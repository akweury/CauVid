from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import config
from src.exp_driving_videos import pipeline_config

_MERGED_INITIAL_RULES_VERSION = 3


def dedupe_rule_evidence_entries(evidence_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]] = set()
    for entry in evidence_entries:
        bindings = dict(entry.get("bindings", {}))
        matched_atoms = dict(entry.get("matched_atoms", {}))
        key = (
            str(entry.get("example_id", "")),
            str(entry.get("matched_atom", "")),
            tuple(sorted((str(k), str(v)) for k, v in bindings.items())),
            tuple(sorted((str(k), str(v)) for k, v in matched_atoms.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def summarize_rule_evidence(evidence_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_firings = len(evidence_entries)
    positive_firings = sum(1 for entry in evidence_entries if bool(entry.get("label", False)))
    negative_firings = total_firings - positive_firings
    positive_example_ids = sorted(
        {
            str(entry.get("example_id", ""))
            for entry in evidence_entries
            if bool(entry.get("label", False)) and str(entry.get("example_id", ""))
        }
    )
    negative_example_ids = sorted(
        {
            str(entry.get("example_id", ""))
            for entry in evidence_entries
            if not bool(entry.get("label", False)) and str(entry.get("example_id", ""))
        }
    )
    total_support = len(set(positive_example_ids) | set(negative_example_ids))
    confidence = float(positive_firings / max(1, total_firings))
    return {
        "positive_support": len(positive_example_ids),
        "negative_support": len(negative_example_ids),
        "total_support": total_support,
        "positive_firings": positive_firings,
        "negative_firings": negative_firings,
        "total_firings": total_firings,
        "confidence": confidence,
        "positive_example_ids": positive_example_ids,
        "negative_example_ids": negative_example_ids,
    }


def summarize_rule_candidate_provenance(
    evidence_entries: List[Dict[str, Any]],
    body_length: int,
) -> Dict[str, Any]:
    candidate_body_templates: set[str] = set()
    accepted_body_templates: set[str] = set()
    candidate_evidence_firings = 0
    accepted_evidence_firings = 0
    matched_prior_ids_involved = sorted(
        {
            str(prior_id)
            for entry in evidence_entries
            for prior_ids in dict(entry.get("matched_atom_prior_ids", {})).values()
            for prior_id in list(prior_ids)
            if str(prior_id)
        }
    )
    for entry in evidence_entries:
        matched_atom_sources = dict(entry.get("matched_atom_sources", {}))
        if not matched_atom_sources:
            body_atom_template = str(entry.get("body_atom_template", "")).strip()
            source = str(entry.get("body_atom_source", "")).strip()
            if body_atom_template and source:
                matched_atom_sources = {body_atom_template: source}
        for body_atom_template, source in matched_atom_sources.items():
            if str(source) == "candidate":
                candidate_body_templates.add(str(body_atom_template))
                candidate_evidence_firings += 1
            elif str(source) == "accepted":
                accepted_body_templates.add(str(body_atom_template))
                accepted_evidence_firings += 1

    uses_candidate_atoms = bool(candidate_body_templates)
    uses_only_candidate_atoms = uses_candidate_atoms and not accepted_body_templates
    mixes_accepted_and_candidate_atoms = bool(candidate_body_templates) and bool(accepted_body_templates)
    num_candidate_body_atoms = len(candidate_body_templates)
    if mixes_accepted_and_candidate_atoms:
        body_source_mix = "mixed_accepted_and_candidate"
    elif uses_only_candidate_atoms:
        body_source_mix = "candidate_only"
    elif accepted_body_templates:
        body_source_mix = "accepted_only"
    else:
        body_source_mix = "non_object_or_segment_only"

    return {
        "uses_candidate_atoms": uses_candidate_atoms,
        "num_candidate_body_atoms": num_candidate_body_atoms,
        "candidate_body_atom_ratio": float(num_candidate_body_atoms / max(1, int(body_length))),
        "matched_prior_ids_involved": matched_prior_ids_involved,
        "mixes_accepted_and_candidate_atoms": mixes_accepted_and_candidate_atoms,
        "uses_only_candidate_atoms": uses_only_candidate_atoms,
        "body_source_mix": body_source_mix,
        "candidate_evidence_firings": candidate_evidence_firings,
        "accepted_evidence_firings": accepted_evidence_firings,
    }


def rule_candidate_category(rule: Dict[str, Any]) -> str:
    if bool(rule.get("mixes_accepted_and_candidate_atoms", False)):
        return "mixed_accepted_candidate"
    if bool(rule.get("uses_only_candidate_atoms", False)):
        return "candidate_only"
    if bool(rule.get("uses_candidate_atoms", False)):
        return "candidate_only"
    return "accepted_only"


def empty_rule_category_counts() -> Dict[str, int]:
    return {
        "accepted_only_rules": 0,
        "candidate_only_rules": 0,
        "mixed_accepted_candidate_rules": 0,
        "candidate_involving_rules": 0,
        "all_rules": 0,
    }


def count_rule_categories(rules: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = empty_rule_category_counts()
    for rule in rules:
        category = str(rule.get("candidate_rule_category", "")) or rule_candidate_category(rule)
        counts["all_rules"] += 1
        if category == "mixed_accepted_candidate":
            counts["mixed_accepted_candidate_rules"] += 1
            counts["candidate_involving_rules"] += 1
        elif category == "candidate_only":
            counts["candidate_only_rules"] += 1
            counts["candidate_involving_rules"] += 1
        else:
            counts["accepted_only_rules"] += 1
    return counts


def merge_candidate_rules(candidate_rule_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    out_root = pipeline_config.get_merged_candidate_rules_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "merged_initial_rules.json"
    csv_path = out_root / "merged_initial_rules.csv"
    manifest_path = out_root / "merged_initial_rules_manifest.json"

    merged_rule_map: Dict[str, Dict[str, Any]] = {}
    video_summaries: List[Dict[str, Any]] = []
    target_predicates: set[str] = set()
    total_examples = 0
    total_positive_examples = 0
    total_negative_examples = 0
    total_rules_using_candidate_atoms = 0
    total_candidate_only_rules = 0
    total_mixed_source_rules = 0
    input_generated_counts = empty_rule_category_counts()
    atom_availability_counts = {
        "num_body_atom_templates": 0,
        "num_accepted_body_atom_templates": 0,
        "num_candidate_body_atom_templates": 0,
        "num_candidate_semantic_body_atom_templates": 0,
        "num_candidate_provenance_only_body_atom_templates": 0,
    }
    candidate_semantic_body_atom_templates: set[str] = set()
    candidate_provenance_only_body_atom_templates: set[str] = set()

    for video_result in sorted(candidate_rule_results, key=lambda item: str(item.get("video_id", ""))):
        video_id = str(video_result.get("video_id", "unknown"))
        target_predicate = str(video_result.get("target_predicate", ""))
        if target_predicate:
            target_predicates.add(target_predicate)
        total_examples += int(video_result.get("num_examples", 0))
        total_positive_examples += int(video_result.get("num_positive_examples", 0))
        total_negative_examples += int(video_result.get("num_negative_examples", 0))

        candidate_rules = list(video_result.get("initial_rules", video_result.get("candidate_rules", [])))
        video_summaries.append(
            {
                "video_id": video_id,
                "target_predicate": target_predicate,
                "num_initial_rules": len(candidate_rules),
                "num_examples": int(video_result.get("num_examples", 0)),
                "num_positive_examples": int(video_result.get("num_positive_examples", 0)),
                "num_negative_examples": int(video_result.get("num_negative_examples", 0)),
                "num_rules_using_candidate_atoms": int(video_result.get("num_rules_using_candidate_atoms", 0)),
                "num_candidate_only_rules": int(video_result.get("num_candidate_only_rules", 0)),
                "num_mixed_source_rules": int(video_result.get("num_mixed_source_rules", 0)),
            }
        )
        total_rules_using_candidate_atoms += int(video_result.get("num_rules_using_candidate_atoms", 0))
        total_candidate_only_rules += int(video_result.get("num_candidate_only_rules", 0))
        total_mixed_source_rules += int(video_result.get("num_mixed_source_rules", 0))
        atom_availability = dict(
            dict(video_result.get("candidate_rule_stage_stats", {})).get("atom_availability", {})
        )
        for key in atom_availability_counts:
            atom_availability_counts[key] += int(atom_availability.get(key, 0))
        candidate_semantic_body_atom_templates.update(
            str(value)
            for value in list(atom_availability.get("candidate_semantic_body_atom_templates", []))
            if str(value)
        )
        candidate_provenance_only_body_atom_templates.update(
            str(value)
            for value in list(atom_availability.get("candidate_provenance_only_body_atom_templates", []))
            if str(value)
        )
        generated_counts = dict(
            dict(video_result.get("candidate_rule_stage_stats", {})).get("generated_rule_counts", {})
        )
        if not generated_counts:
            generated_counts = {
                "accepted_only_rules": max(
                    0,
                    len(candidate_rules) - int(video_result.get("num_rules_using_candidate_atoms", 0)),
                ),
                "candidate_only_rules": int(video_result.get("num_candidate_only_rules", 0)),
                "mixed_accepted_candidate_rules": int(video_result.get("num_mixed_source_rules", 0)),
                "candidate_involving_rules": int(video_result.get("num_rules_using_candidate_atoms", 0)),
                "all_rules": len(candidate_rules),
            }
        for key in input_generated_counts:
            input_generated_counts[key] += int(generated_counts.get(key, 0))

        for rule in candidate_rules:
            clause = str(rule.get("clause", "")).strip()
            if not clause:
                continue

            merged_rule = merged_rule_map.get(clause)
            if merged_rule is None:
                merged_rule = {
                    "merged_rule_index": -1,
                    "rule_id": f"merged_rule_{len(merged_rule_map):04d}",
                    "head_predicate": rule.get("head_predicate", ""),
                    "head_atom_template": rule.get("head_atom_template", ""),
                    "body_atom_template": rule.get("body_atom_template", ""),
                    "body_atom_templates": list(rule.get("body_atom_templates", []))
                    if isinstance(rule.get("body_atom_templates"), list)
                    else ([rule.get("body_atom_template", "")] if rule.get("body_atom_template", "") else []),
                    "body_length": int(rule.get("body_length", 1)),
                    "clause": clause,
                    "positive_support": 0,
                    "negative_support": 0,
                    "total_support": 0,
                    "positive_firings": 0,
                    "negative_firings": 0,
                    "total_firings": 0,
                    "confidence": 0.0,
                    "positive_example_ids": [],
                    "negative_example_ids": [],
                    "evidence_set": [],
                    "source_video_ids": [],
                    "source_rule_ids": [],
                    "source_rule_indices": [],
                    "num_source_rules": 0,
                }
                merged_rule_map[clause] = merged_rule

            merged_rule["evidence_set"].extend(list(rule.get("evidence_set", [])))
            merged_rule["source_video_ids"].append(video_id)
            merged_rule["source_rule_ids"].append(rule.get("rule_id", ""))
            merged_rule["source_rule_indices"].append(rule.get("rule_index"))
            merged_rule["num_source_rules"] += 1

    merged_rules = sorted(merged_rule_map.values(), key=lambda item: str(item.get("clause", "")))
    for idx, merged_rule in enumerate(merged_rules):
        merged_rule["merged_rule_index"] = idx
        merged_rule["evidence_set"] = dedupe_rule_evidence_entries(list(merged_rule.get("evidence_set", [])))
        evidence_summary = summarize_rule_evidence(list(merged_rule.get("evidence_set", [])))
        provenance_summary = summarize_rule_candidate_provenance(
            list(merged_rule.get("evidence_set", [])),
            int(merged_rule.get("body_length", 1)),
        )
        merged_rule["positive_support"] = int(evidence_summary["positive_support"])
        merged_rule["negative_support"] = int(evidence_summary["negative_support"])
        merged_rule["total_support"] = int(evidence_summary["total_support"])
        merged_rule["positive_firings"] = int(evidence_summary["positive_firings"])
        merged_rule["negative_firings"] = int(evidence_summary["negative_firings"])
        merged_rule["total_firings"] = int(evidence_summary["total_firings"])
        merged_rule["confidence"] = float(evidence_summary["confidence"])
        merged_rule["positive_example_ids"] = list(evidence_summary["positive_example_ids"])
        merged_rule["negative_example_ids"] = list(evidence_summary["negative_example_ids"])
        merged_rule.update(provenance_summary)
        merged_rule["candidate_rule_category"] = rule_candidate_category(merged_rule)
        merged_rule["source_video_ids"] = sorted(set(str(v) for v in merged_rule["source_video_ids"]))
        merged_rule["source_rule_ids"] = sorted(
            set(str(rule_id) for rule_id in merged_rule["source_rule_ids"] if str(rule_id))
        )
        merged_rule["source_rule_indices"] = [
            source_index for source_index in merged_rule["source_rule_indices"] if source_index is not None
        ]

    merged_category_counts = count_rule_categories(merged_rules)
    aggregate_atom_availability = {
        **atom_availability_counts,
        "candidate_semantic_body_atom_templates": sorted(candidate_semantic_body_atom_templates),
        "candidate_provenance_only_body_atom_templates": sorted(candidate_provenance_only_body_atom_templates),
    }
    merged_result: Dict[str, Any] = {
        "version": _MERGED_INITIAL_RULES_VERSION,
        "num_videos": len(candidate_rule_results),
        "num_examples": total_examples,
        "num_positive_examples": total_positive_examples,
        "num_negative_examples": total_negative_examples,
        "num_rules": len(merged_rules),
        "num_rules_using_candidate_atoms": merged_category_counts["candidate_involving_rules"],
        "num_candidate_only_rules": merged_category_counts["candidate_only_rules"],
        "num_mixed_source_rules": merged_category_counts["mixed_accepted_candidate_rules"],
        "candidate_rule_stage_stats": {
            "stage": "step15_merge_initial_rules",
            "atom_availability": aggregate_atom_availability,
            "input_generated_rule_counts": input_generated_counts,
            "merged_rule_counts": merged_category_counts,
        },
        "candidate_rule_flow_summary": {
            "atom_availability": aggregate_atom_availability,
            "initial_rule_generation": input_generated_counts,
            "merged_after_step15": merged_category_counts,
            "pruning": empty_rule_category_counts(),
            "extension": empty_rule_category_counts(),
            "final_selection": empty_rule_category_counts(),
            "evaluation": empty_rule_category_counts(),
        },
        "target_predicates": sorted(target_predicates),
        "rules": merged_rules,
    }

    manifest: Dict[str, Any] = {
        "version": _MERGED_INITIAL_RULES_VERSION,
        "num_videos": len(candidate_rule_results),
        "num_examples": total_examples,
        "num_positive_examples": total_positive_examples,
        "num_negative_examples": total_negative_examples,
        "num_rules": len(merged_rules),
        "input_num_rules_using_candidate_atoms": total_rules_using_candidate_atoms,
        "input_num_candidate_only_rules": total_candidate_only_rules,
        "input_num_mixed_source_rules": total_mixed_source_rules,
        "num_rules_using_candidate_atoms": merged_result["num_rules_using_candidate_atoms"],
        "num_candidate_only_rules": merged_result["num_candidate_only_rules"],
        "num_mixed_source_rules": merged_result["num_mixed_source_rules"],
        "candidate_rule_stage_stats": merged_result["candidate_rule_stage_stats"],
        "target_predicates": sorted(target_predicates),
        "videos": video_summaries,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(merged_result, fh, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "merged_rule_index",
                "rule_id",
                "head_predicate",
                "head_atom_template",
                "body_atom_template",
                "body_atom_templates",
                "body_length",
                "clause",
                "positive_support",
                "negative_support",
                "total_support",
                "positive_firings",
                "negative_firings",
                "total_firings",
                "confidence",
                "uses_candidate_atoms",
                "num_candidate_body_atoms",
                "candidate_body_atom_ratio",
                "matched_prior_ids_involved",
                "mixes_accepted_and_candidate_atoms",
                "uses_only_candidate_atoms",
                "body_source_mix",
                "candidate_rule_category",
                "positive_example_ids",
                "negative_example_ids",
                "source_video_ids",
                "source_rule_ids",
                "source_rule_indices",
                "num_source_rules",
            ],
        )
        writer.writeheader()
        for rule in merged_rules:
            writer.writerow(
                {
                    "merged_rule_index": rule.get("merged_rule_index", ""),
                    "rule_id": rule.get("rule_id", ""),
                    "head_predicate": rule.get("head_predicate", ""),
                    "head_atom_template": rule.get("head_atom_template", ""),
                    "body_atom_template": rule.get("body_atom_template", ""),
                    "body_atom_templates": json.dumps(rule.get("body_atom_templates", [])),
                    "body_length": rule.get("body_length", 1),
                    "clause": rule.get("clause", ""),
                    "positive_support": rule.get("positive_support", 0),
                    "negative_support": rule.get("negative_support", 0),
                    "total_support": rule.get("total_support", 0),
                    "positive_firings": rule.get("positive_firings", 0),
                    "negative_firings": rule.get("negative_firings", 0),
                    "total_firings": rule.get("total_firings", 0),
                    "confidence": rule.get("confidence", 0.0),
                    "uses_candidate_atoms": rule.get("uses_candidate_atoms", False),
                    "num_candidate_body_atoms": rule.get("num_candidate_body_atoms", 0),
                    "candidate_body_atom_ratio": rule.get("candidate_body_atom_ratio", 0.0),
                    "matched_prior_ids_involved": json.dumps(rule.get("matched_prior_ids_involved", [])),
                    "mixes_accepted_and_candidate_atoms": rule.get("mixes_accepted_and_candidate_atoms", False),
                    "uses_only_candidate_atoms": rule.get("uses_only_candidate_atoms", False),
                    "body_source_mix": rule.get("body_source_mix", ""),
                    "candidate_rule_category": rule.get("candidate_rule_category", ""),
                    "positive_example_ids": json.dumps(rule.get("positive_example_ids", [])),
                    "negative_example_ids": json.dumps(rule.get("negative_example_ids", [])),
                    "source_video_ids": json.dumps(rule.get("source_video_ids", [])),
                    "source_rule_ids": json.dumps(rule.get("source_rule_ids", [])),
                    "source_rule_indices": json.dumps(rule.get("source_rule_indices", [])),
                    "num_source_rules": rule.get("num_source_rules", 0),
                }
            )

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Merged initial rule JSON written to {json_path}")
    print(f"Merged initial rule CSV written to {csv_path}")
    print(f"Merged initial rule manifest written to {manifest_path}")
    return merged_result


def select_video_results(
    video_results: List[Dict[str, Any]],
    selected_video_ids: List[str],
) -> List[Dict[str, Any]]:
    selected_video_id_set = {str(video_id) for video_id in selected_video_ids}
    return [
        result
        for result in video_results
        if str(result.get("video_id", "")) in selected_video_id_set
    ]


def build_train_eval_split(
    video_ids: List[str],
    train_video_count: int,
    eval_video_count: int,
    strategy: str = "eval_fraction",
    eval_fraction: float = 0.2,
) -> Dict[str, Any]:
    unique_video_ids = sorted({str(video_id) for video_id in video_ids if str(video_id)})
    total_videos = len(unique_video_ids)
    if total_videos < 2:
        raise ValueError(
            "At least 2 videos are required to create train/evaluation splits. "
            f"Found {total_videos}."
        )
    if train_video_count < 1:
        raise ValueError(f"train_video_count must be >= 1. Found {train_video_count}.")
    if eval_video_count < 1:
        raise ValueError(f"eval_video_count must be >= 1. Found {eval_video_count}.")

    strategy = str(strategy or "eval_fraction")
    if strategy == "fixed_counts":
        requested_total = train_video_count + eval_video_count
        if total_videos >= requested_total:
            effective_train_count = train_video_count
            effective_eval_count = eval_video_count
            resolved_strategy = f"fixed_counts_first_{train_video_count}_train_{eval_video_count}_eval"
        else:
            effective_train_count = max(1, total_videos - 1)
            effective_eval_count = total_videos - effective_train_count
            resolved_strategy = "fallback_last_video_eval"
    elif strategy == "eval_fraction":
        eval_fraction = float(eval_fraction)
        if not 0.0 < eval_fraction < 1.0:
            raise ValueError(f"eval_fraction must be between 0 and 1. Found {eval_fraction}.")
        effective_eval_count = max(1, min(total_videos - 1, int(math.ceil(total_videos * eval_fraction))))
        effective_train_count = total_videos - effective_eval_count
        resolved_strategy = f"eval_fraction_{eval_fraction:g}"
    else:
        raise ValueError(f"Unsupported data split strategy: {strategy}")

    train_video_ids = unique_video_ids[:effective_train_count]
    eval_video_ids = unique_video_ids[effective_train_count : effective_train_count + effective_eval_count]
    if not eval_video_ids:
        raise ValueError("Failed to assign evaluation videos for the train/eval split.")

    split_manifest = {
        "num_total_videos": total_videos,
        "num_train_videos": len(train_video_ids),
        "num_eval_videos": len(eval_video_ids),
        "requested_train_video_count": train_video_count,
        "requested_eval_video_count": eval_video_count,
        "requested_eval_fraction": eval_fraction if strategy == "eval_fraction" else None,
        "strategy": resolved_strategy,
        "train_video_ids": train_video_ids,
        "eval_video_ids": eval_video_ids,
        "unused_video_ids": unique_video_ids[effective_train_count + effective_eval_count :],
    }

    out_root = pipeline_config.get_split_output_root()
    manifest_path = out_root / "train_eval_split.json"
    split_manifest["manifest_path"] = str(manifest_path)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(split_manifest, fh, indent=2)
    print(
        "Train/eval split: "
        f"train={train_video_ids} | "
        f"eval={eval_video_ids}"
    )
    print(f"Split manifest written to {manifest_path}")
    return split_manifest


def get_default_driving_mini_video_ids(limit: int | None = None) -> List[str]:
    selection_cfg = pipeline_config.get_video_selection_cfg()
    raw_limit = selection_cfg.get("default_video_limit")
    default_limit = int(raw_limit) if raw_limit is not None else None
    effective_limit = limit if limit is not None else default_limit

    dataset_root = config.get_dataset_path("driving_mini")
    frames_root = dataset_root / "frames"
    if frames_root.exists():
        video_ids = sorted(path.name for path in frames_root.iterdir() if path.is_dir())
        if video_ids:
            return video_ids[:effective_limit] if effective_limit and effective_limit > 0 else video_ids

    videos_root = dataset_root / "videos"
    if videos_root.exists():
        video_ids = sorted(path.stem for path in videos_root.glob("*.mov"))
        return video_ids[:effective_limit] if effective_limit and effective_limit > 0 else video_ids
    return []


def resolve_video_ids(
    video_ids: List[str] | None = None,
    video_count: int | None = None,
) -> List[str] | None:
    if video_ids:
        resolved_video_ids = list(video_ids)
        return resolved_video_ids[:video_count] if video_count and video_count > 0 else resolved_video_ids
    default_video_ids = get_default_driving_mini_video_ids(limit=video_count)
    return default_video_ids or None
