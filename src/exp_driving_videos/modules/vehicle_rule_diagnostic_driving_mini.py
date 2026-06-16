"""
Diagnose whether vehicle-centered braking rules were generated, pruned/scored out,
or missed due to predicate representation / selection.

Outputs:
    pipeline_output/20_driving_mini_vehicle_rule_diagnostic/
        vehicle_centered_rule_pool_report.csv
        vehicle_centered_pool_summary.csv
        vehicle_centered_eval_context_summary.csv
        vehicle_centered_diagnostic_summary.json
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_DIAGNOSTIC_VERSION = 1


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "20_driving_mini_vehicle_rule_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "vehicle_classes": sorted(str(v) for v in cfg.get("vehicle_classes", [])),
        "near_states": sorted(str(v) for v in cfg.get("near_states", [])),
        "center_states": sorted(str(v) for v in cfg.get("center_states", [])),
        "primary_rule_set": str(cfg.get("primary_rule_set", "original")),
    }


def _parse_atom(atom: str) -> Optional[Tuple[str, List[str]]]:
    text = str(atom).strip()
    match = re.match(r"^([a-z0-9_]+)\((.*)\)\.$", text)
    if not match:
        return None
    predicate = match.group(1)
    args_text = match.group(2).strip()
    if not args_text:
        return predicate, []
    return predicate, [part.strip() for part in args_text.split(",")]


def _get_rule_body_atom_templates(rule: Dict[str, Any]) -> List[str]:
    body_atom_templates = rule.get("body_atom_templates")
    if isinstance(body_atom_templates, list):
        return [str(atom).strip() for atom in body_atom_templates if str(atom).strip()]
    body_atom_template = str(rule.get("body_atom_template", "")).strip()
    return [body_atom_template] if body_atom_template else []


def _sort_rules(all_kept_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        all_kept_rules,
        key=lambda rule: (
            -float(rule.get("confidence", 0.0)),
            -int(rule.get("positive_support", 0)),
            int(rule.get("kept_round_index", 0)),
            str(rule.get("clause", "")),
        ),
    )


def _rule_family_signature(rule: Dict[str, Any]) -> str:
    predicates = set()
    for atom in _get_rule_body_atom_templates(rule):
        parsed = _parse_atom(atom)
        if parsed is None:
            continue
        predicates.add(str(parsed[0]))
    return "|".join(sorted(predicates))


def _rule_vehicle_match(rule: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    vehicle_classes = {str(v) for v in cfg.get("vehicle_classes", ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"])}
    near_states = {str(v) for v in cfg.get("near_states", ["near"])}
    center_states = {str(v) for v in cfg.get("center_states", ["centered"])}

    matched_vehicle_classes: Set[str] = set()
    has_near = False
    has_centered = False
    matched_predicates: Set[str] = set()

    for atom in _get_rule_body_atom_templates(rule):
        parsed = _parse_atom(atom)
        if parsed is None:
            continue
        predicate, args = parsed
        if predicate == "object_class" and len(args) >= 3 and str(args[2]) in vehicle_classes:
            matched_vehicle_classes.add(str(args[2]))
            matched_predicates.add("vehicle")
        elif predicate == "object_distance_state" and len(args) >= 3 and str(args[2]) in near_states:
            has_near = True
            matched_predicates.add("near")
        elif predicate == "object_x_position_state" and len(args) >= 3 and str(args[2]) in center_states:
            has_centered = True
            matched_predicates.add("centered")

    if matched_predicates == {"vehicle", "near", "centered"}:
        match_level = "exact_vehicle_near_centered"
    elif matched_predicates == {"vehicle", "near"}:
        match_level = "vehicle_near_partial"
    elif matched_predicates == {"vehicle", "centered"}:
        match_level = "vehicle_centered_partial"
    elif matched_predicates == {"near", "centered"}:
        match_level = "near_centered_partial"
    elif matched_predicates == {"vehicle"}:
        match_level = "vehicle_only"
    elif matched_predicates == {"near"}:
        match_level = "near_only"
    elif matched_predicates == {"centered"}:
        match_level = "centered_only"
    else:
        match_level = "no_match"

    return {
        "match_level": match_level,
        "has_vehicle_class": bool(matched_vehicle_classes),
        "has_near": has_near,
        "has_centered": has_centered,
        "matched_vehicle_classes": sorted(matched_vehicle_classes),
    }


def _iter_eval_examples(temporal_rule_results: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for video_result in temporal_rule_results:
        video_id = str(video_result.get("video_id", ""))
        for example in list(video_result.get("examples", [])):
            row = dict(example)
            row["video_id"] = video_id
            yield row


def _example_vehicle_context(example: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    vehicle_classes = {str(v) for v in cfg.get("vehicle_classes", ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"])}
    near_states = {str(v) for v in cfg.get("near_states", ["near"])}
    center_states = {str(v) for v in cfg.get("center_states", ["centered"])}

    object_rows: Dict[str, Dict[str, str]] = {}
    ego_forward_state = "unknown"
    ego_lateral_state = "unknown"
    ego_motion_state = "unknown"

    for atom in list(example.get("body_atoms", [])):
        parsed = _parse_atom(str(atom))
        if parsed is None:
            continue
        predicate, args = parsed
        if predicate == "segment_forward_state" and len(args) >= 2:
            ego_forward_state = str(args[1])
        elif predicate == "segment_lateral_state" and len(args) >= 2:
            ego_lateral_state = str(args[1])
        elif predicate == "segment_motion_state" and len(args) >= 2:
            ego_motion_state = str(args[1])
        elif predicate.startswith("object_") and len(args) >= 2:
            object_id = str(args[1])
            obj = object_rows.setdefault(object_id, {})
            if predicate == "object_class" and len(args) >= 3:
                obj["object_class"] = str(args[2])
            elif predicate == "object_distance_state" and len(args) >= 3:
                obj["distance_state"] = str(args[2])
            elif predicate == "object_x_position_state" and len(args) >= 3:
                obj["x_position_state"] = str(args[2])

    has_vehicle = any(str(obj.get("object_class", "")) in vehicle_classes for obj in object_rows.values())
    has_near = any(str(obj.get("distance_state", "")) in near_states for obj in object_rows.values())
    has_centered = any(str(obj.get("x_position_state", "")) in center_states for obj in object_rows.values())
    has_vehicle_near_centered = any(
        str(obj.get("object_class", "")) in vehicle_classes
        and str(obj.get("distance_state", "")) in near_states
        and str(obj.get("x_position_state", "")) in center_states
        for obj in object_rows.values()
    )
    return {
        "video_id": str(example.get("video_id", "")),
        "example_id": str(example.get("example_id", "")),
        "label": bool(example.get("label", False)),
        "ego_forward_state": ego_forward_state,
        "ego_lateral_state": ego_lateral_state,
        "ego_motion_state": ego_motion_state,
        "has_vehicle": has_vehicle,
        "has_near": has_near,
        "has_centered": has_centered,
        "has_vehicle_near_centered": has_vehicle_near_centered,
    }


def _predicted_positive_ids(evaluation_results: Dict[str, Any]) -> Set[str]:
    path = Path(str(evaluation_results.get("example_predictions_csv_path", "")))
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        return {str(row["example_id"]) for row in rows if str(row.get("predicted_positive", "")) == "True"}
    tp = set()
    return tp


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def process_diagnostic(
    merged_initial_rules: Dict[str, Any],
    extended_rule_results: Dict[str, Any],
    original_final_rule_results: Dict[str, Any],
    diverse_final_rule_results: Dict[str, Any],
    eval_temporal_rule_results: List[Dict[str, Any]],
    evaluation_results_by_name: Dict[str, Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "vehicle_centered_diagnostic_summary.json"
    pool_report_path = out_root / "vehicle_centered_rule_pool_report.csv"
    pool_summary_path = out_root / "vehicle_centered_pool_summary.csv"
    eval_context_path = out_root / "vehicle_centered_eval_context_summary.csv"

    if not force_recompute and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _DIAGNOSTIC_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {manifest_path.name}")
            return cached

    original_ranked_rules = _sort_rules(list(extended_rule_results.get("all_kept_rules", [])))
    original_rank_map = {str(rule.get("rule_id", "")): idx for idx, rule in enumerate(original_ranked_rules)}
    selected_original_ids = {str(rule.get("rule_id", "")) for rule in list(original_final_rule_results.get("final_rules", []))}
    selected_diverse_ids = {str(rule.get("rule_id", "")) for rule in list(diverse_final_rule_results.get("final_rules", []))}

    pools = [
        ("merged_initial", list(merged_initial_rules.get("rules", []))),
        ("scored_all_kept", list(extended_rule_results.get("all_kept_rules", []))),
        ("selected_original", list(original_final_rule_results.get("final_rules", []))),
        ("selected_diverse", list(diverse_final_rule_results.get("final_rules", []))),
    ]

    pool_report_rows: List[Dict[str, Any]] = []
    pool_summary_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    exact_scored_rules: List[Dict[str, Any]] = []

    for pool_name, rules in pools:
        for rule in rules:
            match = _rule_vehicle_match(rule, cfg)
            match_level = str(match["match_level"])
            if match_level == "no_match":
                continue
            row = {
                "pool_name": pool_name,
                "rule_id": str(rule.get("rule_id", "")),
                "clause": str(rule.get("clause", "")),
                "confidence": float(rule.get("confidence", 0.0)),
                "positive_support": int(rule.get("positive_support", 0)),
                "negative_support": int(rule.get("negative_support", 0)),
                "total_support": int(rule.get("total_support", 0)),
                "match_level": match_level,
                "vehicle_classes": json.dumps(match["matched_vehicle_classes"]),
                "family_signature": _rule_family_signature(rule),
                "original_rank": original_rank_map.get(str(rule.get("rule_id", "")), ""),
                "selected_original": str(rule.get("rule_id", "")) in selected_original_ids,
                "selected_diverse": str(rule.get("rule_id", "")) in selected_diverse_ids,
            }
            pool_report_rows.append(row)

            summary = pool_summary_map.setdefault(
                (pool_name, match_level),
                {
                    "pool_name": pool_name,
                    "match_level": match_level,
                    "num_rules": 0,
                    "best_confidence": 0.0,
                    "best_positive_support": 0,
                    "representative_clauses": [],
                },
            )
            summary["num_rules"] += 1
            summary["best_confidence"] = max(summary["best_confidence"], float(rule.get("confidence", 0.0)))
            summary["best_positive_support"] = max(summary["best_positive_support"], int(rule.get("positive_support", 0)))
            if len(summary["representative_clauses"]) < 3 and str(rule.get("clause", "")):
                summary["representative_clauses"].append(str(rule.get("clause", "")))

            if pool_name == "scored_all_kept" and match_level == "exact_vehicle_near_centered":
                exact_scored_rules.append(row)

    _write_csv(
        pool_report_path,
        [
            "pool_name",
            "rule_id",
            "clause",
            "confidence",
            "positive_support",
            "negative_support",
            "total_support",
            "match_level",
            "vehicle_classes",
            "family_signature",
            "original_rank",
            "selected_original",
            "selected_diverse",
        ],
        sorted(pool_report_rows, key=lambda row: (row["pool_name"], row["match_level"], row["original_rank"] if row["original_rank"] != "" else 10**9)),
    )

    pool_summary_rows = []
    for summary in sorted(pool_summary_map.values(), key=lambda row: (row["pool_name"], row["match_level"])):
        pool_summary_rows.append(
            {
                "pool_name": summary["pool_name"],
                "match_level": summary["match_level"],
                "num_rules": int(summary["num_rules"]),
                "best_confidence": float(summary["best_confidence"]),
                "best_positive_support": int(summary["best_positive_support"]),
                "representative_clauses": json.dumps(summary["representative_clauses"]),
            }
        )
    _write_csv(
        pool_summary_path,
        [
            "pool_name",
            "match_level",
            "num_rules",
            "best_confidence",
            "best_positive_support",
            "representative_clauses",
        ],
        pool_summary_rows,
    )

    eval_examples = [_example_vehicle_context(example, cfg) for example in _iter_eval_examples(eval_temporal_rule_results)]
    primary_rule_set = str(cfg.get("primary_rule_set", "original"))
    predicted_ids_by_name = {
        name: _predicted_positive_ids(results)
        for name, results in evaluation_results_by_name.items()
    }

    context_groups: Dict[Tuple[str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "num_positive_examples": 0,
            "num_vehicle_near_centered_positive_examples": 0,
            "predicted_positive_original": 0,
            "predicted_positive_diverse": 0,
        }
    )
    vehicle_centered_positive_ids: Set[str] = set()
    for row in eval_examples:
        if not bool(row["label"]):
            continue
        key = (str(row["ego_forward_state"]), str(row["ego_lateral_state"]), str(row["ego_motion_state"]))
        group = context_groups[key]
        group["ego_forward_state"] = str(row["ego_forward_state"])
        group["ego_lateral_state"] = str(row["ego_lateral_state"])
        group["ego_motion_state"] = str(row["ego_motion_state"])
        group["num_positive_examples"] += 1
        if bool(row["has_vehicle_near_centered"]):
            group["num_vehicle_near_centered_positive_examples"] += 1
            vehicle_centered_positive_ids.add(str(row["example_id"]))
        if str(row["example_id"]) in predicted_ids_by_name.get("original", set()):
            group["predicted_positive_original"] += 1
        if str(row["example_id"]) in predicted_ids_by_name.get("diverse", set()):
            group["predicted_positive_diverse"] += 1

    eval_context_rows = []
    for group in sorted(
        context_groups.values(),
        key=lambda item: (-int(item["num_vehicle_near_centered_positive_examples"]), -int(item["num_positive_examples"]), str(item["ego_motion_state"])),
    ):
        eval_context_rows.append(
            {
                "ego_forward_state": group["ego_forward_state"],
                "ego_lateral_state": group["ego_lateral_state"],
                "ego_motion_state": group["ego_motion_state"],
                "num_positive_examples": int(group["num_positive_examples"]),
                "num_vehicle_near_centered_positive_examples": int(group["num_vehicle_near_centered_positive_examples"]),
                "predicted_positive_original": int(group["predicted_positive_original"]),
                "predicted_positive_diverse": int(group["predicted_positive_diverse"]),
                "recall_original": float(
                    group["predicted_positive_original"] / max(1, group["num_vehicle_near_centered_positive_examples"])
                ),
                "recall_diverse": float(
                    group["predicted_positive_diverse"] / max(1, group["num_vehicle_near_centered_positive_examples"])
                ),
            }
        )
    _write_csv(
        eval_context_path,
        [
            "ego_forward_state",
            "ego_lateral_state",
            "ego_motion_state",
            "num_positive_examples",
            "num_vehicle_near_centered_positive_examples",
            "predicted_positive_original",
            "predicted_positive_diverse",
            "recall_original",
            "recall_diverse",
        ],
        eval_context_rows,
    )

    exact_scored_count = sum(1 for row in pool_report_rows if row["pool_name"] == "scored_all_kept" and row["match_level"] == "exact_vehicle_near_centered")
    exact_selected_original_count = sum(1 for row in pool_report_rows if row["pool_name"] == "selected_original" and row["match_level"] == "exact_vehicle_near_centered")
    exact_selected_diverse_count = sum(1 for row in pool_report_rows if row["pool_name"] == "selected_diverse" and row["match_level"] == "exact_vehicle_near_centered")
    initial_vehicle_count = sum(1 for row in pool_report_rows if row["pool_name"] == "merged_initial" and row["match_level"] == "vehicle_only")
    initial_near_count = sum(1 for row in pool_report_rows if row["pool_name"] == "merged_initial" and row["match_level"] == "near_only")
    initial_centered_count = sum(1 for row in pool_report_rows if row["pool_name"] == "merged_initial" and row["match_level"] == "centered_only")

    vehicle_centered_positive_count = len(vehicle_centered_positive_ids)
    fn_original = vehicle_centered_positive_ids - predicted_ids_by_name.get("original", set())
    fn_diverse = vehicle_centered_positive_ids - predicted_ids_by_name.get("diverse", set())
    best_exact_original_rank = min(
        [int(row["original_rank"]) for row in exact_scored_rules if row["original_rank"] != ""],
        default=None,
    )

    if vehicle_centered_positive_count == 0:
        primary_diagnosis = "insufficient_vehicle_centered_eval_signal"
        rationale = "No held-out positive examples with vehicle+near+centered context were found."
    elif exact_scored_count > 0 and exact_selected_original_count == 0:
        primary_diagnosis = "rule_selection"
        rationale = (
            "Vehicle-centered rules exist in the scored pool but were not selected into the original final rule set. "
            "This points primarily to post-mining rule selection."
        )
    elif exact_scored_count == 0 and initial_vehicle_count > 0 and initial_near_count > 0 and initial_centered_count > 0:
        primary_diagnosis = "rule_generation_or_pruning"
        rationale = (
            "Vehicle, near, and centered unary rule atoms were generated, but no exact vehicle+near+centered rules survived "
            "into the scored rule pool. This points to extension / pruning pressure."
        )
    elif min(initial_vehicle_count, initial_near_count, initial_centered_count) == 0 and vehicle_centered_positive_count > 0:
        primary_diagnosis = "predicate_representation"
        rationale = (
            "Held-out positives contain vehicle+near+centered contexts, but the train-side initial rule pool lacks one or more "
            "foundational unary predicates. This points to representation or upstream symbolic extraction gaps."
        )
    else:
        primary_diagnosis = "mixed_or_unclear"
        rationale = "Signals are mixed; no single failure mode dominates the vehicle-centered rule audit."

    if exact_selected_diverse_count > exact_selected_original_count or len(fn_diverse) < len(fn_original):
        selection_effect = "diverse_selection_helpful"
    else:
        selection_effect = "diverse_selection_not_helpful"

    manifest = {
        "version": _DIAGNOSTIC_VERSION,
        "config": _cfg_key_subset(cfg),
        "primary_diagnosis": primary_diagnosis,
        "selection_effect": selection_effect,
        "rationale": rationale,
        "vehicle_centered_positive_eval_examples": vehicle_centered_positive_count,
        "vehicle_centered_fn_original": len(fn_original),
        "vehicle_centered_fn_diverse": len(fn_diverse),
        "exact_scored_rule_count": exact_scored_count,
        "exact_selected_original_rule_count": exact_selected_original_count,
        "exact_selected_diverse_rule_count": exact_selected_diverse_count,
        "best_exact_original_rank": best_exact_original_rank,
        "initial_vehicle_unary_rule_count": initial_vehicle_count,
        "initial_near_unary_rule_count": initial_near_count,
        "initial_centered_unary_rule_count": initial_centered_count,
        "pool_report_path": str(pool_report_path),
        "pool_summary_path": str(pool_summary_path),
        "eval_context_summary_path": str(eval_context_path),
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(
        "  vehicle_rule_diagnostic: "
        f"diagnosis={primary_diagnosis} | "
        f"vehicle_centered_positive_eval_examples={vehicle_centered_positive_count} | "
        f"exact_scored_rules={exact_scored_count} | "
        f"exact_selected_original={exact_selected_original_count} | "
        f"exact_selected_diverse={exact_selected_diverse_count}"
    )
    print(f"Vehicle-centered rule pool report written to {pool_report_path}")
    print(f"Vehicle-centered pool summary written to {pool_summary_path}")
    print(f"Vehicle-centered eval context summary written to {eval_context_path}")
    print(f"Vehicle-centered diagnostic summary written to {manifest_path}")
    return manifest


def run(
    merged_initial_rules: Dict[str, Any],
    extended_rule_results: Dict[str, Any],
    original_final_rule_results: Dict[str, Any],
    diverse_final_rule_results: Dict[str, Any],
    eval_temporal_rule_results: List[Dict[str, Any]],
    evaluation_results_by_name: Dict[str, Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_diagnostic(
        merged_initial_rules=merged_initial_rules,
        extended_rule_results=extended_rule_results,
        original_final_rule_results=original_final_rule_results,
        diverse_final_rule_results=diverse_final_rule_results,
        eval_temporal_rule_results=eval_temporal_rule_results,
        evaluation_results_by_name=evaluation_results_by_name,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
