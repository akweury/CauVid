from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import config
from src.exp_driving_videos import pipeline_config

_MERGED_INITIAL_RULES_VERSION = 3
_PRUNED_INITIAL_RULES_VERSION = 1


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
        "candidate_candidate_rules": 0,
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
            if (
                str(rule.get("initial_rule_pair_category", rule.get("extension_rule_category", "")))
                == "candidate_candidate"
                or (
                    bool(rule.get("uses_only_candidate_atoms", False))
                    and int(rule.get("num_candidate_body_atoms", 0)) >= 2
                )
            ):
                counts["candidate_candidate_rules"] += 1
        else:
            counts["accepted_only_rules"] += 1
    return counts


def _rule_body_length(rule: Dict[str, Any]) -> int:
    body_atom_templates = rule.get("body_atom_templates")
    if isinstance(body_atom_templates, list):
        return len([str(atom) for atom in body_atom_templates if str(atom)])
    if str(rule.get("body_atom_template", "")):
        return 1
    return int(rule.get("body_length", 0))


def _as_sorted_str_tuple(values: Any) -> Tuple[str, ...]:
    if isinstance(values, list):
        return tuple(sorted(str(value) for value in values if str(value)))
    if isinstance(values, tuple):
        return tuple(sorted(str(value) for value in values if str(value)))
    if isinstance(values, set):
        return tuple(sorted(str(value) for value in values if str(value)))
    return ()


def _jsonable_signature_value(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)


def _rule_firing_sets(rule: Dict[str, Any]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    positive_firings: set[str] = set()
    negative_firings: set[str] = set()
    for entry in list(rule.get("evidence_set", [])):
        firing_key = "|".join(
            [
                str(entry.get("example_id", "")),
                str(entry.get("current_segment_id", "")),
                str(entry.get("next_segment_id", "")),
                _jsonable_signature_value(dict(entry.get("bindings", {}))),
                _jsonable_signature_value(dict(entry.get("matched_atoms", {}))),
            ]
        )
        if bool(entry.get("label", False)):
            positive_firings.add(firing_key)
        else:
            negative_firings.add(firing_key)

    if not positive_firings and not negative_firings:
        positive_firings.update(_as_sorted_str_tuple(rule.get("positive_example_ids", [])))
        negative_firings.update(_as_sorted_str_tuple(rule.get("negative_example_ids", [])))

    return tuple(sorted(positive_firings)), tuple(sorted(negative_firings))


def _positive_coverage_key(rule: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    return (
        str(rule.get("head_predicate", "")),
        _as_sorted_str_tuple(rule.get("positive_example_ids", [])),
    )


def _initial_pruning_category(rule: Dict[str, Any]) -> str:
    pair_category = str(rule.get("initial_rule_pair_category", rule.get("extension_rule_category", "")))
    if pair_category == "candidate_candidate":
        return "candidate_candidate"
    category = str(rule.get("candidate_rule_category", "")) or rule_candidate_category(rule)
    if category == "mixed_accepted_candidate":
        return "accepted_candidate"
    if category == "candidate_only":
        if bool(rule.get("uses_only_candidate_atoms", False)) and int(rule.get("num_candidate_body_atoms", 0)) >= 2:
            return "candidate_candidate"
        return "candidate_only"
    return "accepted_only"


def _computed_rule_ratios(
    rule: Dict[str, Any],
    candidate_semantic_templates: set[str],
    candidate_provenance_templates: set[str],
) -> Dict[str, float]:
    body_templates = [
        str(atom)
        for atom in list(rule.get("body_atom_templates", []))
        if str(atom)
    ]
    if not body_templates and str(rule.get("body_atom_template", "")):
        body_templates = [str(rule.get("body_atom_template", ""))]
    body_length = max(1, int(rule.get("body_length", len(body_templates) or 1)))

    provenance_atom_count = sum(1 for atom in body_templates if atom in candidate_provenance_templates)
    semantic_atom_count = sum(1 for atom in body_templates if atom in candidate_semantic_templates)
    candidate_ratio = float(rule.get("candidate_body_atom_ratio", 0.0))
    provenance_ratio = float(provenance_atom_count / body_length)
    semantic_ratio = float(semantic_atom_count / body_length)
    return {
        "candidate_body_atom_ratio": candidate_ratio,
        "candidate_provenance_atom_ratio": provenance_ratio,
        "candidate_semantic_atom_ratio": semantic_ratio,
        "semantic_to_provenance_atom_ratio": float(semantic_atom_count / max(1, provenance_atom_count)),
    }


def _initial_rule_rank_key(rule: Dict[str, Any]) -> Tuple[Any, ...]:
    positive_support = int(rule.get("positive_support", 0))
    positive_firings = int(rule.get("positive_firings", 0))
    negative_firings = int(rule.get("negative_firings", 0))
    body_length = int(rule.get("body_length", len(list(rule.get("body_atom_templates", []))) or 1))
    candidate_ratio = float(rule.get("candidate_body_atom_ratio", 0.0))
    provenance_ratio = float(rule.get("candidate_provenance_atom_ratio", 0.0))
    confidence = float(rule.get("confidence", 0.0))
    source_video_count = len(_as_sorted_str_tuple(rule.get("source_video_ids", [])))
    source_rule_count = int(rule.get("num_source_rules", len(list(rule.get("source_rule_ids", [])))))
    category = _initial_pruning_category(rule)
    category_penalty = {
        "accepted_only": 0,
        "accepted_candidate": 1,
        "candidate_only": 2,
        "candidate_candidate": 3,
    }.get(category, 4)
    return (
        negative_firings,
        body_length,
        candidate_ratio,
        provenance_ratio,
        -confidence,
        -positive_support,
        -positive_firings,
        -source_video_count,
        -source_rule_count,
        category_penalty,
        str(rule.get("clause", "")),
    )


def _increment_initial_pruning_count(counts: Dict[str, int], category: str) -> None:
    counts["all_rules"] = counts.get("all_rules", 0) + 1
    if category == "accepted_candidate":
        counts["mixed_accepted_candidate_rules"] = counts.get("mixed_accepted_candidate_rules", 0) + 1
        counts["candidate_involving_rules"] = counts.get("candidate_involving_rules", 0) + 1
    elif category == "candidate_only":
        counts["candidate_only_rules"] = counts.get("candidate_only_rules", 0) + 1
        counts["candidate_involving_rules"] = counts.get("candidate_involving_rules", 0) + 1
    elif category == "candidate_candidate":
        counts["candidate_candidate_rules"] = counts.get("candidate_candidate_rules", 0) + 1
        counts["candidate_only_rules"] = counts.get("candidate_only_rules", 0) + 1
        counts["candidate_involving_rules"] = counts.get("candidate_involving_rules", 0) + 1
    else:
        counts["accepted_only_rules"] = counts.get("accepted_only_rules", 0) + 1


def _initial_pruning_counts(rules: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = empty_rule_category_counts()
    for rule in rules:
        _increment_initial_pruning_count(counts, _initial_pruning_category(rule))
    return counts


def _initial_budget_limit(cfg: Dict[str, Any], key: str, default: int) -> int:
    value = int(cfg.get(key, default))
    return value if value >= 0 else 10**12


def _cluster_diverse_select(
    rules: Sequence[Dict[str, Any]],
    max_rules: int,
    cluster_key_name: str,
) -> List[Dict[str, Any]]:
    if max_rules <= 0 or not rules:
        return []

    clusters: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for rule in rules:
        if cluster_key_name == "firing_signature":
            positive_firings, negative_firings = _rule_firing_sets(rule)
            cluster_key = (
                str(rule.get("head_predicate", "")),
                positive_firings,
                negative_firings,
            )
        else:
            cluster_key = _positive_coverage_key(rule)
        clusters.setdefault(cluster_key, []).append(rule)

    for cluster_rules in clusters.values():
        cluster_rules.sort(key=_initial_rule_rank_key)

    ordered_clusters = sorted(
        clusters.values(),
        key=lambda cluster_rules: (
            _initial_rule_rank_key(cluster_rules[0]),
            len(cluster_rules),
        ),
    )

    selected: List[Dict[str, Any]] = []
    while len(selected) < max_rules:
        made_progress = False
        for cluster_rules in ordered_clusters:
            if not cluster_rules:
                continue
            selected.append(cluster_rules.pop(0))
            made_progress = True
            if len(selected) >= max_rules:
                break
        if not made_progress:
            break
    return selected


def _initial_rule_csv_row(rule: Dict[str, Any]) -> Dict[str, Any]:
    return {
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
        "candidate_provenance_atom_ratio": rule.get("candidate_provenance_atom_ratio", 0.0),
        "candidate_semantic_atom_ratio": rule.get("candidate_semantic_atom_ratio", 0.0),
        "semantic_to_provenance_atom_ratio": rule.get("semantic_to_provenance_atom_ratio", 0.0),
        "matched_prior_ids_involved": json.dumps(rule.get("matched_prior_ids_involved", [])),
        "mixes_accepted_and_candidate_atoms": rule.get("mixes_accepted_and_candidate_atoms", False),
        "uses_only_candidate_atoms": rule.get("uses_only_candidate_atoms", False),
        "body_source_mix": rule.get("body_source_mix", ""),
        "candidate_rule_category": rule.get("candidate_rule_category", ""),
        "initial_rule_pair_category": rule.get("initial_rule_pair_category", ""),
        "initial_pruning_category": rule.get("initial_pruning_category", ""),
        "positive_example_ids": json.dumps(rule.get("positive_example_ids", [])),
        "negative_example_ids": json.dumps(rule.get("negative_example_ids", [])),
        "source_video_ids": json.dumps(rule.get("source_video_ids", [])),
        "source_rule_ids": json.dumps(rule.get("source_rule_ids", [])),
        "source_rule_indices": json.dumps(rule.get("source_rule_indices", [])),
        "num_source_rules": rule.get("num_source_rules", 0),
    }


def _write_pruned_initial_rules_csv(csv_path: Path, rules: Sequence[Dict[str, Any]]) -> None:
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
                "candidate_provenance_atom_ratio",
                "candidate_semantic_atom_ratio",
                "semantic_to_provenance_atom_ratio",
                "matched_prior_ids_involved",
                "mixes_accepted_and_candidate_atoms",
                "uses_only_candidate_atoms",
                "body_source_mix",
                "candidate_rule_category",
                "initial_rule_pair_category",
                "initial_pruning_category",
                "positive_example_ids",
                "negative_example_ids",
                "source_video_ids",
                "source_rule_ids",
                "source_rule_indices",
                "num_source_rules",
            ],
        )
        writer.writeheader()
        for rule in rules:
            writer.writerow(_initial_rule_csv_row(rule))


def prune_initial_rules(
    merged_initial_rules: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or pipeline_config.get_merged_candidate_rules_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "pruned_initial_rules.json"
    csv_path = out_root / "pruned_initial_rules.csv"
    summary_path = out_root / "initial_rule_pruning_summary.json"

    atom_availability = dict(
        dict(merged_initial_rules.get("candidate_rule_stage_stats", {})).get("atom_availability", {})
    )
    candidate_semantic_templates = {
        str(value)
        for value in list(atom_availability.get("candidate_semantic_body_atom_templates", []))
        if str(value)
    }
    candidate_provenance_templates = {
        str(value)
        for value in list(atom_availability.get("candidate_provenance_only_body_atom_templates", []))
        if str(value)
    }

    input_rules = list(merged_initial_rules.get("rules", []))
    annotated_rules: List[Dict[str, Any]] = []
    for rule in input_rules:
        annotated_rule = dict(rule)
        annotated_rule.update(
            _computed_rule_ratios(
                annotated_rule,
                candidate_semantic_templates=candidate_semantic_templates,
                candidate_provenance_templates=candidate_provenance_templates,
            )
        )
        annotated_rule["initial_pruning_category"] = _initial_pruning_category(annotated_rule)
        annotated_rules.append(annotated_rule)

    input_counts = _initial_pruning_counts(annotated_rules)

    signature_best: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], Dict[str, Any]] = {}
    signature_pruned_counts = empty_rule_category_counts()
    for rule in annotated_rules:
        positive_firings, negative_firings = _rule_firing_sets(rule)
        key = (str(rule.get("head_predicate", "")), positive_firings, negative_firings)
        existing = signature_best.get(key)
        if existing is None:
            signature_best[key] = rule
            continue
        if _initial_rule_rank_key(rule) < _initial_rule_rank_key(existing):
            _increment_initial_pruning_count(
                signature_pruned_counts,
                _initial_pruning_category(existing),
            )
            signature_best[key] = rule
        else:
            _increment_initial_pruning_count(
                signature_pruned_counts,
                _initial_pruning_category(rule),
            )

    deduplicated_rules = list(signature_best.values())
    deduplicated_counts = _initial_pruning_counts(deduplicated_rules)

    coverage_best: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
    dominance_pruned_counts = empty_rule_category_counts()
    for rule in deduplicated_rules:
        key = _positive_coverage_key(rule)
        existing = coverage_best.get(key)
        if existing is None:
            coverage_best[key] = rule
            continue
        if _initial_rule_rank_key(rule) < _initial_rule_rank_key(existing):
            _increment_initial_pruning_count(
                dominance_pruned_counts,
                _initial_pruning_category(existing),
            )
            coverage_best[key] = rule
        else:
            _increment_initial_pruning_count(
                dominance_pruned_counts,
                _initial_pruning_category(rule),
            )

    dominance_rules = list(coverage_best.values())
    dominance_counts = _initial_pruning_counts(dominance_rules)

    max_total_rules = _initial_budget_limit(cfg, "max_total_initial_rules", 8000)
    category_budgets = {
        "accepted_only": _initial_budget_limit(cfg, "max_accepted_only_initial_rules", 4000),
        "accepted_candidate": _initial_budget_limit(cfg, "max_mixed_candidate_initial_rules", 2000),
        "candidate_only": _initial_budget_limit(cfg, "max_candidate_only_initial_rules", 1500),
        "candidate_candidate": _initial_budget_limit(cfg, "max_candidate_candidate_initial_rules", 500),
    }
    diversity_key = str(cfg.get("diversity_key", "positive_coverage"))

    category_rules: Dict[str, List[Dict[str, Any]]] = {
        "accepted_only": [],
        "accepted_candidate": [],
        "candidate_only": [],
        "candidate_candidate": [],
    }
    for rule in dominance_rules:
        category_rules.setdefault(_initial_pruning_category(rule), []).append(rule)

    category_selected: Dict[str, List[Dict[str, Any]]] = {}
    budget_pruned_counts = empty_rule_category_counts()
    for category, rules in category_rules.items():
        selected = _cluster_diverse_select(
            rules=rules,
            max_rules=min(len(rules), category_budgets.get(category, 0)),
            cluster_key_name=diversity_key,
        )
        selected_ids = {id(rule) for rule in selected}
        for rule in rules:
            if id(rule) not in selected_ids:
                _increment_initial_pruning_count(budget_pruned_counts, category)
        category_selected[category] = selected

    accepted_order = [
        "accepted_only",
        "accepted_candidate",
        "candidate_only",
        "candidate_candidate",
    ]
    budget_candidates: List[Dict[str, Any]] = []
    for category in accepted_order:
        budget_candidates.extend(category_selected.get(category, []))

    kept_rules = _cluster_diverse_select(
        rules=budget_candidates,
        max_rules=min(len(budget_candidates), max_total_rules),
        cluster_key_name=diversity_key,
    )
    kept_ids = {id(rule) for rule in kept_rules}
    for rule in budget_candidates:
        if id(rule) not in kept_ids:
            _increment_initial_pruning_count(
                budget_pruned_counts,
                _initial_pruning_category(rule),
            )

    kept_rules = sorted(
        kept_rules,
        key=lambda rule: (
            str(rule.get("head_predicate", "")),
            str(rule.get("clause", "")),
        ),
    )
    for idx, rule in enumerate(kept_rules):
        rule["merged_rule_index"] = idx
        rule["pruned_initial_rule_index"] = idx

    kept_counts = _initial_pruning_counts(kept_rules)
    pruned_result = dict(merged_initial_rules)
    pruned_result["version"] = _PRUNED_INITIAL_RULES_VERSION
    pruned_result["source_version"] = merged_initial_rules.get("version")
    pruned_result["num_rules"] = len(kept_rules)
    pruned_result["num_rules_using_candidate_atoms"] = kept_counts["candidate_involving_rules"]
    pruned_result["num_candidate_only_rules"] = kept_counts["candidate_only_rules"]
    pruned_result["num_mixed_source_rules"] = kept_counts["mixed_accepted_candidate_rules"]
    pruned_result["rules"] = kept_rules

    summary = {
        "version": _PRUNED_INITIAL_RULES_VERSION,
        "stage": "step15b_initial_rule_pruning",
        "enabled": True,
        "config": {
            "max_total_initial_rules": max_total_rules,
            "category_budgets": category_budgets,
            "diversity_key": diversity_key,
        },
        "input_num_rules": len(annotated_rules),
        "input_rule_counts": input_counts,
        "deduplicated_num_rules": len(deduplicated_rules),
        "deduplicated_rule_counts": deduplicated_counts,
        "firing_signature_deduplicated_num_rules": input_counts["all_rules"] - deduplicated_counts["all_rules"],
        "firing_signature_deduplicated_rule_counts": signature_pruned_counts,
        "dominance_pruned_num_rules": deduplicated_counts["all_rules"] - dominance_counts["all_rules"],
        "dominance_pruned_rule_counts": dominance_pruned_counts,
        "dominance_pruned_remaining_num_rules": len(dominance_rules),
        "dominance_pruned_remaining_rule_counts": dominance_counts,
        "budget_pruned_num_rules": dominance_counts["all_rules"] - kept_counts["all_rules"],
        "budget_pruned_rule_counts": budget_pruned_counts,
        "kept_num_rules": len(kept_rules),
        "kept_rule_counts": kept_counts,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }

    stage_stats = dict(pruned_result.get("candidate_rule_stage_stats", {}))
    stage_stats["initial_rule_pruning"] = summary
    stage_stats["pruned_initial_rule_counts"] = kept_counts
    pruned_result["candidate_rule_stage_stats"] = stage_stats

    flow_summary = dict(pruned_result.get("candidate_rule_flow_summary", {}))
    flow_summary["pruning"] = {
        "initial_rule_pruning_pruned_rule_counts": {
            key: int(signature_pruned_counts.get(key, 0))
            + int(dominance_pruned_counts.get(key, 0))
            + int(budget_pruned_counts.get(key, 0))
            for key in empty_rule_category_counts()
        },
        "initial_rule_pruning_kept_rule_counts": kept_counts,
    }
    pruned_result["candidate_rule_flow_summary"] = flow_summary

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(pruned_result, fh, indent=2)
    _write_pruned_initial_rules_csv(csv_path, kept_rules)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Pruned initial rule JSON written to {json_path}")
    print(f"Pruned initial rule CSV written to {csv_path}")
    print(f"Initial rule pruning summary written to {summary_path}")
    return pruned_result


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
    total_skipped_non_unary_input_rules = 0
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
        unary_candidate_rules = [
            rule for rule in candidate_rules
            if _rule_body_length(rule) == 1
        ]
        skipped_non_unary_input_rules = len(candidate_rules) - len(unary_candidate_rules)
        video_summaries.append(
            {
                "video_id": video_id,
                "target_predicate": target_predicate,
                "num_initial_rules": len(unary_candidate_rules),
                "num_input_initial_rules_before_unary_filter": len(candidate_rules),
                "num_skipped_non_unary_input_rules": skipped_non_unary_input_rules,
                "num_examples": int(video_result.get("num_examples", 0)),
                "num_positive_examples": int(video_result.get("num_positive_examples", 0)),
                "num_negative_examples": int(video_result.get("num_negative_examples", 0)),
                "num_rules_using_candidate_atoms": int(video_result.get("num_rules_using_candidate_atoms", 0)),
                "num_candidate_only_rules": int(video_result.get("num_candidate_only_rules", 0)),
                "num_mixed_source_rules": int(video_result.get("num_mixed_source_rules", 0)),
            }
        )
        total_skipped_non_unary_input_rules += skipped_non_unary_input_rules
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
                    len(unary_candidate_rules) - int(video_result.get("num_rules_using_candidate_atoms", 0)),
                ),
                "candidate_only_rules": int(video_result.get("num_candidate_only_rules", 0)),
                "mixed_accepted_candidate_rules": int(video_result.get("num_mixed_source_rules", 0)),
                "candidate_involving_rules": int(video_result.get("num_rules_using_candidate_atoms", 0)),
                "all_rules": len(unary_candidate_rules),
            }
        for key in input_generated_counts:
            input_generated_counts[key] += int(generated_counts.get(key, 0))

        for rule in unary_candidate_rules:
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
                    "initial_rule_pair_category": rule.get("initial_rule_pair_category", ""),
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
            if not str(merged_rule.get("initial_rule_pair_category", "")):
                merged_rule["initial_rule_pair_category"] = rule.get("initial_rule_pair_category", "")
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
        if not str(merged_rule.get("initial_rule_pair_category", "")):
            if bool(merged_rule.get("mixes_accepted_and_candidate_atoms", False)):
                merged_rule["initial_rule_pair_category"] = "accepted_candidate"
            elif (
                bool(merged_rule.get("uses_only_candidate_atoms", False))
                and int(merged_rule.get("num_candidate_body_atoms", 0)) >= 2
            ):
                merged_rule["initial_rule_pair_category"] = "candidate_candidate"
            elif bool(merged_rule.get("uses_candidate_atoms", False)):
                merged_rule["initial_rule_pair_category"] = "candidate_only"
            else:
                merged_rule["initial_rule_pair_category"] = "accepted_only"
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
        "num_skipped_non_unary_input_rules": total_skipped_non_unary_input_rules,
        "num_rules_using_candidate_atoms": merged_category_counts["candidate_involving_rules"],
        "num_candidate_only_rules": merged_category_counts["candidate_only_rules"],
        "num_mixed_source_rules": merged_category_counts["mixed_accepted_candidate_rules"],
        "candidate_rule_stage_stats": {
            "stage": "step15_merge_initial_rules",
            "skipped_non_unary_input_rules": total_skipped_non_unary_input_rules,
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
        "num_skipped_non_unary_input_rules": total_skipped_non_unary_input_rules,
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
                "initial_rule_pair_category",
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
                    "initial_rule_pair_category": rule.get("initial_rule_pair_category", ""),
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
