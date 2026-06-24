"""
Select a diverse final rule set by greedily maximizing new positive coverage
while penalizing overlap and repeated rule families.

Consumes:
  - Step 16 output: aggregated kept rules from initial + extended rounds

Output layout:
    pipeline_output/17b_driving_mini_diverse_final_rules/
        diverse_final_rules.json
        diverse_final_rules.csv
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_DIVERSE_FINAL_RULES_VERSION = 6
_STRONG_SEMANTIC_BODY_PREDICATES = {
    "object_class",
    "object_distance_state",
    "object_vz_state",
    "object_vx_state",
    "object_speed_state",
    "object_visibility_state",
    "traffic_control_type",
    "traffic_control_relevance_state",
    "traffic_control_relevant",
    "traffic_light_relevant",
    "stop_sign_relevant",
    "traffic_light_state",
}
_BROAD_CANDIDATE_BODY_PREDICATES = {
    "object_x_position_state",
    "object_source_type",
    "object_is_candidate",
    "object_candidate_score_state",
    "object_prior_relevance_state",
    "object_matched_prior",
    "traffic_control_front_center_region",
    "traffic_light_position_state",
}


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "17b_driving_mini_diverse_final_rules"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    semantic_min_family_counts = cfg.get("semantic_min_family_counts", {})
    return {
        "top_k": int(cfg.get("top_k", 50)),
        "score_mode": str(cfg.get("score_mode", "legacy_diverse_positive_coverage")),
        "selection_method_name": str(cfg.get("selection_method_name", "greedy_diverse_positive_coverage")),
        "output_prefix": str(cfg.get("output_prefix", "diverse_final_rules")),
        "new_positive_weight": float(cfg.get("new_positive_weight", 1.0)),
        "confidence_weight": float(cfg.get("confidence_weight", 0.25)),
        "coverage_weight": float(cfg.get("coverage_weight", 1.0)),
        "quality_weight": float(cfg.get("quality_weight", 1.0)),
        "overlap_penalty": float(cfg.get("overlap_penalty", 0.35)),
        "family_penalty": float(cfg.get("family_penalty", 0.75)),
        "family_diversity_bonus": float(cfg.get("family_diversity_bonus", 0.75)),
        "negative_support_penalty": float(cfg.get("negative_support_penalty", 0.1)),
        "weak_candidate_rule_penalty": float(cfg.get("weak_candidate_rule_penalty", 1.0)),
        "semantic_hard_constraints": bool(cfg.get("semantic_hard_constraints", False)),
        "semantic_bonus_weight": float(cfg.get("semantic_bonus_weight", 1.0)),
        "semantic_min_positive_support": int(cfg.get("semantic_min_positive_support", 1)),
        "semantic_min_total_support": int(cfg.get("semantic_min_total_support", 1)),
        "semantic_min_confidence": float(cfg.get("semantic_min_confidence", 0.0)),
        "semantic_min_family_counts": {
            str(key): int(value)
            for key, value in sorted(dict(semantic_min_family_counts).items())
            if int(value) > 0
        },
        "vehicle_classes": sorted(str(v) for v in cfg.get("vehicle_classes", [])),
        "near_states": sorted(str(v) for v in cfg.get("near_states", [])),
        "center_states": sorted(str(v) for v in cfg.get("center_states", [])),
    }


def _sort_rules(all_kept_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        all_kept_rules,
        key=lambda rule: (
            int(bool(rule.get("is_weak_candidate_rule", False))),
            -float(rule.get("confidence", 0.0)),
            -int(rule.get("positive_support", 0)),
            int(bool(rule.get("is_broad_candidate_rule", False))),
            int(rule.get("kept_round_index", 0)),
            str(rule.get("clause", "")),
        ),
    )


def _rule_candidate_category(rule: Dict[str, Any]) -> str:
    if bool(rule.get("mixes_accepted_and_candidate_atoms", False)):
        return "mixed_accepted_candidate"
    if bool(rule.get("uses_only_candidate_atoms", False)):
        return "candidate_only"
    if bool(rule.get("uses_candidate_atoms", False)):
        return "candidate_only"
    return "accepted_only"


def _candidate_rule_semantic_profile(rule: Dict[str, Any]) -> Dict[str, Any]:
    predicates = {
        predicate
        for predicate in (_parse_atom(atom)[0] if _parse_atom(atom) is not None else None for atom in _get_rule_body_atom_templates(rule))
        if predicate
    }
    non_segment_predicates = {
        predicate
        for predicate in predicates
        if not str(predicate).startswith("segment_")
        and str(predicate) not in {"object_in_segment", "object_track"}
    }
    has_strong_semantic_atom = bool(non_segment_predicates & _STRONG_SEMANTIC_BODY_PREDICATES)
    uses_candidate_atoms = bool(rule.get("uses_candidate_atoms", False))
    is_weak_candidate_rule = uses_candidate_atoms and not has_strong_semantic_atom
    broad_candidate_predicates = sorted(non_segment_predicates & _BROAD_CANDIDATE_BODY_PREDICATES)
    object_x_only = non_segment_predicates == {"object_x_position_state"}
    if not uses_candidate_atoms:
        broad_pattern = ""
    elif has_strong_semantic_atom:
        broad_pattern = "strong_semantic_candidate"
    elif object_x_only:
        broad_pattern = "object_x_position_state_only"
    elif non_segment_predicates and non_segment_predicates <= _BROAD_CANDIDATE_BODY_PREDICATES:
        if "object_x_position_state" in non_segment_predicates:
            broad_pattern = "position_provenance_prior_score_only"
        else:
            broad_pattern = "prior_score_provenance_only"
    else:
        broad_pattern = "weak_candidate_other"
    return {
        "body_predicates": sorted(predicates),
        "strong_candidate_semantic_predicates": sorted(
            non_segment_predicates & _STRONG_SEMANTIC_BODY_PREDICATES
        ),
        "broad_candidate_predicates": broad_candidate_predicates,
        "has_strong_candidate_semantic_atom": has_strong_semantic_atom,
        "is_weak_candidate_rule": is_weak_candidate_rule,
        "is_broad_candidate_rule": uses_candidate_atoms and is_weak_candidate_rule,
        "broad_candidate_rule_pattern": broad_pattern,
    }


def _enrich_rule_candidate_semantics(rule: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(rule)
    enriched.update(_candidate_rule_semantic_profile(enriched))
    return enriched


def _summarize_candidate_rule_subset(rules: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rules_list = list(rules)
    matched_prior_id_counts: Dict[str, int] = {}
    broad_candidate_pattern_counts: Dict[str, int] = {}
    total_candidate_body_atoms = 0
    candidate_body_atom_ratios: List[float] = []
    candidate_rule_count = 0
    weak_candidate_rule_count = 0
    broad_candidate_rule_count = 0
    for rule in rules_list:
        if bool(rule.get("uses_candidate_atoms", False)):
            candidate_rule_count += 1
        if bool(rule.get("is_weak_candidate_rule", False)):
            weak_candidate_rule_count += 1
        if bool(rule.get("is_broad_candidate_rule", False)):
            broad_candidate_rule_count += 1
        pattern = str(rule.get("broad_candidate_rule_pattern", "")).strip()
        if pattern:
            broad_candidate_pattern_counts[pattern] = broad_candidate_pattern_counts.get(pattern, 0) + 1
        total_candidate_body_atoms += max(0, int(rule.get("num_candidate_body_atoms", 0)))
        candidate_body_atom_ratios.append(max(0.0, float(rule.get("candidate_body_atom_ratio", 0.0))))
        for prior_id in list(rule.get("matched_prior_ids_involved", [])):
            prior_text = str(prior_id).strip()
            if prior_text:
                matched_prior_id_counts[prior_text] = matched_prior_id_counts.get(prior_text, 0) + 1
    return {
        "num_rules": len(rules_list),
        "num_rules_using_candidate_atoms": candidate_rule_count,
        "num_weak_candidate_rules": weak_candidate_rule_count,
        "num_broad_candidate_rules": broad_candidate_rule_count,
        "total_candidate_body_atoms": total_candidate_body_atoms,
        "avg_candidate_body_atom_ratio": float(sum(candidate_body_atom_ratios) / max(1, len(candidate_body_atom_ratios))),
        "avg_num_candidate_body_atoms": float(total_candidate_body_atoms / max(1, len(rules_list))),
        "broad_candidate_rule_pattern_counts": {
            key: broad_candidate_pattern_counts[key]
            for key in sorted(broad_candidate_pattern_counts)
        },
        "matched_prior_id_counts": {
            key: matched_prior_id_counts[key]
            for key in sorted(matched_prior_id_counts)
        },
    }


def _summarize_candidate_rule_selection(rules: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rules_list = list(rules)
    subset_rules = {
        "all_rules": rules_list,
        "accepted_only_rules": [rule for rule in rules_list if _rule_candidate_category(rule) == "accepted_only"],
        "candidate_only_rules": [rule for rule in rules_list if _rule_candidate_category(rule) == "candidate_only"],
        "mixed_accepted_candidate_rules": [
            rule for rule in rules_list if _rule_candidate_category(rule) == "mixed_accepted_candidate"
        ],
    }
    return {
        "category_counts": {
            "accepted_only_rules": len(subset_rules["accepted_only_rules"]),
            "candidate_only_rules": len(subset_rules["candidate_only_rules"]),
            "mixed_accepted_candidate_rules": len(subset_rules["mixed_accepted_candidate_rules"]),
            "all_rules": len(subset_rules["all_rules"]),
        },
        "subsets": {
            subset_name: _summarize_candidate_rule_subset(subset)
            for subset_name, subset in subset_rules.items()
        },
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


def _rule_family_signature(rule: Dict[str, Any]) -> str:
    predicates: Set[str] = set()
    for atom in _get_rule_body_atom_templates(rule):
        parsed = _parse_atom(atom)
        if parsed is None:
            continue
        predicates.add(str(parsed[0]))
    return "|".join(sorted(predicates))


def _positive_example_ids(rule: Dict[str, Any]) -> Set[str]:
    return {
        str(example_id)
        for example_id in list(rule.get("positive_example_ids", []))
        if str(example_id)
    }


def _base_quality_score(rule: Dict[str, Any]) -> float:
    confidence = max(0.0, float(rule.get("confidence", 0.0)))
    positive_support = max(0, int(rule.get("positive_support", 0)))
    return confidence * math.log1p(positive_support)


def _rule_vehicle_match_level(rule: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    vehicle_classes = {
        str(v)
        for v in cfg.get("vehicle_classes", ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"])
    }
    near_states = {str(v) for v in cfg.get("near_states", ["near"])}
    center_states = {str(v) for v in cfg.get("center_states", ["centered"])}

    has_vehicle = False
    has_near = False
    has_centered = False
    for atom in _get_rule_body_atom_templates(rule):
        parsed = _parse_atom(atom)
        if parsed is None:
            continue
        predicate, args = parsed
        if predicate == "object_class" and len(args) >= 3 and str(args[2]) in vehicle_classes:
            has_vehicle = True
        elif predicate == "object_distance_state" and len(args) >= 3 and str(args[2]) in near_states:
            has_near = True
        elif predicate == "object_x_position_state" and len(args) >= 3 and str(args[2]) in center_states:
            has_centered = True

    if has_vehicle and has_near and has_centered:
        return "exact_vehicle_near_centered"
    if has_vehicle and has_near:
        return "vehicle_near_partial"
    if has_vehicle and has_centered:
        return "vehicle_centered_partial"
    if has_near and has_centered:
        return "near_centered_partial"
    if has_vehicle:
        return "vehicle_only"
    if has_near:
        return "near_only"
    if has_centered:
        return "centered_only"
    return "no_match"


def _legacy_diverse_utility(
    rule: Dict[str, Any],
    covered_positive_ids: Set[str],
    family_counts: Dict[str, int],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    positive_ids = _positive_example_ids(rule)
    family_signature = _rule_family_signature(rule)
    new_positive_ids = positive_ids - covered_positive_ids
    overlap_positive_ids = positive_ids & covered_positive_ids
    family_reuse_count = int(family_counts.get(family_signature, 0))
    confidence = float(rule.get("confidence", 0.0))
    negative_support = int(rule.get("negative_support", 0))

    new_positive_gain = len(new_positive_ids)
    overlap_positive_count = len(overlap_positive_ids)
    weak_candidate_penalty_value = (
        float(cfg.get("weak_candidate_rule_penalty", 1.0))
        if bool(rule.get("is_weak_candidate_rule", False))
        else 0.0
    )
    utility = (
        float(cfg.get("new_positive_weight", 1.0)) * new_positive_gain
        + float(cfg.get("confidence_weight", 0.25)) * confidence
        - float(cfg.get("overlap_penalty", 0.35)) * overlap_positive_count
        - float(cfg.get("family_penalty", 0.75)) * family_reuse_count
        - float(cfg.get("negative_support_penalty", 0.1)) * negative_support
        - weak_candidate_penalty_value
    )
    return {
        "utility": utility,
        "family_signature": family_signature,
        "new_positive_ids": new_positive_ids,
        "new_positive_gain": new_positive_gain,
        "overlap_positive_count": overlap_positive_count,
        "family_reuse_count": family_reuse_count,
        "base_quality_score": _base_quality_score(rule),
        "family_diversity_bonus": 0.0,
        "negative_support_penalty": float(cfg.get("negative_support_penalty", 0.1)) * negative_support,
        "weak_candidate_penalty_value": weak_candidate_penalty_value,
        "overlap_penalty_value": float(cfg.get("overlap_penalty", 0.35)) * overlap_positive_count,
        "family_penalty_value": float(cfg.get("family_penalty", 0.75)) * family_reuse_count,
    }


def _coverage_family_aware_utility(
    rule: Dict[str, Any],
    covered_positive_ids: Set[str],
    family_counts: Dict[str, int],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    positive_ids = _positive_example_ids(rule)
    family_signature = _rule_family_signature(rule)
    new_positive_ids = positive_ids - covered_positive_ids
    overlap_positive_ids = positive_ids & covered_positive_ids
    family_reuse_count = int(family_counts.get(family_signature, 0))
    negative_support = max(0, int(rule.get("negative_support", 0)))

    new_positive_gain = len(new_positive_ids)
    overlap_positive_count = len(overlap_positive_ids)
    quality_score = _base_quality_score(rule)
    family_diversity_bonus = float(cfg.get("family_diversity_bonus", 0.75)) / float(1 + family_reuse_count)
    weak_candidate_penalty_value = (
        float(cfg.get("weak_candidate_rule_penalty", 1.0))
        if bool(rule.get("is_weak_candidate_rule", False))
        else 0.0
    )
    utility = (
        float(cfg.get("coverage_weight", 1.0)) * new_positive_gain
        + float(cfg.get("quality_weight", 1.0)) * quality_score
        + family_diversity_bonus
        - float(cfg.get("overlap_penalty", 0.35)) * overlap_positive_count
        - float(cfg.get("family_penalty", 0.75)) * family_reuse_count
        - float(cfg.get("negative_support_penalty", 0.1)) * negative_support
        - weak_candidate_penalty_value
    )
    return {
        "utility": utility,
        "family_signature": family_signature,
        "new_positive_ids": new_positive_ids,
        "new_positive_gain": new_positive_gain,
        "overlap_positive_count": overlap_positive_count,
        "family_reuse_count": family_reuse_count,
        "base_quality_score": quality_score,
        "family_diversity_bonus": family_diversity_bonus,
        "negative_support_penalty": float(cfg.get("negative_support_penalty", 0.1)) * negative_support,
        "weak_candidate_penalty_value": weak_candidate_penalty_value,
        "overlap_penalty_value": float(cfg.get("overlap_penalty", 0.35)) * overlap_positive_count,
        "family_penalty_value": float(cfg.get("family_penalty", 0.75)) * family_reuse_count,
    }


def _semantic_family_deficits(
    selected_semantic_families: Dict[str, Set[str]],
    effective_semantic_min_family_counts: Dict[str, int],
) -> Dict[str, int]:
    deficits: Dict[str, int] = {}
    for match_level, min_count in effective_semantic_min_family_counts.items():
        required = max(0, int(min_count))
        if required <= 0:
            continue
        selected_count = len(selected_semantic_families.get(str(match_level), set()))
        deficits[str(match_level)] = max(0, required - selected_count)
    return deficits


def _rule_is_semantic_quota_qualified(rule: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    positive_support = max(0, int(rule.get("positive_support", 0)))
    total_support = max(0, int(rule.get("total_support", positive_support + max(0, int(rule.get("negative_support", 0))))))
    confidence = max(0.0, float(rule.get("confidence", 0.0)))
    return (
        positive_support >= int(cfg.get("semantic_min_positive_support", 1))
        and total_support >= int(cfg.get("semantic_min_total_support", 1))
        and confidence >= float(cfg.get("semantic_min_confidence", 0.0))
    )


def _effective_semantic_min_family_counts(
    candidate_rules: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Dict[str, int]:
    requested_counts = {
        str(match_level): max(0, int(min_count))
        for match_level, min_count in dict(cfg.get("semantic_min_family_counts", {})).items()
        if int(min_count) > 0
    }
    qualified_families_by_level: Dict[str, Set[str]] = {}
    for rule in candidate_rules:
        if not _rule_is_semantic_quota_qualified(rule, cfg):
            continue
        semantic_match_level = _rule_vehicle_match_level(rule, cfg)
        if semantic_match_level == "no_match":
            continue
        qualified_families_by_level.setdefault(semantic_match_level, set()).add(_rule_family_signature(rule))

    return {
        match_level: min(requested_count, len(qualified_families_by_level.get(match_level, set())))
        for match_level, requested_count in requested_counts.items()
    }


def _semantic_constrained_diverse_utility(
    rule: Dict[str, Any],
    covered_positive_ids: Set[str],
    family_counts: Dict[str, int],
    selected_qualified_semantic_families: Dict[str, Set[str]],
    effective_semantic_min_family_counts: Dict[str, int],
    remaining_slots: int,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    base = _legacy_diverse_utility(rule, covered_positive_ids, family_counts, cfg)
    family_signature = str(base["family_signature"])
    semantic_match_level = _rule_vehicle_match_level(rule, cfg)
    semantic_is_quota_qualified = _rule_is_semantic_quota_qualified(rule, cfg)
    deficits = _semantic_family_deficits(selected_qualified_semantic_families, effective_semantic_min_family_counts)
    total_remaining_deficits = sum(deficits.values())
    already_selected_for_level = family_signature in selected_qualified_semantic_families.get(semantic_match_level, set())
    semantic_deficit_reduction = 0
    if (
        semantic_is_quota_qualified
        and semantic_match_level in deficits
        and deficits[semantic_match_level] > 0
        and not already_selected_for_level
    ):
        semantic_deficit_reduction = 1

    hard_constraint_active = bool(cfg.get("semantic_hard_constraints", False)) and remaining_slots <= total_remaining_deficits and total_remaining_deficits > 0
    if hard_constraint_active and semantic_deficit_reduction == 0:
        utility = -1e12
    else:
        utility = float(base["utility"]) + float(cfg.get("semantic_bonus_weight", 1.0)) * semantic_deficit_reduction

    result = dict(base)
    result["utility"] = utility
    result["semantic_match_level"] = semantic_match_level
    result["semantic_is_quota_qualified"] = semantic_is_quota_qualified
    result["semantic_deficit_reduction"] = semantic_deficit_reduction
    result["semantic_hard_constraint_active"] = hard_constraint_active
    result["semantic_total_remaining_deficits"] = total_remaining_deficits
    return result


def _rule_utility(
    rule: Dict[str, Any],
    covered_positive_ids: Set[str],
    family_counts: Dict[str, int],
    selected_qualified_semantic_families: Dict[str, Set[str]],
    effective_semantic_min_family_counts: Dict[str, int],
    remaining_slots: int,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    score_mode = str(cfg.get("score_mode", "legacy_diverse_positive_coverage"))
    if score_mode == "legacy_diverse_positive_coverage":
        return _legacy_diverse_utility(rule, covered_positive_ids, family_counts, cfg)
    if score_mode == "coverage_family_aware":
        return _coverage_family_aware_utility(rule, covered_positive_ids, family_counts, cfg)
    if score_mode == "semantic_constrained_diverse":
        return _semantic_constrained_diverse_utility(
            rule,
            covered_positive_ids,
            family_counts,
            selected_qualified_semantic_families,
            effective_semantic_min_family_counts,
            remaining_slots,
            cfg,
        )
    raise ValueError(f"Unsupported diverse final rule score_mode: {score_mode}")


def process_rules(
    extended_rule_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    top_k = int(cfg.get("top_k", 50))
    output_prefix = str(cfg.get("output_prefix", "diverse_final_rules")).strip() or "diverse_final_rules"
    selection_method_name = str(cfg.get("selection_method_name", "greedy_diverse_positive_coverage"))

    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / f"{output_prefix}.json"
    csv_path = out_root / f"{output_prefix}.csv"

    if not force_recompute and json_path.exists():
        with json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _DIVERSE_FINAL_RULES_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {json_path.name}")
            return cached

    candidate_rules = _sort_rules(
        [_enrich_rule_candidate_semantics(rule) for rule in list(extended_rule_results.get("all_kept_rules", []))]
    )
    selected_rules: List[Dict[str, Any]] = []
    selection_trace: List[Dict[str, Any]] = []
    selected_rule_ids: Set[str] = set()
    covered_positive_ids: Set[str] = set()
    family_counts: Dict[str, int] = {}
    selected_semantic_families: Dict[str, Set[str]] = {}
    selected_qualified_semantic_families: Dict[str, Set[str]] = {}
    effective_semantic_min_family_counts = _effective_semantic_min_family_counts(candidate_rules, cfg)

    while len(selected_rules) < max(0, top_k):
        best_rule: Optional[Dict[str, Any]] = None
        best_trace: Optional[Dict[str, Any]] = None
        remaining_slots = max(0, top_k - len(selected_rules))

        for rule in candidate_rules:
            rule_id = str(rule.get("rule_id", ""))
            if rule_id in selected_rule_ids:
                continue
            utility = _rule_utility(
                rule,
                covered_positive_ids,
                family_counts,
                selected_qualified_semantic_families,
                effective_semantic_min_family_counts,
                remaining_slots,
                cfg,
            )
            trace = {
                "rule_id": rule_id,
                "clause": str(rule.get("clause", "")),
                "utility": float(utility["utility"]),
                "new_positive_gain": int(utility["new_positive_gain"]),
                "overlap_positive_count": int(utility["overlap_positive_count"]),
                "family_reuse_count": int(utility["family_reuse_count"]),
                "family_signature": str(utility["family_signature"]),
                "base_quality_score": float(utility["base_quality_score"]),
                "family_diversity_bonus": float(utility["family_diversity_bonus"]),
                "overlap_penalty_value": float(utility["overlap_penalty_value"]),
                "family_penalty_value": float(utility["family_penalty_value"]),
                "negative_support_penalty_value": float(utility["negative_support_penalty"]),
                "weak_candidate_penalty_value": float(utility.get("weak_candidate_penalty_value", 0.0)),
                "semantic_match_level": str(utility.get("semantic_match_level", "no_match")),
                "semantic_is_quota_qualified": bool(utility.get("semantic_is_quota_qualified", False)),
                "semantic_deficit_reduction": int(utility.get("semantic_deficit_reduction", 0)),
                "semantic_hard_constraint_active": bool(utility.get("semantic_hard_constraint_active", False)),
                "semantic_total_remaining_deficits": int(utility.get("semantic_total_remaining_deficits", 0)),
                "confidence": float(rule.get("confidence", 0.0)),
                "positive_support": int(rule.get("positive_support", 0)),
                "negative_support": int(rule.get("negative_support", 0)),
                "new_positive_ids": sorted(str(v) for v in utility["new_positive_ids"]),
            }
            if best_rule is None:
                best_rule = rule
                best_trace = trace
                continue
            assert best_trace is not None
            current_key = (
                float(trace["utility"]),
                int(trace["new_positive_gain"]),
                -int(trace["family_reuse_count"]),
                float(trace["confidence"]),
                int(trace["positive_support"]),
                str(trace["clause"]),
            )
            best_key = (
                float(best_trace["utility"]),
                int(best_trace["new_positive_gain"]),
                -int(best_trace["family_reuse_count"]),
                float(best_trace["confidence"]),
                int(best_trace["positive_support"]),
                str(best_trace["clause"]),
            )
            if current_key > best_key:
                best_rule = rule
                best_trace = trace

        if best_rule is None or best_trace is None:
            break

        selected_rule = dict(best_rule)
        selected_rule["selection_rank"] = len(selected_rules)
        selected_rule["selection_method"] = selection_method_name
        selected_rule["selection_utility"] = float(best_trace["utility"])
        selected_rule["selection_family_signature"] = str(best_trace["family_signature"])
        selected_rule["selection_new_positive_gain"] = int(best_trace["new_positive_gain"])
        selected_rule["selection_overlap_positive_count"] = int(best_trace["overlap_positive_count"])
        selected_rule["selection_family_reuse_count"] = int(best_trace["family_reuse_count"])
        selected_rule["selection_base_quality_score"] = float(best_trace["base_quality_score"])
        selected_rule["selection_family_diversity_bonus"] = float(best_trace["family_diversity_bonus"])
        selected_rule["selection_overlap_penalty_value"] = float(best_trace["overlap_penalty_value"])
        selected_rule["selection_family_penalty_value"] = float(best_trace["family_penalty_value"])
        selected_rule["selection_negative_support_penalty_value"] = float(best_trace["negative_support_penalty_value"])
        selected_rule["selection_weak_candidate_penalty_value"] = float(best_trace.get("weak_candidate_penalty_value", 0.0))
        selected_rule["selection_semantic_match_level"] = str(best_trace["semantic_match_level"])
        selected_rule["selection_semantic_is_quota_qualified"] = bool(best_trace["semantic_is_quota_qualified"])
        selected_rule["selection_semantic_deficit_reduction"] = int(best_trace["semantic_deficit_reduction"])
        selected_rule["candidate_rule_category"] = _rule_candidate_category(selected_rule)
        selected_rules.append(selected_rule)

        selected_rule_ids.add(str(best_rule.get("rule_id", "")))
        covered_positive_ids.update(best_trace["new_positive_ids"])
        family_signature = str(best_trace["family_signature"])
        family_counts[family_signature] = family_counts.get(family_signature, 0) + 1
        semantic_match_level = str(best_trace["semantic_match_level"])
        if semantic_match_level != "no_match":
            selected_semantic_families.setdefault(semantic_match_level, set()).add(family_signature)
            if bool(best_trace["semantic_is_quota_qualified"]):
                selected_qualified_semantic_families.setdefault(semantic_match_level, set()).add(family_signature)

        selection_trace.append(
            {
                "selection_rank": len(selected_rules) - 1,
                "rule_id": str(best_rule.get("rule_id", "")),
                "clause": str(best_rule.get("clause", "")),
                "selection_utility": float(best_trace["utility"]),
                "new_positive_gain": int(best_trace["new_positive_gain"]),
                "overlap_positive_count": int(best_trace["overlap_positive_count"]),
                "family_reuse_count": int(best_trace["family_reuse_count"]),
                "family_signature": family_signature,
                "base_quality_score": float(best_trace["base_quality_score"]),
                "family_diversity_bonus": float(best_trace["family_diversity_bonus"]),
                "overlap_penalty_value": float(best_trace["overlap_penalty_value"]),
                "family_penalty_value": float(best_trace["family_penalty_value"]),
                "negative_support_penalty_value": float(best_trace["negative_support_penalty_value"]),
                "weak_candidate_penalty_value": float(best_trace.get("weak_candidate_penalty_value", 0.0)),
                "semantic_match_level": semantic_match_level,
                "semantic_is_quota_qualified": bool(best_trace["semantic_is_quota_qualified"]),
                "semantic_deficit_reduction": int(best_trace["semantic_deficit_reduction"]),
                "cumulative_positive_coverage": len(covered_positive_ids),
            }
        )

    candidate_rule_diagnostics = _summarize_candidate_rule_selection(selected_rules)

    result: Dict[str, Any] = {
        "version": _DIVERSE_FINAL_RULES_VERSION,
        "config": _cfg_key_subset(cfg),
        "selection_method": selection_method_name,
        "num_input_rules": len(candidate_rules),
        "num_final_rules": len(selected_rules),
        "num_distinct_families": len(family_counts),
        "covered_training_positive_examples": len(covered_positive_ids),
        "effective_semantic_min_family_counts": effective_semantic_min_family_counts,
        "selected_semantic_family_counts": {
            match_level: len(families)
            for match_level, families in sorted(selected_semantic_families.items())
        },
        "selected_qualified_semantic_family_counts": {
            match_level: len(families)
            for match_level, families in sorted(selected_qualified_semantic_families.items())
        },
        "candidate_rule_diagnostics": candidate_rule_diagnostics,
        "final_rules": selected_rules,
        "selection_trace": selection_trace,
    }

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "selection_rank",
                "rule_id",
                "kept_stage",
                "kept_round_index",
                "clause",
                "confidence",
                "positive_support",
                "negative_support",
                "total_support",
                "selection_utility",
                "selection_family_signature",
                "selection_new_positive_gain",
                "selection_overlap_positive_count",
                "selection_family_reuse_count",
                "selection_base_quality_score",
                "selection_family_diversity_bonus",
                "selection_overlap_penalty_value",
                "selection_family_penalty_value",
                "selection_negative_support_penalty_value",
                "selection_semantic_match_level",
                "selection_semantic_is_quota_qualified",
                "selection_semantic_deficit_reduction",
                "selection_weak_candidate_penalty_value",
                "candidate_rule_category",
                "uses_candidate_atoms",
                "num_candidate_body_atoms",
                "candidate_body_atom_ratio",
                "mixes_accepted_and_candidate_atoms",
                "uses_only_candidate_atoms",
                "body_source_mix",
                "matched_prior_ids_involved",
                "has_strong_candidate_semantic_atom",
                "is_weak_candidate_rule",
                "is_broad_candidate_rule",
                "broad_candidate_rule_pattern",
                "strong_candidate_semantic_predicates",
            ],
        )
        writer.writeheader()
        for rule in selected_rules:
            writer.writerow(
                {
                    "selection_rank": rule.get("selection_rank", ""),
                    "rule_id": rule.get("rule_id", ""),
                    "kept_stage": rule.get("kept_stage", ""),
                    "kept_round_index": rule.get("kept_round_index", ""),
                    "clause": rule.get("clause", ""),
                    "confidence": rule.get("confidence", 0.0),
                    "positive_support": rule.get("positive_support", 0),
                    "negative_support": rule.get("negative_support", 0),
                    "total_support": rule.get("total_support", 0),
                    "selection_utility": rule.get("selection_utility", 0.0),
                    "selection_family_signature": rule.get("selection_family_signature", ""),
                    "selection_new_positive_gain": rule.get("selection_new_positive_gain", 0),
                    "selection_overlap_positive_count": rule.get("selection_overlap_positive_count", 0),
                    "selection_family_reuse_count": rule.get("selection_family_reuse_count", 0),
                    "selection_base_quality_score": rule.get("selection_base_quality_score", 0.0),
                    "selection_family_diversity_bonus": rule.get("selection_family_diversity_bonus", 0.0),
                    "selection_overlap_penalty_value": rule.get("selection_overlap_penalty_value", 0.0),
                    "selection_family_penalty_value": rule.get("selection_family_penalty_value", 0.0),
                    "selection_negative_support_penalty_value": rule.get("selection_negative_support_penalty_value", 0.0),
                    "selection_semantic_match_level": rule.get("selection_semantic_match_level", ""),
                    "selection_semantic_is_quota_qualified": rule.get("selection_semantic_is_quota_qualified", False),
                    "selection_semantic_deficit_reduction": rule.get("selection_semantic_deficit_reduction", 0),
                    "selection_weak_candidate_penalty_value": rule.get("selection_weak_candidate_penalty_value", 0.0),
                    "candidate_rule_category": rule.get("candidate_rule_category", "accepted_only"),
                    "uses_candidate_atoms": rule.get("uses_candidate_atoms", False),
                    "num_candidate_body_atoms": rule.get("num_candidate_body_atoms", 0),
                    "candidate_body_atom_ratio": rule.get("candidate_body_atom_ratio", 0.0),
                    "mixes_accepted_and_candidate_atoms": rule.get("mixes_accepted_and_candidate_atoms", False),
                    "uses_only_candidate_atoms": rule.get("uses_only_candidate_atoms", False),
                    "body_source_mix": rule.get("body_source_mix", ""),
                    "matched_prior_ids_involved": json.dumps(rule.get("matched_prior_ids_involved", [])),
                    "has_strong_candidate_semantic_atom": rule.get("has_strong_candidate_semantic_atom", False),
                    "is_weak_candidate_rule": rule.get("is_weak_candidate_rule", False),
                    "is_broad_candidate_rule": rule.get("is_broad_candidate_rule", False),
                    "broad_candidate_rule_pattern": rule.get("broad_candidate_rule_pattern", ""),
                    "strong_candidate_semantic_predicates": json.dumps(
                        rule.get("strong_candidate_semantic_predicates", [])
                    ),
                }
            )

    print(
        f"  {selection_method_name}: "
        f"input={len(candidate_rules)} | "
        f"top_k={top_k} | "
        f"kept={len(selected_rules)} | "
        f"families={len(family_counts)} | "
        f"covered_training_positive_examples={len(covered_positive_ids)}"
    )
    print(f"Diverse final rules JSON written to {json_path}")
    print(f"Diverse final rules CSV written to {csv_path}")
    return result


def run(
    extended_rule_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_rules(
        extended_rule_results=extended_rule_results,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
