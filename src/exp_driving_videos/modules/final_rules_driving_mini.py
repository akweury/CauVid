"""
Select the final rule set by ranking all kept rules and retaining the top-k.

Consumes:
  - Step 16 output: aggregated kept rules from initial + extended rounds

Output layout:
    pipeline_output/17_driving_mini_final_rules/
        final_rules.json
        final_rules.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_FINAL_RULES_VERSION = 4
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
    out = config.get_output_path("pipeline_output") / "17_driving_mini_final_rules"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "top_k": int(cfg.get("top_k", 50)),
    }


def _parse_atom(atom: str) -> Optional[str]:
    text = str(atom).strip()
    match = re.match(r"^([a-z0-9_]+)\((.*)\)\.$", text)
    if not match:
        return None
    return str(match.group(1))


def _get_rule_body_atom_templates(rule: Dict[str, Any]) -> List[str]:
    body_atom_templates = rule.get("body_atom_templates")
    if isinstance(body_atom_templates, list):
        return [str(atom).strip() for atom in body_atom_templates if str(atom).strip()]
    body_atom_template = str(rule.get("body_atom_template", "")).strip()
    return [body_atom_template] if body_atom_template else []


def _candidate_rule_semantic_profile(rule: Dict[str, Any]) -> Dict[str, Any]:
    predicates = {
        predicate
        for predicate in (_parse_atom(atom) for atom in _get_rule_body_atom_templates(rule))
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


def _summarize_candidate_rule_subset(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    matched_prior_id_counts: Dict[str, int] = {}
    broad_candidate_pattern_counts: Dict[str, int] = {}
    total_candidate_body_atoms = 0
    candidate_body_atom_ratios: List[float] = []
    candidate_rule_count = 0
    weak_candidate_rule_count = 0
    broad_candidate_rule_count = 0
    for rule in rules:
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
        "num_rules": len(rules),
        "num_rules_using_candidate_atoms": candidate_rule_count,
        "num_weak_candidate_rules": weak_candidate_rule_count,
        "num_broad_candidate_rules": broad_candidate_rule_count,
        "total_candidate_body_atoms": total_candidate_body_atoms,
        "avg_candidate_body_atom_ratio": float(sum(candidate_body_atom_ratios) / max(1, len(candidate_body_atom_ratios))),
        "avg_num_candidate_body_atoms": float(total_candidate_body_atoms / max(1, len(rules))),
        "broad_candidate_rule_pattern_counts": {
            key: broad_candidate_pattern_counts[key]
            for key in sorted(broad_candidate_pattern_counts)
        },
        "matched_prior_id_counts": {
            key: matched_prior_id_counts[key]
            for key in sorted(matched_prior_id_counts)
        },
    }


def _summarize_candidate_rule_selection(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    subset_rules = {
        "all_rules": list(rules),
        "accepted_only_rules": [rule for rule in rules if _rule_candidate_category(rule) == "accepted_only"],
        "candidate_only_rules": [rule for rule in rules if _rule_candidate_category(rule) == "candidate_only"],
        "mixed_accepted_candidate_rules": [
            rule for rule in rules if _rule_candidate_category(rule) == "mixed_accepted_candidate"
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


def _empty_category_counts() -> Dict[str, int]:
    return {
        "accepted_only_rules": 0,
        "candidate_only_rules": 0,
        "mixed_accepted_candidate_rules": 0,
        "candidate_involving_rules": 0,
        "all_rules": 0,
    }


def _count_rule_categories(rules: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = _empty_category_counts()
    for rule in rules:
        category = str(rule.get("candidate_rule_category", "")) or _rule_candidate_category(rule)
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


def process_rules(
    extended_rule_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    top_k = int(cfg.get("top_k", 50))

    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "final_rules.json"
    csv_path = out_root / "final_rules.csv"

    if not force_recompute and json_path.exists():
        with json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _FINAL_RULES_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset({"top_k": top_k}):
            print(f"  [cache] loading {json_path.name}")
            return cached

    all_kept_rules = [_enrich_rule_candidate_semantics(rule) for rule in list(extended_rule_results.get("all_kept_rules", []))]
    ranked_rules = _sort_rules(all_kept_rules)
    final_rules = ranked_rules[: max(0, top_k)]
    for rule in final_rules:
        rule["candidate_rule_category"] = _rule_candidate_category(rule)
    candidate_rule_diagnostics = _summarize_candidate_rule_selection(final_rules)
    selected_rule_counts = _count_rule_categories(final_rules)
    input_rule_counts = _count_rule_categories(all_kept_rules)
    candidate_rule_flow_summary = dict(extended_rule_results.get("candidate_rule_flow_summary", {}))
    if not candidate_rule_flow_summary:
        candidate_rule_flow_summary = {
            "atom_availability": {},
            "initial_rule_generation": _empty_category_counts(),
            "merged_after_step15": _empty_category_counts(),
            "pruning": _empty_category_counts(),
            "extension": {
                "all_kept_after_step16_rule_counts": input_rule_counts,
            },
            "final_selection": _empty_category_counts(),
            "evaluation": _empty_category_counts(),
        }
    candidate_rule_flow_summary["final_selection"] = selected_rule_counts

    result: Dict[str, Any] = {
        "version": _FINAL_RULES_VERSION,
        "config": {
            "top_k": top_k,
        },
        "selection_method": "score_top_k",
        "num_input_rules": len(all_kept_rules),
        "num_final_rules": len(final_rules),
        "candidate_rule_stage_stats": {
            "stage": "step17_final_selection",
            "input_kept_after_step16_rule_counts": input_rule_counts,
            "selected_rule_counts": selected_rule_counts,
        },
        "candidate_rule_flow_summary": candidate_rule_flow_summary,
        "candidate_rule_diagnostics": candidate_rule_diagnostics,
        "final_rules": final_rules,
    }

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rule_id",
                "kept_stage",
                "kept_round_index",
                "clause",
                "confidence",
                "positive_support",
                "negative_support",
                "total_support",
                "positive_firings",
                "negative_firings",
                "total_firings",
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
        for rule in final_rules:
            writer.writerow(
                {
                    "rule_id": rule.get("rule_id", ""),
                    "kept_stage": rule.get("kept_stage", ""),
                    "kept_round_index": rule.get("kept_round_index", ""),
                    "clause": rule.get("clause", ""),
                    "confidence": rule.get("confidence", 0.0),
                    "positive_support": rule.get("positive_support", 0),
                    "negative_support": rule.get("negative_support", 0),
                    "total_support": rule.get("total_support", 0),
                    "positive_firings": rule.get("positive_firings", 0),
                    "negative_firings": rule.get("negative_firings", 0),
                    "total_firings": rule.get("total_firings", 0),
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
        "  final_rules: "
        f"input={len(all_kept_rules)} | "
        f"top_k={top_k} | "
        f"kept={len(final_rules)}"
    )
    print(f"Final rules JSON written to {json_path}")
    print(f"Final rules CSV written to {csv_path}")
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
