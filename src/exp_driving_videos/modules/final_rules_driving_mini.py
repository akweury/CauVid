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
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_FINAL_RULES_VERSION = 7
_RULE_CATEGORY_ORDER = (
    "accepted_only",
    "mixed_accepted_candidate",
    "candidate_only",
    "candidate_candidate",
)
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

_RANKED_RULE_FIELDS = [
    "rule_id",
    "clause",
    "rank",
    "original_score",
    "confidence",
    "support",
    "positive_support",
    "negative_support",
    "rule_source",
    "kept_stage",
    "kept_round_index",
    "candidate_rule_category",
    "selection_rule_category",
    "uses_candidate_atoms",
    "num_candidate_body_atoms",
    "candidate_body_atom_ratio",
    "mixes_accepted_and_candidate_atoms",
    "uses_only_candidate_atoms",
    "body_source_mix",
    "initial_pruning_category",
    "pruned_initial_rule_index",
    "selection_rank",
    "selection_method",
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
    "selection_weak_candidate_penalty_value",
    "selection_semantic_match_level",
    "selection_semantic_is_quota_qualified",
    "selection_semantic_deficit_reduction",
]


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "17_driving_mini_final_rules"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "top_k": int(cfg.get("top_k", 50)),
        "category_budgets": _normalized_category_budgets(cfg),
    }


def _normalized_category_budgets(cfg: Dict[str, Any]) -> Dict[str, int]:
    defaults = {
        "accepted_only": 25,
        "mixed_accepted_candidate": 20,
        "candidate_only": 5,
        "candidate_candidate": 0,
    }
    resolved = dict(defaults)
    override = cfg.get("category_budgets")
    if isinstance(override, dict):
        for key, value in override.items():
            resolved[str(key)] = int(value)
    return resolved


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


def _build_example_provenance_lookup(
    temporal_rule_results: Optional[Sequence[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for video_result in list(temporal_rule_results or []):
        for example in list(video_result.get("examples", [])):
            example_id = str(example.get("example_id", "")).strip()
            if not example_id:
                continue
            lookup[example_id] = {
                str(atom): dict(provenance)
                for atom, provenance in dict(example.get("body_atom_provenance_map", {})).items()
                if str(atom) and isinstance(provenance, dict)
            }
    return lookup


def _enrich_rule_evidence_with_provenance(
    rule: Dict[str, Any],
    example_provenance_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if not example_provenance_lookup:
        return dict(rule)
    updated_rule = dict(rule)
    evidence_out: List[Dict[str, Any]] = []
    for entry in list(rule.get("evidence_set", [])):
        updated_entry = dict(entry)
        if dict(updated_entry.get("matched_atom_provenance", {})):
            evidence_out.append(updated_entry)
            continue
        example_id = str(updated_entry.get("example_id", "")).strip()
        atom_lookup = example_provenance_lookup.get(example_id, {})
        matched_atom_provenance = {}
        for atom_template, concrete_atom in dict(updated_entry.get("matched_atoms", {})).items():
            provenance = atom_lookup.get(str(concrete_atom), {})
            if isinstance(provenance, dict) and provenance:
                matched_atom_provenance[str(atom_template)] = dict(provenance)
        if matched_atom_provenance:
            updated_entry["matched_atom_provenance"] = matched_atom_provenance
        evidence_out.append(updated_entry)
    updated_rule["evidence_set"] = evidence_out
    return updated_rule


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
    explicit = str(rule.get("candidate_rule_category", "")).strip()
    if explicit == "candidate_candidate":
        return "candidate_candidate"
    if explicit == "mixed_accepted_candidate":
        return "mixed_accepted_candidate"
    extension_category = str(rule.get("extension_rule_category", "")).strip()
    if extension_category == "candidate_candidate":
        return "candidate_candidate"
    if extension_category == "accepted_candidate":
        return "mixed_accepted_candidate"
    if bool(rule.get("mixes_accepted_and_candidate_atoms", False)):
        return "mixed_accepted_candidate"
    if bool(rule.get("uses_only_candidate_atoms", False)) and int(rule.get("num_candidate_body_atoms", 0)) >= 2:
        return "candidate_candidate"
    if bool(rule.get("uses_only_candidate_atoms", False)):
        return "candidate_only"
    if bool(rule.get("uses_candidate_atoms", False)):
        return "candidate_only"
    return "accepted_only"


def _post_pruned_rule_pool(extended_rule_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    all_kept_rules = list(extended_rule_results.get("all_kept_rules", []))
    if all_kept_rules:
        return [dict(rule) for rule in all_kept_rules]

    rules: List[Dict[str, Any]] = []
    for round_payload in list(extended_rule_results.get("rounds", [])):
        for rule in list(round_payload.get("rules", [])):
            kept_rule = dict(rule)
            kept_rule.setdefault("kept_stage", "extended")
            kept_rule.setdefault("kept_round_index", int(round_payload.get("round_index", 0)))
            rules.append(kept_rule)
    return rules


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
        "candidate_candidate_rules": [rule for rule in rules if _rule_candidate_category(rule) == "candidate_candidate"],
        "mixed_accepted_candidate_rules": [
            rule for rule in rules if _rule_candidate_category(rule) == "mixed_accepted_candidate"
        ],
    }
    return {
        "category_counts": {
            "accepted_only_rules": len(subset_rules["accepted_only_rules"]),
            "candidate_only_rules": len(subset_rules["candidate_only_rules"]),
            "candidate_candidate_rules": len(subset_rules["candidate_candidate_rules"]),
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
        "candidate_candidate_rules": 0,
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
        elif category == "candidate_candidate":
            counts["candidate_candidate_rules"] += 1
            counts["candidate_only_rules"] += 1
            counts["candidate_involving_rules"] += 1
        elif category == "candidate_only":
            counts["candidate_only_rules"] += 1
            counts["candidate_involving_rules"] += 1
        else:
            counts["accepted_only_rules"] += 1
    return counts


def _category_count_key(category: str) -> str:
    if category == "accepted_only":
        return "accepted_only_rules"
    if category == "mixed_accepted_candidate":
        return "mixed_accepted_candidate_rules"
    if category == "candidate_candidate":
        return "candidate_candidate_rules"
    return "candidate_only_rules"


def _budgeted_final_rule_selection(
    ranked_rules: Sequence[Dict[str, Any]],
    *,
    top_k: int,
    category_budgets: Dict[str, int],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    selected: List[Dict[str, Any]] = []
    selected_counts = _empty_category_counts()
    budget_pruned_counts = _empty_category_counts()
    for rule in ranked_rules:
        if len(selected) >= max(0, top_k):
            break
        category = _rule_candidate_category(rule)
        budget = int(category_budgets.get(category, -1))
        if budget >= 0 and selected_counts.get(_category_count_key(category), 0) >= budget:
            budget_pruned_counts[_category_count_key(category)] += 1
            budget_pruned_counts["all_rules"] += 1
            if category != "accepted_only":
                budget_pruned_counts["candidate_involving_rules"] += 1
            continue
        selected.append(dict(rule))
        rule_count_key = _category_count_key(category)
        selected_counts["all_rules"] += 1
        selected_counts[rule_count_key] += 1
        if category in {"mixed_accepted_candidate", "candidate_only", "candidate_candidate"}:
            selected_counts["candidate_involving_rules"] += 1
    return selected, budget_pruned_counts


def _rule_original_score(rule: Dict[str, Any]) -> float:
    if "selection_utility" in rule:
        return float(rule.get("selection_utility", 0.0))
    return float(rule.get("confidence", 0.0))


def _ranked_rule_payload(rule: Dict[str, Any], rank: int) -> Dict[str, Any]:
    payload = dict(rule)
    category = _rule_candidate_category(payload)
    payload["rank"] = int(rank)
    payload["original_rank"] = int(rank)
    payload["original_score"] = _rule_original_score(payload)
    payload["support"] = int(payload.get("total_support", payload.get("support", 0)) or 0)
    payload["candidate_rule_category"] = category
    payload["selection_rule_category"] = str(payload.get("selection_rule_category", category))
    payload["rule_source"] = str(payload.get("kept_stage", payload.get("source", "")))
    return payload


def _write_ranked_rules_csv(path: Path, ranked_rules: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_RANKED_RULE_FIELDS)
        writer.writeheader()
        for rule in ranked_rules:
            writer.writerow({key: rule.get(key, "") for key in _RANKED_RULE_FIELDS})


def process_rules(
    extended_rule_results: Dict[str, Any],
    temporal_rule_results: Optional[Sequence[Dict[str, Any]]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    top_k = int(cfg.get("top_k", 50))
    category_budgets = _normalized_category_budgets(cfg)

    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "final_rules.json"
    csv_path = out_root / "final_rules.csv"
    ranked_json_path = out_root / "ranked_rules.json"
    ranked_csv_path = out_root / "ranked_rules.csv"

    if not force_recompute and json_path.exists():
        with json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _FINAL_RULES_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset({"top_k": top_k, "category_budgets": category_budgets}):
            print(f"  [cache] loading {json_path.name}")
            return cached

    post_pruned_rule_pool = _post_pruned_rule_pool(extended_rule_results)
    example_provenance_lookup = _build_example_provenance_lookup(temporal_rule_results)
    all_kept_rules = [
        _enrich_rule_candidate_semantics(
            _enrich_rule_evidence_with_provenance(rule, example_provenance_lookup)
        )
        for rule in post_pruned_rule_pool
    ]
    ranked_rules = _sort_rules(all_kept_rules)
    final_rules, budget_pruned_counts = _budgeted_final_rule_selection(
        ranked_rules,
        top_k=top_k,
        category_budgets=category_budgets,
    )
    ranked_rule_payloads = [
        _ranked_rule_payload(rule, rank)
        for rank, rule in enumerate(ranked_rules, start=1)
    ]
    for rule in final_rules:
        rule["candidate_rule_category"] = _rule_candidate_category(rule)
        rule["selection_rule_category"] = rule["candidate_rule_category"]
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
            "category_budgets": category_budgets,
        },
        "selection_method": "score_top_k_category_budgeted",
        "selection_input_stage": "step16_post_pruned_kept_pool",
        "num_input_rules": len(all_kept_rules),
        "num_final_rules": len(final_rules),
        "candidate_rule_stage_stats": {
            "stage": "step17_final_selection",
            "input_kept_after_step16_rule_counts": input_rule_counts,
            "budget_pruned_rule_counts": budget_pruned_counts,
            "selected_rule_counts": selected_rule_counts,
        },
        "candidate_rule_flow_summary": candidate_rule_flow_summary,
        "candidate_rule_diagnostics": candidate_rule_diagnostics,
        "final_rules": final_rules,
        "ranked_rules": ranked_rule_payloads,
        "output_paths": {
            "final_rules_json": str(json_path),
            "final_rules_csv": str(csv_path),
            "ranked_rules_json": str(ranked_json_path),
            "ranked_rules_csv": str(ranked_csv_path),
        },
    }

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    with ranked_json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "version": _FINAL_RULES_VERSION,
                "config": result["config"],
                "selection_method": result["selection_method"],
                "selection_input_stage": result["selection_input_stage"],
                "num_ranked_rules": len(ranked_rule_payloads),
                "ranked_rules": ranked_rule_payloads,
            },
            fh,
            indent=2,
        )
    _write_ranked_rules_csv(ranked_csv_path, ranked_rule_payloads)

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
                "selection_rule_category",
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
                    "selection_rule_category": rule.get("selection_rule_category", "accepted_only"),
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
        f"kept={len(final_rules)} | "
        f"category_budgets={category_budgets}"
    )
    print(f"Final rules JSON written to {json_path}")
    print(f"Final rules CSV written to {csv_path}")
    print(f"Ranked rules JSON written to {ranked_json_path}")
    print(f"Ranked rules CSV written to {ranked_csv_path}")
    return result


def run(
    extended_rule_results: Dict[str, Any],
    temporal_rule_results: Optional[Sequence[Dict[str, Any]]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_rules(
        extended_rule_results=extended_rule_results,
        temporal_rule_results=temporal_rule_results,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
