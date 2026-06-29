"""
Extend merged initial temporal rules for a fixed number of rounds.

Current scope:
  - Each round extends every current rule with one body atom drawn from the
    unary subset of the merged initial rule set.
  - The added atom must be new to the rule body.
  - Evaluation uses binding-aware evidence-set intersections.
  - Pruning is dispatched through a strategy interface.

Output layout:
    pipeline_output/16_driving_mini_extended_rules/
        extended_rules_manifest.json
        extended_rules_round_<n>.json
        extended_rules_round_<n>.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config
from src.exp_driving_videos.modules.extended_rules_pruning import (
    apply_pruning_strategies,
    normalize_strategy_names,
)


_EXTENDED_RULES_VERSION = 5
_PROVENANCE_ONLY_BODY_PREDICATES = {
    "object_is_candidate",
    "object_source_type",
    "object_candidate_score_state",
    "object_prior_relevance_state",
    "object_matched_prior",
}
_SEMANTIC_OBJECT_BODY_PREDICATES = {
    "object_class",
    "object_distance_state",
    "object_x_position_state",
    "object_vz_state",
    "object_vx_state",
    "object_speed_state",
    "object_visibility_state",
    "traffic_control_type",
    "traffic_control_relevance_state",
    "traffic_control_front_center_region",
    "traffic_control_relevant",
    "traffic_light_relevant",
    "stop_sign_relevant",
    "traffic_light_position_state",
    "traffic_light_state",
}


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "16_driving_mini_extended_rules"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_prune_strategies(cfg: Dict[str, Any]) -> List[str]:
    raw_list = cfg.get("prune_strategies")
    if isinstance(raw_list, list):
        strategies = normalize_strategy_names([str(item) for item in raw_list])
        if strategies:
            return strategies
    return normalize_strategy_names([str(cfg.get("prune_strategy", "empty_evidence"))])


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "input_initial_rule_pool_key": str(cfg.get("input_initial_rule_pool_key", "")),
        "num_rounds": int(cfg.get("num_rounds", 3)),
        "evaluation_strategy": str(cfg.get("evaluation_strategy", "binding_aware_intersection")),
        "prune_strategies": _resolve_prune_strategies(cfg),
        "min_positive_support_to_extend": int(cfg.get("min_positive_support_to_extend", 1)),
        "same_confidence_smaller_evidence_enabled": bool(
            cfg.get("same_confidence_smaller_evidence_enabled", True)
        ),
        "per_parent_extension_top_k": int(cfg.get("per_parent_extension_top_k", 40)),
        "max_round_rules": int(cfg.get("max_round_rules", 50000)),
        "max_round_accepted_only_rules": int(cfg.get("max_round_accepted_only_rules", 30000)),
        "max_round_mixed_candidate_rules": int(cfg.get("max_round_mixed_candidate_rules", 15000)),
        "max_round_candidate_only_rules": int(cfg.get("max_round_candidate_only_rules", 3000)),
        "max_round_candidate_candidate_rules": int(cfg.get("max_round_candidate_candidate_rules", 500)),
    }


def _initial_rule_pool_cache_key(rules: Sequence[Dict[str, Any]]) -> str:
    payload = [
        {
            "rule_id": str(rule.get("rule_id", "")),
            "clause": str(rule.get("clause", "")),
            "positive_example_ids": list(rule.get("positive_example_ids", [])),
            "negative_example_ids": list(rule.get("negative_example_ids", [])),
            "positive_firings": int(rule.get("positive_firings", 0)),
            "negative_firings": int(rule.get("negative_firings", 0)),
        }
        for rule in rules
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strip_trailing_dot(atom_text: str) -> str:
    return str(atom_text).strip().rstrip(".").strip()


def _normalize_atom_template(atom_text: str) -> str:
    normalized = _strip_trailing_dot(atom_text)
    return f"{normalized}." if normalized else ""


def _predicate_of_atom_template(atom_text: str) -> str:
    text = _strip_trailing_dot(atom_text)
    if "(" not in text:
        return text
    return text.split("(", 1)[0].strip()


def _canonical_body_atoms(body_atoms: Iterable[str]) -> Tuple[str, ...]:
    normalized = {
        _normalize_atom_template(atom_text)
        for atom_text in body_atoms
        if _normalize_atom_template(atom_text)
    }
    return tuple(sorted(normalized))


def _build_clause(head_atom_template: str, body_atom_templates: Sequence[str]) -> str:
    head = _strip_trailing_dot(head_atom_template)
    body = ", ".join(_strip_trailing_dot(atom) for atom in body_atom_templates if _strip_trailing_dot(atom))
    if not head:
        return ""
    if not body:
        return f"{head}."
    return f"{head} :- {body}."


def _get_rule_body_atom_templates(rule: Dict[str, Any]) -> Tuple[str, ...]:
    body_atoms = rule.get("body_atom_templates")
    if isinstance(body_atoms, list):
        return _canonical_body_atoms(body_atoms)

    body_atom_template = str(rule.get("body_atom_template", "")).strip()
    if body_atom_template:
        return _canonical_body_atoms([body_atom_template])
    return ()


def _is_unary_initial_rule(rule: Dict[str, Any]) -> bool:
    body_atoms = _get_rule_body_atom_templates(rule)
    return len(body_atoms) == 1


def _group_evidence_by_example(
    evidence_entries: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in evidence_entries:
        example_id = str(entry.get("example_id", ""))
        grouped.setdefault(example_id, []).append(entry)
    return grouped


def _bindings_compatible(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> bool:
    for key, value in left.items():
        if key in right and str(right[key]) != str(value):
            return False
    return True


def _merge_bindings(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> Dict[str, str]:
    merged = {str(key): str(value) for key, value in left.items()}
    for key, value in right.items():
        merged[str(key)] = str(value)
    return merged


def _merge_matched_atoms(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> Dict[str, str]:
    merged = {str(key): str(value) for key, value in left.items()}
    for key, value in right.items():
        merged[str(key)] = str(value)
    return merged


def _dedupe_evidence_entries(
    evidence_entries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[str, Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...]]] = set()
    for entry in evidence_entries:
        bindings = tuple(sorted((str(k), str(v)) for k, v in dict(entry.get("bindings", {})).items()))
        matched_atoms = tuple(
            sorted((str(k), str(v)) for k, v in dict(entry.get("matched_atoms", {})).items())
        )
        key = (str(entry.get("example_id", "")), bindings, matched_atoms)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _intersect_evidence_sets(
    parent_evidence_set: Sequence[Dict[str, Any]],
    added_atom_evidence_set: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    parent_by_example = _group_evidence_by_example(parent_evidence_set)
    added_by_example = _group_evidence_by_example(added_atom_evidence_set)
    shared_example_ids = sorted(set(parent_by_example) & set(added_by_example))

    intersected: List[Dict[str, Any]] = []
    for example_id in shared_example_ids:
        parent_entries = parent_by_example.get(example_id, [])
        added_entries = added_by_example.get(example_id, [])
        for parent_entry in parent_entries:
            parent_bindings = dict(parent_entry.get("bindings", {}))
            parent_matched_atoms = dict(parent_entry.get("matched_atoms", {}))
            parent_matched_atom_sources = dict(parent_entry.get("matched_atom_sources", {}))
            parent_matched_atom_prior_ids = dict(parent_entry.get("matched_atom_prior_ids", {}))
            for added_entry in added_entries:
                added_bindings = dict(added_entry.get("bindings", {}))
                if not _bindings_compatible(parent_bindings, added_bindings):
                    continue

                matched_atoms = _merge_matched_atoms(
                    parent_matched_atoms,
                    dict(added_entry.get("matched_atoms", {})),
                )
                matched_atom_sources = _merge_matched_atom_sources(
                    parent_matched_atom_sources,
                    dict(added_entry.get("matched_atom_sources", {})),
                )
                matched_atom_prior_ids = _merge_matched_atom_prior_ids(
                    parent_matched_atom_prior_ids,
                    dict(added_entry.get("matched_atom_prior_ids", {})),
                )
                intersected.append(
                    {
                        "video_id": str(parent_entry.get("video_id", added_entry.get("video_id", ""))),
                        "example_id": example_id,
                        "current_segment_id": str(
                            parent_entry.get("current_segment_id", added_entry.get("current_segment_id", ""))
                        ),
                        "next_segment_id": str(
                            parent_entry.get("next_segment_id", added_entry.get("next_segment_id", ""))
                        ),
                        "target_predicate": str(
                            parent_entry.get("target_predicate", added_entry.get("target_predicate", ""))
                        ),
                        "label": bool(parent_entry.get("label", added_entry.get("label", False))),
                        "bindings": _merge_bindings(parent_bindings, added_bindings),
                        "matched_atoms": matched_atoms,
                        "matched_atom_sources": matched_atom_sources,
                        "matched_atom_prior_ids": matched_atom_prior_ids,
                    }
                )

    return _dedupe_evidence_entries(intersected)


def _merge_matched_atom_sources(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> Dict[str, str]:
    merged = {str(key): str(value) for key, value in left.items()}
    for key, value in right.items():
        merged[str(key)] = str(value)
    return merged


def _merge_matched_atom_prior_ids(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {
        str(key): [str(item) for item in list(value)]
        for key, value in left.items()
    }
    for key, value in right.items():
        merged_key = str(key)
        existing = merged.setdefault(merged_key, [])
        for item in list(value):
            item_text = str(item)
            if item_text and item_text not in existing:
                existing.append(item_text)
    return merged


def _summarize_evidence(
    evidence_entries: Sequence[Dict[str, Any]],
    total_positive_examples: int,
) -> Dict[str, Any]:
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
    confidence = float(positive_firings / max(1, total_firings))
    recall = float(len(positive_example_ids) / max(1, total_positive_examples))
    return {
        "positive_support": len(positive_example_ids),
        "negative_support": len(negative_example_ids),
        "total_support": len(set(positive_example_ids) | set(negative_example_ids)),
        "positive_firings": positive_firings,
        "negative_firings": negative_firings,
        "total_firings": total_firings,
        "confidence": confidence,
        "recall": recall,
        "positive_example_ids": positive_example_ids,
        "negative_example_ids": negative_example_ids,
    }


def _is_body_semantically_grounded(body_atom_templates: Sequence[str]) -> bool:
    provenance_only_count = 0
    semantic_count = 0
    for body_atom_template in body_atom_templates:
        predicate = _predicate_of_atom_template(body_atom_template)
        if predicate in _PROVENANCE_ONLY_BODY_PREDICATES:
            provenance_only_count += 1
        if predicate in _SEMANTIC_OBJECT_BODY_PREDICATES:
            semantic_count += 1
    if provenance_only_count == 0:
        return True
    return semantic_count >= 1


def _rule_candidate_category(rule: Dict[str, Any]) -> str:
    if bool(rule.get("mixes_accepted_and_candidate_atoms", False)):
        return "mixed_accepted_candidate"
    if bool(rule.get("uses_only_candidate_atoms", False)):
        return "candidate_only"
    if bool(rule.get("uses_candidate_atoms", False)):
        return "candidate_only"
    body_atom_templates = list(rule.get("body_atom_templates", []))
    has_candidate_template = any("(C" in str(atom) or ",C" in str(atom) for atom in body_atom_templates)
    has_accepted_template = any("(O" in str(atom) or ",O" in str(atom) for atom in body_atom_templates)
    if has_candidate_template and has_accepted_template:
        return "mixed_accepted_candidate"
    if has_candidate_template:
        return "candidate_only"
    return "accepted_only"


def _empty_category_counts() -> Dict[str, int]:
    return {
        "accepted_only_rules": 0,
        "candidate_only_rules": 0,
        "candidate_candidate_rules": 0,
        "mixed_accepted_candidate_rules": 0,
        "candidate_involving_rules": 0,
        "all_rules": 0,
    }


def _count_rule_categories(rules: Sequence[Dict[str, Any]]) -> Dict[str, int]:
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
            if str(rule.get("extension_rule_category", "")) == "candidate_candidate":
                counts["candidate_candidate_rules"] += 1
        else:
            counts["accepted_only_rules"] += 1
    return counts


def _extension_rule_category(rule: Dict[str, Any]) -> str:
    if bool(rule.get("mixes_accepted_and_candidate_atoms", False)):
        return "accepted_candidate"
    if bool(rule.get("uses_only_candidate_atoms", False)):
        return "candidate_candidate" if int(rule.get("num_candidate_body_atoms", 0)) >= 2 else "candidate_only"
    if bool(rule.get("uses_candidate_atoms", False)):
        return "candidate_only"
    return "accepted_only"


def _candidate_category_count_key(category: str) -> str:
    if category == "accepted_candidate":
        return "mixed_accepted_candidate_rules"
    if category == "candidate_only":
        return "candidate_only_rules"
    if category == "candidate_candidate":
        return "candidate_candidate_rules"
    return "accepted_only_rules"


def _increment_category_count(counts: Dict[str, int], category: str) -> None:
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


def _firing_signature(evidence_entries: Sequence[Dict[str, Any]]) -> Tuple[Tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                str(entry.get("example_id", "")),
                "pos" if bool(entry.get("label", False)) else "neg",
            )
            for entry in evidence_entries
        )
    )


def _semantic_predicate_strength(body_atom_templates: Sequence[str]) -> int:
    return sum(
        1
        for body_atom_template in body_atom_templates
        if _predicate_of_atom_template(body_atom_template) in _SEMANTIC_OBJECT_BODY_PREDICATES
    )


def _extension_rank_key(rule: Dict[str, Any]) -> Tuple[float, int, int, int, int, float, str]:
    confidence = float(rule.get("confidence", 0.0))
    positive_support = int(rule.get("positive_support", 0))
    negative_support = int(rule.get("negative_support", 0))
    support_improvement = int(rule.get("support_improvement", 0))
    body_atoms = list(rule.get("body_atom_templates", []))
    semantic_strength = _semantic_predicate_strength(body_atoms)
    candidate_ratio = float(rule.get("candidate_body_atom_ratio", 0.0))
    provenance_penalty = 1.0 if _extension_rule_category(rule) == "candidate_candidate" else 0.0
    score = (
        confidence * 100.0
        + positive_support * 10.0
        + max(0, support_improvement) * 6.0
        - negative_support * 4.0
        - len(body_atoms) * 0.5
        + semantic_strength * 2.0
        - candidate_ratio * 3.0
        - provenance_penalty * 4.0
    )
    return (
        -score,
        -positive_support,
        negative_support,
        len(body_atoms),
        -semantic_strength,
        candidate_ratio,
        str(rule.get("clause", "")),
    )


def _budget_allows_rule(
    category: str,
    kept_counts: Dict[str, int],
    max_round_rules: int,
    category_budgets: Dict[str, int],
) -> bool:
    if max_round_rules and kept_counts.get("all_rules", 0) >= max_round_rules:
        return False
    budget = int(category_budgets.get(category, -1))
    if budget >= 0:
        return kept_counts.get(_candidate_category_count_key(category), 0) < budget
    return True


def _summarize_rule_candidate_provenance(
    body_atom_templates: Sequence[str],
    evidence_entries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_body_templates: set[str] = set()
    accepted_body_templates: set[str] = set()
    matched_prior_ids_involved: set[str] = set()
    for entry in evidence_entries:
        matched_atom_sources = dict(entry.get("matched_atom_sources", {}))
        matched_atom_prior_ids = dict(entry.get("matched_atom_prior_ids", {}))
        for body_atom_template in body_atom_templates:
            source = str(matched_atom_sources.get(body_atom_template, ""))
            if source == "candidate":
                candidate_body_templates.add(str(body_atom_template))
                for prior_id in list(matched_atom_prior_ids.get(body_atom_template, [])):
                    prior_id_text = str(prior_id)
                    if prior_id_text:
                        matched_prior_ids_involved.add(prior_id_text)
            elif source == "accepted":
                accepted_body_templates.add(str(body_atom_template))

    uses_candidate_atoms = bool(candidate_body_templates)
    uses_only_candidate_atoms = uses_candidate_atoms and not accepted_body_templates
    mixes_accepted_and_candidate_atoms = bool(candidate_body_templates) and bool(accepted_body_templates)
    num_candidate_body_atoms = len(candidate_body_templates)
    body_length = max(1, len(list(body_atom_templates)))
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
        "candidate_body_atom_ratio": float(num_candidate_body_atoms / float(body_length)),
        "matched_prior_ids_involved": sorted(matched_prior_ids_involved),
        "mixes_accepted_and_candidate_atoms": mixes_accepted_and_candidate_atoms,
        "uses_only_candidate_atoms": uses_only_candidate_atoms,
        "body_source_mix": body_source_mix,
    }


def _seed_rule_evidence(rule: Dict[str, Any]) -> List[Dict[str, Any]]:
    seeded: List[Dict[str, Any]] = []
    for entry in list(rule.get("evidence_set", [])):
        if isinstance(entry.get("matched_atoms"), dict):
            seeded.append(
                {
                    "video_id": str(entry.get("video_id", "")),
                    "example_id": str(entry.get("example_id", "")),
                    "current_segment_id": str(entry.get("current_segment_id", "")),
                    "next_segment_id": str(entry.get("next_segment_id", "")),
                    "target_predicate": str(entry.get("target_predicate", "")),
                    "label": bool(entry.get("label", False)),
                    "bindings": {str(k): str(v) for k, v in dict(entry.get("bindings", {})).items()},
                    "matched_atoms": {
                        str(k): str(v)
                        for k, v in dict(entry.get("matched_atoms", {})).items()
                        if str(k) and str(v)
                    },
                    "matched_atom_sources": {
                        str(k): str(v)
                        for k, v in dict(entry.get("matched_atom_sources", {})).items()
                        if str(k) and str(v)
                    },
                    "matched_atom_prior_ids": {
                        str(k): [str(item) for item in list(v) if str(item)]
                        for k, v in dict(entry.get("matched_atom_prior_ids", {})).items()
                        if str(k)
                    },
                }
            )
            continue
        body_atom_template = str(entry.get("body_atom_template", rule.get("body_atom_template", ""))).strip()
        matched_atom = str(entry.get("matched_atom", "")).strip()
        body_atom_source = str(entry.get("body_atom_source", "")).strip()
        matched_prior_ids = [str(value) for value in list(entry.get("matched_prior_ids", [])) if str(value)]
        seeded.append(
            {
                "video_id": str(entry.get("video_id", "")),
                "example_id": str(entry.get("example_id", "")),
                "current_segment_id": str(entry.get("current_segment_id", "")),
                "next_segment_id": str(entry.get("next_segment_id", "")),
                "target_predicate": str(entry.get("target_predicate", "")),
                "label": bool(entry.get("label", False)),
                "bindings": {str(k): str(v) for k, v in dict(entry.get("bindings", {})).items()},
                "matched_atoms": {body_atom_template: matched_atom} if body_atom_template and matched_atom else {},
                "matched_atom_sources": (
                    {body_atom_template: body_atom_source}
                    if body_atom_template and matched_atom and body_atom_source
                    else {}
                ),
                "matched_atom_prior_ids": (
                    {body_atom_template: matched_prior_ids}
                    if body_atom_template and matched_atom and matched_prior_ids
                    else {}
                ),
            }
        )
    return _dedupe_evidence_entries(seeded)


def _single_atom_evidence_from_rule(
    rule: Dict[str, Any],
    body_atom_template: str,
) -> List[Dict[str, Any]]:
    atom_template = _normalize_atom_template(body_atom_template)
    seeded: List[Dict[str, Any]] = []
    for entry in list(rule.get("evidence_set", [])):
        matched_atoms = dict(entry.get("matched_atoms", {}))
        matched_atom = str(matched_atoms.get(atom_template, "")).strip()
        if matched_atom:
            matched_atom_sources = dict(entry.get("matched_atom_sources", {}))
            matched_atom_prior_ids = dict(entry.get("matched_atom_prior_ids", {}))
            seeded.append(
                {
                    "video_id": str(entry.get("video_id", "")),
                    "example_id": str(entry.get("example_id", "")),
                    "current_segment_id": str(entry.get("current_segment_id", "")),
                    "next_segment_id": str(entry.get("next_segment_id", "")),
                    "target_predicate": str(entry.get("target_predicate", "")),
                    "label": bool(entry.get("label", False)),
                    "bindings": {str(k): str(v) for k, v in dict(entry.get("bindings", {})).items()},
                    "matched_atoms": {atom_template: matched_atom},
                    "matched_atom_sources": {
                        atom_template: str(matched_atom_sources.get(atom_template, ""))
                    },
                    "matched_atom_prior_ids": {
                        atom_template: [
                            str(item)
                            for item in list(matched_atom_prior_ids.get(atom_template, []))
                            if str(item)
                        ]
                    },
                }
            )
            continue

        if _normalize_atom_template(str(entry.get("body_atom_template", ""))) != atom_template:
            continue
        matched_atom = str(entry.get("matched_atom", "")).strip()
        if not matched_atom:
            continue
        body_atom_source = str(entry.get("body_atom_source", "")).strip()
        matched_prior_ids = [str(value) for value in list(entry.get("matched_prior_ids", [])) if str(value)]
        seeded.append(
            {
                "video_id": str(entry.get("video_id", "")),
                "example_id": str(entry.get("example_id", "")),
                "current_segment_id": str(entry.get("current_segment_id", "")),
                "next_segment_id": str(entry.get("next_segment_id", "")),
                "target_predicate": str(entry.get("target_predicate", "")),
                "label": bool(entry.get("label", False)),
                "bindings": {str(k): str(v) for k, v in dict(entry.get("bindings", {})).items()},
                "matched_atoms": {atom_template: matched_atom},
                "matched_atom_sources": {atom_template: body_atom_source} if body_atom_source else {},
                "matched_atom_prior_ids": {atom_template: matched_prior_ids} if matched_prior_ids else {},
            }
        )
    return _dedupe_evidence_entries(seeded)


def _serialize_rule(
    rule_index: int,
    round_index: int,
    head_predicate: str,
    head_atom_template: str,
    body_atom_templates: Sequence[str],
    parent_rule_id: str,
    added_body_atom_template: str,
    source_initial_rule_id: str,
    evidence_set: Sequence[Dict[str, Any]],
    evidence_summary: Dict[str, Any],
    provenance_summary: Dict[str, Any],
    parent_confidence: float,
    parent_positive_support: int,
    prune_reason: str,
    prune_status: str,
    kept_after_prune: bool,
) -> Dict[str, Any]:
    clause = _build_clause(head_atom_template, body_atom_templates)
    rule_id = f"extended_round_{round_index}_rule_{rule_index:06d}"
    return {
        "rule_index": rule_index,
        "rule_id": rule_id,
        "round_index": round_index,
        "head_predicate": head_predicate,
        "head_atom_template": head_atom_template,
        "body_atom_templates": list(body_atom_templates),
        "body_length": len(body_atom_templates),
        "body_atom_template": body_atom_templates[0] if len(body_atom_templates) == 1 else "",
        "clause": clause,
        "parent_rule_id": parent_rule_id,
        "added_body_atom_template": added_body_atom_template,
        "source_initial_rule_id": source_initial_rule_id,
        "positive_support": int(evidence_summary.get("positive_support", 0)),
        "negative_support": int(evidence_summary.get("negative_support", 0)),
        "total_support": int(evidence_summary.get("total_support", 0)),
        "positive_firings": int(evidence_summary.get("positive_firings", 0)),
        "negative_firings": int(evidence_summary.get("negative_firings", 0)),
        "total_firings": int(evidence_summary.get("total_firings", 0)),
        "confidence": float(evidence_summary.get("confidence", 0.0)),
        "recall": float(evidence_summary.get("recall", 0.0)),
        "positive_example_ids": list(evidence_summary.get("positive_example_ids", [])),
        "negative_example_ids": list(evidence_summary.get("negative_example_ids", [])),
        "uses_candidate_atoms": bool(provenance_summary.get("uses_candidate_atoms", False)),
        "num_candidate_body_atoms": int(provenance_summary.get("num_candidate_body_atoms", 0)),
        "candidate_body_atom_ratio": float(provenance_summary.get("candidate_body_atom_ratio", 0.0)),
        "matched_prior_ids_involved": list(provenance_summary.get("matched_prior_ids_involved", [])),
        "mixes_accepted_and_candidate_atoms": bool(
            provenance_summary.get("mixes_accepted_and_candidate_atoms", False)
        ),
        "uses_only_candidate_atoms": bool(provenance_summary.get("uses_only_candidate_atoms", False)),
        "body_source_mix": str(provenance_summary.get("body_source_mix", "")),
        "candidate_rule_category": _rule_candidate_category(provenance_summary),
        "parent_confidence": float(parent_confidence),
        "parent_positive_support": int(parent_positive_support),
        "confidence_improvement": float(evidence_summary.get("confidence", 0.0)) - float(parent_confidence),
        "support_improvement": int(evidence_summary.get("positive_support", 0)) - int(parent_positive_support),
        "evaluation_status": "implemented",
        "prune_reason": prune_reason,
        "prune_status": prune_status,
        "kept_after_prune": kept_after_prune,
        "evidence_set": list(evidence_set),
    }


def _write_round_outputs(
    out_root: Path,
    round_index: int,
    payload: Dict[str, Any],
) -> Tuple[Path, Path]:
    json_path = out_root / f"extended_rules_round_{round_index}.json"
    csv_path = out_root / f"extended_rules_round_{round_index}.csv"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rule_index",
                "rule_id",
                "round_index",
                "head_predicate",
                "head_atom_template",
                "body_atom_templates",
                "body_length",
                "body_atom_template",
                "clause",
                "parent_rule_id",
                "added_body_atom_template",
                "source_initial_rule_id",
                "positive_support",
                "negative_support",
                "total_support",
                "positive_firings",
                "negative_firings",
                "total_firings",
                "confidence",
                "recall",
                "uses_candidate_atoms",
                "num_candidate_body_atoms",
                "candidate_body_atom_ratio",
                "matched_prior_ids_involved",
                "mixes_accepted_and_candidate_atoms",
                "uses_only_candidate_atoms",
                "body_source_mix",
                "candidate_rule_category",
                "parent_confidence",
                "confidence_improvement",
                "evaluation_status",
                "prune_reason",
                "prune_status",
                "kept_after_prune",
            ],
        )
        writer.writeheader()
        for rule in payload.get("rules", []):
            writer.writerow(
                {
                    "rule_index": rule.get("rule_index", ""),
                    "rule_id": rule.get("rule_id", ""),
                    "round_index": rule.get("round_index", ""),
                    "head_predicate": rule.get("head_predicate", ""),
                    "head_atom_template": rule.get("head_atom_template", ""),
                    "body_atom_templates": json.dumps(rule.get("body_atom_templates", [])),
                    "body_length": rule.get("body_length", 0),
                    "body_atom_template": rule.get("body_atom_template", ""),
                    "clause": rule.get("clause", ""),
                    "parent_rule_id": rule.get("parent_rule_id", ""),
                    "added_body_atom_template": rule.get("added_body_atom_template", ""),
                    "source_initial_rule_id": rule.get("source_initial_rule_id", ""),
                    "positive_support": rule.get("positive_support", 0),
                    "negative_support": rule.get("negative_support", 0),
                    "total_support": rule.get("total_support", 0),
                    "positive_firings": rule.get("positive_firings", 0),
                    "negative_firings": rule.get("negative_firings", 0),
                    "total_firings": rule.get("total_firings", 0),
                    "confidence": rule.get("confidence", 0.0),
                    "recall": rule.get("recall", 0.0),
                    "uses_candidate_atoms": rule.get("uses_candidate_atoms", False),
                    "num_candidate_body_atoms": rule.get("num_candidate_body_atoms", 0),
                    "candidate_body_atom_ratio": rule.get("candidate_body_atom_ratio", 0.0),
                    "matched_prior_ids_involved": json.dumps(rule.get("matched_prior_ids_involved", [])),
                    "mixes_accepted_and_candidate_atoms": rule.get("mixes_accepted_and_candidate_atoms", False),
                    "uses_only_candidate_atoms": rule.get("uses_only_candidate_atoms", False),
                    "body_source_mix": rule.get("body_source_mix", ""),
                    "candidate_rule_category": rule.get("candidate_rule_category", ""),
                    "parent_confidence": rule.get("parent_confidence", 0.0),
                    "confidence_improvement": rule.get("confidence_improvement", 0.0),
                    "evaluation_status": rule.get("evaluation_status", ""),
                    "prune_reason": rule.get("prune_reason", ""),
                    "prune_status": rule.get("prune_status", ""),
                    "kept_after_prune": rule.get("kept_after_prune", True),
                }
            )

    return json_path, csv_path


def _build_all_kept_rules(
    initial_rules: Sequence[Dict[str, Any]],
    rounds: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    all_kept_rules: List[Dict[str, Any]] = []

    for rule_index, rule in enumerate(initial_rules):
        initial_rule = dict(rule)
        initial_rule["kept_stage"] = "initial"
        initial_rule["kept_round_index"] = 0
        initial_rule["kept_rule_index"] = rule_index
        all_kept_rules.append(initial_rule)

    for round_payload in rounds:
        round_index = int(round_payload.get("round_index", 0))
        for rule in list(round_payload.get("rules", [])):
            kept_rule = dict(rule)
            kept_rule["kept_stage"] = "extended"
            kept_rule["kept_round_index"] = round_index
            kept_rule["kept_rule_index"] = int(kept_rule.get("rule_index", -1))
            all_kept_rules.append(kept_rule)

    return all_kept_rules


def process_rules(
    merged_initial_rules: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    num_rounds = int(cfg.get("num_rounds", 3))
    evaluation_strategy = str(cfg.get("evaluation_strategy", "binding_aware_intersection"))
    prune_strategies = _resolve_prune_strategies(cfg)
    min_positive_support_to_extend = int(cfg.get("min_positive_support_to_extend", 1))
    same_confidence_smaller_evidence_enabled = bool(
        cfg.get("same_confidence_smaller_evidence_enabled", True)
    )
    per_parent_extension_top_k = int(cfg.get("per_parent_extension_top_k", 40))
    max_round_rules = int(cfg.get("max_round_rules", 50000))
    category_budgets = {
        "accepted_only": int(cfg.get("max_round_accepted_only_rules", 30000)),
        "accepted_candidate": int(cfg.get("max_round_mixed_candidate_rules", 15000)),
        "candidate_only": int(cfg.get("max_round_candidate_only_rules", 3000)),
        "candidate_candidate": int(cfg.get("max_round_candidate_candidate_rules", 500)),
    }
    all_input_initial_rules = list(merged_initial_rules.get("rules", []))
    initial_rules = [rule for rule in all_input_initial_rules if _is_unary_initial_rule(rule)]
    num_skipped_non_unary_initial_rules = len(all_input_initial_rules) - len(initial_rules)
    input_initial_rule_pool_key = _initial_rule_pool_cache_key(initial_rules)

    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "extended_rules_manifest.json"

    if not force_recompute and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _EXTENDED_RULES_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset(
            {
                "input_initial_rule_pool_key": input_initial_rule_pool_key,
                "num_rounds": num_rounds,
                "evaluation_strategy": evaluation_strategy,
                "prune_strategies": prune_strategies,
                "min_positive_support_to_extend": min_positive_support_to_extend,
                "same_confidence_smaller_evidence_enabled": same_confidence_smaller_evidence_enabled,
                "per_parent_extension_top_k": per_parent_extension_top_k,
                "max_round_rules": max_round_rules,
                "max_round_accepted_only_rules": category_budgets["accepted_only"],
                "max_round_mixed_candidate_rules": category_budgets["accepted_candidate"],
                "max_round_candidate_only_rules": category_budgets["candidate_only"],
                "max_round_candidate_candidate_rules": category_budgets["candidate_candidate"],
            }
        ):
            print(f"  [cache] loading {manifest_path.name}")
            return cached

    total_positive_examples = int(merged_initial_rules.get("num_positive_examples", 0))
    initial_rule_atom_map: Dict[str, Dict[str, Any]] = {}
    for rule in initial_rules:
        for body_atom_template in _get_rule_body_atom_templates(rule):
            normalized_body_atom_template = _normalize_atom_template(body_atom_template)
            if not normalized_body_atom_template:
                continue
            entry = initial_rule_atom_map.setdefault(
                normalized_body_atom_template,
                {
                    "body_atom_template": normalized_body_atom_template,
                    "source_initial_rule_ids": [],
                    "evidence_set": [],
                },
            )
            rule_id = str(rule.get("rule_id", ""))
            if rule_id and rule_id not in entry["source_initial_rule_ids"]:
                entry["source_initial_rule_ids"].append(rule_id)
            entry["evidence_set"].extend(_single_atom_evidence_from_rule(rule, normalized_body_atom_template))

    initial_rule_atoms: List[Tuple[str, str, List[Dict[str, Any]]]] = []
    for body_atom_template, atom_entry in sorted(initial_rule_atom_map.items()):
        initial_rule_atoms.append(
            (
                body_atom_template,
                ",".join(sorted(atom_entry["source_initial_rule_ids"])),
                _dedupe_evidence_entries(list(atom_entry.get("evidence_set", []))),
            )
        )

    current_rules = []
    for rule in initial_rules:
        seeded_rule = dict(rule)
        seeded_rule["body_atom_templates"] = list(_get_rule_body_atom_templates(rule))
        seeded_rule["evidence_set"] = _seed_rule_evidence(rule)
        seeded_rule["candidate_rule_category"] = _rule_candidate_category(seeded_rule)
        current_rules.append(seeded_rule)
    unary_initial_rule_counts = _count_rule_categories(initial_rules)

    print(
        "  initial_rules: "
        f"input_total={len(all_input_initial_rules)} | "
        f"unary_used={len(initial_rules)} | "
        f"skipped_non_unary={num_skipped_non_unary_initial_rules} | "
        f"unique_body_atoms={len(initial_rule_atoms)}"
    )
    print(f"  prune_strategies: {prune_strategies}")

    round_summaries: List[Dict[str, Any]] = []
    rounds: List[Dict[str, Any]] = []
    total_rules_using_candidate_atoms = 0
    total_candidate_only_rules = 0
    total_mixed_source_rules = 0
    total_pruned_candidate_rule_counts = _empty_category_counts()

    for round_index in range(1, num_rounds + 1):
        next_rule_map: Dict[Tuple[str, str, Tuple[str, ...]], Dict[str, Any]] = {}
        num_candidates_generated = 0
        num_pruned = 0
        num_pruned_low_evidence = 0
        num_pruned_empty_evidence = 0
        num_pruned_no_positive = 0
        num_pruned_same_firings_as_parent = 0
        num_pruned_same_confidence_smaller_evidence = 0
        num_pruned_provenance_dominance = 0
        num_pruned_parent_top_k = 0
        num_pruned_round_budget = 0
        num_deduplicated_body = 0
        num_deduplicated_firing_signature = 0
        num_parent_rules_skipped = 0
        round_rules_using_candidate_atoms = 0
        round_candidate_only_rules = 0
        round_mixed_source_rules = 0
        round_candidate_pool: List[Dict[str, Any]] = []
        round_generated_category_counts = _empty_category_counts()
        round_pruned_budget_category_counts = _empty_category_counts()
        for parent_rule in current_rules:
            parent_candidate_rules: List[Dict[str, Any]] = []
            head_predicate = str(parent_rule.get("head_predicate", ""))
            head_atom_template = str(parent_rule.get("head_atom_template", f"{head_predicate}(S)."))
            parent_rule_id = str(parent_rule.get("rule_id", ""))
            parent_body_atoms = _get_rule_body_atom_templates(parent_rule)
            parent_evidence_set = list(parent_rule.get("evidence_set", []))
            parent_confidence = float(parent_rule.get("confidence", 0.0))
            parent_positive_support = int(parent_rule.get("positive_support", 0))
            parent_positive_firings = int(parent_rule.get("positive_firings", 0))
            parent_negative_firings = int(parent_rule.get("negative_firings", 0))
            parent_total_firings = int(parent_rule.get("total_firings", 0))

            for initial_body_atom_template, source_initial_rule_id, initial_evidence_set in initial_rule_atoms:
                if initial_body_atom_template in parent_body_atoms:
                    continue

                new_body_atoms = _canonical_body_atoms(parent_body_atoms + (initial_body_atom_template,))
                if not _is_body_semantically_grounded(new_body_atoms):
                    num_pruned += 1
                    num_pruned_provenance_dominance += 1
                    provenance_summary = _summarize_rule_candidate_provenance(
                        body_atom_templates=new_body_atoms,
                        evidence_entries=[],
                    )
                    category = _rule_candidate_category(
                        {**provenance_summary, "body_atom_templates": list(new_body_atoms)}
                    )
                    total_pruned_candidate_rule_counts["all_rules"] += 1
                    if category == "mixed_accepted_candidate":
                        total_pruned_candidate_rule_counts["mixed_accepted_candidate_rules"] += 1
                        total_pruned_candidate_rule_counts["candidate_involving_rules"] += 1
                    elif category == "candidate_only":
                        total_pruned_candidate_rule_counts["candidate_only_rules"] += 1
                        total_pruned_candidate_rule_counts["candidate_involving_rules"] += 1
                    else:
                        total_pruned_candidate_rule_counts["accepted_only_rules"] += 1
                    continue
                key = (head_predicate, head_atom_template, new_body_atoms)
                if key in next_rule_map:
                    continue

                num_candidates_generated += 1
                intersected_evidence = _intersect_evidence_sets(parent_evidence_set, initial_evidence_set)

                evidence_summary = _summarize_evidence(intersected_evidence, total_positive_examples)
                provenance_summary = _summarize_rule_candidate_provenance(
                    body_atom_templates=new_body_atoms,
                    evidence_entries=intersected_evidence,
                )
                evidence_summary["parent_confidence"] = parent_confidence
                evidence_summary["parent_positive_firings"] = parent_positive_firings
                evidence_summary["parent_negative_firings"] = parent_negative_firings
                evidence_summary["parent_total_firings"] = parent_total_firings
                evidence_summary["is_last_round"] = bool(round_index == num_rounds)
                evidence_summary["min_positive_support_to_extend"] = min_positive_support_to_extend
                evidence_summary["same_confidence_smaller_evidence_enabled"] = (
                    same_confidence_smaller_evidence_enabled
                )
                prune_decision = apply_pruning_strategies(
                    rule={
                        "head_predicate": head_predicate,
                        "head_atom_template": head_atom_template,
                        "body_atom_templates": list(new_body_atoms),
                        "parent_rule_id": parent_rule_id,
                        "added_body_atom_template": initial_body_atom_template,
                    },
                    metrics=evidence_summary,
                    strategies=prune_strategies,
                )
                if not bool(prune_decision.get("kept", False)):
                    num_pruned += 1
                    pruned_category = _rule_candidate_category(
                        {**provenance_summary, "body_atom_templates": list(new_body_atoms)}
                    )
                    total_pruned_candidate_rule_counts["all_rules"] += 1
                    if pruned_category == "mixed_accepted_candidate":
                        total_pruned_candidate_rule_counts["mixed_accepted_candidate_rules"] += 1
                        total_pruned_candidate_rule_counts["candidate_involving_rules"] += 1
                    elif pruned_category == "candidate_only":
                        total_pruned_candidate_rule_counts["candidate_only_rules"] += 1
                        total_pruned_candidate_rule_counts["candidate_involving_rules"] += 1
                    else:
                        total_pruned_candidate_rule_counts["accepted_only_rules"] += 1
                    prune_reason = str(prune_decision.get("prune_reason", ""))
                    if prune_reason == "low_evidence":
                        num_pruned_low_evidence += 1
                    elif prune_reason == "empty_evidence":
                        num_pruned_empty_evidence += 1
                    elif prune_reason == "no_positive_firings":
                        num_pruned_no_positive += 1
                    elif prune_reason == "same_firings_as_parent":
                        num_pruned_same_firings_as_parent += 1
                    elif prune_reason == "same_confidence_smaller_evidence":
                        num_pruned_same_confidence_smaller_evidence += 1
                    continue

                serialized_rule = _serialize_rule(
                    rule_index=-1,
                    round_index=round_index,
                    head_predicate=head_predicate,
                    head_atom_template=head_atom_template,
                    body_atom_templates=new_body_atoms,
                    parent_rule_id=parent_rule_id,
                    added_body_atom_template=initial_body_atom_template,
                    source_initial_rule_id=source_initial_rule_id,
                    evidence_set=intersected_evidence,
                    evidence_summary=evidence_summary,
                    provenance_summary=provenance_summary,
                    parent_confidence=parent_confidence,
                    parent_positive_support=parent_positive_support,
                    prune_reason=str(prune_decision.get("prune_reason", "")),
                    prune_status=str(prune_decision.get("prune_status", "kept")),
                    kept_after_prune=bool(prune_decision.get("kept", True)),
                )
                serialized_rule["extension_rule_category"] = _extension_rule_category(serialized_rule)
                serialized_rule["firing_signature"] = _firing_signature(intersected_evidence)
                parent_candidate_rules.append(serialized_rule)

            parent_candidate_rules.sort(key=_extension_rank_key)
            if per_parent_extension_top_k >= 0 and len(parent_candidate_rules) > per_parent_extension_top_k:
                num_pruned_parent_top_k += len(parent_candidate_rules) - per_parent_extension_top_k
                parent_candidate_rules = parent_candidate_rules[:per_parent_extension_top_k]
            round_candidate_pool.extend(parent_candidate_rules)

        body_dedup_map: Dict[Tuple[str, str, Tuple[str, ...]], Dict[str, Any]] = {}
        for candidate_rule in round_candidate_pool:
            candidate_category = _extension_rule_category(candidate_rule)
            _increment_category_count(round_generated_category_counts, candidate_category)
            key = (
                str(candidate_rule.get("head_predicate", "")),
                str(candidate_rule.get("head_atom_template", "")),
                tuple(candidate_rule.get("body_atom_templates", [])),
            )
            existing = body_dedup_map.get(key)
            if existing is None or _extension_rank_key(candidate_rule) < _extension_rank_key(existing):
                if existing is not None:
                    num_deduplicated_body += 1
                body_dedup_map[key] = candidate_rule
            else:
                num_deduplicated_body += 1

        signature_dedup_map: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Dict[str, Any]] = {}
        for candidate_rule in body_dedup_map.values():
            key = (
                str(candidate_rule.get("head_predicate", "")),
                tuple(candidate_rule.get("firing_signature", ())),
            )
            existing = signature_dedup_map.get(key)
            if existing is None or _extension_rank_key(candidate_rule) < _extension_rank_key(existing):
                if existing is not None:
                    num_deduplicated_firing_signature += 1
                signature_dedup_map[key] = candidate_rule
            else:
                num_deduplicated_firing_signature += 1

        kept_budget_counts = _empty_category_counts()
        for candidate_rule in sorted(signature_dedup_map.values(), key=_extension_rank_key):
            category = _extension_rule_category(candidate_rule)
            if not _budget_allows_rule(
                category=category,
                kept_counts=kept_budget_counts,
                max_round_rules=max_round_rules,
                category_budgets=category_budgets,
            ):
                num_pruned_round_budget += 1
                _increment_category_count(round_pruned_budget_category_counts, category)
                continue
            _increment_category_count(kept_budget_counts, category)
            key = (
                str(candidate_rule.get("head_predicate", "")),
                str(candidate_rule.get("head_atom_template", "")),
                tuple(candidate_rule.get("body_atom_templates", [])),
            )
            next_rule_map[key] = candidate_rule

        round_rules = sorted(
            next_rule_map.values(),
            key=lambda rule: (
                int(rule.get("body_length", 0)),
                str(rule.get("clause", "")),
            ),
        )
        for rule_index, rule in enumerate(round_rules):
            rule["rule_index"] = rule_index
            rule["rule_id"] = f"extended_round_{round_index}_rule_{rule_index:06d}"
            if bool(rule.get("uses_candidate_atoms", False)):
                round_rules_using_candidate_atoms += 1
            if bool(rule.get("uses_only_candidate_atoms", False)):
                round_candidate_only_rules += 1
            if bool(rule.get("mixes_accepted_and_candidate_atoms", False)):
                round_mixed_source_rules += 1

        total_rules_using_candidate_atoms += round_rules_using_candidate_atoms
        total_candidate_only_rules += round_candidate_only_rules
        total_mixed_source_rules += round_mixed_source_rules
        round_candidate_rule_counts = _count_rule_categories(round_rules)

        round_payload: Dict[str, Any] = {
            "version": _EXTENDED_RULES_VERSION,
            "round_index": round_index,
            "config": {
                "input_initial_rule_pool_key": input_initial_rule_pool_key,
                "num_rounds": num_rounds,
                "evaluation_strategy": evaluation_strategy,
                "prune_strategies": prune_strategies,
                "min_positive_support_to_extend": min_positive_support_to_extend,
                "same_confidence_smaller_evidence_enabled": same_confidence_smaller_evidence_enabled,
                "per_parent_extension_top_k": per_parent_extension_top_k,
                "max_round_rules": max_round_rules,
                "category_budgets": category_budgets,
            },
            "num_candidates_generated": num_candidates_generated,
            "num_parent_top_k_pruned": num_pruned_parent_top_k,
            "num_body_deduplicated": num_deduplicated_body,
            "num_firing_signature_deduplicated": num_deduplicated_firing_signature,
            "num_round_budget_pruned": num_pruned_round_budget,
            "num_rules": len(round_rules),
            "evaluation": {
                "status": "implemented",
                "strategy": evaluation_strategy,
            },
            "pruning": {
                "status": "implemented",
                "strategies": prune_strategies,
                "parent_rules_skipped_num": num_parent_rules_skipped,
                "pruned_num_rules": num_pruned,
                "pruned_low_evidence": num_pruned_low_evidence,
                "pruned_empty_evidence": num_pruned_empty_evidence,
                "pruned_no_positive_firings": num_pruned_no_positive,
                "pruned_same_firings_as_parent": num_pruned_same_firings_as_parent,
                "pruned_same_confidence_smaller_evidence": num_pruned_same_confidence_smaller_evidence,
                "pruned_provenance_dominance": num_pruned_provenance_dominance,
                "pruned_parent_top_k": num_pruned_parent_top_k,
                "pruned_round_budget": num_pruned_round_budget,
                "deduplicated_body": num_deduplicated_body,
                "deduplicated_firing_signature": num_deduplicated_firing_signature,
                "kept_num_rules": len(round_rules),
            },
            "candidate_rule_provenance": {
                "num_rules_using_candidate_atoms": round_rules_using_candidate_atoms,
                "num_candidate_only_rules": round_candidate_only_rules,
                "num_mixed_source_rules": round_mixed_source_rules,
                "generated_rule_counts": round_generated_category_counts,
                "budget_pruned_rule_counts": round_pruned_budget_category_counts,
                "kept_rule_counts": round_candidate_rule_counts,
            },
            "rules": round_rules,
        }
        json_path, csv_path = _write_round_outputs(out_root, round_index, round_payload)

        round_summaries.append(
            {
                "round_index": round_index,
                "num_candidates_generated": num_candidates_generated,
                "num_rules": len(round_rules),
                "json_path": str(json_path),
                "csv_path": str(csv_path),
                "evaluation_status": "implemented",
                "prune_status": "implemented",
                "parent_rules_skipped_num": num_parent_rules_skipped,
                "pruned_num_rules": num_pruned,
                "pruned_low_evidence": num_pruned_low_evidence,
                "pruned_empty_evidence": num_pruned_empty_evidence,
                "pruned_no_positive_firings": num_pruned_no_positive,
                "pruned_same_firings_as_parent": num_pruned_same_firings_as_parent,
                "pruned_same_confidence_smaller_evidence": num_pruned_same_confidence_smaller_evidence,
                "pruned_provenance_dominance": num_pruned_provenance_dominance,
                "pruned_parent_top_k": num_pruned_parent_top_k,
                "pruned_round_budget": num_pruned_round_budget,
                "deduplicated_body": num_deduplicated_body,
                "deduplicated_firing_signature": num_deduplicated_firing_signature,
                "num_rules_using_candidate_atoms": round_rules_using_candidate_atoms,
                "num_candidate_only_rules": round_candidate_only_rules,
                "num_mixed_source_rules": round_mixed_source_rules,
                "generated_rule_counts": round_generated_category_counts,
                "budget_pruned_rule_counts": round_pruned_budget_category_counts,
                "kept_rule_counts": round_candidate_rule_counts,
            }
        )
        rounds.append(round_payload)
        current_rules = round_rules

        print(
            f"  round={round_index}: candidates={num_candidates_generated} | "
            f"parent_topk_pruned={num_pruned_parent_top_k} | "
            f"deduped={num_deduplicated_body + num_deduplicated_firing_signature} | "
            f"budget_pruned={num_pruned_round_budget} | "
            f"kept={len(round_rules)}"
        )
        print(
            "    parent_rules_skipped: "
            f"total={num_parent_rules_skipped}"
        )
        print(
            "    pruned_by_strategy: "
            f"total={num_pruned} | "
            f"low_evidence={num_pruned_low_evidence} | "
            f"empty={num_pruned_empty_evidence} | "
            f"no_positive={num_pruned_no_positive} | "
            f"same_firings_as_parent={num_pruned_same_firings_as_parent} | "
            "same_confidence_smaller_evidence="
            f"{num_pruned_same_confidence_smaller_evidence} | "
            f"provenance_dominance={num_pruned_provenance_dominance}"
        )

        if not current_rules:
            break

    all_kept_rules = _build_all_kept_rules(initial_rules, rounds)
    all_kept_counts = _count_rule_categories(all_kept_rules)
    extension_only_counts = _count_rule_categories(
        [rule for round_payload in rounds for rule in list(round_payload.get("rules", []))]
    )
    upstream_flow_summary = dict(merged_initial_rules.get("candidate_rule_flow_summary", {}))
    candidate_rule_flow_summary = {
        "atom_availability": dict(upstream_flow_summary.get("atom_availability", {})),
        "initial_rule_generation": dict(
            upstream_flow_summary.get(
                "initial_rule_generation",
                merged_initial_rules.get("candidate_rule_stage_stats", {}).get(
                    "input_generated_rule_counts", _empty_category_counts()
                ),
            )
        ),
        "merged_after_step15": dict(
            upstream_flow_summary.get(
                "merged_after_step15",
                merged_initial_rules.get("candidate_rule_stage_stats", {}).get(
                    "merged_rule_counts", _count_rule_categories(initial_rules)
                ),
            )
        ),
        "pruning": total_pruned_candidate_rule_counts,
        "extension": {
            "extension_only_kept_rule_counts": extension_only_counts,
            "all_kept_after_step16_rule_counts": all_kept_counts,
        },
        "final_selection": _empty_category_counts(),
        "evaluation": _empty_category_counts(),
    }

    manifest = {
        "version": _EXTENDED_RULES_VERSION,
        "config": {
            "input_initial_rule_pool_key": input_initial_rule_pool_key,
            "num_rounds": num_rounds,
            "evaluation_strategy": evaluation_strategy,
            "prune_strategies": prune_strategies,
            "min_positive_support_to_extend": min_positive_support_to_extend,
            "same_confidence_smaller_evidence_enabled": same_confidence_smaller_evidence_enabled,
            "per_parent_extension_top_k": per_parent_extension_top_k,
            "max_round_rules": max_round_rules,
            "max_round_accepted_only_rules": category_budgets["accepted_only"],
            "max_round_mixed_candidate_rules": category_budgets["accepted_candidate"],
            "max_round_candidate_only_rules": category_budgets["candidate_only"],
            "max_round_candidate_candidate_rules": category_budgets["candidate_candidate"],
        },
        "input_num_examples": int(merged_initial_rules.get("num_examples", 0)),
        "input_num_positive_examples": total_positive_examples,
        "input_num_negative_examples": int(merged_initial_rules.get("num_negative_examples", 0)),
        "input_num_initial_rules": len(all_input_initial_rules),
        "input_num_unary_initial_rules_used": len(initial_rules),
        "input_num_skipped_non_unary_initial_rules": num_skipped_non_unary_initial_rules,
        "input_num_unique_initial_body_atoms": len(initial_rule_atoms),
        "input_num_rules_using_candidate_atoms": int(unary_initial_rule_counts.get("candidate_involving_rules", 0)),
        "input_num_candidate_only_rules": int(unary_initial_rule_counts.get("candidate_only_rules", 0)),
        "input_num_mixed_source_rules": int(unary_initial_rule_counts.get("mixed_accepted_candidate_rules", 0)),
        "num_rounds_completed": len(rounds),
        "num_all_kept_rules": len(all_kept_rules),
        "num_rules_using_candidate_atoms": total_rules_using_candidate_atoms,
        "num_candidate_only_rules": total_candidate_only_rules,
        "num_mixed_source_rules": total_mixed_source_rules,
        "candidate_rule_stage_stats": {
            "stage": "step16_extension",
            "input_initial_rule_counts": unary_initial_rule_counts,
            "extension_only_kept_rule_counts": extension_only_counts,
            "all_kept_after_step16_rule_counts": all_kept_counts,
            "pruned_rule_counts": total_pruned_candidate_rule_counts,
        },
        "candidate_rule_flow_summary": candidate_rule_flow_summary,
        "all_kept_rules": all_kept_rules,
        "round_summaries": round_summaries,
        "rounds": rounds,
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Extended rules manifest written to {manifest_path}")
    return manifest


def run(
    merged_initial_rules: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_rules(
        merged_initial_rules=merged_initial_rules,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
