"""
Generate ranked perception re-check feedback requests from downstream reasoning failures.

Outputs:
    pipeline_output/23_driving_mini_reasoning_feedback_signal/
        background_causal_prior.json
        feedback_signal_summary.json
        reasoning_feedback_requests.csv
        reasoning_feedback_requests.json
        unexplained_positive_examples.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from ast import literal_eval
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config
from src.exp_driving_videos.modules.evaluate_rules_driving_mini import (
    _find_rule_matches_for_example,
    _get_rule_body_atom_templates,
    _parse_atom,
)


_FEEDBACK_VERSION = 6
_SUPPORTING_FACT_PREDICATES: Set[str] = {
    "object_class",
    "object_distance_state",
    "object_x_position_state",
    "object_visibility_state",
    "object_vx_state",
    "object_speed_state",
    "traffic_control_type",
    "traffic_control_relevant",
    "traffic_light_relevant",
    "traffic_light_state",
    "traffic_light_position_state",
    "stop_sign_relevant",
}
_EXPLAINABILITY_GAP_LEVELS: Set[str] = {
    "missing_rule_or_predicate_dense_context",
    "missing_rule_or_predicate_sparse_context",
    "unexplained_noise_or_symbol_gap",
    "unexplained_noise_no_objects",
}
_HIGH_PRIORITY_DISTANCE_STATES: Set[str] = {"near", "very_near", "close"}
_HIGH_PRIORITY_POSITION_STATES: Set[str] = {"centered", "front_center"}
_LOW_VISIBILITY_STATES: Set[str] = {"brief", "intermittent"}
_WEAK_DISTANCE_STATES: Set[str] = {"medium", "far", "unknown"}
_WEAK_POSITION_STATES: Set[str] = {"left_of_ego", "right_of_ego", "unknown"}
_NON_INFORMATIVE_SPEED_STATES: Set[str] = {"", "unknown", "rel_static", "constant_speed", "moving", "uncertain"}
_NON_INFORMATIVE_VX_STATES: Set[str] = {"", "unknown", "vx_unknown", "vx_stable", "stable", "uncertain"}
_DYNAMIC_RELEVANT_PRIOR_IDS: Set[str] = {"lead_vehicle", "pedestrian", "cyclist", "obstacle"}
_CONTROL_PRIOR_IDS: Set[str] = {"traffic_light", "stop_sign"}
_PRIOR_RULE_HINTS: Dict[str, Set[str]] = {
    "lead_vehicle": {"object_distance_state", "object_x_position_state", "blocking_lane", "stopped", "slowing"},
    "pedestrian": {"person", "pedestrian", "crossing", "near_road_boundary_presence"},
    "cyclist": {"bicycle", "motorcycle", "rider", "crossing", "merging", "sharing_lane"},
    "traffic_light": {"traffic_light", "traffic_light_state", "traffic_light_relevant", "traffic_control_relevant"},
    "stop_sign": {"stop_sign", "stop_sign_relevant", "traffic_control_relevant"},
    "obstacle": {"unknown", "barrier", "debris", "cone", "blocking_lane"},
    "crosswalk": {"crosswalk", "pedestrian", "yield"},
    "intersection": {"intersection", "junction", "traffic_light", "stop_sign"},
}


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "23_driving_mini_reasoning_feedback_signal"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target_predicate": str(cfg.get("target_predicate", "brake_next")),
        "primary_rule_set": str(cfg.get("primary_rule_set", "original")),
        "max_feedback_requests": int(cfg.get("max_feedback_requests", 200)),
        "max_prior_requests_per_example": int(cfg.get("max_prior_requests_per_example", 3)),
        "low_confidence_positive_margin": float(cfg.get("low_confidence_positive_margin", 0.08)),
        "min_existing_supporting_facts": int(cfg.get("min_existing_supporting_facts", 2)),
        "max_supporting_facts_per_request": int(cfg.get("max_supporting_facts_per_request", 16)),
        "max_fired_rules_per_request": int(cfg.get("max_fired_rules_per_request", 6)),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    def _serialize(value: Any) -> Any:
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True)
        if isinstance(value, set):
            return json.dumps(sorted(value))
        if isinstance(value, (list, tuple)):
            return json.dumps(list(value))
        return value

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize(row.get(key, "")) for key in fieldnames})


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not str(path) or not path.exists() or path.is_dir():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _parse_list_like(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    raw = str(value or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return list(parsed) if isinstance(parsed, list) else []
    except Exception:
        pass
    try:
        parsed = literal_eval(raw)
        return list(parsed) if isinstance(parsed, (list, tuple, set)) else []
    except Exception:
        return []


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _segment_lookup(logic_atom_results: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for video_result in logic_atom_results:
        video_id = str(video_result.get("video_id", ""))
        for segment in list(video_result.get("segments", [])):
            segment_id = str(segment.get("segment_id", ""))
            if not video_id or not segment_id:
                continue
            lookup[(video_id, segment_id)] = {
                "segment_index": _safe_int(segment.get("segment_index", -1), -1),
                "segment_label": str(segment.get("segment_label", "")),
                "segment_forward_label": str(segment.get("segment_forward_label", "")),
                "segment_lateral_label": str(segment.get("segment_lateral_label", "")),
                "start_frame": _safe_int(segment.get("start_frame", -1), -1),
                "end_frame": _safe_int(segment.get("end_frame", -1), -1),
            }
    return lookup


def _example_lookup(eval_temporal_rule_results: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for video_result in eval_temporal_rule_results:
        video_id = str(video_result.get("video_id", ""))
        for example in list(video_result.get("examples", [])):
            example_id = str(example.get("example_id", ""))
            if not example_id:
                continue
            lookup[example_id] = {
                "video_id": video_id,
                "example_id": example_id,
                "label": bool(example.get("label", False)),
                "target_predicate": str(example.get("target_predicate", "")),
                "current_segment_id": str(example.get("current_segment_id", "")),
                "next_segment_id": str(example.get("next_segment_id", "")),
                "current_segment_index": _safe_int(example.get("current_segment_index", -1), -1),
                "current_segment_label": str(example.get("current_segment_label", "")),
                "body_atoms": list(example.get("body_atoms", [])),
            }
    return lookup


def _rule_clause_lookup(primary_rule_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for rule in list(primary_rule_results.get("final_rules", [])):
        rule_id = str(rule.get("rule_id", ""))
        if rule_id:
            lookup[rule_id] = dict(rule)
    return lookup


def _rule_prediction_lookup(
    evaluation_results: Dict[str, Any],
    primary_rule_results: Dict[str, Any],
    example_lookup: Dict[str, Dict[str, Any]],
    max_rules_per_request: int,
) -> Dict[str, Dict[str, Any]]:
    csv_path = Path(str(evaluation_results.get("example_predictions_csv_path", "")))
    rows = _read_csv(csv_path)
    rule_lookup = _rule_clause_lookup(primary_rule_results)
    predictions: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id", ""))
        example = example_lookup.get(example_id)
        matching_rule_ids = [str(rule_id) for rule_id in _parse_list_like(row.get("matching_rule_ids", "[]"))]
        fired_rules: List[Dict[str, Any]] = []
        if example is not None:
            for rule_id in matching_rule_ids[:max_rules_per_request]:
                rule = rule_lookup.get(str(rule_id))
                if not rule:
                    fired_rules.append({"rule_id": str(rule_id), "clause": "", "matched_atoms": {}})
                    continue
                match_states = _find_rule_matches_for_example(
                    body_atom_templates=_get_rule_body_atom_templates(rule),
                    body_atoms=list(example.get("body_atoms", [])),
                )
                matched_atoms = dict(match_states[0].get("matched_atoms", {})) if match_states else {}
                fired_rules.append(
                    {
                        "rule_id": str(rule_id),
                        "clause": str(rule.get("clause", "")),
                        "matched_atoms": matched_atoms,
                    }
                )
        predictions[example_id] = {
            "predicted_positive": _parse_bool(row.get("predicted_positive", False)),
            "num_matching_rules": _safe_int(row.get("num_matching_rules", 0), 0),
            "matching_rule_ids": [str(rule_id) for rule_id in matching_rule_ids],
            "fired_rules": fired_rules,
        }
    return predictions


def _aggregation_prediction_lookup(rule_aggregation_baseline_results: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], float]:
    output_paths = dict(rule_aggregation_baseline_results.get("output_paths", {}))
    csv_path = Path(str(output_paths.get("prediction_examples_csv", "")))
    rows = _read_csv(csv_path)
    threshold = _safe_float(rule_aggregation_baseline_results.get("best_validation_threshold", 0.5), 0.5)
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id", ""))
        lookup[example_id] = {
            "predicted_probability": _safe_float(row.get("predicted_probability", 0.0), 0.0),
            "predicted_label_at_best_validation_threshold": _parse_bool(
                row.get("predicted_label_at_best_validation_threshold", False)
            ),
            "predicted_label_at_0_5": _parse_bool(row.get("predicted_label_at_0_5", False)),
        }
    return lookup, threshold


def _fn_lookup(error_analysis_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    fn_path = Path(str(error_analysis_results.get("fn_examples_path", "")))
    rows = _read_csv(fn_path)
    return {str(row.get("example_id", "")): dict(row) for row in rows if str(row.get("example_id", ""))}


def _uncovered_positive_example_ids(error_analysis_results: Dict[str, Any]) -> Set[str]:
    uncovered_path = Path(str(error_analysis_results.get("uncovered_positive_summary_path", "")))
    example_ids: Set[str] = set()
    for row in _read_csv(uncovered_path):
        raw = str(row.get("example_ids", "[]"))
        try:
            values = list(json.loads(raw))
        except Exception:
            values = []
        for value in values:
            if str(value):
                example_ids.add(str(value))
    return example_ids


def _flatten_prior_classes(prior_entries: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    object_classes: List[str] = []
    context_entities: List[str] = []
    seen_object = set()
    seen_context = set()
    for entry in sorted(prior_entries, key=lambda row: -_safe_float(row.get("priority", 0.0), 0.0)):
        candidate_classes = [str(value) for value in list(entry.get("candidate_classes", []))]
        if str(entry.get("entity_kind", "")) == "scene_context":
            for candidate_class in candidate_classes:
                normalized = _normalize_text(candidate_class)
                if normalized and normalized not in seen_context:
                    seen_context.add(normalized)
                    context_entities.append(str(candidate_class))
            continue
        for candidate_class in candidate_classes:
            normalized = _normalize_text(candidate_class)
            if normalized and normalized not in seen_object:
                seen_object.add(normalized)
                object_classes.append(str(candidate_class))
    return object_classes, context_entities


def _body_atom_signals(body_atoms: Sequence[str]) -> Dict[str, Any]:
    observed_classes: Set[str] = set()
    object_classes_by_id: Dict[str, str] = {}
    object_facts_by_id: Dict[str, Dict[str, str]] = {}
    object_fact_atoms_by_id: Dict[str, List[str]] = {}
    supporting_facts: List[str] = []
    traffic_control_fact_present = False
    traffic_control_facts: List[str] = []
    traffic_light_states: Set[str] = set()
    traffic_control_relevant = False

    for atom in body_atoms:
        parsed = _parse_atom(str(atom))
        if parsed is None:
            continue
        predicate, args = parsed
        if predicate == "object_class" and len(args) >= 3:
            object_id = str(args[1])
            class_name = str(args[2])
            observed_classes.add(_normalize_text(class_name))
            object_classes_by_id[object_id] = class_name
            object_facts_by_id.setdefault(object_id, {})["object_class"] = class_name
            object_fact_atoms_by_id.setdefault(object_id, []).append(str(atom))
        if predicate in _SUPPORTING_FACT_PREDICATES:
            if predicate.startswith("traffic_") or predicate == "stop_sign_relevant":
                supporting_facts.append(str(atom))
                traffic_control_facts.append(str(atom))
                traffic_control_fact_present = True
                if predicate in {"traffic_control_relevant", "traffic_light_relevant", "stop_sign_relevant"}:
                    traffic_control_relevant = True
                if predicate == "traffic_light_state" and len(args) >= 3:
                    traffic_light_states.add(_normalize_text(args[2]))
                continue
            object_id = str(args[1]) if len(args) >= 2 else ""
            if object_id:
                supporting_facts.append(str(atom))
                object_fact_atoms_by_id.setdefault(object_id, []).append(str(atom))
                if len(args) >= 3:
                    object_facts_by_id.setdefault(object_id, {})[predicate] = str(args[2])

    return {
        "observed_classes": observed_classes,
        "supporting_facts": supporting_facts,
        "traffic_control_fact_present": traffic_control_fact_present,
        "traffic_control_facts": traffic_control_facts,
        "traffic_light_states": traffic_light_states,
        "traffic_control_relevant": traffic_control_relevant,
        "object_classes_by_id": object_classes_by_id,
        "object_facts_by_id": object_facts_by_id,
        "object_fact_atoms_by_id": object_fact_atoms_by_id,
    }


def _prior_missing_candidates(
    prior_entry: Dict[str, Any],
    observed_classes: Set[str],
) -> Tuple[List[str], List[str]]:
    candidate_classes = [str(value) for value in list(prior_entry.get("candidate_classes", [])) if str(value)]
    if str(prior_entry.get("entity_kind", "")) == "scene_context":
        missing_context_entities = [
            class_name for class_name in candidate_classes if _normalize_text(class_name) not in observed_classes
        ]
        return [], missing_context_entities
    if any(_normalize_text(class_name) in observed_classes for class_name in candidate_classes):
        return [], []
    missing_candidate_classes = [
        class_name for class_name in candidate_classes if _normalize_text(class_name) not in observed_classes
    ]
    return missing_candidate_classes, []


def _candidate_object_classes(prior_entries: Sequence[Dict[str, Any]]) -> List[str]:
    object_classes, _ = _flatten_prior_classes(prior_entries)
    return object_classes


def _prior_matched_object_ids(prior_entry: Dict[str, Any], body_signal: Dict[str, Any]) -> List[str]:
    candidate_classes = {_normalize_text(value) for value in list(prior_entry.get("candidate_classes", [])) if str(value)}
    matched: List[str] = []
    for object_id, class_name in dict(body_signal.get("object_classes_by_id", {})).items():
        if _normalize_text(class_name) in candidate_classes:
            matched.append(str(object_id))
    return matched


def _prior_matching_supporting_facts(
    prior_entry: Dict[str, Any],
    body_signal: Dict[str, Any],
    *,
    max_supporting_facts_per_request: int,
) -> List[str]:
    matched_object_ids = _prior_matched_object_ids(prior_entry, body_signal)
    facts: List[str] = []
    for object_id in matched_object_ids:
        facts.extend(list(dict(body_signal.get("object_fact_atoms_by_id", {})).get(object_id, [])))
    prior_id = str(prior_entry.get("prior_id", ""))
    if prior_id in {"traffic_light", "stop_sign"}:
        facts.extend(list(body_signal.get("traffic_control_facts", [])))
    if prior_id == "crosswalk":
        facts.extend(
            [
                fact
                for fact in list(body_signal.get("traffic_control_facts", []))
                if "traffic_light" in _normalize_text(fact) or "stop_sign" in _normalize_text(fact)
            ]
        )
    if prior_id == "intersection":
        facts.extend(list(body_signal.get("traffic_control_facts", [])))
    deduped: List[str] = []
    seen: Set[str] = set()
    for fact in facts:
        fact_text = str(fact)
        if fact_text and fact_text not in seen:
            seen.add(fact_text)
            deduped.append(fact_text)
    return deduped[:max_supporting_facts_per_request]


def _requested_recheck_focus(
    prior_entry: Dict[str, Any],
    body_signal: Dict[str, Any],
    matched_object_ids: Sequence[str],
) -> List[str]:
    prior_id = str(prior_entry.get("prior_id", ""))
    default_focus = [str(value) for value in list(prior_entry.get("perception_recheck_focus", [])) if str(value)]
    object_facts_by_id = dict(body_signal.get("object_facts_by_id", {}))

    if prior_id == "traffic_light" and matched_object_ids:
        return [
            "state_classification",
            "control_relevance",
            "ego_relevance_refinement",
            "temporal_alignment",
        ]
    if prior_id == "stop_sign" and (
        matched_object_ids or bool(body_signal.get("traffic_control_fact_present", False))
    ):
        return [
            "control_relevance",
            "ego_relevance_refinement",
            "distance_to_stop",
        ]
    if prior_id == "lead_vehicle" and matched_object_ids:
        has_near_centered = False
        for object_id in matched_object_ids:
            fact_map = dict(object_facts_by_id.get(object_id, {}))
            if (
                _normalize_text(fact_map.get("object_distance_state", "")) in _HIGH_PRIORITY_DISTANCE_STATES
                and _normalize_text(fact_map.get("object_x_position_state", "")) in _HIGH_PRIORITY_POSITION_STATES
            ):
                has_near_centered = True
                break
        if has_near_centered:
            return [
                "closing_in",
                "distance_decreasing",
                "persistent_near",
                "lead_vehicle_relevance",
            ]
    return default_focus


def _prior_signal_profile(
    prior_entry: Dict[str, Any],
    body_signal: Dict[str, Any],
    matched_object_ids: Sequence[str],
    supporting_facts: Sequence[str],
) -> Dict[str, Any]:
    prior_id = str(prior_entry.get("prior_id", ""))
    object_facts_by_id = dict(body_signal.get("object_facts_by_id", {}))
    observed_classes = set(body_signal.get("observed_classes", set()))
    traffic_states = set(body_signal.get("traffic_light_states", set()))

    has_distance_state = False
    has_position_state = False
    has_visibility_state = False
    has_dynamic_state = False
    has_informative_dynamic_state = False
    has_strong_distance = False
    has_strong_position = False
    has_weak_distance = False
    has_weak_position = False

    for object_id in matched_object_ids:
        fact_map = dict(object_facts_by_id.get(object_id, {}))
        distance_state = _normalize_text(fact_map.get("object_distance_state", ""))
        position_state = _normalize_text(fact_map.get("object_x_position_state", ""))
        visibility_state = _normalize_text(fact_map.get("object_visibility_state", ""))
        speed_state = _normalize_text(fact_map.get("object_speed_state", ""))
        vx_state = _normalize_text(fact_map.get("object_vx_state", ""))

        has_distance_state = has_distance_state or bool(distance_state)
        has_position_state = has_position_state or bool(position_state)
        has_visibility_state = has_visibility_state or bool(visibility_state)
        has_dynamic_state = has_dynamic_state or bool(speed_state or vx_state)
        has_informative_dynamic_state = has_informative_dynamic_state or (
            (speed_state not in _NON_INFORMATIVE_SPEED_STATES)
            or (vx_state not in _NON_INFORMATIVE_VX_STATES)
        )
        has_strong_distance = has_strong_distance or (distance_state in _HIGH_PRIORITY_DISTANCE_STATES)
        has_strong_position = has_strong_position or (position_state in _HIGH_PRIORITY_POSITION_STATES)
        has_weak_distance = has_weak_distance or (distance_state in _WEAK_DISTANCE_STATES)
        has_weak_position = has_weak_position or (position_state in _WEAK_POSITION_STATES)

    has_control_relevance = bool(body_signal.get("traffic_control_relevant", False))
    has_control_state = bool(traffic_states) or any(
        "stop_sign_relevant" in _normalize_text(fact) for fact in supporting_facts
    )
    has_strong_control_signal = (
        ("red" in traffic_states)
        or ("yellow" in traffic_states)
        or any("stop_sign_relevant" in _normalize_text(fact) for fact in supporting_facts)
        or has_control_relevance
    )
    has_useful_state = (
        has_distance_state
        or has_position_state
        or has_visibility_state
        or has_dynamic_state
        or has_control_relevance
        or has_control_state
    )
    has_strong_relevance = has_strong_distance or has_strong_position or has_strong_control_signal
    has_weak_or_ambiguous_relevance = (
        has_weak_distance
        or has_weak_position
        or any(_normalize_text(fact).find("green") >= 0 for fact in supporting_facts)
        or (
            prior_id == "crosswalk"
            and any(_normalize_text(class_name) in observed_classes for class_name in ("person", "pedestrian"))
        )
    )
    has_any_prior_cue = bool(matched_object_ids) or has_control_state or has_control_relevance
    if str(prior_entry.get("entity_kind", "")) == "scene_context":
        crosswalk_cue = prior_id == "crosswalk" and any(
            _normalize_text(class_name) in observed_classes for class_name in ("person", "pedestrian")
        )
        intersection_cue = prior_id == "intersection" and (
            bool(body_signal.get("traffic_control_fact_present", False))
            or any(_normalize_text(class_name) in observed_classes for class_name in ("traffic_light", "stop_sign"))
        )
        has_any_prior_cue = has_any_prior_cue or crosswalk_cue or intersection_cue
        has_strong_relevance = has_strong_relevance or (
            prior_id == "intersection" and bool(body_signal.get("traffic_control_fact_present", False))
        )
        has_useful_state = has_useful_state or has_any_prior_cue

    return {
        "has_any_prior_cue": has_any_prior_cue,
        "has_useful_state": has_useful_state,
        "has_strong_relevance": has_strong_relevance,
        "has_weak_or_ambiguous_relevance": has_weak_or_ambiguous_relevance,
        "has_informative_dynamic_state": has_informative_dynamic_state,
        "has_control_state": has_control_state,
        "has_control_relevance": has_control_relevance,
    }


def _feedback_case_type(
    *,
    prior_entry: Dict[str, Any],
    body_signal: Dict[str, Any],
    matched_object_ids: Sequence[str],
    missing_candidate_classes: Sequence[str],
    missing_context_entities: Sequence[str],
    supporting_facts: Sequence[str],
    fired_rules: Sequence[Dict[str, Any]],
) -> str:
    has_any_object_facts = bool(body_signal.get("object_facts_by_id", {}))
    if not has_any_object_facts and not bool(body_signal.get("traffic_control_facts", [])):
        return "no_object_context"

    signal_profile = _prior_signal_profile(
        prior_entry=prior_entry,
        body_signal=body_signal,
        matched_object_ids=matched_object_ids,
        supporting_facts=supporting_facts,
    )
    if not signal_profile["has_any_prior_cue"] and (missing_candidate_classes or missing_context_entities):
        return "missing_causal_object"

    if not signal_profile["has_useful_state"]:
        return "existing_object_missing_state"

    if not fired_rules and signal_profile["has_strong_relevance"]:
        if str(prior_entry.get("prior_id", "")) in _DYNAMIC_RELEVANT_PRIOR_IDS and not signal_profile["has_informative_dynamic_state"]:
            return "existing_object_missing_dynamic_predicate"
        return "rule_abstraction_gap"

    if signal_profile["has_weak_or_ambiguous_relevance"]:
        return "existing_object_weak_relevance"

    if str(prior_entry.get("prior_id", "")) in _CONTROL_PRIOR_IDS and not (
        signal_profile["has_control_state"] or signal_profile["has_control_relevance"]
    ):
        return "existing_object_missing_state"

    if matched_object_ids and not signal_profile["has_strong_relevance"]:
        return "existing_object_weak_relevance"

    if missing_candidate_classes or missing_context_entities:
        return "missing_causal_object"

    return "rule_abstraction_gap" if not fired_rules else "existing_object_weak_relevance"


def _should_request_prior(
    prior_entry: Dict[str, Any],
    prior_score: Dict[str, Any],
    body_signal: Dict[str, Any],
) -> bool:
    case_type = str(prior_score.get("feedback_case_type", ""))
    if case_type == "no_object_context":
        return True
    if case_type == "missing_causal_object":
        return bool(prior_score.get("missing_candidate_classes")) or bool(prior_score.get("missing_context_entities"))
    if case_type in {
        "existing_object_missing_state",
        "existing_object_weak_relevance",
        "existing_object_missing_dynamic_predicate",
        "rule_abstraction_gap",
    }:
        if prior_score.get("matched_object_ids"):
            return True
        if str(prior_entry.get("prior_id", "")) in _CONTROL_PRIOR_IDS:
            return bool(body_signal.get("traffic_control_fact_present", False))
        if str(prior_entry.get("entity_kind", "")) == "scene_context":
            return bool(body_signal.get("traffic_control_fact_present", False)) or bool(body_signal.get("observed_classes", set()))
    return False


def _feedback_request_group_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("video_id", "")),
        str(row.get("example_id", "")),
        str(row.get("segment_id", "")),
    )


def _rule_alignment_score(prior_entry: Dict[str, Any], fired_rules: Sequence[Dict[str, Any]]) -> Tuple[float, int]:
    prior_id = str(prior_entry.get("prior_id", ""))
    candidate_classes = {_normalize_text(value) for value in list(prior_entry.get("candidate_classes", [])) if str(value)}
    rule_hints = set(_PRIOR_RULE_HINTS.get(prior_id, set()))
    matches = 0
    for rule in fired_rules:
        clause_text = _normalize_text(rule.get("clause", ""))
        matched_atoms_text = _normalize_text(json.dumps(rule.get("matched_atoms", {}), sort_keys=True))
        haystack = f"{clause_text} {matched_atoms_text}"
        if any(candidate_class and candidate_class in haystack for candidate_class in candidate_classes) or any(
            hint and _normalize_text(hint) in haystack for hint in rule_hints
        ):
            matches += 1
    return min(0.18, 0.09 * float(matches)), matches


def _common_feedback_score(
    *,
    rule_fn: bool,
    agg_fn: bool,
    low_conf_positive: bool,
    unexplained_positive: bool,
    supporting_facts_count: int,
) -> float:
    score = 0.0
    if unexplained_positive:
        score += 1.0
    if rule_fn:
        score += 0.5
    if agg_fn:
        score += 0.35
    if low_conf_positive:
        score += 0.2
    score += max(0.0, 0.2 - (0.03 * supporting_facts_count))
    return score


def _score_prior_request(
    *,
    prior_entry: Dict[str, Any],
    body_signal: Dict[str, Any],
    fired_rules: Sequence[Dict[str, Any]],
    explainability_level: str,
    rule_fn: bool,
    agg_fn: bool,
    low_conf_positive: bool,
    unexplained_positive: bool,
    max_supporting_facts_per_request: int,
) -> Dict[str, Any]:
    observed_classes = set(body_signal.get("observed_classes", set()))
    matched_object_ids = _prior_matched_object_ids(prior_entry, body_signal)
    object_facts_by_id = dict(body_signal.get("object_facts_by_id", {}))
    supporting_facts = _prior_matching_supporting_facts(
        prior_entry,
        body_signal,
        max_supporting_facts_per_request=max_supporting_facts_per_request,
    )
    missing_candidate_classes, missing_context_entities = _prior_missing_candidates(prior_entry, observed_classes)
    prior_id = str(prior_entry.get("prior_id", ""))
    if prior_id in _CONTROL_PRIOR_IDS and (
        matched_object_ids
        or any(prior_id in _normalize_text(fact) for fact in supporting_facts)
        or (prior_id == "traffic_light" and bool(body_signal.get("traffic_light_states", set())))
    ):
        missing_candidate_classes = []
    requested_recheck_focus = _requested_recheck_focus(prior_entry, body_signal, matched_object_ids)
    feedback_case_type = _feedback_case_type(
        prior_entry=prior_entry,
        body_signal=body_signal,
        matched_object_ids=matched_object_ids,
        missing_candidate_classes=missing_candidate_classes,
        missing_context_entities=missing_context_entities,
        supporting_facts=supporting_facts,
        fired_rules=fired_rules,
    )
    common_score = _common_feedback_score(
        rule_fn=rule_fn,
        agg_fn=agg_fn,
        low_conf_positive=low_conf_positive,
        unexplained_positive=unexplained_positive,
        supporting_facts_count=len(supporting_facts),
    )

    score = common_score + (0.35 * _safe_float(prior_entry.get("priority", 0.0), 0.0))
    if missing_candidate_classes:
        score += min(0.24, 0.12 * float(len(missing_candidate_classes)))
    if missing_context_entities:
        score += min(0.18, 0.09 * float(len(missing_context_entities)))
    if matched_object_ids:
        score += 0.28

    for object_id in matched_object_ids:
        fact_map = dict(object_facts_by_id.get(object_id, {}))
        distance_state = _normalize_text(fact_map.get("object_distance_state", ""))
        if distance_state in _HIGH_PRIORITY_DISTANCE_STATES:
            score += 0.16
        elif distance_state:
            score += 0.05

        position_state = _normalize_text(fact_map.get("object_x_position_state", ""))
        if position_state in _HIGH_PRIORITY_POSITION_STATES:
            score += 0.12
        elif position_state:
            score += 0.04

        visibility_state = _normalize_text(fact_map.get("object_visibility_state", ""))
        if visibility_state in _LOW_VISIBILITY_STATES:
            score += 0.1
        elif visibility_state:
            score += 0.03

        speed_state = _normalize_text(fact_map.get("object_speed_state", ""))
        vx_state = _normalize_text(fact_map.get("object_vx_state", ""))
        candidate_state_text = " ".join(_normalize_text(value) for value in list(prior_entry.get("candidate_states", [])))
        if speed_state and speed_state in candidate_state_text:
            score += 0.08
        if vx_state and vx_state in candidate_state_text:
            score += 0.05

    if prior_id == "traffic_light":
        if bool(body_signal.get("traffic_control_relevant", False)):
            score += 0.2
        traffic_states = set(body_signal.get("traffic_light_states", set()))
        if "red" in traffic_states:
            score += 0.2
        elif "yellow" in traffic_states:
            score += 0.14
        elif "green" in traffic_states:
            score += 0.02
    elif prior_id == "stop_sign":
        if any("stop_sign_relevant" in _normalize_text(fact) for fact in supporting_facts):
            score += 0.24
        elif bool(body_signal.get("traffic_control_relevant", False)):
            score += 0.08
    elif prior_id == "crosswalk":
        if any(_normalize_text(class_name) in observed_classes for class_name in ("person", "pedestrian")):
            score += 0.14
        if bool(body_signal.get("traffic_control_fact_present", False)):
            score += 0.08
    elif prior_id == "intersection":
        if bool(body_signal.get("traffic_control_fact_present", False)):
            score += 0.14
        if any(_normalize_text(class_name) in observed_classes for class_name in ("traffic_light", "stop_sign")):
            score += 0.08

    rule_alignment_score, matched_rule_count = _rule_alignment_score(prior_entry, fired_rules)
    score += rule_alignment_score

    if feedback_case_type == "no_object_context":
        score += 0.22
    elif feedback_case_type == "missing_causal_object":
        score += 0.18
    elif feedback_case_type == "existing_object_missing_state":
        score += 0.16
    elif feedback_case_type == "existing_object_weak_relevance":
        score += 0.12
    elif feedback_case_type == "existing_object_missing_dynamic_predicate":
        score += 0.2
    elif feedback_case_type == "rule_abstraction_gap":
        score += 0.18

    if explainability_level == "missing_rule_or_predicate_dense_context":
        score += 0.12
    elif explainability_level == "missing_rule_or_predicate_sparse_context":
        score += 0.08
    elif explainability_level == "unexplained_noise_or_symbol_gap":
        score += 0.05

    return {
        "feedback_score": float(score),
        "existing_supporting_facts": supporting_facts,
        "matched_object_ids": matched_object_ids,
        "missing_candidate_classes": missing_candidate_classes,
        "missing_context_entities": missing_context_entities,
        "matched_rule_count": int(matched_rule_count),
        "feedback_case_type": feedback_case_type,
        "requested_recheck_focus": requested_recheck_focus,
    }


def _current_prediction_text(
    rule_prediction: Dict[str, Any],
    agg_prediction: Dict[str, Any],
    agg_threshold: float,
) -> str:
    rule_text = (
        f"step18_rules={'positive' if bool(rule_prediction.get('predicted_positive', False)) else 'negative'}"
        f"(matches={_safe_int(rule_prediction.get('num_matching_rules', 0), 0)})"
    )
    if agg_prediction:
        agg_text = (
            "step18c_agg="
            f"{'positive' if bool(agg_prediction.get('predicted_label_at_best_validation_threshold', False)) else 'negative'}"
            f"(p={_safe_float(agg_prediction.get('predicted_probability', 0.0), 0.0):.3f},"
            f"thr={float(agg_threshold):.3f})"
        )
    else:
        agg_text = "step18c_agg=unavailable"
    return f"{rule_text}; {agg_text}"


def _error_type_and_reason(
    *,
    rule_fn: bool,
    agg_fn: bool,
    low_conf_positive: bool,
    unexplained_positive: bool,
    insufficient_causal_evidence: bool,
) -> Tuple[str, str]:
    if unexplained_positive:
        return "unexplained_positive_example", "unexplained_braking_positive"
    if rule_fn and agg_fn:
        return "false_negative_step18_and_step18c", "false_negative_both_reasoners"
    if rule_fn:
        return "false_negative_step18_rules", "rule_reasoning_false_negative"
    if low_conf_positive:
        return "low_confidence_positive_step18c", "low_confidence_positive"
    if agg_fn:
        return "false_negative_step18c_aggregation", "aggregation_false_negative"
    if insufficient_causal_evidence:
        return "positive_without_causal_support", "positive_without_sufficient_causal_support"
    return "positive_recheck_request", "reasoning_feedback_recheck"


def process_feedback_signals(
    background_causal_prior_results: Dict[str, Any],
    primary_rule_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    rule_aggregation_baseline_results: Optional[Dict[str, Any]] = None,
    error_analysis_results: Optional[Dict[str, Any]] = None,
    eval_temporal_rule_results: Optional[Sequence[Dict[str, Any]]] = None,
    logic_atom_results: Optional[Sequence[Dict[str, Any]]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)

    background_prior_json_path = out_root / "background_causal_prior.json"
    summary_json_path = out_root / "feedback_signal_summary.json"
    feedback_csv_path = out_root / "reasoning_feedback_requests.csv"
    requests_json_path = out_root / "reasoning_feedback_requests.json"
    unexplained_positive_csv_path = out_root / "unexplained_positive_examples.csv"

    if not force_recompute and summary_json_path.exists():
        with summary_json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _FEEDBACK_VERSION and _cfg_key_subset(cached.get("config", {})) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {summary_json_path.name}")
            return cached

    prior_entries = list(background_causal_prior_results.get("entries", []))
    background_prior_payload = {
        "usage_constraints": {
            "perception_recheck_prior_only": True,
            "not_direct_rule": True,
            "not_object_fact": True,
            "not_training_label": True,
        },
        "background_causal_prior": prior_entries,
    }
    with background_prior_json_path.open("w", encoding="utf-8") as fh:
        json.dump(background_prior_payload, fh, indent=2)

    example_lookup = _example_lookup(eval_temporal_rule_results or [])
    segment_lookup = _segment_lookup(logic_atom_results or [])
    rule_prediction_lookup = _rule_prediction_lookup(
        evaluation_results=evaluation_results,
        primary_rule_results=primary_rule_results,
        example_lookup=example_lookup,
        max_rules_per_request=int(cfg.get("max_fired_rules_per_request", 6)),
    )
    agg_prediction_lookup, agg_threshold = _aggregation_prediction_lookup(rule_aggregation_baseline_results or {})
    fn_lookup = _fn_lookup(error_analysis_results or {})
    uncovered_positive_ids = _uncovered_positive_example_ids(error_analysis_results or {})

    min_existing_supporting_facts = max(1, int(cfg.get("min_existing_supporting_facts", 2)))
    low_confidence_positive_margin = max(0.0, float(cfg.get("low_confidence_positive_margin", 0.08)))
    max_supporting_facts_per_request = max(1, int(cfg.get("max_supporting_facts_per_request", 16)))
    max_feedback_requests = max(1, int(cfg.get("max_feedback_requests", 200)))
    max_prior_requests_per_example = max(1, int(cfg.get("max_prior_requests_per_example", 3)))
    sorted_prior_entries = sorted(prior_entries, key=lambda row: -_safe_float(row.get("priority", 0.0), 0.0))
    candidate_object_classes = _candidate_object_classes(prior_entries)
    candidate_class_present_counts = Counter()
    candidate_class_missing_counts = Counter()
    unexplained_candidate_class_present_counts = Counter()
    unexplained_candidate_class_missing_counts = Counter()

    feedback_rows: List[Dict[str, Any]] = []
    unexplained_positive_rows: List[Dict[str, Any]] = []
    for example_id, example in sorted(example_lookup.items()):
        if not bool(example.get("label", False)):
            continue

        rule_prediction = dict(rule_prediction_lookup.get(example_id, {}))
        agg_prediction = dict(agg_prediction_lookup.get(example_id, {}))
        fn_row = dict(fn_lookup.get(example_id, {}))
        explainability_level = str(fn_row.get("explainability_level", ""))
        unexplained_positive = example_id in uncovered_positive_ids or explainability_level in _EXPLAINABILITY_GAP_LEVELS

        body_signal = _body_atom_signals(list(example.get("body_atoms", [])))
        observed_classes = set(body_signal["observed_classes"])
        for candidate_class in candidate_object_classes:
            normalized_class = _normalize_text(candidate_class)
            if normalized_class in observed_classes:
                candidate_class_present_counts[candidate_class] += 1
            else:
                candidate_class_missing_counts[candidate_class] += 1
        supporting_facts = list(body_signal["supporting_facts"])[:max_supporting_facts_per_request]
        fired_rules = list(rule_prediction.get("fired_rules", []))

        has_candidate_object_class = bool(observed_classes)
        sufficient_causal_evidence = (
            len(supporting_facts) >= min_existing_supporting_facts
            and (has_candidate_object_class or bool(body_signal.get("traffic_control_fact_present", False)))
            and bool(fired_rules)
        )
        insufficient_causal_evidence = not sufficient_causal_evidence or unexplained_positive

        rule_fn = not bool(rule_prediction.get("predicted_positive", False))
        agg_has_prediction = bool(agg_prediction)
        agg_fn = agg_has_prediction and not bool(agg_prediction.get("predicted_label_at_best_validation_threshold", False))
        low_conf_positive = (
            agg_has_prediction
            and bool(agg_prediction.get("predicted_label_at_best_validation_threshold", False))
            and _safe_float(agg_prediction.get("predicted_probability", 0.0), 0.0)
            <= float(agg_threshold + low_confidence_positive_margin)
        )

        should_request = insufficient_causal_evidence
        if not should_request:
            continue

        video_id = str(example.get("video_id", ""))
        current_segment_id = str(example.get("current_segment_id", ""))
        segment_meta = dict(segment_lookup.get((video_id, current_segment_id), {}))
        error_type, feedback_reason = _error_type_and_reason(
            rule_fn=rule_fn,
            agg_fn=agg_fn,
            low_conf_positive=low_conf_positive,
            unexplained_positive=unexplained_positive,
            insufficient_causal_evidence=insufficient_causal_evidence,
        )
        current_prediction = _current_prediction_text(
            rule_prediction=rule_prediction,
            agg_prediction=agg_prediction,
            agg_threshold=agg_threshold,
        )
        frame_range = (
            ""
            if _safe_int(segment_meta.get("start_frame", -1), -1) < 0
            else f"frame_{_safe_int(segment_meta.get('start_frame', -1), -1):05d}-frame_{_safe_int(segment_meta.get('end_frame', -1), -1):05d}"
        )

        ranked_prior_rows: List[Dict[str, Any]] = []
        for prior_entry in sorted_prior_entries:
            prior_score = _score_prior_request(
                prior_entry=prior_entry,
                body_signal=body_signal,
                fired_rules=fired_rules,
                explainability_level=explainability_level,
                rule_fn=rule_fn,
                agg_fn=agg_fn,
                low_conf_positive=low_conf_positive,
                unexplained_positive=unexplained_positive,
                max_supporting_facts_per_request=max_supporting_facts_per_request,
            )
            if not _should_request_prior(prior_entry, prior_score, body_signal):
                continue
            ranked_prior_rows.append(
                {
                    "example_id": example_id,
                    "video_id": video_id,
                    "segment_id": current_segment_id,
                    "current_segment_index": _safe_int(example.get("current_segment_index", -1), -1),
                    "frame_range": frame_range,
                    "start_frame": _safe_int(segment_meta.get("start_frame", -1), -1),
                    "end_frame": _safe_int(segment_meta.get("end_frame", -1), -1),
                    "error_type": error_type,
                    "feedback_reason": feedback_reason,
                    "target_label": True,
                    "current_prediction": current_prediction,
                    "step18_predicted_positive": bool(rule_prediction.get("predicted_positive", False)),
                    "step18_num_matching_rules": _safe_int(rule_prediction.get("num_matching_rules", 0), 0),
                    "step18c_predicted_probability": _safe_float(agg_prediction.get("predicted_probability", 0.0), 0.0),
                    "step18c_best_validation_threshold": float(agg_threshold),
                    "step18c_predicted_positive": bool(agg_prediction.get("predicted_label_at_best_validation_threshold", False)),
                    "low_confidence_positive": bool(low_conf_positive),
                    "unexplained_positive": bool(unexplained_positive),
                    "feedback_case_type": str(prior_score.get("feedback_case_type", "")),
                    "prior_id": str(prior_entry.get("prior_id", "")),
                    "prior_display_name": str(prior_entry.get("display_name", "")),
                    "prior_scope": str(prior_entry.get("prior_scope", "")),
                    "prior_entity_kind": str(prior_entry.get("entity_kind", "")),
                    "prior_priority": _safe_float(prior_entry.get("priority", 0.0), 0.0),
                    "prior_candidate_classes_json": json.dumps(list(prior_entry.get("candidate_classes", []))),
                    "prior_candidate_states_json": json.dumps(list(prior_entry.get("candidate_states", []))),
                    "prior_perception_recheck_focus_json": json.dumps(list(prior_entry.get("perception_recheck_focus", []))),
                    "requested_recheck_focus_json": json.dumps(list(prior_score.get("requested_recheck_focus", []))),
                    "missing_candidate_classes_json": json.dumps(prior_score["missing_candidate_classes"]),
                    "missing_context_entities_json": json.dumps(prior_score["missing_context_entities"]),
                    "requested_prior_ids_json": json.dumps([str(prior_entry.get("prior_id", ""))]),
                    "existing_supporting_facts_json": json.dumps(prior_score["existing_supporting_facts"]),
                    "matched_object_ids_json": json.dumps(prior_score["matched_object_ids"]),
                    "fired_rules_json": json.dumps(fired_rules),
                    "matched_rule_count": int(prior_score["matched_rule_count"]),
                    "explainability_level": explainability_level,
                    "feedback_score": float(prior_score["feedback_score"]),
                    "must_not_be_used_as_rule_or_fact": True,
                }
            )

        ranked_prior_rows = sorted(
            ranked_prior_rows,
            key=lambda row: (
                -_safe_float(row.get("feedback_score", 0.0), 0.0),
                -_safe_float(row.get("prior_priority", 0.0), 0.0),
                str(row.get("prior_id", "")),
            ),
        )
        requested_prior_ids_ranked = [str(row.get("prior_id", "")) for row in ranked_prior_rows]
        per_prior_score_dict = {
            str(row.get("prior_id", "")): _safe_float(row.get("feedback_score", 0.0), 0.0)
            for row in ranked_prior_rows
        }
        ranked_prior_rows = ranked_prior_rows[:max_prior_requests_per_example]
        top_feedback_prior_ids = [str(row.get("prior_id", "")) for row in ranked_prior_rows]

        if unexplained_positive:
            for candidate_class in candidate_object_classes:
                normalized_class = _normalize_text(candidate_class)
                if normalized_class in observed_classes:
                    unexplained_candidate_class_present_counts[candidate_class] += 1
                else:
                    unexplained_candidate_class_missing_counts[candidate_class] += 1
            unexplained_positive_rows.append(
                {
                    "example_id": example_id,
                    "video_id": video_id,
                    "segment_id": current_segment_id,
                    "current_segment_index": _safe_int(example.get("current_segment_index", -1), -1),
                    "frame_range": frame_range,
                    "start_frame": _safe_int(segment_meta.get("start_frame", -1), -1),
                    "end_frame": _safe_int(segment_meta.get("end_frame", -1), -1),
                    "target_label": True,
                    "current_prediction": current_prediction,
                    "error_type": error_type,
                    "feedback_reason": feedback_reason,
                    "explainability_level": explainability_level,
                    "missing_candidate_classes_json": json.dumps(
                        sorted(
                            {
                                candidate_class
                                for row in ranked_prior_rows
                                for candidate_class in _parse_list_like(row.get("missing_candidate_classes_json", "[]"))
                            }
                        )
                    ),
                    "missing_context_entities_json": json.dumps(
                        sorted(
                            {
                                entity_name
                                for row in ranked_prior_rows
                                for entity_name in _parse_list_like(row.get("missing_context_entities_json", "[]"))
                            }
                        )
                    ),
                    "existing_supporting_facts_json": json.dumps(supporting_facts),
                    "fired_rules_json": json.dumps(fired_rules),
                    "observed_classes_json": json.dumps(sorted(observed_classes)),
                    "selected_prior_ids_json": json.dumps([str(row.get("prior_id", "")) for row in ranked_prior_rows]),
                    "selected_prior_scores_json": json.dumps(
                        [
                            {
                                "prior_id": str(row.get("prior_id", "")),
                                "feedback_score": _safe_float(row.get("feedback_score", 0.0), 0.0),
                            }
                            for row in ranked_prior_rows
                        ]
                    ),
                    "selected_recheck_focus_json": json.dumps(
                        [
                            {
                                "prior_id": str(row.get("prior_id", "")),
                                "requested_recheck_focus": _parse_list_like(row.get("requested_recheck_focus_json", "[]")),
                            }
                            for row in ranked_prior_rows
                        ]
                    ),
                    "requested_prior_ids_ranked": requested_prior_ids_ranked,
                    "top_feedback_prior_ids": top_feedback_prior_ids,
                    "per_prior_score_dict": per_prior_score_dict,
                }
            )
        if not ranked_prior_rows:
            continue

        for rank_index, row in enumerate(ranked_prior_rows, start=1):
            row["prior_rank_for_example"] = rank_index
            row["max_prior_requests_per_example"] = max_prior_requests_per_example
            row["requested_prior_ids_ranked"] = requested_prior_ids_ranked
            row["top_feedback_prior_ids"] = top_feedback_prior_ids
            row["per_prior_score_dict"] = per_prior_score_dict
            feedback_rows.append(row)

    feedback_rows = sorted(
        feedback_rows,
        key=lambda row: (
            -_safe_float(row.get("feedback_score", 0.0), 0.0),
            str(row.get("video_id", "")),
            _safe_int(row.get("current_segment_index", -1), -1),
        ),
    )[:max_feedback_requests]

    grouped_feedback_rows: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in feedback_rows:
        grouped_feedback_rows.setdefault(_feedback_request_group_key(row), []).append(row)
    for grouped_rows in grouped_feedback_rows.values():
        grouped_rows.sort(
            key=lambda row: (
                -_safe_float(row.get("feedback_score", 0.0), 0.0),
                -_safe_float(row.get("prior_priority", 0.0), 0.0),
                str(row.get("prior_id", "")),
            )
        )
        requested_prior_ids_ranked = [str(row.get("prior_id", "")) for row in grouped_rows]
        top_feedback_prior_ids = list(requested_prior_ids_ranked)
        per_prior_score_dict = {
            str(row.get("prior_id", "")): _safe_float(row.get("feedback_score", 0.0), 0.0)
            for row in grouped_rows
        }
        for rank_index, row in enumerate(grouped_rows, start=1):
            row["prior_rank_for_example"] = rank_index
            row["num_selected_priors_for_request"] = len(grouped_rows)
            row["requested_prior_ids_ranked"] = requested_prior_ids_ranked
            row["top_feedback_prior_ids"] = top_feedback_prior_ids
            row["per_prior_score_dict"] = per_prior_score_dict

    _write_csv(
        feedback_csv_path,
        [
            "example_id",
            "video_id",
            "segment_id",
            "current_segment_index",
            "frame_range",
            "start_frame",
            "end_frame",
            "error_type",
            "feedback_reason",
            "target_label",
            "current_prediction",
            "step18_predicted_positive",
            "step18_num_matching_rules",
            "step18c_predicted_probability",
            "step18c_best_validation_threshold",
            "step18c_predicted_positive",
            "low_confidence_positive",
            "unexplained_positive",
            "feedback_case_type",
            "prior_id",
            "prior_display_name",
            "prior_scope",
            "prior_entity_kind",
            "prior_priority",
            "prior_rank_for_example",
            "max_prior_requests_per_example",
            "num_selected_priors_for_request",
            "prior_candidate_classes_json",
            "prior_candidate_states_json",
            "prior_perception_recheck_focus_json",
            "requested_recheck_focus_json",
            "missing_candidate_classes_json",
            "missing_context_entities_json",
            "requested_prior_ids_json",
            "existing_supporting_facts_json",
            "matched_object_ids_json",
            "fired_rules_json",
            "matched_rule_count",
            "explainability_level",
            "feedback_score",
            "requested_prior_ids_ranked",
            "top_feedback_prior_ids",
            "per_prior_score_dict",
            "must_not_be_used_as_rule_or_fact",
        ],
        feedback_rows,
    )

    with requests_json_path.open("w", encoding="utf-8") as fh:
        json.dump(feedback_rows, fh, indent=2)

    _write_csv(
        unexplained_positive_csv_path,
        [
            "example_id",
            "video_id",
            "segment_id",
            "current_segment_index",
            "frame_range",
            "start_frame",
            "end_frame",
            "target_label",
            "current_prediction",
            "error_type",
            "feedback_reason",
            "explainability_level",
            "missing_candidate_classes_json",
            "missing_context_entities_json",
            "existing_supporting_facts_json",
            "fired_rules_json",
            "observed_classes_json",
            "selected_prior_ids_json",
            "selected_prior_scores_json",
            "selected_recheck_focus_json",
            "requested_prior_ids_ranked",
            "top_feedback_prior_ids",
            "per_prior_score_dict",
        ],
        unexplained_positive_rows,
    )

    error_type_counts = Counter(str(row.get("error_type", "")) for row in feedback_rows)
    feedback_case_type_counts = Counter(str(row.get("feedback_case_type", "")) for row in feedback_rows)
    feedback_request_count_by_candidate_class = Counter()
    feedback_request_count_by_prior = Counter()
    rank_1_prior_counts = Counter()
    selected_prior_count_distribution = Counter()
    selected_top_k_prior_distribution = Counter()
    prior_selection_reason_summary: Dict[str, Dict[str, Any]] = {}
    for row in feedback_rows:
        feedback_request_count_by_prior[str(row.get("prior_id", ""))] += 1
        for candidate_class in _parse_list_like(row.get("missing_candidate_classes_json", "[]")):
            feedback_request_count_by_candidate_class[str(candidate_class)] += 1
        prior_id = str(row.get("prior_id", ""))
        prior_summary = prior_selection_reason_summary.setdefault(
            prior_id,
            {
                "total_selected_count": 0,
                "missing_causal_object_count": 0,
                "existing_object_refinement_count": 0,
                "no_object_context_count": 0,
                "other_case_count": 0,
                "feedback_case_type_counts": {},
            },
        )
        prior_summary["total_selected_count"] = int(prior_summary["total_selected_count"]) + 1
        feedback_case_type = str(row.get("feedback_case_type", ""))
        case_counts = dict(prior_summary.get("feedback_case_type_counts", {}))
        case_counts[feedback_case_type] = int(case_counts.get(feedback_case_type, 0)) + 1
        prior_summary["feedback_case_type_counts"] = case_counts
        if feedback_case_type == "missing_causal_object":
            prior_summary["missing_causal_object_count"] = int(prior_summary["missing_causal_object_count"]) + 1
        elif feedback_case_type in {
            "existing_object_missing_state",
            "existing_object_weak_relevance",
            "existing_object_missing_dynamic_predicate",
            "rule_abstraction_gap",
        }:
            prior_summary["existing_object_refinement_count"] = (
                int(prior_summary["existing_object_refinement_count"]) + 1
            )
        elif feedback_case_type == "no_object_context":
            prior_summary["no_object_context_count"] = int(prior_summary["no_object_context_count"]) + 1
        else:
            prior_summary["other_case_count"] = int(prior_summary["other_case_count"]) + 1

    for grouped_rows in grouped_feedback_rows.values():
        num_selected = len(grouped_rows)
        selected_prior_count_distribution[str(num_selected)] += 1
        top_prior_ids = [str(row.get("prior_id", "")) for row in grouped_rows]
        selected_top_k_prior_distribution[json.dumps(top_prior_ids)] += 1
        if top_prior_ids:
            rank_1_prior_counts[top_prior_ids[0]] += 1

    average_selected_priors_per_request = (
        float(sum(len(rows) for rows in grouped_feedback_rows.values())) / float(len(grouped_feedback_rows))
        if grouped_feedback_rows
        else 0.0
    )

    candidate_class_summary = {}
    for candidate_class in candidate_object_classes:
        candidate_class_summary[candidate_class] = {
            "feedback_request_count": int(feedback_request_count_by_candidate_class.get(candidate_class, 0)),
            "present_in_positive_examples": int(candidate_class_present_counts.get(candidate_class, 0)),
            "missing_in_positive_examples": int(candidate_class_missing_counts.get(candidate_class, 0)),
            "present_in_unexplained_positive_examples": int(unexplained_candidate_class_present_counts.get(candidate_class, 0)),
            "missing_in_unexplained_positive_examples": int(unexplained_candidate_class_missing_counts.get(candidate_class, 0)),
        }

    summary = {
        "version": _FEEDBACK_VERSION,
        "config": _cfg_key_subset(cfg),
        "usage_constraints": {
            "reasoning_feedback_only": True,
            "not_direct_rule": True,
            "not_object_fact": True,
            "not_training_signal": True,
            "does_not_insert_new_facts": True,
            "does_not_change_detections": True,
            "does_not_tune_parameters_from_evaluation_labels": True,
        },
        "target_predicate": str(cfg.get("target_predicate", "brake_next")),
        "primary_rule_set": str(cfg.get("primary_rule_set", "original")),
        "num_eval_examples": len(example_lookup),
        "num_positive_examples": sum(1 for example in example_lookup.values() if bool(example.get("label", False))),
        "num_unexplained_positive_examples": len(unexplained_positive_rows),
        "num_feedback_requests": len(feedback_rows),
        "num_feedback_request_groups": len(grouped_feedback_rows),
        "max_prior_requests_per_example": max_prior_requests_per_example,
        "error_type_counts": dict(sorted(error_type_counts.items())),
        "feedback_case_type_counts": dict(sorted(feedback_case_type_counts.items())),
        "average_selected_priors_per_request": average_selected_priors_per_request,
        "selected_prior_count_distribution": dict(sorted(selected_prior_count_distribution.items())),
        "selected_top_k_prior_distribution": dict(sorted(selected_top_k_prior_distribution.items())),
        "rank_1_prior_counts": dict(sorted(rank_1_prior_counts.items())),
        "feedback_request_count_by_prior": dict(sorted(feedback_request_count_by_prior.items())),
        "prior_selection_reason_summary": {
            prior_id: {
                **{
                    key: value
                    for key, value in prior_summary.items()
                    if key != "feedback_case_type_counts"
                },
                "feedback_case_type_counts": dict(sorted(dict(prior_summary.get("feedback_case_type_counts", {})).items())),
            }
            for prior_id, prior_summary in sorted(prior_selection_reason_summary.items())
        },
        "feedback_request_count_by_candidate_class": dict(sorted(feedback_request_count_by_candidate_class.items())),
        "candidate_class_presence_summary": candidate_class_summary,
        "source_diagnostics": {
            "background_causal_prior_summary": str(background_causal_prior_results.get("output_paths", {}).get("summary_json", "")),
            "step18_example_predictions_csv": str(evaluation_results.get("example_predictions_csv_path", "")),
            "step18c_prediction_examples_csv": str((rule_aggregation_baseline_results or {}).get("output_paths", {}).get("prediction_examples_csv", "")),
            "step19_fn_examples_csv": str((error_analysis_results or {}).get("fn_examples_path", "")),
            "step19_uncovered_positive_summary_csv": str((error_analysis_results or {}).get("uncovered_positive_summary_path", "")),
        },
        "requests": feedback_rows,
        "output_paths": {
            "background_causal_prior_json": str(background_prior_json_path),
            "summary_json": str(summary_json_path),
            "feedback_csv": str(feedback_csv_path),
            "requests_json": str(requests_json_path),
            "unexplained_positive_examples_csv": str(unexplained_positive_csv_path),
        },
    }
    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        "  reasoning_feedback_signal: "
        f"requests={len(feedback_rows)} | "
        f"top_reason={feedback_rows[0]['feedback_reason'] if feedback_rows else 'none'}"
    )
    print(f"Background causal prior JSON written to {background_prior_json_path}")
    print(f"Feedback signal summary JSON written to {summary_json_path}")
    print(f"Reasoning feedback requests CSV written to {feedback_csv_path}")
    print(f"Reasoning feedback requests JSON written to {requests_json_path}")
    print(f"Unexplained positive examples CSV written to {unexplained_positive_csv_path}")
    return summary


def run(
    background_causal_prior_results: Dict[str, Any],
    primary_rule_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    rule_aggregation_baseline_results: Optional[Dict[str, Any]] = None,
    error_analysis_results: Optional[Dict[str, Any]] = None,
    eval_temporal_rule_results: Optional[Sequence[Dict[str, Any]]] = None,
    logic_atom_results: Optional[Sequence[Dict[str, Any]]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_feedback_signals(
        background_causal_prior_results=background_causal_prior_results,
        primary_rule_results=primary_rule_results,
        evaluation_results=evaluation_results,
        rule_aggregation_baseline_results=rule_aggregation_baseline_results,
        error_analysis_results=error_analysis_results,
        eval_temporal_rule_results=eval_temporal_rule_results,
        logic_atom_results=logic_atom_results,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
