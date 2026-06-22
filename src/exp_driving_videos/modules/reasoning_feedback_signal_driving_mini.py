"""
Generate perception re-check feedback requests from downstream reasoning failures.

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


_FEEDBACK_VERSION = 3
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


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "23_driving_mini_reasoning_feedback_signal"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target_predicate": str(cfg.get("target_predicate", "brake_next")),
        "primary_rule_set": str(cfg.get("primary_rule_set", "original")),
        "max_feedback_requests": int(cfg.get("max_feedback_requests", 200)),
        "low_confidence_positive_margin": float(cfg.get("low_confidence_positive_margin", 0.08)),
        "min_existing_supporting_facts": int(cfg.get("min_existing_supporting_facts", 2)),
        "max_supporting_facts_per_request": int(cfg.get("max_supporting_facts_per_request", 16)),
        "max_fired_rules_per_request": int(cfg.get("max_fired_rules_per_request", 6)),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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
    object_ids_for_candidate_classes: Set[str] = set()
    candidate_objects: Dict[str, Dict[str, str]] = {}
    supporting_facts: List[str] = []
    traffic_control_fact_present = False

    for atom in body_atoms:
        parsed = _parse_atom(str(atom))
        if parsed is None:
            continue
        predicate, args = parsed
        if predicate == "object_class" and len(args) >= 3:
            object_id = str(args[1])
            class_name = str(args[2])
            observed_classes.add(_normalize_text(class_name))
            object_ids_for_candidate_classes.add(object_id)
            candidate_objects.setdefault(object_id, {})["object_class"] = class_name
        if predicate in _SUPPORTING_FACT_PREDICATES:
            if predicate.startswith("traffic_") or predicate == "stop_sign_relevant":
                supporting_facts.append(str(atom))
                traffic_control_fact_present = True
                continue
            object_id = str(args[1]) if len(args) >= 2 else ""
            if object_id and object_id in object_ids_for_candidate_classes:
                supporting_facts.append(str(atom))
                if len(args) >= 3:
                    candidate_objects.setdefault(object_id, {})[predicate] = str(args[2])

    return {
        "observed_classes": observed_classes,
        "supporting_facts": supporting_facts,
        "traffic_control_fact_present": traffic_control_fact_present,
        "candidate_objects": candidate_objects,
    }


def _missing_prior_candidates(
    prior_entries: Sequence[Dict[str, Any]],
    observed_classes: Set[str],
) -> Tuple[List[str], List[str], List[str]]:
    object_classes, context_entities = _flatten_prior_classes(prior_entries)
    missing_candidate_classes = [
        class_name
        for class_name in object_classes
        if _normalize_text(class_name) not in observed_classes
    ]
    requested_prior_ids = [
        str(entry.get("prior_id", ""))
        for entry in sorted(prior_entries, key=lambda row: -_safe_float(row.get("priority", 0.0), 0.0))
        if str(entry.get("prior_id", ""))
    ]
    return missing_candidate_classes, context_entities, requested_prior_ids


def _candidate_object_classes(prior_entries: Sequence[Dict[str, Any]]) -> List[str]:
    object_classes, _ = _flatten_prior_classes(prior_entries)
    return object_classes


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
        missing_candidate_classes, missing_context_entities, requested_prior_ids = _missing_prior_candidates(
            prior_entries=prior_entries,
            observed_classes=observed_classes,
        )
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
                    "missing_candidate_classes_json": json.dumps(missing_candidate_classes),
                    "missing_context_entities_json": json.dumps(missing_context_entities),
                    "existing_supporting_facts_json": json.dumps(supporting_facts),
                    "fired_rules_json": json.dumps(fired_rules),
                    "observed_classes_json": json.dumps(sorted(observed_classes)),
                }
            )
        if not missing_candidate_classes and not missing_context_entities:
            continue

        feedback_score = 0.0
        if unexplained_positive:
            feedback_score += 1.0
        if rule_fn:
            feedback_score += 0.5
        if agg_fn:
            feedback_score += 0.35
        if low_conf_positive:
            feedback_score += 0.2
        feedback_score += max(0.0, 0.2 - (0.03 * len(supporting_facts)))

        feedback_rows.append(
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
                "missing_candidate_classes_json": json.dumps(missing_candidate_classes),
                "missing_context_entities_json": json.dumps(missing_context_entities),
                "requested_prior_ids_json": json.dumps(requested_prior_ids),
                "existing_supporting_facts_json": json.dumps(supporting_facts),
                "fired_rules_json": json.dumps(fired_rules),
                "explainability_level": explainability_level,
                "feedback_score": float(feedback_score),
                "must_not_be_used_as_rule_or_fact": True,
            }
        )

    feedback_rows = sorted(
        feedback_rows,
        key=lambda row: (
            -_safe_float(row.get("feedback_score", 0.0), 0.0),
            str(row.get("video_id", "")),
            _safe_int(row.get("current_segment_index", -1), -1),
        ),
    )[:max_feedback_requests]

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
            "missing_candidate_classes_json",
            "missing_context_entities_json",
            "requested_prior_ids_json",
            "existing_supporting_facts_json",
            "fired_rules_json",
            "explainability_level",
            "feedback_score",
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
        ],
        unexplained_positive_rows,
    )

    error_type_counts = Counter(str(row.get("error_type", "")) for row in feedback_rows)
    feedback_request_count_by_candidate_class = Counter()
    for row in feedback_rows:
        for candidate_class in _parse_list_like(row.get("missing_candidate_classes_json", "[]")):
            feedback_request_count_by_candidate_class[str(candidate_class)] += 1

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
        "error_type_counts": dict(sorted(error_type_counts.items())),
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
