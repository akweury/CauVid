from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.exp_driving_videos.modules.evaluate_rules_driving_mini import (
    _find_rule_matches_for_example,
    _get_rule_body_atom_templates,
    _parse_atom,
)
from src.exp_driving_videos.modules import od_calibration_policy_utils


_PSEUDO_LABEL_VERSION = 1
_POSITION_OR_PROVENANCE_ONLY_PATTERNS = {
    "object_x_position_state_only",
    "position_provenance_prior_score_only",
    "prior_score_provenance_only",
    "weak_candidate_other",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "selector_mode": str(cfg.get("selector_mode", "primary_plus_best")),
        "max_match_states_per_example_rule": int(cfg.get("max_match_states_per_example_rule", 12)),
        "include_neutral_selected_candidate_detections": bool(
            cfg.get("include_neutral_selected_candidate_detections", True)
        ),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple, set)):
                    serialized[key] = json.dumps(value, sort_keys=isinstance(value, dict))
                else:
                    serialized[key] = value
            writer.writerow(serialized)


def _selector_names_to_trace(
    primary_rule_set: str,
    evaluation_results_by_name: Dict[str, Dict[str, Any]],
    candidate_contribution_summary_results: Dict[str, Any],
    cfg: Dict[str, Any],
) -> List[str]:
    selector_mode = str(cfg.get("selector_mode", "primary_plus_best"))
    names: List[str] = []
    if selector_mode == "all_available":
        names = sorted(evaluation_results_by_name)
    else:
        primary_name = _safe_text(primary_rule_set)
        if primary_name:
            names.append(primary_name)
        best_name = _safe_text(candidate_contribution_summary_results.get("best_selector_by_delta_f1", ""))
        if best_name:
            names.append(best_name)
    deduped: List[str] = []
    seen: Set[str] = set()
    for name in names:
        if name in evaluation_results_by_name and name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def _example_lookup(eval_temporal_rule_results: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for video_result in eval_temporal_rule_results:
        video_id = _safe_text(video_result.get("video_id", ""))
        for example in list(video_result.get("examples", [])):
            example_id = _safe_text(example.get("example_id", ""))
            if not example_id:
                continue
            lookup[example_id] = {
                "video_id": video_id,
                "example_id": example_id,
                "label": bool(example.get("label", False)),
                "current_segment_id": _safe_text(example.get("current_segment_id", "")),
                "body_atoms": list(example.get("body_atoms", [])),
            }
    return lookup


def _candidate_object_lookup(
    logic_atom_results: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for video_result in logic_atom_results:
        video_id = _safe_text(video_result.get("video_id", ""))
        for segment in list(video_result.get("segments", [])):
            segment_id = _safe_text(segment.get("segment_id", ""))
            for obj in list(segment.get("candidate_objects", [])):
                object_id = _safe_text(obj.get("object_id", ""))
                if not video_id or not segment_id or not object_id:
                    continue
                lookup[(video_id, segment_id, object_id)] = {
                    "video_id": video_id,
                    "segment_id": segment_id,
                    "object_id": object_id,
                    "candidate_track_id": _safe_int(obj.get("candidate_track_id", -1), -1),
                    "candidate_object_id": _safe_text(obj.get("candidate_object_id", "")),
                    "object_class": _safe_text(obj.get("object_class", "unknown")) or "unknown",
                    "frame_detection_id": _safe_text(obj.get("frame_detection_id", "")),
                    "source_detection_ids": [
                        _safe_text(value)
                        for value in list(obj.get("source_detection_ids", []))
                        if _safe_text(value)
                    ],
                    "candidate_source": _safe_text(obj.get("candidate_source", "")),
                    "selection_score": _safe_float(obj.get("selection_score", 0.0), 0.0),
                    "prior_metadata": dict(obj.get("prior_metadata", {})),
                    "track_quality": dict(obj.get("track_quality", {})),
                }
    return lookup


def _selected_candidate_detection_ids(
    tracking_results: Sequence[Dict[str, Any]],
) -> Set[str]:
    detection_ids: Set[str] = set()
    for video_result in tracking_results:
        track_summaries = (
            video_result.get("candidate_tracks", {})
            .get("selected_candidate_tracks", {})
            .get("track_summaries", [])
        )
        for summary in list(track_summaries):
            for detection_id in list(summary.get("source_detection_ids", [])):
                detection_text = _safe_text(detection_id)
                if detection_text:
                    detection_ids.add(detection_text)
    return detection_ids


def _candidate_object_ids_from_matched_atoms(matched_atoms: Dict[str, Any]) -> Set[str]:
    object_ids: Set[str] = set()
    for concrete_atom in matched_atoms.values():
        parsed = _parse_atom(_safe_text(concrete_atom))
        if parsed is None:
            continue
        _predicate, args = parsed
        for arg in args:
            arg_text = _safe_text(arg)
            if arg_text.startswith("obj_candidate_"):
                object_ids.add(arg_text)
    return object_ids


def _rule_feedback_profile(rule: Dict[str, Any]) -> Dict[str, Any]:
    has_strong_semantics = bool(rule.get("has_strong_candidate_semantic_atom", False))
    broad_pattern = _safe_text(rule.get("broad_candidate_rule_pattern", ""))
    fn_gain_count = _safe_int(rule.get("eval_fn_coverage_gain_count_vs_accepted_only", 0), 0)
    fp_contribution_count = _safe_int(rule.get("eval_fp_contribution_count_vs_accepted_only", 0), 0)
    positive_rule = has_strong_semantics and fn_gain_count > 0 and fp_contribution_count == 0
    broad_or_provenance_only = bool(rule.get("is_broad_candidate_rule", False)) or (
        broad_pattern in _POSITION_OR_PROVENANCE_ONLY_PATTERNS
    )
    negative_rule = fp_contribution_count > 0 or (broad_or_provenance_only and fn_gain_count <= 0)
    return {
        "positive_rule": positive_rule,
        "negative_rule": negative_rule,
        "has_strong_semantics": has_strong_semantics,
        "broad_or_provenance_only": broad_or_provenance_only,
        "broad_pattern": broad_pattern,
        "fn_gain_example_ids": {
            _safe_text(value)
            for value in list(rule.get("eval_fn_coverage_gain_example_ids_vs_accepted_only", []))
            if _safe_text(value)
        },
        "fp_example_ids": {
            _safe_text(value)
            for value in list(rule.get("eval_fp_contribution_example_ids_vs_accepted_only", []))
            if _safe_text(value)
        },
    }


def _evidence_label(
    *,
    example_id: str,
    example_label: bool,
    rule_profile: Dict[str, Any],
) -> Tuple[str, str]:
    if example_id in set(rule_profile.get("fn_gain_example_ids", set())) and bool(rule_profile.get("positive_rule", False)):
        return "positive", "semantic_fn_recovery_without_fp"
    if example_id in set(rule_profile.get("fp_example_ids", set())):
        return "negative", "introduced_false_positive"
    if bool(rule_profile.get("negative_rule", False)):
        if not example_label:
            return "negative", "noisy_or_broad_rule_on_negative_example"
        return "neutral", "broad_or_weak_rule_without_clear_fn_gain"
    return "neutral", "candidate_rule_without_direct_task_utility_change"


def _update_aggregate(
    aggregate: Dict[str, Any],
    *,
    selector_name: str,
    rule: Dict[str, Any],
    example: Dict[str, Any],
    label: str,
    reason: str,
) -> None:
    aggregate.setdefault("selectors", set()).add(selector_name)
    aggregate.setdefault("rule_ids", set()).add(_safe_text(rule.get("rule_id", "")))
    aggregate.setdefault("reasons", Counter()).update([reason])
    aggregate.setdefault("example_ids", set()).add(_safe_text(example.get("example_id", "")))
    if label == "positive":
        aggregate["positive_count"] = _safe_int(aggregate.get("positive_count", 0), 0) + 1
        aggregate.setdefault("recovered_fn_example_ids", set()).add(_safe_text(example.get("example_id", "")))
    elif label == "negative":
        aggregate["negative_count"] = _safe_int(aggregate.get("negative_count", 0), 0) + 1
        if not bool(example.get("label", False)):
            aggregate.setdefault("introduced_fp_example_ids", set()).add(_safe_text(example.get("example_id", "")))
    else:
        aggregate["neutral_count"] = _safe_int(aggregate.get("neutral_count", 0), 0) + 1


def _finalize_pseudo_label_row(
    key: str,
    aggregate: Dict[str, Any],
    detection_lookup: Dict[str, Dict[str, Any]],
    *,
    row_type: str,
) -> Dict[str, Any]:
    positive_count = _safe_int(aggregate.get("positive_count", 0), 0)
    negative_count = _safe_int(aggregate.get("negative_count", 0), 0)
    neutral_count = _safe_int(aggregate.get("neutral_count", 0), 0)
    recovered_fn_example_ids = sorted(
        _safe_text(value) for value in aggregate.get("recovered_fn_example_ids", set()) if _safe_text(value)
    )
    introduced_fp_example_ids = sorted(
        _safe_text(value) for value in aggregate.get("introduced_fp_example_ids", set()) if _safe_text(value)
    )
    if positive_count > 0 and negative_count == 0 and not introduced_fp_example_ids:
        pseudo_label = "positive"
    elif negative_count > 0 and positive_count == 0:
        pseudo_label = "negative"
    elif positive_count > 0 and negative_count > positive_count:
        pseudo_label = "negative"
    else:
        pseudo_label = "neutral"

    pseudo_label_score = float(positive_count - negative_count)
    first_detection = dict(detection_lookup.get(key, {}))
    return {
        f"{row_type}_id": key,
        "pseudo_label": pseudo_label,
        "pseudo_label_score": pseudo_label_score,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "selectors": sorted(_safe_text(value) for value in aggregate.get("selectors", set()) if _safe_text(value)),
        "rule_ids": sorted(_safe_text(value) for value in aggregate.get("rule_ids", set()) if _safe_text(value)),
        "reason_counts": dict(sorted(dict(aggregate.get("reasons", Counter())).items())),
        "example_ids": sorted(_safe_text(value) for value in aggregate.get("example_ids", set()) if _safe_text(value)),
        "recovered_fn_example_ids": recovered_fn_example_ids,
        "introduced_fp_example_ids": introduced_fp_example_ids,
        "class_name": _safe_text(first_detection.get("class", first_detection.get("label", aggregate.get("object_class", "unknown"))))
        or _safe_text(aggregate.get("object_class", "unknown")),
        "candidate_source": _safe_text(first_detection.get("candidate_source", aggregate.get("candidate_source", ""))),
        "raw_score": _safe_float(first_detection.get("raw_score", first_detection.get("score", 0.0)), 0.0),
        "calibrated_score": _safe_float(
            first_detection.get("calibrated_score", first_detection.get("raw_score", first_detection.get("score", 0.0))),
            0.0,
        ),
        "matched_prior_ids": od_calibration_policy_utils.matched_prior_ids_from_detection(first_detection)
        if first_detection
        else list(aggregate.get("matched_prior_ids", [])),
    }


def process(
    detection_results: Sequence[Dict[str, Any]],
    tracking_results: Sequence[Dict[str, Any]],
    logic_atom_results: Sequence[Dict[str, Any]],
    eval_temporal_rule_results: Sequence[Dict[str, Any]],
    evaluation_results_by_name: Dict[str, Dict[str, Any]],
    candidate_contribution_summary_results: Dict[str, Any],
    primary_rule_set: str,
    *,
    iteration_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = od_calibration_policy_utils.get_iteration_root(iteration_id)
    if output_root is not None:
        out_root = Path(output_root) / iteration_id
        out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "reasoning_to_od_pseudo_labels.json"
    detection_csv_path = out_root / "reasoning_to_od_detection_pseudo_labels.csv"
    object_csv_path = out_root / "reasoning_to_od_object_pseudo_labels.csv"
    track_csv_path = out_root / "reasoning_to_od_track_pseudo_labels.csv"
    contribution_report_path = out_root / "low_confidence_od_contribution_report.json"
    feedback_summary_path = out_root / "candidate_feedback_summary.json"

    if not force_recompute and json_path.exists():
        cached = od_calibration_policy_utils.load_json(json_path, default={})
        if int(cached.get("version", 0)) == _PSEUDO_LABEL_VERSION and _cfg_key_subset(cached.get("config", {})) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {json_path.name}")
            return cached

    detection_lookup = od_calibration_policy_utils.build_detection_lookup(detection_results)
    example_lookup = _example_lookup(eval_temporal_rule_results)
    object_lookup = _candidate_object_lookup(logic_atom_results)
    selector_names = _selector_names_to_trace(
        primary_rule_set,
        evaluation_results_by_name,
        candidate_contribution_summary_results,
        cfg,
    )
    max_match_states = max(1, int(cfg.get("max_match_states_per_example_rule", 12)))

    object_aggregates: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    detection_aggregates: Dict[str, Dict[str, Any]] = {}
    track_aggregates: Dict[str, Dict[str, Any]] = {}

    for selector_name in selector_names:
        evaluation_result = dict(evaluation_results_by_name.get(selector_name, {}))
        for rule in list(evaluation_result.get("rule_evaluations", [])):
            if not bool(rule.get("uses_candidate_atoms", False)):
                continue
            rule_profile = _rule_feedback_profile(rule)
            triggered_example_ids = [
                _safe_text(example_id)
                for example_id in list(rule.get("eval_triggered_example_ids", []))
                if _safe_text(example_id)
            ]
            for example_id in triggered_example_ids:
                example = dict(example_lookup.get(example_id, {}))
                if not example:
                    continue
                match_states = _find_rule_matches_for_example(
                    body_atom_templates=_get_rule_body_atom_templates(rule),
                    body_atoms=list(example.get("body_atoms", [])),
                )
                if not match_states:
                    continue
                label, reason = _evidence_label(
                    example_id=example_id,
                    example_label=bool(example.get("label", False)),
                    rule_profile=rule_profile,
                )
                for match_state in match_states[:max_match_states]:
                    candidate_object_ids = _candidate_object_ids_from_matched_atoms(
                        dict(match_state.get("matched_atoms", {}))
                    )
                    for object_id in candidate_object_ids:
                        object_key = (
                            _safe_text(example.get("video_id", "")),
                            _safe_text(example.get("current_segment_id", "")),
                            object_id,
                        )
                        object_meta = dict(object_lookup.get(object_key, {}))
                        object_aggregate = object_aggregates.setdefault(
                            object_key,
                            {
                                "object_class": _safe_text(object_meta.get("object_class", "unknown")) or "unknown",
                                "candidate_source": _safe_text(object_meta.get("candidate_source", "")),
                                "matched_prior_ids": set(
                                    od_calibration_policy_utils.matched_prior_ids_from_detection(object_meta)
                                ),
                                "positive_count": 0,
                                "negative_count": 0,
                                "neutral_count": 0,
                            },
                        )
                        _update_aggregate(
                            object_aggregate,
                            selector_name=selector_name,
                            rule=rule,
                            example=example,
                            label=label,
                            reason=reason,
                        )
                        detection_ids = [
                            _safe_text(value)
                            for value in object_meta.get("source_detection_ids", [])
                            if _safe_text(value)
                        ]
                        if not detection_ids and _safe_text(object_meta.get("frame_detection_id", "")):
                            detection_ids = [_safe_text(object_meta.get("frame_detection_id", ""))]
                        for detection_id in detection_ids:
                            detection_aggregate = detection_aggregates.setdefault(
                                detection_id,
                                {
                                    "object_class": _safe_text(object_meta.get("object_class", "unknown")) or "unknown",
                                    "candidate_source": _safe_text(object_meta.get("candidate_source", "")),
                                    "matched_prior_ids": set(
                                        od_calibration_policy_utils.matched_prior_ids_from_detection(object_meta)
                                    ),
                                    "candidate_track_ids": set(),
                                    "positive_count": 0,
                                    "negative_count": 0,
                                    "neutral_count": 0,
                                },
                            )
                            if _safe_int(object_meta.get("candidate_track_id", -1), -1) >= 0:
                                detection_aggregate.setdefault("candidate_track_ids", set()).add(
                                    _safe_text(object_meta.get("candidate_track_id", ""))
                                )
                            _update_aggregate(
                                detection_aggregate,
                                selector_name=selector_name,
                                rule=rule,
                                example=example,
                                label=label,
                                reason=reason,
                            )
                            track_id_text = _safe_text(object_meta.get("candidate_track_id", ""))
                            if track_id_text:
                                track_aggregate = track_aggregates.setdefault(
                                    track_id_text,
                                    {
                                        "object_class": _safe_text(object_meta.get("object_class", "unknown")) or "unknown",
                                        "candidate_source": _safe_text(object_meta.get("candidate_source", "")),
                                        "matched_prior_ids": set(
                                            od_calibration_policy_utils.matched_prior_ids_from_detection(object_meta)
                                        ),
                                        "positive_count": 0,
                                        "negative_count": 0,
                                        "neutral_count": 0,
                                    },
                                )
                                _update_aggregate(
                                    track_aggregate,
                                    selector_name=selector_name,
                                    rule=rule,
                                    example=example,
                                    label=label,
                                    reason=reason,
                                )

    if bool(cfg.get("include_neutral_selected_candidate_detections", True)):
        for detection_id in _selected_candidate_detection_ids(tracking_results):
            detection_aggregates.setdefault(
                detection_id,
                {
                    "object_class": _safe_text(detection_lookup.get(detection_id, {}).get("class", "unknown")) or "unknown",
                    "candidate_source": _safe_text(detection_lookup.get(detection_id, {}).get("candidate_source", "")),
                    "matched_prior_ids": set(
                        od_calibration_policy_utils.matched_prior_ids_from_detection(detection_lookup.get(detection_id, {}))
                    ),
                    "candidate_track_ids": set(),
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 1,
                    "selectors": set(),
                    "rule_ids": set(),
                    "reasons": Counter({"no_direct_reasoning_signal": 1}),
                    "example_ids": set(),
                    "recovered_fn_example_ids": set(),
                    "introduced_fp_example_ids": set(),
                },
            )

    detection_rows = [
        _finalize_pseudo_label_row(detection_id, aggregate, detection_lookup, row_type="detection")
        for detection_id, aggregate in sorted(detection_aggregates.items())
    ]
    object_rows = []
    for object_key, aggregate in sorted(object_aggregates.items()):
        key_text = f"{object_key[0]}::{object_key[1]}::{object_key[2]}"
        object_rows.append(
            {
                **_finalize_pseudo_label_row("", aggregate, {}, row_type="object"),
                "object_id": object_key[2],
                "video_id": object_key[0],
                "segment_id": object_key[1],
                "object_key": key_text,
            }
        )
    track_rows = [
        {
            **_finalize_pseudo_label_row(track_id, aggregate, {}, row_type="track"),
            "track_id": track_id,
        }
        for track_id, aggregate in sorted(track_aggregates.items())
    ]

    label_counts = Counter(row.get("pseudo_label", "neutral") for row in detection_rows)
    contribution_by_class: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())
    contribution_by_source: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())
    for row in detection_rows:
        contribution_by_class[_safe_text(row.get("class_name", "unknown"))][row.get("pseudo_label", "neutral")] += 1
        contribution_by_source[_safe_text(row.get("candidate_source", "unknown"))][row.get("pseudo_label", "neutral")] += 1

    contribution_report = {
        "num_detection_pseudo_labels": len(detection_rows),
        "num_positive_detection_pseudo_labels": int(label_counts.get("positive", 0)),
        "num_negative_detection_pseudo_labels": int(label_counts.get("negative", 0)),
        "num_neutral_detection_pseudo_labels": int(label_counts.get("neutral", 0)),
        "by_class": {
            key: dict(sorted(dict(counter).items()))
            for key, counter in sorted(contribution_by_class.items())
        },
        "by_candidate_source": {
            key: dict(sorted(dict(counter).items()))
            for key, counter in sorted(contribution_by_source.items())
        },
    }
    feedback_summary = {
        "selector_names_used": selector_names,
        "top_positive_detection_ids": [
            row.get("detection_id", "")
            for row in sorted(
                detection_rows,
                key=lambda row: (
                    row.get("pseudo_label", "") != "positive",
                    -_safe_float(row.get("pseudo_label_score", 0.0), 0.0),
                    row.get("detection_id", ""),
                ),
            )[:10]
        ],
        "top_negative_detection_ids": [
            row.get("detection_id", "")
            for row in sorted(
                detection_rows,
                key=lambda row: (
                    row.get("pseudo_label", "") != "negative",
                    _safe_float(row.get("pseudo_label_score", 0.0), 0.0),
                    row.get("detection_id", ""),
                ),
            )[:10]
        ],
    }

    result = {
        "version": _PSEUDO_LABEL_VERSION,
        "iteration_id": str(iteration_id),
        "config": _cfg_key_subset(cfg),
        "selector_names_used": selector_names,
        "detection_pseudo_labels": detection_rows,
        "object_pseudo_labels": object_rows,
        "track_pseudo_labels": track_rows,
        "low_confidence_od_contribution_report": contribution_report,
        "candidate_feedback_summary": feedback_summary,
        "output_paths": {
            "json": str(json_path),
            "detection_csv": str(detection_csv_path),
            "object_csv": str(object_csv_path),
            "track_csv": str(track_csv_path),
            "low_confidence_od_contribution_report_json": str(contribution_report_path),
            "candidate_feedback_summary_json": str(feedback_summary_path),
        },
    }

    od_calibration_policy_utils.save_json_atomic(json_path, result)
    od_calibration_policy_utils.save_json_atomic(contribution_report_path, contribution_report)
    od_calibration_policy_utils.save_json_atomic(feedback_summary_path, feedback_summary)
    _write_csv(
        detection_csv_path,
        [
            "detection_id",
            "pseudo_label",
            "pseudo_label_score",
            "positive_count",
            "negative_count",
            "neutral_count",
            "class_name",
            "candidate_source",
            "raw_score",
            "calibrated_score",
            "selectors",
            "rule_ids",
            "reason_counts",
            "example_ids",
            "recovered_fn_example_ids",
            "introduced_fp_example_ids",
            "matched_prior_ids",
        ],
        detection_rows,
    )
    _write_csv(
        object_csv_path,
        [
            "object_key",
            "video_id",
            "segment_id",
            "object_id",
            "pseudo_label",
            "pseudo_label_score",
            "positive_count",
            "negative_count",
            "neutral_count",
            "class_name",
            "candidate_source",
            "selectors",
            "rule_ids",
            "reason_counts",
            "example_ids",
            "recovered_fn_example_ids",
            "introduced_fp_example_ids",
        ],
        object_rows,
    )
    _write_csv(
        track_csv_path,
        [
            "track_id",
            "pseudo_label",
            "pseudo_label_score",
            "positive_count",
            "negative_count",
            "neutral_count",
            "class_name",
            "candidate_source",
            "selectors",
            "rule_ids",
            "reason_counts",
            "example_ids",
            "recovered_fn_example_ids",
            "introduced_fp_example_ids",
        ],
        track_rows,
    )

    print(
        "  reasoning_to_od_pseudo_labels: "
        f"selectors={len(selector_names)} | "
        f"detections={len(detection_rows)} | "
        f"positive={int(label_counts.get('positive', 0))} | "
        f"negative={int(label_counts.get('negative', 0))}"
    )
    print(f"Reasoning-to-OD pseudo labels JSON written to {json_path}")
    return result


def run(
    detection_results: Sequence[Dict[str, Any]],
    tracking_results: Sequence[Dict[str, Any]],
    logic_atom_results: Sequence[Dict[str, Any]],
    eval_temporal_rule_results: Sequence[Dict[str, Any]],
    evaluation_results_by_name: Dict[str, Dict[str, Any]],
    candidate_contribution_summary_results: Dict[str, Any],
    primary_rule_set: str,
    *,
    iteration_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process(
        detection_results=detection_results,
        tracking_results=tracking_results,
        logic_atom_results=logic_atom_results,
        eval_temporal_rule_results=eval_temporal_rule_results,
        evaluation_results_by_name=evaluation_results_by_name,
        candidate_contribution_summary_results=candidate_contribution_summary_results,
        primary_rule_set=primary_rule_set,
        iteration_id=iteration_id,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
