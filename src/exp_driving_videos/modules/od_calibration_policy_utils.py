from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import config


_OD_CALIBRATION_POLICY_VERSION = 1
_OD_CALIBRATION_STATE_VERSION = 1

_NUMERIC_FEATURE_NAMES: Tuple[str, ...] = (
    "raw_score",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_aspect_ratio",
    "bbox_center_x",
    "bbox_center_y",
    "prior_relevance_score",
    "matched_prior_id_count",
    "has_matched_prior",
    "track_length",
    "track_temporal_consistency",
    "track_selection_score",
    "track_mean_score",
    "track_max_score",
    "track_prior_relevance_mean",
    "reasoning_feedback_score",
)


def get_od_calibration_loop_root() -> Path:
    root = config.get_output_path("pipeline_output") / "18hij_driving_mini_od_calibration_loop"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_od_calibration_iterations_root() -> Path:
    root = get_od_calibration_loop_root() / "iterations"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_active_od_calibration_policy_path() -> Path:
    return get_od_calibration_loop_root() / "active_policy.json"


def get_active_od_calibration_state_path() -> Path:
    return get_od_calibration_loop_root() / "active_policy_state.json"


def get_iteration_root(iteration_id: str) -> Path:
    root = get_od_calibration_iterations_root() / str(iteration_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def next_iteration_id() -> str:
    existing_numbers: List[int] = []
    for child in get_od_calibration_iterations_root().iterdir():
        if not child.is_dir():
            continue
        parts = str(child.name).split("_")
        if len(parts) != 2 or parts[0] != "iteration":
            continue
        try:
            existing_numbers.append(int(parts[1]))
        except ValueError:
            continue
    next_number = max(existing_numbers, default=0) + 1
    return f"iteration_{next_number:04d}"


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


def load_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists() or path.is_dir():
        return dict(default or {})
    try:
        with path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except Exception:
        return dict(default or {})
    return dict(loaded) if isinstance(loaded, dict) else dict(default or {})


def save_json_atomic(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    tmp_path.replace(path)
    return path


def load_active_od_calibration_policy() -> Dict[str, Any]:
    payload = load_json(get_active_od_calibration_policy_path(), default={})
    if int(payload.get("version", 0)) != _OD_CALIBRATION_POLICY_VERSION:
        return {}
    return payload


def load_active_od_calibration_state() -> Dict[str, Any]:
    payload = load_json(get_active_od_calibration_state_path(), default={})
    if int(payload.get("version", 0)) != _OD_CALIBRATION_STATE_VERSION:
        return {}
    return payload


def policy_id(policy: Optional[Dict[str, Any]]) -> str:
    if not isinstance(policy, dict):
        return ""
    return _safe_text(policy.get("policy_id", ""))


def top_prior_id_from_detection(detection: Dict[str, Any]) -> str:
    prior_metadata = dict(detection.get("prior_metadata", {}))
    matched_prior_ids = [
        _safe_text(value)
        for value in list(prior_metadata.get("matched_prior_ids", []))
        if _safe_text(value)
    ]
    if matched_prior_ids:
        return matched_prior_ids[0]
    matched_track_prior_ids = [
        _safe_text(value)
        for value in list(prior_metadata.get("track_matched_prior_ids", []))
        if _safe_text(value)
    ]
    return matched_track_prior_ids[0] if matched_track_prior_ids else ""


def matched_prior_ids_from_detection(detection: Dict[str, Any]) -> List[str]:
    prior_metadata = dict(detection.get("prior_metadata", {}))
    values = {
        _safe_text(value)
        for value in list(prior_metadata.get("matched_prior_ids", []))
        + list(prior_metadata.get("track_matched_prior_ids", []))
        if _safe_text(value)
    }
    return sorted(values)


def bbox_features(bbox: Sequence[Any]) -> Dict[str, float]:
    if len(bbox) < 4:
        return {
            "bbox_width": 0.0,
            "bbox_height": 0.0,
            "bbox_area": 0.0,
            "bbox_aspect_ratio": 0.0,
            "bbox_center_x": 0.0,
            "bbox_center_y": 0.0,
        }
    x1, y1, x2, y2 = [_safe_float(value, 0.0) for value in bbox[:4]]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": width * height,
        "bbox_aspect_ratio": float(width / max(height, 1e-6)),
        "bbox_center_x": float((x1 + x2) * 0.5),
        "bbox_center_y": float((y1 + y2) * 0.5),
    }


def build_detection_lookup(
    detection_results: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for video_result in detection_results:
        video_id = _safe_text(video_result.get("video_id", ""))
        for frame in list(video_result.get("frames", [])):
            frame_index = _safe_int(frame.get("frame_index", -1), -1)
            image_path = _safe_text(frame.get("image_path", ""))
            for collection_name in ("accepted_detections", "candidate_detections"):
                for det in list(frame.get(collection_name, [])):
                    detection_id = _safe_text(det.get("detection_id", ""))
                    if not detection_id:
                        continue
                    entry = dict(det)
                    entry.setdefault("video_id", video_id)
                    entry.setdefault("frame_index", frame_index)
                    entry.setdefault("image_path", image_path)
                    entry.setdefault("accepted", collection_name == "accepted_detections")
                    lookup[detection_id] = entry
    return lookup


def build_candidate_track_feature_lookup(
    tracking_results: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for video_result in tracking_results:
        track_summaries = (
            video_result.get("candidate_tracks", {})
            .get("selected_candidate_tracks", {})
            .get("track_summaries", [])
        )
        for summary in list(track_summaries):
            features = {
                "candidate_track_id": _safe_int(summary.get("track_id", -1), -1),
                "track_length": _safe_float(summary.get("track_length", 0.0), 0.0),
                "track_temporal_consistency": _safe_float(summary.get("temporal_consistency", 0.0), 0.0),
                "track_selection_score": _safe_float(summary.get("selection_score", 0.0), 0.0),
                "track_mean_score": _safe_float(summary.get("mean_score", 0.0), 0.0),
                "track_max_score": _safe_float(summary.get("max_score", 0.0), 0.0),
                "track_prior_relevance_mean": _safe_float(summary.get("prior_relevance_mean", 0.0), 0.0),
            }
            for detection_id in list(summary.get("source_detection_ids", [])):
                detection_text = _safe_text(detection_id)
                if detection_text:
                    lookup[detection_text] = dict(features)
    return lookup


def extract_detection_calibration_features(
    detection: Dict[str, Any],
    *,
    track_features: Optional[Dict[str, Any]] = None,
    reasoning_feedback_score: float = 0.0,
) -> Dict[str, Any]:
    raw_score = _safe_float(detection.get("raw_score", detection.get("score", 0.0)), 0.0)
    prior_metadata = dict(detection.get("prior_metadata", {}))
    matched_prior_ids = matched_prior_ids_from_detection(detection)
    top_prior_id = top_prior_id_from_detection(detection)
    bbox = bbox_features(list(detection.get("bbox", [])))
    track_features = dict(track_features or {})
    features: Dict[str, Any] = {
        "raw_score": raw_score,
        **bbox,
        "prior_relevance_score": _safe_float(prior_metadata.get("prior_relevance_score", 0.0), 0.0),
        "matched_prior_id_count": float(len(matched_prior_ids)),
        "has_matched_prior": float(1.0 if matched_prior_ids else 0.0),
        "track_length": _safe_float(track_features.get("track_length", 0.0), 0.0),
        "track_temporal_consistency": _safe_float(track_features.get("track_temporal_consistency", 0.0), 0.0),
        "track_selection_score": _safe_float(track_features.get("track_selection_score", 0.0), 0.0),
        "track_mean_score": _safe_float(track_features.get("track_mean_score", 0.0), 0.0),
        "track_max_score": _safe_float(track_features.get("track_max_score", 0.0), 0.0),
        "track_prior_relevance_mean": _safe_float(track_features.get("track_prior_relevance_mean", 0.0), 0.0),
        "reasoning_feedback_score": _safe_float(reasoning_feedback_score, 0.0),
        "class_name": _safe_text(detection.get("class", detection.get("label", "unknown"))) or "unknown",
        "candidate_source": _safe_text(detection.get("candidate_source", "")) or "candidate",
        "top_prior_id": top_prior_id or "none",
    }
    return features


def numeric_feature_names() -> List[str]:
    return list(_NUMERIC_FEATURE_NAMES)


def _sigmoid(value: float) -> float:
    clipped = max(-50.0, min(50.0, float(value)))
    return float(1.0 / (1.0 + math.exp(-clipped)))


def _standardize_numeric(value: float, mean: float, scale: float) -> float:
    safe_scale = float(scale) if abs(float(scale)) > 1e-9 else 1.0
    return float((float(value) - float(mean)) / safe_scale)


def policy_ready(policy: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(policy, dict):
        return False
    if int(policy.get("version", 0)) != _OD_CALIBRATION_POLICY_VERSION:
        return False
    return bool(policy_id(policy))


def calibrate_candidate_detection(
    detection: Dict[str, Any],
    policy: Optional[Dict[str, Any]],
    *,
    track_features: Optional[Dict[str, Any]] = None,
    reasoning_feedback_score: float = 0.0,
) -> Dict[str, Any]:
    enriched = dict(detection)
    raw_score = _safe_float(detection.get("raw_score", detection.get("score", 0.0)), 0.0)
    enriched["raw_score"] = raw_score
    if bool(enriched.get("accepted", False)):
        enriched["calibrated_score"] = raw_score
        enriched["feedback_bonus"] = 0.0
        enriched["score_used_for_candidate_ranking"] = raw_score
        enriched["od_calibration"] = {
            "policy_id": policy_id(policy),
            "policy_applied": False,
            "calibration_branch": "protected_accepted_baseline",
        }
        return enriched

    if not policy_ready(policy):
        enriched["calibrated_score"] = raw_score
        enriched["feedback_bonus"] = 0.0
        enriched["score_used_for_candidate_ranking"] = raw_score
        enriched["od_calibration"] = {
            "policy_id": "",
            "policy_applied": False,
            "calibration_branch": "candidate_exploration",
        }
        return enriched

    features = extract_detection_calibration_features(
        enriched,
        track_features=track_features,
        reasoning_feedback_score=reasoning_feedback_score,
    )
    normalization = dict(policy.get("normalization", {}))
    means = dict(normalization.get("means", {}))
    scales = dict(normalization.get("scales", {}))
    model = dict(policy.get("model", {}))
    feature_coefficients = dict(model.get("feature_coefficients", {}))
    logit = _safe_float(model.get("intercept", 0.0), 0.0)
    for feature_name in _NUMERIC_FEATURE_NAMES:
        standardized = _standardize_numeric(
            _safe_float(features.get(feature_name, 0.0), 0.0),
            _safe_float(means.get(feature_name, 0.0), 0.0),
            _safe_float(scales.get(feature_name, 1.0), 1.0),
        )
        logit += _safe_float(feature_coefficients.get(f"num:{feature_name}", 0.0), 0.0) * standardized
    for categorical_name in ("class_name", "candidate_source", "top_prior_id"):
        categorical_value = _safe_text(features.get(categorical_name, "")) or "none"
        logit += _safe_float(
            feature_coefficients.get(f"cat:{categorical_name}={categorical_value}", 0.0),
            0.0,
        )

    heuristics = dict(policy.get("heuristics", {}))
    heuristic_bonus = (
        _safe_float(heuristics.get("global_bias", 0.0), 0.0)
        + _safe_float(dict(heuristics.get("class_bonus", {})).get(_safe_text(features.get("class_name", "")), 0.0), 0.0)
        + _safe_float(
            dict(heuristics.get("candidate_source_bonus", {})).get(
                _safe_text(features.get("candidate_source", "")),
                0.0,
            ),
            0.0,
        )
        + _safe_float(
            dict(heuristics.get("top_prior_bonus", {})).get(
                _safe_text(features.get("top_prior_id", "")),
                0.0,
            ),
            0.0,
        )
    )
    policy_type = _safe_text(policy.get("policy_type", "logistic_detection_calibrator")) or "logistic_detection_calibrator"
    if policy_type == "heuristic_additive":
        calibrated_score = raw_score + heuristic_bonus
    else:
        calibrated_score = _sigmoid(logit) + heuristic_bonus
    calibrated_score = max(0.0, min(1.0, float(calibrated_score)))
    feedback_bonus = float(calibrated_score - raw_score)
    enriched["calibrated_score"] = calibrated_score
    enriched["feedback_bonus"] = feedback_bonus
    enriched["score_used_for_candidate_ranking"] = float(
        max(0.0, min(1.0, calibrated_score + _safe_float(policy.get("ranking_bonus_bias", 0.0), 0.0)))
    )
    enriched["od_calibration"] = {
        "policy_id": policy_id(policy),
        "policy_applied": True,
        "calibration_branch": "candidate_exploration",
        "reasoning_feedback_score": _safe_float(reasoning_feedback_score, 0.0),
    }
    return enriched


def apply_policy_to_frame_record(
    frame_record: Dict[str, Any],
    policy: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    updated = dict(frame_record)
    updated["accepted_detections"] = [
        calibrate_candidate_detection(det, policy)
        for det in list(frame_record.get("accepted_detections", []))
    ]
    updated["candidate_detections"] = [
        calibrate_candidate_detection(det, policy)
        for det in list(frame_record.get("candidate_detections", []))
    ]
    return updated


def current_policy_marker(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "policy_id": policy_id(policy),
        "policy_version": int(policy.get("version", 0)) if isinstance(policy, dict) else 0,
        "policy_available": bool(policy_ready(policy)),
    }


def write_active_policy_state(
    *,
    active_policy: Dict[str, Any],
    latest_iteration_id: str,
    accepted_reference_metrics: Dict[str, Any],
    gate_decision: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "version": _OD_CALIBRATION_STATE_VERSION,
        "active_policy_id": policy_id(active_policy),
        "active_policy_path": str(get_active_od_calibration_policy_path()),
        "latest_iteration_id": str(latest_iteration_id),
        "accepted_reference_metrics": dict(accepted_reference_metrics),
        "gate_decision": dict(gate_decision),
    }
    save_json_atomic(get_active_od_calibration_state_path(), payload)
    return payload


def validate_iteration_artifacts(
    iteration_id: str,
    *,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    iteration_root = get_iteration_root(iteration_id)
    if output_root is not None:
        iteration_root = Path(output_root) / str(iteration_id)

    required_json_fields = {
        "reasoning_to_od_pseudo_labels.json": [
            "version",
            "iteration_id",
            "config",
            "selector_names_used",
            "detection_pseudo_labels",
            "object_pseudo_labels",
            "track_pseudo_labels",
            "low_confidence_od_contribution_report",
            "candidate_feedback_summary",
            "output_paths",
        ],
        "low_confidence_od_contribution_report.json": [
            "num_detection_pseudo_labels",
            "num_positive_detection_pseudo_labels",
            "num_negative_detection_pseudo_labels",
            "num_neutral_detection_pseudo_labels",
            "by_class",
            "by_candidate_source",
        ],
        "candidate_feedback_summary.json": [
            "selector_names_used",
            "top_positive_detection_ids",
            "top_negative_detection_ids",
        ],
        "od_confidence_calibration.json": [
            "version",
            "iteration_id",
            "config",
            "num_training_rows",
            "num_labeled_rows",
            "policy",
            "training_summary",
            "output_paths",
        ],
        "proposed_od_calibration_policy.json": [
            "version",
            "policy_id",
            "source_iteration_id",
            "policy_type",
            "normalization",
            "model",
            "heuristics",
            "feature_schema",
            "training_summary",
        ],
        "baseline_preservation_audit.json": [
            "version",
            "iteration_id",
            "config",
            "reference_metrics",
            "current_metrics",
            "same_run_audit",
        ],
        "calibration_gate_decision.json": [
            "version",
            "iteration_id",
            "config",
            "decision",
            "decision_reason",
            "proposed_policy_id",
            "active_policy_after_id",
            "audit_path",
            "active_state_after",
        ],
    }
    required_csv_fields = {
        "reasoning_to_od_detection_pseudo_labels.csv": [
            "detection_id",
            "pseudo_label",
            "pseudo_label_score",
            "class_name",
            "candidate_source",
            "raw_score",
            "calibrated_score",
        ],
        "reasoning_to_od_object_pseudo_labels.csv": [
            "object_key",
            "video_id",
            "segment_id",
            "object_id",
            "pseudo_label",
        ],
        "reasoning_to_od_track_pseudo_labels.csv": [
            "track_id",
            "pseudo_label",
            "pseudo_label_score",
        ],
        "od_calibration_training_rows.csv": [
            "detection_id",
            "label_name",
            "sample_weight",
            "class_name",
            "candidate_source",
            "top_prior_id",
            "raw_score",
            "reasoning_feedback_score",
        ],
    }

    errors: List[str] = []
    verified_json: Dict[str, Any] = {}
    verified_csv: Dict[str, Any] = {}

    for filename, field_names in required_json_fields.items():
        path = iteration_root / filename
        payload = load_json(path, default={})
        if not path.exists():
            errors.append(f"missing_json:{filename}")
            continue
        missing = [field for field in field_names if field not in payload]
        if missing:
            errors.append(f"json_missing_fields:{filename}:{','.join(missing)}")
            continue
        verified_json[filename] = {"path": str(path), "fields": list(field_names)}

    for filename, field_names in required_csv_fields.items():
        path = iteration_root / filename
        if not path.exists():
            errors.append(f"missing_csv:{filename}")
            continue
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                header = []
        missing = [field for field in field_names if field not in header]
        if missing:
            errors.append(f"csv_missing_fields:{filename}:{','.join(missing)}")
            continue
        verified_csv[filename] = {"path": str(path), "fields": list(field_names)}

    return {
        "iteration_id": str(iteration_id),
        "iteration_root": str(iteration_root),
        "ok": not errors,
        "errors": errors,
        "verified_json": verified_json,
        "verified_csv": verified_csv,
    }
