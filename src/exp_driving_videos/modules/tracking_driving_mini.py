"""
Prior-guided candidate-aware multi-object tracking over driving_mini detections.

Accepted detections keep the existing ByteTrack-based tracking behavior.
Candidate detections are tracked in a separate lightweight branch using detector
score, Step 0 prior metadata, and simple spatial/temporal plausibility gates.
Candidate outputs are split into raw, deduplicated, selected, and rejected
track groups so downstream reasoning can diagnose candidate-track explosion and
where pruning occurred.

Output layout:
    pipeline_output/02_driving_mini_tracking/<video_id>/
        tracks.json          — per-frame accepted/candidate tracking results
        tracks_manifest.json — summary manifest (written by run())
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import cv2
except Exception as exc:  # pragma: no cover - optional runtime dependency guard
    cv2 = None
    _CV2_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _CV2_IMPORT_ERROR = None
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# ---------------------------------------------------------------------------
# ByteTrack via ultralytics
# ---------------------------------------------------------------------------

try:
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.engine.results import Boxes
except Exception as exc:  # pragma: no cover - optional runtime dependency guard
    BYTETracker = None
    Boxes = Any
    _BYTE_TRACKER_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _BYTE_TRACKER_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Default tracker parameters (mirrors bytetrack.yaml defaults)
# ---------------------------------------------------------------------------

_DEFAULT_TRACKER_ARGS = SimpleNamespace(
    tracker_type="bytetrack",
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    track_buffer=30,   # keep lost tracks alive for up to 30 frames before dropping
    match_thresh=0.8,
    fuse_score=True,
)

# Placeholder image shape used when no image is provided to the tracker.
# ByteTracker needs orig_shape to build Boxes; we use the first available
# detection to derive it when possible, otherwise fall back to this.
_FALLBACK_SHAPE = (1080, 1920)  # (height, width)
_TRACKING_SCHEMA_VERSION = 7
_CANDIDATE_TRACK_ID_OFFSET = 1_000_000
_DEFAULT_CANDIDATE_SCORING_POLICY: Dict[str, Any] = {
    "visual_score_weight": 1.0,
    "prior_relevance_weight": 0.35,
    "retention_bias_weight": 0.2,
    "source_bonus": {
        "nms_relaxed_candidate": 0.25,
        "borderline_confidence": 0.18,
        "low_confidence": 0.08,
        "discarded_detector_output": 0.0,
        "candidate": 0.0,
    },
    "tracking_priority_bonus": {
        "highest": 0.2,
        "high": 0.15,
        "medium_high": 0.12,
        "medium": 0.08,
        "low": 0.03,
    },
    "spatial_plausibility_gate": {
        "min_bbox_width": 4.0,
        "min_bbox_height": 4.0,
        "min_bbox_area": 25.0,
        "frame_margin_ratio": 0.05,
    },
    "temporal_consistency_gate": {
        "match_iou_threshold": 0.3,
        "max_frame_gap": 1,
        "max_idle_frames": 2,
    },
    "candidate_selection_thresholds": {
        "min_visual_score": 0.01,
        "min_combined_score": 0.35,
        "min_score_without_prior": 0.12,
        "min_new_track_combined_score": 0.45,
    },
    "exploration_quota": {
        "non_prior_high_score_candidates_per_frame": 1,
        "non_prior_high_score_min_score": 0.35,
    },
    "candidate_track_deduplication": {
        "accepted_duplicate_mean_iou": 0.6,
        "accepted_duplicate_shared_frame_ratio": 0.5,
        "candidate_duplicate_mean_iou": 0.55,
        "candidate_duplicate_shared_frame_ratio": 0.5,
        "duplicate_penalty_cap": 1.0,
        "time_bucket_size": 12,
        "grid_size": [4, 3],
        "center_distance_threshold_ratio": 0.12,
    },
    "candidate_track_selection": {
        "visual_score_weight": 0.35,
        "prior_relevance_weight": 0.2,
        "temporal_consistency_weight": 0.2,
        "spatial_plausibility_weight": 0.2,
        "duplicate_penalty_weight": 0.25,
        "min_selection_score": 0.15,
    },
    "pre_tracking_gate": {
        "visual_score_weight": 0.45,
        "prior_relevance_weight": 0.25,
        "tracking_priority_weight": 0.15,
        "spatial_plausibility_weight": 0.1,
        "accepted_track_suppression_weight": 0.35,
        "accepted_track_suppression_iou": 0.4,
        "min_gate_score": 0.2,
        "max_candidates_per_frame": 18,
        "max_candidates_per_class_per_frame_default": 3,
        "max_candidates_per_class_per_frame": {
            "car": 4,
            "person": 4,
            "traffic light": 4,
        },
        "max_tracking_inputs_per_video": 320,
    },
    "early_fragment_merge": {
        "enabled": True,
        "max_frame_gap": 2,
        "min_end_start_iou": 0.2,
        "max_center_distance_ratio": 0.1,
        "time_bucket_size": 12,
        "grid_size": [4, 3],
    },
    "selection_budgets": {
        "max_tracking_inputs_per_video": 320,
        "max_raw_tracks": 160,
        "max_deduplicated_tracks": 64,
        "max_selected_tracks": 24,
        "max_tracks_per_video": 24,
        "max_tracks_per_class_default": 4,
        "max_tracks_per_class": {
            "car": 6,
            "truck": 4,
            "bus": 3,
            "motorcycle": 4,
            "bicycle": 4,
            "person": 6,
            "traffic light": 4,
            "stop sign": 3,
        },
    },
}


def _format_dependency_error(exc: Optional[BaseException]) -> str:
    if exc is None:
        return ""
    return f"{exc.__class__.__name__}: {exc}"


def ensure_tracking_runtime_available() -> bool:
    if BYTETracker is None:
        raise RuntimeError(
            "Tracking dependencies are unavailable. "
            "ByteTrack requires `ultralytics` and its runtime dependencies. "
            f"Import error: {_format_dependency_error(_BYTE_TRACKER_IMPORT_ERROR)}"
        )
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "02_driving_mini_tracking"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    tmp_path.replace(path)


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Return a stable BGR color for a given track ID."""
    h = (track_id * 37) % 360
    s, v = 0.85, 0.95
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))


def render_tracking_video(
    video_id: str,
    tracked_frames: List[Dict[str, Any]],
    output_path: Path,
    fps: float = 10.0,
) -> Optional[str]:
    """Render an annotated MP4 with bounding boxes colored by track ID."""
    if cv2 is None:
        return None
    if not tracked_frames:
        return None

    first = cv2.imread(tracked_frames[0]["image_path"])
    if first is None:
        return None

    h, w = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        return None

    try:
        for rec in tracked_frames:
            img = cv2.imread(rec["image_path"])
            if img is None:
                continue

            for box, score, label, track_id in zip(
                rec.get("boxes", []),
                rec.get("scores", []),
                rec.get("labels", []),
                rec.get("track_ids", []),
            ):
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                color = _track_color(track_id)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                text = f"#{track_id} {label} {float(score):.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_y = max(y1 - 6, th + 6)
                cv2.rectangle(img, (x1, text_y - th - 4), (x1 + tw + 4, text_y), color, -1)
                cv2.putText(
                    img, text, (x1 + 2, text_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
                )

            cv2.putText(
                img,
                f"{video_id} | frame {rec.get('frame_index', -1):04d}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
            writer.write(img)
    finally:
        writer.release()

    return str(output_path)


def _make_tracker(frame_rate: int = 10, args: Optional[SimpleNamespace] = None) -> BYTETracker:
    """Create a fresh BYTETracker instance."""
    tracker_args = args or _DEFAULT_TRACKER_ARGS
    try:
        return BYTETracker(tracker_args, frame_rate=frame_rate)
    except TypeError as exc:
        # Newer ultralytics releases removed the frame_rate kwarg.
        if "frame_rate" not in str(exc):
            raise
        return BYTETracker(tracker_args)


def _infer_orig_shape(frame_records: List[Dict[str, Any]]) -> tuple[int, int]:
    """Try to read image dimensions from the first available frame image."""
    try:
        import cv2
        for rec in frame_records:
            img = cv2.imread(rec.get("image_path", ""))
            if img is not None:
                h, w = img.shape[:2]
                return (h, w)
    except Exception:
        pass
    return _FALLBACK_SHAPE


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _detection_id(frame_index: int, collection_name: str, det_index: int) -> str:
    return f"{int(frame_index):06d}:{collection_name}:{int(det_index):04d}"


def _coerce_bbox(box: Sequence[Any]) -> List[float]:
    if len(box) < 4:
        return []
    return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]


def _center_xy(box: Sequence[float]) -> Tuple[float, float]:
    return ((float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5)


def _bbox_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _match_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = _bbox_area(box_a)
    area_b = _bbox_area(box_b)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0.0 else 0.0


def _accepted_detection_records(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    accepted_detections = list(rec.get("accepted_detections", []))
    if accepted_detections:
        return accepted_detections
    boxes = list(rec.get("boxes", []))
    scores = list(rec.get("scores", []))
    labels = list(rec.get("labels", []))
    fallback: List[Dict[str, Any]] = []
    for det_index, box in enumerate(boxes):
        fallback.append(
            {
                "bbox": list(box),
                "score": _safe_float(scores[det_index] if det_index < len(scores) else 0.0),
                "class": str(labels[det_index] if det_index < len(labels) else "unknown"),
                "accepted": True,
                "candidate_source": "accepted_high_confidence",
                "threshold_info": {},
                "prior_metadata": {},
            }
        )
    return fallback


def _candidate_detection_records(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(rec.get("candidate_detections", []))


def _candidate_branch_enabled(video_result: Dict[str, Any]) -> bool:
    return bool(video_result.get("candidate_branch_enabled", True))


def _empty_candidate_track_bundle(policy: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_tracking_input_candidates": 0,
        "tracking_input_candidates": {
            "num_candidates": 0,
            "frames": [],
        },
        "raw_candidate_tracks": {
            "num_tracks": 0,
            "frames": [],
            "track_summaries": [],
        },
        "deduplicated_candidate_tracks": {
            "num_tracks": 0,
            "frames": [],
            "track_summaries": [],
        },
        "selected_candidate_tracks": {
            "num_tracks": 0,
            "frames": [],
            "track_summaries": [],
        },
        "rejected_candidate_tracks": {
            "num_tracks": 0,
            "tracks": [],
            "rejection_reason_counts": {},
        },
        "rejected_candidate_detections": {
            "num_rejected": 0,
            "frames": [],
            "rejection_reason_counts": {},
            "selection_policy": policy,
        },
        "selection_policy": policy,
        "selection_budgets": {},
    }


def _load_candidate_scoring_policy(video_result: Dict[str, Any]) -> Dict[str, Any]:
    policy = json.loads(json.dumps(_DEFAULT_CANDIDATE_SCORING_POLICY))
    policy["od_calibration_policy_id"] = str(
        dict(video_result.get("od_calibration", {})).get("policy_id", "")
    )
    prior_json_path = str(video_result.get("output_paths", {}).get("background_rule_relevance_prior_json", ""))
    if not prior_json_path:
        return policy
    try:
        with Path(prior_json_path).open("r", encoding="utf-8") as fh:
            prior_payload = json.load(fh)
        override = dict(prior_payload.get("candidate_scoring_policy", {}))
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(policy.get(key), dict):
                policy[key].update(value)
            else:
                policy[key] = value
    except Exception:
        return policy
    return policy


def _retention_bias(detection: Dict[str, Any]) -> float:
    prior_metadata = dict(detection.get("prior_metadata", {}))
    matched_prior_ids = [str(value) for value in list(prior_metadata.get("matched_prior_ids", [])) if str(value)]
    retention_weights = dict(prior_metadata.get("matched_prior_retention_weights", {}))
    biases = [
        _safe_float(dict(retention_weights.get(prior_id, {})).get("retention_bias", 0.0))
        for prior_id in matched_prior_ids
    ]
    return max(biases) if biases else 0.0


def _raw_detection_score(detection: Dict[str, Any]) -> float:
    return _safe_float(detection.get("raw_score", detection.get("score", 0.0)), 0.0)


def _candidate_ranking_score(detection: Dict[str, Any]) -> float:
    return _safe_float(
        detection.get(
            "score_used_for_candidate_ranking",
            detection.get("calibrated_score", detection.get("raw_score", detection.get("score", 0.0))),
        ),
        0.0,
    )


def _candidate_spatially_plausible(
    detection: Dict[str, Any],
    orig_shape: Tuple[int, int],
    policy: Dict[str, Any],
) -> bool:
    bbox = list(detection.get("bbox", []))
    if len(bbox) < 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return False
    width = x2 - x1
    height = y2 - y1
    spatial_cfg = dict(policy.get("spatial_plausibility_gate", {}))
    if width < _safe_float(spatial_cfg.get("min_bbox_width", 4.0), 4.0) or height < _safe_float(spatial_cfg.get("min_bbox_height", 4.0), 4.0):
        return False
    if _bbox_area(bbox) < _safe_float(spatial_cfg.get("min_bbox_area", 25.0), 25.0):
        return False
    frame_h, frame_w = orig_shape
    margin_ratio = _safe_float(spatial_cfg.get("frame_margin_ratio", 0.05), 0.05)
    cx, cy = _center_xy(bbox)
    return -margin_ratio * frame_w <= cx <= (1.0 + margin_ratio) * frame_w and -margin_ratio * frame_h <= cy <= (1.0 + margin_ratio) * frame_h


def _candidate_combined_score(detection: Dict[str, Any], policy: Dict[str, Any]) -> float:
    prior_metadata = dict(detection.get("prior_metadata", {}))
    matched_prior_ids = [str(value) for value in list(prior_metadata.get("matched_prior_ids", [])) if str(value)]
    matched_priority_map = dict(prior_metadata.get("matched_prior_tracking_priorities", {}))
    top_priority = matched_priority_map.get(matched_prior_ids[0], "") if matched_prior_ids else ""
    source_bonus = dict(policy.get("source_bonus", {}))
    priority_bonus = dict(policy.get("tracking_priority_bonus", {}))
    return (
        _safe_float(policy.get("visual_score_weight", 1.0), 1.0) * _candidate_ranking_score(detection)
        + _safe_float(policy.get("prior_relevance_weight", 0.35), 0.35) * _safe_float(prior_metadata.get("prior_relevance_score", 0.0))
        + _safe_float(source_bonus.get(str(detection.get("candidate_source", "")), 0.0))
        + _safe_float(priority_bonus.get(str(top_priority), 0.0))
        + _safe_float(policy.get("retention_bias_weight", 0.2), 0.2) * _retention_bias(detection)
    )


def _top_tracking_priority(detection: Dict[str, Any]) -> str:
    prior_metadata = dict(detection.get("prior_metadata", {}))
    matched_prior_ids = [str(value) for value in list(prior_metadata.get("matched_prior_ids", [])) if str(value)]
    matched_priority_map = dict(prior_metadata.get("matched_prior_tracking_priorities", {}))
    return str(matched_priority_map.get(matched_prior_ids[0], "")) if matched_prior_ids else ""


def _tracking_priority_value(priority: str, policy: Dict[str, Any]) -> float:
    return _safe_float(dict(policy.get("tracking_priority_bonus", {})).get(str(priority), 0.0))


def _accepted_frame_detections(accepted_frame: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(accepted_frame, dict):
        return []
    detections: List[Dict[str, Any]] = []
    for box, score, label, track_id, detection_id in zip(
        accepted_frame.get("boxes", []),
        accepted_frame.get("scores", []),
        accepted_frame.get("labels", []),
        accepted_frame.get("track_ids", []),
        accepted_frame.get("detection_ids", []),
    ):
        detections.append(
            {
                "bbox": list(box),
                "score": _safe_float(score),
                "class": str(label),
                "track_id": int(track_id),
                "detection_id": str(detection_id),
            }
        )
    return detections


def _candidate_rejection_payload(
    detection: Dict[str, Any],
    reason: str,
    **extra: Any,
) -> Dict[str, Any]:
    return {
        "detection_id": str(detection.get("detection_id", "")),
        "bbox": list(detection.get("bbox", [])),
        "class": str(detection.get("class", "unknown")),
        "score": _safe_float(detection.get("score", 0.0)),
        "raw_score": _raw_detection_score(detection),
        "calibrated_score": _safe_float(detection.get("calibrated_score", detection.get("score", 0.0)), 0.0),
        "feedback_bonus": _safe_float(detection.get("feedback_bonus", 0.0), 0.0),
        "score_used_for_candidate_ranking": _candidate_ranking_score(detection),
        "candidate_source": str(detection.get("candidate_source", "")),
        "combined_tracking_score": _safe_float(detection.get("combined_tracking_score", 0.0)),
        "pre_tracking_score": _safe_float(detection.get("pre_tracking_score", 0.0)),
        "matched_prior_ids": list(detection.get("matched_prior_ids", [])),
        "prior_relevance_score": _safe_float(detection.get("prior_relevance_score", 0.0)),
        "tracking_priority": str(detection.get("tracking_priority", "")),
        "accepted_track_suppression_iou": _safe_float(detection.get("accepted_track_suppression_iou", 0.0)),
        "accepted_track_suppression_track_id": detection.get("accepted_track_suppression_track_id"),
        "prior_metadata": dict(detection.get("prior_metadata", {})),
        "rejection_reason": str(reason),
        **extra,
    }


def _accepted_track_suppression(
    detection: Dict[str, Any],
    accepted_frame_detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    label = str(detection.get("class", "unknown"))
    bbox = list(detection.get("bbox", []))
    best_match: Dict[str, Any] = {}
    for accepted in accepted_frame_detections:
        if str(accepted.get("class", "")) != label:
            continue
        iou = _match_iou(bbox, list(accepted.get("bbox", [])))
        if iou > _safe_float(best_match.get("iou", 0.0)):
            best_match = {
                "iou": float(iou),
                "track_id": int(accepted.get("track_id", -1)),
                "detection_id": str(accepted.get("detection_id", "")),
            }
    return best_match


def _candidate_pretracking_score(
    detection: Dict[str, Any],
    orig_shape: Tuple[int, int],
    policy: Dict[str, Any],
) -> float:
    gate_cfg = dict(policy.get("pre_tracking_gate", {}))
    spatial_plausibility = 1.0 if _candidate_spatially_plausible(detection, orig_shape, policy) else 0.0
    accepted_suppression = _safe_float(detection.get("accepted_track_suppression_iou", 0.0))
    return (
        _safe_float(gate_cfg.get("visual_score_weight", 0.45), 0.45) * _candidate_ranking_score(detection)
        + _safe_float(gate_cfg.get("prior_relevance_weight", 0.25), 0.25) * _safe_float(detection.get("prior_relevance_score", 0.0))
        + _safe_float(gate_cfg.get("tracking_priority_weight", 0.15), 0.15) * _tracking_priority_value(
            str(detection.get("tracking_priority", "")),
            policy,
        )
        + _safe_float(gate_cfg.get("spatial_plausibility_weight", 0.1), 0.1) * spatial_plausibility
        - _safe_float(gate_cfg.get("accepted_track_suppression_weight", 0.35), 0.35) * accepted_suppression
    )


def _candidate_selection_decision(
    detection: Dict[str, Any],
    orig_shape: Tuple[int, int],
    policy: Dict[str, Any],
) -> Tuple[bool, str]:
    score = _candidate_ranking_score(detection)
    threshold_cfg = dict(policy.get("candidate_selection_thresholds", {}))
    if score < _safe_float(threshold_cfg.get("min_visual_score", 0.01), 0.01):
        return False, "below_min_visual_score"
    if not _candidate_spatially_plausible(detection, orig_shape, policy):
        return False, "spatially_implausible"
    prior_metadata = dict(detection.get("prior_metadata", {}))
    matched_prior_ids = [str(value) for value in list(prior_metadata.get("matched_prior_ids", [])) if str(value)]
    combined_score = _candidate_combined_score(detection, policy)
    if combined_score < _safe_float(threshold_cfg.get("min_combined_score", 0.35), 0.35):
        return False, "below_combined_score_threshold"
    if not matched_prior_ids and score < _safe_float(threshold_cfg.get("min_score_without_prior", 0.12), 0.12):
        return False, "non_prior_low_score"
    return True, "selected"


def _candidate_summary_from_track(track: Dict[str, Any]) -> Dict[str, Any]:
    observations = list(track.get("observations", []))
    detection_ids = [str(obs.get("detection_id", "")) for obs in observations if str(obs.get("detection_id", ""))]
    candidate_sources = [str(obs.get("candidate_source", "")) for obs in observations if str(obs.get("candidate_source", ""))]
    scores = [_safe_float(obs.get("score", 0.0)) for obs in observations]
    raw_scores = [_safe_float(obs.get("raw_score", obs.get("score", 0.0))) for obs in observations]
    calibrated_scores = [
        _safe_float(obs.get("calibrated_score", obs.get("raw_score", obs.get("score", 0.0))))
        for obs in observations
    ]
    ranking_scores = [
        _safe_float(
            obs.get(
                "score_used_for_candidate_ranking",
                obs.get("calibrated_score", obs.get("raw_score", obs.get("score", 0.0))),
            )
        )
        for obs in observations
    ]
    prior_scores = [_safe_float(obs.get("prior_relevance_score", 0.0)) for obs in observations]
    ious = [_safe_float(value, 0.0) for value in list(track.get("match_ious", []))]
    matched_prior_ids: set[str] = set()
    prior_id_frequency: Counter[str] = Counter()
    for obs in observations:
        for prior_id in list(obs.get("matched_prior_ids", [])):
            prior_id_text = str(prior_id)
            if not prior_id_text:
                continue
            matched_prior_ids.add(prior_id_text)
            prior_id_frequency.update([prior_id_text])
    label_counts = Counter(str(obs.get("label", "unknown")) for obs in observations)
    summary = {
        "track_id": int(track.get("track_id", -1)),
        "label": label_counts.most_common(1)[0][0] if label_counts else "unknown",
        "source_detection_ids": detection_ids,
        "candidate_sources": sorted(set(candidate_sources)),
        "candidate_source_counts": dict(sorted(Counter(candidate_sources).items())),
        "mean_score": float(sum(scores) / max(1, len(scores))),
        "max_score": float(max(scores) if scores else 0.0),
        "mean_raw_score": float(sum(raw_scores) / max(1, len(raw_scores))),
        "mean_calibrated_score": float(sum(calibrated_scores) / max(1, len(calibrated_scores))),
        "mean_ranking_score": float(sum(ranking_scores) / max(1, len(ranking_scores))),
        "max_calibrated_score": float(max(calibrated_scores) if calibrated_scores else 0.0),
        "track_length": len(observations),
        "temporal_consistency": float(sum(ious) / max(1, len(ious))) if ious else 1.0,
        "matched_prior_ids": sorted(matched_prior_ids),
        "matched_prior_id_counts": dict(sorted(prior_id_frequency.items())),
        "prior_relevance_mean": float(sum(prior_scores) / max(1, len(prior_scores))),
        "prior_relevance_max": float(max(prior_scores) if prior_scores else 0.0),
        "prior_relevance_min": float(min(prior_scores) if prior_scores else 0.0),
        "first_frame_index": int(observations[0].get("frame_index", -1)) if observations else -1,
        "last_frame_index": int(observations[-1].get("frame_index", -1)) if observations else -1,
    }
    if isinstance(track.get("duplicate_with_accepted"), dict):
        summary["duplicate_with_accepted"] = dict(track.get("duplicate_with_accepted", {}))
    if isinstance(track.get("duplicate_with_candidate"), dict):
        summary["duplicate_with_candidate"] = dict(track.get("duplicate_with_candidate", {}))
    if isinstance(track.get("score_breakdown"), dict):
        summary["score_breakdown"] = dict(track.get("score_breakdown", {}))
        summary["selection_score"] = _safe_float(dict(track.get("score_breakdown", {})).get("selection_score", 0.0))
    return summary


def _accepted_track_summaries(tracked_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    track_rows: Dict[int, Dict[str, Any]] = {}
    for rec in tracked_frames:
        frame_index = int(rec.get("frame_index", -1))
        for box, score, label, track_id in zip(
            rec.get("boxes", []),
            rec.get("scores", []),
            rec.get("labels", []),
            rec.get("track_ids", []),
        ):
            bucket = track_rows.setdefault(
                int(track_id),
                {"track_id": int(track_id), "labels": Counter(), "scores": [], "frames": [], "boxes": []},
            )
            bucket["labels"].update([str(label)])
            bucket["scores"].append(_safe_float(score))
            bucket["frames"].append(frame_index)
            bucket["boxes"].append(list(box))
    summaries: List[Dict[str, Any]] = []
    for track_id, bucket in sorted(track_rows.items()):
        scores = list(bucket["scores"])
        frames = list(bucket["frames"])
        summaries.append(
            {
                "track_id": int(track_id),
                "label": bucket["labels"].most_common(1)[0][0] if bucket["labels"] else "unknown",
                "mean_score": float(sum(scores) / max(1, len(scores))),
                "max_score": float(max(scores) if scores else 0.0),
                "track_length": len(scores),
                "first_frame_index": min(frames) if frames else -1,
                "last_frame_index": max(frames) if frames else -1,
            }
        )
    return summaries


def _accepted_track_records(tracked_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    track_rows: Dict[int, Dict[str, Any]] = {}
    for rec in tracked_frames:
        frame_index = int(rec.get("frame_index", -1))
        image_path = str(rec.get("image_path", ""))
        for box, score, label, track_id, detection_id in zip(
            rec.get("boxes", []),
            rec.get("scores", []),
            rec.get("labels", []),
            rec.get("track_ids", []),
            rec.get("detection_ids", []),
        ):
            bucket = track_rows.setdefault(
                int(track_id),
                {
                    "track_id": int(track_id),
                    "label": str(label),
                    "observations": [],
                },
            )
            bucket["observations"].append(
                {
                    "frame_index": frame_index,
                    "image_path": image_path,
                    "bbox": list(box),
                    "score": _safe_float(score),
                    "label": str(label),
                    "detection_id": str(detection_id),
                }
            )
    return [track_rows[track_id] for track_id in sorted(track_rows)]


def _tracks_to_frame_records(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    frame_map: Dict[int, Dict[str, Any]] = {}
    for track in tracks:
        track_id = int(track.get("track_id", -1))
        for obs in list(track.get("observations", [])):
            frame_index = int(obs.get("frame_index", -1))
            frame_rec = frame_map.setdefault(
                frame_index,
                {
                    "frame": obs.get("frame", ""),
                    "frame_index": frame_index,
                    "image_path": obs.get("image_path", ""),
                    "boxes": [],
                    "scores": [],
                    "labels": [],
                    "track_ids": [],
                    "detection_ids": [],
                    "candidate_sources": [],
                    "prior_relevance_scores": [],
                    "matched_prior_ids": [],
                },
            )
            frame_rec["boxes"].append(list(obs.get("bbox", [])))
            frame_rec["scores"].append(_safe_float(obs.get("score", 0.0)))
            frame_rec["labels"].append(str(obs.get("label", "unknown")))
            frame_rec["track_ids"].append(track_id)
            frame_rec["detection_ids"].append(str(obs.get("detection_id", "")))
            frame_rec["candidate_sources"].append(str(obs.get("candidate_source", "")))
            frame_rec["prior_relevance_scores"].append(_safe_float(obs.get("prior_relevance_score", 0.0)))
            frame_rec["matched_prior_ids"].append(list(obs.get("matched_prior_ids", [])))
    return [frame_map[frame_index] for frame_index in sorted(frame_map)]


def _track_overlap_metrics(track_a: Dict[str, Any], track_b: Dict[str, Any]) -> Dict[str, Any]:
    observations_a = list(track_a.get("observations", []))
    observations_b = list(track_b.get("observations", []))
    index_b = {int(obs.get("frame_index", -1)): obs for obs in observations_b}
    shared_ious: List[float] = []
    shared_frames: List[int] = []
    for obs_a in observations_a:
        frame_index = int(obs_a.get("frame_index", -1))
        obs_b = index_b.get(frame_index)
        if obs_b is None:
            continue
        shared_frames.append(frame_index)
        shared_ious.append(_match_iou(list(obs_a.get("bbox", [])), list(obs_b.get("bbox", []))))
    shared_count = len(shared_ious)
    denom = max(1, min(len(observations_a), len(observations_b)))
    mean_iou = float(sum(shared_ious) / shared_count) if shared_count else 0.0
    max_iou = float(max(shared_ious) if shared_ious else 0.0)
    return {
        "shared_frame_count": shared_count,
        "shared_frame_ratio": float(shared_count / denom),
        "mean_iou": mean_iou,
        "max_iou": max_iou,
        "overlap_score": float(mean_iou * (shared_count / denom)),
        "shared_frame_indices": shared_frames,
        "other_track_id": int(track_b.get("track_id", -1)),
    }


def _track_spatial_plausibility(track: Dict[str, Any], orig_shape: Tuple[int, int], policy: Dict[str, Any]) -> float:
    observations = list(track.get("observations", []))
    if not observations:
        return 0.0
    plausible_count = sum(
        1
        for obs in observations
        if _candidate_spatially_plausible({"bbox": list(obs.get("bbox", []))}, orig_shape, policy)
    )
    return float(plausible_count / max(1, len(observations)))


def _candidate_track_score_breakdown(track: Dict[str, Any], orig_shape: Tuple[int, int], policy: Dict[str, Any]) -> Dict[str, Any]:
    selection_cfg = dict(policy.get("candidate_track_selection", {}))
    dedupe_cfg = dict(policy.get("candidate_track_deduplication", {}))
    summary = _candidate_summary_from_track(track)
    spatial_plausibility = _track_spatial_plausibility(track, orig_shape, policy)
    accepted_overlap_score = _safe_float(dict(track.get("duplicate_with_accepted", {})).get("overlap_score", 0.0))
    candidate_overlap_score = _safe_float(dict(track.get("duplicate_with_candidate", {})).get("overlap_score", 0.0))
    duplicate_penalty_raw = max(accepted_overlap_score, candidate_overlap_score)
    duplicate_penalty = min(
        duplicate_penalty_raw,
        _safe_float(dedupe_cfg.get("duplicate_penalty_cap", 1.0), 1.0),
    )
    visual_component = _safe_float(selection_cfg.get("visual_score_weight", 0.35), 0.35) * _safe_float(
        summary.get("mean_ranking_score", summary.get("mean_score", 0.0))
    )
    prior_component = _safe_float(selection_cfg.get("prior_relevance_weight", 0.2), 0.2) * _safe_float(
        summary.get("prior_relevance_mean", 0.0)
    )
    temporal_component = _safe_float(selection_cfg.get("temporal_consistency_weight", 0.2), 0.2) * _safe_float(
        summary.get("temporal_consistency", 0.0)
    )
    spatial_component = _safe_float(selection_cfg.get("spatial_plausibility_weight", 0.2), 0.2) * spatial_plausibility
    duplicate_penalty_component = _safe_float(
        selection_cfg.get("duplicate_penalty_weight", 0.25),
        0.25,
    ) * duplicate_penalty
    selection_score = (
        visual_component
        + prior_component
        + temporal_component
        + spatial_component
        - duplicate_penalty_component
    )
    return {
        "selection_score": float(selection_score),
        "visual_component": float(visual_component),
        "prior_component": float(prior_component),
        "temporal_component": float(temporal_component),
        "spatial_component": float(spatial_component),
        "duplicate_penalty_component": float(duplicate_penalty_component),
        "spatial_plausibility": float(spatial_plausibility),
        "duplicate_penalty_raw": float(duplicate_penalty_raw),
        "duplicate_penalty": float(duplicate_penalty),
        "mean_score": _safe_float(summary.get("mean_score", 0.0)),
        "mean_raw_score": _safe_float(summary.get("mean_raw_score", 0.0)),
        "mean_calibrated_score": _safe_float(summary.get("mean_calibrated_score", 0.0)),
        "mean_ranking_score": _safe_float(summary.get("mean_ranking_score", 0.0)),
        "prior_relevance_mean": _safe_float(summary.get("prior_relevance_mean", 0.0)),
        "temporal_consistency": _safe_float(summary.get("temporal_consistency", 0.0)),
    }


def _candidate_track_rejection_entry(track: Dict[str, Any], reason: str, **extra: Any) -> Dict[str, Any]:
    summary = _candidate_summary_from_track(track)
    return {
        "track_id": int(track.get("track_id", -1)),
        "label": str(summary.get("label", "unknown")),
        "rejection_reason": str(reason),
        "selection_score": _safe_float(dict(track.get("score_breakdown", {})).get("selection_score", 0.0)),
        "score_breakdown": dict(track.get("score_breakdown", {})),
        "duplicate_with_accepted": dict(track.get("duplicate_with_accepted", {})),
        "duplicate_with_candidate": dict(track.get("duplicate_with_candidate", {})),
        "summary": summary,
        "provenance": {
            "observations": list(track.get("observations", [])),
            "match_ious": list(track.get("match_ious", [])),
        },
        **extra,
    }


def _center_distance_ratio(box_a: Sequence[float], box_b: Sequence[float], orig_shape: Tuple[int, int]) -> float:
    ax, ay = _center_xy(box_a)
    bx, by = _center_xy(box_b)
    frame_h, frame_w = orig_shape
    norm_x = max(1.0, float(frame_w))
    norm_y = max(1.0, float(frame_h))
    dx = (ax - bx) / norm_x
    dy = (ay - by) / norm_y
    return float((dx * dx + dy * dy) ** 0.5)


def _track_prior_signature(track: Dict[str, Any], max_ids: int = 2) -> Tuple[str, ...]:
    summary = _candidate_summary_from_track(track)
    prior_ids = [str(value) for value in list(summary.get("matched_prior_ids", [])) if str(value)]
    return tuple(sorted(prior_ids[:max_ids])) if prior_ids else ("no_prior",)


def _track_mean_center(track: Dict[str, Any]) -> Tuple[float, float]:
    observations = list(track.get("observations", []))
    if not observations:
        return (0.0, 0.0)
    centers = [_center_xy(list(obs.get("bbox", []))) for obs in observations if len(list(obs.get("bbox", []))) >= 4]
    if not centers:
        return (0.0, 0.0)
    return (
        float(sum(center[0] for center in centers) / len(centers)),
        float(sum(center[1] for center in centers) / len(centers)),
    )


def _track_bucket_keys(
    track: Dict[str, Any],
    orig_shape: Tuple[int, int],
    time_bucket_size: int,
    grid_size: Sequence[int],
) -> List[Tuple[Any, ...]]:
    summary = _candidate_summary_from_track(track)
    label = str(summary.get("label", "unknown"))
    first_frame = int(summary.get("first_frame_index", -1))
    last_frame = int(summary.get("last_frame_index", -1))
    midpoint_frame = max(0, (first_frame + last_frame) // 2)
    time_bucket = midpoint_frame // max(1, time_bucket_size)
    mean_cx, mean_cy = _track_mean_center(track)
    frame_h, frame_w = orig_shape
    grid_w = max(1, _safe_int(grid_size[0] if len(grid_size) > 0 else 4, 4))
    grid_h = max(1, _safe_int(grid_size[1] if len(grid_size) > 1 else 3, 3))
    cell_x = min(grid_w - 1, max(0, int((mean_cx / max(1.0, float(frame_w))) * grid_w)))
    cell_y = min(grid_h - 1, max(0, int((mean_cy / max(1.0, float(frame_h))) * grid_h)))
    prior_sig = _track_prior_signature(track)
    keys: List[Tuple[Any, ...]] = []
    for delta_t in (-1, 0, 1):
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                keys.append(
                    (
                        label,
                        prior_sig,
                        time_bucket + delta_t,
                        min(grid_w - 1, max(0, cell_x + delta_x)),
                        min(grid_h - 1, max(0, cell_y + delta_y)),
                    )
                )
    return keys


def _merge_track_fragments(
    tracks: List[Dict[str, Any]],
    orig_shape: Tuple[int, int],
    policy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    merge_cfg = dict(policy.get("early_fragment_merge", {}))
    if not bool(merge_cfg.get("enabled", True)) or len(tracks) < 2:
        return tracks
    max_frame_gap = max(0, _safe_int(merge_cfg.get("max_frame_gap", 2), 2))
    min_end_start_iou = _safe_float(merge_cfg.get("min_end_start_iou", 0.2), 0.2)
    max_center_distance = _safe_float(merge_cfg.get("max_center_distance_ratio", 0.1), 0.1)
    time_bucket_size = max(1, _safe_int(merge_cfg.get("time_bucket_size", 12), 12))
    grid_size = list(merge_cfg.get("grid_size", [4, 3]))

    sorted_tracks = sorted(
        tracks,
        key=lambda track: (
            str(track.get("label", "unknown")),
            int(_candidate_summary_from_track(track).get("first_frame_index", -1)),
            int(track.get("track_id", -1)),
        ),
    )
    bucket_index: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    merged_tracks: List[Dict[str, Any]] = []
    for track in sorted_tracks:
        summary = _candidate_summary_from_track(track)
        label = str(summary.get("label", "unknown"))
        first_frame = int(summary.get("first_frame_index", -1))
        start_box = list(track.get("observations", [{}])[0].get("bbox", [])) if track.get("observations") else []
        best_merge_target: Optional[Dict[str, Any]] = None
        best_merge_score = -1.0
        for bucket_key in _track_bucket_keys(track, orig_shape, time_bucket_size, grid_size):
            for existing in bucket_index.get(bucket_key, []):
                existing_summary = _candidate_summary_from_track(existing)
                if str(existing_summary.get("label", "unknown")) != label:
                    continue
                last_frame = int(existing_summary.get("last_frame_index", -1))
                frame_gap = first_frame - last_frame
                if frame_gap < 1 or frame_gap > max_frame_gap:
                    continue
                if _track_prior_signature(existing) != _track_prior_signature(track):
                    continue
                end_box = list(existing.get("observations", [{}])[-1].get("bbox", [])) if existing.get("observations") else []
                if len(start_box) < 4 or len(end_box) < 4:
                    continue
                end_start_iou = _match_iou(start_box, end_box)
                center_distance = _center_distance_ratio(start_box, end_box, orig_shape)
                if end_start_iou < min_end_start_iou and center_distance > max_center_distance:
                    continue
                merge_score = end_start_iou - center_distance
                if merge_score > best_merge_score:
                    best_merge_target = existing
                    best_merge_score = merge_score
        if best_merge_target is None:
            merged_tracks.append(track)
            for bucket_key in _track_bucket_keys(track, orig_shape, time_bucket_size, grid_size):
                bucket_index[bucket_key].append(track)
            continue
        best_merge_target["observations"].extend(list(track.get("observations", [])))
        best_merge_target["observations"].sort(key=lambda obs: int(obs.get("frame_index", -1)))
        best_merge_target["match_ious"].extend(list(track.get("match_ious", [])))
        best_merge_target["last_bbox"] = list(best_merge_target.get("observations", [{}])[-1].get("bbox", []))
        best_merge_target["last_frame_index"] = int(
            best_merge_target.get("observations", [{}])[-1].get("frame_index", best_merge_target.get("last_frame_index", -1))
        )
        best_merge_target.setdefault("merged_fragment_track_ids", []).append(int(track.get("track_id", -1)))
    return merged_tracks


# ---------------------------------------------------------------------------
# Per-video tracking
# ---------------------------------------------------------------------------

def _run_accepted_tracking(
    frame_records: List[Dict[str, Any]],
    orig_shape: Tuple[int, int],
    frame_rate: int,
    tracker_args: Optional[SimpleNamespace],
) -> Tuple[List[Dict[str, Any]], set[int]]:
    tracker = _make_tracker(frame_rate=frame_rate, args=tracker_args)
    tracked_frames: List[Dict[str, Any]] = []
    all_track_ids: set[int] = set()

    for rec in frame_records:
        accepted_detections = _accepted_detection_records(rec)
        boxes_raw = [list(det.get("bbox", [])) for det in accepted_detections]
        scores_raw = [_safe_float(det.get("score", 0.0)) for det in accepted_detections]
        labels_raw = [str(det.get("class", "unknown")) for det in accepted_detections]
        detection_ids = [
            _detection_id(int(rec.get("frame_index", -1)), "accepted", det_index)
            for det_index in range(len(accepted_detections))
        ]

        if boxes_raw:
            arr = np.zeros((len(boxes_raw), 6), dtype=np.float32)
            for det_index, (box, score) in enumerate(zip(boxes_raw, scores_raw)):
                arr[det_index, :4] = box
                arr[det_index, 4] = float(score)
                arr[det_index, 5] = float(det_index)
            track_output: np.ndarray = tracker.update(Boxes(arr, orig_shape))
        else:
            tracker.update(Boxes(np.empty((0, 6), dtype=np.float32), orig_shape))
            track_output = np.empty((0, 7), dtype=np.float32)

        frame_boxes: List[List[float]] = []
        frame_scores: List[float] = []
        frame_labels: List[str] = []
        frame_track_ids: List[int] = []
        frame_detection_ids: List[str] = []

        if track_output is not None and len(track_output):
            for row in track_output:
                cls_idx = int(round(float(row[6])))
                track_id = int(row[4])
                frame_boxes.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
                frame_scores.append(float(row[5]))
                frame_labels.append(labels_raw[cls_idx] if cls_idx < len(labels_raw) else "unknown")
                frame_track_ids.append(track_id)
                frame_detection_ids.append(detection_ids[cls_idx] if cls_idx < len(detection_ids) else "")
                all_track_ids.add(track_id)

        tracked_frames.append(
            {
                "frame": rec.get("frame", ""),
                "frame_index": rec.get("frame_index", -1),
                "image_path": rec.get("image_path", ""),
                "boxes": frame_boxes,
                "scores": frame_scores,
                "labels": frame_labels,
                "track_ids": frame_track_ids,
                "detection_ids": frame_detection_ids,
            }
        )

    return tracked_frames, all_track_ids


def _run_candidate_tracking(
    frame_records: List[Dict[str, Any]],
    accepted_frames: List[Dict[str, Any]],
    orig_shape: Tuple[int, int],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    active_tracks: Dict[int, Dict[str, Any]] = {}
    completed_tracks: List[Dict[str, Any]] = []
    rejected_frames: List[Dict[str, Any]] = []
    rejection_reason_counts: Counter[str] = Counter()
    accepted_frame_map = {
        int(rec.get("frame_index", -1)): rec
        for rec in accepted_frames
    }
    temporal_cfg = dict(policy.get("temporal_consistency_gate", {}))
    threshold_cfg = dict(policy.get("candidate_selection_thresholds", {}))
    exploration_cfg = dict(policy.get("exploration_quota", {}))
    gate_cfg = dict(policy.get("pre_tracking_gate", {}))
    budget_cfg = dict(policy.get("selection_budgets", {}))
    next_track_id = _CANDIDATE_TRACK_ID_OFFSET
    total_tracking_inputs = 0
    tracking_input_frames: List[Dict[str, Any]] = []

    for rec in frame_records:
        frame_index = int(rec.get("frame_index", -1))
        max_idle_frames = _safe_int(temporal_cfg.get("max_idle_frames", 2), 2)
        max_frame_gap = _safe_int(temporal_cfg.get("max_frame_gap", 1), 1)
        match_iou_threshold = _safe_float(temporal_cfg.get("match_iou_threshold", 0.3), 0.3)
        exploration_quota = max(0, _safe_int(exploration_cfg.get("non_prior_high_score_candidates_per_frame", 1), 1))
        exploration_min_score = _safe_float(exploration_cfg.get("non_prior_high_score_min_score", 0.35), 0.35)
        max_candidates_per_frame = max(0, _safe_int(gate_cfg.get("max_candidates_per_frame", 18), 18))
        max_candidates_per_class_default = max(
            0,
            _safe_int(gate_cfg.get("max_candidates_per_class_per_frame_default", 3), 3),
        )
        max_tracking_inputs_per_video = max(
            0,
            _safe_int(
                gate_cfg.get(
                    "max_tracking_inputs_per_video",
                    budget_cfg.get("max_tracking_inputs_per_video", 320),
                ),
                320,
            ),
        )
        per_class_frame_budget_cfg = {
            str(key): max(0, _safe_int(value, max_candidates_per_class_default))
            for key, value in dict(gate_cfg.get("max_candidates_per_class_per_frame", {})).items()
        }
        accepted_suppression_iou = _safe_float(gate_cfg.get("accepted_track_suppression_iou", 0.4), 0.4)
        min_gate_score = _safe_float(gate_cfg.get("min_gate_score", 0.2), 0.2)

        filtered_active_tracks: Dict[int, Dict[str, Any]] = {}
        for track_id, track in active_tracks.items():
            if frame_index - int(track.get("last_frame_index", frame_index)) <= max_idle_frames:
                filtered_active_tracks[track_id] = track
            else:
                completed_tracks.append(track)
        active_tracks = filtered_active_tracks

        candidate_input_pool: List[Dict[str, Any]] = []
        rejected_candidates: List[Dict[str, Any]] = []
        accepted_frame_detections = _accepted_frame_detections(accepted_frame_map.get(frame_index))
        for det_index, det in enumerate(_candidate_detection_records(rec)):
            detection = dict(det)
            detection["bbox"] = _coerce_bbox(detection.get("bbox", []))
            detection["score"] = _safe_float(detection.get("score", 0.0))
            detection["raw_score"] = _raw_detection_score(detection)
            detection["calibrated_score"] = _safe_float(
                detection.get("calibrated_score", detection.get("raw_score", detection.get("score", 0.0))),
                0.0,
            )
            detection["feedback_bonus"] = _safe_float(detection.get("feedback_bonus", 0.0), 0.0)
            detection["score_used_for_candidate_ranking"] = _candidate_ranking_score(detection)
            detection["class"] = str(detection.get("class", "unknown"))
            detection["candidate_source"] = str(detection.get("candidate_source", "candidate"))
            detection["prior_metadata"] = dict(detection.get("prior_metadata", {}))
            detection["detection_id"] = str(
                detection.get("detection_id", _detection_id(frame_index, "candidate", det_index))
            )
            detection["combined_tracking_score"] = _candidate_combined_score(detection, policy)
            detection["matched_prior_ids"] = [
                str(value)
                for value in list(detection["prior_metadata"].get("matched_prior_ids", []))
                if str(value)
            ]
            detection["prior_relevance_score"] = _safe_float(
                detection["prior_metadata"].get("prior_relevance_score", 0.0)
            )
            detection["tracking_priority"] = _top_tracking_priority(detection)
            accepted_suppression = _accepted_track_suppression(detection, accepted_frame_detections)
            detection["accepted_track_suppression_iou"] = _safe_float(accepted_suppression.get("iou", 0.0))
            detection["accepted_track_suppression_track_id"] = accepted_suppression.get("track_id")
            detection["accepted_track_suppression_detection_id"] = accepted_suppression.get("detection_id")
            detection["pre_tracking_score"] = _candidate_pretracking_score(detection, orig_shape, policy)
            is_selected, rejection_reason = _candidate_selection_decision(detection, orig_shape, policy)
            detection["exploration_eligible"] = (
                not detection["matched_prior_ids"]
                and _candidate_ranking_score(detection) >= exploration_min_score
            )
            if not is_selected:
                rejected_candidates.append(_candidate_rejection_payload(detection, rejection_reason))
                rejection_reason_counts.update([rejection_reason])
                continue
            if detection["accepted_track_suppression_iou"] >= accepted_suppression_iou:
                rejected_candidates.append(
                    _candidate_rejection_payload(
                        detection,
                        "suppressed_by_accepted_track",
                    )
                )
                rejection_reason_counts.update(["suppressed_by_accepted_track"])
                continue
            if _safe_float(detection.get("pre_tracking_score", 0.0)) < min_gate_score:
                rejected_candidates.append(
                    _candidate_rejection_payload(
                        detection,
                        "below_pre_tracking_gate_score",
                    )
                )
                rejection_reason_counts.update(["below_pre_tracking_gate_score"])
                continue
            candidate_input_pool.append(detection)

        candidate_input_pool.sort(
            key=lambda row: (
                -_safe_float(row.get("pre_tracking_score", 0.0)),
                -_safe_float(row.get("combined_tracking_score", 0.0)),
                -_candidate_ranking_score(row),
            )
        )
        candidate_inputs: List[Dict[str, Any]] = []
        selected_per_class: Counter[str] = Counter()
        for detection in candidate_input_pool:
            label = str(detection.get("class", "unknown"))
            class_budget = per_class_frame_budget_cfg.get(label, max_candidates_per_class_default)
            if class_budget and selected_per_class[label] >= class_budget:
                rejected_candidates.append(
                    _candidate_rejection_payload(detection, "frame_class_input_budget_exceeded", class_budget=class_budget)
                )
                rejection_reason_counts.update(["frame_class_input_budget_exceeded"])
                continue
            if max_candidates_per_frame and len(candidate_inputs) >= max_candidates_per_frame:
                rejected_candidates.append(_candidate_rejection_payload(detection, "frame_input_budget_exceeded"))
                rejection_reason_counts.update(["frame_input_budget_exceeded"])
                continue
            if max_tracking_inputs_per_video and total_tracking_inputs >= max_tracking_inputs_per_video:
                rejected_candidates.append(_candidate_rejection_payload(detection, "video_input_budget_exceeded"))
                rejection_reason_counts.update(["video_input_budget_exceeded"])
                continue
            candidate_inputs.append(detection)
            selected_per_class.update([label])
            total_tracking_inputs += 1

        tracking_input_frames.append(
            {
                "frame": rec.get("frame", ""),
                "frame_index": frame_index,
                "image_path": rec.get("image_path", ""),
                "tracking_input_detection_ids": [str(row.get("detection_id", "")) for row in candidate_inputs],
                "num_tracking_input_candidates": len(candidate_inputs),
            }
        )

        selected_non_prior_exploration = 0
        used_track_ids: set[int] = set()

        for detection in candidate_inputs:
            label = str(detection.get("class", "unknown"))
            bbox = list(detection.get("bbox", []))
            is_non_prior = not bool(detection.get("matched_prior_ids"))
            if is_non_prior and not bool(detection.get("exploration_eligible", False)):
                rejected_candidates.append(_candidate_rejection_payload(detection, "non_prior_not_exploration_eligible"))
                rejection_reason_counts.update(["non_prior_not_exploration_eligible"])
                continue
            if is_non_prior and selected_non_prior_exploration >= exploration_quota:
                rejected_candidates.append(_candidate_rejection_payload(detection, "exploration_quota_exceeded"))
                rejection_reason_counts.update(["exploration_quota_exceeded"])
                continue
            best_track: Optional[Dict[str, Any]] = None
            best_track_id: Optional[int] = None
            best_iou = 0.0
            for track_id, track in active_tracks.items():
                if track_id in used_track_ids:
                    continue
                if frame_index - int(track.get("last_frame_index", frame_index)) > max_frame_gap:
                    continue
                if str(track.get("label", "")) != label:
                    continue
                iou = _match_iou(bbox, list(track.get("last_bbox", [])))
                if iou >= match_iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track = track
                    best_track_id = track_id

            if best_track is None:
                if _safe_float(detection.get("combined_tracking_score", 0.0)) < _safe_float(
                    threshold_cfg.get("min_new_track_combined_score", 0.45),
                    0.45,
                ):
                    rejected_candidates.append(_candidate_rejection_payload(detection, "below_new_track_combined_score"))
                    rejection_reason_counts.update(["below_new_track_combined_score"])
                    continue
                best_track_id = next_track_id
                next_track_id += 1
                best_track = {
                    "track_id": best_track_id,
                    "label": label,
                    "last_bbox": bbox,
                    "last_frame_index": frame_index,
                    "observations": [],
                    "match_ious": [],
                }
                active_tracks[best_track_id] = best_track
                best_iou = 1.0

            used_track_ids.add(int(best_track_id))
            if is_non_prior:
                selected_non_prior_exploration += 1
            observation = {
                "frame": rec.get("frame", ""),
                "frame_index": frame_index,
                "image_path": rec.get("image_path", ""),
                "bbox": bbox,
                "score": _safe_float(detection.get("score", 0.0)),
                "raw_score": _raw_detection_score(detection),
                "calibrated_score": _safe_float(detection.get("calibrated_score", detection.get("score", 0.0)), 0.0),
                "feedback_bonus": _safe_float(detection.get("feedback_bonus", 0.0), 0.0),
                "score_used_for_candidate_ranking": _candidate_ranking_score(detection),
                "label": label,
                "detection_id": str(detection.get("detection_id", "")),
                "candidate_source": str(detection.get("candidate_source", "")),
                "matched_prior_ids": list(detection.get("matched_prior_ids", [])),
                "prior_relevance_score": _safe_float(detection.get("prior_relevance_score", 0.0)),
                "combined_tracking_score": _safe_float(detection.get("combined_tracking_score", 0.0)),
                "selected_by_exploration_quota": bool(is_non_prior),
                "candidate_retention_weights": dict(
                    detection.get("prior_metadata", {}).get("matched_prior_retention_weights", {})
                ),
                "expected_predicates": dict(
                    detection.get("prior_metadata", {}).get("matched_prior_expected_predicates", {})
                ),
            }
            best_track["observations"].append(observation)
            best_track["match_ious"].append(best_iou)
            best_track["last_bbox"] = bbox
            best_track["last_frame_index"] = frame_index
        rejected_frames.append(
            {
                "frame": rec.get("frame", ""),
                "frame_index": frame_index,
                "image_path": rec.get("image_path", ""),
                "rejected_candidates": rejected_candidates,
            }
        )

    completed_tracks.extend(active_tracks.values())
    raw_tracks = [track for track in completed_tracks if track.get("observations")]
    raw_tracks = _merge_track_fragments(raw_tracks, orig_shape, policy)
    accepted_track_records = _accepted_track_records(accepted_frames)
    dedupe_cfg = dict(policy.get("candidate_track_deduplication", {}))
    selection_cfg = dict(policy.get("candidate_track_selection", {}))

    candidate_duplicate_threshold = _safe_float(
        dedupe_cfg.get("candidate_duplicate_mean_iou", 0.55),
        0.55,
    )
    candidate_duplicate_shared_ratio = _safe_float(
        dedupe_cfg.get("candidate_duplicate_shared_frame_ratio", 0.5),
        0.5,
    )
    accepted_duplicate_threshold = _safe_float(
        dedupe_cfg.get("accepted_duplicate_mean_iou", 0.6),
        0.6,
    )
    accepted_duplicate_shared_ratio = _safe_float(
        dedupe_cfg.get("accepted_duplicate_shared_frame_ratio", 0.5),
        0.5,
    )
    dedupe_time_bucket_size = max(1, _safe_int(dedupe_cfg.get("time_bucket_size", 12), 12))
    dedupe_grid_size = list(dedupe_cfg.get("grid_size", [4, 3]))
    dedupe_center_distance_threshold = _safe_float(
        dedupe_cfg.get("center_distance_threshold_ratio", 0.12),
        0.12,
    )

    prefiltered_tracks: List[Dict[str, Any]] = []
    rejected_track_entries: List[Dict[str, Any]] = []
    rejected_track_reason_counts: Counter[str] = Counter()

    for track in raw_tracks:
        best_accepted_overlap: Dict[str, Any] = {}
        for accepted_track in accepted_track_records:
            if str(accepted_track.get("label", "")) != str(track.get("label", "")):
                continue
            overlap = _track_overlap_metrics(track, accepted_track)
            if _safe_float(overlap.get("overlap_score", 0.0)) > _safe_float(
                best_accepted_overlap.get("overlap_score", 0.0)
            ):
                best_accepted_overlap = {
                    **overlap,
                    "accepted_track_id": int(accepted_track.get("track_id", -1)),
                }
        track["duplicate_with_accepted"] = best_accepted_overlap
        track["duplicate_with_candidate"] = {}
        track["score_breakdown"] = _candidate_track_score_breakdown(track, orig_shape, policy)
        if (
            best_accepted_overlap
            and _safe_float(best_accepted_overlap.get("mean_iou", 0.0)) >= accepted_duplicate_threshold
            and _safe_float(best_accepted_overlap.get("shared_frame_ratio", 0.0)) >= accepted_duplicate_shared_ratio
        ):
            rejected_track_entries.append(
                _candidate_track_rejection_entry(
                    track,
                    "duplicate_of_accepted_track",
                    duplicate_track_id=int(best_accepted_overlap.get("accepted_track_id", -1)),
                )
            )
            rejected_track_reason_counts.update(["duplicate_of_accepted_track"])
            continue
        prefiltered_tracks.append(track)

    max_raw_tracks = max(0, _safe_int(budget_cfg.get("max_raw_tracks", 160), 160))
    if max_raw_tracks and len(prefiltered_tracks) > max_raw_tracks:
        prefiltered_tracks.sort(
            key=lambda track: (
                -_safe_float(dict(track.get("score_breakdown", {})).get("selection_score", 0.0)),
                -len(list(track.get("observations", []))),
                int(track.get("track_id", -1)),
            )
        )
        kept_prefiltered = prefiltered_tracks[:max_raw_tracks]
        overflow_prefiltered = prefiltered_tracks[max_raw_tracks:]
        for track in overflow_prefiltered:
            rejected_track_entries.append(_candidate_track_rejection_entry(track, "raw_track_budget_exceeded"))
            rejected_track_reason_counts.update(["raw_track_budget_exceeded"])
        prefiltered_tracks = kept_prefiltered

    prefiltered_tracks.sort(
        key=lambda track: (
            -_safe_float(dict(track.get("score_breakdown", {})).get("selection_score", 0.0)),
            -len(list(track.get("observations", []))),
            int(track.get("track_id", -1)),
        )
    )

    deduplicated_tracks: List[Dict[str, Any]] = []
    dedupe_bucket_index: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for track in prefiltered_tracks:
        best_candidate_overlap: Dict[str, Any] = {}
        candidate_neighbors: List[Dict[str, Any]] = []
        seen_neighbor_ids: set[int] = set()
        for bucket_key in _track_bucket_keys(track, orig_shape, dedupe_time_bucket_size, dedupe_grid_size):
            for kept_track in dedupe_bucket_index.get(bucket_key, []):
                kept_track_id = int(kept_track.get("track_id", -1))
                if kept_track_id in seen_neighbor_ids:
                    continue
                seen_neighbor_ids.add(kept_track_id)
                candidate_neighbors.append(kept_track)
        for kept_track in candidate_neighbors:
            if str(kept_track.get("label", "")) != str(track.get("label", "")):
                continue
            overlap = _track_overlap_metrics(track, kept_track)
            last_box = list(kept_track.get("observations", [{}])[-1].get("bbox", [])) if kept_track.get("observations") else []
            current_box = list(track.get("observations", [{}])[0].get("bbox", [])) if track.get("observations") else []
            center_distance = _center_distance_ratio(last_box, current_box, orig_shape) if len(last_box) >= 4 and len(current_box) >= 4 else 1.0
            overlap["center_distance_ratio"] = float(center_distance)
            if _safe_float(overlap.get("overlap_score", 0.0)) > _safe_float(
                best_candidate_overlap.get("overlap_score", 0.0)
            ):
                best_candidate_overlap = {
                    **overlap,
                    "candidate_track_id": int(kept_track.get("track_id", -1)),
                }
        track["duplicate_with_candidate"] = best_candidate_overlap
        track["score_breakdown"] = _candidate_track_score_breakdown(track, orig_shape, policy)
        if (
            best_candidate_overlap
            and _safe_float(best_candidate_overlap.get("mean_iou", 0.0)) >= candidate_duplicate_threshold
            and _safe_float(best_candidate_overlap.get("shared_frame_ratio", 0.0)) >= candidate_duplicate_shared_ratio
            and _safe_float(best_candidate_overlap.get("center_distance_ratio", 0.0)) <= dedupe_center_distance_threshold
        ):
            rejected_track_entries.append(
                _candidate_track_rejection_entry(
                    track,
                    "duplicate_of_candidate_track",
                    duplicate_track_id=int(best_candidate_overlap.get("candidate_track_id", -1)),
                )
            )
            rejected_track_reason_counts.update(["duplicate_of_candidate_track"])
            continue
        deduplicated_tracks.append(track)
        for bucket_key in _track_bucket_keys(track, orig_shape, dedupe_time_bucket_size, dedupe_grid_size):
            dedupe_bucket_index[bucket_key].append(track)

    max_deduplicated_tracks = max(0, _safe_int(budget_cfg.get("max_deduplicated_tracks", 64), 64))
    if max_deduplicated_tracks and len(deduplicated_tracks) > max_deduplicated_tracks:
        deduplicated_tracks.sort(
            key=lambda track: (
                -_safe_float(dict(track.get("score_breakdown", {})).get("selection_score", 0.0)),
                -_safe_float(dict(track.get("score_breakdown", {})).get("temporal_consistency", 0.0)),
                int(track.get("track_id", -1)),
            )
        )
        kept_deduplicated = deduplicated_tracks[:max_deduplicated_tracks]
        overflow_deduplicated = deduplicated_tracks[max_deduplicated_tracks:]
        for track in overflow_deduplicated:
            rejected_track_entries.append(_candidate_track_rejection_entry(track, "deduplicated_track_budget_exceeded"))
            rejected_track_reason_counts.update(["deduplicated_track_budget_exceeded"])
        deduplicated_tracks = kept_deduplicated

    for track in deduplicated_tracks:
        track["score_breakdown"] = _candidate_track_score_breakdown(track, orig_shape, policy)

    deduplicated_tracks.sort(
        key=lambda track: (
            -_safe_float(dict(track.get("score_breakdown", {})).get("selection_score", 0.0)),
            -_safe_float(dict(track.get("score_breakdown", {})).get("temporal_consistency", 0.0)),
            -_safe_float(dict(track.get("score_breakdown", {})).get("prior_relevance_mean", 0.0)),
            int(track.get("track_id", -1)),
        )
    )

    max_tracks_per_video = max(
        0,
        _safe_int(
            budget_cfg.get("max_selected_tracks", budget_cfg.get("max_tracks_per_video", 24)),
            24,
        ),
    )
    max_tracks_per_class_default = max(0, _safe_int(budget_cfg.get("max_tracks_per_class_default", 4), 4))
    per_class_budget_cfg = {
        str(key): max(0, _safe_int(value, max_tracks_per_class_default))
        for key, value in dict(budget_cfg.get("max_tracks_per_class", {})).items()
    }
    min_selection_score = _safe_float(selection_cfg.get("min_selection_score", 0.15), 0.15)
    selected_tracks: List[Dict[str, Any]] = []
    selected_counts_by_class: Counter[str] = Counter()
    for track in deduplicated_tracks:
        label = str(track.get("label", "unknown"))
        selection_score = _safe_float(dict(track.get("score_breakdown", {})).get("selection_score", 0.0))
        if selection_score < min_selection_score:
            rejected_track_entries.append(_candidate_track_rejection_entry(track, "below_track_selection_score"))
            rejected_track_reason_counts.update(["below_track_selection_score"])
            continue
        if max_tracks_per_video and len(selected_tracks) >= max_tracks_per_video:
            rejected_track_entries.append(_candidate_track_rejection_entry(track, "video_budget_exceeded"))
            rejected_track_reason_counts.update(["video_budget_exceeded"])
            continue
        class_budget = per_class_budget_cfg.get(label, max_tracks_per_class_default)
        if class_budget and selected_counts_by_class[label] >= class_budget:
            rejected_track_entries.append(
                _candidate_track_rejection_entry(
                    track,
                    "class_budget_exceeded",
                    class_budget=class_budget,
                )
            )
            rejected_track_reason_counts.update(["class_budget_exceeded"])
            continue
        selected_tracks.append(track)
        selected_counts_by_class.update([label])

    rejected_detections_bundle = {
        "num_rejected": sum(len(frame.get("rejected_candidates", [])) for frame in rejected_frames),
        "frames": rejected_frames,
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "selection_policy": policy,
    }
    rejected_track_entries.sort(
        key=lambda row: (
            row.get("rejection_reason", ""),
            -_safe_float(row.get("selection_score", 0.0)),
            int(row.get("track_id", -1)),
        )
    )
    return {
        "num_tracking_input_candidates": total_tracking_inputs,
        "tracking_input_candidates": {
            "num_candidates": total_tracking_inputs,
            "frames": tracking_input_frames,
        },
        "raw_candidate_tracks": {
            "num_tracks": len(raw_tracks),
            "frames": _tracks_to_frame_records(raw_tracks),
            "track_summaries": sorted(
                (_candidate_summary_from_track(track) for track in raw_tracks),
                key=lambda row: int(row.get("track_id", -1)),
            ),
        },
        "deduplicated_candidate_tracks": {
            "num_tracks": len(deduplicated_tracks),
            "frames": _tracks_to_frame_records(deduplicated_tracks),
            "track_summaries": [
                _candidate_summary_from_track(track) for track in deduplicated_tracks
            ],
        },
        "selected_candidate_tracks": {
            "num_tracks": len(selected_tracks),
            "frames": _tracks_to_frame_records(selected_tracks),
            "track_summaries": [
                _candidate_summary_from_track(track) for track in selected_tracks
            ],
        },
        "rejected_candidate_tracks": {
            "num_tracks": len(rejected_track_entries),
            "tracks": rejected_track_entries,
            "rejection_reason_counts": dict(sorted(rejected_track_reason_counts.items())),
        },
        "rejected_candidate_detections": rejected_detections_bundle,
        "selection_policy": policy,
        "selection_budgets": {
            "max_tracking_inputs_per_video": max(
                0,
                _safe_int(
                    gate_cfg.get(
                        "max_tracking_inputs_per_video",
                        budget_cfg.get("max_tracking_inputs_per_video", 320),
                    ),
                    320,
                ),
            ),
            "max_raw_tracks": max_raw_tracks,
            "max_deduplicated_tracks": max_deduplicated_tracks,
            "max_selected_tracks": max_tracks_per_video,
            "max_tracks_per_video": max_tracks_per_video,
            "max_tracks_per_class_default": max_tracks_per_class_default,
            "max_tracks_per_class": dict(sorted(per_class_budget_cfg.items())),
            "selected_counts_by_class": dict(sorted(selected_counts_by_class.items())),
        },
    }

def track_video(
    video_result: Dict[str, Any],
    output_root: Optional[Path] = None,
    frame_rate: int = 10,
    tracker_args: Optional[SimpleNamespace] = None,
    force_recompute: bool = False,
    render_video: bool = True,
) -> Dict[str, Any]:
    """Run ByteTrack on a single video's detection records.

    Args:
        video_result:   One entry from detect_driving_mini.run() output.
        output_root:    Root directory for persisting tracks.json.
        frame_rate:     Video frame rate passed to the tracker.
        tracker_args:   Override tracker hyper-parameters.
        force_recompute: Recompute even if tracks.json already exists.

    Returns:
        Tracking result dict (see module docstring).
    """
    video_id: str = video_result["video_id"]
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_file = out_dir / "tracks.json"
    cache_status = "force_recompute" if force_recompute else "missing"

    if not force_recompute and tracks_file.exists():
        cache_status = "invalid"
        try:
            with tracks_file.open("r", encoding="utf-8") as fh:
                cached = json.load(fh)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            print(f"[warn] Ignoring invalid tracking cache for {video_id}: {exc}")
            cached = {}
        cache_is_current = (
            int(cached.get("schema_version", 1)) >= _TRACKING_SCHEMA_VERSION
            and isinstance(cached.get("accepted_tracks"), dict)
            and isinstance(cached.get("candidate_tracks"), dict)
        )
        if not cache_is_current:
            cache_status = "stale_schema_or_shape"
        current_policy_id = str(dict(video_result.get("od_calibration", {})).get("policy_id", ""))
        cached_policy_id = str(cached.get("od_calibration_policy_id", ""))
        if cache_is_current and cached_policy_id != current_policy_id:
            cache_is_current = False
            cache_status = "stale_od_policy"
        if cache_is_current and bool(cached.get("candidate_branch_enabled", True)) != _candidate_branch_enabled(video_result):
            cache_is_current = False
            cache_status = "stale_candidate_branch"
        if not cache_is_current:
            cached = {}
        else:
            cache_status = "hit"
            tracked_video_file = out_dir / "tracks_boxed.mp4"
            if render_video and not tracked_video_file.exists() and cached.get("frames"):
                render_path = render_tracking_video(
                    video_id=video_id,
                    tracked_frames=cached["frames"],
                    output_path=tracked_video_file,
                    fps=float(frame_rate),
                )
                if render_path:
                    cached["tracked_video_path"] = render_path
                    _write_json_atomic(tracks_file, cached)
            cached["from_cache"] = True
            cached["tracking_cache_status"] = cache_status
            return cached

    frame_records: List[Dict[str, Any]] = video_result.get("frames", [])
    orig_shape = _infer_orig_shape(frame_records)
    candidate_scoring_policy = _load_candidate_scoring_policy(video_result)
    accepted_frames, accepted_track_ids = _run_accepted_tracking(
        frame_records=frame_records,
        orig_shape=orig_shape,
        frame_rate=frame_rate,
        tracker_args=tracker_args,
    )
    if _candidate_branch_enabled(video_result):
        candidate_track_bundle = _run_candidate_tracking(
            frame_records=frame_records,
            accepted_frames=accepted_frames,
            orig_shape=orig_shape,
            policy=candidate_scoring_policy,
        )
    else:
        candidate_track_bundle = _empty_candidate_track_bundle(candidate_scoring_policy)
    selected_candidate_tracks = dict(candidate_track_bundle.get("selected_candidate_tracks", {}))
    rejected_candidate_detections = dict(candidate_track_bundle.get("rejected_candidate_detections", {}))
    raw_candidate_tracks = dict(candidate_track_bundle.get("raw_candidate_tracks", {}))
    deduplicated_candidate_tracks = dict(candidate_track_bundle.get("deduplicated_candidate_tracks", {}))
    rejected_candidate_tracks = dict(candidate_track_bundle.get("rejected_candidate_tracks", {}))
    tracking_input_candidates = dict(candidate_track_bundle.get("tracking_input_candidates", {}))

    result: Dict[str, Any] = {
        "schema_version": _TRACKING_SCHEMA_VERSION,
        "video_id": video_id,
        "from_cache": False,
        "tracking_cache_status": cache_status,
        "candidate_branch_enabled": _candidate_branch_enabled(video_result),
        "od_calibration_policy_id": str(dict(video_result.get("od_calibration", {})).get("policy_id", "")),
        "num_frames": len(accepted_frames),
        "num_tracks": len(accepted_track_ids),
        "num_tracking_input_candidate_detections": int(candidate_track_bundle.get("num_tracking_input_candidates", 0)),
        "num_candidate_tracks": int(selected_candidate_tracks.get("num_tracks", 0)),
        "num_raw_candidate_tracks": int(raw_candidate_tracks.get("num_tracks", 0)),
        "num_deduplicated_candidate_tracks": int(deduplicated_candidate_tracks.get("num_tracks", 0)),
        "num_rejected_candidate_tracks": int(rejected_candidate_tracks.get("num_tracks", 0)),
        "num_rejected_candidate_detections": int(rejected_candidate_detections.get("num_rejected", 0)),
        "frames": accepted_frames,
        "accepted_tracks": {
            "num_tracks": len(accepted_track_ids),
            "frames": accepted_frames,
            "track_summaries": _accepted_track_summaries(accepted_frames),
        },
        "candidate_tracks": {
            "num_tracks": int(selected_candidate_tracks.get("num_tracks", 0)),
            "frames": list(selected_candidate_tracks.get("frames", [])),
            "track_summaries": list(selected_candidate_tracks.get("track_summaries", [])),
            "tracking_input_candidates": tracking_input_candidates,
            "raw_candidate_tracks": raw_candidate_tracks,
            "deduplicated_candidate_tracks": deduplicated_candidate_tracks,
            "selected_candidate_tracks": selected_candidate_tracks,
            "rejected_candidate_tracks": rejected_candidate_tracks,
            "rejected_candidate_detections": rejected_candidate_detections,
            "selection_policy": candidate_scoring_policy,
            "selection_budgets": dict(candidate_track_bundle.get("selection_budgets", {})),
        },
        "output_paths": {
            "tracks_json": str(tracks_file),
        },
    }

    _write_json_atomic(tracks_file, result)

    # Render annotated video with track IDs
    tracked_video_file = out_dir / "tracks_boxed.mp4"
    if render_video:
        render_path = render_tracking_video(
            video_id=video_id,
            tracked_frames=accepted_frames,
            output_path=tracked_video_file,
            fps=float(frame_rate),
        )
        if render_path:
            result["tracked_video_path"] = render_path
            _write_json_atomic(tracks_file, result)
    return result


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(
    detection_results: List[Dict[str, Any]],
    output_root: Optional[Path] = None,
    frame_rate: int = 10,
    tracker_args: Optional[SimpleNamespace] = None,
    force_recompute: bool = False,
    render_video: bool = True,
) -> List[Dict[str, Any]]:
    """Run ByteTrack over all videos in detection_results.

    Args:
        detection_results: Output of detect_driving_mini.run() — a list of
                           per-video detection dicts.
        output_root:       Override default output root.
        frame_rate:        Frame rate of the source videos.
        tracker_args:      Override default ByteTrack hyper-parameters.
        force_recompute:   Ignore cached tracks.json files and recompute.

    Returns:
        List of per-video tracking result dicts.
    """
    effective_render_video = bool(render_video)
    if effective_render_video and cv2 is None:
        print(
            "[warn] OpenCV (`cv2`) is not available; tracking rendering will be skipped for this run. "
            f"Import error: {_format_dependency_error(_CV2_IMPORT_ERROR)}"
        )
        effective_render_video = False
    ensure_tracking_runtime_available()
    effective_output_root = output_root or get_output_root()

    print(f"Videos to track: {[r['video_id'] for r in detection_results]}")

    tracking_results: List[Dict[str, Any]] = []
    total_videos = len(detection_results)
    for index, video_result in enumerate(detection_results, start=1):
        video_id = str(video_result.get("video_id", ""))
        print(f"Tracking progress: {index}/{total_videos} | {video_id} | starting")
        result = track_video(
            video_result=video_result,
            output_root=effective_output_root,
            frame_rate=frame_rate,
            tracker_args=tracker_args,
            force_recompute=force_recompute,
            render_video=effective_render_video,
        )
        tracking_results.append(result)
        cache_tag = "cached" if bool(result.get("from_cache", False)) else "recomputed"
        print(
            "Tracking progress: "
            f"{index}/{total_videos} | {video_id} | {cache_tag} "
            f"status={result.get('tracking_cache_status', '')}"
        )

    # Write summary manifest
    manifest = {
        "schema_version": _TRACKING_SCHEMA_VERSION,
        "candidate_branch_enabled": all(bool(r.get("candidate_branch_enabled", True)) for r in tracking_results),
        "num_videos": len(tracking_results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r["num_frames"],
                "num_tracks": r["num_tracks"],
                "num_tracking_input_candidate_detections": int(r.get("num_tracking_input_candidate_detections", 0)),
                "num_candidate_tracks": int(r.get("num_candidate_tracks", 0)),
                "num_raw_candidate_tracks": int(r.get("num_raw_candidate_tracks", 0)),
                "num_deduplicated_candidate_tracks": int(r.get("num_deduplicated_candidate_tracks", 0)),
                "num_rejected_candidate_tracks": int(r.get("num_rejected_candidate_tracks", 0)),
                "num_rejected_candidate_detections": int(r.get("num_rejected_candidate_detections", 0)),
            }
            for r in tracking_results
        ],
    }
    manifest_path = effective_output_root / "tracks_manifest.json"
    _write_json_atomic(manifest_path, manifest)

    total_tracks = sum(r["num_tracks"] for r in tracking_results)
    total_tracking_input_candidates = sum(
        int(r.get("num_tracking_input_candidate_detections", 0)) for r in tracking_results
    )
    total_raw_candidate_tracks = sum(int(r.get("num_raw_candidate_tracks", 0)) for r in tracking_results)
    total_deduplicated_candidate_tracks = sum(
        int(r.get("num_deduplicated_candidate_tracks", 0)) for r in tracking_results
    )
    total_candidate_tracks = sum(int(r.get("num_candidate_tracks", 0)) for r in tracking_results)
    total_rejected_candidate_tracks = sum(
        int(r.get("num_rejected_candidate_tracks", 0)) for r in tracking_results
    )
    total_rejected_candidate_detections = sum(
        int(r.get("num_rejected_candidate_detections", 0))
        for r in tracking_results
    )
    print(f"\nSaved tracking manifest to {manifest_path}")
    print(f"Total unique tracks across all videos: {total_tracks}")
    print(f"Total tracking-input candidate detections across all videos: {total_tracking_input_candidates}")
    print(f"Total raw candidate tracks across all videos: {total_raw_candidate_tracks}")
    print(f"Total deduplicated candidate tracks across all videos: {total_deduplicated_candidate_tracks}")
    print(f"Total selected candidate tracks across all videos: {total_candidate_tracks}")
    print(f"Total rejected candidate tracks across all videos: {total_rejected_candidate_tracks}")
    print(f"Total rejected candidate detections across all videos: {total_rejected_candidate_detections}")
    return tracking_results
