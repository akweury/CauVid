"""
Prior-guided candidate-aware multi-object tracking over driving_mini detections.

Accepted detections keep the existing ByteTrack-based tracking behavior.
Candidate detections are tracked in a separate lightweight branch using detector
score, Step 0 prior metadata, and simple spatial/temporal plausibility gates.

Output layout:
    pipeline_output/02_driving_mini_tracking/<video_id>/
        tracks.json          — per-frame accepted/candidate tracking results
        tracks_manifest.json — summary manifest (written by run())
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
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
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ultralytics is required for ByteTrack tracking. "
        "Install with: pip install ultralytics"
    ) from exc


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
_TRACKING_SCHEMA_VERSION = 3
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
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "02_driving_mini_tracking"
    out.mkdir(parents=True, exist_ok=True)
    return out


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


def _load_candidate_scoring_policy(video_result: Dict[str, Any]) -> Dict[str, Any]:
    policy = json.loads(json.dumps(_DEFAULT_CANDIDATE_SCORING_POLICY))
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
        _safe_float(policy.get("visual_score_weight", 1.0), 1.0) * _safe_float(detection.get("score", 0.0))
        + _safe_float(policy.get("prior_relevance_weight", 0.35), 0.35) * _safe_float(prior_metadata.get("prior_relevance_score", 0.0))
        + _safe_float(source_bonus.get(str(detection.get("candidate_source", "")), 0.0))
        + _safe_float(priority_bonus.get(str(top_priority), 0.0))
        + _safe_float(policy.get("retention_bias_weight", 0.2), 0.2) * _retention_bias(detection)
    )


def _candidate_selection_decision(
    detection: Dict[str, Any],
    orig_shape: Tuple[int, int],
    policy: Dict[str, Any],
) -> Tuple[bool, str]:
    score = _safe_float(detection.get("score", 0.0))
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
    return {
        "track_id": int(track.get("track_id", -1)),
        "label": label_counts.most_common(1)[0][0] if label_counts else "unknown",
        "source_detection_ids": detection_ids,
        "candidate_sources": sorted(set(candidate_sources)),
        "candidate_source_counts": dict(sorted(Counter(candidate_sources).items())),
        "mean_score": float(sum(scores) / max(1, len(scores))),
        "max_score": float(max(scores) if scores else 0.0),
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
    orig_shape: Tuple[int, int],
    policy: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    active_tracks: Dict[int, Dict[str, Any]] = {}
    completed_tracks: List[Dict[str, Any]] = []
    candidate_frames: List[Dict[str, Any]] = []
    rejected_frames: List[Dict[str, Any]] = []
    rejection_reason_counts: Counter[str] = Counter()
    next_track_id = _CANDIDATE_TRACK_ID_OFFSET
    for rec in frame_records:
        frame_index = int(rec.get("frame_index", -1))
        temporal_cfg = dict(policy.get("temporal_consistency_gate", {}))
        threshold_cfg = dict(policy.get("candidate_selection_thresholds", {}))
        exploration_cfg = dict(policy.get("exploration_quota", {}))
        max_idle_frames = _safe_int(temporal_cfg.get("max_idle_frames", 2), 2)
        max_frame_gap = _safe_int(temporal_cfg.get("max_frame_gap", 1), 1)
        match_iou_threshold = _safe_float(temporal_cfg.get("match_iou_threshold", 0.3), 0.3)
        exploration_quota = max(0, _safe_int(exploration_cfg.get("non_prior_high_score_candidates_per_frame", 1), 1))
        exploration_min_score = _safe_float(exploration_cfg.get("non_prior_high_score_min_score", 0.35), 0.35)

        filtered_active_tracks: Dict[int, Dict[str, Any]] = {}
        for track_id, track in active_tracks.items():
            if frame_index - int(track.get("last_frame_index", frame_index)) <= max_idle_frames:
                filtered_active_tracks[track_id] = track
            else:
                completed_tracks.append(track)
        active_tracks = filtered_active_tracks

        candidate_inputs: List[Dict[str, Any]] = []
        rejected_candidates: List[Dict[str, Any]] = []
        for det_index, det in enumerate(_candidate_detection_records(rec)):
            detection = dict(det)
            detection["bbox"] = _coerce_bbox(detection.get("bbox", []))
            detection["score"] = _safe_float(detection.get("score", 0.0))
            detection["class"] = str(detection.get("class", "unknown"))
            detection["candidate_source"] = str(detection.get("candidate_source", "candidate"))
            detection["prior_metadata"] = dict(detection.get("prior_metadata", {}))
            detection["detection_id"] = _detection_id(frame_index, "candidate", det_index)
            detection["combined_tracking_score"] = _candidate_combined_score(detection, policy)
            detection["matched_prior_ids"] = [
                str(value)
                for value in list(detection["prior_metadata"].get("matched_prior_ids", []))
                if str(value)
            ]
            detection["prior_relevance_score"] = _safe_float(
                detection["prior_metadata"].get("prior_relevance_score", 0.0)
            )
            is_selected, rejection_reason = _candidate_selection_decision(detection, orig_shape, policy)
            detection["exploration_eligible"] = (
                not detection["matched_prior_ids"]
                and detection["score"] >= exploration_min_score
            )
            if is_selected:
                candidate_inputs.append(detection)
            else:
                rejected_candidates.append(
                    {
                        "detection_id": str(detection.get("detection_id", "")),
                        "bbox": list(detection.get("bbox", [])),
                        "class": str(detection.get("class", "unknown")),
                        "score": _safe_float(detection.get("score", 0.0)),
                        "candidate_source": str(detection.get("candidate_source", "")),
                        "combined_tracking_score": _safe_float(detection.get("combined_tracking_score", 0.0)),
                        "matched_prior_ids": list(detection.get("matched_prior_ids", [])),
                        "prior_relevance_score": _safe_float(detection.get("prior_relevance_score", 0.0)),
                        "prior_metadata": dict(detection.get("prior_metadata", {})),
                        "rejection_reason": rejection_reason,
                    }
                )
                rejection_reason_counts.update([rejection_reason])

        candidate_inputs.sort(
            key=lambda row: (
                -_safe_float(row.get("combined_tracking_score", 0.0)),
                -_safe_float(row.get("score", 0.0)),
            )
        )

        selected_non_prior_exploration = 0
        used_track_ids: set[int] = set()
        frame_candidate_boxes: List[List[float]] = []
        frame_candidate_scores: List[float] = []
        frame_candidate_labels: List[str] = []
        frame_candidate_track_ids: List[int] = []
        frame_candidate_detection_ids: List[str] = []
        frame_candidate_sources: List[str] = []
        frame_candidate_prior_scores: List[float] = []
        frame_candidate_prior_ids: List[List[str]] = []

        for detection in candidate_inputs:
            label = str(detection.get("class", "unknown"))
            bbox = list(detection.get("bbox", []))
            is_non_prior = not bool(detection.get("matched_prior_ids"))
            if is_non_prior and not bool(detection.get("exploration_eligible", False)):
                rejected_candidates.append(
                    {
                        "detection_id": str(detection.get("detection_id", "")),
                        "bbox": bbox,
                        "class": label,
                        "score": _safe_float(detection.get("score", 0.0)),
                        "candidate_source": str(detection.get("candidate_source", "")),
                        "combined_tracking_score": _safe_float(detection.get("combined_tracking_score", 0.0)),
                        "matched_prior_ids": [],
                        "prior_relevance_score": 0.0,
                        "prior_metadata": dict(detection.get("prior_metadata", {})),
                        "rejection_reason": "non_prior_not_exploration_eligible",
                    }
                )
                rejection_reason_counts.update(["non_prior_not_exploration_eligible"])
                continue
            if is_non_prior and selected_non_prior_exploration >= exploration_quota:
                rejected_candidates.append(
                    {
                        "detection_id": str(detection.get("detection_id", "")),
                        "bbox": bbox,
                        "class": label,
                        "score": _safe_float(detection.get("score", 0.0)),
                        "candidate_source": str(detection.get("candidate_source", "")),
                        "combined_tracking_score": _safe_float(detection.get("combined_tracking_score", 0.0)),
                        "matched_prior_ids": [],
                        "prior_relevance_score": 0.0,
                        "prior_metadata": dict(detection.get("prior_metadata", {})),
                        "rejection_reason": "exploration_quota_exceeded",
                    }
                )
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
                    rejected_candidates.append(
                        {
                            "detection_id": str(detection.get("detection_id", "")),
                            "bbox": bbox,
                            "class": label,
                            "score": _safe_float(detection.get("score", 0.0)),
                            "candidate_source": str(detection.get("candidate_source", "")),
                            "combined_tracking_score": _safe_float(detection.get("combined_tracking_score", 0.0)),
                            "matched_prior_ids": list(detection.get("matched_prior_ids", [])),
                            "prior_relevance_score": _safe_float(detection.get("prior_relevance_score", 0.0)),
                            "prior_metadata": dict(detection.get("prior_metadata", {})),
                            "rejection_reason": "below_new_track_combined_score",
                        }
                    )
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
            elif best_iou < match_iou_threshold:
                rejected_candidates.append(
                    {
                        "detection_id": str(detection.get("detection_id", "")),
                        "bbox": bbox,
                        "class": label,
                        "score": _safe_float(detection.get("score", 0.0)),
                        "candidate_source": str(detection.get("candidate_source", "")),
                        "combined_tracking_score": _safe_float(detection.get("combined_tracking_score", 0.0)),
                        "matched_prior_ids": list(detection.get("matched_prior_ids", [])),
                        "prior_relevance_score": _safe_float(detection.get("prior_relevance_score", 0.0)),
                        "prior_metadata": dict(detection.get("prior_metadata", {})),
                        "rejection_reason": "temporal_match_below_threshold",
                    }
                )
                rejection_reason_counts.update(["temporal_match_below_threshold"])
                continue

            used_track_ids.add(int(best_track_id))
            if is_non_prior:
                selected_non_prior_exploration += 1
            observation = {
                "frame": rec.get("frame", ""),
                "frame_index": frame_index,
                "image_path": rec.get("image_path", ""),
                "bbox": bbox,
                "score": _safe_float(detection.get("score", 0.0)),
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

            frame_candidate_boxes.append(bbox)
            frame_candidate_scores.append(_safe_float(detection.get("score", 0.0)))
            frame_candidate_labels.append(label)
            frame_candidate_track_ids.append(int(best_track_id))
            frame_candidate_detection_ids.append(str(detection.get("detection_id", "")))
            frame_candidate_sources.append(str(detection.get("candidate_source", "")))
            frame_candidate_prior_scores.append(_safe_float(detection.get("prior_relevance_score", 0.0)))
            frame_candidate_prior_ids.append(list(detection.get("matched_prior_ids", [])))

        candidate_frames.append(
            {
                "frame": rec.get("frame", ""),
                "frame_index": frame_index,
                "image_path": rec.get("image_path", ""),
                "boxes": frame_candidate_boxes,
                "scores": frame_candidate_scores,
                "labels": frame_candidate_labels,
                "track_ids": frame_candidate_track_ids,
                "detection_ids": frame_candidate_detection_ids,
                "candidate_sources": frame_candidate_sources,
                "prior_relevance_scores": frame_candidate_prior_scores,
                "matched_prior_ids": frame_candidate_prior_ids,
            }
        )
        rejected_frames.append(
            {
                "frame": rec.get("frame", ""),
                "frame_index": frame_index,
                "image_path": rec.get("image_path", ""),
                "rejected_candidates": rejected_candidates,
            }
        )

    completed_tracks.extend(active_tracks.values())
    candidate_track_summaries = sorted(
        (_candidate_summary_from_track(track) for track in completed_tracks if track.get("observations")),
        key=lambda row: (int(row.get("track_id", -1))),
    )
    rejected_bundle = {
        "num_rejected": sum(len(frame.get("rejected_candidates", [])) for frame in rejected_frames),
        "frames": rejected_frames,
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "selection_policy": policy,
    }
    return candidate_frames, candidate_track_summaries, rejected_bundle

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

    if not force_recompute and tracks_file.exists():
        with tracks_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        cache_is_current = (
            int(cached.get("schema_version", 1)) >= _TRACKING_SCHEMA_VERSION
            and isinstance(cached.get("accepted_tracks"), dict)
            and isinstance(cached.get("candidate_tracks"), dict)
        )
        if not cache_is_current:
            cached = {}
        else:
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
                    with tracks_file.open("w", encoding="utf-8") as fh:
                        json.dump(cached, fh, indent=2)
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
    candidate_frames, candidate_track_summaries, rejected_candidate_detections = _run_candidate_tracking(
        frame_records=frame_records,
        orig_shape=orig_shape,
        policy=candidate_scoring_policy,
    )

    result: Dict[str, Any] = {
        "schema_version": _TRACKING_SCHEMA_VERSION,
        "video_id": video_id,
        "num_frames": len(accepted_frames),
        "num_tracks": len(accepted_track_ids),
        "num_candidate_tracks": len(candidate_track_summaries),
        "num_rejected_candidate_detections": int(rejected_candidate_detections.get("num_rejected", 0)),
        "frames": accepted_frames,
        "accepted_tracks": {
            "num_tracks": len(accepted_track_ids),
            "frames": accepted_frames,
            "track_summaries": _accepted_track_summaries(accepted_frames),
        },
        "candidate_tracks": {
            "num_tracks": len(candidate_track_summaries),
            "frames": candidate_frames,
            "track_summaries": candidate_track_summaries,
            "rejected_candidate_detections": rejected_candidate_detections,
            "selection_policy": candidate_scoring_policy,
        },
        "output_paths": {
            "tracks_json": str(tracks_file),
        },
    }

    with tracks_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

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
            with tracks_file.open("w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
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
    effective_output_root = output_root or get_output_root()

    print(f"Videos to track: {[r['video_id'] for r in detection_results]}")

    tracking_results: List[Dict[str, Any]] = []
    for video_result in detection_results:
        result = track_video(
            video_result=video_result,
            output_root=effective_output_root,
            frame_rate=frame_rate,
            tracker_args=tracker_args,
            force_recompute=force_recompute,
            render_video=render_video,
        )
        tracking_results.append(result)

    # Write summary manifest
    manifest = {
        "schema_version": _TRACKING_SCHEMA_VERSION,
        "num_videos": len(tracking_results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r["num_frames"],
                "num_tracks": r["num_tracks"],
                "num_candidate_tracks": int(r.get("num_candidate_tracks", 0)),
                "num_rejected_candidate_detections": int(r.get("num_rejected_candidate_detections", 0)),
            }
            for r in tracking_results
        ],
    }
    manifest_path = effective_output_root / "tracks_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    total_tracks = sum(r["num_tracks"] for r in tracking_results)
    total_candidate_tracks = sum(int(r.get("num_candidate_tracks", 0)) for r in tracking_results)
    total_rejected_candidate_detections = sum(
        int(r.get("num_rejected_candidate_detections", 0))
        for r in tracking_results
    )
    print(f"\nSaved tracking manifest to {manifest_path}")
    print(f"Total unique tracks across all videos: {total_tracks}")
    print(f"Total candidate tracks across all videos: {total_candidate_tracks}")
    print(f"Total rejected candidate detections across all videos: {total_rejected_candidate_detections}")
    return tracking_results
