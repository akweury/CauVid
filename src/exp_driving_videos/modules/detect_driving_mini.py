"""
YOLO-World object detection over the driving_mini dataset.

The driving_mini dataset has pre-extracted frames stored as:
    dataset/driving_mini/frames/<video_id>/frame_NNNNN.jpg

This script runs YOLO-World on every frame of every video clip and writes
per-video JSON results alongside a summary manifest.

Output layout:
    pipeline_output/01_driving_mini_detection/
        detection_manifest.json
        detection_predictions.csv
        detection_class_summary.csv
        detection_video_summary.csv
        <video_id>/
            detections.json
                - legacy accepted boxes in boxes/scores/labels
                - accepted_detections per frame
                - candidate_detections per frame
            detection_predictions.csv
            detection_class_summary.csv

Usage:
    python src/exp_driving_videos/detect_driving_mini.py
    python src/exp_driving_videos/detect_driving_mini.py \\
        --video 0153f03b-3b26c404 \\
        --confidence-threshold 0.25 \\
        --yolo-model yolov8s-worldv2.pt \\
        --force-recompute
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import cv2
except Exception as exc:  # pragma: no cover - optional runtime dependency guard
    cv2 = None
    _CV2_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _CV2_IMPORT_ERROR = None

import config
from src.exp_driving_videos.modules import od_calibration_policy_utils

# Reuse the shared detector classes from the nuScenes detection pipeline
try:
    from src.exp_nuScenes.detection_pipeline import (
        YOLO_WORLD_DEFAULT_CLASSES,
        DetectionResult,
        ObjectDetector,
        YOLOWorldDetector,
    )
except Exception as exc:  # pragma: no cover - optional runtime dependency guard
    YOLO_WORLD_DEFAULT_CLASSES = []
    DetectionResult = Any
    ObjectDetector = Any
    YOLOWorldDetector = None
    _DETECTOR_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _DETECTOR_IMPORT_ERROR = None


_DETECTION_SCHEMA_VERSION = 6

_PREDICTION_CSV_FIELDS: List[str] = [
    "video_id", "frame", "frame_index", "image_path", "detection_id", "class_name", "score",
    "raw_score", "calibrated_score", "feedback_bonus", "score_used_for_candidate_ranking",
    "accepted",
    "candidate_source", "quality_bucket",
    "od_calibration_policy_id", "od_calibration_policy_applied", "od_calibration_branch",
    "matched_prior_ids_json", "top_prior_id", "prior_relevance_score",
    "matched_prior_scores_json", "matched_prior_tracking_priorities_json",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "bbox_width", "bbox_height", "bbox_area",
    "accepted_confidence_threshold", "candidate_confidence_threshold",
    "borderline_confidence_threshold", "accepted_nms_iou_threshold",
    "candidate_nms_iou_threshold",
]

_CLASS_SUMMARY_CSV_FIELDS: List[str] = [
    "class_name", "num_total_bboxes", "num_accepted_bboxes", "num_candidate_bboxes",
    "num_videos_with_detections", "num_frames_with_detections",
    "num_videos_with_accepted", "num_frames_with_accepted",
    "num_videos_with_candidates", "num_frames_with_candidates",
    "avg_score_total", "avg_score_accepted", "avg_score_candidate", "avg_bbox_area_total",
    "quality_counts_json", "num_high_quality", "num_borderline_quality",
    "num_low_quality", "num_nms_relaxed_quality", "num_discarded_quality",
]

_VIDEO_SUMMARY_CSV_FIELDS: List[str] = [
    "video_id", "num_frames", "num_classes", "num_accepted_bboxes",
    "num_candidate_bboxes", "num_total_bboxes", "quality_counts_json",
    "num_high_quality", "num_borderline_quality", "num_low_quality",
    "num_nms_relaxed_quality", "num_discarded_quality", "detected_classes_json",
]


def _format_dependency_error(exc: Optional[BaseException]) -> str:
    if exc is None:
        return ""
    return f"{exc.__class__.__name__}: {exc}"


def get_detector_dependency_status(*, render_video: bool = True) -> Dict[str, Any]:
    return {
        "cv2_available": cv2 is not None,
        "cv2_error": _format_dependency_error(_CV2_IMPORT_ERROR),
        "detector_backend_available": YOLOWorldDetector is not None,
        "detector_backend_error": _format_dependency_error(_DETECTOR_IMPORT_ERROR),
        "render_video_requested": bool(render_video),
        "render_video_available": bool(render_video and cv2 is not None),
    }


def detector_dependency_warnings(*, render_video: bool = True) -> List[str]:
    status = get_detector_dependency_status(render_video=render_video)
    warnings: List[str] = []
    if not bool(status.get("cv2_available", False)):
        if render_video:
            warnings.append(
                "OpenCV (`cv2`) is not available; detection rendering will be disabled. "
                f"Import error: {status.get('cv2_error', 'unknown')}"
            )
        else:
            warnings.append(
                "OpenCV (`cv2`) is not available. Detection can still use cached or non-rendered outputs, "
                "but boxed-video rendering is unavailable."
            )
    if not bool(status.get("detector_backend_available", False)):
        warnings.append(
            "Detector backend dependencies are missing; Step 1 detection cannot run until they are installed. "
            f"Import error: {status.get('detector_backend_error', 'unknown')}"
        )
    return warnings


def ensure_detector_runtime_available(*, render_video: bool = True) -> bool:
    if YOLOWorldDetector is None:
        raise RuntimeError(
            "Step 1 detection dependencies are unavailable. "
            + " ".join(detector_dependency_warnings(render_video=render_video))
        )
    return True

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

def get_frames_root() -> Path:
    return config.get_dataset_path("driving_mini") / "frames"


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "01_driving_mini_detection"
    out.mkdir(parents=True, exist_ok=True)
    return out


def list_video_ids(frames_root: Optional[Path] = None) -> List[str]:
    root = frames_root or get_frames_root()
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def list_frames(video_id: str, frames_root: Optional[Path] = None) -> List[Path]:
    root = frames_root or get_frames_root()
    return sorted((root / video_id).glob("frame_*.jpg"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _thresholds_match(left: Dict[str, Any], right: Dict[str, Any], *, tol: float = 1e-12) -> bool:
    keys = (
        "accepted_confidence_threshold",
        "candidate_confidence_threshold",
        "accepted_nms_iou_threshold",
        "candidate_nms_iou_threshold",
        "borderline_confidence_margin",
    )
    for key in keys:
        if abs(_safe_float(left.get(key, 0.0)) - _safe_float(right.get(key, 0.0))) > tol:
            return False
    return True


def _load_manifest_fast_path(
    *,
    manifest_path: Path,
    target_videos: List[str],
    model_name: str,
    effective_classes: List[str],
    confidence_threshold: float,
    nms_iou_threshold: float,
    candidate_confidence_threshold: float,
    candidate_nms_iou_threshold: float,
    borderline_confidence_margin: float,
    policy_marker: Dict[str, Any],
    output_root: Path,
) -> Optional[List[Dict[str, Any]]]:
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception:
        return None

    if int(manifest.get("schema_version", 0)) < _DETECTION_SCHEMA_VERSION:
        return None
    if str(manifest.get("model", "")) != str(model_name):
        return None
    manifest_classes = list(manifest.get("classes", []))
    if manifest_classes and manifest_classes != list(effective_classes):
        return None
    if not _thresholds_match(
        dict(manifest.get("thresholds", {})),
        {
            "accepted_confidence_threshold": confidence_threshold,
            "candidate_confidence_threshold": candidate_confidence_threshold,
            "accepted_nms_iou_threshold": nms_iou_threshold,
            "candidate_nms_iou_threshold": candidate_nms_iou_threshold,
            "borderline_confidence_margin": borderline_confidence_margin,
        },
    ):
        return None
    if not _policy_marker_matches(dict(manifest.get("od_calibration", {})), policy_marker):
        return None

    manifest_videos = {
        str(entry.get("video_id", "")): dict(entry)
        for entry in list(manifest.get("videos", []))
        if str(entry.get("video_id", ""))
    }
    if any(video_id not in manifest_videos for video_id in target_videos):
        return None

    aggregate_paths = dict(manifest.get("output_paths", {}))
    required_aggregate_paths = (
        output_root / "detection_predictions.csv",
        output_root / "detection_class_summary.csv",
        output_root / "detection_video_summary.csv",
    )
    for aggregate_path in required_aggregate_paths:
        if not aggregate_path.exists():
            return None
    if str(aggregate_paths.get("detection_predictions_csv", "")) and Path(
        str(aggregate_paths.get("detection_predictions_csv", ""))
    ) != required_aggregate_paths[0]:
        return None

    loaded_results: List[Dict[str, Any]] = []
    for video_id in target_videos:
        video_entry = manifest_videos[video_id]
        detections_json = str(video_entry.get("detections_json", "")).strip()
        detections_path = Path(detections_json) if detections_json else (output_root / video_id / "detections.json")
        if not detections_path.exists():
            return None
        try:
            with detections_path.open("r", encoding="utf-8") as fh:
                cached = json.load(fh)
        except Exception:
            return None
        if int(cached.get("schema_version", 0)) < _DETECTION_SCHEMA_VERSION:
            return None
        cached["from_cache"] = True
        loaded_results.append(cached)
    return loaded_results


def _label_color(label: str) -> tuple[int, int, int]:
    # Stable pseudo-random color per class label.
    h = abs(hash(label)) % 360
    s = 0.75
    v = 0.95
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


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _quality_bucket(detection: Dict[str, Any]) -> str:
    if bool(detection.get("accepted", False)):
        return "accepted_high_confidence"
    return str(detection.get("candidate_source", "candidate"))


def _detection_id(frame_index: int, collection_name: str, det_index: int) -> str:
    return f"{int(frame_index):06d}:{collection_name}:{int(det_index):04d}"


def _iter_frame_detections(frame_records: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for frame_record in frame_records:
        for field_name in ("accepted_detections", "candidate_detections"):
            for detection in list(frame_record.get(field_name, [])):
                yield detection


def _policy_marker_matches(
    cached_marker: Dict[str, Any],
    expected_marker: Dict[str, Any],
) -> bool:
    return (
        str(cached_marker.get("policy_id", "")) == str(expected_marker.get("policy_id", ""))
        and int(cached_marker.get("policy_version", 0)) == int(expected_marker.get("policy_version", 0))
        and bool(cached_marker.get("policy_available", False)) == bool(expected_marker.get("policy_available", False))
    )


def _cache_has_detection_identity_and_calibration(
    frame_records: List[Dict[str, Any]],
    expected_policy_marker: Dict[str, Any],
) -> bool:
    for frame_record in frame_records:
        if not _policy_marker_matches(
            dict(frame_record.get("od_calibration", {})),
            expected_policy_marker,
        ):
            return False
        for detection in _iter_frame_detections([frame_record]):
            if not str(detection.get("detection_id", "")).strip():
                return False
            if "raw_score" not in detection:
                return False
            if "calibrated_score" not in detection:
                return False
            if "feedback_bonus" not in detection:
                return False
            if "score_used_for_candidate_ranking" not in detection:
                return False
    return True


def _ensure_detection_identity_and_calibration(
    frame_records: List[Dict[str, Any]],
    policy: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    updated_records: List[Dict[str, Any]] = []
    policy_marker = od_calibration_policy_utils.current_policy_marker(policy)
    for frame_record in frame_records:
        frame_index = int(frame_record.get("frame_index", -1))
        updated_frame = od_calibration_policy_utils.apply_policy_to_frame_record(frame_record, policy)
        for collection_name, field_name in (
            ("accepted", "accepted_detections"),
            ("candidate", "candidate_detections"),
        ):
            detections = list(updated_frame.get(field_name, []))
            for det_index, detection in enumerate(detections):
                detection.setdefault("detection_id", _detection_id(frame_index, collection_name, det_index))
                detection.setdefault("raw_score", float(detection.get("score", 0.0)))
                detection.setdefault("calibrated_score", float(detection.get("raw_score", detection.get("score", 0.0))))
                detection.setdefault("feedback_bonus", float(detection.get("calibrated_score", 0.0)) - float(detection.get("raw_score", 0.0)))
                detection.setdefault(
                    "score_used_for_candidate_ranking",
                    float(detection.get("calibrated_score", detection.get("raw_score", detection.get("score", 0.0)))),
                )
        updated_frame["od_calibration"] = dict(policy_marker)
        updated_records.append(updated_frame)
    return updated_records


def _detection_quality_rank(quality_bucket: str) -> int:
    order = {
        "accepted_high_confidence": 0,
        "nms_relaxed_candidate": 1,
        "borderline_confidence": 2,
        "low_confidence": 3,
        "discarded_detector_output": 4,
        "candidate": 5,
    }
    return int(order.get(str(quality_bucket), 9))


def _build_prediction_rows(frame_records: List[Dict[str, Any]], video_id: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for frame_record in frame_records:
        frame_name = str(frame_record.get("frame", ""))
        frame_index = int(frame_record.get("frame_index", -1))
        image_path = str(frame_record.get("image_path", ""))
        calibration_marker = dict(frame_record.get("od_calibration", {}))
        for collection_name in ("accepted_detections", "candidate_detections"):
            for detection in list(frame_record.get(collection_name, [])):
                bbox = list(detection.get("bbox", []))
                threshold_info = dict(detection.get("threshold_info", {}))
                prior_metadata = dict(detection.get("prior_metadata", {}))
                od_calibration = dict(detection.get("od_calibration", {}))
                matched_prior_ids = [str(value) for value in list(prior_metadata.get("matched_prior_ids", [])) if str(value)]
                quality_bucket = _quality_bucket(detection)
                rows.append(
                    {
                        "video_id": video_id,
                        "frame": frame_name,
                        "frame_index": frame_index,
                        "image_path": image_path,
                        "detection_id": str(detection.get("detection_id", "")),
                        "class_name": str(detection.get("class", "")),
                        "score": float(detection.get("score", 0.0)),
                        "raw_score": float(detection.get("raw_score", detection.get("score", 0.0))),
                        "calibrated_score": float(detection.get("calibrated_score", detection.get("score", 0.0))),
                        "feedback_bonus": float(detection.get("feedback_bonus", 0.0)),
                        "score_used_for_candidate_ranking": float(
                            detection.get(
                                "score_used_for_candidate_ranking",
                                detection.get("calibrated_score", detection.get("score", 0.0)),
                            )
                        ),
                        "accepted": bool(detection.get("accepted", False)),
                        "candidate_source": str(detection.get("candidate_source", "")),
                        "quality_bucket": quality_bucket,
                        "od_calibration_policy_id": str(
                            od_calibration.get("policy_id", calibration_marker.get("policy_id", ""))
                        ),
                        "od_calibration_policy_applied": bool(
                            od_calibration.get("policy_applied", calibration_marker.get("policy_available", False))
                        ),
                        "od_calibration_branch": str(
                            od_calibration.get("calibration_branch", "candidate_exploration")
                        ),
                        "matched_prior_ids_json": json.dumps(matched_prior_ids),
                        "top_prior_id": matched_prior_ids[0] if matched_prior_ids else "",
                        "prior_relevance_score": float(prior_metadata.get("prior_relevance_score", 0.0) or 0.0),
                        "matched_prior_scores_json": json.dumps(dict(prior_metadata.get("matched_prior_scores", {})), sort_keys=True),
                        "matched_prior_tracking_priorities_json": json.dumps(
                            dict(prior_metadata.get("matched_prior_tracking_priorities", {})),
                            sort_keys=True,
                        ),
                        "bbox_x1": float(bbox[0]) if len(bbox) > 0 else "",
                        "bbox_y1": float(bbox[1]) if len(bbox) > 1 else "",
                        "bbox_x2": float(bbox[2]) if len(bbox) > 2 else "",
                        "bbox_y2": float(bbox[3]) if len(bbox) > 3 else "",
                        "bbox_width": float(bbox[2] - bbox[0]) if len(bbox) >= 4 else "",
                        "bbox_height": float(bbox[3] - bbox[1]) if len(bbox) >= 4 else "",
                        "bbox_area": float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])) if len(bbox) >= 4 else "",
                        "accepted_confidence_threshold": threshold_info.get("accepted_confidence_threshold", ""),
                        "candidate_confidence_threshold": threshold_info.get("candidate_confidence_threshold", ""),
                        "borderline_confidence_threshold": threshold_info.get("borderline_confidence_threshold", ""),
                        "accepted_nms_iou_threshold": threshold_info.get("accepted_nms_iou_threshold", ""),
                        "candidate_nms_iou_threshold": threshold_info.get("candidate_nms_iou_threshold", ""),
                    }
                )
    rows.sort(
        key=lambda row: (
            int(row.get("frame_index", -1)),
            int(not bool(row.get("accepted", False))),
            _detection_quality_rank(str(row.get("quality_bucket", ""))),
            -float(row.get("score", 0.0)),
            str(row.get("class_name", "")),
        )
    )
    return rows


def _build_class_summary_rows(prediction_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_class: Dict[str, Dict[str, Any]] = {}
    for row in prediction_rows:
        class_name = str(row.get("class_name", ""))
        bucket = by_class.setdefault(
            class_name,
            {
                "class_name": class_name,
                "num_total_bboxes": 0,
                "num_accepted_bboxes": 0,
                "num_candidate_bboxes": 0,
                "num_videos_with_detections": set(),
                "num_frames_with_detections": set(),
                "num_videos_with_accepted": set(),
                "num_frames_with_accepted": set(),
                "num_videos_with_candidates": set(),
                "num_frames_with_candidates": set(),
                "score_sum_total": 0.0,
                "score_sum_accepted": 0.0,
                "score_sum_candidate": 0.0,
                "bbox_area_sum_total": 0.0,
                "quality_counts": Counter(),
            },
        )
        bucket["num_total_bboxes"] += 1
        bucket["num_videos_with_detections"].add(str(row.get("video_id", "")))
        bucket["num_frames_with_detections"].add(int(row.get("frame_index", -1)))
        bucket["score_sum_total"] += float(row.get("score", 0.0))
        bucket["bbox_area_sum_total"] += float(row.get("bbox_area", 0.0) or 0.0)
        quality_bucket = str(row.get("quality_bucket", ""))
        bucket["quality_counts"][quality_bucket] += 1
        if bool(row.get("accepted", False)):
            bucket["num_accepted_bboxes"] += 1
            bucket["num_videos_with_accepted"].add(str(row.get("video_id", "")))
            bucket["num_frames_with_accepted"].add(int(row.get("frame_index", -1)))
            bucket["score_sum_accepted"] += float(row.get("score", 0.0))
        else:
            bucket["num_candidate_bboxes"] += 1
            bucket["num_videos_with_candidates"].add(str(row.get("video_id", "")))
            bucket["num_frames_with_candidates"].add(int(row.get("frame_index", -1)))
            bucket["score_sum_candidate"] += float(row.get("score", 0.0))

    rows: List[Dict[str, Any]] = []
    for class_name, bucket in sorted(by_class.items()):
        num_total = int(bucket["num_total_bboxes"])
        num_accepted = int(bucket["num_accepted_bboxes"])
        num_candidate = int(bucket["num_candidate_bboxes"])
        quality_counts = dict(sorted(dict(bucket["quality_counts"]).items()))
        rows.append(
            {
                "class_name": class_name,
                "num_total_bboxes": num_total,
                "num_accepted_bboxes": num_accepted,
                "num_candidate_bboxes": num_candidate,
                "num_videos_with_detections": len({v for v in bucket["num_videos_with_detections"] if v}),
                "num_frames_with_detections": len(bucket["num_frames_with_detections"]),
                "num_videos_with_accepted": len({v for v in bucket["num_videos_with_accepted"] if v}),
                "num_frames_with_accepted": len(bucket["num_frames_with_accepted"]),
                "num_videos_with_candidates": len({v for v in bucket["num_videos_with_candidates"] if v}),
                "num_frames_with_candidates": len(bucket["num_frames_with_candidates"]),
                "avg_score_total": float(bucket["score_sum_total"] / max(1, num_total)),
                "avg_score_accepted": float(bucket["score_sum_accepted"] / max(1, num_accepted)),
                "avg_score_candidate": float(bucket["score_sum_candidate"] / max(1, num_candidate)),
                "avg_bbox_area_total": float(bucket["bbox_area_sum_total"] / max(1, num_total)),
                "quality_counts_json": json.dumps(quality_counts),
                "num_high_quality": int(quality_counts.get("accepted_high_confidence", 0)),
                "num_borderline_quality": int(quality_counts.get("borderline_confidence", 0)),
                "num_low_quality": int(quality_counts.get("low_confidence", 0)),
                "num_nms_relaxed_quality": int(quality_counts.get("nms_relaxed_candidate", 0)),
                "num_discarded_quality": int(quality_counts.get("discarded_detector_output", 0)),
            }
        )
    rows.sort(key=lambda row: (-int(row["num_total_bboxes"]), str(row["class_name"])))
    return rows


def _build_video_summary_rows(video_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in video_results:
        prediction_rows = _build_prediction_rows(result.get("frames", []), video_id=str(result.get("video_id", "")))
        quality_counts = Counter(str(row.get("quality_bucket", "")) for row in prediction_rows)
        rows.append(
            {
                "video_id": str(result.get("video_id", "")),
                "num_frames": int(result.get("num_frames", 0)),
                "num_classes": len(result.get("detected_classes", {})),
                "num_accepted_bboxes": int(result.get("num_detections", 0)),
                "num_candidate_bboxes": int(result.get("num_candidate_detections", 0)),
                "num_total_bboxes": int(result.get("num_total_saved_detections", result.get("num_detections", 0))),
                "quality_counts_json": json.dumps(dict(sorted(quality_counts.items()))),
                "num_high_quality": int(quality_counts.get("accepted_high_confidence", 0)),
                "num_borderline_quality": int(quality_counts.get("borderline_confidence", 0)),
                "num_low_quality": int(quality_counts.get("low_confidence", 0)),
                "num_nms_relaxed_quality": int(quality_counts.get("nms_relaxed_candidate", 0)),
                "num_discarded_quality": int(quality_counts.get("discarded_detector_output", 0)),
                "detected_classes_json": json.dumps(result.get("detected_classes", {}), sort_keys=True),
            }
        )
    rows.sort(key=lambda row: str(row["video_id"]))
    return rows


def _format_step_progress(completed: int, total: int) -> str:
    if total <= 0:
        return "0/0 videos (0.0%)"
    percentage = 100.0 * float(completed) / float(total)
    return f"{completed}/{total} videos ({percentage:.1f}%)"


def render_detection_video(
    video_id: str,
    frame_records: List[Dict[str, Any]],
    output_path: Path,
    fps: float = 10.0,
) -> Optional[str]:
    """Render one annotated mp4 for a video using detection frame records."""
    if cv2 is None:
        return None
    if not frame_records:
        return None

    first = cv2.imread(frame_records[0]["image_path"])
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
        for rec in frame_records:
            img = cv2.imread(rec["image_path"])
            if img is None:
                continue

            for box, score, label in zip(rec.get("boxes", []), rec.get("scores", []), rec.get("labels", [])):
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                color = _label_color(str(label))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {float(score):.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_y = max(y1 - 6, th + 6)
                cv2.rectangle(img, (x1, text_y - th - 4), (x1 + tw + 4, text_y), color, -1)
                cv2.putText(
                    img,
                    text,
                    (x1 + 2, text_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                img,
                f"{video_id} | frame {rec.get('frame_index', -1):04d}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(img)
    finally:
        writer.release()

    return str(output_path)


# ---------------------------------------------------------------------------
# Per-video detection
# ---------------------------------------------------------------------------

def process_video(
    video_id: str,
    detector: Optional[ObjectDetector],
    od_calibration_policy: Optional[Dict[str, Any]] = None,
    background_rule_relevance_prior_results: Optional[Dict[str, Any]] = None,
    frames_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
    render_video: bool = True,
) -> Dict[str, Any]:
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    detections_file = out_dir / "detections.json"
    predictions_csv_file = out_dir / "detection_predictions.csv"
    class_summary_csv_file = out_dir / "detection_class_summary.csv"
    boxed_video_file = out_dir / "detections_boxed.mp4"
    policy_marker = od_calibration_policy_utils.current_policy_marker(od_calibration_policy)

    if not force_recompute and detections_file.exists():
        with detections_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        cached_frames = list(cached.get("frames", []))
        cache_is_current = int(cached.get("schema_version", 1)) >= _DETECTION_SCHEMA_VERSION and all(
            "accepted_detections" in frame and "candidate_detections" in frame
            for frame in cached_frames
        )
        if cache_is_current:
            cached_output_paths = dict(cached.get("output_paths", {}))
            artifacts_ready = (
                predictions_csv_file.exists()
                and class_summary_csv_file.exists()
                and (not render_video or boxed_video_file.exists() or not cached_frames)
            )
            if (
                artifacts_ready
                and _policy_marker_matches(dict(cached.get("od_calibration", {})), policy_marker)
                and _cache_has_detection_identity_and_calibration(cached_frames, policy_marker)
            ):
                cached["output_paths"] = {
                    **cached_output_paths,
                    "detections_json": str(detections_file),
                    "detection_predictions_csv": str(predictions_csv_file),
                    "detection_class_summary_csv": str(class_summary_csv_file),
                    "background_rule_relevance_prior_json": str(
                        background_rule_relevance_prior_results.get("output_paths", {}).get("prior_json", "")
                    )
                    if isinstance(background_rule_relevance_prior_results, dict)
                    else "",
                }
                cached["od_calibration"] = dict(policy_marker)
                cached["from_cache"] = True
                return cached
            cached_frames = _ensure_detection_identity_and_calibration(cached_frames, od_calibration_policy)
            cached["frames"] = cached_frames
            cached["od_calibration"] = dict(policy_marker)
            prediction_rows = _build_prediction_rows(cached_frames, video_id=video_id)
            class_summary_rows = _build_class_summary_rows(prediction_rows)
            _write_csv(
                predictions_csv_file,
                _PREDICTION_CSV_FIELDS,
                prediction_rows,
            )
            _write_csv(
                class_summary_csv_file,
                _CLASS_SUMMARY_CSV_FIELDS,
                class_summary_rows,
            )
            cached.setdefault("output_paths", {})
            cached["output_paths"]["detections_json"] = str(detections_file)
            cached["output_paths"]["detection_predictions_csv"] = str(predictions_csv_file)
            cached["output_paths"]["detection_class_summary_csv"] = str(class_summary_csv_file)
            cached["output_paths"]["background_rule_relevance_prior_json"] = str(
                background_rule_relevance_prior_results.get("output_paths", {}).get("prior_json", "")
            ) if isinstance(background_rule_relevance_prior_results, dict) else ""
            with detections_file.open("w", encoding="utf-8") as fh:
                json.dump(cached, fh, indent=2)
            if render_video and not boxed_video_file.exists() and cached.get("frames"):
                render_path = render_detection_video(
                    video_id=video_id,
                    frame_records=cached["frames"],
                    output_path=boxed_video_file,
                )
                if render_path:
                    cached["boxed_video_path"] = render_path
                    with detections_file.open("w", encoding="utf-8") as fh:
                        json.dump(cached, fh, indent=2)
            cached["from_cache"] = True
            return cached

    frames = list_frames(video_id, frames_root)

    if not frames:
        result: Dict[str, Any] = {
            "schema_version": _DETECTION_SCHEMA_VERSION,
            "video_id": video_id,
            "num_frames": 0,
            "num_detections": 0,
            "num_candidate_detections": 0,
            "num_total_saved_detections": 0,
            "detected_classes": {},
            "frames": [],
            "output_paths": {
                "detections_json": str(detections_file),
                "detection_predictions_csv": str(predictions_csv_file),
                "detection_class_summary_csv": str(class_summary_csv_file),
                "background_rule_relevance_prior_json": str(
                    background_rule_relevance_prior_results.get("output_paths", {}).get("prior_json", "")
                )
                if isinstance(background_rule_relevance_prior_results, dict)
                else "",
            },
            "od_calibration": od_calibration_policy_utils.current_policy_marker(od_calibration_policy),
            "from_cache": False,
        }
        _write_csv(
            predictions_csv_file,
            _PREDICTION_CSV_FIELDS,
            [],
        )
        _write_csv(
            class_summary_csv_file,
            _CLASS_SUMMARY_CSV_FIELDS,
            [],
        )
        with detections_file.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        return result

    # Batch all frames in one model.predict() call for speed
    if detector is None:
        return {
            "video_id": video_id,
            "from_cache": False,
            "_requires_detection": True,
        }
    image_paths = [str(f) for f in frames]
    results: List[DetectionResult] = detector.detect_batch(image_paths)

    frame_records: List[Dict[str, Any]] = []
    all_labels: List[str] = []
    total_candidate_detections = 0
    total_saved_detections = 0

    for frame_path, det in zip(frames, results):
        all_labels.extend(det.labels)
        frame_record = {
            "frame": frame_path.name,
            "frame_index": int(frame_path.stem.split("_")[-1]),
            **det.to_dict(),
        }
        frame_records.append(frame_record)
    frame_records = _ensure_detection_identity_and_calibration(frame_records, od_calibration_policy)
    for frame_record in frame_records:
        total_candidate_detections += int(frame_record.get("num_candidate_detections", 0))
        total_saved_detections += int(
            frame_record.get("num_total_saved_detections", frame_record.get("num_detections", 0))
        )

    label_counts = dict(Counter(all_labels).most_common())
    prediction_rows = _build_prediction_rows(frame_records, video_id=video_id)
    class_summary_rows = _build_class_summary_rows(prediction_rows)

    video_result: Dict[str, Any] = {
        "schema_version": _DETECTION_SCHEMA_VERSION,
        "video_id": video_id,
        "num_frames": len(frame_records),
        "num_detections": sum(len(f["boxes"]) for f in frame_records),
        "num_candidate_detections": total_candidate_detections,
        "num_total_saved_detections": total_saved_detections,
        "detected_classes": label_counts,
        "frames": frame_records,
        "output_paths": {
            "detections_json": str(detections_file),
            "detection_predictions_csv": str(predictions_csv_file),
            "detection_class_summary_csv": str(class_summary_csv_file),
            "background_rule_relevance_prior_json": str(
                background_rule_relevance_prior_results.get("output_paths", {}).get("prior_json", "")
            )
            if isinstance(background_rule_relevance_prior_results, dict)
            else "",
        },
        "od_calibration": od_calibration_policy_utils.current_policy_marker(od_calibration_policy),
        "from_cache": False,
    }

    render_path: Optional[str] = None
    if render_video:
        render_path = render_detection_video(
            video_id=video_id,
            frame_records=frame_records,
            output_path=boxed_video_file,
        )
        if render_path:
            video_result["boxed_video_path"] = render_path

    with detections_file.open("w", encoding="utf-8") as fh:
        json.dump(video_result, fh, indent=2)
    _write_csv(
        predictions_csv_file,
        _PREDICTION_CSV_FIELDS,
        prediction_rows,
    )
    _write_csv(
        class_summary_csv_file,
        _CLASS_SUMMARY_CSV_FIELDS,
        class_summary_rows,
    )

    return video_result


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run(
    video_ids: Optional[List[str]] = None,
    model_name: str = "yolov8s-worldv2.pt",
    classes: Optional[List[str]] = None,
    confidence_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5,
    candidate_confidence_threshold: float = 0.01,
    candidate_nms_iou_threshold: float = 0.99,
    borderline_confidence_margin: float = 0.05,
    device: Optional[str] = None,
    predict_batch_size: Optional[int] = None,
    inference_imgsz: int = 640,
    use_half_precision: Optional[bool] = None,
    od_calibration_policy: Optional[Dict[str, Any]] = None,
    background_rule_relevance_prior_results: Optional[Dict[str, Any]] = None,
    frames_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
    render_video: bool = True,
) -> List[Dict[str, Any]]:
    effective_frames_root = frames_root or get_frames_root()
    effective_output_root = output_root or get_output_root()
    effective_render_video = bool(render_video)
    if effective_render_video and cv2 is None:
        print(
            "[warn] OpenCV (`cv2`) is not available; detection rendering will be skipped for this run. "
            f"Import error: {_format_dependency_error(_CV2_IMPORT_ERROR)}"
        )
        effective_render_video = False
    all_videos = list_video_ids(effective_frames_root)
    target_videos = list(video_ids) if video_ids else all_videos

    unknown = set(target_videos) - set(all_videos)
    if unknown:
        raise ValueError(f"Unknown video IDs: {sorted(unknown)}. Available: {all_videos}")

    effective_classes = list(classes) if classes else list(YOLO_WORLD_DEFAULT_CLASSES)
    resolved_calibration_policy = (
        dict(od_calibration_policy)
        if isinstance(od_calibration_policy, dict)
        else od_calibration_policy_utils.load_active_od_calibration_policy()
    )
    policy_marker = od_calibration_policy_utils.current_policy_marker(resolved_calibration_policy)
    manifest_path = effective_output_root / "detection_manifest.json"

    print(f"Step 1 detection: {_format_step_progress(0, len(target_videos))}")
    print(f"Model: {model_name} | classes: {len(effective_classes)}")
    print(f"Render video: {effective_render_video}")
    print(
        "Detection cfg: "
        f"accept_conf={confidence_threshold:.3f}, "
        f"candidate_conf={candidate_confidence_threshold:.3f}, "
        f"accept_iou={nms_iou_threshold:.3f}, "
        f"candidate_iou={candidate_nms_iou_threshold:.3f}"
    )
    print(
        "OD calibration policy: "
        f"{od_calibration_policy_utils.policy_id(resolved_calibration_policy) or 'none'}"
    )

    if not force_recompute:
        fast_cached_results = _load_manifest_fast_path(
            manifest_path=manifest_path,
            target_videos=target_videos,
            model_name=model_name,
            effective_classes=effective_classes,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            candidate_confidence_threshold=candidate_confidence_threshold,
            candidate_nms_iou_threshold=candidate_nms_iou_threshold,
            borderline_confidence_margin=borderline_confidence_margin,
            policy_marker=policy_marker,
            output_root=effective_output_root,
        )
        if fast_cached_results is not None:
            print(
                "Cache fast path: "
                f"reusing manifest-backed detection cache for {len(fast_cached_results)} videos"
            )
            print(f"Step 1 complete: {_format_step_progress(len(fast_cached_results), len(target_videos))}")
            print(f"Outputs: {manifest_path.parent}")
            return fast_cached_results

    detector: Optional[ObjectDetector] = None
    runtime_profile: Dict[str, Any] = {}

    def _ensure_detector() -> ObjectDetector:
        nonlocal detector, runtime_profile
        if detector is not None:
            return detector
        ensure_detector_runtime_available(render_video=effective_render_video)
        detector = YOLOWorldDetector(
            model_name=model_name,
            classes=effective_classes,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            candidate_confidence_threshold=candidate_confidence_threshold,
            candidate_nms_iou_threshold=candidate_nms_iou_threshold,
            borderline_confidence_margin=borderline_confidence_margin,
            device=device,
            predict_batch_size=predict_batch_size,
            inference_imgsz=inference_imgsz,
            use_half_precision=use_half_precision,
            background_rule_relevance_prior_entries=list(background_rule_relevance_prior_results.get("entries", []))
            if isinstance(background_rule_relevance_prior_results, dict)
            else [],
        )
        detector.warmup()
        runtime_profile = detector.get_runtime_profile() if isinstance(detector, YOLOWorldDetector) else {}
        print(
            "Inference profile: "
            f"device={runtime_profile.get('device', device)}, "
            f"gpu={runtime_profile.get('gpu_name', 'n/a')}, "
            f"batch={runtime_profile.get('predict_batch_size', predict_batch_size)}, "
            f"imgsz={runtime_profile.get('inference_imgsz', inference_imgsz)}, "
            f"half={runtime_profile.get('half_precision', use_half_precision)}"
        )
        return detector

    video_results: List[Dict[str, Any]] = []
    try:
        total_videos = len(target_videos)
        for index, video_id in enumerate(target_videos, start=1):
            result = process_video(
                video_id=video_id,
                detector=detector,
                od_calibration_policy=resolved_calibration_policy,
                background_rule_relevance_prior_results=background_rule_relevance_prior_results,
                frames_root=effective_frames_root,
                output_root=effective_output_root,
                force_recompute=force_recompute,
                render_video=effective_render_video,
            )
            if (
                bool(result.get("_requires_detection", False))
            ):
                result = process_video(
                    video_id=video_id,
                    detector=_ensure_detector(),
                    od_calibration_policy=resolved_calibration_policy,
                    background_rule_relevance_prior_results=background_rule_relevance_prior_results,
                    frames_root=effective_frames_root,
                    output_root=effective_output_root,
                    force_recompute=force_recompute,
                    render_video=effective_render_video,
                )
            video_results.append(result)
            cache_tag = "cached" if bool(result.get("from_cache", False)) else "done"
            print(f"Progress: {_format_step_progress(index, total_videos)} | {video_id} | {cache_tag}")
    finally:
        if detector is not None:
            detector.teardown()

    # Aggregate class counts across all videos
    total_labels: Counter[str] = Counter()
    for r in video_results:
        total_labels.update(r["detected_classes"])

    all_prediction_rows: List[Dict[str, Any]] = []
    for r in video_results:
        all_prediction_rows.extend(_build_prediction_rows(r.get("frames", []), video_id=str(r.get("video_id", ""))))
    aggregate_class_summary_rows = _build_class_summary_rows(all_prediction_rows)
    video_summary_rows = _build_video_summary_rows(video_results)

    manifest = {
        "schema_version": _DETECTION_SCHEMA_VERSION,
        "model": model_name,
        "classes": list(effective_classes),
        "num_videos": len(video_results),
        "num_frames_total": sum(r["num_frames"] for r in video_results),
        "num_detections_total": sum(r["num_detections"] for r in video_results),
        "num_candidate_detections_total": sum(r.get("num_candidate_detections", 0) for r in video_results),
        "num_total_saved_detections": sum(r.get("num_total_saved_detections", r["num_detections"]) for r in video_results),
        "detected_classes_total": dict(total_labels.most_common()),
        "thresholds": {
            "accepted_confidence_threshold": float(confidence_threshold),
            "candidate_confidence_threshold": float(candidate_confidence_threshold),
            "accepted_nms_iou_threshold": float(nms_iou_threshold),
            "candidate_nms_iou_threshold": float(candidate_nms_iou_threshold),
            "borderline_confidence_margin": float(borderline_confidence_margin),
        },
        "inference_profile": runtime_profile,
        "background_rule_relevance_prior": {
            "applied": bool(background_rule_relevance_prior_results),
            "prior_json": str(background_rule_relevance_prior_results.get("output_paths", {}).get("prior_json", ""))
            if isinstance(background_rule_relevance_prior_results, dict)
            else "",
        },
        "od_calibration": policy_marker,
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r["num_frames"],
                "num_detections": r["num_detections"],
                "num_candidate_detections": r.get("num_candidate_detections", 0),
                "num_total_saved_detections": r.get("num_total_saved_detections", r["num_detections"]),
                "num_classes": len(r["detected_classes"]),
                "detections_json": str(r.get("output_paths", {}).get("detections_json", "")),
            }
            for r in video_results
        ],
        "output_paths": {
            "detection_predictions_csv": str(effective_output_root / "detection_predictions.csv"),
            "detection_class_summary_csv": str(effective_output_root / "detection_class_summary.csv"),
            "detection_video_summary_csv": str(effective_output_root / "detection_video_summary.csv"),
        },
    }
    aggregate_predictions_csv_path = effective_output_root / "detection_predictions.csv"
    aggregate_class_summary_csv_path = effective_output_root / "detection_class_summary.csv"
    video_summary_csv_path = effective_output_root / "detection_video_summary.csv"
    _write_csv(
        aggregate_predictions_csv_path,
        _PREDICTION_CSV_FIELDS,
        all_prediction_rows,
    )
    _write_csv(
        aggregate_class_summary_csv_path,
        _CLASS_SUMMARY_CSV_FIELDS,
        aggregate_class_summary_rows,
    )
    _write_csv(
        video_summary_csv_path,
        _VIDEO_SUMMARY_CSV_FIELDS,
        video_summary_rows,
    )
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Step 1 complete: {_format_step_progress(len(video_results), len(target_videos))}")
    print(f"Outputs: {manifest_path.parent}")
    print(
        f"Detections: {manifest['num_detections_total']} accepted + "
        f"{manifest['num_candidate_detections_total']} candidates across "
        f"{manifest['num_frames_total']} frames"
    )
    print("Top classes: " + ", ".join(
        f"{k}({v})" for k, v in list(manifest["detected_classes_total"].items())[:10]
    ))
    return video_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO-World detection over driving_mini frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", action="append", default=[], dest="video",
                        help="Video ID(s) to process. Repeat for multiple. Default: all.")
    parser.add_argument("--yolo-model", default="yolov8s-worldv2.pt", dest="yolo_model")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Custom class vocabulary. Default: YOLO_WORLD_DEFAULT_CLASSES.")
    parser.add_argument("--confidence-threshold", type=float, default=0.3, dest="confidence_threshold")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5, dest="nms_iou_threshold")
    parser.add_argument("--candidate-confidence-threshold", type=float, default=0.01, dest="candidate_confidence_threshold")
    parser.add_argument("--candidate-nms-iou-threshold", type=float, default=0.99, dest="candidate_nms_iou_threshold")
    parser.add_argument("--borderline-confidence-margin", type=float, default=0.05, dest="borderline_confidence_margin")
    parser.add_argument("--device", default=None,
                        help="Inference device, e.g. cpu, cuda:0, mps.")
    parser.add_argument("--predict-batch-size", type=int, default=None, dest="predict_batch_size",
                        help="Explicit Ultralytics inference batch size. Default auto-tunes: A100=128, other CUDA=32, CPU=1.")
    parser.add_argument("--imgsz", type=int, default=640, dest="inference_imgsz",
                        help="Ultralytics inference image size.")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=None, dest="use_half_precision",
                        help="Enable or disable FP16 inference. Default auto-enables on CUDA.")
    parser.add_argument("--frames-root", type=Path, default=None, dest="frames_root",
                        help="Override default frames root.")
    parser.add_argument("--output-root", type=Path, default=None, dest="output_root",
                        help="Override default output root.")
    parser.add_argument("--force-recompute", action="store_true", dest="force_recompute")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        video_ids=args.video or None,
        model_name=args.yolo_model,
        classes=args.classes,
        confidence_threshold=args.confidence_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        candidate_confidence_threshold=args.candidate_confidence_threshold,
        candidate_nms_iou_threshold=args.candidate_nms_iou_threshold,
        borderline_confidence_margin=args.borderline_confidence_margin,
        device=args.device,
        predict_batch_size=args.predict_batch_size,
        inference_imgsz=args.inference_imgsz,
        use_half_precision=args.use_half_precision,
        frames_root=args.frames_root,
        output_root=args.output_root,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
