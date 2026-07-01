"""
Segment-level symbolic object motion summaries for driving_mini videos.

Consumes:
  - Step 7 output: per-frame relative object motion
  - Step 8 output: merged temporal segmentation

Output layout:
    pipeline_output/09_driving_mini_segment_object_motion/
        segment_object_motion_manifest.json
        <video_id>/
            segment_object_motion.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_SEGMENT_OBJECT_MOTION_VERSION = 8


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "09_driving_mini_segment_object_motion"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_local_image_path(image_path: str) -> str:
    """Map container-saved driving_mini frame paths to the current local dataset root."""
    if not image_path:
        return image_path

    path = Path(image_path)
    if path.exists():
        return str(path)

    parts = list(path.parts)
    if "frames" not in parts:
        return image_path

    frames_idx = parts.index("frames")
    rel_parts = parts[frames_idx + 1 :]
    if not rel_parts:
        return image_path

    candidate = config.get_dataset_path("driving_mini") / "frames"
    for part in rel_parts:
        candidate = candidate / part
    return str(candidate)


def _safe_slug(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "unknown")).strip("_")
    return clean or "unknown"


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


def _split_combined_label(label: str) -> Dict[str, str]:
    text = str(label or "")
    if "|" in text:
        forward, lateral = text.split("|", 1)
    else:
        forward, lateral = text, "straightforward"
    return {"forward": forward, "lateral": lateral}


def _majority_label(labels: List[str]) -> str:
    if not labels:
        return "unknown"
    return Counter(str(v) for v in labels).most_common(1)[0][0]


def _classify_rel_vz(values: List[float], threshold: float, dominance_ratio: float) -> Dict[str, Any]:
    if not values:
        return {
            "vz_state": "vz_unknown",
            "vz_approaching_ratio": 0.0,
            "vz_awaying_ratio": 0.0,
            "vz_stable_ratio": 0.0,
        }

    arr = np.asarray(values, dtype=np.float32)
    approaching_ratio = float(np.mean(arr < -threshold))
    awaying_ratio = float(np.mean(arr > threshold))
    stable_ratio = float(np.mean(np.abs(arr) <= threshold))
    mean_value = float(np.mean(arr))

    if approaching_ratio >= dominance_ratio:
        state = "vz_approaching"
    elif awaying_ratio >= dominance_ratio:
        state = "vz_awaying"
    elif stable_ratio >= dominance_ratio:
        state = "vz_stable"
    else:
        state = "vz_approaching" if mean_value < -threshold else (
            "vz_awaying" if mean_value > threshold else "vz_stable"
        )

    return {
        "vz_state": state,
        "vz_approaching_ratio": approaching_ratio,
        "vz_awaying_ratio": awaying_ratio,
        "vz_stable_ratio": stable_ratio,
    }


def _classify_rel_vx(values: List[float], threshold: float, dominance_ratio: float) -> Dict[str, Any]:
    if not values:
        return {
            "vx_state": "vx_unknown",
            "vx_turning_left_ratio": 0.0,
            "vx_turning_right_ratio": 0.0,
            "vx_stable_ratio": 0.0,
        }

    arr = np.asarray(values, dtype=np.float32)
    # Keep the older object-relative convention: negative = left, positive = right.
    left_ratio = float(np.mean(arr < -threshold))
    right_ratio = float(np.mean(arr > threshold))
    stable_ratio = float(np.mean(np.abs(arr) <= threshold))
    mean_value = float(np.mean(arr))

    if left_ratio >= dominance_ratio:
        state = "vx_turning_left"
    elif right_ratio >= dominance_ratio:
        state = "vx_turning_right"
    elif stable_ratio >= dominance_ratio:
        state = "vx_stable"
    else:
        state = "vx_turning_left" if mean_value < -threshold else (
            "vx_turning_right" if mean_value > threshold else "vx_stable"
        )

    return {
        "vx_state": state,
        "vx_turning_left_ratio": left_ratio,
        "vx_turning_right_ratio": right_ratio,
        "vx_stable_ratio": stable_ratio,
    }


def _classify_speed_state(mean_rel_speed: float, threshold: float) -> str:
    return "rel_moving" if float(mean_rel_speed) > float(threshold) else "rel_static"


def _classify_distance_state(mean_z: float, near_threshold: float, medium_threshold: float) -> str:
    if mean_z <= near_threshold:
        return "near"
    if mean_z <= medium_threshold:
        return "medium"
    return "far"


def _visibility_state(visibility_ratio: float, persistent_threshold: float = 0.8, present_threshold: float = 0.3) -> str:
    ratio = float(visibility_ratio)
    if ratio >= persistent_threshold:
        return "persistent"
    if ratio >= present_threshold:
        return "intermittent"
    return "brief"


def _score_stats(values: List[float]) -> Dict[str, float]:
    cleaned = [float(value) for value in values]
    if not cleaned:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(min(cleaned)),
        "mean": float(sum(cleaned) / max(1, len(cleaned))),
        "max": float(max(cleaned)),
    }


def _temporal_consistency_proxy(frame_indices: List[int], num_frames_in_segment: int) -> float:
    ordered = sorted(int(value) for value in frame_indices if int(value) >= 0)
    if not ordered:
        return 0.0
    visibility_ratio = float(len(ordered) / max(1, num_frames_in_segment))
    if len(ordered) == 1:
        return visibility_ratio
    consecutive_ratio = float(
        sum(1 for left, right in zip(ordered, ordered[1:]) if int(right) - int(left) == 1)
        / max(1, len(ordered) - 1)
    )
    return float((visibility_ratio + consecutive_ratio) / 2.0)


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "rel_vz_threshold",
        "rel_vx_threshold",
        "compare_rel_vx_thresholds",
        "rel_speed_threshold",
        "dominance_ratio_threshold",
        "distance_near_threshold",
        "distance_medium_threshold",
        "top_k_visualized_objects",
        "visualization_fps",
    ]
    return {k: cfg.get(k) for k in keys}


def _summarize_segment_object_group(
    *,
    grouped: Dict[int, List[Dict[str, Any]]],
    num_frames_in_segment: int,
    rel_vz_threshold: float,
    rel_vx_threshold: float,
    dominance_ratio_threshold: float,
    rel_speed_threshold: float,
    distance_near_threshold: float,
    distance_medium_threshold: float,
    include_candidate_provenance: bool = False,
) -> List[Dict[str, Any]]:
    objects_out: List[Dict[str, Any]] = []
    for track_id, samples in sorted(grouped.items()):
        labels = [str(s.get("label", "unknown")) for s in samples]
        positions = [s.get("position_3d", [0.0, 0.0, 0.0]) for s in samples]
        frame_indices = [int(s.get("frame_index", -1)) for s in samples]
        has_rel_motion_mask = [bool(s.get("has_rel_motion", False)) for s in samples]
        rel_vx_series = [float(s.get("rel_vx", 0.0)) for s in samples]
        rel_vz_series = [float(s.get("rel_vz", 0.0)) for s in samples]
        rel_speed_series = [float(s.get("rel_speed", 0.0)) for s in samples]
        obj_vx_series = [float(s.get("obj_vx", 0.0)) for s in samples]
        obj_vz_series = [float(s.get("obj_vz", 0.0)) for s in samples]
        ego_vx_series = [float(s.get("ego_vx", 0.0)) for s in samples]
        ego_vz_series = [float(s.get("ego_vz", 0.0)) for s in samples]
        mean_position = [
            float(np.mean([float(p[0]) for p in positions])) if positions else 0.0,
            float(np.mean([float(p[1]) for p in positions])) if positions else 0.0,
            float(np.mean([float(p[2]) for p in positions])) if positions else 0.0,
        ]

        rel_motion_samples = [s for s in samples if bool(s.get("has_rel_motion", False))]
        rel_vx_vals = [float(s.get("rel_vx", 0.0)) for s in rel_motion_samples]
        rel_vz_vals = [float(s.get("rel_vz", 0.0)) for s in rel_motion_samples]
        rel_speed_vals = [float(s.get("rel_speed", 0.0)) for s in rel_motion_samples]
        obj_vx_vals = [float(s.get("obj_vx", 0.0)) for s in rel_motion_samples]
        obj_vz_vals = [float(s.get("obj_vz", 0.0)) for s in rel_motion_samples]

        vz_symbolic = _classify_rel_vz(
            values=rel_vz_vals,
            threshold=rel_vz_threshold,
            dominance_ratio=dominance_ratio_threshold,
        )
        vx_symbolic = _classify_rel_vx(
            values=rel_vx_vals,
            threshold=rel_vx_threshold,
            dominance_ratio=dominance_ratio_threshold,
        )

        mean_rel_vx = float(np.mean(rel_vx_vals)) if rel_vx_vals else 0.0
        mean_rel_vz = float(np.mean(rel_vz_vals)) if rel_vz_vals else 0.0
        mean_rel_speed = float(np.mean(rel_speed_vals)) if rel_speed_vals else 0.0

        summary: Dict[str, Any] = {
            "track_id": int(track_id),
            "object_class": _majority_label(labels),
            "num_visible_frames": len(samples),
            "num_motion_frames": len(rel_motion_samples),
            "visibility_ratio": float(len(samples) / max(1, num_frames_in_segment)),
            "frame_indices": frame_indices,
            "has_rel_motion_mask": has_rel_motion_mask,
            "position_3d_series": positions,
            "rel_vx_series": rel_vx_series,
            "rel_vz_series": rel_vz_series,
            "rel_speed_series": rel_speed_series,
            "obj_vx_series": obj_vx_series,
            "obj_vz_series": obj_vz_series,
            "ego_vx_series": ego_vx_series,
            "ego_vz_series": ego_vz_series,
            "mean_position_3d": mean_position,
            "mean_rel_vx": mean_rel_vx,
            "mean_rel_vz": mean_rel_vz,
            "mean_rel_speed": mean_rel_speed,
            "mean_obj_vx": float(np.mean(obj_vx_vals)) if obj_vx_vals else 0.0,
            "mean_obj_vz": float(np.mean(obj_vz_vals)) if obj_vz_vals else 0.0,
            "vz_state": vz_symbolic["vz_state"],
            "vx_state": vx_symbolic["vx_state"],
            "speed_state": _classify_speed_state(mean_rel_speed, rel_speed_threshold),
            "distance_state": _classify_distance_state(
                mean_z=mean_position[2],
                near_threshold=distance_near_threshold,
                medium_threshold=distance_medium_threshold,
            ),
            "symbolic": {
                **vz_symbolic,
                **vx_symbolic,
            },
        }

        first = samples[0] if samples else {}
        source_detection_ids = sorted(
            {
                str(detection_id)
                for sample in samples
                for detection_id in list(sample.get("source_detection_ids", []))
                if str(detection_id)
            }
            | {
                str(sample.get("detection_id", ""))
                for sample in samples
                if str(sample.get("detection_id", ""))
            }
        )
        bbox_ids = sorted(
            {
                str(bbox_id)
                for sample in samples
                for bbox_id in list(sample.get("bbox_ids", []))
                if str(bbox_id)
            }
            | {
                str(sample.get("bbox_id", ""))
                for sample in samples
                if str(sample.get("bbox_id", ""))
            }
        )
        score_values = [float(sample.get("score", 0.0)) for sample in samples if sample.get("score") is not None]
        detector_score_stats = _score_stats(score_values)
        frame_range = [
            int(min(frame_indices)) if frame_indices else -1,
            int(max(frame_indices)) if frame_indices else -1,
        ]
        track_temporal_consistency = _temporal_consistency_proxy(frame_indices, num_frames_in_segment)
        accepted_source_type = "accepted_bbox"
        summary.update(
            {
                "accepted": True,
                "source_type": accepted_source_type,
                "frame_detection_id": str(first.get("detection_id", "")) if samples else "",
                "source_detection_ids": source_detection_ids,
                "bbox_ids": bbox_ids,
                "bbox_id_available": bool(bbox_ids),
                "detector_score_min": float(detector_score_stats["min"]),
                "detector_score_mean": float(detector_score_stats["mean"]),
                "detector_score_max": float(detector_score_stats["max"]),
                "track_length": len(frame_indices),
                "track_temporal_consistency": float(track_temporal_consistency),
                "frame_range": frame_range,
                "upstream_source_type": str(first.get("source_type", accepted_source_type)) if samples else accepted_source_type,
            }
        )

        if include_candidate_provenance:
            candidate_object_ids = sorted([
                str(candidate_object_id)
                for candidate_object_id in {
                    str(s.get("candidate_object_id", ""))
                    for s in samples
                    if s.get("candidate_object_id")
                }
            ])
            visibility_ratio = float(len(samples) / max(1, num_frames_in_segment))
            distance_series_meters = [
                float(position[2]) if len(position) > 2 else 0.0
                for position in positions
            ]
            distance_state_series = [
                str(sample.get("distance_state", _classify_distance_state(
                    mean_z=float(position[2]) if len(position) > 2 else 0.0,
                    near_threshold=distance_near_threshold,
                    medium_threshold=distance_medium_threshold,
                )))
                for sample, position in zip(samples, positions)
            ]
            vx_state_series = [
                str(sample.get("vx_state", "vx_unknown"))
                for sample in samples
            ]
            vz_state_series = [
                str(sample.get("vz_state", "vz_unknown"))
                for sample in samples
            ]
            speed_state_series = [
                str(sample.get("speed_state", "speed_unknown"))
                for sample in samples
            ]
            visibility_state = _visibility_state(visibility_ratio)
            summary.update(
                {
                    "accepted": False,
                    "source_type": str(first.get("source_type", "selected_candidate_track")),
                    "candidate_track_id": int(first.get("candidate_track_id", track_id)),
                    "candidate_object_id": candidate_object_ids[0] if candidate_object_ids else "",
                    "candidate_object_ids": candidate_object_ids,
                    "source_detection_ids": source_detection_ids,
                    "bbox_ids": bbox_ids,
                    "bbox_id_available": bool(bbox_ids),
                    "candidate_source": str(first.get("candidate_source", "")),
                    "prior_metadata": dict(first.get("prior_metadata", {})),
                    "score_breakdown": dict(first.get("score_breakdown", {})),
                    "selection_score": float(first.get("selection_score", 0.0)),
                    "track_quality": dict(first.get("track_quality", {})),
                    "detector_score_min": float(detector_score_stats["min"]),
                    "detector_score_mean": float(detector_score_stats["mean"]),
                    "detector_score_max": float(detector_score_stats["max"]),
                    "track_length": len(frame_indices),
                    "track_temporal_consistency": float(
                        dict(first.get("track_quality", {})).get("temporal_consistency", track_temporal_consistency)
                    ),
                    "frame_range": frame_range,
                    "position_3d_provenance": dict(first.get("position_3d_provenance", {})),
                    "relative_position_3d_series": positions,
                    "distance_series_meters": distance_series_meters,
                    "distance_state_series": distance_state_series,
                    "vx_state_series": vx_state_series,
                    "vz_state_series": vz_state_series,
                    "speed_state_series": speed_state_series,
                    "visibility_state": visibility_state,
                    "segment_ready_motion_features": {
                        "visibility_ratio": visibility_ratio,
                        "visibility_state": visibility_state,
                        "mean_position_3d": mean_position,
                        "mean_rel_vx": mean_rel_vx,
                        "mean_rel_vz": mean_rel_vz,
                        "mean_rel_speed": mean_rel_speed,
                        "distance_state": summary["distance_state"],
                        "vx_state": summary["vx_state"],
                        "vz_state": summary["vz_state"],
                        "speed_state": summary["speed_state"],
                        "num_motion_frames": len(rel_motion_samples),
                    },
                }
            )

        objects_out.append(summary)

    return objects_out


def _find_segment_object_summary(
    segment_summaries: List[Dict[str, Any]],
    frame_index: int,
    track_id: int,
) -> Optional[Dict[str, Any]]:
    for segment in segment_summaries:
        start_frame = int(segment.get("start_frame", 0))
        end_frame = int(segment.get("end_frame", start_frame))
        if frame_index < start_frame or frame_index > end_frame:
            continue
        for obj in segment.get("objects", []):
            if int(obj.get("track_id", -1)) == int(track_id):
                return {
                    "segment_index": int(segment.get("segment_index", -1)),
                    "segment_label": str(segment.get("segment_label", "")),
                    "segment_forward_label": str(segment.get("segment_forward_label", "")),
                    "segment_lateral_label": str(segment.get("segment_lateral_label", "")),
                    "object": obj,
                }
    return None


def _collect_object_segment_summaries(
    segment_summaries: List[Dict[str, Any]],
    track_id: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for segment in segment_summaries:
        for obj in segment.get("objects", []):
            if int(obj.get("track_id", -1)) != int(track_id):
                continue
            results.append(
                {
                    "segment_index": int(segment.get("segment_index", -1)),
                    "segment_label": str(segment.get("segment_label", "")),
                    "segment_forward_label": str(segment.get("segment_forward_label", "")),
                    "segment_lateral_label": str(segment.get("segment_lateral_label", "")),
                    "start_frame": int(segment.get("start_frame", 0)),
                    "end_frame": int(segment.get("end_frame", 0)),
                    "length": int(segment.get("length", 0)),
                    "object": obj,
                }
            )
            break
    return results


def _render_segment_summary_panel(
    width: int,
    height: int,
    object_segments: List[Dict[str, Any]],
    active_segment_index: int,
    header_text: str,
) -> np.ndarray:
    import cv2

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)
    cv2.putText(
        panel,
        header_text,
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    if not object_segments:
        cv2.putText(
            panel,
            "segment summary unavailable",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return panel

    top = 34
    gap = 8
    n = len(object_segments)
    usable_w = max(1, width - 24)
    section_w = max(140, int((usable_w - gap * max(0, n - 1)) / max(1, n)))
    total_w = n * section_w + gap * max(0, n - 1)
    if total_w > usable_w:
        section_w = max(100, int((usable_w - gap * max(0, n - 1)) / max(1, n)))
        total_w = n * section_w + gap * max(0, n - 1)
    start_x = max(12, 12 + (usable_w - total_w) // 2)
    section_h = height - top - 10

    for idx, seg in enumerate(object_segments):
        x0 = start_x + idx * (section_w + gap)
        x1 = min(width - 12, x0 + section_w)
        is_active = idx == active_segment_index
        fill = (62, 78, 110) if is_active else (34, 34, 34)
        border = (0, 255, 255) if is_active else (110, 110, 110)
        text = (255, 255, 255) if is_active else (215, 215, 215)

        cv2.rectangle(panel, (x0, top), (x1, top + section_h), fill, -1)
        cv2.rectangle(panel, (x0, top), (x1, top + section_h), border, 2 if is_active else 1)

        obj = seg["object"]
        lines = [
            f"seg {seg['segment_index']}",
            seg["segment_label"],
            f"vz: {obj.get('vz_state', 'vz_unknown')}",
            f"vx: {obj.get('vx_state', 'vx_unknown')}",
            f"spd: {obj.get('speed_state', 'rel_static')}",
            f"dist: {obj.get('distance_state', 'unknown')}",
            f"rvz={float(obj.get('mean_rel_vz', 0.0)):+.2f}",
            f"rvx={float(obj.get('mean_rel_vx', 0.0)):+.2f}",
        ]
        y = top + 18
        for line in lines:
            cv2.putText(
                panel,
                line,
                (x0 + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                text,
                1,
                cv2.LINE_AA,
            )
            y += 18

        if is_active:
            cv2.putText(
                panel,
                "ACTIVE",
                (x0 + 8, top + section_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return panel


def _build_object_segment_slots(
    samples: List[Dict[str, Any]],
    object_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sample_frame_indices = [int(s.get("frame_index", -1)) for s in samples]
    slots: List[Dict[str, Any]] = []
    for seg_idx, seg in enumerate(object_segments):
        start_frame = int(seg.get("start_frame", 0))
        end_frame = int(seg.get("end_frame", start_frame))
        covered = [
            i for i, frame_index in enumerate(sample_frame_indices)
            if start_frame <= frame_index <= end_frame
        ]
        if not covered:
            continue
        slots.append(
            {
                "segment_slot_index": seg_idx,
                "segment_index": int(seg.get("segment_index", -1)),
                "label": str(seg.get("segment_label", "")),
                "start_sample_index": covered[0],
                "end_sample_index": covered[-1],
            }
        )
    return slots


def _build_rel_vx_segment_comparison(
    samples: List[Dict[str, Any]],
    object_segments: List[Dict[str, Any]],
    thresholds: List[float],
    dominance_ratio_threshold: float,
) -> Dict[str, List[Dict[str, Any]]]:
    sample_frame_indices = [int(s.get("frame_index", -1)) for s in samples]
    rel_vx_by_frame_index = {
        int(s.get("frame_index", -1)): float(s.get("rel_vx", 0.0))
        for s in samples
        if bool(s.get("has_rel_motion", False))
    }
    comparison: Dict[str, List[Dict[str, Any]]] = {}
    for threshold in thresholds:
        slots: List[Dict[str, Any]] = []
        for seg_idx, seg in enumerate(object_segments):
            start_frame = int(seg.get("start_frame", 0))
            end_frame = int(seg.get("end_frame", start_frame))
            covered = [
                i for i, frame_index in enumerate(sample_frame_indices)
                if start_frame <= frame_index <= end_frame
            ]
            if not covered:
                continue
            values = [
                rel_vx_by_frame_index[frame_index]
                for frame_index in sample_frame_indices[covered[0] : covered[-1] + 1]
                if frame_index in rel_vx_by_frame_index
            ]
            vx_symbolic = _classify_rel_vx(
                values=values,
                threshold=float(threshold),
                dominance_ratio=float(dominance_ratio_threshold),
            )
            slots.append(
                {
                    "segment_slot_index": seg_idx,
                    "segment_index": int(seg.get("segment_index", -1)),
                    "label": vx_symbolic["vx_state"],
                    "start_sample_index": covered[0],
                    "end_sample_index": covered[-1],
                }
            )
        comparison[f"{float(threshold):g}"] = slots
    return comparison


def _render_object_signal_chart(
    width: int,
    height: int,
    title: str,
    signal_values: List[float],
    current_index: int,
    segment_slots: List[Dict[str, Any]],
    active_segment_slot: int,
    value_text: str = "",
) -> np.ndarray:
    import cv2

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (16, 16, 16)
    cv2.putText(
        panel,
        title,
        (12, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    if not signal_values:
        return panel

    top_pad = 26
    bottom_pad = 18
    left_pad = 18
    right_pad = 12
    chart_y0 = top_pad
    chart_y1 = max(chart_y0 + 12, height - bottom_pad)
    chart_h = chart_y1 - chart_y0
    chart_w = max(1, width - left_pad - right_pad)
    n = len(signal_values)

    overlay = panel.copy()
    for slot in segment_slots:
        start_idx = int(slot["start_sample_index"])
        end_idx = int(slot["end_sample_index"])
        x0 = left_pad + int(round(start_idx * chart_w / max(1, n)))
        x1 = left_pad + int(round((end_idx + 1) * chart_w / max(1, n)))
        is_active = int(slot["segment_slot_index"]) == int(active_segment_slot)
        color = (68, 90, 126) if is_active else (38, 38, 38)
        cv2.rectangle(overlay, (x0, chart_y0), (max(x0 + 1, x1), chart_y1), color, -1)
    panel = cv2.addWeighted(overlay, 0.28, panel, 0.72, 0.0)

    vals = np.asarray(signal_values, dtype=np.float32)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if abs(vmax - vmin) < 1e-6:
        vmin -= 1.0
        vmax += 1.0
    margin = 0.08 * (vmax - vmin)
    vmin -= margin
    vmax += margin

    def _to_y(v: float) -> int:
        alpha = (float(v) - vmin) / max(1e-6, (vmax - vmin))
        alpha = min(1.0, max(0.0, alpha))
        return int(round(chart_y1 - alpha * chart_h))

    if vmin < 0.0 < vmax:
        zero_y = _to_y(0.0)
        cv2.line(panel, (left_pad, zero_y), (left_pad + chart_w, zero_y), (88, 88, 88), 1)

    points = []
    for j, value in enumerate(vals):
        x = left_pad + int(round((j + 0.5) * chart_w / max(1, n)))
        y = _to_y(float(value))
        points.append([x, y])
    if len(points) >= 2:
        cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, (255, 255, 255), 2)
    elif points:
        cv2.circle(panel, tuple(points[0]), 2, (255, 255, 255), -1)

    cursor_idx = min(max(0, int(current_index)), n - 1)
    cursor_x = left_pad + int(round((cursor_idx + 0.5) * chart_w / max(1, n)))
    cv2.line(panel, (cursor_x, chart_y0 - 3), (cursor_x, chart_y1 + 3), (0, 255, 255), 2)
    cv2.circle(panel, tuple(points[cursor_idx]), 4, (0, 255, 255), -1)

    info_text = value_text or f"value={float(vals[cursor_idx]):+.3f}"
    cv2.putText(
        panel,
        info_text,
        (max(12, width - 360), 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return panel


def _render_top_object_videos(
    video_id: str,
    frames: List[Dict[str, Any]],
    segment_summaries: List[Dict[str, Any]],
    output_dir: Path,
    top_k: int = 20,
    fps: float = 10.0,
    compare_rel_vx_thresholds: Optional[List[float]] = None,
    dominance_ratio_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    try:
        import cv2
    except ModuleNotFoundError:
        return []

    if not frames:
        return []

    compare_rel_vx_thresholds = [
        float(v) for v in (compare_rel_vx_thresholds or [10.0, 20.0, 50.0])
    ]

    track_to_samples: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    track_to_labels: Dict[int, List[str]] = defaultdict(list)
    for frame in frames:
        frame_index = int(frame.get("frame_index", -1))
        for obj in frame.get("objects", []):
            sample = dict(obj)
            sample["frame_index"] = frame_index
            sample["image_path"] = frame.get("image_path", "")
            track_id = int(obj.get("track_id", -1))
            track_to_samples[track_id].append(sample)
            track_to_labels[track_id].append(str(obj.get("label", "unknown")))

    ranked_track_ids = sorted(
        track_to_samples.keys(),
        key=lambda tid: (-len(track_to_samples[tid]), tid),
    )[: max(0, int(top_k))]

    vis_dir = output_dir / "object_videos"
    vis_dir.mkdir(parents=True, exist_ok=True)
    rendered: List[Dict[str, Any]] = []

    for track_id in ranked_track_ids:
        samples = sorted(track_to_samples[track_id], key=lambda s: int(s.get("frame_index", -1)))
        if not samples:
            continue
        object_segments = _collect_object_segment_summaries(segment_summaries, track_id)
        segment_slots = _build_object_segment_slots(samples, object_segments)
        rel_vx_comparison = _build_rel_vx_segment_comparison(
            samples=samples,
            object_segments=object_segments,
            thresholds=compare_rel_vx_thresholds,
            dominance_ratio_threshold=dominance_ratio_threshold,
        )

        first_img = None
        for sample in samples:
            first_img = cv2.imread(_resolve_local_image_path(str(sample.get("image_path", ""))))
            if first_img is not None:
                break
        if first_img is None:
            continue

        object_class = _majority_label(track_to_labels[track_id])
        h, w = first_img.shape[:2]
        summary_h = 220
        chart_h = 86
        panel_h = summary_h + chart_h * (1 + len(compare_rel_vx_thresholds))
        out_path = vis_dir / f"track_{int(track_id):04d}_{_safe_slug(object_class)}.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (w, h + panel_h),
        )
        if not writer.isOpened():
            continue

        color = _track_color(int(track_id))
        visible_start = int(samples[0].get("frame_index", -1))
        visible_end = int(samples[-1].get("frame_index", -1))
        current_summary_cache: Optional[Dict[str, Any]] = None
        rel_vz_series = [float(s.get("rel_vz", 0.0)) if bool(s.get("has_rel_motion", False)) else 0.0 for s in samples]
        rel_vx_series = [float(s.get("rel_vx", 0.0)) if bool(s.get("has_rel_motion", False)) else 0.0 for s in samples]

        try:
            for sample_i, sample in enumerate(samples):
                image_path = _resolve_local_image_path(str(sample.get("image_path", "")))
                img = cv2.imread(image_path)
                if img is None:
                    continue
                if img.shape[0] != h or img.shape[1] != w:
                    img = cv2.resize(img, (w, h))

                frame_index = int(sample.get("frame_index", -1))
                box = [int(round(v)) for v in sample.get("box", [0, 0, 0, 0])]
                x1, y1, x2, y2 = box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                title = f"{video_id} | #{track_id} {object_class} | frame {frame_index:04d}"
                cv2.putText(
                    img,
                    title,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                current_summary = _find_segment_object_summary(segment_summaries, frame_index, track_id)
                if current_summary is not None:
                    current_summary_cache = current_summary
                active_summary = current_summary_cache

                active_segment_slot = 0
                if active_summary is not None:
                    active_seg_idx = int(active_summary.get("segment_index", -1))
                    for seg_i, seg in enumerate(object_segments):
                        if int(seg.get("segment_index", -1)) == active_seg_idx:
                            active_segment_slot = seg_i
                            break

                header_text = (
                    f"track_id={track_id} | class={object_class} | "
                    f"visible_frames={len(samples)} | span={visible_start}-{visible_end}"
                )
                summary_panel = _render_segment_summary_panel(
                    width=w,
                    height=summary_h,
                    object_segments=object_segments,
                    active_segment_index=active_segment_slot,
                    header_text=header_text,
                )
                vz_chart = _render_object_signal_chart(
                    width=w,
                    height=chart_h,
                    title="object rel_vz",
                    signal_values=rel_vz_series,
                    current_index=sample_i,
                    segment_slots=segment_slots,
                    active_segment_slot=active_segment_slot,
                    value_text=f"value={rel_vz_series[sample_i]:+.3f}",
                )
                vx_charts: List[np.ndarray] = []
                for threshold in compare_rel_vx_thresholds:
                    slots_for_threshold = rel_vx_comparison.get(f"{float(threshold):g}", [])
                    current_state = "vx_unknown"
                    for slot in slots_for_threshold:
                        if int(slot.get("segment_slot_index", -1)) == int(active_segment_slot):
                            current_state = str(slot.get("label", "vx_unknown"))
                            break
                    vx_charts.append(
                        _render_object_signal_chart(
                            width=w,
                            height=chart_h,
                            title=f"object rel_vx | th={float(threshold):g}",
                            signal_values=rel_vx_series,
                            current_index=sample_i,
                            segment_slots=slots_for_threshold,
                            active_segment_slot=active_segment_slot,
                            value_text=(
                                f"th={float(threshold):g} | value={rel_vx_series[sample_i]:+.3f} "
                                f"| label={current_state}"
                            ),
                        )
                    )
                panel = np.vstack([summary_panel, vz_chart, *vx_charts])

                canvas = np.vstack([img, panel])
                writer.write(canvas)
        finally:
            writer.release()

        rendered.append(
            {
                "track_id": int(track_id),
                "object_class": object_class,
                "num_visible_frames": len(samples),
                "num_segments": len(object_segments),
                "compare_rel_vx_thresholds": compare_rel_vx_thresholds,
                "start_frame": visible_start,
                "end_frame": visible_end,
                "path": str(out_path),
            }
        )

    return rendered


def process_video(
    relative_motion_video_result: Dict[str, Any],
    temporal_segmentation_video_result: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    rel_vz_threshold = float(cfg.get("rel_vz_threshold", 0.2))
    rel_vx_threshold = float(cfg.get("rel_vx_threshold", 0.2))
    compare_rel_vx_thresholds = list(cfg.get("compare_rel_vx_thresholds", [10.0, 20.0, 50.0]))
    rel_speed_threshold = float(cfg.get("rel_speed_threshold", 0.3))
    dominance_ratio_threshold = float(cfg.get("dominance_ratio_threshold", 0.6))
    distance_near_threshold = float(cfg.get("distance_near_threshold", 15.0))
    distance_medium_threshold = float(cfg.get("distance_medium_threshold", 30.0))
    top_k_visualized_objects = int(cfg.get("top_k_visualized_objects", 20))
    visualization_fps = float(cfg.get("visualization_fps", 10.0))
    render_videos = bool(cfg.get("render_videos", True))

    video_id = temporal_segmentation_video_result["video_id"]
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "segment_object_motion.json"

    if not force_recompute and out_file.exists():
        with out_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        cache_cfg = cached.get("config", {}) if isinstance(cached, dict) else {}
        if (
            int(cached.get("version", 0)) == _SEGMENT_OBJECT_MOTION_VERSION
            and _cfg_key_subset(cache_cfg) == _cfg_key_subset({
                "rel_vz_threshold": rel_vz_threshold,
                "rel_vx_threshold": rel_vx_threshold,
                "compare_rel_vx_thresholds": compare_rel_vx_thresholds,
                "rel_speed_threshold": rel_speed_threshold,
                "dominance_ratio_threshold": dominance_ratio_threshold,
                "distance_near_threshold": distance_near_threshold,
                "distance_medium_threshold": distance_medium_threshold,
                "top_k_visualized_objects": top_k_visualized_objects,
                "visualization_fps": visualization_fps,
            })
        ):
            print(f"  [cache] {video_id} - loading {out_file.name}")
            return cached

    frames = relative_motion_video_result.get("frames", [])
    segments = temporal_segmentation_video_result.get("segments", [])

    segment_summaries: List[Dict[str, Any]] = []
    objects_total = 0
    candidate_objects_total = 0

    for segment_index, segment in enumerate(segments):
        start_frame = int(segment.get("start_frame", 0))
        end_frame = int(segment.get("end_frame", start_frame))
        segment_label = str(segment.get("event", ""))
        split_label = _split_combined_label(segment_label)

        frames_in_segment = [
            frame
            for frame in frames
            if start_frame <= int(frame.get("frame_index", -1)) <= end_frame
        ]
        grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        grouped_candidates: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for frame in frames_in_segment:
            frame_index = int(frame.get("frame_index", -1))
            for obj in frame.get("objects", []):
                enriched = dict(obj)
                enriched["frame_index"] = frame_index
                grouped[int(obj.get("track_id", -1))].append(enriched)
            for candidate_obj in frame.get("candidate_objects", []):
                enriched_candidate = dict(candidate_obj)
                enriched_candidate["frame_index"] = frame_index
                grouped_candidates[int(candidate_obj.get("candidate_track_id", candidate_obj.get("track_id", -1)))].append(
                    enriched_candidate
                )

        objects_out = _summarize_segment_object_group(
            grouped=grouped,
            num_frames_in_segment=len(frames_in_segment),
            rel_vz_threshold=rel_vz_threshold,
            rel_vx_threshold=rel_vx_threshold,
            dominance_ratio_threshold=dominance_ratio_threshold,
            rel_speed_threshold=rel_speed_threshold,
            distance_near_threshold=distance_near_threshold,
            distance_medium_threshold=distance_medium_threshold,
            include_candidate_provenance=False,
        )
        candidate_objects_out = _summarize_segment_object_group(
            grouped=grouped_candidates,
            num_frames_in_segment=len(frames_in_segment),
            rel_vz_threshold=rel_vz_threshold,
            rel_vx_threshold=rel_vx_threshold,
            dominance_ratio_threshold=dominance_ratio_threshold,
            rel_speed_threshold=rel_speed_threshold,
            distance_near_threshold=distance_near_threshold,
            distance_medium_threshold=distance_medium_threshold,
            include_candidate_provenance=True,
        )

        objects_total += len(objects_out)
        candidate_objects_total += len(candidate_objects_out)
        segment_summaries.append(
            {
                "segment_index": int(segment_index),
                "segment_label": segment_label,
                "segment_forward_label": split_label["forward"],
                "segment_lateral_label": split_label["lateral"],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "length": int(segment.get("length", max(1, end_frame - start_frame + 1))),
                "num_frames_covered": len(frames_in_segment),
                "num_objects": len(objects_out),
                "num_candidate_objects": len(candidate_objects_out),
                "objects": objects_out,
                "candidate_objects": candidate_objects_out,
            }
        )

    object_video_paths: List[Dict[str, Any]] = []
    if render_videos:
        object_video_paths = _render_top_object_videos(
            video_id=video_id,
            frames=frames,
            segment_summaries=segment_summaries,
            output_dir=out_dir,
            top_k=top_k_visualized_objects,
            fps=visualization_fps,
            compare_rel_vx_thresholds=compare_rel_vx_thresholds,
            dominance_ratio_threshold=dominance_ratio_threshold,
        )

    result: Dict[str, Any] = {
        "version": _SEGMENT_OBJECT_MOTION_VERSION,
        "video_id": video_id,
        "num_segments": len(segment_summaries),
        "num_objects_total": objects_total,
        "num_candidate_objects_total": candidate_objects_total,
        "config": {
            "rel_vz_threshold": rel_vz_threshold,
            "rel_vx_threshold": rel_vx_threshold,
            "compare_rel_vx_thresholds": compare_rel_vx_thresholds,
            "rel_speed_threshold": rel_speed_threshold,
            "dominance_ratio_threshold": dominance_ratio_threshold,
            "distance_near_threshold": distance_near_threshold,
            "distance_medium_threshold": distance_medium_threshold,
            "top_k_visualized_objects": top_k_visualized_objects,
            "visualization_fps": visualization_fps,
            "render_videos": render_videos,
        },
        "segments": segment_summaries,
        "object_videos": object_video_paths,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(
        f"  {video_id}: {result['num_segments']} segments, "
        f"{result['num_objects_total']} segment-object summaries, "
        f"{result['num_candidate_objects_total']} candidate segment-object summaries"
    )
    return result


def run(
    relative_motion_results: List[Dict[str, Any]],
    temporal_segmentation_results: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    out_root = output_root or get_output_root()
    temporal_by_video = {
        r.get("video_id", ""): r for r in temporal_segmentation_results
    }

    results: List[Dict[str, Any]] = []
    for relative_result in relative_motion_results:
        video_id = relative_result.get("video_id", "unknown")
        temporal_result = temporal_by_video.get(video_id)
        if temporal_result is None:
            print(f"  [warn] Missing temporal segmentation for video {video_id}; skipping.")
            continue
        result = process_video(
            relative_motion_video_result=relative_result,
            temporal_segmentation_video_result=temporal_result,
            cfg=cfg,
            output_root=out_root,
            force_recompute=force_recompute,
        )
        results.append(result)

    manifest = {
        "version": _SEGMENT_OBJECT_MOTION_VERSION,
        "num_videos": len(results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_segments": r.get("num_segments", 0),
                "num_objects_total": r.get("num_objects_total", 0),
                "num_candidate_objects_total": r.get("num_candidate_objects_total", 0),
            }
            for r in results
        ],
    }
    manifest_path = out_root / "segment_object_motion_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Segment object motion manifest written to {manifest_path}")
    return results
