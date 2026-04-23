"""
Frame-level ego motion segmentation for nuScenes.

The segmentation here is intentionally lightweight: it converts the ego-local
forward and lateral velocity signals into discrete masks that later reasoning
stages can consume.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


FORWARD_LABELS = {
    0: "speed_up",
    1: "slow_down",
    2: "stable",
    3: "stop",
}

LATERAL_LABELS = {
    0: "left",
    1: "right",
    2: "stable",
}


def _to_1d_array(values: Sequence[float], expected_len: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if expected_len is None or arr.size == expected_len:
        return arr
    if arr.size > expected_len:
        return arr[:expected_len]
    if arr.size == 0:
        return np.zeros(expected_len, dtype=np.float64)
    pad = np.repeat(arr[-1], expected_len - arr.size)
    return np.concatenate([arr, pad])


def _centered_moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if values.size == 0 or window_size <= 1:
        return values.astype(np.float64, copy=True)

    window_size = int(window_size)
    if window_size % 2 == 0:
        window_size += 1

    half = window_size // 2
    padded = np.pad(values, (half, half), mode="edge")
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.convolve(padded, kernel, mode="valid")


def _forward_acceleration(forward_speed: np.ndarray, delta_t_s: np.ndarray) -> np.ndarray:
    acceleration = np.zeros_like(forward_speed, dtype=np.float64)
    if forward_speed.size <= 1:
        return acceleration

    dt = _to_1d_array(delta_t_s, expected_len=forward_speed.size)
    positive_dt = dt[dt > 0]
    fallback_dt = float(np.median(positive_dt)) if positive_dt.size else 0.5

    for index in range(1, forward_speed.size):
        frame_dt = float(dt[index]) if dt[index] > 0 else fallback_dt
        acceleration[index] = (forward_speed[index] - forward_speed[index - 1]) / frame_dt
    acceleration[0] = acceleration[1]
    return acceleration


def _smooth_short_label_runs(labels: np.ndarray, min_segment_frames: int) -> np.ndarray:
    if labels.size == 0 or min_segment_frames <= 1:
        return labels.astype(np.int64, copy=True)

    smoothed = labels.astype(np.int64, copy=True)
    index = 0
    while index < smoothed.size:
        start = index
        label = smoothed[index]
        while index < smoothed.size and smoothed[index] == label:
            index += 1
        end = index
        run_len = end - start
        if run_len >= min_segment_frames:
            continue

        prev_label = smoothed[start - 1] if start > 0 else None
        next_label = smoothed[end] if end < smoothed.size else None
        if prev_label is not None and next_label is not None and prev_label == next_label:
            smoothed[start:end] = prev_label
        elif prev_label is None and next_label is not None:
            smoothed[start:end] = next_label
        elif next_label is None and prev_label is not None:
            smoothed[start:end] = prev_label

    return smoothed


def _label_names(labels: np.ndarray, label_map: Mapping[int, str]) -> List[str]:
    return [label_map[int(label)] for label in labels]


def _segments_from_labels(
    labels: np.ndarray,
    label_map: Mapping[int, str],
    stats: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    if labels.size == 0:
        return []

    segments: List[Dict[str, Any]] = []
    start = 0
    current_label = int(labels[0])
    for index in range(1, labels.size + 1):
        if index < labels.size and int(labels[index]) == current_label:
            continue

        end = index - 1
        segment: Dict[str, Any] = {
            "start_frame": int(start),
            "end_frame": int(end),
            "duration_frames": int(end - start + 1),
            "label": current_label,
            "label_name": label_map[current_label],
        }
        for stat_name, values in stats.items():
            if values.size:
                segment[f"mean_{stat_name}"] = float(np.mean(values[start : end + 1]))
                segment[f"min_{stat_name}"] = float(np.min(values[start : end + 1]))
                segment[f"max_{stat_name}"] = float(np.max(values[start : end + 1]))
        segments.append(segment)

        if index < labels.size:
            start = index
            current_label = int(labels[index])

    return segments


def _timeline_segments_from_axis_labels(
    forward_labels: np.ndarray,
    lateral_labels: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Build maximal timeline segments where both axis labels are constant.

    A new segment starts whenever either the forward-axis label or the
    lateral-axis label changes.
    """
    frame_count = min(forward_labels.size, lateral_labels.size)
    if frame_count == 0:
        return []

    forward_labels = forward_labels[:frame_count]
    lateral_labels = lateral_labels[:frame_count]

    segments: List[Dict[str, Any]] = []
    start = 0
    current_forward = int(forward_labels[0])
    current_lateral = int(lateral_labels[0])

    for index in range(1, frame_count + 1):
        same_forward = index < frame_count and int(forward_labels[index]) == current_forward
        same_lateral = index < frame_count and int(lateral_labels[index]) == current_lateral
        if same_forward and same_lateral:
            continue

        end = index - 1
        forward_name = FORWARD_LABELS[current_forward]
        lateral_name = LATERAL_LABELS[current_lateral]
        segments.append(
            {
                "start_frame": int(start),
                "end_frame": int(end),
                "duration_frames": int(end - start + 1),
                "forward_label": current_forward,
                "forward_label_name": forward_name,
                "lateral_label": current_lateral,
                "lateral_label_name": lateral_name,
                "combined_label_name": f"{forward_name}+{lateral_name}",
            }
        )

        if index < frame_count:
            start = index
            current_forward = int(forward_labels[index])
            current_lateral = int(lateral_labels[index])

    return segments


def segment_ego_motion_signals(
    ego_motion: Dict[str, Any],
    forward_stop_threshold_mps: float = 0.25,
    forward_accel_threshold_mps2: float = 0.5,
    lateral_speed_threshold_mps: float = 0.1,
    min_segment_frames: int = 2,
    smoothing_window: int = 3,
) -> Dict[str, Any]:
    """
    Segment ego forward and lateral velocity into per-frame class masks.

    Forward mask labels:
        0 = speed_up, 1 = slow_down, 2 = stable, 3 = stop

    Lateral mask labels:
        0 = left, 1 = right, 2 = stable
    """
    forward_speed = _to_1d_array(ego_motion.get("forward_speed_mps", []))
    lateral_speed = _to_1d_array(ego_motion.get("lateral_speed_mps", []), expected_len=forward_speed.size)
    delta_t_s = _to_1d_array(ego_motion.get("delta_t_s", []), expected_len=forward_speed.size)

    forward_smoothed = _centered_moving_average(forward_speed, smoothing_window)
    lateral_smoothed = _centered_moving_average(lateral_speed, smoothing_window)
    forward_accel = _forward_acceleration(forward_smoothed, delta_t_s)

    forward_labels = np.full(forward_speed.size, 2, dtype=np.int64)
    forward_labels[forward_accel > forward_accel_threshold_mps2] = 0
    forward_labels[forward_accel < -forward_accel_threshold_mps2] = 1
    forward_labels[np.abs(forward_smoothed) <= forward_stop_threshold_mps] = 3
    forward_labels = _smooth_short_label_runs(forward_labels, min_segment_frames)

    lateral_labels = np.full(lateral_speed.size, 2, dtype=np.int64)
    lateral_labels[lateral_smoothed > lateral_speed_threshold_mps] = 0
    lateral_labels[lateral_smoothed < -lateral_speed_threshold_mps] = 1
    lateral_labels = _smooth_short_label_runs(lateral_labels, min_segment_frames)

    forward_segments = _segments_from_labels(
        forward_labels,
        FORWARD_LABELS,
        {
            "forward_speed_mps": forward_speed,
            "forward_speed_smoothed_mps": forward_smoothed,
            "forward_acceleration_mps2": forward_accel,
        },
    )
    lateral_segments = _segments_from_labels(
        lateral_labels,
        LATERAL_LABELS,
        {
            "lateral_speed_mps": lateral_speed,
            "lateral_speed_smoothed_mps": lateral_smoothed,
        },
    )
    timeline_segments = _timeline_segments_from_axis_labels(forward_labels, lateral_labels)

    return {
        "label_maps": {
            "forward": {str(key): value for key, value in FORWARD_LABELS.items()},
            "lateral": {str(key): value for key, value in LATERAL_LABELS.items()},
        },
        "params": {
            "forward_stop_threshold_mps": float(forward_stop_threshold_mps),
            "forward_accel_threshold_mps2": float(forward_accel_threshold_mps2),
            "lateral_speed_threshold_mps": float(lateral_speed_threshold_mps),
            "min_segment_frames": int(min_segment_frames),
            "smoothing_window": int(smoothing_window),
        },
        "signals": {
            "forward_speed_mps": forward_speed.tolist(),
            "forward_speed_smoothed_mps": forward_smoothed.tolist(),
            "forward_acceleration_mps2": forward_accel.tolist(),
            "lateral_speed_mps": lateral_speed.tolist(),
            "lateral_speed_smoothed_mps": lateral_smoothed.tolist(),
        },
        "forward_mask": forward_labels.astype(int).tolist(),
        "forward_label_names": _label_names(forward_labels, FORWARD_LABELS),
        "forward_segments": forward_segments,
        "lateral_mask": lateral_labels.astype(int).tolist(),
        "lateral_label_names": _label_names(lateral_labels, LATERAL_LABELS),
        "lateral_segments": lateral_segments,
        "timeline_segment_ranges": [
            [segment["start_frame"], segment["end_frame"]]
            for segment in timeline_segments
        ],
        "timeline_segments": timeline_segments,
    }


def summarize_motion_segments(segmentation: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact counts for scene summary JSON files."""
    summary: Dict[str, Any] = {}
    for axis_name, labels_key, segments_key in (
        ("forward", "forward_mask", "forward_segments"),
        ("lateral", "lateral_mask", "lateral_segments"),
    ):
        label_map = {int(key): value for key, value in segmentation["label_maps"][axis_name].items()}
        labels = [int(label) for label in segmentation.get(labels_key, [])]
        counts = {name: 0 for name in label_map.values()}
        for label in labels:
            counts[label_map[label]] += 1
        summary[axis_name] = {
            "frame_counts": counts,
            "num_segments": len(segmentation.get(segments_key, [])),
        }
    summary["timeline"] = {
        "num_segments": len(segmentation.get("timeline_segments", [])),
        "segment_ranges": segmentation.get("timeline_segment_ranges", []),
    }
    return summary
