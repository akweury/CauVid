"""
Temporal segmentation of ego-motion signals for driving_mini videos.

Segments the ego timeline into motion events and extracts cut points.
Primary event labels:
  - stop
  - turning
  - braking
  - speed_up
  - constant_speed

Consumes:
  - Step 6 output from ego_motion_driving_mini.run(...)

Output layout:
    pipeline_output/driving_mini_temporal_segmentation/
        temporal_segmentation_manifest.json
        <video_id>/
            temporal_segmentation.json
            temporal_segmentation_vis.mp4
"""

from __future__ import annotations

import json
import math
import sys
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


_EVENT_COLORS_BGR: Dict[str, tuple] = {
    "stop": (80, 80, 255),
    "turning": (30, 180, 255),
    "braking": (70, 70, 200),
    "speed_up": (70, 200, 70),
    "constant_speed": (180, 180, 180),
}

_SEGMENTATION_VIS_VERSION = 6


def _event_color_bgr(event_label: str) -> tuple:
    """Map combined event labels (forward|lateral) to stable timeline colors."""
    text = str(event_label or "").strip().lower()
    forward, lateral = text, ""
    if "|" in text:
        forward, lateral = text.split("|", 1)

    base = {
        "stopping": (80, 80, 255),
        "forward_speedup": (70, 200, 70),
        "forward_slowdown": (70, 70, 200),
        "forward_static_moving": (180, 180, 180),
        "backward_speedup": (40, 170, 255),
        "backward_slowdown": (190, 120, 255),
        "backward_static_moving": (120, 150, 190),
        "speedup": (70, 200, 70),
        "slowdown": (70, 70, 200),
        "static_moving": (180, 180, 180),
        "stop": (80, 80, 255),
        "speed_up": (70, 200, 70),
        "braking": (70, 70, 200),
        "constant_speed": (180, 180, 180),
    }.get(forward, (180, 180, 180))

    # Slight hue cue for lateral state.
    b, g, r = base
    if lateral in {"turning_left", "left"}:
        b = min(255, b + 35)
    elif lateral in {"turning_right", "right"}:
        g = min(255, g + 35)
    return (b, g, r)


def _lateral_event_color_bgr(event_label: str) -> tuple:
    text = str(event_label or "").strip().lower()
    return {
        "left": (30, 140, 255),            # orange
        "right": (255, 200, 40),           # cyan/yellow
        "straightforward": (150, 150, 150) # gray
    }.get(text, (150, 150, 150))


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "driving_mini_temporal_segmentation"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _bool_runs(mask: List[bool], min_len: int = 1) -> List[Dict[str, int]]:
    runs: List[Dict[str, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if (j - i) >= max(1, min_len):
            runs.append({"start": i, "end": j - 1, "length": j - i})
        i = j
    return runs


def _mask_from_runs(length: int, runs: List[Dict[str, int]]) -> List[bool]:
    out = [False] * length
    for run in runs:
        for idx in range(run["start"], run["end"] + 1):
            if 0 <= idx < length:
                out[idx] = True
    return out


def _segment_from_events(events: List[str], frame_indices: List[int]) -> List[Dict[str, Any]]:
    if not events:
        return []

    segments: List[Dict[str, Any]] = []
    start = 0
    for i in range(1, len(events)):
        if events[i] != events[i - 1]:
            segments.append(
                {
                    "event": events[start],
                    "start_idx": start,
                    "end_idx": i - 1,
                    "start_frame": frame_indices[start],
                    "end_frame": frame_indices[i - 1],
                    "length": i - start,
                }
            )
            start = i

    segments.append(
        {
            "event": events[start],
            "start_idx": start,
            "end_idx": len(events) - 1,
            "start_frame": frame_indices[start],
            "end_frame": frame_indices[-1],
            "length": len(events) - start,
        }
    )
    return segments


def _merge_short_segments(
    events: List[str],
    frame_indices: List[int],
    min_segment_length: int,
) -> Dict[str, Any]:
    """Merge segments shorter than min_segment_length into neighboring segments."""
    if not events:
        return {
            "events": events,
            "num_merged_segments": 0,
            "min_segment_length": int(min_segment_length),
        }

    min_len = max(1, int(min_segment_length))
    if min_len <= 1:
        return {
            "events": list(events),
            "num_merged_segments": 0,
            "min_segment_length": min_len,
        }

    merged_events = list(events)
    merged_count = 0

    for _ in range(len(events)):
        segments = _segment_from_events(merged_events, frame_indices)
        changed = False
        for i, seg in enumerate(segments):
            if seg["length"] >= min_len:
                continue

            # Pick neighbor label to absorb this short segment.
            left_seg = segments[i - 1] if i > 0 else None
            right_seg = segments[i + 1] if i + 1 < len(segments) else None

            if left_seg is None and right_seg is None:
                continue
            if left_seg is None:
                target_label = right_seg["event"]
            elif right_seg is None:
                target_label = left_seg["event"]
            else:
                if right_seg["length"] > left_seg["length"]:
                    target_label = right_seg["event"]
                else:
                    target_label = left_seg["event"]

            start = seg["start_idx"]
            end = seg["end_idx"]
            for idx in range(start, end + 1):
                merged_events[idx] = target_label

            merged_count += 1
            changed = True
            break

        if not changed:
            break

    return {
        "events": merged_events,
        "num_merged_segments": merged_count,
        "min_segment_length": min_len,
    }


def _combine_axis_events(forward_events: List[str], lateral_events: List[str]) -> List[str]:
    """Combine forward/lateral symbolic states into a single per-frame label."""
    n = min(len(forward_events), len(lateral_events))
    return [f"{forward_events[i]}|{lateral_events[i]}" for i in range(n)]


def _centered_moving_mean(values: List[float], window_size: int) -> List[float]:
    if not values:
        return []
    win = max(1, int(window_size))
    if win <= 1:
        return [float(v) for v in values]

    arr = np.asarray(values, dtype=np.float32)
    pad_left = win // 2
    pad_right = win - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return [float(v) for v in smoothed]


def _classify_forward_events(
    vz: List[float],
    speed: List[float],
    accel_threshold: float,
    stop_enter_threshold: float,
    stop_exit_threshold: float,
    stop_total_speed_enter: float,
    stop_total_speed_exit: float,
    min_stop_duration: int,
    stop_window_size: int,
    motion_window_size: int,
    direction_epsilon: float,
) -> Dict[str, Any]:
    """Classify signed forward-axis motion into forward/backward symbolic states."""
    n = len(vz)
    vz_trend = _centered_moving_mean(vz, motion_window_size)
    forward_accel: List[float] = [0.0] * n
    for i in range(1, n):
        forward_accel[i] = vz_trend[i] - vz_trend[i - 1]

    abs_vz_mean = _centered_moving_mean([abs(v) for v in vz], stop_window_size)
    total_speed_mean = _centered_moving_mean(speed, stop_window_size)

    stop_raw: List[bool] = [False] * n
    is_stopped = False
    for i in range(n):
        abs_vz = abs_vz_mean[i]
        total_speed = total_speed_mean[i]
        enter_stop = (abs_vz <= stop_enter_threshold) and (total_speed <= stop_total_speed_enter)
        stay_stop = (abs_vz <= stop_exit_threshold) and (total_speed <= stop_total_speed_exit)
        if is_stopped:
            is_stopped = stay_stop
        else:
            is_stopped = enter_stop
        stop_raw[i] = is_stopped

    stop_runs = _bool_runs(stop_raw, min_len=min_stop_duration)
    forward_stop_mask = _mask_from_runs(n, stop_runs)

    forward_speedup_mask: List[bool] = [False] * n
    forward_slowdown_mask: List[bool] = [False] * n
    forward_static_moving_mask: List[bool] = [False] * n
    backward_speedup_mask: List[bool] = [False] * n
    backward_slowdown_mask: List[bool] = [False] * n
    backward_static_moving_mask: List[bool] = [False] * n
    forward_event_raw: List[str] = []
    last_direction = 1

    for i in range(n):
        if forward_stop_mask[i]:
            forward_event_raw.append("stopping")
            continue

        if vz_trend[i] > direction_epsilon:
            moving_forward = True
            last_direction = 1
        elif vz_trend[i] < -direction_epsilon:
            moving_forward = False
            last_direction = -1
        else:
            moving_forward = last_direction >= 0
        accel = forward_accel[i]
        if moving_forward:
            if accel > accel_threshold:
                forward_speedup_mask[i] = True
                forward_event_raw.append("forward_speedup")
            elif accel < -accel_threshold:
                forward_slowdown_mask[i] = True
                forward_event_raw.append("forward_slowdown")
            else:
                forward_static_moving_mask[i] = True
                forward_event_raw.append("forward_static_moving")
        else:
            if accel < -accel_threshold:
                backward_speedup_mask[i] = True
                forward_event_raw.append("backward_speedup")
            elif accel > accel_threshold:
                backward_slowdown_mask[i] = True
                forward_event_raw.append("backward_slowdown")
            else:
                backward_static_moving_mask[i] = True
                forward_event_raw.append("backward_static_moving")

    return {
        "vz_trend": vz_trend,
        "forward_accel": forward_accel,
        "abs_vz_mean": abs_vz_mean,
        "total_speed_mean": total_speed_mean,
        "stop_raw": stop_raw,
        "forward_stop_mask": forward_stop_mask,
        "forward_speedup_mask": forward_speedup_mask,
        "forward_slowdown_mask": forward_slowdown_mask,
        "forward_static_moving_mask": forward_static_moving_mask,
        "backward_speedup_mask": backward_speedup_mask,
        "backward_slowdown_mask": backward_slowdown_mask,
        "backward_static_moving_mask": backward_static_moving_mask,
        "forward_event_raw": forward_event_raw,
    }


def _classify_lateral_events(
    vx: List[float],
    lateral_turn_threshold: float,
    lateral_straight_threshold: float,
    min_turn_duration: int,
    motion_window_size: int,
    direction_epsilon: float,
) -> Dict[str, Any]:
    """Classify lateral motion into left/right/straightforward using a smoothed trend."""
    vx_trend = _centered_moving_mean(vx, motion_window_size)
    straight_threshold = max(0.0, float(lateral_straight_threshold))
    turn_threshold = max(float(lateral_turn_threshold), float(direction_epsilon), straight_threshold + 1e-6)

    lateral_left_raw = [vx_i > turn_threshold for vx_i in vx_trend]
    lateral_right_raw = [vx_i < -turn_threshold for vx_i in vx_trend]

    left_runs = _bool_runs(lateral_left_raw, min_len=min_turn_duration)
    right_runs = _bool_runs(lateral_right_raw, min_len=min_turn_duration)
    lateral_left_mask = _mask_from_runs(len(vx), left_runs)
    lateral_right_mask = _mask_from_runs(len(vx), right_runs)
    lateral_straight_mask = [
        (not lateral_left_mask[i]) and (not lateral_right_mask[i])
        for i in range(len(vx))
    ]

    lateral_event_raw: List[str] = []
    for i in range(len(vx)):
        if abs(vx_trend[i]) <= straight_threshold:
            lateral_event_raw.append("straightforward")
        elif lateral_left_mask[i]:
            lateral_event_raw.append("left")
        elif lateral_right_mask[i]:
            lateral_event_raw.append("right")
        else:
            lateral_event_raw.append("straightforward")

    return {
        "vx_trend": vx_trend,
        "lateral_left_mask": lateral_left_mask,
        "lateral_right_mask": lateral_right_mask,
        "lateral_straight_mask": lateral_straight_mask,
        "lateral_event_raw": lateral_event_raw,
    }


def _build_merged_segments_from_cutpoints(
    frame_indices: List[int],
    forward_events: List[str],
    lateral_events: List[str],
) -> Dict[str, Any]:
    """
    Merge axis cut points (from forward + lateral event changes) into timeline segments.
    """
    n = min(len(frame_indices), len(forward_events), len(lateral_events))
    if n == 0:
        return {"segments": [], "cut_points": []}

    cut_idx = {0}
    for i in range(1, n):
        if forward_events[i] != forward_events[i - 1] or lateral_events[i] != lateral_events[i - 1]:
            cut_idx.add(i)
    cut_idx_sorted = sorted(cut_idx)

    segments: List[Dict[str, Any]] = []
    for k, start in enumerate(cut_idx_sorted):
        end = (cut_idx_sorted[k + 1] - 1) if (k + 1 < len(cut_idx_sorted)) else (n - 1)
        f_ev = forward_events[start]
        l_ev = lateral_events[start]
        segments.append(
            {
                "event": f"{f_ev}|{l_ev}",
                "forward_event": f_ev,
                "lateral_event": l_ev,
                "start_idx": start,
                "end_idx": end,
                "start_frame": frame_indices[start],
                "end_frame": frame_indices[end],
                "length": end - start + 1,
            }
        )

    cut_points = [frame_indices[idx] for idx in cut_idx_sorted]
    return {"segments": segments, "cut_points": cut_points}


def _is_static_label(label: Any, keywords: List[str]) -> bool:
    text = str(label).strip().lower()
    if not text:
        return False
    return any(k in text for k in keywords)


def _build_relative_frame_map(relative_video_result: Optional[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    if not relative_video_result:
        return {}
    return {
        int(frame.get("frame_index", idx)): frame
        for idx, frame in enumerate(relative_video_result.get("frames", []))
    }


def _apply_symbolic_static_correction(
    primary_event: List[str],
    frame_indices: List[int],
    relative_video_result: Optional[Dict[str, Any]],
    cfg: Dict[str, Any],
    stop_label: str = "stop",
) -> Dict[str, Any]:
    symbolic_cfg = cfg.get("symbolic_correction", {}) if isinstance(cfg, dict) else {}
    enabled = bool(symbolic_cfg.get("enabled", True))

    corrected = list(primary_event)
    if not enabled:
        return {
            "events": corrected,
            "num_corrected": 0,
            "corrected_frame_indices": [],
            "config": symbolic_cfg,
        }

    static_keywords = [
        str(v).strip().lower()
        for v in symbolic_cfg.get("static_object_keywords", ["traffic light", "building"])
    ]
    min_static_objects = int(symbolic_cfg.get("min_static_objects", 1))
    max_static_rel_vz_abs = float(symbolic_cfg.get("max_static_rel_vz_abs", 0.03))
    max_static_rel_speed = float(symbolic_cfg.get("max_static_rel_speed", 0.05))

    rel_by_frame = _build_relative_frame_map(relative_video_result)
    corrected_frame_indices: List[int] = []

    for i, frame_index in enumerate(frame_indices):
        rel_frame = rel_by_frame.get(frame_index, {})
        objects = rel_frame.get("objects", [])
        static_objs = [
            obj for obj in objects
            if obj.get("has_rel_motion", False)
            and _is_static_label(obj.get("label", ""), static_keywords)
        ]
        if len(static_objs) < min_static_objects:
            continue

        rel_vz_vals = [abs(float(obj.get("rel_vz", 0.0))) for obj in static_objs]
        rel_speed_vals = [abs(float(obj.get("rel_speed", 0.0))) for obj in static_objs]
        med_rel_vz_abs = float(np.median(rel_vz_vals)) if rel_vz_vals else 0.0
        med_rel_speed = float(np.median(rel_speed_vals)) if rel_speed_vals else 0.0

        # Symbolic rule: if distance/motion to static references is near-constant,
        # ego should be considered stopped.
        if med_rel_vz_abs <= max_static_rel_vz_abs and med_rel_speed <= max_static_rel_speed:
            if corrected[i] != stop_label:
                corrected[i] = stop_label
                corrected_frame_indices.append(frame_index)

    return {
        "events": corrected,
        "num_corrected": len(corrected_frame_indices),
        "corrected_frame_indices": corrected_frame_indices,
        "config": {
            "enabled": enabled,
            "static_object_keywords": static_keywords,
            "min_static_objects": min_static_objects,
            "max_static_rel_vz_abs": max_static_rel_vz_abs,
            "max_static_rel_speed": max_static_rel_speed,
        },
    }


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


def _render_signal_chart_panel(
    width: int,
    height: int,
    title: str,
    signal_values: List[float],
    event_series: List[str],
    frame_indices: List[int],
    current_idx: int,
    cut_points: List[int],
    color_fn,
    fallback_event: str,
    signal_label: str,
    footer_prefix: str = "",
) -> np.ndarray:
    """Render a line chart with segmentation-colored background and current-frame cursor."""
    try:
        import cv2
    except ModuleNotFoundError:
        return np.zeros((height, width, 3), dtype=np.uint8)

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)
    cv2.putText(
        panel,
        title,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    if not signal_values:
        return panel

    top_pad = 34
    bottom_pad = 26
    left_pad = 18
    right_pad = 12
    chart_y0 = top_pad
    chart_y1 = max(chart_y0 + 20, height - bottom_pad)
    chart_h = chart_y1 - chart_y0
    chart_w = max(1, width - left_pad - right_pad)

    n = max(1, len(signal_values), len(event_series), len(frame_indices))
    cut_points_set = set(int(v) for v in cut_points)

    overlay = panel.copy()
    for j in range(n):
        x0 = left_pad + int(round(j * chart_w / n))
        x1 = left_pad + int(round((j + 1) * chart_w / n))
        ev = event_series[j] if j < len(event_series) else fallback_event
        cv2.rectangle(
            overlay,
            (x0, chart_y0),
            (max(x0 + 1, x1), chart_y1),
            color_fn(ev),
            thickness=-1,
        )
    panel = cv2.addWeighted(overlay, 0.22, panel, 0.78, 0.0)

    vals = np.asarray(signal_values, dtype=np.float32)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if abs(vmax - vmin) < 1e-6:
        vmax += 1.0
        vmin -= 1.0
    margin = 0.08 * (vmax - vmin)
    vmin -= margin
    vmax += margin

    def _to_y(v: float) -> int:
        alpha = (float(v) - vmin) / max(1e-6, (vmax - vmin))
        alpha = min(1.0, max(0.0, alpha))
        return int(round(chart_y1 - alpha * chart_h))

    if vmin < 0.0 < vmax:
        zero_y = _to_y(0.0)
        cv2.line(panel, (left_pad, zero_y), (left_pad + chart_w, zero_y), (80, 80, 80), 1)

    points = []
    for j, v in enumerate(vals):
        x = left_pad + int(round((j + 0.5) * chart_w / n))
        y = _to_y(float(v))
        points.append([x, y])
    if len(points) >= 2:
        cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], isClosed=False, color=(255, 255, 255), thickness=2)
    elif points:
        cv2.circle(panel, tuple(points[0]), 2, (255, 255, 255), -1)

    for j, fidx in enumerate(frame_indices):
        if fidx in cut_points_set:
            x = left_pad + int(round((j + 0.5) * chart_w / n))
            cv2.line(panel, (x, chart_y0 - 4), (x, chart_y1 + 4), (180, 180, 180), 1)

    cursor_x = left_pad + int(round((min(current_idx, n - 1) + 0.5) * chart_w / n))
    cv2.line(panel, (cursor_x, chart_y0 - 6), (cursor_x, chart_y1 + 6), (0, 255, 255), 2)
    if current_idx < len(points):
        cv2.circle(panel, tuple(points[current_idx]), 4, (0, 255, 255), -1)

    current_value = float(vals[min(current_idx, len(vals) - 1)])
    current_event = (
        event_series[min(current_idx, len(event_series) - 1)]
        if event_series else fallback_event
    )
    footer_text = f"{signal_label}: {current_value:+.4f}"
    footer_text = f"{footer_text} | label={current_event}"
    if footer_prefix:
        footer_text = f"{footer_prefix} | {footer_text}"
    cv2.putText(
        panel,
        footer_text,
        (10, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    return panel


def _render_single_temporal_segmentation_video(
    video_id: str,
    ego_frames: List[Dict[str, Any]],
    output_path: Path,
    chart_specs: List[Dict[str, Any]],
    fps: float = 10.0,
) -> Optional[str]:
    """Render one timeline segmentation stream as a standalone MP4."""
    try:
        import cv2
    except ModuleNotFoundError:
        return None

    if not ego_frames:
        return None

    first_img = None
    for frame in ego_frames:
        first_img = cv2.imread(_resolve_local_image_path(frame.get("image_path", "")))
        if first_img is not None:
            break
    if first_img is None:
        return None

    h, w = first_img.shape[:2]
    if not chart_specs:
        return None
    panel_h = max(90, min(160, 900 // max(1, len(chart_specs))))
    timeline_h = panel_h * len(chart_specs)
    out_size = (w, h + timeline_h)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        out_size,
    )
    if not writer.isOpened():
        return None

    frame_indices = [int(f.get("frame_index", i)) for i, f in enumerate(ego_frames)]

    try:
        for i, frame in enumerate(ego_frames):
            img = cv2.imread(_resolve_local_image_path(frame.get("image_path", "")))
            if img is None:
                img = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (w, h))

            cv2.putText(
                img,
                f"{video_id} | frame {frame_indices[i]:04d}",
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if len(chart_specs) == 1:
                spec = chart_specs[0]
                event_series = spec["event_series"]
                fallback_event = spec["fallback_event"]
                event = event_series[i] if i < len(event_series) else fallback_event
                color = spec["color_fn"](event)
                cv2.putText(
                    img,
                    f"{spec['current_label_prefix']}: {event}",
                    (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            panels = []
            for spec in chart_specs:
                panels.append(
                    _render_signal_chart_panel(
                        width=w,
                        height=panel_h,
                        title=spec["title"],
                        signal_values=spec["signal_values"],
                        event_series=spec["event_series"],
                        frame_indices=frame_indices,
                        current_idx=i,
                        cut_points=spec["cut_points"],
                        color_fn=spec["color_fn"],
                        fallback_event=spec["fallback_event"],
                        signal_label=spec["signal_label"],
                        footer_prefix=spec.get("footer_prefix", ""),
                    )
                )

            panel = np.vstack(panels)
            out_frame = np.vstack([img, panel])
            writer.write(out_frame)
    finally:
        writer.release()

    return str(output_path)


def _render_temporal_segmentation_videos(
    video_id: str,
    ego_frames: List[Dict[str, Any]],
    segmentation_result: Dict[str, Any],
    output_path: Path,
    fps: float = 10.0,
) -> Dict[str, str]:
    """Render separate MP4 files for each segmentation stream."""
    forward_events_raw = segmentation_result.get(
        "forward_event_raw",
        segmentation_result.get("forward_event", []),
    )
    forward_events_processed = segmentation_result.get(
        "forward_event",
        forward_events_raw,
    )
    lateral_events = segmentation_result.get("lateral_event", [])
    events_raw = segmentation_result.get("primary_event_raw", segmentation_result.get("primary_event", []))
    events_processed = segmentation_result.get("primary_event", [])
    frame_indices = [int(f.get("frame_index", i)) for i, f in enumerate(ego_frames)]
    speed_values = [float(v) for v in segmentation_result.get("signals", {}).get("speed", [])]
    forward_signal = [float(f.get("ego_vz_smoothed", f.get("ego_vz", 0.0))) for f in ego_frames]
    lateral_signal = [
        float(v) for v in segmentation_result.get(
            "signals", {}
        ).get(
            "vx_trend",
            [float(f.get("ego_vx_smoothed", f.get("ego_vx", 0.0))) for f in ego_frames],
        )
    ]

    raw_segments = _segment_from_events(events_raw, frame_indices) if events_raw else []
    cut_points_raw = [int(seg.get("start_frame", 0)) for seg in raw_segments]
    forward_raw_segments = _segment_from_events(forward_events_raw, frame_indices) if forward_events_raw else []
    cut_points_forward_raw = [int(seg.get("start_frame", 0)) for seg in forward_raw_segments]
    forward_processed_segments = _segment_from_events(forward_events_processed, frame_indices) if forward_events_processed else []
    cut_points_forward_processed = [int(seg.get("start_frame", 0)) for seg in forward_processed_segments]
    lateral_segments = _segment_from_events(lateral_events, frame_indices) if lateral_events else []
    cut_points_lateral = [int(seg.get("start_frame", 0)) for seg in lateral_segments]
    cut_points_processed = [int(v) for v in segmentation_result.get("cut_points", [])]

    forward_compare_specs = segmentation_result.get("forward_comparison_variants", [])
    if forward_compare_specs:
        hydrated_specs = []
        for spec in forward_compare_specs:
            hydrated = dict(spec)
            hydrated["color_fn"] = lambda ev: _event_color_bgr(f"{ev}|straightforward")
            hydrated.setdefault("fallback_event", "forward_static_moving")
            hydrated.setdefault("signal_label", "vz")
            stop_threshold = hydrated.get("stop_threshold")
            stop_duration = hydrated.get("min_stop_duration")
            segment_length = hydrated.get("min_segment_length")
            if stop_threshold is not None and stop_duration is not None and segment_length is not None:
                hydrated.setdefault(
                    "footer_prefix",
                    f"stop_th={float(stop_threshold):.3f} | min_stop={int(stop_duration)} | min_seg={int(segment_length)}",
                )
            elif stop_duration is not None and segment_length is not None:
                hydrated.setdefault(
                    "footer_prefix",
                    f"min_stop={int(stop_duration)} | min_seg={int(segment_length)}",
                )
            hydrated_specs.append(hydrated)
        forward_compare_specs = hydrated_specs
    if not forward_compare_specs:
        forward_compare_specs = [
            {
                "title": "forward segmentation | active config",
                "signal_values": forward_signal,
                "event_series": forward_events_processed,
                "cut_points": cut_points_forward_processed,
                "color_fn": lambda ev: _event_color_bgr(f"{ev}|straightforward"),
                "fallback_event": "forward_static_moving",
                "current_label_prefix": "forward",
                "signal_label": "vz",
            }
        ]

    lateral_compare_specs = segmentation_result.get("lateral_comparison_variants", [])
    if lateral_compare_specs:
        hydrated_specs = []
        for spec in lateral_compare_specs:
            hydrated = dict(spec)
            hydrated["color_fn"] = _lateral_event_color_bgr
            hydrated.setdefault("fallback_event", "straightforward")
            hydrated.setdefault("signal_label", "vx")
            straight_threshold = hydrated.get("lateral_straight_threshold")
            motion_window_size = hydrated.get("lateral_motion_window_size")
            if straight_threshold is not None and motion_window_size is not None:
                hydrated.setdefault(
                    "footer_prefix",
                    f"lat_straight={float(straight_threshold):.3f} | lat_win={int(motion_window_size)}",
                )
            hydrated_specs.append(hydrated)
        lateral_compare_specs = hydrated_specs
    if not lateral_compare_specs:
        lateral_compare_specs = [
            {
                "title": "lateral segmentation",
                "signal_values": lateral_signal,
                "event_series": lateral_events,
                "cut_points": cut_points_lateral,
                "color_fn": _lateral_event_color_bgr,
                "fallback_event": "straightforward",
                "current_label_prefix": "lateral",
                "signal_label": "vx",
                "footer_prefix": (
                    f"lat_eps={float(segmentation_result.get('config', {}).get('lateral_direction_epsilon', 0.0)):.3f} | "
                    f"lat_straight={float(segmentation_result.get('config', {}).get('lateral_straight_threshold', 0.0)):.3f} | "
                    f"lat_win={int(segmentation_result.get('config', {}).get('lateral_motion_window_size', 1))}"
                ),
            }
        ]

    vis_specs = [
        (
            "primary_raw",
            [
                {
                    "title": "raw primary segmentation",
                    "signal_values": speed_values,
                    "event_series": events_raw,
                    "cut_points": cut_points_raw,
                    "color_fn": lambda ev: _event_color_bgr(ev),
                    "fallback_event": "constant_speed",
                    "current_label_prefix": "raw",
                    "signal_label": "speed",
                }
            ],
        ),
        (
            "primary",
            [
                {
                    "title": "processed primary segmentation",
                    "signal_values": speed_values,
                    "event_series": events_processed,
                    "cut_points": cut_points_processed,
                    "color_fn": lambda ev: _event_color_bgr(ev),
                    "fallback_event": "constant_speed",
                    "current_label_prefix": "processed",
                    "signal_label": "speed",
                }
            ],
        ),
        (
            "forward_raw",
            [
                {
                    "title": "forward raw segmentation",
                    "signal_values": forward_signal,
                    "event_series": forward_events_raw,
                    "cut_points": cut_points_forward_raw,
                    "color_fn": lambda ev: _event_color_bgr(f"{ev}|straightforward"),
                    "fallback_event": "forward_static_moving",
                    "current_label_prefix": "forward raw",
                    "signal_label": "vz",
                }
            ],
        ),
        (
            "forward",
            forward_compare_specs,
        ),
        (
            "lateral",
            lateral_compare_specs,
        ),
    ]

    rendered: Dict[str, str] = {}
    for stem, chart_specs in vis_specs:
        single_path = output_path.parent / f"temporal_segmentation_{stem}_vis.mp4"
        out = _render_single_temporal_segmentation_video(
            video_id=video_id,
            ego_frames=ego_frames,
            output_path=single_path,
            chart_specs=chart_specs,
            fps=fps,
        )
        if out:
            rendered[stem] = out
    return rendered

def process_video(
    ego_video_result: Dict[str, Any],
    relative_motion_video_result: Optional[Dict[str, Any]] = None,
    seg_cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = seg_cfg or {}
    stop_speed_threshold = float(cfg.get("forward_stop_threshold", cfg.get("stop_speed_threshold", 0.05)))
    stop_enter_threshold = float(cfg.get("forward_stop_enter_threshold", stop_speed_threshold))
    stop_exit_threshold = float(cfg.get("forward_stop_exit_threshold", stop_speed_threshold * 1.6))
    stop_total_speed_enter = float(cfg.get("stop_total_speed_enter_threshold", stop_enter_threshold * 1.8))
    stop_total_speed_exit = float(cfg.get("stop_total_speed_exit_threshold", stop_exit_threshold * 1.8))
    accel_threshold = float(cfg.get("forward_accel_threshold", cfg.get("accel_threshold", 0.03)))
    lateral_turn_threshold = float(cfg.get("lateral_turn_threshold", 0.03))
    stop_window_size = int(cfg.get("stop_window_size", 5))
    motion_window_size = int(cfg.get("motion_window_size", 5))
    direction_epsilon = float(cfg.get("forward_direction_epsilon", max(1e-3, stop_enter_threshold * 0.5)))
    lateral_motion_window_size = int(cfg.get("lateral_motion_window_size", motion_window_size))
    lateral_direction_epsilon = float(cfg.get("lateral_direction_epsilon", max(1e-3, lateral_turn_threshold)))
    lateral_straight_threshold = float(cfg.get("lateral_straight_threshold", 45.0))
    compare_lateral_straight_thresholds = [
        float(v) for v in cfg.get("compare_lateral_straight_thresholds", [15, 25, 35, 45])
    ]
    min_stop_duration = int(cfg.get("min_stop_duration", 3))
    min_turn_duration = int(cfg.get("min_turn_duration", 3))
    min_segment_length = int(cfg.get("min_segment_length", 1))
    compare_forward_stop_thresholds = [
        float(v) for v in cfg.get("compare_forward_stop_thresholds", [1.0])
    ]
    compare_min_segment_lengths = [
        int(v) for v in cfg.get("compare_min_segment_lengths", [7])
    ]
    compare_forward_stop_thresholds = sorted({max(1e-6, v) for v in compare_forward_stop_thresholds})
    compare_min_segment_lengths = sorted({max(1, v) for v in compare_min_segment_lengths})
    compare_lateral_straight_thresholds = sorted({max(0.0, v) for v in compare_lateral_straight_thresholds})

    video_id = ego_video_result["video_id"]
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "temporal_segmentation.json"

    if not force_recompute and out_file.exists():
        print(f"  [cache] {video_id} - loading {out_file.name}")
        with out_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)

        symbolic_enabled = bool((cfg.get("symbolic_correction", {}) or {}).get("enabled", True))
        has_symbolic_meta = "symbolic_correction" in cached and "primary_event_raw" in cached
        cache_stale_for_symbolic = symbolic_enabled and (relative_motion_video_result is not None) and (not has_symbolic_meta)
        cached_min_segment = int(cached.get("config", {}).get("min_segment_length", 1))
        cache_stale_for_min_segment = cached_min_segment != min_segment_length
        cache_stale_for_vis_version = int(cached.get("visualization_version", 1)) < _SEGMENTATION_VIS_VERSION
        has_axis_meta = ("forward_event" in cached) and ("lateral_event" in cached)
        cache_stale_for_axis_mode = not has_axis_meta
        cache_cfg = cached.get("config", {}) if isinstance(cached, dict) else {}
        cache_stale_for_cfg_change = (
            float(cache_cfg.get("forward_stop_threshold", cache_cfg.get("stop_speed_threshold", stop_speed_threshold)))
            != float(stop_speed_threshold)
            or float(cache_cfg.get("forward_stop_enter_threshold", stop_enter_threshold))
            != float(stop_enter_threshold)
            or float(cache_cfg.get("forward_stop_exit_threshold", stop_exit_threshold))
            != float(stop_exit_threshold)
            or float(cache_cfg.get("stop_total_speed_enter_threshold", stop_total_speed_enter))
            != float(stop_total_speed_enter)
            or float(cache_cfg.get("stop_total_speed_exit_threshold", stop_total_speed_exit))
            != float(stop_total_speed_exit)
            or float(cache_cfg.get("lateral_turn_threshold", lateral_turn_threshold))
            != float(lateral_turn_threshold)
            or float(cache_cfg.get("forward_accel_threshold", cache_cfg.get("accel_threshold", accel_threshold)))
            != float(accel_threshold)
            or int(cache_cfg.get("stop_window_size", stop_window_size)) != int(stop_window_size)
            or int(cache_cfg.get("motion_window_size", motion_window_size)) != int(motion_window_size)
            or float(cache_cfg.get("forward_direction_epsilon", direction_epsilon)) != float(direction_epsilon)
            or int(cache_cfg.get("lateral_motion_window_size", lateral_motion_window_size)) != int(lateral_motion_window_size)
            or float(cache_cfg.get("lateral_direction_epsilon", lateral_direction_epsilon)) != float(lateral_direction_epsilon)
            or float(cache_cfg.get("lateral_straight_threshold", lateral_straight_threshold)) != float(lateral_straight_threshold)
            or int(cache_cfg.get("min_stop_duration", min_stop_duration)) != int(min_stop_duration)
            or int(cache_cfg.get("min_turn_duration", min_turn_duration)) != int(min_turn_duration)
        )

        if cache_stale_for_axis_mode:
            print(f"  [cache] {video_id} - cache is stale; recomputing axis-wise segmentation")
        elif cache_stale_for_cfg_change:
            print(f"  [cache] {video_id} - cache is stale; recomputing with updated segmentation thresholds")
        elif cache_stale_for_symbolic:
            print(f"  [cache] {video_id} - cache is stale; recomputing symbolic correction")
        elif cache_stale_for_min_segment:
            print(f"  [cache] {video_id} - cache is stale; recomputing short-segment merge")
        elif cache_stale_for_vis_version:
            print(f"  [cache] {video_id} - cache visualization is stale; rerendering comparison view")
        else:
            cached_vis_paths = cached.get("visualization_paths", {})
            if (
                isinstance(cached_vis_paths, dict)
                and cached_vis_paths
                and all(Path(p).exists() for p in cached_vis_paths.values())
            ):
                return cached

            rendered_cached = _render_temporal_segmentation_videos(
                video_id=video_id,
                ego_frames=ego_video_result.get("frames", []),
                segmentation_result=cached,
                output_path=out_dir / "temporal_segmentation_vis.mp4",
                fps=float(cfg.get("visualization_fps", 10.0)),
            )
            if rendered_cached:
                cached["visualization_paths"] = rendered_cached
                cached["visualization_path"] = rendered_cached.get("primary", "")
                cached["visualization_version"] = _SEGMENTATION_VIS_VERSION
                with out_file.open("w", encoding="utf-8") as fh:
                    json.dump(cached, fh, indent=2)
            return cached

    frames = ego_video_result.get("frames", [])
    frame_indices = [int(f.get("frame_index", i)) for i, f in enumerate(frames)]

    vx = [float(f.get("ego_vx_smoothed", f.get("ego_vx", 0.0))) for f in frames]
    vz = [float(f.get("ego_vz_smoothed", f.get("ego_vz", 0.0))) for f in frames]
    yaw = [float(f.get("ego_yaw_rate_smoothed", f.get("ego_yaw_rate", 0.0))) for f in frames]

    speed = [math.sqrt(vx_i * vx_i + vz_i * vz_i) for vx_i, vz_i in zip(vx, vz)]

    # Forward-axis segmentation (z): signed forward/backward motion plus stopping.
    forward_axis = _classify_forward_events(
        vz=vz,
        speed=speed,
        accel_threshold=accel_threshold,
        stop_enter_threshold=stop_enter_threshold,
        stop_exit_threshold=stop_exit_threshold,
        stop_total_speed_enter=stop_total_speed_enter,
        stop_total_speed_exit=stop_total_speed_exit,
        min_stop_duration=min_stop_duration,
        stop_window_size=stop_window_size,
        motion_window_size=motion_window_size,
        direction_epsilon=direction_epsilon,
    )
    forward_accel = forward_axis["forward_accel"]
    stop_raw = forward_axis["stop_raw"]
    forward_stop_mask = forward_axis["forward_stop_mask"]
    forward_speedup_mask = forward_axis["forward_speedup_mask"]
    forward_slowdown_mask = forward_axis["forward_slowdown_mask"]
    forward_static_moving_mask = forward_axis["forward_static_moving_mask"]
    backward_speedup_mask = forward_axis["backward_speedup_mask"]
    backward_slowdown_mask = forward_axis["backward_slowdown_mask"]
    backward_static_moving_mask = forward_axis["backward_static_moving_mask"]
    forward_event_raw = forward_axis["forward_event_raw"]

    # Lateral-axis segmentation (x): left / right / straightforward
    lateral_axis = _classify_lateral_events(
        vx=vx,
        lateral_turn_threshold=lateral_turn_threshold,
        lateral_straight_threshold=lateral_straight_threshold,
        min_turn_duration=min_turn_duration,
        motion_window_size=lateral_motion_window_size,
        direction_epsilon=lateral_direction_epsilon,
    )
    lateral_left_mask = lateral_axis["lateral_left_mask"]
    lateral_right_mask = lateral_axis["lateral_right_mask"]
    lateral_straight_mask = lateral_axis["lateral_straight_mask"]
    lateral_event_raw = lateral_axis["lateral_event_raw"]

    # Symbolic correction is applied to forward axis only.
    symbolic = _apply_symbolic_static_correction(
        primary_event=forward_event_raw,
        frame_indices=frame_indices,
        relative_video_result=relative_motion_video_result,
        cfg=cfg,
        stop_label="stopping",
    )

    combined_raw = _combine_axis_events(forward_event_raw, lateral_event_raw)
    combined_symbolic = _combine_axis_events(symbolic["events"], lateral_event_raw)

    short_merge = _merge_short_segments(
        events=combined_symbolic,
        frame_indices=frame_indices,
        min_segment_length=min_segment_length,
    )
    primary_event_corrected = short_merge["events"]

    # Merge cut points from forward + lateral axis segmentations.
    merged_axis = _build_merged_segments_from_cutpoints(
        frame_indices=frame_indices,
        forward_events=symbolic["events"],
        lateral_events=lateral_event_raw,
    )

    forward_segments_final = _segment_from_events(symbolic["events"], frame_indices)
    lateral_segments_final = _segment_from_events(lateral_event_raw, frame_indices)
    segments = _segment_from_events(primary_event_corrected, frame_indices)
    cut_points = merged_axis["cut_points"]

    stop_exit_ratio = stop_exit_threshold / max(1e-6, stop_enter_threshold)
    total_enter_ratio = stop_total_speed_enter / max(1e-6, stop_enter_threshold)
    total_exit_ratio = stop_total_speed_exit / max(1e-6, stop_exit_threshold)

    comparison_rows: List[Dict[str, Any]] = []
    forward_comparison_variants: List[Dict[str, Any]] = []
    lateral_comparison_variants: List[Dict[str, Any]] = []
    for compare_stop_threshold in compare_forward_stop_thresholds:
        compare_stop_exit_threshold = compare_stop_threshold * stop_exit_ratio
        compare_total_speed_enter = compare_stop_threshold * total_enter_ratio
        compare_total_speed_exit = compare_stop_exit_threshold * total_exit_ratio
        forward_cmp = _classify_forward_events(
            vz=vz,
            speed=speed,
            accel_threshold=accel_threshold,
            stop_enter_threshold=compare_stop_threshold,
            stop_exit_threshold=compare_stop_exit_threshold,
            stop_total_speed_enter=compare_total_speed_enter,
            stop_total_speed_exit=compare_total_speed_exit,
            min_stop_duration=min_stop_duration,
            stop_window_size=stop_window_size,
            motion_window_size=motion_window_size,
            direction_epsilon=direction_epsilon,
        )

        symbolic_cmp = _apply_symbolic_static_correction(
            primary_event=forward_cmp["forward_event_raw"],
            frame_indices=frame_indices,
            relative_video_result=relative_motion_video_result,
            cfg=cfg,
            stop_label="stopping",
        )

        segment_length_counts: List[Dict[str, int]] = []
        for compare_segment_length in compare_min_segment_lengths:
            forward_events_cmp = _merge_short_segments(
                events=symbolic_cmp["events"],
                frame_indices=frame_indices,
                min_segment_length=compare_segment_length,
            )["events"]
            forward_segments_cmp = _segment_from_events(forward_events_cmp, frame_indices)
            lateral_events_cmp = _merge_short_segments(
                events=lateral_event_raw,
                frame_indices=frame_indices,
                min_segment_length=compare_segment_length,
            )["events"]
            segment_length_counts.append(
                {
                    "min_segment_length": compare_segment_length,
                    "num_forward_segments": len(forward_segments_cmp),
                    "num_lateral_segments": len(_segment_from_events(lateral_events_cmp, frame_indices)),
                }
            )
            forward_comparison_variants.append(
                {
                    "title": (
                        f"forward | stop_th={compare_stop_threshold:.3f}, "
                        f"seg={compare_segment_length}"
                    ),
                    "signal_values": vz,
                    "event_series": forward_events_cmp,
                    "cut_points": [int(seg.get("start_frame", 0)) for seg in forward_segments_cmp],
                    "fallback_event": "forward_static_moving",
                    "current_label_prefix": (
                        f"stop_th={compare_stop_threshold:.3f}, seg={compare_segment_length}"
                    ),
                    "stop_threshold": compare_stop_threshold,
                    "min_stop_duration": min_stop_duration,
                    "min_segment_length": compare_segment_length,
                    "signal_label": "vz",
                }
            )

        comparison_rows.append(
            {
                "stop_threshold": compare_stop_threshold,
                "segment_length_counts": segment_length_counts,
            }
        )

    for compare_lateral_straight_threshold in compare_lateral_straight_thresholds:
        lateral_cmp = _classify_lateral_events(
            vx=vx,
            lateral_turn_threshold=lateral_turn_threshold,
            lateral_straight_threshold=compare_lateral_straight_threshold,
            min_turn_duration=min_turn_duration,
            motion_window_size=lateral_motion_window_size,
            direction_epsilon=lateral_direction_epsilon,
        )
        lateral_segments_cmp = _segment_from_events(lateral_cmp["lateral_event_raw"], frame_indices)
        lateral_comparison_variants.append(
            {
                "title": f"lateral | straight_th={compare_lateral_straight_threshold:.3f}",
                "signal_values": lateral_cmp["vx_trend"],
                "event_series": lateral_cmp["lateral_event_raw"],
                "cut_points": [int(seg.get("start_frame", 0)) for seg in lateral_segments_cmp],
                "fallback_event": "straightforward",
                "current_label_prefix": f"straight_th={compare_lateral_straight_threshold:.3f}",
                "signal_label": "vx",
                "lateral_straight_threshold": compare_lateral_straight_threshold,
                "lateral_motion_window_size": lateral_motion_window_size,
            }
        )

    result: Dict[str, Any] = {
        "video_id": video_id,
        "num_frames": len(frames),
        "config": {
            "stop_speed_threshold": stop_speed_threshold,
            "forward_stop_threshold": stop_speed_threshold,
            "forward_stop_enter_threshold": stop_enter_threshold,
            "forward_stop_exit_threshold": stop_exit_threshold,
            "stop_total_speed_enter_threshold": stop_total_speed_enter,
            "stop_total_speed_exit_threshold": stop_total_speed_exit,
            "accel_threshold": accel_threshold,
            "forward_accel_threshold": accel_threshold,
            "lateral_turn_threshold": lateral_turn_threshold,
            "stop_window_size": stop_window_size,
            "motion_window_size": motion_window_size,
            "forward_direction_epsilon": direction_epsilon,
            "lateral_motion_window_size": lateral_motion_window_size,
            "lateral_direction_epsilon": lateral_direction_epsilon,
            "lateral_straight_threshold": lateral_straight_threshold,
            "min_stop_duration": min_stop_duration,
            "min_turn_duration": min_turn_duration,
            "min_segment_length": min_segment_length,
        },
        "signals": {
            "speed": speed,
            "vz_trend": forward_axis["vz_trend"],
            "vx_trend": lateral_axis["vx_trend"],
            "forward_acceleration": forward_accel,
            "stop_abs_vz_mean": forward_axis["abs_vz_mean"],
            "stop_total_speed_mean": forward_axis["total_speed_mean"],
            "yaw_rate": yaw,
        },
            "masks": {
            "forward_stopping": forward_stop_mask,
            "forward_speedup": forward_speedup_mask,
            "forward_slowdown": forward_slowdown_mask,
            "forward_static_moving": forward_static_moving_mask,
            "backward_speedup": backward_speedup_mask,
            "backward_slowdown": backward_slowdown_mask,
            "backward_static_moving": backward_static_moving_mask,
            "lateral_left": lateral_left_mask,
            "lateral_right": lateral_right_mask,
            "lateral_straightforward": lateral_straight_mask,
            "lateral_turning_left": lateral_left_mask,
            "lateral_turning_right": lateral_right_mask,
        },
        "forward_event_raw": forward_event_raw,
        "forward_event": symbolic["events"],
        "lateral_event": lateral_event_raw,
        "primary_event_raw": combined_raw,
        "primary_event_symbolic": combined_symbolic,
        "primary_event": primary_event_corrected,
        "symbolic_correction": {
            "num_corrected": symbolic["num_corrected"],
            "corrected_frame_indices": symbolic["corrected_frame_indices"],
            "config": symbolic["config"],
        },
        "axis_merged_cut_points": {
            "cut_points": merged_axis["cut_points"],
            "segments": merged_axis["segments"],
        },
        "short_segment_merge": {
            "num_merged_segments": short_merge["num_merged_segments"],
            "min_segment_length": short_merge["min_segment_length"],
        },
        "forward_segments": forward_segments_final,
        "lateral_segments": lateral_segments_final,
        "num_forward_segments": len(forward_segments_final),
        "num_lateral_segments": len(lateral_segments_final),
        "segment_count_comparison": {
            "compare_forward_stop_thresholds": compare_forward_stop_thresholds,
            "compare_min_segment_lengths": compare_min_segment_lengths,
            "rows": comparison_rows,
        },
        "forward_comparison_variants": forward_comparison_variants,
        "lateral_comparison_variants": lateral_comparison_variants,
        "segments": segments,
        "cut_points": cut_points,
        "num_segments": len(segments),
        "visualization_version": _SEGMENTATION_VIS_VERSION,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    rendered = _render_temporal_segmentation_videos(
        video_id=video_id,
        ego_frames=frames,
        segmentation_result=result,
        output_path=out_dir / "temporal_segmentation_vis.mp4",
        fps=float(cfg.get("visualization_fps", 10.0)),
    )
    if rendered:
        result["visualization_paths"] = rendered
        result["visualization_path"] = rendered.get("primary", "")
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    print(f"  {video_id}")
    for row in comparison_rows:
        stop_threshold = float(row["stop_threshold"])
        for counts in row["segment_length_counts"]:
            print(
                f"    stop_threshold={stop_threshold:.3f} | "
                f"min_stop_duration={min_stop_duration} | "
                f"min_segment_length={int(counts['min_segment_length'])} | "
                f"vz={int(counts['num_forward_segments'])}"
            )
    return result


def run(
    ego_motion_results: List[Dict[str, Any]],
    relative_motion_results: Optional[List[Dict[str, Any]]] = None,
    seg_cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    out_root = output_root or get_output_root()

    relative_by_video: Dict[str, Dict[str, Any]] = {
        r.get("video_id", ""): r for r in (relative_motion_results or [])
    }

    results: List[Dict[str, Any]] = []
    for ego_video_result in ego_motion_results:
        video_id = ego_video_result.get("video_id", "unknown")
        result = process_video(
            ego_video_result=ego_video_result,
            relative_motion_video_result=relative_by_video.get(video_id),
            seg_cfg=seg_cfg,
            output_root=out_root,
            force_recompute=force_recompute,
        )
        results.append(result)

    manifest = {
        "num_videos": len(results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r.get("num_frames", 0),
                "num_segments": r.get("num_segments", 0),
                "num_cut_points": len(r.get("cut_points", [])),
            }
            for r in results
        ],
    }
    manifest_path = out_root / "temporal_segmentation_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Temporal segmentation manifest written to {manifest_path}")

    return results
