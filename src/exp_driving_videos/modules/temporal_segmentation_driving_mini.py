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

_SEGMENTATION_VIS_VERSION = 3


def _event_color_bgr(event_label: str) -> tuple:
    """Map combined event labels (forward|lateral) to stable timeline colors."""
    text = str(event_label or "").strip().lower()
    forward, lateral = text, ""
    if "|" in text:
        forward, lateral = text.split("|", 1)

    base = {
        "stopping": (80, 80, 255),
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
    if lateral == "turning_left":
        b = min(255, b + 35)
    elif lateral == "turning_right":
        g = min(255, g + 35)
    return (b, g, r)


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


def _render_temporal_segmentation_video(
    video_id: str,
    ego_frames: List[Dict[str, Any]],
    segmentation_result: Dict[str, Any],
    output_path: Path,
    fps: float = 10.0,
) -> Optional[str]:
    """Render side-by-side segmentation visualization as MP4."""
    try:
        import cv2
    except ModuleNotFoundError:
        return None

    if not ego_frames:
        return None

    first_img = None
    for frame in ego_frames:
        first_img = cv2.imread(frame.get("image_path", ""))
        if first_img is not None:
            break
    if first_img is None:
        return None

    h, w = first_img.shape[:2]
    timeline_h = 180
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
    cut_points_processed = set(int(v) for v in segmentation_result.get("cut_points", []))
    frame_indices = [int(f.get("frame_index", i)) for i, f in enumerate(ego_frames)]
    n = max(
        1,
        len(forward_events_raw),
        len(forward_events_processed),
        len(lateral_events),
        len(events_raw),
        len(events_processed),
    )

    raw_segments = _segment_from_events(events_raw, frame_indices) if events_raw else []
    cut_points_raw = set(int(seg.get("start_frame", 0)) for seg in raw_segments)
    forward_raw_segments = _segment_from_events(forward_events_raw, frame_indices) if forward_events_raw else []
    cut_points_forward_raw = set(int(seg.get("start_frame", 0)) for seg in forward_raw_segments)
    forward_processed_segments = _segment_from_events(forward_events_processed, frame_indices) if forward_events_processed else []
    cut_points_forward_processed = set(int(seg.get("start_frame", 0)) for seg in forward_processed_segments)
    lateral_segments = _segment_from_events(lateral_events, frame_indices) if lateral_events else []
    cut_points_lateral = set(int(seg.get("start_frame", 0)) for seg in lateral_segments)

    try:
        for i, frame in enumerate(ego_frames):
            img = cv2.imread(frame.get("image_path", ""))
            if img is None:
                img = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (w, h))

            forward_raw = forward_events_raw[i] if i < len(forward_events_raw) else "static_moving"
            forward_processed = (
                forward_events_processed[i] if i < len(forward_events_processed) else forward_raw
            )
            lateral_event = lateral_events[i] if i < len(lateral_events) else "straightforward"
            event_raw = events_raw[i] if i < len(events_raw) else f"{forward_raw}|{lateral_event}"
            event_processed = (
                events_processed[i] if i < len(events_processed) else f"{forward_processed}|{lateral_event}"
            )
            color_raw = _event_color_bgr(event_raw)
            color_processed = _event_color_bgr(event_processed)
            color_forward_raw = _event_color_bgr(f"{forward_raw}|straightforward")
            color_forward_processed = _event_color_bgr(f"{forward_processed}|straightforward")
            color_lateral = _event_color_bgr(f"static_moving|{lateral_event}")

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
            cv2.putText(
                img,
                f"raw: {event_raw}",
                (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color_raw,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"processed: {event_processed}",
                (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color_processed,
                2,
                cv2.LINE_AA,
            )

            # Timeline panel
            panel = np.zeros((timeline_h, w, 3), dtype=np.uint8)
            panel[:] = (20, 20, 20)

            raw_events = events_raw
            bar_specs = [
                ("raw timeline (gray ticks=cut points)", raw_events, cut_points_raw, 18, 34, lambda ev: _event_color_bgr(ev), (170, 170, 170)),
                ("processed timeline (white ticks=cut points)", events_processed, cut_points_processed, 50, 66, lambda ev: _event_color_bgr(ev), (255, 255, 255)),
                ("forward raw (gray ticks=cut points)", forward_events_raw, cut_points_forward_raw, 82, 98, lambda ev: _event_color_bgr(f"{ev}|straightforward"), (170, 170, 170)),
                ("forward processed (white ticks=cut points)", forward_events_processed, cut_points_forward_processed, 114, 130, lambda ev: _event_color_bgr(f"{ev}|straightforward"), (255, 255, 255)),
                ("lateral axis (amber ticks=cut points)", lateral_events, cut_points_lateral, 146, 162, lambda ev: _event_color_bgr(f"static_moving|{ev}"), (255, 210, 120)),
            ]

            for label, event_series, cut_points, y0, y1, color_fn, tick_color in bar_specs:
                cv2.putText(
                    panel,
                    label,
                    (10, max(14, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                for j in range(n):
                    x0 = int(round(j * w / n))
                    x1 = int(round((j + 1) * w / n))
                    if event_series is raw_events:
                        fallback = "constant_speed"
                    elif event_series is lateral_events:
                        fallback = "straightforward"
                    else:
                        fallback = "static_moving"
                    ev = event_series[j] if j < len(event_series) else fallback
                    cv2.rectangle(panel, (x0, y0), (max(x0 + 1, x1), y1), color_fn(ev), thickness=-1)
                for j, fidx in enumerate(frame_indices):
                    if fidx in cut_points:
                        x = int(round(j * w / n))
                        cv2.line(panel, (x, y0 - 4), (x, y1 + 4), tick_color, 1)

            out_frame = np.vstack([img, panel])
            writer.write(out_frame)
    finally:
        writer.release()

    return str(output_path)

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
    min_stop_duration = int(cfg.get("min_stop_duration", 3))
    min_turn_duration = int(cfg.get("min_turn_duration", 3))
    min_segment_length = int(cfg.get("min_segment_length", 1))

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
            vis_path = out_dir / "temporal_segmentation_vis.mp4"
            if vis_path.exists():
                return cached

            rendered_cached = _render_temporal_segmentation_video(
                video_id=video_id,
                ego_frames=ego_video_result.get("frames", []),
                segmentation_result=cached,
                output_path=vis_path,
                fps=float(cfg.get("visualization_fps", 10.0)),
            )
            if rendered_cached:
                cached["visualization_path"] = rendered_cached
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

    # Forward-axis segmentation (z): stopping / static_moving / speedup / slowdown
    forward_accel: List[float] = [0.0] * len(vz)
    for i in range(1, len(vz)):
        forward_accel[i] = vz[i] - vz[i - 1]

    stop_raw: List[bool] = [False] * len(frames)
    is_stopped = False
    for i in range(len(frames)):
        abs_vz = abs(vz[i])
        total_speed = speed[i]
        enter_stop = (abs_vz <= stop_enter_threshold) and (total_speed <= stop_total_speed_enter)
        stay_stop = (abs_vz <= stop_exit_threshold) and (total_speed <= stop_total_speed_exit)
        if is_stopped:
            is_stopped = stay_stop
        else:
            is_stopped = enter_stop
        stop_raw[i] = is_stopped
    stop_runs = _bool_runs(stop_raw, min_len=min_stop_duration)
    forward_stop_mask = _mask_from_runs(len(frames), stop_runs)

    forward_speedup_mask = [
        (forward_accel[i] > accel_threshold) and (not forward_stop_mask[i])
        for i in range(len(frames))
    ]
    forward_slowdown_mask = [
        (forward_accel[i] < -accel_threshold) and (not forward_stop_mask[i])
        for i in range(len(frames))
    ]
    forward_static_moving_mask = [
        (not forward_stop_mask[i])
        and (not forward_speedup_mask[i])
        and (not forward_slowdown_mask[i])
        for i in range(len(frames))
    ]

    forward_event_raw: List[str] = []
    for i in range(len(frames)):
        if forward_stop_mask[i]:
            forward_event_raw.append("stopping")
        elif forward_speedup_mask[i]:
            forward_event_raw.append("speedup")
        elif forward_slowdown_mask[i]:
            forward_event_raw.append("slowdown")
        else:
            forward_event_raw.append("static_moving")

    # Lateral-axis segmentation (x): turning_left / turning_right / straightforward
    lateral_left_raw = [vx_i < -lateral_turn_threshold for vx_i in vx]
    lateral_right_raw = [vx_i > lateral_turn_threshold for vx_i in vx]

    # Optional run filtering for turns to suppress flicker.
    left_runs = _bool_runs(lateral_left_raw, min_len=min_turn_duration)
    right_runs = _bool_runs(lateral_right_raw, min_len=min_turn_duration)
    lateral_left_mask = _mask_from_runs(len(frames), left_runs)
    lateral_right_mask = _mask_from_runs(len(frames), right_runs)
    lateral_straight_mask = [
        (not lateral_left_mask[i]) and (not lateral_right_mask[i])
        for i in range(len(frames))
    ]

    lateral_event_raw: List[str] = []
    for i in range(len(frames)):
        if lateral_left_mask[i]:
            lateral_event_raw.append("turning_left")
        elif lateral_right_mask[i]:
            lateral_event_raw.append("turning_right")
        else:
            lateral_event_raw.append("straightforward")

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

    segments = _segment_from_events(primary_event_corrected, frame_indices)
    cut_points = merged_axis["cut_points"]

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
            "min_stop_duration": min_stop_duration,
            "min_turn_duration": min_turn_duration,
            "min_segment_length": min_segment_length,
        },
        "signals": {
            "speed": speed,
            "forward_acceleration": forward_accel,
            "yaw_rate": yaw,
        },
        "masks": {
            "forward_stopping": forward_stop_mask,
            "forward_speedup": forward_speedup_mask,
            "forward_slowdown": forward_slowdown_mask,
            "forward_static_moving": forward_static_moving_mask,
            "lateral_turning_left": lateral_left_mask,
            "lateral_turning_right": lateral_right_mask,
            "lateral_straightforward": lateral_straight_mask,
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
        "segments": segments,
        "cut_points": cut_points,
        "num_segments": len(segments),
        "visualization_version": _SEGMENTATION_VIS_VERSION,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    vis_path = out_dir / "temporal_segmentation_vis.mp4"
    rendered = _render_temporal_segmentation_video(
        video_id=video_id,
        ego_frames=frames,
        segmentation_result=result,
        output_path=vis_path,
        fps=float(cfg.get("visualization_fps", 10.0)),
    )
    if rendered:
        result["visualization_path"] = rendered
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"    visualization saved -> {vis_path.name}")

    print(
        f"  {video_id}: {len(segments)} segments, "
        f"{len(cut_points)} cut points"
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
        print(f"  Processing temporal segmentation: {video_id}")
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
