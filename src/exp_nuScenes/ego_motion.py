"""
Ego motion helpers for nuScenes scenes.

nuScenes stores ego poses per sample_data record. These helpers convert an
ordered scene's poses into frame-aligned 3D velocity signals in meters/second.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import numpy as np


DEFAULT_CHANNEL_PRIORITY = (
    "LIDAR_TOP",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

FORWARD_CHART_CLASS_COLORS = {
    0: (80, 190, 90),    # speed_up
    1: (70, 150, 245),   # slow_down
    2: (225, 225, 225),  # stable
    3: (185, 115, 215),  # stop
}

LATERAL_CHART_CLASS_COLORS = {
    0: (235, 130, 70),   # left
    1: (85, 120, 235),   # right
    2: (225, 225, 225),  # stable
}


def quaternion_to_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Return a 3x3 rotation matrix from a nuScenes [w, x, y, z] quaternion."""
    w, x, y, z = [float(value) for value in quaternion]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _select_media_record(
    sample: Dict[str, Any],
    channel: Optional[str] = None,
    channel_priority: Sequence[str] = DEFAULT_CHANNEL_PRIORITY,
) -> Dict[str, Any]:
    media = sample.get("media") or {}
    if not media:
        raise ValueError(f"Sample {sample.get('sample_token', '<unknown>')} has no media records.")

    if channel is not None:
        channel = channel.upper()
        if channel not in media:
            raise ValueError(
                f"Sample {sample.get('sample_token', '<unknown>')} has no media channel {channel!r}. "
                f"Available channels: {sorted(media)}"
            )
        return media[channel]

    for preferred_channel in channel_priority:
        if preferred_channel in media:
            return media[preferred_channel]
    return media[sorted(media)[0]]


def _sample_pose(
    sample: Dict[str, Any],
    channel: Optional[str] = None,
    channel_priority: Sequence[str] = DEFAULT_CHANNEL_PRIORITY,
) -> Dict[str, Any]:
    media_record = _select_media_record(sample, channel=channel, channel_priority=channel_priority)
    ego_pose = media_record.get("ego_pose")
    if not ego_pose:
        raise ValueError(
            f"Media record {media_record.get('sample_data_token', '<unknown>')} "
            f"for sample {sample.get('sample_token', '<unknown>')} has no ego_pose."
        )
    return {
        "sample_token": sample["sample_token"],
        "scene_name": sample.get("scene_name"),
        "channel": media_record["channel"],
        "timestamp": int(media_record.get("timestamp", sample["timestamp"])),
        "translation": np.asarray(ego_pose["translation"], dtype=np.float64),
        "rotation": np.asarray(ego_pose["rotation"], dtype=np.float64),
    }


def extract_ego_speed_3d(
    scene_samples: Sequence[Dict[str, Any]],
    channel: Optional[str] = None,
    channel_priority: Sequence[str] = DEFAULT_CHANNEL_PRIORITY,
) -> Dict[str, Any]:
    """
    Extract frame-aligned ego 3D speed from an ordered nuScenes scene.

    Args:
        scene_samples: Samples from one scene, ordered or unordered.
        channel: Optional sample_data channel to use for ego poses. If omitted,
            LIDAR_TOP is preferred, then CAM_FRONT, then the remaining cameras.
        channel_priority: Fallback channel order when channel is omitted.

    Returns:
        Dict containing sample tokens, timestamps, positions, global velocity,
        ego-local velocity, scalar speeds, and delta time arrays/lists.

    Notes:
        - Positions are global nuScenes coordinates in meters.
        - Timestamps are microseconds in the raw data.
        - global_velocity_mps[i] is the backward-difference velocity from
          frame i-1 to i. The first frame uses the first forward difference.
        - ego_velocity_mps rotates each global velocity into that frame's
          ego-local coordinate system. When at least two frames exist, frame 0
          ego velocity is copied from frame 1 so the first displayed vx/vy/vz
          values match the first measurable speed.
    """
    if not scene_samples:
        return {
            "sample_tokens": [],
            "timestamps_us": [],
            "channels": [],
            "positions_m": [],
            "global_velocity_mps": [],
            "ego_velocity_mps": [],
            "speed_mps": [],
            "planar_speed_mps": [],
            "delta_t_s": [],
        }

    ordered_samples = sorted(scene_samples, key=lambda sample: sample["timestamp"])
    poses = [
        _sample_pose(sample, channel=channel, channel_priority=channel_priority)
        for sample in ordered_samples
    ]

    timestamps_us = np.asarray([pose["timestamp"] for pose in poses], dtype=np.float64)
    positions = np.vstack([pose["translation"] for pose in poses])
    rotations = [pose["rotation"] for pose in poses]

    delta_t_s = np.zeros(len(poses), dtype=np.float64)
    global_velocity = np.zeros_like(positions, dtype=np.float64)

    if len(poses) > 1:
        raw_delta_t = np.diff(timestamps_us) / 1_000_000.0
        delta_positions = np.diff(positions, axis=0)
        valid_delta_t = np.where(raw_delta_t > 0, raw_delta_t, np.nan)
        global_velocity[1:] = delta_positions / valid_delta_t[:, None]
        global_velocity[~np.isfinite(global_velocity)] = 0.0
        global_velocity[0] = global_velocity[1]
        delta_t_s[1:] = np.where(raw_delta_t > 0, raw_delta_t, 0.0)
        delta_t_s[0] = delta_t_s[1]

    ego_velocity = np.zeros_like(global_velocity, dtype=np.float64)
    for index, rotation in enumerate(rotations):
        rotation_matrix = quaternion_to_rotation_matrix(rotation)
        ego_velocity[index] = rotation_matrix.T @ global_velocity[index]
    if len(poses) > 1:
        ego_velocity[0] = ego_velocity[1]

    speed = np.linalg.norm(global_velocity, axis=1)
    planar_speed = np.linalg.norm(global_velocity[:, :2], axis=1)

    # nuScenes ego/local coordinates are x-forward, y-left, z-up.
    forward_speed = ego_velocity[:, 0]
    lateral_speed = ego_velocity[:, 1]
    vertical_speed = ego_velocity[:, 2]

    return {
        "sample_tokens": [pose["sample_token"] for pose in poses],
        "scene_name": poses[0]["scene_name"],
        "timestamps_us": timestamps_us.astype(np.int64).tolist(),
        "channels": [pose["channel"] for pose in poses],
        "positions_m": positions.tolist(),
        "global_velocity_mps": global_velocity.tolist(),
        "ego_velocity_mps": ego_velocity.tolist(),
        "forward_speed_mps": forward_speed.tolist(),
        "lateral_speed_mps": lateral_speed.tolist(),
        "vertical_speed_mps": vertical_speed.tolist(),
        "speed_mps": speed.tolist(),
        "planar_speed_mps": planar_speed.tolist(),
        "delta_t_s": delta_t_s.tolist(),
    }


def summarize_ego_speed(ego_motion: Dict[str, Any]) -> Dict[str, float]:
    """Return compact scalar stats for an ego speed signal."""
    speeds = np.asarray(ego_motion.get("speed_mps", []), dtype=np.float64)
    planar_speeds = np.asarray(ego_motion.get("planar_speed_mps", []), dtype=np.float64)
    forward_speeds = np.asarray(ego_motion.get("forward_speed_mps", []), dtype=np.float64)
    lateral_speeds = np.asarray(ego_motion.get("lateral_speed_mps", []), dtype=np.float64)
    vertical_speeds = np.asarray(ego_motion.get("vertical_speed_mps", []), dtype=np.float64)
    if speeds.size == 0:
        return {
            "mean_speed_mps": 0.0,
            "max_speed_mps": 0.0,
            "mean_planar_speed_mps": 0.0,
            "max_planar_speed_mps": 0.0,
            "mean_forward_speed_mps": 0.0,
            "mean_lateral_speed_mps": 0.0,
            "mean_vertical_speed_mps": 0.0,
        }
    return {
        "mean_speed_mps": float(np.mean(speeds)),
        "max_speed_mps": float(np.max(speeds)),
        "mean_planar_speed_mps": float(np.mean(planar_speeds)),
        "max_planar_speed_mps": float(np.max(planar_speeds)),
        "mean_forward_speed_mps": float(np.mean(forward_speeds)),
        "mean_lateral_speed_mps": float(np.mean(lateral_speeds)),
        "mean_vertical_speed_mps": float(np.mean(vertical_speeds)),
    }


def _draw_signal_chart(
    values: np.ndarray,
    current_index: int,
    title: str,
    color: tuple[int, int, int],
    width: int,
    height: int,
    class_mask: Optional[Sequence[int]] = None,
    class_label_map: Optional[Mapping[int, str]] = None,
    class_colors: Optional[Mapping[int, tuple[int, int, int]]] = None,
) -> np.ndarray:
    chart = np.full((height, width, 3), 245, dtype=np.uint8)
    pad_left = 82
    pad_right = 26
    pad_top = 58
    pad_bottom = 34
    plot_x0 = pad_left
    plot_y0 = pad_top
    plot_x1 = width - pad_right
    plot_y1 = height - pad_bottom

    if values.size == 0:
        cv2.rectangle(chart, (plot_x0, plot_y0), (plot_x1, plot_y1), (255, 255, 255), -1)
        cv2.rectangle(chart, (plot_x0, plot_y0), (plot_x1, plot_y1), (190, 190, 190), 1)
        return chart

    cv2.rectangle(chart, (plot_x0, plot_y0), (plot_x1, plot_y1), (255, 255, 255), -1)
    x_positions = np.linspace(plot_x0, plot_x1, max(1, values.size))

    current_label_name = None
    if class_mask is not None and class_colors is not None:
        mask = np.asarray(class_mask, dtype=np.int64).reshape(-1)
        if mask.size < values.size:
            pad_value = mask[-1] if mask.size else 0
            mask = np.concatenate([mask, np.repeat(pad_value, values.size - mask.size)])
        mask = mask[: values.size]

        overlay = chart.copy()
        run_start = 0
        for index in range(1, mask.size + 1):
            if index < mask.size and int(mask[index]) == int(mask[run_start]):
                continue

            run_end = index - 1
            label = int(mask[run_start])
            bg_color = class_colors.get(label)
            if bg_color is not None:
                if values.size == 1:
                    left_x, right_x = plot_x0, plot_x1
                else:
                    left_x = plot_x0 if run_start == 0 else int((x_positions[run_start - 1] + x_positions[run_start]) / 2)
                    right_x = plot_x1 if run_end == values.size - 1 else int((x_positions[run_end] + x_positions[run_end + 1]) / 2)
                cv2.rectangle(overlay, (int(left_x), plot_y0), (int(right_x), plot_y1), bg_color, -1)

            if index < mask.size:
                run_start = index

        cv2.addWeighted(overlay, 0.28, chart, 0.72, 0, chart)
        current_index_for_label = max(0, min(current_index, mask.size - 1))
        current_label = int(mask[current_index_for_label])
        if class_label_map is not None:
            current_label_name = class_label_map.get(current_label)

    cv2.rectangle(chart, (plot_x0, plot_y0), (plot_x1, plot_y1), (190, 190, 190), 1)

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        y_min, y_max = -1.0, 1.0
    else:
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        if abs(y_max - y_min) < 1e-6:
            margin = max(1.0, abs(y_max) * 0.2)
            y_min -= margin
            y_max += margin
        else:
            margin = (y_max - y_min) * 0.14
            y_min -= margin
            y_max += margin

    zero_y = None
    if y_min <= 0 <= y_max:
        zero_y = int(plot_y1 - (0 - y_min) / (y_max - y_min) * (plot_y1 - plot_y0))
        cv2.line(chart, (plot_x0, zero_y), (plot_x1, zero_y), (215, 215, 215), 1)

    points = []
    for x, value in zip(x_positions, values):
        y = int(plot_y1 - (float(value) - y_min) / (y_max - y_min) * (plot_y1 - plot_y0))
        points.append((int(x), y))

    if len(points) > 1:
        cv2.polylines(chart, [np.asarray(points, dtype=np.int32)], False, color, 3, cv2.LINE_AA)

    current_index = max(0, min(current_index, len(points) - 1))
    current_point = points[current_index]
    cv2.circle(chart, current_point, 11, color, -1, cv2.LINE_AA)
    cv2.circle(chart, current_point, 15, (255, 255, 255), 3, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(chart, title, (16, 25), font, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
    if current_label_name:
        label_text = f"class: {current_label_name}"
        cv2.putText(chart, label_text, (plot_x0, 51), font, 0.62, (55, 55, 55), 2, cv2.LINE_AA)
    current_value = values[current_index]
    cv2.putText(
        chart,
        f"{current_value: .2f} m/s",
        (plot_x1 - 150, 25),
        font,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(chart, f"{y_max:.1f}", (8, plot_y0 + 8), font, 0.48, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(chart, f"{y_min:.1f}", (8, plot_y1), font, 0.48, (80, 80, 80), 1, cv2.LINE_AA)
    if zero_y is not None:
        cv2.putText(chart, "0", (48, zero_y + 5), font, 0.45, (130, 130, 130), 1, cv2.LINE_AA)

    return chart


def _find_segment_for_frame(
    frame_index: int,
    segments_payload: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not segments_payload:
        return None
    for segment in segments_payload.get("segments", []):
        start_frame = int(segment.get("start_frame", -1))
        end_frame = int(segment.get("end_frame", -1))
        if start_frame <= frame_index <= end_frame:
            return segment
    return None


def _short_category_name(category_name: str) -> str:
    parts = str(category_name).split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return str(category_name)


def _predicate_text_color(label: str) -> tuple[int, int, int]:
    color_map = {
        "approaching": (40, 90, 235),
        "moving_away": (70, 170, 70),
        "stable": (210, 210, 210),
        "left": (235, 130, 70),
        "right": (85, 120, 235),
        "moving": (80, 200, 90),
        "stopped": (185, 115, 215),
        "turning": (90, 200, 255),
        "not_turning": (210, 210, 210),
        "unknown": (155, 155, 155),
        "uncertain": (155, 155, 155),
    }
    return color_map.get(str(label), (225, 225, 225))


def _object_display_color(predicate_obj: Optional[Dict[str, Any]]) -> tuple[int, int, int]:
    if not predicate_obj:
        return (225, 225, 225)
    for key in ("relative_motion", "lateral_motion", "is_moving", "is_turning"):
        label = predicate_obj.get(key, {}).get("label")
        if label and label not in {"unknown", "uncertain"}:
            return _predicate_text_color(label)
    return (225, 225, 225)


def _lookup_object_value_for_frame(
    obj: Dict[str, Any],
    frame_index: int,
    key: str,
) -> Any:
    frame_indices = [int(value) for value in obj.get("present_frame_indices", [])]
    if frame_index not in frame_indices:
        return None
    value_index = frame_indices.index(frame_index)
    values = obj.get(key, [])
    if value_index >= len(values):
        return None
    return values[value_index]


def _annotation_box_corners_global(annotation: Dict[str, Any]) -> Optional[np.ndarray]:
    size = annotation.get("size", [])
    translation = annotation.get("translation", [])
    rotation = annotation.get("rotation_wxyz", [])
    if len(size) != 3 or len(translation) != 3 or len(rotation) != 4:
        return None

    width, length, height = [float(value) for value in size]
    x_corners = np.array([length, length, length, length, -length, -length, -length, -length], dtype=np.float64) / 2.0
    y_corners = np.array([width, -width, -width, width, width, -width, -width, width], dtype=np.float64) / 2.0
    z_corners = np.array([height, height, -height, -height, height, height, -height, -height], dtype=np.float64) / 2.0
    corners_local = np.stack([x_corners, y_corners, z_corners], axis=1)

    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    translation_vec = np.asarray(translation, dtype=np.float64)
    corners_global = (rotation_matrix @ corners_local.T).T + translation_vec
    return corners_global


def _project_global_corners_to_image(
    corners_global: np.ndarray,
    media_record: Dict[str, Any],
) -> Optional[np.ndarray]:
    calibrated_sensor = media_record.get("calibrated_sensor", {})
    ego_pose = media_record.get("ego_pose", {})
    camera_intrinsic = np.asarray(calibrated_sensor.get("camera_intrinsic", []), dtype=np.float64)
    if camera_intrinsic.shape != (3, 3):
        return None

    ego_translation = np.asarray(ego_pose.get("translation", []), dtype=np.float64)
    ego_rotation = np.asarray(ego_pose.get("rotation", []), dtype=np.float64)
    sensor_translation = np.asarray(calibrated_sensor.get("translation", []), dtype=np.float64)
    sensor_rotation = np.asarray(calibrated_sensor.get("rotation", []), dtype=np.float64)
    if ego_translation.shape != (3,) or ego_rotation.shape != (4,) or sensor_translation.shape != (3,) or sensor_rotation.shape != (4,):
        return None

    ego_rotation_matrix = quaternion_to_rotation_matrix(ego_rotation)
    sensor_rotation_matrix = quaternion_to_rotation_matrix(sensor_rotation)

    corners_ego = (ego_rotation_matrix.T @ (corners_global - ego_translation).T).T
    corners_sensor = (sensor_rotation_matrix.T @ (corners_ego - sensor_translation).T).T
    valid = corners_sensor[:, 2] > 0.1
    if not np.any(valid):
        return None

    projected = (camera_intrinsic @ corners_sensor[valid].T).T
    projected_xy = projected[:, :2] / projected[:, 2:3]
    return projected_xy


def _project_annotation_bbox(
    annotation: Dict[str, Any],
    media_record: Dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> Optional[tuple[int, int, int, int]]:
    corners_global = _annotation_box_corners_global(annotation)
    if corners_global is None:
        return None
    projected_xy = _project_global_corners_to_image(corners_global, media_record)
    if projected_xy is None or projected_xy.size == 0:
        return None

    x_min = int(np.floor(np.min(projected_xy[:, 0])))
    y_min = int(np.floor(np.min(projected_xy[:, 1])))
    x_max = int(np.ceil(np.max(projected_xy[:, 0])))
    y_max = int(np.ceil(np.max(projected_xy[:, 1])))

    x_min = max(0, min(frame_width - 1, x_min))
    y_min = max(0, min(frame_height - 1, y_min))
    x_max = max(0, min(frame_width - 1, x_max))
    y_max = max(0, min(frame_height - 1, y_max))
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def _draw_projected_object_bboxes(
    frame: np.ndarray,
    sample: Dict[str, Any],
    channel: str,
    segment_objects: Optional[Dict[str, Any]],
    segment_predicates: Optional[Dict[str, Any]],
    frame_index: int,
) -> np.ndarray:
    panel_segment = _find_segment_for_frame(frame_index, segment_objects)
    if panel_segment is None:
        return frame
    predicate_segment = _find_segment_for_frame(frame_index, segment_predicates)
    predicate_by_id = {}
    if predicate_segment is not None:
        predicate_by_id = {
            obj["id"]: obj
            for obj in predicate_segment.get("objects", [])
        }

    media = sample.get("media", {})
    if channel not in media:
        return frame
    media_record = media[channel]
    frame_h, frame_w = frame.shape[:2]

    active_object_by_id = {}
    for obj in panel_segment.get("objects", []):
        present_frame_indices = [int(value) for value in obj.get("present_frame_indices", [])]
        if frame_index in present_frame_indices:
            active_object_by_id[obj["id"]] = obj

    for annotation in sample.get("annotations", []):
        obj_id = annotation.get("instance_token")
        if obj_id not in active_object_by_id:
            continue

        bbox = _project_annotation_bbox(annotation, media_record, frame_w, frame_h)
        if bbox is None:
            continue

        predicate_obj = predicate_by_id.get(obj_id)
        bbox_color = _object_display_color(predicate_obj)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2, cv2.LINE_AA)

        obj_meta = active_object_by_id[obj_id]
        rank = int(obj_meta.get("nearest_distance_rank", 0))
        category = _short_category_name(annotation.get("category_name", obj_meta.get("category_name", "object")))
        rel_label = predicate_obj.get("relative_motion", {}).get("label", "unknown") if predicate_obj else "unknown"
        bbox_label = f"#{rank} {category} | {rel_label}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(bbox_label, font, font_scale, thickness)
        label_x = max(0, min(frame_w - text_w - 8, x1))
        label_y = max(text_h + 8, y1 - 8)
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (label_x - 4, label_y - text_h - 4),
            (label_x + text_w + 4, label_y + baseline + 4),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)
        cv2.putText(frame, bbox_label, (label_x, label_y), font, font_scale, bbox_color, thickness, cv2.LINE_AA)

    return frame


def _draw_object_motion_panel(
    frame: np.ndarray,
    frame_index: int,
    segment_objects: Optional[Dict[str, Any]],
    segment_predicates: Optional[Dict[str, Any]],
    max_objects: int = 5,
) -> np.ndarray:
    panel_segment = _find_segment_for_frame(frame_index, segment_objects)
    if panel_segment is None:
        return frame

    predicate_segment = _find_segment_for_frame(frame_index, segment_predicates)
    predicate_by_id = {}
    if predicate_segment is not None:
        predicate_by_id = {
            obj["id"]: obj
            for obj in predicate_segment.get("objects", [])
        }

    active_objects = []
    for obj in panel_segment.get("objects", []):
        if frame_index not in [int(value) for value in obj.get("present_frame_indices", [])]:
            continue
        active_objects.append(obj)
    if not active_objects:
        return frame

    active_objects.sort(
        key=lambda obj: (
            int(obj.get("nearest_distance_rank", 10**9)),
            obj.get("id", ""),
        )
    )
    active_objects = active_objects[:max_objects]

    frame_h, frame_w = frame.shape[:2]
    panel_w = min(520, max(360, int(frame_w * 0.34)))
    panel_x1 = frame_w - 18
    panel_x0 = max(18, panel_x1 - panel_w)
    panel_y0 = 72
    panel_y1 = min(frame_h - 18, panel_y0 + 54 + 74 * len(active_objects) + 34)

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x0, panel_y0), (panel_x1, panel_y1), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (panel_x0, panel_y0), (panel_x1, panel_y1), (75, 75, 75), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    segment_title = (
        f"segment {int(panel_segment.get('segment_index', 0)) + 1} | "
        f"{panel_segment.get('combined_label_name', 'unknown')}"
    )
    cv2.putText(frame, "object motion", (panel_x0 + 14, panel_y0 + 24), font, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, segment_title, (panel_x0 + 14, panel_y0 + 48), font, 0.56, (190, 190, 190), 1, cv2.LINE_AA)

    ego_predicates = predicate_segment.get("ego", {}) if predicate_segment else {}
    ego_move = ego_predicates.get("ego_is_moving", {}).get("label", "unknown")
    ego_turn = ego_predicates.get("ego_is_turning", {}).get("label", "unknown")
    ego_line = f"ego: {ego_move} | {ego_turn}"
    cv2.putText(frame, ego_line, (panel_x0 + 14, panel_y0 + 70), font, 0.54, (110, 210, 255), 1, cv2.LINE_AA)

    row_y = panel_y0 + 102
    for obj in active_objects:
        obj_id = obj.get("id", "")
        pred = predicate_by_id.get(obj_id, {})
        category = _short_category_name(obj.get("category_name", "object"))
        rank = int(obj.get("nearest_distance_rank", 0))
        current_distance = _lookup_object_value_for_frame(obj, frame_index, "relative_distance_to_ego_m")
        current_speed = _lookup_object_value_for_frame(obj, frame_index, "ego_relative_speed_mps")
        if current_distance is None:
            current_distance = obj.get("mean_relative_distance_to_ego_m")
        if current_speed is None:
            current_speed = obj.get("mean_ego_relative_speed_mps")

        display_color = _object_display_color(pred)
        title_line = (
            f"#{rank} {category}  "
            f"d={float(current_distance):.1f}m  "
            f"v={float(current_speed):.1f}m/s"
            if current_distance is not None and current_speed is not None
            else f"#{rank} {category}"
        )
        cv2.putText(frame, title_line, (panel_x0 + 14, row_y), font, 0.54, display_color, 1, cv2.LINE_AA)

        rel_label = pred.get("relative_motion", {}).get("label", "unknown")
        lat_label = pred.get("lateral_motion", {}).get("label", "unknown")
        move_label = pred.get("is_moving", {}).get("label", "unknown")
        turn_label = pred.get("is_turning", {}).get("label", "unknown")

        labels = [
            (rel_label, _predicate_text_color(rel_label)),
            (lat_label, _predicate_text_color(lat_label)),
            (move_label, _predicate_text_color(move_label)),
            (turn_label, _predicate_text_color(turn_label)),
        ]
        text_x = panel_x0 + 14
        text_y = row_y + 24
        for index, (label, label_color) in enumerate(labels):
            if index > 0:
                sep = " | "
                cv2.putText(frame, sep, (text_x, text_y), font, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
                sep_size = cv2.getTextSize(sep, font, 0.5, 1)[0]
                text_x += sep_size[0]
            cv2.putText(frame, label, (text_x, text_y), font, 0.5, label_color, 1, cv2.LINE_AA)
            label_size = cv2.getTextSize(label, font, 0.5, 1)[0]
            text_x += label_size[0]

        row_y += 74
        if row_y + 20 > panel_y1:
            break

    return frame


def render_ego_motion_video(
    scene_samples: Sequence[Dict[str, Any]],
    ego_motion: Dict[str, Any],
    output_path: str | "Path",
    channel: str = "CAM_FRONT",
    fps: float = 2.0,
    chart_height: int = 190,
    ego_motion_segments: Optional[Dict[str, Any]] = None,
    segment_objects: Optional[Dict[str, Any]] = None,
    segment_predicates: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Render camera frames plus ego vx/vy/vz signal charts to an MP4.

    The top panel is the selected camera frame. The lower three panels show
    ego-local x/y/z velocity signals, each with a circle marking the current
    frame's value. If ego_motion_segments is supplied, the x and y chart
    backgrounds are shaded by their per-frame class masks. If segment object
    data is supplied, the camera frame also shows a compact object-motion panel.
    """
    from pathlib import Path

    if not scene_samples:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_samples = sorted(scene_samples, key=lambda sample: sample["timestamp"])
    velocities = np.asarray(ego_motion.get("ego_velocity_mps", []), dtype=np.float64)
    if velocities.ndim != 2 or velocities.shape[1] != 3:
        raise ValueError("ego_motion must contain ego_velocity_mps as an Nx3 sequence.")

    if velocities.shape[0] < len(ordered_samples):
        pad_count = len(ordered_samples) - velocities.shape[0]
        pad_value = velocities[-1:] if velocities.size else np.zeros((1, 3), dtype=np.float64)
        velocities = np.vstack([velocities, np.repeat(pad_value, pad_count, axis=0)])
    velocities = velocities[: len(ordered_samples)]

    first_frame = None
    for sample in ordered_samples:
        media = sample.get("media", {})
        if channel not in media:
            continue
        first_frame = cv2.imread(media[channel]["path"])
        if first_frame is not None:
            break
    if first_frame is None:
        raise FileNotFoundError(f"No readable frames found for channel {channel}.")

    frame_h, frame_w = first_frame.shape[:2]
    total_h = frame_h + chart_height * 3
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, total_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer: {output_path}")

    colors = {
        "x": (40, 90, 235),
        "y": (45, 170, 75),
        "z": (220, 120, 40),
    }
    forward_mask = None
    forward_label_map = None
    lateral_mask = None
    lateral_label_map = None
    if ego_motion_segments:
        forward_mask = ego_motion_segments.get("forward_mask")
        lateral_mask = ego_motion_segments.get("lateral_mask")
        label_maps = ego_motion_segments.get("label_maps", {})
        forward_label_map = {
            int(label): name
            for label, name in label_maps.get("forward", {}).items()
        }
        lateral_label_map = {
            int(label): name
            for label, name in label_maps.get("lateral", {}).items()
        }

    for frame_index, sample in enumerate(ordered_samples):
        media = sample.get("media", {})
        if channel not in media:
            frame = np.full((frame_h, frame_w, 3), 40, dtype=np.uint8)
        else:
            frame = cv2.imread(media[channel]["path"])
            if frame is None:
                frame = np.full((frame_h, frame_w, 3), 40, dtype=np.uint8)
            elif frame.shape[:2] != (frame_h, frame_w):
                frame = cv2.resize(frame, (frame_w, frame_h))

        header = (
            f"{sample.get('scene_name', '')} | {channel} | "
            f"frame {frame_index + 1}/{len(ordered_samples)} | "
            f"ego speed {np.linalg.norm(velocities[frame_index]):.2f} m/s"
        )
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, 58), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, header, (18, 39), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        frame = _draw_projected_object_bboxes(
            frame,
            sample,
            channel=channel,
            segment_objects=segment_objects,
            segment_predicates=segment_predicates,
            frame_index=frame_index,
        )
        frame = _draw_object_motion_panel(
            frame,
            frame_index,
            segment_objects=segment_objects,
            segment_predicates=segment_predicates,
        )

        x_chart = _draw_signal_chart(
            velocities[:, 0],
            frame_index,
            "ego velocity x (forward)",
            colors["x"],
            frame_w,
            chart_height,
            class_mask=forward_mask,
            class_label_map=forward_label_map,
            class_colors=FORWARD_CHART_CLASS_COLORS,
        )
        y_chart = _draw_signal_chart(
            velocities[:, 1],
            frame_index,
            "ego velocity y (left / lateral)",
            colors["y"],
            frame_w,
            chart_height,
            class_mask=lateral_mask,
            class_label_map=lateral_label_map,
            class_colors=LATERAL_CHART_CLASS_COLORS,
        )
        z_chart = _draw_signal_chart(velocities[:, 2], frame_index, "ego velocity z (up / vertical)", colors["z"], frame_w, chart_height)

        writer.write(np.vstack([frame, x_chart, y_chart, z_chart]))

    writer.release()
    return str(output_path)
