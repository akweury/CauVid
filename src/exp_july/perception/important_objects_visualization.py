"""Read-only Step 11 scene, ego-state, and object-motion visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import cv2
import numpy as np
from tqdm import tqdm

import config


VX_COLORS = {
    "right": (244, 126, 71),
    "straight": (80, 205, 105),
    "left": (214, 92, 222),
}
VZ_COLORS = {
    "backward": (70, 130, 245),
    "static": (155, 160, 170),
    "forward": (35, 215, 235),
}
UNKNOWN_COLOR = (85, 90, 100)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _ego_frame_maps(ego_result: Mapping[str, Any]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    final = dict(ego_result.get("final_segmentation", {}))
    output: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for axis in ("vx", "vz"):
        axis_result = dict(final.get(axis, {}))
        output[axis] = {
            int(row.get("frame_index", -1)): dict(row)
            for row in axis_result.get("frames", [])
            if int(row.get("frame_index", -1)) >= 0
        }
    return output


def _put(
    canvas: np.ndarray,
    text: str,
    origin: Sequence[int],
    scale: float = 0.65,
    color=(240, 240, 240),
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas,
        str(text),
        (int(origin[0]), int(origin[1])),
        cv2.FONT_HERSHEY_DUPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _fit_frame(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    scale = min(width / max(1, image.shape[1]), height / max(1, image.shape[0]))
    resized = cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel, scale, x, y


def _draw_scene(
    canvas: np.ndarray,
    frame: Mapping[str, Any],
    ego_labels: Mapping[str, str],
    box: Sequence[int],
) -> None:
    left, top, width, height = map(int, box)
    image = cv2.imread(str(frame.get("image_path", "")))
    if image is None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        _put(image, "SOURCE FRAME UNAVAILABLE", (330, 360), 1.0, (80, 100, 235), 2)
    panel, scale, offset_x, offset_y = _fit_frame(image, width, height)
    for obj in frame.get("objects", []):
        bbox = list(obj.get("bbox", obj.get("box", [])))
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [_safe_float(value) for value in bbox[:4]]
        p1 = (int(offset_x + x1 * scale), int(offset_y + y1 * scale))
        p2 = (int(offset_x + x2 * scale), int(offset_y + y2 * scale))
        color = (80, 235, 130) if obj.get("is_observed", False) else (65, 180, 245)
        cv2.rectangle(panel, p1, p2, color, 3)
        label = f"{obj.get('label', 'object')} | id={obj.get('track_id', '?')}"
        _put(panel, label, (p1[0], max(22, p1[1] - 7)), 0.55, color, 2)
    canvas[top : top + height, left : left + width] = panel
    label_text = f"EGO: {ego_labels.get('vz', 'unavailable')} | {ego_labels.get('vx', 'unavailable')}"
    cv2.rectangle(canvas, (left + 12, top + 12), (left + 520, top + 52), (20, 24, 31), -1)
    _put(canvas, label_text, (left + 24, top + 41), 0.72, (255, 255, 255), 2)


def _draw_timeline(
    canvas: np.ndarray,
    frame_indices: Sequence[int],
    ego_maps: Mapping[str, Mapping[int, Mapping[str, Any]]],
    current_position: int,
    box: Sequence[int],
) -> None:
    left, top, width, height = map(int, box)
    _put(canvas, "FINAL EGO-MOTION SEGMENTATION [STEP 7B]", (left, top + 25), 0.67, (245, 245, 245), 2)
    bar_left = left + 125
    bar_width = width - 145
    bar_height = 42
    for row, axis in enumerate(("vx", "vz")):
        y = top + 48 + row * 72
        _put(canvas, axis.upper(), (left + 18, y + 29), 0.72, (220, 225, 235), 2)
        color_map = VX_COLORS if axis == "vx" else VZ_COLORS
        axis_map = ego_maps.get(axis, {})
        for index, frame_index in enumerate(frame_indices):
            x1 = bar_left + int(index * bar_width / max(1, len(frame_indices)))
            x2 = bar_left + int((index + 1) * bar_width / max(1, len(frame_indices)))
            state = str(axis_map.get(int(frame_index), {}).get("state", "unavailable"))
            cv2.rectangle(
                canvas,
                (x1, y),
                (max(x1 + 1, x2), y + bar_height),
                color_map.get(state, UNKNOWN_COLOR),
                -1,
            )
        marker_x = bar_left + int((current_position + 0.5) * bar_width / max(1, len(frame_indices)))
        cv2.line(canvas, (marker_x, y - 6), (marker_x, y + bar_height + 6), (255, 255, 255), 3)
        cv2.rectangle(canvas, (bar_left, y), (bar_left + bar_width, y + bar_height), (185, 190, 200), 1)
    legend_y = top + 202
    legend = [
        ("right", VX_COLORS["right"]),
        ("straight", VX_COLORS["straight"]),
        ("left", VX_COLORS["left"]),
        ("backward", VZ_COLORS["backward"]),
        ("static", VZ_COLORS["static"]),
        ("forward", VZ_COLORS["forward"]),
    ]
    x = left + 12
    for name, color in legend:
        cv2.rectangle(canvas, (x, legend_y), (x + 22, legend_y + 18), color, -1)
        _put(canvas, name, (x + 29, legend_y + 16), 0.42, (210, 215, 225), 1)
        x += 132


def _object_priority(obj: Mapping[str, Any]) -> tuple:
    bbox = list(obj.get("bbox", []))
    area = 0.0
    if len(bbox) >= 4:
        area = max(0.0, _safe_float(bbox[2]) - _safe_float(bbox[0])) * max(
            0.0, _safe_float(bbox[3]) - _safe_float(bbox[1])
        )
    return (
        bool(obj.get("is_observed", False)),
        bool(obj.get("has_rel_motion", False)),
        area,
        _safe_float(obj.get("score", 0.0)),
    )


def _draw_object_cards(
    canvas: np.ndarray,
    source_image: np.ndarray | None,
    objects: Iterable[Mapping[str, Any]],
    box: Sequence[int],
) -> None:
    left, top, width, height = map(int, box)
    _put(canvas, "CURRENT-FRAME OBJECT MOTION [STEP 8A/8B]", (left + 16, top + 34), 0.72, (250, 250, 250), 2)
    selected = sorted((dict(row) for row in objects), key=_object_priority, reverse=True)[:4]
    card_top = top + 55
    card_height = max(120, (height - 70) // 4)
    for slot in range(4):
        y = card_top + slot * card_height
        cv2.rectangle(canvas, (left + 10, y), (left + width - 10, y + card_height - 10), (36, 42, 52), -1)
        cv2.rectangle(canvas, (left + 10, y), (left + width - 10, y + card_height - 10), (85, 95, 110), 1)
        if slot >= len(selected):
            _put(canvas, f"OBJECT {slot + 1}: none in current frame", (left + 30, y + 55), 0.58, (130, 140, 155), 1)
            continue
        obj = selected[slot]
        crop_left = left + 24
        crop_top = y + 16
        crop_width = 190
        crop_height = card_height - 42
        crop = None
        bbox = list(obj.get("bbox", []))
        if source_image is not None and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(round(_safe_float(value))) for value in bbox[:4]]
            x1, x2 = sorted((max(0, x1), min(source_image.shape[1], x2)))
            y1, y2 = sorted((max(0, y1), min(source_image.shape[0], y2)))
            if x2 > x1 and y2 > y1:
                crop = source_image[y1:y2, x1:x2]
        if crop is not None and crop.size:
            crop_panel, _, _, _ = _fit_frame(crop, crop_width, crop_height)
            canvas[crop_top : crop_top + crop_height, crop_left : crop_left + crop_width] = crop_panel
        else:
            cv2.rectangle(
                canvas,
                (crop_left, crop_top),
                (crop_left + crop_width, crop_top + crop_height),
                (22, 26, 32),
                -1,
            )
            _put(canvas, "NO CROP", (crop_left + 45, crop_top + crop_height // 2), 0.5, (110, 120, 135), 1)
        text_x = crop_left + crop_width + 22
        label = str(obj.get("label", "object"))
        _put(canvas, f"{label} | track {obj.get('track_id', '?')}", (text_x, y + 36), 0.68, (90, 235, 145), 2)
        _put(
            canvas,
            f"motion: {obj.get('vx_state', 'unavailable')} | {obj.get('vz_state', 'unavailable')} | {obj.get('speed_state', 'unavailable')}",
            (text_x, y + 72),
            0.55,
            (235, 220, 115),
            1,
        )
        _put(
            canvas,
            f"rel vx={_safe_float(obj.get('rel_vx')):+.3f}  rel vz={_safe_float(obj.get('rel_vz')):+.3f}  speed={_safe_float(obj.get('rel_speed')):.3f}",
            (text_x, y + 105),
            0.51,
            (215, 220, 230),
            1,
        )
        _put(
            canvas,
            f"position: {obj.get('x_position_state', 'unavailable')} | {obj.get('distance_state', 'unavailable')}  conf={_safe_float(obj.get('score')):.2f}",
            (text_x, y + 136),
            0.49,
            (175, 190, 210),
            1,
        )


def render_step11_video(
    relative_video: Mapping[str, Any],
    ego_result: Mapping[str, Any],
    output_path: Path,
    fps: float = 10.0,
) -> Dict[str, Any]:
    frames = list(relative_video.get("frames", []))
    if not frames:
        return {"status": "skipped", "reason": "no_frames", "path": None}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1920, 1080
    left_width = 1120
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(fps)),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open Step 11 MP4 writer: {output_path}")
    frame_indices = [int(row.get("frame_index", index)) for index, row in enumerate(frames)]
    ego_maps = _ego_frame_maps(ego_result)
    try:
        for position, frame in enumerate(frames):
            canvas = np.full((height, width, 3), (22, 26, 33), dtype=np.uint8)
            frame_index = frame_indices[position]
            labels = {
                axis: str(ego_maps.get(axis, {}).get(frame_index, {}).get("state", "unavailable"))
                for axis in ("vx", "vz")
            }
            _draw_scene(canvas, frame, labels, (20, 48, left_width - 40, 710))
            _draw_timeline(canvas, frame_indices, ego_maps, position, (30, 775, left_width - 60, 260))
            source_image = cv2.imread(str(frame.get("image_path", "")))
            _draw_object_cards(
                canvas,
                source_image,
                frame.get("objects", []),
                (left_width + 10, 18, width - left_width - 20, height - 36),
            )
            cv2.line(canvas, (left_width, 0), (left_width, height), (100, 110, 125), 2)
            _put(
                canvas,
                f"STEP 11 AUDIT | {relative_video.get('video_id', '')} | FRAME {frame_index}",
                (24, 32),
                0.72,
                (245, 245, 245),
                2,
            )
            writer.write(canvas)
    finally:
        writer.release()
    return {
        "status": "rendered",
        "video_id": str(relative_video.get("video_id", "")),
        "path": str(output_path),
        "fps": float(fps),
        "num_frames": len(frames),
        "max_objects_per_frame": 4,
        "layout": "scene_and_ego_timeline_left_current_object_motion_right",
    }


def render_step11_visualizations(
    state: Mapping[str, Any],
    max_videos: int = 5,
    fps: float = 10.0,
) -> Dict[str, Any]:
    output_root = config.get_output_path("pipeline_output") / "11_important_objects_visualization"
    output_root.mkdir(parents=True, exist_ok=True)
    ego_by_video = {
        str(row.get("video_id", "")): row
        for row in state.get("final_ego_symbols", [])
    }
    relative_videos = sorted(
        state.get("relative_object_motion", []),
        key=lambda row: str(row.get("video_id", "")),
    )
    selected = relative_videos[: max(0, int(max_videos))]
    outputs: List[Dict[str, Any]] = []
    for relative_video in tqdm(selected, desc="[step 11] visualization", unit="video"):
        video_id = str(relative_video.get("video_id", ""))
        outputs.append(
            render_step11_video(
                relative_video,
                ego_by_video.get(video_id, {}),
                output_root / f"{video_id}_important_objects.mp4",
                fps=fps,
            )
        )
    manifest = {
        "version": 1,
        "stage": "11_important_objects_visualization",
        "selection_decisions_modified": False,
        "source_video_count": len(relative_videos),
        "max_visualization_videos": int(max_videos),
        "num_rendered": sum(row.get("status") == "rendered" for row in outputs),
        "outputs": outputs,
    }
    manifest_path = output_root / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
    }
