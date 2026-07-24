"""MP4 visualization for Step 8B low-level uncertain signal evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


_VISUALIZATION_VERSION = 2
_SELECTION_POLICY = "step8b-lowest-confidence-first-v1"
_MAX_TRACK_VIDEOS_PER_SOURCE_VIDEO = 3
_OUTPUT_WIDTH = 1920
_OUTPUT_HEIGHT = 1440
_LEFT_SCENE_WIDTH = 1100
_CUE_ORDER = (
    "leftness",
    "rightness",
    "approach",
    "recede",
    "acceleration",
    "deceleration",
)


def _number(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _track_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _stable_tie_breaker(video_id, track_id):
    value = f"{_SELECTION_POLICY}\0{video_id}\0{track_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _cue_strength(evidence):
    cues = dict(evidence.get("observable_cues", {}))
    return max((_number(cues.get(name, 0.0)) for name in _CUE_ORDER), default=0.0)


def select_step8b_visualization_tracks(
    evidence_videos,
    max_tracks_per_video=3,
):
    """Select the same most-uncertain tracks independent of input ordering."""
    limit = min(
        _MAX_TRACK_VIDEOS_PER_SOURCE_VIDEO,
        max(0, int(max_tracks_per_video)),
    )
    selected = []
    for video in sorted(
        evidence_videos,
        key=lambda row: str(row.get("video_id", "")),
    ):
        video_id = str(video.get("video_id", ""))
        unique = {}
        for evidence in video.get("track_signal_evidence", []):
            track_id = _track_id(evidence.get("track_id"))
            if video_id and track_id >= 0:
                unique.setdefault(track_id, evidence)
        ranked = sorted(
            unique.values(),
            key=lambda row: (
                _cue_strength(row),
                _stable_tie_breaker(video_id, _track_id(row.get("track_id"))),
                _track_id(row.get("track_id")),
            ),
        )
        selected.extend(
            {
                "video_id": video_id,
                "track_id": _track_id(evidence.get("track_id")),
                "evidence": evidence,
            }
            for evidence in ranked[:limit]
        )
    return selected


def _confidence_color(confidence):
    """BGR red -> yellow -> green ramp for confidence magnitude."""
    value = max(0.0, min(1.0, _number(confidence)))
    if value <= 0.5:
        fraction = value / 0.5
        return (35, int(round(70 + 175 * fraction)), 235)
    fraction = (value - 0.5) / 0.5
    return (35, int(round(245 - 35 * fraction)), int(round(235 - 180 * fraction)))


def _put_text(cv2, image, text, x, y, scale=0.48, color=(235, 235, 235), thickness=1):
    cv2.putText(
        image,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _fit_text(cv2, text, max_width, scale=0.46, thickness=1):
    value = str(text)
    if cv2.getTextSize(
        value, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )[0][0] <= max_width:
        return value
    suffix = " ..."
    while value and cv2.getTextSize(
        value + suffix,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )[0][0] > max_width:
        value = value[:-1]
    return value + suffix


def _build_evidence_panel(cv2, np, evidence, width, height):
    panel = np.full((height, width, 3), (20, 23, 29), dtype=np.uint8)
    margin = 38
    _put_text(
        cv2,
        panel,
        "STEP 8B",
        margin,
        74,
        1.80,
        (250, 250, 250),
        4,
    )
    _put_text(
        cv2,
        panel,
        "SIGNAL EVIDENCE",
        margin,
        137,
        1.50,
        (215, 222, 232),
        3,
    )
    _put_text(
        cv2,
        panel,
        (
            f"track {evidence.get('track_id', -1)} | "
            f"{evidence.get('primary_label', 'unknown')}"
        ),
        margin,
        207,
        1.42,
        (242, 242, 242),
        3,
    )
    cv2.line(
        panel,
        (margin, 255),
        (width - margin, 255),
        (75, 82, 94),
        3,
        cv2.LINE_AA,
    )

    cues = dict(evidence.get("observable_cues", {}))
    y = 330
    for name in _CUE_ORDER:
        value = max(0.0, min(1.0, _number(cues.get(name, 0.0))))
        color = _confidence_color(value)
        title = name.upper()
        _put_text(
            cv2, panel, title, margin, y, 1.55, (235, 238, 243), 4
        )
        _put_text(
            cv2,
            panel,
            f"{value:.3f}",
            margin,
            y + 67,
            1.70,
            color,
            4,
        )
        y += 170

    _put_text(
        cv2,
        panel,
        "RED low   YELLOW medium   GREEN high",
        margin,
        height - 35,
        0.90,
        (205, 210, 220),
        2,
    )
    return panel


def _frame_map(relative_video):
    return {
        int(frame.get("frame_index", offset)): frame
        for offset, frame in enumerate(relative_video.get("frames", []))
    }


def _track_object_map(relative_video, track_id):
    objects = {}
    for frame_index, frame in _frame_map(relative_video).items():
        for obj in frame.get("objects", []):
            if _track_id(obj.get("track_id")) == int(track_id):
                objects[frame_index] = obj
                break
    return objects


def _valid_bbox(obj):
    if not obj:
        return None
    box = list(obj.get("bbox", obj.get("box", [])))
    if len(box) != 4:
        return None
    values = [_number(value) for value in box]
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def _current_signal_lines(frame_index, obj):
    if not obj:
        return (
            f"frame {frame_index:05d}",
            "object absent in this frame",
        )
    position = list(
        obj.get("position_3d", obj.get("relative_position_3d", []))
    )
    x_value = _number(position[0]) if len(position) >= 3 else 0.0
    z_value = _number(position[2]) if len(position) >= 3 else 0.0
    return (
        f"frame {frame_index:05d}   x {x_value:+.3f}   z {z_value:+.3f}",
        (
            f"vx {_number(obj.get('rel_vx')):+.3f}   "
            f"vz {_number(obj.get('rel_vz')):+.3f}   "
            f"speed {_number(obj.get('rel_speed')):.3f}"
        ),
    )


def _draw_track_progress_bar(
    cv2,
    panel,
    frame_indices,
    current_frame,
    track_objects,
    width,
    present_color,
    *,
    top=0,
    bar_left=None,
    bar_right=None,
):
    """Draw frame-level object presence immediately below the scene."""
    indices = sorted(frame_indices)
    if not indices:
        return
    left = (
        max(0, int(bar_left))
        if bar_left is not None
        else 24
    )
    right = (
        min(int(width), int(bar_right))
        if bar_right is not None
        else int(width) - 24
    )
    right = max(left + 1, right)
    bar_top = top + 47
    bar_bottom = top + 83
    bar_width = right - left
    _put_text(
        cv2,
        panel,
        "TRACK PRESENCE | white marker = current frame",
        left,
        top + 31,
        1.15,
        (225, 230, 238),
        3,
    )
    cv2.rectangle(
        panel, (left, bar_top), (right, bar_bottom), (54, 58, 66), -1
    )
    count = len(indices)
    for offset, frame_index in enumerate(indices):
        x1 = left + int(math.floor(offset * bar_width / count))
        x2 = left + int(math.ceil((offset + 1) * bar_width / count))
        color = present_color if frame_index in track_objects else (72, 76, 84)
        cv2.rectangle(
            panel,
            (x1, bar_top + 2),
            (max(x1, x2 - 1), bar_bottom - 2),
            color,
            -1,
        )
    try:
        current_offset = indices.index(current_frame)
    except ValueError:
        current_offset = 0
    marker_x = left + int(
        round((current_offset + 0.5) * bar_width / count)
    )
    cv2.line(
        panel,
        (marker_x, bar_top - 4),
        (marker_x, bar_bottom + 4),
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    current_position = current_offset + 1
    _put_text(
        cv2,
        panel,
        f"{current_position}/{count}",
        max(left, right - 120),
        top + 31,
        1.10,
        (255, 255, 255),
        3,
    )


def _draw_bbox_label(
    cv2,
    scene,
    *,
    box,
    track_id,
    object_label,
    color,
):
    x1, y1, x2, y2 = box
    scene_height, scene_width = scene.shape[:2]
    x1 = max(0, min(scene_width - 1, int(x1)))
    x2 = max(0, min(scene_width - 1, int(x2)))
    y1 = max(0, min(scene_height - 1, int(y1)))
    y2 = max(0, min(scene_height - 1, int(y2)))
    thickness = max(
        3, int(round(min(scene_width, scene_height) / 260))
    )
    cv2.rectangle(
        scene, (x1, y1), (x2, y2), (0, 0, 0), thickness + 3
    )
    cv2.rectangle(scene, (x1, y1), (x2, y2), color, thickness)
    label = f"{object_label} | ID {track_id}"
    font_scale = 1.75
    text_thickness = 4
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_thickness,
    )
    text_x = max(4, min(x1, scene_width - text_width - 12))
    text_y = y1 - 10 if y1 >= text_height + baseline + 18 else y1 + text_height + 16
    text_y = max(text_height + 8, min(scene_height - baseline - 5, text_y))
    cv2.rectangle(
        scene,
        (text_x - 4, text_y - text_height - 7),
        (text_x + text_width + 7, text_y + baseline + 5),
        (0, 0, 0),
        -1,
    )
    _put_text(
        cv2,
        scene,
        label,
        text_x,
        text_y,
        font_scale,
        color,
        text_thickness,
    )


def _render_step8b_track_video(
    relative_video,
    evidence,
    output_path,
    *,
    fps=10.0,
    progress_callback=None,
):
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError:
        return None, "missing_cv2_or_numpy"

    frames = _frame_map(relative_video)
    if not frames:
        return None, "no_frames"
    first_image = None
    for frame_index in sorted(frames):
        image_path = str(frames[frame_index].get("image_path", ""))
        first_image = cv2.imread(image_path) if image_path else None
        if first_image is not None:
            break
    if first_image is None:
        return None, "missing_frame_images"

    canvas_width = _OUTPUT_WIDTH
    total_height = _OUTPUT_HEIGHT
    left_width = _LEFT_SCENE_WIDTH
    panel_width = canvas_width - left_width
    max_scene_width = left_width - 40
    max_scene_height = 980
    track_id = _track_id(evidence.get("track_id"))
    track_objects = _track_object_map(relative_video, track_id)
    strongest_cue = _cue_strength(evidence)
    confidence_color = _confidence_color(strongest_cue)
    static_panel = _build_evidence_panel(
        cv2, np, evidence, panel_width, total_height
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(0.1, float(fps)),
        (canvas_width, total_height),
    )
    if not writer.isOpened():
        return None, "writer_open_failed"

    try:
        frame_indices = sorted(frames)
        for frame_index in frame_indices:
            frame = frames[frame_index]
            image_path = str(frame.get("image_path", ""))
            image = cv2.imread(image_path) if image_path else None
            if image is None:
                image = np.zeros_like(first_image)
            frame_height, frame_width = image.shape[:2]
            scale = min(
                max_scene_width / max(1, frame_width),
                max_scene_height / max(1, frame_height),
            )
            scene_width = max(2, int(round(frame_width * scale)))
            scene_height = max(2, int(round(frame_height * scale)))
            if scene_width % 2:
                scene_width -= 1
            if scene_height % 2:
                scene_height -= 1
            scene = cv2.resize(image, (scene_width, scene_height))
            obj = track_objects.get(frame_index)
            box = _valid_bbox(obj)
            if box:
                scale_x = scene_width / max(1, frame_width)
                scale_y = scene_height / max(1, frame_height)
                x1, y1, x2, y2 = [
                    int(round(value))
                    for value in (
                        box[0] * scale_x,
                        box[1] * scale_y,
                        box[2] * scale_x,
                        box[3] * scale_y,
                    )
                ]
                object_label = str(
                    obj.get(
                        "frame_label",
                        obj.get(
                            "label",
                            evidence.get("primary_label", "unknown"),
                        ),
                    )
                )
                _draw_bbox_label(
                    cv2,
                    scene,
                    box=(x1, y1, x2, y2),
                    track_id=track_id,
                    object_label=object_label,
                    color=confidence_color,
                )
            header = (
                f"{relative_video.get('video_id', '')} | "
                f"track {track_id}"
            )
            cv2.rectangle(
                scene, (0, 0), (scene_width, 78), (0, 0, 0), -1
            )
            _put_text(
                cv2, scene, header, 20, 54, 1.50, confidence_color, 4
            )

            canvas = np.full(
                (total_height, canvas_width, 3),
                (12, 14, 18),
                dtype=np.uint8,
            )
            scene_x = (left_width - scene_width) // 2
            scene_y = 42
            canvas[
                scene_y : scene_y + scene_height,
                scene_x : scene_x + scene_width,
            ] = scene
            canvas[:, left_width:] = static_panel
            cv2.line(
                canvas,
                (left_width, 0),
                (left_width, total_height),
                (82, 88, 100),
                4,
                cv2.LINE_AA,
            )
            progress_top = scene_y + scene_height + 12
            _draw_track_progress_bar(
                cv2,
                canvas,
                frame_indices,
                frame_index,
                track_objects,
                left_width,
                confidence_color,
                top=progress_top,
                bar_left=scene_x,
                bar_right=scene_x + scene_width,
            )
            current_title_y = progress_top + 146
            _put_text(
                cv2,
                canvas,
                "CURRENT SIGNAL",
                24,
                current_title_y,
                1.40,
                (225, 230, 238),
                4,
            )
            for line_offset, line in enumerate(
                _current_signal_lines(frame_index, obj)
            ):
                _put_text(
                    cv2,
                    canvas,
                    _fit_text(cv2, line, left_width - 48, 1.35, 4),
                    24,
                    current_title_y + 67 + line_offset * 62,
                    1.35,
                    (245, 245, 245),
                    4,
                )
            writer.write(canvas)
            if progress_callback:
                progress_callback(1)
    finally:
        writer.release()
    return str(output_path), "rendered"


def render_step8b_signal_evidence_videos(
    relative_motion,
    evidence_videos,
    output_root,
    *,
    fps=10.0,
    max_tracks_per_video=3,
):
    """Render at most three deterministic, low-confidence tracks per video."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    limit = min(
        _MAX_TRACK_VIDEOS_PER_SOURCE_VIDEO,
        max(0, int(max_tracks_per_video)),
    )
    selected = select_step8b_visualization_tracks(
        evidence_videos,
        max_tracks_per_video=limit,
    )
    relative_by_video = {
        str(video.get("video_id", "")): video for video in relative_motion
    }
    available_by_video = defaultdict(set)
    selected_by_video = defaultdict(list)
    for video in evidence_videos:
        video_id = str(video.get("video_id", ""))
        for evidence in video.get("track_signal_evidence", []):
            track_id = _track_id(evidence.get("track_id"))
            if video_id and track_id >= 0:
                available_by_video[video_id].add(track_id)
    for row in selected:
        selected_by_video[row["video_id"]].append(row["track_id"])

    pruned_artifacts = []
    for video_id in sorted(available_by_video):
        selected_ids = set(selected_by_video.get(video_id, []))
        video_root = output_root / video_id
        if not video_root.exists():
            continue
        for track_root in video_root.glob("track_*"):
            if not track_root.is_dir():
                continue
            track_id = _track_id(track_root.name.removeprefix("track_"))
            if track_id in selected_ids:
                continue
            for artifact in (
                track_root / f"track_{track_id:04d}_step8b_evidence.mp4",
                track_root / f"track_{track_id:04d}_evidence.json",
            ):
                if artifact.is_file():
                    artifact.unlink()
                    pruned_artifacts.append(str(artifact))

    frame_counts = {
        video_id: len(_frame_map(video))
        for video_id, video in relative_by_video.items()
    }
    total_frames = sum(
        frame_counts.get(row["video_id"], 0) for row in selected
    )
    print(
        f"[step 8b][visualization] MP4_START tracks={len(selected)} "
        f"frames={total_frames} max_tracks_per_video={limit} "
        f"fps={float(fps):.2f} output_root={output_root}",
        flush=True,
    )
    started = time.perf_counter()
    rendered = []
    skipped = []
    with tqdm(
        total=total_frames,
        desc="[step 8b] evidence MP4",
        unit="frame",
        dynamic_ncols=True,
    ) as progress:
        for index, row in enumerate(selected, start=1):
            video_id = row["video_id"]
            track_id = row["track_id"]
            expected_frames = frame_counts.get(video_id, 0)
            encoded = [0]

            def update(count=1):
                increment = max(0, int(count))
                encoded[0] += increment
                progress.update(increment)

            progress.set_postfix_str(
                f"track={index}/{len(selected)} video={video_id} id={track_id}",
                refresh=True,
            )
            track_started = time.perf_counter()
            print(
                f"[step 8b][visualization] MP4_TRACK_START "
                f"track={index}/{len(selected)} video={video_id} "
                f"track_id={track_id} frames={expected_frames}",
                flush=True,
            )
            track_root = output_root / video_id / f"track_{track_id:04d}"
            evidence_path = track_root / f"track_{track_id:04d}_evidence.json"
            track_root.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(
                json.dumps(row["evidence"], indent=2, default=str),
                encoding="utf-8",
            )
            output_path = (
                track_root / f"track_{track_id:04d}_step8b_evidence.mp4"
            )
            try:
                path, status = _render_step8b_track_video(
                    relative_by_video.get(video_id, {}),
                    row["evidence"],
                    output_path,
                    fps=fps,
                    progress_callback=update,
                )
            except Exception as exc:
                path = None
                status = (
                    f"render_failed:{type(exc).__name__}:"
                    f"{str(exc)[:240]}"
                )
            if encoded[0] < expected_frames:
                progress.update(expected_frames - encoded[0])
            result = {
                "video_id": video_id,
                "track_id": track_id,
                "strongest_cue": _cue_strength(row["evidence"]),
                "status": status,
                "evidence_path": str(evidence_path),
            }
            if path:
                result["visualization_path"] = str(path)
                rendered.append(result)
            else:
                skipped.append(result)
            print(
                f"[step 8b][visualization] MP4_TRACK_DONE "
                f"track={index}/{len(selected)} video={video_id} "
                f"track_id={track_id} status={status} "
                f"encoded_frames={encoded[0]} "
                f"latency={time.perf_counter() - track_started:.2f}s",
                flush=True,
            )
    print(
        f"[step 8b][visualization] MP4_DONE rendered={len(rendered)} "
        f"skipped={len(skipped)} latency={time.perf_counter() - started:.2f}s",
        flush=True,
    )

    selections = []
    for video_id in sorted(available_by_video):
        selected_ids = selected_by_video.get(video_id, [])
        selected_set = set(selected_ids)
        selections.append(
            {
                "video_id": video_id,
                "available_track_ids": sorted(available_by_video[video_id]),
                "selected_track_ids": list(selected_ids),
                "unselected_track_ids": sorted(
                    available_by_video[video_id] - selected_set
                ),
            }
        )
    manifest = {
        "version": _VISUALIZATION_VERSION,
        "format": "mp4",
        "layout": "scene_left_evidence_right",
        "track_progress_position": "directly_below_original_video",
        "canvas_resolution": [_OUTPUT_WIDTH, _OUTPUT_HEIGHT],
        "canvas_aspect_ratio": "4:3",
        "selection_policy": _SELECTION_POLICY,
        "max_tracks_per_video": limit,
        "confidence_color_scale": {
            "low": "red",
            "medium": "yellow",
            "high": "green",
        },
        "num_available_tracks": sum(
            len(track_ids) for track_ids in available_by_video.values()
        ),
        "num_selected_tracks": len(selected),
        "num_rendered_videos": len(rendered),
        "num_skipped_videos": len(skipped),
        "num_pruned_stale_artifacts": len(pruned_artifacts),
        "pruned_stale_artifacts": pruned_artifacts,
        "selections": selections,
        "rendered": rendered,
        "skipped": skipped,
    }
    manifest_path = output_root / "step8b_evidence_video_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def configured_step8b_visualization_limit():
    try:
        configured = int(os.environ.get("CAUVID_STEP8B_MAX_TRACK_VIDEOS", "3"))
    except ValueError:
        configured = 3
    return min(_MAX_TRACK_VIDEOS_PER_SOURCE_VIDEO, max(0, configured))
