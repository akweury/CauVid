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


_VISUALIZATION_VERSION = 1
_SELECTION_POLICY = "step8b-lowest-confidence-first-v1"
_MAX_TRACK_VIDEOS_PER_SOURCE_VIDEO = 3
_DESCRIPTOR_ORDER = (
    "observation_quality",
    "longitudinal_trend",
    "lateral_trend",
    "speed_trend",
    "temporal_coherence",
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
                _number(row.get("evidence_confidence", 0.0)),
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


def _format_metric_lines(metrics, items_per_line=4):
    parts = []
    for key, value in metrics.items():
        label = str(key).replace("_", " ")
        if isinstance(value, float):
            rendered = f"{value:.3f}"
        else:
            rendered = str(value)
        parts.append(f"{label}={rendered}")
    return [
        " | ".join(parts[offset : offset + items_per_line])
        for offset in range(0, len(parts), items_per_line)
    ] or ["no metrics"]


def _signal_summary(prefix, signal):
    return [
        (
            f"{prefix}: trend={signal.get('trend', 'unobserved')} | "
            f"level={signal.get('level', 'unobserved')} | "
            f"confidence={_number(signal.get('confidence')):.3f} | "
            f"n={int(_number(signal.get('sample_count', 0)))} | "
            f"start={_number(signal.get('start')):+.3f} | "
            f"end={_number(signal.get('end')):+.3f} | "
            f"delta={_number(signal.get('delta')):+.3f}"
        ),
        (
            f"{prefix}: slope/frame={_number(signal.get('slope_per_frame')):+.3f} | "
            f"mean={_number(signal.get('mean')):+.3f} | "
            f"std={_number(signal.get('standard_deviation')):.3f} | "
            f"linear fit={_number(signal.get('linear_fit_coherence')):.3f} | "
            f"step sign consistency={_number(signal.get('step_sign_consistency')):.3f}"
        ),
    ]


def _descriptor_detail_lines(name, descriptor):
    if name == "observation_quality":
        return _format_metric_lines(dict(descriptor.get("metrics", {})))
    if name in {"longitudinal_trend", "lateral_trend"}:
        return [
            *_signal_summary(
                "position", dict(descriptor.get("position_signal", {}))
            ),
            *_signal_summary(
                "velocity", dict(descriptor.get("velocity_signal", {}))
            ),
        ]
    if name == "speed_trend":
        return _signal_summary(
            "speed", dict(descriptor.get("speed_signal", {}))
        )
    return _format_metric_lines(dict(descriptor.get("metrics", {})))


def _build_evidence_panel(cv2, np, evidence, width, height):
    panel = np.full((height, width, 3), (22, 25, 31), dtype=np.uint8)
    confidence = _number(evidence.get("evidence_confidence", 0.0))
    confidence_color = _confidence_color(confidence)
    margin = 24
    _put_text(
        cv2,
        panel,
        "STEP 8B | UNCERTAIN LOW-LEVEL SIGNAL EVIDENCE",
        margin,
        38,
        0.72,
        (250, 250, 250),
        2,
    )
    _put_text(
        cv2,
        panel,
        (
            f"track {evidence.get('track_id', -1)} | "
            f"observed label={evidence.get('primary_label', 'unknown')} | "
            f"overall evidence confidence={confidence:.3f}"
        ),
        margin,
        70,
        0.56,
        confidence_color,
        2,
    )
    bar_x = margin
    bar_y = 86
    bar_width = width - margin * 2
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (65, 70, 78), -1)
    cv2.rectangle(
        panel,
        (bar_x, bar_y),
        (bar_x + int(round(bar_width * max(0.0, min(1.0, confidence)))), bar_y + 15),
        confidence_color,
        -1,
    )

    descriptors = dict(evidence.get("descriptors", {}))
    y = 142
    for name in _DESCRIPTOR_ORDER:
        descriptor = dict(descriptors.get(name, {}))
        descriptor_confidence = _number(descriptor.get("confidence", 0.0))
        color = _confidence_color(descriptor_confidence)
        title = (
            f"{name.replace('_', ' ').upper()} | "
            f"state={descriptor.get('state', 'unobserved')} | "
            f"confidence={descriptor_confidence:.3f}"
        )
        _put_text(cv2, panel, title, margin, y, 0.56, color, 2)
        small_bar_x = max(margin, width - 330)
        cv2.rectangle(panel, (small_bar_x, y - 16), (width - margin, y - 5), (65, 70, 78), -1)
        cv2.rectangle(
            panel,
            (small_bar_x, y - 16),
            (
                small_bar_x
                + int(round((width - margin - small_bar_x) * max(0.0, min(1.0, descriptor_confidence)))),
                y - 5,
            ),
            color,
            -1,
        )
        y += 28
        for detail in _descriptor_detail_lines(name, descriptor):
            _put_text(
                cv2,
                panel,
                _fit_text(cv2, detail, width - margin * 2, 0.48),
                margin + 14,
                y,
                0.48,
                (215, 220, 228),
                1,
            )
            y += 28
        y += 12

    signal_reference = dict(evidence.get("signal_reference", {}))
    provenance = dict(evidence.get("provenance", {}))
    label_summary = dict(evidence.get("observed_label_summary", {}))
    _put_text(
        cv2,
        panel,
        _fit_text(
            cv2,
            (
                "SIGNAL REFERENCE | "
                f"frames={signal_reference.get('frame_start', -1)}.."
                f"{signal_reference.get('frame_end', -1)} | "
                f"observations={signal_reference.get('observation_count', 0)} | "
                f"source={signal_reference.get('source_state_key', '')}"
            ),
            width - margin * 2,
            0.43,
        ),
        margin,
        y,
        0.43,
        (195, 205, 218),
        1,
    )
    y += 25
    _put_text(
        cv2,
        panel,
        _fit_text(
            cv2,
            (
                "PROVENANCE | "
                f"observed={provenance.get('observed_count', 0)} | "
                f"repaired={provenance.get('repaired_count', 0)} | "
                f"merged={provenance.get('merged_count', 0)} | "
                f"sources={provenance.get('source_counts', {})}"
            ),
            width - margin * 2,
            0.43,
        ),
        margin,
        y,
        0.43,
        (195, 205, 218),
        1,
    )
    y += 25
    _put_text(
        cv2,
        panel,
        _fit_text(
            cv2,
            f"OBSERVED LABEL COUNTS | {label_summary.get('label_counts', {})}",
            width - margin * 2,
            0.43,
        ),
        margin,
        y,
        0.43,
        (195, 205, 218),
        1,
    )

    legend_y = height - 34
    _put_text(cv2, panel, "confidence magnitude:", margin, legend_y, 0.44, (220, 220, 220), 1)
    legend_x = margin + 185
    for label, value in (("low", 0.15), ("medium", 0.50), ("high", 0.90)):
        color = _confidence_color(value)
        cv2.rectangle(panel, (legend_x, legend_y - 14), (legend_x + 24, legend_y + 2), color, -1)
        _put_text(cv2, panel, label, legend_x + 31, legend_y, 0.42, (220, 220, 220), 1)
        legend_x += 118
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


def _current_signal_text(frame_index, obj):
    if not obj:
        return f"frame={frame_index:05d} | track absent in this frame"
    position = list(obj.get("position_3d", obj.get("relative_position_3d", [])))
    x_value = _number(position[0]) if len(position) >= 3 else 0.0
    z_value = _number(position[2]) if len(position) >= 3 else 0.0
    return (
        f"frame={frame_index:05d} | x={x_value:+.3f} | z={z_value:+.3f} | "
        f"vx={_number(obj.get('rel_vx')):+.3f} | "
        f"vz={_number(obj.get('rel_vz')):+.3f} | "
        f"speed={_number(obj.get('rel_speed')):.3f}"
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

    source_height, source_width = first_image.shape[:2]
    canvas_width = min(1920, max(1280, int(source_width)))
    if canvas_width % 2:
        canvas_width += 1
    scene_height = int(round(source_height * canvas_width / source_width))
    if scene_height % 2:
        scene_height += 1
    panel_height = 900
    total_height = scene_height + panel_height
    track_id = _track_id(evidence.get("track_id"))
    track_objects = _track_object_map(relative_video, track_id)
    confidence = _number(evidence.get("evidence_confidence", 0.0))
    confidence_color = _confidence_color(confidence)
    static_panel = _build_evidence_panel(
        cv2, np, evidence, canvas_width, panel_height
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
        for frame_index in sorted(frames):
            frame = frames[frame_index]
            image_path = str(frame.get("image_path", ""))
            image = cv2.imread(image_path) if image_path else None
            if image is None:
                image = np.zeros_like(first_image)
            frame_height, frame_width = image.shape[:2]
            scene = cv2.resize(image, (canvas_width, scene_height))
            obj = track_objects.get(frame_index)
            box = _valid_bbox(obj)
            if box:
                scale_x = canvas_width / max(1, frame_width)
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
                thickness = max(3, int(round(min(canvas_width, scene_height) / 260)))
                cv2.rectangle(scene, (x1, y1), (x2, y2), (0, 0, 0), thickness + 3)
                cv2.rectangle(scene, (x1, y1), (x2, y2), confidence_color, thickness)
            header = (
                f"{relative_video.get('video_id', '')} | track {track_id} | "
                f"evidence confidence {confidence:.3f}"
            )
            cv2.rectangle(scene, (0, 0), (canvas_width, 48), (0, 0, 0), -1)
            _put_text(cv2, scene, header, 14, 33, 0.68, confidence_color, 2)

            panel = static_panel.copy()
            cv2.rectangle(panel, (0, 104), (canvas_width, 128), (33, 37, 44), -1)
            _put_text(
                cv2,
                panel,
                _fit_text(
                    cv2,
                    _current_signal_text(frame_index, obj),
                    canvas_width - 48,
                    0.48,
                ),
                24,
                122,
                0.48,
                (245, 245, 245),
                1,
            )
            writer.write(cv2.vconcat([scene, panel]))
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
                "evidence_confidence": _number(
                    row["evidence"].get("evidence_confidence", 0.0)
                ),
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
        "layout": "scene_above_evidence_below",
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
