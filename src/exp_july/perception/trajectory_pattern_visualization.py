"""Image and MP4 diagnostics for the Step 8B/8C trajectory pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


PATTERNS = (
    "stationary", "same_direction", "opposite_direction", "approaching",
    "receding", "crossing", "turning", "lane_entry", "overtaking", "unknown",
)
RESIDUALS = (
    "position", "direction", "speed", "acceleration", "path_intersection",
    "ttc", "continuity", "depth_consistency", "ego_motion_consistency",
)
_TRACK_VIDEO_SELECTION_NAMESPACE = "step8bc-track-video-v1"
_MAX_TRACK_VIDEOS_PER_VIDEO = 10


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(cv2, image, text, x, y, scale=0.46, color=(225, 225, 225), thickness=1):
    cv2.putText(
        image, str(text), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )


def _fit_text(cv2, text, width, scale=0.40, thickness=1):
    words = str(text).split()
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )[0][0] > width:
            return current + " ..."
        current = candidate
    return current


def _stable_track_rank(video_id, track_id):
    payload = (
        f"{_TRACK_VIDEO_SELECTION_NAMESPACE}\0{video_id}\0{int(track_id)}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), int(track_id)


def select_deterministic_track_records(records, max_tracks_per_video=10):
    """Select a stable, order-independent sample capped separately per video."""
    limit = min(
        _MAX_TRACK_VIDEOS_PER_VIDEO,
        max(0, int(max_tracks_per_video)),
    )
    records_by_video = defaultdict(dict)
    for record in records:
        video_id = str(record.get("video_id", ""))
        try:
            track_id = int(record.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if not video_id or track_id < 0:
            continue
        records_by_video[video_id].setdefault(track_id, record)

    selected = []
    for video_id in sorted(records_by_video):
        ranked_track_ids = sorted(
            records_by_video[video_id],
            key=lambda track_id: _stable_track_rank(video_id, track_id),
        )
        selected.extend(
            records_by_video[video_id][track_id]
            for track_id in ranked_track_ids[:limit]
        )
    return selected


def _residual_map(candidates):
    rows = {}
    for candidate in candidates:
        pattern_id = str(candidate.get("pattern_id", "unknown"))
        vector = dict(candidate.get("residual_vector", {}))
        rows[pattern_id] = {
            residual_id: _number(vector.get(residual_id))
            for residual_id in RESIDUALS
        }
    return rows


def _residual_delta(before, final):
    return {
        pattern_id: {
            residual_id: (
                _number(final.get(pattern_id, {}).get(residual_id))
                - _number(before.get(pattern_id, {}).get(residual_id))
            )
            for residual_id in RESIDUALS
        }
        for pattern_id in PATTERNS
        if pattern_id in before or pattern_id in final
    }


def _threshold_distance_rows(track, validation):
    """Expose signed margins to the 8B uncertain/invalid rule boundaries."""
    statistics = dict(track.get("trajectory_statistics", {}))
    metrics = dict(validation.get("step_metrics", {}))
    thresholds = dict(validation.get("thresholds", {}))
    checks = dict(validation.get("checks", {}))

    def check_value(check_id, key, default=0.0):
        return _number(dict(checks.get(check_id, {})).get(key, default))

    upper_bound_rows = (
        (
            "frame_gap",
            _number(
                statistics.get(
                    "max_frame_gap",
                    check_value(
                        "trajectory_discontinuity", "max_frame_gap", 0.0
                    ),
                )
            ),
            "max_uncertain_frame_gap",
            "max_valid_frame_gap",
        ),
        (
            "bbox_center_step_diag_ratio",
            _number(metrics.get("max_bbox_center_step_diag_ratio")),
            "max_uncertain_center_step_diag_ratio",
            "max_invalid_center_step_diag_ratio",
        ),
        (
            "bbox_size_ratio",
            _number(metrics.get("max_bbox_size_ratio")),
            "max_uncertain_bbox_size_ratio",
            "max_invalid_bbox_size_ratio",
        ),
        (
            "depth_step_per_frame",
            _number(metrics.get("max_depth_step_per_frame")),
            "max_uncertain_depth_step_per_frame",
            "max_invalid_depth_step_per_frame",
        ),
        (
            "relative_velocity_delta",
            _number(metrics.get("max_rel_velocity_delta")),
            "max_uncertain_rel_velocity_delta",
            "max_invalid_rel_velocity_delta",
        ),
        (
            "relative_speed",
            _number(metrics.get("max_rel_speed")),
            "max_uncertain_rel_speed",
            "max_invalid_rel_speed",
        ),
    )
    rows = []
    for rule_id, value, uncertain_key, invalid_key in upper_bound_rows:
        uncertain = _number(thresholds.get(uncertain_key))
        invalid = _number(thresholds.get(invalid_key))
        rows.append(
            {
                "rule_id": rule_id,
                "comparison": "upper_bound",
                "value": value,
                "uncertain_threshold": uncertain,
                "invalid_threshold": invalid,
                "signed_distance_to_uncertain": value - uncertain,
                "signed_distance_to_invalid": value - invalid,
            }
        )

    motion_ratio = _number(
        statistics.get(
            "has_motion_ratio",
            check_value("motion_evidence", "has_motion_ratio", 0.0),
        )
    )
    minimum_motion_ratio = _number(thresholds.get("min_motion_ratio"))
    rows.append(
        {
            "rule_id": "motion_evidence_ratio",
            "comparison": "lower_bound",
            "value": motion_ratio,
            "uncertain_threshold": minimum_motion_ratio,
            "invalid_threshold": None,
            "signed_distance_to_uncertain": minimum_motion_ratio - motion_ratio,
            "signed_distance_to_invalid": None,
        }
    )
    reversals = _number(metrics.get("direction_reversal_count"))
    rows.append(
        {
            "rule_id": "direction_reversal_count",
            "comparison": "upper_bound",
            "value": reversals,
            "uncertain_threshold": 1.0,
            "invalid_threshold": 2.0,
            "signed_distance_to_uncertain": reversals - 1.0,
            "signed_distance_to_invalid": reversals - 2.0,
        }
    )
    return rows


def build_step8bc_track_video_payload(record):
    """Build the complete, JSON-safe 8B/8C diagnostic payload for one track."""
    track = copy.deepcopy(dict(record.get("symbolic_track", {})))
    validation = copy.deepcopy(dict(track.get("source_validation", {})))
    trajectory_statistics = copy.deepcopy(
        dict(
            track.get(
                "trajectory_statistics",
                record.get("step8b_trajectory_statistics", {}),
            )
        )
    )
    uncertainty = copy.deepcopy(
        dict(track.get("uncertainty", record.get("step8b_uncertainty", {})))
    )
    motion_significance = copy.deepcopy(
        dict(
            track.get(
                "motion_significance_assessment",
                record.get("step8b_motion_significance_assessment", {}),
            )
        )
    )
    fact_decision = copy.deepcopy(
        dict(
            track.get(
                "fact_decision",
                record.get("step8b_fact_decision", {}),
            )
        )
    )
    track["trajectory_statistics"] = trajectory_statistics
    before = _residual_map(record.get("pattern_candidates", []))
    final = _residual_map(record.get("final_pattern_candidates", []))
    for pattern_id, vector in before.items():
        final.setdefault(pattern_id, copy.deepcopy(vector))
    selected = copy.deepcopy(dict(record.get("selected_candidate", {})))

    step8b_metrics = {
        "track_facts": {
            key: copy.deepcopy(track.get(key))
            for key in (
                "object_class",
                "position",
                "bbox_size",
                "relative_motion",
                "direction",
                "persistence",
                "confidence",
                "provenance",
                "source_decision",
            )
        },
        "trajectory_statistics": trajectory_statistics,
        "uncertainty": uncertainty,
        "validation": validation,
        "motion_significance_assessment": motion_significance,
        "fact_decision": fact_decision,
        "threshold_distances": _threshold_distance_rows(track, validation),
    }
    step8c_residual_distances = {
        "pattern_order": list(PATTERNS),
        "residual_order": list(RESIDUALS),
        "before": before,
        "final": final,
        "delta_final_minus_before": _residual_delta(before, final),
        "selected_candidate_before": copy.deepcopy(
            dict(selected.get("residual_vector_before", {}))
        ),
        "selected_candidate_after": copy.deepcopy(
            dict(selected.get("residual_vector_after", {}))
        ),
        "selected_candidate_improvement": selected.get(
            "residual_improvement"
        ),
    }
    return {
        "schema_version": 1,
        "video_id": str(record.get("video_id", "")),
        "track_id": int(record.get("track_id", -1)),
        "step8b_metrics": step8b_metrics,
        "step8c_residual_distances": step8c_residual_distances,
        "step8c": {
            "pattern_candidates": copy.deepcopy(
                list(record.get("pattern_candidates", []))
            ),
            "final_pattern_candidates": copy.deepcopy(
                list(record.get("final_pattern_candidates", []))
            ),
            "candidate_repairs": copy.deepcopy(
                list(record.get("candidate_repairs", []))
            ),
            "selected_candidate": selected,
            "repair_applied": bool(record.get("repair_applied", False)),
            "validated_pattern": str(
                record.get("validated_pattern", "unknown")
            ),
            "final_pattern": str(record.get("final_pattern", "unknown")),
            "final_validation_status": str(
                record.get("final_validation_status", "unknown")
            ),
            "final_selection_reason": str(
                record.get("final_selection_reason", "")
            ),
            "provenance": copy.deepcopy(dict(record.get("provenance", {}))),
        },
    }


def _flatten_display_scalars(value, prefix=""):
    rows = []
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_display_scalars(value[key], child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            rows.extend(
                _flatten_display_scalars(child, f"{prefix}[{index}]")
            )
    elif isinstance(value, (str, int, float, bool)) or value is None:
        rows.append((prefix, value))
    return rows


def _display_value(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value == 0.0:
            return "0"
        if abs(value) >= 10000 or abs(value) < 0.001:
            return f"{value:.2e}"
        return f"{value:.4g}"
    return str(value)


def _clip_text_to_width(cv2, text, width, scale, thickness=1):
    text = str(text)
    if cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )[0][0] <= width:
        return text
    suffix = "..."
    while text and cv2.getTextSize(
        text + suffix, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )[0][0] > width:
        text = text[:-1]
    return text + suffix


def _draw_metric_grid(cv2, panel, rows, x, y, width, height):
    if not rows:
        _text(cv2, panel, "No 8B metrics available", x, y + 22, 0.36)
        return
    row_height = 18
    rows_per_column = max(1, int(height // row_height))
    column_count = max(1, int(math.ceil(len(rows) / rows_per_column)))
    column_width = max(120, int(width / column_count))
    scale = 0.25 if column_width < 220 else 0.29
    for index, (key, value) in enumerate(rows):
        column = index // rows_per_column
        row = index % rows_per_column
        left = x + column * column_width
        baseline = y + row * row_height + 14
        text = f"{key}={_display_value(value)}"
        _text(
            cv2,
            panel,
            _clip_text_to_width(
                cv2, text, column_width - 8, scale, thickness=1
            ),
            left,
            baseline,
            scale,
            (218, 221, 226),
        )


def _draw_residual_table(cv2, panel, payload, x, y, width):
    residual_payload = dict(payload.get("step8c_residual_distances", {}))
    before = dict(residual_payload.get("before", {}))
    final = dict(residual_payload.get("final", {}))
    pattern_width = 150
    cell_width = max(80, int((width - pattern_width) / len(RESIDUALS)))
    row_height = 25
    selected_pattern = str(
        dict(payload.get("step8c", {})).get("final_pattern", "unknown")
    )
    _text(
        cv2,
        panel,
        "8C ALL PATTERN RESIDUAL DISTANCES (before/final)",
        x,
        y,
        0.39,
        (225, 190, 70),
        1,
    )
    header_y = y + 28
    _text(cv2, panel, "pattern", x + 2, header_y, 0.28, (180, 185, 195))
    for residual_index, residual_id in enumerate(RESIDUALS):
        left = x + pattern_width + residual_index * cell_width
        heading = _clip_text_to_width(
            cv2, residual_id, cell_width - 5, 0.25
        )
        _text(
            cv2, panel, heading, left + 2, header_y, 0.25, (180, 185, 195)
        )

    for pattern_index, pattern_id in enumerate(PATTERNS):
        baseline = header_y + (pattern_index + 1) * row_height
        if pattern_id == selected_pattern:
            cv2.rectangle(
                panel,
                (x, baseline - 18),
                (x + width, baseline + 5),
                (39, 59, 47),
                -1,
            )
        _text(
            cv2,
            panel,
            ("* " if pattern_id == selected_pattern else "  ") + pattern_id,
            x + 2,
            baseline,
            0.29,
            (70, 220, 100)
            if pattern_id == selected_pattern
            else (235, 235, 235),
            1,
        )
        before_vector = dict(before.get(pattern_id, {}))
        final_vector = dict(final.get(pattern_id, before_vector))
        for residual_index, residual_id in enumerate(RESIDUALS):
            left = x + pattern_width + residual_index * cell_width
            value_text = (
                f"{_number(before_vector.get(residual_id)):.3g}/"
                f"{_number(final_vector.get(residual_id)):.3g}"
            )
            _text(
                cv2,
                panel,
                _clip_text_to_width(
                    cv2, value_text, cell_width - 5, 0.26
                ),
                left + 2,
                baseline,
                0.26,
                (235, 235, 235),
            )
    return header_y + (len(PATTERNS) + 1) * row_height


def _draw_repair_candidate_table(cv2, panel, payload, x, y, width, height):
    step8c = dict(payload.get("step8c", {}))
    repairs = list(step8c.get("candidate_repairs", []))
    selected_id = str(
        dict(step8c.get("selected_candidate", {})).get("candidate_id", "")
    )
    _text(
        cv2,
        panel,
        "8C ALL REPAIR CANDIDATES | score, residual improvement, retention, issue cost",
        x,
        y,
        0.37,
        (225, 190, 70),
        1,
    )
    if not repairs:
        _text(cv2, panel, "No repair candidates", x + 4, y + 28, 0.31)
        return

    row_height = 18
    rows_per_column = max(1, int((height - 28) // row_height))
    column_count = max(1, int(math.ceil(len(repairs) / rows_per_column)))
    column_width = int(width / column_count)
    for index, repair in enumerate(repairs):
        column = index // rows_per_column
        row = index % rows_per_column
        left = x + column * column_width
        baseline = y + 28 + row * row_height
        candidate_id = str(repair.get("candidate_id", "unknown"))
        selected = candidate_id == selected_id
        score = repair.get("final_score")
        score_text = "-" if score is None else f"{_number(score):.3f}"
        summary = (
            f"{'*' if selected else ' '} {candidate_id} "
            f"{repair.get('symbolic_verdict', repair.get('decision', '?'))} "
            f"s={score_text} ri={_number(repair.get('residual_improvement')):+.3f} "
            f"ret={_number(repair.get('observation_retention')):.3f} "
            f"issue={_number(repair.get('issue_cost_before')):.2g}/"
            f"{_number(repair.get('issue_cost_after')):.2g}"
        )
        _text(
            cv2,
            panel,
            _clip_text_to_width(
                cv2, summary, column_width - 8, 0.27
            ),
            left,
            baseline,
            0.27,
            (70, 220, 100) if selected else (220, 222, 226),
            1,
        )


def _step8b_display_metrics(payload):
    step8b = copy.deepcopy(dict(payload.get("step8b_metrics", {})))
    validation = dict(step8b.get("validation", {}))
    display = {
        "facts": dict(step8b.get("track_facts", {})),
        "statistics": dict(step8b.get("trajectory_statistics", {})),
        "uncertainty": dict(step8b.get("uncertainty", {})),
        "validation": {
            "status": validation.get(
                "validation_status", validation.get("status", "unknown")
            ),
            "step_metrics": dict(validation.get("step_metrics", {})),
            "thresholds": dict(validation.get("thresholds", {})),
            "checks": dict(validation.get("checks", {})),
            "ego_motion_consistency": dict(
                validation.get("ego_motion_consistency", {})
            ),
        },
        "motion_significance": dict(
            step8b.get("motion_significance_assessment", {})
        ),
        "fact_decision": dict(step8b.get("fact_decision", {})),
        "threshold_distances": list(
            step8b.get("threshold_distances", [])
        ),
    }
    return _flatten_display_scalars(display)


def _build_step8bc_static_panel(cv2, np, payload, width, height):
    panel = np.full((height, width, 3), (22, 24, 28), dtype=np.uint8)
    step8b = dict(payload.get("step8b_metrics", {}))
    validation = dict(step8b.get("validation", {}))
    step8c = dict(payload.get("step8c", {}))
    _text(
        cv2,
        panel,
        (
            f"STEP 8B + 8C | video {payload.get('video_id', '')} | "
            f"track {payload.get('track_id', -1)}"
        ),
        18,
        32,
        0.62,
        (242, 242, 242),
        2,
    )
    _text(
        cv2,
        panel,
        (
            f"8B={validation.get('validation_status', validation.get('status', 'unknown'))} "
            f"decision={dict(step8b.get('track_facts', {})).get('source_decision', '')} | "
            f"8C pattern={step8c.get('final_pattern', 'unknown')} "
            f"repair={step8c.get('repair_applied', False)} "
            f"validation={step8c.get('final_validation_status', 'unknown')}"
        ),
        18,
        60,
        0.39,
        (80, 215, 240),
        1,
    )
    _text(
        cv2,
        panel,
        "8B ALL METRICS, THRESHOLDS, CHECK VALUES, AND SIGNED RULE DISTANCES",
        18,
        108,
        0.39,
        (225, 190, 70),
        1,
    )
    _draw_metric_grid(
        cv2,
        panel,
        _step8b_display_metrics(payload),
        18,
        122,
        width - 36,
        420,
    )
    residual_bottom = _draw_residual_table(
        cv2, panel, payload, 18, 575, width - 36
    )
    _draw_repair_candidate_table(
        cv2,
        panel,
        payload,
        18,
        residual_bottom + 15,
        width - 36,
        max(120, height - residual_bottom - 70),
    )
    _text(
        cv2,
        panel,
        (
            "Residual cells are before/final. Complete unabridged metrics and "
            "every candidate residual vector are saved in the sibling JSON."
        ),
        18,
        height - 18,
        0.30,
        (165, 170, 180),
        1,
    )
    return panel


def _video_frame_map(video):
    return {
        int(frame.get("frame_index", index)): dict(frame)
        for index, frame in enumerate(dict(video or {}).get("frames", []))
    }


def _track_objects_by_frame(video, track_id):
    rows = {}
    for frame_index, frame in _video_frame_map(video).items():
        for obj in frame.get("objects", []):
            try:
                object_track_id = int(obj.get("track_id", -1))
            except (TypeError, ValueError):
                continue
            if object_track_id == int(track_id):
                rows[frame_index] = dict(obj)
                break
    return rows


def _valid_box(value):
    try:
        values = [float(item) for item in list(value)]
    except (TypeError, ValueError):
        return None
    if (
        len(values) != 4
        or not all(math.isfinite(item) for item in values)
        or values[2] <= values[0]
        or values[3] <= values[1]
    ):
        return None
    return values


def _draw_scaled_box(
    cv2,
    image,
    obj,
    source_width,
    source_height,
    color,
    label,
    text_y_offset=0,
):
    if not obj:
        return
    box = _valid_box(obj.get("bbox", obj.get("box", [])))
    if box is None:
        return
    image_height, image_width = image.shape[:2]
    scale_x = image_width / max(1.0, float(source_width))
    scale_y = image_height / max(1.0, float(source_height))
    x1, y1, x2, y2 = [
        int(round(value * scale))
        for value, scale in zip(box, (scale_x, scale_y, scale_x, scale_y))
    ]
    x1 = max(0, min(image_width - 1, x1))
    x2 = max(0, min(image_width - 1, x2))
    y1 = max(0, min(image_height - 1, y1))
    y2 = max(0, min(image_height - 1, y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 5)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text_y = max(24, y1 - 8 - text_y_offset)
    cv2.rectangle(
        image,
        (x1, max(0, text_y - 18)),
        (min(image_width - 1, x1 + 230), min(image_height - 1, text_y + 5)),
        (0, 0, 0),
        -1,
    )
    _text(cv2, image, label, x1 + 3, text_y, 0.43, color, 1)


def _current_motion_text(frame_index, pre_obj, final_obj):
    def motion(obj):
        if not obj:
            return "absent"
        position = list(
            obj.get("position_3d", obj.get("relative_position_3d", []))
        )
        x_value = _number(position[0]) if len(position) >= 3 else 0.0
        z_value = _number(position[2]) if len(position) >= 3 else 0.0
        return (
            f"x={x_value:+.2f} z={z_value:.2f} "
            f"vx={_number(obj.get('rel_vx')):+.2f} "
            f"vz={_number(obj.get('rel_vz')):+.2f} "
            f"speed={_number(obj.get('rel_speed')):.2f}"
        )

    return (
        f"frame={int(frame_index):05d} | 8B {motion(pre_obj)} | "
        f"8C {motion(final_obj)}"
    )


def _render_step8bc_track_video(
    *,
    record,
    payload,
    pre_pattern_video,
    final_video,
    output_path,
    fps=10.0,
):
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError:
        return None, "missing_cv2_or_numpy"

    pre_frames = _video_frame_map(pre_pattern_video)
    final_frames = _video_frame_map(final_video)
    frame_indices = sorted(set(pre_frames) | set(final_frames))
    if not frame_indices:
        return None, "no_frames"

    first_image = None
    for frame_index in frame_indices:
        frame = final_frames.get(frame_index, pre_frames.get(frame_index, {}))
        image_path = str(frame.get("image_path", ""))
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
    panel_height = 1220
    total_height = scene_height + panel_height
    pre_track = _track_objects_by_frame(
        pre_pattern_video, int(record.get("track_id", -1))
    )
    final_track = _track_objects_by_frame(
        final_video, int(record.get("track_id", -1))
    )
    static_panel = _build_step8bc_static_panel(
        cv2, np, payload, canvas_width, panel_height
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
        for frame_index in frame_indices:
            frame = final_frames.get(
                frame_index, pre_frames.get(frame_index, {})
            )
            image_path = str(frame.get("image_path", ""))
            image = cv2.imread(image_path) if image_path else None
            if image is None:
                image = np.zeros_like(first_image)
            source_frame_height, source_frame_width = image.shape[:2]
            scene = cv2.resize(image, (canvas_width, scene_height))
            pre_obj = pre_track.get(frame_index)
            final_obj = final_track.get(frame_index)
            _draw_scaled_box(
                cv2,
                scene,
                pre_obj,
                source_frame_width,
                source_frame_height,
                (40, 185, 245),
                "8B original",
                text_y_offset=23,
            )
            _draw_scaled_box(
                cv2,
                scene,
                final_obj,
                source_frame_width,
                source_frame_height,
                (70, 220, 100),
                "8C final",
            )
            header = (
                f"{payload.get('video_id', '')} | "
                f"track {payload.get('track_id', -1)} | "
                f"frame {frame_index:05d}"
            )
            cv2.rectangle(scene, (0, 0), (canvas_width, 42), (0, 0, 0), -1)
            _text(cv2, scene, header, 12, 29, 0.62, (245, 245, 245), 2)

            panel = static_panel.copy()
            cv2.rectangle(
                panel, (0, 69), (canvas_width, 94), (31, 35, 42), -1
            )
            _text(
                cv2,
                panel,
                _clip_text_to_width(
                    cv2,
                    _current_motion_text(frame_index, pre_obj, final_obj),
                    canvas_width - 36,
                    0.36,
                ),
                18,
                87,
                0.36,
                (235, 235, 235),
                1,
            )
            writer.write(cv2.vconcat([scene, panel]))
    finally:
        writer.release()
    return str(output_path), "rendered"


def render_step8bc_track_videos(
    state,
    output_root,
    fps=10.0,
    max_tracks_per_video=10,
):
    """Render stable, capped per-track Step 8B/8C diagnostic MP4s."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = list(state.get("trajectory_pattern_records", []))
    selected_records = select_deterministic_track_records(
        records,
        max_tracks_per_video=max_tracks_per_video,
    )
    pre_by_video = {
        str(video.get("video_id", "")): video
        for video in state.get("pre_pattern_relative_object_motion", [])
    }
    final_by_video = {
        str(video.get("video_id", "")): video
        for video in state.get("relative_object_motion", [])
    }
    available_by_video = defaultdict(set)
    selected_by_video = defaultdict(list)
    for record in records:
        video_id = str(record.get("video_id", ""))
        try:
            track_id = int(record.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if video_id and track_id >= 0:
            available_by_video[video_id].add(track_id)
    for record in selected_records:
        selected_by_video[str(record.get("video_id", ""))].append(
            int(record.get("track_id", -1))
        )

    rendered = []
    skipped = []
    for record in selected_records:
        video_id = str(record.get("video_id", ""))
        track_id = int(record.get("track_id", -1))
        track_root = output_root / video_id / f"track_{track_id:04d}"
        track_root.mkdir(parents=True, exist_ok=True)
        payload = build_step8bc_track_video_payload(record)
        metrics_path = (
            track_root / f"track_{track_id:04d}_8b_8c_metrics.json"
        )
        metrics_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        output_path = track_root / f"track_{track_id:04d}_8b_8c.mp4"
        try:
            path, status = _render_step8bc_track_video(
                record=record,
                payload=payload,
                pre_pattern_video=pre_by_video.get(video_id, {}),
                final_video=final_by_video.get(
                    video_id, pre_by_video.get(video_id, {})
                ),
                output_path=output_path,
                fps=fps,
            )
        except Exception as exc:
            path = None
            status = (
                f"render_failed:{type(exc).__name__}:"
                f"{str(exc)[:240]}"
            )
        row = {
            "video_id": video_id,
            "track_id": track_id,
            "status": status,
            "metrics_path": str(metrics_path),
        }
        if path:
            row["visualization_path"] = str(path)
            rendered.append(row)
        else:
            skipped.append(row)

    selections = []
    for video_id in sorted(available_by_video):
        selected_ids = list(selected_by_video.get(video_id, []))
        selected_set = set(selected_ids)
        selection = {
            "video_id": video_id,
            "available_track_ids": sorted(available_by_video[video_id]),
            "selected_track_ids": selected_ids,
            "unselected_track_ids": sorted(
                available_by_video[video_id] - selected_set
            ),
        }
        selections.append(selection)
        video_root = output_root / video_id
        video_root.mkdir(parents=True, exist_ok=True)
        (video_root / "step8bc_track_video_selection.json").write_text(
            json.dumps(
                {
                    "selection_policy": _TRACK_VIDEO_SELECTION_NAMESPACE,
                    "max_tracks_per_video": min(
                        _MAX_TRACK_VIDEOS_PER_VIDEO,
                        max(0, int(max_tracks_per_video)),
                    ),
                    **selection,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest = {
        "version": 1,
        "selection_policy": _TRACK_VIDEO_SELECTION_NAMESPACE,
        "max_tracks_per_video": min(
            _MAX_TRACK_VIDEOS_PER_VIDEO,
            max(0, int(max_tracks_per_video)),
        ),
        "num_available_tracks": sum(
            len(values) for values in available_by_video.values()
        ),
        "num_selected_tracks": len(selected_records),
        "num_rendered_videos": len(rendered),
        "num_skipped_videos": len(skipped),
        "selections": selections,
        "rendered": rendered,
        "skipped": skipped,
    }
    manifest_path = output_root / "step8bc_track_video_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {**manifest, "manifest_path": str(manifest_path)}


def _render_track(record, output_path):
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError:
        return False, "missing_cv2_or_numpy"

    image = np.full((1500, 1800, 3), (22, 24, 28), dtype=np.uint8)
    white = (242, 242, 242)
    muted = (170, 175, 185)
    green = (70, 220, 100)
    red = (75, 90, 235)
    amber = (60, 190, 245)
    cyan = (225, 190, 70)

    video_id = str(record.get("video_id", ""))
    track_id = int(record.get("track_id", -1))
    _text(
        cv2, image,
        f"STEP 8C | TRAJECTORY PATTERN CLOSED LOOP | video {video_id} | track {track_id}",
        28, 48, 0.78, white, 2,
    )

    flow = (
        "symbolic abstraction", "all-pattern residuals", "LLM interpretation",
        "multi-repair", "symbolic validation", "final selection",
    )
    x = 28
    for index, label in enumerate(flow):
        box_width = 260 if index < 5 else 250
        cv2.rectangle(image, (x, 72), (x + box_width, 122), (55, 60, 70), -1)
        _text(cv2, image, label, x + 12, 104, 0.45, white)
        if index < len(flow) - 1:
            cv2.arrowedLine(
                image, (x + box_width + 5, 97), (x + box_width + 33, 97),
                amber, 2, tipLength=0.25,
            )
        x += box_width + 38

    track = dict(record.get("symbolic_track", {}))
    source_validation = dict(track.get("source_validation", {})).get(
        "validation_status", track.get("source_decision", "unknown")
    )
    facts = (
        f"class={track.get('object_class', 'unknown')}  "
        f"direction={track.get('direction', 'unknown')}  "
        f"persistence={_number(track.get('persistence')):.3f}  "
        f"confidence={_number(track.get('confidence')):.3f}  "
        f"source_validation={source_validation}"
    )
    _text(cv2, image, "1. SYMBOLIC TRACK", 28, 162, 0.58, cyan, 2)
    _text(cv2, image, facts, 48, 198, 0.50, white)

    interpretations = {
        str(row.get("pattern_id", "")): row
        for row in record.get("llm_residual_interpretation", [])
    }
    candidates = list(record.get("pattern_candidates", []))
    _text(
        cv2, image,
        "2-3. EVERY PATTERN: NUMERICAL RESIDUAL VECTOR + LLM INTERPRETATION",
        28, 248, 0.58, cyan, 2,
    )
    for heading, hx in (
        ("pattern", 48), ("plausibility", 285), ("residual sum", 470),
        ("largest residual components", 650), ("LLM conflicts / explanation", 1120),
    ):
        _text(cv2, image, heading, hx, 282, 0.43, muted)

    row_y = 318
    for candidate in candidates:
        pattern_id = str(candidate.get("pattern_id", "unknown"))
        residuals = dict(candidate.get("residual_vector", {}))
        interpretation = interpretations.get(pattern_id, {})
        plausibility = _number(interpretation.get("plausibility"))
        residual_sum = sum(_number(value) for value in residuals.values())
        largest = sorted(
            residuals.items(), key=lambda item: _number(item[1]), reverse=True
        )[:3]
        residual_text = " | ".join(
            f"{name}={_number(value):.3g}" for name, value in largest
        )
        conflicts = ", ".join(
            map(str, interpretation.get("structural_conflicts", []))
        ) or str(interpretation.get("explanation", "no conflict reported"))
        final = pattern_id == str(record.get("final_pattern", ""))
        cv2.rectangle(
            image, (32, row_y - 23), (1760, row_y + 17),
            (42, 46, 54) if final else (29, 32, 38), -1,
        )
        _text(
            cv2, image, ("* " if final else "  ") + pattern_id,
            48, row_y, 0.47, green if final else white, 2 if final else 1,
        )
        _text(
            cv2, image, f"{plausibility:.3f}", 305, row_y, 0.47,
            green if plausibility >= 0.5 else amber,
        )
        _text(cv2, image, f"{residual_sum:.3g}", 490, row_y, 0.47, white)
        _text(cv2, image, residual_text, 650, row_y, 0.42, white)
        _text(cv2, image, _fit_text(cv2, conflicts, 610), 1120, row_y, 0.40, muted)
        row_y += 43

    repairs = sorted(
        record.get("candidate_repairs", []),
        key=lambda row: _number(row.get("final_score"), -1e9), reverse=True,
    )
    y = max(790, row_y + 30)
    _text(
        cv2, image,
        "4-5. DETERMINISTIC REPAIR CANDIDATES + SYMBOLIC VALIDATION",
        28, y, 0.58, cyan, 2,
    )
    y += 38
    for heading, hx in (
        ("candidate", 48), ("decision", 400), ("score", 565),
        ("residual improvement", 690), ("retention", 955),
        ("new anomalies / modified frames", 1120),
    ):
        _text(cv2, image, heading, hx, y, 0.43, muted)
    y += 34

    selected_id = str(record.get("selected_candidate", {}).get("candidate_id", ""))
    shown = repairs[:10]
    for repair in shown:
        decision = str(repair.get("symbolic_verdict", repair.get("decision", "unknown")))
        candidate_id = str(repair.get("candidate_id", ""))
        selected = candidate_id == selected_id
        decision_color = green if decision in {"accept", "pass"} else amber if decision == "uncertain" else red
        cv2.rectangle(
            image, (32, y - 23), (1760, y + 17),
            (42, 55, 45) if selected else (29, 32, 38), -1,
        )
        _text(
            cv2, image, ("SELECTED  " if selected else "") + candidate_id,
            48, y, 0.43, green if selected else white, 2 if selected else 1,
        )
        _text(cv2, image, decision.upper(), 400, y, 0.43, decision_color, 2)
        _text(cv2, image, f"{_number(repair.get('final_score')):.3f}", 565, y, 0.43, white)
        _text(cv2, image, f"{_number(repair.get('residual_improvement')):+.3f}", 740, y, 0.43, white)
        _text(cv2, image, f"{_number(repair.get('observation_retention')):.3f}", 970, y, 0.43, white)
        details = (
            (", ".join(map(str, repair.get("new_anomalies", []))) or "none")
            + f" | frames={repair.get('modified_frame_ids', [])}"
        )
        _text(cv2, image, _fit_text(cv2, details, 620, 0.38), 1120, y, 0.38, muted)
        y += 42
    if len(repairs) > len(shown):
        _text(
            cv2, image,
            f"+ {len(repairs) - len(shown)} additional candidates in the per-track JSON",
            48, y, 0.40, muted,
        )
        y += 34

    y = min(1300, max(y + 18, 1240))
    cv2.rectangle(image, (28, y), (1770, 1472), (38, 44, 54), -1)
    _text(cv2, image, "6. FINAL RESULT", 48, y + 40, 0.62, (230, 135, 75), 2)
    repair_applied = bool(record.get("repair_applied", False))
    llm_preferred = str(record.get("LLM_preferred_pattern", "unknown"))
    validated = str(record.get("validated_pattern", record.get("final_pattern", "unknown")))
    mismatch = llm_preferred != validated
    selected_score = min((sum(_number(v) for v in row.get("post_repair_pattern_scores", {}).get(row.get("pattern_id"), {}).values()) for row in repairs if row.get("candidate_id") == selected_id), default=float("inf"))
    lower_rejected = any(row.get("symbolic_verdict") == "reject" and sum(_number(v) for v in row.get("post_repair_pattern_scores", {}).get(row.get("pattern_id"), {}).values()) < selected_score for row in repairs)
    result = (
        f"pattern={record.get('final_pattern', 'unknown')} | "
        f"repair_applied={repair_applied} | "
        f"validation={record.get('final_validation_status', 'unknown')} | "
        f"selected={selected_id or 'none'}"
    )
    _text(cv2, image, result, 48, y + 82, 0.58, green if repair_applied else amber, 2)
    alert = f"LLM preferred={llm_preferred} | validated={validated}"
    if lower_rejected:
        alert += " | LOWER-RESIDUAL CANDIDATE REJECTED BY HARD CONSTRAINTS"
    _text(cv2, image, alert, 48, y + 120, 0.46, red if mismatch or lower_rejected else green, 2)
    _text(cv2, image, f"reason: {record.get('final_selection_reason', '')}", 48, y + 150, 0.43, white)
    provenance = dict(record.get("provenance", {}))
    provenance_text = (
        f"provenance: LLM={provenance.get('llm_role', '')} | "
        f"repair={provenance.get('numeric_repair_role', '')} | "
        f"original preserved={provenance.get('original_observations_preserved', False)}"
    )
    _text(cv2, image, provenance_text, 900, y + 150, 0.38, muted)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        return False, "image_write_failed"
    return True, "rendered"


def render_trajectory_pattern_visualizations(state, output_root):
    """Render Step 8C sheets plus capped per-track Step 8B/8C MP4s."""
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError:
        return {
            **state,
            "trajectory_pattern_visualizations": [],
            "trajectory_pattern_video_summaries": [],
            "trajectory_pattern_track_videos": [],
            "trajectory_pattern_track_video_skipped": [],
            "trajectory_pattern_track_video_selections": [],
            "trajectory_pattern_visualization_skipped": [{"reason": "missing_cv2_or_numpy"}],
            "trajectory_pattern_visualization_output_root": None,
        }

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rendered = []
    skipped = []
    by_video = defaultdict(list)
    for record in state.get("trajectory_pattern_records", []):
        video_id = str(record.get("video_id", ""))
        track_id = int(record.get("track_id", -1))
        by_video[video_id].append(record)
        path = output_root / video_id / f"track_{track_id:04d}_pattern_process.png"
        success, status = _render_track(record, path)
        row = {"video_id": video_id, "track_id": track_id, "status": status}
        if success:
            row["visualization_path"] = str(path)
            rendered.append(row)
        else:
            skipped.append(row)

    summaries = []
    for video_id, records in sorted(by_video.items()):
        image = np.full((720, 1400, 3), (22, 24, 28), dtype=np.uint8)
        white = (242, 242, 242)
        green = (70, 220, 100)
        amber = (60, 190, 245)
        _text(cv2, image, f"STEP 8C VIDEO SUMMARY | {video_id}", 30, 50, 0.82, white, 2)
        repaired = sum(bool(row.get("repair_applied")) for row in records)
        disagreements = sum(str(row.get("LLM_preferred_pattern")) != str(row.get("validated_pattern")) for row in records)
        non_invalid = sum(
            str(row.get("final_validation_status", "")) != "invalid" for row in records
        )
        _text(
            cv2, image,
            f"tracks={len(records)}  repairs_applied={repaired}  final_non_invalid={non_invalid}  LLM_validated_disagreements={disagreements}",
            30, 95, 0.58, green,
        )
        counts = defaultdict(int)
        for record in records:
            counts[str(record.get("final_pattern", "unknown"))] += 1
        _text(cv2, image, "Final pattern distribution", 30, 150, 0.58, amber, 2)
        y = 190
        for pattern_id in PATTERNS:
            count = counts[pattern_id]
            bar_width = int(700 * count / max(1, len(records)))
            _text(cv2, image, pattern_id, 50, y, 0.46, white)
            cv2.rectangle(image, (270, y - 19), (270 + bar_width, y + 4), (70, 150, 220), -1)
            _text(cv2, image, count, 990, y, 0.46, white)
            y += 42
        promotion = dict(state.get("trajectory_pattern_statistics_promotion", {}))
        _text(
            cv2, image,
            f"statistics table: {promotion.get('decision', 'unknown')} | {promotion.get('reason', '')}",
            30, 650, 0.50, green if promotion.get("decision") == "accept" else red if promotion.get("reason") == "validation_regression" else amber, 2,
        )
        path = output_root / video_id / "video_pattern_summary.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(path), image):
            summaries.append({"video_id": video_id, "visualization_path": str(path)})

    track_video_manifest = render_step8bc_track_videos(
        state,
        output_root,
        fps=max(0.1, _number(state.get("step8bc_visualization_fps", 10.0), 10.0)),
        max_tracks_per_video=_MAX_TRACK_VIDEOS_PER_VIDEO,
    )
    manifest = {
        "version": 3,
        "num_track_images": len(rendered),
        "num_summary_images": len(summaries),
        "num_selected_track_videos": int(
            track_video_manifest.get("num_selected_tracks", 0)
        ),
        "num_track_videos": int(
            track_video_manifest.get("num_rendered_videos", 0)
        ),
        "num_skipped_track_videos": int(
            track_video_manifest.get("num_skipped_videos", 0)
        ),
        "max_track_videos_per_video": _MAX_TRACK_VIDEOS_PER_VIDEO,
        "num_skipped": len(skipped),
        "track_images": rendered,
        "summary_images": summaries,
        "track_videos": list(track_video_manifest.get("rendered", [])),
        "track_video_skipped": list(track_video_manifest.get("skipped", [])),
        "track_video_selections": list(
            track_video_manifest.get("selections", [])
        ),
        "skipped": skipped,
    }
    (output_root / "trajectory_pattern_visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        **state,
        "trajectory_pattern_visualizations": rendered,
        "trajectory_pattern_video_summaries": summaries,
        "trajectory_pattern_track_videos": list(
            track_video_manifest.get("rendered", [])
        ),
        "trajectory_pattern_track_video_skipped": list(
            track_video_manifest.get("skipped", [])
        ),
        "trajectory_pattern_track_video_selections": list(
            track_video_manifest.get("selections", [])
        ),
        "trajectory_pattern_track_video_manifest_path": str(
            track_video_manifest.get("manifest_path", "")
        ),
        "trajectory_pattern_visualization_skipped": skipped,
        "trajectory_pattern_visualization_output_root": output_root,
    }
