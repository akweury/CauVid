"""HTML and MP4 diagnostics for the Step 8B/8C trajectory pipeline."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


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
    signal_evidence = copy.deepcopy(
        dict(track.get("source_signal_evidence", {}))
    )
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
        "threshold_distances": (
            _threshold_distance_rows(track, validation) if validation else []
        ),
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
        "schema_version": 2,
        "video_id": str(record.get("video_id", "")),
        "track_id": int(record.get("track_id", -1)),
        "step8b_signal_evidence": signal_evidence,
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
    signal_evidence = copy.deepcopy(
        dict(payload.get("step8b_signal_evidence", {}))
    )
    if signal_evidence:
        return _flatten_display_scalars(
            {
                "evidence_type": "uncertain_signal_evidence",
                "observable_cues": dict(
                    signal_evidence.get("observable_cues", {})
                ),
            }
        )
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
    signal_evidence = dict(payload.get("step8b_signal_evidence", {}))
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
            (
                "8B observable cues: "
                + ", ".join(
                    f"{key}={_number(value):.2f}"
                    for key, value in dict(
                        signal_evidence.get("observable_cues", {})
                    ).items()
                )
            )
            if signal_evidence
            else (
                f"8B={validation.get('validation_status', validation.get('status', 'unknown'))} "
                f"decision={dict(step8b.get('track_facts', {})).get('source_decision', '')}"
            )
        )
        + (
            f" | 8C pattern={step8c.get('final_pattern', 'unknown')} "
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
        (
            "8B LOW-LEVEL SIGNAL EVIDENCE DESCRIPTORS AND CONFIDENCE"
            if signal_evidence
            else "8B ALL METRICS, THRESHOLDS, CHECK VALUES, AND SIGNED RULE DISTANCES"
        ),
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
    progress_callback=None,
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
            if progress_callback is not None:
                progress_callback(1)
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
    frame_counts_by_video = {}
    for video_id in set(pre_by_video) | set(final_by_video):
        frame_counts_by_video[video_id] = len(
            set(_video_frame_map(pre_by_video.get(video_id, {})))
            | set(_video_frame_map(final_by_video.get(video_id, {})))
        )
    total_expected_frames = sum(
        frame_counts_by_video.get(str(record.get("video_id", "")), 0)
        for record in selected_records
    )
    print(
        f"[step 8c][visualization] MP4_START "
        f"tracks={len(selected_records)} frames={total_expected_frames} "
        f"max_tracks_per_video={min(_MAX_TRACK_VIDEOS_PER_VIDEO, max(0, int(max_tracks_per_video)))} "
        f"fps={float(fps):.2f} output_root={output_root}",
        flush=True,
    )
    mp4_started = time.perf_counter()
    with tqdm(
        total=total_expected_frames,
        desc="[step 8c] 8B+8C MP4",
        unit="frame",
        dynamic_ncols=True,
    ) as frame_progress:
        for track_index, record in enumerate(selected_records, start=1):
            video_id = str(record.get("video_id", ""))
            track_id = int(record.get("track_id", -1))
            expected_frames = frame_counts_by_video.get(video_id, 0)
            track_progress = [0]

            def update_frames(count=1):
                increment = max(0, int(count))
                track_progress[0] += increment
                frame_progress.update(increment)

            frame_progress.set_postfix_str(
                f"track={track_index}/{len(selected_records)} "
                f"video={video_id} id={track_id}",
                refresh=True,
            )
            track_started = time.perf_counter()
            print(
                f"[step 8c][visualization] MP4_TRACK_START "
                f"track={track_index}/{len(selected_records)} "
                f"video={video_id} track_id={track_id} "
                f"frames={expected_frames}",
                flush=True,
            )
            track_root = output_root / video_id / f"track_{track_id:04d}"
            track_root.mkdir(parents=True, exist_ok=True)
            payload = build_step8bc_track_video_payload(record)
            metrics_path = (
                track_root / f"track_{track_id:04d}_8b_8c_metrics.json"
            )
            metrics_path.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
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
                    progress_callback=update_frames,
                )
            except Exception as exc:
                path = None
                status = (
                    f"render_failed:{type(exc).__name__}:"
                    f"{str(exc)[:240]}"
                )
            if track_progress[0] < expected_frames:
                frame_progress.update(expected_frames - track_progress[0])
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
            print(
                f"[step 8c][visualization] MP4_TRACK_DONE "
                f"track={track_index}/{len(selected_records)} "
                f"video={video_id} track_id={track_id} "
                f"status={status} encoded_frames={track_progress[0]} "
                f"latency={time.perf_counter() - track_started:.2f}s",
                flush=True,
            )
    print(
        f"[step 8c][visualization] MP4_DONE "
        f"rendered={len(rendered)} skipped={len(skipped)} "
        f"latency={time.perf_counter() - mp4_started:.2f}s",
        flush=True,
    )

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


_HTML_REPORT_STYLE = """
:root {
  color-scheme: dark;
  --bg: #11151b;
  --panel: #1b222c;
  --panel-2: #242d39;
  --line: #344050;
  --text: #edf2f7;
  --muted: #aab4c2;
  --good: #54d98c;
  --bad: #ff6b76;
  --warn: #ffc857;
  --blue: #62b5ff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { max-width: 1700px; margin: 0 auto; padding: 24px; }
h1 { margin: 0 0 6px; font-size: 24px; }
h2 { margin: 0 0 14px; font-size: 18px; }
a { color: var(--blue); }
.muted { color: var(--muted); }
.flow, .cards { display: flex; flex-wrap: wrap; gap: 9px; }
.flow { margin: 18px 0; counter-reset: stage; }
.flow li {
  list-style: none;
  counter-increment: stage;
  padding: 10px 13px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: var(--panel-2);
}
.flow li::before { content: counter(stage) ". "; color: var(--warn); }
.panel {
  margin: 14px 0;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 9px;
  background: var(--panel);
  overflow-x: auto;
}
.card {
  min-width: 145px;
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: var(--panel-2);
}
.card b { display: block; margin-top: 3px; font-size: 16px; }
.badge {
  display: inline-block;
  margin: 1px 3px 1px 0;
  padding: 2px 7px;
  border-radius: 10px;
  background: #313b48;
}
.good { color: var(--good); }
.bad { color: var(--bad); }
.warn { color: var(--warn); }
.selected { background: #21382e; }
.alert {
  margin: 10px 0;
  padding: 10px 12px;
  border-left: 4px solid var(--warn);
  background: #2d291a;
}
.alert.bad { border-color: var(--bad); background: #331d22; }
table { width: 100%; border-collapse: collapse; }
th, td {
  padding: 7px 8px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}
th { position: sticky; top: 0; background: var(--panel); color: var(--muted); }
code, pre { font: 12px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace; }
pre {
  margin: 8px 0 0;
  padding: 10px;
  border-radius: 6px;
  background: #10151b;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.bar-track {
  width: min(620px, 70vw);
  height: 14px;
  border-radius: 7px;
  background: #10151b;
  overflow: hidden;
}
.bar { height: 100%; min-width: 0; background: var(--blue); }
video { width: min(100%, 1200px); max-height: 72vh; background: #090b0e; }
@media (max-width: 760px) {
  main { padding: 14px; }
  th, td { padding: 6px; }
}
"""


def _html_text(value):
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return html.escape(str(value if value is not None else "-"), quote=True)


def _html_json(value):
    return html.escape(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        quote=True,
    )


def _status_class(value):
    normalized = str(value).strip().lower()
    if normalized in {
        "accept", "accepted", "completed", "keep", "pass", "passed",
        "rendered", "valid",
    }:
        return "good"
    if normalized in {
        "fail", "failed", "invalid", "reject", "rejected",
    }:
        return "bad"
    return "warn"


def _html_document(title, body):
    return (
        "<!doctype html>\n"
        "<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<meta http-equiv=\"Content-Security-Policy\" "
        "content=\"default-src 'none'; style-src 'unsafe-inline'; "
        "media-src 'self' file:; base-uri 'none'; form-action 'none'\">"
        f"<title>{_html_text(title)}</title><style>{_HTML_REPORT_STYLE}</style>"
        f"</head><body><main>{body}</main></body></html>\n"
    )


def _residual_total(vector):
    return sum(_number(value) for value in dict(vector).values())


def _render_track_html(record, output_path, media=None):
    video_id = str(record.get("video_id", ""))
    track_id = int(record.get("track_id", -1))
    track = dict(record.get("symbolic_track", {}))
    signal_evidence = dict(track.get("source_signal_evidence", {}))
    validation = dict(track.get("source_validation", {}))
    source_validation = validation.get(
        "validation_status", track.get("source_decision", "unknown")
    )
    final_pattern = str(record.get("final_pattern", "unknown"))
    selected = dict(record.get("selected_candidate", {}))
    selected_id = str(selected.get("candidate_id", ""))
    llm_preferred = str(record.get("LLM_preferred_pattern", "unknown"))
    validated = str(record.get("validated_pattern", final_pattern))
    interpretations = {
        str(row.get("pattern_id", "")): dict(row)
        for row in record.get("llm_residual_interpretation", [])
    }
    before_by_pattern = {
        str(row.get("pattern_id", "unknown")): dict(
            row.get("residual_vector", {})
        )
        for row in record.get("pattern_candidates", [])
    }
    final_by_pattern = {
        str(row.get("pattern_id", "unknown")): dict(
            row.get("residual_vector", {})
        )
        for row in record.get("final_pattern_candidates", [])
    }
    repairs = sorted(
        record.get("candidate_repairs", []),
        key=lambda row: _number(row.get("final_score"), -1e9),
        reverse=True,
    )

    selected_score = min(
        (
            _residual_total(
                dict(row.get("post_repair_pattern_scores", {})).get(
                    row.get("pattern_id"), {}
                )
            )
            for row in repairs
            if str(row.get("candidate_id", "")) == selected_id
        ),
        default=float("inf"),
    )
    lower_rejected = any(
        str(row.get("symbolic_verdict", "")) == "reject"
        and _residual_total(
            dict(row.get("post_repair_pattern_scores", {})).get(
                row.get("pattern_id"), {}
            )
        )
        < selected_score
        for row in repairs
    )

    pattern_rows = []
    interpretation_rows = []
    pattern_order = list(PATTERNS)
    pattern_order.extend(
        pattern_id
        for pattern_id in before_by_pattern
        if pattern_id not in PATTERNS
    )
    for pattern_id in pattern_order:
        before = before_by_pattern.get(pattern_id, {})
        final = final_by_pattern.get(pattern_id, before)
        interpretation = interpretations.get(pattern_id, {})
        plausibility = interpretation.get("plausibility")
        residual_cells = []
        for residual_id in RESIDUALS:
            before_value = before.get(residual_id)
            final_value = final.get(residual_id, before_value)
            residual_cells.append(
                f'<td data-residual="{_html_text(residual_id)}">'
                f"{_html_text(_display_value(before_value))}"
                " &rarr; "
                f"{_html_text(_display_value(final_value))}</td>"
            )
        conflicts = list(interpretation.get("structural_conflicts", []))
        explanation = interpretation.get(
            "explanation", "no interpretation reported"
        )
        pattern_rows.append(
            f'<tr data-pattern="{_html_text(pattern_id)}" '
            f'class="{"selected" if pattern_id == final_pattern else ""}">'
            f"<td><b>{'* ' if pattern_id == final_pattern else ''}"
            f"{_html_text(pattern_id)}</b></td>"
            f"<td>{_html_text(_display_value(plausibility))}</td>"
            f"<td>{_html_text(_display_value(_residual_total(before)))}"
            " &rarr; "
            f"{_html_text(_display_value(_residual_total(final)))}</td>"
            + "".join(residual_cells)
            + f"<td>{_html_text(conflicts or explanation)}</td></tr>"
        )
        interpretation_rows.append(
            f'<details data-pattern="{_html_text(pattern_id)}">'
            f"<summary>{_html_text(pattern_id)} — plausibility "
            f"{_html_text(_display_value(plausibility))}</summary>"
            f"<p><b>Structural conflicts:</b> "
            f"{_html_text(conflicts or 'none')}</p>"
            f"<p><b>Explanation:</b> {_html_text(explanation)}</p>"
            "</details>"
        )

    repair_rows = []
    for repair in repairs:
        candidate_id = str(repair.get("candidate_id", "unknown"))
        verdict = str(
            repair.get("symbolic_verdict", repair.get("decision", "unknown"))
        )
        repair_hypothesis = dict(repair.get("repair_hypothesis", {}))
        operation = repair.get(
            "repair_operation", repair_hypothesis.get("operation", "unknown")
        )
        score = repair.get("final_score")
        repair_rows.append(
            f'<tr data-candidate-id="{_html_text(candidate_id)}" '
            f'class="{"selected" if candidate_id == selected_id else ""}">'
            f"<td><b>{'SELECTED — ' if candidate_id == selected_id else ''}"
            f"{_html_text(candidate_id)}</b></td>"
            f"<td>{_html_text(repair.get('pattern_id', 'unknown'))}</td>"
            f"<td>{_html_text(operation)}</td>"
            f'<td class="{_status_class(verdict)}">{_html_text(verdict)}</td>'
            f"<td>{_html_text(_display_value(score))}</td>"
            f"<td>{_html_text(_display_value(repair.get('residual_improvement')))}</td>"
            f"<td>{_html_text(_display_value(repair.get('observation_retention')))}</td>"
            f"<td>{_html_text(repair.get('new_anomalies', []) or 'none')}</td>"
            f"<td>{_html_text(repair.get('modified_frame_ids', []))}</td>"
            f"<td>{_html_text(repair.get('final_selection_reason', ''))}</td>"
            "</tr>"
        )
    if not repair_rows:
        repair_rows.append(
            '<tr><td colspan="10" class="muted">No repair candidates</td></tr>'
        )

    hard_constraints = dict(selected.get("hard_constraint_results", {}))
    constraint_rows = "".join(
        "<tr>"
        f"<td>{_html_text(constraint_id)}</td>"
        f'<td class="{_status_class("pass" if passed else "fail")}">'
        f"{'PASS' if passed else 'FAIL'}</td></tr>"
        for constraint_id, passed in sorted(hard_constraints.items())
    ) or '<tr><td colspan="2" class="muted">No selected candidate</td></tr>'

    media_section = ""
    media = dict(media or {})
    if media.get("video_href"):
        media_section = (
            '<section class="panel" id="track-video"><h2>8B/8C track video</h2>'
            f'<video controls preload="metadata" src="{_html_text(media["video_href"])}">'
            "Your browser cannot play this MP4.</video>"
            f'<p><a href="{_html_text(media["video_href"])}">Download MP4</a>'
        )
        if media.get("metrics_href"):
            media_section += (
                f' · <a href="{_html_text(media["metrics_href"])}">'
                "Open complete metrics JSON</a>"
            )
        media_section += "</p></section>"
    elif media.get("metrics_href"):
        media_section = (
            '<section class="panel" id="track-video"><h2>8B/8C track video</h2>'
            '<p class="warn">MP4 encoding was skipped or failed. '
            f'<a href="{_html_text(media["metrics_href"])}">'
            "Open complete metrics JSON</a>.</p></section>"
        )
    else:
        media_section = (
            '<section class="panel" id="track-video"><h2>8B/8C track video</h2>'
            '<p class="muted">Not selected by the deterministic '
            f"{_MAX_TRACK_VIDEOS_PER_VIDEO}-track-per-video cap.</p></section>"
        )

    alert_rows = []
    if llm_preferred != validated:
        alert_rows.append(
            '<div class="alert bad">LLM preferred '
            f"<b>{_html_text(llm_preferred)}</b>, but symbolic validation "
            f"selected <b>{_html_text(validated)}</b>.</div>"
        )
    if lower_rejected:
        alert_rows.append(
            '<div class="alert bad">A lower-residual candidate was rejected '
            "by hard symbolic constraints.</div>"
        )

    residual_headers = "".join(
        f"<th>{_html_text(residual_id)}<br><span class=\"muted\">before → final</span></th>"
        for residual_id in RESIDUALS
    )
    if signal_evidence:
        source_evidence_html = (
            f'<div class="card">Step 8B observable cues<b>'
            f'{_html_text(signal_evidence.get("observable_cues", {}))}'
            "</b></div></div>"
            "<details><summary>Complete Step 8B six-cue evidence"
            "</summary>"
            f"<pre>{_html_json(signal_evidence)}</pre></details></section>"
        )
    else:
        source_evidence_html = (
            f'<div class="card">Source validation<b class="{_status_class(source_validation)}">'
            f"{_html_text(source_validation)}</b></div></div>"
            "<details><summary>Complete legacy Step 8B validation</summary>"
            f"<pre>{_html_json(validation)}</pre></details></section>"
        )
    body = (
        f"<h1>Step 8C trajectory pattern process</h1>"
        f'<p class="muted">Video {_html_text(video_id)} · track {track_id}</p>'
        '<ol class="flow"><li>symbolic abstraction</li>'
        "<li>all-pattern residuals</li><li>LLM interpretation</li>"
        "<li>multi-repair</li><li>symbolic validation</li>"
        "<li>final selection</li></ol>"
        + "".join(alert_rows)
        + '<section class="panel" id="symbolic-track"><h2>Symbolic track</h2>'
        '<div class="cards">'
        f'<div class="card">Class<b>{_html_text(track.get("object_class", "unknown"))}</b></div>'
        f'<div class="card">Direction<b>{_html_text(track.get("direction", "unknown"))}</b></div>'
        f'<div class="card">Persistence<b>{_html_text(_display_value(track.get("persistence")))}</b></div>'
        f'<div class="card">Confidence<b>{_html_text(_display_value(track.get("confidence")))}</b></div>'
        + source_evidence_html
        + media_section
        + '<section class="panel" id="pattern-residuals">'
        "<h2>All-pattern residual distances</h2>"
        '<p class="muted">Each cell shows before → final.</p><table><thead><tr>'
        "<th>Pattern</th><th>Plausibility</th><th>Residual sum</th>"
        + residual_headers
        + "<th>LLM conflicts / explanation</th></tr></thead><tbody>"
        + "".join(pattern_rows)
        + "</tbody></table></section>"
        + '<section class="panel" id="llm-interpretation">'
        "<h2>LLM interpretation</h2>"
        + "".join(interpretation_rows)
        + "</section>"
        + '<section class="panel" id="repair-candidates">'
        "<h2>Deterministic repair candidates</h2><table><thead><tr>"
        "<th>Candidate</th><th>Pattern</th><th>Operation</th><th>Verdict</th>"
        "<th>Score</th><th>Improvement</th><th>Retention</th>"
        "<th>New anomalies</th><th>Modified frames</th><th>Reason</th>"
        "</tr></thead><tbody>"
        + "".join(repair_rows)
        + "</tbody></table></section>"
        + '<section class="panel" id="symbolic-validation">'
        "<h2>Symbolic validation</h2><table><thead><tr>"
        "<th>Hard constraint</th><th>Result</th></tr></thead><tbody>"
        + constraint_rows
        + "</tbody></table></section>"
        + '<section class="panel" id="final-result"><h2>Final result</h2>'
        '<div class="cards">'
        f'<div class="card">Final pattern<b>{_html_text(final_pattern)}</b></div>'
        f'<div class="card">Repair applied<b>{_html_text(bool(record.get("repair_applied", False)))}</b></div>'
        f'<div class="card">Validation<b class="{_status_class(record.get("final_validation_status", "unknown"))}">'
        f'{_html_text(record.get("final_validation_status", "unknown"))}</b></div>'
        f'<div class="card">Selected candidate<b>{_html_text(selected_id or "none")}</b></div>'
        f'<div class="card">LLM preferred<b>{_html_text(llm_preferred)}</b></div>'
        f'<div class="card">Validated pattern<b>{_html_text(validated)}</b></div>'
        "</div><p><b>Reason:</b> "
        f'{_html_text(record.get("final_selection_reason", ""))}</p></section>'
        + '<section class="panel" id="provenance"><h2>Provenance</h2>'
        f'<pre>{_html_json(dict(record.get("provenance", {})))}</pre></section>'
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(
            _html_document(
                f"Step 8C pattern process — {video_id} / track {track_id}",
                body,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        return False, f"html_write_failed:{type(exc).__name__}:{str(exc)[:160]}"
    return True, "rendered"


def _render_video_summary_html(
    video_id,
    video_records,
    promotion,
    output_path,
    track_media=None,
):
    repaired = sum(bool(row.get("repair_applied")) for row in video_records)
    disagreements = sum(
        str(row.get("LLM_preferred_pattern"))
        != str(row.get("validated_pattern"))
        for row in video_records
    )
    non_invalid = sum(
        str(row.get("final_validation_status", "")) != "invalid"
        for row in video_records
    )
    counts = defaultdict(int)
    for record in video_records:
        counts[str(record.get("final_pattern", "unknown"))] += 1
    total = max(1, len(video_records))
    distribution_rows = "".join(
        f'<tr data-pattern="{_html_text(pattern_id)}">'
        f"<td>{_html_text(pattern_id)}</td><td>{counts[pattern_id]}</td>"
        '<td><div class="bar-track"><div class="bar" '
        f'style="width:{100.0 * counts[pattern_id] / total:.3f}%"></div>'
        "</div></td></tr>"
        for pattern_id in PATTERNS
    )

    track_media = dict(track_media or {})
    track_rows = []
    for record in sorted(
        video_records, key=lambda row: int(row.get("track_id", -1))
    ):
        track_id = int(record.get("track_id", -1))
        report_href = f"track_{track_id:04d}_pattern_process.html"
        media = dict(track_media.get((video_id, track_id), {}))
        media_link = (
            f'<a href="{_html_text(media["video_href"])}">MP4</a>'
            if media.get("video_href")
            else '<span class="muted">not rendered</span>'
        )
        track_rows.append(
            f'<tr data-track-id="{track_id}"><td>{track_id}</td>'
            f'<td><a href="{report_href}">pattern process</a></td>'
            f"<td>{_html_text(record.get('final_pattern', 'unknown'))}</td>"
            f'<td class="{_status_class(record.get("final_validation_status", "unknown"))}">'
            f'{_html_text(record.get("final_validation_status", "unknown"))}</td>'
            f"<td>{_html_text(bool(record.get('repair_applied', False)))}</td>"
            f"<td>{media_link}</td></tr>"
        )

    decision = str(promotion.get("decision", "unknown"))
    body = (
        f"<h1>Step 8C video pattern summary</h1>"
        f'<p class="muted">Video {_html_text(video_id)}</p>'
        '<section class="panel"><div class="cards">'
        f'<div class="card">Tracks<b>{len(video_records)}</b></div>'
        f'<div class="card">Repairs applied<b>{repaired}</b></div>'
        f'<div class="card">Final non-invalid<b>{non_invalid}</b></div>'
        f'<div class="card">LLM/validated disagreements<b>{disagreements}</b></div>'
        "</div></section>"
        '<section class="panel" id="pattern-distribution">'
        "<h2>Final pattern distribution</h2><table><thead><tr>"
        "<th>Pattern</th><th>Count</th><th>Share</th>"
        "</tr></thead><tbody>"
        + distribution_rows
        + "</tbody></table></section>"
        + '<section class="panel" id="tracks"><h2>Track reports</h2>'
        "<table><thead><tr><th>Track</th><th>HTML report</th>"
        "<th>Final pattern</th><th>Validation</th><th>Repair</th>"
        "<th>Video</th></tr></thead><tbody>"
        + "".join(track_rows)
        + "</tbody></table></section>"
        + '<section class="panel" id="statistics-promotion">'
        "<h2>Statistics promotion</h2>"
        f'<p><span class="badge {_status_class(decision)}">'
        f"{_html_text(decision)}</span> "
        f'{_html_text(promotion.get("reason", ""))}</p>'
        f"<pre>{_html_json(promotion)}</pre></section>"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(
            _html_document(f"Step 8C summary — {video_id}", body),
            encoding="utf-8",
        )
    except OSError as exc:
        return False, f"html_write_failed:{type(exc).__name__}:{str(exc)[:160]}"
    return True, "rendered"


def render_trajectory_pattern_visualizations(state, output_root):
    """Write Step 8C HTML reports plus capped per-track Step 8B/8C MP4s."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visualization_started = time.perf_counter()
    records = list(state.get("trajectory_pattern_records", []))
    print(
        f"[step 8c][visualization] START tracks={len(records)} "
        f"output_root={output_root}",
        flush=True,
    )

    by_video = defaultdict(list)
    for record in records:
        by_video[str(record.get("video_id", ""))].append(record)

    track_video_manifest = render_step8bc_track_videos(
        state,
        output_root,
        fps=max(
            0.1,
            _number(state.get("step8bc_visualization_fps", 10.0), 10.0),
        ),
        max_tracks_per_video=_MAX_TRACK_VIDEOS_PER_VIDEO,
    )
    track_media = {}
    for row in (
        list(track_video_manifest.get("rendered", []))
        + list(track_video_manifest.get("skipped", []))
    ):
        video_id = str(row.get("video_id", ""))
        track_id = int(row.get("track_id", -1))
        track_dir = f"track_{track_id:04d}"
        media = {
            "metrics_href": (
                f"{track_dir}/track_{track_id:04d}_8b_8c_metrics.json"
            ),
            "status": str(row.get("status", "unknown")),
        }
        if row.get("visualization_path"):
            media["video_href"] = (
                f"{track_dir}/track_{track_id:04d}_8b_8c.mp4"
            )
        track_media[(video_id, track_id)] = media

    rendered = []
    skipped = []
    report_started = time.perf_counter()
    for record in tqdm(
        records,
        desc="[step 8c] HTML track reports",
        unit="track",
        dynamic_ncols=True,
    ):
        video_id = str(record.get("video_id", ""))
        track_id = int(record.get("track_id", -1))
        path = (
            output_root
            / video_id
            / f"track_{track_id:04d}_pattern_process.html"
        )
        success, status = _render_track_html(
            record,
            path,
            media=track_media.get((video_id, track_id)),
        )
        row = {
            "video_id": video_id,
            "track_id": track_id,
            "status": status,
            "media_type": "text/html",
        }
        if success:
            row["visualization_path"] = str(path)
            row["report_path"] = str(path)
            rendered.append(row)
        else:
            skipped.append(row)
    print(
        f"[step 8c][visualization] HTML_REPORTS_DONE "
        f"rendered={len(rendered)} skipped={len(skipped)} "
        f"latency={time.perf_counter() - report_started:.2f}s",
        flush=True,
    )

    summaries = []
    summary_skipped = []
    summary_started = time.perf_counter()
    for video_id, video_records in tqdm(
        sorted(by_video.items()),
        desc="[step 8c] HTML video summaries",
        unit="video",
        dynamic_ncols=True,
    ):
        path = output_root / video_id / "video_pattern_summary.html"
        success, status = _render_video_summary_html(
            video_id,
            video_records,
            dict(state.get("trajectory_pattern_statistics_promotion", {})),
            path,
            track_media=track_media,
        )
        row = {
            "video_id": video_id,
            "status": status,
            "media_type": "text/html",
        }
        if success:
            row["visualization_path"] = str(path)
            row["report_path"] = str(path)
            summaries.append(row)
        else:
            summary_skipped.append(row)
    print(
        f"[step 8c][visualization] HTML_SUMMARIES_DONE "
        f"rendered={len(summaries)} skipped={len(summary_skipped)} "
        f"latency={time.perf_counter() - summary_started:.2f}s",
        flush=True,
    )

    manifest = {
        "version": 4,
        "report_format": "html",
        "self_contained_reports": True,
        "num_track_reports": len(rendered),
        "num_summary_reports": len(summaries),
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
        "num_skipped": len(skipped) + len(summary_skipped),
        "track_reports": rendered,
        "summary_reports": summaries,
        "track_videos": list(track_video_manifest.get("rendered", [])),
        "track_video_skipped": list(track_video_manifest.get("skipped", [])),
        "track_video_selections": list(
            track_video_manifest.get("selections", [])
        ),
        "skipped": skipped + summary_skipped,
    }
    (output_root / "trajectory_pattern_visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"[step 8c][visualization] DONE "
        f"track_html_reports={len(rendered)} "
        f"summary_html_reports={len(summaries)} "
        f"track_videos={len(track_video_manifest.get('rendered', []))} "
        f"track_video_skipped={len(track_video_manifest.get('skipped', []))} "
        f"latency={time.perf_counter() - visualization_started:.2f}s "
        f"manifest={output_root / 'trajectory_pattern_visualization_manifest.json'}",
        flush=True,
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
        "trajectory_pattern_visualization_skipped": skipped + summary_skipped,
        "trajectory_pattern_visualization_output_root": output_root,
    }
