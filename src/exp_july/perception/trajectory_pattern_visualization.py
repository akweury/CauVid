"""HTML and MP4 diagnostics for the Step 8B/8C trajectory pipeline."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import time
from collections import Counter, defaultdict
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
_TRACK_VIDEO_SELECTION_NAMESPACE = "step8bc-global-track-video-five-v2"
_MAX_TRACK_VIDEOS_TOTAL = 5
_OUTPUT_WIDTH = 1920
_OUTPUT_HEIGHT = 1440
_LEFT_SCENE_WIDTH = 1100


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


def select_deterministic_track_records(records, max_tracks_per_video=5):
    """Select at most five stable track records across the complete run."""
    del max_tracks_per_video  # Retained for compatibility; global budget is fixed.
    unique = {}
    for record in records:
        video_id = str(record.get("video_id", ""))
        try:
            track_id = int(record.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if video_id and track_id >= 0:
            unique.setdefault((video_id, track_id), record)
    ranked_keys = sorted(
        unique,
        key=lambda key: (_stable_track_rank(key[0], key[1]), key[0], key[1]),
    )
    return [unique[key] for key in ranked_keys[:_MAX_TRACK_VIDEOS_TOTAL]]


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
        "schema_version": 3,
        "video_id": str(record.get("video_id", "")),
        "track_id": int(record.get("track_id", -1)),
        "step8b_signal_evidence": signal_evidence,
        "step8b_metrics": step8b_metrics,
        "step8c_residual_distances": step8c_residual_distances,
        "step8c": {
            "trajectory_cohort_id": str(
                record.get("trajectory_cohort_id", "unknown")
            ),
            "activated_rule": copy.deepcopy(
                dict(record.get("activated_rule", {}))
            ),
            "cohort_static_metadata": copy.deepcopy(
                dict(record.get("cohort_static_metadata", {}))
            ),
            "cohort_statistical_summary": copy.deepcopy(
                dict(record.get("cohort_statistical_summary", {}))
            ),
            "cohort_operator_plan": copy.deepcopy(
                dict(record.get("cohort_operator_plan", {}))
            ),
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
            "resolution_status": str(
                record.get("resolution_status", "unknown")
            ),
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
    """Build the concise right-side Step 8C cohort and repair panel."""
    panel = np.full((height, width, 3), (20, 23, 29), dtype=np.uint8)
    step8c = dict(payload.get("step8c", {}))
    rule = dict(step8c.get("activated_rule", {}))
    cohort_summary = dict(step8c.get("cohort_statistical_summary", {}))
    plan = dict(step8c.get("cohort_operator_plan", {}))
    calibration = dict(plan.get("calibration", {}))
    selected_measurement = dict(
        calibration.get("selected_measurement", {})
    )
    margin = 38

    _text(
        cv2,
        panel,
        "STEP 8C",
        margin,
        74,
        1.80,
        (250, 250, 250),
        4,
    )
    _text(
        cv2,
        panel,
        "STATISTICAL REPAIR",
        margin,
        137,
        1.35,
        (215, 222, 232),
        3,
    )
    object_class = str(
        dict(payload.get("step8b_metrics", {}))
        .get("track_facts", {})
        .get("object_class", "unknown")
    )
    _text(
        cv2,
        panel,
        (
            f"[8B] track {payload.get('track_id', -1)} | class {object_class}"
        ),
        margin,
        204,
        1.22,
        (242, 242, 242),
        3,
    )
    cv2.line(
        panel,
        (margin, 248),
        (width - margin, 248),
        (75, 82, 94),
        3,
        cv2.LINE_AA,
    )

    _text(
        cv2,
        panel,
        "SEMANTIC COHORT  [8C]",
        margin,
        306,
        1.08,
        (225, 190, 70),
        3,
    )
    _text(
        cv2,
        panel,
        _clip_text_to_width(
            cv2,
            step8c.get("trajectory_cohort_id", "unknown"),
            width - 2 * margin,
            1.32,
            3,
        ),
        margin,
        363,
        1.32,
        (245, 245, 245),
        3,
    )
    _text(
        cv2,
        panel,
        _clip_text_to_width(
            cv2,
            f"rule: {rule.get('rule_id', 'unknown')}",
            width - 2 * margin,
            0.92,
            2,
        ),
        margin,
        415,
        0.92,
        (205, 215, 228),
        2,
    )
    _text(
        cv2,
        panel,
        _clip_text_to_width(
            cv2,
            rule.get("description", ""),
            width - 2 * margin,
            0.78,
            2,
        ),
        margin,
        458,
        0.78,
        (175, 185, 198),
        2,
    )

    _text(
        cv2,
        panel,
        "DETERMINISTIC OPERATOR  [8C]",
        margin,
        535,
        1.08,
        (225, 190, 70),
        3,
    )
    operator = str(plan.get("operator", "no_repair"))
    operator_color = (
        (70, 220, 100) if operator != "no_repair" else (80, 215, 240)
    )
    _text(
        cv2,
        panel,
        operator.upper(),
        margin,
        596,
        1.38,
        operator_color,
        4,
    )
    requested = str(plan.get("llm_requested_operator", operator))
    _text(
        cv2,
        panel,
        _clip_text_to_width(
            cv2,
            f"LLM proposed: {requested}",
            width - 2 * margin,
            0.80,
            2,
        ),
        margin,
        641,
        0.80,
        (205, 215, 228),
        2,
    )
    parameters = dict(plan.get("calibrated_parameters", {}))
    parameter_text = (
        ", ".join(f"{key}={_display_value(value)}" for key, value in parameters.items())
        or "none"
    )
    _text(
        cv2,
        panel,
        _clip_text_to_width(
            cv2,
            f"parameters: {parameter_text}",
            width - 2 * margin,
            0.76,
            2,
        ),
        margin,
        684,
        0.76,
        (205, 215, 228),
        2,
    )

    _text(
        cv2,
        panel,
        "STATISTICAL VALIDATION  [8C]",
        margin,
        765,
        1.08,
        (225, 190, 70),
        3,
    )
    anomalies = list(cohort_summary.get("systematic_anomalies", []))
    _text(
        cv2,
        panel,
        _clip_text_to_width(
            cv2,
            "anomalies: " + (", ".join(map(str, anomalies)) or "none"),
            width - 2 * margin,
            0.78,
            2,
        ),
        margin,
        816,
        0.78,
        (225, 228, 234),
        2,
    )
    _text(
        cv2,
        panel,
        (
            f"cohort tracks: {int(cohort_summary.get('track_count', 0))}    "
            f"validation samples: {int(selected_measurement.get('sample_count', 0))}"
        ),
        margin,
        860,
        0.76,
        (205, 215, 228),
        2,
    )
    _text(
        cv2,
        panel,
        (
            f"success: {_number(selected_measurement.get('success_rate')):.3f}    "
            f"issue gain: {_number(selected_measurement.get('mean_issue_cost_improvement')):+.3f}"
        ),
        margin,
        903,
        0.76,
        (205, 215, 228),
        2,
    )
    _text(
        cv2,
        panel,
        _clip_text_to_width(
            cv2,
            f"decision: {calibration.get('promotion_decision', 'unknown')}",
            width - 2 * margin,
            0.82,
            2,
        ),
        margin,
        949,
        0.82,
        operator_color,
        2,
    )

    _text(
        cv2,
        panel,
        "FINAL OUTCOME  [8C]",
        margin,
        1042,
        1.08,
        (225, 190, 70),
        3,
    )
    status = str(step8c.get("resolution_status", "unknown"))
    status_color = (
        (70, 220, 100)
        if status.startswith("validated")
        else (60, 190, 245)
    )
    _text(
        cv2,
        panel,
        status.upper(),
        margin,
        1103,
        1.20,
        status_color,
        3,
    )
    _text(
        cv2,
        panel,
        (
            f"repair applied: {bool(step8c.get('repair_applied', False))}    "
            f"validation: {step8c.get('final_validation_status', 'unknown')}"
        ),
        margin,
        1150,
        0.78,
        (220, 225, 232),
        2,
    )
    reason = str(step8c.get("final_selection_reason", ""))
    _text(
        cv2,
        panel,
        _clip_text_to_width(
            cv2, reason, width - 2 * margin, 0.72, 2
        ),
        margin,
        1194,
        0.72,
        (185, 195, 208),
        2,
    )
    fingerprint = str(
        dict(step8c.get("provenance", {})).get(
            "cohort_policy_fingerprint", ""
        )
    )
    _text(
        cv2,
        panel,
        f"policy: {fingerprint or 'not available'}",
        margin,
        1250,
        0.72,
        (165, 175, 188),
        2,
    )
    _text(
        cv2,
        panel,
        "Full cohort statistics and repair provenance are in the JSON.",
        margin,
        height - 35,
        0.64,
        (165, 175, 188),
        2,
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
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
    if not label:
        return
    text_y = max(24, y1 - 8 - text_y_offset)
    cv2.rectangle(
        image,
        (x1, max(0, text_y - 18)),
        (min(image_width - 1, x1 + 230), min(image_height - 1, text_y + 5)),
        (0, 0, 0),
        -1,
    )
    _text(cv2, image, label, x1 + 3, text_y, 0.43, color, 1)


def _draw_scaled_track_path(
    cv2,
    image,
    track,
    frame_indices,
    current_frame,
    source_width,
    source_height,
    color,
):
    """Overlay the observed bbox-center path up to the current frame."""
    image_height, image_width = image.shape[:2]
    scale_x = image_width / max(1.0, float(source_width))
    scale_y = image_height / max(1.0, float(source_height))
    points = []
    for frame_index in frame_indices:
        if int(frame_index) > int(current_frame):
            break
        obj = track.get(frame_index)
        if not obj:
            continue
        box = _valid_box(obj.get("bbox", obj.get("box", [])))
        if box is None:
            continue
        center_x = int(round((box[0] + box[2]) * 0.5 * scale_x))
        center_y = int(round((box[1] + box[3]) * 0.5 * scale_y))
        points.append(
            (
                max(0, min(image_width - 1, center_x)),
                max(0, min(image_height - 1, center_y)),
            )
        )
    points = points[-80:]
    for start, end in zip(points, points[1:]):
        cv2.line(image, start, end, color, 3, cv2.LINE_AA)
    if points:
        cv2.circle(image, points[-1], 6, color, -1, cv2.LINE_AA)


def _signal_values(obj):
    if not obj:
        return None
    position = list(
        obj.get("position_3d", obj.get("relative_position_3d", []))
    )
    x_value = _number(position[0]) if len(position) >= 3 else 0.0
    z_value = _number(position[2]) if len(position) >= 3 else 0.0
    vx_value = _number(obj.get("rel_vx"))
    vz_value = _number(obj.get("rel_vz"))
    speed_value = _number(
        obj.get("rel_speed", math.hypot(vx_value, vz_value))
    )
    return (x_value, z_value, vx_value, vz_value, speed_value)


def _cue_visual_state(name, raw_value, object_observed):
    """Return cue text/style; absent objects can never activate a cue."""
    try:
        cue_value = float(raw_value)
        cue_available = math.isfinite(cue_value)
    except (TypeError, ValueError):
        cue_value = 0.0
        cue_available = False
    if not cue_available:
        return f"{name}=N/A", (70, 90, 235), 2, False
    if object_observed and cue_value > 0.0:
        return f"{name}={cue_value:.2f}", (70, 220, 100), 2, True
    return f"{name}={cue_value:.2f}", (145, 152, 163), 1, False


def _bbox_difference_metrics(pre_obj, final_obj):
    if not pre_obj or not final_obj:
        return None
    before = _valid_box(pre_obj.get("bbox", pre_obj.get("box", [])))
    after = _valid_box(final_obj.get("bbox", final_obj.get("box", [])))
    if before is None or after is None:
        return None
    before_center = ((before[0] + before[2]) * 0.5, (before[1] + before[3]) * 0.5)
    after_center = ((after[0] + after[2]) * 0.5, (after[1] + after[3]) * 0.5)
    center_shift = math.hypot(
        after_center[0] - before_center[0],
        after_center[1] - before_center[1],
    )
    intersection_width = max(0.0, min(before[2], after[2]) - max(before[0], after[0]))
    intersection_height = max(0.0, min(before[3], after[3]) - max(before[1], after[1]))
    intersection = intersection_width * intersection_height
    before_area = (before[2] - before[0]) * (before[3] - before[1])
    after_area = (after[2] - after[0]) * (after[3] - after[1])
    union = before_area + after_area - intersection
    return {
        "center_shift_px": center_shift,
        "iou": intersection / union if union > 0.0 else 0.0,
    }


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


def _ego_speed_series(ego_video, frame_indices):
    """Return frame-aligned ego vx/vz values using the best available fields."""
    frames = {
        int(frame.get("frame_index", index)): dict(frame)
        for index, frame in enumerate(dict(ego_video or {}).get("frames", []))
    }

    def value(frame, names):
        for name in names:
            if name not in frame:
                continue
            try:
                result = float(frame[name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(result):
                return result
        return None

    vx_names = ("refined_ego_vx", "ego_vx_smoothed", "ego_vx")
    vz_names = ("refined_ego_vz", "ego_vz_smoothed", "ego_vz")
    return {
        "vx": [value(frames.get(int(frame_index), {}), vx_names) for frame_index in frame_indices],
        "vz": [value(frames.get(int(frame_index), {}), vz_names) for frame_index in frame_indices],
    }


def _track_motion_series(track, frame_indices):
    """Return aligned object and relative velocities for one rendered track."""
    track = dict(track or {})

    def value(obj, names):
        if not obj:
            return None
        motion = dict(obj.get("motion", {}))
        for source in (obj, motion):
            for name in names:
                if name not in source:
                    continue
                try:
                    result = float(source[name])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(result):
                    return result
        return None

    indices = list(frame_indices)
    return {
        "obj_vx": [value(track.get(int(index)), ("obj_vx", "object_vx")) for index in indices],
        "obj_vz": [value(track.get(int(index)), ("obj_vz", "object_vz")) for index in indices],
        "rel_vx": [value(track.get(int(index)), ("rel_vx",)) for index in indices],
        "rel_vz": [value(track.get(int(index)), ("rel_vz",)) for index in indices],
    }


def _draw_bright_zero_baseline(cv2, canvas, left, right, y):
    """Draw a high-contrast dashed zero reference without masking the signal."""
    dash_length = 12
    gap_length = 7
    cursor = int(left)
    while cursor <= int(right):
        segment_right = min(int(right), cursor + dash_length)
        cv2.line(
            canvas,
            (cursor, int(y)),
            (segment_right, int(y)),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cursor = segment_right + gap_length


def _draw_motion_speed_charts(
    cv2,
    canvas,
    ego_video,
    object_track,
    frame_indices,
    current_frame,
    *,
    top,
    left,
    right,
    height,
):
    """Draw a synchronized 3x2 ego/object/relative vx-vz chart grid."""
    indices = list(sorted(frame_indices))
    if not indices:
        return
    column_gap = 14
    row_gap = 8
    total_width = max(2, int(right) - int(left))
    panel_width = max(1, (total_width - column_gap) // 2)
    row_height = max(62, (int(height) - row_gap * 2) // 3)
    ego = _ego_speed_series(ego_video, indices)
    track = _track_motion_series(object_track, indices)
    series = {
        "ego_vx": ego["vx"],
        "ego_vz": ego["vz"],
        **track,
    }
    specifications = (
        (0, 0, "ego_vx", "EGO VX [7]", (80, 215, 240)),
        (0, 1, "ego_vz", "EGO VZ [7]", (80, 220, 130)),
        (1, 0, "obj_vx", "OBJ VX [8C]", (245, 180, 70)),
        (1, 1, "obj_vz", "OBJ VZ [8C]", (210, 125, 245)),
        (2, 0, "rel_vx", "REL VX [8A/8C]", (70, 220, 100)),
        (2, 1, "rel_vz", "REL VZ [8A/8C]", (245, 125, 90)),
    )
    try:
        current_offset = indices.index(current_frame)
    except ValueError:
        current_offset = 0

    for row, column, series_id, title, color in specifications:
        panel_left = int(left) + column * (panel_width + column_gap)
        panel_right = panel_left + panel_width
        panel_top = int(top) + row * (row_height + row_gap)
        panel_bottom = panel_top + row_height
        cv2.rectangle(canvas, (panel_left, panel_top), (panel_right, panel_bottom), (55, 62, 74), 2)
        _text(cv2, canvas, title, panel_left + 8, panel_top + 19, 0.43, (220, 225, 232), 2)
        values = series[series_id]
        current_value = values[current_offset] if current_offset < len(values) else None
        value_text = f"{current_value:+.2f} m/s" if current_value is not None else "N/A"
        _text(
            cv2,
            canvas,
            value_text,
            max(panel_left + 120, panel_right - 112),
            panel_top + 19,
            0.39,
            color if current_value is not None else (145, 152, 163),
            1,
        )
        plot_left = panel_left + 38
        plot_right = panel_right - 10
        plot_top = panel_top + 27
        plot_bottom = panel_bottom - 14
        valid_values = [value for value in values if value is not None]

        def x_position(offset):
            return plot_left + int(round(offset * (plot_right - plot_left) / max(1, len(indices) - 1)))

        if valid_values:
            minimum = min(valid_values + [0.0])
            maximum = max(valid_values + [0.0])
            padding = max(0.05, (maximum - minimum) * 0.12)
            minimum -= padding
            maximum += padding

            def point(offset, value):
                y = plot_bottom - int(round((value - minimum) * (plot_bottom - plot_top) / max(1e-9, maximum - minimum)))
                return x_position(offset), y

            zero_y = point(0, 0.0)[1]
            _draw_bright_zero_baseline(cv2, canvas, plot_left, plot_right, zero_y)
            previous = None
            previous_offset = None
            for offset, value_row in enumerate(values):
                if value_row is None:
                    previous = None
                    previous_offset = None
                    continue
                current_point = point(offset, value_row)
                if previous is not None and previous_offset == offset - 1:
                    cv2.line(canvas, previous, current_point, color, 2, cv2.LINE_AA)
                previous = current_point
                previous_offset = offset
        else:
            _text(cv2, canvas, "no samples", plot_left + 12, plot_top + 22, 0.39, (145, 152, 163), 1)

        marker_x = x_position(current_offset)
        cv2.line(canvas, (marker_x, plot_top), (marker_x, plot_bottom), (255, 255, 255), 2, cv2.LINE_AA)
        if current_value is not None and valid_values:
            cv2.circle(canvas, point(current_offset, current_value), 4, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_step8c_track_progress(
    cv2,
    canvas,
    frame_indices,
    current_frame,
    pre_track,
    final_track,
    modified_frames,
    *,
    top,
    left,
    right,
):
    """Draw Step 8C track presence directly below the scene."""
    indices = sorted(frame_indices)
    if not indices:
        return
    left = max(0, int(left))
    right = max(left + 1, int(right))
    bar_top = int(top) + 47
    bar_bottom = int(top) + 83
    bar_width = right - left
    _text(
        cv2,
        canvas,
        "TRACK PRESENCE [8B -> 8C]",
        left,
        int(top) + 31,
        1.15,
        (225, 230, 238),
        3,
    )
    cv2.rectangle(
        canvas, (left, bar_top), (right, bar_bottom), (54, 58, 66), -1
    )
    for offset, frame_index in enumerate(indices):
        x1 = left + int(math.floor(offset * bar_width / len(indices)))
        x2 = left + int(
            math.ceil((offset + 1) * bar_width / len(indices))
        )
        pre_present = frame_index in pre_track
        final_present = frame_index in final_track
        if frame_index in modified_frames or (
            final_present and not pre_present
        ):
            color = (70, 220, 100)
        elif pre_present:
            color = (40, 185, 245)
        else:
            color = (72, 76, 84)
        cv2.rectangle(
            canvas,
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
        round((current_offset + 0.5) * bar_width / len(indices))
    )
    cv2.line(
        canvas,
        (marker_x, bar_top - 4),
        (marker_x, bar_bottom + 4),
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    _text(
        cv2,
        canvas,
        f"frame {current_offset + 1}/{len(indices)}",
        max(left, right - 185),
        int(top) + 29,
        0.78,
        (255, 255, 255),
        3,
    )


def _render_step8bc_track_video(
    *,
    record,
    payload,
    pre_pattern_video,
    final_video,
    ego_video,
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

    canvas_width = _OUTPUT_WIDTH
    total_height = _OUTPUT_HEIGHT
    left_width = _LEFT_SCENE_WIDTH
    panel_width = canvas_width - left_width
    max_scene_width = left_width - 40
    max_scene_height = 620
    pre_track = _track_objects_by_frame(
        pre_pattern_video, int(record.get("track_id", -1))
    )
    final_track = _track_objects_by_frame(
        final_video, int(record.get("track_id", -1))
    )
    step8c_payload = dict(payload.get("step8c", {}))
    selected_candidate = dict(
        step8c_payload.get("selected_candidate", {})
    )
    modified_frames = {
        int(value)
        for value in selected_candidate.get("modified_frame_ids", [])
    }
    repair_applied = bool(step8c_payload.get("repair_applied"))
    static_panel = _build_step8bc_static_panel(
        cv2, np, payload, panel_width, total_height
    )
    observable_cues = dict(
        dict(payload.get("step8b_signal_evidence", {})).get(
            "observable_cues", {}
        )
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
            scale = min(
                max_scene_width / max(1, source_frame_width),
                max_scene_height / max(1, source_frame_height),
            )
            scene_width = max(2, int(round(source_frame_width * scale)))
            scene_height = max(2, int(round(source_frame_height * scale)))
            if scene_width % 2:
                scene_width -= 1
            if scene_height % 2:
                scene_height -= 1
            scene = cv2.resize(image, (scene_width, scene_height))
            pre_obj = pre_track.get(frame_index)
            final_obj = final_track.get(frame_index)
            _draw_scaled_track_path(
                cv2,
                scene,
                pre_track,
                frame_indices,
                frame_index,
                source_frame_width,
                source_frame_height,
                (40, 185, 245),
            )
            if repair_applied:
                _draw_scaled_track_path(
                    cv2,
                    scene,
                    final_track,
                    frame_indices,
                    frame_index,
                    source_frame_width,
                    source_frame_height,
                    (70, 220, 100),
                )
            _draw_scaled_box(
                cv2,
                scene,
                pre_obj,
                source_frame_width,
                source_frame_height,
                (40, 185, 245),
                "",
            )
            draw_repaired_box = bool(
                repair_applied
                and (
                    not modified_frames
                    or frame_index in modified_frames
                    or (final_obj is not None and pre_obj is None)
                )
            )
            if draw_repaired_box:
                _draw_scaled_box(
                    cv2,
                    scene,
                    final_obj,
                    source_frame_width,
                    source_frame_height,
                    (70, 220, 100),
                    "",
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
            motion_chart_top = scene_y + scene_height + 10
            motion_chart_height = 286
            _draw_motion_speed_charts(
                cv2,
                canvas,
                ego_video,
                final_track,
                frame_indices,
                frame_index,
                top=motion_chart_top,
                left=scene_x,
                right=scene_x + scene_width,
                height=motion_chart_height,
            )
            progress_top = motion_chart_top + motion_chart_height + 8
            _draw_step8c_track_progress(
                cv2,
                canvas,
                frame_indices,
                frame_index,
                pre_track,
                final_track,
                modified_frames,
                top=progress_top,
                left=scene_x,
                right=scene_x + scene_width,
            )
            object_label = str(
                (final_obj or pre_obj or {}).get(
                    "frame_label",
                    (final_obj or pre_obj or {}).get(
                        "label",
                        dict(payload.get("step8b_metrics", {}))
                        .get("track_facts", {})
                        .get("object_class", "unknown"),
                    ),
                )
            )
            info_top = progress_top + 112
            _text(
                cv2,
                canvas,
                _clip_text_to_width(
                    cv2,
                    (
                        f"[8B] {object_label} | "
                        f"track {payload.get('track_id', -1)}"
                    ),
                    left_width - 48,
                    0.88,
                    2,
                ),
                24,
                info_top,
                0.88,
                (80, 215, 240),
                2,
            )

            table_top = info_top + 32
            table_left = 24
            table_right = left_width - 24
            row_height = 34
            column_x = (table_left + 8, 235, 395, 555, 715, 875)
            headers = ("SOURCE", "x", "z", "vx", "vz", "speed")
            for column, header in enumerate(headers):
                _text(
                    cv2,
                    canvas,
                    header,
                    column_x[column],
                    table_top + 23,
                    0.56,
                    (185, 195, 208),
                    2,
                )
            cv2.line(
                canvas,
                (table_left, table_top + row_height),
                (table_right, table_top + row_height),
                (65, 72, 84),
                2,
                cv2.LINE_AA,
            )

            before_values = _signal_values(pre_obj)
            after_values = _signal_values(final_obj)

            def formatted_signal(values):
                if values is None:
                    return ("—", "—", "—", "—", "—")
                return (
                    f"{values[0]:+.2f}",
                    f"{values[1]:+.2f}",
                    f"{values[2]:+.2f}",
                    f"{values[3]:+.2f}",
                    f"{values[4]:.2f}",
                )

            if before_values is not None and after_values is not None:
                delta_values = tuple(
                    abs(after - before)
                    for before, after in zip(before_values, after_values)
                )
                delta_text = tuple(f"{value:.2f}" for value in delta_values)
                position_delta = math.hypot(delta_values[0], delta_values[1])
                velocity_delta = math.hypot(delta_values[2], delta_values[3])
            else:
                delta_values = None
                delta_text = ("N/A", "N/A", "N/A", "N/A", "N/A")
                position_delta = None
                velocity_delta = None

            signal_rows = (
                ("ORIGINAL [8A]", formatted_signal(before_values), (40, 185, 245)),
                ("REPAIRED [8C]", formatted_signal(after_values), (70, 220, 100)),
                ("ABS DELTA", delta_text, (80, 215, 240)),
            )
            for row_index, (row_label, row_values, color) in enumerate(signal_rows):
                baseline = table_top + row_height * (row_index + 1) + 24
                values = (row_label,) + row_values
                for column, value in enumerate(values):
                    _text(
                        cv2,
                        canvas,
                        str(value),
                        column_x[column],
                        baseline,
                        0.58,
                        color,
                        2,
                    )
            table_bottom = table_top + row_height * 4
            cv2.rectangle(
                canvas,
                (table_left, table_top),
                (table_right, table_bottom),
                (65, 72, 84),
                2,
            )
            cue_groups = (
                ("leftness", "rightness", "approach"),
                ("recede", "acceleration", "deceleration"),
                (
                    "relative_static",
                    "relative_moving",
                    "relative_motion_uncertain",
                ),
            )
            object_observed_in_current_frame = pre_obj is not None
            cue_header = (
                "CUES [8B] (green=active)"
                if object_observed_in_current_frame
                else "CUES [8B] (inactive: object absent)"
            )
            _text(
                cv2,
                canvas,
                cue_header,
                24,
                table_bottom + 24,
                0.50,
                (
                    (220, 225, 232)
                    if object_observed_in_current_frame
                    else (145, 152, 163)
                ),
                1,
            )
            cue_column_x = (285, 555, 825)
            for cue_row, cue_names in enumerate(cue_groups):
                for cue_column, name in enumerate(cue_names):
                    cue_text, cue_color, cue_thickness, _ = _cue_visual_state(
                        name,
                        observable_cues.get(name),
                        object_observed_in_current_frame,
                    )
                    _text(
                        cv2,
                        canvas,
                        cue_text,
                        cue_column_x[cue_column],
                        table_bottom + 24 + cue_row * 25,
                        0.54,
                        cue_color,
                        cue_thickness,
                    )

            bbox_difference = _bbox_difference_metrics(pre_obj, final_obj)
            magnitude_parts = [
                (
                    f"|Δposition|={position_delta:.3f}m"
                    if position_delta is not None
                    else "|Δposition|=N/A"
                ),
                (
                    f"|Δvelocity|={velocity_delta:.3f}m/s"
                    if velocity_delta is not None
                    else "|Δvelocity|=N/A"
                ),
            ]
            if bbox_difference is not None:
                magnitude_parts.extend(
                    (
                        f"bbox shift={bbox_difference['center_shift_px']:.1f}px",
                        f"bbox IoU={bbox_difference['iou']:.3f}",
                    )
                )
            else:
                magnitude_parts.extend(("bbox shift=N/A", "bbox IoU=N/A"))
            _text(
                cv2,
                canvas,
                "   |   ".join(magnitude_parts),
                24,
                min(total_height - 38, table_bottom + 78),
                0.55,
                (225, 228, 234),
                1,
            )
            _text(
                cv2,
                canvas,
                "paths/bboxes: ORANGE original [8A] | GREEN repaired [8C]",
                24,
                min(total_height - 16, table_bottom + 101),
                0.54,
                (185, 195, 208),
                1,
            )
            writer.write(canvas)
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
    """Render stable, capped per-track Step 8C statistical-repair MP4s."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = list(state.get("trajectory_pattern_records", []))
    del max_tracks_per_video  # The visualization budget is global and fixed.
    limit = _MAX_TRACK_VIDEOS_TOTAL
    selected_records = select_deterministic_track_records(records)
    pre_by_video = {
        str(video.get("video_id", "")): video
        for video in state.get("pre_pattern_relative_object_motion", [])
    }
    final_by_video = {
        str(video.get("video_id", "")): video
        for video in state.get("relative_object_motion", [])
    }
    ego_by_video = {
        str(video.get("video_id", "")): video
        for video in state.get("ego_motion", [])
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

    selected_stems = {
        f"{record.get('video_id','')}_track_{int(record.get('track_id',-1)):04d}"
        for record in selected_records
    }
    pruned_artifacts = []
    for artifact in list(output_root.glob("*/track_*/*_8b_8c.mp4")):
        if artifact.is_file():artifact.unlink();pruned_artifacts.append(str(artifact))
    for artifact in list(output_root.glob("*_track_*_8b_8c.mp4")):
        stem=artifact.name.removesuffix("_8b_8c.mp4")
        if stem not in selected_stems and artifact.is_file():artifact.unlink();pruned_artifacts.append(str(artifact))

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
        f"max_videos_total={limit} output_layout=flat "
        f"fps={float(fps):.2f} output_root={output_root}",
        flush=True,
    )
    mp4_started = time.perf_counter()
    with tqdm(
        total=total_expected_frames,
        desc="[step 8c] statistical repair MP4",
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
            artifact_stem = f"{video_id}_track_{track_id:04d}"
            track_root = output_root
            payload = build_step8bc_track_video_payload(record)
            output_path = track_root / f"{artifact_stem}_8b_8c.mp4"
            try:
                path, status = _render_step8bc_track_video(
                    record=record,
                    payload=payload,
                    pre_pattern_video=pre_by_video.get(video_id, {}),
                    final_video=final_by_video.get(
                        video_id, pre_by_video.get(video_id, {})
                    ),
                    ego_video=ego_by_video.get(video_id, {}),
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
    manifest = {
        "version": 3,
        "selection_policy": _TRACK_VIDEO_SELECTION_NAMESPACE,
        "layout": "scene_left_statistical_repair_right",
        "scene_column_width": _LEFT_SCENE_WIDTH,
        "track_progress_position": "directly_below_scene",
        "canvas_resolution": [_OUTPUT_WIDTH, _OUTPUT_HEIGHT],
        "canvas_aspect_ratio": "4:3",
        "scene_bbox_labels": False,
        "progress_colors": {
            "original_presence": "orange",
            "modified_or_added": "green",
            "current_frame": "white",
        },
        "max_tracks_per_video": None,
        "max_visualization_videos_total": limit,
        "output_folder": str(output_root),
        "num_available_tracks": sum(
            len(values) for values in available_by_video.values()
        ),
        "num_selected_tracks": len(selected_records),
        "num_rendered_videos": len(rendered),
        "num_skipped_videos": len(skipped),
        "num_pruned_stale_artifacts": len(pruned_artifacts),
        "pruned_stale_artifacts": pruned_artifacts,
        "selections": selections,
        "rendered": rendered,
        "skipped": skipped,
    }
    return {**manifest, "manifest_path": ""}


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
            f"{_MAX_TRACK_VIDEOS_TOTAL}-video global cap.</p></section>"
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
            "<details><summary>Complete Step 8B nine-cue evidence"
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
        '<ol class="flow"><li>[8B → 8C] symbolic abstraction</li>'
        "<li>[8C] all-pattern residuals</li><li>[8C] LLM interpretation</li>"
        "<li>[8C] multi-repair</li><li>[8C] symbolic validation</li>"
        "<li>[8C] final selection</li></ol>"
        + "".join(alert_rows)
        + '<section class="panel" id="symbolic-track"><h2>Symbolic track [8B → 8C]</h2>'
        '<div class="cards">'
        f'<div class="card">Class [8B]<b>{_html_text(track.get("object_class", "unknown"))}</b></div>'
        f'<div class="card">Direction [8A → 8C]<b>{_html_text(track.get("direction", "unknown"))}</b></div>'
        f'<div class="card">Persistence [8B]<b>{_html_text(_display_value(track.get("persistence")))}</b></div>'
        f'<div class="card">Confidence [8B]<b>{_html_text(_display_value(track.get("confidence")))}</b></div>'
        + source_evidence_html
        + media_section
        + '<section class="panel" id="pattern-residuals">'
        "<h2>All-pattern residual distances [8C]</h2>"
        '<p class="muted">Each cell shows before → final.</p><table><thead><tr>'
        "<th>Pattern</th><th>Plausibility</th><th>Residual sum</th>"
        + residual_headers
        + "<th>LLM conflicts / explanation</th></tr></thead><tbody>"
        + "".join(pattern_rows)
        + "</tbody></table></section>"
        + '<section class="panel" id="llm-interpretation">'
        "<h2>LLM interpretation [8C]</h2>"
        + "".join(interpretation_rows)
        + "</section>"
        + '<section class="panel" id="repair-candidates">'
        "<h2>Deterministic repair candidates [8C]</h2><table><thead><tr>"
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


def _percentage_summary(counts):
    """Return JSON-safe counts and percentages for one categorical result."""
    total = sum(int(value) for value in counts.values())
    return {
        str(key): {
            "count": int(value),
            "percentage": (100.0 * int(value) / total) if total else 0.0,
        }
        for key, value in counts.items()
    }


def _plot_percentage_bars(ax, counts, title, *, max_items=14):
    """Draw a readable horizontal count/percentage chart."""
    rows = sorted(
        ((str(key), int(value)) for key, value in counts.items()),
        key=lambda row: (-row[1], row[0]),
    )
    if len(rows) > max_items:
        kept = rows[: max_items - 1]
        kept.append(("other", sum(value for _, value in rows[max_items - 1 :])))
        rows = kept
    total = sum(value for _, value in rows)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    if not rows or total <= 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    rows = list(reversed(rows))
    labels = [label for label, _ in rows]
    values = [value for _, value in rows]
    bars = ax.barh(range(len(rows)), values, color="#4C9F70")
    ax.set_yticks(range(len(rows)), labels=labels)
    ax.set_xlabel("tracks / candidates")
    ax.grid(axis="x", alpha=0.22)
    maximum = max(values)
    ax.set_xlim(0, max(1.0, maximum * 1.32))
    for bar, value in zip(bars, values):
        ax.text(
            value + max(0.05, maximum * 0.015),
            bar.get_y() + bar.get_height() / 2,
            f"{value} ({100.0 * value / total:.1f}%)",
            va="center",
            fontsize=9,
        )


def render_step8c_statistical_pdfs(records, promotion, output_root):
    """Render read-only dataset-level Step 8C statistical PDF reports."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = list(records)
    output_root = Path(output_root) / "statistics_pdfs"
    output_root.mkdir(parents=True, exist_ok=True)

    repair_counts = Counter(
        "repaired" if bool(row.get("repair_applied")) else "not_repaired"
        for row in records
    )
    resolution_counts = Counter(
        str(row.get("resolution_status", "unknown")) for row in records
    )
    final_validation_counts = Counter(
        str(row.get("final_validation_status", "unknown")) for row in records
    )
    validation_transition_counts = Counter(
        f"{row.get('initial_8c_validation_status', 'unknown')} → "
        f"{row.get('final_validation_status', 'unknown')}"
        for row in records
    )
    final_pattern_counts = Counter(
        str(row.get("final_pattern", "unknown")) for row in records
    )
    llm_agreement_counts = Counter(
        "agree"
        if str(row.get("LLM_preferred_pattern", "unknown"))
        == str(row.get("validated_pattern", "unknown"))
        else "disagree"
        for row in records
    )
    class_counts = Counter(
        str(dict(row.get("symbolic_track", {})).get("object_class", "unknown"))
        for row in records
    )
    cohort_counts = Counter(
        str(row.get("trajectory_cohort_id", "unknown")) for row in records
    )
    selected_operator_counts = Counter()
    candidate_verdict_counts = Counter()
    hard_constraint_failure_counts = Counter()
    record_status_counts = Counter(
        str(row.get("record_status", "unknown")) for row in records
    )
    for row in records:
        if bool(row.get("repair_applied")):
            selected = dict(row.get("selected_candidate", {}))
            hypothesis = dict(selected.get("repair_hypothesis", {}))
            operator = selected.get(
                "repair_operation", hypothesis.get("operation", "unknown")
            )
            selected_operator_counts[str(operator)] += 1
        else:
            selected_operator_counts["no_repair"] += 1
        for candidate in row.get("candidate_repairs", []):
            verdict = str(
                candidate.get("symbolic_verdict", candidate.get("decision", "unknown"))
            )
            candidate_verdict_counts[verdict] += 1
            for constraint, passed in dict(
                candidate.get("hard_constraint_results", {})
            ).items():
                if not bool(passed):
                    hard_constraint_failure_counts[str(constraint)] += 1

    chart_groups = (
        (
            "01_track_outcomes.pdf",
            "Step 8C — Track outcomes",
            (
                (repair_counts, "Repair application"),
                (resolution_counts, "Resolution status"),
                (final_validation_counts, "Final validation status"),
                (validation_transition_counts, "Initial → final validation"),
            ),
        ),
        (
            "02_patterns_and_cohorts.pdf",
            "Step 8C — Patterns and semantic groups",
            (
                (final_pattern_counts, "Validated trajectory patterns"),
                (llm_agreement_counts, "LLM prior vs validated pattern"),
                (class_counts, "Object classes"),
                (cohort_counts, "Trajectory cohorts"),
            ),
        ),
        (
            "03_repair_and_validation.pdf",
            "Step 8C — Repair and validation diagnostics",
            (
                (selected_operator_counts, "Selected repair operation"),
                (candidate_verdict_counts, "Candidate symbolic verdicts"),
                (hard_constraint_failure_counts, "Hard-constraint failures"),
                (record_status_counts, "Completed record status"),
            ),
        ),
    )

    reports = []
    for filename, title, plots in chart_groups:
        figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
        figure.suptitle(
            f"{title}\ntracks={len(records)}",
            fontsize=17,
            fontweight="bold",
        )
        for axis, (counts, subtitle) in zip(axes.flat, plots):
            _plot_percentage_bars(axis, counts, subtitle)
        path = output_root / filename
        figure.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        reports.append(
            {
                "report_id": path.stem,
                "visualization_path": str(path),
                "report_path": str(path),
                "media_type": "application/pdf",
                "num_tracks": len(records),
            }
        )

    summary = {
        "schema_version": 1,
        "num_tracks": len(records),
        "repair_application": _percentage_summary(repair_counts),
        "resolution_status": _percentage_summary(resolution_counts),
        "final_validation_status": _percentage_summary(final_validation_counts),
        "validation_transitions": _percentage_summary(validation_transition_counts),
        "final_patterns": _percentage_summary(final_pattern_counts),
        "llm_validated_pattern_agreement": _percentage_summary(llm_agreement_counts),
        "object_classes": _percentage_summary(class_counts),
        "trajectory_cohorts": _percentage_summary(cohort_counts),
        "selected_repair_operations": _percentage_summary(selected_operator_counts),
        "candidate_symbolic_verdicts": _percentage_summary(candidate_verdict_counts),
        "hard_constraint_failures": _percentage_summary(hard_constraint_failure_counts),
        "record_status": _percentage_summary(record_status_counts),
        "statistics_promotion": copy.deepcopy(dict(promotion or {})),
        "pdf_reports": reports,
    }
    return {
        "reports": reports,
        "summary": summary,
        "summary_path": "",
        "output_root": str(output_root),
    }


def _retain_step8h_media_only(output_root):
    """Keep Step 8H disk output restricted to MP4 videos and PDF reports."""
    output_root = Path(output_root)
    if not output_root.exists():
        return []
    removed = []
    for artifact in sorted(output_root.rglob("*"), reverse=True):
        if artifact.is_file() and artifact.suffix.lower() not in {".mp4", ".pdf"}:
            artifact.unlink()
            removed.append(str(artifact))
    for directory in sorted(
        (path for path in output_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    return removed


def render_trajectory_pattern_visualizations(state, output_root):
    """Write only capped statistical-repair MP4s and statistical PDFs."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    removed_non_media = _retain_step8h_media_only(output_root)
    started = time.perf_counter()
    records = list(state.get("trajectory_pattern_records", []))
    print(
        f"[step 8h][visualization] START tracks={len(records)} "
        f"output_root={output_root}",
        flush=True,
    )
    track_video_manifest = render_step8bc_track_videos(
        state,
        output_root,
        fps=max(0.1, _number(state.get("step8bc_visualization_fps", 10.0), 10.0)),
        max_tracks_per_video=_MAX_TRACK_VIDEOS_TOTAL,
    )
    statistical_pdfs = render_step8c_statistical_pdfs(
        records,
        dict(state.get("trajectory_pattern_statistics_promotion", {})),
        output_root,
    )
    removed_non_media.extend(_retain_step8h_media_only(output_root))
    print(
        f"[step 8h][visualization] DONE "
        f"statistical_pdf_reports={len(statistical_pdfs.get('reports', []))} "
        f"track_videos={len(track_video_manifest.get('rendered', []))} "
        f"track_video_skipped={len(track_video_manifest.get('skipped', []))} "
        f"removed_non_media={len(removed_non_media)} "
        f"latency={time.perf_counter() - started:.2f}s output_root={output_root}",
        flush=True,
    )
    return {
        **state,
        "trajectory_pattern_visualizations": [],
        "trajectory_pattern_video_summaries": [],
        "trajectory_pattern_statistical_pdf_reports": list(statistical_pdfs.get("reports", [])),
        "trajectory_pattern_statistical_summary": dict(statistical_pdfs.get("summary", {})),
        "trajectory_pattern_statistical_summary_path": "",
        "trajectory_pattern_statistical_pdf_output_root": str(statistical_pdfs.get("output_root", "")),
        "trajectory_pattern_track_videos": list(track_video_manifest.get("rendered", [])),
        "trajectory_pattern_track_video_skipped": list(track_video_manifest.get("skipped", [])),
        "trajectory_pattern_track_video_selections": list(track_video_manifest.get("selections", [])),
        "trajectory_pattern_track_video_manifest_path": "",
        "trajectory_pattern_visualization_skipped": [],
        "trajectory_pattern_visualization_output_root": output_root,
    }
