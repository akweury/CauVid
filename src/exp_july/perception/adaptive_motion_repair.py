"""Diagnostic Step 8D adaptive repair for semantically protected trajectories."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


_VERSION = 1
_IMPROVEMENT_MARGIN = 8.0
_SEVERITY = {"valid": 0, "repaired": 0, "uncertain": 1, "invalid": 3}
_ISSUE_SEVERITY = {"uncertain": 1, "invalid": 3}
_DEPENDENCY_ORDER = (
    "id_switch",
    "trajectory_discontinuity",
    "track_drift",
    "bbox_jump",
    "depth_jump",
    "speed_abnormal_change",
    "motion_direction_abrupt_change",
)
_STRATEGIES = {
    "id_switch": ("track_split", "fragment_reassociation"),
    "trajectory_discontinuity": ("gap_interpolation", "kalman_smoothing", "robust_polynomial_regression"),
    "track_drift": ("outlier_removal", "bbox_stabilization", "kalman_smoothing", "robust_polynomial_regression"),
    "bbox_jump": ("bbox_stabilization", "kalman_smoothing"),
    "depth_jump": ("depth_reestimation", "outlier_removal", "robust_polynomial_regression", "kalman_smoothing"),
    "speed_abnormal_change": ("multi_frame_velocity_recomputation", "outlier_removal", "robust_polynomial_regression", "kalman_smoothing"),
    "motion_direction_abrupt_change": ("multi_frame_velocity_recomputation", "kalman_smoothing", "robust_polynomial_regression"),
}
_DEFAULT_PARAMETERS = {
    "track_split": {"minimum_segment_length": 2},
    "fragment_reassociation": {"normalize_to_dominant_label": True},
    "gap_interpolation": {"maximum_gap": 12},
    "kalman_smoothing": {"alpha": 0.55},
    "robust_polynomial_regression": {"degree": 1, "residual_clip_mad": 3.0},
    "outlier_removal": {"median_radius": 2, "mad_scale": 3.0},
    "bbox_stabilization": {"median_radius": 2},
    "depth_reestimation": {"median_radius": 2, "mad_scale": 2.5},
    "multi_frame_velocity_recomputation": {"window": 3},
}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _median(values: Iterable[float]) -> float:
    rows = sorted(_f(value) for value in values)
    if not rows:
        return 0.0
    middle = len(rows) // 2
    return rows[middle] if len(rows) % 2 else (rows[middle - 1] + rows[middle]) / 2.0


def _mad(values: Sequence[float]) -> float:
    center = _median(values)
    return _median(abs(_f(value) - center) for value in values)


def _rule_group(trajectory: Dict[str, Any]) -> str:
    protection = dict(trajectory.get("symbol_grounded_protection", {}))
    rule_ids = sorted(str(value) for value in protection.get("matched_rule_ids", []) if str(value))
    label = str(trajectory.get("primary_label", "unknown")).strip().lower().replace(" ", "_")
    return "+".join(rule_ids) if rule_ids else f"class:{label}"


def _refined_ego_by_frame(refined_ego: Dict[str, Any]) -> Dict[int, Tuple[float, float]]:
    return {
        int(row.get("frame_index", index)): (
            _f(row.get("refined_ego_vx", row.get("ego_vx", 0.0))),
            _f(row.get("refined_ego_vz", row.get("ego_vz", 0.0))),
        )
        for index, row in enumerate(refined_ego.get("frames", []))
    }


def _interpolate(left: Sequence[float], right: Sequence[float], ratio: float) -> List[float]:
    size = min(len(left), len(right))
    return [_f(left[idx]) + ratio * (_f(right[idx]) - _f(left[idx])) for idx in range(size)]


def _smooth_series(values: Sequence[float], alpha: float) -> List[float]:
    if not values:
        return []
    forward = [_f(values[0])]
    for value in values[1:]:
        forward.append(alpha * _f(value) + (1.0 - alpha) * forward[-1])
    backward = [_f(forward[-1])]
    for value in reversed(forward[:-1]):
        backward.append(alpha * _f(value) + (1.0 - alpha) * backward[-1])
    backward.reverse()
    return [(left + right) / 2.0 for left, right in zip(forward, backward)]


def _linear_fit(xs: Sequence[float], ys: Sequence[float], residual_clip_mad: float = 3.0) -> List[float]:
    if len(xs) < 2:
        return list(ys)
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / max(1e-9, denominator)
    intercept = y_mean - slope * x_mean
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    limit = max(1e-6, residual_clip_mad * _mad(residuals))
    kept = [(x, y) for x, y, residual in zip(xs, ys, residuals) if abs(residual) <= limit]
    if 2 <= len(kept) < len(xs):
        return _linear_fit([row[0] for row in kept], [row[1] for row in kept], residual_clip_mad)
    return [intercept + slope * x for x in xs]


def _median_filtered(values: Sequence[float], radius: int = 2) -> List[float]:
    return [
        _median(values[max(0, idx - radius) : min(len(values), idx + radius + 1)])
        for idx in range(len(values))
    ]


def _modified_frames(before: Sequence[Dict[str, Any]], after: Sequence[Dict[str, Any]]) -> List[int]:
    before_by_frame = {int(row.get("frame_index", -1)): row for row in before}
    modified = []
    for row in after:
        frame_id = int(row.get("frame_index", -1))
        original = before_by_frame.get(frame_id)
        if original is None or original.get("position_3d") != row.get("position_3d") or original.get("bbox") != row.get("bbox") or original.get("motion") != row.get("motion") or original.get("frame_label") != row.get("frame_label"):
            modified.append(frame_id)
    return sorted(set(modified))


def _apply_strategy(
    observations: Sequence[Dict[str, Any]],
    strategy: str,
    parameters: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = sorted(copy.deepcopy(list(observations)), key=lambda row: int(row.get("frame_index", -1)))
    if not rows:
        return rows
    if strategy == "track_split":
        counts = Counter(str(row.get("frame_label", "unknown")) for row in rows)
        dominant = counts.most_common(1)[0][0]
        rows = [row for row in rows if str(row.get("frame_label", "unknown")) == dominant]
    elif strategy == "fragment_reassociation":
        dominant = Counter(str(row.get("frame_label", "unknown")) for row in rows).most_common(1)[0][0]
        for row in rows:
            row["frame_label"] = dominant
            row.setdefault("repair_annotations", []).append("label_reassociated")
    elif strategy == "gap_interpolation":
        expanded = []
        maximum_gap = int(parameters.get("maximum_gap", 12))
        for left, right in zip(rows, rows[1:]):
            expanded.append(left)
            left_frame = int(left.get("frame_index", -1))
            right_frame = int(right.get("frame_index", -1))
            gap = right_frame - left_frame
            if 1 < gap <= maximum_gap:
                for frame_id in range(left_frame + 1, right_frame):
                    ratio = (frame_id - left_frame) / gap
                    inserted = copy.deepcopy(left)
                    inserted["frame_index"] = frame_id
                    inserted["position_3d"] = _interpolate(left.get("position_3d", []), right.get("position_3d", []), ratio)
                    inserted["bbox"] = _interpolate(left.get("bbox", []), right.get("bbox", []), ratio)
                    inserted["provenance"] = {
                        **dict(inserted.get("provenance", {})),
                        "source": "repaired",
                        "is_observed": False,
                        "is_repaired": True,
                        "repair_provenance": {"strategy": strategy, "left_frame": left_frame, "right_frame": right_frame},
                    }
                    inserted["uncertainty"] = {**dict(inserted.get("uncertainty", {})), "source_uncertainty": 0.35}
                    expanded.append(inserted)
        expanded.append(rows[-1])
        rows = sorted(expanded, key=lambda row: int(row.get("frame_index", -1)))
    elif strategy in {"kalman_smoothing", "robust_polynomial_regression", "outlier_removal", "depth_reestimation"}:
        frames = [_f(row.get("frame_index", index)) for index, row in enumerate(rows)]
        for coordinate in (0, 2):
            values = [_f(row.get("position_3d", [0.0, 0.0, 0.0])[coordinate]) for row in rows]
            if strategy == "kalman_smoothing":
                repaired = _smooth_series(values, _f(parameters.get("alpha", 0.55), 0.55))
            elif strategy == "robust_polynomial_regression":
                repaired = _linear_fit(frames, values, _f(parameters.get("residual_clip_mad", 3.0), 3.0))
            else:
                baseline = _median_filtered(values, int(parameters.get("median_radius", 2)))
                scale = max(1e-6, _mad([value - base for value, base in zip(values, baseline)]))
                limit = _f(parameters.get("mad_scale", 3.0), 3.0) * scale
                repaired = [base if abs(value - base) > limit else value for value, base in zip(values, baseline)]
            for row, value in zip(rows, repaired):
                position = list(row.get("position_3d", []))
                while len(position) < 3:
                    position.append(0.0)
                position[coordinate] = float(value)
                row["position_3d"] = position
    elif strategy == "bbox_stabilization":
        features = []
        for row in rows:
            box = list(row.get("bbox", [0.0, 0.0, 0.0, 0.0]))
            while len(box) < 4:
                box.append(0.0)
            features.append(((_f(box[0]) + _f(box[2])) / 2.0, (_f(box[1]) + _f(box[3])) / 2.0, abs(_f(box[2]) - _f(box[0])), abs(_f(box[3]) - _f(box[1]))))
        radius = int(parameters.get("median_radius", 2))
        repaired_features = [_median_filtered([row[idx] for row in features], radius) for idx in range(4)]
        for index, row in enumerate(rows):
            cx, cy, width, height = [series[index] for series in repaired_features]
            row["bbox"] = [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]
    return rows


def _recompute_motion(
    observations: Sequence[Dict[str, Any]],
    ego_by_frame: Dict[int, Tuple[float, float]],
    velocity_window: int = 1,
) -> List[Dict[str, Any]]:
    rows = sorted(copy.deepcopy(list(observations)), key=lambda row: int(row.get("frame_index", -1)))
    for index, row in enumerate(rows):
        frame_id = int(row.get("frame_index", -1))
        ego_vx, ego_vz = ego_by_frame.get(frame_id, (
            _f(dict(row.get("motion", {})).get("ego_vx", 0.0)),
            _f(dict(row.get("motion", {})).get("ego_vz", 0.0)),
        ))
        motion = dict(row.get("motion", {}))
        motion.update({"ego_vx": ego_vx, "ego_vz": ego_vz})
        if index == 0:
            motion.update({"obj_vx": 0.0, "obj_vz": 0.0, "rel_vx": 0.0, "rel_vz": 0.0, "rel_speed": 0.0, "has_rel_motion": False})
        else:
            previous = rows[max(0, index - max(1, int(velocity_window)))]
            gap = max(1, frame_id - int(previous.get("frame_index", frame_id - 1)))
            left = list(previous.get("position_3d", []))
            right = list(row.get("position_3d", []))
            if len(left) >= 3 and len(right) >= 3:
                obj_vx = (_f(right[0]) - _f(left[0])) / gap
                obj_vz = (_f(right[2]) - _f(left[2])) / gap
                rel_vx = obj_vx - ego_vx
                rel_vz = obj_vz - ego_vz
                motion.update({
                    "obj_vx": obj_vx,
                    "obj_vz": obj_vz,
                    "rel_vx": rel_vx,
                    "rel_vz": rel_vz,
                    "rel_speed": math.hypot(rel_vx, rel_vz),
                    "has_rel_motion": True,
                    "distance_meters": _f(right[2]),
                })
        row["motion"] = motion
        row.setdefault("repair_annotations", []).append("downstream_motion_recomputed")
    return rows


def _evaluate(observations: Sequence[Dict[str, Any]], num_frames: int) -> Dict[str, Any]:
    from src.exp_july.perception import pipeline as core

    rows = list(observations)
    statistics = core._trajectory_statistics(rows, num_frames)
    uncertainty = core._trajectory_uncertainty(rows, statistics)
    validation = core._trajectory_reality_validation(rows, statistics, uncertainty)
    provenance = {
        "observed_count": int(statistics.get("observed_count", 0)),
        "repaired_count": int(statistics.get("repaired_count", 0)),
        "merged_count": int(statistics.get("merged_count", 0)),
        "observed_ratio": _f(statistics.get("observed_ratio", 0.0)),
        "repaired_ratio": _f(statistics.get("repaired_ratio", 0.0)),
        "merged_ratio": _f(statistics.get("merged_ratio", 0.0)),
    }
    significance = core._motion_significance_assessment(statistics, provenance, uncertainty, validation)
    decision = core._fact_decision_for_trajectory(validation, significance, provenance, uncertainty)
    return {
        "statistics": statistics,
        "uncertainty": uncertainty,
        "validation": validation,
        "significance": significance,
        "decision": decision,
    }


def _issue_cost(evaluation: Dict[str, Any]) -> float:
    validation = dict(evaluation.get("validation", {}))
    issues = list(validation.get("issues", []))
    cost = 35.0 * sum(_ISSUE_SEVERITY.get(str(issue.get("severity", "uncertain")), 1) for issue in issues)
    metrics = dict(validation.get("step_metrics", {}))
    cost += min(25.0, _f(metrics.get("max_bbox_center_step_diag_ratio", 0.0)) * 4.0)
    cost += min(25.0, _f(metrics.get("max_depth_step_per_frame", 0.0)))
    cost += min(25.0, _f(metrics.get("max_rel_velocity_delta", 0.0)))
    cost += min(25.0, _f(metrics.get("max_rel_speed", 0.0)))
    cost += 15.0 * _f(dict(evaluation.get("uncertainty", {})).get("uncertainty_score", 0.0))
    return float(cost)


def _snapshot(observations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "frame_id": int(row.get("frame_index", -1)),
            "position": list(row.get("position_3d", [])),
            "depth": _f(list(row.get("position_3d", [0.0, 0.0, 0.0]))[2] if len(row.get("position_3d", [])) >= 3 else 0.0),
            "bbox": list(row.get("bbox", [])),
            "confidence": _f(dict(row.get("uncertainty", {})).get("score", 0.0)),
            "vx": _f(dict(row.get("motion", {})).get("obj_vx", 0.0)),
            "vz": _f(dict(row.get("motion", {})).get("obj_vz", 0.0)),
        }
        for row in observations
    ]


def _rmse(original: Sequence[Dict[str, Any]], repaired: Sequence[Dict[str, Any]]) -> float:
    repaired_by_frame = {int(row.get("frame_index", -1)): row for row in repaired}
    errors = []
    for row in original:
        candidate = repaired_by_frame.get(int(row.get("frame_index", -1)))
        if candidate is None:
            continue
        left = list(row.get("position_3d", []))
        right = list(candidate.get("position_3d", []))
        if len(left) >= 3 and len(right) >= 3:
            errors.append(sum((_f(left[idx]) - _f(right[idx])) ** 2 for idx in (0, 2)))
    return math.sqrt(sum(errors) / max(1, len(errors)))


def _synthetic_corruption(observations: Sequence[Dict[str, Any]], rejection: str) -> List[Dict[str, Any]]:
    rows = copy.deepcopy(list(observations))
    if len(rows) < 4:
        return rows
    middle = len(rows) // 2
    if rejection == "id_switch":
        rows[middle]["frame_label"] = "__synthetic_switch__"
    elif rejection == "trajectory_discontinuity":
        rows = rows[:middle] + rows[min(len(rows), middle + 3) :]
    elif rejection == "track_drift":
        rows[middle]["bbox"] = [_f(value) + 400.0 for value in rows[middle].get("bbox", [])]
    elif rejection == "bbox_jump":
        box = list(rows[middle].get("bbox", []))
        if len(box) >= 4:
            box[2] = box[0] + max(1.0, (box[2] - box[0]) * 4.0)
            box[3] = box[1] + max(1.0, (box[3] - box[1]) * 4.0)
            rows[middle]["bbox"] = box
    elif rejection == "depth_jump":
        rows[middle]["position_3d"][2] = _f(rows[middle]["position_3d"][2]) + 20.0
    elif rejection == "speed_abnormal_change":
        rows[middle]["position_3d"][0] = _f(rows[middle]["position_3d"][0]) + 30.0
    elif rejection == "motion_direction_abrupt_change":
        for index in (middle, min(len(rows) - 1, middle + 1)):
            rows[index]["position_3d"][0] = -_f(rows[index]["position_3d"][0])
    return rows


def _parameter_candidates(strategy: str) -> List[Dict[str, Any]]:
    base = dict(_DEFAULT_PARAMETERS[strategy])
    variants = [base]
    if strategy == "kalman_smoothing":
        variants = [{**base, "alpha": value} for value in (0.35, 0.55, 0.75)]
    elif strategy in {"outlier_removal", "depth_reestimation"}:
        variants = [{**base, "mad_scale": value} for value in (2.0, 2.5, 3.0)]
    elif strategy == "bbox_stabilization":
        variants = [{**base, "median_radius": value} for value in (1, 2, 3)]
    elif strategy == "gap_interpolation":
        variants = [{**base, "maximum_gap": value} for value in (4, 8, 12)]
    elif strategy == "robust_polynomial_regression":
        variants = [{**base, "residual_clip_mad": value} for value in (2.0, 3.0, 4.0)]
    elif strategy == "multi_frame_velocity_recomputation":
        variants = [{**base, "window": value} for value in (2, 3, 4)]
    return variants


def _calibrate(
    evidence_videos: Sequence[Dict[str, Any]],
    refined_by_video: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    reliable = []
    for video in evidence_videos:
        video_id = str(video.get("video_id", ""))
        for trajectory in video.get("trajectory_motion_evidence", []):
            decision = str(trajectory.get("fact_decision_status", ""))
            confidence = _f(dict(trajectory.get("uncertainty", {})).get("confidence_score", 0.0))
            if decision == "Keep" or (decision == "Repair" and confidence >= 0.70):
                reliable.append((video_id, trajectory))
    measurements = defaultdict(list)
    for video_id, trajectory in reliable:
        original = list(trajectory.get("trajectory_observations", []))
        if len(original) < 4:
            continue
        ego = _refined_ego_by_frame(refined_by_video.get(video_id, {}))
        group = _rule_group(trajectory)
        for rejection, strategies in _STRATEGIES.items():
            corrupted = _recompute_motion(_synthetic_corruption(original, rejection), ego)
            for strategy in strategies:
                for parameters in _parameter_candidates(strategy):
                    window = int(parameters.get("window", 1)) if strategy == "multi_frame_velocity_recomputation" else 1
                    repaired = _recompute_motion(
                        _apply_strategy(corrupted, strategy, parameters),
                        ego,
                        velocity_window=window,
                    )
                    evaluation = _evaluate(repaired, int(trajectory.get("trajectory_statistics", {}).get("video_num_frames", len(repaired))))
                    success = rejection not in evaluation["validation"].get("rejection_reasons", []) and evaluation["validation"].get("validation_status") != "invalid"
                    signature = json.dumps(parameters, sort_keys=True)
                    measurements[(rejection, strategy, group, signature)].append((success, _rmse(original, repaired)))
                    measurements[(rejection, strategy, "global", signature)].append((success, _rmse(original, repaired)))
    parameter_rows = []
    for (rejection, strategy, group, signature), values in sorted(measurements.items()):
        if group != "global" and len(values) < 3:
            continue
        parameter_rows.append({
            "rejection_type": rejection,
            "strategy": strategy,
            "reference_group": group,
            "sample_count": len(values),
            "success_rate": sum(1 for success, _ in values if success) / max(1, len(values)),
            "expected_reconstruction_error": sum(error for _, error in values) / max(1, len(values)),
            "parameters": json.loads(signature),
        })
    best_by_group = {}
    for row in parameter_rows:
        key = (row["rejection_type"], row["strategy"], row["reference_group"])
        current = best_by_group.get(key)
        rank = (row["success_rate"], -row["expected_reconstruction_error"])
        if current is None or rank > (current["success_rate"], -current["expected_reconstruction_error"]):
            best_by_group[key] = row
    rows = [best_by_group[key] for key in sorted(best_by_group)]
    return {"num_reliable_reference_tracks": len(reliable), "strategy_measurements": rows}


def _strategy_ranking(reasons: Sequence[str], calibration: Dict[str, Any], group: str) -> List[Tuple[str, Dict[str, Any]]]:
    measurements = list(calibration.get("strategy_measurements", []))
    candidates = []
    for reason in _DEPENDENCY_ORDER:
        if reason not in reasons:
            continue
        for strategy in _STRATEGIES.get(reason, ()):
            matching = [
                row for row in measurements
                if row.get("rejection_type") == reason
                and row.get("strategy") == strategy
                and row.get("reference_group") in {group, "global"}
            ]
            best = max(matching, key=lambda row: (row.get("reference_group") == group, _f(row.get("success_rate", 0.0))), default={})
            candidates.append((strategy, {
                **dict(_DEFAULT_PARAMETERS[strategy]),
                "calibrated_success_rate": _f(best.get("success_rate", 0.0)),
                "expected_reconstruction_error": _f(best.get("expected_reconstruction_error", 0.0)),
                "calibration_group": str(best.get("reference_group", "default")),
                "addresses_rejection": reason,
            }))
    unique = []
    seen = set()
    for strategy, parameters in sorted(candidates, key=lambda row: (-_f(row[1].get("calibrated_success_rate", 0.0)), _f(row[1].get("expected_reconstruction_error", 0.0)))):
        if strategy not in seen:
            seen.add(strategy)
            unique.append((strategy, parameters))
    return unique


def _repair_track(
    trajectory: Dict[str, Any],
    refined_ego: Dict[str, Any],
    calibration: Dict[str, Any],
) -> Dict[str, Any]:
    original = copy.deepcopy(list(trajectory.get("trajectory_observations", [])))
    num_frames = int(trajectory.get("trajectory_statistics", {}).get("video_num_frames", len(original)))
    original_eval = _evaluate(original, num_frames)
    reasons = list(dict(trajectory.get("causal_motion_fact_validation", {})).get("rejection_reasons", original_eval["validation"].get("rejection_reasons", [])))
    group = _rule_group(trajectory)
    ego = _refined_ego_by_frame(refined_ego)
    candidates = []
    ranked = _strategy_ranking(reasons, calibration, group)
    pipelines = [[item] for item in ranked]
    if len(ranked) > 1:
        ordered = sorted(ranked, key=lambda row: _DEPENDENCY_ORDER.index(str(row[1].get("addresses_rejection"))) if str(row[1].get("addresses_rejection")) in _DEPENDENCY_ORDER else 99)
        pipelines.append(ordered)
    for pipeline in pipelines:
        repaired = copy.deepcopy(original)
        stages = []
        velocity_window = 1
        for strategy, parameters in pipeline:
            before_stage = repaired
            repaired = _apply_strategy(repaired, strategy, parameters)
            if strategy == "multi_frame_velocity_recomputation":
                velocity_window = max(2, int(parameters.get("window", 3)))
            repaired = _recompute_motion(repaired, ego, velocity_window=velocity_window)
            stage_eval = _evaluate(repaired, num_frames)
            stages.append({
                "strategy": strategy,
                "parameters": parameters,
                "modified_frame_ids": _modified_frames(before_stage, repaired),
                "rejection_reasons_after_stage": list(stage_eval["validation"].get("rejection_reasons", [])),
                "validation_status_after_stage": stage_eval["validation"].get("validation_status"),
            })
        evaluation = _evaluate(repaired, num_frames)
        remaining = list(evaluation["validation"].get("rejection_reasons", []))
        original_cost = _issue_cost(original_eval)
        repaired_cost = _issue_cost(evaluation)
        calibration_penalty = sum(
            (1.0 - _f(parameters.get("calibrated_success_rate", 0.0))) * 5.0
            + min(10.0, _f(parameters.get("expected_reconstruction_error", 0.0)))
            for _, parameters in pipeline
        )
        candidate_score = repaired_cost + calibration_penalty
        improvement = original_cost - repaired_cost
        original_frames = {int(row.get("frame_index", -1)) for row in original}
        repaired_frames = {int(row.get("frame_index", -1)) for row in repaired}
        removes_observations = bool(original_frames - repaired_frames)
        new_invalid_reasons = sorted(set(remaining) - set(reasons))
        introduces_more_severe = bool(new_invalid_reasons) or _SEVERITY.get(str(evaluation["validation"].get("validation_status", "invalid")), 3) > _SEVERITY.get(str(original_eval["validation"].get("validation_status", "invalid")), 3)
        accepted = improvement >= _IMPROVEMENT_MARGIN and not removes_observations and not introduces_more_severe
        candidates.append({
            "candidate_id": f"candidate_{len(candidates) + 1}",
            "strategies": stages,
            "accepted": accepted,
            "score": float(candidate_score),
            "score_components": {"validation_signal_cost": float(repaired_cost), "reference_group_consistency_penalty": float(calibration_penalty)},
            "improvement": float(improvement),
            "remaining_rejection_reasons": remaining,
            "modified_frame_ids": _modified_frames(original, repaired),
            "removes_original_observations": removes_observations,
            "introduces_more_severe_rejection": introduces_more_severe,
            "new_invalid_rejection_reasons": new_invalid_reasons,
            "evaluation": evaluation,
            "repaired_observations": repaired,
        })
    acceptable = [row for row in candidates if row["accepted"]]
    selected = min(acceptable, key=lambda row: row["score"], default=None)
    final_observations = selected["repaired_observations"] if selected else original
    final_eval = selected["evaluation"] if selected else original_eval
    remaining = list(final_eval["validation"].get("rejection_reasons", []))
    if selected and not remaining:
        final_decision = "Repair"
    elif selected:
        final_decision = "Keep_with_uncertainty"
    else:
        final_decision = "Unrepairable_but_semantically_retained"
    improvement = _f(selected.get("improvement", 0.0)) if selected else 0.0
    confidence = max(0.0, min(1.0, improvement / max(_IMPROVEMENT_MARGIN, _issue_cost(original_eval)))) if selected else 0.0
    return {
        "track_id": int(trajectory.get("track_id", -1)),
        "semantic_protection": copy.deepcopy(trajectory.get("symbol_grounded_protection", {})),
        "matched_protection_rules": list(dict(trajectory.get("symbol_grounded_protection", {})).get("matched_rule_ids", [])),
        "selected_reference_group": group,
        "original_decision": str(dict(trajectory.get("fact_decision", {})).get("original_decision_before_protection", trajectory.get("fact_decision_status", "Discard"))),
        "final_decision": final_decision,
        "rejection_reasons_before_repair": reasons,
        "rejection_reasons_after_repair": remaining,
        "attempted_strategies": [{key: value for key, value in row.items() if key != "repaired_observations"} for row in candidates],
        "accepted_strategy": [stage["strategy"] for stage in selected["strategies"]] if selected else [],
        "candidate_rejection_reason": "" if selected else "no_candidate_met_improvement_and_safety_constraints",
        "modified_frame_ids": selected["modified_frame_ids"] if selected else [],
        "before_signals": _snapshot(original),
        "after_signals": _snapshot(final_observations),
        "validation_metrics": {
            "before": original_eval,
            "after": final_eval,
            "score_before": _issue_cost(original_eval),
            "score_after": _issue_cost(final_eval),
            "required_improvement_margin": _IMPROVEMENT_MARGIN,
        },
        "repair_confidence": float(confidence),
        "remaining_uncertainty": dict(final_eval.get("uncertainty", {})),
        "diagnostic_only": True,
    }


def run_adaptive_motion_repair(
    relative_motion_state: Dict[str, Any],
    output_root: Path,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_videos = list(relative_motion_state.get("trajectory_motion_evidence", []))
    refined_by_video = {
        str(row.get("video_id", "")): row
        for row in relative_motion_state.get("refined_ego_motion", [])
    }
    calibration = _calibrate(evidence_videos, refined_by_video)
    with (output_root / "repair_strategy_calibration.json").open("w", encoding="utf-8") as file:
        json.dump(calibration, file, indent=2)
    video_results = []
    all_tracks = []
    for evidence_video in evidence_videos:
        video_id = str(evidence_video.get("video_id", ""))
        queued = []
        for trajectory in evidence_video.get("trajectory_motion_evidence", []):
            fact = dict(trajectory.get("fact_decision", {}))
            original_decision = str(fact.get("original_decision_before_protection", trajectory.get("fact_decision_status", "")))
            if bool(trajectory.get("symbol_grounded_protected", False)) and original_decision == "Discard":
                queued.append(trajectory)
        tracks = [
            _repair_track(trajectory, refined_by_video.get(video_id, {}), calibration)
            for trajectory in queued
        ]
        result = {
            "version": _VERSION,
            "video_id": video_id,
            "status": "diagnostic_completed",
            "diagnostic_only": True,
            "source_step8b_preserved": True,
            "num_queued": len(queued),
            "num_attempted": sum(len(row.get("attempted_strategies", [])) for row in tracks),
            "num_repaired": sum(row.get("final_decision") == "Repair" for row in tracks),
            "num_uncertain": sum(row.get("final_decision") == "Keep_with_uncertainty" for row in tracks),
            "num_unrepairable": sum(row.get("final_decision") == "Unrepairable_but_semantically_retained" for row in tracks),
            "tracks": tracks,
        }
        video_dir = output_root / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        with (video_dir / "adaptive_motion_repair.json").open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
        video_results.append(result)
        all_tracks.extend({"video_id": video_id, **row} for row in tracks)
    manifest = {
        "version": _VERSION,
        "method": "adaptive_protected_object_motion_repair",
        "diagnostic_only": True,
        "num_videos": len(video_results),
        "queued": sum(row["num_queued"] for row in video_results),
        "attempted": sum(row["num_attempted"] for row in video_results),
        "repaired": sum(row["num_repaired"] for row in video_results),
        "uncertain": sum(row["num_uncertain"] for row in video_results),
        "unrepairable": sum(row["num_unrepairable"] for row in video_results),
        "calibration": calibration,
        "videos": video_results,
    }
    with (output_root / "adaptive_motion_repair_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return {
        **relative_motion_state,
        "adaptive_motion_repair": video_results,
        "adaptive_motion_repair_tracks": all_tracks,
        "adaptive_motion_repair_calibration": calibration,
        "adaptive_motion_repair_manifest": manifest,
        "adaptive_motion_repair_output_root": output_root,
    }
