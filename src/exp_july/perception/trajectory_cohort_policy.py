"""Prior-guided statistical cohort policy for Step 8C.

The LLM is restricted to proposing interpretable static-metadata rules and
choosing an operator family. Numeric calibration, anomaly gating, validation,
and repair execution are deterministic.
"""

from __future__ import annotations

import copy
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.exp_july.perception.adaptive_motion_repair import (
    _apply_strategy,
    _evaluate,
    _issue_cost,
    _modified_frames,
    _recompute_motion,
)


POLICY_VERSION = 1
ALLOWED_ATTRIBUTES = {
    "category",
    "bbox_size_bucket",
    "image_location",
    "track_length_bucket",
    "confidence_bucket",
    "source_kind",
    "track_length",
    "bbox_area_mean",
    "bbox_center_x_mean",
    "detection_confidence_mean",
    "repaired_ratio",
}
ALLOWED_CONDITION_OPERATORS = {"eq", "neq", "in", "not_in", "lt", "lte", "gt", "gte"}
ANOMALY_TYPES = {
    "id_switch",
    "trajectory_discontinuity",
    "track_drift",
    "bbox_jump",
    "depth_jump",
    "speed_abnormal_change",
    "motion_direction_abrupt_change",
}
DEFAULT_RULES = [
    {
        "rule_id": "traffic_control_and_vulnerable",
        "description": "Traffic-control and vulnerable-road-user tracks.",
        "priority": 100,
        "all": [
            {
                "attribute": "category",
                "operator": "in",
                "value": [
                    "traffic light", "traffic sign", "stop sign", "person",
                    "pedestrian", "bicycle", "cyclist", "motorcycle",
                ],
            }
        ],
    },
    {
        "rule_id": "short_small_vehicle",
        "description": "Short vehicle tracks with small image support.",
        "priority": 80,
        "all": [
            {"attribute": "category", "operator": "in", "value": ["car", "truck", "bus", "van", "vehicle"]},
            {"attribute": "track_length_bucket", "operator": "eq", "value": "short"},
            {"attribute": "bbox_size_bucket", "operator": "in", "value": ["tiny", "small"]},
        ],
    },
    {
        "rule_id": "persistent_vehicle",
        "description": "Medium or long vehicle trajectories.",
        "priority": 60,
        "all": [
            {"attribute": "category", "operator": "in", "value": ["car", "truck", "bus", "van", "vehicle"]},
            {"attribute": "track_length_bucket", "operator": "in", "value": ["medium", "long"]},
        ],
    },
    {
        "rule_id": "other_tracks",
        "description": "Fallback cohort for tracks not covered above.",
        "priority": 0,
        "all": [],
    },
]


def _f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    rows = [_f(value) for value in values]
    return sum(rows) / max(1, len(rows))


def _quantile(values: Iterable[float], q: float) -> float:
    rows = sorted(_f(value) for value in values)
    if not rows:
        return 0.0
    position = max(0.0, min(1.0, q)) * (len(rows) - 1)
    left = int(position)
    right = min(len(rows) - 1, left + 1)
    ratio = position - left
    return rows[left] * (1.0 - ratio) + rows[right] * ratio


def _normalized_label(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def _bucket(value: float, low: float, high: float, labels: Sequence[str]) -> str:
    if value <= low:
        return labels[0]
    if value <= high:
        return labels[1]
    return labels[2]


def _track_base_metadata(track: Mapping[str, Any]) -> Dict[str, Any]:
    statistics_row = dict(track.get("trajectory_statistics", {}))
    observations = list(track.get("observations", []))
    bbox_areas = []
    bbox_centers = []
    confidences = []
    source_counts = Counter()
    for observation in observations:
        bbox = list(observation.get("bbox", []))
        if len(bbox) >= 4:
            width = max(0.0, _f(bbox[2]) - _f(bbox[0]))
            height = max(0.0, _f(bbox[3]) - _f(bbox[1]))
            bbox_areas.append(width * height)
            bbox_centers.append((_f(bbox[0]) + _f(bbox[2])) / 2.0)
        confidences.append(_f(dict(observation.get("uncertainty", {})).get("score", 0.0)))
        source_counts[str(dict(observation.get("provenance", {})).get("source", "observed"))] += 1
    track_length = len(observations)
    return {
        "category": _normalized_label(track.get("object_class", "unknown")),
        "track_length": track_length,
        "bbox_area_mean": _mean(bbox_areas),
        "bbox_center_x_mean": _mean(bbox_centers),
        "detection_confidence_mean": _mean(confidences),
        "repaired_ratio": _f(statistics_row.get("repaired_ratio", 0.0)),
        "source_kind": source_counts.most_common(1)[0][0] if source_counts else "unknown",
    }


def attach_static_metadata(
    tracks: Sequence[Dict[str, Any]],
    *,
    bucket_boundaries: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Attach static-only attributes using fitted or supplied image buckets.

    Supplying ``bucket_boundaries`` applies a previously fitted transform. It
    is used by the August strict-holdout path so eval/test tracks cannot change
    quantiles fitted on the training split.
    """
    base_rows = [_track_base_metadata(track) for track in tracks]
    area_rows = [row["bbox_area_mean"] for row in base_rows]
    center_rows = [row["bbox_center_x_mean"] for row in base_rows]
    fitted_boundaries = dict(bucket_boundaries or {})
    area_q25 = _f(fitted_boundaries.get("bbox_area_q25"), _quantile(area_rows, 0.25))
    area_q75 = _f(fitted_boundaries.get("bbox_area_q75"), _quantile(area_rows, 0.75))
    center_q33 = _f(
        fitted_boundaries.get("bbox_center_x_q33"),
        _quantile(center_rows, 1.0 / 3.0),
    )
    center_q67 = _f(
        fitted_boundaries.get("bbox_center_x_q67"),
        _quantile(center_rows, 2.0 / 3.0),
    )
    for track, metadata in zip(tracks, base_rows):
        length = int(metadata["track_length"])
        metadata.update(
            {
                "track_length_bucket": "short" if length <= 3 else "medium" if length <= 10 else "long",
                "bbox_size_bucket": (
                    "tiny"
                    if metadata["bbox_area_mean"] <= max(64.0, area_q25 * 0.5)
                    else _bucket(metadata["bbox_area_mean"], area_q25, area_q75, ("small", "medium", "large"))
                ),
                "image_location": _bucket(
                    metadata["bbox_center_x_mean"], center_q33, center_q67, ("left", "center", "right")
                ),
                "confidence_bucket": (
                    "low"
                    if metadata["detection_confidence_mean"] < 0.40
                    else "medium"
                    if metadata["detection_confidence_mean"] < 0.70
                    else "high"
                ),
            }
        )
        track["static_metadata"] = metadata
    catalog = {
        "track_count": len(tracks),
        "categories": dict(sorted(Counter(row["category"] for row in base_rows).items())),
        "attribute_schema": {
            "category": "categorical",
            "bbox_size_bucket": ["tiny", "small", "medium", "large"],
            "image_location": ["left", "center", "right"],
            "track_length_bucket": ["short", "medium", "long"],
            "confidence_bucket": ["low", "medium", "high"],
            "source_kind": sorted({row["source_kind"] for row in base_rows}),
        },
        "numeric_ranges": {
            key: {
                "min": min((_f(row[key]) for row in base_rows), default=0.0),
                "median": _quantile((_f(row[key]) for row in base_rows), 0.5),
                "max": max((_f(row[key]) for row in base_rows), default=0.0),
            }
            for key in (
                "track_length", "bbox_area_mean", "bbox_center_x_mean",
                "detection_confidence_mean", "repaired_ratio",
            )
        },
        "bucket_boundaries": {
            "bbox_area_q25": area_q25,
            "bbox_area_q75": area_q75,
            "bbox_center_x_q33": center_q33,
            "bbox_center_x_q67": center_q67,
        },
    }
    return catalog


def rule_generation_prompt(dataset: str, catalog: Mapping[str, Any]) -> str:
    return (
        "Generate interpretable, ordered trajectory cohort rules from STATIC metadata only. "
        "Do not use position trends, velocity, acceleration, motion labels, residuals, repairs, "
        "threshold outcomes, or corrected values. Every rule must describe a semantically coherent "
        "cohort and use only the supplied attribute names and condition operators. Include a final "
        "catch-all rule with an empty all list. Return JSON: "
        '{"rules":[{"rule_id":"short_small_vehicle","description":"...",'
        '"priority":80,"all":[{"attribute":"category","operator":"in",'
        '"value":["car"]}]}],"rationale":"..."}. '
        f"allowed_attributes={json.dumps(sorted(ALLOWED_ATTRIBUTES))}; "
        f"allowed_operators={json.dumps(sorted(ALLOWED_CONDITION_OPERATORS))}; "
        f"dataset={dataset}; metadata_catalog={json.dumps(catalog, separators=(',', ':'))}"
    )


def _safe_rule_id(value: Any) -> str:
    text = "_".join(_normalized_label(value).split())
    return "".join(character for character in text if character.isalnum() or character == "_")[:80]


def compile_rules(raw: Mapping[str, Any]) -> Dict[str, Any]:
    compiled = []
    errors = []
    for index, source in enumerate(raw.get("rules", []) if isinstance(raw, Mapping) else []):
        if not isinstance(source, Mapping):
            errors.append(f"rule_{index}:not_object")
            continue
        rule_id = _safe_rule_id(source.get("rule_id", f"rule_{index}"))
        if not rule_id or any(row["rule_id"] == rule_id for row in compiled):
            errors.append(f"rule_{index}:invalid_or_duplicate_id")
            continue
        conditions = []
        valid = True
        for condition_index, condition in enumerate(source.get("all", [])):
            if not isinstance(condition, Mapping):
                valid = False
                errors.append(f"{rule_id}.condition_{condition_index}:not_object")
                break
            attribute = str(condition.get("attribute", ""))
            operator = str(condition.get("operator", ""))
            value = condition.get("value")
            if attribute not in ALLOWED_ATTRIBUTES or operator not in ALLOWED_CONDITION_OPERATORS:
                valid = False
                errors.append(f"{rule_id}.condition_{condition_index}:unsupported")
                break
            if operator in {"in", "not_in"} and not isinstance(value, list):
                valid = False
                errors.append(f"{rule_id}.condition_{condition_index}:list_required")
                break
            if attribute == "category":
                value = (
                    [_normalized_label(item) for item in value]
                    if isinstance(value, list)
                    else _normalized_label(value)
                )
            conditions.append({"attribute": attribute, "operator": operator, "value": value})
        if valid:
            compiled.append(
                {
                    "rule_id": rule_id,
                    "description": str(source.get("description", ""))[:500],
                    "priority": int(source.get("priority", 0) or 0),
                    "all": conditions,
                    "source": "llm_static_metadata_rule",
                }
            )
    if not compiled:
        compiled = copy.deepcopy(DEFAULT_RULES)
        errors.append("no_valid_llm_rules:deterministic_fallback")
    if not any(not row["all"] for row in compiled):
        compiled.append(copy.deepcopy(DEFAULT_RULES[-1]))
        errors.append("catch_all_rule_added")
    compiled.sort(key=lambda row: (-int(row["priority"]), row["rule_id"]))
    return {
        "version": POLICY_VERSION,
        "rules": compiled,
        "compile_errors": errors,
        "llm_rationale": str(raw.get("rationale", ""))[:1000] if isinstance(raw, Mapping) else "",
    }


def _condition_matches(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "eq":
        return actual == expected
    if operator == "neq":
        return actual != expected
    if operator == "in":
        return actual in expected
    if operator == "not_in":
        return actual not in expected
    left, right = _f(actual), _f(expected)
    return {
        "lt": left < right,
        "lte": left <= right,
        "gt": left > right,
        "gte": left >= right,
    }.get(operator, False)


def assign_cohorts(tracks: Sequence[Dict[str, Any]], rules: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    cohorts = defaultdict(list)
    for track in tracks:
        metadata = dict(track.get("static_metadata", {}))
        matched = next(
            (
                rule
                for rule in rules
                if all(
                    _condition_matches(metadata.get(row["attribute"]), row["operator"], row["value"])
                    for row in rule.get("all", [])
                )
            ),
            rules[-1],
        )
        track["activated_cohort_rule"] = copy.deepcopy(matched)
        track["cohort_id"] = str(matched["rule_id"])
        cohorts[str(matched["rule_id"])].append(track)
    return dict(cohorts)


def _numeric_summary(values: Iterable[float]) -> Dict[str, float]:
    rows = [_f(value) for value in values]
    return {
        "count": len(rows),
        "mean": _mean(rows),
        "std": statistics.pstdev(rows) if len(rows) > 1 else 0.0,
        "q05": _quantile(rows, 0.05),
        "q50": _quantile(rows, 0.50),
        "q95": _quantile(rows, 0.95),
    }


def cohort_statistics(
    cohorts: Mapping[str, Sequence[Dict[str, Any]]],
    validation_thresholds: Mapping[str, Any] | None,
    systematic_anomaly_rate: float = 0.20,
) -> Dict[str, Dict[str, Any]]:
    systematic_anomaly_rate = max(
        0.05, min(0.80, _f(systematic_anomaly_rate, 0.20))
    )
    summaries = {}
    for cohort_id, tracks in sorted(cohorts.items()):
        evaluations = []
        rejection_counts = Counter()
        for track in tracks:
            num_frames = int(dict(track.get("trajectory_statistics", {})).get("video_num_frames", len(track.get("observations", []))))
            evaluation = _evaluate(track.get("observations", []), num_frames, thresholds=validation_thresholds)
            reasons = list(evaluation["validation"].get("rejection_reasons", []))
            rejection_counts.update(reasons)
            evaluations.append(evaluation)
            track["pre_repair_evaluation"] = evaluation
            track["cohort_anomaly_reasons"] = reasons
        size = len(tracks)
        summaries[cohort_id] = {
            "cohort_id": cohort_id,
            "activated_rule": copy.deepcopy(tracks[0].get("activated_cohort_rule", {})) if tracks else {},
            "track_count": size,
            "video_count": len({str(track.get("video_id", "")) for track in tracks}),
            "category_counts": dict(sorted(Counter(dict(track.get("static_metadata", {})).get("category", "unknown") for track in tracks).items())),
            "static_metadata": {
                key: _numeric_summary(dict(track.get("static_metadata", {})).get(key, 0.0) for track in tracks)
                for key in ("track_length", "bbox_area_mean", "bbox_center_x_mean", "detection_confidence_mean", "repaired_ratio")
            },
            "motion_statistics": {
                "relative_speed_mean": _numeric_summary(dict(track.get("trajectory_statistics", {})).get("rel_speed", {}).get("mean", 0.0) for track in tracks),
                "relative_speed_max": _numeric_summary(dict(track.get("trajectory_statistics", {})).get("rel_speed", {}).get("max", 0.0) for track in tracks),
                "depth_delta": _numeric_summary(dict(track.get("trajectory_statistics", {})).get("position_z_depth", {}).get("delta", 0.0) for track in tracks),
                "path_length_xz": _numeric_summary(dict(track.get("trajectory_statistics", {})).get("path_length_xz", 0.0) for track in tracks),
            },
            "validation_status_counts": dict(sorted(Counter(str(row["validation"].get("validation_status", "unknown")) for row in evaluations).items())),
            "anomaly_counts": dict(sorted(rejection_counts.items())),
            "anomaly_rates": {key: count / max(1, size) for key, count in sorted(rejection_counts.items())},
            "systematic_anomaly_rate_threshold": systematic_anomaly_rate,
            "systematic_anomalies": [
                key
                for key, count in sorted(rejection_counts.items())
                if count / max(1, size) >= systematic_anomaly_rate
            ],
        }
    return summaries


def repair_selection_prompt(
    cohort_summaries: Mapping[str, Any],
    operator_library: Mapping[str, Any],
) -> str:
    public_library = {
        name: {"executor": row["executor"], "default_parameters": row["parameters"]}
        for name, row in operator_library.items()
    }
    return (
        "For each trajectory cohort, select at most one repair operator from the predefined library "
        "and propose initial parameters. The selection must be justified only by the aggregated cohort "
        "statistics and systematic anomaly types. Do not output corrected trajectories, per-track "
        "decisions, numeric signal values, new operators, or final thresholds. Deterministic code will "
        "calibrate parameters and decide whether each track actually needs repair. Return JSON: "
        '{"plans":[{"cohort_id":"...","operator":"outlier_removal",'
        '"initial_parameters":{"median_radius":2,"mad_scale":3.0},'
        '"anomaly_types":["depth_jump"],"rationale":"..."}]}. '
        f"operator_library={json.dumps(public_library, separators=(',', ':'))}; "
        f"cohort_summaries={json.dumps(cohort_summaries, separators=(',', ':'))}"
    )


def _bounded_parameters(operator: str, supplied: Mapping[str, Any], defaults: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(defaults)
    result.update({key: value for key, value in supplied.items() if key in defaults})
    if operator == "kalman_smoothing":
        result["alpha"] = max(0.20, min(0.90, _f(result.get("alpha", 0.55))))
    elif operator in {"outlier_removal", "depth_refinement"}:
        result["median_radius"] = max(1, min(4, int(result.get("median_radius", 2))))
        result["mad_scale"] = max(1.5, min(5.0, _f(result.get("mad_scale", 3.0))))
    elif operator == "interpolation":
        result["maximum_gap"] = max(2, min(20, int(result.get("maximum_gap", 8))))
    elif operator == "motion_recomputation":
        result["window"] = max(2, min(6, int(result.get("window", 3))))
    return result


def compile_operator_plans(
    raw: Mapping[str, Any],
    cohort_summaries: Mapping[str, Any],
    operator_library: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    supplied = {
        str(row.get("cohort_id", "")): row
        for row in raw.get("plans", [])
        if isinstance(row, Mapping)
    }
    plans = {}
    for cohort_id, summary in sorted(cohort_summaries.items()):
        source = supplied.get(cohort_id, {})
        requested_operator = str(source.get("operator", ""))
        operator = requested_operator
        if operator not in operator_library:
            operator = "outlier_removal" if summary.get("systematic_anomalies") else "no_repair"
        systematic_anomalies = {
            str(value)
            for value in summary.get("systematic_anomalies", [])
            if str(value) in ANOMALY_TYPES
        }
        if not systematic_anomalies:
            operator = "no_repair"
        library_row = operator_library[operator]
        anomaly_types = [
            str(value)
            for value in source.get("anomaly_types", summary.get("systematic_anomalies", []))
            if str(value) in systematic_anomalies
        ]
        if not anomaly_types:
            anomaly_types = [
                str(value)
                for value in summary.get("systematic_anomalies", [])
                if str(value) in ANOMALY_TYPES
            ]
        if operator == "no_repair":
            anomaly_types = []
        plans[cohort_id] = {
            "cohort_id": cohort_id,
            "operator": operator,
            "llm_requested_operator": requested_operator,
            "executor": library_row["executor"],
            "initial_parameters": _bounded_parameters(
                operator,
                dict(source.get("initial_parameters", {})),
                library_row["parameters"],
            ),
            "anomaly_types": anomaly_types,
            "llm_rationale": str(source.get("rationale", ""))[:1000],
            "selection_source": (
                "deterministic_no_systematic_anomaly"
                if operator == "no_repair" and requested_operator != "no_repair"
                else "llm_cohort_statistics"
                if source
                else "deterministic_safe_default"
            ),
        }
    return plans


def _parameter_variants(operator: str, initial: Mapping[str, Any]) -> List[Dict[str, Any]]:
    variants = [dict(initial)]
    if operator == "kalman_smoothing":
        variants.extend({**initial, "alpha": value} for value in (0.35, 0.55, 0.75))
    elif operator in {"outlier_removal", "depth_refinement"}:
        variants.extend({**initial, "mad_scale": value} for value in (2.0, 2.5, 3.0, 4.0))
    elif operator == "interpolation":
        variants.extend({**initial, "maximum_gap": value} for value in (4, 8, 12))
    elif operator == "motion_recomputation":
        variants.extend({**initial, "window": value} for value in (2, 3, 4))
    unique = {}
    for row in variants:
        unique[json.dumps(row, sort_keys=True)] = row
    return [unique[key] for key in sorted(unique)]


def _apply_operator(
    observations: Sequence[Dict[str, Any]],
    operator: str,
    executor: str,
    parameters: Mapping[str, Any],
    ego: Mapping[int, Any],
) -> List[Dict[str, Any]]:
    if operator == "no_repair":
        return copy.deepcopy(list(observations))
    repaired = _apply_strategy(observations, executor, dict(parameters))
    return _recompute_motion(
        repaired,
        dict(ego),
        velocity_window=int(parameters.get("window", 1)),
    )


def calibrate_operator_plans(
    plans: Mapping[str, Dict[str, Any]],
    cohorts: Mapping[str, Sequence[Dict[str, Any]]],
    ego_by_video: Mapping[str, Mapping[int, Any]],
    validation_thresholds: Mapping[str, Any] | None,
    downstream_feedback: Mapping[str, Any] | None = None,
    calibration_video_ids: Sequence[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    downstream_feedback = dict(downstream_feedback or {})
    calibration_video_set = {
        str(value) for value in (calibration_video_ids or [])
    }
    calibrated = {}
    for cohort_id, source_plan in sorted(plans.items()):
        plan = copy.deepcopy(source_plan)
        all_tracks = list(cohorts.get(cohort_id, []))
        tracks = (
            [
                track for track in all_tracks
                if str(track.get("video_id", "")) in calibration_video_set
            ]
            if calibration_video_set
            else all_tracks
        )
        variants = _parameter_variants(plan["operator"], plan["initial_parameters"])
        measurements = []
        for parameters in variants:
            rows = []
            for track in tracks:
                reasons = set(track.get("cohort_anomaly_reasons", []))
                triggered = bool(reasons & set(plan.get("anomaly_types", [])))
                if not triggered:
                    continue
                original = list(track.get("observations", []))
                num_frames = int(dict(track.get("trajectory_statistics", {})).get("video_num_frames", len(original)))
                before = dict(track.get("pre_repair_evaluation", {})) or _evaluate(original, num_frames, thresholds=validation_thresholds)
                repaired = _apply_operator(
                    original,
                    plan["operator"],
                    plan["executor"],
                    parameters,
                    ego_by_video.get(str(track.get("video_id", "")), {}),
                )
                after = _evaluate(repaired, num_frames, thresholds=validation_thresholds)
                new_anomalies = set(after["validation"].get("rejection_reasons", [])) - reasons
                retention = len({int(row.get("frame_index", -1)) for row in repaired} & {int(row.get("frame_index", -1)) for row in original}) / max(1, len(original))
                improvement = _issue_cost(before) - _issue_cost(after)
                success = improvement > 0.0 and retention >= 0.95 and not new_anomalies
                rows.append(
                    {
                        "success": success,
                        "issue_cost_improvement": improvement,
                        "retention": retention,
                        "modified_frames": len(_modified_frames(original, repaired)),
                    }
                )
            success_rate = sum(bool(row["success"]) for row in rows) / max(1, len(rows))
            mean_improvement = _mean(row["issue_cost_improvement"] for row in rows)
            mean_retention = _mean(row["retention"] for row in rows) if rows else 1.0
            modification_rate = _mean(row["modified_frames"] for row in rows)
            prior = dict(downstream_feedback.get("cohorts", {})).get(cohort_id, {})
            prior_operators = dict(prior.get("operators", {}))
            same_operator_feedback = plan["operator"] in prior_operators
            critical_regressions = (
                int(prior.get("critical_regressions", 0))
                if same_operator_feedback
                else 0
            )
            downstream_score = _f(prior.get("downstream_success_rate", 0.5), 0.5)
            score = (
                2.0 * success_rate
                + mean_improvement
                + 0.25 * downstream_score
                - 0.01 * modification_rate
                - 10.0 * critical_regressions
            )
            measurements.append(
                {
                    "parameters": parameters,
                    "sample_count": len(rows),
                    "success_rate": success_rate,
                    "mean_issue_cost_improvement": mean_improvement,
                    "mean_observation_retention": mean_retention,
                    "mean_modified_frames": modification_rate,
                    "downstream_success_rate": downstream_score,
                    "critical_regressions": critical_regressions,
                    "same_operator_feedback": same_operator_feedback,
                    "score": score,
                }
            )
        selected = max(
            measurements,
            key=lambda row: (
                row["critical_regressions"] == 0,
                row["success_rate"],
                row["score"],
                json.dumps(row["parameters"], sort_keys=True),
            ),
            default={
                "parameters": plan["initial_parameters"],
                "sample_count": 0,
                "success_rate": 0.0,
                "mean_issue_cost_improvement": 0.0,
                "mean_observation_retention": 1.0,
                "mean_modified_frames": 0.0,
                "downstream_success_rate": 0.5,
                "critical_regressions": 0,
                "same_operator_feedback": False,
                "score": 0.0,
            },
        )
        rollback_due_to_critical_regression = bool(
            selected["same_operator_feedback"]
            and selected["critical_regressions"] > 0
        )
        rollback_due_to_validation = bool(
            plan["operator"] != "no_repair"
            and (
                selected["sample_count"] == 0
                or selected["success_rate"] <= 0.0
                or selected["mean_issue_cost_improvement"] <= 0.0
                or selected["mean_observation_retention"] < 0.95
            )
        )
        rollback_to_no_repair = (
            rollback_due_to_critical_regression or rollback_due_to_validation
        )
        if rollback_to_no_repair:
            plan.update(
                {
                    "operator": "no_repair",
                    "executor": "identity",
                    "anomaly_types": [],
                }
            )
        plan.update(
            {
                "calibrated_parameters": (
                    {} if rollback_to_no_repair
                    else dict(selected["parameters"])
                ),
                "calibration": {
                    "selection_metric": "validation_success+issue_reduction+prior_downstream_performance",
                    "validation_video_ids": sorted(calibration_video_set),
                    "independent_validation_split": bool(calibration_video_set),
                    "cohort_track_count": len(all_tracks),
                    "calibration_track_count": len(tracks),
                    "candidate_measurements": measurements,
                    "selected_measurement": selected,
                    "promotion_decision": (
                        "rollback_to_no_repair"
                        if rollback_to_no_repair
                        else "no_repair_required"
                        if plan["operator"] == "no_repair"
                        else "accept_calibrated_parameters"
                    ),
                    "promotion_reason": (
                        "prior_downstream_critical_regression"
                        if rollback_due_to_critical_regression
                        else "insufficient_or_non_improving_validation_evidence"
                        if rollback_due_to_validation
                        else "cohort_has_no_systematic_repair_trigger"
                        if plan["operator"] == "no_repair"
                        else "quality_improved_without_critical_regression"
                    ),
                    "downstream_feedback_source": (
                        "previous_epoch"
                        if dict(downstream_feedback.get("cohorts", {})).get(cohort_id)
                        else "trajectory_validation_proxy"
                    ),
                },
                "apply_policy": (
                    "matching_rule_and_deterministic_anomaly_trigger_and_hard_validation"
                ),
            }
        )
        calibrated[cohort_id] = plan
    return calibrated


def operator_library(executors: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    library = {
        name: {"executor": executor, "parameters": dict(parameters)}
        for name, (executor, parameters) in executors.items()
    }
    library["no_repair"] = {"executor": "identity", "parameters": {}}
    return library


def write_downstream_feedback(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Persist final Step 8 outcomes for the next Step 8C calibration epoch."""
    records = list(state.get("trajectory_pattern_records", []))
    manifest = dict(state.get("trajectory_pattern_manifest", {}))
    validation_video_ids = {
        str(value)
        for value in manifest.get("statistics_validation_video_ids", [])
    }
    if validation_video_ids:
        records = [
            record for record in records
            if str(record.get("video_id", "")) in validation_video_ids
        ]
    final_evidence = {}
    for video in state.get("trajectory_motion_evidence", []):
        video_id = str(video.get("video_id", ""))
        for row in video.get("trajectory_motion_evidence", []):
            final_evidence[
                (video_id, _safe_track_id(row.get("track_id", -1)))
            ] = row
    protected = {
        (str(row.get("video_id", "")), _safe_track_id(row.get("track_id", -1)))
        for row in state.get("protected_objects", [])
        if _safe_track_id(row.get("track_id", -1)) >= 0
    }
    cohort_rows = defaultdict(list)
    for record in records:
        key = (
            str(record.get("video_id", "")),
            _safe_track_id(record.get("track_id", -1)),
        )
        evidence = dict(final_evidence.get(key, {}))
        causal_validation = evidence.get("causal_motion_fact_validation", {})
        if not isinstance(causal_validation, Mapping):
            causal_validation = {}
        validation_status = str(
            evidence.get(
                "validation_status",
                causal_validation.get(
                    "validation_status", record.get("final_validation_status", "")
                ),
            )
        )
        decision = str(evidence.get("fact_decision_status", ""))
        is_protected = key in protected
        critical_regression = is_protected and (
            validation_status == "invalid" or decision == "Discard"
        )
        success = validation_status != "invalid" and decision != "Discard"
        cohort_rows[str(record.get("trajectory_cohort_id", "unknown"))].append(
            {
                "video_id": key[0],
                "track_id": key[1],
                "operator": str(
                    dict(record.get("cohort_operator_plan", {})).get(
                        "operator", "no_repair"
                    )
                ),
                "repair_applied": bool(record.get("repair_applied", False)),
                "validation_status": validation_status,
                "fact_decision_status": decision,
                "semantic_protected": is_protected,
                "success": success,
                "critical_regression": critical_regression,
            }
        )
    cohorts = {}
    for cohort_id, rows in sorted(cohort_rows.items()):
        cohorts[cohort_id] = {
            "sample_count": len(rows),
            "downstream_success_rate": sum(bool(row["success"]) for row in rows)
            / max(1, len(rows)),
            "critical_regressions": sum(
                bool(row["critical_regression"]) for row in rows
            ),
            "semantic_protected_count": sum(
                bool(row["semantic_protected"]) for row in rows
            ),
            "repair_applied_count": sum(
                bool(row["repair_applied"]) for row in rows
            ),
            "operators": dict(
                sorted(Counter(row["operator"] for row in rows).items())
            ),
            "tracks": rows,
        }
    feedback = {
        "version": POLICY_VERSION,
        "source": "steps_8d_through_8i_final_outcomes",
        "validation_video_ids": sorted(validation_video_ids),
        "independent_validation_split": bool(validation_video_ids),
        "cohorts": cohorts,
    }
    output_root = state.get("trajectory_pattern_output_root")
    if output_root:
        path = Path(output_root) / "policies" / "downstream_feedback.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(feedback, indent=2), encoding="utf-8")
        feedback["path"] = str(path)
    return feedback


def _safe_track_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1
