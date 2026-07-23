"""Epoch-frozen, weakly supervised calibration of trajectory thresholds."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


SCHEMA_VERSION = 1
POLICY_FILENAME = "active_threshold_policy.json"
PENDING_FILENAME = "pending_threshold_policy.json"

# These are soft, continuous limits. Identity, continuity, and direction-change
# checks remain hard and cannot be changed by semantic supervision.
CALIBRATABLE_THRESHOLDS = {
    "max_invalid_center_step_diag_ratio": {
        "issue": "track_drift",
        "metric": "max_bbox_center_step_diag_ratio",
    },
    "max_invalid_bbox_size_ratio": {
        "issue": "bbox_jump",
        "metric": "max_bbox_size_ratio",
    },
    "max_invalid_depth_step_per_frame": {
        "issue": "depth_jump",
        "metric": "max_depth_step_per_frame",
    },
    "max_invalid_rel_velocity_delta": {
        "issue": "speed_abnormal_change",
        "metric": "max_rel_velocity_delta",
    },
    "max_invalid_rel_speed": {
        "issue": "speed_abnormal_change",
        "metric": "max_rel_speed",
    },
}
HARD_INVALID_REASONS = {
    "physical_invalidity",
    "id_switch",
    "trajectory_discontinuity",
    "motion_direction_abrupt_change",
}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(_f(value) for value in values)
    if not ordered:
        return 0.0
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def default_policy(default_thresholds: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "version": 1,
        "status": "active",
        "parent_version": None,
        "thresholds": copy.deepcopy(default_thresholds),
        "calibratable_thresholds": sorted(CALIBRATABLE_THRESHOLDS),
        "hard_invalid_reasons": sorted(HARD_INVALID_REASONS),
        "calibration": {
            "source": "defaults",
            "weak_label": "symbol_grounded_protected",
        },
    }


def validate_policy(
    policy: Dict[str, Any],
    default_thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        raise ValueError("threshold policy must be an object")
    thresholds = policy.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(default_thresholds):
        raise ValueError("threshold policy must cover the fixed threshold schema")
    normalized = copy.deepcopy(policy)
    normalized["schema_version"] = SCHEMA_VERSION
    normalized["version"] = max(1, int(policy.get("version", 1)))
    normalized["thresholds"] = {}
    for name, default in default_thresholds.items():
        try:
            value = float(thresholds.get(name))
        except (TypeError, ValueError):
            raise ValueError(f"invalid threshold value: {name}") from None
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid threshold value: {name}")
        if name not in CALIBRATABLE_THRESHOLDS and value != _f(default):
            raise ValueError(f"hard/non-calibratable threshold changed: {name}")
        if name in CALIBRATABLE_THRESHOLDS and not (
            _f(default) <= value <= 2.0 * _f(default)
        ):
            raise ValueError(f"calibratable threshold outside safety bounds: {name}")
        normalized["thresholds"][name] = int(round(value)) if isinstance(default, int) else float(value)
    pairs = (
        ("max_uncertain_center_step_diag_ratio", "max_invalid_center_step_diag_ratio"),
        ("max_uncertain_bbox_size_ratio", "max_invalid_bbox_size_ratio"),
        ("max_uncertain_depth_step_per_frame", "max_invalid_depth_step_per_frame"),
        ("max_uncertain_rel_velocity_delta", "max_invalid_rel_velocity_delta"),
        ("max_uncertain_rel_speed", "max_invalid_rel_speed"),
    )
    for uncertain, invalid in pairs:
        if _f(normalized["thresholds"][uncertain]) >= _f(normalized["thresholds"][invalid]):
            raise ValueError(f"{uncertain} must remain below {invalid}")
    if normalized["thresholds"]["max_uncertain_frame_gap"] > normalized["thresholds"]["max_valid_frame_gap"]:
        raise ValueError("uncertain frame-gap threshold must not exceed invalid frame-gap threshold")
    normalized["calibratable_thresholds"] = sorted(CALIBRATABLE_THRESHOLDS)
    normalized["hard_invalid_reasons"] = sorted(HARD_INVALID_REASONS)
    return normalized


def begin_threshold_epoch(
    policy_root: Path,
    default_thresholds: Dict[str, Any],
) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    """Activate a pending policy at an epoch boundary and freeze its snapshot."""
    root = Path(policy_root)
    root.mkdir(parents=True, exist_ok=True)
    active_path = root / POLICY_FILENAME
    pending_path = root / PENDING_FILENAME
    activated = False
    if pending_path.exists():
        pending = validate_policy(
            json.loads(pending_path.read_text(encoding="utf-8")),
            default_thresholds,
        )
        pending["status"] = "active"
        active_path.write_text(json.dumps(pending, indent=2), encoding="utf-8")
        pending_path.unlink()
        activated = True
    if active_path.exists():
        active = validate_policy(
            json.loads(active_path.read_text(encoding="utf-8")),
            default_thresholds,
        )
    else:
        active = default_policy(default_thresholds)
        active_path.write_text(json.dumps(active, indent=2), encoding="utf-8")
    epoch_paths = sorted(root.glob("threshold_epoch_*.json"))
    epoch_id = len(epoch_paths) + 1
    frozen = copy.deepcopy(active)
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "epoch_id": epoch_id,
        "status": "processing",
        "policy_frozen": True,
        "activated_pending_policy": activated,
        "policy": frozen,
    }
    (root / f"threshold_epoch_{epoch_id:04d}.json").write_text(
        json.dumps(snapshot, indent=2),
        encoding="utf-8",
    )
    return epoch_id, frozen, snapshot


def fixed_video_split(video_ids: Iterable[str], fraction: float = 0.2) -> Tuple[List[str], List[str]]:
    ordered = sorted(
        {str(value) for value in video_ids if str(value)},
        key=lambda value: hashlib.sha256(
            ("step8-threshold-validation-v1:" + value).encode()
        ).hexdigest(),
    )
    if len(ordered) < 2:
        return ordered, []
    validation_count = max(
        1,
        min(len(ordered) - 1, int(round(len(ordered) * max(0.05, min(0.5, fraction))))),
    )
    return ordered[:-validation_count], ordered[-validation_count:]


def collect_conflicts(evidence_videos: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collect protected tracks whose raw threshold decision was Discard."""
    conflicts = []
    for video in evidence_videos:
        video_id = str(video.get("video_id", ""))
        for trajectory in video.get("trajectory_motion_evidence", []):
            validation = dict(trajectory.get("causal_motion_fact_validation", {}))
            decision = dict(trajectory.get("fact_decision", {}))
            protection = dict(trajectory.get("symbol_grounded_protection", {}))
            original = str(
                decision.get(
                    "original_decision_before_protection",
                    decision.get("trajectory_decision", ""),
                )
            )
            if not trajectory.get("symbol_grounded_protected") or original != "Discard":
                continue
            rejection_reasons = [str(value) for value in validation.get("rejection_reasons", [])]
            metrics = dict(validation.get("step_metrics", {}))
            active_thresholds = dict(validation.get("thresholds", {}))
            calibration_measurements = []
            for threshold_name, spec in CALIBRATABLE_THRESHOLDS.items():
                if spec["issue"] not in rejection_reasons:
                    continue
                measured = _f(metrics.get(spec["metric"]))
                active_threshold = _f(active_thresholds.get(threshold_name))
                calibration_measurements.append(
                    {
                        "threshold_name": threshold_name,
                        "issue": spec["issue"],
                        "metric": spec["metric"],
                        "measured_value": measured,
                        "active_threshold": active_threshold,
                        "distance_to_threshold": measured - active_threshold,
                    }
                )
            conflicts.append(
                {
                    "video_id": video_id,
                    "track_id": int(trajectory.get("track_id", -1)),
                    "object_class": str(trajectory.get("primary_label", "unknown")),
                    "validation_status": str(validation.get("validation_status", "")),
                    "original_decision": original,
                    "final_decision": str(decision.get("decision", "")),
                    "rejection_reasons": rejection_reasons,
                    "hard_invalid_reasons": sorted(set(rejection_reasons) & HARD_INVALID_REASONS),
                    "step_metrics": metrics,
                    "active_thresholds": active_thresholds,
                    "calibration_measurements": calibration_measurements,
                    "matched_rule_ids": list(protection.get("matched_rule_ids", [])),
                    "grounding_evidence": list(protection.get("grounding_evidence", [])),
                    "protection_reason": str(protection.get("protection_reason", "")),
                    "confidence_score": _f(
                        dict(trajectory.get("uncertainty", {})).get("confidence_score")
                    ),
                    "num_observations": int(
                        dict(trajectory.get("trajectory_statistics", {})).get(
                            "num_observations", 0
                        )
                    ),
                }
            )
    return sorted(conflicts, key=lambda row: (row["video_id"], row["track_id"]))


def compile_candidate(
    active_policy: Dict[str, Any],
    update_conflicts: Sequence[Dict[str, Any]],
    default_thresholds: Dict[str, Any],
    *,
    min_samples: int,
    target_quantile: float,
    max_relative_change: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Calculate a bounded candidate from non-hard protected conflicts."""
    candidate = copy.deepcopy(active_policy)
    candidate["version"] = int(active_policy["version"]) + 1
    candidate["parent_version"] = int(active_policy["version"])
    candidate["status"] = "candidate"
    changes = {}
    evidence = {}
    eligible = [
        row for row in update_conflicts if not row.get("hard_invalid_reasons")
    ]
    for threshold_name, spec in CALIBRATABLE_THRESHOLDS.items():
        values = [
            _f(dict(row.get("step_metrics", {})).get(spec["metric"]))
            for row in eligible
            if spec["issue"] in row.get("rejection_reasons", [])
            and _f(dict(row.get("step_metrics", {})).get(spec["metric"])) > 0
        ]
        evidence[threshold_name] = {
            "issue": spec["issue"],
            "metric": spec["metric"],
            "sample_count": len(values),
            "values": sorted(values),
        }
        if len(values) < min_samples:
            continue
        current = _f(active_policy["thresholds"][threshold_name])
        observed_target = _quantile(values, target_quantile)
        epoch_cap = current * (1.0 + max_relative_change)
        absolute_cap = _f(default_thresholds[threshold_name]) * 2.0
        proposed = min(observed_target, epoch_cap, absolute_cap)
        if proposed <= current + max(1e-9, abs(current) * 1e-6):
            continue
        candidate["thresholds"][threshold_name] = float(proposed)
        changes[threshold_name] = {
            "from": current,
            "to": float(proposed),
            "observed_quantile": float(observed_target),
            "sample_count": len(values),
            "bounded_by_epoch_cap": observed_target > epoch_cap,
            "bounded_by_absolute_cap": observed_target > absolute_cap,
        }
    candidate["calibration"] = {
        "source": "protected_invalid_conflict_batch",
        "weak_label": "symbol_grounded_protected",
        "min_samples": min_samples,
        "target_quantile": target_quantile,
        "max_relative_change": max_relative_change,
        "changes": changes,
    }
    return validate_policy(candidate, default_thresholds), {
        "eligible_conflict_count": len(eligible),
        "excluded_hard_conflict_count": len(update_conflicts) - len(eligible),
        "changes": changes,
        "threshold_evidence": evidence,
    }


def _evaluate_records(
    evidence_videos: Sequence[Dict[str, Any]],
    thresholds: Dict[str, Any],
    validation_fn: Callable[..., Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[Tuple[str, int], str]]:
    total = protected = protected_invalid = invalid = hard_invalid = 0
    statuses: Dict[Tuple[str, int], str] = {}
    for video in evidence_videos:
        video_id = str(video.get("video_id", ""))
        for row in video.get("trajectory_motion_evidence", []):
            total += 1
            is_protected = bool(row.get("symbol_grounded_protected"))
            validation = validation_fn(
                list(row.get("trajectory_observations", [])),
                dict(row.get("trajectory_statistics", {})),
                dict(row.get("uncertainty", {})),
                thresholds=thresholds,
            )
            status = str(validation.get("validation_status", "invalid"))
            key = (video_id, int(row.get("track_id", -1)))
            statuses[key] = status
            reasons = set(map(str, validation.get("rejection_reasons", [])))
            invalid += status == "invalid"
            protected += is_protected
            protected_invalid += is_protected and status == "invalid"
            hard_invalid += bool(reasons & HARD_INVALID_REASONS)
    return {
        "track_count": total,
        "protected_track_count": protected,
        "invalid_count": invalid,
        "protected_invalid_conflict_count": protected_invalid,
        "semantic_alignment_rate": (
            (protected - protected_invalid) / max(1, protected)
        ),
        "hard_invalid_count": hard_invalid,
    }, statuses


def evaluate_candidate(
    validation_videos: Sequence[Dict[str, Any]],
    active_policy: Dict[str, Any],
    candidate_policy: Dict[str, Any],
    validation_fn: Callable[..., Dict[str, Any]],
    *,
    max_unprotected_flip_rate: float,
) -> Dict[str, Any]:
    current, current_status = _evaluate_records(
        validation_videos, active_policy["thresholds"], validation_fn
    )
    candidate, candidate_status = _evaluate_records(
        validation_videos, candidate_policy["thresholds"], validation_fn
    )
    protected_keys = {
        (str(video.get("video_id", "")), int(row.get("track_id", -1)))
        for video in validation_videos
        for row in video.get("trajectory_motion_evidence", [])
        if bool(row.get("symbol_grounded_protected"))
    }
    flips = [
        key
        for key, before in current_status.items()
        if before == "invalid" and candidate_status.get(key) != "invalid"
    ]
    protected_flips = [key for key in flips if key in protected_keys]
    unprotected_flips = [key for key in flips if key not in protected_keys]
    unprotected_count = max(1, current["track_count"] - current["protected_track_count"])
    unprotected_flip_rate = len(unprotected_flips) / unprotected_count
    independent = bool(validation_videos)
    improved = (
        candidate["protected_invalid_conflict_count"]
        < current["protected_invalid_conflict_count"]
    )
    no_critical_regression = (
        candidate["hard_invalid_count"] == current["hard_invalid_count"]
        and candidate["invalid_count"] <= current["invalid_count"]
        and unprotected_flip_rate <= max_unprotected_flip_rate
    )
    promoted = independent and improved and no_critical_regression
    return {
        "promoted": promoted,
        "decision": "stage_for_next_epoch" if promoted else "reject",
        "reason": (
            "semantic_alignment_improved_without_critical_regression"
            if promoted
            else "independent_validation_split_unavailable"
            if not independent
            else "critical_regression"
            if not no_critical_regression
            else "validation_semantic_alignment_did_not_improve"
        ),
        "independent_validation_split": independent,
        "current_metrics": current,
        "candidate_metrics": candidate,
        "protected_invalid_to_non_invalid": [
            {"video_id": key[0], "track_id": key[1]} for key in protected_flips
        ],
        "unprotected_invalid_to_non_invalid": [
            {"video_id": key[0], "track_id": key[1]} for key in unprotected_flips
        ],
        "unprotected_flip_rate": unprotected_flip_rate,
        "max_unprotected_flip_rate": max_unprotected_flip_rate,
        "no_critical_regression": no_critical_regression,
    }


def run_threshold_calibration(
    state: Dict[str, Any],
    output_root: Path,
    default_thresholds: Dict[str, Any],
    validation_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    policy_root = root / "policies"
    policy_root.mkdir(parents=True, exist_ok=True)
    active = validate_policy(
        dict(
            state.get(
                "trajectory_validation_threshold_policy",
                default_policy(default_thresholds),
            )
        ),
        default_thresholds,
    )
    evidence = list(state.get("trajectory_motion_evidence", []))
    epoch_id = int(state.get("trajectory_validation_threshold_epoch_id", 0))
    conflicts = collect_conflicts(evidence)
    update_ids, validation_ids = fixed_video_split(
        [str(row.get("video_id", "")) for row in evidence],
        fraction=_f(os.environ.get("CAUVID_STEP8_THRESHOLD_VALIDATION_FRACTION", "0.2"), 0.2),
    )
    update_set, validation_set = set(update_ids), set(validation_ids)
    update_conflicts = [row for row in conflicts if row["video_id"] in update_set]
    validation_videos = [
        row for row in evidence if str(row.get("video_id", "")) in validation_set
    ]
    min_samples = max(
        1, int(os.environ.get("CAUVID_STEP8_THRESHOLD_MIN_CONFLICTS", "10"))
    )
    quantile = max(
        0.5,
        min(
            0.99,
            _f(os.environ.get("CAUVID_STEP8_THRESHOLD_TARGET_QUANTILE", "0.9"), 0.9),
        ),
    )
    max_change = max(
        0.01,
        min(
            0.5,
            _f(os.environ.get("CAUVID_STEP8_THRESHOLD_MAX_RELATIVE_CHANGE", "0.1"), 0.1),
        ),
    )
    max_unprotected_flip_rate = max(
        0.0,
        min(
            0.25,
            _f(os.environ.get("CAUVID_STEP8_THRESHOLD_MAX_UNPROTECTED_FLIP_RATE", "0.02"), 0.02),
        ),
    )
    candidate, compilation = compile_candidate(
        active,
        update_conflicts,
        default_thresholds,
        min_samples=min_samples,
        target_quantile=quantile,
        max_relative_change=max_change,
    )
    if compilation["changes"]:
        evaluation = evaluate_candidate(
            validation_videos,
            active,
            candidate,
            validation_fn,
            max_unprotected_flip_rate=max_unprotected_flip_rate,
        )
    else:
        evaluation = {
            "promoted": False,
            "decision": "reject",
            "reason": "insufficient_eligible_conflict_evidence",
            "independent_validation_split": bool(validation_videos),
            "current_metrics": {},
            "candidate_metrics": {},
            "no_critical_regression": True,
        }
    candidate_record = {
        **candidate,
        "threshold_epoch_id": epoch_id,
        "update_video_ids": update_ids,
        "validation_video_ids": validation_ids,
        "compilation": compilation,
        "evaluation": evaluation,
    }
    conflict_payload = json.dumps(conflicts, indent=2)
    (root / "protected_invalid_conflicts.json").write_text(
        conflict_payload, encoding="utf-8"
    )
    (root / f"protected_invalid_conflicts_epoch_{epoch_id:04d}.json").write_text(
        conflict_payload,
        encoding="utf-8",
    )
    candidate_path = (
        root
        / f"candidate_threshold_policy_epoch_{epoch_id:04d}_v{candidate['version']:04d}.json"
    )
    candidate_path.write_text(
        json.dumps(candidate_record, indent=2), encoding="utf-8"
    )
    (root / f"candidate_threshold_policy_v{candidate['version']:04d}.json").write_text(
        json.dumps(candidate_record, indent=2),
        encoding="utf-8",
    )
    if evaluation.get("promoted"):
        (policy_root / PENDING_FILENAME).write_text(
            json.dumps({**candidate, "status": "pending"}, indent=2),
            encoding="utf-8",
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": "batch_weak_supervision_threshold_calibration",
        "threshold_epoch_id": epoch_id,
        "weak_label_semantics": (
            "symbol_grounded_protected means retain for safety review; it is not "
            "ground truth that the trajectory is physically valid"
        ),
        "active_policy_version": active["version"],
        "candidate_policy_version": candidate["version"],
        "num_conflicts": len(conflicts),
        "num_update_conflicts": len(update_conflicts),
        "update_video_ids": update_ids,
        "validation_video_ids": validation_ids,
        "calibratable_thresholds": sorted(CALIBRATABLE_THRESHOLDS),
        "hard_invalid_reasons": sorted(HARD_INVALID_REASONS),
        "compilation": compilation,
        "promotion": evaluation,
        "activation": "next_epoch_only" if evaluation.get("promoted") else "not_staged",
    }
    manifest_payload = json.dumps(manifest, indent=2)
    (root / "threshold_calibration_manifest.json").write_text(
        manifest_payload, encoding="utf-8"
    )
    (root / f"threshold_calibration_manifest_epoch_{epoch_id:04d}.json").write_text(
        manifest_payload, encoding="utf-8"
    )
    epoch_path = policy_root / f"threshold_epoch_{epoch_id:04d}.json"
    if epoch_id > 0 and epoch_path.exists():
        try:
            epoch = json.loads(epoch_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            epoch = {"epoch_id": epoch_id}
        epoch.update({"status": "completed", "calibration": manifest})
        epoch_path.write_text(json.dumps(epoch, indent=2), encoding="utf-8")
    return {
        **state,
        "trajectory_threshold_conflicts": conflicts,
        "trajectory_threshold_candidate_policy": candidate_record,
        "trajectory_threshold_calibration_manifest": manifest,
        "trajectory_threshold_calibration_output_root": root,
    }
