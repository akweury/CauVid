"""Deterministic, versioned epoch policy management for Step 8C."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from src.exp_july.perception.trajectory_pattern_llm_batch import PATTERNS, RESIDUALS, REPAIRS

SCHEMA_VERSION = 1


def _f(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def default_policy():
    return {
        "schema_version": SCHEMA_VERSION,
        "version": 1,
        "status": "active",
        "residual_weights": {name: 1.0 for name in RESIDUALS},
        "pattern_biases": {name: 0.0 for name in PATTERNS},
        "repair_preferences": {name: [] for name in PATTERNS},
        "parent_version": None,
    }


def _validate_policy(policy):
    if not isinstance(policy, dict):
        raise ValueError("policy must be an object")
    weights = policy.get("residual_weights")
    biases = policy.get("pattern_biases")
    preferences = policy.get("repair_preferences")
    if not isinstance(weights, dict) or set(weights) != set(RESIDUALS):
        raise ValueError("policy residual_weights must cover the fixed residual schema")
    if not isinstance(biases, dict) or set(biases) != set(PATTERNS):
        raise ValueError("policy pattern_biases must cover the fixed pattern schema")
    if not isinstance(preferences, dict) or set(preferences) != set(PATTERNS):
        raise ValueError("policy repair_preferences must cover the fixed pattern schema")
    if any(not 0.1 <= _f(value, -1) <= 10.0 for value in weights.values()):
        raise ValueError("residual weight outside [0.1, 10]")
    if any(not -2.0 <= _f(value, 99) <= 2.0 for value in biases.values()):
        raise ValueError("pattern bias outside [-2, 2]")
    for operations in preferences.values():
        if not isinstance(operations, list) or any(op not in REPAIRS for op in operations):
            raise ValueError("unsupported repair preference")
    return policy


def begin_epoch(root):
    """Activate a previously promoted policy, then freeze a snapshot for this epoch."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    active_path = root / "active_policy.json"; pending_path = root / "pending_policy.json"
    activated = False
    if pending_path.exists():
        pending = _validate_policy(json.loads(pending_path.read_text(encoding="utf-8")))
        pending["status"] = "active"
        active_path.write_text(json.dumps(pending, indent=2), encoding="utf-8")
        pending_path.unlink()
        activated = True
    if active_path.exists():
        active = _validate_policy(json.loads(active_path.read_text(encoding="utf-8")))
    else:
        active = default_policy()
        active_path.write_text(json.dumps(active, indent=2), encoding="utf-8")
    epochs = sorted(root.glob("epoch_*.json"))
    epoch_id = len(epochs) + 1
    frozen = copy.deepcopy(active)
    snapshot = {"epoch_id": epoch_id, "policy": frozen, "activated_pending_policy": activated,
                "policy_frozen": True, "status": "processing"}
    (root / f"epoch_{epoch_id:04d}.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return epoch_id, frozen, snapshot


def deterministic_interpretation(candidates, policy):
    weights = policy["residual_weights"]; biases = policy["pattern_biases"]
    costs = {}
    for candidate in candidates:
        pid = candidate["pattern_id"]
        costs[pid] = sum(_f(candidate["residual_vector"].get(name)) * _f(weights[name], 1.0) for name in RESIDUALS) - _f(biases[pid])
    values = list(costs.values()); low = min(values or [0]); high = max(values or [1]); span = max(1e-9, high-low)
    return [{"pattern_id": pid, "plausibility": max(0.0, min(1.0, 1-(costs[pid]-low)/span)),
             "ignorable_errors": [], "structural_conflicts": [],
             "recommended_repairs": list(policy["repair_preferences"].get(pid, [])),
             "explanation": f"deterministic epoch policy v{policy['version']} weighted residual ranking"}
            for pid in PATTERNS]


def fixed_video_split(video_ids, fraction=0.2):
    ordered = sorted(set(map(str, video_ids)), key=lambda value: hashlib.sha256(("step8c-validation-v1:"+value).encode()).hexdigest())
    if len(ordered) < 2:
        return ordered, []
    count = max(1, min(len(ordered)-1, int(round(len(ordered)*fraction))))
    return ordered[:-count], ordered[-count:]


def _case(record):
    selected = record.get("selected_candidate", {})
    return {"video_id": record.get("video_id"), "track_id": record.get("track_id"),
            "object_class": record.get("symbolic_track", {}).get("object_class"),
            "source_validation": record.get("symbolic_track", {}).get("source_validation", {}).get("validation_status"),
            "final_validation": record.get("final_validation_status"), "repair_applied": bool(record.get("repair_applied")),
            "pattern": record.get("final_pattern"), "residual_improvement": _f(selected.get("residual_improvement")),
            "observation_retention": _f(selected.get("observation_retention", 1), 1),
            "selection_reason": record.get("final_selection_reason")}


def review_package(records, epoch_id, interval_index, policy, validation_video_ids, max_cases=5):
    successes = [row for row in records if row.get("repair_applied") and row.get("final_validation_status") != "invalid"]
    failures = [row for row in records if not row.get("repair_applied") or row.get("final_validation_status") == "invalid"]
    successes.sort(key=lambda row: (-_f(row.get("selected_candidate", {}).get("residual_improvement")), str(row.get("video_id")), int(row.get("track_id", -1))))
    failures.sort(key=lambda row: (str(row.get("final_selection_reason")), str(row.get("video_id")), int(row.get("track_id", -1))))
    pattern_counts = {}
    for row in records:
        pattern_counts[row.get("final_pattern", "unknown")] = pattern_counts.get(row.get("final_pattern", "unknown"), 0) + 1
    return {"schema_version": SCHEMA_VERSION, "epoch_id": epoch_id, "interval_index": interval_index,
            "frozen_policy_version": policy["version"], "track_count": len(records),
            "video_ids": sorted({str(row.get("video_id")) for row in records}),
            "validation_video_ids": sorted(validation_video_ids), "aggregates": {
                "repairs_applied": sum(bool(row.get("repair_applied")) for row in records),
                "invalid_after": sum(row.get("final_validation_status") == "invalid" for row in records),
                "mean_residual_improvement": sum(_f(row.get("selected_candidate", {}).get("residual_improvement")) for row in records)/max(1, len(records)),
                "pattern_counts": pattern_counts},
            "representative_successes": [_case(row) for row in successes[:max_cases]],
            "representative_failures": [_case(row) for row in failures[:max_cases]]}


def review_prompt(package):
    return ("Review this aggregated Step 8C epoch interval. Propose only a structured policy_patch; never make track decisions. "
            "Allowed patch keys are residual_weights, pattern_biases, repair_preferences. Include only changed entries. "
            "Weights must be in [0.1,10], biases in [-2,2], and repairs must use: "+",".join(REPAIRS)+
            '. JSON only: {"policy_patch":{"residual_weights":{},"pattern_biases":{},"repair_preferences":{}},"rationale":"...","critical_regressions":[]}. review_package='+
            json.dumps(package, separators=(",", ":"), default=str))


def compile_patch(active, response):
    if not isinstance(response, dict):
        raise ValueError("policy review response must be an object")
    patch = response.get("policy_patch", {})
    if not isinstance(patch, dict) or set(patch)-{"residual_weights", "pattern_biases", "repair_preferences"}:
        raise ValueError("invalid policy patch fields")
    candidate = copy.deepcopy(active)
    for field, allowed in (("residual_weights", RESIDUALS), ("pattern_biases", PATTERNS), ("repair_preferences", PATTERNS)):
        updates = patch.get(field, {})
        if not isinstance(updates, dict) or set(updates)-set(allowed):
            raise ValueError(f"invalid {field} patch")
        candidate[field].update(updates)
    candidate.update({"version": int(active["version"])+1, "status": "candidate", "parent_version": int(active["version"])})
    return _validate_policy(candidate)


def policy_metrics(records, policy):
    accepted = invalid = critical = 0; improvement = retention = 0.0
    for record in records:
        candidates = [row for row in record.get("candidate_repairs", []) if row.get("symbolic_verdict") == "pass"]
        ranked = []
        for row in candidates:
            pid = row.get("pattern_id", "unknown"); vector = row.get("residual_vector_after", {})
            policy_cost = sum(_f(vector.get(name))*_f(policy["residual_weights"][name], 1) for name in RESIDUALS)-_f(policy["pattern_biases"][pid])
            preference = policy["repair_preferences"].get(pid, [])
            repair_bonus = 0.05 if row.get("repair_operation") in preference else 0.0
            ranked.append((-policy_cost+repair_bonus, row))
        chosen = max(ranked, key=lambda item: (item[0], str(item[1].get("candidate_id"))))[1] if ranked else None
        accepted += chosen is not None
        if chosen:
            status = chosen.get("validation", {}).get("validation", {}).get("validation_status", "invalid")
            invalid += status == "invalid"; improvement += _f(chosen.get("residual_improvement")); retention += _f(chosen.get("observation_retention", 1), 1)
            critical += bool(chosen.get("new_anomalies")) or not chosen.get("class_consistent", True)
        else:
            invalid += record.get("symbolic_track", {}).get("source_validation", {}).get("validation_status") == "invalid"
            retention += 1.0
    count = len(records)
    return {"track_count": count, "accepted_repairs": accepted, "invalid_after": invalid,
            "critical_regressions": critical, "mean_residual_improvement": improvement/max(1, count),
            "mean_observation_retention": retention/max(1, count),
            "overall_score": accepted + improvement/max(1, count) + retention/max(1, count) - 2*invalid - 4*critical}


def evaluate_and_stage(root, active, candidate, validation_records, llm_response):
    current = policy_metrics(validation_records, active); proposed = policy_metrics(validation_records, candidate)
    independent = bool(validation_records)
    no_regression = (proposed["invalid_after"] <= current["invalid_after"] and proposed["critical_regressions"] == 0
                     and proposed["mean_observation_retention"] >= 0.95
                     and not (llm_response.get("critical_regressions", []) if isinstance(llm_response, dict) else ["malformed_review"]))
    improved = proposed["overall_score"] > current["overall_score"] + 1e-9
    promoted = independent and no_regression and improved
    decision = {"promoted": promoted, "decision": "stage_for_next_epoch" if promoted else "reject",
                "reason": "quality_improved_without_critical_regression" if promoted else
                          "independent_validation_split_unavailable" if not independent else
                          "critical_regression" if not no_regression else "metrics_did_not_improve",
                "active_policy_version": active["version"], "candidate_policy_version": candidate["version"],
                "current_metrics": current, "candidate_metrics": proposed,
                "activation_epoch": "next_epoch" if promoted else None}
    root = Path(root); (root/f"candidate_policy_v{candidate['version']:04d}.json").write_text(json.dumps({**candidate, "evaluation": decision}, indent=2), encoding="utf-8")
    if promoted:
        (root/"pending_policy.json").write_text(json.dumps({**candidate, "status": "pending"}, indent=2), encoding="utf-8")
    return decision
