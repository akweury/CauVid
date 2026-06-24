from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.exp_driving_videos.modules import od_calibration_policy_utils


_GATE_VERSION = 1


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "baseline_tolerance": float(cfg.get("baseline_tolerance", 1e-9)),
        "final_f1_tolerance": float(cfg.get("final_f1_tolerance", 1e-9)),
    }


def _current_metrics_snapshot(
    evaluation_results_by_name: Dict[str, Dict[str, Any]],
    primary_rule_set: str,
    rule_aggregation_baseline_results: Dict[str, Any],
) -> Dict[str, Any]:
    primary_evaluation = dict(evaluation_results_by_name.get(primary_rule_set, {}))
    ablation = dict(primary_evaluation.get("candidate_rule_ablation", {}))
    accepted_only = dict(ablation.get("baseline_metrics", {}))
    accepted_plus_candidate = dict(ablation.get("augmented_metrics", {}))
    aggregation_eval = dict(
        dict(rule_aggregation_baseline_results.get("metrics_by_split", {}))
        .get("eval", {})
        .get("best_validation_threshold", {})
    )
    accepted_only_f1 = _safe_float(accepted_only.get("f1", 0.0), 0.0)
    candidate_augmented_f1 = _safe_float(accepted_plus_candidate.get("f1", 0.0), 0.0)
    aggregation_f1 = _safe_float(aggregation_eval.get("f1", 0.0), 0.0)
    return {
        "primary_rule_set": str(primary_rule_set),
        "accepted_only_metrics": accepted_only,
        "accepted_plus_candidate_metrics": accepted_plus_candidate,
        "rule_aggregation_eval_metrics": aggregation_eval,
        "accepted_only_f1": accepted_only_f1,
        "accepted_plus_candidate_f1": candidate_augmented_f1,
        "rule_aggregation_f1": aggregation_f1,
        "final_f1": max(candidate_augmented_f1, aggregation_f1),
    }


def process(
    evaluation_results_by_name: Dict[str, Dict[str, Any]],
    primary_rule_set: str,
    rule_aggregation_baseline_results: Dict[str, Any],
    calibration_results: Dict[str, Any],
    *,
    iteration_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = od_calibration_policy_utils.get_iteration_root(iteration_id)
    if output_root is not None:
        out_root = Path(output_root) / iteration_id
        out_root.mkdir(parents=True, exist_ok=True)
    audit_json_path = out_root / "baseline_preservation_audit.json"
    gate_json_path = out_root / "calibration_gate_decision.json"

    if not force_recompute and gate_json_path.exists():
        cached = od_calibration_policy_utils.load_json(gate_json_path, default={})
        if int(cached.get("version", 0)) == _GATE_VERSION and _cfg_key_subset(cached.get("config", {})) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {gate_json_path.name}")
            return cached

    proposed_policy = dict(calibration_results.get("policy", {}))
    active_state_before = od_calibration_policy_utils.load_active_od_calibration_state()
    active_policy_before = od_calibration_policy_utils.load_active_od_calibration_policy()
    reference_metrics = dict(active_state_before.get("accepted_reference_metrics", {}))
    current_metrics = _current_metrics_snapshot(
        evaluation_results_by_name=evaluation_results_by_name,
        primary_rule_set=primary_rule_set,
        rule_aggregation_baseline_results=rule_aggregation_baseline_results,
    )

    baseline_tolerance = _safe_float(cfg.get("baseline_tolerance", 1e-9), 1e-9)
    final_f1_tolerance = _safe_float(cfg.get("final_f1_tolerance", 1e-9), 1e-9)

    if not reference_metrics:
        decision = "accept"
        decision_reason = "bootstrap_accept_no_prior_reference"
    else:
        baseline_regressed = (
            _safe_float(current_metrics.get("accepted_only_f1", 0.0), 0.0)
            + baseline_tolerance
            < _safe_float(reference_metrics.get("accepted_only_f1", 0.0), 0.0)
        )
        final_regressed = (
            _safe_float(current_metrics.get("final_f1", 0.0), 0.0)
            + final_f1_tolerance
            < _safe_float(reference_metrics.get("final_f1", 0.0), 0.0)
        )
        if baseline_regressed:
            decision = "reject"
            decision_reason = "protected_baseline_regressed"
        elif final_regressed:
            decision = "reject"
            decision_reason = "final_f1_regressed"
        else:
            decision = "accept"
            decision_reason = "baseline_safe_and_final_f1_non_decreasing"

    audit = {
        "version": _GATE_VERSION,
        "iteration_id": str(iteration_id),
        "config": _cfg_key_subset(cfg),
        "active_policy_before": {
            "policy_id": od_calibration_policy_utils.policy_id(active_policy_before),
            "latest_iteration_id": _safe_text(active_state_before.get("latest_iteration_id", "")),
        },
        "proposed_policy_id": od_calibration_policy_utils.policy_id(proposed_policy),
        "reference_metrics": reference_metrics,
        "current_metrics": current_metrics,
        "same_run_audit": {
            "accepted_only_f1": _safe_float(current_metrics.get("accepted_only_f1", 0.0), 0.0),
            "accepted_plus_candidate_f1": _safe_float(current_metrics.get("accepted_plus_candidate_f1", 0.0), 0.0),
            "rule_aggregation_f1": _safe_float(current_metrics.get("rule_aggregation_f1", 0.0), 0.0),
            "final_f1": _safe_float(current_metrics.get("final_f1", 0.0), 0.0),
        },
    }

    if decision == "accept":
        od_calibration_policy_utils.save_json_atomic(
            od_calibration_policy_utils.get_active_od_calibration_policy_path(),
            proposed_policy,
        )
        active_state_after = od_calibration_policy_utils.write_active_policy_state(
            active_policy=proposed_policy,
            latest_iteration_id=str(iteration_id),
            accepted_reference_metrics=current_metrics,
            gate_decision={
                "decision": decision,
                "decision_reason": decision_reason,
                "proposed_policy_id": od_calibration_policy_utils.policy_id(proposed_policy),
            },
        )
        active_policy_after_id = od_calibration_policy_utils.policy_id(proposed_policy)
    else:
        active_state_after = dict(active_state_before)
        active_policy_after_id = od_calibration_policy_utils.policy_id(active_policy_before)

    gate_result = {
        "version": _GATE_VERSION,
        "iteration_id": str(iteration_id),
        "config": _cfg_key_subset(cfg),
        "decision": decision,
        "decision_reason": decision_reason,
        "proposed_policy_id": od_calibration_policy_utils.policy_id(proposed_policy),
        "active_policy_after_id": active_policy_after_id,
        "audit_path": str(audit_json_path),
        "active_state_after": active_state_after,
    }

    od_calibration_policy_utils.save_json_atomic(audit_json_path, audit)
    od_calibration_policy_utils.save_json_atomic(gate_json_path, gate_result)

    print(
        "  baseline_safe_calibration_gate: "
        f"decision={decision} | "
        f"reason={decision_reason} | "
        f"active_policy_after={active_policy_after_id or 'none'}"
    )
    print(f"Baseline preservation audit JSON written to {audit_json_path}")
    print(f"Calibration gate decision JSON written to {gate_json_path}")
    return gate_result


def run(
    evaluation_results_by_name: Dict[str, Dict[str, Any]],
    primary_rule_set: str,
    rule_aggregation_baseline_results: Dict[str, Any],
    calibration_results: Dict[str, Any],
    *,
    iteration_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process(
        evaluation_results_by_name=evaluation_results_by_name,
        primary_rule_set=primary_rule_set,
        rule_aggregation_baseline_results=rule_aggregation_baseline_results,
        calibration_results=calibration_results,
        iteration_id=iteration_id,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
