from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


_FORCED_RECOMPUTE_KEYS = {
    "traffic_control_attributes",
    "logic_atoms",
    "target_head_atoms",
    "temporal_rule_examples",
    "candidate_rules",
    "extended_rules",
    "final_rules",
    "diverse_final_rules",
    "semantic_constrained_diverse_final_rules",
    "coverage_family_aware_final_rules",
    "rule_pool_upper_bound_diagnostic",
    "oracle_rule_selection_gap_diagnostic",
    "rule_evaluation",
    "candidate_contribution_summary",
    "reasoning_to_od_pseudo_labels",
    "od_confidence_calibration",
    "rule_aggregation_baseline",
    "baseline_safe_calibration_gate",
    "object_to_atom_coverage_diagnostic",
    "traffic_control_rule_utility_diagnostic",
    "traffic_control_temporal_alignment_diagnostic",
    "traffic_light_detection_quality_audit",
    "neural_symbolic_baseline",
    "error_and_explainability_analysis",
    "vehicle_rule_diagnostic",
    "fn_categorization_diagnostic",
    "rule_selection_visualization",
    "integrated_method_visualization",
    "background_causal_prior",
    "reasoning_feedback_signal",
}


@dataclass(frozen=True)
class ODCalibrationLoopConfig:
    max_iterations: int
    force_full_recompute_on_policy_change: bool
    stop_on_gate_reject: bool
    stop_on_final_f1_plateau: bool
    improvement_tolerance: float


@dataclass(frozen=True)
class ODCalibrationLoopDecision:
    continue_loop: bool
    reason: str
    current_final_f1: float
    reference_final_f1: float | None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_loop_cfg(
    cfg: Mapping[str, Any] | None,
    *,
    improvement_tolerance: float,
) -> ODCalibrationLoopConfig:
    mapping = dict(cfg or {})
    max_iterations = max(1, _safe_int(mapping.get("max_iterations", 1), 1))
    return ODCalibrationLoopConfig(
        max_iterations=max_iterations,
        force_full_recompute_on_policy_change=_safe_bool(
            mapping.get("force_full_recompute_on_policy_change", True),
            True,
        ),
        stop_on_gate_reject=_safe_bool(mapping.get("stop_on_gate_reject", True), True),
        stop_on_final_f1_plateau=_safe_bool(mapping.get("stop_on_final_f1_plateau", True), True),
        improvement_tolerance=max(0.0, _safe_float(improvement_tolerance, 0.0)),
    )


def iteration_requires_full_recompute(
    iteration_index: int,
    *,
    force_full_recompute_on_policy_change: bool,
) -> bool:
    return bool(force_full_recompute_on_policy_change and int(iteration_index) > 1)


def apply_iteration_recompute_overrides(
    recompute_cfg: Mapping[str, Any] | None,
    *,
    force_full_recompute: bool,
) -> Dict[str, Any]:
    resolved = {str(key): bool(value) for key, value in dict(recompute_cfg or {}).items()}
    if not force_full_recompute:
        return resolved
    for key in _FORCED_RECOMPUTE_KEYS:
        resolved[key] = True
    return resolved


def should_continue_after_iteration(
    gate_result: Mapping[str, Any] | None,
    *,
    iteration_index: int,
    loop_cfg: ODCalibrationLoopConfig,
) -> ODCalibrationLoopDecision:
    payload = dict(gate_result or {})
    current_metrics = dict(payload.get("current_metrics", {}))
    reference_metrics = dict(payload.get("reference_metrics", {}))
    current_final_f1 = _safe_float(current_metrics.get("final_f1", 0.0), 0.0)
    reference_final_f1 = (
        _safe_float(reference_metrics.get("final_f1", 0.0), 0.0)
        if reference_metrics
        else None
    )
    decision = str(payload.get("decision", "")).strip().lower()

    if int(iteration_index) >= int(loop_cfg.max_iterations):
        return ODCalibrationLoopDecision(
            continue_loop=False,
            reason="max_iterations_reached",
            current_final_f1=current_final_f1,
            reference_final_f1=reference_final_f1,
        )
    if loop_cfg.stop_on_gate_reject and decision and decision != "accept":
        return ODCalibrationLoopDecision(
            continue_loop=False,
            reason="gate_rejected",
            current_final_f1=current_final_f1,
            reference_final_f1=reference_final_f1,
        )
    if (
        loop_cfg.stop_on_final_f1_plateau
        and reference_final_f1 is not None
        and current_final_f1 <= reference_final_f1 + loop_cfg.improvement_tolerance
    ):
        return ODCalibrationLoopDecision(
            continue_loop=False,
            reason="final_f1_not_improved",
            current_final_f1=current_final_f1,
            reference_final_f1=reference_final_f1,
        )
    return ODCalibrationLoopDecision(
        continue_loop=True,
        reason="continue_with_updated_active_policy",
        current_final_f1=current_final_f1,
        reference_final_f1=reference_final_f1,
    )
