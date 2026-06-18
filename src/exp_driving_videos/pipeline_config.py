from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import config
from src.exp_driving_videos.modules.pipe_utils.exp_driving_utils import load_pattern_cfg_file


DRIVING_MINI_OD_MODEL = "yolov8l-worldv2.pt"
DEFAULT_TRAIN_VIDEO_COUNT = 8
DEFAULT_EVAL_VIDEO_COUNT = 2
DRIVING_MINI_OD_CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person",
    "traffic light",
    "stop sign",
    "building",
    "tree",
    "crosswalk",
]


def _load_exp_driving_cfg() -> Dict[str, Any]:
    cfg_path = config.get_config_path("exp_driving")
    return dict(load_pattern_cfg_file(cfg_path))


def _get_nested(mapping: Dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _load_cfg_section(
    defaults: Dict[str, Any],
    *,
    path: Sequence[str],
    warn_label: str,
) -> Dict[str, Any]:
    resolved = dict(defaults)
    try:
        cfg = _load_exp_driving_cfg()
        override = _get_nested(cfg, path)
        if isinstance(override, dict):
            resolved.update(override)
    except Exception as exc:
        print(f"[warn] Could not load {warn_label} config: {exc}. Using defaults.")
    return resolved


def _load_output_root(*parts: str) -> Path:
    out = config.get_output_path("pipeline_output")
    for part in parts:
        out = out / part
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_ego_motion_smoothing_window(default: int = 5) -> int:
    try:
        cfg = _load_exp_driving_cfg()
        return int(_get_nested(cfg, ("ego_motion", "smoothing_window")) or default)
    except Exception as exc:
        print(f"[warn] Could not load exp_driving config: {exc}. Using default={default}.")
        return default


def get_ego_static_adjustment_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": True,
            "static_object_keywords": ["building", "traffic light"],
            "blend_weight": 0.7,
            "min_static_pixels": 300,
        },
        path=("ego_motion", "static_adjustment"),
        warn_label="ego static adjustment",
    )


def get_temporal_segmentation_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "forward_stop_threshold": 0.05,
            "forward_stop_enter_threshold": 0.05,
            "forward_stop_exit_threshold": 0.08,
            "stop_total_speed_enter_threshold": 0.09,
            "stop_total_speed_exit_threshold": 0.14,
            "forward_accel_threshold": 0.03,
            "lateral_turn_threshold": 0.03,
            "stop_window_size": 5,
            "motion_window_size": 5,
            "forward_direction_epsilon": 0.025,
            "lateral_motion_window_size": 10,
            "lateral_direction_epsilon": 0.03,
            "lateral_straight_threshold": 35.0,
            "compare_lateral_straight_thresholds": [15, 25, 35, 45],
            "min_stop_duration": 5,
            "min_turn_duration": 3,
            "min_segment_length": 7,
            "compare_forward_stop_thresholds": [1.0],
            "compare_min_segment_lengths": [7],
        },
        path=("temporal_segmentation",),
        warn_label="temporal segmentation",
    )


def get_segment_object_motion_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "rel_vz_threshold": 0.2,
            "rel_vx_threshold": 0.2,
            "compare_rel_vx_thresholds": [10.0, 20.0, 50.0],
            "rel_speed_threshold": 0.3,
            "dominance_ratio_threshold": 0.6,
            "distance_near_threshold": 15.0,
            "distance_medium_threshold": 30.0,
            "top_k_visualized_objects": 20,
            "visualization_fps": 10.0,
        },
        path=("segment_object_motion",),
        warn_label="segment object motion",
    )


def get_important_objects_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "selection_strategy": "not_implemented",
            "passthrough_selected_objects": True,
        },
        path=("important_objects",),
        warn_label="important objects",
    )


def get_logic_atoms_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "lateral_position_threshold": 2.0,
            "visibility_persistent_threshold": 0.8,
            "visibility_present_threshold": 0.3,
            "include_segment_boundary_atoms": True,
            "include_object_identity_atoms": True,
        },
        path=("logic_atoms",),
        warn_label="logic atoms",
    )


def get_target_head_atoms_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "target_predicate": "brake_next",
            "negative_target_predicate": "not_brake_next",
            "positive_forward_states": ["forward_slowdown", "stopping"],
            "include_negative_examples": True,
        },
        path=("target_head_atoms",),
        warn_label="target head atoms",
    )


def get_temporal_rule_examples_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "deduplicate_body_atoms": True,
            "sort_body_atoms": True,
            "include_clause_text": True,
            "include_negative_examples": True,
        },
        path=("temporal_rule_examples",),
        warn_label="temporal rule examples",
    )


def get_candidate_rules_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "target_predicate": "brake_next",
            "negative_target_predicate": "not_brake_next",
            "use_only_positive_examples": True,
            "min_positive_support": 1,
            "include_example_ids": True,
            "ignored_body_predicates": [
                "segment",
                "segment_start_frame",
                "segment_end_frame",
                "object_in_segment",
                "object_track",
            ],
        },
        path=("candidate_rules",),
        warn_label="candidate rules",
    )


def get_extended_rules_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "num_rounds": 3,
            "evaluation_strategy": "binding_aware_intersection",
            "prune_strategies": [
                "low_evidence",
                "empty_evidence",
                "same_firings_as_parent",
                "same_confidence_smaller_evidence",
            ],
            "min_positive_support_to_extend": 2,
            "same_confidence_smaller_evidence_enabled": True,
        },
        path=("extended_rules",),
        warn_label="extended rules",
    )


def get_final_rules_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {"top_k": 50},
        path=("final_rules",),
        warn_label="final rules",
    )


def get_diverse_final_rules_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "top_k": 50,
            "score_mode": "legacy_diverse_positive_coverage",
            "selection_method_name": "greedy_diverse_positive_coverage",
            "output_prefix": "diverse_final_rules",
            "new_positive_weight": 1.0,
            "confidence_weight": 0.25,
            "overlap_penalty": 0.35,
            "family_penalty": 0.75,
            "negative_support_penalty": 0.1,
        },
        path=("diverse_final_rules",),
        warn_label="diverse final rules",
    )


def get_semantic_constrained_diverse_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "top_k": 50,
            "score_mode": "semantic_constrained_diverse",
            "selection_method_name": "semantic_constrained_diverse",
            "output_prefix": "semantic_constrained_diverse_final_rules",
            "new_positive_weight": 1.0,
            "confidence_weight": 0.25,
            "overlap_penalty": 0.35,
            "family_penalty": 0.75,
            "negative_support_penalty": 0.1,
            "semantic_bonus_weight": 1.5,
            "semantic_hard_constraints": True,
            "semantic_min_positive_support": 2,
            "semantic_min_total_support": 2,
            "semantic_min_confidence": 0.6,
            "semantic_min_family_counts": {
                "vehicle_centered_partial": 2,
                "near_centered_partial": 2,
                "vehicle_near_partial": 2,
                "exact_vehicle_near_centered": 1,
            },
            "vehicle_classes": ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"],
            "near_states": ["near"],
            "center_states": ["centered"],
        },
        path=("semantic_constrained_diverse_final_rules",),
        warn_label="semantic constrained diverse",
    )


def get_coverage_family_aware_final_rules_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "top_k": 50,
            "score_mode": "coverage_family_aware",
            "selection_method_name": "greedy_coverage_family_aware",
            "output_prefix": "coverage_family_aware_final_rules",
            "coverage_weight": 1.0,
            "quality_weight": 1.0,
            "overlap_penalty": 0.5,
            "family_penalty": 0.6,
            "family_diversity_bonus": 0.75,
            "negative_support_penalty": 0.1,
        },
        path=("coverage_family_aware_final_rules",),
        warn_label="coverage-aware final rules",
    )


def get_rule_pool_upper_bound_diagnostic_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "top_single_rules": 100,
            "precision_thresholds": [0.5, 0.7, 0.9],
            "f1_thresholds": [0.1, 0.2, 0.3, 0.4],
            "min_recall_thresholds": [0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
            "oracle_k_values": [1, 5, 10, 20, 50, 100],
            "selection_gap_threshold": 0.05,
            "selection_pool_min_f1": 0.35,
            "low_pool_f1_threshold": 0.35,
            "low_single_rule_f1_threshold": 0.2,
            "high_precision_threshold": 0.8,
            "low_recall_threshold": 0.1,
            "high_recall_threshold": 0.2,
            "low_precision_threshold": 0.5,
            "vehicle_classes": ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"],
            "near_states": ["near"],
            "center_states": ["centered"],
        },
        path=("rule_pool_upper_bound_diagnostic",),
        warn_label="rule-pool upper-bound diagnostic",
    )


def get_data_split_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "train_video_count": DEFAULT_TRAIN_VIDEO_COUNT,
            "eval_video_count": DEFAULT_EVAL_VIDEO_COUNT,
            "strategy": "eval_fraction",
            "eval_fraction": 0.2,
        },
        path=("data_split",),
        warn_label="data split",
    )


def get_rule_evaluation_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "prediction_mode": "any_rule_positive",
            "rule_set_mode": "all",
            "primary_rule_set": "original",
        },
        path=("rule_evaluation",),
        warn_label="rule evaluation",
    )


def get_neural_symbolic_baseline_cfg() -> Dict[str, Any]:
    defaults = {
        "min_feature_count": 1,
        "probability_threshold": 0.5,
        "random_seed": 0,
        "device": "auto",
        "single_segment_mlp": {
            "hidden_dims": [128, 64],
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "num_epochs": 100,
            "early_stopping_patience": 12,
        },
        "temporal_model": {
            "architecture": "gru",
            "history_window": 4,
            "hidden_dims": [128, 64],
            "gru_hidden_dim": 128,
            "gru_num_layers": 1,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "num_epochs": 100,
            "early_stopping_patience": 12,
        },
    }
    resolved = _load_cfg_section(
        defaults,
        path=("neural_symbolic_baseline",),
        warn_label="neural symbolic baseline",
    )
    for nested_key in ("single_segment_mlp", "temporal_model"):
        nested_defaults = dict(defaults[nested_key])
        nested_override = resolved.get(nested_key, {})
        if isinstance(nested_override, dict):
            nested_defaults.update(nested_override)
        resolved[nested_key] = nested_defaults
    return resolved


def get_rule_selection_visualization_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "top_rule_families": 8,
            "dpi": 160,
            "figure_format": "png",
        },
        path=("rule_selection_visualization",),
        warn_label="rule selection visualization",
    )


def get_fn_categorization_diagnostic_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": False,
            "vehicle_context_match_levels": [
                "exact_vehicle_near_centered",
                "vehicle_near_partial",
                "vehicle_centered_partial",
            ],
            "predicate_gap_levels": [
                "missing_rule_or_predicate_dense_context",
                "missing_rule_or_predicate_sparse_context",
                "unexplained_noise_or_symbol_gap",
            ],
            "noisy_levels": [
                "unexplained_noise_no_objects",
                "unexplained_noise_or_symbol_gap",
            ],
        },
        path=("fn_categorization_diagnostic",),
        warn_label="FN categorization diagnostic",
    )


def get_pipeline_recompute_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "candidate_rules": False,
            "extended_rules": True,
            "final_rules": True,
            "diverse_final_rules": True,
            "semantic_constrained_diverse_final_rules": True,
            "coverage_family_aware_final_rules": True,
            "rule_pool_upper_bound_diagnostic": True,
            "rule_evaluation": True,
            "neural_symbolic_baseline": True,
            "error_and_explainability_analysis": True,
            "vehicle_rule_diagnostic": True,
            "fn_categorization_diagnostic": True,
            "rule_selection_visualization": True,
        },
        path=("pipeline_recompute",),
        warn_label="pipeline recompute",
    )


def get_error_and_explainability_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "vehicle_classes": ["car", "truck", "bus", "motorcycle", "bicycle"],
            "dense_context_min_objects": 2,
            "overlap_rule_threshold": 10,
        },
        path=("error_and_explainability_analysis",),
        warn_label="error analysis",
    )


def get_vehicle_rule_diagnostic_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "vehicle_classes": ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"],
            "near_states": ["near"],
            "center_states": ["centered"],
            "primary_rule_set": "original",
        },
        path=("vehicle_rule_diagnostic",),
        warn_label="vehicle rule diagnostic",
    )


def get_merged_candidate_rules_output_root() -> Path:
    return _load_output_root("15_driving_mini_merged_initial_rules")


def get_split_output_root() -> Path:
    return _load_output_root("driving_mini_split")


def get_rule_evaluation_output_root() -> Path:
    return _load_output_root("18_driving_mini_rule_evaluation")


def get_error_and_explainability_output_root() -> Path:
    return _load_output_root("19_driving_mini_error_and_explainability_analysis")


def get_neural_symbolic_baseline_output_root() -> Path:
    return _load_output_root("neural_baselines_driving_mini")


def get_coverage_family_aware_final_rules_output_root() -> Path:
    return _load_output_root("17c_driving_mini_coverage_family_aware_final_rules")


def get_semantic_constrained_diverse_output_root() -> Path:
    return _load_output_root("17b2_driving_mini_semantic_constrained_diverse_final_rules")


def get_rule_pool_upper_bound_diagnostic_output_root() -> Path:
    return _load_output_root("17d_driving_mini_rule_pool_upper_bound_diagnostic")


def get_vehicle_rule_diagnostic_output_root() -> Path:
    return _load_output_root("20_driving_mini_vehicle_rule_diagnostic")


def get_rule_selection_visualization_output_root() -> Path:
    return _load_output_root("21_rule_selection_visualization")


def get_fn_categorization_diagnostic_output_root() -> Path:
    return _load_output_root("20b_rule_selection_fn_diagnostic")
