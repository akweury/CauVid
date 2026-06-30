from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import config
from src.exp_driving_videos.modules.pipe_utils.exp_driving_utils import load_pattern_cfg_file


DRIVING_MINI_OD_MODEL = "yolov8l-worldv2.pt"
DEFAULT_TRAIN_VIDEO_COUNT = 100
DEFAULT_EVAL_VIDEO_COUNT = 2
DEFAULT_PIPELINE_VIDEO_LIMIT = DEFAULT_TRAIN_VIDEO_COUNT + DEFAULT_EVAL_VIDEO_COUNT
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


def _get_nested_bool(path: Sequence[str], default: bool) -> bool:
    try:
        cfg = _load_exp_driving_cfg()
        value = _get_nested(cfg, path)
        if value is None:
            return bool(default)
        return bool(value)
    except Exception as exc:
        print(f"[warn] Could not load boolean config at {'.'.join(path)}: {exc}. Using default={default}.")
        return bool(default)


def get_detection_render_video_enabled(default: bool = True) -> bool:
    return _get_nested_bool(("detection", "render_video"), default)


def get_detection_check_cache_enabled(default: bool = False) -> bool:
    return _get_nested_bool(("detection", "check_cache"), default)


def get_tracking_render_video_enabled(default: bool = True) -> bool:
    return _get_nested_bool(("tracking", "render_video"), default)


def get_merge_render_video_enabled(default: bool = True) -> bool:
    return _get_nested_bool(("merge_annotations", "render_video"), default)


def get_ego_motion_render_video_enabled(default: bool = True) -> bool:
    return _get_nested_bool(("ego_motion", "render_video"), default)


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
            "render_videos": True,
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
            "render_videos": True,
        },
        path=("segment_object_motion",),
        warn_label="segment object motion",
    )


def get_important_objects_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "selection_strategy": "not_implemented",
            "passthrough_selected_objects": True,
            "candidate_selection": {
                "enabled": True,
                "max_per_segment": 8,
                "max_per_class": 3,
                "max_per_prior": 2,
                "min_score": 0.0,
                "score_weights": {
                    "calibrated_score": 0.30,
                    "raw_score": 0.15,
                    "track_quality": 0.20,
                    "temporal_consistency": 0.15,
                    "prior_relevance": 0.10,
                    "semantic_relevance": 0.10,
                },
            },
        },
        path=("important_objects",),
        warn_label="important objects",
    )


def get_video_selection_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "default_video_limit": DEFAULT_PIPELINE_VIDEO_LIMIT,
        },
        path=("video_selection",),
        warn_label="video selection",
    )


def get_traffic_control_attributes_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": True,
            "min_crop_size": 8,
            "traffic_light_state_threshold": 0.06,
            "traffic_light_min_confidence_margin": 0.025,
            "front_distance_far_threshold": 40.0,
            "center_x_threshold_meters": 3.0,
            "relevance_high_threshold": 0.67,
            "relevance_medium_threshold": 0.4,
            "relevance_weights": {
                "visibility": 0.3,
                "center": 0.2,
                "size_or_distance": 0.25,
                "front_center": 0.25,
            },
        },
        path=("traffic_control_attributes",),
        warn_label="traffic control attributes",
    )


def get_logic_atoms_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "lateral_position_threshold": 2.0,
            "visibility_persistent_threshold": 0.8,
            "visibility_present_threshold": 0.3,
            "include_segment_boundary_atoms": True,
            "include_object_identity_atoms": True,
            "include_traffic_control_atoms": True,
            "traffic_control_relevance_threshold": 0.4,
            "include_unknown_traffic_light_state_atoms": False,
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
            "generate_mixed_accepted_candidate_rules": False,
            "generate_candidate_candidate_rules": False,
            "max_candidate_only_initial_rules": 500,
            "max_mixed_candidate_initial_rules": 3000,
            "max_candidate_candidate_initial_rules": 0,
            "max_total_initial_rules": 10000,
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


def get_initial_rule_pruning_cfg() -> Dict[str, Any]:
    defaults = {
        "enabled": True,
        "max_total_initial_rules": 300,
        "category_budgets": {
            "accepted_only": 180,
            "accepted_candidate": 0,
            "candidate_only": 80,
            "candidate_candidate": 0,
        },
        "diversity_key": "positive_coverage",
    }
    resolved = _load_cfg_section(
        defaults,
        path=("initial_rule_pruning",),
        warn_label="initial rule pruning",
    )
    override_budgets = resolved.get("category_budgets")
    merged_budgets = dict(defaults["category_budgets"])
    if isinstance(override_budgets, dict):
        merged_budgets.update(override_budgets)
    resolved["category_budgets"] = merged_budgets
    return resolved


def get_extended_rules_cfg() -> Dict[str, Any]:
    defaults = {
        "num_rounds": 2,
        "evaluation_strategy": "binding_aware_intersection",
        "prune_strategies": [
            "low_evidence",
            "empty_evidence",
            "same_firings_as_parent",
            "same_confidence_smaller_evidence",
        ],
        "min_positive_support_to_extend": 2,
        "same_confidence_smaller_evidence_enabled": True,
        "skip_perfect_precision_parents_without_new_positive_recovery": False,
        "allow_segment_context_extension_atoms": True,
        "allow_provenance_extension_atoms": False,
        "allow_candidate_candidate_extension": False,
        "max_segment_context_atoms_per_rule": 1,
        "per_parent_extension_top_k": 25,
        "post_extension_pruning": {
            "enabled": True,
            "deduplicate_by_firing_signature": True,
            "dominance_prune_same_positive_coverage": True,
            "max_rules_per_firing_signature": 1,
            "max_rules_per_positive_coverage_set": 2,
        },
        "max_round_rules": {
            "round_1": 800,
            "round_2": 200,
        },
        "max_round_rules_by_category": {
            "round_1": {
                "accepted_only": 300,
                "mixed_accepted_candidate": 350,
                "candidate_only": 100,
                "candidate_candidate": 0,
            },
            "round_2": {
                "accepted_only": 80,
                "mixed_accepted_candidate": 100,
                "candidate_only": 20,
                "candidate_candidate": 0,
            },
        },
        "parents_for_next_round": {
            "max_total": 300,
            "accepted_only": 120,
            "mixed_accepted_candidate": 150,
            "candidate_only": 30,
            "candidate_candidate": 0,
        },
    }
    resolved = _load_cfg_section(
        defaults,
        path=("extended_rules",),
        warn_label="extended rules",
    )
    for key in ("post_extension_pruning", "max_round_rules", "max_round_rules_by_category", "parents_for_next_round"):
        merged = dict(defaults[key])
        override = resolved.get(key)
        if isinstance(override, dict):
            for subkey, value in override.items():
                if isinstance(merged.get(subkey), dict) and isinstance(value, dict):
                    nested = dict(merged[subkey])
                    nested.update(value)
                    merged[subkey] = nested
                else:
                    merged[subkey] = value
        resolved[key] = merged
    return resolved


def get_final_rules_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "selector_mode": "coverage_family_aware",
            "top_k": 20,
            "category_budgets": {
                "accepted_only": 10,
                "mixed_accepted_candidate": 8,
                "candidate_only": 2,
                "candidate_candidate": 0,
            },
        },
        path=("final_rules",),
        warn_label="final rules",
    )


def get_rule_pool_and_selector_diagnostic_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": False,
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
            "oracle_target_mode": "peak_f1",
            "max_missing_rules_per_selector": 20,
            "max_extra_rules_per_selector": 20,
            "vehicle_classes": ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"],
            "near_states": ["near"],
            "center_states": ["centered"],
        },
        path=("rule_pool_and_selector_diagnostic",),
        warn_label="rule-pool and selector diagnostic",
    )


def get_diverse_final_rules_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "top_k": 50,
            "category_budgets": {
                "accepted_only": 25,
                "mixed_accepted_candidate": 20,
                "candidate_only": 5,
                "candidate_candidate": 0,
            },
            "score_mode": "legacy_diverse_positive_coverage",
            "selection_method_name": "greedy_diverse_positive_coverage",
            "output_prefix": "diverse_final_rules",
            "new_positive_weight": 1.0,
            "confidence_weight": 0.25,
            "overlap_penalty": 0.35,
            "family_penalty": 0.75,
            "negative_support_penalty": 0.1,
            "weak_candidate_rule_penalty": 1.0,
        },
        path=("diverse_final_rules",),
        warn_label="diverse final rules",
    )


def get_semantic_constrained_diverse_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "top_k": 50,
            "category_budgets": {
                "accepted_only": 25,
                "mixed_accepted_candidate": 20,
                "candidate_only": 5,
                "candidate_candidate": 0,
            },
            "score_mode": "semantic_constrained_diverse",
            "selection_method_name": "semantic_constrained_diverse",
            "output_prefix": "semantic_constrained_diverse_final_rules",
            "new_positive_weight": 1.0,
            "confidence_weight": 0.25,
            "overlap_penalty": 0.35,
            "family_penalty": 0.75,
            "negative_support_penalty": 0.1,
            "weak_candidate_rule_penalty": 1.0,
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
            "category_budgets": {
                "accepted_only": 25,
                "mixed_accepted_candidate": 20,
                "candidate_only": 5,
                "candidate_candidate": 0,
            },
            "score_mode": "coverage_family_aware",
            "selection_method_name": "greedy_coverage_family_aware",
            "output_prefix": "coverage_family_aware_final_rules",
            "coverage_weight": 1.0,
            "quality_weight": 1.0,
            "overlap_penalty": 0.5,
            "family_penalty": 0.6,
            "family_diversity_bonus": 0.75,
            "negative_support_penalty": 0.1,
            "weak_candidate_rule_penalty": 1.0,
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


def get_oracle_rule_selection_gap_diagnostic_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "oracle_target_mode": "peak_f1",
            "max_missing_rules_per_selector": 20,
            "max_extra_rules_per_selector": 20,
        },
        path=("oracle_rule_selection_gap_diagnostic",),
        warn_label="oracle rule-selection gap diagnostic",
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
            "rule_set_mode": "selected",
            "primary_rule_set": "selected",
        },
        path=("rule_evaluation",),
        warn_label="rule evaluation",
    )


def get_baseline_comparison_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": False,
            "run_neural_symbolic": True,
            "run_rule_aggregation": True,
        },
        path=("baseline_comparison",),
        warn_label="baseline comparison",
    )


def get_reasoning_supervised_od_calibration_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": False,
            "min_delta_f1": 0.0,
            "min_fn_gain": 1,
            "allow_nonpositive_delta_with_fn_recovery": True,
        },
        path=("reasoning_supervised_od_calibration",),
        warn_label="reasoning-supervised OD calibration",
    )


def get_candidate_contribution_summary_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "top_useful_candidate_rules": 10,
            "top_noisy_candidate_rules": 10,
            "top_broad_candidate_rules": 10,
            "top_matched_priors": 10,
        },
        path=("candidate_contribution_summary",),
        warn_label="candidate contribution summary",
    )


def get_reasoning_to_od_pseudo_labels_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "selector_mode": "primary_plus_best",
            "max_match_states_per_example_rule": 12,
            "include_neutral_selected_candidate_detections": True,
        },
        path=("reasoning_to_od_pseudo_labels",),
        warn_label="reasoning-to-OD pseudo labels",
    )


def get_od_confidence_calibration_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "min_labeled_detections": 8,
            "min_positive_detections": 2,
            "min_negative_detections": 2,
            "logistic_c": 1.0,
            "max_iter": 500,
            "class_weight": "balanced",
        },
        path=("od_confidence_calibration",),
        warn_label="OD confidence calibration",
    )


def get_baseline_safe_calibration_gate_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "baseline_tolerance": 1e-9,
            "final_f1_tolerance": 1e-9,
        },
        path=("baseline_safe_calibration_gate",),
        warn_label="baseline-safe calibration gate",
    )


def get_od_calibration_loop_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "max_iterations": 3,
            "force_full_recompute_on_policy_change": True,
            "stop_on_gate_reject": True,
            "stop_on_final_f1_plateau": True,
        },
        path=("od_calibration_loop",),
        warn_label="OD calibration loop",
    )


def get_rule_aggregation_baseline_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": False,
            "validation_fraction": 0.25,
            "class_weight": "balanced",
            "c_values": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            "solver": "liblinear",
            "max_iter": 2000,
            "random_seed": 0,
            "active_rule_min_train_support": 1,
            "top_weighted_rules": 30,
            "vehicle_classes": ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"],
            "near_states": ["near"],
            "center_states": ["centered"],
        },
        path=("rule_aggregation_baseline",),
        warn_label="rule aggregation baseline",
    )


def get_neural_symbolic_baseline_cfg() -> Dict[str, Any]:
    defaults = {
        "enabled": False,
        "min_feature_count": 1,
        "probability_threshold": 0.5,
        "validation_fraction": 0.25,
        "imbalance_strategy": "pos_weight",
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
            "enabled": False,
            "top_rule_families": 8,
            "dpi": 160,
            "figure_format": "png",
        },
        path=("rule_selection_visualization",),
        warn_label="rule selection visualization",
    )


def get_integrated_method_visualization_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": False,
            "dpi": 170,
            "figure_format": "png",
        },
        path=("integrated_method_visualization",),
        warn_label="integrated method visualization",
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


def get_object_to_atom_coverage_diagnostic_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": False,
            "class_aliases": {
                "traffic light": "traffic_light",
                "traffic_light": "traffic_light",
                "traffic signal": "traffic_light",
                "traffic signals": "traffic_light",
            },
            "max_rule_examples_per_class": 5,
        },
        path=("object_to_atom_coverage_diagnostic",),
        warn_label="object-to-atom coverage diagnostic",
    )


def get_traffic_control_rule_utility_diagnostic_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": False,
            "highlight_predicates": [
                "traffic_light_state",
                "traffic_light_relevant",
                "stop_sign_relevant",
                "traffic_control_relevant",
            ],
            "tracked_states": ["red", "yellow", "green"],
            "top_rules_per_key": 5,
            "vehicle_classes": ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"],
            "near_states": ["near"],
            "center_states": ["centered"],
        },
        path=("traffic_control_rule_utility_diagnostic",),
        warn_label="traffic-control rule utility diagnostic",
    )


def get_traffic_control_temporal_alignment_diagnostic_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": False,
            "highlight_predicates": [
                "traffic_light_state",
                "traffic_light_relevant",
                "traffic_control_relevant",
                "stop_sign_relevant",
            ],
            "tracked_states": ["red", "yellow", "green"],
            "future_horizons": [1, 2, 3, 5],
            "positive_forward_states": ["forward_slowdown", "stopping"],
        },
        path=("traffic_control_temporal_alignment_diagnostic",),
        warn_label="traffic-control temporal alignment diagnostic",
    )


def get_traffic_light_detection_quality_audit_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": False,
            "max_samples_total": 0,
            "max_samples_per_state": 50,
            "max_samples_per_video": 12,
            "include_unknown_state": True,
            "min_state_confidence": 0.0,
            "random_seed": 0,
            "separate_confidence_bands": False,
            "confidence_split_threshold": 0.75,
            "max_samples_per_state_per_confidence_band": 0,
            "frame_path_roots": [],
        },
        path=("traffic_light_detection_quality_audit",),
        warn_label="traffic-light detection quality audit",
    )


def get_background_causal_prior_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": False,
            "target_predicate": "brake_next",
        },
        path=("background_causal_prior",),
        warn_label="background causal prior",
    )


def get_background_rule_relevance_prior_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": False,
            "target_predicate": "brake_next",
        },
        path=("background_rule_relevance_prior",),
        warn_label="background rule relevance prior",
    )


def get_reasoning_feedback_signal_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": False,
            "target_predicate": "brake_next",
            "primary_rule_set": "original",
            "max_feedback_requests": 200,
            "max_prior_requests_per_example": 3,
            "low_confidence_positive_margin": 0.08,
            "min_existing_supporting_facts": 2,
            "max_supporting_facts_per_request": 16,
            "max_fired_rules_per_request": 6,
        },
        path=("reasoning_feedback_signal",),
        warn_label="reasoning feedback signal",
    )


def get_pipeline_recompute_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "logic_atoms": False,
            "target_head_atoms": False,
            "temporal_rule_examples": False,
            "candidate_rules": False,
            "extended_rules": False,
            "traffic_control_attributes": False,
            "background_rule_relevance_prior": False,
            "final_rules": True,
            "diverse_final_rules": True,
            "semantic_constrained_diverse_final_rules": True,
            "coverage_family_aware_final_rules": True,
            "rule_pool_upper_bound_diagnostic": True,
            "oracle_rule_selection_gap_diagnostic": True,
            "rule_evaluation": True,
            "candidate_contribution_summary": True,
            "reasoning_to_od_pseudo_labels": True,
            "od_confidence_calibration": True,
            "rule_aggregation_baseline": True,
            "baseline_safe_calibration_gate": True,
            "object_to_atom_coverage_diagnostic": True,
            "traffic_control_rule_utility_diagnostic": True,
            "traffic_control_temporal_alignment_diagnostic": True,
            "traffic_light_detection_quality_audit": True,
            "neural_symbolic_baseline": True,
            "error_and_explainability_analysis": True,
            "vehicle_rule_diagnostic": True,
            "fn_categorization_diagnostic": True,
            "rule_selection_visualization": True,
            "integrated_method_visualization": True,
            "background_causal_prior": True,
            "reasoning_feedback_signal": True,
        },
        path=("pipeline_recompute",),
        warn_label="pipeline recompute",
    )


def get_error_and_explainability_cfg() -> Dict[str, Any]:
    return _load_cfg_section(
        {
            "enabled": False,
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
            "enabled": False,
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


def get_candidate_contribution_summary_output_root() -> Path:
    return _load_output_root("18a_driving_mini_candidate_contribution_summary")


def get_od_confidence_calibration_loop_output_root() -> Path:
    return _load_output_root("18hij_driving_mini_od_calibration_loop")


def get_rule_aggregation_baseline_output_root() -> Path:
    return _load_output_root("18c_driving_mini_rule_aggregation_baseline")


def get_object_to_atom_coverage_diagnostic_output_root() -> Path:
    return _load_output_root("18d_driving_mini_object_to_atom_coverage_diagnostic")


def get_traffic_control_rule_utility_diagnostic_output_root() -> Path:
    return _load_output_root("18e_driving_mini_traffic_control_rule_utility")


def get_traffic_control_temporal_alignment_diagnostic_output_root() -> Path:
    return _load_output_root("18f_driving_mini_traffic_control_temporal_alignment")


def get_traffic_light_detection_quality_audit_output_root() -> Path:
    return _load_output_root("18g_driving_mini_traffic_light_detection_quality_audit")


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


def get_oracle_rule_selection_gap_diagnostic_output_root() -> Path:
    return _load_output_root("17e_driving_mini_oracle_rule_selection_gap_diagnostic")


def get_vehicle_rule_diagnostic_output_root() -> Path:
    return _load_output_root("20_driving_mini_vehicle_rule_diagnostic")


def get_rule_selection_visualization_output_root() -> Path:
    return _load_output_root("21_rule_selection_visualization")


def get_integrated_method_visualization_output_root() -> Path:
    return _load_output_root("22_driving_mini_integrated_method_visualization")


def get_background_rule_relevance_prior_output_root() -> Path:
    return _load_output_root("00_driving_mini_background_rule_relevance_prior")


def get_background_causal_prior_output_root() -> Path:
    return _load_output_root("23a_driving_mini_background_causal_prior")


def get_reasoning_feedback_signal_output_root() -> Path:
    return _load_output_root("23_driving_mini_reasoning_feedback_signal")


def get_fn_categorization_diagnostic_output_root() -> Path:
    return _load_output_root("20b_rule_selection_fn_diagnostic")
