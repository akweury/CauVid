"""
Experiment pipeline for the driving_mini dataset.

Steps:
  0. background_rule_relevance_prior_driving_mini — define a manual
                            task-level brake_next prior table for
                            candidate-ranking and tracking-priority use only;
                            never emit it as detections, facts, or rule labels.
  1. detect_driving_mini  — run YOLO-World detection on every video clip and
                            return per-video detection records.
  2. tracking_driving_mini — run ByteTrack on the detection results to assign
                             persistent track IDs and return object tracks.
    3. dataset_annotations_driving_mini — load dataset-provided object
                                                                                annotations/tracks (ground truth).
    4. merge_gt_and_detected_driving_mini — merge GT + tracked detections;
                                                                                    when both exist, GT is kept.
    5. prepare_3d_positions_driving_mini — generate/use Depth Anything depth maps
                                                                                    and infer per-object 3D positions.
    6. ego_motion_driving_mini — estimate per-frame ego motion (vx, vz, yaw_rate)
                                    from successive-frame optical flow (RAFT) filtered
                                    to background/static pixels using bg_mask.
    7. relative_object_motion_driving_mini — estimate per-object motion relative
                                    to ego using 3D object trajectories and ego motion.
    8. temporal_segmentation_driving_mini — segment ego-motion signals into
                    event spans and cut points.
    9. segment_object_motion_driving_mini — summarize per-object relative
                    motion symbolically for each merged temporal segment.
    10. important_objects_driving_mini — analyze/filter important objects per
                    segment. Strategy placeholder for now.
    10B. traffic_control_attributes_driving_mini — enrich traffic lights and
                    stop signs with heuristic state and relevance attributes.
    11. logic_atoms_driving_mini — convert filtered segment-level symbolic facts
                    into logic atoms for downstream reasoning.
    12. target_head_atoms_driving_mini — derive future-action target/head atoms
                    for rule learning from consecutive segments.
    13. temporal_rule_examples_driving_mini — combine current-segment symbolic
                    atoms with target/head atoms into final rule-learning examples.
    14. candidate_rules_driving_mini — generate unary-body temporal initial
                    rules whose head is the target predicate.
    15. merge_initial_rules — flatten all per-video initial rules into a
                    single persisted list for downstream scoring/selection.
    15B. initial_rule_pruning — prune the merged initial rule pool using
                    dataset-agnostic coverage, category, confidence, and
                    provenance/source metadata before extension.
    16. extended_rules_driving_mini — iteratively extend rules with
                    unary initial-rule body atoms for a fixed number of rounds.
    17. final_rules_driving_mini — run one configured selector mode against
                    the Step 16 post-pruned extended rule pool and emit the
                    final selected rule set.
    17D. rule_pool_and_selector_diagnostic_driving_mini — optional combined
                    selector diagnostic that reports category-aware upper
                    bounds, oracle curves, selector-vs-oracle overlap, and
                    bottleneck reasons.
    18. evaluate_rules_driving_mini — evaluate the selected Step 17 rule set
                    and its accepted-only, accepted-plus-mixed,
                    candidate-augmented, and balanced-core subset views on
                    the held-out evaluation split.
    18B. baseline_comparison_driving_mini — optional comparison against
                    neural-symbolic and learned rule-aggregation baselines.
    19. reasoning_supervised_od_calibration_driving_mini — optional
                    candidate-derived pseudo labeling, OD confidence
                    calibration, and baseline-safe gating; runs only when
                    enabled and Step 18 indicates useful candidate signal.
    18D. object_to_atom_coverage_diagnostic_driving_mini — optional
                    object-to-atom coverage tracing across perception,
                    symbolic conversion, extended rules, and selected rules.
    18E. traffic_control_rule_utility_diagnostic_driving_mini — optional
                    traffic-control predicate audit across the Step 16 pool,
                    selected rules, and learned baseline outputs.
    18F. traffic_control_temporal_alignment_diagnostic_driving_mini —
                    optional temporal alignment audit for traffic-control
                    predicates against near-future braking labels.
    18G. traffic_light_detection_quality_audit_driving_mini — optional
                    sampled full-frame visual audits for detected
                    traffic-light objects and predicted state/context.
    20. error_and_explainability_analysis_driving_mini — summarize false
                    negatives / false positives and generate explainability-
                    oriented diagnostics for held-out evaluation examples.
    21. vehicle_rule_diagnostic_driving_mini — audit whether vehicle-centered
                    rules were generated, pruned/scored out, or missed due to
                    predicate representation.
    21B. fn_categorization_diagnostic_driving_mini — categorize false
                    negatives for each selector using the step 16 rule pool,
                    step 18 predictions, step 20 FN examples, and step 21
                    vehicle-context diagnostics.
    22. rule_selection_visualization_driving_mini — generate comparison plots
                    from the step 18/20/21 selector summaries and save a
                    visualization manifest.
    23. integrated_method_visualization_driving_mini — generate integrated
                    publication-style figures comparing NeSy selectors,
                    neural symbolic baselines, learned rule aggregation, and
                    the oracle rule-pool upper bound.
    24A. background_causal_prior_driving_mini — define a fixed background
                    causal prior for brake_next that is used only to steer
                    perception re-checks and never as direct rules or facts.
    24B. reasoning_feedback_signal_driving_mini — combine the background
                    causal prior with existing diagnostics to surface
                    perception re-check signals for likely missed causes of
                    braking.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from rtpt.rtpt import RTPT
except Exception:  # pragma: no cover - optional progress monitor dependency
    RTPT = None

import config
from src.exp_driving_videos import od_calibration_loop as od_calibration_loop_utils
from src.exp_driving_videos import pipeline_config as driving_pipeline_config
from src.exp_driving_videos import pipeline_data as driving_pipeline_data
from src.exp_driving_videos.modules import detect_driving_mini
from src.exp_driving_videos.modules import candidate_rules_driving_mini
from src.exp_driving_videos.modules import candidate_contribution_summary_driving_mini
from src.exp_driving_videos.modules import reasoning_to_od_pseudo_labels_driving_mini
from src.exp_driving_videos.modules import od_confidence_calibration_driving_mini
from src.exp_driving_videos.modules import baseline_safe_calibration_gate_driving_mini
from src.exp_driving_videos.modules import dataset_annotations_driving_mini
from src.exp_driving_videos.modules import diverse_final_rules_driving_mini
from src.exp_driving_videos.modules import evaluate_rules_driving_mini
from src.exp_driving_videos.modules import error_and_explainability_analysis_driving_mini
from src.exp_driving_videos.modules import extended_rules_driving_mini
from src.exp_driving_videos.modules import final_rules_driving_mini
from src.exp_driving_videos.modules import fn_categorization_diagnostic_driving_mini
from src.exp_driving_videos.modules import merge_gt_and_detected_driving_mini
from src.exp_driving_videos.modules import neural_symbolic_baseline_driving_mini
from src.exp_driving_videos.modules import object_to_atom_coverage_diagnostic_driving_mini
from src.exp_driving_videos.modules import prepare_3d_positions_driving_mini
from src.exp_driving_videos.modules import tracking_driving_mini
from src.exp_driving_videos.modules import ego_motion_driving_mini
from src.exp_driving_videos.modules import important_objects_driving_mini
from src.exp_driving_videos.modules import traffic_control_rule_utility_diagnostic_driving_mini
from src.exp_driving_videos.modules import traffic_control_attributes_driving_mini
from src.exp_driving_videos.modules import logic_atoms_driving_mini
from src.exp_driving_videos.modules import relative_object_motion_driving_mini
from src.exp_driving_videos.modules import oracle_rule_selection_gap_diagnostic_driving_mini
from src.exp_driving_videos.modules import rule_pool_upper_bound_diagnostic_driving_mini
from src.exp_driving_videos.modules import rule_aggregation_baseline_driving_mini
from src.exp_driving_videos.modules import integrated_method_visualization_driving_mini
from src.exp_driving_videos.modules import rule_selection_visualization_driving_mini
from src.exp_driving_videos.modules import segment_object_motion_driving_mini
from src.exp_driving_videos.modules import target_head_atoms_driving_mini
from src.exp_driving_videos.modules import temporal_rule_examples_driving_mini
from src.exp_driving_videos.modules import temporal_segmentation_driving_mini
from src.exp_driving_videos.modules import traffic_light_detection_quality_audit_driving_mini
from src.exp_driving_videos.modules import traffic_control_temporal_alignment_diagnostic_driving_mini
from src.exp_driving_videos.modules import vehicle_rule_diagnostic_driving_mini
from src.exp_driving_videos.modules import background_causal_prior_driving_mini
from src.exp_driving_videos.modules import background_rule_relevance_prior_driving_mini
from src.exp_driving_videos.modules import reasoning_feedback_signal_driving_mini
from src.exp_driving_videos.modules import od_calibration_policy_utils

DRIVING_MINI_OD_MODEL = driving_pipeline_config.DRIVING_MINI_OD_MODEL
DEFAULT_TRAIN_VIDEO_COUNT = driving_pipeline_config.DEFAULT_TRAIN_VIDEO_COUNT
DEFAULT_EVAL_VIDEO_COUNT = driving_pipeline_config.DEFAULT_EVAL_VIDEO_COUNT
DRIVING_MINI_OD_CLASSES = driving_pipeline_config.DRIVING_MINI_OD_CLASSES

_get_ego_motion_smoothing_window = driving_pipeline_config.get_ego_motion_smoothing_window
_get_detection_render_video_enabled = driving_pipeline_config.get_detection_render_video_enabled
_get_detection_check_cache_enabled = driving_pipeline_config.get_detection_check_cache_enabled
_get_tracking_render_video_enabled = driving_pipeline_config.get_tracking_render_video_enabled
_get_merge_render_video_enabled = driving_pipeline_config.get_merge_render_video_enabled
_get_ego_motion_render_video_enabled = driving_pipeline_config.get_ego_motion_render_video_enabled
_get_ego_static_adjustment_cfg = driving_pipeline_config.get_ego_static_adjustment_cfg
_get_temporal_segmentation_cfg = driving_pipeline_config.get_temporal_segmentation_cfg
_get_segment_object_motion_cfg = driving_pipeline_config.get_segment_object_motion_cfg
_get_important_objects_cfg = driving_pipeline_config.get_important_objects_cfg
_get_traffic_control_attributes_cfg = driving_pipeline_config.get_traffic_control_attributes_cfg
_get_logic_atoms_cfg = driving_pipeline_config.get_logic_atoms_cfg
_get_target_head_atoms_cfg = driving_pipeline_config.get_target_head_atoms_cfg
_get_temporal_rule_examples_cfg = driving_pipeline_config.get_temporal_rule_examples_cfg
_get_candidate_rules_cfg = driving_pipeline_config.get_candidate_rules_cfg
_get_initial_rule_pruning_cfg = driving_pipeline_config.get_initial_rule_pruning_cfg
_get_extended_rules_cfg = driving_pipeline_config.get_extended_rules_cfg
_get_final_rules_cfg = driving_pipeline_config.get_final_rules_cfg
_get_rule_pool_and_selector_diagnostic_cfg = (
    driving_pipeline_config.get_rule_pool_and_selector_diagnostic_cfg
)
_get_diverse_final_rules_cfg = driving_pipeline_config.get_diverse_final_rules_cfg
_get_semantic_constrained_diverse_cfg = driving_pipeline_config.get_semantic_constrained_diverse_cfg
_get_coverage_family_aware_final_rules_cfg = driving_pipeline_config.get_coverage_family_aware_final_rules_cfg
_get_rule_pool_upper_bound_diagnostic_cfg = driving_pipeline_config.get_rule_pool_upper_bound_diagnostic_cfg
_get_oracle_rule_selection_gap_diagnostic_cfg = driving_pipeline_config.get_oracle_rule_selection_gap_diagnostic_cfg
_get_data_split_cfg = driving_pipeline_config.get_data_split_cfg
_get_rule_evaluation_cfg = driving_pipeline_config.get_rule_evaluation_cfg
_get_baseline_comparison_cfg = driving_pipeline_config.get_baseline_comparison_cfg
_get_candidate_contribution_summary_cfg = driving_pipeline_config.get_candidate_contribution_summary_cfg
_get_reasoning_supervised_od_calibration_cfg = (
    driving_pipeline_config.get_reasoning_supervised_od_calibration_cfg
)
_get_reasoning_to_od_pseudo_labels_cfg = driving_pipeline_config.get_reasoning_to_od_pseudo_labels_cfg
_get_od_confidence_calibration_cfg = driving_pipeline_config.get_od_confidence_calibration_cfg
_get_baseline_safe_calibration_gate_cfg = driving_pipeline_config.get_baseline_safe_calibration_gate_cfg
_get_od_calibration_loop_cfg = driving_pipeline_config.get_od_calibration_loop_cfg
_get_rule_aggregation_baseline_cfg = driving_pipeline_config.get_rule_aggregation_baseline_cfg
_get_object_to_atom_coverage_diagnostic_cfg = driving_pipeline_config.get_object_to_atom_coverage_diagnostic_cfg
_get_traffic_control_rule_utility_diagnostic_cfg = (
    driving_pipeline_config.get_traffic_control_rule_utility_diagnostic_cfg
)
_get_traffic_control_temporal_alignment_diagnostic_cfg = (
    driving_pipeline_config.get_traffic_control_temporal_alignment_diagnostic_cfg
)
_get_traffic_light_detection_quality_audit_cfg = (
    driving_pipeline_config.get_traffic_light_detection_quality_audit_cfg
)
_get_background_causal_prior_cfg = driving_pipeline_config.get_background_causal_prior_cfg
_get_background_rule_relevance_prior_cfg = driving_pipeline_config.get_background_rule_relevance_prior_cfg
_get_reasoning_feedback_signal_cfg = driving_pipeline_config.get_reasoning_feedback_signal_cfg
_get_neural_symbolic_baseline_cfg = driving_pipeline_config.get_neural_symbolic_baseline_cfg
_get_rule_selection_visualization_cfg = driving_pipeline_config.get_rule_selection_visualization_cfg
_get_integrated_method_visualization_cfg = driving_pipeline_config.get_integrated_method_visualization_cfg
_get_fn_categorization_diagnostic_cfg = driving_pipeline_config.get_fn_categorization_diagnostic_cfg
_get_pipeline_recompute_cfg = driving_pipeline_config.get_pipeline_recompute_cfg
_get_error_and_explainability_cfg = driving_pipeline_config.get_error_and_explainability_cfg
_get_vehicle_rule_diagnostic_cfg = driving_pipeline_config.get_vehicle_rule_diagnostic_cfg
_get_rule_evaluation_output_root = driving_pipeline_config.get_rule_evaluation_output_root
_get_merged_candidate_rules_output_root = driving_pipeline_config.get_merged_candidate_rules_output_root
_get_candidate_contribution_summary_output_root = (
    driving_pipeline_config.get_candidate_contribution_summary_output_root
)
_get_od_confidence_calibration_loop_output_root = (
    driving_pipeline_config.get_od_confidence_calibration_loop_output_root
)
_get_rule_aggregation_baseline_output_root = driving_pipeline_config.get_rule_aggregation_baseline_output_root
_get_object_to_atom_coverage_diagnostic_output_root = driving_pipeline_config.get_object_to_atom_coverage_diagnostic_output_root
_get_traffic_control_rule_utility_diagnostic_output_root = (
    driving_pipeline_config.get_traffic_control_rule_utility_diagnostic_output_root
)
_get_traffic_control_temporal_alignment_diagnostic_output_root = (
    driving_pipeline_config.get_traffic_control_temporal_alignment_diagnostic_output_root
)
_get_traffic_light_detection_quality_audit_output_root = (
    driving_pipeline_config.get_traffic_light_detection_quality_audit_output_root
)
_get_neural_symbolic_baseline_output_root = driving_pipeline_config.get_neural_symbolic_baseline_output_root
_get_error_and_explainability_output_root = driving_pipeline_config.get_error_and_explainability_output_root
_get_coverage_family_aware_final_rules_output_root = driving_pipeline_config.get_coverage_family_aware_final_rules_output_root
_get_semantic_constrained_diverse_output_root = driving_pipeline_config.get_semantic_constrained_diverse_output_root
_get_rule_pool_upper_bound_diagnostic_output_root = driving_pipeline_config.get_rule_pool_upper_bound_diagnostic_output_root
_get_oracle_rule_selection_gap_diagnostic_output_root = driving_pipeline_config.get_oracle_rule_selection_gap_diagnostic_output_root
_get_vehicle_rule_diagnostic_output_root = driving_pipeline_config.get_vehicle_rule_diagnostic_output_root
_get_rule_selection_visualization_output_root = driving_pipeline_config.get_rule_selection_visualization_output_root
_get_integrated_method_visualization_output_root = driving_pipeline_config.get_integrated_method_visualization_output_root
_get_fn_categorization_diagnostic_output_root = driving_pipeline_config.get_fn_categorization_diagnostic_output_root
_get_background_causal_prior_output_root = driving_pipeline_config.get_background_causal_prior_output_root
_get_background_rule_relevance_prior_output_root = driving_pipeline_config.get_background_rule_relevance_prior_output_root
_get_reasoning_feedback_signal_output_root = driving_pipeline_config.get_reasoning_feedback_signal_output_root

_merge_candidate_rules = driving_pipeline_data.merge_candidate_rules
_prune_initial_rules = driving_pipeline_data.prune_initial_rules
_select_video_results = driving_pipeline_data.select_video_results
_build_train_eval_split = driving_pipeline_data.build_train_eval_split
_resolve_video_ids = driving_pipeline_data.resolve_video_ids

_PIPELINE_STAGE_SEQUENCE: List[Dict[str, Any]] = [
    {"tag": "0", "stop_after": 0, "label": "background_rule_relevance_prior_driving_mini"},
    {"tag": "1", "stop_after": 1, "label": "detect_driving_mini"},
    {"tag": "2", "stop_after": 2, "label": "tracking_driving_mini"},
    {"tag": "3", "stop_after": 3, "label": "dataset_annotations_driving_mini"},
    {"tag": "4", "stop_after": 4, "label": "merge_gt_and_detected_driving_mini"},
    {"tag": "5", "stop_after": 5, "label": "prepare_3d_positions_driving_mini"},
    {"tag": "6", "stop_after": 6, "label": "ego_motion_driving_mini"},
    {"tag": "7", "stop_after": 7, "label": "relative_object_motion_driving_mini"},
    {"tag": "8", "stop_after": 8, "label": "temporal_segmentation_driving_mini"},
    {"tag": "9", "stop_after": 9, "label": "segment_object_motion_driving_mini"},
    {"tag": "10", "stop_after": 10, "label": "important_objects_driving_mini"},
    {"tag": "10B", "stop_after": None, "label": "traffic_control_attributes_driving_mini"},
    {"tag": "11", "stop_after": 11, "label": "logic_atoms_driving_mini"},
    {"tag": "12", "stop_after": 12, "label": "target_head_atoms_driving_mini"},
    {"tag": "13", "stop_after": 13, "label": "temporal_rule_examples_driving_mini"},
    {"tag": "14", "stop_after": 14, "label": "candidate_rules_driving_mini"},
    {"tag": "15", "stop_after": 15, "label": "merge_initial_rules"},
    {"tag": "15B", "stop_after": None, "label": "initial_rule_pruning"},
    {"tag": "16", "stop_after": 16, "label": "extended_rules_driving_mini"},
    {"tag": "17", "stop_after": 17, "label": "final_rules_driving_mini"},
    {"tag": "17D", "stop_after": None, "label": "rule_pool_and_selector_diagnostic_driving_mini"},
    {"tag": "18", "stop_after": None, "label": "evaluate_rules_driving_mini"},
    {"tag": "18B", "stop_after": None, "label": "baseline_comparison_driving_mini"},
    {"tag": "19", "stop_after": 19, "label": "reasoning_supervised_od_calibration_driving_mini"},
    {"tag": "18D", "stop_after": None, "label": "object_to_atom_coverage_diagnostic_driving_mini"},
    {"tag": "18E", "stop_after": None, "label": "traffic_control_rule_utility_diagnostic_driving_mini"},
    {"tag": "18F", "stop_after": None, "label": "traffic_control_temporal_alignment_diagnostic_driving_mini"},
    {"tag": "18G", "stop_after": None, "label": "traffic_light_detection_quality_audit_driving_mini"},
    {"tag": "20", "stop_after": 20, "label": "error_and_explainability_analysis_driving_mini"},
    {"tag": "21", "stop_after": 21, "label": "vehicle_rule_diagnostic_driving_mini"},
    {"tag": "21B", "stop_after": None, "label": "fn_categorization_diagnostic_driving_mini"},
    {"tag": "22", "stop_after": 22, "label": "rule_selection_visualization_driving_mini"},
    {"tag": "23", "stop_after": 23, "label": "integrated_method_visualization_driving_mini"},
    {"tag": "24A", "stop_after": None, "label": "background_causal_prior_driving_mini"},
    {"tag": "24B", "stop_after": 24, "label": "reasoning_feedback_signal_driving_mini"},
]
_PIPELINE_STAGE_TAGS: List[str] = [str(stage["tag"]) for stage in _PIPELINE_STAGE_SEQUENCE]
_PIPELINE_STAGE_BY_TAG: Dict[str, Dict[str, Any]] = {
    str(stage["tag"]): dict(stage) for stage in _PIPELINE_STAGE_SEQUENCE
}


class _PipelineStopRequested(RuntimeError):
    """Raised internally to stop the orchestrator after the requested step."""


class _FilteredStepWriter(io.TextIOBase):
    _UUIDISH_LINE_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{8}(?::|\b)")
    _VIDEO_SUMMARY_RE = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{8}: .*"
    )
    _NOISY_PREFIXES = (
        "processing ",
        "videos to process:",
        "videos to track:",
        "dataset annotation videos to process:",
        "videos to prepare 3d positions for:",
        "rendered ",
        "saved boxed video:",
        "saved tracked video:",
        "saved merged video:",
        "visualization saved",
        "stop_threshold=",
    )
    _NOISY_SUBSTRINGS = (
        " [cache] ",
        "[cache]",
        "cache is stale",
        "cache visualization is missing",
        "cache visualization is stale",
        "frames tracked,",
        "with ego motion estimates",
        "segment-object summaries",
        "objects=",
    )
    _ALLOW_SUBSTRINGS = (
        "[warn]",
        "[error]",
        "traceback",
        "manifest written",
        "json written",
        "csv written",
        "pdf written",
        "report written",
        "summary written",
        "table written",
        "figure written",
        "figure data written",
        "written to ",
        "total detections",
        "top classes",
        "total unique tracks",
    )

    def __init__(
        self,
        base_stream: Any,
        step_tag: str,
        suppressed_substrings: Sequence[str] = ("[cache]",),
    ) -> None:
        self._base_stream = base_stream
        self._step_tag = str(step_tag)
        self._suppressed_substrings = tuple(str(value) for value in suppressed_substrings)
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit_line(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._emit_line(self._buffer)
            self._buffer = ""
        self._base_stream.flush()

    def _emit_line(self, line: str) -> None:
        stripped = str(line).strip()
        if not stripped:
            return
        lowered = stripped.lower()
        if any(token in stripped for token in self._suppressed_substrings):
            return
        if any(token in lowered for token in self._ALLOW_SUBSTRINGS):
            self._base_stream.write(f"[Step {self._step_tag}] {stripped}\n")
            return
        if self._UUIDISH_LINE_RE.match(stripped):
            return
        if self._VIDEO_SUMMARY_RE.match(stripped):
            return
        if any(lowered.startswith(prefix) for prefix in self._NOISY_PREFIXES):
            return
        if any(token in lowered for token in self._NOISY_SUBSTRINGS):
            return
        self._base_stream.write(f"[Step {self._step_tag}] {stripped}\n")


@dataclass(slots=True)
class PipelineContext:
    effective_video_ids: List[str]
    has_explicit_video_selection: bool = False
    recompute_cfg: Dict[str, Any] = field(default_factory=dict)
    od_loop_iteration_index: int = 1
    force_full_iteration_recompute: bool = False
    defer_stop_after_gate: bool = False
    od_loop_stop_reason: str = ""
    background_rule_relevance_prior_results: Dict[str, Any] = field(default_factory=dict)
    detection_results: Optional[List[Dict[str, Any]]] = None
    tracking_results: Optional[List[Dict[str, Any]]] = None
    dataset_annotation_results: Optional[List[Dict[str, Any]]] = None
    merged_results: Optional[List[Dict[str, Any]]] = None
    positions_3d_results: Optional[List[Dict[str, Any]]] = None
    ego_motion_results: Optional[List[Dict[str, Any]]] = None
    relative_motion_results: Optional[List[Dict[str, Any]]] = None
    temporal_seg_results: Optional[List[Dict[str, Any]]] = None
    segment_object_results: Optional[List[Dict[str, Any]]] = None
    important_object_results: Optional[List[Dict[str, Any]]] = None
    traffic_control_attribute_results: Optional[List[Dict[str, Any]]] = None
    logic_atom_results: Optional[List[Dict[str, Any]]] = None
    target_head_results: Optional[List[Dict[str, Any]]] = None
    temporal_rule_results: Optional[List[Dict[str, Any]]] = None
    split_manifest: Dict[str, Any] = field(default_factory=dict)
    train_temporal_rule_results: Optional[List[Dict[str, Any]]] = None
    eval_temporal_rule_results: Optional[List[Dict[str, Any]]] = None
    candidate_rule_results: Optional[List[Dict[str, Any]]] = None
    merged_candidate_rules: Dict[str, Any] = field(default_factory=dict)
    initial_rules_for_extension: Dict[str, Any] = field(default_factory=dict)
    extended_rule_results: Dict[str, Any] = field(default_factory=dict)
    final_rule_results: Dict[str, Any] = field(default_factory=dict)
    diverse_final_rule_results: Dict[str, Any] = field(default_factory=dict)
    semantic_constrained_diverse_rule_results: Dict[str, Any] = field(default_factory=dict)
    coverage_family_aware_rule_results: Dict[str, Any] = field(default_factory=dict)
    rule_results_by_name: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rule_pool_upper_bound_results: Dict[str, Any] = field(default_factory=dict)
    oracle_rule_selection_gap_results: Dict[str, Any] = field(default_factory=dict)
    rule_set_mode: str = "all"
    primary_rule_set: str = "original"
    evaluation_rule_sets: List[str] = field(default_factory=list)
    evaluation_results_by_name: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    candidate_contribution_summary_results: Dict[str, Any] = field(default_factory=dict)
    od_calibration_iteration_id: str = ""
    reasoning_to_od_pseudo_label_results: Dict[str, Any] = field(default_factory=dict)
    od_confidence_calibration_results: Dict[str, Any] = field(default_factory=dict)
    neural_symbolic_baseline_results: Dict[str, Any] = field(default_factory=dict)
    rule_aggregation_baseline_results: Dict[str, Any] = field(default_factory=dict)
    baseline_safe_calibration_gate_results: Dict[str, Any] = field(default_factory=dict)
    object_to_atom_coverage_results: Dict[str, Any] = field(default_factory=dict)
    traffic_control_rule_utility_results: Dict[str, Any] = field(default_factory=dict)
    traffic_control_temporal_alignment_results: Dict[str, Any] = field(default_factory=dict)
    traffic_light_detection_quality_audit_results: Dict[str, Any] = field(default_factory=dict)
    error_analysis_results_by_name: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error_analysis_results: Dict[str, Any] = field(default_factory=dict)
    vehicle_rule_diagnostic_results: Dict[str, Any] = field(default_factory=dict)
    fn_categorization_results: Dict[str, Any] = field(default_factory=dict)
    rule_selection_visualization_results: Dict[str, Any] = field(default_factory=dict)
    integrated_method_visualization_results: Dict[str, Any] = field(default_factory=dict)
    background_causal_prior_results: Dict[str, Any] = field(default_factory=dict)
    reasoning_feedback_signal_results: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepRunner:
    stop_target: str
    stop_label: str
    rtpt: Optional[Any]

    @classmethod
    def create(cls, requested_step: int | str, *, total_rtpt_iterations: int | None = None) -> "StepRunner":
        stop_target, stop_label = _resolve_stop_request(requested_step)
        return cls(
            stop_target=stop_target,
            stop_label=stop_label,
            rtpt=_start_rtpt(stop_target, total_iterations=total_rtpt_iterations),
        )

    def announce_step(self, tag: str, name: str, *, leading_newline: bool = True) -> None:
        prefix = "\n" if leading_newline else ""
        print(f"{prefix}[Step {tag}] {name}")

    def log(self, tag: str, message: str) -> None:
        print(f"[Step {tag}] {message}")

    def complete_step(self, tag: str, subtitle: str = "", *, allow_stop: bool = True) -> None:
        _rtpt_step(self.rtpt, tag, subtitle=subtitle)
        if allow_stop and str(tag) == self.stop_target:
            self.stop_now()

    def stop_now(self) -> None:
        print(f"\nStopping after step {self.stop_label} by request.")
        raise _PipelineStopRequested()

    @contextmanager
    def module_output(self, tag: str) -> Any:
        writer = _FilteredStepWriter(sys.stdout, step_tag=tag)
        with redirect_stdout(writer), redirect_stderr(writer):
            yield
        writer.flush()


def _normalize_requested_step(value: int | str) -> str:
    return str(value).strip().upper()


def _resolve_stop_request(requested_step: int | str) -> Tuple[str, str]:
    normalized = _normalize_requested_step(requested_step)
    if normalized.isdigit():
        step_number = int(normalized)
        if step_number < 0 or step_number > 24:
            raise ValueError(f"Unsupported max_step: {requested_step}")
        default_targets = [
            str(stage["tag"])
            for stage in _PIPELINE_STAGE_SEQUENCE
            if stage["stop_after"] == step_number
        ]
        resolved_target = default_targets[-1] if default_targets else str(step_number)
        return resolved_target, str(step_number)
    if normalized not in _PIPELINE_STAGE_BY_TAG:
        raise ValueError(f"Unsupported step id: {requested_step}")
    return normalized, normalized


def _parse_max_step_arg(value: str) -> str:
    normalized = _normalize_requested_step(value)
    try:
        _resolve_stop_request(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return normalized


def _rtpt_max_iterations(stop_target: str, total_iterations: int | None = None) -> int:
    if total_iterations is not None and total_iterations > 0:
        return int(total_iterations)
    return _PIPELINE_STAGE_TAGS.index(stop_target) + 1


def _start_rtpt(stop_target: str, *, total_iterations: int | None = None) -> Optional[Any]:
    if RTPT is None:
        print("RTPT unavailable; continuing without remote progress monitor.")
        return None
    rtpt = RTPT(
        name_initials="JI",
        experiment_name="DrivingMiniPipeline",
        max_iterations=_rtpt_max_iterations(stop_target, total_iterations=total_iterations),
    )
    rtpt.start()
    return rtpt


def _rtpt_step(rtpt: Optional[Any], stage_tag: str, subtitle: str = "") -> None:
    if rtpt is None:
        return
    stage_subtitle = f"step={stage_tag}"
    if subtitle:
        stage_subtitle = f"{stage_subtitle} | {subtitle}"
    rtpt.step(subtitle=stage_subtitle)


def _stage_index(tag: str) -> int:
    return _PIPELINE_STAGE_TAGS.index(str(tag))


def _stop_target_reaches(tag: str, stop_target: str) -> bool:
    return _stage_index(stop_target) >= _stage_index(tag)


def _start_target_reaches(tag: str, start_target: str) -> bool:
    return _stage_index(start_target) <= _stage_index(tag)


def _step_is_requested(tag: str, *, start_target: str, stop_target: str) -> bool:
    step_index = _stage_index(tag)
    return _stage_index(start_target) <= step_index <= _stage_index(stop_target)


def _force_recompute(ctx: PipelineContext, key: str | None = None, default: bool = False) -> bool:
    if ctx.force_full_iteration_recompute:
        return True
    if key is None:
        return bool(default)
    return bool(ctx.recompute_cfg.get(key, default))


def _resolve_od_calibration_loop_cfg(
    *,
    max_iterations_override: int | None = None,
) -> od_calibration_loop_utils.ODCalibrationLoopConfig:
    gate_cfg = _get_baseline_safe_calibration_gate_cfg()
    loop_cfg = dict(_get_od_calibration_loop_cfg())
    if max_iterations_override is not None:
        loop_cfg["max_iterations"] = max_iterations_override
    return od_calibration_loop_utils.normalize_loop_cfg(
        loop_cfg,
        improvement_tolerance=float(gate_cfg.get("final_f1_tolerance", 1e-9)),
    )


def _should_run_od_calibration_loop(
    stop_target: str,
    loop_cfg: od_calibration_loop_utils.ODCalibrationLoopConfig,
) -> bool:
    calibration_cfg = _get_reasoning_supervised_od_calibration_cfg()
    return (
        loop_cfg.max_iterations > 1
        and bool(calibration_cfg.get("enabled", False))
        and _stop_target_reaches("19", stop_target)
    )


def _estimated_rtpt_iterations(
    stop_target: str,
    loop_cfg: od_calibration_loop_utils.ODCalibrationLoopConfig,
) -> int:
    if not _should_run_od_calibration_loop(stop_target, loop_cfg):
        return _rtpt_max_iterations(stop_target)
    through_gate = _stage_index("19") + 1
    post_loop_steps = max(0, _stage_index(stop_target) - _stage_index("19"))
    return through_gate * int(loop_cfg.max_iterations) + post_loop_steps


def _build_pipeline_context(
    *,
    video_ids: Optional[List[str]],
    video_count: int | None,
    iteration_index: int = 1,
    force_full_recompute: bool = False,
    defer_stop_after_gate: bool = False,
) -> PipelineContext:
    return PipelineContext(
        effective_video_ids=_resolve_video_ids(video_ids, video_count),
        has_explicit_video_selection=bool(video_ids) or video_count is not None,
        recompute_cfg=od_calibration_loop_utils.apply_iteration_recompute_overrides(
            _get_pipeline_recompute_cfg(),
            force_full_recompute=force_full_recompute,
        ),
        od_loop_iteration_index=int(iteration_index),
        force_full_iteration_recompute=bool(force_full_recompute),
        defer_stop_after_gate=bool(defer_stop_after_gate),
    )


def _run_object_detection_step(
    force_recompute: bool = False,
    video_ids: Optional[List[str]] = None,
    render_video: bool = True,
    check_cache: bool = False,
    od_calibration_policy: Optional[Dict[str, Any]] = None,
    background_rule_relevance_prior_results: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    return detect_driving_mini.run(
        video_ids=video_ids,
        model_name=DRIVING_MINI_OD_MODEL,
        classes=DRIVING_MINI_OD_CLASSES,
        force_recompute=force_recompute,
        render_video=render_video,
        check_cache=check_cache,
        od_calibration_policy=od_calibration_policy,
        background_rule_relevance_prior_results=background_rule_relevance_prior_results,
    )


def _resolve_rule_evaluation_plan(rule_evaluation_cfg: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    rule_set_mode = str(rule_evaluation_cfg.get("rule_set_mode", "all"))
    primary_rule_set = str(rule_evaluation_cfg.get("primary_rule_set", "original"))
    valid_rule_sets = {"original", "diverse", "semantic_constrained_diverse", "coverage_family_aware"}
    valid_modes = valid_rule_sets | {"both", "all"}
    if rule_set_mode not in valid_modes:
        raise ValueError(f"Unsupported rule_evaluation.rule_set_mode: {rule_set_mode}")
    if primary_rule_set not in valid_rule_sets:
        raise ValueError(f"Unsupported rule_evaluation.primary_rule_set: {primary_rule_set}")
    if rule_set_mode not in {"both", "all"} and primary_rule_set != rule_set_mode:
        primary_rule_set = rule_set_mode

    if rule_set_mode == "both":
        evaluation_rule_sets = ["original", "diverse"]
    elif rule_set_mode == "all":
        evaluation_rule_sets = ["original", "diverse", "semantic_constrained_diverse", "coverage_family_aware"]
    else:
        evaluation_rule_sets = [rule_set_mode]
    if primary_rule_set not in evaluation_rule_sets:
        primary_rule_set = evaluation_rule_sets[0]
    return rule_set_mode, primary_rule_set, evaluation_rule_sets


def _selected_selector_cfg(final_rules_cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[Path], Any]:
    selector_mode = str(final_rules_cfg.get("selector_mode", "coverage_family_aware")).strip() or "coverage_family_aware"
    normalized_mode = selector_mode.lower()
    shared_overrides = dict(final_rules_cfg)
    shared_overrides.pop("selector_mode", None)

    output_root = final_rules_driving_mini.get_output_root()
    if normalized_mode in {"score_top_k", "top_k", "marginal_f1_score_top_k"}:
        selector_cfg = dict(shared_overrides)
        return normalized_mode, selector_cfg, None, final_rules_driving_mini

    if normalized_mode in {"diverse_greedy", "legacy_diverse_positive_coverage", "greedy_diverse_positive_coverage"}:
        selector_cfg = dict(_get_diverse_final_rules_cfg())
        selector_cfg.update(shared_overrides)
        selector_cfg["output_prefix"] = "final_rules"
        selector_cfg.setdefault("selection_method_name", "greedy_diverse_positive_coverage")
        return normalized_mode, selector_cfg, output_root, diverse_final_rules_driving_mini

    if normalized_mode in {"semantic_constrained_diverse", "semantic_constrained"}:
        selector_cfg = dict(_get_semantic_constrained_diverse_cfg())
        selector_cfg.update(shared_overrides)
        selector_cfg["output_prefix"] = "final_rules"
        return normalized_mode, selector_cfg, output_root, diverse_final_rules_driving_mini

    if normalized_mode in {
        "coverage_family_aware",
        "validation_greedy",
        "marginal_f1",
        "small_cardinality_validation_greedy",
    }:
        selector_cfg = dict(_get_coverage_family_aware_final_rules_cfg())
        selector_cfg.update(shared_overrides)
        selector_cfg["output_prefix"] = "final_rules"
        if normalized_mode == "validation_greedy":
            selector_cfg["selection_method_name"] = "validation_greedy"
        elif normalized_mode == "marginal_f1":
            selector_cfg["selection_method_name"] = "marginal_f1"
        return normalized_mode, selector_cfg, output_root, diverse_final_rules_driving_mini

    raise ValueError(f"Unsupported final_rules.selector_mode: {selector_mode}")


def _synthesize_selected_candidate_contribution_summary(
    final_rule_results: Dict[str, Any],
    evaluation_result: Dict[str, Any],
) -> Dict[str, Any]:
    ablation = dict(evaluation_result.get("candidate_rule_ablation", {}))
    selector_row = {
        "rule_set_name": "selected",
        "selection_method": str(final_rule_results.get("selection_method", "")),
        "delta_precision": float(ablation.get("delta_precision", 0.0)),
        "delta_recall": float(ablation.get("delta_recall", 0.0)),
        "delta_f1": float(ablation.get("delta_f1", 0.0)),
        "mixed_delta_f1": float(ablation.get("mixed_delta_f1", 0.0)),
        "fn_coverage_gain_count": int(ablation.get("fn_coverage_gain_count", 0)),
        "fp_contribution_count": int(ablation.get("fp_contribution_count", 0)),
        "mixed_fn_coverage_gain_count": int(ablation.get("mixed_fn_coverage_gain_count", 0)),
        "mixed_fp_contribution_count": int(ablation.get("mixed_fp_contribution_count", 0)),
        "candidate_only_fn_coverage_gain_count": int(ablation.get("candidate_only_fn_coverage_gain_count", 0)),
        "candidate_only_fp_contribution_count": int(ablation.get("candidate_only_fp_contribution_count", 0)),
        "candidate_candidate_fn_coverage_gain_count": int(
            ablation.get("candidate_candidate_fn_coverage_gain_count", 0)
        ),
        "candidate_candidate_fp_contribution_count": int(
            ablation.get("candidate_candidate_fp_contribution_count", 0)
        ),
    }
    return {
        "best_selector_by_delta_f1": "selected",
        "selectors": [selector_row],
    }


def _candidate_signal_is_useful(
    evaluation_result: Dict[str, Any],
    calibration_cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    ablation = dict(evaluation_result.get("candidate_rule_ablation", {}))
    delta_f1 = float(ablation.get("delta_f1", 0.0))
    fn_gain = int(ablation.get("fn_coverage_gain_count", 0))
    min_delta_f1 = float(calibration_cfg.get("min_delta_f1", 0.0))
    min_fn_gain = int(calibration_cfg.get("min_fn_gain", 1))
    allow_nonpositive = bool(calibration_cfg.get("allow_nonpositive_delta_with_fn_recovery", True))
    if delta_f1 > min_delta_f1:
        return True, f"delta_f1={delta_f1:.3f}"
    if allow_nonpositive and fn_gain >= min_fn_gain:
        return True, f"fn_gain={fn_gain}"
    return False, f"delta_f1={delta_f1:.3f}, fn_gain={fn_gain}"


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            serialized: Dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple, set)):
                    serialized[key] = json.dumps(value, sort_keys=isinstance(value, dict))
                else:
                    serialized[key] = value
            writer.writerow(serialized)


def _write_manifest_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _read_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}.")
    return dict(payload)


def _read_cached_json(path: Path, *, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(
            f"Missing cached {label} at {path}. "
            "Run the upstream steps once or restart the pipeline from an earlier step."
        )
    return _read_json_file(path)


def _load_cached_video_results(
    *,
    output_root: Path,
    manifest_name: str,
    per_video_filename: str,
    selected_video_ids: Optional[Sequence[str]] = None,
    label: str,
) -> List[Dict[str, Any]]:
    manifest_path = output_root / manifest_name
    manifest = _read_cached_json(manifest_path, label=f"{label} manifest")
    manifest_video_ids = [
        str(entry.get("video_id", "")).strip()
        for entry in list(manifest.get("videos", []))
        if str(entry.get("video_id", "")).strip()
    ]
    requested_video_ids = (
        [str(video_id).strip() for video_id in selected_video_ids if str(video_id).strip()]
        if selected_video_ids
        else manifest_video_ids
    )
    missing_video_ids = sorted(set(requested_video_ids) - set(manifest_video_ids))
    if missing_video_ids:
        raise RuntimeError(
            f"Cached {label} does not contain requested videos: {missing_video_ids}. "
            "Restart from an earlier step to rebuild the cached artifacts for this selection."
        )
    results: List[Dict[str, Any]] = []
    for video_id in requested_video_ids:
        result_path = output_root / video_id / per_video_filename
        results.append(_read_cached_json(result_path, label=f"{label} result for {video_id}"))
    return results


def _cached_manifest_video_ids(
    *,
    output_root: Path,
    manifest_name: str,
) -> List[str]:
    manifest = _read_cached_json(output_root / manifest_name, label=f"{manifest_name} manifest")
    return [
        str(entry.get("video_id", "")).strip()
        for entry in list(manifest.get("videos", []))
        if str(entry.get("video_id", "")).strip()
    ]


def _load_cached_detection_results(selected_video_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    output_root = detect_driving_mini.get_output_root()
    manifest = _read_cached_json(output_root / "detection_manifest.json", label="step 1 detection manifest")
    video_entries = {
        str(entry.get("video_id", "")).strip(): dict(entry)
        for entry in list(manifest.get("videos", []))
        if str(entry.get("video_id", "")).strip()
    }
    requested_video_ids = (
        [str(video_id).strip() for video_id in selected_video_ids if str(video_id).strip()]
        if selected_video_ids
        else list(video_entries.keys())
    )
    missing_video_ids = sorted(set(requested_video_ids) - set(video_entries.keys()))
    if missing_video_ids:
        raise RuntimeError(
            f"Cached step 1 detection results do not contain requested videos: {missing_video_ids}. "
            "Restart from step 1 to rebuild the cache for this selection."
        )
    results: List[Dict[str, Any]] = []
    for video_id in requested_video_ids:
        entry = video_entries[video_id]
        detections_path_text = str(entry.get("detections_json", "")).strip()
        detections_path = Path(detections_path_text) if detections_path_text else output_root / video_id / "detections.json"
        results.append(_read_cached_json(detections_path, label=f"step 1 detection result for {video_id}"))
    return results


def _load_cached_step17_result() -> Dict[str, Any]:
    manifest = _read_cached_json(
        final_rules_driving_mini.get_output_root() / "17_final_rule_selection_manifest.json",
        label="step 17 final rule selection manifest",
    )
    for artifact in list(manifest.get("secondary_debug_artifacts", [])):
        artifact_path = Path(str(artifact))
        if artifact_path.suffix.lower() == ".json" and artifact_path.name == "final_rules.json":
            return _read_cached_json(artifact_path, label="step 17 final rule selection result")
    return _read_cached_json(
        final_rules_driving_mini.get_output_root() / "final_rules.json",
        label="step 17 final rule selection result",
    )


def _load_cached_step18_result() -> Dict[str, Any]:
    manifest = _read_cached_json(
        _get_rule_evaluation_output_root() / "18_rule_evaluation_manifest.json",
        label="step 18 rule evaluation manifest",
    )
    for artifact in list(manifest.get("secondary_debug_artifacts", [])):
        artifact_path = Path(str(artifact))
        if artifact_path.suffix.lower() == ".json" and artifact_path.name == "rule_evaluation.json":
            return _read_cached_json(artifact_path, label="step 18 rule evaluation result")
    return _read_cached_json(
        _get_rule_evaluation_output_root() / "rule_evaluation.json",
        label="step 18 rule evaluation result",
    )


def _load_cached_upstream_context(
    ctx: PipelineContext,
    *,
    start_target: str,
    stop_target: str,
) -> None:
    if _stage_index(start_target) <= _stage_index("16"):
        return
    if ctx.has_explicit_video_selection:
        cached_video_ids = _cached_manifest_video_ids(
            output_root=temporal_rule_examples_driving_mini.get_output_root(),
            manifest_name="temporal_rule_examples_manifest.json",
        )
        requested_video_ids = [str(video_id).strip() for video_id in ctx.effective_video_ids if str(video_id).strip()]
        if requested_video_ids != cached_video_ids:
            raise RuntimeError(
                "Warm-starting after step 16 only supports an explicit video selection when it exactly "
                "matches the cached upstream video set. "
                f"requested_videos={len(requested_video_ids)} cached_videos={len(cached_video_ids)}. "
                "Restart from step 0-16 to rebuild for a different subset."
            )

    ctx.extended_rule_results = _read_cached_json(
        extended_rules_driving_mini.get_output_root() / "extended_rules_manifest.json",
        label="step 16 extended rules result",
    )

    needs_temporal_examples = any(
        _stop_target_reaches(tag, stop_target)
        for tag in ("17D", "18", "18B", "19", "18E", "18F", "18G", "20", "21", "21B", "24B")
    )
    if needs_temporal_examples:
        ctx.temporal_rule_results = _load_cached_video_results(
            output_root=temporal_rule_examples_driving_mini.get_output_root(),
            manifest_name="temporal_rule_examples_manifest.json",
            per_video_filename="temporal_rule_examples.json",
            selected_video_ids=ctx.effective_video_ids,
            label="step 13 temporal rule examples",
        )
        _prepare_rule_learning_inputs(ctx)

    needs_logic_atoms = (
        (
            bool(_get_reasoning_supervised_od_calibration_cfg().get("enabled", False))
            and _stop_target_reaches("19", stop_target)
        )
        or (
            bool(_get_traffic_control_temporal_alignment_diagnostic_cfg().get("enabled", False))
            and _stop_target_reaches("18F", stop_target)
        )
        or (
            bool(_get_traffic_light_detection_quality_audit_cfg().get("enabled", False))
            and _stop_target_reaches("18G", stop_target)
        )
        or (
            bool(_get_reasoning_feedback_signal_cfg().get("enabled", False))
            and _stop_target_reaches("24B", stop_target)
        )
    )
    if needs_logic_atoms:
        ctx.logic_atom_results = _load_cached_video_results(
            output_root=logic_atoms_driving_mini.get_output_root(),
            manifest_name="logic_atoms_manifest.json",
            per_video_filename="logic_atoms.json",
            selected_video_ids=ctx.effective_video_ids,
            label="step 11 logic atoms",
        )

    if (
        bool(_get_traffic_light_detection_quality_audit_cfg().get("enabled", False))
        and _stop_target_reaches("18G", stop_target)
    ):
        ctx.traffic_control_attribute_results = _load_cached_video_results(
            output_root=traffic_control_attributes_driving_mini.get_output_root(),
            manifest_name="traffic_control_attributes_manifest.json",
            per_video_filename="traffic_control_attributes.json",
            selected_video_ids=ctx.effective_video_ids,
            label="step 10B traffic control attributes",
        )

    if (
        bool(_get_object_to_atom_coverage_diagnostic_cfg().get("enabled", False))
        and _stop_target_reaches("18D", stop_target)
    ):
        ctx.detection_results = _load_cached_detection_results(ctx.effective_video_ids)
        ctx.dataset_annotation_results = _load_cached_video_results(
            output_root=dataset_annotations_driving_mini.get_output_root(),
            manifest_name="dataset_annotations_manifest.json",
            per_video_filename="dataset_annotations.json",
            selected_video_ids=ctx.effective_video_ids,
            label="step 3 dataset annotations",
        )
        ctx.merged_results = _load_cached_video_results(
            output_root=merge_gt_and_detected_driving_mini.get_output_root(),
            manifest_name="merged_annotations_manifest.json",
            per_video_filename="merged_annotations.json",
            selected_video_ids=ctx.effective_video_ids,
            label="step 4 merged annotations",
        )
        ctx.important_object_results = _load_cached_video_results(
            output_root=important_objects_driving_mini.get_output_root(),
            manifest_name="important_objects_manifest.json",
            per_video_filename="important_objects.json",
            selected_video_ids=ctx.effective_video_ids,
            label="step 10 important objects",
        )

    if (
        bool(_get_vehicle_rule_diagnostic_cfg().get("enabled", False))
        and _stop_target_reaches("21", stop_target)
    ):
        ctx.merged_candidate_rules = _read_cached_json(
            _get_merged_candidate_rules_output_root() / "merged_initial_rules.json",
            label="step 15 merged initial rules result",
        )

    if _stage_index(start_target) > _stage_index("17"):
        ctx.final_rule_results = _load_cached_step17_result()
        ctx.rule_results_by_name = {"selected": ctx.final_rule_results}
        ctx.primary_rule_set = "selected"
        ctx.rule_set_mode = "selected"
        ctx.evaluation_rule_sets = ["selected"]

    if _stage_index(start_target) > _stage_index("18"):
        ctx.evaluation_results = _load_cached_step18_result()
        ctx.evaluation_results_by_name = {"selected": ctx.evaluation_results}
        ctx.candidate_contribution_summary_results = _synthesize_selected_candidate_contribution_summary(
            ctx.final_rule_results,
            ctx.evaluation_results,
        )

def _rule_body_length(rule: Dict[str, Any]) -> int:
    templates = rule.get("body_atom_templates")
    if isinstance(templates, list):
        return len([str(atom).strip() for atom in templates if str(atom).strip()])
    template = str(rule.get("body_atom_template", "")).strip()
    return 1 if template else 0


def _rule_signature_key(rule: Dict[str, Any]) -> str:
    firing_signature = rule.get("firing_signature")
    if firing_signature:
        try:
            return json.dumps(firing_signature, sort_keys=True)
        except TypeError:
            return str(firing_signature)
    return str(rule.get("rule_id", ""))


def _rule_positive_coverage_key(rule: Dict[str, Any]) -> str:
    positive_ids = sorted(str(value) for value in list(rule.get("positive_example_ids", [])) if str(value))
    return "|".join(positive_ids)


def _step17_summary_row(
    *,
    rule_set_name: str,
    selector_name: str,
    final_rule_results: Dict[str, Any],
) -> Dict[str, Any]:
    final_rules = list(final_rule_results.get("final_rules", []))
    diagnostics = dict(final_rule_results.get("candidate_rule_diagnostics", {}))
    category_counts = dict(diagnostics.get("category_counts", {}))
    num_rules = len(final_rules)
    avg_body_length = float(sum(_rule_body_length(rule) for rule in final_rules) / max(1, num_rules))
    avg_confidence = float(sum(float(rule.get("confidence", 0.0)) for rule in final_rules) / max(1, num_rules))
    avg_positive_support = float(
        sum(int(rule.get("positive_support", 0)) for rule in final_rules) / max(1, num_rules)
    )
    avg_negative_support = float(
        sum(int(rule.get("negative_support", 0)) for rule in final_rules) / max(1, num_rules)
    )
    unique_firing_signatures = len({_rule_signature_key(rule) for rule in final_rules})
    unique_positive_coverage_sets = len({_rule_positive_coverage_key(rule) for rule in final_rules})
    accepted_only_count = int(category_counts.get("accepted_only_rules", 0))
    mixed_count = int(category_counts.get("mixed_accepted_candidate_rules", 0))
    candidate_only_count = int(category_counts.get("candidate_only_rules", 0))
    candidate_candidate_count = int(category_counts.get("candidate_candidate_rules", 0))
    candidate_involving = mixed_count + candidate_only_count + candidate_candidate_count
    if candidate_involving > 0:
        main_conclusion = (
            f"Selected {candidate_involving} candidate-involving rules while keeping "
            f"{accepted_only_count} accepted-only baseline rules."
        )
    else:
        main_conclusion = "Selection stayed on the accepted-only baseline branch."
    warning_flag = ""
    if accepted_only_count == 0:
        warning_flag = "no_accepted_only_baseline"
    elif candidate_candidate_count > 0:
        warning_flag = "candidate_candidate_selected"
    elif unique_firing_signatures < max(1, num_rules // 2):
        warning_flag = "low_signature_diversity"
    return {
        "rule_set_name": rule_set_name,
        "selector_name": selector_name,
        "num_rules": num_rules,
        "accepted_only_count": accepted_only_count,
        "mixed_accepted_candidate_count": mixed_count,
        "candidate_only_count": candidate_only_count,
        "candidate_candidate_count": candidate_candidate_count,
        "average_body_length": avg_body_length,
        "average_confidence": avg_confidence,
        "average_train_positive_support": avg_positive_support,
        "average_train_negative_support": avg_negative_support,
        "num_unique_firing_signatures": unique_firing_signatures,
        "num_unique_positive_coverage_sets": unique_positive_coverage_sets,
        "selection_objective": str(final_rule_results.get("selection_method", selector_name)),
        "main_conclusion": main_conclusion,
        "warning_flag": warning_flag,
    }


def _write_step17_primary_summary(
    final_rule_results: Dict[str, Any],
    *,
    output_root: Path,
    rule_set_name: str = "selected",
) -> Tuple[Path, Dict[str, Any]]:
    summary_path = output_root / "17_final_rule_selection_summary.csv"
    row = _step17_summary_row(
        rule_set_name=rule_set_name,
        selector_name=str(final_rule_results.get("selection_method", "")) or rule_set_name,
        final_rule_results=final_rule_results,
    )
    _write_csv_rows(summary_path, list(row.keys()), [row])
    return summary_path, row


def _write_step17d_primary_summary(
    rule_pool_upper_bound_results: Dict[str, Any],
    *,
    output_root: Path,
) -> Tuple[Path, List[Dict[str, Any]]]:
    summary_path = output_root / "17d_rule_pool_diagnostic_summary.csv"
    category_upper_bounds = dict(rule_pool_upper_bound_results.get("category_upper_bounds", {}))
    gap_by_selector = dict(rule_pool_upper_bound_results.get("category_oracle_gap_by_selector", {}))
    selector_name = str(rule_pool_upper_bound_results.get("best_actual_selector_name", "selected")) or "selected"
    selector_gap_rows = dict(gap_by_selector.get(selector_name, {}))
    bottleneck_label = str(rule_pool_upper_bound_results.get("bottleneck_label", "unknown"))
    rows: List[Dict[str, Any]] = []
    for subset_name in (
        "accepted_only",
        "candidate_only",
        "mixed_accepted_candidate",
        "accepted_plus_mixed",
        "all_candidate_involving",
        "all_rules",
    ):
        subset = dict(category_upper_bounds.get(subset_name, {}))
        selector_gap = dict(selector_gap_rows.get(subset_name, {}))
        oracle_f1 = float(subset.get("f1", 0.0))
        selector_f1 = float(selector_gap.get("selector_f1", 0.0))
        oracle_gap_f1 = float(selector_gap.get("oracle_gap_f1", oracle_f1 - selector_f1))
        fn_recovery = int(subset.get("fn_recovery_beyond_accepted_only", 0))
        if oracle_gap_f1 > 0.05:
            main_conclusion = f"Selection is trailing the oracle upper bound for {subset_name}."
        elif fn_recovery > 0:
            main_conclusion = f"{subset_name} recovers extra positives beyond accepted-only with limited selector gap."
        else:
            main_conclusion = f"{subset_name} offers limited additional held-out upside."
        warning_flag = ""
        if int(subset.get("num_pool_rules", 0)) == 0:
            warning_flag = "empty_subset_pool"
        elif oracle_gap_f1 > 0.10:
            warning_flag = "large_oracle_gap"
        elif int(subset.get("fp_cost", 0)) > int(subset.get("max_positive_coverage_count", 0)):
            warning_flag = "high_fp_cost"
        rows.append(
            {
                "rule_subset_name": subset_name,
                "num_pool_rules": int(subset.get("num_pool_rules", 0)),
                "best_oracle_k": int(subset.get("best_oracle_k", 0)),
                "oracle_precision": float(subset.get("precision", 0.0)),
                "oracle_recall": float(subset.get("recall", 0.0)),
                "oracle_f1": oracle_f1,
                "max_positive_coverage": int(subset.get("max_positive_coverage_count", 0)),
                "fp_cost": int(subset.get("fp_cost", 0)),
                "fn_recovery_beyond_accepted_only": fn_recovery,
                "best_actual_selector_f1": selector_f1,
                "oracle_gap_f1": oracle_gap_f1,
                "bottleneck_label": bottleneck_label,
                "main_conclusion": main_conclusion,
                "warning_flag": warning_flag,
            }
        )
    _write_csv_rows(summary_path, list(rows[0].keys()) if rows else [], rows)
    return summary_path, rows


def _write_step18_primary_summary(
    evaluation_result: Dict[str, Any],
    *,
    output_root: Path,
) -> Tuple[Path, List[Dict[str, Any]]]:
    summary_path = output_root / "18_rule_evaluation_summary.csv"
    subset_metrics = dict(evaluation_result.get("rule_subset_metrics", {}))
    ablation = dict(evaluation_result.get("candidate_rule_ablation", {}))
    candidate_only_net = int(ablation.get("candidate_only_fn_coverage_gain_count", 0)) - int(
        ablation.get("candidate_only_fp_contribution_count", 0)
    )
    mixed_net = int(ablation.get("mixed_fn_coverage_gain_count", 0)) - int(
        ablation.get("mixed_fp_contribution_count", 0)
    )
    rows: List[Dict[str, Any]] = []
    for subset_name, subset in subset_metrics.items():
        subset = dict(subset)
        overall = dict(subset.get("overall_metrics", {}))
        fn_recovery = int(subset.get("fn_coverage_gain_count_vs_accepted_only", 0))
        fp_added = int(subset.get("fp_contribution_count_vs_accepted_only", 0))
        precision = float(overall.get("precision", 0.0))
        recall = float(overall.get("recall", 0.0))
        f1 = float(overall.get("f1", 0.0))
        if subset_name == "accepted_plus_all_candidate_rules":
            if fn_recovery > fp_added:
                main_conclusion = "Candidate augmentation adds more FN recovery than FP cost."
            else:
                main_conclusion = "Candidate augmentation adds limited net value over accepted-only."
        elif subset_name == "accepted_only_rules":
            main_conclusion = "Protected accepted-only baseline reference."
        else:
            main_conclusion = f"{subset_name} reaches F1={f1:.3f} on held-out evaluation."
        warning_flag = ""
        if subset_name != "accepted_only_rules" and fp_added > fn_recovery:
            warning_flag = "fp_cost_exceeds_fn_recovery"
        elif precision < 0.5 and int(subset.get("num_rules", 0)) > 0:
            warning_flag = "low_precision_subset"
        rows.append(
            {
                "rule_subset_name": subset_name,
                "num_rules": int(subset.get("num_rules", 0)),
                "TP": int(overall.get("true_positive", 0)),
                "FP": int(overall.get("false_positive", 0)),
                "FN": int(overall.get("false_negative", 0)),
                "TN": int(overall.get("true_negative", 0)),
                "precision": precision,
                "recall": recall,
                "F1": f1,
                "accuracy": float(overall.get("accuracy", 0.0)),
                "FN_recovery_vs_accepted_only": fn_recovery,
                "FP_added_vs_accepted_only": fp_added,
                "candidate_only_contribution": candidate_only_net,
                "mixed_rule_contribution": mixed_net,
                "main_conclusion": main_conclusion,
                "warning_flag": warning_flag,
            }
        )
    _write_csv_rows(summary_path, list(rows[0].keys()) if rows else [], rows)
    return summary_path, rows


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _upsert_csv_row(path: Path, fieldnames: Sequence[str], row: Dict[str, Any], key_name: str) -> None:
    existing_rows = _read_csv_rows(path)
    updated = False
    key_value = str(row.get(key_name, ""))
    for index, existing in enumerate(existing_rows):
        if str(existing.get(key_name, "")) == key_value:
            existing_rows[index] = dict(row)
            updated = True
            break
    if not updated:
        existing_rows.append(dict(row))
    _write_csv_rows(path, fieldnames, existing_rows)


def _step19_summary_row(ctx: PipelineContext) -> Dict[str, Any]:
    iteration_id = str(ctx.od_calibration_iteration_id or f"iteration_{ctx.od_loop_iteration_index}")
    pseudo_report = dict(ctx.reasoning_to_od_pseudo_label_results.get("low_confidence_od_contribution_report", {}))
    calibration = dict(ctx.od_confidence_calibration_results)
    gate = dict(ctx.baseline_safe_calibration_gate_results)
    policy = dict(calibration.get("policy", {}))
    reference_metrics = dict(gate.get("reference_metrics", {}))
    current_metrics = dict(gate.get("current_metrics", {}))
    positive_count = int(pseudo_report.get("num_positive_detection_pseudo_labels", 0))
    negative_count = int(pseudo_report.get("num_negative_detection_pseudo_labels", 0))
    total_polar = positive_count + negative_count
    before_accepted_only = float(reference_metrics.get("accepted_only_f1", current_metrics.get("accepted_only_f1", 0.0)))
    before_candidate = float(
        reference_metrics.get("accepted_plus_candidate_f1", current_metrics.get("accepted_plus_candidate_f1", 0.0))
    )
    before_final = float(reference_metrics.get("final_f1", current_metrics.get("final_f1", 0.0)))
    after_accepted_only = float(current_metrics.get("accepted_only_f1", 0.0))
    after_candidate = float(current_metrics.get("accepted_plus_candidate_f1", 0.0))
    after_final = float(current_metrics.get("final_f1", 0.0))
    decision = str(gate.get("decision", "reject"))
    decision_reason = str(gate.get("decision_reason", "not_run"))
    protected_baseline_regressed = decision_reason == "protected_baseline_regressed"
    policy_promoted = decision == "accept" and bool(str(gate.get("active_policy_after_id", "")).strip())
    if decision == "accept":
        main_conclusion = f"Calibration policy promoted with final F1={after_final:.3f}."
    else:
        main_conclusion = f"Calibration not promoted ({decision_reason})."
    warning_flag = ""
    if protected_baseline_regressed:
        warning_flag = "protected_baseline_regressed"
    elif decision_reason == "final_f1_regressed":
        warning_flag = "final_f1_regressed"
    elif total_polar == 0:
        warning_flag = "no_polar_pseudo_labels"
    return {
        "iteration_id": iteration_id,
        "active_policy_before": str(gate.get("active_policy_before_id", "")),
        "active_policy_after": str(gate.get("active_policy_after_id", "")),
        "proposed_policy": str(gate.get("proposed_policy_id", "")),
        "gate_decision": decision,
        "decision_reason": decision_reason,
        "pseudo_label_count": int(pseudo_report.get("num_detection_pseudo_labels", 0)),
        "positive_pseudo_labels": positive_count,
        "negative_pseudo_labels": negative_count,
        "positive_negative_ratio": float(positive_count / max(1, negative_count)),
        "policy_type": str(policy.get("policy_type", "")),
        "num_training_rows": int(calibration.get("num_training_rows", 0)),
        "before_accepted_only_f1": before_accepted_only,
        "after_accepted_only_f1": after_accepted_only,
        "before_accepted_plus_candidate_f1": before_candidate,
        "after_accepted_plus_candidate_f1": after_candidate,
        "before_final_f1": before_final,
        "after_final_f1": after_final,
        "protected_baseline_regressed": protected_baseline_regressed,
        "policy_promoted": policy_promoted,
        "main_conclusion": main_conclusion,
        "warning_flag": warning_flag,
    }


def _write_rule_set_evaluation_comparison(
    evaluation_output_root: Path,
    primary_rule_set: str,
    rule_set_mode: str,
    evaluation_rule_sets: Sequence[str],
    rule_results_by_name: Dict[str, Dict[str, Any]],
    evaluation_results_by_name: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    comparison_rows: List[Dict[str, Any]] = []
    original_fn_ids = set(evaluation_results_by_name.get("original", {}).get("false_negative_example_ids", []))
    for rule_set_name in evaluation_rule_sets:
        overall = dict(evaluation_results_by_name[rule_set_name].get("overall_metrics", {}))
        rule_eval_result = evaluation_results_by_name[rule_set_name]
        ablation = dict(rule_eval_result.get("candidate_rule_ablation", {}))
        candidate_rule_diagnostics = dict(rule_results_by_name[rule_set_name].get("candidate_rule_diagnostics", {}))
        category_counts = dict(candidate_rule_diagnostics.get("category_counts", {}))
        all_subset = dict(dict(candidate_rule_diagnostics.get("subsets", {})).get("all_rules", {}))
        predicted_positive_ids = set(rule_eval_result.get("predicted_positive_example_ids", []))
        covered_original_fn_ids = original_fn_ids & predicted_positive_ids
        comparison_rows.append(
            {
                "rule_set_name": rule_set_name,
                "selection_method": str(rule_results_by_name[rule_set_name].get("selection_method", "score_top_k")),
                "num_final_rules": int(rule_results_by_name[rule_set_name].get("num_final_rules", 0)),
                "num_accepted_only_rules": int(category_counts.get("accepted_only_rules", 0)),
                "num_candidate_only_rules": int(category_counts.get("candidate_only_rules", 0)),
                "num_mixed_accepted_candidate_rules": int(category_counts.get("mixed_accepted_candidate_rules", 0)),
                "avg_candidate_body_atom_ratio": float(all_subset.get("avg_candidate_body_atom_ratio", 0.0)),
                "covered_eval_positive_examples": len(rule_eval_result.get("covered_positive_example_ids", [])),
                "num_fn_examples": len(rule_eval_result.get("false_negative_example_ids", [])),
                "fn_coverage_vs_original": float(len(covered_original_fn_ids) / max(1, len(original_fn_ids))),
                "precision": float(overall.get("precision", 0.0)),
                "recall": float(overall.get("recall", 0.0)),
                "f1": float(overall.get("f1", 0.0)),
                "accuracy": float(overall.get("accuracy", 0.0)),
                "accepted_only_precision": float(dict(ablation.get("baseline_metrics", {})).get("precision", 0.0)),
                "accepted_only_recall": float(dict(ablation.get("baseline_metrics", {})).get("recall", 0.0)),
                "accepted_only_f1": float(dict(ablation.get("baseline_metrics", {})).get("f1", 0.0)),
                "candidate_augmented_precision_delta": float(ablation.get("delta_precision", 0.0)),
                "candidate_augmented_recall_delta": float(ablation.get("delta_recall", 0.0)),
                "candidate_augmented_f1_delta": float(ablation.get("delta_f1", 0.0)),
                "candidate_fn_coverage_gain_count": int(ablation.get("fn_coverage_gain_count", 0)),
                "candidate_fp_contribution_count": int(ablation.get("fp_contribution_count", 0)),
            }
        )

    comparison_json_path = evaluation_output_root / "rule_set_comparison_summary.json"
    comparison_csv_path = evaluation_output_root / "rule_set_comparison_summary.csv"
    with comparison_json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "primary_rule_set": primary_rule_set,
                "rule_set_mode": rule_set_mode,
                "rows": comparison_rows,
            },
            fh,
            indent=2,
        )
    with comparison_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rule_set_name",
                "selection_method",
                "num_final_rules",
                "num_accepted_only_rules",
                "num_candidate_only_rules",
                "num_mixed_accepted_candidate_rules",
                "avg_candidate_body_atom_ratio",
                "covered_eval_positive_examples",
                "num_fn_examples",
                "fn_coverage_vs_original",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "accepted_only_precision",
                "accepted_only_recall",
                "accepted_only_f1",
                "candidate_augmented_precision_delta",
                "candidate_augmented_recall_delta",
                "candidate_augmented_f1_delta",
                "candidate_fn_coverage_gain_count",
                "candidate_fp_contribution_count",
            ],
        )
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)
    return {
        "comparison_json_path": str(comparison_json_path),
        "comparison_csv_path": str(comparison_csv_path),
    }


def _write_rule_set_error_comparison(
    error_analysis_output_root: Path,
    primary_rule_set: str,
    rule_set_mode: str,
    evaluation_rule_sets: Sequence[str],
    rule_results_by_name: Dict[str, Dict[str, Any]],
    evaluation_results_by_name: Dict[str, Dict[str, Any]],
    error_analysis_results_by_name: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    comparison_rows: List[Dict[str, Any]] = []
    original_fn_ids = set(evaluation_results_by_name.get("original", {}).get("false_negative_example_ids", []))
    for rule_set_name in evaluation_rule_sets:
        overall = dict(evaluation_results_by_name[rule_set_name].get("overall_metrics", {}))
        analysis_manifest = dict(error_analysis_results_by_name[rule_set_name])
        rule_eval_result = evaluation_results_by_name[rule_set_name]
        predicted_positive_ids = set(rule_eval_result.get("predicted_positive_example_ids", []))
        covered_original_fn_ids = original_fn_ids & predicted_positive_ids
        comparison_rows.append(
            {
                "rule_set_name": rule_set_name,
                "selection_method": str(rule_results_by_name[rule_set_name].get("selection_method", "score_top_k")),
                "num_rules": int(analysis_manifest.get("num_rules", 0)),
                "num_rule_families": int(analysis_manifest.get("num_rule_families", 0)),
                "max_rules_in_family": int(analysis_manifest.get("max_rules_in_family", 0)),
                "redundancy_ratio": float(analysis_manifest.get("redundancy_ratio", 0.0)),
                "covered_eval_positive_examples": len(rule_eval_result.get("covered_positive_example_ids", [])),
                "precision": float(overall.get("precision", 0.0)),
                "recall": float(overall.get("recall", 0.0)),
                "f1": float(overall.get("f1", 0.0)),
                "accuracy": float(overall.get("accuracy", 0.0)),
                "num_fn_examples": int(analysis_manifest.get("num_fn_examples", 0)),
                "fn_coverage_vs_original": float(len(covered_original_fn_ids) / max(1, len(original_fn_ids))),
                "num_fp_examples": int(analysis_manifest.get("num_fp_examples", 0)),
                "num_uncovered_positive_patterns": int(analysis_manifest.get("num_uncovered_positive_patterns", 0)),
            }
        )

    comparison_json_path = error_analysis_output_root / "rule_set_comparison_summary.json"
    comparison_csv_path = error_analysis_output_root / "rule_set_comparison_summary.csv"
    with comparison_json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "primary_rule_set": primary_rule_set,
                "rule_set_mode": rule_set_mode,
                "rows": comparison_rows,
            },
            fh,
            indent=2,
        )
    with comparison_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rule_set_name",
                "selection_method",
                "num_rules",
                "num_rule_families",
                "max_rules_in_family",
                "redundancy_ratio",
                "covered_eval_positive_examples",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "num_fn_examples",
                "fn_coverage_vs_original",
                "num_fp_examples",
                "num_uncovered_positive_patterns",
            ],
        )
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)
    return {
        "comparison_json_path": str(comparison_json_path),
        "comparison_csv_path": str(comparison_csv_path),
    }


def _prepare_rule_learning_inputs(ctx: PipelineContext) -> None:
    split_cfg = _get_data_split_cfg()
    temporal_rule_results = list(ctx.temporal_rule_results or [])
    total_videos = len({str(result.get("video_id", "")) for result in temporal_rule_results if str(result.get("video_id", ""))})
    strategy = str(split_cfg.get("strategy", "eval_fraction"))
    train_video_count = int(split_cfg.get("train_video_count", DEFAULT_TRAIN_VIDEO_COUNT))
    eval_video_count = int(split_cfg.get("eval_video_count", DEFAULT_EVAL_VIDEO_COUNT))
    eval_fraction = float(split_cfg.get("eval_fraction", 0.2))

    if strategy == "fixed_counts":
        requested_total = train_video_count + eval_video_count
        if total_videos >= 2 and total_videos != requested_total:
            base_eval_fraction = eval_video_count / max(1, requested_total)
            eval_video_count = max(1, min(total_videos - 1, int(math.ceil(total_videos * base_eval_fraction))))
            train_video_count = total_videos - eval_video_count

    ctx.split_manifest = _build_train_eval_split(
        video_ids=[str(result.get("video_id", "")) for result in temporal_rule_results],
        train_video_count=train_video_count,
        eval_video_count=eval_video_count,
        strategy=strategy,
        eval_fraction=eval_fraction,
    )
    ctx.train_temporal_rule_results = _select_video_results(
        temporal_rule_results,
        selected_video_ids=list(ctx.split_manifest.get("train_video_ids", [])),
    )
    ctx.eval_temporal_rule_results = _select_video_results(
        temporal_rule_results,
        selected_video_ids=list(ctx.split_manifest.get("eval_video_ids", [])),
    )
    if not ctx.train_temporal_rule_results:
        raise RuntimeError("Train split is empty; cannot continue with rule learning.")


def run_step_0_background_rule_relevance_prior(ctx: PipelineContext, runner: StepRunner) -> None:
    background_rule_relevance_prior_cfg = _get_background_rule_relevance_prior_cfg()
    runner.announce_step("0", "background_rule_relevance_prior_driving_mini", leading_newline=False)
    runner.log("0", f"cfg={background_rule_relevance_prior_cfg}")
    runner.log("0", f"recompute={bool(_get_pipeline_recompute_cfg().get('background_rule_relevance_prior', True))}")
    with runner.module_output("0"):
        ctx.background_rule_relevance_prior_results = background_rule_relevance_prior_driving_mini.run(
            cfg=background_rule_relevance_prior_cfg,
            output_root=_get_background_rule_relevance_prior_output_root(),
            force_recompute=bool(_get_pipeline_recompute_cfg().get("background_rule_relevance_prior", True)),
        )
    runner.log("0", f"entries={int(ctx.background_rule_relevance_prior_results.get('num_prior_entries', 0))}")
    runner.complete_step("0", subtitle=f"entries={int(ctx.background_rule_relevance_prior_results.get('num_prior_entries', 0))}")


def run_step_1_detection(ctx: PipelineContext, runner: StepRunner) -> None:
    render_video = _get_detection_render_video_enabled(default=True)
    check_cache = _get_detection_check_cache_enabled(default=False)
    active_od_policy = od_calibration_policy_utils.load_active_od_calibration_policy()
    runner.announce_step("1", "detect_driving_mini", leading_newline=False)
    runner.log("1", f"model={DRIVING_MINI_OD_MODEL}")
    runner.log("1", f"classes={len(DRIVING_MINI_OD_CLASSES)} configured")
    runner.log("1", f"render_video={render_video}")
    runner.log("1", f"check_cache={check_cache}")
    runner.log("1", f"step0_prior_entries={int(ctx.background_rule_relevance_prior_results.get('num_prior_entries', 0))}")
    runner.log("1", f"od_calibration_policy={od_calibration_policy_utils.policy_id(active_od_policy) or 'none'}")
    runner.log("1", f"force_recompute={_force_recompute(ctx)}")
    for warning in detect_driving_mini.detector_dependency_warnings(render_video=render_video):
        runner.log("1", f"[warn] {warning}")
    if ctx.effective_video_ids:
        runner.log("1", f"video_filter={ctx.effective_video_ids}")
    with runner.module_output("1"):
        ctx.detection_results = _run_object_detection_step(
            force_recompute=_force_recompute(ctx),
            video_ids=ctx.effective_video_ids,
            render_video=render_video,
            check_cache=check_cache,
            od_calibration_policy=active_od_policy,
            background_rule_relevance_prior_results=ctx.background_rule_relevance_prior_results,
        )
    runner.log("1", f"completed videos={len(ctx.detection_results)}")
    runner.complete_step("1", subtitle=f"videos={len(ctx.detection_results)}")


def run_step_2_tracking(ctx: PipelineContext, runner: StepRunner) -> None:
    render_video = _get_tracking_render_video_enabled(default=True)
    runner.announce_step("2", "tracking_driving_mini")
    runner.log("2", f"render_video={render_video}")
    runner.log("2", f"force_recompute={_force_recompute(ctx)}")
    if ctx.detection_results:
        runner.log(
            "2",
            "od_calibration_policy="
            f"{dict(ctx.detection_results[0].get('od_calibration', {})).get('policy_id', '') or 'none'}",
        )
    with runner.module_output("2"):
        ctx.tracking_results = tracking_driving_mini.run(
            ctx.detection_results or [],
            render_video=render_video,
            force_recompute=_force_recompute(ctx),
        )
    runner.log("2", f"completed videos={len(ctx.tracking_results)}")
    runner.log(
        "2",
        f"tracking_input_candidate_detections={sum(int(row.get('num_tracking_input_candidate_detections', 0)) for row in (ctx.tracking_results or []))}",
    )
    runner.log(
        "2",
        f"raw_candidate_tracks={sum(int(row.get('num_raw_candidate_tracks', 0)) for row in (ctx.tracking_results or []))}",
    )
    runner.log(
        "2",
        f"deduplicated_candidate_tracks={sum(int(row.get('num_deduplicated_candidate_tracks', 0)) for row in (ctx.tracking_results or []))}",
    )
    runner.log(
        "2",
        f"selected_candidate_tracks={sum(int(row.get('num_candidate_tracks', 0)) for row in (ctx.tracking_results or []))}",
    )
    runner.log(
        "2",
        f"rejected_candidate_tracks={sum(int(row.get('num_rejected_candidate_tracks', 0)) for row in (ctx.tracking_results or []))}",
    )
    runner.log(
        "2",
        f"rejected_candidate_detections={sum(int(row.get('num_rejected_candidate_detections', 0)) for row in (ctx.tracking_results or []))}",
    )
    runner.complete_step("2", subtitle=f"videos={len(ctx.tracking_results)}")


def run_step_3_dataset_annotations(ctx: PipelineContext, runner: StepRunner) -> None:
    runner.announce_step("3", "dataset_annotations_driving_mini")
    runner.log("3", f"force_recompute={_force_recompute(ctx)}")
    with runner.module_output("3"):
        ctx.dataset_annotation_results = dataset_annotations_driving_mini.run(
            video_ids=[r["video_id"] for r in (ctx.tracking_results or [])],
            tracking_results=ctx.tracking_results or [],
            force_recompute=_force_recompute(ctx),
        )
    runner.log("3", f"completed videos={len(ctx.dataset_annotation_results)}")
    runner.log(
        "3",
        f"candidate_objects={sum(int(row.get('num_candidate_objects', 0)) for row in (ctx.dataset_annotation_results or []))}",
    )
    runner.complete_step("3", subtitle=f"videos={len(ctx.dataset_annotation_results)}")


def run_step_4_merge_annotations(ctx: PipelineContext, runner: StepRunner) -> None:
    render_video = _get_merge_render_video_enabled(default=True)
    runner.announce_step("4", "merge_gt_and_detected_driving_mini")
    runner.log("4", f"render_video={render_video}")
    runner.log("4", f"force_recompute={_force_recompute(ctx)}")
    with runner.module_output("4"):
        ctx.merged_results = merge_gt_and_detected_driving_mini.run(
            tracking_results=ctx.tracking_results or [],
            dataset_annotation_results=ctx.dataset_annotation_results or [],
            render_video=render_video,
            force_recompute=_force_recompute(ctx),
        )
    runner.log("4", f"completed videos={len(ctx.merged_results)}")
    runner.log(
        "4",
        f"candidate_objects={sum(int(row.get('num_candidate_objects', 0)) for row in (ctx.merged_results or []))}",
    )
    runner.complete_step("4", subtitle=f"videos={len(ctx.merged_results)}")


def run_step_5_prepare_3d_positions(ctx: PipelineContext, runner: StepRunner) -> None:
    runner.announce_step("5", "prepare_3d_positions_driving_mini")
    runner.log("5", f"force_recompute={_force_recompute(ctx)}")
    with runner.module_output("5"):
        ctx.positions_3d_results = prepare_3d_positions_driving_mini.run(
            merged_results=ctx.merged_results or [],
            force_recompute=_force_recompute(ctx),
        )
    runner.log("5", f"completed videos={len(ctx.positions_3d_results)}")
    runner.log(
        "5",
        f"candidate_objects_with_3d={sum(int(row.get('num_candidate_objects_with_3d', 0)) for row in (ctx.positions_3d_results or []))}",
    )
    runner.complete_step("5", subtitle=f"videos={len(ctx.positions_3d_results)}")


def run_step_6_ego_motion(ctx: PipelineContext, runner: StepRunner) -> None:
    smoothing_window = _get_ego_motion_smoothing_window(default=5)
    static_adjust_cfg = _get_ego_static_adjustment_cfg()
    render_video = _get_ego_motion_render_video_enabled(default=True)
    runner.announce_step("6", "ego_motion_driving_mini")
    runner.log("6", f"smoothing_window={smoothing_window}")
    runner.log("6", f"static_adjustment={static_adjust_cfg}")
    runner.log("6", f"render_video={render_video}")
    runner.log("6", f"force_recompute={_force_recompute(ctx)}")
    with runner.module_output("6"):
        ctx.ego_motion_results = ego_motion_driving_mini.run(
            merged_results=ctx.merged_results or [],
            force_recompute=_force_recompute(ctx),
            smoothing_window=smoothing_window,
            static_adjust_cfg=static_adjust_cfg,
            render_video=render_video,
        )
    runner.log("6", f"completed videos={len(ctx.ego_motion_results)}")
    runner.complete_step("6", subtitle=f"videos={len(ctx.ego_motion_results)}")


def run_step_7_relative_object_motion(ctx: PipelineContext, runner: StepRunner) -> None:
    runner.announce_step("7", "relative_object_motion_driving_mini")
    runner.log("7", f"force_recompute={_force_recompute(ctx)}")
    with runner.module_output("7"):
        ctx.relative_motion_results = relative_object_motion_driving_mini.run(
            positions_3d_results=ctx.positions_3d_results or [],
            ego_motion_results=ctx.ego_motion_results or [],
            force_recompute=_force_recompute(ctx),
        )
    runner.log("7", f"completed videos={len(ctx.relative_motion_results)}")
    runner.log(
        "7",
        f"candidate_objects_with_motion={sum(int(row.get('num_candidate_objects_with_rel_motion', 0)) for row in (ctx.relative_motion_results or []))}",
    )
    runner.complete_step("7", subtitle=f"videos={len(ctx.relative_motion_results)}")


def run_step_8_temporal_segmentation(ctx: PipelineContext, runner: StepRunner) -> None:
    temporal_seg_cfg = _get_temporal_segmentation_cfg()
    runner.announce_step("8", "temporal_segmentation_driving_mini")
    runner.log("8", f"cfg={temporal_seg_cfg}")
    runner.log("8", f"force_recompute={_force_recompute(ctx)}")
    with runner.module_output("8"):
        ctx.temporal_seg_results = temporal_segmentation_driving_mini.run(
            ego_motion_results=ctx.ego_motion_results or [],
            relative_motion_results=ctx.relative_motion_results or [],
            seg_cfg=temporal_seg_cfg,
            force_recompute=_force_recompute(ctx),
        )
    runner.log("8", f"completed videos={len(ctx.temporal_seg_results)}")
    runner.complete_step("8", subtitle=f"videos={len(ctx.temporal_seg_results)}")


def run_step_9_segment_object_motion(ctx: PipelineContext, runner: StepRunner) -> None:
    segment_object_cfg = _get_segment_object_motion_cfg()
    runner.announce_step("9", "segment_object_motion_driving_mini")
    runner.log("9", f"cfg={segment_object_cfg}")
    runner.log("9", f"force_recompute={_force_recompute(ctx)}")
    with runner.module_output("9"):
        ctx.segment_object_results = segment_object_motion_driving_mini.run(
            relative_motion_results=ctx.relative_motion_results or [],
            temporal_segmentation_results=ctx.temporal_seg_results or [],
            cfg=segment_object_cfg,
            force_recompute=_force_recompute(ctx),
        )
    runner.log("9", f"completed videos={len(ctx.segment_object_results)}")
    runner.log(
        "9",
        f"candidate_segment_objects={sum(int(row.get('num_candidate_objects_total', 0)) for row in (ctx.segment_object_results or []))}",
    )
    runner.complete_step("9", subtitle=f"videos={len(ctx.segment_object_results)}")


def run_step_10_important_objects(ctx: PipelineContext, runner: StepRunner) -> None:
    important_objects_cfg = _get_important_objects_cfg()
    runner.announce_step("10", "important_objects_driving_mini")
    runner.log("10", f"cfg={important_objects_cfg}")
    runner.log("10", f"force_recompute={_force_recompute(ctx)}")
    with runner.module_output("10"):
        ctx.important_object_results = important_objects_driving_mini.run(
            segment_object_motion_results=ctx.segment_object_results or [],
            cfg=important_objects_cfg,
            force_recompute=_force_recompute(ctx),
        )
    runner.log("10", f"completed videos={len(ctx.important_object_results)}")
    runner.complete_step("10", subtitle=f"videos={len(ctx.important_object_results)}")


def run_step_10b_traffic_control_attributes(ctx: PipelineContext, runner: StepRunner) -> None:
    traffic_control_attributes_cfg = _get_traffic_control_attributes_cfg()
    runner.announce_step("10B", "traffic_control_attributes_driving_mini")
    runner.log("10B", f"cfg={traffic_control_attributes_cfg}")
    runner.log("10B", f"recompute={_force_recompute(ctx, 'traffic_control_attributes', False)}")
    with runner.module_output("10B"):
        ctx.traffic_control_attribute_results = traffic_control_attributes_driving_mini.run(
            important_object_results=ctx.important_object_results or [],
            relative_motion_results=ctx.relative_motion_results or [],
            cfg=traffic_control_attributes_cfg,
            force_recompute=_force_recompute(ctx, "traffic_control_attributes", False),
        )
    runner.log("10B", f"completed videos={len(ctx.traffic_control_attribute_results)}")
    runner.complete_step("10B", subtitle=f"videos={len(ctx.traffic_control_attribute_results)}")


def run_step_11_logic_atoms(ctx: PipelineContext, runner: StepRunner) -> None:
    logic_atoms_cfg = _get_logic_atoms_cfg()
    runner.announce_step("11", "logic_atoms_driving_mini")
    runner.log("11", f"cfg={logic_atoms_cfg}")
    runner.log("11", f"recompute={_force_recompute(ctx, 'logic_atoms', False)}")
    with runner.module_output("11"):
        ctx.logic_atom_results = logic_atoms_driving_mini.run(
            segment_object_motion_results=ctx.traffic_control_attribute_results or ctx.important_object_results or [],
            cfg=logic_atoms_cfg,
            force_recompute=_force_recompute(ctx, "logic_atoms", False),
        )
    runner.log("11", f"completed videos={len(ctx.logic_atom_results)}")
    runner.complete_step("11", subtitle=f"videos={len(ctx.logic_atom_results)}")


def run_step_12_target_head_atoms(ctx: PipelineContext, runner: StepRunner) -> None:
    target_head_cfg = _get_target_head_atoms_cfg()
    runner.announce_step("12", "target_head_atoms_driving_mini")
    runner.log("12", f"cfg={target_head_cfg}")
    runner.log("12", f"recompute={_force_recompute(ctx, 'target_head_atoms', False)}")
    with runner.module_output("12"):
        ctx.target_head_results = target_head_atoms_driving_mini.run(
            logic_atom_results=ctx.logic_atom_results or [],
            cfg=target_head_cfg,
            force_recompute=_force_recompute(ctx, "target_head_atoms", False),
        )
    runner.log("12", f"completed videos={len(ctx.target_head_results)}")
    runner.complete_step("12", subtitle=f"videos={len(ctx.target_head_results)}")


def run_step_13_temporal_rule_examples(ctx: PipelineContext, runner: StepRunner) -> None:
    rule_examples_cfg = _get_temporal_rule_examples_cfg()
    runner.announce_step("13", "temporal_rule_examples_driving_mini")
    runner.log("13", f"cfg={rule_examples_cfg}")
    runner.log("13", f"recompute={_force_recompute(ctx, 'temporal_rule_examples', False)}")
    with runner.module_output("13"):
        ctx.temporal_rule_results = temporal_rule_examples_driving_mini.run(
            target_head_results=ctx.target_head_results or [],
            cfg=rule_examples_cfg,
            force_recompute=_force_recompute(ctx, "temporal_rule_examples", False),
        )
    runner.log("13", f"completed videos={len(ctx.temporal_rule_results)}")
    runner.complete_step("13", subtitle=f"videos={len(ctx.temporal_rule_results)}")


def run_step_14_candidate_rules(ctx: PipelineContext, runner: StepRunner) -> None:
    candidate_rules_cfg = _get_candidate_rules_cfg()
    runner.announce_step("14", "candidate_rules_driving_mini")
    runner.log("14", f"cfg={candidate_rules_cfg}")
    runner.log("14", f"recompute={bool(ctx.recompute_cfg.get('candidate_rules', False))}")
    with runner.module_output("14"):
        ctx.candidate_rule_results = candidate_rules_driving_mini.run(
            temporal_rule_results=ctx.train_temporal_rule_results or [],
            cfg=candidate_rules_cfg,
            force_recompute=bool(ctx.recompute_cfg.get("candidate_rules", False)),
        )
    runner.log("14", f"completed videos={len(ctx.candidate_rule_results)}")
    runner.complete_step("14", subtitle=f"videos={len(ctx.candidate_rule_results)}")


def run_step_15_merge_initial_rules(ctx: PipelineContext, runner: StepRunner) -> None:
    runner.announce_step("15", "merge_initial_rules")
    ctx.merged_candidate_rules = _merge_candidate_rules(ctx.candidate_rule_results or [])
    ctx.initial_rules_for_extension = ctx.merged_candidate_rules
    runner.log(
        "15",
        "merged "
        f"rules={ctx.merged_candidate_rules['num_rules']} "
        f"videos={ctx.merged_candidate_rules['num_videos']}",
    )
    runner.complete_step("15", subtitle=f"rules={ctx.merged_candidate_rules['num_rules']}")


def run_step_15b_initial_rule_pruning(ctx: PipelineContext, runner: StepRunner) -> None:
    pruning_cfg = _get_initial_rule_pruning_cfg()
    runner.announce_step("15B", "initial_rule_pruning")
    runner.log("15B", f"cfg={pruning_cfg}")
    if not bool(pruning_cfg.get("enabled", True)):
        ctx.initial_rules_for_extension = ctx.merged_candidate_rules
        runner.log("15B", "disabled; Step 16 will use merged initial rules")
        runner.complete_step("15B", subtitle="disabled")
        return

    ctx.initial_rules_for_extension = _prune_initial_rules(
        ctx.merged_candidate_rules,
        cfg=pruning_cfg,
    )
    summary = dict(
        dict(ctx.initial_rules_for_extension.get("candidate_rule_stage_stats", {})).get(
            "initial_rule_pruning", {}
        )
    )
    runner.log(
        "15B",
        "pruned "
        f"input={int(summary.get('input_num_rules', 0))} "
        f"kept={int(summary.get('kept_num_rules', ctx.initial_rules_for_extension.get('num_rules', 0)))}",
    )
    runner.complete_step(
        "15B",
        subtitle=f"kept={int(ctx.initial_rules_for_extension.get('num_rules', 0))}",
    )


def run_step_16_extended_rules(ctx: PipelineContext, runner: StepRunner) -> None:
    extended_rules_cfg = _get_extended_rules_cfg()
    initial_rules_payload = ctx.initial_rules_for_extension or ctx.merged_candidate_rules
    runner.announce_step("16", "extended_rules_driving_mini")
    runner.log("16", f"cfg={extended_rules_cfg}")
    runner.log("16", f"input_initial_rules={int(initial_rules_payload.get('num_rules', 0))}")
    runner.log("16", f"recompute={bool(ctx.recompute_cfg.get('extended_rules', True))}")
    with runner.module_output("16"):
        ctx.extended_rule_results = extended_rules_driving_mini.run(
            merged_initial_rules=initial_rules_payload,
            cfg=extended_rules_cfg,
            force_recompute=bool(ctx.recompute_cfg.get("extended_rules", True)),
        )
    runner.log("16", f"rounds={ctx.extended_rule_results.get('num_rounds_completed', 0)}")
    runner.complete_step(
        "16",
        subtitle=f"kept={int(ctx.extended_rule_results.get('num_kept_rules', 0))}",
    )


def run_step_17_final_rules(ctx: PipelineContext, runner: StepRunner) -> None:
    final_rules_cfg = _get_final_rules_cfg()
    selector_mode, selector_cfg, output_root, selector_module = _selected_selector_cfg(final_rules_cfg)
    final_rules_output_root = final_rules_driving_mini.get_output_root()
    runner.announce_step("17", "final_rules_driving_mini")
    runner.log("17", f"selector_mode={selector_mode} cfg={selector_cfg}")
    runner.log("17", f"recompute={bool(ctx.recompute_cfg.get('final_rules', True))}")
    with runner.module_output("17"):
        if output_root is None:
            ctx.final_rule_results = selector_module.run(
                extended_rule_results=ctx.extended_rule_results,
                cfg=selector_cfg,
                force_recompute=bool(ctx.recompute_cfg.get("final_rules", True)),
            )
        else:
            ctx.final_rule_results = selector_module.run(
                extended_rule_results=ctx.extended_rule_results,
                cfg=selector_cfg,
                output_root=output_root,
                force_recompute=bool(ctx.recompute_cfg.get("final_rules", True)),
            )
    ctx.rule_results_by_name = {"selected": ctx.final_rule_results}
    ctx.primary_rule_set = "selected"
    ctx.rule_set_mode = "selected"
    ctx.evaluation_rule_sets = ["selected"]
    summary_path, summary_row = _write_step17_primary_summary(
        ctx.final_rule_results,
        output_root=final_rules_output_root,
        rule_set_name="selected",
    )
    manifest_path = final_rules_output_root / "17_final_rule_selection_manifest.json"
    manifest = {
        "step": "17",
        "primary_summary_csv": str(summary_path),
        "main_conclusion": str(summary_row.get("main_conclusion", "")),
        "secondary_debug_artifacts": [
            str(final_rules_output_root / "final_rules.json"),
            str(final_rules_output_root / "final_rules.csv"),
        ],
    }
    _write_manifest_json(manifest_path, manifest)
    ctx.final_rule_results["primary_summary_csv"] = str(summary_path)
    ctx.final_rule_results["manifest_json"] = str(manifest_path)
    ctx.final_rule_results["main_conclusion"] = str(summary_row.get("main_conclusion", ""))
    ctx.final_rule_results["secondary_debug_artifacts"] = list(manifest["secondary_debug_artifacts"])
    runner.log(
        "17",
        "selected "
        f"rules={ctx.final_rule_results.get('num_final_rules', 0)} "
        f"method={ctx.final_rule_results.get('selection_method', 'n/a')}",
    )
    runner.log("17", f"summary_csv={summary_path}")
    runner.log("17", str(summary_row.get("main_conclusion", "")))
    runner.complete_step("17", subtitle=f"rules={int(ctx.final_rule_results.get('num_final_rules', 0))}")


def run_step_17d_rule_pool_and_selector_diagnostic(ctx: PipelineContext, runner: StepRunner) -> None:
    diagnostic_cfg = _get_rule_pool_and_selector_diagnostic_cfg()
    diagnostic_output_root = _get_rule_pool_upper_bound_diagnostic_output_root()
    runner.announce_step("17D", "rule_pool_and_selector_diagnostic_driving_mini")
    if not bool(diagnostic_cfg.get("enabled", False)):
        runner.log("17D", "disabled by config")
        runner.complete_step("17D", subtitle="skipped")
        return
    runner.log("17D", f"cfg={diagnostic_cfg}")
    runner.log("17D", f"recompute={bool(ctx.recompute_cfg.get('rule_pool_and_selector_diagnostic', True))}")
    with runner.module_output("17D"):
        ctx.rule_pool_upper_bound_results = rule_pool_upper_bound_diagnostic_driving_mini.run(
            extended_rule_results=ctx.extended_rule_results,
            temporal_rule_results=ctx.eval_temporal_rule_results or [],
            eval_video_ids=list(ctx.split_manifest.get("eval_video_ids", [])),
            split_manifest=ctx.split_manifest,
            rule_results_by_name=ctx.rule_results_by_name,
            cfg=diagnostic_cfg,
            output_root=_get_rule_pool_upper_bound_diagnostic_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("rule_pool_and_selector_diagnostic", True)),
        )
        ctx.oracle_rule_selection_gap_results = oracle_rule_selection_gap_diagnostic_driving_mini.run(
            extended_rule_results=ctx.extended_rule_results,
            rule_pool_upper_bound_results=ctx.rule_pool_upper_bound_results,
            rule_results_by_name=ctx.rule_results_by_name,
            cfg=diagnostic_cfg,
            output_root=_get_oracle_rule_selection_gap_diagnostic_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("rule_pool_and_selector_diagnostic", True)),
        )
    summary_path, summary_rows = _write_step17d_primary_summary(
        ctx.rule_pool_upper_bound_results,
        output_root=diagnostic_output_root,
    )
    manifest_path = diagnostic_output_root / "17d_rule_pool_and_selector_diagnostic_manifest.json"
    secondary_debug_artifacts = []
    secondary_debug_artifacts.extend(
        str(path)
        for path in dict(ctx.rule_pool_upper_bound_results.get("output_paths", {})).values()
        if str(path)
    )
    secondary_debug_artifacts.append(str(diagnostic_output_root / "pool_upper_bound_summary.json"))
    secondary_debug_artifacts.extend(
        str(path)
        for path in dict(ctx.oracle_rule_selection_gap_results.get("output_paths", {})).values()
        if str(path)
    )
    secondary_debug_artifacts.append(
        str(_get_oracle_rule_selection_gap_diagnostic_output_root() / "oracle_selection_gap_summary.json")
    )
    manifest = {
        "step": "17D",
        "primary_summary_csv": str(summary_path),
        "main_conclusion": str(summary_rows[0].get("main_conclusion", "")) if summary_rows else "",
        "secondary_debug_artifacts": secondary_debug_artifacts,
    }
    _write_manifest_json(manifest_path, manifest)
    ctx.rule_pool_upper_bound_results["primary_summary_csv"] = str(summary_path)
    ctx.rule_pool_upper_bound_results["manifest_json"] = str(manifest_path)
    ctx.rule_pool_upper_bound_results["secondary_debug_artifacts"] = secondary_debug_artifacts
    runner.log(
        "17D",
        "bottleneck="
        f"{ctx.rule_pool_upper_bound_results.get('bottleneck_label', 'unknown')} "
        f"oracle_f1={float(ctx.oracle_rule_selection_gap_results.get('oracle_target_f1', 0.0)):.3f}",
    )
    if summary_rows:
        runner.log("17D", f"summary_csv={summary_path}")
        runner.log("17D", str(summary_rows[0].get("main_conclusion", "")))
    runner.complete_step("17D", subtitle="enabled")


def run_step_18_rule_evaluation(ctx: PipelineContext, runner: StepRunner) -> None:
    rule_evaluation_cfg = _get_rule_evaluation_cfg()
    evaluation_output_root = _get_rule_evaluation_output_root()
    runner.announce_step("18", "evaluate_rules_driving_mini")
    runner.log("18", f"cfg={rule_evaluation_cfg}")
    ctx.rule_set_mode = "selected"
    ctx.primary_rule_set = "selected"
    ctx.evaluation_rule_sets = ["selected"]
    evaluation_cfg_for_run = dict(rule_evaluation_cfg)
    evaluation_cfg_for_run["evaluated_rule_set_name"] = "selected"
    with runner.module_output("18"):
        evaluation_result = evaluate_rules_driving_mini.run(
            final_rule_results=ctx.final_rule_results,
            temporal_rule_results=ctx.eval_temporal_rule_results or [],
            eval_video_ids=list(ctx.split_manifest.get("eval_video_ids", [])),
            split_manifest=ctx.split_manifest,
            cfg=evaluation_cfg_for_run,
            output_root=evaluation_output_root,
            force_recompute=bool(ctx.recompute_cfg.get("rule_evaluation", True)),
        )
    evaluation_result["evaluated_rule_set_name"] = "selected"
    ctx.evaluation_results_by_name = {"selected": evaluation_result}
    ctx.evaluation_results = evaluation_result
    ctx.candidate_contribution_summary_results = _synthesize_selected_candidate_contribution_summary(
        ctx.final_rule_results,
        evaluation_result,
    )
    summary_path, summary_rows = _write_step18_primary_summary(
        ctx.evaluation_results,
        output_root=evaluation_output_root,
    )
    manifest_path = evaluation_output_root / "18_rule_evaluation_manifest.json"
    secondary_debug_artifacts = [
        str(evaluation_output_root / "rule_evaluation.json"),
        str(evaluation_output_root / "rule_evaluation.csv"),
        str(evaluation_output_root / "example_predictions.csv"),
        str(evaluation_output_root / "rule_subset_metrics.csv"),
        str(evaluation_output_root / "candidate_rule_flow_summary.json"),
        str(evaluation_output_root / "rule_evaluation_summary.pdf"),
    ]
    manifest = {
        "step": "18",
        "primary_summary_csv": str(summary_path),
        "main_conclusion": str(summary_rows[0].get("main_conclusion", "")) if summary_rows else "",
        "secondary_debug_artifacts": secondary_debug_artifacts,
    }
    _write_manifest_json(manifest_path, manifest)
    ctx.evaluation_results["primary_summary_csv"] = str(summary_path)
    ctx.evaluation_results["manifest_json"] = str(manifest_path)
    ctx.evaluation_results["secondary_debug_artifacts"] = secondary_debug_artifacts
    overall_metrics = dict(ctx.evaluation_results.get("overall_metrics", {}))
    subset_metrics = dict(ctx.evaluation_results.get("rule_subset_metrics", {}))
    candidate_rule_ablation = dict(ctx.evaluation_results.get("candidate_rule_ablation", {}))
    for subset_name in (
        "accepted_only_rules",
        "accepted_plus_mixed_rules",
        "accepted_plus_all_candidate_rules",
        "all_rules",
    ):
        subset_overall = dict(dict(subset_metrics.get(subset_name, {})).get("overall_metrics", {}))
        runner.log(
            "18",
            f"{subset_name} "
            f"p={float(subset_overall.get('precision', 0.0)):.3f} "
            f"r={float(subset_overall.get('recall', 0.0)):.3f} "
            f"f1={float(subset_overall.get('f1', 0.0)):.3f}",
        )
    runner.log(
        "18",
        f"rule_set=selected "
        f"precision={float(overall_metrics.get('precision', 0.0)):.3f} "
        f"recall={float(overall_metrics.get('recall', 0.0)):.3f} "
        f"f1={float(overall_metrics.get('f1', 0.0)):.3f} "
        f"delta_f1_vs_accepted_only={float(candidate_rule_ablation.get('delta_f1', 0.0)):.3f}",
    )
    if summary_rows:
        runner.log("18", f"summary_csv={summary_path}")
        runner.log("18", str(summary_rows[0].get("main_conclusion", "")))
    runner.complete_step("18", subtitle=f"f1={float(overall_metrics.get('f1', 0.0)):.3f}")


def _ensure_od_calibration_iteration_id(ctx: PipelineContext) -> str:
    if not ctx.od_calibration_iteration_id:
        ctx.od_calibration_iteration_id = od_calibration_policy_utils.next_iteration_id()
    return ctx.od_calibration_iteration_id


def run_step_19_reasoning_supervised_od_calibration(ctx: PipelineContext, runner: StepRunner) -> None:
    calibration_cfg = _get_reasoning_supervised_od_calibration_cfg()
    iteration_id = _ensure_od_calibration_iteration_id(ctx)
    calibration_output_root = _get_od_confidence_calibration_loop_output_root()
    current_f1 = float(dict(ctx.evaluation_results.get("overall_metrics", {})).get("f1", 0.0))
    runner.announce_step("19", "reasoning_supervised_od_calibration_driving_mini")
    if not bool(calibration_cfg.get("enabled", False)):
        ctx.baseline_safe_calibration_gate_results = {
            "iteration_id": iteration_id,
            "decision": "reject",
            "decision_reason": "reasoning_supervised_od_calibration_disabled",
            "active_policy_before_id": "",
            "active_policy_after_id": "",
            "proposed_policy_id": "",
            "current_metrics": {"final_f1": current_f1},
            "reference_metrics": {"final_f1": current_f1},
        }
        summary_row = _step19_summary_row(ctx)
        summary_path = calibration_output_root / "19_od_calibration_summary.csv"
        fieldnames = list(summary_row.keys())
        _upsert_csv_row(summary_path, fieldnames, summary_row, "iteration_id")
        manifest_path = calibration_output_root / "19_reasoning_supervised_od_calibration_manifest.json"
        manifest = {
            "step": "19",
            "primary_summary_csv": str(summary_path),
            "main_conclusion": str(summary_row.get("main_conclusion", "")),
            "secondary_debug_artifacts": [],
        }
        _write_manifest_json(manifest_path, manifest)
        ctx.baseline_safe_calibration_gate_results["primary_summary_csv"] = str(summary_path)
        runner.log("19", "disabled by config")
        runner.log("19", f"summary_csv={summary_path}")
        runner.log("19", str(summary_row.get("main_conclusion", "")))
        runner.complete_step("19", subtitle="skipped", allow_stop=not ctx.defer_stop_after_gate)
        return
    useful_signal, reason = _candidate_signal_is_useful(ctx.evaluation_results, calibration_cfg)
    if not useful_signal:
        ctx.baseline_safe_calibration_gate_results = {
            "iteration_id": iteration_id,
            "decision": "reject",
            "decision_reason": "candidate_signal_not_useful",
            "active_policy_before_id": "",
            "active_policy_after_id": "",
            "proposed_policy_id": "",
            "current_metrics": {"final_f1": current_f1},
            "reference_metrics": {"final_f1": current_f1},
        }
        summary_row = _step19_summary_row(ctx)
        summary_path = calibration_output_root / "19_od_calibration_summary.csv"
        fieldnames = list(summary_row.keys())
        _upsert_csv_row(summary_path, fieldnames, summary_row, "iteration_id")
        manifest_path = calibration_output_root / "19_reasoning_supervised_od_calibration_manifest.json"
        manifest = {
            "step": "19",
            "primary_summary_csv": str(summary_path),
            "main_conclusion": str(summary_row.get("main_conclusion", "")),
            "secondary_debug_artifacts": [],
        }
        _write_manifest_json(manifest_path, manifest)
        ctx.baseline_safe_calibration_gate_results["primary_summary_csv"] = str(summary_path)
        runner.log("19", f"skipped: candidate signal not useful ({reason})")
        runner.log("19", f"summary_csv={summary_path}")
        runner.log("19", str(summary_row.get("main_conclusion", "")))
        runner.complete_step("19", subtitle="skipped", allow_stop=not ctx.defer_stop_after_gate)
        return

    reasoning_to_od_pseudo_labels_cfg = _get_reasoning_to_od_pseudo_labels_cfg()
    runner.log("19", f"gate={reason} iteration_id={iteration_id}")
    runner.log("19", f"pseudo_label_cfg={reasoning_to_od_pseudo_labels_cfg}")
    with runner.module_output("19"):
        ctx.reasoning_to_od_pseudo_label_results = reasoning_to_od_pseudo_labels_driving_mini.run(
            detection_results=ctx.detection_results or [],
            tracking_results=ctx.tracking_results or [],
            logic_atom_results=ctx.logic_atom_results or [],
            eval_temporal_rule_results=ctx.eval_temporal_rule_results or [],
            evaluation_results_by_name=ctx.evaluation_results_by_name,
            candidate_contribution_summary_results=ctx.candidate_contribution_summary_results,
            primary_rule_set=ctx.primary_rule_set,
            iteration_id=iteration_id,
            cfg=reasoning_to_od_pseudo_labels_cfg,
            output_root=_get_od_confidence_calibration_loop_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("reasoning_to_od_pseudo_labels", True)),
        )
    od_confidence_calibration_cfg = _get_od_confidence_calibration_cfg()
    runner.log("19", f"calibration_cfg={od_confidence_calibration_cfg}")
    with runner.module_output("19"):
        ctx.od_confidence_calibration_results = od_confidence_calibration_driving_mini.run(
            pseudo_label_results=ctx.reasoning_to_od_pseudo_label_results,
            detection_results=ctx.detection_results or [],
            tracking_results=ctx.tracking_results or [],
            iteration_id=iteration_id,
            cfg=od_confidence_calibration_cfg,
            output_root=_get_od_confidence_calibration_loop_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("od_confidence_calibration", True)),
        )
    baseline_safe_calibration_gate_cfg = _get_baseline_safe_calibration_gate_cfg()
    runner.log("19", f"gate_cfg={baseline_safe_calibration_gate_cfg}")
    with runner.module_output("19"):
        ctx.baseline_safe_calibration_gate_results = baseline_safe_calibration_gate_driving_mini.run(
            evaluation_results_by_name=ctx.evaluation_results_by_name,
            primary_rule_set=ctx.primary_rule_set,
            rule_aggregation_baseline_results=ctx.rule_aggregation_baseline_results,
            calibration_results=ctx.od_confidence_calibration_results,
            iteration_id=iteration_id,
            cfg=baseline_safe_calibration_gate_cfg,
            output_root=_get_od_confidence_calibration_loop_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("baseline_safe_calibration_gate", True)),
        )
    runner.log(
        "19",
        f"decision={ctx.baseline_safe_calibration_gate_results.get('decision', 'n/a')} "
        f"policy={str(dict(ctx.od_confidence_calibration_results.get('policy', {})).get('policy_type', '')) or 'n/a'}",
    )
    summary_row = _step19_summary_row(ctx)
    summary_path = calibration_output_root / "19_od_calibration_summary.csv"
    fieldnames = list(summary_row.keys())
    _upsert_csv_row(summary_path, fieldnames, summary_row, "iteration_id")
    iteration_root = calibration_output_root / iteration_id
    secondary_debug_artifacts = [
        str(path)
        for path in dict(ctx.reasoning_to_od_pseudo_label_results.get("output_paths", {})).values()
        if str(path)
    ]
    secondary_debug_artifacts.extend(
        str(path)
        for path in dict(ctx.od_confidence_calibration_results.get("output_paths", {})).values()
        if str(path)
    )
    secondary_debug_artifacts.extend(
        [
            str(ctx.baseline_safe_calibration_gate_results.get("audit_path", "")),
            str(iteration_root / "calibration_gate_decision.json"),
        ]
    )
    secondary_debug_artifacts = [path for path in secondary_debug_artifacts if path]
    manifest_path = calibration_output_root / "19_reasoning_supervised_od_calibration_manifest.json"
    manifest = {
        "step": "19",
        "primary_summary_csv": str(summary_path),
        "main_conclusion": str(summary_row.get("main_conclusion", "")),
        "secondary_debug_artifacts": secondary_debug_artifacts,
    }
    _write_manifest_json(manifest_path, manifest)
    ctx.baseline_safe_calibration_gate_results["primary_summary_csv"] = str(summary_path)
    ctx.baseline_safe_calibration_gate_results["manifest_json"] = str(manifest_path)
    ctx.baseline_safe_calibration_gate_results["secondary_debug_artifacts"] = secondary_debug_artifacts
    runner.log("19", f"summary_csv={summary_path}")
    runner.log("19", str(summary_row.get("main_conclusion", "")))
    runner.complete_step(
        "19",
        subtitle=f"decision={ctx.baseline_safe_calibration_gate_results.get('decision', 'n/a')}",
        allow_stop=not ctx.defer_stop_after_gate,
    )


def run_step_18b_baseline_comparison(ctx: PipelineContext, runner: StepRunner) -> None:
    baseline_cfg = _get_baseline_comparison_cfg()
    runner.announce_step("18B", "baseline_comparison_driving_mini")
    if not bool(baseline_cfg.get("enabled", False)):
        runner.log("18B", "disabled by config")
        runner.complete_step("18B", subtitle="skipped")
        return

    ran_any = False
    if bool(baseline_cfg.get("run_neural_symbolic", True)):
        neural_symbolic_baseline_cfg = _get_neural_symbolic_baseline_cfg()
        runner.log("18B", f"neural_symbolic_cfg={neural_symbolic_baseline_cfg}")
        with runner.module_output("18B"):
            ctx.neural_symbolic_baseline_results = neural_symbolic_baseline_driving_mini.run(
                train_temporal_rule_results=ctx.train_temporal_rule_results or [],
                eval_temporal_rule_results=ctx.eval_temporal_rule_results or [],
                split_manifest=ctx.split_manifest,
                cfg=neural_symbolic_baseline_cfg,
                output_root=_get_neural_symbolic_baseline_output_root(),
                force_recompute=bool(ctx.recompute_cfg.get("neural_symbolic_baseline", True)),
            )
        ran_any = True

    if bool(baseline_cfg.get("run_rule_aggregation", True)):
        rule_aggregation_baseline_cfg = _get_rule_aggregation_baseline_cfg()
        runner.log("18B", f"rule_aggregation_cfg={rule_aggregation_baseline_cfg}")
        with runner.module_output("18B"):
            ctx.rule_aggregation_baseline_results = rule_aggregation_baseline_driving_mini.run(
                extended_rule_results=ctx.extended_rule_results,
                train_temporal_rule_results=ctx.train_temporal_rule_results or [],
                eval_temporal_rule_results=ctx.eval_temporal_rule_results or [],
                split_manifest=ctx.split_manifest,
                cfg=rule_aggregation_baseline_cfg,
                output_root=_get_rule_aggregation_baseline_output_root(),
                force_recompute=bool(ctx.recompute_cfg.get("rule_aggregation_baseline", True)),
            )
        ran_any = True

    if ctx.neural_symbolic_baseline_results:
        comparison_rows = list(ctx.neural_symbolic_baseline_results.get("comparison", []))
        if comparison_rows:
            runner.log("18B", f"neural_models={len(comparison_rows)}")
    if ctx.rule_aggregation_baseline_results:
        eval_best = dict(
            ctx.rule_aggregation_baseline_results.get("metrics_by_split", {})
            .get("eval", {})
            .get("best_validation_threshold", {})
        )
        runner.log(
            "18B",
            "rule_aggregation "
            f"f1={float(eval_best.get('f1', 0.0)):.3f} "
            f"nonzero_rules={int(ctx.rule_aggregation_baseline_results.get('num_nonzero_rules', 0))}",
        )
    runner.complete_step("18B", subtitle="enabled" if ran_any else "skipped")


def run_step_18c_rule_aggregation_baseline(ctx: PipelineContext, runner: StepRunner) -> None:
    rule_aggregation_baseline_cfg = _get_rule_aggregation_baseline_cfg()
    runner.announce_step("18C", "rule_aggregation_baseline_driving_mini")
    if not bool(rule_aggregation_baseline_cfg.get("enabled", False)):
        runner.log("18C", "disabled by config")
        runner.complete_step("18C", subtitle="skipped")
        return
    runner.log("18C", f"cfg={rule_aggregation_baseline_cfg}")
    runner.log("18C", f"recompute={bool(ctx.recompute_cfg.get('rule_aggregation_baseline', True))}")
    with runner.module_output("18C"):
        ctx.rule_aggregation_baseline_results = rule_aggregation_baseline_driving_mini.run(
            extended_rule_results=ctx.extended_rule_results,
            train_temporal_rule_results=ctx.train_temporal_rule_results or [],
            eval_temporal_rule_results=ctx.eval_temporal_rule_results or [],
            split_manifest=ctx.split_manifest,
            cfg=rule_aggregation_baseline_cfg,
            output_root=_get_rule_aggregation_baseline_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("rule_aggregation_baseline", True)),
        )
    rule_aggregation_eval_best = dict(
        ctx.rule_aggregation_baseline_results.get("metrics_by_split", {})
        .get("eval", {})
        .get("best_validation_threshold", {})
    )
    runner.log(
        "18C",
        f"f1={float(rule_aggregation_eval_best.get('f1', 0.0)):.3f} "
        f"auroc={float(rule_aggregation_eval_best.get('auroc', 0.0)):.3f} "
        f"auprc={float(rule_aggregation_eval_best.get('auprc', 0.0)):.3f} "
        f"nonzero_rules={int(ctx.rule_aggregation_baseline_results.get('num_nonzero_rules', 0))}",
    )
    runner.complete_step(
        "18C",
        subtitle=(
            f"f1={float(rule_aggregation_eval_best.get('f1', 0.0)):.3f},"
            f"nonzero={int(ctx.rule_aggregation_baseline_results.get('num_nonzero_rules', 0))}"
        ),
    )


def run_step_18d_object_to_atom_coverage(ctx: PipelineContext, runner: StepRunner) -> None:
    object_to_atom_coverage_cfg = _get_object_to_atom_coverage_diagnostic_cfg()
    runner.announce_step("18D", "object_to_atom_coverage_diagnostic_driving_mini")
    if not bool(object_to_atom_coverage_cfg.get("enabled", False)):
        runner.log("18D", "disabled by config")
        runner.complete_step("18D", subtitle="skipped")
        return
    runner.log("18D", f"cfg={object_to_atom_coverage_cfg}")
    runner.log("18D", f"recompute={bool(ctx.recompute_cfg.get('object_to_atom_coverage_diagnostic', True))}")
    with runner.module_output("18D"):
        ctx.object_to_atom_coverage_results = object_to_atom_coverage_diagnostic_driving_mini.run(
            detection_results=ctx.detection_results or [],
            dataset_annotation_results=ctx.dataset_annotation_results or [],
            merged_results=ctx.merged_results or [],
            important_object_results=ctx.important_object_results or [],
            logic_atom_results=ctx.logic_atom_results or [],
            extended_rule_results=ctx.extended_rule_results,
            original_final_rule_results=ctx.final_rule_results,
            diverse_final_rule_results=ctx.diverse_final_rule_results,
            semantic_constrained_diverse_final_rule_results=ctx.semantic_constrained_diverse_rule_results,
            coverage_family_aware_final_rule_results=ctx.coverage_family_aware_rule_results,
            rule_aggregation_baseline_results=ctx.rule_aggregation_baseline_results,
            cfg=object_to_atom_coverage_cfg,
            output_root=_get_object_to_atom_coverage_diagnostic_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("object_to_atom_coverage_diagnostic", True)),
        )
    runner.log("18D", f"classes={int(ctx.object_to_atom_coverage_results.get('num_classes', 0))}")
    runner.complete_step("18D", subtitle=f"classes={int(ctx.object_to_atom_coverage_results.get('num_classes', 0))}")


def run_step_18e_traffic_control_rule_utility(ctx: PipelineContext, runner: StepRunner) -> None:
    traffic_control_rule_utility_cfg = _get_traffic_control_rule_utility_diagnostic_cfg()
    runner.announce_step("18E", "traffic_control_rule_utility_diagnostic_driving_mini")
    if not bool(traffic_control_rule_utility_cfg.get("enabled", False)):
        runner.log("18E", "disabled by config")
        runner.complete_step("18E", subtitle="skipped")
        return
    runner.log("18E", f"cfg={traffic_control_rule_utility_cfg}")
    runner.log("18E", f"recompute={bool(ctx.recompute_cfg.get('traffic_control_rule_utility_diagnostic', True))}")
    with runner.module_output("18E"):
        ctx.traffic_control_rule_utility_results = traffic_control_rule_utility_diagnostic_driving_mini.run(
            extended_rule_results=ctx.extended_rule_results,
            eval_temporal_rule_results=ctx.eval_temporal_rule_results or [],
            rule_results_by_name=ctx.rule_results_by_name,
            rule_aggregation_baseline_results=ctx.rule_aggregation_baseline_results,
            cfg=traffic_control_rule_utility_cfg,
            output_root=_get_traffic_control_rule_utility_diagnostic_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("traffic_control_rule_utility_diagnostic", True)),
        )
    runner.log("18E", f"keys={len(list(ctx.traffic_control_rule_utility_results.get('rows', [])))}")
    runner.complete_step(
        "18E",
        subtitle=f"keys={len(list(ctx.traffic_control_rule_utility_results.get('rows', [])))}",
    )


def run_step_18f_traffic_control_temporal_alignment(ctx: PipelineContext, runner: StepRunner) -> None:
    temporal_alignment_cfg = _get_traffic_control_temporal_alignment_diagnostic_cfg()
    runner.announce_step("18F", "traffic_control_temporal_alignment_diagnostic_driving_mini")
    if not bool(temporal_alignment_cfg.get("enabled", False)):
        runner.log("18F", "disabled by config")
        runner.complete_step("18F", subtitle="skipped")
        return
    runner.log("18F", f"cfg={temporal_alignment_cfg}")
    runner.log(
        "18F",
        f"recompute={bool(ctx.recompute_cfg.get('traffic_control_temporal_alignment_diagnostic', True))}",
    )
    with runner.module_output("18F"):
        ctx.traffic_control_temporal_alignment_results = (
            traffic_control_temporal_alignment_diagnostic_driving_mini.run(
                logic_atom_results=ctx.logic_atom_results or [],
                eval_temporal_rule_results=ctx.eval_temporal_rule_results or [],
                cfg=temporal_alignment_cfg,
                output_root=_get_traffic_control_temporal_alignment_diagnostic_output_root(),
                force_recompute=bool(
                    ctx.recompute_cfg.get("traffic_control_temporal_alignment_diagnostic", True)
                ),
            )
        )
    runner.log(
        "18F",
        f"segments={int(ctx.traffic_control_temporal_alignment_results.get('num_segments_with_any_traffic_control', 0))}",
    )
    runner.complete_step(
        "18F",
        subtitle=(
            "segments="
            f"{int(ctx.traffic_control_temporal_alignment_results.get('num_segments_with_any_traffic_control', 0))}"
        ),
    )


def run_step_18g_traffic_light_detection_quality_audit(ctx: PipelineContext, runner: StepRunner) -> None:
    detection_quality_audit_cfg = _get_traffic_light_detection_quality_audit_cfg()
    runner.announce_step("18G", "traffic_light_detection_quality_audit_driving_mini")
    if not bool(detection_quality_audit_cfg.get("enabled", False)):
        runner.log("18G", "disabled by config")
        runner.complete_step("18G", subtitle="skipped")
        return
    runner.log("18G", f"cfg={detection_quality_audit_cfg}")
    runner.log(
        "18G",
        f"recompute={bool(ctx.recompute_cfg.get('traffic_light_detection_quality_audit', True))}",
    )
    with runner.module_output("18G"):
        ctx.traffic_light_detection_quality_audit_results = (
            traffic_light_detection_quality_audit_driving_mini.run(
                traffic_control_attribute_results=ctx.traffic_control_attribute_results or [],
                logic_atom_results=ctx.logic_atom_results or [],
                eval_temporal_rule_results=ctx.eval_temporal_rule_results or [],
                cfg=detection_quality_audit_cfg,
                output_root=_get_traffic_light_detection_quality_audit_output_root(),
                force_recompute=bool(ctx.recompute_cfg.get("traffic_light_detection_quality_audit", True)),
            )
        )
    runner.log(
        "18G",
        f"saved={int(ctx.traffic_light_detection_quality_audit_results.get('num_saved_audit_images', 0))}",
    )
    runner.complete_step(
        "18G",
        subtitle=f"saved={int(ctx.traffic_light_detection_quality_audit_results.get('num_saved_audit_images', 0))}",
    )


def run_step_18j_baseline_safe_calibration_gate(ctx: PipelineContext, runner: StepRunner) -> None:
    baseline_safe_calibration_gate_cfg = _get_baseline_safe_calibration_gate_cfg()
    iteration_id = _ensure_od_calibration_iteration_id(ctx)
    runner.announce_step("18J", "baseline_safe_calibration_gate_driving_mini")
    runner.log("18J", f"cfg={baseline_safe_calibration_gate_cfg}")
    runner.log("18J", f"iteration_id={iteration_id}")
    runner.log("18J", f"recompute={bool(ctx.recompute_cfg.get('baseline_safe_calibration_gate', True))}")
    with runner.module_output("18J"):
        ctx.baseline_safe_calibration_gate_results = baseline_safe_calibration_gate_driving_mini.run(
            evaluation_results_by_name=ctx.evaluation_results_by_name,
            primary_rule_set=ctx.primary_rule_set,
            rule_aggregation_baseline_results=ctx.rule_aggregation_baseline_results,
            calibration_results=ctx.od_confidence_calibration_results,
            iteration_id=iteration_id,
            cfg=baseline_safe_calibration_gate_cfg,
            output_root=_get_od_confidence_calibration_loop_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("baseline_safe_calibration_gate", True)),
        )
    runner.log(
        "18J",
        f"decision={ctx.baseline_safe_calibration_gate_results.get('decision', 'n/a')} "
        f"active_policy_after={ctx.baseline_safe_calibration_gate_results.get('active_policy_after_id', '') or 'none'}",
    )
    runner.complete_step(
        "18J",
        subtitle=f"decision={ctx.baseline_safe_calibration_gate_results.get('decision', 'n/a')}",
        allow_stop=not ctx.defer_stop_after_gate,
    )


def run_step_20_error_analysis(ctx: PipelineContext, runner: StepRunner) -> None:
    error_analysis_cfg = _get_error_and_explainability_cfg()
    runner.announce_step("20", "error_and_explainability_analysis_driving_mini")
    if not bool(error_analysis_cfg.get("enabled", False)):
        runner.log("20", "disabled by config")
        runner.complete_step("20", subtitle="skipped")
        return
    runner.log("20", f"cfg={error_analysis_cfg}")
    error_analysis_output_root = _get_error_and_explainability_output_root()
    ctx.error_analysis_results_by_name = {}
    for rule_set_name in ctx.evaluation_rule_sets:
        error_cfg_for_run = dict(error_analysis_cfg)
        error_cfg_for_run["rule_set_name"] = rule_set_name
        output_root = (
            error_analysis_output_root
            if rule_set_name == ctx.primary_rule_set
            else error_analysis_output_root / rule_set_name
        )
        runner.log("20", f"analyzing rule_set={rule_set_name}")
        with runner.module_output("20"):
            error_analysis_result = error_and_explainability_analysis_driving_mini.run(
                final_rule_results=ctx.rule_results_by_name[rule_set_name],
                temporal_rule_results=ctx.eval_temporal_rule_results or [],
                evaluation_results=ctx.evaluation_results_by_name[rule_set_name],
                cfg=error_cfg_for_run,
                output_root=output_root,
                force_recompute=bool(ctx.recompute_cfg.get("error_and_explainability_analysis", True)),
            )
        ctx.error_analysis_results_by_name[rule_set_name] = error_analysis_result

    if ctx.rule_set_mode in {"both", "all"}:
        comparison_paths = _write_rule_set_error_comparison(
            error_analysis_output_root=error_analysis_output_root,
            primary_rule_set=ctx.primary_rule_set,
            rule_set_mode=ctx.rule_set_mode,
            evaluation_rule_sets=ctx.evaluation_rule_sets,
            rule_results_by_name=ctx.rule_results_by_name,
            evaluation_results_by_name=ctx.evaluation_results_by_name,
            error_analysis_results_by_name=ctx.error_analysis_results_by_name,
        )
        runner.log("20", f"comparison={comparison_paths['comparison_json_path']}")

    ctx.error_analysis_results = ctx.error_analysis_results_by_name[ctx.primary_rule_set]
    runner.log(
        "20",
        f"rule_set={ctx.primary_rule_set} "
        f"fn={int(ctx.error_analysis_results.get('num_fn_examples', 0))} "
        f"fp={int(ctx.error_analysis_results.get('num_fp_examples', 0))}",
    )
    runner.complete_step(
        "20",
        subtitle=f"fn={int(ctx.error_analysis_results.get('num_fn_examples', 0))},fp={int(ctx.error_analysis_results.get('num_fp_examples', 0))}",
    )


def run_step_21_vehicle_rule_diagnostic(ctx: PipelineContext, runner: StepRunner) -> None:
    vehicle_rule_diagnostic_cfg = _get_vehicle_rule_diagnostic_cfg()
    vehicle_rule_diagnostic_cfg["primary_rule_set"] = ctx.primary_rule_set
    runner.announce_step("21", "vehicle_rule_diagnostic_driving_mini")
    if not bool(vehicle_rule_diagnostic_cfg.get("enabled", False)):
        runner.log("21", "disabled by config")
        runner.complete_step("21", subtitle="skipped")
        return
    runner.log("21", f"cfg={vehicle_rule_diagnostic_cfg}")
    runner.log("21", f"recompute={bool(ctx.recompute_cfg.get('vehicle_rule_diagnostic', True))}")
    with runner.module_output("21"):
        ctx.vehicle_rule_diagnostic_results = vehicle_rule_diagnostic_driving_mini.run(
            merged_initial_rules=ctx.merged_candidate_rules,
            extended_rule_results=ctx.extended_rule_results,
            original_final_rule_results=ctx.final_rule_results,
            diverse_final_rule_results=ctx.diverse_final_rule_results,
            semantic_constrained_diverse_final_rule_results=ctx.semantic_constrained_diverse_rule_results,
            coverage_family_aware_final_rule_results=ctx.coverage_family_aware_rule_results,
            eval_temporal_rule_results=ctx.eval_temporal_rule_results or [],
            evaluation_results_by_name=ctx.evaluation_results_by_name,
            cfg=vehicle_rule_diagnostic_cfg,
            output_root=_get_vehicle_rule_diagnostic_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("vehicle_rule_diagnostic", True)),
        )
    runner.log("21", f"diagnosis={ctx.vehicle_rule_diagnostic_results.get('primary_diagnosis', 'unknown')}")
    runner.complete_step(
        "21",
        subtitle=f"diagnosis={ctx.vehicle_rule_diagnostic_results.get('primary_diagnosis', 'unknown')}",
    )


def run_step_21b_fn_categorization(ctx: PipelineContext, runner: StepRunner) -> None:
    fn_categorization_cfg = _get_fn_categorization_diagnostic_cfg()
    runner.announce_step("21B", "fn_categorization_diagnostic_driving_mini")
    if bool(fn_categorization_cfg.get("enabled", False)):
        runner.log("21B", f"cfg={fn_categorization_cfg}")
        runner.log("21B", f"recompute={bool(ctx.recompute_cfg.get('fn_categorization_diagnostic', True))}")
        with runner.module_output("21B"):
            ctx.fn_categorization_results = fn_categorization_diagnostic_driving_mini.run(
                extended_rule_results=ctx.extended_rule_results,
                rule_results_by_name=ctx.rule_results_by_name,
                evaluation_results_by_name=ctx.evaluation_results_by_name,
                error_analysis_results_by_name=ctx.error_analysis_results_by_name,
                temporal_rule_results=ctx.eval_temporal_rule_results or [],
                vehicle_rule_diagnostic_results=ctx.vehicle_rule_diagnostic_results,
                cfg=fn_categorization_cfg,
                output_root=_get_fn_categorization_diagnostic_output_root(),
                force_recompute=bool(ctx.recompute_cfg.get("fn_categorization_diagnostic", True)),
            )
        runner.log("21B", f"summary={ctx.fn_categorization_results.get('summary_path', '')}")
        runner.complete_step("21B", subtitle="enabled")
        return
    runner.log("21B", "disabled by config")
    runner.complete_step("21B", subtitle="skipped")


def run_step_22_rule_selection_visualization(ctx: PipelineContext, runner: StepRunner) -> None:
    rule_selection_visualization_cfg = _get_rule_selection_visualization_cfg()
    runner.announce_step("22", "rule_selection_visualization_driving_mini")
    if not bool(rule_selection_visualization_cfg.get("enabled", False)):
        runner.log("22", "disabled by config")
        runner.complete_step("22", subtitle="skipped")
        return
    runner.log("22", f"cfg={rule_selection_visualization_cfg}")
    runner.log("22", f"recompute={bool(ctx.recompute_cfg.get('rule_selection_visualization', True))}")
    with runner.module_output("22"):
        ctx.rule_selection_visualization_results = rule_selection_visualization_driving_mini.run(
            cfg=rule_selection_visualization_cfg,
            output_root=_get_rule_selection_visualization_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("rule_selection_visualization", True)),
        )
    runner.log("22", f"figures={len(ctx.rule_selection_visualization_results.get('figure_paths', {}))}")
    runner.complete_step("22", subtitle="figures=ready")


def run_step_23_integrated_visualization(ctx: PipelineContext, runner: StepRunner) -> None:
    integrated_method_visualization_cfg = _get_integrated_method_visualization_cfg()
    runner.announce_step("23", "integrated_method_visualization_driving_mini")
    if not bool(integrated_method_visualization_cfg.get("enabled", False)):
        runner.log("23", "disabled by config")
        runner.complete_step("23", subtitle="skipped")
        return
    runner.log("23", f"cfg={integrated_method_visualization_cfg}")
    runner.log("23", f"recompute={bool(ctx.recompute_cfg.get('integrated_method_visualization', True))}")
    with runner.module_output("23"):
        ctx.integrated_method_visualization_results = integrated_method_visualization_driving_mini.run(
            cfg=integrated_method_visualization_cfg,
            output_root=_get_integrated_method_visualization_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("integrated_method_visualization", True)),
        )
    runner.log("23", f"figures={len(ctx.integrated_method_visualization_results.get('figure_paths', {}))}")
    runner.complete_step("23", subtitle="figures=ready")


def run_step_24a_background_causal_prior(ctx: PipelineContext, runner: StepRunner) -> None:
    background_causal_prior_cfg = _get_background_causal_prior_cfg()
    runner.announce_step("24A", "background_causal_prior_driving_mini")
    if not bool(background_causal_prior_cfg.get("enabled", False)):
        runner.log("24A", "disabled by config")
        runner.complete_step("24A", subtitle="skipped")
        return
    runner.log("24A", f"cfg={background_causal_prior_cfg}")
    runner.log("24A", f"recompute={bool(ctx.recompute_cfg.get('background_causal_prior', True))}")
    with runner.module_output("24A"):
        ctx.background_causal_prior_results = background_causal_prior_driving_mini.run(
            cfg=background_causal_prior_cfg,
            output_root=_get_background_causal_prior_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("background_causal_prior", True)),
        )
    runner.log("24A", f"entries={int(ctx.background_causal_prior_results.get('num_prior_entries', 0))}")
    runner.complete_step("24A", subtitle=f"entries={int(ctx.background_causal_prior_results.get('num_prior_entries', 0))}")


def run_step_24b_reasoning_feedback_signal(ctx: PipelineContext, runner: StepRunner) -> None:
    reasoning_feedback_signal_cfg = _get_reasoning_feedback_signal_cfg()
    runner.announce_step("24B", "reasoning_feedback_signal_driving_mini")
    if not bool(reasoning_feedback_signal_cfg.get("enabled", False)):
        runner.log("24B", "disabled by config")
        runner.complete_step("24B", subtitle="skipped")
        return
    runner.log("24B", f"cfg={reasoning_feedback_signal_cfg}")
    runner.log("24B", f"recompute={bool(ctx.recompute_cfg.get('reasoning_feedback_signal', True))}")
    with runner.module_output("24B"):
        ctx.reasoning_feedback_signal_results = reasoning_feedback_signal_driving_mini.run(
            background_causal_prior_results=ctx.background_causal_prior_results,
            primary_rule_results=ctx.rule_results_by_name.get(ctx.primary_rule_set, {}),
            evaluation_results=ctx.evaluation_results,
            rule_aggregation_baseline_results=ctx.rule_aggregation_baseline_results,
            error_analysis_results=ctx.error_analysis_results,
            eval_temporal_rule_results=ctx.eval_temporal_rule_results or [],
            logic_atom_results=ctx.logic_atom_results or [],
            cfg=reasoning_feedback_signal_cfg,
            output_root=_get_reasoning_feedback_signal_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("reasoning_feedback_signal", True)),
        )
    runner.log("24B", f"requests={int(ctx.reasoning_feedback_signal_results.get('num_feedback_requests', 0))}")
    runner.complete_step("24B", subtitle=f"requests={int(ctx.reasoning_feedback_signal_results.get('num_feedback_requests', 0))}")


def _run_steps_through_19(
    ctx: PipelineContext,
    runner: StepRunner,
    *,
    start_target: str,
    stop_target: str,
) -> None:
    if _step_is_requested("0", start_target=start_target, stop_target=stop_target):
        run_step_0_background_rule_relevance_prior(ctx, runner)

    # Perception
    if _step_is_requested("1", start_target=start_target, stop_target=stop_target):
        run_step_1_detection(ctx, runner)
    if _step_is_requested("2", start_target=start_target, stop_target=stop_target):
        run_step_2_tracking(ctx, runner)
    if _step_is_requested("3", start_target=start_target, stop_target=stop_target):
        run_step_3_dataset_annotations(ctx, runner)
    if _step_is_requested("4", start_target=start_target, stop_target=stop_target):
        run_step_4_merge_annotations(ctx, runner)
    if _step_is_requested("5", start_target=start_target, stop_target=stop_target):
        run_step_5_prepare_3d_positions(ctx, runner)

    # Motion and symbolic scene abstraction
    if _step_is_requested("6", start_target=start_target, stop_target=stop_target):
        run_step_6_ego_motion(ctx, runner)
    if _step_is_requested("7", start_target=start_target, stop_target=stop_target):
        run_step_7_relative_object_motion(ctx, runner)
    if _step_is_requested("8", start_target=start_target, stop_target=stop_target):
        run_step_8_temporal_segmentation(ctx, runner)
    if _step_is_requested("9", start_target=start_target, stop_target=stop_target):
        run_step_9_segment_object_motion(ctx, runner)
    if _step_is_requested("10", start_target=start_target, stop_target=stop_target):
        run_step_10_important_objects(ctx, runner)
    if _step_is_requested("10B", start_target=start_target, stop_target=stop_target):
        run_step_10b_traffic_control_attributes(ctx, runner)

    # Logic and rule mining
    if _step_is_requested("11", start_target=start_target, stop_target=stop_target):
        run_step_11_logic_atoms(ctx, runner)
    if _step_is_requested("12", start_target=start_target, stop_target=stop_target):
        run_step_12_target_head_atoms(ctx, runner)
    if _step_is_requested("13", start_target=start_target, stop_target=stop_target):
        run_step_13_temporal_rule_examples(ctx, runner)
    if _start_target_reaches("13", start_target) and _stop_target_reaches("13", stop_target):
        _prepare_rule_learning_inputs(ctx)
        runner.log(
            "13",
            "split "
            f"train_videos={len(ctx.split_manifest.get('train_video_ids', []))} "
            f"eval_videos={len(ctx.split_manifest.get('eval_video_ids', []))}",
        )
        runner.log("13", f"recompute_cfg={ctx.recompute_cfg}")
    if _step_is_requested("14", start_target=start_target, stop_target=stop_target):
        run_step_14_candidate_rules(ctx, runner)
    if _step_is_requested("15", start_target=start_target, stop_target=stop_target):
        run_step_15_merge_initial_rules(ctx, runner)
    if _step_is_requested("15B", start_target=start_target, stop_target=stop_target):
        run_step_15b_initial_rule_pruning(ctx, runner)
    if _step_is_requested("16", start_target=start_target, stop_target=stop_target):
        run_step_16_extended_rules(ctx, runner)

    # Rule selection and evaluation
    if _step_is_requested("17", start_target=start_target, stop_target=stop_target):
        run_step_17_final_rules(ctx, runner)
    if _step_is_requested("17D", start_target=start_target, stop_target=stop_target):
        run_step_17d_rule_pool_and_selector_diagnostic(ctx, runner)
    if _step_is_requested("18", start_target=start_target, stop_target=stop_target):
        run_step_18_rule_evaluation(ctx, runner)
    if _step_is_requested("18B", start_target=start_target, stop_target=stop_target):
        run_step_18b_baseline_comparison(ctx, runner)
    if _step_is_requested("19", start_target=start_target, stop_target=stop_target):
        run_step_19_reasoning_supervised_od_calibration(ctx, runner)


def _run_post_loop_steps(
    ctx: PipelineContext,
    runner: StepRunner,
    *,
    start_target: str,
    stop_target: str,
) -> None:
    if _step_is_requested("18D", start_target=start_target, stop_target=stop_target):
        run_step_18d_object_to_atom_coverage(ctx, runner)
    if _step_is_requested("18E", start_target=start_target, stop_target=stop_target):
        run_step_18e_traffic_control_rule_utility(ctx, runner)
    if _step_is_requested("18F", start_target=start_target, stop_target=stop_target):
        run_step_18f_traffic_control_temporal_alignment(ctx, runner)
    if _step_is_requested("18G", start_target=start_target, stop_target=stop_target):
        run_step_18g_traffic_light_detection_quality_audit(ctx, runner)
    if _step_is_requested("20", start_target=start_target, stop_target=stop_target):
        run_step_20_error_analysis(ctx, runner)
    if _step_is_requested("21", start_target=start_target, stop_target=stop_target):
        run_step_21_vehicle_rule_diagnostic(ctx, runner)
    if _step_is_requested("21B", start_target=start_target, stop_target=stop_target):
        run_step_21b_fn_categorization(ctx, runner)
    if _step_is_requested("22", start_target=start_target, stop_target=stop_target):
        run_step_22_rule_selection_visualization(ctx, runner)
    if _step_is_requested("23", start_target=start_target, stop_target=stop_target):
        run_step_23_integrated_visualization(ctx, runner)
    if _step_is_requested("24A", start_target=start_target, stop_target=stop_target):
        run_step_24a_background_causal_prior(ctx, runner)
    if _step_is_requested("24B", start_target=start_target, stop_target=stop_target):
        run_step_24b_reasoning_feedback_signal(ctx, runner)


def _run_single_pass_pipeline(
    runner: StepRunner,
    *,
    start_target: str,
    stop_target: str,
    video_ids: Optional[List[str]],
    video_count: int | None,
) -> None:
    ctx = _build_pipeline_context(
        video_ids=video_ids,
        video_count=video_count,
    )
    if _stage_index(start_target) > _stage_index("16"):
        runner.log(
            start_target,
            f"warm_start_from={start_target} using cached upstream artifacts through step 16",
        )
    _load_cached_upstream_context(ctx, start_target=start_target, stop_target=stop_target)
    _run_steps_through_19(ctx, runner, start_target=start_target, stop_target=stop_target)
    if _stop_target_reaches("18D", stop_target):
        _run_post_loop_steps(ctx, runner, start_target=start_target, stop_target=stop_target)


def _run_od_calibration_loop(
    runner: StepRunner,
    *,
    start_target: str,
    stop_target: str,
    video_ids: Optional[List[str]],
    video_count: int | None,
    loop_cfg: od_calibration_loop_utils.ODCalibrationLoopConfig,
) -> PipelineContext:
    final_ctx: Optional[PipelineContext] = None
    for iteration_index in range(1, int(loop_cfg.max_iterations) + 1):
        force_full_recompute = od_calibration_loop_utils.iteration_requires_full_recompute(
            iteration_index,
            force_full_recompute_on_policy_change=loop_cfg.force_full_recompute_on_policy_change,
        )
        ctx = _build_pipeline_context(
            video_ids=video_ids,
            video_count=video_count,
            iteration_index=iteration_index,
            force_full_recompute=force_full_recompute,
            defer_stop_after_gate=True,
        )
        if _stage_index(start_target) > _stage_index("16"):
            runner.log(
                "19",
                f"warm_start_from={start_target} using cached upstream artifacts through step 16",
            )
        _load_cached_upstream_context(ctx, start_target=start_target, stop_target=stop_target)
        active_policy_before = od_calibration_policy_utils.load_active_od_calibration_policy()
        runner.log(
            "19",
            "od_calibration_loop "
            f"iteration={iteration_index}/{loop_cfg.max_iterations} "
            f"input_policy={od_calibration_policy_utils.policy_id(active_policy_before) or 'identity'} "
            f"force_recompute={force_full_recompute}",
        )
        _run_steps_through_19(ctx, runner, start_target=start_target, stop_target=stop_target)
        decision = od_calibration_loop_utils.should_continue_after_iteration(
            ctx.baseline_safe_calibration_gate_results,
            iteration_index=iteration_index,
            loop_cfg=loop_cfg,
        )
        ctx.od_loop_stop_reason = decision.reason
        reference_final_f1 = (
            f"{decision.reference_final_f1:.3f}"
            if decision.reference_final_f1 is not None
            else "n/a"
        )
        runner.log(
            "19",
            "od_calibration_loop_result "
            f"reason={decision.reason} "
            f"current_final_f1={decision.current_final_f1:.3f} "
            f"reference_final_f1={reference_final_f1}",
        )
        final_ctx = ctx
        if not decision.continue_loop:
            return ctx
    if final_ctx is None:
        raise RuntimeError("OD calibration loop did not execute any iterations.")
    return final_ctx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the driving_mini experiment pipeline up to a selected step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "max_step",
        nargs="?",
        type=_parse_max_step_arg,
        default="24",
        help="Run through this step number or exact step id such as 17D or 18D.",
    )
    parser.add_argument(
        "--start-step",
        dest="start_step",
        type=_parse_max_step_arg,
        default="17",
        help="Start from this step number or exact step id. Defaults to warm-starting at step 17.",
    )
    parser.add_argument(
        "--video-id",
        dest="video_ids",
        action="append",
        help=(
            "Restrict the pipeline to one or more video IDs. "
            "If omitted, the pipeline uses the default subset sized for 100 train and 2 eval videos."
        ),
    )
    parser.add_argument(
        "--data",
        "--video-count",
        dest="video_count",
        type=int,
        help=(
            "Restrict the pipeline to the first N videos. "
            "When used, downstream train/eval splits scale to consume the selected N videos."
        ),
    )
    parser.add_argument(
        "--od-calibration-iterations",
        dest="od_calibration_iterations",
        type=int,
        help="Override the configured maximum number of OD calibration loop iterations.",
    )
    return parser.parse_args()


def main(
    start_step: int | str = 17,
    max_step: int | str = 24,
    video_ids: Optional[List[str]] = None,
    video_count: int | None = None,
    od_calibration_iterations: int | None = None,
) -> None:
    if video_count is not None and video_count < 1:
        raise ValueError(f"video_count must be >= 1. Found {video_count}.")
    if od_calibration_iterations is not None and od_calibration_iterations < 1:
        raise ValueError(
            "od_calibration_iterations must be >= 1. "
            f"Found {od_calibration_iterations}."
        )
    resolved_start_target, _ = _resolve_stop_request(start_step)
    resolved_stop_target, _ = _resolve_stop_request(max_step)
    if _stage_index(resolved_start_target) > _stage_index(resolved_stop_target):
        raise ValueError(
            f"start_step={start_step} must be at or before max_step={max_step}."
        )
    loop_cfg = _resolve_od_calibration_loop_cfg(
        max_iterations_override=od_calibration_iterations,
    )
    runner = StepRunner.create(
        max_step,
        total_rtpt_iterations=_estimated_rtpt_iterations(resolved_stop_target, loop_cfg),
    )
    try:
        if _should_run_od_calibration_loop(resolved_stop_target, loop_cfg):
            ctx = _run_od_calibration_loop(
                runner,
                start_target=resolved_start_target,
                stop_target=resolved_stop_target,
                video_ids=video_ids,
                video_count=video_count,
                loop_cfg=loop_cfg,
            )
            runner.log("19", f"od_calibration_loop_stop_reason={ctx.od_loop_stop_reason or 'completed'}")
            if resolved_stop_target == "19":
                runner.stop_now()
            if _stop_target_reaches("18D", resolved_stop_target):
                _run_post_loop_steps(
                    ctx,
                    runner,
                    start_target=resolved_start_target,
                    stop_target=resolved_stop_target,
                )
            return
        _run_single_pass_pipeline(
            runner,
            start_target=resolved_start_target,
            stop_target=resolved_stop_target,
            video_ids=video_ids,
            video_count=video_count,
        )
    except _PipelineStopRequested:
        return


if __name__ == "__main__":
    args = parse_args()
    main(
        start_step=args.start_step,
        max_step=args.max_step,
        video_ids=args.video_ids,
        video_count=args.video_count,
        od_calibration_iterations=args.od_calibration_iterations,
    )
