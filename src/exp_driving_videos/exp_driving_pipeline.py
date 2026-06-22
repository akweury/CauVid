"""
Experiment pipeline for the driving_mini dataset.

Steps:
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
    16. extended_rules_driving_mini — iteratively extend merged rules with
                    initial-rule body atoms for a fixed number of rounds.
    17. final_rules_driving_mini — rank all kept rules by confidence and keep
                    the top-k as the final rule set.
    17B. diverse_final_rules_driving_mini — greedily choose a diverse final
                    rule set that expands positive coverage while penalizing
                    overlap and repeated rule families.
    17B2. semantic_constrained_diverse_final_rules_driving_mini — greedy
                    positive-coverage selection with semantic minimums for
                    vehicle/near/centered-related rule families.
    17C. coverage_family_aware_final_rules_driving_mini — greedily choose a
                    coverage-aware final rule set using rule quality
                    confidence * log(1 + positive_support) together with
                    redundancy and family-diversity penalties.
    17D. rule_pool_upper_bound_diagnostic_driving_mini — evaluate the full
                    extended-rule pool on held-out examples and estimate
                    single-rule and oracle-greedy upper bounds.
    17E. oracle_rule_selection_gap_diagnostic_driving_mini — inspect the
                    oracle top-K pool rules from step 17D, compare them
                    against actual selector outputs, and summarize why the
                    current selectors missed oracle rules.
    18. evaluate_rules_driving_mini — evaluate the learned final rules on the
                    held-out evaluation split.
    18B. neural_symbolic_baseline_driving_mini — train single-segment and
                    short-history symbolic neural baselines on the same
                    train/eval split and compare held-out classification
                    metrics.
    18C. rule_aggregation_baseline_driving_mini — train a sparse logistic
                    regression over Step 16 rule firings, tuned on
                    validation data, and evaluate on the held-out split.
    18D. object_to_atom_coverage_diagnostic_driving_mini — trace normalized
                    object classes from detections/annotations through
                    merged objects, important objects, logic atoms, the
                    extended rule pool, selected hard-OR rule sets, and the
                    top weighted learned aggregation rules.
    18E. traffic_control_rule_utility_diagnostic_driving_mini — audit
                    traffic-control predicates and traffic-light states
                    across the Step 16 pool, hard-OR selectors, and
                    learned Step 18C aggregation weights.
    18F. traffic_control_temporal_alignment_diagnostic_driving_mini —
                    test whether traffic-control predicates align with
                    the immediate brake_next target or delayed braking
                    within 2/3/5 future segments using diagnostic-only
                    future labels.
    19. error_and_explainability_analysis_driving_mini — summarize false
                    negatives / false positives and generate explainability-
                    oriented diagnostics for held-out evaluation examples.
    20. vehicle_rule_diagnostic_driving_mini — audit whether vehicle-centered
                    rules were generated, pruned/scored out, or missed due to
                    predicate representation.
    20B. fn_categorization_diagnostic_driving_mini — categorize false
                    negatives for each selector using the step 16 rule pool,
                    step 18 predictions, step 19 FN examples, and step 20
                    vehicle-context diagnostics.
    21. rule_selection_visualization_driving_mini — generate comparison plots
                    from the step 18/19/20 selector summaries and save a
                    visualization manifest.
    22. integrated_method_visualization_driving_mini — generate integrated
                    publication-style figures comparing NeSy selectors,
                    neural symbolic baselines, learned rule aggregation, and
                    the oracle rule-pool upper bound.

"""

from __future__ import annotations

import argparse
import csv
import io
import json
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
from src.exp_driving_videos import pipeline_config as driving_pipeline_config
from src.exp_driving_videos import pipeline_data as driving_pipeline_data
from src.exp_driving_videos.modules import detect_driving_mini
from src.exp_driving_videos.modules import candidate_rules_driving_mini
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
from src.exp_driving_videos.modules import traffic_control_temporal_alignment_diagnostic_driving_mini
from src.exp_driving_videos.modules import vehicle_rule_diagnostic_driving_mini

DRIVING_MINI_OD_MODEL = driving_pipeline_config.DRIVING_MINI_OD_MODEL
DEFAULT_TRAIN_VIDEO_COUNT = driving_pipeline_config.DEFAULT_TRAIN_VIDEO_COUNT
DEFAULT_EVAL_VIDEO_COUNT = driving_pipeline_config.DEFAULT_EVAL_VIDEO_COUNT
DRIVING_MINI_OD_CLASSES = driving_pipeline_config.DRIVING_MINI_OD_CLASSES

_get_ego_motion_smoothing_window = driving_pipeline_config.get_ego_motion_smoothing_window
_get_detection_render_video_enabled = driving_pipeline_config.get_detection_render_video_enabled
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
_get_extended_rules_cfg = driving_pipeline_config.get_extended_rules_cfg
_get_final_rules_cfg = driving_pipeline_config.get_final_rules_cfg
_get_diverse_final_rules_cfg = driving_pipeline_config.get_diverse_final_rules_cfg
_get_semantic_constrained_diverse_cfg = driving_pipeline_config.get_semantic_constrained_diverse_cfg
_get_coverage_family_aware_final_rules_cfg = driving_pipeline_config.get_coverage_family_aware_final_rules_cfg
_get_rule_pool_upper_bound_diagnostic_cfg = driving_pipeline_config.get_rule_pool_upper_bound_diagnostic_cfg
_get_oracle_rule_selection_gap_diagnostic_cfg = driving_pipeline_config.get_oracle_rule_selection_gap_diagnostic_cfg
_get_data_split_cfg = driving_pipeline_config.get_data_split_cfg
_get_rule_evaluation_cfg = driving_pipeline_config.get_rule_evaluation_cfg
_get_rule_aggregation_baseline_cfg = driving_pipeline_config.get_rule_aggregation_baseline_cfg
_get_object_to_atom_coverage_diagnostic_cfg = driving_pipeline_config.get_object_to_atom_coverage_diagnostic_cfg
_get_traffic_control_rule_utility_diagnostic_cfg = (
    driving_pipeline_config.get_traffic_control_rule_utility_diagnostic_cfg
)
_get_traffic_control_temporal_alignment_diagnostic_cfg = (
    driving_pipeline_config.get_traffic_control_temporal_alignment_diagnostic_cfg
)
_get_neural_symbolic_baseline_cfg = driving_pipeline_config.get_neural_symbolic_baseline_cfg
_get_rule_selection_visualization_cfg = driving_pipeline_config.get_rule_selection_visualization_cfg
_get_integrated_method_visualization_cfg = driving_pipeline_config.get_integrated_method_visualization_cfg
_get_fn_categorization_diagnostic_cfg = driving_pipeline_config.get_fn_categorization_diagnostic_cfg
_get_pipeline_recompute_cfg = driving_pipeline_config.get_pipeline_recompute_cfg
_get_error_and_explainability_cfg = driving_pipeline_config.get_error_and_explainability_cfg
_get_vehicle_rule_diagnostic_cfg = driving_pipeline_config.get_vehicle_rule_diagnostic_cfg
_get_rule_evaluation_output_root = driving_pipeline_config.get_rule_evaluation_output_root
_get_rule_aggregation_baseline_output_root = driving_pipeline_config.get_rule_aggregation_baseline_output_root
_get_object_to_atom_coverage_diagnostic_output_root = driving_pipeline_config.get_object_to_atom_coverage_diagnostic_output_root
_get_traffic_control_rule_utility_diagnostic_output_root = (
    driving_pipeline_config.get_traffic_control_rule_utility_diagnostic_output_root
)
_get_traffic_control_temporal_alignment_diagnostic_output_root = (
    driving_pipeline_config.get_traffic_control_temporal_alignment_diagnostic_output_root
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

_merge_candidate_rules = driving_pipeline_data.merge_candidate_rules
_select_video_results = driving_pipeline_data.select_video_results
_build_train_eval_split = driving_pipeline_data.build_train_eval_split
_resolve_video_ids = driving_pipeline_data.resolve_video_ids

_PIPELINE_STAGE_SEQUENCE: List[Dict[str, Any]] = [
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
    {"tag": "16", "stop_after": 16, "label": "extended_rules_driving_mini"},
    {"tag": "17", "stop_after": 17, "label": "final_rules_driving_mini"},
    {"tag": "17B", "stop_after": None, "label": "diverse_final_rules_driving_mini"},
    {"tag": "17B2", "stop_after": None, "label": "semantic_constrained_diverse_final_rules_driving_mini"},
    {"tag": "17C", "stop_after": None, "label": "coverage_family_aware_final_rules_driving_mini"},
    {"tag": "17D", "stop_after": None, "label": "rule_pool_upper_bound_diagnostic_driving_mini"},
    {"tag": "17E", "stop_after": None, "label": "oracle_rule_selection_gap_diagnostic_driving_mini"},
    {"tag": "18", "stop_after": None, "label": "evaluate_rules_driving_mini"},
    {"tag": "18B", "stop_after": None, "label": "neural_symbolic_baseline_driving_mini"},
    {"tag": "18C", "stop_after": None, "label": "rule_aggregation_baseline_driving_mini"},
    {"tag": "18D", "stop_after": None, "label": "object_to_atom_coverage_diagnostic_driving_mini"},
    {"tag": "18E", "stop_after": None, "label": "traffic_control_rule_utility_diagnostic_driving_mini"},
    {"tag": "18F", "stop_after": 18, "label": "traffic_control_temporal_alignment_diagnostic_driving_mini"},
    {"tag": "19", "stop_after": 19, "label": "error_and_explainability_analysis_driving_mini"},
    {"tag": "20", "stop_after": 20, "label": "vehicle_rule_diagnostic_driving_mini"},
    {"tag": "20B", "stop_after": None, "label": "fn_categorization_diagnostic_driving_mini"},
    {"tag": "21", "stop_after": 21, "label": "rule_selection_visualization_driving_mini"},
    {"tag": "22", "stop_after": 22, "label": "integrated_method_visualization_driving_mini"},
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
    recompute_cfg: Dict[str, Any] = field(default_factory=dict)
    candidate_rule_results: Optional[List[Dict[str, Any]]] = None
    merged_candidate_rules: Dict[str, Any] = field(default_factory=dict)
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
    neural_symbolic_baseline_results: Dict[str, Any] = field(default_factory=dict)
    rule_aggregation_baseline_results: Dict[str, Any] = field(default_factory=dict)
    object_to_atom_coverage_results: Dict[str, Any] = field(default_factory=dict)
    traffic_control_rule_utility_results: Dict[str, Any] = field(default_factory=dict)
    traffic_control_temporal_alignment_results: Dict[str, Any] = field(default_factory=dict)
    error_analysis_results_by_name: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error_analysis_results: Dict[str, Any] = field(default_factory=dict)
    vehicle_rule_diagnostic_results: Dict[str, Any] = field(default_factory=dict)
    fn_categorization_results: Dict[str, Any] = field(default_factory=dict)
    rule_selection_visualization_results: Dict[str, Any] = field(default_factory=dict)
    integrated_method_visualization_results: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepRunner:
    stop_target: str
    stop_label: str
    rtpt: Optional[Any]

    @classmethod
    def create(cls, requested_step: int | str) -> "StepRunner":
        stop_target, stop_label = _resolve_stop_request(requested_step)
        return cls(
            stop_target=stop_target,
            stop_label=stop_label,
            rtpt=_start_rtpt(stop_target),
        )

    def announce_step(self, tag: str, name: str, *, leading_newline: bool = True) -> None:
        prefix = "\n" if leading_newline else ""
        print(f"{prefix}[Step {tag}] {name}")

    def log(self, tag: str, message: str) -> None:
        print(f"[Step {tag}] {message}")

    def complete_step(self, tag: str, subtitle: str = "") -> None:
        _rtpt_step(self.rtpt, tag, subtitle=subtitle)
        if str(tag) == self.stop_target:
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
        if step_number < 1 or step_number > 22:
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


def _rtpt_max_iterations(stop_target: str) -> int:
    return _PIPELINE_STAGE_TAGS.index(stop_target) + 1


def _start_rtpt(stop_target: str) -> Optional[Any]:
    if RTPT is None:
        print("RTPT unavailable; continuing without remote progress monitor.")
        return None
    rtpt = RTPT(
        name_initials="JI",
        experiment_name="DrivingMiniPipeline",
        max_iterations=_rtpt_max_iterations(stop_target),
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


def _run_object_detection_step(
    force_recompute: bool = False,
    video_ids: Optional[List[str]] = None,
    render_video: bool = True,
) -> List[Dict[str, Any]]:
    return detect_driving_mini.run(
        video_ids=video_ids,
        model_name=DRIVING_MINI_OD_MODEL,
        classes=DRIVING_MINI_OD_CLASSES,
        force_recompute=force_recompute,
        render_video=render_video,
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
        predicted_positive_ids = set(rule_eval_result.get("predicted_positive_example_ids", []))
        covered_original_fn_ids = original_fn_ids & predicted_positive_ids
        comparison_rows.append(
            {
                "rule_set_name": rule_set_name,
                "selection_method": str(rule_results_by_name[rule_set_name].get("selection_method", "score_top_k")),
                "num_final_rules": int(rule_results_by_name[rule_set_name].get("num_final_rules", 0)),
                "covered_eval_positive_examples": len(rule_eval_result.get("covered_positive_example_ids", [])),
                "num_fn_examples": len(rule_eval_result.get("false_negative_example_ids", [])),
                "fn_coverage_vs_original": float(len(covered_original_fn_ids) / max(1, len(original_fn_ids))),
                "precision": float(overall.get("precision", 0.0)),
                "recall": float(overall.get("recall", 0.0)),
                "f1": float(overall.get("f1", 0.0)),
                "accuracy": float(overall.get("accuracy", 0.0)),
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
                "covered_eval_positive_examples",
                "num_fn_examples",
                "fn_coverage_vs_original",
                "precision",
                "recall",
                "f1",
                "accuracy",
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
    ctx.split_manifest = _build_train_eval_split(
        video_ids=[str(result.get("video_id", "")) for result in temporal_rule_results],
        train_video_count=int(split_cfg.get("train_video_count", DEFAULT_TRAIN_VIDEO_COUNT)),
        eval_video_count=int(split_cfg.get("eval_video_count", DEFAULT_EVAL_VIDEO_COUNT)),
        strategy=str(split_cfg.get("strategy", "eval_fraction")),
        eval_fraction=float(split_cfg.get("eval_fraction", 0.2)),
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
    ctx.recompute_cfg = _get_pipeline_recompute_cfg()


def run_step_1_detection(ctx: PipelineContext, runner: StepRunner) -> None:
    render_video = _get_detection_render_video_enabled(default=True)
    runner.announce_step("1", "detect_driving_mini", leading_newline=False)
    runner.log("1", f"model={DRIVING_MINI_OD_MODEL}")
    runner.log("1", f"classes={len(DRIVING_MINI_OD_CLASSES)} configured")
    runner.log("1", f"render_video={render_video}")
    if ctx.effective_video_ids:
        runner.log("1", f"video_filter={ctx.effective_video_ids}")
    with runner.module_output("1"):
        ctx.detection_results = _run_object_detection_step(
            force_recompute=False,
            video_ids=ctx.effective_video_ids,
            render_video=render_video,
        )
    runner.log("1", f"completed videos={len(ctx.detection_results)}")
    runner.complete_step("1", subtitle=f"videos={len(ctx.detection_results)}")


def run_step_2_tracking(ctx: PipelineContext, runner: StepRunner) -> None:
    render_video = _get_tracking_render_video_enabled(default=True)
    runner.announce_step("2", "tracking_driving_mini")
    runner.log("2", f"render_video={render_video}")
    with runner.module_output("2"):
        ctx.tracking_results = tracking_driving_mini.run(
            ctx.detection_results or [],
            render_video=render_video,
        )
    runner.log("2", f"completed videos={len(ctx.tracking_results)}")
    runner.complete_step("2", subtitle=f"videos={len(ctx.tracking_results)}")


def run_step_3_dataset_annotations(ctx: PipelineContext, runner: StepRunner) -> None:
    runner.announce_step("3", "dataset_annotations_driving_mini")
    with runner.module_output("3"):
        ctx.dataset_annotation_results = dataset_annotations_driving_mini.run(
            video_ids=[r["video_id"] for r in (ctx.tracking_results or [])]
        )
    runner.log("3", f"completed videos={len(ctx.dataset_annotation_results)}")
    runner.complete_step("3", subtitle=f"videos={len(ctx.dataset_annotation_results)}")


def run_step_4_merge_annotations(ctx: PipelineContext, runner: StepRunner) -> None:
    render_video = _get_merge_render_video_enabled(default=True)
    runner.announce_step("4", "merge_gt_and_detected_driving_mini")
    runner.log("4", f"render_video={render_video}")
    with runner.module_output("4"):
        ctx.merged_results = merge_gt_and_detected_driving_mini.run(
            tracking_results=ctx.tracking_results or [],
            dataset_annotation_results=ctx.dataset_annotation_results or [],
            render_video=render_video,
        )
    runner.log("4", f"completed videos={len(ctx.merged_results)}")
    runner.complete_step("4", subtitle=f"videos={len(ctx.merged_results)}")


def run_step_5_prepare_3d_positions(ctx: PipelineContext, runner: StepRunner) -> None:
    runner.announce_step("5", "prepare_3d_positions_driving_mini")
    with runner.module_output("5"):
        ctx.positions_3d_results = prepare_3d_positions_driving_mini.run(
            merged_results=ctx.merged_results or [],
        )
    runner.log("5", f"completed videos={len(ctx.positions_3d_results)}")
    runner.complete_step("5", subtitle=f"videos={len(ctx.positions_3d_results)}")


def run_step_6_ego_motion(ctx: PipelineContext, runner: StepRunner) -> None:
    smoothing_window = _get_ego_motion_smoothing_window(default=5)
    static_adjust_cfg = _get_ego_static_adjustment_cfg()
    render_video = _get_ego_motion_render_video_enabled(default=True)
    runner.announce_step("6", "ego_motion_driving_mini")
    runner.log("6", f"smoothing_window={smoothing_window}")
    runner.log("6", f"static_adjustment={static_adjust_cfg}")
    runner.log("6", f"render_video={render_video}")
    with runner.module_output("6"):
        ctx.ego_motion_results = ego_motion_driving_mini.run(
            merged_results=ctx.merged_results or [],
            force_recompute=False,
            smoothing_window=smoothing_window,
            static_adjust_cfg=static_adjust_cfg,
            render_video=render_video,
        )
    runner.log("6", f"completed videos={len(ctx.ego_motion_results)}")
    runner.complete_step("6", subtitle=f"videos={len(ctx.ego_motion_results)}")


def run_step_7_relative_object_motion(ctx: PipelineContext, runner: StepRunner) -> None:
    runner.announce_step("7", "relative_object_motion_driving_mini")
    with runner.module_output("7"):
        ctx.relative_motion_results = relative_object_motion_driving_mini.run(
            positions_3d_results=ctx.positions_3d_results or [],
            ego_motion_results=ctx.ego_motion_results or [],
            force_recompute=False,
        )
    runner.log("7", f"completed videos={len(ctx.relative_motion_results)}")
    runner.complete_step("7", subtitle=f"videos={len(ctx.relative_motion_results)}")


def run_step_8_temporal_segmentation(ctx: PipelineContext, runner: StepRunner) -> None:
    temporal_seg_cfg = _get_temporal_segmentation_cfg()
    runner.announce_step("8", "temporal_segmentation_driving_mini")
    runner.log("8", f"cfg={temporal_seg_cfg}")
    with runner.module_output("8"):
        ctx.temporal_seg_results = temporal_segmentation_driving_mini.run(
            ego_motion_results=ctx.ego_motion_results or [],
            relative_motion_results=ctx.relative_motion_results or [],
            seg_cfg=temporal_seg_cfg,
            force_recompute=False,
        )
    runner.log("8", f"completed videos={len(ctx.temporal_seg_results)}")
    runner.complete_step("8", subtitle=f"videos={len(ctx.temporal_seg_results)}")


def run_step_9_segment_object_motion(ctx: PipelineContext, runner: StepRunner) -> None:
    segment_object_cfg = _get_segment_object_motion_cfg()
    runner.announce_step("9", "segment_object_motion_driving_mini")
    runner.log("9", f"cfg={segment_object_cfg}")
    with runner.module_output("9"):
        ctx.segment_object_results = segment_object_motion_driving_mini.run(
            relative_motion_results=ctx.relative_motion_results or [],
            temporal_segmentation_results=ctx.temporal_seg_results or [],
            cfg=segment_object_cfg,
            force_recompute=False,
        )
    runner.log("9", f"completed videos={len(ctx.segment_object_results)}")
    runner.complete_step("9", subtitle=f"videos={len(ctx.segment_object_results)}")


def run_step_10_important_objects(ctx: PipelineContext, runner: StepRunner) -> None:
    important_objects_cfg = _get_important_objects_cfg()
    runner.announce_step("10", "important_objects_driving_mini")
    runner.log("10", f"cfg={important_objects_cfg}")
    with runner.module_output("10"):
        ctx.important_object_results = important_objects_driving_mini.run(
            segment_object_motion_results=ctx.segment_object_results or [],
            cfg=important_objects_cfg,
            force_recompute=False,
        )
    runner.log("10", f"completed videos={len(ctx.important_object_results)}")
    runner.complete_step("10", subtitle=f"videos={len(ctx.important_object_results)}")


def run_step_10b_traffic_control_attributes(ctx: PipelineContext, runner: StepRunner) -> None:
    traffic_control_attributes_cfg = _get_traffic_control_attributes_cfg()
    runner.announce_step("10B", "traffic_control_attributes_driving_mini")
    runner.log("10B", f"cfg={traffic_control_attributes_cfg}")
    runner.log(
        "10B",
        f"recompute={bool(ctx.recompute_cfg.get('traffic_control_attributes', False))}",
    )
    with runner.module_output("10B"):
        ctx.traffic_control_attribute_results = traffic_control_attributes_driving_mini.run(
            important_object_results=ctx.important_object_results or [],
            relative_motion_results=ctx.relative_motion_results or [],
            cfg=traffic_control_attributes_cfg,
            force_recompute=bool(ctx.recompute_cfg.get("traffic_control_attributes", False)),
        )
    runner.log("10B", f"completed videos={len(ctx.traffic_control_attribute_results)}")
    runner.complete_step("10B", subtitle=f"videos={len(ctx.traffic_control_attribute_results)}")


def run_step_11_logic_atoms(ctx: PipelineContext, runner: StepRunner) -> None:
    logic_atoms_cfg = _get_logic_atoms_cfg()
    runner.announce_step("11", "logic_atoms_driving_mini")
    runner.log("11", f"cfg={logic_atoms_cfg}")
    with runner.module_output("11"):
        ctx.logic_atom_results = logic_atoms_driving_mini.run(
            segment_object_motion_results=ctx.traffic_control_attribute_results or ctx.important_object_results or [],
            cfg=logic_atoms_cfg,
            force_recompute=False,
        )
    runner.log("11", f"completed videos={len(ctx.logic_atom_results)}")
    runner.complete_step("11", subtitle=f"videos={len(ctx.logic_atom_results)}")


def run_step_12_target_head_atoms(ctx: PipelineContext, runner: StepRunner) -> None:
    target_head_cfg = _get_target_head_atoms_cfg()
    runner.announce_step("12", "target_head_atoms_driving_mini")
    runner.log("12", f"cfg={target_head_cfg}")
    with runner.module_output("12"):
        ctx.target_head_results = target_head_atoms_driving_mini.run(
            logic_atom_results=ctx.logic_atom_results or [],
            cfg=target_head_cfg,
            force_recompute=False,
        )
    runner.log("12", f"completed videos={len(ctx.target_head_results)}")
    runner.complete_step("12", subtitle=f"videos={len(ctx.target_head_results)}")


def run_step_13_temporal_rule_examples(ctx: PipelineContext, runner: StepRunner) -> None:
    rule_examples_cfg = _get_temporal_rule_examples_cfg()
    runner.announce_step("13", "temporal_rule_examples_driving_mini")
    runner.log("13", f"cfg={rule_examples_cfg}")
    with runner.module_output("13"):
        ctx.temporal_rule_results = temporal_rule_examples_driving_mini.run(
            target_head_results=ctx.target_head_results or [],
            cfg=rule_examples_cfg,
            force_recompute=False,
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
    runner.log(
        "15",
        "merged "
        f"rules={ctx.merged_candidate_rules['num_rules']} "
        f"videos={ctx.merged_candidate_rules['num_videos']}",
    )
    runner.complete_step("15", subtitle=f"rules={ctx.merged_candidate_rules['num_rules']}")


def run_step_16_extended_rules(ctx: PipelineContext, runner: StepRunner) -> None:
    extended_rules_cfg = _get_extended_rules_cfg()
    runner.announce_step("16", "extended_rules_driving_mini")
    runner.log("16", f"cfg={extended_rules_cfg}")
    runner.log("16", f"recompute={bool(ctx.recompute_cfg.get('extended_rules', True))}")
    with runner.module_output("16"):
        ctx.extended_rule_results = extended_rules_driving_mini.run(
            merged_initial_rules=ctx.merged_candidate_rules,
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
    runner.announce_step("17", "final_rules_driving_mini")
    runner.log("17", f"cfg={final_rules_cfg}")
    runner.log("17", f"recompute={bool(ctx.recompute_cfg.get('final_rules', True))}")
    with runner.module_output("17"):
        ctx.final_rule_results = final_rules_driving_mini.run(
            extended_rule_results=ctx.extended_rule_results,
            cfg=final_rules_cfg,
            force_recompute=bool(ctx.recompute_cfg.get("final_rules", True)),
        )
    runner.log("17", f"selected rules={ctx.final_rule_results.get('num_final_rules', 0)}")
    runner.complete_step("17", subtitle=f"rules={int(ctx.final_rule_results.get('num_final_rules', 0))}")


def run_step_17b_diverse_final_rules(ctx: PipelineContext, runner: StepRunner) -> None:
    diverse_final_rules_cfg = _get_diverse_final_rules_cfg()
    runner.announce_step("17B", "diverse_final_rules_driving_mini")
    runner.log("17B", f"cfg={diverse_final_rules_cfg}")
    runner.log("17B", f"recompute={bool(ctx.recompute_cfg.get('diverse_final_rules', True))}")
    with runner.module_output("17B"):
        ctx.diverse_final_rule_results = diverse_final_rules_driving_mini.run(
            extended_rule_results=ctx.extended_rule_results,
            cfg=diverse_final_rules_cfg,
            force_recompute=bool(ctx.recompute_cfg.get("diverse_final_rules", True)),
        )
    runner.log("17B", f"selected rules={ctx.diverse_final_rule_results.get('num_final_rules', 0)}")
    runner.complete_step("17B", subtitle=f"rules={int(ctx.diverse_final_rule_results.get('num_final_rules', 0))}")


def run_step_17b2_semantic_constrained(ctx: PipelineContext, runner: StepRunner) -> None:
    semantic_constrained_diverse_cfg = _get_semantic_constrained_diverse_cfg()
    runner.announce_step("17B2", "semantic_constrained_diverse_final_rules_driving_mini")
    runner.log("17B2", f"cfg={semantic_constrained_diverse_cfg}")
    runner.log(
        "17B2",
        f"recompute={bool(ctx.recompute_cfg.get('semantic_constrained_diverse_final_rules', True))}",
    )
    with runner.module_output("17B2"):
        ctx.semantic_constrained_diverse_rule_results = diverse_final_rules_driving_mini.run(
            extended_rule_results=ctx.extended_rule_results,
            cfg=semantic_constrained_diverse_cfg,
            output_root=_get_semantic_constrained_diverse_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("semantic_constrained_diverse_final_rules", True)),
        )
    runner.log(
        "17B2",
        f"selected rules={ctx.semantic_constrained_diverse_rule_results.get('num_final_rules', 0)}",
    )
    runner.complete_step(
        "17B2",
        subtitle=f"rules={int(ctx.semantic_constrained_diverse_rule_results.get('num_final_rules', 0))}",
    )


def run_step_17c_coverage_family_aware(ctx: PipelineContext, runner: StepRunner) -> None:
    coverage_family_aware_cfg = _get_coverage_family_aware_final_rules_cfg()
    runner.announce_step("17C", "coverage_family_aware_final_rules_driving_mini")
    runner.log("17C", f"cfg={coverage_family_aware_cfg}")
    runner.log(
        "17C",
        f"recompute={bool(ctx.recompute_cfg.get('coverage_family_aware_final_rules', True))}",
    )
    with runner.module_output("17C"):
        ctx.coverage_family_aware_rule_results = diverse_final_rules_driving_mini.run(
            extended_rule_results=ctx.extended_rule_results,
            cfg=coverage_family_aware_cfg,
            output_root=_get_coverage_family_aware_final_rules_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("coverage_family_aware_final_rules", True)),
        )
    runner.log("17C", f"selected rules={ctx.coverage_family_aware_rule_results.get('num_final_rules', 0)}")
    runner.complete_step(
        "17C",
        subtitle=f"rules={int(ctx.coverage_family_aware_rule_results.get('num_final_rules', 0))}",
    )


def run_step_17d_rule_pool_upper_bound(ctx: PipelineContext, runner: StepRunner) -> None:
    ctx.rule_results_by_name = {
        "original": ctx.final_rule_results,
        "diverse": ctx.diverse_final_rule_results,
        "semantic_constrained_diverse": ctx.semantic_constrained_diverse_rule_results,
        "coverage_family_aware": ctx.coverage_family_aware_rule_results,
    }
    rule_pool_upper_bound_cfg = _get_rule_pool_upper_bound_diagnostic_cfg()
    runner.announce_step("17D", "rule_pool_upper_bound_diagnostic_driving_mini")
    runner.log("17D", f"cfg={rule_pool_upper_bound_cfg}")
    runner.log("17D", f"recompute={bool(ctx.recompute_cfg.get('rule_pool_upper_bound_diagnostic', True))}")
    with runner.module_output("17D"):
        ctx.rule_pool_upper_bound_results = rule_pool_upper_bound_diagnostic_driving_mini.run(
            extended_rule_results=ctx.extended_rule_results,
            temporal_rule_results=ctx.eval_temporal_rule_results or [],
            eval_video_ids=list(ctx.split_manifest.get("eval_video_ids", [])),
            split_manifest=ctx.split_manifest,
            rule_results_by_name=ctx.rule_results_by_name,
            cfg=rule_pool_upper_bound_cfg,
            output_root=_get_rule_pool_upper_bound_diagnostic_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("rule_pool_upper_bound_diagnostic", True)),
        )
    runner.log("17D", f"bottleneck={ctx.rule_pool_upper_bound_results.get('bottleneck_label', 'unknown')}")
    runner.complete_step(
        "17D",
        subtitle=f"bottleneck={ctx.rule_pool_upper_bound_results.get('bottleneck_label', 'unknown')}",
    )


def run_step_17e_oracle_gap(ctx: PipelineContext, runner: StepRunner) -> None:
    oracle_rule_selection_gap_cfg = _get_oracle_rule_selection_gap_diagnostic_cfg()
    runner.announce_step("17E", "oracle_rule_selection_gap_diagnostic_driving_mini")
    runner.log("17E", f"cfg={oracle_rule_selection_gap_cfg}")
    runner.log("17E", f"recompute={bool(ctx.recompute_cfg.get('oracle_rule_selection_gap_diagnostic', True))}")
    with runner.module_output("17E"):
        ctx.oracle_rule_selection_gap_results = oracle_rule_selection_gap_diagnostic_driving_mini.run(
            extended_rule_results=ctx.extended_rule_results,
            rule_pool_upper_bound_results=ctx.rule_pool_upper_bound_results,
            rule_results_by_name=ctx.rule_results_by_name,
            cfg=oracle_rule_selection_gap_cfg,
            output_root=_get_oracle_rule_selection_gap_diagnostic_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("oracle_rule_selection_gap_diagnostic", True)),
        )
    runner.log(
        "17E",
        f"oracle_target_f1={float(ctx.oracle_rule_selection_gap_results.get('oracle_target_f1', 0.0)):.3f}",
    )
    runner.complete_step(
        "17E",
        subtitle=f"oracle_f1={float(ctx.oracle_rule_selection_gap_results.get('oracle_target_f1', 0.0)):.3f}",
    )


def run_step_18_rule_evaluation(ctx: PipelineContext, runner: StepRunner) -> None:
    rule_evaluation_cfg = _get_rule_evaluation_cfg()
    runner.announce_step("18", "evaluate_rules_driving_mini")
    runner.log("18", f"cfg={rule_evaluation_cfg}")
    ctx.rule_set_mode, ctx.primary_rule_set, ctx.evaluation_rule_sets = _resolve_rule_evaluation_plan(rule_evaluation_cfg)
    runner.log("18", f"rule_sets={ctx.evaluation_rule_sets} primary={ctx.primary_rule_set}")

    evaluation_output_root = _get_rule_evaluation_output_root()
    ctx.evaluation_results_by_name = {}
    for rule_set_name in ctx.evaluation_rule_sets:
        evaluation_cfg_for_run = dict(rule_evaluation_cfg)
        evaluation_cfg_for_run["evaluated_rule_set_name"] = rule_set_name
        output_root = (
            evaluation_output_root
            if rule_set_name == ctx.primary_rule_set
            else evaluation_output_root / rule_set_name
        )
        runner.log("18", f"evaluating rule_set={rule_set_name}")
        with runner.module_output("18"):
            evaluation_result = evaluate_rules_driving_mini.run(
                final_rule_results=ctx.rule_results_by_name[rule_set_name],
                temporal_rule_results=ctx.eval_temporal_rule_results or [],
                eval_video_ids=list(ctx.split_manifest.get("eval_video_ids", [])),
                split_manifest=ctx.split_manifest,
                cfg=evaluation_cfg_for_run,
                output_root=output_root,
                force_recompute=bool(ctx.recompute_cfg.get("rule_evaluation", True)),
            )
        evaluation_result["evaluated_rule_set_name"] = rule_set_name
        ctx.evaluation_results_by_name[rule_set_name] = evaluation_result

    if ctx.rule_set_mode in {"both", "all"}:
        comparison_paths = _write_rule_set_evaluation_comparison(
            evaluation_output_root=evaluation_output_root,
            primary_rule_set=ctx.primary_rule_set,
            rule_set_mode=ctx.rule_set_mode,
            evaluation_rule_sets=ctx.evaluation_rule_sets,
            rule_results_by_name=ctx.rule_results_by_name,
            evaluation_results_by_name=ctx.evaluation_results_by_name,
        )
        runner.log("18", f"comparison={comparison_paths['comparison_json_path']}")

    ctx.evaluation_results = ctx.evaluation_results_by_name[ctx.primary_rule_set]
    overall_metrics = dict(ctx.evaluation_results.get("overall_metrics", {}))
    runner.log(
        "18",
        f"rule_set={ctx.primary_rule_set} "
        f"precision={float(overall_metrics.get('precision', 0.0)):.3f} "
        f"recall={float(overall_metrics.get('recall', 0.0)):.3f} "
        f"f1={float(overall_metrics.get('f1', 0.0)):.3f}",
    )
    runner.complete_step("18", subtitle=f"f1={float(overall_metrics.get('f1', 0.0)):.3f}")


def run_step_18b_neural_symbolic_baseline(ctx: PipelineContext, runner: StepRunner) -> None:
    neural_symbolic_baseline_cfg = _get_neural_symbolic_baseline_cfg()
    runner.announce_step("18B", "neural_symbolic_baseline_driving_mini")
    runner.log("18B", f"cfg={neural_symbolic_baseline_cfg}")
    runner.log("18B", f"recompute={bool(ctx.recompute_cfg.get('neural_symbolic_baseline', True))}")
    with runner.module_output("18B"):
        ctx.neural_symbolic_baseline_results = neural_symbolic_baseline_driving_mini.run(
            train_temporal_rule_results=ctx.train_temporal_rule_results or [],
            eval_temporal_rule_results=ctx.eval_temporal_rule_results or [],
            split_manifest=ctx.split_manifest,
            cfg=neural_symbolic_baseline_cfg,
            output_root=_get_neural_symbolic_baseline_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("neural_symbolic_baseline", True)),
        )
    comparison_rows = list(ctx.neural_symbolic_baseline_results.get("comparison", []))
    if comparison_rows:
        comparison_parts = []
        for row in comparison_rows:
            threshold_name = str(row.get("threshold_name", "")).strip()
            threshold_suffix = f" [{threshold_name}]" if threshold_name else ""
            comparison_parts.append(
                f"{str(row.get('model_name', 'model'))}{threshold_suffix}: "
                f"F1={float(row.get('f1', 0.0)):.3f}, "
                f"AUROC={float(row.get('auroc', 0.0)):.3f}, "
                f"AUPRC={float(row.get('auprc', 0.0)):.3f}"
            )
        runner.log("18B", " | ".join(comparison_parts))
    runner.complete_step("18B", subtitle=f"models={len(comparison_rows)}")


def run_step_18c_rule_aggregation_baseline(ctx: PipelineContext, runner: StepRunner) -> None:
    rule_aggregation_baseline_cfg = _get_rule_aggregation_baseline_cfg()
    runner.announce_step("18C", "rule_aggregation_baseline_driving_mini")
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


def run_step_19_error_analysis(ctx: PipelineContext, runner: StepRunner) -> None:
    error_analysis_cfg = _get_error_and_explainability_cfg()
    runner.announce_step("19", "error_and_explainability_analysis_driving_mini")
    runner.log("19", f"cfg={error_analysis_cfg}")
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
        runner.log("19", f"analyzing rule_set={rule_set_name}")
        with runner.module_output("19"):
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
        runner.log("19", f"comparison={comparison_paths['comparison_json_path']}")

    ctx.error_analysis_results = ctx.error_analysis_results_by_name[ctx.primary_rule_set]
    runner.log(
        "19",
        f"rule_set={ctx.primary_rule_set} "
        f"fn={int(ctx.error_analysis_results.get('num_fn_examples', 0))} "
        f"fp={int(ctx.error_analysis_results.get('num_fp_examples', 0))}",
    )
    runner.complete_step(
        "19",
        subtitle=f"fn={int(ctx.error_analysis_results.get('num_fn_examples', 0))},fp={int(ctx.error_analysis_results.get('num_fp_examples', 0))}",
    )


def run_step_20_vehicle_rule_diagnostic(ctx: PipelineContext, runner: StepRunner) -> None:
    vehicle_rule_diagnostic_cfg = _get_vehicle_rule_diagnostic_cfg()
    vehicle_rule_diagnostic_cfg["primary_rule_set"] = ctx.primary_rule_set
    runner.announce_step("20", "vehicle_rule_diagnostic_driving_mini")
    runner.log("20", f"cfg={vehicle_rule_diagnostic_cfg}")
    runner.log("20", f"recompute={bool(ctx.recompute_cfg.get('vehicle_rule_diagnostic', True))}")
    with runner.module_output("20"):
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
    runner.log("20", f"diagnosis={ctx.vehicle_rule_diagnostic_results.get('primary_diagnosis', 'unknown')}")
    runner.complete_step(
        "20",
        subtitle=f"diagnosis={ctx.vehicle_rule_diagnostic_results.get('primary_diagnosis', 'unknown')}",
    )


def run_step_20b_fn_categorization(ctx: PipelineContext, runner: StepRunner) -> None:
    fn_categorization_cfg = _get_fn_categorization_diagnostic_cfg()
    runner.announce_step("20B", "fn_categorization_diagnostic_driving_mini")
    if bool(fn_categorization_cfg.get("enabled", False)):
        runner.log("20B", f"cfg={fn_categorization_cfg}")
        runner.log("20B", f"recompute={bool(ctx.recompute_cfg.get('fn_categorization_diagnostic', True))}")
        with runner.module_output("20B"):
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
        runner.log("20B", f"summary={ctx.fn_categorization_results.get('summary_path', '')}")
        runner.complete_step("20B", subtitle="enabled")
        return
    runner.log("20B", "disabled by config")
    runner.complete_step("20B", subtitle="skipped")


def run_step_21_rule_selection_visualization(ctx: PipelineContext, runner: StepRunner) -> None:
    rule_selection_visualization_cfg = _get_rule_selection_visualization_cfg()
    runner.announce_step("21", "rule_selection_visualization_driving_mini")
    runner.log("21", f"cfg={rule_selection_visualization_cfg}")
    runner.log("21", f"recompute={bool(ctx.recompute_cfg.get('rule_selection_visualization', True))}")
    with runner.module_output("21"):
        ctx.rule_selection_visualization_results = rule_selection_visualization_driving_mini.run(
            cfg=rule_selection_visualization_cfg,
            output_root=_get_rule_selection_visualization_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("rule_selection_visualization", True)),
        )
    runner.log("21", f"figures={len(ctx.rule_selection_visualization_results.get('figure_paths', {}))}")
    runner.complete_step("21", subtitle="figures=ready")


def run_step_22_integrated_visualization(ctx: PipelineContext, runner: StepRunner) -> None:
    integrated_method_visualization_cfg = _get_integrated_method_visualization_cfg()
    runner.announce_step("22", "integrated_method_visualization_driving_mini")
    runner.log("22", f"cfg={integrated_method_visualization_cfg}")
    runner.log("22", f"recompute={bool(ctx.recompute_cfg.get('integrated_method_visualization', True))}")
    with runner.module_output("22"):
        ctx.integrated_method_visualization_results = integrated_method_visualization_driving_mini.run(
            cfg=integrated_method_visualization_cfg,
            output_root=_get_integrated_method_visualization_output_root(),
            force_recompute=bool(ctx.recompute_cfg.get("integrated_method_visualization", True)),
        )
    runner.log("22", f"figures={len(ctx.integrated_method_visualization_results.get('figure_paths', {}))}")
    runner.complete_step("22", subtitle="figures=ready")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the driving_mini experiment pipeline up to a selected step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "max_step",
        nargs="?",
        type=_parse_max_step_arg,
        default="22",
        help="Run through this step number or exact step id such as 17E or 18C.",
    )
    parser.add_argument(
        "--video-id",
        dest="video_ids",
        action="append",
        help=(
            "Restrict the pipeline to one or more video IDs. "
            "If omitted, the pipeline processes all available videos."
        ),
    )
    return parser.parse_args()


def main(max_step: int | str = 22, video_ids: Optional[List[str]] = None) -> None:
    ctx = PipelineContext(effective_video_ids=_resolve_video_ids(video_ids))
    runner = StepRunner.create(max_step)
    try:
        # Perception
        run_step_1_detection(ctx, runner)
        run_step_2_tracking(ctx, runner)
        run_step_3_dataset_annotations(ctx, runner)
        run_step_4_merge_annotations(ctx, runner)
        run_step_5_prepare_3d_positions(ctx, runner)

        # Motion and symbolic scene abstraction
        run_step_6_ego_motion(ctx, runner)
        run_step_7_relative_object_motion(ctx, runner)
        run_step_8_temporal_segmentation(ctx, runner)
        run_step_9_segment_object_motion(ctx, runner)
        run_step_10_important_objects(ctx, runner)
        run_step_10b_traffic_control_attributes(ctx, runner)

        # Logic and rule mining
        run_step_11_logic_atoms(ctx, runner)
        run_step_12_target_head_atoms(ctx, runner)
        run_step_13_temporal_rule_examples(ctx, runner)
        _prepare_rule_learning_inputs(ctx)
        runner.log(
            "13",
            "split "
            f"train_videos={len(ctx.split_manifest.get('train_video_ids', []))} "
            f"eval_videos={len(ctx.split_manifest.get('eval_video_ids', []))}",
        )
        runner.log("13", f"recompute_cfg={ctx.recompute_cfg}")
        run_step_14_candidate_rules(ctx, runner)
        run_step_15_merge_initial_rules(ctx, runner)
        run_step_16_extended_rules(ctx, runner)

        # Rule selection
        run_step_17_final_rules(ctx, runner)
        run_step_17b_diverse_final_rules(ctx, runner)
        run_step_17b2_semantic_constrained(ctx, runner)
        run_step_17c_coverage_family_aware(ctx, runner)
        run_step_17d_rule_pool_upper_bound(ctx, runner)
        run_step_17e_oracle_gap(ctx, runner)

        # Evaluation and baselines
        run_step_18_rule_evaluation(ctx, runner)
        run_step_18b_neural_symbolic_baseline(ctx, runner)
        run_step_18c_rule_aggregation_baseline(ctx, runner)
        run_step_18d_object_to_atom_coverage(ctx, runner)
        run_step_18e_traffic_control_rule_utility(ctx, runner)
        run_step_18f_traffic_control_temporal_alignment(ctx, runner)

        # Diagnostics
        run_step_19_error_analysis(ctx, runner)
        run_step_20_vehicle_rule_diagnostic(ctx, runner)
        run_step_20b_fn_categorization(ctx, runner)

        # Visualization
        run_step_21_rule_selection_visualization(ctx, runner)
        run_step_22_integrated_visualization(ctx, runner)
    except _PipelineStopRequested:
        return


if __name__ == "__main__":
    args = parse_args()
    main(max_step=args.max_step, video_ids=args.video_ids)
