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

"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

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
from src.exp_driving_videos.modules import prepare_3d_positions_driving_mini
from src.exp_driving_videos.modules import tracking_driving_mini
from src.exp_driving_videos.modules import ego_motion_driving_mini
from src.exp_driving_videos.modules import important_objects_driving_mini
from src.exp_driving_videos.modules import logic_atoms_driving_mini
from src.exp_driving_videos.modules import relative_object_motion_driving_mini
from src.exp_driving_videos.modules import oracle_rule_selection_gap_diagnostic_driving_mini
from src.exp_driving_videos.modules import rule_pool_upper_bound_diagnostic_driving_mini
from src.exp_driving_videos.modules import rule_aggregation_baseline_driving_mini
from src.exp_driving_videos.modules import rule_selection_visualization_driving_mini
from src.exp_driving_videos.modules import segment_object_motion_driving_mini
from src.exp_driving_videos.modules import target_head_atoms_driving_mini
from src.exp_driving_videos.modules import temporal_rule_examples_driving_mini
from src.exp_driving_videos.modules import temporal_segmentation_driving_mini
from src.exp_driving_videos.modules import vehicle_rule_diagnostic_driving_mini

DRIVING_MINI_OD_MODEL = driving_pipeline_config.DRIVING_MINI_OD_MODEL
DEFAULT_TRAIN_VIDEO_COUNT = driving_pipeline_config.DEFAULT_TRAIN_VIDEO_COUNT
DEFAULT_EVAL_VIDEO_COUNT = driving_pipeline_config.DEFAULT_EVAL_VIDEO_COUNT
DRIVING_MINI_OD_CLASSES = driving_pipeline_config.DRIVING_MINI_OD_CLASSES

_get_ego_motion_smoothing_window = driving_pipeline_config.get_ego_motion_smoothing_window
_get_ego_static_adjustment_cfg = driving_pipeline_config.get_ego_static_adjustment_cfg
_get_temporal_segmentation_cfg = driving_pipeline_config.get_temporal_segmentation_cfg
_get_segment_object_motion_cfg = driving_pipeline_config.get_segment_object_motion_cfg
_get_important_objects_cfg = driving_pipeline_config.get_important_objects_cfg
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
_get_neural_symbolic_baseline_cfg = driving_pipeline_config.get_neural_symbolic_baseline_cfg
_get_rule_selection_visualization_cfg = driving_pipeline_config.get_rule_selection_visualization_cfg
_get_fn_categorization_diagnostic_cfg = driving_pipeline_config.get_fn_categorization_diagnostic_cfg
_get_pipeline_recompute_cfg = driving_pipeline_config.get_pipeline_recompute_cfg
_get_error_and_explainability_cfg = driving_pipeline_config.get_error_and_explainability_cfg
_get_vehicle_rule_diagnostic_cfg = driving_pipeline_config.get_vehicle_rule_diagnostic_cfg
_get_rule_evaluation_output_root = driving_pipeline_config.get_rule_evaluation_output_root
_get_rule_aggregation_baseline_output_root = driving_pipeline_config.get_rule_aggregation_baseline_output_root
_get_neural_symbolic_baseline_output_root = driving_pipeline_config.get_neural_symbolic_baseline_output_root
_get_error_and_explainability_output_root = driving_pipeline_config.get_error_and_explainability_output_root
_get_coverage_family_aware_final_rules_output_root = driving_pipeline_config.get_coverage_family_aware_final_rules_output_root
_get_semantic_constrained_diverse_output_root = driving_pipeline_config.get_semantic_constrained_diverse_output_root
_get_rule_pool_upper_bound_diagnostic_output_root = driving_pipeline_config.get_rule_pool_upper_bound_diagnostic_output_root
_get_oracle_rule_selection_gap_diagnostic_output_root = driving_pipeline_config.get_oracle_rule_selection_gap_diagnostic_output_root
_get_vehicle_rule_diagnostic_output_root = driving_pipeline_config.get_vehicle_rule_diagnostic_output_root
_get_rule_selection_visualization_output_root = driving_pipeline_config.get_rule_selection_visualization_output_root
_get_fn_categorization_diagnostic_output_root = driving_pipeline_config.get_fn_categorization_diagnostic_output_root

_merge_candidate_rules = driving_pipeline_data.merge_candidate_rules
_select_video_results = driving_pipeline_data.select_video_results
_build_train_eval_split = driving_pipeline_data.build_train_eval_split
_resolve_video_ids = driving_pipeline_data.resolve_video_ids


def _run_object_detection_step(
    force_recompute: bool = False,
    video_ids: List[str] | None = None,
) -> List[Dict[str, Any]]:
    print("=== Step 1: detect_driving_mini ===")
    print(f"OD model           : {DRIVING_MINI_OD_MODEL}")
    print(f"OD classes         : {DRIVING_MINI_OD_CLASSES}")
    print(f"OD force_recompute : {force_recompute}")
    if video_ids:
        print(f"OD video_ids       : {video_ids}")
    detection_results: List[Dict[str, Any]] = detect_driving_mini.run(
        video_ids=video_ids,
        model_name=DRIVING_MINI_OD_MODEL,
        classes=DRIVING_MINI_OD_CLASSES,
        force_recompute=force_recompute,
    )
    print(f"Detection complete. Processed {len(detection_results)} video(s).")
    print("*** Vis Path: ", Path(config.get_output_path("pipeline_output")) / "01_driving_mini_detection")
    return detection_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the driving_mini experiment pipeline up to a selected step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "max_step",
        nargs="?",
        type=int,
        default=21,
        choices=range(1, 22),
        help="Run the pipeline through this step number.",
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


def main(max_step: int = 21, video_ids: List[str] | None = None) -> None:
    effective_video_ids = _resolve_video_ids(video_ids)
    if effective_video_ids:
        print(f"Video filter: {effective_video_ids}")

    # Step 1: object detection over driving_mini frames
    detection_results = _run_object_detection_step(
        force_recompute=False,
        video_ids=effective_video_ids,
    )
    if max_step == 1:
        print("\nStopping after step 1 by request.")
        return

    # Step 2: multi-object tracking with ByteTrack
    print("\n=== Step 2: tracking_driving_mini ===")
    tracking_results: List[Dict[str, Any]] = tracking_driving_mini.run(detection_results)
    print(f"Tracking complete. Processed {len(tracking_results)} video(s).")
    if max_step == 2:
        print("\nStopping after step 2 by request.")
        return

    # Step 3: import dataset-provided object annotations/tracks (ground truth)
    print("\n=== Step 3: dataset_annotations_driving_mini ===")
    video_ids = [r["video_id"] for r in tracking_results]
    dataset_annotation_results: List[Dict[str, Any]] = dataset_annotations_driving_mini.run(
        video_ids=video_ids
    )
    print(
        "Dataset annotation import complete. "
        f"Processed {len(dataset_annotation_results)} video(s)."
    )
    if max_step == 3:
        print("\nStopping after step 3 by request.")
        return

    # Step 4: merge GT annotations with detected/tracked results
    print("\n=== Step 4: merge_gt_and_detected_driving_mini ===")
    merged_results: List[Dict[str, Any]] = merge_gt_and_detected_driving_mini.run(
        tracking_results=tracking_results,
        dataset_annotation_results=dataset_annotation_results,
    )
    print(
        "Merge complete. "
        f"Processed {len(merged_results)} video(s)."
    )
    if max_step == 4:
        print("\nStopping after step 4 by request.")
        return

    # Step 5: prepare per-object 3D positions using depth maps
    print("\n=== Step 5: prepare_3d_positions_driving_mini ===")
    positions_3d_results: List[Dict[str, Any]] = prepare_3d_positions_driving_mini.run(
        merged_results=merged_results,
    )
    print(
        "3D position preparation complete. "
        f"Processed {len(positions_3d_results)} video(s)."
    )
    if max_step == 5:
        print("\nStopping after step 5 by request.")
        return

    # Step 6: ego motion estimation from optical flow + background mask + depth
    smoothing_window = _get_ego_motion_smoothing_window(default=5)
    static_adjust_cfg = _get_ego_static_adjustment_cfg()
    print("\n=== Step 6: ego_motion_driving_mini ===")
    print(f"Using ego motion smoothing_window={smoothing_window}")
    print(f"Using ego static adjustment cfg={static_adjust_cfg}")
    ego_motion_results: List[Dict[str, Any]] = ego_motion_driving_mini.run(
        merged_results=merged_results,
        force_recompute=False,
        smoothing_window=smoothing_window,
        static_adjust_cfg=static_adjust_cfg,
    )
    print(
        "Ego motion estimation complete. "
        f"Processed {len(ego_motion_results)} video(s)."
    )
    if max_step == 6:
        print("\nStopping after step 6 by request.")
        return

    # Step 7: object motion relative to ego motion in camera 3D frame
    print("\n=== Step 7: relative_object_motion_driving_mini ===")
    relative_motion_results: List[Dict[str, Any]] = relative_object_motion_driving_mini.run(
        positions_3d_results=positions_3d_results,
        ego_motion_results=ego_motion_results,
        force_recompute=False,
    )
    print(
        "Relative object motion estimation complete. "
        f"Processed {len(relative_motion_results)} video(s)."
    )
    if max_step == 7:
        print("\nStopping after step 7 by request.")
        return

    # Step 8: temporal segmentation of ego-motion signals to event spans/cut points
    temporal_seg_cfg = _get_temporal_segmentation_cfg()
    print("\n=== Step 8: temporal_segmentation_driving_mini ===")
    print(f"Temporal segmentation cfg: {temporal_seg_cfg}")
    temporal_seg_results: List[Dict[str, Any]] = temporal_segmentation_driving_mini.run(
        ego_motion_results=ego_motion_results,
        relative_motion_results=relative_motion_results,
        seg_cfg=temporal_seg_cfg,
        force_recompute=False,
    )
    print(
        "Temporal segmentation complete. "
        f"Processed {len(temporal_seg_results)} video(s)."
    )
    if max_step == 8:
        print("\nStopping after step 8 by request.")
        return

    # Step 9: summarize object relative motion per merged temporal segment
    segment_object_cfg = _get_segment_object_motion_cfg()
    print("\n=== Step 9: segment_object_motion_driving_mini ===")
    print(f"Segment object motion cfg: {segment_object_cfg}")
    segment_object_results: List[Dict[str, Any]] = segment_object_motion_driving_mini.run(
        relative_motion_results=relative_motion_results,
        temporal_segmentation_results=temporal_seg_results,
        cfg=segment_object_cfg,
        force_recompute=False,
    )
    print(
        "Segment object motion summary complete. "
        f"Processed {len(segment_object_results)} video(s)."
    )
    if max_step == 9:
        print("\nStopping after step 9 by request.")
        return

    # Step 10: analyze/select important objects per merged temporal segment
    important_objects_cfg = _get_important_objects_cfg()
    print("\n=== Step 10: important_objects_driving_mini ===")
    print(f"Important objects cfg: {important_objects_cfg}")
    important_object_results: List[Dict[str, Any]] = important_objects_driving_mini.run(
        segment_object_motion_results=segment_object_results,
        cfg=important_objects_cfg,
        force_recompute=False,
    )
    print(
        "Important object analysis complete. "
        f"Processed {len(important_object_results)} video(s)."
    )
    if max_step == 10:
        print("\nStopping after step 10 by request.")
        return

    # Step 11: convert filtered symbolic segment facts into logic atoms
    logic_atoms_cfg = _get_logic_atoms_cfg()
    print("\n=== Step 11: logic_atoms_driving_mini ===")
    print(f"Logic atoms cfg: {logic_atoms_cfg}")
    logic_atom_results: List[Dict[str, Any]] = logic_atoms_driving_mini.run(
        segment_object_motion_results=important_object_results,
        cfg=logic_atoms_cfg,
        force_recompute=False,
    )
    print(
        "Logic atom conversion complete. "
        f"Processed {len(logic_atom_results)} video(s)."
    )
    if max_step == 11:
        print("\nStopping after step 11 by request.")
        return

    # Step 12: derive temporal target/head atoms for rule learning
    target_head_cfg = _get_target_head_atoms_cfg()
    print("\n=== Step 12: target_head_atoms_driving_mini ===")
    print(f"Target head atoms cfg: {target_head_cfg}")
    target_head_results: List[Dict[str, Any]] = target_head_atoms_driving_mini.run(
        logic_atom_results=logic_atom_results,
        cfg=target_head_cfg,
        force_recompute=False,
    )
    print(
        "Target head atom derivation complete. "
        f"Processed {len(target_head_results)} video(s)."
    )
    if max_step == 12:
        print("\nStopping after step 12 by request.")
        return

    # Step 13: build final temporal rule-learning examples
    rule_examples_cfg = _get_temporal_rule_examples_cfg()
    print("\n=== Step 13: temporal_rule_examples_driving_mini ===")
    print(f"Temporal rule examples cfg: {rule_examples_cfg}")
    temporal_rule_results: List[Dict[str, Any]] = temporal_rule_examples_driving_mini.run(
        target_head_results=target_head_results,
        cfg=rule_examples_cfg,
        force_recompute=False,
    )
    print(
        "Temporal rule-learning example build complete. "
        f"Processed {len(temporal_rule_results)} video(s)."
    )
    if max_step == 13:
        print("\nStopping after step 13 by request.")
        return

    split_cfg = _get_data_split_cfg()
    split_manifest = _build_train_eval_split(
        video_ids=[str(result.get("video_id", "")) for result in temporal_rule_results],
        train_video_count=int(split_cfg.get("train_video_count", DEFAULT_TRAIN_VIDEO_COUNT)),
        eval_video_count=int(split_cfg.get("eval_video_count", DEFAULT_EVAL_VIDEO_COUNT)),
        strategy=str(split_cfg.get("strategy", "eval_fraction")),
        eval_fraction=float(split_cfg.get("eval_fraction", 0.2)),
    )
    train_temporal_rule_results = _select_video_results(
        temporal_rule_results,
        selected_video_ids=list(split_manifest.get("train_video_ids", [])),
    )
    eval_temporal_rule_results = _select_video_results(
        temporal_rule_results,
        selected_video_ids=list(split_manifest.get("eval_video_ids", [])),
    )
    if not train_temporal_rule_results:
        raise RuntimeError("Train split is empty; cannot continue with rule learning.")
    recompute_cfg = _get_pipeline_recompute_cfg()
    print(f"Pipeline recompute cfg: {recompute_cfg}")

    # Step 14: generate short unary-body initial temporal rules
    candidate_rules_cfg = _get_candidate_rules_cfg()
    print("\n=== Step 14: candidate_rules_driving_mini ===")
    print(f"Initial rules cfg: {candidate_rules_cfg}")
    candidate_rule_results: List[Dict[str, Any]] = candidate_rules_driving_mini.run(
        temporal_rule_results=train_temporal_rule_results,
        cfg=candidate_rules_cfg,
        force_recompute=bool(recompute_cfg.get("candidate_rules", False)),
    )
    print(
        "Initial rule generation complete. "
        f"Processed {len(candidate_rule_results)} video(s)."
    )
    if max_step == 14:
        print("\nStopping after step 14 by request.")
        return

    # Step 15: Merge all the initial rules into a single list for downstream processing
    print("\n=== Step 15: merge_initial_rules ===")
    merged_candidate_rules = _merge_candidate_rules(candidate_rule_results)
    print(
        "Initial rule merge complete. "
        f"Merged {merged_candidate_rules['num_rules']} rule(s) from "
        f"{merged_candidate_rules['num_videos']} video(s)."
    )
    if max_step == 15:
        print("\nStopping after step 15 by request.")
        return
    
    # Step 16: iteratively extend merged initial rules for N rounds
    extended_rules_cfg = _get_extended_rules_cfg()
    print("\n=== Step 16: extended_rules_driving_mini ===")
    print(f"Extended rules cfg: {extended_rules_cfg}")
    extended_rule_results: Dict[str, Any] = extended_rules_driving_mini.run(
        merged_initial_rules=merged_candidate_rules,
        cfg=extended_rules_cfg,
        force_recompute=bool(recompute_cfg.get("extended_rules", True)),
    )
    print(
        "Extended rule generation complete. "
        f"Completed {extended_rule_results.get('num_rounds_completed', 0)} round(s)."
    )
    if max_step == 16:
        print("\nStopping after step 16 by request.")
        return

    # Step 17: rank all kept rules by confidence and keep top-k
    final_rules_cfg = _get_final_rules_cfg()
    print("\n=== Step 17: final_rules_driving_mini ===")
    print(f"Final rules cfg: {final_rules_cfg}")
    final_rule_results: Dict[str, Any] = final_rules_driving_mini.run(
        extended_rule_results=extended_rule_results,
        cfg=final_rules_cfg,
        force_recompute=bool(recompute_cfg.get("final_rules", True)),
    )
    print(
        "Final rule selection complete. "
        f"Selected {final_rule_results.get('num_final_rules', 0)} rule(s)."
    )
    if max_step == 17:
        print("\nStopping after step 17 by request.")
        return

    # Step 17B: greedy diverse post-mining rule selection
    diverse_final_rules_cfg = _get_diverse_final_rules_cfg()
    print("\n=== Step 17B: diverse_final_rules_driving_mini ===")
    print(f"Diverse final rules cfg: {diverse_final_rules_cfg}")
    diverse_final_rule_results: Dict[str, Any] = diverse_final_rules_driving_mini.run(
        extended_rule_results=extended_rule_results,
        cfg=diverse_final_rules_cfg,
        force_recompute=bool(recompute_cfg.get("diverse_final_rules", True)),
    )
    print(
        "Diverse final rule selection complete. "
        f"Selected {diverse_final_rule_results.get('num_final_rules', 0)} rule(s)."
    )

    # Step 17B2: semantic constrained diverse rule selection
    semantic_constrained_diverse_cfg = _get_semantic_constrained_diverse_cfg()
    print("\n=== Step 17B2: semantic_constrained_diverse_final_rules_driving_mini ===")
    print(f"Semantic constrained diverse cfg: {semantic_constrained_diverse_cfg}")
    semantic_constrained_diverse_rule_results: Dict[str, Any] = diverse_final_rules_driving_mini.run(
        extended_rule_results=extended_rule_results,
        cfg=semantic_constrained_diverse_cfg,
        output_root=_get_semantic_constrained_diverse_output_root(),
        force_recompute=bool(recompute_cfg.get("semantic_constrained_diverse_final_rules", True)),
    )
    print(
        "Semantic constrained diverse rule selection complete. "
        f"Selected {semantic_constrained_diverse_rule_results.get('num_final_rules', 0)} rule(s)."
    )

    # Step 17C: coverage-aware / family-aware post-mining rule selection
    coverage_family_aware_cfg = _get_coverage_family_aware_final_rules_cfg()
    print("\n=== Step 17C: coverage_family_aware_final_rules_driving_mini ===")
    print(f"Coverage-aware final rules cfg: {coverage_family_aware_cfg}")
    coverage_family_aware_rule_results: Dict[str, Any] = diverse_final_rules_driving_mini.run(
        extended_rule_results=extended_rule_results,
        cfg=coverage_family_aware_cfg,
        output_root=_get_coverage_family_aware_final_rules_output_root(),
        force_recompute=bool(recompute_cfg.get("coverage_family_aware_final_rules", True)),
    )
    print(
        "Coverage-aware final rule selection complete. "
        f"Selected {coverage_family_aware_rule_results.get('num_final_rules', 0)} rule(s)."
    )

    # Step 17D: diagnose held-out upper bounds of the mined rule pool
    rule_results_by_name: Dict[str, Dict[str, Any]] = {
        "original": final_rule_results,
        "diverse": diverse_final_rule_results,
        "semantic_constrained_diverse": semantic_constrained_diverse_rule_results,
        "coverage_family_aware": coverage_family_aware_rule_results,
    }
    rule_pool_upper_bound_cfg = _get_rule_pool_upper_bound_diagnostic_cfg()
    print("\n=== Step 17D: rule_pool_upper_bound_diagnostic_driving_mini ===")
    print(f"Rule-pool upper-bound cfg: {rule_pool_upper_bound_cfg}")
    rule_pool_upper_bound_results: Dict[str, Any] = rule_pool_upper_bound_diagnostic_driving_mini.run(
        extended_rule_results=extended_rule_results,
        temporal_rule_results=eval_temporal_rule_results,
        eval_video_ids=list(split_manifest.get("eval_video_ids", [])),
        split_manifest=split_manifest,
        rule_results_by_name=rule_results_by_name,
        cfg=rule_pool_upper_bound_cfg,
        output_root=_get_rule_pool_upper_bound_diagnostic_output_root(),
        force_recompute=bool(recompute_cfg.get("rule_pool_upper_bound_diagnostic", True)),
    )
    print(
        "Rule-pool upper-bound diagnostic complete. "
        f"bottleneck={rule_pool_upper_bound_results.get('bottleneck_label', 'unknown')}"
    )

    # Step 17E: compare oracle top-K rules against actual selector outputs
    oracle_rule_selection_gap_cfg = _get_oracle_rule_selection_gap_diagnostic_cfg()
    print("\n=== Step 17E: oracle_rule_selection_gap_diagnostic_driving_mini ===")
    print(f"Oracle rule-selection gap cfg: {oracle_rule_selection_gap_cfg}")
    oracle_rule_selection_gap_results: Dict[str, Any] = (
        oracle_rule_selection_gap_diagnostic_driving_mini.run(
            extended_rule_results=extended_rule_results,
            rule_pool_upper_bound_results=rule_pool_upper_bound_results,
            rule_results_by_name=rule_results_by_name,
            cfg=oracle_rule_selection_gap_cfg,
            output_root=_get_oracle_rule_selection_gap_diagnostic_output_root(),
            force_recompute=bool(recompute_cfg.get("oracle_rule_selection_gap_diagnostic", True)),
        )
    )
    print(
        "Oracle rule-selection gap diagnostic complete. "
        f"oracle_target_f1={float(oracle_rule_selection_gap_results.get('oracle_target_f1', 0.0)):.3f}"
    )

    # Step 18: evaluate final rules on held-out evaluation videos
    rule_evaluation_cfg = _get_rule_evaluation_cfg()
    print("\n=== Step 18: evaluate_rules_driving_mini ===")
    print(f"Rule evaluation cfg: {rule_evaluation_cfg}")
    rule_set_mode = str(rule_evaluation_cfg.get("rule_set_mode", "all"))
    primary_rule_set = str(rule_evaluation_cfg.get("primary_rule_set", "original"))
    if rule_set_mode not in {"original", "diverse", "semantic_constrained_diverse", "coverage_family_aware", "both", "all"}:
        raise ValueError(f"Unsupported rule_evaluation.rule_set_mode: {rule_set_mode}")
    if primary_rule_set not in {"original", "diverse", "semantic_constrained_diverse", "coverage_family_aware"}:
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

    evaluation_output_root = _get_rule_evaluation_output_root()
    evaluation_results_by_name: Dict[str, Dict[str, Any]] = {}
    for rule_set_name in evaluation_rule_sets:
        evaluation_cfg_for_run = dict(rule_evaluation_cfg)
        evaluation_cfg_for_run["evaluated_rule_set_name"] = rule_set_name
        output_root = (
            evaluation_output_root
            if rule_set_name == primary_rule_set
            else evaluation_output_root / rule_set_name
        )
        evaluation_result = evaluate_rules_driving_mini.run(
            final_rule_results=rule_results_by_name[rule_set_name],
            temporal_rule_results=eval_temporal_rule_results,
            eval_video_ids=list(split_manifest.get("eval_video_ids", [])),
            split_manifest=split_manifest,
            cfg=evaluation_cfg_for_run,
            output_root=output_root,
            force_recompute=bool(recompute_cfg.get("rule_evaluation", True)),
        )
        evaluation_result["evaluated_rule_set_name"] = rule_set_name
        evaluation_results_by_name[rule_set_name] = evaluation_result

    if rule_set_mode in {"both", "all"}:
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
        print(f"Rule-set evaluation comparison JSON written to {comparison_json_path}")
        print(f"Rule-set evaluation comparison CSV written to {comparison_csv_path}")

    evaluation_results = evaluation_results_by_name[primary_rule_set]
    overall_metrics = dict(evaluation_results.get("overall_metrics", {}))
    print(
        "Held-out evaluation complete. "
        f"rule_set={primary_rule_set} | "
        f"Precision={float(overall_metrics.get('precision', 0.0)):.3f} | "
        f"Recall={float(overall_metrics.get('recall', 0.0)):.3f} | "
        f"F1={float(overall_metrics.get('f1', 0.0)):.3f}"
    )

    # Step 18B: symbolic neural baselines on the same train/eval split
    neural_symbolic_baseline_cfg = _get_neural_symbolic_baseline_cfg()
    print("\n=== Step 18B: neural_symbolic_baseline_driving_mini ===")
    print(f"Neural symbolic baseline cfg: {neural_symbolic_baseline_cfg}")
    neural_symbolic_baseline_results: Dict[str, Any] = neural_symbolic_baseline_driving_mini.run(
        train_temporal_rule_results=train_temporal_rule_results,
        eval_temporal_rule_results=eval_temporal_rule_results,
        split_manifest=split_manifest,
        cfg=neural_symbolic_baseline_cfg,
        output_root=_get_neural_symbolic_baseline_output_root(),
        force_recompute=bool(recompute_cfg.get("neural_symbolic_baseline", True)),
    )
    comparison_rows = list(neural_symbolic_baseline_results.get("comparison", []))
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
        print("Neural symbolic baselines complete. " + " | ".join(comparison_parts))

    # Step 18C: learned sparse logistic aggregation over Step 16 rule firings
    rule_aggregation_baseline_cfg = _get_rule_aggregation_baseline_cfg()
    print("\n=== Step 18C: rule_aggregation_baseline_driving_mini ===")
    print(f"Rule aggregation baseline cfg: {rule_aggregation_baseline_cfg}")
    rule_aggregation_baseline_results: Dict[str, Any] = rule_aggregation_baseline_driving_mini.run(
        extended_rule_results=extended_rule_results,
        train_temporal_rule_results=train_temporal_rule_results,
        eval_temporal_rule_results=eval_temporal_rule_results,
        split_manifest=split_manifest,
        cfg=rule_aggregation_baseline_cfg,
        output_root=_get_rule_aggregation_baseline_output_root(),
        force_recompute=bool(recompute_cfg.get("rule_aggregation_baseline", True)),
    )
    rule_aggregation_eval_best = dict(
        rule_aggregation_baseline_results.get("metrics_by_split", {})
        .get("eval", {})
        .get("best_validation_threshold", {})
    )
    print(
        "Rule aggregation baseline complete. "
        f"F1={float(rule_aggregation_eval_best.get('f1', 0.0)):.3f} | "
        f"AUROC={float(rule_aggregation_eval_best.get('auroc', 0.0)):.3f} | "
        f"AUPRC={float(rule_aggregation_eval_best.get('auprc', 0.0)):.3f} | "
        f"nonzero_rules={int(rule_aggregation_baseline_results.get('num_nonzero_rules', 0))}"
    )
    if max_step == 18:
        print("\nStopping after step 18 by request.")
        return

    # Step 19: error analysis + explainability diagnostics on held-out examples
    error_analysis_cfg = _get_error_and_explainability_cfg()
    print("\n=== Step 19: error_and_explainability_analysis_driving_mini ===")
    print(f"Error analysis cfg: {error_analysis_cfg}")
    error_analysis_output_root = _get_error_and_explainability_output_root()
    error_analysis_results_by_name: Dict[str, Dict[str, Any]] = {}
    for rule_set_name in evaluation_rule_sets:
        error_cfg_for_run = dict(error_analysis_cfg)
        error_cfg_for_run["rule_set_name"] = rule_set_name
        output_root = (
            error_analysis_output_root
            if rule_set_name == primary_rule_set
            else error_analysis_output_root / rule_set_name
        )
        error_analysis_result = error_and_explainability_analysis_driving_mini.run(
            final_rule_results=rule_results_by_name[rule_set_name],
            temporal_rule_results=eval_temporal_rule_results,
            evaluation_results=evaluation_results_by_name[rule_set_name],
            cfg=error_cfg_for_run,
            output_root=output_root,
            force_recompute=bool(recompute_cfg.get("error_and_explainability_analysis", True)),
        )
        error_analysis_results_by_name[rule_set_name] = error_analysis_result

    if rule_set_mode in {"both", "all"}:
        comparison_rows = []
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
        print(f"Rule-set error comparison JSON written to {comparison_json_path}")
        print(f"Rule-set error comparison CSV written to {comparison_csv_path}")

    error_analysis_results = error_analysis_results_by_name[primary_rule_set]
    print(
        "Held-out error analysis complete. "
        f"rule_set={primary_rule_set} | "
        f"FN={int(error_analysis_results.get('num_fn_examples', 0))} | "
        f"FP={int(error_analysis_results.get('num_fp_examples', 0))}"
    )
    if max_step == 19:
        print("\nStopping after step 19 by request.")
        return

    # Step 20: diagnose vehicle-centered rule coverage through the rule pool
    vehicle_rule_diagnostic_cfg = _get_vehicle_rule_diagnostic_cfg()
    vehicle_rule_diagnostic_cfg["primary_rule_set"] = primary_rule_set
    print("\n=== Step 20: vehicle_rule_diagnostic_driving_mini ===")
    print(f"Vehicle rule diagnostic cfg: {vehicle_rule_diagnostic_cfg}")
    vehicle_rule_diagnostic_results: Dict[str, Any] = vehicle_rule_diagnostic_driving_mini.run(
        merged_initial_rules=merged_candidate_rules,
        extended_rule_results=extended_rule_results,
        original_final_rule_results=final_rule_results,
        diverse_final_rule_results=diverse_final_rule_results,
        semantic_constrained_diverse_final_rule_results=semantic_constrained_diverse_rule_results,
        coverage_family_aware_final_rule_results=coverage_family_aware_rule_results,
        eval_temporal_rule_results=eval_temporal_rule_results,
        evaluation_results_by_name=evaluation_results_by_name,
        cfg=vehicle_rule_diagnostic_cfg,
        output_root=_get_vehicle_rule_diagnostic_output_root(),
        force_recompute=bool(recompute_cfg.get("vehicle_rule_diagnostic", True)),
    )
    print(
        "Vehicle-centered rule diagnostic complete. "
        f"diagnosis={vehicle_rule_diagnostic_results.get('primary_diagnosis', 'unknown')}"
    )
    if max_step == 20:
        print("\nStopping after step 20 by request.")
        return

    # Step 20B: categorize selector false negatives using rule-pool and context diagnostics
    fn_categorization_cfg = _get_fn_categorization_diagnostic_cfg()
    if bool(fn_categorization_cfg.get("enabled", False)):
        print("\n=== Step 20B: fn_categorization_diagnostic_driving_mini ===")
        print(f"FN categorization cfg: {fn_categorization_cfg}")
        fn_categorization_results: Dict[str, Any] = fn_categorization_diagnostic_driving_mini.run(
            extended_rule_results=extended_rule_results,
            rule_results_by_name=rule_results_by_name,
            evaluation_results_by_name=evaluation_results_by_name,
            error_analysis_results_by_name=error_analysis_results_by_name,
            temporal_rule_results=eval_temporal_rule_results,
            vehicle_rule_diagnostic_results=vehicle_rule_diagnostic_results,
            cfg=fn_categorization_cfg,
            output_root=_get_fn_categorization_diagnostic_output_root(),
            force_recompute=bool(recompute_cfg.get("fn_categorization_diagnostic", True)),
        )
        print(
            "FN categorization diagnostic complete. "
            f"summary={fn_categorization_results.get('summary_path', '')}"
        )
    else:
        print("\n=== Step 20B: fn_categorization_diagnostic_driving_mini ===")
        print("FN categorization diagnostic disabled by config. Skipping Step 20B.")

    # Step 21: generate rule-selection visualization figures from summary artifacts
    rule_selection_visualization_cfg = _get_rule_selection_visualization_cfg()
    print("\n=== Step 21: rule_selection_visualization_driving_mini ===")
    print(f"Rule selection visualization cfg: {rule_selection_visualization_cfg}")
    rule_selection_visualization_results: Dict[str, Any] = rule_selection_visualization_driving_mini.run(
        cfg=rule_selection_visualization_cfg,
        output_root=_get_rule_selection_visualization_output_root(),
        force_recompute=bool(recompute_cfg.get("rule_selection_visualization", True)),
    )
    print(
        "Rule selection visualization complete. "
        f"manifest={rule_selection_visualization_results.get('figure_paths', {})}"
    )
    if max_step == 21:
        print("\nStopping after step 21 by request.")
        return

if __name__ == "__main__":
    args = parse_args()
    main(max_step=args.max_step, video_ids=args.video_ids)
