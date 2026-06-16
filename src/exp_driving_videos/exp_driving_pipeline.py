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
    18. evaluate_rules_driving_mini — evaluate the learned final rules on the
                    held-out evaluation split.
    19. error_and_explainability_analysis_driving_mini — summarize false
                    negatives / false positives and generate explainability-
                    oriented diagnostics for held-out evaluation examples.

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import config
from src.exp_driving_videos.modules import detect_driving_mini
from src.exp_driving_videos.modules import candidate_rules_driving_mini
from src.exp_driving_videos.modules import dataset_annotations_driving_mini
from src.exp_driving_videos.modules import evaluate_rules_driving_mini
from src.exp_driving_videos.modules import error_and_explainability_analysis_driving_mini
from src.exp_driving_videos.modules import extended_rules_driving_mini
from src.exp_driving_videos.modules import final_rules_driving_mini
from src.exp_driving_videos.modules import merge_gt_and_detected_driving_mini
from src.exp_driving_videos.modules import prepare_3d_positions_driving_mini
from src.exp_driving_videos.modules import tracking_driving_mini
from src.exp_driving_videos.modules import ego_motion_driving_mini
from src.exp_driving_videos.modules import important_objects_driving_mini
from src.exp_driving_videos.modules import logic_atoms_driving_mini
from src.exp_driving_videos.modules import relative_object_motion_driving_mini
from src.exp_driving_videos.modules import segment_object_motion_driving_mini
from src.exp_driving_videos.modules import target_head_atoms_driving_mini
from src.exp_driving_videos.modules import temporal_rule_examples_driving_mini
from src.exp_driving_videos.modules import temporal_segmentation_driving_mini
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


def _get_ego_motion_smoothing_window(default: int = 5) -> int:
    """Load ego motion smoothing window from configs/exp_driving/default.yaml."""
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        return int(cfg.get("ego_motion", {}).get("smoothing_window", default))
    except Exception as exc:
        print(f"[warn] Could not load exp_driving config: {exc}. Using default={default}.")
        return default


def _get_ego_static_adjustment_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "enabled": True,
        "static_object_keywords": ["building", "traffic light"],
        "blend_weight": 0.7,
        "min_static_pixels": 300,
    }
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        override = cfg.get("ego_motion", {}).get("static_adjustment", {})
        if isinstance(override, dict):
            defaults.update(override)
    except Exception as exc:
        print(f"[warn] Could not load ego static adjustment config: {exc}. Using defaults.")
    return defaults


def _get_temporal_segmentation_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
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
    }
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        override = cfg.get("temporal_segmentation", {})
        if isinstance(override, dict):
            defaults.update(override)
    except Exception as exc:
        print(f"[warn] Could not load temporal segmentation config: {exc}. Using defaults.")
    return defaults


def _get_segment_object_motion_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "rel_vz_threshold": 0.2,
        "rel_vx_threshold": 0.2,
        "compare_rel_vx_thresholds": [10.0, 20.0, 50.0],
        "rel_speed_threshold": 0.3,
        "dominance_ratio_threshold": 0.6,
        "distance_near_threshold": 15.0,
        "distance_medium_threshold": 30.0,
        "top_k_visualized_objects": 20,
        "visualization_fps": 10.0,
    }
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        override = cfg.get("segment_object_motion", {})
        if isinstance(override, dict):
            defaults.update(override)
    except Exception as exc:
        print(f"[warn] Could not load segment object motion config: {exc}. Using defaults.")
    return defaults


def _get_important_objects_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "selection_strategy": "not_implemented",
        "passthrough_selected_objects": True,
    }
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        override = cfg.get("important_objects", {})
        if isinstance(override, dict):
            defaults.update(override)
    except Exception as exc:
        print(f"[warn] Could not load important objects config: {exc}. Using defaults.")
    return defaults


def _get_logic_atoms_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "lateral_position_threshold": 2.0,
        "visibility_persistent_threshold": 0.8,
        "visibility_present_threshold": 0.3,
        "include_segment_boundary_atoms": True,
        "include_object_identity_atoms": True,
    }
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        override = cfg.get("logic_atoms", {})
        if isinstance(override, dict):
            defaults.update(override)
    except Exception as exc:
        print(f"[warn] Could not load logic atoms config: {exc}. Using defaults.")
    return defaults


def _get_target_head_atoms_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "target_predicate": "brake_next",
        "negative_target_predicate": "not_brake_next",
        "positive_forward_states": ["forward_slowdown", "stopping"],
        "include_negative_examples": True,
    }
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        override = cfg.get("target_head_atoms", {})
        if isinstance(override, dict):
            defaults.update(override)
    except Exception as exc:
        print(f"[warn] Could not load target head atoms config: {exc}. Using defaults.")
    return defaults


def _get_temporal_rule_examples_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "deduplicate_body_atoms": True,
        "sort_body_atoms": True,
        "include_clause_text": True,
        "include_negative_examples": True,
    }
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        override = cfg.get("temporal_rule_examples", {})
        if isinstance(override, dict):
            defaults.update(override)
    except Exception as exc:
        print(f"[warn] Could not load temporal rule examples config: {exc}. Using defaults.")
    return defaults


def _get_candidate_rules_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
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
    }
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        override = cfg.get("candidate_rules", {})
        if isinstance(override, dict):
            defaults.update(override)
    except Exception as exc:
        print(f"[warn] Could not load candidate rules config: {exc}. Using defaults.")
    return defaults


def _get_extended_rules_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
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
    }
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        override = cfg.get("extended_rules", {})
        if isinstance(override, dict):
            defaults.update(override)
    except Exception as exc:
        print(f"[warn] Could not load extended rules config: {exc}. Using defaults.")
    return defaults


def _get_final_rules_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "top_k": 50,
    }
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        override = cfg.get("final_rules", {})
        if isinstance(override, dict):
            defaults.update(override)
    except Exception as exc:
        print(f"[warn] Could not load final rules config: {exc}. Using defaults.")
    return defaults


def _get_data_split_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "train_video_count": DEFAULT_TRAIN_VIDEO_COUNT,
        "eval_video_count": DEFAULT_EVAL_VIDEO_COUNT,
        "strategy": "eval_fraction",
        "eval_fraction": 0.2,
    }
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        override = cfg.get("data_split", {})
        if isinstance(override, dict):
            defaults.update(override)
    except Exception as exc:
        print(f"[warn] Could not load data split config: {exc}. Using defaults.")
    return defaults


def _get_rule_evaluation_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "prediction_mode": "any_rule_positive",
    }
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        override = cfg.get("rule_evaluation", {})
        if isinstance(override, dict):
            defaults.update(override)
    except Exception as exc:
        print(f"[warn] Could not load rule evaluation config: {exc}. Using defaults.")
    return defaults


def _get_pipeline_recompute_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        # Step 14 is per-video and safe to load from cache after split changes.
        "candidate_rules": False,
        # Steps 16-18 depend on the selected train/eval split and should refresh.
        "extended_rules": True,
        "final_rules": True,
        "rule_evaluation": True,
        "error_and_explainability_analysis": True,
    }
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        override = cfg.get("pipeline_recompute", {})
        if isinstance(override, dict):
            defaults.update(override)
    except Exception as exc:
        print(f"[warn] Could not load pipeline recompute config: {exc}. Using defaults.")
    return defaults


def _get_merged_candidate_rules_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "15_driving_mini_merged_initial_rules"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _get_split_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "driving_mini_split"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _get_error_and_explainability_cfg() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "vehicle_classes": ["car", "truck", "bus", "motorcycle", "bicycle"],
        "dense_context_min_objects": 2,
        "overlap_rule_threshold": 10,
    }
    try:
        cfg_path = config.get_config_path("exp_driving")
        cfg = load_pattern_cfg_file(cfg_path)
        override = cfg.get("error_and_explainability_analysis", {})
        if isinstance(override, dict):
            defaults.update(override)
    except Exception as exc:
        print(f"[warn] Could not load error analysis config: {exc}. Using defaults.")
    return defaults


def _dedupe_rule_evidence_entries(evidence_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
    for entry in evidence_entries:
        bindings = dict(entry.get("bindings", {}))
        key = (
            str(entry.get("example_id", "")),
            str(entry.get("matched_atom", "")),
            tuple(sorted((str(k), str(v)) for k, v in bindings.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _summarize_rule_evidence(evidence_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_firings = len(evidence_entries)
    positive_firings = sum(1 for entry in evidence_entries if bool(entry.get("label", False)))
    negative_firings = total_firings - positive_firings
    positive_example_ids = sorted(
        {
            str(entry.get("example_id", ""))
            for entry in evidence_entries
            if bool(entry.get("label", False)) and str(entry.get("example_id", ""))
        }
    )
    negative_example_ids = sorted(
        {
            str(entry.get("example_id", ""))
            for entry in evidence_entries
            if not bool(entry.get("label", False)) and str(entry.get("example_id", ""))
        }
    )
    total_support = len(set(positive_example_ids) | set(negative_example_ids))
    confidence = float(positive_firings / max(1, total_firings))
    return {
        "positive_support": len(positive_example_ids),
        "negative_support": len(negative_example_ids),
        "total_support": total_support,
        "positive_firings": positive_firings,
        "negative_firings": negative_firings,
        "total_firings": total_firings,
        "confidence": confidence,
        "positive_example_ids": positive_example_ids,
        "negative_example_ids": negative_example_ids,
    }


def _merge_candidate_rules(candidate_rule_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    out_root = _get_merged_candidate_rules_output_root()
    json_path = out_root / "merged_initial_rules.json"
    csv_path = out_root / "merged_initial_rules.csv"
    manifest_path = out_root / "merged_initial_rules_manifest.json"

    merged_rule_map: Dict[str, Dict[str, Any]] = {}
    video_summaries: List[Dict[str, Any]] = []
    target_predicates: set[str] = set()
    total_examples = 0
    total_positive_examples = 0
    total_negative_examples = 0

    for video_result in sorted(candidate_rule_results, key=lambda item: str(item.get("video_id", ""))):
        video_id = str(video_result.get("video_id", "unknown"))
        target_predicate = str(video_result.get("target_predicate", ""))
        if target_predicate:
            target_predicates.add(target_predicate)
        total_examples += int(video_result.get("num_examples", 0))
        total_positive_examples += int(video_result.get("num_positive_examples", 0))
        total_negative_examples += int(video_result.get("num_negative_examples", 0))

        candidate_rules = list(video_result.get("initial_rules", video_result.get("candidate_rules", [])))
        video_summaries.append(
            {
                "video_id": video_id,
                "target_predicate": target_predicate,
                "num_initial_rules": len(candidate_rules),
                "num_examples": int(video_result.get("num_examples", 0)),
                "num_positive_examples": int(video_result.get("num_positive_examples", 0)),
                "num_negative_examples": int(video_result.get("num_negative_examples", 0)),
            }
        )

        for rule in candidate_rules:
            clause = str(rule.get("clause", "")).strip()
            if not clause:
                continue

            merged_rule = merged_rule_map.get(clause)
            if merged_rule is None:
                merged_rule = {
                    "merged_rule_index": -1,
                    "rule_id": f"merged_rule_{len(merged_rule_map):04d}",
                    "head_predicate": rule.get("head_predicate", ""),
                    "head_atom_template": rule.get("head_atom_template", ""),
                    "body_atom_template": rule.get("body_atom_template", ""),
                    "clause": clause,
                    "positive_support": 0,
                    "negative_support": 0,
                    "total_support": 0,
                    "positive_firings": 0,
                    "negative_firings": 0,
                    "total_firings": 0,
                    "confidence": 0.0,
                    "positive_example_ids": [],
                    "negative_example_ids": [],
                    "evidence_set": [],
                    "source_video_ids": [],
                    "source_rule_ids": [],
                    "source_rule_indices": [],
                    "num_source_rules": 0,
                }
                merged_rule_map[clause] = merged_rule

            merged_rule["evidence_set"].extend(list(rule.get("evidence_set", [])))
            merged_rule["source_video_ids"].append(video_id)
            merged_rule["source_rule_ids"].append(rule.get("rule_id", ""))
            merged_rule["source_rule_indices"].append(rule.get("rule_index"))
            merged_rule["num_source_rules"] += 1

    merged_rules = sorted(merged_rule_map.values(), key=lambda item: str(item.get("clause", "")))
    for idx, merged_rule in enumerate(merged_rules):
        merged_rule["merged_rule_index"] = idx
        merged_rule["evidence_set"] = _dedupe_rule_evidence_entries(list(merged_rule.get("evidence_set", [])))
        evidence_summary = _summarize_rule_evidence(list(merged_rule.get("evidence_set", [])))
        merged_rule["positive_support"] = int(evidence_summary["positive_support"])
        merged_rule["negative_support"] = int(evidence_summary["negative_support"])
        merged_rule["total_support"] = int(evidence_summary["total_support"])
        merged_rule["positive_firings"] = int(evidence_summary["positive_firings"])
        merged_rule["negative_firings"] = int(evidence_summary["negative_firings"])
        merged_rule["total_firings"] = int(evidence_summary["total_firings"])
        merged_rule["confidence"] = float(evidence_summary["confidence"])
        merged_rule["positive_example_ids"] = list(evidence_summary["positive_example_ids"])
        merged_rule["negative_example_ids"] = list(evidence_summary["negative_example_ids"])
        merged_rule["source_video_ids"] = sorted(set(str(v) for v in merged_rule["source_video_ids"]))
        merged_rule["source_rule_ids"] = sorted(
            set(str(rule_id) for rule_id in merged_rule["source_rule_ids"] if str(rule_id))
        )
        merged_rule["source_rule_indices"] = [
            idx for idx in merged_rule["source_rule_indices"] if idx is not None
        ]

    merged_result: Dict[str, Any] = {
        "num_videos": len(candidate_rule_results),
        "num_examples": total_examples,
        "num_positive_examples": total_positive_examples,
        "num_negative_examples": total_negative_examples,
        "num_rules": len(merged_rules),
        "target_predicates": sorted(target_predicates),
        "rules": merged_rules,
    }

    manifest: Dict[str, Any] = {
        "num_videos": len(candidate_rule_results),
        "num_examples": total_examples,
        "num_positive_examples": total_positive_examples,
        "num_negative_examples": total_negative_examples,
        "num_rules": len(merged_rules),
        "target_predicates": sorted(target_predicates),
        "videos": video_summaries,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(merged_result, fh, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "merged_rule_index",
                "rule_id",
                "head_predicate",
                "head_atom_template",
                "body_atom_template",
                "clause",
                "positive_support",
                "negative_support",
                "total_support",
                "positive_firings",
                "negative_firings",
                "total_firings",
                "confidence",
                "positive_example_ids",
                "negative_example_ids",
                "source_video_ids",
                "source_rule_ids",
                "source_rule_indices",
                "num_source_rules",
            ],
        )
        writer.writeheader()
        for rule in merged_rules:
            writer.writerow(
                {
                    "merged_rule_index": rule.get("merged_rule_index", ""),
                    "rule_id": rule.get("rule_id", ""),
                    "head_predicate": rule.get("head_predicate", ""),
                    "head_atom_template": rule.get("head_atom_template", ""),
                    "body_atom_template": rule.get("body_atom_template", ""),
                    "clause": rule.get("clause", ""),
                    "positive_support": rule.get("positive_support", 0),
                    "negative_support": rule.get("negative_support", 0),
                    "total_support": rule.get("total_support", 0),
                    "positive_firings": rule.get("positive_firings", 0),
                    "negative_firings": rule.get("negative_firings", 0),
                    "total_firings": rule.get("total_firings", 0),
                    "confidence": rule.get("confidence", 0.0),
                    "positive_example_ids": json.dumps(rule.get("positive_example_ids", [])),
                    "negative_example_ids": json.dumps(rule.get("negative_example_ids", [])),
                    "source_video_ids": json.dumps(rule.get("source_video_ids", [])),
                    "source_rule_ids": json.dumps(rule.get("source_rule_ids", [])),
                    "source_rule_indices": json.dumps(rule.get("source_rule_indices", [])),
                    "num_source_rules": rule.get("num_source_rules", 0),
                }
            )

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Merged initial rule JSON written to {json_path}")
    print(f"Merged initial rule CSV written to {csv_path}")
    print(f"Merged initial rule manifest written to {manifest_path}")
    return merged_result


def _select_video_results(
    video_results: List[Dict[str, Any]],
    selected_video_ids: List[str],
) -> List[Dict[str, Any]]:
    selected_video_id_set = {str(video_id) for video_id in selected_video_ids}
    return [
        result
        for result in video_results
        if str(result.get("video_id", "")) in selected_video_id_set
    ]


def _build_train_eval_split(
    video_ids: List[str],
    train_video_count: int,
    eval_video_count: int,
    strategy: str = "eval_fraction",
    eval_fraction: float = 0.2,
) -> Dict[str, Any]:
    unique_video_ids = sorted({str(video_id) for video_id in video_ids if str(video_id)})
    total_videos = len(unique_video_ids)
    if total_videos < 2:
        raise ValueError(
            "At least 2 videos are required to create train/evaluation splits. "
            f"Found {total_videos}."
        )
    if train_video_count < 1:
        raise ValueError(f"train_video_count must be >= 1. Found {train_video_count}.")
    if eval_video_count < 1:
        raise ValueError(f"eval_video_count must be >= 1. Found {eval_video_count}.")

    strategy = str(strategy or "eval_fraction")
    if strategy == "fixed_counts":
        requested_total = train_video_count + eval_video_count
        if total_videos >= requested_total:
            effective_train_count = train_video_count
            effective_eval_count = eval_video_count
            resolved_strategy = f"fixed_counts_first_{train_video_count}_train_{eval_video_count}_eval"
        else:
            effective_train_count = max(1, total_videos - 1)
            effective_eval_count = total_videos - effective_train_count
            resolved_strategy = "fallback_last_video_eval"
    elif strategy == "eval_fraction":
        eval_fraction = float(eval_fraction)
        if not 0.0 < eval_fraction < 1.0:
            raise ValueError(f"eval_fraction must be between 0 and 1. Found {eval_fraction}.")
        effective_eval_count = max(1, min(total_videos - 1, int(math.ceil(total_videos * eval_fraction))))
        effective_train_count = total_videos - effective_eval_count
        resolved_strategy = f"eval_fraction_{eval_fraction:g}"
    else:
        raise ValueError(f"Unsupported data split strategy: {strategy}")

    train_video_ids = unique_video_ids[:effective_train_count]
    eval_video_ids = unique_video_ids[effective_train_count : effective_train_count + effective_eval_count]
    if not eval_video_ids:
        raise ValueError("Failed to assign evaluation videos for the train/eval split.")

    split_manifest = {
        "num_total_videos": total_videos,
        "num_train_videos": len(train_video_ids),
        "num_eval_videos": len(eval_video_ids),
        "requested_train_video_count": train_video_count,
        "requested_eval_video_count": eval_video_count,
        "requested_eval_fraction": eval_fraction if strategy == "eval_fraction" else None,
        "strategy": resolved_strategy,
        "train_video_ids": train_video_ids,
        "eval_video_ids": eval_video_ids,
        "unused_video_ids": unique_video_ids[effective_train_count + effective_eval_count :],
    }

    out_root = _get_split_output_root()
    manifest_path = out_root / "train_eval_split.json"
    split_manifest["manifest_path"] = str(manifest_path)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(split_manifest, fh, indent=2)
    print(
        "Train/eval split: "
        f"train={train_video_ids} | "
        f"eval={eval_video_ids}"
    )
    print(f"Split manifest written to {manifest_path}")
    return split_manifest


def _get_default_driving_mini_video_ids() -> List[str]:
    dataset_root = config.get_dataset_path("driving_mini")
    frames_root = dataset_root / "frames"
    if frames_root.exists():
        video_ids = sorted(path.name for path in frames_root.iterdir() if path.is_dir())
        if video_ids:
            return video_ids

    videos_root = dataset_root / "videos"
    if videos_root.exists():
        return sorted(path.stem for path in videos_root.glob("*.mov"))
    return []


def _resolve_video_ids(video_ids: List[str] | None = None) -> List[str] | None:
    if video_ids:
        return list(video_ids)
    default_video_ids = _get_default_driving_mini_video_ids()
    return default_video_ids or None


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
        default=19,
        choices=range(1, 20),
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


def main(max_step: int = 19, video_ids: List[str] | None = None) -> None:
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

    # Step 18: evaluate final rules on held-out evaluation videos
    rule_evaluation_cfg = _get_rule_evaluation_cfg()
    print("\n=== Step 18: evaluate_rules_driving_mini ===")
    print(f"Rule evaluation cfg: {rule_evaluation_cfg}")
    evaluation_results: Dict[str, Any] = evaluate_rules_driving_mini.run(
        final_rule_results=final_rule_results,
        temporal_rule_results=eval_temporal_rule_results,
        eval_video_ids=list(split_manifest.get("eval_video_ids", [])),
        split_manifest=split_manifest,
        cfg=rule_evaluation_cfg,
        force_recompute=bool(recompute_cfg.get("rule_evaluation", True)),
    )
    overall_metrics = dict(evaluation_results.get("overall_metrics", {}))
    print(
        "Held-out evaluation complete. "
        f"Precision={float(overall_metrics.get('precision', 0.0)):.3f} | "
        f"Recall={float(overall_metrics.get('recall', 0.0)):.3f} | "
        f"F1={float(overall_metrics.get('f1', 0.0)):.3f}"
    )
    if max_step == 18:
        print("\nStopping after step 18 by request.")
        return

    # Step 19: error analysis + explainability diagnostics on held-out examples
    error_analysis_cfg = _get_error_and_explainability_cfg()
    print("\n=== Step 19: error_and_explainability_analysis_driving_mini ===")
    print(f"Error analysis cfg: {error_analysis_cfg}")
    error_analysis_results: Dict[str, Any] = error_and_explainability_analysis_driving_mini.run(
        final_rule_results=final_rule_results,
        temporal_rule_results=eval_temporal_rule_results,
        evaluation_results=evaluation_results,
        cfg=error_analysis_cfg,
        force_recompute=bool(recompute_cfg.get("error_and_explainability_analysis", True)),
    )
    print(
        "Held-out error analysis complete. "
        f"FN={int(error_analysis_results.get('num_fn_examples', 0))} | "
        f"FP={int(error_analysis_results.get('num_fp_examples', 0))}"
    )
    if max_step == 19:
        print("\nStopping after step 19 by request.")
        return

if __name__ == "__main__":
    args = parse_args()
    main(max_step=args.max_step, video_ids=args.video_ids)
