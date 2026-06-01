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
        9. (future) pattern_mining_driving_mini: beam search rules;
        
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import config
from src.exp_driving_videos.modules import detect_driving_mini
from src.exp_driving_videos.modules import dataset_annotations_driving_mini
from src.exp_driving_videos.modules import merge_gt_and_detected_driving_mini
from src.exp_driving_videos.modules import prepare_3d_positions_driving_mini
from src.exp_driving_videos.modules import tracking_driving_mini
from src.exp_driving_videos.modules import ego_motion_driving_mini
from src.exp_driving_videos.modules import relative_object_motion_driving_mini
from src.exp_driving_videos.modules import temporal_segmentation_driving_mini
from src.exp_driving_videos.modules.pipe_utils.exp_driving_utils import load_pattern_cfg_file


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
        "min_stop_duration": 3,
        "min_turn_duration": 3,
        "min_segment_length": 3,
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


def main() -> None:
    smoothing_window = _get_ego_motion_smoothing_window(default=5)
    static_adjust_cfg = _get_ego_static_adjustment_cfg()
    temporal_seg_cfg = _get_temporal_segmentation_cfg()

    # Step 1: object detection over driving_mini frames
    print("=== Step 1: detect_driving_mini ===")
    detection_results: List[Dict[str, Any]] = detect_driving_mini.run()
    print(f"Detection complete. Processed {len(detection_results)} video(s).")
    print("*** Vis Path: ", Path(config.get_output_path("pipeline_output")) / "driving_mini_detection")
    # Step 2: multi-object tracking with ByteTrack
    print("\n=== Step 2: tracking_driving_mini ===")
    tracking_results: List[Dict[str, Any]] = tracking_driving_mini.run(detection_results)
    print(f"Tracking complete. Processed {len(tracking_results)} video(s).")

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

    # Step 5: prepare per-object 3D positions using depth maps
    print("\n=== Step 5: prepare_3d_positions_driving_mini ===")
    positions_3d_results: List[Dict[str, Any]] = prepare_3d_positions_driving_mini.run(
        merged_results=merged_results,
    )
    print(
        "3D position preparation complete. "
        f"Processed {len(positions_3d_results)} video(s)."
    )

    # Step 6: ego motion estimation from optical flow + background mask + depth
    print("\n=== Step 6: ego_motion_driving_mini ===")
    print(f"Using ego motion smoothing_window={smoothing_window}")
    print(f"Using ego static adjustment cfg={static_adjust_cfg}")
    ego_motion_results: List[Dict[str, Any]] = ego_motion_driving_mini.run(
        merged_results=merged_results,
        smoothing_window=smoothing_window,
        static_adjust_cfg=static_adjust_cfg,
    )
    print(
        "Ego motion estimation complete. "
        f"Processed {len(ego_motion_results)} video(s)."
    )

    # Step 7: object motion relative to ego motion in camera 3D frame
    print("\n=== Step 7: relative_object_motion_driving_mini ===")
    relative_motion_results: List[Dict[str, Any]] = relative_object_motion_driving_mini.run(
        positions_3d_results=positions_3d_results,
        ego_motion_results=ego_motion_results,
    )
    print(
        "Relative object motion estimation complete. "
        f"Processed {len(relative_motion_results)} video(s)."
    )

    # Step 8: temporal segmentation of ego-motion signals to event spans/cut points
    print("\n=== Step 8: temporal_segmentation_driving_mini ===")
    print(f"Temporal segmentation cfg: {temporal_seg_cfg}")
    temporal_seg_results: List[Dict[str, Any]] = temporal_segmentation_driving_mini.run(
        ego_motion_results=ego_motion_results,
        relative_motion_results=relative_motion_results,
        seg_cfg=temporal_seg_cfg,
    )
    print(
        "Temporal segmentation complete. "
        f"Processed {len(temporal_seg_results)} video(s)."
    )
    
    # TODO: segmentation should also be 
    # TODO: generate atom rules based on the symbolized segments
    
    # TODO: rule mining based on the atom rules


if __name__ == "__main__":
    main()
