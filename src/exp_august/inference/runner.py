"""Runner for the target annotation-free ``exp_august`` inference pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import config
from src.exp_driving_videos import pipeline_config

from src.exp_august.inference.step01_init import run_step1
from src.exp_august.inference.depth_backend import (
    Da3DepthEvidenceBackend,
    DisabledDepthBackend,
)
from src.exp_august.inference.flow_backend import (
    DisabledFlowBackend,
    RaftFlowEvidenceBackend,
)
from src.exp_august.inference.mask_backend import (
    DisabledMaskBackend,
    Sam2MaskEvidenceBackend,
)
from src.exp_august.inference.step02_neural_evidence import (
    DisabledObjectBackend,
    YoloWorldEvidenceBackend,
    run_step2,
)
from src.exp_august.inference.step03_object_tracking import run_step3
from src.exp_august.inference.step03_visualization import render_step3_visualizations
from src.exp_august.inference.step04_geometry_scale import run_step4
from src.exp_august.inference.step04_visualization import render_step4_visualizations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the target exp_august training-free inference pipeline"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video-paths", nargs="+", type=Path)
    source.add_argument("--video-ids", nargs="+")
    parser.add_argument("--video-count", type=int)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=config.get_dataset_path("driving_mini"),
    )
    parser.add_argument("--dataset-name", default="driving_mini")
    parser.add_argument("--canonical-fps", type=float, default=10.0)
    parser.add_argument(
        "--decode-validation",
        choices=("none", "sample", "full"),
        default="sample",
    )
    parser.add_argument("--decode-sample-count", type=int, default=7)
    parser.add_argument("--seed", type=int, default=726381)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=config.get_output_path("output") / "pipeline_august" / "world_state",
    )
    parser.add_argument("--max-step", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument(
        "--objects-backend",
        choices=("yolo_world", "disabled"),
        default="yolo_world",
    )
    parser.add_argument(
        "--yolo-model",
        default="weights/yolo/yolov8l-worldv2.pt",
    )
    parser.add_argument("--object-classes", nargs="+")
    parser.add_argument("--primary-confidence", type=float, default=0.30)
    parser.add_argument("--candidate-confidence", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--inference-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument(
        "--masks-backend",
        choices=("sam2", "disabled"),
        default="disabled",
    )
    parser.add_argument("--sam2-model", default="weights/sam2/sam2_t.pt")
    parser.add_argument("--sam-prompt-candidates", action="store_true")
    parser.add_argument(
        "--flow-backend",
        choices=("raft", "disabled"),
        default="disabled",
    )
    parser.add_argument("--flow-consistency-threshold-px", type=float, default=1.5)
    parser.add_argument(
        "--depth-backend",
        choices=("da3", "disabled"),
        default="disabled",
    )
    parser.add_argument("--depth-model", default="depth-anything/DA3-Large")
    parser.add_argument("--depth-process-resolution", type=int, default=504)
    parser.add_argument("--tracking-max-age-frames", type=int, default=2)
    parser.add_argument("--tracking-min-score", type=float, default=0.30)
    parser.add_argument("--tracking-max-center-distance-ratio", type=float, default=0.25)
    parser.add_argument("--tracking-soft-class-gate", action="store_true")
    parser.add_argument("--tracking-bootstrap-candidates", action="store_true")
    parser.add_argument("--tracking-min-mask-area", type=int, default=16)
    parser.add_argument("--tracking-depth-erosion-pixels", type=int, default=2)
    parser.add_argument("--tracking-mask-iou-weight", type=float, default=0.40)
    parser.add_argument("--tracking-flow-iou-weight", type=float, default=0.20)
    parser.add_argument("--tracking-box-iou-weight", type=float, default=0.20)
    parser.add_argument("--tracking-class-weight", type=float, default=0.10)
    parser.add_argument("--tracking-depth-weight", type=float, default=0.10)
    parser.add_argument("--evidence-policy-seed", type=int, default=726381)
    parser.add_argument("--depth-check-fraction", type=float, default=0.20)
    parser.add_argument("--visualize-step3", action="store_true")
    parser.add_argument("--step3-example-frame-count", type=int, default=4)
    parser.add_argument("--no-step3-video", action="store_true")
    parser.add_argument("--camera-fx-px", type=float)
    parser.add_argument("--camera-fy-px", type=float)
    parser.add_argument("--camera-cx-px", type=float)
    parser.add_argument("--camera-cy-px", type=float)
    parser.add_argument("--horizontal-fov-degrees", type=float, default=90.0)
    parser.add_argument("--horizontal-fov-min-degrees", type=float, default=60.0)
    parser.add_argument("--horizontal-fov-max-degrees", type=float, default=120.0)
    parser.add_argument("--geometry-min-support-pixels", type=int, default=16)
    parser.add_argument("--geometry-min-valid-depth-fraction", type=float, default=0.25)
    parser.add_argument("--geometry-background-flow-stride", type=int, default=16)
    parser.add_argument("--geometry-min-pose-correspondences", type=int, default=32)
    parser.add_argument("--visualize-step4", action="store_true")
    parser.add_argument("--step4-example-frame-count", type=int, default=4)
    parser.add_argument("--step4-maximum-tracks", type=int, default=12)
    parser.add_argument("--no-step4-video", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_step1(
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        video_paths=args.video_paths,
        video_ids=args.video_ids,
        video_count=args.video_count,
        canonical_fps=args.canonical_fps,
        decode_validation_mode=args.decode_validation,
        decode_sample_count=args.decode_sample_count,
        random_seed=args.seed,
        ffprobe_executable=args.ffprobe,
    )
    print(f"Step 1 completed: {len(result.manifests)} video(s)")
    print(f"Run ID: {result.bundle.run_id}")
    print(f"Manifest bundle: {result.bundle_path}")
    if args.max_step == 1:
        return 0
    classes = (
        list(args.object_classes)
        if args.object_classes
        else list(pipeline_config.DRIVING_MINI_OD_CLASSES)
    )
    if args.objects_backend == "disabled":
        object_backend = DisabledObjectBackend()
        classes = []
    else:
        object_backend = YoloWorldEvidenceBackend(
            model_name=args.yolo_model,
            classes=classes,
            primary_confidence=args.primary_confidence,
            candidate_confidence=args.candidate_confidence,
            nms_iou=args.nms_iou,
            inference_size=args.inference_size,
            device=args.device,
            allow_model_download=args.allow_model_download,
        )
    if args.masks_backend == "sam2":
        mask_backend = Sam2MaskEvidenceBackend(
            model_name=args.sam2_model,
            device=args.device,
            prompt_candidates=args.sam_prompt_candidates,
            allow_model_download=args.allow_model_download,
        )
    else:
        mask_backend = DisabledMaskBackend()
    if args.flow_backend == "raft":
        flow_backend = RaftFlowEvidenceBackend(
            device=args.device,
            consistency_threshold_px=args.flow_consistency_threshold_px,
            allow_model_download=args.allow_model_download,
        )
    else:
        flow_backend = DisabledFlowBackend()
    if args.depth_backend == "da3":
        depth_backend = Da3DepthEvidenceBackend(
            model_name=args.depth_model,
            device=args.device,
            process_resolution=args.depth_process_resolution,
        )
    else:
        depth_backend = DisabledDepthBackend()
    step2 = run_step2(
        init_bundle_path=result.bundle_path,
        object_backend=object_backend,
        object_classes=classes,
        primary_confidence=args.primary_confidence,
        candidate_confidence=args.candidate_confidence,
        nms_iou=args.nms_iou,
        inference_size=args.inference_size,
        batch_size=args.batch_size,
        device=getattr(object_backend, "device", args.device),
        mask_backend=mask_backend,
        flow_backend=flow_backend,
        depth_backend=depth_backend,
    )
    print(f"Step 2 completed: {len(step2.video_manifests)} video(s)")
    print(f"Neural evidence store: {step2.store_path}")
    if args.max_step == 2:
        return 0
    step3 = run_step3(
        neural_evidence_store_path=step2.store_path,
        max_age_frames=args.tracking_max_age_frames,
        minimum_assignment_score=args.tracking_min_score,
        maximum_center_distance_ratio=args.tracking_max_center_distance_ratio,
        hard_class_gate=not args.tracking_soft_class_gate,
        bootstrap_primary_only=not args.tracking_bootstrap_candidates,
        minimum_mask_area=args.tracking_min_mask_area,
        depth_erosion_pixels=args.tracking_depth_erosion_pixels,
        mask_iou_weight=args.tracking_mask_iou_weight,
        flow_iou_weight=args.tracking_flow_iou_weight,
        box_iou_weight=args.tracking_box_iou_weight,
        class_weight=args.tracking_class_weight,
        depth_weight=args.tracking_depth_weight,
        evidence_policy_seed=args.evidence_policy_seed,
        depth_check_fraction=args.depth_check_fraction,
    )
    print(f"Step 3 completed: {len(step3.video_manifests)} video(s)")
    print(f"Tracking store: {step3.store_path}")
    if args.visualize_step3:
        visualization_manifest = render_step3_visualizations(
            tracking_store_path=step3.store_path,
            example_frame_count=args.step3_example_frame_count,
            render_video=not args.no_step3_video,
        )
        print(f"Step 3 visualizations: {visualization_manifest}")
    if args.max_step == 3:
        return 0
    step4 = run_step4(
        tracking_store_path=step3.store_path,
        camera_fx_px=args.camera_fx_px,
        camera_fy_px=args.camera_fy_px,
        camera_cx_px=args.camera_cx_px,
        camera_cy_px=args.camera_cy_px,
        horizontal_fov_degrees=args.horizontal_fov_degrees,
        horizontal_fov_min_degrees=args.horizontal_fov_min_degrees,
        horizontal_fov_max_degrees=args.horizontal_fov_max_degrees,
        minimum_support_pixels=args.geometry_min_support_pixels,
        minimum_valid_depth_fraction=args.geometry_min_valid_depth_fraction,
        background_flow_sample_stride=args.geometry_background_flow_stride,
        minimum_pose_correspondences=args.geometry_min_pose_correspondences,
    )
    print(f"Step 4 completed: {len(step4.video_manifests)} video(s)")
    print(f"Geometry store: {step4.store_path}")
    if args.visualize_step4:
        visualization_manifest = render_step4_visualizations(
            geometry_store_path=step4.store_path,
            example_frame_count=args.step4_example_frame_count,
            maximum_tracks=args.step4_maximum_tracks,
            render_video=not args.no_step4_video,
        )
        print(f"Step 4 visualizations: {visualization_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
