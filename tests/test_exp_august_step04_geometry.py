import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.exp_august.contracts import (
    CoordinateSpace,
    DepthRepresentation,
    DetectionTier,
    GeometryStore,
    LocalReestimationStore,
    Observability,
    ResidualStore,
    RepairProposalStore,
    ResidualFamily,
    EvaluationBasis,
    VideoWorldStateManifest,
    VideoRepairProposalManifest,
    VideoLocalReestimationManifest,
    VideoGeometryManifest,
    WorldStateStore,
)
from src.exp_august.contracts.codec import read_contract, sha256_file
from src.exp_august.inference.depth_backend import DepthFrameOutput
from src.exp_august.inference.flow_backend import DirectionalFlowOutput, FlowPairOutput
from src.exp_august.inference.mask_backend import MaskCandidateOutput
from src.exp_august.inference.step01_init import run_step1
from src.exp_august.inference.step02_neural_evidence import ObjectCandidate, run_step2
from src.exp_august.inference.step03_object_tracking import run_step3
from src.exp_august.inference.step04_geometry_scale import run_step4
from src.exp_august.inference.step04_visualization import (
    _relative_static_scene,
    render_step4_visualizations,
)
from src.exp_august.inference.step05_joint_world_reconstruction import run_step5
from src.exp_august.inference.step05_visualization import render_step5_visualizations
from src.exp_august.inference.step06_predict_verify import run_step6
from src.exp_august.inference.step06_visualization import render_step6_visualizations
from src.exp_august.inference.step07_diagnose_propose import run_step7
from src.exp_august.inference.step07_visualization import render_step7_visualizations
from src.exp_august.inference.step08_local_reestimation import run_step8
from src.exp_august.inference.step08_visualization import render_step8_visualizations


class _Objects:
    backend_name = "fake_objects"
    model_name = "fake"
    model_id = "fake-objects@v1"
    available = True
    unavailable_reason = None
    tool_versions = ()

    def warmup(self):
        return None

    def predict_batch(self, frames):
        return tuple(
            (
                ObjectCandidate(
                    bbox_xyxy=(8.0, 8.0, 40.0, 38.0),
                    class_name="car",
                    confidence=0.9,
                    tier=DetectionTier.PRIMARY,
                ),
            )
            if frame.frame_index % 2 == 0
            else ()
            for frame in frames
        )

    def teardown(self):
        return None


class _Masks:
    backend_name = "fake_masks"
    model_name = "fake"
    model_id = "fake-masks@v1"
    available = True
    unavailable_reason = None
    prompt_candidates = False
    tool_versions = ()

    def warmup(self):
        return None

    def predict_frame(self, frame, detections):
        outputs = []
        for detection in detections:
            mask = np.zeros(frame.image_bgr.shape[:2], dtype=bool)
            mask[8:38, 8:40] = True
            outputs.append(
                MaskCandidateOutput(
                    prompt_detection_id=detection.detection_id,
                    mask=mask,
                    confidence=0.85,
                )
            )
        return tuple(outputs)

    def teardown(self):
        return None


class _Flow:
    backend_name = "fake_flow"
    model_name = "fake"
    model_id = "fake-flow@v1"
    available = True
    unavailable_reason = None
    consistency_threshold_px = 1.5
    tool_versions = ()

    def warmup(self):
        return None

    def predict_pair(self, earlier, later):
        height, width = earlier.image_bgr.shape[:2]
        valid = np.ones((height, width), dtype=bool)
        error = np.zeros((height, width), dtype=np.float32)

        def direction(dx):
            flow = np.zeros((height, width, 2), dtype=np.float32)
            flow[:, :, 0] = dx
            return DirectionalFlowOutput(
                flow=flow,
                domain_valid=valid,
                consistency_valid=valid,
                fb_error=error,
            )

        return FlowPairOutput(forward=direction(1.0), backward=direction(-1.0))

    def teardown(self):
        return None


class _Depth:
    backend_name = "fake_relative_depth"
    model_name = "fake"
    model_id = "fake-depth@v1"
    available = True
    unavailable_reason = None
    process_resolution = 64
    representation = DepthRepresentation.RELATIVE
    tool_versions = ()

    def warmup(self):
        return None

    def predict_frame(self, frame):
        height, width = frame.image_bgr.shape[:2]
        rows, columns = np.indices((height, width), dtype=np.float32)
        depth = 2.0 + 0.1 * frame.frame_index + rows / height + columns / width
        return DepthFrameOutput(
            depth=depth.astype(np.float32),
            valid=np.ones((height, width), dtype=bool),
            confidence=np.full((height, width), 0.8, dtype=np.float32),
            representation=DepthRepresentation.RELATIVE,
        )

    def teardown(self):
        return None


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 6.0, (64, 48)
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer could not open")
    for index in range(12):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:, :, 1] = index * 10
        writer.write(image)
    writer.release()


class ExpAugustStep04GeometryTests(unittest.TestCase):
    def test_relative_static_scene_recovers_normalized_ego_and_landmark(self):
        def observation(frame_index, z):
            return SimpleNamespace(
                frame_index=frame_index,
                points=SimpleNamespace(
                    median=SimpleNamespace(x=2.0, y=0.5, z=z)
                ),
            )

        track = SimpleNamespace(
            track_id="track:000001",
            primary_class="traffic light",
            observations=(observation(0, 10.0), observation(1, 9.0), observation(2, 8.0)),
        )

        def pose(source, target):
            return SimpleNamespace(
                pose_id=f"pose:{source}:{target}",
                source_frame_index=source,
                target_frame_index=target,
                rotation_source_to_target=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                translation_direction_source_to_target=SimpleNamespace(x=0.0, y=0.0, z=-1.0),
                inlier_fraction=0.9,
                median_epipolar_residual_px=0.1,
            )

        manifest = SimpleNamespace(
            video_id="synthetic-static-scene",
            canonical_fps=5.0,
            tracks=(track,),
            camera_motion=SimpleNamespace(poses=(pose(0, 1), pose(1, 2))),
        )
        scene = _relative_static_scene(manifest)
        self.assertEqual(scene["summary"]["component_count"], 1)
        self.assertEqual(scene["summary"]["static_landmark_count"], 1)
        self.assertEqual(scene["summary"]["consistent_static_landmark_count"], 1)
        centers = [
            row["camera_center_world"]
            for row in scene["ego_components"][0]["poses"]
        ]
        self.assertTrue(np.allclose(centers, ((0, 0, 0), (0, 0, 1), (0, 0, 2))))
        landmark = scene["static_landmarks"][0]
        self.assertTrue(np.allclose(landmark["median_world_position"], (2.0, 0.5, 10.0)))
        self.assertEqual(landmark["static_consistency"], "supported")

    def test_step4_emits_relative_geometry_and_honors_depth_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "dataset" / "videos"
            videos.mkdir(parents=True)
            _video(videos / "tiny.avi")
            step1 = run_step1(
                output_root=root / "output",
                dataset_root=root / "dataset",
                video_ids=["tiny"],
                canonical_fps=3.0,
                decode_validation_mode="sample",
                decode_sample_count=3,
            )
            step2 = run_step2(
                init_bundle_path=step1.bundle_path,
                object_backend=_Objects(),
                object_classes=["car"],
                primary_confidence=0.3,
                candidate_confidence=0.05,
                nms_iou=0.7,
                inference_size=640,
                batch_size=2,
                device="cpu",
                mask_backend=_Masks(),
                flow_backend=_Flow(),
                depth_backend=_Depth(),
            )
            step3 = run_step3(
                neural_evidence_store_path=step2.store_path,
                max_age_frames=2,
                minimum_assignment_score=0.25,
                evidence_policy_seed=1,
                depth_check_fraction=0.5,
            )
            step4 = run_step4(
                tracking_store_path=step3.store_path,
                camera_fx_px=50.0,
                camera_fy_px=50.0,
                minimum_support_pixels=16,
                background_flow_sample_stride=8,
                minimum_pose_correspondences=16,
            )

            self.assertTrue(step4.store_path.is_file())
            restored_store = read_contract(step4.store_path, GeometryStore)
            self.assertEqual(restored_store, step4.store)
            reference = restored_store.video_geometry[0]
            manifest_path = step4.stage_root / reference.relative_path
            self.assertEqual(sha256_file(manifest_path), reference.sha256)
            manifest = read_contract(manifest_path, VideoGeometryManifest)
            self.assertEqual(manifest.video_id, "tiny")
            self.assertEqual(manifest.intrinsics.source, "provided_cli")
            self.assertFalse(manifest.intrinsics.validated)
            self.assertIn(
                manifest.camera_motion.observability,
                {Observability.RELATIVE, Observability.UNOBSERVABLE},
            )
            scale = manifest.scale_hypotheses[0]
            self.assertEqual(scale.observability, Observability.RELATIVE)
            self.assertIsNone(scale.scale_to_meters)
            self.assertIsNone(scale.scale_interval_to_meters)

            available = [
                observation
                for track in manifest.tracks
                for observation in track.observations
            ]
            unavailable = [
                observation
                for track in manifest.tracks
                for observation in track.unavailable_observations
            ]
            self.assertTrue(available)
            self.assertTrue(
                any(row.reason == "depth_reserved_as_check_only" for row in unavailable)
            )
            self.assertTrue(
                all(row.coordinate_space == CoordinateSpace.CAMERA_3D for row in available)
            )
            self.assertTrue(all(row.depth_representation == DepthRepresentation.RELATIVE for row in available))
            self.assertTrue(all(row.validation_passed for row in available))
            self.assertEqual(
                manifest.validation.requested_observations,
                len(available) + len(unavailable),
            )

            visualization_path = render_step4_visualizations(
                geometry_store_path=step4.store_path,
                example_frame_count=4,
                maximum_tracks=5,
                render_video=False,
            )
            self.assertTrue(visualization_path.is_file())
            visualization = json.loads(visualization_path.read_text(encoding="utf-8"))
            self.assertEqual(visualization["schema_name"], "step4_visualization_manifest")
            row = visualization["videos"][0]
            self.assertFalse(row["world_trajectory_claimed"])
            self.assertEqual(len(row["frame_paths"]), manifest.frame_count)
            self.assertIsNone(row["video"])
            self.assertTrue(row["depth_geometry_examples"])
            visual_root = visualization_path.parent
            for key in (
                "contact_sheet",
                "camera_centric_points_3d",
                "geometry_timeline",
                "camera_motion_diagnostics",
                "relative_static_scene",
                "relative_static_sandbox_3d",
            ):
                self.assertTrue((visual_root / row[key]).is_file())
            static_scene = json.loads(
                (visual_root / row["relative_static_scene"]).read_text(encoding="utf-8")
            )
            self.assertFalse(static_scene["metric_scale_claimed"])
            self.assertFalse(static_scene["world_trajectory_claimed"])
            self.assertEqual(
                static_scene["coordinate_unit"],
                "normalized_relative_translation_step",
            )
            self.assertEqual(
                len(row["relative_static_sandbox_components"]),
                static_scene["summary"]["component_count"],
            )
            self.assertTrue(
                all(
                    (visual_root / path).is_file()
                    for path in row["relative_static_sandbox_components"]
                )
            )

            step5 = run_step5(geometry_store_path=step4.store_path, top_k=3)
            restored_world_store = read_contract(step5.store_path, WorldStateStore)
            self.assertEqual(restored_world_store, step5.store)
            world_reference = restored_world_store.video_world_states[0]
            world_manifest_path = step5.stage_root / world_reference.relative_path
            world_manifest = read_contract(world_manifest_path, VideoWorldStateManifest)
            self.assertEqual(world_manifest.video_id, "tiny")
            self.assertTrue(world_manifest.initial_beam.hypotheses)
            self.assertFalse(
                world_manifest.initial_beam.hypotheses[0].metric_scale_claimed
            )
            step5_visualization_path = render_step5_visualizations(
                world_state_store_path=step5.store_path,
                maximum_objects=5,
            )
            self.assertTrue(step5_visualization_path.is_file())
            step5_visualization = json.loads(
                step5_visualization_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                step5_visualization["schema_name"],
                "step5_visualization_manifest",
            )
            step5_row = step5_visualization["videos"][0]
            for key in ("world_3d", "summary"):
                self.assertTrue(
                    (step5_visualization_path.parent / step5_row[key]).is_file()
                )
            self.assertTrue(step5_row["motion_intervals"])
            self.assertTrue(
                all(
                    (step5_visualization_path.parent / path).is_file()
                    for path in step5_row["motion_intervals"]
                )
            )

            step6 = run_step6(world_state_store_path=step5.store_path)
            restored_residual_store = read_contract(step6.store_path, ResidualStore)
            self.assertEqual(restored_residual_store, step6.store)
            residual_manifest = step6.video_manifests[0]
            self.assertEqual(
                len(residual_manifest.packets),
                len(world_manifest.initial_beam.hypotheses),
            )
            self.assertTrue(residual_manifest.validation.overall_pass)
            for packet in residual_manifest.packets:
                self.assertFalse(packet.repair_applied)
                self.assertFalse(packet.selection_applied)
                self.assertEqual(
                    tuple(row.family for row in packet.family_summaries),
                    tuple(ResidualFamily),
                )
                self.assertTrue(packet.residuals)
                self.assertTrue(
                    all(
                        row.evidence_role is not None and row.evidence_artifacts
                        for row in packet.residuals
                        if row.evaluation_basis == EvaluationBasis.CHECK_EVIDENCE
                    )
                )
            step6_visualization_path = render_step6_visualizations(
                residual_store_path=step6.store_path,
                maximum_hypotheses=2,
            )
            self.assertTrue(step6_visualization_path.is_file())
            step6_visualization = json.loads(
                step6_visualization_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                step6_visualization["schema_name"],
                "step6_visualization_manifest",
            )
            self.assertFalse(step6_visualization["selection_applied"])
            self.assertFalse(step6_visualization["repair_applied"])
            step6_visual_root = step6_visualization_path.parent
            step6_video_row = step6_visualization["videos"][0]
            self.assertTrue(
                (step6_visual_root / step6_video_row["hypothesis_comparison"]).is_file()
            )
            for packet_row in step6_video_row["packets"]:
                self.assertTrue(
                    (step6_visual_root / packet_row["conflict_audit"]).is_file()
                )
                if packet_row["conflict_overview"] is not None:
                    overview_path = step6_visual_root / packet_row["conflict_overview"]
                    self.assertTrue(overview_path.is_file())
                    overview_image = cv2.imread(str(overview_path))
                    self.assertEqual(overview_image.shape[:2], (2160, 3840))
                self.assertTrue(
                    all(
                        (step6_visual_root / path).is_file()
                        for path in packet_row["conflict_panels"]
                    )
                )
                for path in packet_row["conflict_panels"]:
                    panel_image = cv2.imread(str(step6_visual_root / path))
                    self.assertEqual(panel_image.shape[:2], (1080, 1920))

            world_store_sha256_before_step7 = sha256_file(step5.store_path)
            step7 = run_step7(residual_store_path=step6.store_path)
            self.assertEqual(
                sha256_file(step5.store_path), world_store_sha256_before_step7
            )
            restored_repair_store = read_contract(
                step7.store_path, RepairProposalStore
            )
            self.assertEqual(restored_repair_store, step7.store)
            repair_reference = restored_repair_store.video_repair_proposals[0]
            repair_manifest_path = step7.stage_root / repair_reference.relative_path
            repair_manifest = read_contract(
                repair_manifest_path, VideoRepairProposalManifest
            )
            self.assertEqual(repair_manifest.video_id, "tiny")
            self.assertTrue(repair_manifest.validation.overall_pass)
            self.assertEqual(
                len(repair_manifest.packets), len(residual_manifest.packets)
            )
            self.assertTrue(
                all(not row.world_state_mutated for row in repair_manifest.packets)
            )
            self.assertTrue(
                all(
                    not effect.optimized_by_step8
                    for packet in repair_manifest.packets
                    for proposal in packet.proposals
                    for effect in proposal.expected_residual_effects
                    if effect.evaluation_basis
                    in {
                        EvaluationBasis.CHECK_EVIDENCE,
                        EvaluationBasis.NOT_EVALUABLE,
                    }
                )
            )
            step7_visualization_path = render_step7_visualizations(
                repair_proposal_store_path=step7.store_path,
                maximum_hypotheses=2,
                maximum_proposal_panels=2,
            )
            self.assertTrue(step7_visualization_path.is_file())
            step7_visualization = json.loads(
                step7_visualization_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                step7_visualization["schema_name"],
                "step7_visualization_manifest",
            )
            self.assertFalse(step7_visualization["world_state_mutated"])
            self.assertFalse(step7_visualization["selection_applied"])
            self.assertFalse(step7_visualization["check_evidence_optimized"])
            step7_visual_root = step7_visualization_path.parent
            step7_video_row = step7_visualization["videos"][0]
            self.assertTrue(
                (
                    step7_visual_root
                    / step7_video_row["diagnosis_operator_summary"]
                ).is_file()
            )
            for packet_row in step7_video_row["packets"]:
                self.assertTrue(
                    (step7_visual_root / packet_row["timeline"]).is_file()
                )
                self.assertTrue(
                    (step7_visual_root / packet_row["proposal_audit"]).is_file()
                )
                for proposal_number, path in enumerate(
                    packet_row["proposal_panels"], start=1
                ):
                    self.assertIn(f"proposal_{proposal_number:02d}_", Path(path).name)
                    panel_image = cv2.imread(str(step7_visual_root / path))
                    panel_width, panel_height = packet_row[
                        "proposal_panel_resolution"
                    ]
                    self.assertEqual(
                        panel_image.shape[:2],
                        (panel_height, panel_width),
                    )
                self.assertIsNone(packet_row["proposal_overview"])
                self.assertEqual(packet_row["proposal_overview_resolution"], [0, 0])

            world_store_sha256_before_step8 = sha256_file(step5.store_path)
            world_manifest_sha256_before_step8 = sha256_file(world_manifest_path)
            step8 = run_step8(
                repair_proposal_store_path=step7.store_path,
                maximum_candidates_per_proposal=2,
            )
            self.assertEqual(
                sha256_file(step5.store_path), world_store_sha256_before_step8
            )
            self.assertEqual(
                sha256_file(world_manifest_path), world_manifest_sha256_before_step8
            )
            restored_reestimation_store = read_contract(
                step8.store_path, LocalReestimationStore
            )
            self.assertEqual(restored_reestimation_store, step8.store)
            reestimation_reference = (
                restored_reestimation_store.video_local_reestimations[0]
            )
            reestimation_manifest_path = (
                step8.stage_root / reestimation_reference.relative_path
            )
            reestimation_manifest = read_contract(
                reestimation_manifest_path, VideoLocalReestimationManifest
            )
            self.assertEqual(reestimation_manifest.video_id, "tiny")
            self.assertTrue(reestimation_manifest.validation.overall_pass)
            self.assertEqual(
                reestimation_manifest.validation.check_evidence_optimization_violations,
                0,
            )
            self.assertEqual(reestimation_manifest.validation.parent_mutation_count, 0)
            self.assertEqual(reestimation_manifest.validation.selection_count, 0)
            self.assertTrue(
                all(not packet.parent_mutated for packet in reestimation_manifest.packets)
            )
            for packet in reestimation_manifest.packets:
                for result in packet.proposal_results:
                    for candidate in result.candidates:
                        self.assertFalse(candidate.raw_evidence_mutated)
                        self.assertFalse(candidate.selection_applied)
                        self.assertTrue(
                            set(candidate.optimized_residual_ids).isdisjoint(
                                candidate.excluded_check_residual_ids
                            )
                        )
                        if candidate.status == "instantiated":
                            self.assertNotEqual(
                                candidate.child_hypothesis.hypothesis_id,
                                candidate.parent_hypothesis_id,
                            )
                            self.assertTrue(
                                candidate.numerical_changes
                                or candidate.discrete_changes
                            )
                            self.assertTrue(candidate.boundary_preserved)
                            self.assertTrue(candidate.parameter_bounds_satisfied)
                            self.assertTrue(candidate.compute_budget_honored)
            step8_visualization_path = render_step8_visualizations(
                local_reestimation_store_path=step8.store_path,
                maximum_hypotheses=2,
                maximum_proposal_panels=2,
            )
            self.assertTrue(step8_visualization_path.is_file())
            step8_visualization = json.loads(
                step8_visualization_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                step8_visualization["schema_name"],
                "step8_visualization_manifest",
            )
            self.assertFalse(step8_visualization["parent_state_mutated"])
            self.assertFalse(step8_visualization["raw_evidence_mutated"])
            self.assertFalse(step8_visualization["selection_applied"])
            self.assertFalse(step8_visualization["check_evidence_optimized"])
            for packet_row in step8_visualization["videos"][0]["packets"]:
                self.assertIsNone(packet_row["proposal_overview"])
                self.assertEqual(packet_row["proposal_overview_resolution"], [0, 0])
                self.assertEqual(
                    packet_row["proposal_panel_resolution"], [1920, 1220]
                )
                for proposal_row in packet_row["proposals"]:
                    panel_path = (
                        step8_visualization_path.parent / proposal_row["panel"]
                    )
                    self.assertTrue(panel_path.is_file())
                    panel = cv2.imread(str(panel_path))
                    panel_width, panel_height = packet_row[
                        "proposal_panel_resolution"
                    ]
                    self.assertEqual(panel.shape[:2], (panel_height, panel_width))
            repair_manifest_path.write_bytes(
                repair_manifest_path.read_bytes() + b" "
            )
            with self.assertRaisesRegex(
                RuntimeError, "missing or truncated|integrity check"
            ):
                run_step8(repair_proposal_store_path=step7.store_path)

    def test_step4_rejects_tampered_tracking_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "dataset" / "videos"
            videos.mkdir(parents=True)
            _video(videos / "tiny.avi")
            step1 = run_step1(
                output_root=root / "output",
                dataset_root=root / "dataset",
                video_ids=["tiny"],
                canonical_fps=3.0,
                decode_validation_mode="sample",
                decode_sample_count=3,
            )
            step2 = run_step2(
                init_bundle_path=step1.bundle_path,
                object_backend=_Objects(),
                object_classes=["car"],
                primary_confidence=0.3,
                candidate_confidence=0.05,
                nms_iou=0.7,
                inference_size=640,
                batch_size=2,
                device="cpu",
            )
            step3 = run_step3(neural_evidence_store_path=step2.store_path)
            tracking_manifest = step3.stage_root / step3.store.video_tracking[0].relative_path
            tracking_manifest.write_bytes(tracking_manifest.read_bytes() + b" ")
            with self.assertRaisesRegex(RuntimeError, "missing or truncated|integrity check"):
                run_step4(tracking_store_path=step3.store_path)


if __name__ == "__main__":
    unittest.main()
