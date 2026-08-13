import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.exp_august.contracts import (
    CueStatus,
    DepthRepresentation,
    DetectionTier,
    FlowDirection,
    NeuralEvidenceStore,
    TrackingStore,
    VideoEvidenceManifest,
    VideoTrackingManifest,
)
from src.exp_august.contracts.codec import read_contract, sha256_file
from src.exp_august.inference.depth_backend import DepthFrameOutput
from src.exp_august.inference.flow_backend import (
    DirectionalFlowOutput,
    FlowPairOutput,
)
from src.exp_august.inference.frames import CanonicalFrameProvider
from src.exp_august.inference.mask_backend import MaskCandidateOutput
from src.exp_august.inference.step01_init import run_step1
from src.exp_august.inference.step03_object_tracking import run_step3
from src.exp_august.inference.step03_visualization import render_step3_visualizations
from src.exp_august.inference.step02_neural_evidence import (
    ObjectCandidate,
    run_step2,
)


class _FakeObjectBackend:
    backend_name = "fake_objects"
    model_name = "fake-model"
    model_id = "fake-model@v1"
    available = True
    unavailable_reason = None
    tool_versions = ()

    def __init__(self):
        self.warmed = False
        self.closed = False

    def warmup(self):
        self.warmed = True

    def predict_batch(self, frames):
        if not self.warmed:
            raise RuntimeError("backend was not warmed")
        return tuple(
            (
                ObjectCandidate(
                    bbox_xyxy=(1.0, 2.0, 20.0, 30.0),
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
        self.closed = True


class _FakeMaskBackend:
    backend_name = "fake_masks"
    model_name = "fake-mask-model"
    model_id = "fake-mask-model@v1"
    available = True
    unavailable_reason = None
    prompt_candidates = False
    tool_versions = ()

    def __init__(self):
        self.closed = False

    def warmup(self):
        return None

    def predict_frame(self, frame, detections):
        outputs = []
        height, width = frame.image_bgr.shape[:2]
        for detection in detections:
            mask = np.zeros((height, width), dtype=bool)
            box = detection.bbox
            x1, y1 = int(np.floor(box.x1)), int(np.floor(box.y1))
            x2, y2 = int(np.ceil(box.x2)), int(np.ceil(box.y2))
            mask[y1:y2, x1:x2] = True
            outputs.append(
                MaskCandidateOutput(
                    prompt_detection_id=detection.detection_id,
                    mask=mask,
                    confidence=0.8,
                )
            )
        return tuple(outputs)

    def teardown(self):
        self.closed = True


class _FakeFlowBackend:
    backend_name = "fake_bidirectional_flow"
    model_name = "fake-flow-model"
    model_id = "fake-flow-model@v1"
    available = True
    unavailable_reason = None
    consistency_threshold_px = 1.5
    tool_versions = ()

    def __init__(self):
        self.closed = False

    def warmup(self):
        return None

    def predict_pair(self, earlier, later):
        height, width = earlier.image_bgr.shape[:2]
        valid = np.ones((height, width), dtype=bool)
        error = np.zeros((height, width), dtype=np.float32)

        def directional(delta_x):
            flow = np.zeros((height, width, 2), dtype=np.float32)
            flow[:, :, 0] = delta_x
            return DirectionalFlowOutput(
                flow=flow,
                domain_valid=valid,
                consistency_valid=valid,
                fb_error=error,
            )

        return FlowPairOutput(forward=directional(1.0), backward=directional(-1.0))

    def teardown(self):
        self.closed = True


class _FakeDepthBackend:
    backend_name = "fake_relative_depth"
    model_name = "fake-depth-model"
    model_id = "fake-depth-model@v1"
    available = True
    unavailable_reason = None
    process_resolution = 64
    representation = DepthRepresentation.RELATIVE
    tool_versions = ()

    def __init__(self):
        self.closed = False

    def warmup(self):
        return None

    def predict_frame(self, frame):
        height, width = frame.image_bgr.shape[:2]
        rows, columns = np.indices((height, width), dtype=np.float32)
        depth = 1.0 + frame.frame_index + rows / height + columns / width
        return DepthFrameOutput(
            depth=depth.astype(np.float32),
            valid=np.ones((height, width), dtype=bool),
            confidence=np.full((height, width), 0.75, dtype=np.float32),
            representation=self.representation,
        )

    def teardown(self):
        self.closed = True


def _write_tiny_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        6.0,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer could not open")
    for index in range(12):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:, :, 1] = index * 10
        writer.write(image)
    writer.release()


class ExpAugustStep02EvidenceTests(unittest.TestCase):
    def test_provider_and_step2_preserve_canonical_timeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_dir = root / "dataset" / "videos"
            video_dir.mkdir(parents=True)
            video_path = video_dir / "tiny.avi"
            _write_tiny_video(video_path)
            step1 = run_step1(
                output_root=root / "output",
                dataset_root=root / "dataset",
                video_ids=["tiny"],
                canonical_fps=3.0,
                decode_validation_mode="sample",
                decode_sample_count=3,
            )

            provider = CanonicalFrameProvider(step1.manifests[0])
            frames = tuple(provider.iter_frames())
            self.assertEqual(len(frames), 6)
            self.assertEqual([frame.frame_index for frame in frames], list(range(6)))
            self.assertEqual(
                [frame.source_frame_index for frame in frames],
                [0, 2, 4, 6, 8, 10],
            )
            self.assertTrue(all(frame.image_bgr.shape == (48, 64, 3) for frame in frames))

            backend = _FakeObjectBackend()
            step2 = run_step2(
                init_bundle_path=step1.bundle_path,
                object_backend=backend,
                object_classes=["car"],
                primary_confidence=0.3,
                candidate_confidence=0.05,
                nms_iou=0.7,
                inference_size=640,
                batch_size=2,
                device="cpu",
            )
            self.assertTrue(backend.closed)
            self.assertEqual(step2.store.video_ids, ("tiny",))
            evidence = step2.video_manifests[0]
            self.assertEqual(evidence.frame_count, 6)
            self.assertEqual(
                [frame.object_cue.status for frame in evidence.frames],
                [
                    CueStatus.AVAILABLE,
                    CueStatus.EMPTY,
                    CueStatus.AVAILABLE,
                    CueStatus.EMPTY,
                    CueStatus.AVAILABLE,
                    CueStatus.EMPTY,
                ],
            )
            self.assertEqual(
                evidence.frames[0].backward_flow_cue.status,
                CueStatus.NOT_APPLICABLE,
            )
            self.assertEqual(
                evidence.frames[-1].forward_flow_cue.status,
                CueStatus.NOT_APPLICABLE,
            )
            self.assertTrue(
                all(frame.mask_cue.status == CueStatus.UNAVAILABLE for frame in evidence.frames)
            )
            self.assertTrue(
                all(frame.depth_cue.status == CueStatus.UNAVAILABLE for frame in evidence.frames)
            )

            restored_store = read_contract(step2.store_path, NeuralEvidenceStore)
            self.assertEqual(restored_store, step2.store)
            evidence_ref = restored_store.video_evidence[0]
            evidence_path = step2.stage_root / evidence_ref.relative_path
            self.assertEqual(sha256_file(evidence_path), evidence_ref.sha256)
            restored_evidence = read_contract(evidence_path, VideoEvidenceManifest)
            self.assertEqual(restored_evidence, evidence)
            serialized = evidence_path.read_text(encoding="utf-8")
            self.assertNotIn("image_bgr", serialized)

            partial_step3 = run_step3(
                neural_evidence_store_path=step2.store_path,
                max_age_frames=2,
                minimum_assignment_score=0.25,
            )
            partial_package = partial_step3.video_manifests[0]
            self.assertTrue(partial_package.retention_report.overall_pass)
            self.assertEqual(len(partial_package.tracks), 1)
            self.assertTrue(
                all(
                    observation.selected_mask_candidate_id is not None
                    for observation in partial_package.tracks[0].observations
                )
            )
            self.assertTrue(
                all(
                    candidate.source.value == "explicit_unobservable"
                    for candidate in partial_package.mask_candidate_bank
                )
            )

    def test_step2_persists_typed_dense_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_dir = root / "dataset" / "videos"
            video_dir.mkdir(parents=True)
            _write_tiny_video(video_dir / "tiny.avi")
            step1 = run_step1(
                output_root=root / "output",
                dataset_root=root / "dataset",
                video_ids=["tiny"],
                canonical_fps=3.0,
                decode_validation_mode="sample",
                decode_sample_count=3,
            )

            objects = _FakeObjectBackend()
            masks = _FakeMaskBackend()
            flow = _FakeFlowBackend()
            depth = _FakeDepthBackend()
            step2 = run_step2(
                init_bundle_path=step1.bundle_path,
                object_backend=objects,
                object_classes=["car"],
                primary_confidence=0.3,
                candidate_confidence=0.05,
                nms_iou=0.7,
                inference_size=640,
                batch_size=2,
                device="cpu",
                mask_backend=masks,
                flow_backend=flow,
                depth_backend=depth,
            )
            self.assertTrue(objects.closed)
            self.assertTrue(masks.closed)
            self.assertTrue(flow.closed)
            self.assertTrue(depth.closed)

            evidence = step2.video_manifests[0]
            self.assertEqual(evidence.frame_count, 6)
            self.assertEqual(
                [frame.mask_cue.status for frame in evidence.frames],
                [
                    CueStatus.AVAILABLE,
                    CueStatus.NOT_APPLICABLE,
                    CueStatus.AVAILABLE,
                    CueStatus.NOT_APPLICABLE,
                    CueStatus.AVAILABLE,
                    CueStatus.NOT_APPLICABLE,
                ],
            )
            self.assertEqual(
                [len(frame.masks) for frame in evidence.frames],
                [1, 0, 1, 0, 1, 0],
            )
            self.assertEqual(
                [frame.forward_flow is not None for frame in evidence.frames],
                [True, True, True, True, True, False],
            )
            self.assertEqual(
                [frame.backward_flow is not None for frame in evidence.frames],
                [False, True, True, True, True, True],
            )
            self.assertTrue(all(frame.depth is not None for frame in evidence.frames))
            self.assertTrue(
                all(
                    frame.depth.representation == DepthRepresentation.RELATIVE
                    for frame in evidence.frames
                )
            )

            first_mask = evidence.frames[0].masks[0]
            self.assertEqual(first_mask.area_pixels, 19 * 28)
            mask_path = step2.stage_root / first_mask.mask_ref.relative_path
            self.assertEqual(sha256_file(mask_path), first_mask.mask_ref.sha256)
            persisted_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(int(np.count_nonzero(persisted_mask)), first_mask.area_pixels)

            forward = evidence.frames[0].forward_flow
            self.assertEqual(forward.direction, FlowDirection.FORWARD)
            self.assertEqual(forward.source_frame_index, 0)
            self.assertEqual(forward.target_frame_index, 1)
            flow_path = step2.stage_root / forward.field_ref.relative_path
            self.assertEqual(sha256_file(flow_path), forward.field_ref.sha256)
            with np.load(flow_path) as persisted_flow:
                self.assertEqual(
                    set(persisted_flow.files),
                    {"flow", "domain_valid", "consistency_valid", "fb_error"},
                )
                self.assertEqual(persisted_flow["flow"].shape, (48, 64, 2))
                self.assertTrue(np.allclose(persisted_flow["flow"][:, :, 0], 1.0))

            first_depth = evidence.frames[0].depth
            depth_path = step2.stage_root / first_depth.field_ref.relative_path
            self.assertEqual(sha256_file(depth_path), first_depth.field_ref.sha256)
            with np.load(depth_path) as persisted_depth:
                self.assertEqual(
                    set(persisted_depth.files),
                    {"depth", "valid", "confidence"},
                )
                self.assertEqual(persisted_depth["depth"].shape, (48, 64))
                self.assertTrue(np.all(persisted_depth["valid"] == 1))

    def test_step3_builds_replayable_tracks_across_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_dir = root / "dataset" / "videos"
            video_dir.mkdir(parents=True)
            _write_tiny_video(video_dir / "tiny.avi")
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
                object_backend=_FakeObjectBackend(),
                object_classes=["car"],
                primary_confidence=0.3,
                candidate_confidence=0.05,
                nms_iou=0.7,
                inference_size=640,
                batch_size=2,
                device="cpu",
                mask_backend=_FakeMaskBackend(),
                flow_backend=_FakeFlowBackend(),
                depth_backend=_FakeDepthBackend(),
            )

            step3 = run_step3(
                neural_evidence_store_path=step2.store_path,
                max_age_frames=2,
                minimum_assignment_score=0.25,
                evidence_policy_seed=7,
            )
            package = step3.video_manifests[0]
            self.assertTrue(package.retention_report.overall_pass)
            self.assertEqual(package.retention_report.expected_candidate_pairs, 2)
            self.assertEqual(package.retention_report.ledger_rows, 2)
            self.assertEqual(len(package.tracks), 1)
            track = package.tracks[0]
            self.assertEqual(
                [observation.frame_index for observation in track.observations],
                [0, 2, 4],
            )
            self.assertEqual(
                [marker.marker_type.value for marker in track.state_markers],
                [
                    "first_observed",
                    "missed",
                    "reobserved",
                    "missed",
                    "reobserved",
                    "missed",
                    "video_end",
                ],
            )
            self.assertEqual(len(package.gap_records), 3)
            self.assertEqual(
                [gap.status.value for gap in package.gap_records],
                ["reobserved", "reobserved", "video_end"],
            )
            sources = [candidate.source.value for candidate in package.mask_candidate_bank]
            self.assertIn("direct_instance", sources)
            self.assertIn("flow_forward", sources)
            self.assertIn("flow_backward", sources)
            self.assertTrue(package.derived_artifacts)
            self.assertTrue(
                all(row.decision.value == "matched" for row in package.association_ledger)
            )
            self.assertTrue(
                any(
                    assignment.role.value == "check_only"
                    and assignment.cue_family.value == "flow_backward"
                    for assignment in package.evidence_use_plan.assignments
                )
            )

            restored_store = read_contract(step3.store_path, TrackingStore)
            self.assertEqual(restored_store, step3.store)
            package_ref = restored_store.video_tracking[0]
            package_path = step3.stage_root / package_ref.relative_path
            self.assertEqual(sha256_file(package_path), package_ref.sha256)
            restored_package = read_contract(package_path, VideoTrackingManifest)
            self.assertEqual(restored_package, package)
            for link in package.derived_artifacts:
                artifact_path = step3.stage_root / link.artifact.relative_path
                self.assertEqual(sha256_file(artifact_path), link.artifact.sha256)

            visualization_manifest = render_step3_visualizations(
                tracking_store_path=step3.store_path,
                example_frame_count=4,
                render_video=False,
            )
            self.assertTrue(visualization_manifest.is_file())
            visualization_root = visualization_manifest.parent / "tiny"
            self.assertTrue((visualization_root / "tiny_step3_examples.png").is_file())
            self.assertEqual(
                len(tuple((visualization_root / "frames").glob("frame_*.png"))),
                package.frame_count,
            )

            source_manifest_path = (
                step1.bundle_path.parent
                / step1.bundle.video_manifests[0].relative_path
            )
            source_manifest_path.write_bytes(source_manifest_path.read_bytes() + b" ")
            with self.assertRaisesRegex(RuntimeError, "retention gate failed"):
                run_step3(neural_evidence_store_path=step2.store_path)


if __name__ == "__main__":
    unittest.main()
