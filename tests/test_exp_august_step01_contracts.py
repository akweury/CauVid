import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from src.exp_august.contracts import (
    ArtifactRef,
    DecodeValidationMode,
    InitBundle,
    Step1ConfigSnapshot,
)
from src.exp_august.contracts.codec import read_contract, sha256_file
from src.exp_august.inference.step01_init import (
    VideoValidationError,
    _VideoProbe,
    _normalized_frame_mapping,
    run_step1,
)
from src.exp_august.contracts import TimelineSource


SHA256 = "a" * 64


class ExpAugustStep01ContractTests(unittest.TestCase):
    def test_contracts_are_frozen_and_reject_unknown_fields(self):
        config = Step1ConfigSnapshot(
            dataset_name="unit-test",
            canonical_fps=10.0,
            decode_validation_mode=DecodeValidationMode.SAMPLE,
            decode_sample_count=3,
            random_seed=7,
        )
        with self.assertRaises(ValidationError):
            config.canonical_fps = 5.0
        with self.assertRaises(ValidationError):
            Step1ConfigSnapshot(
                dataset_name="unit-test",
                canonical_fps=10.0,
                decode_validation_mode=DecodeValidationMode.SAMPLE,
                decode_sample_count=3,
                random_seed=7,
                annotation_path="forbidden.json",
            )

    def test_artifact_references_cannot_escape_stage_root(self):
        with self.assertRaises(ValidationError):
            ArtifactRef(
                artifact_id="bad",
                relative_path="../outside.json",
                sha256=SHA256,
                byte_size=1,
                media_type="application/json",
            )

    def test_timeline_normalization_is_monotonic_and_reversible(self):
        probe = _VideoProbe(
            width=64,
            height=48,
            display_rotation_degrees_clockwise=0,
            fps=6.0,
            duration_s=2.0,
            frame_count=12,
            timestamps_s=tuple(index / 6.0 for index in range(12)),
            timeline_source=TimelineSource.CONTAINER_PTS,
            codec="test",
            backend="unit-test",
            tool_versions=(),
        )
        mapping = _normalized_frame_mapping(probe, 3.0)
        self.assertEqual(len(mapping), 6)
        self.assertEqual([row[0] for row in mapping], [0, 2, 4, 6, 8, 10])
        self.assertEqual([row[1] for row in mapping], [index / 3.0 for index in range(6)])
        self.assertTrue(all(row[3] < 1e-9 for row in mapping))
        with self.assertRaises(VideoValidationError):
            _normalized_frame_mapping(probe, 12.0)

    def test_real_video_produces_round_trippable_manifest_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            video_dir = dataset_root / "videos"
            video_dir.mkdir(parents=True)
            video_path = video_dir / "tiny.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                6.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for index in range(12):
                frame = np.full((48, 64, 3), index * 10, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            # A label file is deliberately present; Step 1 has no parameter or
            # contract field through which it could enter inference.
            (dataset_root / "labels.csv").write_text("annotation,not,used\n", encoding="utf-8")
            result = run_step1(
                output_root=root / "output",
                dataset_root=dataset_root,
                dataset_name="synthetic",
                video_ids=["tiny"],
                canonical_fps=3.0,
                decode_validation_mode="sample",
                decode_sample_count=3,
                random_seed=11,
            )

            self.assertEqual(result.bundle.video_ids, ("tiny",))
            self.assertEqual(len(result.manifests), 1)
            manifest = result.manifests[0]
            self.assertEqual(manifest.canonical_frame_count, 6)
            self.assertEqual(manifest.image_size.width, 64)
            self.assertEqual(manifest.image_size.height, 48)
            self.assertEqual(len(manifest.decode_validation.checked_frame_indices), 3)
            self.assertEqual(manifest.input_sha256, sha256_file(video_path))

            restored_bundle = read_contract(result.bundle_path, InitBundle)
            self.assertEqual(restored_bundle, result.bundle)
            manifest_ref = restored_bundle.video_manifests[0]
            manifest_path = result.bundle_path.parent / manifest_ref.relative_path
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(sha256_file(manifest_path), manifest_ref.sha256)

    def test_contract_json_schema_forbids_additional_fields(self):
        schema = InitBundle.model_json_schema()
        self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
