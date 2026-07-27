import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.exp_july.perception.pipeline import step7c_video_local_evidence_calibration
from src.exp_july.perception.video_local_evidence_calibration import calibrate_video


def _vector(index, magnitude, state, region):
    radial = magnitude * (0.8 if state == "expansion" else -0.8 if state == "contraction" else 0.0)
    return {
        "patch_id": index,
        "region_id": region,
        "magnitude": magnitude,
        "radial_projection": radial,
        "radial_state": state,
        "forward_backward_error": magnitude * 0.04,
        "local_vector_residual": magnitude * 0.08,
    }


def _raw_video(scale=1.0):
    segments = []
    specs = [
        (0, [1.0, 1.2, 0.9, 1.1, 1.3, 1.0], "expansion"),
        (1, [2.0, 2.2, 1.8, 2.1, 2.3, 1.9], "contraction"),
    ]
    for segment_id, magnitudes, state in specs:
        vectors = [
            _vector(index, value * scale, state, f"r{index % 2}c{index % 3}")
            for index, value in enumerate(magnitudes)
        ]
        segments.append({
            "segment_id": segment_id,
            "provisional_action": "forward" if segment_id == 0 else "backward",
            "start_frame": segment_id * 5,
            "end_frame": segment_id * 5 + 4,
            "status": "completed",
            "patch_vectors": vectors,
            "frame_pair_evidence": [
                {
                    "start_frame": segment_id * 5,
                    "end_frame": segment_id * 5 + 1,
                    "patch_vectors": vectors[:3],
                },
                {
                    "start_frame": segment_id * 5 + 1,
                    "end_frame": segment_id * 5 + 2,
                    "patch_vectors": vectors[3:],
                },
            ],
            "covered_regions": sorted({row["region_id"] for row in vectors}),
            "spatial_coverage": 6 / 9,
            "temporal_persistence": 0.8,
            "tracking_reliability": 0.75,
        })
    return {
        "version": 1,
        "video_id": "demo",
        "input_label_status": "provisional",
        "segments": segments,
    }


class Step7CVideoLocalCalibrationTests(unittest.TestCase):
    def test_outputs_complete_video_local_normalized_audit(self):
        result = calibrate_video(_raw_video())
        self.assertEqual(result["calibration_scope"], "video_local")
        self.assertFalse(result["dataset_specific_absolute_thresholds_used"])
        self.assertIn("motion_magnitude", result["calibration_statistics"])
        self.assertGreater(
            result["calibration_statistics"]["motion_magnitude"]["robust_scale"], 0
        )
        for segment in result["normalized_segment_evidence"]:
            self.assertIn("normalized_motion_magnitude", segment)
            self.assertIn("motion_magnitude_robust_z", segment)
            self.assertGreaterEqual(segment["direction_support_ratio"], 0.0)
            self.assertLessEqual(segment["direction_support_ratio"], 1.0)
            self.assertAlmostEqual(segment["region_support_ratio"], 6 / 9)
            self.assertAlmostEqual(segment["temporal_persistence"], 0.8)
            self.assertGreater(segment["estimator_agreement"], 0.0)
            self.assertGreaterEqual(segment["uncertainty"], 0.0)
            self.assertLessEqual(segment["uncertainty"], 1.0)
            self.assertFalse(
                segment["provenance"]["dataset_specific_absolute_thresholds_used"]
            )
        self.assertEqual(
            result["audit"]["raw_evidence_preserved"]["segments"],
            _raw_video()["segments"],
        )

    def test_normalization_is_invariant_to_video_wide_motion_units(self):
        baseline = calibrate_video(_raw_video(1.0))
        scaled = calibrate_video(_raw_video(17.0))
        for left, right in zip(
            baseline["normalized_segment_evidence"],
            scaled["normalized_segment_evidence"],
        ):
            self.assertAlmostEqual(
                left["normalized_motion_magnitude"],
                right["normalized_motion_magnitude"],
                places=9,
            )
            self.assertAlmostEqual(
                left["motion_magnitude_robust_z"],
                right["motion_magnitude_robust_z"],
                places=9,
            )
            self.assertEqual(
                left["dominant_radial_direction"],
                right["dominant_radial_direction"],
            )
            self.assertAlmostEqual(
                left["estimator_agreement"], right["estimator_agreement"]
            )

    def test_stage_cache_is_fingerprinted_and_deterministic(self):
        state = {"background_motion_evidence": [_raw_video()]}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.exp_july.perception.pipeline.get_pipeline_output_root",
            return_value=Path(tmp),
        ):
            first = step7c_video_local_evidence_calibration(state)
            second = step7c_video_local_evidence_calibration(copy.deepcopy(state))
        self.assertEqual(
            first["video_local_evidence_calibration_manifest"]["cached_videos"], 0
        )
        self.assertEqual(
            second["video_local_evidence_calibration_manifest"]["cached_videos"], 1
        )
        self.assertEqual(
            first["video_local_calibrated_evidence"],
            second["video_local_calibrated_evidence"],
        )


if __name__ == "__main__":
    unittest.main()
