import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.exp_july.perception.background_motion_evidence import extract_video_evidence
from src.exp_july.perception.pipeline import step7b_background_motion_evidence


def _inputs(root):
    rng = np.random.default_rng(17)
    first = rng.integers(0, 256, size=(180, 240), dtype=np.uint8)
    first = cv2.GaussianBlur(first, (3, 3), 0)
    transform = cv2.getRotationMatrix2D((120, 90), 0.0, 1.025)
    second = cv2.warpAffine(first, transform, (240, 180))
    paths = []
    for index, image in enumerate((first, second)):
        path = root / f"frame_{index:05d}.png"
        cv2.imwrite(str(path), image)
        paths.append(str(path))
    position_video = {
        "video_id": "demo",
        "frames": [
            {
                "frame_index": index,
                "image_path": path,
                "objects": [{"bbox": [95, 65, 145, 115]}],
            }
            for index, path in enumerate(paths)
        ],
    }
    provisional = {
        "video_id": "demo",
        "label_status": "provisional",
        "final_action_segments": [
            {
                "segment_id": 0,
                "action": "forward",
                "start_frame": 0,
                "end_frame": 1,
                "duration_frames": 2,
            }
        ],
    }
    return position_video, provisional


class Step7BBackgroundMotionEvidenceTests(unittest.TestCase):
    def test_extracts_independent_radial_multiregion_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            position_video, provisional = _inputs(Path(tmp))
            result = extract_video_evidence(position_video, provisional)
            self.assertEqual(result["input_label_status"], "provisional")
            segment = result["segments"][0]
            self.assertGreater(segment["num_accepted_vectors"], 4)
            self.assertGreater(segment["radial_expansion_support"], 0.5)
            self.assertGreater(segment["spatial_coverage"], 0.3)
            self.assertGreater(segment["temporal_persistence"], 0.0)
            self.assertGreater(segment["tracking_reliability"], 0.0)
            self.assertGreater(segment["estimator_confidence"], 0.0)
            for vector in segment["patch_vectors"]:
                x, y = vector["start_xy"]
                self.assertFalse(87 <= x <= 153 and 57 <= y <= 123)
                self.assertTrue(
                    vector["provenance"]["independent_from_existing_ego_vz"]
                )

    def test_stage_cache_reuses_identical_segment_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            position_video, provisional = _inputs(root)
            position_state = {
                "positions_3d": [position_video],
                "dataset_root": str(root),
            }
            ego_state = {
                "ego_symbol_prior": [provisional],
                "ego_motion": [],
            }
            output_root = root / "outputs"
            with patch(
                "src.exp_july.perception.pipeline.get_pipeline_output_root",
                return_value=output_root,
            ):
                first = step7b_background_motion_evidence(position_state, ego_state)
                second = step7b_background_motion_evidence(position_state, ego_state)
            self.assertEqual(
                first["background_motion_evidence_manifest"]["cached_videos"], 0
            )
            self.assertEqual(
                second["background_motion_evidence_manifest"]["cached_videos"], 1
            )
            self.assertEqual(
                first["background_motion_evidence"][0]["segments"],
                second["background_motion_evidence"][0]["segments"],
            )


if __name__ == "__main__":
    unittest.main()
