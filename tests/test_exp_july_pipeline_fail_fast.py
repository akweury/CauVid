import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import MagicMock, Mock
from unittest.mock import patch

if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

import config
from src.exp_july.pipeline import _require_step_data, _tracked_step


class PipelineFailFastTests(unittest.TestCase):
    def test_driving_mini_discovery_accepts_frames_without_video_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp)
            first_frames = dataset_root / "frames" / "video_b"
            second_frames = dataset_root / "frames" / "video_a"
            empty_frames = dataset_root / "frames" / "empty_video"
            first_frames.mkdir(parents=True)
            second_frames.mkdir(parents=True)
            empty_frames.mkdir(parents=True)
            (first_frames / "frame_00001.jpg").touch()
            (second_frames / "frame_00002.jpg").touch()

            with patch.object(config, "get_dataset_path", return_value=dataset_root):
                self.assertEqual(
                    config.get_mini_video_ids(),
                    ["video_a", "video_b"],
                )

    def test_driving_mini_discovery_unions_frame_and_video_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp)
            frame_dir = dataset_root / "frames" / "shared"
            video_dir = dataset_root / "videos"
            frame_dir.mkdir(parents=True)
            video_dir.mkdir(parents=True)
            (frame_dir / "frame_00001.jpg").touch()
            (video_dir / "shared.mov").touch()
            (video_dir / "video_only.mp4").touch()

            with patch.object(config, "get_dataset_path", return_value=dataset_root):
                self.assertEqual(
                    config.get_mini_video_ids(),
                    ["shared", "video_only"],
                )

    def test_empty_initial_video_selection_prints_error_and_raises(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaisesRegex(
            RuntimeError, "01_init: produced no videos"
        ):
            _require_step_data(
                "01_init",
                {"videos": [], "dataset_root": "/dataset/driving_mini"},
            )

        message = stderr.getvalue()
        self.assertIn("[pipeline][error]", message)
        self.assertIn("check CAUVID_DRIVING_MINI_HOST", message)
        self.assertIn("stopping pipeline", message)

    def test_zero_step8b_tracks_are_rejected(self):
        state = {
            "videos": ["scene"],
            "uncertain_signal_evidence_manifest": {
                "num_videos": 1,
                "num_tracks": 0,
                "num_observations": 0,
            },
        }
        with self.assertRaisesRegex(
            RuntimeError, "produced zero signal-evidence tracks"
        ):
            _require_step_data("08b_uncertain_signal_evidence", state)

    def test_valid_zero_optional_outcomes_are_not_rejected(self):
        state = {
            "videos": ["scene"],
            "relative_object_motion": [
                {
                    "video_id": "scene",
                    "num_objects_total": 2,
                    "num_objects_with_rel_motion": 2,
                }
            ],
            "protected_objects": [],
            "trajectory_threshold_conflicts": [],
        }
        self.assertIs(
            _require_step_data("08e_semantic_protection", state),
            state,
        )
        self.assertIs(
            _require_step_data("08i_threshold_calibration", state),
            state,
        )

    def test_empty_downstream_primary_collection_is_rejected(self):
        with self.assertRaisesRegex(
            RuntimeError, "09_temporal_segmentation: produced no temporal segments"
        ):
            _require_step_data(
                "09_temporal_segmentation",
                {"videos": ["scene"], "temporal_segments": []},
            )

    def test_tracked_step_logs_failure_before_propagating(self):
        tracker = Mock()
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaisesRegex(
            RuntimeError, "02_detection: produced zero object detections"
        ):
            _tracked_step(
                tracker,
                "02_detection",
                lambda: {
                    "videos": ["scene"],
                    "detections": [
                        {
                            "video_id": "scene",
                            "num_detections": 0,
                            "num_candidate_detections": 0,
                        }
                    ],
                },
            )

        tracker.log_failure.assert_called_once()
        tracker.log_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
