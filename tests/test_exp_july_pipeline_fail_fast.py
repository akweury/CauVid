import io
import sys
import unittest
from contextlib import redirect_stderr
from unittest.mock import MagicMock, Mock

if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

from src.exp_july.pipeline import _require_step_data, _tracked_step


class PipelineFailFastTests(unittest.TestCase):
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
        self.assertIn("check the dataset mount/path", message)
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
