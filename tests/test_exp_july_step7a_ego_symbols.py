import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.exp_july.perception.pipeline import (
    _EGO_CUE_NAMES,
    _ego_symbol_prior_video,
    step7a_ego_symbol_prior,
)


def _ego_video():
    return {
        "video_id": "demo",
        "frames": [
            {"frame_index": 0, "ego_vx_smoothed": 0.01, "ego_vz_smoothed": 0.02, "ego_yaw_rate_smoothed": 0.0, "has_ego_motion": True},
            {"frame_index": 1, "ego_vx_smoothed": 0.01, "ego_vz_smoothed": 1.0, "ego_yaw_rate_smoothed": 0.0, "has_ego_motion": True},
            {"frame_index": 2, "ego_vx_smoothed": 0.45, "ego_vz_smoothed": 1.0, "ego_yaw_rate_smoothed": 0.08, "has_ego_motion": True},
            {"frame_index": 3, "ego_vx_smoothed": -0.45, "ego_vz_smoothed": -1.0, "ego_yaw_rate_smoothed": -0.08, "has_ego_motion": True},
            {"frame_index": 4, "has_ego_motion": False},
        ],
    }


class Step7AEgoSymbolTests(unittest.TestCase):
    def test_frame_aligned_prior_preserves_schema_and_full_threshold_audit(self):
        result = _ego_symbol_prior_video(_ego_video())
        self.assertEqual(set(result["aggregate_cues"]), set(_EGO_CUE_NAMES))
        frames = result["frames"]
        self.assertGreater(frames[0]["observable_cues"]["ego_static"], 0.0)
        self.assertGreater(frames[1]["observable_cues"]["ego_driving_forward"], 0.0)
        self.assertGreater(frames[4]["observable_cues"]["ego_motion_uncertain"], 0.0)
        self.assertEqual(
            result["threshold_status"],
            "provisional_frozen_for_evidence_validation",
        )
        self.assertEqual(result["label_status"], "provisional")
        self.assertFalse(result["downstream_usable_as_final"])
        self.assertEqual(result["selected_threshold"], result["selected_thresholds"])
        self.assertTrue(result["candidate_scores"])
        self.assertTrue(result["final_action_segments"])
        self.assertEqual(len(result["continuous_signals"]), len(frames))
        self.assertIn("minimum global score", result["audit_explanation"])
        required_components = {
            "signal_fit_error",
            "state_transitions",
            "short_segment_count",
            "short_segment_duration",
            "rapid_left_right_reversals",
            "action_complexity",
        }
        self.assertEqual(
            set(result["candidate_scores"][0]["score_components"]),
            required_components,
        )

    def test_noisy_straight_video_avoids_false_lateral_and_turning_segments(self):
        frames = []
        for frame_index in range(80):
            frames.append(
                {
                    "frame_index": frame_index,
                    "ego_vx_smoothed": (
                        0.11 * math.sin(frame_index * 1.7)
                        + 0.035 * math.sin(frame_index * 4.1)
                    ),
                    "ego_vz_smoothed": 4.0,
                    "ego_yaw_rate_smoothed": 0.012 * math.sin(frame_index * 1.9),
                    "has_ego_motion": True,
                }
            )
        result = _ego_symbol_prior_video(
            {"video_id": "noisy-straight", "frames": frames}
        )
        lateral_actions = {
            "left", "right", "turning_left", "turning_right"
        }
        self.assertFalse(
            lateral_actions
            & {segment["action"] for segment in result["final_action_segments"]}
        )
        self.assertEqual(len(result["final_action_segments"]), 1)
        self.assertEqual(result["final_action_segments"][0]["action"], "forward")
        self.assertTrue(
            all(
                frame["observable_cues"]["ego_turning_left"] == 0.0
                and frame["observable_cues"]["ego_turning_right"] == 0.0
                for frame in result["frames"]
            )
        )

    def test_sustained_turn_remains_detectable(self):
        frames = []
        for frame_index in range(80):
            turning = 25 <= frame_index < 55
            frames.append(
                {
                    "frame_index": frame_index,
                    "ego_vx_smoothed": (
                        0.55 if turning else 0.02 * math.sin(frame_index / 3.0)
                    ),
                    "ego_vz_smoothed": 4.0,
                    "ego_yaw_rate_smoothed": (
                        0.085 if turning else 0.003 * math.sin(frame_index / 4.0)
                    ),
                    "has_ego_motion": True,
                }
            )
        result = _ego_symbol_prior_video(
            {"video_id": "sustained-turn", "frames": frames}
        )
        turning_segments = [
            segment
            for segment in result["final_action_segments"]
            if segment["action"] == "turning_left"
        ]
        self.assertEqual(len(turning_segments), 1)
        self.assertGreaterEqual(turning_segments[0]["duration_frames"], 25)
        self.assertGreaterEqual(
            sum(
                frame["observable_cues"]["ego_turning_left"] > 0.0
                for frame in result["frames"]
            ),
            25,
        )

    def test_step_persists_and_reuses_fingerprinted_cache(self):
        state = {"videos": ["demo"], "ego_motion": [_ego_video()]}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.exp_july.perception.pipeline.get_pipeline_output_root",
            return_value=Path(tmp),
        ):
            first = step7a_ego_symbol_prior(state)
            second = step7a_ego_symbol_prior(state)
            self.assertEqual(first["ego_symbol_prior_manifest"]["cached_videos"], 0)
            self.assertEqual(second["ego_symbol_prior_manifest"]["cached_videos"], 1)
            self.assertTrue(
                (Path(tmp) / "07a_ego_symbol_prior" / "demo" / "ego_symbol_prior.json").exists()
            )
            self.assertEqual(
                second["ego_symbol_prior"][0]["role"],
                "provisional_ego_symbol_hypothesis",
            )
            self.assertLessEqual(
                len(second["ego_symbol_prior"][0]["candidate_scores"]), 64
            )
            reconfigured = step7a_ego_symbol_prior(
                state,
                config={"candidate_lateral_thresholds": [0.2]},
            )
            self.assertEqual(
                reconfigured["ego_symbol_prior_manifest"]["cached_videos"], 0
            )
            self.assertEqual(
                len(reconfigured["ego_symbol_prior"][0]["candidate_scores"]),
                12,
            )


if __name__ == "__main__":
    unittest.main()
