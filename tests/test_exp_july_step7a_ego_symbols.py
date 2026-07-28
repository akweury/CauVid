import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.exp_july.perception.ego_symbol_finalization import _ego_timeline_states
from src.exp_july.perception.pipeline import (
    _EGO_CUE_NAMES,
    _coarse_to_fine_ego_candidate_scores,
    _ego_action,
    _ego_symbol_config,
    _ego_symbol_prior_video,
    _score_ego_threshold_candidate,
    _shortlist_step7e_candidates,
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
    def test_static_action_fallback_populates_all_visual_axes(self):
        self.assertEqual(
            _ego_timeline_states({"action": "static", "observable_cues": {}}),
            ("static", "static", "static"),
        )

    def test_visual_axes_use_uncertain_instead_of_unknown(self):
        self.assertEqual(
            _ego_timeline_states({"action": "unknown", "observable_cues": {}}),
            ("uncertain", "uncertain", "uncertain"),
        )

    def test_valid_low_margin_vz_uses_signed_fallback_instead_of_unknown(self):
        thresholds = {
            "static_speed_threshold": 0.15,
            "lateral_threshold": 0.20,
            "yaw_threshold": 0.03,
        }
        base = {"available": True, "ego_speed": 0.18, "ego_vx": 0.12, "ego_yaw_rate": 0.0}
        self.assertEqual(_ego_action({**base, "ego_vz": 0.12}, thresholds), "forward")
        self.assertEqual(_ego_action({**base, "ego_vz": -0.12}, thresholds), "backward")
        self.assertEqual(_ego_action({**base, "ego_vz": 0.0}, thresholds), "static")
        self.assertEqual(_ego_action({**base, "available": False, "ego_vz": 0.12}, thresholds), "unknown")

    def test_change_point_output_has_robust_statistics_and_provenance(self):
        values = [2.0] * 12 + [0.01] * 10 + [-2.0] * 12
        result = _ego_symbol_prior_video({
            "video_id": "three-state",
            "frames": [
                {"frame_index": index, "ego_vx_smoothed": 0.0, "ego_vz_smoothed": value, "has_ego_motion": True}
                for index, value in enumerate(values)
            ],
        })
        segments = result["final_action_segments"]
        self.assertEqual([row["state"] for row in segments], ["forward", "static", "backward"])
        self.assertTrue(all(row["duration_frames"] >= 5 for row in segments))
        self.assertTrue(all("confidence" in row for row in segments))
        self.assertTrue(all("robust_level" in row for row in segments))
        self.assertTrue(all("residual_variance" in row for row in segments))
        self.assertTrue(all("uncertain" not in row for row in segments))
        self.assertTrue(all("uncertainty_reasons" not in row for row in segments))
        self.assertEqual(result["change_point_segmentation"]["provenance"]["source_signal"], "ego_vz")

    def test_frame_aligned_prior_preserves_schema_and_full_threshold_audit(self):
        result = _ego_symbol_prior_video(_ego_video())
        self.assertEqual(set(result["aggregate_cues"]), set(_EGO_CUE_NAMES))
        frames = result["frames"]
        self.assertGreater(frames[0]["observable_cues"]["ego_static"], 0.0)
        self.assertTrue(any(
            frame["observable_cues"]["ego_static"] > 0.0
            or frame["observable_cues"]["ego_driving_forward"] > 0.0
            or frame["observable_cues"]["ego_driving_backward"] > 0.0
            for frame in frames[:-1]
        ))
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
        self.assertIn("change-point dynamic programming", result["audit_explanation"])
        self.assertEqual(
            result["candidate_scores"][0]["method"],
            "constrained_change_point_dynamic_programming",
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

    def test_lateral_and_yaw_do_not_change_vz_only_state(self):
        frames = []
        for frame_index in range(80):
            turning = 25 <= frame_index < 55
            frames.append({
                "frame_index": frame_index,
                "ego_vx_smoothed": 0.55 if turning else 0.0,
                "ego_vz_smoothed": 4.0,
                "ego_yaw_rate_smoothed": 0.085 if turning else 0.0,
                "has_ego_motion": True,
            })
        result = _ego_symbol_prior_video({"video_id": "sustained-turn", "frames": frames})
        self.assertEqual({segment["action"] for segment in result["final_action_segments"]}, {"forward"})
        self.assertTrue(all(frame["observable_cues"]["ego_turning_left"] == 0.0 for frame in result["frames"]))

    def test_direct_forward_backward_transition_is_forbidden(self):
        values = [2.0] * 12 + [-2.0] * 12
        result = _ego_symbol_prior_video({
            "video_id": "direct-reversal",
            "frames": [
                {"frame_index": index, "ego_vx_smoothed": 0.0, "ego_vz_smoothed": value, "has_ego_motion": True}
                for index, value in enumerate(values)
            ],
        })
        states = [segment["state"] for segment in result["final_action_segments"]]
        self.assertFalse(any(
            {left, right} == {"forward", "backward"}
            for left, right in zip(states, states[1:])
        ))

    def test_step7e_shortlist_is_bounded_and_preserves_winner_and_rounds(self):
        candidates = [
            {
                "candidate_id": f"c{index}",
                "score": float(index),
                "search_round": index % 3,
                "num_forward_backward_reversals": 0,
                "num_acceleration_deceleration_reversals": 0,
            }
            for index in range(12)
        ]
        shortlist = _shortlist_step7e_candidates(candidates, "c10", 6)
        self.assertLessEqual(len(shortlist), 6)
        self.assertIn("c10", {row["candidate_id"] for row in shortlist})
        self.assertEqual({row["search_round"] for row in shortlist}, {0, 1, 2})
        self.assertEqual(shortlist, _shortlist_step7e_candidates(candidates, "c10", 6))

    def test_direction_and_acceleration_oscillations_receive_explicit_penalties(self):
        vz_values = [1.0, -1.0, 1.0, -1.0, 1.0]
        deltas = [None, 0.5, -0.5, 0.5, -0.5]
        samples = [
            {
                "frame_index": index,
                "available": True,
                "ego_speed": abs(vz),
                "ego_vx": 0.0,
                "ego_vz": vz,
                "ego_yaw_rate": 0.0,
                "ego_speed_delta": deltas[index],
            }
            for index, vz in enumerate(vz_values)
        ]
        config = _ego_symbol_config({"threshold_search_rounds": 1})
        candidate = _score_ego_threshold_candidate(
            samples,
            {"static_speed_threshold": 0.15, "lateral_threshold": 0.15, "yaw_threshold": 0.03},
            config,
            "oscillating",
        )
        self.assertGreater(candidate["num_longitudinal_state_transitions"], 0)
        self.assertGreater(candidate["num_forward_backward_reversals"], 0)
        self.assertGreater(candidate["num_acceleration_state_transitions"], 0)
        self.assertGreater(candidate["num_acceleration_deceleration_reversals"], 0)
        for name in (
            "longitudinal_state_transitions",
            "forward_backward_reversals",
            "acceleration_state_transitions",
            "acceleration_deceleration_reversals",
        ):
            self.assertGreater(candidate["weighted_score_components"][name], 0.0)

    def test_coarse_to_fine_search_is_deterministic_and_refines_resolution(self):
        values = [0.05, 0.12, 0.18, 0.22, 0.28, 0.34, 0.42, 0.55, -0.10, -0.20]
        samples = [
            {
                "frame_index": index,
                "available": True,
                "ego_speed": abs(value),
                "ego_vx": 0.0,
                "ego_vz": value,
                "ego_yaw_rate": 0.0,
            }
            for index, value in enumerate(values)
        ]
        config = _ego_symbol_config({"threshold_search_rounds": 3})
        first = _coarse_to_fine_ego_candidate_scores(samples, config)
        second = _coarse_to_fine_ego_candidate_scores(samples, config)
        self.assertEqual(first, second)
        self.assertIn(1, {row["search_round"] for row in first})
        coarse = next(row for row in first if row["search_round"] == 0)
        refined = next(row for row in first if row["search_round"] == 1)
        self.assertLess(
            refined["refinement_steps"]["static_speed_threshold"],
            coarse["refinement_steps"]["static_speed_threshold"],
        )
        self.assertTrue(refined["parent_candidate_ids"])
        self.assertEqual(len({tuple(row["actions"]) for row in first}), len(first))

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
            candidate_scores = reconfigured["ego_symbol_prior"][0]["candidate_scores"]
            self.assertGreaterEqual(len(candidate_scores), 1)
            self.assertLessEqual(len(candidate_scores), 12)


if __name__ == "__main__":
    unittest.main()
