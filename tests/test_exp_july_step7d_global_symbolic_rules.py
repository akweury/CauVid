import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.exp_july.perception.global_ego_symbolic_rules import (
    RULE_POLICY_ID,
    RULES,
    evaluate_segment,
    evaluate_video,
)
from src.exp_july.perception.pipeline import step7d_global_symbolic_rule_evaluation


def _segment(**updates):
    row = {
        "segment_id": 0,
        "start_frame": 0,
        "end_frame": 20,
        "provisional_action": "forward",
        "normalized_motion_magnitude": 2.0,
        "motion_magnitude_robust_z": 1.2,
        "dominant_radial_direction": "expansion",
        "direction_support_ratio": 0.82,
        "signed_direction_balance": 0.70,
        "background_flow_left_support_ratio": 0.10,
        "background_flow_right_support_ratio": 0.12,
        "background_flow_neutral_support_ratio": 0.78,
        "horizontal_flow_balance": 0.02,
        "normalized_horizontal_motion": 0.03,
        "turning_structure_support": 0.08,
        "region_support_ratio": 0.78,
        "temporal_persistence": 0.90,
        "estimator_agreement": 0.85,
        "uncertainty": 0.12,
    }
    row.update(updates)
    return row


def _video(video_id="demo", segment=None):
    return {
        "version": 2,
        "video_id": video_id,
        "input_label_status": "provisional",
        "normalized_segment_evidence": [segment or _segment()],
    }


class Step7DGlobalSymbolicRulesTests(unittest.TestCase):
    def test_forward_rules_fire_with_atom_level_evidence_chain(self):
        result = evaluate_segment(_segment())
        fired_ids = {row["rule_id"] for row in result["fired_rules"]}
        self.assertIn("forward_radial_consensus", fired_ids)
        self.assertIn("forward_motion_agreement", fired_ids)
        self.assertEqual(result["hypothesis_ranking"][0], "forward")
        self.assertGreater(
            result["hypothesis_scores"]["forward"],
            result["hypothesis_scores"]["backward"],
        )
        self.assertTrue(result["reasoning_chain"])
        self.assertEqual(
            result["decision_authority"],
            "evidence_evaluation_only_no_label_change",
        )
        fired = next(
            row for row in result["fired_rules"]
            if row["rule_id"] == "forward_radial_consensus"
        )
        self.assertTrue(all(atom["passed"] for atom in fired["atoms"]))
        self.assertTrue(all("observed" in atom for atom in fired["atoms"]))

    def test_static_and_turning_hypotheses_use_normalized_evidence(self):
        static = evaluate_segment(_segment(
            provisional_action="static",
            normalized_motion_magnitude=0.30,
            dominant_radial_direction="neutral",
            direction_support_ratio=0.78,
            signed_direction_balance=0.02,
            estimator_agreement=0.82,
        ))
        self.assertEqual(static["hypothesis_ranking"][0], "static")
        self.assertIn(
            "static_low_motion", {row["rule_id"] for row in static["fired_rules"]}
        )
        turning = evaluate_segment(_segment(
            provisional_action="turning_left",
            background_flow_right_support_ratio=0.78,
            horizontal_flow_balance=0.65,
            turning_structure_support=0.62,
            normalized_horizontal_motion=1.4,
        ))
        fired = {row["rule_id"] for row in turning["fired_rules"]}
        self.assertIn("left_background_flow", fired)
        self.assertIn("turning_multiregion_motion", fired)
        self.assertGreater(turning["hypothesis_scores"]["left"], 0.7)
        self.assertGreater(turning["hypothesis_scores"]["turning"], 0.7)

    def test_mutually_exclusive_support_is_reported_as_conflict(self):
        result = evaluate_segment(_segment(
            provisional_action="left",
            background_flow_left_support_ratio=0.68,
            background_flow_right_support_ratio=0.68,
            horizontal_flow_balance=0.0,
            turning_structure_support=0.60,
        ))
        self.assertTrue(any(
            row["type"] == "mutually_exclusive_hypotheses"
            and row["hypotheses"] == ["left", "right"]
            for row in result["conflicts"]
        ))

    def test_rule_policy_is_identical_across_videos_and_stage_is_cached(self):
        first_video = evaluate_video(_video("a"))
        second_video = evaluate_video(_video("b"))
        self.assertEqual(first_video["rule_policy_id"], RULE_POLICY_ID)
        self.assertEqual(first_video["shared_rule_policy"], second_video["shared_rule_policy"])
        self.assertEqual(tuple(first_video["shared_rule_policy"]), RULES)
        state = {"video_local_calibrated_evidence": [_video("demo")]}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.exp_july.perception.pipeline.get_pipeline_output_root",
            return_value=Path(tmp),
        ):
            first = step7d_global_symbolic_rule_evaluation(state)
            second = step7d_global_symbolic_rule_evaluation(copy.deepcopy(state))
        self.assertEqual(
            first["global_ego_symbolic_rule_evaluation_manifest"]["cached_videos"], 0
        )
        self.assertEqual(
            second["global_ego_symbolic_rule_evaluation_manifest"]["cached_videos"], 1
        )
        self.assertFalse(
            second["global_ego_symbolic_rule_evaluations"][0]["provenance"]["labels_modified"]
        )


if __name__ == "__main__":
    unittest.main()
