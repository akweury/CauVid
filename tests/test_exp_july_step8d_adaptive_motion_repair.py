import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.exp_july.perception.adaptive_motion_repair import (
    _evaluate,
    _recompute_motion,
    run_adaptive_motion_repair,
)


def _observations(depth_spike=False):
    rows = []
    for frame_id in range(8):
        depth = 10.0 + 0.1 * frame_id
        if depth_spike and frame_id == 4:
            depth += 25.0
        rows.append(
            {
                "frame_index": frame_id,
                "object_index": 0,
                "frame_label": "car",
                "bbox": [100.0 + frame_id, 80.0, 150.0 + frame_id, 130.0],
                "position_3d": [0.1 * frame_id, 0.0, depth],
                "motion": {
                    "ego_vx": 0.0,
                    "ego_vz": 0.0,
                    "has_rel_motion": frame_id > 0,
                },
                "provenance": {
                    "source": "observed",
                    "is_observed": True,
                    "is_repaired": False,
                    "is_merged": False,
                },
                "uncertainty": {
                    "score": 0.95,
                    "source_uncertainty": 0.01,
                    "has_rel_motion": frame_id > 0,
                },
            }
        )
    return _recompute_motion(rows, {})


def _trajectory(track_id, observations, protected=False):
    evaluation = _evaluate(observations, 8)
    fact = copy.deepcopy(evaluation["decision"])
    if protected:
        fact.update(
            {
                "original_decision_before_protection": "Discard",
                "trajectory_decision": "Discard",
                "final_decision_after_protection": "Keep_with_uncertainty",
                "send_to_motion_signal_refinement": True,
            }
        )
        fact["decision"] = "Keep with uncertainty"
    return {
        "track_id": track_id,
        "primary_label": "car",
        "trajectory_observations": observations,
        "trajectory_statistics": evaluation["statistics"],
        "uncertainty": evaluation["uncertainty"],
        "causal_motion_fact_validation": evaluation["validation"],
        "motion_significance_assessment": evaluation["significance"],
        "fact_decision": fact,
        "fact_decision_status": fact["decision"],
        "symbolic_layer_eligible": bool(fact.get("symbolic_layer_eligible", False)),
        "symbol_grounded_protected": protected,
        "symbol_grounded_protection": (
            {
                "track_id": track_id,
                "matched_rule_ids": ["protect_frontal_car"],
                "grounding_evidence": [{"rule_id": "protect_frontal_car"}],
            }
            if protected
            else {}
        ),
    }


def _relative_video(track_id, observations):
    frames = []
    for observation in observations:
        motion = dict(observation.get("motion", {}))
        frames.append(
            {
                "frame_index": int(observation["frame_index"]),
                "image_path": "",
                "objects": [
                    {
                        "track_id": track_id,
                        "frame_label": observation["frame_label"],
                        "label": observation["frame_label"],
                        "bbox": list(observation["bbox"]),
                        "box": list(observation["bbox"]),
                        "position_3d": list(observation["position_3d"]),
                        "relative_position_3d": list(observation["position_3d"]),
                        "score": 0.95,
                        "source": "observed",
                        "is_observed": True,
                        "is_repaired": False,
                        **motion,
                    }
                ],
            }
        )
    return {"video_id": "demo", "num_frames": 8, "frames": frames}


class Step8DAdaptiveMotionRepairTests(unittest.TestCase):
    def test_diagnostic_repair_preserves_step8b_and_writes_comparison(self):
        reliable = _trajectory(1, _observations(False))
        invalid = _trajectory(2, _observations(True), protected=False)
        self.assertIn(
            "depth_jump",
            invalid["causal_motion_fact_validation"]["rejection_reasons"],
        )
        state = {
            "videos": ["demo"],
            "trajectory_motion_evidence": [
                {
                    "video_id": "demo",
                    "num_frames": 8,
                    "trajectory_motion_evidence": [reliable, invalid],
                }
            ],
            "relative_object_motion": [_relative_video(2, invalid["trajectory_observations"])],
            "ego_motion": [{"video_id": "demo", "frames": []}],
        }
        original = json.dumps(state["trajectory_motion_evidence"], sort_keys=True)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_adaptive_motion_repair(state, Path(tmp))
            report = json.loads(
                (Path(tmp) / "demo" / "adaptive_motion_repair.json").read_text()
            )
            self.assertTrue((Path(tmp) / "repair_strategy_calibration.json").exists())
        self.assertEqual(
            original,
            json.dumps(result["trajectory_motion_evidence"], sort_keys=True),
        )
        self.assertEqual(report["num_queued"], 1)
        track = report["tracks"][0]
        self.assertEqual(track["original_decision"], "Discard")
        self.assertTrue(track["attempted_strategies"])
        attempted = {
            stage["strategy"]
            for candidate in track["attempted_strategies"]
            for stage in candidate["strategies"]
        }
        self.assertTrue(
            attempted.issubset(
                {"track_split", "fragment_reassociation", "outlier_removal"}
            )
        )
        self.assertEqual(
            report["repair_profile"],
            "identity_fragment_and_outlier_only",
        )
        self.assertEqual(
            set(report["enabled_strategies"]),
            {"track_split", "fragment_reassociation", "outlier_removal"},
        )
        self.assertIn("before_signals", track)
        self.assertIn("after_signals", track)
        self.assertTrue(track["diagnostic_only"])
        self.assertEqual(track["final_decision"], "Repair")
        self.assertEqual(
            result["relative_object_motion"][0]["step8d_modified_track_ids"],
            [2],
        )
        before_depth = state["relative_object_motion"][0]["frames"][4]["objects"][0]["position_3d"][2]
        after_depth = result["relative_object_motion"][0]["frames"][4]["objects"][0]["position_3d"][2]
        self.assertNotEqual(before_depth, after_depth)
        self.assertEqual(
            result["pre_repair_relative_object_motion"],
            state["relative_object_motion"],
        )


if __name__ == "__main__":
    unittest.main()
