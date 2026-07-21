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


class Step8DAdaptiveMotionRepairTests(unittest.TestCase):
    def test_diagnostic_repair_preserves_step8b_and_writes_comparison(self):
        reliable = _trajectory(1, _observations(False))
        invalid = _trajectory(2, _observations(True), protected=True)
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
            "refined_ego_motion": [{"video_id": "demo", "frames": []}],
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
        self.assertIn("before_signals", track)
        self.assertIn("after_signals", track)
        self.assertTrue(track["diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
