import tempfile
import unittest
from pathlib import Path

from src.exp_july.perception.trajectory_cohort_policy import (
    ALLOWED_ATTRIBUTES,
    assign_cohorts,
    attach_static_metadata,
    compile_operator_plans,
    compile_rules,
    operator_library,
    write_downstream_feedback,
)


def _track(track_id=1, category="car", frames=6):
    return {
        "video_id": "demo",
        "track_id": track_id,
        "object_class": category,
        "trajectory_statistics": {"repaired_ratio": 0.0},
        "observations": [
            {
                "frame_index": frame,
                "bbox": [10 + frame, 20, 40 + frame, 60],
                "uncertainty": {"score": 0.9},
                "provenance": {"source": "observed"},
            }
            for frame in range(frames)
        ],
    }


class TrajectoryCohortPolicyTests(unittest.TestCase):
    def test_rule_compiler_rejects_motion_attributes_and_adds_catch_all(self):
        compiled = compile_rules(
            {
                "rules": [
                    {
                        "rule_id": "illegal_motion_rule",
                        "priority": 10,
                        "all": [
                            {
                                "attribute": "relative_speed",
                                "operator": "gt",
                                "value": 1.0,
                            }
                        ],
                    }
                ]
            }
        )
        self.assertTrue(compiled["compile_errors"])
        self.assertTrue(any(not rule["all"] for rule in compiled["rules"]))
        for rule in compiled["rules"]:
            for condition in rule["all"]:
                self.assertIn(condition["attribute"], ALLOWED_ATTRIBUTES)

    def test_static_metadata_rules_assign_a_reproducible_cohort(self):
        tracks = [_track()]
        catalog = attach_static_metadata(tracks)
        compiled = compile_rules(
            {
                "rules": [
                    {
                        "rule_id": "persistent_vehicle",
                        "priority": 20,
                        "all": [
                            {
                                "attribute": "category",
                                "operator": "eq",
                                "value": "car",
                            },
                            {
                                "attribute": "track_length_bucket",
                                "operator": "eq",
                                "value": "medium",
                            },
                        ],
                    },
                    {
                        "rule_id": "other",
                        "priority": 0,
                        "all": [],
                    },
                ]
            }
        )
        cohorts = assign_cohorts(tracks, compiled["rules"])
        self.assertEqual(catalog["track_count"], 1)
        self.assertEqual(list(cohorts), ["persistent_vehicle"])
        self.assertEqual(tracks[0]["cohort_id"], "persistent_vehicle")
        self.assertNotIn("relative_speed", tracks[0]["static_metadata"])

    def test_llm_operator_is_suppressed_without_systematic_anomaly(self):
        plans = compile_operator_plans(
            {
                "plans": [
                    {
                        "cohort_id": "cars",
                        "operator": "kalman_smoothing",
                        "initial_parameters": {"alpha": 0.7},
                        "anomaly_types": ["depth_jump"],
                    }
                ]
            },
            {
                "cars": {
                    "systematic_anomalies": [],
                }
            },
            operator_library(
                {
                    "kalman_smoothing": (
                        "kalman_smoothing",
                        {"alpha": 0.55},
                    )
                }
            ),
        )
        self.assertEqual(plans["cars"]["llm_requested_operator"], "kalman_smoothing")
        self.assertEqual(plans["cars"]["operator"], "no_repair")
        self.assertEqual(
            plans["cars"]["selection_source"],
            "deterministic_no_systematic_anomaly",
        )

    def test_final_downstream_feedback_is_persisted_for_next_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "trajectory_pattern_output_root": Path(tmp),
                "trajectory_pattern_records": [
                    {
                        "video_id": "demo",
                        "track_id": 1,
                        "trajectory_cohort_id": "persistent_vehicle",
                        "cohort_operator_plan": {"operator": "outlier_removal"},
                        "repair_applied": True,
                        "final_validation_status": "valid",
                    }
                ],
                "trajectory_motion_evidence": [
                    {
                        "video_id": "demo",
                        "trajectory_motion_evidence": [
                            {
                                "track_id": 1,
                                "validation_status": "invalid",
                                "fact_decision_status": "Discard",
                            }
                        ],
                    }
                ],
                "protected_objects": [{"video_id": "demo", "track_id": 1}],
            }
            feedback = write_downstream_feedback(state)
            cohort = feedback["cohorts"]["persistent_vehicle"]
            self.assertEqual(cohort["critical_regressions"], 1)
            self.assertEqual(cohort["downstream_success_rate"], 0.0)
            self.assertTrue(
                (Path(tmp) / "policies" / "downstream_feedback.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
