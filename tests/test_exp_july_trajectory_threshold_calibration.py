import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.exp_july.perception.trajectory_threshold_calibration import (
    PENDING_FILENAME,
    begin_threshold_epoch,
    collect_conflicts,
    compile_candidate,
    default_policy,
    evaluate_candidate,
    fixed_video_split,
    run_threshold_calibration,
    validate_policy,
)


DEFAULTS = {
    "max_valid_frame_gap": 3,
    "max_uncertain_frame_gap": 1,
    "max_invalid_center_step_diag_ratio": 2.0,
    "max_uncertain_center_step_diag_ratio": 1.0,
    "max_invalid_bbox_size_ratio": 3.0,
    "max_uncertain_bbox_size_ratio": 2.0,
    "max_invalid_depth_step_per_frame": 8.0,
    "max_uncertain_depth_step_per_frame": 4.0,
    "max_invalid_rel_velocity_delta": 10.0,
    "max_uncertain_rel_velocity_delta": 5.0,
    "max_invalid_rel_speed": 25.0,
    "max_uncertain_rel_speed": 12.0,
    "min_motion_ratio": 0.5,
}


def conflict(value=2.08, *, video_id="update", reasons=None):
    reasons = reasons or ["track_drift"]
    return {
        "video_id": video_id,
        "track_id": 7,
        "rejection_reasons": reasons,
        "hard_invalid_reasons": [
            reason
            for reason in reasons
            if reason in {"physical_invalidity", "id_switch", "trajectory_discontinuity",
                          "motion_direction_abrupt_change"}
        ],
        "step_metrics": {
            "max_bbox_center_step_diag_ratio": value,
            "max_bbox_size_ratio": 1.0,
            "max_depth_step_per_frame": 0.0,
            "max_rel_velocity_delta": 0.0,
            "max_rel_speed": 0.0,
        },
    }


class TrajectoryThresholdCalibrationTests(unittest.TestCase):
    def test_collect_conflicts_records_auditable_margin(self):
        protected = {
            "track_id": 7,
            "primary_label": "pedestrian",
            "symbol_grounded_protected": True,
            "symbol_grounded_protection": {
                "matched_rule_ids": ["protect_pedestrian"],
                "grounding_evidence": [{"rule_id": "protect_pedestrian"}],
                "protection_reason": "safety relevant",
            },
            "uncertainty": {"confidence_score": 0.9},
            "trajectory_statistics": {"num_observations": 8},
            "causal_motion_fact_validation": {
                "validation_status": "invalid",
                "rejection_reasons": ["track_drift"],
                "step_metrics": {"max_bbox_center_step_diag_ratio": 2.08},
                "thresholds": DEFAULTS,
            },
            "fact_decision": {
                "original_decision_before_protection": "Discard",
                "decision": "Keep with uncertainty",
            },
        }
        unprotected = copy.deepcopy(protected)
        unprotected.update({"track_id": 8, "symbol_grounded_protected": False})
        rows = collect_conflicts([{
            "video_id": "demo",
            "trajectory_motion_evidence": [protected, unprotected],
        }])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["matched_rule_ids"], ["protect_pedestrian"])
        self.assertEqual(row["original_decision"], "Discard")
        self.assertEqual(row["final_decision"], "Keep with uncertainty")
        measurement = row["calibration_measurements"][0]
        self.assertEqual(measurement["threshold_name"], "max_invalid_center_step_diag_ratio")
        self.assertAlmostEqual(measurement["measured_value"], 2.08)
        self.assertAlmostEqual(measurement["active_threshold"], 2.0)
        self.assertAlmostEqual(measurement["distance_to_threshold"], 0.08)

    def test_hard_conflict_is_audited_but_not_calibratable(self):
        active = default_policy(DEFAULTS)
        candidate, audit = compile_candidate(
            active,
            [conflict(reasons=["id_switch", "track_drift"])],
            DEFAULTS,
            min_samples=1,
            target_quantile=0.9,
            max_relative_change=0.1,
        )
        self.assertEqual(candidate["thresholds"], active["thresholds"])
        self.assertEqual(audit["excluded_hard_conflict_count"], 1)
        self.assertFalse(audit["changes"])

    def test_candidate_is_order_independent_and_capped(self):
        rows = [conflict(2.08), conflict(2.12), conflict(50.0)]
        active = default_policy(DEFAULTS)
        first, first_audit = compile_candidate(
            active, rows, DEFAULTS,
            min_samples=2, target_quantile=0.9, max_relative_change=0.1,
        )
        second, second_audit = compile_candidate(
            active, list(reversed(rows)), DEFAULTS,
            min_samples=2, target_quantile=0.9, max_relative_change=0.1,
        )
        self.assertEqual(first, second)
        self.assertEqual(first_audit, second_audit)
        self.assertAlmostEqual(
            first["thresholds"]["max_invalid_center_step_diag_ratio"], 2.2
        )

    def test_policy_validation_rejects_unsafe_pending_values(self):
        for key, value in (
            ("max_invalid_center_step_diag_ratio", float("nan")),
            ("max_invalid_center_step_diag_ratio", float("inf")),
            ("max_invalid_center_step_diag_ratio", 1000.0),
            ("max_valid_frame_gap", 4),
            ("min_motion_ratio", 0.1),
        ):
            policy = default_policy(DEFAULTS)
            policy["thresholds"][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    validate_policy(policy, DEFAULTS)

    def test_fixed_split_is_stable_disjoint_and_grouped(self):
        first = fixed_video_split(["a", "b", "c", "a", "d"])
        second = fixed_video_split(["d", "c", "b", "a"])
        self.assertEqual(first, second)
        self.assertTrue(first[0])
        self.assertTrue(first[1])
        self.assertFalse(set(first[0]) & set(first[1]))

    def test_candidate_promotes_only_on_safe_validation_improvement(self):
        active = default_policy(DEFAULTS)
        candidate = copy.deepcopy(active)
        candidate["version"] = 2
        candidate["thresholds"]["max_invalid_center_step_diag_ratio"] = 2.2

        def validation_fn(_observations, statistics, _uncertainty, *, thresholds):
            value = float(statistics["synthetic_center_ratio"])
            hard = bool(statistics.get("hard"))
            reasons = ["id_switch"] if hard else (
                ["track_drift"]
                if value > thresholds["max_invalid_center_step_diag_ratio"]
                else []
            )
            return {
                "validation_status": "invalid" if reasons else "valid",
                "rejection_reasons": reasons,
            }

        validation_video = {
            "video_id": "validation",
            "trajectory_motion_evidence": [
                {
                    "track_id": 1,
                    "symbol_grounded_protected": True,
                    "trajectory_observations": [],
                    "trajectory_statistics": {"synthetic_center_ratio": 2.1},
                    "uncertainty": {},
                },
                {
                    "track_id": 2,
                    "symbol_grounded_protected": False,
                    "trajectory_observations": [],
                    "trajectory_statistics": {"synthetic_center_ratio": 1.0},
                    "uncertainty": {},
                },
            ],
        }
        decision = evaluate_candidate(
            [validation_video], active, candidate, validation_fn,
            max_unprotected_flip_rate=0.0,
        )
        self.assertTrue(decision["promoted"])
        self.assertEqual(decision["decision"], "stage_for_next_epoch")

        validation_video["trajectory_motion_evidence"][1]["trajectory_statistics"][
            "synthetic_center_ratio"
        ] = 2.1
        rejected = evaluate_candidate(
            [validation_video], active, candidate, validation_fn,
            max_unprotected_flip_rate=0.0,
        )
        self.assertFalse(rejected["promoted"])
        self.assertEqual(rejected["reason"], "critical_regression")

    def test_pending_policy_activates_only_at_next_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            epoch_id, frozen, _ = begin_threshold_epoch(root, DEFAULTS)
            self.assertEqual(epoch_id, 1)
            pending = copy.deepcopy(frozen)
            pending.update({"version": 2, "parent_version": 1, "status": "pending"})
            pending["thresholds"]["max_invalid_center_step_diag_ratio"] = 2.1
            (root / PENDING_FILENAME).write_text(json.dumps(pending), encoding="utf-8")
            self.assertEqual(frozen["thresholds"]["max_invalid_center_step_diag_ratio"], 2.0)
            epoch_id, next_frozen, snapshot = begin_threshold_epoch(root, DEFAULTS)
            self.assertEqual(epoch_id, 2)
            self.assertTrue(snapshot["activated_pending_policy"])
            self.assertEqual(next_frozen["version"], 2)
            self.assertAlmostEqual(
                next_frozen["thresholds"]["max_invalid_center_step_diag_ratio"], 2.1
            )

    def test_batch_calibration_stages_only_validation_proven_candidate(self):
        video_ids = ["video_a", "video_b", "video_c", "video_d"]
        update_ids, validation_ids = fixed_video_split(video_ids)
        evidence = []
        for video_id in video_ids:
            ratio = 2.1 if video_id in update_ids else 2.05
            protected = {
                "track_id": 1,
                "primary_label": "pedestrian",
                "symbol_grounded_protected": True,
                "symbol_grounded_protection": {
                    "matched_rule_ids": ["protect_pedestrian"],
                    "grounding_evidence": [],
                },
                "trajectory_observations": [],
                "trajectory_statistics": {
                    "num_observations": 8,
                    "synthetic_center_ratio": ratio,
                },
                "uncertainty": {},
                "causal_motion_fact_validation": {
                    "validation_status": "invalid",
                    "rejection_reasons": ["track_drift"],
                    "step_metrics": {
                        "max_bbox_center_step_diag_ratio": ratio,
                        "max_bbox_size_ratio": 1.0,
                        "max_depth_step_per_frame": 0.0,
                        "max_rel_velocity_delta": 0.0,
                        "max_rel_speed": 0.0,
                    },
                    "thresholds": DEFAULTS,
                },
                "fact_decision": {
                    "original_decision_before_protection": "Discard",
                    "decision": "Keep with uncertainty",
                },
            }
            evidence.append({
                "video_id": video_id,
                "trajectory_motion_evidence": [protected],
            })

        def validation_fn(_observations, statistics, _uncertainty, *, thresholds):
            invalid = (
                statistics["synthetic_center_ratio"]
                > thresholds["max_invalid_center_step_diag_ratio"]
            )
            return {
                "validation_status": "invalid" if invalid else "valid",
                "rejection_reasons": ["track_drift"] if invalid else [],
            }

        state = {
            "trajectory_motion_evidence": evidence,
            "trajectory_validation_threshold_policy": default_policy(DEFAULTS),
            "trajectory_validation_threshold_epoch_id": 1,
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "CAUVID_STEP8_THRESHOLD_MIN_CONFLICTS": "1",
                "CAUVID_STEP8_THRESHOLD_MAX_UNPROTECTED_FLIP_RATE": "0",
            },
        ):
            root = Path(tmp)
            result = run_threshold_calibration(
                state, root, DEFAULTS, validation_fn
            )
            manifest = result["trajectory_threshold_calibration_manifest"]
            self.assertEqual(manifest["update_video_ids"], update_ids)
            self.assertEqual(manifest["validation_video_ids"], validation_ids)
            self.assertTrue(manifest["promotion"]["promoted"])
            self.assertEqual(manifest["activation"], "next_epoch_only")
            self.assertTrue((root / "policies" / PENDING_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
