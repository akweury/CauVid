from __future__ import annotations

import unittest

from src.exp_driving_videos import od_calibration_loop


class ODCalibrationLoopControllerTest(unittest.TestCase):
    def test_normalize_cfg_applies_defaults(self) -> None:
        cfg = od_calibration_loop.normalize_loop_cfg({}, improvement_tolerance=1e-6)
        self.assertEqual(cfg.max_iterations, 1)
        self.assertTrue(cfg.force_full_recompute_on_policy_change)
        self.assertTrue(cfg.stop_on_gate_reject)
        self.assertTrue(cfg.stop_on_final_f1_plateau)
        self.assertAlmostEqual(cfg.improvement_tolerance, 1e-6)

    def test_force_recompute_overrides_candidate_dependent_keys(self) -> None:
        resolved = od_calibration_loop.apply_iteration_recompute_overrides(
            {"logic_atoms": False, "rule_evaluation": False, "background_rule_relevance_prior": False},
            force_full_recompute=True,
        )
        self.assertTrue(resolved["logic_atoms"])
        self.assertTrue(resolved["rule_evaluation"])
        self.assertFalse(resolved["background_rule_relevance_prior"])

    def test_bootstrap_accept_continues_when_budget_remains(self) -> None:
        loop_cfg = od_calibration_loop.normalize_loop_cfg(
            {"max_iterations": 3},
            improvement_tolerance=1e-9,
        )
        decision = od_calibration_loop.should_continue_after_iteration(
            {
                "decision": "accept",
                "current_metrics": {"final_f1": 0.62},
                "reference_metrics": {},
            },
            iteration_index=1,
            loop_cfg=loop_cfg,
        )
        self.assertTrue(decision.continue_loop)
        self.assertEqual(decision.reason, "continue_with_updated_active_policy")

    def test_plateau_stops_loop(self) -> None:
        loop_cfg = od_calibration_loop.normalize_loop_cfg(
            {"max_iterations": 3, "stop_on_final_f1_plateau": True},
            improvement_tolerance=1e-9,
        )
        decision = od_calibration_loop.should_continue_after_iteration(
            {
                "decision": "accept",
                "current_metrics": {"final_f1": 0.62},
                "reference_metrics": {"final_f1": 0.62},
            },
            iteration_index=2,
            loop_cfg=loop_cfg,
        )
        self.assertFalse(decision.continue_loop)
        self.assertEqual(decision.reason, "final_f1_not_improved")

    def test_gate_reject_stops_loop(self) -> None:
        loop_cfg = od_calibration_loop.normalize_loop_cfg(
            {"max_iterations": 3},
            improvement_tolerance=1e-9,
        )
        decision = od_calibration_loop.should_continue_after_iteration(
            {
                "decision": "reject",
                "current_metrics": {"final_f1": 0.58},
                "reference_metrics": {"final_f1": 0.60},
            },
            iteration_index=2,
            loop_cfg=loop_cfg,
        )
        self.assertFalse(decision.continue_loop)
        self.assertEqual(decision.reason, "gate_rejected")

    def test_max_iterations_stops_loop(self) -> None:
        loop_cfg = od_calibration_loop.normalize_loop_cfg(
            {"max_iterations": 2},
            improvement_tolerance=1e-9,
        )
        decision = od_calibration_loop.should_continue_after_iteration(
            {
                "decision": "accept",
                "current_metrics": {"final_f1": 0.70},
                "reference_metrics": {"final_f1": 0.60},
            },
            iteration_index=2,
            loop_cfg=loop_cfg,
        )
        self.assertFalse(decision.continue_loop)
        self.assertEqual(decision.reason, "max_iterations_reached")


if __name__ == "__main__":
    unittest.main()
