import unittest
from types import SimpleNamespace

import numpy as np
from pydantic import ValidationError

from src.exp_august.contracts import (
    EvaluationBasis,
    LocalReestimationCandidate,
    RepairOperator,
)
from src.exp_august.inference.step08_local_reestimation import (
    _excluded_check_ids,
    _optimized_source_ids,
    _smooth_positions,
)


def _state(frame, x, std=0.5):
    return SimpleNamespace(
        frame_index=frame,
        timestamp_s=float(frame),
        position=SimpleNamespace(x=float(x), y=0.0, z=0.0),
        position_std=SimpleNamespace(x=std, y=std, z=std),
    )


class ExpAugustStep08LocalReestimationTests(unittest.TestCase):
    def test_local_refit_preserves_boundaries_and_clips_by_declared_sigma(self):
        states = tuple(
            _state(frame, x)
            for frame, x in enumerate((0.0, 1.0, 8.0, 3.0, 4.0))
        )
        updates = _smooth_positions(
            states,
            start_frame=0,
            end_frame=4,
            strength=1.0,
            maximum_sigma=2.0,
            minimum_std=1e-6,
        )

        self.assertNotIn(0, updates)
        self.assertNotIn(4, updates)
        updated_position, standardized_delta = updates[2]
        self.assertTrue(np.allclose(updated_position, (7.0, 0.0, 0.0)))
        self.assertAlmostEqual(standardized_delta, 2.0)

    def test_check_only_residual_is_audited_but_never_optimized(self):
        proposal = SimpleNamespace(
            expected_residual_effects=(
                SimpleNamespace(
                    residual_id="residual:fit",
                    evaluation_basis=EvaluationBasis.FIT_EVIDENCE,
                    optimized_by_step8=True,
                ),
                SimpleNamespace(
                    residual_id="residual:check",
                    evaluation_basis=EvaluationBasis.CHECK_EVIDENCE,
                    optimized_by_step8=False,
                ),
            )
        )

        self.assertEqual(_optimized_source_ids(proposal), ("residual:fit",))
        self.assertEqual(_excluded_check_ids(proposal), ("residual:check",))

    def test_instantiated_candidate_requires_child_and_reversible_diff(self):
        with self.assertRaises(ValidationError):
            LocalReestimationCandidate.model_validate(
                {
                    "candidate_id": "candidate:test",
                    "proposal_id": "proposal:test",
                    "parent_hypothesis_id": "world:test",
                    "operator": RepairOperator.REFIT_LOCAL_DYNAMICS,
                    "status": "instantiated",
                    "child_hypothesis": None,
                    "solver_method": "test",
                    "solver_iterations": 1,
                    "boundary_preserved": False,
                    "parameter_bounds_satisfied": True,
                    "compute_budget_honored": True,
                    "self_consistency_only": True,
                }
            )


if __name__ == "__main__":
    unittest.main()
