import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.exp_august.contracts import (
    ArtifactLink,
    ArtifactOwner,
    ArtifactRef,
    ConflictWindow,
    CueFamily,
    EvaluationBasis,
    EvidenceRole,
    FailureCategory,
    HypothesisResidualPacket,
    RepairOperator,
    ResidualFamily,
    ResidualRecord,
    ResidualSeverity,
    Step7ConfigSnapshot,
)
from src.exp_august.inference.step07_diagnose_propose import _hypothesis_packet
from src.exp_august.inference.frames import CanonicalFrame
from src.exp_august.inference.step07_visualization import _proposal_panel


def _artifact_link():
    return ArtifactLink(
        owner=ArtifactOwner.STEP2_NEURAL_EVIDENCE,
        artifact=ArtifactRef(
            artifact_id="flow:0",
            relative_path="flow/0.npz",
            sha256="0" * 64,
            byte_size=1,
            media_type="application/x-npz",
        ),
    )


def _packet(residual):
    conflict = ConflictWindow(
        conflict_id=f"conflict:{residual.constraint_id}",
        hypothesis_id="world:test",
        family=residual.family,
        constraint_id=residual.constraint_id,
        start_frame_index=residual.start_frame_index,
        end_frame_index=residual.end_frame_index,
        residual_ids=(residual.residual_id,),
        peak_normalized_residual=residual.normalized_residual,
        severity=residual.severity,
        component_ids=("component:0",),
        track_ids=("track:0",),
        check_evidence_supported=(
            residual.evaluation_basis == EvaluationBasis.CHECK_EVIDENCE
        ),
    )
    return HypothesisResidualPacket(
        packet_id="packet:world:test",
        hypothesis_id="world:test",
        hypothesis_rank=1,
        residuals=(residual,),
        conflict_windows=(conflict,),
        family_summaries=(),
        evaluable_fraction=1.0,
        check_evidence_residual_count=(
            1 if residual.evaluation_basis == EvaluationBasis.CHECK_EVIDENCE else 0
        ),
        check_supported_conflict_count=(
            1 if residual.evaluation_basis == EvaluationBasis.CHECK_EVIDENCE else 0
        ),
        hard_violation=residual.severity == ResidualSeverity.HARD_VIOLATION,
        status="conflicts_detected",
    )


def _context():
    tracking = SimpleNamespace(
        mask_candidate_bank=(),
        association_ledger=(),
        gap_records=(),
    )
    hypothesis = SimpleNamespace(hypothesis_id="world:test", scale_id="scale:0")
    world = SimpleNamespace(initial_beam=SimpleNamespace(hypotheses=(hypothesis,)))
    source = SimpleNamespace(
        source_path="/dataset/test.mp4",
        frames=tuple(
            SimpleNamespace(
                timestamp_s=float(index),
                source_frame_index=index,
                source_timestamp_s=float(index),
            )
            for index in range(8)
        ),
    )
    config = Step7ConfigSnapshot(
        maximum_proposals_per_hypothesis=16,
        maximum_keyframes_per_evidence_packet=8,
        conflict_context_frames=1,
        cross_family_merge_gap_frames=1,
        maximum_discrete_candidates=8,
        default_solver_iterations=50,
        default_maximum_child_hypotheses=2,
        default_wall_time_seconds=2.0,
    )
    return tracking, world, source, config


class ExpAugustStep07DiagnosisTests(unittest.TestCase):
    def test_empty_residual_packet_is_marked_unobservable_without_invented_values(self):
        packet = HypothesisResidualPacket(
            packet_id="packet:world:test",
            hypothesis_id="world:test",
            hypothesis_rank=1,
            residuals=(),
            conflict_windows=(),
            family_summaries=(),
            evaluable_fraction=0.0,
            check_evidence_residual_count=0,
            check_supported_conflict_count=0,
            hard_violation=False,
            status="insufficient_evidence",
        )
        tracking, world, source, config = _context()
        result = _hypothesis_packet(
            packet=packet,
            tracking=tracking,
            world=world,
            source=source,
            config=config,
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(
            result.diagnoses[0].category, FailureCategory.UNOBSERVABLE_EVIDENCE
        )
        proposal = result.proposals[0]
        self.assertEqual(proposal.operator, RepairOperator.MARK_UNOBSERVABLE)
        self.assertEqual(proposal.target_residual_ids, ())
        self.assertEqual(proposal.expected_residual_effects, ())
        self.assertFalse(proposal.numeric_values_supplied_by_diagnoser)

    def test_physics_conflict_proposes_bounded_local_refit(self):
        residual = ResidualRecord(
            residual_id="residual:physics",
            hypothesis_id="world:test",
            family=ResidualFamily.PHYSICS,
            constraint_id="object_acceleration_bound",
            evaluation_basis=EvaluationBasis.FROZEN_KNOWLEDGE,
            component_id="component:0",
            track_id="track:0",
            start_frame_index=3,
            end_frame_index=3,
            start_timestamp_s=3.0,
            end_timestamp_s=3.0,
            metric_name="acceleration",
            metric_unit="meter_per_second_squared",
            predicted_values=(45.0,),
            observed_values=(0.0,),
            raw_residual=45.0,
            normalized_residual=3.5,
            uncertainty=15.0,
            threshold=45.0,
            severity=ResidualSeverity.VIOLATION,
            evaluable=True,
            hard_constraint=True,
            reason="test physics conflict",
        )
        tracking, world, source, config = _context()
        result = _hypothesis_packet(
            packet=_packet(residual),
            tracking=tracking,
            world=world,
            source=source,
            config=config,
        )

        self.assertEqual(result.status, "proposals_ready")
        self.assertEqual(result.diagnoses[0].category, FailureCategory.DYNAMICS_MISMATCH)
        proposal = result.proposals[0]
        self.assertEqual(proposal.operator, RepairOperator.REFIT_LOCAL_DYNAMICS)
        self.assertEqual((proposal.start_frame_index, proposal.end_frame_index), (2, 4))
        self.assertFalse(result.world_state_mutated)
        self.assertFalse(proposal.raw_evidence_mutation_allowed)
        self.assertFalse(proposal.numeric_values_supplied_by_diagnoser)

    def test_check_only_flow_can_diagnose_but_cannot_be_optimized(self):
        evidence = _artifact_link()
        residual = ResidualRecord(
            residual_id="residual:flow",
            hypothesis_id="world:test",
            family=ResidualFamily.OBJECT_IDENTITY,
            constraint_id="heldout_object_backward_flow",
            evaluation_basis=EvaluationBasis.CHECK_EVIDENCE,
            evidence_role=EvidenceRole.CHECK_ONLY,
            cue_family=CueFamily.FLOW_BACKWARD,
            component_id="component:0",
            track_id="track:0",
            start_frame_index=4,
            end_frame_index=4,
            start_timestamp_s=4.0,
            end_timestamp_s=4.0,
            metric_name="flow_endpoint_error",
            metric_unit="pixel",
            predicted_values=(2.0, 0.0),
            observed_values=(-2.0, 0.0),
            raw_residual=4.0,
            normalized_residual=4.0,
            uncertainty=1.0,
            threshold=3.0,
            flow_direction_error_deg=180.0,
            flow_magnitude_ratio=1.0,
            severity=ResidualSeverity.VIOLATION,
            evaluable=True,
            hard_constraint=False,
            evidence_keys=("flow:backward:4",),
            evidence_artifacts=(evidence,),
            reason="test held-out flow conflict",
        )
        tracking, world, source, config = _context()
        result = _hypothesis_packet(
            packet=_packet(residual),
            tracking=tracking,
            world=world,
            source=source,
            config=config,
        )

        self.assertEqual(result.diagnoses[0].category, FailureCategory.IDENTITY_ERROR)
        self.assertEqual(result.proposals[0].operator, RepairOperator.SPLIT_TRACK)
        effect = result.proposals[0].expected_residual_effects[0]
        self.assertEqual(effect.evaluation_basis, EvaluationBasis.CHECK_EVIDENCE)
        self.assertFalse(effect.optimized_by_step8)
        self.assertTrue(result.evidence.keyframes)
        self.assertEqual(result.evidence.evidence_artifacts, (evidence,))

    def test_proposal_panel_renders_canonical_frame_without_mutating_state(self):
        residual_packet = HypothesisResidualPacket(
            packet_id="packet:world:test",
            hypothesis_id="world:test",
            hypothesis_rank=1,
            residuals=(),
            conflict_windows=(),
            family_summaries=(),
            evaluable_fraction=0.0,
            check_evidence_residual_count=0,
            check_supported_conflict_count=0,
            hard_violation=False,
            status="insufficient_evidence",
        )
        tracking, world, source, config = _context()
        tracking.tracks = ()
        packet = _hypothesis_packet(
            packet=residual_packet,
            tracking=tracking,
            world=world,
            source=source,
            config=config,
        )
        source_image = np.full((180, 320, 3), 190, dtype=np.uint8)
        frame = CanonicalFrame(
            video_id="test",
            frame_index=0,
            timestamp_s=0.0,
            source_frame_index=0,
            source_timestamp_s=0.0,
            image_bgr=source_image,
        )
        panel = _proposal_panel(
            frame=frame,
            packet=packet,
            diagnosis=packet.diagnoses[0],
            proposal=packet.proposals[0],
            residual=None,
            tracking=tracking,
            frame_count=8,
            step2_root=Path("/unused/step2"),
            step3_root=Path("/unused/step3"),
        )

        self.assertEqual(panel.shape, (1080, 1920, 3))
        self.assertTrue(np.all(source_image == 190))
        self.assertFalse(packet.world_state_mutated)


if __name__ == "__main__":
    unittest.main()
