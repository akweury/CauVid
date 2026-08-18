"""Step 7 contracts for failure diagnosis and bounded repair proposals."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .artifacts import ArtifactRef
from .base import (
    ArtifactOwner,
    ContractId,
    ContractModel,
    CueFamily,
    EvaluationBasis,
    ExpectedEffectDirection,
    FailureCategory,
    FiniteFloat,
    ImageSize,
    NonNegativeFloat,
    PositiveFloat,
    Probability,
    RepairOperator,
    Sha256,
    ToolVersion,
)
from .tracking import ArtifactLink


class Step7InputSnapshot(ContractModel):
    source_step6_relative_root: str
    residual_store: ArtifactLink
    video_residual_manifest: ArtifactLink
    source_step5_relative_root: str
    video_world_state_manifest: ArtifactLink
    source_step3_relative_root: str
    video_tracking_manifest: ArtifactLink
    source_video_manifest: ArtifactLink

    @field_validator(
        "source_step6_relative_root",
        "source_step5_relative_root",
        "source_step3_relative_root",
    )
    @classmethod
    def validate_relative_root(cls, value: str) -> str:
        if not value or "\\" in value:
            raise ValueError("source stage root must be a POSIX relative path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source stage root must stay inside the run directory")
        return path.as_posix()

    @model_validator(mode="after")
    def validate_owners(self) -> "Step7InputSnapshot":
        expected = (
            (self.residual_store, ArtifactOwner.STEP6_VERIFICATION),
            (self.video_residual_manifest, ArtifactOwner.STEP6_VERIFICATION),
            (self.video_world_state_manifest, ArtifactOwner.STEP5_WORLD_RECONSTRUCTION),
            (self.video_tracking_manifest, ArtifactOwner.STEP3_OBJECT_TRACKING),
            (self.source_video_manifest, ArtifactOwner.STEP1_INIT),
        )
        if any(link.owner != owner for link, owner in expected):
            raise ValueError("Step 7 input artifact owner mismatch")
        return self


class Step7ConfigSnapshot(ContractModel):
    schema_name: Literal["step7_config"] = "step7_config"
    schema_version: Literal[1] = 1
    implementation_version: Literal["step07_diagnose_propose_v1"] = (
        "step07_diagnose_propose_v1"
    )
    diagnosis_policy_version: Literal["deterministic_diagnosis_v1"] = (
        "deterministic_diagnosis_v1"
    )
    repair_allow_list_version: Literal["bounded_repair_operators_v1"] = (
        "bounded_repair_operators_v1"
    )

    maximum_proposals_per_hypothesis: Annotated[int, Field(gt=0, le=64)]
    maximum_keyframes_per_evidence_packet: Annotated[int, Field(gt=0, le=32)]
    conflict_context_frames: Annotated[int, Field(ge=0, le=100)]
    cross_family_merge_gap_frames: Annotated[int, Field(ge=0, le=100)]
    maximum_discrete_candidates: Annotated[int, Field(gt=0, le=64)]
    default_solver_iterations: Annotated[int, Field(gt=0, le=100000)]
    default_maximum_child_hypotheses: Annotated[int, Field(gt=0, le=64)]
    default_wall_time_seconds: PositiveFloat


class EvidenceKeyframe(ContractModel):
    frame_index: Annotated[int, Field(ge=0)]
    timestamp_s: NonNegativeFloat
    source_frame_index: Annotated[int, Field(ge=0)]
    source_timestamp_s: FiniteFloat
    selection_reasons: tuple[ContractId, ...]
    residual_ids: tuple[ContractId, ...]
    evidence_artifacts: tuple[ArtifactLink, ...] = ()

    @model_validator(mode="after")
    def validate_keyframe(self) -> "EvidenceKeyframe":
        if not self.selection_reasons or not self.residual_ids:
            raise ValueError("evidence keyframes require reasons and residuals")
        if len(self.residual_ids) != len(set(self.residual_ids)):
            raise ValueError("evidence keyframe residual IDs must be unique")
        return self


class EvidencePacket(ContractModel):
    evidence_packet_id: ContractId
    hypothesis_id: ContractId
    hypothesis_rank: Annotated[int, Field(gt=0)]
    source_path: ContractId
    conflict_ids: tuple[ContractId, ...]
    residual_ids: tuple[ContractId, ...]
    component_ids: tuple[ContractId, ...]
    track_ids: tuple[ContractId, ...]
    cue_families: tuple[CueFamily, ...]
    keyframes: tuple[EvidenceKeyframe, ...]
    evidence_artifacts: tuple[ArtifactLink, ...]
    limitations: tuple[ContractId, ...] = ()

    @model_validator(mode="after")
    def validate_packet(self) -> "EvidencePacket":
        for values in (
            self.conflict_ids,
            self.residual_ids,
            self.component_ids,
            self.track_ids,
        ):
            if len(values) != len(set(values)):
                raise ValueError("evidence packet identifiers must be unique")
        if self.conflict_ids and not self.residual_ids:
            raise ValueError("conflict evidence packets require residuals")
        known = set(self.residual_ids)
        if any(set(row.residual_ids) - known for row in self.keyframes):
            raise ValueError("evidence keyframe references an unknown residual")
        return self


class FailureDiagnosis(ContractModel):
    diagnosis_id: ContractId
    hypothesis_id: ContractId
    category: FailureCategory
    confidence: Probability
    source_conflict_ids: tuple[ContractId, ...]
    source_residual_ids: tuple[ContractId, ...]
    component_ids: tuple[ContractId, ...]
    track_ids: tuple[ContractId, ...]
    supporting_cue_families: tuple[CueFamily, ...]
    check_evidence_supported: bool
    rationale: ContractId
    alternative_categories: tuple[FailureCategory, ...] = ()
    policy_version: Literal["deterministic_diagnosis_v1"] = "deterministic_diagnosis_v1"

    @model_validator(mode="after")
    def validate_diagnosis(self) -> "FailureDiagnosis":
        if not self.source_residual_ids and self.category != FailureCategory.UNOBSERVABLE_EVIDENCE:
            raise ValueError("diagnoses require at least one source residual")
        if not self.source_conflict_ids and self.category != FailureCategory.UNOBSERVABLE_EVIDENCE:
            raise ValueError("only unobservable diagnoses may omit a source conflict")
        if self.category in self.alternative_categories:
            raise ValueError("primary diagnosis cannot also be an alternative")
        return self


class RepairParameterBound(ContractModel):
    parameter_name: ContractId
    lower_bound: FiniteFloat | None = None
    upper_bound: FiniteFloat | None = None
    allowed_values: tuple[ContractId, ...] = ()
    unit: ContractId

    @model_validator(mode="after")
    def validate_bound(self) -> "RepairParameterBound":
        numeric = self.lower_bound is not None or self.upper_bound is not None
        if numeric:
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError("numeric repair bounds require both endpoints")
            if self.lower_bound > self.upper_bound:
                raise ValueError("repair parameter bounds must be ordered")
            if self.allowed_values:
                raise ValueError("repair bounds cannot be numeric and discrete")
        elif not self.allowed_values:
            raise ValueError("repair bounds require a numeric range or allowed values")
        if len(self.allowed_values) != len(set(self.allowed_values)):
            raise ValueError("allowed repair values must be unique")
        return self


class ExpectedResidualEffect(ContractModel):
    residual_id: ContractId
    evaluation_basis: EvaluationBasis
    direction: ExpectedEffectDirection
    optimized_by_step8: bool
    minimum_normalized_improvement: NonNegativeFloat | None = None
    rationale: ContractId

    @model_validator(mode="after")
    def validate_effect(self) -> "ExpectedResidualEffect":
        if (
            self.evaluation_basis
            in {EvaluationBasis.CHECK_EVIDENCE, EvaluationBasis.NOT_EVALUABLE}
            and self.optimized_by_step8
        ):
            raise ValueError(
                "check-only and non-evaluable residuals cannot be Step 8 optimization targets"
            )
        if self.direction != ExpectedEffectDirection.DECREASE:
            if self.minimum_normalized_improvement is not None:
                raise ValueError("only decreasing residuals may claim an improvement")
        elif self.minimum_normalized_improvement is None:
            raise ValueError("decreasing residuals require a minimum improvement")
        return self


class RepairComputeBudget(ContractModel):
    maximum_solver_iterations: Annotated[int, Field(ge=0)]
    maximum_child_hypotheses: Annotated[int, Field(ge=0, le=64)]
    maximum_wall_time_seconds: NonNegativeFloat


class RepairProposal(ContractModel):
    proposal_id: ContractId
    diagnosis_id: ContractId
    evidence_packet_id: ContractId
    parent_hypothesis_id: ContractId
    operator: RepairOperator
    affected_variables: tuple[ContractId, ...]
    start_frame_index: Annotated[int, Field(ge=0)]
    end_frame_index: Annotated[int, Field(ge=0)]
    parameter_bounds: tuple[RepairParameterBound, ...]
    target_residual_ids: tuple[ContractId, ...]
    expected_residual_effects: tuple[ExpectedResidualEffect, ...]
    compute_budget: RepairComputeBudget
    source_conflict_ids: tuple[ContractId, ...]
    status: Literal["ready", "leave_unresolved"]
    immutable_parent: Literal[True] = True
    raw_evidence_mutation_allowed: Literal[False] = False
    numeric_values_supplied_by_diagnoser: Literal[False] = False
    allow_list_version: Literal["bounded_repair_operators_v1"] = (
        "bounded_repair_operators_v1"
    )

    @model_validator(mode="after")
    def validate_proposal(self) -> "RepairProposal":
        if self.end_frame_index < self.start_frame_index:
            raise ValueError("repair proposal frame window must be ordered")
        if not self.affected_variables or not self.parameter_bounds:
            raise ValueError("repair proposals require variables and bounds")
        no_residual_mark = (
            self.operator == RepairOperator.MARK_UNOBSERVABLE
            and not self.target_residual_ids
            and not self.expected_residual_effects
        )
        if (
            not no_residual_mark
            and (not self.target_residual_ids or not self.expected_residual_effects)
        ):
            raise ValueError("repair proposals require residual targets and effects")
        effect_ids = {row.residual_id for row in self.expected_residual_effects}
        if set(self.target_residual_ids) - effect_ids:
            raise ValueError("every target residual requires an expected effect")
        unresolved = self.operator == RepairOperator.LEAVE_UNRESOLVED
        if unresolved != (self.status == "leave_unresolved"):
            raise ValueError("leave_unresolved status must match its operator")
        if unresolved:
            if self.compute_budget.maximum_solver_iterations != 0:
                raise ValueError("unresolved proposals cannot request solver iterations")
            if self.compute_budget.maximum_child_hypotheses != 0:
                raise ValueError("unresolved proposals cannot request child hypotheses")
        elif self.compute_budget.maximum_child_hypotheses == 0:
            raise ValueError("ready proposals require at least one child budget")
        return self


class HypothesisDiagnosisPacket(ContractModel):
    packet_id: ContractId
    hypothesis_id: ContractId
    hypothesis_rank: Annotated[int, Field(gt=0)]
    evidence: EvidencePacket
    diagnoses: tuple[FailureDiagnosis, ...]
    proposals: tuple[RepairProposal, ...]
    deferred_conflict_ids: tuple[ContractId, ...] = ()
    status: Literal["no_conflict", "proposals_ready", "unresolved", "insufficient_evidence"]
    world_state_mutated: Literal[False] = False
    selection_applied: Literal[False] = False

    @model_validator(mode="after")
    def validate_diagnosis_packet(self) -> "HypothesisDiagnosisPacket":
        if self.evidence.hypothesis_id != self.hypothesis_id:
            raise ValueError("evidence packet belongs to another hypothesis")
        if any(row.hypothesis_id != self.hypothesis_id for row in self.diagnoses):
            raise ValueError("diagnosis belongs to another hypothesis")
        if any(row.parent_hypothesis_id != self.hypothesis_id for row in self.proposals):
            raise ValueError("repair proposal belongs to another parent")
        diagnosis_ids = {row.diagnosis_id for row in self.diagnoses}
        if any(row.diagnosis_id not in diagnosis_ids for row in self.proposals):
            raise ValueError("repair proposal references an unknown diagnosis")
        if self.status == "no_conflict" and (self.diagnoses or self.proposals):
            raise ValueError("no-conflict packets cannot contain diagnoses or proposals")
        if self.status == "proposals_ready" and not any(
            row.status == "ready" for row in self.proposals
        ):
            raise ValueError("proposals_ready requires a ready proposal")
        if self.status in {"unresolved", "insufficient_evidence"} and not self.proposals:
            raise ValueError("unresolved packets require an explicit proposal")
        return self


class Step7ValidationSummary(ContractModel):
    input_hypothesis_count: Annotated[int, Field(gt=0)]
    diagnosed_hypothesis_count: Annotated[int, Field(ge=0)]
    conflict_window_count: Annotated[int, Field(ge=0)]
    diagnosis_count: Annotated[int, Field(ge=0)]
    proposal_count: Annotated[int, Field(ge=0)]
    ready_proposal_count: Annotated[int, Field(ge=0)]
    unresolved_proposal_count: Annotated[int, Field(ge=0)]
    deferred_conflict_count: Annotated[int, Field(ge=0)]
    check_evidence_optimization_violations: Literal[0] = 0
    world_state_mutation_count: Literal[0] = 0
    overall_pass: bool

    @model_validator(mode="after")
    def validate_counts(self) -> "Step7ValidationSummary":
        if self.diagnosed_hypothesis_count > self.input_hypothesis_count:
            raise ValueError("diagnosed hypotheses exceed Step 7 input")
        if self.ready_proposal_count + self.unresolved_proposal_count != self.proposal_count:
            raise ValueError("Step 7 proposal accounting does not close")
        if not self.overall_pass:
            raise ValueError("published Step 7 output requires closed accounting")
        return self


class VideoRepairProposalManifest(ContractModel):
    schema_name: Literal["video_repair_proposal_manifest"] = (
        "video_repair_proposal_manifest"
    )
    schema_version: Literal[1] = 1

    run_id: ContractId
    video_id: ContractId
    source_residual_manifest_sha256: Sha256
    config_sha256: Sha256
    canonical_fps: PositiveFloat
    image_size: ImageSize
    frame_count: Annotated[int, Field(gt=0)]
    input_snapshot: Step7InputSnapshot
    packets: tuple[HypothesisDiagnosisPacket, ...]
    validation: Step7ValidationSummary
    tool_versions: tuple[ToolVersion, ...] = ()

    @model_validator(mode="after")
    def validate_packets(self) -> "VideoRepairProposalManifest":
        if not self.packets:
            raise ValueError("Step 7 requires at least one hypothesis packet")
        ids = tuple(row.hypothesis_id for row in self.packets)
        if len(ids) != len(set(ids)):
            raise ValueError("Step 7 hypothesis packets must be unique")
        return self


class RepairProposalStore(ContractModel):
    schema_name: Literal["repair_proposal_store"] = "repair_proposal_store"
    schema_version: Literal[1] = 1

    run_id: ContractId
    source_residual_store_sha256: Sha256
    config: Step7ConfigSnapshot
    config_sha256: Sha256
    video_ids: tuple[ContractId, ...]
    video_repair_proposals: tuple[ArtifactRef, ...]

    @model_validator(mode="after")
    def validate_videos(self) -> "RepairProposalStore":
        if not self.video_ids or len(self.video_ids) != len(set(self.video_ids)):
            raise ValueError("repair proposal store video IDs must be non-empty and unique")
        if len(self.video_ids) != len(self.video_repair_proposals):
            raise ValueError("every video must have one Step 7 manifest")
        return self


__all__ = [
    "EvidenceKeyframe",
    "EvidencePacket",
    "ExpectedResidualEffect",
    "FailureDiagnosis",
    "HypothesisDiagnosisPacket",
    "RepairComputeBudget",
    "RepairParameterBound",
    "RepairProposal",
    "RepairProposalStore",
    "Step7ConfigSnapshot",
    "Step7InputSnapshot",
    "Step7ValidationSummary",
    "VideoRepairProposalManifest",
]
