"""Step 8 contracts for bounded local numerical re-estimation."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .artifacts import ArtifactRef
from .base import (
    ArtifactOwner,
    ContractId,
    ContractModel,
    FiniteFloat,
    ImageSize,
    NonNegativeFloat,
    PositiveFloat,
    RepairOperator,
    Sha256,
    ToolVersion,
)
from .tracking import ArtifactLink
from .world_state import WorldHypothesis


class Step8InputSnapshot(ContractModel):
    source_step7_relative_root: str
    repair_proposal_store: ArtifactLink
    video_repair_proposal_manifest: ArtifactLink
    source_step6_relative_root: str
    residual_store: ArtifactLink
    video_residual_manifest: ArtifactLink
    source_step5_relative_root: str
    world_state_store: ArtifactLink
    video_world_state_manifest: ArtifactLink
    source_step4_relative_root: str
    video_geometry_manifest: ArtifactLink
    source_step3_relative_root: str
    video_tracking_manifest: ArtifactLink

    @field_validator(
        "source_step7_relative_root",
        "source_step6_relative_root",
        "source_step5_relative_root",
        "source_step4_relative_root",
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
    def validate_owners(self) -> "Step8InputSnapshot":
        expected = (
            (self.repair_proposal_store, ArtifactOwner.STEP7_DIAGNOSIS),
            (self.video_repair_proposal_manifest, ArtifactOwner.STEP7_DIAGNOSIS),
            (self.residual_store, ArtifactOwner.STEP6_VERIFICATION),
            (self.video_residual_manifest, ArtifactOwner.STEP6_VERIFICATION),
            (self.world_state_store, ArtifactOwner.STEP5_WORLD_RECONSTRUCTION),
            (self.video_world_state_manifest, ArtifactOwner.STEP5_WORLD_RECONSTRUCTION),
            (self.video_geometry_manifest, ArtifactOwner.STEP4_GEOMETRY_SCALE),
            (self.video_tracking_manifest, ArtifactOwner.STEP3_OBJECT_TRACKING),
        )
        if any(link.owner != owner for link, owner in expected):
            raise ValueError("Step 8 input artifact owner mismatch")
        return self


class Step8ConfigSnapshot(ContractModel):
    schema_name: Literal["step8_config"] = "step8_config"
    schema_version: Literal[1] = 1
    implementation_version: Literal["step08_local_reestimation_v1"] = (
        "step08_local_reestimation_v1"
    )
    solver_policy_version: Literal["bounded_local_solver_v1"] = (
        "bounded_local_solver_v1"
    )

    maximum_candidates_per_proposal: Annotated[int, Field(gt=0, le=16)]
    fit_objective_weight: PositiveFloat
    physics_objective_weight: PositiveFloat
    minimum_position_std: PositiveFloat


class ObjectiveTerms(ContractModel):
    fit_evidence_error: NonNegativeFloat
    physics_error: NonNegativeFloat
    total: NonNegativeFloat
    fit_residual_count: Annotated[int, Field(ge=0)]
    physics_residual_count: Annotated[int, Field(ge=0)]


class NumericStateChange(ContractModel):
    field_path: ContractId
    frame_index: Annotated[int, Field(ge=0)]
    before_values: tuple[FiniteFloat, ...]
    after_values: tuple[FiniteFloat, ...]
    unit: ContractId
    maximum_standardized_delta: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def validate_change(self) -> "NumericStateChange":
        if not self.before_values or len(self.before_values) != len(self.after_values):
            raise ValueError("numeric diffs require equally sized non-empty values")
        if self.before_values == self.after_values:
            raise ValueError("numeric diffs must change a value")
        return self


class DiscreteStateChange(ContractModel):
    field_path: ContractId
    before_value: ContractId
    after_value: ContractId

    @model_validator(mode="after")
    def validate_change(self) -> "DiscreteStateChange":
        if self.before_value == self.after_value:
            raise ValueError("discrete diffs must change a value")
        return self


class LocalReestimationCandidate(ContractModel):
    candidate_id: ContractId
    proposal_id: ContractId
    parent_hypothesis_id: ContractId
    operator: RepairOperator
    status: Literal["instantiated", "no_change", "unsupported", "unresolved"]
    child_hypothesis: WorldHypothesis | None = None
    numerical_changes: tuple[NumericStateChange, ...] = ()
    discrete_changes: tuple[DiscreteStateChange, ...] = ()
    objective_before: ObjectiveTerms | None = None
    objective_after: ObjectiveTerms | None = None
    optimized_residual_ids: tuple[ContractId, ...] = ()
    excluded_check_residual_ids: tuple[ContractId, ...] = ()
    solver_method: ContractId
    solver_iterations: Annotated[int, Field(ge=0)]
    boundary_preserved: bool
    parameter_bounds_satisfied: bool
    compute_budget_honored: bool
    self_consistency_only: bool
    limitations: tuple[ContractId, ...] = ()
    parent_immutable: Literal[True] = True
    raw_evidence_mutated: Literal[False] = False
    selection_applied: Literal[False] = False

    @model_validator(mode="after")
    def validate_candidate(self) -> "LocalReestimationCandidate":
        instantiated = self.status == "instantiated"
        if instantiated != (self.child_hypothesis is not None):
            raise ValueError("only instantiated candidates may carry a child hypothesis")
        if instantiated and not (self.numerical_changes or self.discrete_changes):
            raise ValueError("instantiated candidates require a reversible diff")
        if instantiated and self.child_hypothesis.hypothesis_id == self.parent_hypothesis_id:
            raise ValueError("child hypothesis ID must differ from its parent")
        if set(self.optimized_residual_ids) & set(self.excluded_check_residual_ids):
            raise ValueError("check-only residuals cannot be optimized by Step 8")
        if len(self.optimized_residual_ids) != len(set(self.optimized_residual_ids)):
            raise ValueError("optimized residual IDs must be unique")
        if len(self.excluded_check_residual_ids) != len(
            set(self.excluded_check_residual_ids)
        ):
            raise ValueError("excluded check residual IDs must be unique")
        if instantiated and (self.objective_before is None or self.objective_after is None):
            raise ValueError("instantiated candidates require objective accounting")
        if instantiated and not (
            self.boundary_preserved
            and self.parameter_bounds_satisfied
            and self.compute_budget_honored
        ):
            raise ValueError("instantiated candidates must satisfy every Step 8 guard")
        return self


class ProposalReestimationResult(ContractModel):
    proposal_id: ContractId
    parent_hypothesis_id: ContractId
    operator: RepairOperator
    status: Literal["candidates_generated", "no_change", "unsupported", "unresolved"]
    candidates: tuple[LocalReestimationCandidate, ...]

    @model_validator(mode="after")
    def validate_result(self) -> "ProposalReestimationResult":
        if not self.candidates:
            raise ValueError("every Step 8 proposal result requires an audit candidate")
        if any(row.proposal_id != self.proposal_id for row in self.candidates):
            raise ValueError("candidate references another repair proposal")
        if any(
            row.parent_hypothesis_id != self.parent_hypothesis_id
            for row in self.candidates
        ):
            raise ValueError("candidate references another parent hypothesis")
        if any(row.operator != self.operator for row in self.candidates):
            raise ValueError("candidate operator differs from its proposal")
        candidate_ids = tuple(row.candidate_id for row in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique inside a proposal")
        generated = any(row.status == "instantiated" for row in self.candidates)
        if generated != (self.status == "candidates_generated"):
            raise ValueError("proposal result status does not match instantiated children")
        expected_candidate_status = {
            "no_change": "no_change",
            "unsupported": "unsupported",
            "unresolved": "unresolved",
        }.get(self.status)
        if expected_candidate_status is not None and any(
            row.status != expected_candidate_status for row in self.candidates
        ):
            raise ValueError("proposal result and candidate audit statuses differ")
        return self


class HypothesisReestimationPacket(ContractModel):
    packet_id: ContractId
    parent_hypothesis_id: ContractId
    parent_hypothesis_rank: Annotated[int, Field(gt=0)]
    proposal_results: tuple[ProposalReestimationResult, ...]
    child_candidate_count: Annotated[int, Field(ge=0)]
    parent_mutated: Literal[False] = False
    selection_applied: Literal[False] = False

    @model_validator(mode="after")
    def validate_packet(self) -> "HypothesisReestimationPacket":
        if any(
            row.parent_hypothesis_id != self.parent_hypothesis_id
            for row in self.proposal_results
        ):
            raise ValueError("proposal result belongs to another parent")
        proposal_ids = tuple(row.proposal_id for row in self.proposal_results)
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("Step 8 proposal results must be unique")
        child_ids = tuple(
            candidate.child_hypothesis.hypothesis_id
            for row in self.proposal_results
            for candidate in row.candidates
            if candidate.child_hypothesis is not None
        )
        if len(child_ids) != len(set(child_ids)):
            raise ValueError("Step 8 child hypothesis IDs must be unique")
        expected = sum(
            candidate.status == "instantiated"
            for row in self.proposal_results
            for candidate in row.candidates
        )
        if self.child_candidate_count != expected:
            raise ValueError("Step 8 child candidate accounting does not close")
        return self


class Step8ValidationSummary(ContractModel):
    input_parent_count: Annotated[int, Field(gt=0)]
    input_proposal_count: Annotated[int, Field(ge=0)]
    input_ready_proposal_count: Annotated[int, Field(ge=0)]
    input_unresolved_proposal_count: Annotated[int, Field(ge=0)]
    generated_proposal_count: Annotated[int, Field(ge=0)]
    generated_candidate_count: Annotated[int, Field(ge=0)]
    no_change_proposal_count: Annotated[int, Field(ge=0)]
    unsupported_proposal_count: Annotated[int, Field(ge=0)]
    output_unresolved_proposal_count: Annotated[int, Field(ge=0)]
    check_evidence_optimization_violations: Literal[0] = 0
    parent_mutation_count: Literal[0] = 0
    raw_evidence_mutation_count: Literal[0] = 0
    selection_count: Literal[0] = 0
    overall_pass: bool

    @model_validator(mode="after")
    def validate_counts(self) -> "Step8ValidationSummary":
        input_accounted = (
            self.input_ready_proposal_count
            + self.input_unresolved_proposal_count
        )
        output_accounted = (
            self.generated_proposal_count
            + self.no_change_proposal_count
            + self.unsupported_proposal_count
            + self.output_unresolved_proposal_count
        )
        if input_accounted != self.input_proposal_count:
            raise ValueError("Step 8 input proposal-status accounting does not close")
        if output_accounted != self.input_proposal_count:
            raise ValueError("Step 8 output proposal-status accounting does not close")
        if self.generated_candidate_count < self.generated_proposal_count:
            raise ValueError("generated proposals require at least one child candidate")
        if not self.overall_pass:
            raise ValueError("published Step 8 output requires closed accounting")
        return self


class VideoLocalReestimationManifest(ContractModel):
    schema_name: Literal["video_local_reestimation_manifest"] = (
        "video_local_reestimation_manifest"
    )
    schema_version: Literal[1] = 1

    run_id: ContractId
    video_id: ContractId
    source_repair_manifest_sha256: Sha256
    config_sha256: Sha256
    canonical_fps: PositiveFloat
    image_size: ImageSize
    frame_count: Annotated[int, Field(gt=0)]
    input_snapshot: Step8InputSnapshot
    packets: tuple[HypothesisReestimationPacket, ...]
    validation: Step8ValidationSummary
    tool_versions: tuple[ToolVersion, ...] = ()

    @model_validator(mode="after")
    def validate_packets(self) -> "VideoLocalReestimationManifest":
        if not self.packets:
            raise ValueError("Step 8 requires at least one parent packet")
        ids = tuple(row.parent_hypothesis_id for row in self.packets)
        if len(ids) != len(set(ids)):
            raise ValueError("Step 8 parent packets must be unique")
        return self


class LocalReestimationStore(ContractModel):
    schema_name: Literal["local_reestimation_store"] = "local_reestimation_store"
    schema_version: Literal[1] = 1

    run_id: ContractId
    source_repair_proposal_store_sha256: Sha256
    config: Step8ConfigSnapshot
    config_sha256: Sha256
    video_ids: tuple[ContractId, ...]
    video_local_reestimations: tuple[ArtifactRef, ...]

    @model_validator(mode="after")
    def validate_videos(self) -> "LocalReestimationStore":
        if not self.video_ids or len(self.video_ids) != len(set(self.video_ids)):
            raise ValueError("Step 8 video IDs must be non-empty and unique")
        if len(self.video_ids) != len(self.video_local_reestimations):
            raise ValueError("every video must have one Step 8 manifest")
        return self


__all__ = [
    "DiscreteStateChange",
    "HypothesisReestimationPacket",
    "LocalReestimationCandidate",
    "LocalReestimationStore",
    "NumericStateChange",
    "ObjectiveTerms",
    "ProposalReestimationResult",
    "Step8ConfigSnapshot",
    "Step8InputSnapshot",
    "Step8ValidationSummary",
    "VideoLocalReestimationManifest",
]
