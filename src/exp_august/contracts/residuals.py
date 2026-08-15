"""Step 6 contracts for forward prediction and consistency verification."""

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
    EvidenceRole,
    FiniteFloat,
    ImageSize,
    NonNegativeFloat,
    PositiveFloat,
    Probability,
    ResidualFamily,
    ResidualSeverity,
    Sha256,
    ToolVersion,
)
from .tracking import ArtifactLink


class Step6InputSnapshot(ContractModel):
    source_step5_relative_root: str
    world_state_store: ArtifactLink
    video_world_state_manifest: ArtifactLink
    source_step4_relative_root: str
    video_geometry_manifest: ArtifactLink
    source_step3_relative_root: str
    video_tracking_manifest: ArtifactLink
    source_step2_relative_root: str
    video_evidence_manifest: ArtifactLink

    @field_validator(
        "source_step5_relative_root",
        "source_step4_relative_root",
        "source_step3_relative_root",
        "source_step2_relative_root",
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
    def validate_owners(self) -> "Step6InputSnapshot":
        expected = (
            (self.world_state_store, ArtifactOwner.STEP5_WORLD_RECONSTRUCTION),
            (self.video_world_state_manifest, ArtifactOwner.STEP5_WORLD_RECONSTRUCTION),
            (self.video_geometry_manifest, ArtifactOwner.STEP4_GEOMETRY_SCALE),
            (self.video_tracking_manifest, ArtifactOwner.STEP3_OBJECT_TRACKING),
            (self.video_evidence_manifest, ArtifactOwner.STEP2_NEURAL_EVIDENCE),
        )
        if any(link.owner != owner for link, owner in expected):
            raise ValueError("Step 6 input artifact owner mismatch")
        return self


class Step6ConfigSnapshot(ContractModel):
    schema_name: Literal["step6_config"] = "step6_config"
    schema_version: Literal[1] = 1
    implementation_version: Literal["step06_predict_verify_v2"] = (
        "step06_predict_verify_v2"
    )

    maximum_hypotheses: Annotated[int, Field(gt=0, le=64)]
    projection_sigma_px: PositiveFloat
    depth_log_sigma: PositiveFloat
    flow_sigma_px: PositiveFloat
    background_sample_stride: Annotated[int, Field(gt=0)]
    maximum_background_samples: Annotated[int, Field(gt=0)]
    maximum_prediction_gap_frames: Annotated[int, Field(ge=0)]
    conflict_z_threshold: PositiveFloat
    hard_z_threshold: PositiveFloat
    conflict_merge_gap_frames: Annotated[int, Field(ge=0)]
    metric_max_ego_speed_mps: PositiveFloat
    metric_max_object_speed_mps: PositiveFloat
    metric_max_acceleration_mps2: PositiveFloat
    relative_acceleration_scale: PositiveFloat
    minimum_evaluable_fraction: Probability

    @model_validator(mode="after")
    def validate_thresholds(self) -> "Step6ConfigSnapshot":
        if self.hard_z_threshold <= self.conflict_z_threshold:
            raise ValueError("hard residual threshold must exceed conflict threshold")
        return self


class ResidualRecord(ContractModel):
    residual_id: ContractId
    hypothesis_id: ContractId
    family: ResidualFamily
    constraint_id: ContractId
    evaluation_basis: EvaluationBasis
    evidence_role: EvidenceRole | None = None
    cue_family: CueFamily | None = None
    component_id: str | None = None
    track_id: str | None = None
    start_frame_index: Annotated[int, Field(ge=0)]
    end_frame_index: Annotated[int, Field(ge=0)]
    start_timestamp_s: NonNegativeFloat
    end_timestamp_s: NonNegativeFloat
    metric_name: ContractId
    metric_unit: ContractId
    predicted_values: tuple[FiniteFloat, ...] = ()
    observed_values: tuple[FiniteFloat, ...] = ()
    raw_residual: NonNegativeFloat | None = None
    normalized_residual: NonNegativeFloat | None = None
    uncertainty: NonNegativeFloat | None = None
    threshold: NonNegativeFloat | None = None
    flow_direction_error_deg: NonNegativeFloat | None = None
    flow_magnitude_ratio: NonNegativeFloat | None = None
    severity: ResidualSeverity
    evaluable: bool
    hard_constraint: bool
    evidence_keys: tuple[ContractId, ...] = ()
    evidence_artifacts: tuple[ArtifactLink, ...] = ()
    reason: ContractId
    limitations: tuple[ContractId, ...] = ()

    @model_validator(mode="after")
    def validate_residual(self) -> "ResidualRecord":
        if self.end_frame_index < self.start_frame_index:
            raise ValueError("residual frame window must be ordered")
        if self.end_timestamp_s < self.start_timestamp_s:
            raise ValueError("residual timestamps must be ordered")
        numerical = (
            self.raw_residual,
            self.normalized_residual,
            self.uncertainty,
            self.threshold,
        )
        if self.evaluable and any(value is None for value in numerical):
            raise ValueError("evaluable residuals require complete numerical fields")
        if not self.evaluable and any(value is not None for value in numerical):
            raise ValueError("non-evaluable residuals cannot carry numerical claims")
        flow_diagnostics = (self.flow_direction_error_deg, self.flow_magnitude_ratio)
        if not self.evaluable and any(value is not None for value in flow_diagnostics):
            raise ValueError("non-evaluable residuals cannot carry flow diagnostics")
        if self.flow_direction_error_deg is not None and self.flow_direction_error_deg > 180.0:
            raise ValueError("flow direction error must be within [0, 180] degrees")
        if self.evaluation_basis == EvaluationBasis.CHECK_EVIDENCE:
            if self.evidence_role != EvidenceRole.CHECK_ONLY:
                raise ValueError("check-evidence residuals require check_only role")
            if not self.evidence_keys or not self.evidence_artifacts:
                raise ValueError("check-evidence residuals require evidence lineage")
        if self.evaluation_basis == EvaluationBasis.FIT_EVIDENCE:
            if self.evidence_role != EvidenceRole.FIT:
                raise ValueError("fit-evidence residuals require fit role")
        if self.evaluation_basis == EvaluationBasis.NOT_EVALUABLE and self.evaluable:
            raise ValueError("not_evaluable basis cannot be evaluable")
        if self.severity == ResidualSeverity.HARD_VIOLATION and not self.hard_constraint:
            raise ValueError("hard violation severity requires a hard constraint")
        return self


class ConflictWindow(ContractModel):
    conflict_id: ContractId
    hypothesis_id: ContractId
    family: ResidualFamily
    constraint_id: ContractId
    start_frame_index: Annotated[int, Field(ge=0)]
    end_frame_index: Annotated[int, Field(ge=0)]
    residual_ids: tuple[ContractId, ...]
    peak_normalized_residual: NonNegativeFloat
    severity: ResidualSeverity
    component_ids: tuple[ContractId, ...]
    track_ids: tuple[ContractId, ...]
    check_evidence_supported: bool

    @model_validator(mode="after")
    def validate_window(self) -> "ConflictWindow":
        if self.end_frame_index < self.start_frame_index:
            raise ValueError("conflict window must be ordered")
        if not self.residual_ids:
            raise ValueError("conflict windows require at least one residual")
        return self


class ResidualFamilySummary(ContractModel):
    family: ResidualFamily
    total_count: Annotated[int, Field(ge=0)]
    evaluable_count: Annotated[int, Field(ge=0)]
    check_evidence_count: Annotated[int, Field(ge=0)]
    violation_count: Annotated[int, Field(ge=0)]
    hard_violation_count: Annotated[int, Field(ge=0)]
    peak_normalized_residual: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def validate_counts(self) -> "ResidualFamilySummary":
        if self.evaluable_count > self.total_count:
            raise ValueError("evaluable residual count exceeds total")
        if self.check_evidence_count > self.evaluable_count:
            raise ValueError("check residual count exceeds evaluable count")
        if self.hard_violation_count > self.violation_count:
            raise ValueError("hard violations must be included in violations")
        if (self.evaluable_count > 0) != (self.peak_normalized_residual is not None):
            raise ValueError("peak residual presence must match evaluable residuals")
        return self


class HypothesisResidualPacket(ContractModel):
    packet_id: ContractId
    hypothesis_id: ContractId
    hypothesis_rank: Annotated[int, Field(gt=0)]
    residuals: tuple[ResidualRecord, ...]
    conflict_windows: tuple[ConflictWindow, ...]
    family_summaries: tuple[ResidualFamilySummary, ...]
    evaluable_fraction: Probability
    check_evidence_residual_count: Annotated[int, Field(ge=0)]
    check_supported_conflict_count: Annotated[int, Field(ge=0)]
    hard_violation: bool
    status: Literal["no_conflict", "conflicts_detected", "insufficient_evidence"]
    repair_applied: Literal[False] = False
    selection_applied: Literal[False] = False

    @model_validator(mode="after")
    def validate_packet(self) -> "HypothesisResidualPacket":
        if any(row.hypothesis_id != self.hypothesis_id for row in self.residuals):
            raise ValueError("residual packet contains another hypothesis")
        residual_ids = {row.residual_id for row in self.residuals}
        if any(set(window.residual_ids) - residual_ids for window in self.conflict_windows):
            raise ValueError("conflict window references an unknown residual")
        expected_hard = any(
            row.severity == ResidualSeverity.HARD_VIOLATION for row in self.residuals
        )
        if self.hard_violation != expected_hard:
            raise ValueError("packet hard-violation flag does not match residuals")
        expected_check = sum(
            row.evaluation_basis == EvaluationBasis.CHECK_EVIDENCE
            for row in self.residuals
            if row.evaluable
        )
        if self.check_evidence_residual_count != expected_check:
            raise ValueError("packet check-evidence count does not match residuals")
        expected_conflicts = sum(row.check_evidence_supported for row in self.conflict_windows)
        if self.check_supported_conflict_count != expected_conflicts:
            raise ValueError("packet check-supported conflict count does not match windows")
        return self


class Step6ValidationSummary(ContractModel):
    input_hypothesis_count: Annotated[int, Field(gt=0)]
    evaluated_hypothesis_count: Annotated[int, Field(gt=0)]
    residual_count: Annotated[int, Field(ge=0)]
    evaluable_residual_count: Annotated[int, Field(ge=0)]
    check_evidence_residual_count: Annotated[int, Field(ge=0)]
    conflict_window_count: Annotated[int, Field(ge=0)]
    overall_pass: bool

    @model_validator(mode="after")
    def validate_counts(self) -> "Step6ValidationSummary":
        if self.evaluated_hypothesis_count > self.input_hypothesis_count:
            raise ValueError("evaluated hypotheses exceed input beam")
        if self.evaluable_residual_count > self.residual_count:
            raise ValueError("evaluable residuals exceed total residuals")
        if self.check_evidence_residual_count > self.evaluable_residual_count:
            raise ValueError("check residuals exceed evaluable residuals")
        if not self.overall_pass:
            raise ValueError("published Step 6 output requires closed accounting")
        return self


class VideoResidualManifest(ContractModel):
    schema_name: Literal["video_residual_manifest"] = "video_residual_manifest"
    schema_version: Literal[1] = 1

    run_id: ContractId
    video_id: ContractId
    source_world_state_sha256: Sha256
    config_sha256: Sha256
    canonical_fps: PositiveFloat
    image_size: ImageSize
    frame_count: Annotated[int, Field(gt=0)]
    input_snapshot: Step6InputSnapshot
    packets: tuple[HypothesisResidualPacket, ...]
    validation: Step6ValidationSummary
    tool_versions: tuple[ToolVersion, ...] = ()

    @model_validator(mode="after")
    def validate_packets(self) -> "VideoResidualManifest":
        if not self.packets:
            raise ValueError("Step 6 requires at least one hypothesis packet")
        ids = tuple(row.hypothesis_id for row in self.packets)
        if len(ids) != len(set(ids)):
            raise ValueError("Step 6 hypothesis packets must be unique")
        return self


class ResidualStore(ContractModel):
    schema_name: Literal["residual_store"] = "residual_store"
    schema_version: Literal[1] = 1

    run_id: ContractId
    source_world_state_store_sha256: Sha256
    config: Step6ConfigSnapshot
    config_sha256: Sha256
    video_ids: tuple[ContractId, ...]
    video_residuals: tuple[ArtifactRef, ...]

    @model_validator(mode="after")
    def validate_videos(self) -> "ResidualStore":
        if not self.video_ids or len(self.video_ids) != len(set(self.video_ids)):
            raise ValueError("residual store video IDs must be non-empty and unique")
        if len(self.video_ids) != len(self.video_residuals):
            raise ValueError("every video must have one residual manifest")
        return self
