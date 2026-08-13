"""Step 3 contracts for replayable multi-evidence object tracking."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .artifacts import ArtifactRef
from .base import (
    ArtifactOwner,
    AssociationDecision,
    ContractId,
    ContractModel,
    CueFamily,
    CueStatus,
    DepthRepresentation,
    DepthUnit,
    DetectionTier,
    EvidenceDisposition,
    EvidenceRole,
    FiniteFloat,
    GapStatus,
    ImageSize,
    MaskCandidateSource,
    NonNegativeFloat,
    PositiveFloat,
    Probability,
    Sha256,
    SupportObservability,
    ToolVersion,
    TrackMarkerType,
    TrackState,
)
from .evidence import BoundingBoxXYXY


class ArtifactLink(ContractModel):
    """Artifact reference plus the stage directory that owns its relative path."""

    owner: ArtifactOwner
    artifact: ArtifactRef


class Step3InputSnapshot(ContractModel):
    source_step2_relative_root: str
    source_video_manifest: ArtifactLink
    neural_evidence_store: ArtifactLink
    video_evidence_manifest: ArtifactLink
    input_artifacts: tuple[ArtifactLink, ...]

    @field_validator("source_step2_relative_root")
    @classmethod
    def validate_relative_root(cls, value: str) -> str:
        if not value or "\\" in value:
            raise ValueError("source Step 2 root must be a POSIX relative path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source Step 2 root must stay inside the run directory")
        return path.as_posix()

    @model_validator(mode="after")
    def validate_input_artifacts(self) -> "Step3InputSnapshot":
        if self.source_video_manifest.owner != ArtifactOwner.STEP1_INIT:
            raise ValueError("source video manifest must be owned by Step 1")
        if self.neural_evidence_store.owner != ArtifactOwner.STEP2_NEURAL_EVIDENCE:
            raise ValueError("neural evidence store must be owned by Step 2")
        if self.video_evidence_manifest.owner != ArtifactOwner.STEP2_NEURAL_EVIDENCE:
            raise ValueError("video evidence manifest must be owned by Step 2")
        keys = tuple(
            (link.owner, link.artifact.artifact_id, link.artifact.sha256)
            for link in self.input_artifacts
        )
        if len(keys) != len(set(keys)):
            raise ValueError("input artifact links must be unique")
        if any(link.owner != ArtifactOwner.STEP2_NEURAL_EVIDENCE for link in self.input_artifacts):
            raise ValueError("Step 3 input artifacts must be owned by Step 2")
        return self


class DepthDescriptor(ContractModel):
    representation: DepthRepresentation
    unit: DepthUnit
    support_source: Literal["eroded_mask", "inner_box"]
    valid_fraction: Probability
    minimum: FiniteFloat
    q25: FiniteFloat
    median: FiniteFloat
    q75: FiniteFloat
    maximum: FiniteFloat
    mad: NonNegativeFloat

    @model_validator(mode="after")
    def validate_statistics(self) -> "DepthDescriptor":
        if not self.minimum <= self.q25 <= self.median <= self.q75 <= self.maximum:
            raise ValueError("depth descriptor quantiles are inconsistent")
        expected_unit = (
            DepthUnit.METER
            if self.representation == DepthRepresentation.METRIC
            else DepthUnit.RELATIVE_UNIT
        )
        if self.unit != expected_unit:
            raise ValueError("depth descriptor unit does not match representation")
        return self


class FlowDescriptor(ContractModel):
    median_dx_px: FiniteFloat
    median_dy_px: FiniteFloat
    mad_px: NonNegativeFloat
    valid_fraction: Probability


class AssociationCueValue(ContractModel):
    cue_name: Literal["mask_iou", "flow_iou", "box_iou", "class", "depth"]
    available: bool
    value: Probability | None = None
    configured_weight: NonNegativeFloat
    effective_weight: Probability
    missing_reason: str | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> "AssociationCueValue":
        if self.available:
            if self.value is None or self.missing_reason is not None:
                raise ValueError("available association cues require a value only")
        else:
            if self.value is not None or not self.missing_reason:
                raise ValueError("missing association cues require a reason and no value")
            if self.effective_weight != 0.0:
                raise ValueError("missing association cues cannot carry effective weight")
        return self


class AssociationGate(ContractModel):
    gate_name: Literal["class", "center_distance"]
    passed: bool
    measured_value: FiniteFloat
    threshold: FiniteFloat
    reason: str


class AssociationLedgerRow(ContractModel):
    ledger_id: ContractId
    frame_index: Annotated[int, Field(ge=0)]
    track_id: ContractId
    detection_id: ContractId
    proposal_id: str | None = None
    track_age_frames: Annotated[int, Field(ge=0)]
    cues: tuple[AssociationCueValue, ...]
    gates: tuple[AssociationGate, ...]
    feasible: bool
    total_score: Probability
    rank_for_track: Annotated[int, Field(gt=0)]
    decision: AssociationDecision
    selected: bool
    decision_reason: str

    @model_validator(mode="after")
    def validate_decision(self) -> "AssociationLedgerRow":
        cue_names = tuple(cue.cue_name for cue in self.cues)
        if len(cue_names) != len(set(cue_names)):
            raise ValueError("association cue names must be unique")
        effective_sum = sum(cue.effective_weight for cue in self.cues)
        if abs(effective_sum - 1.0) > 1e-6:
            raise ValueError("association effective weights must sum to one")
        if self.feasible != all(gate.passed for gate in self.gates):
            raise ValueError("association feasibility must match its gates")
        if self.selected != (self.decision == AssociationDecision.MATCHED):
            raise ValueError("only matched ledger rows may be selected")
        if self.selected and not self.feasible:
            raise ValueError("a gated association cannot be selected")
        return self


class MaskCandidateRecord(ContractModel):
    candidate_id: ContractId
    track_id: ContractId
    frame_index: Annotated[int, Field(ge=0)]
    source: MaskCandidateSource
    observability: SupportObservability
    mask: ArtifactLink | None = None
    confidence: Probability | None = None
    detection_id: str | None = None
    proposal_id: str | None = None
    anchor_frame_index: Annotated[int, Field(ge=0)] | None = None
    parent_evidence_keys: tuple[ContractId, ...] = ()
    transform_id: ContractId
    selected: bool
    reason: str

    @model_validator(mode="after")
    def validate_support(self) -> "MaskCandidateRecord":
        no_pixels = {
            MaskCandidateSource.EMPTY_OR_OUTSIDE,
            MaskCandidateSource.EXPLICIT_UNOBSERVABLE,
        }
        if self.source in no_pixels:
            if self.mask is not None or self.observability != SupportObservability.UNOBSERVABLE:
                raise ValueError("empty/unobservable candidates cannot reference mask pixels")
        elif self.mask is None:
            raise ValueError("spatial mask candidates require an artifact link")
        if self.source == MaskCandidateSource.DIRECT_INSTANCE:
            if self.observability != SupportObservability.OBSERVED:
                raise ValueError("direct instance masks must be marked observed")
            if self.proposal_id is None:
                raise ValueError("direct instance masks require a proposal ID")
        return self


class TrackObservation(ContractModel):
    frame_index: Annotated[int, Field(ge=0)]
    timestamp_s: NonNegativeFloat
    detection_id: ContractId
    proposal_id: str | None = None
    class_name: ContractId
    detection_tier: DetectionTier
    confidence: Probability
    bbox: BoundingBoxXYXY
    selected_mask_candidate_id: str | None = None
    forward_flow: ArtifactLink | None = None
    backward_flow: ArtifactLink | None = None
    depth: ArtifactLink | None = None
    depth_descriptor: DepthDescriptor | None = None
    flow_descriptor: FlowDescriptor | None = None
    missing_cues: tuple[CueFamily, ...] = ()
    association_ledger_id: str | None = None


class TrackStateMarker(ContractModel):
    marker_id: ContractId
    marker_type: TrackMarkerType
    frame_index: Annotated[int, Field(ge=0)]
    state_after: TrackState
    operational_trigger: ContractId
    evidence_keys: tuple[ContractId, ...] = ()


class ObjectMaskTrack(ContractModel):
    track_id: ContractId
    primary_class: ContractId
    terminal_state: TrackState
    first_observed_frame: Annotated[int, Field(ge=0)]
    last_observed_frame: Annotated[int, Field(ge=0)]
    observations: tuple[TrackObservation, ...]
    state_markers: tuple[TrackStateMarker, ...]

    @model_validator(mode="after")
    def validate_timeline(self) -> "ObjectMaskTrack":
        if not self.observations:
            raise ValueError("tracks require at least one direct detection observation")
        frames = tuple(item.frame_index for item in self.observations)
        if frames != tuple(sorted(frames)) or len(frames) != len(set(frames)):
            raise ValueError("track observations must be unique and time ordered")
        if self.first_observed_frame != frames[0] or self.last_observed_frame != frames[-1]:
            raise ValueError("track endpoint fields must match observations")
        marker_ids = tuple(marker.marker_id for marker in self.state_markers)
        if len(marker_ids) != len(set(marker_ids)):
            raise ValueError("track marker IDs must be unique")
        return self


class UnassignedEvidenceRecord(ContractModel):
    unassigned_id: ContractId
    frame_index: Annotated[int, Field(ge=0)]
    detection_id: str | None = None
    proposal_id: str | None = None
    mask: ArtifactLink | None = None
    related_ledger_ids: tuple[ContractId, ...] = ()
    reason: ContractId

    @model_validator(mode="after")
    def validate_identity(self) -> "UnassignedEvidenceRecord":
        if self.detection_id is None and self.proposal_id is None:
            raise ValueError("unassigned evidence requires a detection or proposal")
        return self


class GapEvidenceRecord(ContractModel):
    gap_id: ContractId
    track_id: ContractId
    last_reliable_frame: Annotated[int, Field(ge=0)]
    next_reliable_frame: Annotated[int, Field(ge=0)] | None = None
    gap_frames: tuple[Annotated[int, Field(ge=0)], ...]
    status: GapStatus
    mask_candidate_ids: tuple[ContractId, ...]
    unassigned_evidence_ids: tuple[ContractId, ...]
    marker_ids: tuple[ContractId, ...]
    context_artifacts: tuple[ArtifactLink, ...]

    @model_validator(mode="after")
    def validate_gap(self) -> "GapEvidenceRecord":
        if not self.gap_frames:
            raise ValueError("gap records require at least one missing frame")
        if self.gap_frames != tuple(range(self.gap_frames[0], self.gap_frames[-1] + 1)):
            raise ValueError("gap frames must be contiguous")
        if self.gap_frames[0] <= self.last_reliable_frame:
            raise ValueError("gap must begin after the last reliable frame")
        if self.status == GapStatus.REOBSERVED:
            if self.next_reliable_frame is None or self.next_reliable_frame <= self.gap_frames[-1]:
                raise ValueError("reobserved gaps require a later reliable anchor")
        elif self.next_reliable_frame is not None:
            raise ValueError("open-ended gaps cannot have a next reliable frame")
        return self


class EvidenceDispositionRecord(ContractModel):
    evidence_key: ContractId
    cue_family: CueFamily
    frame_index: Annotated[int, Field(ge=0)]
    disposition: EvidenceDisposition
    track_id: str | None = None
    reason: ContractId


class ModalityRetentionCount(ContractModel):
    frame_index: Annotated[int, Field(ge=0)]
    cue_family: CueFamily
    input_count: Annotated[int, Field(ge=0)]
    selected_count: Annotated[int, Field(ge=0)]
    unselected_count: Annotated[int, Field(ge=0)]
    invalid_count: Annotated[int, Field(ge=0)]
    input_status: CueStatus

    @model_validator(mode="after")
    def validate_accounting(self) -> "ModalityRetentionCount":
        if self.input_count != self.selected_count + self.unselected_count + self.invalid_count:
            raise ValueError("retention counts do not close")
        if self.input_status == CueStatus.AVAILABLE and self.input_count == 0:
            raise ValueError("available modalities require at least one input record")
        if self.input_status != CueStatus.AVAILABLE and self.input_count != 0:
            raise ValueError("non-available modality status cannot carry input records")
        return self


class RetentionReport(ContractModel):
    modality_counts: tuple[ModalityRetentionCount, ...]
    input_artifact_count: Annotated[int, Field(ge=0)]
    verified_artifact_count: Annotated[int, Field(ge=0)]
    unresolved_evidence_keys: tuple[ContractId, ...] = ()
    hash_mismatch_evidence_keys: tuple[ContractId, ...] = ()
    shape_mismatch_evidence_keys: tuple[ContractId, ...] = ()
    expected_candidate_pairs: Annotated[int, Field(ge=0)]
    ledger_rows: Annotated[int, Field(ge=0)]
    required_track_frames: Annotated[int, Field(ge=0)]
    covered_track_frames: Annotated[int, Field(ge=0)]
    coverage_violations: tuple[ContractId, ...] = ()
    disposition_complete: bool
    overall_pass: bool

    @model_validator(mode="after")
    def validate_report(self) -> "RetentionReport":
        computed_pass = (
            self.input_artifact_count == self.verified_artifact_count
            and not self.unresolved_evidence_keys
            and not self.hash_mismatch_evidence_keys
            and not self.shape_mismatch_evidence_keys
            and self.expected_candidate_pairs == self.ledger_rows
            and self.required_track_frames == self.covered_track_frames
            and not self.coverage_violations
            and self.disposition_complete
        )
        if self.overall_pass != computed_pass:
            raise ValueError("retention report pass flag does not match its checks")
        return self


class EvidenceRoleAssignment(ContractModel):
    assignment_id: ContractId
    evidence_key: ContractId
    cue_family: CueFamily | None = None
    frame_index: Annotated[int, Field(ge=0)] | None = None
    artifact: ArtifactLink | None = None
    role: EvidenceRole
    allowed_consumers: tuple[ContractId, ...]
    prohibited_optimizers: tuple[ContractId, ...]
    selection_reason: ContractId


class EvidenceUsePlan(ContractModel):
    plan_id: ContractId
    policy_version: Literal["step3_evidence_roles_v1"] = "step3_evidence_roles_v1"
    random_seed: Annotated[int, Field(ge=0)]
    assignments: tuple[EvidenceRoleAssignment, ...]
    plan_sha256: Sha256


class TransformRecord(ContractModel):
    transform_id: ContractId
    transform_type: Literal["identity", "flow_forward_splat", "flow_backward_splat"]
    source_space: Literal["canonical_image_pixels"] = "canonical_image_pixels"
    target_space: Literal["canonical_image_pixels"] = "canonical_image_pixels"
    reversible: bool
    description: str


class Step3ConfigSnapshot(ContractModel):
    schema_name: Literal["step3_config"] = "step3_config"
    schema_version: Literal[1] = 1
    implementation_version: Literal["step03_object_tracking_v1"] = "step03_object_tracking_v1"

    max_age_frames: Annotated[int, Field(ge=0)]
    minimum_assignment_score: Probability
    maximum_center_distance_ratio: PositiveFloat
    hard_class_gate: bool
    bootstrap_primary_only: bool
    minimum_mask_area: Annotated[int, Field(gt=0)]
    depth_erosion_pixels: Annotated[int, Field(ge=0)]
    mask_iou_weight: NonNegativeFloat
    flow_iou_weight: NonNegativeFloat
    box_iou_weight: NonNegativeFloat
    class_weight: NonNegativeFloat
    depth_weight: NonNegativeFloat
    evidence_policy_seed: Annotated[int, Field(ge=0)]
    depth_check_fraction: Probability

    @model_validator(mode="after")
    def validate_weights(self) -> "Step3ConfigSnapshot":
        if sum(
            (
                self.mask_iou_weight,
                self.flow_iou_weight,
                self.box_iou_weight,
                self.class_weight,
                self.depth_weight,
            )
        ) <= 0.0:
            raise ValueError("at least one association weight must be positive")
        return self


class VideoTrackingManifest(ContractModel):
    schema_name: Literal["video_tracking_manifest"] = "video_tracking_manifest"
    schema_version: Literal[1] = 1

    run_id: ContractId
    video_id: ContractId
    source_evidence_sha256: Sha256
    config_sha256: Sha256
    canonical_fps: PositiveFloat
    image_size: ImageSize
    frame_count: Annotated[int, Field(gt=0)]
    input_snapshot: Step3InputSnapshot
    tracks: tuple[ObjectMaskTrack, ...]
    association_ledger: tuple[AssociationLedgerRow, ...]
    mask_candidate_bank: tuple[MaskCandidateRecord, ...]
    gap_records: tuple[GapEvidenceRecord, ...]
    unassigned_evidence: tuple[UnassignedEvidenceRecord, ...]
    evidence_dispositions: tuple[EvidenceDispositionRecord, ...]
    derived_artifacts: tuple[ArtifactLink, ...]
    transform_registry: tuple[TransformRecord, ...]
    evidence_use_plan: EvidenceUsePlan
    retention_report: RetentionReport
    tool_versions: tuple[ToolVersion, ...] = ()

    @model_validator(mode="after")
    def validate_package(self) -> "VideoTrackingManifest":
        track_ids = tuple(track.track_id for track in self.tracks)
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("track IDs must be unique")
        candidate_ids = tuple(item.candidate_id for item in self.mask_candidate_bank)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("mask candidate IDs must be unique")
        ledger_ids = tuple(item.ledger_id for item in self.association_ledger)
        if len(ledger_ids) != len(set(ledger_ids)):
            raise ValueError("association ledger IDs must be unique")
        disposition_keys = tuple(item.evidence_key for item in self.evidence_dispositions)
        if len(disposition_keys) != len(set(disposition_keys)):
            raise ValueError("every input evidence key must have one disposition")
        if not self.retention_report.overall_pass:
            raise ValueError("a published Step 3 package requires a passing retention report")
        return self


class TrackingStore(ContractModel):
    schema_name: Literal["tracking_store"] = "tracking_store"
    schema_version: Literal[1] = 1

    run_id: ContractId
    source_neural_evidence_store_sha256: Sha256
    config: Step3ConfigSnapshot
    config_sha256: Sha256
    video_ids: tuple[ContractId, ...]
    video_tracking: tuple[ArtifactRef, ...]

    @model_validator(mode="after")
    def validate_videos(self) -> "TrackingStore":
        if not self.video_ids or len(self.video_ids) != len(set(self.video_ids)):
            raise ValueError("tracking store video IDs must be non-empty and unique")
        if len(self.video_ids) != len(self.video_tracking):
            raise ValueError("every video must have one tracking manifest")
        return self
