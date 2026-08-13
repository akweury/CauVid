"""Step 2 contracts for frame-local neural evidence."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .artifacts import ArtifactRef
from .base import (
    ContractId,
    ContractModel,
    CueFamily,
    CueStatus,
    DepthRepresentation,
    DepthUnit,
    DetectionTier,
    FiniteFloat,
    FlowDirection,
    ImageSize,
    NonNegativeFloat,
    PositiveFloat,
    Probability,
    Sha256,
    ToolVersion,
)


class BoundingBoxXYXY(ContractModel):
    x1: NonNegativeFloat
    y1: NonNegativeFloat
    x2: NonNegativeFloat
    y2: NonNegativeFloat

    @model_validator(mode="after")
    def validate_extent(self) -> "BoundingBoxXYXY":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding boxes must have positive width and height")
        return self


class CueDescriptor(ContractModel):
    cue_family: CueFamily
    status: CueStatus
    backend: ContractId
    model_id: str | None = None
    reason: str | None = None
    artifact_refs: tuple[ArtifactRef, ...] = ()

    @model_validator(mode="after")
    def validate_status(self) -> "CueDescriptor":
        if self.status in {CueStatus.UNAVAILABLE, CueStatus.FAILED, CueStatus.NOT_APPLICABLE}:
            if not self.reason:
                raise ValueError(f"cue status {self.status.value} requires a reason")
        elif self.reason:
            raise ValueError("available/empty cues cannot carry a failure reason")
        if self.status in {CueStatus.UNAVAILABLE, CueStatus.FAILED, CueStatus.NOT_APPLICABLE}:
            if self.artifact_refs:
                raise ValueError("non-evaluable cues cannot reference generated artifacts")
        return self


class DetectionObservation(ContractModel):
    detection_id: ContractId
    class_name: ContractId
    confidence: Probability
    bbox: BoundingBoxXYXY
    tier: DetectionTier
    model_id: ContractId
    rank: Annotated[int, Field(ge=0)]


class MaskObservation(ContractModel):
    proposal_id: ContractId
    prompt_detection_id: str | None = None
    confidence: Probability | None = None
    mask_ref: ArtifactRef
    model_id: ContractId
    area_pixels: Annotated[int, Field(gt=0)]


class FlowObservation(ContractModel):
    direction: FlowDirection
    source_frame_index: Annotated[int, Field(ge=0)]
    target_frame_index: Annotated[int, Field(ge=0)]
    source_timestamp_s: NonNegativeFloat
    target_timestamp_s: NonNegativeFloat
    field_ref: ArtifactRef
    valid_fraction: Probability
    median_fb_error_px: NonNegativeFloat | None = None
    p95_fb_error_px: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def validate_pair(self) -> "FlowObservation":
        expected_delta = 1 if self.direction == FlowDirection.FORWARD else -1
        if self.target_frame_index - self.source_frame_index != expected_delta:
            raise ValueError("flow direction does not match its source/target frames")
        if self.direction == FlowDirection.FORWARD:
            if self.target_timestamp_s <= self.source_timestamp_s:
                raise ValueError("forward flow target must be later than its source")
        elif self.target_timestamp_s >= self.source_timestamp_s:
            raise ValueError("backward flow target must be earlier than its source")
        if (self.median_fb_error_px is None) != (self.p95_fb_error_px is None):
            raise ValueError("flow consistency statistics must be present together")
        if (
            self.median_fb_error_px is not None
            and self.p95_fb_error_px is not None
            and self.median_fb_error_px > self.p95_fb_error_px + 1e-9
        ):
            raise ValueError("median flow error cannot exceed p95 flow error")
        return self


class DepthObservation(ContractModel):
    representation: DepthRepresentation
    unit: DepthUnit
    field_ref: ArtifactRef
    valid_fraction: Probability
    minimum: FiniteFloat
    median: FiniteFloat
    maximum: FiniteFloat
    has_confidence: bool
    model_id: ContractId
    context: Literal["single_frame"] = "single_frame"

    @model_validator(mode="after")
    def validate_depth_semantics(self) -> "DepthObservation":
        expected_unit = (
            DepthUnit.METER
            if self.representation == DepthRepresentation.METRIC
            else DepthUnit.RELATIVE_UNIT
        )
        if self.unit != expected_unit:
            raise ValueError("depth unit does not match its representation")
        if not self.minimum <= self.median <= self.maximum:
            raise ValueError("depth summary statistics are inconsistent")
        return self


class FrameEvidence(ContractModel):
    frame_index: Annotated[int, Field(ge=0)]
    timestamp_s: NonNegativeFloat
    source_frame_index: Annotated[int, Field(ge=0)]
    source_timestamp_s: FiniteFloat
    image_size: ImageSize

    object_cue: CueDescriptor
    detections: tuple[DetectionObservation, ...]
    mask_cue: CueDescriptor
    masks: tuple[MaskObservation, ...]
    forward_flow_cue: CueDescriptor
    forward_flow: FlowObservation | None = None
    backward_flow_cue: CueDescriptor
    backward_flow: FlowObservation | None = None
    depth_cue: CueDescriptor
    depth: DepthObservation | None = None

    @model_validator(mode="after")
    def validate_cue_payloads(self) -> "FrameEvidence":
        expected = (
            (self.object_cue, CueFamily.OBJECTS),
            (self.mask_cue, CueFamily.MASKS),
            (self.forward_flow_cue, CueFamily.FLOW_FORWARD),
            (self.backward_flow_cue, CueFamily.FLOW_BACKWARD),
            (self.depth_cue, CueFamily.DEPTH),
        )
        if any(cue.cue_family != family for cue, family in expected):
            raise ValueError("cue descriptors are attached to the wrong frame fields")
        if self.object_cue.status not in {
            CueStatus.AVAILABLE,
            CueStatus.EMPTY,
            CueStatus.UNAVAILABLE,
            CueStatus.FAILED,
        }:
            raise ValueError("object evidence cannot be marked not_applicable")
        if self.mask_cue.status not in {
            CueStatus.AVAILABLE,
            CueStatus.EMPTY,
            CueStatus.UNAVAILABLE,
            CueStatus.FAILED,
            CueStatus.NOT_APPLICABLE,
        }:
            raise ValueError("invalid mask evidence status")
        if bool(self.detections) != (self.object_cue.status == CueStatus.AVAILABLE):
            if not (not self.detections and self.object_cue.status in {CueStatus.EMPTY, CueStatus.UNAVAILABLE, CueStatus.FAILED}):
                raise ValueError("object cue status does not match detection payload")
        if bool(self.masks) != (self.mask_cue.status == CueStatus.AVAILABLE):
            if not (
                not self.masks
                and self.mask_cue.status
                in {
                    CueStatus.EMPTY,
                    CueStatus.UNAVAILABLE,
                    CueStatus.FAILED,
                    CueStatus.NOT_APPLICABLE,
                }
            ):
                raise ValueError("mask cue status does not match mask payload")
        for cue, observation in (
            (self.forward_flow_cue, self.forward_flow),
            (self.backward_flow_cue, self.backward_flow),
            (self.depth_cue, self.depth),
        ):
            if (observation is not None) != (cue.status == CueStatus.AVAILABLE):
                raise ValueError(
                    f"{cue.cue_family.value} cue status does not match its observation"
                )
        detection_ids = tuple(item.detection_id for item in self.detections)
        if len(detection_ids) != len(set(detection_ids)):
            raise ValueError("detection IDs must be unique within a frame")
        if any(
            detection.bbox.x2 > self.image_size.width
            or detection.bbox.y2 > self.image_size.height
            for detection in self.detections
        ):
            raise ValueError("detection bounding box exceeds canonical image bounds")
        if any(
            mask.prompt_detection_id is not None
            and mask.prompt_detection_id not in detection_ids
            for mask in self.masks
        ):
            raise ValueError("mask prompt references an unknown frame-local detection")
        return self


class Step2ConfigSnapshot(ContractModel):
    schema_name: Literal["step2_config"] = "step2_config"
    schema_version: Literal[2] = 2
    implementation_version: Literal["step02_neural_evidence_v2"] = "step02_neural_evidence_v2"

    object_backend: ContractId
    object_model: ContractId
    object_classes: tuple[ContractId, ...]
    primary_confidence: Probability
    candidate_confidence: Probability
    nms_iou: Probability
    inference_size: Annotated[int, Field(gt=0)]
    batch_size: Annotated[int, Field(gt=0)]
    device: ContractId
    masks_backend: ContractId
    masks_model: ContractId
    mask_prompt_candidates: bool
    flow_backend: ContractId
    flow_model: ContractId
    flow_consistency_threshold_px: PositiveFloat
    depth_backend: ContractId
    depth_model: ContractId
    depth_process_resolution: Annotated[int, Field(gt=0)]
    depth_representation: DepthRepresentation

    @model_validator(mode="after")
    def validate_thresholds(self) -> "Step2ConfigSnapshot":
        if self.candidate_confidence > self.primary_confidence:
            raise ValueError("candidate confidence cannot exceed primary confidence")
        return self


class VideoEvidenceManifest(ContractModel):
    schema_name: Literal["video_evidence_manifest"] = "video_evidence_manifest"
    schema_version: Literal[2] = 2

    run_id: ContractId
    video_id: ContractId
    source_manifest: ArtifactRef
    source_manifest_sha256: Sha256
    config_sha256: Sha256
    canonical_fps: PositiveFloat
    image_size: ImageSize
    frame_count: Annotated[int, Field(gt=0)]
    frames: tuple[FrameEvidence, ...]
    tool_versions: tuple[ToolVersion, ...] = ()

    @model_validator(mode="after")
    def validate_frames(self) -> "VideoEvidenceManifest":
        if self.frame_count != len(self.frames):
            raise ValueError("frame_count must equal the number of evidence frames")
        if tuple(frame.frame_index for frame in self.frames) != tuple(range(self.frame_count)):
            raise ValueError("evidence frames must be contiguous and zero-based")
        if any(right.timestamp_s <= left.timestamp_s for left, right in zip(self.frames, self.frames[1:])):
            raise ValueError("evidence timestamps must be strictly increasing")
        return self


class NeuralEvidenceStore(ContractModel):
    schema_name: Literal["neural_evidence_store"] = "neural_evidence_store"
    schema_version: Literal[2] = 2

    run_id: ContractId
    source_init_bundle_sha256: Sha256
    config: Step2ConfigSnapshot
    config_sha256: Sha256
    video_ids: tuple[ContractId, ...]
    video_evidence: tuple[ArtifactRef, ...]

    @model_validator(mode="after")
    def validate_video_references(self) -> "NeuralEvidenceStore":
        if not self.video_ids:
            raise ValueError("Step 2 must contain at least one video")
        if len(self.video_ids) != len(set(self.video_ids)):
            raise ValueError("video IDs must be unique")
        if len(self.video_ids) != len(self.video_evidence):
            raise ValueError("every video must have exactly one evidence manifest")
        return self
