"""Step 1 contracts for validated videos and normalized timelines."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .artifacts import ArtifactRef
from .base import (
    ContractId,
    ContractModel,
    DecodeStatus,
    DecodeValidationMode,
    FiniteFloat,
    ImageSize,
    NonNegativeFloat,
    PositiveFloat,
    Sha256,
    TimelineSource,
    ToolVersion,
)


class FrameRecord(ContractModel):
    """One canonical frame and its reversible source-frame mapping."""

    frame_index: Annotated[int, Field(ge=0)]
    timestamp_s: NonNegativeFloat
    source_frame_index: Annotated[int, Field(ge=0)]
    source_timestamp_s: FiniteFloat
    sampling_error_s: NonNegativeFloat
    decode_status: DecodeStatus


class TimelineTransform(ContractModel):
    method: Literal["nearest_monotonic_timestamp"] = "nearest_monotonic_timestamp"
    source: TimelineSource
    source_time_origin_s: FiniteFloat
    canonical_fps: PositiveFloat
    mean_sampling_error_s: NonNegativeFloat
    max_sampling_error_s: NonNegativeFloat

    @model_validator(mode="after")
    def validate_error_summary(self) -> "TimelineTransform":
        if self.mean_sampling_error_s > self.max_sampling_error_s + 1e-12:
            raise ValueError("mean sampling error cannot exceed maximum sampling error")
        return self


class DecodeValidation(ContractModel):
    mode: DecodeValidationMode
    checked_frame_indices: tuple[Annotated[int, Field(ge=0)], ...]

    @model_validator(mode="after")
    def validate_checked_frames(self) -> "DecodeValidation":
        if tuple(sorted(set(self.checked_frame_indices))) != self.checked_frame_indices:
            raise ValueError("checked frame indices must be sorted and unique")
        if self.mode == DecodeValidationMode.NONE and self.checked_frame_indices:
            raise ValueError("decode mode 'none' cannot contain checked frames")
        return self


class VideoManifest(ContractModel):
    """Validated source video with a canonical, reversible frame timeline."""

    schema_name: Literal["video_manifest"] = "video_manifest"
    schema_version: Literal[1] = 1

    run_id: ContractId
    video_id: ContractId
    dataset_name: ContractId
    source_path: ContractId
    input_sha256: Sha256
    input_byte_size: Annotated[int, Field(gt=0)]
    container: ContractId
    codec: str | None = None

    encoded_image_size: ImageSize
    image_size: ImageSize
    display_rotation_degrees_clockwise: Annotated[int, Field(ge=0, lt=360)]
    orientation_applied: bool
    source_fps: PositiveFloat
    canonical_fps: PositiveFloat
    duration_s: PositiveFloat
    source_frame_count: Annotated[int, Field(gt=0)]
    canonical_frame_count: Annotated[int, Field(gt=0)]
    frames: tuple[FrameRecord, ...]

    timeline_transform: TimelineTransform
    decode_validation: DecodeValidation
    probe_backend: ContractId
    config_sha256: Sha256
    random_seed: int
    tool_versions: tuple[ToolVersion, ...] = ()

    @model_validator(mode="after")
    def validate_timeline(self) -> "VideoManifest":
        if self.canonical_frame_count != len(self.frames):
            raise ValueError("canonical_frame_count must equal the number of frame records")
        if self.orientation_applied != (self.display_rotation_degrees_clockwise != 0):
            raise ValueError("orientation_applied must match the display rotation")
        if self.display_rotation_degrees_clockwise not in {0, 90, 180, 270}:
            raise ValueError("Step 1 supports right-angle display rotations only")
        if self.canonical_fps > self.source_fps + 1e-6:
            raise ValueError("Step 1 currently supports downsampling only")
        expected_indices = tuple(range(len(self.frames)))
        if tuple(frame.frame_index for frame in self.frames) != expected_indices:
            raise ValueError("canonical frame indices must be contiguous and zero-based")
        canonical_times = tuple(frame.timestamp_s for frame in self.frames)
        source_indices = tuple(frame.source_frame_index for frame in self.frames)
        if any(right <= left for left, right in zip(canonical_times, canonical_times[1:])):
            raise ValueError("canonical timestamps must be strictly increasing")
        if any(right <= left for left, right in zip(source_indices, source_indices[1:])):
            raise ValueError("source frame mappings must be strictly increasing")
        checked = set(self.decode_validation.checked_frame_indices)
        if any(index >= len(self.frames) for index in checked):
            raise ValueError("decode validation references a frame outside the manifest")
        decoded = {frame.frame_index for frame in self.frames if frame.decode_status == DecodeStatus.DECODED}
        if decoded != checked:
            raise ValueError("per-frame decode status must match decode validation records")
        return self


class Step1ConfigSnapshot(ContractModel):
    schema_name: Literal["step1_config"] = "step1_config"
    schema_version: Literal[1] = 1
    implementation_version: Literal["step01_init_v1"] = "step01_init_v1"

    dataset_name: ContractId
    canonical_fps: PositiveFloat
    decode_validation_mode: DecodeValidationMode
    decode_sample_count: Annotated[int, Field(gt=0)]
    random_seed: int


class InitBundle(ContractModel):
    """Compact Step 1 output whose video manifests remain separate artifacts."""

    schema_name: Literal["init_bundle"] = "init_bundle"
    schema_version: Literal[1] = 1

    run_id: ContractId
    config: Step1ConfigSnapshot
    config_sha256: Sha256
    video_ids: tuple[ContractId, ...]
    video_manifests: tuple[ArtifactRef, ...]

    @model_validator(mode="after")
    def validate_video_references(self) -> "InitBundle":
        if not self.video_ids:
            raise ValueError("Step 1 must contain at least one video")
        if len(self.video_ids) != len(set(self.video_ids)):
            raise ValueError("video IDs must be unique")
        if len(self.video_ids) != len(self.video_manifests):
            raise ValueError("every video must have exactly one manifest reference")
        return self
