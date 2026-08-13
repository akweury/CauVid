"""Step 4 contracts for observable, uncertainty-aware monocular geometry."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .artifacts import ArtifactRef
from .base import (
    ArtifactOwner,
    ContractId,
    ContractModel,
    CoordinateSpace,
    DepthRepresentation,
    DepthUnit,
    FiniteFloat,
    ImageSize,
    NonNegativeFloat,
    Observability,
    PositiveFloat,
    Probability,
    Sha256,
    ToolVersion,
)
from .evidence import BoundingBoxXYXY
from .tracking import ArtifactLink


class PixelCoordinate(ContractModel):
    u: FiniteFloat
    v: FiniteFloat


class Vector3D(ContractModel):
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat


class NonNegativeVector3D(ContractModel):
    x: NonNegativeFloat
    y: NonNegativeFloat
    z: NonNegativeFloat


class PointDistribution3D(ContractModel):
    q25: Vector3D
    median: Vector3D
    q75: Vector3D
    mad: NonNegativeVector3D

    @model_validator(mode="after")
    def validate_quantiles(self) -> "PointDistribution3D":
        for axis in ("x", "y", "z"):
            if not (
                getattr(self.q25, axis)
                <= getattr(self.median, axis)
                <= getattr(self.q75, axis)
            ):
                raise ValueError(f"3D {axis}-axis quantiles are inconsistent")
        return self


class CameraIntrinsicsHypothesis(ContractModel):
    intrinsics_id: ContractId
    source: Literal["provided_cli", "horizontal_fov_prior"]
    image_size: ImageSize
    fx_px: PositiveFloat
    fy_px: PositiveFloat
    cx_px: FiniteFloat
    cy_px: FiniteFloat
    horizontal_fov_deg: PositiveFloat | None = None
    horizontal_fov_interval_deg: tuple[PositiveFloat, PositiveFloat] | None = None
    assumption_driven: bool
    validated: bool
    coordinate_convention: Literal["x_right_y_down_z_forward"] = (
        "x_right_y_down_z_forward"
    )

    @model_validator(mode="after")
    def validate_intrinsics(self) -> "CameraIntrinsicsHypothesis":
        if not 0.0 <= self.cx_px <= self.image_size.width:
            raise ValueError("principal point cx must lie within the image")
        if not 0.0 <= self.cy_px <= self.image_size.height:
            raise ValueError("principal point cy must lie within the image")
        if self.source == "provided_cli":
            if self.horizontal_fov_interval_deg is not None or self.assumption_driven:
                raise ValueError("provided intrinsics cannot carry a prior interval")
        else:
            if self.horizontal_fov_deg is None or self.horizontal_fov_interval_deg is None:
                raise ValueError("FOV-prior intrinsics require nominal and interval values")
            lower, upper = self.horizontal_fov_interval_deg
            if not lower <= self.horizontal_fov_deg <= upper < 180.0:
                raise ValueError("horizontal FOV prior interval is inconsistent")
            if not self.assumption_driven or self.validated:
                raise ValueError("FOV-prior intrinsics must remain assumption-driven")
        return self


class ScaleHypothesis(ContractModel):
    scale_id: ContractId
    rank: Annotated[int, Field(gt=0)]
    source: Literal["metric_depth", "relative_monocular_depth", "no_depth"]
    observability: Observability
    depth_representation: DepthRepresentation
    scale_to_meters: PositiveFloat | None = None
    scale_interval_to_meters: tuple[PositiveFloat, PositiveFloat] | None = None
    evidence: tuple[ContractId, ...]
    limitations: tuple[ContractId, ...]

    @model_validator(mode="after")
    def validate_scale(self) -> "ScaleHypothesis":
        if self.observability == Observability.METRIC:
            if self.depth_representation != DepthRepresentation.METRIC:
                raise ValueError("metric scale requires metric depth")
            if self.scale_to_meters is None or self.scale_interval_to_meters is None:
                raise ValueError("metric scale requires a factor and interval")
            lower, upper = self.scale_interval_to_meters
            if not lower <= self.scale_to_meters <= upper:
                raise ValueError("metric scale interval does not contain its estimate")
        elif self.scale_to_meters is not None or self.scale_interval_to_meters is not None:
            raise ValueError("non-metric scale cannot claim a meters conversion")
        return self


class RelativeCameraPose(ContractModel):
    pose_id: ContractId
    source_frame_index: Annotated[int, Field(ge=0)]
    target_frame_index: Annotated[int, Field(ge=0)]
    source_timestamp_s: NonNegativeFloat
    target_timestamp_s: NonNegativeFloat
    rotation_source_to_target: tuple[FiniteFloat, ...]
    translation_direction_source_to_target: Vector3D
    correspondence_count: Annotated[int, Field(ge=5)]
    inlier_count: Annotated[int, Field(ge=0)]
    inlier_fraction: Probability
    median_epipolar_residual_px: NonNegativeFloat
    method: Literal["essential_matrix_background_flow"] = (
        "essential_matrix_background_flow"
    )

    @model_validator(mode="after")
    def validate_pose(self) -> "RelativeCameraPose":
        if self.target_frame_index <= self.source_frame_index:
            raise ValueError("relative pose target must follow its source")
        if self.target_timestamp_s <= self.source_timestamp_s:
            raise ValueError("relative pose timestamps must increase")
        if len(self.rotation_source_to_target) != 9:
            raise ValueError("rotation matrix must contain nine values")
        if self.inlier_count > self.correspondence_count:
            raise ValueError("pose inliers cannot exceed correspondences")
        expected = self.inlier_count / self.correspondence_count
        if abs(expected - self.inlier_fraction) > 1e-6:
            raise ValueError("pose inlier fraction does not match its counts")
        return self


class CameraMotionEstimate(ContractModel):
    observability: Observability
    poses: tuple[RelativeCameraPose, ...]
    failed_frame_pairs: tuple[tuple[int, int], ...]
    translation_scale: Literal["metric", "up_to_scale", "unobservable"]
    reason: str

    @model_validator(mode="after")
    def validate_motion(self) -> "CameraMotionEstimate":
        if self.poses:
            if self.observability not in {Observability.RELATIVE, Observability.METRIC}:
                raise ValueError("estimated poses require relative or metric observability")
        elif self.observability != Observability.UNOBSERVABLE:
            raise ValueError("an empty camera-motion estimate must be unobservable")
        return self


class GroundPlaneEstimate(ContractModel):
    observability: Observability
    normal_camera: Vector3D | None = None
    offset: FiniteFloat | None = None
    unit: DepthUnit | None = None
    method: Literal["not_estimated", "depth_road_plane"]
    reason: str

    @model_validator(mode="after")
    def validate_ground(self) -> "GroundPlaneEstimate":
        has_plane = self.normal_camera is not None
        if has_plane != (self.offset is not None and self.unit is not None):
            raise ValueError("ground plane normal, offset and unit must appear together")
        if not has_plane and self.observability != Observability.UNOBSERVABLE:
            raise ValueError("missing ground plane must be unobservable")
        return self


class GeometryObservation(ContractModel):
    observation_id: ContractId
    track_id: ContractId
    frame_index: Annotated[int, Field(ge=0)]
    timestamp_s: NonNegativeFloat
    detection_id: ContractId
    class_name: ContractId
    bbox: BoundingBoxXYXY
    coordinate_space: Literal[CoordinateSpace.CAMERA_3D] = CoordinateSpace.CAMERA_3D
    coordinate_unit: DepthUnit
    depth_representation: DepthRepresentation
    intrinsics_id: ContractId
    scale_id: ContractId
    support_source: Literal["eroded_mask", "inner_box"]
    support_pixel_count: Annotated[int, Field(gt=0)]
    valid_depth_pixel_count: Annotated[int, Field(gt=0)]
    valid_depth_fraction: Probability
    confidence_median: Probability | None = None
    pixel_centroid: PixelCoordinate
    points: PointDistribution3D
    median_reprojection_error_px: NonNegativeFloat
    source_artifacts: tuple[ArtifactLink, ...]
    validation_passed: bool
    validation_notes: tuple[ContractId, ...]

    @model_validator(mode="after")
    def validate_observation(self) -> "GeometryObservation":
        if self.valid_depth_pixel_count > self.support_pixel_count:
            raise ValueError("valid depth pixels cannot exceed support pixels")
        expected = self.valid_depth_pixel_count / self.support_pixel_count
        if abs(expected - self.valid_depth_fraction) > 1e-6:
            raise ValueError("valid depth fraction does not match its counts")
        expected_unit = (
            DepthUnit.METER
            if self.depth_representation == DepthRepresentation.METRIC
            else DepthUnit.RELATIVE_UNIT
        )
        if self.coordinate_unit != expected_unit:
            raise ValueError("3D coordinate unit does not match depth representation")
        if not self.source_artifacts:
            raise ValueError("geometry observations require source artifacts")
        return self


class UnavailableGeometryObservation(ContractModel):
    unavailable_id: ContractId
    track_id: ContractId
    frame_index: Annotated[int, Field(ge=0)]
    timestamp_s: NonNegativeFloat
    detection_id: ContractId
    reason: ContractId
    source_artifacts: tuple[ArtifactLink, ...] = ()


class ObjectGeometryTrack(ContractModel):
    track_id: ContractId
    primary_class: ContractId
    observations: tuple[GeometryObservation, ...]
    unavailable_observations: tuple[UnavailableGeometryObservation, ...]

    @model_validator(mode="after")
    def validate_timeline(self) -> "ObjectGeometryTrack":
        frames = tuple(item.frame_index for item in self.observations)
        if frames != tuple(sorted(frames)) or len(frames) != len(set(frames)):
            raise ValueError("geometry observations must be unique and time ordered")
        unavailable = tuple(item.frame_index for item in self.unavailable_observations)
        if unavailable != tuple(sorted(unavailable)) or len(unavailable) != len(set(unavailable)):
            raise ValueError("unavailable geometry observations must be unique and ordered")
        if set(frames) & set(unavailable):
            raise ValueError("a track frame cannot be both available and unavailable")
        return self


class GeometryValidationSummary(ContractModel):
    requested_observations: Annotated[int, Field(ge=0)]
    emitted_observations: Annotated[int, Field(ge=0)]
    unavailable_observations: Annotated[int, Field(ge=0)]
    verified_source_artifacts: Annotated[int, Field(ge=0)]
    failed_source_artifacts: tuple[ContractId, ...]
    passed_observations: Annotated[int, Field(ge=0)]
    failed_observations: Annotated[int, Field(ge=0)]
    overall_pass: bool

    @model_validator(mode="after")
    def validate_accounting(self) -> "GeometryValidationSummary":
        if self.requested_observations != (
            self.emitted_observations + self.unavailable_observations
        ):
            raise ValueError("Step 4 observation accounting does not close")
        if self.emitted_observations != self.passed_observations + self.failed_observations:
            raise ValueError("Step 4 validation accounting does not close")
        expected_pass = not self.failed_source_artifacts and self.failed_observations == 0
        if self.overall_pass != expected_pass:
            raise ValueError("Step 4 overall pass flag does not match validation")
        return self


class Step4InputSnapshot(ContractModel):
    source_step3_relative_root: str
    tracking_store: ArtifactLink
    video_tracking_manifest: ArtifactLink
    source_step2_relative_root: str

    @field_validator("source_step3_relative_root", "source_step2_relative_root")
    @classmethod
    def validate_relative_root(cls, value: str) -> str:
        if not value or "\\" in value:
            raise ValueError("source stage root must be a POSIX relative path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source stage root must stay inside the run directory")
        return path.as_posix()

    @model_validator(mode="after")
    def validate_owners(self) -> "Step4InputSnapshot":
        if self.tracking_store.owner != ArtifactOwner.STEP3_OBJECT_TRACKING:
            raise ValueError("tracking store must be owned by Step 3")
        if self.video_tracking_manifest.owner != ArtifactOwner.STEP3_OBJECT_TRACKING:
            raise ValueError("tracking manifest must be owned by Step 3")
        return self


class Step4ConfigSnapshot(ContractModel):
    schema_name: Literal["step4_config"] = "step4_config"
    schema_version: Literal[1] = 1
    implementation_version: Literal["step04_geometry_scale_v1"] = (
        "step04_geometry_scale_v1"
    )

    intrinsics_mode: Literal["provided_cli", "horizontal_fov_prior"]
    camera_fx_px: PositiveFloat | None = None
    camera_fy_px: PositiveFloat | None = None
    camera_cx_px: FiniteFloat | None = None
    camera_cy_px: FiniteFloat | None = None
    horizontal_fov_degrees: PositiveFloat
    horizontal_fov_min_degrees: PositiveFloat
    horizontal_fov_max_degrees: PositiveFloat
    support_erosion_pixels: Annotated[int, Field(ge=0)]
    bbox_inset_fraction: Annotated[float, Field(ge=0.0, lt=0.5, allow_inf_nan=False)]
    minimum_support_pixels: Annotated[int, Field(gt=0)]
    minimum_valid_depth_fraction: Probability
    maximum_median_reprojection_error_px: NonNegativeFloat
    background_flow_sample_stride: Annotated[int, Field(gt=0)]
    minimum_pose_correspondences: Annotated[int, Field(ge=8)]
    pose_ransac_threshold_px: PositiveFloat

    @model_validator(mode="after")
    def validate_config(self) -> "Step4ConfigSnapshot":
        provided = self.intrinsics_mode == "provided_cli"
        if provided != (self.camera_fx_px is not None and self.camera_fy_px is not None):
            raise ValueError("provided intrinsics mode requires fx and fy")
        if not (
            0.0
            < self.horizontal_fov_min_degrees
            <= self.horizontal_fov_degrees
            <= self.horizontal_fov_max_degrees
            < 180.0
        ):
            raise ValueError("horizontal FOV configuration is inconsistent")
        return self


class VideoGeometryManifest(ContractModel):
    schema_name: Literal["video_geometry_manifest"] = "video_geometry_manifest"
    schema_version: Literal[1] = 1

    run_id: ContractId
    video_id: ContractId
    source_tracking_sha256: Sha256
    config_sha256: Sha256
    canonical_fps: PositiveFloat
    image_size: ImageSize
    frame_count: Annotated[int, Field(gt=0)]
    input_snapshot: Step4InputSnapshot
    intrinsics: CameraIntrinsicsHypothesis
    camera_motion: CameraMotionEstimate
    ground_plane: GroundPlaneEstimate
    scale_hypotheses: tuple[ScaleHypothesis, ...]
    tracks: tuple[ObjectGeometryTrack, ...]
    validation: GeometryValidationSummary
    tool_versions: tuple[ToolVersion, ...] = ()

    @model_validator(mode="after")
    def validate_manifest(self) -> "VideoGeometryManifest":
        scale_ids = tuple(item.scale_id for item in self.scale_hypotheses)
        if not scale_ids or len(scale_ids) != len(set(scale_ids)):
            raise ValueError("geometry manifest requires unique scale hypotheses")
        track_ids = tuple(track.track_id for track in self.tracks)
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("geometry track IDs must be unique")
        known_scales = set(scale_ids)
        if any(
            observation.scale_id not in known_scales
            for track in self.tracks
            for observation in track.observations
        ):
            raise ValueError("geometry observation references an unknown scale")
        return self


class GeometryStore(ContractModel):
    schema_name: Literal["geometry_store"] = "geometry_store"
    schema_version: Literal[1] = 1

    run_id: ContractId
    source_tracking_store_sha256: Sha256
    config: Step4ConfigSnapshot
    config_sha256: Sha256
    video_ids: tuple[ContractId, ...]
    video_geometry: tuple[ArtifactRef, ...]

    @model_validator(mode="after")
    def validate_videos(self) -> "GeometryStore":
        if not self.video_ids or len(self.video_ids) != len(set(self.video_ids)):
            raise ValueError("geometry store video IDs must be non-empty and unique")
        if len(self.video_ids) != len(self.video_geometry):
            raise ValueError("every video must have one geometry manifest")
        return self
