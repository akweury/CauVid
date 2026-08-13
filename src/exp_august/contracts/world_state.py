"""Step 5 contracts for uncertainty-aware ego/object world reconstruction."""

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
    DepthUnit,
    FiniteFloat,
    ImageSize,
    MotionState,
    NonNegativeFloat,
    Observability,
    PositiveFloat,
    Probability,
    Sha256,
    ToolVersion,
)
from .geometry import NonNegativeVector3D, Vector3D
from .tracking import ArtifactLink


class Step5InputSnapshot(ContractModel):
    source_step4_relative_root: str
    geometry_store: ArtifactLink
    video_geometry_manifest: ArtifactLink
    source_step3_relative_root: str
    tracking_store: ArtifactLink
    video_tracking_manifest: ArtifactLink

    @field_validator("source_step4_relative_root", "source_step3_relative_root")
    @classmethod
    def validate_relative_root(cls, value: str) -> str:
        if not value or "\\" in value:
            raise ValueError("source stage root must be a POSIX relative path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source stage root must stay inside the run directory")
        return path.as_posix()

    @model_validator(mode="after")
    def validate_owners(self) -> "Step5InputSnapshot":
        if self.geometry_store.owner != ArtifactOwner.STEP4_GEOMETRY_SCALE:
            raise ValueError("geometry store must be owned by Step 4")
        if self.video_geometry_manifest.owner != ArtifactOwner.STEP4_GEOMETRY_SCALE:
            raise ValueError("geometry manifest must be owned by Step 4")
        if self.tracking_store.owner != ArtifactOwner.STEP3_OBJECT_TRACKING:
            raise ValueError("tracking store must be owned by Step 3")
        if self.video_tracking_manifest.owner != ArtifactOwner.STEP3_OBJECT_TRACKING:
            raise ValueError("tracking manifest must be owned by Step 3")
        return self


class Step5ConfigSnapshot(ContractModel):
    schema_name: Literal["step5_config"] = "step5_config"
    schema_version: Literal[1] = 1
    implementation_version: Literal["step05_joint_world_reconstruction_v1"] = (
        "step05_joint_world_reconstruction_v1"
    )

    top_k: Annotated[int, Field(gt=0, le=64)]
    minimum_motion_observations: Annotated[int, Field(ge=2)]
    static_displacement_threshold: PositiveFloat
    moving_displacement_threshold: PositiveFloat
    static_scale_residual_threshold: PositiveFloat
    fallback_scale_residual_threshold: PositiveFloat
    uncertainty_sigma_multiplier: PositiveFloat

    @model_validator(mode="after")
    def validate_thresholds(self) -> "Step5ConfigSnapshot":
        if self.moving_displacement_threshold <= self.static_displacement_threshold:
            raise ValueError("moving threshold must exceed static threshold")
        if self.static_scale_residual_threshold < self.fallback_scale_residual_threshold:
            raise ValueError("semantic-static scale gate cannot be tighter than fallback")
        return self


class EgoPoseState(ContractModel):
    pose_state_id: ContractId
    component_id: ContractId
    frame_index: Annotated[int, Field(ge=0)]
    timestamp_s: NonNegativeFloat
    coordinate_space: Literal[CoordinateSpace.COMPONENT_LOCAL_3D] = (
        CoordinateSpace.COMPONENT_LOCAL_3D
    )
    coordinate_unit: DepthUnit
    position: Vector3D
    position_std: NonNegativeVector3D
    rotation_world_to_camera: tuple[FiniteFloat, ...]
    translation_world_to_camera: Vector3D
    velocity: Vector3D | None = None
    speed: NonNegativeFloat | None = None
    speed_interval: tuple[NonNegativeFloat, NonNegativeFloat] | None = None
    source_pose_ids: tuple[ContractId, ...]
    observability: Observability

    @model_validator(mode="after")
    def validate_pose(self) -> "EgoPoseState":
        if len(self.rotation_world_to_camera) != 9:
            raise ValueError("ego pose rotation must contain nine values")
        velocity_fields = (self.velocity, self.speed, self.speed_interval)
        if any(value is None for value in velocity_fields) and not all(
            value is None for value in velocity_fields
        ):
            raise ValueError("ego velocity, speed and interval must appear together")
        if self.speed_interval is not None:
            lower, upper = self.speed_interval
            if not lower <= self.speed <= upper:
                raise ValueError("ego speed interval must contain its estimate")
        return self


class EgoPoseComponent(ContractModel):
    component_id: ContractId
    origin_frame_index: Annotated[int, Field(ge=0)]
    frame_indices: tuple[Annotated[int, Field(ge=0)], ...]
    pose_edge_ids: tuple[ContractId, ...]
    poses: tuple[EgoPoseState, ...]
    coordinate_unit: DepthUnit
    independent_origin: bool = True
    connection_status: Literal["component_local", "globally_aligned"] = (
        "component_local"
    )

    @model_validator(mode="after")
    def validate_component(self) -> "EgoPoseComponent":
        pose_frames = tuple(pose.frame_index for pose in self.poses)
        if not pose_frames or pose_frames != self.frame_indices:
            raise ValueError("component frame indices must match its ordered poses")
        if pose_frames != tuple(sorted(pose_frames)) or len(pose_frames) != len(set(pose_frames)):
            raise ValueError("component pose frames must be unique and ordered")
        if self.origin_frame_index != pose_frames[0]:
            raise ValueError("component origin must be its first pose frame")
        if any(pose.component_id != self.component_id for pose in self.poses):
            raise ValueError("ego poses must reference their enclosing component")
        if any(pose.coordinate_unit != self.coordinate_unit for pose in self.poses):
            raise ValueError("ego component coordinate units must be consistent")
        return self


class ObjectWorldObservation(ContractModel):
    state_id: ContractId
    geometry_observation_id: ContractId
    track_id: ContractId
    component_id: ContractId
    frame_index: Annotated[int, Field(ge=0)]
    timestamp_s: NonNegativeFloat
    coordinate_space: Literal[CoordinateSpace.COMPONENT_LOCAL_3D] = (
        CoordinateSpace.COMPONENT_LOCAL_3D
    )
    coordinate_unit: DepthUnit
    position: Vector3D
    position_std: NonNegativeVector3D
    velocity: Vector3D | None = None
    speed: NonNegativeFloat | None = None
    speed_interval: tuple[NonNegativeFloat, NonNegativeFloat] | None = None

    @model_validator(mode="after")
    def validate_motion(self) -> "ObjectWorldObservation":
        values = (self.velocity, self.speed, self.speed_interval)
        if any(value is None for value in values) and not all(value is None for value in values):
            raise ValueError("object velocity, speed and interval must appear together")
        if self.speed_interval is not None:
            lower, upper = self.speed_interval
            if not lower <= self.speed <= upper:
                raise ValueError("object speed interval must contain its estimate")
        return self


class ObjectTrajectoryHypothesis(ContractModel):
    trajectory_id: ContractId
    track_id: ContractId
    class_name: ContractId
    component_id: ContractId
    coordinate_unit: DepthUnit
    observations: tuple[ObjectWorldObservation, ...]
    unplaced_frame_indices: tuple[Annotated[int, Field(ge=0)], ...]
    motion_state: MotionState
    motion_score: NonNegativeFloat | None = None
    semantic_static_prior: bool
    evidence: tuple[ContractId, ...]
    limitations: tuple[ContractId, ...]

    @model_validator(mode="after")
    def validate_trajectory(self) -> "ObjectTrajectoryHypothesis":
        frames = tuple(row.frame_index for row in self.observations)
        if frames != tuple(sorted(frames)) or len(frames) != len(set(frames)):
            raise ValueError("object world observations must be unique and ordered")
        if any(row.track_id != self.track_id for row in self.observations):
            raise ValueError("object states must reference their enclosing track")
        if any(row.component_id != self.component_id for row in self.observations):
            raise ValueError("object states must remain inside one pose component")
        if any(row.coordinate_unit != self.coordinate_unit for row in self.observations):
            raise ValueError("object trajectory coordinate units must be consistent")
        if self.motion_state == MotionState.UNOBSERVABLE and self.motion_score is not None:
            raise ValueError("unobservable motion cannot claim a motion score")
        if self.motion_state != MotionState.UNOBSERVABLE and self.motion_score is None:
            raise ValueError("observable motion requires a motion score")
        return self


class WorldConstructionScore(ContractModel):
    geometry_support: Probability
    trajectory_coverage: Probability
    observability_support: Probability
    assumption_penalty: Probability
    total: Probability


class WorldHypothesis(ContractModel):
    hypothesis_id: ContractId
    rank: Annotated[int, Field(gt=0)]
    scale_id: ContractId
    scale_observability: Observability
    coordinate_unit: DepthUnit
    world_frame_status: Literal["global", "component_local", "unobservable"]
    metric_scale_claimed: bool
    ego_components: tuple[EgoPoseComponent, ...]
    object_trajectories: tuple[ObjectTrajectoryHypothesis, ...]
    unresolved_ego_frame_indices: tuple[Annotated[int, Field(ge=0)], ...]
    unresolved_object_observation_ids: tuple[ContractId, ...]
    discrete_choices: tuple[ContractId, ...]
    construction_score: WorldConstructionScore
    limitations: tuple[ContractId, ...]

    @model_validator(mode="after")
    def validate_hypothesis(self) -> "WorldHypothesis":
        component_ids = tuple(component.component_id for component in self.ego_components)
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("ego component IDs must be unique")
        known_components = set(component_ids)
        if any(row.component_id not in known_components for row in self.object_trajectories):
            raise ValueError("object trajectories must reference a known ego component")
        if self.metric_scale_claimed != (self.coordinate_unit == DepthUnit.METER):
            raise ValueError("metric claim must match the coordinate unit")
        if self.metric_scale_claimed and self.scale_observability != Observability.METRIC:
            raise ValueError("metric world hypotheses require metric scale observability")
        if not self.ego_components and self.world_frame_status != "unobservable":
            raise ValueError("missing ego components require an unobservable world frame")
        if len(self.ego_components) > 1 and self.world_frame_status == "global":
            raise ValueError("disconnected ego components cannot claim one global frame")
        return self


class HypothesisBeam(ContractModel):
    beam_id: ContractId
    iteration: Literal[0] = 0
    top_k: Annotated[int, Field(gt=0, le=64)]
    hypotheses: tuple[WorldHypothesis, ...]
    selection_method: Literal["initial_construction_score"] = "initial_construction_score"

    @model_validator(mode="after")
    def validate_beam(self) -> "HypothesisBeam":
        if not self.hypotheses:
            raise ValueError("initial hypothesis beam cannot be empty")
        if len(self.hypotheses) > self.top_k:
            raise ValueError("hypothesis beam exceeds top_k")
        ids = tuple(row.hypothesis_id for row in self.hypotheses)
        if len(ids) != len(set(ids)):
            raise ValueError("hypothesis IDs must be unique")
        ranks = tuple(row.rank for row in self.hypotheses)
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("hypothesis beam ranks must be contiguous and ordered")
        scores = tuple(row.construction_score.total for row in self.hypotheses)
        if scores != tuple(sorted(scores, reverse=True)):
            raise ValueError("initial hypothesis beam must be sorted by score")
        return self


class Step5ValidationSummary(ContractModel):
    input_pose_edges: Annotated[int, Field(ge=0)]
    emitted_pose_components: Annotated[int, Field(ge=0)]
    emitted_ego_poses: Annotated[int, Field(ge=0)]
    requested_object_observations: Annotated[int, Field(ge=0)]
    placed_object_observations: Annotated[int, Field(ge=0)]
    unplaced_object_observations: Annotated[int, Field(ge=0)]
    emitted_object_trajectories: Annotated[int, Field(ge=0)]
    hypothesis_count: Annotated[int, Field(gt=0)]
    overall_pass: bool

    @model_validator(mode="after")
    def validate_accounting(self) -> "Step5ValidationSummary":
        if self.requested_object_observations != (
            self.placed_object_observations + self.unplaced_object_observations
        ):
            raise ValueError("Step 5 object observation accounting does not close")
        if self.overall_pass != (self.hypothesis_count > 0):
            raise ValueError("Step 5 pass flag must match hypothesis availability")
        return self


class VideoWorldStateManifest(ContractModel):
    schema_name: Literal["video_world_state_manifest"] = "video_world_state_manifest"
    schema_version: Literal[1] = 1

    run_id: ContractId
    video_id: ContractId
    source_geometry_sha256: Sha256
    config_sha256: Sha256
    canonical_fps: PositiveFloat
    image_size: ImageSize
    frame_count: Annotated[int, Field(gt=0)]
    input_snapshot: Step5InputSnapshot
    initial_beam: HypothesisBeam
    validation: Step5ValidationSummary
    tool_versions: tuple[ToolVersion, ...] = ()


class WorldStateStore(ContractModel):
    schema_name: Literal["world_state_store"] = "world_state_store"
    schema_version: Literal[1] = 1

    run_id: ContractId
    source_geometry_store_sha256: Sha256
    config: Step5ConfigSnapshot
    config_sha256: Sha256
    video_ids: tuple[ContractId, ...]
    video_world_states: tuple[ArtifactRef, ...]

    @model_validator(mode="after")
    def validate_videos(self) -> "WorldStateStore":
        if not self.video_ids or len(self.video_ids) != len(set(self.video_ids)):
            raise ValueError("world-state store video IDs must be non-empty and unique")
        if len(self.video_ids) != len(self.video_world_states):
            raise ValueError("every video must have one world-state manifest")
        return self
