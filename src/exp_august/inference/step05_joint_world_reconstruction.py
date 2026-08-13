"""Target Step 5: construct initial ego/object world-state hypotheses.

The implementation is deliberately conservative.  It accumulates only camera
pose edges supplied by Step 4, keeps disconnected components independent, and
never converts relative monocular coordinates into meters.  The result is the
initial hypothesis beam B0; evidence prediction and iterative repair belong to
Steps 6-9.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.exp_august.contracts import (
    ArtifactLink,
    ArtifactOwner,
    ArtifactRef,
    CoordinateSpace,
    DepthUnit,
    EgoPoseComponent,
    EgoPoseState,
    GeometryStore,
    HypothesisBeam,
    MotionState,
    NonNegativeVector3D,
    ObjectTrajectoryHypothesis,
    ObjectWorldObservation,
    Observability,
    Step5ConfigSnapshot,
    Step5InputSnapshot,
    Step5ValidationSummary,
    ToolVersion,
    Vector3D,
    VideoGeometryManifest,
    VideoWorldStateManifest,
    WorldConstructionScore,
    WorldHypothesis,
    WorldStateStore,
)
from src.exp_august.contracts.codec import (
    hash_payload,
    read_contract,
    sha256_file,
    write_contract,
)


_STATIC_CLASS_TERMS = (
    "traffic light",
    "stop sign",
    "traffic sign",
    "road sign",
    "street sign",
    "fire hydrant",
    "parking meter",
    "lamp post",
    "lamppost",
    "utility pole",
    "building",
)


@dataclass(frozen=True)
class Step5Result:
    store: WorldStateStore
    video_manifests: tuple[VideoWorldStateManifest, ...]
    stage_root: Path
    store_path: Path


@dataclass(frozen=True)
class _LoadedStep4:
    store_path: Path
    stage_root: Path
    run_root: Path
    store: GeometryStore
    manifests: tuple[VideoGeometryManifest, ...]
    manifest_refs: tuple[ArtifactRef, ...]


@dataclass(frozen=True)
class _PoseEdge:
    pose_id: str
    source_frame: int
    target_frame: int
    rotation: np.ndarray
    direction: np.ndarray
    translation_scale: float
    translation_scale_std: float
    evidence_mode: str
    evidence_track_ids: tuple[str, ...]
    support_score: float


@dataclass(frozen=True)
class _FrameTransform:
    rotation_world_to_camera: np.ndarray
    translation_world_to_camera: np.ndarray
    component_id: str
    position_std: float
    source_pose_ids: tuple[str, ...]


def _vector(values: np.ndarray) -> Vector3D:
    return Vector3D(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _nonnegative_vector(values: np.ndarray) -> NonNegativeVector3D:
    return NonNegativeVector3D(
        x=float(max(0.0, values[0])),
        y=float(max(0.0, values[1])),
        z=float(max(0.0, values[2])),
    )


def _file_reference(*, path: Path, stage_root: Path, artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        relative_path=path.relative_to(stage_root).as_posix(),
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        media_type="application/json",
        coordinate_space=None,
    )


def _step4_link(reference: ArtifactRef) -> ArtifactLink:
    return ArtifactLink(owner=ArtifactOwner.STEP4_GEOMETRY_SCALE, artifact=reference)


def _load_step4(store_path: Path | str) -> _LoadedStep4:
    resolved = Path(store_path).expanduser().resolve()
    store = read_contract(resolved, GeometryStore)
    stage_root = resolved.parent
    if stage_root.parent.name != "04_geometry_scale":
        raise RuntimeError("Step 4 store must live inside 04_geometry_scale/config_<hash>")
    run_root = stage_root.parent.parent
    manifests = []
    for video_id, reference in zip(store.video_ids, store.video_geometry):
        path = stage_root / reference.relative_path
        if not path.is_file() or path.stat().st_size != reference.byte_size:
            raise RuntimeError(f"Step 4 geometry manifest is missing or truncated: {path}")
        if sha256_file(path) != reference.sha256:
            raise RuntimeError(f"Step 4 geometry manifest failed integrity check: {path}")
        manifest = read_contract(path, VideoGeometryManifest)
        if manifest.video_id != video_id or manifest.run_id != store.run_id:
            raise RuntimeError(f"Step 4 geometry identity mismatch: {path}")
        manifests.append(manifest)
    return _LoadedStep4(
        store_path=resolved,
        stage_root=stage_root,
        run_root=run_root,
        store=store,
        manifests=tuple(manifests),
        manifest_refs=store.video_geometry,
    )


def _semantic_static_prior(class_name: str) -> bool:
    normalized = " ".join(class_name.lower().replace("_", " ").split())
    return any(term in normalized for term in _STATIC_CLASS_TERMS)


def _median_point(observation) -> np.ndarray:
    point = observation.points.median
    return np.asarray((point.x, point.y, point.z), dtype=np.float64)


def _point_mad(observation) -> np.ndarray:
    point = observation.points.mad
    return np.asarray((point.x, point.y, point.z), dtype=np.float64)


def _pose_edges(
    manifest: VideoGeometryManifest,
    *,
    scale_id: str,
    coordinate_unit: DepthUnit,
    config: Step5ConfigSnapshot,
) -> tuple[tuple[_PoseEdge, ...], tuple[str, ...]]:
    tracks = {
        track.track_id: track
        for track in manifest.tracks
        if any(row.scale_id == scale_id for row in track.observations)
    }
    frame_points = {
        track_id: {
            row.frame_index: _median_point(row)
            for row in track.observations
            if row.scale_id == scale_id and row.validation_passed
        }
        for track_id, track in tracks.items()
    }
    semantic_static_ids = {
        track_id
        for track_id, track in tracks.items()
        if _semantic_static_prior(track.primary_class) and len(frame_points[track_id]) >= 2
    }
    rows: list[dict] = []
    supported_scales: list[float] = []
    limitations: set[str] = set()
    for pose in sorted(
        manifest.camera_motion.poses,
        key=lambda item: (item.source_frame_index, item.target_frame_index),
    ):
        rotation = np.asarray(pose.rotation_source_to_target, dtype=np.float64).reshape(3, 3)
        direction = np.asarray(
            (
                pose.translation_direction_source_to_target.x,
                pose.translation_direction_source_to_target.y,
                pose.translation_direction_source_to_target.z,
            ),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-9:
            limitations.add("zero_translation_direction_edge_omitted")
            continue
        direction /= norm
        candidates = []
        for track_id, points in frame_points.items():
            if pose.source_frame_index not in points or pose.target_frame_index not in points:
                continue
            source_point = points[pose.source_frame_index]
            target_point = points[pose.target_frame_index]
            difference = target_point - rotation @ source_point
            raw_scale = float(np.dot(difference, direction))
            perpendicular = difference - raw_scale * direction
            reference_depth = max(
                1e-6,
                0.5 * (float(np.linalg.norm(source_point)) + float(np.linalg.norm(target_point))),
            )
            candidates.append(
                {
                    "track_id": track_id,
                    "semantic_static": track_id in semantic_static_ids,
                    "raw_scale": raw_scale,
                    "normalized_residual": float(np.linalg.norm(perpendicular) / reference_depth),
                }
            )
        semantic = [
            row
            for row in candidates
            if row["semantic_static"]
            and row["raw_scale"] > 1e-6
            and row["normalized_residual"] <= config.static_scale_residual_threshold
        ]
        fallback = [
            row
            for row in candidates
            if row["raw_scale"] > 1e-6
            and row["normalized_residual"] <= config.fallback_scale_residual_threshold
        ]
        selected = semantic or fallback
        evidence_mode = (
            "semantic_static_tracks"
            if semantic
            else "low_motion_residual_fallback"
            if fallback
            else "unit_translation_fallback"
        )
        if selected:
            values = np.asarray([row["raw_scale"] for row in selected], dtype=np.float64)
            raw_scale = float(np.median(values))
            raw_scale_std = float(np.median(np.abs(values - raw_scale)))
            supported_scales.append(raw_scale)
        else:
            raw_scale = None
            raw_scale_std = None
        rows.append(
            {
                "pose": pose,
                "rotation": rotation,
                "direction": direction,
                "raw_scale": raw_scale,
                "raw_scale_std": raw_scale_std,
                "evidence_mode": evidence_mode,
                "track_ids": tuple(sorted(row["track_id"] for row in selected)),
            }
        )

    normalizer = 1.0
    if coordinate_unit == DepthUnit.RELATIVE_UNIT and supported_scales:
        normalizer = float(np.median(np.asarray(supported_scales, dtype=np.float64)))
        if not np.isfinite(normalizer) or normalizer <= 1e-9:
            normalizer = 1.0
    edges = []
    for row in rows:
        raw_scale = row["raw_scale"]
        if raw_scale is None and coordinate_unit == DepthUnit.METER:
            limitations.add("metric_pose_edge_without_translation_magnitude_omitted")
            continue
        translation_scale = float(raw_scale / normalizer) if raw_scale is not None else 1.0
        raw_std = row["raw_scale_std"]
        if raw_std is None:
            translation_scale_std = max(0.5, abs(translation_scale) * 0.5)
            limitations.add("unit_translation_fallback_used")
        else:
            translation_scale_std = max(
                0.02 * abs(translation_scale), float(raw_std / normalizer)
            )
        pose = row["pose"]
        support_score = float(
            np.clip(
                pose.inlier_fraction
                / (1.0 + pose.median_epipolar_residual_px)
                * (1.0 if row["evidence_mode"] == "semantic_static_tracks" else 0.8),
                0.0,
                1.0,
            )
        )
        edges.append(
            _PoseEdge(
                pose_id=pose.pose_id,
                source_frame=pose.source_frame_index,
                target_frame=pose.target_frame_index,
                rotation=row["rotation"],
                direction=row["direction"],
                translation_scale=translation_scale,
                translation_scale_std=translation_scale_std,
                evidence_mode=row["evidence_mode"],
                evidence_track_ids=row["track_ids"],
                support_score=support_score,
            )
        )
    if coordinate_unit == DepthUnit.RELATIVE_UNIT:
        limitations.add("monocular_metric_scale_unobservable")
        limitations.add("relative_depth_scale_may_drift_between_frames")
    return tuple(edges), tuple(sorted(limitations))


def _accumulate_components(
    edges: tuple[_PoseEdge, ...],
) -> tuple[dict[int, _FrameTransform], dict[str, tuple[_PoseEdge, ...]], tuple[str, ...]]:
    adjacency: dict[int, list[tuple[int, _PoseEdge, bool]]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.source_frame].append((edge.target_frame, edge, True))
        adjacency[edge.target_frame].append((edge.source_frame, edge, False))
    transforms: dict[int, _FrameTransform] = {}
    component_edges: dict[str, list[_PoseEdge]] = defaultdict(list)
    limitations: set[str] = set()
    component_index = 0
    for origin in sorted(adjacency):
        if origin in transforms:
            continue
        component_id = f"component:{component_index:03d}"
        component_index += 1
        transforms[origin] = _FrameTransform(
            rotation_world_to_camera=np.eye(3, dtype=np.float64),
            translation_world_to_camera=np.zeros(3, dtype=np.float64),
            component_id=component_id,
            position_std=0.0,
            source_pose_ids=(),
        )
        queue: deque[int] = deque((origin,))
        seen_edges: set[str] = set()
        while queue:
            frame = queue.popleft()
            current = transforms[frame]
            for neighbor, edge, forward in sorted(adjacency[frame], key=lambda row: row[0]):
                if edge.pose_id not in seen_edges:
                    component_edges[component_id].append(edge)
                    seen_edges.add(edge.pose_id)
                if forward:
                    rotation = edge.rotation @ current.rotation_world_to_camera
                    translation = (
                        edge.rotation @ current.translation_world_to_camera
                        + edge.translation_scale * edge.direction
                    )
                else:
                    rotation = edge.rotation.T @ current.rotation_world_to_camera
                    translation = edge.rotation.T @ (
                        current.translation_world_to_camera
                        - edge.translation_scale * edge.direction
                    )
                candidate = _FrameTransform(
                    rotation_world_to_camera=rotation,
                    translation_world_to_camera=translation,
                    component_id=component_id,
                    position_std=float(
                        np.hypot(current.position_std, edge.translation_scale_std)
                    ),
                    source_pose_ids=(*current.source_pose_ids, edge.pose_id),
                )
                existing = transforms.get(neighbor)
                if existing is None:
                    transforms[neighbor] = candidate
                    queue.append(neighbor)
                elif existing.component_id == component_id:
                    existing_center = -existing.rotation_world_to_camera.T @ existing.translation_world_to_camera
                    candidate_center = -candidate.rotation_world_to_camera.T @ candidate.translation_world_to_camera
                    if float(np.linalg.norm(existing_center - candidate_center)) > max(
                        0.25, 2.0 * (existing.position_std + candidate.position_std)
                    ):
                        limitations.add("pose_cycle_closure_inconsistent")
    return (
        transforms,
        {key: tuple(value) for key, value in component_edges.items()},
        tuple(sorted(limitations)),
    )


def _motion_derivatives(
    positions: np.ndarray,
    timestamps: np.ndarray,
    position_std: np.ndarray,
    sigma_multiplier: float,
) -> list[tuple[np.ndarray | None, float | None, tuple[float, float] | None]]:
    count = len(positions)
    if count < 2:
        return [(None, None, None)] * count
    output = []
    for index in range(count):
        left = max(0, index - 1)
        right = min(count - 1, index + 1)
        if left == right:
            output.append((None, None, None))
            continue
        delta_t = float(timestamps[right] - timestamps[left])
        if delta_t <= 1e-9:
            output.append((None, None, None))
            continue
        velocity = (positions[right] - positions[left]) / delta_t
        speed = float(np.linalg.norm(velocity))
        displacement_std = np.sqrt(
            np.square(position_std[right]) + np.square(position_std[left])
        )
        uncertainty = float(
            sigma_multiplier * np.linalg.norm(displacement_std) / delta_t
        )
        output.append((velocity, speed, (max(0.0, speed - uncertainty), speed + uncertainty)))
    return output


def _ego_components(
    *,
    manifest: VideoGeometryManifest,
    transforms: dict[int, _FrameTransform],
    component_edges: dict[str, tuple[_PoseEdge, ...]],
    coordinate_unit: DepthUnit,
    config: Step5ConfigSnapshot,
) -> tuple[EgoPoseComponent, ...]:
    timestamp_by_frame: dict[int, float] = {}
    for pose in manifest.camera_motion.poses:
        timestamp_by_frame[pose.source_frame_index] = pose.source_timestamp_s
        timestamp_by_frame[pose.target_frame_index] = pose.target_timestamp_s
    by_component: dict[str, list[tuple[int, _FrameTransform]]] = defaultdict(list)
    for frame, transform in sorted(transforms.items()):
        by_component[transform.component_id].append((frame, transform))
    components = []
    for component_id, rows in sorted(by_component.items()):
        frames = np.asarray([frame for frame, _ in rows], dtype=np.int64)
        timestamps = np.asarray(
            [timestamp_by_frame.get(int(frame), float(frame) / manifest.canonical_fps) for frame in frames],
            dtype=np.float64,
        )
        positions = np.asarray(
            [
                -transform.rotation_world_to_camera.T @ transform.translation_world_to_camera
                for _, transform in rows
            ],
            dtype=np.float64,
        )
        std = np.asarray(
            [np.full(3, transform.position_std, dtype=np.float64) for _, transform in rows]
        )
        derivatives = _motion_derivatives(
            positions, timestamps, std, config.uncertainty_sigma_multiplier
        )
        poses = []
        for (frame, transform), timestamp, position, sigma, derivative in zip(
            rows, timestamps, positions, std, derivatives
        ):
            velocity, speed, interval = derivative
            poses.append(
                EgoPoseState(
                    pose_state_id=f"ego:{component_id}:{frame}",
                    component_id=component_id,
                    frame_index=frame,
                    timestamp_s=float(timestamp),
                    coordinate_unit=coordinate_unit,
                    position=_vector(position),
                    position_std=_nonnegative_vector(sigma),
                    rotation_world_to_camera=tuple(
                        float(value) for value in transform.rotation_world_to_camera.reshape(-1)
                    ),
                    translation_world_to_camera=_vector(transform.translation_world_to_camera),
                    velocity=_vector(velocity) if velocity is not None else None,
                    speed=speed,
                    speed_interval=interval,
                    source_pose_ids=transform.source_pose_ids,
                    observability=(
                        Observability.METRIC
                        if coordinate_unit == DepthUnit.METER
                        else Observability.RELATIVE
                    ),
                )
            )
        edge_ids = tuple(edge.pose_id for edge in component_edges.get(component_id, ()))
        components.append(
            EgoPoseComponent(
                component_id=component_id,
                origin_frame_index=int(frames[0]),
                frame_indices=tuple(int(value) for value in frames),
                pose_edge_ids=edge_ids,
                poses=tuple(poses),
                coordinate_unit=coordinate_unit,
            )
        )
    return tuple(components)


def _motion_classification(
    *,
    positions: np.ndarray,
    position_std: np.ndarray,
    config: Step5ConfigSnapshot,
) -> tuple[MotionState, float | None]:
    if len(positions) < config.minimum_motion_observations:
        return MotionState.UNOBSERVABLE, None
    center = np.median(positions, axis=0)
    radial = np.linalg.norm(positions - center, axis=1)
    endpoint_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    motion_score = max(endpoint_displacement, float(2.0 * np.median(radial)))
    uncertainty = float(
        config.uncertainty_sigma_multiplier
        * np.median(np.linalg.norm(position_std, axis=1))
    )
    lower = max(0.0, motion_score - uncertainty)
    upper = motion_score + uncertainty
    if upper <= config.static_displacement_threshold:
        return MotionState.STATIC, motion_score
    if lower >= config.moving_displacement_threshold:
        return MotionState.MOVING, motion_score
    return MotionState.AMBIGUOUS, motion_score


def _object_trajectories(
    *,
    manifest: VideoGeometryManifest,
    scale_id: str,
    transforms: dict[int, _FrameTransform],
    coordinate_unit: DepthUnit,
    config: Step5ConfigSnapshot,
) -> tuple[tuple[ObjectTrajectoryHypothesis, ...], tuple[str, ...], int, int, int]:
    trajectories = []
    unresolved_ids: list[str] = []
    requested = placed = 0
    for track in manifest.tracks:
        matching = [row for row in track.observations if row.scale_id == scale_id]
        unavailable = list(track.unavailable_observations)
        requested += len(matching) + len(unavailable)
        unresolved_ids.extend(row.unavailable_id for row in unavailable)
        by_component: dict[str, list[tuple[object, np.ndarray, np.ndarray]]] = defaultdict(list)
        unplaced_frames = []
        for row in matching:
            transform = transforms.get(row.frame_index)
            if transform is None:
                unresolved_ids.append(row.observation_id)
                unplaced_frames.append(row.frame_index)
                continue
            camera_point = _median_point(row)
            world_point = transform.rotation_world_to_camera.T @ (
                camera_point - transform.translation_world_to_camera
            )
            rotated_mad = np.abs(transform.rotation_world_to_camera.T) @ _point_mad(row)
            sigma = np.hypot(rotated_mad, transform.position_std)
            by_component[transform.component_id].append((row, world_point, sigma))
            placed += 1
        for component_id, rows in sorted(by_component.items()):
            rows.sort(key=lambda item: item[0].frame_index)
            timestamps = np.asarray([row.timestamp_s for row, _, _ in rows], dtype=np.float64)
            positions = np.asarray([point for _, point, _ in rows], dtype=np.float64)
            std = np.asarray([sigma for _, _, sigma in rows], dtype=np.float64)
            derivatives = _motion_derivatives(
                positions, timestamps, std, config.uncertainty_sigma_multiplier
            )
            observations = []
            for (row, position, sigma), derivative in zip(rows, derivatives):
                velocity, speed, interval = derivative
                observations.append(
                    ObjectWorldObservation(
                        state_id=f"object-state:{track.track_id}:{component_id}:{row.frame_index}",
                        geometry_observation_id=row.observation_id,
                        track_id=track.track_id,
                        component_id=component_id,
                        frame_index=row.frame_index,
                        timestamp_s=row.timestamp_s,
                        coordinate_unit=coordinate_unit,
                        position=_vector(position),
                        position_std=_nonnegative_vector(sigma),
                        velocity=_vector(velocity) if velocity is not None else None,
                        speed=speed,
                        speed_interval=interval,
                    )
                )
            motion_state, motion_score = _motion_classification(
                positions=positions, position_std=std, config=config
            )
            limitations = ["motion_classification_is_initial_not_step6_verified"]
            limitations.append("velocity_may_span_missing_intermediate_3d_observations")
            if coordinate_unit == DepthUnit.RELATIVE_UNIT:
                limitations.append("speed_is_relative_units_per_second_not_meters_per_second")
                limitations.append("per_frame_depth_scale_drift_can_mimic_object_motion")
            trajectories.append(
                ObjectTrajectoryHypothesis(
                    trajectory_id=f"object-trajectory:{track.track_id}:{component_id}",
                    track_id=track.track_id,
                    class_name=track.primary_class,
                    component_id=component_id,
                    coordinate_unit=coordinate_unit,
                    observations=tuple(observations),
                    unplaced_frame_indices=tuple(sorted(set(unplaced_frames))),
                    motion_state=motion_state,
                    motion_score=motion_score,
                    semantic_static_prior=_semantic_static_prior(track.primary_class),
                    evidence=tuple(row.observation_id for row, _, _ in rows),
                    limitations=tuple(limitations),
                )
            )
    return (
        tuple(trajectories),
        tuple(sorted(set(unresolved_ids))),
        requested,
        placed,
        requested - placed,
    )


def _hypothesis(
    *,
    manifest: VideoGeometryManifest,
    scale,
    config: Step5ConfigSnapshot,
    config_sha256: str,
    source_geometry_sha256: str,
) -> tuple[WorldHypothesis, tuple[int, int, int, int]]:
    coordinate_unit = (
        DepthUnit.METER
        if scale.observability == Observability.METRIC
        else DepthUnit.RELATIVE_UNIT
    )
    edges, scale_limitations = _pose_edges(
        manifest,
        scale_id=scale.scale_id,
        coordinate_unit=coordinate_unit,
        config=config,
    )
    transforms, component_edges, graph_limitations = _accumulate_components(edges)
    components = _ego_components(
        manifest=manifest,
        transforms=transforms,
        component_edges=component_edges,
        coordinate_unit=coordinate_unit,
        config=config,
    )
    trajectories, unresolved_objects, requested, placed, unplaced = _object_trajectories(
        manifest=manifest,
        scale_id=scale.scale_id,
        transforms=transforms,
        coordinate_unit=coordinate_unit,
        config=config,
    )
    pose_frames = set(transforms)
    evidence_frames = {
        row.frame_index
        for track in manifest.tracks
        for row in track.observations
        if row.scale_id == scale.scale_id
    }
    unresolved_ego = tuple(sorted(evidence_frames - pose_frames))
    geometry_support = float(np.mean([edge.support_score for edge in edges])) if edges else 0.0
    coverage = placed / requested if requested else 0.0
    observability_support = {
        Observability.METRIC: 1.0,
        Observability.RELATIVE: 0.75,
        Observability.AMBIGUOUS: 0.4,
        Observability.UNOBSERVABLE: 0.2,
    }[scale.observability]
    fallback_count = sum(edge.evidence_mode != "semantic_static_tracks" for edge in edges)
    assumption_penalty = fallback_count / len(edges) if edges else 1.0
    total = float(
        np.clip(
            0.45 * geometry_support
            + 0.30 * coverage
            + 0.25 * observability_support
            - 0.25 * assumption_penalty,
            0.0,
            1.0,
        )
    )
    limitations = {
        *scale.limitations,
        *scale_limitations,
        *graph_limitations,
        "initial_hypothesis_not_forward_verified",
        "rotation_uncertainty_not_yet_propagated",
    }
    if len(components) > 1:
        limitations.add("disconnected_pose_components_not_aligned")
    if unresolved_ego:
        limitations.add("ego_pose_unobservable_for_some_object_frames")
    world_status = (
        "unobservable" if not components else "global" if len(components) == 1 else "component_local"
    )
    payload = {
        "video_id": manifest.video_id,
        "scale_id": scale.scale_id,
        "config_sha256": config_sha256,
        "source_geometry_sha256": source_geometry_sha256,
        "edge_ids": [edge.pose_id for edge in edges],
    }
    hypothesis = WorldHypothesis(
        hypothesis_id=f"world:{hash_payload(payload)[:20]}",
        rank=1,
        scale_id=scale.scale_id,
        scale_observability=scale.observability,
        coordinate_unit=coordinate_unit,
        world_frame_status=world_status,
        metric_scale_claimed=coordinate_unit == DepthUnit.METER,
        ego_components=components,
        object_trajectories=trajectories,
        unresolved_ego_frame_indices=unresolved_ego,
        unresolved_object_observation_ids=unresolved_objects,
        discrete_choices=(f"scale_choice:{scale.scale_id}",),
        construction_score=WorldConstructionScore(
            geometry_support=geometry_support,
            trajectory_coverage=coverage,
            observability_support=observability_support,
            assumption_penalty=assumption_penalty,
            total=total,
        ),
        limitations=tuple(sorted(limitations)),
    )
    return hypothesis, (len(edges), len(components), requested, placed)


def _static_trajectory_alternative(
    trajectory: ObjectTrajectoryHypothesis,
) -> ObjectTrajectoryHypothesis:
    positions = np.asarray(
        [
            (row.position.x, row.position.y, row.position.z)
            for row in trajectory.observations
        ],
        dtype=np.float64,
    )
    center = np.median(positions, axis=0)
    scatter = np.median(np.abs(positions - center), axis=0)
    source_std = np.asarray(
        [
            (row.position_std.x, row.position_std.y, row.position_std.z)
            for row in trajectory.observations
        ],
        dtype=np.float64,
    )
    combined_std = np.hypot(np.median(source_std, axis=0), scatter)
    speed_upper = max(
        (
            row.speed_interval[1]
            for row in trajectory.observations
            if row.speed_interval is not None
        ),
        default=0.0,
    )
    observations = []
    for row in trajectory.observations:
        payload = row.model_dump(mode="python")
        payload.update(
            position=_vector(center),
            position_std=_nonnegative_vector(combined_std),
            velocity=Vector3D(x=0.0, y=0.0, z=0.0),
            speed=0.0,
            speed_interval=(0.0, speed_upper),
        )
        observations.append(ObjectWorldObservation.model_validate(payload))
    payload = trajectory.model_dump(mode="python")
    payload.update(
        observations=tuple(observations),
        motion_state=MotionState.STATIC,
        motion_score=0.0,
        limitations=tuple(
            sorted(
                {
                    *trajectory.limitations,
                    "ambiguous_motion_constrained_static_initial_branch",
                }
            )
        ),
    )
    return ObjectTrajectoryHypothesis.model_validate(payload)


def _moving_trajectory_alternative(
    trajectory: ObjectTrajectoryHypothesis,
) -> ObjectTrajectoryHypothesis:
    payload = trajectory.model_dump(mode="python")
    payload.update(
        motion_state=MotionState.MOVING,
        limitations=tuple(
            sorted(
                {
                    *trajectory.limitations,
                    "ambiguous_motion_retained_as_moving_initial_branch",
                }
            )
        ),
    )
    return ObjectTrajectoryHypothesis.model_validate(payload)


def _motion_alternative_hypotheses(
    parent: WorldHypothesis,
) -> tuple[WorldHypothesis, ...]:
    """Create auditable one-variable alternatives without combinatorial growth."""

    alternatives = []
    ambiguous = sorted(
        (
            row
            for row in parent.object_trajectories
            if row.motion_state == MotionState.AMBIGUOUS
        ),
        key=lambda row: (-len(row.observations), row.trajectory_id),
    )
    for trajectory in ambiguous:
        for state in (MotionState.STATIC, MotionState.MOVING):
            replacement = (
                _static_trajectory_alternative(trajectory)
                if state == MotionState.STATIC
                else _moving_trajectory_alternative(trajectory)
            )
            trajectories = tuple(
                replacement if row.trajectory_id == trajectory.trajectory_id else row
                for row in parent.object_trajectories
            )
            penalty_increment = (
                0.01
                if state == MotionState.STATIC and trajectory.semantic_static_prior
                else 0.08
                if state == MotionState.MOVING and trajectory.semantic_static_prior
                else 0.03
            )
            score_payload = parent.construction_score.model_dump(mode="python")
            score_payload["assumption_penalty"] = min(
                1.0,
                parent.construction_score.assumption_penalty + penalty_increment,
            )
            score_payload["total"] = max(
                0.0,
                parent.construction_score.total - penalty_increment,
            )
            choice = f"motion:{trajectory.trajectory_id}:{state.value}"
            payload = parent.model_dump(mode="python")
            payload.update(
                hypothesis_id=(
                    f"world:{hash_payload({'parent': parent.hypothesis_id, 'choice': choice})[:20]}"
                ),
                object_trajectories=trajectories,
                discrete_choices=(*parent.discrete_choices, choice),
                construction_score=WorldConstructionScore.model_validate(score_payload),
                limitations=tuple(
                    sorted(
                        {
                            *parent.limitations,
                            "single_variable_motion_alternative_not_forward_verified",
                        }
                    )
                ),
            )
            alternatives.append(WorldHypothesis.model_validate(payload))
    return tuple(alternatives)


def _rank_hypotheses(
    hypotheses: list[WorldHypothesis], top_k: int
) -> tuple[WorldHypothesis, ...]:
    selected = sorted(
        hypotheses,
        key=lambda row: (-row.construction_score.total, row.hypothesis_id),
    )[:top_k]
    ranked = []
    for rank, row in enumerate(selected, start=1):
        payload = row.model_dump(mode="python")
        payload["rank"] = rank
        ranked.append(WorldHypothesis.model_validate(payload))
    return tuple(ranked)


def _video_world_state(
    *,
    loaded: _LoadedStep4,
    manifest: VideoGeometryManifest,
    manifest_reference: ArtifactRef,
    geometry_store_reference: ArtifactRef,
    config: Step5ConfigSnapshot,
    config_sha256: str,
) -> VideoWorldStateManifest:
    candidates = []
    accounting = []
    for scale in manifest.scale_hypotheses:
        hypothesis, counts = _hypothesis(
            manifest=manifest,
            scale=scale,
            config=config,
            config_sha256=config_sha256,
            source_geometry_sha256=manifest_reference.sha256,
        )
        candidates.append(hypothesis)
        accounting.append((hypothesis.hypothesis_id, counts))
        alternatives = _motion_alternative_hypotheses(hypothesis)
        candidates.extend(alternatives)
        accounting.extend((row.hypothesis_id, counts) for row in alternatives)
    ranked = _rank_hypotheses(candidates, config.top_k)
    top_counts = dict(accounting)[ranked[0].hypothesis_id]
    edge_count, component_count, requested, placed = top_counts
    top = ranked[0]
    input_snapshot = Step5InputSnapshot(
        source_step4_relative_root=loaded.stage_root.relative_to(loaded.run_root).as_posix(),
        geometry_store=_step4_link(geometry_store_reference),
        video_geometry_manifest=_step4_link(manifest_reference),
        source_step3_relative_root=manifest.input_snapshot.source_step3_relative_root,
        tracking_store=manifest.input_snapshot.tracking_store,
        video_tracking_manifest=manifest.input_snapshot.video_tracking_manifest,
    )
    beam_id = f"beam0:{hash_payload({'video_id': manifest.video_id, 'hypotheses': [row.hypothesis_id for row in ranked]})[:20]}"
    return VideoWorldStateManifest(
        run_id=manifest.run_id,
        video_id=manifest.video_id,
        source_geometry_sha256=manifest_reference.sha256,
        config_sha256=config_sha256,
        canonical_fps=manifest.canonical_fps,
        image_size=manifest.image_size,
        frame_count=manifest.frame_count,
        input_snapshot=input_snapshot,
        initial_beam=HypothesisBeam(
            beam_id=beam_id,
            top_k=config.top_k,
            hypotheses=ranked,
        ),
        validation=Step5ValidationSummary(
            input_pose_edges=len(manifest.camera_motion.poses),
            emitted_pose_components=component_count,
            emitted_ego_poses=sum(len(component.poses) for component in top.ego_components),
            requested_object_observations=requested,
            placed_object_observations=placed,
            unplaced_object_observations=requested - placed,
            emitted_object_trajectories=len(top.object_trajectories),
            hypothesis_count=len(ranked),
            overall_pass=True,
        ),
        tool_versions=(ToolVersion(name="numpy", version=np.__version__),),
    )


def run_step5(
    *,
    geometry_store_path: Path | str,
    top_k: int = 5,
    minimum_motion_observations: int = 2,
    static_displacement_threshold: float = 0.25,
    moving_displacement_threshold: float = 0.75,
    static_scale_residual_threshold: float = 0.50,
    fallback_scale_residual_threshold: float = 0.15,
    uncertainty_sigma_multiplier: float = 2.0,
) -> Step5Result:
    """Create the immutable initial world-hypothesis beam ``B0``."""

    loaded = _load_step4(geometry_store_path)
    config = Step5ConfigSnapshot(
        top_k=top_k,
        minimum_motion_observations=minimum_motion_observations,
        static_displacement_threshold=static_displacement_threshold,
        moving_displacement_threshold=moving_displacement_threshold,
        static_scale_residual_threshold=static_scale_residual_threshold,
        fallback_scale_residual_threshold=fallback_scale_residual_threshold,
        uncertainty_sigma_multiplier=uncertainty_sigma_multiplier,
    )
    config_sha256 = hash_payload(config)
    source_geometry_sha256 = sha256_file(loaded.store_path)
    stage_root = (
        loaded.run_root
        / "05_world_reconstruction"
        / f"input_{source_geometry_sha256[:16]}"
        / f"config_{config_sha256[:16]}"
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    geometry_store_reference = _file_reference(
        path=loaded.store_path,
        stage_root=loaded.stage_root,
        artifact_id=f"geometry-store:{loaded.store.run_id}",
    )
    manifests, references = [], []
    for manifest, manifest_reference in zip(loaded.manifests, loaded.manifest_refs):
        world_state = _video_world_state(
            loaded=loaded,
            manifest=manifest,
            manifest_reference=manifest_reference,
            geometry_store_reference=geometry_store_reference,
            config=config,
            config_sha256=config_sha256,
        )
        relative_path = Path("videos") / f"{manifest.video_id}.world_state.json"
        path = stage_root / relative_path
        sha256, byte_size = write_contract(path, world_state)
        references.append(
            ArtifactRef(
                artifact_id=f"video-world-state:{manifest.video_id}",
                relative_path=relative_path.as_posix(),
                sha256=sha256,
                byte_size=byte_size,
                media_type="application/json",
                coordinate_space=None,
            )
        )
        manifests.append(world_state)
    store = WorldStateStore(
        run_id=loaded.store.run_id,
        source_geometry_store_sha256=source_geometry_sha256,
        config=config,
        config_sha256=config_sha256,
        video_ids=loaded.store.video_ids,
        video_world_states=tuple(references),
    )
    store_path = stage_root / "world_state_store.json"
    write_contract(store_path, store)
    return Step5Result(
        store=store,
        video_manifests=tuple(manifests),
        stage_root=stage_root,
        store_path=store_path,
    )


__all__ = ["Step5Result", "run_step5"]
