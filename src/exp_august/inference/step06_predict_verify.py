"""Step 6: predict evidence and verify immutable Step 5 hypotheses.

This stage is intentionally read-only with respect to the world-state beam.  It
projects each hypothesis into held-out evidence, evaluates physical/semantic
constraints, and emits residual packets.  Repair and hypothesis selection are
owned by later stages.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.exp_august.contracts import (
    ArtifactLink,
    ArtifactOwner,
    ArtifactRef,
    ConflictWindow,
    CueFamily,
    DepthUnit,
    EvaluationBasis,
    EvidenceRole,
    HypothesisResidualPacket,
    MotionState,
    ResidualFamily,
    ResidualFamilySummary,
    ResidualRecord,
    ResidualSeverity,
    ResidualStore,
    Step6ConfigSnapshot,
    Step6InputSnapshot,
    Step6ValidationSummary,
    ToolVersion,
    VideoEvidenceManifest,
    VideoGeometryManifest,
    VideoResidualManifest,
    VideoTrackingManifest,
    VideoWorldStateManifest,
    WorldStateStore,
)
from src.exp_august.contracts.codec import (
    hash_payload,
    read_contract,
    sha256_file,
    write_contract,
)
from src.exp_august.inference.artifact_io import read_image_artifact


@dataclass(frozen=True)
class Step6Result:
    store: ResidualStore
    video_manifests: tuple[VideoResidualManifest, ...]
    stage_root: Path
    store_path: Path


@dataclass(frozen=True)
class _LoadedVideo:
    world: VideoWorldStateManifest
    world_reference: ArtifactRef
    geometry: VideoGeometryManifest
    geometry_path: Path
    tracking: VideoTrackingManifest
    tracking_path: Path
    evidence: VideoEvidenceManifest
    evidence_path: Path


@dataclass(frozen=True)
class _LoadedStep5:
    store_path: Path
    stage_root: Path
    run_root: Path
    store: WorldStateStore
    videos: tuple[_LoadedVideo, ...]


class _ArtifactResolver:
    def __init__(self, *, run_root: Path, step2_root: Path, step3_root: Path) -> None:
        self._roots = {
            ArtifactOwner.STEP2_NEURAL_EVIDENCE: step2_root,
            ArtifactOwner.STEP3_OBJECT_TRACKING: step3_root,
        }
        self._verified: set[tuple[str, str]] = set()

    def path(self, link: ArtifactLink) -> Path:
        if link.owner not in self._roots:
            raise RuntimeError(f"Step 6 cannot resolve {link.owner.value} artifact")
        path = self._roots[link.owner] / link.artifact.relative_path
        key = (link.owner.value, link.artifact.sha256)
        if key not in self._verified:
            if not path.is_file():
                raise RuntimeError(f"Step 6 source artifact is missing: {path}")
            if path.stat().st_size != link.artifact.byte_size:
                raise RuntimeError(f"Step 6 source artifact size mismatch: {path}")
            if sha256_file(path) != link.artifact.sha256:
                raise RuntimeError(f"Step 6 source artifact hash mismatch: {path}")
            self._verified.add(key)
        return path


def _verified_contract(path: Path, reference: ArtifactRef, model):
    if not path.is_file() or path.stat().st_size != reference.byte_size:
        raise RuntimeError(f"Step 6 input is missing or truncated: {path}")
    if sha256_file(path) != reference.sha256:
        raise RuntimeError(f"Step 6 input failed integrity check: {path}")
    return read_contract(path, model)


def _find_run_root(store_path: Path) -> Path:
    for parent in store_path.parents:
        if parent.name == "05_world_reconstruction":
            return parent.parent
    raise RuntimeError("Step 5 store must live below 05_world_reconstruction")


def _load_step5(store_path: Path | str) -> _LoadedStep5:
    resolved = Path(store_path).expanduser().resolve()
    store = read_contract(resolved, WorldStateStore)
    stage_root = resolved.parent
    run_root = _find_run_root(resolved)
    videos = []
    for video_id, world_ref in zip(store.video_ids, store.video_world_states):
        world_path = stage_root / world_ref.relative_path
        world = _verified_contract(world_path, world_ref, VideoWorldStateManifest)
        if world.video_id != video_id or world.run_id != store.run_id:
            raise RuntimeError(f"Step 5 world-state identity mismatch: {world_path}")
        snapshot = world.input_snapshot
        geometry_path = (
            run_root
            / snapshot.source_step4_relative_root
            / snapshot.video_geometry_manifest.artifact.relative_path
        )
        geometry = _verified_contract(
            geometry_path,
            snapshot.video_geometry_manifest.artifact,
            VideoGeometryManifest,
        )
        tracking_path = (
            run_root
            / snapshot.source_step3_relative_root
            / snapshot.video_tracking_manifest.artifact.relative_path
        )
        tracking = _verified_contract(
            tracking_path,
            snapshot.video_tracking_manifest.artifact,
            VideoTrackingManifest,
        )
        step2_root = run_root / tracking.input_snapshot.source_step2_relative_root
        evidence_path = (
            step2_root / tracking.input_snapshot.video_evidence_manifest.artifact.relative_path
        )
        evidence = _verified_contract(
            evidence_path,
            tracking.input_snapshot.video_evidence_manifest.artifact,
            VideoEvidenceManifest,
        )
        if {geometry.video_id, tracking.video_id, evidence.video_id} != {video_id}:
            raise RuntimeError(f"Step 6 cross-stage video identity mismatch: {video_id}")
        videos.append(
            _LoadedVideo(
                world=world,
                world_reference=world_ref,
                geometry=geometry,
                geometry_path=geometry_path,
                tracking=tracking,
                tracking_path=tracking_path,
                evidence=evidence,
                evidence_path=evidence_path,
            )
        )
    return _LoadedStep5(
        store_path=resolved,
        stage_root=stage_root,
        run_root=run_root,
        store=store,
        videos=tuple(videos),
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


def _link(owner: ArtifactOwner, reference: ArtifactRef) -> ArtifactLink:
    return ArtifactLink(owner=owner, artifact=reference)


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except Exception as error:
        raise RuntimeError(f"Step 6 could not decode NPZ artifact: {path}") from error


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = read_image_artifact(path, cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != shape:
        raise RuntimeError(f"Step 6 mask shape/decode failure: {path}")
    return image > 0


def _vector3(value) -> np.ndarray:
    return np.asarray((value.x, value.y, value.z), dtype=np.float64)


def _pose_map(hypothesis) -> dict[tuple[str, int], object]:
    return {
        (component.component_id, pose.frame_index): pose
        for component in hypothesis.ego_components
        for pose in component.poses
    }


def _project(position_world: np.ndarray, pose, intrinsics) -> tuple[np.ndarray, float] | None:
    rotation = np.asarray(pose.rotation_world_to_camera, dtype=np.float64).reshape(3, 3)
    camera = rotation @ position_world + _vector3(pose.translation_world_to_camera)
    if not np.all(np.isfinite(camera)) or camera[2] <= 1e-8:
        return None
    pixel = np.asarray(
        (
            intrinsics.fx_px * camera[0] / camera[2] + intrinsics.cx_px,
            intrinsics.fy_px * camera[1] / camera[2] + intrinsics.cy_px,
        ),
        dtype=np.float64,
    )
    return pixel, float(camera[2])


def _interpolate_observation(trajectory, frame_index: int, maximum_gap: int):
    exact = {row.frame_index: row for row in trajectory.observations}
    if frame_index in exact:
        return _vector3(exact[frame_index].position), exact[frame_index].timestamp_s, "exact"
    before = [row for row in trajectory.observations if row.frame_index < frame_index]
    after = [row for row in trajectory.observations if row.frame_index > frame_index]
    if not before or not after:
        return None
    left, right = before[-1], after[0]
    if frame_index - left.frame_index > maximum_gap or right.frame_index - frame_index > maximum_gap:
        return None
    fraction = (frame_index - left.frame_index) / (right.frame_index - left.frame_index)
    point = (1.0 - fraction) * _vector3(left.position) + fraction * _vector3(right.position)
    timestamp = (1.0 - fraction) * left.timestamp_s + fraction * right.timestamp_s
    return point, float(timestamp), "interpolated"


def _check_assignments(tracking, cue_family: CueFamily):
    rows = [
        row
        for row in tracking.evidence_use_plan.assignments
        if row.role == EvidenceRole.CHECK_ONLY
        and row.cue_family == cue_family
        and row.artifact is not None
        and row.frame_index is not None
    ]
    required_prohibitions = {"step4_geometry", "step5_world_state"}
    for row in rows:
        if "step6_verification" not in row.allowed_consumers:
            raise RuntimeError(
                f"check-only evidence is not authorized for Step 6: {row.evidence_key}"
            )
        if not required_prohibitions.issubset(row.prohibited_optimizers):
            raise RuntimeError(
                f"check-only evidence was not isolated from Step 4/5: {row.evidence_key}"
            )
    return rows


def _severity(normalized: float | None, *, hard: bool, config: Step6ConfigSnapshot):
    if normalized is None:
        return ResidualSeverity.INFO
    if hard and normalized >= config.hard_z_threshold:
        return ResidualSeverity.HARD_VIOLATION
    if normalized >= config.conflict_z_threshold:
        return ResidualSeverity.VIOLATION
    if normalized >= 1.0:
        return ResidualSeverity.WARNING
    return ResidualSeverity.INFO


def _residual(
    *, hypothesis_id: str, family: ResidualFamily, constraint_id: str,
    basis: EvaluationBasis, frame: int, timestamp: float, metric_name: str,
    metric_unit: str, reason: str, config: Step6ConfigSnapshot,
    raw: float | None = None, normalized: float | None = None,
    uncertainty: float | None = None, threshold: float | None = None,
    predicted: Iterable[float] = (), observed: Iterable[float] = (),
    role: EvidenceRole | None = None, cue: CueFamily | None = None,
    component_id: str | None = None, track_id: str | None = None,
    evidence_keys: Iterable[str] = (), evidence_artifacts: Iterable[ArtifactLink] = (),
    hard: bool = False, limitations: Iterable[str] = (), end_frame: int | None = None,
    end_timestamp: float | None = None,
) -> ResidualRecord:
    evaluable = raw is not None and normalized is not None
    return ResidualRecord(
        residual_id=f"residual:{hypothesis_id}:{constraint_id}:{track_id or component_id or 'world'}:{frame}",
        hypothesis_id=hypothesis_id,
        family=family,
        constraint_id=constraint_id,
        evaluation_basis=basis if evaluable else EvaluationBasis.NOT_EVALUABLE,
        evidence_role=role,
        cue_family=cue,
        component_id=component_id,
        track_id=track_id,
        start_frame_index=frame,
        end_frame_index=frame if end_frame is None else end_frame,
        start_timestamp_s=max(0.0, timestamp),
        end_timestamp_s=max(0.0, timestamp if end_timestamp is None else end_timestamp),
        metric_name=metric_name,
        metric_unit=metric_unit,
        predicted_values=tuple(float(value) for value in predicted) if evaluable else (),
        observed_values=tuple(float(value) for value in observed) if evaluable else (),
        raw_residual=float(raw) if evaluable else None,
        normalized_residual=float(normalized) if evaluable else None,
        uncertainty=float(uncertainty) if evaluable else None,
        threshold=float(threshold) if evaluable else None,
        severity=_severity(normalized if evaluable else None, hard=hard, config=config),
        evaluable=evaluable,
        hard_constraint=hard,
        evidence_keys=tuple(evidence_keys),
        evidence_artifacts=tuple(evidence_artifacts),
        reason=reason,
        limitations=tuple(limitations),
    )


def _fit_reprojection_residuals(*, hypothesis, geometry, config):
    rows = []
    poses = _pose_map(hypothesis)
    geometry_by_id = {
        row.observation_id: row for track in geometry.tracks for row in track.observations
    }
    for trajectory in hypothesis.object_trajectories:
        for state in trajectory.observations:
            observation = geometry_by_id.get(state.geometry_observation_id)
            pose = poses.get((trajectory.component_id, state.frame_index))
            projection = _project(_vector3(state.position), pose, geometry.intrinsics) if pose else None
            if observation is None or projection is None:
                rows.append(_residual(
                    hypothesis_id=hypothesis.hypothesis_id, family=ResidualFamily.OBSERVATION,
                    constraint_id="object_reprojection", basis=EvaluationBasis.SELF_CONSISTENCY,
                    frame=state.frame_index, timestamp=state.timestamp_s,
                    metric_name="pixel_reprojection_error", metric_unit="pixel",
                    reason="world state cannot be reprojected at this frame", config=config,
                    component_id=trajectory.component_id, track_id=trajectory.track_id,
                    limitations=("self_consistency_not_independent_evidence",),
                ))
                continue
            pixel, _ = projection
            observed = np.asarray((observation.pixel_centroid.u, observation.pixel_centroid.v))
            error = float(np.linalg.norm(pixel - observed))
            rows.append(_residual(
                hypothesis_id=hypothesis.hypothesis_id, family=ResidualFamily.OBSERVATION,
                constraint_id="object_reprojection", basis=EvaluationBasis.FIT_EVIDENCE,
                role=EvidenceRole.FIT, cue=CueFamily.DEPTH,
                frame=state.frame_index, timestamp=state.timestamp_s,
                metric_name="pixel_reprojection_error", metric_unit="pixel",
                raw=error, normalized=error / config.projection_sigma_px,
                uncertainty=config.projection_sigma_px,
                threshold=config.conflict_z_threshold * config.projection_sigma_px,
                predicted=pixel, observed=observed,
                reason="forward projection into the Step 4 fitted observation", config=config,
                component_id=trajectory.component_id, track_id=trajectory.track_id,
                limitations=("fit_evidence_self_consistency_not_holdout_validation",),
            ))
    return rows


def _selected_support(*, tracking, resolver, track_id: str, frame: int, shape):
    track = next((row for row in tracking.tracks if row.track_id == track_id), None)
    observation = next((row for row in track.observations if row.frame_index == frame), None) if track else None
    if observation is None:
        return None, None
    candidates = {row.candidate_id: row for row in tracking.mask_candidate_bank}
    candidate = candidates.get(observation.selected_mask_candidate_id)
    if candidate is not None and candidate.mask is not None:
        return _read_mask(resolver.path(candidate.mask), shape), observation
    mask = np.zeros(shape, dtype=bool)
    x1, y1 = int(observation.bbox.x1), int(observation.bbox.y1)
    x2, y2 = int(np.ceil(observation.bbox.x2)), int(np.ceil(observation.bbox.y2))
    mask[max(0, y1):min(shape[0], y2), max(0, x1):min(shape[1], x2)] = True
    return mask, observation


def _check_depth_residuals(*, hypothesis, video, resolver, config):
    rows = []
    poses = _pose_map(hypothesis)
    assignments = _check_assignments(video.tracking, CueFamily.DEPTH)
    shape = (video.world.image_size.height, video.world.image_size.width)
    for assignment in assignments:
        payload = _read_npz(resolver.path(assignment.artifact))
        depth = np.asarray(payload.get("depth"))
        valid = np.asarray(payload.get("valid"), dtype=bool)
        for trajectory in hypothesis.object_trajectories:
            estimate = _interpolate_observation(
                trajectory, assignment.frame_index, config.maximum_prediction_gap_frames
            )
            pose = poses.get((trajectory.component_id, assignment.frame_index))
            support, _ = _selected_support(
                tracking=video.tracking, resolver=resolver, track_id=trajectory.track_id,
                frame=assignment.frame_index, shape=shape,
            )
            if estimate is None or pose is None or support is None or depth.shape != shape or valid.shape != shape:
                rows.append(_residual(
                    hypothesis_id=hypothesis.hypothesis_id, family=ResidualFamily.OBSERVATION,
                    constraint_id="heldout_object_depth", basis=EvaluationBasis.CHECK_EVIDENCE,
                    frame=assignment.frame_index, timestamp=video.evidence.frames[assignment.frame_index].timestamp_s,
                    metric_name="log_depth_error", metric_unit="log_ratio", reason="held-out depth cannot be evaluated for this trajectory",
                    config=config, component_id=trajectory.component_id, track_id=trajectory.track_id,
                    role=EvidenceRole.CHECK_ONLY, cue=CueFamily.DEPTH,
                    evidence_keys=(assignment.evidence_key,), evidence_artifacts=(assignment.artifact,),
                    limitations=("missing_pose_track_support_or_prediction",),
                ))
                continue
            projection = _project(estimate[0], pose, video.geometry.intrinsics)
            selected = support & valid & np.isfinite(depth) & (depth > 0)
            if projection is None or not np.any(selected):
                continue
            predicted_depth = projection[1]
            observed_depth = float(np.median(depth[selected]))
            raw = float(abs(np.log(max(predicted_depth, 1e-8) / max(observed_depth, 1e-8))))
            rows.append(_residual(
                hypothesis_id=hypothesis.hypothesis_id, family=ResidualFamily.OBSERVATION,
                constraint_id="heldout_object_depth", basis=EvaluationBasis.CHECK_EVIDENCE,
                role=EvidenceRole.CHECK_ONLY, cue=CueFamily.DEPTH,
                frame=assignment.frame_index, timestamp=estimate[1], metric_name="log_depth_error",
                metric_unit="log_ratio", raw=raw, normalized=raw / config.depth_log_sigma,
                uncertainty=config.depth_log_sigma,
                threshold=config.conflict_z_threshold * config.depth_log_sigma,
                predicted=(predicted_depth,), observed=(observed_depth,),
                reason="predicted object depth checked against seeded held-out depth", config=config,
                component_id=trajectory.component_id, track_id=trajectory.track_id,
                evidence_keys=(assignment.evidence_key,), evidence_artifacts=(assignment.artifact,),
                limitations=("relative_monocular_depth_may_have_per_frame_scale_drift", estimate[2]),
            ))
    return rows


def _check_object_flow_residuals(*, hypothesis, video, resolver, config):
    rows = []
    poses = _pose_map(hypothesis)
    shape = (video.world.image_size.height, video.world.image_size.width)
    assignments = _check_assignments(video.tracking, CueFamily.FLOW_BACKWARD)
    for assignment in assignments:
        target_frame, source_frame = assignment.frame_index, assignment.frame_index - 1
        payload = _read_npz(resolver.path(assignment.artifact))
        flow = np.asarray(payload.get("flow"))
        valid = np.asarray(payload.get("domain_valid"), dtype=bool)
        if "consistency_valid" in payload:
            valid &= np.asarray(payload["consistency_valid"], dtype=bool)
        for trajectory in hypothesis.object_trajectories:
            target = _interpolate_observation(trajectory, target_frame, config.maximum_prediction_gap_frames)
            source = _interpolate_observation(trajectory, source_frame, config.maximum_prediction_gap_frames)
            target_pose = poses.get((trajectory.component_id, target_frame))
            source_pose = poses.get((trajectory.component_id, source_frame))
            support, _ = _selected_support(
                tracking=video.tracking, resolver=resolver, track_id=trajectory.track_id,
                frame=target_frame, shape=shape,
            )
            if target is None or source is None or target_pose is None or source_pose is None or support is None:
                continue
            projected_target = _project(target[0], target_pose, video.geometry.intrinsics)
            projected_source = _project(source[0], source_pose, video.geometry.intrinsics)
            selected = support & valid & np.all(np.isfinite(flow), axis=2) if flow.shape == (*shape, 2) and valid.shape == shape else np.zeros(shape, bool)
            if projected_target is None or projected_source is None or not np.any(selected):
                continue
            predicted = projected_source[0] - projected_target[0]
            observed = np.median(flow[selected], axis=0)
            raw = float(np.linalg.norm(predicted - observed))
            rows.append(_residual(
                hypothesis_id=hypothesis.hypothesis_id, family=ResidualFamily.OBJECT_IDENTITY,
                constraint_id="heldout_object_backward_flow", basis=EvaluationBasis.CHECK_EVIDENCE,
                role=EvidenceRole.CHECK_ONLY, cue=CueFamily.FLOW_BACKWARD,
                frame=target_frame, timestamp=target[1], metric_name="flow_endpoint_error",
                metric_unit="pixel", raw=raw, normalized=raw / config.flow_sigma_px,
                uncertainty=config.flow_sigma_px, threshold=config.conflict_z_threshold * config.flow_sigma_px,
                predicted=predicted, observed=observed,
                reason="world trajectory displacement checked against held-out backward flow", config=config,
                component_id=trajectory.component_id, track_id=trajectory.track_id,
                evidence_keys=(assignment.evidence_key,), evidence_artifacts=(assignment.artifact,),
                limitations=(target[2], source[2], "mask_support_was_not_held_out"),
            ))
    return rows


def _foreground_mask(*, tracking, resolver, frame: int, shape: tuple[int, int]):
    foreground = np.zeros(shape, dtype=bool)
    for track in tracking.tracks:
        support, observation = _selected_support(
            tracking=tracking,
            resolver=resolver,
            track_id=track.track_id,
            frame=frame,
            shape=shape,
        )
        if observation is not None and support is not None:
            foreground |= support
    return foreground


def _check_background_flow_residuals(*, hypothesis, video, resolver, config):
    """Render rigid background motion and compare it with held-out backward flow."""
    rows = []
    poses = _pose_map(hypothesis)
    shape = (video.world.image_size.height, video.world.image_size.width)
    assignments = _check_assignments(video.tracking, CueFamily.FLOW_BACKWARD)
    for assignment in assignments:
        current_frame, previous_frame = assignment.frame_index, assignment.frame_index - 1
        frame_evidence = video.evidence.frames[current_frame]
        if frame_evidence.depth is None:
            rows.append(_residual(
                hypothesis_id=hypothesis.hypothesis_id,
                family=ResidualFamily.EGO_BACKGROUND,
                constraint_id="heldout_background_backward_flow",
                basis=EvaluationBasis.CHECK_EVIDENCE,
                frame=current_frame,
                timestamp=frame_evidence.timestamp_s,
                metric_name="background_flow_endpoint_error",
                metric_unit="pixel",
                reason="background prediction requires a depth field at the flow source frame",
                config=config,
                role=EvidenceRole.CHECK_ONLY,
                cue=CueFamily.FLOW_BACKWARD,
                evidence_keys=(assignment.evidence_key,),
                evidence_artifacts=(assignment.artifact,),
                limitations=("missing_depth_is_not_a_physical_violation",),
            ))
            continue
        depth_link = ArtifactLink(
            owner=ArtifactOwner.STEP2_NEURAL_EVIDENCE,
            artifact=frame_evidence.depth.field_ref,
        )
        flow_payload = _read_npz(resolver.path(assignment.artifact))
        depth_payload = _read_npz(resolver.path(depth_link))
        flow = np.asarray(flow_payload.get("flow"))
        flow_valid = np.asarray(flow_payload.get("domain_valid"), dtype=bool)
        if "consistency_valid" in flow_payload:
            flow_valid &= np.asarray(flow_payload["consistency_valid"], dtype=bool)
        depth = np.asarray(depth_payload.get("depth"))
        depth_valid = np.asarray(depth_payload.get("valid"), dtype=bool)
        if (
            flow.shape != (*shape, 2)
            or flow_valid.shape != shape
            or depth.shape != shape
            or depth_valid.shape != shape
        ):
            continue
        valid = (
            flow_valid
            & depth_valid
            & np.all(np.isfinite(flow), axis=2)
            & np.isfinite(depth)
            & (depth > 0.0)
            & ~_foreground_mask(
                tracking=video.tracking,
                resolver=resolver,
                frame=current_frame,
                shape=shape,
            )
        )
        grid = np.zeros(shape, dtype=bool)
        grid[:: config.background_sample_stride, :: config.background_sample_stride] = True
        yy, xx = np.nonzero(valid & grid)
        if yy.size > config.maximum_background_samples:
            chosen = np.linspace(
                0, yy.size - 1, config.maximum_background_samples, dtype=np.int64
            )
            yy, xx = yy[chosen], xx[chosen]
        matched_component = False
        for component in hypothesis.ego_components:
            current_pose = poses.get((component.component_id, current_frame))
            previous_pose = poses.get((component.component_id, previous_frame))
            if current_pose is None or previous_pose is None:
                continue
            matched_component = True
            if yy.size < 8:
                continue
            z = depth[yy, xx].astype(np.float64)
            intrinsics = video.geometry.intrinsics
            camera_current = np.column_stack(
                (
                    (xx - intrinsics.cx_px) * z / intrinsics.fx_px,
                    (yy - intrinsics.cy_px) * z / intrinsics.fy_px,
                    z,
                )
            )
            rotation_current = np.asarray(
                current_pose.rotation_world_to_camera, dtype=np.float64
            ).reshape(3, 3)
            translation_current = _vector3(current_pose.translation_world_to_camera)
            world = (rotation_current.T @ (camera_current - translation_current).T).T
            rotation_previous = np.asarray(
                previous_pose.rotation_world_to_camera, dtype=np.float64
            ).reshape(3, 3)
            translation_previous = _vector3(previous_pose.translation_world_to_camera)
            camera_previous = (rotation_previous @ world.T).T + translation_previous
            in_front = camera_previous[:, 2] > 1e-8
            if np.count_nonzero(in_front) < 8:
                continue
            predicted = np.column_stack(
                (
                    intrinsics.fx_px
                    * camera_previous[in_front, 0]
                    / camera_previous[in_front, 2]
                    + intrinsics.cx_px
                    - xx[in_front],
                    intrinsics.fy_px
                    * camera_previous[in_front, 1]
                    / camera_previous[in_front, 2]
                    + intrinsics.cy_px
                    - yy[in_front],
                )
            )
            observed = flow[yy[in_front], xx[in_front]].astype(np.float64)
            endpoint_errors = np.linalg.norm(predicted - observed, axis=1)
            raw = float(np.median(endpoint_errors))
            rows.append(_residual(
                hypothesis_id=hypothesis.hypothesis_id,
                family=ResidualFamily.EGO_BACKGROUND,
                constraint_id="heldout_background_backward_flow",
                basis=EvaluationBasis.CHECK_EVIDENCE,
                role=EvidenceRole.CHECK_ONLY,
                cue=CueFamily.FLOW_BACKWARD,
                frame=current_frame,
                timestamp=frame_evidence.timestamp_s,
                metric_name="median_background_flow_endpoint_error",
                metric_unit="pixel",
                raw=raw,
                normalized=raw / config.flow_sigma_px,
                uncertainty=config.flow_sigma_px,
                threshold=config.conflict_z_threshold * config.flow_sigma_px,
                predicted=np.median(predicted, axis=0),
                observed=np.median(observed, axis=0),
                reason="ego motion predicts rigid background flow checked against held-out RAFT flow",
                config=config,
                component_id=component.component_id,
                evidence_keys=(assignment.evidence_key,),
                evidence_artifacts=(assignment.artifact,),
                limitations=(
                    "depth_is_support_not_independent_holdout_evidence",
                    "foreground_exclusion_uses_step3_masks",
                    "relative_depth_and_translation_share_nonmetric_scale",
                ),
            ))
        if not matched_component:
            rows.append(_residual(
                hypothesis_id=hypothesis.hypothesis_id,
                family=ResidualFamily.EGO_BACKGROUND,
                constraint_id="heldout_background_backward_flow",
                basis=EvaluationBasis.CHECK_EVIDENCE,
                frame=current_frame,
                timestamp=frame_evidence.timestamp_s,
                metric_name="background_flow_endpoint_error",
                metric_unit="pixel",
                reason="adjacent ego poses do not share a Step 5 component",
                config=config,
                role=EvidenceRole.CHECK_ONLY,
                cue=CueFamily.FLOW_BACKWARD,
                evidence_keys=(assignment.evidence_key,),
                evidence_artifacts=(assignment.artifact,),
                limitations=("disconnected_pose_component_is_not_a_violation",),
            ))
    return rows


def _physics_residuals(*, hypothesis, config):
    rows = []
    metric = hypothesis.coordinate_unit == DepthUnit.METER
    for component in hypothesis.ego_components:
        poses = component.poses
        for left, right in zip(poses, poses[1:]):
            dt = right.timestamp_s - left.timestamp_s
            if dt <= 0 or left.velocity is None or right.velocity is None:
                continue
            acceleration = float(np.linalg.norm(_vector3(right.velocity) - _vector3(left.velocity)) / dt)
            scale = config.metric_max_acceleration_mps2 if metric else config.relative_acceleration_scale
            rows.append(_residual(
                hypothesis_id=hypothesis.hypothesis_id, family=ResidualFamily.PHYSICS,
                constraint_id="ego_acceleration_bound", basis=EvaluationBasis.FROZEN_KNOWLEDGE,
                frame=right.frame_index, timestamp=right.timestamp_s, metric_name="acceleration",
                metric_unit="meter_per_second_squared" if metric else "relative_unit_per_second_squared",
                raw=acceleration, normalized=acceleration / scale, uncertainty=scale,
                threshold=config.conflict_z_threshold * scale, predicted=(acceleration,), observed=(0.0,),
                reason="finite-difference acceleration checked against a frozen plausibility scale",
                config=config, component_id=component.component_id, hard=metric,
                limitations=() if metric else ("relative_scale_prevents_metric_physics_claim",),
            ))
        if metric:
            for pose in poses:
                if pose.speed is None:
                    continue
                rows.append(_residual(
                    hypothesis_id=hypothesis.hypothesis_id, family=ResidualFamily.PHYSICS,
                    constraint_id="ego_speed_bound", basis=EvaluationBasis.FROZEN_KNOWLEDGE,
                    frame=pose.frame_index, timestamp=pose.timestamp_s, metric_name="speed",
                    metric_unit="meter_per_second", raw=float(pose.speed),
                    normalized=float(pose.speed / config.metric_max_ego_speed_mps),
                    uncertainty=config.metric_max_ego_speed_mps,
                    threshold=config.conflict_z_threshold * config.metric_max_ego_speed_mps,
                    predicted=(pose.speed,), observed=(0.0,), reason="metric ego speed plausibility bound",
                    config=config, component_id=component.component_id, hard=True,
                ))
    for trajectory in hypothesis.object_trajectories:
        observations = trajectory.observations
        for left, right in zip(observations, observations[1:]):
            dt = right.timestamp_s - left.timestamp_s
            if dt <= 0 or left.velocity is None or right.velocity is None:
                continue
            acceleration = float(np.linalg.norm(_vector3(right.velocity) - _vector3(left.velocity)) / dt)
            scale = config.metric_max_acceleration_mps2 if metric else config.relative_acceleration_scale
            rows.append(_residual(
                hypothesis_id=hypothesis.hypothesis_id, family=ResidualFamily.PHYSICS,
                constraint_id="object_acceleration_bound", basis=EvaluationBasis.FROZEN_KNOWLEDGE,
                frame=right.frame_index, timestamp=right.timestamp_s, metric_name="acceleration",
                metric_unit="meter_per_second_squared" if metric else "relative_unit_per_second_squared",
                raw=acceleration, normalized=acceleration / scale, uncertainty=scale,
                threshold=config.conflict_z_threshold * scale, predicted=(acceleration,), observed=(0.0,),
                reason="object acceleration checked against a frozen plausibility scale", config=config,
                component_id=trajectory.component_id, track_id=trajectory.track_id, hard=metric,
                limitations=() if metric else ("relative_scale_prevents_metric_physics_claim",),
            ))
        if metric:
            for state in observations:
                if state.speed is None:
                    continue
                rows.append(_residual(
                    hypothesis_id=hypothesis.hypothesis_id, family=ResidualFamily.PHYSICS,
                    constraint_id="object_speed_bound", basis=EvaluationBasis.FROZEN_KNOWLEDGE,
                    frame=state.frame_index, timestamp=state.timestamp_s, metric_name="speed",
                    metric_unit="meter_per_second", raw=float(state.speed),
                    normalized=float(state.speed / config.metric_max_object_speed_mps),
                    uncertainty=config.metric_max_object_speed_mps,
                    threshold=config.conflict_z_threshold * config.metric_max_object_speed_mps,
                    predicted=(state.speed,), observed=(0.0,), reason="metric object speed plausibility bound",
                    config=config, component_id=trajectory.component_id, track_id=trajectory.track_id, hard=True,
                ))
    return rows


def _identity_semantic_residuals(*, hypothesis, tracking, config):
    rows = []
    tracks = {row.track_id: row for row in tracking.tracks}
    for trajectory in hypothesis.object_trajectories:
        frames = [row.frame_index for row in trajectory.observations]
        if len(frames) >= 2:
            largest_gap = max(right - left - 1 for left, right in zip(frames, frames[1:]))
            if largest_gap > 0:
                rows.append(_residual(
                    hypothesis_id=hypothesis.hypothesis_id, family=ResidualFamily.OBJECT_IDENTITY,
                    constraint_id="trajectory_temporal_gap", basis=EvaluationBasis.SELF_CONSISTENCY,
                    frame=frames[0], end_frame=frames[-1], timestamp=trajectory.observations[0].timestamp_s,
                    end_timestamp=trajectory.observations[-1].timestamp_s, metric_name="largest_gap",
                    metric_unit="frame", raw=float(largest_gap),
                    normalized=float(largest_gap / max(1, config.maximum_prediction_gap_frames)),
                    uncertainty=float(max(1, config.maximum_prediction_gap_frames)),
                    threshold=float(config.conflict_z_threshold * max(1, config.maximum_prediction_gap_frames)),
                    predicted=(largest_gap,), observed=(0.0,), reason="object trajectory contains a temporal support gap",
                    config=config, component_id=trajectory.component_id, track_id=trajectory.track_id,
                    limitations=("gap_is_not_proof_of_identity_failure",),
                ))
        track = tracks.get(trajectory.track_id)
        if track is None:
            continue
        marker_types = {row.marker_type.value for row in track.state_markers}
        if not marker_types & {"retired", "video_end"}:
            rows.append(_residual(
                hypothesis_id=hypothesis.hypothesis_id, family=ResidualFamily.OBJECT_IDENTITY,
                constraint_id="track_endpoint_accounting", basis=EvaluationBasis.NOT_EVALUABLE,
                frame=track.last_observed_frame,
                timestamp=track.observations[-1].timestamp_s, metric_name="endpoint_explanation",
                metric_unit="categorical", reason="track endpoint has no terminal marker",
                config=config, component_id=trajectory.component_id, track_id=trajectory.track_id,
            ))
        if trajectory.semantic_static_prior and trajectory.motion_state == MotionState.MOVING:
            rows.append(_residual(
                hypothesis_id=hypothesis.hypothesis_id, family=ResidualFamily.SEMANTIC,
                constraint_id="semantic_static_motion", basis=EvaluationBasis.FROZEN_KNOWLEDGE,
                frame=frames[0] if frames else track.first_observed_frame,
                timestamp=trajectory.observations[0].timestamp_s if trajectory.observations else track.observations[0].timestamp_s,
                metric_name="semantic_motion_disagreement", metric_unit="binary", raw=1.0,
                normalized=2.0, uncertainty=0.5, threshold=config.conflict_z_threshold * 0.5,
                predicted=(1.0,), observed=(0.0,), reason="semantic-static class is reconstructed as moving",
                config=config, component_id=trajectory.component_id, track_id=trajectory.track_id,
                limitations=("semantic_prior_is_soft_not_a_physical_law",),
            ))
    return rows


def _conflict_windows(hypothesis_id: str, residuals, config):
    conflicting = [
        row for row in residuals if row.evaluable and row.normalized_residual is not None
        and row.normalized_residual >= config.conflict_z_threshold
    ]
    groups = defaultdict(list)
    for row in conflicting:
        groups[(row.family, row.constraint_id, row.track_id, row.component_id)].append(row)
    windows = []
    for (family, constraint, _, _), group in groups.items():
        ordered = sorted(group, key=lambda row: row.start_frame_index)
        chunks = [[ordered[0]]]
        for row in ordered[1:]:
            if row.start_frame_index <= chunks[-1][-1].end_frame_index + config.conflict_merge_gap_frames + 1:
                chunks[-1].append(row)
            else:
                chunks.append([row])
        for index, chunk in enumerate(chunks):
            peak = max(float(row.normalized_residual) for row in chunk)
            hard = any(row.severity == ResidualSeverity.HARD_VIOLATION for row in chunk)
            windows.append(ConflictWindow(
                conflict_id=f"conflict:{hypothesis_id}:{constraint}:{index}",
                hypothesis_id=hypothesis_id, family=family, constraint_id=constraint,
                start_frame_index=min(row.start_frame_index for row in chunk),
                end_frame_index=max(row.end_frame_index for row in chunk),
                residual_ids=tuple(row.residual_id for row in chunk), peak_normalized_residual=peak,
                severity=ResidualSeverity.HARD_VIOLATION if hard else ResidualSeverity.VIOLATION,
                component_ids=tuple(sorted({row.component_id for row in chunk if row.component_id})),
                track_ids=tuple(sorted({row.track_id for row in chunk if row.track_id})),
                check_evidence_supported=any(row.evaluation_basis == EvaluationBasis.CHECK_EVIDENCE for row in chunk),
            ))
    return tuple(windows)


def _packet(*, hypothesis, video, resolver, config):
    residuals = []
    residuals.extend(_fit_reprojection_residuals(hypothesis=hypothesis, geometry=video.geometry, config=config))
    residuals.extend(_check_depth_residuals(hypothesis=hypothesis, video=video, resolver=resolver, config=config))
    residuals.extend(_check_object_flow_residuals(hypothesis=hypothesis, video=video, resolver=resolver, config=config))
    residuals.extend(_check_background_flow_residuals(hypothesis=hypothesis, video=video, resolver=resolver, config=config))
    residuals.extend(_physics_residuals(hypothesis=hypothesis, config=config))
    residuals.extend(_identity_semantic_residuals(hypothesis=hypothesis, tracking=video.tracking, config=config))
    residuals = tuple(residuals)
    conflicts = _conflict_windows(hypothesis.hypothesis_id, residuals, config)
    summaries = []
    for family in ResidualFamily:
        family_rows = [row for row in residuals if row.family == family]
        evaluable = [row for row in family_rows if row.evaluable]
        violations = [row for row in evaluable if row.severity in {ResidualSeverity.VIOLATION, ResidualSeverity.HARD_VIOLATION}]
        summaries.append(ResidualFamilySummary(
            family=family, total_count=len(family_rows), evaluable_count=len(evaluable),
            check_evidence_count=sum(row.evaluation_basis == EvaluationBasis.CHECK_EVIDENCE for row in evaluable),
            violation_count=len(violations),
            hard_violation_count=sum(row.severity == ResidualSeverity.HARD_VIOLATION for row in violations),
            peak_normalized_residual=max((float(row.normalized_residual) for row in evaluable), default=None),
        ))
    evaluable_fraction = sum(row.evaluable for row in residuals) / max(1, len(residuals))
    status = "insufficient_evidence" if evaluable_fraction < config.minimum_evaluable_fraction else ("conflicts_detected" if conflicts else "no_conflict")
    return HypothesisResidualPacket(
        packet_id=f"packet:{hypothesis.hypothesis_id}", hypothesis_id=hypothesis.hypothesis_id,
        hypothesis_rank=hypothesis.rank, residuals=residuals, conflict_windows=conflicts,
        family_summaries=tuple(summaries), evaluable_fraction=evaluable_fraction,
        check_evidence_residual_count=sum(row.evaluable and row.evaluation_basis == EvaluationBasis.CHECK_EVIDENCE for row in residuals),
        check_supported_conflict_count=sum(row.check_evidence_supported for row in conflicts),
        hard_violation=any(row.severity == ResidualSeverity.HARD_VIOLATION for row in residuals),
        status=status,
    )


def _video_manifest(*, loaded, video, world_store_ref, config, config_sha256):
    step2_root = loaded.run_root / video.tracking.input_snapshot.source_step2_relative_root
    step3_root = loaded.run_root / video.world.input_snapshot.source_step3_relative_root
    resolver = _ArtifactResolver(run_root=loaded.run_root, step2_root=step2_root, step3_root=step3_root)
    hypotheses = video.world.initial_beam.hypotheses[: config.maximum_hypotheses]
    packets = tuple(_packet(hypothesis=row, video=video, resolver=resolver, config=config) for row in hypotheses)
    residuals = [row for packet in packets for row in packet.residuals]
    input_snapshot = Step6InputSnapshot(
        source_step5_relative_root=loaded.stage_root.relative_to(loaded.run_root).as_posix(),
        world_state_store=_link(ArtifactOwner.STEP5_WORLD_RECONSTRUCTION, world_store_ref),
        video_world_state_manifest=_link(ArtifactOwner.STEP5_WORLD_RECONSTRUCTION, video.world_reference),
        source_step4_relative_root=video.world.input_snapshot.source_step4_relative_root,
        video_geometry_manifest=video.world.input_snapshot.video_geometry_manifest,
        source_step3_relative_root=video.world.input_snapshot.source_step3_relative_root,
        video_tracking_manifest=video.world.input_snapshot.video_tracking_manifest,
        source_step2_relative_root=video.tracking.input_snapshot.source_step2_relative_root,
        video_evidence_manifest=video.tracking.input_snapshot.video_evidence_manifest,
    )
    return VideoResidualManifest(
        run_id=video.world.run_id, video_id=video.world.video_id,
        source_world_state_sha256=video.world_reference.sha256, config_sha256=config_sha256,
        canonical_fps=video.world.canonical_fps, image_size=video.world.image_size,
        frame_count=video.world.frame_count, input_snapshot=input_snapshot, packets=packets,
        validation=Step6ValidationSummary(
            input_hypothesis_count=len(video.world.initial_beam.hypotheses),
            evaluated_hypothesis_count=len(packets), residual_count=len(residuals),
            evaluable_residual_count=sum(row.evaluable for row in residuals),
            check_evidence_residual_count=sum(row.evaluable and row.evaluation_basis == EvaluationBasis.CHECK_EVIDENCE for row in residuals),
            conflict_window_count=sum(len(packet.conflict_windows) for packet in packets), overall_pass=True,
        ),
        tool_versions=(ToolVersion(name="numpy", version=np.__version__), ToolVersion(name="opencv", version=cv2.__version__)),
    )


def run_step6(
    *, world_state_store_path: Path | str, maximum_hypotheses: int = 64,
    projection_sigma_px: float = 5.0, depth_log_sigma: float = 0.35,
    flow_sigma_px: float = 3.0, background_sample_stride: int = 24,
    maximum_background_samples: int = 2048, maximum_prediction_gap_frames: int = 2,
    conflict_z_threshold: float = 3.0, hard_z_threshold: float = 6.0,
    conflict_merge_gap_frames: int = 1, metric_max_ego_speed_mps: float = 70.0,
    metric_max_object_speed_mps: float = 80.0, metric_max_acceleration_mps2: float = 15.0,
    relative_acceleration_scale: float = 10.0, minimum_evaluable_fraction: float = 0.05,
) -> Step6Result:
    """Evaluate every available Step 5 hypothesis without mutating or ranking it."""
    loaded = _load_step5(world_state_store_path)
    config = Step6ConfigSnapshot(
        maximum_hypotheses=maximum_hypotheses, projection_sigma_px=projection_sigma_px,
        depth_log_sigma=depth_log_sigma, flow_sigma_px=flow_sigma_px,
        background_sample_stride=background_sample_stride,
        maximum_background_samples=maximum_background_samples,
        maximum_prediction_gap_frames=maximum_prediction_gap_frames,
        conflict_z_threshold=conflict_z_threshold, hard_z_threshold=hard_z_threshold,
        conflict_merge_gap_frames=conflict_merge_gap_frames,
        metric_max_ego_speed_mps=metric_max_ego_speed_mps,
        metric_max_object_speed_mps=metric_max_object_speed_mps,
        metric_max_acceleration_mps2=metric_max_acceleration_mps2,
        relative_acceleration_scale=relative_acceleration_scale,
        minimum_evaluable_fraction=minimum_evaluable_fraction,
    )
    config_sha256 = hash_payload(config)
    source_sha256 = sha256_file(loaded.store_path)
    stage_root = loaded.run_root / "06_predict_verify" / f"input_{source_sha256[:16]}" / f"config_{config_sha256[:16]}"
    stage_root.mkdir(parents=True, exist_ok=True)
    world_store_ref = _file_reference(
        path=loaded.store_path, stage_root=loaded.stage_root,
        artifact_id=f"world-state-store:{loaded.store.run_id}",
    )
    manifests, references = [], []
    for video in loaded.videos:
        manifest = _video_manifest(
            loaded=loaded, video=video, world_store_ref=world_store_ref,
            config=config, config_sha256=config_sha256,
        )
        relative_path = Path("videos") / f"{video.world.video_id}.residuals.json"
        path = stage_root / relative_path
        sha256, byte_size = write_contract(path, manifest)
        references.append(ArtifactRef(
            artifact_id=f"video-residuals:{video.world.video_id}",
            relative_path=relative_path.as_posix(), sha256=sha256, byte_size=byte_size,
            media_type="application/vnd.cauvid.residual-packet+json", coordinate_space=None,
        ))
        manifests.append(manifest)
    store = ResidualStore(
        run_id=loaded.store.run_id, source_world_state_store_sha256=source_sha256,
        config=config, config_sha256=config_sha256, video_ids=loaded.store.video_ids,
        video_residuals=tuple(references),
    )
    store_path_out = stage_root / "residual_store.json"
    write_contract(store_path_out, store)
    return Step6Result(store=store, video_manifests=tuple(manifests), stage_root=stage_root, store_path=store_path_out)


__all__ = ["Step6Result", "run_step6"]
