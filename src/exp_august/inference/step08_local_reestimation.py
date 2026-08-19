"""Step 8: instantiate bounded local repair proposals as child hypotheses.

The numerical solver reads only fit evidence, frozen physical constraints, and
the parent state.  Check-only residuals are recorded as excluded inputs and are
left to Step 9 acceptance and ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.exp_august.contracts import (
    ArtifactLink,
    ArtifactOwner,
    ArtifactRef,
    DiscreteStateChange,
    EvaluationBasis,
    HypothesisReestimationPacket,
    LocalReestimationCandidate,
    LocalReestimationStore,
    MotionState,
    NumericStateChange,
    ObjectiveTerms,
    ProposalReestimationResult,
    RepairOperator,
    RepairProposalStore,
    ResidualStore,
    Step8ConfigSnapshot,
    Step8InputSnapshot,
    Step8ValidationSummary,
    ToolVersion,
    VideoGeometryManifest,
    VideoLocalReestimationManifest,
    VideoRepairProposalManifest,
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
from src.exp_august.inference.step06_predict_verify import (
    _fit_reprojection_residuals,
    _physics_residuals,
)


@dataclass(frozen=True)
class Step8Result:
    store: LocalReestimationStore
    video_manifests: tuple[VideoLocalReestimationManifest, ...]
    stage_root: Path
    store_path: Path


@dataclass(frozen=True)
class _LoadedVideo:
    repair: VideoRepairProposalManifest
    repair_reference: ArtifactRef
    residual: VideoResidualManifest
    world: VideoWorldStateManifest
    world_path: Path
    geometry: VideoGeometryManifest
    tracking: VideoTrackingManifest


@dataclass(frozen=True)
class _LoadedStep7:
    store_path: Path
    stage_root: Path
    run_root: Path
    store: RepairProposalStore
    videos: tuple[_LoadedVideo, ...]


@dataclass(frozen=True)
class _ObjectiveEvaluation:
    terms: ObjectiveTerms
    has_fit_evidence: bool


def _verified_contract(path: Path, reference: ArtifactRef, model, *, label: str):
    if not path.is_file() or path.stat().st_size != reference.byte_size:
        raise RuntimeError(f"Step 8 {label} is missing or truncated: {path}")
    if sha256_file(path) != reference.sha256:
        raise RuntimeError(f"Step 8 {label} failed integrity check: {path}")
    return read_contract(path, model)


def _find_run_root(store_path: Path) -> Path:
    for parent in store_path.parents:
        if parent.name == "07_diagnose_propose":
            return parent.parent
    raise RuntimeError("Step 7 store must live below 07_diagnose_propose")


def _load_step7(store_path: Path | str) -> _LoadedStep7:
    resolved = Path(store_path).expanduser().resolve()
    store = read_contract(resolved, RepairProposalStore)
    stage_root = resolved.parent
    run_root = _find_run_root(resolved)
    videos = []
    for video_id, reference in zip(store.video_ids, store.video_repair_proposals):
        repair_path = stage_root / reference.relative_path
        repair = _verified_contract(
            repair_path,
            reference,
            VideoRepairProposalManifest,
            label="repair proposal manifest",
        )
        if repair.video_id != video_id or repair.run_id != store.run_id:
            raise RuntimeError(f"Step 8 repair identity mismatch: {repair_path}")
        snapshot = repair.input_snapshot
        residual_store_path = (
            run_root
            / snapshot.source_step6_relative_root
            / snapshot.residual_store.artifact.relative_path
        )
        _verified_contract(
            residual_store_path,
            snapshot.residual_store.artifact,
            ResidualStore,
            label="residual store",
        )
        if sha256_file(residual_store_path) != store.source_residual_store_sha256:
            raise RuntimeError("Step 8 found inconsistent Step 6 store lineage")
        residual_path = (
            run_root
            / snapshot.source_step6_relative_root
            / snapshot.video_residual_manifest.artifact.relative_path
        )
        residual = _verified_contract(
            residual_path,
            snapshot.video_residual_manifest.artifact,
            VideoResidualManifest,
            label="residual manifest",
        )
        world_path = (
            run_root
            / snapshot.source_step5_relative_root
            / snapshot.video_world_state_manifest.artifact.relative_path
        )
        world = _verified_contract(
            world_path,
            snapshot.video_world_state_manifest.artifact,
            VideoWorldStateManifest,
            label="world-state manifest",
        )
        world_store_link = residual.input_snapshot.world_state_store
        world_store_path = (
            run_root
            / snapshot.source_step5_relative_root
            / world_store_link.artifact.relative_path
        )
        world_store = _verified_contract(
            world_store_path,
            world_store_link.artifact,
            WorldStateStore,
            label="world-state store",
        )
        if world_store.run_id != store.run_id or video_id not in world_store.video_ids:
            raise RuntimeError(f"Step 8 world-state store identity mismatch: {video_id}")
        geometry_link = residual.input_snapshot.video_geometry_manifest
        geometry_path = (
            run_root
            / residual.input_snapshot.source_step4_relative_root
            / geometry_link.artifact.relative_path
        )
        geometry = _verified_contract(
            geometry_path,
            geometry_link.artifact,
            VideoGeometryManifest,
            label="geometry manifest",
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
            label="tracking manifest",
        )
        if {residual.video_id, world.video_id, geometry.video_id, tracking.video_id} != {
            video_id
        }:
            raise RuntimeError(f"Step 8 cross-stage video identity mismatch: {video_id}")
        parent_ids = tuple(row.hypothesis_id for row in repair.packets)
        if parent_ids != tuple(
            row.hypothesis_id for row in residual.packets[: len(parent_ids)]
        ):
            raise RuntimeError(f"Step 8 Step 6/7 hypothesis lineage mismatch: {video_id}")
        if not set(parent_ids).issubset(
            {row.hypothesis_id for row in world.initial_beam.hypotheses}
        ):
            raise RuntimeError(f"Step 8 parent hypothesis is absent from Step 5: {video_id}")
        videos.append(
            _LoadedVideo(
                repair=repair,
                repair_reference=reference,
                residual=residual,
                world=world,
                world_path=world_path,
                geometry=geometry,
                tracking=tracking,
            )
        )
    return _LoadedStep7(
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


def _replace(model, **updates):
    payload = model.model_dump(mode="python")
    payload.update(updates)
    return type(model).model_validate(payload)


def _vector(value) -> np.ndarray:
    return np.asarray((value.x, value.y, value.z), dtype=np.float64)


def _vector_payload(value: np.ndarray) -> dict[str, float]:
    return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}


def _unique(values: Iterable[str | None]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value is not None))


def _target_subjects(proposal, residual_packet) -> tuple[tuple[str, ...], tuple[str, ...]]:
    by_id = {row.residual_id: row for row in residual_packet.residuals}
    rows = [by_id[value] for value in proposal.target_residual_ids if value in by_id]
    return (
        _unique(row.component_id for row in rows),
        _unique(row.track_id for row in rows),
    )


def _overlaps(row, proposal) -> bool:
    return not (
        row.end_frame_index < proposal.start_frame_index
        or row.start_frame_index > proposal.end_frame_index
    )


def _objective(
    *, hypothesis, proposal, residual_packet, video: _LoadedVideo,
    step6_config, config: Step8ConfigSnapshot,
) -> _ObjectiveEvaluation:
    component_ids, track_ids = _target_subjects(proposal, residual_packet)

    def relevant(row) -> bool:
        if not _overlaps(row, proposal):
            return False
        if track_ids:
            return row.track_id in set(track_ids)
        if component_ids:
            return row.component_id in set(component_ids)
        return True

    fit_rows = [
        row
        for row in _fit_reprojection_residuals(
            hypothesis=hypothesis,
            geometry=video.geometry,
            config=step6_config,
        )
        if row.evaluable and relevant(row)
    ]
    physics_rows = [
        row
        for row in _physics_residuals(hypothesis=hypothesis, config=step6_config)
        if row.evaluable and relevant(row)
    ]

    def mean_square(rows) -> float:
        values = [float(row.normalized_residual) ** 2 for row in rows]
        return float(np.mean(values)) if values else 0.0

    fit_error = mean_square(fit_rows)
    physics_error = mean_square(physics_rows)
    total = (
        config.fit_objective_weight * fit_error
        + config.physics_objective_weight * physics_error
    )
    return _ObjectiveEvaluation(
        terms=ObjectiveTerms(
            fit_evidence_error=fit_error,
            physics_error=physics_error,
            total=float(total),
            fit_residual_count=len(fit_rows),
            physics_residual_count=len(physics_rows),
        ),
        has_fit_evidence=bool(fit_rows),
    )


def _numeric_upper(proposal, names: set[str], default: float) -> float:
    for bound in proposal.parameter_bounds:
        if bound.parameter_name in names and bound.upper_bound is not None:
            return float(bound.upper_bound)
    return default


def _candidate_count(proposal, config: Step8ConfigSnapshot) -> int:
    return max(
        1,
        min(
            proposal.compute_budget.maximum_child_hypotheses,
            config.maximum_candidates_per_proposal,
        ),
    )


def _change(
    *, path: str, frame: int, before: Iterable[float], after: Iterable[float],
    unit: str, standardized: float | None = None,
) -> NumericStateChange | None:
    before_values = tuple(float(value) for value in before)
    after_values = tuple(float(value) for value in after)
    if np.allclose(before_values, after_values, rtol=0.0, atol=1e-12):
        return None
    return NumericStateChange(
        field_path=path,
        frame_index=frame,
        before_values=before_values,
        after_values=after_values,
        unit=unit,
        maximum_standardized_delta=(
            float(standardized) if standardized is not None else None
        ),
    )


def _smooth_positions(
    states,
    *,
    start_frame: int,
    end_frame: int,
    strength: float,
    maximum_sigma: float,
    minimum_std: float,
) -> dict[int, tuple[np.ndarray, float]]:
    scoped = [
        row for row in states if start_frame <= row.frame_index <= end_frame
    ]
    if len(scoped) < 3:
        return {}
    left, right = scoped[0], scoped[-1]
    left_position = _vector(left.position)
    right_position = _vector(right.position)
    duration = max(right.timestamp_s - left.timestamp_s, 1e-12)
    updates = {}
    for row in scoped[1:-1]:
        fraction = np.clip((row.timestamp_s - left.timestamp_s) / duration, 0.0, 1.0)
        target = (1.0 - fraction) * left_position + fraction * right_position
        current = _vector(row.position)
        requested = strength * (target - current)
        std = np.maximum(_vector(row.position_std), minimum_std)
        bounded = np.clip(requested, -maximum_sigma * std, maximum_sigma * std)
        updated = current + bounded
        standardized = float(np.max(np.abs(bounded) / std))
        if not np.allclose(current, updated, rtol=0.0, atol=1e-12):
            updates[row.frame_index] = (updated, standardized)
    return updates


def _motion_fields(states, positions: dict[int, np.ndarray], scoped_frames: set[int]):
    ordered = list(states)
    output = {}
    for index, row in enumerate(ordered):
        if row.frame_index not in scoped_frames or row.velocity is None:
            continue
        if index == 0 or index == len(ordered) - 1:
            continue
        left, right = ordered[index - 1], ordered[index + 1]
        dt = right.timestamp_s - left.timestamp_s
        if dt <= 0:
            continue
        left_position = positions.get(left.frame_index, _vector(left.position))
        right_position = positions.get(right.frame_index, _vector(right.position))
        velocity = (right_position - left_position) / dt
        speed = float(np.linalg.norm(velocity))
        if row.speed_interval is None:
            interval = None
        else:
            lower_margin = max(0.0, float(row.speed) - float(row.speed_interval[0]))
            upper_margin = max(0.0, float(row.speed_interval[1]) - float(row.speed))
            interval = (max(0.0, speed - lower_margin), speed + upper_margin)
        output[row.frame_index] = (velocity, speed, interval)
    return output


def _refit_ego_component(
    component,
    *,
    proposal,
    strength: float,
    maximum_sigma: float,
    minimum_std: float,
):
    updates = _smooth_positions(
        component.poses,
        start_frame=proposal.start_frame_index,
        end_frame=proposal.end_frame_index,
        strength=strength,
        maximum_sigma=maximum_sigma,
        minimum_std=minimum_std,
    )
    if not updates:
        return component, ()
    positions = {frame: value for frame, (value, _) in updates.items()}
    scoped_frames = set(updates)
    motions = _motion_fields(component.poses, positions, scoped_frames)
    changes = []
    poses = []
    unit = component.coordinate_unit.value
    for pose in component.poses:
        if pose.frame_index not in updates:
            poses.append(pose)
            continue
        position, standardized = updates[pose.frame_index]
        rotation = np.asarray(pose.rotation_world_to_camera, dtype=np.float64).reshape(3, 3)
        translation = -(rotation @ position)
        path = f"ego_components.{component.component_id}.poses.{pose.frame_index}"
        changes.append(
            _change(
                path=f"{path}.position",
                frame=pose.frame_index,
                before=_vector(pose.position),
                after=position,
                unit=unit,
                standardized=standardized,
            )
        )
        changes.append(
            _change(
                path=f"{path}.translation_world_to_camera",
                frame=pose.frame_index,
                before=_vector(pose.translation_world_to_camera),
                after=translation,
                unit=unit,
            )
        )
        updates_for_pose = {
            "position": _vector_payload(position),
            "translation_world_to_camera": _vector_payload(translation),
        }
        if pose.frame_index in motions:
            velocity, speed, interval = motions[pose.frame_index]
            changes.append(
                _change(
                    path=f"{path}.velocity",
                    frame=pose.frame_index,
                    before=_vector(pose.velocity),
                    after=velocity,
                    unit=f"{unit}_per_second",
                )
            )
            changes.append(
                _change(
                    path=f"{path}.speed",
                    frame=pose.frame_index,
                    before=(pose.speed,),
                    after=(speed,),
                    unit=f"{unit}_per_second",
                )
            )
            updates_for_pose.update(
                velocity=_vector_payload(velocity),
                speed=speed,
                speed_interval=interval,
            )
        poses.append(_replace(pose, **updates_for_pose))
    return _replace(component, poses=tuple(poses)), tuple(row for row in changes if row)


def _refit_object_trajectory(
    trajectory,
    *,
    proposal,
    strength: float,
    maximum_sigma: float,
    minimum_std: float,
):
    updates = _smooth_positions(
        trajectory.observations,
        start_frame=proposal.start_frame_index,
        end_frame=proposal.end_frame_index,
        strength=strength,
        maximum_sigma=maximum_sigma,
        minimum_std=minimum_std,
    )
    if not updates:
        return trajectory, ()
    positions = {frame: value for frame, (value, _) in updates.items()}
    motions = _motion_fields(trajectory.observations, positions, set(updates))
    changes = []
    observations = []
    unit = trajectory.coordinate_unit.value
    for state in trajectory.observations:
        if state.frame_index not in updates:
            observations.append(state)
            continue
        position, standardized = updates[state.frame_index]
        path = f"object_trajectories.{trajectory.track_id}.states.{state.frame_index}"
        changes.append(
            _change(
                path=f"{path}.position",
                frame=state.frame_index,
                before=_vector(state.position),
                after=position,
                unit=unit,
                standardized=standardized,
            )
        )
        updates_for_state = {"position": _vector_payload(position)}
        if state.frame_index in motions:
            velocity, speed, interval = motions[state.frame_index]
            changes.append(
                _change(
                    path=f"{path}.velocity",
                    frame=state.frame_index,
                    before=_vector(state.velocity),
                    after=velocity,
                    unit=f"{unit}_per_second",
                )
            )
            changes.append(
                _change(
                    path=f"{path}.speed",
                    frame=state.frame_index,
                    before=(state.speed,),
                    after=(speed,),
                    unit=f"{unit}_per_second",
                )
            )
            updates_for_state.update(
                velocity=_vector_payload(velocity),
                speed=speed,
                speed_interval=interval,
            )
        observations.append(_replace(state, **updates_for_state))
    return (
        _replace(trajectory, observations=tuple(observations)),
        tuple(row for row in changes if row),
    )


def _child_id(*, parent_id: str, proposal_id: str, ordinal: int, changes) -> str:
    digest = hash_payload(
        {
            "parent": parent_id,
            "proposal": proposal_id,
            "ordinal": ordinal,
            "changes": [row.model_dump(mode="json") for row in changes],
        }
    )[:20]
    return f"world:step8:{digest}"


def _candidate_id(proposal_id: str, ordinal: int, status: str) -> str:
    return "candidate:" + hash_payload(
        {"proposal": proposal_id, "ordinal": ordinal, "status": status}
    )[:20]


def _excluded_check_ids(proposal) -> tuple[str, ...]:
    return tuple(
        row.residual_id
        for row in proposal.expected_residual_effects
        if row.evaluation_basis == EvaluationBasis.CHECK_EVIDENCE
    )


def _optimized_source_ids(proposal) -> tuple[str, ...]:
    return tuple(
        row.residual_id
        for row in proposal.expected_residual_effects
        if row.optimized_by_step8
    )


def _encoded_ids(values: Iterable[int | str]) -> str:
    encoded = ",".join(str(value) for value in values)
    return encoded or "none"


def _audit_candidate(
    *, proposal, status: str, method: str, limitations: tuple[str, ...],
) -> LocalReestimationCandidate:
    return LocalReestimationCandidate(
        candidate_id=_candidate_id(proposal.proposal_id, 0, status),
        proposal_id=proposal.proposal_id,
        parent_hypothesis_id=proposal.parent_hypothesis_id,
        operator=proposal.operator,
        status=status,
        solver_method=method,
        solver_iterations=0,
        boundary_preserved=True,
        parameter_bounds_satisfied=True,
        compute_budget_honored=True,
        self_consistency_only=True,
        excluded_check_residual_ids=_excluded_check_ids(proposal),
        limitations=limitations,
    )


def _refit_candidates(
    *, parent, proposal, residual_packet, video, step6_config, config,
) -> tuple[LocalReestimationCandidate, ...]:
    component_ids, track_ids = _target_subjects(proposal, residual_packet)
    affects_ego = any(value.startswith("ego_components") for value in proposal.affected_variables)
    affects_objects = any(
        value.startswith("object_trajectories") for value in proposal.affected_variables
    )
    maximum_sigma = _numeric_upper(
        proposal,
        {"maximum_pose_delta_sigma", "maximum_state_delta_sigma"},
        3.0,
    )
    before = _objective(
        hypothesis=parent,
        proposal=proposal,
        residual_packet=residual_packet,
        video=video,
        step6_config=step6_config,
        config=config,
    )
    candidates = []
    count = _candidate_count(proposal, config)
    for ordinal in range(1, count + 1):
        strength = ordinal / count
        changes = []
        components = []
        for component in parent.ego_components:
            selected = affects_ego and (
                not component_ids or component.component_id in set(component_ids)
            )
            if selected:
                updated, component_changes = _refit_ego_component(
                    component,
                    proposal=proposal,
                    strength=strength,
                    maximum_sigma=maximum_sigma,
                    minimum_std=config.minimum_position_std,
                )
                components.append(updated)
                changes.extend(component_changes)
            else:
                components.append(component)
        trajectories = []
        for trajectory in parent.object_trajectories:
            selected = affects_objects and (
                not track_ids or trajectory.track_id in set(track_ids)
            )
            if selected:
                updated, trajectory_changes = _refit_object_trajectory(
                    trajectory,
                    proposal=proposal,
                    strength=strength,
                    maximum_sigma=maximum_sigma,
                    minimum_std=config.minimum_position_std,
                )
                trajectories.append(updated)
                changes.extend(trajectory_changes)
            else:
                trajectories.append(trajectory)
        if not changes:
            continue
        child_id = _child_id(
            parent_id=parent.hypothesis_id,
            proposal_id=proposal.proposal_id,
            ordinal=ordinal,
            changes=changes,
        )
        child = _replace(
            parent,
            hypothesis_id=child_id,
            ego_components=tuple(components),
            object_trajectories=tuple(trajectories),
            limitations=tuple(parent.limitations)
            + (
                "step8_local_reestimation_candidate",
                "rotation_held_fixed_by_bounded_local_solver_v1",
            ),
        )
        after = _objective(
            hypothesis=child,
            proposal=proposal,
            residual_packet=residual_packet,
            video=video,
            step6_config=step6_config,
            config=config,
        )
        candidates.append(
            LocalReestimationCandidate(
                candidate_id=_candidate_id(proposal.proposal_id, ordinal, "instantiated"),
                proposal_id=proposal.proposal_id,
                parent_hypothesis_id=parent.hypothesis_id,
                operator=proposal.operator,
                status="instantiated",
                child_hypothesis=child,
                numerical_changes=tuple(changes),
                objective_before=before.terms,
                objective_after=after.terms,
                optimized_residual_ids=_optimized_source_ids(proposal),
                excluded_check_residual_ids=_excluded_check_ids(proposal),
                solver_method="bounded_linear_state_refit_v1",
                solver_iterations=1,
                boundary_preserved=True,
                parameter_bounds_satisfied=all(
                    row.maximum_standardized_delta is None
                    or row.maximum_standardized_delta <= maximum_sigma + 1e-9
                    for row in changes
                ),
                compute_budget_honored=(
                    proposal.compute_budget.maximum_solver_iterations >= 1
                ),
                self_consistency_only=not after.has_fit_evidence,
                limitations=(
                    "check_only_evidence_excluded_from_optimizer",
                    "candidate_requires_step9_acceptance",
                ),
            )
        )
    if candidates:
        return tuple(candidates)
    return (
        _audit_candidate(
            proposal=proposal,
            status="no_change",
            method="bounded_linear_state_refit_v1",
            limitations=(
                "fewer_than_three_mutable_states_or_zero_bounded_delta",
                "check_only_evidence_excluded_from_optimizer",
            ),
        ),
    )


def _adjust_noise_candidates(
    *, parent, proposal, residual_packet, video, step6_config, config,
) -> tuple[LocalReestimationCandidate, ...]:
    component_ids, track_ids = _target_subjects(proposal, residual_packet)
    lower, upper = 1.0, 3.0
    for bound in proposal.parameter_bounds:
        if bound.parameter_name == "process_noise_multiplier":
            lower, upper = float(bound.lower_bound), float(bound.upper_bound)
    before = _objective(
        hypothesis=parent,
        proposal=proposal,
        residual_packet=residual_packet,
        video=video,
        step6_config=step6_config,
        config=config,
    )
    candidates = []
    count = _candidate_count(proposal, config)
    values = (
        np.linspace(lower, upper, count + 1)[1:]
        if np.isclose(lower, 1.0) and not np.isclose(upper, lower)
        else np.linspace(lower, upper, count)
    )
    for ordinal, multiplier in enumerate(values, start=1):
        changes = []
        components = []
        for component in parent.ego_components:
            poses = []
            for pose in component.poses:
                selected = (
                    proposal.start_frame_index <= pose.frame_index <= proposal.end_frame_index
                    and (not component_ids or component.component_id in set(component_ids))
                    and not track_ids
                )
                if not selected:
                    poses.append(pose)
                    continue
                old = _vector(pose.position_std)
                new = old * float(multiplier)
                row = _change(
                    path=f"ego_components.{component.component_id}.poses.{pose.frame_index}.position_std",
                    frame=pose.frame_index,
                    before=old,
                    after=new,
                    unit=component.coordinate_unit.value,
                )
                if row:
                    changes.append(row)
                poses.append(_replace(pose, position_std=_vector_payload(new)))
            components.append(_replace(component, poses=tuple(poses)))
        trajectories = []
        for trajectory in parent.object_trajectories:
            observations = []
            for state in trajectory.observations:
                selected = (
                    proposal.start_frame_index <= state.frame_index <= proposal.end_frame_index
                    and (not track_ids or trajectory.track_id in set(track_ids))
                )
                if not selected:
                    observations.append(state)
                    continue
                old = _vector(state.position_std)
                new = old * float(multiplier)
                row = _change(
                    path=f"object_trajectories.{trajectory.track_id}.states.{state.frame_index}.position_std",
                    frame=state.frame_index,
                    before=old,
                    after=new,
                    unit=trajectory.coordinate_unit.value,
                )
                if row:
                    changes.append(row)
                observations.append(_replace(state, position_std=_vector_payload(new)))
            trajectories.append(_replace(trajectory, observations=tuple(observations)))
        if not changes:
            continue
        child_id = _child_id(
            parent_id=parent.hypothesis_id,
            proposal_id=proposal.proposal_id,
            ordinal=ordinal,
            changes=changes,
        )
        child = _replace(
            parent,
            hypothesis_id=child_id,
            ego_components=tuple(components),
            object_trajectories=tuple(trajectories),
            limitations=tuple(parent.limitations)
            + ("step8_process_noise_candidate",),
        )
        after = _objective(
            hypothesis=child,
            proposal=proposal,
            residual_packet=residual_packet,
            video=video,
            step6_config=step6_config,
            config=config,
        )
        candidates.append(
            LocalReestimationCandidate(
                candidate_id=_candidate_id(proposal.proposal_id, ordinal, "instantiated"),
                proposal_id=proposal.proposal_id,
                parent_hypothesis_id=parent.hypothesis_id,
                operator=proposal.operator,
                status="instantiated",
                child_hypothesis=child,
                numerical_changes=tuple(changes),
                objective_before=before.terms,
                objective_after=after.terms,
                optimized_residual_ids=_optimized_source_ids(proposal),
                excluded_check_residual_ids=_excluded_check_ids(proposal),
                solver_method="bounded_process_noise_scaling_v1",
                solver_iterations=1,
                boundary_preserved=True,
                parameter_bounds_satisfied=lower <= float(multiplier) <= upper,
                compute_budget_honored=(
                    proposal.compute_budget.maximum_solver_iterations >= 1
                ),
                self_consistency_only=not after.has_fit_evidence,
                limitations=(
                    "check_only_evidence_excluded_from_optimizer",
                    "candidate_requires_step9_acceptance",
                ),
            )
        )
    if candidates:
        return tuple(candidates)
    return (
        _audit_candidate(
            proposal=proposal,
            status="no_change",
            method="bounded_process_noise_scaling_v1",
            limitations=("selected process-noise multiplier leaves state unchanged",),
        ),
    )


def _mark_unobservable_candidate(
    *, parent, proposal, residual_packet, video, step6_config, config,
) -> tuple[LocalReestimationCandidate, ...]:
    _, track_ids = _target_subjects(proposal, residual_packet)
    unresolved_frames = set(parent.unresolved_ego_frame_indices)
    if not track_ids:
        unresolved_frames.update(
            range(proposal.start_frame_index, proposal.end_frame_index + 1)
        )
    unresolved_objects = set(parent.unresolved_object_observation_ids)
    trajectories = []
    changes = []
    for trajectory in parent.object_trajectories:
        selected = not track_ids or trajectory.track_id in set(track_ids)
        if not selected:
            trajectories.append(trajectory)
            continue
        unresolved_objects.update(
            row.state_id
            for row in trajectory.observations
            if proposal.start_frame_index <= row.frame_index <= proposal.end_frame_index
        )
        if trajectory.motion_state != MotionState.UNOBSERVABLE:
            changes.append(
                DiscreteStateChange(
                    field_path=f"object_trajectories.{trajectory.track_id}.motion_state",
                    before_value=trajectory.motion_state.value,
                    after_value=MotionState.UNOBSERVABLE.value,
                )
            )
            trajectory = _replace(
                trajectory,
                motion_state=MotionState.UNOBSERVABLE,
                motion_score=None,
                limitations=tuple(trajectory.limitations)
                + ("step8_marked_unobservable",),
            )
        trajectories.append(trajectory)
    if tuple(sorted(unresolved_frames)) != parent.unresolved_ego_frame_indices:
        changes.append(
            DiscreteStateChange(
                field_path="unresolved_ego_frame_indices",
                before_value=_encoded_ids(parent.unresolved_ego_frame_indices),
                after_value=_encoded_ids(sorted(unresolved_frames)),
            )
        )
    if tuple(sorted(unresolved_objects)) != parent.unresolved_object_observation_ids:
        changes.append(
            DiscreteStateChange(
                field_path="unresolved_object_observation_ids",
                before_value=_encoded_ids(parent.unresolved_object_observation_ids),
                after_value=_encoded_ids(sorted(unresolved_objects)),
            )
        )
    mark_world_unobservable = not track_ids
    if mark_world_unobservable and parent.world_frame_status != "unobservable":
        changes.append(
            DiscreteStateChange(
                field_path="world_frame_status",
                before_value=parent.world_frame_status,
                after_value="unobservable",
            )
        )
    if not changes:
        return (
            _audit_candidate(
                proposal=proposal,
                status="no_change",
                method="mark_unobservable_v1",
                limitations=("parent already carries the requested unobservable state",),
            ),
        )
    child_id = _child_id(
        parent_id=parent.hypothesis_id,
        proposal_id=proposal.proposal_id,
        ordinal=1,
        changes=changes,
    )
    child = _replace(
        parent,
        hypothesis_id=child_id,
        world_frame_status=(
            "unobservable" if mark_world_unobservable else parent.world_frame_status
        ),
        object_trajectories=tuple(trajectories),
        unresolved_ego_frame_indices=tuple(sorted(unresolved_frames)),
        unresolved_object_observation_ids=tuple(sorted(unresolved_objects)),
        limitations=tuple(parent.limitations) + ("step8_marked_unobservable",),
    )
    before = _objective(
        hypothesis=parent,
        proposal=proposal,
        residual_packet=residual_packet,
        video=video,
        step6_config=step6_config,
        config=config,
    )
    after = _objective(
        hypothesis=child,
        proposal=proposal,
        residual_packet=residual_packet,
        video=video,
        step6_config=step6_config,
        config=config,
    )
    return (
        LocalReestimationCandidate(
            candidate_id=_candidate_id(proposal.proposal_id, 1, "instantiated"),
            proposal_id=proposal.proposal_id,
            parent_hypothesis_id=parent.hypothesis_id,
            operator=proposal.operator,
            status="instantiated",
            child_hypothesis=child,
            discrete_changes=tuple(changes),
            objective_before=before.terms,
            objective_after=after.terms,
            optimized_residual_ids=_optimized_source_ids(proposal),
            excluded_check_residual_ids=_excluded_check_ids(proposal),
            solver_method="mark_unobservable_v1",
            solver_iterations=1,
            boundary_preserved=True,
            parameter_bounds_satisfied=True,
            compute_budget_honored=True,
            self_consistency_only=not after.has_fit_evidence,
            limitations=(
                "missing evidence is represented explicitly rather than imputed",
                "candidate_requires_step9_acceptance",
            ),
        ),
    )


def _mark_occluded_candidate(
    *, parent, proposal, residual_packet, video, step6_config, config,
) -> tuple[LocalReestimationCandidate, ...]:
    """Represent an evidence gap explicitly without deleting or imputing observations."""

    _, track_ids = _target_subjects(proposal, residual_packet)
    if not track_ids:
        return (
            _audit_candidate(
                proposal=proposal,
                status="unresolved",
                method="mark_occluded_v1",
                limitations=("occlusion proposal has no target track",),
            ),
        )
    marker = (
        f"step8_occluded_window_{proposal.start_frame_index}_"
        f"{proposal.end_frame_index}"
    )
    trajectories = []
    changes = []
    unresolved_objects = set(parent.unresolved_object_observation_ids)
    for trajectory in parent.object_trajectories:
        if trajectory.track_id not in set(track_ids):
            trajectories.append(trajectory)
            continue
        unresolved_objects.update(
            row.state_id
            for row in trajectory.observations
            if proposal.start_frame_index <= row.frame_index <= proposal.end_frame_index
        )
        if marker not in trajectory.limitations:
            changes.append(
                DiscreteStateChange(
                    field_path=f"object_trajectories.{trajectory.track_id}.occlusion_window",
                    before_value="not_declared_occluded",
                    after_value=marker,
                )
            )
            trajectory = _replace(
                trajectory,
                limitations=tuple(trajectory.limitations) + (marker,),
            )
        trajectories.append(trajectory)
    if tuple(sorted(unresolved_objects)) != parent.unresolved_object_observation_ids:
        changes.append(
            DiscreteStateChange(
                field_path="unresolved_object_observation_ids",
                before_value=_encoded_ids(parent.unresolved_object_observation_ids),
                after_value=_encoded_ids(sorted(unresolved_objects)),
            )
        )
    if not changes:
        return (
            _audit_candidate(
                proposal=proposal,
                status="no_change",
                method="mark_occluded_v1",
                limitations=("requested occlusion window is already represented",),
            ),
        )
    child_id = _child_id(
        parent_id=parent.hypothesis_id,
        proposal_id=proposal.proposal_id,
        ordinal=1,
        changes=changes,
    )
    child = _replace(
        parent,
        hypothesis_id=child_id,
        object_trajectories=tuple(trajectories),
        unresolved_object_observation_ids=tuple(sorted(unresolved_objects)),
        limitations=tuple(parent.limitations) + (marker,),
    )
    before = _objective(
        hypothesis=parent,
        proposal=proposal,
        residual_packet=residual_packet,
        video=video,
        step6_config=step6_config,
        config=config,
    )
    after = _objective(
        hypothesis=child,
        proposal=proposal,
        residual_packet=residual_packet,
        video=video,
        step6_config=step6_config,
        config=config,
    )
    return (
        LocalReestimationCandidate(
            candidate_id=_candidate_id(proposal.proposal_id, 1, "instantiated"),
            proposal_id=proposal.proposal_id,
            parent_hypothesis_id=parent.hypothesis_id,
            operator=proposal.operator,
            status="instantiated",
            child_hypothesis=child,
            discrete_changes=tuple(changes),
            objective_before=before.terms,
            objective_after=after.terms,
            optimized_residual_ids=_optimized_source_ids(proposal),
            excluded_check_residual_ids=_excluded_check_ids(proposal),
            solver_method="mark_occluded_v1",
            solver_iterations=1,
            boundary_preserved=True,
            parameter_bounds_satisfied=True,
            compute_budget_honored=True,
            self_consistency_only=not after.has_fit_evidence,
            limitations=(
                "observations were preserved and no missing values were imputed",
                "candidate_requires_step9_acceptance",
            ),
        ),
    )


def _instantiate_proposal(
    *, parent, proposal, residual_packet, video, step6_config, config,
) -> ProposalReestimationResult:
    if proposal.status == "leave_unresolved" or proposal.operator == RepairOperator.LEAVE_UNRESOLVED:
        candidates = (
            _audit_candidate(
                proposal=proposal,
                status="unresolved",
                method="no_solver",
                limitations=("Step 7 explicitly left this conflict unresolved",),
            ),
        )
        status = "unresolved"
    elif (
        proposal.compute_budget.maximum_solver_iterations < 1
        or proposal.compute_budget.maximum_wall_time_seconds <= 0.0
    ):
        candidates = (
            _audit_candidate(
                proposal=proposal,
                status="unsupported",
                method="compute_budget_guard_v1",
                limitations=(
                    "proposal budget does not permit one bounded solver iteration",
                ),
            ),
        )
        status = "unsupported"
    elif proposal.operator == RepairOperator.REFIT_LOCAL_DYNAMICS:
        candidates = _refit_candidates(
            parent=parent,
            proposal=proposal,
            residual_packet=residual_packet,
            video=video,
            step6_config=step6_config,
            config=config,
        )
        status = (
            "candidates_generated"
            if any(row.status == "instantiated" for row in candidates)
            else "no_change"
        )
    elif proposal.operator == RepairOperator.ADJUST_PROCESS_NOISE:
        candidates = _adjust_noise_candidates(
            parent=parent,
            proposal=proposal,
            residual_packet=residual_packet,
            video=video,
            step6_config=step6_config,
            config=config,
        )
        status = (
            "candidates_generated"
            if any(row.status == "instantiated" for row in candidates)
            else "no_change"
        )
    elif proposal.operator == RepairOperator.MARK_UNOBSERVABLE:
        candidates = _mark_unobservable_candidate(
            parent=parent,
            proposal=proposal,
            residual_packet=residual_packet,
            video=video,
            step6_config=step6_config,
            config=config,
        )
        status = (
            "candidates_generated"
            if any(row.status == "instantiated" for row in candidates)
            else "no_change"
        )
    elif proposal.operator == RepairOperator.MARK_OCCLUDED:
        candidates = _mark_occluded_candidate(
            parent=parent,
            proposal=proposal,
            residual_packet=residual_packet,
            video=video,
            step6_config=step6_config,
            config=config,
        )
        if any(row.status == "instantiated" for row in candidates):
            status = "candidates_generated"
        elif any(row.status == "unresolved" for row in candidates):
            status = "unresolved"
        else:
            status = "no_change"
    else:
        candidates = (
            _audit_candidate(
                proposal=proposal,
                status="unsupported",
                method="unsupported_operator_audit_v1",
                limitations=(
                    "operator requires a mutable tracking/candidate-bank child contract",
                    "no world-state value was invented",
                ),
            ),
        )
        status = "unsupported"
    return ProposalReestimationResult(
        proposal_id=proposal.proposal_id,
        parent_hypothesis_id=parent.hypothesis_id,
        operator=proposal.operator,
        status=status,
        candidates=candidates,
    )


def _video_manifest(
    *, loaded: _LoadedStep7, video: _LoadedVideo, repair_store_ref,
    step6_config, config, config_sha256,
) -> VideoLocalReestimationManifest:
    parents = {
        row.hypothesis_id: row for row in video.world.initial_beam.hypotheses
    }
    residual_packets = {
        row.hypothesis_id: row for row in video.residual.packets
    }
    packets = []
    for diagnosis_packet in video.repair.packets:
        parent = parents[diagnosis_packet.hypothesis_id]
        residual_packet = residual_packets[parent.hypothesis_id]
        results = tuple(
            _instantiate_proposal(
                parent=parent,
                proposal=proposal,
                residual_packet=residual_packet,
                video=video,
                step6_config=step6_config,
                config=config,
            )
            for proposal in diagnosis_packet.proposals
        )
        packets.append(
            HypothesisReestimationPacket(
                packet_id=f"reestimation-packet:{parent.hypothesis_id}",
                parent_hypothesis_id=parent.hypothesis_id,
                parent_hypothesis_rank=parent.rank,
                proposal_results=results,
                child_candidate_count=sum(
                    candidate.status == "instantiated"
                    for row in results
                    for candidate in row.candidates
                ),
            )
        )
    results = [row for packet in packets for row in packet.proposal_results]
    proposals = [row for packet in video.repair.packets for row in packet.proposals]
    candidates = [
        candidate
        for result in results
        for candidate in result.candidates
    ]
    check_violations = sum(
        bool(
            set(candidate.optimized_residual_ids)
            & set(candidate.excluded_check_residual_ids)
        )
        for candidate in candidates
    )
    guard_violations = sum(
        candidate.status == "instantiated"
        and not (
            candidate.boundary_preserved
            and candidate.parameter_bounds_satisfied
            and candidate.compute_budget_honored
        )
        for candidate in candidates
    )
    if check_violations or guard_violations:
        raise RuntimeError("Step 8 candidate validation failed before publication")
    input_snapshot = Step8InputSnapshot(
        source_step7_relative_root=loaded.stage_root.relative_to(loaded.run_root).as_posix(),
        repair_proposal_store=_link(ArtifactOwner.STEP7_DIAGNOSIS, repair_store_ref),
        video_repair_proposal_manifest=_link(
            ArtifactOwner.STEP7_DIAGNOSIS, video.repair_reference
        ),
        source_step6_relative_root=video.repair.input_snapshot.source_step6_relative_root,
        residual_store=video.repair.input_snapshot.residual_store,
        video_residual_manifest=video.repair.input_snapshot.video_residual_manifest,
        source_step5_relative_root=video.repair.input_snapshot.source_step5_relative_root,
        world_state_store=video.residual.input_snapshot.world_state_store,
        video_world_state_manifest=video.repair.input_snapshot.video_world_state_manifest,
        source_step4_relative_root=video.residual.input_snapshot.source_step4_relative_root,
        video_geometry_manifest=video.residual.input_snapshot.video_geometry_manifest,
        source_step3_relative_root=video.repair.input_snapshot.source_step3_relative_root,
        video_tracking_manifest=video.repair.input_snapshot.video_tracking_manifest,
    )
    return VideoLocalReestimationManifest(
        run_id=video.repair.run_id,
        video_id=video.repair.video_id,
        source_repair_manifest_sha256=video.repair_reference.sha256,
        config_sha256=config_sha256,
        canonical_fps=video.repair.canonical_fps,
        image_size=video.repair.image_size,
        frame_count=video.repair.frame_count,
        input_snapshot=input_snapshot,
        packets=tuple(packets),
        validation=Step8ValidationSummary(
            input_parent_count=len(packets),
            input_proposal_count=len(proposals),
            input_ready_proposal_count=sum(
                row.status == "ready" for row in proposals
            ),
            input_unresolved_proposal_count=sum(
                row.status == "leave_unresolved" for row in proposals
            ),
            generated_proposal_count=sum(
                row.status == "candidates_generated" for row in results
            ),
            generated_candidate_count=sum(packet.child_candidate_count for packet in packets),
            no_change_proposal_count=sum(row.status == "no_change" for row in results),
            unsupported_proposal_count=sum(row.status == "unsupported" for row in results),
            output_unresolved_proposal_count=sum(
                row.status == "unresolved" for row in results
            ),
            overall_pass=True,
        ),
        tool_versions=(ToolVersion(name="numpy", version=np.__version__),),
    )


def run_step8(
    *,
    repair_proposal_store_path: Path | str,
    maximum_candidates_per_proposal: int = 3,
    fit_objective_weight: float = 1.0,
    physics_objective_weight: float = 0.25,
    minimum_position_std: float = 1e-6,
) -> Step8Result:
    """Instantiate bounded Step 7 proposals without selecting a winner."""

    loaded = _load_step7(repair_proposal_store_path)
    config = Step8ConfigSnapshot(
        maximum_candidates_per_proposal=maximum_candidates_per_proposal,
        fit_objective_weight=fit_objective_weight,
        physics_objective_weight=physics_objective_weight,
        minimum_position_std=minimum_position_std,
    )
    config_sha256 = hash_payload(config)
    source_sha256 = sha256_file(loaded.store_path)
    stage_root = (
        loaded.run_root
        / "08_local_reestimation"
        / f"input_{source_sha256[:16]}"
        / f"config_{config_sha256[:16]}"
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    repair_store_ref = _file_reference(
        path=loaded.store_path,
        stage_root=loaded.stage_root,
        artifact_id=f"repair-proposal-store:{loaded.store.run_id}",
    )
    manifests = []
    references = []
    for video in loaded.videos:
        residual_store_path = (
            loaded.run_root
            / video.repair.input_snapshot.source_step6_relative_root
            / video.repair.input_snapshot.residual_store.artifact.relative_path
        )
        residual_store = read_contract(residual_store_path, ResidualStore)
        parent_sha256_before = sha256_file(video.world_path)
        manifest = _video_manifest(
            loaded=loaded,
            video=video,
            repair_store_ref=repair_store_ref,
            step6_config=residual_store.config,
            config=config,
            config_sha256=config_sha256,
        )
        if sha256_file(video.world_path) != parent_sha256_before:
            raise RuntimeError("Step 8 mutated the immutable Step 5 parent manifest")
        relative_path = Path("videos") / f"{video.repair.video_id}.reestimation.json"
        path = stage_root / relative_path
        sha256, byte_size = write_contract(path, manifest)
        references.append(
            ArtifactRef(
                artifact_id=f"video-local-reestimation:{video.repair.video_id}",
                relative_path=relative_path.as_posix(),
                sha256=sha256,
                byte_size=byte_size,
                media_type="application/vnd.cauvid.local-reestimation+json",
                coordinate_space=None,
            )
        )
        manifests.append(manifest)
    store = LocalReestimationStore(
        run_id=loaded.store.run_id,
        source_repair_proposal_store_sha256=source_sha256,
        config=config,
        config_sha256=config_sha256,
        video_ids=loaded.store.video_ids,
        video_local_reestimations=tuple(references),
    )
    store_path_out = stage_root / "local_reestimation_store.json"
    write_contract(store_path_out, store)
    return Step8Result(
        store=store,
        video_manifests=tuple(manifests),
        stage_root=stage_root,
        store_path=store_path_out,
    )


__all__ = ["Step8Result", "run_step8"]
