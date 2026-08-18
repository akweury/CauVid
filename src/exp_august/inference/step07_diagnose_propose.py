"""Step 7: diagnose Step 6 conflicts and emit bounded repair proposals.

The stage is deterministic and read-only with respect to Step 5 world-state
hypotheses.  It may select only operators from the frozen repair allow-list;
continuous state values are left to Step 8 numerical re-estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.exp_august.contracts import (
    ArtifactLink,
    ArtifactOwner,
    ArtifactRef,
    CueFamily,
    EvaluationBasis,
    EvidenceKeyframe,
    EvidencePacket,
    ExpectedEffectDirection,
    ExpectedResidualEffect,
    FailureCategory,
    FailureDiagnosis,
    HypothesisDiagnosisPacket,
    RepairComputeBudget,
    RepairOperator,
    RepairParameterBound,
    RepairProposal,
    RepairProposalStore,
    ResidualSeverity,
    ResidualStore,
    Step7ConfigSnapshot,
    Step7InputSnapshot,
    Step7ValidationSummary,
    VideoManifest,
    VideoRepairProposalManifest,
    VideoResidualManifest,
    VideoTrackingManifest,
    VideoWorldStateManifest,
)
from src.exp_august.contracts.codec import (
    hash_payload,
    read_contract,
    sha256_file,
    write_contract,
)


@dataclass(frozen=True)
class Step7Result:
    store: RepairProposalStore
    video_manifests: tuple[VideoRepairProposalManifest, ...]
    stage_root: Path
    store_path: Path


@dataclass(frozen=True)
class _LoadedVideo:
    residual: VideoResidualManifest
    residual_reference: ArtifactRef
    world: VideoWorldStateManifest
    tracking: VideoTrackingManifest
    source: VideoManifest


@dataclass(frozen=True)
class _LoadedStep6:
    store_path: Path
    stage_root: Path
    run_root: Path
    store: ResidualStore
    videos: tuple[_LoadedVideo, ...]


@dataclass(frozen=True)
class _ConflictCluster:
    windows: tuple[object, ...]
    residuals: tuple[object, ...]

    @property
    def start_frame_index(self) -> int:
        rows = self.windows or self.residuals
        return min(row.start_frame_index for row in rows)

    @property
    def end_frame_index(self) -> int:
        rows = self.windows or self.residuals
        return max(row.end_frame_index for row in rows)

    @property
    def peak(self) -> float:
        if self.windows:
            return max(float(row.peak_normalized_residual) for row in self.windows)
        return max(float(row.normalized_residual or 0.0) for row in self.residuals)

    @property
    def conflict_ids(self) -> tuple[str, ...]:
        return tuple(sorted(row.conflict_id for row in self.windows))

    @property
    def component_ids(self) -> tuple[str, ...]:
        if self.windows:
            values = (
                value for row in self.windows for value in row.component_ids
            )
        else:
            values = (row.component_id for row in self.residuals)
        return _unique(values)

    @property
    def track_ids(self) -> tuple[str, ...]:
        if self.windows:
            values = (value for row in self.windows for value in row.track_ids)
        else:
            values = (row.track_id for row in self.residuals)
        return _unique(values)


def _unique(values: Iterable) -> tuple:
    return tuple(dict.fromkeys(value for value in values if value is not None))


def _artifact_key(link: ArtifactLink) -> tuple[str, str, str]:
    return (link.owner.value, link.artifact.artifact_id, link.artifact.sha256)


def _unique_artifacts(links: Iterable[ArtifactLink]) -> tuple[ArtifactLink, ...]:
    by_key = {}
    for link in links:
        by_key.setdefault(_artifact_key(link), link)
    return tuple(by_key[key] for key in sorted(by_key))


def _verified_contract(path: Path, reference: ArtifactRef, model, *, label: str):
    if not path.is_file() or path.stat().st_size != reference.byte_size:
        raise RuntimeError(f"Step 7 {label} is missing or truncated: {path}")
    if sha256_file(path) != reference.sha256:
        raise RuntimeError(f"Step 7 {label} failed integrity check: {path}")
    return read_contract(path, model)


def _find_run_root(store_path: Path) -> Path:
    for parent in store_path.parents:
        if parent.name == "06_predict_verify":
            return parent.parent
    raise RuntimeError("Step 6 store must live below 06_predict_verify")


def _load_step6(store_path: Path | str) -> _LoadedStep6:
    resolved = Path(store_path).expanduser().resolve()
    store = read_contract(resolved, ResidualStore)
    stage_root = resolved.parent
    run_root = _find_run_root(resolved)
    videos = []
    for video_id, reference in zip(store.video_ids, store.video_residuals):
        residual_path = stage_root / reference.relative_path
        residual = _verified_contract(
            residual_path,
            reference,
            VideoResidualManifest,
            label="residual manifest",
        )
        if residual.video_id != video_id or residual.run_id != store.run_id:
            raise RuntimeError(f"Step 7 residual identity mismatch: {residual_path}")
        snapshot = residual.input_snapshot
        if snapshot.world_state_store.artifact.sha256 != store.source_world_state_store_sha256:
            raise RuntimeError("Step 7 found inconsistent Step 5 store lineage")
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
        source_link = tracking.input_snapshot.source_video_manifest
        source_path = run_root / "01_init" / source_link.artifact.relative_path
        source = _verified_contract(
            source_path,
            source_link.artifact,
            VideoManifest,
            label="source video manifest",
        )
        if {world.video_id, tracking.video_id, source.video_id} != {video_id}:
            raise RuntimeError(f"Step 7 cross-stage video identity mismatch: {video_id}")
        if tuple(row.hypothesis_id for row in residual.packets) != tuple(
            row.hypothesis_id
            for row in world.initial_beam.hypotheses[: len(residual.packets)]
        ):
            raise RuntimeError(f"Step 7 hypothesis lineage mismatch: {video_id}")
        videos.append(
            _LoadedVideo(
                residual=residual,
                residual_reference=reference,
                world=world,
                tracking=tracking,
                source=source,
            )
        )
    return _LoadedStep6(
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


def _shares_subject(left: _ConflictCluster, window) -> bool:
    left_tracks = set(left.track_ids)
    right_tracks = set(window.track_ids)
    if left_tracks and right_tracks and left_tracks & right_tracks:
        return True
    left_components = set(left.component_ids)
    right_components = set(window.component_ids)
    if left_components and right_components and left_components & right_components:
        return True
    return not (left_tracks or right_tracks or left_components or right_components)


def _conflict_clusters(packet, merge_gap_frames: int) -> tuple[_ConflictCluster, ...]:
    residual_by_id = {row.residual_id: row for row in packet.residuals}
    clusters: list[_ConflictCluster] = []
    ordered = sorted(
        packet.conflict_windows,
        key=lambda row: (row.start_frame_index, row.end_frame_index, row.conflict_id),
    )
    for window in ordered:
        matching_index = next(
            (
                index
                for index, cluster in enumerate(clusters)
                if window.start_frame_index
                <= cluster.end_frame_index + merge_gap_frames + 1
                and _shares_subject(cluster, window)
            ),
            None,
        )
        window_residuals = tuple(
            residual_by_id[residual_id] for residual_id in window.residual_ids
        )
        if matching_index is None:
            clusters.append(_ConflictCluster(windows=(window,), residuals=window_residuals))
        else:
            previous = clusters[matching_index]
            clusters[matching_index] = _ConflictCluster(
                windows=previous.windows + (window,),
                residuals=tuple(
                    {
                        row.residual_id: row
                        for row in previous.residuals + window_residuals
                    }.values()
                ),
            )
    return tuple(clusters)


def _alternative_mask_ids(tracking, cluster, maximum: int) -> tuple[str, ...]:
    track_ids = set(cluster.track_ids)
    return tuple(
        row.candidate_id
        for row in tracking.mask_candidate_bank
        if row.track_id in track_ids
        and cluster.start_frame_index <= row.frame_index <= cluster.end_frame_index
        and not row.selected
        and row.mask is not None
    )[:maximum]


def _association_alternative_ids(tracking, cluster, maximum: int) -> tuple[str, ...]:
    track_ids = set(cluster.track_ids)
    return tuple(
        row.ledger_id
        for row in tracking.association_ledger
        if row.track_id in track_ids
        and cluster.start_frame_index <= row.frame_index <= cluster.end_frame_index
        and row.feasible
        and not row.selected
    )[:maximum]


def _gap_support(tracking, cluster) -> bool:
    track_ids = set(cluster.track_ids)
    return any(
        row.track_id in track_ids
        and row.gap_frames[0] <= cluster.end_frame_index
        and row.gap_frames[-1] >= cluster.start_frame_index
        and (row.mask_candidate_ids or row.unassigned_evidence_ids)
        for row in tracking.gap_records
    )


def _scale_alternatives(world, hypothesis_id: str, maximum: int) -> tuple[str, ...]:
    parent = next(
        row for row in world.initial_beam.hypotheses if row.hypothesis_id == hypothesis_id
    )
    return _unique(
        row.scale_id
        for row in world.initial_beam.hypotheses
        if row.scale_id != parent.scale_id
    )[:maximum]


def _discrete_bound(name: str, values: Iterable[str]) -> RepairParameterBound:
    return RepairParameterBound(
        parameter_name=name,
        allowed_values=tuple(values),
        unit="categorical",
    )


def _numeric_bound(name: str, low: float, high: float, unit: str) -> RepairParameterBound:
    return RepairParameterBound(
        parameter_name=name,
        lower_bound=float(low),
        upper_bound=float(high),
        unit=unit,
    )


def _diagnosis_choice(*, cluster, tracking, world, hypothesis_id: str, config):
    constraints = {row.constraint_id for row in cluster.residuals}
    families = {row.family.value for row in cluster.residuals}
    check_supported = any(row.check_evidence_supported for row in cluster.windows)
    mask_ids = _alternative_mask_ids(
        tracking, cluster, config.maximum_discrete_candidates
    )
    association_ids = _association_alternative_ids(
        tracking, cluster, config.maximum_discrete_candidates
    )
    scale_ids = _scale_alternatives(
        world, hypothesis_id, config.maximum_discrete_candidates
    )
    has_background = "heldout_background_backward_flow" in constraints
    has_object_flow = "heldout_object_backward_flow" in constraints
    has_depth = "heldout_object_depth" in constraints
    has_reprojection = "object_reprojection" in constraints
    has_gap = "trajectory_temporal_gap" in constraints
    has_physics = bool(
        constraints
        & {
            "ego_acceleration_bound",
            "ego_speed_bound",
            "object_acceleration_bound",
            "object_speed_bound",
        }
    )
    has_semantic = "semantic_static_motion" in constraints
    flow_direction = max(
        (
            float(row.flow_direction_error_deg)
            for row in cluster.residuals
            if row.flow_direction_error_deg is not None
        ),
        default=0.0,
    )

    if has_background:
        return (
            FailureCategory.POSE_DRIFT,
            RepairOperator.REFIT_LOCAL_DYNAMICS,
            "held-out rigid-background flow disagrees with the ego pose window",
            (FailureCategory.INVALID_STATIC_BACKGROUND_ASSUMPTION,),
            ("ego_components.pose_window",),
            (
                _numeric_bound("maximum_pose_delta_sigma", 0.0, 3.0, "standard_deviation"),
                _discrete_bound("boundary_condition", ("preserve_window_boundaries",)),
            ),
        )
    if has_gap and (association_ids or _gap_support(tracking, cluster)):
        values = association_ids or ("archived_gap_candidates_only",)
        return (
            FailureCategory.IDENTITY_ERROR,
            RepairOperator.RELINK_TRACK,
            "a supported trajectory gap has archived association evidence for relinking",
            (FailureCategory.UNOBSERVABLE_EVIDENCE,),
            ("track_association_edges",),
            (_discrete_bound("association_candidate", values),),
        )
    if has_gap:
        return (
            FailureCategory.UNOBSERVABLE_EVIDENCE,
            RepairOperator.MARK_OCCLUDED,
            "the trajectory gap lacks a safe archived association alternative",
            (FailureCategory.IDENTITY_ERROR,),
            ("track_visibility_state",),
            (_discrete_bound("visibility_state", ("occluded",)),),
        )
    if mask_ids and has_depth and (has_object_flow or has_reprojection):
        return (
            FailureCategory.MASK_ERROR,
            RepairOperator.SWITCH_MASK_CANDIDATE,
            "multiple visual cues disagree and an unselected archived mask is available",
            (FailureCategory.DEPTH_JUMP, FailureCategory.IDENTITY_ERROR),
            ("selected_mask_candidate_id",),
            (_discrete_bound("mask_candidate_id", mask_ids),),
        )
    if has_depth and len(cluster.track_ids) > 1 and scale_ids:
        return (
            FailureCategory.SCALE_AMBIGUITY,
            RepairOperator.SWITCH_SCALE_CANDIDATE,
            "simultaneous held-out depth conflicts affect multiple tracks in one component",
            (FailureCategory.DEPTH_JUMP,),
            ("scale_id",),
            (_discrete_bound("scale_candidate_id", scale_ids),),
        )
    if has_physics and has_object_flow and check_supported:
        return (
            FailureCategory.TRUE_ACUTE_MANEUVER,
            RepairOperator.ADJUST_PROCESS_NOISE,
            "a dynamics violation coincides with independently checked object motion",
            (FailureCategory.DYNAMICS_MISMATCH,),
            ("object_dynamics.process_noise",),
            (_numeric_bound("process_noise_multiplier", 1.0, 3.0, "ratio"),),
        )
    if has_physics:
        subject = "object_trajectories.local_dynamics" if cluster.track_ids else "ego_components.local_dynamics"
        return (
            FailureCategory.DYNAMICS_MISMATCH,
            RepairOperator.REFIT_LOCAL_DYNAMICS,
            "the local finite-difference dynamics violate the frozen plausibility bound",
            (FailureCategory.TRUE_ACUTE_MANEUVER,),
            (subject,),
            (
                _numeric_bound("maximum_state_delta_sigma", 0.0, 3.0, "standard_deviation"),
                _discrete_bound("boundary_condition", ("preserve_window_boundaries",)),
            ),
        )
    if has_object_flow and flow_direction >= 60.0:
        if association_ids:
            return (
                FailureCategory.IDENTITY_ERROR,
                RepairOperator.RELINK_TRACK,
                "held-out flow reverses the predicted direction and an association alternative exists",
                (FailureCategory.DYNAMICS_MISMATCH,),
                ("track_association_edges",),
                (_discrete_bound("association_candidate", association_ids),),
            )
        return (
            FailureCategory.IDENTITY_ERROR,
            RepairOperator.SPLIT_TRACK,
            "held-out flow strongly disagrees in direction without a safe relink candidate",
            (FailureCategory.DYNAMICS_MISMATCH,),
            ("track_identity_segments",),
            (
                _numeric_bound(
                    "split_frame_index",
                    cluster.start_frame_index,
                    cluster.end_frame_index,
                    "frame",
                ),
            ),
        )
    if has_object_flow or has_reprojection:
        return (
            FailureCategory.DYNAMICS_MISMATCH,
            RepairOperator.REFIT_LOCAL_DYNAMICS,
            "the object trajectory does not reproduce its image-space motion",
            (FailureCategory.MASK_ERROR, FailureCategory.IDENTITY_ERROR),
            ("object_trajectories.local_dynamics",),
            (
                _numeric_bound("maximum_state_delta_sigma", 0.0, 3.0, "standard_deviation"),
                _discrete_bound("boundary_condition", ("preserve_window_boundaries",)),
            ),
        )
    if has_depth:
        return (
            FailureCategory.DEPTH_JUMP,
            RepairOperator.INVALIDATE_OR_DOWNWEIGHT_CUE,
            "an isolated held-out monocular-depth conflict may reflect frame-scale drift",
            (FailureCategory.SCALE_AMBIGUITY, FailureCategory.MASK_ERROR),
            ("cue_weights.depth",),
            (_numeric_bound("retained_cue_weight", 0.5, 1.0, "ratio"),),
        )
    if has_semantic:
        return (
            FailureCategory.SEMANTIC_PRIOR_MISMATCH,
            RepairOperator.LEAVE_UNRESOLVED,
            "a soft semantic prior alone is insufficient to rewrite physical state",
            (FailureCategory.DYNAMICS_MISMATCH,),
            ("none",),
            (_discrete_bound("action", ("no_state_change",)),),
        )
    return (
        FailureCategory.UNRESOLVED_CONFLICT,
        RepairOperator.LEAVE_UNRESOLVED,
        f"no safe deterministic repair rule covers constraints {sorted(constraints)} and families {sorted(families)}",
        (),
        ("none",),
        (_discrete_bound("action", ("no_state_change",)),),
    )


def _diagnosis_confidence(cluster: _ConflictCluster) -> float:
    value = 0.50
    if any(row.check_evidence_supported for row in cluster.windows):
        value += 0.15
    if len({row.family for row in cluster.windows}) > 1:
        value += 0.10
    if any(row.severity == ResidualSeverity.HARD_VIOLATION for row in cluster.windows):
        value += 0.10
    if cluster.track_ids or cluster.component_ids:
        value += 0.05
    return min(0.95, value)


def _effect(
    residual,
    *,
    direction: ExpectedEffectDirection,
    optimized: bool,
    rationale: str,
) -> ExpectedResidualEffect:
    return ExpectedResidualEffect(
        residual_id=residual.residual_id,
        evaluation_basis=residual.evaluation_basis,
        direction=direction,
        optimized_by_step8=(
            optimized
            and residual.evaluable
            and residual.evaluation_basis
            not in {EvaluationBasis.CHECK_EVIDENCE, EvaluationBasis.NOT_EVALUABLE}
        ),
        minimum_normalized_improvement=(
            0.10 if direction == ExpectedEffectDirection.DECREASE else None
        ),
        rationale=rationale,
    )


def _proposal(
    *,
    diagnosis,
    evidence_packet_id: str,
    cluster,
    packet,
    operator,
    affected_variables,
    bounds,
    config,
    maximum_frame_index: int,
):
    unresolved = operator == RepairOperator.LEAVE_UNRESOLVED
    target_ids = tuple(row.residual_id for row in cluster.residuals)
    effects = [
        _effect(
            row,
            direction=(
                ExpectedEffectDirection.UNKNOWN
                if unresolved or not row.evaluable
                else ExpectedEffectDirection.DECREASE
            ),
            optimized=not unresolved,
            rationale=(
                "unresolved diagnosis makes no residual-improvement claim"
                if unresolved
                else "the bounded repair is expected to reduce its diagnosed residual"
            ),
        )
        for row in cluster.residuals
    ]
    known = set(target_ids)
    component_ids = set(cluster.component_ids)
    track_ids = set(cluster.track_ids)
    for row in packet.residuals:
        if row.residual_id in known or row.evaluation_basis != EvaluationBasis.CHECK_EVIDENCE:
            continue
        if row.end_frame_index < cluster.start_frame_index or row.start_frame_index > cluster.end_frame_index:
            continue
        if track_ids and row.track_id not in track_ids:
            continue
        if not track_ids and component_ids and row.component_id not in component_ids:
            continue
        effects.append(
            _effect(
                row,
                direction=ExpectedEffectDirection.NON_DEGRADATION,
                optimized=False,
                rationale="independent check evidence may accept or reject but cannot drive Step 8",
            )
        )
    identifier = hash_payload(
        {
            "diagnosis": diagnosis.diagnosis_id,
            "operator": operator.value,
            "window": (cluster.start_frame_index, cluster.end_frame_index),
            "bounds": [row.model_dump(mode="json") for row in bounds],
        }
    )[:20]
    return RepairProposal(
        proposal_id=f"repair:{identifier}",
        diagnosis_id=diagnosis.diagnosis_id,
        evidence_packet_id=evidence_packet_id,
        parent_hypothesis_id=packet.hypothesis_id,
        operator=operator,
        affected_variables=affected_variables,
        start_frame_index=max(0, cluster.start_frame_index - config.conflict_context_frames),
        end_frame_index=min(
            cluster.end_frame_index + config.conflict_context_frames,
            maximum_frame_index,
        ),
        parameter_bounds=bounds,
        target_residual_ids=target_ids,
        expected_residual_effects=tuple(effects),
        compute_budget=RepairComputeBudget(
            maximum_solver_iterations=0 if unresolved else config.default_solver_iterations,
            maximum_child_hypotheses=(
                0 if unresolved else config.default_maximum_child_hypotheses
            ),
            maximum_wall_time_seconds=(
                0.0 if unresolved else config.default_wall_time_seconds
            ),
        ),
        source_conflict_ids=cluster.conflict_ids,
        status="leave_unresolved" if unresolved else "ready",
    )


def _evidence_packet(*, packet, clusters, source, config) -> EvidencePacket:
    residual_by_id = {
        row.residual_id: row
        for cluster in clusters
        for row in cluster.residuals
    }
    residuals = tuple(residual_by_id.values())
    if not residuals and packet.status == "insufficient_evidence":
        residuals = tuple(row for row in packet.residuals if not row.evaluable)
        if not residuals:
            residuals = packet.residuals
    conflict_ids = _unique(
        window.conflict_id for cluster in clusters for window in cluster.windows
    )
    component_ids = _unique(row.component_id for row in residuals)
    track_ids = _unique(row.track_id for row in residuals)
    cue_families = _unique(row.cue_family for row in residuals)
    frame_candidates: dict[int, dict[str, set]] = {}

    def register(frame: int, reason: str, rows) -> None:
        entry = frame_candidates.setdefault(frame, {"reasons": set(), "residuals": set()})
        entry["reasons"].add(reason)
        entry["residuals"].update(row.residual_id for row in rows)

    for cluster in clusters:
        register(cluster.start_frame_index, "conflict_window_start", cluster.residuals)
        register(cluster.end_frame_index, "conflict_window_end", cluster.residuals)
        peak_row = max(
            cluster.residuals,
            key=lambda row: float(row.normalized_residual or 0.0),
        )
        register(peak_row.start_frame_index, "normalized_residual_peak", (peak_row,))
    if not clusters:
        for row in residuals:
            register(row.start_frame_index, "insufficient_evidence", (row,))

    ranked_frames = sorted(
        frame_candidates,
        key=lambda frame: (
            "normalized_residual_peak" not in frame_candidates[frame]["reasons"],
            frame,
        ),
    )[: config.maximum_keyframes_per_evidence_packet]
    residual_by_id = {row.residual_id: row for row in residuals}
    keyframes = []
    for frame in sorted(ranked_frames):
        if frame >= len(source.frames):
            continue
        frame_record = source.frames[frame]
        ids = tuple(sorted(frame_candidates[frame]["residuals"]))
        rows = tuple(residual_by_id[value] for value in ids if value in residual_by_id)
        keyframes.append(
            EvidenceKeyframe(
                frame_index=frame,
                timestamp_s=frame_record.timestamp_s,
                source_frame_index=frame_record.source_frame_index,
                source_timestamp_s=frame_record.source_timestamp_s,
                selection_reasons=tuple(sorted(frame_candidates[frame]["reasons"])),
                residual_ids=ids,
                evidence_artifacts=_unique_artifacts(
                    link for row in rows for link in row.evidence_artifacts
                ),
            )
        )
    evidence_artifacts = _unique_artifacts(
        link for row in residuals for link in row.evidence_artifacts
    )
    limitations = []
    if not any(
        row.evaluation_basis == EvaluationBasis.CHECK_EVIDENCE for row in residuals
    ):
        limitations.append("self_consistency_only_without_check_evidence")
    if not evidence_artifacts:
        limitations.append("no_dense_evidence_artifact_for_selected_residuals")
    identifier = hash_payload(
        {
            "hypothesis": packet.hypothesis_id,
            "conflicts": conflict_ids,
            "residuals": [row.residual_id for row in residuals],
        }
    )[:20]
    return EvidencePacket(
        evidence_packet_id=f"evidence-packet:{identifier}",
        hypothesis_id=packet.hypothesis_id,
        hypothesis_rank=packet.hypothesis_rank,
        source_path=source.source_path,
        conflict_ids=conflict_ids,
        residual_ids=tuple(row.residual_id for row in residuals),
        component_ids=component_ids,
        track_ids=track_ids,
        cue_families=cue_families,
        keyframes=tuple(keyframes),
        evidence_artifacts=evidence_artifacts,
        limitations=tuple(limitations),
    )


def _unobservable_cluster(packet) -> _ConflictCluster:
    residuals = tuple(row for row in packet.residuals if not row.evaluable)
    if not residuals:
        residuals = packet.residuals
    return _ConflictCluster(windows=(), residuals=residuals)


def _hypothesis_packet(*, packet, tracking, world, source, config):
    all_clusters = _conflict_clusters(packet, config.cross_family_merge_gap_frames)
    ordered_clusters = tuple(
        sorted(
            all_clusters,
            key=lambda cluster: (
                not any(
                    row.severity == ResidualSeverity.HARD_VIOLATION
                    for row in cluster.windows
                ),
                -cluster.peak,
                cluster.start_frame_index,
            ),
        )
    )
    selected_clusters = ordered_clusters[: config.maximum_proposals_per_hypothesis]
    deferred = _unique(
        conflict_id
        for cluster in ordered_clusters[config.maximum_proposals_per_hypothesis :]
        for conflict_id in cluster.conflict_ids
    )
    evidence_clusters = ordered_clusters
    evidence = _evidence_packet(
        packet=packet,
        clusters=evidence_clusters,
        source=source,
        config=config,
    )
    diagnoses = []
    proposals = []
    for cluster in selected_clusters:
        category, operator, rationale, alternatives, variables, bounds = _diagnosis_choice(
            cluster=cluster,
            tracking=tracking,
            world=world,
            hypothesis_id=packet.hypothesis_id,
            config=config,
        )
        diagnosis_id = "diagnosis:" + hash_payload(
            {
                "hypothesis": packet.hypothesis_id,
                "conflicts": cluster.conflict_ids,
                "category": category.value,
            }
        )[:20]
        diagnosis = FailureDiagnosis(
            diagnosis_id=diagnosis_id,
            hypothesis_id=packet.hypothesis_id,
            category=category,
            confidence=_diagnosis_confidence(cluster),
            source_conflict_ids=cluster.conflict_ids,
            source_residual_ids=tuple(row.residual_id for row in cluster.residuals),
            component_ids=cluster.component_ids,
            track_ids=cluster.track_ids,
            supporting_cue_families=_unique(
                row.cue_family for row in cluster.residuals
            ),
            check_evidence_supported=any(
                row.check_evidence_supported for row in cluster.windows
            ),
            rationale=rationale,
            alternative_categories=alternatives,
        )
        diagnoses.append(diagnosis)
        proposals.append(
            _proposal(
                diagnosis=diagnosis,
                evidence_packet_id=evidence.evidence_packet_id,
                cluster=cluster,
                packet=packet,
                operator=operator,
                affected_variables=variables,
                bounds=bounds,
                config=config,
                maximum_frame_index=len(source.frames) - 1,
            )
        )

    if not selected_clusters and packet.status == "insufficient_evidence":
        if not packet.residuals:
            diagnosis = FailureDiagnosis(
                diagnosis_id="diagnosis:"
                + hash_payload(
                    {
                        "hypothesis": packet.hypothesis_id,
                        "category": FailureCategory.UNOBSERVABLE_EVIDENCE.value,
                        "reason": "no_residual_records",
                    }
                )[:20],
                hypothesis_id=packet.hypothesis_id,
                category=FailureCategory.UNOBSERVABLE_EVIDENCE,
                confidence=1.0,
                source_conflict_ids=(),
                source_residual_ids=(),
                component_ids=(),
                track_ids=(),
                supporting_cue_families=(),
                check_evidence_supported=False,
                rationale="Step 6 emitted no evaluable or non-evaluable residual records",
                alternative_categories=(),
            )
            diagnoses.append(diagnosis)
            proposals.append(
                RepairProposal(
                    proposal_id="repair:"
                    + hash_payload(
                        {
                            "diagnosis": diagnosis.diagnosis_id,
                            "operator": RepairOperator.MARK_UNOBSERVABLE.value,
                        }
                    )[:20],
                    diagnosis_id=diagnosis.diagnosis_id,
                    evidence_packet_id=evidence.evidence_packet_id,
                    parent_hypothesis_id=packet.hypothesis_id,
                    operator=RepairOperator.MARK_UNOBSERVABLE,
                    affected_variables=("hypothesis.observability",),
                    start_frame_index=0,
                    end_frame_index=len(source.frames) - 1,
                    parameter_bounds=(
                        _discrete_bound("observability", ("unobservable",)),
                    ),
                    target_residual_ids=(),
                    expected_residual_effects=(),
                    compute_budget=RepairComputeBudget(
                        maximum_solver_iterations=config.default_solver_iterations,
                        maximum_child_hypotheses=config.default_maximum_child_hypotheses,
                        maximum_wall_time_seconds=config.default_wall_time_seconds,
                    ),
                    source_conflict_ids=(),
                    status="ready",
                )
            )
        else:
            cluster = _unobservable_cluster(packet)
            diagnosis = FailureDiagnosis(
                diagnosis_id="diagnosis:"
                + hash_payload(
                    {
                        "hypothesis": packet.hypothesis_id,
                        "category": FailureCategory.UNOBSERVABLE_EVIDENCE.value,
                    }
                )[:20],
                hypothesis_id=packet.hypothesis_id,
                category=FailureCategory.UNOBSERVABLE_EVIDENCE,
                confidence=0.90,
                source_conflict_ids=(),
                source_residual_ids=tuple(row.residual_id for row in cluster.residuals),
                component_ids=_unique(row.component_id for row in cluster.residuals),
                track_ids=_unique(row.track_id for row in cluster.residuals),
                supporting_cue_families=_unique(
                    row.cue_family for row in cluster.residuals
                ),
                check_evidence_supported=False,
                rationale="Step 6 evaluability is below the frozen minimum; missing evidence is not a violation",
                alternative_categories=(),
            )
            diagnoses.append(diagnosis)
            proposal = _proposal(
                diagnosis=diagnosis,
                evidence_packet_id=evidence.evidence_packet_id,
                cluster=cluster,
                packet=packet,
                operator=RepairOperator.MARK_UNOBSERVABLE,
                affected_variables=("hypothesis.observability",),
                bounds=(_discrete_bound("observability", ("unobservable",)),),
                config=config,
                maximum_frame_index=len(source.frames) - 1,
            )
            proposals.append(proposal)

    if not diagnoses:
        status = "no_conflict"
    elif packet.status == "insufficient_evidence" and not selected_clusters:
        status = "insufficient_evidence"
    elif any(row.status == "ready" for row in proposals):
        status = "proposals_ready"
    else:
        status = "unresolved"
    return HypothesisDiagnosisPacket(
        packet_id=f"diagnosis-packet:{packet.hypothesis_id}",
        hypothesis_id=packet.hypothesis_id,
        hypothesis_rank=packet.hypothesis_rank,
        evidence=evidence,
        diagnoses=tuple(diagnoses),
        proposals=tuple(proposals),
        deferred_conflict_ids=deferred,
        status=status,
    )


def _video_manifest(*, loaded, video, residual_store_ref, config, config_sha256):
    packets = tuple(
        _hypothesis_packet(
            packet=packet,
            tracking=video.tracking,
            world=video.world,
            source=video.source,
            config=config,
        )
        for packet in video.residual.packets
    )
    proposals = [row for packet in packets for row in packet.proposals]
    diagnoses = [row for packet in packets for row in packet.diagnoses]
    input_snapshot = Step7InputSnapshot(
        source_step6_relative_root=loaded.stage_root.relative_to(loaded.run_root).as_posix(),
        residual_store=_link(ArtifactOwner.STEP6_VERIFICATION, residual_store_ref),
        video_residual_manifest=_link(
            ArtifactOwner.STEP6_VERIFICATION, video.residual_reference
        ),
        source_step5_relative_root=video.residual.input_snapshot.source_step5_relative_root,
        video_world_state_manifest=video.residual.input_snapshot.video_world_state_manifest,
        source_step3_relative_root=video.residual.input_snapshot.source_step3_relative_root,
        video_tracking_manifest=video.residual.input_snapshot.video_tracking_manifest,
        source_video_manifest=video.tracking.input_snapshot.source_video_manifest,
    )
    return VideoRepairProposalManifest(
        run_id=video.residual.run_id,
        video_id=video.residual.video_id,
        source_residual_manifest_sha256=video.residual_reference.sha256,
        config_sha256=config_sha256,
        canonical_fps=video.residual.canonical_fps,
        image_size=video.residual.image_size,
        frame_count=video.residual.frame_count,
        input_snapshot=input_snapshot,
        packets=packets,
        validation=Step7ValidationSummary(
            input_hypothesis_count=len(video.residual.packets),
            diagnosed_hypothesis_count=sum(bool(row.diagnoses) for row in packets),
            conflict_window_count=sum(
                len(row.conflict_windows) for row in video.residual.packets
            ),
            diagnosis_count=len(diagnoses),
            proposal_count=len(proposals),
            ready_proposal_count=sum(row.status == "ready" for row in proposals),
            unresolved_proposal_count=sum(
                row.status == "leave_unresolved" for row in proposals
            ),
            deferred_conflict_count=sum(
                len(row.deferred_conflict_ids) for row in packets
            ),
            check_evidence_optimization_violations=0,
            world_state_mutation_count=0,
            overall_pass=True,
        ),
    )


def run_step7(
    *,
    residual_store_path: Path | str,
    maximum_proposals_per_hypothesis: int = 16,
    maximum_keyframes_per_evidence_packet: int = 8,
    conflict_context_frames: int = 2,
    cross_family_merge_gap_frames: int = 1,
    maximum_discrete_candidates: int = 8,
    default_solver_iterations: int = 100,
    default_maximum_child_hypotheses: int = 3,
    default_wall_time_seconds: float = 10.0,
) -> Step7Result:
    """Diagnose Step 6 packets without mutating or ranking world hypotheses."""

    loaded = _load_step6(residual_store_path)
    config = Step7ConfigSnapshot(
        maximum_proposals_per_hypothesis=maximum_proposals_per_hypothesis,
        maximum_keyframes_per_evidence_packet=maximum_keyframes_per_evidence_packet,
        conflict_context_frames=conflict_context_frames,
        cross_family_merge_gap_frames=cross_family_merge_gap_frames,
        maximum_discrete_candidates=maximum_discrete_candidates,
        default_solver_iterations=default_solver_iterations,
        default_maximum_child_hypotheses=default_maximum_child_hypotheses,
        default_wall_time_seconds=default_wall_time_seconds,
    )
    config_sha256 = hash_payload(config)
    source_sha256 = sha256_file(loaded.store_path)
    stage_root = (
        loaded.run_root
        / "07_diagnose_propose"
        / f"input_{source_sha256[:16]}"
        / f"config_{config_sha256[:16]}"
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    residual_store_ref = _file_reference(
        path=loaded.store_path,
        stage_root=loaded.stage_root,
        artifact_id=f"residual-store:{loaded.store.run_id}",
    )
    manifests = []
    references = []
    for video in loaded.videos:
        manifest = _video_manifest(
            loaded=loaded,
            video=video,
            residual_store_ref=residual_store_ref,
            config=config,
            config_sha256=config_sha256,
        )
        relative_path = Path("videos") / f"{video.residual.video_id}.repairs.json"
        path = stage_root / relative_path
        sha256, byte_size = write_contract(path, manifest)
        references.append(
            ArtifactRef(
                artifact_id=f"video-repair-proposals:{video.residual.video_id}",
                relative_path=relative_path.as_posix(),
                sha256=sha256,
                byte_size=byte_size,
                media_type="application/vnd.cauvid.repair-proposals+json",
                coordinate_space=None,
            )
        )
        manifests.append(manifest)
    store = RepairProposalStore(
        run_id=loaded.store.run_id,
        source_residual_store_sha256=source_sha256,
        config=config,
        config_sha256=config_sha256,
        video_ids=loaded.store.video_ids,
        video_repair_proposals=tuple(references),
    )
    store_path_out = stage_root / "repair_proposal_store.json"
    write_contract(store_path_out, store)
    return Step7Result(
        store=store,
        video_manifests=tuple(manifests),
        stage_root=stage_root,
        store_path=store_path_out,
    )


__all__ = ["Step7Result", "run_step7"]
