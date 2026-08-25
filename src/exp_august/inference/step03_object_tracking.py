"""Target Step 3: deterministic, replayable multi-evidence object tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from src.exp_august.contracts import (
    ArtifactLink,
    ArtifactOwner,
    ArtifactRef,
    AssociationCueValue,
    AssociationDecision,
    AssociationGate,
    AssociationLedgerRow,
    BoundingBoxXYXY,
    CueFamily,
    DepthDescriptor,
    DetectionObservation,
    DetectionTier,
    EvidenceDisposition,
    EvidenceDispositionRecord,
    EvidenceRole,
    EvidenceRoleAssignment,
    EvidenceUsePlan,
    FlowDescriptor,
    FrameEvidence,
    GapEvidenceRecord,
    GapStatus,
    MaskCandidateRecord,
    MaskCandidateSource,
    MaskObservation,
    ModalityRetentionCount,
    NeuralEvidenceStore,
    ObjectMaskTrack,
    RetentionReport,
    Step3ConfigSnapshot,
    Step3InputSnapshot,
    SupportObservability,
    ToolVersion,
    TrackMarkerType,
    TrackObservation,
    TrackState,
    TrackStateMarker,
    TrackingStore,
    TransformRecord,
    UnassignedEvidenceRecord,
    VideoEvidenceManifest,
    VideoTrackingManifest,
)
from src.exp_august.inference.association import assign_one_to_one_matches
from src.exp_august.contracts.codec import (
    hash_payload,
    read_contract,
    sha256_file,
    write_contract,
)
from src.exp_august.inference.artifact_io import (
    read_image_artifact,
    write_mask_artifact,
)


@dataclass(frozen=True)
class Step3Result:
    store: TrackingStore
    video_manifests: tuple[VideoTrackingManifest, ...]
    stage_root: Path
    store_path: Path


@dataclass(frozen=True)
class _Instance:
    detection: DetectionObservation
    raw_mask: MaskObservation | None
    mask: MaskObservation | None
    mask_array: np.ndarray | None
    depth: DepthDescriptor | None


@dataclass
class _TrackWork:
    track_id: str
    primary_class: str
    state: TrackState
    first_observed_frame: int
    last_observed_frame: int
    missed_count: int
    observations: list[TrackObservation] = field(default_factory=list)
    markers: list[TrackStateMarker] = field(default_factory=list)
    last_mask: np.ndarray | None = None
    last_bbox: BoundingBoxXYXY | None = None
    last_depth: DepthDescriptor | None = None
    support_mask: np.ndarray | None = None
    support_mask_link: ArtifactLink | None = None
    support_frame: int | None = None
    open_gap_frames: list[int] = field(default_factory=list)
    open_gap_candidate_ids: list[str] = field(default_factory=list)
    open_gap_marker_ids: list[str] = field(default_factory=list)
    open_gap_context: list[ArtifactLink] = field(default_factory=list)


@dataclass(frozen=True)
class _CandidateEvaluation:
    track_id: str
    detection_id: str
    proposal_id: str | None
    track_age_frames: int
    cues: tuple[AssociationCueValue, ...]
    gates: tuple[AssociationGate, ...]
    feasible: bool
    total_score: float


@dataclass(frozen=True)
class _LoadedStep2:
    store_path: Path
    stage_root: Path
    run_root: Path
    store: NeuralEvidenceStore
    manifests: tuple[VideoEvidenceManifest, ...]
    manifest_refs: tuple[ArtifactRef, ...]


def _step2_link(reference: ArtifactRef) -> ArtifactLink:
    return ArtifactLink(owner=ArtifactOwner.STEP2_NEURAL_EVIDENCE, artifact=reference)


def _step1_link(reference: ArtifactRef) -> ArtifactLink:
    return ArtifactLink(owner=ArtifactOwner.STEP1_INIT, artifact=reference)


def _step3_link(reference: ArtifactRef) -> ArtifactLink:
    return ArtifactLink(owner=ArtifactOwner.STEP3_OBJECT_TRACKING, artifact=reference)


def _load_step2(store_path: Path | str) -> _LoadedStep2:
    resolved = Path(store_path).expanduser().resolve()
    store = read_contract(resolved, NeuralEvidenceStore)
    stage_root = resolved.parent
    manifests = []
    for video_id, reference in zip(store.video_ids, store.video_evidence):
        path = stage_root / reference.relative_path
        if not path.is_file() or sha256_file(path) != reference.sha256:
            raise RuntimeError(f"Step 2 evidence manifest failed integrity check: {path}")
        manifest = read_contract(path, VideoEvidenceManifest)
        if manifest.video_id != video_id or manifest.run_id != store.run_id:
            raise RuntimeError(f"Step 2 evidence identity mismatch: {path}")
        manifests.append(manifest)
    if stage_root.parent.name != "02_neural_evidence":
        raise RuntimeError("Step 2 store must live inside 02_neural_evidence/config_<hash>")
    return _LoadedStep2(
        store_path=resolved,
        stage_root=stage_root,
        run_root=stage_root.parent.parent,
        store=store,
        manifests=tuple(manifests),
        manifest_refs=store.video_evidence,
    )


def _artifact_key(cue: CueFamily, frame_index: int, artifact_id: str) -> str:
    return f"evidence:{cue.value}:{frame_index}:{artifact_id}"


def _detection_key(frame_index: int, detection_id: str) -> str:
    return f"evidence:objects:{frame_index}:{detection_id}"


def _mask_key(frame_index: int, proposal_id: str) -> str:
    return f"evidence:masks:{frame_index}:{proposal_id}"


def _load_mask(stage_root: Path, mask: MaskObservation | None) -> np.ndarray | None:
    if mask is None:
        return None
    image = read_image_artifact(
        stage_root / mask.mask_ref.relative_path,
        cv2.IMREAD_GRAYSCALE,
    )
    if image is None:
        raise RuntimeError(f"mask artifact is unreadable: {mask.mask_ref.relative_path}")
    binary = image > 0
    if int(binary.sum()) != mask.area_pixels:
        raise RuntimeError(f"mask area contract mismatch: {mask.mask_ref.artifact_id}")
    return binary


def _load_npz(stage_root: Path, reference: ArtifactRef) -> dict[str, np.ndarray]:
    path = stage_root / reference.relative_path
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _box_iou(left: BoundingBoxXYXY, right: BoundingBoxXYXY) -> float:
    ix1, iy1 = max(left.x1, right.x1), max(left.y1, right.y1)
    ix2, iy2 = min(left.x2, right.x2), min(left.y2, right.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_left = (left.x2 - left.x1) * (left.y2 - left.y1)
    area_right = (right.x2 - right.x1) * (right.y2 - right.y1)
    union = area_left + area_right - intersection
    return float(intersection / union) if union > 0 else 0.0


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("mask IoU requires canonical masks with identical shapes")
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def _bbox_from_mask(mask: np.ndarray) -> BoundingBoxXYXY | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return BoundingBoxXYXY(
        x1=float(xs.min()),
        y1=float(ys.min()),
        x2=float(xs.max() + 1),
        y2=float(ys.max() + 1),
    )


def _center_distance_ratio(
    left: BoundingBoxXYXY,
    right: BoundingBoxXYXY,
    image_width: int,
    image_height: int,
) -> float:
    left_center = np.array(((left.x1 + left.x2) / 2, (left.y1 + left.y2) / 2))
    right_center = np.array(((right.x1 + right.x2) / 2, (right.y1 + right.y2) / 2))
    diagonal = float(np.hypot(image_width, image_height))
    return float(np.linalg.norm(left_center - right_center) / diagonal)


def _warp_mask(mask: np.ndarray, flow: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Forward splat a source mask through source-to-target optical flow."""

    if flow.shape[:2] != mask.shape or flow.shape[2] != 2 or valid.shape != mask.shape:
        raise ValueError("mask, flow and validity must share canonical coordinates")
    ys, xs = np.nonzero(mask & valid.astype(bool))
    result = np.zeros_like(mask, dtype=bool)
    if not len(xs):
        return result
    vectors = flow[ys, xs]
    finite = np.isfinite(vectors).all(axis=1)
    ys, xs, vectors = ys[finite], xs[finite], vectors[finite]
    if not len(xs):
        return result
    target_x = np.rint(xs + vectors[:, 0]).astype(np.int64)
    target_y = np.rint(ys + vectors[:, 1]).astype(np.int64)
    inside = (
        (target_x >= 0)
        & (target_x < mask.shape[1])
        & (target_y >= 0)
        & (target_y < mask.shape[0])
    )
    result[target_y[inside], target_x[inside]] = True
    if np.any(result):
        result = cv2.morphologyEx(
            result.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        ).astype(bool)
    return result


def _depth_descriptor(
    *,
    frame: FrameEvidence,
    mask: np.ndarray | None,
    bbox: BoundingBoxXYXY,
    step2_root: Path,
    erosion_pixels: int,
) -> DepthDescriptor | None:
    if frame.depth is None:
        return None
    payload = _load_npz(step2_root, frame.depth.field_ref)
    depth = np.asarray(payload["depth"], dtype=np.float32)
    valid = np.asarray(payload["valid"], dtype=bool) & np.isfinite(depth)
    support_source = "eroded_mask"
    if mask is not None and np.any(mask):
        support = mask.copy()
        if erosion_pixels > 0:
            size = 2 * erosion_pixels + 1
            eroded = cv2.erode(
                support.astype(np.uint8),
                np.ones((size, size), dtype=np.uint8),
            ).astype(bool)
            if np.any(eroded):
                support = eroded
    else:
        support_source = "inner_box"
        support = np.zeros(depth.shape, dtype=bool)
        x1, y1 = int(np.floor(bbox.x1)), int(np.floor(bbox.y1))
        x2, y2 = int(np.ceil(bbox.x2)), int(np.ceil(bbox.y2))
        inset_x, inset_y = max(1, (x2 - x1) // 4), max(1, (y2 - y1) // 4)
        support[y1 + inset_y : y2 - inset_y, x1 + inset_x : x2 - inset_x] = True
    values = depth[support & valid]
    support_count = int(np.count_nonzero(support))
    if not values.size or support_count == 0:
        return None
    q25, median, q75 = (float(value) for value in np.percentile(values, (25, 50, 75)))
    return DepthDescriptor(
        representation=frame.depth.representation,
        unit=frame.depth.unit,
        support_source=support_source,
        valid_fraction=float(values.size / support_count),
        minimum=float(values.min()),
        q25=q25,
        median=median,
        q75=q75,
        maximum=float(values.max()),
        mad=float(np.median(np.abs(values - median))),
    )


def _flow_descriptor(
    mask: np.ndarray | None,
    flow_payload: dict[str, np.ndarray] | None,
) -> FlowDescriptor | None:
    if mask is None or flow_payload is None:
        return None
    flow = np.asarray(flow_payload["flow"], dtype=np.float32)
    valid = np.asarray(flow_payload["consistency_valid"], dtype=bool)
    support = mask & valid & np.isfinite(flow).all(axis=2)
    values = flow[support]
    mask_count = int(mask.sum())
    if not len(values) or mask_count == 0:
        return None
    median_vector = np.median(values, axis=0)
    residuals = np.linalg.norm(values - median_vector, axis=1)
    return FlowDescriptor(
        median_dx_px=float(median_vector[0]),
        median_dy_px=float(median_vector[1]),
        mad_px=float(np.median(residuals)),
        valid_fraction=float(len(values) / mask_count),
    )


def _depth_similarity(left: DepthDescriptor, right: DepthDescriptor) -> float:
    scale = max(abs(left.median), abs(right.median), left.mad, right.mad, 1e-6)
    return float(np.exp(-abs(left.median - right.median) / scale))


def _cue(
    name: str,
    configured_weight: float,
    value: float | None,
    missing_reason: str,
    available_weight_sum: float,
) -> AssociationCueValue:
    available = value is not None
    return AssociationCueValue(
        cue_name=name,
        available=available,
        value=None if value is None else max(0.0, min(1.0, float(value))),
        configured_weight=float(configured_weight),
        effective_weight=(
            float(configured_weight / available_weight_sum)
            if available and available_weight_sum > 0
            else 0.0
        ),
        missing_reason=None if available else missing_reason,
    )


def _evaluate_pair(
    *,
    track: _TrackWork,
    instance: _Instance,
    predicted_mask: np.ndarray | None,
    predicted_box: BoundingBoxXYXY,
    flow_available: bool,
    config: Step3ConfigSnapshot,
    image_width: int,
    image_height: int,
) -> _CandidateEvaluation:
    raw = {
        "mask_iou": (
            _mask_iou(track.last_mask, instance.mask_array)
            if track.last_mask is not None and instance.mask_array is not None
            else None
        ),
        "flow_iou": (
            _mask_iou(predicted_mask, instance.mask_array)
            if flow_available and predicted_mask is not None and instance.mask_array is not None
            else None
        ),
        "box_iou": _box_iou(predicted_box, instance.detection.bbox),
        "class": 1.0 if track.primary_class == instance.detection.class_name else 0.0,
        "depth": (
            _depth_similarity(track.last_depth, instance.depth)
            if track.last_depth is not None and instance.depth is not None
            else None
        ),
    }
    weights = {
        "mask_iou": config.mask_iou_weight,
        "flow_iou": config.flow_iou_weight,
        "box_iou": config.box_iou_weight,
        "class": config.class_weight,
        "depth": config.depth_weight,
    }
    available_weight_sum = sum(weights[name] for name, value in raw.items() if value is not None)
    if available_weight_sum <= 0:
        raise RuntimeError("association pair has no usable cues")
    cues = tuple(
        _cue(
            name,
            weights[name],
            raw[name],
            {
                "mask_iou": "one or both direct masks unavailable",
                "flow_iou": "forward flow or target mask unavailable",
                "box_iou": "box unavailable",
                "class": "class unavailable",
                "depth": "one or both masked depth descriptors unavailable",
            }[name],
            available_weight_sum,
        )
        for name in ("mask_iou", "flow_iou", "box_iou", "class", "depth")
    )
    class_value = 1.0 if track.primary_class == instance.detection.class_name else 0.0
    center_ratio = _center_distance_ratio(
        predicted_box,
        instance.detection.bbox,
        image_width,
        image_height,
    )
    gates = (
        AssociationGate(
            gate_name="class",
            passed=not config.hard_class_gate or bool(class_value),
            measured_value=class_value,
            threshold=1.0 if config.hard_class_gate else 0.0,
            reason=(
                "class labels agree"
                if class_value
                else "class mismatch" if config.hard_class_gate else "class gate disabled"
            ),
        ),
        AssociationGate(
            gate_name="center_distance",
            passed=center_ratio <= config.maximum_center_distance_ratio,
            measured_value=center_ratio,
            threshold=config.maximum_center_distance_ratio,
            reason="normalized center displacement gate",
        ),
    )
    return _CandidateEvaluation(
        track_id=track.track_id,
        detection_id=instance.detection.detection_id,
        proposal_id=instance.mask.proposal_id if instance.mask else None,
        track_age_frames=track.missed_count,
        cues=cues,
        gates=gates,
        feasible=all(gate.passed for gate in gates),
        total_score=float(sum((cue.value or 0.0) * cue.effective_weight for cue in cues)),
    )


def _marker(
    track: _TrackWork,
    marker_type: TrackMarkerType,
    frame_index: int,
    state_after: TrackState,
    trigger: str,
    evidence_keys: Sequence[str] = (),
) -> TrackStateMarker:
    marker = TrackStateMarker(
        marker_id=f"marker:{track.track_id}:{frame_index}:{marker_type.value}",
        marker_type=marker_type,
        frame_index=frame_index,
        state_after=state_after,
        operational_trigger=trigger,
        evidence_keys=tuple(evidence_keys),
    )
    track.markers.append(marker)
    return marker


def _mask_candidate(
    *,
    track_id: str,
    frame_index: int,
    source: MaskCandidateSource,
    observability: SupportObservability,
    mask_link: ArtifactLink | None,
    confidence: float | None,
    detection_id: str | None,
    proposal_id: str | None,
    anchor_frame: int | None,
    parent_keys: Sequence[str],
    transform_id: str,
    selected: bool,
    reason: str,
    ordinal: int,
) -> MaskCandidateRecord:
    return MaskCandidateRecord(
        candidate_id=f"mask_candidate:{track_id}:{frame_index}:{source.value}:{ordinal}",
        track_id=track_id,
        frame_index=frame_index,
        source=source,
        observability=observability,
        mask=mask_link,
        confidence=confidence,
        detection_id=detection_id,
        proposal_id=proposal_id,
        anchor_frame_index=anchor_frame,
        parent_evidence_keys=tuple(parent_keys),
        transform_id=transform_id,
        selected=selected,
        reason=reason,
    )


def _frame_artifact_links(frame: FrameEvidence) -> tuple[ArtifactLink, ...]:
    refs = []
    for cue in (
        frame.mask_cue,
        frame.forward_flow_cue,
        frame.backward_flow_cue,
        frame.depth_cue,
    ):
        refs.extend(_step2_link(reference) for reference in cue.artifact_refs)
    return tuple(refs)


def _unique_links(links: Iterable[ArtifactLink]) -> tuple[ArtifactLink, ...]:
    unique = {}
    for link in links:
        key = (link.owner, link.artifact.artifact_id, link.artifact.sha256)
        unique[key] = link
    return tuple(unique[key] for key in sorted(unique, key=lambda item: (item[0].value, item[1], item[2])))


def _input_artifact_links(manifest: VideoEvidenceManifest) -> tuple[ArtifactLink, ...]:
    return _unique_links(link for frame in manifest.frames for link in _frame_artifact_links(frame))


def _verify_input_artifacts(
    stage_root: Path,
    links: Sequence[ArtifactLink],
) -> tuple[int, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    verified = 0
    unresolved, hash_mismatch, shape_mismatch = [], [], []
    for link in links:
        reference = link.artifact
        path = stage_root / reference.relative_path
        if not path.is_file():
            unresolved.append(reference.artifact_id)
            continue
        if sha256_file(path) != reference.sha256:
            hash_mismatch.append(reference.artifact_id)
            continue
        if reference.shape:
            try:
                expected_shape = tuple(int(value) for value in reference.shape)
                if reference.media_type == "image/png":
                    image = read_image_artifact(path, cv2.IMREAD_UNCHANGED)
                    actual_shape = (
                        None
                        if image is None
                        else tuple(int(value) for value in image.shape)
                    )
                elif path.suffix.lower() == ".npz":
                    with np.load(path, allow_pickle=False) as payload:
                        primary = (
                            payload["flow"]
                            if "flow" in payload
                            else payload["depth"] if "depth" in payload else None
                        )
                    actual_shape = (
                        None
                        if primary is None
                        else tuple(int(value) for value in primary.shape)
                    )
                else:
                    actual_shape = expected_shape
                if actual_shape != expected_shape:
                    shape_mismatch.append(reference.artifact_id)
                    continue
            except (OSError, ValueError, KeyError):
                shape_mismatch.append(reference.artifact_id)
                continue
        verified += 1
    return verified, tuple(unresolved), tuple(hash_mismatch), tuple(shape_mismatch)


def _instances_for_frame(
    frame: FrameEvidence,
    step2_root: Path,
    config: Step3ConfigSnapshot,
) -> tuple[_Instance, ...]:
    masks_by_detection = {
        mask.prompt_detection_id: mask
        for mask in frame.masks
        if mask.prompt_detection_id is not None
    }
    instances = []
    for detection in frame.detections:
        raw_mask = masks_by_detection.get(detection.detection_id)
        mask = (
            raw_mask
            if raw_mask is not None and raw_mask.area_pixels >= config.minimum_mask_area
            else None
        )
        mask_array = _load_mask(step2_root, mask)
        depth = _depth_descriptor(
            frame=frame,
            mask=mask_array,
            bbox=detection.bbox,
            step2_root=step2_root,
            erosion_pixels=config.depth_erosion_pixels,
        )
        instances.append(
            _Instance(
                detection=detection,
                raw_mask=raw_mask,
                mask=mask,
                mask_array=mask_array,
                depth=depth,
            )
        )
    return tuple(instances)


def _create_direct_candidate(
    *,
    track: _TrackWork,
    instance: _Instance,
    frame_index: int,
    selected: bool,
    ordinal: int,
) -> MaskCandidateRecord:
    if instance.mask is None:
        return _mask_candidate(
            track_id=track.track_id,
            frame_index=frame_index,
            source=MaskCandidateSource.EXPLICIT_UNOBSERVABLE,
            observability=SupportObservability.UNOBSERVABLE,
            mask_link=None,
            confidence=None,
            detection_id=instance.detection.detection_id,
            proposal_id=None,
            anchor_frame=frame_index,
            parent_keys=(_detection_key(frame_index, instance.detection.detection_id),),
            transform_id="identity:canonical_image_pixels",
            selected=selected,
            reason=(
                "Step 2 supplied no eligible mask for this selected detection"
                if instance.raw_mask is None
                else "Step 2 mask is below the configured minimum area"
            ),
            ordinal=ordinal,
        )
    return _mask_candidate(
        track_id=track.track_id,
        frame_index=frame_index,
        source=MaskCandidateSource.DIRECT_INSTANCE,
        observability=SupportObservability.OBSERVED,
        mask_link=_step2_link(instance.mask.mask_ref),
        confidence=instance.mask.confidence,
        detection_id=instance.detection.detection_id,
        proposal_id=instance.mask.proposal_id,
        anchor_frame=frame_index,
        parent_keys=(
            _detection_key(frame_index, instance.detection.detection_id),
            _mask_key(frame_index, instance.mask.proposal_id),
        ),
        transform_id="identity:canonical_image_pixels",
        selected=selected,
        reason="frame-local SAM 2 instance prompted by this detection",
        ordinal=ordinal,
    )


def _persist_warp_candidate(
    *,
    stage_root: Path,
    video_id: str,
    track: _TrackWork,
    frame_index: int,
    mask: np.ndarray,
    source: MaskCandidateSource,
    flow_reference: ArtifactRef,
    direction: str,
    ordinal: int,
) -> tuple[MaskCandidateRecord, ArtifactLink | None]:
    if not np.any(mask):
        candidate = _mask_candidate(
            track_id=track.track_id,
            frame_index=frame_index,
            source=MaskCandidateSource.EMPTY_OR_OUTSIDE,
            observability=SupportObservability.UNOBSERVABLE,
            mask_link=None,
            confidence=None,
            detection_id=None,
            proposal_id=None,
            anchor_frame=track.last_observed_frame,
            parent_keys=(flow_reference.artifact_id,),
            transform_id=f"flow_{direction}_splat:canonical_image_pixels",
            selected=False,
            reason="flow-warped support is empty or wholly outside the frame",
            ordinal=ordinal,
        )
        return candidate, None
    relative_path = (
        Path("artifacts")
        / "mask_candidates"
        / video_id
        / track.track_id.replace(":", "_")
        / f"frame_{frame_index:06d}_{source.value}_{ordinal:03d}.png"
    )
    reference = write_mask_artifact(
        stage_root=stage_root,
        relative_path=relative_path,
        artifact_id=f"derived_mask:{video_id}:{track.track_id}:{frame_index}:{source.value}:{ordinal}",
        mask=mask,
    )
    link = _step3_link(reference)
    candidate = _mask_candidate(
        track_id=track.track_id,
        frame_index=frame_index,
        source=source,
        observability=SupportObservability.LATENT_SUPPORT,
        mask_link=link,
        confidence=None,
        detection_id=None,
        proposal_id=None,
        anchor_frame=track.last_observed_frame,
        parent_keys=(
            track.support_mask_link.artifact.artifact_id
            if track.support_mask_link
            else "input_missing:support_mask",
            flow_reference.artifact_id,
        ),
        transform_id=f"flow_{direction}_splat:canonical_image_pixels",
        selected=False,
        reason="latent support warped from the last reliable observed mask",
        ordinal=ordinal,
    )
    return candidate, link


def _archive_backward_gap_candidates(
    *,
    stage_root: Path,
    step2_root: Path,
    manifest: VideoEvidenceManifest,
    track: _TrackWork,
    current_frame: FrameEvidence,
    current_instance: _Instance,
    mask_candidates: list[MaskCandidateRecord],
    derived_links: list[ArtifactLink],
) -> None:
    """Warp the next reliable mask backward and retain it beside forward proposals."""

    if not track.open_gap_frames or current_instance.mask_array is None:
        return
    support = current_instance.mask_array
    parent_artifact_id = current_instance.mask.mask_ref.artifact_id
    source_index = current_frame.frame_index
    for target_index in reversed(track.open_gap_frames):
        if source_index != target_index + 1:
            break
        source_frame = manifest.frames[source_index]
        if source_frame.backward_flow is None:
            break
        flow_reference = source_frame.backward_flow.field_ref
        payload = _load_npz(step2_root, flow_reference)
        warped = _warp_mask(
            support,
            payload["flow"],
            payload["consistency_valid"],
        )
        ordinal = len(mask_candidates)
        if np.any(warped):
            relative_path = (
                Path("artifacts")
                / "mask_candidates"
                / manifest.video_id
                / track.track_id.replace(":", "_")
                / f"frame_{target_index:06d}_flow_backward_{ordinal:03d}.png"
            )
            reference = write_mask_artifact(
                stage_root=stage_root,
                relative_path=relative_path,
                artifact_id=(
                    f"derived_mask:{manifest.video_id}:{track.track_id}:"
                    f"{target_index}:flow_backward:{ordinal}"
                ),
                mask=warped,
            )
            link = _step3_link(reference)
            candidate = _mask_candidate(
                track_id=track.track_id,
                frame_index=target_index,
                source=MaskCandidateSource.FLOW_BACKWARD,
                observability=SupportObservability.LATENT_SUPPORT,
                mask_link=link,
                confidence=None,
                detection_id=None,
                proposal_id=None,
                anchor_frame=current_frame.frame_index,
                parent_keys=(parent_artifact_id, flow_reference.artifact_id),
                transform_id="flow_backward_splat:canonical_image_pixels",
                selected=False,
                reason="backward latent support from the next reliable mask anchor",
                ordinal=ordinal,
            )
            derived_links.append(link)
            parent_artifact_id = reference.artifact_id
        else:
            candidate = _mask_candidate(
                track_id=track.track_id,
                frame_index=target_index,
                source=MaskCandidateSource.EMPTY_OR_OUTSIDE,
                observability=SupportObservability.UNOBSERVABLE,
                mask_link=None,
                confidence=None,
                detection_id=None,
                proposal_id=None,
                anchor_frame=current_frame.frame_index,
                parent_keys=(parent_artifact_id, flow_reference.artifact_id),
                transform_id="flow_backward_splat:canonical_image_pixels",
                selected=False,
                reason="backward flow-warped support is empty or outside the frame",
                ordinal=ordinal,
            )
        mask_candidates.append(candidate)
        track.open_gap_candidate_ids.append(candidate.candidate_id)
        track.open_gap_context.append(_step2_link(flow_reference))
        support = warped
        source_index = target_index


def _observation(
    *,
    frame: FrameEvidence,
    instance: _Instance,
    direct_candidate: MaskCandidateRecord,
    ledger_id: str | None,
    flow_descriptor: FlowDescriptor | None,
) -> TrackObservation:
    missing = []
    if instance.mask is None:
        missing.append(CueFamily.MASKS)
    if frame.forward_flow is None:
        missing.append(CueFamily.FLOW_FORWARD)
    if frame.backward_flow is None:
        missing.append(CueFamily.FLOW_BACKWARD)
    if frame.depth is None or instance.depth is None:
        missing.append(CueFamily.DEPTH)
    return TrackObservation(
        frame_index=frame.frame_index,
        timestamp_s=frame.timestamp_s,
        detection_id=instance.detection.detection_id,
        proposal_id=instance.mask.proposal_id if instance.mask else None,
        class_name=instance.detection.class_name,
        detection_tier=instance.detection.tier,
        confidence=instance.detection.confidence,
        bbox=instance.detection.bbox,
        selected_mask_candidate_id=direct_candidate.candidate_id,
        forward_flow=(
            _step2_link(frame.forward_flow.field_ref) if frame.forward_flow else None
        ),
        backward_flow=(
            _step2_link(frame.backward_flow.field_ref) if frame.backward_flow else None
        ),
        depth=_step2_link(frame.depth.field_ref) if frame.depth else None,
        depth_descriptor=instance.depth,
        flow_descriptor=flow_descriptor,
        missing_cues=tuple(missing),
        association_ledger_id=ledger_id,
    )


def _frame_evidence_entries(
    frame: FrameEvidence,
) -> tuple[tuple[str, CueFamily, ArtifactRef | None], ...]:
    entries = []
    for detection in frame.detections:
        entries.append((_detection_key(frame.frame_index, detection.detection_id), CueFamily.OBJECTS, None))
    for mask in frame.masks:
        entries.append((_mask_key(frame.frame_index, mask.proposal_id), CueFamily.MASKS, mask.mask_ref))
    for family, observation in (
        (CueFamily.FLOW_FORWARD, frame.forward_flow),
        (CueFamily.FLOW_BACKWARD, frame.backward_flow),
        (CueFamily.DEPTH, frame.depth),
    ):
        if observation is not None:
            reference = observation.field_ref
            entries.append((_artifact_key(family, frame.frame_index, reference.artifact_id), family, reference))
    return tuple(entries)


def _close_gap(
    *,
    track: _TrackWork,
    next_reliable_frame: int | None,
    status: GapStatus,
    unassigned: Sequence[UnassignedEvidenceRecord],
) -> GapEvidenceRecord | None:
    if not track.open_gap_frames:
        return None
    first_gap, last_gap = track.open_gap_frames[0], track.open_gap_frames[-1]
    relevant_unassigned = tuple(
        item.unassigned_id
        for item in unassigned
        if first_gap <= item.frame_index <= (next_reliable_frame or last_gap)
    )
    gap = GapEvidenceRecord(
        gap_id=f"gap:{track.track_id}:{first_gap}:{last_gap}",
        track_id=track.track_id,
        last_reliable_frame=first_gap - 1,
        next_reliable_frame=next_reliable_frame,
        gap_frames=tuple(track.open_gap_frames),
        status=status,
        mask_candidate_ids=tuple(track.open_gap_candidate_ids),
        unassigned_evidence_ids=relevant_unassigned,
        marker_ids=tuple(track.open_gap_marker_ids),
        context_artifacts=_unique_links(track.open_gap_context),
    )
    track.open_gap_frames.clear()
    track.open_gap_candidate_ids.clear()
    track.open_gap_marker_ids.clear()
    track.open_gap_context.clear()
    return gap


def _evidence_use_plan(
    *,
    video_id: str,
    manifest: VideoEvidenceManifest,
    seed: int,
    depth_check_fraction: float,
) -> EvidenceUsePlan:
    rng = np.random.default_rng(seed)
    assignments = []
    for frame in manifest.frames:
        for evidence_key, cue_family, reference in _frame_evidence_entries(frame):
            if cue_family == CueFamily.FLOW_BACKWARD:
                role = EvidenceRole.CHECK_ONLY
                reason = "backward flow is reserved for independent temporal checks"
            elif cue_family == CueFamily.DEPTH and rng.random() < depth_check_fraction:
                role = EvidenceRole.CHECK_ONLY
                reason = "seeded depth holdout for independent consistency checks"
            else:
                role = EvidenceRole.FIT
                reason = "available to association or later world-state fitting"
            assignments.append(
                EvidenceRoleAssignment(
                    assignment_id=f"role:{video_id}:{len(assignments)}",
                    evidence_key=evidence_key,
                    cue_family=cue_family,
                    frame_index=frame.frame_index,
                    artifact=_step2_link(reference) if reference else None,
                    role=role,
                    allowed_consumers=("step4_geometry", "step5_world_state", "step6_verification"),
                    prohibited_optimizers=(
                        ("step4_geometry", "step5_world_state", "step8_local_reestimation")
                        if role == EvidenceRole.CHECK_ONLY
                        else ()
                    ),
                    selection_reason=reason,
                )
            )
    payload = {
        "policy_version": "step3_evidence_roles_v1",
        "random_seed": seed,
        "assignments": [item.model_dump(mode="json") for item in assignments],
    }
    return EvidenceUsePlan(
        plan_id=f"evidence_plan:{video_id}:{hash_payload(payload)[:16]}",
        random_seed=seed,
        assignments=tuple(assignments),
        plan_sha256=hash_payload(payload),
    )


def _retention_counts(
    manifest: VideoEvidenceManifest,
    dispositions: Sequence[EvidenceDispositionRecord],
) -> tuple[ModalityRetentionCount, ...]:
    by_key = {item.evidence_key: item for item in dispositions}
    counts = []
    for frame in manifest.frames:
        entries = _frame_evidence_entries(frame)
        status_by_family = {
            CueFamily.OBJECTS: frame.object_cue.status,
            CueFamily.MASKS: frame.mask_cue.status,
            CueFamily.FLOW_FORWARD: frame.forward_flow_cue.status,
            CueFamily.FLOW_BACKWARD: frame.backward_flow_cue.status,
            CueFamily.DEPTH: frame.depth_cue.status,
        }
        for family in CueFamily:
            family_entries = [item for item in entries if item[1] == family]
            selected = sum(
                by_key[item[0]].disposition == EvidenceDisposition.SELECTED
                for item in family_entries
            )
            unselected = sum(
                by_key[item[0]].disposition == EvidenceDisposition.UNSELECTED
                for item in family_entries
            )
            invalid = sum(
                by_key[item[0]].disposition == EvidenceDisposition.INVALID
                for item in family_entries
            )
            counts.append(
                ModalityRetentionCount(
                    frame_index=frame.frame_index,
                    cue_family=family,
                    input_count=len(family_entries),
                    selected_count=selected,
                    unselected_count=unselected,
                    invalid_count=invalid,
                    input_status=status_by_family[family],
                )
            )
    return tuple(counts)


def _build_video_package(
    *,
    loaded: _LoadedStep2,
    manifest: VideoEvidenceManifest,
    manifest_reference: ArtifactRef,
    config: Step3ConfigSnapshot,
    config_sha256: str,
    stage_root: Path,
) -> VideoTrackingManifest:
    input_links = _input_artifact_links(manifest)
    verified, unresolved, hash_mismatch, shape_mismatch = _verify_input_artifacts(
        loaded.stage_root,
        input_links,
    )
    source_manifest_path = (
        loaded.run_root / "01_init" / manifest.source_manifest.relative_path
    )
    if not source_manifest_path.is_file():
        unresolved = (*unresolved, manifest.source_manifest.artifact_id)
    elif sha256_file(source_manifest_path) != manifest.source_manifest.sha256:
        hash_mismatch = (*hash_mismatch, manifest.source_manifest.artifact_id)
    else:
        verified += 1
    # The Step 2 store and per-video manifest were parsed and hash-validated by
    # _load_step2; count both boundary records in the retention closure.
    verified += 2
    boundary_record_count = 3
    if unresolved or hash_mismatch or shape_mismatch:
        raise RuntimeError(
            "Step 3 input retention gate failed before tracking: "
            f"unresolved={unresolved}, hash={hash_mismatch}, shape={shape_mismatch}"
        )
    store_reference = ArtifactRef(
        artifact_id="neural_evidence_store",
        relative_path="neural_evidence_store.json",
        sha256=sha256_file(loaded.store_path),
        byte_size=loaded.store_path.stat().st_size,
        media_type="application/vnd.cauvid.neural-evidence-store+json",
    )
    snapshot = Step3InputSnapshot(
        source_step2_relative_root=loaded.stage_root.relative_to(loaded.run_root).as_posix(),
        source_video_manifest=_step1_link(manifest.source_manifest),
        neural_evidence_store=_step2_link(store_reference),
        video_evidence_manifest=_step2_link(manifest_reference),
        input_artifacts=input_links,
    )
    tracks: dict[str, _TrackWork] = {}
    mask_candidates: list[MaskCandidateRecord] = []
    derived_links: list[ArtifactLink] = []
    ledger: list[AssociationLedgerRow] = []
    unassigned: list[UnassignedEvidenceRecord] = []
    gaps: list[GapEvidenceRecord] = []
    dispositions: dict[str, EvidenceDispositionRecord] = {}
    expected_pairs = 0
    next_track_number = 1

    for frame in manifest.frames:
        instances = _instances_for_frame(frame, loaded.stage_root, config)
        for evidence_key, cue_family, _ in _frame_evidence_entries(frame):
            dispositions[evidence_key] = EvidenceDispositionRecord(
                evidence_key=evidence_key,
                cue_family=cue_family,
                frame_index=frame.frame_index,
                disposition=EvidenceDisposition.UNSELECTED,
                reason="retained in Step 3 input snapshot; not yet selected",
            )

        active_tracks = [
            track
            for track in tracks.values()
            if track.state in {TrackState.ACTIVE, TrackState.LOST}
            and track.missed_count <= config.max_age_frames
        ]
        predictions: dict[str, tuple[np.ndarray | None, BoundingBoxXYXY, bool]] = {}
        for track in active_tracks:
            predicted_mask = None
            flow_available = False
            if (
                track.support_frame == frame.frame_index - 1
                and track.support_mask is not None
                and frame.frame_index > 0
            ):
                previous_frame = manifest.frames[frame.frame_index - 1]
                if previous_frame.forward_flow is not None:
                    forward_payload = _load_npz(
                        loaded.stage_root,
                        previous_frame.forward_flow.field_ref,
                    )
                    predicted_mask = _warp_mask(
                        track.support_mask,
                        forward_payload["flow"],
                        forward_payload["consistency_valid"],
                    )
                    flow_available = True
            predicted_box = _bbox_from_mask(predicted_mask) if predicted_mask is not None else None
            predictions[track.track_id] = (
                predicted_mask,
                predicted_box or track.last_bbox,
                flow_available,
            )

        evaluations: dict[tuple[str, str], _CandidateEvaluation] = {}
        for track in active_tracks:
            predicted_mask, predicted_box, flow_available = predictions[track.track_id]
            for instance in instances:
                evaluation = _evaluate_pair(
                    track=track,
                    instance=instance,
                    predicted_mask=predicted_mask,
                    predicted_box=predicted_box,
                    flow_available=flow_available,
                    config=config,
                    image_width=manifest.image_size.width,
                    image_height=manifest.image_size.height,
                )
                evaluations[(track.track_id, instance.detection.detection_id)] = evaluation
        expected_pairs += len(evaluations)

        assignment = assign_one_to_one_matches(
            track_ids=[track.track_id for track in active_tracks],
            detection_ids=[instance.detection.detection_id for instance in instances],
            score_by_pair={pair: evaluation.total_score for pair, evaluation in evaluations.items()},
            feasible_by_pair={pair: evaluation.feasible for pair, evaluation in evaluations.items()},
            minimum_score=config.minimum_assignment_score,
        )
        selected_pairs = set(assignment.selected_pairs)
        ranks = assignment.rank_by_pair
        selected_detection_ids = {detection_id for _, detection_id in selected_pairs}
        for evaluation in sorted(evaluations.values(), key=lambda item: (item.track_id, item.detection_id)):
            pair = (evaluation.track_id, evaluation.detection_id)
            selected = pair in selected_pairs
            if selected:
                decision = AssociationDecision.MATCHED
                reason = "selected by one-to-one Hungarian assignment"
            elif not evaluation.feasible:
                decision = AssociationDecision.REJECTED_GATE
                reason = "one or more deterministic gates failed"
            elif evaluation.total_score < config.minimum_assignment_score:
                decision = AssociationDecision.REJECTED_THRESHOLD
                reason = "score below minimum assignment threshold"
            else:
                decision = AssociationDecision.REJECTED_CONFLICT
                reason = "feasible pair lost one-to-one assignment conflict"
            ledger_id = f"association:{manifest.video_id}:{frame.frame_index}:{len(ledger)}"
            ledger.append(
                AssociationLedgerRow(
                    ledger_id=ledger_id,
                    frame_index=frame.frame_index,
                    track_id=evaluation.track_id,
                    detection_id=evaluation.detection_id,
                    proposal_id=evaluation.proposal_id,
                    track_age_frames=evaluation.track_age_frames,
                    cues=evaluation.cues,
                    gates=evaluation.gates,
                    feasible=evaluation.feasible,
                    total_score=evaluation.total_score,
                    rank_for_track=ranks[pair],
                    decision=decision,
                    selected=selected,
                    decision_reason=reason,
                )
            )

        ledger_by_selected = {
            (row.track_id, row.detection_id): row.ledger_id for row in ledger if row.selected
        }
        instances_by_id = {item.detection.detection_id: item for item in instances}
        for track in active_tracks:
            matched_detection = next(
                (detection_id for track_id, detection_id in selected_pairs if track_id == track.track_id),
                None,
            )
            if matched_detection is not None:
                instance = instances_by_id[matched_detection]
                direct = _create_direct_candidate(
                    track=track,
                    instance=instance,
                    frame_index=frame.frame_index,
                    selected=True,
                    ordinal=len(mask_candidates),
                )
                mask_candidates.append(direct)
                previous_frame = manifest.frames[frame.frame_index - 1] if frame.frame_index else None
                forward_payload = (
                    _load_npz(loaded.stage_root, previous_frame.forward_flow.field_ref)
                    if previous_frame is not None and previous_frame.forward_flow is not None
                    else None
                )
                flow_desc = _flow_descriptor(track.last_mask, forward_payload)
                observation = _observation(
                    frame=frame,
                    instance=instance,
                    direct_candidate=direct,
                    ledger_id=ledger_by_selected[(track.track_id, matched_detection)],
                    flow_descriptor=flow_desc,
                )
                was_lost = track.state == TrackState.LOST
                if was_lost:
                    _archive_backward_gap_candidates(
                        stage_root=stage_root,
                        step2_root=loaded.stage_root,
                        manifest=manifest,
                        track=track,
                        current_frame=frame,
                        current_instance=instance,
                        mask_candidates=mask_candidates,
                        derived_links=derived_links,
                    )
                    track.open_gap_context.extend(_frame_artifact_links(frame))
                track.observations.append(observation)
                track.state = TrackState.ACTIVE
                track.last_observed_frame = frame.frame_index
                track.missed_count = 0
                track.last_mask = instance.mask_array
                track.last_bbox = instance.detection.bbox
                track.last_depth = instance.depth
                track.support_mask = instance.mask_array
                track.support_mask_link = (
                    _step2_link(instance.mask.mask_ref) if instance.mask is not None else None
                )
                track.support_frame = frame.frame_index
                marker = _marker(
                    track,
                    TrackMarkerType.REOBSERVED if was_lost else TrackMarkerType.MATCHED,
                    frame.frame_index,
                    TrackState.ACTIVE,
                    "association_selected_after_gap" if was_lost else "association_selected",
                    (matched_detection,),
                )
                if was_lost:
                    track.open_gap_marker_ids.append(marker.marker_id)
                    gap = _close_gap(
                        track=track,
                        next_reliable_frame=frame.frame_index,
                        status=GapStatus.REOBSERVED,
                        unassigned=unassigned,
                    )
                    if gap:
                        gaps.append(gap)
                dispositions[_detection_key(frame.frame_index, matched_detection)] = EvidenceDispositionRecord(
                    evidence_key=_detection_key(frame.frame_index, matched_detection),
                    cue_family=CueFamily.OBJECTS,
                    frame_index=frame.frame_index,
                    disposition=EvidenceDisposition.SELECTED,
                    track_id=track.track_id,
                    reason="selected track observation",
                )
                if instance.mask:
                    dispositions[_mask_key(frame.frame_index, instance.mask.proposal_id)] = EvidenceDispositionRecord(
                        evidence_key=_mask_key(frame.frame_index, instance.mask.proposal_id),
                        cue_family=CueFamily.MASKS,
                        frame_index=frame.frame_index,
                        disposition=EvidenceDisposition.SELECTED,
                        track_id=track.track_id,
                        reason="selected direct mask observation",
                    )
                elif instance.raw_mask is not None:
                    dispositions[_mask_key(frame.frame_index, instance.raw_mask.proposal_id)] = EvidenceDispositionRecord(
                        evidence_key=_mask_key(frame.frame_index, instance.raw_mask.proposal_id),
                        cue_family=CueFamily.MASKS,
                        frame_index=frame.frame_index,
                        disposition=EvidenceDisposition.INVALID,
                        track_id=track.track_id,
                        reason="mask area below Step 3 minimum",
                    )
            else:
                track.missed_count += 1
                track.state = TrackState.LOST
                marker = _marker(
                    track,
                    TrackMarkerType.MISSED,
                    frame.frame_index,
                    TrackState.LOST,
                    "no_assignment_selected",
                )
                track.open_gap_frames.append(frame.frame_index)
                track.open_gap_marker_ids.append(marker.marker_id)
                track.open_gap_context.extend(_frame_artifact_links(frame))
                predicted_mask, _, flow_available = predictions[track.track_id]
                previous_frame = manifest.frames[frame.frame_index - 1] if frame.frame_index else None
                if flow_available and predicted_mask is not None and previous_frame.forward_flow is not None:
                    candidate, derived_link = _persist_warp_candidate(
                        stage_root=stage_root,
                        video_id=manifest.video_id,
                        track=track,
                        frame_index=frame.frame_index,
                        mask=predicted_mask,
                        source=MaskCandidateSource.FLOW_FORWARD,
                        flow_reference=previous_frame.forward_flow.field_ref,
                        direction="forward",
                        ordinal=len(mask_candidates),
                    )
                else:
                    candidate = _mask_candidate(
                        track_id=track.track_id,
                        frame_index=frame.frame_index,
                        source=MaskCandidateSource.EXPLICIT_UNOBSERVABLE,
                        observability=SupportObservability.UNOBSERVABLE,
                        mask_link=None,
                        confidence=None,
                        detection_id=None,
                        proposal_id=None,
                        anchor_frame=track.last_observed_frame,
                        parent_keys=(),
                        transform_id="identity:canonical_image_pixels",
                        selected=False,
                        reason="no adjacent forward flow and no direct selected mask",
                        ordinal=len(mask_candidates),
                    )
                    derived_link = None
                mask_candidates.append(candidate)
                track.open_gap_candidate_ids.append(candidate.candidate_id)
                if derived_link:
                    derived_links.append(derived_link)
                    track.support_mask = predicted_mask
                    track.support_mask_link = derived_link
                    track.support_frame = frame.frame_index
                else:
                    track.support_mask = None
                    track.support_mask_link = None
                    track.support_frame = frame.frame_index
                if track.missed_count > config.max_age_frames:
                    track.state = TrackState.RETIRED
                    retired = _marker(
                        track,
                        TrackMarkerType.RETIRED,
                        frame.frame_index,
                        TrackState.RETIRED,
                        "max_age_exceeded",
                    )
                    track.open_gap_marker_ids.append(retired.marker_id)
                    gap = _close_gap(
                        track=track,
                        next_reliable_frame=None,
                        status=GapStatus.RETIRED,
                        unassigned=unassigned,
                    )
                    if gap:
                        gaps.append(gap)

        for instance in instances:
            if instance.detection.detection_id in selected_detection_ids:
                continue
            eligible_for_new = (
                instance.detection.tier == DetectionTier.PRIMARY
                or not config.bootstrap_primary_only
            )
            related = tuple(
                row.ledger_id
                for row in ledger
                if row.frame_index == frame.frame_index
                and row.detection_id == instance.detection.detection_id
            )
            if eligible_for_new:
                track_id = f"track:{manifest.video_id}:{next_track_number:06d}"
                next_track_number += 1
                track = _TrackWork(
                    track_id=track_id,
                    primary_class=instance.detection.class_name,
                    state=TrackState.ACTIVE,
                    first_observed_frame=frame.frame_index,
                    last_observed_frame=frame.frame_index,
                    missed_count=0,
                    last_mask=instance.mask_array,
                    last_bbox=instance.detection.bbox,
                    last_depth=instance.depth,
                    support_mask=instance.mask_array,
                    support_mask_link=(
                        _step2_link(instance.mask.mask_ref) if instance.mask is not None else None
                    ),
                    support_frame=frame.frame_index,
                )
                direct = _create_direct_candidate(
                    track=track,
                    instance=instance,
                    frame_index=frame.frame_index,
                    selected=True,
                    ordinal=len(mask_candidates),
                )
                mask_candidates.append(direct)
                track.observations.append(
                    _observation(
                        frame=frame,
                        instance=instance,
                        direct_candidate=direct,
                        ledger_id=None,
                        flow_descriptor=None,
                    )
                )
                _marker(
                    track,
                    TrackMarkerType.FIRST_OBSERVED,
                    frame.frame_index,
                    TrackState.ACTIVE,
                    "unassigned_instance_bootstrap",
                    (instance.detection.detection_id,),
                )
                tracks[track_id] = track
                dispositions[_detection_key(frame.frame_index, instance.detection.detection_id)] = EvidenceDispositionRecord(
                    evidence_key=_detection_key(frame.frame_index, instance.detection.detection_id),
                    cue_family=CueFamily.OBJECTS,
                    frame_index=frame.frame_index,
                    disposition=EvidenceDisposition.SELECTED,
                    track_id=track_id,
                    reason="bootstrapped a new track",
                )
                if instance.mask:
                    dispositions[_mask_key(frame.frame_index, instance.mask.proposal_id)] = EvidenceDispositionRecord(
                        evidence_key=_mask_key(frame.frame_index, instance.mask.proposal_id),
                        cue_family=CueFamily.MASKS,
                        frame_index=frame.frame_index,
                        disposition=EvidenceDisposition.SELECTED,
                        track_id=track_id,
                        reason="selected direct mask for new track",
                    )
                elif instance.raw_mask is not None:
                    dispositions[_mask_key(frame.frame_index, instance.raw_mask.proposal_id)] = EvidenceDispositionRecord(
                        evidence_key=_mask_key(frame.frame_index, instance.raw_mask.proposal_id),
                        cue_family=CueFamily.MASKS,
                        frame_index=frame.frame_index,
                        disposition=EvidenceDisposition.INVALID,
                        track_id=track_id,
                        reason="mask area below Step 3 minimum",
                    )
            else:
                unassigned_id = f"unassigned:{manifest.video_id}:{frame.frame_index}:{len(unassigned)}"
                record = UnassignedEvidenceRecord(
                    unassigned_id=unassigned_id,
                    frame_index=frame.frame_index,
                    detection_id=instance.detection.detection_id,
                    proposal_id=(
                        instance.raw_mask.proposal_id if instance.raw_mask else None
                    ),
                    mask=(
                        _step2_link(instance.raw_mask.mask_ref)
                        if instance.raw_mask
                        else None
                    ),
                    related_ledger_ids=related,
                    reason="candidate-tier observation cannot bootstrap a track",
                )
                unassigned.append(record)
                if instance.raw_mask is not None and instance.mask is None:
                    dispositions[_mask_key(frame.frame_index, instance.raw_mask.proposal_id)] = EvidenceDispositionRecord(
                        evidence_key=_mask_key(frame.frame_index, instance.raw_mask.proposal_id),
                        cue_family=CueFamily.MASKS,
                        frame_index=frame.frame_index,
                        disposition=EvidenceDisposition.INVALID,
                        reason="unassigned mask area below Step 3 minimum",
                    )
                for track in active_tracks:
                    candidate = _mask_candidate(
                        track_id=track.track_id,
                        frame_index=frame.frame_index,
                        source=MaskCandidateSource.UNASSIGNED_INSTANCE,
                        observability=SupportObservability.OBSERVED,
                        mask_link=_step2_link(instance.mask.mask_ref) if instance.mask else None,
                        confidence=instance.mask.confidence if instance.mask else None,
                        detection_id=instance.detection.detection_id,
                        proposal_id=instance.mask.proposal_id if instance.mask else None,
                        anchor_frame=frame.frame_index,
                        parent_keys=(_detection_key(frame.frame_index, instance.detection.detection_id),),
                        transform_id="identity:canonical_image_pixels",
                        selected=False,
                        reason="unassigned current instance retained as a relinking candidate",
                        ordinal=len(mask_candidates),
                    ) if instance.mask else None
                    if candidate:
                        mask_candidates.append(candidate)
                        if track.open_gap_frames:
                            track.open_gap_candidate_ids.append(candidate.candidate_id)

        # Dense flow/depth artifacts remain selected context whenever they are
        # available; they are never copied or consumed destructively.
        for family, observation in (
            (CueFamily.FLOW_FORWARD, frame.forward_flow),
            (CueFamily.FLOW_BACKWARD, frame.backward_flow),
            (CueFamily.DEPTH, frame.depth),
        ):
            if observation is not None:
                key = _artifact_key(family, frame.frame_index, observation.field_ref.artifact_id)
                dispositions[key] = EvidenceDispositionRecord(
                    evidence_key=key,
                    cue_family=family,
                    frame_index=frame.frame_index,
                    disposition=EvidenceDisposition.SELECTED,
                    reason="retained as shared tracking and downstream context",
                )

    final_frame = manifest.frame_count - 1
    for track in tracks.values():
        if track.state != TrackState.RETIRED:
            track.state = TrackState.RETIRED
            marker = _marker(
                track,
                TrackMarkerType.VIDEO_END,
                final_frame,
                TrackState.RETIRED,
                "video_end",
            )
            if track.open_gap_frames:
                track.open_gap_marker_ids.append(marker.marker_id)
                gap = _close_gap(
                    track=track,
                    next_reliable_frame=None,
                    status=GapStatus.VIDEO_END,
                    unassigned=unassigned,
                )
                if gap:
                    gaps.append(gap)

    track_view = tuple(
        ObjectMaskTrack(
            track_id=track.track_id,
            primary_class=track.primary_class,
            terminal_state=track.state,
            first_observed_frame=track.first_observed_frame,
            last_observed_frame=track.last_observed_frame,
            observations=tuple(track.observations),
            state_markers=tuple(track.markers),
        )
        for track in sorted(tracks.values(), key=lambda item: item.track_id)
    )
    dispositions_tuple = tuple(dispositions[key] for key in sorted(dispositions))
    expected_evidence_keys = {
        entry[0] for frame in manifest.frames for entry in _frame_evidence_entries(frame)
    }
    disposition_complete = expected_evidence_keys == set(dispositions)
    candidate_coverage = {
        (candidate.track_id, candidate.frame_index) for candidate in mask_candidates
    }
    observation_coverage = {
        (track.track_id, observation.frame_index)
        for track in track_view
        for observation in track.observations
        if observation.selected_mask_candidate_id is not None
        or CueFamily.MASKS in observation.missing_cues
    }
    required_pairs = {
        (track.track_id, observation.frame_index)
        for track in track_view
        for observation in track.observations
    } | {
        (track.track_id, marker.frame_index)
        for track in track_view
        for marker in track.state_markers
        if marker.marker_type == TrackMarkerType.MISSED
    }
    covered_pairs = required_pairs & (candidate_coverage | observation_coverage)
    coverage_violations = tuple(
        f"coverage:{track_id}:{frame_index}"
        for track_id, frame_index in sorted(required_pairs - covered_pairs)
    )
    report = RetentionReport(
        modality_counts=_retention_counts(manifest, dispositions_tuple),
        input_artifact_count=len(input_links) + boundary_record_count,
        verified_artifact_count=verified,
        unresolved_evidence_keys=unresolved,
        hash_mismatch_evidence_keys=hash_mismatch,
        shape_mismatch_evidence_keys=shape_mismatch,
        expected_candidate_pairs=expected_pairs,
        ledger_rows=len(ledger),
        required_track_frames=len(required_pairs),
        covered_track_frames=len(covered_pairs),
        coverage_violations=coverage_violations,
        disposition_complete=disposition_complete,
        overall_pass=(
            len(input_links) + boundary_record_count == verified
            and not unresolved
            and not hash_mismatch
            and not shape_mismatch
            and expected_pairs == len(ledger)
            and len(required_pairs) == len(covered_pairs)
            and not coverage_violations
            and disposition_complete
        ),
    )
    plan = _evidence_use_plan(
        video_id=manifest.video_id,
        manifest=manifest,
        seed=config.evidence_policy_seed,
        depth_check_fraction=config.depth_check_fraction,
    )
    transforms = (
        TransformRecord(
            transform_id="identity:canonical_image_pixels",
            transform_type="identity",
            reversible=True,
            description="Step 2 and Step 3 artifacts share canonical image coordinates",
        ),
        TransformRecord(
            transform_id="flow_forward_splat:canonical_image_pixels",
            transform_type="flow_forward_splat",
            reversible=False,
            description="nearest-pixel forward splat through RAFT flow with consistency validity",
        ),
        TransformRecord(
            transform_id="flow_backward_splat:canonical_image_pixels",
            transform_type="flow_backward_splat",
            reversible=False,
            description="nearest-pixel backward splat retained as a separate candidate",
        ),
    )
    return VideoTrackingManifest(
        run_id=manifest.run_id,
        video_id=manifest.video_id,
        source_evidence_sha256=manifest_reference.sha256,
        config_sha256=config_sha256,
        canonical_fps=manifest.canonical_fps,
        image_size=manifest.image_size,
        frame_count=manifest.frame_count,
        input_snapshot=snapshot,
        tracks=track_view,
        association_ledger=tuple(ledger),
        mask_candidate_bank=tuple(mask_candidates),
        gap_records=tuple(gaps),
        unassigned_evidence=tuple(unassigned),
        evidence_dispositions=dispositions_tuple,
        derived_artifacts=_unique_links(derived_links),
        transform_registry=transforms,
        evidence_use_plan=plan,
        retention_report=report,
        tool_versions=(
            ToolVersion(name="opencv", version=cv2.__version__),
            ToolVersion(name="numpy", version=np.__version__),
        ),
    )


def run_step3(
    *,
    neural_evidence_store_path: Path | str,
    max_age_frames: int = 2,
    minimum_assignment_score: float = 0.30,
    maximum_center_distance_ratio: float = 0.25,
    hard_class_gate: bool = True,
    bootstrap_primary_only: bool = True,
    minimum_mask_area: int = 16,
    depth_erosion_pixels: int = 2,
    mask_iou_weight: float = 0.40,
    flow_iou_weight: float = 0.20,
    box_iou_weight: float = 0.20,
    class_weight: float = 0.10,
    depth_weight: float = 0.10,
    evidence_policy_seed: int = 726381,
    depth_check_fraction: float = 0.20,
) -> Step3Result:
    """Build stable image-space mask tracks and a lossless evidence index."""

    loaded = _load_step2(neural_evidence_store_path)
    config = Step3ConfigSnapshot(
        max_age_frames=max_age_frames,
        minimum_assignment_score=minimum_assignment_score,
        maximum_center_distance_ratio=maximum_center_distance_ratio,
        hard_class_gate=hard_class_gate,
        bootstrap_primary_only=bootstrap_primary_only,
        minimum_mask_area=minimum_mask_area,
        depth_erosion_pixels=depth_erosion_pixels,
        mask_iou_weight=mask_iou_weight,
        flow_iou_weight=flow_iou_weight,
        box_iou_weight=box_iou_weight,
        class_weight=class_weight,
        depth_weight=depth_weight,
        evidence_policy_seed=evidence_policy_seed,
        depth_check_fraction=depth_check_fraction,
    )
    config_sha256 = hash_payload(config)
    stage_root = loaded.run_root / "03_object_tracking" / f"config_{config_sha256[:16]}"
    manifests, references = [], []
    for manifest, manifest_reference in zip(loaded.manifests, loaded.manifest_refs):
        package = _build_video_package(
            loaded=loaded,
            manifest=manifest,
            manifest_reference=manifest_reference,
            config=config,
            config_sha256=config_sha256,
            stage_root=stage_root,
        )
        relative_path = Path("videos") / f"{manifest.video_id}.tracking.json"
        path = stage_root / relative_path
        digest, byte_size = write_contract(path, package)
        manifests.append(package)
        references.append(
            ArtifactRef(
                artifact_id=f"tracking_package:{manifest.video_id}",
                relative_path=relative_path.as_posix(),
                sha256=digest,
                byte_size=byte_size,
                media_type="application/vnd.cauvid.tracking-package+json",
            )
        )
    store = TrackingStore(
        run_id=loaded.store.run_id,
        source_neural_evidence_store_sha256=sha256_file(loaded.store_path),
        config=config,
        config_sha256=config_sha256,
        video_ids=loaded.store.video_ids,
        video_tracking=tuple(references),
    )
    store_path = stage_root / "tracking_store.json"
    write_contract(store_path, store)
    return Step3Result(
        store=store,
        video_manifests=tuple(manifests),
        stage_root=stage_root,
        store_path=store_path,
    )
