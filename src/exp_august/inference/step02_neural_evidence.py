"""Target Step 2: frame-local neural evidence extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from src.exp_august.contracts import (
    ArtifactRef,
    BoundingBoxXYXY,
    CueDescriptor,
    CueFamily,
    CueStatus,
    DepthObservation,
    DepthRepresentation,
    DepthUnit,
    DetectionObservation,
    DetectionTier,
    FrameEvidence,
    FlowDirection,
    FlowObservation,
    MaskObservation,
    NeuralEvidenceStore,
    Step2ConfigSnapshot,
    ToolVersion,
    VideoManifest,
    VideoEvidenceManifest,
)
from src.exp_august.contracts.codec import hash_payload, sha256_file, write_contract
from src.exp_august.inference.frames import (
    CanonicalFrame,
    CanonicalFrameProvider,
    load_init_bundle,
)
from src.exp_august.inference.artifact_io import (
    write_mask_artifact,
    write_npz_artifact,
)
from src.exp_august.inference.depth_backend import (
    DisabledDepthBackend,
)
from src.exp_august.inference.flow_backend import (
    DisabledFlowBackend,
    DirectionalFlowOutput,
)
from src.exp_august.inference.mask_backend import (
    DisabledMaskBackend,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ObjectCandidate:
    bbox_xyxy: tuple[float, float, float, float]
    class_name: str
    confidence: float
    tier: DetectionTier


class ObjectEvidenceBackend(Protocol):
    backend_name: str
    model_name: str
    model_id: str
    available: bool
    unavailable_reason: str | None
    tool_versions: tuple[ToolVersion, ...]

    def warmup(self) -> None: ...

    def predict_batch(
        self, frames: Sequence[CanonicalFrame]
    ) -> tuple[tuple[ObjectCandidate, ...], ...]: ...

    def teardown(self) -> None: ...


class DisabledObjectBackend:
    backend_name = "disabled"
    model_name = "none"
    model_id = "none"
    available = False
    unavailable_reason = "object backend explicitly disabled"
    tool_versions: tuple[ToolVersion, ...] = ()

    def warmup(self) -> None:
        return None

    def predict_batch(
        self, frames: Sequence[CanonicalFrame]
    ) -> tuple[tuple[ObjectCandidate, ...], ...]:
        return tuple(() for _ in frames)

    def teardown(self) -> None:
        return None


class YoloWorldEvidenceBackend:
    """High-recall YOLO-World extraction without a persistent JPEG frame cache."""

    backend_name = "yolo_world"
    available = True
    unavailable_reason = None

    def __init__(
        self,
        *,
        model_name: str,
        classes: Sequence[str],
        primary_confidence: float,
        candidate_confidence: float,
        nms_iou: float,
        inference_size: int,
        device: str = "auto",
        allow_model_download: bool = False,
    ) -> None:
        import torch
        import ultralytics

        candidate_path = Path(model_name).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = PROJECT_ROOT / candidate_path
        if candidate_path.is_file():
            self.model_source = str(candidate_path.resolve())
            model_hash = sha256_file(candidate_path)
            self.model_name = str(model_name)
            self.model_id = f"{candidate_path.name}@sha256:{model_hash[:16]}"
        elif allow_model_download:
            self.model_source = model_name
            self.model_name = model_name
            self.model_id = f"{model_name}@download-resolved-at-runtime"
        else:
            raise FileNotFoundError(
                f"YOLO model is not local: {candidate_path}; "
                "pass allow_model_download=True to permit runtime resolution"
            )
        if not 0.0 <= candidate_confidence <= primary_confidence <= 1.0:
            raise ValueError("expected 0 <= candidate confidence <= primary confidence <= 1")
        self.classes = tuple(str(value) for value in classes)
        self.primary_confidence = float(primary_confidence)
        self.candidate_confidence = float(candidate_confidence)
        self.nms_iou = float(nms_iou)
        self.inference_size = int(inference_size)
        self.device = (
            "cuda:0" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        self.tool_versions = (
            ToolVersion(name="ultralytics", version=ultralytics.__version__),
            ToolVersion(name="torch", version=torch.__version__),
        )
        self._model = None

    def warmup(self) -> None:
        from ultralytics import YOLOWorld

        self._model = YOLOWorld(self.model_source)
        if self.classes:
            self._model.set_classes(list(self.classes))

    def predict_batch(
        self, frames: Sequence[CanonicalFrame]
    ) -> tuple[tuple[ObjectCandidate, ...], ...]:
        if self._model is None:
            self.warmup()
        results = self._model.predict(
            source=[frame.image_bgr for frame in frames],
            conf=self.candidate_confidence,
            iou=self.nms_iou,
            imgsz=self.inference_size,
            device=self.device,
            verbose=False,
            stream=False,
        )
        batch: list[tuple[ObjectCandidate, ...]] = []
        for frame, result in zip(frames, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                batch.append(())
                continue
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            class_indices = boxes.cls.detach().cpu().tolist()
            names = result.names
            candidates: list[ObjectCandidate] = []
            height, width = frame.image_bgr.shape[:2]
            for coordinates, confidence, class_index in zip(
                xyxy, confidences, class_indices
            ):
                x1, y1, x2, y2 = (float(value) for value in coordinates[:4])
                x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
                y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                class_id = int(class_index)
                class_name = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
                score = max(0.0, min(1.0, float(confidence)))
                candidates.append(
                    ObjectCandidate(
                        bbox_xyxy=(x1, y1, x2, y2),
                        class_name=class_name,
                        confidence=score,
                        tier=(
                            DetectionTier.PRIMARY
                            if score >= self.primary_confidence
                            else DetectionTier.CANDIDATE
                        ),
                    )
                )
            candidates.sort(key=lambda item: (-item.confidence, item.class_name, item.bbox_xyxy))
            batch.append(tuple(candidates))
        if len(batch) != len(frames):
            raise RuntimeError("YOLO returned a different number of results than input frames")
        return tuple(batch)

    def teardown(self) -> None:
        self._model = None


@dataclass(frozen=True)
class Step2Result:
    store: NeuralEvidenceStore
    video_manifests: tuple[VideoEvidenceManifest, ...]
    stage_root: Path
    store_path: Path


@dataclass
class _VideoWork:
    source_reference: ArtifactRef
    manifest: VideoManifest
    detections: list[tuple[DetectionObservation, ...]]
    object_cues: list[CueDescriptor]
    masks: list[tuple[MaskObservation, ...]]
    mask_cues: list[CueDescriptor]
    forward_flows: list[FlowObservation | None]
    forward_flow_cues: list[CueDescriptor]
    backward_flows: list[FlowObservation | None]
    backward_flow_cues: list[CueDescriptor]
    depths: list[DepthObservation | None]
    depth_cues: list[CueDescriptor]


def _missing_cue(
    family: CueFamily,
    *,
    backend: str,
    reason: str,
    not_applicable: bool = False,
) -> CueDescriptor:
    return CueDescriptor(
        cue_family=family,
        status=CueStatus.NOT_APPLICABLE if not_applicable else CueStatus.UNAVAILABLE,
        backend=backend,
        reason=reason,
    )


def _detection_observations(
    video_id: str,
    frame_index: int,
    candidates: Sequence[ObjectCandidate],
    *,
    backend: ObjectEvidenceBackend,
) -> tuple[DetectionObservation, ...]:
    return tuple(
        DetectionObservation(
            detection_id=f"det:{video_id}:{frame_index}:{rank}",
            class_name=candidate.class_name,
            confidence=candidate.confidence,
            bbox=BoundingBoxXYXY(
                x1=candidate.bbox_xyxy[0],
                y1=candidate.bbox_xyxy[1],
                x2=candidate.bbox_xyxy[2],
                y2=candidate.bbox_xyxy[3],
            ),
            tier=candidate.tier,
            model_id=backend.model_id,
            rank=rank,
        )
        for rank, candidate in enumerate(candidates)
    )


def _object_cue(
    detections: Sequence[DetectionObservation],
    backend: ObjectEvidenceBackend,
) -> CueDescriptor:
    if backend.available:
        return CueDescriptor(
            cue_family=CueFamily.OBJECTS,
            status=CueStatus.AVAILABLE if detections else CueStatus.EMPTY,
            backend=backend.backend_name,
            model_id=backend.model_id,
        )
    return _missing_cue(
        CueFamily.OBJECTS,
        backend=backend.backend_name,
        reason=backend.unavailable_reason or "object backend unavailable",
    )


def _initial_work(
    source_reference: ArtifactRef,
    manifest: VideoManifest,
    *,
    object_backend,
    mask_backend,
    flow_backend,
    depth_backend,
) -> _VideoWork:
    count = manifest.canonical_frame_count
    object_missing = _missing_cue(
        CueFamily.OBJECTS,
        backend=object_backend.backend_name,
        reason=object_backend.unavailable_reason or "object backend unavailable",
    )
    mask_missing = _missing_cue(
        CueFamily.MASKS,
        backend=mask_backend.backend_name,
        reason=mask_backend.unavailable_reason or "mask backend unavailable",
    )
    depth_missing = _missing_cue(
        CueFamily.DEPTH,
        backend=depth_backend.backend_name,
        reason=depth_backend.unavailable_reason or "depth backend unavailable",
    )
    forward_cues: list[CueDescriptor] = []
    backward_cues: list[CueDescriptor] = []
    for frame_index in range(count):
        forward_cues.append(
            _missing_cue(
                CueFamily.FLOW_FORWARD,
                backend=flow_backend.backend_name,
                reason=(
                    "no next canonical frame"
                    if frame_index == count - 1
                    else flow_backend.unavailable_reason or "flow backend unavailable"
                ),
                not_applicable=frame_index == count - 1,
            )
        )
        backward_cues.append(
            _missing_cue(
                CueFamily.FLOW_BACKWARD,
                backend=flow_backend.backend_name,
                reason=(
                    "no previous canonical frame"
                    if frame_index == 0
                    else flow_backend.unavailable_reason or "flow backend unavailable"
                ),
                not_applicable=frame_index == 0,
            )
        )
    return _VideoWork(
        source_reference=source_reference,
        manifest=manifest,
        detections=[() for _ in range(count)],
        object_cues=[object_missing for _ in range(count)],
        masks=[() for _ in range(count)],
        mask_cues=[mask_missing for _ in range(count)],
        forward_flows=[None for _ in range(count)],
        forward_flow_cues=forward_cues,
        backward_flows=[None for _ in range(count)],
        backward_flow_cues=backward_cues,
        depths=[None for _ in range(count)],
        depth_cues=[depth_missing for _ in range(count)],
    )


def _run_object_pass(
    works: Sequence[_VideoWork],
    backend: ObjectEvidenceBackend,
    *,
    batch_size: int,
) -> None:
    if not backend.available:
        return
    backend.warmup()
    try:
        for work in works:
            provider = CanonicalFrameProvider(work.manifest, verify_source_hash=False)
            for frame_batch in provider.iter_batches(batch_size):
                candidates_batch = backend.predict_batch(frame_batch)
                if len(candidates_batch) != len(frame_batch):
                    raise RuntimeError("object backend returned the wrong batch length")
                for frame, candidates in zip(frame_batch, candidates_batch):
                    detections = _detection_observations(
                        frame.video_id,
                        frame.frame_index,
                        candidates,
                        backend=backend,
                    )
                    work.detections[frame.frame_index] = detections
                    work.object_cues[frame.frame_index] = _object_cue(detections, backend)
    finally:
        backend.teardown()


def _run_mask_pass(
    works: Sequence[_VideoWork],
    backend,
    *,
    stage_root: Path,
) -> None:
    if not backend.available:
        return
    backend.warmup()
    try:
        for work in works:
            provider = CanonicalFrameProvider(work.manifest, verify_source_hash=False)
            for frame in provider.iter_frames():
                detections = work.detections[frame.frame_index]
                prompts = tuple(
                    detection
                    for detection in detections
                    if backend.prompt_candidates
                    or detection.tier == DetectionTier.PRIMARY
                )
                if not prompts:
                    work.mask_cues[frame.frame_index] = _missing_cue(
                        CueFamily.MASKS,
                        backend=backend.backend_name,
                        reason="no eligible object prompts in this frame",
                        not_applicable=True,
                    )
                    continue
                outputs = backend.predict_frame(frame, detections)
                if len(outputs) != len(prompts):
                    raise RuntimeError("mask backend did not return one mask per eligible prompt")
                observations: list[MaskObservation] = []
                references: list[ArtifactRef] = []
                for rank, output in enumerate(outputs):
                    relative_path = (
                        Path("artifacts")
                        / "masks"
                        / frame.video_id
                        / f"frame_{frame.frame_index:06d}"
                        / f"mask_{rank:04d}.png"
                    )
                    proposal_id = f"mask:{frame.video_id}:{frame.frame_index}:{rank}"
                    reference = write_mask_artifact(
                        stage_root=stage_root,
                        relative_path=relative_path,
                        artifact_id=proposal_id,
                        mask=output.mask,
                    )
                    references.append(reference)
                    observations.append(
                        MaskObservation(
                            proposal_id=proposal_id,
                            prompt_detection_id=output.prompt_detection_id,
                            confidence=output.confidence,
                            mask_ref=reference,
                            model_id=backend.model_id,
                            area_pixels=int(np.count_nonzero(output.mask)),
                        )
                    )
                work.masks[frame.frame_index] = tuple(observations)
                work.mask_cues[frame.frame_index] = CueDescriptor(
                    cue_family=CueFamily.MASKS,
                    status=CueStatus.AVAILABLE,
                    backend=backend.backend_name,
                    model_id=backend.model_id,
                    artifact_refs=tuple(references),
                )
    finally:
        backend.teardown()


def _flow_statistics(
    output: DirectionalFlowOutput,
) -> tuple[float, float | None, float | None]:
    valid_fraction = float(np.mean(output.consistency_valid))
    errors = output.fb_error[output.domain_valid]
    if not errors.size:
        return valid_fraction, None, None
    return (
        valid_fraction,
        float(np.median(errors)),
        float(np.percentile(errors, 95)),
    )


def _persist_flow(
    *,
    stage_root: Path,
    video_id: str,
    source: CanonicalFrame,
    target: CanonicalFrame,
    direction: FlowDirection,
    output: DirectionalFlowOutput,
    backend,
) -> tuple[FlowObservation, CueDescriptor]:
    direction_value = direction.value
    relative_path = (
        Path("artifacts")
        / "flow"
        / video_id
        / (
            f"frame_{source.frame_index:06d}_to_{target.frame_index:06d}_"
            f"{direction_value}.npz"
        )
    )
    artifact_id = (
        f"flow:{direction_value}:{video_id}:"
        f"{source.frame_index}:{target.frame_index}"
    )
    reference = write_npz_artifact(
        stage_root=stage_root,
        relative_path=relative_path,
        artifact_id=artifact_id,
        arrays={
            "flow": output.flow.astype(np.float32, copy=False),
            "domain_valid": output.domain_valid.astype(np.uint8),
            "consistency_valid": output.consistency_valid.astype(np.uint8),
            "fb_error": output.fb_error.astype(np.float32, copy=False),
        },
        primary_shape=tuple(int(value) for value in output.flow.shape),
        dtype_description=(
            "npz[flow=float32,domain_valid=uint8,"
            "consistency_valid=uint8,fb_error=float32]"
        ),
        media_type="application/vnd.cauvid.optical-flow+npz",
    )
    valid_fraction, median_error, p95_error = _flow_statistics(output)
    observation = FlowObservation(
        direction=direction,
        source_frame_index=source.frame_index,
        target_frame_index=target.frame_index,
        source_timestamp_s=source.timestamp_s,
        target_timestamp_s=target.timestamp_s,
        field_ref=reference,
        valid_fraction=valid_fraction,
        median_fb_error_px=median_error,
        p95_fb_error_px=p95_error,
    )
    cue = CueDescriptor(
        cue_family=(
            CueFamily.FLOW_FORWARD
            if direction == FlowDirection.FORWARD
            else CueFamily.FLOW_BACKWARD
        ),
        status=CueStatus.AVAILABLE,
        backend=backend.backend_name,
        model_id=backend.model_id,
        artifact_refs=(reference,),
    )
    return observation, cue


def _run_flow_pass(
    works: Sequence[_VideoWork],
    backend,
    *,
    stage_root: Path,
) -> None:
    if not backend.available:
        return
    backend.warmup()
    try:
        for work in works:
            provider = CanonicalFrameProvider(work.manifest, verify_source_hash=False)
            previous: CanonicalFrame | None = None
            for current in provider.iter_frames():
                if previous is None:
                    previous = current
                    continue
                pair = backend.predict_pair(previous, current)
                forward, forward_cue = _persist_flow(
                    stage_root=stage_root,
                    video_id=work.manifest.video_id,
                    source=previous,
                    target=current,
                    direction=FlowDirection.FORWARD,
                    output=pair.forward,
                    backend=backend,
                )
                backward, backward_cue = _persist_flow(
                    stage_root=stage_root,
                    video_id=work.manifest.video_id,
                    source=current,
                    target=previous,
                    direction=FlowDirection.BACKWARD,
                    output=pair.backward,
                    backend=backend,
                )
                work.forward_flows[previous.frame_index] = forward
                work.forward_flow_cues[previous.frame_index] = forward_cue
                work.backward_flows[current.frame_index] = backward
                work.backward_flow_cues[current.frame_index] = backward_cue
                previous = current
    finally:
        backend.teardown()


def _run_depth_pass(
    works: Sequence[_VideoWork],
    backend,
    *,
    stage_root: Path,
) -> None:
    if not backend.available:
        return
    backend.warmup()
    try:
        for work in works:
            provider = CanonicalFrameProvider(work.manifest, verify_source_hash=False)
            for frame in provider.iter_frames():
                output = backend.predict_frame(frame)
                valid_values = output.depth[output.valid]
                if not valid_values.size:
                    raise RuntimeError("depth backend returned no valid pixels")
                arrays = {
                    "depth": output.depth.astype(np.float32, copy=False),
                    "valid": output.valid.astype(np.uint8),
                }
                if output.confidence is not None:
                    arrays["confidence"] = output.confidence.astype(np.float32, copy=False)
                relative_path = (
                    Path("artifacts")
                    / "depth"
                    / frame.video_id
                    / f"frame_{frame.frame_index:06d}.npz"
                )
                artifact_id = f"depth:{frame.video_id}:{frame.frame_index}"
                reference = write_npz_artifact(
                    stage_root=stage_root,
                    relative_path=relative_path,
                    artifact_id=artifact_id,
                    arrays=arrays,
                    primary_shape=tuple(int(value) for value in output.depth.shape),
                    dtype_description=(
                        "npz[depth=float32,valid=uint8"
                        + (",confidence=float32]" if output.confidence is not None else "]")
                    ),
                    media_type="application/vnd.cauvid.depth+npz",
                )
                unit = (
                    DepthUnit.METER
                    if output.representation == DepthRepresentation.METRIC
                    else DepthUnit.RELATIVE_UNIT
                )
                observation = DepthObservation(
                    representation=output.representation,
                    unit=unit,
                    field_ref=reference,
                    valid_fraction=float(np.mean(output.valid)),
                    minimum=float(np.min(valid_values)),
                    median=float(np.median(valid_values)),
                    maximum=float(np.max(valid_values)),
                    has_confidence=output.confidence is not None,
                    model_id=backend.model_id,
                )
                work.depths[frame.frame_index] = observation
                work.depth_cues[frame.frame_index] = CueDescriptor(
                    cue_family=CueFamily.DEPTH,
                    status=CueStatus.AVAILABLE,
                    backend=backend.backend_name,
                    model_id=backend.model_id,
                    artifact_refs=(reference,),
                )
    finally:
        backend.teardown()


def _finalize_frames(work: _VideoWork) -> tuple[FrameEvidence, ...]:
    size = work.manifest.image_size
    return tuple(
        FrameEvidence(
            frame_index=record.frame_index,
            timestamp_s=record.timestamp_s,
            source_frame_index=record.source_frame_index,
            source_timestamp_s=record.source_timestamp_s,
            image_size=size,
            object_cue=work.object_cues[index],
            detections=work.detections[index],
            mask_cue=work.mask_cues[index],
            masks=work.masks[index],
            forward_flow_cue=work.forward_flow_cues[index],
            forward_flow=work.forward_flows[index],
            backward_flow_cue=work.backward_flow_cues[index],
            backward_flow=work.backward_flows[index],
            depth_cue=work.depth_cues[index],
            depth=work.depths[index],
        )
        for index, record in enumerate(work.manifest.frames)
    )


def _unique_tool_versions(*backends) -> tuple[ToolVersion, ...]:
    values = [ToolVersion(name="opencv", version=cv2.__version__)]
    for backend in backends:
        values.extend(backend.tool_versions)
    unique: dict[tuple[str, str], ToolVersion] = {}
    for value in values:
        unique[(value.name, value.version)] = value
    return tuple(unique[key] for key in sorted(unique))


def run_step2(
    *,
    init_bundle_path: Path | str,
    object_backend: ObjectEvidenceBackend,
    object_classes: Sequence[str],
    primary_confidence: float,
    candidate_confidence: float,
    nms_iou: float,
    inference_size: int,
    batch_size: int,
    device: str,
    mask_backend=None,
    flow_backend=None,
    depth_backend=None,
    verify_source_hash: bool = True,
) -> Step2Result:
    """Extract independent frame evidence with immutable dense artifacts."""

    loaded = load_init_bundle(init_bundle_path, verify_artifacts=True)
    mask_backend = mask_backend or DisabledMaskBackend()
    flow_backend = flow_backend or DisabledFlowBackend()
    depth_backend = depth_backend or DisabledDepthBackend()
    config = Step2ConfigSnapshot(
        object_backend=object_backend.backend_name,
        object_model=object_backend.model_id,
        object_classes=tuple(object_classes),
        primary_confidence=float(primary_confidence),
        candidate_confidence=float(candidate_confidence),
        nms_iou=float(nms_iou),
        inference_size=int(inference_size),
        batch_size=int(batch_size),
        device=device,
        masks_backend=mask_backend.backend_name,
        masks_model=mask_backend.model_id,
        mask_prompt_candidates=mask_backend.prompt_candidates,
        flow_backend=flow_backend.backend_name,
        flow_model=flow_backend.model_id,
        flow_consistency_threshold_px=flow_backend.consistency_threshold_px,
        depth_backend=depth_backend.backend_name,
        depth_model=depth_backend.model_id,
        depth_process_resolution=depth_backend.process_resolution,
        depth_representation=depth_backend.representation,
    )
    config_sha256 = hash_payload(config)
    # Step 1 identifies the immutable input/timeline run. Step 2 has its own
    # configuration namespace so alternative models or thresholds never
    # overwrite evidence produced for the same source video.
    stage_root = (
        loaded.run_root
        / "02_neural_evidence"
        / f"config_{config_sha256[:16]}"
    )
    works = [
        _initial_work(
            source_reference,
            manifest,
            object_backend=object_backend,
            mask_backend=mask_backend,
            flow_backend=flow_backend,
            depth_backend=depth_backend,
        )
        for source_reference, manifest in zip(
            loaded.bundle.video_manifests, loaded.manifests
        )
    ]
    for work in works:
        CanonicalFrameProvider(
            work.manifest,
            verify_source_hash=verify_source_hash,
        )
    _run_object_pass(works, object_backend, batch_size=config.batch_size)
    _run_mask_pass(works, mask_backend, stage_root=stage_root)
    _run_flow_pass(works, flow_backend, stage_root=stage_root)
    _run_depth_pass(works, depth_backend, stage_root=stage_root)

    video_manifests: list[VideoEvidenceManifest] = []
    references: list[ArtifactRef] = []
    tool_versions = _unique_tool_versions(
        object_backend,
        mask_backend,
        flow_backend,
        depth_backend,
    )
    for work in works:
        evidence_frames = _finalize_frames(work)
        video_evidence = VideoEvidenceManifest(
            run_id=loaded.bundle.run_id,
            video_id=work.manifest.video_id,
            source_manifest=work.source_reference,
            source_manifest_sha256=work.source_reference.sha256,
            config_sha256=config_sha256,
            canonical_fps=work.manifest.canonical_fps,
            image_size=work.manifest.image_size,
            frame_count=len(evidence_frames),
            frames=tuple(evidence_frames),
            tool_versions=tool_versions,
        )
        relative_path = Path("videos") / f"{work.manifest.video_id}.evidence.json"
        evidence_path = stage_root / relative_path
        evidence_sha256, evidence_size = write_contract(evidence_path, video_evidence)
        video_manifests.append(video_evidence)
        references.append(
            ArtifactRef(
                artifact_id=f"neural_evidence:{work.manifest.video_id}",
                relative_path=relative_path.as_posix(),
                sha256=evidence_sha256,
                byte_size=evidence_size,
                media_type="application/vnd.cauvid.neural-evidence+json",
            )
        )
    store = NeuralEvidenceStore(
        run_id=loaded.bundle.run_id,
        source_init_bundle_sha256=sha256_file(loaded.bundle_path),
        config=config,
        config_sha256=config_sha256,
        video_ids=loaded.bundle.video_ids,
        video_evidence=tuple(references),
    )
    store_path = stage_root / "neural_evidence_store.json"
    write_contract(store_path, store)
    return Step2Result(
        store=store,
        video_manifests=tuple(video_manifests),
        stage_root=stage_root,
        store_path=store_path,
    )
