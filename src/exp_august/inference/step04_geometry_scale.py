"""Target Step 4: lift mask tracks into observable monocular geometry.

The stage deliberately separates camera-centric relative geometry from metric
claims.  A DA3 relative-depth field can support 3D shape and cross-frame
geometry, but it cannot by itself establish a meters conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.exp_august.contracts import (
    ArtifactLink,
    ArtifactOwner,
    ArtifactRef,
    CameraIntrinsicsHypothesis,
    CameraMotionEstimate,
    CoordinateSpace,
    DepthRepresentation,
    DepthUnit,
    EvidenceRole,
    GeometryObservation,
    GeometryStore,
    GeometryValidationSummary,
    GroundPlaneEstimate,
    ImageSize,
    NonNegativeVector3D,
    ObjectGeometryTrack,
    Observability,
    PixelCoordinate,
    PointDistribution3D,
    RelativeCameraPose,
    ScaleHypothesis,
    Step4ConfigSnapshot,
    Step4InputSnapshot,
    ToolVersion,
    TrackingStore,
    UnavailableGeometryObservation,
    Vector3D,
    VideoEvidenceManifest,
    VideoGeometryManifest,
    VideoTrackingManifest,
)
from src.exp_august.contracts.codec import (
    hash_payload,
    read_contract,
    sha256_file,
    write_contract,
)


@dataclass(frozen=True)
class Step4Result:
    store: GeometryStore
    video_manifests: tuple[VideoGeometryManifest, ...]
    stage_root: Path
    store_path: Path


@dataclass(frozen=True)
class _LoadedStep3:
    store_path: Path
    stage_root: Path
    run_root: Path
    store: TrackingStore
    manifests: tuple[VideoTrackingManifest, ...]
    manifest_refs: tuple[ArtifactRef, ...]


class _ArtifactResolver:
    def __init__(self, *, run_root: Path, step2_root: Path, step3_root: Path) -> None:
        self.run_root = run_root
        self.step2_root = step2_root
        self.step3_root = step3_root
        self._verified: set[tuple[ArtifactOwner, str, str]] = set()

    @property
    def verified_count(self) -> int:
        return len(self._verified)

    def path(self, link: ArtifactLink) -> Path:
        roots = {
            ArtifactOwner.STEP1_INIT: self.run_root / "01_init",
            ArtifactOwner.STEP2_NEURAL_EVIDENCE: self.step2_root,
            ArtifactOwner.STEP3_OBJECT_TRACKING: self.step3_root,
        }
        if link.owner not in roots:
            raise RuntimeError(f"Step 4 cannot resolve artifact owner {link.owner.value}")
        path = roots[link.owner] / link.artifact.relative_path
        key = (link.owner, link.artifact.relative_path, link.artifact.sha256)
        if key not in self._verified:
            if not path.is_file():
                raise RuntimeError(f"Step 4 source artifact is missing: {path}")
            if path.stat().st_size != link.artifact.byte_size:
                raise RuntimeError(f"Step 4 source artifact size mismatch: {path}")
            if sha256_file(path) != link.artifact.sha256:
                raise RuntimeError(f"Step 4 source artifact hash mismatch: {path}")
            self._verified.add(key)
        return path


def _file_reference(*, path: Path, stage_root: Path, artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        relative_path=path.relative_to(stage_root).as_posix(),
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        media_type="application/json",
        coordinate_space=None,
    )


def _step3_link(reference: ArtifactRef) -> ArtifactLink:
    return ArtifactLink(owner=ArtifactOwner.STEP3_OBJECT_TRACKING, artifact=reference)


def _load_step3(store_path: Path | str) -> _LoadedStep3:
    resolved = Path(store_path).expanduser().resolve()
    store = read_contract(resolved, TrackingStore)
    stage_root = resolved.parent
    if stage_root.parent.name != "03_object_tracking":
        raise RuntimeError("Step 3 store must live inside 03_object_tracking/config_<hash>")
    run_root = stage_root.parent.parent
    manifests = []
    for video_id, reference in zip(store.video_ids, store.video_tracking):
        path = stage_root / reference.relative_path
        if not path.is_file() or path.stat().st_size != reference.byte_size:
            raise RuntimeError(f"Step 3 tracking manifest is missing or truncated: {path}")
        if sha256_file(path) != reference.sha256:
            raise RuntimeError(f"Step 3 tracking manifest failed integrity check: {path}")
        manifest = read_contract(path, VideoTrackingManifest)
        if manifest.video_id != video_id or manifest.run_id != store.run_id:
            raise RuntimeError(f"Step 3 tracking identity mismatch: {path}")
        manifests.append(manifest)
    return _LoadedStep3(
        store_path=resolved,
        stage_root=stage_root,
        run_root=run_root,
        store=store,
        manifests=tuple(manifests),
        manifest_refs=store.video_tracking,
    )


def _intrinsics(
    *, image_size: ImageSize, config: Step4ConfigSnapshot
) -> CameraIntrinsicsHypothesis:
    width, height = image_size.width, image_size.height
    if config.intrinsics_mode == "provided_cli":
        fx = float(config.camera_fx_px)
        fy = float(config.camera_fy_px)
        cx = float(config.camera_cx_px) if config.camera_cx_px is not None else width / 2.0
        cy = float(config.camera_cy_px) if config.camera_cy_px is not None else height / 2.0
        payload = {"source": "provided_cli", "fx": fx, "fy": fy, "cx": cx, "cy": cy}
        return CameraIntrinsicsHypothesis(
            intrinsics_id=f"intrinsics:{hash_payload(payload)[:16]}",
            source="provided_cli",
            image_size=image_size,
            fx_px=fx,
            fy_px=fy,
            cx_px=cx,
            cy_px=cy,
            assumption_driven=False,
            validated=False,
        )
    fov = config.horizontal_fov_degrees
    fx = width / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    payload = {
        "source": "horizontal_fov_prior",
        "image_size": image_size.model_dump(mode="json"),
        "fov": fov,
        "interval": [config.horizontal_fov_min_degrees, config.horizontal_fov_max_degrees],
    }
    return CameraIntrinsicsHypothesis(
        intrinsics_id=f"intrinsics:{hash_payload(payload)[:16]}",
        source="horizontal_fov_prior",
        image_size=image_size,
        fx_px=float(fx),
        fy_px=float(fx),
        cx_px=width / 2.0,
        cy_px=height / 2.0,
        horizontal_fov_deg=fov,
        horizontal_fov_interval_deg=(
            config.horizontal_fov_min_degrees,
            config.horizontal_fov_max_degrees,
        ),
        assumption_driven=True,
        validated=False,
    )


def _camera_matrix(intrinsics: CameraIntrinsicsHypothesis) -> np.ndarray:
    return np.asarray(
        [
            [intrinsics.fx_px, 0.0, intrinsics.cx_px],
            [0.0, intrinsics.fy_px, intrinsics.cy_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _read_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Step 4 could not decode mask: {path}")
    mask = image > 0
    if mask.shape != expected_shape:
        raise RuntimeError(f"Step 4 mask shape mismatch: {path}")
    return mask


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except Exception as error:
        raise RuntimeError(f"Step 4 could not decode NPZ artifact: {path}") from error


def _selected_candidates(package: VideoTrackingManifest) -> dict[str, object]:
    return {candidate.candidate_id: candidate for candidate in package.mask_candidate_bank}


def _reserved_artifact_ids(
    package: VideoTrackingManifest, *, consumer: str
) -> set[str]:
    return {
        assignment.artifact.artifact.artifact_id
        for assignment in package.evidence_use_plan.assignments
        if assignment.role == EvidenceRole.CHECK_ONLY
        and consumer in assignment.prohibited_optimizers
        and assignment.artifact is not None
    }


def _support_from_bbox(
    *, bbox, shape: tuple[int, int], inset_fraction: float
) -> np.ndarray:
    height, width = shape
    x1 = max(0, min(width, int(np.floor(bbox.x1))))
    y1 = max(0, min(height, int(np.floor(bbox.y1))))
    x2 = max(0, min(width, int(np.ceil(bbox.x2))))
    y2 = max(0, min(height, int(np.ceil(bbox.y2))))
    inset_x = int(np.floor((x2 - x1) * inset_fraction))
    inset_y = int(np.floor((y2 - y1) * inset_fraction))
    support = np.zeros(shape, dtype=bool)
    if x1 + inset_x < x2 - inset_x and y1 + inset_y < y2 - inset_y:
        support[y1 + inset_y : y2 - inset_y, x1 + inset_x : x2 - inset_x] = True
    return support


def _vector(values: np.ndarray) -> Vector3D:
    return Vector3D(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _nonnegative_vector(values: np.ndarray) -> NonNegativeVector3D:
    return NonNegativeVector3D(
        x=float(max(0.0, values[0])),
        y=float(max(0.0, values[1])),
        z=float(max(0.0, values[2])),
    )


def _scale_hypotheses(package: VideoTrackingManifest) -> tuple[ScaleHypothesis, ...]:
    representations = {
        observation.depth_descriptor.representation
        for track in package.tracks
        for observation in track.observations
        if observation.depth_descriptor is not None
    }
    hypotheses = []
    if DepthRepresentation.METRIC in representations:
        hypotheses.append(
            ScaleHypothesis(
                scale_id="scale:metric-depth",
                rank=len(hypotheses) + 1,
                source="metric_depth",
                observability=Observability.METRIC,
                depth_representation=DepthRepresentation.METRIC,
                scale_to_meters=1.0,
                scale_interval_to_meters=(1.0, 1.0),
                evidence=("depth_backend_declares_metric",),
                limitations=("metric_depth_requires_external_backend_validation",),
            )
        )
    if DepthRepresentation.RELATIVE in representations:
        hypotheses.append(
            ScaleHypothesis(
                scale_id="scale:relative-monocular",
                rank=len(hypotheses) + 1,
                source="relative_monocular_depth",
                observability=Observability.RELATIVE,
                depth_representation=DepthRepresentation.RELATIVE,
                evidence=("relative_depth_field", "camera_intrinsics_hypothesis"),
                limitations=(
                    "no_supported_meters_conversion",
                    "per_frame_relative_depth_scale_may_drift",
                ),
            )
        )
    if not hypotheses:
        hypotheses.append(
            ScaleHypothesis(
                scale_id="scale:unobservable",
                rank=1,
                source="no_depth",
                observability=Observability.UNOBSERVABLE,
                depth_representation=DepthRepresentation.RELATIVE,
                evidence=(),
                limitations=("no_usable_depth_observation",),
            )
        )
    return tuple(hypotheses)


def _scale_id(
    representation: DepthRepresentation, hypotheses: tuple[ScaleHypothesis, ...]
) -> str:
    for hypothesis in hypotheses:
        if hypothesis.depth_representation == representation and hypothesis.source != "no_depth":
            return hypothesis.scale_id
    raise RuntimeError(f"no scale hypothesis for {representation.value} depth")


def _geometry_observation(
    *,
    track_id: str,
    observation,
    candidate,
    resolver: _ArtifactResolver,
    intrinsics: CameraIntrinsicsHypothesis,
    scale_hypotheses: tuple[ScaleHypothesis, ...],
    config: Step4ConfigSnapshot,
) -> GeometryObservation | UnavailableGeometryObservation:
    source_links = tuple(
        link
        for link in (
            candidate.mask if candidate is not None else None,
            observation.depth,
        )
        if link is not None
    )
    unavailable = lambda reason: UnavailableGeometryObservation(
        unavailable_id=f"geometry-unavailable:{track_id}:{observation.frame_index}",
        track_id=track_id,
        frame_index=observation.frame_index,
        timestamp_s=observation.timestamp_s,
        detection_id=observation.detection_id,
        reason=reason,
        source_artifacts=source_links,
    )
    if observation.depth is None or observation.depth_descriptor is None:
        return unavailable("depth_unavailable")
    depth_path = resolver.path(observation.depth)
    payload = _read_npz(depth_path)
    if "depth" not in payload or "valid" not in payload:
        return unavailable("depth_payload_missing_required_arrays")
    depth = np.asarray(payload["depth"], dtype=np.float32)
    valid = np.asarray(payload["valid"], dtype=bool) & np.isfinite(depth) & (depth > 0.0)
    if depth.ndim != 2 or valid.shape != depth.shape:
        return unavailable("depth_shape_invalid")

    support_source = "inner_box"
    support = None
    if candidate is not None and candidate.mask is not None:
        support = _read_mask(resolver.path(candidate.mask), depth.shape)
        if config.support_erosion_pixels > 0:
            size = 2 * config.support_erosion_pixels + 1
            eroded = cv2.erode(
                support.astype(np.uint8), np.ones((size, size), dtype=np.uint8)
            ).astype(bool)
            if np.any(eroded):
                support = eroded
        support_source = "eroded_mask"
    if support is None:
        support = _support_from_bbox(
            bbox=observation.bbox,
            shape=depth.shape,
            inset_fraction=config.bbox_inset_fraction,
        )
    support_count = int(np.count_nonzero(support))
    if support_count < config.minimum_support_pixels:
        return unavailable("insufficient_spatial_support")
    selected = support & valid
    valid_count = int(np.count_nonzero(selected))
    valid_fraction = valid_count / support_count
    if (
        valid_count < config.minimum_support_pixels
        or valid_fraction < config.minimum_valid_depth_fraction
    ):
        return unavailable("insufficient_valid_depth_support")

    rows, columns = np.nonzero(selected)
    z = depth[selected].astype(np.float64)
    x = (columns.astype(np.float64) - intrinsics.cx_px) * z / intrinsics.fx_px
    y = (rows.astype(np.float64) - intrinsics.cy_px) * z / intrinsics.fy_px
    points = np.column_stack((x, y, z))
    q25 = np.percentile(points, 25.0, axis=0)
    median = np.median(points, axis=0)
    q75 = np.percentile(points, 75.0, axis=0)
    mad = np.median(np.abs(points - median), axis=0)
    pixel_centroid = np.asarray(
        [float(np.median(columns)), float(np.median(rows))], dtype=np.float64
    )
    projected_u = intrinsics.fx_px * points[:, 0] / points[:, 2] + intrinsics.cx_px
    projected_v = intrinsics.fy_px * points[:, 1] / points[:, 2] + intrinsics.cy_px
    reprojection_error = float(
        np.median(
            np.hypot(
                projected_u - columns.astype(np.float64),
                projected_v - rows.astype(np.float64),
            )
        )
    )
    representation = observation.depth_descriptor.representation
    unit = observation.depth_descriptor.unit
    confidence_median = None
    if "confidence" in payload:
        confidence = np.asarray(payload["confidence"], dtype=np.float32)
        if confidence.shape == depth.shape:
            finite_confidence = confidence[selected & np.isfinite(confidence)]
            if finite_confidence.size:
                confidence_median = float(np.clip(np.median(finite_confidence), 0.0, 1.0))
    passed = reprojection_error <= config.maximum_median_reprojection_error_px
    notes = [
        "masked_depth_backprojection",
        "relative_coordinates_not_meters"
        if representation == DepthRepresentation.RELATIVE
        else "metric_depth_coordinates",
    ]
    if intrinsics.assumption_driven:
        notes.append("intrinsics_from_frozen_fov_prior")
    if not passed:
        notes.append("median_reprojection_error_exceeds_threshold")
    return GeometryObservation(
        observation_id=f"geometry:{track_id}:{observation.frame_index}",
        track_id=track_id,
        frame_index=observation.frame_index,
        timestamp_s=observation.timestamp_s,
        detection_id=observation.detection_id,
        class_name=observation.class_name,
        bbox=observation.bbox,
        coordinate_space=CoordinateSpace.CAMERA_3D,
        coordinate_unit=unit,
        depth_representation=representation,
        intrinsics_id=intrinsics.intrinsics_id,
        scale_id=_scale_id(representation, scale_hypotheses),
        support_source=support_source,
        support_pixel_count=support_count,
        valid_depth_pixel_count=valid_count,
        valid_depth_fraction=valid_fraction,
        confidence_median=confidence_median,
        pixel_centroid=PixelCoordinate(u=pixel_centroid[0], v=pixel_centroid[1]),
        points=PointDistribution3D(
            q25=_vector(q25),
            median=_vector(median),
            q75=_vector(q75),
            mad=_nonnegative_vector(mad),
        ),
        median_reprojection_error_px=reprojection_error,
        source_artifacts=source_links,
        validation_passed=passed,
        validation_notes=tuple(notes),
    )


def _foreground_masks(
    *,
    package: VideoTrackingManifest,
    resolver: _ArtifactResolver,
) -> dict[int, np.ndarray]:
    candidates = _selected_candidates(package)
    masks: dict[int, np.ndarray] = {}
    shape = (package.image_size.height, package.image_size.width)
    for track in package.tracks:
        for observation in track.observations:
            candidate = candidates.get(observation.selected_mask_candidate_id)
            if candidate is None or candidate.mask is None:
                continue
            mask = _read_mask(resolver.path(candidate.mask), shape)
            masks.setdefault(observation.frame_index, np.zeros(shape, dtype=bool))
            masks[observation.frame_index] |= mask
    return masks


def _epipolar_residuals_px(
    essential: np.ndarray,
    camera_matrix: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
) -> np.ndarray:
    inverse = np.linalg.inv(camera_matrix)
    fundamental = inverse.T @ essential @ inverse
    ones = np.ones((points1.shape[0], 1), dtype=np.float64)
    p1 = np.hstack((points1, ones))
    p2 = np.hstack((points2, ones))
    fp1 = (fundamental @ p1.T).T
    ftp2 = (fundamental.T @ p2.T).T
    numerator = np.square(np.sum(p2 * fp1, axis=1))
    denominator = fp1[:, 0] ** 2 + fp1[:, 1] ** 2 + ftp2[:, 0] ** 2 + ftp2[:, 1] ** 2
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def _camera_motion(
    *,
    evidence: VideoEvidenceManifest,
    package: VideoTrackingManifest,
    resolver: _ArtifactResolver,
    intrinsics: CameraIntrinsicsHypothesis,
    reserved_artifact_ids: set[str],
    config: Step4ConfigSnapshot,
) -> CameraMotionEstimate:
    foreground = _foreground_masks(package=package, resolver=resolver)
    height, width = package.image_size.height, package.image_size.width
    camera_matrix = _camera_matrix(intrinsics)
    poses, failed = [], []
    for frame in evidence.frames:
        flow_observation = frame.forward_flow
        if flow_observation is None:
            continue
        target_index = flow_observation.target_frame_index
        pair = (frame.frame_index, target_index)
        if flow_observation.field_ref.artifact_id in reserved_artifact_ids:
            failed.append(pair)
            continue
        link = ArtifactLink(
            owner=ArtifactOwner.STEP2_NEURAL_EVIDENCE,
            artifact=flow_observation.field_ref,
        )
        payload = _read_npz(resolver.path(link))
        if not {"flow", "domain_valid", "consistency_valid"}.issubset(payload):
            failed.append(pair)
            continue
        flow = np.asarray(payload["flow"], dtype=np.float32)
        valid = np.asarray(payload["domain_valid"], dtype=bool) & np.asarray(
            payload["consistency_valid"], dtype=bool
        )
        if flow.shape != (height, width, 2) or valid.shape != (height, width):
            failed.append(pair)
            continue
        background = valid & np.all(np.isfinite(flow), axis=2)
        if frame.frame_index in foreground:
            background &= ~foreground[frame.frame_index]
        rows, columns = np.indices((height, width))
        sampled = (
            background
            & (rows % config.background_flow_sample_stride == 0)
            & (columns % config.background_flow_sample_stride == 0)
        )
        ys, xs = np.nonzero(sampled)
        if xs.size:
            target_x = xs.astype(np.float64) + flow[ys, xs, 0]
            target_y = ys.astype(np.float64) + flow[ys, xs, 1]
            inside = (
                (target_x >= 0.0)
                & (target_x < width)
                & (target_y >= 0.0)
                & (target_y < height)
            )
            xs, ys, target_x, target_y = (
                value[inside] for value in (xs, ys, target_x, target_y)
            )
        if xs.size < config.minimum_pose_correspondences:
            failed.append(pair)
            continue
        points1 = np.column_stack((xs, ys)).astype(np.float64)
        points2 = np.column_stack((target_x, target_y)).astype(np.float64)
        essential, initial_mask = cv2.findEssentialMat(
            points1,
            points2,
            camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=config.pose_ransac_threshold_px,
        )
        if essential is None or initial_mask is None:
            failed.append(pair)
            continue
        essential = np.asarray(essential, dtype=np.float64)
        if essential.shape != (3, 3):
            essential = essential[:3, :3]
        try:
            _, rotation, translation, pose_mask = cv2.recoverPose(
                essential,
                points1,
                points2,
                camera_matrix,
                mask=initial_mask,
            )
        except cv2.error:
            failed.append(pair)
            continue
        inliers = np.asarray(pose_mask).reshape(-1) > 0
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 5:
            failed.append(pair)
            continue
        direction = translation.reshape(3).astype(np.float64)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-12:
            failed.append(pair)
            continue
        direction /= norm
        residuals = _epipolar_residuals_px(
            essential, camera_matrix, points1[inliers], points2[inliers]
        )
        poses.append(
            RelativeCameraPose(
                pose_id=f"camera-pose:{package.video_id}:{pair[0]}:{pair[1]}",
                source_frame_index=pair[0],
                target_frame_index=pair[1],
                source_timestamp_s=flow_observation.source_timestamp_s,
                target_timestamp_s=flow_observation.target_timestamp_s,
                rotation_source_to_target=tuple(float(value) for value in rotation.reshape(-1)),
                translation_direction_source_to_target=_vector(direction),
                correspondence_count=int(points1.shape[0]),
                inlier_count=inlier_count,
                inlier_fraction=inlier_count / int(points1.shape[0]),
                median_epipolar_residual_px=float(np.median(residuals)),
            )
        )
    if poses:
        return CameraMotionEstimate(
            observability=Observability.RELATIVE,
            poses=tuple(poses),
            failed_frame_pairs=tuple(failed),
            translation_scale="up_to_scale",
            reason=(
                "essential matrices fitted to RAFT background correspondences; "
                "translation magnitude remains unobservable"
            ),
        )
    return CameraMotionEstimate(
        observability=Observability.UNOBSERVABLE,
        poses=(),
        failed_frame_pairs=tuple(failed),
        translation_scale="unobservable",
        reason="insufficient valid background-flow geometry for an essential matrix",
    )


def _video_package(
    *,
    loaded: _LoadedStep3,
    package: VideoTrackingManifest,
    package_reference: ArtifactRef,
    tracking_store_reference: ArtifactRef,
    config: Step4ConfigSnapshot,
    config_sha256: str,
) -> VideoGeometryManifest:
    step2_root = loaded.run_root / package.input_snapshot.source_step2_relative_root
    step2_manifest_link = package.input_snapshot.video_evidence_manifest
    step2_manifest_path = step2_root / step2_manifest_link.artifact.relative_path
    if not step2_manifest_path.is_file() or sha256_file(step2_manifest_path) != step2_manifest_link.artifact.sha256:
        raise RuntimeError(f"Step 4 could not verify Step 2 manifest: {step2_manifest_path}")
    evidence = read_contract(step2_manifest_path, VideoEvidenceManifest)
    if evidence.video_id != package.video_id:
        raise RuntimeError("Step 2/Step 3 video identity mismatch at Step 4")
    resolver = _ArtifactResolver(
        run_root=loaded.run_root,
        step2_root=step2_root,
        step3_root=loaded.stage_root,
    )
    intrinsics = _intrinsics(image_size=package.image_size, config=config)
    scales = _scale_hypotheses(package)
    candidates = _selected_candidates(package)
    reserved = _reserved_artifact_ids(package, consumer="step4_geometry")
    geometry_tracks = []
    emitted, unavailable, passed, failed = 0, 0, 0, 0
    for track in package.tracks:
        available_rows, unavailable_rows = [], []
        for observation in track.observations:
            if observation.depth is not None and observation.depth.artifact.artifact_id in reserved:
                row = UnavailableGeometryObservation(
                    unavailable_id=f"geometry-unavailable:{track.track_id}:{observation.frame_index}",
                    track_id=track.track_id,
                    frame_index=observation.frame_index,
                    timestamp_s=observation.timestamp_s,
                    detection_id=observation.detection_id,
                    reason="depth_reserved_as_check_only",
                    source_artifacts=(observation.depth,),
                )
            else:
                row = _geometry_observation(
                    track_id=track.track_id,
                    observation=observation,
                    candidate=candidates.get(observation.selected_mask_candidate_id),
                    resolver=resolver,
                    intrinsics=intrinsics,
                    scale_hypotheses=scales,
                    config=config,
                )
            if isinstance(row, GeometryObservation):
                available_rows.append(row)
                emitted += 1
                if row.validation_passed:
                    passed += 1
                else:
                    failed += 1
            else:
                unavailable_rows.append(row)
                unavailable += 1
        geometry_tracks.append(
            ObjectGeometryTrack(
                track_id=track.track_id,
                primary_class=track.primary_class,
                observations=tuple(available_rows),
                unavailable_observations=tuple(unavailable_rows),
            )
        )
    motion = _camera_motion(
        evidence=evidence,
        package=package,
        resolver=resolver,
        intrinsics=intrinsics,
        reserved_artifact_ids=reserved,
        config=config,
    )
    validation = GeometryValidationSummary(
        requested_observations=emitted + unavailable,
        emitted_observations=emitted,
        unavailable_observations=unavailable,
        verified_source_artifacts=resolver.verified_count,
        failed_source_artifacts=(),
        passed_observations=passed,
        failed_observations=failed,
        overall_pass=failed == 0,
    )
    return VideoGeometryManifest(
        run_id=package.run_id,
        video_id=package.video_id,
        source_tracking_sha256=package_reference.sha256,
        config_sha256=config_sha256,
        canonical_fps=package.canonical_fps,
        image_size=package.image_size,
        frame_count=package.frame_count,
        input_snapshot=Step4InputSnapshot(
            source_step3_relative_root=loaded.stage_root.relative_to(loaded.run_root).as_posix(),
            tracking_store=_step3_link(tracking_store_reference),
            video_tracking_manifest=_step3_link(package_reference),
            source_step2_relative_root=package.input_snapshot.source_step2_relative_root,
        ),
        intrinsics=intrinsics,
        camera_motion=motion,
        ground_plane=GroundPlaneEstimate(
            observability=Observability.UNOBSERVABLE,
            method="not_estimated",
            reason=(
                "road/ground semantic support is not yet available; no plane or camera-height "
                "scale is asserted"
            ),
        ),
        scale_hypotheses=scales,
        tracks=tuple(geometry_tracks),
        validation=validation,
        tool_versions=(
            ToolVersion(name="opencv", version=cv2.__version__),
            ToolVersion(name="numpy", version=np.__version__),
        ),
    )


def run_step4(
    *,
    tracking_store_path: Path | str,
    camera_fx_px: float | None = None,
    camera_fy_px: float | None = None,
    camera_cx_px: float | None = None,
    camera_cy_px: float | None = None,
    horizontal_fov_degrees: float = 90.0,
    horizontal_fov_min_degrees: float = 60.0,
    horizontal_fov_max_degrees: float = 120.0,
    support_erosion_pixels: int = 2,
    bbox_inset_fraction: float = 0.25,
    minimum_support_pixels: int = 16,
    minimum_valid_depth_fraction: float = 0.25,
    maximum_median_reprojection_error_px: float = 1e-3,
    background_flow_sample_stride: int = 16,
    minimum_pose_correspondences: int = 32,
    pose_ransac_threshold_px: float = 1.5,
) -> Step4Result:
    """Create camera-centric geometry without inventing monocular metric scale."""

    if (camera_fx_px is None) != (camera_fy_px is None):
        raise ValueError("camera_fx_px and camera_fy_px must be provided together")
    loaded = _load_step3(tracking_store_path)
    config = Step4ConfigSnapshot(
        intrinsics_mode=(
            "provided_cli" if camera_fx_px is not None else "horizontal_fov_prior"
        ),
        camera_fx_px=camera_fx_px,
        camera_fy_px=camera_fy_px,
        camera_cx_px=camera_cx_px,
        camera_cy_px=camera_cy_px,
        horizontal_fov_degrees=horizontal_fov_degrees,
        horizontal_fov_min_degrees=horizontal_fov_min_degrees,
        horizontal_fov_max_degrees=horizontal_fov_max_degrees,
        support_erosion_pixels=support_erosion_pixels,
        bbox_inset_fraction=bbox_inset_fraction,
        minimum_support_pixels=minimum_support_pixels,
        minimum_valid_depth_fraction=minimum_valid_depth_fraction,
        maximum_median_reprojection_error_px=maximum_median_reprojection_error_px,
        background_flow_sample_stride=background_flow_sample_stride,
        minimum_pose_correspondences=minimum_pose_correspondences,
        pose_ransac_threshold_px=pose_ransac_threshold_px,
    )
    config_sha256 = hash_payload(config)
    stage_root = loaded.run_root / "04_geometry_scale" / f"config_{config_sha256[:16]}"
    stage_root.mkdir(parents=True, exist_ok=True)
    tracking_store_reference = _file_reference(
        path=loaded.store_path,
        stage_root=loaded.stage_root,
        artifact_id=f"tracking-store:{loaded.store.run_id}",
    )
    manifests, references = [], []
    for package, package_reference in zip(loaded.manifests, loaded.manifest_refs):
        manifest = _video_package(
            loaded=loaded,
            package=package,
            package_reference=package_reference,
            tracking_store_reference=tracking_store_reference,
            config=config,
            config_sha256=config_sha256,
        )
        relative_path = Path("videos") / f"{package.video_id}.geometry.json"
        path = stage_root / relative_path
        sha256, byte_size = write_contract(path, manifest)
        references.append(
            ArtifactRef(
                artifact_id=f"video-geometry:{package.video_id}",
                relative_path=relative_path.as_posix(),
                sha256=sha256,
                byte_size=byte_size,
                media_type="application/json",
                coordinate_space=CoordinateSpace.CAMERA_3D,
            )
        )
        manifests.append(manifest)
    store = GeometryStore(
        run_id=loaded.store.run_id,
        source_tracking_store_sha256=sha256_file(loaded.store_path),
        config=config,
        config_sha256=config_sha256,
        video_ids=loaded.store.video_ids,
        video_geometry=tuple(references),
    )
    store_path = stage_root / "geometry_store.json"
    write_contract(store_path, store)
    return Step4Result(
        store=store,
        video_manifests=tuple(manifests),
        stage_root=stage_root,
        store_path=store_path,
    )


__all__ = ["Step4Result", "run_step4"]
