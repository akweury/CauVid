"""Auditable visualizations for Step 4 camera-centric geometry.

The plots in this module deliberately do not call camera-frame point sequences
"world trajectories".  Step 4 has not yet accumulated a metric world pose;
the visualization therefore exposes the coordinate frame, unit, uncertainty,
and observability assumptions on every relevant output.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.exp_august.contracts import (
    ArtifactOwner,
    GeometryStore,
    VideoGeometryManifest,
    VideoManifest,
    VideoTrackingManifest,
)
from src.exp_august.contracts.codec import read_contract, sha256_file
from src.exp_august.inference.frames import CanonicalFrameProvider


def _rgb_color(track_id: str) -> tuple[float, float, float]:
    digest = hashlib.sha256(track_id.encode("utf-8")).digest()
    return tuple((70 + value % 176) / 255.0 for value in digest[:3])


def _bgr_color(track_id: str) -> tuple[int, int, int]:
    red, green, blue = _rgb_color(track_id)
    return int(255 * blue), int(255 * green), int(255 * red)


def _short_id(track_id: str) -> str:
    value = track_id.rsplit(":", 1)[-1].lstrip("0")
    return value or "0"


def _unit_label(manifest: VideoGeometryManifest) -> str:
    units = {
        observation.coordinate_unit.value
        for track in manifest.tracks
        for observation in track.observations
    }
    if units == {"meter"}:
        return "m"
    if units == {"relative_unit"}:
        return "relative units"
    if not units:
        return "unobservable"
    return "mixed units"


class _ArtifactReader:
    def __init__(self, *, step2_root: Path, step3_root: Path) -> None:
        self.step2_root = step2_root
        self.step3_root = step3_root
        self._verified: set[tuple[str, str]] = set()

    def path(self, link) -> Path:
        if link.owner == ArtifactOwner.STEP2_NEURAL_EVIDENCE:
            root = self.step2_root
        elif link.owner == ArtifactOwner.STEP3_OBJECT_TRACKING:
            root = self.step3_root
        else:
            raise RuntimeError(f"unsupported Step 4 source owner: {link.owner}")
        path = root / link.artifact.relative_path
        key = (link.owner.value, link.artifact.artifact_id)
        if key not in self._verified:
            if not path.is_file() or sha256_file(path) != link.artifact.sha256:
                raise RuntimeError(f"Step 4 visualization source failed integrity check: {path}")
            self._verified.add(key)
        return path

    def mask(self, link, shape: tuple[int, int]) -> np.ndarray:
        image = cv2.imread(str(self.path(link)), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"could not decode mask artifact: {self.path(link)}")
        if image.shape != shape:
            image = cv2.resize(
                image,
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return image > 0

    def depth(self, link) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self.path(link), allow_pickle=False) as payload:
            if "depth" not in payload or "valid" not in payload:
                raise RuntimeError(f"depth artifact lacks depth/valid arrays: {self.path(link)}")
            depth = np.asarray(payload["depth"], dtype=np.float32)
            valid = np.asarray(payload["valid"], dtype=bool)
        valid &= np.isfinite(depth) & (depth > 0.0)
        return depth, valid


def _source_links(observation):
    mask_link = next(
        (
            link
            for link in observation.source_artifacts
            if link.artifact.media_type == "image/png"
            or link.artifact.relative_path.lower().endswith(".png")
        ),
        None,
    )
    depth_link = next(
        (
            link
            for link in observation.source_artifacts
            if "depth" in link.artifact.media_type.lower()
            or link.artifact.relative_path.lower().endswith(".npz")
        ),
        None,
    )
    return mask_link, depth_link


def _overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.34,
) -> None:
    image[mask] = np.clip(
        image[mask].astype(np.float32) * (1.0 - alpha)
        + np.asarray(color, dtype=np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, color, 2)


def _draw_label(
    image: np.ndarray,
    lines: tuple[str, ...],
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    scale = 0.52
    thickness = 1
    metrics = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        for line in lines
    ]
    width = max(item[0][0] for item in metrics) + 10
    advance = max(item[0][1] + item[1] + 5 for item in metrics)
    height = advance * len(lines) + 5
    x = max(0, min(origin[0], image.shape[1] - width - 1))
    y = max(height + 1, min(origin[1], image.shape[0] - 1))
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y - height), (x + width, y), (16, 21, 30), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0.0, image)
    cv2.rectangle(image, (x, y - height), (x + 4, y), color, -1)
    baseline = y - height + metrics[0][0][1] + 4
    for line in lines:
        cv2.putText(
            image,
            line,
            (x + 8, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )
        baseline += advance


def _observations_by_frame(manifest: VideoGeometryManifest) -> dict[int, list]:
    rows: dict[int, list] = defaultdict(list)
    for track in manifest.tracks:
        for observation in track.observations:
            rows[observation.frame_index].append(observation)
    return rows


def _render_geometry_frame(
    *,
    image: np.ndarray,
    frame_index: int,
    timestamp_s: float,
    manifest: VideoGeometryManifest,
    observations: list,
    reader: _ArtifactReader,
) -> np.ndarray:
    canvas = image.copy()
    detailed_ids = {
        row.observation_id
        for row in sorted(
            observations,
            key=lambda item: -(
                (item.bbox.x2 - item.bbox.x1) * (item.bbox.y2 - item.bbox.y1)
            ),
        )[:8]
    }
    for observation in observations:
        color = _bgr_color(observation.track_id)
        mask_link, _ = _source_links(observation)
        if mask_link is not None:
            _overlay_mask(canvas, reader.mask(mask_link, canvas.shape[:2]), color)
        box = observation.bbox
        x1, y1, x2, y2 = map(
            int, (round(box.x1), round(box.y1), round(box.x2), round(box.y2))
        )
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        center = (int(round(observation.pixel_centroid.u)), int(round(observation.pixel_centroid.v)))
        cv2.drawMarker(canvas, center, color, cv2.MARKER_CROSS, 13, 2)
        if observation.observation_id in detailed_ids:
            point = observation.points.median
            q25, q75 = observation.points.q25, observation.points.q75
            unit = "m" if observation.coordinate_unit.value == "meter" else "rel"
            _draw_label(
                canvas,
                (
                    f"ID {_short_id(observation.track_id)} | {observation.class_name}",
                    f"camera XYZ=({point.x:.2f}, {point.y:.2f}, {point.z:.2f}) {unit}",
                    f"Z IQR={q25.z:.2f}-{q75.z:.2f} | valid={observation.valid_depth_fraction:.0%}",
                ),
                (x1, max(78, y1)),
                color,
            )
        else:
            text = f"ID {_short_id(observation.track_id)}"
            cv2.putText(
                canvas,
                text,
                (x1, max(72, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (12, 12, 12),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                text,
                (x1, max(72, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

    header_height = max(58, image.shape[0] // 12)
    cv2.rectangle(canvas, (0, 0), (image.shape[1], header_height), (16, 21, 30), -1)
    cv2.putText(
        canvas,
        f"Step 4 | {manifest.video_id} | frame {frame_index:04d}/{manifest.frame_count - 1:04d} "
        f"| t={timestamp_s:.2f}s | geometry={len(observations)}",
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (248, 248, 248),
        2,
        cv2.LINE_AA,
    )
    scale_text = (
        f"camera-centric {_unit_label(manifest)} | x right, y down, z forward | "
        "not a world trajectory"
    )
    cv2.putText(
        canvas,
        scale_text,
        (14, min(header_height - 9, 49)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (178, 218, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _select_example_frames(manifest: VideoGeometryManifest, count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("example_frame_count must be positive")
    observed = sorted(_observations_by_frame(manifest))
    candidates = observed or list(range(manifest.frame_count))
    positions = np.linspace(0, len(candidates) - 1, min(count, len(candidates)), dtype=int)
    return tuple(candidates[int(position)] for position in positions)


def _contact_sheet(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    if not images:
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    columns = min(2, len(images))
    rows = math.ceil(len(images) / columns)
    tiles = []
    for image, label in zip(images, labels):
        tile = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 326), (640, 360), (16, 21, 30), -1)
        cv2.putText(
            tile,
            label,
            (12, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    while len(tiles) < rows * columns:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack(
        [np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)]
    )


def _top_tracks(manifest: VideoGeometryManifest, maximum: int) -> list:
    if maximum <= 0:
        raise ValueError("maximum_tracks must be positive")
    return sorted(
        (track for track in manifest.tracks if track.observations),
        key=lambda track: (-len(track.observations), track.track_id),
    )[:maximum]


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


def _static_semantic_prior(class_name: str) -> bool:
    normalized = " ".join(class_name.lower().replace("_", " ").split())
    return any(term in normalized for term in _STATIC_CLASS_TERMS)


def _median_point(observation) -> np.ndarray:
    point = observation.points.median
    return np.asarray((point.x, point.y, point.z), dtype=np.float64)


def _relative_static_scene(manifest: VideoGeometryManifest) -> dict:
    """Build a conservative normalized ego/static-landmark reconstruction.

    The relative camera pose contract has translation direction but not
    magnitude.  Static multi-frame object observations provide a diagnostic
    magnitude estimate.  When they do not, a unit step is retained and marked
    as a fallback.  Independent pose-graph components are never silently joined.
    """

    tracks = {track.track_id: track for track in manifest.tracks}
    frame_points = {
        track.track_id: {row.frame_index: _median_point(row) for row in track.observations}
        for track in manifest.tracks
    }
    semantic_static_ids = {
        track.track_id
        for track in manifest.tracks
        if _static_semantic_prior(track.primary_class) and len(track.observations) >= 2
    }
    pose_rows = []
    fallback_static_ids: set[str] = set()
    raw_supported_scales = []
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
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-9:
            direction = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        else:
            direction /= direction_norm

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
                    "perpendicular_residual": float(np.linalg.norm(perpendicular)),
                    "normalized_perpendicular_residual": float(
                        np.linalg.norm(perpendicular) / reference_depth
                    ),
                }
            )
        semantic_candidates = [
            row
            for row in candidates
            if row["semantic_static"]
            and row["raw_scale"] > 1e-6
            and row["normalized_perpendicular_residual"] <= 0.50
        ]
        if semantic_candidates:
            selected = semantic_candidates
            evidence_mode = "semantic_static_tracks"
        else:
            selected = [
                row
                for row in candidates
                if row["raw_scale"] > 1e-6
                and row["normalized_perpendicular_residual"] <= 0.15
            ]
            evidence_mode = (
                "low_motion_residual_fallback" if selected else "unit_translation_fallback"
            )
            fallback_static_ids.update(row["track_id"] for row in selected)
        if selected:
            values = np.asarray([row["raw_scale"] for row in selected], dtype=np.float64)
            raw_scale = float(np.median(values))
            scale_mad = float(np.median(np.abs(values - raw_scale)))
            raw_supported_scales.append(raw_scale)
        else:
            raw_scale = None
            scale_mad = None
        pose_rows.append(
            {
                "pose_id": pose.pose_id,
                "source_frame_index": pose.source_frame_index,
                "target_frame_index": pose.target_frame_index,
                "rotation_source_to_target": rotation.tolist(),
                "translation_direction_source_to_target": direction.tolist(),
                "raw_translation_scale": raw_scale,
                "raw_translation_scale_mad": scale_mad,
                "scale_evidence_mode": evidence_mode,
                "scale_evidence_track_ids": [row["track_id"] for row in selected],
                "candidate_count": len(candidates),
                "inlier_fraction": pose.inlier_fraction,
                "median_epipolar_residual_px": pose.median_epipolar_residual_px,
            }
        )

    scale_normalizer = (
        float(np.median(np.asarray(raw_supported_scales, dtype=np.float64)))
        if raw_supported_scales
        else 1.0
    )
    if not np.isfinite(scale_normalizer) or scale_normalizer <= 1e-9:
        scale_normalizer = 1.0
    for row in pose_rows:
        raw_scale = row["raw_translation_scale"]
        row["normalized_translation_scale"] = (
            float(raw_scale / scale_normalizer) if raw_scale is not None else 1.0
        )

    transforms: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    component_count = 0
    for row in pose_rows:
        source_frame = row["source_frame_index"]
        target_frame = row["target_frame_index"]
        if source_frame not in transforms:
            transforms[source_frame] = (
                np.eye(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                component_count,
            )
            component_count += 1
        source_rotation, source_translation, component_id = transforms[source_frame]
        relative_rotation = np.asarray(row["rotation_source_to_target"], dtype=np.float64)
        relative_direction = np.asarray(
            row["translation_direction_source_to_target"], dtype=np.float64
        )
        relative_scale = float(row["normalized_translation_scale"])
        target_rotation = relative_rotation @ source_rotation
        target_translation = (
            relative_rotation @ source_translation + relative_scale * relative_direction
        )
        if target_frame not in transforms:
            transforms[target_frame] = (
                target_rotation,
                target_translation,
                component_id,
            )
        row["component_id"] = component_id

    ego_components: dict[int, list[dict]] = defaultdict(list)
    for frame_index, (rotation, translation, component_id) in sorted(transforms.items()):
        center = -rotation.T @ translation
        ego_components[component_id].append(
            {
                "frame_index": frame_index,
                "timestamp_s": frame_index / manifest.canonical_fps,
                "camera_center_world": center.tolist(),
                "rotation_world_to_camera": rotation.tolist(),
                "translation_world_to_camera": translation.tolist(),
            }
        )

    # Do not populate the visible sandbox with generic vehicles when explicit
    # stationary semantic anchors exist.  Low-motion residual tracks remain a
    # clearly marked fallback only for videos without such anchors.
    candidate_static_ids = semantic_static_ids or fallback_static_ids
    landmark_rows = []
    for track_id in sorted(candidate_static_ids):
        track = tracks[track_id]
        by_component: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for observation in track.observations:
            transform = transforms.get(observation.frame_index)
            if transform is None:
                continue
            rotation, translation, component_id = transform
            world_point = rotation.T @ (_median_point(observation) - translation)
            by_component[component_id].append((observation.frame_index, world_point))
        for component_id, rows in sorted(by_component.items()):
            if len(rows) < 2:
                continue
            values = np.asarray([point for _, point in rows], dtype=np.float64)
            median = np.median(values, axis=0)
            deviations = np.linalg.norm(values - median, axis=1)
            spread = float(np.median(deviations))
            relative_spread = spread / max(1.0, float(np.linalg.norm(median)))
            landmark_rows.append(
                {
                    "landmark_id": f"static:{track_id}:component:{component_id}",
                    "track_id": track_id,
                    "class_name": track.primary_class,
                    "component_id": component_id,
                    "selection_basis": (
                        "semantic_static_prior"
                        if track_id in semantic_static_ids
                        else "low_motion_residual_fallback"
                    ),
                    "observation_count": len(rows),
                    "frame_indices": [frame for frame, _ in rows],
                    "world_observations": [point.tolist() for _, point in rows],
                    "median_world_position": median.tolist(),
                    "axis_q25": np.percentile(values, 25.0, axis=0).tolist(),
                    "axis_q75": np.percentile(values, 75.0, axis=0).tolist(),
                    "median_radial_spread": spread,
                    "relative_spread": relative_spread,
                    "static_consistency": (
                        "supported" if relative_spread <= 0.25 else "inconsistent"
                    ),
                }
            )

    return {
        "schema_name": "relative_static_scene",
        "schema_version": 1,
        "video_id": manifest.video_id,
        "coordinate_frame": "component_local_world_from_first_camera",
        "coordinate_convention": "x_right_y_down_z_forward_at_component_origin",
        "coordinate_unit": "normalized_relative_translation_step",
        "metric_scale_claimed": False,
        "world_trajectory_claimed": False,
        "method": "pairwise_pose_accumulation_with_static_track_scale_diagnostics",
        "limitations": [
            "monocular_metric_scale_unobservable",
            "depth_scale_may_drift_between_frames",
            "disconnected_pose_components_are_not_aligned",
            "static_semantics_are_priors_and_require_residual_validation",
            "this_is_a_step4_diagnostic_not_step5_physical_motion",
        ],
        "translation_scale_normalizer": scale_normalizer,
        "pose_scale_estimates": pose_rows,
        "ego_components": [
            {"component_id": component_id, "poses": poses}
            for component_id, poses in sorted(ego_components.items())
        ],
        "static_landmarks": landmark_rows,
        "summary": {
            "pose_count": len(pose_rows),
            "component_count": len(ego_components),
            "semantic_static_track_count": len(semantic_static_ids),
            "fallback_static_track_count": len(fallback_static_ids - semantic_static_ids),
            "static_landmark_count": len(landmark_rows),
            "consistent_static_landmark_count": sum(
                row["static_consistency"] == "supported" for row in landmark_rows
            ),
        },
    }


def _sandbox_coordinates(point: list[float] | np.ndarray) -> np.ndarray:
    """Map x-right/y-down/z-forward contract axes to a sandbox x/z/y-up view."""

    x, y, z = np.asarray(point, dtype=np.float64)
    return np.asarray((x, z, -y), dtype=np.float64)


def _plot_relative_static_sandbox_legacy(scene: dict, output_path: Path) -> None:
    figure = plt.figure(figsize=(12.8, 7.2), dpi=150)
    axis = figure.add_subplot(111, projection="3d")
    for component in scene["ego_components"]:
        poses = component["poses"]
        points = np.asarray(
            [_sandbox_coordinates(row["camera_center_world"]) for row in poses],
            dtype=np.float64,
        )
        if not points.size:
            continue
        component_id = component["component_id"]
        axis.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            "o-",
            color="#d62728",
            linewidth=2.4,
            markersize=6,
            label=f"ego component {component_id}",
        )
        for point, pose in zip(points, poses):
            axis.text(
                point[0],
                point[1],
                point[2],
                f" f{pose['frame_index']}",
                color="#8b1a1a",
                fontsize=7,
            )

    for landmark in scene["static_landmarks"]:
        observations = np.asarray(
            [_sandbox_coordinates(row) for row in landmark["world_observations"]],
            dtype=np.float64,
        )
        median = _sandbox_coordinates(landmark["median_world_position"])
        color = _rgb_color(landmark["track_id"])
        axis.scatter(
            observations[:, 0],
            observations[:, 1],
            observations[:, 2],
            color=[color],
            s=18,
            alpha=0.25,
        )
        marker = "s" if landmark["static_consistency"] == "supported" else "X"
        axis.scatter(
            [median[0]],
            [median[1]],
            [median[2]],
            color=[color],
            edgecolors="black",
            linewidths=0.6,
            marker=marker,
            s=78,
            label=(
                f"ID {_short_id(landmark['track_id'])} {landmark['class_name']} "
                f"[{landmark['static_consistency']}]"
            ),
        )
        axis.text(
            median[0],
            median[1],
            median[2],
            f" ID {_short_id(landmark['track_id'])}",
            fontsize=7,
        )
        q25 = _sandbox_coordinates(landmark["axis_q25"])
        q75 = _sandbox_coordinates(landmark["axis_q75"])
        for dimension in range(3):
            lower, upper = sorted((q25[dimension], q75[dimension]))
            segment = np.vstack((median, median))
            segment[0, dimension] = lower
            segment[1, dimension] = upper
            axis.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                color=color,
                alpha=0.55,
                linewidth=1.3,
            )

    summary = scene["summary"]
    if not scene["ego_components"]:
        axis.text2D(
            0.5,
            0.52,
            "No connected relative camera poses are observable",
            transform=axis.transAxes,
            ha="center",
        )
    axis.set_xlabel("World X: right (normalized relative)", labelpad=10)
    axis.set_ylabel("World Z: forward (normalized relative)", labelpad=10)
    axis.set_zlabel("World Y: up (normalized relative)", labelpad=10)
    axis.set_title(
        f"{scene['video_id']}: relative ego + static-object sandbox\n"
        "Red: ego camera centers; squares: consistent static candidates; X: inconsistent. Not metric ground truth.",
        pad=18,
    )
    axis.text2D(
        0.01,
        0.98,
        (
            f"poses={summary['pose_count']} | components={summary['component_count']} | "
            f"static landmarks={summary['static_landmark_count']} | "
            f"consistent={summary['consistent_static_landmark_count']}\n"
            "Each disconnected component has its own origin; depth and translation are normalized."
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#bbbbbb"},
    )
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            handles[:14],
            labels[:14],
            loc="upper left",
            bbox_to_anchor=(1.02, 0.93),
            fontsize=7.5,
        )
    axis.view_init(elev=24, azim=-58)
    figure.subplots_adjust(left=0.03, right=0.77, bottom=0.06, top=0.88)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _draw_static_landmark(axis, landmark: dict, *, compact: bool) -> list[np.ndarray]:
    observations = np.asarray(
        [_sandbox_coordinates(row) for row in landmark["world_observations"]],
        dtype=np.float64,
    )
    median = _sandbox_coordinates(landmark["median_world_position"])
    color = _rgb_color(landmark["track_id"])
    axis.scatter(
        observations[:, 0], observations[:, 1], observations[:, 2],
        color=[color], s=12 if compact else 18, alpha=0.25,
    )
    marker = "s" if landmark["static_consistency"] == "supported" else "X"
    axis.scatter(
        [median[0]], [median[1]], [median[2]], color=[color],
        edgecolors="black", linewidths=0.6, marker=marker,
        s=48 if compact else 78,
    )
    axis.text(
        median[0], median[1], median[2],
        f" ID {_short_id(landmark['track_id'])}",
        fontsize=6 if compact else 8,
    )
    q25 = _sandbox_coordinates(landmark["axis_q25"])
    q75 = _sandbox_coordinates(landmark["axis_q75"])
    for dimension in range(3):
        lower, upper = sorted((q25[dimension], q75[dimension]))
        segment = np.vstack((median, median))
        segment[0, dimension] = lower
        segment[1, dimension] = upper
        axis.plot(
            segment[:, 0], segment[:, 1], segment[:, 2],
            color=color, alpha=0.55, linewidth=1.0 if compact else 1.3,
        )
    return [*observations, q25, q75]


def _set_sandbox_axis_geometry(axis, points: list[np.ndarray]) -> None:
    """Use data-aware, forward-emphasized geometry instead of a cubic box."""

    if points:
        values = np.asarray(points, dtype=np.float64)
        lower = np.min(values, axis=0)
        upper = np.max(values, axis=0)
        spans = np.maximum(upper - lower, 1e-3)
        padding = np.maximum(spans * 0.08, 0.04)
        axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
        axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
        axis.set_zlim(lower[2] - padding[2], upper[2] + padding[2])
    else:
        spans = np.ones(3, dtype=np.float64)
    # Axis order is lateral X, forward Z, vertical Y. Preserve measured
    # proportions while keeping the semantic forward dimension visibly long.
    lateral = max(float(spans[0]), 0.08 * float(spans[1]))
    vertical = max(float(spans[2]), 0.06 * float(spans[1]))
    forward = max(float(spans[1]), 2.5 * lateral, 4.0 * vertical)
    axis.set_box_aspect((lateral, forward, vertical))


def _draw_sandbox_component(
    axis,
    component: dict,
    landmarks: list[dict],
    *,
    compact: bool,
) -> None:
    poses = component["poses"]
    points = np.asarray(
        [_sandbox_coordinates(row["camera_center_world"]) for row in poses],
        dtype=np.float64,
    )
    plot_points: list[np.ndarray] = [*points]
    if points.size:
        axis.plot(
            points[:, 0], points[:, 1], points[:, 2], "o-",
            color="#d62728", linewidth=1.7 if compact else 2.4,
            markersize=3.5 if compact else 5.5,
        )
        endpoints = [(0, "start")]
        if len(poses) > 1:
            endpoints.append((-1, "end"))
        for index, role in endpoints:
            point = points[index]
            pose = poses[index]
            axis.text(
                point[0], point[1], point[2],
                f" {role} f{pose['frame_index']}",
                color="#8b1a1a", fontsize=6 if compact else 8,
            )
    for landmark in landmarks:
        plot_points.extend(_draw_static_landmark(axis, landmark, compact=compact))
    _set_sandbox_axis_geometry(axis, plot_points)
    first_frame = poses[0]["frame_index"] if poses else "?"
    last_frame = poses[-1]["frame_index"] if poses else "?"
    axis.set_title(
        f"Component {component['component_id']} | frames {first_frame}-{last_frame} | "
        f"{len(poses)} ego points",
        fontsize=9 if compact else 13,
        pad=8 if compact else 16,
    )
    axis.set_xlabel("X right", fontsize=7 if compact else 10, labelpad=2 if compact else 8)
    axis.set_ylabel(
        "Z forward (normalized)", fontsize=7 if compact else 10,
        labelpad=3 if compact else 9,
    )
    axis.set_zlabel("Y up", fontsize=7 if compact else 10, labelpad=2 if compact else 8)
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(6))
    axis.zaxis.set_major_locator(MaxNLocator(4))
    axis.tick_params(labelsize=6 if compact else 8, pad=0 if compact else 2)
    # View almost across the lateral axis so the elongated forward dimension
    # occupies the horizontal canvas instead of collapsing into a thin diagonal.
    axis.view_init(elev=18, azim=-8)


def _plot_relative_static_sandbox(
    scene: dict,
    output_path: Path,
    component_root: Path,
) -> list[Path]:
    """Render a component overview and one independent figure per component."""

    components = scene["ego_components"]
    landmarks_by_component: dict[int, list[dict]] = defaultdict(list)
    for landmark in scene["static_landmarks"]:
        landmarks_by_component[int(landmark["component_id"])].append(landmark)

    columns = min(3, max(1, len(components)))
    rows = max(1, math.ceil(len(components) / columns))
    figure = plt.figure(figsize=(6.4 * columns, 5.0 * rows), dpi=150)
    for plot_index, component in enumerate(components, start=1):
        axis = figure.add_subplot(rows, columns, plot_index, projection="3d")
        _draw_sandbox_component(
            axis, component,
            landmarks_by_component[int(component["component_id"])],
            compact=True,
        )
    figure.suptitle(
        f"{scene['video_id']}: disconnected relative ego components\n"
        "Each panel has its own origin and display scale; forward is elongated. "
        "Only segment endpoints are labeled.",
        fontsize=15, y=0.995,
    )
    figure.text(
        0.5, 0.008,
        "Red: ego camera centers | square: consistent static candidate | "
        "X: inconsistent | normalized relative, not metric ground truth",
        ha="center", fontsize=9,
    )
    figure.subplots_adjust(
        left=0.02, right=0.98, bottom=0.04, top=0.94,
        wspace=0.05, hspace=0.16,
    )
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)

    component_root.mkdir(parents=True, exist_ok=True)
    component_paths = []
    for component in components:
        component_id = int(component["component_id"])
        poses = component["poses"]
        first_frame = int(poses[0]["frame_index"])
        last_frame = int(poses[-1]["frame_index"])
        component_path = component_root / (
            f"component_{component_id:02d}_frames_{first_frame:04d}_{last_frame:04d}.png"
        )
        detail = plt.figure(figsize=(12.8, 7.2), dpi=150)
        axis = detail.add_subplot(111, projection="3d")
        component_landmarks = landmarks_by_component[component_id]
        _draw_sandbox_component(
            axis, component, component_landmarks, compact=False,
        )
        detail.suptitle(
            f"{scene['video_id']}: relative ego component {component_id}\n"
            "Independent local origin; forward-axis geometry is not constrained to a cube.",
            fontsize=15, y=0.98,
        )
        detail.text(
            0.5, 0.025,
            f"static candidates={len(component_landmarks)} | "
            "normalized relative coordinates | not metric ground truth",
            ha="center", fontsize=9,
        )
        detail.subplots_adjust(left=0.04, right=0.96, bottom=0.08, top=0.86)
        # Preserve the requested 16:9 canvas. Tight bounding-box export can
        # crop titles and turn a forward-elongated 3D box into a nearly square
        # raster even when the underlying axes are correctly proportioned.
        detail.savefig(component_path)
        plt.close(detail)
        component_paths.append(component_path)
    return component_paths


def _plot_camera_points_3d(
    manifest: VideoGeometryManifest,
    output_path: Path,
    maximum_tracks: int,
) -> None:
    tracks = _top_tracks(manifest, maximum_tracks)
    figure = plt.figure(figsize=(12.8, 7.2), dpi=150)
    axis = figure.add_subplot(111, projection="3d")
    for track in tracks:
        observations = list(track.observations)
        color = _rgb_color(track.track_id)
        points = np.asarray(
            [[row.points.median.x, row.points.median.y, row.points.median.z] for row in observations]
        )
        label = f"ID {_short_id(track.track_id)} {track.primary_class} (n={len(observations)})"
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], color=[color], s=28, label=label)
        for left, right in zip(observations, observations[1:]):
            values = np.asarray(
                [
                    [left.points.median.x, left.points.median.y, left.points.median.z],
                    [right.points.median.x, right.points.median.y, right.points.median.z],
                ]
            )
            style = "-" if right.frame_index == left.frame_index + 1 else "--"
            axis.plot(values[:, 0], values[:, 1], values[:, 2], style, color=color, alpha=0.75)
        for row in observations:
            point = row.points.median
            axis.plot(
                [point.x, point.x],
                [point.y, point.y],
                [row.points.q25.z, row.points.q75.z],
                color=color,
                alpha=0.28,
                linewidth=1.0,
            )
    if not tracks:
        axis.text2D(0.5, 0.5, "No usable 3D object observations", transform=axis.transAxes, ha="center")
    unit = _unit_label(manifest)
    axis.set_xlabel(f"X: right ({unit})", labelpad=10)
    axis.set_ylabel(f"Y: down ({unit})", labelpad=10)
    axis.set_zlabel(f"Z: forward ({unit})", labelpad=10)
    axis.invert_yaxis()
    axis.set_title(
        f"{manifest.video_id}: camera-frame 3D point sequences\n"
        "Dashed segments cross missing observations; whiskers show Z IQR. Not world trajectories.",
        pad=18,
    )
    if tracks:
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    figure.subplots_adjust(left=0.04, right=0.78, bottom=0.08, top=0.88)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_geometry_timeline(
    manifest: VideoGeometryManifest,
    output_path: Path,
    maximum_tracks: int,
) -> None:
    tracks = _top_tracks(manifest, maximum_tracks)
    figure, axes = plt.subplots(3, 1, figsize=(12.8, 7.2), dpi=150, sharex=True)
    for track in tracks:
        color = _rgb_color(track.track_id)
        times = np.asarray([row.timestamp_s for row in track.observations])
        label = f"ID {_short_id(track.track_id)} {track.primary_class}"
        for axis_name, axis in zip(("x", "y", "z"), axes):
            medians = np.asarray([getattr(row.points.median, axis_name) for row in track.observations])
            lower = np.asarray([getattr(row.points.q25, axis_name) for row in track.observations])
            upper = np.asarray([getattr(row.points.q75, axis_name) for row in track.observations])
            axis.plot(times, medians, color=color, marker="o", markersize=3, label=label)
            axis.fill_between(times, lower, upper, color=color, alpha=0.13)
    unit = _unit_label(manifest)
    for name, axis in zip(("X right", "Y down", "Z forward"), axes):
        axis.set_ylabel(f"{name}\n({unit})")
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Canonical time (s)")
    figure.suptitle(
        f"{manifest.video_id}: camera-frame XYZ observations with interquartile bands\n"
        "Discontinuities are preserved; no physical smoothing is applied in Step 4.",
        fontsize=13,
    )
    if tracks:
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No usable geometry observations", transform=axes[1].transAxes, ha="center")
    figure.subplots_adjust(left=0.09, right=0.80, bottom=0.09, top=0.84, hspace=0.12)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_camera_motion(manifest: VideoGeometryManifest, output_path: Path) -> None:
    poses = list(manifest.camera_motion.poses)
    figure, axes = plt.subplots(3, 1, figsize=(12.8, 7.2), dpi=150, sharex=True)
    if poses:
        frames = np.asarray([pose.target_frame_index for pose in poses])
        axes[0].plot(frames, [pose.inlier_fraction for pose in poses], "o-", color="#1874b4")
        axes[0].set_ylim(-0.03, 1.03)
        axes[1].plot(
            frames,
            [pose.median_epipolar_residual_px for pose in poses],
            "o-",
            color="#d95f02",
        )
        for component, color in zip(("x", "y", "z"), ("#1b9e77", "#7570b3", "#e7298a")):
            axes[2].plot(
                frames,
                [getattr(pose.translation_direction_source_to_target, component) for pose in poses],
                "o-",
                label=f"t_{component}",
                color=color,
            )
        axes[2].legend(loc="upper right", ncol=3)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Relative camera motion is unobservable for this video",
            transform=axes[1].transAxes,
            ha="center",
        )
    for source, target in manifest.camera_motion.failed_frame_pairs:
        for axis in axes:
            axis.axvspan(source, target, color="#cc3333", alpha=0.10)
    axes[0].set_ylabel("Inlier fraction")
    axes[1].set_ylabel("Median epipolar\nresidual (px)")
    axes[2].set_ylabel("Translation\ndirection")
    axes[2].set_xlabel("Target canonical frame")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        f"{manifest.video_id}: background-flow camera-motion diagnostics\n"
        f"Observability={manifest.camera_motion.observability.value}; translation scale="
        f"{manifest.camera_motion.translation_scale}. Red spans indicate failed pairs.",
        fontsize=13,
    )
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.09, top=0.84, hspace=0.14)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _depth_panel(
    *,
    image: np.ndarray,
    frame_index: int,
    observations: list,
    reader: _ArtifactReader,
    manifest: VideoGeometryManifest,
) -> np.ndarray | None:
    depth_link = next(
        (link for row in observations for link in [_source_links(row)[1]] if link is not None),
        None,
    )
    if depth_link is None:
        return None
    depth, valid = reader.depth(depth_link)
    finite = depth[valid]
    if not finite.size:
        return None
    low, high = np.percentile(finite, (2.0, 98.0))
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if high > low:
        normalized[valid] = np.clip((depth[valid] - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    heatmap[~valid] = 0
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = _render_geometry_frame(
        image=image,
        frame_index=frame_index,
        timestamp_s=frame_index / manifest.canonical_fps,
        manifest=manifest,
        observations=observations,
        reader=reader,
    )
    for row in observations:
        mask_link, _ = _source_links(row)
        if mask_link is None:
            continue
        mask = reader.mask(mask_link, heatmap.shape[:2])
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(heatmap, contours, -1, _bgr_color(row.track_id), 2)
    cv2.rectangle(heatmap, (0, 0), (heatmap.shape[1], 58), (16, 21, 30), -1)
    cv2.putText(heatmap, "Depth evidence used for back-projection", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (248, 248, 248), 2, cv2.LINE_AA)
    cv2.putText(heatmap, f"per-frame display range: {low:.2f} to {high:.2f} ({_unit_label(manifest)})", (14, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (178, 218, 255), 1, cv2.LINE_AA)
    target_height = 540
    width = int(round(image.shape[1] * target_height / image.shape[0]))
    overlay = cv2.resize(overlay, (width, target_height), interpolation=cv2.INTER_AREA)
    heatmap = cv2.resize(heatmap, (width, target_height), interpolation=cv2.INTER_AREA)
    return np.hstack((overlay, heatmap))


def render_step4_visualizations(
    *,
    geometry_store_path: Path | str,
    example_frame_count: int = 4,
    maximum_tracks: int = 12,
    render_video: bool = True,
) -> Path:
    """Render Step 4 evidence overlays and diagnostic plots for every video."""

    store_path = Path(geometry_store_path).expanduser().resolve()
    store = read_contract(store_path, GeometryStore)
    stage_root = store_path.parent
    run_root = stage_root.parent.parent
    output_root = stage_root / "visualizations"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for reference in store.video_geometry:
        geometry_path = stage_root / reference.relative_path
        if not geometry_path.is_file() or sha256_file(geometry_path) != reference.sha256:
            raise RuntimeError(f"geometry manifest hash mismatch: {geometry_path}")
        geometry = read_contract(geometry_path, VideoGeometryManifest)
        step3_root = run_root / geometry.input_snapshot.source_step3_relative_root
        tracking_ref = geometry.input_snapshot.video_tracking_manifest.artifact
        tracking_path = step3_root / tracking_ref.relative_path
        if not tracking_path.is_file() or sha256_file(tracking_path) != tracking_ref.sha256:
            raise RuntimeError(f"tracking manifest hash mismatch: {tracking_path}")
        tracking = read_contract(tracking_path, VideoTrackingManifest)
        source_ref = tracking.input_snapshot.source_video_manifest.artifact
        source_path = run_root / "01_init" / source_ref.relative_path
        if not source_path.is_file() or sha256_file(source_path) != source_ref.sha256:
            raise RuntimeError(f"source video manifest hash mismatch: {source_path}")
        source = read_contract(source_path, VideoManifest)
        step2_root = run_root / geometry.input_snapshot.source_step2_relative_root
        reader = _ArtifactReader(step2_root=step2_root, step3_root=step3_root)
        provider = CanonicalFrameProvider(source, verify_source_hash=True)
        observations_by_frame = _observations_by_frame(geometry)
        selected_frames = _select_example_frames(geometry, example_frame_count)

        video_root = output_root / geometry.video_id
        frame_root = video_root / "frames"
        depth_root = video_root / "depth_geometry_examples"
        frame_root.mkdir(parents=True, exist_ok=True)
        depth_root.mkdir(parents=True, exist_ok=True)
        video_path = video_root / f"{geometry.video_id}_step4_geometry.mp4"
        writer = None
        frame_paths: list[Path] = []
        example_images: dict[int, np.ndarray] = {}
        depth_paths: list[Path] = []
        try:
            for canonical in provider.iter_frames():
                rows = observations_by_frame.get(canonical.frame_index, [])
                rendered = _render_geometry_frame(
                    image=canonical.image_bgr,
                    frame_index=canonical.frame_index,
                    timestamp_s=canonical.timestamp_s,
                    manifest=geometry,
                    observations=rows,
                    reader=reader,
                )
                frame_path = frame_root / f"frame_{canonical.frame_index:06d}.png"
                if not cv2.imwrite(str(frame_path), rendered):
                    raise RuntimeError(f"could not write Step 4 frame: {frame_path}")
                frame_paths.append(frame_path)
                if canonical.frame_index in selected_frames:
                    example_images[canonical.frame_index] = rendered
                    panel = _depth_panel(
                        image=canonical.image_bgr,
                        frame_index=canonical.frame_index,
                        observations=rows,
                        reader=reader,
                        manifest=geometry,
                    )
                    if panel is not None:
                        panel_path = depth_root / f"frame_{canonical.frame_index:06d}_depth_geometry.png"
                        if not cv2.imwrite(str(panel_path), panel):
                            raise RuntimeError(f"could not write depth geometry panel: {panel_path}")
                        depth_paths.append(panel_path)
                if render_video:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            max(0.1, geometry.canonical_fps),
                            (rendered.shape[1], rendered.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"could not open Step 4 video: {video_path}")
                    writer.write(rendered)
        finally:
            if writer is not None:
                writer.release()

        ordered_examples = [index for index in selected_frames if index in example_images]
        sheet = _contact_sheet(
            [example_images[index] for index in ordered_examples],
            [f"frame {index} | t={source.frames[index].timestamp_s:.2f}s" for index in ordered_examples],
        )
        sheet_path = video_root / f"{geometry.video_id}_step4_examples.png"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise RuntimeError(f"could not write Step 4 contact sheet: {sheet_path}")
        points_path = video_root / f"{geometry.video_id}_camera_centric_points_3d.png"
        timeline_path = video_root / f"{geometry.video_id}_geometry_timeline.png"
        motion_path = video_root / f"{geometry.video_id}_camera_motion_diagnostics.png"
        sandbox_path = video_root / f"{geometry.video_id}_relative_static_sandbox_3d.png"
        sandbox_component_root = video_root / "relative_static_sandbox_components"
        static_scene_path = video_root / f"{geometry.video_id}_relative_static_scene.json"
        _plot_camera_points_3d(geometry, points_path, maximum_tracks)
        _plot_geometry_timeline(geometry, timeline_path, maximum_tracks)
        _plot_camera_motion(geometry, motion_path)
        static_scene = _relative_static_scene(geometry)
        static_scene_path.write_text(
            json.dumps(static_scene, indent=2),
            encoding="utf-8",
        )
        sandbox_component_paths = _plot_relative_static_sandbox(
            static_scene,
            sandbox_path,
            sandbox_component_root,
        )

        manifest_rows.append(
            {
                "video_id": geometry.video_id,
                "coordinate_space": "camera_3d",
                "coordinate_convention": geometry.intrinsics.coordinate_convention,
                "coordinate_unit": _unit_label(geometry),
                "world_trajectory_claimed": False,
                "frame_count": geometry.frame_count,
                "track_count": len(geometry.tracks),
                "geometry_observation_count": sum(len(track.observations) for track in geometry.tracks),
                "unavailable_observation_count": sum(len(track.unavailable_observations) for track in geometry.tracks),
                "camera_pose_count": len(geometry.camera_motion.poses),
                "frame_paths": [path.relative_to(output_root).as_posix() for path in frame_paths],
                "contact_sheet": sheet_path.relative_to(output_root).as_posix(),
                "depth_geometry_examples": [path.relative_to(output_root).as_posix() for path in depth_paths],
                "camera_centric_points_3d": points_path.relative_to(output_root).as_posix(),
                "geometry_timeline": timeline_path.relative_to(output_root).as_posix(),
                "camera_motion_diagnostics": motion_path.relative_to(output_root).as_posix(),
                "relative_static_scene": static_scene_path.relative_to(output_root).as_posix(),
                "relative_static_sandbox_3d": sandbox_path.relative_to(output_root).as_posix(),
                "relative_static_sandbox_components": [
                    path.relative_to(output_root).as_posix()
                    for path in sandbox_component_paths
                ],
                "relative_static_scene_summary": static_scene["summary"],
                "video": video_path.relative_to(output_root).as_posix() if render_video else None,
            }
        )

    manifest_path = output_root / "step4_visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_name": "step4_visualization_manifest",
                "schema_version": 3,
                "run_id": store.run_id,
                "geometry_store_sha256": sha256_file(store_path),
                "semantic_warning": (
                    "Point sequences are per-frame camera-centric geometry. They are not "
                    "metric world trajectories until a later stage estimates and validates world motion. "
                    "The static-scene sandbox uses normalized relative translation and preserves "
                    "disconnected pose components."
                ),
                "videos": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path
