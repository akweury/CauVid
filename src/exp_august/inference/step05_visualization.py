"""Diagnostic visualizations for the initial Step 5 world-hypothesis beam."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.exp_august.contracts import VideoWorldStateManifest, WorldStateStore
from src.exp_august.contracts.codec import read_contract, sha256_file


def _color(key: str) -> tuple[float, float, float]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return tuple((55 + value % 180) / 255.0 for value in digest[:3])


def _point(row) -> np.ndarray:
    return np.asarray((row.position.x, row.position.z, -row.position.y), dtype=np.float64)


def _set_axis_geometry(axis, points: list[np.ndarray]) -> None:
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
    lateral = max(float(spans[0]), 0.08 * float(spans[1]))
    vertical = max(float(spans[2]), 0.06 * float(spans[1]))
    forward = max(float(spans[1]), 2.5 * lateral, 4.0 * vertical)
    axis.set_box_aspect((lateral, forward, vertical))


def _draw_component(axis, component, trajectories: list, *, compact: bool) -> None:
    plot_points: list[np.ndarray] = []
    ego = np.asarray([_point(row) for row in component.poses], dtype=np.float64)
    if ego.size:
        plot_points.extend(ego)
        axis.plot(
            ego[:, 0], ego[:, 1], ego[:, 2], "o-",
            color="#d62728", linewidth=1.8 if compact else 2.5,
            markersize=3.5 if compact else 5.5, label="ego",
        )
        endpoints = [(0, "start")]
        if len(ego) > 1:
            endpoints.append((-1, "end"))
        for index, role in endpoints:
            row = component.poses[index]
            axis.text(
                ego[index, 0], ego[index, 1], ego[index, 2],
                f" {role} f{row.frame_index}", color="#8b1a1a",
                fontsize=6 if compact else 8,
            )
    for trajectory in trajectories:
        points = np.asarray([_point(row) for row in trajectory.observations], dtype=np.float64)
        if not points.size:
            continue
        plot_points.extend(points)
        color = _color(trajectory.track_id)
        marker = {"static": "s", "moving": "o", "ambiguous": "D", "unobservable": "x"}[
            trajectory.motion_state.value
        ]
        axis.plot(
            points[:, 0], points[:, 1], points[:, 2], "-",
            color=color, linewidth=1.2 if compact else 1.8, alpha=0.82,
        )
        axis.scatter(
            points[:, 0], points[:, 1], points[:, 2], marker=marker,
            color=[color], s=18 if compact else 35,
            label=f"{trajectory.class_name} {trajectory.motion_state.value}",
        )
        if not compact:
            axis.text(
                points[-1, 0], points[-1, 1], points[-1, 2],
                f" {trajectory.track_id.rsplit(':', 1)[-1]}", fontsize=7,
            )
    _set_axis_geometry(axis, plot_points)
    axis.set_title(
        f"{component.component_id} | frames {component.frame_indices[0]}-"
        f"{component.frame_indices[-1]} | objects={len(trajectories)}",
        fontsize=9 if compact else 13,
    )
    axis.set_xlabel("X right", fontsize=7 if compact else 10)
    axis.set_ylabel("Z forward", fontsize=7 if compact else 10)
    axis.set_zlabel("Y up", fontsize=7 if compact else 10)
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(6))
    axis.zaxis.set_major_locator(MaxNLocator(4))
    axis.tick_params(labelsize=6 if compact else 8)
    axis.view_init(elev=18, azim=-8)


def _plot_world(
    hypothesis,
    overview_path: Path,
    component_root: Path,
    maximum_objects: int,
) -> list[Path]:
    ranked_trajectories = sorted(
        hypothesis.object_trajectories,
        key=lambda row: (-len(row.observations), row.trajectory_id),
    )[:maximum_objects]
    by_component: dict[str, list] = defaultdict(list)
    for row in ranked_trajectories:
        by_component[row.component_id].append(row)
    components = hypothesis.ego_components
    columns = min(3, max(1, len(components)))
    rows = max(1, math.ceil(len(components) / columns))
    figure = plt.figure(figsize=(6.4 * columns, 5.0 * rows), dpi=150)
    if components:
        for index, component in enumerate(components, start=1):
            axis = figure.add_subplot(rows, columns, index, projection="3d")
            _draw_component(axis, component, by_component[component.component_id], compact=True)
    else:
        axis = figure.add_subplot(111, projection="3d")
        axis.text2D(0.5, 0.5, "World reconstruction unobservable", transform=axis.transAxes, ha="center")
    figure.suptitle(
        f"Initial world hypothesis rank {hypothesis.rank} | "
        f"frame={hypothesis.world_frame_status} | unit={hypothesis.coordinate_unit.value}\n"
        "Red: ego; objects are transformed after ego-motion compensation. Not Step 6 verified.",
        fontsize=14, y=0.995,
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.90, wspace=0.05)
    figure.savefig(overview_path)
    plt.close(figure)

    component_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for component in components:
        path = component_root / f"{component.component_id.replace(':', '_')}.png"
        detail = plt.figure(figsize=(12.8, 7.2), dpi=150)
        axis = detail.add_subplot(111, projection="3d")
        _draw_component(axis, component, by_component[component.component_id], compact=False)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles[:12], labels[:12], loc="upper left", fontsize=8)
        detail.suptitle(
            f"Step 5 component-local ego/object reconstruction: {component.component_id}\n"
            f"{hypothesis.coordinate_unit.value}; independent origin; uncertainty retained",
            fontsize=15, y=0.98,
        )
        detail.subplots_adjust(left=0.04, right=0.96, bottom=0.08, top=0.86)
        detail.savefig(path)
        plt.close(detail)
        paths.append(path)
    return paths


def _plot_motion(hypothesis, output_path: Path, maximum_objects: int) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), dpi=150, sharex=False)
    for component in hypothesis.ego_components:
        rows = [row for row in component.poses if row.speed is not None]
        if not rows:
            continue
        times = np.asarray([row.timestamp_s for row in rows])
        speeds = np.asarray([row.speed for row in rows])
        lower = np.asarray([row.speed_interval[0] for row in rows])
        upper = np.asarray([row.speed_interval[1] for row in rows])
        axes[0].plot(times, speeds, linewidth=1.8, label=component.component_id)
        axes[0].fill_between(times, lower, upper, alpha=0.18)
    trajectories = sorted(
        hypothesis.object_trajectories,
        key=lambda row: (-len(row.observations), row.trajectory_id),
    )[:maximum_objects]
    for trajectory in trajectories:
        rows = [row for row in trajectory.observations if row.speed is not None]
        if not rows:
            continue
        times = np.asarray([row.timestamp_s for row in rows])
        speeds = np.asarray([row.speed for row in rows])
        lower = np.asarray([row.speed_interval[0] for row in rows])
        upper = np.asarray([row.speed_interval[1] for row in rows])
        color = _color(trajectory.track_id)
        track_label = trajectory.track_id.rsplit(":", 1)[-1]
        label = (
            f"{trajectory.class_name} ID {track_label} "
            f"{trajectory.component_id.rsplit(':', 1)[-1]} "
            f"{trajectory.motion_state.value}"
        )
        axes[1].plot(times, speeds, color=color, linewidth=1.5, label=label)
        axes[1].fill_between(times, lower, upper, color=color, alpha=0.14)
    axes[0].set_title("Relative ego speed with propagated interval")
    axes[1].set_title("Ego-compensated object speed with propagated interval")
    for axis in axes:
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(f"Speed ({hypothesis.coordinate_unit.value}/s)")
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles[:12], labels[:12], fontsize=8, ncol=2)
    figure.suptitle(
        "Step 5 initial motion estimates — intervals are not metric when unit=relative_unit",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path)
    plt.close(figure)


def render_step5_visualizations(
    *,
    world_state_store_path: Path | str,
    maximum_objects: int = 12,
) -> Path:
    """Render the best initial hypothesis without treating it as verified truth."""

    if maximum_objects <= 0:
        raise ValueError("maximum_objects must be positive")
    store_path = Path(world_state_store_path).expanduser().resolve()
    store = read_contract(store_path, WorldStateStore)
    stage_root = store_path.parent
    visualization_root = stage_root / "visualizations"
    visualization_root.mkdir(parents=True, exist_ok=True)
    videos = []
    for video_id, reference in zip(store.video_ids, store.video_world_states):
        source_path = stage_root / reference.relative_path
        if not source_path.is_file() or source_path.stat().st_size != reference.byte_size:
            raise RuntimeError(f"Step 5 world-state manifest is missing or truncated: {source_path}")
        if sha256_file(source_path) != reference.sha256:
            raise RuntimeError(f"Step 5 world-state manifest failed integrity check: {source_path}")
        manifest = read_contract(source_path, VideoWorldStateManifest)
        hypothesis = manifest.initial_beam.hypotheses[0]
        video_root = visualization_root / video_id
        video_root.mkdir(parents=True, exist_ok=True)
        world_path = video_root / "initial_world_hypothesis_3d.png"
        motion_path = video_root / "initial_motion_intervals.png"
        component_paths = _plot_world(
            hypothesis,
            world_path,
            video_root / "components",
            maximum_objects,
        )
        _plot_motion(hypothesis, motion_path, maximum_objects)
        summary_path = video_root / "step5_summary.json"
        summary = {
            "schema_name": "step5_visualization_summary",
            "schema_version": 1,
            "video_id": video_id,
            "beam_id": manifest.initial_beam.beam_id,
            "visualized_hypothesis_id": hypothesis.hypothesis_id,
            "visualized_rank": hypothesis.rank,
            "hypothesis_count": len(manifest.initial_beam.hypotheses),
            "world_frame_status": hypothesis.world_frame_status,
            "coordinate_unit": hypothesis.coordinate_unit.value,
            "metric_scale_claimed": hypothesis.metric_scale_claimed,
            "step6_verified": False,
            "component_count": len(hypothesis.ego_components),
            "object_trajectory_count": len(hypothesis.object_trajectories),
            "motion_state_counts": {
                state: sum(row.motion_state.value == state for row in hypothesis.object_trajectories)
                for state in ("static", "moving", "ambiguous", "unobservable")
            },
            "limitations": list(hypothesis.limitations),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        videos.append(
            {
                "video_id": video_id,
                "visualized_hypothesis_id": hypothesis.hypothesis_id,
                "world_3d": world_path.relative_to(visualization_root).as_posix(),
                "motion_intervals": motion_path.relative_to(visualization_root).as_posix(),
                "component_world_3d": [
                    path.relative_to(visualization_root).as_posix() for path in component_paths
                ],
                "summary": summary_path.relative_to(visualization_root).as_posix(),
            }
        )
    manifest_path = visualization_root / "step5_visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_name": "step5_visualization_manifest",
                "schema_version": 1,
                "source_world_state_store": store_path.as_posix(),
                "videos": videos,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


__all__ = ["render_step5_visualizations"]
