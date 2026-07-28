"""Per-video ego vx/vz segmentation by stable segment-count threshold plateaus."""

from __future__ import annotations

import math
from pathlib import Path


VERSION = 10
NUM_THRESHOLDS = 100


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _signal(frame, axis):
    for key in (f"refined_ego_{axis}", f"ego_{axis}_smoothed", f"ego_{axis}"):
        value = _finite(frame.get(key))
        if value is not None:
            return value
    return None


def _label(value, threshold, negative, center, positive):
    if value < -threshold:
        return negative
    if value > threshold:
        return positive
    return center


def _segments(frames, axis, threshold, labels):
    negative, center, positive = labels
    rows = []
    active = None
    previous_frame = None
    for offset, frame in enumerate(frames):
        frame_index = int(frame.get("frame_index", offset))
        value = _signal(frame, axis)
        if value is None:
            if active is not None:
                rows.append(active)
                active = None
            previous_frame = None
            continue
        state = _label(value, threshold, negative, center, positive)
        contiguous = previous_frame is not None and frame_index == previous_frame + 1
        if active is None or active["state"] != state or not contiguous:
            if active is not None:
                rows.append(active)
            active = {
                "state": state,
                "start_frame": frame_index,
                "end_frame": frame_index,
                "duration_frames": 1,
            }
        else:
            active["end_frame"] = frame_index
            active["duration_frames"] += 1
        previous_frame = frame_index
    if active is not None:
        rows.append(active)
    for segment_id, row in enumerate(rows):
        row["segment_id"] = segment_id
    return rows


def filter_short_state_interruptions(segments, tolerance_frames):
    """Bridge short state interruptions between two persistent equal-state segments."""
    tolerance = max(0, int(tolerance_frames))
    rows = [dict(row) for row in segments]
    if tolerance <= 0 or len(rows) < 3:
        return rows
    changed = True
    while changed:
        changed = False
        for left_index, left in enumerate(rows[:-2]):
            if int(left.get("duration_frames", 0)) <= tolerance:
                continue
            for right_index in range(left_index + 2, len(rows)):
                right = rows[right_index]
                interruption_frames = int(right["start_frame"]) - int(left["end_frame"]) - 1
                if interruption_frames > tolerance:
                    break
                if (right.get("state") == left.get("state")
                        and int(right.get("duration_frames", 0)) > tolerance):
                    merged = dict(left)
                    merged["end_frame"] = int(right["end_frame"])
                    merged["duration_frames"] = merged["end_frame"] - int(merged["start_frame"]) + 1
                    merged["noise_filter_merged"] = True
                    merged["absorbed_interruption_frames"] = interruption_frames
                    merged["absorbed_states"] = [str(row.get("state", "")) for row in rows[left_index + 1:right_index]]
                    rows[left_index:right_index + 1] = [merged]
                    changed = True
                    break
            if changed:
                break
    for segment_id, row in enumerate(rows):
        row["segment_id"] = segment_id
    return rows


def _filtered_segments(frames, axis, threshold, labels, tolerance_frames):
    return filter_short_state_interruptions(
        _segments(frames, axis, threshold, labels), tolerance_frames,
    )


def _plateaus(candidate_rows):
    plateaus = []
    start = 0
    for index in range(1, len(candidate_rows) + 1):
        if (
            index < len(candidate_rows)
            and candidate_rows[index]["segment_count"]
            == candidate_rows[start]["segment_count"]
        ):
            continue
        chunk = candidate_rows[start:index]
        middle = len(chunk) // 2
        midpoint = (
            chunk[middle]["threshold"]
            if len(chunk) % 2
            else 0.5 * (chunk[middle - 1]["threshold"] + chunk[middle]["threshold"])
        )
        plateaus.append({
            "plateau_id": len(plateaus),
            "start_index": start,
            "end_index": index - 1,
            "num_n_values": len(chunk),
            "segment_count": int(chunk[0]["segment_count"]),
            "threshold_start": float(chunk[0]["threshold"]),
            "threshold_end": float(chunk[-1]["threshold"]),
            "midpoint_n": float(midpoint),
        })
        start = index
    return plateaus


def segment_axis(frames, axis, labels, num_thresholds=NUM_THRESHOLDS, noise_tolerance_frames=5):
    values = [
        value
        for frame in frames
        for value in [_signal(frame, axis)]
        if value is not None
    ]
    maximum = max((abs(value) for value in values), default=0.0)
    if maximum <= 0.0:
        thresholds = [0.0]
    else:
        thresholds = [
            maximum * index / (num_thresholds + 1)
            for index in range(1, num_thresholds + 1)
        ]
    candidates = []
    for index, threshold in enumerate(thresholds):
        candidates.append({
            "candidate_index": index,
            "threshold": float(threshold),
            "segment_count": len(_filtered_segments(frames, axis, threshold, labels, noise_tolerance_frames)),
            "raw_segment_count": len(_segments(frames, axis, threshold, labels)),
        })
    all_plateaus = _plateaus(candidates)
    qualifying = []
    for plateau in all_plateaus:
        # More than five sampled N values and more than one temporal segment.
        if plateau["num_n_values"] <= 5 or plateau["segment_count"] <= 1:
            continue
        row = dict(plateau)
        row["candidate_optimal_n"] = float(row["midpoint_n"])
        row["segments"] = _filtered_segments(
            frames, axis, row["midpoint_n"], labels, noise_tolerance_frames,
        )
        qualifying.append(row)
    return {
        "axis": axis,
        "labels": {
            "negative": labels[0],
            "center": labels[1],
            "positive": labels[2],
        },
        "maximum_absolute_signal": float(maximum),
        "num_threshold_candidates": len(candidates),
        "threshold_candidates": candidates,
        "all_plateaus": all_plateaus,
        "qualifying_plateaus": qualifying,
        "noise_filter": {
            "method": "bridge_short_interruptions_between_persistent_equal_states",
            "tolerance_frames": max(0, int(noise_tolerance_frames)),
            "persistent_anchor_minimum_frames_exclusive": max(0, int(noise_tolerance_frames)),
            "interruption_measure": "total_frame_span_between_anchor_segments",
        },
        "plateau_filter": {
            "minimum_n_values_exclusive": 5,
            "exclude_single_segment_plateaus": True,
        },
    }


def segment_video(ego_video, vx_noise_tolerance_frames=5, vz_noise_tolerance_frames=5):
    frames = list(ego_video.get("frames", []))
    vz = segment_axis(
        frames, "vz", ("backward", "static", "forward"),
        noise_tolerance_frames=vz_noise_tolerance_frames,
    )
    vx = segment_axis(
        frames, "vx", ("right", "straight", "left"),
        noise_tolerance_frames=vx_noise_tolerance_frames,
    )
    frame_rows = []
    for offset, frame in enumerate(frames):
        frame_rows.append({
            "frame_index": int(frame.get("frame_index", offset)),
            "ego_vx": _signal(frame, "vx"),
            "ego_vz": _signal(frame, "vz"),
        })
    return {
        "version": VERSION,
        "video_id": str(ego_video.get("video_id", "")),
        "status": "completed",
        "method": "multi_plateau_segment_count_stability",
        "num_frames": len(frames),
        "vz_segmentation": vz,
        "vx_segmentation": vx,
        "frames": frame_rows,
        "provenance": {
            "source": "continuous_ego_motion",
            "threshold_candidates_per_axis": NUM_THRESHOLDS,
            "selection": "all_plateaus_over_five_n_values_excluding_single_segment",
            "single_final_n_selected": False,
            "noise_tolerance_frames": {
                "vx": max(0, int(vx_noise_tolerance_frames)),
                "vz": max(0, int(vz_noise_tolerance_frames)),
            },
            "deterministic": True,
        },
    }


def render_segment_count_chart(result, output_path):
    """Render all qualifying vx/vz segment-count plateaus for one video."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for axis, key, title, color in (
        (axes[0], "vx_segmentation", "VX: right | straight | left", "#2878B5"),
        (axes[1], "vz_segmentation", "VZ: backward | static | forward", "#D95319"),
    ):
        data = result[key]
        candidates = data["threshold_candidates"]
        thresholds = [row["threshold"] for row in candidates]
        counts = [row["segment_count"] for row in candidates]
        axis.plot(thresholds, counts, color=color, linewidth=2.3, marker=".", markersize=4)
        for plateau_index, plateau in enumerate(data["qualifying_plateaus"]):
            midpoint = plateau["midpoint_n"]
            count = plateau["segment_count"]
            axis.axvspan(
                plateau["threshold_start"], plateau["threshold_end"],
                color="#65C18C", alpha=0.22,
                label="qualifying plateau" if plateau_index == 0 else None,
            )
            axis.axvline(midpoint, color="#7A1FA2", linestyle="--", linewidth=1.8)
            axis.scatter([midpoint], [count], s=75, color="#7A1FA2", edgecolors="white", linewidths=1.2, zorder=5)
            axis.annotate(
                f"middle N={midpoint:.5g}\nsegments={count}",
                xy=(midpoint, count), xytext=(7, 13), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color="#4A1268",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#7A1FA2", "alpha": 0.9},
            )
        axis.set_title(f"{title}\nqualifying plateaus={len(data['qualifying_plateaus'])}", fontsize=13, fontweight="bold")
        axis.set_xlabel("Threshold N", fontsize=11)
        axis.set_ylabel("Number of temporal segments", fontsize=11)
        axis.grid(True, alpha=0.25)
        if data["qualifying_plateaus"]:
            axis.legend(fontsize=9)
    figure.suptitle(
        f"Step 7A multi-plateau threshold stability | video={result.get('video_id', '')}",
        fontsize=15, fontweight="bold",
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return str(output_path)


def _confidence_surface(rows, grid_size=90, bounds=None):
    """Fit normalized Gaussian confidence c(middle_N, segment_count)."""
    if not rows:
        return None
    import numpy as np

    points = np.asarray(
        [[float(row["midpoint_n"]), float(row["segment_count"])] for row in rows],
        dtype=float,
    )
    count = len(points)
    x_values, y_values = points[:, 0], points[:, 1]
    x_span = float(np.ptp(x_values))
    y_span = float(np.ptp(y_values))
    x_reference = max(x_span, abs(float(np.median(x_values))) * 0.1, 1.0)
    y_reference = max(y_span, 1.0)
    factor = count ** (-1.0 / 6.0)
    bandwidth_x = max(float(np.std(x_values)) * factor, x_reference / 25.0, 1e-6)
    bandwidth_y = max(float(np.std(y_values)) * factor, y_reference / 25.0, 0.15)
    if bounds is None:
        x_min = float(np.min(x_values) - 2.5 * bandwidth_x)
        x_max = float(np.max(x_values) + 2.5 * bandwidth_x)
        y_min = float(np.min(y_values) - 2.5 * bandwidth_y)
        y_max = float(np.max(y_values) + 2.5 * bandwidth_y)
    else:
        x_min, x_max, y_min, y_max = (float(value) for value in bounds)
    x_grid = np.linspace(x_min, x_max, max(30, int(grid_size)))
    y_grid = np.linspace(y_min, y_max, max(30, int(grid_size)))
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    density = np.zeros_like(grid_x, dtype=float)
    for x_value, y_value in points:
        density += np.exp(-0.5 * (
            ((grid_x - x_value) / bandwidth_x) ** 2
            + ((grid_y - y_value) / bandwidth_y) ** 2
        ))
    density /= max(1, count)
    maximum = float(np.max(density))
    confidence = density / maximum if maximum > 0.0 else density
    peak_index = np.unravel_index(int(np.argmax(confidence)), confidence.shape)
    audit = {
        "method": "normalized_gaussian_kernel_confidence",
        "function": "c(plateau_middle_threshold, number_of_temporal_segments)",
        "range": [0.0, 1.0],
        "training_point_count": count,
        "bandwidth_middle_n": float(bandwidth_x),
        "bandwidth_segment_count": float(bandwidth_y),
        "peak_middle_n": float(grid_x[peak_index]),
        "peak_segment_count": float(grid_y[peak_index]),
        "peak_confidence": float(confidence[peak_index]),
        "grid_size": int(confidence.shape[0]),
        "bounds": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "gradient": "continuous_gaussian_decay",
    }
    return {
        "points": points,
        "x": grid_x,
        "y": grid_y,
        "confidence": confidence,
        "maximum_density": maximum,
        "bandwidth_x": bandwidth_x,
        "bandwidth_y": bandwidth_y,
        "audit": audit,
    }


def _confidence_at(model, row):
    """Evaluate normalized confidence for one plateau point."""
    if model is None or model["maximum_density"] <= 0.0:
        return None
    import numpy as np

    x_value = float(row["midpoint_n"])
    y_value = float(row["segment_count"])
    points = model["points"]
    density = float(np.mean(np.exp(-0.5 * (
        ((points[:, 0] - x_value) / model["bandwidth_x"]) ** 2
        + ((points[:, 1] - y_value) / model["bandwidth_y"]) ** 2
    ))))
    return float(max(0.0, min(1.0, density / model["maximum_density"])))


def render_all_video_plateau_scatter(
    results, output_path, eval_results=None,
    vx_seg_max_count=8, vz_seg_max_count=5,
    max_plateau_middle_th_vx=250.0, max_plateau_middle_th_vz=70.0,
):
    """Fit training confidence heat maps and evaluate held-out points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_results = list(eval_results or [])
    axis_limits = {"vx": max(1, int(vx_seg_max_count)), "vz": max(1, int(vz_seg_max_count))}
    midpoint_limits = {
        "vx": max(0.0, float(max_plateau_middle_th_vx)),
        "vz": max(0.0, float(max_plateau_middle_th_vz)),
    }
    plot_limits = {
        signal_axis: {
            "x_min": 0.0,
            "x_max": 1.2 * midpoint_limits[signal_axis],
            "y_min": 0.0,
            "y_max": 1.2 * axis_limits[signal_axis],
        }
        for signal_axis in ("vx", "vz")
    }

    def collect(source_results, split):
        rows = []
        for result in source_results:
            video_id = str(result.get("video_id", ""))
            for signal_axis, key in (("vx", "vx_segmentation"), ("vz", "vz_segmentation")):
                for plateau in result.get(key, {}).get("qualifying_plateaus", []):
                    segment_count = int(plateau["segment_count"])
                    midpoint_n = float(plateau["midpoint_n"])
                    disabled_reasons = []
                    if segment_count > axis_limits[signal_axis]:
                        disabled_reasons.append("segment_count_above_seg_max_count")
                    if midpoint_n > midpoint_limits[signal_axis]:
                        disabled_reasons.append("plateau_middle_n_above_maximum")
                    rows.append({
                        "video_id": video_id, "split": split, "axis": signal_axis,
                        "midpoint_n": midpoint_n, "segment_count": segment_count,
                        "plateau_id": int(plateau["plateau_id"]),
                        "enabled": not disabled_reasons,
                        "disabled_reason": disabled_reasons[0] if disabled_reasons else None,
                        "disabled_reasons": disabled_reasons,
                    })
        return rows

    points = collect(results, "train") + collect(eval_results, "eval")
    figure, axes = plt.subplots(1, 2, figsize=(17, 7.5), constrained_layout=True)
    styles = {"vx": {"color": "#2878B5", "marker": "o"}, "vz": {"color": "#D95319", "marker": "^"}}
    confidence_regions = {}
    evaluation_metrics = {}
    for plot_axis, signal_axis in zip(axes, ("vx", "vz")):
        rows = [row for row in points if row["axis"] == signal_axis]
        train_enabled = [row for row in rows if row["split"] == "train" and row["enabled"]]
        eval_enabled = [row for row in rows if row["split"] == "eval" and row["enabled"]]
        disabled_rows = [row for row in rows if not row["enabled"]]
        limits = plot_limits[signal_axis]
        model = _confidence_surface(
            train_enabled,
            bounds=(limits["x_min"], limits["x_max"], limits["y_min"], limits["y_max"]),
        )
        confidence_regions[signal_axis] = None if model is None else model["audit"]
        if model is not None:
            heatmap = plot_axis.contourf(
                model["x"], model["y"], model["confidence"],
                levels=np.linspace(0.0, 1.0, 13), cmap="YlOrRd", alpha=0.48, zorder=0,
            )
            figure.colorbar(heatmap, ax=plot_axis, fraction=0.046, pad=0.03, label="Confidence c(N, segments)")
            plot_axis.scatter(
                [model["audit"]["peak_middle_n"]], [model["audit"]["peak_segment_count"]],
                s=150, marker="*", color="#8B0000", edgecolors="white", linewidths=1.0,
                zorder=5, label="confidence peak",
            )
        for row in rows:
            row["confidence"] = _confidence_at(model, row) if row["enabled"] else None
        style = styles[signal_axis]
        plot_axis.scatter(
            [row["midpoint_n"] for row in train_enabled], [row["segment_count"] for row in train_enabled],
            s=70, alpha=0.82, color=style["color"], marker=style["marker"],
            edgecolors="white", linewidths=0.8, label="train enabled", zorder=4,
        )
        plot_axis.scatter(
            [row["midpoint_n"] for row in eval_enabled], [row["segment_count"] for row in eval_enabled],
            s=88, alpha=0.92, color="#8E44AD", marker="D",
            edgecolors="white", linewidths=1.0, label="eval enabled", zorder=5,
        )
        eval_confidences = [row["confidence"] for row in eval_enabled if row["confidence"] is not None]
        mean_confidence = None if not eval_confidences else float(sum(eval_confidences) / len(eval_confidences))
        metric = {
            "metric": "mean_eval_confidence",
            "value": mean_confidence,
            "enabled_eval_points": len(eval_enabled),
            "scored_eval_points": len(eval_confidences),
            "definition": "mean_c_N_segments_for_enabled_eval_plateaus_under_train_fitted_surface",
        }
        evaluation_metrics[signal_axis] = metric
        metric_text = "Mean eval confidence: N/A" if mean_confidence is None else f"Mean eval confidence: {mean_confidence:.3f}"
        plot_axis.text(
            0.02, 0.98, metric_text, transform=plot_axis.transAxes, va="top", ha="left",
            fontsize=11, fontweight="bold", color="#5B2C6F",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#8E44AD", "alpha": 0.92}, zorder=8,
        )
        plot_axis.scatter(
            [row["midpoint_n"] for row in disabled_rows], [row["segment_count"] for row in disabled_rows],
            s=65, alpha=0.55, color="#8A8A8A", marker="x", linewidths=1.2, label="disabled", zorder=4,
        )
        plot_axis.axhline(axis_limits[signal_axis], color="#666666", linestyle="--", linewidth=1.3)
        plot_axis.axvline(midpoint_limits[signal_axis], color="#999999", linestyle=":", linewidth=1.5)
        plot_axis.set_xlim(limits["x_min"], limits["x_max"])
        plot_axis.set_ylim(limits["y_min"], limits["y_max"])
        plot_axis.set_title(f"{signal_axis.upper()} confidence | train={len(train_enabled)} | eval={len(eval_enabled)}", fontsize=14, fontweight="bold")
        plot_axis.set_xlabel("Plateau middle threshold N", fontsize=12)
        plot_axis.set_ylabel("Number of temporal segments at N", fontsize=12)
        plot_axis.grid(True, alpha=0.2)
        plot_axis.legend(fontsize=8.5, loc="best")
    figure.suptitle("Step 7A train-fitted confidence c(N, temporal segments)", fontsize=17, fontweight="bold")
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return {
        "path": str(output_path), "split_ratio": "4:1",
        "num_train_videos": len(results), "num_eval_videos": len(eval_results),
        "vx_seg_max_count": axis_limits["vx"], "vz_seg_max_count": axis_limits["vz"],
        "seg_max_count_by_axis": dict(axis_limits),
        "max_plateau_middle_th_vx": midpoint_limits["vx"],
        "max_plateau_middle_th_vz": midpoint_limits["vz"],
        "max_plateau_middle_threshold_by_axis": dict(midpoint_limits),
        "plot_limits_by_axis": plot_limits,
        "confidence_regions": confidence_regions,
        "evaluation_metrics": evaluation_metrics,
        "num_points": len(points),
        "num_enabled_points": sum(row["enabled"] for row in points),
        "num_disabled_points": sum(not row["enabled"] for row in points),
        "points": points,
    }
