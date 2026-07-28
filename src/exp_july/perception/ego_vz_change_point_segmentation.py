"""Deterministic constrained change-point segmentation for ego longitudinal motion."""

from __future__ import annotations

import math
from statistics import median


VERSION = 1
STATES = ("backward", "static", "forward")
STATE_INDEX = {state: index for index, state in enumerate(STATES)}

DEFAULT_CONFIG = {
    "min_segment_length": 5,
    "segment_penalty": 3.0,
    "static_band_floor": 0.05,
    "static_band_noise_multiplier": 2.5,
    "robust_clip_multiplier": 4.0,
    "uncertain_variance_multiplier": 9.0,
    "boundary_change_multiplier": 2.0,
    "boundary_stride": 12,
    "max_candidate_boundaries": 320,
}


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _mad(values, center=None):
    if not values:
        return 0.0
    center = float(median(values) if center is None else center)
    return float(median(abs(float(value) - center) for value in values))


def _prefix(values):
    total = [0.0]
    square = [0.0]
    for value in values:
        total.append(total[-1] + value)
        square.append(square[-1] + value * value)
    return total, square


def _range_stats(prefix_sum, prefix_square, start, end):
    count = end - start
    total = prefix_sum[end] - prefix_sum[start]
    square = prefix_square[end] - prefix_square[start]
    mean = total / max(1, count)
    variance = max(0.0, square / max(1, count) - mean * mean)
    return mean, variance


def _state(level, static_band):
    if level > static_band:
        return "forward"
    if level < -static_band:
        return "backward"
    return "static"


def _transition_allowed(previous, current):
    return previous is None or abs(STATE_INDEX[previous] - STATE_INDEX[current]) <= 1


def _candidate_boundaries(values, minimum, noise_scale, config):
    """Prune candidate boundaries while retaining endpoints and strong changes."""
    count = len(values)
    boundaries = {0, count}
    stride = max(minimum, int(config["boundary_stride"]))
    boundaries.update(range(minimum, count, stride))
    trigger = float(config["boundary_change_multiplier"]) * max(noise_scale, 1e-9)
    for index in range(minimum, count - minimum + 1):
        if abs(values[index] - values[index - 1]) >= trigger:
            boundaries.add(index)
    ordered = sorted(boundaries)
    limit = max(2, int(config["max_candidate_boundaries"]))
    if len(ordered) > limit:
        interior = ordered[1:-1]
        keep = limit - 2
        selected = {
            interior[round(index * (len(interior) - 1) / max(1, keep - 1))]
            for index in range(keep)
        }
        ordered = [0, *sorted(selected), count]
    return ordered


def _merge_identical(segments, values):
    merged = []
    for segment in segments:
        if merged and merged[-1]["state"] == segment["state"]:
            merged[-1]["end_index"] = segment["end_index"]
        else:
            merged.append(dict(segment))
    for segment in merged:
        subset = values[segment["start_index"] : segment["end_index"]]
        level = float(median(subset))
        segment["robust_level"] = level
        segment["residual_variance"] = float(
            sum((value - level) ** 2 for value in subset) / max(1, len(subset))
        )
    return merged


def _segment_run(values, frame_indices, config, global_noise):
    count = len(values)
    minimum = max(1, min(int(config["min_segment_length"]), count))
    center = float(median(values))
    scale = max(global_noise, 1.4826 * _mad(values, center), 1e-6)
    static_band = max(
        float(config["static_band_floor"]),
        float(config["static_band_noise_multiplier"]) * global_noise,
    )
    clip = float(config["robust_clip_multiplier"]) * scale
    clipped = [min(center + clip, max(center - clip, value)) for value in values]
    prefix_sum, prefix_square = _prefix(clipped)
    boundaries = _candidate_boundaries(clipped, minimum, global_noise, config)
    penalty_scale = max(global_noise, float(config["static_band_floor"]), 1e-6)
    penalty = float(config["segment_penalty"]) * penalty_scale * penalty_scale

    # dp[(end, state)] = (cost, start, previous_state)
    dp = {}
    for end_pos, end in enumerate(boundaries[1:], start=1):
        for start in boundaries[:end_pos]:
            if end - start < minimum and not (start == 0 and end == count):
                continue
            fit_level, fit_variance = _range_stats(
                prefix_sum, prefix_square, start, end
            )
            state = _state(fit_level, static_band)
            fit_cost = fit_variance * (end - start)
            if start == 0:
                proposal = (fit_cost, start, None)
            else:
                predecessors = [
                    (dp[(start, previous)][0], previous)
                    for previous in STATES
                    if (start, previous) in dp
                    and _transition_allowed(previous, state)
                ]
                if not predecessors:
                    continue
                previous_cost, previous_state = min(
                    predecessors, key=lambda row: (row[0], STATE_INDEX[row[1]])
                )
                proposal = (previous_cost + penalty + fit_cost, start, previous_state)
            key = (end, state)
            if key not in dp or proposal[0] < dp[key][0]:
                dp[key] = proposal

    endings = [(dp[(count, state)][0], state) for state in STATES if (count, state) in dp]
    if not endings:
        level = float(median(values))
        raw = [{
            "start_index": 0,
            "end_index": count,
            "state": _state(level, static_band),
        }]
        objective = 0.0
    else:
        objective, state = min(endings, key=lambda row: (row[0], STATE_INDEX[row[1]]))
        raw = []
        end = count
        while end > 0:
            _, start, previous_state = dp[(end, state)]
            raw.append({"start_index": start, "end_index": end, "state": state})
            end, state = start, previous_state
        raw.reverse()
    segments = _merge_identical(raw, values)
    variance_reference = max(global_noise, static_band / max(1.0, float(config["static_band_noise_multiplier"])), 1e-6)
    variance_limit = float(config["uncertain_variance_multiplier"]) * variance_reference * variance_reference
    for segment in segments:
        duration = segment["end_index"] - segment["start_index"]
        variance = float(segment["residual_variance"])
        unreliable = variance > variance_limit
        separation = abs(float(segment["robust_level"])) / max(static_band, 1e-9)
        if segment["state"] == "static":
            separation = max(0.0, 1.0 - separation)
        else:
            separation = min(1.0, max(0.0, separation - 1.0))
        stability = 1.0 / (1.0 + variance / max(scale * scale, 1e-12))
        length_support = min(1.0, duration / max(1.0, 2.0 * minimum))
        segment["confidence"] = float(
            max(0.0, min(1.0, 0.45 * separation + 0.4 * stability + 0.15 * length_support))
        )
        if unreliable:
            segment["confidence"] = min(segment["confidence"], 0.49)
        segment["start_frame"] = int(frame_indices[segment["start_index"]])
        segment["end_frame"] = int(frame_indices[segment["end_index"] - 1])
        segment["duration_frames"] = int(duration)
        segment["adaptive_static_band"] = float(static_band)
        segment["noise_scale"] = float(scale)
    return segments, {
        "objective": float(objective),
        "adaptive_static_band": float(static_band),
        "noise_scale": float(scale),
        "candidate_boundaries": [int(value) for value in boundaries],
        "candidate_boundary_count": len(boundaries),
    }


def segment_ego_vz(samples, config=None):
    """Segment available, frame-contiguous ego-vz runs with constrained DP."""
    resolved = dict(DEFAULT_CONFIG)
    resolved.update(dict(config or {}))
    available_values = [
        float(sample["ego_vz"])
        for sample in samples
        if sample.get("available") and _finite(sample.get("ego_vz")) is not None
    ]
    differences = [
        right - left for left, right in zip(available_values, available_values[1:])
    ]
    global_noise = max(1e-6, 1.4826 * _mad(differences) / math.sqrt(2.0))
    segments = []
    run_samples = []

    def flush():
        if not run_samples:
            return
        values = [float(row["ego_vz"]) for row in run_samples]
        indices = [int(row["frame_index"]) for row in run_samples]
        run_segments, run_audit = _segment_run(values, indices, resolved, global_noise)
        for row in run_segments:
            row["run_audit"] = run_audit
            row["sample_indices"] = [
                int(run_samples[index]["sample_index"])
                for index in range(row.pop("start_index"), row.pop("end_index"))
            ]
            segments.append(row)
        run_samples.clear()

    previous_frame = None
    for sample_index, sample in enumerate(samples):
        frame = int(sample.get("frame_index", sample_index))
        value = _finite(sample.get("ego_vz"))
        contiguous = previous_frame is None or frame == previous_frame + 1
        if not sample.get("available") or value is None or not contiguous:
            flush()
        if sample.get("available") and value is not None:
            run_samples.append({**sample, "sample_index": sample_index, "ego_vz": value})
        else:
            segments.append({
                "state": "unknown",
                "start_frame": frame,
                "end_frame": frame,
                "duration_frames": 1,
                "sample_indices": [sample_index],
                "confidence": 0.0,
                "robust_level": None,
                "residual_variance": None,
                "adaptive_static_band": None,
                "noise_scale": float(global_noise),
                "run_audit": {"candidate_boundaries": [], "candidate_boundary_count": 0},
            })
        previous_frame = frame
    flush()

    # Merge only adjacent identical states; missing observations remain explicit.
    merged = []
    for row in sorted(segments, key=lambda item: item["sample_indices"][0]):
        if (
            merged
            and row["state"] == merged[-1]["state"]
            and row["start_frame"] == merged[-1]["end_frame"] + 1
        ):
            merged[-1]["end_frame"] = row["end_frame"]
            merged[-1]["duration_frames"] += row["duration_frames"]
            merged[-1]["sample_indices"].extend(row["sample_indices"])
        else:
            merged.append(dict(row))
    for segment_id, row in enumerate(merged):
        row["segment_id"] = segment_id
        row["action"] = row["state"]
        row["provenance"] = {
            "source_step": "07_ego_motion",
            "source_signal": "ego_vz",
            "method": "constrained_change_point_dynamic_programming",
            "robust_fit": "globally_winsorized_prefix_sse",
            "transition_constraint": "forward<->static<->backward",
            "candidate_boundary_pruning": True,
        }
    return {
        "segments": merged,
        "global_noise_scale": float(global_noise),
        "configuration": resolved,
        "method": "constrained_change_point_dynamic_programming",
    }
