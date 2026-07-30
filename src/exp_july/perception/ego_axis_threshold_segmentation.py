"""Per-video ego vx/vz segmentation by stable segment-count threshold plateaus."""

from __future__ import annotations

import math
from pathlib import Path


VERSION = 18
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
                "signal_sum": float(value),
            }
        else:
            active["end_frame"] = frame_index
            active["duration_frames"] += 1
            active["signal_sum"] += float(value)
        previous_frame = frame_index
    if active is not None:
        rows.append(active)
    for segment_id, row in enumerate(rows):
        row["segment_id"] = segment_id
        row["mean_signal"] = float(row.pop("signal_sum") / max(1, int(row["duration_frames"])))
    return rows


def _weighted_mean_signal(rows):
    weighted = [(float(row["mean_signal"]), int(row.get("duration_frames", 0))) for row in rows if row.get("mean_signal") is not None]
    total = sum(weight for _, weight in weighted)
    return None if total <= 0 else float(sum(value * weight for value, weight in weighted) / total)


def _coalesce_assigned_segments(rows):
    output = []
    for source in rows:
        row = dict(source)
        if output and output[-1].get("state") == row.get("state"):
            previous = output[-1]
            combined_mean = _weighted_mean_signal([previous, row])
            previous["end_frame"] = int(row["end_frame"])
            previous["duration_frames"] = previous["end_frame"] - int(previous["start_frame"]) + 1
            previous["mean_signal"] = combined_mean
            assignments = list(previous.get("residual_short_assignments", []))
            assignments.extend(row.get("residual_short_assignments", []))
            if assignments:
                previous["residual_short_assignments"] = assignments
        else:
            output.append(row)
    for segment_id, row in enumerate(output):
        row["segment_id"] = segment_id
    return output


def merge_remaining_short_segments(segments, tolerance_frames):
    """Assign every residual short island to neighboring long segments."""
    tolerance = max(0, int(tolerance_frames))
    rows = [dict(row) for row in segments]
    if tolerance <= 0 or not rows:
        return rows
    long_indices = [index for index, row in enumerate(rows) if int(row.get("duration_frames", 0)) > tolerance]
    if not long_indices:
        dominant = max(rows, key=lambda row: (int(row.get("duration_frames", 0)), -int(row.get("segment_id", 0))))
        assignment_rows = []
        for row in rows:
            assigned = dict(row)
            assigned["residual_short_assignments"] = [{
                "original_state": str(row.get("state", "")), "assigned_state": str(dominant.get("state", "")),
                "duration_frames": int(row.get("duration_frames", 0)), "assignment_method": "global_dominant_state_no_long_anchor",
                "mean_signal": row.get("mean_signal"),
            }]
            assigned["state"] = dominant.get("state")
            assignment_rows.append(assigned)
        output = _coalesce_assigned_segments(assignment_rows)
        if output and int(output[0].get("duration_frames", 0)) <= tolerance:
            output[0]["short_segment_unavoidable_entire_sequence"] = True
        return output
    index = 0
    while index < len(rows):
        if int(rows[index].get("duration_frames", 0)) > tolerance:
            index += 1
            continue
        start = index
        while index < len(rows) and int(rows[index].get("duration_frames", 0)) <= tolerance:
            index += 1
        end = index
        island = rows[start:end]
        left = rows[start - 1] if start > 0 and int(rows[start - 1].get("duration_frames", 0)) > tolerance else None
        right = rows[end] if end < len(rows) and int(rows[end].get("duration_frames", 0)) > tolerance else None
        if left is not None and right is not None:
            left_mean, right_mean = left.get("mean_signal"), right.get("mean_signal")
            def distance(row, anchor_mean):
                if row.get("mean_signal") is None or anchor_mean is None:
                    return 0.0
                return abs(float(row["mean_signal"]) - float(anchor_mean)) * max(1, int(row.get("duration_frames", 0)))
            split_candidates = []
            for split in range(len(island) + 1):
                cost = sum(distance(row, left_mean) for row in island[:split]) + sum(distance(row, right_mean) for row in island[split:])
                split_candidates.append((cost, abs(split - len(island) / 2.0), split))
            _, _, split = min(split_candidates)
        elif left is not None:
            split = len(island)
        else:
            split = 0
        for offset, row in enumerate(island):
            target = left if offset < split else right
            if target is None:
                target = left or right
            assigned_state = str(target.get("state", row.get("state", ""))) if target is not None else str(row.get("state", ""))
            row_mean = row.get("mean_signal")
            left_distance = None if left is None or row_mean is None or left.get("mean_signal") is None else abs(float(row_mean) - float(left["mean_signal"]))
            right_distance = None if right is None or row_mean is None or right.get("mean_signal") is None else abs(float(row_mean) - float(right["mean_signal"]))
            row["residual_short_assignments"] = [{
                "original_state": str(row.get("state", "")), "assigned_state": assigned_state,
                "duration_frames": int(row.get("duration_frames", 0)), "mean_signal": row_mean,
                "left_anchor_state": None if left is None else str(left.get("state", "")),
                "right_anchor_state": None if right is None else str(right.get("state", "")),
                "left_anchor_mean_signal": None if left is None else left.get("mean_signal"),
                "right_anchor_mean_signal": None if right is None else right.get("mean_signal"),
                "left_signal_distance": left_distance, "right_signal_distance": right_distance,
                "selected_side": "left" if target is left else "right",
                "assignment_method": "nearest_neighbor_mean_signal_monotonic_split" if left is not None and right is not None else "single_available_long_neighbor",
            }]
            row["state"] = assigned_state
    return _coalesce_assigned_segments(rows)


def filter_short_state_interruptions(
    segments, tolerance_frames, bridge_total_max_frames=15,
    anchor_min_frames=8, bridge_max_segments=5,
    bridge_max_anchor_ratio=0.75,
):
    """Bridge a bounded sequence of individually short states between equal anchors."""
    tolerance = max(0, int(tolerance_frames))
    total_limit = max(0, int(bridge_total_max_frames))
    anchor_minimum = max(1, int(anchor_min_frames))
    maximum_segments = max(1, int(bridge_max_segments))
    maximum_ratio = max(0.0, float(bridge_max_anchor_ratio))
    rows = [dict(row) for row in segments]
    if tolerance <= 0 or total_limit <= 0 or maximum_ratio <= 0.0 or len(rows) < 3:
        return rows
    changed = True
    while changed:
        changed = False
        for left_index, left in enumerate(rows[:-2]):
            left_duration = int(left.get("duration_frames", 0))
            if left_duration < anchor_minimum:
                continue
            last_right_index = min(len(rows), left_index + maximum_segments + 2)
            for right_index in range(left_index + 2, last_right_index):
                right = rows[right_index]
                interruption_frames = int(right["start_frame"]) - int(left["end_frame"]) - 1
                if interruption_frames > total_limit:
                    break
                right_duration = int(right.get("duration_frames", 0))
                if right.get("state") != left.get("state") or right_duration < anchor_minimum:
                    continue
                interruptions = rows[left_index + 1:right_index]
                if not interruptions or any(
                    int(row.get("duration_frames", 0)) > tolerance
                    for row in interruptions
                ):
                    continue
                anchor_ratio = interruption_frames / max(1, min(left_duration, right_duration))
                if anchor_ratio > maximum_ratio:
                    continue
                merged = dict(left)
                merged["end_frame"] = int(right["end_frame"])
                merged["duration_frames"] = merged["end_frame"] - int(merged["start_frame"]) + 1
                merged["mean_signal"] = _weighted_mean_signal(rows[left_index:right_index + 1])
                merged["noise_filter_merged"] = True
                merged["absorbed_interruption_frames"] = interruption_frames
                merged["absorbed_segment_count"] = len(interruptions)
                merged["absorbed_states"] = [str(row.get("state", "")) for row in interruptions]
                merged["bridge_anchor_ratio"] = float(anchor_ratio)
                rows[left_index:right_index + 1] = [merged]
                changed = True
                break
            if changed:
                break
    for segment_id, row in enumerate(rows):
        row["segment_id"] = segment_id
    return rows


def _filtered_segments(frames, axis, threshold, labels, tolerance_frames, bridge_config=None):
    config = dict(bridge_config or {})
    bridged = filter_short_state_interruptions(
        _segments(frames, axis, threshold, labels), tolerance_frames,
        bridge_total_max_frames=config.get("bridge_total_max_frames", 15),
        anchor_min_frames=config.get("anchor_min_frames", 8),
        bridge_max_segments=config.get("bridge_max_segments", 5),
        bridge_max_anchor_ratio=config.get("bridge_max_anchor_ratio", 0.75),
    )
    return merge_remaining_short_segments(bridged, tolerance_frames)

def changed_label_confidence(segment_length, frame_offset, minimum_long_length):
    """Symmetric confidence valley for a relabeled source segment."""
    length = max(1, int(segment_length))
    offset = min(max(0, int(frame_offset)), length - 1)
    minimum_long = max(1, int(minimum_long_length))
    depth = min(1.0, length / minimum_long)
    if length <= 2:
        shape = 1.0
    else:
        distance_to_edge = min(offset, length - 1 - offset)
        middle_distance = max(1, (length - 1) // 2)
        shape = min(1.0, distance_to_edge / middle_distance)
    return float(max(0.0, min(1.0, 1.0 - depth * shape)))


def frame_label_confidences(frames, raw_segments, filtered_segments, minimum_long_length):
    """Return original/final labels and confidence for every observed frame."""
    raw_by_frame = {}
    final_by_frame = {}
    for segment in raw_segments:
        for frame_index in range(int(segment["start_frame"]), int(segment["end_frame"]) + 1):
            raw_by_frame[frame_index] = segment
    for segment in filtered_segments:
        for frame_index in range(int(segment["start_frame"]), int(segment["end_frame"]) + 1):
            final_by_frame[frame_index] = segment
    output = []
    for offset, frame in enumerate(frames):
        frame_index = int(frame.get("frame_index", offset))
        raw = raw_by_frame.get(frame_index)
        final = final_by_frame.get(frame_index)
        if raw is None or final is None:
            continue
        original_label = str(raw.get("state", ""))
        final_label = str(final.get("state", ""))
        changed = original_label != final_label
        source_length = int(raw.get("duration_frames", 1))
        source_offset = frame_index - int(raw["start_frame"])
        confidence = changed_label_confidence(
            source_length, source_offset, minimum_long_length,
        ) if changed else 1.0
        output.append({
            "frame_index": frame_index,
            "original_label": original_label,
            "label": final_label,
            "label_changed": changed,
            "original_label_confidence": 1.0,
            "filtered_label_confidence": float(confidence),
            "confidence": float(confidence),
            "source_segment_start_frame": int(raw["start_frame"]),
            "source_segment_end_frame": int(raw["end_frame"]),
            "source_segment_duration_frames": source_length,
            "minimum_long_segment_length": max(1, int(minimum_long_length)),
            "confidence_method": "symmetric_triangular_valley_by_source_segment_length" if changed else "unchanged_label_identity",
        })
    return output


def _candidate_frame_evidence(candidates, labels):
    """Aggregate confidence-weighted candidate votes for each observed frame."""
    state_order = list(labels)
    by_frame = {}
    for candidate in candidates:
        candidate_confidence = candidate.get("candidate_confidence")
        candidate_weight = (
            1.0 if candidate_confidence is None
            else max(0.0, min(1.0, float(candidate_confidence)))
        )
        for row in candidate.get("frame_labels", []):
            frame_index = int(row["frame_index"])
            state = str(row.get("label", ""))
            if state not in state_order:
                continue
            frame_confidence = max(0.0, min(1.0, float(
                row.get("semantic_corrected_confidence", row.get("confidence", 0.0))
            )))
            confidence = frame_confidence * candidate_weight
            frame = by_frame.setdefault(frame_index, {
                "weighted": {name: 0.0 for name in state_order},
                "votes": {name: 0 for name in state_order},
                "candidate_count": 0,
                "weight_sum": 0.0,
            })
            frame["weighted"][state] += confidence
            frame["votes"][state] += 1
            frame["candidate_count"] += 1
            frame["weight_sum"] += confidence
    return by_frame


def _decode_contiguous_evidence_block(rows, state_order, minimum_segment_length):
    """Maximum-emission semi-Markov decoding with a hard minimum run length."""
    length = len(rows)
    minimum = max(1, int(minimum_segment_length))
    if not rows:
        return []
    emission = [
        [math.log(max(1e-12, float(row["normalized_evidence"][state]))) for state in state_order]
        for row in rows
    ]
    prefix = [[0.0] * (length + 1) for _ in state_order]
    for state_index in range(len(state_order)):
        for frame_offset in range(length):
            prefix[state_index][frame_offset + 1] = (
                prefix[state_index][frame_offset] + emission[frame_offset][state_index]
            )
    if length < minimum:
        scores = [prefix[index][length] for index in range(len(state_order))]
        selected = max(range(len(state_order)), key=lambda index: (scores[index], -index))
        return [state_order[selected]] * length

    negative_infinity = float("-inf")
    dp = [[negative_infinity] * len(state_order) for _ in range(length)]
    back = [[None] * len(state_order) for _ in range(length)]
    for end in range(minimum - 1, length):
        for state_index in range(len(state_order)):
            for start in range(0, end - minimum + 2):
                segment_score = prefix[state_index][end + 1] - prefix[state_index][start]
                if start == 0:
                    score, pointer = segment_score, None
                else:
                    previous_end = start - 1
                    previous = [
                        (dp[previous_end][other], other)
                        for other in range(len(state_order))
                        if other != state_index and dp[previous_end][other] != negative_infinity
                    ]
                    if not previous:
                        continue
                    previous_score, previous_state = max(previous, key=lambda item: (item[0], -item[1]))
                    score = previous_score + segment_score
                    pointer = (previous_end, previous_state, start)
                if score > dp[end][state_index]:
                    dp[end][state_index] = score
                    back[end][state_index] = pointer

    final_state = max(
        range(len(state_order)), key=lambda index: (dp[length - 1][index], -index)
    )
    decoded = [None] * length
    end, state_index = length - 1, final_state
    while end >= 0:
        pointer = back[end][state_index]
        start = 0 if pointer is None else int(pointer[2])
        decoded[start:end + 1] = [state_order[state_index]] * (end - start + 1)
        if pointer is None:
            break
        end, state_index, _ = pointer
    return decoded


def _final_segments(frame_rows, minimum_segment_length):
    segments = []
    for row in frame_rows:
        frame_index = int(row["frame_index"])
        state = str(row["state"])
        contiguous = segments and frame_index == int(segments[-1]["end_frame"]) + 1
        if not segments or segments[-1]["state"] != state or not contiguous:
            segments.append({
                "state": state, "start_frame": frame_index, "end_frame": frame_index,
                "duration_frames": 1, "_rows": [row],
            })
        else:
            segments[-1]["end_frame"] = frame_index
            segments[-1]["duration_frames"] += 1
            segments[-1]["_rows"].append(row)
    for segment_id, segment in enumerate(segments):
        rows = segment.pop("_rows")
        segment["segment_id"] = segment_id
        for field, output in (
            ("confidence", "mean_confidence"),
            ("consensus", "mean_consensus"),
            ("margin", "mean_margin"),
            ("candidate_disagreement", "mean_candidate_disagreement"),
        ):
            segment[output] = float(sum(float(row[field]) for row in rows) / len(rows))
        segment["confidence"] = segment["mean_confidence"]
        segment["consensus"] = segment["mean_consensus"]
        segment["margin"] = segment["mean_margin"]
        segment["candidate_disagreement"] = segment["mean_candidate_disagreement"]
        segment["minimum_length_constraint_satisfied"] = (
            int(segment["duration_frames"]) >= max(1, int(minimum_segment_length))
        )
    return segments


def confidence_weighted_consensus(candidates, labels, minimum_segment_length):
    """Return one authoritative state sequence from all threshold candidates."""
    state_order = list(labels)
    evidence_by_frame = _candidate_frame_evidence(candidates, state_order)
    evidence_rows = []
    for frame_index in sorted(evidence_by_frame):
        evidence = evidence_by_frame[frame_index]
        total_weight = float(evidence["weight_sum"])
        normalized = (
            {state: float(evidence["weighted"][state] / total_weight) for state in state_order}
            if total_weight > 0.0
            else {state: 1.0 / len(state_order) for state in state_order}
        )
        evidence_rows.append({
            "frame_index": frame_index,
            "normalized_evidence": normalized,
            "weighted_evidence": {
                state: float(evidence["weighted"][state]) for state in state_order
            },
            "vote_counts": {state: int(evidence["votes"][state]) for state in state_order},
            "candidate_count": int(evidence["candidate_count"]),
            "candidate_weight_sum": total_weight,
        })

    decoded_by_frame = {}
    block_start = 0
    while block_start < len(evidence_rows):
        block_end = block_start + 1
        while (
            block_end < len(evidence_rows)
            and int(evidence_rows[block_end]["frame_index"])
            == int(evidence_rows[block_end - 1]["frame_index"]) + 1
        ):
            block_end += 1
        block = evidence_rows[block_start:block_end]
        states = _decode_contiguous_evidence_block(block, state_order, minimum_segment_length)
        decoded_by_frame.update({
            int(row["frame_index"]): state for row, state in zip(block, states)
        })
        block_start = block_end

    final_frames = []
    for row in evidence_rows:
        decoded_state = decoded_by_frame[int(row["frame_index"])]
        normalized = row["normalized_evidence"]
        alternatives = [normalized[state] for state in state_order if state != decoded_state]
        candidate_count = max(1, int(row["candidate_count"]))
        local_top = max(normalized, key=lambda state: (normalized[state], -state_order.index(state)))
        final_frames.append({
            **row,
            "state": decoded_state,
            "label": decoded_state,
            "confidence": float(normalized[decoded_state]),
            "consensus": float(row["vote_counts"][decoded_state] / candidate_count),
            "margin": float(normalized[decoded_state] - max(alternatives, default=0.0)),
            "candidate_disagreement": float(1.0 - max(row["vote_counts"].values()) / candidate_count),
            "local_evidence_winner": local_top,
            "dp_overrode_local_winner": decoded_state != local_top,
        })
    return {
        "status": "completed" if candidates else "unavailable_no_enabled_candidates",
        "method": "confidence_weighted_candidate_evidence_plus_min_length_dp",
        "authoritative": bool(candidates),
        "candidate_scope": "enabled_qualifying_plateau_middle_candidates",
        "num_candidates": len(candidates),
        "enabled_candidate_ids": [int(row.get("candidate_index", -1)) for row in candidates],
        "enabled_candidate_thresholds": [float(row.get("threshold", 0.0)) for row in candidates],
        "enabled_candidate_confidences": [
            None if row.get("candidate_confidence") is None
            else float(row["candidate_confidence"])
            for row in candidates
        ],
        "state_order": state_order,
        "minimum_segment_length": max(1, int(minimum_segment_length)),
        "frames": final_frames,
        "segments": _final_segments(final_frames, minimum_segment_length),
    }


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
            "raw_segment_count_min": min(int(row.get("raw_segment_count", row["segment_count"])) for row in chunk),
            "raw_segment_count_max": max(int(row.get("raw_segment_count", row["segment_count"])) for row in chunk),
            "threshold_start": float(chunk[0]["threshold"]),
            "threshold_end": float(chunk[-1]["threshold"]),
            "midpoint_n": float(midpoint),
        })
        start = index
    return plateaus


def segment_axis(frames, axis, labels, num_thresholds=NUM_THRESHOLDS, noise_tolerance_frames=5, bridge_config=None, plateau_min_n_values=3, consensus_min_segment_length=None):
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
    minimum_long_length = max(1, int(noise_tolerance_frames) + 1)
    for index, threshold in enumerate(thresholds):
        raw_segments = _segments(frames, axis, threshold, labels)
        filtered_segments = _filtered_segments(
            frames, axis, threshold, labels, noise_tolerance_frames, bridge_config,
        )
        candidates.append({
            "candidate_index": index,
            "threshold": float(threshold),
            "segment_count": len(filtered_segments),
            "raw_segment_count": len(raw_segments),
            "frame_labels": frame_label_confidences(
                frames, raw_segments, filtered_segments, minimum_long_length,
            ),
        })
    consensus_minimum = (
        minimum_long_length
        if consensus_min_segment_length is None
        else max(1, int(consensus_min_segment_length))
    )
    all_plateaus = _plateaus(candidates)
    qualifying = []
    for plateau in all_plateaus:
        # Retain plateaus spanning the configured minimum N samples and >1 segment.
        if plateau["num_n_values"] < max(1, int(plateau_min_n_values)) or plateau["segment_count"] <= 1:
            continue
        row = dict(plateau)
        row["candidate_optimal_n"] = float(row["midpoint_n"])
        raw_segments = _segments(frames, axis, row["midpoint_n"], labels)
        row["segments"] = _filtered_segments(
            frames, axis, row["midpoint_n"], labels, noise_tolerance_frames, bridge_config,
        )
        row["frame_labels"] = frame_label_confidences(
            frames, raw_segments, row["segments"], minimum_long_length,
        )
        qualifying.append(row)
    final_segmentation = {
        "status": "pending_enabled_candidate_audit",
        "authoritative": False,
        "candidate_scope": "enabled_qualifying_plateau_middle_candidates",
        "num_candidates": 0,
        "state_order": list(labels),
        "minimum_segment_length": consensus_minimum,
        "frames": [],
        "segments": [],
    }
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
        "final_segmentation": final_segmentation,
        "frame_label_confidence": {
            "raw_confidence": 1.0,
            "unchanged_filtered_confidence": 1.0,
            "minimum_long_segment_length": minimum_long_length,
            "changed_label_profile": "symmetric_triangular_valley",
            "maximum_valley_depth": "min(1, source_segment_length / minimum_long_segment_length)",
        },
        "noise_filter": {
            "method": "robust_multi_segment_bridge_between_equal_state_anchors",
            "tolerance_frames": max(0, int(noise_tolerance_frames)),
            "bridge_total_max_frames": max(0, int((bridge_config or {}).get("bridge_total_max_frames", 15))),
            "anchor_min_frames": max(1, int((bridge_config or {}).get("anchor_min_frames", 8))),
            "bridge_max_segments": max(1, int((bridge_config or {}).get("bridge_max_segments", 5))),
            "bridge_max_anchor_ratio": max(0.0, float((bridge_config or {}).get("bridge_max_anchor_ratio", 0.75))),
            "interruption_measure": "total_frame_span_between_anchor_segments",
            "requirements": ["equal_outer_states", "every_inner_segment_is_short", "bounded_total_span", "bounded_inner_segment_count", "bounded_anchor_ratio"],
            "residual_short_cleanup": "monotonic_mean_signal_assignment_to_neighboring_long_segments",
            "output_invariant": "no_short_segments_when_total_observed_span_exceeds_tolerance",
        },
        "plateau_filter": {
            "minimum_n_values_inclusive": max(1, int(plateau_min_n_values)),
            "exclude_single_segment_plateaus": True,
        },
    }


def segment_video(ego_video, vx_noise_tolerance_frames=5, vz_noise_tolerance_frames=5, vx_bridge_config=None, vz_bridge_config=None, plateau_min_n_values=3, vx_consensus_min_segment_length=None, vz_consensus_min_segment_length=None):
    frames = list(ego_video.get("frames", []))
    vz = segment_axis(
        frames, "vz", ("backward", "static", "forward"),
        noise_tolerance_frames=vz_noise_tolerance_frames,
        bridge_config=vz_bridge_config,
        plateau_min_n_values=plateau_min_n_values,
        consensus_min_segment_length=vz_consensus_min_segment_length,
    )
    vx = segment_axis(
        frames, "vx", ("right", "straight", "left"),
        noise_tolerance_frames=vx_noise_tolerance_frames,
        bridge_config=vx_bridge_config,
        plateau_min_n_values=plateau_min_n_values,
        consensus_min_segment_length=vx_consensus_min_segment_length,
    )
    frame_rows = []
    for offset, frame in enumerate(frames):
        frame_rows.append({
            "frame_index": int(frame.get("frame_index", offset)),
            "ego_vx": _signal(frame, "vx"),
            "ego_vz": _signal(frame, "vz"),
        })
    vx_final_by_frame = {
        int(row["frame_index"]): row for row in vx["final_segmentation"]["frames"]
    }
    vz_final_by_frame = {
        int(row["frame_index"]): row for row in vz["final_segmentation"]["frames"]
    }
    final_frame_labels = []
    for frame_index in sorted(set(vx_final_by_frame) | set(vz_final_by_frame)):
        vx_row = vx_final_by_frame.get(frame_index)
        vz_row = vz_final_by_frame.get(frame_index)
        final_frame_labels.append({
            "frame_index": frame_index,
            "vx": vx_row,
            "vz": vz_row,
        })
    return {
        "version": VERSION,
        "video_id": str(ego_video.get("video_id", "")),
        "status": "completed",
        "method": "confidence_weighted_candidate_consensus_dp",
        "num_frames": len(frames),
        "vz_segmentation": vz,
        "vx_segmentation": vx,
        "final_segmentation": {
            "authoritative": True,
            "vx": vx["final_segmentation"],
            "vz": vz["final_segmentation"],
            "frames": final_frame_labels,
        },
        "frames": frame_rows,
        "provenance": {
            "source": "continuous_ego_motion",
            "threshold_candidates_per_axis": NUM_THRESHOLDS,
            "plateau_min_n_values": max(1, int(plateau_min_n_values)),
            "selection": "plateaus_meeting_minimum_n_values_excluding_single_segment",
            "single_final_n_selected": False,
            "single_final_segmentation_returned": False,
            "final_decoding": "deferred_to_step7b_enabled_candidate_consensus",
            "noise_tolerance_frames": {
                "vx": max(0, int(vx_noise_tolerance_frames)),
                "vz": max(0, int(vz_noise_tolerance_frames)),
            },
            "bridge_config": {
                "vx": dict(vx_bridge_config or {}),
                "vz": dict(vz_bridge_config or {}),
            },
            "deterministic": True,
        },
    }


def materialize_enabled_candidates(result, plateau_audit):
    """Attach enabled plateau-middle candidates without merging their labels."""
    video_id = str(result.get("video_id", ""))
    for axis in ("vx", "vz"):
        axis_result = result.get(f"{axis}_segmentation", {})
        plateaus = {
            int(row["plateau_id"]): row
            for row in axis_result.get("qualifying_plateaus", [])
        }
        axis_points = sorted(
            (
                row for row in plateau_audit.get("points", [])
                if str(row.get("video_id", "")) == video_id
                and str(row.get("axis", "")) == axis
                and int(row.get("plateau_id", -1)) in plateaus
            ),
            key=lambda row: (float(row.get("midpoint_n", 0.0)), int(row.get("plateau_id", -1))),
        )
        enabled_candidates = []
        disabled_candidates = []
        for point in axis_points:
            plateau = plateaus[int(point["plateau_id"])]
            candidate = {
                "candidate_index": int(plateau["plateau_id"]),
                "plateau_id": int(plateau["plateau_id"]),
                "threshold": float(plateau["midpoint_n"]),
                "candidate_confidence": (
                    None if point.get("confidence") is None
                    else float(point["confidence"])
                ),
                "segment_count": int(plateau["segment_count"]),
                "segments": [dict(row) for row in plateau.get("segments", [])],
                "frame_labels": [dict(row) for row in plateau.get("frame_labels", [])],
                "enabled": bool(point.get("enabled", False)),
                "disabled_reasons": list(point.get("disabled_reasons", [])),
                "enablement_source": "all_video_plateau_scatter_audit",
            }
            (enabled_candidates if candidate["enabled"] else disabled_candidates).append(candidate)
        axis_result["enabled_segmentation_candidates"] = enabled_candidates
        axis_result["disabled_segmentation_candidates"] = disabled_candidates
        axis_result["candidate_selection_summary"] = {
            "status": "completed",
            "num_qualifying_candidates": len(plateaus),
            "num_enabled_candidates": len(enabled_candidates),
            "num_disabled_candidates": len(disabled_candidates),
            "enabled_candidate_ids": [row["candidate_index"] for row in enabled_candidates],
            "enabled_candidate_thresholds": [row["threshold"] for row in enabled_candidates],
            "final_merge_performed": False,
            "final_merge_step": "7b",
        }
        axis_result["final_segmentation"] = {
            "status": "pending_step7b_consensus_merge",
            "authoritative": False,
            "candidate_scope": "enabled_qualifying_plateau_middle_candidates",
            "num_candidates": len(enabled_candidates),
            "frames": [],
            "segments": [],
        }
    result["final_segmentation"] = {
        "status": "pending_step7b_consensus_merge",
        "authoritative": False,
        "candidate_scope": "enabled_qualifying_plateau_middle_candidates_only",
        "frames": [],
    }
    result.setdefault("provenance", {})["final_decoding"] = "deferred_to_step7b"
    result["provenance"]["disabled_candidates_excluded"] = True
    return result


def apply_semantic_candidate_confidence_correction(result, opposite_transition_penalty=0.5):
    """Penalize both segments at direct forward/backward candidate transitions."""
    penalty = max(0.0, min(1.0, float(opposite_transition_penalty)))
    total_violations = 0
    corrected_candidates = 0
    axis_summaries = {}
    for axis in ("vx", "vz"):
        candidates = result.get(f"{axis}_segmentation", {}).get(
            "enabled_segmentation_candidates", []
        )
        axis_violations = 0
        axis_corrected = 0
        for candidate in candidates:
            segments = sorted(
                candidate.get("segments", []),
                key=lambda row: (int(row.get("start_frame", 0)), int(row.get("end_frame", 0))),
            )
            violation_counts = [0] * len(segments)
            violations = []
            if axis == "vz":
                forbidden = {("forward", "backward"), ("backward", "forward")}
                for left_index, (left, right) in enumerate(zip(segments, segments[1:])):
                    transition = (str(left.get("state", "")), str(right.get("state", "")))
                    if transition not in forbidden:
                        continue
                    right_index = left_index + 1
                    violation_counts[left_index] += 1
                    violation_counts[right_index] += 1
                    violations.append({
                        "rule_id": "no_direct_forward_backward_transition",
                        "transition": f"{transition[0]}->{transition[1]}",
                        "left_segment_id": int(left.get("segment_id", left_index)),
                        "right_segment_id": int(right.get("segment_id", right_index)),
                        "left_frame_range": [int(left["start_frame"]), int(left["end_frame"])],
                        "right_frame_range": [int(right["start_frame"]), int(right["end_frame"])],
                        "penalty_per_incident": penalty,
                        "affected_segments": "both",
                    })
            frame_labels = candidate.get("frame_labels", [])
            original_values = []
            corrected_values = []
            for frame in frame_labels:
                frame_index = int(frame["frame_index"])
                incidents = sum(
                    count
                    for segment, count in zip(segments, violation_counts)
                    if count > 0 and int(segment["start_frame"]) <= frame_index <= int(segment["end_frame"])
                )
                multiplier = float((1.0 - penalty) ** incidents)
                original = max(0.0, min(1.0, float(frame.get("confidence", 0.0))))
                corrected = float(original * multiplier)
                frame["semantic_confidence_before"] = original
                frame["semantic_confidence_multiplier"] = multiplier
                frame["semantic_corrected_confidence"] = corrected
                frame["semantic_violation_count"] = incidents
                frame["semantic_correction_applied"] = incidents > 0
                original_values.append(original)
                corrected_values.append(corrected)
            for index, segment in enumerate(segments):
                incidents = violation_counts[index]
                multiplier = float((1.0 - penalty) ** incidents)
                segment_frames = [
                    row for row in frame_labels
                    if int(segment["start_frame"]) <= int(row["frame_index"]) <= int(segment["end_frame"])
                ]
                segment_before = (
                    sum(float(row["semantic_confidence_before"]) for row in segment_frames) / len(segment_frames)
                    if segment_frames else 0.0
                )
                segment_after = (
                    sum(float(row["semantic_corrected_confidence"]) for row in segment_frames) / len(segment_frames)
                    if segment_frames else 0.0
                )
                segment["semantic_violation_count"] = incidents
                segment["semantic_confidence_before"] = float(segment_before)
                segment["semantic_confidence_after"] = float(segment_after)
                segment["semantic_confidence_multiplier"] = multiplier
                segment["semantic_correction_applied"] = incidents > 0
                segment["semantic_rule_status"] = "violated" if incidents else "satisfied"
            original_mean = (
                float(sum(original_values) / len(original_values)) if original_values else 0.0
            )
            corrected_mean = (
                float(sum(corrected_values) / len(corrected_values)) if corrected_values else 0.0
            )
            candidate["semantic_correction"] = {
                "status": "penalized" if violations else "unchanged",
                "rules_evaluated": ["no_direct_forward_backward_transition"] if axis == "vz" else [],
                "violations": violations,
                "num_violations": len(violations),
                "penalty_per_incident": penalty,
                "original_mean_frame_confidence": original_mean,
                "corrected_mean_frame_confidence": corrected_mean,
                "candidate_confidence_multiplier": (
                    corrected_mean / original_mean if original_mean > 0.0 else 1.0
                ),
                "correction_step": "7b_pre_merge_semantic_correction",
            }
            axis_violations += len(violations)
            if violations:
                axis_corrected += 1
        total_violations += axis_violations
        corrected_candidates += axis_corrected
        axis_summaries[axis] = {
            "num_candidates": len(candidates),
            "num_penalized_candidates": axis_corrected,
            "num_violations": axis_violations,
        }
    summary = {
        "status": "completed",
        "step": "7b_pre_merge_semantic_correction",
        "rule_ids": ["no_direct_forward_backward_transition"],
        "opposite_transition_penalty": penalty,
        "num_penalized_candidates": corrected_candidates,
        "num_violations": total_violations,
        "axis_summaries": axis_summaries,
        "final_merge_uses": "semantic_corrected_confidence",
    }
    result["step7b_semantic_confidence_correction"] = summary
    return summary


def finalize_enabled_consensus(result, plateau_audit=None, vx_minimum_segment_length=6, vz_minimum_segment_length=6):
    """Step 7B merge of the enabled candidates materialized by Step 7A."""
    if plateau_audit is not None:
        materialize_enabled_candidates(result, plateau_audit)
    minimum_by_axis = {
        "vx": max(1, int(vx_minimum_segment_length)),
        "vz": max(1, int(vz_minimum_segment_length)),
    }
    for axis in ("vx", "vz"):
        axis_result = result.get(f"{axis}_segmentation", {})
        labels = axis_result.get("labels", {})
        state_order = (
            str(labels.get("negative", "negative")),
            str(labels.get("center", "center")),
            str(labels.get("positive", "positive")),
        )
        enabled_candidates = list(axis_result.get("enabled_segmentation_candidates", []))
        final = confidence_weighted_consensus(
            enabled_candidates, state_order, minimum_by_axis[axis],
        )
        summary = axis_result.get("candidate_selection_summary", {})
        final["enablement_source"] = "step7a_enabled_segmentation_candidates"
        final["disabled_candidates_excluded"] = True
        final["num_qualifying_candidates"] = int(summary.get("num_qualifying_candidates", len(enabled_candidates)))
        final["num_disabled_candidates"] = int(summary.get("num_disabled_candidates", 0))
        final["merge_step"] = "7b"
        axis_result["final_segmentation"] = final

    vx_final = result.get("vx_segmentation", {}).get("final_segmentation", {})
    vz_final = result.get("vz_segmentation", {}).get("final_segmentation", {})
    vx_by_frame = {int(row["frame_index"]): row for row in vx_final.get("frames", [])}
    vz_by_frame = {int(row["frame_index"]): row for row in vz_final.get("frames", [])}
    final_frames = [
        {"frame_index": frame_index, "vx": vx_by_frame.get(frame_index), "vz": vz_by_frame.get(frame_index)}
        for frame_index in sorted(set(vx_by_frame) | set(vz_by_frame))
    ]
    result["final_segmentation"] = {
        "authoritative": bool(vx_final.get("authoritative")) and bool(vz_final.get("authoritative")),
        "status": (
            "completed" if vx_final.get("status") == "completed" and vz_final.get("status") == "completed"
            else "partial_or_unavailable_no_enabled_candidates"
        ),
        "candidate_scope": "step7a_enabled_segmentation_candidates_only",
        "merge_step": "7b",
        "vx": vx_final,
        "vz": vz_final,
        "frames": final_frames,
    }
    result.setdefault("provenance", {})["final_decoding"] = (
        "step7b_enabled_candidates_confidence_weighting_and_min_length_dp"
    )
    result["provenance"]["disabled_candidates_excluded"] = True
    return result


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
        raw_counts = [row.get("raw_segment_count", row["segment_count"]) for row in candidates]
        axis.plot(
            thresholds, raw_counts, color="#6C757D", linewidth=1.9,
            linestyle="--", marker="x", markersize=3.5,
            label="raw segments before short-merge filter", zorder=2,
        )
        axis.plot(
            thresholds, counts, color=color, linewidth=2.5,
            marker=".", markersize=4,
            label="filtered segments after short-merge filter", zorder=3,
        )
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
            raw_min = int(plateau.get("raw_segment_count_min", count))
            raw_max = int(plateau.get("raw_segment_count_max", count))
            raw_text = str(raw_min) if raw_min == raw_max else f"{raw_min}-{raw_max}"
            axis.annotate(
                f"middle N={midpoint:.5g}\nfiltered={count} | raw={raw_text}",
                xy=(midpoint, count), xytext=(7, 13), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color="#4A1268",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#7A1FA2", "alpha": 0.9},
            )
        axis.set_title(f"{title}\nqualifying plateaus={len(data['qualifying_plateaus'])}", fontsize=13, fontweight="bold")
        axis.set_xlabel("Threshold N", fontsize=11)
        axis.set_ylabel("Number of temporal segments", fontsize=11)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8.5, loc="best")
    figure.suptitle(
        f"Step 7A raw vs filtered segment counts | video={result.get('video_id', '')}",
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



def select_optimal_n_by_final_similarity(result):
    """Select the enabled candidate most similar to the final merged sequence."""
    video_id = str(result.get("video_id", ""))
    selections = {}
    for axis in ("vx", "vz"):
        axis_result = result.get(f"{axis}_segmentation", {})
        final_frames = {
            int(row["frame_index"]): row
            for row in axis_result.get("final_segmentation", {}).get("frames", [])
        }
        comparisons = []
        for candidate in axis_result.get("enabled_segmentation_candidates", []):
            candidate_frames = {
                int(row["frame_index"]): row
                for row in candidate.get("frame_labels", [])
            }
            common = sorted(set(final_frames) & set(candidate_frames))
            final_count = len(final_frames)
            matched = [
                frame_index for frame_index in common
                if str(final_frames[frame_index].get("state", ""))
                == str(candidate_frames[frame_index].get("label", ""))
            ]
            weights = {
                frame_index: max(0.0, float(final_frames[frame_index].get("confidence", 0.0)))
                for frame_index in common
            }
            total_weight = sum(weights.values())
            weighted_similarity = (
                sum(weights[frame_index] for frame_index in matched) / total_weight
                if total_weight > 0.0 else 0.0
            )
            raw_similarity = len(matched) / len(common) if common else 0.0
            coverage = len(common) / final_count if final_count else 0.0
            comparison = {
                "candidate_id": int(candidate.get("candidate_index", -1)),
                "plateau_id": int(candidate.get("plateau_id", candidate.get("candidate_index", -1))),
                "threshold_n": float(candidate.get("threshold", 0.0)),
                "segment_count": int(candidate.get("segment_count", 0)),
                "candidate_confidence": (
                    None if candidate.get("candidate_confidence") is None
                    else float(candidate["candidate_confidence"])
                ),
                "weighted_state_similarity": float(weighted_similarity),
                "raw_state_similarity": float(raw_similarity),
                "frame_coverage": float(coverage),
                "num_common_frames": len(common),
                "num_matching_frames": len(matched),
                "num_disagreeing_frames": len(common) - len(matched),
                "semantic_correction_status": str(candidate.get("semantic_correction", {}).get("status", "not_applied")),
            }
            comparisons.append(comparison)
        comparisons.sort(key=lambda row: (row["threshold_n"], row["candidate_id"]))
        if comparisons:
            selected = max(comparisons, key=lambda row: (
                row["weighted_state_similarity"],
                row["raw_state_similarity"],
                row["frame_coverage"],
                -1.0 if row["candidate_confidence"] is None else row["candidate_confidence"],
                -row["threshold_n"],
                -row["candidate_id"],
            ))
            status = "selected"
        else:
            selected = None
            status = "unavailable_no_enabled_candidates"
        selection = {
            "status": status,
            "video_id": video_id,
            "axis": axis,
            "method": "final_confidence_weighted_frame_state_agreement",
            "tie_break_order": [
                "weighted_state_similarity", "raw_state_similarity", "frame_coverage",
                "candidate_confidence", "lower_threshold_n", "lower_candidate_id",
            ],
            "num_compared_candidates": len(comparisons),
            "optimal_n": None if selected is None else float(selected["threshold_n"]),
            "selected_candidate_id": None if selected is None else int(selected["candidate_id"]),
            "selected_segment_count": None if selected is None else int(selected["segment_count"]),
            "selected_similarity": None if selected is None else float(selected["weighted_state_similarity"]),
            "selected_candidate": selected,
            "candidate_similarities": comparisons,
        }
        axis_result["optimal_n_selection"] = selection
        selections[axis] = selection
    result["optimal_n_selection"] = selections
    return selections


def render_train_optimal_n_scatter(
    train_results, eval_results, output_path,
    vx_seg_max_count=8, vz_seg_max_count=5,
    max_plateau_middle_th_vx=250.0, max_plateau_middle_th_vz=70.0,
):
    """Render one optimal-N point/video, fitting heat maps from train only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    limits = {
        "vx": {"x": max(0.0, float(max_plateau_middle_th_vx)), "y": max(1, int(vx_seg_max_count))},
        "vz": {"x": max(0.0, float(max_plateau_middle_th_vz)), "y": max(1, int(vz_seg_max_count))},
    }

    def collect(results, axis, split):
        rows = []
        for result in results:
            selection = result.get(f"{axis}_segmentation", {}).get("optimal_n_selection", {})
            if selection.get("status") != "selected":
                continue
            rows.append({
                "video_id": str(result.get("video_id", "")),
                "split": split,
                "axis": axis,
                "midpoint_n": float(selection["optimal_n"]),
                "segment_count": int(selection["selected_segment_count"]),
                "similarity": float(selection["selected_similarity"]),
                "candidate_id": int(selection["selected_candidate_id"]),
            })
        return rows

    figure, axes = plt.subplots(1, 2, figsize=(17, 7.5), constrained_layout=True)
    all_points = []
    confidence_regions = {}
    evaluation_metrics = {}
    plot_limits_by_axis = {}
    for plot_axis, axis in zip(axes, ("vx", "vz")):
        train_rows = collect(train_results, axis, "train")
        eval_rows = collect(eval_results, axis, "eval")
        all_points.extend(train_rows + eval_rows)
        x_source_rows = eval_rows if eval_rows else train_rows
        x_values = [float(row["midpoint_n"]) for row in x_source_rows]
        if x_values:
            x_min_data, x_max_data = min(x_values), max(x_values)
            x_span = x_max_data - x_min_data
            x_padding = max(
                0.10 * x_span,
                0.05 * max(abs(x_min_data), abs(x_max_data), 1.0),
                1e-3,
            )
            x_min = max(0.0, x_min_data - x_padding)
            x_max = x_max_data + x_padding
            if x_max <= x_min:
                x_max = x_min + max(1.0, x_padding)
        else:
            x_min, x_max = 0.0, max(1.0, 1.2 * limits[axis]["x"])
        bounds = (x_min, x_max, 0.0, 1.2 * limits[axis]["y"])
        plot_limits_by_axis[axis] = {
            "x_min": float(x_min), "x_max": float(x_max),
            "y_min": float(bounds[2]), "y_max": float(bounds[3]),
            "x_range_source": "eval_optimal_n" if eval_rows else "train_optimal_n_fallback",
        }
        model = _confidence_surface(train_rows, bounds=bounds)
        confidence_regions[axis] = None if model is None else model["audit"]
        if model is not None:
            heatmap = plot_axis.contourf(
                model["x"], model["y"], model["confidence"],
                levels=np.linspace(0.0, 1.0, 25), cmap="viridis",
                alpha=1.0, antialiased=True, zorder=0,
            )
            figure.colorbar(
                heatmap, ax=plot_axis, fraction=0.046, pad=0.03,
                label="Train optimal-N confidence",
            )
            for row in eval_rows:
                row["train_density_confidence"] = _confidence_at(model, row)
        else:
            for row in eval_rows:
                row["train_density_confidence"] = None
        if eval_rows:
            plot_axis.scatter(
                [row["midpoint_n"] for row in eval_rows],
                [row["segment_count"] for row in eval_rows],
                s=115, color="#D24FA4", marker="D", edgecolors="white",
                linewidths=1.1, alpha=0.96, label="eval optimal N", zorder=7,
            )
        eval_confidences = [
            row["train_density_confidence"] for row in eval_rows
            if row["train_density_confidence"] is not None
        ]
        evaluation_metrics[axis] = {
            "metric": "mean_eval_optimal_n_train_density_confidence",
            "value": None if not eval_confidences else float(sum(eval_confidences) / len(eval_confidences)),
            "num_train_optimal_points": len(train_rows),
            "num_eval_optimal_points": len(eval_rows),
            "train_points_used_for_heatmap_only": True,
            "visible_scatter_split": "eval_only",
        }
        plot_axis.axhline(limits[axis]["y"], color="#666666", linestyle="--", linewidth=1.2)
        plot_axis.axvline(limits[axis]["x"], color="#999999", linestyle=":", linewidth=1.4)
        plot_axis.set_xlim(bounds[0], bounds[1])
        plot_axis.set_ylim(bounds[2], bounds[3])
        plot_axis.set_title(
            f"{axis.upper()} optimal N | heatmap train={len(train_rows)} | shown eval={len(eval_rows)}",
            fontsize=14, fontweight="bold",
        )
        plot_axis.set_xlabel("Optimal threshold N", fontsize=12)
        plot_axis.set_ylabel("Segments in most-similar candidate", fontsize=12)
        plot_axis.grid(True, color="white", alpha=0.12, linewidth=0.7)
        if eval_rows:
            plot_axis.legend(fontsize=9, loc="best")
    figure.suptitle(
        "Step 7B optimal N per video | train-fitted heat map + eval overlay",
        fontsize=17, fontweight="bold",
    )
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return {
        "status": "rendered",
        "path": str(output_path),
        "method": "one_most_similar_candidate_optimal_n_per_video",
        "heatmap_fit_split": "train_only",
        "eval_usage": "visible_scatter_and_held_out_density_evaluation_only",
        "visible_scatter_split": "eval_only",
        "train_points_visible": False,
        "x_range_policy": "eval_optimal_n_range_with_padding_else_train_fallback",
        "heatmap_style": {
            "colormap": "viridis",
            "opacity": 1.0,
            "contour_levels": 25,
            "grid_color": "white",
            "grid_alpha": 0.12,
        },
        "num_train_videos": len(train_results),
        "num_eval_videos": len(eval_results),
        "num_train_optimal_points": sum(row["split"] == "train" for row in all_points),
        "num_eval_optimal_points": sum(row["split"] == "eval" for row in all_points),
        "confidence_regions": confidence_regions,
        "evaluation_metrics": evaluation_metrics,
        "plot_limits_by_axis": plot_limits_by_axis,
        "points": all_points,
    }
