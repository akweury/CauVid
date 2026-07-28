"""Deterministic video-local normalization for Step 7B background evidence."""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict


VERSION = 2


def _finite(values):
    rows = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            rows.append(number)
    return rows


def _quantile(sorted_values, probability):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, float(probability))) * (len(sorted_values) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return float(sorted_values[left])
    alpha = position - left
    return float(sorted_values[left] * (1.0 - alpha) + sorted_values[right] * alpha)


def robust_statistics(values):
    rows = sorted(_finite(values))
    if not rows:
        return {
            "sample_count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "mad": 0.0,
            "robust_scale": 1.0,
            "robust_scale_method": "degenerate_unit_scale",
            "quantiles": {"q05": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q95": 0.0},
        }
    count = len(rows)
    mean = sum(rows) / count
    variance = sum((value - mean) ** 2 for value in rows) / count
    median = _quantile(rows, 0.5)
    deviations = sorted(abs(value - median) for value in rows)
    mad = _quantile(deviations, 0.5)
    q25 = _quantile(rows, 0.25)
    q75 = _quantile(rows, 0.75)
    scale = 1.4826 * mad
    method = "scaled_mad"
    if scale <= 0.0:
        scale = (q75 - q25) / 1.349
        method = "iqr_scale"
    if scale <= 0.0:
        scale = max(abs(median), math.sqrt(variance))
        method = "median_or_std_scale"
    if scale <= 0.0:
        scale = 1.0
        method = "degenerate_unit_scale"
    return {
        "sample_count": count,
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "median": float(median),
        "mad": float(mad),
        "robust_scale": float(scale),
        "robust_scale_method": method,
        "quantiles": {
            "q05": _quantile(rows, 0.05),
            "q25": q25,
            "q50": median,
            "q75": q75,
            "q95": _quantile(rows, 0.95),
        },
    }


def _dominant_state(vectors):
    counts = Counter(str(row.get("radial_state", "neutral")) for row in vectors)
    order = {"expansion": 0, "contraction": 1, "neutral": 2}
    if not counts:
        return "unavailable", 0.0, counts
    state, count = min(
        counts.items(), key=lambda item: (-item[1], order.get(item[0], 9), item[0])
    )
    return state, float(count / max(1, len(vectors))), counts


def _estimator_agreement(segment, global_state):
    if global_state == "unavailable":
        return 0.0, {"global_state": global_state, "group_votes": []}
    group_votes = []
    for pair in segment.get("frame_pair_evidence", []):
        vectors = list(pair.get("patch_vectors", []))
        state, support, _ = _dominant_state(vectors)
        if state != "unavailable":
            group_votes.append({
                "group_type": "frame_pair",
                "group_id": f"{pair.get('start_frame')}:{pair.get('end_frame')}",
                "dominant_state": state,
                "support": support,
            })
    by_region = defaultdict(list)
    for vector in segment.get("patch_vectors", []):
        by_region[str(vector.get("region_id", "unknown"))].append(vector)
    for region_id, vectors in sorted(by_region.items()):
        state, support, _ = _dominant_state(vectors)
        group_votes.append({
            "group_type": "region",
            "group_id": region_id,
            "dominant_state": state,
            "support": support,
        })
    if not group_votes:
        return 0.0, {"global_state": global_state, "group_votes": []}
    agreement = sum(
        float(row["support"]) if row["dominant_state"] == global_state else 0.0
        for row in group_votes
    ) / len(group_votes)
    return float(max(0.0, min(1.0, agreement))), {
        "global_state": global_state,
        "group_votes": group_votes,
    }


def calibrate_video(raw_video, preserve_raw_evidence=True):
    segments = list(raw_video.get("segments", []))
    vectors = [
        vector for segment in segments for vector in segment.get("patch_vectors", [])
    ]
    calibration = {
        "motion_magnitude": robust_statistics(row.get("magnitude") for row in vectors),
        "absolute_radial_projection": robust_statistics(
            abs(float(row.get("radial_projection", 0.0))) for row in vectors
        ),
        "forward_backward_error": robust_statistics(
            row.get("forward_backward_error") for row in vectors
        ),
        "local_vector_residual": robust_statistics(
            row.get("local_vector_residual") for row in vectors
        ),
    }
    motion_stats = calibration["motion_magnitude"]
    motion_scale = float(motion_stats["robust_scale"])
    motion_center = float(motion_stats["median"])
    normalized_segments = []
    for segment in segments:
        segment_vectors = list(segment.get("patch_vectors", []))
        magnitudes = _finite(row.get("magnitude") for row in segment_vectors)
        magnitude_stats = robust_statistics(magnitudes)
        median_magnitude = float(magnitude_stats["median"])
        robust_z = float((median_magnitude - motion_center) / motion_scale)
        noise_units = float(median_magnitude / motion_scale)
        dominant_state, direction_support, direction_counts = _dominant_state(segment_vectors)
        direction_balance = float(
            (direction_counts["expansion"] - direction_counts["contraction"])
            / max(1, len(segment_vectors))
        )
        horizontal_values = _finite(row.get("dx", 0.0) for row in segment_vectors)
        horizontal_stats = robust_statistics(horizontal_values)
        horizontal_deadband = float(
            calibration["local_vector_residual"]["robust_scale"]
        )
        flow_left = sum(value < -horizontal_deadband for value in horizontal_values)
        flow_right = sum(value > horizontal_deadband for value in horizontal_values)
        flow_neutral = max(0, len(horizontal_values) - flow_left - flow_right)
        horizontal_count = max(1, len(horizontal_values))
        background_flow_left_support = float(flow_left / horizontal_count)
        background_flow_right_support = float(flow_right / horizontal_count)
        horizontal_flow_balance = float((flow_right - flow_left) / horizontal_count)
        normalized_horizontal_motion = float(
            horizontal_stats["median"] / motion_scale
        )
        turning_structure_support = float(
            max(background_flow_left_support, background_flow_right_support)
            * max(0.0, min(1.0, segment.get("spatial_coverage", 0.0)))
        )
        agreement, agreement_audit = _estimator_agreement(segment, dominant_state)
        region_support = float(max(0.0, min(1.0, segment.get("spatial_coverage", 0.0))))
        persistence = float(max(0.0, min(1.0, segment.get("temporal_persistence", 0.0))))
        reliability = float(max(0.0, min(1.0, segment.get("tracking_reliability", 0.0))))
        sampling_uncertainty = float(1.0 / math.sqrt(len(segment_vectors) + 1.0))
        uncertainty_components = {
            "tracking": 1.0 - reliability,
            "spatial_coverage": 1.0 - region_support,
            "temporal_persistence": 1.0 - persistence,
            "estimator_disagreement": 1.0 - agreement,
            "finite_sample": sampling_uncertainty,
        }
        uncertainty = float(sum(uncertainty_components.values()) / len(uncertainty_components))
        normalized_segments.append({
            "segment_id": int(segment.get("segment_id", len(normalized_segments))),
            "provisional_action": str(segment.get("provisional_action", "unknown")),
            "start_frame": int(segment.get("start_frame", 0)),
            "end_frame": int(segment.get("end_frame", 0)),
            "status": str(segment.get("status", "unknown")),
            "normalized_motion_magnitude": noise_units,
            "motion_magnitude_robust_z": robust_z,
            "raw_median_motion_magnitude": median_magnitude,
            "direction_support_ratio": direction_support,
            "dominant_radial_direction": dominant_state,
            "signed_direction_balance": direction_balance,
            "background_flow_left_support_ratio": background_flow_left_support,
            "background_flow_right_support_ratio": background_flow_right_support,
            "background_flow_neutral_support_ratio": float(flow_neutral / horizontal_count),
            "horizontal_flow_balance": horizontal_flow_balance,
            "normalized_horizontal_motion": normalized_horizontal_motion,
            "turning_structure_support": turning_structure_support,
            "region_support_ratio": region_support,
            "temporal_persistence": persistence,
            "estimator_agreement": agreement,
            "uncertainty": uncertainty,
            "uncertainty_components": uncertainty_components,
            "estimator_agreement_audit": agreement_audit,
            "source_evidence_counts": {
                "patch_vectors": len(segment_vectors),
                "covered_regions": len(segment.get("covered_regions", [])),
                "frame_pairs": len(segment.get("frame_pair_evidence", [])),
            },
            "provenance": {
                "source_step": "7c_video_local_evidence_calibration",
                "source_evidence_step": "7b_background_motion_evidence",
                "normalization_scope": "single_video",
                "dataset_specific_absolute_thresholds_used": False,
                "deterministic": True,
            },
        })
    return {
        "version": VERSION,
        "video_id": str(raw_video.get("video_id", "")),
        "status": "completed",
        "input_label_status": "provisional",
        "output_role": "video_normalized_evidence_not_final_labels",
        "calibration_scope": "video_local",
        "dataset_specific_absolute_thresholds_used": False,
        "calibration_statistics": calibration,
        "normalized_segment_evidence": normalized_segments,
        "num_segments": len(normalized_segments),
        "num_patch_vectors": len(vectors),
        "audit": {
            "method": "median_mad_with_iqr_and_degenerate_fallback",
            "motion_normalization": "segment_median_in_video_noise_scale_units",
            "uncertainty_aggregation": "unweighted_mean_of_bounded_measurement_components",
            "raw_evidence_preserved": (
                copy.deepcopy(raw_video)
                if preserve_raw_evidence
                else {
                    "video_id": str(raw_video.get("video_id", "")),
                    "segments": [],
                    "omitted_during_candidate_search": True,
                }
            ),
        },
    }
