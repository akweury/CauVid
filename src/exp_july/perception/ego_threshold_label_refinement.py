"""Deterministic closed-loop threshold and ego-label refinement for Step 7E."""

from __future__ import annotations

import copy
import math
from collections import Counter

from src.exp_july.perception.global_ego_symbolic_rules import evaluate_video
from src.exp_july.perception.video_local_evidence_calibration import calibrate_video


VERSION = 1
MAX_ITERATIONS = 4


def _candidate_segment_evidence(candidate_segments, raw_video):
    pair_rows = []
    cfg = dict(raw_video.get("configuration", {}))
    total_regions = max(1, int(cfg.get("region_rows", 3)) * int(cfg.get("region_cols", 3)))
    for original in raw_video.get("segments", []):
        pair_rows.extend(copy.deepcopy(original.get("frame_pair_evidence", [])))
    pair_rows.sort(key=lambda row: (int(row.get("start_frame", 0)), int(row.get("end_frame", 0))))
    results = []
    for candidate in candidate_segments:
        start = int(candidate.get("start_frame", 0))
        end = int(candidate.get("end_frame", start))
        pairs = [
            row for row in pair_rows
            if start <= int(row.get("start_frame", -1))
            and int(row.get("end_frame", -1)) <= end
        ]
        vectors = [vector for pair in pairs for vector in pair.get("patch_vectors", [])]
        raw_count = sum(int(pair.get("raw_patch_count", len(pair.get("patch_vectors", [])))) for pair in pairs)
        reliable_pairs = [pair for pair in pairs if pair.get("status") == "completed"]
        covered = sorted({str(vector.get("region_id", "unknown")) for vector in vectors})
        radial = Counter(str(vector.get("radial_state", "neutral")) for vector in vectors)
        denominator = max(1, len(vectors))
        possible_pairs = max(1, end - start)
        reliability = float(len(vectors) / max(1, raw_count))
        coverage = float(len(covered) / total_regions)
        persistence = float(len(reliable_pairs) / possible_pairs)
        results.append({
            "segment_id": int(candidate.get("segment_id", len(results))),
            "provisional_action": str(candidate.get("action", "unknown")),
            "start_frame": start,
            "end_frame": end,
            "duration_frames": int(candidate.get("duration_frames", end - start + 1)),
            "status": "completed" if vectors else "insufficient_evidence",
            "patch_vectors": vectors,
            "frame_pair_evidence": pairs,
            "radial_expansion_support": float(radial["expansion"] / denominator),
            "radial_contraction_support": float(radial["contraction"] / denominator),
            "radial_neutral_support": float(radial["neutral"] / denominator),
            "spatial_coverage": coverage,
            "covered_regions": covered,
            "temporal_persistence": persistence,
            "tracking_reliability": reliability,
            "estimator_confidence": float(max(0.0, min(1.0, persistence * math.sqrt(max(0.0, reliability * coverage))))),
            "num_raw_patches": raw_count,
            "num_accepted_vectors": len(vectors),
            "provenance": {
                "source_step": "7e_threshold_label_refinement",
                "resegmented_from": "7b_frame_pair_evidence",
                "numerical_motion_recomputed": True,
            },
        })
    return {
        "version": int(raw_video.get("version", 1)),
        "video_id": str(raw_video.get("video_id", "")),
        "input_label_status": "candidate",
        "segments": results,
        "configuration": cfg,
    }


def _action_hypotheses(action):
    action = str(action)
    if action in {"forward", "backward", "static"}:
        return (action,)
    if action in {"left", "turning_left"}:
        return ("left", "turning")
    if action in {"right", "turning_right"}:
        return ("right", "turning")
    return ()


def _candidate_rule_metrics(rule_video):
    hard_violations = 0
    soft_severity = 0.0
    unexplained = 0
    uncertain_segments = []
    for segment in rule_video.get("segment_evaluations", []):
        targets = _action_hypotheses(segment.get("provisional_action"))
        target_rules = [
            row for collection in (segment.get("fired_rules", []), segment.get("violated_rules", []))
            for row in collection if row.get("hypothesis") in targets
        ]
        for row in target_rules:
            severity = 1.0 - float(row.get("rule_match_score", 0.0))
            if float(row.get("weight", 0.0)) >= 1.0 and row.get("violated_atoms"):
                hard_violations += 1
            elif row.get("violated_atoms"):
                soft_severity += severity * float(row.get("weight", 1.0))
        scores = dict(segment.get("hypothesis_scores", {}))
        best_score = max(scores.values(), default=0.0)
        target_score = max((float(scores.get(target, 0.0)) for target in targets), default=0.0)
        reasons = []
        if not targets:
            reasons.append("unknown_candidate_action")
        if best_score < 0.50:
            unexplained += 1
            reasons.append("no_supported_global_hypothesis")
        if targets and target_score < 0.50:
            reasons.append("candidate_action_not_rule_supported")
        if segment.get("conflicts"):
            reasons.append("symbolic_conflict")
        if reasons:
            uncertain_segments.append({
                "segment_id": segment.get("segment_id"),
                "reasons": reasons,
                "best_hypothesis": segment.get("hypothesis_ranking", ["unknown"])[0],
                "best_score": best_score,
                "target_score": target_score,
            })
    return hard_violations, float(soft_severity), unexplained, uncertain_segments


def evaluate_candidate(video_id, candidate_score, raw_video):
    candidate_segments = list(candidate_score.get("segments", []))
    resegmented = _candidate_segment_evidence(candidate_segments, raw_video)
    calibrated = calibrate_video(resegmented)
    rule_video = evaluate_video(calibrated)
    hard, soft, unexplained, uncertain_segments = _candidate_rule_metrics(rule_video)
    components = dict(candidate_score.get("score_components", {}))
    rank_key = (
        int(hard),
        round(float(soft), 12),
        int(unexplained),
        int(candidate_score.get("num_rapid_left_right_reversals", 0)),
        int(candidate_score.get("num_short_segments", 0)),
        round(float(components.get("action_complexity", 0.0)), 12),
        round(float(components.get("signal_fit_error", 0.0)), 12),
        str(candidate_score.get("candidate_id", "")),
    )
    return {
        "candidate_id": str(candidate_score.get("candidate_id", "")),
        "thresholds": copy.deepcopy(candidate_score.get("thresholds", {})),
        "segments": copy.deepcopy(candidate_segments),
        "actions": list(candidate_score.get("actions", [])),
        "hard_rule_violations": int(hard),
        "soft_rule_violation_severity": float(soft),
        "unexplained_segments": int(unexplained),
        "rapid_state_reversals": int(candidate_score.get("num_rapid_left_right_reversals", 0)),
        "short_segments": int(candidate_score.get("num_short_segments", 0)),
        "action_complexity": float(components.get("action_complexity", 0.0)),
        "signal_fit_error": float(components.get("signal_fit_error", 0.0)),
        "rank_key": list(rank_key),
        "uncertain_segments": uncertain_segments,
        "normalized_evidence": calibrated,
        "global_rule_evaluation": rule_video,
        "source_candidate_score": {
            key: copy.deepcopy(value)
            for key, value in candidate_score.items()
            if key not in {"actions", "segments"}
        },
    }


def refine_video(video_id, candidate_scores, raw_video, provisional_video, max_iterations=MAX_ITERATIONS):
    evaluated = [evaluate_candidate(video_id, row, raw_video) for row in candidate_scores]
    if not evaluated:
        raise RuntimeError(f"Step 7E generated no candidates for video {video_id}")
    evaluated.sort(key=lambda row: tuple(row["rank_key"]))
    selected = evaluated[0]
    history = []
    previous_signature = None
    stabilized = False
    for iteration in range(1, max(2, int(max_iterations)) + 1):
        signature = (
            selected["candidate_id"],
            tuple(sorted(selected["thresholds"].items())),
            tuple((row.get("action"), row.get("start_frame"), row.get("end_frame")) for row in selected["segments"]),
        )
        history.append({
            "iteration": iteration,
            "selected_candidate_id": selected["candidate_id"],
            "selected_thresholds": copy.deepcopy(selected["thresholds"]),
            "rank_key": copy.deepcopy(selected["rank_key"]),
            "stable_with_previous_iteration": signature == previous_signature,
        })
        if signature == previous_signature:
            stabilized = True
            break
        previous_signature = signature
    provisional_segments = list(provisional_video.get("final_action_segments", []))
    corrections = []
    for segment in selected["segments"]:
        overlaps = [
            row for row in provisional_segments
            if int(row.get("start_frame", 0)) <= int(segment.get("end_frame", 0))
            and int(row.get("end_frame", 0)) >= int(segment.get("start_frame", 0))
        ]
        prior_actions = sorted({str(row.get("action", "unknown")) for row in overlaps})
        if prior_actions != [str(segment.get("action", "unknown"))]:
            rule_segment = next(
                (row for row in selected["global_rule_evaluation"]["segment_evaluations"] if int(row.get("segment_id", -1)) == int(segment.get("segment_id", -2))),
                {},
            )
            corrections.append({
                "segment_id": segment.get("segment_id"),
                "provisional_actions": prior_actions,
                "corrected_action": segment.get("action"),
                "reason": "lower_global_rule_violation_rank",
                "best_rule_hypothesis": (rule_segment.get("hypothesis_ranking") or ["unknown"])[0],
                "hypothesis_scores": copy.deepcopy(rule_segment.get("hypothesis_scores", {})),
                "fired_rule_ids": [row.get("rule_id") for row in rule_segment.get("fired_rules", [])],
            })
    uncertain_ids = {int(row["segment_id"]) for row in selected["uncertain_segments"]}
    final_segments = []
    for segment in selected["segments"]:
        row = copy.deepcopy(segment)
        row["validation_status"] = "uncertain" if int(row.get("segment_id", -1)) in uncertain_ids else "validated"
        row["uncertainty_reasons"] = next((item["reasons"] for item in selected["uncertain_segments"] if int(item["segment_id"]) == int(row.get("segment_id", -2))), [])
        final_segments.append(row)
    return {
        "version": VERSION,
        "video_id": str(video_id),
        "status": "completed",
        "input_label_status": "provisional",
        "output_label_status": "refined_candidate",
        "stabilized": stabilized,
        "stop_reason": "thresholds_and_labels_stable" if stabilized else "deterministic_iteration_limit",
        "iterations": history,
        "selected_candidate_id": selected["candidate_id"],
        "selected_thresholds": copy.deepcopy(selected["thresholds"]),
        "provisional_thresholds": copy.deepcopy(provisional_video.get("selected_thresholds", {})),
        "threshold_changes": {
            key: float(selected["thresholds"].get(key, 0.0)) - float(provisional_video.get("selected_thresholds", {}).get(key, 0.0))
            for key in sorted(set(selected["thresholds"]) | set(provisional_video.get("selected_thresholds", {})))
        },
        "provisional_segments": copy.deepcopy(provisional_segments),
        "refined_segments": final_segments,
        "corrections": corrections,
        "uncertain_segments": copy.deepcopy(selected["uncertain_segments"]),
        "selected_normalized_evidence": selected["normalized_evidence"],
        "selected_global_rule_evaluation": selected["global_rule_evaluation"],
        "candidate_rankings": [
            {
                key: copy.deepcopy(value)
                for key, value in candidate.items()
                if key not in {"normalized_evidence", "global_rule_evaluation", "actions"}
            }
            for candidate in evaluated
        ],
        "selection_reason": "lexicographic_minimum: hard violations, soft severity, unexplained, reversals, short segments, complexity, signal fit",
        "provenance": {
            "source_steps": ["7a", "7b", "7c", "7d"],
            "deterministic": True,
            "candidate_count": len(evaluated),
            "max_iterations": max_iterations,
            "llm_used": False,
        },
    }
