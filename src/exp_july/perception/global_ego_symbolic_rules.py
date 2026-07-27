"""Shared, deterministic symbolic rules for Step 7D ego hypotheses."""

from __future__ import annotations

import copy
import math
from collections import defaultdict


VERSION = 2
RULE_POLICY_ID = "global_ego_motion_rules_v2"

# Values are dimensionless normalized ratios or video-local noise units.
RULES = (
    {"rule_id": "forward_radial_consensus", "hypothesis": "forward", "weight": 1.0, "conditions": (
        ("dominant_radial_direction", "eq", "expansion"),
        ("direction_support_ratio", "ge", 0.55),
        ("region_support_ratio", "ge", 1.0 / 3.0),
        ("temporal_persistence", "ge", 0.50),
    ), "reason": "Forward motion should produce persistent multi-region radial expansion."},
    {"rule_id": "forward_motion_agreement", "hypothesis": "forward", "weight": 0.8, "conditions": (
        ("normalized_motion_magnitude", "ge", 1.0),
        ("signed_direction_balance", "ge", 0.25),
        ("estimator_agreement", "ge", 0.50),
    ), "reason": "Motion above video noise should agree across regional and temporal estimators."},
    {"rule_id": "backward_radial_consensus", "hypothesis": "backward", "weight": 1.0, "conditions": (
        ("dominant_radial_direction", "eq", "contraction"),
        ("direction_support_ratio", "ge", 0.55),
        ("region_support_ratio", "ge", 1.0 / 3.0),
        ("temporal_persistence", "ge", 0.50),
    ), "reason": "Backward motion should produce persistent multi-region radial contraction."},
    {"rule_id": "backward_motion_agreement", "hypothesis": "backward", "weight": 0.8, "conditions": (
        ("normalized_motion_magnitude", "ge", 1.0),
        ("signed_direction_balance", "le", -0.25),
        ("estimator_agreement", "ge", 0.50),
    ), "reason": "Contraction above video noise should agree across estimators."},
    {"rule_id": "static_low_motion", "hypothesis": "static", "weight": 1.0, "conditions": (
        ("normalized_motion_magnitude", "le", 0.75),
        ("signed_direction_balance", "abs_le", 0.25),
        ("temporal_persistence", "ge", 0.50),
    ), "reason": "Static evidence requires persistent motion within the video-local noise scale."},
    {"rule_id": "static_neutral_regions", "hypothesis": "static", "weight": 0.8, "conditions": (
        ("dominant_radial_direction", "eq", "neutral"),
        ("region_support_ratio", "ge", 1.0 / 3.0),
        ("estimator_agreement", "ge", 0.50),
    ), "reason": "Neutral motion should be supported across multiple regions."},
    {"rule_id": "left_background_flow", "hypothesis": "left", "weight": 1.0, "conditions": (
        ("background_flow_right_support_ratio", "ge", 0.55),
        ("region_support_ratio", "ge", 1.0 / 3.0),
        ("temporal_persistence", "ge", 0.50),
    ), "reason": "A left ego turn is supported by coherent rightward background flow."},
    {"rule_id": "left_turning_structure", "hypothesis": "left", "weight": 0.8, "conditions": (
        ("horizontal_flow_balance", "ge", 0.25),
        ("turning_structure_support", "ge", 0.25),
        ("estimator_agreement", "ge", 0.40),
    ), "reason": "Rightward flow balance and spatial structure support a left hypothesis."},
    {"rule_id": "right_background_flow", "hypothesis": "right", "weight": 1.0, "conditions": (
        ("background_flow_left_support_ratio", "ge", 0.55),
        ("region_support_ratio", "ge", 1.0 / 3.0),
        ("temporal_persistence", "ge", 0.50),
    ), "reason": "A right ego turn is supported by coherent leftward background flow."},
    {"rule_id": "right_turning_structure", "hypothesis": "right", "weight": 0.8, "conditions": (
        ("horizontal_flow_balance", "le", -0.25),
        ("turning_structure_support", "ge", 0.25),
        ("estimator_agreement", "ge", 0.40),
    ), "reason": "Leftward flow balance and spatial structure support a right hypothesis."},
    {"rule_id": "turning_multiregion_motion", "hypothesis": "turning", "weight": 1.0, "conditions": (
        ("turning_structure_support", "ge", 0.25),
        ("region_support_ratio", "ge", 1.0 / 3.0),
        ("temporal_persistence", "ge", 0.50),
        ("normalized_motion_magnitude", "ge", 0.75),
    ), "reason": "Turning requires persistent structured horizontal motion across regions."},
    {"rule_id": "turning_estimator_consensus", "hypothesis": "turning", "weight": 0.8, "conditions": (
        ("estimator_agreement", "ge", 0.50),
        ("horizontal_flow_balance", "abs_ge", 0.25),
        ("direction_support_ratio", "ge", 0.40),
    ), "reason": "Turning structure must be corroborated by independent estimators."},
)

HYPOTHESES = ("forward", "backward", "static", "left", "right", "turning")


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _condition(field, operator, expected, evidence):
    observed = evidence.get(field)
    if operator == "eq":
        passed = str(observed) == str(expected)
        satisfaction = 1.0 if passed else 0.0
    else:
        value = _number(observed)
        threshold = float(expected)
        if operator == "ge":
            passed = value >= threshold
            satisfaction = 1.0 if threshold <= 0 else max(0.0, min(1.0, value / threshold))
        elif operator == "le":
            passed = value <= threshold
            if threshold < 0:
                satisfaction = max(0.0, min(1.0, value / threshold)) if value < 0 else 0.0
            else:
                satisfaction = 1.0 if value <= threshold else max(0.0, threshold / max(value, 1e-12))
        elif operator == "abs_le":
            passed = abs(value) <= threshold
            satisfaction = 1.0 if passed else max(0.0, threshold / max(abs(value), 1e-12))
        elif operator == "abs_ge":
            passed = abs(value) >= threshold
            satisfaction = 1.0 if threshold <= 0 else max(0.0, min(1.0, abs(value) / threshold))
        else:
            raise ValueError(f"Unsupported symbolic operator: {operator}")
    return {
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "satisfaction": float(satisfaction),
    }


def _provisional_family(action):
    action = str(action)
    if action in {"forward", "backward", "static"}:
        return action
    if action in {"left", "turning_left"}:
        return "left"
    if action in {"right", "turning_right"}:
        return "right"
    return "unknown"


def evaluate_segment(segment):
    fired = []
    violated = []
    scores_by_hypothesis = defaultdict(list)
    for rule in RULES:
        atoms = [
            _condition(field, operator, expected, segment)
            for field, operator, expected in rule["conditions"]
        ]
        match_score = math.prod(atom["satisfaction"] for atom in atoms)
        row = {
            "rule_id": rule["rule_id"],
            "hypothesis": rule["hypothesis"],
            "weight": rule["weight"],
            "reason": rule["reason"],
            "atoms": atoms,
            "matched_atoms": [atom["field"] for atom in atoms if atom["passed"]],
            "violated_atoms": [atom["field"] for atom in atoms if not atom["passed"]],
            "rule_match_score": float(match_score),
        }
        scores_by_hypothesis[rule["hypothesis"]].append((float(rule["weight"]), match_score))
        (fired if all(atom["passed"] for atom in atoms) else violated).append(row)
    scores = {}
    for hypothesis in HYPOTHESES:
        rows = scores_by_hypothesis[hypothesis]
        denominator = sum(weight for weight, _ in rows)
        scores[hypothesis] = float(sum(weight * score for weight, score in rows) / max(denominator, 1e-12))
    ranking = sorted(scores, key=lambda hypothesis: (-scores[hypothesis], HYPOTHESES.index(hypothesis)))
    conflicts = []
    for left, right in (("forward", "backward"), ("forward", "static"), ("backward", "static"), ("left", "right")):
        if scores[left] >= 0.50 and scores[right] >= 0.50:
            conflicts.append({
                "type": "mutually_exclusive_hypotheses",
                "hypotheses": [left, right],
                "scores": [scores[left], scores[right]],
            })
    provisional = _provisional_family(segment.get("provisional_action"))
    best = ranking[0]
    if provisional in scores and best != provisional and scores[best] - scores[provisional] >= 0.15:
        conflicts.append({
            "type": "provisional_label_disagreement",
            "provisional_hypothesis": provisional,
            "best_evidence_hypothesis": best,
            "score_margin": float(scores[best] - scores[provisional]),
        })
    chain = [
        f"Segment {segment.get('segment_id')} starts from provisional '{segment.get('provisional_action', 'unknown')}'.",
        f"Evaluated {len(RULES)} shared rules using video-normalized evidence only.",
        f"Fired {len(fired)} rules; {len(violated)} rules contain one or more violated atoms.",
        f"Highest evidence score is {best}={scores[best]:.3f}.",
    ]
    if fired:
        chain.append("Fired: " + ", ".join(row["rule_id"] for row in fired) + ".")
    if conflicts:
        chain.append("Conflicts: " + ", ".join(row["type"] for row in conflicts) + ".")
    else:
        chain.append("No symbolic hypothesis conflict was detected.")
    return {
        "segment_id": int(segment.get("segment_id", 0)),
        "start_frame": int(segment.get("start_frame", 0)),
        "end_frame": int(segment.get("end_frame", 0)),
        "provisional_action": str(segment.get("provisional_action", "unknown")),
        "fired_rules": fired,
        "violated_rules": violated,
        "hypothesis_scores": scores,
        "hypothesis_ranking": ranking,
        "conflicts": conflicts,
        "supporting_evidence_values": {
            key: copy.deepcopy(value)
            for key, value in segment.items()
            if key not in {"estimator_agreement_audit", "uncertainty_components", "provenance"}
        },
        "reasoning_chain": chain,
        "reasoning_text": " ".join(chain),
        "decision_authority": "evidence_evaluation_only_no_label_change",
    }


def evaluate_video(calibrated_video):
    evaluations = [
        evaluate_segment(segment)
        for segment in calibrated_video.get("normalized_segment_evidence", [])
    ]
    return {
        "version": VERSION,
        "rule_policy_id": RULE_POLICY_ID,
        "video_id": str(calibrated_video.get("video_id", "")),
        "status": "completed",
        "input_label_status": "provisional",
        "output_role": "symbolic_hypothesis_evaluation_not_final_labels",
        "shared_rule_policy": copy.deepcopy(RULES),
        "hypotheses": list(HYPOTHESES),
        "segment_evaluations": evaluations,
        "num_segments": len(evaluations),
        "num_fired_rules": sum(len(row["fired_rules"]) for row in evaluations),
        "num_violated_rules": sum(len(row["violated_rules"]) for row in evaluations),
        "num_conflicts": sum(len(row["conflicts"]) for row in evaluations),
        "provenance": {
            "source_step": "7d_global_symbolic_rule_evaluation",
            "source_evidence_step": "7c_video_local_evidence_calibration",
            "rule_scope": "global_shared_across_all_videos",
            "deterministic": True,
            "labels_modified": False,
        },
    }
