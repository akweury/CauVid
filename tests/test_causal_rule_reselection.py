import csv
import json
from pathlib import Path

from src.exp_driving_videos.modules import causal_rule_reselection_driving_mini as reselection


def _write_rule_causal_summary(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rule_id",
        "clause",
        "confidence",
        "trigger_count",
        "prediction_flip_count",
        "helpful_count",
        "harmful_count",
        "non_decisive_contribution_count",
        "redundant_count",
        "necessary_true_positive_count",
        "causal_false_positive_count",
        "score_delta_sum",
        "net_helpful_minus_harmful",
        "dominant_influence_type",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _rule(rule_id, clause, confidence=0.7, positive_support=1, rank=None):
    payload = {
        "rule_id": rule_id,
        "clause": clause,
        "confidence": confidence,
        "positive_support": positive_support,
        "negative_support": 0,
        "total_support": positive_support,
        "body_atom_templates": [clause.split(":-", 1)[1].strip()] if ":-" in clause else [],
    }
    if rank is not None:
        payload["rank"] = rank
        payload["original_rank"] = rank
        payload["original_score"] = confidence
    return payload


def test_reselection_removes_harmful_and_refills_from_ranked_pool(tmp_path):
    output_root = tmp_path / "18n"
    causal_csv = tmp_path / "18m" / "rule_causal_summary.csv"
    _write_rule_causal_summary(
        causal_csv,
        [
            {
                "rule_id": "harmful",
                "clause": "brake_next(S) :- object_distance_state(S,O,far).",
                "confidence": 0.5,
                "trigger_count": 3,
                "prediction_flip_count": 3,
                "helpful_count": 0,
                "harmful_count": 3,
                "necessary_true_positive_count": 0,
                "causal_false_positive_count": 3,
                "dominant_influence_type": "harmful_causal_source",
            },
            {
                "rule_id": "necessary",
                "clause": "brake_next(S) :- object_distance_state(S,O,near).",
                "confidence": 0.9,
                "trigger_count": 2,
                "prediction_flip_count": 2,
                "helpful_count": 2,
                "harmful_count": 0,
                "necessary_true_positive_count": 2,
                "dominant_influence_type": "helpful_for_correct_prediction",
            },
            {
                "rule_id": "mixed",
                "clause": "brake_next(S) :- object_speed_state(S,O,rel_static).",
                "confidence": 0.8,
                "trigger_count": 4,
                "prediction_flip_count": 4,
                "helpful_count": 1,
                "harmful_count": 3,
                "necessary_true_positive_count": 1,
                "causal_false_positive_count": 3,
                "dominant_influence_type": "harmful_causal_source",
            },
            {
                "rule_id": "orphan18m",
                "clause": "brake_next(S) :- object_visibility_state(S,O,intermittent_visibility).",
                "confidence": 0.3,
                "trigger_count": 1,
                "prediction_flip_count": 1,
                "helpful_count": 0,
                "harmful_count": 1,
                "necessary_true_positive_count": 0,
                "causal_false_positive_count": 1,
                "dominant_influence_type": "harmful_causal_source",
            },
        ],
    )
    final_rules = [
        _rule("harmful", "brake_next(S) :- object_distance_state(S,O,far).", 0.5, rank=1),
        _rule("necessary", "brake_next(S) :- object_distance_state(S,O,near).", 0.9, rank=2),
        _rule("mixed", "brake_next(S) :- object_speed_state(S,O,rel_static).", 0.8, rank=3),
        _rule("missing", "brake_next(S) :- object_vz_state(S,O,vz_awaying).", 0.6, rank=4),
        _rule("weak", "brake_next(S) :- object_vz_state(S,O,vz_approaching).", 0.4, rank=5),
    ]
    ranked_rules = final_rules + [
        _rule("duplicate_harmful", "brake_next(S) :- object_distance_state(S,O,far).", 0.95, rank=6),
        _rule("zero_support", "brake_next(S) :- segment_motion_state(S,forward_static).", 0.93, 0, rank=7),
        _rule("refill", "brake_next(S) :- segment_motion_state(S,forward_slowdown_left).", 0.7, 2, rank=8),
    ]

    result = reselection.process_reselection(
        final_rule_results={"final_rules": final_rules, "ranked_rules": ranked_rules},
        evaluation_results={"rule_evaluations": []},
        rule_level_causal_masking_results={"output_paths": {"rule_causal_summary_csv": str(causal_csv)}},
        cfg={
            "top_k": 5,
            "broad_weak_predicates": ["vz_approaching", "vz_awaying", "intermittent_visibility"],
            "ego_motion_only_predicates": ["ego_forward_slowdown"],
            "skip_zero_positive_support_refill": True,
            "remove_mixed_harmful_dominant": False,
        },
        output_root=output_root,
        force_recompute=True,
    )

    rows = {row["rule_id"]: row for row in result["reselection_rows"]}
    assert rows["harmful"]["assigned_reselection_category"] == "harmful_false_positive_rule"
    assert rows["harmful"]["reselection_decision"] == "remove"
    assert rows["harmful"]["blacklist_status"] == "blacklisted_harmful"
    assert rows["necessary"]["reselection_decision"] == "keep"
    assert rows["mixed"]["assigned_reselection_category"] == "mixed_rule"
    assert rows["mixed"]["reselection_decision"] == "refine"
    assert rows["missing"]["found_in_step18m"] is False
    assert rows["missing"]["assigned_reselection_category"] != "redundant_rule"
    assert rows["missing"]["reselection_decision"] in {"keep_for_review", "refine"}
    assert rows["refill"]["reselection_decision"] == "refill"
    assert rows["refill"]["selection_source"] == "ranked_pool_refill"
    assert rows["refill"]["found_in_step17"] is True
    assert rows["refill"]["found_in_step18m"] is False
    assert "causal effect statistics are missing" in rows["refill"]["warning_message"]
    assert rows["refill"]["replaced_removed_rule_id"] == "harmful"

    assert result["num_initial_final_rules"] == 5
    assert result["num_removed_rules"] == 1
    assert result["num_refilled_rules"] == 1
    assert result["num_refined_final_rules"] == 5
    assert result["refined_final_rules_reached_top_k"] is True
    assert result["ranked_pool_candidates_scanned_for_refill"] == 8
    assert result["num_refill_skipped_duplicate"] == 4
    assert result["num_refill_skipped_blacklist"] == 2
    assert result["num_refill_skipped_zero_positive_support"] == 1
    assert result["warning_section"]["num_step17_rules_missing_from_step18m"] == 2
    assert result["warning_section"]["num_step18m_nonzero_causal_effect_rules_missing_from_step17"] == 1
    assert result["warning_section"]["step18m_nonzero_causal_effect_rules_missing_from_step17"][0]["rule_id"] == "orphan18m"

    refined = json.loads(Path(result["output_paths"]["refined_final_rules_json"]).read_text(encoding="utf-8"))
    refined_rule_ids = {rule["rule_id"] for rule in refined["final_rules"]}
    assert "harmful" not in refined_rule_ids
    assert "refill" in refined_rule_ids
    assert len(refined_rule_ids) == 5

    with Path(result["output_paths"]["refined_final_rules_csv"]).open("r", encoding="utf-8", newline="") as fh:
        refined_csv_rows = list(csv.DictReader(fh))
    for column in [
        "selection_source",
        "reselection_decision",
        "blacklist_status",
        "refill_reason",
        "replaced_removed_rule_id",
    ]:
        assert column in refined_csv_rows[0]

    with Path(result["output_paths"]["refilled_rules_csv"]).open("r", encoding="utf-8", newline="") as fh:
        refilled_rows = list(csv.DictReader(fh))
    assert [row["rule_id"] for row in refilled_rows] == ["refill"]

    with Path(result["output_paths"]["removed_rules_csv"]).open("r", encoding="utf-8", newline="") as fh:
        removed_rows = list(csv.DictReader(fh))
    assert removed_rows[0]["rule_id"] == "harmful"
    assert removed_rows[0]["removal_reason"] == "pure harmful causal false-positive source"

    with Path(result["output_paths"]["refinement_targets_csv"]).open("r", encoding="utf-8", newline="") as fh:
        refinement_rows = {row["rule_id"]: row for row in csv.DictReader(fh)}
    assert "mixed" in refinement_rows
    assert "weak" in refinement_rows
    assert refinement_rows["mixed"]["suggested_action"] == "refine_with_more_specific_descendant"
