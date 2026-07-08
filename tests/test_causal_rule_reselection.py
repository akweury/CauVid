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


def test_reselection_distinguishes_missing_step18m_from_redundant(tmp_path):
    output_root = tmp_path / "18n"
    causal_csv = tmp_path / "18m" / "rule_causal_summary.csv"
    _write_rule_causal_summary(
        causal_csv,
        [
            {
                "rule_id": "redundant_backup",
                "clause": "brake_next(S) :- object_distance_state(S,O,near).",
                "confidence": 0.7,
                "trigger_count": 12,
                "prediction_flip_count": 0,
                "redundant_count": 12,
                "dominant_influence_type": "redundant_trigger",
            },
            {
                "rule_id": "harmful",
                "clause": "brake_next(S) :- object_distance_state(S,O,far).",
                "confidence": 0.5,
                "trigger_count": 3,
                "prediction_flip_count": 3,
                "helpful_count": 0,
                "harmful_count": 3,
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
                "rule_id": "weak_vz",
                "clause": "brake_next(S) :- object_vz_state(S,O,vz_approaching).",
                "confidence": 0.4,
                "trigger_count": 4,
                "prediction_flip_count": 0,
                "redundant_count": 4,
                "dominant_influence_type": "redundant_trigger",
            },
            {
                "rule_id": "orphan_effect",
                "clause": "brake_next(S) :- object_visibility_state(S,O,intermittent).",
                "confidence": 0.3,
                "trigger_count": 1,
                "prediction_flip_count": 1,
                "harmful_count": 1,
                "causal_false_positive_count": 1,
                "dominant_influence_type": "harmful_causal_source",
            },
        ],
    )
    final_rules = [
        {
            "rule_id": "missing",
            "clause": "brake_next(S) :- object_vz_state(S,O,vz_awaying).",
            "confidence": 0.6,
            "body_atom_templates": ["object_vz_state(S,O,vz_awaying)."],
        },
        {
            "rule_id": "redundant_backup",
            "clause": "brake_next(S) :- object_distance_state(S,O,near).",
            "confidence": 0.7,
            "body_atom_templates": ["object_distance_state(S,O,near)."],
        },
        {
            "rule_id": "harmful",
            "clause": "brake_next(S) :- object_distance_state(S,O,far).",
            "confidence": 0.5,
            "body_atom_templates": ["object_distance_state(S,O,far)."],
        },
        {
            "rule_id": "necessary",
            "clause": "brake_next(S) :- object_distance_state(S,O,near).",
            "confidence": 0.9,
            "body_atom_templates": ["object_distance_state(S,O,near)."],
        },
        {
            "rule_id": "weak_vz",
            "clause": "brake_next(S) :- object_vz_state(S,O,vz_approaching).",
            "confidence": 0.4,
            "body_atom_templates": ["object_vz_state(S,O,vz_approaching)."],
        },
    ]

    result = reselection.process_reselection(
        final_rule_results={"final_rules": final_rules},
        evaluation_results={
            "rule_evaluations": [
                {"rule_id": "missing", "eval_total_firings": 5},
                {"rule_id": "redundant_backup", "eval_total_firings": 12},
            ]
        },
        rule_level_causal_masking_results={"output_paths": {"rule_causal_summary_csv": str(causal_csv)}},
        cfg={
            "top_k": 10,
            "broad_weak_predicates": ["vz_approaching", "vz_awaying", "intermittent_visibility"],
            "ego_motion_only_predicates": ["ego_forward_slowdown"],
            "backup_explanation_min_trigger_count": 5,
        },
        output_root=output_root,
        force_recompute=True,
    )

    rows = {row["rule_id"]: row for row in result["reselection_rows"]}
    assert rows["missing"]["assigned_reselection_category"] == "causal_effect_missing"
    assert rows["missing"]["reselection_decision"] == "keep_for_review"
    assert rows["missing"]["found_in_step18m"] is False
    assert rows["missing"]["missing_from_step18m"] is True
    assert "not found in Step 18M" in rows["missing"]["warning_message"]

    assert rows["harmful"]["reselection_decision"] == "remove"
    assert rows["necessary"]["reselection_decision"] == "keep"
    assert rows["weak_vz"]["assigned_reselection_category"] == "weak_causal_grounding_rule"
    assert rows["weak_vz"]["reselection_decision"] == "refine"
    assert rows["redundant_backup"]["reselection_decision"] == "backup_explanation"
    assert rows["redundant_backup"]["backup_explanation_candidate"] is True

    assert rows["orphan_effect"]["found_in_step17"] is False
    assert rows["orphan_effect"]["reselection_decision"] == "warning_only"
    assert result["warning_section"]["num_step17_rules_missing_from_step18m"] == 1
    assert result["warning_section"]["num_step18m_nonzero_causal_effect_rules_missing_from_step17"] == 1

    with Path(result["output_paths"]["causal_rule_reselection_summary_csv"]).open(
        "r", encoding="utf-8", newline=""
    ) as fh:
        csv_rows = list(csv.DictReader(fh))
    assert "found_in_step18m" in csv_rows[0]
    assert "missing_from_step18m" in csv_rows[0]
    assert "found_in_step17" in csv_rows[0]
    assert "causal_effect_status" in csv_rows[0]
    assert "backup_explanation_candidate" in csv_rows[0]
    assert "warning_message" in csv_rows[0]

    refined = json.loads(Path(result["output_paths"]["refined_final_rules_json"]).read_text(encoding="utf-8"))
    refined_rule_ids = {rule["rule_id"] for rule in refined["final_rules"]}
    assert "orphan_effect" not in refined_rule_ids
    assert "harmful" not in refined_rule_ids
