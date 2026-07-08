"""
Evaluate the Step 18N refined rule set with the Step 18 protocol.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config
from src.exp_driving_videos.modules import evaluate_rules_driving_mini


_REFINED_EVALUATION_VERSION = 1
_PER_ROUND_RULE_EVALUATION_FIELDS = [
    "round_index",
    "rule_set_name",
    "num_rules",
    "num_removed_rules",
    "num_refilled_rules",
    "num_blacklisted_rules",
    "top_k_reached",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "selected_as_best",
    "early_stop_reason",
]
_PER_ROUND_REMOVED_FIELDS = [
    "round_index",
    "rule_id",
    "clause",
    "original_rank",
    "removal_reason",
    "helpful_count",
    "harmful_count",
    "necessary_true_positive_count",
    "causal_false_positive_count",
    "prediction_flip_count",
]
_PER_ROUND_REFILLED_FIELDS = [
    "round_index",
    "rule_id",
    "clause",
    "original_rank",
    "original_score",
    "refill_reason",
    "replaced_removed_rule_id",
    "selection_source",
]
_BEST_RULE_FIELDS = [
    "rule_id",
    "clause",
    "original_rank",
    "original_score",
    "selection_source",
    "reselection_decision",
    "assigned_reselection_category",
    "helpful_count",
    "harmful_count",
    "necessary_true_positive_count",
    "causal_false_positive_count",
    "prediction_flip_count",
    "found_in_step18m",
    "blacklist_status",
    "refill_reason",
    "replaced_removed_rule_id",
    "refined_rank",
]


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "18o_driving_mini_refined_rule_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prediction_mode": str(cfg.get("prediction_mode", "any_rule_positive")),
        "evaluated_rule_set_name": "refined",
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _metric(metrics: Dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _comparison_rows(
    original_evaluation_results: Dict[str, Any],
    refined_evaluation_results: Dict[str, Any],
    refined_rule_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    original = dict(original_evaluation_results.get("overall_metrics", {}))
    refined = dict(refined_evaluation_results.get("overall_metrics", {}))
    reselection = dict(refined_rule_results)
    keys = [
        ("TP", "true_positive"),
        ("FP", "false_positive"),
        ("FN", "false_negative"),
        ("TN", "true_negative"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("F1", "f1"),
        ("accuracy", "accuracy"),
    ]
    rows: List[Dict[str, Any]] = []
    for label, key in keys:
        original_value = _metric(original, key)
        refined_value = _metric(refined, key)
        rows.append(
            {
                "metric": label,
                "original_step18": original_value,
                "refined_step18o": refined_value,
                "delta_refined_minus_original": refined_value - original_value,
            }
        )
    original_rules = int(original_evaluation_results.get("num_final_rules", 0))
    refined_rules = int(refined_evaluation_results.get("num_final_rules", 0))
    rows.extend(
        [
            {
                "metric": "number_of_selected_rules",
                "original_step18": original_rules,
                "refined_step18o": refined_rules,
                "delta_refined_minus_original": refined_rules - original_rules,
            },
            {
                "metric": "removed_harmful_rules",
                "original_step18": 0,
                "refined_step18o": int(reselection.get("num_removed_harmful_rules", 0)),
                "delta_refined_minus_original": int(reselection.get("num_removed_harmful_rules", 0)),
            },
            {
                "metric": "refilled_replacement_rules",
                "original_step18": 0,
                "refined_step18o": int(reselection.get("num_refilled_rules", 0)),
                "delta_refined_minus_original": int(reselection.get("num_refilled_rules", 0)),
            },
            {
                "metric": "refinement_targets",
                "original_step18": 0,
                "refined_step18o": int(reselection.get("num_refinement_targets", 0)),
                "delta_refined_minus_original": int(reselection.get("num_refinement_targets", 0)),
            },
            {
                "metric": "retained_necessary_rules",
                "original_step18": 0,
                "refined_step18o": int(reselection.get("num_retained_necessary_rules", 0)),
                "delta_refined_minus_original": int(reselection.get("num_retained_necessary_rules", 0)),
            },
            {
                "metric": "mixed_refinement_targets",
                "original_step18": 0,
                "refined_step18o": int(reselection.get("num_mixed_refinement_targets", 0)),
                "delta_refined_minus_original": int(reselection.get("num_mixed_refinement_targets", 0)),
            },
        ]
    )
    return rows


def _main_conclusion(comparison_rows: Sequence[Dict[str, Any]]) -> str:
    by_metric = {str(row.get("metric", "")): row for row in comparison_rows}
    fp_delta = _metric(by_metric.get("FP", {}), "delta_refined_minus_original")
    recall_delta = _metric(by_metric.get("recall", {}), "delta_refined_minus_original")
    f1_delta = _metric(by_metric.get("F1", {}), "delta_refined_minus_original")
    if fp_delta < 0 and recall_delta >= -1e-9:
        return (
            "Causal-aware top-k maintenance reduced false positives while preserving recall "
            f"(delta_fp={fp_delta:.0f}, delta_recall={recall_delta:.3f}, delta_f1={f1_delta:.3f})."
        )
    if fp_delta < 0:
        return (
            "Causal-aware top-k maintenance reduced false positives but changed recall "
            f"(delta_fp={fp_delta:.0f}, delta_recall={recall_delta:.3f}, delta_f1={f1_delta:.3f})."
        )
    return (
        "Causal-aware top-k maintenance did not reduce false positives "
        f"(delta_fp={fp_delta:.0f}, delta_recall={recall_delta:.3f}, delta_f1={f1_delta:.3f})."
    )


def _per_round_rows(refined_rule_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = list(refined_rule_results.get("per_round_evaluation_rows", []))
    if rows:
        return [dict(row) for row in rows]
    output_rows: List[Dict[str, Any]] = []
    for record in list(refined_rule_results.get("rounds", [])):
        metrics = dict(record.get("metrics", {}))
        output_rows.append(
            {
                "round_index": int(record.get("round_index", 0)),
                "rule_set_name": str(record.get("rule_set_name", "")),
                "num_rules": len(list(record.get("final_rules", []))),
                "num_removed_rules": int(record.get("num_removed_rules", 0)),
                "num_refilled_rules": int(record.get("num_refilled_rules", 0)),
                "num_blacklisted_rules": int(record.get("num_blacklisted_rules", 0)),
                "top_k_reached": bool(record.get("top_k_reached", False)),
                "true_positive": int(metrics.get("true_positive", 0)),
                "false_positive": int(metrics.get("false_positive", 0)),
                "false_negative": int(metrics.get("false_negative", 0)),
                "true_negative": int(metrics.get("true_negative", 0)),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "selected_as_best": bool(record.get("selected_as_best", False)),
                "early_stop_reason": str(record.get("early_stop_reason", "")),
            }
        )
    return output_rows


def _per_round_detail_rows(refined_rule_results: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in list(refined_rule_results.get("rounds", [])):
        round_index = int(record.get("round_index", 0))
        for row in list(record.get(key, [])):
            rows.append({"round_index": round_index, **dict(row)})
    return rows


def _best_rule_rows(refined_rule_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    best_round_index = int(refined_rule_results.get("best_round_index", 0))
    for record in list(refined_rule_results.get("rounds", [])):
        if int(record.get("round_index", -1)) == best_round_index and list(record.get("selected_rows", [])):
            return [dict(row) for row in list(record.get("selected_rows", []))]
    rows: List[Dict[str, Any]] = []
    for index, rule in enumerate(list(refined_rule_results.get("final_rules", [])), start=1):
        reselection = dict(rule.get("causal_reselection", {}))
        rows.append(
            {
                "rule_id": str(rule.get("rule_id", "")),
                "clause": str(rule.get("clause", "")),
                "original_rank": reselection.get("original_rank", rule.get("original_rank", index)),
                "original_score": reselection.get("original_score", rule.get("original_score", rule.get("confidence", 0.0))),
                "selection_source": reselection.get("selection_source", "best_round"),
                "reselection_decision": reselection.get("reselection_decision", "keep"),
                "assigned_reselection_category": reselection.get("assigned_reselection_category", ""),
                "helpful_count": reselection.get("helpful_count", 0),
                "harmful_count": reselection.get("harmful_count", 0),
                "necessary_true_positive_count": reselection.get("necessary_true_positive_count", 0),
                "causal_false_positive_count": reselection.get("causal_false_positive_count", 0),
                "prediction_flip_count": reselection.get("prediction_flip_count", 0),
                "found_in_step18m": reselection.get("found_in_step18m", ""),
                "blacklist_status": reselection.get("blacklist_status", ""),
                "refill_reason": reselection.get("refill_reason", ""),
                "replaced_removed_rule_id": reselection.get("replaced_removed_rule_id", ""),
                "refined_rank": int(rule.get("refined_rank", index)),
            }
        )
    return rows


def process_refined_evaluation(
    refined_rule_results: Dict[str, Any],
    original_evaluation_results: Dict[str, Any],
    temporal_rule_results: Sequence[Dict[str, Any]],
    logic_atom_results: Sequence[Dict[str, Any]],
    eval_video_ids: Sequence[str],
    split_manifest: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    comparison_csv_path = out_root / "refined_rule_evaluation_comparison.csv"
    comparison_json_path = out_root / "refined_rule_evaluation_comparison.json"
    per_round_rule_evaluation_csv_path = out_root / "per_round_rule_evaluation.csv"
    per_round_removed_rules_csv_path = out_root / "per_round_removed_rules.csv"
    per_round_refilled_rules_csv_path = out_root / "per_round_refilled_rules.csv"
    iterative_reselection_summary_csv_path = out_root / "iterative_reselection_summary.csv"
    best_round_refined_final_rules_csv_path = out_root / "best_round_refined_final_rules.csv"
    best_round_manifest_path = out_root / "best_round_manifest.json"

    eval_cfg = {
        "prediction_mode": str(cfg.get("prediction_mode", "any_rule_positive")),
        "evaluated_rule_set_name": "refined",
    }
    refined_evaluation = evaluate_rules_driving_mini.run(
        final_rule_results=refined_rule_results,
        temporal_rule_results=temporal_rule_results,
        logic_atom_results=logic_atom_results,
        eval_video_ids=list(eval_video_ids),
        split_manifest=split_manifest,
        cfg=eval_cfg,
        output_root=out_root,
        force_recompute=force_recompute,
    )
    comparison_rows = _comparison_rows(
        original_evaluation_results=original_evaluation_results,
        refined_evaluation_results=refined_evaluation,
        refined_rule_results=refined_rule_results,
    )
    _write_csv(
        comparison_csv_path,
        ["metric", "original_step18", "refined_step18o", "delta_refined_minus_original"],
        comparison_rows,
    )
    per_round_rows = _per_round_rows(refined_rule_results)
    per_round_removed_rows = _per_round_detail_rows(refined_rule_results, "removed_rules")
    per_round_refilled_rows = _per_round_detail_rows(refined_rule_results, "refilled_rules")
    best_rule_rows = _best_rule_rows(refined_rule_results)
    _write_csv(per_round_rule_evaluation_csv_path, _PER_ROUND_RULE_EVALUATION_FIELDS, per_round_rows)
    _write_csv(iterative_reselection_summary_csv_path, _PER_ROUND_RULE_EVALUATION_FIELDS, per_round_rows)
    _write_csv(per_round_removed_rules_csv_path, _PER_ROUND_REMOVED_FIELDS, per_round_removed_rows)
    _write_csv(per_round_refilled_rules_csv_path, _PER_ROUND_REFILLED_FIELDS, per_round_refilled_rows)
    _write_csv(best_round_refined_final_rules_csv_path, _BEST_RULE_FIELDS, best_rule_rows)
    best_round_manifest = {
        "best_round_index": int(refined_rule_results.get("best_round_index", 0)),
        "selected_on_validation_performance": bool(refined_rule_results.get("selected_on_validation_performance", False)),
        "selection_metric": str(refined_rule_results.get("selection_metric", "f1")),
        "tie_breaker": str(refined_rule_results.get("tie_breaker", "precision")),
        "best_round_metrics": dict(refined_rule_results.get("best_round_metrics", {})),
        "final_test_evaluation_uses_best_validation_round_once": True,
    }
    with best_round_manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(best_round_manifest, fh, indent=2)
    comparison = {
        "version": _REFINED_EVALUATION_VERSION,
        "config": _cfg_key_subset(cfg),
        "comparison_rows": comparison_rows,
        "main_conclusion": _main_conclusion(comparison_rows),
        "usage_constraints": {
            "same_protocol_as_step18": True,
            "refined_rules_from_step18n_only": True,
            "does_not_add_rules_or_facts": True,
        },
        "output_paths": {
            "comparison_csv": str(comparison_csv_path),
            "comparison_json": str(comparison_json_path),
            "rule_evaluation_json": str(out_root / "rule_evaluation.json"),
            "rule_evaluation_csv": str(out_root / "rule_evaluation.csv"),
            "example_predictions_csv": str(out_root / "example_predictions.csv"),
            "per_round_rule_evaluation_csv": str(per_round_rule_evaluation_csv_path),
            "per_round_removed_rules_csv": str(per_round_removed_rules_csv_path),
            "per_round_refilled_rules_csv": str(per_round_refilled_rules_csv_path),
            "iterative_reselection_summary_csv": str(iterative_reselection_summary_csv_path),
            "best_round_refined_final_rules_csv": str(best_round_refined_final_rules_csv_path),
            "best_round_manifest_json": str(best_round_manifest_path),
        },
    }
    with comparison_json_path.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)
    refined_evaluation["refined_comparison"] = comparison
    refined_evaluation["output_paths"] = {
        **dict(refined_evaluation.get("output_paths", {})),
        **comparison["output_paths"],
    }
    print(
        "  refined_rule_evaluation: "
        f"precision={float(dict(refined_evaluation.get('overall_metrics', {})).get('precision', 0.0)):.3f} | "
        f"recall={float(dict(refined_evaluation.get('overall_metrics', {})).get('recall', 0.0)):.3f} | "
        f"f1={float(dict(refined_evaluation.get('overall_metrics', {})).get('f1', 0.0)):.3f}"
    )
    return refined_evaluation


def run(
    refined_rule_results: Dict[str, Any],
    original_evaluation_results: Dict[str, Any],
    temporal_rule_results: Sequence[Dict[str, Any]],
    logic_atom_results: Sequence[Dict[str, Any]],
    eval_video_ids: Sequence[str],
    split_manifest: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_refined_evaluation(
        refined_rule_results=refined_rule_results,
        original_evaluation_results=original_evaluation_results,
        temporal_rule_results=temporal_rule_results,
        logic_atom_results=logic_atom_results,
        eval_video_ids=eval_video_ids,
        split_manifest=split_manifest,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
