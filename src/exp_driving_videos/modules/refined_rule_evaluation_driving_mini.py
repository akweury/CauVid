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
                "metric": "number_of_removed_harmful_rules",
                "original_step18": 0,
                "refined_step18o": int(reselection.get("num_removed_harmful_rules", 0)),
                "delta_refined_minus_original": int(reselection.get("num_removed_harmful_rules", 0)),
            },
            {
                "metric": "number_of_retained_necessary_rules",
                "original_step18": 0,
                "refined_step18o": int(reselection.get("num_retained_necessary_rules", 0)),
                "delta_refined_minus_original": int(reselection.get("num_retained_necessary_rules", 0)),
            },
            {
                "metric": "number_of_mixed_refinement_targets",
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
            "Causal masking-guided reselection reduced false positives while preserving recall "
            f"(delta_fp={fp_delta:.0f}, delta_recall={recall_delta:.3f}, delta_f1={f1_delta:.3f})."
        )
    if fp_delta < 0:
        return (
            "Causal masking-guided reselection reduced false positives but changed recall "
            f"(delta_fp={fp_delta:.0f}, delta_recall={recall_delta:.3f}, delta_f1={f1_delta:.3f})."
        )
    return (
        "Causal masking-guided reselection did not reduce false positives "
        f"(delta_fp={fp_delta:.0f}, delta_recall={recall_delta:.3f}, delta_f1={f1_delta:.3f})."
    )


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
