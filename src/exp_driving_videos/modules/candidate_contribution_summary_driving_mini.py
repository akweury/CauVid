"""
Summarize whether candidate-derived atoms help or hurt brake_next rule selection.

Consumes:
  - Step 17 selected-rule outputs with candidate provenance metadata
  - Step 18 evaluation outputs with candidate-aware ablation diagnostics

Output layout:
    pipeline_output/18a_driving_mini_candidate_contribution_summary/
        candidate_contribution_summary.json
        candidate_contribution_summary.csv
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


_SUMMARY_VERSION = 1


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "18a_driving_mini_candidate_contribution_summary"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "top_useful_candidate_rules": int(cfg.get("top_useful_candidate_rules", 10)),
        "top_noisy_candidate_rules": int(cfg.get("top_noisy_candidate_rules", 10)),
        "top_matched_priors": int(cfg.get("top_matched_priors", 10)),
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_impact_label(
    delta_f1: float,
    delta_precision: float,
    delta_recall: float,
    fn_gain_count: int,
    fp_contribution_count: int,
) -> str:
    if abs(delta_f1) < 1e-9 and fn_gain_count == 0 and fp_contribution_count == 0:
        return "no_effect"
    if delta_f1 > 0.0 and fn_gain_count >= fp_contribution_count and delta_recall >= -1e-9:
        return "improves_reasoning"
    if delta_f1 < 0.0 and fp_contribution_count > fn_gain_count and delta_precision <= 1e-9:
        return "mostly_adds_noise"
    return "mixed_tradeoff"


def _candidate_rule_rows(rule_evaluations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rule in rule_evaluations:
        if not bool(rule.get("uses_candidate_atoms", False)):
            continue
        fn_gain = _safe_int(rule.get("eval_fn_coverage_gain_count_vs_accepted_only", 0))
        fp_contrib = _safe_int(rule.get("eval_fp_contribution_count_vs_accepted_only", 0))
        rows.append(
            {
                "rule_id": str(rule.get("rule_id", "")),
                "clause": str(rule.get("clause", "")),
                "candidate_rule_category": str(rule.get("candidate_rule_category", "")),
                "matched_prior_ids_involved": [
                    str(prior_id)
                    for prior_id in list(rule.get("matched_prior_ids_involved", []))
                    if str(prior_id)
                ],
                "candidate_body_atom_ratio": _safe_float(rule.get("candidate_body_atom_ratio", 0.0)),
                "num_candidate_body_atoms": _safe_int(rule.get("num_candidate_body_atoms", 0)),
                "eval_precision": _safe_float(rule.get("eval_precision", 0.0)),
                "eval_recall": _safe_float(rule.get("eval_recall", 0.0)),
                "eval_f1": _safe_float(rule.get("eval_f1", 0.0)),
                "eval_positive_support": _safe_int(rule.get("eval_positive_support", 0)),
                "eval_negative_support": _safe_int(rule.get("eval_negative_support", 0)),
                "eval_total_firings": _safe_int(rule.get("eval_total_firings", 0)),
                "fn_coverage_gain_count": fn_gain,
                "fp_contribution_count": fp_contrib,
                "net_candidate_utility": fn_gain - fp_contrib,
                "fn_coverage_gain_example_ids": list(
                    rule.get("eval_fn_coverage_gain_example_ids_vs_accepted_only", [])
                ),
                "fp_contribution_example_ids": list(
                    rule.get("eval_fp_contribution_example_ids_vs_accepted_only", [])
                ),
            }
        )
    return rows


def _top_useful_candidate_rules(
    candidate_rule_rows: Sequence[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        candidate_rule_rows,
        key=lambda row: (
            -_safe_int(row.get("net_candidate_utility", 0)),
            -_safe_int(row.get("fn_coverage_gain_count", 0)),
            -_safe_float(row.get("eval_f1", 0.0)),
            -_safe_float(row.get("eval_precision", 0.0)),
            _safe_int(row.get("fp_contribution_count", 0)),
            str(row.get("rule_id", "")),
        ),
    )
    useful = [
        row for row in ranked
        if _safe_int(row.get("fn_coverage_gain_count", 0)) > 0 or _safe_int(row.get("net_candidate_utility", 0)) > 0
    ]
    return useful[: max(0, limit)]


def _top_noisy_candidate_rules(
    candidate_rule_rows: Sequence[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        candidate_rule_rows,
        key=lambda row: (
            -_safe_int(row.get("fp_contribution_count", 0)),
            _safe_float(row.get("eval_precision", 0.0)),
            -_safe_int(row.get("eval_negative_support", 0)),
            -_safe_float(row.get("candidate_body_atom_ratio", 0.0)),
            str(row.get("rule_id", "")),
        ),
    )
    noisy = [
        row for row in ranked
        if _safe_int(row.get("fp_contribution_count", 0)) > 0 or _safe_int(row.get("eval_negative_support", 0)) > 0
    ]
    return noisy[: max(0, limit)]


def _matched_prior_utility_stats(
    candidate_rule_rows: Sequence[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    by_prior: Dict[str, Dict[str, Any]] = {}
    for row in candidate_rule_rows:
        prior_ids = list(row.get("matched_prior_ids_involved", []))
        for prior_id in prior_ids:
            prior_text = str(prior_id).strip()
            if not prior_text:
                continue
            entry = by_prior.setdefault(
                prior_text,
                {
                    "prior_id": prior_text,
                    "selected_rule_count": 0,
                    "candidate_only_rule_count": 0,
                    "mixed_rule_count": 0,
                    "sum_fn_coverage_gain_count": 0,
                    "sum_fp_contribution_count": 0,
                    "sum_eval_precision": 0.0,
                    "sum_eval_recall": 0.0,
                    "sum_eval_f1": 0.0,
                    "sum_eval_positive_support": 0,
                    "sum_eval_negative_support": 0,
                    "rule_ids": [],
                },
            )
            entry["selected_rule_count"] += 1
            if str(row.get("candidate_rule_category", "")) == "candidate_only":
                entry["candidate_only_rule_count"] += 1
            if str(row.get("candidate_rule_category", "")) == "mixed_accepted_candidate":
                entry["mixed_rule_count"] += 1
            entry["sum_fn_coverage_gain_count"] += _safe_int(row.get("fn_coverage_gain_count", 0))
            entry["sum_fp_contribution_count"] += _safe_int(row.get("fp_contribution_count", 0))
            entry["sum_eval_precision"] += _safe_float(row.get("eval_precision", 0.0))
            entry["sum_eval_recall"] += _safe_float(row.get("eval_recall", 0.0))
            entry["sum_eval_f1"] += _safe_float(row.get("eval_f1", 0.0))
            entry["sum_eval_positive_support"] += _safe_int(row.get("eval_positive_support", 0))
            entry["sum_eval_negative_support"] += _safe_int(row.get("eval_negative_support", 0))
            entry["rule_ids"].append(str(row.get("rule_id", "")))

    rows: List[Dict[str, Any]] = []
    for prior_id, entry in by_prior.items():
        selected_rule_count = max(1, _safe_int(entry.get("selected_rule_count", 0)))
        fn_gain = _safe_int(entry.get("sum_fn_coverage_gain_count", 0))
        fp_contrib = _safe_int(entry.get("sum_fp_contribution_count", 0))
        rows.append(
            {
                "prior_id": prior_id,
                "selected_rule_count": selected_rule_count,
                "candidate_only_rule_count": _safe_int(entry.get("candidate_only_rule_count", 0)),
                "mixed_rule_count": _safe_int(entry.get("mixed_rule_count", 0)),
                "sum_fn_coverage_gain_count": fn_gain,
                "sum_fp_contribution_count": fp_contrib,
                "net_candidate_utility": fn_gain - fp_contrib,
                "avg_eval_precision": _safe_float(entry.get("sum_eval_precision", 0.0)) / selected_rule_count,
                "avg_eval_recall": _safe_float(entry.get("sum_eval_recall", 0.0)) / selected_rule_count,
                "avg_eval_f1": _safe_float(entry.get("sum_eval_f1", 0.0)) / selected_rule_count,
                "sum_eval_positive_support": _safe_int(entry.get("sum_eval_positive_support", 0)),
                "sum_eval_negative_support": _safe_int(entry.get("sum_eval_negative_support", 0)),
                "rule_ids": sorted({rule_id for rule_id in entry.get("rule_ids", []) if str(rule_id)}),
            }
        )

    rows.sort(
        key=lambda row: (
            -_safe_int(row.get("net_candidate_utility", 0)),
            -_safe_int(row.get("sum_fn_coverage_gain_count", 0)),
            _safe_int(row.get("sum_fp_contribution_count", 0)),
            -_safe_float(row.get("avg_eval_f1", 0.0)),
            str(row.get("prior_id", "")),
        )
    )
    return rows[: max(0, limit)]


def _selector_summary(
    rule_set_name: str,
    selection_method: str,
    rule_result: Dict[str, Any],
    evaluation_result: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    ablation = dict(evaluation_result.get("candidate_rule_ablation", {}))
    baseline_metrics = dict(ablation.get("baseline_metrics", {}))
    augmented_metrics = dict(ablation.get("augmented_metrics", {}))
    category_counts = dict(dict(rule_result.get("candidate_rule_diagnostics", {})).get("category_counts", {}))
    rule_evaluations = list(evaluation_result.get("rule_evaluations", []))
    candidate_rows = _candidate_rule_rows(rule_evaluations)
    useful_rules = _top_useful_candidate_rules(
        candidate_rows,
        limit=int(cfg.get("top_useful_candidate_rules", 10)),
    )
    noisy_rules = _top_noisy_candidate_rules(
        candidate_rows,
        limit=int(cfg.get("top_noisy_candidate_rules", 10)),
    )
    matched_prior_stats = _matched_prior_utility_stats(
        candidate_rows,
        limit=int(cfg.get("top_matched_priors", 10)),
    )
    delta_precision = _safe_float(ablation.get("delta_precision", 0.0))
    delta_recall = _safe_float(ablation.get("delta_recall", 0.0))
    delta_f1 = _safe_float(ablation.get("delta_f1", 0.0))
    fn_gain_count = _safe_int(ablation.get("fn_coverage_gain_count", 0))
    fp_contribution_count = _safe_int(ablation.get("fp_contribution_count", 0))
    impact_label = _candidate_impact_label(
        delta_f1=delta_f1,
        delta_precision=delta_precision,
        delta_recall=delta_recall,
        fn_gain_count=fn_gain_count,
        fp_contribution_count=fp_contribution_count,
    )
    return {
        "rule_set_name": rule_set_name,
        "selection_method": selection_method,
        "candidate_impact_label": impact_label,
        "num_final_rules": _safe_int(rule_result.get("num_final_rules", 0)),
        "selected_accepted_only_rule_count": _safe_int(category_counts.get("accepted_only_rules", 0)),
        "selected_candidate_only_rule_count": _safe_int(category_counts.get("candidate_only_rules", 0)),
        "selected_mixed_rule_count": _safe_int(category_counts.get("mixed_accepted_candidate_rules", 0)),
        "selected_candidate_involving_rule_count": _safe_int(category_counts.get("candidate_only_rules", 0))
        + _safe_int(category_counts.get("mixed_accepted_candidate_rules", 0)),
        "accepted_only_precision": _safe_float(baseline_metrics.get("precision", 0.0)),
        "accepted_only_recall": _safe_float(baseline_metrics.get("recall", 0.0)),
        "accepted_only_f1": _safe_float(baseline_metrics.get("f1", 0.0)),
        "accepted_plus_candidate_precision": _safe_float(augmented_metrics.get("precision", 0.0)),
        "accepted_plus_candidate_recall": _safe_float(augmented_metrics.get("recall", 0.0)),
        "accepted_plus_candidate_f1": _safe_float(augmented_metrics.get("f1", 0.0)),
        "delta_precision": delta_precision,
        "delta_recall": delta_recall,
        "delta_f1": delta_f1,
        "fn_coverage_gain_count": fn_gain_count,
        "fn_coverage_gain_rate": _safe_float(ablation.get("fn_coverage_gain_rate", 0.0)),
        "fp_contribution_count": fp_contribution_count,
        "fp_contribution_rate": _safe_float(ablation.get("fp_contribution_rate", 0.0)),
        "recovered_fn_examples": list(ablation.get("recovered_false_negative_example_ids", [])),
        "introduced_fp_examples": list(ablation.get("added_false_positive_example_ids", [])),
        "top_useful_candidate_involving_rules": useful_rules,
        "top_noisy_candidate_involving_rules": noisy_rules,
        "matched_prior_utility_statistics": matched_prior_stats,
    }


def process(
    rule_results_by_name: Dict[str, Dict[str, Any]],
    evaluation_results_by_name: Dict[str, Dict[str, Any]],
    primary_rule_set: str,
    evaluation_rule_sets: Sequence[str],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "candidate_contribution_summary.json"
    csv_path = out_root / "candidate_contribution_summary.csv"

    if not force_recompute and json_path.exists():
        with json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _SUMMARY_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {json_path.name}")
            return cached

    selector_summaries: List[Dict[str, Any]] = []
    for rule_set_name in evaluation_rule_sets:
        if rule_set_name not in rule_results_by_name or rule_set_name not in evaluation_results_by_name:
            continue
        selector_summaries.append(
            _selector_summary(
                rule_set_name=rule_set_name,
                selection_method=str(rule_results_by_name[rule_set_name].get("selection_method", "score_top_k")),
                rule_result=rule_results_by_name[rule_set_name],
                evaluation_result=evaluation_results_by_name[rule_set_name],
                cfg=cfg,
            )
        )

    best_selector_by_delta_f1 = ""
    if selector_summaries:
        best_selector_by_delta_f1 = str(
            max(
                selector_summaries,
                key=lambda row: (
                    _safe_float(row.get("delta_f1", 0.0)),
                    _safe_int(row.get("fn_coverage_gain_count", 0)),
                    -_safe_int(row.get("fp_contribution_count", 0)),
                    str(row.get("rule_set_name", "")),
                ),
            ).get("rule_set_name", "")
        )

    result: Dict[str, Any] = {
        "version": _SUMMARY_VERSION,
        "config": _cfg_key_subset(cfg),
        "primary_rule_set": str(primary_rule_set),
        "evaluation_rule_sets": [str(name) for name in evaluation_rule_sets],
        "best_selector_by_delta_f1": best_selector_by_delta_f1,
        "selectors": selector_summaries,
        "csv_path": str(csv_path),
    }

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rule_set_name",
                "selection_method",
                "candidate_impact_label",
                "num_final_rules",
                "selected_accepted_only_rule_count",
                "selected_candidate_only_rule_count",
                "selected_mixed_rule_count",
                "selected_candidate_involving_rule_count",
                "accepted_only_precision",
                "accepted_only_recall",
                "accepted_only_f1",
                "accepted_plus_candidate_precision",
                "accepted_plus_candidate_recall",
                "accepted_plus_candidate_f1",
                "delta_precision",
                "delta_recall",
                "delta_f1",
                "fn_coverage_gain_count",
                "fn_coverage_gain_rate",
                "fp_contribution_count",
                "fp_contribution_rate",
                "recovered_fn_examples",
                "introduced_fp_examples",
                "top_useful_candidate_involving_rule_ids",
                "top_noisy_candidate_involving_rule_ids",
                "top_matched_prior_utility_ids",
            ],
        )
        writer.writeheader()
        for row in selector_summaries:
            writer.writerow(
                {
                    "rule_set_name": row.get("rule_set_name", ""),
                    "selection_method": row.get("selection_method", ""),
                    "candidate_impact_label": row.get("candidate_impact_label", ""),
                    "num_final_rules": row.get("num_final_rules", 0),
                    "selected_accepted_only_rule_count": row.get("selected_accepted_only_rule_count", 0),
                    "selected_candidate_only_rule_count": row.get("selected_candidate_only_rule_count", 0),
                    "selected_mixed_rule_count": row.get("selected_mixed_rule_count", 0),
                    "selected_candidate_involving_rule_count": row.get("selected_candidate_involving_rule_count", 0),
                    "accepted_only_precision": row.get("accepted_only_precision", 0.0),
                    "accepted_only_recall": row.get("accepted_only_recall", 0.0),
                    "accepted_only_f1": row.get("accepted_only_f1", 0.0),
                    "accepted_plus_candidate_precision": row.get("accepted_plus_candidate_precision", 0.0),
                    "accepted_plus_candidate_recall": row.get("accepted_plus_candidate_recall", 0.0),
                    "accepted_plus_candidate_f1": row.get("accepted_plus_candidate_f1", 0.0),
                    "delta_precision": row.get("delta_precision", 0.0),
                    "delta_recall": row.get("delta_recall", 0.0),
                    "delta_f1": row.get("delta_f1", 0.0),
                    "fn_coverage_gain_count": row.get("fn_coverage_gain_count", 0),
                    "fn_coverage_gain_rate": row.get("fn_coverage_gain_rate", 0.0),
                    "fp_contribution_count": row.get("fp_contribution_count", 0),
                    "fp_contribution_rate": row.get("fp_contribution_rate", 0.0),
                    "recovered_fn_examples": json.dumps(row.get("recovered_fn_examples", [])),
                    "introduced_fp_examples": json.dumps(row.get("introduced_fp_examples", [])),
                    "top_useful_candidate_involving_rule_ids": json.dumps(
                        [rule.get("rule_id", "") for rule in row.get("top_useful_candidate_involving_rules", [])]
                    ),
                    "top_noisy_candidate_involving_rule_ids": json.dumps(
                        [rule.get("rule_id", "") for rule in row.get("top_noisy_candidate_involving_rules", [])]
                    ),
                    "top_matched_prior_utility_ids": json.dumps(
                        [prior.get("prior_id", "") for prior in row.get("matched_prior_utility_statistics", [])]
                    ),
                }
            )

    print(
        "  candidate_contribution_summary: "
        f"rule_sets={len(selector_summaries)} | "
        f"best_delta_f1_selector={best_selector_by_delta_f1 or 'n/a'}"
    )
    print(f"Candidate contribution summary JSON written to {json_path}")
    print(f"Candidate contribution summary CSV written to {csv_path}")
    return result


def run(
    rule_results_by_name: Dict[str, Dict[str, Any]],
    evaluation_results_by_name: Dict[str, Dict[str, Any]],
    primary_rule_set: str,
    evaluation_rule_sets: Sequence[str],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process(
        rule_results_by_name=rule_results_by_name,
        evaluation_results_by_name=evaluation_results_by_name,
        primary_rule_set=primary_rule_set,
        evaluation_rule_sets=evaluation_rule_sets,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
