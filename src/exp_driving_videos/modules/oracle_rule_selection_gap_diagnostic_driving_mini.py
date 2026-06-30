"""
Inspect the oracle-selected top-K rules from the rule-pool upper-bound
diagnostic, compare them with the actual selector outputs, and explain why the
current selectors missed oracle rules that materially improve held-out F1.

Outputs:
    pipeline_output/17e_driving_mini_oracle_rule_selection_gap_diagnostic/
        oracle_selection_gap_summary.json
        oracle_target_rule_set.csv
        selector_oracle_overlap_summary.csv
        selector_missing_oracle_rules.csv
        selector_extra_selected_rules.csv
        selector_missing_reason_summary.csv
        oracle_selection_gap_report.md
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config
from src.exp_driving_videos.modules.diverse_final_rules_driving_mini import (
    _effective_semantic_min_family_counts,
    _positive_example_ids,
    _rule_family_signature,
    _rule_is_semantic_quota_qualified,
    _rule_utility,
    _rule_vehicle_match_level,
)
from src.exp_driving_videos.modules.final_rules_driving_mini import _sort_rules as _sort_original_rules
from src.exp_driving_videos.modules.final_rules_driving_mini import _post_pruned_rule_pool


_ORACLE_GAP_VERSION = 2


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "17e_driving_mini_oracle_rule_selection_gap_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "oracle_target_mode": str(cfg.get("oracle_target_mode", "peak_f1")),
        "max_missing_rules_per_selector": int(cfg.get("max_missing_rules_per_selector", 20)),
        "max_extra_rules_per_selector": int(cfg.get("max_extra_rules_per_selector", 20)),
    }


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return dict(json.load(fh))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def _selector_selected_rules(selector_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules = [dict(rule) for rule in list(selector_result.get("final_rules", []))]
    if rules and any("selection_rank" in rule for rule in rules):
        return sorted(
            rules,
            key=lambda rule: (
                _safe_int(rule.get("selection_rank", 10**9)),
                str(rule.get("rule_id", "")),
            ),
        )
    return list(rules)


def _oracle_target_rows(
    oracle_curve_rows: Sequence[Dict[str, Any]],
    target_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not oracle_curve_rows:
        return [], {}

    sorted_rows = sorted(
        [dict(row) for row in oracle_curve_rows],
        key=lambda row: _safe_int(row.get("selection_rank", 0)),
    )
    if str(target_mode).strip().lower() == "selection_reference_k":
        target_row = sorted_rows[-1]
    else:
        target_row = max(
            sorted_rows,
            key=lambda row: (
                _safe_float(row.get("f1", 0.0)),
                _safe_float(row.get("recall", 0.0)),
                _safe_float(row.get("precision", 0.0)),
                -_safe_int(row.get("selection_rank", 0)),
            ),
        )
    target_k = _safe_int(target_row.get("selection_rank", 0))
    return sorted_rows[:target_k], target_row


def _rule_lookup(extended_rule_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(rule.get("rule_id", "")): dict(rule)
        for rule in _post_pruned_rule_pool(extended_rule_results)
        if str(rule.get("rule_id", ""))
    }


def _metric_lookup(rule_pool_upper_bound_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    scatter_path = Path(
        str(
            rule_pool_upper_bound_results.get("output_paths", {}).get(
                "rule_pool_precision_recall_scatter_csv",
                "",
            )
        )
    )
    if not scatter_path.exists():
        return {}
    rows = _read_csv(scatter_path)
    return {str(row.get("rule_id", "")): dict(row) for row in rows if str(row.get("rule_id", ""))}


def _update_semantic_selection_state(
    rule: Dict[str, Any],
    selector_cfg: Dict[str, Any],
    selected_qualified_semantic_families: Dict[str, Set[str]],
) -> None:
    family_signature = _rule_family_signature(rule)
    semantic_match_level = _rule_vehicle_match_level(rule, selector_cfg)
    if semantic_match_level == "no_match":
        return
    if _rule_is_semantic_quota_qualified(rule, selector_cfg):
        selected_qualified_semantic_families.setdefault(semantic_match_level, set()).add(family_signature)


def _build_greedy_selector_contexts(
    selector_result: Dict[str, Any],
    selector_cfg: Dict[str, Any],
    candidate_rules: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    selected_rules = _selector_selected_rules(selector_result)
    contexts: List[Dict[str, Any]] = []
    covered_positive_ids: Set[str] = set()
    family_counts: Dict[str, int] = {}
    selected_qualified_semantic_families: Dict[str, Set[str]] = {}
    effective_semantic_counts = _effective_semantic_min_family_counts(candidate_rules, selector_cfg)
    top_k = max(0, _safe_int(selector_cfg.get("top_k", len(selected_rules)), len(selected_rules)))

    for step_index, chosen_rule in enumerate(selected_rules):
        contexts.append(
            {
                "step_index": step_index,
                "remaining_slots": max(0, top_k - step_index),
                "covered_positive_ids": set(covered_positive_ids),
                "family_counts": dict(family_counts),
                "selected_qualified_semantic_families": {
                    key: set(value)
                    for key, value in selected_qualified_semantic_families.items()
                },
                "effective_semantic_min_family_counts": dict(effective_semantic_counts),
                "chosen_rule": dict(chosen_rule),
            }
        )
        family_signature = _rule_family_signature(chosen_rule)
        family_counts[family_signature] = family_counts.get(family_signature, 0) + 1
        covered_positive_ids.update(_positive_example_ids(chosen_rule))
        _update_semantic_selection_state(
            chosen_rule,
            selector_cfg,
            selected_qualified_semantic_families,
        )
    return contexts


def _infer_original_reason(
    oracle_rule: Dict[str, Any],
    oracle_metrics: Dict[str, Any],
    sorted_rules: Sequence[Dict[str, Any]],
    selected_ids: Set[str],
    top_k: int,
) -> Dict[str, Any]:
    rule_id = str(oracle_rule.get("rule_id", ""))
    ranked_ids = [str(rule.get("rule_id", "")) for rule in sorted_rules]
    rank_index = ranked_ids.index(rule_id) + 1 if rule_id in ranked_ids else -1
    cutoff_rule = dict(sorted_rules[top_k - 1]) if top_k > 0 and top_k <= len(sorted_rules) else {}
    reason_label = "ranked_below_confidence_top_k_cutoff"
    reason_text = (
        "The original selector is pure confidence/support ranking, and this oracle rule was below the top-k cutoff."
    )
    if rank_index <= 0:
        reason_label = "rule_not_present_in_ranked_pool"
        reason_text = "The oracle rule could not be located in the ranked final-rule input."
    return {
        "reason_label": reason_label,
        "reason_text": reason_text,
        "rank_within_original_sort": rank_index,
        "top_k_cutoff_rank": top_k,
        "cutoff_rule_id": str(cutoff_rule.get("rule_id", "")),
        "cutoff_rule_confidence": _safe_float(cutoff_rule.get("confidence", 0.0)),
        "cutoff_rule_positive_support": _safe_int(cutoff_rule.get("positive_support", 0)),
        "oracle_rule_confidence": _safe_float(oracle_rule.get("confidence", 0.0)),
        "oracle_rule_positive_support": _safe_int(oracle_rule.get("positive_support", 0)),
        "oracle_rule_eval_f1": _safe_float(oracle_metrics.get("f1", 0.0)),
        "oracle_rule_eval_precision": _safe_float(oracle_metrics.get("precision", 0.0)),
        "oracle_rule_eval_recall": _safe_float(oracle_metrics.get("recall", 0.0)),
        "is_selected": rule_id in selected_ids,
    }


def _reason_from_greedy_trace(
    trace: Dict[str, Any],
    chosen_rule: Dict[str, Any],
) -> Tuple[str, str]:
    if bool(trace.get("semantic_hard_constraint_active", False)) and _safe_int(trace.get("semantic_deficit_reduction", 0)) <= 0:
        return (
            "semantic_constraint_blocked_rule",
            "Semantic quota pressure was active at this step, so the selector deprioritized a rule that did not reduce an outstanding semantic deficit.",
        )
    if _safe_int(trace.get("new_positive_gain", 0)) <= 0:
        return (
            "redundant_after_earlier_coverage",
            "Earlier selections had already covered nearly all of this rule's positives, leaving little or no marginal gain under the greedy objective.",
        )
    if _safe_int(trace.get("family_reuse_count", 0)) > 0 and _safe_float(trace.get("family_penalty_value", 0.0)) >= _safe_float(trace.get("overlap_penalty_value", 0.0)):
        return (
            "family_reuse_penalty",
            "The selector had already taken rules from the same predicate family, so family reuse penalties pushed this oracle rule below the chosen alternative.",
        )
    if _safe_float(trace.get("overlap_penalty_value", 0.0)) > 0.0 and _safe_int(trace.get("overlap_positive_count", 0)) >= _safe_int(trace.get("new_positive_gain", 0)):
        return (
            "overlap_penalty_and_low_marginal_gain",
            "The rule overlaps heavily with already covered positives, so its marginal gain was too small after overlap penalties.",
        )
    if _safe_float(trace.get("negative_support_penalty_value", 0.0)) > 0.0 and _safe_int(trace.get("negative_support", 0)) > 0:
        return (
            "negative_support_penalty",
            "The rule carried non-trivial negative support, and the selector's negative-support penalty reduced its utility.",
        )
    if _safe_float(trace.get("base_quality_score", 0.0)) < _safe_float(chosen_rule.get("selection_base_quality_score", 0.0)):
        return (
            "lower_quality_score_than_chosen_rule",
            "Its confidence * log(1 + positive_support) quality score was weaker than the rule actually chosen at the step where it had its best chance.",
        )
    return (
        "outscored_by_higher_utility_rule",
        "At its best opportunity, the rule was still outscored by another rule with better combined marginal coverage and selector utility.",
    )


def _infer_greedy_reason(
    selector_name: str,
    oracle_rule: Dict[str, Any],
    oracle_metrics: Dict[str, Any],
    selector_result: Dict[str, Any],
    candidate_rules: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    selector_cfg = dict(selector_result.get("config", {}))
    contexts = _build_greedy_selector_contexts(selector_result, selector_cfg, candidate_rules)
    best_trace: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, float, float, float]] = None

    for context in contexts:
        utility = _rule_utility(
            oracle_rule,
            context["covered_positive_ids"],
            context["family_counts"],
            context["selected_qualified_semantic_families"],
            context["effective_semantic_min_family_counts"],
            _safe_int(context.get("remaining_slots", 0)),
            selector_cfg,
        )
        chosen_rule = dict(context.get("chosen_rule", {}))
        utility_gap = _safe_float(chosen_rule.get("selection_utility", 0.0)) - _safe_float(utility.get("utility", 0.0))
        trace = {
            "step_index": _safe_int(context.get("step_index", 0)),
            "chosen_rule": chosen_rule,
            "utility": _safe_float(utility.get("utility", 0.0)),
            "new_positive_gain": _safe_int(utility.get("new_positive_gain", 0)),
            "overlap_positive_count": _safe_int(utility.get("overlap_positive_count", 0)),
            "family_reuse_count": _safe_int(utility.get("family_reuse_count", 0)),
            "base_quality_score": _safe_float(utility.get("base_quality_score", 0.0)),
            "family_diversity_bonus": _safe_float(utility.get("family_diversity_bonus", 0.0)),
            "overlap_penalty_value": _safe_float(utility.get("overlap_penalty_value", 0.0)),
            "family_penalty_value": _safe_float(utility.get("family_penalty_value", 0.0)),
            "negative_support_penalty_value": _safe_float(utility.get("negative_support_penalty", 0.0)),
            "semantic_match_level": str(utility.get("semantic_match_level", "no_match")),
            "semantic_is_quota_qualified": bool(utility.get("semantic_is_quota_qualified", False)),
            "semantic_deficit_reduction": _safe_int(utility.get("semantic_deficit_reduction", 0)),
            "semantic_hard_constraint_active": bool(utility.get("semantic_hard_constraint_active", False)),
            "semantic_total_remaining_deficits": _safe_int(utility.get("semantic_total_remaining_deficits", 0)),
            "utility_gap_to_chosen_rule": utility_gap,
            "remaining_slots": _safe_int(context.get("remaining_slots", 0)),
            "negative_support": _safe_int(oracle_rule.get("negative_support", 0)),
        }
        key = (
            trace["utility"],
            trace["new_positive_gain"],
            -trace["utility_gap_to_chosen_rule"],
            -trace["step_index"],
        )
        if best_trace is None or key > best_key:
            best_trace = trace
            best_key = key

    if best_trace is None:
        return {
            "reason_label": "selector_trace_unavailable",
            "reason_text": "Could not reconstruct the selector trace for this oracle rule.",
        }

    reason_label, reason_text = _reason_from_greedy_trace(best_trace, dict(best_trace.get("chosen_rule", {})))
    chosen_rule = dict(best_trace.get("chosen_rule", {}))
    return {
        "reason_label": reason_label,
        "reason_text": reason_text,
        "best_chance_step": _safe_int(best_trace.get("step_index", 0)),
        "oracle_rule_utility_at_best_chance": _safe_float(best_trace.get("utility", 0.0)),
        "chosen_rule_id_at_best_chance": str(chosen_rule.get("rule_id", "")),
        "chosen_rule_utility_at_best_chance": _safe_float(chosen_rule.get("selection_utility", 0.0)),
        "utility_gap_to_chosen_rule": _safe_float(best_trace.get("utility_gap_to_chosen_rule", 0.0)),
        "new_positive_gain_at_best_chance": _safe_int(best_trace.get("new_positive_gain", 0)),
        "overlap_positive_count_at_best_chance": _safe_int(best_trace.get("overlap_positive_count", 0)),
        "family_reuse_count_at_best_chance": _safe_int(best_trace.get("family_reuse_count", 0)),
        "base_quality_score_at_best_chance": _safe_float(best_trace.get("base_quality_score", 0.0)),
        "family_diversity_bonus_at_best_chance": _safe_float(best_trace.get("family_diversity_bonus", 0.0)),
        "overlap_penalty_value_at_best_chance": _safe_float(best_trace.get("overlap_penalty_value", 0.0)),
        "family_penalty_value_at_best_chance": _safe_float(best_trace.get("family_penalty_value", 0.0)),
        "negative_support_penalty_value_at_best_chance": _safe_float(best_trace.get("negative_support_penalty_value", 0.0)),
        "semantic_match_level": str(best_trace.get("semantic_match_level", "no_match")),
        "semantic_is_quota_qualified": bool(best_trace.get("semantic_is_quota_qualified", False)),
        "semantic_deficit_reduction_at_best_chance": _safe_int(best_trace.get("semantic_deficit_reduction", 0)),
        "semantic_hard_constraint_active_at_best_chance": bool(best_trace.get("semantic_hard_constraint_active", False)),
        "semantic_total_remaining_deficits_at_best_chance": _safe_int(best_trace.get("semantic_total_remaining_deficits", 0)),
        "oracle_rule_eval_f1": _safe_float(oracle_metrics.get("f1", 0.0)),
        "oracle_rule_eval_precision": _safe_float(oracle_metrics.get("precision", 0.0)),
        "oracle_rule_eval_recall": _safe_float(oracle_metrics.get("recall", 0.0)),
        "selector_name": selector_name,
    }


def _selector_explanation_rows(
    selector_name: str,
    selector_result: Dict[str, Any],
    oracle_target_rows: Sequence[Dict[str, Any]],
    rule_by_id: Dict[str, Dict[str, Any]],
    metric_by_rule_id: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    selected_rules = _selector_selected_rules(selector_result)
    selected_ids = {str(rule.get("rule_id", "")) for rule in selected_rules if str(rule.get("rule_id", ""))}
    oracle_target_ids = [str(row.get("rule_id", "")) for row in oracle_target_rows if str(row.get("rule_id", ""))]
    oracle_target_id_set = set(oracle_target_ids)

    overlap_ids = [rule_id for rule_id in oracle_target_ids if rule_id in selected_ids]
    missed_ids = [rule_id for rule_id in oracle_target_ids if rule_id not in selected_ids]
    extra_ids = [str(rule.get("rule_id", "")) for rule in selected_rules if str(rule.get("rule_id", "")) and str(rule.get("rule_id", "")) not in oracle_target_id_set]

    candidate_rules = list(rule_by_id.values())
    selector_metrics = dict(selector_result.get("overall_metrics", {}))
    if not selector_metrics:
        selector_metrics = {}

    sorted_all_rules = _sort_original_rules(list(rule_by_id.values()))
    missing_rows: List[Dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    for oracle_rank, rule_id in enumerate(missed_ids, start=1):
        oracle_rule = dict(rule_by_id.get(rule_id, {}))
        oracle_metrics = dict(metric_by_rule_id.get(rule_id, {}))
        if selector_name == "original":
            reason_info = _infer_original_reason(
                oracle_rule,
                oracle_metrics,
                sorted_all_rules,
                selected_ids,
                _safe_int(selector_result.get("num_final_rules", len(selected_rules))),
            )
        else:
            reason_info = _infer_greedy_reason(
                selector_name,
                oracle_rule,
                oracle_metrics,
                selector_result,
                candidate_rules,
            )
        reason_counter[str(reason_info.get("reason_label", "unknown"))] += 1
        missing_rows.append(
            {
                "selector_name": selector_name,
                "oracle_rank": oracle_rank,
                "rule_id": rule_id,
                "clause": str(oracle_rule.get("clause", "")),
                "semantic_family": str(oracle_metrics.get("semantic_family", "")),
                "oracle_rule_confidence": _safe_float(oracle_rule.get("confidence", 0.0)),
                "oracle_rule_train_positive_support": _safe_int(oracle_rule.get("positive_support", 0)),
                "oracle_rule_eval_precision": _safe_float(oracle_metrics.get("precision", 0.0)),
                "oracle_rule_eval_recall": _safe_float(oracle_metrics.get("recall", 0.0)),
                "oracle_rule_eval_f1": _safe_float(oracle_metrics.get("f1", 0.0)),
                **reason_info,
            }
        )

    extra_rows: List[Dict[str, Any]] = []
    for selector_rank, rule_id in enumerate(extra_ids, start=1):
        selected_rule = dict(rule_by_id.get(rule_id, {}))
        selected_metrics = dict(metric_by_rule_id.get(rule_id, {}))
        extra_rows.append(
            {
                "selector_name": selector_name,
                "selector_extra_rank": selector_rank,
                "rule_id": rule_id,
                "clause": str(selected_rule.get("clause", "")),
                "semantic_family": str(selected_metrics.get("semantic_family", "")),
                "confidence": _safe_float(selected_rule.get("confidence", 0.0)),
                "train_positive_support": _safe_int(selected_rule.get("positive_support", 0)),
                "eval_precision": _safe_float(selected_metrics.get("precision", 0.0)),
                "eval_recall": _safe_float(selected_metrics.get("recall", 0.0)),
                "eval_f1": _safe_float(selected_metrics.get("f1", 0.0)),
            }
        )

    summary_row = {
        "selector_name": selector_name,
        "selection_method": str(selector_result.get("selection_method", "score_top_k")),
        "num_selected_rules": len(selected_rules),
        "oracle_target_k": len(oracle_target_rows),
        "num_overlap_with_oracle": len(overlap_ids),
        "num_missed_oracle_rules": len(missed_ids),
        "num_extra_selected_rules": len(extra_ids),
        "oracle_overlap_fraction": float(len(overlap_ids) / max(1, len(oracle_target_rows))),
        "selector_f1": _safe_float(selector_metrics.get("f1", 0.0)),
        "selector_precision": _safe_float(selector_metrics.get("precision", 0.0)),
        "selector_recall": _safe_float(selector_metrics.get("recall", 0.0)),
        "primary_miss_reason": reason_counter.most_common(1)[0][0] if reason_counter else "",
    }
    return summary_row, missing_rows, extra_rows, dict(reason_counter)


def _write_markdown_report(
    path: Path,
    *,
    oracle_target_row: Dict[str, Any],
    selector_summary_rows: Sequence[Dict[str, Any]],
    selector_category_gap_rows: Sequence[Dict[str, Any]],
    reason_counts_by_selector: Dict[str, Dict[str, int]],
    missing_rows: Sequence[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    target_k = _safe_int(oracle_target_row.get("selection_rank", 0))
    target_f1 = _safe_float(oracle_target_row.get("f1", 0.0))
    target_precision = _safe_float(oracle_target_row.get("precision", 0.0))
    target_recall = _safe_float(oracle_target_row.get("recall", 0.0))
    lines.append("# Oracle Rule Selection Gap Diagnostic")
    lines.append("")
    lines.append(
        f"The oracle greedy pool diagnostic peaks at K={target_k} with F1={target_f1:.3f}, "
        f"precision={target_precision:.3f}, recall={target_recall:.3f}."
    )
    lines.append("")
    lines.append("## Selector Comparison")
    lines.append("")
    for row in selector_summary_rows:
        selector_name = str(row.get("selector_name", ""))
        lines.append(
            f"- {selector_name}: F1={_safe_float(row.get('selector_f1', 0.0)):.3f}, "
            f"overlap_with_oracle={_safe_int(row.get('num_overlap_with_oracle', 0))}/{_safe_int(row.get('oracle_target_k', 0))}, "
            f"missed_oracle_rules={_safe_int(row.get('num_missed_oracle_rules', 0))}, "
            f"primary_miss_reason={str(row.get('primary_miss_reason', '')) or 'n/a'}."
        )
        reason_counts = reason_counts_by_selector.get(selector_name, {})
        if reason_counts:
            formatted = ", ".join(f"{name}: {count}" for name, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:4])
            lines.append(f"  Top miss reasons: {formatted}.")
    if selector_category_gap_rows:
        lines.append("")
        lines.append("## Category Oracle Gaps")
        lines.append("")
        for selector_name in sorted({str(row.get("selector_name", "")) for row in selector_category_gap_rows}):
            selector_rows = [
                row for row in selector_category_gap_rows if str(row.get("selector_name", "")) == selector_name
            ]
            if not selector_rows:
                continue
            lines.append(f"### {selector_name}")
            for row in selector_rows:
                lines.append(
                    f"- {str(row.get('subset_name', ''))}: oracle_f1={_safe_float(row.get('oracle_upper_bound_f1', 0.0)):.3f}, "
                    f"selector_f1={_safe_float(row.get('selector_f1', 0.0)):.3f}, "
                    f"gap={_safe_float(row.get('oracle_gap_f1', 0.0)):.3f}, "
                    f"fn_recovery={_safe_int(row.get('fn_recovery_beyond_accepted_only', 0))}, "
                    f"fp_cost={_safe_int(row.get('fp_cost', 0))}."
                )
            lines.append("")
    lines.append("")
    lines.append("## Representative Missed Oracle Rules")
    lines.append("")
    for selector_name in sorted({str(row.get("selector_name", "")) for row in missing_rows}):
        selector_rows = [row for row in missing_rows if str(row.get("selector_name", "")) == selector_name][:5]
        if not selector_rows:
            continue
        lines.append(f"### {selector_name}")
        lines.append("")
        for row in selector_rows:
            lines.append(
                f"- {str(row.get('rule_id', ''))}: eval_f1={_safe_float(row.get('oracle_rule_eval_f1', 0.0)):.3f}, "
                f"family={str(row.get('semantic_family', '')) or 'unknown'}, "
                f"reason={str(row.get('reason_label', ''))}. "
                f"{str(row.get('reason_text', ''))}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def process_diagnostic(
    extended_rule_results: Dict[str, Any],
    rule_pool_upper_bound_results: Dict[str, Any],
    rule_results_by_name: Dict[str, Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)

    summary_path = out_root / "oracle_selection_gap_summary.json"
    oracle_target_path = out_root / "oracle_target_rule_set.csv"
    overlap_summary_path = out_root / "selector_oracle_overlap_summary.csv"
    category_gap_path = out_root / "selector_category_oracle_gap_summary.csv"
    missing_rules_path = out_root / "selector_missing_oracle_rules.csv"
    extra_rules_path = out_root / "selector_extra_selected_rules.csv"
    reason_summary_path = out_root / "selector_missing_reason_summary.csv"
    report_path = out_root / "oracle_selection_gap_report.md"

    if not force_recompute and summary_path.exists():
        cached = _read_json(summary_path)
        if int(cached.get("version", 0)) == _ORACLE_GAP_VERSION and _cfg_key_subset(cached.get("config", {})) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {summary_path.name}")
            return cached

    oracle_curve_path = Path(
        str(
            rule_pool_upper_bound_results.get("output_paths", {}).get(
                "oracle_greedy_rule_set_curve_csv",
                "",
            )
        )
    )
    if not oracle_curve_path.exists():
        raise FileNotFoundError(f"Missing oracle curve CSV for Step 17E: {oracle_curve_path}")

    oracle_curve_rows = _read_csv(oracle_curve_path)
    oracle_target_rows, oracle_target_row = _oracle_target_rows(
        oracle_curve_rows,
        str(cfg.get("oracle_target_mode", "peak_f1")),
    )

    rule_by_id = _rule_lookup(extended_rule_results)
    metric_by_rule_id = _metric_lookup(rule_pool_upper_bound_results)

    oracle_target_export_rows: List[Dict[str, Any]] = []
    for row in oracle_target_rows:
        rule_id = str(row.get("rule_id", ""))
        full_rule = dict(rule_by_id.get(rule_id, {}))
        oracle_target_export_rows.append(
            {
                "oracle_rank": _safe_int(row.get("selection_rank", 0)),
                "rule_id": rule_id,
                "clause": str(full_rule.get("clause", row.get("clause", ""))),
                "semantic_family": str(row.get("semantic_family", "")),
                "confidence": _safe_float(full_rule.get("confidence", 0.0)),
                "train_positive_support": _safe_int(full_rule.get("positive_support", 0)),
                "train_negative_support": _safe_int(full_rule.get("negative_support", 0)),
                "added_positive_examples": _safe_int(row.get("added_positive_examples", 0)),
                "added_negative_examples": _safe_int(row.get("added_negative_examples", 0)),
                "precision": _safe_float(row.get("precision", 0.0)),
                "recall": _safe_float(row.get("recall", 0.0)),
                "f1": _safe_float(row.get("f1", 0.0)),
            }
        )

    selector_summary_rows: List[Dict[str, Any]] = []
    selector_category_gap_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    extra_rows: List[Dict[str, Any]] = []
    reason_summary_rows: List[Dict[str, Any]] = []
    reason_counts_by_selector: Dict[str, Dict[str, int]] = {}

    selector_metrics_by_name = dict(rule_pool_upper_bound_results.get("actual_selected_rule_set_metrics_by_name", {}))
    category_upper_bounds = dict(rule_pool_upper_bound_results.get("category_upper_bounds", {}))
    category_oracle_gap_by_selector = dict(rule_pool_upper_bound_results.get("category_oracle_gap_by_selector", {}))
    for selector_name, selector_result in rule_results_by_name.items():
        selector_result = dict(selector_result)
        selector_result["overall_metrics"] = dict(selector_metrics_by_name.get(selector_name, {}))
        summary_row, selector_missing_rows, selector_extra_rows, reason_counts = _selector_explanation_rows(
            selector_name,
            selector_result,
            oracle_target_rows,
            rule_by_id,
            metric_by_rule_id,
        )
        selector_summary_rows.append(summary_row)
        for subset_name, gap_info in dict(category_oracle_gap_by_selector.get(selector_name, {})).items():
            subset_upper_bound = dict(category_upper_bounds.get(subset_name, {}))
            selector_category_gap_rows.append(
                {
                    "selector_name": selector_name,
                    "subset_name": subset_name,
                    "oracle_upper_bound_f1": _safe_float(gap_info.get("oracle_upper_bound_f1", 0.0)),
                    "selector_f1": _safe_float(gap_info.get("selector_f1", 0.0)),
                    "oracle_gap_f1": _safe_float(gap_info.get("oracle_gap_f1", 0.0)),
                    "precision": _safe_float(subset_upper_bound.get("precision", 0.0)),
                    "recall": _safe_float(subset_upper_bound.get("recall", 0.0)),
                    "fn_recovery_beyond_accepted_only": _safe_int(
                        subset_upper_bound.get("fn_recovery_beyond_accepted_only", 0)
                    ),
                    "fp_cost": _safe_int(subset_upper_bound.get("fp_cost", 0)),
                    "max_positive_coverage_count": _safe_int(
                        subset_upper_bound.get("max_positive_coverage_count", 0)
                    ),
                }
            )
        missing_rows.extend(selector_missing_rows[: max(1, _safe_int(cfg.get("max_missing_rules_per_selector", 20)))])
        extra_rows.extend(selector_extra_rows[: max(1, _safe_int(cfg.get("max_extra_rules_per_selector", 20)))])
        reason_counts_by_selector[selector_name] = dict(reason_counts)
        for reason_label, count in sorted(reason_counts.items()):
            reason_summary_rows.append(
                {
                    "selector_name": selector_name,
                    "reason_label": reason_label,
                    "count": int(count),
                }
            )

    _write_csv(
        oracle_target_path,
        [
            "oracle_rank",
            "rule_id",
            "clause",
            "semantic_family",
            "confidence",
            "train_positive_support",
            "train_negative_support",
            "added_positive_examples",
            "added_negative_examples",
            "precision",
            "recall",
            "f1",
        ],
        oracle_target_export_rows,
    )
    _write_csv(
        overlap_summary_path,
        [
            "selector_name",
            "selection_method",
            "num_selected_rules",
            "oracle_target_k",
            "num_overlap_with_oracle",
            "num_missed_oracle_rules",
            "num_extra_selected_rules",
            "oracle_overlap_fraction",
            "selector_f1",
            "selector_precision",
            "selector_recall",
            "primary_miss_reason",
        ],
        selector_summary_rows,
    )
    _write_csv(
        category_gap_path,
        [
            "selector_name",
            "subset_name",
            "oracle_upper_bound_f1",
            "selector_f1",
            "oracle_gap_f1",
            "precision",
            "recall",
            "fn_recovery_beyond_accepted_only",
            "fp_cost",
            "max_positive_coverage_count",
        ],
        selector_category_gap_rows,
    )
    _write_csv(
        missing_rules_path,
        [
            "selector_name",
            "oracle_rank",
            "rule_id",
            "clause",
            "semantic_family",
            "oracle_rule_confidence",
            "oracle_rule_train_positive_support",
            "oracle_rule_eval_precision",
            "oracle_rule_eval_recall",
            "oracle_rule_eval_f1",
            "reason_label",
            "reason_text",
            "rank_within_original_sort",
            "top_k_cutoff_rank",
            "cutoff_rule_id",
            "cutoff_rule_confidence",
            "cutoff_rule_positive_support",
            "best_chance_step",
            "oracle_rule_utility_at_best_chance",
            "chosen_rule_id_at_best_chance",
            "chosen_rule_utility_at_best_chance",
            "utility_gap_to_chosen_rule",
            "new_positive_gain_at_best_chance",
            "overlap_positive_count_at_best_chance",
            "family_reuse_count_at_best_chance",
            "base_quality_score_at_best_chance",
            "family_diversity_bonus_at_best_chance",
            "overlap_penalty_value_at_best_chance",
            "family_penalty_value_at_best_chance",
            "negative_support_penalty_value_at_best_chance",
            "semantic_match_level",
            "semantic_is_quota_qualified",
            "semantic_deficit_reduction_at_best_chance",
            "semantic_hard_constraint_active_at_best_chance",
            "semantic_total_remaining_deficits_at_best_chance",
        ],
        missing_rows,
    )
    _write_csv(
        extra_rules_path,
        [
            "selector_name",
            "selector_extra_rank",
            "rule_id",
            "clause",
            "semantic_family",
            "confidence",
            "train_positive_support",
            "eval_precision",
            "eval_recall",
            "eval_f1",
        ],
        extra_rows,
    )
    _write_csv(
        reason_summary_path,
        [
            "selector_name",
            "reason_label",
            "count",
        ],
        reason_summary_rows,
    )
    _write_markdown_report(
        report_path,
        oracle_target_row=oracle_target_row,
        selector_summary_rows=selector_summary_rows,
        selector_category_gap_rows=selector_category_gap_rows,
        reason_counts_by_selector=reason_counts_by_selector,
        missing_rows=missing_rows,
    )

    summary = {
        "version": _ORACLE_GAP_VERSION,
        "config": _cfg_key_subset(cfg),
        "oracle_target_mode": str(cfg.get("oracle_target_mode", "peak_f1")),
        "oracle_target_k": _safe_int(oracle_target_row.get("selection_rank", 0)),
        "oracle_target_f1": _safe_float(oracle_target_row.get("f1", 0.0)),
        "oracle_target_precision": _safe_float(oracle_target_row.get("precision", 0.0)),
        "oracle_target_recall": _safe_float(oracle_target_row.get("recall", 0.0)),
        "best_actual_selector_name": str(rule_pool_upper_bound_results.get("best_actual_selector_name", "")),
        "best_actual_selector_f1": _safe_float(rule_pool_upper_bound_results.get("best_actual_selector_f1", 0.0)),
        "selector_overlap_summary": selector_summary_rows,
        "selector_category_oracle_gap_summary": selector_category_gap_rows,
        "reason_counts_by_selector": reason_counts_by_selector,
        "output_paths": {
            "oracle_target_rule_set_csv": str(oracle_target_path),
            "selector_oracle_overlap_summary_csv": str(overlap_summary_path),
            "selector_category_oracle_gap_summary_csv": str(category_gap_path),
            "selector_missing_oracle_rules_csv": str(missing_rules_path),
            "selector_extra_selected_rules_csv": str(extra_rules_path),
            "selector_missing_reason_summary_csv": str(reason_summary_path),
            "oracle_selection_gap_report_md": str(report_path),
        },
    }

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        "  oracle_rule_selection_gap_diagnostic: "
        f"oracle_target_k={summary['oracle_target_k']} | "
        f"oracle_target_f1={summary['oracle_target_f1']:.3f} | "
        f"best_actual_selector={summary['best_actual_selector_name'] or 'unknown'} | "
        f"best_actual_f1={summary['best_actual_selector_f1']:.3f}"
    )
    print(f"Oracle selection gap summary written to {summary_path}")
    print(f"Oracle selection gap report written to {report_path}")
    return summary


def run(
    extended_rule_results: Dict[str, Any],
    rule_pool_upper_bound_results: Dict[str, Any],
    rule_results_by_name: Dict[str, Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_diagnostic(
        extended_rule_results=extended_rule_results,
        rule_pool_upper_bound_results=rule_pool_upper_bound_results,
        rule_results_by_name=rule_results_by_name,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
