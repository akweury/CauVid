"""
Diagnose whether held-out performance is limited by rule selection or by the
quality/coverage of the mined rule pool itself.

Outputs:
    pipeline_output/17d_driving_mini_rule_pool_upper_bound_diagnostic/
        best_single_rules_by_f1.csv
        best_rules_by_precision_at_min_recall.csv
        rule_pool_precision_recall_scatter.csv
        oracle_greedy_rule_set_curve.csv
        pool_upper_bound_summary.json
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config
from src.exp_driving_videos.modules.evaluate_rules_driving_mini import _find_rule_matches_for_example
from src.exp_driving_videos.modules.evaluate_rules_driving_mini import _get_rule_body_atom_templates


_POOL_UPPER_BOUND_VERSION = 1


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "17d_driving_mini_rule_pool_upper_bound_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "top_single_rules": int(cfg.get("top_single_rules", 100)),
        "precision_thresholds": [float(v) for v in cfg.get("precision_thresholds", [])],
        "f1_thresholds": [float(v) for v in cfg.get("f1_thresholds", [])],
        "min_recall_thresholds": [float(v) for v in cfg.get("min_recall_thresholds", [])],
        "oracle_k_values": [int(v) for v in cfg.get("oracle_k_values", [])],
        "selection_gap_threshold": float(cfg.get("selection_gap_threshold", 0.05)),
        "selection_pool_min_f1": float(cfg.get("selection_pool_min_f1", 0.35)),
        "low_pool_f1_threshold": float(cfg.get("low_pool_f1_threshold", 0.35)),
        "low_single_rule_f1_threshold": float(cfg.get("low_single_rule_f1_threshold", 0.2)),
        "high_precision_threshold": float(cfg.get("high_precision_threshold", 0.8)),
        "low_recall_threshold": float(cfg.get("low_recall_threshold", 0.1)),
        "high_recall_threshold": float(cfg.get("high_recall_threshold", 0.2)),
        "low_precision_threshold": float(cfg.get("low_precision_threshold", 0.5)),
        "vehicle_classes": sorted(str(v) for v in cfg.get("vehicle_classes", [])),
        "near_states": sorted(str(v) for v in cfg.get("near_states", [])),
        "center_states": sorted(str(v) for v in cfg.get("center_states", [])),
    }


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return dict(json.load(fh))


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


def _parse_atom(atom: str) -> Optional[Tuple[str, List[str]]]:
    text = str(atom).strip()
    match = re.match(r"^([a-z0-9_]+)\((.*)\)\.$", text)
    if not match:
        return None
    predicate = match.group(1)
    args_text = match.group(2).strip()
    if not args_text:
        return predicate, []
    return predicate, [part.strip() for part in args_text.split(",")]


def _iter_eval_examples(temporal_rule_results: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for video_result in temporal_rule_results:
        video_id = str(video_result.get("video_id", ""))
        for example in list(video_result.get("examples", [])):
            row = dict(example)
            row["video_id"] = video_id
            yield row


def _compute_binary_metrics(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> Dict[str, float | int]:
    precision = float(true_positive / max(1, true_positive + false_positive))
    recall = float(true_positive / max(1, true_positive + false_negative))
    f1 = float(2 * precision * recall / max(1e-12, precision + recall))
    accuracy = float((true_positive + true_negative) / max(1, true_positive + false_positive + false_negative + true_negative))
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _rule_semantic_family(rule: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    vehicle_classes = {
        str(v)
        for v in cfg.get("vehicle_classes", ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"])
    }
    near_states = {str(v) for v in cfg.get("near_states", ["near"])}
    center_states = {str(v) for v in cfg.get("center_states", ["centered"])}

    has_vehicle = False
    has_near = False
    has_centered = False
    has_generic_motion_object = False

    for atom in _get_rule_body_atom_templates(rule):
        parsed = _parse_atom(atom)
        if parsed is None:
            continue
        predicate, args = parsed
        if predicate == "object_class" and len(args) >= 3 and str(args[2]) in vehicle_classes:
            has_vehicle = True
        elif predicate == "object_distance_state" and len(args) >= 3 and str(args[2]) in near_states:
            has_near = True
        elif predicate == "object_x_position_state" and len(args) >= 3 and str(args[2]) in center_states:
            has_centered = True
        elif predicate.startswith("object_") or predicate.startswith("segment_"):
            has_generic_motion_object = True

    if has_vehicle and has_near and has_centered:
        return "vehicle+near+centered"
    if has_vehicle and has_near:
        return "vehicle+near"
    if has_vehicle and has_centered:
        return "vehicle+centered"
    if has_near and has_centered:
        return "near+centered"
    if has_vehicle:
        return "vehicle"
    if has_near:
        return "near"
    if has_centered:
        return "centered"
    if has_generic_motion_object:
        return "generic motion/object"
    return "other"


def _evaluate_selected_rule_set(
    selected_rule_ids: Sequence[str],
    rule_matches_by_id: Dict[str, Dict[str, Set[str]]],
    positive_example_ids: Set[str],
    negative_example_ids: Set[str],
) -> Dict[str, Any]:
    predicted_positive_ids: Set[str] = set()
    for rule_id in selected_rule_ids:
        match_info = rule_matches_by_id.get(str(rule_id), {})
        predicted_positive_ids.update(match_info.get("matched_example_ids", set()))

    tp = len(predicted_positive_ids & positive_example_ids)
    fp = len(predicted_positive_ids & negative_example_ids)
    fn = len(positive_example_ids - predicted_positive_ids)
    tn = len(negative_example_ids - predicted_positive_ids)
    metrics = _compute_binary_metrics(tp, fp, fn, tn)
    metrics["num_rules"] = len([rule_id for rule_id in selected_rule_ids if str(rule_id) in rule_matches_by_id])
    metrics["predicted_positive_example_ids"] = sorted(predicted_positive_ids)
    return metrics


def _count_thresholds(rule_rows: Sequence[Dict[str, Any]], field_name: str, thresholds: Sequence[float]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for threshold in thresholds:
        counts[f"{float(threshold):.3f}"] = sum(1 for row in rule_rows if float(row.get(field_name, 0.0)) >= float(threshold))
    return counts


def _diagnose_bottleneck(
    best_single_rule_f1: float,
    oracle_f1_at_reference_k: float,
    best_actual_selector_name: str,
    best_actual_selector_f1: float,
    selection_reference_k: int,
    cfg: Dict[str, Any],
) -> Tuple[str, str]:
    selection_gap = oracle_f1_at_reference_k - best_actual_selector_f1
    if (
        selection_reference_k > 0
        and oracle_f1_at_reference_k >= float(cfg.get("selection_pool_min_f1", 0.35))
        and selection_gap >= float(cfg.get("selection_gap_threshold", 0.05))
    ):
        return (
            "rule_selection",
            "The held-out oracle greedy rule set materially outperforms the best selected rule set at the same K, "
            "so selection appears to leave usable pool quality on the table.",
        )
    if (
        oracle_f1_at_reference_k < float(cfg.get("low_pool_f1_threshold", 0.35))
        and best_single_rule_f1 < float(cfg.get("low_single_rule_f1_threshold", 0.2))
    ):
        return (
            "insufficient_rule_quality_in_pool",
            "Even the best single rule and oracle greedy subsets remain weak on held-out data, suggesting the pool lacks "
            "sufficiently strong rules rather than merely selecting the wrong ones.",
        )
    return (
        "mixed_or_inconclusive",
        f"The best selected set is {best_actual_selector_name} with F1={best_actual_selector_f1:.3f}, while the oracle upper bound "
        f"at K={selection_reference_k} is {oracle_f1_at_reference_k:.3f}; this does not cleanly isolate selection from pool quality.",
    )


def process_diagnostic(
    extended_rule_results: Dict[str, Any],
    temporal_rule_results: List[Dict[str, Any]],
    eval_video_ids: Optional[List[str]] = None,
    split_manifest: Optional[Dict[str, Any]] = None,
    rule_results_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)

    best_single_path = out_root / "best_single_rules_by_f1.csv"
    precision_recall_path = out_root / "best_rules_by_precision_at_min_recall.csv"
    scatter_path = out_root / "rule_pool_precision_recall_scatter.csv"
    oracle_curve_path = out_root / "oracle_greedy_rule_set_curve.csv"
    summary_path = out_root / "pool_upper_bound_summary.json"

    if not force_recompute and summary_path.exists():
        cached = _read_json(summary_path)
        if int(cached.get("version", 0)) == _POOL_UPPER_BOUND_VERSION and _cfg_key_subset(cached.get("config", {})) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {summary_path.name}")
            return cached

    eval_video_id_set = {str(video_id) for video_id in (eval_video_ids or [])}
    filtered_results = [
        result
        for result in temporal_rule_results
        if not eval_video_id_set or str(result.get("video_id", "")) in eval_video_id_set
    ]

    eval_examples: List[Dict[str, Any]] = []
    positive_example_ids: Set[str] = set()
    negative_example_ids: Set[str] = set()
    for example in _iter_eval_examples(filtered_results):
        example_id = str(example.get("example_id", ""))
        if not example_id:
            continue
        eval_examples.append(
            {
                "video_id": str(example.get("video_id", "")),
                "example_id": example_id,
                "label": bool(example.get("label", False)),
                "body_atoms": list(example.get("body_atoms", [])),
            }
        )
        if bool(example.get("label", False)):
            positive_example_ids.add(example_id)
        else:
            negative_example_ids.add(example_id)

    total_positive_examples = len(positive_example_ids)
    total_negative_examples = len(negative_example_ids)
    all_kept_rules = list(extended_rule_results.get("all_kept_rules", []))
    oracle_k_values = sorted({max(1, int(v)) for v in cfg.get("oracle_k_values", [1, 5, 10, 20, 50, 100])})
    max_oracle_k = max(oracle_k_values, default=1)

    rule_rows: List[Dict[str, Any]] = []
    rule_matches_by_id: Dict[str, Dict[str, Set[str]]] = {}

    for rule in all_kept_rules:
        rule_id = str(rule.get("rule_id", ""))
        matched_positive_ids: Set[str] = set()
        matched_negative_ids: Set[str] = set()
        body_atom_templates = _get_rule_body_atom_templates(rule)
        for example in eval_examples:
            match_states = _find_rule_matches_for_example(
                body_atom_templates=body_atom_templates,
                body_atoms=list(example.get("body_atoms", [])),
            )
            if not match_states:
                continue
            example_id = str(example.get("example_id", ""))
            if bool(example.get("label", False)):
                matched_positive_ids.add(example_id)
            else:
                matched_negative_ids.add(example_id)

        tp = len(matched_positive_ids)
        fp = len(matched_negative_ids)
        fn = total_positive_examples - tp
        tn = total_negative_examples - fp
        metrics = _compute_binary_metrics(tp, fp, fn, tn)
        semantic_family = _rule_semantic_family(rule, cfg)
        matched_example_ids = matched_positive_ids | matched_negative_ids
        rule_matches_by_id[rule_id] = {
            "positive_example_ids": set(matched_positive_ids),
            "negative_example_ids": set(matched_negative_ids),
            "matched_example_ids": set(matched_example_ids),
        }
        rule_rows.append(
            {
                "rule_id": rule_id,
                "clause": str(rule.get("clause", "")),
                "confidence": float(rule.get("confidence", 0.0)),
                "train_positive_support": int(rule.get("positive_support", 0)),
                "train_negative_support": int(rule.get("negative_support", 0)),
                "train_total_support": int(rule.get("total_support", int(rule.get("positive_support", 0)) + int(rule.get("negative_support", 0)))),
                "eval_positive_support": tp,
                "eval_negative_support": fp,
                "eval_total_support": len(matched_example_ids),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "accuracy": float(metrics["accuracy"]),
                "semantic_family": semantic_family,
            }
        )

    sorted_by_f1 = sorted(
        rule_rows,
        key=lambda row: (
            -float(row.get("f1", 0.0)),
            -float(row.get("precision", 0.0)),
            -float(row.get("recall", 0.0)),
            -float(row.get("confidence", 0.0)),
            str(row.get("rule_id", "")),
        ),
    )
    top_single_rules = sorted_by_f1[: max(1, int(cfg.get("top_single_rules", 100)))]

    min_recall_thresholds = [float(v) for v in cfg.get("min_recall_thresholds", [0.02, 0.05, 0.1, 0.15, 0.2, 0.3])]
    best_precision_rows: List[Dict[str, Any]] = []
    for threshold in min_recall_thresholds:
        eligible_rows = [row for row in rule_rows if float(row.get("recall", 0.0)) >= threshold]
        if not eligible_rows:
            best_precision_rows.append(
                {
                    "min_recall_threshold": threshold,
                    "rule_id": "",
                    "clause": "",
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "eval_positive_support": "",
                    "eval_negative_support": "",
                    "semantic_family": "",
                    "confidence": "",
                }
            )
            continue
        best_row = max(
            eligible_rows,
            key=lambda row: (
                float(row.get("precision", 0.0)),
                float(row.get("f1", 0.0)),
                float(row.get("recall", 0.0)),
                float(row.get("confidence", 0.0)),
                -int(row.get("eval_negative_support", 0)),
                str(row.get("rule_id", "")),
            ),
        )
        best_precision_rows.append(
            {
                "min_recall_threshold": threshold,
                "rule_id": str(best_row.get("rule_id", "")),
                "clause": str(best_row.get("clause", "")),
                "precision": float(best_row.get("precision", 0.0)),
                "recall": float(best_row.get("recall", 0.0)),
                "f1": float(best_row.get("f1", 0.0)),
                "eval_positive_support": int(best_row.get("eval_positive_support", 0)),
                "eval_negative_support": int(best_row.get("eval_negative_support", 0)),
                "semantic_family": str(best_row.get("semantic_family", "")),
                "confidence": float(best_row.get("confidence", 0.0)),
            }
        )

    oracle_curve_rows: List[Dict[str, Any]] = []
    selected_rule_ids: Set[str] = set()
    predicted_positive_ids: Set[str] = set()
    current_tp = 0
    current_fp = 0

    for rank in range(1, min(max_oracle_k, len(rule_rows)) + 1):
        best_candidate: Optional[Dict[str, Any]] = None
        best_candidate_metrics: Optional[Dict[str, Any]] = None
        for row in rule_rows:
            rule_id = str(row.get("rule_id", ""))
            if rule_id in selected_rule_ids:
                continue
            match_info = rule_matches_by_id.get(rule_id, {})
            candidate_pos_ids = set(match_info.get("positive_example_ids", set()))
            candidate_neg_ids = set(match_info.get("negative_example_ids", set()))
            additional_tp = len(candidate_pos_ids - predicted_positive_ids)
            additional_fp = len(candidate_neg_ids - predicted_positive_ids)
            tp = current_tp + additional_tp
            fp = current_fp + additional_fp
            fn = total_positive_examples - tp
            tn = total_negative_examples - fp
            metrics = _compute_binary_metrics(tp, fp, fn, tn)
            candidate = {
                "rule_id": rule_id,
                "clause": str(row.get("clause", "")),
                "semantic_family": str(row.get("semantic_family", "")),
                "added_positive_examples": additional_tp,
                "added_negative_examples": additional_fp,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "true_positive": int(metrics["true_positive"]),
                "false_positive": int(metrics["false_positive"]),
                "false_negative": int(metrics["false_negative"]),
                "true_negative": int(metrics["true_negative"]),
                "confidence": float(row.get("confidence", 0.0)),
            }
            key = (
                float(candidate["f1"]),
                float(candidate["recall"]),
                float(candidate["precision"]),
                int(candidate["added_positive_examples"]) - int(candidate["added_negative_examples"]),
                float(candidate["confidence"]),
                str(candidate["rule_id"]),
            )
            best_key = None
            if best_candidate is not None:
                best_key = (
                    float(best_candidate["f1"]),
                    float(best_candidate["recall"]),
                    float(best_candidate["precision"]),
                    int(best_candidate["added_positive_examples"]) - int(best_candidate["added_negative_examples"]),
                    float(best_candidate["confidence"]),
                    str(best_candidate["rule_id"]),
                )
            if best_candidate is None or key > best_key:
                best_candidate = candidate
                best_candidate_metrics = match_info

        if best_candidate is None or best_candidate_metrics is None:
            break

        best_rule_id = str(best_candidate["rule_id"])
        selected_rule_ids.add(best_rule_id)
        predicted_positive_ids.update(best_candidate_metrics.get("positive_example_ids", set()))
        predicted_positive_ids.update(best_candidate_metrics.get("negative_example_ids", set()))
        current_tp = int(best_candidate["true_positive"])
        current_fp = int(best_candidate["false_positive"])

        oracle_curve_rows.append(
            {
                "selection_rank": rank,
                "rule_id": best_rule_id,
                "clause": str(best_candidate["clause"]),
                "semantic_family": str(best_candidate["semantic_family"]),
                "added_positive_examples": int(best_candidate["added_positive_examples"]),
                "added_negative_examples": int(best_candidate["added_negative_examples"]),
                "cumulative_true_positive": int(best_candidate["true_positive"]),
                "cumulative_false_positive": int(best_candidate["false_positive"]),
                "cumulative_false_negative": int(best_candidate["false_negative"]),
                "cumulative_true_negative": int(best_candidate["true_negative"]),
                "precision": float(best_candidate["precision"]),
                "recall": float(best_candidate["recall"]),
                "f1": float(best_candidate["f1"]),
                "is_reported_k": rank in oracle_k_values,
            }
        )

    oracle_curve_by_k = {int(row["selection_rank"]): row for row in oracle_curve_rows}
    oracle_f1_by_k: Dict[str, float] = {}
    for k in oracle_k_values:
        row = oracle_curve_by_k.get(k)
        oracle_f1_by_k[str(k)] = float(row.get("f1", 0.0)) if row is not None else 0.0

    selector_metrics_by_name: Dict[str, Dict[str, Any]] = {}
    if rule_results_by_name:
        for selector_name, selector_result in rule_results_by_name.items():
            selected_ids = [str(rule.get("rule_id", "")) for rule in list(selector_result.get("final_rules", []))]
            selector_metrics_by_name[selector_name] = _evaluate_selected_rule_set(
                selected_rule_ids=selected_ids,
                rule_matches_by_id=rule_matches_by_id,
                positive_example_ids=positive_example_ids,
                negative_example_ids=negative_example_ids,
            )

    selection_reference_k = max((int(metrics.get("num_rules", 0)) for metrics in selector_metrics_by_name.values()), default=max(oracle_k_values, default=0))
    best_actual_selector_name = ""
    best_actual_selector_f1 = 0.0
    if selector_metrics_by_name:
        best_actual_selector_name = max(
            selector_metrics_by_name,
            key=lambda name: (
                float(selector_metrics_by_name[name].get("f1", 0.0)),
                float(selector_metrics_by_name[name].get("recall", 0.0)),
                float(selector_metrics_by_name[name].get("precision", 0.0)),
                -int(selector_metrics_by_name[name].get("num_rules", 0)),
                str(name),
            ),
        )
        best_actual_selector_f1 = float(selector_metrics_by_name[best_actual_selector_name].get("f1", 0.0))

    oracle_reference_row = oracle_curve_by_k.get(selection_reference_k)
    if oracle_reference_row is None and oracle_curve_rows:
        fallback_rank = max((int(row["selection_rank"]) for row in oracle_curve_rows if int(row["selection_rank"]) <= selection_reference_k), default=0)
        if fallback_rank > 0:
            oracle_reference_row = oracle_curve_by_k.get(fallback_rank)
    oracle_f1_at_reference_k = float(oracle_reference_row.get("f1", 0.0)) if oracle_reference_row is not None else 0.0
    best_single_rule_f1 = float(sorted_by_f1[0].get("f1", 0.0)) if sorted_by_f1 else 0.0
    bottleneck_label, bottleneck_rationale = _diagnose_bottleneck(
        best_single_rule_f1=best_single_rule_f1,
        oracle_f1_at_reference_k=oracle_f1_at_reference_k,
        best_actual_selector_name=best_actual_selector_name,
        best_actual_selector_f1=best_actual_selector_f1,
        selection_reference_k=selection_reference_k,
        cfg=cfg,
    )

    _write_csv(
        best_single_path,
        [
            "rule_id",
            "clause",
            "confidence",
            "train_positive_support",
            "train_negative_support",
            "train_total_support",
            "eval_positive_support",
            "eval_negative_support",
            "eval_total_support",
            "precision",
            "recall",
            "f1",
            "semantic_family",
        ],
        top_single_rules,
    )
    _write_csv(
        precision_recall_path,
        [
            "min_recall_threshold",
            "rule_id",
            "clause",
            "precision",
            "recall",
            "f1",
            "eval_positive_support",
            "eval_negative_support",
            "semantic_family",
            "confidence",
        ],
        best_precision_rows,
    )
    _write_csv(
        scatter_path,
        [
            "rule_id",
            "clause",
            "confidence",
            "train_positive_support",
            "train_negative_support",
            "train_total_support",
            "eval_positive_support",
            "eval_negative_support",
            "eval_total_support",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "semantic_family",
        ],
        sorted_by_f1,
    )
    _write_csv(
        oracle_curve_path,
        [
            "selection_rank",
            "rule_id",
            "clause",
            "semantic_family",
            "added_positive_examples",
            "added_negative_examples",
            "cumulative_true_positive",
            "cumulative_false_positive",
            "cumulative_false_negative",
            "cumulative_true_negative",
            "precision",
            "recall",
            "f1",
            "is_reported_k",
        ],
        oracle_curve_rows,
    )

    summary: Dict[str, Any] = {
        "version": _POOL_UPPER_BOUND_VERSION,
        "config": _cfg_key_subset(cfg),
        "split": split_manifest or {},
        "num_pool_rules": len(rule_rows),
        "num_eval_videos": len(filtered_results),
        "num_eval_examples": len(eval_examples),
        "num_eval_positive_examples": total_positive_examples,
        "num_eval_negative_examples": total_negative_examples,
        "num_rules_above_precision_thresholds": _count_thresholds(rule_rows, "precision", cfg.get("precision_thresholds", [0.5, 0.7, 0.9])),
        "num_rules_above_f1_thresholds": _count_thresholds(rule_rows, "f1", cfg.get("f1_thresholds", [0.1, 0.2, 0.3, 0.4])),
        "narrow_high_precision_low_recall_rule_count": sum(
            1
            for row in rule_rows
            if float(row.get("precision", 0.0)) >= float(cfg.get("high_precision_threshold", 0.8))
            and float(row.get("recall", 0.0)) < float(cfg.get("low_recall_threshold", 0.1))
        ),
        "broad_high_recall_low_precision_rule_count": sum(
            1
            for row in rule_rows
            if float(row.get("recall", 0.0)) >= float(cfg.get("high_recall_threshold", 0.2))
            and float(row.get("precision", 0.0)) < float(cfg.get("low_precision_threshold", 0.5))
        ),
        "best_single_rule": dict(top_single_rules[0]) if top_single_rules else {},
        "best_single_rule_f1": best_single_rule_f1,
        "best_oracle_greedy_rule_set_f1_by_k": oracle_f1_by_k,
        "actual_selected_rule_set_metrics_by_name": selector_metrics_by_name,
        "best_actual_selector_name": best_actual_selector_name,
        "best_actual_selector_f1": best_actual_selector_f1,
        "selection_reference_k": selection_reference_k,
        "oracle_f1_at_selection_reference_k": oracle_f1_at_reference_k,
        "bottleneck_label": bottleneck_label,
        "bottleneck_rationale": bottleneck_rationale,
        "output_paths": {
            "best_single_rules_by_f1_csv": str(best_single_path),
            "best_rules_by_precision_at_min_recall_csv": str(precision_recall_path),
            "rule_pool_precision_recall_scatter_csv": str(scatter_path),
            "oracle_greedy_rule_set_curve_csv": str(oracle_curve_path),
        },
    }

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        "  rule_pool_upper_bound_diagnostic: "
        f"pool_rules={len(rule_rows)} | "
        f"best_single_f1={best_single_rule_f1:.3f} | "
        f"oracle_f1_at_k={oracle_f1_at_reference_k:.3f} | "
        f"bottleneck={bottleneck_label}"
    )
    print(f"Rule-pool scatter CSV written to {scatter_path}")
    print(f"Oracle greedy curve CSV written to {oracle_curve_path}")
    print(f"Pool upper-bound summary written to {summary_path}")
    return summary


def run(
    extended_rule_results: Dict[str, Any],
    temporal_rule_results: List[Dict[str, Any]],
    eval_video_ids: Optional[List[str]] = None,
    split_manifest: Optional[Dict[str, Any]] = None,
    rule_results_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_diagnostic(
        extended_rule_results=extended_rule_results,
        temporal_rule_results=temporal_rule_results,
        eval_video_ids=eval_video_ids,
        split_manifest=split_manifest,
        rule_results_by_name=rule_results_by_name,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
