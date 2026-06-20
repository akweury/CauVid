"""
Train a learned rule-aggregation baseline over Step 16 rule firings.

The baseline builds a binary rule-firing matrix from the Step 16 rule pool,
trains an L1-sparse logistic regression on train videos, tunes regularization
and decision threshold on validation videos, and evaluates on the held-out
evaluation split.

Output layout:
    pipeline_output/18c_driving_mini_rule_aggregation_baseline/
        rule_aggregation_baseline_summary.json
        rule_aggregation_baseline_metrics.csv
        per_video_metrics.csv
        prediction_examples.csv
        top_weighted_rules_with_clauses.csv
        rule_aggregation_ablation_metrics.csv
        rule_aggregation_subset_threshold_metrics.csv
        top_k_weighted_rule_sets.csv
        rule_aggregation_family_summary.json
        rule_aggregation_family_summary.csv
        split_leakage_check.json
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config
from src.exp_driving_videos.modules.rule_pool_upper_bound_diagnostic_driving_mini import (
    _compute_binary_metrics,
    _parse_atom,
    _rule_matches_example_fast,
    _rule_semantic_family,
)
from src.exp_driving_videos.modules.evaluate_rules_driving_mini import _get_rule_body_atom_templates


_RULE_AGGREGATION_BASELINE_VERSION = 4
_TOP_WEIGHT_ABLATION_SIZES: Tuple[Any, ...] = (1, 3, 5, 10, 20, 50, "all_nonzero")
_SUMMARY_FAMILY_ORDER: Tuple[str, ...] = (
    "transition",
    "ego-motion",
    "vehicle-centered",
    "near/centered object",
    "suppressor",
    "other",
)


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "18c_driving_mini_rule_aggregation_baseline"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "validation_fraction": float(cfg.get("validation_fraction", 0.25)),
        "class_weight": str(cfg.get("class_weight", "balanced")),
        "c_values": [float(v) for v in cfg.get("c_values", [])],
        "solver": str(cfg.get("solver", "liblinear")),
        "max_iter": int(cfg.get("max_iter", 2000)),
        "random_seed": int(cfg.get("random_seed", 0)),
        "active_rule_min_train_support": int(cfg.get("active_rule_min_train_support", 1)),
        "top_weighted_rules": int(cfg.get("top_weighted_rules", 30)),
        "vehicle_classes": sorted(str(v) for v in cfg.get("vehicle_classes", [])),
        "near_states": sorted(str(v) for v in cfg.get("near_states", [])),
        "center_states": sorted(str(v) for v in cfg.get("center_states", [])),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _prepare_examples(temporal_rule_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for video_result in temporal_rule_results:
        video_id = str(video_result.get("video_id", ""))
        for example in list(video_result.get("examples", [])):
            example_id = str(example.get("example_id", ""))
            if not example_id:
                continue
            predicate_counts: Dict[str, int] = {}
            atoms_by_predicate: Dict[str, List[Tuple[str, ...]]] = {}
            for atom in list(example.get("body_atoms", [])):
                parsed = _parse_atom(str(atom))
                if parsed is None:
                    continue
                predicate, args = parsed
                predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
                atoms_by_predicate.setdefault(predicate, []).append(tuple(str(arg) for arg in args))
            examples.append(
                {
                    "video_id": video_id,
                    "example_id": example_id,
                    "label": bool(example.get("label", False)),
                    "current_segment_index": int(example.get("current_segment_index", -1)),
                    "current_segment_label": str(example.get("current_segment_label", "")),
                    "num_body_atoms": int(example.get("num_body_atoms", len(example.get("body_atoms", [])))),
                    "predicate_counts": predicate_counts,
                    "atoms_by_predicate": atoms_by_predicate,
                }
            )
    return examples


def _group_examples_by_video(examples: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for example in examples:
        grouped.setdefault(str(example.get("video_id", "")), []).append(example)
    for video_id in grouped:
        grouped[video_id] = sorted(
            grouped[video_id],
            key=lambda row: (
                int(row.get("current_segment_index", -1)),
                str(row.get("example_id", "")),
            ),
        )
    return grouped


def _split_train_validation(
    train_examples: Sequence[Dict[str, Any]],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    grouped = _group_examples_by_video(train_examples)
    video_ids = sorted(grouped)
    rng = random.Random(int(seed))
    shuffled_video_ids = list(video_ids)
    rng.shuffle(shuffled_video_ids)

    if len(shuffled_video_ids) > 1:
        requested = int(round(len(shuffled_video_ids) * float(validation_fraction)))
        val_video_count = min(len(shuffled_video_ids) - 1, max(1, requested))
        val_video_ids = set(shuffled_video_ids[:val_video_count])
        train_subset = [example for example in train_examples if str(example.get("video_id", "")) not in val_video_ids]
        val_subset = [example for example in train_examples if str(example.get("video_id", "")) in val_video_ids]
        return train_subset, val_subset, {
            "strategy": "video_level",
            "train_video_ids": sorted({str(example.get("video_id", "")) for example in train_subset}),
            "validation_video_ids": sorted(val_video_ids),
            "num_train_examples": len(train_subset),
            "num_validation_examples": len(val_subset),
        }

    ordered = sorted(
        train_examples,
        key=lambda row: (
            int(row.get("current_segment_index", -1)),
            str(row.get("example_id", "")),
        ),
    )
    if len(ordered) <= 1:
        return list(ordered), list(ordered), {
            "strategy": "single_example_fallback",
            "train_video_ids": list(video_ids),
            "validation_video_ids": list(video_ids),
            "num_train_examples": len(ordered),
            "num_validation_examples": len(ordered),
        }
    val_count = min(len(ordered) - 1, max(1, int(round(len(ordered) * float(validation_fraction)))))
    train_subset = ordered[:-val_count]
    val_subset = ordered[-val_count:]
    return train_subset, val_subset, {
        "strategy": "example_level_fallback",
        "train_video_ids": list(video_ids),
        "validation_video_ids": list(video_ids),
        "num_train_examples": len(train_subset),
        "num_validation_examples": len(val_subset),
    }


def _prepare_rules(extended_rule_results: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    for rule in list(extended_rule_results.get("all_kept_rules", [])):
        rule_id = str(rule.get("rule_id", ""))
        if not rule_id:
            continue
        templates: List[Tuple[str, Tuple[str, ...]]] = []
        predicate_counts: Dict[str, int] = {}
        for atom in _get_rule_body_atom_templates(rule):
            parsed = _parse_atom(atom)
            if parsed is None:
                continue
            predicate, args = parsed
            templates.append((predicate, tuple(str(arg) for arg in args)))
            predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
        if not templates:
            continue
        rules.append(
            {
                "rule_id": rule_id,
                "clause": str(rule.get("clause", "")),
                "confidence": float(rule.get("confidence", 0.0)),
                "positive_support": int(rule.get("positive_support", 0)),
                "negative_support": int(rule.get("negative_support", 0)),
                "total_support": int(rule.get("total_support", 0)),
                "semantic_family": _rule_semantic_family(rule, cfg),
                "rule_templates": templates,
                "rule_predicate_counts": predicate_counts,
            }
        )
    return rules


def _import_ml_dependencies() -> Tuple[Any, Any, Any, Any, Any]:
    try:
        from scipy import sparse
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "Rule aggregation baseline requires scipy and scikit-learn in the runtime environment."
        ) from exc
    return sparse, LogisticRegression, average_precision_score, roc_auc_score, confusion_matrix


def _build_rule_firing_matrix(
    examples: Sequence[Dict[str, Any]],
    prepared_rules: Sequence[Dict[str, Any]],
    sparse_module: Any,
) -> Any:
    row_indices: List[int] = []
    col_indices: List[int] = []
    data: List[int] = []
    for col_index, rule in enumerate(prepared_rules):
        rule_templates = list(rule.get("rule_templates", []))
        rule_predicate_counts = dict(rule.get("rule_predicate_counts", {}))
        for row_index, example in enumerate(examples):
            if _rule_matches_example_fast(
                rule_templates=rule_templates,
                rule_predicate_counts=rule_predicate_counts,
                example_atoms_by_predicate=dict(example.get("atoms_by_predicate", {})),
                example_predicate_counts=dict(example.get("predicate_counts", {})),
            ):
                row_indices.append(row_index)
                col_indices.append(col_index)
                data.append(1)
    return sparse_module.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(examples), len(prepared_rules)),
        dtype=np.float32,
    )


def _compute_threshold_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
    average_precision_score_fn: Any,
    roc_auc_score_fn: Any,
    confusion_matrix_fn: Any,
) -> Dict[str, Any]:
    predicted = [1 if float(probability) >= float(threshold) else 0 for probability in probabilities]
    confusion = confusion_matrix_fn(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in confusion.ravel()]
    metrics = _compute_binary_metrics(tp, fp, fn, tn)
    try:
        auroc = float(roc_auc_score_fn(labels, probabilities))
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score_fn(labels, probabilities))
    except Exception:
        auprc = float("nan")
    metrics["auroc"] = auroc
    metrics["auprc"] = auprc
    metrics["threshold"] = float(threshold)
    metrics["predicted_labels"] = predicted
    return metrics


def _select_best_threshold(
    labels: Sequence[int],
    probabilities: Sequence[float],
    average_precision_score_fn: Any,
    roc_auc_score_fn: Any,
    confusion_matrix_fn: Any,
    default_threshold: float = 0.5,
) -> Tuple[float, Dict[str, Any]]:
    candidates = {0.0, 1.0, float(default_threshold)}
    for probability in probabilities:
        p = float(probability)
        candidates.add(p)
        candidates.add(min(1.0, max(0.0, p + 1e-8)))
        candidates.add(min(1.0, max(0.0, p - 1e-8)))

    best_threshold = float(default_threshold)
    best_metrics = _compute_threshold_metrics(
        labels,
        probabilities,
        best_threshold,
        average_precision_score_fn,
        roc_auc_score_fn,
        confusion_matrix_fn,
    )
    best_key = (
        float(best_metrics.get("f1", 0.0)),
        float(best_metrics.get("precision", 0.0)),
        float(best_metrics.get("recall", 0.0)),
        -abs(best_threshold - default_threshold),
        -best_threshold,
    )
    for threshold in sorted(candidates):
        metrics = _compute_threshold_metrics(
            labels,
            probabilities,
            float(threshold),
            average_precision_score_fn,
            roc_auc_score_fn,
            confusion_matrix_fn,
        )
        key = (
            float(metrics.get("f1", 0.0)),
            float(metrics.get("precision", 0.0)),
            float(metrics.get("recall", 0.0)),
            -abs(float(threshold) - default_threshold),
            -float(threshold),
        )
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


def _per_video_metrics(
    model_name: str,
    examples: Sequence[Dict[str, Any]],
    probabilities: Sequence[float],
    threshold_name: str,
    threshold_value: float,
    average_precision_score_fn: Any,
    roc_auc_score_fn: Any,
    confusion_matrix_fn: Any,
) -> List[Dict[str, Any]]:
    grouped = _group_examples_by_video(examples)
    probability_lookup = {
        str(example.get("example_id", "")): float(probability)
        for example, probability in zip(examples, probabilities)
    }
    rows: List[Dict[str, Any]] = []
    for video_id in sorted(grouped):
        video_examples = grouped[video_id]
        labels = [1 if bool(example.get("label", False)) else 0 for example in video_examples]
        video_probabilities = [probability_lookup[str(example.get("example_id", ""))] for example in video_examples]
        metrics = _compute_threshold_metrics(
            labels,
            video_probabilities,
            threshold_value,
            average_precision_score_fn,
            roc_auc_score_fn,
            confusion_matrix_fn,
        )
        rows.append(
            {
                "model_name": model_name,
                "video_id": video_id,
                "num_examples": len(video_examples),
                "threshold_name": threshold_name,
                "threshold_value": float(threshold_value),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "auroc": float(metrics.get("auroc", float("nan"))),
                "auprc": float(metrics.get("auprc", float("nan"))),
                "true_positive": int(metrics.get("true_positive", 0)),
                "false_positive": int(metrics.get("false_positive", 0)),
                "false_negative": int(metrics.get("false_negative", 0)),
                "true_negative": int(metrics.get("true_negative", 0)),
            }
        )
    return rows


def _prediction_rows(
    examples: Sequence[Dict[str, Any]],
    probabilities: Sequence[float],
    threshold_05: float,
    best_threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for example, probability in zip(examples, probabilities):
        rows.append(
            {
                "video_id": str(example.get("video_id", "")),
                "example_id": str(example.get("example_id", "")),
                "current_segment_index": int(example.get("current_segment_index", -1)),
                "label": bool(example.get("label", False)),
                "predicted_probability": float(probability),
                "threshold_at_0_5": float(threshold_05),
                "best_validation_threshold": float(best_threshold),
                "predicted_label_at_0_5": bool(float(probability) >= float(threshold_05)),
                "predicted_label_at_best_validation_threshold": bool(float(probability) >= float(best_threshold)),
                "current_segment_label": str(example.get("current_segment_label", "")),
                "num_body_atoms": int(example.get("num_body_atoms", 0)),
            }
        )
    return rows


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _split_separation_checks(
    train_examples_all: Sequence[Dict[str, Any]],
    train_examples: Sequence[Dict[str, Any]],
    val_examples: Sequence[Dict[str, Any]],
    eval_examples: Sequence[Dict[str, Any]],
    validation_split: Dict[str, Any],
    split_manifest: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    train_all_video_ids = {str(example.get("video_id", "")) for example in train_examples_all}
    train_video_ids = {str(example.get("video_id", "")) for example in train_examples}
    validation_video_ids = {str(example.get("video_id", "")) for example in val_examples}
    eval_video_ids = {str(example.get("video_id", "")) for example in eval_examples}

    train_example_ids = {str(example.get("example_id", "")) for example in train_examples}
    validation_example_ids = {str(example.get("example_id", "")) for example in val_examples}
    eval_example_ids = {str(example.get("example_id", "")) for example in eval_examples}

    train_manifest_video_ids = {str(value) for value in list((split_manifest or {}).get("train_video_ids", []))}
    eval_manifest_video_ids = {str(value) for value in list((split_manifest or {}).get("eval_video_ids", []))}

    train_eval_video_overlap = sorted(train_all_video_ids & eval_video_ids)
    validation_eval_video_overlap = sorted(validation_video_ids & eval_video_ids)
    train_validation_video_overlap = sorted(train_video_ids & validation_video_ids)
    train_eval_example_overlap = sorted(train_example_ids & eval_example_ids)
    validation_eval_example_overlap = sorted(validation_example_ids & eval_example_ids)
    train_validation_example_overlap = sorted(train_example_ids & validation_example_ids)

    return {
        "validation_strategy": str(validation_split.get("strategy", "")),
        "train_manifest_video_count": len(train_manifest_video_ids),
        "eval_manifest_video_count": len(eval_manifest_video_ids),
        "train_video_count_all": len(train_all_video_ids),
        "train_video_count_after_split": len(train_video_ids),
        "validation_video_count": len(validation_video_ids),
        "eval_video_count": len(eval_video_ids),
        "train_eval_video_overlap_count": len(train_eval_video_overlap),
        "validation_eval_video_overlap_count": len(validation_eval_video_overlap),
        "train_validation_video_overlap_count": len(train_validation_video_overlap),
        "train_eval_example_overlap_count": len(train_eval_example_overlap),
        "validation_eval_example_overlap_count": len(validation_eval_example_overlap),
        "train_validation_example_overlap_count": len(train_validation_example_overlap),
        "train_eval_video_overlap_ids": train_eval_video_overlap,
        "validation_eval_video_overlap_ids": validation_eval_video_overlap,
        "train_validation_video_overlap_ids": train_validation_video_overlap,
        "eval_video_ids_match_manifest": sorted(eval_video_ids) == sorted(eval_manifest_video_ids) if eval_manifest_video_ids else None,
        "train_video_ids_subset_of_manifest": sorted(train_all_video_ids - train_manifest_video_ids) == [] if train_manifest_video_ids else None,
        "validation_video_ids_subset_of_manifest": sorted(validation_video_ids - train_manifest_video_ids) == [] if train_manifest_video_ids else None,
        "eval_examples_seen_in_train_or_validation": len((train_example_ids | validation_example_ids) & eval_example_ids),
        "train_validation_overlap_expected_due_to_fallback": str(validation_split.get("strategy", "")) != "video_level",
        "eval_is_disjoint_from_train_and_validation": not train_eval_video_overlap
        and not validation_eval_video_overlap
        and not train_eval_example_overlap
        and not validation_eval_example_overlap,
        "notes": [
            "Validation examples are drawn only from the train-side split before any model selection.",
            "Train/validation overlap may appear in fallback modes when the train side contains too few videos to hold out a separate validation video set.",
        ],
    }


def _subset_probabilities(
    feature_matrix: Any,
    selected_indices: Sequence[int],
    coefficients: np.ndarray,
    intercept: float,
) -> List[float]:
    subset_logits = np.full(shape=(feature_matrix.shape[0],), fill_value=float(intercept), dtype=np.float64)
    if selected_indices:
        subset_logits += np.asarray(feature_matrix[:, list(selected_indices)].dot(coefficients[list(selected_indices)])).reshape(-1)
    return _sigmoid(subset_logits).tolist()


def _summary_family_labels(
    rule: Dict[str, Any],
    weight: float,
    cfg: Dict[str, Any],
) -> List[str]:
    vehicle_classes = {
        str(v)
        for v in cfg.get("vehicle_classes", ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"])
    }
    near_states = {str(v) for v in cfg.get("near_states", ["near"])}
    center_states = {str(v) for v in cfg.get("center_states", ["centered"])}

    has_transition = False
    has_ego_motion = False
    has_vehicle = False
    has_near = False
    has_centered = False

    for predicate, args in list(rule.get("rule_templates", [])):
        predicate_text = str(predicate).strip().lower()
        lowered_args = [str(arg).strip().lower() for arg in args]
        if "transition" in predicate_text or any("transition" in arg for arg in lowered_args):
            has_transition = True
        if predicate_text.startswith("ego_") or "ego_" in predicate_text or "ego" in lowered_args:
            has_ego_motion = True
        if predicate_text == "object_class" and len(args) >= 3 and str(args[2]) in vehicle_classes:
            has_vehicle = True
        if predicate_text == "object_distance_state" and len(args) >= 3 and str(args[2]) in near_states:
            has_near = True
        if predicate_text == "object_x_position_state" and len(args) >= 3 and str(args[2]) in center_states:
            has_centered = True

    labels: List[str] = []
    if has_transition:
        labels.append("transition")
    if has_ego_motion:
        labels.append("ego-motion")
    if has_vehicle and has_centered:
        labels.append("vehicle-centered")
    if has_near or has_centered:
        labels.append("near/centered object")
    if float(weight) < 0.0:
        labels.append("suppressor")
    if not labels:
        labels.append("other")
    return labels


def _signed_weight_sort_key(row: Dict[str, Any], sign: str) -> Tuple[float, float, str]:
    weight = float(row.get("weight", 0.0))
    rule_id = str(row.get("rule_id", ""))
    if sign == "positive":
        return (-weight, -abs(weight), rule_id)
    return (weight, -abs(weight), rule_id)


def _family_summary_rule_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rank": _safe_int(row.get("rank", 0)),
        "rule_id": str(row.get("rule_id", "")),
        "weight": float(row.get("weight", 0.0)),
        "abs_weight": float(row.get("abs_weight", abs(float(row.get("weight", 0.0))))),
        "semantic_family": str(row.get("semantic_family", "")),
        "summary_families": list(row.get("summary_families_list", [])),
        "confidence": float(row.get("confidence", 0.0)),
        "train_positive_support": int(row.get("train_positive_support", 0)),
        "train_negative_support": int(row.get("train_negative_support", 0)),
        "clause": str(row.get("clause", "")),
    }


def _build_family_summary(
    weighted_rule_rows: Sequence[Dict[str, Any]],
    top_rules_per_family: int = 5,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    discovered_families: List[str] = []
    for row in weighted_rule_rows:
        for family in list(row.get("summary_families_list", [])):
            if family not in _SUMMARY_FAMILY_ORDER and family not in discovered_families:
                discovered_families.append(family)
    family_names = list(_SUMMARY_FAMILY_ORDER) + discovered_families

    payload: Dict[str, Any] = {
        "top_rules_per_family": int(top_rules_per_family),
        "family_order": family_names,
        "positive": {},
        "negative": {},
    }
    csv_rows: List[Dict[str, Any]] = []

    for sign in ("positive", "negative"):
        signed_rows = [
            dict(row)
            for row in weighted_rule_rows
            if (float(row.get("weight", 0.0)) > 0.0 if sign == "positive" else float(row.get("weight", 0.0)) < 0.0)
        ]
        signed_rows = sorted(signed_rows, key=lambda row: _signed_weight_sort_key(row, sign))
        sign_payload = {
            "num_rules": len(signed_rows),
            "overall_top_rules": [_family_summary_rule_row(row) for row in signed_rows[:top_rules_per_family]],
            "families": [],
        }
        for family_name in family_names:
            family_rows = [
                dict(row) for row in signed_rows if family_name in list(row.get("summary_families_list", []))
            ]
            ordered_family_rows = sorted(family_rows, key=lambda row: _signed_weight_sort_key(row, sign))
            net_weight = float(sum(float(row.get("weight", 0.0)) for row in ordered_family_rows))
            total_abs_weight = float(sum(abs(float(row.get("weight", 0.0))) for row in ordered_family_rows))
            top_family_rows = ordered_family_rows[:top_rules_per_family]
            sign_payload["families"].append(
                {
                    "family": family_name,
                    "num_rules": len(ordered_family_rows),
                    "net_weight": net_weight,
                    "total_abs_weight": total_abs_weight,
                    "top_rules": [_family_summary_rule_row(row) for row in top_family_rows],
                }
            )
            for family_rank, row in enumerate(top_family_rows, start=1):
                csv_rows.append(
                    {
                        "sign": sign,
                        "family": family_name,
                        "family_rank": family_rank,
                        "num_rules_in_family": len(ordered_family_rows),
                        "net_weight": net_weight,
                        "total_abs_weight": total_abs_weight,
                        "rule_id": str(row.get("rule_id", "")),
                        "weight": float(row.get("weight", 0.0)),
                        "abs_weight": float(row.get("abs_weight", abs(float(row.get("weight", 0.0))))),
                        "semantic_family": str(row.get("semantic_family", "")),
                        "summary_families": " || ".join(str(v) for v in list(row.get("summary_families_list", []))),
                        "confidence": float(row.get("confidence", 0.0)),
                        "train_positive_support": int(row.get("train_positive_support", 0)),
                        "train_negative_support": int(row.get("train_negative_support", 0)),
                        "clause": str(row.get("clause", "")),
                    }
                )
        payload[sign] = sign_payload
    return payload, csv_rows


def _top_weight_subset_ablation_rows(
    val_matrix: Any,
    eval_matrix: Any,
    y_val: np.ndarray,
    y_eval: np.ndarray,
    active_rules: Sequence[Dict[str, Any]],
    coefficients: np.ndarray,
    intercept: float,
    shared_threshold: float,
    average_precision_score_fn: Any,
    roc_auc_score_fn: Any,
    confusion_matrix_fn: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nonzero_indices = np.flatnonzero(np.abs(coefficients) > 1e-12)
    ordered_nonzero_indices = sorted(
        nonzero_indices.tolist(),
        key=lambda index: (-abs(coefficients[index]), -coefficients[index], str(active_rules[index].get("rule_id", ""))),
    )

    summary_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    for subset_spec in _TOP_WEIGHT_ABLATION_SIZES:
        if subset_spec == "all_nonzero":
            selected_indices = list(ordered_nonzero_indices)
            subset_label = "all_nonzero"
        else:
            subset_k = int(subset_spec)
            selected_indices = list(ordered_nonzero_indices[: min(subset_k, len(ordered_nonzero_indices))])
            subset_label = str(subset_k)
        if not selected_indices:
            continue

        val_probabilities = _subset_probabilities(
            feature_matrix=val_matrix,
            selected_indices=selected_indices,
            coefficients=coefficients,
            intercept=intercept,
        )
        eval_probabilities = _subset_probabilities(
            feature_matrix=eval_matrix,
            selected_indices=selected_indices,
            coefficients=coefficients,
            intercept=intercept,
        )
        subset_specific_threshold, subset_specific_validation_metrics = _select_best_threshold(
            y_val.tolist(),
            val_probabilities,
            average_precision_score_fn,
            roc_auc_score_fn,
            confusion_matrix_fn,
            default_threshold=shared_threshold,
        )
        selected_rules = [dict(active_rules[index]) for index in selected_indices]
        selected_rule_ids = [str(rule.get("rule_id", "")) for rule in selected_rules]
        selected_rule_clauses = [str(rule.get("clause", "")) for rule in selected_rules]
        selected_rule_families = [str(rule.get("semantic_family", "")) for rule in selected_rules]

        per_split_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for split_name, labels, probabilities in [
            ("validation", y_val.tolist(), val_probabilities),
            ("eval", y_eval.tolist(), eval_probabilities),
        ]:
            per_split_metrics[split_name] = {}
            for threshold_name, threshold_value in [
                ("threshold_0_5", 0.5),
                ("shared_validation_threshold", shared_threshold),
                ("subset_specific_validation_threshold", subset_specific_threshold),
            ]:
                metrics = _compute_threshold_metrics(
                    labels,
                    probabilities,
                    float(threshold_value),
                    average_precision_score_fn,
                    roc_auc_score_fn,
                    confusion_matrix_fn,
                )
                per_split_metrics[split_name][threshold_name] = metrics
                metric_rows.append(
                    {
                        "subset_k": subset_label,
                        "subset_size": len(selected_indices),
                        "split_name": split_name,
                        "threshold_name": threshold_name,
                        "threshold_value": float(threshold_value),
                        "precision": float(metrics.get("precision", 0.0)),
                        "recall": float(metrics.get("recall", 0.0)),
                        "f1": float(metrics.get("f1", 0.0)),
                        "accuracy": float(metrics.get("accuracy", 0.0)),
                        "auroc": float(metrics.get("auroc", float("nan"))),
                        "auprc": float(metrics.get("auprc", float("nan"))),
                        "true_positive": int(metrics.get("true_positive", 0)),
                        "false_positive": int(metrics.get("false_positive", 0)),
                        "false_negative": int(metrics.get("false_negative", 0)),
                        "true_negative": int(metrics.get("true_negative", 0)),
                        "selected_rule_ids": " || ".join(selected_rule_ids),
                        "selected_rule_clauses": " || ".join(selected_rule_clauses),
                        "selected_rule_families": " || ".join(selected_rule_families),
                    }
                )

        threshold_05_eval_metrics = per_split_metrics["eval"]["threshold_0_5"]
        shared_eval_metrics = per_split_metrics["eval"]["shared_validation_threshold"]
        shared_validation_metrics = per_split_metrics["validation"]["shared_validation_threshold"]
        subset_eval_metrics = per_split_metrics["eval"]["subset_specific_validation_threshold"]

        summary_rows.append(
            {
                "subset_k": subset_label,
                "subset_size": len(selected_indices),
                "threshold_name": "shared_validation_threshold",
                "threshold_value": float(shared_threshold),
                "threshold_0_5_precision": float(threshold_05_eval_metrics.get("precision", 0.0)),
                "threshold_0_5_recall": float(threshold_05_eval_metrics.get("recall", 0.0)),
                "threshold_0_5_f1": float(threshold_05_eval_metrics.get("f1", 0.0)),
                "precision": float(shared_eval_metrics.get("precision", 0.0)),
                "recall": float(shared_eval_metrics.get("recall", 0.0)),
                "f1": float(shared_eval_metrics.get("f1", 0.0)),
                "accuracy": float(shared_eval_metrics.get("accuracy", 0.0)),
                "auroc": float(shared_eval_metrics.get("auroc", float("nan"))),
                "auprc": float(shared_eval_metrics.get("auprc", float("nan"))),
                "true_positive": int(shared_eval_metrics.get("true_positive", 0)),
                "false_positive": int(shared_eval_metrics.get("false_positive", 0)),
                "false_negative": int(shared_eval_metrics.get("false_negative", 0)),
                "true_negative": int(shared_eval_metrics.get("true_negative", 0)),
                "shared_threshold_validation_precision": float(shared_validation_metrics.get("precision", 0.0)),
                "shared_threshold_validation_recall": float(shared_validation_metrics.get("recall", 0.0)),
                "shared_threshold_validation_f1": float(shared_validation_metrics.get("f1", 0.0)),
                "shared_threshold_validation_accuracy": float(shared_validation_metrics.get("accuracy", 0.0)),
                "shared_threshold_validation_auroc": float(shared_validation_metrics.get("auroc", float("nan"))),
                "shared_threshold_validation_auprc": float(shared_validation_metrics.get("auprc", float("nan"))),
                "subset_specific_threshold_name": "subset_specific_validation_threshold",
                "subset_specific_threshold_value": float(subset_specific_threshold),
                "subset_specific_validation_precision": float(subset_specific_validation_metrics.get("precision", 0.0)),
                "subset_specific_validation_recall": float(subset_specific_validation_metrics.get("recall", 0.0)),
                "subset_specific_validation_f1": float(subset_specific_validation_metrics.get("f1", 0.0)),
                "subset_specific_validation_accuracy": float(subset_specific_validation_metrics.get("accuracy", 0.0)),
                "subset_specific_validation_auroc": float(subset_specific_validation_metrics.get("auroc", float("nan"))),
                "subset_specific_validation_auprc": float(subset_specific_validation_metrics.get("auprc", float("nan"))),
                "subset_specific_precision": float(subset_eval_metrics.get("precision", 0.0)),
                "subset_specific_recall": float(subset_eval_metrics.get("recall", 0.0)),
                "subset_specific_f1": float(subset_eval_metrics.get("f1", 0.0)),
                "subset_specific_accuracy": float(subset_eval_metrics.get("accuracy", 0.0)),
                "subset_specific_auroc": float(subset_eval_metrics.get("auroc", float("nan"))),
                "subset_specific_auprc": float(subset_eval_metrics.get("auprc", float("nan"))),
                "subset_specific_true_positive": int(subset_eval_metrics.get("true_positive", 0)),
                "subset_specific_false_positive": int(subset_eval_metrics.get("false_positive", 0)),
                "subset_specific_false_negative": int(subset_eval_metrics.get("false_negative", 0)),
                "subset_specific_true_negative": int(subset_eval_metrics.get("true_negative", 0)),
                "selected_rule_ids": " || ".join(selected_rule_ids),
                "selected_rule_clauses": " || ".join(selected_rule_clauses),
                "selected_rule_families": " || ".join(selected_rule_families),
            }
        )
    return summary_rows, metric_rows


def process_baseline(
    extended_rule_results: Dict[str, Any],
    train_temporal_rule_results: Sequence[Dict[str, Any]],
    eval_temporal_rule_results: Sequence[Dict[str, Any]],
    split_manifest: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)

    summary_path = out_root / "rule_aggregation_baseline_summary.json"
    metrics_csv_path = out_root / "rule_aggregation_baseline_metrics.csv"
    per_video_csv_path = out_root / "per_video_metrics.csv"
    prediction_csv_path = out_root / "prediction_examples.csv"
    top_rules_csv_path = out_root / "top_weighted_rules_with_clauses.csv"
    all_weights_csv_path = out_root / "all_nonzero_weighted_rules_with_clauses.csv"
    ablation_csv_path = out_root / "rule_aggregation_ablation_metrics.csv"
    subset_threshold_metrics_csv_path = out_root / "rule_aggregation_subset_threshold_metrics.csv"
    top_k_rule_sets_csv_path = out_root / "top_k_weighted_rule_sets.csv"
    family_summary_json_path = out_root / "rule_aggregation_family_summary.json"
    family_summary_csv_path = out_root / "rule_aggregation_family_summary.csv"
    split_leakage_json_path = out_root / "split_leakage_check.json"

    if not force_recompute and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _RULE_AGGREGATION_BASELINE_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {summary_path.name}")
            return cached

    sparse, LogisticRegression, average_precision_score, roc_auc_score, confusion_matrix = _import_ml_dependencies()

    random_seed = int(cfg.get("random_seed", 0))
    np.random.seed(random_seed)
    random.seed(random_seed)

    train_examples_all = _prepare_examples(train_temporal_rule_results)
    eval_examples = _prepare_examples(eval_temporal_rule_results)
    if not train_examples_all:
        raise RuntimeError("Rule aggregation baseline received an empty training split.")
    if not eval_examples:
        raise RuntimeError("Rule aggregation baseline received an empty evaluation split.")

    train_examples, val_examples, validation_split = _split_train_validation(
        train_examples_all,
        validation_fraction=float(cfg.get("validation_fraction", 0.25)),
        seed=random_seed,
    )
    if not train_examples or not val_examples:
        raise RuntimeError("Rule aggregation baseline could not create non-empty train/validation subsets.")
    split_checks = _split_separation_checks(
        train_examples_all=train_examples_all,
        train_examples=train_examples,
        val_examples=val_examples,
        eval_examples=eval_examples,
        validation_split=validation_split,
        split_manifest=split_manifest,
    )
    if not bool(split_checks.get("eval_is_disjoint_from_train_and_validation", False)):
        raise RuntimeError(
            "Rule aggregation baseline detected train/eval or validation/eval overlap; aborting to avoid eval leakage."
        )

    prepared_rules = _prepare_rules(extended_rule_results, cfg)
    if not prepared_rules:
        raise RuntimeError("Rule aggregation baseline found no valid rules in the Step 16 rule pool.")

    train_matrix_full = _build_rule_firing_matrix(train_examples, prepared_rules, sparse)
    val_matrix_full = _build_rule_firing_matrix(val_examples, prepared_rules, sparse)
    eval_matrix_full = _build_rule_firing_matrix(eval_examples, prepared_rules, sparse)

    min_train_support = max(1, int(cfg.get("active_rule_min_train_support", 1)))
    active_rule_mask = np.asarray(train_matrix_full.getnnz(axis=0)).reshape(-1) >= min_train_support
    active_rule_indices = np.flatnonzero(active_rule_mask)
    if active_rule_indices.size == 0:
        raise RuntimeError("Rule aggregation baseline found no active train-time rule features.")

    train_matrix = train_matrix_full[:, active_rule_indices]
    val_matrix = val_matrix_full[:, active_rule_indices]
    eval_matrix = eval_matrix_full[:, active_rule_indices]
    active_rules = [prepared_rules[int(index)] for index in active_rule_indices.tolist()]

    y_train = np.asarray([1 if bool(example.get("label", False)) else 0 for example in train_examples], dtype=np.int64)
    y_val = np.asarray([1 if bool(example.get("label", False)) else 0 for example in val_examples], dtype=np.int64)
    y_eval = np.asarray([1 if bool(example.get("label", False)) else 0 for example in eval_examples], dtype=np.int64)

    c_values = [float(v) for v in cfg.get("c_values", [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])]
    solver = str(cfg.get("solver", "liblinear"))
    class_weight_cfg = str(cfg.get("class_weight", "balanced")).strip().lower()
    class_weight = "balanced" if class_weight_cfg == "balanced" else None
    max_iter = int(cfg.get("max_iter", 2000))

    candidate_rows: List[Dict[str, Any]] = []
    best_model: Optional[Any] = None
    best_threshold = 0.5
    best_key: Optional[Tuple[float, float, float, float, float]] = None
    best_val_metrics_best: Dict[str, Any] = {}
    best_val_metrics_05: Dict[str, Any] = {}
    best_c = c_values[0] if c_values else 1.0

    for c_value in c_values:
        model = LogisticRegression(
            penalty="l1",
            solver=solver,
            C=float(c_value),
            max_iter=max_iter,
            random_state=random_seed,
            class_weight=class_weight,
        )
        model.fit(train_matrix, y_train)
        val_probabilities = model.predict_proba(val_matrix)[:, 1].astype(np.float64).tolist()
        threshold_05 = _compute_threshold_metrics(
            y_val.tolist(),
            val_probabilities,
            0.5,
            average_precision_score,
            roc_auc_score,
            confusion_matrix,
        )
        tuned_threshold, tuned_metrics = _select_best_threshold(
            y_val.tolist(),
            val_probabilities,
            average_precision_score,
            roc_auc_score,
            confusion_matrix,
            default_threshold=0.5,
        )
        nonzero_rule_count = int(np.count_nonzero(np.abs(model.coef_[0]) > 1e-12))
        candidate_rows.append(
            {
                "c_value": float(c_value),
                "validation_threshold_at_0_5_f1": float(threshold_05.get("f1", 0.0)),
                "validation_threshold_at_0_5_precision": float(threshold_05.get("precision", 0.0)),
                "validation_threshold_at_0_5_recall": float(threshold_05.get("recall", 0.0)),
                "best_validation_threshold": float(tuned_threshold),
                "best_validation_f1": float(tuned_metrics.get("f1", 0.0)),
                "best_validation_precision": float(tuned_metrics.get("precision", 0.0)),
                "best_validation_recall": float(tuned_metrics.get("recall", 0.0)),
                "nonzero_rule_count": nonzero_rule_count,
            }
        )
        key = (
            float(tuned_metrics.get("f1", 0.0)),
            float(tuned_metrics.get("precision", 0.0)),
            float(tuned_metrics.get("recall", 0.0)),
            -float(nonzero_rule_count),
            -float(c_value),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_model = model
            best_threshold = float(tuned_threshold)
            best_val_metrics_best = tuned_metrics
            best_val_metrics_05 = threshold_05
            best_c = float(c_value)

    if best_model is None:
        raise RuntimeError("Rule aggregation baseline could not fit any logistic-regression model.")

    train_probabilities = best_model.predict_proba(train_matrix)[:, 1].astype(np.float64).tolist()
    val_probabilities = best_model.predict_proba(val_matrix)[:, 1].astype(np.float64).tolist()
    eval_probabilities = best_model.predict_proba(eval_matrix)[:, 1].astype(np.float64).tolist()

    threshold_05 = 0.5
    metric_rows: List[Dict[str, Any]] = []
    split_payloads = [
        ("train", train_examples, y_train.tolist(), train_probabilities),
        ("validation", val_examples, y_val.tolist(), val_probabilities),
        ("eval", eval_examples, y_eval.tolist(), eval_probabilities),
    ]
    metrics_by_split: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for split_name, split_examples, labels, probabilities in split_payloads:
        metrics_by_split[split_name] = {}
        for threshold_name, threshold_value in [
            ("threshold_0_5", threshold_05),
            ("best_validation_threshold", best_threshold),
        ]:
            metrics = _compute_threshold_metrics(
                labels,
                probabilities,
                threshold_value,
                average_precision_score,
                roc_auc_score,
                confusion_matrix,
            )
            row = {
                "model_name": "rule_aggregation_logistic_regression",
                "split_name": split_name,
                "threshold_name": threshold_name,
                "threshold_value": float(threshold_value),
                "num_examples": len(split_examples),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "auroc": float(metrics.get("auroc", float("nan"))),
                "auprc": float(metrics.get("auprc", float("nan"))),
                "true_positive": int(metrics.get("true_positive", 0)),
                "false_positive": int(metrics.get("false_positive", 0)),
                "false_negative": int(metrics.get("false_negative", 0)),
                "true_negative": int(metrics.get("true_negative", 0)),
            }
            metric_rows.append(row)
            metrics_by_split[split_name][threshold_name] = dict(row)

    per_video_rows: List[Dict[str, Any]] = []
    for threshold_name, threshold_value in [
        ("threshold_0_5", threshold_05),
        ("best_validation_threshold", best_threshold),
    ]:
        per_video_rows.extend(
            _per_video_metrics(
                "rule_aggregation_logistic_regression",
                eval_examples,
                eval_probabilities,
                threshold_name,
                threshold_value,
                average_precision_score,
                roc_auc_score,
                confusion_matrix,
            )
        )

    prediction_rows = _prediction_rows(eval_examples, eval_probabilities, threshold_05, best_threshold)

    coefficients = np.asarray(best_model.coef_[0], dtype=np.float64)
    intercept = float(np.asarray(best_model.intercept_, dtype=np.float64).reshape(-1)[0])
    nonzero_indices = np.flatnonzero(np.abs(coefficients) > 1e-12)
    top_k_rules = max(1, int(cfg.get("top_weighted_rules", 30)))
    top_weight_rows: List[Dict[str, Any]] = []
    all_weight_rows: List[Dict[str, Any]] = []
    ordered_nonzero_indices = sorted(
        nonzero_indices.tolist(),
        key=lambda index: (-abs(coefficients[index]), -coefficients[index], active_rules[index].get("rule_id", "")),
    )
    for rank, feature_index in enumerate(ordered_nonzero_indices, start=1):
        rule = dict(active_rules[feature_index])
        weight = float(coefficients[feature_index])
        row = {
            "rank": rank,
            "rule_id": str(rule.get("rule_id", "")),
            "clause": str(rule.get("clause", "")),
            "weight": weight,
            "abs_weight": abs(weight),
            "sign": "positive" if weight >= 0.0 else "negative",
            "confidence": float(rule.get("confidence", 0.0)),
            "train_positive_support": int(rule.get("positive_support", 0)),
            "train_negative_support": int(rule.get("negative_support", 0)),
            "semantic_family": str(rule.get("semantic_family", "")),
        }
        row["summary_families_list"] = _summary_family_labels(rule, weight, cfg)
        row["summary_families"] = " || ".join(row["summary_families_list"])
        all_weight_rows.append(row)
    top_weight_rows = [dict(row) for row in all_weight_rows[:top_k_rules]]

    subset_ablation_rows, subset_threshold_metric_rows = _top_weight_subset_ablation_rows(
        val_matrix=val_matrix,
        eval_matrix=eval_matrix,
        y_val=y_val,
        y_eval=y_eval,
        active_rules=active_rules,
        coefficients=coefficients,
        intercept=intercept,
        shared_threshold=best_threshold,
        average_precision_score_fn=average_precision_score,
        roc_auc_score_fn=roc_auc_score,
        confusion_matrix_fn=confusion_matrix,
    )
    family_summary_payload, family_summary_rows = _build_family_summary(all_weight_rows)

    _write_csv(
        metrics_csv_path,
        [
            "model_name",
            "split_name",
            "threshold_name",
            "threshold_value",
            "num_examples",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "auroc",
            "auprc",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        ],
        metric_rows,
    )
    _write_csv(
        per_video_csv_path,
        [
            "model_name",
            "video_id",
            "num_examples",
            "threshold_name",
            "threshold_value",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "auroc",
            "auprc",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        ],
        per_video_rows,
    )
    _write_csv(
        prediction_csv_path,
        [
            "video_id",
            "example_id",
            "current_segment_index",
            "label",
            "predicted_probability",
            "threshold_at_0_5",
            "best_validation_threshold",
            "predicted_label_at_0_5",
            "predicted_label_at_best_validation_threshold",
            "current_segment_label",
            "num_body_atoms",
        ],
        prediction_rows,
    )
    _write_csv(
        all_weights_csv_path,
        [
            "rank",
            "rule_id",
            "clause",
            "weight",
            "abs_weight",
            "sign",
            "confidence",
            "train_positive_support",
            "train_negative_support",
            "semantic_family",
            "summary_families",
        ],
        all_weight_rows,
    )
    _write_csv(
        top_rules_csv_path,
        [
            "rank",
            "rule_id",
            "clause",
            "weight",
            "abs_weight",
            "sign",
            "confidence",
            "train_positive_support",
            "train_negative_support",
            "semantic_family",
            "summary_families",
        ],
        top_weight_rows,
    )
    _write_csv(
        ablation_csv_path,
        [
            "subset_k",
            "subset_size",
            "threshold_name",
            "threshold_value",
            "threshold_0_5_precision",
            "threshold_0_5_recall",
            "threshold_0_5_f1",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "auroc",
            "auprc",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "shared_threshold_validation_precision",
            "shared_threshold_validation_recall",
            "shared_threshold_validation_f1",
            "shared_threshold_validation_accuracy",
            "shared_threshold_validation_auroc",
            "shared_threshold_validation_auprc",
            "subset_specific_threshold_name",
            "subset_specific_threshold_value",
            "subset_specific_validation_precision",
            "subset_specific_validation_recall",
            "subset_specific_validation_f1",
            "subset_specific_validation_accuracy",
            "subset_specific_validation_auroc",
            "subset_specific_validation_auprc",
            "subset_specific_precision",
            "subset_specific_recall",
            "subset_specific_f1",
            "subset_specific_accuracy",
            "subset_specific_auroc",
            "subset_specific_auprc",
            "subset_specific_true_positive",
            "subset_specific_false_positive",
            "subset_specific_false_negative",
            "subset_specific_true_negative",
            "selected_rule_ids",
            "selected_rule_clauses",
            "selected_rule_families",
        ],
        subset_ablation_rows,
    )
    _write_csv(
        subset_threshold_metrics_csv_path,
        [
            "subset_k",
            "subset_size",
            "split_name",
            "threshold_name",
            "threshold_value",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "auroc",
            "auprc",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "selected_rule_ids",
            "selected_rule_clauses",
            "selected_rule_families",
        ],
        subset_threshold_metric_rows,
    )
    top_k_rule_set_rows = [
        {
            "subset_k": row.get("subset_k", ""),
            "subset_size": row.get("subset_size", 0),
            "threshold_name": row.get("threshold_name", ""),
            "threshold_value": row.get("threshold_value", 0.0),
            "subset_specific_threshold_name": row.get("subset_specific_threshold_name", ""),
            "subset_specific_threshold_value": row.get("subset_specific_threshold_value", 0.0),
            "selected_rule_ids": row.get("selected_rule_ids", ""),
            "selected_rule_clauses": row.get("selected_rule_clauses", ""),
            "selected_rule_families": row.get("selected_rule_families", ""),
        }
        for row in subset_ablation_rows
    ]
    _write_csv(
        top_k_rule_sets_csv_path,
        [
            "subset_k",
            "subset_size",
            "threshold_name",
            "threshold_value",
            "subset_specific_threshold_name",
            "subset_specific_threshold_value",
            "selected_rule_ids",
            "selected_rule_clauses",
            "selected_rule_families",
        ],
        top_k_rule_set_rows,
    )
    with family_summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(family_summary_payload, fh, indent=2)
    _write_csv(
        family_summary_csv_path,
        [
            "sign",
            "family",
            "family_rank",
            "num_rules_in_family",
            "net_weight",
            "total_abs_weight",
            "rule_id",
            "weight",
            "abs_weight",
            "semantic_family",
            "summary_families",
            "confidence",
            "train_positive_support",
            "train_negative_support",
            "clause",
        ],
        family_summary_rows,
    )

    tuning_checks = {
        "regularization_tuned_on": "train_validation_only",
        "threshold_tuned_on": "validation_only",
        "eval_used_for_regularization_search": False,
        "eval_used_for_threshold_search": False,
        "best_model_selected_from_validation_search": True,
        "validation_candidate_count": len(candidate_rows),
        "validation_search_fields": [
            "best_validation_f1",
            "best_validation_precision",
            "best_validation_recall",
            "best_validation_threshold",
        ],
        "held_out_eval_applied_only_after_model_selection": True,
        "notes": [
            "Each C candidate is fit on the train subset only.",
            "Best C and decision threshold are selected using validation metrics only.",
            "Held-out eval probabilities are computed only after best C and threshold are fixed.",
            "Top-|weight| subset thresholds are also tuned on validation only, then applied to held-out eval.",
        ],
    }
    split_leakage_payload = {
        "split_separation_checks": split_checks,
        "tuning_checks": tuning_checks,
    }
    with split_leakage_json_path.open("w", encoding="utf-8") as fh:
        json.dump(split_leakage_payload, fh, indent=2)

    summary: Dict[str, Any] = {
        "version": _RULE_AGGREGATION_BASELINE_VERSION,
        "config": _cfg_key_subset(cfg),
        "split": split_manifest or {},
        "validation_split": validation_split,
        "model_type": "sparse_l1_logistic_regression",
        "solver": solver,
        "class_weight": class_weight if class_weight is not None else "none",
        "best_c": best_c,
        "best_validation_threshold": float(best_threshold),
        "num_pool_rules": len(prepared_rules),
        "num_active_train_rules": len(active_rules),
        "num_nonzero_rules": int(nonzero_indices.size),
        "tuning_checks": tuning_checks,
        "split_separation_checks": split_checks,
        "metrics_by_split": metrics_by_split,
        "validation_search": candidate_rows,
        "top_weighted_rules": top_weight_rows,
        "weighted_rule_family_summary": family_summary_payload,
        "top_weight_rule_subset_ablations": subset_ablation_rows,
        "top_weight_rule_subset_threshold_metrics": subset_threshold_metric_rows,
        "output_paths": {
            "summary_json": str(summary_path),
            "metrics_csv": str(metrics_csv_path),
            "per_video_metrics_csv": str(per_video_csv_path),
            "prediction_examples_csv": str(prediction_csv_path),
            "all_nonzero_weighted_rules_with_clauses_csv": str(all_weights_csv_path),
            "top_weighted_rules_with_clauses_csv": str(top_rules_csv_path),
            "rule_aggregation_ablation_metrics_csv": str(ablation_csv_path),
            "rule_aggregation_subset_threshold_metrics_csv": str(subset_threshold_metrics_csv_path),
            "top_k_weighted_rule_sets_csv": str(top_k_rule_sets_csv_path),
            "rule_aggregation_family_summary_json": str(family_summary_json_path),
            "rule_aggregation_family_summary_csv": str(family_summary_csv_path),
            "split_leakage_check_json": str(split_leakage_json_path),
        },
    }

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    selected_eval = dict(metrics_by_split["eval"]["best_validation_threshold"])
    print(
        "  rule_aggregation_baseline: "
        f"pool_rules={len(prepared_rules)} | "
        f"active_train_rules={len(active_rules)} | "
        f"nonzero_rules={int(nonzero_indices.size)} | "
        f"best_c={best_c:.4g} | "
        f"eval_f1={float(selected_eval.get('f1', 0.0)):.3f}"
    )
    print(f"Rule aggregation baseline summary JSON written to {summary_path}")
    print(f"Rule aggregation baseline all nonzero weighted rules CSV written to {all_weights_csv_path}")
    print(f"Rule aggregation baseline top weighted rules CSV written to {top_rules_csv_path}")
    print(f"Rule aggregation baseline subset ablations CSV written to {ablation_csv_path}")
    print(f"Rule aggregation baseline subset threshold metrics CSV written to {subset_threshold_metrics_csv_path}")
    print(f"Rule aggregation baseline top-K weighted rule sets CSV written to {top_k_rule_sets_csv_path}")
    print(f"Rule aggregation baseline family summary JSON written to {family_summary_json_path}")
    print(f"Rule aggregation baseline family summary CSV written to {family_summary_csv_path}")
    print(f"Rule aggregation baseline split leakage check JSON written to {split_leakage_json_path}")
    return summary


def run(
    extended_rule_results: Dict[str, Any],
    train_temporal_rule_results: Sequence[Dict[str, Any]],
    eval_temporal_rule_results: Sequence[Dict[str, Any]],
    split_manifest: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_baseline(
        extended_rule_results=extended_rule_results,
        train_temporal_rule_results=train_temporal_rule_results,
        eval_temporal_rule_results=eval_temporal_rule_results,
        split_manifest=split_manifest,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
