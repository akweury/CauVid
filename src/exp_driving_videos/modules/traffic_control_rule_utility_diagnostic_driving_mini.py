"""
Inspect traffic-control rule utility across the Step 16 pool, selected rule sets,
and Step 18C learned rule aggregation.

Outputs:
    pipeline_output/18e_driving_mini_traffic_control_rule_utility/
        traffic_control_rule_utility_summary.json
        traffic_light_state_rule_counts.csv
        traffic_control_selected_rule_matches.csv
        traffic_control_weighted_rule_matches.csv
        traffic_control_firing_by_error_type.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config
from src.exp_driving_videos.modules.evaluate_rules_driving_mini import _compute_binary_metrics
from src.exp_driving_videos.modules.rule_aggregation_baseline_driving_mini import (
    _build_rule_firing_matrix,
    _import_ml_dependencies,
    _prepare_examples,
    _prepare_rules,
)


_TRAFFIC_CONTROL_RULE_UTILITY_VERSION = 1
_DEFAULT_HIGHLIGHT_PREDICATES: Tuple[str, ...] = (
    "traffic_light_state",
    "traffic_light_relevant",
    "stop_sign_relevant",
    "traffic_control_relevant",
)
_DEFAULT_STATE_VALUES: Tuple[str, ...] = ("red", "yellow", "green")
_RULE_SET_ORDER: Tuple[Tuple[str, str], ...] = (
    ("original", "step17_original"),
    ("diverse", "step17b_diverse"),
    ("semantic_constrained_diverse", "step17b2_semantic_constrained"),
    ("coverage_family_aware", "step17c_coverage_family_aware"),
)
_LR_SOURCE_ORDER: Tuple[str, ...] = (
    "step18c_lr_nonzero_all",
    "step18c_lr_nonzero_positive",
    "step18c_lr_nonzero_negative",
)


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "18e_driving_mini_traffic_control_rule_utility"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "highlight_predicates": sorted(str(v) for v in cfg.get("highlight_predicates", [])),
        "tracked_states": sorted(str(v) for v in cfg.get("tracked_states", [])),
        "top_rules_per_key": int(cfg.get("top_rules_per_key", 5)),
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


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_traffic_control_predicate(predicate: str) -> bool:
    normalized = _normalize_text(predicate)
    return normalized.startswith("traffic_") or normalized == "stop_sign_relevant"


def _rule_key_rows(rule: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    seen: Set[Tuple[str, str, str, str]] = set()
    for predicate, args in list(rule.get("rule_templates", [])):
        predicate_name = _normalize_text(predicate)
        if not _is_traffic_control_predicate(predicate_name):
            continue
        predicate_key = ("predicate", predicate_name, "", predicate_name)
        if predicate_key not in seen:
            rows.append(predicate_key)
            seen.add(predicate_key)
        if predicate_name == "traffic_light_state" and len(args) >= 3:
            state_name = _normalize_text(args[2])
            state_key = (
                "state",
                predicate_name,
                state_name,
                f"{predicate_name}={state_name}",
            )
            if state_key not in seen:
                rows.append(state_key)
                seen.add(state_key)
    return rows


def _rule_key_map(rules: Sequence[Dict[str, Any]]) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = {}
    for rule in rules:
        rule_id = str(rule.get("rule_id", ""))
        if not rule_id:
            continue
        for key in _rule_key_rows(rule):
            mapping.setdefault(rule_id, set()).add("||".join(key))
    return mapping


def _rule_stats_from_matrix(
    prepared_rules: Sequence[Dict[str, Any]],
    eval_examples: Sequence[Dict[str, Any]],
    eval_matrix: Any,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    example_ids = [str(example.get("example_id", "")) for example in eval_examples]
    positive_example_ids = {
        str(example.get("example_id", ""))
        for example in eval_examples
        if bool(example.get("label", False))
    }
    negative_example_ids = set(example_ids) - positive_example_ids
    rule_stats: Dict[str, Dict[str, Any]] = {}
    fired_example_ids_by_rule: Dict[str, Set[str]] = {}
    positive_example_ids_by_rule: Dict[str, Set[str]] = {}
    matrix_csc = eval_matrix.tocsc()
    total_positive = max(1, len(positive_example_ids))
    for index, rule in enumerate(prepared_rules):
        rule_id = str(rule.get("rule_id", ""))
        column_start = int(matrix_csc.indptr[index])
        column_end = int(matrix_csc.indptr[index + 1])
        fired_indices = matrix_csc.indices[column_start:column_end].tolist()
        fired_example_ids = {example_ids[row_index] for row_index in fired_indices}
        fired_positive_ids = fired_example_ids & positive_example_ids
        fired_negative_ids = fired_example_ids & negative_example_ids
        fired_example_ids_by_rule[rule_id] = fired_example_ids
        positive_example_ids_by_rule[rule_id] = fired_positive_ids
        rule_stats[rule_id] = {
            "eval_positive_support": len(fired_positive_ids),
            "eval_negative_support": len(fired_negative_ids),
            "eval_total_support": len(fired_example_ids),
            "eval_recall": float(len(fired_positive_ids) / total_positive),
        }
    return rule_stats, fired_example_ids_by_rule, positive_example_ids_by_rule


def _prediction_sets_from_mask(
    eval_examples: Sequence[Dict[str, Any]],
    predicted_positive_mask: np.ndarray,
) -> Dict[str, Set[str]]:
    example_ids = [str(example.get("example_id", "")) for example in eval_examples]
    labels = [bool(example.get("label", False)) for example in eval_examples]
    tp: Set[str] = set()
    fp: Set[str] = set()
    fn: Set[str] = set()
    tn: Set[str] = set()
    for example_id, label, predicted_positive in zip(example_ids, labels, predicted_positive_mask.tolist()):
        if predicted_positive and label:
            tp.add(example_id)
        elif predicted_positive and not label:
            fp.add(example_id)
        elif not predicted_positive and label:
            fn.add(example_id)
        else:
            tn.add(example_id)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def _selected_rule_prediction_sets(
    eval_examples: Sequence[Dict[str, Any]],
    eval_matrix: Any,
    rule_id_to_index: Dict[str, int],
    selected_rule_ids: Iterable[str],
) -> Dict[str, Any]:
    selected_indices = [
        int(rule_id_to_index[rule_id])
        for rule_id in selected_rule_ids
        if str(rule_id) in rule_id_to_index
    ]
    if selected_indices:
        predicted_positive_mask = np.asarray(eval_matrix[:, selected_indices].getnnz(axis=1)).reshape(-1) > 0
    else:
        predicted_positive_mask = np.zeros(shape=(len(eval_examples),), dtype=bool)
    outcome_sets = _prediction_sets_from_mask(eval_examples, predicted_positive_mask)
    metrics = _compute_binary_metrics(
        true_positive=len(outcome_sets["TP"]),
        false_positive=len(outcome_sets["FP"]),
        false_negative=len(outcome_sets["FN"]),
        true_negative=len(outcome_sets["TN"]),
    )
    return {
        "selected_rule_count": len(selected_indices),
        "selected_rule_ids": [str(rule_id) for rule_id in selected_rule_ids if str(rule_id) in rule_id_to_index],
        "outcome_example_ids": {key: sorted(value) for key, value in outcome_sets.items()},
        "metrics": metrics,
    }


def _bool_from_csv(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _read_lr_prediction_sets(rule_aggregation_baseline_results: Dict[str, Any]) -> Dict[str, Any]:
    prediction_csv_path = Path(
        str(rule_aggregation_baseline_results.get("output_paths", {}).get("prediction_examples_csv", ""))
    )
    if not prediction_csv_path.exists():
        raise RuntimeError(
            "Traffic-control rule utility diagnostic expected Step 18C prediction_examples.csv to exist."
        )

    outcome_sets = {"TP": set(), "FP": set(), "FN": set(), "TN": set()}
    with prediction_csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            example_id = str(row.get("example_id", ""))
            if not example_id:
                continue
            label = _bool_from_csv(row.get("label", False))
            predicted_positive = _bool_from_csv(row.get("predicted_label_at_best_validation_threshold", False))
            if predicted_positive and label:
                outcome_sets["TP"].add(example_id)
            elif predicted_positive and not label:
                outcome_sets["FP"].add(example_id)
            elif not predicted_positive and label:
                outcome_sets["FN"].add(example_id)
            else:
                outcome_sets["TN"].add(example_id)
    metrics = _compute_binary_metrics(
        true_positive=len(outcome_sets["TP"]),
        false_positive=len(outcome_sets["FP"]),
        false_negative=len(outcome_sets["FN"]),
        true_negative=len(outcome_sets["TN"]),
    )
    return {
        "outcome_example_ids": {key: sorted(value) for key, value in outcome_sets.items()},
        "metrics": metrics,
    }


def _read_all_nonzero_weight_rows(rule_aggregation_baseline_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    all_weights_csv_path = Path(
        str(
            rule_aggregation_baseline_results.get("output_paths", {}).get(
                "all_nonzero_weighted_rules_with_clauses_csv", ""
            )
        )
    )
    if not all_weights_csv_path.exists():
        raise RuntimeError(
            "Traffic-control rule utility diagnostic expected Step 18C all_nonzero_weighted_rules_with_clauses.csv to exist."
        )

    rows: List[Dict[str, Any]] = []
    with all_weights_csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            normalized = dict(row)
            normalized["rank"] = _safe_int(row.get("rank", 0))
            normalized["weight"] = _safe_float(row.get("weight", 0.0))
            normalized["abs_weight"] = _safe_float(row.get("abs_weight", abs(_safe_float(row.get("weight", 0.0)))))
            normalized["confidence"] = _safe_float(row.get("confidence", 0.0))
            normalized["train_positive_support"] = _safe_int(row.get("train_positive_support", 0))
            normalized["train_negative_support"] = _safe_int(row.get("train_negative_support", 0))
            rows.append(normalized)
    return rows


def _source_rule_ids_for_key(
    key_id: str,
    key_rule_ids: Set[str],
    rule_set_predictions: Dict[str, Dict[str, Any]],
    lr_weight_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Set[str]]:
    source_rule_ids: Dict[str, Set[str]] = {}
    for rule_set_name, stage_name in _RULE_SET_ORDER:
        selected = set(rule_set_predictions.get(rule_set_name, {}).get("selected_rule_ids", []))
        source_rule_ids[stage_name] = key_rule_ids & selected

    nonzero_all = {str(row.get("rule_id", "")) for row in lr_weight_rows if str(row.get("rule_id", "")) in key_rule_ids}
    nonzero_positive = {
        str(row.get("rule_id", ""))
        for row in lr_weight_rows
        if str(row.get("rule_id", "")) in key_rule_ids and _safe_float(row.get("weight", 0.0)) > 0.0
    }
    nonzero_negative = {
        str(row.get("rule_id", ""))
        for row in lr_weight_rows
        if str(row.get("rule_id", "")) in key_rule_ids and _safe_float(row.get("weight", 0.0)) < 0.0
    }
    source_rule_ids["step18c_lr_nonzero_all"] = nonzero_all
    source_rule_ids["step18c_lr_nonzero_positive"] = nonzero_positive
    source_rule_ids["step18c_lr_nonzero_negative"] = nonzero_negative
    return source_rule_ids


def _firing_stat_rows_for_source(
    key_tuple: Tuple[str, str, str, str],
    source_name: str,
    matched_rule_ids: Set[str],
    outcome_example_ids: Dict[str, Sequence[str]],
    fired_example_ids_by_rule: Dict[str, Set[str]],
) -> List[Dict[str, Any]]:
    entry_kind, predicate_name, state_name, display_name = key_tuple
    rows: List[Dict[str, Any]] = []
    for outcome_name in ("TP", "FP", "FN", "TN"):
        outcome_ids = set(str(value) for value in outcome_example_ids.get(outcome_name, []))
        covered_examples: Set[str] = set()
        example_level_firings = 0
        for rule_id in matched_rule_ids:
            fired_example_ids = set(fired_example_ids_by_rule.get(rule_id, set()))
            overlap = fired_example_ids & outcome_ids
            covered_examples.update(overlap)
            example_level_firings += len(overlap)
        rows.append(
            {
                "entry_kind": entry_kind,
                "predicate_name": predicate_name,
                "state_name": state_name,
                "display_name": display_name,
                "source_name": source_name,
                "outcome_bucket": outcome_name,
                "matched_rule_count": len(matched_rule_ids),
                "covered_example_count": len(covered_examples),
                "example_level_rule_firing_count": int(example_level_firings),
            }
        )
    return rows


def _top_weight_rows_for_key(
    key_tuple: Tuple[str, str, str, str],
    key_rule_ids: Set[str],
    lr_weight_rows: Sequence[Dict[str, Any]],
    top_rules_per_key: int,
) -> List[Dict[str, Any]]:
    entry_kind, predicate_name, state_name, display_name = key_tuple
    positive_rows = sorted(
        [
            dict(row)
            for row in lr_weight_rows
            if str(row.get("rule_id", "")) in key_rule_ids and _safe_float(row.get("weight", 0.0)) > 0.0
        ],
        key=lambda row: (-_safe_float(row.get("weight", 0.0)), -_safe_float(row.get("abs_weight", 0.0)), str(row.get("rule_id", ""))),
    )[:top_rules_per_key]
    negative_rows = sorted(
        [
            dict(row)
            for row in lr_weight_rows
            if str(row.get("rule_id", "")) in key_rule_ids and _safe_float(row.get("weight", 0.0)) < 0.0
        ],
        key=lambda row: (_safe_float(row.get("weight", 0.0)), -_safe_float(row.get("abs_weight", 0.0)), str(row.get("rule_id", ""))),
    )[:top_rules_per_key]

    output_rows: List[Dict[str, Any]] = []
    for sign_name, signed_rows in (("positive", positive_rows), ("negative", negative_rows)):
        for rank, row in enumerate(signed_rows, start=1):
            output_rows.append(
                {
                    "entry_kind": entry_kind,
                    "predicate_name": predicate_name,
                    "state_name": state_name,
                    "display_name": display_name,
                    "sign": sign_name,
                    "rank": rank,
                    "rule_id": str(row.get("rule_id", "")),
                    "weight": _safe_float(row.get("weight", 0.0)),
                    "abs_weight": _safe_float(row.get("abs_weight", 0.0)),
                    "confidence": _safe_float(row.get("confidence", 0.0)),
                    "train_positive_support": _safe_int(row.get("train_positive_support", 0)),
                    "train_negative_support": _safe_int(row.get("train_negative_support", 0)),
                    "semantic_family": str(row.get("semantic_family", "")),
                    "summary_families": str(row.get("summary_families", "")),
                    "clause": str(row.get("clause", "")),
                }
            )
    return output_rows


def _weighted_match_rows_for_key(
    key_tuple: Tuple[str, str, str, str],
    key_rule_ids: Set[str],
    lr_weight_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    entry_kind, predicate_name, state_name, display_name = key_tuple
    rows: List[Dict[str, Any]] = []
    matched_rows = [
        dict(row)
        for row in lr_weight_rows
        if str(row.get("rule_id", "")) in key_rule_ids
    ]
    matched_rows = sorted(
        matched_rows,
        key=lambda row: (
            -_safe_float(row.get("abs_weight", 0.0)),
            -_safe_float(row.get("weight", 0.0)),
            str(row.get("rule_id", "")),
        ),
    )
    for row in matched_rows:
        rows.append(
            {
                "entry_kind": entry_kind,
                "predicate_name": predicate_name,
                "state_name": state_name,
                "display_name": display_name,
                "rule_id": str(row.get("rule_id", "")),
                "rank": _safe_int(row.get("rank", 0)),
                "sign": str(row.get("sign", "")),
                "weight": _safe_float(row.get("weight", 0.0)),
                "abs_weight": _safe_float(row.get("abs_weight", 0.0)),
                "confidence": _safe_float(row.get("confidence", 0.0)),
                "train_positive_support": _safe_int(row.get("train_positive_support", 0)),
                "train_negative_support": _safe_int(row.get("train_negative_support", 0)),
                "semantic_family": str(row.get("semantic_family", "")),
                "summary_families": str(row.get("summary_families", "")),
                "clause": str(row.get("clause", "")),
            }
        )
    return rows


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / max(1, len(values)))


def process_diagnostic(
    extended_rule_results: Dict[str, Any],
    eval_temporal_rule_results: Sequence[Dict[str, Any]],
    rule_results_by_name: Dict[str, Dict[str, Any]],
    rule_aggregation_baseline_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_json_path = out_root / "traffic_control_rule_utility_summary.json"
    traffic_light_state_counts_csv_path = out_root / "traffic_light_state_rule_counts.csv"
    selected_rule_matches_csv_path = out_root / "traffic_control_selected_rule_matches.csv"
    weighted_rule_matches_csv_path = out_root / "traffic_control_weighted_rule_matches.csv"
    firing_by_error_type_csv_path = out_root / "traffic_control_firing_by_error_type.csv"

    if not force_recompute and summary_json_path.exists():
        with summary_json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _TRAFFIC_CONTROL_RULE_UTILITY_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {summary_json_path.name}")
            return cached

    sparse, _, _, _, _ = _import_ml_dependencies()
    prepared_rules = _prepare_rules(extended_rule_results, cfg)
    eval_examples = _prepare_examples(eval_temporal_rule_results)
    if not prepared_rules:
        raise RuntimeError("Traffic-control rule utility diagnostic found no Step 16 rules to inspect.")
    if not eval_examples:
        raise RuntimeError("Traffic-control rule utility diagnostic found no evaluation examples to inspect.")

    eval_matrix = _build_rule_firing_matrix(eval_examples, prepared_rules, sparse)
    rule_id_to_index = {str(rule.get("rule_id", "")): index for index, rule in enumerate(prepared_rules)}
    rule_stats, fired_example_ids_by_rule, _ = _rule_stats_from_matrix(prepared_rules, eval_examples, eval_matrix)
    rule_key_membership = _rule_key_map(prepared_rules)

    rule_set_predictions: Dict[str, Dict[str, Any]] = {}
    for rule_set_name, _ in _RULE_SET_ORDER:
        selected_rule_ids = [
            str(rule.get("rule_id", ""))
            for rule in list(rule_results_by_name.get(rule_set_name, {}).get("final_rules", []))
            if str(rule.get("rule_id", ""))
        ]
        rule_set_predictions[rule_set_name] = _selected_rule_prediction_sets(
            eval_examples=eval_examples,
            eval_matrix=eval_matrix,
            rule_id_to_index=rule_id_to_index,
            selected_rule_ids=selected_rule_ids,
        )

    lr_weight_rows = _read_all_nonzero_weight_rows(rule_aggregation_baseline_results)
    lr_prediction_sets = _read_lr_prediction_sets(rule_aggregation_baseline_results)

    forced_predicates = {
        _normalize_text(value)
        for value in cfg.get("highlight_predicates", list(_DEFAULT_HIGHLIGHT_PREDICATES))
        if _normalize_text(value)
    }
    forced_states = {
        _normalize_text(value)
        for value in cfg.get("tracked_states", list(_DEFAULT_STATE_VALUES))
        if _normalize_text(value)
    }

    discovered_predicates: Set[str] = set(forced_predicates)
    discovered_states: Set[str] = set(forced_states)
    for rule in prepared_rules:
        for entry_kind, predicate_name, state_name, _ in _rule_key_rows(rule):
            discovered_predicates.add(predicate_name)
            if entry_kind == "state" and state_name:
                discovered_states.add(state_name)

    key_tuples: List[Tuple[str, str, str, str]] = []
    for predicate_name in sorted(discovered_predicates):
        key_tuples.append(("predicate", predicate_name, "", predicate_name))
    for state_name in sorted(discovered_states):
        key_tuples.append(("state", "traffic_light_state", state_name, f"traffic_light_state={state_name}"))

    actual_positive_eval = sum(1 for example in eval_examples if bool(example.get("label", False)))
    positive_eval_example_ids = {
        str(example.get("example_id", ""))
        for example in eval_examples
        if bool(example.get("label", False))
    }
    negative_eval_example_ids = {
        str(example.get("example_id", ""))
        for example in eval_examples
        if not bool(example.get("label", False))
    }
    summary_rows: List[Dict[str, Any]] = []
    firing_stat_rows: List[Dict[str, Any]] = []
    weighted_rule_match_rows: List[Dict[str, Any]] = []
    selected_rule_match_rows: List[Dict[str, Any]] = []
    details_by_key: Dict[str, Any] = {}

    for key_tuple in key_tuples:
        key_id = "||".join(key_tuple)
        matching_rule_ids = {
            rule_id for rule_id, key_ids in rule_key_membership.items() if key_id in key_ids
        }
        matching_rules = [dict(prepared_rules[rule_id_to_index[rule_id]]) for rule_id in sorted(matching_rule_ids) if rule_id in rule_id_to_index]
        positive_supports = [float(rule.get("positive_support", 0)) for rule in matching_rules]
        negative_supports = [float(rule.get("negative_support", 0)) for rule in matching_rules]
        confidences = [float(rule.get("confidence", 0.0)) for rule in matching_rules]
        eval_recalls = [float(rule_stats.get(str(rule.get("rule_id", "")), {}).get("eval_recall", 0.0)) for rule in matching_rules]
        eval_positive_cover = set()
        eval_negative_cover = set()
        for rule_id in matching_rule_ids:
            eval_positive_cover.update(
                set(fired_example_ids_by_rule.get(rule_id, set())) & positive_eval_example_ids
            )
            eval_negative_cover.update(
                set(fired_example_ids_by_rule.get(rule_id, set())) & negative_eval_example_ids
            )

        source_rule_ids = _source_rule_ids_for_key(
            key_id=key_id,
            key_rule_ids=matching_rule_ids,
            rule_set_predictions=rule_set_predictions,
            lr_weight_rows=lr_weight_rows,
        )
        for rule_set_name, stage_name in _RULE_SET_ORDER:
            firing_stat_rows.extend(
                _firing_stat_rows_for_source(
                    key_tuple=key_tuple,
                    source_name=stage_name,
                    matched_rule_ids=source_rule_ids.get(stage_name, set()),
                    outcome_example_ids=rule_set_predictions.get(rule_set_name, {}).get("outcome_example_ids", {}),
                    fired_example_ids_by_rule=fired_example_ids_by_rule,
                )
            )
        for source_name in _LR_SOURCE_ORDER:
            firing_stat_rows.extend(
                _firing_stat_rows_for_source(
                    key_tuple=key_tuple,
                    source_name=source_name,
                    matched_rule_ids=source_rule_ids.get(source_name, set()),
                    outcome_example_ids=lr_prediction_sets.get("outcome_example_ids", {}),
                    fired_example_ids_by_rule=fired_example_ids_by_rule,
                )
            )

        top_rows = _top_weight_rows_for_key(
            key_tuple=key_tuple,
            key_rule_ids=matching_rule_ids,
            lr_weight_rows=lr_weight_rows,
            top_rules_per_key=max(1, int(cfg.get("top_rules_per_key", 5))),
        )
        weighted_rule_match_rows.extend(
            _weighted_match_rows_for_key(
                key_tuple=key_tuple,
                key_rule_ids=matching_rule_ids,
                lr_weight_rows=lr_weight_rows,
            )
        )
        positive_top = next((row for row in top_rows if str(row.get("sign", "")) == "positive"), {})
        negative_top = next((row for row in top_rows if str(row.get("sign", "")) == "negative"), {})

        for _, stage_name in _RULE_SET_ORDER:
            selected_rule_ids = sorted(source_rule_ids.get(stage_name, set()))
            for rule_id in selected_rule_ids:
                rule = prepared_rules[rule_id_to_index[rule_id]]
                selected_rule_match_rows.append(
                    {
                        "entry_kind": key_tuple[0],
                        "predicate_name": key_tuple[1],
                        "state_name": key_tuple[2],
                        "display_name": key_tuple[3],
                        "source_name": stage_name,
                        "rule_id": rule_id,
                        "confidence": _safe_float(rule.get("confidence", 0.0)),
                        "positive_support": _safe_int(rule.get("positive_support", 0)),
                        "negative_support": _safe_int(rule.get("negative_support", 0)),
                        "total_support": _safe_int(rule.get("total_support", 0)),
                        "eval_recall": _safe_float(rule_stats.get(rule_id, {}).get("eval_recall", 0.0)),
                        "clause": str(rule.get("clause", "")),
                    }
                )

        summary_rows.append(
            {
                "entry_kind": key_tuple[0],
                "predicate_name": key_tuple[1],
                "state_name": key_tuple[2],
                "display_name": key_tuple[3],
                "step16_rule_pool_rule_count": len(matching_rule_ids),
                "step16_mean_positive_support": _mean(positive_supports),
                "step16_max_positive_support": max(positive_supports) if positive_supports else 0.0,
                "step16_mean_negative_support": _mean(negative_supports),
                "step16_max_negative_support": max(negative_supports) if negative_supports else 0.0,
                "step16_mean_confidence": _mean(confidences),
                "step16_max_confidence": max(confidences) if confidences else 0.0,
                "step16_mean_eval_recall": _mean(eval_recalls),
                "step16_max_eval_recall": max(eval_recalls) if eval_recalls else 0.0,
                "step16_eval_positive_example_coverage": len(eval_positive_cover),
                "step16_eval_negative_example_coverage": len(eval_negative_cover),
                "step17_original_selected_rule_count": len(source_rule_ids.get("step17_original", set())),
                "step17b_diverse_selected_rule_count": len(source_rule_ids.get("step17b_diverse", set())),
                "step17b2_semantic_constrained_selected_rule_count": len(
                    source_rule_ids.get("step17b2_semantic_constrained", set())
                ),
                "step17c_coverage_family_aware_selected_rule_count": len(
                    source_rule_ids.get("step17c_coverage_family_aware", set())
                ),
                "step18c_nonzero_weight_rule_count": len(source_rule_ids.get("step18c_lr_nonzero_all", set())),
                "step18c_positive_weight_rule_count": len(source_rule_ids.get("step18c_lr_nonzero_positive", set())),
                "step18c_negative_weight_rule_count": len(source_rule_ids.get("step18c_lr_nonzero_negative", set())),
                "step18c_top_positive_rule_id": str(positive_top.get("rule_id", "")),
                "step18c_top_positive_weight": _safe_float(positive_top.get("weight", 0.0)),
                "step18c_top_negative_rule_id": str(negative_top.get("rule_id", "")),
                "step18c_top_negative_weight": _safe_float(negative_top.get("weight", 0.0)),
            }
        )
        details_by_key[key_tuple[3]] = {
            "matching_rule_ids": sorted(matching_rule_ids),
            "top_weighted_rules": top_rows,
            "weighted_rule_matches": [
                row for row in weighted_rule_match_rows if str(row.get("display_name", "")) == key_tuple[3]
            ],
            "prediction_sources": {
                stage_name: {
                    "matched_rule_ids": sorted(source_rule_ids.get(stage_name, set())),
                    "selected_rule_count": len(source_rule_ids.get(stage_name, set())),
                }
                for _, stage_name in _RULE_SET_ORDER
            },
            "lr_sources": {
                source_name: {
                    "matched_rule_ids": sorted(source_rule_ids.get(source_name, set())),
                    "selected_rule_count": len(source_rule_ids.get(source_name, set())),
                }
                for source_name in _LR_SOURCE_ORDER
            },
        }

    state_summary_rows = [
        dict(row)
        for row in summary_rows
        if str(row.get("entry_kind", "")) == "state"
    ]
    _write_csv(
        traffic_light_state_counts_csv_path,
        [
            "entry_kind",
            "predicate_name",
            "state_name",
            "display_name",
            "step16_rule_pool_rule_count",
            "step16_mean_positive_support",
            "step16_max_positive_support",
            "step16_mean_negative_support",
            "step16_max_negative_support",
            "step16_mean_confidence",
            "step16_max_confidence",
            "step16_mean_eval_recall",
            "step16_max_eval_recall",
            "step16_eval_positive_example_coverage",
            "step16_eval_negative_example_coverage",
            "step17_original_selected_rule_count",
            "step17b_diverse_selected_rule_count",
            "step17b2_semantic_constrained_selected_rule_count",
            "step17c_coverage_family_aware_selected_rule_count",
            "step18c_nonzero_weight_rule_count",
            "step18c_positive_weight_rule_count",
            "step18c_negative_weight_rule_count",
            "step18c_top_positive_rule_id",
            "step18c_top_positive_weight",
            "step18c_top_negative_rule_id",
            "step18c_top_negative_weight",
        ],
        state_summary_rows,
    )
    _write_csv(
        firing_by_error_type_csv_path,
        [
            "entry_kind",
            "predicate_name",
            "state_name",
            "display_name",
            "source_name",
            "outcome_bucket",
            "matched_rule_count",
            "covered_example_count",
            "example_level_rule_firing_count",
        ],
        firing_stat_rows,
    )
    _write_csv(
        selected_rule_matches_csv_path,
        [
            "entry_kind",
            "predicate_name",
            "state_name",
            "display_name",
            "source_name",
            "rule_id",
            "confidence",
            "positive_support",
            "negative_support",
            "total_support",
            "eval_recall",
            "clause",
        ],
        selected_rule_match_rows,
    )
    _write_csv(
        weighted_rule_matches_csv_path,
        [
            "entry_kind",
            "predicate_name",
            "state_name",
            "display_name",
            "rule_id",
            "rank",
            "sign",
            "weight",
            "abs_weight",
            "confidence",
            "train_positive_support",
            "train_negative_support",
            "semantic_family",
            "summary_families",
            "clause",
        ],
        weighted_rule_match_rows,
    )

    traffic_light_state_predicate_row = next(
        (row for row in summary_rows if str(row.get("display_name", "")) == "traffic_light_state"),
        {},
    )
    traffic_light_state_usage_summary = {
        "predicate_in_rule_pool": bool(_safe_int(traffic_light_state_predicate_row.get("step16_rule_pool_rule_count", 0)) > 0),
        "predicate_used_by_any_final_rule_selection": bool(
            _safe_int(traffic_light_state_predicate_row.get("step17_original_selected_rule_count", 0)) > 0
            or _safe_int(traffic_light_state_predicate_row.get("step17b_diverse_selected_rule_count", 0)) > 0
            or _safe_int(traffic_light_state_predicate_row.get("step17b2_semantic_constrained_selected_rule_count", 0)) > 0
            or _safe_int(traffic_light_state_predicate_row.get("step17c_coverage_family_aware_selected_rule_count", 0)) > 0
        ),
        "predicate_used_by_learned_rule_aggregation": bool(
            _safe_int(traffic_light_state_predicate_row.get("step18c_nonzero_weight_rule_count", 0)) > 0
        ),
        "answer": (
            "traffic-light state predicates are only present in the Step 16 rule pool"
            if _safe_int(traffic_light_state_predicate_row.get("step16_rule_pool_rule_count", 0)) > 0
            and _safe_int(traffic_light_state_predicate_row.get("step17_original_selected_rule_count", 0)) == 0
            and _safe_int(traffic_light_state_predicate_row.get("step17b_diverse_selected_rule_count", 0)) == 0
            and _safe_int(traffic_light_state_predicate_row.get("step17b2_semantic_constrained_selected_rule_count", 0)) == 0
            and _safe_int(traffic_light_state_predicate_row.get("step17c_coverage_family_aware_selected_rule_count", 0)) == 0
            and _safe_int(traffic_light_state_predicate_row.get("step18c_nonzero_weight_rule_count", 0)) == 0
            else "traffic-light state predicates are used beyond mere rule-pool presence"
        ),
        "by_state": {
            str(row.get("state_name", "")): {
                "in_rule_pool": bool(_safe_int(row.get("step16_rule_pool_rule_count", 0)) > 0),
                "used_by_any_final_rule_selection": bool(
                    _safe_int(row.get("step17_original_selected_rule_count", 0)) > 0
                    or _safe_int(row.get("step17b_diverse_selected_rule_count", 0)) > 0
                    or _safe_int(row.get("step17b2_semantic_constrained_selected_rule_count", 0)) > 0
                    or _safe_int(row.get("step17c_coverage_family_aware_selected_rule_count", 0)) > 0
                ),
                "used_by_learned_rule_aggregation": bool(
                    _safe_int(row.get("step18c_nonzero_weight_rule_count", 0)) > 0
                ),
                "selected_rule_count_total": (
                    _safe_int(row.get("step17_original_selected_rule_count", 0))
                    + _safe_int(row.get("step17b_diverse_selected_rule_count", 0))
                    + _safe_int(row.get("step17b2_semantic_constrained_selected_rule_count", 0))
                    + _safe_int(row.get("step17c_coverage_family_aware_selected_rule_count", 0))
                ),
                "lr_nonzero_rule_count": _safe_int(row.get("step18c_nonzero_weight_rule_count", 0)),
            }
            for row in state_summary_rows
        },
    }

    summary = {
        "version": _TRAFFIC_CONTROL_RULE_UTILITY_VERSION,
        "config": _cfg_key_subset(cfg),
        "num_eval_examples": len(eval_examples),
        "num_eval_positive_examples": actual_positive_eval,
        "traffic_light_state_usage_summary": traffic_light_state_usage_summary,
        "prediction_sources": {
            rule_set_name: dict(payload.get("metrics", {})) for rule_set_name, payload in rule_set_predictions.items()
        },
        "lr_prediction_metrics": dict(lr_prediction_sets.get("metrics", {})),
        "rows": summary_rows,
        "details_by_key": details_by_key,
        "output_paths": {
            "summary_json": str(summary_json_path),
            "traffic_light_state_rule_counts_csv": str(traffic_light_state_counts_csv_path),
            "traffic_control_selected_rule_matches_csv": str(selected_rule_matches_csv_path),
            "traffic_control_weighted_rule_matches_csv": str(weighted_rule_matches_csv_path),
            "traffic_control_firing_by_error_type_csv": str(firing_by_error_type_csv_path),
        },
    }
    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        "  traffic_control_rule_utility_diagnostic: "
        f"keys={len(summary_rows)} | "
        f"pool_rules={sum(int(row.get('step16_rule_pool_rule_count', 0)) for row in summary_rows)} | "
        f"lr_nonzero={sum(int(row.get('step18c_nonzero_weight_rule_count', 0)) for row in summary_rows)}"
    )
    print(f"Traffic-control rule utility summary JSON written to {summary_json_path}")
    print(f"Traffic-light-state rule counts CSV written to {traffic_light_state_counts_csv_path}")
    print(f"Traffic-control selected-rule matches CSV written to {selected_rule_matches_csv_path}")
    print(f"Traffic-control weighted-rule matches CSV written to {weighted_rule_matches_csv_path}")
    print(f"Traffic-control firing-by-error-type CSV written to {firing_by_error_type_csv_path}")
    return summary


def run(
    extended_rule_results: Dict[str, Any],
    eval_temporal_rule_results: Sequence[Dict[str, Any]],
    rule_results_by_name: Dict[str, Dict[str, Any]],
    rule_aggregation_baseline_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_diagnostic(
        extended_rule_results=extended_rule_results,
        eval_temporal_rule_results=eval_temporal_rule_results,
        rule_results_by_name=rule_results_by_name,
        rule_aggregation_baseline_results=rule_aggregation_baseline_results,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
