"""
Post-hoc rule-level causal masking over Step 18 final rule evaluation outputs.

This module does not rerun perception, atom extraction, rule matching, or rule
training. It starts from Step 18 example-level triggered rule ids, masks one
triggered rule at a time, and reuses the Step 18 aggregation logic to measure
how much each triggered rule affected the final prediction.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config
from src.exp_driving_videos.modules.evaluate_rules_driving_mini import aggregate_rule_prediction


_MASKING_VERSION = 1


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "18m_driving_mini_rule_level_causal_masking"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prediction_mode": str(cfg.get("prediction_mode", "any_rule_positive")),
        "rule_set_name": str(cfg.get("rule_set_name", "")),
        "score_epsilon": float(cfg.get("score_epsilon", 1e-9)),
        "many_redundant_rule_fraction": float(cfg.get("many_redundant_rule_fraction", 0.5)),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return []


def _write_csv(path: Path, fieldnames: List[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _load_example_prediction_rows(evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    csv_path = Path(str(evaluation_results.get("example_predictions_csv_path", "")))
    if not csv_path.exists():
        raise FileNotFoundError(
            "Rule-level causal masking expected Step 18 example_predictions.csv. "
            f"Missing: {csv_path}"
        )

    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "video_id": str(row.get("video_id", "")),
                    "example_id": str(row.get("example_id", "")),
                    "label": _as_bool(row.get("label", False)),
                    "step18_predicted_positive": _as_bool(row.get("predicted_positive", False)),
                    "matching_rule_ids": _json_list(row.get("matching_rule_ids", "")),
                }
            )
    return rows


def _build_rule_lookup(
    final_rule_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    rule_lookup: Dict[str, Dict[str, Any]] = {}
    for rule in list(final_rule_results.get("final_rules", [])):
        rule_id = str(rule.get("rule_id", "")).strip()
        if rule_id:
            rule_lookup[rule_id] = dict(rule)
    for rule in list(evaluation_results.get("rule_evaluations", [])):
        rule_id = str(rule.get("rule_id", "")).strip()
        if rule_id:
            rule_lookup[rule_id] = {**rule_lookup.get(rule_id, {}), **dict(rule)}
    return rule_lookup


def _classify_masking_effect(
    *,
    label: bool,
    original_predicted_positive: bool,
    masked_predicted_positive: bool,
    score_delta: float,
    score_epsilon: float,
) -> str:
    original_correct = original_predicted_positive == label
    masked_correct = masked_predicted_positive == label
    if original_correct and not masked_correct:
        return "helpful_for_correct_prediction"
    if (not original_correct) and masked_correct:
        return "harmful_causal_source"
    if abs(score_delta) > score_epsilon:
        return "non_decisive_contribution"
    return "redundant_trigger"


def _dominant_influence(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return counter.most_common(1)[0][0]


def _rule_summary_rows(
    example_rule_rows: Sequence[Dict[str, Any]],
    rule_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aggregates: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "trigger_count": 0,
            "prediction_flip_count": 0,
            "helpful_count": 0,
            "harmful_count": 0,
            "non_decisive_contribution_count": 0,
            "redundant_count": 0,
            "necessary_true_positive_count": 0,
            "causal_false_positive_count": 0,
            "score_delta_sum": 0.0,
            "score_delta_max": 0.0,
            "influence_counter": Counter(),
            "example_ids": [],
        }
    )
    for row in example_rule_rows:
        rule_id = str(row.get("masked_rule_id", ""))
        aggregate = aggregates[rule_id]
        influence_type = str(row.get("influence_type", ""))
        score_delta = float(row.get("score_delta", 0.0))
        aggregate["trigger_count"] += 1
        aggregate["prediction_flip_count"] += int(bool(row.get("prediction_flipped", False)))
        aggregate["helpful_count"] += int(influence_type == "helpful_for_correct_prediction")
        aggregate["harmful_count"] += int(influence_type == "harmful_causal_source")
        aggregate["non_decisive_contribution_count"] += int(influence_type == "non_decisive_contribution")
        aggregate["redundant_count"] += int(influence_type == "redundant_trigger")
        aggregate["necessary_true_positive_count"] += int(
            influence_type == "helpful_for_correct_prediction"
            and bool(row.get("label", False))
            and bool(row.get("original_predicted_positive", False))
        )
        aggregate["causal_false_positive_count"] += int(
            influence_type == "harmful_causal_source"
            and (not bool(row.get("label", False)))
            and bool(row.get("original_predicted_positive", False))
        )
        aggregate["score_delta_sum"] += score_delta
        aggregate["score_delta_max"] = max(float(aggregate["score_delta_max"]), score_delta)
        aggregate["influence_counter"][influence_type] += 1
        if len(aggregate["example_ids"]) < 50:
            aggregate["example_ids"].append(str(row.get("example_id", "")))

    rows: List[Dict[str, Any]] = []
    for rule_id, aggregate in sorted(
        aggregates.items(),
        key=lambda item: (
            -int(item[1]["prediction_flip_count"]),
            -float(item[1]["score_delta_sum"]),
            item[0],
        ),
    ):
        rule = rule_lookup.get(rule_id, {})
        trigger_count = int(aggregate["trigger_count"])
        rows.append(
            {
                "rule_id": rule_id,
                "clause": str(rule.get("clause", "")),
                "candidate_rule_category": str(rule.get("candidate_rule_category", "")),
                "confidence": float(rule.get("confidence", 0.0)),
                "trigger_count": trigger_count,
                "prediction_flip_count": int(aggregate["prediction_flip_count"]),
                "prediction_flip_rate": float(aggregate["prediction_flip_count"] / max(1, trigger_count)),
                "helpful_count": int(aggregate["helpful_count"]),
                "harmful_count": int(aggregate["harmful_count"]),
                "non_decisive_contribution_count": int(aggregate["non_decisive_contribution_count"]),
                "redundant_count": int(aggregate["redundant_count"]),
                "necessary_true_positive_count": int(aggregate["necessary_true_positive_count"]),
                "causal_false_positive_count": int(aggregate["causal_false_positive_count"]),
                "score_delta_sum": float(aggregate["score_delta_sum"]),
                "score_delta_avg": float(aggregate["score_delta_sum"] / max(1, trigger_count)),
                "score_delta_max": float(aggregate["score_delta_max"]),
                "net_helpful_minus_harmful": int(aggregate["helpful_count"]) - int(aggregate["harmful_count"]),
                "dominant_influence_type": _dominant_influence(aggregate["influence_counter"]),
                "example_ids_sample": json.dumps(aggregate["example_ids"]),
            }
        )
    return rows


def _example_summary_rows(
    example_rows: Sequence[Dict[str, Any]],
    example_rule_rows: Sequence[Dict[str, Any]],
    *,
    many_redundant_rule_fraction: float,
) -> List[Dict[str, Any]]:
    rows_by_example = {str(row.get("example_id", "")): dict(row) for row in example_rows}
    masked_by_example: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in example_rule_rows:
        masked_by_example[str(row.get("example_id", ""))].append(dict(row))

    summary_rows: List[Dict[str, Any]] = []
    for example_id in sorted(rows_by_example):
        example = rows_by_example[example_id]
        masked_rows = masked_by_example.get(example_id, [])
        influence_counter = Counter(str(row.get("influence_type", "")) for row in masked_rows)
        flip_rows = [row for row in masked_rows if bool(row.get("prediction_flipped", False))]
        key_candidates = flip_rows or masked_rows
        key_row = max(
            key_candidates,
            key=lambda row: (
                bool(row.get("prediction_flipped", False)),
                float(row.get("score_delta", 0.0)),
                str(row.get("masked_rule_id", "")),
            ),
            default={},
        )
        num_triggered_rules = len(masked_rows)
        redundant_count = int(influence_counter.get("redundant_trigger", 0))
        redundant_fraction = float(redundant_count / max(1, num_triggered_rules))
        causal_error_rule_ids = [
            str(row.get("masked_rule_id", ""))
            for row in masked_rows
            if str(row.get("influence_type", "")) == "harmful_causal_source"
        ]
        necessary_support_rule_ids = [
            str(row.get("masked_rule_id", ""))
            for row in masked_rows
            if str(row.get("influence_type", "")) == "helpful_for_correct_prediction"
        ]
        summary_rows.append(
            {
                "video_id": str(example.get("video_id", "")),
                "example_id": example_id,
                "label": bool(example.get("label", False)),
                "original_predicted_positive": bool(example.get("original_predicted_positive", False)),
                "original_correct": bool(example.get("original_correct", False)),
                "original_score": float(example.get("original_score", 0.0)),
                "num_triggered_rules": num_triggered_rules,
                "num_prediction_flip_rules": len(flip_rows),
                "num_helpful_rules": int(influence_counter.get("helpful_for_correct_prediction", 0)),
                "num_harmful_rules": int(influence_counter.get("harmful_causal_source", 0)),
                "num_non_decisive_contribution_rules": int(
                    influence_counter.get("non_decisive_contribution", 0)
                ),
                "num_redundant_rules": redundant_count,
                "redundant_fraction": redundant_fraction,
                "has_key_causal_rule": bool(flip_rows),
                "has_many_redundant_rules": bool(
                    num_triggered_rules > 0 and redundant_fraction >= many_redundant_rule_fraction
                ),
                "key_causal_rule_id": str(key_row.get("masked_rule_id", "")),
                "key_causal_influence_type": str(key_row.get("influence_type", "none")),
                "max_score_delta": float(key_row.get("score_delta", 0.0)),
                "causal_error_rule_ids": json.dumps(causal_error_rule_ids),
                "necessary_support_rule_ids": json.dumps(necessary_support_rule_ids),
            }
        )
    return summary_rows


def process_masking(
    final_rule_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    result_path = out_root / "rule_level_causal_masking.json"
    example_rule_path = out_root / "example_rule_masking.csv"
    rule_summary_path = out_root / "rule_causal_summary.csv"
    example_summary_path = out_root / "example_causal_summary.csv"

    cfg_subset = _cfg_key_subset(cfg)
    if not force_recompute and result_path.exists():
        with result_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _MASKING_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == cfg_subset:
            print(f"  [cache] loading {result_path.name}")
            return cached

    prediction_mode = str(cfg.get("prediction_mode", "any_rule_positive"))
    score_epsilon = float(cfg.get("score_epsilon", 1e-9))
    many_redundant_rule_fraction = float(cfg.get("many_redundant_rule_fraction", 0.5))
    rule_lookup = _build_rule_lookup(final_rule_results, evaluation_results)
    raw_example_rows = _load_example_prediction_rows(evaluation_results)

    example_rows: List[Dict[str, Any]] = []
    example_rule_rows: List[Dict[str, Any]] = []
    for row in raw_example_rows:
        example_id = str(row.get("example_id", ""))
        triggered_rule_ids = [rule_id for rule_id in list(row.get("matching_rule_ids", [])) if rule_id in rule_lookup]
        original = aggregate_rule_prediction(
            triggered_rule_ids,
            rule_lookup,
            prediction_mode=prediction_mode,
        )
        original_predicted_positive = bool(original.get("predicted_positive", False))
        label = bool(row.get("label", False))
        original_score = float(original.get("prediction_score", 0.0))
        example_rows.append(
            {
                **row,
                "matching_rule_ids": triggered_rule_ids,
                "original_predicted_positive": original_predicted_positive,
                "original_score": original_score,
                "original_correct": original_predicted_positive == label,
            }
        )
        for masked_rule_id in triggered_rule_ids:
            masked_triggered_rule_ids = [
                rule_id for rule_id in triggered_rule_ids if str(rule_id) != str(masked_rule_id)
            ]
            masked = aggregate_rule_prediction(
                masked_triggered_rule_ids,
                rule_lookup,
                prediction_mode=prediction_mode,
            )
            masked_predicted_positive = bool(masked.get("predicted_positive", False))
            masked_score = float(masked.get("prediction_score", 0.0))
            score_delta = float(original_score - masked_score)
            influence_type = _classify_masking_effect(
                label=label,
                original_predicted_positive=original_predicted_positive,
                masked_predicted_positive=masked_predicted_positive,
                score_delta=score_delta,
                score_epsilon=score_epsilon,
            )
            rule = rule_lookup.get(masked_rule_id, {})
            example_rule_rows.append(
                {
                    "video_id": str(row.get("video_id", "")),
                    "example_id": example_id,
                    "label": label,
                    "masked_rule_id": masked_rule_id,
                    "masked_rule_confidence": float(rule.get("confidence", 0.0)),
                    "masked_rule_category": str(rule.get("candidate_rule_category", "")),
                    "original_predicted_positive": original_predicted_positive,
                    "masked_predicted_positive": masked_predicted_positive,
                    "original_correct": original_predicted_positive == label,
                    "masked_correct": masked_predicted_positive == label,
                    "original_score": original_score,
                    "masked_score": masked_score,
                    "score_delta": score_delta,
                    "prediction_flipped": original_predicted_positive != masked_predicted_positive,
                    "num_original_triggered_rules": len(triggered_rule_ids),
                    "num_masked_triggered_rules": len(masked_triggered_rule_ids),
                    "influence_type": influence_type,
                }
            )

    rule_rows = _rule_summary_rows(example_rule_rows, rule_lookup)
    example_summary_rows = _example_summary_rows(
        example_rows,
        example_rule_rows,
        many_redundant_rule_fraction=many_redundant_rule_fraction,
    )

    _write_csv(
        example_rule_path,
        [
            "video_id",
            "example_id",
            "label",
            "masked_rule_id",
            "masked_rule_confidence",
            "masked_rule_category",
            "original_predicted_positive",
            "masked_predicted_positive",
            "original_correct",
            "masked_correct",
            "original_score",
            "masked_score",
            "score_delta",
            "prediction_flipped",
            "num_original_triggered_rules",
            "num_masked_triggered_rules",
            "influence_type",
        ],
        example_rule_rows,
    )
    _write_csv(
        rule_summary_path,
        [
            "rule_id",
            "clause",
            "candidate_rule_category",
            "confidence",
            "trigger_count",
            "prediction_flip_count",
            "prediction_flip_rate",
            "helpful_count",
            "harmful_count",
            "non_decisive_contribution_count",
            "redundant_count",
            "necessary_true_positive_count",
            "causal_false_positive_count",
            "score_delta_sum",
            "score_delta_avg",
            "score_delta_max",
            "net_helpful_minus_harmful",
            "dominant_influence_type",
            "example_ids_sample",
        ],
        rule_rows,
    )
    _write_csv(
        example_summary_path,
        [
            "video_id",
            "example_id",
            "label",
            "original_predicted_positive",
            "original_correct",
            "original_score",
            "num_triggered_rules",
            "num_prediction_flip_rules",
            "num_helpful_rules",
            "num_harmful_rules",
            "num_non_decisive_contribution_rules",
            "num_redundant_rules",
            "redundant_fraction",
            "has_key_causal_rule",
            "has_many_redundant_rules",
            "key_causal_rule_id",
            "key_causal_influence_type",
            "max_score_delta",
            "causal_error_rule_ids",
            "necessary_support_rule_ids",
        ],
        example_summary_rows,
    )

    influence_counter = Counter(str(row.get("influence_type", "")) for row in example_rule_rows)
    result = {
        "version": _MASKING_VERSION,
        "config": cfg_subset,
        "num_examples": len(example_rows),
        "num_triggered_example_rules": len(example_rule_rows),
        "num_rules_with_triggered_masks": len(rule_rows),
        "num_prediction_flips": int(influence_counter.get("helpful_for_correct_prediction", 0))
        + int(influence_counter.get("harmful_causal_source", 0)),
        "num_helpful_masks": int(influence_counter.get("helpful_for_correct_prediction", 0)),
        "num_harmful_masks": int(influence_counter.get("harmful_causal_source", 0)),
        "num_non_decisive_contribution_masks": int(
            influence_counter.get("non_decisive_contribution", 0)
        ),
        "num_redundant_masks": int(influence_counter.get("redundant_trigger", 0)),
        "output_paths": {
            "result_json": str(result_path),
            "example_rule_masking_csv": str(example_rule_path),
            "rule_causal_summary_csv": str(rule_summary_path),
            "example_causal_summary_csv": str(example_summary_path),
        },
    }
    with result_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(
        "  rule_level_causal_masking: "
        f"examples={len(example_rows)} | "
        f"triggered_masks={len(example_rule_rows)} | "
        f"flips={result['num_prediction_flips']} | "
        f"redundant={result['num_redundant_masks']}"
    )
    return result


def run(
    final_rule_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_masking(
        final_rule_results=final_rule_results,
        evaluation_results=evaluation_results,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
