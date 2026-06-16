"""
Evaluate final driving_mini rules on held-out temporal rule examples.

Consumes:
  - Step 13 output: temporal rule-learning examples for evaluation videos
  - Step 17 output: final rules learned from the training split

Output layout:
    pipeline_output/18_driving_mini_rule_evaluation/
        rule_evaluation.json
        rule_evaluation.csv
        example_predictions.csv
        rule_evaluation_summary.pdf
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_RULE_EVALUATION_VERSION = 2


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "18_driving_mini_rule_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prediction_mode": str(cfg.get("prediction_mode", "any_rule_positive")),
        "evaluated_rule_set_name": str(cfg.get("evaluated_rule_set_name", "")),
    }


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


def _extract_bindings(
    body_atom_template: str,
    concrete_atom: str,
) -> Optional[Dict[str, str]]:
    template_parsed = _parse_atom(body_atom_template)
    concrete_parsed = _parse_atom(concrete_atom)
    if template_parsed is None or concrete_parsed is None:
        return None

    template_predicate, template_args = template_parsed
    concrete_predicate, concrete_args = concrete_parsed
    if template_predicate != concrete_predicate or len(template_args) != len(concrete_args):
        return None

    bindings: Dict[str, str] = {}
    for template_arg, concrete_arg in zip(template_args, concrete_args):
        if template_arg in {"S", "O", "T", "F"}:
            existing = bindings.get(template_arg)
            if existing is not None and existing != concrete_arg:
                return None
            bindings[template_arg] = concrete_arg
            continue
        if template_arg != concrete_arg:
            return None
    return bindings


def _bindings_compatible(
    left: Dict[str, str],
    right: Dict[str, str],
) -> bool:
    for key, value in left.items():
        if key in right and str(right[key]) != str(value):
            return False
    return True


def _merge_bindings(
    left: Dict[str, str],
    right: Dict[str, str],
) -> Dict[str, str]:
    merged = {str(key): str(value) for key, value in left.items()}
    for key, value in right.items():
        merged[str(key)] = str(value)
    return merged


def _dedupe_match_states(
    states: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...]]] = set()
    for state in states:
        bindings = tuple(sorted((str(k), str(v)) for k, v in dict(state.get("bindings", {})).items()))
        matched_atoms = tuple(
            sorted((str(k), str(v)) for k, v in dict(state.get("matched_atoms", {})).items())
        )
        key = (bindings, matched_atoms)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(state)
    return deduped


def _find_rule_matches_for_example(
    body_atom_templates: Sequence[str],
    body_atoms: Sequence[str],
) -> List[Dict[str, Any]]:
    normalized_templates = [str(atom).strip() for atom in body_atom_templates if str(atom).strip()]
    normalized_body_atoms = [str(atom).strip() for atom in body_atoms if str(atom).strip()]
    if not normalized_templates or not normalized_body_atoms:
        return []

    match_states: List[Dict[str, Any]] = [{"bindings": {}, "matched_atoms": {}}]
    for template in normalized_templates:
        next_states: List[Dict[str, Any]] = []
        for state in match_states:
            state_bindings = dict(state.get("bindings", {}))
            state_matches = dict(state.get("matched_atoms", {}))
            for concrete_atom in normalized_body_atoms:
                new_bindings = _extract_bindings(template, concrete_atom)
                if new_bindings is None or not _bindings_compatible(state_bindings, new_bindings):
                    continue
                next_state = {
                    "bindings": _merge_bindings(state_bindings, new_bindings),
                    "matched_atoms": {**state_matches, template: concrete_atom},
                }
                next_states.append(next_state)
        match_states = _dedupe_match_states(next_states)
        if not match_states:
            return []
    return match_states


def _summarize_rule_evidence(
    evidence_entries: Sequence[Dict[str, Any]],
    total_positive_examples: int,
) -> Dict[str, Any]:
    total_firings = len(evidence_entries)
    positive_firings = sum(1 for entry in evidence_entries if bool(entry.get("label", False)))
    negative_firings = total_firings - positive_firings
    positive_example_ids = sorted(
        {
            str(entry.get("example_id", ""))
            for entry in evidence_entries
            if bool(entry.get("label", False)) and str(entry.get("example_id", ""))
        }
    )
    negative_example_ids = sorted(
        {
            str(entry.get("example_id", ""))
            for entry in evidence_entries
            if not bool(entry.get("label", False)) and str(entry.get("example_id", ""))
        }
    )
    precision = float(positive_firings / max(1, total_firings))
    recall = float(len(positive_example_ids) / max(1, total_positive_examples))
    return {
        "positive_support": len(positive_example_ids),
        "negative_support": len(negative_example_ids),
        "total_support": len(set(positive_example_ids) | set(negative_example_ids)),
        "positive_firings": positive_firings,
        "negative_firings": negative_firings,
        "total_firings": total_firings,
        "precision": precision,
        "recall": recall,
        "positive_example_ids": positive_example_ids,
        "negative_example_ids": negative_example_ids,
    }


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


def _get_rule_body_atom_templates(rule: Dict[str, Any]) -> List[str]:
    body_atom_templates = rule.get("body_atom_templates")
    if isinstance(body_atom_templates, list):
        return [str(atom).strip() for atom in body_atom_templates if str(atom).strip()]
    body_atom_template = str(rule.get("body_atom_template", "")).strip()
    return [body_atom_template] if body_atom_template else []


def _save_evaluation_pdf(
    result: Dict[str, Any],
    pdf_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = dict(result.get("overall_metrics", {}))
    per_video = list(result.get("per_video_metrics", []))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("Driving Mini Rule Evaluation", fontsize=16, fontweight="bold")

    confusion_ax = axes[0]
    tp = int(overall.get("true_positive", 0))
    fp = int(overall.get("false_positive", 0))
    fn = int(overall.get("false_negative", 0))
    tn = int(overall.get("true_negative", 0))
    confusion = [[tp, fn], [fp, tn]]
    image = confusion_ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=confusion_ax, fraction=0.046, pad=0.04)
    confusion_ax.set_xticks([0, 1], labels=["Actual +", "Actual -"])
    confusion_ax.set_yticks([0, 1], labels=["Pred +", "Pred -"])
    confusion_ax.set_title("Confusion Matrix", loc="left", fontweight="bold")
    for row_index, row in enumerate(confusion):
        for col_index, value in enumerate(row):
            confusion_ax.text(col_index, row_index, str(value), ha="center", va="center", color="black")

    metrics_ax = axes[1]
    metric_names = ["accuracy", "precision", "recall", "f1"]
    metric_values = [float(overall.get(name, 0.0)) for name in metric_names]
    metric_colors = ["#2a9d8f", "#457b9d", "#e76f51", "#264653"]
    metrics_ax.bar(metric_names, metric_values, color=metric_colors)
    metrics_ax.set_ylim(0.0, 1.1)
    metrics_ax.set_title("Overall Metrics", loc="left", fontweight="bold")
    metrics_ax.set_ylabel("Score")
    for idx, value in enumerate(metric_values):
        metrics_ax.text(idx, min(value + 0.03, 1.06), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    per_video_ax = axes[2]
    video_ids = [str(item.get("video_id", "")) for item in per_video]
    video_scores = [float(item.get("accuracy", 0.0)) for item in per_video]
    if video_ids:
        per_video_ax.bar(video_ids, video_scores, color="#8d99ae")
        per_video_ax.set_ylim(0.0, 1.1)
        per_video_ax.set_ylabel("Accuracy")
        per_video_ax.set_title("Per-Video Accuracy", loc="left", fontweight="bold")
        per_video_ax.tick_params(axis="x", rotation=25, labelsize=8)
        for idx, value in enumerate(video_scores):
            per_video_ax.text(idx, min(value + 0.03, 1.06), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    else:
        per_video_ax.axis("off")
        per_video_ax.text(0.5, 0.5, "No eval videos", ha="center", va="center")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def process_rules(
    final_rule_results: Dict[str, Any],
    temporal_rule_results: List[Dict[str, Any]],
    eval_video_ids: Optional[List[str]] = None,
    split_manifest: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    prediction_mode = str(cfg.get("prediction_mode", "any_rule_positive"))
    evaluated_rule_set_name = str(cfg.get("evaluated_rule_set_name", ""))
    if prediction_mode != "any_rule_positive":
        raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")

    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "rule_evaluation.json"
    csv_path = out_root / "rule_evaluation.csv"
    example_csv_path = out_root / "example_predictions.csv"
    pdf_path = out_root / "rule_evaluation_summary.pdf"

    if not force_recompute and json_path.exists():
        with json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _RULE_EVALUATION_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset(
            {
                "prediction_mode": prediction_mode,
                "evaluated_rule_set_name": evaluated_rule_set_name,
            }
        ):
            print(f"  [cache] loading {json_path.name}")
            return cached

    eval_video_id_set = {str(video_id) for video_id in (eval_video_ids or [])}
    filtered_results = [
        result
        for result in temporal_rule_results
        if not eval_video_id_set or str(result.get("video_id", "")) in eval_video_id_set
    ]

    eval_examples: List[Dict[str, Any]] = []
    examples_by_id: Dict[str, Dict[str, Any]] = {}
    per_video_example_ids: Dict[str, List[str]] = {}
    total_positive_examples = 0
    for video_result in filtered_results:
        video_id = str(video_result.get("video_id", ""))
        video_examples = list(video_result.get("examples", []))
        per_video_example_ids[video_id] = []
        for example in video_examples:
            example_id = str(example.get("example_id", ""))
            if not example_id:
                continue
            eval_example = {
                "video_id": video_id,
                "example_id": example_id,
                "label": bool(example.get("label", False)),
                "target_predicate": str(example.get("target_predicate", "")),
                "current_segment_id": str(example.get("current_segment_id", "")),
                "next_segment_id": str(example.get("next_segment_id", "")),
                "body_atoms": list(example.get("body_atoms", [])),
            }
            eval_examples.append(eval_example)
            examples_by_id[example_id] = eval_example
            per_video_example_ids[video_id].append(example_id)
            if eval_example["label"]:
                total_positive_examples += 1

    final_rules = list(final_rule_results.get("final_rules", []))
    triggered_rules_by_example: Dict[str, List[str]] = {}
    rule_evaluations: List[Dict[str, Any]] = []

    for rule in final_rules:
        rule_id = str(rule.get("rule_id", ""))
        body_atom_templates = _get_rule_body_atom_templates(rule)
        evidence_entries: List[Dict[str, Any]] = []
        for example in eval_examples:
            match_states = _find_rule_matches_for_example(
                body_atom_templates=body_atom_templates,
                body_atoms=list(example.get("body_atoms", [])),
            )
            if not match_states:
                continue
            triggered_rules_by_example.setdefault(str(example["example_id"]), []).append(rule_id)
            for match_state in match_states:
                evidence_entries.append(
                    {
                        "video_id": str(example.get("video_id", "")),
                        "example_id": str(example.get("example_id", "")),
                        "current_segment_id": str(example.get("current_segment_id", "")),
                        "next_segment_id": str(example.get("next_segment_id", "")),
                        "target_predicate": str(example.get("target_predicate", "")),
                        "label": bool(example.get("label", False)),
                        "bindings": dict(match_state.get("bindings", {})),
                        "matched_atoms": dict(match_state.get("matched_atoms", {})),
                    }
                )

        evidence_summary = _summarize_rule_evidence(
            evidence_entries=evidence_entries,
            total_positive_examples=total_positive_examples,
        )
        evaluated_rule = dict(rule)
        evaluated_rule["eval_num_fired_examples"] = int(evidence_summary["total_support"])
        evaluated_rule["eval_positive_support"] = int(evidence_summary["positive_support"])
        evaluated_rule["eval_negative_support"] = int(evidence_summary["negative_support"])
        evaluated_rule["eval_positive_firings"] = int(evidence_summary["positive_firings"])
        evaluated_rule["eval_negative_firings"] = int(evidence_summary["negative_firings"])
        evaluated_rule["eval_total_firings"] = int(evidence_summary["total_firings"])
        evaluated_rule["eval_precision"] = float(evidence_summary["precision"])
        evaluated_rule["eval_recall"] = float(evidence_summary["recall"])
        evaluated_rule["eval_positive_example_ids"] = list(evidence_summary["positive_example_ids"])
        evaluated_rule["eval_negative_example_ids"] = list(evidence_summary["negative_example_ids"])
        rule_evaluations.append(evaluated_rule)

    predicted_positive_example_ids = {
        example_id for example_id, rule_ids in triggered_rules_by_example.items() if rule_ids
    }
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    example_prediction_rows: List[Dict[str, Any]] = []

    for example in eval_examples:
        example_id = str(example.get("example_id", ""))
        predicted_positive = example_id in predicted_positive_example_ids
        label = bool(example.get("label", False))
        if predicted_positive and label:
            true_positive += 1
        elif predicted_positive and not label:
            false_positive += 1
        elif not predicted_positive and label:
            false_negative += 1
        else:
            true_negative += 1

        example_prediction_rows.append(
            {
                "video_id": str(example.get("video_id", "")),
                "example_id": example_id,
                "label": label,
                "predicted_positive": predicted_positive,
                "matching_rule_ids": list(triggered_rules_by_example.get(example_id, [])),
                "num_matching_rules": len(triggered_rules_by_example.get(example_id, [])),
            }
        )

    overall_metrics = _compute_binary_metrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
    )

    per_video_metrics: List[Dict[str, Any]] = []
    for video_id in sorted(per_video_example_ids):
        tp = fp = fn = tn = 0
        for example_id in per_video_example_ids[video_id]:
            example = examples_by_id[example_id]
            predicted_positive = example_id in predicted_positive_example_ids
            label = bool(example.get("label", False))
            if predicted_positive and label:
                tp += 1
            elif predicted_positive and not label:
                fp += 1
            elif not predicted_positive and label:
                fn += 1
            else:
                tn += 1
        metrics = _compute_binary_metrics(
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            true_negative=tn,
        )
        metrics["video_id"] = video_id
        metrics["num_examples"] = len(per_video_example_ids[video_id])
        per_video_metrics.append(metrics)

    result: Dict[str, Any] = {
        "version": _RULE_EVALUATION_VERSION,
        "config": {
            "prediction_mode": prediction_mode,
            "evaluated_rule_set_name": evaluated_rule_set_name,
        },
        "target_predicate": "brake_next",
        "split": split_manifest or {},
        "num_eval_videos": len(filtered_results),
        "eval_video_ids": sorted(str(result.get("video_id", "")) for result in filtered_results),
        "num_eval_examples": len(eval_examples),
        "num_eval_positive_examples": total_positive_examples,
        "num_eval_negative_examples": len(eval_examples) - total_positive_examples,
        "num_final_rules": len(final_rules),
        "num_rules_fired": sum(1 for rule in rule_evaluations if int(rule.get("eval_total_firings", 0)) > 0),
        "overall_metrics": overall_metrics,
        "per_video_metrics": per_video_metrics,
        "rule_evaluations": rule_evaluations,
        "example_predictions_csv_path": str(example_csv_path),
        "pdf_path": str(pdf_path),
    }

    _save_evaluation_pdf(result, pdf_path)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rule_id",
                "clause",
                "confidence",
                "positive_support",
                "negative_support",
                "eval_num_fired_examples",
                "eval_positive_support",
                "eval_negative_support",
                "eval_positive_firings",
                "eval_negative_firings",
                "eval_total_firings",
                "eval_precision",
                "eval_recall",
            ],
        )
        writer.writeheader()
        for rule in rule_evaluations:
            writer.writerow(
                {
                    "rule_id": rule.get("rule_id", ""),
                    "clause": rule.get("clause", ""),
                    "confidence": rule.get("confidence", 0.0),
                    "positive_support": rule.get("positive_support", 0),
                    "negative_support": rule.get("negative_support", 0),
                    "eval_num_fired_examples": rule.get("eval_num_fired_examples", 0),
                    "eval_positive_support": rule.get("eval_positive_support", 0),
                    "eval_negative_support": rule.get("eval_negative_support", 0),
                    "eval_positive_firings": rule.get("eval_positive_firings", 0),
                    "eval_negative_firings": rule.get("eval_negative_firings", 0),
                    "eval_total_firings": rule.get("eval_total_firings", 0),
                    "eval_precision": rule.get("eval_precision", 0.0),
                    "eval_recall": rule.get("eval_recall", 0.0),
                }
            )

    with example_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "video_id",
                "example_id",
                "label",
                "predicted_positive",
                "num_matching_rules",
                "matching_rule_ids",
            ],
        )
        writer.writeheader()
        for row in example_prediction_rows:
            writer.writerow(
                {
                    "video_id": row.get("video_id", ""),
                    "example_id": row.get("example_id", ""),
                    "label": row.get("label", False),
                    "predicted_positive": row.get("predicted_positive", False),
                    "num_matching_rules": row.get("num_matching_rules", 0),
                    "matching_rule_ids": json.dumps(row.get("matching_rule_ids", [])),
                }
            )

    print(
        "  rule_evaluation: "
        f"eval_videos={len(filtered_results)} | "
        f"eval_examples={len(eval_examples)} | "
        f"precision={float(overall_metrics['precision']):.3f} | "
        f"recall={float(overall_metrics['recall']):.3f} | "
        f"f1={float(overall_metrics['f1']):.3f}"
    )
    print(f"Rule evaluation JSON written to {json_path}")
    print(f"Rule evaluation CSV written to {csv_path}")
    print(f"Example predictions CSV written to {example_csv_path}")
    print(f"Rule evaluation PDF written to {pdf_path}")
    return result


def run(
    final_rule_results: Dict[str, Any],
    temporal_rule_results: List[Dict[str, Any]],
    eval_video_ids: Optional[List[str]] = None,
    split_manifest: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_rules(
        final_rule_results=final_rule_results,
        temporal_rule_results=temporal_rule_results,
        eval_video_ids=eval_video_ids,
        split_manifest=split_manifest,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
