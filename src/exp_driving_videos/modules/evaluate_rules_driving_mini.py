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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config
from src.exp_driving_videos.modules.final_rules_driving_mini import (
    _rule_candidate_category as _shared_rule_candidate_category,
)


_RULE_EVALUATION_VERSION = 6
_VARIABLE_ARGS = {"S", "O", "C", "T", "F"}


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
        if template_arg in _VARIABLE_ARGS:
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


def _rule_candidate_category(rule: Dict[str, Any]) -> str:
    return _shared_rule_candidate_category(rule)


def _build_rule_subset_views(
    rules: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    rules_list = list(rules)
    accepted_only_rules = [rule for rule in rules_list if _rule_candidate_category(rule) == "accepted_only"]
    mixed_rules = [rule for rule in rules_list if _rule_candidate_category(rule) == "mixed_accepted_candidate"]
    candidate_only_rules = [rule for rule in rules_list if _rule_candidate_category(rule) == "candidate_only"]
    candidate_candidate_rules = [rule for rule in rules_list if _rule_candidate_category(rule) == "candidate_candidate"]
    return {
        "all_rules": rules_list,
        "accepted_only_rules": accepted_only_rules,
        "candidate_only_rules": candidate_only_rules,
        "candidate_candidate_rules": candidate_candidate_rules,
        "mixed_accepted_candidate_rules": mixed_rules,
        "accepted_plus_mixed_rules": accepted_only_rules + mixed_rules,
        "accepted_plus_all_candidate_rules": (
            accepted_only_rules + mixed_rules + candidate_only_rules + candidate_candidate_rules
        ),
        "all_candidate_involving_rules": mixed_rules + candidate_only_rules + candidate_candidate_rules,
    }


def _empty_category_counts() -> Dict[str, int]:
    return {
        "accepted_only_rules": 0,
        "candidate_only_rules": 0,
        "candidate_candidate_rules": 0,
        "mixed_accepted_candidate_rules": 0,
        "candidate_involving_rules": 0,
        "all_rules": 0,
    }


def _count_rule_categories(rules: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = _empty_category_counts()
    for rule in rules:
        category = str(rule.get("candidate_rule_category", "")) or _rule_candidate_category(rule)
        counts["all_rules"] += 1
        if category == "mixed_accepted_candidate":
            counts["mixed_accepted_candidate_rules"] += 1
            counts["candidate_involving_rules"] += 1
        elif category == "candidate_candidate":
            counts["candidate_candidate_rules"] += 1
            counts["candidate_only_rules"] += 1
            counts["candidate_involving_rules"] += 1
        elif category == "candidate_only":
            counts["candidate_only_rules"] += 1
            counts["candidate_involving_rules"] += 1
        else:
            counts["accepted_only_rules"] += 1
    return counts


def _predicted_positive_ids_for_rules(
    rules: Sequence[Dict[str, Any]],
) -> Set[str]:
    predicted_ids: Set[str] = set()
    for rule in rules:
        predicted_ids.update(
            str(example_id)
            for example_id in list(rule.get("eval_triggered_example_ids", []))
            if str(example_id)
        )
    return predicted_ids


def _matched_prior_id_counts(rules: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rule in rules:
        for prior_id in list(rule.get("matched_prior_ids_involved", [])):
            prior_text = str(prior_id).strip()
            if prior_text:
                counts[prior_text] = counts.get(prior_text, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _subset_rule_composition(rules: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rules_list = list(rules)
    total_candidate_body_atoms = sum(max(0, int(rule.get("num_candidate_body_atoms", 0))) for rule in rules_list)
    candidate_body_atom_ratios = [
        max(0.0, float(rule.get("candidate_body_atom_ratio", 0.0)))
        for rule in rules_list
    ]
    return {
        "num_rules": len(rules_list),
        "accepted_only_rule_count": sum(1 for rule in rules_list if _rule_candidate_category(rule) == "accepted_only"),
        "candidate_only_rule_count": sum(1 for rule in rules_list if _rule_candidate_category(rule) == "candidate_only"),
        "candidate_candidate_rule_count": sum(
            1 for rule in rules_list if _rule_candidate_category(rule) == "candidate_candidate"
        ),
        "mixed_rule_count": sum(1 for rule in rules_list if _rule_candidate_category(rule) == "mixed_accepted_candidate"),
        "candidate_usage_rule_count": sum(1 for rule in rules_list if bool(rule.get("uses_candidate_atoms", False))),
        "total_candidate_body_atoms": total_candidate_body_atoms,
        "avg_num_candidate_body_atoms": float(total_candidate_body_atoms / max(1, len(rules_list))),
        "avg_candidate_body_atom_ratio": float(sum(candidate_body_atom_ratios) / max(1, len(candidate_body_atom_ratios))),
        "matched_prior_id_counts": _matched_prior_id_counts(rules_list),
    }


def _evaluate_rule_subset(
    subset_name: str,
    rules: Sequence[Dict[str, Any]],
    eval_examples: Sequence[Dict[str, Any]],
    examples_by_id: Dict[str, Dict[str, Any]],
    per_video_example_ids: Dict[str, List[str]],
    total_positive_examples: int,
    accepted_only_predicted_positive_ids: Set[str],
) -> Dict[str, Any]:
    rules_list = list(rules)
    predicted_positive_example_ids = _predicted_positive_ids_for_rules(rules_list)
    covered_positive_example_ids = {
        example_id
        for example_id in predicted_positive_example_ids
        if bool(examples_by_id.get(example_id, {}).get("label", False))
    }
    false_negative_example_ids = sorted(
        str(example.get("example_id", ""))
        for example in eval_examples
        if bool(example.get("label", False)) and str(example.get("example_id", "")) not in predicted_positive_example_ids
    )
    false_positive_example_ids = sorted(
        str(example.get("example_id", ""))
        for example in eval_examples
        if not bool(example.get("label", False)) and str(example.get("example_id", "")) in predicted_positive_example_ids
    )
    true_positive = len(covered_positive_example_ids)
    false_positive = len(false_positive_example_ids)
    false_negative = len(false_negative_example_ids)
    true_negative = len(eval_examples) - true_positive - false_positive - false_negative
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

    accepted_only_fn_ids = {
        str(example.get("example_id", ""))
        for example in eval_examples
        if bool(example.get("label", False)) and str(example.get("example_id", "")) not in accepted_only_predicted_positive_ids
    }
    fn_coverage_gain_example_ids = sorted(predicted_positive_example_ids & accepted_only_fn_ids)
    fp_contribution_example_ids = sorted(
        example_id
        for example_id in predicted_positive_example_ids - accepted_only_predicted_positive_ids
        if not bool(examples_by_id.get(example_id, {}).get("label", False))
    )
    composition = _subset_rule_composition(rules_list)
    return {
        "subset_name": subset_name,
        **composition,
        "num_rules_fired": sum(1 for rule in rules_list if int(rule.get("eval_total_firings", 0)) > 0),
        "num_eval_positive_examples": total_positive_examples,
        "num_predicted_positive_examples": len(predicted_positive_example_ids),
        "covered_positive_example_ids": sorted(covered_positive_example_ids),
        "false_negative_example_ids": false_negative_example_ids,
        "false_positive_example_ids": false_positive_example_ids,
        "predicted_positive_example_ids": sorted(predicted_positive_example_ids),
        "fn_coverage_gain_count_vs_accepted_only": len(fn_coverage_gain_example_ids),
        "fn_coverage_gain_rate_vs_accepted_only": float(
            len(fn_coverage_gain_example_ids) / max(1, len(accepted_only_fn_ids))
        ),
        "fn_coverage_gain_example_ids_vs_accepted_only": fn_coverage_gain_example_ids,
        "fp_contribution_count_vs_accepted_only": len(fp_contribution_example_ids),
        "fp_contribution_rate_vs_accepted_only": float(
            len(fp_contribution_example_ids) / max(1, len(eval_examples) - total_positive_examples)
        ),
        "fp_contribution_example_ids_vs_accepted_only": fp_contribution_example_ids,
        "overall_metrics": overall_metrics,
        "per_video_metrics": per_video_metrics,
    }


def _build_candidate_rule_ablation(
    subset_metrics: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    baseline = dict(subset_metrics.get("accepted_only_rules", {}))
    augmented = dict(subset_metrics.get("accepted_plus_all_candidate_rules", {}))
    mixed_augmented = dict(subset_metrics.get("accepted_plus_mixed_rules", {}))
    mixed_only = dict(subset_metrics.get("mixed_accepted_candidate_rules", {}))
    candidate_only = dict(subset_metrics.get("candidate_only_rules", {}))
    candidate_candidate = dict(subset_metrics.get("candidate_candidate_rules", {}))
    baseline_overall = dict(baseline.get("overall_metrics", {}))
    augmented_overall = dict(augmented.get("overall_metrics", {}))
    mixed_augmented_overall = dict(mixed_augmented.get("overall_metrics", {}))
    return {
        "baseline_subset_name": "accepted_only_rules",
        "augmented_subset_name": "accepted_plus_all_candidate_rules",
        "mixed_augmented_subset_name": "accepted_plus_mixed_rules",
        "baseline_metrics": baseline_overall,
        "augmented_metrics": augmented_overall,
        "mixed_augmented_metrics": mixed_augmented_overall,
        "delta_precision": float(augmented_overall.get("precision", 0.0) - baseline_overall.get("precision", 0.0)),
        "delta_recall": float(augmented_overall.get("recall", 0.0) - baseline_overall.get("recall", 0.0)),
        "delta_f1": float(augmented_overall.get("f1", 0.0) - baseline_overall.get("f1", 0.0)),
        "delta_accuracy": float(augmented_overall.get("accuracy", 0.0) - baseline_overall.get("accuracy", 0.0)),
        "mixed_delta_precision": float(
            mixed_augmented_overall.get("precision", 0.0) - baseline_overall.get("precision", 0.0)
        ),
        "mixed_delta_recall": float(
            mixed_augmented_overall.get("recall", 0.0) - baseline_overall.get("recall", 0.0)
        ),
        "mixed_delta_f1": float(mixed_augmented_overall.get("f1", 0.0) - baseline_overall.get("f1", 0.0)),
        "fn_coverage_gain_count": int(augmented.get("fn_coverage_gain_count_vs_accepted_only", 0)),
        "fn_coverage_gain_rate": float(augmented.get("fn_coverage_gain_rate_vs_accepted_only", 0.0)),
        "fp_contribution_count": int(augmented.get("fp_contribution_count_vs_accepted_only", 0)),
        "fp_contribution_rate": float(augmented.get("fp_contribution_rate_vs_accepted_only", 0.0)),
        "mixed_fn_coverage_gain_count": int(mixed_only.get("fn_coverage_gain_count_vs_accepted_only", 0)),
        "mixed_fp_contribution_count": int(mixed_only.get("fp_contribution_count_vs_accepted_only", 0)),
        "candidate_only_fn_coverage_gain_count": int(
            candidate_only.get("fn_coverage_gain_count_vs_accepted_only", 0)
        ),
        "candidate_only_fp_contribution_count": int(
            candidate_only.get("fp_contribution_count_vs_accepted_only", 0)
        ),
        "candidate_candidate_fn_coverage_gain_count": int(
            candidate_candidate.get("fn_coverage_gain_count_vs_accepted_only", 0)
        ),
        "candidate_candidate_fp_contribution_count": int(
            candidate_candidate.get("fp_contribution_count_vs_accepted_only", 0)
        ),
        "recovered_false_negative_example_ids": list(
            augmented.get("fn_coverage_gain_example_ids_vs_accepted_only", [])
        ),
        "added_false_positive_example_ids": list(
            augmented.get("fp_contribution_example_ids_vs_accepted_only", [])
        ),
    }


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
    subset_csv_path = out_root / "rule_subset_metrics.csv"
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
    triggered_rule_ids_by_example_by_subset: Dict[str, Dict[str, List[str]]] = {}
    rule_evaluations: List[Dict[str, Any]] = []

    for rule in final_rules:
        rule_id = str(rule.get("rule_id", ""))
        body_atom_templates = _get_rule_body_atom_templates(rule)
        evidence_entries: List[Dict[str, Any]] = []
        triggered_example_ids: Set[str] = set()
        candidate_rule_category = _rule_candidate_category(rule)
        for example in eval_examples:
            match_states = _find_rule_matches_for_example(
                body_atom_templates=body_atom_templates,
                body_atoms=list(example.get("body_atoms", [])),
            )
            if not match_states:
                continue
            example_id = str(example["example_id"])
            triggered_example_ids.add(example_id)
            triggered_rules_by_example.setdefault(example_id, []).append(rule_id)
            subset_trigger_map = triggered_rule_ids_by_example_by_subset.setdefault(example_id, {})
            subset_trigger_map.setdefault("all_rules", []).append(rule_id)
            subset_trigger_map.setdefault(f"{candidate_rule_category}_rules", []).append(rule_id)
            if candidate_rule_category == "mixed_accepted_candidate":
                subset_trigger_map.setdefault("accepted_plus_mixed_rules", []).append(rule_id)
                subset_trigger_map.setdefault("accepted_plus_all_candidate_rules", []).append(rule_id)
                subset_trigger_map.setdefault("all_candidate_involving_rules", []).append(rule_id)
            elif candidate_rule_category in {"candidate_only", "candidate_candidate"}:
                subset_trigger_map.setdefault("accepted_plus_all_candidate_rules", []).append(rule_id)
                subset_trigger_map.setdefault("all_candidate_involving_rules", []).append(rule_id)
            elif candidate_rule_category == "accepted_only":
                subset_trigger_map.setdefault("accepted_plus_mixed_rules", []).append(rule_id)
                subset_trigger_map.setdefault("accepted_plus_all_candidate_rules", []).append(rule_id)
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
        eval_f1 = float(
            2.0
            * float(evidence_summary["precision"])
            * float(evidence_summary["recall"])
            / max(1e-12, float(evidence_summary["precision"]) + float(evidence_summary["recall"]))
        )
        evaluated_rule = dict(rule)
        evaluated_rule["candidate_rule_category"] = candidate_rule_category
        evaluated_rule["eval_num_fired_examples"] = int(evidence_summary["total_support"])
        evaluated_rule["eval_positive_support"] = int(evidence_summary["positive_support"])
        evaluated_rule["eval_negative_support"] = int(evidence_summary["negative_support"])
        evaluated_rule["eval_positive_firings"] = int(evidence_summary["positive_firings"])
        evaluated_rule["eval_negative_firings"] = int(evidence_summary["negative_firings"])
        evaluated_rule["eval_total_firings"] = int(evidence_summary["total_firings"])
        evaluated_rule["eval_precision"] = float(evidence_summary["precision"])
        evaluated_rule["eval_recall"] = float(evidence_summary["recall"])
        evaluated_rule["eval_f1"] = eval_f1
        evaluated_rule["eval_positive_example_ids"] = list(evidence_summary["positive_example_ids"])
        evaluated_rule["eval_negative_example_ids"] = list(evidence_summary["negative_example_ids"])
        evaluated_rule["eval_triggered_example_ids"] = sorted(triggered_example_ids)
        rule_evaluations.append(evaluated_rule)

    rule_subset_views = _build_rule_subset_views(rule_evaluations)
    accepted_only_predicted_positive_ids = _predicted_positive_ids_for_rules(rule_subset_views["accepted_only_rules"])
    rule_subset_metrics = {
        subset_name: _evaluate_rule_subset(
            subset_name=subset_name,
            rules=subset_rules,
            eval_examples=eval_examples,
            examples_by_id=examples_by_id,
            per_video_example_ids=per_video_example_ids,
            total_positive_examples=total_positive_examples,
            accepted_only_predicted_positive_ids=accepted_only_predicted_positive_ids,
        )
        for subset_name, subset_rules in rule_subset_views.items()
    }
    candidate_rule_ablation = _build_candidate_rule_ablation(rule_subset_metrics)

    accepted_only_false_negative_example_ids = set(
        rule_subset_metrics["accepted_only_rules"].get("false_negative_example_ids", [])
    )
    accepted_only_negative_example_ids = {
        example_id
        for example_id in examples_by_id
        if example_id not in accepted_only_predicted_positive_ids
        and not bool(examples_by_id.get(example_id, {}).get("label", False))
    }
    for evaluated_rule in rule_evaluations:
        positive_ids = {
            str(example_id)
            for example_id in list(evaluated_rule.get("eval_positive_example_ids", []))
            if str(example_id)
        }
        negative_ids = {
            str(example_id)
            for example_id in list(evaluated_rule.get("eval_negative_example_ids", []))
            if str(example_id)
        }
        fn_coverage_gain_example_ids = sorted(positive_ids & accepted_only_false_negative_example_ids)
        fp_contribution_example_ids = sorted(negative_ids & accepted_only_negative_example_ids)
        evaluated_rule["eval_fn_coverage_gain_count_vs_accepted_only"] = len(fn_coverage_gain_example_ids)
        evaluated_rule["eval_fn_coverage_gain_rate_vs_accepted_only"] = float(
            len(fn_coverage_gain_example_ids) / max(1, len(accepted_only_false_negative_example_ids))
        )
        evaluated_rule["eval_fn_coverage_gain_example_ids_vs_accepted_only"] = fn_coverage_gain_example_ids
        evaluated_rule["eval_fp_contribution_count_vs_accepted_only"] = len(fp_contribution_example_ids)
        evaluated_rule["eval_fp_contribution_rate_vs_accepted_only"] = float(
            len(fp_contribution_example_ids) / max(1, len(accepted_only_negative_example_ids))
        )
        evaluated_rule["eval_fp_contribution_example_ids_vs_accepted_only"] = fp_contribution_example_ids

    overall_metrics = dict(rule_subset_metrics["accepted_plus_all_candidate_rules"].get("overall_metrics", {}))
    per_video_metrics = list(rule_subset_metrics["accepted_plus_all_candidate_rules"].get("per_video_metrics", []))
    predicted_positive_example_ids = set(
        rule_subset_metrics["accepted_plus_all_candidate_rules"].get("predicted_positive_example_ids", [])
    )
    covered_positive_example_ids = set(
        rule_subset_metrics["accepted_plus_all_candidate_rules"].get("covered_positive_example_ids", [])
    )
    false_negative_example_ids = list(
        rule_subset_metrics["accepted_plus_all_candidate_rules"].get("false_negative_example_ids", [])
    )
    false_positive_example_ids = list(
        rule_subset_metrics["accepted_plus_all_candidate_rules"].get("false_positive_example_ids", [])
    )
    evaluated_rule_counts = _count_rule_categories(rule_evaluations)
    fired_rule_counts = _count_rule_categories(
        [rule for rule in rule_evaluations if int(rule.get("eval_total_firings", 0)) > 0]
    )
    candidate_rule_flow_summary = dict(final_rule_results.get("candidate_rule_flow_summary", {}))
    if not candidate_rule_flow_summary:
        candidate_rule_flow_summary = {
            "atom_availability": {},
            "initial_rule_generation": _empty_category_counts(),
            "merged_after_step15": _empty_category_counts(),
            "pruning": _empty_category_counts(),
            "extension": {},
            "final_selection": _count_rule_categories(final_rules),
            "evaluation": _empty_category_counts(),
        }
    candidate_rule_flow_summary["evaluation"] = {
        "evaluated_rule_counts": evaluated_rule_counts,
        "fired_rule_counts": fired_rule_counts,
    }

    example_prediction_rows: List[Dict[str, Any]] = []
    for example in eval_examples:
        example_id = str(example.get("example_id", ""))
        predicted_positive = example_id in predicted_positive_example_ids
        subset_rule_ids = triggered_rule_ids_by_example_by_subset.get(example_id, {})
        example_prediction_rows.append(
            {
                "video_id": str(example.get("video_id", "")),
                "example_id": example_id,
                "label": bool(example.get("label", False)),
                "predicted_positive": predicted_positive,
                "predicted_positive_accepted_only": example_id in accepted_only_predicted_positive_ids,
                "predicted_positive_candidate_only": bool(subset_rule_ids.get("candidate_only_rules", [])),
                "predicted_positive_candidate_candidate": bool(subset_rule_ids.get("candidate_candidate_rules", [])),
                "predicted_positive_mixed_accepted_candidate": bool(
                    subset_rule_ids.get("mixed_accepted_candidate_rules", [])
                ),
                "predicted_positive_accepted_plus_mixed": bool(subset_rule_ids.get("accepted_plus_mixed_rules", [])),
                "predicted_positive_accepted_plus_all_candidate": bool(
                    subset_rule_ids.get("accepted_plus_all_candidate_rules", [])
                ),
                "num_matching_rules": len(triggered_rules_by_example.get(example_id, [])),
                "num_matching_rules_accepted_only": len(subset_rule_ids.get("accepted_only_rules", [])),
                "num_matching_rules_candidate_only": len(subset_rule_ids.get("candidate_only_rules", [])),
                "num_matching_rules_candidate_candidate": len(
                    subset_rule_ids.get("candidate_candidate_rules", [])
                ),
                "num_matching_rules_mixed_accepted_candidate": len(
                    subset_rule_ids.get("mixed_accepted_candidate_rules", [])
                ),
                "matching_rule_ids": list(triggered_rules_by_example.get(example_id, [])),
            }
        )

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
        "candidate_rule_stage_stats": {
            "stage": "step18_rule_evaluation",
            "evaluated_rule_counts": evaluated_rule_counts,
            "fired_rule_counts": fired_rule_counts,
        },
        "candidate_rule_flow_summary": candidate_rule_flow_summary,
        "predicted_positive_example_ids": sorted(predicted_positive_example_ids),
        "covered_positive_example_ids": sorted(covered_positive_example_ids),
        "false_negative_example_ids": sorted(false_negative_example_ids),
        "false_positive_example_ids": sorted(false_positive_example_ids),
        "overall_metrics": overall_metrics,
        "per_video_metrics": per_video_metrics,
        "rule_subset_metrics": rule_subset_metrics,
        "candidate_rule_ablation": candidate_rule_ablation,
        "rule_evaluations": rule_evaluations,
        "example_predictions_csv_path": str(example_csv_path),
        "rule_subset_metrics_csv_path": str(subset_csv_path),
        "candidate_rule_flow_summary_json_path": str(out_root / "candidate_rule_flow_summary.json"),
        "pdf_path": str(pdf_path),
    }

    _save_evaluation_pdf(result, pdf_path)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    candidate_flow_path = out_root / "candidate_rule_flow_summary.json"
    with candidate_flow_path.open("w", encoding="utf-8") as fh:
        json.dump(candidate_rule_flow_summary, fh, indent=2)

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
                "eval_f1",
                "candidate_rule_category",
                "uses_candidate_atoms",
                "num_candidate_body_atoms",
                "candidate_body_atom_ratio",
                "mixes_accepted_and_candidate_atoms",
                "uses_only_candidate_atoms",
                "body_source_mix",
                "matched_prior_ids_involved",
                "eval_fn_coverage_gain_count_vs_accepted_only",
                "eval_fn_coverage_gain_rate_vs_accepted_only",
                "eval_fp_contribution_count_vs_accepted_only",
                "eval_fp_contribution_rate_vs_accepted_only",
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
                    "eval_f1": rule.get("eval_f1", 0.0),
                    "candidate_rule_category": rule.get("candidate_rule_category", "accepted_only"),
                    "uses_candidate_atoms": rule.get("uses_candidate_atoms", False),
                    "num_candidate_body_atoms": rule.get("num_candidate_body_atoms", 0),
                    "candidate_body_atom_ratio": rule.get("candidate_body_atom_ratio", 0.0),
                    "mixes_accepted_and_candidate_atoms": rule.get("mixes_accepted_and_candidate_atoms", False),
                    "uses_only_candidate_atoms": rule.get("uses_only_candidate_atoms", False),
                    "body_source_mix": rule.get("body_source_mix", ""),
                    "matched_prior_ids_involved": json.dumps(rule.get("matched_prior_ids_involved", [])),
                    "eval_fn_coverage_gain_count_vs_accepted_only": rule.get(
                        "eval_fn_coverage_gain_count_vs_accepted_only", 0
                    ),
                    "eval_fn_coverage_gain_rate_vs_accepted_only": rule.get(
                        "eval_fn_coverage_gain_rate_vs_accepted_only", 0.0
                    ),
                    "eval_fp_contribution_count_vs_accepted_only": rule.get(
                        "eval_fp_contribution_count_vs_accepted_only", 0
                    ),
                    "eval_fp_contribution_rate_vs_accepted_only": rule.get(
                        "eval_fp_contribution_rate_vs_accepted_only", 0.0
                    ),
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
                "predicted_positive_accepted_only",
                "predicted_positive_candidate_only",
                "predicted_positive_candidate_candidate",
                "predicted_positive_mixed_accepted_candidate",
                "predicted_positive_accepted_plus_mixed",
                "predicted_positive_accepted_plus_all_candidate",
                "num_matching_rules",
                "num_matching_rules_accepted_only",
                "num_matching_rules_candidate_only",
                "num_matching_rules_candidate_candidate",
                "num_matching_rules_mixed_accepted_candidate",
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
                    "predicted_positive_accepted_only": row.get("predicted_positive_accepted_only", False),
                    "predicted_positive_candidate_only": row.get("predicted_positive_candidate_only", False),
                    "predicted_positive_candidate_candidate": row.get(
                        "predicted_positive_candidate_candidate", False
                    ),
                    "predicted_positive_mixed_accepted_candidate": row.get(
                        "predicted_positive_mixed_accepted_candidate", False
                    ),
                    "predicted_positive_accepted_plus_mixed": row.get(
                        "predicted_positive_accepted_plus_mixed", False
                    ),
                    "predicted_positive_accepted_plus_all_candidate": row.get(
                        "predicted_positive_accepted_plus_all_candidate", False
                    ),
                    "num_matching_rules": row.get("num_matching_rules", 0),
                    "num_matching_rules_accepted_only": row.get("num_matching_rules_accepted_only", 0),
                    "num_matching_rules_candidate_only": row.get("num_matching_rules_candidate_only", 0),
                    "num_matching_rules_candidate_candidate": row.get(
                        "num_matching_rules_candidate_candidate", 0
                    ),
                    "num_matching_rules_mixed_accepted_candidate": row.get(
                        "num_matching_rules_mixed_accepted_candidate", 0
                    ),
                    "matching_rule_ids": json.dumps(row.get("matching_rule_ids", [])),
                }
            )

    with subset_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "subset_name",
                "num_rules",
                "num_rules_fired",
                "accepted_only_rule_count",
                "candidate_only_rule_count",
                "candidate_candidate_rule_count",
                "mixed_rule_count",
                "candidate_usage_rule_count",
                "total_candidate_body_atoms",
                "avg_num_candidate_body_atoms",
                "avg_candidate_body_atom_ratio",
                "num_predicted_positive_examples",
                "num_covered_positive_examples",
                "num_false_negative_examples",
                "num_false_positive_examples",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "fn_coverage_gain_count_vs_accepted_only",
                "fn_coverage_gain_rate_vs_accepted_only",
                "fp_contribution_count_vs_accepted_only",
                "fp_contribution_rate_vs_accepted_only",
                "matched_prior_id_counts",
            ],
        )
        writer.writeheader()
        for subset_name, subset_result in rule_subset_metrics.items():
            subset_overall = dict(subset_result.get("overall_metrics", {}))
            writer.writerow(
                {
                    "subset_name": subset_name,
                    "num_rules": subset_result.get("num_rules", 0),
                    "num_rules_fired": subset_result.get("num_rules_fired", 0),
                    "accepted_only_rule_count": subset_result.get("accepted_only_rule_count", 0),
                    "candidate_only_rule_count": subset_result.get("candidate_only_rule_count", 0),
                    "candidate_candidate_rule_count": subset_result.get("candidate_candidate_rule_count", 0),
                    "mixed_rule_count": subset_result.get("mixed_rule_count", 0),
                    "candidate_usage_rule_count": subset_result.get("candidate_usage_rule_count", 0),
                    "total_candidate_body_atoms": subset_result.get("total_candidate_body_atoms", 0),
                    "avg_num_candidate_body_atoms": subset_result.get("avg_num_candidate_body_atoms", 0.0),
                    "avg_candidate_body_atom_ratio": subset_result.get("avg_candidate_body_atom_ratio", 0.0),
                    "num_predicted_positive_examples": subset_result.get("num_predicted_positive_examples", 0),
                    "num_covered_positive_examples": len(subset_result.get("covered_positive_example_ids", [])),
                    "num_false_negative_examples": len(subset_result.get("false_negative_example_ids", [])),
                    "num_false_positive_examples": len(subset_result.get("false_positive_example_ids", [])),
                    "precision": subset_overall.get("precision", 0.0),
                    "recall": subset_overall.get("recall", 0.0),
                    "f1": subset_overall.get("f1", 0.0),
                    "accuracy": subset_overall.get("accuracy", 0.0),
                    "fn_coverage_gain_count_vs_accepted_only": subset_result.get(
                        "fn_coverage_gain_count_vs_accepted_only", 0
                    ),
                    "fn_coverage_gain_rate_vs_accepted_only": subset_result.get(
                        "fn_coverage_gain_rate_vs_accepted_only", 0.0
                    ),
                    "fp_contribution_count_vs_accepted_only": subset_result.get(
                        "fp_contribution_count_vs_accepted_only", 0
                    ),
                    "fp_contribution_rate_vs_accepted_only": subset_result.get(
                        "fp_contribution_rate_vs_accepted_only", 0.0
                    ),
                    "matched_prior_id_counts": json.dumps(subset_result.get("matched_prior_id_counts", {})),
                }
            )

    print(
        "  rule_evaluation: "
        f"eval_videos={len(filtered_results)} | "
        f"eval_examples={len(eval_examples)} | "
        f"precision={float(overall_metrics['precision']):.3f} | "
        f"recall={float(overall_metrics['recall']):.3f} | "
        f"f1={float(overall_metrics['f1']):.3f} | "
        f"delta_f1_vs_accepted_only={float(candidate_rule_ablation.get('delta_f1', 0.0)):.3f}"
    )
    print(f"Rule evaluation JSON written to {json_path}")
    print(f"Rule evaluation CSV written to {csv_path}")
    print(f"Example predictions CSV written to {example_csv_path}")
    print(f"Rule subset metrics CSV written to {subset_csv_path}")
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
