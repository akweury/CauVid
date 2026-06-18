"""
Generate integrated publication-style figures comparing NeSy rule selectors,
neural symbolic baselines, learned rule aggregation, and the oracle rule-pool
upper bound.

Reads existing artifacts from:
  - Step 17D rule-pool upper-bound diagnostic
  - Step 17E oracle rule-selection gap diagnostic
  - Step 18 held-out rule evaluation
  - Step 18B neural symbolic baseline
  - Step 18C rule aggregation baseline

Outputs:
    pipeline_output/22_driving_mini_integrated_method_visualization/
        method_f1_ladder.png/.pdf
        method_f1_ladder_data.csv
        method_precision_recall_f1.png/.pdf
        method_precision_recall_f1_data.csv
        oracle_gap_curve.png/.pdf
        oracle_gap_curve_data.csv
        top_weighted_rule_contributions.png/.pdf
        top_weighted_rule_contributions_data.csv
        method_characteristics_table.csv
        integrated_visualization_manifest.json
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_VISUALIZATION_VERSION = 2
_SELECTOR_ORDER = ["original", "diverse", "semantic_constrained_diverse", "coverage_family_aware"]
_SELECTOR_LABELS = {
    "original": "Original Rules",
    "diverse": "Diverse Rules",
    "semantic_constrained_diverse": "Semantic-Constrained Rules",
    "coverage_family_aware": "Coverage/Family-Aware Rules",
}
_SELECTOR_COLORS = {
    "original": "#264653",
    "diverse": "#2a9d8f",
    "semantic_constrained_diverse": "#e9c46a",
    "coverage_family_aware": "#e76f51",
}
_METHOD_COLORS = {
    "hard_or_rule_selector": "#355070",
    "neural_symbolic": "#43aa8b",
    "learned_rule_aggregation": "#bc4749",
    "oracle_upper_bound": "#6d597a",
}
_METHOD_LINESTYLES = {
    "hard_or_rule_selector": "--",
    "neural_symbolic": "-.",
    "learned_rule_aggregation": ":",
    "oracle_upper_bound": "-",
}
_METHOD_MARKERS = {
    "hard_or_rule_selector": "o",
    "neural_symbolic": "s",
    "learned_rule_aggregation": "D",
    "oracle_upper_bound": "*",
}
_SHORT_METHOD_LABELS = {
    "Original Rules": "Original",
    "Diverse Rules": "Diverse",
    "Semantic-Constrained Rules": "Sem-Constr",
    "Coverage/Family-Aware Rules": "Cov/Fam",
    "Single Segment Mlp": "Seg-MLP",
    "Temporal Gru": "Temp-GRU",
    "Temporal Mlp": "Temp-MLP",
    "Rule Aggregation LR": "RuleAgg-LR",
    "Oracle Pool Upper Bound": "Oracle",
    "Oracle Pool Upper Bound (Diagnostic Only)": "Oracle",
}


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "22_driving_mini_integrated_method_visualization"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dpi": int(cfg.get("dpi", 170)),
        "figure_format": str(cfg.get("figure_format", "png")),
    }


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return dict(json.load(fh))


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


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


def _figure_paths(out_root: Path, figure_name: str) -> Dict[str, Path]:
    return {
        "png": out_root / f"{figure_name}.png",
        "pdf": out_root / f"{figure_name}.pdf",
        "csv": out_root / f"{figure_name}_data.csv",
    }


def _short_method_label(label: str) -> str:
    return _SHORT_METHOD_LABELS.get(str(label), str(label))


def _save_figure(fig: Any, output_paths: Dict[str, Path], dpi: int) -> None:
    fig.savefig(output_paths["png"], dpi=dpi, bbox_inches="tight")
    fig.savefig(output_paths["pdf"], bbox_inches="tight")


def _selector_rows_from_step18(step18_root: Path) -> Tuple[List[Dict[str, Any]], str]:
    json_path = step18_root / "rule_set_comparison_summary.json"
    if json_path.exists():
        payload = _read_json(json_path)
        return list(payload.get("rows", [])), str(payload.get("primary_rule_set", "original"))
    csv_path = step18_root / "rule_set_comparison_summary.csv"
    if csv_path.exists():
        return _read_csv(csv_path), "original"
    raise FileNotFoundError(f"Missing Step 18 comparison summary under {step18_root}")


def _method_rows_from_step18(
    step18_rows: Sequence[Dict[str, Any]],
    selector_overlap_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    overlap_by_selector = {
        str(row.get("selector_name", "")): dict(row)
        for row in selector_overlap_rows
        if str(row.get("selector_name", ""))
    }
    rows: List[Dict[str, Any]] = []
    for row in step18_rows:
        selector_name = str(row.get("rule_set_name", ""))
        overlap_row = overlap_by_selector.get(selector_name, {})
        rows.append(
            {
                "method_id": selector_name,
                "method_label": _SELECTOR_LABELS.get(selector_name, selector_name),
                "method_family": "hard_or_rule_selector",
                "precision": _safe_float(row.get("precision", 0.0)),
                "recall": _safe_float(row.get("recall", 0.0)),
                "f1": _safe_float(row.get("f1", 0.0)),
                "accuracy": _safe_float(row.get("accuracy", 0.0)),
                "auroc": float("nan"),
                "auprc": float("nan"),
                "selection_method": str(row.get("selection_method", "")),
                "oracle_overlap_fraction": _safe_float(overlap_row.get("oracle_overlap_fraction", float("nan")), float("nan")),
                "num_overlap_with_oracle": _safe_int(overlap_row.get("num_overlap_with_oracle", 0)),
                "num_missed_oracle_rules": _safe_int(overlap_row.get("num_missed_oracle_rules", 0)),
                "threshold_name": "rule_set",
                "uses_symbolic_rules": True,
                "uses_learned_weights": False,
                "uses_eval_labels": False,
            }
        )
    return rows


def _method_rows_from_neural_baseline(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    model_results = dict(summary.get("model_results", {}))
    for model_name, model_payload in model_results.items():
        eval_metrics = (
            dict(model_payload.get("metrics_by_split", {}))
            .get("eval", {})
            .get("best_validation_threshold", {})
        )
        rows.append(
            {
                "method_id": str(model_name),
                "method_label": str(model_name).replace("_", " ").title(),
                "method_family": "neural_symbolic",
                "precision": _safe_float(eval_metrics.get("precision", 0.0)),
                "recall": _safe_float(eval_metrics.get("recall", 0.0)),
                "f1": _safe_float(eval_metrics.get("f1", 0.0)),
                "accuracy": _safe_float(eval_metrics.get("accuracy", 0.0)),
                "auroc": _safe_float(eval_metrics.get("auroc", float("nan")), float("nan")),
                "auprc": _safe_float(eval_metrics.get("auprc", float("nan")), float("nan")),
                "selection_method": str(model_payload.get("architecture", "")),
                "oracle_overlap_fraction": float("nan"),
                "num_overlap_with_oracle": 0,
                "num_missed_oracle_rules": 0,
                "threshold_name": "best_validation_threshold",
                "uses_symbolic_rules": False,
                "uses_learned_weights": True,
                "uses_eval_labels": False,
            }
        )
    return rows


def _method_row_from_rule_aggregation(summary: Dict[str, Any]) -> Dict[str, Any]:
    eval_metrics = dict(summary.get("metrics_by_split", {})).get("eval", {}).get("best_validation_threshold", {})
    return {
        "method_id": "rule_aggregation_logistic_regression",
        "method_label": "Rule Aggregation LR",
        "method_family": "learned_rule_aggregation",
        "precision": _safe_float(eval_metrics.get("precision", 0.0)),
        "recall": _safe_float(eval_metrics.get("recall", 0.0)),
        "f1": _safe_float(eval_metrics.get("f1", 0.0)),
        "accuracy": _safe_float(eval_metrics.get("accuracy", 0.0)),
        "auroc": _safe_float(eval_metrics.get("auroc", float("nan")), float("nan")),
        "auprc": _safe_float(eval_metrics.get("auprc", float("nan")), float("nan")),
        "selection_method": "l1_logistic_regression",
        "oracle_overlap_fraction": float("nan"),
        "num_overlap_with_oracle": 0,
        "num_missed_oracle_rules": 0,
        "threshold_name": "best_validation_threshold",
        "uses_symbolic_rules": True,
        "uses_learned_weights": True,
        "uses_eval_labels": False,
        "num_nonzero_rules": _safe_int(summary.get("num_nonzero_rules", 0)),
    }


def _oracle_method_row(step17e_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method_id": "oracle_rule_pool_upper_bound",
        "method_label": "Oracle Pool Upper Bound (Diagnostic Only)",
        "method_family": "oracle_upper_bound",
        "precision": _safe_float(step17e_summary.get("oracle_target_precision", 0.0)),
        "recall": _safe_float(step17e_summary.get("oracle_target_recall", 0.0)),
        "f1": _safe_float(step17e_summary.get("oracle_target_f1", 0.0)),
        "accuracy": float("nan"),
        "auroc": float("nan"),
        "auprc": float("nan"),
        "selection_method": "oracle_greedy_eval",
        "oracle_overlap_fraction": 1.0,
        "num_overlap_with_oracle": _safe_int(step17e_summary.get("oracle_target_k", 0)),
        "num_missed_oracle_rules": 0,
        "threshold_name": "oracle_target",
        "uses_symbolic_rules": True,
        "uses_learned_weights": False,
        "uses_eval_labels": True,
        "diagnostic_only": True,
        "legal_model_result": False,
        "eval_label_usage_note": "Uses eval labels for oracle upper-bound diagnosis only; not a legal deployable model result.",
    }


def _method_order(method_rows: Sequence[Dict[str, Any]]) -> List[str]:
    family_priority = {
        "oracle_upper_bound": 0,
        "learned_rule_aggregation": 1,
        "neural_symbolic": 2,
        "hard_or_rule_selector": 3,
    }
    ordered_rows = sorted(
        method_rows,
        key=lambda row: (
            -_safe_float(row.get("f1", 0.0)),
            family_priority.get(str(row.get("method_family", "")), 99),
            str(row.get("method_label", "")),
        ),
    )
    return [str(row.get("method_id", "")) for row in ordered_rows]


def _plot_method_f1_ladder(
    method_rows: Sequence[Dict[str, Any]],
    output_paths: Dict[str, Path],
    dpi: int,
) -> List[Dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered_ids = _method_order(method_rows)
    ordered_rows = [next(row for row in method_rows if str(row.get("method_id", "")) == method_id) for method_id in ordered_ids]
    data_rows = [dict(row) for row in ordered_rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    y_positions = list(range(len(ordered_rows)))
    colors = [_METHOD_COLORS.get(str(row.get("method_family", "")), "#888888") for row in ordered_rows]
    values = [_safe_float(row.get("f1", 0.0)) for row in ordered_rows]
    bars = ax.barh(y_positions, values, color=colors)
    ax.set_yticks(y_positions, [str(row.get("method_label", "")) for row in ordered_rows])
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(0.85, max(values, default=0.0) + 0.05))
    ax.set_xlabel("Held-Out F1")
    ax.set_title("Method F1 Ladder", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    for bar, row in zip(bars, ordered_rows):
        value = _safe_float(row.get("f1", 0.0))
        suffix = ""
        overlap = row.get("oracle_overlap_fraction")
        if isinstance(overlap, (int, float)) and not math.isnan(float(overlap)):
            suffix = f" | oracle overlap={float(overlap):.2f}"
        ax.text(
            min(ax.get_xlim()[1] - 0.01, value + 0.01),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.3f}{suffix}",
            va="center",
            ha="left",
            fontsize=9,
        )
    fig.tight_layout()
    _save_figure(fig, output_paths, dpi)
    plt.close(fig)
    _write_csv(
        output_paths["csv"],
        [
            "method_id",
            "method_label",
            "method_family",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "auroc",
            "auprc",
            "selection_method",
            "oracle_overlap_fraction",
            "num_overlap_with_oracle",
            "num_missed_oracle_rules",
            "threshold_name",
            "uses_symbolic_rules",
            "uses_learned_weights",
            "uses_eval_labels",
        ],
        data_rows,
    )
    return data_rows


def _plot_method_precision_recall_f1(
    method_rows: Sequence[Dict[str, Any]],
    output_paths: Dict[str, Path],
    dpi: int,
) -> List[Dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered_ids = _method_order(method_rows)
    ordered_rows = [next(row for row in method_rows if str(row.get("method_id", "")) == method_id) for method_id in ordered_ids]
    x_positions = list(range(len(ordered_rows)))
    width = 0.22
    metric_names = ["precision", "recall", "f1"]
    metric_offsets = [-width, 0.0, width]
    metric_colors = ["#355070", "#43aa8b", "#bc4749"]
    data_rows = [dict(row) for row in ordered_rows]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    for metric_name, offset, metric_color in zip(metric_names, metric_offsets, metric_colors):
        values = [_safe_float(row.get(metric_name, 0.0)) for row in ordered_rows]
        positions = [x + offset for x in x_positions]
        bars = ax.bar(positions, values, width=width, color=metric_color, label=metric_name.upper())
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, min(1.015, value + 0.018), f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(
        x_positions,
        [_short_method_label(str(row.get("method_label", ""))) for row in ordered_rows],
        rotation=28,
        ha="right",
    )
    ax.set_ylim(0.0, 1.06)
    ax.set_ylabel("Score")
    ax.set_title("Held-Out Precision, Recall, and F1 by Method", loc="left", fontweight="bold")
    ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="#dddddd",
        facecolor="white",
        ncol=1,
        loc="upper right",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, output_paths, dpi)
    plt.close(fig)
    _write_csv(
        output_paths["csv"],
        [
            "method_id",
            "method_label",
            "method_family",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "auroc",
            "auprc",
            "selection_method",
            "threshold_name",
        ],
        data_rows,
    )
    return data_rows


def _plot_oracle_gap_curve(
    oracle_curve_rows: Sequence[Dict[str, Any]],
    method_rows: Sequence[Dict[str, Any]],
    step17e_summary: Dict[str, Any],
    output_paths: Dict[str, Path],
    dpi: int,
) -> List[Dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve_rows = sorted(
        [
            {
                "selection_rank": _safe_int(row.get("selection_rank", 0)),
                "precision": _safe_float(row.get("precision", 0.0)),
                "recall": _safe_float(row.get("recall", 0.0)),
                "f1": _safe_float(row.get("f1", 0.0)),
                "rule_id": str(row.get("rule_id", "")),
            }
            for row in oracle_curve_rows
        ],
        key=lambda row: int(row["selection_rank"]),
    )
    data_rows: List[Dict[str, Any]] = [dict(row, row_type="oracle_curve", method_id="", method_label="", method_family="") for row in curve_rows]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    x_values = [int(row["selection_rank"]) for row in curve_rows]
    f1_values = [float(row["f1"]) for row in curve_rows]
    ax.plot(
        x_values,
        f1_values,
        color=_METHOD_COLORS["oracle_upper_bound"],
        linestyle=_METHOD_LINESTYLES["oracle_upper_bound"],
        linewidth=2.4,
        label="Oracle F1 Curve",
    )

    target_k = _safe_int(step17e_summary.get("oracle_target_k", 0))
    target_f1 = _safe_float(step17e_summary.get("oracle_target_f1", 0.0))
    if target_k > 0:
        ax.scatter([target_k], [target_f1], color=_METHOD_COLORS["oracle_upper_bound"], s=90, marker="*", zorder=5, label=f"Oracle Peak (K={target_k})")

    for row in method_rows:
        method_family = str(row.get("method_family", ""))
        if method_family == "oracle_upper_bound":
            continue
        method_label = str(row.get("method_label", ""))
        method_f1 = _safe_float(row.get("f1", 0.0))
        ax.axhline(
            method_f1,
            color=_METHOD_COLORS.get(method_family, "#888888"),
            linestyle=_METHOD_LINESTYLES.get(method_family, "--"),
            linewidth=1.3,
            alpha=0.7,
            label=method_label,
        )
        data_rows.append(
            {
                "row_type": "method_reference",
                "selection_rank": "",
                "precision": _safe_float(row.get("precision", 0.0)),
                "recall": _safe_float(row.get("recall", 0.0)),
                "f1": method_f1,
                "rule_id": "",
                "method_id": str(row.get("method_id", "")),
                "method_label": method_label,
                "method_family": method_family,
            }
        )

    ax.set_xlabel("Oracle Greedy Selection Rank (K)")
    ax.set_ylabel("Held-Out F1")
    ax.set_title("Oracle Gap Curve vs Real Methods", loc="left", fontweight="bold")
    ax.grid(alpha=0.2)
    ax.set_xlim(1, max(x_values[-1], target_k))
    ax.set_ylim(0.0, max(0.85, max(f1_values + [_safe_float(row.get("f1", 0.0)) for row in method_rows]) + 0.05))
    handles, labels = ax.get_legend_handles_labels()
    dedup_handles = []
    dedup_labels = []
    seen_labels = set()
    for handle, label in zip(handles, labels):
        if label in seen_labels:
            continue
        seen_labels.add(label)
        dedup_handles.append(handle)
        dedup_labels.append(label)
    ax.legend(
        dedup_handles,
        dedup_labels,
        frameon=True,
        fancybox=False,
        edgecolor="#dddddd",
        facecolor="white",
        loc="lower right",
        ncol=1,
    )
    fig.tight_layout()
    _save_figure(fig, output_paths, dpi)
    plt.close(fig)
    _write_csv(
        output_paths["csv"],
        [
            "row_type",
            "selection_rank",
            "precision",
            "recall",
            "f1",
            "rule_id",
            "method_id",
            "method_label",
            "method_family",
        ],
        data_rows,
    )
    return data_rows


def _plot_top_weighted_rule_contributions(
    top_weight_rows: Sequence[Dict[str, Any]],
    output_paths: Dict[str, Path],
    dpi: int,
) -> List[Dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not top_weight_rows:
        _write_csv(
            output_paths["csv"],
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
                "short_clause",
            ],
            [],
        )
        fig, ax = plt.subplots(figsize=(9.2, 4.6))
        ax.text(0.5, 0.5, "No nonzero learned rule weights available.", ha="center", va="center", fontsize=12)
        ax.axis("off")
        ax.set_title("Top Weighted Rule Contributions", loc="left", fontweight="bold")
        fig.tight_layout()
        _save_figure(fig, output_paths, dpi)
        plt.close(fig)
        return []

    ordered_rows = sorted(
        [dict(row) for row in top_weight_rows],
        key=lambda row: _safe_int(row.get("rank", 0)),
    )
    display_rows = ordered_rows[: min(12, len(ordered_rows))]
    data_rows: List[Dict[str, Any]] = []
    for row in display_rows:
        clause = str(row.get("clause", ""))
        short_clause = clause if len(clause) <= 88 else f"{clause[:85]}..."
        data_rows.append(
            {
                **row,
                "short_clause": short_clause,
            }
        )

    fig, ax = plt.subplots(figsize=(11.4, 6.6))
    y_positions = list(range(len(data_rows)))
    weights = [_safe_float(row.get("weight", 0.0)) for row in data_rows]
    colors = ["#bc4749" if weight >= 0.0 else "#577590" for weight in weights]
    bars = ax.barh(y_positions, weights, color=colors)
    ax.set_yticks(y_positions, [str(row.get("short_clause", "")) for row in data_rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0.0, color="#444444", linewidth=1.0, alpha=0.7)
    max_abs_weight = max(abs(weight) for weight in weights) if weights else 1.0
    ax.set_xlim(-max_abs_weight * 1.18, max_abs_weight * 1.18)
    ax.set_xlabel("Learned Logistic Weight")
    ax.set_title("Top Weighted Rule Contributions", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    for bar, row in zip(bars, data_rows):
        weight = _safe_float(row.get("weight", 0.0))
        family = str(row.get("semantic_family", ""))
        anchor_x = weight + (0.015 * max_abs_weight if weight >= 0.0 else -0.015 * max_abs_weight)
        ax.text(
            anchor_x,
            bar.get_y() + bar.get_height() / 2.0,
            f"{weight:+.3f} | {family}",
            va="center",
            ha="left" if weight >= 0.0 else "right",
            fontsize=8.5,
        )
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#bc4749", label="Positive contribution toward brake_next"),
        Patch(facecolor="#577590", label="Negative contribution against brake_next"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#dddddd",
        facecolor="white",
    )
    fig.tight_layout()
    _save_figure(fig, output_paths, dpi)
    plt.close(fig)
    _write_csv(
        output_paths["csv"],
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
            "short_clause",
        ],
        data_rows,
    )
    return data_rows


def process_visualization(
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "integrated_visualization_manifest.json"
    characteristics_csv_path = out_root / "method_characteristics_table.csv"
    figure_outputs = {
        "method_f1_ladder": _figure_paths(out_root, "method_f1_ladder"),
        "method_precision_recall_f1": _figure_paths(out_root, "method_precision_recall_f1"),
        "oracle_gap_curve": _figure_paths(out_root, "oracle_gap_curve"),
        "top_weighted_rule_contributions": _figure_paths(out_root, "top_weighted_rule_contributions"),
    }
    dpi = int(cfg.get("dpi", 170))

    if not force_recompute and manifest_path.exists():
        cached = _read_json(manifest_path)
        if int(cached.get("version", 0)) == _VISUALIZATION_VERSION and _cfg_key_subset(cached.get("config", {})) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {manifest_path.name}")
            return cached

    pipeline_root = config.get_output_path("pipeline_output")
    step17d_root = pipeline_root / "17d_driving_mini_rule_pool_upper_bound_diagnostic"
    step17e_root = pipeline_root / "17e_driving_mini_oracle_rule_selection_gap_diagnostic"
    step18_root = pipeline_root / "18_driving_mini_rule_evaluation"
    step18b_root = pipeline_root / "neural_baselines_driving_mini"
    step18c_root = pipeline_root / "18c_driving_mini_rule_aggregation_baseline"

    step17d_summary = _read_json(step17d_root / "pool_upper_bound_summary.json")
    step17e_summary = _read_json(step17e_root / "oracle_selection_gap_summary.json")
    step17e_overlap_rows = _read_csv(step17e_root / "selector_oracle_overlap_summary.csv")
    step18_rows, step18_primary = _selector_rows_from_step18(step18_root)
    step18b_summary = _read_json(step18b_root / "neural_baseline_summary.json")
    step18c_summary = _read_json(step18c_root / "rule_aggregation_baseline_summary.json")
    oracle_curve_rows = _read_csv(step17d_root / "oracle_greedy_rule_set_curve.csv")
    step18c_top_weight_rows = list(step18c_summary.get("top_weighted_rules", []))

    method_rows: List[Dict[str, Any]] = []
    method_rows.extend(_method_rows_from_step18(step18_rows, step17e_overlap_rows))
    method_rows.extend(_method_rows_from_neural_baseline(step18b_summary))
    method_rows.append(_method_row_from_rule_aggregation(step18c_summary))
    method_rows.append(_oracle_method_row(step17e_summary))

    characteristics_rows = [
        {
            "method_id": str(row.get("method_id", "")),
            "method_label": str(row.get("method_label", "")),
            "method_family": str(row.get("method_family", "")),
            "uses_symbolic_rules": bool(row.get("uses_symbolic_rules", False)),
            "uses_learned_weights": bool(row.get("uses_learned_weights", False)),
            "uses_eval_labels": bool(row.get("uses_eval_labels", False)),
            "diagnostic_only": bool(row.get("diagnostic_only", False)),
            "legal_model_result": bool(row.get("legal_model_result", True)),
            "selection_method": str(row.get("selection_method", "")),
            "threshold_name": str(row.get("threshold_name", "")),
            "eval_label_usage_note": str(row.get("eval_label_usage_note", "")),
        }
        for row in method_rows
    ]
    _write_csv(
        characteristics_csv_path,
        [
            "method_id",
            "method_label",
            "method_family",
            "uses_symbolic_rules",
            "uses_learned_weights",
            "uses_eval_labels",
            "diagnostic_only",
            "legal_model_result",
            "selection_method",
            "threshold_name",
            "eval_label_usage_note",
        ],
        characteristics_rows,
    )

    figure_data = {
        "method_f1_ladder": _plot_method_f1_ladder(method_rows, figure_outputs["method_f1_ladder"], dpi),
        "method_precision_recall_f1": _plot_method_precision_recall_f1(method_rows, figure_outputs["method_precision_recall_f1"], dpi),
        "oracle_gap_curve": _plot_oracle_gap_curve(oracle_curve_rows, method_rows, step17e_summary, figure_outputs["oracle_gap_curve"], dpi),
        "top_weighted_rule_contributions": _plot_top_weighted_rule_contributions(
            step18c_top_weight_rows,
            figure_outputs["top_weighted_rule_contributions"],
            dpi,
        ),
    }

    manifest: Dict[str, Any] = {
        "version": _VISUALIZATION_VERSION,
        "config": _cfg_key_subset(cfg),
        "step18_primary_rule_set": step18_primary,
        "input_paths": {
            "step17d_summary_json": str(step17d_root / "pool_upper_bound_summary.json"),
            "step17d_oracle_curve_csv": str(step17d_root / "oracle_greedy_rule_set_curve.csv"),
            "step17e_summary_json": str(step17e_root / "oracle_selection_gap_summary.json"),
            "step17e_overlap_csv": str(step17e_root / "selector_oracle_overlap_summary.csv"),
            "step18_summary_json": str(step18_root / "rule_set_comparison_summary.json"),
            "step18b_summary_json": str(step18b_root / "neural_baseline_summary.json"),
            "step18c_summary_json": str(step18c_root / "rule_aggregation_baseline_summary.json"),
            "step18c_top_weighted_rules_csv": str(step18c_root / "top_weighted_rules_with_clauses.csv"),
            "step18c_subset_ablations_csv": str(step18c_root / "rule_aggregation_ablation_metrics.csv"),
            "step18c_top_k_rule_sets_csv": str(step18c_root / "top_k_weighted_rule_sets.csv"),
            "step18c_split_leakage_json": str(step18c_root / "split_leakage_check.json"),
        },
        "figure_paths": {
            figure_name: {suffix: str(path) for suffix, path in paths.items()}
            for figure_name, paths in figure_outputs.items()
        },
        "table_paths": {
            "method_characteristics_table_csv": str(characteristics_csv_path),
        },
        "method_rows": method_rows,
        "oracle_target_f1": _safe_float(step17e_summary.get("oracle_target_f1", 0.0)),
        "best_actual_selector_f1": _safe_float(step17d_summary.get("best_actual_selector_f1", 0.0)),
        "diagnostic_notes": {
            "oracle_upper_bound": "Oracle upper bound uses eval labels for diagnostic analysis only and is not a legal model result."
        },
        "figure_data_row_counts": {name: len(rows) for name, rows in figure_data.items()},
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(
        "  integrated_method_visualization: "
        f"methods={len(method_rows)} | "
        f"figures={len(figure_outputs)} | "
        f"oracle_target_f1={_safe_float(step17e_summary.get('oracle_target_f1', 0.0)):.3f}"
    )
    print(f"Integrated method visualization manifest written to {manifest_path}")
    for figure_name, paths in figure_outputs.items():
        print(f"Figure written to {paths['png']}")
        print(f"Figure written to {paths['pdf']}")
        print(f"Figure data written to {paths['csv']}")
    print(f"Method characteristics table written to {characteristics_csv_path}")
    return manifest


def run(
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_visualization(
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
