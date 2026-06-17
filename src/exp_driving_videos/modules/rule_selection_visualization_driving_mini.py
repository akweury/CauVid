"""
Generate publication-style rule-selection comparison plots from step 18/19/20
summary artifacts.

Outputs:
    pipeline_output/21_rule_selection_visualization/
        selector_metrics.png/.pdf
        selector_metrics_data.csv
        fn_comparison.png/.pdf
        fn_comparison_data.csv
        rule_family_composition.png/.pdf
        rule_family_composition_data.csv
        confidence_vs_positive_support.png/.pdf
        confidence_vs_positive_support_data.csv
        vehicle_family_coverage.png/.pdf
        vehicle_family_coverage_data.csv
        rule_selection_visualization_manifest.json
"""

from __future__ import annotations

import csv
import json
import math
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


_VISUALIZATION_VERSION = 2
_SELECTOR_ORDER = ["original", "diverse", "semantic_constrained_diverse", "coverage_family_aware"]
_SELECTOR_LABELS = {
    "original": "Original",
    "diverse": "Diverse",
    "semantic_constrained_diverse": "Semantic Constrained",
    "coverage_family_aware": "Coverage/Family Aware",
}
_SELECTOR_COLORS = {
    "original": "#264653",
    "diverse": "#2a9d8f",
    "semantic_constrained_diverse": "#e9c46a",
    "coverage_family_aware": "#e76f51",
}
_MATCH_LEVEL_ORDER = [
    "exact_vehicle_near_centered",
    "vehicle_near_partial",
    "vehicle_centered_partial",
    "near_centered_partial",
    "vehicle_only",
    "near_only",
    "centered_only",
]
_MATCH_LEVEL_LABELS = {
    "exact_vehicle_near_centered": "vehicle+near+centered",
    "vehicle_near_partial": "vehicle+near",
    "vehicle_centered_partial": "vehicle+centered",
    "near_centered_partial": "near+centered",
    "vehicle_only": "vehicle",
    "near_only": "near",
    "centered_only": "centered",
}
_COARSE_FAMILY_ORDER = [
    "centered",
    "near",
    "vehicle",
    "near+centered",
    "vehicle+centered",
    "vehicle+near",
    "vehicle+near+centered",
    "generic motion/object",
    "other",
]
_COARSE_FAMILY_COLORS = {
    "centered": "#90be6d",
    "near": "#43aa8b",
    "vehicle": "#4d908e",
    "near+centered": "#577590",
    "vehicle+centered": "#277da1",
    "vehicle+near": "#f8961e",
    "vehicle+near+centered": "#f3722c",
    "generic motion/object": "#8d99ae",
    "other": "#ced4da",
}
_EXPLICIT_MATCH_LEVELS = {
    "vehicle_centered_partial",
    "near_centered_partial",
    "vehicle_near_partial",
    "exact_vehicle_near_centered",
}


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "21_rule_selection_visualization"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "top_rule_families": int(cfg.get("top_rule_families", 8)),
        "dpi": int(cfg.get("dpi", 160)),
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


def _step_root(step_dir_name: str) -> Path:
    return config.get_output_path("pipeline_output") / step_dir_name


def _selector_file(step_root: Path, selector_name: str, primary_rule_set: str, filename: str) -> Path:
    if selector_name == primary_rule_set:
        return step_root / filename
    return step_root / selector_name / filename


def _load_summary_rows(step_root: Path, summary_json_name: str, summary_csv_name: str) -> Tuple[List[Dict[str, Any]], str]:
    json_path = step_root / summary_json_name
    if json_path.exists():
        payload = _read_json(json_path)
        return list(payload.get("rows", [])), str(payload.get("primary_rule_set", "original"))
    csv_path = step_root / summary_csv_name
    if csv_path.exists():
        return _read_csv(csv_path), "original"
    raise FileNotFoundError(f"Missing summary artifacts under {step_root}")


def _selector_rule_family_rows(step19_root: Path, primary_rule_set: str, selector_name: str) -> List[Dict[str, Any]]:
    path = _selector_file(step19_root, selector_name, primary_rule_set, "rule_family_summary.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing rule family summary: {path}")
    return _read_csv(path)


def _selector_rule_eval_rows(step18_root: Path, primary_rule_set: str, selector_name: str) -> List[Dict[str, Any]]:
    path = _selector_file(step18_root, selector_name, primary_rule_set, "rule_evaluation.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing rule evaluation CSV: {path}")
    return _read_csv(path)


def _figure_paths(out_root: Path, figure_name: str) -> Dict[str, Path]:
    return {
        "png": out_root / f"{figure_name}.png",
        "pdf": out_root / f"{figure_name}.pdf",
        "csv": out_root / f"{figure_name}_data.csv",
    }


def _save_figure(fig: Any, output_paths: Dict[str, Path], dpi: int) -> None:
    fig.savefig(output_paths["png"], dpi=dpi, bbox_inches="tight")
    fig.savefig(output_paths["pdf"], bbox_inches="tight")


def _coarse_semantic_family(predicate_signature: str) -> str:
    predicates = {part.strip() for part in str(predicate_signature).split("|") if part.strip()}
    has_vehicle = "object_class" in predicates
    has_near = "object_distance_state" in predicates
    has_centered = "object_x_position_state" in predicates
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
    if any(predicate.startswith("object_") or predicate.startswith("segment_") for predicate in predicates):
        return "generic motion/object"
    return "other"


def _vehicle_centered_fn_count(vehicle_manifest: Dict[str, Any], selector_name: str) -> int:
    key = f"vehicle_centered_fn_{selector_name}"
    return _safe_int(vehicle_manifest.get(key, 0))


def _explicit_vehicle_rule_count(
    pool_summary_rows: Sequence[Dict[str, Any]],
    selector_name: str,
) -> int:
    selector_pool_names = {
        "original": "selected_original",
        "diverse": "selected_diverse",
        "semantic_constrained_diverse": "selected_semantic_constrained_diverse",
        "coverage_family_aware": "selected_coverage_family_aware",
    }
    pool_name = selector_pool_names[selector_name]
    return sum(
        _safe_int(row.get("num_rules", 0))
        for row in pool_summary_rows
        if str(row.get("pool_name", "")) == pool_name and str(row.get("match_level", "")) in _EXPLICIT_MATCH_LEVELS
    )


def _plot_selector_metrics(
    comparison_rows: Sequence[Dict[str, Any]],
    output_paths: Dict[str, Path],
    dpi: int,
) -> List[Dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selectors = [row["rule_set_name"] for row in comparison_rows]
    x_positions = list(range(len(selectors)))
    width = 0.22
    metric_names = ["precision", "recall", "f1"]
    metric_offsets = [-width, 0.0, width]
    metric_colors = ["#355070", "#43aa8b", "#bc4749"]
    data_rows = [
        {
            "selector_name": row["rule_set_name"],
            "selector_label": _SELECTOR_LABELS.get(row["rule_set_name"], row["rule_set_name"]),
            "precision": _safe_float(row.get("precision", 0.0)),
            "recall": _safe_float(row.get("recall", 0.0)),
            "f1": _safe_float(row.get("f1", 0.0)),
        }
        for row in comparison_rows
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for metric_name, offset, metric_color in zip(metric_names, metric_offsets, metric_colors):
        values = [_safe_float(row.get(metric_name, 0.0)) for row in comparison_rows]
        positions = [x + offset for x in x_positions]
        bars = ax.bar(positions, values, width=width, color=metric_color, label=metric_name.upper())
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, min(1.015, value + 0.018), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x_positions, [_SELECTOR_LABELS.get(name, name) for name in selectors])
    ax.set_ylim(0.0, 1.06)
    ax.set_ylabel("Score")
    ax.set_title("Held-Out Rule-Selection Performance", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, output_paths, dpi)
    plt.close(fig)
    _write_csv(output_paths["csv"], ["selector_name", "selector_label", "precision", "recall", "f1"], data_rows)
    return data_rows


def _plot_fn_metrics(
    comparison_rows: Sequence[Dict[str, Any]],
    output_paths: Dict[str, Path],
    dpi: int,
) -> List[Dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selectors = [row["rule_set_name"] for row in comparison_rows]
    labels = [_SELECTOR_LABELS.get(name, name) for name in selectors]
    fn_counts = [_safe_int(row.get("num_fn_examples", 0)) for row in comparison_rows]
    fp_counts = [_safe_int(row.get("num_fp_examples", 0)) for row in comparison_rows]
    fn_recovery = [_safe_float(row.get("fn_coverage_vs_original", 0.0)) for row in comparison_rows]
    colors = [_SELECTOR_COLORS.get(name, "#888888") for name in selectors]
    data_rows = [
        {
            "selector_name": selector_name,
            "selector_label": label,
            "num_fn_examples": fn_count,
            "num_fp_examples": fp_count,
            "fn_recovery_vs_original": recovery,
        }
        for selector_name, label, fn_count, fp_count, recovery in zip(selectors, labels, fn_counts, fp_counts, fn_recovery)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    x_positions = list(range(len(labels)))
    width = 0.34

    fn_bars = axes[0].bar([x - width / 2.0 for x in x_positions], fn_counts, width=width, color="#bc4749", label="FN")
    fp_bars = axes[0].bar([x + width / 2.0 for x in x_positions], fp_counts, width=width, color="#577590", label="FP")
    axes[0].set_xticks(x_positions, labels)
    axes[0].set_ylabel("Count")
    axes[0].set_title("False-Negative and False-Positive Counts", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    axes[0].grid(axis="y", alpha=0.2)
    for bars in [fn_bars, fp_bars]:
        for bar in bars:
            value = int(round(bar.get_height()))
            axes[0].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.2, str(value), ha="center", va="bottom", fontsize=9)

    bars = axes[1].bar(labels, fn_recovery, color=colors)
    axes[1].set_title("False-Negative Recovery Relative to Original", loc="left", fontweight="bold")
    axes[1].set_ylabel("Recovered Fraction")
    axes[1].set_ylim(0.0, 1.06)
    axes[1].grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, fn_recovery):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, min(1.015, value + 0.018), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    _save_figure(fig, output_paths, dpi)
    plt.close(fig)
    _write_csv(
        output_paths["csv"],
        ["selector_name", "selector_label", "num_fn_examples", "num_fp_examples", "fn_recovery_vs_original"],
        data_rows,
    )
    return data_rows


def _plot_rule_family_composition(
    family_rows_by_selector: Dict[str, List[Dict[str, Any]]],
    output_paths: Dict[str, Path],
    dpi: int,
) -> List[Dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selectors = [name for name in _SELECTOR_ORDER if name in family_rows_by_selector]
    labels = [_SELECTOR_LABELS.get(name, name) for name in selectors]
    data_rows: List[Dict[str, Any]] = []
    coarse_counts_by_selector: Dict[str, Dict[str, int]] = {}

    for selector_name in selectors:
        coarse_counts = {family_name: 0 for family_name in _COARSE_FAMILY_ORDER}
        for row in family_rows_by_selector.get(selector_name, []):
            coarse_family = _coarse_semantic_family(str(row.get("predicate_signature", "")))
            coarse_counts[coarse_family] = coarse_counts.get(coarse_family, 0) + _safe_int(row.get("num_rules", 0))
        coarse_counts_by_selector[selector_name] = coarse_counts
        for coarse_family in _COARSE_FAMILY_ORDER:
            data_rows.append(
                {
                    "selector_name": selector_name,
                    "selector_label": _SELECTOR_LABELS.get(selector_name, selector_name),
                    "coarse_family": coarse_family,
                    "num_rules": coarse_counts.get(coarse_family, 0),
                }
            )

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    bottoms = [0 for _ in selectors]
    for coarse_family in _COARSE_FAMILY_ORDER:
        values = [coarse_counts_by_selector[selector_name].get(coarse_family, 0) for selector_name in selectors]
        ax.bar(
            labels,
            values,
            bottom=bottoms,
            color=_COARSE_FAMILY_COLORS.get(coarse_family, "#cccccc"),
            label=coarse_family,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_title("Selected Rule Composition by Coarse Semantic Family", loc="left", fontweight="bold")
    ax.set_ylabel("Selected Rules")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    _save_figure(fig, output_paths, dpi)
    plt.close(fig)
    _write_csv(output_paths["csv"], ["selector_name", "selector_label", "coarse_family", "num_rules"], data_rows)
    return data_rows


def _plot_confidence_support_scatter(
    rule_eval_rows_by_selector: Dict[str, List[Dict[str, Any]]],
    output_paths: Dict[str, Path],
    dpi: int,
) -> List[Dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selectors = [name for name in _SELECTOR_ORDER if name in rule_eval_rows_by_selector]
    data_rows: List[Dict[str, Any]] = []
    x_max = 0.0
    y_min = 0.0
    y_max = 1.0
    size_max_source = 1

    for selector_name in selectors:
        for row in rule_eval_rows_by_selector.get(selector_name, []):
            positive_support = max(0, _safe_int(row.get("positive_support", 0)))
            confidence = _safe_float(row.get("confidence", 0.0))
            eval_positive_support = max(0, _safe_int(row.get("eval_positive_support", 0)))
            log_positive_support = math.log10(1.0 + positive_support)
            size_max_source = max(size_max_source, eval_positive_support)
            x_max = max(x_max, log_positive_support)
            y_max = max(y_max, confidence)
            data_rows.append(
                {
                    "selector_name": selector_name,
                    "selector_label": _SELECTOR_LABELS.get(selector_name, selector_name),
                    "rule_id": str(row.get("rule_id", "")),
                    "positive_support": positive_support,
                    "log10_positive_support": log_positive_support,
                    "confidence": confidence,
                    "eval_positive_support": eval_positive_support,
                }
            )

    for row in data_rows:
        bubble_size = 26.0 + 18.0 * math.sqrt(float(row["eval_positive_support"]))
        row["bubble_size"] = min(220.0, bubble_size)

    fig, axes = plt.subplots(1, max(1, len(selectors)), figsize=(5.0 * max(1, len(selectors)), 4.8), squeeze=False, sharex=True, sharey=True)
    axes_row = axes[0]
    x_upper = max(0.2, x_max + 0.08)
    y_upper = max(1.0, y_max + 0.03)

    for axis, selector_name in zip(axes_row, selectors):
        rows = [row for row in data_rows if row["selector_name"] == selector_name]
        axis.scatter(
            [float(row["log10_positive_support"]) for row in rows],
            [float(row["confidence"]) for row in rows],
            s=[float(row["bubble_size"]) for row in rows],
            alpha=0.58,
            color=_SELECTOR_COLORS.get(selector_name, "#888888"),
            edgecolors="white",
            linewidths=0.5,
        )
        axis.set_title(_SELECTOR_LABELS.get(selector_name, selector_name), loc="left", fontweight="bold")
        axis.set_xlabel(r"$\log_{10}(1 + \mathrm{positive\ support})$")
        axis.set_ylabel("Confidence")
        axis.set_xlim(0.0, x_upper)
        axis.set_ylim(y_min, y_upper)
        axis.grid(alpha=0.2)

    for axis in axes_row[len(selectors):]:
        axis.axis("off")

    fig.suptitle("Selected Rule Confidence vs Training Support", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, output_paths, dpi)
    plt.close(fig)
    _write_csv(
        output_paths["csv"],
        [
            "selector_name",
            "selector_label",
            "rule_id",
            "positive_support",
            "log10_positive_support",
            "confidence",
            "eval_positive_support",
            "bubble_size",
        ],
        data_rows,
    )
    return data_rows


def _plot_vehicle_family_coverage(
    pool_summary_rows: Sequence[Dict[str, Any]],
    vehicle_manifest: Dict[str, Any],
    output_paths: Dict[str, Path],
    dpi: int,
) -> List[Dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selectors = [name for name in _SELECTOR_ORDER]
    selector_pool_names = {
        "original": "selected_original",
        "diverse": "selected_diverse",
        "semantic_constrained_diverse": "selected_semantic_constrained_diverse",
        "coverage_family_aware": "selected_coverage_family_aware",
    }
    labels = [_SELECTOR_LABELS.get(name, name) for name in selectors]
    colors = list(plt.cm.Set2.colors) + list(plt.cm.Set3.colors)
    data_rows: List[Dict[str, Any]] = []

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    bottoms = [0 for _ in selectors]
    for idx, match_level in enumerate(_MATCH_LEVEL_ORDER):
        values: List[int] = []
        for selector_name in selectors:
            pool_name = selector_pool_names[selector_name]
            value = 0
            for row in pool_summary_rows:
                if str(row.get("pool_name", "")) == pool_name and str(row.get("match_level", "")) == match_level:
                    value = _safe_int(row.get("num_rules", 0))
                    break
            values.append(value)
            data_rows.append(
                {
                    "selector_name": selector_name,
                    "selector_label": _SELECTOR_LABELS.get(selector_name, selector_name),
                    "series": "match_level_coverage",
                    "category": match_level,
                    "category_label": _MATCH_LEVEL_LABELS.get(match_level, match_level),
                    "value": value,
                }
            )
        axes[0].bar(labels, values, bottom=bottoms, color=colors[idx % len(colors)], label=_MATCH_LEVEL_LABELS.get(match_level, match_level))
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    axes[0].set_title("Vehicle/Near/Centered Rule Coverage", loc="left", fontweight="bold")
    axes[0].set_ylabel("Selected Rules")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))

    vehicle_centered_fn_counts = [_vehicle_centered_fn_count(vehicle_manifest, selector_name) for selector_name in selectors]
    explicit_vehicle_rule_counts = [_explicit_vehicle_rule_count(pool_summary_rows, selector_name) for selector_name in selectors]
    x_positions = list(range(len(labels)))
    width = 0.34
    fn_bars = axes[1].bar([x - width / 2.0 for x in x_positions], vehicle_centered_fn_counts, width=width, color="#bc4749", label="Vehicle-centered FN")
    explicit_bars = axes[1].bar([x + width / 2.0 for x in x_positions], explicit_vehicle_rule_counts, width=width, color="#355070", label="Explicit vehicle rules")
    axes[1].set_xticks(x_positions, labels)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Vehicle-Centered Errors vs Explicit Rule Coverage", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    axes[1].grid(axis="y", alpha=0.2)
    for bars in [fn_bars, explicit_bars]:
        for bar in bars:
            value = int(round(bar.get_height()))
            axes[1].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.2, str(value), ha="center", va="bottom", fontsize=9)

    for selector_name, label, fn_count, explicit_count in zip(selectors, labels, vehicle_centered_fn_counts, explicit_vehicle_rule_counts):
        data_rows.append(
            {
                "selector_name": selector_name,
                "selector_label": label,
                "series": "vehicle_centered_diagnostic",
                "category": "vehicle_centered_fn_count",
                "category_label": "Vehicle-centered FN count",
                "value": fn_count,
            }
        )
        data_rows.append(
            {
                "selector_name": selector_name,
                "selector_label": label,
                "series": "vehicle_centered_diagnostic",
                "category": "explicit_vehicle_rule_count",
                "category_label": "Explicit vehicle-rule count",
                "value": explicit_count,
            }
        )

    fig.tight_layout()
    _save_figure(fig, output_paths, dpi)
    plt.close(fig)
    _write_csv(
        output_paths["csv"],
        ["selector_name", "selector_label", "series", "category", "category_label", "value"],
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

    dpi = int(cfg.get("dpi", 160))
    manifest_path = out_root / "rule_selection_visualization_manifest.json"
    figure_outputs = {
        "selector_metrics": _figure_paths(out_root, "selector_metrics"),
        "fn_comparison": _figure_paths(out_root, "fn_comparison"),
        "rule_family_composition": _figure_paths(out_root, "rule_family_composition"),
        "confidence_vs_positive_support": _figure_paths(out_root, "confidence_vs_positive_support"),
        "vehicle_family_coverage": _figure_paths(out_root, "vehicle_family_coverage"),
    }

    if not force_recompute and manifest_path.exists():
        cached = _read_json(manifest_path)
        if int(cached.get("version", 0)) == _VISUALIZATION_VERSION and _cfg_key_subset(cached.get("config", {})) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {manifest_path.name}")
            return cached

    step18_root = _step_root("18_driving_mini_rule_evaluation")
    step19_root = _step_root("19_driving_mini_error_and_explainability_analysis")
    step20_root = _step_root("20_driving_mini_vehicle_rule_diagnostic")

    step18_rows, step18_primary = _load_summary_rows(
        step18_root,
        "rule_set_comparison_summary.json",
        "rule_set_comparison_summary.csv",
    )
    step19_rows, step19_primary = _load_summary_rows(
        step19_root,
        "rule_set_comparison_summary.json",
        "rule_set_comparison_summary.csv",
    )
    pool_summary_rows = _read_csv(step20_root / "vehicle_centered_pool_summary.csv")
    vehicle_manifest = _read_json(step20_root / "vehicle_centered_diagnostic_summary.json")

    family_rows_by_selector = {
        selector_name: _selector_rule_family_rows(step19_root, step19_primary, selector_name)
        for selector_name in _SELECTOR_ORDER
    }
    rule_eval_rows_by_selector = {
        selector_name: _selector_rule_eval_rows(step18_root, step18_primary, selector_name)
        for selector_name in _SELECTOR_ORDER
    }

    figure_data = {
        "selector_metrics": _plot_selector_metrics(step18_rows, figure_outputs["selector_metrics"], dpi),
        "fn_comparison": _plot_fn_metrics(step19_rows, figure_outputs["fn_comparison"], dpi),
        "rule_family_composition": _plot_rule_family_composition(family_rows_by_selector, figure_outputs["rule_family_composition"], dpi),
        "confidence_vs_positive_support": _plot_confidence_support_scatter(rule_eval_rows_by_selector, figure_outputs["confidence_vs_positive_support"], dpi),
        "vehicle_family_coverage": _plot_vehicle_family_coverage(pool_summary_rows, vehicle_manifest, figure_outputs["vehicle_family_coverage"], dpi),
    }

    manifest: Dict[str, Any] = {
        "version": _VISUALIZATION_VERSION,
        "config": _cfg_key_subset(cfg),
        "selectors": list(_SELECTOR_ORDER),
        "step18_primary_rule_set": step18_primary,
        "step19_primary_rule_set": step19_primary,
        "step20_primary_diagnosis": str(vehicle_manifest.get("primary_diagnosis", "unknown")),
        "input_paths": {
            "step18_root": str(step18_root),
            "step19_root": str(step19_root),
            "step20_root": str(step20_root),
            "step18_summary_json": str(step18_root / "rule_set_comparison_summary.json"),
            "step19_summary_json": str(step19_root / "rule_set_comparison_summary.json"),
            "step20_pool_summary_csv": str(step20_root / "vehicle_centered_pool_summary.csv"),
            "step20_manifest_json": str(step20_root / "vehicle_centered_diagnostic_summary.json"),
        },
        "figure_paths": {
            figure_name: {suffix: str(path) for suffix, path in paths.items()}
            for figure_name, paths in figure_outputs.items()
        },
        "figure_data_row_counts": {figure_name: len(rows) for figure_name, rows in figure_data.items()},
        "step18_rows": step18_rows,
        "step19_rows": step19_rows,
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(
        "  rule_selection_visualization: "
        f"selectors={len(_SELECTOR_ORDER)} | "
        f"figures={len(figure_outputs)} | "
        f"diagnosis={vehicle_manifest.get('primary_diagnosis', 'unknown')}"
    )
    print(f"Rule selection visualization manifest written to {manifest_path}")
    for figure_name, paths in figure_outputs.items():
        print(f"Figure written to {paths['png']}")
        print(f"Figure written to {paths['pdf']}")
        print(f"Figure data written to {paths['csv']}")
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
