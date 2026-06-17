"""
Generate rule-selection comparison plots from step 18/19/20 summary artifacts.

Outputs:
    pipeline_output/21_rule_selection_visualization/
        selector_metrics.png
        fn_comparison.png
        rule_family_composition.png
        confidence_vs_positive_support.png
        vehicle_family_coverage.png
        rule_selection_visualization_manifest.json
"""

from __future__ import annotations

import csv
import json
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


_VISUALIZATION_VERSION = 1
_SELECTOR_ORDER = ["original", "diverse", "coverage_family_aware"]
_SELECTOR_LABELS = {
    "original": "Original",
    "diverse": "Diverse",
    "coverage_family_aware": "Coverage/Family Aware",
}
_SELECTOR_COLORS = {
    "original": "#264653",
    "diverse": "#2a9d8f",
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


def _load_vehicle_context_totals(eval_context_rows: Sequence[Dict[str, Any]], selector_name: str) -> Tuple[int, int]:
    total_positive = 0
    predicted_positive = 0
    predicted_field = f"predicted_positive_{selector_name}"
    for row in eval_context_rows:
        total_positive += _safe_int(row.get("num_vehicle_near_centered_positive_examples", 0))
        predicted_positive += _safe_int(row.get(predicted_field, 0))
    return total_positive, predicted_positive


def _plot_selector_metrics(
    comparison_rows: Sequence[Dict[str, Any]],
    figure_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selectors = [row["rule_set_name"] for row in comparison_rows]
    x_positions = list(range(len(selectors)))
    width = 0.22
    metric_names = ["precision", "recall", "f1"]
    metric_offsets = [-width, 0.0, width]
    metric_colors = ["#457b9d", "#2a9d8f", "#e76f51"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for metric_name, offset, metric_color in zip(metric_names, metric_offsets, metric_colors):
        values = [_safe_float(row.get(metric_name, 0.0)) for row in comparison_rows]
        positions = [x + offset for x in x_positions]
        bars = ax.bar(positions, values, width=width, color=metric_color, label=metric_name.upper())
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, min(1.02, value + 0.02), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x_positions, [_SELECTOR_LABELS.get(name, name) for name in selectors])
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Selector Comparison: Precision / Recall / F1", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_fn_metrics(
    comparison_rows: Sequence[Dict[str, Any]],
    figure_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selectors = [row["rule_set_name"] for row in comparison_rows]
    labels = [_SELECTOR_LABELS.get(name, name) for name in selectors]
    fn_counts = [_safe_int(row.get("num_fn_examples", 0)) for row in comparison_rows]
    fn_recovery = [_safe_float(row.get("fn_coverage_vs_original", 0.0)) for row in comparison_rows]
    colors = [_SELECTOR_COLORS.get(name, "#888888") for name in selectors]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    bars = axes[0].bar(labels, fn_counts, color=colors)
    axes[0].set_title("False Negatives", loc="left", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, fn_counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, value + 0.2, str(value), ha="center", va="bottom", fontsize=9)

    bars = axes[1].bar(labels, fn_recovery, color=colors)
    axes[1].set_title("FN Recovery Vs Original", loc="left", fontweight="bold")
    axes[1].set_ylabel("Recovered Fraction")
    axes[1].set_ylim(0.0, 1.08)
    axes[1].grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, fn_recovery):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, min(1.02, value + 0.02), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_rule_family_composition(
    family_rows_by_selector: Dict[str, List[Dict[str, Any]]],
    figure_path: Path,
    dpi: int,
    top_rule_families: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate_counts: Dict[str, int] = {}
    for rows in family_rows_by_selector.values():
        for row in rows:
            signature = str(row.get("predicate_signature", "unknown"))
            aggregate_counts[signature] = aggregate_counts.get(signature, 0) + _safe_int(row.get("num_rules", 0))

    top_signatures = [
        signature
        for signature, _count in sorted(aggregate_counts.items(), key=lambda item: (-item[1], item[0]))[: max(1, top_rule_families)]
    ]
    if not top_signatures:
        top_signatures = ["no_families"]

    selectors = [name for name in _SELECTOR_ORDER if name in family_rows_by_selector]
    labels = [_SELECTOR_LABELS.get(name, name) for name in selectors]
    colors = list(plt.cm.tab20.colors)

    fig, ax = plt.subplots(figsize=(12, 6))
    bottoms = [0 for _ in selectors]
    for idx, signature in enumerate(top_signatures):
        values: List[int] = []
        for selector_name in selectors:
            rows = family_rows_by_selector.get(selector_name, [])
            value = 0
            for row in rows:
                if str(row.get("predicate_signature", "")) == signature:
                    value = _safe_int(row.get("num_rules", 0))
                    break
            values.append(value)
        ax.bar(labels, values, bottom=bottoms, color=colors[idx % len(colors)], label=signature)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    other_values: List[int] = []
    for selector_name in selectors:
        rows = family_rows_by_selector.get(selector_name, [])
        total_rules = sum(_safe_int(row.get("num_rules", 0)) for row in rows)
        top_rules = sum(
            _safe_int(row.get("num_rules", 0))
            for row in rows
            if str(row.get("predicate_signature", "")) in top_signatures
        )
        other_values.append(max(0, total_rules - top_rules))
    ax.bar(labels, other_values, bottom=bottoms, color="#d9d9d9", label="other")

    ax.set_title("Selected Rule-Family Composition", loc="left", fontweight="bold")
    ax.set_ylabel("Selected Rules")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_confidence_support_scatter(
    rule_eval_rows_by_selector: Dict[str, List[Dict[str, Any]]],
    figure_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selectors = [name for name in _SELECTOR_ORDER if name in rule_eval_rows_by_selector]
    fig, axes = plt.subplots(1, max(1, len(selectors)), figsize=(5.2 * max(1, len(selectors)), 4.8), squeeze=False)
    axes_row = axes[0]

    for axis, selector_name in zip(axes_row, selectors):
        rows = rule_eval_rows_by_selector.get(selector_name, [])
        x_values = [_safe_int(row.get("positive_support", 0)) for row in rows]
        y_values = [_safe_float(row.get("confidence", 0.0)) for row in rows]
        sizes = [30 + 12 * _safe_int(row.get("eval_positive_support", 0)) for row in rows]
        color = _SELECTOR_COLORS.get(selector_name, "#888888")
        axis.scatter(x_values, y_values, s=sizes, alpha=0.55, color=color, edgecolors="white", linewidths=0.5)
        axis.set_title(_SELECTOR_LABELS.get(selector_name, selector_name), loc="left", fontweight="bold")
        axis.set_xlabel("Positive Support")
        axis.set_ylabel("Confidence")
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.2)

    for axis in axes_row[len(selectors):]:
        axis.axis("off")

    fig.suptitle("Confidence Vs Positive Support", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_vehicle_family_coverage(
    pool_summary_rows: Sequence[Dict[str, Any]],
    eval_context_rows: Sequence[Dict[str, Any]],
    figure_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selectors = [name for name in _SELECTOR_ORDER]
    selector_pool_names = {
        "original": "selected_original",
        "diverse": "selected_diverse",
        "coverage_family_aware": "selected_coverage_family_aware",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    labels = [_SELECTOR_LABELS.get(name, name) for name in selectors]
    bottoms = [0 for _ in selectors]
    colors = list(plt.cm.Set2.colors) + list(plt.cm.Set3.colors)
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
        axes[0].bar(labels, values, bottom=bottoms, color=colors[idx % len(colors)], label=_MATCH_LEVEL_LABELS.get(match_level, match_level))
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    axes[0].set_title("Vehicle/Near/Centered Rule Coverage", loc="left", fontweight="bold")
    axes[0].set_ylabel("Selected Rules")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1.0), loc="upper left")

    vehicle_recalls: List[float] = []
    for selector_name in selectors:
        total_positive, predicted_positive = _load_vehicle_context_totals(eval_context_rows, selector_name)
        vehicle_recalls.append(float(predicted_positive / max(1, total_positive)))
    bars = axes[1].bar(labels, vehicle_recalls, color=[_SELECTOR_COLORS.get(name, "#888888") for name in selectors])
    axes[1].set_title("Vehicle-Centered Positive Recall", loc="left", fontweight="bold")
    axes[1].set_ylabel("Recall")
    axes[1].set_ylim(0.0, 1.08)
    axes[1].grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, vehicle_recalls):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, min(1.02, value + 0.02), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_visualization(
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)

    figure_format = str(cfg.get("figure_format", "png")).strip().lower() or "png"
    dpi = int(cfg.get("dpi", 160))
    top_rule_families = int(cfg.get("top_rule_families", 8))

    manifest_path = out_root / "rule_selection_visualization_manifest.json"
    figure_paths = {
        "selector_metrics": out_root / f"selector_metrics.{figure_format}",
        "fn_comparison": out_root / f"fn_comparison.{figure_format}",
        "rule_family_composition": out_root / f"rule_family_composition.{figure_format}",
        "confidence_vs_positive_support": out_root / f"confidence_vs_positive_support.{figure_format}",
        "vehicle_family_coverage": out_root / f"vehicle_family_coverage.{figure_format}",
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
    eval_context_rows = _read_csv(step20_root / "vehicle_centered_eval_context_summary.csv")
    vehicle_manifest = _read_json(step20_root / "vehicle_centered_diagnostic_summary.json")

    family_rows_by_selector = {
        selector_name: _selector_rule_family_rows(step19_root, step19_primary, selector_name)
        for selector_name in _SELECTOR_ORDER
    }
    rule_eval_rows_by_selector = {
        selector_name: _selector_rule_eval_rows(step18_root, step18_primary, selector_name)
        for selector_name in _SELECTOR_ORDER
    }

    _plot_selector_metrics(step18_rows, figure_paths["selector_metrics"], dpi)
    _plot_fn_metrics(step19_rows, figure_paths["fn_comparison"], dpi)
    _plot_rule_family_composition(family_rows_by_selector, figure_paths["rule_family_composition"], dpi, top_rule_families)
    _plot_confidence_support_scatter(rule_eval_rows_by_selector, figure_paths["confidence_vs_positive_support"], dpi)
    _plot_vehicle_family_coverage(pool_summary_rows, eval_context_rows, figure_paths["vehicle_family_coverage"], dpi)

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
            "step20_eval_context_csv": str(step20_root / "vehicle_centered_eval_context_summary.csv"),
            "step20_manifest_json": str(step20_root / "vehicle_centered_diagnostic_summary.json"),
        },
        "figure_paths": {name: str(path) for name, path in figure_paths.items()},
        "step18_rows": step18_rows,
        "step19_rows": step19_rows,
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(
        "  rule_selection_visualization: "
        f"selectors={len(_SELECTOR_ORDER)} | "
        f"figures={len(figure_paths)} | "
        f"diagnosis={vehicle_manifest.get('primary_diagnosis', 'unknown')}"
    )
    print(f"Rule selection visualization manifest written to {manifest_path}")
    for figure_path in figure_paths.values():
        print(f"Figure written to {figure_path}")
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
