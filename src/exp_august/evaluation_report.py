"""Publication-ready PDF charts for the Step 8 test evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def _result_table(axis, rows, title):
    axis.axis("off")
    axis.set_title(title, fontweight="bold", loc="left", pad=8)
    table = axis.table(cellText=rows, colLabels=("Measure", "Value"), cellLoc="left", loc="center", colWidths=(0.67, 0.28))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#D8DEEA")
        if row == 0:
            cell.set_facecolor("#EAF0FA")
            cell.set_text_props(weight="bold", color="#172033")


def write_test_evaluation_pdf(results: dict[str, Any], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = results["aggregate"]
    frame = aggregate["frame_classification"]
    segment = aggregate["segment_evaluation"]
    boundaries = aggregate["boundary_detection"]
    matching = results["matching"]

    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(16, 10), constrained_layout=True)
        grid = figure.add_gridspec(3, 3, height_ratios=(0.16, 0.39, 0.45), width_ratios=(1, 1, 1.25))
        title = figure.add_subplot(grid[0, :])
        title.axis("off")
        title.text(0, 0.82, "CauVid Step 8 - Test Split Evaluation", fontsize=22, fontweight="bold", color="#172033")
        title.text(
            0,
            0.34,
            f"Seed {results['config']['seed']} | annotated test videos only | generated from the persisted run split",
            fontsize=12,
            color="#536079",
        )
        _result_table(
            figure.add_subplot(grid[1, 0]),
            [
                ("Selected test videos", str(matching["selected_for_split"])),
                ("Matched predictions", str(matching["matched"])),
                ("Missing predictions", str(matching["missing_predictions"])),
                ("Invalid predictions", str(matching["invalid_predictions"])),
                ("Evaluated frames", str(frame["num_frames"])),
                ("GT / predicted segments", f"{segment['num_gt_segments']} / {segment['num_predicted_segments']}"),
            ],
            "Evaluation Coverage",
        )
        _result_table(
            figure.add_subplot(grid[1, 1]),
            [
                ("Frame accuracy", f"{frame['accuracy']:.4f}"),
                ("Macro F1", f"{frame['macro_f1']:.4f}"),
                ("Weighted F1", f"{frame['weighted_f1']:.4f}"),
                ("Mean matched IoU", f"{segment['mean_matched_iou']:.4f}"),
                ("Segment IoU", f"{segment['segment_iou']:.4f}"),
                ("Label-aware segment IoU", f"{segment['label_aware_segment_iou']:.4f}"),
            ],
            "Aggregate Results",
        )
        boundary_axis = figure.add_subplot(grid[2, 0])
        tolerances = sorted(boundaries, key=lambda value: int(value))
        x = [int(value) for value in tolerances]
        for metric, color in (("precision", "#2563EB"), ("recall", "#E8792E"), ("f1", "#0F9D75")):
            boundary_axis.plot(x, [boundaries[value][metric] for value in tolerances], marker="o", linewidth=2.2, label=metric.title(), color=color)
        boundary_axis.set_ylim(0, 1.05)
        boundary_axis.set_xlabel("Boundary tolerance (frames)")
        boundary_axis.set_ylabel("Score")
        boundary_axis.set_title("Boundary Detection", fontweight="bold")
        boundary_axis.grid(alpha=0.22)
        boundary_axis.legend(fontsize=8)
        class_axis = figure.add_subplot(grid[2, 1])
        classes = list(frame["per_class"])
        values = [frame["per_class"][label]["f1"] for label in classes]
        class_axis.barh(classes, values, color="#6D5BD0")
        class_axis.set_xlim(0, 1.05)
        class_axis.set_xlabel("F1")
        class_axis.set_title("Per-class Frame F1", fontweight="bold")
        class_axis.grid(axis="x", alpha=0.22)
        matrix = np.asarray(frame["confusion_matrix"], dtype=float)
        confusion_axis = figure.add_subplot(grid[1:, 2])
        image = confusion_axis.imshow(matrix, cmap="Blues", aspect="equal")
        labels = frame["confusion_matrix_labels"]
        confusion_axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        confusion_axis.set_yticks(range(len(labels)), labels)
        confusion_axis.set_xlabel("Predicted")
        confusion_axis.set_ylabel("Ground truth")
        confusion_axis.set_title("Confusion Matrix", fontweight="bold")
        figure.colorbar(image, ax=confusion_axis, label="Frames", shrink=0.76)
        pdf.savefig(figure, dpi=180)
        plt.close(figure)
    return path
