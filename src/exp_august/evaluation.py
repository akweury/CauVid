"""Standalone evaluation for exp_august temporal video segmentation.

The evaluator is deliberately read-only with respect to predictions.  Manual
annotations are used only for scoring after a deterministic dev/test split has
been constructed from video IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


CLASSES = ("forward", "backward", "left", "right", "static")
DEFAULT_TOLERANCES = (1, 3, 5, 10)

_LABEL_ALIASES = {
    "moving_forward": "forward",
    "forward": "forward",
    "driving_forward": "forward",
    "moving_backward": "backward",
    "backward": "backward",
    "driving_backward": "backward",
    "turning_left": "left",
    "turn_left": "left",
    "left": "left",
    "turning_right": "right",
    "turn_right": "right",
    "right": "right",
    "stationary": "static",
    "stopping": "static",
    "stopped": "static",
    "stop": "static",
    "static": "static",
}


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    label: str

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class VideoTimeline:
    video_id: str
    num_frames: int
    segments: tuple[Segment, ...]
    source_path: str = ""
    fps: Optional[float] = None


def _safe_int(value: Any) -> Optional[int]:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clean_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def canonical_label(value: Any, *, forward_event: Any = None, lateral_event: Any = None) -> Optional[str]:
    """Map annotation or August combined motion labels to the five classes."""
    direct = _LABEL_ALIASES.get(_clean_label(value))
    if direct is not None:
        return direct

    text = _clean_label(value)
    forward = _clean_label(forward_event)
    lateral = _clean_label(lateral_event)
    if "|" in str(value):
        raw_forward, raw_lateral = str(value).split("|", 1)
        forward = forward or _clean_label(raw_forward)
        lateral = lateral or _clean_label(raw_lateral)

    # Lateral motion is the semantic class whenever a turn is active.
    if lateral in {"left", "turning_left", "turn_left"} or "turning_left" in text:
        return "left"
    if lateral in {"right", "turning_right", "turn_right"} or "turning_right" in text:
        return "right"
    if forward.startswith("backward") or text.startswith("backward"):
        return "backward"
    if forward in {"stop", "stopping", "stopped", "stationary", "static"}:
        return "static"
    if any(token in forward for token in ("forward", "speedup", "slowdown", "moving")):
        return "forward"
    if any(token in text for token in ("forward", "speedup", "slowdown")):
        return "forward"
    return None


def _merge_adjacent(segments: Sequence[Segment]) -> tuple[Segment, ...]:
    merged: list[Segment] = []
    for segment in segments:
        if merged and merged[-1].label == segment.label and merged[-1].end + 1 == segment.start:
            previous = merged[-1]
            merged[-1] = Segment(previous.start, segment.end, previous.label)
        else:
            merged.append(segment)
    return tuple(merged)


def _video_id_from_annotation(payload: dict[str, Any], path: Path) -> str:
    video = payload.get("video", {})
    if isinstance(video, dict):
        source_path = str(video.get("path", "")).strip()
        if source_path:
            return Path(source_path).stem
        raw_id = str(video.get("id", "")).strip()
        if raw_id:
            raw_id = raw_id.removeprefix("videos__")
            match = re.match(r"^([0-9a-f]{8}-[0-9a-f]{8})", raw_id, re.IGNORECASE)
            return match.group(1) if match else raw_id
    name = path.stem.removeprefix("videos__")
    match = re.match(r"^([0-9a-f]{8}-[0-9a-f]{8})", name, re.IGNORECASE)
    return match.group(1) if match else name


def load_annotation(path: Path) -> tuple[Optional[VideoTimeline], Optional[str]]:
    """Infer and validate the repository's existing annotation schema."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable_json: {exc}"
    if not isinstance(payload, dict):
        return None, "top_level_must_be_object"

    video = payload.get("video")
    raw_segments = payload.get("segments")
    if not isinstance(video, dict) or not isinstance(raw_segments, list):
        return None, "unsupported_schema: expected video object and segments list"
    frame_count = _safe_int(video.get("frame_count"))
    if frame_count is None or frame_count <= 0:
        return None, "invalid_video_frame_count"
    if not raw_segments:
        return None, "no_manual_segments"

    segments: list[Segment] = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            return None, f"segment_{index}_must_be_object"
        start = _safe_int(raw.get("start_frame"))
        end = _safe_int(raw.get("end_frame"))
        label = canonical_label(raw.get("label"))
        if start is None or end is None or start < 0 or end < start or end >= frame_count:
            return None, f"segment_{index}_invalid_bounds"
        if label is None:
            return None, f"segment_{index}_unknown_label:{raw.get('label')}"
        if segments and start != segments[-1].end + 1:
            return None, f"segment_{index}_non_contiguous"
        segments.append(Segment(start, end, label))
    if segments[0].start != 0 or segments[-1].end != frame_count - 1:
        return None, "segments_do_not_cover_video"

    return (
        VideoTimeline(
            video_id=_video_id_from_annotation(payload, path),
            num_frames=frame_count,
            segments=_merge_adjacent(segments),
            source_path=str(path),
            fps=_safe_float(video.get("fps")),
        ),
        None,
    )


def _segments_from_frame_labels(labels: Sequence[Any]) -> tuple[Segment, ...]:
    segments: list[Segment] = []
    for index, raw_label in enumerate(labels):
        label = canonical_label(raw_label)
        if label is None:
            raise ValueError(f"unknown frame label at {index}: {raw_label}")
        if segments and segments[-1].label == label:
            previous = segments[-1]
            segments[-1] = Segment(previous.start, index, label)
        else:
            segments.append(Segment(index, index, label))
    return tuple(segments)


def adapt_prediction(payload: dict[str, Any], source_path: Path) -> tuple[Optional[VideoTimeline], Optional[str]]:
    """Adapt an exp_august temporal-segmentation result without changing it."""
    video_id = str(payload.get("video_id", "")).strip()
    if not video_id:
        return None, "missing_video_id"
    num_frames = _safe_int(payload.get("num_frames"))
    raw_segments = payload.get("segments")

    try:
        if isinstance(raw_segments, list) and raw_segments:
            segments: list[Segment] = []
            for index, raw in enumerate(raw_segments):
                if not isinstance(raw, dict):
                    return None, f"segment_{index}_must_be_object"
                start = _safe_int(raw.get("start_frame", raw.get("start_idx")))
                end = _safe_int(raw.get("end_frame", raw.get("end_idx")))
                label = canonical_label(
                    raw.get("label", raw.get("event")),
                    forward_event=raw.get("forward_event"),
                    lateral_event=raw.get("lateral_event"),
                )
                if start is None or end is None or start < 0 or end < start:
                    return None, f"segment_{index}_invalid_bounds"
                if label is None:
                    return None, f"segment_{index}_unknown_label"
                segments.append(Segment(start, end, label))
            segments.sort(key=lambda row: (row.start, row.end))
            if num_frames is None:
                num_frames = max(row.end for row in segments) + 1
        else:
            frame_labels = payload.get("primary_event")
            if not isinstance(frame_labels, list) or not frame_labels:
                return None, "no_segments_or_primary_event"
            segments = list(_segments_from_frame_labels(frame_labels))
            num_frames = len(frame_labels)
    except ValueError as exc:
        return None, str(exc)

    if num_frames is None or num_frames <= 0:
        return None, "invalid_num_frames"
    if any(segment.end >= num_frames for segment in segments):
        return None, "segment_outside_prediction_timeline"
    for previous, current in zip(segments, segments[1:]):
        if current.start <= previous.end:
            return None, "overlapping_prediction_segments"
    return (
        VideoTimeline(
            video_id=video_id,
            num_frames=num_frames,
            segments=_merge_adjacent(segments),
            source_path=str(source_path),
            fps=_safe_float(payload.get("fps")),
        ),
        None,
    )


def _prediction_payloads(payload: Any, *, allow_direct: bool = False) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        if allow_direct and payload.get("video_id") and (
            isinstance(payload.get("segments"), list)
            or isinstance(payload.get("primary_event"), list)
        ):
            yield payload
        temporal = payload.get("temporal_segments")
        if isinstance(temporal, list):
            for row in temporal:
                if isinstance(row, dict):
                    yield row
    elif isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict) and row.get("video_id"):
                yield row


def discover_predictions(root: Path) -> tuple[dict[str, VideoTimeline], list[dict[str, str]]]:
    files = [root] if root.is_file() else sorted(root.rglob("*.json"))
    predictions: dict[str, VideoTimeline] = {}
    invalid: list[dict[str, str]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        allow_direct = root.is_file() or path.name == "temporal_segmentation.json"
        for candidate in _prediction_payloads(payload, allow_direct=allow_direct):
            timeline, error = adapt_prediction(candidate, path)
            if timeline is None:
                invalid.append(
                    {
                        "video_id": str(candidate.get("video_id", "unknown")),
                        "path": str(path),
                        "reason": str(error),
                    }
                )
                continue
            existing = predictions.get(timeline.video_id)
            if existing is None or path.name == "temporal_segmentation.json":
                predictions[timeline.video_id] = timeline
    return predictions, invalid


def deterministic_split(video_ids: Sequence[str], seed: int, test_ratio: float) -> dict[str, list[str]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be strictly between 0 and 1")
    ordered = sorted(
        set(video_ids),
        key=lambda value: (
            hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )
    if len(ordered) <= 1:
        test_count = len(ordered)
    else:
        test_count = max(1, min(len(ordered) - 1, int(round(len(ordered) * test_ratio))))
    return {"test": sorted(ordered[:test_count]), "dev": sorted(ordered[test_count:])}


def _rescale_prediction(prediction: VideoTimeline, target_frames: int) -> tuple[Segment, ...]:
    """Map the processed prediction timeline onto the annotated raw timeline."""
    if prediction.num_frames == target_frames:
        return prediction.segments
    scale = target_frames / prediction.num_frames
    scaled: list[Segment] = []
    for segment in prediction.segments:
        start = max(0, min(target_frames - 1, int(math.floor(segment.start * scale))))
        end = max(start, min(target_frames - 1, int(math.ceil((segment.end + 1) * scale) - 1)))
        scaled.append(Segment(start, end, segment.label))
    return _merge_adjacent(scaled)


def _frame_metrics(gt: VideoTimeline, predicted_segments: Sequence[Segment]) -> dict[str, Any]:
    gt_labels: list[Optional[str]] = [None] * gt.num_frames
    pred_labels: list[Optional[str]] = [None] * gt.num_frames
    for segment in gt.segments:
        gt_labels[segment.start : segment.end + 1] = [segment.label] * segment.length
    for segment in predicted_segments:
        pred_labels[segment.start : segment.end + 1] = [segment.label] * segment.length

    confusion = [[0 for _ in CLASSES] for _ in CLASSES]
    missing_by_class = [0 for _ in CLASSES]
    class_index = {label: index for index, label in enumerate(CLASSES)}
    correct = 0
    for truth, prediction in zip(gt_labels, pred_labels):
        truth_index = class_index[str(truth)]
        if prediction is None:
            missing_by_class[truth_index] += 1
        else:
            prediction_index = class_index[prediction]
            confusion[truth_index][prediction_index] += 1
            correct += truth == prediction

    per_class: dict[str, dict[str, Any]] = {}
    f1_values = []
    weighted_f1_numerator = 0.0
    for index, label in enumerate(CLASSES):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(CLASSES)) if row != index)
        support = sum(confusion[index]) + missing_by_class[index]
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        f1_values.append(f1)
        weighted_f1_numerator += f1 * support
    return {
        "num_frames": gt.num_frames,
        "predicted_frames": gt.num_frames - sum(missing_by_class),
        "unpredicted_frames": sum(missing_by_class),
        "accuracy": correct / gt.num_frames,
        "macro_f1": sum(f1_values) / len(CLASSES),
        "weighted_f1": weighted_f1_numerator / gt.num_frames,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "confusion_matrix_labels": list(CLASSES),
        "unpredicted_by_class": dict(zip(CLASSES, missing_by_class)),
    }


def _boundaries(segments: Sequence[Segment]) -> list[int]:
    return [segment.start for segment in segments[1:]]


def match_boundaries(gt: Sequence[int], prediction: Sequence[int], tolerance: int) -> list[tuple[int, int]]:
    """One-to-one optimal ordered matching: max matches, then min distance."""
    n, m = len(gt), len(prediction)
    scores: list[list[tuple[int, int, tuple[tuple[int, int], ...]]]] = [
        [(0, 0, ()) for _ in range(m + 1)] for _ in range(n + 1)
    ]

    def better(left, right):
        return left if (left[0], -left[1], left[2]) >= (right[0], -right[1], right[2]) else right

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = better(scores[i - 1][j], scores[i][j - 1])
            distance = abs(int(gt[i - 1]) - int(prediction[j - 1]))
            if distance <= tolerance:
                previous = scores[i - 1][j - 1]
                candidate = (
                    previous[0] + 1,
                    previous[1] + distance,
                    previous[2] + ((int(gt[i - 1]), int(prediction[j - 1])),),
                )
                best = better(best, candidate)
            scores[i][j] = best
    return list(scores[n][m][2])


def _boundary_metrics(gt_segments: Sequence[Segment], pred_segments: Sequence[Segment], tolerances: Sequence[int]) -> dict[str, Any]:
    gt = _boundaries(gt_segments)
    prediction = _boundaries(pred_segments)
    results: dict[str, Any] = {}
    for tolerance in tolerances:
        pairs = match_boundaries(gt, prediction, int(tolerance))
        tp = len(pairs)
        fp = len(prediction) - tp
        fn = len(gt) - tp
        precision = tp / (tp + fp) if tp + fp else (1.0 if not gt else 0.0)
        recall = tp / (tp + fn) if tp + fn else (1.0 if not prediction else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results[str(int(tolerance))] = {
            "tolerance_frames": int(tolerance),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "matches": [
                {"gt_frame": a, "predicted_frame": b, "distance": abs(a - b)}
                for a, b in pairs
            ],
        }
    return results


def temporal_iou(left: Segment, right: Segment) -> float:
    intersection = max(0, min(left.end, right.end) - max(left.start, right.start) + 1)
    union = left.length + right.length - intersection
    return intersection / union if union else 0.0


def _maximum_weight_assignment(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Dependency-free Hungarian assignment maximizing total matrix weight."""
    rows = len(weights)
    cols = max((len(row) for row in weights), default=0)
    if rows == 0 or cols == 0:
        return []
    size = max(rows, cols)
    maximum = max((max(row, default=0.0) for row in weights), default=0.0)
    cost = [[maximum for _ in range(size)] for _ in range(size)]
    for i, row in enumerate(weights):
        for j, value in enumerate(row):
            cost[i][j] = maximum - float(value)

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j] = current
                    way[j] = j0
                if minimum[j] < delta:
                    delta = minimum[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [(p[j] - 1, j - 1) for j in range(1, size + 1) if p[j]]
    return [(i, j) for i, j in assignment if i < rows and j < cols]


def _segment_metrics(gt_segments: Sequence[Segment], pred_segments: Sequence[Segment]) -> dict[str, Any]:
    weights = [[temporal_iou(gt, pred) for pred in pred_segments] for gt in gt_segments]
    assignment = _maximum_weight_assignment(weights)
    label_weights = [
        [value if gt_segments[i].label == pred_segments[j].label else 0.0 for j, value in enumerate(row)]
        for i, row in enumerate(weights)
    ]
    label_assignment = _maximum_weight_assignment(label_weights)
    matches = []
    matched_sum = 0.0
    positive_matches = 0
    for gt_index, pred_index in assignment:
        iou = weights[gt_index][pred_index]
        if iou <= 0:
            continue
        gt, pred = gt_segments[gt_index], pred_segments[pred_index]
        label_match = gt.label == pred.label
        matches.append(
            {
                "gt_segment_index": gt_index,
                "predicted_segment_index": pred_index,
                "gt": asdict(gt),
                "prediction": asdict(pred),
                "temporal_iou": iou,
                "label_match": label_match,
                "label_aware_iou": iou if label_match else 0.0,
            }
        )
        matched_sum += iou
        positive_matches += 1
    denominator = max(len(gt_segments), len(pred_segments), 1)
    label_aware_sum = sum(
        label_weights[gt_index][pred_index]
        for gt_index, pred_index in label_assignment
        if label_weights[gt_index][pred_index] > 0
    )
    return {
        "num_gt_segments": len(gt_segments),
        "num_predicted_segments": len(pred_segments),
        "num_matched_segments": positive_matches,
        "mean_matched_iou": matched_sum / positive_matches if positive_matches else 0.0,
        "segment_iou": matched_sum / denominator,
        "label_aware_segment_iou": label_aware_sum / denominator,
        "matches": matches,
    }


def evaluate_video(gt: VideoTimeline, prediction: VideoTimeline, tolerances: Sequence[int] = DEFAULT_TOLERANCES) -> dict[str, Any]:
    scaled = _rescale_prediction(prediction, gt.num_frames)
    return {
        "video_id": gt.video_id,
        "annotation_path": gt.source_path,
        "prediction_path": prediction.source_path,
        "alignment": {
            "method": "identity" if gt.num_frames == prediction.num_frames else "normalized_full_duration",
            "annotation_frames": gt.num_frames,
            "prediction_frames": prediction.num_frames,
            "scale": gt.num_frames / prediction.num_frames,
        },
        "frame_classification": _frame_metrics(gt, scaled),
        "boundary_detection": _boundary_metrics(gt.segments, scaled, tolerances),
        "segment_evaluation": _segment_metrics(gt.segments, scaled),
    }


def _aggregate(per_video: Sequence[dict[str, Any]], tolerances: Sequence[int]) -> dict[str, Any]:
    confusion = [[0 for _ in CLASSES] for _ in CLASSES]
    missing = {label: 0 for label in CLASSES}
    total_frames = 0
    for result in per_video:
        frame = result["frame_classification"]
        total_frames += frame["num_frames"]
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                confusion[i][j] += frame["confusion_matrix"][i][j]
            missing[CLASSES[i]] += frame["unpredicted_by_class"][CLASSES[i]]

    # Recompute classification summaries directly from the pooled confusion.
    per_class = {}
    weighted = 0.0
    correct = 0
    for i, label in enumerate(CLASSES):
        tp = confusion[i][i]
        fp = sum(confusion[row][i] for row in range(len(CLASSES)) if row != i)
        support = sum(confusion[i]) + missing[label]
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support, "tp": tp, "fp": fp, "fn": fn}
        weighted += f1 * support
        correct += tp
    frame_aggregate = {
        "num_frames": total_frames,
        "accuracy": correct / total_frames if total_frames else 0.0,
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(CLASSES),
        "weighted_f1": weighted / total_frames if total_frames else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "confusion_matrix_labels": list(CLASSES),
        "unpredicted_frames": sum(missing.values()),
        "unpredicted_by_class": missing,
    }

    boundary_aggregate = {}
    for tolerance in tolerances:
        rows = [result["boundary_detection"][str(int(tolerance))] for result in per_video]
        tp, fp, fn = sum(row["tp"] for row in rows), sum(row["fp"] for row in rows), sum(row["fn"] for row in rows)
        precision = tp / (tp + fp) if tp + fp else (1.0 if rows and fn == 0 else 0.0)
        recall = tp / (tp + fn) if tp + fn else (1.0 if rows and fp == 0 else 0.0)
        boundary_aggregate[str(int(tolerance))] = {
            "tolerance_frames": int(tolerance), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
        }

    segment_rows = [result["segment_evaluation"] for result in per_video]
    match_rows = [match for row in segment_rows for match in row["matches"]]
    denominator = sum(max(row["num_gt_segments"], row["num_predicted_segments"]) for row in segment_rows)
    matched_iou_sum = sum(match["temporal_iou"] for match in match_rows)
    label_iou_sum = sum(match["label_aware_iou"] for match in match_rows)
    segment_aggregate = {
        "num_gt_segments": sum(row["num_gt_segments"] for row in segment_rows),
        "num_predicted_segments": sum(row["num_predicted_segments"] for row in segment_rows),
        "num_matched_segments": len(match_rows),
        "mean_matched_iou": matched_iou_sum / len(match_rows) if match_rows else 0.0,
        "segment_iou": matched_iou_sum / denominator if denominator else 0.0,
        "label_aware_segment_iou": label_iou_sum / denominator if denominator else 0.0,
        "mean_per_video_matched_iou": sum(row["mean_matched_iou"] for row in segment_rows) / len(segment_rows) if segment_rows else 0.0,
    }
    return {"frame_classification": frame_aggregate, "boundary_detection": boundary_aggregate, "segment_evaluation": segment_aggregate}


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _svg_text(x: float, y: float, value: Any, size: int = 14, anchor: str = "middle", weight: str = "normal") -> str:
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}">{html.escape(str(value))}</text>'


def _write_confusion_svg(path: Path, matrix: Sequence[Sequence[int]]) -> None:
    width, height, left, top, cell = 820, 720, 170, 100, 92
    maximum = max((max(row, default=0) for row in matrix), default=1) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(_svg_text(width / 2, 38, "Frame-level confusion matrix", 24, weight="bold"))
    parts.append(_svg_text(width / 2, 66, "Rows: ground truth · Columns: prediction", 14))
    for i, label in enumerate(CLASSES):
        parts.append(_svg_text(left + i * cell + cell / 2, top - 16, label, 14))
        parts.append(_svg_text(left - 16, top + i * cell + cell / 2 + 5, label, 14, anchor="end"))
        row_total = sum(matrix[i])
        for j in range(len(CLASSES)):
            value = matrix[i][j]
            intensity = value / maximum
            blue = int(245 - 155 * intensity)
            color = f"rgb({blue},{blue + 4},{255})"
            x, y = left + j * cell, top + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff"/>')
            parts.append(_svg_text(x + cell / 2, y + cell / 2 - 3, value, 18, weight="bold"))
            percentage = 100 * value / row_total if row_total else 0.0
            parts.append(_svg_text(x + cell / 2, y + cell / 2 + 20, f"{percentage:.1f}%", 12))
    parts.append(_svg_text(left + len(CLASSES) * cell / 2, top + len(CLASSES) * cell + 48, "Predicted class", 16, weight="bold"))
    parts.append(f'<text x="38" y="{top + len(CLASSES) * cell / 2}" transform="rotate(-90 38 {top + len(CLASSES) * cell / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold">Ground-truth class</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_summary_svg(path: Path, aggregate: dict[str, Any], tolerances: Sequence[int]) -> None:
    metrics = [
        ("Accuracy", aggregate["frame_classification"]["accuracy"]),
        ("Macro-F1", aggregate["frame_classification"]["macro_f1"]),
        ("Weighted F1", aggregate["frame_classification"]["weighted_f1"]),
        ("Matched tIoU", aggregate["segment_evaluation"]["mean_matched_iou"]),
        ("Label-aware sIoU", aggregate["segment_evaluation"]["label_aware_segment_iou"]),
    ] + [
        (f"Boundary F1 ±{int(t)}", aggregate["boundary_detection"][str(int(t))]["f1"])
        for t in tolerances
    ]
    width, height, left, top = 980, 620, 230, 80
    bar_height, gap, chart_width = 38, 18, 650
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', _svg_text(width / 2, 38, "exp_august segmentation evaluation", 24, weight="bold")]
    for index, (label, value) in enumerate(metrics):
        y = top + index * (bar_height + gap)
        parts.append(_svg_text(left - 14, y + 25, label, 14, anchor="end"))
        parts.append(f'<rect x="{left}" y="{y}" width="{chart_width}" height="{bar_height}" fill="#eef1f5" rx="3"/>')
        parts.append(f'<rect x="{left}" y="{y}" width="{chart_width * max(0.0, min(1.0, value)):.1f}" height="{bar_height}" fill="#3569a8" rx="3"/>')
        parts.append(_svg_text(left + chart_width + 12, y + 25, f"{value:.3f}", 14, anchor="start", weight="bold"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _save_outputs(output_root: Path, results: dict[str, Any], tolerances: Sequence[int]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "evaluation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    per_video_rows = []
    boundary_rows = []
    matching_rows = []
    per_video_class_rows = []
    for result in results["per_video"]:
        frame, segment = result["frame_classification"], result["segment_evaluation"]
        row = {
            "video_id": result["video_id"], "accuracy": frame["accuracy"], "macro_f1": frame["macro_f1"],
            "weighted_f1": frame["weighted_f1"], "mean_matched_iou": segment["mean_matched_iou"],
            "segment_iou": segment["segment_iou"], "label_aware_segment_iou": segment["label_aware_segment_iou"],
            "annotation_frames": result["alignment"]["annotation_frames"], "prediction_frames": result["alignment"]["prediction_frames"],
        }
        for tolerance in tolerances:
            row[f"boundary_f1_t{int(tolerance)}"] = result["boundary_detection"][str(int(tolerance))]["f1"]
            boundary_rows.append({"scope": "video", "video_id": result["video_id"], **result["boundary_detection"][str(int(tolerance))]})
        per_video_rows.append(row)
        for label, class_metrics in frame["per_class"].items():
            per_video_class_rows.append(
                {"video_id": result["video_id"], "class": label, **class_metrics}
            )
        for match in segment["matches"]:
            matching_rows.append({"video_id": result["video_id"], **{key: value for key, value in match.items() if key not in {"gt", "prediction"}}, "gt_label": match["gt"]["label"], "predicted_label": match["prediction"]["label"], "gt_start": match["gt"]["start"], "gt_end": match["gt"]["end"], "predicted_start": match["prediction"]["start"], "predicted_end": match["prediction"]["end"]})
    aggregate = results["aggregate"]
    for tolerance in tolerances:
        boundary_rows.append({"scope": "aggregate", "video_id": "ALL", **aggregate["boundary_detection"][str(int(tolerance))]})
    _write_csv(output_root / "per_video_metrics.csv", per_video_rows)
    _write_csv(output_root / "per_video_class_metrics.csv", per_video_class_rows)
    _write_csv(output_root / "boundary_metrics.csv", boundary_rows)
    _write_csv(output_root / "segment_matches.csv", matching_rows)
    confusion_rows = [{"ground_truth": label, **dict(zip(CLASSES, aggregate["frame_classification"]["confusion_matrix"][i]))} for i, label in enumerate(CLASSES)]
    _write_csv(output_root / "confusion_matrix.csv", confusion_rows)
    aggregate_rows = [
        {"level": "frame", "metric": "accuracy", "value": aggregate["frame_classification"]["accuracy"]},
        {"level": "frame", "metric": "macro_f1", "value": aggregate["frame_classification"]["macro_f1"]},
        {"level": "frame", "metric": "weighted_f1", "value": aggregate["frame_classification"]["weighted_f1"]},
        {"level": "segment", "metric": "mean_matched_iou", "value": aggregate["segment_evaluation"]["mean_matched_iou"]},
        {"level": "segment", "metric": "segment_iou", "value": aggregate["segment_evaluation"]["segment_iou"]},
        {"level": "segment", "metric": "label_aware_segment_iou", "value": aggregate["segment_evaluation"]["label_aware_segment_iou"]},
    ]
    for label, metrics in aggregate["frame_classification"]["per_class"].items():
        for metric in ("precision", "recall", "f1"):
            aggregate_rows.append({"level": "frame_class", "class": label, "metric": metric, "value": metrics[metric]})
    for tolerance in tolerances:
        for metric in ("precision", "recall", "f1"):
            aggregate_rows.append({"level": "boundary", "tolerance": int(tolerance), "metric": metric, "value": aggregate["boundary_detection"][str(int(tolerance))][metric]})
    _write_csv(output_root / "aggregate_metrics.csv", aggregate_rows)
    _write_confusion_svg(output_root / "confusion_matrix.svg", aggregate["frame_classification"]["confusion_matrix"])
    _write_summary_svg(output_root / "metric_summary.svg", aggregate, tolerances)


def evaluate_dataset(
    predictions_root: Path | str,
    annotations_root: Path | str,
    output_root: Path | str,
    *,
    split: str = "test",
    seed: int = 20260809,
    test_ratio: float = 0.2,
    tolerances: Sequence[int] = DEFAULT_TOLERANCES,
) -> dict[str, Any]:
    annotations_path = Path(annotations_root)
    prediction_path = Path(predictions_root)
    output_path = Path(output_root)
    if split not in {"dev", "test", "all"}:
        raise ValueError("split must be dev, test, or all")
    tolerances = tuple(sorted({int(value) for value in tolerances if int(value) >= 0}))
    if not tolerances:
        raise ValueError("at least one non-negative boundary tolerance is required")

    valid_annotations: dict[str, VideoTimeline] = {}
    invalid_annotations = []
    for path in sorted(annotations_path.glob("*.json")):
        timeline, error = load_annotation(path)
        if timeline is None:
            invalid_annotations.append({"path": str(path), "reason": str(error)})
        elif timeline.video_id in valid_annotations:
            invalid_annotations.append({"path": str(path), "video_id": timeline.video_id, "reason": "duplicate_video_id"})
        else:
            valid_annotations[timeline.video_id] = timeline

    assignments = deterministic_split(list(valid_annotations), seed, test_ratio)
    selected_ids = sorted(valid_annotations) if split == "all" else assignments[split]
    predictions, invalid_predictions = discover_predictions(prediction_path)
    matched_ids = [video_id for video_id in selected_ids if video_id in predictions]
    invalid_prediction_ids = {
        str(row.get("video_id", "")) for row in invalid_predictions if row.get("video_id")
    }
    invalid_selected_ids = [
        video_id
        for video_id in selected_ids
        if video_id not in predictions and video_id in invalid_prediction_ids
    ]
    missing_ids = [
        video_id
        for video_id in selected_ids
        if video_id not in predictions and video_id not in invalid_prediction_ids
    ]
    per_video = [evaluate_video(valid_annotations[video_id], predictions[video_id], tolerances) for video_id in matched_ids]
    aggregate = _aggregate(per_video, tolerances)
    split_manifest = {
        "version": 1,
        "method": "sha256(seed:video_id)_ordered_holdout",
        "seed": int(seed),
        "test_ratio": float(test_ratio),
        "dev_video_ids": assignments["dev"],
        "test_video_ids": assignments["test"],
        "policy": "test IDs are evaluation-only and must not be used for threshold or parameter selection",
    }
    results = {
        "schema_version": 1,
        "evaluation": "exp_august_video_segmentation",
        "config": {
            "predictions": str(prediction_path), "annotations": str(annotations_path), "output": str(output_path),
            "split": split, "seed": int(seed), "test_ratio": float(test_ratio), "boundary_tolerances_frames": list(tolerances),
            "prediction_alignment": "normalized full-duration mapping to annotation frame timeline",
            "class_order": list(CLASSES), "ground_truth_used_for_parameter_selection": False,
        },
        "annotation_schema_detected": {
            "top_level": ["video", "segments"], "video_id_source": "video.path stem",
            "frame_interval": "inclusive start_frame/end_frame", "labels": sorted(_LABEL_ALIASES),
        },
        "split_manifest": split_manifest,
        "matching": {
            "annotation_files": len(list(annotations_path.glob("*.json"))),
            "valid_annotations": len(valid_annotations), "invalid_annotations": len(invalid_annotations),
            "selected_for_split": len(selected_ids), "matched": len(matched_ids), "missing_predictions": len(missing_ids),
            "invalid_predictions": len(invalid_selected_ids),
            "invalid_predictions_discovered": len(invalid_predictions), "matched_video_ids": matched_ids,
            "missing_video_ids": missing_ids, "invalid_annotation_files": invalid_annotations,
            "invalid_prediction_video_ids": invalid_selected_ids,
            "invalid_prediction_files": invalid_predictions,
        },
        "per_video": per_video,
        "aggregate": aggregate,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    _save_outputs(output_path, results, tolerances)
    return results


def compact_summary(results: dict[str, Any]) -> str:
    matching, aggregate = results["matching"], results["aggregate"]
    frame, segment = aggregate["frame_classification"], aggregate["segment_evaluation"]
    boundary = aggregate["boundary_detection"]
    boundary_text = " ".join(f"B-F1@±{key}={row['f1']:.3f}" for key, row in boundary.items())
    return (
        f"exp_august-eval split={results['config']['split']} matched={matching['matched']}/"
        f"{matching['selected_for_split']} missing={matching['missing_predictions']} "
        f"invalid_pred={matching['invalid_predictions']} invalid_ann={matching['invalid_annotations']} "
        f"Acc={frame['accuracy']:.3f} "
        f"Macro-F1={frame['macro_f1']:.3f} Weighted-F1={frame['weighted_f1']:.3f} "
        f"{boundary_text} mIoU={segment['mean_matched_iou']:.3f} "
        f"label-sIoU={segment['label_aware_segment_iou']:.3f}"
    )


def _default_annotations() -> Path:
    for candidate in (Path("annotaions/video_segmentation"), Path("annotations/video_segmentation")):
        if candidate.is_dir():
            return candidate
    return Path("annotations/video_segmentation")


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate exp_august temporal video segmentation")
    parser.add_argument("--predictions", type=Path, required=True, help="August output root, state JSON, or temporal_segmentation.json")
    parser.add_argument("--annotations", type=Path, default=_default_annotations())
    parser.add_argument("--output", type=Path, default=Path("evaluation/exp_august_video_segmentation"))
    parser.add_argument("--split", choices=("dev", "test", "all"), default="test")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--tolerances", type=int, nargs="+", default=list(DEFAULT_TOLERANCES))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = evaluate_dataset(
        args.predictions, args.annotations, args.output,
        split=args.split, seed=args.seed, test_ratio=args.test_ratio, tolerances=args.tolerances,
    )
    print(compact_summary(results))
    print(f"results={args.output / 'evaluation_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
