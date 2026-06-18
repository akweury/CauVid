"""
Train symbolic neural baselines for brake_next prediction.

Baselines:
  - single-segment MLP over fixed symbolic features
  - temporal model over short segment-history windows (GRU or temporal MLP)

Consumes:
  - Step 13 output: temporal rule-learning examples for train/eval videos
  - Train/eval split manifest from the main pipeline

Output layout:
    pipeline_output/neural_baselines_driving_mini/
        neural_baselines_summary.json
        model_comparison.csv
        model_comparison.json
        feature_vocab.json
        single_segment_mlp/
            baseline_result.json
            metrics.csv
            per_video_metrics.csv
            example_predictions.csv
            training_history.csv
        temporal_gru/ or temporal_mlp/
            baseline_result.json
            metrics.csv
            per_video_metrics.csv
            example_predictions.csv
            training_history.csv
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_BASELINE_VERSION = 2
_VARIABLE_NAMES = {"S", "O", "T", "F"}


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "neural_baselines_driving_mini"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _default_cfg() -> Dict[str, Any]:
    return {
        "min_feature_count": 1,
        "probability_threshold": 0.5,
        "random_seed": 0,
        "device": "auto",
        "single_segment_mlp": {
            "hidden_dims": [128, 64],
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "num_epochs": 100,
            "early_stopping_patience": 12,
        },
        "temporal_model": {
            "architecture": "gru",
            "history_window": 4,
            "hidden_dims": [128, 64],
            "gru_hidden_dim": 128,
            "gru_num_layers": 1,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "num_epochs": 100,
            "early_stopping_patience": 12,
        },
    }


def _normalize_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _default_cfg()
    normalized: Dict[str, Any] = {
        "min_feature_count": int(cfg.get("min_feature_count", defaults["min_feature_count"])),
        "probability_threshold": float(cfg.get("probability_threshold", defaults["probability_threshold"])),
        "random_seed": int(cfg.get("random_seed", defaults["random_seed"])),
        "device": str(cfg.get("device", defaults["device"])),
    }

    single_cfg = dict(defaults["single_segment_mlp"])
    if isinstance(cfg.get("single_segment_mlp"), dict):
        single_cfg.update(cfg["single_segment_mlp"])
    for key in list(defaults["single_segment_mlp"].keys()):
        if key in cfg:
            single_cfg[key] = cfg[key]
    single_cfg["hidden_dims"] = [int(v) for v in single_cfg.get("hidden_dims", [])]
    single_cfg["dropout"] = float(single_cfg.get("dropout", 0.1))
    single_cfg["learning_rate"] = float(single_cfg.get("learning_rate", 1e-3))
    single_cfg["weight_decay"] = float(single_cfg.get("weight_decay", 1e-4))
    single_cfg["batch_size"] = int(single_cfg.get("batch_size", 32))
    single_cfg["num_epochs"] = int(single_cfg.get("num_epochs", 100))
    single_cfg["early_stopping_patience"] = int(single_cfg.get("early_stopping_patience", 12))
    normalized["single_segment_mlp"] = single_cfg

    temporal_cfg = dict(defaults["temporal_model"])
    if isinstance(cfg.get("temporal_model"), dict):
        temporal_cfg.update(cfg["temporal_model"])
    temporal_cfg["architecture"] = str(temporal_cfg.get("architecture", "gru")).strip().lower()
    temporal_cfg["history_window"] = int(temporal_cfg.get("history_window", 4))
    temporal_cfg["hidden_dims"] = [int(v) for v in temporal_cfg.get("hidden_dims", [])]
    temporal_cfg["gru_hidden_dim"] = int(temporal_cfg.get("gru_hidden_dim", 128))
    temporal_cfg["gru_num_layers"] = int(temporal_cfg.get("gru_num_layers", 1))
    temporal_cfg["dropout"] = float(temporal_cfg.get("dropout", 0.1))
    temporal_cfg["learning_rate"] = float(temporal_cfg.get("learning_rate", 1e-3))
    temporal_cfg["weight_decay"] = float(temporal_cfg.get("weight_decay", 1e-4))
    temporal_cfg["batch_size"] = int(temporal_cfg.get("batch_size", 32))
    temporal_cfg["num_epochs"] = int(temporal_cfg.get("num_epochs", 100))
    temporal_cfg["early_stopping_patience"] = int(temporal_cfg.get("early_stopping_patience", 12))
    normalized["temporal_model"] = temporal_cfg
    return normalized


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_cfg(cfg)
    return {
        "min_feature_count": normalized["min_feature_count"],
        "probability_threshold": normalized["probability_threshold"],
        "random_seed": normalized["random_seed"],
        "device": normalized["device"],
        "single_segment_mlp": dict(normalized["single_segment_mlp"]),
        "temporal_model": dict(normalized["temporal_model"]),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def _iter_examples(video_results: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for video_result in video_results:
        video_id = str(video_result.get("video_id", ""))
        for example in list(video_result.get("examples", [])):
            row = dict(example)
            row["video_id"] = video_id
            yield row


def _example_sort_key(example: Dict[str, Any]) -> Tuple[int, int, str]:
    return (
        int(example.get("current_segment_index", -1)),
        int(example.get("example_index", -1)),
        str(example.get("example_id", "")),
    )


def _prepare_examples(video_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples = [dict(example) for example in _iter_examples(video_results)]
    for row_index, example in enumerate(examples):
        example["_row_index"] = row_index
    return examples


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


def _compute_roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    n_pos = sum(1 for label in y_true if int(label) == 1)
    n_neg = sum(1 for label in y_true if int(label) == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sorted_pairs = sorted(zip(y_score, y_true), key=lambda item: float(item[0]))
    rank_sum = 0.0
    index = 0
    while index < len(sorted_pairs):
        next_index = index + 1
        while next_index < len(sorted_pairs) and float(sorted_pairs[next_index][0]) == float(sorted_pairs[index][0]):
            next_index += 1
        avg_rank = (index + 1 + next_index) / 2.0
        num_positive = sum(1 for _, label in sorted_pairs[index:next_index] if int(label) == 1)
        rank_sum += avg_rank * num_positive
        index = next_index
    return float((rank_sum - (n_pos * (n_pos + 1) / 2.0)) / max(1e-12, n_pos * n_neg))


def _compute_average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    total_positive = sum(1 for label in y_true if int(label) == 1)
    if total_positive == 0:
        return float("nan")
    sorted_pairs = sorted(zip(y_score, y_true), key=lambda item: float(item[0]), reverse=True)
    true_positive = 0
    false_positive = 0
    precision_recall_points: List[Tuple[float, float]] = []
    for _, label in sorted_pairs:
        if int(label) == 1:
            true_positive += 1
        else:
            false_positive += 1
        precision = float(true_positive / max(1, true_positive + false_positive))
        recall = float(true_positive / total_positive)
        precision_recall_points.append((recall, precision))

    average_precision = 0.0
    previous_recall = 0.0
    for recall, precision in precision_recall_points:
        average_precision += precision * max(0.0, recall - previous_recall)
        previous_recall = recall
    return float(average_precision)


def _evaluate_scores(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> Dict[str, Any]:
    predicted_labels = [1 if float(probability) >= float(threshold) else 0 for probability in probabilities]
    true_positive = sum(1 for label, pred in zip(labels, predicted_labels) if int(label) == 1 and int(pred) == 1)
    false_positive = sum(1 for label, pred in zip(labels, predicted_labels) if int(label) == 0 and int(pred) == 1)
    false_negative = sum(1 for label, pred in zip(labels, predicted_labels) if int(label) == 1 and int(pred) == 0)
    true_negative = sum(1 for label, pred in zip(labels, predicted_labels) if int(label) == 0 and int(pred) == 0)
    metrics = _compute_binary_metrics(true_positive, false_positive, false_negative, true_negative)
    metrics["auroc"] = _compute_roc_auc(labels, probabilities)
    metrics["auprc"] = _compute_average_precision(labels, probabilities)
    metrics["predicted_labels"] = predicted_labels
    return metrics


def _example_feature_counts(example: Dict[str, Any]) -> Dict[str, float]:
    counts: Dict[str, float] = {}
    current_segment_label = str(example.get("current_segment_label", "unknown")).strip() or "unknown"
    counts[f"segment_label::{current_segment_label}"] = 1.0

    for atom in list(example.get("body_atoms", [])):
        parsed = _parse_atom(str(atom))
        if parsed is None:
            continue
        predicate, args = parsed
        counts[f"predicate::{predicate}"] = counts.get(f"predicate::{predicate}", 0.0) + 1.0

        if args:
            state_value = str(args[-1]).strip()
            if state_value and state_value not in _VARIABLE_NAMES:
                key = f"state::{predicate}::{state_value}"
                counts[key] = counts.get(key, 0.0) + 1.0

            if predicate == "segment_forward_state" and len(args) >= 2 and args[-1] not in _VARIABLE_NAMES:
                counts[f"segment_forward_state::{args[-1]}"] = 1.0
            elif predicate == "segment_lateral_state" and len(args) >= 2 and args[-1] not in _VARIABLE_NAMES:
                counts[f"segment_lateral_state::{args[-1]}"] = 1.0
            elif predicate == "segment_motion_state" and len(args) >= 2 and args[-1] not in _VARIABLE_NAMES:
                counts[f"segment_motion_state::{args[-1]}"] = 1.0

    return counts


def _build_feature_vocab(train_examples: Sequence[Dict[str, Any]], min_feature_count: int) -> List[str]:
    feature_counts: Counter[str] = Counter()
    for example in train_examples:
        for feature_name in _example_feature_counts(example).keys():
            feature_counts[feature_name] += 1
    return sorted(
        feature_name
        for feature_name, count in feature_counts.items()
        if int(count) >= max(1, int(min_feature_count))
    )


def _vectorize_examples(examples: Sequence[Dict[str, Any]], feature_vocab: Sequence[str]) -> np.ndarray:
    feature_index = {feature_name: idx for idx, feature_name in enumerate(feature_vocab)}
    features = np.zeros((len(examples), len(feature_vocab)), dtype=np.float32)
    for row_index, example in enumerate(examples):
        counts = _example_feature_counts(example)
        for feature_name, value in counts.items():
            idx = feature_index.get(feature_name)
            if idx is not None:
                features[row_index, idx] = float(value)
    return features


def _group_examples_by_video(examples: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for example in examples:
        grouped.setdefault(str(example.get("video_id", "")), []).append(example)
    for video_id in grouped:
        grouped[video_id] = sorted(grouped[video_id], key=_example_sort_key)
    return grouped


def _build_temporal_sequences(
    examples: Sequence[Dict[str, Any]],
    feature_matrix: np.ndarray,
    history_window: int,
) -> Tuple[np.ndarray, List[int]]:
    sequence_length = max(1, int(history_window) + 1)
    sequences = np.zeros((len(examples), sequence_length, feature_matrix.shape[1]), dtype=np.float32)
    available_history = [0 for _ in examples]
    grouped = _group_examples_by_video(examples)

    for video_examples in grouped.values():
        for local_index, example in enumerate(video_examples):
            row_index = int(example["_row_index"])
            start_index = max(0, local_index - int(history_window))
            history_examples = video_examples[start_index : local_index + 1]
            available_history[row_index] = len(history_examples) - 1
            dest_start = sequence_length - len(history_examples)
            for offset, history_example in enumerate(history_examples):
                source_row = int(history_example["_row_index"])
                sequences[row_index, dest_start + offset, :] = feature_matrix[source_row]

    return sequences, available_history


class SymbolicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        dims = [input_dim] + [int(dim) for dim in hidden_dims if int(dim) > 0]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
        final_input_dim = dims[-1] if dims else input_dim
        layers.append(nn.Linear(final_input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class TemporalMLPClassifier(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.input_dim = int(input_dim)
        self.classifier = SymbolicMLP(
            input_dim=self.sequence_length * self.input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x.reshape(x.shape[0], self.sequence_length * self.input_dim))


class TemporalGRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        head_hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        gru_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=max(1, int(num_layers)),
            dropout=gru_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.head = SymbolicMLP(
            input_dim=int(hidden_dim),
            hidden_dims=head_hidden_dims,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        final_hidden = self.dropout(hidden[-1])
        return self.head(final_hidden)


def _choose_device(device_name: str) -> torch.device:
    requested = str(device_name or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested for neural baselines but is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_indices(num_examples: int, batch_size: int) -> Iterable[np.ndarray]:
    permutation = np.random.permutation(num_examples)
    for start in range(0, num_examples, batch_size):
        yield permutation[start : start + batch_size]


def _train_model(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    train_cfg: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    train_labels_cpu = train_targets.detach().cpu().numpy().astype(np.float32)
    num_positive = int(train_labels_cpu.sum())
    num_negative = int(len(train_labels_cpu) - num_positive)
    pos_weight_value = float(num_negative / max(1, num_positive)) if num_positive > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    batch_size = max(1, int(train_cfg.get("batch_size", 32)))
    num_epochs = max(1, int(train_cfg.get("num_epochs", 100)))
    early_stopping_patience = max(1, int(train_cfg.get("early_stopping_patience", 12)))

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss = float("inf")
    epochs_without_improvement = 0
    epoch_history: List[Dict[str, Any]] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        batch_losses: List[float] = []
        for batch_index in _batch_indices(len(train_labels_cpu), batch_size):
            batch_tensor = torch.as_tensor(batch_index, dtype=torch.long, device=device)
            batch_inputs = train_inputs.index_select(0, batch_tensor)
            batch_targets = train_targets.index_select(0, batch_tensor)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            train_logits = model(train_inputs)
            train_loss = float(criterion(train_logits, train_targets).detach().cpu().item())
        epoch_history.append(
            {
                "epoch": epoch,
                "avg_batch_loss": float(sum(batch_losses) / max(1, len(batch_losses))),
                "train_loss": train_loss,
            }
        )

        if train_loss + 1e-8 < best_loss:
            best_loss = train_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return epoch_history


def _predict_probabilities(model: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(inputs)).detach().cpu().numpy().astype(np.float64)
    return probabilities


def _compute_per_video_metrics(eval_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows_by_video: Dict[str, List[Dict[str, Any]]] = {}
    for row in eval_rows:
        rows_by_video.setdefault(str(row["video_id"]), []).append(row)

    per_video_metrics: List[Dict[str, Any]] = []
    for video_id in sorted(rows_by_video):
        rows = rows_by_video[video_id]
        tp = sum(1 for row in rows if bool(row["label"]) and bool(row["predicted_label"]))
        fp = sum(1 for row in rows if (not bool(row["label"])) and bool(row["predicted_label"]))
        fn = sum(1 for row in rows if bool(row["label"]) and (not bool(row["predicted_label"])))
        tn = sum(1 for row in rows if (not bool(row["label"])) and (not bool(row["predicted_label"])))
        metrics = _compute_binary_metrics(tp, fp, fn, tn)
        metrics["video_id"] = video_id
        metrics["num_examples"] = len(rows)
        metrics["auroc"] = _compute_roc_auc(
            [1 if bool(row["label"]) else 0 for row in rows],
            [float(row["predicted_probability"]) for row in rows],
        )
        metrics["auprc"] = _compute_average_precision(
            [1 if bool(row["label"]) else 0 for row in rows],
            [float(row["predicted_probability"]) for row in rows],
        )
        per_video_metrics.append(metrics)
    return per_video_metrics


def _temporal_model_name(architecture: str) -> str:
    return "temporal_mlp" if str(architecture).strip().lower() == "temporal_mlp" else "temporal_gru"


def _serialize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _format_metric(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(numeric):
        return "nan"
    return f"{numeric:.3f}"


def _build_eval_rows(
    examples: Sequence[Dict[str, Any]],
    probabilities: Sequence[float],
    predicted_labels: Sequence[int],
    *,
    available_history: Optional[Sequence[int]] = None,
    sequence_length: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, (example, probability, predicted_label) in enumerate(zip(examples, probabilities, predicted_labels)):
        row: Dict[str, Any] = {
            "video_id": str(example.get("video_id", "")),
            "example_id": str(example.get("example_id", "")),
            "current_segment_index": int(example.get("current_segment_index", -1)),
            "label": bool(example.get("label", False)),
            "predicted_probability": float(probability),
            "predicted_label": bool(predicted_label),
            "current_segment_label": str(example.get("current_segment_label", "")),
            "num_body_atoms": int(example.get("num_body_atoms", len(example.get("body_atoms", [])))),
        }
        if available_history is not None:
            row["available_history"] = int(available_history[index])
            row["sequence_length"] = int(sequence_length or 0)
        rows.append(row)
    return rows


def _write_model_outputs(
    model_dir: Path,
    result: Dict[str, Any],
    train_metrics: Dict[str, Any],
    overall_metrics: Dict[str, Any],
    per_video_metrics: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    training_history: Sequence[Dict[str, Any]],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    result_json_path = model_dir / "baseline_result.json"
    metrics_csv_path = model_dir / "metrics.csv"
    per_video_csv_path = model_dir / "per_video_metrics.csv"
    example_csv_path = model_dir / "example_predictions.csv"
    history_csv_path = model_dir / "training_history.csv"

    with result_json_path.open("w", encoding="utf-8") as fh:
        json.dump(_serialize_json(result), fh, indent=2)

    _write_csv(
        metrics_csv_path,
        [
            "split_name",
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
        [
            {
                "split_name": "train",
                "num_examples": int(result.get("num_train_examples", 0)),
                **{key: value for key, value in train_metrics.items() if key != "predicted_labels"},
            },
            {
                "split_name": "eval",
                "num_examples": int(result.get("num_eval_examples", 0)),
                **{key: value for key, value in overall_metrics.items() if key != "predicted_labels"},
            },
        ],
    )
    _write_csv(
        per_video_csv_path,
        [
            "video_id",
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
        per_video_metrics,
    )

    example_fieldnames = [
        "video_id",
        "example_id",
        "current_segment_index",
        "label",
        "predicted_probability",
        "predicted_label",
        "current_segment_label",
        "num_body_atoms",
    ]
    if eval_rows and "available_history" in eval_rows[0]:
        example_fieldnames.extend(["available_history", "sequence_length"])
    _write_csv(example_csv_path, example_fieldnames, eval_rows)

    _write_csv(history_csv_path, ["epoch", "avg_batch_loss", "train_loss"], training_history)


def _run_single_segment_model(
    *,
    train_matrix: np.ndarray,
    eval_matrix: np.ndarray,
    train_examples: Sequence[Dict[str, Any]],
    eval_examples: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
    probability_threshold: float,
    model_dir: Path,
) -> Dict[str, Any]:
    model = SymbolicMLP(
        input_dim=train_matrix.shape[1],
        hidden_dims=cfg.get("hidden_dims", []),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)

    train_inputs = torch.from_numpy(train_matrix).to(device)
    eval_inputs = torch.from_numpy(eval_matrix).to(device)
    train_targets = torch.from_numpy(
        np.asarray([1 if bool(example.get("label", False)) else 0 for example in train_examples], dtype=np.float32)
    ).to(device)
    eval_labels = [1 if bool(example.get("label", False)) else 0 for example in eval_examples]

    training_history = _train_model(model, train_inputs, train_targets, cfg, device)
    train_probabilities = _predict_probabilities(model, train_inputs)
    eval_probabilities = _predict_probabilities(model, eval_inputs)
    train_metrics = _evaluate_scores(train_targets.detach().cpu().numpy().astype(int).tolist(), train_probabilities.tolist(), probability_threshold)
    overall_metrics = _evaluate_scores(eval_labels, eval_probabilities.tolist(), probability_threshold)
    eval_rows = _build_eval_rows(eval_examples, eval_probabilities.tolist(), overall_metrics["predicted_labels"])
    per_video_metrics = _compute_per_video_metrics(eval_rows)

    result = {
        "model_name": "single_segment_mlp",
        "architecture": "mlp",
        "device": str(device),
        "num_train_examples": len(train_examples),
        "num_eval_examples": len(eval_examples),
        "num_input_features": int(train_matrix.shape[1]),
        "config": dict(cfg),
        "train_metrics": {key: value for key, value in train_metrics.items() if key != "predicted_labels"},
        "overall_metrics": {key: value for key, value in overall_metrics.items() if key != "predicted_labels"},
        "confusion_matrix": {
            "true_positive": int(overall_metrics["true_positive"]),
            "false_positive": int(overall_metrics["false_positive"]),
            "false_negative": int(overall_metrics["false_negative"]),
            "true_negative": int(overall_metrics["true_negative"]),
        },
        "per_video_metrics": list(per_video_metrics),
        "training_history": list(training_history),
    }
    _write_model_outputs(model_dir, result, train_metrics, overall_metrics, per_video_metrics, eval_rows, training_history)
    return result


def _run_temporal_model(
    *,
    train_sequences: np.ndarray,
    eval_sequences: np.ndarray,
    train_examples: Sequence[Dict[str, Any]],
    eval_examples: Sequence[Dict[str, Any]],
    train_available_history: Sequence[int],
    eval_available_history: Sequence[int],
    cfg: Dict[str, Any],
    device: torch.device,
    probability_threshold: float,
    model_dir: Path,
) -> Dict[str, Any]:
    architecture = str(cfg.get("architecture", "gru")).strip().lower()
    sequence_length = int(train_sequences.shape[1])
    input_dim = int(train_sequences.shape[2])

    if architecture == "temporal_mlp":
        model = TemporalMLPClassifier(
            sequence_length=sequence_length,
            input_dim=input_dim,
            hidden_dims=cfg.get("hidden_dims", []),
            dropout=float(cfg.get("dropout", 0.1)),
        ).to(device)
    else:
        model = TemporalGRUClassifier(
            input_dim=input_dim,
            hidden_dim=int(cfg.get("gru_hidden_dim", 128)),
            num_layers=int(cfg.get("gru_num_layers", 1)),
            head_hidden_dims=cfg.get("hidden_dims", []),
            dropout=float(cfg.get("dropout", 0.1)),
        ).to(device)

    train_inputs = torch.from_numpy(train_sequences).to(device)
    eval_inputs = torch.from_numpy(eval_sequences).to(device)
    train_targets = torch.from_numpy(
        np.asarray([1 if bool(example.get("label", False)) else 0 for example in train_examples], dtype=np.float32)
    ).to(device)
    eval_labels = [1 if bool(example.get("label", False)) else 0 for example in eval_examples]

    training_history = _train_model(model, train_inputs, train_targets, cfg, device)
    train_probabilities = _predict_probabilities(model, train_inputs)
    eval_probabilities = _predict_probabilities(model, eval_inputs)
    train_metrics = _evaluate_scores(train_targets.detach().cpu().numpy().astype(int).tolist(), train_probabilities.tolist(), probability_threshold)
    overall_metrics = _evaluate_scores(eval_labels, eval_probabilities.tolist(), probability_threshold)
    eval_rows = _build_eval_rows(
        eval_examples,
        eval_probabilities.tolist(),
        overall_metrics["predicted_labels"],
        available_history=eval_available_history,
        sequence_length=sequence_length,
    )
    per_video_metrics = _compute_per_video_metrics(eval_rows)

    result = {
        "model_name": _temporal_model_name(architecture),
        "architecture": architecture,
        "device": str(device),
        "history_window": int(cfg.get("history_window", sequence_length - 1)),
        "sequence_length": sequence_length,
        "num_train_examples": len(train_examples),
        "num_eval_examples": len(eval_examples),
        "num_input_features": input_dim,
        "config": dict(cfg),
        "train_metrics": {key: value for key, value in train_metrics.items() if key != "predicted_labels"},
        "overall_metrics": {key: value for key, value in overall_metrics.items() if key != "predicted_labels"},
        "confusion_matrix": {
            "true_positive": int(overall_metrics["true_positive"]),
            "false_positive": int(overall_metrics["false_positive"]),
            "false_negative": int(overall_metrics["false_negative"]),
            "true_negative": int(overall_metrics["true_negative"]),
        },
        "history_summary": {
            "avg_train_available_history": float(sum(train_available_history) / max(1, len(train_available_history))),
            "avg_eval_available_history": float(sum(eval_available_history) / max(1, len(eval_available_history))),
        },
        "per_video_metrics": list(per_video_metrics),
        "training_history": list(training_history),
    }
    _write_model_outputs(model_dir, result, train_metrics, overall_metrics, per_video_metrics, eval_rows, training_history)
    return result


def process_baseline(
    train_temporal_rule_results: Sequence[Dict[str, Any]],
    eval_temporal_rule_results: Sequence[Dict[str, Any]],
    split_manifest: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    normalized_cfg = _normalize_cfg(cfg or {})
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)

    summary_json_path = out_root / "neural_baselines_summary.json"
    comparison_csv_path = out_root / "model_comparison.csv"
    comparison_json_path = out_root / "model_comparison.json"
    feature_vocab_path = out_root / "feature_vocab.json"

    if not force_recompute and summary_json_path.exists():
        with summary_json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _BASELINE_VERSION and _cfg_key_subset(cached.get("config", {})) == _cfg_key_subset(normalized_cfg):
            print(f"  [cache] loading {summary_json_path.name}")
            return cached

    train_examples = _prepare_examples(train_temporal_rule_results)
    eval_examples = _prepare_examples(eval_temporal_rule_results)
    if not train_examples:
        raise RuntimeError("Neural baselines received an empty training split.")
    if not eval_examples:
        raise RuntimeError("Neural baselines received an empty evaluation split.")

    _set_random_seed(int(normalized_cfg["random_seed"]))

    feature_vocab = _build_feature_vocab(train_examples, min_feature_count=int(normalized_cfg["min_feature_count"]))
    if not feature_vocab:
        raise RuntimeError("Neural baselines feature vocabulary is empty.")

    train_matrix = _vectorize_examples(train_examples, feature_vocab)
    eval_matrix = _vectorize_examples(eval_examples, feature_vocab)

    feature_mean = train_matrix.mean(axis=0, keepdims=True)
    feature_std = train_matrix.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    train_matrix = (train_matrix - feature_mean) / feature_std
    eval_matrix = (eval_matrix - feature_mean) / feature_std

    history_window = int(normalized_cfg["temporal_model"].get("history_window", 4))
    train_sequences, train_available_history = _build_temporal_sequences(train_examples, train_matrix, history_window)
    eval_sequences, eval_available_history = _build_temporal_sequences(eval_examples, eval_matrix, history_window)

    with feature_vocab_path.open("w", encoding="utf-8") as fh:
        json.dump(
            _serialize_json(
                {
                    "feature_vocab": feature_vocab,
                    "feature_mean": feature_mean.reshape(-1).tolist(),
                    "feature_std": feature_std.reshape(-1).tolist(),
                }
            ),
            fh,
            indent=2,
        )

    device = _choose_device(str(normalized_cfg.get("device", "auto")))
    probability_threshold = float(normalized_cfg.get("probability_threshold", 0.5))

    single_result = _run_single_segment_model(
        train_matrix=train_matrix,
        eval_matrix=eval_matrix,
        train_examples=train_examples,
        eval_examples=eval_examples,
        cfg=normalized_cfg["single_segment_mlp"],
        device=device,
        probability_threshold=probability_threshold,
        model_dir=out_root / "single_segment_mlp",
    )

    temporal_result = _run_temporal_model(
        train_sequences=train_sequences,
        eval_sequences=eval_sequences,
        train_examples=train_examples,
        eval_examples=eval_examples,
        train_available_history=train_available_history,
        eval_available_history=eval_available_history,
        cfg=normalized_cfg["temporal_model"],
        device=device,
        probability_threshold=probability_threshold,
        model_dir=out_root / _temporal_model_name(str(normalized_cfg["temporal_model"].get("architecture", "gru"))),
    )

    comparison_rows: List[Dict[str, Any]] = []
    for model_result in [single_result, temporal_result]:
        overall_metrics = dict(model_result.get("overall_metrics", {}))
        comparison_rows.append(
            {
                "model_name": str(model_result.get("model_name", "")),
                "architecture": str(model_result.get("architecture", "")),
                "precision": overall_metrics.get("precision", 0.0),
                "recall": overall_metrics.get("recall", 0.0),
                "f1": overall_metrics.get("f1", 0.0),
                "accuracy": overall_metrics.get("accuracy", 0.0),
                "auroc": overall_metrics.get("auroc", float("nan")),
                "auprc": overall_metrics.get("auprc", float("nan")),
                "true_positive": overall_metrics.get("true_positive", 0),
                "false_positive": overall_metrics.get("false_positive", 0),
                "false_negative": overall_metrics.get("false_negative", 0),
                "true_negative": overall_metrics.get("true_negative", 0),
            }
        )

    _write_csv(
        comparison_csv_path,
        [
            "model_name",
            "architecture",
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
        comparison_rows,
    )
    with comparison_json_path.open("w", encoding="utf-8") as fh:
        json.dump(_serialize_json({"models": comparison_rows}), fh, indent=2)

    result = {
        "version": _BASELINE_VERSION,
        "config": dict(normalized_cfg),
        "split": split_manifest or {},
        "output_root": str(out_root),
        "feature_vocab_path": str(feature_vocab_path),
        "comparison_csv_path": str(comparison_csv_path),
        "comparison_json_path": str(comparison_json_path),
        "num_train_examples": len(train_examples),
        "num_eval_examples": len(eval_examples),
        "num_features": len(feature_vocab),
        "temporal_sequence_length": int(history_window + 1),
        "model_results": {
            "single_segment_mlp": single_result,
            _temporal_model_name(str(normalized_cfg["temporal_model"].get("architecture", "gru"))): temporal_result,
        },
        "comparison": comparison_rows,
    }

    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(_serialize_json(result), fh, indent=2)

    print(
        "  neural_baselines: "
        f"train_examples={len(train_examples)} | "
        f"eval_examples={len(eval_examples)} | "
        f"features={len(feature_vocab)} | "
        f"single_f1={_format_metric(single_result.get('overall_metrics', {}).get('f1'))} | "
        f"{temporal_result.get('model_name', 'temporal_model')}_f1={_format_metric(temporal_result.get('overall_metrics', {}).get('f1'))}"
    )
    print(f"Neural baselines summary JSON written to {summary_json_path}")
    print(f"Neural baselines comparison CSV written to {comparison_csv_path}")
    return result


def run(
    train_temporal_rule_results: Sequence[Dict[str, Any]],
    eval_temporal_rule_results: Sequence[Dict[str, Any]],
    split_manifest: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_baseline(
        train_temporal_rule_results=train_temporal_rule_results,
        eval_temporal_rule_results=eval_temporal_rule_results,
        split_manifest=split_manifest,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
