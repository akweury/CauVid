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
        neural_baseline_summary.json
        neural_baseline_metrics.csv
        per_video_metrics.csv
        prediction_examples.csv
        training_curves.csv
        feature_vocab.json
        single_segment_mlp/
            neural_baseline_summary.json
            neural_baseline_metrics.csv
            per_video_metrics.csv
            prediction_examples.csv
            training_curves.csv
            best_model_checkpoint.pt
            final_model_checkpoint.pt
        temporal_gru/ or temporal_mlp/
            ...
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


_BASELINE_VERSION = 3
_VARIABLE_NAMES = {"S", "O", "T", "F"}
_DEFAULT_THRESHOLD = 0.5


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "neural_baselines_driving_mini"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _default_cfg() -> Dict[str, Any]:
    return {
        "min_feature_count": 1,
        "probability_threshold": 0.5,
        "validation_fraction": 0.25,
        "imbalance_strategy": "pos_weight",
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
        "validation_fraction": float(cfg.get("validation_fraction", defaults["validation_fraction"])),
        "imbalance_strategy": str(cfg.get("imbalance_strategy", defaults["imbalance_strategy"])).strip().lower(),
        "random_seed": int(cfg.get("random_seed", defaults["random_seed"])),
        "device": str(cfg.get("device", defaults["device"])),
    }

    single_cfg = dict(defaults["single_segment_mlp"])
    if isinstance(cfg.get("single_segment_mlp"), dict):
        single_cfg.update(cfg["single_segment_mlp"])
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
        "validation_fraction": normalized["validation_fraction"],
        "imbalance_strategy": normalized["imbalance_strategy"],
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


def _choose_device(device_name: str) -> torch.device:
    requested = str(device_name or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_negative": int(true_negative),
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


def _evaluate_scores(labels: Sequence[int], probabilities: Sequence[float], threshold: float) -> Dict[str, Any]:
    predicted_labels = [1 if float(probability) >= float(threshold) else 0 for probability in probabilities]
    true_positive = sum(1 for label, pred in zip(labels, predicted_labels) if int(label) == 1 and int(pred) == 1)
    false_positive = sum(1 for label, pred in zip(labels, predicted_labels) if int(label) == 0 and int(pred) == 1)
    false_negative = sum(1 for label, pred in zip(labels, predicted_labels) if int(label) == 1 and int(pred) == 0)
    true_negative = sum(1 for label, pred in zip(labels, predicted_labels) if int(label) == 0 and int(pred) == 0)
    metrics = _compute_binary_metrics(true_positive, false_positive, false_negative, true_negative)
    metrics["auroc"] = _compute_roc_auc(labels, probabilities)
    metrics["auprc"] = _compute_average_precision(labels, probabilities)
    metrics["predicted_labels"] = predicted_labels
    metrics["threshold"] = float(threshold)
    return metrics


def _select_best_f1_threshold(
    labels: Sequence[int],
    probabilities: Sequence[float],
    default_threshold: float = _DEFAULT_THRESHOLD,
) -> Tuple[float, Dict[str, Any]]:
    if not labels:
        metrics = _evaluate_scores([], [], default_threshold)
        return float(default_threshold), metrics

    candidates = {0.0, 1.0, float(default_threshold)}
    for probability in probabilities:
        p = float(probability)
        candidates.add(p)
        candidates.add(min(1.0, max(0.0, p + 1e-8)))
        candidates.add(min(1.0, max(0.0, p - 1e-8)))

    best_threshold = float(default_threshold)
    best_metrics = _evaluate_scores(labels, probabilities, best_threshold)
    best_key = (
        float(best_metrics.get("f1", 0.0)),
        float(best_metrics.get("precision", 0.0)),
        -abs(best_threshold - default_threshold),
        best_threshold,
    )
    for threshold in sorted(candidates):
        metrics = _evaluate_scores(labels, probabilities, float(threshold))
        key = (
            float(metrics.get("f1", 0.0)),
            float(metrics.get("precision", 0.0)),
            -abs(float(threshold) - default_threshold),
            float(threshold),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


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
                state_key = f"state::{predicate}::{state_value}"
                counts[state_key] = counts.get(state_key, 0.0) + 1.0

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


def _split_train_validation(
    train_examples: Sequence[Dict[str, Any]],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    grouped = _group_examples_by_video(train_examples)
    video_ids = sorted(grouped)
    rng = random.Random(int(seed))
    shuffled_video_ids = list(video_ids)
    rng.shuffle(shuffled_video_ids)

    if len(shuffled_video_ids) > 1:
        requested = int(round(len(shuffled_video_ids) * float(validation_fraction)))
        val_video_count = min(len(shuffled_video_ids) - 1, max(1, requested))
        val_video_ids = set(shuffled_video_ids[:val_video_count])
        train_subset = [example for example in train_examples if str(example.get("video_id", "")) not in val_video_ids]
        val_subset = [example for example in train_examples if str(example.get("video_id", "")) in val_video_ids]
        return train_subset, val_subset, {
            "strategy": "video_level",
            "train_video_ids": sorted({str(example.get("video_id", "")) for example in train_subset}),
            "validation_video_ids": sorted(val_video_ids),
            "num_train_examples": len(train_subset),
            "num_validation_examples": len(val_subset),
        }

    ordered_examples = sorted(train_examples, key=_example_sort_key)
    if len(ordered_examples) <= 1:
        return list(ordered_examples), list(ordered_examples), {
            "strategy": "single_example_fallback",
            "train_video_ids": list(video_ids),
            "validation_video_ids": list(video_ids),
            "num_train_examples": len(ordered_examples),
            "num_validation_examples": len(ordered_examples),
        }

    val_count = min(len(ordered_examples) - 1, max(1, int(round(len(ordered_examples) * float(validation_fraction)))))
    val_indices = set(range(len(ordered_examples) - val_count, len(ordered_examples)))
    train_subset = [example for idx, example in enumerate(ordered_examples) if idx not in val_indices]
    val_subset = [example for idx, example in enumerate(ordered_examples) if idx in val_indices]
    return train_subset, val_subset, {
        "strategy": "example_level_fallback",
        "train_video_ids": list(video_ids),
        "validation_video_ids": list(video_ids),
        "num_train_examples": len(train_subset),
        "num_validation_examples": len(val_subset),
    }


def _build_temporal_sequences(
    examples: Sequence[Dict[str, Any]],
    feature_matrix: np.ndarray,
    history_window: int,
) -> Tuple[np.ndarray, List[int]]:
    sequence_length = max(1, int(history_window) + 1)
    sequences = np.zeros((len(examples), sequence_length, feature_matrix.shape[1]), dtype=np.float32)
    available_history = [0 for _ in examples]
    grouped = _group_examples_by_video(examples)
    row_lookup = {str(example.get("example_id", "")): idx for idx, example in enumerate(examples)}

    for video_examples in grouped.values():
        for local_index, example in enumerate(video_examples):
            example_id = str(example.get("example_id", ""))
            row_index = row_lookup[example_id]
            start_index = max(0, local_index - int(history_window))
            history_examples = video_examples[start_index : local_index + 1]
            available_history[row_index] = len(history_examples) - 1
            dest_start = sequence_length - len(history_examples)
            for offset, history_example in enumerate(history_examples):
                source_row = row_lookup[str(history_example.get("example_id", ""))]
                sequences[row_index, dest_start + offset, :] = feature_matrix[source_row]

    return sequences, available_history


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


class SymbolicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(dim) for dim in hidden_dims if int(dim) > 0]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(dims[-1], 1))
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
        flattened = x.reshape(x.shape[0], self.sequence_length * self.input_dim)
        return self.classifier(flattened)


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


def _build_sampling_probabilities(labels: Sequence[int]) -> np.ndarray:
    label_array = np.asarray(labels, dtype=np.int64)
    class_counts = {
        0: max(1, int(np.sum(label_array == 0))),
        1: max(1, int(np.sum(label_array == 1))),
    }
    weights = np.asarray([1.0 / class_counts[int(label)] for label in label_array], dtype=np.float64)
    weights /= max(np.sum(weights), 1e-12)
    return weights


def _batch_indices(num_examples: int, batch_size: int, sample_probabilities: Optional[np.ndarray]) -> Iterable[np.ndarray]:
    if sample_probabilities is None:
        permutation = np.random.permutation(num_examples)
    else:
        permutation = np.random.choice(
            num_examples,
            size=num_examples,
            replace=True,
            p=sample_probabilities,
        )
    for start in range(0, num_examples, batch_size):
        yield permutation[start : start + batch_size]


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    epoch: int,
    model_name: str,
    config_dict: Dict[str, Any],
    train_summary: Dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_name": str(model_name),
            "config": _serialize_json(config_dict),
            "train_summary": _serialize_json(train_summary),
            "state_dict": model.state_dict(),
        },
        path,
    )


def _train_model(
    *,
    model: nn.Module,
    model_name: str,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    train_cfg: Dict[str, Any],
    global_cfg: Dict[str, Any],
    device: torch.device,
    model_dir: Path,
) -> Dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = model_dir / "best_model_checkpoint.pt"
    final_checkpoint_path = model_dir / "final_model_checkpoint.pt"

    train_labels = train_targets.detach().cpu().numpy().astype(np.int64)
    val_labels = val_targets.detach().cpu().numpy().astype(np.int64)
    imbalance_strategy = str(global_cfg.get("imbalance_strategy", "pos_weight")).strip().lower()

    pos_weight_value = 1.0
    sample_probabilities: Optional[np.ndarray] = None
    if imbalance_strategy == "balanced_sampling":
        criterion = nn.BCEWithLogitsLoss()
        sample_probabilities = _build_sampling_probabilities(train_labels.tolist())
    else:
        num_positive = int(np.sum(train_labels == 1))
        num_negative = int(np.sum(train_labels == 0))
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
    best_epoch = 0
    best_monitor_loss = float("inf")
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        batch_losses: List[float] = []
        for batch_index in _batch_indices(len(train_labels), batch_size, sample_probabilities):
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
            val_logits = model(val_inputs)
            train_loss = float(criterion(train_logits, train_targets).detach().cpu().item())
            val_loss = float(criterion(val_logits, val_targets).detach().cpu().item())
            train_probabilities = torch.sigmoid(train_logits).detach().cpu().numpy().astype(np.float64)
            val_probabilities = torch.sigmoid(val_logits).detach().cpu().numpy().astype(np.float64)

        train_metrics_05 = _evaluate_scores(train_labels.tolist(), train_probabilities.tolist(), _DEFAULT_THRESHOLD)
        val_metrics_05 = _evaluate_scores(val_labels.tolist(), val_probabilities.tolist(), _DEFAULT_THRESHOLD)
        tuned_threshold, val_metrics_best = _select_best_f1_threshold(val_labels.tolist(), val_probabilities.tolist(), _DEFAULT_THRESHOLD)

        history.append(
            {
                "model_name": model_name,
                "epoch": int(epoch),
                "avg_batch_loss": float(sum(batch_losses) / max(1, len(batch_losses))),
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "train_precision_at_0_5": float(train_metrics_05.get("precision", 0.0)),
                "train_recall_at_0_5": float(train_metrics_05.get("recall", 0.0)),
                "train_f1_at_0_5": float(train_metrics_05.get("f1", 0.0)),
                "validation_precision_at_0_5": float(val_metrics_05.get("precision", 0.0)),
                "validation_recall_at_0_5": float(val_metrics_05.get("recall", 0.0)),
                "validation_f1_at_0_5": float(val_metrics_05.get("f1", 0.0)),
                "best_validation_threshold": float(tuned_threshold),
                "validation_precision_at_best_threshold": float(val_metrics_best.get("precision", 0.0)),
                "validation_recall_at_best_threshold": float(val_metrics_best.get("recall", 0.0)),
                "validation_f1_at_best_threshold": float(val_metrics_best.get("f1", 0.0)),
            }
        )

        if val_loss + 1e-8 < best_monitor_loss:
            best_monitor_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            _save_checkpoint(
                best_checkpoint_path,
                model=model,
                epoch=epoch,
                model_name=model_name,
                config_dict=train_cfg,
                train_summary={
                    "imbalance_strategy": imbalance_strategy,
                    "pos_weight": pos_weight_value,
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    _save_checkpoint(
        final_checkpoint_path,
        model=model,
        epoch=len(history),
        model_name=model_name,
        config_dict=train_cfg,
        train_summary={
            "imbalance_strategy": imbalance_strategy,
            "pos_weight": pos_weight_value,
        },
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_monitor_loss),
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "imbalance_strategy": imbalance_strategy,
        "pos_weight": float(pos_weight_value),
        "used_balanced_sampling": imbalance_strategy == "balanced_sampling",
    }


def _predict_probabilities(model: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(inputs)).detach().cpu().numpy().astype(np.float64)
    return probabilities


def _compute_thresholded_metrics(
    split_name: str,
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold_name: str,
    threshold_value: float,
) -> Dict[str, Any]:
    metrics = _evaluate_scores(labels, probabilities, threshold_value)
    return {
        "split_name": split_name,
        "threshold_name": threshold_name,
        "threshold_value": float(threshold_value),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "f1": float(metrics.get("f1", 0.0)),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "auroc": float(metrics.get("auroc", float("nan"))),
        "auprc": float(metrics.get("auprc", float("nan"))),
        "true_positive": int(metrics.get("true_positive", 0)),
        "false_positive": int(metrics.get("false_positive", 0)),
        "false_negative": int(metrics.get("false_negative", 0)),
        "true_negative": int(metrics.get("true_negative", 0)),
    }


def _compute_per_video_metrics(
    *,
    model_name: str,
    examples: Sequence[Dict[str, Any]],
    probabilities: Sequence[float],
    threshold_name: str,
    threshold_value: float,
) -> List[Dict[str, Any]]:
    grouped = _group_examples_by_video(examples)
    probability_lookup = {
        str(example.get("example_id", "")): float(probability)
        for example, probability in zip(examples, probabilities)
    }

    rows: List[Dict[str, Any]] = []
    for video_id in sorted(grouped):
        video_examples = grouped[video_id]
        labels = [1 if bool(example.get("label", False)) else 0 for example in video_examples]
        video_probabilities = [probability_lookup[str(example.get("example_id", ""))] for example in video_examples]
        metrics = _compute_thresholded_metrics(
            "eval",
            labels,
            video_probabilities,
            threshold_name,
            threshold_value,
        )
        rows.append(
            {
                "model_name": model_name,
                "video_id": video_id,
                "num_examples": len(video_examples),
                **metrics,
            }
        )
    return rows


def _build_prediction_rows(
    *,
    model_name: str,
    examples: Sequence[Dict[str, Any]],
    probabilities: Sequence[float],
    threshold_05: float,
    best_threshold: float,
    available_history: Optional[Sequence[int]] = None,
    sequence_length: Optional[int] = None,
) -> List[Dict[str, Any]]:
    predicted_05 = [1 if float(probability) >= float(threshold_05) else 0 for probability in probabilities]
    predicted_best = [1 if float(probability) >= float(best_threshold) else 0 for probability in probabilities]
    rows: List[Dict[str, Any]] = []
    for index, example in enumerate(examples):
        row: Dict[str, Any] = {
            "model_name": model_name,
            "video_id": str(example.get("video_id", "")),
            "example_id": str(example.get("example_id", "")),
            "current_segment_index": int(example.get("current_segment_index", -1)),
            "label": bool(example.get("label", False)),
            "predicted_probability": float(probabilities[index]),
            "threshold_at_0_5": float(threshold_05),
            "best_validation_threshold": float(best_threshold),
            "predicted_label_at_0_5": bool(predicted_05[index]),
            "predicted_label_at_best_validation_threshold": bool(predicted_best[index]),
            "current_segment_label": str(example.get("current_segment_label", "")),
            "num_body_atoms": int(example.get("num_body_atoms", len(example.get("body_atoms", [])))),
        }
        if available_history is not None:
            row["available_history"] = int(available_history[index])
            row["sequence_length"] = int(sequence_length or 0)
        rows.append(row)
    return rows


def _write_model_outputs(
    *,
    model_dir: Path,
    summary: Dict[str, Any],
    metrics_rows: Sequence[Dict[str, Any]],
    per_video_rows: Sequence[Dict[str, Any]],
    prediction_rows: Sequence[Dict[str, Any]],
    training_curves: Sequence[Dict[str, Any]],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = model_dir / "neural_baseline_summary.json"
    metrics_csv_path = model_dir / "neural_baseline_metrics.csv"
    per_video_csv_path = model_dir / "per_video_metrics.csv"
    prediction_csv_path = model_dir / "prediction_examples.csv"
    curves_csv_path = model_dir / "training_curves.csv"

    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(_serialize_json(summary), fh, indent=2)

    _write_csv(
        metrics_csv_path,
        [
            "model_name",
            "split_name",
            "threshold_name",
            "threshold_value",
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
        metrics_rows,
    )
    _write_csv(
        per_video_csv_path,
        [
            "model_name",
            "video_id",
            "num_examples",
            "split_name",
            "threshold_name",
            "threshold_value",
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
        per_video_rows,
    )

    prediction_fieldnames = [
        "model_name",
        "video_id",
        "example_id",
        "current_segment_index",
        "label",
        "predicted_probability",
        "threshold_at_0_5",
        "best_validation_threshold",
        "predicted_label_at_0_5",
        "predicted_label_at_best_validation_threshold",
        "current_segment_label",
        "num_body_atoms",
    ]
    if prediction_rows and "available_history" in prediction_rows[0]:
        prediction_fieldnames.extend(["available_history", "sequence_length"])
    _write_csv(prediction_csv_path, prediction_fieldnames, prediction_rows)
    _write_csv(
        curves_csv_path,
        [
            "model_name",
            "epoch",
            "avg_batch_loss",
            "train_loss",
            "validation_loss",
            "train_precision_at_0_5",
            "train_recall_at_0_5",
            "train_f1_at_0_5",
            "validation_precision_at_0_5",
            "validation_recall_at_0_5",
            "validation_f1_at_0_5",
            "best_validation_threshold",
            "validation_precision_at_best_threshold",
            "validation_recall_at_best_threshold",
            "validation_f1_at_best_threshold",
        ],
        training_curves,
    )


def _run_model(
    *,
    model: nn.Module,
    model_name: str,
    architecture: str,
    train_array: np.ndarray,
    val_array: np.ndarray,
    eval_array: np.ndarray,
    train_examples: Sequence[Dict[str, Any]],
    val_examples: Sequence[Dict[str, Any]],
    eval_examples: Sequence[Dict[str, Any]],
    train_cfg: Dict[str, Any],
    global_cfg: Dict[str, Any],
    device: torch.device,
    model_dir: Path,
    eval_available_history: Optional[Sequence[int]] = None,
    sequence_length: Optional[int] = None,
    extra_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    train_inputs = torch.from_numpy(train_array).to(device)
    val_inputs = torch.from_numpy(val_array).to(device)
    eval_inputs = torch.from_numpy(eval_array).to(device)
    train_labels = [1 if bool(example.get("label", False)) else 0 for example in train_examples]
    val_labels = [1 if bool(example.get("label", False)) else 0 for example in val_examples]
    eval_labels = [1 if bool(example.get("label", False)) else 0 for example in eval_examples]
    train_targets = torch.from_numpy(np.asarray(train_labels, dtype=np.float32)).to(device)
    val_targets = torch.from_numpy(np.asarray(val_labels, dtype=np.float32)).to(device)

    training_info = _train_model(
        model=model,
        model_name=model_name,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        train_cfg=train_cfg,
        global_cfg=global_cfg,
        device=device,
        model_dir=model_dir,
    )

    train_probabilities = _predict_probabilities(model, train_inputs).tolist()
    val_probabilities = _predict_probabilities(model, val_inputs).tolist()
    eval_probabilities = _predict_probabilities(model, eval_inputs).tolist()

    threshold_05 = float(global_cfg.get("probability_threshold", _DEFAULT_THRESHOLD))
    best_threshold, best_val_metrics = _select_best_f1_threshold(val_labels, val_probabilities, threshold_05)
    thresholds = {
        "threshold_0_5": threshold_05,
        "best_validation_threshold": float(best_threshold),
    }

    metrics_rows: List[Dict[str, Any]] = []
    metrics_by_split: Dict[str, Dict[str, Dict[str, Any]]] = {}
    split_payloads = [
        ("train", train_examples, train_labels, train_probabilities),
        ("validation", val_examples, val_labels, val_probabilities),
        ("eval", eval_examples, eval_labels, eval_probabilities),
    ]
    for split_name, split_examples, split_labels, split_probabilities in split_payloads:
        metrics_by_split[split_name] = {}
        for threshold_name, threshold_value in thresholds.items():
            row = _compute_thresholded_metrics(
                split_name,
                split_labels,
                split_probabilities,
                threshold_name,
                threshold_value,
            )
            row["model_name"] = model_name
            row["num_examples"] = len(split_examples)
            metrics_rows.append(row)
            metrics_by_split[split_name][threshold_name] = dict(row)

    per_video_rows: List[Dict[str, Any]] = []
    for threshold_name, threshold_value in thresholds.items():
        per_video_rows.extend(
            _compute_per_video_metrics(
                model_name=model_name,
                examples=eval_examples,
                probabilities=eval_probabilities,
                threshold_name=threshold_name,
                threshold_value=threshold_value,
            )
        )

    prediction_rows = _build_prediction_rows(
        model_name=model_name,
        examples=eval_examples,
        probabilities=eval_probabilities,
        threshold_05=threshold_05,
        best_threshold=best_threshold,
        available_history=eval_available_history,
        sequence_length=sequence_length,
    )

    summary: Dict[str, Any] = {
        "version": _BASELINE_VERSION,
        "model_name": model_name,
        "architecture": architecture,
        "device": str(device),
        "config": dict(train_cfg),
        "global_baseline_config": {
            "probability_threshold": threshold_05,
            "validation_fraction": float(global_cfg.get("validation_fraction", 0.25)),
            "imbalance_strategy": str(global_cfg.get("imbalance_strategy", "pos_weight")),
        },
        "thresholds": thresholds,
        "validation_threshold_tuning": {
            "selected_threshold": float(best_threshold),
            "validation_metrics_at_selected_threshold": {
                key: value for key, value in best_val_metrics.items() if key != "predicted_labels"
            },
        },
        "num_train_examples": len(train_examples),
        "num_validation_examples": len(val_examples),
        "num_eval_examples": len(eval_examples),
        "metrics_by_split": metrics_by_split,
        "training": training_info,
        "checkpoint_paths": {
            "best_model_checkpoint": training_info["best_checkpoint_path"],
            "final_model_checkpoint": training_info["final_checkpoint_path"],
        },
    }
    if extra_summary:
        summary.update(extra_summary)

    _write_model_outputs(
        model_dir=model_dir,
        summary=summary,
        metrics_rows=metrics_rows,
        per_video_rows=per_video_rows,
        prediction_rows=prediction_rows,
        training_curves=training_info["history"],
    )

    return {
        **summary,
        "metrics_rows": metrics_rows,
        "per_video_rows": per_video_rows,
        "prediction_rows": prediction_rows,
        "training_curves": training_info["history"],
        "selected_eval_metrics": metrics_by_split["eval"]["best_validation_threshold"],
    }


def _run_single_segment_model(
    *,
    train_matrix: np.ndarray,
    val_matrix: np.ndarray,
    eval_matrix: np.ndarray,
    train_examples: Sequence[Dict[str, Any]],
    val_examples: Sequence[Dict[str, Any]],
    eval_examples: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    global_cfg: Dict[str, Any],
    device: torch.device,
    model_dir: Path,
) -> Dict[str, Any]:
    model = SymbolicMLP(
        input_dim=train_matrix.shape[1],
        hidden_dims=cfg.get("hidden_dims", []),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    return _run_model(
        model=model,
        model_name="single_segment_mlp",
        architecture="mlp",
        train_array=train_matrix,
        val_array=val_matrix,
        eval_array=eval_matrix,
        train_examples=train_examples,
        val_examples=val_examples,
        eval_examples=eval_examples,
        train_cfg=cfg,
        global_cfg=global_cfg,
        device=device,
        model_dir=model_dir,
    )


def _temporal_model_name(architecture: str) -> str:
    return "temporal_mlp" if str(architecture).strip().lower() == "temporal_mlp" else "temporal_gru"


def _run_temporal_model(
    *,
    train_sequences: np.ndarray,
    val_sequences: np.ndarray,
    eval_sequences: np.ndarray,
    train_examples: Sequence[Dict[str, Any]],
    val_examples: Sequence[Dict[str, Any]],
    eval_examples: Sequence[Dict[str, Any]],
    eval_available_history: Sequence[int],
    cfg: Dict[str, Any],
    global_cfg: Dict[str, Any],
    device: torch.device,
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

    return _run_model(
        model=model,
        model_name=_temporal_model_name(architecture),
        architecture=architecture,
        train_array=train_sequences,
        val_array=val_sequences,
        eval_array=eval_sequences,
        train_examples=train_examples,
        val_examples=val_examples,
        eval_examples=eval_examples,
        train_cfg=cfg,
        global_cfg=global_cfg,
        device=device,
        model_dir=model_dir,
        eval_available_history=eval_available_history,
        sequence_length=sequence_length,
        extra_summary={
            "history_window": int(cfg.get("history_window", sequence_length - 1)),
            "sequence_length": sequence_length,
            "num_input_features": input_dim,
        },
    )


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

    summary_json_path = out_root / "neural_baseline_summary.json"
    metrics_csv_path = out_root / "neural_baseline_metrics.csv"
    per_video_csv_path = out_root / "per_video_metrics.csv"
    prediction_csv_path = out_root / "prediction_examples.csv"
    training_curves_csv_path = out_root / "training_curves.csv"
    feature_vocab_path = out_root / "feature_vocab.json"

    if not force_recompute and summary_json_path.exists():
        with summary_json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _BASELINE_VERSION and _cfg_key_subset(cached.get("config", {})) == _cfg_key_subset(normalized_cfg):
            print(f"  [cache] loading {summary_json_path.name}")
            return cached

    _set_random_seed(int(normalized_cfg["random_seed"]))

    full_train_examples = _prepare_examples(train_temporal_rule_results)
    eval_examples = _prepare_examples(eval_temporal_rule_results)
    if not full_train_examples:
        raise RuntimeError("Neural baselines received an empty training split.")
    if not eval_examples:
        raise RuntimeError("Neural baselines received an empty evaluation split.")

    train_examples, val_examples, validation_split = _split_train_validation(
        full_train_examples,
        validation_fraction=float(normalized_cfg.get("validation_fraction", 0.25)),
        seed=int(normalized_cfg.get("random_seed", 0)),
    )
    if not train_examples or not val_examples:
        raise RuntimeError("Neural baselines could not create non-empty train/validation subsets.")

    feature_vocab = _build_feature_vocab(train_examples, min_feature_count=int(normalized_cfg["min_feature_count"]))
    if not feature_vocab:
        raise RuntimeError("Neural baselines feature vocabulary is empty.")

    train_raw = _vectorize_examples(train_examples, feature_vocab)
    val_raw = _vectorize_examples(val_examples, feature_vocab)
    eval_raw = _vectorize_examples(eval_examples, feature_vocab)
    feature_mean = train_raw.mean(axis=0, keepdims=True)
    feature_std = train_raw.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    train_matrix = (train_raw - feature_mean) / feature_std
    val_matrix = (val_raw - feature_mean) / feature_std
    eval_matrix = (eval_raw - feature_mean) / feature_std

    history_window = int(normalized_cfg["temporal_model"].get("history_window", 4))
    train_sequences, _ = _build_temporal_sequences(train_examples, train_matrix, history_window)
    val_sequences, _ = _build_temporal_sequences(val_examples, val_matrix, history_window)
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
    single_result = _run_single_segment_model(
        train_matrix=train_matrix,
        val_matrix=val_matrix,
        eval_matrix=eval_matrix,
        train_examples=train_examples,
        val_examples=val_examples,
        eval_examples=eval_examples,
        cfg=normalized_cfg["single_segment_mlp"],
        global_cfg=normalized_cfg,
        device=device,
        model_dir=out_root / "single_segment_mlp",
    )
    temporal_model_dir = out_root / _temporal_model_name(str(normalized_cfg["temporal_model"].get("architecture", "gru")))
    temporal_result = _run_temporal_model(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        eval_sequences=eval_sequences,
        train_examples=train_examples,
        val_examples=val_examples,
        eval_examples=eval_examples,
        eval_available_history=eval_available_history,
        cfg=normalized_cfg["temporal_model"],
        global_cfg=normalized_cfg,
        device=device,
        model_dir=temporal_model_dir,
    )

    all_metrics_rows = list(single_result["metrics_rows"]) + list(temporal_result["metrics_rows"])
    all_per_video_rows = list(single_result["per_video_rows"]) + list(temporal_result["per_video_rows"])
    all_prediction_rows = list(single_result["prediction_rows"]) + list(temporal_result["prediction_rows"])
    all_training_curves = list(single_result["training_curves"]) + list(temporal_result["training_curves"])

    _write_csv(
        metrics_csv_path,
        [
            "model_name",
            "split_name",
            "threshold_name",
            "threshold_value",
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
        all_metrics_rows,
    )
    _write_csv(
        per_video_csv_path,
        [
            "model_name",
            "video_id",
            "num_examples",
            "split_name",
            "threshold_name",
            "threshold_value",
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
        all_per_video_rows,
    )
    prediction_fieldnames = [
        "model_name",
        "video_id",
        "example_id",
        "current_segment_index",
        "label",
        "predicted_probability",
        "threshold_at_0_5",
        "best_validation_threshold",
        "predicted_label_at_0_5",
        "predicted_label_at_best_validation_threshold",
        "current_segment_label",
        "num_body_atoms",
        "available_history",
        "sequence_length",
    ]
    _write_csv(prediction_csv_path, prediction_fieldnames, all_prediction_rows)
    _write_csv(
        training_curves_csv_path,
        [
            "model_name",
            "epoch",
            "avg_batch_loss",
            "train_loss",
            "validation_loss",
            "train_precision_at_0_5",
            "train_recall_at_0_5",
            "train_f1_at_0_5",
            "validation_precision_at_0_5",
            "validation_recall_at_0_5",
            "validation_f1_at_0_5",
            "best_validation_threshold",
            "validation_precision_at_best_threshold",
            "validation_recall_at_best_threshold",
            "validation_f1_at_best_threshold",
        ],
        all_training_curves,
    )

    comparison_rows = [
        row
        for row in all_metrics_rows
        if str(row.get("split_name", "")) == "eval"
    ]

    result = {
        "version": _BASELINE_VERSION,
        "config": dict(normalized_cfg),
        "split": split_manifest or {},
        "validation_split": validation_split,
        "output_root": str(out_root),
        "feature_vocab_path": str(feature_vocab_path),
        "metrics_csv_path": str(metrics_csv_path),
        "per_video_csv_path": str(per_video_csv_path),
        "prediction_csv_path": str(prediction_csv_path),
        "training_curves_csv_path": str(training_curves_csv_path),
        "num_full_train_examples": len(full_train_examples),
        "num_train_examples": len(train_examples),
        "num_validation_examples": len(val_examples),
        "num_eval_examples": len(eval_examples),
        "num_features": len(feature_vocab),
        "temporal_sequence_length": int(history_window + 1),
        "model_results": {
            "single_segment_mlp": {
                key: value
                for key, value in single_result.items()
                if key not in {"metrics_rows", "per_video_rows", "prediction_rows", "training_curves"}
            },
            str(temporal_result.get("model_name", "temporal_model")): {
                key: value
                for key, value in temporal_result.items()
                if key not in {"metrics_rows", "per_video_rows", "prediction_rows", "training_curves"}
            },
        },
        "comparison": comparison_rows,
    }

    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(_serialize_json(result), fh, indent=2)

    print(
        "  neural_baselines: "
        f"train_examples={len(train_examples)} | "
        f"val_examples={len(val_examples)} | "
        f"eval_examples={len(eval_examples)} | "
        f"features={len(feature_vocab)} | "
        f"single_f1_best={_format_metric(single_result['selected_eval_metrics'].get('f1'))} | "
        f"{temporal_result.get('model_name', 'temporal_model')}_f1_best={_format_metric(temporal_result['selected_eval_metrics'].get('f1'))}"
    )
    print(f"Neural baselines summary JSON written to {summary_json_path}")
    print(f"Neural baselines metrics CSV written to {metrics_csv_path}")
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
