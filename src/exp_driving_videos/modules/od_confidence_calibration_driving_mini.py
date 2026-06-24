from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.exp_driving_videos.modules import od_calibration_policy_utils


_CALIBRATION_STAGE_VERSION = 1
_POLICY_VERSION = 1


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "min_labeled_detections": int(cfg.get("min_labeled_detections", 8)),
        "min_positive_detections": int(cfg.get("min_positive_detections", 2)),
        "min_negative_detections": int(cfg.get("min_negative_detections", 2)),
        "logistic_c": float(cfg.get("logistic_c", 1.0)),
        "max_iter": int(cfg.get("max_iter", 500)),
        "class_weight": str(cfg.get("class_weight", "balanced")),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple, set)):
                    serialized[key] = json.dumps(value, sort_keys=isinstance(value, dict))
                else:
                    serialized[key] = value
            writer.writerow(serialized)


def _heuristic_bonus_maps(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    class_scores: Dict[str, List[float]] = defaultdict(list)
    source_scores: Dict[str, List[float]] = defaultdict(list)
    prior_scores: Dict[str, List[float]] = defaultdict(list)
    overall_scores: List[float] = []
    for row in rows:
        if row.get("label_name") not in {"positive", "negative"}:
            continue
        utility = 1.0 if row.get("label_name") == "positive" else -1.0
        utility *= max(0.5, min(3.0, abs(_safe_float(row.get("sample_weight", 1.0), 1.0))))
        class_scores[_safe_text(row.get("class_name", "unknown")) or "unknown"].append(utility)
        source_scores[_safe_text(row.get("candidate_source", "candidate")) or "candidate"].append(utility)
        prior_id = _safe_text(row.get("top_prior_id", "none")) or "none"
        prior_scores[prior_id].append(utility)
        overall_scores.append(utility)

    def _mean_bonus(values: Sequence[float], scale: float) -> float:
        if not values:
            return 0.0
        return float(max(-0.2, min(0.2, (sum(values) / max(1, len(values))) * scale)))

    return {
        "global_bias": _mean_bonus(overall_scores, 0.03),
        "class_bonus": {
            key: _mean_bonus(values, 0.05)
            for key, values in sorted(class_scores.items())
        },
        "candidate_source_bonus": {
            key: _mean_bonus(values, 0.05)
            for key, values in sorted(source_scores.items())
        },
        "top_prior_bonus": {
            key: _mean_bonus(values, 0.04)
            for key, values in sorted(prior_scores.items())
        },
    }


def _build_training_rows(
    pseudo_label_results: Dict[str, Any],
    detection_results: Sequence[Dict[str, Any]],
    tracking_results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    detection_lookup = od_calibration_policy_utils.build_detection_lookup(detection_results)
    track_lookup = od_calibration_policy_utils.build_candidate_track_feature_lookup(tracking_results)
    rows: List[Dict[str, Any]] = []
    for label_row in list(pseudo_label_results.get("detection_pseudo_labels", [])):
        detection_id = _safe_text(label_row.get("detection_id", ""))
        detection = dict(detection_lookup.get(detection_id, {}))
        if not detection_id or not detection or bool(detection.get("accepted", False)):
            continue
        label_name = _safe_text(label_row.get("pseudo_label", "neutral")) or "neutral"
        reasoning_feedback_score = _safe_float(label_row.get("pseudo_label_score", 0.0), 0.0)
        track_features = dict(track_lookup.get(detection_id, {}))
        features = od_calibration_policy_utils.extract_detection_calibration_features(
            detection,
            track_features=track_features,
            reasoning_feedback_score=reasoning_feedback_score,
        )
        sample_weight = 1.0 + min(3.0, abs(reasoning_feedback_score))
        if label_name == "positive":
            sample_weight += 0.5 * len(list(label_row.get("recovered_fn_example_ids", [])))
        elif label_name == "negative":
            sample_weight += 0.5 * len(list(label_row.get("introduced_fp_example_ids", [])))
        rows.append(
            {
                "detection_id": detection_id,
                "label_name": label_name,
                "label_binary": 1 if label_name == "positive" else 0,
                "sample_weight": sample_weight,
                "class_name": _safe_text(features.get("class_name", "unknown")) or "unknown",
                "candidate_source": _safe_text(features.get("candidate_source", "candidate")) or "candidate",
                "top_prior_id": _safe_text(features.get("top_prior_id", "none")) or "none",
                "matched_prior_ids": od_calibration_policy_utils.matched_prior_ids_from_detection(detection),
                **features,
            }
        )
    return rows


def _fit_logistic_policy(
    rows: Sequence[Dict[str, Any]],
    *,
    iteration_id: str,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("scikit-learn is required for logistic OD calibration policy fitting.") from exc

    numeric_names = od_calibration_policy_utils.numeric_feature_names()
    categorical_names = ("class_name", "candidate_source", "top_prior_id")
    labeled_rows = [row for row in rows if row.get("label_name") in {"positive", "negative"}]
    means: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    for name in numeric_names:
        values = np.asarray([_safe_float(row.get(name, 0.0), 0.0) for row in labeled_rows], dtype=np.float64)
        means[name] = float(values.mean()) if values.size else 0.0
        scale = float(values.std()) if values.size else 1.0
        scales[name] = scale if abs(scale) > 1e-9 else 1.0

    feature_names: List[str] = [f"num:{name}" for name in numeric_names]
    categorical_feature_names: List[str] = []
    for categorical_name in categorical_names:
        values = sorted({_safe_text(row.get(categorical_name, "none")) or "none" for row in labeled_rows})
        categorical_feature_names.extend(
            [f"cat:{categorical_name}={value}" for value in values]
        )
    feature_names.extend(categorical_feature_names)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    x = np.zeros((len(labeled_rows), len(feature_names)), dtype=np.float64)
    y = np.asarray([_safe_int(row.get("label_binary", 0), 0) for row in labeled_rows], dtype=np.int64)
    weights = np.asarray([_safe_float(row.get("sample_weight", 1.0), 1.0) for row in labeled_rows], dtype=np.float64)
    for row_index, row in enumerate(labeled_rows):
        for name in numeric_names:
            standardized = (
                (_safe_float(row.get(name, 0.0), 0.0) - means[name]) / max(scales[name], 1e-9)
            )
            x[row_index, feature_index[f"num:{name}"]] = standardized
        for categorical_name in categorical_names:
            value = _safe_text(row.get(categorical_name, "none")) or "none"
            feature_name = f"cat:{categorical_name}={value}"
            if feature_name in feature_index:
                x[row_index, feature_index[feature_name]] = 1.0

    class_weight_cfg = _safe_text(cfg.get("class_weight", "balanced")).lower()
    class_weight = "balanced" if class_weight_cfg == "balanced" else None
    model = LogisticRegression(
        penalty="l2",
        C=float(cfg.get("logistic_c", 1.0)),
        solver="liblinear",
        max_iter=int(cfg.get("max_iter", 500)),
        class_weight=class_weight,
        random_state=0,
    )
    model.fit(x, y, sample_weight=weights)
    logits = np.asarray(model.decision_function(x), dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    feature_coefficients = {
        feature_name: float(model.coef_[0][feature_index[feature_name]])
        for feature_name in feature_names
    }
    model_summary = {
        "num_training_rows": len(labeled_rows),
        "num_positive_rows": int(np.sum(y == 1)),
        "num_negative_rows": int(np.sum(y == 0)),
        "training_probability_mean": float(probabilities.mean()) if probabilities.size else 0.0,
    }
    policy = {
        "version": _POLICY_VERSION,
        "policy_id": f"od_calibration_{iteration_id}",
        "source_iteration_id": str(iteration_id),
        "policy_type": "logistic_detection_calibrator",
        "normalization": {
            "means": means,
            "scales": scales,
        },
        "model": {
            "intercept": float(model.intercept_[0]),
            "feature_coefficients": feature_coefficients,
        },
        "heuristics": _heuristic_bonus_maps(labeled_rows),
        "feature_schema": {
            "numeric_features": numeric_names,
            "categorical_features": list(categorical_names),
            "feature_names": feature_names,
        },
        "training_summary": model_summary,
    }
    return policy, model_summary


def _heuristic_policy(
    rows: Sequence[Dict[str, Any]],
    *,
    iteration_id: str,
    reason: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    labeled_rows = [row for row in rows if row.get("label_name") in {"positive", "negative"}]
    summary = {
        "num_training_rows": len(labeled_rows),
        "num_positive_rows": sum(1 for row in labeled_rows if row.get("label_name") == "positive"),
        "num_negative_rows": sum(1 for row in labeled_rows if row.get("label_name") == "negative"),
        "fallback_reason": reason,
    }
    policy = {
        "version": _POLICY_VERSION,
        "policy_id": f"od_calibration_{iteration_id}",
        "source_iteration_id": str(iteration_id),
        "policy_type": "heuristic_additive",
        "normalization": {"means": {}, "scales": {}},
        "model": {"intercept": 0.0, "feature_coefficients": {}},
        "heuristics": _heuristic_bonus_maps(labeled_rows),
        "feature_schema": {
            "numeric_features": od_calibration_policy_utils.numeric_feature_names(),
            "categorical_features": ["class_name", "candidate_source", "top_prior_id"],
            "feature_names": [],
        },
        "training_summary": summary,
    }
    return policy, summary


def process(
    pseudo_label_results: Dict[str, Any],
    detection_results: Sequence[Dict[str, Any]],
    tracking_results: Sequence[Dict[str, Any]],
    *,
    iteration_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = od_calibration_policy_utils.get_iteration_root(iteration_id)
    if output_root is not None:
        out_root = Path(output_root) / iteration_id
        out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "od_confidence_calibration.json"
    policy_path = out_root / "proposed_od_calibration_policy.json"
    training_csv_path = out_root / "od_calibration_training_rows.csv"

    if not force_recompute and json_path.exists():
        cached = od_calibration_policy_utils.load_json(json_path, default={})
        if int(cached.get("version", 0)) == _CALIBRATION_STAGE_VERSION and _cfg_key_subset(cached.get("config", {})) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {json_path.name}")
            return cached

    training_rows = _build_training_rows(pseudo_label_results, detection_results, tracking_results)
    labeled_rows = [row for row in training_rows if row.get("label_name") in {"positive", "negative"}]
    label_counter = Counter(row.get("label_name", "neutral") for row in training_rows)

    min_labeled = max(1, int(cfg.get("min_labeled_detections", 8)))
    min_positive = max(1, int(cfg.get("min_positive_detections", 2)))
    min_negative = max(1, int(cfg.get("min_negative_detections", 2)))

    if len(labeled_rows) < min_labeled:
        policy, training_summary = _heuristic_policy(
            training_rows,
            iteration_id=iteration_id,
            reason="insufficient_labeled_detections",
        )
    elif int(label_counter.get("positive", 0)) < min_positive:
        policy, training_summary = _heuristic_policy(
            training_rows,
            iteration_id=iteration_id,
            reason="insufficient_positive_detections",
        )
    elif int(label_counter.get("negative", 0)) < min_negative:
        policy, training_summary = _heuristic_policy(
            training_rows,
            iteration_id=iteration_id,
            reason="insufficient_negative_detections",
        )
    else:
        policy, training_summary = _fit_logistic_policy(
            training_rows,
            iteration_id=iteration_id,
            cfg=cfg,
        )

    result = {
        "version": _CALIBRATION_STAGE_VERSION,
        "iteration_id": str(iteration_id),
        "config": _cfg_key_subset(cfg),
        "num_training_rows": len(training_rows),
        "num_labeled_rows": len(labeled_rows),
        "num_positive_rows": int(label_counter.get("positive", 0)),
        "num_negative_rows": int(label_counter.get("negative", 0)),
        "num_neutral_rows": int(label_counter.get("neutral", 0)),
        "policy": policy,
        "training_summary": training_summary,
        "output_paths": {
            "json": str(json_path),
            "proposed_policy_json": str(policy_path),
            "training_rows_csv": str(training_csv_path),
        },
    }

    od_calibration_policy_utils.save_json_atomic(json_path, result)
    od_calibration_policy_utils.save_json_atomic(policy_path, policy)
    _write_csv(
        training_csv_path,
        [
            "detection_id",
            "label_name",
            "label_binary",
            "sample_weight",
            "class_name",
            "candidate_source",
            "top_prior_id",
            "matched_prior_ids",
            *od_calibration_policy_utils.numeric_feature_names(),
        ],
        training_rows,
    )

    print(
        "  od_confidence_calibration: "
        f"rows={len(training_rows)} | "
        f"labeled={len(labeled_rows)} | "
        f"policy_type={_safe_text(policy.get('policy_type', ''))}"
    )
    print(f"OD confidence calibration JSON written to {json_path}")
    print(f"Proposed OD calibration policy JSON written to {policy_path}")
    return result


def run(
    pseudo_label_results: Dict[str, Any],
    detection_results: Sequence[Dict[str, Any]],
    tracking_results: Sequence[Dict[str, Any]],
    *,
    iteration_id: str,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process(
        pseudo_label_results=pseudo_label_results,
        detection_results=detection_results,
        tracking_results=tracking_results,
        iteration_id=iteration_id,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
