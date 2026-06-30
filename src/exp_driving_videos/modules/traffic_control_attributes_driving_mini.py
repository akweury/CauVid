"""
Attach traffic-control attributes to important objects for driving_mini videos.

Consumes:
  - Step 10 output: segment-level important objects
  - Step 7 output: frame-level relative object motion for crop evidence

Output layout:
    pipeline_output/10b_driving_mini_traffic_control_attributes/
        traffic_control_attributes_manifest.json
        traffic_light_state_counts.csv
        traffic_control_relevance_counts.csv
        debug_examples/
            traffic_light_states/<state>/*.jpg
            relevance/<bucket>/*.jpg
        <video_id>/
            traffic_control_attributes.json
"""

from __future__ import annotations

import json
import math
import sys
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_TRAFFIC_CONTROL_ATTRIBUTES_VERSION = 4
_TRAFFIC_LIGHT_ALIASES = {"traffic light", "traffic_light", "traffic signal", "traffic signals"}
_STOP_SIGN_ALIASES = {"stop sign", "stop_sign"}
_DEBUG_STATES: Tuple[str, ...] = ("red", "yellow", "green", "unknown")
_DEBUG_RELEVANCE_BUCKETS: Tuple[str, ...] = ("relevant", "irrelevant")
_MAX_DEBUG_EXAMPLES_PER_BUCKET = 6


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "10b_driving_mini_traffic_control_attributes"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "enabled",
        "min_crop_size",
        "traffic_light_state_threshold",
        "traffic_light_min_confidence_margin",
        "front_distance_far_threshold",
        "center_x_threshold_meters",
        "relevance_high_threshold",
        "relevance_medium_threshold",
        "relevance_weights",
    ]
    return {k: cfg.get(k) for k in keys}


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _normalize_control_class(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    if text in _TRAFFIC_LIGHT_ALIASES:
        return "traffic_light"
    if text in _STOP_SIGN_ALIASES:
        return "stop_sign"
    return text.replace(" ", "_")


def _resolve_local_image_path(image_path: str) -> str:
    if not image_path:
        return image_path
    path = Path(image_path)
    if path.exists():
        return str(path)
    parts = path.parts
    try:
        dataset_idx = parts.index("dataset")
    except ValueError:
        return image_path
    candidate = PROJECT_ROOT.joinpath(*parts[dataset_idx:])
    if candidate.exists():
        return str(candidate)
    return image_path


def _sanitize_filename(value: Any) -> str:
    text = str(value or "unknown").strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text) or "unknown"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _build_relative_frame_lookup(relative_motion_video_result: Dict[str, Any]) -> Dict[int, Dict[int, Dict[str, Any]]]:
    lookup: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for frame in relative_motion_video_result.get("frames", []):
        frame_index = int(frame.get("frame_index", -1))
        if frame_index < 0:
            continue
        by_track: Dict[int, Dict[str, Any]] = {}
        image_path = str(frame.get("image_path", ""))
        for obj in frame.get("objects", []):
            enriched = dict(obj)
            enriched["frame_index"] = frame_index
            enriched["image_path"] = image_path
            by_track[int(obj.get("track_id", -1))] = enriched
        lookup[frame_index] = by_track
    return lookup


def _collect_object_samples(
    relative_lookup: Dict[int, Dict[int, Dict[str, Any]]],
    track_id: int,
    start_frame: int,
    end_frame: int,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for frame_index in range(int(start_frame), int(end_frame) + 1):
        frame_objects = relative_lookup.get(frame_index, {})
        sample = frame_objects.get(int(track_id))
        if sample is None:
            continue
        samples.append(
            {
                "frame_index": int(sample.get("frame_index", frame_index)),
                "image_path": str(sample.get("image_path", "")),
                "label": str(sample.get("label", "unknown")),
                "box": list(sample.get("box", [0, 0, 0, 0])),
                "position_3d": list(sample.get("position_3d", [0.0, 0.0, 0.0])),
                "has_rel_motion": bool(sample.get("has_rel_motion", False)),
            }
        )
    return samples


def _box_metrics(sample: Dict[str, Any]) -> Tuple[float, float]:
    box = list(sample.get("box", [0, 0, 0, 0]))
    if len(box) != 4:
        return 0.0, 0.0
    x1, y1, x2, y2 = [float(v) for v in box]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    center_x = (x1 + x2) * 0.5
    return area, center_x


def _compute_relevance(
    obj: Dict[str, Any],
    samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    mean_position = list(obj.get("mean_position_3d", [0.0, 0.0, 0.0]))
    mean_x = float(mean_position[0]) if len(mean_position) > 0 else 0.0
    mean_z = float(mean_position[2]) if len(mean_position) > 2 else 0.0

    visibility_score = _clamp01(float(obj.get("visibility_ratio", 0.0)))
    center_x_threshold = max(1e-6, float(cfg.get("center_x_threshold_meters", 3.0)))
    center_score = 1.0 - _clamp01(abs(mean_x) / center_x_threshold)

    far_threshold = max(1e-6, float(cfg.get("front_distance_far_threshold", 40.0)))
    if mean_z <= 0.0:
        front_score = 0.0
        distance_score = 0.0
    else:
        front_score = 1.0
        distance_score = 1.0 - _clamp01(mean_z / far_threshold)

    areas = []
    center_scores = []
    for sample in samples:
        area, center_x = _box_metrics(sample)
        if area > 0.0:
            areas.append(area)
        box = list(sample.get("box", [0, 0, 0, 0]))
        if len(box) == 4:
            x1, _, x2, _ = [float(v) for v in box]
            width = max(1.0, x2 - x1)
            object_center = (x1 + x2) * 0.5
            image_path = _resolve_local_image_path(str(sample.get("image_path", "")))
            try:
                import cv2

                img = cv2.imread(image_path)
                if img is not None and img.shape[1] > 0:
                    norm_center = object_center / float(img.shape[1])
                    center_scores.append(1.0 - min(1.0, abs(norm_center - 0.5) / 0.5))
            except Exception:
                _ = width

    area_score = 0.0
    if areas:
        area_score = _clamp01(math.sqrt(float(sum(areas) / len(areas))) / 120.0)
    size_or_distance_score = max(area_score, distance_score)
    frame_center_score = float(sum(center_scores) / len(center_scores)) if center_scores else center_score
    front_center_score = front_score * max(center_score, frame_center_score)

    weights = dict(cfg.get("relevance_weights", {}))
    visibility_weight = float(weights.get("visibility", 0.3))
    center_weight = float(weights.get("center", 0.2))
    size_distance_weight = float(weights.get("size_or_distance", 0.25))
    front_center_weight = float(weights.get("front_center", 0.25))
    weight_sum = max(
        1e-6,
        visibility_weight + center_weight + size_distance_weight + front_center_weight,
    )
    relevance_score = (
        visibility_score * visibility_weight
        + max(center_score, frame_center_score) * center_weight
        + size_or_distance_score * size_distance_weight
        + front_center_score * front_center_weight
    ) / weight_sum
    relevance_score = _clamp01(relevance_score)

    high_threshold = float(cfg.get("relevance_high_threshold", 0.67))
    medium_threshold = float(cfg.get("relevance_medium_threshold", 0.4))
    if relevance_score >= high_threshold:
        relevance_label = "high"
    elif relevance_score >= medium_threshold:
        relevance_label = "medium"
    else:
        relevance_label = "low"

    return {
        "relevance_score": relevance_score,
        "relevance_label": relevance_label,
        "components": {
            "visibility_score": visibility_score,
            "center_score": max(center_score, frame_center_score),
            "size_or_distance_score": size_or_distance_score,
            "front_center_score": front_center_score,
        },
        "is_front_center_region": bool(front_center_score >= 0.5),
    }


def _classify_traffic_light_crop(
    image_path: str,
    box: List[float],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except Exception:
        return {"state": "unknown", "confidence": 0.0, "scores": {}}

    min_crop_size = int(cfg.get("min_crop_size", 8))
    img = cv2.imread(_resolve_local_image_path(image_path))
    if img is None or len(box) != 4:
        return {"state": "unknown", "confidence": 0.0, "scores": {}}

    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return {"state": "unknown", "confidence": 0.0, "scores": {}}

    crop = img[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < min_crop_size or crop.shape[1] < min_crop_size:
        return {"state": "unknown", "confidence": 0.0, "scores": {}}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    hue = hsv[:, :, 0]
    color_energy = sat * val

    red_mask = ((hue <= 12) | (hue >= 165)) & (sat >= 0.25) & (val >= 0.2)
    yellow_mask = (hue >= 15) & (hue <= 40) & (sat >= 0.2) & (val >= 0.2)
    green_mask = (hue >= 40) & (hue <= 100) & (sat >= 0.2) & (val >= 0.2)

    rows = crop.shape[0]
    top_slice = slice(0, max(1, rows // 3))
    mid_slice = slice(max(0, rows // 3), max(1, (2 * rows) // 3))
    bottom_slice = slice(max(0, (2 * rows) // 3), rows)

    def _region_score(mask: Any, row_slice: slice, region_weight: float) -> float:
        region_energy = color_energy[row_slice, :]
        region_mask = mask[row_slice, :]
        if region_energy.size == 0:
            return 0.0
        return float((region_energy * region_mask.astype(np.float32)).mean() * region_weight)

    rgb_float = rgb.astype(np.float32) / 255.0
    r = rgb_float[:, :, 0]
    g = rgb_float[:, :, 1]
    b = rgb_float[:, :, 2]

    red_rgb = np.maximum(r - g, 0.0) + np.maximum(r - b, 0.0)
    green_rgb = np.maximum(g - r, 0.0) + np.maximum(g - b, 0.0)
    yellow_rgb = np.maximum(np.minimum(r, g) - b, 0.0)

    score_red = (
        _region_score(red_mask, top_slice, 1.2)
        + _region_score(red_mask, slice(0, rows), 0.4)
        + float(red_rgb[top_slice, :].mean() * 0.25)
    )
    score_yellow = (
        _region_score(yellow_mask, mid_slice, 1.2)
        + _region_score(yellow_mask, slice(0, rows), 0.4)
        + float(yellow_rgb[mid_slice, :].mean() * 0.25)
    )
    score_green = (
        _region_score(green_mask, bottom_slice, 1.2)
        + _region_score(green_mask, slice(0, rows), 0.4)
        + float(green_rgb[bottom_slice, :].mean() * 0.25)
    )

    scores = {"red": score_red, "yellow": score_yellow, "green": score_green}
    best_state, best_score = max(scores.items(), key=lambda item: item[1])
    sorted_scores = sorted(scores.values(), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    total_score = max(1e-6, sum(scores.values()))
    confidence = best_score / total_score

    min_threshold = float(cfg.get("traffic_light_state_threshold", 0.06))
    min_margin = float(cfg.get("traffic_light_min_confidence_margin", 0.025))
    if best_score < min_threshold or (best_score - second_score) < min_margin:
        return {"state": "unknown", "confidence": float(confidence * 0.5), "scores": scores}
    return {"state": best_state, "confidence": float(confidence), "scores": scores}


def _load_crop(image_path: str, box: List[float]) -> Optional[Any]:
    try:
        import cv2
    except Exception:
        return None

    img = cv2.imread(_resolve_local_image_path(image_path))
    if img is None or len(box) != 4:
        return None

    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    return crop if crop is not None and crop.size > 0 else None


def _aggregate_traffic_light_state(
    samples: List[Dict[str, Any]],
    relevance: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    state_scores = {"red": 0.0, "yellow": 0.0, "green": 0.0, "unknown": 0.0}
    sample_results: List[Dict[str, Any]] = []
    sample_weight_base = max(0.25, float(relevance.get("relevance_score", 0.0)))

    for sample in samples:
        result = _classify_traffic_light_crop(
            image_path=str(sample.get("image_path", "")),
            box=list(sample.get("box", [0, 0, 0, 0])),
            cfg=cfg,
        )
        area, _ = _box_metrics(sample)
        area_weight = _clamp01(math.sqrt(max(0.0, area)) / 120.0)
        weight = max(0.15, sample_weight_base * max(0.25, area_weight))
        state_scores[result["state"]] += float(result.get("confidence", 0.0)) * weight
        sample_results.append(
            {
                "frame_index": int(sample.get("frame_index", -1)),
                "state": str(result.get("state", "unknown")),
                "confidence": float(result.get("confidence", 0.0)),
            }
        )

    best_state, best_score = max(state_scores.items(), key=lambda item: item[1])
    total = max(1e-6, sum(state_scores.values()))
    confidence = best_score / total
    if best_state == "unknown" or best_score <= 0.0:
        confidence = min(confidence, 0.5)
    return {
        "traffic_light_state": best_state,
        "traffic_light_state_confidence": float(confidence),
        "traffic_light_state_scores": {k: float(v) for k, v in state_scores.items()},
        "traffic_light_state_samples": sample_results,
    }


def _enrich_object(
    obj: Dict[str, Any],
    segment: Dict[str, Any],
    relative_lookup: Dict[int, Dict[int, Dict[str, Any]]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    enriched = dict(obj)
    object_class = _normalize_control_class(obj.get("object_class", "unknown"))
    if object_class not in {"traffic_light", "stop_sign"}:
        return enriched

    track_id = int(obj.get("track_id", -1))
    samples = _collect_object_samples(
        relative_lookup=relative_lookup,
        track_id=track_id,
        start_frame=int(segment.get("start_frame", 0)),
        end_frame=int(segment.get("end_frame", 0)),
    )
    relevance = _compute_relevance(obj=enriched, samples=samples, cfg=cfg)
    traffic_control_attributes: Dict[str, Any] = {
        "traffic_control_type": object_class,
        "traffic_control_relevance_score": float(relevance["relevance_score"]),
        "traffic_control_relevance_label": str(relevance["relevance_label"]),
        "traffic_control_relevance_components": dict(relevance["components"]),
        "is_front_center_region": bool(relevance["is_front_center_region"]),
        "num_evidence_samples": len(samples),
    }
    if object_class == "traffic_light":
        traffic_control_attributes.update(
            _aggregate_traffic_light_state(samples=samples, relevance=relevance, cfg=cfg)
        )

    enriched["object_class"] = object_class
    enriched["traffic_control_attributes"] = traffic_control_attributes
    enriched["traffic_control_sample_evidence"] = samples
    return enriched


def _print_video_summary(result: Dict[str, Any]) -> None:
    print(
        f"  {result.get('video_id', 'unknown')}: "
        f"traffic_control_objects={int(result.get('num_traffic_control_objects', 0))} | "
        f"traffic_lights={int(result.get('num_traffic_lights', 0))} | "
        f"stop_signs={int(result.get('num_stop_signs', 0))}"
    )


def _iter_selected_traffic_control_objects(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for video_result in results:
        video_id = str(video_result.get("video_id", "unknown"))
        for segment in list(video_result.get("segments", [])):
            segment_index = int(segment.get("segment_index", -1))
            segment_id = str(segment.get("segment_id", ""))
            for obj in list(segment.get("selected_objects", [])):
                attrs = obj.get("traffic_control_attributes")
                if not isinstance(attrs, dict):
                    continue
                rows.append(
                    {
                        "video_id": video_id,
                        "segment_index": segment_index,
                        "segment_id": segment_id,
                        "track_id": int(obj.get("track_id", -1)),
                        "object_class": str(obj.get("object_class", "unknown")),
                        "object": obj,
                        "attributes": attrs,
                    }
                )
    return rows


def _build_state_count_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, float]] = {
        state: {
            "object_count": 0.0,
            "mean_confidence_total": 0.0,
            "high_relevance_count": 0.0,
            "front_center_count": 0.0,
        }
        for state in _DEBUG_STATES
    }
    for row in rows:
        attrs = row["attributes"]
        if str(attrs.get("traffic_control_type", "")) != "traffic_light":
            continue
        state = str(attrs.get("traffic_light_state", "unknown"))
        if state not in buckets:
            state = "unknown"
        bucket = buckets[state]
        bucket["object_count"] += 1.0
        bucket["mean_confidence_total"] += float(attrs.get("traffic_light_state_confidence", 0.0))
        if str(attrs.get("traffic_control_relevance_label", "low")) in {"high", "medium"}:
            bucket["high_relevance_count"] += 1.0
        if bool(attrs.get("is_front_center_region", False)):
            bucket["front_center_count"] += 1.0

    output_rows: List[Dict[str, Any]] = []
    for state in _DEBUG_STATES:
        bucket = buckets[state]
        count = int(bucket["object_count"])
        output_rows.append(
            {
                "traffic_light_state": state,
                "object_count": count,
                "mean_confidence": (
                    float(bucket["mean_confidence_total"] / max(1, count)) if count else 0.0
                ),
                "high_or_medium_relevance_count": int(bucket["high_relevance_count"]),
                "front_center_count": int(bucket["front_center_count"]),
            }
        )
    return output_rows


def _build_relevance_count_rows(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    relevance_threshold = float(cfg.get("relevance_medium_threshold", 0.4))
    counts: Dict[Tuple[str, str, str], int] = {}
    for row in rows:
        attrs = row["attributes"]
        object_class = str(attrs.get("traffic_control_type", row.get("object_class", "unknown")))
        relevance_label = str(attrs.get("traffic_control_relevance_label", "low"))
        is_relevant = float(attrs.get("traffic_control_relevance_score", 0.0)) >= relevance_threshold
        key = (object_class, relevance_label, "relevant" if is_relevant else "irrelevant")
        counts[key] = counts.get(key, 0) + 1
    output_rows = []
    for (object_class, relevance_label, relevance_bucket), count in sorted(counts.items()):
        output_rows.append(
            {
                "object_class": object_class,
                "relevance_label": relevance_label,
                "relevance_bucket": relevance_bucket,
                "object_count": int(count),
            }
        )
    return output_rows


def _remove_old_debug_examples(debug_root: Path) -> None:
    if not debug_root.exists():
        return
    for path in sorted(debug_root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def _choose_debug_sample(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    samples = list(obj.get("traffic_control_sample_evidence", []))
    if not samples:
        return None
    ranked = sorted(
        samples,
        key=lambda sample: (_box_metrics(sample)[0], int(sample.get("frame_index", -1))),
        reverse=True,
    )
    return ranked[0] if ranked else None


def _write_debug_examples(results: List[Dict[str, Any]], out_root: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    debug_root = out_root / "debug_examples"
    _remove_old_debug_examples(debug_root)
    debug_root.mkdir(parents=True, exist_ok=True)

    state_root = debug_root / "traffic_light_states"
    relevance_root = debug_root / "relevance"
    for state in _DEBUG_STATES:
        (state_root / state).mkdir(parents=True, exist_ok=True)
    for bucket in _DEBUG_RELEVANCE_BUCKETS:
        (relevance_root / bucket).mkdir(parents=True, exist_ok=True)

    state_counts = {state: 0 for state in _DEBUG_STATES}
    relevance_counts = {bucket: 0 for bucket in _DEBUG_RELEVANCE_BUCKETS}
    relevance_threshold = float(cfg.get("relevance_medium_threshold", 0.4))

    try:
        import cv2
    except Exception:
        return {
            "enabled": False,
            "reason": "cv2_not_available",
            "output_root": str(debug_root),
            "saved_examples": {"traffic_light_states": state_counts, "relevance": relevance_counts},
        }

    for row in _iter_selected_traffic_control_objects(results):
        obj = row["object"]
        attrs = row["attributes"]
        sample = _choose_debug_sample(obj)
        if sample is None:
            continue
        crop = _load_crop(str(sample.get("image_path", "")), list(sample.get("box", [0, 0, 0, 0])))
        if crop is None:
            continue

        state = str(attrs.get("traffic_light_state", "unknown"))
        if state not in state_counts:
            state = "unknown"
        relevance_bucket = (
            "relevant"
            if float(attrs.get("traffic_control_relevance_score", 0.0)) >= relevance_threshold
            else "irrelevant"
        )

        stem = (
            f"{_sanitize_filename(row['video_id'])}_seg{int(row['segment_index']):03d}"
            f"_track{int(row['track_id']):04d}_{_sanitize_filename(row['object_class'])}"
            f"_f{int(sample.get('frame_index', -1)):04d}"
        )

        if str(attrs.get("traffic_control_type", "")) == "traffic_light" and state_counts[state] < _MAX_DEBUG_EXAMPLES_PER_BUCKET:
            out_path = state_root / state / f"{stem}.jpg"
            cv2.imwrite(str(out_path), crop)
            state_counts[state] += 1

        if relevance_counts[relevance_bucket] < _MAX_DEBUG_EXAMPLES_PER_BUCKET:
            out_path = relevance_root / relevance_bucket / f"{stem}.jpg"
            cv2.imwrite(str(out_path), crop)
            relevance_counts[relevance_bucket] += 1

        if all(count >= _MAX_DEBUG_EXAMPLES_PER_BUCKET for count in state_counts.values()) and all(
            count >= _MAX_DEBUG_EXAMPLES_PER_BUCKET for count in relevance_counts.values()
        ):
            break

    return {
        "enabled": True,
        "output_root": str(debug_root),
        "saved_examples": {"traffic_light_states": state_counts, "relevance": relevance_counts},
    }


def process_video(
    important_objects_video_result: Dict[str, Any],
    relative_motion_video_result: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    enabled = bool(cfg.get("enabled", True))

    video_id = str(important_objects_video_result["video_id"])
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "traffic_control_attributes.json"

    if not enabled:
        result = dict(important_objects_video_result)
        result["traffic_control_attributes_enabled"] = False
        return result

    if not force_recompute and out_file.exists():
        with out_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        cache_cfg = cached.get("config", {}) if isinstance(cached, dict) else {}
        if (
            int(cached.get("version", 0)) == _TRAFFIC_CONTROL_ATTRIBUTES_VERSION
            and _cfg_key_subset(cache_cfg) == _cfg_key_subset(cfg)
        ):
            print(f"  [cache] {video_id} - loading {out_file.name}")
            _print_video_summary(cached)
            return cached

    relative_lookup = _build_relative_frame_lookup(relative_motion_video_result)
    segments_out: List[Dict[str, Any]] = []
    num_traffic_control_objects = 0
    num_traffic_lights = 0
    num_stop_signs = 0

    for segment in important_objects_video_result.get("segments", []):
        segment_out = dict(segment)
        objects = [
            _enrich_object(obj, segment, relative_lookup, cfg)
            for obj in list(segment.get("objects", []))
        ]
        selected_objects = [
            _enrich_object(obj, segment, relative_lookup, cfg)
            for obj in list(segment.get("selected_objects", segment.get("objects", [])))
        ]
        filtered_objects = [
            _enrich_object(obj, segment, relative_lookup, cfg)
            for obj in list(segment.get("filtered_objects", []))
        ]
        candidate_objects = [dict(obj) for obj in list(segment.get("candidate_objects", []))]
        selected_candidate_objects = [
            dict(obj)
            for obj in list(segment.get("selected_candidate_objects", segment.get("candidate_objects", [])))
        ]
        filtered_candidate_objects = [
            dict(obj)
            for obj in list(segment.get("filtered_candidate_objects", []))
        ]
        segment_out["objects"] = objects
        segment_out["selected_objects"] = selected_objects
        segment_out["filtered_objects"] = filtered_objects
        segment_out["candidate_objects"] = candidate_objects
        segment_out["selected_candidate_objects"] = selected_candidate_objects
        segment_out["filtered_candidate_objects"] = filtered_candidate_objects

        traffic_control_objects = []
        for obj in selected_objects:
            attrs = obj.get("traffic_control_attributes")
            if not isinstance(attrs, dict):
                continue
            traffic_control_objects.append(
                {
                    "track_id": int(obj.get("track_id", -1)),
                    "object_class": str(obj.get("object_class", "unknown")),
                    "traffic_control_relevance_score": float(attrs.get("traffic_control_relevance_score", 0.0)),
                    "traffic_control_relevance_label": str(attrs.get("traffic_control_relevance_label", "low")),
                    "traffic_light_state": str(attrs.get("traffic_light_state", "")),
                    "traffic_light_state_confidence": float(attrs.get("traffic_light_state_confidence", 0.0)),
                }
            )
            num_traffic_control_objects += 1
            if str(attrs.get("traffic_control_type", "")) == "traffic_light":
                num_traffic_lights += 1
            elif str(attrs.get("traffic_control_type", "")) == "stop_sign":
                num_stop_signs += 1
        segment_out["traffic_control_objects"] = traffic_control_objects
        segments_out.append(segment_out)

    result: Dict[str, Any] = {
        "version": _TRAFFIC_CONTROL_ATTRIBUTES_VERSION,
        "video_id": video_id,
        "traffic_control_attributes_enabled": True,
        "num_segments": len(segments_out),
        "num_traffic_control_objects": num_traffic_control_objects,
        "num_traffic_lights": num_traffic_lights,
        "num_stop_signs": num_stop_signs,
        "config": dict(cfg),
        "segments": segments_out,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    _print_video_summary(result)
    return result


def run(
    important_object_results: List[Dict[str, Any]],
    relative_motion_results: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    out_root = output_root or get_output_root()
    relative_by_video = {str(v.get("video_id", "")): v for v in relative_motion_results}
    results: List[Dict[str, Any]] = []

    for important_result in important_object_results:
        video_id = str(important_result.get("video_id", "unknown"))
        relative_motion_result = relative_by_video.get(video_id)
        if relative_motion_result is None:
            print(f"  [warn] Missing relative motion for video {video_id}; passing important objects through.")
            results.append(dict(important_result))
            continue
        results.append(
            process_video(
                important_objects_video_result=important_result,
                relative_motion_video_result=relative_motion_result,
                cfg=cfg,
                output_root=out_root,
                force_recompute=force_recompute,
            )
        )

    manifest = {
        "version": _TRAFFIC_CONTROL_ATTRIBUTES_VERSION,
        "num_videos": len(results),
        "videos": [
            {
                "video_id": r.get("video_id", ""),
                "num_segments": r.get("num_segments", 0),
                "num_traffic_control_objects": r.get("num_traffic_control_objects", 0),
                "num_traffic_lights": r.get("num_traffic_lights", 0),
                "num_stop_signs": r.get("num_stop_signs", 0),
            }
            for r in results
        ],
    }
    manifest_path = out_root / "traffic_control_attributes_manifest.json"
    state_counts_csv_path = out_root / "traffic_light_state_counts.csv"
    relevance_counts_csv_path = out_root / "traffic_control_relevance_counts.csv"

    selected_rows = _iter_selected_traffic_control_objects(results)
    state_count_rows = _build_state_count_rows(selected_rows)
    relevance_count_rows = _build_relevance_count_rows(selected_rows, cfg or {})
    _write_csv(
        state_counts_csv_path,
        [
            "traffic_light_state",
            "object_count",
            "mean_confidence",
            "high_or_medium_relevance_count",
            "front_center_count",
        ],
        state_count_rows,
    )
    _write_csv(
        relevance_counts_csv_path,
        ["object_class", "relevance_label", "relevance_bucket", "object_count"],
        relevance_count_rows,
    )
    debug_summary = _write_debug_examples(results, out_root, cfg or {})
    manifest["aggregate_outputs"] = {
        "traffic_light_state_counts_csv": str(state_counts_csv_path),
        "traffic_control_relevance_counts_csv": str(relevance_counts_csv_path),
        "debug_examples": debug_summary,
        "num_selected_traffic_control_objects": len(selected_rows),
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Traffic control attributes manifest written to {manifest_path}")
    return results
