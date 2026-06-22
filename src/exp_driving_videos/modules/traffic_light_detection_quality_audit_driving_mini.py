"""
Export a balanced full-frame visual audit set for detected traffic lights.

Outputs:
    pipeline_output/18g_driving_mini_traffic_light_detection_quality_audit/
        traffic_light_detection_quality_audit_summary.json
        traffic_light_detection_quality_audit_manifest.json
        traffic_light_detection_quality_audit_samples.csv
        traffic_light_detection_quality_audit_index.csv
        red/*.jpg
        yellow/*.jpg
        green/*.jpg
        unknown/*.jpg
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_TRAFFIC_LIGHT_DETECTION_QUALITY_AUDIT_VERSION = 3
_STATE_ORDER: Tuple[str, ...] = ("red", "yellow", "green", "unknown")
_FUTURE_HORIZONS: Tuple[int, ...] = (1, 2, 3, 5)
_POSITIVE_FORWARD_STATES: Tuple[str, ...] = ("forward_slowdown", "stopping")
_RELEVANCE_THRESHOLD = 0.4


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "18g_driving_mini_traffic_light_detection_quality_audit"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "max_samples_total": int(cfg.get("max_samples_total", 0)),
        "max_samples_per_state": int(cfg.get("max_samples_per_state", 50)),
        "max_samples_per_video": int(cfg.get("max_samples_per_video", 12)),
        "include_unknown_state": bool(cfg.get("include_unknown_state", True)),
        "min_state_confidence": float(cfg.get("min_state_confidence", 0.0)),
        "random_seed": int(cfg.get("random_seed", 0)),
        "separate_confidence_bands": bool(cfg.get("separate_confidence_bands", False)),
        "confidence_split_threshold": float(cfg.get("confidence_split_threshold", 0.75)),
        "max_samples_per_state_per_confidence_band": int(
            cfg.get("max_samples_per_state_per_confidence_band", 0)
        ),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


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


def _safe_divide(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator / denominator)


def _sanitize_filename(value: Any) -> str:
    text = str(value or "unknown").strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text) or "unknown"


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


def _box_area(box: Sequence[float]) -> float:
    if len(box) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _frame_id(frame_index: int) -> str:
    return f"frame_{int(frame_index):04d}"


def _object_id(track_id: int) -> str:
    return f"track_{int(track_id):04d}"


def _state_sort_key(state_name: str) -> Tuple[int, str]:
    normalized = _normalize_text(state_name)
    try:
        return (_STATE_ORDER.index(normalized), normalized)
    except ValueError:
        return (len(_STATE_ORDER), normalized)


def _choose_best_sample(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    samples = list(obj.get("traffic_control_sample_evidence", []))
    if not samples:
        return None
    ranked = sorted(
        samples,
        key=lambda sample: (
            _box_area(list(sample.get("box", [0, 0, 0, 0]))),
            _safe_int(sample.get("frame_index", -1), -1),
        ),
        reverse=True,
    )
    return dict(ranked[0]) if ranked else None


def _segment_positions_by_video(logic_atom_results: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    positions: Dict[str, Dict[str, Any]] = {}
    for video_result in logic_atom_results:
        video_id = str(video_result.get("video_id", ""))
        segments = sorted(
            list(video_result.get("segments", [])),
            key=lambda segment: (
                _safe_int(segment.get("segment_index", -1), -1),
                str(segment.get("segment_id", "")),
            ),
        )
        positions[video_id] = {
            "segments": segments,
            "position_by_segment_id": {
                str(segment.get("segment_id", "")): index for index, segment in enumerate(segments)
            },
        }
    return positions


def _future_brake_flags(
    segments: Sequence[Dict[str, Any]],
    current_position: int,
) -> Dict[int, bool]:
    future_labels = [
        _normalize_text(segment.get("segment_forward_label", "unknown"))
        for segment in list(segments[current_position + 1 :])
    ]
    positive_forward_states = set(_POSITIVE_FORWARD_STATES)
    return {
        horizon: any(label in positive_forward_states for label in future_labels[:horizon])
        for horizon in _FUTURE_HORIZONS
    }


def _build_brake_lookup(
    logic_atom_results: Sequence[Dict[str, Any]],
    eval_temporal_rule_results: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    positions_by_video = _segment_positions_by_video(logic_atom_results)
    for video_result in eval_temporal_rule_results:
        video_id = str(video_result.get("video_id", ""))
        video_positions = positions_by_video.get(video_id, {})
        positions_by_segment_id = dict(video_positions.get("position_by_segment_id", {}))
        segments = list(video_positions.get("segments", []))
        for example in list(video_result.get("examples", [])):
            segment_id = str(example.get("current_segment_id", ""))
            position = positions_by_segment_id.get(segment_id)
            if position is None:
                continue
            horizon_flags = _future_brake_flags(segments, position)
            lookup[(video_id, segment_id)] = {
                "eval_example_id": str(example.get("example_id", "")),
                "brake_label": "brake_next" if bool(example.get("label", False)) else "not_brake_next",
                "brake_next": bool(horizon_flags.get(1, False)),
                "brake_within_2_segments": bool(horizon_flags.get(2, False)),
                "brake_within_3_segments": bool(horizon_flags.get(3, False)),
                "brake_within_5_segments": bool(horizon_flags.get(5, False)),
            }
    return lookup


def _candidate_rows(
    traffic_control_attribute_results: Sequence[Dict[str, Any]],
    brake_lookup: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    restrict_to_eval_segments = bool(brake_lookup)
    for video_result in traffic_control_attribute_results:
        video_id = str(video_result.get("video_id", "unknown"))
        for segment in list(video_result.get("segments", [])):
            segment_index = _safe_int(segment.get("segment_index", -1), -1)
            segment_id = str(segment.get("segment_id", ""))
            brake_meta = brake_lookup.get((video_id, segment_id))
            if restrict_to_eval_segments and brake_meta is None:
                continue
            for obj in list(segment.get("selected_objects", [])):
                attrs = obj.get("traffic_control_attributes")
                if not isinstance(attrs, dict):
                    continue
                if str(attrs.get("traffic_control_type", "")) != "traffic_light":
                    continue
                sample = _choose_best_sample(obj)
                if sample is None:
                    continue
                relevance_score = _safe_float(attrs.get("traffic_control_relevance_score", 0.0))
                relevance_label = str(attrs.get("traffic_control_relevance_label", "low"))
                is_relevant = relevance_score >= _RELEVANCE_THRESHOLD or relevance_label in {"medium", "high"}
                track_id = _safe_int(obj.get("track_id", -1), -1)
                row = {
                    "video_id": video_id,
                    "segment_index": segment_index,
                    "segment_id": segment_id,
                    "frame_index": _safe_int(sample.get("frame_index", -1), -1),
                    "frame_id": _frame_id(_safe_int(sample.get("frame_index", -1), -1)),
                    "track_id": track_id,
                    "object_id": _object_id(track_id),
                    "object_class": str(obj.get("object_class", "traffic_light")),
                    "predicted_state": str(attrs.get("traffic_light_state", "unknown")),
                    "state_confidence": _safe_float(attrs.get("traffic_light_state_confidence", 0.0)),
                    "traffic_light_relevant": bool(is_relevant),
                    "traffic_control_relevant": bool(is_relevant),
                    "position_state": "front_center" if bool(attrs.get("is_front_center_region", False)) else "not_front_center",
                    "relevance_score": relevance_score,
                    "relevance_label": relevance_label,
                    "is_front_center_region": bool(attrs.get("is_front_center_region", False)),
                    "num_evidence_samples": _safe_int(attrs.get("num_evidence_samples", 0), 0),
                    "image_path": str(sample.get("image_path", "")),
                    "box": list(sample.get("box", [0, 0, 0, 0])),
                    "sample_label": str(sample.get("label", "unknown")),
                    "eval_example_id": "",
                    "brake_label": "unknown",
                    "brake_next": "",
                    "brake_within_2_segments": "",
                    "brake_within_3_segments": "",
                    "brake_within_5_segments": "",
                }
                if brake_meta is not None:
                    row.update(brake_meta)
                rows.append(row)
    return rows


def _confidence_band_name(row: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    if not bool(cfg.get("separate_confidence_bands", False)):
        return "all"
    threshold = float(cfg.get("confidence_split_threshold", 0.75))
    return "high_confidence" if _safe_float(row.get("state_confidence", 0.0)) >= threshold else "low_confidence"


def _bucket_key_for_row(row: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[str, str]:
    return (_normalize_text(row.get("predicted_state", "unknown")), _confidence_band_name(row, cfg))


def _bucket_sort_key(bucket_key: Tuple[str, str]) -> Tuple[Tuple[int, str], int, str]:
    state_name, band_name = bucket_key
    band_rank = 0 if band_name == "high_confidence" else 1 if band_name == "low_confidence" else 0
    return (_state_sort_key(state_name), band_rank, band_name)


def _bucket_sample_limit(cfg: Dict[str, Any]) -> int:
    per_state_limit = max(1, int(cfg.get("max_samples_per_state", 50)))
    if not bool(cfg.get("separate_confidence_bands", False)):
        return per_state_limit
    per_band_limit = int(cfg.get("max_samples_per_state_per_confidence_band", 0))
    if per_band_limit > 0:
        return per_band_limit
    return max(1, per_state_limit // 2)


def _effective_max_samples_total(cfg: Dict[str, Any], filtered_rows: Sequence[Dict[str, Any]]) -> int:
    configured = int(cfg.get("max_samples_total", 0))
    if configured > 0:
        return configured
    bucket_keys = {_bucket_key_for_row(row, cfg) for row in filtered_rows}
    return max(1, len(bucket_keys) * _bucket_sample_limit(cfg))


def _build_diverse_bucket_order(
    rows: Sequence[Dict[str, Any]],
    bucket_key: Tuple[str, str],
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(f"{int(seed)}::{bucket_key[0]}::{bucket_key[1]}")
    rows_by_video_and_segment: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        rows_by_video_and_segment[str(row.get("video_id", ""))][_safe_int(row.get("segment_index", -1), -1)].append(
            dict(row)
        )

    video_ids = list(rows_by_video_and_segment.keys())
    rng.shuffle(video_ids)
    per_video_segments: Dict[str, List[List[Dict[str, Any]]]] = {}
    for video_id in video_ids:
        segment_ids = list(rows_by_video_and_segment[video_id].keys())
        rng.shuffle(segment_ids)
        queues: List[List[Dict[str, Any]]] = []
        for segment_id in segment_ids:
            segment_rows = list(rows_by_video_and_segment[video_id][segment_id])
            rng.shuffle(segment_rows)
            segment_rows.sort(
                key=lambda row: (
                    -_box_area(list(row.get("box", [0, 0, 0, 0]))),
                    -_safe_float(row.get("state_confidence", 0.0)),
                    _safe_int(row.get("frame_index", -1), -1),
                )
            )
            queues.append(segment_rows)
        per_video_segments[video_id] = queues

    ordered: List[Dict[str, Any]] = []
    active_videos = list(video_ids)
    while active_videos:
        next_active_videos: List[str] = []
        for video_id in active_videos:
            segment_queues = [queue for queue in per_video_segments.get(video_id, []) if queue]
            if not segment_queues:
                continue
            row = segment_queues[0].pop(0)
            ordered.append(row)
            rotated = segment_queues[1:] + segment_queues[:1]
            per_video_segments[video_id] = rotated
            if any(queue for queue in rotated):
                next_active_videos.append(video_id)
        active_videos = next_active_videos
    return ordered


def _select_samples(
    rows: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    include_unknown_state = bool(cfg.get("include_unknown_state", True))
    min_state_confidence = float(cfg.get("min_state_confidence", 0.0))
    max_samples_per_video = max(1, int(cfg.get("max_samples_per_video", 12)))
    random_seed = int(cfg.get("random_seed", 0))

    filtered_rows: List[Dict[str, Any]] = []
    skipped = {
        "filtered_unknown_state": 0,
        "filtered_below_min_confidence": 0,
        "dropped_by_sampling": 0,
    }
    for row in rows:
        state_name = _normalize_text(row.get("predicted_state", "unknown"))
        if state_name == "unknown" and not include_unknown_state:
            skipped["filtered_unknown_state"] += 1
            continue
        if _safe_float(row.get("state_confidence", 0.0)) < min_state_confidence:
            skipped["filtered_below_min_confidence"] += 1
            continue
        filtered_rows.append(dict(row))

    max_samples_total = _effective_max_samples_total(cfg, filtered_rows)
    bucket_limit = _bucket_sample_limit(cfg)
    rows_by_bucket: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in filtered_rows:
        bucket_key = _bucket_key_for_row(row, cfg)
        row["confidence_band"] = bucket_key[1]
        rows_by_bucket[bucket_key].append(row)

    bucket_order = sorted(rows_by_bucket.keys(), key=_bucket_sort_key)
    per_bucket_orders: Dict[Tuple[str, str], List[Dict[str, Any]]] = {
        bucket_key: _build_diverse_bucket_order(rows_by_bucket[bucket_key], bucket_key, random_seed)
        for bucket_key in bucket_order
    }
    bucket_indices: Dict[Tuple[str, str], int] = {bucket_key: 0 for bucket_key in bucket_order}

    selected: List[Dict[str, Any]] = []
    by_bucket_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    by_video_counts: Dict[str, int] = defaultdict(int)
    seen_keys = set()
    while len(selected) < max_samples_total:
        progress = False
        for bucket_key in bucket_order:
            if len(selected) >= max_samples_total:
                break
            if by_bucket_counts[bucket_key] >= bucket_limit:
                continue
            bucket_rows = per_bucket_orders.get(bucket_key, [])
            start_index = bucket_indices.get(bucket_key, 0)
            while start_index < len(bucket_rows):
                row = bucket_rows[start_index]
                start_index += 1
                video_id = str(row.get("video_id", ""))
                sample_key = (
                    video_id,
                    _safe_int(row.get("segment_index", -1), -1),
                    _safe_int(row.get("frame_index", -1), -1),
                    _safe_int(row.get("track_id", -1), -1),
                )
                if sample_key in seen_keys:
                    continue
                if by_video_counts[video_id] >= max_samples_per_video:
                    continue
                selected.append(row)
                seen_keys.add(sample_key)
                by_bucket_counts[bucket_key] += 1
                by_video_counts[video_id] += 1
                progress = True
                break
            bucket_indices[bucket_key] = start_index
        if not progress:
            break
    skipped["dropped_by_sampling"] = max(0, len(filtered_rows) - len(selected))
    return selected, skipped


def _state_color_bgr(state_name: str) -> Tuple[int, int, int]:
    normalized = _normalize_text(state_name)
    if normalized == "red":
        return (40, 40, 230)
    if normalized == "yellow":
        return (0, 210, 255)
    if normalized == "green":
        return (40, 180, 60)
    return (220, 220, 220)


def _draw_label_block(
    image: Any,
    lines: Sequence[str],
    color: Tuple[int, int, int],
) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 2
    x0 = 10
    y0 = 12
    line_height = 22
    widths = []
    for line in lines:
        (width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        widths.append(width)
    block_w = max(widths) + 16 if widths else 120
    block_h = line_height * len(lines) + 12
    cv2.rectangle(image, (x0, y0), (x0 + block_w, y0 + block_h), (20, 20, 20), -1)
    cv2.rectangle(image, (x0, y0), (x0 + block_w, y0 + block_h), color, 2)
    for idx, line in enumerate(lines):
        y = y0 + 24 + idx * line_height
        cv2.putText(image, line, (x0 + 8, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _render_audit_frame(row: Dict[str, Any], output_path: Path) -> bool:
    try:
        import cv2
    except Exception:
        return False

    image_path = _resolve_local_image_path(str(row.get("image_path", "")))
    frame = cv2.imread(image_path)
    if frame is None:
        return False

    h, w = frame.shape[:2]
    box = list(row.get("box", [0, 0, 0, 0]))
    if len(box) != 4:
        return False
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, max(0, w - 1)))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, max(0, h - 1)))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return False

    state_name = str(row.get("predicted_state", "unknown"))
    color = _state_color_bgr(state_name)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    label_lines = [
        f"video={row.get('video_id', 'unknown')} segment={row.get('segment_id', 'unknown')} frame={row.get('frame_id', 'unknown')}",
        f"track={row.get('track_id', -1)} object={row.get('object_id', 'unknown')} brake={row.get('brake_label', 'unknown')}",
        f"state={state_name} conf={_safe_float(row.get('state_confidence', 0.0)):.2f}",
        f"relevance tl={bool(row.get('traffic_light_relevant', False))} tc={bool(row.get('traffic_control_relevant', False))} pos={row.get('position_state', 'unknown')}",
    ]
    _draw_label_block(frame, label_lines, color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), frame))


def _index_fieldnames() -> List[str]:
    return [
        "video_id",
        "segment_id",
        "frame_id",
        "object_id",
        "track_id",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_area",
        "frame_with_bbox_path",
        "predicted_state",
        "state_confidence",
        "traffic_light_relevant",
        "traffic_control_relevant",
        "position_state",
        "brake_next",
        "brake_within_2_segments",
        "brake_within_3_segments",
        "brake_within_5_segments",
    ]


def process_audit(
    traffic_control_attribute_results: Sequence[Dict[str, Any]],
    logic_atom_results: Optional[Sequence[Dict[str, Any]]] = None,
    eval_temporal_rule_results: Optional[Sequence[Dict[str, Any]]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_json_path = out_root / "traffic_light_detection_quality_audit_summary.json"
    manifest_json_path = out_root / "traffic_light_detection_quality_audit_manifest.json"
    samples_csv_path = out_root / "traffic_light_detection_quality_audit_samples.csv"
    index_csv_path = out_root / "traffic_light_detection_quality_audit_index.csv"

    if not force_recompute and summary_json_path.exists() and manifest_json_path.exists() and index_csv_path.exists():
        with summary_json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _TRAFFIC_LIGHT_DETECTION_QUALITY_AUDIT_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {summary_json_path.name}")
            return cached

    brake_lookup = _build_brake_lookup(logic_atom_results or [], eval_temporal_rule_results or [])
    candidate_rows_all = _candidate_rows(traffic_control_attribute_results, {})
    candidate_rows = _candidate_rows(traffic_control_attribute_results, brake_lookup)
    selected_rows, skipped_counts = _select_samples(candidate_rows, cfg)

    saved_rows: List[Dict[str, Any]] = []
    state_saved_counts: Dict[str, int] = defaultdict(int)
    bucket_saved_counts: Dict[str, int] = defaultdict(int)
    render_failure_count = 0
    for row in selected_rows:
        state_name = _normalize_text(row.get("predicted_state", "unknown"))
        confidence_band = str(row.get("confidence_band", "all"))
        confidence_text = f"{_safe_float(row.get('state_confidence', 0.0)):.2f}".replace(".", "p")
        file_name = (
            f"{_sanitize_filename(row.get('video_id', 'unknown'))}"
            f"_{_sanitize_filename(row.get('segment_id', 'unknown'))}"
            f"_{_sanitize_filename(row.get('frame_id', 'unknown'))}"
            f"_{_sanitize_filename(row.get('object_id', 'unknown'))}"
            f"_{_sanitize_filename(row.get('predicted_state', 'unknown'))}"
            f"_conf{confidence_text}"
            f"_{_sanitize_filename(row.get('brake_label', 'unknown'))}.jpg"
        )
        output_path = out_root / state_name / file_name
        if not _render_audit_frame(row, output_path):
            render_failure_count += 1
            continue

        box = list(row.get("box", [0, 0, 0, 0]))
        out_row = dict(row)
        out_row["resolved_image_path"] = _resolve_local_image_path(str(row.get("image_path", "")))
        out_row["frame_with_bbox_path"] = str(output_path)
        out_row["bbox_x1"] = _safe_float(box[0] if len(box) > 0 else 0.0)
        out_row["bbox_y1"] = _safe_float(box[1] if len(box) > 1 else 0.0)
        out_row["bbox_x2"] = _safe_float(box[2] if len(box) > 2 else 0.0)
        out_row["bbox_y2"] = _safe_float(box[3] if len(box) > 3 else 0.0)
        out_row["bbox_area"] = _box_area(box)
        saved_rows.append(out_row)
        state_saved_counts[state_name] += 1
        bucket_saved_counts[f"{state_name}::{confidence_band}"] += 1

    saved_rows = sorted(
        saved_rows,
        key=lambda row: (
            _state_sort_key(_normalize_text(row.get("predicted_state", "unknown"))),
            str(row.get("video_id", "")),
            str(row.get("segment_id", "")),
            str(row.get("frame_id", "")),
            _safe_int(row.get("track_id", -1), -1),
        ),
    )

    index_rows = [
        {
            "video_id": str(row.get("video_id", "")),
            "segment_id": str(row.get("segment_id", "")),
            "frame_id": str(row.get("frame_id", "")),
            "object_id": str(row.get("object_id", "")),
            "track_id": _safe_int(row.get("track_id", -1), -1),
            "bbox_x1": _safe_float(row.get("bbox_x1", 0.0)),
            "bbox_y1": _safe_float(row.get("bbox_y1", 0.0)),
            "bbox_x2": _safe_float(row.get("bbox_x2", 0.0)),
            "bbox_y2": _safe_float(row.get("bbox_y2", 0.0)),
            "bbox_area": _safe_float(row.get("bbox_area", 0.0)),
            "frame_with_bbox_path": str(row.get("frame_with_bbox_path", "")),
            "predicted_state": str(row.get("predicted_state", "unknown")),
            "state_confidence": _safe_float(row.get("state_confidence", 0.0)),
            "traffic_light_relevant": bool(row.get("traffic_light_relevant", False)),
            "traffic_control_relevant": bool(row.get("traffic_control_relevant", False)),
            "position_state": str(row.get("position_state", "unknown")),
            "brake_next": row.get("brake_next", ""),
            "brake_within_2_segments": row.get("brake_within_2_segments", ""),
            "brake_within_3_segments": row.get("brake_within_3_segments", ""),
            "brake_within_5_segments": row.get("brake_within_5_segments", ""),
        }
        for row in saved_rows
    ]
    _write_csv(index_csv_path, _index_fieldnames(), index_rows)
    _write_csv(samples_csv_path, _index_fieldnames(), index_rows)

    requested_sample_size_per_state = {
        state: int(cfg.get("max_samples_per_state", 50))
        for state in _STATE_ORDER
    }
    actual_exported_count_per_state = {
        state: int(state_saved_counts.get(state, 0))
        for state in _STATE_ORDER
    }
    skipped_examples = {
        **{key: int(value) for key, value in skipped_counts.items()},
        "render_failed_or_missing_image": int(render_failure_count),
        "selected_but_not_exported": max(0, len(selected_rows) - len(saved_rows)),
        "non_eval_segments_excluded": max(0, len(candidate_rows_all) - len(candidate_rows)),
    }

    output_paths = {
        "summary_json": str(summary_json_path),
        "manifest_json": str(manifest_json_path),
        "samples_csv": str(samples_csv_path),
        "index_csv": str(index_csv_path),
        "state_directories_root": str(out_root),
    }

    manifest = {
        "version": _TRAFFIC_LIGHT_DETECTION_QUALITY_AUDIT_VERSION,
        "requested_sample_size_per_state": requested_sample_size_per_state,
        "actual_exported_count_per_state": actual_exported_count_per_state,
        "skipped_examples": skipped_examples,
        "random_seed": int(cfg.get("random_seed", 0)),
        "output_paths": output_paths,
    }
    with manifest_json_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    summary = {
        "version": _TRAFFIC_LIGHT_DETECTION_QUALITY_AUDIT_VERSION,
        "config": _cfg_key_subset(cfg),
        "uses_full_frame_visualizations": True,
        "saves_cropped_bboxes": False,
        "sampling_strategy": (
            "balanced_state_and_confidence_band_round_robin"
            if bool(cfg.get("separate_confidence_bands", False))
            else "balanced_state_round_robin"
        ),
        "scope": "evaluation_split_only" if brake_lookup else "all_available_videos",
        "num_candidate_traffic_lights": len(candidate_rows),
        "num_selected_audit_samples": len(selected_rows),
        "num_saved_audit_images": len(saved_rows),
        "predicted_state_counts_saved": actual_exported_count_per_state,
        "bucket_counts_saved": {key: int(value) for key, value in sorted(bucket_saved_counts.items())},
        "skipped_examples": skipped_examples,
        "output_paths": output_paths,
        "samples": [
            {
                "video_id": str(row.get("video_id", "")),
                "segment_id": str(row.get("segment_id", "")),
                "frame_id": str(row.get("frame_id", "")),
                "object_id": str(row.get("object_id", "")),
                "track_id": _safe_int(row.get("track_id", -1), -1),
                "predicted_state": str(row.get("predicted_state", "unknown")),
                "state_confidence": _safe_float(row.get("state_confidence", 0.0)),
                "brake_label": str(row.get("brake_label", "unknown")),
                "frame_with_bbox_path": str(row.get("frame_with_bbox_path", "")),
            }
            for row in saved_rows
        ],
    }
    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        "  traffic_light_detection_quality_audit: "
        f"candidates={len(candidate_rows)} | selected={len(selected_rows)} | saved={len(saved_rows)}"
    )
    print(f"Traffic-light detection quality audit summary JSON written to {summary_json_path}")
    print(f"Traffic-light detection quality audit manifest JSON written to {manifest_json_path}")
    print(f"Traffic-light detection quality audit index CSV written to {index_csv_path}")
    print(f"Traffic-light detection quality audit frames written to {out_root}")
    return summary


def run(
    traffic_control_attribute_results: Sequence[Dict[str, Any]],
    logic_atom_results: Optional[Sequence[Dict[str, Any]]] = None,
    eval_temporal_rule_results: Optional[Sequence[Dict[str, Any]]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_audit(
        traffic_control_attribute_results=traffic_control_attribute_results,
        logic_atom_results=logic_atom_results,
        eval_temporal_rule_results=eval_temporal_rule_results,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
