"""
Analyze and select important objects for each temporal segment.

Current status:
  - Structural placeholder only.
  - The actual importance/filtering strategy is intentionally not implemented yet.
  - For now, all objects are passed through as selected objects so downstream
    steps can be wired against the final schema.

Consumes:
  - Step 9 output: segment-level object motion summaries

Output layout:
    pipeline_output/10_driving_mini_important_objects/
        important_objects_manifest.json
        <video_id>/
            important_objects.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_IMPORTANT_OBJECTS_VERSION = 3


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "10_driving_mini_important_objects"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "selection_strategy",
        "passthrough_selected_objects",
        "candidate_selection",
    ]
    return {k: cfg.get(k) for k in keys}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _candidate_class(candidate_object: Dict[str, Any]) -> str:
    return str(
        candidate_object.get("object_class", candidate_object.get("label", "unknown"))
        or "unknown"
    ).strip() or "unknown"


def _candidate_prior_ids(candidate_object: Dict[str, Any]) -> List[str]:
    prior_metadata = dict(candidate_object.get("prior_metadata", {}))
    prior_ids = {
        str(value).strip()
        for value in list(prior_metadata.get("matched_prior_ids", []))
        + list(prior_metadata.get("track_matched_prior_ids", []))
        if str(value).strip()
    }
    return sorted(prior_ids) or ["none"]


def _prior_relevance(candidate_object: Dict[str, Any]) -> float:
    prior_metadata = dict(candidate_object.get("prior_metadata", {}))
    values = [
        _safe_float(prior_metadata.get("prior_relevance_score", 0.0)),
        _safe_float(prior_metadata.get("prior_relevance_mean", 0.0)),
        _safe_float(prior_metadata.get("prior_relevance_max", 0.0)),
    ]
    return max(values) if values else 0.0


def _semantic_relevance(candidate_object: Dict[str, Any]) -> float:
    traffic_attrs = candidate_object.get("traffic_control_attributes", {})
    if isinstance(traffic_attrs, dict):
        relevance = _safe_float(traffic_attrs.get("traffic_control_relevance_score", 0.0))
        if relevance > 0.0:
            return relevance
    class_name = _candidate_class(candidate_object).lower()
    if class_name in {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle", "pedestrian"}:
        return 0.6
    if class_name in {"traffic light", "traffic_light", "stop sign", "stop_sign"}:
        return 0.8
    return 0.3


def _candidate_selection_score(candidate_object: Dict[str, Any], weights: Dict[str, Any]) -> float:
    track_quality = dict(candidate_object.get("track_quality", {}))
    raw_score = _safe_float(candidate_object.get("score", candidate_object.get("raw_score", 0.0)))
    calibrated_score = _safe_float(candidate_object.get("calibrated_score", raw_score))
    selection_score = _safe_float(candidate_object.get("selection_score", 0.0))
    track_quality_score = max(
        _safe_float(track_quality.get("mean_score", 0.0)),
        _safe_float(track_quality.get("max_score", 0.0)),
        selection_score,
    )
    temporal_consistency = _safe_float(track_quality.get("temporal_consistency", 0.0))
    prior_relevance = _prior_relevance(candidate_object)
    semantic_relevance = _semantic_relevance(candidate_object)
    score = (
        _safe_float(weights.get("calibrated_score", 0.30), 0.30) * calibrated_score
        + _safe_float(weights.get("raw_score", 0.15), 0.15) * raw_score
        + _safe_float(weights.get("track_quality", 0.20), 0.20) * track_quality_score
        + _safe_float(weights.get("temporal_consistency", 0.15), 0.15) * temporal_consistency
        + _safe_float(weights.get("prior_relevance", 0.10), 0.10) * prior_relevance
        + _safe_float(weights.get("semantic_relevance", 0.10), 0.10) * semantic_relevance
    )
    return float(score)


def _empty_candidate_counts() -> Dict[str, Any]:
    return {
        "input": 0,
        "selected": 0,
        "rejected": 0,
        "by_class": {},
        "by_prior": {},
        "rejection_reasons": {},
    }


def _increment_count(bucket: Dict[str, Any], section: str, key: str, field: str) -> None:
    section_map = bucket.setdefault(section, {})
    entry = section_map.setdefault(str(key), {"input": 0, "selected": 0, "rejected": 0})
    entry[field] = int(entry.get(field, 0)) + 1


def _record_rejection(bucket: Dict[str, Any], reason: str) -> None:
    reasons = bucket.setdefault("rejection_reasons", {})
    reasons[reason] = int(reasons.get(reason, 0)) + 1


def _select_candidate_objects_for_segment(
    candidate_objects: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    selection_cfg = dict(cfg.get("candidate_selection", {}))
    enabled = bool(selection_cfg.get("enabled", True))
    max_per_segment = max(0, _safe_int(selection_cfg.get("max_per_segment", 8), 8))
    max_per_class = max(0, _safe_int(selection_cfg.get("max_per_class", 3), 3))
    max_per_prior = max(0, _safe_int(selection_cfg.get("max_per_prior", 2), 2))
    min_score = _safe_float(selection_cfg.get("min_score", 0.0), 0.0)
    weights = dict(selection_cfg.get("score_weights", {}))

    stats = _empty_candidate_counts()
    scored: List[Dict[str, Any]] = []
    for index, candidate_object in enumerate(candidate_objects):
        class_name = _candidate_class(candidate_object)
        prior_ids = _candidate_prior_ids(candidate_object)
        stats["input"] += 1
        _increment_count(stats, "by_class", class_name, "input")
        for prior_id in prior_ids:
            _increment_count(stats, "by_prior", prior_id, "input")
        enriched = dict(candidate_object)
        score = _candidate_selection_score(enriched, weights)
        enriched["candidate_selection_score"] = score
        enriched["candidate_selection_rank_key"] = {
            "score": score,
            "calibrated_score": _safe_float(enriched.get("calibrated_score", enriched.get("score", 0.0))),
            "raw_score": _safe_float(enriched.get("score", enriched.get("raw_score", 0.0))),
            "track_quality": dict(enriched.get("track_quality", {})),
            "prior_relevance": _prior_relevance(enriched),
            "semantic_relevance": _semantic_relevance(enriched),
            "input_index": index,
        }
        scored.append(enriched)

    if not enabled:
        selected = scored
        rejected: List[Dict[str, Any]] = []
    else:
        selected = []
        rejected = []
        class_counts: Dict[str, int] = defaultdict(int)
        prior_counts: Dict[str, int] = defaultdict(int)
        for candidate_object in sorted(
            scored,
            key=lambda item: (
                -_safe_float(item.get("candidate_selection_score", 0.0)),
                -_safe_float(item.get("calibrated_score", item.get("score", 0.0))),
                str(item.get("candidate_object_id", "")),
            ),
        ):
            class_name = _candidate_class(candidate_object)
            prior_ids = _candidate_prior_ids(candidate_object)
            reason = ""
            if _safe_float(candidate_object.get("candidate_selection_score", 0.0)) < min_score:
                reason = "below_candidate_selection_score"
            elif max_per_segment and len(selected) >= max_per_segment:
                reason = "segment_candidate_budget_exceeded"
            elif max_per_class and class_counts[class_name] >= max_per_class:
                reason = "class_candidate_budget_exceeded"
            elif max_per_prior and any(prior_counts[prior_id] >= max_per_prior for prior_id in prior_ids):
                reason = "prior_candidate_budget_exceeded"

            if reason:
                rejected_entry = dict(candidate_object)
                rejected_entry["candidate_rejection_reason"] = reason
                rejected.append(rejected_entry)
                _record_rejection(stats, reason)
                continue

            selected.append(candidate_object)
            class_counts[class_name] += 1
            for prior_id in prior_ids:
                prior_counts[prior_id] += 1

    selected_ids = {
        str(obj.get("candidate_object_id", obj.get("candidate_track_id", index)))
        for index, obj in enumerate(selected)
    }
    if not enabled:
        rejected = []
    else:
        already_rejected = {
            str(obj.get("candidate_object_id", obj.get("candidate_track_id", index)))
            for index, obj in enumerate(rejected)
        }
        for index, candidate_object in enumerate(scored):
            key = str(candidate_object.get("candidate_object_id", candidate_object.get("candidate_track_id", index)))
            if key not in selected_ids and key not in already_rejected:
                rejected_entry = dict(candidate_object)
                rejected_entry["candidate_rejection_reason"] = "not_selected"
                rejected.append(rejected_entry)
                _record_rejection(stats, "not_selected")

    for candidate_object in selected:
        class_name = _candidate_class(candidate_object)
        prior_ids = _candidate_prior_ids(candidate_object)
        stats["selected"] += 1
        _increment_count(stats, "by_class", class_name, "selected")
        for prior_id in prior_ids:
            _increment_count(stats, "by_prior", prior_id, "selected")
    for candidate_object in rejected:
        class_name = _candidate_class(candidate_object)
        prior_ids = _candidate_prior_ids(candidate_object)
        stats["rejected"] += 1
        _increment_count(stats, "by_class", class_name, "rejected")
        for prior_id in prior_ids:
            _increment_count(stats, "by_prior", prior_id, "rejected")

    return selected, rejected, stats


def _merge_candidate_count_stats(total: Dict[str, Any], segment_stats: Dict[str, Any]) -> None:
    total["input"] = int(total.get("input", 0)) + int(segment_stats.get("input", 0))
    total["selected"] = int(total.get("selected", 0)) + int(segment_stats.get("selected", 0))
    total["rejected"] = int(total.get("rejected", 0)) + int(segment_stats.get("rejected", 0))
    for section in ("by_class", "by_prior"):
        target_section = total.setdefault(section, {})
        for key, counts in dict(segment_stats.get(section, {})).items():
            target_counts = target_section.setdefault(str(key), {"input": 0, "selected": 0, "rejected": 0})
            for field in ("input", "selected", "rejected"):
                target_counts[field] = int(target_counts.get(field, 0)) + int(dict(counts).get(field, 0))
    reasons = total.setdefault("rejection_reasons", {})
    for reason, count in dict(segment_stats.get("rejection_reasons", {})).items():
        reasons[str(reason)] = int(reasons.get(str(reason), 0)) + int(count)


def _print_video_summary(result: Dict[str, Any]) -> None:
    print(
        f"  {result.get('video_id', 'unknown')}: "
        f"objects={int(result.get('num_objects', 0))} | "
        f"candidate_objects={int(result.get('num_candidate_objects', 0))} | "
        f"selected_objects={int(result.get('num_selected_objects', 0))} | "
        f"selected_candidate_objects={int(result.get('num_selected_candidate_objects', 0))} | "
        f"strategy_applied={bool(result.get('selection_strategy_applied', False))}"
    )


def process_video(
    segment_object_motion_video_result: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    selection_strategy = str(cfg.get("selection_strategy", "not_implemented"))
    passthrough_selected_objects = bool(cfg.get("passthrough_selected_objects", True))

    video_id = str(segment_object_motion_video_result["video_id"])
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "important_objects.json"

    if not force_recompute and out_file.exists():
        with out_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        cache_cfg = cached.get("config", {}) if isinstance(cached, dict) else {}
        if (
            int(cached.get("version", 0)) == _IMPORTANT_OBJECTS_VERSION
            and _cfg_key_subset(cache_cfg) == _cfg_key_subset(
                {
                    "selection_strategy": selection_strategy,
                    "passthrough_selected_objects": passthrough_selected_objects,
                }
            )
        ):
            print(f"  [cache] {video_id} - loading {out_file.name}")
            _print_video_summary(cached)
            return cached

    segments_in = list(segment_object_motion_video_result.get("segments", []))
    segments_out: List[Dict[str, Any]] = []
    num_objects = 0
    num_candidate_objects = 0
    num_selected_objects = 0
    num_selected_candidate_objects = 0
    candidate_selection_stats = _empty_candidate_counts()

    for segment in segments_in:
        objects = list(segment.get("objects", []))
        candidate_objects = list(segment.get("candidate_objects", []))
        selected_objects = list(objects) if passthrough_selected_objects else []
        if passthrough_selected_objects:
            (
                selected_candidate_objects,
                rejected_candidate_objects,
                segment_candidate_selection_stats,
            ) = _select_candidate_objects_for_segment(candidate_objects, cfg)
        else:
            selected_candidate_objects = []
            rejected_candidate_objects = list(candidate_objects)
            segment_candidate_selection_stats = _empty_candidate_counts()
            for candidate_object in rejected_candidate_objects:
                class_name = _candidate_class(candidate_object)
                prior_ids = _candidate_prior_ids(candidate_object)
                segment_candidate_selection_stats["input"] += 1
                segment_candidate_selection_stats["rejected"] += 1
                _increment_count(segment_candidate_selection_stats, "by_class", class_name, "input")
                _increment_count(segment_candidate_selection_stats, "by_class", class_name, "rejected")
                for prior_id in prior_ids:
                    _increment_count(segment_candidate_selection_stats, "by_prior", prior_id, "input")
                    _increment_count(segment_candidate_selection_stats, "by_prior", prior_id, "rejected")
                _record_rejection(segment_candidate_selection_stats, "passthrough_disabled")
        filtered_objects: List[Dict[str, Any]] = []
        filtered_candidate_objects: List[Dict[str, Any]] = rejected_candidate_objects

        num_objects += len(objects)
        num_candidate_objects += len(candidate_objects)
        num_selected_objects += len(selected_objects)
        num_selected_candidate_objects += len(selected_candidate_objects)
        _merge_candidate_count_stats(candidate_selection_stats, segment_candidate_selection_stats)

        segment_out = dict(segment)
        segment_out["objects"] = objects
        segment_out["candidate_objects"] = candidate_objects
        segment_out["selected_objects"] = selected_objects
        segment_out["selected_candidate_objects"] = selected_candidate_objects
        segment_out["filtered_objects"] = filtered_objects
        segment_out["filtered_candidate_objects"] = filtered_candidate_objects
        segment_out["rejected_candidate_objects"] = rejected_candidate_objects
        segment_out["candidate_selection_stats"] = segment_candidate_selection_stats
        segment_out["selection_strategy"] = selection_strategy
        segment_out["selection_strategy_applied"] = False
        segment_out["num_objects"] = len(objects)
        segment_out["num_candidate_objects"] = len(candidate_objects)
        segment_out["num_selected_objects"] = len(selected_objects)
        segment_out["num_selected_candidate_objects"] = len(selected_candidate_objects)
        segments_out.append(segment_out)

    result: Dict[str, Any] = {
        "version": _IMPORTANT_OBJECTS_VERSION,
        "video_id": video_id,
        "selection_strategy_applied": False,
        "num_segments": len(segments_out),
        "num_objects": num_objects,
        "num_candidate_objects": num_candidate_objects,
        "num_selected_objects": num_selected_objects,
        "num_selected_candidate_objects": num_selected_candidate_objects,
        "num_rejected_candidate_objects": int(candidate_selection_stats.get("rejected", 0)),
        "candidate_selection_stats": candidate_selection_stats,
        "config": {
            "selection_strategy": selection_strategy,
            "passthrough_selected_objects": passthrough_selected_objects,
            "candidate_selection": dict(cfg.get("candidate_selection", {})),
        },
        "segments": segments_out,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    _print_video_summary(result)
    return result


def run(
    segment_object_motion_results: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    out_root = output_root or get_output_root()
    results: List[Dict[str, Any]] = []

    for segment_object_result in segment_object_motion_results:
        result = process_video(
            segment_object_motion_video_result=segment_object_result,
            cfg=cfg,
            output_root=out_root,
            force_recompute=force_recompute,
        )
        results.append(result)

    aggregate_candidate_selection_stats = _empty_candidate_counts()
    for result in results:
        _merge_candidate_count_stats(
            aggregate_candidate_selection_stats,
            dict(result.get("candidate_selection_stats", {})),
        )

    manifest = {
        "version": _IMPORTANT_OBJECTS_VERSION,
        "num_videos": len(results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_segments": r.get("num_segments", 0),
                "num_objects": r.get("num_objects", 0),
                "num_candidate_objects": r.get("num_candidate_objects", 0),
                "num_selected_objects": r.get("num_selected_objects", 0),
                "num_selected_candidate_objects": r.get("num_selected_candidate_objects", 0),
                "num_rejected_candidate_objects": r.get("num_rejected_candidate_objects", 0),
                "selection_strategy_applied": r.get("selection_strategy_applied", False),
            }
            for r in results
        ],
        "candidate_selection_stats": aggregate_candidate_selection_stats,
    }
    manifest_path = out_root / "important_objects_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Important objects manifest written to {manifest_path}")
    return results
