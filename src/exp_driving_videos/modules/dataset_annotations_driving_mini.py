"""
Load dataset-provided object annotations (ground-truth tracks) for driving_mini.

Source:
    dataset/driving_mini/labels.csv

The CSV includes per-object rows with fields such as:
    videoName, frameIndex, id, category, box2d.x1, box2d.x2, box2d.y1, box2d.y2

Output layout:
    pipeline_output/03_driving_mini_dataset_annotations/
        dataset_annotations_manifest.json
        <video_id>/
            dataset_annotations.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


_DATASET_ANNOTATIONS_VERSION = 3


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def get_labels_csv() -> Path:
    return config.get_dataset_path("driving_mini") / "labels.csv"


def get_frames_root() -> Path:
    return config.get_dataset_path("driving_mini") / "frames"


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "03_driving_mini_dataset_annotations"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iter_rows(labels_csv: Path) -> Iterable[Dict[str, str]]:
    with labels_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def _selected_candidate_summary_by_track(
    tracking_video_result: Optional[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    if not isinstance(tracking_video_result, dict):
        return {}
    track_summaries = (
        tracking_video_result.get("candidate_tracks", {})
        .get("selected_candidate_tracks", {})
        .get("track_summaries", [])
    )
    return {
        int(row.get("track_id", -1)): dict(row)
        for row in track_summaries
        if int(row.get("track_id", -1)) >= 0
    }


def _selected_candidate_objects_by_frame(
    tracking_video_result: Optional[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    if not isinstance(tracking_video_result, dict):
        return {}
    frame_map: Dict[int, Dict[str, Any]] = {}
    summary_by_track = _selected_candidate_summary_by_track(tracking_video_result)
    candidate_frames = (
        tracking_video_result.get("candidate_tracks", {})
        .get("selected_candidate_tracks", {})
        .get("frames", [])
    )
    for frame in candidate_frames:
        frame_index = _to_int(frame.get("frame_index", -1), default=-1)
        if frame_index < 0:
            continue
        frame_entry = frame_map.setdefault(
            frame_index,
            {
                "frame": str(frame.get("frame", "")),
                "frame_index": frame_index,
                "image_path": str(frame.get("image_path", "")),
                "candidate_objects": [],
            },
        )
        boxes = list(frame.get("boxes", []))
        scores = list(frame.get("scores", []))
        labels = list(frame.get("labels", []))
        track_ids = list(frame.get("track_ids", []))
        detection_ids = list(frame.get("detection_ids", []))
        candidate_sources = list(frame.get("candidate_sources", []))
        prior_relevance_scores = list(frame.get("prior_relevance_scores", []))
        matched_prior_ids = list(frame.get("matched_prior_ids", []))
        for index, box in enumerate(boxes):
            candidate_track_id = _to_int(track_ids[index] if index < len(track_ids) else -1, default=-1)
            track_summary = dict(summary_by_track.get(candidate_track_id, {}))
            frame_entry["candidate_objects"].append(
                {
                    "bbox": list(box),
                    "score": _to_float(scores[index] if index < len(scores) else 0.0),
                    "label": str(labels[index] if index < len(labels) else "unknown"),
                    "accepted": False,
                    "source_type": "selected_candidate_track",
                    "candidate_track_id": candidate_track_id,
                    "track_id": candidate_track_id,
                    "frame_detection_id": str(detection_ids[index] if index < len(detection_ids) else ""),
                    "source_detection_ids": list(track_summary.get("source_detection_ids", [])),
                    "candidate_source": str(candidate_sources[index] if index < len(candidate_sources) else ""),
                    "prior_metadata": {
                        "matched_prior_ids": list(matched_prior_ids[index] if index < len(matched_prior_ids) else []),
                        "track_matched_prior_ids": list(track_summary.get("matched_prior_ids", [])),
                        "matched_prior_id_counts": dict(track_summary.get("matched_prior_id_counts", {})),
                        "prior_relevance_score": _to_float(
                            prior_relevance_scores[index] if index < len(prior_relevance_scores) else 0.0
                        ),
                        "prior_relevance_mean": _to_float(track_summary.get("prior_relevance_mean", 0.0)),
                        "prior_relevance_max": _to_float(track_summary.get("prior_relevance_max", 0.0)),
                        "prior_relevance_min": _to_float(track_summary.get("prior_relevance_min", 0.0)),
                    },
                    "score_breakdown": dict(track_summary.get("score_breakdown", {})),
                    "track_quality": {
                        "mean_score": _to_float(track_summary.get("mean_score", 0.0)),
                        "max_score": _to_float(track_summary.get("max_score", 0.0)),
                        "track_length": _to_int(track_summary.get("track_length", 0), default=0),
                        "temporal_consistency": _to_float(track_summary.get("temporal_consistency", 0.0)),
                        "selection_score": _to_float(track_summary.get("selection_score", 0.0)),
                    },
                }
            )
    return frame_map


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------


def process_video(
    video_id: str,
    rows: List[Dict[str, str]],
    tracking_video_result: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dataset_annotations.json"

    if not force_recompute and out_file.exists():
        print(f"  [cache] {video_id} - loading {out_file.name}")
        with out_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _DATASET_ANNOTATIONS_VERSION:
            cached["from_cache"] = True
            return cached

    frame_dir = get_frames_root() / video_id
    available_frames = {
        int(frame_path.stem.split("_")[-1]): frame_path
        for frame_path in sorted(frame_dir.glob("frame_*.jpg"))
    }
    candidate_frame_map = _selected_candidate_objects_by_frame(tracking_video_result)

    frame_map: Dict[int, Dict[str, Any]] = {}
    all_track_ids: set[int] = set()
    candidate_track_ids: set[int] = set()
    skipped_rows = 0

    for row in rows:
        frame_index = _to_int(row.get("frameIndex", "0"), default=0)
        frame_path = available_frames.get(frame_index)
        if frame_path is None:
            skipped_rows += 1
            continue

        frame_entry = frame_map.setdefault(
            frame_index,
            {
                "frame": frame_path.name,
                "frame_index": frame_index,
                "image_path": str(frame_path),
                "boxes": [],
                "scores": [],
                "labels": [],
                "track_ids": [],
                "objects": [],
                "candidate_objects": list(candidate_frame_map.get(frame_index, {}).get("candidate_objects", [])),
                "is_ground_truth": True,
            },
        )

        x1 = _to_float(row.get("box2d.x1", "0"))
        x2 = _to_float(row.get("box2d.x2", "0"))
        y1 = _to_float(row.get("box2d.y1", "0"))
        y2 = _to_float(row.get("box2d.y2", "0"))
        track_id = _to_int(row.get("id", "-1"), default=-1)
        category = (row.get("category") or "unknown").strip()

        frame_entry["boxes"].append([x1, y1, x2, y2])
        frame_entry["scores"].append(1.0)
        frame_entry["labels"].append(category)
        frame_entry["track_ids"].append(track_id)
        frame_entry["objects"].append(
            {
                "bbox": [x1, y1, x2, y2],
                "score": 1.0,
                "label": category,
                "track_id": track_id,
                "source_type": "ground_truth_annotation",
                "accepted": True,
                "is_ground_truth": True,
            }
        )

        if track_id >= 0:
            all_track_ids.add(track_id)

    for frame_index, candidate_entry in candidate_frame_map.items():
        frame_path = available_frames.get(frame_index)
        if frame_path is None:
            continue
        frame_entry = frame_map.setdefault(
            frame_index,
            {
                "frame": str(candidate_entry.get("frame", frame_path.name or f"frame_{frame_index:05d}.jpg")),
                "frame_index": frame_index,
                "image_path": str(candidate_entry.get("image_path", str(frame_path))),
                "boxes": [],
                "scores": [],
                "labels": [],
                "track_ids": [],
                "objects": [],
                "candidate_objects": [],
                "is_ground_truth": True,
            },
        )
        frame_entry["candidate_objects"] = list(candidate_entry.get("candidate_objects", []))

    for frame in frame_map.values():
        for candidate_object in list(frame.get("candidate_objects", [])):
            candidate_track_id = _to_int(candidate_object.get("candidate_track_id", -1), default=-1)
            if candidate_track_id >= 0:
                candidate_track_ids.add(candidate_track_id)

    frames = [frame_map[k] for k in sorted(frame_map.keys())]
    result: Dict[str, Any] = {
        "version": _DATASET_ANNOTATIONS_VERSION,
        "video_id": video_id,
        "from_cache": False,
        "source": "dataset/driving_mini/labels.csv",
        "num_frames": len(frames),
        "num_tracks": len(all_track_ids),
        "num_objects": sum(len(frame.get("objects", [])) for frame in frames),
        "num_candidate_tracks": len(candidate_track_ids),
        "num_candidate_objects": sum(len(frame.get("candidate_objects", [])) for frame in frames),
        "frames": frames,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(
        f"  {video_id}: {result['num_frames']} frames, "
        f"{result['num_tracks']} GT tracks"
    )
    if skipped_rows:
        print(f"  [warn] Skipped {skipped_rows} GT rows without extracted frames")
    return result


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def run(
    video_ids: Optional[List[str]] = None,
    tracking_results: Optional[List[Dict[str, Any]]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    labels_csv = get_labels_csv()
    if not labels_csv.exists():
        raise FileNotFoundError(f"Ground-truth labels CSV not found: {labels_csv}")

    rows_by_video: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in _iter_rows(labels_csv):
        video_name = (row.get("videoName") or "").strip()
        if not video_name:
            continue
        rows_by_video[video_name].append(row)

    available_videos = sorted(rows_by_video.keys())
    tracking_by_video = {
        str(result.get("video_id", "")): result
        for result in (tracking_results or [])
        if str(result.get("video_id", ""))
    }
    target_videos = list(video_ids) if video_ids else available_videos

    unknown = set(target_videos) - set(available_videos)
    if unknown:
        print(f"  [warn] Skipping {len(unknown)} video(s) without GT annotations: {sorted(unknown)}")
        target_videos = [v for v in target_videos if v in available_videos]
        if not target_videos:
            print("  [warn] No videos with GT annotations found. Returning empty results.")
            return []

    effective_output_root = output_root or get_output_root()
    print(f"Dataset annotation videos to process: {target_videos}")

    results: List[Dict[str, Any]] = []
    total_videos = len(target_videos)
    for index, video_id in enumerate(target_videos, start=1):
        print(f"Dataset annotation progress: {index}/{total_videos} | {video_id} | starting")
        result = process_video(
            video_id=video_id,
            rows=rows_by_video.get(video_id, []),
            tracking_video_result=tracking_by_video.get(video_id),
            output_root=effective_output_root,
            force_recompute=force_recompute,
        )
        results.append(result)
        cache_tag = "cached" if bool(result.get("from_cache", False)) else "recomputed"
        print(f"Dataset annotation progress: {index}/{total_videos} | {video_id} | {cache_tag}")

    manifest = {
        "version": _DATASET_ANNOTATIONS_VERSION,
        "source": str(labels_csv),
        "num_videos": len(results),
        "num_frames_total": sum(r["num_frames"] for r in results),
        "num_tracks_total": sum(r["num_tracks"] for r in results),
        "num_candidate_tracks_total": sum(r.get("num_candidate_tracks", 0) for r in results),
        "num_candidate_objects_total": sum(r.get("num_candidate_objects", 0) for r in results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r["num_frames"],
                "num_tracks": r["num_tracks"],
                "num_candidate_tracks": r.get("num_candidate_tracks", 0),
                "num_candidate_objects": r.get("num_candidate_objects", 0),
            }
            for r in results
        ],
    }
    manifest_path = effective_output_root / "dataset_annotations_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Saved dataset annotation manifest to {manifest_path}")
    return results
