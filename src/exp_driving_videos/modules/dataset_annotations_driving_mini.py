"""
Load dataset-provided object annotations (ground-truth tracks) for driving_mini.

Source:
    dataset/driving_mini/labels.csv

The CSV includes per-object rows with fields such as:
    videoName, frameIndex, id, category, box2d.x1, box2d.x2, box2d.y1, box2d.y2

Output layout:
    pipeline_output/driving_mini_dataset_annotations/
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


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def get_labels_csv() -> Path:
    return PROJECT_ROOT / "dataset" / "driving_mini" / "labels.csv"


def get_frames_root() -> Path:
    return PROJECT_ROOT / "dataset" / "driving_mini" / "frames"


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "driving_mini_dataset_annotations"
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


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------


def process_video(
    video_id: str,
    rows: List[Dict[str, str]],
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dataset_annotations.json"

    if not force_recompute and out_file.exists():
        print(f"  [cache] {video_id} - loading {out_file.name}")
        with out_file.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    frame_dir = get_frames_root() / video_id
    available_frames = {
        int(frame_path.stem.split("_")[-1]): frame_path
        for frame_path in sorted(frame_dir.glob("frame_*.jpg"))
    }

    frame_map: Dict[int, Dict[str, Any]] = {}
    all_track_ids: set[int] = set()
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

        if track_id >= 0:
            all_track_ids.add(track_id)

    frames = [frame_map[k] for k in sorted(frame_map.keys())]
    result: Dict[str, Any] = {
        "video_id": video_id,
        "source": "dataset/driving_mini/labels.csv",
        "num_frames": len(frames),
        "num_tracks": len(all_track_ids),
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
    for video_id in target_videos:
        result = process_video(
            video_id=video_id,
            rows=rows_by_video.get(video_id, []),
            output_root=effective_output_root,
            force_recompute=force_recompute,
        )
        results.append(result)

    manifest = {
        "source": str(labels_csv),
        "num_videos": len(results),
        "num_frames_total": sum(r["num_frames"] for r in results),
        "num_tracks_total": sum(r["num_tracks"] for r in results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r["num_frames"],
                "num_tracks": r["num_tracks"],
            }
            for r in results
        ],
    }
    manifest_path = effective_output_root / "dataset_annotations_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Saved dataset annotation manifest to {manifest_path}")
    return results
