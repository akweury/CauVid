"""
Estimate relative-to-ego object motion for driving_mini videos.

Consumes:
  - Step 5 output: per-object 3D positions per frame
  - Step 6 output: per-frame ego motion (smoothed fields preferred)

For each tracked object (track_id):
  1. Compute object motion from consecutive 3D positions.
  2. Subtract ego motion to obtain relative-to-ego motion.

Output layout:
    pipeline_output/07_driving_mini_relative_object_motion/
        relative_object_motion_manifest.json
        <video_id>/
            relative_object_motion.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "07_driving_mini_relative_object_motion"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_ego_frame_map(ego_video_result: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {
        int(frame.get("frame_index", idx)): frame
        for idx, frame in enumerate(ego_video_result.get("frames", []))
    }


def _ego_vx_vz(frame_ego: Dict[str, Any]) -> Tuple[float, float]:
    """Read smoothed ego motion if present, otherwise raw values."""
    vx = float(frame_ego.get("ego_vx_smoothed", frame_ego.get("ego_vx", 0.0)))
    vz = float(frame_ego.get("ego_vz_smoothed", frame_ego.get("ego_vz", 0.0)))
    return vx, vz


def process_video(
    positions_3d_video_result: Dict[str, Any],
    ego_video_result: Dict[str, Any],
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    video_id = positions_3d_video_result["video_id"]
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "relative_object_motion.json"

    if not force_recompute and out_file.exists():
        print(f"  [cache] {video_id} - loading {out_file.name}")
        with out_file.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    ego_by_frame = _build_ego_frame_map(ego_video_result)

    frames_out: List[Dict[str, Any]] = []
    # track_id -> (frame_index, (x, y, z))
    prev_track_state: Dict[int, Tuple[int, Tuple[float, float, float]]] = {}

    for idx, frame in enumerate(positions_3d_video_result.get("frames", [])):
        frame_index = int(frame.get("frame_index", idx))
        image_path = frame.get("image_path", "")

        boxes = frame.get("boxes", [])
        labels = frame.get("labels", [])
        track_ids = frame.get("track_ids", [])
        positions_3d = frame.get("positions_3d", [])

        n = min(len(positions_3d), len(track_ids), len(labels), len(boxes))

        frame_ego = ego_by_frame.get(frame_index, {})
        ego_vx, ego_vz = _ego_vx_vz(frame_ego)

        objects: List[Dict[str, Any]] = []
        for obj_i in range(n):
            track_id = int(track_ids[obj_i])
            label = str(labels[obj_i])
            box = boxes[obj_i]
            x, y, z = [float(v) for v in positions_3d[obj_i]]

            obj_vx = 0.0
            obj_vz = 0.0
            rel_vx = 0.0
            rel_vz = 0.0
            rel_speed = 0.0
            has_rel_motion = False

            if track_id in prev_track_state:
                prev_frame_index, (px, py, pz) = prev_track_state[track_id]
                d_frame = max(1, frame_index - prev_frame_index)
                obj_vx = (x - px) / float(d_frame)
                obj_vz = (z - pz) / float(d_frame)

                rel_vx = obj_vx - ego_vx
                rel_vz = obj_vz - ego_vz
                rel_speed = float(np.hypot(rel_vx, rel_vz))
                has_rel_motion = True

            objects.append(
                {
                    "track_id": track_id,
                    "label": label,
                    "box": box,
                    "position_3d": [x, y, z],
                    "obj_vx": float(obj_vx),
                    "obj_vz": float(obj_vz),
                    "ego_vx": float(ego_vx),
                    "ego_vz": float(ego_vz),
                    "rel_vx": float(rel_vx),
                    "rel_vz": float(rel_vz),
                    "rel_speed": float(rel_speed),
                    "has_rel_motion": has_rel_motion,
                }
            )

            prev_track_state[track_id] = (frame_index, (x, y, z))

        frames_out.append(
            {
                "frame_index": frame_index,
                "image_path": image_path,
                "num_objects": len(objects),
                "objects": objects,
            }
        )

    result: Dict[str, Any] = {
        "video_id": video_id,
        "num_frames": len(frames_out),
        "num_frames_with_objects": sum(1 for f in frames_out if f["num_objects"] > 0),
        "num_objects_total": sum(f["num_objects"] for f in frames_out),
        "num_objects_with_rel_motion": sum(
            1
            for f in frames_out
            for obj in f.get("objects", [])
            if obj.get("has_rel_motion", False)
        ),
        "frames": frames_out,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(
        f"  {video_id}: {result['num_frames']} frames, "
        f"{result['num_objects_with_rel_motion']} object motions relative to ego"
    )
    return result


def run(
    positions_3d_results: List[Dict[str, Any]],
    ego_motion_results: List[Dict[str, Any]],
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    out_root = output_root or get_output_root()

    ego_by_video: Dict[str, Dict[str, Any]] = {
        r.get("video_id", ""): r for r in ego_motion_results
    }

    results: List[Dict[str, Any]] = []
    for pos_result in positions_3d_results:
        video_id = pos_result.get("video_id", "unknown")
        ego_result = ego_by_video.get(video_id)
        if ego_result is None:
            print(f"  [warn] Missing ego motion for video {video_id}; skipping.")
            continue

        print(f"  Processing relative object motion: {video_id}")
        result = process_video(
            positions_3d_video_result=pos_result,
            ego_video_result=ego_result,
            output_root=out_root,
            force_recompute=force_recompute,
        )
        results.append(result)

    manifest = {
        "num_videos": len(results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r.get("num_frames", 0),
                "num_objects_total": r.get("num_objects_total", 0),
                "num_objects_with_rel_motion": r.get("num_objects_with_rel_motion", 0),
            }
            for r in results
        ],
    }
    manifest_path = out_root / "relative_object_motion_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Relative object motion manifest written to {manifest_path}")

    return results
