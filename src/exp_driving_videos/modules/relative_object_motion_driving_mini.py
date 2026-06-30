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


_RELATIVE_OBJECT_MOTION_VERSION = 4
_REL_VZ_THRESHOLD = 0.2
_REL_VX_THRESHOLD = 0.2
_REL_SPEED_THRESHOLD = 0.3
_DISTANCE_NEAR_THRESHOLD = 15.0
_DISTANCE_MEDIUM_THRESHOLD = 30.0
_X_POSITION_THRESHOLD = 2.0
_VISIBILITY_PERSISTENT_THRESHOLD = 0.8
_VISIBILITY_PRESENT_THRESHOLD = 0.3


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


def _distance_state(distance_meters: float) -> str:
    z = float(distance_meters)
    if z <= _DISTANCE_NEAR_THRESHOLD:
        return "near"
    if z <= _DISTANCE_MEDIUM_THRESHOLD:
        return "medium"
    return "far"


def _position_x_state(x_meters: float) -> str:
    x = float(x_meters)
    if x < -_X_POSITION_THRESHOLD:
        return "left_of_ego"
    if x > _X_POSITION_THRESHOLD:
        return "right_of_ego"
    return "centered"


def _instantaneous_vz_state(rel_vz: float, has_rel_motion: bool) -> str:
    if not has_rel_motion:
        return "vz_unknown"
    if rel_vz < -_REL_VZ_THRESHOLD:
        return "vz_approaching"
    if rel_vz > _REL_VZ_THRESHOLD:
        return "vz_awaying"
    return "vz_stable"


def _instantaneous_vx_state(rel_vx: float, has_rel_motion: bool) -> str:
    if not has_rel_motion:
        return "vx_unknown"
    if rel_vx < -_REL_VX_THRESHOLD:
        return "vx_turning_left"
    if rel_vx > _REL_VX_THRESHOLD:
        return "vx_turning_right"
    return "vx_stable"


def _speed_state(rel_speed: float, has_rel_motion: bool) -> str:
    if not has_rel_motion:
        return "speed_unknown"
    return "rel_moving" if float(rel_speed) > _REL_SPEED_THRESHOLD else "rel_static"


def _visibility_state(track_quality: Dict[str, Any]) -> str:
    ratio = float(dict(track_quality).get("temporal_consistency", 0.0))
    if ratio >= _VISIBILITY_PERSISTENT_THRESHOLD:
        return "persistent"
    if ratio >= _VISIBILITY_PRESENT_THRESHOLD:
        return "intermittent"
    return "brief"


def _relative_motion_entry(
    *,
    track_key: int,
    label: str,
    box: List[float],
    position_3d: List[float],
    ego_vx: float,
    ego_vz: float,
    prev_state: Dict[int, Tuple[int, Tuple[float, float, float]]],
    frame_index: int,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    x, y, z = [float(v) for v in position_3d]
    obj_vx = 0.0
    obj_vz = 0.0
    rel_vx = 0.0
    rel_vz = 0.0
    rel_speed = 0.0
    has_rel_motion = False
    if track_key in prev_state:
        prev_frame_index, (px, py, pz) = prev_state[track_key]
        d_frame = max(1, frame_index - prev_frame_index)
        obj_vx = (x - px) / float(d_frame)
        obj_vz = (z - pz) / float(d_frame)
        rel_vx = obj_vx - ego_vx
        rel_vz = obj_vz - ego_vz
        rel_speed = float(np.hypot(rel_vx, rel_vz))
        has_rel_motion = True
    prev_state[track_key] = (frame_index, (x, y, z))
    return {
        "track_id": int(track_key),
        "label": str(label),
        "box": list(box),
        "position_3d": [x, y, z],
        "obj_vx": float(obj_vx),
        "obj_vz": float(obj_vz),
        "ego_vx": float(ego_vx),
        "ego_vz": float(ego_vz),
        "rel_vx": float(rel_vx),
        "rel_vz": float(rel_vz),
        "rel_speed": float(rel_speed),
        "has_rel_motion": has_rel_motion,
        **(extra_fields or {}),
    }


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
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _RELATIVE_OBJECT_MOTION_VERSION:
            return cached

    ego_by_frame = _build_ego_frame_map(ego_video_result)

    frames_out: List[Dict[str, Any]] = []
    # track_id -> (frame_index, (x, y, z))
    prev_track_state: Dict[int, Tuple[int, Tuple[float, float, float]]] = {}
    prev_candidate_track_state: Dict[int, Tuple[int, Tuple[float, float, float]]] = {}

    for idx, frame in enumerate(positions_3d_video_result.get("frames", [])):
        frame_index = int(frame.get("frame_index", idx))
        image_path = frame.get("image_path", "")

        boxes = frame.get("boxes", [])
        labels = frame.get("labels", [])
        track_ids = frame.get("track_ids", [])
        positions_3d = frame.get("positions_3d", [])
        candidate_objects_in = [dict(obj) for obj in list(frame.get("candidate_objects", []))]

        n = min(len(positions_3d), len(track_ids), len(labels), len(boxes))

        frame_ego = ego_by_frame.get(frame_index, {})
        ego_vx, ego_vz = _ego_vx_vz(frame_ego)

        objects: List[Dict[str, Any]] = []
        for obj_i in range(n):
            track_id = int(track_ids[obj_i])
            label = str(labels[obj_i])
            box = boxes[obj_i]
            objects.append(
                _relative_motion_entry(
                    track_key=track_id,
                    label=label,
                    box=list(box),
                    position_3d=list(positions_3d[obj_i]),
                    ego_vx=ego_vx,
                    ego_vz=ego_vz,
                    prev_state=prev_track_state,
                    frame_index=frame_index,
                    extra_fields={
                        "accepted": True,
                        "source_type": "accepted_object",
                    },
                )
            )

        candidate_objects: List[Dict[str, Any]] = []
        for candidate_object in candidate_objects_in:
            candidate_track_id = int(candidate_object.get("candidate_track_id", candidate_object.get("track_id", -1)))
            position_3d = list(candidate_object.get("position_3d", []))
            if len(position_3d) < 3:
                continue
            track_quality = dict(candidate_object.get("track_quality", {}))
            motion_entry = _relative_motion_entry(
                track_key=candidate_track_id,
                label=str(candidate_object.get("label", "unknown")),
                box=list(candidate_object.get("bbox", [])),
                position_3d=position_3d,
                ego_vx=ego_vx,
                ego_vz=ego_vz,
                prev_state=prev_candidate_track_state,
                frame_index=frame_index,
                extra_fields={
                    "accepted": False,
                    "source_type": str(candidate_object.get("source_type", "selected_candidate_track")),
                    "candidate_object_id": str(
                        candidate_object.get(
                            "candidate_object_id",
                            f"candidate_track_{candidate_track_id}_frame_{frame_index:05d}",
                        )
                    ),
                    "candidate_track_id": candidate_track_id,
                    "source_detection_ids": list(candidate_object.get("source_detection_ids", [])),
                    "candidate_source": str(candidate_object.get("candidate_source", "")),
                    "prior_metadata": dict(candidate_object.get("prior_metadata", {})),
                    "score_breakdown": dict(candidate_object.get("score_breakdown", {})),
                    "selection_score": float(
                        dict(candidate_object.get("score_breakdown", {})).get(
                            "selection_score",
                            track_quality.get("selection_score", 0.0),
                        )
                    ),
                    "track_quality": track_quality,
                    "has_3d_position": bool(candidate_object.get("has_3d_position", True)),
                    "position_3d_provenance": dict(candidate_object.get("position_3d_provenance", {})),
                },
            )
            x_pos = float(motion_entry["position_3d"][0]) if len(motion_entry["position_3d"]) > 0 else 0.0
            z_pos = float(motion_entry["position_3d"][2]) if len(motion_entry["position_3d"]) > 2 else 0.0
            has_rel_motion = bool(motion_entry.get("has_rel_motion", False))
            rel_vx = float(motion_entry.get("rel_vx", 0.0))
            rel_vz = float(motion_entry.get("rel_vz", 0.0))
            rel_speed = float(motion_entry.get("rel_speed", 0.0))
            distance_state = _distance_state(z_pos)
            vx_state = _instantaneous_vx_state(rel_vx, has_rel_motion)
            vz_state = _instantaneous_vz_state(rel_vz, has_rel_motion)
            speed_state = _speed_state(rel_speed, has_rel_motion)
            visibility_state = _visibility_state(track_quality)
            motion_entry.update(
                {
                    "relative_position_3d": list(motion_entry.get("position_3d", [])),
                    "distance_meters": z_pos,
                    "distance_state": distance_state,
                    "x_position_state": _position_x_state(x_pos),
                    "vx_state": vx_state,
                    "vz_state": vz_state,
                    "speed_state": speed_state,
                    "visibility_state": visibility_state,
                    "motion_state": "observed_with_rel_motion" if has_rel_motion else "observed_without_rel_motion",
                    "segment_ready_motion_features": {
                        "relative_position_3d": list(motion_entry.get("position_3d", [])),
                        "distance_meters": z_pos,
                        "distance_state": distance_state,
                        "x_position_state": _position_x_state(x_pos),
                        "vx_state": vx_state,
                        "vz_state": vz_state,
                        "speed_state": speed_state,
                        "visibility_state": visibility_state,
                        "has_rel_motion": has_rel_motion,
                        "has_3d_position": bool(motion_entry.get("has_3d_position", False)),
                    },
                }
            )
            candidate_objects.append(motion_entry)

        frames_out.append(
            {
                "frame_index": frame_index,
                "image_path": image_path,
                "num_objects": len(objects),
                "num_candidate_objects": len(candidate_objects),
                "objects": objects,
                "candidate_objects": candidate_objects,
            }
        )

    result: Dict[str, Any] = {
        "version": _RELATIVE_OBJECT_MOTION_VERSION,
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
        "num_frames_with_candidate_objects": sum(1 for f in frames_out if f["num_candidate_objects"] > 0),
        "num_candidate_objects_total": sum(f["num_candidate_objects"] for f in frames_out),
        "num_candidate_objects_with_rel_motion": sum(
            1
            for f in frames_out
            for obj in f.get("candidate_objects", [])
            if obj.get("has_rel_motion", False)
        ),
        "frames": frames_out,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(
        f"  {video_id}: {result['num_frames']} frames, "
        f"{result['num_objects_with_rel_motion']} object motions relative to ego, "
        f"{result['num_candidate_objects_with_rel_motion']} candidate object motions"
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
        "version": _RELATIVE_OBJECT_MOTION_VERSION,
        "num_videos": len(results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r.get("num_frames", 0),
                "num_objects_total": r.get("num_objects_total", 0),
                "num_objects_with_rel_motion": r.get("num_objects_with_rel_motion", 0),
                "num_candidate_objects_total": r.get("num_candidate_objects_total", 0),
                "num_candidate_objects_with_rel_motion": r.get("num_candidate_objects_with_rel_motion", 0),
            }
            for r in results
        ],
    }
    manifest_path = out_root / "relative_object_motion_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Relative object motion manifest written to {manifest_path}")

    return results
