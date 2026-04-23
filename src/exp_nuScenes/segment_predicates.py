"""
Predicate extraction for nuScenes ego-motion segments.

For now this module keeps the predicate set intentionally small:
1. Ego moving state
2. Ego turning state
3. Object moving state
4. Object turning state
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _state(label: str, confidence: float) -> Dict[str, Any]:
    return {
        "label": str(label),
        "confidence": float(max(0.0, min(1.0, confidence))),
    }


def ego_is_moving(segment: Dict[str, Any]) -> Dict[str, Any]:
    forward_label = segment.get("forward_label_name", "stable")
    if forward_label == "stop":
        return _state("stopped", 1.0)
    return _state("moving", 1.0)


def ego_is_turning(segment: Dict[str, Any]) -> Dict[str, Any]:
    lateral_label = segment.get("lateral_label_name", "stable")
    if lateral_label in {"left", "right"}:
        return _state("turning", 1.0)
    return _state("not_turning", 1.0)


def obj_is_moving(
    obj: Dict[str, Any],
    speed_threshold_mps: float = 0.5,
) -> Dict[str, Any]:
    speed_values = [
        float(value)
        for value in obj.get("ego_relative_speed_mps", [])
        if value is not None
    ]
    if not speed_values:
        return _state("unknown", 0.0)

    moving_ratio = float(np.mean(np.asarray(speed_values) > speed_threshold_mps))
    stopped_ratio = 1.0 - moving_ratio
    if moving_ratio >= 0.7:
        return _state("moving", moving_ratio)
    if stopped_ratio >= 0.7:
        return _state("stopped", stopped_ratio)
    return _state("uncertain", abs(moving_ratio - stopped_ratio))


def _planar_headings(
    velocity_series: Sequence[Optional[Sequence[float]]],
    speed_threshold_mps: float,
) -> np.ndarray:
    headings: List[float] = []
    for vector in velocity_series:
        if vector is None:
            continue
        arr = np.asarray(vector, dtype=np.float64)
        planar_speed = float(np.linalg.norm(arr[:2]))
        if planar_speed <= speed_threshold_mps:
            continue
        headings.append(float(math.atan2(arr[1], arr[0])))
    return np.asarray(headings, dtype=np.float64)


def obj_is_turning(
    obj: Dict[str, Any],
    speed_threshold_mps: float = 0.5,
    heading_change_threshold_rad: float = 0.12,
) -> Dict[str, Any]:
    velocity_series = obj.get("motion_ego_relative_velocity_mps", [])
    headings = _planar_headings(velocity_series, speed_threshold_mps=speed_threshold_mps)
    if headings.size < 2:
        moving_state = obj_is_moving(obj, speed_threshold_mps=speed_threshold_mps)
        if moving_state["label"] == "stopped":
            return _state("not_turning", moving_state["confidence"])
        return _state("unknown", 0.0)

    heading_diffs = np.diff(np.unwrap(headings))
    turning_ratio = float(np.mean(np.abs(heading_diffs) > heading_change_threshold_rad))
    not_turning_ratio = 1.0 - turning_ratio
    if turning_ratio >= 0.5:
        return _state("turning", turning_ratio)
    if not_turning_ratio >= 0.7:
        return _state("not_turning", not_turning_ratio)
    return _state("uncertain", abs(turning_ratio - not_turning_ratio))


def obj_relative_motion(
    obj: Dict[str, Any],
    distance_change_threshold_m: float = 0.25,
    ratio_threshold: float = 0.6,
) -> Dict[str, Any]:
    distance_values = [
        float(value)
        for value in obj.get("relative_distance_to_ego_m", [])
        if value is not None
    ]
    if len(distance_values) < 2:
        return _state("unknown", 0.0)

    distance_diffs = np.diff(np.asarray(distance_values, dtype=np.float64))
    approaching_ratio = float(np.mean(distance_diffs < -distance_change_threshold_m))
    moving_away_ratio = float(np.mean(distance_diffs > distance_change_threshold_m))
    stable_ratio = float(
        np.mean(np.abs(distance_diffs) <= distance_change_threshold_m)
    )

    if approaching_ratio >= ratio_threshold:
        return _state("approaching", approaching_ratio)
    if moving_away_ratio >= ratio_threshold:
        return _state("moving_away", moving_away_ratio)
    if stable_ratio >= ratio_threshold:
        return _state("stable", stable_ratio)

    total_change = float(distance_values[-1] - distance_values[0])
    if total_change < -distance_change_threshold_m:
        return _state("approaching", max(approaching_ratio, 0.5))
    if total_change > distance_change_threshold_m:
        return _state("moving_away", max(moving_away_ratio, 0.5))
    return _state("stable", max(stable_ratio, 0.5))


def obj_lateral_motion(
    obj: Dict[str, Any],
    lateral_velocity_threshold_mps: float = 0.2,
    lateral_position_change_threshold_m: float = 0.3,
    ratio_threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Classify object lateral motion in ego coordinates.

    nuScenes ego coordinates use:
      - x: forward
      - y: left
      - z: up

    So positive y trend means the object moves left relative to ego, and
    negative y trend means it moves right.
    """
    lateral_velocity_values = []
    for vector in obj.get("motion_ego_relative_velocity_mps", []):
        if vector is None:
            continue
        arr = np.asarray(vector, dtype=np.float64)
        lateral_velocity_values.append(float(arr[1]))

    if lateral_velocity_values:
        lateral_velocity = np.asarray(lateral_velocity_values, dtype=np.float64)
        left_ratio = float(np.mean(lateral_velocity > lateral_velocity_threshold_mps))
        right_ratio = float(np.mean(lateral_velocity < -lateral_velocity_threshold_mps))
        stable_ratio = float(np.mean(np.abs(lateral_velocity) <= lateral_velocity_threshold_mps))

        if left_ratio >= ratio_threshold:
            return _state("left", left_ratio)
        if right_ratio >= ratio_threshold:
            return _state("right", right_ratio)
        if stable_ratio >= ratio_threshold:
            return _state("stable", stable_ratio)

    lateral_positions = []
    for position in obj.get("ego_positions_m", []):
        if position is None:
            continue
        arr = np.asarray(position, dtype=np.float64)
        lateral_positions.append(float(arr[1]))

    if len(lateral_positions) < 2:
        return _state("unknown", 0.0)

    total_change = float(lateral_positions[-1] - lateral_positions[0])
    if total_change > lateral_position_change_threshold_m:
        return _state("left", 0.5)
    if total_change < -lateral_position_change_threshold_m:
        return _state("right", 0.5)
    return _state("stable", 0.5)


def extract_segment_predicates(
    segment_objects: Dict[str, Any],
    object_moving_speed_threshold_mps: float = 0.5,
    object_turn_heading_change_threshold_rad: float = 0.12,
    object_relative_distance_change_threshold_m: float = 0.25,
    object_lateral_velocity_threshold_mps: float = 0.2,
    object_lateral_position_change_threshold_m: float = 0.3,
) -> Dict[str, Any]:
    """Extract per-segment ego and object predicates."""
    results: List[Dict[str, Any]] = []
    for segment in segment_objects.get("segments", []):
        ego_predicates = {
            "ego_is_moving": ego_is_moving(segment),
            "ego_is_turning": ego_is_turning(segment),
        }

        object_predicates = []
        for obj in segment.get("objects", []):
            object_predicates.append(
                {
                    "id": obj["id"],
                    "instance_token": obj["instance_token"],
                    "category_name": obj["category_name"],
                    "nearest_distance_rank": obj.get("nearest_distance_rank"),
                    "mean_relative_distance_to_ego_m": obj.get("mean_relative_distance_to_ego_m"),
                    "mean_ego_relative_speed_mps": obj.get("mean_ego_relative_speed_mps"),
                    "is_moving": obj_is_moving(
                        obj,
                        speed_threshold_mps=object_moving_speed_threshold_mps,
                    ),
                    "is_turning": obj_is_turning(
                        obj,
                        speed_threshold_mps=object_moving_speed_threshold_mps,
                        heading_change_threshold_rad=object_turn_heading_change_threshold_rad,
                    ),
                    "relative_motion": obj_relative_motion(
                        obj,
                        distance_change_threshold_m=object_relative_distance_change_threshold_m,
                    ),
                    "lateral_motion": obj_lateral_motion(
                        obj,
                        lateral_velocity_threshold_mps=object_lateral_velocity_threshold_mps,
                        lateral_position_change_threshold_m=object_lateral_position_change_threshold_m,
                    ),
                }
            )

        results.append(
            {
                "segment_index": segment["segment_index"],
                "start_frame": segment["start_frame"],
                "end_frame": segment["end_frame"],
                "duration_frames": segment["duration_frames"],
                "combined_label_name": segment["combined_label_name"],
                "ego": ego_predicates,
                "objects": object_predicates,
            }
        )

    return {
        "scene_name": segment_objects.get("scene_name"),
        "frame_count": segment_objects.get("frame_count", 0),
        "num_segments": len(results),
        "segments": results,
        "params": {
            "object_moving_speed_threshold_mps": float(object_moving_speed_threshold_mps),
            "object_turn_heading_change_threshold_rad": float(object_turn_heading_change_threshold_rad),
            "object_relative_distance_change_threshold_m": float(object_relative_distance_change_threshold_m),
            "object_lateral_velocity_threshold_mps": float(object_lateral_velocity_threshold_mps),
            "object_lateral_position_change_threshold_m": float(object_lateral_position_change_threshold_m),
        },
    }


def summarize_segment_predicates(segment_predicates: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact summary of extracted predicates."""
    segments = segment_predicates.get("segments", [])
    ego_moving_counts = {"moving": 0, "stopped": 0, "unknown": 0}
    ego_turning_counts = {"turning": 0, "not_turning": 0, "unknown": 0}
    object_predicate_count = 0
    object_relative_motion_counts = {
        "approaching": 0,
        "moving_away": 0,
        "stable": 0,
        "unknown": 0,
    }
    object_lateral_motion_counts = {
        "left": 0,
        "right": 0,
        "stable": 0,
        "unknown": 0,
    }

    for segment in segments:
        ego_move_label = segment.get("ego", {}).get("ego_is_moving", {}).get("label", "unknown")
        ego_turn_label = segment.get("ego", {}).get("ego_is_turning", {}).get("label", "unknown")
        ego_moving_counts.setdefault(ego_move_label, 0)
        ego_turning_counts.setdefault(ego_turn_label, 0)
        ego_moving_counts[ego_move_label] += 1
        ego_turning_counts[ego_turn_label] += 1
        objects = segment.get("objects", [])
        object_predicate_count += len(objects)
        for obj in objects:
            rel_label = obj.get("relative_motion", {}).get("label", "unknown")
            lateral_label = obj.get("lateral_motion", {}).get("label", "unknown")
            object_relative_motion_counts.setdefault(rel_label, 0)
            object_lateral_motion_counts.setdefault(lateral_label, 0)
            object_relative_motion_counts[rel_label] += 1
            object_lateral_motion_counts[lateral_label] += 1

    return {
        "num_segments": len(segments),
        "num_object_predicates": int(object_predicate_count),
        "ego_moving_counts": ego_moving_counts,
        "ego_turning_counts": ego_turning_counts,
        "object_relative_motion_counts": object_relative_motion_counts,
        "object_lateral_motion_counts": object_lateral_motion_counts,
    }
