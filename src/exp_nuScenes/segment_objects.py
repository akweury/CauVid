"""
Segment-level object extraction for nuScenes scenes.

This module groups object motion tracks by the ego-motion timeline segments so
later reasoning can work one segment at a time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _slice_present_values(
    values: Sequence[Any],
    frame_indices: Sequence[int],
) -> List[Any]:
    return [values[index] for index in frame_indices]


def _distance_series_from_positions(ego_positions_m: Sequence[Optional[Sequence[float]]]) -> List[Optional[float]]:
    distances: List[Optional[float]] = []
    for position in ego_positions_m:
        if position is None:
            distances.append(None)
            continue
        vector = np.asarray(position, dtype=np.float64)
        distances.append(float(np.linalg.norm(vector)))
    return distances


def _mean_vector(vectors: Sequence[Optional[Sequence[float]]]) -> Optional[List[float]]:
    valid_vectors = [np.asarray(vector, dtype=np.float64) for vector in vectors if vector is not None]
    if not valid_vectors:
        return None
    return [float(value) for value in np.mean(np.vstack(valid_vectors), axis=0).tolist()]


def extract_objects_in_ego_segments(
    ego_motion_segments: Dict[str, Any],
    object_motion: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract per-segment object summaries using ego-motion timeline segments.

    Each segment contains the objects visible inside that frame range, along
    with their motion vectors, speed, and distance-to-ego measurements.
    """
    timeline_segments = ego_motion_segments.get("timeline_segments", [])
    tracks = object_motion.get("tracks", [])
    scene_name = object_motion.get("scene_name")
    frame_count = int(object_motion.get("frame_count", 0))

    segment_records: List[Dict[str, Any]] = []
    for segment_index, timeline_segment in enumerate(timeline_segments):
        start_frame = int(timeline_segment["start_frame"])
        end_frame = int(timeline_segment["end_frame"])
        segment_frame_indices = list(range(start_frame, end_frame + 1))
        objects_in_segment: List[Dict[str, Any]] = []

        for track in tracks:
            present_mask = [bool(value) for value in track.get("present_mask", [])]
            if not present_mask:
                continue

            present_frame_indices = [
                frame_index
                for frame_index in segment_frame_indices
                if frame_index < len(present_mask) and present_mask[frame_index]
            ]
            if not present_frame_indices:
                continue

            ego_relative_velocity = track.get("ego_relative_velocity_mps", [])
            global_velocity = track.get("global_velocity_mps", [])
            ego_relative_speed = track.get("ego_relative_speed_mps", [])
            global_speed = track.get("global_speed_mps", [])
            ego_positions = track.get("ego_positions_m", [])
            relative_distance = _distance_series_from_positions(ego_positions)

            segment_ego_relative_velocity = _slice_present_values(ego_relative_velocity, present_frame_indices)
            segment_global_velocity = _slice_present_values(global_velocity, present_frame_indices)
            segment_ego_relative_speed = _slice_present_values(ego_relative_speed, present_frame_indices)
            segment_global_speed = _slice_present_values(global_speed, present_frame_indices)
            segment_ego_positions = _slice_present_values(ego_positions, present_frame_indices)
            segment_relative_distance = _slice_present_values(relative_distance, present_frame_indices)

            valid_ego_relative_speed = [float(value) for value in segment_ego_relative_speed if value is not None]
            valid_global_speed = [float(value) for value in segment_global_speed if value is not None]
            valid_relative_distance = [float(value) for value in segment_relative_distance if value is not None]

            objects_in_segment.append(
                {
                    "id": track["instance_token"],
                    "instance_token": track["instance_token"],
                    "category_name": track["category_name"],
                    "top_category": track.get("top_category"),
                    "start_frame": present_frame_indices[0],
                    "end_frame": present_frame_indices[-1],
                    "present_frame_indices": present_frame_indices,
                    "num_observed_frames": len(present_frame_indices),
                    "motion_global_velocity_mps": segment_global_velocity,
                    "motion_ego_relative_velocity_mps": segment_ego_relative_velocity,
                    "mean_global_velocity_mps": _mean_vector(segment_global_velocity),
                    "mean_ego_relative_velocity_mps": _mean_vector(segment_ego_relative_velocity),
                    "global_speed_mps": segment_global_speed,
                    "ego_relative_speed_mps": segment_ego_relative_speed,
                    "mean_global_speed_mps": float(np.mean(valid_global_speed)) if valid_global_speed else None,
                    "mean_ego_relative_speed_mps": float(np.mean(valid_ego_relative_speed)) if valid_ego_relative_speed else None,
                    "ego_positions_m": segment_ego_positions,
                    "relative_distance_to_ego_m": segment_relative_distance,
                    "mean_relative_distance_to_ego_m": float(np.mean(valid_relative_distance)) if valid_relative_distance else None,
                    "min_relative_distance_to_ego_m": float(np.min(valid_relative_distance)) if valid_relative_distance else None,
                    "max_relative_distance_to_ego_m": float(np.max(valid_relative_distance)) if valid_relative_distance else None,
                }
            )

        objects_in_segment.sort(
            key=lambda obj: (
                obj["min_relative_distance_to_ego_m"] is None,
                float(obj["min_relative_distance_to_ego_m"]) if obj["min_relative_distance_to_ego_m"] is not None else float("inf"),
                obj["id"],
            )
        )
        for rank_index, obj in enumerate(objects_in_segment, start=1):
            obj["nearest_distance_rank"] = int(rank_index)

        segment_records.append(
            {
                "segment_index": segment_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": int(end_frame - start_frame + 1),
                "forward_label": timeline_segment["forward_label"],
                "forward_label_name": timeline_segment["forward_label_name"],
                "lateral_label": timeline_segment["lateral_label"],
                "lateral_label_name": timeline_segment["lateral_label_name"],
                "combined_label_name": timeline_segment["combined_label_name"],
                "num_objects": len(objects_in_segment),
                "objects": objects_in_segment,
            }
        )

    return {
        "scene_name": scene_name,
        "frame_count": frame_count,
        "num_segments": len(segment_records),
        "segments": segment_records,
    }


def summarize_segment_objects(segment_objects: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact summary for scene-level reporting."""
    segments = segment_objects.get("segments", [])
    object_counts = [int(segment.get("num_objects", 0)) for segment in segments]

    return {
        "num_segments": len(segments),
        "mean_objects_per_segment": float(np.mean(object_counts)) if object_counts else 0.0,
        "max_objects_per_segment": int(max(object_counts)) if object_counts else 0,
        "segment_ranges": [
            [int(segment["start_frame"]), int(segment["end_frame"])]
            for segment in segments
        ],
    }
