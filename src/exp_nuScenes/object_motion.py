"""
Object motion extraction for nuScenes scenes.

This module turns per-frame nuScenes annotations into object-centric motion
tracks spanning the full scene timeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.exp_nuScenes.ego_motion import DEFAULT_CHANNEL_PRIORITY, quaternion_to_rotation_matrix


def _select_sample_media(sample: Dict[str, Any], channel: Optional[str] = None) -> Dict[str, Any]:
    media = sample.get("media") or {}
    if not media:
        raise ValueError(f"Sample {sample.get('sample_token', '<unknown>')} has no media records.")

    if channel is not None:
        channel = channel.upper()
        if channel in media:
            return media[channel]

    for preferred_channel in DEFAULT_CHANNEL_PRIORITY:
        if preferred_channel in media:
            return media[preferred_channel]
    return media[sorted(media)[0]]


def _sample_ego_pose(
    sample: Dict[str, Any],
    channel: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    media_record = _select_sample_media(sample, channel=channel)
    ego_pose = media_record.get("ego_pose")
    if not ego_pose:
        raise ValueError(
            f"Media record {media_record.get('sample_data_token', '<unknown>')} "
            f"for sample {sample.get('sample_token', '<unknown>')} has no ego_pose."
        )
    return {
        "translation": np.asarray(ego_pose["translation"], dtype=np.float64),
        "rotation": np.asarray(ego_pose["rotation"], dtype=np.float64),
    }


def _vector_to_json(vector: Optional[np.ndarray]) -> Optional[List[float]]:
    if vector is None:
        return None
    return [float(value) for value in vector.tolist()]


def _vector_series_to_json(vectors: Sequence[Optional[np.ndarray]]) -> List[Optional[List[float]]]:
    return [_vector_to_json(vector) for vector in vectors]


def extract_object_motion_3d(
    scene_samples: Sequence[Dict[str, Any]],
    ego_motion: Optional[Dict[str, Any]] = None,
    channel: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract object-centric 3D motion tracks across one ordered nuScenes scene.

    Returns a timeline-aligned track for each annotated instance. Missing frames
    are preserved so every track spans frame 0 to the last frame of the scene.
    """
    if not scene_samples:
        return {
            "scene_name": None,
            "frame_count": 0,
            "timestamps_us": [],
            "sample_tokens": [],
            "tracks": [],
        }

    ordered_samples = sorted(scene_samples, key=lambda sample: sample["timestamp"])
    frame_count = len(ordered_samples)
    scene_name = ordered_samples[0].get("scene_name")
    timestamps_us = np.asarray([sample["timestamp"] for sample in ordered_samples], dtype=np.float64)
    sample_tokens = [sample["sample_token"] for sample in ordered_samples]

    ego_translations = []
    ego_rotations = []
    ego_global_velocity = None
    if ego_motion is not None:
        ego_global_velocity = np.asarray(ego_motion.get("global_velocity_mps", []), dtype=np.float64)
        if ego_global_velocity.ndim != 2 or ego_global_velocity.shape[1] != 3:
            ego_global_velocity = None

    for sample in ordered_samples:
        pose = _sample_ego_pose(sample, channel=channel)
        ego_translations.append(pose["translation"])
        ego_rotations.append(pose["rotation"])

    ego_translations = np.vstack(ego_translations)

    tracks_by_instance: Dict[str, Dict[str, Any]] = {}

    for frame_index, sample in enumerate(ordered_samples):
        for annotation in sample.get("annotations", []):
            instance_token = annotation["instance_token"]
            track = tracks_by_instance.get(instance_token)
            if track is None:
                track = {
                    "instance_token": instance_token,
                    "category_name": annotation["category_name"],
                    "top_category": annotation.get("top_category"),
                    "annotation_tokens": [None] * frame_count,
                    "present_mask": [False] * frame_count,
                    "attribute_names_by_frame": [None] * frame_count,
                    "visibility_by_frame": [None] * frame_count,
                    "size_by_frame": [None] * frame_count,
                    "global_positions": [None] * frame_count,
                    "ego_positions": [None] * frame_count,
                    "global_velocity": [None] * frame_count,
                    "ego_relative_velocity": [None] * frame_count,
                }
                tracks_by_instance[instance_token] = track

            global_position = np.asarray(annotation["translation"], dtype=np.float64)
            rotation_matrix = quaternion_to_rotation_matrix(ego_rotations[frame_index])
            ego_position = rotation_matrix.T @ (global_position - ego_translations[frame_index])

            track["annotation_tokens"][frame_index] = annotation["annotation_token"]
            track["present_mask"][frame_index] = True
            track["attribute_names_by_frame"][frame_index] = list(annotation.get("attribute_names", []))
            visibility_level = annotation.get("visibility_level", "")
            visibility_description = annotation.get("visibility_description", "")
            track["visibility_by_frame"][frame_index] = {
                "level": visibility_level,
                "description": visibility_description,
            }
            track["size_by_frame"][frame_index] = [float(value) for value in annotation.get("size", [])]
            track["global_positions"][frame_index] = global_position
            track["ego_positions"][frame_index] = ego_position

    tracks = []
    for track in sorted(tracks_by_instance.values(), key=lambda item: (item["category_name"], item["instance_token"])):
        present_indices = [index for index, is_present in enumerate(track["present_mask"]) if is_present]
        if not present_indices:
            continue

        for index_in_track, frame_index in enumerate(present_indices[1:], start=1):
            prev_frame_index = present_indices[index_in_track - 1]
            prev_position = track["global_positions"][prev_frame_index]
            curr_position = track["global_positions"][frame_index]
            delta_t_s = float(timestamps_us[frame_index] - timestamps_us[prev_frame_index]) / 1_000_000.0
            if delta_t_s <= 0:
                velocity = np.zeros(3, dtype=np.float64)
            else:
                velocity = (curr_position - prev_position) / delta_t_s
            track["global_velocity"][frame_index] = velocity

        if len(present_indices) > 1:
            first_idx = present_indices[0]
            second_idx = present_indices[1]
            track["global_velocity"][first_idx] = np.asarray(track["global_velocity"][second_idx], dtype=np.float64)

        for frame_index in present_indices:
            velocity = track["global_velocity"][frame_index]
            if velocity is None:
                continue
            ego_rotation_matrix = quaternion_to_rotation_matrix(ego_rotations[frame_index])
            relative_global_velocity = velocity.copy()
            if ego_global_velocity is not None and frame_index < ego_global_velocity.shape[0]:
                relative_global_velocity = relative_global_velocity - ego_global_velocity[frame_index]
            track["ego_relative_velocity"][frame_index] = ego_rotation_matrix.T @ relative_global_velocity

        global_speed = [
            float(np.linalg.norm(vector)) if vector is not None else None
            for vector in track["global_velocity"]
        ]
        ego_relative_speed = [
            float(np.linalg.norm(vector)) if vector is not None else None
            for vector in track["ego_relative_velocity"]
        ]

        tracks.append(
            {
                "instance_token": track["instance_token"],
                "category_name": track["category_name"],
                "top_category": track["top_category"],
                "first_frame": int(present_indices[0]),
                "last_frame": int(present_indices[-1]),
                "num_observed_frames": len(present_indices),
                "present_frame_indices": [int(index) for index in present_indices],
                "annotation_tokens": track["annotation_tokens"],
                "present_mask": [bool(value) for value in track["present_mask"]],
                "attribute_names_by_frame": track["attribute_names_by_frame"],
                "visibility_by_frame": track["visibility_by_frame"],
                "size_by_frame": track["size_by_frame"],
                "global_positions_m": _vector_series_to_json(track["global_positions"]),
                "ego_positions_m": _vector_series_to_json(track["ego_positions"]),
                "global_velocity_mps": _vector_series_to_json(track["global_velocity"]),
                "ego_relative_velocity_mps": _vector_series_to_json(track["ego_relative_velocity"]),
                "global_speed_mps": global_speed,
                "ego_relative_speed_mps": ego_relative_speed,
            }
        )

    frame_objects = []
    for frame_index, sample in enumerate(ordered_samples):
        objects = []
        for annotation in sample.get("annotations", []):
            objects.append(
                {
                    "instance_token": annotation["instance_token"],
                    "annotation_token": annotation["annotation_token"],
                    "category_name": annotation["category_name"],
                }
            )
        frame_objects.append(
            {
                "frame_index": frame_index,
                "sample_token": sample["sample_token"],
                "timestamp_us": int(sample["timestamp"]),
                "num_objects": len(objects),
                "objects": objects,
            }
        )

    return {
        "scene_name": scene_name,
        "frame_count": frame_count,
        "timestamps_us": timestamps_us.astype(np.int64).tolist(),
        "sample_tokens": sample_tokens,
        "tracks": tracks,
        "frames": frame_objects,
    }


def summarize_object_motion(object_motion: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact summary of extracted object motion tracks."""
    tracks = object_motion.get("tracks", [])
    categories: Dict[str, int] = {}
    observed_frame_counts = []
    full_span_track_count = 0
    frame_count = int(object_motion.get("frame_count", 0))

    for track in tracks:
        category_name = track.get("category_name", "unknown")
        categories[category_name] = categories.get(category_name, 0) + 1
        observed_frames = int(track.get("num_observed_frames", 0))
        observed_frame_counts.append(observed_frames)
        if observed_frames == frame_count and frame_count > 0:
            full_span_track_count += 1

    return {
        "num_tracks": len(tracks),
        "num_categories": len(categories),
        "tracks_per_category": dict(sorted(categories.items())),
        "mean_observed_frames_per_track": float(np.mean(observed_frame_counts)) if observed_frame_counts else 0.0,
        "max_observed_frames_per_track": int(max(observed_frame_counts)) if observed_frame_counts else 0,
        "full_span_track_count": int(full_span_track_count),
    }
