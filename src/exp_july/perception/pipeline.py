import copy
import json
import math
import os
import sys
from collections import Counter
from contextlib import redirect_stderr
from contextlib import redirect_stdout
import io
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.exp_driving_videos import pipeline_config as driving_pipeline_config
from src.exp_driving_videos.modules import detect_driving_mini
from src.exp_driving_videos.modules import ego_motion_driving_mini
from src.exp_driving_videos.modules import merge_gt_and_detected_driving_mini
from src.exp_driving_videos.modules import prepare_3d_positions_driving_mini
from src.exp_driving_videos.modules import tracking_driving_mini
from tqdm import tqdm


_TRACKLET_REPAIR_VERSION = 1
_TRACKLET_REPAIR_DEFAULT_CFG = {
    "max_gap_frames": 2,
    "min_endpoint_score": 0.2,
    "max_center_step_fraction_of_diag": 0.55,
    "max_center_step_px": 120.0,
    "max_size_ratio": 1.45,
    "max_velocity_delta_fraction_of_diag": 0.45,
    "max_velocity_delta_px": 90.0,
    "conflict_iou_threshold": 0.25,
}
_RELATIVE_OBJECT_MOTION_VERSION = 1
_REL_VZ_THRESHOLD = 0.2
_REL_VX_THRESHOLD = 0.2
_REL_SPEED_THRESHOLD = 0.3
_DISTANCE_NEAR_THRESHOLD = 15.0
_DISTANCE_MEDIUM_THRESHOLD = 30.0
_X_POSITION_THRESHOLD = 2.0
_RELATIVE_MOTION_VIS_VERSION = 1
_VIS_OBSERVED_COLOR = (70, 220, 70)
_VIS_REPAIRED_COLOR = (220, 60, 255)
_VIS_ABSENT_COLOR = (58, 58, 58)
_VIS_DECISION_COLORS = {
    "Keep": (80, 220, 80),
    "Keep with uncertainty": (80, 200, 255),
    "Repair": (220, 60, 255),
    "Discard": (40, 40, 230),
}
_VIS_EGO_METHOD_COLORS = {
    "original": (220, 220, 220),
    "weighted_median": (255, 190, 40),
    "refined": (80, 220, 80),
    "ransac": (60, 140, 255),
}
_CAUSAL_FILTER_OUT_VERSION = 1
_TRAJECTORY_VALIDATION_THRESHOLDS = {
    "max_valid_frame_gap": 3,
    "max_uncertain_frame_gap": 1,
    "max_invalid_center_step_diag_ratio": 2.0,
    "max_uncertain_center_step_diag_ratio": 1.0,
    "max_invalid_bbox_size_ratio": 3.0,
    "max_uncertain_bbox_size_ratio": 2.0,
    "max_invalid_depth_step_per_frame": 8.0,
    "max_uncertain_depth_step_per_frame": 4.0,
    "max_invalid_rel_velocity_delta": 10.0,
    "max_uncertain_rel_velocity_delta": 5.0,
    "max_invalid_rel_speed": 25.0,
    "max_uncertain_rel_speed": 12.0,
    "min_motion_ratio": 0.5,
}
_MOTION_SIGNIFICANCE_THRESHOLDS = {
    "min_observations": 3,
    "min_has_motion_ratio": 0.6,
    "max_repaired_ratio": 0.5,
    "max_uncertainty_score": 0.55,
    "min_rel_speed_mean": 0.05,
    "min_rel_speed_max": 0.12,
    "min_path_length_xz": 0.25,
    "min_displacement_xz": 0.15,
    "min_depth_abs_delta": 0.15,
    "min_bbox_center_path_px": 3.0,
    "noise_rel_speed": 0.05,
    "noise_position_xz_step": 0.03,
}
_EGO_REFINEMENT_VERSION = 1
_STATIC_OBJECT_PRIOR = {
    "traffic light": "static",
    "traffic_light": "static",
    "traffic sign": "static",
    "traffic_sign": "static",
    "stop sign": "static",
    "stop_sign": "static",
    "sign": "static",
    "pole": "static",
    "utility pole": "static",
    "street light": "static",
    "street_light": "static",
    "building": "static",
    "wall": "static",
    "fence": "static",
    "road": "static",
    "lane": "static",
    "crosswalk": "static",
    "sidewalk": "static",
    "parking meter": "static",
    "parking_meter": "static",
}
_LOW_DYNAMIC_OBJECT_PRIOR = {
    "parked car": "low_dynamic",
    "parked_car": "low_dynamic",
    "car": "low_dynamic",
    "truck": "low_dynamic",
    "bus": "low_dynamic",
    "trailer": "low_dynamic",
}
_REFERENCE_OBJECT_THRESHOLDS = {
    "min_observation_ratio": 0.25,
    "max_uncertainty_score": 0.45,
    "max_repaired_ratio": 0.5,
    "max_rel_speed_mean_static": 0.12,
    "max_rel_speed_mean_low_dynamic": 0.25,
    "max_depth_abs_delta_static": 1.0,
    "max_bbox_center_step_diag_ratio": 0.8,
}


def get_pipeline_output_root():
    return Path(os.environ.get("CAUVID_PIPELINE_OUTPUT_PATH", ROOT / "output_july"))


def normalize_detection_image_paths(video_result, dataset_root):
    video_id = str(video_result.get("video_id", "")).strip()
    if not video_id:
        return video_result, False

    frames_root = Path(dataset_root) / "frames" / video_id
    changed = False
    updated = dict(video_result)
    updated_frames = []
    for frame in video_result.get("frames", []):
        frame_record = dict(frame)
        image_path_text = str(frame_record.get("image_path", "")).strip()
        image_path = Path(image_path_text) if image_path_text else None
        if image_path_text and image_path and not image_path.exists():
            candidate = frames_root / image_path.name
            if candidate.exists():
                frame_record["image_path"] = str(candidate)
                changed = True
        updated_frames.append(frame_record)
    updated["frames"] = updated_frames
    return updated, changed


def write_detection_cache_if_needed(video_result, source_path=None):
    detections_json = str(video_result.get("output_paths", {}).get("detections_json", "")).strip()
    path = Path(source_path) if source_path is not None else (Path(detections_json) if detections_json else None)
    if path is None:
        return
    updated = dict(video_result)
    output_paths = dict(updated.get("output_paths", {}))
    output_paths["detections_json"] = str(path)
    updated["output_paths"] = output_paths
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)


def _safe_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _valid_bbox(box):
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    out = [_safe_float(v) for v in box[:4]]
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def _valid_position_3d(position):
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return None
    out = [_safe_float(v) for v in position[:3]]
    if not all(math.isfinite(v) for v in out):
        return None
    return out


def _bbox_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _bbox_size(box):
    return (box[2] - box[0], box[3] - box[1])


def _bbox_diag(box):
    width, height = _bbox_size(box)
    return math.hypot(width, height)


def _bbox_area(box):
    width, height = _bbox_size(box)
    return max(0.0, width) * max(0.0, height)


def _bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    denom = _bbox_area(box_a) + _bbox_area(box_b) - inter
    return float(inter / denom) if denom > 0.0 else 0.0


def _lerp_list(start, end, alpha):
    return [float((1.0 - alpha) * s + alpha * e) for s, e in zip(start, end)]


def _vector_between(obs_a, obs_b):
    ax, ay = _bbox_center(obs_a["bbox"])
    bx, by = _bbox_center(obs_b["bbox"])
    delta = max(1, int(obs_b["frame_index"]) - int(obs_a["frame_index"]))
    return ((bx - ax) / delta, (by - ay) / delta)


def _vector_delta_norm(vec_a, vec_b):
    return math.hypot(vec_a[0] - vec_b[0], vec_a[1] - vec_b[1])


def _same_label(label_a, label_b):
    return str(label_a).strip().lower() == str(label_b).strip().lower()


def _frame_object_observations(video_result):
    tracks = {}
    duplicate_track_frames = set()
    frame_indices = {}
    for frame_pos, frame in enumerate(video_result.get("frames", [])):
        frame_index = int(frame.get("frame_index", frame_pos))
        frame_indices[frame_index] = frame_pos
        boxes = list(frame.get("boxes", []))
        scores = list(frame.get("scores", []))
        labels = list(frame.get("labels", []))
        track_ids = list(frame.get("track_ids", []))
        positions_3d = list(frame.get("positions_3d", []))
        seen_track_ids = set()
        for obj_idx, track_id_raw in enumerate(track_ids):
            try:
                track_id = int(track_id_raw)
            except (TypeError, ValueError):
                continue
            if track_id < 0:
                continue
            if track_id in seen_track_ids:
                duplicate_track_frames.add((track_id, frame_index))
            seen_track_ids.add(track_id)
            box = _valid_bbox(boxes[obj_idx] if obj_idx < len(boxes) else None)
            if box is None:
                continue
            position_3d = _valid_position_3d(positions_3d[obj_idx] if obj_idx < len(positions_3d) else None)
            label = labels[obj_idx] if obj_idx < len(labels) else "unknown"
            score = _safe_float(scores[obj_idx] if obj_idx < len(scores) else 1.0, 1.0)
            tracks.setdefault(track_id, []).append(
                {
                    "track_id": track_id,
                    "frame_pos": frame_pos,
                    "frame_index": frame_index,
                    "object_index": obj_idx,
                    "bbox": box,
                    "position_3d": position_3d,
                    "label": str(label),
                    "score": score,
                }
            )
    for track_obs in tracks.values():
        track_obs.sort(key=lambda row: (int(row["frame_index"]), int(row["object_index"])))
    return tracks, frame_indices, duplicate_track_frames


def _tracklet_gap_is_safe(track_obs, obs_idx, frame_indices, video_frames, cfg):
    prev_obs = track_obs[obs_idx]
    next_obs = track_obs[obs_idx + 1]
    start = int(prev_obs["frame_index"])
    end = int(next_obs["frame_index"])
    gap = end - start - 1
    if gap <= 0 or gap > int(cfg["max_gap_frames"]):
        return False, "gap_length", []
    missing_indices = list(range(start + 1, end))
    if any(frame_index not in frame_indices for frame_index in missing_indices):
        return False, "missing_frame_record", []
    if not _same_label(prev_obs["label"], next_obs["label"]):
        return False, "label_mismatch", []
    if prev_obs["position_3d"] is None or next_obs["position_3d"] is None:
        return False, "missing_3d_endpoint", []
    if min(prev_obs["score"], next_obs["score"]) < float(cfg["min_endpoint_score"]):
        return False, "low_endpoint_score", []

    prev_width, prev_height = _bbox_size(prev_obs["bbox"])
    next_width, next_height = _bbox_size(next_obs["bbox"])
    width_ratio = max(prev_width, next_width) / max(1e-6, min(prev_width, next_width))
    height_ratio = max(prev_height, next_height) / max(1e-6, min(prev_height, next_height))
    if width_ratio > float(cfg["max_size_ratio"]) or height_ratio > float(cfg["max_size_ratio"]):
        return False, "box_scale_change", []

    prev_center = _bbox_center(prev_obs["bbox"])
    next_center = _bbox_center(next_obs["bbox"])
    center_step = math.hypot(next_center[0] - prev_center[0], next_center[1] - prev_center[1]) / float(gap + 1)
    avg_diag = max(1.0, (_bbox_diag(prev_obs["bbox"]) + _bbox_diag(next_obs["bbox"])) / 2.0)
    max_center_step = min(
        float(cfg["max_center_step_px"]),
        float(cfg["max_center_step_fraction_of_diag"]) * avg_diag,
    )
    if center_step > max_center_step:
        return False, "center_motion_too_large", []

    gap_velocity = _vector_between(prev_obs, next_obs)
    max_velocity_delta = min(
        float(cfg["max_velocity_delta_px"]),
        float(cfg["max_velocity_delta_fraction_of_diag"]) * avg_diag,
    )
    if obs_idx > 0:
        before_obs = track_obs[obs_idx - 1]
        if _same_label(before_obs["label"], prev_obs["label"]):
            prior_velocity = _vector_between(before_obs, prev_obs)
            if _vector_delta_norm(prior_velocity, gap_velocity) > max_velocity_delta:
                return False, "incoming_motion_not_smooth", []
    if obs_idx + 2 < len(track_obs):
        after_obs = track_obs[obs_idx + 2]
        if _same_label(next_obs["label"], after_obs["label"]):
            post_velocity = _vector_between(next_obs, after_obs)
            if _vector_delta_norm(gap_velocity, post_velocity) > max_velocity_delta:
                return False, "outgoing_motion_not_smooth", []

    proposed = []
    for offset, frame_index in enumerate(missing_indices, start=1):
        alpha = offset / float(gap + 1)
        interp_box = _lerp_list(prev_obs["bbox"], next_obs["bbox"], alpha)
        interp_position_3d = _lerp_list(prev_obs["position_3d"], next_obs["position_3d"], alpha)
        frame = video_frames[frame_indices[frame_index]]
        frame_track_ids = set()
        for track_id_raw in frame.get("track_ids", []):
            try:
                frame_track_ids.add(int(track_id_raw))
            except (TypeError, ValueError):
                continue
        if int(prev_obs["track_id"]) in frame_track_ids:
            return False, "track_already_present", []
        for existing_box_raw in frame.get("boxes", []):
            existing_box = _valid_bbox(existing_box_raw)
            if existing_box is None:
                continue
            if _bbox_iou(interp_box, existing_box) >= float(cfg["conflict_iou_threshold"]):
                return False, "overlap_conflict", []
        proposed.append(
            {
                "frame_index": frame_index,
                "frame_pos": frame_indices[frame_index],
                "track_id": int(prev_obs["track_id"]),
                "label": str(prev_obs["label"]),
                "score": float(min(prev_obs["score"], next_obs["score"]) * 0.95),
                "bbox": interp_box,
                "position_3d": interp_position_3d,
                "gap_start_frame_index": start,
                "gap_end_frame_index": end,
                "gap_size": gap,
                "alpha": float(alpha),
            }
        )
    return True, "accepted", proposed


def _append_repaired_object(frame, repair):
    frame.setdefault("boxes", []).append(list(repair["bbox"]))
    frame.setdefault("scores", []).append(float(repair["score"]))
    frame.setdefault("labels", []).append(str(repair["label"]))
    frame.setdefault("track_ids", []).append(int(repair["track_id"]))
    if "sources" in frame:
        frame.setdefault("sources", []).append("tracklet_repair")
    if "positions_3d" in frame:
        frame.setdefault("positions_3d", []).append(list(repair["position_3d"]))
    if "detection_ids" in frame:
        frame.setdefault("detection_ids", []).append(
            f"tracklet_repair:{repair['track_id']}:{repair['frame_index']}"
        )
    object_record = {
        "bbox": list(repair["bbox"]),
        "score": float(repair["score"]),
        "label": str(repair["label"]),
        "track_id": int(repair["track_id"]),
        "accepted": True,
        "source": "tracklet_repair",
        "source_type": "interpolated_tracklet",
        "is_ground_truth": False,
        "position_3d": list(repair["position_3d"]),
        "has_3d_position": True,
        "repair_provenance": {
            "version": _TRACKLET_REPAIR_VERSION,
            "method": "bounded_linear_interpolation",
            "gap_start_frame_index": int(repair["gap_start_frame_index"]),
            "gap_end_frame_index": int(repair["gap_end_frame_index"]),
            "gap_size": int(repair["gap_size"]),
            "alpha": float(repair["alpha"]),
        },
    }
    frame.setdefault("objects", []).append(object_record)
    frame["has_3d_positions"] = bool(frame.get("positions_3d", []))


def _split_tracklets_at_large_gaps(video_result, cfg):
    """Give post-gap track segments new IDs before short-gap interpolation."""
    tracks, _, _ = _frame_object_observations(video_result)
    used_track_ids = {
        int(track_id)
        for frame in video_result.get("frames", [])
        for track_id in frame.get("track_ids", [])
        if str(track_id).lstrip("-").isdigit() and int(track_id) >= 0
    }
    next_track_id = max(used_track_ids, default=-1) + 1
    split_events = []

    for original_track_id, track_obs in sorted(tracks.items()):
        segments = [[track_obs[0]]] if track_obs else []
        boundary_gaps = []
        for prev_obs, obs in zip(track_obs, track_obs[1:]):
            gap_size = int(obs["frame_index"]) - int(prev_obs["frame_index"]) - 1
            if gap_size > int(cfg["max_gap_frames"]):
                segments.append([obs])
                boundary_gaps.append(
                    {
                        "gap_start_frame_index": int(prev_obs["frame_index"]),
                        "gap_end_frame_index": int(obs["frame_index"]),
                        "gap_size": int(gap_size),
                    }
                )
            else:
                segments[-1].append(obs)

        for segment_index, segment in enumerate(segments[1:], start=1):
            new_track_id = next_track_id
            next_track_id += 1
            for obs in segment:
                frame = video_result["frames"][int(obs["frame_pos"])]
                object_index = int(obs["object_index"])
                frame["track_ids"][object_index] = int(new_track_id)
                objects = frame.get("objects", [])
                if object_index < len(objects):
                    objects[object_index]["track_id"] = int(new_track_id)
                    objects[object_index]["track_split_provenance"] = {
                        "version": _TRACKLET_REPAIR_VERSION,
                        "method": "split_at_large_temporal_gap",
                        "original_track_id": int(original_track_id),
                        "segment_index": int(segment_index),
                    }
            boundary = boundary_gaps[segment_index - 1]
            split_events.append(
                {
                    "original_track_id": int(original_track_id),
                    "new_track_id": int(new_track_id),
                    "segment_index": int(segment_index),
                    "segment_start_frame_index": int(segment[0]["frame_index"]),
                    "segment_end_frame_index": int(segment[-1]["frame_index"]),
                    **boundary,
                    "method": "split_at_large_temporal_gap",
                }
            )

    if "num_tracks" in video_result:
        video_result["num_tracks"] = int(video_result.get("num_tracks", 0)) + len(split_events)
    return split_events


def _repair_video_tracklets(video_result, ego_result=None, repair_cfg=None):
    cfg = dict(_TRACKLET_REPAIR_DEFAULT_CFG)
    if repair_cfg:
        cfg.update(repair_cfg)
    repaired = copy.deepcopy(video_result)
    split_events = _split_tracklets_at_large_gaps(repaired, cfg)
    frames = repaired.get("frames", [])
    tracks, frame_indices, duplicate_track_frames = _frame_object_observations(repaired)
    repair_events = []
    skipped_gaps = []

    for track_id, track_obs in sorted(tracks.items()):
        duplicate_frames = {frame_index for dup_track_id, frame_index in duplicate_track_frames if dup_track_id == track_id}
        for obs_idx in range(max(0, len(track_obs) - 1)):
            prev_obs = track_obs[obs_idx]
            next_obs = track_obs[obs_idx + 1]
            gap = int(next_obs["frame_index"]) - int(prev_obs["frame_index"]) - 1
            if gap <= 0:
                continue
            if int(prev_obs["frame_index"]) in duplicate_frames or int(next_obs["frame_index"]) in duplicate_frames:
                skipped_gaps.append(
                    {
                        "track_id": int(track_id),
                        "gap_start_frame_index": int(prev_obs["frame_index"]),
                        "gap_end_frame_index": int(next_obs["frame_index"]),
                        "gap_size": gap,
                        "reason": "duplicate_track_endpoint",
                    }
                )
                continue
            is_safe, reason, proposed = _tracklet_gap_is_safe(track_obs, obs_idx, frame_indices, frames, cfg)
            if not is_safe:
                skipped_gaps.append(
                    {
                        "track_id": int(track_id),
                        "gap_start_frame_index": int(prev_obs["frame_index"]),
                        "gap_end_frame_index": int(next_obs["frame_index"]),
                        "gap_size": gap,
                        "reason": reason,
                    }
                )
                continue
            for repair in proposed:
                _append_repaired_object(frames[int(repair["frame_pos"])], repair)
            repair_events.append(
                {
                    "track_id": int(track_id),
                    "label": str(prev_obs["label"]),
                    "gap_start_frame_index": int(prev_obs["frame_index"]),
                    "gap_end_frame_index": int(next_obs["frame_index"]),
                    "gap_size": gap,
                    "inserted_frame_indices": [int(row["frame_index"]) for row in proposed],
                    "method": "bounded_linear_interpolation",
                }
            )

    num_interpolated = sum(len(event["inserted_frame_indices"]) for event in repair_events)
    repaired["num_objects"] = int(repaired.get("num_objects", 0)) + num_interpolated
    repaired["num_objects_with_3d"] = int(repaired.get("num_objects_with_3d", 0)) + num_interpolated
    repaired["tracklet_repair"] = {
        "version": _TRACKLET_REPAIR_VERSION,
        "method": "safest_short_gap_linear_interpolation",
        "policy": cfg,
        "ego_motion_video_id": str((ego_result or {}).get("video_id", "")),
        "num_repaired_gaps": len(repair_events),
        "num_interpolated_objects": num_interpolated,
        "repair_events": repair_events,
        "num_split_events": len(split_events),
        "num_new_track_ids": len(split_events),
        "split_events": split_events,
        "num_skipped_gaps": len(skipped_gaps),
        "skipped_gaps": skipped_gaps,
    }
    return repaired


def _build_ego_frame_map(ego_video_result):
    return {
        int(frame.get("frame_index", idx)): frame
        for idx, frame in enumerate((ego_video_result or {}).get("frames", []))
    }


def _ego_vx_vz(frame_ego):
    return (
        _safe_float(frame_ego.get("ego_vx_smoothed", frame_ego.get("ego_vx", 0.0))),
        _safe_float(frame_ego.get("ego_vz_smoothed", frame_ego.get("ego_vz", 0.0))),
    )


def _distance_state(distance_meters):
    z = _safe_float(distance_meters)
    if z <= _DISTANCE_NEAR_THRESHOLD:
        return "near"
    if z <= _DISTANCE_MEDIUM_THRESHOLD:
        return "medium"
    return "far"


def _position_x_state(x_meters):
    x = _safe_float(x_meters)
    if x < -_X_POSITION_THRESHOLD:
        return "left_of_ego"
    if x > _X_POSITION_THRESHOLD:
        return "right_of_ego"
    return "centered"


def _instantaneous_vz_state(rel_vz, has_rel_motion):
    if not has_rel_motion:
        return "vz_unknown"
    if rel_vz < -_REL_VZ_THRESHOLD:
        return "vz_approaching"
    if rel_vz > _REL_VZ_THRESHOLD:
        return "vz_awaying"
    return "vz_stable"


def _instantaneous_vx_state(rel_vx, has_rel_motion):
    if not has_rel_motion:
        return "vx_unknown"
    if rel_vx < -_REL_VX_THRESHOLD:
        return "vx_turning_left"
    if rel_vx > _REL_VX_THRESHOLD:
        return "vx_turning_right"
    return "vx_stable"


def _speed_state(rel_speed, has_rel_motion):
    if not has_rel_motion:
        return "speed_unknown"
    return "rel_moving" if _safe_float(rel_speed) > _REL_SPEED_THRESHOLD else "rel_static"


def _object_source_state(obj):
    source_type = str(obj.get("source_type", "") or obj.get("source", "")).strip()
    is_repaired = source_type == "interpolated_tracklet" or str(obj.get("source", "")) == "tracklet_repair"
    return {
        "source": "repaired" if is_repaired else "observed",
        "source_type": source_type or ("interpolated_tracklet" if is_repaired else "accepted_track"),
        "is_observed": not is_repaired,
        "is_repaired": is_repaired,
    }


def _frame_objects_with_positions(frame):
    objects = [dict(obj) for obj in list(frame.get("objects", []))]
    boxes = list(frame.get("boxes", []))
    scores = list(frame.get("scores", []))
    labels = list(frame.get("labels", []))
    track_ids = list(frame.get("track_ids", []))
    positions_3d = list(frame.get("positions_3d", []))
    sources = list(frame.get("sources", []))
    detection_ids = list(frame.get("detection_ids", [])) if isinstance(frame.get("detection_ids", []), list) else []
    rows = []
    n = max(len(objects), len(track_ids), len(boxes), len(positions_3d))
    for obj_idx in range(n):
        obj = dict(objects[obj_idx]) if obj_idx < len(objects) else {}
        box = _valid_bbox(obj.get("bbox", boxes[obj_idx] if obj_idx < len(boxes) else None))
        position_3d = _valid_position_3d(obj.get("position_3d", positions_3d[obj_idx] if obj_idx < len(positions_3d) else None))
        track_id_raw = obj.get("track_id", track_ids[obj_idx] if obj_idx < len(track_ids) else -1)
        try:
            track_id = int(track_id_raw)
        except (TypeError, ValueError):
            track_id = -1
        if track_id < 0 or box is None or position_3d is None:
            continue
        label = str(obj.get("label", labels[obj_idx] if obj_idx < len(labels) else "unknown"))
        score = _safe_float(obj.get("score", scores[obj_idx] if obj_idx < len(scores) else 0.0))
        source_type = str(obj.get("source_type", "") or obj.get("source", ""))
        if not source_type and obj_idx < len(sources):
            source_type = str(sources[obj_idx])
        source_obj = dict(obj)
        source_obj["source_type"] = source_type
        if obj_idx < len(detection_ids) and "detection_id" not in source_obj:
            source_obj["detection_id"] = str(detection_ids[obj_idx])
        rows.append(
            {
                "object_index": obj_idx,
                "track_id": track_id,
                "label": label,
                "bbox": box,
                "position_3d": position_3d,
                "score": score,
                "object": source_obj,
            }
        )
    return rows


def _relative_motion_object_entry(row, frame_index, ego_vx, ego_vz, prev_track_state):
    x, y, z = row["position_3d"]
    obj_vx = 0.0
    obj_vz = 0.0
    rel_vx = 0.0
    rel_vz = 0.0
    rel_speed = 0.0
    has_rel_motion = False
    track_id = int(row["track_id"])
    if track_id in prev_track_state:
        prev_frame_index, prev_position = prev_track_state[track_id]
        px, _, pz = prev_position
        d_frame = max(1, int(frame_index) - int(prev_frame_index))
        obj_vx = (x - px) / float(d_frame)
        obj_vz = (z - pz) / float(d_frame)
        rel_vx = obj_vx - ego_vx
        rel_vz = obj_vz - ego_vz
        rel_speed = math.hypot(rel_vx, rel_vz)
        has_rel_motion = True
    prev_track_state[track_id] = (int(frame_index), (x, y, z))

    source_state = _object_source_state(row["object"])
    vx_state = _instantaneous_vx_state(rel_vx, has_rel_motion)
    vz_state = _instantaneous_vz_state(rel_vz, has_rel_motion)
    speed_state = _speed_state(rel_speed, has_rel_motion)
    distance_state = _distance_state(z)
    x_position_state = _position_x_state(x)
    detection_id = str(row["object"].get("detection_id", ""))
    return {
        "track_id": track_id,
        "object_index": int(row["object_index"]),
        "frame_label": str(row["label"]),
        "label": str(row["label"]),
        "box": list(row["bbox"]),
        "bbox": list(row["bbox"]),
        "position_3d": [float(x), float(y), float(z)],
        "relative_position_3d": [float(x), float(y), float(z)],
        "obj_vx": float(obj_vx),
        "obj_vz": float(obj_vz),
        "ego_vx": float(ego_vx),
        "ego_vz": float(ego_vz),
        "rel_vx": float(rel_vx),
        "rel_vz": float(rel_vz),
        "rel_speed": float(rel_speed),
        "has_rel_motion": has_rel_motion,
        "motion_state": f"{source_state['source']}_with_rel_motion" if has_rel_motion else f"{source_state['source']}_without_rel_motion",
        "distance_meters": float(z),
        "distance_state": distance_state,
        "x_position_state": x_position_state,
        "vx_state": vx_state,
        "vz_state": vz_state,
        "speed_state": speed_state,
        "accepted": True,
        "score": float(row["score"]),
        "detection_id": detection_id,
        "bbox_id": detection_id,
        "source_detection_ids": [detection_id] if detection_id else [],
        "bbox_ids": [detection_id] if detection_id else [],
        "source": source_state["source"],
        "source_type": source_state["source_type"],
        "is_observed": bool(source_state["is_observed"]),
        "is_repaired": bool(source_state["is_repaired"]),
        "repair_provenance": dict(row["object"].get("repair_provenance", {})),
        "segment_ready_motion_features": {
            "relative_position_3d": [float(x), float(y), float(z)],
            "distance_meters": float(z),
            "distance_state": distance_state,
            "x_position_state": x_position_state,
            "vx_state": vx_state,
            "vz_state": vz_state,
            "speed_state": speed_state,
            "has_rel_motion": has_rel_motion,
            "has_3d_position": True,
            "source": source_state["source"],
            "source_type": source_state["source_type"],
            "is_observed": bool(source_state["is_observed"]),
            "is_repaired": bool(source_state["is_repaired"]),
            "frame_label": str(row["label"]),
        },
    }


def _relative_motion_video(video_result, ego_result):
    ego_by_frame = _build_ego_frame_map(ego_result)
    prev_track_state = {}
    frames_out = []
    for idx, frame in enumerate(video_result.get("frames", [])):
        frame_index = int(frame.get("frame_index", idx))
        frame_ego = ego_by_frame.get(frame_index, {})
        ego_vx, ego_vz = _ego_vx_vz(frame_ego)
        rows = _frame_objects_with_positions(frame)
        objects = [
            _relative_motion_object_entry(row, frame_index, ego_vx, ego_vz, prev_track_state)
            for row in rows
        ]
        frame_labels = [str(obj["frame_label"]) for obj in objects]
        frames_out.append(
            {
                "frame_index": frame_index,
                "image_path": frame.get("image_path", ""),
                "ego_vx": float(ego_vx),
                "ego_vz": float(ego_vz),
                "num_objects": len(objects),
                "num_observed_objects": sum(1 for obj in objects if obj.get("is_observed", False)),
                "num_repaired_objects": sum(1 for obj in objects if obj.get("is_repaired", False)),
                "labels": frame_labels,
                "frame_labels": frame_labels,
                "objects": objects,
            }
        )
    return {
        "version": _RELATIVE_OBJECT_MOTION_VERSION,
        "video_id": str(video_result.get("video_id", "")),
        "num_frames": len(frames_out),
        "num_frames_with_objects": sum(1 for frame in frames_out if frame["num_objects"] > 0),
        "num_objects_total": sum(int(frame["num_objects"]) for frame in frames_out),
        "num_observed_objects_total": sum(int(frame["num_observed_objects"]) for frame in frames_out),
        "num_repaired_objects_total": sum(int(frame["num_repaired_objects"]) for frame in frames_out),
        "num_objects_with_rel_motion": sum(
            1
            for frame in frames_out
            for obj in frame.get("objects", [])
            if obj.get("has_rel_motion", False)
        ),
        "frames": frames_out,
    }


def _relative_motion_track_index(relative_motion_video_result):
    tracks = {}
    frame_indices = []
    for frame in relative_motion_video_result.get("frames", []):
        frame_index = int(frame.get("frame_index", len(frame_indices)))
        frame_indices.append(frame_index)
        for obj in frame.get("objects", []):
            try:
                track_id = int(obj.get("track_id", -1))
            except (TypeError, ValueError):
                continue
            if track_id < 0:
                continue
            bucket = tracks.setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "label": str(obj.get("frame_label", obj.get("label", "unknown"))),
                    "frames": {},
                },
            )
            # Prefer repaired records if duplicate track objects ever appear in a frame:
            # these are the only records that need explicit visual audit.
            existing = bucket["frames"].get(frame_index)
            if existing is None or bool(obj.get("is_repaired", False)):
                bucket["frames"][frame_index] = dict(obj)
    return sorted(set(frame_indices)), tracks


def _visual_source_color(obj):
    if obj is None:
        return _VIS_ABSENT_COLOR
    return _VIS_REPAIRED_COLOR if bool(obj.get("is_repaired", False)) else _VIS_OBSERVED_COLOR


def _visual_source_label(obj):
    if obj is None:
        return "absent"
    return "repaired" if bool(obj.get("is_repaired", False)) else "observed"


def _put_text_with_background(cv2, image, text, org, scale, color, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(
        image,
        (x - 4, y - text_h - baseline - 4),
        (x + text_w + 4, y + baseline + 4),
        color,
        -1,
    )
    text_color = (0, 0, 0) if sum(color) > 360 else (255, 255, 255)
    cv2.putText(image, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)


def _draw_track_progress_bar(cv2, panel, frame_indices, current_frame_index, track_frames, width):
    bar_x = 18
    bar_w = max(1, width - 2 * bar_x)
    bar_h = 18
    # Keep the temporal presence/source bar immediately below the video.
    bar_y = 8
    n = max(1, len(frame_indices))
    for idx, frame_index in enumerate(frame_indices):
        x1 = bar_x + int(round(idx * bar_w / n))
        x2 = bar_x + int(round((idx + 1) * bar_w / n))
        color = _visual_source_color(track_frames.get(frame_index))
        cv2.rectangle(panel, (x1, bar_y), (max(x1 + 1, x2), bar_y + bar_h), color, -1)
    current_pos = frame_indices.index(current_frame_index) if current_frame_index in frame_indices else 0
    marker_x = bar_x + int(round((current_pos + 0.5) * bar_w / n))
    cv2.line(panel, (marker_x, bar_y - 7), (marker_x, bar_y + bar_h + 7), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (210, 210, 210), 1)


def _trajectory_filter_reason_codes(trajectory_evidence):
    """Collect the concrete Step 8B reason codes shown in track videos."""
    evidence = dict(trajectory_evidence or {})
    fact_decision = dict(evidence.get("fact_decision", {}))
    validation = dict(evidence.get("causal_motion_fact_validation", {}))
    significance = dict(evidence.get("motion_significance_assessment", {}))
    reason_codes = []

    def add(value):
        value = str(value or "").strip()
        if value and value not in reason_codes:
            reason_codes.append(value)

    for reason in fact_decision.get("decision_reasons", []):
        reason = dict(reason or {})
        add(reason.get("kind"))
        for key in ("validation_reasons", "uncertain_reasons", "significance_reasons"):
            for nested_reason in reason.get(key, []):
                add(nested_reason)
    for reason in validation.get("rejection_reasons", []):
        add(reason)
    for reason in validation.get("uncertain_reasons", []):
        add(reason)
    for reason in significance.get("reasons", []):
        add(dict(reason or {}).get("kind"))
    return reason_codes


def _trajectory_decision_reason_table(trajectory_evidence):
    """Build the fixed eight-cell quantitative decision-reason table."""
    evidence = dict(trajectory_evidence or {})
    fact_decision = dict(evidence.get("fact_decision", {}))
    validation = dict(evidence.get("causal_motion_fact_validation", {}))
    significance = dict(evidence.get("motion_significance_assessment", {}))
    provenance = dict(evidence.get("provenance", {}))
    active_codes = {
        str(reason.get("kind", ""))
        for reason in fact_decision.get("decision_reasons", [])
        if isinstance(reason, dict)
    }
    validation_status = str(validation.get("validation_status", validation.get("status", "uncertain")))
    significance_status = str(significance.get("significance", evidence.get("motion_significance", "low_significance")))
    invalid_count = len(validation.get("rejection_reasons", []))
    uncertain_count = len(validation.get("uncertain_reasons", []))
    significance_count = len(significance.get("reasons", []))
    repaired_count = int(provenance.get("repaired_count", 0))
    merged_count = int(provenance.get("merged_count", 0))
    valid_flag = int(validation_status == "valid")
    high_flag = int(significance_status == "high_significance")
    low_flag = int(significance_status == "low_significance")
    credibility_uncertainty_count = uncertain_count + int(validation_status == "uncertain") + significance_count
    specs = (
        ("invalid_trajectory", invalid_count, 1, f"invalid_issues={invalid_count}"),
        ("repaired_trajectory_kept", repaired_count + merged_count, 1, f"repairs+merges={repaired_count + merged_count}"),
        ("low_motion_significance", significance_count, 1, f"low_motion_reasons={significance_count}"),
        ("valid_high_significance", valid_flag + high_flag, 2, f"valid={valid_flag}, high={high_flag}"),
        ("valid_low_motion_retained", valid_flag + low_flag, 2, f"valid={valid_flag}, low={low_flag}"),
        ("credible_but_uncertain", credibility_uncertainty_count, 1, f"uncertainty_evidence={credibility_uncertainty_count}"),
        ("validation_uncertainty", uncertain_count, 1, f"uncertain_issues={uncertain_count}"),
        ("significance_uncertainty", significance_count, 1, f"significance_reasons={significance_count}"),
    )
    rows = []
    for reason_name, measured_value, threshold, evidence_text in specs:
        active = reason_name in active_codes
        rows.append(
            {
                "reason": reason_name,
                "active": active,
                "measured_value": float(measured_value),
                "threshold": float(threshold),
                "distance_to_threshold": float(measured_value - threshold),
                "evidence": evidence_text,
            }
        )
    return rows


def _draw_decision_reason_table(cv2, panel, entries, x, y, width, row_height=88):
    columns = 4
    cell_width = max(1, width // columns)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, entry in enumerate(entries):
        row = index // columns
        column = index % columns
        cell_x = x + column * cell_width
        cell_y = y + row * row_height
        cell_w = cell_width if column < columns - 1 else width - column * cell_width
        active = bool(entry.get("active", False))
        border_color = (80, 220, 80) if active else (92, 92, 92)
        fill_color = (34, 58, 34) if active else (32, 32, 32)
        cv2.rectangle(panel, (cell_x, cell_y), (cell_x + cell_w, cell_y + row_height), fill_color, -1)
        cv2.rectangle(panel, (cell_x, cell_y), (cell_x + cell_w, cell_y + row_height), border_color, 2 if active else 1)
        reason = str(entry.get("reason", ""))
        words = reason.split("_")
        midpoint = max(1, (len(words) + 1) // 2)
        title_lines = ["_".join(words[:midpoint]), "_".join(words[midpoint:])]
        title_lines = [line for line in title_lines if line]
        for line_index, title_line in enumerate(title_lines[:2]):
            cv2.putText(panel, title_line, (cell_x + 6, cell_y + 15 + line_index * 13), font, 0.32, (235, 235, 235), 1, cv2.LINE_AA)
        status_color = (100, 255, 100) if active else (165, 165, 165)
        cv2.putText(panel, "ACTIVE" if active else "inactive", (cell_x + 6, cell_y + 43), font, 0.38, status_color, 1, cv2.LINE_AA)
        value = float(entry.get("measured_value", 0.0))
        threshold = float(entry.get("threshold", 1.0))
        distance = float(entry.get("distance_to_threshold", value - threshold))
        cv2.putText(
            panel,
            f"value={value:.0f}  threshold={threshold:.0f}  delta={distance:+.0f}",
            (cell_x + 6, cell_y + 61),
            font,
            0.31,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(panel, str(entry.get("evidence", "")), (cell_x + 6, cell_y + 79), font, 0.30, (190, 190, 190), 1, cv2.LINE_AA)


def _ego_refinement_series(refined_ego_motion_video):
    frame_rows = {
        int(frame.get("frame_index", idx)): dict(frame)
        for idx, frame in enumerate((refined_ego_motion_video or {}).get("frames", []))
    }
    return {
        "method": str((refined_ego_motion_video or {}).get("method", "")),
        "frames": frame_rows,
        "methods": {
            "original": {
                "label": "original",
                "vx_field": "original_ego_vx",
                "vz_field": "original_ego_vz",
                "color": _VIS_EGO_METHOD_COLORS["original"],
            },
            "weighted_median": {
                "label": "median vote",
                "vx_field": "reference_estimated_ego_vx",
                "vz_field": "reference_estimated_ego_vz",
                "color": _VIS_EGO_METHOD_COLORS["weighted_median"],
            },
            "refined": {
                "label": "refined",
                "vx_field": "refined_ego_vx",
                "vz_field": "refined_ego_vz",
                "color": _VIS_EGO_METHOD_COLORS["refined"],
            },
            "ransac": {
                "label": "RANSAC",
                "vx_field": "ransac_ego_vx",
                "vz_field": "ransac_ego_vz",
                "color": _VIS_EGO_METHOD_COLORS["ransac"],
            },
        },
    }


def _series_values_for_field(frame_indices, frame_rows, field):
    values = []
    available = False
    for frame_index in frame_indices:
        row = frame_rows.get(frame_index, {})
        value = row.get(field)
        if value is None:
            values.append(None)
            continue
        available = True
        values.append(_safe_float(value))
    return values if available else []


def _draw_line_chart(cv2, panel, title, frame_indices, current_frame_index, method_values, x, y, w, h):
    cv2.rectangle(panel, (x, y), (x + w, y + h), (34, 34, 34), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (90, 90, 90), 1)
    cv2.putText(panel, title, (x + 6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    all_values = [
        value
        for series in method_values.values()
        for value in series.get("values", [])
        if value is not None
    ]
    if not all_values or not frame_indices:
        cv2.putText(panel, "ego refinement unavailable", (x + 8, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        return
    lo = min(all_values)
    hi = max(all_values)
    if abs(hi - lo) < 1e-6:
        pad = max(0.1, abs(hi) * 0.25)
        lo -= pad
        hi += pad
    else:
        pad = (hi - lo) * 0.12
        lo -= pad
        hi += pad
    plot_x = x + 8
    plot_y = y + 22
    plot_w = max(1, w - 16)
    plot_h = max(1, h - 32)
    n = max(1, len(frame_indices) - 1)

    def point(idx, value):
        px = plot_x + int(round(idx * plot_w / n))
        py = plot_y + plot_h - int(round((value - lo) * plot_h / max(1e-6, hi - lo)))
        return px, py

    zero_y = None
    if lo <= 0.0 <= hi:
        zero_y = point(0, 0.0)[1]
        cv2.line(panel, (plot_x, zero_y), (plot_x + plot_w, zero_y), (70, 70, 70), 1, cv2.LINE_AA)

    for method_name, series in method_values.items():
        values = series.get("values", [])
        color = series.get("color", (220, 220, 220))
        prev_pt = None
        for idx, value in enumerate(values):
            if value is None:
                prev_pt = None
                continue
            pt = point(idx, value)
            cv2.circle(panel, pt, 2, color, -1)
            if prev_pt is not None:
                cv2.line(panel, prev_pt, pt, color, 2, cv2.LINE_AA)
            prev_pt = pt

    current_pos = frame_indices.index(current_frame_index) if current_frame_index in frame_indices else 0
    cursor_x = plot_x + int(round(current_pos * plot_w / n))
    cv2.line(panel, (cursor_x, plot_y), (cursor_x, plot_y + plot_h), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{lo:+.2f}", (x + w - 46, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{hi:+.2f}", (x + w - 46, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 170), 1, cv2.LINE_AA)


def _draw_ego_motion_comparison_charts(cv2, panel, frame_indices, current_frame_index, ego_series):
    if not ego_series:
        return
    frame_rows = dict(ego_series.get("frames", {}))
    methods = dict(ego_series.get("methods", {}))
    vx_values = {}
    vz_values = {}
    for method_name, meta in methods.items():
        vx_series = _series_values_for_field(frame_indices, frame_rows, meta.get("vx_field", ""))
        vz_series = _series_values_for_field(frame_indices, frame_rows, meta.get("vz_field", ""))
        if vx_series:
            vx_values[method_name] = {
                "values": vx_series,
                "color": meta.get("color", (220, 220, 220)),
            }
        if vz_series:
            vz_values[method_name] = {
                "values": vz_series,
                "color": meta.get("color", (220, 220, 220)),
            }
    panel_h, panel_w = panel.shape[:2]
    chart_y = 128
    chart_h = max(44, panel_h - chart_y - 70)
    gap = 10
    chart_w = max(80, (panel_w - 36 - gap) // 2)
    _draw_line_chart(cv2, panel, "ego vx", frame_indices, current_frame_index, vx_values, 18, chart_y, chart_w, chart_h)
    _draw_line_chart(cv2, panel, "ego vz", frame_indices, current_frame_index, vz_values, 18 + chart_w + gap, chart_y, chart_w, chart_h)
    legend_y = min(panel_h - 48, chart_y + chart_h + 18)
    legend_x = 18
    for method_name in ("original", "weighted_median", "refined", "ransac"):
        meta = methods.get(method_name, {})
        color = meta.get("color", (180, 180, 180))
        label = str(meta.get("label", method_name))
        is_available = bool(
            _series_values_for_field(frame_indices, frame_rows, meta.get("vx_field", ""))
            or _series_values_for_field(frame_indices, frame_rows, meta.get("vz_field", ""))
        )
        if not is_available and method_name == "ransac":
            label = "RANSAC n/a"
        cv2.rectangle(panel, (legend_x, legend_y - 10), (legend_x + 14, legend_y + 2), color, -1)
        cv2.putText(panel, label, (legend_x + 18, legend_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
        legend_x += 104 if method_name != "weighted_median" else 132


def _bgr_to_mpl_rgb(color):
    b, g, r = color
    return (r / 255.0, g / 255.0, b / 255.0)


def _pdf_escape_text(text):
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_num(value):
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _bgr_to_pdf_rgb(color):
    b, g, r = color
    return (r / 255.0, g / 255.0, b / 255.0)


def _pdf_text(commands, x, y, text, size=10, color=(0.0, 0.0, 0.0)):
    r, g, b = color
    commands.append(f"{_pdf_num(r)} {_pdf_num(g)} {_pdf_num(b)} rg")
    commands.append(f"BT /F1 {int(size)} Tf {_pdf_num(x)} {_pdf_num(y)} Td ({_pdf_escape_text(text)}) Tj ET")


def _pdf_line(commands, x1, y1, x2, y2, color=(0.0, 0.0, 0.0), width=1.0):
    r, g, b = color
    commands.append(f"{_pdf_num(r)} {_pdf_num(g)} {_pdf_num(b)} RG")
    commands.append(f"{_pdf_num(width)} w")
    commands.append(f"{_pdf_num(x1)} {_pdf_num(y1)} m {_pdf_num(x2)} {_pdf_num(y2)} l S")


def _pdf_rect(commands, x, y, w, h, color=(0.0, 0.0, 0.0), width=1.0):
    r, g, b = color
    commands.append(f"{_pdf_num(r)} {_pdf_num(g)} {_pdf_num(b)} RG")
    commands.append(f"{_pdf_num(width)} w")
    commands.append(f"{_pdf_num(x)} {_pdf_num(y)} {_pdf_num(w)} {_pdf_num(h)} re S")


def _build_ego_chart_series(axis_field, frame_indices, frame_rows, methods, method_names=None):
    chart_series = []
    all_values = []
    if method_names is None:
        method_names = ("original", "weighted_median", "refined", "ransac")
    for method_name in method_names:
        meta = dict(methods.get(method_name, {}))
        values = _series_values_for_field(frame_indices, frame_rows, meta.get(axis_field, ""))
        if not values:
            continue
        points = [(frame, value) for frame, value in zip(frame_indices, values) if value is not None]
        if not points:
            continue
        all_values.extend(value for _, value in points)
        chart_series.append(
            {
                "label": str(meta.get("label", method_name)),
                "points": points,
                "color": meta.get("color", (220, 220, 220)),
            }
        )
    return chart_series, all_values


def _ego_motion_chart_rows():
    """Return Step 8 chart rows in display order.

    Each row is rendered with vx in the left column and vz in the right
    column.  The final row retains the original all-method comparison.
    """
    return (
        ("original", ("original",)),
        ("median vote", ("weighted_median",)),
        ("refined", ("refined",)),
        ("combined comparison", ("original", "weighted_median", "refined", "ransac")),
    )


def _draw_pdf_ego_axis(commands, x, y, w, h, title, frame_indices, chart_series, all_values):
    _pdf_text(commands, x, y + h + 18, title, size=11, color=(0.0, 0.0, 0.0))
    _pdf_rect(commands, x, y, w, h, color=(0.2, 0.2, 0.2), width=0.8)
    if not chart_series or not all_values:
        _pdf_text(commands, x + 12, y + h / 2, "ego refinement unavailable", size=10, color=(0.35, 0.35, 0.35))
        return

    lo = min(all_values)
    hi = max(all_values)
    if abs(hi - lo) < 1e-9:
        pad = max(0.1, abs(hi) * 0.25)
        lo -= pad
        hi += pad
    else:
        pad = (hi - lo) * 0.12
        lo -= pad
        hi += pad

    first_frame = min(frame_indices)
    last_frame = max(frame_indices)
    frame_span = max(1, last_frame - first_frame)

    def point(frame, value):
        px = x + ((float(frame) - first_frame) / frame_span) * w
        py = y + ((float(value) - lo) / max(1e-9, hi - lo)) * h
        return px, py

    if lo <= 0.0 <= hi:
        _, zero_y = point(first_frame, 0.0)
        _pdf_line(commands, x, zero_y, x + w, zero_y, color=(0.72, 0.72, 0.72), width=0.6)

    _pdf_text(commands, x - 42, y - 4, _pdf_num(lo), size=8, color=(0.35, 0.35, 0.35))
    _pdf_text(commands, x - 42, y + h - 4, _pdf_num(hi), size=8, color=(0.35, 0.35, 0.35))
    _pdf_text(commands, x, y - 18, str(first_frame), size=8, color=(0.35, 0.35, 0.35))
    _pdf_text(commands, x + w - 24, y - 18, str(last_frame), size=8, color=(0.35, 0.35, 0.35))

    legend_x = x + w - 150
    legend_y = y + h + 18
    for series in chart_series:
        rgb = _bgr_to_pdf_rgb(series["color"])
        _pdf_line(commands, legend_x, legend_y + 3, legend_x + 18, legend_y + 3, color=rgb, width=2.0)
        _pdf_text(commands, legend_x + 24, legend_y, series["label"], size=8, color=(0.0, 0.0, 0.0))
        legend_x += 82

    for series in chart_series:
        points = [point(frame, value) for frame, value in series["points"]]
        if len(points) == 1:
            px, py = points[0]
            _pdf_line(commands, px - 1.5, py, px + 1.5, py, color=_bgr_to_pdf_rgb(series["color"]), width=2.0)
            continue
        r, g, b = _bgr_to_pdf_rgb(series["color"])
        commands.append(f"{_pdf_num(r)} {_pdf_num(g)} {_pdf_num(b)} RG")
        commands.append("1.8 w")
        first_x, first_y = points[0]
        path = [f"{_pdf_num(first_x)} {_pdf_num(first_y)} m"]
        path.extend(f"{_pdf_num(px)} {_pdf_num(py)} l" for px, py in points[1:])
        path.append("S")
        commands.append(" ".join(path))


def _save_ego_motion_comparison_pdf_simple(refined_ego_motion_video, output_path):
    ego_series = _ego_refinement_series(refined_ego_motion_video)
    frame_rows = dict(ego_series.get("frames", {}))
    if not frame_rows:
        return None, "no_refined_ego_motion"
    frame_indices = sorted(frame_rows)
    methods = dict(ego_series.get("methods", {}))
    chart_rows = []
    for row_label, method_names in _ego_motion_chart_rows():
        vx_series, vx_values = _build_ego_chart_series(
            "vx_field", frame_indices, frame_rows, methods, method_names
        )
        vz_series, vz_values = _build_ego_chart_series(
            "vz_field", frame_indices, frame_rows, methods, method_names
        )
        chart_rows.append((row_label, vx_series, vx_values, vz_series, vz_values))
    if not any(vx_series or vz_series for _, vx_series, _, vz_series, _ in chart_rows):
        return None, "no_available_chart_series"

    page_w = 792.0
    page_h = 612.0
    commands = []
    video_id = str((refined_ego_motion_video or {}).get("video_id", ""))
    _pdf_text(commands, 42, page_h - 30, f"ego motion comparison | {video_id}", size=15, color=(0.0, 0.0, 0.0))
    chart_w = 328.0
    chart_h = 92.0
    left_x = 48.0
    right_x = 432.0
    for row_index, (row_label, vx_series, vx_values, vz_series, vz_values) in enumerate(chart_rows):
        chart_y = 438.0 - row_index * 132.0
        _draw_pdf_ego_axis(
            commands, left_x, chart_y, chart_w, chart_h,
            f"{row_label} | ego vx", frame_indices, vx_series, vx_values,
        )
        _draw_pdf_ego_axis(
            commands, right_x, chart_y, chart_w, chart_h,
            f"{row_label} | ego vz", frame_indices, vz_series, vz_values,
        )
    _pdf_text(commands, 378, 12, "frame", size=10, color=(0.0, 0.0, 0.0))

    content = "\n".join(commands).encode("ascii", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 792 612] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pdf)
    return str(output_path), "rendered"


def _save_ego_motion_comparison_pdf(refined_ego_motion_video, output_path):
    if plt is None:
        return _save_ego_motion_comparison_pdf_simple(refined_ego_motion_video, output_path)
    ego_series = _ego_refinement_series(refined_ego_motion_video)
    frame_rows = dict(ego_series.get("frames", {}))
    if not frame_rows:
        return None, "no_refined_ego_motion"

    frame_indices = sorted(frame_rows)
    methods = dict(ego_series.get("methods", {}))
    chart_rows = _ego_motion_chart_rows()
    fig, axes = plt.subplots(4, 2, figsize=(14.0, 12.0), sharex=True, squeeze=False)
    plotted = False
    for row_index, (row_label, method_names) in enumerate(chart_rows):
        for column_index, (axis_name, field_name) in enumerate((("ego vx", "vx_field"), ("ego vz", "vz_field"))):
            axis = axes[row_index][column_index]
            axis_plotted = False
            for method_name in method_names:
                meta = dict(methods.get(method_name, {}))
                values = _series_values_for_field(frame_indices, frame_rows, meta.get(field_name, ""))
                if not values:
                    continue
                xs = [frame for frame, value in zip(frame_indices, values) if value is not None]
                ys = [value for value in values if value is not None]
                if not xs:
                    continue
                axis.plot(
                    xs,
                    ys,
                    label=str(meta.get("label", method_name)),
                    color=_bgr_to_mpl_rgb(meta.get("color", (220, 220, 220))),
                    linewidth=1.8,
                )
                plotted = True
                axis_plotted = True
            axis.axhline(0.0, color="#888888", linewidth=0.8, alpha=0.55)
            axis.set_title(f"{row_label} | {axis_name}")
            axis.set_ylabel(axis_name)
            axis.grid(True, color="#dddddd", linewidth=0.6, alpha=0.75)
            if axis_plotted:
                axis.legend(loc="best", frameon=True)
            else:
                axis.text(0.5, 0.5, "ego refinement unavailable", ha="center", va="center", transform=axis.transAxes)

    if not plotted:
        plt.close(fig)
        return None, "no_available_chart_series"

    video_id = str((refined_ego_motion_video or {}).get("video_id", ""))
    axes[-1][0].set_xlabel("frame")
    axes[-1][1].set_xlabel("frame")
    fig.suptitle(f"ego motion comparison | {video_id}".strip(), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    return str(output_path), "rendered"


def _render_relative_motion_track_video(
    relative_motion_video_result,
    track_id,
    track_data,
    frame_indices,
    output_path,
    fps=10.0,
    trajectory_evidence=None,
):
    try:
        import cv2
    except ModuleNotFoundError:
        return None, "missing_cv2"

    frames = list(relative_motion_video_result.get("frames", []))
    if not frames or not frame_indices:
        return None, "no_frames"
    frame_by_index = {
        int(frame.get("frame_index", idx)): frame
        for idx, frame in enumerate(frames)
    }

    first_img = None
    for frame_index in frame_indices:
        image_path = str(frame_by_index.get(frame_index, {}).get("image_path", ""))
        if image_path:
            first_img = cv2.imread(image_path)
        if first_img is not None:
            break
    if first_img is None:
        return None, "missing_frame_images"

    frame_h, frame_w = first_img.shape[:2]
    panel_h = max(400, int(frame_h * 0.55))
    total_h = frame_h + panel_h
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frame_w, total_h),
    )
    if not writer.isOpened():
        return None, "writer_open_failed"

    track_frames = dict(track_data.get("frames", {}))
    thickness = max(5, int(round(min(frame_w, frame_h) / 140.0)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    video_id = str(relative_motion_video_result.get("video_id", ""))
    label = str(track_data.get("label", "unknown"))
    trajectory_evidence = dict(trajectory_evidence or {})
    fact_decision = dict(trajectory_evidence.get("fact_decision", {}))
    decision_status = str(fact_decision.get("decision", trajectory_evidence.get("fact_decision_status", "not_available")))
    decision_reason_entries = _trajectory_decision_reason_table(trajectory_evidence)
    decision_color = _VIS_DECISION_COLORS.get(decision_status, (180, 180, 180))

    try:
        for frame_index in frame_indices:
            frame = frame_by_index.get(frame_index, {})
            img = cv2.imread(str(frame.get("image_path", "")))
            if img is None:
                img = first_img.copy()
                img[:] = 0
            elif img.shape[:2] != (frame_h, frame_w):
                img = cv2.resize(img, (frame_w, frame_h))

            obj = track_frames.get(frame_index)
            source_label = _visual_source_label(obj)
            source_color = _visual_source_color(obj)
            motion_state = "not_present"
            if obj is not None:
                box = _valid_bbox(obj.get("bbox", obj.get("box", [])))
                motion_state = str(obj.get("motion_state", "unknown"))
                if box is not None:
                    x1, y1, x2, y2 = [int(round(value)) for value in box]
                    x1 = max(0, min(frame_w - 1, x1))
                    x2 = max(0, min(frame_w - 1, x2))
                    y1 = max(0, min(frame_h - 1, y1))
                    y2 = max(0, min(frame_h - 1, y2))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness + 3)
                    cv2.rectangle(img, (x1, y1), (x2, y2), source_color, thickness)
                    text_y = max(y1 - 10, 28)
                    _put_text_with_background(
                        cv2,
                        img,
                        f"track {track_id} | {label} | {source_label} | {decision_status}",
                        (max(8, x1), text_y),
                        0.72,
                        decision_color if decision_status == "Discard" else source_color,
                        2,
                    )

            header = f"{video_id} | track {track_id} | frame {frame_index:05d}"
            cv2.putText(img, header, (12, 30), font, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, header, (12, 30), font, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

            panel = cv2.resize(first_img[:1, :1], (frame_w, panel_h))
            panel[:] = (24, 24, 24)
            _draw_track_progress_bar(cv2, panel, frame_indices, frame_index, track_frames, frame_w)
            cv2.rectangle(panel, (0, 38), (frame_w, 46), decision_color, -1)
            cv2.putText(
                panel,
                f"relative motion: {motion_state}",
                (18, 70),
                font,
                0.7,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
            if obj is not None:
                metrics = (
                    f"source={source_label}  rel_vx={_safe_float(obj.get('rel_vx', 0.0)):+.3f}  "
                    f"rel_vz={_safe_float(obj.get('rel_vz', 0.0)):+.3f}  "
                    f"speed={_safe_float(obj.get('rel_speed', 0.0)):.3f}"
                )
            else:
                metrics = "source=absent"
            cv2.putText(panel, metrics, (18, 98), font, 0.58, (210, 210, 210), 1, cv2.LINE_AA)
            decision_text = f"causal filter: {decision_status}"
            cv2.putText(panel, decision_text, (18, 124), font, 0.56, decision_color, 2, cv2.LINE_AA)
            cv2.putText(panel, "decision reason diagnostics (2 x 4)", (18, 148), font, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
            _draw_decision_reason_table(
                cv2,
                panel,
                decision_reason_entries,
                x=18,
                y=158,
                width=max(1, frame_w - 36),
                row_height=88,
            )
            legend_y = panel_h - 62
            cv2.rectangle(panel, (18, legend_y), (42, legend_y + 16), _VIS_OBSERVED_COLOR, -1)
            cv2.putText(panel, "observed", (48, legend_y + 15), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (142, legend_y), (166, legend_y + 16), _VIS_REPAIRED_COLOR, -1)
            cv2.putText(panel, "repaired", (172, legend_y + 15), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (266, legend_y), (290, legend_y + 16), _VIS_ABSENT_COLOR, -1)
            cv2.putText(panel, "absent", (296, legend_y + 15), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (382, legend_y), (406, legend_y + 16), decision_color, -1)
            cv2.putText(panel, decision_status, (412, legend_y + 15), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            writer.write(cv2.vconcat([img, panel]))
    finally:
        writer.release()

    return str(output_path), "rendered"


def _render_relative_motion_track_videos(
    relative_motion_video_result,
    output_root,
    fps=10.0,
    trajectory_evidence_by_track=None,
):
    frame_indices, tracks = _relative_motion_track_index(relative_motion_video_result)
    video_id = str(relative_motion_video_result.get("video_id", ""))
    trajectory_evidence_by_track = dict(trajectory_evidence_by_track or {})
    rendered = []
    skipped = []
    for track_id, track_data in sorted(tracks.items()):
        trajectory_evidence = dict(trajectory_evidence_by_track.get(int(track_id), {}))
        output_path = Path(output_root) / video_id / f"track_{track_id:04d}_relative_motion.mp4"
        path, status = _render_relative_motion_track_video(
            relative_motion_video_result=relative_motion_video_result,
            track_id=track_id,
            track_data=track_data,
            frame_indices=frame_indices,
            output_path=output_path,
            fps=fps,
            trajectory_evidence=trajectory_evidence,
        )
        row = {
            "video_id": video_id,
            "track_id": int(track_id),
            "label": str(track_data.get("label", "unknown")),
            "num_present_frames": len(track_data.get("frames", {})),
            "status": status,
            "fact_decision_status": str(trajectory_evidence.get("fact_decision_status", "")),
            "validation_status": str(trajectory_evidence.get("validation_status", "")),
            "motion_significance": str(trajectory_evidence.get("motion_significance", "")),
            "symbolic_layer_eligible": bool(trajectory_evidence.get("symbolic_layer_eligible", False)),
        }
        if path:
            row["visualization_path"] = path
            rendered.append(row)
        else:
            skipped.append(row)
    return rendered, skipped


def _numeric_stats(values):
    vals = [_safe_float(value) for value in values if value is not None and math.isfinite(_safe_float(value))]
    if not vals:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "first": 0.0,
            "last": 0.0,
            "delta": 0.0,
            "abs_delta": 0.0,
            "mean_abs_step": 0.0,
            "max_abs_step": 0.0,
        }
    mean = sum(vals) / len(vals)
    variance = sum((value - mean) ** 2 for value in vals) / len(vals)
    abs_steps = [abs(right - left) for left, right in zip(vals, vals[1:])]
    return {
        "count": len(vals),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "first": float(vals[0]),
        "last": float(vals[-1]),
        "delta": float(vals[-1] - vals[0]),
        "abs_delta": float(abs(vals[-1] - vals[0])),
        "mean_abs_step": float(sum(abs_steps) / len(abs_steps)) if abs_steps else 0.0,
        "max_abs_step": float(max(abs_steps)) if abs_steps else 0.0,
    }


def _bbox_features(box):
    bbox = _valid_bbox(box)
    if bbox is None:
        return {
            "width": 0.0,
            "height": 0.0,
            "area": 0.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "diag": 0.0,
        }
    width, height = _bbox_size(bbox)
    center_x, center_y = _bbox_center(bbox)
    return {
        "width": float(width),
        "height": float(height),
        "area": float(_bbox_area(bbox)),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "diag": float(_bbox_diag(bbox)),
    }


def _source_kind_from_motion_object(obj):
    source = str(obj.get("source", "")).strip().lower()
    source_type = str(obj.get("source_type", "")).strip().lower()
    if bool(obj.get("is_repaired", False)) or source == "repaired" or source_type == "interpolated_tracklet":
        return "repaired"
    if source == "merged" or "merged" in source_type:
        return "merged"
    return "observed"


def _trajectory_observation_from_motion_object(frame_index, obj):
    bbox = list(obj.get("bbox", obj.get("box", [])))
    position_3d = list(obj.get("position_3d", obj.get("relative_position_3d", [])))
    source_kind = _source_kind_from_motion_object(obj)
    score = _safe_float(obj.get("score", 0.0), 0.0)
    has_rel_motion = bool(obj.get("has_rel_motion", False))
    source_uncertainty = 0.0
    if source_kind == "repaired":
        source_uncertainty += 0.35
    elif source_kind == "merged":
        source_uncertainty += 0.2
    if not has_rel_motion:
        source_uncertainty += 0.25
    if score > 0.0:
        source_uncertainty += max(0.0, 1.0 - score) * 0.2
    return {
        "frame_index": int(frame_index),
        "object_index": int(obj.get("object_index", -1)),
        "frame_label": str(obj.get("frame_label", obj.get("label", "unknown"))),
        "bbox": bbox,
        "position_3d": position_3d,
        "motion": {
            "obj_vx": _safe_float(obj.get("obj_vx", 0.0)),
            "obj_vz": _safe_float(obj.get("obj_vz", 0.0)),
            "ego_vx": _safe_float(obj.get("ego_vx", 0.0)),
            "ego_vz": _safe_float(obj.get("ego_vz", 0.0)),
            "rel_vx": _safe_float(obj.get("rel_vx", 0.0)),
            "rel_vz": _safe_float(obj.get("rel_vz", 0.0)),
            "rel_speed": _safe_float(obj.get("rel_speed", 0.0)),
            "has_rel_motion": has_rel_motion,
            "motion_state": str(obj.get("motion_state", "unknown")),
            "vx_state": str(obj.get("vx_state", "vx_unknown")),
            "vz_state": str(obj.get("vz_state", "vz_unknown")),
            "speed_state": str(obj.get("speed_state", "speed_unknown")),
            "distance_meters": _safe_float(obj.get("distance_meters", position_3d[2] if len(position_3d) > 2 else 0.0)),
            "distance_state": str(obj.get("distance_state", "unknown")),
            "x_position_state": str(obj.get("x_position_state", "unknown")),
        },
        "provenance": {
            "source": source_kind,
            "source_type": str(obj.get("source_type", "")),
            "is_observed": source_kind == "observed",
            "is_repaired": source_kind == "repaired",
            "is_merged": source_kind == "merged",
            "detection_id": str(obj.get("detection_id", "")),
            "bbox_id": str(obj.get("bbox_id", "")),
            "source_detection_ids": list(obj.get("source_detection_ids", [])),
            "bbox_ids": list(obj.get("bbox_ids", [])),
            "repair_provenance": dict(obj.get("repair_provenance", {})),
        },
        "uncertainty": {
            "score": float(score),
            "source_uncertainty": float(min(1.0, source_uncertainty)),
            "has_rel_motion": has_rel_motion,
        },
    }


def _trajectory_statistics(observations, video_num_frames):
    frame_indices = [int(obs["frame_index"]) for obs in observations]
    frame_gaps = [right - left for left, right in zip(frame_indices, frame_indices[1:])]
    positions = [list(obs.get("position_3d", [])) for obs in observations]
    valid_positions = [pos for pos in positions if len(pos) >= 3]
    bboxes = [list(obs.get("bbox", [])) for obs in observations]
    bbox_rows = [_bbox_features(box) for box in bboxes]
    motions = [dict(obs.get("motion", {})) for obs in observations]

    path_length_3d = 0.0
    path_length_xz = 0.0
    for left, right in zip(valid_positions, valid_positions[1:]):
        dx = _safe_float(right[0]) - _safe_float(left[0])
        dy = _safe_float(right[1]) - _safe_float(left[1])
        dz = _safe_float(right[2]) - _safe_float(left[2])
        path_length_3d += math.sqrt(dx * dx + dy * dy + dz * dz)
        path_length_xz += math.sqrt(dx * dx + dz * dz)
    displacement_3d = 0.0
    displacement_xz = 0.0
    if len(valid_positions) >= 2:
        first = valid_positions[0]
        last = valid_positions[-1]
        dx = _safe_float(last[0]) - _safe_float(first[0])
        dy = _safe_float(last[1]) - _safe_float(first[1])
        dz = _safe_float(last[2]) - _safe_float(first[2])
        displacement_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        displacement_xz = math.sqrt(dx * dx + dz * dz)

    center_path_px = 0.0
    centers = [(row["center_x"], row["center_y"]) for row in bbox_rows]
    for left, right in zip(centers, centers[1:]):
        center_path_px += math.hypot(right[0] - left[0], right[1] - left[1])
    center_displacement_px = 0.0
    if len(centers) >= 2:
        center_displacement_px = math.hypot(centers[-1][0] - centers[0][0], centers[-1][1] - centers[0][1])

    source_counts = Counter(str(obs.get("provenance", {}).get("source", "observed")) for obs in observations)
    motion_state_counts = Counter(str(motion.get("motion_state", "unknown")) for motion in motions)
    label_counts = Counter(str(obs.get("frame_label", "unknown")) for obs in observations)
    num_observations = len(observations)
    frame_span = (max(frame_indices) - min(frame_indices) + 1) if frame_indices else 0
    rel_motion_count = sum(1 for motion in motions if bool(motion.get("has_rel_motion", False)))

    return {
        "num_observations": int(num_observations),
        "frame_start": int(min(frame_indices)) if frame_indices else -1,
        "frame_end": int(max(frame_indices)) if frame_indices else -1,
        "frame_span": int(frame_span),
        "video_num_frames": int(video_num_frames),
        "temporal_coverage_in_span": float(num_observations / max(1, frame_span)),
        "temporal_coverage_in_video": float(num_observations / max(1, int(video_num_frames))),
        "num_temporal_gaps": int(sum(1 for gap in frame_gaps if gap > 1)),
        "max_frame_gap": int(max(frame_gaps) if frame_gaps else 0),
        "mean_frame_gap": float(sum(frame_gaps) / len(frame_gaps)) if frame_gaps else 0.0,
        "has_motion_ratio": float(rel_motion_count / max(1, num_observations)),
        "label_counts": dict(sorted(label_counts.items())),
        "primary_label": label_counts.most_common(1)[0][0] if label_counts else "unknown",
        "source_counts": dict(sorted(source_counts.items())),
        "observed_count": int(source_counts.get("observed", 0)),
        "repaired_count": int(source_counts.get("repaired", 0)),
        "merged_count": int(source_counts.get("merged", 0)),
        "observed_ratio": float(source_counts.get("observed", 0) / max(1, num_observations)),
        "repaired_ratio": float(source_counts.get("repaired", 0) / max(1, num_observations)),
        "merged_ratio": float(source_counts.get("merged", 0) / max(1, num_observations)),
        "position_x": _numeric_stats([pos[0] for pos in valid_positions]),
        "position_y": _numeric_stats([pos[1] for pos in valid_positions]),
        "position_z_depth": _numeric_stats([pos[2] for pos in valid_positions]),
        "depth_change": _numeric_stats([motion.get("distance_meters", 0.0) for motion in motions]),
        "path_length_3d": float(path_length_3d),
        "path_length_xz": float(path_length_xz),
        "displacement_3d": float(displacement_3d),
        "displacement_xz": float(displacement_xz),
        "bbox_width": _numeric_stats([row["width"] for row in bbox_rows]),
        "bbox_height": _numeric_stats([row["height"] for row in bbox_rows]),
        "bbox_area": _numeric_stats([row["area"] for row in bbox_rows]),
        "bbox_center_x": _numeric_stats([row["center_x"] for row in bbox_rows]),
        "bbox_center_y": _numeric_stats([row["center_y"] for row in bbox_rows]),
        "bbox_center_path_px": float(center_path_px),
        "bbox_center_displacement_px": float(center_displacement_px),
        "obj_vx": _numeric_stats([motion.get("obj_vx", 0.0) for motion in motions if bool(motion.get("has_rel_motion", False))]),
        "obj_vz": _numeric_stats([motion.get("obj_vz", 0.0) for motion in motions if bool(motion.get("has_rel_motion", False))]),
        "rel_vx": _numeric_stats([motion.get("rel_vx", 0.0) for motion in motions if bool(motion.get("has_rel_motion", False))]),
        "rel_vz": _numeric_stats([motion.get("rel_vz", 0.0) for motion in motions if bool(motion.get("has_rel_motion", False))]),
        "rel_speed": _numeric_stats([motion.get("rel_speed", 0.0) for motion in motions if bool(motion.get("has_rel_motion", False))]),
        "motion_state_counts": dict(sorted(motion_state_counts.items())),
    }


def _trajectory_uncertainty(observations, statistics):
    scores = [obs.get("uncertainty", {}).get("score", 0.0) for obs in observations]
    source_uncertainties = [obs.get("uncertainty", {}).get("source_uncertainty", 0.0) for obs in observations]
    score_stats = _numeric_stats(scores)
    repaired_ratio = _safe_float(statistics.get("repaired_ratio", 0.0))
    merged_ratio = _safe_float(statistics.get("merged_ratio", 0.0))
    missing_motion_ratio = 1.0 - _safe_float(statistics.get("has_motion_ratio", 0.0))
    gap_penalty = min(1.0, _safe_float(statistics.get("num_temporal_gaps", 0)) / max(1.0, _safe_float(statistics.get("num_observations", 1))))
    low_score_penalty = max(0.0, 1.0 - _safe_float(score_stats.get("mean", 0.0))) if score_stats.get("count", 0) else 0.3
    source_uncertainty_mean = _safe_float(_numeric_stats(source_uncertainties).get("mean", 0.0))
    uncertainty_score = min(
        1.0,
        0.25 * repaired_ratio
        + 0.15 * merged_ratio
        + 0.25 * missing_motion_ratio
        + 0.15 * gap_penalty
        + 0.1 * low_score_penalty
        + 0.1 * source_uncertainty_mean,
    )
    return {
        "score_stats": score_stats,
        "repaired_ratio": float(repaired_ratio),
        "merged_ratio": float(merged_ratio),
        "missing_motion_ratio": float(missing_motion_ratio),
        "temporal_gap_penalty": float(gap_penalty),
        "source_uncertainty_mean": float(source_uncertainty_mean),
        "uncertainty_score": float(uncertainty_score),
        "confidence_score": float(max(0.0, 1.0 - uncertainty_score)),
        "notes": [
            "Scores are detector/interpolation confidence proxies when available.",
            "Uncertainty is heuristic and intended for causal fact validation, not final filtering.",
        ],
    }


def _ratio_larger_to_smaller(value_a, value_b):
    a = max(1e-6, abs(_safe_float(value_a)))
    b = max(1e-6, abs(_safe_float(value_b)))
    return float(max(a, b) / min(a, b))


def _trajectory_step_metrics(observations):
    metrics = {
        "frame_gaps": [],
        "bbox_center_step_px_per_frame": [],
        "bbox_center_step_diag_ratio": [],
        "bbox_width_ratio": [],
        "bbox_height_ratio": [],
        "bbox_area_ratio": [],
        "depth_step_per_frame": [],
        "ego_compensated_depth_step_per_frame": [],
        "ego_minus_depth_step_per_frame": [],
        "ego_plus_depth_step_per_frame": [],
        "position_xz_step_per_frame": [],
        "rel_velocity_delta": [],
        "rel_speed_delta": [],
        "ego_minus_velocity_delta": [],
        "ego_plus_velocity_delta": [],
        "ego_minus_speed": [],
        "ego_plus_speed": [],
        "direction_reversal_count": 0,
    }
    ordered = sorted(observations, key=lambda row: int(row.get("frame_index", -1)))
    for obs in ordered:
        motion = dict(obs.get("motion", {}))
        if not bool(motion.get("has_rel_motion", False)):
            continue
        obj_vx = _safe_float(motion.get("obj_vx", 0.0))
        obj_vz = _safe_float(motion.get("obj_vz", 0.0))
        ego_vx = _safe_float(motion.get("ego_vx", 0.0))
        ego_vz = _safe_float(motion.get("ego_vz", 0.0))
        metrics["ego_minus_speed"].append(float(math.hypot(obj_vx - ego_vx, obj_vz - ego_vz)))
        metrics["ego_plus_speed"].append(float(math.hypot(obj_vx + ego_vx, obj_vz + ego_vz)))

    for left, right in zip(ordered, ordered[1:]):
        left_frame = int(left.get("frame_index", -1))
        right_frame = int(right.get("frame_index", -1))
        frame_gap = max(1, right_frame - left_frame)
        metrics["frame_gaps"].append(frame_gap)

        left_bbox = _bbox_features(left.get("bbox", []))
        right_bbox = _bbox_features(right.get("bbox", []))
        center_step = math.hypot(
            right_bbox["center_x"] - left_bbox["center_x"],
            right_bbox["center_y"] - left_bbox["center_y"],
        ) / float(frame_gap)
        avg_diag = max(1.0, (left_bbox["diag"] + right_bbox["diag"]) / 2.0)
        metrics["bbox_center_step_px_per_frame"].append(float(center_step))
        metrics["bbox_center_step_diag_ratio"].append(float(center_step / avg_diag))
        metrics["bbox_width_ratio"].append(_ratio_larger_to_smaller(left_bbox["width"], right_bbox["width"]))
        metrics["bbox_height_ratio"].append(_ratio_larger_to_smaller(left_bbox["height"], right_bbox["height"]))
        metrics["bbox_area_ratio"].append(_ratio_larger_to_smaller(left_bbox["area"], right_bbox["area"]))

        left_pos = list(left.get("position_3d", []))
        right_pos = list(right.get("position_3d", []))
        if len(left_pos) >= 3 and len(right_pos) >= 3:
            dx = _safe_float(right_pos[0]) - _safe_float(left_pos[0])
            dz = _safe_float(right_pos[2]) - _safe_float(left_pos[2])
            dz_per_frame = dz / float(frame_gap)
            right_motion_for_ego = dict(right.get("motion", {}))
            ego_vz = _safe_float(right_motion_for_ego.get("ego_vz", 0.0))
            depth_step = float(abs(dz_per_frame))
            ego_minus_depth_step = float(abs(dz_per_frame - ego_vz))
            ego_plus_depth_step = float(abs(dz_per_frame + ego_vz))
            metrics["depth_step_per_frame"].append(depth_step)
            metrics["ego_minus_depth_step_per_frame"].append(ego_minus_depth_step)
            metrics["ego_plus_depth_step_per_frame"].append(ego_plus_depth_step)
            metrics["ego_compensated_depth_step_per_frame"].append(
                float(min(depth_step, ego_minus_depth_step, ego_plus_depth_step))
            )
            metrics["position_xz_step_per_frame"].append(float(math.hypot(dx, dz) / frame_gap))

        left_motion = dict(left.get("motion", {}))
        right_motion = dict(right.get("motion", {}))
        if bool(left_motion.get("has_rel_motion", False)) and bool(right_motion.get("has_rel_motion", False)):
            left_v = (_safe_float(left_motion.get("rel_vx", 0.0)), _safe_float(left_motion.get("rel_vz", 0.0)))
            right_v = (_safe_float(right_motion.get("rel_vx", 0.0)), _safe_float(right_motion.get("rel_vz", 0.0)))
            left_speed = math.hypot(left_v[0], left_v[1])
            right_speed = math.hypot(right_v[0], right_v[1])
            metrics["rel_velocity_delta"].append(float(math.hypot(right_v[0] - left_v[0], right_v[1] - left_v[1])))
            metrics["rel_speed_delta"].append(float(abs(right_speed - left_speed)))
            left_obj_v = (_safe_float(left_motion.get("obj_vx", 0.0)), _safe_float(left_motion.get("obj_vz", 0.0)))
            right_obj_v = (_safe_float(right_motion.get("obj_vx", 0.0)), _safe_float(right_motion.get("obj_vz", 0.0)))
            left_ego_v = (_safe_float(left_motion.get("ego_vx", 0.0)), _safe_float(left_motion.get("ego_vz", 0.0)))
            right_ego_v = (_safe_float(right_motion.get("ego_vx", 0.0)), _safe_float(right_motion.get("ego_vz", 0.0)))
            left_minus_v = (left_obj_v[0] - left_ego_v[0], left_obj_v[1] - left_ego_v[1])
            right_minus_v = (right_obj_v[0] - right_ego_v[0], right_obj_v[1] - right_ego_v[1])
            left_plus_v = (left_obj_v[0] + left_ego_v[0], left_obj_v[1] + left_ego_v[1])
            right_plus_v = (right_obj_v[0] + right_ego_v[0], right_obj_v[1] + right_ego_v[1])
            metrics["ego_minus_velocity_delta"].append(
                float(math.hypot(right_minus_v[0] - left_minus_v[0], right_minus_v[1] - left_minus_v[1]))
            )
            metrics["ego_plus_velocity_delta"].append(
                float(math.hypot(right_plus_v[0] - left_plus_v[0], right_plus_v[1] - left_plus_v[1]))
            )
            if left_speed > _REL_SPEED_THRESHOLD and right_speed > _REL_SPEED_THRESHOLD:
                dot = left_v[0] * right_v[0] + left_v[1] * right_v[1]
                if dot < 0.0:
                    metrics["direction_reversal_count"] += 1
    return metrics


def _trajectory_validation_velocity_profile(step_metrics, statistics):
    legacy_max_speed = _safe_float(dict(statistics.get("rel_speed", {})).get("max", 0.0))
    legacy_max_delta = max(step_metrics["rel_velocity_delta"]) if step_metrics["rel_velocity_delta"] else 0.0
    profiles = [
        {
            "name": "ego_minus",
            "description": "existing step 8 residual: obj_v - ego_v",
            "max_speed": float(max(step_metrics["ego_minus_speed"]) if step_metrics["ego_minus_speed"] else legacy_max_speed),
            "mean_speed": float(
                sum(step_metrics["ego_minus_speed"]) / len(step_metrics["ego_minus_speed"])
                if step_metrics["ego_minus_speed"]
                else legacy_max_speed
            ),
            "max_velocity_delta": float(max(step_metrics["ego_minus_velocity_delta"]) if step_metrics["ego_minus_velocity_delta"] else legacy_max_delta),
        },
        {
            "name": "ego_plus",
            "description": "reverse/physical ego residual: obj_v + ego_v",
            "max_speed": float(max(step_metrics["ego_plus_speed"]) if step_metrics["ego_plus_speed"] else legacy_max_speed),
            "mean_speed": float(
                sum(step_metrics["ego_plus_speed"]) / len(step_metrics["ego_plus_speed"])
                if step_metrics["ego_plus_speed"]
                else legacy_max_speed
            ),
            "max_velocity_delta": float(max(step_metrics["ego_plus_velocity_delta"]) if step_metrics["ego_plus_velocity_delta"] else legacy_max_delta),
        },
    ]
    best = min(profiles, key=lambda row: (row["max_speed"], row["max_velocity_delta"]))
    return {
        "selected_profile": best["name"],
        "selected_description": best["description"],
        "max_speed": float(best["max_speed"]),
        "max_velocity_delta": float(best["max_velocity_delta"]),
        "profiles": profiles,
        "legacy_rel_speed_max": float(legacy_max_speed),
        "legacy_rel_velocity_delta_max": float(legacy_max_delta),
        "notes": (
            "Validation uses the lower residual across ego sign conventions so reverse ego motion "
            "does not by itself invalidate otherwise continuous tracks."
        ),
    }


def _validation_issue(kind, severity, message, value=None, threshold=None):
    issue = {
        "kind": str(kind),
        "severity": str(severity),
        "message": str(message),
    }
    if value is not None:
        issue["value"] = value
    if threshold is not None:
        issue["threshold"] = threshold
    return issue


def _trajectory_reality_validation(observations, statistics, uncertainty):
    thresholds = dict(_TRAJECTORY_VALIDATION_THRESHOLDS)
    step_metrics = _trajectory_step_metrics(observations)
    issues = []
    label_counts = dict(statistics.get("label_counts", {}))
    num_observations = int(statistics.get("num_observations", len(observations)))
    max_frame_gap = int(statistics.get("max_frame_gap", 0))
    has_motion_ratio = _safe_float(statistics.get("has_motion_ratio", 0.0))
    repaired_count = int(statistics.get("repaired_count", 0))
    merged_count = int(statistics.get("merged_count", 0))

    if num_observations < 2:
        issues.append(_validation_issue("trajectory_too_short", "uncertain", "Only one observation; continuity cannot be verified.", num_observations, 2))
    if len(label_counts) > 1:
        issues.append(_validation_issue("id_switch", "invalid", "Track contains multiple frame-level labels.", label_counts))
    if max_frame_gap > int(thresholds["max_valid_frame_gap"]):
        issues.append(_validation_issue("trajectory_discontinuity", "invalid", "Large frame gap inside trajectory.", max_frame_gap, thresholds["max_valid_frame_gap"]))
    elif max_frame_gap > int(thresholds["max_uncertain_frame_gap"]):
        issues.append(_validation_issue("trajectory_discontinuity", "uncertain", "Non-consecutive trajectory observations.", max_frame_gap, thresholds["max_uncertain_frame_gap"]))
    if has_motion_ratio < float(thresholds["min_motion_ratio"]):
        issues.append(_validation_issue("insufficient_motion_evidence", "uncertain", "Too few observations have relative motion.", has_motion_ratio, thresholds["min_motion_ratio"]))

    max_center_ratio = max(step_metrics["bbox_center_step_diag_ratio"]) if step_metrics["bbox_center_step_diag_ratio"] else 0.0
    if max_center_ratio > float(thresholds["max_invalid_center_step_diag_ratio"]):
        issues.append(_validation_issue("track_drift", "invalid", "BBox center jump is too large relative to object size.", max_center_ratio, thresholds["max_invalid_center_step_diag_ratio"]))
    elif max_center_ratio > float(thresholds["max_uncertain_center_step_diag_ratio"]):
        issues.append(_validation_issue("track_drift", "uncertain", "BBox center motion is high relative to object size.", max_center_ratio, thresholds["max_uncertain_center_step_diag_ratio"]))

    max_bbox_ratio = max(
        step_metrics["bbox_width_ratio"] + step_metrics["bbox_height_ratio"] + step_metrics["bbox_area_ratio"]
    ) if (step_metrics["bbox_width_ratio"] or step_metrics["bbox_height_ratio"] or step_metrics["bbox_area_ratio"]) else 1.0
    if max_bbox_ratio > float(thresholds["max_invalid_bbox_size_ratio"]):
        issues.append(_validation_issue("bbox_jump", "invalid", "BBox size or area changes abruptly.", max_bbox_ratio, thresholds["max_invalid_bbox_size_ratio"]))
    elif max_bbox_ratio > float(thresholds["max_uncertain_bbox_size_ratio"]):
        issues.append(_validation_issue("bbox_jump", "uncertain", "BBox size or area change is high.", max_bbox_ratio, thresholds["max_uncertain_bbox_size_ratio"]))

    raw_max_depth_step = max(step_metrics["depth_step_per_frame"]) if step_metrics["depth_step_per_frame"] else 0.0
    max_depth_step = (
        max(step_metrics["ego_compensated_depth_step_per_frame"])
        if step_metrics["ego_compensated_depth_step_per_frame"]
        else raw_max_depth_step
    )
    if max_depth_step > float(thresholds["max_invalid_depth_step_per_frame"]):
        issues.append(
            _validation_issue(
                "depth_jump",
                "invalid",
                "Ego-compensated depth changes too abruptly.",
                {"ego_compensated": max_depth_step, "raw": raw_max_depth_step},
                thresholds["max_invalid_depth_step_per_frame"],
            )
        )
    elif max_depth_step > float(thresholds["max_uncertain_depth_step_per_frame"]):
        issues.append(
            _validation_issue(
                "depth_jump",
                "uncertain",
                "Ego-compensated depth change is high.",
                {"ego_compensated": max_depth_step, "raw": raw_max_depth_step},
                thresholds["max_uncertain_depth_step_per_frame"],
            )
        )

    velocity_profile = _trajectory_validation_velocity_profile(step_metrics, statistics)
    max_velocity_delta = _safe_float(velocity_profile.get("max_velocity_delta", 0.0))
    max_rel_speed = _safe_float(velocity_profile.get("max_speed", 0.0))
    if max_velocity_delta > float(thresholds["max_invalid_rel_velocity_delta"]) or max_rel_speed > float(thresholds["max_invalid_rel_speed"]):
        issues.append(
            _validation_issue(
                "speed_abnormal_change",
                "invalid",
                "Ego-compensated velocity delta or residual speed is too large.",
                {
                    "max_validation_velocity_delta": max_velocity_delta,
                    "max_validation_speed": max_rel_speed,
                    "selected_profile": velocity_profile["selected_profile"],
                    "legacy_max_rel_velocity_delta": velocity_profile["legacy_rel_velocity_delta_max"],
                    "legacy_max_rel_speed": velocity_profile["legacy_rel_speed_max"],
                },
                {"max_rel_velocity_delta": thresholds["max_invalid_rel_velocity_delta"], "max_rel_speed": thresholds["max_invalid_rel_speed"]},
            )
        )
    elif max_velocity_delta > float(thresholds["max_uncertain_rel_velocity_delta"]) or max_rel_speed > float(thresholds["max_uncertain_rel_speed"]):
        issues.append(
            _validation_issue(
                "speed_abnormal_change",
                "uncertain",
                "Ego-compensated velocity delta or residual speed is high.",
                {
                    "max_validation_velocity_delta": max_velocity_delta,
                    "max_validation_speed": max_rel_speed,
                    "selected_profile": velocity_profile["selected_profile"],
                    "legacy_max_rel_velocity_delta": velocity_profile["legacy_rel_velocity_delta_max"],
                    "legacy_max_rel_speed": velocity_profile["legacy_rel_speed_max"],
                },
                {"max_rel_velocity_delta": thresholds["max_uncertain_rel_velocity_delta"], "max_rel_speed": thresholds["max_uncertain_rel_speed"]},
            )
        )

    direction_reversals = int(step_metrics["direction_reversal_count"])
    if direction_reversals >= 2:
        issues.append(_validation_issue("motion_direction_abrupt_change", "invalid", "Multiple relative motion direction reversals.", direction_reversals, 2))
    elif direction_reversals == 1:
        issues.append(_validation_issue("motion_direction_abrupt_change", "uncertain", "One relative motion direction reversal.", direction_reversals, 1))

    invalid_issues = [issue for issue in issues if issue["severity"] == "invalid"]
    uncertain_issues = [issue for issue in issues if issue["severity"] == "uncertain"]
    if invalid_issues:
        status = "invalid"
    elif uncertain_issues:
        status = "uncertain"
    elif repaired_count > 0 or merged_count > 0:
        status = "repaired"
    else:
        status = "valid"

    checks = {
        "id_switch": {"passed": len(label_counts) <= 1, "label_counts": label_counts},
        "trajectory_discontinuity": {"passed": max_frame_gap <= int(thresholds["max_uncertain_frame_gap"]), "max_frame_gap": max_frame_gap},
        "track_drift": {"passed": max_center_ratio <= float(thresholds["max_uncertain_center_step_diag_ratio"]), "max_bbox_center_step_diag_ratio": float(max_center_ratio)},
        "bbox_depth_jump": {
            "passed": (
                max_bbox_ratio <= float(thresholds["max_uncertain_bbox_size_ratio"])
                and max_depth_step <= float(thresholds["max_uncertain_depth_step_per_frame"])
            ),
            "max_bbox_size_ratio": float(max_bbox_ratio),
            "max_depth_step_per_frame": float(max_depth_step),
            "raw_max_depth_step_per_frame": float(raw_max_depth_step),
        },
        "motion_direction_abrupt_change": {"passed": direction_reversals == 0, "direction_reversal_count": direction_reversals},
        "speed_abnormal_change": {
            "passed": (
                max_velocity_delta <= float(thresholds["max_uncertain_rel_velocity_delta"])
                and max_rel_speed <= float(thresholds["max_uncertain_rel_speed"])
            ),
            "max_validation_velocity_delta": float(max_velocity_delta),
            "max_validation_speed": float(max_rel_speed),
            "selected_velocity_profile": velocity_profile["selected_profile"],
            "legacy_max_rel_velocity_delta": float(velocity_profile["legacy_rel_velocity_delta_max"]),
            "legacy_max_rel_speed": float(velocity_profile["legacy_rel_speed_max"]),
        },
        "motion_evidence": {"passed": has_motion_ratio >= float(thresholds["min_motion_ratio"]), "has_motion_ratio": float(has_motion_ratio)},
    }
    return {
        "status": status,
        "validation_status": status,
        "valid": status in {"valid", "repaired"},
        "repaired": status == "repaired",
        "uncertain": status == "uncertain",
        "invalid": status == "invalid",
        "issues": issues,
        "rejection_reasons": [issue["kind"] for issue in invalid_issues],
        "uncertain_reasons": [issue["kind"] for issue in uncertain_issues],
        "checks": checks,
        "step_metrics": {
            "max_bbox_center_step_px_per_frame": float(max(step_metrics["bbox_center_step_px_per_frame"]) if step_metrics["bbox_center_step_px_per_frame"] else 0.0),
            "max_bbox_center_step_diag_ratio": float(max_center_ratio),
            "max_bbox_size_ratio": float(max_bbox_ratio),
            "max_depth_step_per_frame": float(max_depth_step),
            "raw_max_depth_step_per_frame": float(raw_max_depth_step),
            "max_rel_velocity_delta": float(max_velocity_delta),
            "max_rel_speed": float(max_rel_speed),
            "legacy_max_rel_velocity_delta": float(velocity_profile["legacy_rel_velocity_delta_max"]),
            "legacy_max_rel_speed": float(velocity_profile["legacy_rel_speed_max"]),
            "direction_reversal_count": direction_reversals,
        },
        "ego_motion_consistency": {
            "selected_velocity_profile": velocity_profile["selected_profile"],
            "velocity_profiles": velocity_profile["profiles"],
            "raw_max_depth_step_per_frame": float(raw_max_depth_step),
            "max_ego_compensated_depth_step_per_frame": float(max_depth_step),
            "notes": velocity_profile["notes"],
        },
        "thresholds": thresholds,
        "notes": "Heuristic continuity validation for causal motion fact validation, with ego-aware reverse-motion handling.",
    }


def _motion_significance_assessment(statistics, provenance, uncertainty, validation):
    thresholds = dict(_MOTION_SIGNIFICANCE_THRESHOLDS)
    reasons = []
    supporting_metrics = {}

    validation_status = str(validation.get("validation_status", validation.get("status", "uncertain")))
    num_observations = int(statistics.get("num_observations", 0))
    has_motion_ratio = _safe_float(statistics.get("has_motion_ratio", 0.0))
    repaired_ratio = _safe_float(provenance.get("repaired_ratio", statistics.get("repaired_ratio", 0.0)))
    uncertainty_score = _safe_float(uncertainty.get("uncertainty_score", 1.0), 1.0)
    rel_speed_mean = _safe_float(dict(statistics.get("rel_speed", {})).get("mean", 0.0))
    rel_speed_max = _safe_float(dict(statistics.get("rel_speed", {})).get("max", 0.0))
    path_length_xz = _safe_float(statistics.get("path_length_xz", 0.0))
    displacement_xz = _safe_float(statistics.get("displacement_xz", 0.0))
    depth_abs_delta = _safe_float(dict(statistics.get("position_z_depth", {})).get("abs_delta", 0.0))
    bbox_center_path_px = _safe_float(statistics.get("bbox_center_path_px", 0.0))
    position_xz_step_mean = _safe_float(dict(statistics.get("position_x", {})).get("mean_abs_step", 0.0)) + _safe_float(
        dict(statistics.get("position_z_depth", {})).get("mean_abs_step", 0.0)
    )

    supporting_metrics.update(
        {
            "validation_status": validation_status,
            "num_observations": num_observations,
            "has_motion_ratio": float(has_motion_ratio),
            "repaired_ratio": float(repaired_ratio),
            "uncertainty_score": float(uncertainty_score),
            "rel_speed_mean": float(rel_speed_mean),
            "rel_speed_max": float(rel_speed_max),
            "path_length_xz": float(path_length_xz),
            "displacement_xz": float(displacement_xz),
            "depth_abs_delta": float(depth_abs_delta),
            "bbox_center_path_px": float(bbox_center_path_px),
            "position_xz_step_mean_proxy": float(position_xz_step_mean),
        }
    )

    if validation_status in {"invalid", "uncertain"}:
        reasons.append(
            {
                "kind": "motion_not_reliably_validated",
                "message": "Trajectory reality validation is not stable enough for high-significance motion facts.",
                "value": validation_status,
            }
        )
    if num_observations < int(thresholds["min_observations"]):
        reasons.append(
            {
                "kind": "extremely_short_trajectory",
                "message": "Trajectory has too few observations to support a stable motion fact.",
                "value": num_observations,
                "threshold": int(thresholds["min_observations"]),
            }
        )
    if has_motion_ratio < float(thresholds["min_has_motion_ratio"]):
        reasons.append(
            {
                "kind": "motion_unstable_or_missing",
                "message": "Too few observations have usable relative motion.",
                "value": float(has_motion_ratio),
                "threshold": float(thresholds["min_has_motion_ratio"]),
            }
        )
    if repaired_ratio > float(thresholds["max_repaired_ratio"]):
        reasons.append(
            {
                "kind": "mostly_interpolated",
                "message": "Most of the trajectory comes from repaired/interpolated observations.",
                "value": float(repaired_ratio),
                "threshold": float(thresholds["max_repaired_ratio"]),
            }
        )
    if uncertainty_score > float(thresholds["max_uncertainty_score"]):
        reasons.append(
            {
                "kind": "high_uncertainty",
                "message": "Trajectory uncertainty is too high for a high-significance motion fact.",
                "value": float(uncertainty_score),
                "threshold": float(thresholds["max_uncertainty_score"]),
            }
        )

    near_static = (
        rel_speed_mean < float(thresholds["min_rel_speed_mean"])
        and rel_speed_max < float(thresholds["min_rel_speed_max"])
        and path_length_xz < float(thresholds["min_path_length_xz"])
        and displacement_xz < float(thresholds["min_displacement_xz"])
        and depth_abs_delta < float(thresholds["min_depth_abs_delta"])
        and bbox_center_path_px < float(thresholds["min_bbox_center_path_px"])
    )
    below_noise = (
        rel_speed_mean < float(thresholds["noise_rel_speed"])
        and position_xz_step_mean < float(thresholds["noise_position_xz_step"])
    )
    if near_static:
        reasons.append(
            {
                "kind": "nearly_static",
                "message": "Trajectory motion is close to static across 3D, relative speed, and bbox evidence.",
            }
        )
    if below_noise:
        reasons.append(
            {
                "kind": "below_estimated_noise",
                "message": "Motion magnitude is below the configured noise floor.",
            }
        )

    high_motion_signal = (
        rel_speed_mean >= float(thresholds["min_rel_speed_mean"])
        or rel_speed_max >= float(thresholds["min_rel_speed_max"])
        or path_length_xz >= float(thresholds["min_path_length_xz"])
        or displacement_xz >= float(thresholds["min_displacement_xz"])
        or depth_abs_delta >= float(thresholds["min_depth_abs_delta"])
        or bbox_center_path_px >= float(thresholds["min_bbox_center_path_px"])
    )
    significance = "high_significance" if not reasons and high_motion_signal else "low_significance"
    return {
        "significance": significance,
        "is_high_significance": significance == "high_significance",
        "is_low_significance": significance == "low_significance",
        "reasons": reasons,
        "supporting_metrics": supporting_metrics,
        "thresholds": thresholds,
        "notes": "Motion significance is a label for information content; it does not remove trajectories.",
    }


def _fact_decision_for_trajectory(validation, significance, provenance, uncertainty):
    validation_status = str(validation.get("validation_status", validation.get("status", "uncertain")))
    motion_significance = str(significance.get("significance", "low_significance"))
    confidence_score = _safe_float(uncertainty.get("confidence_score", 0.0), 0.0)
    repaired_count = int(provenance.get("repaired_count", 0))
    merged_count = int(provenance.get("merged_count", 0))
    reasons = []

    if validation_status == "invalid":
        decision = "Discard"
        symbolic_layer_eligible = False
        reasons.append(
            {
                "kind": "invalid_trajectory",
                "message": "Trajectory failed reality validation and should not enter the symbolic layer.",
                "validation_reasons": list(validation.get("rejection_reasons", [])),
            }
        )
    elif validation_status == "repaired" or repaired_count > 0 or merged_count > 0:
        decision = "Repair"
        symbolic_layer_eligible = True
        reasons.append(
            {
                "kind": "repaired_trajectory_kept",
                "message": "Trajectory contains repaired or merged evidence and is retained with repair provenance.",
                "repaired_count": repaired_count,
                "merged_count": merged_count,
            }
        )
        if motion_significance == "low_significance":
            reasons.append(
                {
                    "kind": "low_motion_significance",
                    "message": "Trajectory is retained after repair but carries a low motion-significance label.",
                    "significance_reasons": [row.get("kind", "") for row in significance.get("reasons", [])],
                }
            )
    elif validation_status == "valid" and motion_significance == "high_significance":
        decision = "Keep"
        symbolic_layer_eligible = True
        reasons.append(
            {
                "kind": "valid_high_significance",
                "message": "Trajectory is realistic and has enough motion information.",
            }
        )
    elif validation_status == "valid" and motion_significance == "low_significance":
        decision = "Keep"
        symbolic_layer_eligible = True
        reasons.append(
            {
                "kind": "valid_low_motion_retained",
                "message": (
                    "Trajectory is realistic and retained even though its motion is low; "
                    "static objects remain valid facts and can support ego-motion refinement."
                ),
                "significance_reasons": [row.get("kind", "") for row in significance.get("reasons", [])],
            }
        )
    else:
        decision = "Keep with uncertainty"
        symbolic_layer_eligible = True
        reasons.append(
            {
                "kind": "credible_but_uncertain",
                "message": "Trajectory is not clearly invalid, but validation/significance/uncertainty is not strong enough for a plain Keep decision.",
                "validation_status": validation_status,
                "motion_significance": motion_significance,
                "confidence_score": confidence_score,
            }
        )
        if validation.get("uncertain_reasons"):
            reasons.append(
                {
                    "kind": "validation_uncertainty",
                    "message": "Trajectory reality validation reported uncertainty.",
                    "uncertain_reasons": list(validation.get("uncertain_reasons", [])),
                }
            )
        if significance.get("reasons"):
            reasons.append(
                {
                    "kind": "significance_uncertainty",
                    "message": "Motion significance assessment reported low-information or unstable motion evidence.",
                    "significance_reasons": [row.get("kind", "") for row in significance.get("reasons", [])],
                }
            )

    return {
        "decision": decision,
        "status": decision,
        "symbolic_layer_eligible": bool(symbolic_layer_eligible),
        "decision_reasons": reasons,
        "provenance_summary": dict(provenance),
        "supporting_status": {
            "validation_status": validation_status,
            "motion_significance": motion_significance,
            "confidence_score": float(confidence_score),
        },
        "notes": "Final 8B fact decision for symbolic-layer admission; it preserves provenance and reasons for explanation.",
    }


def _normalize_label_for_prior(label):
    return str(label).strip().lower().replace("-", " ").replace("_", " ")


def _expected_motion_from_prior(label):
    normalized = _normalize_label_for_prior(label)
    for prior_label, expected_motion in _STATIC_OBJECT_PRIOR.items():
        if _normalize_label_for_prior(prior_label) == normalized or _normalize_label_for_prior(prior_label) in normalized:
            return expected_motion
    for prior_label, expected_motion in _LOW_DYNAMIC_OBJECT_PRIOR.items():
        if _normalize_label_for_prior(prior_label) == normalized or _normalize_label_for_prior(prior_label) in normalized:
            return expected_motion
    return "dynamic"


def _reference_object_candidate(trajectory):
    thresholds = dict(_REFERENCE_OBJECT_THRESHOLDS)
    label = str(trajectory.get("primary_label", "unknown"))
    expected_motion = _expected_motion_from_prior(label)
    statistics = dict(trajectory.get("trajectory_statistics", {}))
    validation = dict(trajectory.get("causal_motion_fact_validation", {}))
    uncertainty = dict(trajectory.get("uncertainty", {}))
    provenance = dict(trajectory.get("provenance", {}))
    significance = dict(trajectory.get("motion_significance_assessment", {}))
    fact_decision = dict(trajectory.get("fact_decision", {}))

    observation_ratio = _safe_float(statistics.get("temporal_coverage_in_video", statistics.get("temporal_coverage_in_span", 0.0)))
    uncertainty_score = _safe_float(uncertainty.get("uncertainty_score", 1.0), 1.0)
    repaired_ratio = _safe_float(provenance.get("repaired_ratio", 0.0))
    rel_speed_mean = _safe_float(dict(statistics.get("rel_speed", {})).get("mean", 0.0))
    ego_motion_consistency = dict(validation.get("ego_motion_consistency", {}))
    selected_velocity_profile = str(ego_motion_consistency.get("selected_velocity_profile", "ego_minus"))
    velocity_profiles = list(ego_motion_consistency.get("velocity_profiles", []))
    selected_profile_metrics = next(
        (dict(row) for row in velocity_profiles if str(row.get("name", "")) == selected_velocity_profile),
        {},
    )
    prior_motion_speed_mean = _safe_float(selected_profile_metrics.get("mean_speed", rel_speed_mean), rel_speed_mean)
    prior_motion_speed_max = _safe_float(selected_profile_metrics.get("max_speed", dict(statistics.get("rel_speed", {})).get("max", 0.0)))
    depth_abs_delta = _safe_float(dict(statistics.get("position_z_depth", {})).get("abs_delta", 0.0))
    ego_compensated_depth_step = _safe_float(
        ego_motion_consistency.get(
            "max_ego_compensated_depth_step_per_frame",
            dict(validation.get("step_metrics", {})).get("max_depth_step_per_frame", 0.0),
        )
    )
    center_step_ratio = _safe_float(
        dict(validation.get("step_metrics", {})).get("max_bbox_center_step_diag_ratio", 0.0)
    )
    validation_status = str(validation.get("validation_status", validation.get("status", "uncertain")))
    symbolic_eligible = bool(fact_decision.get("symbolic_layer_eligible", trajectory.get("symbolic_layer_eligible", False)))
    max_rel_speed_mean = (
        float(thresholds["max_rel_speed_mean_static"])
        if expected_motion == "static"
        else float(thresholds["max_rel_speed_mean_low_dynamic"])
    )
    reasons = []
    disqualifiers = []

    if expected_motion not in {"static", "low_dynamic"}:
        disqualifiers.append("expected_motion_not_reference")
    if validation_status not in {"valid", "repaired"}:
        disqualifiers.append("trajectory_not_reliably_validated")
    if not symbolic_eligible:
        disqualifiers.append("not_symbolic_layer_eligible")
    if observation_ratio < float(thresholds["min_observation_ratio"]):
        disqualifiers.append("low_observation_ratio")
    if uncertainty_score > float(thresholds["max_uncertainty_score"]):
        disqualifiers.append("high_uncertainty")
    if repaired_ratio > float(thresholds["max_repaired_ratio"]):
        disqualifiers.append("mostly_repaired")
    if expected_motion == "low_dynamic" and prior_motion_speed_mean > max_rel_speed_mean:
        disqualifiers.append("too_dynamic_for_prior")
    if center_step_ratio > float(thresholds["max_bbox_center_step_diag_ratio"]):
        disqualifiers.append("bbox_motion_too_large_for_reference")

    if expected_motion in {"static", "low_dynamic"}:
        reasons.append(f"object_motion_prior={expected_motion}")
    if validation_status in {"valid", "repaired"}:
        reasons.append(f"trajectory_valid={validation_status}")
    if observation_ratio >= float(thresholds["min_observation_ratio"]):
        reasons.append("observation_ratio_high_enough")
    if uncertainty_score <= float(thresholds["max_uncertainty_score"]):
        reasons.append("uncertainty_low_enough")
    if expected_motion == "static" or prior_motion_speed_mean <= max_rel_speed_mean:
        reasons.append("relative_motion_consistent_with_prior")

    reference_score = 0.0
    reference_score += 0.25 if expected_motion == "static" else (0.15 if expected_motion == "low_dynamic" else 0.0)
    reference_score += 0.25 if validation_status == "valid" else (0.15 if validation_status == "repaired" else 0.0)
    reference_score += 0.2 * min(1.0, observation_ratio / max(1e-6, float(thresholds["min_observation_ratio"])))
    reference_score += 0.15 * max(0.0, 1.0 - uncertainty_score)
    reference_score += 0.1 * max(0.0, 1.0 - repaired_ratio)
    reference_score += 0.05 if str(significance.get("significance", "")) == "low_significance" else 0.0
    is_reference = not disqualifiers
    return {
        "track_id": int(trajectory.get("track_id", -1)),
        "label": label,
        "expected_motion": expected_motion,
        "is_reliable_reference": bool(is_reference),
        "reference_score": float(reference_score),
        "selection_reasons": reasons,
        "disqualifiers": disqualifiers,
        "metrics": {
            "observation_ratio": float(observation_ratio),
            "uncertainty_score": float(uncertainty_score),
            "repaired_ratio": float(repaired_ratio),
            "rel_speed_mean": float(rel_speed_mean),
            "prior_motion_speed_mean": float(prior_motion_speed_mean),
            "prior_motion_speed_max": float(prior_motion_speed_max),
            "selected_velocity_profile": selected_velocity_profile,
            "depth_abs_delta": float(depth_abs_delta),
            "ego_compensated_depth_step_per_frame": float(ego_compensated_depth_step),
            "max_bbox_center_step_diag_ratio": float(center_step_ratio),
            "validation_status": validation_status,
            "motion_significance": str(trajectory.get("motion_significance", "")),
            "fact_decision_status": str(trajectory.get("fact_decision_status", "")),
        },
        "provenance": {
            **provenance,
            "fact_decision": fact_decision,
        },
    }


def _reliable_reference_objects_video(evidence_video):
    candidates = [
        _reference_object_candidate(trajectory)
        for trajectory in evidence_video.get("trajectory_motion_evidence", [])
    ]
    references = [
        candidate
        for candidate in candidates
        if bool(candidate.get("is_reliable_reference", False))
    ]
    references.sort(key=lambda row: (-_safe_float(row.get("reference_score", 0.0)), int(row.get("track_id", -1))))
    return {
        "version": _EGO_REFINEMENT_VERSION,
        "video_id": str(evidence_video.get("video_id", "")),
        "method": "prior_guided_reference_object_selection",
        "status": "reference_objects_selected",
        "description": (
            "Select reliable static or low-dynamic reference objects from 8B trajectory evidence. "
            "Ego motion refinement is not applied yet."
        ),
        "object_motion_prior": {
            "static_labels": sorted(_STATIC_OBJECT_PRIOR),
            "low_dynamic_labels": sorted(_LOW_DYNAMIC_OBJECT_PRIOR),
            "thresholds": dict(_REFERENCE_OBJECT_THRESHOLDS),
        },
        "num_trajectories": len(candidates),
        "num_reliable_reference_objects": len(references),
        "reliable_reference_objects": references,
        "candidate_reference_objects": candidates,
    }


def _median(values):
    vals = sorted(_safe_float(value) for value in values if math.isfinite(_safe_float(value)))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def _weighted_median(values, weights):
    pairs = sorted(
        (float(value), max(0.0, float(weight)))
        for value, weight in zip(values, weights)
        if math.isfinite(_safe_float(value)) and math.isfinite(_safe_float(weight)) and _safe_float(weight) > 0.0
    )
    if not pairs:
        return 0.0
    total_weight = sum(weight for _, weight in pairs)
    midpoint = total_weight / 2.0
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= midpoint:
            return float(value)
    return float(pairs[-1][0])


def _weighted_mean(values, weights):
    pairs = [
        (_safe_float(value), max(0.0, _safe_float(weight)))
        for value, weight in zip(values, weights)
        if math.isfinite(_safe_float(value)) and math.isfinite(_safe_float(weight)) and _safe_float(weight) > 0.0
    ]
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0.0:
        return 0.0
    return float(sum(value * weight for value, weight in pairs) / total_weight)


def _reference_vote_weight(reference, observation):
    expected_motion = str(reference.get("expected_motion", "dynamic"))
    prior_weight = 1.0 if expected_motion == "static" else 0.55
    reference_score = _safe_float(reference.get("reference_score", 0.0))
    obs_uncertainty = _safe_float(dict(observation.get("uncertainty", {})).get("source_uncertainty", 0.0))
    obs_score = _safe_float(dict(observation.get("uncertainty", {})).get("score", 0.0), 0.0)
    confidence = max(0.05, 1.0 - obs_uncertainty)
    if obs_score > 0.0:
        confidence *= max(0.1, obs_score)
    return float(max(0.0, prior_weight * max(0.05, reference_score) * confidence))


def _ego_vote_from_reference_motion(motion):
    obj_vx = _safe_float(motion.get("obj_vx", 0.0))
    obj_vz = _safe_float(motion.get("obj_vz", 0.0))
    ego_vx = _safe_float(motion.get("ego_vx", 0.0))
    ego_vz = _safe_float(motion.get("ego_vz", 0.0))
    same_sign_vote = (obj_vx, obj_vz)
    opposite_sign_vote = (-obj_vx, -obj_vz)
    same_residual = math.hypot(same_sign_vote[0] - ego_vx, same_sign_vote[1] - ego_vz)
    opposite_residual = math.hypot(opposite_sign_vote[0] - ego_vx, opposite_sign_vote[1] - ego_vz)
    if math.hypot(ego_vx, ego_vz) > _REL_SPEED_THRESHOLD and opposite_residual < same_residual:
        return {
            "ego_vx_vote": float(opposite_sign_vote[0]),
            "ego_vz_vote": float(opposite_sign_vote[1]),
            "ego_vote_sign_convention": "negative_object_velocity",
            "ego_vote_residual_to_original": float(opposite_residual),
        }
    return {
        "ego_vx_vote": float(same_sign_vote[0]),
        "ego_vz_vote": float(same_sign_vote[1]),
        "ego_vote_sign_convention": "object_velocity",
        "ego_vote_residual_to_original": float(same_residual),
    }


def _reference_votes_by_frame(evidence_video, reference_result):
    references_by_track = {
        int(row.get("track_id", -1)): dict(row)
        for row in reference_result.get("reliable_reference_objects", [])
        if int(row.get("track_id", -1)) >= 0
    }
    votes_by_frame = {}
    for trajectory in evidence_video.get("trajectory_motion_evidence", []):
        try:
            track_id = int(trajectory.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        reference = references_by_track.get(track_id)
        if reference is None:
            continue
        for obs in trajectory.get("trajectory_observations", []):
            motion = dict(obs.get("motion", {}))
            if not bool(motion.get("has_rel_motion", False)):
                continue
            frame_index = int(obs.get("frame_index", -1))
            if frame_index < 0:
                continue
            weight = _reference_vote_weight(reference, obs)
            ego_vote = _ego_vote_from_reference_motion(motion)
            votes_by_frame.setdefault(frame_index, []).append(
                {
                    "track_id": track_id,
                    "label": str(reference.get("label", trajectory.get("primary_label", "unknown"))),
                    "expected_motion": str(reference.get("expected_motion", "unknown")),
                    "reference_score": _safe_float(reference.get("reference_score", 0.0)),
                    "vote_weight": weight,
                    "ego_vx_vote": _safe_float(ego_vote.get("ego_vx_vote", 0.0)),
                    "ego_vz_vote": _safe_float(ego_vote.get("ego_vz_vote", 0.0)),
                    "ego_vote_sign_convention": str(ego_vote.get("ego_vote_sign_convention", "object_velocity")),
                    "ego_vote_residual_to_original": _safe_float(ego_vote.get("ego_vote_residual_to_original", 0.0)),
                    "source_frame_index": frame_index,
                    "observation_uncertainty": dict(obs.get("uncertainty", {})),
                    "selection_reasons": list(reference.get("selection_reasons", [])),
                }
            )
    return votes_by_frame


def _vote_agreement(values, estimate):
    vals = [_safe_float(value) for value in values if math.isfinite(_safe_float(value))]
    if not vals:
        return 0.0
    mad = _median([abs(value - estimate) for value in vals])
    return float(1.0 / (1.0 + mad))


def _refined_ego_motion_video(ego_video, evidence_video, reference_result):
    ego_frames = list((ego_video or {}).get("frames", []))
    if not ego_frames:
        ego_frames = [
            {"frame_index": idx, "ego_vx": 0.0, "ego_vz": 0.0, "ego_yaw_rate": 0.0, "has_ego_motion": False}
            for idx in range(int(evidence_video.get("num_frames", 0)))
        ]
    votes_by_frame = _reference_votes_by_frame(evidence_video, reference_result)
    frames_out = []
    for idx, ego_frame in enumerate(ego_frames):
        frame_index = int(ego_frame.get("frame_index", idx))
        original_vx, original_vz = _ego_vx_vz(ego_frame)
        votes = list(votes_by_frame.get(frame_index, []))
        if votes:
            vx_values = [vote["ego_vx_vote"] for vote in votes]
            vz_values = [vote["ego_vz_vote"] for vote in votes]
            weights = [vote["vote_weight"] for vote in votes]
            estimated_vx = _weighted_median(vx_values, weights)
            estimated_vz = _weighted_median(vz_values, weights)
            mean_weight = _weighted_mean(weights, [1.0 for _ in weights])
            support_factor = min(1.0, len(votes) / 3.0)
            weight_factor = min(1.0, sum(weights) / 1.5)
            agreement = (_vote_agreement(vx_values, estimated_vx) + _vote_agreement(vz_values, estimated_vz)) / 2.0
            correction_confidence = float(max(0.0, min(1.0, support_factor * weight_factor * agreement)))
            refined_vx = (1.0 - correction_confidence) * original_vx + correction_confidence * estimated_vx
            refined_vz = (1.0 - correction_confidence) * original_vz + correction_confidence * estimated_vz
        else:
            estimated_vx = original_vx
            estimated_vz = original_vz
            refined_vx = original_vx
            refined_vz = original_vz
            correction_confidence = 0.0
            agreement = 0.0
            mean_weight = 0.0
        frames_out.append(
            {
                **dict(ego_frame),
                "frame_index": frame_index,
                "original_ego_vx": float(original_vx),
                "original_ego_vz": float(original_vz),
                "reference_estimated_ego_vx": float(estimated_vx),
                "reference_estimated_ego_vz": float(estimated_vz),
                "refined_ego_vx": float(refined_vx),
                "refined_ego_vz": float(refined_vz),
                "ego_vx_correction": float(refined_vx - original_vx),
                "ego_vz_correction": float(refined_vz - original_vz),
                "correction_confidence": float(correction_confidence),
                "reference_vote_agreement": float(agreement),
                "reference_vote_mean_weight": float(mean_weight),
                "num_supporting_reference_objects": len(votes),
                "supporting_reference_objects": votes,
            }
        )
    confidence_values = [frame["correction_confidence"] for frame in frames_out]
    correction_magnitudes = [
        math.hypot(frame["ego_vx_correction"], frame["ego_vz_correction"])
        for frame in frames_out
    ]
    return {
        "version": _EGO_REFINEMENT_VERSION,
        "video_id": str((ego_video or {}).get("video_id", evidence_video.get("video_id", ""))),
        "method": "prior_guided_static_reference_weighted_median",
        "status": "refined_ego_motion_estimated",
        "description": (
            "Refine ego vx/vz using reliable static/low-dynamic reference objects. "
            "Reference object apparent motion votes for ego motion; votes are combined with a weighted median."
        ),
        "num_frames": len(frames_out),
        "num_frames_with_reference_votes": sum(1 for frame in frames_out if frame["num_supporting_reference_objects"] > 0),
        "num_reliable_reference_objects": int(reference_result.get("num_reliable_reference_objects", 0)),
        "correction_confidence": {
            "mean": float(sum(confidence_values) / max(1, len(confidence_values))),
            "max": float(max(confidence_values) if confidence_values else 0.0),
            "frames_with_confidence": int(sum(1 for value in confidence_values if value > 0.0)),
        },
        "correction_magnitude": {
            "mean": float(sum(correction_magnitudes) / max(1, len(correction_magnitudes))),
            "max": float(max(correction_magnitudes) if correction_magnitudes else 0.0),
        },
        "frames": frames_out,
    }


def _trajectory_motion_evidence_video(relative_video, ego_video=None):
    frame_indices, tracks = _relative_motion_track_index(relative_video)
    video_num_frames = int(relative_video.get("num_frames", len(frame_indices)))
    trajectories = []
    for track_id, track_data in sorted(tracks.items()):
        observations = [
            _trajectory_observation_from_motion_object(frame_index, track_data["frames"][frame_index])
            for frame_index in sorted(track_data.get("frames", {}))
        ]
        statistics = _trajectory_statistics(observations, video_num_frames)
        uncertainty = _trajectory_uncertainty(observations, statistics)
        validation = _trajectory_reality_validation(observations, statistics, uncertainty)
        provenance = {
            "source_counts": dict(statistics.get("source_counts", {})),
            "observed_count": int(statistics.get("observed_count", 0)),
            "repaired_count": int(statistics.get("repaired_count", 0)),
            "merged_count": int(statistics.get("merged_count", 0)),
            "observed_ratio": float(statistics.get("observed_ratio", 0.0)),
            "repaired_ratio": float(statistics.get("repaired_ratio", 0.0)),
            "merged_ratio": float(statistics.get("merged_ratio", 0.0)),
        }
        significance = _motion_significance_assessment(statistics, provenance, uncertainty, validation)
        fact_decision = _fact_decision_for_trajectory(validation, significance, provenance, uncertainty)
        trajectories.append(
            {
                "track_id": int(track_id),
                "primary_label": str(statistics.get("primary_label", track_data.get("label", "unknown"))),
                "trajectory_observations": observations,
                "trajectory_statistics": statistics,
                "provenance": provenance,
                "uncertainty": uncertainty,
                "causal_motion_fact_validation": validation,
                "motion_significance_assessment": significance,
                "fact_decision": fact_decision,
                "validation_status": validation["validation_status"],
                "motion_significance": significance["significance"],
                "fact_decision_status": fact_decision["decision"],
                "symbolic_layer_eligible": bool(fact_decision["symbolic_layer_eligible"]),
            }
        )
    status_counts = Counter(str(row.get("validation_status", "uncertain")) for row in trajectories)
    significance_counts = Counter(str(row.get("motion_significance", "low_significance")) for row in trajectories)
    decision_counts = Counter(str(row.get("fact_decision_status", "Keep with uncertainty")) for row in trajectories)
    return {
        "version": _CAUSAL_FILTER_OUT_VERSION,
        "evidence_type": "trajectory_motion_evidence",
        "video_id": str(relative_video.get("video_id", "")),
        "num_frames": video_num_frames,
        "num_trajectories": len(trajectories),
        "validation_status_counts": dict(sorted(status_counts.items())),
        "motion_significance_counts": dict(sorted(significance_counts.items())),
        "fact_decision_counts": dict(sorted(decision_counts.items())),
        "num_valid_trajectories": int(status_counts.get("valid", 0)),
        "num_repaired_trajectories": int(status_counts.get("repaired", 0)),
        "num_uncertain_trajectories": int(status_counts.get("uncertain", 0)),
        "num_invalid_trajectories": int(status_counts.get("invalid", 0)),
        "num_high_significance_trajectories": int(significance_counts.get("high_significance", 0)),
        "num_low_significance_trajectories": int(significance_counts.get("low_significance", 0)),
        "num_keep_trajectories": int(decision_counts.get("Keep", 0)),
        "num_keep_with_uncertainty_trajectories": int(decision_counts.get("Keep with uncertainty", 0)),
        "num_repair_decision_trajectories": int(decision_counts.get("Repair", 0)),
        "num_discard_trajectories": int(decision_counts.get("Discard", 0)),
        "num_symbolic_layer_eligible_trajectories": sum(
            1 for row in trajectories if bool(row.get("symbolic_layer_eligible", False))
        ),
        "num_observations": sum(len(row.get("trajectory_observations", [])) for row in trajectories),
        "num_repaired_observations": sum(int(row.get("provenance", {}).get("repaired_count", 0)) for row in trajectories),
        "num_observed_observations": sum(int(row.get("provenance", {}).get("observed_count", 0)) for row in trajectories),
        "inputs": {
            "has_ego_motion": bool(ego_video),
            "has_relative_object_motion": True,
        },
        "trajectory_motion_evidence": trajectories,
    }


def load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_step6_cache_payload(payload, video_id):
    return (
        isinstance(payload, dict)
        and str(payload.get("video_id", "")) == str(video_id)
        and isinstance(payload.get("frames", []), list)
    )


def collect_track_lengths(tracking_results):
    lengths = []
    for video_result in tracking_results:
        summaries = video_result.get("accepted_tracks", {}).get("track_summaries", [])
        for summary in summaries:
            lengths.append(int(summary.get("track_length", 0)))
    return lengths


def track_length_range_counts(track_lengths):
    ranges = [
        ("1-4", 1, 4),
        ("5-9", 5, 9),
        ("10-19", 10, 19),
        ("20-49", 20, 49),
        ("50-99", 50, 99),
        ("100+", 100, None),
    ]
    counts = []
    for label, start, end in ranges:
        if end is None:
            count = sum(1 for length in track_lengths if length >= start)
        else:
            count = sum(1 for length in track_lengths if start <= length <= end)
        counts.append((label, count))
    return counts


def save_track_length_histogram(track_lengths, output_root):
    if not track_lengths or plt is None:
        return None
    track_count = len(track_lengths)
    figure_path = Path(output_root) / f"track_length_histogram_n{track_count}.png"
    plt.figure(figsize=(8, 4.5))
    plt.hist(track_lengths, bins=20, color="#4C78A8", edgecolor="black")
    plt.xlabel("Track length")
    plt.ylabel("Number of tracks")
    plt.yscale("log")
    plt.title(f"Track Length Distribution (n={track_count})")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
    return figure_path


def step1_init(video_ids=None, video_count=None):
    dataset_root = config.get_dataset_path("driving_mini")
    video_dir = dataset_root / "videos"
    all_videos = sorted(config.get_mini_video_ids()) if video_dir.exists() else []
    if video_ids:
        videos = []
        for video_id in video_ids:
            if video_id and video_id not in videos:
                videos.append(video_id)
    else:
        videos = list(all_videos)
    if video_count is not None:
        videos = videos[:video_count]
    print(
        f"[step 1] loaded {len(videos)} videos for this run \n"
        f"[step 1] from {video_dir} \n"
        f"[step 1] (dataset=driving_mini, total_in_dataset={len(all_videos)})"
    )
    detection_args = {
        "video_ids": videos,
        "model_name": driving_pipeline_config.DRIVING_MINI_OD_MODEL,
        "classes": driving_pipeline_config.DRIVING_MINI_OD_CLASSES,
        "output_root": get_pipeline_output_root() / "01_driving_mini_detection",
        "od_calibration_policy": {},
        "force_recompute": False,
        "render_video": driving_pipeline_config.get_detection_render_video_enabled(default=True),
        "check_cache": driving_pipeline_config.get_detection_check_cache_enabled(default=False),
        "enable_candidate_branch": driving_pipeline_config.get_detection_candidate_branch_enabled(default=False),
        "skip_step": driving_pipeline_config.get_detection_skip_step_enabled(default=False),
    }
    tracking_args = {
        "output_root": get_pipeline_output_root() / "02_driving_mini_tracking",
        "frame_rate": 10,
        "tracker_args": None,
        "force_recompute": False,
        "render_video": driving_pipeline_config.get_tracking_render_video_enabled(default=True),
    }
    positions_3d_args = {
        "output_root": get_pipeline_output_root() / "06_driving_mini_3d_positions",
        "model_name": "depth-anything/DA3-Large",
        "batch_size": 4,
        "device": "auto",
        "force_recompute": False,
        "force_recompute_depth": False,
    }
    ego_motion_args = {
        "output_root": get_pipeline_output_root() / "07_driving_mini_ego_motion",
        "force_recompute": False,
        "smoothing_window": driving_pipeline_config.get_ego_motion_smoothing_window(default=5),
        "static_adjust_cfg": driving_pipeline_config.get_ego_static_adjustment_cfg(),
        "render_video": driving_pipeline_config.get_ego_motion_render_video_enabled(default=True),
        "flow_device": "auto",
    }
    return {
        "videos": videos,
        "dataset_root": dataset_root,
        "detection_args": detection_args,
        "tracking_args": tracking_args,
        "positions_3d_args": positions_3d_args,
        "ego_motion_args": ego_motion_args,
    }


def step2_detection(env, args):
    videos = env["videos"]
    if not videos:
        print("[step 2] no videos selected, skip detection")
        return {"videos": [], "detections": [], "detection_output_root": None}

    run_args = dict(args)
    model_name = run_args["model_name"]
    classes = run_args["classes"]
    skip_step = bool(run_args.pop("skip_step", False))
    render_video = bool(run_args.get("render_video", True))
    check_cache = bool(run_args.get("check_cache", False))
    candidate_branch_enabled = bool(run_args.get("enable_candidate_branch", False))
    output_root = Path(run_args["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[step 2] model={model_name}")
    print(f"[step 2] classes={len(classes)} render_video={render_video} check_cache={check_cache}")
    print(f"[step 2] output_root={output_root}")

    if skip_step:
        manifest_path = output_root / "detection_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        video_entries = {
            str(entry.get("video_id", "")).strip(): dict(entry)
            for entry in manifest.get("videos", [])
            if str(entry.get("video_id", "")).strip()
        }
        detections = []
        for video_id in videos:
            entry = video_entries[video_id]
            detections_path = Path(str(entry.get("detections_json", "")).strip() or output_root / video_id / "detections.json")
            with detections_path.open("r", encoding="utf-8") as f:
                video_result = json.load(f)
            if hasattr(detect_driving_mini, "_apply_candidate_branch_mode"):
                video_result = detect_driving_mini._apply_candidate_branch_mode(video_result, candidate_branch_enabled)
            detections.append(video_result)
        print(f"[step 2] loaded cached detection results for {len(detections)} videos")
    else:
        detections = detect_driving_mini.run(**run_args)
        print(f"[step 2] completed detection for {len(detections)} videos")

    accepted_count = sum(int(video_result.get("num_detections", 0)) for video_result in detections)
    candidate_count = sum(int(video_result.get("num_candidate_detections", 0)) for video_result in detections)
    normalized_detections = []
    rewritten_count = 0
    for video_result in detections:
        normalized_video_result, changed = normalize_detection_image_paths(video_result, env["dataset_root"])
        if changed:
            rewritten_count += 1
            write_detection_cache_if_needed(
                normalized_video_result,
                source_path=output_root / str(normalized_video_result.get("video_id", "")).strip() / "detections.json",
            )
        normalized_detections.append(normalized_video_result)
    detections = normalized_detections
    if rewritten_count:
        print(f"[step 2] rewrote frame paths for {rewritten_count} cached detection files")
    print(f"[step 2] accepted_detections={accepted_count}, candidate_detections={candidate_count}")
    return {
        "videos": videos,
        "detections": detections,
        "detection_output_root": output_root,
        "model_name": model_name,
        "classes": classes,
        "tracking_args": env["tracking_args"],
        "positions_3d_args": env["positions_3d_args"],
        "ego_motion_args": env["ego_motion_args"],
    }


def step3_tracking(detection_state):
    videos = detection_state["videos"]
    detections = detection_state["detections"]
    if not videos or not detections:
        print("[step 3] no detection results, skip tracking")
        return {"videos": videos, "tracks": [], "tracking_output_root": None}

    run_args = dict(detection_state["tracking_args"])
    output_root = Path(run_args["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    render_video = bool(run_args.get("render_video", True))
    if render_video and tracking_driving_mini.cv2 is None:
        print("[step 3][warn] OpenCV is unavailable; tracking rendering disabled")
        render_video = False
    tracking_driving_mini.ensure_tracking_runtime_available()

    tracking_results = []
    progress = tqdm(detections, desc="[step 3] tracking", unit="video")
    for video_result in progress:
        progress.set_postfix_str(str(video_result.get("video_id", "")), refresh=False)
        tracking_results.append(
            tracking_driving_mini.track_video(
                video_result=video_result,
                output_root=output_root,
                frame_rate=int(run_args.get("frame_rate", 10)),
                tracker_args=run_args.get("tracker_args"),
                force_recompute=bool(run_args.get("force_recompute", False)),
                render_video=render_video,
            )
        )

    manifest = {
        "schema_version": getattr(tracking_driving_mini, "_TRACKING_SCHEMA_VERSION", 7),
        "candidate_branch_enabled": all(bool(r.get("candidate_branch_enabled", True)) for r in tracking_results),
        "num_videos": len(tracking_results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r["num_frames"],
                "num_tracks": r["num_tracks"],
                "num_tracking_input_candidate_detections": int(r.get("num_tracking_input_candidate_detections", 0)),
                "num_candidate_tracks": int(r.get("num_candidate_tracks", 0)),
                "num_raw_candidate_tracks": int(r.get("num_raw_candidate_tracks", 0)),
                "num_deduplicated_candidate_tracks": int(r.get("num_deduplicated_candidate_tracks", 0)),
                "num_rejected_candidate_tracks": int(r.get("num_rejected_candidate_tracks", 0)),
                "num_rejected_candidate_detections": int(r.get("num_rejected_candidate_detections", 0)),
            }
            for r in tracking_results
        ],
    }
    with (output_root / "tracks_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    track_lengths = collect_track_lengths(tracking_results)
    for label, count in track_length_range_counts(track_lengths):
        print(f"[step 3] track_length_range {label}: {count}")
    figure_path = save_track_length_histogram(track_lengths, output_root)
    print(
        f"[step 3] done videos={len(tracking_results)} "
        f"tracks={sum(int(row.get('num_tracks', 0)) for row in tracking_results)}"
    )
    if figure_path is not None:
        print(f"[step 3] histogram={figure_path}")
    return {
        "videos": videos,
        "tracks": tracking_results,
        "tracking_output_root": output_root,
        "positions_3d_args": detection_state["positions_3d_args"],
        "ego_motion_args": detection_state["ego_motion_args"],
    }

def step6_positions_3d(tracking_state):
    videos = tracking_state["videos"]
    tracking_results = tracking_state["tracks"]
    if not videos or not tracking_results:
        print("[step 6] no tracking results, skip 3d positions")
        return {"videos": videos, "positions_3d": [], "positions_3d_output_root": None}

    run_args = dict(tracking_state["positions_3d_args"])
    output_root = Path(run_args["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[step 6] output_root={output_root}")
    merged_results = [
        merge_gt_and_detected_driving_mini._tracked_video_as_merged_result(video_result)
        for video_result in tracking_results
    ]
    positions_3d = []
    cached_videos = 0
    progress = tqdm(merged_results, desc="[step 6] positions_3d", unit="video")
    for video_result in progress:
        video_id = str(video_result.get("video_id", ""))
        progress.set_postfix_str(video_id, refresh=False)
        cache_path = output_root / video_id / "positions_3d.json"
        cached_result = None
        if not bool(run_args.get("force_recompute", False)):
            payload = load_json_if_exists(cache_path)
            if is_step6_cache_payload(payload, video_id):
                cached_result = payload
        if cached_result is not None:
            cached_videos += 1
            positions_3d.append(cached_result)
            continue
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            positions_3d.append(
                prepare_3d_positions_driving_mini.process_video(
                    video_result=video_result,
                    output_root=output_root,
                    model_name=run_args.get("model_name", "depth-anything/DA3-Large"),
                    batch_size=int(run_args.get("batch_size", 4)),
                    device=str(run_args.get("device", "auto")),
                    force_recompute=bool(run_args.get("force_recompute", False)),
                    force_recompute_depth=bool(run_args.get("force_recompute_depth", False)),
                )
            )
    manifest = {
        "version": getattr(prepare_3d_positions_driving_mini, "_POSITIONS_3D_VERSION", 4),
        "model_name": run_args.get("model_name", "depth-anything/DA3-Large"),
        "num_videos": len(positions_3d),
        "num_frames_total": sum(r.get("num_frames", 0) for r in positions_3d),
        "num_objects_with_3d_total": sum(r.get("num_objects_with_3d", 0) for r in positions_3d),
        "num_candidate_objects_with_3d_total": sum(r.get("num_candidate_objects_with_3d", 0) for r in positions_3d),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r.get("num_frames", 0),
                "num_objects_with_3d": r.get("num_objects_with_3d", 0),
                "num_candidate_objects_with_3d": r.get("num_candidate_objects_with_3d", 0),
                "depth_dir": r.get("depth_dir", ""),
            }
            for r in positions_3d
        ],
    }
    with (output_root / "positions_3d_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[step 6] done videos={len(positions_3d)} "
        f"cached={cached_videos} "
        f"objects_with_3d={sum(int(row.get('num_objects_with_3d', 0)) for row in positions_3d)} "
        f"candidate_objects_with_3d={sum(int(row.get('num_candidate_objects_with_3d', 0)) for row in positions_3d)}"
    )
    return {
        "videos": videos,
        "positions_3d": positions_3d,
        "positions_3d_output_root": output_root,
        "ego_motion_args": tracking_state["ego_motion_args"],
    }


def step7_ego_motion(position_state):
    videos = position_state["videos"]
    positions_3d = position_state["positions_3d"]
    if not videos or not positions_3d:
        print("[step 7] no 3d positions, skip ego motion")
        return {"videos": videos, "ego_motion": [], "ego_motion_output_root": None}

    run_args = dict(position_state["ego_motion_args"])
    output_root = Path(run_args["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    ego_motion = []
    cached_videos = 0
    progress = tqdm(positions_3d, desc="[step 7] ego_motion", unit="video")
    for video_result in progress:
        video_id = str(video_result.get("video_id", ""))
        progress.set_postfix_str(video_id, refresh=False)
        cache_path = output_root / video_id / "ego_motion.json"
        cached_result = None
        if not bool(run_args.get("force_recompute", False)):
            payload = load_json_if_exists(cache_path)
            if (
                payload
                and int(payload.get("version", 0)) == getattr(ego_motion_driving_mini, "_EGO_MOTION_VERSION", 0)
                and str(payload.get("estimation_method", "")) == getattr(ego_motion_driving_mini, "_EGO_MOTION_METHOD", "")
            ):
                cached_result = payload
        if cached_result is not None:
            cached_videos += 1
            ego_motion.append(cached_result)
            continue
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            ego_motion.append(
                ego_motion_driving_mini.process_video(
                    video_result=video_result,
                    output_root=output_root,
                    force_recompute=bool(run_args.get("force_recompute", False)),
                    smoothing_window=int(run_args.get("smoothing_window", 5)),
                    static_adjust_cfg=run_args.get("static_adjust_cfg"),
                    render_video=bool(run_args.get("render_video", True)),
                    flow_device=run_args.get("flow_device"),
                )
            )
    manifest = {
        "num_videos": len(ego_motion),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r["num_frames"],
                "num_frames_with_ego_motion": r["num_frames_with_ego_motion"],
            }
            for r in ego_motion
        ],
    }
    with (output_root / "ego_motion_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[step 7] done videos={len(ego_motion)} "
        f"cached={cached_videos} "
        f"frames_with_ego_motion={sum(int(row.get('num_frames_with_ego_motion', 0)) for row in ego_motion)}"
    )
    return {
        "videos": videos,
        "ego_motion": ego_motion,
        "ego_motion_output_root": output_root,
    }


def step7b_tracklet_repair(position_state, ego_state):
    videos = position_state["videos"]
    positions_3d = position_state.get("positions_3d", [])
    ego_motion = ego_state.get("ego_motion", [])
    if not videos or not positions_3d:
        print("[step 7b] no 3d positions, skip tracklet repair")
        return {
            "videos": videos,
            "tracklet_repair": [],
            "positions_3d": positions_3d,
            "positions_3d_output_root": position_state.get("positions_3d_output_root"),
            "ego_motion": ego_motion,
            "ego_motion_output_root": ego_state.get("ego_motion_output_root"),
        }

    output_root = get_pipeline_output_root() / "07b_driving_mini_tracklet_repair"
    output_root.mkdir(parents=True, exist_ok=True)
    ego_by_video = {
        str(row.get("video_id", "")): row
        for row in ego_motion
        if str(row.get("video_id", ""))
    }
    repaired_positions_3d = []
    progress = tqdm(positions_3d, desc="[step 7b] tracklet_repair", unit="video")
    for video_result in progress:
        video_id = str(video_result.get("video_id", ""))
        progress.set_postfix_str(video_id, refresh=False)
        repaired = _repair_video_tracklets(video_result, ego_by_video.get(video_id))
        out_dir = output_root / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "tracklet_repair.json").open("w", encoding="utf-8") as f:
            json.dump(repaired, f, indent=2)
        repaired_positions_3d.append(repaired)

    repaired_frame_stats = []
    for row in repaired_positions_3d:
        repair_events = list(row.get("tracklet_repair", {}).get("repair_events", []))
        repaired_frame_indices = sorted(
            {
                int(frame_index)
                for event in repair_events
                for frame_index in list(event.get("inserted_frame_indices", []))
            }
        )
        num_frames = int(row.get("num_frames", len(row.get("frames", []))))
        num_repaired_frames = len(repaired_frame_indices)
        repaired_frame_ratio = float(num_repaired_frames / max(1, num_frames))
        repaired_frame_stats.append(
            {
                "video_id": str(row.get("video_id", "")),
                "num_frames": num_frames,
                "num_repaired_frames": num_repaired_frames,
                "repaired_frame_ratio": repaired_frame_ratio,
                "repaired_frame_percentage": float(repaired_frame_ratio * 100.0),
            }
        )
    total_frames = sum(int(row["num_frames"]) for row in repaired_frame_stats)
    total_repaired_frames = sum(int(row["num_repaired_frames"]) for row in repaired_frame_stats)
    repaired_frame_ratio_total = float(total_repaired_frames / max(1, total_frames))
    average_repaired_frame_percentage = float(
        sum(float(row["repaired_frame_percentage"]) for row in repaired_frame_stats)
        / max(1, len(repaired_frame_stats))
    )
    repaired_frame_stats_by_video = {
        str(row["video_id"]): row
        for row in repaired_frame_stats
    }
    manifest = {
        "version": _TRACKLET_REPAIR_VERSION,
        "num_videos": len(repaired_positions_3d),
        "num_repaired_gaps": sum(
            int(row.get("tracklet_repair", {}).get("num_repaired_gaps", 0))
            for row in repaired_positions_3d
        ),
        "num_interpolated_objects": sum(
            int(row.get("tracklet_repair", {}).get("num_interpolated_objects", 0))
            for row in repaired_positions_3d
        ),
        "num_split_events": sum(
            int(row.get("tracklet_repair", {}).get("num_split_events", 0))
            for row in repaired_positions_3d
        ),
        "num_new_track_ids": sum(
            int(row.get("tracklet_repair", {}).get("num_new_track_ids", 0))
            for row in repaired_positions_3d
        ),
        "num_repaired_frames_total": total_repaired_frames,
        "num_frames_total": total_frames,
        "repaired_frame_ratio_total": repaired_frame_ratio_total,
        "repaired_frame_percentage_total": float(repaired_frame_ratio_total * 100.0),
        "average_repaired_frame_percentage": average_repaired_frame_percentage,
        "videos": [
            {
                "video_id": row["video_id"],
                "num_frames": row.get("num_frames", 0),
                "num_repaired_gaps": row.get("tracklet_repair", {}).get("num_repaired_gaps", 0),
                "num_interpolated_objects": row.get("tracklet_repair", {}).get("num_interpolated_objects", 0),
                "num_split_events": row.get("tracklet_repair", {}).get("num_split_events", 0),
                "num_new_track_ids": row.get("tracklet_repair", {}).get("num_new_track_ids", 0),
                "num_repaired_frames": repaired_frame_stats_by_video.get(str(row.get("video_id", "")), {}).get(
                    "num_repaired_frames",
                    0,
                ),
                "repaired_frame_percentage": repaired_frame_stats_by_video.get(str(row.get("video_id", "")), {}).get(
                    "repaired_frame_percentage",
                    0.0,
                ),
                "num_skipped_gaps": row.get("tracklet_repair", {}).get("num_skipped_gaps", 0),
            }
            for row in repaired_positions_3d
        ],
    }
    with (output_root / "tracklet_repair_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[step 7b] done videos={len(repaired_positions_3d)} "
        f"repaired_gaps={manifest['num_repaired_gaps']} "
        f"interpolated_objects={manifest['num_interpolated_objects']} "
        f"split_events={manifest['num_split_events']} "
        f"new_track_ids={manifest['num_new_track_ids']} "
        f"avg_repaired_frame_pct={manifest['average_repaired_frame_percentage']:.2f}% "
        f"repaired_frames={manifest['num_repaired_frames_total']}/{manifest['num_frames_total']} "
        f"({manifest['repaired_frame_percentage_total']:.2f}%)"
    )
    return {
        "videos": videos,
        "tracklet_repair": repaired_positions_3d,
        "positions_3d": repaired_positions_3d,
        "positions_3d_output_root": position_state.get("positions_3d_output_root"),
        "tracklet_repair_output_root": output_root,
        "ego_motion": ego_motion,
        "ego_motion_output_root": ego_state.get("ego_motion_output_root"),
    }


def step8_relative_object_motion(position_state, repaired_state):
    videos = position_state["videos"]
    repaired_positions = repaired_state.get("positions_3d", repaired_state.get("tracklet_repair", []))
    ego_motion = repaired_state.get("ego_motion", [])
    if not videos or not repaired_positions:
        print("[step 8] no repaired object positions, skip relative object motion")
        return {"videos": videos, "relative_object_motion": []}

    output_root = get_pipeline_output_root() / "08_driving_mini_relative_object_motion"
    output_root.mkdir(parents=True, exist_ok=True)
    ego_by_video = {
        str(row.get("video_id", "")): row
        for row in ego_motion
        if str(row.get("video_id", ""))
    }
    relative_motion = []
    progress = tqdm(repaired_positions, desc="[step 8] relative_object_motion", unit="video")
    for video_result in progress:
        video_id = str(video_result.get("video_id", ""))
        progress.set_postfix_str(video_id, refresh=False)
        ego_result = ego_by_video.get(video_id, {})
        result = _relative_motion_video(video_result, ego_result)
        out_dir = output_root / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "relative_object_motion.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        relative_motion.append(result)

    manifest = {
        "version": _RELATIVE_OBJECT_MOTION_VERSION,
        "num_videos": len(relative_motion),
        "num_objects_total": sum(int(row.get("num_objects_total", 0)) for row in relative_motion),
        "num_observed_objects_total": sum(int(row.get("num_observed_objects_total", 0)) for row in relative_motion),
        "num_repaired_objects_total": sum(int(row.get("num_repaired_objects_total", 0)) for row in relative_motion),
        "num_objects_with_rel_motion": sum(int(row.get("num_objects_with_rel_motion", 0)) for row in relative_motion),
        "videos": [
            {
                "video_id": row["video_id"],
                "num_frames": row.get("num_frames", 0),
                "num_objects_total": row.get("num_objects_total", 0),
                "num_observed_objects_total": row.get("num_observed_objects_total", 0),
                "num_repaired_objects_total": row.get("num_repaired_objects_total", 0),
                "num_objects_with_rel_motion": row.get("num_objects_with_rel_motion", 0),
            }
            for row in relative_motion
        ],
    }
    with (output_root / "relative_object_motion_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[step 8] done videos={len(relative_motion)} "
        f"objects={manifest['num_objects_total']} "
        f"observed={manifest['num_observed_objects_total']} "
        f"repaired={manifest['num_repaired_objects_total']} "
        f"with_rel_motion={manifest['num_objects_with_rel_motion']}"
    )
    return {
        "videos": videos,
        "relative_object_motion": relative_motion,
        "relative_object_motion_output_root": output_root,
        "positions_3d": repaired_positions,
        "tracklet_repair": repaired_state.get("tracklet_repair", []),
        "ego_motion": ego_motion,
    }


def step8_visual_relative_motion(relative_motion_state, fps=10.0):
    videos = relative_motion_state.get("videos", [])
    relative_motion = relative_motion_state.get("relative_object_motion", [])
    if not videos or not relative_motion:
        print("[step 8visual] no relative object motion, skip visualization")
        return {
            **relative_motion_state,
            "relative_motion_visualizations": [],
            "relative_motion_visualization_output_root": None,
        }

    output_root = get_pipeline_output_root() / "08visual_relative_motion_tracks"
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_by_video = {}
    for evidence_video in relative_motion_state.get("trajectory_motion_evidence", []):
        video_id = str(evidence_video.get("video_id", ""))
        if not video_id:
            continue
        evidence_by_track = {}
        for row in evidence_video.get("trajectory_motion_evidence", []):
            try:
                track_id = int(row.get("track_id", -1))
            except (TypeError, ValueError):
                continue
            if track_id >= 0:
                evidence_by_track[track_id] = dict(row)
        evidence_by_video[video_id] = evidence_by_track
    refined_ego_by_video = {
        str(row.get("video_id", "")): row
        for row in relative_motion_state.get("refined_ego_motion", [])
        if str(row.get("video_id", ""))
    }
    all_rendered = []
    all_skipped = []
    ego_motion_chart_pdfs = []
    ego_motion_chart_skipped = []
    progress = tqdm(relative_motion, desc="[step 8visual] relative_motion_tracks", unit="video")
    for video_result in progress:
        video_id = str(video_result.get("video_id", ""))
        progress.set_postfix_str(video_id, refresh=False)
        refined_ego_video = refined_ego_by_video.get(video_id, {})
        pdf_path, pdf_status = _save_ego_motion_comparison_pdf(
            refined_ego_video,
            output_root / video_id / "ego_motion_comparison.pdf",
        )
        pdf_row = {
            "video_id": video_id,
            "status": pdf_status,
            "methods": ["original", "weighted_median", "refined", "ransac_if_available"],
        }
        if pdf_path:
            pdf_row["pdf_path"] = pdf_path
            ego_motion_chart_pdfs.append(pdf_row)
        else:
            ego_motion_chart_skipped.append(pdf_row)
        rendered, skipped = _render_relative_motion_track_videos(
            relative_motion_video_result=video_result,
            output_root=output_root,
            fps=float(fps),
            trajectory_evidence_by_track=evidence_by_video.get(video_id, {}),
        )
        all_rendered.extend(rendered)
        all_skipped.extend(skipped)

    manifest = {
        "version": _RELATIVE_MOTION_VIS_VERSION,
        "num_videos": len(relative_motion),
        "fps": float(fps),
        "num_track_videos_rendered": len(all_rendered),
        "num_track_videos_skipped": len(all_skipped),
        "uses_causal_filter_out": bool(evidence_by_video),
        "uses_refined_ego_motion": bool(refined_ego_by_video),
        "ego_motion_charts_in_track_videos": False,
        "ego_motion_chart_methods": ["original", "weighted_median", "refined", "ransac_if_available"],
        "num_ego_motion_chart_pdfs_rendered": len(ego_motion_chart_pdfs),
        "num_ego_motion_chart_pdfs_skipped": len(ego_motion_chart_skipped),
        "ego_motion_chart_pdfs": ego_motion_chart_pdfs,
        "ego_motion_chart_skipped": ego_motion_chart_skipped,
        "rendered": all_rendered,
        "skipped": all_skipped,
    }
    with (output_root / "relative_motion_track_visualization_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[step 8visual] done videos={len(relative_motion)} "
        f"track_videos={manifest['num_track_videos_rendered']} "
        f"skipped={manifest['num_track_videos_skipped']} "
        f"output_root={output_root}"
    )
    return {
        **relative_motion_state,
        "relative_motion_visualizations": all_rendered,
        "relative_motion_visualization_skipped": all_skipped,
        "ego_motion_chart_pdfs": ego_motion_chart_pdfs,
        "ego_motion_chart_skipped": ego_motion_chart_skipped,
        "relative_motion_visualization_output_root": output_root,
    }


def step9_temporal_segmentation(ego_state, relative_motion_state):
    return {"videos": ego_state["videos"], "temporal_segments": []}


def step8b_causal_filter_out(ego_state, relative_motion_state):
    videos = relative_motion_state.get("videos", ego_state.get("videos", []))
    relative_motion = relative_motion_state.get("relative_object_motion", [])
    output_root = get_pipeline_output_root() / "08b_driving_mini_causal_filter_out"
    output_root.mkdir(parents=True, exist_ok=True)

    ego_by_video = {
        str(row.get("video_id", "")): row
        for row in ego_state.get("ego_motion", [])
        if str(row.get("video_id", ""))
    }

    trajectory_motion_evidence = []
    for relative_video in relative_motion:
        video_id = str(relative_video.get("video_id", ""))
        evidence = _trajectory_motion_evidence_video(relative_video, ego_by_video.get(video_id))
        num_objects_in = int(evidence.get("num_observations", 0))
        num_tracks_in = int(evidence.get("num_trajectories", 0))
        evidence.update(
            {
                "method": "causal_motion_fact_validation_evidence_build",
                "status": "trajectory_reality_validated",
                "description": (
                    "Frame-level relative motion has been aggregated into trajectory-level "
                    "evidence and checked for trajectory realism. Final object removal is not applied yet."
                ),
                "num_objects_in": int(num_objects_in),
                "num_objects_kept": int(num_objects_in),
                "num_objects_filtered": 0,
                "num_tracks_in": int(num_tracks_in),
                "num_tracks_kept": int(num_tracks_in),
                "num_tracks_filtered": 0,
                "kept_track_ids": [
                    int(row.get("track_id", -1))
                    for row in evidence.get("trajectory_motion_evidence", [])
                    if int(row.get("track_id", -1)) >= 0
                ],
                "filtered_track_ids": [],
                "filter_decisions": [],
                "causal_reasoning": {
                    "enabled": False,
                    "method": "",
                    "rules": [],
                    "effects": [],
                    "notes": "Trajectory realism validation is available; causal filtering rules are not implemented yet.",
                },
                "relative_object_motion": relative_video,
                "filtered_relative_object_motion": relative_video,
            }
        )
        evidence["trajectory_statistics_summary"] = {
            "mean_confidence_score": _numeric_stats(
                [
                    row.get("uncertainty", {}).get("confidence_score", 0.0)
                    for row in evidence.get("trajectory_motion_evidence", [])
                ]
            ).get("mean", 0.0),
            "mean_repaired_ratio": _numeric_stats(
                [
                    row.get("provenance", {}).get("repaired_ratio", 0.0)
                    for row in evidence.get("trajectory_motion_evidence", [])
                ]
            ).get("mean", 0.0),
            "mean_temporal_coverage": _numeric_stats(
                [
                    row.get("trajectory_statistics", {}).get("temporal_coverage_in_span", 0.0)
                    for row in evidence.get("trajectory_motion_evidence", [])
                ]
            ).get("mean", 0.0),
        }
        evidence["ego_temporal_signal_summary"] = {}
        evidence["object_temporal_signal_summary"] = evidence["trajectory_statistics_summary"]
        out_dir = output_root / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "trajectory_motion_evidence.json").open("w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2)
        # Keep the older filename as an alias while 8B's downstream contract settles.
        with (out_dir / "causal_filter_out.json").open("w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2)
        trajectory_motion_evidence.append(evidence)

    manifest = {
        "version": _CAUSAL_FILTER_OUT_VERSION,
        "method": "causal_motion_fact_validation_evidence_build",
        "evidence_type": "trajectory_motion_evidence",
        "num_videos": len(trajectory_motion_evidence),
        "num_trajectories": sum(int(row.get("num_trajectories", 0)) for row in trajectory_motion_evidence),
        "num_valid_trajectories": sum(int(row.get("num_valid_trajectories", 0)) for row in trajectory_motion_evidence),
        "num_repaired_trajectories": sum(int(row.get("num_repaired_trajectories", 0)) for row in trajectory_motion_evidence),
        "num_uncertain_trajectories": sum(int(row.get("num_uncertain_trajectories", 0)) for row in trajectory_motion_evidence),
        "num_invalid_trajectories": sum(int(row.get("num_invalid_trajectories", 0)) for row in trajectory_motion_evidence),
        "num_high_significance_trajectories": sum(int(row.get("num_high_significance_trajectories", 0)) for row in trajectory_motion_evidence),
        "num_low_significance_trajectories": sum(int(row.get("num_low_significance_trajectories", 0)) for row in trajectory_motion_evidence),
        "num_keep_trajectories": sum(int(row.get("num_keep_trajectories", 0)) for row in trajectory_motion_evidence),
        "num_keep_with_uncertainty_trajectories": sum(int(row.get("num_keep_with_uncertainty_trajectories", 0)) for row in trajectory_motion_evidence),
        "num_repair_decision_trajectories": sum(int(row.get("num_repair_decision_trajectories", 0)) for row in trajectory_motion_evidence),
        "num_discard_trajectories": sum(int(row.get("num_discard_trajectories", 0)) for row in trajectory_motion_evidence),
        "num_symbolic_layer_eligible_trajectories": sum(int(row.get("num_symbolic_layer_eligible_trajectories", 0)) for row in trajectory_motion_evidence),
        "num_observations": sum(int(row.get("num_observations", 0)) for row in trajectory_motion_evidence),
        "num_repaired_observations": sum(int(row.get("num_repaired_observations", 0)) for row in trajectory_motion_evidence),
        "num_observed_observations": sum(int(row.get("num_observed_observations", 0)) for row in trajectory_motion_evidence),
        "num_objects_in": sum(int(row.get("num_objects_in", 0)) for row in trajectory_motion_evidence),
        "num_objects_kept": sum(int(row.get("num_objects_kept", 0)) for row in trajectory_motion_evidence),
        "num_objects_filtered": sum(int(row.get("num_objects_filtered", 0)) for row in trajectory_motion_evidence),
        "num_tracks_in": sum(int(row.get("num_tracks_in", 0)) for row in trajectory_motion_evidence),
        "num_tracks_kept": sum(int(row.get("num_tracks_kept", 0)) for row in trajectory_motion_evidence),
        "num_tracks_filtered": sum(int(row.get("num_tracks_filtered", 0)) for row in trajectory_motion_evidence),
        "videos": [
            {
                "video_id": row["video_id"],
                "num_frames": row.get("num_frames", 0),
                "num_trajectories": row.get("num_trajectories", 0),
                "num_valid_trajectories": row.get("num_valid_trajectories", 0),
                "num_repaired_trajectories": row.get("num_repaired_trajectories", 0),
                "num_uncertain_trajectories": row.get("num_uncertain_trajectories", 0),
                "num_invalid_trajectories": row.get("num_invalid_trajectories", 0),
                "num_high_significance_trajectories": row.get("num_high_significance_trajectories", 0),
                "num_low_significance_trajectories": row.get("num_low_significance_trajectories", 0),
                "num_keep_trajectories": row.get("num_keep_trajectories", 0),
                "num_keep_with_uncertainty_trajectories": row.get("num_keep_with_uncertainty_trajectories", 0),
                "num_repair_decision_trajectories": row.get("num_repair_decision_trajectories", 0),
                "num_discard_trajectories": row.get("num_discard_trajectories", 0),
                "num_symbolic_layer_eligible_trajectories": row.get("num_symbolic_layer_eligible_trajectories", 0),
                "num_observations": row.get("num_observations", 0),
                "num_repaired_observations": row.get("num_repaired_observations", 0),
                "num_observed_observations": row.get("num_observed_observations", 0),
                "num_objects_filtered": row.get("num_objects_filtered", 0),
                "num_tracks_filtered": row.get("num_tracks_filtered", 0),
                "status": row.get("status", ""),
            }
            for row in trajectory_motion_evidence
        ],
    }
    with (output_root / "trajectory_motion_evidence_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with (output_root / "causal_filter_out_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[step 8b] trajectory_motion_evidence "
        f"videos={len(trajectory_motion_evidence)} "
        f"trajectories={manifest['num_trajectories']} "
        f"valid={manifest['num_valid_trajectories']} "
        f"repaired={manifest['num_repaired_trajectories']} "
        f"uncertain={manifest['num_uncertain_trajectories']} "
        f"invalid={manifest['num_invalid_trajectories']} "
        f"high_sig={manifest['num_high_significance_trajectories']} "
        f"low_sig={manifest['num_low_significance_trajectories']} "
        f"keep={manifest['num_keep_trajectories']} "
        f"keep_uncertain={manifest['num_keep_with_uncertainty_trajectories']} "
        f"repair={manifest['num_repair_decision_trajectories']} "
        f"discard={manifest['num_discard_trajectories']} "
        f"observations={manifest['num_observations']} "
        f"repaired_observations={manifest['num_repaired_observations']} "
        f"filtered_objects={manifest['num_objects_filtered']}"
    )
    return {
        **relative_motion_state,
        "trajectory_motion_evidence": trajectory_motion_evidence,
        "trajectory_motion_evidence_output_root": output_root,
        "causal_filter_out": trajectory_motion_evidence,
        "causal_filter_out_output_root": output_root,
        "relative_object_motion": relative_motion,
        "filtered_relative_object_motion": relative_motion,
        "ego_motion": ego_state.get("ego_motion", []),
    }


def step8c_prior_guided_ego_motion_refinement(ego_state, relative_motion_state):
    videos = relative_motion_state.get("videos", ego_state.get("videos", []))
    trajectory_motion_evidence = relative_motion_state.get("trajectory_motion_evidence", [])
    if not videos or not trajectory_motion_evidence:
        print("[step 8c] no trajectory motion evidence, skip prior-guided ego refinement")
        return {
            **relative_motion_state,
            "reliable_reference_objects": [],
            "prior_guided_ego_refinement_output_root": None,
        }

    output_root = get_pipeline_output_root() / "08c_prior_guided_ego_motion_refinement"
    output_root.mkdir(parents=True, exist_ok=True)
    ego_by_video = {
        str(row.get("video_id", "")): row
        for row in ego_state.get("ego_motion", relative_motion_state.get("ego_motion", []))
        if str(row.get("video_id", ""))
    }
    reliable_reference_results = []
    refined_ego_motion_results = []
    for evidence_video in trajectory_motion_evidence:
        video_id = str(evidence_video.get("video_id", ""))
        result = _reliable_reference_objects_video(evidence_video)
        refined_ego_motion = _refined_ego_motion_video(
            ego_video=ego_by_video.get(video_id, {}),
            evidence_video=evidence_video,
            reference_result=result,
        )
        out_dir = output_root / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "reliable_reference_objects.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        with (out_dir / "refined_ego_motion.json").open("w", encoding="utf-8") as f:
            json.dump(refined_ego_motion, f, indent=2)
        reliable_reference_results.append(result)
        refined_ego_motion_results.append(refined_ego_motion)

    manifest = {
        "version": _EGO_REFINEMENT_VERSION,
        "method": "prior_guided_static_reference_weighted_median",
        "num_videos": len(reliable_reference_results),
        "num_trajectories": sum(int(row.get("num_trajectories", 0)) for row in reliable_reference_results),
        "num_reliable_reference_objects": sum(
            int(row.get("num_reliable_reference_objects", 0))
            for row in reliable_reference_results
        ),
        "num_frames": sum(int(row.get("num_frames", 0)) for row in refined_ego_motion_results),
        "num_frames_with_reference_votes": sum(
            int(row.get("num_frames_with_reference_votes", 0))
            for row in refined_ego_motion_results
        ),
        "mean_correction_confidence": float(
            sum(float(row.get("correction_confidence", {}).get("mean", 0.0)) for row in refined_ego_motion_results)
            / max(1, len(refined_ego_motion_results))
        ),
        "max_correction_magnitude": float(
            max([float(row.get("correction_magnitude", {}).get("max", 0.0)) for row in refined_ego_motion_results] or [0.0])
        ),
        "videos": [
            {
                "video_id": row["video_id"],
                "num_trajectories": row.get("num_trajectories", 0),
                "num_reliable_reference_objects": row.get("num_reliable_reference_objects", 0),
                "num_frames": refined_ego_motion_results[idx].get("num_frames", 0),
                "num_frames_with_reference_votes": refined_ego_motion_results[idx].get("num_frames_with_reference_votes", 0),
                "mean_correction_confidence": refined_ego_motion_results[idx].get("correction_confidence", {}).get("mean", 0.0),
                "max_correction_magnitude": refined_ego_motion_results[idx].get("correction_magnitude", {}).get("max", 0.0),
                "status": row.get("status", ""),
            }
            for idx, row in enumerate(reliable_reference_results)
        ],
    }
    with (output_root / "reliable_reference_objects_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with (output_root / "refined_ego_motion_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[step 8c] prior_guided_ego_motion_refinement "
        f"videos={len(reliable_reference_results)} "
        f"reference_objects={manifest['num_reliable_reference_objects']}/"
        f"{manifest['num_trajectories']} "
        f"frames_with_votes={manifest['num_frames_with_reference_votes']}/"
        f"{manifest['num_frames']} "
        f"mean_conf={manifest['mean_correction_confidence']:.3f} "
        f"output_root={output_root}"
    )
    return {
        **relative_motion_state,
        "reliable_reference_objects": reliable_reference_results,
        "refined_ego_motion": refined_ego_motion_results,
        "prior_guided_ego_refinement_output_root": output_root,
        "ego_motion": ego_state.get("ego_motion", relative_motion_state.get("ego_motion", [])),
    }


def step10_segment_object_motion(segment_state):
    return {"videos": segment_state["videos"], "segment_object_motion": []}
