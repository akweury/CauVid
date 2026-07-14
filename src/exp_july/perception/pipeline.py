import copy
import json
import math
import os
import sys
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


def _repair_video_tracklets(video_result, ego_result=None, repair_cfg=None):
    cfg = dict(_TRACKLET_REPAIR_DEFAULT_CFG)
    if repair_cfg:
        cfg.update(repair_cfg)
    repaired = copy.deepcopy(video_result)
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
    panel_h = panel.shape[0]
    bar_x = 18
    bar_w = max(1, width - 2 * bar_x)
    bar_h = 18
    bar_y = panel_h - 34
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


def _render_relative_motion_track_video(
    relative_motion_video_result,
    track_id,
    track_data,
    frame_indices,
    output_path,
    fps=10.0,
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
    panel_h = max(112, int(frame_h * 0.18))
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
                        f"track {track_id} | {label} | {source_label}",
                        (max(8, x1), text_y),
                        0.72,
                        source_color,
                        2,
                    )

            header = f"{video_id} | track {track_id} | frame {frame_index:05d}"
            cv2.putText(img, header, (12, 30), font, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, header, (12, 30), font, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

            panel = cv2.resize(first_img[:1, :1], (frame_w, panel_h))
            panel[:] = (24, 24, 24)
            cv2.putText(
                panel,
                f"relative motion: {motion_state}",
                (18, 30),
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
            cv2.putText(panel, metrics, (18, 58), font, 0.58, (210, 210, 210), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (18, 72), (42, 88), _VIS_OBSERVED_COLOR, -1)
            cv2.putText(panel, "observed", (48, 87), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (142, 72), (166, 88), _VIS_REPAIRED_COLOR, -1)
            cv2.putText(panel, "repaired", (172, 87), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (266, 72), (290, 88), _VIS_ABSENT_COLOR, -1)
            cv2.putText(panel, "absent", (296, 87), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            _draw_track_progress_bar(cv2, panel, frame_indices, frame_index, track_frames, frame_w)
            writer.write(cv2.vconcat([img, panel]))
    finally:
        writer.release()

    return str(output_path), "rendered"


def _render_relative_motion_track_videos(relative_motion_video_result, output_root, fps=10.0):
    frame_indices, tracks = _relative_motion_track_index(relative_motion_video_result)
    video_id = str(relative_motion_video_result.get("video_id", ""))
    rendered = []
    skipped = []
    for track_id, track_data in sorted(tracks.items()):
        output_path = Path(output_root) / video_id / f"track_{track_id:04d}_relative_motion.mp4"
        path, status = _render_relative_motion_track_video(
            relative_motion_video_result=relative_motion_video_result,
            track_id=track_id,
            track_data=track_data,
            frame_indices=frame_indices,
            output_path=output_path,
            fps=fps,
        )
        row = {
            "video_id": video_id,
            "track_id": int(track_id),
            "label": str(track_data.get("label", "unknown")),
            "num_present_frames": len(track_data.get("frames", {})),
            "status": status,
        }
        if path:
            row["visualization_path"] = path
            rendered.append(row)
        else:
            skipped.append(row)
    return rendered, skipped


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
    all_rendered = []
    all_skipped = []
    progress = tqdm(relative_motion, desc="[step 8visual] relative_motion_tracks", unit="video")
    for video_result in progress:
        video_id = str(video_result.get("video_id", ""))
        progress.set_postfix_str(video_id, refresh=False)
        rendered, skipped = _render_relative_motion_track_videos(
            relative_motion_video_result=video_result,
            output_root=output_root,
            fps=float(fps),
        )
        all_rendered.extend(rendered)
        all_skipped.extend(skipped)

    manifest = {
        "version": _RELATIVE_MOTION_VIS_VERSION,
        "num_videos": len(relative_motion),
        "fps": float(fps),
        "num_track_videos_rendered": len(all_rendered),
        "num_track_videos_skipped": len(all_skipped),
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
        "relative_motion_visualization_output_root": output_root,
    }


def step9_temporal_segmentation(ego_state, relative_motion_state):
    return {"videos": ego_state["videos"], "temporal_segments": []}


def step10_segment_object_motion(segment_state):
    return {"videos": segment_state["videos"], "segment_object_motion": []}
