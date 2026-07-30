import copy
import hashlib
import json
import math
import os
import re
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
_EGO_SYMBOL_PRIOR_VERSION = 7
_EGO_CUE_NAMES = (
    "ego_static",
    "ego_driving_forward",
    "ego_driving_backward",
    "ego_turning_left",
    "ego_turning_right",
    "ego_straight",
    "ego_accelerating",
    "ego_decelerating",
    "ego_motion_uncertain",
)
_EGO_SYMBOL_DEFAULT_CONFIG = {
    "candidate_static_speed_thresholds": [0.15, 0.25, 0.40],
    "candidate_lateral_thresholds": [0.08, 0.15, 0.30, 0.50],
    "candidate_yaw_thresholds": [0.015, 0.030, 0.060, 0.100],
    "candidate_acceleration_thresholds": [0.06, 0.12, 0.20],
    "acceleration_threshold": 0.12,
    "min_short_segment_frames": 3,
    "rapid_reversal_window_frames": 6,
    "max_candidates": 64,
    "threshold_search_rounds": 3,
    "threshold_refinement_top_k": 3,
    "threshold_refinement_factor": 0.5,
    "step7e_expensive_candidate_limit": 8,
    "score_weights": {
        "signal_fit_error": 1.0,
        "state_transitions": 0.75,
        "short_segment_count": 1.25,
        "short_segment_duration": 0.75,
        "rapid_left_right_reversals": 1.50,
        "longitudinal_state_transitions": 1.00,
        "forward_backward_reversals": 2.00,
        "acceleration_state_transitions": 0.75,
        "acceleration_deceleration_reversals": 1.50,
        "acceleration_signal_fit_error": 0.75,
        "action_complexity": 0.25,
    },
}
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
_RELATIVE_MOTION_VIS_VERSION = 5
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
_UNCERTAIN_SIGNAL_EVIDENCE_VERSION = 4
_TRACK_USEFULNESS_POLICY_VERSION = 2
_TRACK_USEFULNESS_PROTECTED_LABEL_TOKENS = (
    "traffic light",
    "traffic sign",
    "stop sign",
    "yield sign",
    "sign",
    "pedestrian",
    "person",
    "cyclist",
    "bicycle",
    "motorcycle",
    "emergency",
    "ambulance",
    "police",
    "fire truck",
    "construction",
    "barrier",
    "cone",
    "animal",
)
_TRACK_USEFULNESS_VEHICLE_LABEL_TOKENS = (
    "car",
    "truck",
    "bus",
    "van",
    "trailer",
    "vehicle",
)
_TRACK_USEFULNESS_THRESHOLDS = {
    "max_short_observations": 10,
    "max_tiny_bbox_area_px": 625.0,
    "min_far_depth": 45.0,
    "max_low_detection_score": 0.65,
    "max_weak_cue": 0.20,
    "min_near_depth_protection": 35.0,
    "max_ego_corridor_abs_x": 4.5,
    "max_ego_corridor_depth": 120.0,
    "min_large_bbox_area_px": 900.0,
    "min_strong_detection_score": 0.65,
    "min_informative_cue": 0.20,
    "min_approach_protection": 0.10,
    "min_raw_approach_depth_change": 0.50,
    "min_bbox_growth_ratio": 1.25,
    "min_raw_relative_speed": 1.0,
}
_CAUSAL_FILTER_OUT_VERSION = 2
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
    return Path(
        os.environ.get(
            "CAUVID_PIPELINE_OUTPUT_PATH",
            str(config.get_output_path("pipeline_output")),
        )
    )


_CACHE_STEP_DIR_PATTERN = re.compile(r"^\d{2}[a-z]?_.+")


def _relocated_cache_path(
    value,
    dataset_root,
    pipeline_root,
    *,
    video_id="",
    key_hint="",
):
    text = str(value).strip()
    if not text or "://" in text:
        return text
    source = Path(text)
    if source.exists() or not source.is_absolute():
        return text

    normalized = text.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    candidates = []
    dataset_root = Path(dataset_root) if dataset_root else None
    pipeline_root = Path(pipeline_root) if pipeline_root else None

    if dataset_root is not None:
        for marker in ("frames", "depth_maps"):
            if marker in parts:
                marker_index = parts.index(marker)
                candidates.append(dataset_root / marker / Path(*parts[marker_index + 1 :]))
        filename = source.name
        hint = str(key_hint).lower()
        if video_id and filename:
            if "image" in hint or source.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                candidates.append(dataset_root / "frames" / str(video_id) / filename)
            if "depth" in hint or filename.endswith("_depth.npz"):
                candidates.append(dataset_root / "depth_maps" / str(video_id) / filename)

    if pipeline_root is not None:
        for index, part in enumerate(parts):
            if _CACHE_STEP_DIR_PATTERN.match(part):
                candidates.append(pipeline_root / Path(*parts[index:]))
                break

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return text


def relocate_cached_payload(payload, dataset_root=None, pipeline_root=None):
    """Relocate absolute paths embedded by another host after cache copying."""

    changed_paths = []

    def visit(value, key_hint="", video_id=""):
        if isinstance(value, dict):
            current_video_id = str(value.get("video_id", video_id) or video_id)
            return {
                key: visit(child, key_hint=str(key), video_id=current_video_id)
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [visit(child, key_hint=key_hint, video_id=video_id) for child in value]
        if isinstance(value, str):
            relocated = _relocated_cache_path(
                value,
                dataset_root,
                pipeline_root,
                video_id=video_id,
                key_hint=key_hint,
            )
            if relocated != value:
                changed_paths.append({"from": value, "to": relocated})
            return relocated
        return value

    relocated_payload = visit(payload)
    return relocated_payload, changed_paths


def relocate_json_cache_file(path, dataset_root=None, pipeline_root=None):
    path = Path(path)
    if not path.exists():
        return None, []
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    relocated, changes = relocate_cached_payload(
        payload,
        dataset_root=dataset_root,
        pipeline_root=pipeline_root,
    )
    if changes:
        with path.open("w", encoding="utf-8") as file:
            json.dump(relocated, file, indent=2)
    return relocated, changes


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
    """Build the fixed decision-grouped reason rows used by the track video."""
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
        ("Discard", "invalid_trajectory", invalid_count, 1, f"invalid_issues={invalid_count}"),
        ("Repair", "repaired_trajectory_kept", repaired_count + merged_count, 1, f"repairs+merges={repaired_count + merged_count}"),
        ("Repair", "low_motion_significance", significance_count, 1, f"low_motion_reasons={significance_count}"),
        ("Keep", "valid_high_significance", valid_flag + high_flag, 2, f"valid={valid_flag}, high={high_flag}"),
        ("Keep", "valid_low_motion_retained", valid_flag + low_flag, 2, f"valid={valid_flag}, low={low_flag}"),
        ("Keep with uncertainty", "credible_but_uncertain", credibility_uncertainty_count, 1, f"uncertainty_evidence={credibility_uncertainty_count}"),
        ("Keep with uncertainty", "validation_uncertainty", uncertain_count, 1, f"uncertain_issues={uncertain_count}"),
        ("Keep with uncertainty", "significance_uncertainty", significance_count, 1, f"significance_reasons={significance_count}"),
    )
    rows = []
    for decision, reason_name, measured_value, threshold, evidence_text in specs:
        active = reason_name in active_codes
        rows.append(
            {
                "decision": decision,
                "reason": reason_name,
                "active": active,
                "measured_value": float(measured_value),
                "threshold": float(threshold),
                "distance_to_threshold": float(measured_value - threshold),
                "evidence": evidence_text,
            }
        )
    return rows


def _draw_decision_reason_table(cv2, panel, entries, active_decision, x, y, width, row_height=142):
    """Draw decision groups as ``reason, reason -> decision`` flows."""
    decision_order = ("Discard", "Repair", "Keep", "Keep with uncertainty")
    groups = {
        decision: [entry for entry in entries if entry.get("decision") == decision]
        for decision in decision_order
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    decision_w = max(190, int(width * 0.22))
    arrow_w = 58
    reasons_w = max(1, width - decision_w - arrow_w)
    inactive_color = (105, 105, 105)
    active_reason_color = (80, 220, 80)

    for row, decision in enumerate(decision_order):
        row_y = y + row * row_height
        row_mid = row_y + row_height // 2
        group = groups[decision]
        card_gap = 12
        card_width = max(1, (reasons_w - card_gap * max(0, len(group) - 1)) // max(1, len(group)))
        for index, entry in enumerate(group):
            reason = str(entry.get("reason", "")).replace("_", " ")
            active = bool(entry.get("active", False))
            color = active_reason_color if active else inactive_color
            thickness = 2 if active else 1
            card_x = x + index * (card_width + card_gap)
            card_right = card_x + card_width
            border_color = color if active else (70, 70, 70)
            fill_color = (30, 52, 30) if active else (29, 29, 29)
            cv2.rectangle(panel, (card_x, row_y + 9), (card_right, row_y + row_height - 9), fill_color, -1)
            cv2.rectangle(panel, (card_x, row_y + 9), (card_right, row_y + row_height - 9), border_color, thickness)

            name_scale = 0.88
            max_name_width = max(1, card_width - 16)
            while name_scale > 0.68 and cv2.getTextSize(reason, font, name_scale, thickness)[0][0] > max_name_width:
                name_scale -= 0.04
            cv2.putText(panel, reason, (card_x + 8, row_y + 48), font, name_scale, color, thickness, cv2.LINE_AA)

            value = float(entry.get("measured_value", 0.0))
            threshold = float(entry.get("threshold", 1.0))
            distance = float(entry.get("distance_to_threshold", value - threshold))
            metrics = f"value={value:.0f} | threshold>={threshold:.0f} | delta={distance:+.0f}"
            metric_scale = 0.58
            while metric_scale > 0.40 and cv2.getTextSize(metrics, font, metric_scale, 1)[0][0] > max_name_width:
                metric_scale -= 0.03
            cv2.putText(panel, metrics, (card_x + 8, row_y + 91), font, metric_scale, color, 1, cv2.LINE_AA)

            if index < len(group) - 1:
                cv2.putText(panel, ",", (card_right + 2, row_mid + 8), font, 0.82, (180, 180, 180), 2, cv2.LINE_AA)

        arrow_start = x + reasons_w + 8
        arrow_end = x + reasons_w + arrow_w - 10
        arrow_color = _VIS_DECISION_COLORS.get(decision, inactive_color) if decision == active_decision else inactive_color
        cv2.arrowedLine(panel, (arrow_start, row_mid), (arrow_end, row_mid), arrow_color, 3, cv2.LINE_AA, tipLength=0.25)

        box_x = x + reasons_w + arrow_w
        box_active = decision == active_decision
        decision_color = _VIS_DECISION_COLORS.get(decision, inactive_color)
        fill_color = tuple(max(18, int(channel * 0.24)) for channel in decision_color) if box_active else (32, 32, 32)
        border_color = decision_color if box_active else inactive_color
        cv2.rectangle(panel, (box_x, row_y + 9), (x + width, row_y + row_height - 9), fill_color, -1)
        cv2.rectangle(panel, (box_x, row_y + 9), (x + width, row_y + row_height - 9), border_color, 3 if box_active else 1)
        text_color = decision_color if box_active else inactive_color
        decision_scale = 0.82 if decision == "Keep with uncertainty" else 0.94
        cv2.putText(panel, decision, (box_x + 12, row_mid + 10), font, decision_scale, text_color, 3 if box_active else 2, cv2.LINE_AA)

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
    panel_h = max(820, int(frame_h * 0.75))
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
            if obj is not None:
                summary = (
                    f"motion={motion_state} | source={source_label} | filter={decision_status} | "
                    f"vx={_safe_float(obj.get('rel_vx', 0.0)):+.2f}  "
                    f"vz={_safe_float(obj.get('rel_vz', 0.0)):+.2f}  "
                    f"speed={_safe_float(obj.get('rel_speed', 0.0)):.2f}"
                )
            else:
                summary = f"motion=not present | source=absent | filter={decision_status}"
            cv2.putText(panel, summary, (18, 78), font, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(panel, "reasons -> corresponding decision", (18, 108), font, 0.55, (225, 225, 225), 1, cv2.LINE_AA)
            _draw_decision_reason_table(
                cv2,
                panel,
                decision_reason_entries,
                active_decision=decision_status,
                x=18,
                y=120,
                width=max(1, frame_w - 36),
                row_height=142,
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


def _clamp_unit(value):
    return float(max(0.0, min(1.0, _safe_float(value))))


def _linear_signal_descriptor(samples, evidence_confidence):
    """Summarize one numeric signal without assigning a motion hypothesis."""
    points = sorted(
        (
            int(frame_index),
            _safe_float(value),
        )
        for frame_index, value in samples
        if math.isfinite(_safe_float(value))
    )
    if not points:
        return {
            "trend": "unobserved",
            "level": "unobserved",
            "confidence": 0.0,
            "sample_count": 0,
            "start": None,
            "end": None,
            "delta": None,
            "slope_per_frame": None,
            "mean": None,
            "standard_deviation": None,
            "step_sign_consistency": 0.0,
            "linear_fit_coherence": 0.0,
        }

    frames = [row[0] for row in points]
    values = [row[1] for row in points]
    count = len(values)
    start = values[0]
    end = values[-1]
    delta = end - start
    mean_value = sum(values) / count
    variance = sum((value - mean_value) ** 2 for value in values) / count
    standard_deviation = math.sqrt(max(0.0, variance))
    frame_mean = sum(frames) / count
    denominator = sum((frame - frame_mean) ** 2 for frame in frames)
    slope = (
        sum(
            (frame - frame_mean) * (value - mean_value)
            for frame, value in points
        )
        / denominator
        if denominator > 0.0
        else 0.0
    )
    predictions = [
        mean_value + slope * (frame - frame_mean) for frame in frames
    ]
    rmse = math.sqrt(
        sum(
            (value - prediction) ** 2
            for value, prediction in zip(values, predictions)
        )
        / count
    )
    value_range = max(values) - min(values)
    fit_scale = max(
        1e-6,
        value_range,
        abs(delta),
        standard_deviation,
    )
    fit_coherence = _clamp_unit(1.0 - rmse / fit_scale)

    steps = [
        (right_value - left_value) / max(1, right_frame - left_frame)
        for (left_frame, left_value), (right_frame, right_value)
        in zip(points, points[1:])
    ]
    epsilon = max(
        1e-6,
        1e-3 * max(1.0, max(abs(value) for value in values)),
    )
    nonzero_steps = [step for step in steps if abs(step) > epsilon]
    if abs(delta) <= epsilon and abs(slope) <= epsilon:
        trend = "stable"
        step_sign_consistency = (
            1.0
            if not nonzero_steps
            else _clamp_unit(1.0 - len(nonzero_steps) / max(1, len(steps)))
        )
    else:
        expected_sign = 1.0 if slope >= 0.0 else -1.0
        agreeing = sum(
            1 for step in nonzero_steps if step * expected_sign > 0.0
        )
        step_sign_consistency = (
            agreeing / len(nonzero_steps) if nonzero_steps else 0.0
        )
        if step_sign_consistency >= 0.6:
            trend = "increasing" if expected_sign > 0.0 else "decreasing"
        else:
            trend = "mixed"

    positive = sum(value > epsilon for value in values)
    negative = sum(value < -epsilon for value in values)
    near_zero = count - positive - negative
    if near_zero == count:
        level = "near_zero"
    elif positive / count >= 0.7:
        level = "positive"
    elif negative / count >= 0.7:
        level = "negative"
    else:
        level = "mixed"

    sample_support = min(1.0, max(0.0, (count - 1) / 4.0))
    confidence = _clamp_unit(
        evidence_confidence
        * sample_support
        * (0.55 * fit_coherence + 0.45 * step_sign_consistency)
    )
    return {
        "trend": trend,
        "level": level,
        "confidence": confidence,
        "sample_count": count,
        "start": float(start),
        "end": float(end),
        "delta": float(delta),
        "slope_per_frame": float(slope),
        "mean": float(mean_value),
        "standard_deviation": float(standard_deviation),
        "step_sign_consistency": float(step_sign_consistency),
        "linear_fit_coherence": float(fit_coherence),
    }


def _observation_quality_descriptor(observations, statistics):
    count = int(statistics.get("num_observations", len(observations)))
    coverage_span = _safe_float(
        statistics.get("temporal_coverage_in_span", 0.0)
    )
    coverage_video = _safe_float(
        statistics.get("temporal_coverage_in_video", 0.0)
    )
    observed_ratio = _safe_float(statistics.get("observed_ratio", 0.0))
    repaired_ratio = _safe_float(statistics.get("repaired_ratio", 0.0))
    merged_ratio = _safe_float(statistics.get("merged_ratio", 0.0))
    motion_ratio = _safe_float(statistics.get("has_motion_ratio", 0.0))
    scores = [
        _safe_float(dict(row.get("uncertainty", {})).get("score", 0.0))
        for row in observations
    ]
    source_uncertainties = [
        _safe_float(
            dict(row.get("uncertainty", {})).get(
                "source_uncertainty", 0.0
            )
        )
        for row in observations
    ]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    mean_source_uncertainty = (
        sum(source_uncertainties) / len(source_uncertainties)
        if source_uncertainties
        else 1.0
    )
    sample_support = min(1.0, count / 5.0)
    confidence = _clamp_unit(
        sample_support
        * (
            0.25 * coverage_span
            + 0.20 * observed_ratio
            + 0.20 * mean_score
            + 0.20 * motion_ratio
            + 0.15 * (1.0 - mean_source_uncertainty)
        )
    )
    if count < 2:
        state = "single_observation"
    elif coverage_span < 0.5:
        state = "sparse_observations"
    elif repaired_ratio + merged_ratio >= 0.25:
        state = "repair_supported_observations"
    elif observed_ratio >= 0.8 and coverage_span >= 0.8:
        state = "dense_observed_samples"
    else:
        state = "mixed_observation_sources"
    return {
        "state": state,
        "confidence": confidence,
        "metrics": {
            "observation_count": count,
            "temporal_coverage_in_span": float(coverage_span),
            "temporal_coverage_in_video": float(coverage_video),
            "observed_ratio": float(observed_ratio),
            "repaired_ratio": float(repaired_ratio),
            "merged_ratio": float(merged_ratio),
            "samples_with_velocity_ratio": float(motion_ratio),
            "mean_sample_score": float(mean_score),
            "mean_source_uncertainty": float(mean_source_uncertainty),
        },
    }


def _axis_signal_descriptor(
    observations,
    *,
    position_index,
    velocity_key,
    axis,
    evidence_confidence,
):
    position_samples = []
    velocity_samples = []
    for row in observations:
        frame_index = int(row.get("frame_index", -1))
        position = list(row.get("position_3d", []))
        if frame_index >= 0 and len(position) > position_index:
            position_samples.append((frame_index, position[position_index]))
        motion = dict(row.get("motion", {}))
        if frame_index >= 0 and bool(motion.get("has_rel_motion", False)):
            velocity_samples.append((frame_index, motion.get(velocity_key, 0.0)))
    position = _linear_signal_descriptor(
        position_samples, evidence_confidence
    )
    velocity = _linear_signal_descriptor(
        velocity_samples, evidence_confidence
    )
    available = [
        row["confidence"]
        for row in (position, velocity)
        if row["sample_count"] > 0
    ]
    return {
        "axis": axis,
        "state": position["trend"],
        "confidence": (
            float(sum(available) / len(available)) if available else 0.0
        ),
        "position_signal": position,
        "velocity_signal": velocity,
    }


def _temporal_coherence_descriptor(
    observations,
    statistics,
    longitudinal,
    lateral,
    evidence_confidence,
):
    count = len(observations)
    max_gap = int(statistics.get("max_frame_gap", 0))
    mean_gap = _safe_float(statistics.get("mean_frame_gap", 0.0))
    gap_coherence = _clamp_unit(1.0 / (1.0 + max(0.0, mean_gap - 1.0)))
    signal_coherences = [
        _safe_float(
            descriptor.get("position_signal", {}).get(
                "linear_fit_coherence", 0.0
            )
        )
        for descriptor in (longitudinal, lateral)
    ]
    position_coherence = (
        sum(signal_coherences) / len(signal_coherences)
        if signal_coherences
        else 0.0
    )
    sample_support = min(1.0, max(0.0, (count - 1) / 4.0))
    confidence = _clamp_unit(
        evidence_confidence
        * sample_support
        * (0.55 * gap_coherence + 0.45 * position_coherence)
    )
    if count < 2:
        state = "limited_samples"
    elif max_gap <= 1:
        state = "continuous_samples"
    elif max_gap <= 3:
        state = "intermittent_samples"
    else:
        state = "fragmented_samples"
    return {
        "state": state,
        "confidence": confidence,
        "metrics": {
            "observation_count": count,
            "max_frame_gap": max_gap,
            "mean_frame_gap": float(mean_gap),
            "gap_coherence": float(gap_coherence),
            "position_fit_coherence": float(position_coherence),
        },
    }


def _relative_motion_state_cues(speed_samples, evidence_confidence):
    """Classify whole-track ego-relative motion with an uncertainty-aware zero band."""
    speeds = sorted(
        max(0.0, _safe_float(value))
        for _, value in speed_samples
        if math.isfinite(_safe_float(value))
    )
    confidence = _clamp_unit(evidence_confidence)
    if not speeds:
        return {
            "relative_static": 0.0,
            "relative_moving": 0.0,
            "relative_motion_uncertain": 1.0,
        }

    middle = len(speeds) // 2
    median_speed = (
        speeds[middle]
        if len(speeds) % 2
        else 0.5 * (speeds[middle - 1] + speeds[middle])
    )
    deviations = sorted(abs(value - median_speed) for value in speeds)
    deviation_middle = len(deviations) // 2
    mad = (
        deviations[deviation_middle]
        if len(deviations) % 2
        else 0.5 * (
            deviations[deviation_middle - 1] + deviations[deviation_middle]
        )
    )
    zero_band = (
        _REL_SPEED_THRESHOLD
        + min(0.7, 2.5 * mad)
        + 0.3 * (1.0 - confidence)
    )
    within_ratio = sum(value <= zero_band for value in speeds) / len(speeds)
    above_ratio = sum(value > zero_band for value in speeds) / len(speeds)
    sample_support = min(1.0, len(speeds) / 5.0)
    reliable = confidence >= 0.35 and len(speeds) >= 3

    if reliable and median_speed <= zero_band and within_ratio >= 0.7:
        static_score = _clamp_unit(confidence * sample_support * within_ratio)
        return {
            "relative_static": static_score,
            "relative_moving": 0.0,
            "relative_motion_uncertain": 0.0,
        }
    if reliable and median_speed > zero_band and above_ratio >= 0.7:
        moving_score = _clamp_unit(confidence * sample_support * above_ratio)
        return {
            "relative_static": 0.0,
            "relative_moving": moving_score,
            "relative_motion_uncertain": 0.0,
        }
    ambiguity = 1.0 - abs(within_ratio - above_ratio)
    uncertain_score = _clamp_unit(
        max(0.25, 1.0 - confidence, ambiguity * sample_support)
    )
    return {
        "relative_static": 0.0,
        "relative_moving": 0.0,
        "relative_motion_uncertain": uncertain_score,
    }


def _uncertain_track_signal_evidence(track_id, track_data, video_num_frames):
    observations = [
        _trajectory_observation_from_motion_object(
            frame_index,
            track_data["frames"][frame_index],
        )
        for frame_index in sorted(track_data.get("frames", {}))
    ]
    statistics = _trajectory_statistics(observations, video_num_frames)
    quality = _observation_quality_descriptor(observations, statistics)
    evidence_confidence = _safe_float(quality.get("confidence", 0.0))
    longitudinal = _axis_signal_descriptor(
        observations,
        position_index=2,
        velocity_key="rel_vz",
        axis="z",
        evidence_confidence=evidence_confidence,
    )
    lateral = _axis_signal_descriptor(
        observations,
        position_index=0,
        velocity_key="rel_vx",
        axis="x",
        evidence_confidence=evidence_confidence,
    )
    speed_samples = []
    for row in observations:
        motion = dict(row.get("motion", {}))
        if not bool(motion.get("has_rel_motion", False)):
            continue
        speed_samples.append(
            (
                int(row.get("frame_index", -1)),
                math.hypot(
                    _safe_float(motion.get("rel_vx", 0.0)),
                    _safe_float(motion.get("rel_vz", 0.0)),
                ),
            )
        )
    speed = _linear_signal_descriptor(speed_samples, evidence_confidence)

    def directional_cue(axis_descriptor, position_trend, velocity_level):
        sources = []
        position_signal = dict(
            axis_descriptor.get("position_signal", {})
        )
        velocity_signal = dict(
            axis_descriptor.get("velocity_signal", {})
        )
        if int(position_signal.get("sample_count", 0)) > 1:
            sources.append(
                _safe_float(position_signal.get("confidence", 0.0))
                if position_signal.get("trend") == position_trend
                else 0.0
            )
        if int(velocity_signal.get("sample_count", 0)) > 0:
            sources.append(
                _safe_float(velocity_signal.get("confidence", 0.0))
                if velocity_signal.get("level") == velocity_level
                else 0.0
            )
        return _clamp_unit(
            sum(sources) / len(sources) if sources else 0.0
        )

    relative_state_cues = _relative_motion_state_cues(
        speed_samples, evidence_confidence
    )
    observable_cues = {
        "leftness": directional_cue(
            lateral, "decreasing", "negative"
        ),
        "rightness": directional_cue(
            lateral, "increasing", "positive"
        ),
        "approach": directional_cue(
            longitudinal, "decreasing", "negative"
        ),
        "recede": directional_cue(
            longitudinal, "increasing", "positive"
        ),
        "acceleration": (
            _clamp_unit(speed.get("confidence", 0.0))
            if speed.get("trend") == "increasing"
            else 0.0
        ),
        "deceleration": (
            _clamp_unit(speed.get("confidence", 0.0))
            if speed.get("trend") == "decreasing"
            else 0.0
        ),
        **relative_state_cues,
    }
    if relative_state_cues["relative_static"] > 0.0:
        for cue_name in (
            "leftness",
            "rightness",
            "approach",
            "recede",
            "acceleration",
            "deceleration",
        ):
            observable_cues[cue_name] = 0.0
    return {
        "track_id": int(track_id),
        "primary_label": str(
            statistics.get(
                "primary_label", track_data.get("label", "unknown")
            )
        ),
        "observable_cues": observable_cues,
    }


def _normalized_track_label(label):
    return " ".join(
        str(label).strip().lower().replace("_", " ").replace("-", " ").split()
    )


def _label_has_token(label, tokens):
    normalized = _normalized_track_label(label)
    return any(token in normalized for token in tokens)


def _initial_track_usefulness_decision(
    track_id,
    track_data,
    evidence,
    video_num_frames,
):
    """Conservatively quarantine only unanimously weak, far, tiny vehicles."""
    observations = [
        _trajectory_observation_from_motion_object(
            frame_index,
            track_data["frames"][frame_index],
        )
        for frame_index in sorted(track_data.get("frames", {}))
    ]
    statistics = _trajectory_statistics(observations, video_num_frames)
    label = str(evidence.get("primary_label", track_data.get("label", "unknown")))
    observed_labels = sorted(
        {
            str(row.get("frame_label", "")).strip()
            for row in observations
            if str(row.get("frame_label", "")).strip()
        }
    )
    cues = {
        key: _clamp_unit(value)
        for key, value in dict(evidence.get("observable_cues", {})).items()
    }
    bbox_areas = [
        _bbox_area(row.get("bbox", []))
        for row in observations
        if _valid_bbox(row.get("bbox", [])) is not None
    ]
    depths = [
        _safe_float(row.get("position_3d", [0.0, 0.0, 0.0])[2])
        for row in observations
        if len(row.get("position_3d", [])) >= 3
        and _safe_float(row.get("position_3d", [0.0, 0.0, 0.0])[2]) > 0.0
    ]
    lateral_positions = [
        abs(_safe_float(row.get("position_3d", [0.0])[0]))
        for row in observations
        if len(row.get("position_3d", [])) >= 3
    ]
    scores = [
        _safe_float(dict(row.get("uncertainty", {})).get("score", 0.0))
        for row in observations
    ]
    relative_speeds = [
        abs(_safe_float(dict(row.get("motion", {})).get("rel_speed", 0.0)))
        for row in observations
        if bool(dict(row.get("motion", {})).get("has_rel_motion", False))
    ]
    count = len(observations)
    max_bbox_area = max(bbox_areas or [0.0])
    bbox_growth_ratio = (
        bbox_areas[-1] / max(1e-6, bbox_areas[0])
        if len(bbox_areas) >= 2
        else 1.0
    )
    min_depth = min(depths or [0.0])
    depth_change = (
        depths[-1] - depths[0] if len(depths) >= 2 else 0.0
    )
    min_abs_x = min(lateral_positions or [float("inf")])
    max_score = max(scores or [0.0])
    max_relative_speed = max(relative_speeds or [0.0])
    directional_cue_names = (
        "leftness",
        "rightness",
        "approach",
        "recede",
        "acceleration",
        "deceleration",
    )
    max_cue = max(
        (cues.get(name, 0.0) for name in directional_cue_names),
        default=0.0,
    )
    approach = _safe_float(cues.get("approach", 0.0))
    source_counts = dict(statistics.get("source_counts", {}))
    thresholds = _TRACK_USEFULNESS_THRESHOLDS

    protection_reasons = []
    if any(
        _label_has_token(
            observed_label, _TRACK_USEFULNESS_PROTECTED_LABEL_TOKENS
        )
        for observed_label in observed_labels or [label]
    ):
        protection_reasons.append("protected_semantic_category")
    vehicle_category = bool(observed_labels or [label]) and all(
        _label_has_token(
            observed_label, _TRACK_USEFULNESS_VEHICLE_LABEL_TOKENS
        )
        for observed_label in observed_labels or [label]
    )
    if not vehicle_category:
        protection_reasons.append("non_vehicle_category_preserved")
    if count > int(thresholds["max_short_observations"]):
        protection_reasons.append("sufficient_observation_count")
    if int(source_counts.get("repaired", 0)) > 0:
        protection_reasons.append("repair_supported_track")
    if depths and min_depth <= float(thresholds["min_near_depth_protection"]):
        protection_reasons.append("near_ego")
    if (
        depths
        and lateral_positions
        and min_abs_x <= float(thresholds["max_ego_corridor_abs_x"])
        and min_depth <= float(thresholds["max_ego_corridor_depth"])
    ):
        protection_reasons.append("ego_corridor")
    if max_bbox_area >= float(thresholds["min_large_bbox_area_px"]):
        protection_reasons.append("visually_large")
    if max_score >= float(thresholds["min_strong_detection_score"]):
        protection_reasons.append("strong_detection")
    if max_cue >= float(thresholds["min_informative_cue"]):
        protection_reasons.append("informative_signal_cue")
    if approach >= float(thresholds["min_approach_protection"]):
        protection_reasons.append("approach_evidence")
    if depth_change <= -float(
        thresholds["min_raw_approach_depth_change"]
    ):
        protection_reasons.append("raw_depth_approach")
    if bbox_growth_ratio >= float(thresholds["min_bbox_growth_ratio"]):
        protection_reasons.append("growing_bbox")
    if max_relative_speed >= float(thresholds["min_raw_relative_speed"]):
        protection_reasons.append("meaningful_raw_relative_speed")

    useless_conditions = {
        "short": count <= int(thresholds["max_short_observations"]),
        "tiny": max_bbox_area <= float(thresholds["max_tiny_bbox_area_px"]),
        "far": bool(depths)
        and min_depth >= float(thresholds["min_far_depth"]),
        "low_detection_confidence": max_score
        < float(thresholds["max_low_detection_score"]),
        "weak_cues": max_cue < float(thresholds["max_weak_cue"]),
        "vehicle_category": vehicle_category,
    }
    quarantine = not protection_reasons and all(useless_conditions.values())
    return {
        "track_id": int(track_id),
        "primary_label": label,
        "decision": "quarantine" if quarantine else "active",
        "protected": not quarantine,
        "policy_version": _TRACK_USEFULNESS_POLICY_VERSION,
        "reason_codes": (
            ["unanimous_short_tiny_far_weak_vehicle"]
            if quarantine
            else protection_reasons or ["conservative_default_keep"]
        ),
        "features": {
            "num_observations": count,
            "temporal_coverage_in_video": _safe_float(
                statistics.get("temporal_coverage_in_video", 0.0)
            ),
            "max_bbox_area_px": float(max_bbox_area),
            "bbox_growth_ratio": float(bbox_growth_ratio),
            "min_depth": float(min_depth) if depths else None,
            "depth_change": float(depth_change),
            "min_abs_lateral_position": (
                float(min_abs_x) if lateral_positions else None
            ),
            "max_detection_score": float(max_score),
            "max_relative_speed": float(max_relative_speed),
            "max_observable_cue": float(max_cue),
            "approach": float(approach),
        },
        "conditions": useless_conditions,
        "feedback_status": "pending_downstream_review",
    }


def _relative_signal_fingerprint(relative_video):
    digest = hashlib.sha256()
    digest.update(
        str(relative_video.get("video_id", "")).encode("utf-8")
    )
    for frame_offset, frame in enumerate(relative_video.get("frames", [])):
        frame_index = int(frame.get("frame_index", frame_offset))
        for obj in sorted(
            frame.get("objects", []),
            key=lambda row: (
                str(row.get("track_id", "")),
                str(row.get("object_index", "")),
            ),
        ):
            payload = {
                "frame_index": frame_index,
                "track_id": obj.get("track_id"),
                "object_index": obj.get("object_index"),
                "label": obj.get("frame_label", obj.get("label")),
                "bbox": obj.get("bbox", obj.get("box")),
                "position_3d": obj.get(
                    "position_3d", obj.get("relative_position_3d")
                ),
                "rel_vx": obj.get("rel_vx"),
                "rel_vz": obj.get("rel_vz"),
                "rel_speed": obj.get("rel_speed"),
                "has_rel_motion": obj.get("has_rel_motion"),
                "score": obj.get("score"),
                "source": obj.get("source"),
                "source_type": obj.get("source_type"),
            }
            digest.update(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
    return digest.hexdigest()


def _uncertain_signal_evidence_video(
    relative_video,
    source_signal_fingerprint=None,
):
    frame_indices, tracks = _relative_motion_track_index(relative_video)
    video_num_frames = int(
        relative_video.get("num_frames", len(frame_indices))
    )
    source_track_evidence = [
        (
            track_id,
            track_data,
            _uncertain_track_signal_evidence(
                track_id,
                track_data,
                video_num_frames,
            ),
        )
        for track_id, track_data in sorted(tracks.items())
    ]
    filter_decisions = [
        _initial_track_usefulness_decision(
            track_id,
            track_data,
            evidence,
            video_num_frames,
        )
        for track_id, track_data, evidence in source_track_evidence
    ]
    decision_by_track = {
        int(row["track_id"]): row for row in filter_decisions
    }
    track_evidence = [
        evidence
        for track_id, _, evidence in source_track_evidence
        if decision_by_track[int(track_id)]["decision"] == "active"
    ]
    quarantined_evidence = [
        evidence
        for track_id, _, evidence in source_track_evidence
        if decision_by_track[int(track_id)]["decision"] == "quarantine"
    ]
    return {
        "version": _UNCERTAIN_SIGNAL_EVIDENCE_VERSION,
        "evidence_type": "uncertain_signal_evidence",
        "abstraction_level": "low_level_observable_signal",
        "video_id": str(relative_video.get("video_id", "")),
        "source_signal_fingerprint": (
            source_signal_fingerprint
            or _relative_signal_fingerprint(relative_video)
        ),
        "num_frames": video_num_frames,
        "num_source_tracks": len(source_track_evidence),
        "num_tracks": len(track_evidence),
        "num_active_tracks": len(track_evidence),
        "num_quarantined_tracks": len(quarantined_evidence),
        "num_observations": sum(
            len(track_data.get("frames", {}))
            for track_data in tracks.values()
        ),
        "semantic_motion_classification": False,
        "symbolic_reasoning": False,
        "cue_names": [
            "leftness",
            "rightness",
            "approach",
            "recede",
            "acceleration",
            "deceleration",
        ],
        "track_signal_evidence": track_evidence,
        "quarantined_track_signal_evidence": quarantined_evidence,
        "track_usefulness_filter": {
            "policy_version": _TRACK_USEFULNESS_POLICY_VERSION,
            "policy_kind": "conservative_initial_unanimous_evidence_gate",
            "mode": "quarantine_not_delete",
            "thresholds": dict(_TRACK_USEFULNESS_THRESHOLDS),
            "num_source_tracks": len(source_track_evidence),
            "num_active_tracks": len(track_evidence),
            "num_quarantined_tracks": len(quarantined_evidence),
            "decisions": filter_decisions,
        },
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


def _trajectory_reality_validation(
    observations,
    statistics,
    uncertainty,
    *,
    thresholds=None,
):
    active_thresholds = dict(_TRAJECTORY_VALIDATION_THRESHOLDS)
    if thresholds:
        active_thresholds.update(
            {
                key: value
                for key, value in dict(thresholds).items()
                if key in active_thresholds
            }
        )
    thresholds = active_thresholds
    step_metrics = _trajectory_step_metrics(observations)
    issues = []
    label_counts = dict(statistics.get("label_counts", {}))
    num_observations = int(statistics.get("num_observations", len(observations)))
    max_frame_gap = int(statistics.get("max_frame_gap", 0))
    has_motion_ratio = _safe_float(statistics.get("has_motion_ratio", 0.0))
    repaired_count = int(statistics.get("repaired_count", 0))
    merged_count = int(statistics.get("merged_count", 0))

    physically_invalid = []
    for row in observations:
        position = list(row.get("position_3d", []))
        try:
            position_values = [float(value) for value in position]
        except (TypeError, ValueError):
            position_values = []
        bbox = list(row.get("bbox", row.get("box", [])))
        try:
            bbox_values = [float(value) for value in bbox]
        except (TypeError, ValueError):
            bbox_values = []
        invalid_position = (
            len(position_values) < 3
            or not all(math.isfinite(value) for value in position_values)
            or position_values[2] < 0
        )
        invalid_bbox = bool(bbox) and (
            len(bbox_values) != 4
            or not all(math.isfinite(value) for value in bbox_values)
            or bbox_values[2] <= bbox_values[0]
            or bbox_values[3] <= bbox_values[1]
        )
        if invalid_position or invalid_bbox:
            physically_invalid.append(int(row.get("frame_index", -1)))
    if physically_invalid:
        issues.append(
            _validation_issue(
                "physical_invalidity",
                "invalid",
                "Trajectory contains nonfinite, negative-depth, or malformed geometry.",
                {"frame_ids": physically_invalid[:20], "count": len(physically_invalid)},
            )
        )
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
        "notes": "Post-pattern validation decision for symbolic-layer admission; it preserves provenance and reasons for explanation.",
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


def step7_train_eval_split(position_state, train_ratio=4, eval_ratio=1):
    """Create a deterministic video-level train/eval split before Step 7A."""
    video_ids = sorted(
        {str(video_id) for video_id in position_state.get("videos", []) if str(video_id)},
        key=lambda video_id: (hashlib.sha256(video_id.encode("utf-8")).hexdigest(), video_id),
    )
    total_ratio = max(1, int(train_ratio) + int(eval_ratio))
    if len(video_ids) <= 1:
        eval_count = 0
    else:
        eval_count = max(1, int(round(len(video_ids) * int(eval_ratio) / total_ratio)))
        eval_count = min(len(video_ids) - 1, eval_count)
    eval_ids = sorted(video_ids[:eval_count])
    train_ids = sorted(video_ids[eval_count:])
    payload = {
        "version": 1,
        "stage": "07_train_eval_split",
        "strategy": "deterministic_sha256_video_split",
        "requested_ratio": "4:1",
        "train_video_ids": train_ids,
        "eval_video_ids": eval_ids,
        "num_train_videos": len(train_ids),
        "num_eval_videos": len(eval_ids),
        "num_videos": len(video_ids),
    }
    output_root = get_pipeline_output_root() / "07_train_eval_split"
    output_root.mkdir(parents=True, exist_ok=True)
    split_path = output_root / "train_eval_split.json"
    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[step 7 split] videos={len(video_ids)} train={len(train_ids)} "
        f"eval={len(eval_ids)} ratio=4:1",
        flush=True,
    )
    return {
        **position_state,
        "step7_train_eval_split": payload,
        "step7_train_video_ids": train_ids,
        "step7_eval_video_ids": eval_ids,
        "step7_train_eval_split_path": str(split_path),
    }


def step7a_axis_threshold_segmentation(position_state):
    """Enumerate stable multi-threshold plateaus for ego vz/vx segmentation."""
    from src.exp_july.perception.ego_axis_threshold_segmentation import VERSION, materialize_enabled_candidates, render_all_video_plateau_scatter, render_segment_count_chart, segment_video
    from src.exp_july.perception.ego_axis_threshold_visualization import render_axis_segmentation_mp4, render_eval_candidate_filter_comparisons, render_eval_signal_segmentation_chart

    signal_state = step7_ego_motion(position_state)
    ego_motion = list(signal_state.get("ego_motion", []))
    step7a_config = driving_pipeline_config.get_step7a_axis_threshold_segmentation_cfg()
    vx_seg_max_count = int(step7a_config["vx_seg_max_count"])
    vz_seg_max_count = int(step7a_config["vz_seg_max_count"])
    max_plateau_middle_th_vx = float(step7a_config["max_plateau_middle_th_vx"])
    max_plateau_middle_th_vz = float(step7a_config["max_plateau_middle_th_vz"])
    plateau_min_n_values = int(step7a_config["plateau_min_n_values"])
    noise_tolerance_frames_vx = int(step7a_config["noise_tolerance_frames_vx"])
    noise_tolerance_frames_vz = int(step7a_config["noise_tolerance_frames_vz"])
    bridge_config_by_axis = {
        axis: {
            "bridge_total_max_frames": int(step7a_config[f"bridge_total_max_frames_{axis}"]),
            "anchor_min_frames": int(step7a_config[f"anchor_min_frames_{axis}"]),
            "bridge_max_segments": int(step7a_config[f"bridge_max_segments_{axis}"]),
            "bridge_max_anchor_ratio": float(step7a_config[f"bridge_max_anchor_ratio_{axis}"]),
        }
        for axis in ("vx", "vz")
    }
    filter_comparison_max_candidates = int(step7a_config["filter_comparison_max_candidates"])
    visualization_max_eval_videos = int(
        step7a_config.get("visualization_max_eval_videos", 3)
    )
    consensus_min_segment_length_vx = int(
        step7a_config.get("consensus_min_segment_length_vx", 6)
    )
    consensus_min_segment_length_vz = int(
        step7a_config.get("consensus_min_segment_length_vz", 6)
    )
    output_root = get_pipeline_output_root() / "07a_ego_axis_threshold_segmentation"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "eval").mkdir(parents=True, exist_ok=True)
    train_video_ids = set(str(value) for value in position_state.get("step7_train_video_ids", []))
    eval_video_ids = set(str(value) for value in position_state.get("step7_eval_video_ids", []))
    visualized_eval_video_ids = set(
        sorted(eval_video_ids)[:visualization_max_eval_videos]
    )
    results = []
    cached_videos = 0
    for ego_video in tqdm(ego_motion, desc="[step 7a] axis_threshold_segmentation", unit="video"):
        video_id = str(ego_video.get("video_id", ""))
        source_fingerprint = hashlib.sha256(json.dumps({
            "frames": ego_video.get("frames", []),
            "plateau_min_n_values": plateau_min_n_values,
            "noise_tolerance_frames_vx": noise_tolerance_frames_vx,
            "noise_tolerance_frames_vz": noise_tolerance_frames_vz,
            "bridge_config_by_axis": bridge_config_by_axis,
            "consensus_min_segment_length_vx": consensus_min_segment_length_vx,
            "consensus_min_segment_length_vz": consensus_min_segment_length_vz,
        }, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
        data_split = "eval" if video_id in eval_video_ids else "train"
        video_output_root = output_root / data_split / video_id
        path = video_output_root / "axis_threshold_segmentation.json"
        cached = None
        if path.exists():
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if int(candidate.get("version", 0)) == VERSION and str(candidate.get("source_fingerprint", "")) == source_fingerprint:
                    cached = candidate
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                cached = None
        recomputed = cached is None
        if recomputed:
            cached = segment_video(
                ego_video,
                vx_noise_tolerance_frames=noise_tolerance_frames_vx,
                vz_noise_tolerance_frames=noise_tolerance_frames_vz,
                vx_bridge_config=bridge_config_by_axis["vx"],
                vz_bridge_config=bridge_config_by_axis["vz"],
                plateau_min_n_values=plateau_min_n_values,
                vx_consensus_min_segment_length=consensus_min_segment_length_vx,
                vz_consensus_min_segment_length=consensus_min_segment_length_vz,
            )
            cached["source_fingerprint"] = source_fingerprint
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            cached_videos += 1
        chart_path = path.parent / "axis_threshold_segment_counts.png"
        if video_id in visualized_eval_video_ids and (recomputed or not chart_path.exists()):
            render_segment_count_chart(cached, chart_path)
        cached["data_split"] = data_split
        cached["output_directory"] = str(video_output_root)
        cached["segment_count_chart"] = (
            str(chart_path) if video_id in visualized_eval_video_ids else None
        )
        path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
        results.append(cached)
    train_results = [row for row in results if str(row.get("video_id", "")) in train_video_ids]
    eval_results = [row for row in results if str(row.get("video_id", "")) in eval_video_ids]
    visual_eval_results = [
        row for row in eval_results
        if str(row.get("video_id", "")) in visualized_eval_video_ids
    ]
    # Compatibility fallback for direct Step 7A calls without the pre-split stage.
    if not train_results and not eval_results:
        train_results = results
    overall_scatter = render_all_video_plateau_scatter(
        train_results,
        output_root / "all_videos_plateau_scatter.png",
        eval_results=eval_results,
        vx_seg_max_count=vx_seg_max_count,
        vz_seg_max_count=vz_seg_max_count,
        max_plateau_middle_th_vx=max_plateau_middle_th_vx,
        max_plateau_middle_th_vz=max_plateau_middle_th_vz,
    )
    for result in results:
        materialize_enabled_candidates(result, overall_scatter)
        result_path = (
            Path(result["output_directory"]) / "axis_threshold_segmentation.json"
        )
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    ego_by_video = {str(row.get("video_id", "")): row for row in ego_motion}
    visualization_mp4s = []
    signal_segmentation_charts = []
    filter_comparison_visualizations = []
    for result in tqdm(visual_eval_results, desc="[step 7a] eval visualizations", unit="video"):
        video_id = str(result.get("video_id", ""))
        video_output_root = output_root / "eval" / video_id
        visualization_path = video_output_root / "axis_segmentation_visualization.mp4"
        signal_chart_path = video_output_root / "axis_signal_segmentation.png"
        filter_comparison_root = video_output_root / "candidate_filter_comparisons"
        visualization_fingerprint = hashlib.sha256(json.dumps({
            "version": VERSION,
            "visualization_layout": "eval_only_enabled_candidates_no_final_v10",
            "source_fingerprint": result.get("source_fingerprint"),
            "confidence_points": [row for row in overall_scatter.get("points", []) if str(row.get("video_id", "")) == video_id],
        }, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
        signal_chart_fingerprint = hashlib.sha256(json.dumps({
            "visualization_fingerprint": visualization_fingerprint,
            "signal_chart_layout": "k_by_2_compact_titles_enabled_disabled_v4",
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        filter_comparison_fingerprint = hashlib.sha256(json.dumps({
            "source_fingerprint": result.get("source_fingerprint"),
            "layout": "4x1_viridis_gradient_confidence_v8",
            "max_candidates_per_axis": filter_comparison_max_candidates,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        cached_visual = result.get("visualization", {})
        if (visualization_path.exists() and cached_visual.get("status") == "rendered"
                and cached_visual.get("source_fingerprint") == visualization_fingerprint):
            visual = cached_visual
        else:
            visual = render_axis_segmentation_mp4(result, ego_by_video.get(video_id, {}), overall_scatter, visualization_path, show_final=False, step_label="7A")
            visual["source_fingerprint"] = visualization_fingerprint
        cached_signal_chart = result.get("signal_segmentation_chart", {})
        if (signal_chart_path.exists() and cached_signal_chart.get("status") == "rendered"
                and cached_signal_chart.get("source_fingerprint") == signal_chart_fingerprint):
            signal_chart = cached_signal_chart
        else:
            signal_chart = render_eval_signal_segmentation_chart(result, overall_scatter, signal_chart_path)
            signal_chart["source_fingerprint"] = signal_chart_fingerprint
        cached_filter_comparison = result.get("candidate_filter_comparisons", {})
        cached_filter_paths = [Path(row.get("path", "")) for row in cached_filter_comparison.get("charts", [])]
        expected_filter_charts = sum(min(
            filter_comparison_max_candidates,
            len(result.get(f"{axis}_segmentation", {}).get("threshold_candidates", [])),
        ) for axis in ("vx", "vz"))
        if (cached_filter_comparison.get("status") == "rendered"
                and cached_filter_comparison.get("source_fingerprint") == filter_comparison_fingerprint
                and len(cached_filter_paths) == expected_filter_charts
                and all(path.is_file() for path in cached_filter_paths)):
            filter_comparison = cached_filter_comparison
        else:
            filter_comparison = render_eval_candidate_filter_comparisons(
                result, filter_comparison_root,
                max_candidates=filter_comparison_max_candidates,
            )
            filter_comparison["source_fingerprint"] = filter_comparison_fingerprint
        result["visualization"] = visual
        result["signal_segmentation_chart"] = signal_chart
        result["candidate_filter_comparisons"] = filter_comparison
        visualization_mp4s.append(visual)
        signal_segmentation_charts.append(signal_chart)
        filter_comparison_visualizations.append(filter_comparison)
        (video_output_root / "axis_threshold_segmentation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest = {
        "version": VERSION,
        "stage": "7a_ego_axis_threshold_segmentation",
        "method": "confidence_weighted_candidate_consensus_dp",
        "num_videos": len(results),
        "num_frames": sum(int(row.get("num_frames", 0)) for row in results),
        "threshold_candidates_per_axis": 100,
        "vx_seg_max_count": vx_seg_max_count,
        "vz_seg_max_count": vz_seg_max_count,
        "max_plateau_middle_th_vx": max_plateau_middle_th_vx,
        "max_plateau_middle_th_vz": max_plateau_middle_th_vz,
        "plateau_min_n_values": plateau_min_n_values,
        "noise_tolerance_frames_vx": noise_tolerance_frames_vx,
        "noise_tolerance_frames_vz": noise_tolerance_frames_vz,
        "bridge_config_by_axis": bridge_config_by_axis,
        "filter_comparison_max_candidates": filter_comparison_max_candidates,
        "visualization_max_eval_videos": visualization_max_eval_videos,
        "consensus_min_segment_length_vx": consensus_min_segment_length_vx,
        "consensus_min_segment_length_vz": consensus_min_segment_length_vz,
        "visualized_eval_video_ids": sorted(visualized_eval_video_ids),
        "configuration": step7a_config,
        "train_eval_split": copy.deepcopy(position_state.get("step7_train_eval_split", {})),
        "train_output_root": str(output_root / "train"),
        "eval_output_root": str(output_root / "eval"),
        "num_train_videos": len(train_results),
        "num_eval_videos": len(eval_results),
        "cached_videos": cached_videos,
        "segment_count_charts": [
            str(row["segment_count_chart"])
            for row in visual_eval_results
            if row.get("segment_count_chart")
        ],
        "visualization_scope": "eval_videos_only",
        "num_visualized_eval_videos": len(visualization_mp4s),
        "visualization_mp4s": visualization_mp4s,
        "num_signal_segmentation_charts": len(signal_segmentation_charts),
        "signal_segmentation_charts": signal_segmentation_charts,
        "num_filter_comparison_visualizations": sum(int(row.get("num_charts", 0)) for row in filter_comparison_visualizations),
        "candidate_filter_comparisons": filter_comparison_visualizations,
        "num_enabled_segmentation_candidates": sum(
            len(result.get(f"{axis}_segmentation", {}).get("enabled_segmentation_candidates", []))
            for result in results for axis in ("vx", "vz")
        ),
        "final_merge_performed": False,
        "final_merge_step": "7b",
        "num_qualifying_plateaus": sum(
            len(row.get(axis, {}).get("qualifying_plateaus", []))
            for row in results for axis in ("vx_segmentation", "vz_segmentation")
        ),
        "all_videos_plateau_scatter": overall_scatter,
    }
    manifest_path = output_root / "axis_threshold_segmentation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[step 7a] axis_threshold_segmentation videos={manifest['num_videos']} "
        f"frames={manifest['num_frames']} candidates=100x2 "
        f"plateaus={manifest['num_qualifying_plateaus']} "
        f"enabled_candidates={manifest['num_enabled_segmentation_candidates']} "
        f"final_merge=deferred_to_7b "
        f"cached={cached_videos} scatter={overall_scatter['path']}",
        flush=True,
    )
    return {
        **position_state,
        **signal_state,
        "step7_status": "7a_enabled_candidates",
        "step7_substeps": ["7a_axis_threshold_segmentation"],
        "ego_axis_threshold_segmentation": results,
        "ego_axis_threshold_segmentation_manifest": manifest,
        "ego_axis_threshold_segmentation_manifest_path": str(manifest_path),
        "ego_axis_threshold_segmentation_output_root": output_root,
        "ego_symbol_prior": [],
        "final_ego_symbols": [],
    }



def step7b_optimal_segmentation_selection(step7a_state):
    """Merge Step 7A enabled candidates into one final sequence per axis."""
    from src.exp_july.perception.ego_axis_threshold_segmentation import apply_semantic_candidate_confidence_correction, finalize_enabled_consensus, materialize_enabled_candidates, render_train_optimal_n_scatter, select_optimal_n_by_final_similarity
    from src.exp_july.perception.ego_axis_threshold_visualization import render_axis_segmentation_mp4

    config = driving_pipeline_config.get_step7a_axis_threshold_segmentation_cfg()
    vx_minimum = int(config.get("consensus_min_segment_length_vx", 6))
    vz_minimum = int(config.get("consensus_min_segment_length_vz", 6))
    semantic_penalty = float(config.get("semantic_opposite_transition_penalty", 0.5))
    candidate_results = list(step7a_state.get("ego_axis_threshold_segmentation", []))
    final_results = copy.deepcopy(candidate_results)
    output_root = get_pipeline_output_root() / "07b_ego_axis_consensus_segmentation"
    output_root.mkdir(parents=True, exist_ok=True)
    eval_visualization_root = output_root / "eval_visualizations"
    eval_visualization_root.mkdir(parents=True, exist_ok=True)
    visualization_max_eval_videos = int(config.get("visualization_max_eval_videos", 3))
    visualized_eval_ids = set(
        sorted(str(value) for value in step7a_state.get("step7_eval_video_ids", []))[:visualization_max_eval_videos]
    )
    ego_by_video = {
        str(row.get("video_id", "")): row
        for row in step7a_state.get("ego_motion", [])
    }
    visualization_mp4s = []
    plateau_audit = step7a_state.get(
        "ego_axis_threshold_segmentation_manifest", {}
    ).get("all_videos_plateau_scatter", {})
    final_ego_symbols = []
    for result in tqdm(final_results, desc="[step 7b] consensus_merge", unit="video"):
        has_materialized_candidates = all(
            "enabled_segmentation_candidates" in result.get(f"{axis}_segmentation", {})
            for axis in ("vx", "vz")
        )
        if not has_materialized_candidates:
            materialize_enabled_candidates(result, plateau_audit)
        apply_semantic_candidate_confidence_correction(
            result, opposite_transition_penalty=semantic_penalty,
        )
        finalize_enabled_consensus(
            result,
            None,
            vx_minimum_segment_length=vx_minimum,
            vz_minimum_segment_length=vz_minimum,
        )
        select_optimal_n_by_final_similarity(result)
        video_id = str(result.get("video_id", ""))
        data_split = str(result.get("data_split", "train"))
        video_output_root = output_root / data_split / video_id
        video_output_root.mkdir(parents=True, exist_ok=True)
        final_path = video_output_root / "final_axis_segmentation.json"
        result["step7b_output_directory"] = str(video_output_root)
        result["step7b_final_segmentation_path"] = str(final_path)
        final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if data_split == "eval" and video_id in visualized_eval_ids:
            visualization_path = (
                eval_visualization_root / f"{video_id}_final_consensus.mp4"
            )
            visual = render_axis_segmentation_mp4(
                result, ego_by_video.get(video_id, {}), plateau_audit,
                visualization_path, show_final=True, step_label="7B",
            )
            result["step7b_visualization"] = visual
            visualization_mp4s.append(visual)
            final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        final = copy.deepcopy(result.get("final_segmentation", {}))
        final_ego_symbols.append({
            "video_id": video_id,
            "status": str(final.get("status", "")),
            "source_step": "7b_consensus_merge",
            "method": "semantic_corrected_enabled_candidate_confidence_weighted_min_length_dp",
            "final_segmentation": final,
            "optimal_n_selection": copy.deepcopy(result.get("optimal_n_selection", {})),
            "provenance": {
                "step7a_candidate_source": str(result.get("output_directory", "")),
                "step7b_output": str(final_path),
            },
        })
    train_final_results = [
        result for result in final_results if str(result.get("data_split", "train")) == "train"
    ]
    eval_final_results = [
        result for result in final_results if str(result.get("data_split", "train")) == "eval"
    ]
    optimal_n_scatter = render_train_optimal_n_scatter(
        train_final_results, eval_final_results,
        output_root / "train_optimal_n_with_eval_scatter.png",
        vx_seg_max_count=int(config.get("vx_seg_max_count", 8)),
        vz_seg_max_count=int(config.get("vz_seg_max_count", 5)),
        max_plateau_middle_th_vx=float(config.get("max_plateau_middle_th_vx", 250.0)),
        max_plateau_middle_th_vz=float(config.get("max_plateau_middle_th_vz", 70.0)),
    )
    optimal_n_scatter_audit_path = output_root / "train_optimal_n_with_eval_scatter.json"
    optimal_n_scatter["audit_path"] = str(optimal_n_scatter_audit_path)
    optimal_n_scatter_audit_path.write_text(
        json.dumps(optimal_n_scatter, indent=2), encoding="utf-8",
    )
    manifest = {
        "version": 6,
        "stage": "7b_ego_axis_consensus_segmentation",
        "method": "semantic_corrected_enabled_candidate_confidence_weighted_min_length_dp",
        "num_videos": len(final_results),
        "vx_minimum_segment_length": vx_minimum,
        "vz_minimum_segment_length": vz_minimum,
        "semantic_opposite_transition_penalty": semantic_penalty,
        "semantic_rule_ids": ["no_direct_forward_backward_transition"],
        "num_semantically_penalized_candidates": sum(
            int(result.get("step7b_semantic_confidence_correction", {}).get("num_penalized_candidates", 0))
            for result in final_results
        ),
        "num_semantic_violations": sum(
            int(result.get("step7b_semantic_confidence_correction", {}).get("num_violations", 0))
            for result in final_results
        ),
        "num_input_enabled_candidates": sum(
            len(result.get(f"{axis}_segmentation", {}).get("enabled_segmentation_candidates", []))
            for result in candidate_results for axis in ("vx", "vz")
        ),
        "num_completed_axes": sum(
            result.get(f"{axis}_segmentation", {}).get("final_segmentation", {}).get("status") == "completed"
            for result in final_results for axis in ("vx", "vz")
        ),
        "num_unavailable_axes": sum(
            result.get(f"{axis}_segmentation", {}).get("final_segmentation", {}).get("status") != "completed"
            for result in final_results for axis in ("vx", "vz")
        ),
        "output_root": str(output_root),
        "optimal_n_scatter": optimal_n_scatter,
        "num_selected_train_optimal_n": int(optimal_n_scatter.get("num_train_optimal_points", 0)),
        "num_selected_eval_optimal_n": int(optimal_n_scatter.get("num_eval_optimal_points", 0)),
        "visualization_scope": "at_most_configured_eval_videos",
        "eval_visualization_output_root": str(eval_visualization_root),
        "eval_visualization_layout": "single_shared_folder_video_id_filenames",
        "visualized_eval_video_ids": sorted(visualized_eval_ids),
        "num_visualized_eval_videos": len(visualization_mp4s),
        "visualization_mp4s": visualization_mp4s,
        "videos": [
            {
                "video_id": str(result.get("video_id", "")),
                "data_split": str(result.get("data_split", "train")),
                "path": str(result.get("step7b_final_segmentation_path", "")),
                "status": str(result.get("final_segmentation", {}).get("status", "")),
            }
            for result in final_results
        ],
    }
    manifest_path = output_root / "consensus_segmentation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[step 7b] consensus_merge videos={manifest['num_videos']} "
        f"enabled_candidates={manifest['num_input_enabled_candidates']} "
        f"semantic_penalized={manifest['num_semantically_penalized_candidates']} "
        f"semantic_violations={manifest['num_semantic_violations']} "
        f"optimal_n_train={manifest['num_selected_train_optimal_n']} "
        f"optimal_n_eval={manifest['num_selected_eval_optimal_n']} "
        f"completed_axes={manifest['num_completed_axes']} "
        f"unavailable_axes={manifest['num_unavailable_axes']}",
        flush=True,
    )
    return {
        **step7a_state,
        "step7_status": "7b_final_consensus",
        "step7_substeps": list(step7a_state.get("step7_substeps", [])) + ["7b_consensus_merge"],
        "ego_axis_final_segmentation": final_results,
        "ego_axis_consensus_segmentation_manifest": manifest,
        "ego_axis_consensus_segmentation_manifest_path": str(manifest_path),
        "ego_axis_consensus_segmentation_output_root": output_root,
        "final_ego_symbols": final_ego_symbols,
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


def _trajectory_motion_evidence_video(
    relative_video,
    ego_video=None,
    protected_by_track=None,
    validation_thresholds=None,
):
    frame_indices, tracks = _relative_motion_track_index(relative_video)
    protected_by_track = dict(protected_by_track or {})
    video_num_frames = int(relative_video.get("num_frames", len(frame_indices)))
    trajectories = []
    for track_id, track_data in sorted(tracks.items()):
        observations = [
            _trajectory_observation_from_motion_object(frame_index, track_data["frames"][frame_index])
            for frame_index in sorted(track_data.get("frames", {}))
        ]
        statistics = _trajectory_statistics(observations, video_num_frames)
        uncertainty = _trajectory_uncertainty(observations, statistics)
        validation = _trajectory_reality_validation(
            observations,
            statistics,
            uncertainty,
            thresholds=validation_thresholds,
        )
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
        protection = protected_by_track.get(int(track_id), {})
        original_fact_decision = str(fact_decision.get("decision", ""))
        if protection:
            final_protection_decision = (
                "Keep_with_uncertainty"
                if original_fact_decision == "Discard"
                else original_fact_decision
            )
            protection["original_decision_before_protection"] = original_fact_decision
            protection["trajectory_decision"] = original_fact_decision
            protection["final_decision_after_protection"] = final_protection_decision
            protection["send_to_motion_signal_refinement"] = original_fact_decision == "Discard"
            fact_decision["symbol_grounded_protected"] = True
            fact_decision["original_decision_before_protection"] = original_fact_decision
            fact_decision["trajectory_decision"] = original_fact_decision
            fact_decision["final_decision_after_protection"] = final_protection_decision
            fact_decision["send_to_motion_signal_refinement"] = original_fact_decision == "Discard"
            fact_decision["decision_reasons"].append(
                {
                    "kind": "symbol_grounded_protection",
                    "message": "A deterministically grounded Step 8A semantic rule protects this object track.",
                    "matched_rule_ids": list(protection.get("matched_rule_ids", [])),
                    "grounding_evidence": list(protection.get("grounding_evidence", [])),
                    "protection_reason": str(protection.get("protection_reason", "")),
                }
            )
            if original_fact_decision == "Discard":
                fact_decision["decision"] = "Keep with uncertainty"
                fact_decision["status"] = "Keep with uncertainty"
                fact_decision["symbolic_layer_eligible"] = True
        trajectories.append(
            {
                "track_id": int(track_id),
                "symbol_grounded_protected": bool(protection),
                "symbol_grounded_protection": protection,
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
    frames_dir = dataset_root / "frames"
    all_videos = sorted(config.get_mini_video_ids())
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
        f"[step 1] from videos={video_dir} frames={frames_dir} \n"
        f"[step 1] (dataset=driving_mini, total_in_dataset={len(all_videos)})"
    )
    detection_args = {
        "video_ids": videos,
        "model_name": driving_pipeline_config.DRIVING_MINI_OD_MODEL,
        "classes": driving_pipeline_config.DRIVING_MINI_OD_CLASSES,
        "output_root": get_pipeline_output_root() / "01_driving_mini_detection",
        "od_calibration_policy": {},
        "device": "cuda:0",
        "force_recompute": False,
        "render_video": driving_pipeline_config.get_detection_render_video_enabled(default=True),
        "check_cache": driving_pipeline_config.get_detection_check_cache_enabled(default=False),
        "enable_candidate_branch": driving_pipeline_config.get_detection_candidate_branch_enabled(default=False),
        "skip_step": driving_pipeline_config.get_detection_skip_step_enabled(default=False),
        "reuse_copied_cache": os.environ.get(
            "CAUVID_REUSE_COPIED_STEP1_7_CACHE",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"},
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
        "device": "cuda:0",
        "force_recompute": False,
        "force_recompute_depth": False,
    }
    ego_motion_args = {
        "output_root": get_pipeline_output_root() / "07_driving_mini_ego_motion",
        "force_recompute": False,
        "smoothing_window": driving_pipeline_config.get_ego_motion_smoothing_window(default=5),
        "static_adjust_cfg": driving_pipeline_config.get_ego_static_adjustment_cfg(),
        "render_video": driving_pipeline_config.get_ego_motion_render_video_enabled(default=True),
        "flow_device": "cuda:0",
    }
    return {
        "videos": videos,
        "dataset_name": "driving_mini",
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
    reuse_copied_cache = bool(run_args.pop("reuse_copied_cache", True))
    render_video = bool(run_args.get("render_video", True))
    check_cache = bool(run_args.get("check_cache", False))
    candidate_branch_enabled = bool(run_args.get("enable_candidate_branch", False))
    output_root = Path(run_args["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[step 2] model={model_name}")
    print(f"[step 2] classes={len(classes)} render_video={render_video} check_cache={check_cache}")
    print(f"[step 2] output_root={output_root}")

    manifest_path = output_root / "detection_manifest.json"
    use_cached_detection = skip_step or (
        reuse_copied_cache and manifest_path.exists()
    )
    if use_cached_detection:
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
            local_detections_path = output_root / video_id / "detections.json"
            stored_path = str(entry.get("detections_json", "")).strip()
            detections_path = local_detections_path if local_detections_path.exists() else Path(
                _relocated_cache_path(
                    stored_path,
                    env["dataset_root"],
                    get_pipeline_output_root(),
                    video_id=video_id,
                    key_hint="detections_json",
                )
            )
            with detections_path.open("r", encoding="utf-8") as f:
                video_result = json.load(f)
            if hasattr(detect_driving_mini, "_apply_candidate_branch_mode"):
                video_result = detect_driving_mini._apply_candidate_branch_mode(video_result, candidate_branch_enabled)
            detections.append(video_result)
        cache_reason = "configured_skip" if skip_step else "copied_cache_detected"
        print(
            f"[step 2] loaded cached detection results for {len(detections)} videos "
            f"reason={cache_reason}"
        )
    else:
        detections = detect_driving_mini.run(**run_args)
        print(f"[step 2] completed detection for {len(detections)} videos")

    accepted_count = sum(int(video_result.get("num_detections", 0)) for video_result in detections)
    candidate_count = sum(int(video_result.get("num_candidate_detections", 0)) for video_result in detections)
    normalized_detections = []
    rewritten_count = 0
    for video_result in detections:
        relocated_video_result, path_changes = relocate_cached_payload(
            video_result,
            dataset_root=env["dataset_root"],
            pipeline_root=get_pipeline_output_root(),
        )
        normalized_video_result, changed = normalize_detection_image_paths(
            relocated_video_result,
            env["dataset_root"],
        )
        if changed or path_changes:
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
        "dataset_root": env["dataset_root"],
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
        video_id = str(video_result.get("video_id", ""))
        progress.set_postfix_str(video_id, refresh=False)
        tracks_cache = output_root / video_id / "tracks.json"
        _, cache_path_changes = relocate_json_cache_file(
            tracks_cache,
            dataset_root=detection_state.get("dataset_root"),
            pipeline_root=get_pipeline_output_root(),
        )
        if cache_path_changes:
            print(f"[step 3] relocated cached paths video_id={video_id} paths={len(cache_path_changes)}")
        tracking_result = tracking_driving_mini.track_video(
            video_result=video_result,
            output_root=output_root,
            frame_rate=int(run_args.get("frame_rate", 10)),
            tracker_args=run_args.get("tracker_args"),
            force_recompute=bool(run_args.get("force_recompute", False)),
            render_video=render_video,
        )
        tracking_result, _ = relocate_cached_payload(
            tracking_result,
            dataset_root=detection_state.get("dataset_root"),
            pipeline_root=get_pipeline_output_root(),
        )
        tracking_results.append(tracking_result)

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
        "dataset_root": detection_state.get("dataset_root"),
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
            payload, cache_path_changes = relocate_json_cache_file(
                cache_path,
                dataset_root=tracking_state.get("dataset_root"),
                pipeline_root=get_pipeline_output_root(),
            )
            if cache_path_changes:
                print(f"[step 6] relocated cached paths video_id={video_id} paths={len(cache_path_changes)}")
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
        "dataset_root": tracking_state.get("dataset_root"),
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
    requested_eval_video_ids = sorted({
        str(value)
        for value in position_state.get("step7_eval_video_ids", [])
        if str(value)
    })
    restrict_visuals_to_eval = "step7_eval_video_ids" in position_state
    visualization_max_eval_videos = int(
        driving_pipeline_config.get_step7a_axis_threshold_segmentation_cfg().get(
            "visualization_max_eval_videos", 3
        )
    )
    eval_video_ids = set(
        requested_eval_video_ids[:visualization_max_eval_videos]
    )
    progress = tqdm(positions_3d, desc="[step 7] ego_motion", unit="video")
    for video_result in progress:
        video_id = str(video_result.get("video_id", ""))
        progress.set_postfix_str(video_id, refresh=False)
        cache_path = output_root / video_id / "ego_motion.json"
        cached_result = None
        if not bool(run_args.get("force_recompute", False)):
            payload, cache_path_changes = relocate_json_cache_file(
                cache_path,
                dataset_root=position_state.get("dataset_root"),
                pipeline_root=get_pipeline_output_root(),
            )
            if cache_path_changes:
                print(f"[step 7] relocated cached paths video_id={video_id} paths={len(cache_path_changes)}")
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
                    render_video=(
                        bool(run_args.get("render_video", True))
                        and (
                            not restrict_visuals_to_eval
                            or video_id in eval_video_ids
                        )
                    ),
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



def _median(values):
    rows = sorted(float(value) for value in values)
    if not rows:
        return 0.0
    middle = len(rows) // 2
    return rows[middle] if len(rows) % 2 else 0.5 * (rows[middle - 1] + rows[middle])


def _ego_frame_signal(frame, names):
    for name in names:
        if name not in frame:
            continue
        try:
            value = float(frame[name])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _ego_symbol_config(config=None):
    merged = copy.deepcopy(_EGO_SYMBOL_DEFAULT_CONFIG)
    supplied = dict(config or {})
    for key, value in supplied.items():
        if key == "score_weights":
            merged[key].update(dict(value or {}))
        else:
            merged[key] = copy.deepcopy(value)
    for key in (
        "candidate_static_speed_thresholds",
        "candidate_lateral_thresholds",
        "candidate_yaw_thresholds",
        "candidate_acceleration_thresholds",
    ):
        values = sorted(
            {
                float(value)
                for value in merged.get(key, [])
                if math.isfinite(float(value)) and float(value) > 0.0
            }
        )
        if not values:
            raise ValueError(f"Step 7A requires at least one positive value for {key}")
        merged[key] = values
    merged["max_candidates"] = max(1, int(merged.get("max_candidates", 64)))
    merged["threshold_search_rounds"] = max(1, int(merged.get("threshold_search_rounds", 3)))
    merged["threshold_refinement_top_k"] = max(1, int(merged.get("threshold_refinement_top_k", 3)))
    merged["step7e_expensive_candidate_limit"] = max(
        1, int(merged.get("step7e_expensive_candidate_limit", 8))
    )
    merged["threshold_refinement_factor"] = float(merged.get("threshold_refinement_factor", 0.5))
    if not 0.0 < merged["threshold_refinement_factor"] < 1.0:
        raise ValueError("threshold_refinement_factor must be between 0 and 1")
    merged["min_short_segment_frames"] = max(
        1, int(merged.get("min_short_segment_frames", 3))
    )
    merged["rapid_reversal_window_frames"] = max(
        1, int(merged.get("rapid_reversal_window_frames", 6))
    )
    merged["acceleration_threshold"] = max(
        0.0, float(merged.get("acceleration_threshold", 0.12))
    )
    return merged


def _ego_continuous_samples(ego_video):
    samples = []
    previous_speed = None
    for offset, source in enumerate(ego_video.get("frames", [])):
        frame = dict(source)
        vx = _ego_frame_signal(
            frame, ("refined_ego_vx", "ego_vx_smoothed", "ego_vx")
        )
        vz = _ego_frame_signal(
            frame, ("refined_ego_vz", "ego_vz_smoothed", "ego_vz")
        )
        yaw = _ego_frame_signal(
            frame,
            (
                "refined_ego_yaw_rate",
                "ego_yaw_rate_smoothed",
                "ego_yaw_rate",
            ),
        )
        available = vx is not None and vz is not None and bool(
            frame.get("has_ego_motion", vx is not None and vz is not None)
        )
        speed = math.hypot(vx, vz) if available else None
        speed_delta = (
            None
            if speed is None or previous_speed is None
            else float(speed - previous_speed)
        )
        if speed is not None:
            previous_speed = speed
        samples.append(
            {
                "frame_index": int(frame.get("frame_index", offset)),
                "ego_vx": vx,
                "ego_vz": vz,
                "ego_yaw_rate": yaw,
                "ego_speed": speed,
                "ego_speed_delta": speed_delta,
                "available": available,
            }
        )
    return samples


def _ego_threshold_candidates(config):
    candidates = []
    for static_threshold in config["candidate_static_speed_thresholds"]:
        for lateral_threshold in config["candidate_lateral_thresholds"]:
            for yaw_threshold in config["candidate_yaw_thresholds"]:
                candidates.append(
                    {
                        "static_speed_threshold": float(static_threshold),
                        "lateral_threshold": float(lateral_threshold),
                        "yaw_threshold": float(yaw_threshold),
                    }
                )
    candidates.sort(
        key=lambda row: (
            row["static_speed_threshold"],
            row["lateral_threshold"],
            row["yaw_threshold"],
        )
    )
    return candidates[: config["max_candidates"]]


def _ego_action(sample, thresholds):
    if not sample.get("available", False):
        return "unknown"
    speed = float(sample.get("ego_speed", 0.0))
    vx = float(sample.get("ego_vx", 0.0))
    vz = float(sample.get("ego_vz", 0.0))
    yaw = sample.get("ego_yaw_rate")
    if speed <= thresholds["static_speed_threshold"]:
        return "static"
    if yaw is not None and float(yaw) > thresholds["yaw_threshold"]:
        return "turning_left"
    if yaw is not None and float(yaw) < -thresholds["yaw_threshold"]:
        return "turning_right"
    if vx > thresholds["lateral_threshold"]:
        return "left"
    if vx < -thresholds["lateral_threshold"]:
        return "right"
    if vz > thresholds["static_speed_threshold"]:
        return "forward"
    if vz < -thresholds["static_speed_threshold"]:
        return "backward"
    # A valid low-margin signal should remain a soft numerical prediction.
    # Symbolic evidence may lower confidence later, but must not erase its sign.
    if vz > 1e-9:
        return "forward"
    if vz < -1e-9:
        return "backward"
    return "static"


def _contiguous_action_segments(samples, actions):
    if not samples:
        return []
    segments = []
    start = 0
    for index in range(1, len(actions) + 1):
        temporally_contiguous = bool(
            index < len(actions)
            and int(samples[index]["frame_index"])
            == int(samples[index - 1]["frame_index"]) + 1
        )
        if (
            index < len(actions)
            and actions[index] == actions[start]
            and temporally_contiguous
        ):
            continue
        segment_samples = samples[start:index]
        segments.append(
            {
                "segment_id": len(segments),
                "action": actions[start],
                "start_frame": int(segment_samples[0]["frame_index"]),
                "end_frame": int(segment_samples[-1]["frame_index"]),
                "duration_frames": len(segment_samples),
                "sample_indices": list(range(start, index)),
            }
        )
        start = index
    return segments


def _within_action_signal_fit_error(samples, actions):
    available_indices = [
        index for index, sample in enumerate(samples) if sample.get("available", False)
    ]
    if len(available_indices) <= 1:
        return 0.0
    dimensions = ("ego_vx", "ego_vz", "ego_yaw_rate")
    numerator = 0.0
    denominator = 0.0
    for dimension in dimensions:
        observed = [
            float(samples[index][dimension])
            for index in available_indices
            if samples[index].get(dimension) is not None
        ]
        if len(observed) <= 1:
            continue
        global_mean = sum(observed) / len(observed)
        total = sum((value - global_mean) ** 2 for value in observed)
        denominator += total
        grouped = {}
        for index in available_indices:
            value = samples[index].get(dimension)
            if value is not None:
                grouped.setdefault(actions[index], []).append(float(value))
        for values in grouped.values():
            mean = sum(values) / len(values)
            numerator += sum((value - mean) ** 2 for value in values)
    if denominator <= 1e-12:
        return 0.0
    return float(max(0.0, min(1.0, numerator / denominator)))



def _transition_counts(states, reversal_pair):
    valid = [state for state in states if state != "unknown"]
    transitions = sum(left != right for left, right in zip(valid, valid[1:]))
    reversals = sum(
        {left, right} == set(reversal_pair)
        for left, right in zip(valid, valid[1:])
    )
    return transitions, reversals


def _longitudinal_states(samples, thresholds):
    states = []
    for sample in samples:
        if not sample.get("available", False):
            states.append("unknown")
        elif float(sample.get("ego_speed", 0.0)) <= thresholds["static_speed_threshold"]:
            states.append("static")
        elif float(sample.get("ego_vz", 0.0)) > 1e-9:
            states.append("forward")
        elif float(sample.get("ego_vz", 0.0)) < -1e-9:
            states.append("backward")
        else:
            states.append("static")
    return states


def _speed_change_states(samples, acceleration_threshold):
    states = []
    for sample in samples:
        delta = sample.get("ego_speed_delta")
        if not sample.get("available", False) or delta is None:
            states.append("unknown")
        elif float(delta) > acceleration_threshold:
            states.append("accelerating")
        elif float(delta) < -acceleration_threshold:
            states.append("decelerating")
        else:
            states.append("steady")
    return states


def _speed_change_fit_error(samples, states):
    valid = [
        (float(sample["ego_speed_delta"]), state)
        for sample, state in zip(samples, states)
        if sample.get("ego_speed_delta") is not None and state != "unknown"
    ]
    if len(valid) <= 1:
        return 0.0
    values = [value for value, _ in valid]
    mean = sum(values) / len(values)
    total = sum((value - mean) ** 2 for value in values)
    if total <= 1e-12:
        return 0.0
    residual = 0.0
    for state in {state for _, state in valid}:
        grouped = [value for value, current in valid if current == state]
        group_mean = sum(grouped) / len(grouped)
        residual += sum((value - group_mean) ** 2 for value in grouped)
    return float(max(0.0, min(1.0, residual / total)))


def _score_ego_threshold_candidate(samples, thresholds, config, candidate_id):
    actions = [_ego_action(sample, thresholds) for sample in samples]
    segments = _contiguous_action_segments(samples, actions)
    frame_count = max(1, len(samples))
    transitions = max(0, len(segments) - 1)
    short_segments = [
        segment
        for segment in segments
        if int(segment["duration_frames"]) < config["min_short_segment_frames"]
    ]
    direction_family = {
        "left": "left",
        "turning_left": "left",
        "right": "right",
        "turning_right": "right",
    }
    reversals = 0
    directional_segments = [
        segment for segment in segments if segment["action"] in direction_family
    ]
    for left, right in zip(directional_segments, directional_segments[1:]):
        if (
            direction_family[left["action"]] != direction_family[right["action"]]
            and int(left["duration_frames"]) + int(right["duration_frames"])
            <= config["rapid_reversal_window_frames"]
        ):
            reversals += 1
    longitudinal_states = _longitudinal_states(samples, thresholds)
    longitudinal_transitions, forward_backward_reversals = _transition_counts(
        longitudinal_states, ("forward", "backward")
    )
    acceleration_options = []
    for acceleration_threshold in config["candidate_acceleration_thresholds"]:
        speed_change_states = _speed_change_states(samples, acceleration_threshold)
        acceleration_transitions, acceleration_reversals = _transition_counts(
            speed_change_states, ("accelerating", "decelerating")
        )
        fit_error = _speed_change_fit_error(samples, speed_change_states)
        normalized_transition = acceleration_transitions / max(1, frame_count - 1)
        normalized_reversal = acceleration_reversals / max(1, frame_count - 1)
        acceleration_options.append((
            fit_error + 0.75 * normalized_transition + 1.5 * normalized_reversal,
            float(acceleration_threshold),
            speed_change_states,
            acceleration_transitions,
            acceleration_reversals,
            fit_error,
        ))
    acceleration_options.sort(key=lambda row: (row[0], row[1]))
    _, selected_acceleration_threshold, speed_change_states, acceleration_transitions, acceleration_reversals, acceleration_fit_error = acceleration_options[0]
    thresholds = {**thresholds, "acceleration_threshold": selected_acceleration_threshold}
    unique_actions = {
        action for action in actions if action not in {"unknown"}
    }
    components = {
        "signal_fit_error": _within_action_signal_fit_error(samples, actions),
        "state_transitions": float(transitions / max(1, frame_count - 1)),
        "short_segment_count": float(len(short_segments) / max(1, len(segments))),
        "short_segment_duration": float(
            sum(int(segment["duration_frames"]) for segment in short_segments)
            / frame_count
        ),
        "rapid_left_right_reversals": float(
            reversals / max(1, len(directional_segments) - 1)
        ),
        "longitudinal_state_transitions": float(
            longitudinal_transitions / max(1, frame_count - 1)
        ),
        "forward_backward_reversals": float(
            forward_backward_reversals / max(1, frame_count - 1)
        ),
        "acceleration_state_transitions": float(
            acceleration_transitions / max(1, frame_count - 1)
        ),
        "acceleration_deceleration_reversals": float(
            acceleration_reversals / max(1, frame_count - 1)
        ),
        "acceleration_signal_fit_error": float(acceleration_fit_error),
        "action_complexity": float(
            0.5 * len(unique_actions) / 7.0
            + 0.5 * len(segments) / frame_count
        ),
    }
    weights = config["score_weights"]
    weighted = {
        name: float(value * float(weights.get(name, 0.0)))
        for name, value in components.items()
    }
    score = float(sum(weighted.values()))
    counts = Counter(actions)
    return {
        "candidate_id": candidate_id,
        "thresholds": copy.deepcopy(thresholds),
        "score": score,
        "score_components": components,
        "weighted_score_components": weighted,
        "num_segments": len(segments),
        "num_transitions": transitions,
        "num_short_segments": len(short_segments),
        "short_segment_total_duration": sum(
            int(segment["duration_frames"]) for segment in short_segments
        ),
        "num_rapid_left_right_reversals": reversals,
        "num_longitudinal_state_transitions": longitudinal_transitions,
        "num_forward_backward_reversals": forward_backward_reversals,
        "num_acceleration_state_transitions": acceleration_transitions,
        "num_acceleration_deceleration_reversals": acceleration_reversals,
        "longitudinal_states": longitudinal_states,
        "speed_change_states": speed_change_states,
        "action_counts": dict(sorted(counts.items())),
        "actions": actions,
        "segments": segments,
    }


def _threshold_axis_step(values):
    ordered = sorted(set(float(value) for value in values))
    gaps = [right - left for left, right in zip(ordered, ordered[1:]) if right > left]
    return min(gaps) if gaps else max(1e-4, ordered[0] * 0.5)


def _coarse_to_fine_ego_candidate_scores(samples, config):
    """Generate deterministic, label-aware threshold candidates over several rounds."""
    axes = {
        "static_speed_threshold": config["candidate_static_speed_thresholds"],
        "lateral_threshold": config["candidate_lateral_thresholds"],
        "yaw_threshold": config["candidate_yaw_thresholds"],
    }
    base_steps = {key: _threshold_axis_step(values) for key, values in axes.items()}
    scores = []
    seen_thresholds = set()
    seen_label_sequences = set()

    def add_round(threshold_rows, round_index, parents):
        round_scores = []
        for thresholds in sorted(
            threshold_rows,
            key=lambda row: tuple(float(row[key]) for key in sorted(axes)),
        )[: config["max_candidates"]]:
            signature = tuple(round(float(thresholds[key]), 12) for key in sorted(axes))
            if signature in seen_thresholds:
                continue
            seen_thresholds.add(signature)
            candidate = _score_ego_threshold_candidate(
                samples,
                thresholds,
                config,
                candidate_id=f"ego_threshold_r{round_index}_{len(scores) + len(round_scores):04d}",
            )
            label_signature = (
                tuple(candidate["actions"]),
                tuple(candidate["speed_change_states"]),
            )
            if label_signature in seen_label_sequences:
                continue
            seen_label_sequences.add(label_signature)
            candidate["search_round"] = round_index
            candidate["parent_candidate_ids"] = list(parents)
            candidate["refinement_steps"] = {
                key: float(base_steps[key] * config["threshold_refinement_factor"] ** round_index)
                for key in axes
            }
            round_scores.append(candidate)
        scores.extend(round_scores)
        return round_scores

    current = add_round(_ego_threshold_candidates(config), 0, [])
    for round_index in range(1, config["threshold_search_rounds"]):
        parents = sorted(
            current or scores,
            key=lambda row: (float(row["score"]), row["candidate_id"]),
        )[: config["threshold_refinement_top_k"]]
        proposals = []
        for parent in parents:
            center = parent["thresholds"]
            axis_values = []
            for key in sorted(axes):
                step = base_steps[key] * config["threshold_refinement_factor"] ** round_index
                axis_values.append((key, [max(1e-9, float(center[key]) + delta * step) for delta in (-1, 0, 1)]))
            for static_value in axis_values[1][1]:
                for lateral_value in axis_values[0][1]:
                    for yaw_value in axis_values[2][1]:
                        proposals.append({
                            "static_speed_threshold": static_value,
                            "lateral_threshold": lateral_value,
                            "yaw_threshold": yaw_value,
                        })
        current = add_round(proposals, round_index, [row["candidate_id"] for row in parents])
        if not current:
            break
    return scores


def _select_ego_thresholds(samples, config):
    scores = _coarse_to_fine_ego_candidate_scores(samples, config)
    if not scores:
        raise RuntimeError("Step 7A generated no threshold candidates")
    selected = min(scores, key=lambda row: (float(row["score"]), row["candidate_id"]))
    return selected, scores


def _ego_symbol_prior_video(ego_video, config=None):
    """Segment ego-vz with constrained change-point dynamic programming."""
    from src.exp_july.perception.ego_vz_change_point_segmentation import segment_ego_vz

    resolved_config = _ego_symbol_config(config)
    samples = _ego_continuous_samples(ego_video)
    segmentation = segment_ego_vz(samples, resolved_config)
    segments = list(segmentation["segments"])
    action_by_sample = {}
    segment_by_sample = {}
    for segment in segments:
        for sample_index in segment.get("sample_indices", []):
            action_by_sample[int(sample_index)] = str(segment["state"])
            segment_by_sample[int(sample_index)] = segment
    static_bands = [
        float(segment["adaptive_static_band"])
        for segment in segments
        if segment.get("adaptive_static_band") is not None
    ]
    static_band = _median(static_bands) if static_bands else float(
        resolved_config.get("static_band_floor", 0.05)
    )
    selected_thresholds = {
        "static_speed_threshold": float(static_band),
        # Compatibility-only fields for the existing 7E/7F interface. They do
        # not participate in Step 7A longitudinal state assignment.
        "lateral_threshold": float(resolved_config["candidate_lateral_thresholds"][0]),
        "yaw_threshold": float(resolved_config["candidate_yaw_thresholds"][0]),
        "acceleration_threshold": float(resolved_config["acceleration_threshold"]),
    }
    acceleration_threshold = selected_thresholds["acceleration_threshold"]
    output_frames = []
    for sample_index, sample in enumerate(samples):
        action = action_by_sample.get(sample_index, "unknown")
        segment = segment_by_sample.get(sample_index, {})
        cues = {name: 0.0 for name in _EGO_CUE_NAMES}
        cues["ego_motion_uncertain"] = (
            1.0 if action == "unknown"
            else float(max(0.0, min(1.0, 1.0 - float(segment.get("confidence", 0.0)))))
        )
        if action == "static":
            cues["ego_static"] = 1.0
        elif action == "forward":
            cues["ego_driving_forward"] = 1.0
            cues["ego_straight"] = 1.0
        elif action == "backward":
            cues["ego_driving_backward"] = 1.0
            cues["ego_straight"] = 1.0
        if action not in {"unknown", "static"}:
            delta = sample.get("ego_speed_delta")
            if delta is not None and float(delta) > acceleration_threshold:
                cues["ego_accelerating"] = 1.0
            elif delta is not None and float(delta) < -acceleration_threshold:
                cues["ego_decelerating"] = 1.0
        output_frames.append({
            "frame_index": int(sample["frame_index"]),
            "action": action,
            "confidence": float(segment.get("confidence", 0.0)),
            "segment_id": segment.get("segment_id"),
            "observable_cues": cues,
            "signal_evidence": {
                key: sample.get(key)
                for key in ("ego_vx", "ego_vz", "ego_yaw_rate", "ego_speed", "ego_speed_delta")
            },
        })
    aggregate = {
        name: float(sum(frame["observable_cues"][name] for frame in output_frames) / max(1, len(output_frames)))
        for name in _EGO_CUE_NAMES
    }
    run_audits = []
    seen_run_audits = set()
    for segment in segments:
        run_audit = segment.get("run_audit", {})
        identity = id(run_audit)
        if not run_audit or identity in seen_run_audits:
            continue
        seen_run_audits.add(identity)
        members = [row for row in segments if row.get("run_audit") is run_audit]
        run_audits.append({
            **copy.deepcopy(run_audit),
            "start_frame": min(int(row["start_frame"]) for row in members),
            "end_frame": max(int(row["end_frame"]) for row in members),
        })
    objective = float(sum(float(row.get("objective", 0.0)) for row in run_audits))
    audit_explanation = (
        f"Constrained change-point dynamic programming segmented ego_vz into {len(segments)} "
        f"segments using robust prefix-statistic fitting, an additional-segment penalty, "
        f"minimum length {segmentation['configuration']['min_segment_length']}, and pruned "
        "candidate boundaries. States use an adaptive video-local static band; direct "
        "forward/backward transitions are forbidden, adjacent identical states are merged, "
        "and high residual variance is represented by lower confidence."
    )
    speed_values = [float(sample["ego_speed"]) for sample in samples if sample.get("ego_speed") is not None]
    speed_median = _median(speed_values)
    speed_mad = _median([abs(value - speed_median) for value in speed_values])
    public_segments = [
        {key: copy.deepcopy(value) for key, value in segment.items() if key not in {"sample_indices", "run_audit"}}
        for segment in segments
    ]
    return {
        "version": _EGO_SYMBOL_PRIOR_VERSION,
        "video_id": str(ego_video.get("video_id", "")),
        "status": "completed",
        "source_step": "07_ego_motion",
        "role": "provisional_ego_symbol_hypothesis",
        "label_status": "provisional",
        "downstream_usable_as_final": False,
        "threshold_status": "provisional_frozen_for_evidence_validation",
        "num_frames": len(output_frames),
        "continuous_signals": copy.deepcopy(samples),
        "frames": output_frames,
        "aggregate_cues": aggregate,
        "selected_threshold": copy.deepcopy(selected_thresholds),
        "selected_thresholds": copy.deepcopy(selected_thresholds),
        "selected_candidate_id": "ego_vz_constrained_dp",
        "selected_candidate_score": objective,
        "candidate_scores": [{
            "candidate_id": "ego_vz_constrained_dp",
            "thresholds": copy.deepcopy(selected_thresholds),
            "score": objective,
            "search_round": 0,
            "num_segments": len(public_segments),
            "num_transitions": max(0, len(public_segments) - 1),
            "method": segmentation["method"],
        }],
        "final_action_segments": public_segments,
        "change_point_segmentation": {
            "method": segmentation["method"],
            "global_noise_scale": segmentation["global_noise_scale"],
            "boundaries": [
                {"segment_id": segment["segment_id"], "start_frame": segment["start_frame"], "end_frame": segment["end_frame"]}
                for segment in public_segments
            ],
            "segments": copy.deepcopy(public_segments),
            "runs": run_audits,
            "provenance": {
                "source_step": "07_ego_motion",
                "source_signal": "ego_vz",
                "implementation_version": 1,
                "deterministic": True,
            },
        },
        "audit_explanation": audit_explanation,
        "configuration": copy.deepcopy(segmentation["configuration"]),
        "calibration": {
            "static_speed_band": float(static_band),
            "lateral_turn_band": selected_thresholds["lateral_threshold"],
            "yaw_turn_band": selected_thresholds["yaw_threshold"],
            "acceleration_band": acceleration_threshold,
            "speed_median": float(speed_median),
            "speed_mad": float(speed_mad),
            "ego_vz_noise_scale": float(segmentation["global_noise_scale"]),
            "selection_method": "constrained_change_point_dynamic_programming",
        },
    }


def step7a_ego_symbol_prior(ego_state, config=None):
    """Build and cache a per-video automatically calibrated ego-symbol prior."""
    ego_motion = list(ego_state.get("ego_motion", []))
    resolved_config = _ego_symbol_config(
        config if config is not None else ego_state.get("ego_symbol_prior_config")
    )
    config_fingerprint = hashlib.sha256(
        json.dumps(resolved_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_root = get_pipeline_output_root() / "07a_ego_symbol_prior"
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    cached_videos = 0
    for ego_video in tqdm(
        ego_motion, desc="[step 7a] ego_symbol_prior", unit="video"
    ):
        video_id = str(ego_video.get("video_id", ""))
        source_fingerprint = hashlib.sha256(
            json.dumps(
                ego_video.get("frames", []),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()
        path = output_root / video_id / "ego_symbol_prior.json"
        cached = None
        if path.exists():
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if (
                    int(candidate.get("version", 0))
                    == _EGO_SYMBOL_PRIOR_VERSION
                    and str(candidate.get("source_fingerprint", ""))
                    == source_fingerprint
                    and str(candidate.get("config_fingerprint", ""))
                    == config_fingerprint
                    and set(candidate.get("aggregate_cues", {}))
                    == set(_EGO_CUE_NAMES)
                    and candidate.get("threshold_status")
                    == "provisional_frozen_for_evidence_validation"
                    and candidate.get("label_status") == "provisional"
                ):
                    cached = candidate
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                cached = None
        if cached is not None:
            cached_videos += 1
            results.append(cached)
            continue
        result = _ego_symbol_prior_video(ego_video, resolved_config)
        result["source_fingerprint"] = source_fingerprint
        result["config_fingerprint"] = config_fingerprint
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
    manifest = {
        "version": _EGO_SYMBOL_PRIOR_VERSION,
        "num_videos": len(results),
        "num_frames": sum(int(row.get("num_frames", 0)) for row in results),
        "cached_videos": cached_videos,
        "cue_names": list(_EGO_CUE_NAMES),
        "role": "provisional_ego_symbol_hypothesis",
        "label_status": "provisional",
        "downstream_usable_as_final": False,
        "threshold_selection": "adaptive_static_band_from_constrained_change_points",
        "config_fingerprint": config_fingerprint,
        "selected_thresholds_by_video": {
            str(row.get("video_id", "")): copy.deepcopy(
                row.get("selected_thresholds", {})
            )
            for row in results
        },
    }
    (output_root / "ego_symbol_prior_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"[step 7a] ego_symbol_prior videos={manifest['num_videos']} "
        f"frames={manifest['num_frames']} cached={cached_videos} "
        f"segmentation=constrained_dp"
    )
    return {
        **ego_state,
        "ego_symbol_prior": results,
        "ego_symbol_prior_manifest": manifest,
        "ego_symbol_prior_output_root": output_root,
        "ego_symbol_prior_config": resolved_config,
    }


def step7b_background_motion_evidence(position_state, ego_symbol_state, config=None):
    """Extract independent background patch motion for provisional 7A segments."""
    from src.exp_july.perception.background_motion_evidence import (
        VERSION,
        extract_video_evidence,
        resolved_config,
    )

    cfg = resolved_config(config)
    output_root = get_pipeline_output_root() / "07b_background_motion_evidence"
    output_root.mkdir(parents=True, exist_ok=True)
    positions_by_video = {
        str(row.get("video_id", "")): row
        for row in position_state.get("positions_3d", [])
    }
    provisional_videos = list(ego_symbol_state.get("ego_symbol_prior", []))
    results = []
    cached_videos = 0
    for provisional in tqdm(
        provisional_videos,
        desc="[step 7b] background_motion_evidence",
        unit="video",
    ):
        video_id = str(provisional.get("video_id", ""))
        position_video = positions_by_video.get(video_id, {"video_id": video_id, "frames": []})
        frame_sources = []
        for offset, frame in enumerate(position_video.get("frames", [])):
            image_path = str(frame.get("image_path", ""))
            image_metadata = None
            if image_path and Path(image_path).exists():
                stat = Path(image_path).stat()
                image_metadata = [int(stat.st_size), int(stat.st_mtime_ns)]
            frame_sources.append({
                "frame_index": int(frame.get("frame_index", offset)),
                "image_path": image_path,
                "image_metadata": image_metadata,
                "boxes": frame.get("boxes", frame.get("bboxes", [])),
                "objects": [
                    {"bbox": obj.get("bbox", obj.get("box"))}
                    for obj in frame.get("objects", [])
                ],
            })
        source_fingerprint = _step8_cache_fingerprint({
            "schema": "step7b-background-motion-evidence-v1",
            "video_id": video_id,
            "provisional_segments": provisional.get("final_action_segments", []),
            "frame_sources": frame_sources,
            "config": cfg,
        })
        path = output_root / video_id / "background_motion_evidence.json"
        cached = None
        if path.exists():
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if (
                    int(candidate.get("version", 0)) == VERSION
                    and str(candidate.get("source_fingerprint", "")) == source_fingerprint
                    and candidate.get("input_label_status") == "provisional"
                ):
                    cached = candidate
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                cached = None
        if cached is not None:
            cached_videos += 1
            results.append(cached)
            continue
        result = extract_video_evidence(position_video, provisional, cfg)
        result["source_fingerprint"] = source_fingerprint
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
    manifest = {
        "version": VERSION,
        "stage": "7b_background_motion_evidence",
        "input_label_status": "provisional",
        "output_role": "independent_evidence_not_final_labels",
        "num_videos": len(results),
        "num_segments": sum(int(row.get("num_segments", 0)) for row in results),
        "num_patch_vectors": sum(int(row.get("num_patch_vectors", 0)) for row in results),
        "cached_videos": cached_videos,
        "execution_profile": cfg["execution_profile"],
        "configuration": cfg,
    }
    manifest_path = output_root / "background_motion_evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[step 7b] background_motion_evidence videos={manifest['num_videos']} "
        f"segments={manifest['num_segments']} patches={manifest['num_patch_vectors']} "
        f"cached={cached_videos} profile={cfg['execution_profile']} "
        f"stride={cfg['frame_stride']}",
        flush=True,
    )
    return {
        **ego_symbol_state,
        "background_motion_evidence": results,
        "background_motion_evidence_manifest": manifest,
        "background_motion_evidence_manifest_path": str(manifest_path),
        "background_motion_evidence_output_root": output_root,
    }


def step7c_video_local_evidence_calibration(evidence_state):
    """Normalize Step 7B segment evidence within each video without labels."""
    from src.exp_july.perception.video_local_evidence_calibration import VERSION, calibrate_video

    output_root = get_pipeline_output_root() / "07c_video_local_evidence_calibration"
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    cached_videos = 0
    for raw_video in tqdm(
        evidence_state.get("background_motion_evidence", []),
        desc="[step 7c] video_local_evidence_calibration",
        unit="video",
    ):
        video_id = str(raw_video.get("video_id", ""))
        source_fingerprint = _step8_cache_fingerprint({
            "schema": "step7c-video-local-evidence-calibration-v1",
            "raw_background_evidence": raw_video,
        })
        path = output_root / video_id / "normalized_background_evidence.json"
        cached = None
        if path.exists():
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if (
                    int(candidate.get("version", 0)) == VERSION
                    and str(candidate.get("source_fingerprint", "")) == source_fingerprint
                    and candidate.get("calibration_scope") == "video_local"
                    and not bool(candidate.get("dataset_specific_absolute_thresholds_used", True))
                ):
                    cached = candidate
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                cached = None
        if cached is not None:
            cached_videos += 1
            results.append(cached)
            continue
        result = calibrate_video(raw_video)
        result["source_fingerprint"] = source_fingerprint
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
    manifest = {
        "version": VERSION,
        "stage": "7c_video_local_evidence_calibration",
        "input_label_status": "provisional",
        "output_role": "normalized_evidence_not_final_labels",
        "calibration_scope": "video_local",
        "dataset_specific_absolute_thresholds_used": False,
        "num_videos": len(results),
        "num_segments": sum(int(row.get("num_segments", 0)) for row in results),
        "num_patch_vectors": sum(int(row.get("num_patch_vectors", 0)) for row in results),
        "cached_videos": cached_videos,
    }
    manifest_path = output_root / "video_local_evidence_calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[step 7c] video_local_evidence_calibration videos={manifest['num_videos']} "
        f"segments={manifest['num_segments']} patches={manifest['num_patch_vectors']} "
        f"cached={cached_videos} dataset_thresholds=False",
        flush=True,
    )
    return {
        **evidence_state,
        "video_local_calibrated_evidence": results,
        "video_local_evidence_calibration_manifest": manifest,
        "video_local_evidence_calibration_manifest_path": str(manifest_path),
        "video_local_evidence_calibration_output_root": output_root,
    }


def step7d_global_symbolic_rule_evaluation(calibrated_state):
    """Evaluate shared symbolic ego hypotheses without modifying labels."""
    from src.exp_july.perception.global_ego_symbolic_rules import (
        RULE_POLICY_ID,
        VERSION,
        evaluate_video,
    )

    output_root = get_pipeline_output_root() / "07d_global_symbolic_rule_evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    cached_videos = 0
    for calibrated_video in tqdm(
        calibrated_state.get("video_local_calibrated_evidence", []),
        desc="[step 7d] global_symbolic_rule_evaluation",
        unit="video",
    ):
        video_id = str(calibrated_video.get("video_id", ""))
        source_fingerprint = _step8_cache_fingerprint({
            "schema": "step7d-global-symbolic-rule-evaluation-v1",
            "rule_policy_id": RULE_POLICY_ID,
            "normalized_evidence": calibrated_video,
        })
        path = output_root / video_id / "global_symbolic_rule_evaluation.json"
        cached = None
        if path.exists():
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if (
                    int(candidate.get("version", 0)) == VERSION
                    and candidate.get("rule_policy_id") == RULE_POLICY_ID
                    and str(candidate.get("source_fingerprint", "")) == source_fingerprint
                    and dict(candidate.get("provenance", {})).get("labels_modified") is False
                ):
                    cached = candidate
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                cached = None
        if cached is not None:
            cached_videos += 1
            results.append(cached)
            continue
        result = evaluate_video(calibrated_video)
        result["source_fingerprint"] = source_fingerprint
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
    manifest = {
        "version": VERSION,
        "stage": "7d_global_symbolic_rule_evaluation",
        "rule_policy_id": RULE_POLICY_ID,
        "rule_scope": "global_shared_across_all_videos",
        "input_label_status": "provisional",
        "labels_modified": False,
        "num_videos": len(results),
        "num_segments": sum(int(row.get("num_segments", 0)) for row in results),
        "num_fired_rules": sum(int(row.get("num_fired_rules", 0)) for row in results),
        "num_violated_rules": sum(int(row.get("num_violated_rules", 0)) for row in results),
        "num_conflicts": sum(int(row.get("num_conflicts", 0)) for row in results),
        "cached_videos": cached_videos,
    }
    manifest_path = output_root / "global_symbolic_rule_evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[step 7d] global_symbolic_rule_evaluation videos={manifest['num_videos']} "
        f"segments={manifest['num_segments']} fired={manifest['num_fired_rules']} "
        f"violated={manifest['num_violated_rules']} conflicts={manifest['num_conflicts']} "
        f"cached={cached_videos}",
        flush=True,
    )
    return {
        **calibrated_state,
        "global_ego_symbolic_rule_evaluations": results,
        "global_ego_symbolic_rule_evaluation_manifest": manifest,
        "global_ego_symbolic_rule_evaluation_manifest_path": str(manifest_path),
        "global_ego_symbolic_rule_evaluation_output_root": output_root,
    }



def _shortlist_step7e_candidates(candidates, selected_candidate_id, limit):
    """Keep a deterministic, round-diverse shortlist for expensive evidence evaluation."""
    ordered = sorted(
        candidates,
        key=lambda row: (
            float(row.get("score", float("inf"))),
            int(row.get("num_forward_backward_reversals", 0)),
            int(row.get("num_acceleration_deceleration_reversals", 0)),
            str(row.get("candidate_id", "")),
        ),
    )
    by_id = {str(row.get("candidate_id", "")): row for row in ordered}
    selected = []
    seen = set()

    def add(row):
        if row is None:
            return
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id and candidate_id not in seen:
            seen.add(candidate_id)
            selected.append(row)

    add(by_id.get(str(selected_candidate_id)))
    for round_index in sorted({int(row.get("search_round", 0)) for row in ordered}):
        add(next((row for row in ordered if int(row.get("search_round", 0)) == round_index), None))
    for row in ordered:
        if len(selected) >= max(1, int(limit)):
            break
        add(row)
    return selected[: max(1, int(limit))]


def step7e_threshold_label_refinement(rule_state):
    """Search bounded thresholds until labels and thresholds stabilize."""
    from src.exp_july.perception.ego_threshold_label_refinement import VERSION, refine_video

    output_root = get_pipeline_output_root() / "07e_threshold_label_refinement"
    output_root.mkdir(parents=True, exist_ok=True)
    provisional_by_video = {str(row.get("video_id", "")): row for row in rule_state.get("ego_symbol_prior", [])}
    raw_by_video = {str(row.get("video_id", "")): row for row in rule_state.get("background_motion_evidence", [])}
    results = []
    cached_videos = 0
    max_iterations = max(2, int(os.environ.get("CAUVID_STEP7E_MAX_ITERATIONS", "4")))
    for video_id, provisional in tqdm(
        sorted(provisional_by_video.items()), desc="[step 7e] threshold_label_refinement", unit="video"
    ):
        raw_video = raw_by_video.get(video_id, {"video_id": video_id, "segments": []})
        config = _ego_symbol_config(provisional.get("configuration", {}))
        samples = list(provisional.get("continuous_signals", []))
        generated_candidates = _coarse_to_fine_ego_candidate_scores(samples, config)
        candidates = _shortlist_step7e_candidates(
            generated_candidates,
            provisional.get("selected_candidate_id"),
            config["step7e_expensive_candidate_limit"],
        )
        source_fingerprint = _step8_cache_fingerprint({
            "schema": "step7e-threshold-label-refinement-v2",
            "provisional": provisional,
            "raw_evidence": raw_video,
            "generated_candidate_thresholds": [row["thresholds"] for row in generated_candidates],
            "evaluated_candidate_ids": [row["candidate_id"] for row in candidates],
            "expensive_candidate_limit": config["step7e_expensive_candidate_limit"],
            "max_iterations": max_iterations,
        })
        path = output_root / video_id / "threshold_label_refinement.json"
        cached = None
        if path.exists():
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if int(candidate.get("version", 0)) == VERSION and str(candidate.get("source_fingerprint", "")) == source_fingerprint:
                    cached = candidate
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                cached = None
        if cached is not None:
            cached_videos += 1; results.append(cached); continue
        result = refine_video(video_id, candidates, raw_video, provisional, max_iterations=max_iterations)
        result["generated_candidate_count"] = len(generated_candidates)
        result["evaluated_candidate_count"] = len(candidates)
        result["expensive_candidate_limit"] = config["step7e_expensive_candidate_limit"]
        result["source_fingerprint"] = source_fingerprint
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, indent=2), encoding="utf-8"); results.append(result)
    manifest = {
        "version": VERSION, "stage": "7e_threshold_label_refinement", "deterministic": True,
        "num_videos": len(results), "cached_videos": cached_videos,
        "num_generated_candidates": sum(int(row.get("generated_candidate_count", len(row.get("candidate_rankings", [])))) for row in results),
        "num_candidates": sum(len(row.get("candidate_rankings", [])) for row in results),
        "expensive_candidate_limit": max((int(row.get("expensive_candidate_limit", 0)) for row in results), default=0),
        "num_corrections": sum(len(row.get("corrections", [])) for row in results),
        "num_uncertain_segments": sum(len(row.get("uncertain_segments", [])) for row in results),
        "num_stabilized": sum(bool(row.get("stabilized")) for row in results),
        "max_iterations": max_iterations,
    }
    manifest_path = output_root / "threshold_label_refinement_manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[step 7e] threshold_label_refinement videos={manifest['num_videos']} "
        f"generated={manifest['num_generated_candidates']} evaluated={manifest['num_candidates']} "
        f"limit={manifest['expensive_candidate_limit']} corrections={manifest['num_corrections']} "
        f"uncertain={manifest['num_uncertain_segments']} stabilized={manifest['num_stabilized']} "
        f"cached={cached_videos}",
        flush=True,
    )
    return {**rule_state, "ego_threshold_label_refinement": results, "ego_threshold_label_refinement_manifest": manifest, "ego_threshold_label_refinement_manifest_path": str(manifest_path), "ego_threshold_label_refinement_output_root": output_root}


def step7f_ego_symbol_finalization(position_state, refinement_state):
    """Publish validated final ego symbols and offline HTML/MP4 audit artifacts."""
    from src.exp_july.perception.ego_symbol_finalization import VERSION, build_html, finalize_video, render_mp4s

    output_root = get_pipeline_output_root() / "07f_ego_symbol_finalization"; output_root.mkdir(parents=True, exist_ok=True)
    provisional = list(refinement_state.get("ego_symbol_prior", [])); provisional_by_video = {str(row.get("video_id", "")): row for row in provisional}
    results = []; cached_videos = 0; recomputed_video_ids = set()
    for refinement in tqdm(refinement_state.get("ego_threshold_label_refinement", []), desc="[step 7f] ego_symbol_finalization", unit="video"):
        video_id = str(refinement.get("video_id", "")); prior = provisional_by_video.get(video_id, {})
        source_fingerprint = _step8_cache_fingerprint({"schema": "step7f-ego-symbol-finalization-v1", "refinement": refinement, "provisional": prior})
        path = output_root / video_id / "final_ego_symbols.json"; cached = None
        if path.exists():
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if int(candidate.get("version", 0)) == VERSION and str(candidate.get("source_fingerprint", "")) == source_fingerprint and candidate.get("label_status") == "final": cached = candidate
            except (OSError, TypeError, ValueError, json.JSONDecodeError): cached = None
        if cached is not None: cached_videos += 1; results.append(cached); continue
        recomputed_video_ids.add(video_id)
        result = finalize_video(refinement, prior); result["source_fingerprint"] = source_fingerprint
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, indent=2), encoding="utf-8"); results.append(result)
    audit_root = output_root / "audit"; html_path = build_html(results, audit_root)
    position_by_video = {str(row.get("video_id", "")): row for row in position_state.get("positions_3d", [])}
    video_audit_root = audit_root / "videos"
    for video_id in recomputed_video_ids:
        stale_path = video_audit_root / f"{video_id}_ego_symbol_audit.mp4"
        if stale_path.exists():
            stale_path.unlink()
    mp4_manifest = render_mp4s(results, position_by_video, video_audit_root, fps=max(.1, float(os.environ.get("CAUVID_STEP7F_VIS_FPS", "10"))), limit=5)
    manifest = {
        "version": VERSION, "stage": "7f_ego_symbol_finalization", "label_status": "final", "downstream_usable_as_final": True,
        "num_videos": len(results), "num_frames": sum(int(row.get("num_frames", 0)) for row in results),
        "num_validated_segments": sum(sum(seg.get("validation_status") == "validated" for seg in row.get("final_action_segments", [])) for row in results),
        "num_uncertain_segments": sum(sum(seg.get("validation_status") != "validated" for seg in row.get("final_action_segments", [])) for row in results),
        "num_corrections": sum(len(row.get("corrections", [])) for row in results), "cached_videos": cached_videos,
        "html_audit_path": html_path, "mp4_audit": mp4_manifest,
    }
    manifest_path = output_root / "ego_symbol_finalization_manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[step 7f] ego_symbol_finalization videos={manifest['num_videos']} validated={manifest['num_validated_segments']} uncertain={manifest['num_uncertain_segments']} corrections={manifest['num_corrections']} cached={cached_videos} html={html_path} mp4={len(mp4_manifest['rendered'])}", flush=True)
    return {
        **refinement_state,
        "provisional_ego_symbol_prior": provisional,
        "ego_symbol_prior": results,
        "final_ego_symbols": results,
        "ego_symbol_prior_manifest": manifest,
        "final_ego_symbol_manifest": manifest,
        "final_ego_symbol_manifest_path": str(manifest_path),
        "ego_symbol_prior_output_root": output_root,
        "final_ego_symbol_output_root": output_root,
        "ego_symbol_audit_html_path": html_path,
        "ego_symbol_audit_mp4_manifest": mp4_manifest,
    }


def step7b_tracklet_repair(
    position_state,
    ego_state,
    *,
    output_subdir="07b_driving_mini_tracklet_repair",
    step_label="7b",
    repair_cfg=None,
):
    videos = position_state["videos"]
    positions_3d = position_state.get("positions_3d", [])
    ego_motion = ego_state.get("ego_motion", [])
    if not videos or not positions_3d:
        print(f"[step {step_label}] no 3d positions, skip tracklet repair")
        return {
            "videos": videos,
            "tracklet_repair": [],
            "positions_3d": positions_3d,
            "positions_3d_output_root": position_state.get("positions_3d_output_root"),
            "ego_motion": ego_motion,
            "ego_motion_output_root": ego_state.get("ego_motion_output_root"),
            "ego_symbol_prior": copy.deepcopy(ego_state.get("ego_symbol_prior", [])),
            "ego_symbol_prior_manifest": copy.deepcopy(ego_state.get("ego_symbol_prior_manifest", {})),
            "ego_symbol_prior_output_root": ego_state.get("ego_symbol_prior_output_root"),
            "background_motion_evidence": copy.deepcopy(ego_state.get("background_motion_evidence", [])),
            "background_motion_evidence_manifest": copy.deepcopy(ego_state.get("background_motion_evidence_manifest", {})),
            "background_motion_evidence_output_root": ego_state.get("background_motion_evidence_output_root"),
        "video_local_calibrated_evidence": copy.deepcopy(ego_state.get("video_local_calibrated_evidence", [])),
        "video_local_evidence_calibration_manifest": copy.deepcopy(ego_state.get("video_local_evidence_calibration_manifest", {})),
        "video_local_evidence_calibration_output_root": ego_state.get("video_local_evidence_calibration_output_root"),
        "global_ego_symbolic_rule_evaluations": copy.deepcopy(ego_state.get("global_ego_symbolic_rule_evaluations", [])),
        "global_ego_symbolic_rule_evaluation_manifest": copy.deepcopy(ego_state.get("global_ego_symbolic_rule_evaluation_manifest", {})),
        "global_ego_symbolic_rule_evaluation_output_root": ego_state.get("global_ego_symbolic_rule_evaluation_output_root"),
        "ego_threshold_label_refinement": copy.deepcopy(ego_state.get("ego_threshold_label_refinement", [])),
        "ego_threshold_label_refinement_manifest": copy.deepcopy(ego_state.get("ego_threshold_label_refinement_manifest", {})),
        "final_ego_symbols": copy.deepcopy(ego_state.get("final_ego_symbols", [])),
        "final_ego_symbol_manifest": copy.deepcopy(ego_state.get("final_ego_symbol_manifest", {})),
        "ego_symbol_audit_html_path": ego_state.get("ego_symbol_audit_html_path"),
        "ego_symbol_audit_mp4_manifest": copy.deepcopy(ego_state.get("ego_symbol_audit_mp4_manifest", {})),
        }

    output_root = get_pipeline_output_root() / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    ego_by_video = {
        str(row.get("video_id", "")): row
        for row in ego_motion
        if str(row.get("video_id", ""))
    }
    repaired_positions_3d = []
    cached_videos = 0
    resolved_repair_cfg = {
        **_TRACKLET_REPAIR_DEFAULT_CFG,
        **dict(repair_cfg or {}),
    }
    progress = tqdm(positions_3d, desc=f"[step {step_label}] tracklet_repair", unit="video")
    for video_result in progress:
        video_id = str(video_result.get("video_id", ""))
        progress.set_postfix_str(video_id, refresh=False)
        ego_video = ego_by_video.get(video_id)
        source_fingerprint = _step8_cache_fingerprint(
            {
                "schema": "step8-tracklet-repair-v2",
                "video": video_result,
                "ego_motion": ego_video,
                "repair_config": resolved_repair_cfg,
            }
        )
        out_dir = output_root / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_path = out_dir / "tracklet_repair.json"
        repaired = None
        cached, cache_path_changes = relocate_json_cache_file(
            cache_path,
            dataset_root=position_state.get("dataset_root"),
            pipeline_root=get_pipeline_output_root(),
        )
        if (
            isinstance(cached, dict)
            and int(cached.get("whole_step_cache_version", 0)) == 1
            and str(cached.get("source_fingerprint", ""))
            == source_fingerprint
        ):
            repaired = cached
            cached_videos += 1
            if cache_path_changes:
                print(
                    f"[step {step_label}] relocated cached tracklet paths "
                    f"video_id={video_id} paths={len(cache_path_changes)}"
                )
        if repaired is None:
            repaired = _repair_video_tracklets(video_result, ego_video, repair_cfg)
            repaired["whole_step_cache_version"] = 1
            repaired["source_fingerprint"] = source_fingerprint
            cache_path.write_text(
                json.dumps(repaired, indent=2), encoding="utf-8"
            )
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
        f"[step {step_label}] done videos={len(repaired_positions_3d)} "
        f"repaired_gaps={manifest['num_repaired_gaps']} "
        f"interpolated_objects={manifest['num_interpolated_objects']} "
        f"split_events={manifest['num_split_events']} "
        f"new_track_ids={manifest['num_new_track_ids']} "
        f"cached={cached_videos}/{len(repaired_positions_3d)} "
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
        "ego_symbol_prior": copy.deepcopy(ego_state.get("ego_symbol_prior", [])),
        "ego_symbol_prior_manifest": copy.deepcopy(ego_state.get("ego_symbol_prior_manifest", {})),
        "ego_symbol_prior_output_root": ego_state.get("ego_symbol_prior_output_root"),
        "background_motion_evidence": copy.deepcopy(ego_state.get("background_motion_evidence", [])),
        "background_motion_evidence_manifest": copy.deepcopy(ego_state.get("background_motion_evidence_manifest", {})),
        "background_motion_evidence_output_root": ego_state.get("background_motion_evidence_output_root"),
        "video_local_calibrated_evidence": copy.deepcopy(ego_state.get("video_local_calibrated_evidence", [])),
        "video_local_evidence_calibration_manifest": copy.deepcopy(ego_state.get("video_local_evidence_calibration_manifest", {})),
        "video_local_evidence_calibration_output_root": ego_state.get("video_local_evidence_calibration_output_root"),
        "global_ego_symbolic_rule_evaluations": copy.deepcopy(ego_state.get("global_ego_symbolic_rule_evaluations", [])),
        "global_ego_symbolic_rule_evaluation_manifest": copy.deepcopy(ego_state.get("global_ego_symbolic_rule_evaluation_manifest", {})),
        "global_ego_symbolic_rule_evaluation_output_root": ego_state.get("global_ego_symbolic_rule_evaluation_output_root"),
        "ego_threshold_label_refinement": copy.deepcopy(ego_state.get("ego_threshold_label_refinement", [])),
        "ego_threshold_label_refinement_manifest": copy.deepcopy(ego_state.get("ego_threshold_label_refinement_manifest", {})),
        "final_ego_symbols": copy.deepcopy(ego_state.get("final_ego_symbols", [])),
        "final_ego_symbol_manifest": copy.deepcopy(ego_state.get("final_ego_symbol_manifest", {})),
        "ego_symbol_audit_html_path": ego_state.get("ego_symbol_audit_html_path"),
        "ego_symbol_audit_mp4_manifest": copy.deepcopy(ego_state.get("ego_symbol_audit_mp4_manifest", {})),
    }


def step8_relative_object_motion(
    position_state,
    repaired_state,
    *,
    output_subdir="08_driving_mini_relative_object_motion",
    step_label="8",
):
    videos = position_state["videos"]
    repaired_positions = repaired_state.get("positions_3d", repaired_state.get("tracklet_repair", []))
    ego_motion = repaired_state.get("ego_motion", [])
    if not videos or not repaired_positions:
        print(f"[step {step_label}] no repaired object positions, skip relative object motion")
        return {"videos": videos, "relative_object_motion": []}

    output_root = get_pipeline_output_root() / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    ego_by_video = {
        str(row.get("video_id", "")): row
        for row in ego_motion
        if str(row.get("video_id", ""))
    }
    relative_motion = []
    cached_videos = 0
    progress = tqdm(repaired_positions, desc=f"[step {step_label}] relative_object_motion", unit="video")
    for video_result in progress:
        video_id = str(video_result.get("video_id", ""))
        progress.set_postfix_str(video_id, refresh=False)
        ego_result = ego_by_video.get(video_id, {})
        source_fingerprint = _step8_cache_fingerprint(
            {
                "schema": "step8a-relative-object-motion-v2",
                "relative_motion_version": _RELATIVE_OBJECT_MOTION_VERSION,
                "repaired_positions": video_result,
                "ego_motion": ego_result,
            }
        )
        out_dir = output_root / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_path = out_dir / "relative_object_motion.json"
        result = None
        cached, cache_path_changes = relocate_json_cache_file(
            cache_path,
            dataset_root=position_state.get("dataset_root"),
            pipeline_root=get_pipeline_output_root(),
        )
        if (
            isinstance(cached, dict)
            and int(cached.get("whole_step_cache_version", 0)) == 1
            and str(cached.get("source_fingerprint", ""))
            == source_fingerprint
        ):
            result = cached
            cached_videos += 1
            if cache_path_changes:
                print(
                    f"[step {step_label}] relocated cached relative-motion paths "
                    f"video_id={video_id} paths={len(cache_path_changes)}"
                )
        if result is None:
            result = _relative_motion_video(video_result, ego_result)
            result["whole_step_cache_version"] = 1
            result["source_fingerprint"] = source_fingerprint
            cache_path.write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
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
        f"[step {step_label}] done videos={len(relative_motion)} "
        f"objects={manifest['num_objects_total']} "
        f"observed={manifest['num_observed_objects_total']} "
        f"repaired={manifest['num_repaired_objects_total']} "
        f"with_rel_motion={manifest['num_objects_with_rel_motion']} "
        f"cached={cached_videos}/{len(relative_motion)}"
    )
    return {
        "videos": videos,
        "relative_object_motion": relative_motion,
        "relative_object_motion_output_root": output_root,
        "positions_3d": repaired_positions,
        "tracklet_repair": repaired_state.get("tracklet_repair", []),
        "ego_motion": ego_motion,
        "ego_symbol_prior": copy.deepcopy(repaired_state.get("ego_symbol_prior", [])),
        "ego_symbol_prior_manifest": copy.deepcopy(repaired_state.get("ego_symbol_prior_manifest", {})),
        "ego_symbol_prior_output_root": repaired_state.get("ego_symbol_prior_output_root"),
        "background_motion_evidence": copy.deepcopy(repaired_state.get("background_motion_evidence", [])),
        "background_motion_evidence_manifest": copy.deepcopy(repaired_state.get("background_motion_evidence_manifest", {})),
        "background_motion_evidence_output_root": repaired_state.get("background_motion_evidence_output_root"),
        "video_local_calibrated_evidence": copy.deepcopy(repaired_state.get("video_local_calibrated_evidence", [])),
        "video_local_evidence_calibration_manifest": copy.deepcopy(repaired_state.get("video_local_evidence_calibration_manifest", {})),
        "video_local_evidence_calibration_output_root": repaired_state.get("video_local_evidence_calibration_output_root"),
        "global_ego_symbolic_rule_evaluations": copy.deepcopy(repaired_state.get("global_ego_symbolic_rule_evaluations", [])),
        "global_ego_symbolic_rule_evaluation_manifest": copy.deepcopy(repaired_state.get("global_ego_symbolic_rule_evaluation_manifest", {})),
        "global_ego_symbolic_rule_evaluation_output_root": repaired_state.get("global_ego_symbolic_rule_evaluation_output_root"),
        "ego_threshold_label_refinement": copy.deepcopy(repaired_state.get("ego_threshold_label_refinement", [])),
        "ego_threshold_label_refinement_manifest": copy.deepcopy(repaired_state.get("ego_threshold_label_refinement_manifest", {})),
        "final_ego_symbols": copy.deepcopy(repaired_state.get("final_ego_symbols", [])),
        "final_ego_symbol_manifest": copy.deepcopy(repaired_state.get("final_ego_symbol_manifest", {})),
        "ego_symbol_audit_html_path": repaired_state.get("ego_symbol_audit_html_path"),
        "ego_symbol_audit_mp4_manifest": copy.deepcopy(repaired_state.get("ego_symbol_audit_mp4_manifest", {})),
    }


def step8a_symbol_grounded_refinement(
    relative_motion_state,
    llm_generate=None,
    *,
    output_subdir="08a_symbol_grounded_refinement",
    step_label="8a",
):
    """Generate and execute symbol-grounded semantic protection rules."""
    from src.exp_july.perception.symbol_grounded_refinement import run_symbol_grounded_refinement

    output_root = get_pipeline_output_root() / output_subdir
    result = run_symbol_grounded_refinement(
        relative_motion_state,
        output_root=output_root,
        llm_generate=llm_generate,
    )
    video_results = result.get("symbol_grounded_refinement", [])
    print(
        f"[step {step_label}] symbol_grounded_refinement "
        f"videos={len(video_results)} "
        f"accepted_rules={sum(len(row.get('semantic_protection_rules', [])) for row in video_results)} "
        f"rejected_rules={sum(len(row.get('rejected_rules', [])) for row in video_results)} "
        f"protected={len(result.get('protected_objects', []))} "
        f"uncovered={sum(len(row.get('uncovered_track_ids', [])) for row in video_results)}"
    )
    return result


def step8a_visual_symbol_grounded(
    relative_motion_state,
    *,
    output_subdir="08a_visual_symbol_grounded",
    step_label="8a visual",
):
    """Render one grounded-input and rule-result image per Step 8A track."""
    from src.exp_july.perception.symbol_grounded_refinement import (
        render_symbol_grounded_visualizations,
    )

    output_root = get_pipeline_output_root() / output_subdir
    result = render_symbol_grounded_visualizations(relative_motion_state, output_root)
    print(
        f"[step {step_label}] symbol_grounded "
        f"rendered={len(result.get('symbol_grounded_visualizations', []))} "
        f"skipped={len(result.get('symbol_grounded_visualization_skipped', []))} "
        f"output_root={output_root}"
    )
    return result


def step8_visual_relative_motion(
    relative_motion_state,
    fps=10.0,
    *,
    output_subdir="08visual_relative_motion_tracks",
    step_label="8visual",
    render_video_ids=None,
):
    videos = relative_motion_state.get("videos", [])
    source_relative_motion = list(
        relative_motion_state.get("relative_object_motion", [])
    )
    if not videos or not source_relative_motion:
        print(f"[step {step_label}] no relative object motion, skip visualization")
        return {
            **relative_motion_state,
            "relative_motion_visualizations": [],
            "relative_motion_visualization_output_root": None,
        }
    source_video_ids = [
        str(row.get("video_id", "")) for row in source_relative_motion
    ]
    selected_video_ids = (
        None
        if render_video_ids is None
        else {str(value) for value in render_video_ids if str(value)}
    )
    relative_motion = (
        source_relative_motion
        if selected_video_ids is None
        else [
            row
            for row in source_relative_motion
            if str(row.get("video_id", "")) in selected_video_ids
        ]
    )
    rendered_video_ids = [
        str(row.get("video_id", "")) for row in relative_motion
    ]
    skipped_unimportant_video_ids = [
        video_id
        for video_id in source_video_ids
        if video_id not in set(rendered_video_ids)
    ]

    output_root = get_pipeline_output_root() / output_subdir
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
    progress = tqdm(relative_motion, desc=f"[step {step_label}] relative_motion_tracks", unit="video")
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
        "render_scope": (
            "all_videos" if selected_video_ids is None else "important_videos"
        ),
        "num_source_videos": len(source_relative_motion),
        "num_videos": len(relative_motion),
        "rendered_video_ids": rendered_video_ids,
        "skipped_unimportant_video_ids": skipped_unimportant_video_ids,
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
        f"[step {step_label}] done videos={len(relative_motion)} "
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
        "relative_motion_visualization_video_selection": {
            "render_scope": manifest["render_scope"],
            "source_video_ids": source_video_ids,
            "rendered_video_ids": rendered_video_ids,
            "skipped_unimportant_video_ids": skipped_unimportant_video_ids,
        },
        "relative_motion_visualization_output_root": output_root,
    }


def step8b_uncertain_signal_evidence(
    relative_motion_state,
    *,
    output_subdir="08b_uncertain_signal_evidence",
    step_label="8b",
):
    """Abstract Step 8A samples into low-level uncertainty-aware evidence."""
    relative_motion = relative_motion_state.get("relative_object_motion", [])
    output_root = get_pipeline_output_root() / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    evidence_videos = []
    cached_videos = 0
    progress = tqdm(
        relative_motion,
        desc=f"[step {step_label}] uncertain_signal_evidence",
        unit="video",
    )
    for relative_video in progress:
        video_id = str(relative_video.get("video_id", ""))
        progress.set_postfix_str(video_id, refresh=False)
        output_dir = output_root / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = output_dir / "uncertain_signal_evidence.json"
        source_signal_fingerprint = _relative_signal_fingerprint(
            relative_video
        )
        cached, cache_path_changes = relocate_json_cache_file(
            cache_path,
            dataset_root=relative_motion_state.get("dataset_root"),
            pipeline_root=get_pipeline_output_root(),
        )
        cache_valid = (
            isinstance(cached, dict)
            and int(cached.get("version", 0))
            == _UNCERTAIN_SIGNAL_EVIDENCE_VERSION
            and str(cached.get("evidence_type", ""))
            == "uncertain_signal_evidence"
            and str(cached.get("video_id", "")) == video_id
            and str(cached.get("source_signal_fingerprint", ""))
            == source_signal_fingerprint
            and isinstance(cached.get("track_signal_evidence"), list)
            and isinstance(
                cached.get("quarantined_track_signal_evidence"), list
            )
            and all(
                set(row.get("observable_cues", {}))
                == {
                    "leftness",
                    "rightness",
                    "approach",
                    "recede",
                    "acceleration",
                    "deceleration",
                    "relative_static",
                    "relative_moving",
                    "relative_motion_uncertain",
                }
                and "descriptors" not in row
                for row in cached.get("track_signal_evidence", [])
            )
            and all(
                set(row.get("observable_cues", {}))
                == {
                    "leftness",
                    "rightness",
                    "approach",
                    "recede",
                    "acceleration",
                    "deceleration",
                    "relative_static",
                    "relative_moving",
                    "relative_motion_uncertain",
                }
                and "descriptors" not in row
                for row in cached.get(
                    "quarantined_track_signal_evidence", []
                )
            )
            and int(
                dict(cached.get("track_usefulness_filter", {})).get(
                    "policy_version", 0
                )
            )
            == _TRACK_USEFULNESS_POLICY_VERSION
            and not bool(cached.get("semantic_motion_classification", True))
            and not bool(cached.get("symbolic_reasoning", True))
        )
        if cache_valid:
            cached_videos += 1
            evidence = cached
            if cache_path_changes:
                print(
                    f"[step {step_label}] relocated cached signal paths "
                    f"video_id={video_id} paths={len(cache_path_changes)}"
                )
        else:
            evidence = _uncertain_signal_evidence_video(
                relative_video,
                source_signal_fingerprint=source_signal_fingerprint,
            )
            cache_path.write_text(
                json.dumps(evidence, indent=2),
                encoding="utf-8",
            )
        evidence_videos.append(evidence)

    total_source_tracks = sum(
        int(row.get("num_source_tracks", row.get("num_tracks", 0)))
        for row in evidence_videos
    )
    total_active_tracks = sum(
        int(row.get("num_tracks", 0)) for row in evidence_videos
    )
    total_quarantined_tracks = sum(
        int(row.get("num_quarantined_tracks", 0))
        for row in evidence_videos
    )
    manifest = {
        "version": _UNCERTAIN_SIGNAL_EVIDENCE_VERSION,
        "method": "uncertainty_aware_low_level_signal_abstraction",
        "evidence_type": "uncertain_signal_evidence",
        "abstraction_level": "low_level_observable_signal",
        "semantic_motion_classification": False,
        "symbolic_reasoning": False,
        "cue_names": [
            "leftness",
            "rightness",
            "approach",
            "recede",
            "acceleration",
            "deceleration",
        ],
        "num_videos": len(evidence_videos),
        "num_source_tracks": total_source_tracks,
        "num_tracks": total_active_tracks,
        "num_active_tracks": total_active_tracks,
        "num_quarantined_tracks": total_quarantined_tracks,
        "track_usefulness_filter": {
            "policy_version": _TRACK_USEFULNESS_POLICY_VERSION,
            "policy_kind": "conservative_initial_unanimous_evidence_gate",
            "mode": "quarantine_not_delete",
            "thresholds": dict(_TRACK_USEFULNESS_THRESHOLDS),
        },
        "num_observations": sum(
            int(row.get("num_observations", 0)) for row in evidence_videos
        ),
        "videos": [
            {
                "video_id": row.get("video_id", ""),
                "num_frames": int(row.get("num_frames", 0)),
                "num_source_tracks": int(
                    row.get("num_source_tracks", row.get("num_tracks", 0))
                ),
                "num_tracks": int(row.get("num_tracks", 0)),
                "num_active_tracks": int(
                    row.get("num_active_tracks", row.get("num_tracks", 0))
                ),
                "num_quarantined_tracks": int(
                    row.get("num_quarantined_tracks", 0)
                ),
                "num_observations": int(row.get("num_observations", 0)),
            }
            for row in evidence_videos
        ],
    }
    from src.exp_july.perception.uncertain_signal_evidence_visualization import (
        configured_step8b_visualization_limit,
        render_step8b_signal_evidence_videos,
    )

    try:
        visualization_fps = max(
            0.1,
            _safe_float(os.environ.get("CAUVID_STEP8B_VIS_FPS", "10")),
        )
        visualization_manifest = render_step8b_signal_evidence_videos(
            relative_motion,
            evidence_videos,
            output_root / "visualizations",
            fps=visualization_fps,
            max_tracks_per_video=configured_step8b_visualization_limit(),
        )
    except Exception as exc:
        visualization_manifest = {
            "version": 1,
            "format": "mp4",
            "max_tracks_per_video": None,
            "max_visualization_videos_total": configured_step8b_visualization_limit(),
            "num_selected_tracks": 0,
            "num_rendered_videos": 0,
            "num_skipped_videos": 0,
            "status": (
                f"visualization_failed:{type(exc).__name__}:"
                f"{str(exc)[:240]}"
            ),
        }
        print(
            f"[step {step_label}][visualization] "
            f"{visualization_manifest['status']}",
            flush=True,
        )
    manifest["visualization"] = visualization_manifest
    manifest_path = output_root / "uncertain_signal_evidence_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        f"[step {step_label}] uncertain_signal_evidence "
        f"videos={manifest['num_videos']} "
        f"cached={cached_videos} "
        f"source_tracks={manifest['num_source_tracks']} "
        f"active_tracks={manifest['num_active_tracks']} "
        f"quarantined_tracks={manifest['num_quarantined_tracks']} "
        f"observations={manifest['num_observations']} "
        "semantic_classification=False symbolic_reasoning=False"
    )
    result = {
        **relative_motion_state,
        "uncertain_signal_evidence": evidence_videos,
        "uncertain_signal_evidence_manifest": manifest,
        "uncertain_signal_evidence_manifest_path": manifest_path,
        "uncertain_signal_evidence_output_root": output_root,
        "uncertain_signal_evidence_visualizations": list(
            visualization_manifest.get("rendered", [])
        ),
        "uncertain_signal_evidence_visualization_manifest": (
            visualization_manifest
        ),
        "uncertain_signal_evidence_visualization_manifest_path": (
            visualization_manifest.get("manifest_path")
        ),
        "step8b_evidence_type": "uncertain_signal_evidence",
    }
    for stale_key in (
        "trajectory_motion_evidence",
        "trajectory_motion_evidence_output_root",
        "trajectory_motion_evidence_phase",
        "causal_filter_out",
        "causal_filter_out_output_root",
    ):
        result.pop(stale_key, None)
    return result


def step9_temporal_segmentation(ego_state, relative_motion_state):
    return {"videos": ego_state["videos"], "temporal_segments": []}


def step8_trajectory_validation(
    ego_state,
    relative_motion_state,
    *,
    phase="final",
    output_subdir=None,
    step_label=None,
):
    videos = relative_motion_state.get("videos", ego_state.get("videos", []))
    relative_motion = relative_motion_state.get("relative_object_motion", [])
    output_root = get_pipeline_output_root() / (
        output_subdir or "08b_driving_mini_causal_filter_out"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    threshold_policy = dict(
        relative_motion_state.get("trajectory_validation_threshold_policy", {})
    )
    validation_thresholds = dict(
        threshold_policy.get("thresholds", _TRAJECTORY_VALIDATION_THRESHOLDS)
    )
    threshold_policy_version = int(threshold_policy.get("version", 1))
    threshold_policy_fingerprint = hashlib.sha256(
        json.dumps(validation_thresholds, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    ego_by_video = {
        str(row.get("video_id", "")): row
        for row in ego_state.get("ego_motion", [])
        if str(row.get("video_id", ""))
    }

    protected_by_video = {}
    for row in relative_motion_state.get("protected_objects", []):
        video_id = str(row.get("video_id", ""))
        try:
            track_id = int(row.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if video_id and track_id >= 0:
            protected_by_video.setdefault(video_id, {})[track_id] = row

    trajectory_motion_evidence = []
    cached_videos = 0
    progress = tqdm(
        relative_motion,
        desc=f"[step {step_label}] trajectory_validation",
        unit="video",
    )
    for relative_video in progress:
        video_id = str(relative_video.get("video_id", ""))
        progress.set_postfix_str(video_id, refresh=False)
        video_protection = protected_by_video.get(video_id, {})
        protection_fingerprint = hashlib.sha256(
            json.dumps(
                video_protection,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()
        cache_path = output_root / video_id / "trajectory_motion_evidence.json"
        if not cache_path.exists():
            cache_path = output_root / video_id / "causal_filter_out.json"
        cached_evidence, cache_path_changes = relocate_json_cache_file(
            cache_path,
            dataset_root=relative_motion_state.get("dataset_root"),
            pipeline_root=get_pipeline_output_root(),
        )
        cache_valid = (
            isinstance(cached_evidence, dict)
            and int(cached_evidence.get("version", 0)) == _CAUSAL_FILTER_OUT_VERSION
            and str(cached_evidence.get("video_id", "")) == video_id
            and cached_evidence.get("evidence_type") == "trajectory_motion_evidence"
            and isinstance(cached_evidence.get("trajectory_motion_evidence"), list)
            and int(cached_evidence.get("threshold_policy_version", -1))
            == threshold_policy_version
            and str(cached_evidence.get("threshold_policy_fingerprint", ""))
            == threshold_policy_fingerprint
            and str(cached_evidence.get("semantic_protection_fingerprint", ""))
            == protection_fingerprint
        )
        if cache_valid:
            cached_videos += 1
            trajectory_motion_evidence.append(cached_evidence)
            if cache_path_changes:
                print(
                    f"[step {step_label or ('8b:' + phase)}] relocated cached paths "
                    f"video_id={video_id} paths={len(cache_path_changes)}"
                )
            continue
        evidence = _trajectory_motion_evidence_video(
            relative_video,
            ego_by_video.get(video_id),
            protected_by_track=video_protection,
            validation_thresholds=validation_thresholds,
        )
        num_objects_in = int(evidence.get("num_observations", 0))
        num_tracks_in = int(evidence.get("num_trajectories", 0))
        evidence.update(
            {
                "method": "causal_motion_fact_validation_evidence_build",
                "status": "trajectory_reality_validated",
                "threshold_policy_version": threshold_policy_version,
                "threshold_policy_fingerprint": threshold_policy_fingerprint,
                "semantic_protection_fingerprint": protection_fingerprint,
                "threshold_policy_frozen": bool(
                    relative_motion_state.get(
                        "trajectory_validation_threshold_policy_frozen", False
                    )
                ),
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

        step8a_root = relative_motion_state.get("symbol_grounded_refinement_output_root")
        if step8a_root:
            step8a_path = Path(step8a_root) / video_id / "symbol_grounded_refinement.json"
            step8a_payload = load_json_if_exists(step8a_path)
            if isinstance(step8a_payload, dict):
                step8a_payload["protected_objects"] = list(
                    protected_by_video.get(video_id, {}).values()
                )
                with step8a_path.open("w", encoding="utf-8") as f:
                    json.dump(step8a_payload, f, indent=2)
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
    final_threshold_conflicts = []
    if phase == "final":
        from src.exp_july.perception.trajectory_threshold_calibration import (
            collect_conflicts,
        )

        final_threshold_conflicts = collect_conflicts(trajectory_motion_evidence)
        with (output_root / "protected_invalid_threshold_conflicts.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(final_threshold_conflicts, f, indent=2)
    manifest.update(
        {
            "threshold_policy_version": threshold_policy_version,
            "threshold_policy_fingerprint": threshold_policy_fingerprint,
            "threshold_policy_frozen": bool(
                relative_motion_state.get(
                    "trajectory_validation_threshold_policy_frozen", False
                )
            ),
            "num_protected_invalid_threshold_conflicts": len(
                final_threshold_conflicts
            ),
        }
    )
    with (output_root / "trajectory_motion_evidence_manifest.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)
    with (output_root / "causal_filter_out_manifest.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[step {step_label or ('8b:' + phase)}] trajectory_motion_evidence "
        f"videos={len(trajectory_motion_evidence)} "
        f"cached={cached_videos} "
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
        "motion_signal_refinement_queue": [
            row
            for row in relative_motion_state.get("protected_objects", [])
            if bool(row.get("send_to_motion_signal_refinement", False))
        ],
        "trajectory_motion_evidence": trajectory_motion_evidence,
        "trajectory_threshold_conflicts": final_threshold_conflicts,
        "trajectory_motion_evidence_output_root": output_root,
        "trajectory_motion_evidence_phase": phase,
        f"{phase}_trajectory_motion_evidence": trajectory_motion_evidence,
        f"{phase}_trajectory_motion_evidence_output_root": output_root,
        "causal_filter_out": trajectory_motion_evidence,
        "causal_filter_out_output_root": output_root,
        "relative_object_motion": relative_motion,
        "filtered_relative_object_motion": relative_motion,
        "ego_motion": ego_state.get("ego_motion", []),
    }


def step8b_causal_filter_out(
    ego_state,
    relative_motion_state,
    *,
    phase="final",
    output_subdir=None,
    step_label=None,
):
    """Compatibility alias for the validator now used only after Step 8C."""
    return step8_trajectory_validation(
        ego_state,
        relative_motion_state,
        phase=phase,
        output_subdir=output_subdir,
        step_label=step_label,
    )


def step8c_prior_guided_ego_motion_refinement(
    ego_state,
    relative_motion_state,
    *,
    output_subdir="08c_prior_guided_ego_motion_refinement",
    step_label="8c",
):
    videos = relative_motion_state.get("videos", ego_state.get("videos", []))
    trajectory_motion_evidence = relative_motion_state.get("trajectory_motion_evidence", [])
    if not videos or not trajectory_motion_evidence:
        print(f"[step {step_label}] no trajectory motion evidence, skip prior-guided ego refinement")
        return {
            **relative_motion_state,
            "reliable_reference_objects": [],
            "prior_guided_ego_refinement_output_root": None,
        }

    output_root = get_pipeline_output_root() / output_subdir
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
        f"[step {step_label}] prior_guided_ego_motion_refinement "
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


def step8e_iterative_trajectory_pattern_repair(
    relative_motion_state,
    llm_generate=None,
    *,
    output_subdir="08e_trajectory_pattern_closed_loop",
    step_label="8e",
    postprocess=True,
):
    """Run prior-guided cohort analysis and deterministic signal repair."""
    from src.exp_july.perception.trajectory_pattern_closed_loop import (
        run_trajectory_pattern_closed_loop,
    )

    output_root = get_pipeline_output_root() / output_subdir
    result = run_trajectory_pattern_closed_loop(
        relative_motion_state,
        output_root,
        llm_generate=llm_generate,
        postprocess=postprocess,
    )
    manifest = dict(result.get("trajectory_pattern_manifest", {}))
    visualizations = list(result.get("trajectory_pattern_visualizations", []))
    summaries = list(result.get("trajectory_pattern_video_summaries", []))
    track_videos = list(result.get("trajectory_pattern_track_videos", []))
    track_video_skipped = list(
        result.get("trajectory_pattern_track_video_skipped", [])
    )
    skipped = list(result.get("trajectory_pattern_visualization_skipped", []))
    dashboard_path = str(result.get("trajectory_pattern_dashboard_path", ""))
    runtime_dashboard_path = str(result.get("trajectory_pattern_runtime_dashboard_path", ""))
    print(
        f"[step {step_label}] prior_guided_statistical_signal_repair "
        f"videos={int(manifest.get('num_videos', 0))} "
        f"tracks={int(manifest.get('num_tracks', 0))} "
        f"cohorts={int(manifest.get('num_cohorts', 0))} "
        f"rules={int(manifest.get('num_cohort_rules', 0))} "
        f"source_tracks={int(manifest.get('input_source_tracks', manifest.get('num_tracks', 0)))} "
        f"quarantined_tracks={int(manifest.get('input_quarantined_tracks', 0))} "
        f"patterns={int(manifest.get('num_patterns', 0))} "
        f"candidates={int(manifest.get('num_candidates', 0))} "
        f"repairs={int(manifest.get('num_repairs_applied', 0))} "
        f"stats_version={int(manifest.get('statistics_version', 0))} "
        f"promotion={dict(manifest.get('promotion', {})).get('decision', '')} "
        f"llm_batch={int(manifest.get('llm_batch_size', 0))} "
        f"llm_called={int(manifest.get('llm_called', 0))} "
        f"llm_skipped={int(manifest.get('llm_skipped', 0))} "
        f"llm_cache_hits={int(manifest.get('llm_cache_hits', 0))} "
        f"repair_cache_hits={int(manifest.get('repair_cache_hits', 0))} "
        f"repair_computed={int(manifest.get('repair_tracks_computed', 0))} "
        f"repair_fast_path={int(manifest.get('repair_fast_path_tracks', 0))} "
        f"workers={int(manifest.get('repair_worker_count', 1))} "
        f"single_escalations={int(manifest.get('llm_escalated_to_single', 0))} "
        f"track_visuals={len(visualizations)} "
        f"track_videos={len(track_videos)} "
        f"track_video_skipped={len(track_video_skipped)} "
        f"video_summaries={len(summaries)} "
        f"visual_skipped={len(skipped)} "
        f"dashboard={dashboard_path} "
        f"runtime_dashboard={runtime_dashboard_path}"
    )
    return result


def step8d_adaptive_protected_object_motion_repair(
    relative_motion_state,
    *,
    output_subdir="08d_adaptive_trajectory_motion_repair",
    step_label="8d",
):
    """Repair initially invalid trajectories without overwriting Step 8B evidence."""
    from src.exp_july.perception.adaptive_motion_repair import run_adaptive_motion_repair

    output_root = get_pipeline_output_root() / output_subdir
    result = run_adaptive_motion_repair(relative_motion_state, output_root)
    manifest = dict(result.get("adaptive_motion_repair_manifest", {}))
    print(
        f"[step {step_label}] adaptive_motion_repair "
        f"videos={int(manifest.get('num_videos', 0))} "
        f"queued={int(manifest.get('queued', 0))} "
        f"attempted={int(manifest.get('attempted', 0))} "
        f"repaired={int(manifest.get('repaired', 0))} "
        f"uncertain={int(manifest.get('uncertain', 0))} "
        f"unrepairable={int(manifest.get('unrepairable', 0))}"
    )
    return result


# Canonical Step 8 sequence: ID repair precedes every trajectory validation.
def step8_threshold_epoch_begin(state):
    """Activate pending validation thresholds and freeze them for all Step 8 checks."""
    from src.exp_july.perception.trajectory_threshold_calibration import (
        begin_threshold_epoch,
    )

    output_root = get_pipeline_output_root() / "08i_threshold_calibration"
    epoch_id, policy, snapshot = begin_threshold_epoch(
        output_root / "policies",
        _TRAJECTORY_VALIDATION_THRESHOLDS,
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            policy["thresholds"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    print(
        f"[step 8] threshold_epoch_begin epoch={epoch_id} "
        f"policy_v={policy['version']} "
        f"activated_pending={bool(snapshot['activated_pending_policy'])} "
        f"fingerprint={fingerprint[:12]}"
    )
    return {
        **state,
        "trajectory_validation_threshold_epoch_id": epoch_id,
        "trajectory_validation_threshold_policy": policy,
        "trajectory_validation_threshold_policy_frozen": True,
        "trajectory_validation_threshold_policy_fingerprint": fingerprint,
        "trajectory_validation_threshold_policy_output_root": output_root / "policies",
    }


def step8_trajectory_repair(position_state, ego_state):
    return step7b_tracklet_repair(
        position_state,
        ego_state,
        output_subdir="08_trajectory_repair",
        step_label="8",
        repair_cfg={"max_gap_frames": 0},
    )


def step8a_relative_object_motion(position_state, repaired_state):
    return step8_relative_object_motion(
        position_state,
        repaired_state,
        output_subdir="08a_relative_object_motion",
        step_label="8a",
    )


def step8b_signal_evidence(state):
    return step8b_uncertain_signal_evidence(
        state,
        output_subdir="08b_uncertain_signal_evidence",
        step_label="8b",
    )


def step8b_trajectory_validation(ego_state, state):
    """Compatibility alias for the refactored non-classifying Step 8B."""
    del ego_state
    return step8b_signal_evidence(state)


_STEP8_WHOLE_CACHE_VERSION = 1


def _step8_cache_fingerprint(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _load_step8_whole_cache(path, fingerprint, required_paths=()):
    path = Path(path)
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        int(cached.get("cache_version", 0)) != _STEP8_WHOLE_CACHE_VERSION
        or str(cached.get("input_fingerprint", "")) != fingerprint
        or not isinstance(cached.get("outputs"), dict)
    ):
        return None
    if any(not Path(required).exists() for required in required_paths if required):
        return None
    return dict(cached["outputs"])


def _write_step8_whole_cache(path, fingerprint, outputs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "cache_version": _STEP8_WHOLE_CACHE_VERSION,
                "input_fingerprint": fingerprint,
                "outputs": outputs,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _step8c_input_fingerprint(state):
    evidence = state.get(
        "uncertain_signal_evidence",
        state.get("trajectory_motion_evidence", []),
    )
    evidence_rows = [
        {
            "video_id": row.get("video_id", ""),
            "version": row.get("version", 0),
            "source_signal_fingerprint": row.get(
                "source_signal_fingerprint", ""
            ),
            "num_tracks": row.get("num_tracks", 0),
            "num_observations": row.get("num_observations", 0),
            "track_signal_evidence": row.get("track_signal_evidence", []),
            "quarantined_track_signal_evidence": row.get(
                "quarantined_track_signal_evidence", []
            ),
        }
        for row in evidence
    ]
    return _step8_cache_fingerprint(
        {
            "schema": "step8c-trajectory-clustering-v2",
            "dataset": state.get("dataset_name", "driving_mini"),
            "evidence": evidence_rows,
        }
    )


def _write_step8_stage_manifest(output_subdir, filename, payload):
    output_root = get_pipeline_output_root() / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / filename
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_root, path


def step8c_trajectory_clustering(state, llm_generate=None):
    """Build symbolic tracks and assign cohorts; never repair a trajectory."""
    from src.exp_july.perception.trajectory_pattern_closed_loop import (
        llm_call,
        symbolic_tracks,
    )
    from src.exp_july.perception.trajectory_cohort_policy import (
        assign_cohorts,
        attach_static_metadata,
        cohort_statistics,
        compile_rules,
        rule_generation_prompt,
    )

    output_root = get_pipeline_output_root() / "08c_trajectory_clustering"
    audit_root = output_root / "llm_audit"
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = str(state.get("dataset_name", "driving_mini"))
    input_fingerprint = _step8c_input_fingerprint(state)
    whole_cache_path = output_root / "whole_step_cache.json"
    cached_outputs = _load_step8_whole_cache(
        whole_cache_path,
        input_fingerprint,
        required_paths=(
            output_root / "clustered_tracks.json",
            output_root / "compiled_cohort_rules.json",
            output_root / "cohort_statistics.json",
            output_root / "trajectory_clustering_manifest.json",
        ),
    )
    if cached_outputs is not None:
        manifest = dict(cached_outputs.get("trajectory_clustering_manifest", {}))
        manifest["whole_step_cache_hit"] = True
        cached_outputs["trajectory_clustering_manifest"] = manifest
        cached_outputs["trajectory_clustering_output_root"] = output_root
        print(
            f"[step 8c] trajectory_clustering WHOLE_STEP_CACHE_HIT "
            f"videos={manifest.get('num_videos', 0)} "
            f"tracks={manifest.get('num_tracks', 0)} "
            f"rules={manifest.get('num_rules', 0)} "
            f"cohorts={manifest.get('num_cohorts', 0)}",
            flush=True,
        )
        return {**state, **cached_outputs}
    evidence = state.get(
        "uncertain_signal_evidence",
        state.get("trajectory_motion_evidence", []),
    )
    tracks = symbolic_tracks(evidence, state.get("relative_object_motion", []))
    metadata_catalog = attach_static_metadata(tracks)
    raw_rules = llm_call(
        "cohort_rule_generation",
        rule_generation_prompt(dataset, metadata_catalog),
        audit_root,
        llm_generate,
    )
    compiled_policy = compile_rules(raw_rules)
    cohorts = assign_cohorts(tracks, compiled_policy["rules"])
    summaries = cohort_statistics(cohorts, None)
    manifest = {
        "version": 1,
        "stage": "8c_trajectory_clustering",
        "repairs_performed": 0,
        "num_tracks": len(tracks),
        "num_videos": len({str(row.get("video_id", "")) for row in tracks}),
        "num_rules": len(compiled_policy.get("rules", [])),
        "num_cohorts": len(cohorts),
        "cohort_track_counts": {
            key: len(value) for key, value in sorted(cohorts.items())
        },
        "input_fingerprint": input_fingerprint,
        "whole_step_cache_hit": False,
    }
    (output_root / "clustered_tracks.json").write_text(
        json.dumps(tracks, indent=2, default=str), encoding="utf-8"
    )
    (output_root / "compiled_cohort_rules.json").write_text(
        json.dumps(compiled_policy, indent=2), encoding="utf-8"
    )
    (output_root / "cohort_statistics.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    (output_root / "trajectory_clustering_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"[step 8c] trajectory_clustering videos={manifest['num_videos']} "
        f"tracks={manifest['num_tracks']} rules={manifest['num_rules']} "
        f"cohorts={manifest['num_cohorts']} repairs=0"
    )
    outputs = {
        "trajectory_clustered_tracks": tracks,
        "trajectory_cohort_metadata_catalog": metadata_catalog,
        "trajectory_cohort_rule_policy": compiled_policy,
        "trajectory_cohort_statistics": summaries,
        "trajectory_clustering_manifest": manifest,
        "trajectory_clustering_output_root": str(output_root),
    }
    _write_step8_whole_cache(whole_cache_path, input_fingerprint, outputs)
    outputs["trajectory_clustering_output_root"] = output_root
    return {**state, **outputs}


_STEP8D_CACHE_OUTPUT_KEYS = (
    "pre_pattern_relative_object_motion",
    "relative_object_motion",
    "filtered_relative_object_motion",
    "trajectory_pattern_records",
    "trajectory_cohort_metadata_catalog",
    "trajectory_cohort_rule_policy",
    "trajectory_cohort_statistics",
    "trajectory_cohort_operator_plans",
    "trajectory_cohort_output_root",
    "trajectory_pattern_definitions",
    "trajectory_pattern_statistics_candidate",
    "trajectory_pattern_statistics_review",
    "trajectory_pattern_statistics_promotion",
    "trajectory_pattern_epoch_policy",
    "trajectory_pattern_epoch_reviews",
    "trajectory_pattern_manifest",
    "trajectory_pattern_output_root",
    "trajectory_pattern_runtime_monitor",
    "trajectory_pattern_runtime_dashboard_path",
    "trajectory_pattern_runtime_output_root",
)


def step8d_closed_loop_trajectory_repair(state, llm_generate=None):
    """Consume frozen Step 8C cohorts and run deterministic repair candidates."""
    output_root = get_pipeline_output_root() / "08d_closed_loop_trajectory_repair"
    clustering_manifest = dict(state.get("trajectory_clustering_manifest", {}))
    input_fingerprint = _step8_cache_fingerprint(
        {
            "schema": "step8d-closed-loop-repair-v2",
            "step8c_input_fingerprint": clustering_manifest.get(
                "input_fingerprint", ""
            ),
            "clustered_tracks": state.get("trajectory_clustered_tracks", []),
            "cohort_rule_policy": state.get("trajectory_cohort_rule_policy", {}),
            "ego_motion": state.get("ego_motion", []),
        }
    )
    whole_cache_path = output_root / "whole_step_cache.json"
    cached_outputs = _load_step8_whole_cache(
        whole_cache_path,
        input_fingerprint,
        required_paths=(output_root / "trajectory_pattern_manifest.json",),
    )
    if cached_outputs is not None:
        manifest = dict(cached_outputs.get("trajectory_pattern_manifest", {}))
        manifest["whole_step_cache_hit"] = True
        records = list(cached_outputs.get("trajectory_pattern_records", []))
        for record in records:
            llm_processing = dict(record.get("llm_processing", {}))
            llm_processing["repair_cache_hit"] = True
            record["llm_processing"] = llm_processing
        cached_outputs["trajectory_pattern_records"] = records
        manifest["repair_cache_hits"] = len(records)
        manifest["repair_tracks_computed"] = 0
        cached_outputs["trajectory_pattern_manifest"] = manifest
        print(
            f"[step 8d] closed_loop_trajectory_repair WHOLE_STEP_CACHE_HIT "
            f"videos={manifest.get('num_videos', 0)} "
            f"tracks={manifest.get('num_tracks', 0)} "
            f"repairs={manifest.get('num_repairs_applied', 0)}",
            flush=True,
        )
        return {**state, **cached_outputs}
    result = step8e_iterative_trajectory_pattern_repair(
        state,
        llm_generate=llm_generate,
        output_subdir="08d_closed_loop_trajectory_repair",
        step_label="8d",
        postprocess=False,
    )
    outputs = {
        key: result[key]
        for key in _STEP8D_CACHE_OUTPUT_KEYS
        if key in result
    }
    manifest = dict(outputs.get("trajectory_pattern_manifest", {}))
    manifest["whole_step_cache_hit"] = False
    manifest["whole_step_input_fingerprint"] = input_fingerprint
    outputs["trajectory_pattern_manifest"] = manifest
    result["trajectory_pattern_manifest"] = manifest
    _write_step8_whole_cache(whole_cache_path, input_fingerprint, outputs)
    return result


def step8e_repaired_trajectory_validation(state):
    """Publish repair validation outcomes as an independent audit stage."""
    rows = [
        {
            "video_id": row.get("video_id"),
            "track_id": row.get("track_id"),
            "repair_applied": bool(row.get("repair_applied")),
            "initial_validation_status": row.get("initial_8c_validation_status"),
            "final_validation_status": row.get("final_validation_status"),
            "resolution_status": row.get("resolution_status"),
            "validated_pattern": row.get("validated_pattern"),
            "selected_candidate_id": dict(row.get("selected_candidate", {})).get("candidate_id"),
            "final_selection_reason": row.get("final_selection_reason"),
        }
        for row in state.get("trajectory_pattern_records", [])
    ]
    payload = {
        "version": 1,
        "stage": "8e_repaired_trajectory_validation",
        "num_tracks": len(rows),
        "num_repaired": sum(row["repair_applied"] for row in rows),
        "num_unresolved": sum(row["resolution_status"] == "unresolved_uncertain" for row in rows),
        "tracks": rows,
    }
    root, path = _write_step8_stage_manifest(
        "08e_repaired_trajectory_validation",
        "repaired_trajectory_validation.json",
        payload,
    )
    print(
        f"[step 8e] repaired_trajectory_validation tracks={len(rows)} "
        f"repaired={payload['num_repaired']} unresolved={payload['num_unresolved']}"
    )
    return {**state, "step8e_validation_manifest": payload, "step8e_validation_path": str(path), "step8e_validation_output_root": root}


def step8f_trajectory_statistics(state):
    """Expose versioned statistical aggregation and promotion independently."""
    payload = {
        "version": 1,
        "stage": "8f_trajectory_statistics",
        "candidate_table": state.get("trajectory_pattern_statistics_candidate", {}),
        "reviews": state.get("trajectory_pattern_statistics_review", []),
        "promotion": state.get("trajectory_pattern_statistics_promotion", {}),
    }
    root, path = _write_step8_stage_manifest(
        "08f_trajectory_statistics", "trajectory_statistics.json", payload
    )
    print(
        f"[step 8f] trajectory_statistics reviews={len(payload['reviews'])} "
        f"promotion={dict(payload['promotion']).get('decision', 'unknown')}"
    )
    return {**state, "step8f_statistics_manifest": payload, "step8f_statistics_path": str(path), "step8f_statistics_output_root": root}


def step8g_repaired_track_materialization(state):
    """Checkpoint the repaired relative-motion tracks for downstream consumers."""
    videos = list(state.get("relative_object_motion", []))
    payload = {
        "version": 1,
        "stage": "8g_repaired_track_materialization",
        "num_videos": len(videos),
        "num_tracks": len(state.get("trajectory_pattern_records", [])),
        "source_step": "8d_closed_loop_trajectory_repair",
    }
    root, path = _write_step8_stage_manifest(
        "08g_repaired_track_materialization", "materialization_manifest.json", payload
    )
    print(f"[step 8g] repaired_track_materialization videos={len(videos)} tracks={payload['num_tracks']}")
    return {**state, "step8g_materialization_manifest": payload, "step8g_materialization_path": str(path), "step8g_materialization_output_root": root}


_STEP8H_CACHE_OUTPUT_KEYS = (
    "trajectory_pattern_visualizations",
    "trajectory_pattern_video_summaries",
    "trajectory_pattern_statistical_pdf_reports",
    "trajectory_pattern_statistical_summary",
    "trajectory_pattern_statistical_summary_path",
    "trajectory_pattern_statistical_pdf_output_root",
    "trajectory_pattern_track_videos",
    "trajectory_pattern_track_video_skipped",
    "trajectory_pattern_track_video_selections",
    "trajectory_pattern_track_video_manifest_path",
    "trajectory_pattern_visualization_skipped",
    "trajectory_pattern_visualization_output_root",
)


def _step8h_media_paths(outputs):
    paths = []
    for key in (
        "trajectory_pattern_statistical_pdf_reports",
        "trajectory_pattern_track_videos",
    ):
        for row in outputs.get(key, []):
            if isinstance(row, str):
                paths.append(row)
            elif isinstance(row, dict):
                candidate = row.get("path", row.get("output_path", ""))
                if candidate:
                    paths.append(candidate)
    return paths


def step8h_trajectory_repair_visualization(state):
    """Render Step 8C–8G comparison MP4 videos and statistical PDFs only."""
    from src.exp_july.perception.trajectory_pattern_visualization import (
        render_trajectory_pattern_visualizations,
    )

    root = get_pipeline_output_root() / "08h_trajectory_repair_visualization"
    input_fingerprint = _step8_cache_fingerprint(
        {
            "schema": "step8h-trajectory-repair-visualization-v2",
            "records": state.get("trajectory_pattern_records", []),
            "relative_object_motion": state.get("relative_object_motion", []),
            "pre_pattern_relative_object_motion": state.get(
                "pre_pattern_relative_object_motion", []
            ),
            "ego_motion": state.get("ego_motion", []),
            "ego_symbol_prior": state.get("ego_symbol_prior", []),
            "statistics_promotion": state.get(
                "trajectory_pattern_statistics_promotion", {}
            ),
            "fps": state.get("step8bc_visualization_fps", 10.0),
        }
    )
    cache_path = root / "whole_step_cache.json"
    cached_outputs = _load_step8_whole_cache(cache_path, input_fingerprint)
    if cached_outputs is not None and all(
        Path(path).exists() for path in _step8h_media_paths(cached_outputs)
    ):
        cached_outputs["trajectory_pattern_visualization_output_root"] = root
        print(
            f"[step 8h] trajectory_repair_visualization WHOLE_STEP_CACHE_HIT "
            f"mp4s={len(cached_outputs.get('trajectory_pattern_track_videos', []))} "
            f"pdfs={len(cached_outputs.get('trajectory_pattern_statistical_pdf_reports', []))}",
            flush=True,
        )
        return {**state, **cached_outputs}
    result = render_trajectory_pattern_visualizations(state, root)
    outputs = {
        key: result[key] for key in _STEP8H_CACHE_OUTPUT_KEYS if key in result
    }
    _write_step8_whole_cache(cache_path, input_fingerprint, outputs)
    print(
        f"[step 8h] trajectory_repair_visualization "
        f"reports={len(result.get('trajectory_pattern_visualizations', []))} "
        f"pdfs={len(result.get('trajectory_pattern_statistical_pdf_reports', []))}"
    )
    return result


def step8i_trajectory_audit_dashboard(state):
    """Build the read-only offline audit dashboard in its own stage."""
    from src.exp_july.perception.trajectory_pattern_dashboard import (
        build_trajectory_pattern_dashboard,
    )

    root = get_pipeline_output_root() / "08i_trajectory_audit_dashboard"
    audit_root = Path(state.get("trajectory_pattern_output_root", root)) / "llm_audit"
    input_fingerprint = _step8_cache_fingerprint(
        {
            "schema": "step8i-trajectory-audit-dashboard-v2",
            "records": state.get("trajectory_pattern_records", []),
            "manifest": state.get("trajectory_pattern_manifest", {}),
            "promotion": state.get("trajectory_pattern_statistics_promotion", {}),
            "llm_audit_files": sorted(
                (path.name, path.stat().st_size, path.stat().st_mtime_ns)
                for path in audit_root.glob("*.json")
                if path.is_file()
            ) if audit_root.exists() else [],
        }
    )
    cache_path = root / "whole_step_cache.json"
    cached_outputs = _load_step8_whole_cache(
        cache_path,
        input_fingerprint,
        required_paths=(root / "index.html", root / "dashboard_manifest.json"),
    )
    if cached_outputs is not None:
        cached_outputs["trajectory_pattern_dashboard_path"] = root / "index.html"
        cached_outputs["trajectory_pattern_dashboard_output_root"] = root
        print(
            f"[step 8i] trajectory_audit_dashboard WHOLE_STEP_CACHE_HIT "
            f"path={root / 'index.html'}",
            flush=True,
        )
        return {**state, **cached_outputs}
    result = build_trajectory_pattern_dashboard(state, root, audit_root)
    outputs = {
        key: result[key]
        for key in (
            "trajectory_pattern_dashboard",
            "trajectory_pattern_dashboard_path",
            "trajectory_pattern_dashboard_output_root",
        )
        if key in result
    }
    _write_step8_whole_cache(cache_path, input_fingerprint, outputs)
    print(f"[step 8i] trajectory_audit_dashboard path={result.get('trajectory_pattern_dashboard_path', '')}")
    return result


def step8j_trajectory_provenance_audit(state):
    """Persist the cross-stage provenance map without changing decisions."""
    payload = {
        "version": 1,
        "stage": "8j_trajectory_provenance_audit",
        "flow": ["8c_clustering", "8d_repair", "8e_validation", "8f_statistics", "8g_materialization", "8h_visualization", "8i_dashboard"],
        "clustering_manifest": state.get("trajectory_clustering_manifest", {}),
        "repair_manifest": state.get("trajectory_pattern_manifest", {}),
        "validation_manifest": state.get("step8e_validation_manifest", {}),
        "statistics_promotion": state.get("trajectory_pattern_statistics_promotion", {}),
        "dashboard_path": state.get("trajectory_pattern_dashboard_path", ""),
    }
    root, path = _write_step8_stage_manifest(
        "08j_trajectory_provenance_audit", "trajectory_provenance.json", payload
    )
    print(f"[step 8j] trajectory_provenance_audit path={path}")
    return {**state, "step8j_provenance_manifest": payload, "step8j_provenance_path": str(path), "step8j_provenance_output_root": root}


def step8k_trajectory_handoff(state):
    """Finalize the new Step 8 branch and expose its downstream handoff."""
    payload = {
        "version": 1,
        "stage": "8k_trajectory_handoff",
        "status": "completed",
        "num_videos": len(state.get("relative_object_motion", [])),
        "num_tracks": len(state.get("trajectory_pattern_records", [])),
        "num_repairs": sum(bool(row.get("repair_applied")) for row in state.get("trajectory_pattern_records", [])),
        "legacy_steps_8d_through_8i_enabled": False,
        "threshold_epoch_enabled": False,
    }
    root, path = _write_step8_stage_manifest(
        "08k_trajectory_handoff", "trajectory_handoff_manifest.json", payload
    )
    print(
        f"[step 8k] trajectory_handoff videos={payload['num_videos']} "
        f"tracks={payload['num_tracks']} repairs={payload['num_repairs']}"
    )
    return {**state, "step8k_handoff_manifest": payload, "step8k_handoff_path": str(path), "step8k_handoff_output_root": root}


def step8c_trajectory_pattern_closed_loop(state, llm_generate=None):
    """Compatibility entry point for Step 8C statistical cohort repair."""
    return step8e_iterative_trajectory_pattern_repair(
        state,
        llm_generate=llm_generate,
        output_subdir="08c_trajectory_pattern_closed_loop",
        step_label="8c",
    )


def step8d_pattern_refined_validation(ego_state, state):
    return step8_trajectory_validation(
        ego_state,
        state,
        phase="pattern_refined",
        output_subdir="08d_pattern_refined_trajectory_validation",
        step_label="8d",
    )


def step8e_semantic_protection(state, llm_generate=None):
    return step8a_symbol_grounded_refinement(
        state,
        llm_generate=llm_generate,
        output_subdir="08e_symbol_grounded_refinement",
        step_label="8e",
    )


def step8e_visual_semantic_protection(state):
    return step8a_visual_symbol_grounded(
        state,
        output_subdir="08e_visual_symbol_grounded",
        step_label="8e visual",
    )


def step8f_final_trajectory_validation(ego_state, state):
    return step8_trajectory_validation(
        ego_state,
        state,
        phase="final",
        output_subdir="08f_final_trajectory_validation",
        step_label="8f",
    )


def step8g_prior_guided_ego_motion_refinement(ego_state, state):
    return step8c_prior_guided_ego_motion_refinement(
        ego_state,
        state,
        output_subdir="08g_prior_guided_ego_motion_refinement",
        step_label="8g",
    )


def step8h_visual_relative_motion(state, fps=10.0):
    important_video_ids = {
        str(row.get("video_id", ""))
        for key in ("important_objects", "protected_objects")
        for row in state.get(key, [])
        if str(row.get("video_id", ""))
    }
    return step8_visual_relative_motion(
        state,
        fps=fps,
        output_subdir="08h_relative_motion_tracks",
        step_label="8h",
        render_video_ids=important_video_ids,
    )


def step8i_threshold_calibration(state):
    """Calibrate a pending soft-threshold patch from batched semantic conflicts."""
    from src.exp_july.perception.trajectory_threshold_calibration import (
        run_threshold_calibration,
    )

    output_root = get_pipeline_output_root() / "08i_threshold_calibration"
    result = run_threshold_calibration(
        state,
        output_root,
        _TRAJECTORY_VALIDATION_THRESHOLDS,
        _trajectory_reality_validation,
    )
    from src.exp_july.perception.trajectory_cohort_policy import (
        write_downstream_feedback,
    )

    cohort_feedback = write_downstream_feedback(result)
    result = {
        **result,
        "trajectory_cohort_downstream_feedback": cohort_feedback,
    }
    manifest = dict(result.get("trajectory_threshold_calibration_manifest", {}))
    promotion = dict(manifest.get("promotion", {}))
    print(
        f"[step 8i] threshold_calibration "
        f"conflicts={int(manifest.get('num_conflicts', 0))} "
        f"update_conflicts={int(manifest.get('num_update_conflicts', 0))} "
        f"changes={len(dict(manifest.get('compilation', {})).get('changes', {}))} "
        f"decision={promotion.get('decision', 'reject')} "
        f"reason={promotion.get('reason', '')} "
        f"cohort_feedback={len(dict(cohort_feedback.get('cohorts', {})))} "
        f"critical_regressions={sum(int(row.get('critical_regressions', 0)) for row in dict(cohort_feedback.get('cohorts', {})).values())}"
    )
    return result


def step10_segment_object_motion(segment_state):
    return {"videos": segment_state["videos"], "segment_object_motion": []}
