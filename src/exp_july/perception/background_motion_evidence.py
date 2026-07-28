"""Independent background-motion evidence for provisional ego segments."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path


VERSION = 1
DEFAULT_CONFIG = {
    "region_rows": 3,
    "region_cols": 3,
    "max_patches_per_region": 24,
    "quality_level": 0.015,
    "min_patch_distance_px": 9.0,
    "patch_block_size": 7,
    "object_bbox_margin_px": 8,
    "lk_window_px": 21,
    "lk_max_level": 3,
    "forward_backward_error_px": 1.5,
    "min_vector_magnitude_px": 0.05,
    "local_residual_floor_px": 1.5,
    "local_residual_mad_scale": 3.5,
    "radial_deadband_px": 0.05,
    "min_reliable_tracks_per_pair": 4,
    "execution_profile": "train",
    "frame_stride": 1,
    "forward_backward_check": True,
}

PROFILE_CONFIGS = {
    "train": {},
    "eval-fast": {
        "execution_profile": "eval-fast",
        "frame_stride": 4,
        "region_rows": 2,
        "region_cols": 3,
        "max_patches_per_region": 10,
        "lk_max_level": 2,
        "forward_backward_check": False,
    },
}


def resolved_config(config=None):
    supplied = dict(config or {})
    profile = str(supplied.get("execution_profile", "train"))
    if profile not in PROFILE_CONFIGS:
        raise ValueError(
            f"unknown Step 7 execution profile {profile!r}; "
            f"expected one of {sorted(PROFILE_CONFIGS)}"
        )
    result = dict(DEFAULT_CONFIG)
    result.update(PROFILE_CONFIGS[profile])
    result.update(supplied)
    return result


def _bbox_rows(frame):
    rows = []
    for obj in frame.get("objects", []):
        box = obj.get("bbox", obj.get("box"))
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            rows.append(box[:4])
    for key in ("boxes", "bboxes"):
        for box in frame.get(key, []):
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                rows.append(box[:4])
    unique = []
    seen = set()
    for box in rows:
        try:
            normalized = tuple(float(value) for value in box)
        except (TypeError, ValueError):
            continue
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _background_mask(cv2, np, shape, frame, cfg):
    height, width = shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    margin = int(cfg["object_bbox_margin_px"])
    excluded = []
    for x1, y1, x2, y2 in _bbox_rows(frame):
        left = max(0, int(math.floor(min(x1, x2))) - margin)
        top = max(0, int(math.floor(min(y1, y2))) - margin)
        right = min(width - 1, int(math.ceil(max(x1, x2))) + margin)
        bottom = min(height - 1, int(math.ceil(max(y1, y2))) + margin)
        if right <= left or bottom <= top:
            continue
        cv2.rectangle(mask, (left, top), (right, bottom), 0, -1)
        excluded.append([left, top, right, bottom])
    return mask, excluded


def _region_bounds(width, height, row, col, rows, cols):
    x1 = int(round(col * width / cols))
    x2 = int(round((col + 1) * width / cols))
    y1 = int(round(row * height / rows))
    y2 = int(round((row + 1) * height / rows))
    return x1, y1, x2, y2


def _sample_points(cv2, np, gray, background_mask, cfg):
    height, width = gray.shape[:2]
    points = []
    regions = []
    texture = cv2.cornerMinEigenVal(
        gray, blockSize=max(3, int(cfg["patch_block_size"])), ksize=3
    )
    for row in range(int(cfg["region_rows"])):
        for col in range(int(cfg["region_cols"])):
            x1, y1, x2, y2 = _region_bounds(
                width, height, row, col,
                int(cfg["region_rows"]), int(cfg["region_cols"]),
            )
            region_mask = np.zeros_like(background_mask)
            region_mask[y1:y2, x1:x2] = background_mask[y1:y2, x1:x2]
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=int(cfg["max_patches_per_region"]),
                qualityLevel=float(cfg["quality_level"]),
                minDistance=float(cfg["min_patch_distance_px"]),
                mask=region_mask,
                blockSize=int(cfg["patch_block_size"]),
                useHarrisDetector=False,
            )
            if corners is None:
                continue
            region_id = f"r{row}c{col}"
            for corner in corners.reshape(-1, 2):
                x, y = float(corner[0]), float(corner[1])
                ix = max(0, min(width - 1, int(round(x))))
                iy = max(0, min(height - 1, int(round(y))))
                points.append([x, y])
                regions.append((region_id, float(texture[iy, ix])))
    if not points:
        return None, []
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2), regions


def _track_pair(cv2, np, left_frame, right_frame, cfg):
    left_path = str(left_frame.get("image_path", ""))
    right_path = str(right_frame.get("image_path", ""))
    left_gray = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE) if left_path else None
    right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE) if right_path else None
    result = {
        "start_frame": int(left_frame.get("frame_index", 0)),
        "end_frame": int(right_frame.get("frame_index", 0)),
        "source_image_paths": [left_path, right_path],
        "status": "completed",
        "raw_patch_count": 0,
        "accepted_patch_count": 0,
        "rejected_patch_counts": {},
        "patch_vectors": [],
        "covered_regions": [],
        "excluded_object_bboxes": [],
    }
    if left_gray is None or right_gray is None:
        result["status"] = "missing_image"
        return result
    if left_gray.shape != right_gray.shape:
        result["status"] = "image_shape_mismatch"
        return result
    mask, excluded = _background_mask(cv2, np, left_gray.shape, left_frame, cfg)
    result["excluded_object_bboxes"] = excluded
    points, point_metadata = _sample_points(cv2, np, left_gray, mask, cfg)
    if points is None:
        result["status"] = "no_background_patches"
        return result
    result["raw_patch_count"] = int(len(points))
    lk = {
        "winSize": (int(cfg["lk_window_px"]), int(cfg["lk_window_px"])),
        "maxLevel": int(cfg["lk_max_level"]),
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01
        ),
    }
    forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        left_gray, right_gray, points, None, **lk
    )
    if forward is None:
        result["status"] = "forward_tracking_failed"
        return result
    use_backward = bool(cfg.get("forward_backward_check", True))
    if use_backward:
        backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
            right_gray, left_gray, forward, None, **lk
        )
        if backward is None:
            result["status"] = "backward_tracking_failed"
            return result
    else:
        backward = points
        status_backward = np.ones_like(status_forward)
    height, width = left_gray.shape[:2]
    center_x, center_y = 0.5 * width, 0.5 * height
    candidates = []
    rejected = Counter()
    for index, (start, end, back) in enumerate(
        zip(points.reshape(-1, 2), forward.reshape(-1, 2), backward.reshape(-1, 2))
    ):
        if not bool(status_forward[index]) or not bool(status_backward[index]):
            rejected["lk_status"] += 1
            continue
        fb_error = float(np.linalg.norm(back - start)) if use_backward else 0.0
        if use_backward and fb_error > float(cfg["forward_backward_error_px"]):
            rejected["forward_backward_error"] += 1
            continue
        x2, y2 = float(end[0]), float(end[1])
        if not (0 <= x2 < width and 0 <= y2 < height):
            rejected["outside_frame"] += 1
            continue
        dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
        magnitude = float(math.hypot(dx, dy))
        region_id, texture_score = point_metadata[index]
        candidates.append({
            "patch_id": index,
            "region_id": region_id,
            "start_xy": [float(start[0]), float(start[1])],
            "end_xy": [x2, y2],
            "dx": dx,
            "dy": dy,
            "magnitude": magnitude,
            "forward_backward_error": fb_error,
            "texture_score": texture_score,
        })
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["region_id"]].append(candidate)
    accepted = []
    for region_rows in grouped.values():
        median_dx = float(np.median([row["dx"] for row in region_rows]))
        median_dy = float(np.median([row["dy"] for row in region_rows]))
        residuals = [
            math.hypot(row["dx"] - median_dx, row["dy"] - median_dy)
            for row in region_rows
        ]
        median_residual = float(np.median(residuals))
        mad = float(np.median([abs(value - median_residual) for value in residuals]))
        limit = max(
            float(cfg["local_residual_floor_px"]),
            median_residual + float(cfg["local_residual_mad_scale"]) * max(mad, 1e-6),
        )
        for row, residual in zip(region_rows, residuals):
            if residual > limit:
                rejected["local_inconsistency"] += 1
                continue
            x, y = row["start_xy"]
            radius = max(1.0, math.hypot(x - center_x, y - center_y))
            radial = float(
                (row["dx"] * (x - center_x) + row["dy"] * (y - center_y))
                / radius
            )
            row["local_vector_residual"] = float(residual)
            row["radial_projection"] = radial
            row["radial_state"] = (
                "expansion" if radial > float(cfg["radial_deadband_px"])
                else "contraction" if radial < -float(cfg["radial_deadband_px"])
                else "neutral"
            )
            row["provenance"] = {
                "estimator": (
                    "sparse_lk_forward_backward" if use_backward else "sparse_lk_forward"
                ),
                "independent_from_existing_ego_vz": True,
                "object_bbox_excluded": True,
                "forward_backward_check": use_backward,
            }
            accepted.append(row)
    result["patch_vectors"] = accepted
    result["accepted_patch_count"] = len(accepted)
    result["rejected_patch_counts"] = dict(sorted(rejected.items()))
    result["covered_regions"] = sorted({row["region_id"] for row in accepted})
    if len(accepted) < int(cfg["min_reliable_tracks_per_pair"]):
        result["status"] = "insufficient_reliable_tracks"
    return result


def extract_video_evidence(position_video, provisional_video, config=None):
    import cv2
    import numpy as np

    cfg = resolved_config(config)
    frames = {
        int(frame.get("frame_index", offset)): frame
        for offset, frame in enumerate(position_video.get("frames", []))
    }
    segment_results = []
    total_regions = int(cfg["region_rows"]) * int(cfg["region_cols"])
    stride = max(1, int(cfg.get("frame_stride", 1)))
    for segment in provisional_video.get("final_action_segments", []):
        start = int(segment.get("start_frame", 0))
        end = int(segment.get("end_frame", start))
        frame_ids = [index for index in sorted(frames) if start <= index <= end]
        sampled_ids = frame_ids[::stride]
        if len(frame_ids) > 1 and sampled_ids[-1] != frame_ids[-1]:
            sampled_ids.append(frame_ids[-1])
        pairs = []
        for left_id, right_id in zip(sampled_ids, sampled_ids[1:]):
            pairs.append(_track_pair(cv2, np, frames[left_id], frames[right_id], cfg))
        vectors = [vector for pair in pairs for vector in pair["patch_vectors"]]
        reliable_pairs = [pair for pair in pairs if pair["status"] == "completed"]
        radial_counts = Counter(vector["radial_state"] for vector in vectors)
        vector_count = max(1, len(vectors))
        covered = sorted({region for pair in pairs for region in pair["covered_regions"]})
        persistence = float(len(reliable_pairs) / max(1, len(pairs)))
        raw_count = sum(pair["raw_patch_count"] for pair in pairs)
        reliability = float(len(vectors) / max(1, raw_count))
        coverage = float(len(covered) / max(1, total_regions))
        confidence = float(max(0.0, min(1.0, persistence * math.sqrt(max(0.0, reliability * coverage)))))
        segment_results.append({
            "segment_id": int(segment.get("segment_id", len(segment_results))),
            "provisional_action": str(segment.get("action", "unknown")),
            "start_frame": start,
            "end_frame": end,
            "duration_frames": int(segment.get("duration_frames", len(frame_ids))),
            "sampled_frame_count": len(sampled_ids),
            "sampled_pair_count": len(pairs),
            "status": "completed" if vectors else "insufficient_evidence",
            "patch_vectors": vectors,
            "frame_pair_evidence": pairs,
            "radial_expansion_support": float(radial_counts["expansion"] / vector_count),
            "radial_contraction_support": float(radial_counts["contraction"] / vector_count),
            "radial_neutral_support": float(radial_counts["neutral"] / vector_count),
            "spatial_coverage": coverage,
            "covered_regions": covered,
            "temporal_persistence": persistence,
            "tracking_reliability": reliability,
            "estimator_confidence": confidence,
            "num_raw_patches": raw_count,
            "num_accepted_vectors": len(vectors),
            "provenance": {
                "source_step": "7b_background_motion_evidence",
                "source_provisional_step": "7a_ego_symbol_prior",
                "estimator": "opencv_sparse_lk_forward_backward",
                "independent_from_existing_ego_vz": True,
                "known_object_bboxes_excluded_when_available": True,
                "configuration": cfg,
            },
        })
    return {
        "version": VERSION,
        "video_id": str(position_video.get("video_id", "")),
        "status": "completed",
        "input_label_status": "provisional",
        "evidence_role": "independent_background_motion_evidence",
        "segments": segment_results,
        "num_segments": len(segment_results),
        "num_patch_vectors": sum(row["num_accepted_vectors"] for row in segment_results),
        "execution_profile": cfg["execution_profile"],
        "configuration": cfg,
    }
