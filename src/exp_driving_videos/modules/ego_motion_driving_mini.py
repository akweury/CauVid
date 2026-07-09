"""
Estimate ego motion for driving_mini videos.

For each consecutive frame pair:
  1. Detect sparse background features after masking dynamic object boxes.
  2. Track them with pyramidal Lucas-Kanade optical flow.
  3. Reject outliers with forward-backward consistency and affine RANSAC.
  4. Estimate (vx, vz) lateral and forward ego speed + yaw rate using
     depth-weighted sparse background motion.

Consumes:
  - Step 4 merged annotations (image_path, boxes per frame)
  - Depth maps from pipeline_output/05_driving_mini_3d_positions (or depth_maps root)

Output layout:
    pipeline_output/06_driving_mini_ego_motion/
        ego_motion_manifest.json
        <video_id>/
            ego_motion.json          — raw + smoothed per-frame ego motion signals
            ego_motion_vis.mp4       — side-by-side: original frame + smoothed charts
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
from src.exp_driving_videos.modules.pipe_utils.exp_driving_utils import (
    create_bg_mask,
)
from src.exp_driving_videos.modules.data_preprocessing import get_depth_maps_root

_EGO_MOTION_VERSION = 2
_EGO_MOTION_METHOD = "sparse_lk_ransac_depth"


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "06_driving_mini_ego_motion"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _is_static_label(label: Any, static_keywords: List[str]) -> bool:
    text = str(label).strip().lower()
    if not text:
        return False
    return any(keyword in text for keyword in static_keywords)


def _create_static_object_mask(
    frame_shape: Tuple[int, int],
    boxes: List[List[float]],
    labels: List[Any],
    static_keywords: List[str],
) -> np.ndarray:
    """Create mask for static object boxes (255 where static-object pixels exist)."""
    try:
        import cv2
    except ModuleNotFoundError:
        return np.zeros(frame_shape, dtype=np.uint8)

    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for box, label in zip(boxes, labels):
        if not _is_static_label(label, static_keywords):
            continue
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def _smooth_signal(values: List[float], window_size: int = 5) -> List[float]:
    """Return a centered moving-average smoothed copy of ``values``."""
    if not values:
        return []

    window_size = max(1, int(window_size))
    if window_size % 2 == 0:
        window_size += 1

    arr = np.asarray(values, dtype=np.float32)
    if window_size == 1 or arr.size == 1:
        return arr.astype(float).tolist()

    half = window_size // 2
    smoothed = np.empty_like(arr)
    for idx in range(arr.size):
        start = max(0, idx - half)
        end = min(arr.size, idx + half + 1)
        smoothed[idx] = np.mean(arr[start:end])
    return smoothed.astype(float).tolist()


def _apply_smoothing(
    frames_ego: List[Dict[str, Any]],
    smoothing_window: int = 5,
) -> List[Dict[str, Any]]:
    """Attach smoothed ego motion fields to each frame record."""
    if not frames_ego:
        return frames_ego

    vx_vals = [float(frame.get("ego_vx", 0.0)) for frame in frames_ego]
    vz_vals = [float(frame.get("ego_vz", 0.0)) for frame in frames_ego]
    yaw_vals = [float(frame.get("ego_yaw_rate", 0.0)) for frame in frames_ego]

    vx_smooth = _smooth_signal(vx_vals, window_size=smoothing_window)
    vz_smooth = _smooth_signal(vz_vals, window_size=smoothing_window)
    yaw_smooth = _smooth_signal(yaw_vals, window_size=smoothing_window)

    for idx, frame in enumerate(frames_ego):
        frame["ego_vx_smoothed"] = vx_smooth[idx]
        frame["ego_vz_smoothed"] = vz_smooth[idx]
        frame["ego_yaw_rate_smoothed"] = yaw_smooth[idx]

    return frames_ego


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _chart_panel(
    frames_ego: List[Dict[str, Any]],
    current_idx: int,
    panel_w: int,
    panel_h: int,
) -> np.ndarray:
    """
    Render a (panel_h, panel_w, 3) uint8 RGB image containing three stacked
    line charts (ego_vx, ego_vz, ego_yaw_rate) with a vertical cursor at
    current_idx.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [e["frame_index"] for e in frames_ego]
    vx_vals = [e.get("ego_vx_smoothed", e.get("ego_vx", 0.0)) for e in frames_ego]
    vz_vals = [e.get("ego_vz_smoothed", e.get("ego_vz", 0.0)) for e in frames_ego]
    yaw_vals = [e.get("ego_yaw_rate_smoothed", e.get("ego_yaw_rate", 0.0)) for e in frames_ego]

    dpi = 100
    fig, axes = plt.subplots(3, 1, figsize=(panel_w / dpi, panel_h / dpi), dpi=dpi)
    fig.patch.set_facecolor("#1a1a2e")

    signals: List[Tuple[str, List[float], str]] = [
        ("ego_vx smooth (lateral)", vx_vals, "#e94560"),
        ("ego_vz smooth (forward)", vz_vals, "#0f9b8e"),
        ("yaw_rate smooth", yaw_vals, "#f5a623"),
    ]

    cursor_x = xs[current_idx] if current_idx < len(xs) else (xs[-1] if xs else 0)

    for ax, (title, vals, color) in zip(axes, signals):
        ax.set_facecolor("#16213e")
        ax.plot(xs, vals, color=color, linewidth=1.2, alpha=0.9)
        ax.axvline(x=cursor_x, color="white", linewidth=1.0, linestyle="--", alpha=0.7)
        # Highlight current value
        if current_idx < len(vals):
            ax.scatter([cursor_x], [vals[current_idx]], color="white", s=20, zorder=5)
        ax.set_title(title, color="white", fontsize=7, pad=2)
        ax.tick_params(colors="gray", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.yaxis.label.set_color("gray")

    plt.tight_layout(pad=0.5)
    fig.canvas.draw()
    # Use RGBA canvas buffer for broad Matplotlib compatibility.
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = rgba[:, :, :3]
    plt.close(fig)

    # Resize to exact target size
    import cv2
    return cv2.resize(buf, (panel_w, panel_h), interpolation=cv2.INTER_AREA)


def _render_ego_motion_video(
    video_id: str,
    frames_ego: List[Dict[str, Any]],
    output_path: Path,
    fps: float = 10.0,
    chart_width_ratio: float = 0.45,
) -> Optional[str]:
    """
    Render a side-by-side MP4:
      left  — original video frame
      right — stacked line charts (ego_vx, ego_vz, yaw_rate) with cursor

    Returns the path string on success, None on failure.
    """
    import cv2

    if not frames_ego:
        return None

    # Determine frame size from first available image
    frame_h, frame_w = None, None
    for fe in frames_ego:
        img = cv2.imread(fe.get("image_path", ""))
        if img is not None:
            frame_h, frame_w = img.shape[:2]
            break
    if frame_h is None:
        return None

    chart_w = int(frame_w * chart_width_ratio / (1 - chart_width_ratio))
    total_w = frame_w + chart_w
    total_h = frame_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (total_w, total_h),
    )
    if not writer.isOpened():
        return None

    try:
        for idx, fe in enumerate(frames_ego):
            # Left panel: original frame (BGR)
            img = cv2.imread(fe.get("image_path", ""))
            if img is None:
                img = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (frame_w, frame_h))

            # Overlay current ego values on frame
            vx = fe.get("ego_vx_smoothed", fe.get("ego_vx", 0.0))
            vz = fe.get("ego_vz_smoothed", fe.get("ego_vz", 0.0))
            yaw = fe.get("ego_yaw_rate_smoothed", fe.get("ego_yaw_rate", 0.0))
            overlay_lines = [
                f"frame {fe.get('frame_index', idx):04d}",
                f"smooth vx={vx:+.3f}  vz={vz:.3f}  yaw={yaw:+.4f}",
            ]
            y_off = 24
            for line in overlay_lines:
                cv2.putText(
                    img, line, (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
                )
                y_off += 22

            # Right panel: matplotlib charts (RGB → BGR)
            chart_rgb = _chart_panel(frames_ego, idx, chart_w, total_h)
            chart_bgr = cv2.cvtColor(chart_rgb, cv2.COLOR_RGB2BGR)

            combined = np.concatenate([img, chart_bgr], axis=1)
            writer.write(combined)
    finally:
        writer.release()

    return str(output_path)


def _load_depth_map(depth_map_path: Path) -> Optional[np.ndarray]:
    """Load a .npz depth map and return the array, or None if missing."""
    if not depth_map_path.exists():
        return None
    try:
        data = np.load(depth_map_path)
        key = "depth" if "depth" in data else list(data.keys())[0]
        return data[key].astype(np.float32)
    except Exception as e:
        print(f"    [warn] Could not load depth map {depth_map_path}: {e}")
        return None


def _load_frame_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image as RGB numpy array, or None on failure."""
    try:
        import cv2
        bgr = cv2.imread(image_path)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"    [warn] Could not load frame {image_path}: {e}")
        return None


def _combine_background_masks(
    frame_shape: Tuple[int, int],
    prev_boxes: List[List[float]],
    curr_boxes: List[List[float]],
) -> np.ndarray:
    h, w = frame_shape
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    prev_mask = create_bg_mask(dummy, prev_boxes, [], [])
    curr_mask = create_bg_mask(dummy, curr_boxes, [], [])
    return np.minimum(prev_mask, curr_mask)


def _track_sparse_points(
    img_prev: np.ndarray,
    img_curr: np.ndarray,
    mask: np.ndarray,
    max_corners: int = 600,
    quality_level: float = 0.01,
    min_distance: int = 7,
    fb_error_threshold: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    try:
        import cv2
    except ModuleNotFoundError:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), {"num_features": 0, "num_valid_tracks": 0, "num_inliers": 0}

    gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_RGB2GRAY)
    gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY)
    features = cv2.goodFeaturesToTrack(
        gray_prev,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7,
        mask=mask,
    )
    if features is None or len(features) < 8:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), {"num_features": 0, "num_valid_tracks": 0, "num_inliers": 0}

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_prev,
        gray_curr,
        features,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray_curr,
        gray_prev,
        next_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    prev_pts = features.reshape(-1, 2)
    curr_pts = next_pts.reshape(-1, 2)
    back_pts = back_pts.reshape(-1, 2)
    valid = (status.reshape(-1) == 1) & (back_status.reshape(-1) == 1)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), {"num_features": int(len(features)), "num_valid_tracks": 0, "num_inliers": 0}

    fb_error = np.linalg.norm(prev_pts - back_pts, axis=1)
    valid &= np.isfinite(fb_error) & (fb_error <= fb_error_threshold)
    h, w = gray_curr.shape[:2]
    valid &= np.isfinite(curr_pts).all(axis=1)
    valid &= (curr_pts[:, 0] >= 0) & (curr_pts[:, 0] < w) & (curr_pts[:, 1] >= 0) & (curr_pts[:, 1] < h)
    prev_pts = prev_pts[valid].astype(np.float32)
    curr_pts = curr_pts[valid].astype(np.float32)
    if len(prev_pts) < 8:
        return prev_pts, curr_pts, {"num_features": int(len(features)), "num_valid_tracks": int(len(prev_pts)), "num_inliers": 0}

    affine, inliers = cv2.estimateAffinePartial2D(
        prev_pts,
        curr_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=2000,
        confidence=0.99,
    )
    if affine is not None and inliers is not None and int(inliers.sum()) >= 6:
        keep = inliers.reshape(-1).astype(bool)
        prev_pts = prev_pts[keep]
        curr_pts = curr_pts[keep]
    return prev_pts, curr_pts, {
        "num_features": int(len(features)),
        "num_valid_tracks": int(valid.sum()),
        "num_inliers": int(len(prev_pts)),
    }


def _estimate_sparse_ego_motion(
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    depth_map: np.ndarray,
    frame_shape: Tuple[int, int],
    focal_length: float = 1000.0,
    angle_tol_deg: float = 35.0,
    min_flow_px: float = 0.5,
) -> Dict[str, Any]:
    if len(prev_pts) < 5 or len(curr_pts) < 5:
        return {"vx": 0.0, "vz": 0.0, "yaw_rate": 0.0, "success": False, "num_motion_points": 0}

    depth = np.asarray(depth_map, dtype=np.float32)
    h, w = frame_shape
    sample_pts = np.round(curr_pts).astype(int)
    sample_pts[:, 0] = np.clip(sample_pts[:, 0], 0, w - 1)
    sample_pts[:, 1] = np.clip(sample_pts[:, 1], 0, h - 1)
    Z = depth[sample_pts[:, 1], sample_pts[:, 0]]

    dx = (curr_pts[:, 0] - prev_pts[:, 0]).astype(np.float32)
    dy = (curr_pts[:, 1] - prev_pts[:, 1]).astype(np.float32)
    valid = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(Z) & (Z > 0)
    if not np.any(valid):
        return {"vx": 0.0, "vz": 0.0, "yaw_rate": 0.0, "success": False, "num_motion_points": 0}

    xx = curr_pts[valid, 0].astype(np.float32)
    yy = curr_pts[valid, 1].astype(np.float32)
    dx = dx[valid]
    dy = dy[valid]
    Z = Z[valid].astype(np.float32)

    mag = np.hypot(dx, dy)
    moving = mag > min_flow_px
    if moving.sum() < 5:
        return {"vx": 0.0, "vz": 0.0, "yaw_rate": 0.0, "success": False, "num_motion_points": int(moving.sum())}

    xx = xx[moving]
    yy = yy[moving]
    dx = dx[moving]
    dy = dy[moving]
    Z = Z[moving]
    angles = np.arctan2(dy, dx)
    theta_consensus = np.arctan2(np.median(np.sin(angles)), np.median(np.cos(angles)))
    angle_diff = (angles - theta_consensus + np.pi) % (2.0 * np.pi) - np.pi
    inliers = np.abs(angle_diff) < np.deg2rad(angle_tol_deg)
    if inliers.sum() < max(5, int(0.05 * len(dx))):
        inliers = np.ones(len(dx), dtype=bool)

    dx_in = dx[inliers]
    dy_in = dy[inliers]
    Z_in = Z[inliers]
    xx_in = xx[inliers]
    yy_in = yy[inliers]

    vx_initial = float(np.median(dx_in * Z_in))
    dx_lat_residual = dx_in - (vx_initial / np.maximum(Z_in, 1e-3))
    yaw_rate = float(-np.median(dx_lat_residual) / focal_length)
    dx_no_yaw = dx_in + (focal_length * yaw_rate)
    vx = float(np.median(dx_no_yaw * Z_in))
    dx_forward_residual = dx_no_yaw - (vx / np.maximum(Z_in, 1e-3))
    vz_mag = float(np.median(np.hypot(dx_forward_residual, dy_in) * Z_in))

    cx = 0.5 * float(w - 1)
    cy = 0.5 * float(h - 1)
    radial_x = xx_in - cx
    radial_y = yy_in - cy
    radial_norm = np.hypot(radial_x, radial_y)
    radial_valid = radial_norm > 1e-3
    if np.any(radial_valid):
        radial_proj = (
            (radial_x[radial_valid] * dx_forward_residual[radial_valid])
            + (radial_y[radial_valid] * dy_in[radial_valid])
        ) / radial_norm[radial_valid]
        radial_median = float(np.median(radial_proj))
    else:
        radial_median = float(np.median(dy_in))
    if radial_median > 1e-4:
        vz_sign = 1.0
    elif radial_median < -1e-4:
        vz_sign = -1.0
    else:
        vz_sign = 1.0 if float(np.median(dy_in)) >= 0.0 else -1.0

    return {
        "vx": vx,
        "vz": float(vz_sign * vz_mag),
        "yaw_rate": yaw_rate,
        "success": True,
        "num_motion_points": int(len(dx_in)),
    }


def process_video(
    video_result: Dict[str, Any],
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
    smoothing_window: int = 5,
    static_adjust_cfg: Optional[Dict[str, Any]] = None,
    render_video: bool = True,
    flow_device: Optional[str] = None,
) -> Dict[str, Any]:
    """Estimate ego motion for a single video and cache the result."""
    video_id = video_result["video_id"]
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ego_motion.json"

    if force_recompute:
        print(f"  [recompute] {video_id} - rebuilding {out_file}")
    elif not out_file.exists():
        print(f"  [build] {video_id} - missing {out_file}; computing")

    cfg = static_adjust_cfg or {}
    static_adjust_enabled = bool(cfg.get("enabled", True))
    static_keywords = [
        str(v).strip().lower()
        for v in cfg.get(
            "static_object_keywords",
            ["building", "traffic light"],
        )
    ]
    blend_weight = float(cfg.get("blend_weight", 0.7))
    blend_weight = max(0.0, min(1.0, blend_weight))
    min_static_pixels = int(cfg.get("min_static_pixels", 300))

    if not force_recompute and out_file.exists():
        print(f"  [cache] {video_id} - loading {out_file.name}")
        with out_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)

        frames_ego = cached.get("frames", [])
        cache_version_ok = int(cached.get("version", 0)) == _EGO_MOTION_VERSION
        cache_method_ok = str(cached.get("estimation_method", "")) == _EGO_MOTION_METHOD
        has_smoothed = bool(frames_ego) and all(
            "ego_vx_smoothed" in frame
            and "ego_vz_smoothed" in frame
            and "ego_yaw_rate_smoothed" in frame
            for frame in frames_ego
        )
        has_static_adjusted = bool(frames_ego) and all(
            "static_adjustment_used" in frame
            for frame in frames_ego
        )
        vis_path = out_dir / "ego_motion_vis.mp4"
        cache_ok = cache_version_ok and cache_method_ok and has_smoothed and (
            (not render_video) or vis_path.exists()
        ) and (
            (not static_adjust_enabled) or has_static_adjusted
        )
        if cache_ok:
            print(f"  [cache-hit] {video_id} - using {out_file}")
            return cached
        if not cache_version_ok or not cache_method_ok:
            print(f"  [cache] {video_id} - cache is stale; recomputing with {_EGO_MOTION_METHOD}")
        elif static_adjust_enabled and not has_static_adjusted:
            print(f"  [cache] {video_id} - cache is stale; recomputing with static-object adjustment")
        elif render_video and not vis_path.exists():
            print(f"  [cache] {video_id} - cache visualization is missing; rerendering")

    frames: List[Dict[str, Any]] = video_result.get("frames", [])
    depth_root = get_depth_maps_root() / video_id

    per_frame_ego: List[Dict[str, Any]] = []

    for idx in range(len(frames)):
        frame_info = frames[idx]
        image_path = frame_info.get("image_path", "")
        frame_index = frame_info.get("frame_index", idx)

        # Build ego motion entry; first frame has no previous frame → skip flow
        entry: Dict[str, Any] = {
            "frame_index": frame_index,
            "image_path": image_path,
            "ego_vx": 0.0,
            "ego_vz": 0.0,
            "ego_yaw_rate": 0.0,
            "static_adjustment_used": False,
            "num_static_pixels": 0,
            "has_ego_motion": False,
        }

        if idx == 0:
            per_frame_ego.append(entry)
            continue

        prev_frame_info = frames[idx - 1]
        prev_image_path = prev_frame_info.get("image_path", "")

        # Load frame images
        img_prev = _load_frame_image(prev_image_path)
        img_curr = _load_frame_image(image_path)
        if img_prev is None or img_curr is None:
            per_frame_ego.append(entry)
            continue

        # Build background mask from current frame's object bounding boxes
        boxes = frame_info.get("boxes", [])
        labels = frame_info.get("labels", [])
        track_ids = frame_info.get("track_ids", [])
        prev_boxes = prev_frame_info.get("boxes", [])
        bg_mask = _combine_background_masks(img_curr.shape[:2], prev_boxes, boxes)

        # Load depth map for current frame (used for metric-scale estimation)
        depth_file = depth_root / f"{Path(image_path).stem}_depth.npz"
        depth_map = _load_depth_map(depth_file)

        if depth_map is None:
            # Fallback: uniform depth — still gives direction but no metric scale
            depth_map = np.ones(img_curr.shape[:2], dtype=np.float32)

        try:
            prev_pts, curr_pts, point_stats = _track_sparse_points(img_prev, img_curr, bg_mask)
        except Exception as e:
            print(f"    [warn] {video_id} frame {frame_index}: sparse tracking failed: {e}")
            per_frame_ego.append(entry)
            continue
        motion = _estimate_sparse_ego_motion(prev_pts, curr_pts, depth_map, img_curr.shape[:2])
        vx = float(motion["vx"])
        vz = float(motion["vz"])
        yaw_rate = float(motion["yaw_rate"])
        has_ego_motion = bool(motion.get("success", False))
        entry["num_background_features"] = int(point_stats.get("num_features", 0))
        entry["num_background_tracks"] = int(point_stats.get("num_valid_tracks", 0))
        entry["num_background_inliers"] = int(point_stats.get("num_inliers", 0))
        entry["num_motion_points"] = int(motion.get("num_motion_points", 0))

        # Optional adjustment using static-object regions (e.g., building, traffic light).
        if static_adjust_enabled:
            static_mask = _create_static_object_mask(
                frame_shape=img_curr.shape[:2],
                boxes=boxes,
                labels=labels,
                static_keywords=static_keywords,
            )
            num_static_pixels = int(np.count_nonzero(static_mask == 255))
            entry["num_static_pixels"] = num_static_pixels
            if num_static_pixels >= min_static_pixels:
                static_prev_pts, static_curr_pts, static_stats = _track_sparse_points(img_prev, img_curr, static_mask)
                static_motion = _estimate_sparse_ego_motion(static_prev_pts, static_curr_pts, depth_map, img_curr.shape[:2])
                entry["num_static_features"] = int(static_stats.get("num_features", 0))
                entry["num_static_tracks"] = int(static_stats.get("num_valid_tracks", 0))
                entry["num_static_inliers"] = int(static_stats.get("num_inliers", 0))
                if static_motion.get("success", False):
                    if has_ego_motion:
                        vx = (1.0 - blend_weight) * float(vx) + blend_weight * float(static_motion["vx"])
                        vz = (1.0 - blend_weight) * float(vz) + blend_weight * float(static_motion["vz"])
                        yaw_rate = (1.0 - blend_weight) * float(yaw_rate) + blend_weight * float(static_motion["yaw_rate"])
                    else:
                        vx = float(static_motion["vx"])
                        vz = float(static_motion["vz"])
                        yaw_rate = float(static_motion["yaw_rate"])
                    has_ego_motion = True
                    entry["static_adjustment_used"] = True

        entry["ego_vx"] = float(vx)
        entry["ego_vz"] = float(vz)
        entry["ego_yaw_rate"] = float(yaw_rate)
        entry["has_ego_motion"] = has_ego_motion
        per_frame_ego.append(entry)

    per_frame_ego = _apply_smoothing(per_frame_ego, smoothing_window=smoothing_window)

    result: Dict[str, Any] = {
        "version": _EGO_MOTION_VERSION,
        "video_id": video_id,
        "estimation_method": _EGO_MOTION_METHOD,
        "num_frames": len(frames),
        "num_frames_with_ego_motion": sum(
            1 for e in per_frame_ego if e["has_ego_motion"]
        ),
        "smoothing_window": smoothing_window,
        "static_adjustment_cfg": {
            "enabled": static_adjust_enabled,
            "static_object_keywords": static_keywords,
            "blend_weight": blend_weight,
            "min_static_pixels": min_static_pixels,
        },
        "frames": per_frame_ego,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    if render_video:
        vis_path = out_dir / "ego_motion_vis.mp4"
        rendered = _render_ego_motion_video(
            video_id=video_id,
            frames_ego=per_frame_ego,
            output_path=vis_path,
        )
        if rendered:
            result["visualization_path"] = rendered
            print(f"    visualization saved → {vis_path.name}")

    print(
        f"  {video_id}: {result['num_frames']} frames, "
        f"{result['num_frames_with_ego_motion']} with ego motion estimates"
    )
    return result


def run(
    merged_results: List[Dict[str, Any]],
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
    smoothing_window: int = 5,
    static_adjust_cfg: Optional[Dict[str, Any]] = None,
    render_video: bool = True,
    flow_device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run ego motion estimation for all videos.

    Parameters
    ----------
    merged_results : List[Dict]
        Step 4 output — one dict per video with 'video_id' and 'frames' list.
    output_root : Path, optional
        Override output directory.
    force_recompute : bool
        If True, recompute even if cached result exists.
    smoothing_window : int
        Centered moving-average window size applied after raw ego estimation.
    static_adjust_cfg : Dict, optional
        Configuration for static-object-based ego motion adjustment.

    Returns
    -------
    List[Dict] with per-video ego motion signals.
    """
    out_root = output_root or get_output_root()
    ego_motion_results: List[Dict[str, Any]] = []

    for video_result in merged_results:
        result = process_video(
            video_result=video_result,
            output_root=out_root,
            force_recompute=force_recompute,
            smoothing_window=smoothing_window,
            static_adjust_cfg=static_adjust_cfg,
            render_video=render_video,
            flow_device=flow_device,
        )
        ego_motion_results.append(result)

    # Write manifest
    manifest = {
        "version": _EGO_MOTION_VERSION,
        "estimation_method": _EGO_MOTION_METHOD,
        "num_videos": len(ego_motion_results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r["num_frames"],
                "num_frames_with_ego_motion": r["num_frames_with_ego_motion"],
            }
            for r in ego_motion_results
        ],
    }
    manifest_path = out_root / "ego_motion_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Ego motion manifest written to {manifest_path}")

    return ego_motion_results
