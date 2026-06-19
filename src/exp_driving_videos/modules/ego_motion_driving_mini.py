"""
Estimate ego motion for driving_mini videos.

For each consecutive frame pair:
  1. Compute dense optical flow (RAFT-Small).
  2. Build a background mask by blacking out detected/GT object bounding boxes.
  3. ego_flow = flow[bg_mask]  — use only background pixels.
  4. Estimate (vx, vz) lateral and forward ego speed + yaw rate using
     depth-weighted background flow via estimate_ego_motion / estimate_ego_rotation.

Consumes:
  - Step 4 merged annotations (image_path, boxes per frame)
  - Depth maps from pipeline_output/05_driving_mini_3d_positions (or depth_maps root)

Output layout:
    pipeline_output/06_driving_mini_ego_motion/
        ego_motion_manifest.json
        <video_id>/
            ego_motion.json          — raw + smoothed per-frame ego motion signals
            ego_motion_vis.mp4       — side-by-side: original frame + smoothed charts
            optical_flows/
                flow_NNNNN.npy       — cached RAFT flow arrays (H, W, 2)
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
    compute_optical_flow,
    create_bg_mask,
    estimate_ego_motion,
    estimate_ego_rotation,
)
from src.exp_driving_videos.modules.data_preprocessing import get_depth_maps_root


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


def process_video(
    video_result: Dict[str, Any],
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
    smoothing_window: int = 5,
    static_adjust_cfg: Optional[Dict[str, Any]] = None,
    render_video: bool = True,
) -> Dict[str, Any]:
    """Estimate ego motion for a single video and cache the result."""
    video_id = video_result["video_id"]
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ego_motion.json"

    if force_recompute:
        print(f"  [recompute] {video_id} - rebuilding {out_file.name}")
    elif not out_file.exists():
        print(f"  [build] {video_id} - missing {out_file.name}; computing")

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
        cache_ok = has_smoothed and (
            (not render_video) or vis_path.exists()
        ) and (
            (not static_adjust_enabled) or has_static_adjusted
        )
        if cache_ok:
            return cached
        if static_adjust_enabled and not has_static_adjusted:
            print(f"  [cache] {video_id} - cache is stale; recomputing with static-object adjustment")
        elif render_video and not vis_path.exists():
            print(f"  [cache] {video_id} - cache visualization is missing; rerendering")

    frames: List[Dict[str, Any]] = video_result.get("frames", [])
    depth_root = get_depth_maps_root() / video_id
    flow_cache_dir = out_dir / "optical_flows"
    flow_cache_dir.mkdir(parents=True, exist_ok=True)

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
        bg_mask = create_bg_mask(img_curr, boxes, labels, track_ids)

        # Load depth map for current frame (used for metric-scale estimation)
        depth_file = depth_root / f"{Path(image_path).stem}_depth.npz"
        depth_map = _load_depth_map(depth_file)

        if depth_map is None:
            # Fallback: uniform depth — still gives direction but no metric scale
            depth_map = np.ones(img_curr.shape[:2], dtype=np.float32)

        # --- Compute optical flow (RAFT-Small), with per-frame cache ---
        flow_cache_file = flow_cache_dir / f"flow_{frame_index:05d}.npy"
        if not force_recompute and flow_cache_file.exists():
            flow = np.load(str(flow_cache_file))
        else:
            try:
                flow = compute_optical_flow(img_prev, img_curr)
                np.save(str(flow_cache_file), flow)
            except Exception as e:
                print(f"    [warn] {video_id} frame {frame_index}: optical flow failed: {e}")
                per_frame_ego.append(entry)
                continue

        # ego_flow = flow[bg_mask] uses background pixels only
        # estimate_ego_motion internally applies bg_mask == 255 filtering
        try:
            vx, vz, yaw_rate = estimate_ego_motion(
                flow, bg_mask, depth_map, return_yaw=True
            )
        except TypeError:
            # Older signature without return_yaw
            vx, vz = estimate_ego_motion(flow, bg_mask, depth_map)
            yaw_rate = float(estimate_ego_rotation(flow, bg_mask, depth_map))

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
                try:
                    sx, sz, syaw = estimate_ego_motion(
                        flow,
                        static_mask,
                        depth_map,
                        return_yaw=True,
                    )
                except TypeError:
                    sx, sz = estimate_ego_motion(flow, static_mask, depth_map)
                    syaw = float(estimate_ego_rotation(flow, static_mask, depth_map))

                vx = (1.0 - blend_weight) * float(vx) + blend_weight * float(sx)
                vz = (1.0 - blend_weight) * float(vz) + blend_weight * float(sz)
                yaw_rate = (1.0 - blend_weight) * float(yaw_rate) + blend_weight * float(syaw)
                entry["static_adjustment_used"] = True

        entry["ego_vx"] = float(vx)
        entry["ego_vz"] = float(vz)
        entry["ego_yaw_rate"] = float(yaw_rate)
        entry["has_ego_motion"] = True
        per_frame_ego.append(entry)

    per_frame_ego = _apply_smoothing(per_frame_ego, smoothing_window=smoothing_window)

    result: Dict[str, Any] = {
        "video_id": video_id,
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
        )
        ego_motion_results.append(result)

    # Write manifest
    manifest = {
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
