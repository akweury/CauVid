"""
ByteTrack-based multi-object tracking over driving_mini detection results.

Takes the per-video detection records produced by detect_driving_mini.run() and
runs ByteTracker (via ultralytics) to assign persistent track IDs across frames.

Output layout:
    pipeline_output/02_driving_mini_tracking/<video_id>/
        tracks.json          — per-frame tracking results with track IDs
        tracks_manifest.json — summary manifest (written by run())

Returned structure per video
-----------------------------
{
    "video_id": str,
    "num_frames": int,
    "num_tracks": int,          # unique track IDs seen
    "frames": [
        {
            "frame": str,           # e.g. "frame_00001.jpg"
            "frame_index": int,
            "image_path": str,
            "boxes":    [[x1,y1,x2,y2], ...],
            "scores":   [float, ...],
            "labels":   [str, ...],
            "track_ids":[int, ...],
        },
        ...
    ],
}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# ---------------------------------------------------------------------------
# ByteTrack via ultralytics
# ---------------------------------------------------------------------------

try:
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.engine.results import Boxes
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ultralytics is required for ByteTrack tracking. "
        "Install with: pip install ultralytics"
    ) from exc


# ---------------------------------------------------------------------------
# Default tracker parameters (mirrors bytetrack.yaml defaults)
# ---------------------------------------------------------------------------

_DEFAULT_TRACKER_ARGS = SimpleNamespace(
    tracker_type="bytetrack",
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    track_buffer=30,   # keep lost tracks alive for up to 30 frames before dropping
    match_thresh=0.8,
    fuse_score=True,
)

# Placeholder image shape used when no image is provided to the tracker.
# ByteTracker needs orig_shape to build Boxes; we use the first available
# detection to derive it when possible, otherwise fall back to this.
_FALLBACK_SHAPE = (1080, 1920)  # (height, width)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "02_driving_mini_tracking"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Return a stable BGR color for a given track ID."""
    h = (track_id * 37) % 360
    s, v = 0.85, 0.95
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))


def render_tracking_video(
    video_id: str,
    tracked_frames: List[Dict[str, Any]],
    output_path: Path,
    fps: float = 10.0,
) -> Optional[str]:
    """Render an annotated MP4 with bounding boxes colored by track ID."""
    if not tracked_frames:
        return None

    first = cv2.imread(tracked_frames[0]["image_path"])
    if first is None:
        return None

    h, w = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        return None

    try:
        for rec in tracked_frames:
            img = cv2.imread(rec["image_path"])
            if img is None:
                continue

            for box, score, label, track_id in zip(
                rec.get("boxes", []),
                rec.get("scores", []),
                rec.get("labels", []),
                rec.get("track_ids", []),
            ):
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                color = _track_color(track_id)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                text = f"#{track_id} {label} {float(score):.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_y = max(y1 - 6, th + 6)
                cv2.rectangle(img, (x1, text_y - th - 4), (x1 + tw + 4, text_y), color, -1)
                cv2.putText(
                    img, text, (x1 + 2, text_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
                )

            cv2.putText(
                img,
                f"{video_id} | frame {rec.get('frame_index', -1):04d}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
            writer.write(img)
    finally:
        writer.release()

    return str(output_path)


def _make_tracker(frame_rate: int = 10, args: Optional[SimpleNamespace] = None) -> BYTETracker:
    """Create a fresh BYTETracker instance."""
    tracker_args = args or _DEFAULT_TRACKER_ARGS
    try:
        return BYTETracker(tracker_args, frame_rate=frame_rate)
    except TypeError as exc:
        # Newer ultralytics releases removed the frame_rate kwarg.
        if "frame_rate" not in str(exc):
            raise
        return BYTETracker(tracker_args)


def _infer_orig_shape(frame_records: List[Dict[str, Any]]) -> tuple[int, int]:
    """Try to read image dimensions from the first available frame image."""
    try:
        import cv2
        for rec in frame_records:
            img = cv2.imread(rec.get("image_path", ""))
            if img is not None:
                h, w = img.shape[:2]
                return (h, w)
    except Exception:
        pass
    return _FALLBACK_SHAPE


# ---------------------------------------------------------------------------
# Per-video tracking
# ---------------------------------------------------------------------------

def track_video(
    video_result: Dict[str, Any],
    output_root: Optional[Path] = None,
    frame_rate: int = 10,
    tracker_args: Optional[SimpleNamespace] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """Run ByteTrack on a single video's detection records.

    Args:
        video_result:   One entry from detect_driving_mini.run() output.
        output_root:    Root directory for persisting tracks.json.
        frame_rate:     Video frame rate passed to the tracker.
        tracker_args:   Override tracker hyper-parameters.
        force_recompute: Recompute even if tracks.json already exists.

    Returns:
        Tracking result dict (see module docstring).
    """
    video_id: str = video_result["video_id"]
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_file = out_dir / "tracks.json"

    if not force_recompute and tracks_file.exists():
        print(f"  [cache] {video_id} — loading {tracks_file.name}")
        with tracks_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        tracked_video_file = out_dir / "tracks_boxed.mp4"
        if not tracked_video_file.exists() and cached.get("frames"):
            render_path = render_tracking_video(
                video_id=video_id,
                tracked_frames=cached["frames"],
                output_path=tracked_video_file,
                fps=float(frame_rate),
            )
            if render_path:
                cached["tracked_video_path"] = render_path
                with tracks_file.open("w", encoding="utf-8") as fh:
                    json.dump(cached, fh, indent=2)
                print(f"  Rendered tracked video: {tracked_video_file.name}")
        return cached

    frame_records: List[Dict[str, Any]] = video_result.get("frames", [])
    print(f"\n=== Tracking {video_id} ({len(frame_records)} frames) ===")

    orig_shape = _infer_orig_shape(frame_records)
    tracker = _make_tracker(frame_rate=frame_rate, args=tracker_args)

    tracked_frames: List[Dict[str, Any]] = []
    all_track_ids: set[int] = set()

    for rec in frame_records:
        boxes_raw: List[List[float]] = rec.get("boxes", [])
        scores_raw: List[float] = rec.get("scores", [])
        labels_raw: List[str] = rec.get("labels", [])

        if boxes_raw:
            # Build (N, 6) array: [x1, y1, x2, y2, conf, cls_id]
            # BYTETracker only uses coordinates + confidence; class is preserved
            # so we can map back to label strings after tracking.
            n = len(boxes_raw)
            arr = np.zeros((n, 6), dtype=np.float32)
            for i, (box, score) in enumerate(zip(boxes_raw, scores_raw)):
                arr[i, :4] = box
                arr[i, 4] = float(score)
                arr[i, 5] = float(i)  # use original index as placeholder class

            det_boxes = Boxes(arr, orig_shape)
            track_output: np.ndarray = tracker.update(det_boxes)
            # track_output columns: [x1, y1, x2, y2, track_id, conf, cls, ...]
        else:
            # No detections — update tracker with empty input to age out tracks
            empty = np.empty((0, 6), dtype=np.float32)
            det_boxes = Boxes(empty, orig_shape) if False else None  # skip empty update
            tracker.update(
                Boxes(np.empty((0, 6), dtype=np.float32), orig_shape)
            )
            track_output = np.empty((0, 7), dtype=np.float32)

        # Build per-frame output, preserving original labels
        frame_boxes: List[List[float]] = []
        frame_scores: List[float] = []
        frame_labels: List[str] = []
        frame_track_ids: List[int] = []

        if track_output is not None and len(track_output):
            for row in track_output:
                # ultralytics BYTETracker returns [x1,y1,x2,y2, track_id, conf, cls]
                x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                track_id = int(row[4])
                conf = float(row[5])
                cls_idx = int(round(float(row[6])))

                # Recover original label by class index (we stored original index as cls)
                label = labels_raw[cls_idx] if cls_idx < len(labels_raw) else "unknown"

                frame_boxes.append([x1, y1, x2, y2])
                frame_scores.append(conf)
                frame_labels.append(label)
                frame_track_ids.append(track_id)
                all_track_ids.add(track_id)

        tracked_frames.append({
            "frame": rec.get("frame", ""),
            "frame_index": rec.get("frame_index", -1),
            "image_path": rec.get("image_path", ""),
            "boxes": frame_boxes,
            "scores": frame_scores,
            "labels": frame_labels,
            "track_ids": frame_track_ids,
        })

    result: Dict[str, Any] = {
        "video_id": video_id,
        "num_frames": len(tracked_frames),
        "num_tracks": len(all_track_ids),
        "frames": tracked_frames,
    }

    with tracks_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    # Render annotated video with track IDs
    tracked_video_file = out_dir / "tracks_boxed.mp4"
    render_path = render_tracking_video(
        video_id=video_id,
        tracked_frames=tracked_frames,
        output_path=tracked_video_file,
        fps=float(frame_rate),
    )
    if render_path:
        result["tracked_video_path"] = render_path
        with tracks_file.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"  Saved tracked video: {tracked_video_file.name}")

    print(
        f"  {result['num_frames']} frames tracked, "
        f"{result['num_tracks']} unique tracks"
    )
    return result


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(
    detection_results: List[Dict[str, Any]],
    output_root: Optional[Path] = None,
    frame_rate: int = 10,
    tracker_args: Optional[SimpleNamespace] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    """Run ByteTrack over all videos in detection_results.

    Args:
        detection_results: Output of detect_driving_mini.run() — a list of
                           per-video detection dicts.
        output_root:       Override default output root.
        frame_rate:        Frame rate of the source videos.
        tracker_args:      Override default ByteTrack hyper-parameters.
        force_recompute:   Ignore cached tracks.json files and recompute.

    Returns:
        List of per-video tracking result dicts.
    """
    effective_output_root = output_root or get_output_root()

    print(f"Videos to track: {[r['video_id'] for r in detection_results]}")

    tracking_results: List[Dict[str, Any]] = []
    for video_result in detection_results:
        result = track_video(
            video_result=video_result,
            output_root=effective_output_root,
            frame_rate=frame_rate,
            tracker_args=tracker_args,
            force_recompute=force_recompute,
        )
        tracking_results.append(result)

    # Write summary manifest
    manifest = {
        "num_videos": len(tracking_results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r["num_frames"],
                "num_tracks": r["num_tracks"],
            }
            for r in tracking_results
        ],
    }
    manifest_path = effective_output_root / "tracks_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    total_tracks = sum(r["num_tracks"] for r in tracking_results)
    print(f"\nSaved tracking manifest to {manifest_path}")
    print(f"Total unique tracks across all videos: {total_tracks}")
    return tracking_results
