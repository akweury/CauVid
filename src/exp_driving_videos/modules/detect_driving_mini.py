"""
YOLO-World object detection over the driving_mini dataset.

The driving_mini dataset has pre-extracted frames stored as:
    dataset/driving_mini/frames/<video_id>/frame_NNNNN.jpg

This script runs YOLO-World on every frame of every video clip and writes
per-video JSON results alongside a summary manifest.

Output layout:
    pipeline_output/driving_mini_detection/
        detection_manifest.json
        <video_id>/
            detections.json

Usage:
    python src/exp_driving_videos/detect_driving_mini.py
    python src/exp_driving_videos/detect_driving_mini.py \\
        --video 0153f03b-3b26c404 \\
        --confidence-threshold 0.25 \\
        --yolo-model yolov8s-worldv2.pt \\
        --force-recompute
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2



import config
PROJECT_ROOT = config.PROJECT_ROOT

# Reuse the shared detector classes from the nuScenes detection pipeline
from src.exp_nuScenes.detection_pipeline import (
    YOLO_WORLD_DEFAULT_CLASSES,
    DetectionResult,
    ObjectDetector,
    YOLOWorldDetector,
)

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

def get_frames_root() -> Path:
    return PROJECT_ROOT / "dataset" / "driving_mini" / "frames"


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "driving_mini_detection"
    out.mkdir(parents=True, exist_ok=True)
    return out


def list_video_ids(frames_root: Optional[Path] = None) -> List[str]:
    root = frames_root or get_frames_root()
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def list_frames(video_id: str, frames_root: Optional[Path] = None) -> List[Path]:
    root = frames_root or get_frames_root()
    return sorted((root / video_id).glob("frame_*.jpg"))


def _label_color(label: str) -> tuple[int, int, int]:
    # Stable pseudo-random color per class label.
    h = abs(hash(label)) % 360
    s = 0.75
    v = 0.95
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


def render_detection_video(
    video_id: str,
    frame_records: List[Dict[str, Any]],
    output_path: Path,
    fps: float = 10.0,
) -> Optional[str]:
    """Render one annotated mp4 for a video using detection frame records."""
    if not frame_records:
        return None

    first = cv2.imread(frame_records[0]["image_path"])
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
        for rec in frame_records:
            img = cv2.imread(rec["image_path"])
            if img is None:
                continue

            for box, score, label in zip(rec.get("boxes", []), rec.get("scores", []), rec.get("labels", [])):
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                color = _label_color(str(label))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {float(score):.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_y = max(y1 - 6, th + 6)
                cv2.rectangle(img, (x1, text_y - th - 4), (x1 + tw + 4, text_y), color, -1)
                cv2.putText(
                    img,
                    text,
                    (x1 + 2, text_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                img,
                f"{video_id} | frame {rec.get('frame_index', -1):04d}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(img)
    finally:
        writer.release()

    return str(output_path)


# ---------------------------------------------------------------------------
# Per-video detection
# ---------------------------------------------------------------------------

def process_video(
    video_id: str,
    detector: ObjectDetector,
    frames_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    detections_file = out_dir / "detections.json"
    boxed_video_file = out_dir / "detections_boxed.mp4"

    if not force_recompute and detections_file.exists():
        print(f"  [cache] {video_id} — loading {detections_file.name}")
        with detections_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if not boxed_video_file.exists() and cached.get("frames"):
            render_path = render_detection_video(
                video_id=video_id,
                frame_records=cached["frames"],
                output_path=boxed_video_file,
            )
            if render_path:
                cached["boxed_video_path"] = render_path
                with detections_file.open("w", encoding="utf-8") as fh:
                    json.dump(cached, fh, indent=2)
                print(f"  Rendered boxed video: {boxed_video_file.name}")
        return cached

    frames = list_frames(video_id, frames_root)
    print(f"\n=== {video_id} ({len(frames)} frames) ===")

    if not frames:
        result: Dict[str, Any] = {
            "video_id": video_id,
            "num_frames": 0,
            "num_detections": 0,
            "detected_classes": {},
            "frames": [],
        }
        with detections_file.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        return result

    # Batch all frames in one model.predict() call for speed
    image_paths = [str(f) for f in frames]
    results: List[DetectionResult] = detector.detect_batch(image_paths)

    frame_records: List[Dict[str, Any]] = []
    all_labels: List[str] = []

    for frame_path, det in zip(frames, results):
        all_labels.extend(det.labels)
        frame_records.append({
            "frame": frame_path.name,
            "frame_index": int(frame_path.stem.split("_")[-1]),
            **det.to_dict(),
        })

    label_counts = dict(Counter(all_labels).most_common())

    video_result: Dict[str, Any] = {
        "video_id": video_id,
        "num_frames": len(frame_records),
        "num_detections": sum(len(f["boxes"]) for f in frame_records),
        "detected_classes": label_counts,
        "frames": frame_records,
    }

    render_path = render_detection_video(
        video_id=video_id,
        frame_records=frame_records,
        output_path=boxed_video_file,
    )
    if render_path:
        video_result["boxed_video_path"] = render_path

    with detections_file.open("w", encoding="utf-8") as fh:
        json.dump(video_result, fh, indent=2)

    print(
        f"  {video_result['num_frames']} frames, "
        f"{video_result['num_detections']} detections, "
        f"{len(label_counts)} classes: "
        + ", ".join(f"{k}({v})" for k, v in list(label_counts.items())[:8])
        + ("..." if len(label_counts) > 8 else "")
    )
    if render_path:
        print(f"  Saved boxed video: {boxed_video_file}")
    return video_result


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run(
    video_ids: Optional[List[str]] = None,
    model_name: str = "yolov8s-worldv2.pt",
    classes: Optional[List[str]] = None,
    confidence_threshold: float = 0.3,
    nms_iou_threshold: float = 0.5,
    device: Optional[str] = None,
    frames_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    effective_frames_root = frames_root or get_frames_root()
    effective_output_root = output_root or get_output_root()

    all_videos = list_video_ids(effective_frames_root)
    target_videos = list(video_ids) if video_ids else all_videos

    unknown = set(target_videos) - set(all_videos)
    if unknown:
        raise ValueError(f"Unknown video IDs: {sorted(unknown)}. Available: {all_videos}")

    effective_classes = list(classes) if classes else list(YOLO_WORLD_DEFAULT_CLASSES)

    detector = YOLOWorldDetector(
        model_name=model_name,
        classes=effective_classes,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
        device=device,
    )

    print(f"Videos to process: {target_videos}")
    print(f"YOLO-World model : {model_name}")
    print(f"Vocabulary       : {len(effective_classes)} classes")

    detector.warmup()
    video_results: List[Dict[str, Any]] = []
    try:
        for video_id in target_videos:
            result = process_video(
                video_id=video_id,
                detector=detector,
                frames_root=effective_frames_root,
                output_root=effective_output_root,
                force_recompute=force_recompute,
            )
            video_results.append(result)
    finally:
        detector.teardown()

    # Aggregate class counts across all videos
    total_labels: Counter[str] = Counter()
    for r in video_results:
        total_labels.update(r["detected_classes"])

    manifest = {
        "model": model_name,
        "num_videos": len(video_results),
        "num_frames_total": sum(r["num_frames"] for r in video_results),
        "num_detections_total": sum(r["num_detections"] for r in video_results),
        "detected_classes_total": dict(total_labels.most_common()),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r["num_frames"],
                "num_detections": r["num_detections"],
                "num_classes": len(r["detected_classes"]),
            }
            for r in video_results
        ],
    }
    manifest_path = effective_output_root / "detection_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nSaved manifest to {manifest_path}")
    print(f"Total detections : {manifest['num_detections_total']} across {manifest['num_frames_total']} frames")
    print(f"Top classes      : " + ", ".join(
        f"{k}({v})" for k, v in list(manifest["detected_classes_total"].items())[:10]
    ))
    return video_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO-World detection over driving_mini frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", action="append", default=[], dest="video",
                        help="Video ID(s) to process. Repeat for multiple. Default: all.")
    parser.add_argument("--yolo-model", default="yolov8s-worldv2.pt", dest="yolo_model")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Custom class vocabulary. Default: YOLO_WORLD_DEFAULT_CLASSES.")
    parser.add_argument("--confidence-threshold", type=float, default=0.3, dest="confidence_threshold")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5, dest="nms_iou_threshold")
    parser.add_argument("--device", default=None,
                        help="Inference device, e.g. cpu, cuda:0, mps.")
    parser.add_argument("--frames-root", type=Path, default=None, dest="frames_root",
                        help="Override default frames root.")
    parser.add_argument("--output-root", type=Path, default=None, dest="output_root",
                        help="Override default output root.")
    parser.add_argument("--force-recompute", action="store_true", dest="force_recompute")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        video_ids=args.video or None,
        model_name=args.yolo_model,
        classes=args.classes,
        confidence_threshold=args.confidence_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        device=args.device,
        frames_root=args.frames_root,
        output_root=args.output_root,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
