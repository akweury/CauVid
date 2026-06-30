"""
Merge ByteTrack results with dataset ground-truth annotations for driving_mini.

Rule:
- If an object exists in both GT and detected/tracked results for the same frame,
  keep the GT annotation.
- Keep unmatched detected/tracked objects to maximize object coverage.

Output layout:
    pipeline_output/04_driving_mini_merged_annotations/
        merged_annotations_manifest.json
        <video_id>/
            merged_annotations.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency guard
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


_MERGED_ANNOTATIONS_VERSION = 3


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "04_driving_mini_merged_annotations"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _box_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return (inter / denom) if denom > 0.0 else 0.0


def _find_best_detection_match(
    gt_box: List[float],
    gt_label: str,
    det_boxes: List[List[float]],
    det_labels: List[str],
    available_det_idx: set[int],
    iou_threshold: float,
    require_label_match: bool,
) -> Optional[int]:
    best_idx: Optional[int] = None
    best_iou = 0.0

    for det_idx in available_det_idx:
        if require_label_match and det_idx < len(det_labels) and det_labels[det_idx] != gt_label:
            continue
        iou = _box_iou(gt_box, det_boxes[det_idx])
        if iou >= iou_threshold and iou > best_iou:
            best_iou = iou
            best_idx = det_idx

    return best_idx


def _to_frame_map(frames: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    frame_map: Dict[int, Dict[str, Any]] = {}
    for frame in frames:
        frame_map[int(frame.get("frame_index", -1))] = frame
    return frame_map


def _selected_candidate_summary_by_track(tracked_video: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    track_summaries = (
        tracked_video.get("candidate_tracks", {})
        .get("selected_candidate_tracks", {})
        .get("track_summaries", [])
    )
    return {
        int(row.get("track_id", -1)): dict(row)
        for row in track_summaries
        if int(row.get("track_id", -1)) >= 0
    }


def _selected_candidate_objects_by_frame_from_tracking(
    tracked_video: Dict[str, Any],
) -> Dict[int, List[Dict[str, Any]]]:
    frame_map: Dict[int, List[Dict[str, Any]]] = {}
    summary_by_track = _selected_candidate_summary_by_track(tracked_video)
    candidate_frames = (
        tracked_video.get("candidate_tracks", {})
        .get("selected_candidate_tracks", {})
        .get("frames", [])
    )
    for frame in candidate_frames:
        frame_index = int(frame.get("frame_index", -1))
        if frame_index < 0:
            continue
        boxes = list(frame.get("boxes", []))
        scores = list(frame.get("scores", []))
        labels = list(frame.get("labels", []))
        track_ids = list(frame.get("track_ids", []))
        detection_ids = list(frame.get("detection_ids", []))
        candidate_sources = list(frame.get("candidate_sources", []))
        prior_relevance_scores = list(frame.get("prior_relevance_scores", []))
        matched_prior_ids = list(frame.get("matched_prior_ids", []))
        candidate_objects = frame_map.setdefault(frame_index, [])
        for index, box in enumerate(boxes):
            candidate_track_id = int(track_ids[index]) if index < len(track_ids) else -1
            track_summary = dict(summary_by_track.get(candidate_track_id, {}))
            candidate_objects.append(
                {
                    "bbox": list(box),
                    "score": float(scores[index]) if index < len(scores) else 0.0,
                    "label": str(labels[index]) if index < len(labels) else "unknown",
                    "accepted": False,
                    "source_type": "selected_candidate_track",
                    "candidate_track_id": candidate_track_id,
                    "track_id": candidate_track_id,
                    "frame_detection_id": str(detection_ids[index]) if index < len(detection_ids) else "",
                    "source_detection_ids": list(track_summary.get("source_detection_ids", [])),
                    "candidate_source": str(candidate_sources[index]) if index < len(candidate_sources) else "",
                    "prior_metadata": {
                        "matched_prior_ids": list(matched_prior_ids[index] if index < len(matched_prior_ids) else []),
                        "track_matched_prior_ids": list(track_summary.get("matched_prior_ids", [])),
                        "matched_prior_id_counts": dict(track_summary.get("matched_prior_id_counts", {})),
                        "prior_relevance_score": float(
                            prior_relevance_scores[index] if index < len(prior_relevance_scores) else 0.0
                        ),
                        "prior_relevance_mean": float(track_summary.get("prior_relevance_mean", 0.0)),
                        "prior_relevance_max": float(track_summary.get("prior_relevance_max", 0.0)),
                        "prior_relevance_min": float(track_summary.get("prior_relevance_min", 0.0)),
                    },
                    "score_breakdown": dict(track_summary.get("score_breakdown", {})),
                    "track_quality": {
                        "mean_score": float(track_summary.get("mean_score", 0.0)),
                        "max_score": float(track_summary.get("max_score", 0.0)),
                        "track_length": int(track_summary.get("track_length", 0)),
                        "temporal_consistency": float(track_summary.get("temporal_consistency", 0.0)),
                        "selection_score": float(track_summary.get("selection_score", 0.0)),
                    },
                }
            )
    return frame_map


def _candidate_objects_by_frame(
    gt_video: Optional[Dict[str, Any]],
    tracked_video: Dict[str, Any],
) -> Dict[int, List[Dict[str, Any]]]:
    if isinstance(gt_video, dict):
        frames = list(gt_video.get("frames", []))
        if any("candidate_objects" in frame for frame in frames):
            return {
                int(frame.get("frame_index", -1)): list(frame.get("candidate_objects", []))
                for frame in frames
                if int(frame.get("frame_index", -1)) >= 0
            }
    return _selected_candidate_objects_by_frame_from_tracking(tracked_video)


def _accepted_object_entry(
    bbox: List[float],
    score: float,
    label: str,
    track_id: int,
    source: str,
) -> Dict[str, Any]:
    return {
        "bbox": list(bbox),
        "score": float(score),
        "label": str(label),
        "track_id": int(track_id),
        "accepted": True,
        "source": str(source),
        "source_type": "ground_truth_annotation" if source == "gt" else "accepted_track",
        "is_ground_truth": bool(source == "gt"),
    }


def _tracked_video_as_merged_result(
    tracked_video: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_objects_by_frame = _selected_candidate_objects_by_frame_from_tracking(tracked_video)
    frames_out: List[Dict[str, Any]] = []
    candidate_track_ids: set[int] = set()
    candidate_objects_total = 0
    for frame in list(tracked_video.get("frames", [])):
        frame_index = int(frame.get("frame_index", -1))
        boxes = list(frame.get("boxes", []))
        scores = list(frame.get("scores", []))
        labels = list(frame.get("labels", []))
        track_ids = list(frame.get("track_ids", []))
        sources = ["det"] * len(boxes)
        objects = [
            _accepted_object_entry(
                bbox=list(boxes[index]),
                score=float(scores[index]) if index < len(scores) else 0.0,
                label=str(labels[index]) if index < len(labels) else "unknown",
                track_id=int(track_ids[index]) if index < len(track_ids) else -1,
                source="det",
            )
            for index in range(len(boxes))
        ]
        candidate_objects = list(candidate_objects_by_frame.get(frame_index, []))
        for candidate_object in candidate_objects:
            candidate_track_id = int(candidate_object.get("candidate_track_id", -1))
            if candidate_track_id >= 0:
                candidate_track_ids.add(candidate_track_id)
        candidate_objects_total += len(candidate_objects)
        frames_out.append(
            {
                "frame": frame.get("frame", ""),
                "frame_index": frame_index,
                "image_path": frame.get("image_path", ""),
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "track_ids": track_ids,
                "sources": sources,
                "objects": objects,
                "candidate_objects": candidate_objects,
            }
        )
    return {
        "version": _MERGED_ANNOTATIONS_VERSION,
        "video_id": tracked_video["video_id"],
        "num_frames": len(frames_out),
        "num_tracks": int(tracked_video.get("num_tracks", 0)),
        "num_objects": sum(len(frame.get("objects", [])) for frame in frames_out),
        "num_candidate_tracks": len(candidate_track_ids),
        "num_candidate_objects": candidate_objects_total,
        "frames": frames_out,
        "tracked_only": True,
    }


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Return a stable BGR color for a merged track ID."""
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


def render_merged_video(
    video_id: str,
    merged_frames: List[Dict[str, Any]],
    output_path: Path,
    fps: float = 10.0,
) -> Optional[str]:
    """Render merged tracking results to MP4.

    GT boxes are drawn as solid rectangles, and carried detections are drawn
    with dashed-like corner markers so source is visually distinguishable.
    """
    if cv2 is None:
        return None
    if not merged_frames:
        return None

    first = cv2.imread(merged_frames[0].get("image_path", ""))
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
        for rec in merged_frames:
            img = cv2.imread(rec.get("image_path", ""))
            if img is None:
                continue

            for box, score, label, track_id, source in zip(
                rec.get("boxes", []),
                rec.get("scores", []),
                rec.get("labels", []),
                rec.get("track_ids", []),
                rec.get("sources", []),
            ):
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                color = _track_color(int(track_id))
                if source == "gt":
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                else:
                    # Corner-only rectangle for carried detections.
                    seg = 10
                    cv2.line(img, (x1, y1), (x1 + seg, y1), color, 2)
                    cv2.line(img, (x1, y1), (x1, y1 + seg), color, 2)
                    cv2.line(img, (x2, y1), (x2 - seg, y1), color, 2)
                    cv2.line(img, (x2, y1), (x2, y1 + seg), color, 2)
                    cv2.line(img, (x1, y2), (x1 + seg, y2), color, 2)
                    cv2.line(img, (x1, y2), (x1, y2 - seg), color, 2)
                    cv2.line(img, (x2, y2), (x2 - seg, y2), color, 2)
                    cv2.line(img, (x2, y2), (x2, y2 - seg), color, 2)

                text = f"#{int(track_id)} {label} {float(score):.2f} [{source}]"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                text_y = max(y1 - 6, th + 6)
                cv2.rectangle(img, (x1, text_y - th - 4), (x1 + tw + 4, text_y), color, -1)
                cv2.putText(
                    img,
                    text,
                    (x1 + 2, text_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
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


def merge_video(
    tracked_video: Dict[str, Any],
    gt_video: Dict[str, Any],
    output_root: Optional[Path] = None,
    iou_threshold: float = 0.5,
    require_label_match: bool = True,
    force_recompute: bool = False,
    render_video: bool = True,
) -> Dict[str, Any]:
    video_id = tracked_video["video_id"]
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "merged_annotations.json"
    merged_video_file = out_dir / "merged_tracking_boxed.mp4"

    if not force_recompute and out_file.exists():
        print(f"  [cache] {video_id} - loading {out_file.name}")
        with out_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _MERGED_ANNOTATIONS_VERSION:
            if render_video and not merged_video_file.exists() and cached.get("frames"):
                render_path = render_merged_video(
                    video_id=video_id,
                    merged_frames=cached["frames"],
                    output_path=merged_video_file,
                )
                if render_path:
                    cached["merged_video_path"] = render_path
                    with out_file.open("w", encoding="utf-8") as fh:
                        json.dump(cached, fh, indent=2)
                    print(f"  Rendered merged video: {merged_video_file.name}")
            return cached

    tracked_frames = _to_frame_map(tracked_video.get("frames", []))
    gt_frames = _to_frame_map(gt_video.get("frames", []))
    candidate_objects_by_frame = _candidate_objects_by_frame(gt_video, tracked_video)
    frame_indices = sorted(set(tracked_frames.keys()) | set(gt_frames.keys()))

    merged_frames: List[Dict[str, Any]] = []

    max_gt_track_id = -1
    for fr in gt_video.get("frames", []):
        for tid in fr.get("track_ids", []):
            if isinstance(tid, int) and tid > max_gt_track_id:
                max_gt_track_id = tid

    next_det_track_id = max_gt_track_id + 1
    det_track_id_map: Dict[int, int] = {}

    overlap_count = 0
    carried_det_count = 0
    all_track_ids: set[int] = set()
    candidate_track_ids: set[int] = set()
    candidate_objects_total = 0

    for frame_index in frame_indices:
        gt_frame = gt_frames.get(frame_index, {})
        det_frame = tracked_frames.get(frame_index, {})

        gt_boxes = gt_frame.get("boxes", [])
        gt_scores = gt_frame.get("scores", [])
        gt_labels = gt_frame.get("labels", [])
        gt_track_ids = gt_frame.get("track_ids", [])

        det_boxes = det_frame.get("boxes", [])
        det_scores = det_frame.get("scores", [])
        det_labels = det_frame.get("labels", [])
        det_track_ids = det_frame.get("track_ids", [])

        available_det_idx = set(range(len(det_boxes)))

        out_boxes: List[List[float]] = []
        out_scores: List[float] = []
        out_labels: List[str] = []
        out_track_ids: List[int] = []
        out_sources: List[str] = []
        out_objects: List[Dict[str, Any]] = []

        # GT has priority: keep all GT objects, and consume overlapping detections.
        for idx, gt_box in enumerate(gt_boxes):
            gt_label = gt_labels[idx] if idx < len(gt_labels) else "unknown"
            gt_score = float(gt_scores[idx]) if idx < len(gt_scores) else 1.0
            gt_tid = int(gt_track_ids[idx]) if idx < len(gt_track_ids) else -1

            best_det_idx = _find_best_detection_match(
                gt_box=gt_box,
                gt_label=gt_label,
                det_boxes=det_boxes,
                det_labels=det_labels,
                available_det_idx=available_det_idx,
                iou_threshold=iou_threshold,
                require_label_match=require_label_match,
            )
            if best_det_idx is not None:
                available_det_idx.remove(best_det_idx)
                overlap_count += 1

            out_boxes.append(gt_box)
            out_scores.append(gt_score)
            out_labels.append(gt_label)
            out_track_ids.append(gt_tid)
            out_sources.append("gt")
            out_objects.append(
                _accepted_object_entry(
                    bbox=list(gt_box),
                    score=gt_score,
                    label=gt_label,
                    track_id=gt_tid,
                    source="gt",
                )
            )
            if gt_tid >= 0:
                all_track_ids.add(gt_tid)

        # Keep remaining detected objects to maximize object coverage.
        for det_idx in sorted(available_det_idx):
            det_box = det_boxes[det_idx]
            det_score = float(det_scores[det_idx]) if det_idx < len(det_scores) else 0.0
            det_label = det_labels[det_idx] if det_idx < len(det_labels) else "unknown"
            det_tid = int(det_track_ids[det_idx]) if det_idx < len(det_track_ids) else -1

            # Remap detection track IDs to avoid collisions with GT IDs.
            if det_tid not in det_track_id_map:
                det_track_id_map[det_tid] = next_det_track_id
                next_det_track_id += 1
            merged_tid = det_track_id_map[det_tid]

            out_boxes.append(det_box)
            out_scores.append(det_score)
            out_labels.append(det_label)
            out_track_ids.append(merged_tid)
            out_sources.append("det")
            out_objects.append(
                _accepted_object_entry(
                    bbox=list(det_box),
                    score=det_score,
                    label=det_label,
                    track_id=merged_tid,
                    source="det",
                )
            )
            all_track_ids.add(merged_tid)
            carried_det_count += 1

        frame_name = gt_frame.get("frame") or det_frame.get("frame") or f"frame_{frame_index:05d}.jpg"
        image_path = gt_frame.get("image_path") or det_frame.get("image_path", "")
        candidate_objects = list(candidate_objects_by_frame.get(frame_index, []))
        for candidate_object in candidate_objects:
            candidate_track_id = int(candidate_object.get("candidate_track_id", -1))
            if candidate_track_id >= 0:
                candidate_track_ids.add(candidate_track_id)
        candidate_objects_total += len(candidate_objects)

        merged_frames.append(
            {
                "frame": frame_name,
                "frame_index": frame_index,
                "image_path": image_path,
                "boxes": out_boxes,
                "scores": out_scores,
                "labels": out_labels,
                "track_ids": out_track_ids,
                "sources": out_sources,
                "objects": out_objects,
                "candidate_objects": candidate_objects,
            }
        )

    result: Dict[str, Any] = {
        "version": _MERGED_ANNOTATIONS_VERSION,
        "video_id": video_id,
        "num_frames": len(merged_frames),
        "num_tracks": len(all_track_ids),
        "num_objects": sum(len(frame.get("objects", [])) for frame in merged_frames),
        "num_candidate_tracks": len(candidate_track_ids),
        "num_candidate_objects": candidate_objects_total,
        "num_overlaps_replaced_by_gt": overlap_count,
        "num_unmatched_detections_carried": carried_det_count,
        "frames": merged_frames,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    if render_video:
        render_path = render_merged_video(
            video_id=video_id,
            merged_frames=merged_frames,
            output_path=merged_video_file,
        )
        if render_path:
            result["merged_video_path"] = render_path
            with out_file.open("w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
            print(f"  Saved merged video: {merged_video_file.name}")

    print(
        f"  {video_id}: frames={result['num_frames']} tracks={result['num_tracks']} "
        f"overlap->gt={overlap_count} carry-det={carried_det_count}"
    )
    return result


def run(
    tracking_results: List[Dict[str, Any]],
    dataset_annotation_results: List[Dict[str, Any]],
    output_root: Optional[Path] = None,
    iou_threshold: float = 0.5,
    require_label_match: bool = True,
    force_recompute: bool = False,
    keep_tracked_only: bool = True,
    render_video: bool = True,
) -> List[Dict[str, Any]]:
    """Merge tracked results with GT annotations.

    Args:
        tracking_results: Output from tracking_driving_mini.run()
        dataset_annotation_results: Output from dataset_annotations_driving_mini.run()
        output_root: Override output root
        iou_threshold: IoU threshold for box matching
        require_label_match: Require label match for merging
        force_recompute: Recompute even if cached
        keep_tracked_only: If True, keep videos without GT as tracked-only results
                          (marked with merged_video_path but no merge).
                          If False, skip videos without GT.
    """
    gt_by_video = {r["video_id"]: r for r in dataset_annotation_results}
    effective_output_root = output_root or get_output_root()

    merged_results: List[Dict[str, Any]] = []
    for tracked in tracking_results:
        video_id = tracked["video_id"]
        if video_id not in gt_by_video:
            if keep_tracked_only:
                print(f"  [tracked-only] {video_id} (no GT annotations)")
                merged_results.append(_tracked_video_as_merged_result(tracked))
            else:
                print(f"  [skip] {video_id} not found in GT annotations")
            continue

        merged = merge_video(
            tracked_video=tracked,
            gt_video=gt_by_video[video_id],
            output_root=effective_output_root,
            iou_threshold=iou_threshold,
            require_label_match=require_label_match,
            force_recompute=force_recompute,
            render_video=render_video,
        )
        merged_results.append(merged)

    manifest = {
        "version": _MERGED_ANNOTATIONS_VERSION,
        "num_videos": len(merged_results),
        "num_frames_total": sum(r["num_frames"] for r in merged_results),
        "num_tracks_total": sum(r["num_tracks"] for r in merged_results),
        "num_objects_total": sum(r.get("num_objects", 0) for r in merged_results),
        "num_candidate_tracks_total": sum(r.get("num_candidate_tracks", 0) for r in merged_results),
        "num_candidate_objects_total": sum(r.get("num_candidate_objects", 0) for r in merged_results),
        "num_overlaps_replaced_by_gt_total": sum(
            r.get("num_overlaps_replaced_by_gt", 0) for r in merged_results
        ),
        "num_unmatched_detections_carried_total": sum(
            r.get("num_unmatched_detections_carried", 0) for r in merged_results
        ),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r["num_frames"],
                "num_tracks": r["num_tracks"],
                "num_objects": r.get("num_objects", 0),
                "num_candidate_tracks": r.get("num_candidate_tracks", 0),
                "num_candidate_objects": r.get("num_candidate_objects", 0),
                "num_overlaps_replaced_by_gt": r.get("num_overlaps_replaced_by_gt", 0),
                "num_unmatched_detections_carried": r.get("num_unmatched_detections_carried", 0),
                "is_tracked_only": "num_overlaps_replaced_by_gt" not in r,
            }
            for r in merged_results
        ],
    }

    manifest_path = effective_output_root / "merged_annotations_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Saved merged annotation manifest to {manifest_path}")
    return merged_results
