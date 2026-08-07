from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import Keyframe, keyframes_to_segments, safe_video_id
from .videos import logical_relative_path


@dataclass(frozen=True)
class ValidationIssue:
    video_id: str
    code: str
    message: str


def validate_payload(raw: dict[str, Any], valid_labels: set[str] | None = None) -> list[ValidationIssue]:
    video = raw.get("video", {})
    video_id = str(video.get("id", "<unknown>"))
    issues: list[ValidationIssue] = []
    try:
        frame_count = int(video["frame_count"])
        fps = float(video["fps"])
    except (KeyError, TypeError, ValueError):
        return [ValidationIssue(video_id, "invalid_metadata", "Missing or invalid FPS/frame count")]
    if frame_count <= 0 or fps <= 0:
        issues.append(ValidationIssue(video_id, "invalid_metadata", "FPS and frame count must be positive"))
        return issues
    items = raw.get("keyframes", [])
    if not items:
        return [ValidationIssue(video_id, "unlabeled", "Video has no labeled keyframes")]
    frames: list[int] = []
    keyframes: list[Keyframe] = []
    for item in items:
        try:
            frame = int(item["frame"])
            label = str(item["label"])
        except (KeyError, TypeError, ValueError):
            issues.append(ValidationIssue(video_id, "invalid_keyframe", f"Malformed keyframe: {item!r}"))
            continue
        frames.append(frame)
        keyframes.append(Keyframe(frame, label))
        if not 0 <= frame < frame_count:
            issues.append(ValidationIssue(video_id, "invalid_range", f"Keyframe {frame} is outside 0..{frame_count - 1}"))
        if valid_labels is not None and label not in valid_labels:
            issues.append(ValidationIssue(video_id, "unknown_label", f"Unknown label {label!r} at frame {frame}"))
    duplicates = sorted({frame for frame in frames if frames.count(frame) > 1})
    for frame in duplicates:
        issues.append(ValidationIssue(video_id, "duplicate_keyframe", f"Duplicate keyframe at frame {frame}"))
    if frames and min(frames) > 0:
        issues.append(ValidationIssue(video_id, "gap", f"Frames 0..{min(frames) - 1} are unlabeled"))
    try:
        segments = keyframes_to_segments(keyframes, frame_count)
    except ValueError as error:
        issues.append(ValidationIssue(video_id, "invalid_segments", str(error)))
        return issues
    expected = 0
    for segment in segments:
        if segment.start_frame > expected:
            issues.append(ValidationIssue(video_id, "gap", f"Frames {expected}..{segment.start_frame - 1} are unlabeled"))
        if segment.start_frame < expected:
            issues.append(ValidationIssue(video_id, "overlap", f"Segment starts at {segment.start_frame}, before {expected}"))
        if segment.end_frame < segment.start_frame or segment.end_frame >= frame_count:
            issues.append(ValidationIssue(video_id, "invalid_range", f"Invalid segment {segment.start_frame}..{segment.end_frame}"))
        expected = segment.end_frame + 1
    if expected < frame_count:
        issues.append(ValidationIssue(video_id, "gap", f"Frames {expected}..{frame_count - 1} are unlabeled"))
    return _deduplicate(issues)


def _deduplicate(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    seen: set[tuple[str, str, str]] = set()
    result = []
    for issue in issues:
        key = (issue.video_id, issue.code, issue.message)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def validate_annotation_set(
    video_paths: list[Path], dataset_root: Path, output_dir: Path, valid_labels: set[str] | None = None
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for video_path in video_paths:
        relative = logical_relative_path(video_path, dataset_root)
        video_id = safe_video_id(relative)
        annotation_file = output_dir / f"{video_id}.json"
        if not annotation_file.exists():
            issues.append(ValidationIssue(video_id, "unlabeled", f"No annotation for {relative.as_posix()}"))
            continue
        try:
            with annotation_file.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            issues.append(ValidationIssue(video_id, "invalid_json", str(error)))
            continue
        issues.extend(validate_payload(raw, valid_labels))
    return issues
