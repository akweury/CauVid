from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def frame_timestamp(frame: int, fps: float) -> float:
    return round(frame / fps, 6) if fps > 0 else 0.0


@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    path: str
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return round(self.frame_count / self.fps, 6) if self.fps > 0 else 0.0


@dataclass(frozen=True, order=True)
class Keyframe:
    frame: int
    label: str = field(compare=False)

    def to_dict(self, fps: float) -> dict[str, Any]:
        return {"frame": self.frame, "timestamp": frame_timestamp(self.frame, fps), "label": self.label}


@dataclass(frozen=True)
class Segment:
    start_frame: int
    end_frame: int
    label: str

    def to_dict(self, fps: float) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_timestamp": frame_timestamp(self.start_frame, fps),
            "end_timestamp": frame_timestamp(self.end_frame, fps),
            "label": self.label,
        }


def keyframes_to_segments(keyframes: Iterable[Keyframe], frame_count: int) -> list[Segment]:
    ordered = sorted(keyframes)
    frames = [keyframe.frame for keyframe in ordered]
    duplicate = next((frame for index, frame in enumerate(frames[1:], 1) if frame == frames[index - 1]), None)
    if duplicate is not None:
        raise ValueError(f"Duplicate keyframe at frame {duplicate}")
    segments: list[Segment] = []
    for index, keyframe in enumerate(ordered):
        if not 0 <= keyframe.frame < frame_count:
            raise ValueError(f"Keyframe {keyframe.frame} is outside video range 0..{frame_count - 1}")
        end = ordered[index + 1].frame - 1 if index + 1 < len(ordered) else frame_count - 1
        if end < keyframe.frame:
            raise ValueError(f"Invalid segment range {keyframe.frame}..{end}")
        segments.append(Segment(keyframe.frame, end, keyframe.label))
    return segments


@dataclass
class AnnotationDocument:
    video: VideoMetadata
    annotator: str
    annotation_version: str
    keyframes: list[Keyframe] = field(default_factory=list)

    def _normalize(self) -> None:
        self.keyframes.sort()

    def set_keyframe(self, frame: int, label: str, valid_labels: set[str] | None = None) -> None:
        if not 0 <= frame < self.video.frame_count:
            raise ValueError(f"Frame {frame} is outside video range")
        if valid_labels is not None and label not in valid_labels:
            raise ValueError(f"Unknown label: {label}")
        for index, keyframe in enumerate(self.keyframes):
            if keyframe.frame == frame:
                self.keyframes[index] = Keyframe(frame, label)
                self._normalize()
                return
        self.keyframes.append(Keyframe(frame, label))
        self._normalize()

    def clear_keyframes_after(self, frame: int) -> int:
        """Remove stale annotations after an edited frame and return their count."""
        original_length = len(self.keyframes)
        self.keyframes = [keyframe for keyframe in self.keyframes if keyframe.frame <= frame]
        return original_length - len(self.keyframes)

    def delete_keyframe(self, frame: int) -> bool:
        original_length = len(self.keyframes)
        self.keyframes = [keyframe for keyframe in self.keyframes if keyframe.frame != frame]
        return len(self.keyframes) != original_length

    def label_at(self, frame: int) -> str | None:
        active: str | None = None
        for keyframe in self.keyframes:
            if keyframe.frame > frame:
                break
            active = keyframe.label
        return active

    def previous_keyframe(self, frame: int) -> int | None:
        candidates = [keyframe.frame for keyframe in self.keyframes if keyframe.frame < frame]
        return candidates[-1] if candidates else None

    def next_keyframe(self, frame: int) -> int | None:
        return next((keyframe.frame for keyframe in self.keyframes if keyframe.frame > frame), None)

    def segments(self) -> list[Segment]:
        return keyframes_to_segments(self.keyframes, self.video.frame_count)

    def to_dict(self, updated_at: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "annotation_version": self.annotation_version,
            "annotator": self.annotator,
            "updated_at": updated_at,
            "video": {
                "id": self.video.video_id,
                "path": self.video.path,
                "fps": self.video.fps,
                "frame_count": self.video.frame_count,
                "duration_seconds": self.video.duration_seconds,
            },
            "keyframes": [keyframe.to_dict(self.video.fps) for keyframe in self.keyframes],
            "segments": [segment.to_dict(self.video.fps) for segment in self.segments()],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AnnotationDocument":
        video = raw["video"]
        document = cls(
            video=VideoMetadata(
                video_id=str(video["id"]),
                path=str(video["path"]),
                fps=float(video["fps"]),
                frame_count=int(video["frame_count"]),
            ),
            annotator=str(raw.get("annotator", "")),
            annotation_version=str(raw.get("annotation_version", "1.0")),
            keyframes=[Keyframe(int(item["frame"]), str(item["label"])) for item in raw.get("keyframes", [])],
        )
        document._normalize()
        return document


def safe_video_id(relative_path: str | Path) -> str:
    import hashlib

    normalized = Path(relative_path).as_posix()
    readable = "__".join(Path(normalized).with_suffix("").parts)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{digest}"
