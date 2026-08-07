from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .config import AnnotationConfig, Label
from .model import AnnotationDocument, safe_video_id
from .orientation import infer_prepared_frame_rotation
from .storage import annotation_path, load_document, load_session, save_document, save_session
from .validation import validate_payload
from .videos import inspect_video, logical_relative_path, resolve_video_source
from src.exp_driving_videos.modules.data_preprocessing import (
    _disable_opencv_autorotation,
    get_video_rotation,
    rotate_frame,
)


WINDOW_NAME = "CauVid Segment Annotator"
PANEL_HEIGHT = 205

SPECIAL_KEY_CODES = {
    "LEFT": {65361, 2424832, 63234, 16777234},
    "UP": {65362, 2490368, 63232, 16777235},
    "RIGHT": {65363, 2555904, 63235, 16777236},
    "DOWN": {65364, 2621440, 63233, 16777237},
}


def normalize_key_code(code: int) -> str | None:
    """Normalize OpenCV waitKeyEx values across desktop backends."""
    for name, platform_codes in SPECIAL_KEY_CODES.items():
        if code in platform_codes:
            return name
    if 0 <= code <= 255:
        return chr(code)
    return None


def draw_label_legend(canvas: np.ndarray, panel_y: int, labels: tuple[Label, ...]) -> None:
    """Draw configured label colors and keys below the segmentation timeline."""
    x = 12
    baseline = panel_y + 148
    max_x = canvas.shape[1] - 12
    for label in labels:
        text = f"{label.key}: {label.name}"
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        item_width = 21 + text_width + 18
        if x + item_width > max_x and x > 12:
            x = 12
            baseline += 25
        if baseline > panel_y + 178:
            break
        color_bgr = tuple(reversed(label.color))
        cv2.rectangle(canvas, (x, baseline - 13), (x + 14, baseline + 1), color_bgr, -1)
        cv2.rectangle(canvas, (x, baseline - 13), (x + 14, baseline + 1), (235, 235, 235), 1)
        cv2.putText(
            canvas, text, (x + 20, baseline), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (220, 220, 220), 1, cv2.LINE_AA,
        )
        x += item_width


def _format_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    minutes, remainder = divmod(milliseconds, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


class VideoAnnotator:
    def __init__(
        self,
        video_paths: list[Path],
        dataset_root: Path,
        output_dir: Path,
        config: AnnotationConfig,
        annotator: str,
    ) -> None:
        if not video_paths:
            raise ValueError(f"No supported videos found under {dataset_root}")
        self.video_paths = video_paths
        self.dataset_root = dataset_root.resolve()
        self.output_dir = output_dir
        self.config = config
        self.annotator = annotator
        self.video_index = 0
        self.frame_index = 0
        self.playing = False
        self.speed_index = min(
            range(len(config.playback_speeds)), key=lambda index: abs(config.playback_speeds[index] - 1.0)
        )
        self.capture: cv2.VideoCapture | None = None
        self.document: AnnotationDocument | None = None
        self.current_frame: np.ndarray | None = None
        self.display_rotation = 0
        self.rotation_source = "none"
        self._next_capture_frame = 0
        self._trackbar_update = False
        self.status = ""
        self.completed_count = self._completed_count()

    def _completed_count(self) -> int:
        count = 0
        labels = set(self.config.labels_by_id)
        for video_path in self.video_paths:
            relative = logical_relative_path(video_path, self.dataset_root)
            path = annotation_path(self.output_dir, safe_video_id(relative))
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    if not validate_payload(json.load(handle), labels):
                        count += 1
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return count

    def _save_session(self) -> None:
        if self.document is None:
            return
        save_session(
            self.output_dir,
            self.annotator,
            {
                "annotation_version": self.config.annotation_version,
                "video_id": self.document.video.video_id,
                "video_path": self.document.video.path,
                "frame": self.frame_index,
                "playback_speed": self.config.playback_speeds[self.speed_index],
            },
        )

    def _resume_session(self) -> None:
        session = load_session(self.output_dir, self.annotator)
        if session is None:
            self._load_video(0)
            return
        paths = {
            logical_relative_path(path, self.dataset_root).as_posix(): index
            for index, path in enumerate(self.video_paths)
        }
        index = paths.get(str(session.get("video_path", "")), 0)
        saved_speed = float(session.get("playback_speed", 1.0))
        self.speed_index = min(
            range(len(self.config.playback_speeds)),
            key=lambda candidate: abs(self.config.playback_speeds[candidate] - saved_speed),
        )
        self._load_video(index)
        frame = int(session.get("frame", 0))
        self._read_frame(frame)
        self._save_session()
        self.status = f"Session resumed at frame {self.frame_index}; {len(self.document.keyframes)} keyframes loaded"

    def _load_video(self, index: int) -> None:
        if self.capture is not None:
            self._save_session()
            self.capture.release()
        self.video_index = index % len(self.video_paths)
        path = self.video_paths[self.video_index]
        metadata = inspect_video(path, self.dataset_root)
        source = resolve_video_source(path)
        self.capture = cv2.VideoCapture(str(source))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open {path}")
        _disable_opencv_autorotation(self.capture)
        reference_rotation = infer_prepared_frame_rotation(self.capture, path, self.dataset_root)
        if reference_rotation is not None:
            self.display_rotation = reference_rotation
            self.rotation_source = "prepared frame"
        else:
            self.display_rotation = get_video_rotation(source) if self.config.apply_rotation_metadata else 0
            self.rotation_source = "metadata" if self.display_rotation else "none"
            fallback = self.config.portrait_fallback_rotation
            stored_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            stored_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.display_rotation == 0 and fallback is not None and stored_height > stored_width:
                self.display_rotation = fallback
                self.rotation_source = "portrait fallback"
        saved = annotation_path(self.output_dir, metadata.video_id)
        if saved.exists():
            document = load_document(saved)
            if document.video.path != metadata.path or document.video.frame_count != metadata.frame_count:
                raise ValueError(f"Saved annotation metadata does not match video: {saved}")
            document.annotator = self.annotator or document.annotator
            document.annotation_version = self.config.annotation_version
            self.status = f"Resumed {len(document.keyframes)} keyframes"
        else:
            document = AnnotationDocument(metadata, self.annotator, self.config.annotation_version)
            self.status = "New annotation"
        self.document = document
        self.frame_index = 0
        self._next_capture_frame = 0
        self.playing = False
        cv2.setTrackbarMax("Frame", WINDOW_NAME, max(1, metadata.frame_count - 1))
        self._set_trackbar(0)
        self._read_frame(0)
        self._save_session()

    def _set_trackbar(self, frame: int) -> None:
        self._trackbar_update = True
        cv2.setTrackbarPos("Frame", WINDOW_NAME, int(frame))
        self._trackbar_update = False

    def _on_seek(self, frame: int) -> None:
        if self._trackbar_update or self.document is None:
            return
        self.playing = False
        if self._read_frame(frame):
            self._save_session()

    def _read_frame(self, frame: int) -> bool:
        assert self.capture is not None and self.document is not None
        target = max(0, min(int(frame), self.document.video.frame_count - 1))
        if target != self._next_capture_frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        success, image = self.capture.read()
        if not success:
            self.status = f"Could not decode frame {target}"
            self.playing = False
            return False
        if self.display_rotation:
            image = rotate_frame(image, self.display_rotation)
        self.current_frame = image
        self.frame_index = target
        self._next_capture_frame = target + 1
        self._set_trackbar(target)
        return True

    def _save(self, message: str) -> None:
        assert self.document is not None
        path = save_document(self.document, self.output_dir)
        self.status = f"{message}; autosaved {path.name}"
        self.completed_count = self._completed_count()
        self._save_session()

    def _set_label(self, label: Label) -> None:
        assert self.document is not None
        was_keyframe = any(keyframe.frame == self.frame_index for keyframe in self.document.keyframes)
        self.document.set_keyframe(self.frame_index, label.id, set(self.config.labels_by_id))
        cleared = self.document.clear_keyframes_after(self.frame_index)
        suffix = "s" if cleared != 1 else ""
        reset_message = f"; reset {cleared} later keyframe{suffix}" if cleared else ""
        action = "Changed" if was_keyframe else "Added"
        self._save(
            f"{action} frame {self.frame_index} to {label.name}{reset_message}"
        )

    def _delete(self) -> None:
        assert self.document is not None
        if self.document.delete_keyframe(self.frame_index):
            self._save(f"Deleted keyframe {self.frame_index}")
        else:
            self.status = f"Frame {self.frame_index} is not a keyframe"

    def _go_keyframe(self, previous: bool) -> None:
        assert self.document is not None
        target = (
            self.document.previous_keyframe(self.frame_index)
            if previous
            else self.document.next_keyframe(self.frame_index)
        )
        if target is None:
            self.status = f"No {'previous' if previous else 'next'} keyframe"
        else:
            self.playing = False
            if self._read_frame(target):
                self._save_session()

    def _change_speed(self, delta: int) -> None:
        self.speed_index = max(0, min(self.speed_index + delta, len(self.config.playback_speeds) - 1))
        self.status = f"Playback speed {self.config.playback_speeds[self.speed_index]:g}x"
        self._save_session()

    def _handle_key(self, key: str) -> bool:
        if key in self.config.labels_by_key:
            self._set_label(self.config.labels_by_key[key])
            return True
        action = next((name for name, mapped in self.config.shortcuts.items() if mapped == key), None)
        if action == "quit":
            return False
        if action == "play_pause":
            self.playing = not self.playing
            self._save_session()
        elif action == "previous_frame":
            self.playing = False
            if self._read_frame(self.frame_index - 1):
                self._save_session()
        elif action == "next_frame":
            self.playing = False
            if self._read_frame(self.frame_index + 1):
                self._save_session()
        elif action == "previous_video":
            self._load_video(self.video_index - 1)
        elif action == "next_video":
            self._load_video(self.video_index + 1)
        elif action == "previous_keyframe":
            self._go_keyframe(True)
        elif action == "next_keyframe":
            self._go_keyframe(False)
        elif action == "delete_keyframe":
            self._delete()
        elif action == "speed_down":
            self._change_speed(-1)
        elif action == "speed_up":
            self._change_speed(1)
        return True

    def _render(self) -> np.ndarray:
        assert self.current_frame is not None and self.document is not None
        image = self.current_frame
        max_width, max_height = 1280, 720
        scale = min(max_width / image.shape[1], max_height / image.shape[0], 1.0)
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        canvas = np.zeros((image.shape[0] + PANEL_HEIGHT, image.shape[1], 3), dtype=np.uint8)
        canvas[: image.shape[0]] = image
        panel_y = image.shape[0]
        canvas[panel_y:] = (28, 28, 28)
        active = self.document.label_at(self.frame_index)
        label = self.config.labels_by_id.get(active) if active else None
        timestamp = self.frame_index / self.document.video.fps
        progress = self.completed_count
        lines = [
            f"Video {self.video_index + 1}/{len(self.video_paths)}  complete {progress}/{len(self.video_paths)}  {self.document.video.path}",
            f"Frame {self.frame_index}/{self.document.video.frame_count - 1}  {_format_time(timestamp)} / {_format_time(self.document.video.duration_seconds)}  FPS {self.document.video.fps:.3f}",
            f"{'PLAY' if self.playing else 'PAUSE'} {self.config.playback_speeds[self.speed_index]:g}x  Active: {label.name if label else 'UNLABELED'}  Keyframes: {len(self.document.keyframes)}  Rotation: {self.display_rotation:+d} deg ({self.rotation_source})",
        ]
        for index, line in enumerate(lines):
            cv2.putText(canvas, line, (12, panel_y + 24 + index * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 235, 235), 1, cv2.LINE_AA)
        x0, x1 = 12, image.shape[1] - 12
        y0, y1 = panel_y + 88, panel_y + 120
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (65, 65, 65), -1)
        denominator = max(1, self.document.video.frame_count)
        for segment in self.document.segments():
            segment_label = self.config.labels_by_id.get(segment.label)
            if segment_label is None:
                color = (80, 80, 80)
            else:
                color = tuple(reversed(segment_label.color))
            left = x0 + round(segment.start_frame / denominator * (x1 - x0))
            right = x0 + round((segment.end_frame + 1) / denominator * (x1 - x0))
            cv2.rectangle(canvas, (left, y0), (max(left + 1, right), y1), color, -1)
        cursor = x0 + round(self.frame_index / max(1, self.document.video.frame_count - 1) * (x1 - x0))
        cv2.line(canvas, (cursor, y0 - 5), (cursor, y1 + 5), (255, 255, 255), 2)
        draw_label_legend(canvas, panel_y, self.config.labels)
        cv2.putText(canvas, self.status[:180], (12, panel_y + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 255), 1, cv2.LINE_AA)
        return canvas

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Frame", WINDOW_NAME, 0, 1, self._on_seek)
        self._resume_session()
        last_advance = time.monotonic()
        last_session_checkpoint = last_advance
        try:
            running = True
            while running:
                assert self.document is not None
                if self.playing:
                    interval = 1.0 / (self.document.video.fps * self.config.playback_speeds[self.speed_index])
                    now = time.monotonic()
                    if now - last_advance >= interval:
                        if self.frame_index + 1 >= self.document.video.frame_count:
                            self.playing = False
                            self.status = "End of video"
                        else:
                            self._read_frame(self.frame_index + 1)
                        last_advance = now
                now = time.monotonic()
                if now - last_session_checkpoint >= 5.0:
                    self._save_session()
                    last_session_checkpoint = now
                cv2.imshow(WINDOW_NAME, self._render())
                code = cv2.waitKeyEx(5)
                if code >= 0:
                    key = normalize_key_code(code)
                    if key is not None:
                        running = self._handle_key(key)
                    last_advance = time.monotonic()
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self._save_session()
            if self.capture is not None:
                self.capture.release()
            cv2.destroyAllWindows()
