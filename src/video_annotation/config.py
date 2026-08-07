from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Label:
    id: str
    name: str
    key: str
    color: tuple[int, int, int]


@dataclass(frozen=True)
class AnnotationConfig:
    annotation_version: str
    supported_extensions: tuple[str, ...]
    labels: tuple[Label, ...]
    shortcuts: dict[str, str]
    playback_speeds: tuple[float, ...]
    apply_rotation_metadata: bool
    portrait_fallback_rotation: int | None

    @property
    def labels_by_id(self) -> dict[str, Label]:
        return {label.id: label for label in self.labels}

    @property
    def labels_by_key(self) -> dict[str, Label]:
        return {label.key: label for label in self.labels}


NAMED_SHORTCUT_KEYS = {"LEFT", "RIGHT", "UP", "DOWN"}


REQUIRED_ACTIONS = {
    "quit",
    "play_pause",
    "previous_frame",
    "next_frame",
    "previous_video",
    "next_video",
    "previous_keyframe",
    "next_keyframe",
    "delete_keyframe",
    "speed_down",
    "speed_up",
}


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            value = json.load(handle)
        else:
            value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be an object: {path}")
    return value


def load_config(path: str | Path) -> AnnotationConfig:
    path = Path(path)
    raw = _load_mapping(path)
    labels: list[Label] = []
    for item in raw.get("labels", []):
        color = item.get("color", [])
        if len(color) != 3 or any(not isinstance(channel, int) or not 0 <= channel <= 255 for channel in color):
            raise ValueError(f"Label {item.get('id')!r} must have a three-channel RGB color")
        key = str(item.get("key", ""))
        if len(key) != 1:
            raise ValueError(f"Label {item.get('id')!r} must have a single-character key")
        labels.append(Label(str(item["id"]), str(item.get("name", item["id"])), key, tuple(color)))
    if not labels:
        raise ValueError("At least one label is required")
    shortcuts = {}
    for action, raw_key in raw.get("shortcuts", {}).items():
        key = str(raw_key)
        shortcuts[str(action)] = key.upper() if key.upper() in NAMED_SHORTCUT_KEYS else key
    missing = REQUIRED_ACTIONS - shortcuts.keys()
    if missing:
        raise ValueError(f"Missing shortcut mappings: {', '.join(sorted(missing))}")
    shortcut_keys = list(shortcuts.values())
    if any(len(key) != 1 and key not in NAMED_SHORTCUT_KEYS for key in shortcut_keys):
        raise ValueError("Action shortcuts must be single characters or named arrow keys")
    all_keys = [label.key for label in labels] + shortcut_keys
    if len(set(all_keys)) != len(all_keys):
        raise ValueError("Label and action shortcut keys must be unique")
    label_ids = [label.id for label in labels]
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("Label IDs must be unique")
    extensions = tuple(
        extension.lower() if str(extension).startswith(".") else f".{str(extension).lower()}"
        for extension in raw.get("supported_extensions", [".mp4", ".avi", ".mov", ".mkv"])
    )
    speeds = tuple(float(value) for value in raw.get("playback_speeds", [0.25, 0.5, 1.0, 2.0, 4.0]))
    if not speeds or any(value <= 0 for value in speeds):
        raise ValueError("Playback speeds must be positive")
    display = raw.get("display", {})
    if not isinstance(display, dict):
        raise ValueError("display configuration must be an object")
    fallback_raw = display.get("portrait_fallback_rotation", -90)
    fallback_rotation = None if fallback_raw is None else int(fallback_raw)
    if fallback_rotation not in {None, -90, 90}:
        raise ValueError("display.portrait_fallback_rotation must be -90, 90, or null")
    return AnnotationConfig(
        annotation_version=str(raw.get("annotation_version", "1.0")),
        supported_extensions=extensions,
        labels=tuple(labels),
        shortcuts=shortcuts,
        playback_speeds=tuple(sorted(set(speeds))),
        apply_rotation_metadata=bool(display.get("apply_rotation_metadata", True)),
        portrait_fallback_rotation=fallback_rotation,
    )
