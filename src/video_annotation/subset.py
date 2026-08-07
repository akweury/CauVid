from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .videos import logical_relative_path


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) if path.suffix.lower() == ".json" else yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Video subset manifest must be an object: {path}")
    return payload


def select_manifest_videos(
    video_paths: list[Path], dataset_root: str | Path, manifest_path: str | Path
) -> list[Path]:
    """Filter discovered videos to the exact order recorded in a fixed manifest."""
    manifest_path = Path(manifest_path)
    payload = _load_manifest(manifest_path)
    entries = payload.get("videos", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Video subset has no entries: {manifest_path}")
    normalized = [Path(str(entry)).as_posix() for entry in entries]
    if len(normalized) > 100:
        raise ValueError(f"Video subset contains {len(normalized)} entries; maximum is 100")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Video subset contains duplicate paths")
    expected_count = int(payload.get("count", len(normalized)))
    if expected_count != len(normalized):
        raise ValueError(
            f"Video subset count says {expected_count}, but manifest contains {len(normalized)} paths"
        )
    available = {
        logical_relative_path(path, dataset_root).as_posix(): path
        for path in video_paths
    }
    missing = [entry for entry in normalized if entry not in available]
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(f"Video subset references {len(missing)} unavailable video(s): {preview}")
    return [available[entry] for entry in normalized]
