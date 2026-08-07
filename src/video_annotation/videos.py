from __future__ import annotations

import os
from pathlib import Path

from .model import VideoMetadata, safe_video_id


def logical_relative_path(path: str | Path, dataset_root: str | Path) -> Path:
    """Return the dataset-relative logical path without resolving symlinks."""
    return Path(path).absolute().relative_to(Path(dataset_root).absolute())


def resolve_video_source(path: str | Path) -> Path:
    """Resolve a playable host path, including driving_mini Docker symlinks."""
    path = Path(path)
    if path.is_file():
        return path.resolve()
    if not path.is_symlink():
        return path
    target = Path(os.readlink(path))
    try:
        relative = target.relative_to("/raw_driving_data")
    except ValueError:
        return path
    raw_root_value = os.environ.get("CAUVID_RAW_DRIVING_DATASET")
    if raw_root_value:
        raw_root = Path(raw_root_value).expanduser()
    else:
        import config as project_config

        raw_root = project_config.get_dataset_path("driving_raw")
    candidate = raw_root / relative
    return candidate if candidate.is_file() else path


def discover_videos(dataset_root: str | Path, extensions: tuple[str, ...]) -> list[Path]:
    root = Path(dataset_root).absolute()
    supported = {extension.lower() for extension in extensions}
    scan_root = root / "videos" if (root / "videos").is_dir() else root
    return sorted(
        path
        for path in scan_root.rglob("*")
        if path.suffix.lower() in supported and resolve_video_source(path).is_file()
    )


def inspect_video(path: str | Path, dataset_root: str | Path) -> VideoMetadata:
    import cv2

    path = Path(path).absolute()
    source = resolve_video_source(path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Invalid video metadata for {path}: fps={fps}, frames={frame_count}")
    relative = logical_relative_path(path, dataset_root).as_posix()
    return VideoMetadata(safe_video_id(relative), relative, fps, frame_count)
