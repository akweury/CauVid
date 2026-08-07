from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model import AnnotationDocument


def annotation_path(output_dir: str | Path, video_id: str) -> Path:
    return Path(output_dir) / f"{video_id}.json"


def session_path(output_dir: str | Path, annotator: str) -> Path:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", annotator).strip("._")[:40] or "anonymous"
    digest = hashlib.sha1(annotator.encode("utf-8")).hexdigest()[:8]
    return Path(output_dir) / f".session-{readable}-{digest}.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Some filesystems do not support syncing directory descriptors.
        pass
    return path


def save_document(document: AnnotationDocument, output_dir: str | Path) -> Path:
    path = annotation_path(output_dir, document.video.video_id)
    payload = document.to_dict(datetime.now(timezone.utc).isoformat())
    return _atomic_write_json(path, payload)


def load_document(path: str | Path) -> AnnotationDocument:
    with Path(path).open("r", encoding="utf-8") as handle:
        return AnnotationDocument.from_dict(json.load(handle))


def save_session(output_dir: str | Path, annotator: str, state: dict[str, Any]) -> Path:
    payload = {
        "schema_version": 1,
        "annotator": annotator,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **state,
    }
    return _atomic_write_json(session_path(output_dir, annotator), payload)


def load_session(output_dir: str | Path, annotator: str) -> dict[str, Any] | None:
    path = session_path(output_dir, annotator)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and payload.get("annotator") == annotator else None
