"""Atomic persistence helpers for dense Step 2 evidence artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from src.exp_august.contracts import ArtifactRef, CoordinateSpace
from src.exp_august.contracts.codec import sha256_bytes


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def read_image_artifact(
    path: Path,
    flags: int = cv2.IMREAD_UNCHANGED,
) -> np.ndarray | None:
    """Decode an image without relying on process-global ``cv2.imread``.

    Ultralytics replaces ``cv2.imread`` at import time and expands grayscale
    images from ``(height, width)`` to ``(height, width, 1)``. Dense evidence
    contracts describe masks as two-dimensional arrays, so decode the bytes
    directly through OpenCV instead of using that mutable entry point.
    """

    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


def write_mask_artifact(
    *,
    stage_root: Path,
    relative_path: Path,
    artifact_id: str,
    mask: np.ndarray,
) -> ArtifactRef:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not np.any(binary):
        raise ValueError("mask artifacts must contain a non-empty 2D mask")
    encoded_ok, encoded = cv2.imencode(".png", binary.astype(np.uint8) * 255)
    if not encoded_ok:
        raise RuntimeError("OpenCV could not encode a mask artifact")
    payload = encoded.tobytes()
    path = stage_root / relative_path
    _write_bytes_atomic(path, payload)
    return ArtifactRef(
        artifact_id=artifact_id,
        relative_path=relative_path.as_posix(),
        sha256=sha256_bytes(payload),
        byte_size=len(payload),
        media_type="image/png",
        shape=tuple(int(value) for value in binary.shape),
        dtype="bool",
        coordinate_space=CoordinateSpace.IMAGE_PIXELS,
    )


def write_npz_artifact(
    *,
    stage_root: Path,
    relative_path: Path,
    artifact_id: str,
    arrays: Mapping[str, np.ndarray],
    primary_shape: tuple[int, ...],
    dtype_description: str,
    media_type: str,
) -> ArtifactRef:
    path = stage_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    payload = temporary.read_bytes()
    temporary.replace(path)
    return ArtifactRef(
        artifact_id=artifact_id,
        relative_path=relative_path.as_posix(),
        sha256=sha256_bytes(payload),
        byte_size=len(payload),
        media_type=media_type,
        shape=primary_shape,
        dtype=dtype_description,
        coordinate_space=CoordinateSpace.IMAGE_PIXELS,
    )
