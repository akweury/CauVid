"""Utilities for safely reading and publishing cached depth maps."""

import os
import tempfile
from pathlib import Path

import numpy as np


def is_valid_depth_npz(path) -> bool:
    """Return whether *path* contains a non-empty, finite depth array."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False

    try:
        with np.load(path, allow_pickle=False) as data:
            if not data.files:
                return False
            key = "depth" if "depth" in data.files else data.files[0]
            depth = data[key]
            return (
                depth.ndim >= 2
                and depth.size > 0
                and np.issubdtype(depth.dtype, np.number)
                and bool(np.isfinite(depth).all())
            )
    except (EOFError, OSError, ValueError):
        return False


def atomic_save_depth_npz(path, depth) -> None:
    """Write a depth cache without exposing a partial destination file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(temporary_path, depth=depth)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
