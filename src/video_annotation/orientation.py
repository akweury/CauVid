from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


_ROTATION_TRANSFORMS = {
    0: lambda frame: frame,
    -90: lambda frame: cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE),
    90: lambda frame: cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE),
    180: lambda frame: cv2.rotate(frame, cv2.ROTATE_180),
}


def infer_prepared_frame_rotation(
    capture: cv2.VideoCapture, logical_video_path: str | Path, dataset_root: str | Path
) -> int | None:
    """Infer display rotation by matching raw frame 0 to the pipeline frame 0."""
    logical_video_path = Path(logical_video_path)
    references = sorted(
        (Path(dataset_root) / "frames" / logical_video_path.stem).glob("frame_*.jpg")
    )
    if not references:
        return None
    ok, raw = capture.read()
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ok or raw is None:
        return None
    reference = cv2.imread(str(references[0]))
    if reference is None:
        return None
    scores: dict[int, float] = {}
    for rotation, transform in _ROTATION_TRANSFORMS.items():
        candidate = transform(raw)
        if candidate.shape[:2] != reference.shape[:2]:
            continue
        preview_size = (160, 90)
        candidate_small = cv2.resize(candidate, preview_size, interpolation=cv2.INTER_AREA).astype(np.float32)
        reference_small = cv2.resize(reference, preview_size, interpolation=cv2.INTER_AREA).astype(np.float32)
        scores[rotation] = float(np.mean(np.abs(candidate_small - reference_small)))
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda item: item[1])
    best_rotation, best_score = ranked[0]
    separation = ranked[1][1] - best_score if len(ranked) > 1 else float("inf")
    # Prepared frames are re-encodings of the same pixels. Reject unrelated or
    # ambiguous references rather than applying a speculative transformation.
    if best_score > 12.0 or separation < 3.0:
        return None
    return best_rotation
