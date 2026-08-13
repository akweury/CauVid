"""On-demand canonical frame decoding from a validated Step 1 manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from src.exp_august.contracts import InitBundle, VideoManifest
from src.exp_august.contracts.codec import read_contract, sha256_file


class FrameDecodeError(RuntimeError):
    """Raised when source media no longer satisfies its Step 1 contract."""


@dataclass(frozen=True)
class LoadedInitBundle:
    bundle: InitBundle
    manifests: tuple[VideoManifest, ...]
    bundle_path: Path
    stage_root: Path
    run_root: Path


@dataclass(frozen=True)
class CanonicalFrame:
    video_id: str
    frame_index: int
    timestamp_s: float
    source_frame_index: int
    source_timestamp_s: float
    image_bgr: np.ndarray

    @property
    def image_rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)


def load_init_bundle(bundle_path: Path | str, *, verify_artifacts: bool = True) -> LoadedInitBundle:
    path = Path(bundle_path).expanduser().resolve()
    bundle = read_contract(path, InitBundle)
    stage_root = path.parent
    manifests: list[VideoManifest] = []
    for video_id, reference in zip(bundle.video_ids, bundle.video_manifests):
        manifest_path = stage_root / reference.relative_path
        if verify_artifacts and sha256_file(manifest_path) != reference.sha256:
            raise FrameDecodeError(f"video manifest hash mismatch: {manifest_path}")
        manifest = read_contract(manifest_path, VideoManifest)
        if manifest.video_id != video_id or manifest.run_id != bundle.run_id:
            raise FrameDecodeError(f"video manifest identity mismatch: {manifest_path}")
        manifests.append(manifest)
    return LoadedInitBundle(
        bundle=bundle,
        manifests=tuple(manifests),
        bundle_path=path,
        stage_root=stage_root,
        run_root=stage_root.parent,
    )


class CanonicalFrameProvider:
    """Decode exactly the canonical frame sequence declared by Step 1."""

    def __init__(self, manifest: VideoManifest, *, verify_source_hash: bool = True) -> None:
        self.manifest = manifest
        self.source_path = Path(manifest.source_path)
        if not self.source_path.is_file():
            raise FileNotFoundError(f"source video is missing: {self.source_path}")
        if verify_source_hash and sha256_file(self.source_path) != manifest.input_sha256:
            raise FrameDecodeError(f"source video hash mismatch: {self.source_path}")

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(self.source_path))
        if not capture.isOpened():
            raise FrameDecodeError(f"OpenCV could not open source video: {self.source_path}")
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
        return capture

    def _validate_image(self, image: np.ndarray, frame_index: int) -> None:
        height, width = image.shape[:2]
        expected = self.manifest.image_size
        if (width, height) != (expected.width, expected.height):
            raise FrameDecodeError(
                f"canonical frame {frame_index} has size {width}x{height}; "
                f"expected {expected.width}x{expected.height}"
            )

    def get_frame(self, frame_index: int) -> CanonicalFrame:
        if not 0 <= frame_index < len(self.manifest.frames):
            raise IndexError(f"canonical frame index out of range: {frame_index}")
        record = self.manifest.frames[frame_index]
        capture = self._open()
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(record.source_frame_index))
            ok, image = capture.read()
        finally:
            capture.release()
        if not ok:
            raise FrameDecodeError(
                f"decode failed at source frame {record.source_frame_index}: {self.source_path}"
            )
        self._validate_image(image, frame_index)
        return CanonicalFrame(
            video_id=self.manifest.video_id,
            frame_index=record.frame_index,
            timestamp_s=record.timestamp_s,
            source_frame_index=record.source_frame_index,
            source_timestamp_s=record.source_timestamp_s,
            image_bgr=image,
        )

    def iter_frames(self, frame_indices: Sequence[int] | None = None) -> Iterator[CanonicalFrame]:
        requested = (
            tuple(range(len(self.manifest.frames)))
            if frame_indices is None
            else tuple(frame_indices)
        )
        if tuple(sorted(set(requested))) != requested:
            raise ValueError("requested canonical frame indices must be sorted and unique")
        if not requested:
            return
        records = tuple(self.manifest.frames[index] for index in requested)
        source_to_record = {record.source_frame_index: record for record in records}
        capture = self._open()
        try:
            final_source_index = records[-1].source_frame_index
            for source_index in range(final_source_index + 1):
                ok, image = capture.read()
                if not ok:
                    raise FrameDecodeError(
                        f"decode failed at source frame {source_index}: {self.source_path}"
                    )
                record = source_to_record.get(source_index)
                if record is None:
                    continue
                self._validate_image(image, record.frame_index)
                yield CanonicalFrame(
                    video_id=self.manifest.video_id,
                    frame_index=record.frame_index,
                    timestamp_s=record.timestamp_s,
                    source_frame_index=record.source_frame_index,
                    source_timestamp_s=record.source_timestamp_s,
                    image_bgr=image,
                )
        finally:
            capture.release()

    def iter_batches(self, batch_size: int) -> Iterator[tuple[CanonicalFrame, ...]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batch: list[CanonicalFrame] = []
        for frame in self.iter_frames():
            batch.append(frame)
            if len(batch) == batch_size:
                yield tuple(batch)
                batch = []
        if batch:
            yield tuple(batch)

