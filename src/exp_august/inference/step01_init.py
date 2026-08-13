"""Annotation-free Step 1: video validation and timeline normalization."""

from __future__ import annotations

import bisect
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import cv2

from src.exp_august.contracts import (
    ArtifactRef,
    DecodeStatus,
    DecodeValidation,
    DecodeValidationMode,
    FrameRecord,
    ImageSize,
    InitBundle,
    Step1ConfigSnapshot,
    TimelineSource,
    TimelineTransform,
    ToolVersion,
    VideoManifest,
)
from src.exp_august.contracts.codec import (
    hash_payload,
    sha256_file,
    write_contract,
)


VIDEO_EXTENSIONS = (".mov", ".mp4", ".avi", ".mkv")


class VideoValidationError(RuntimeError):
    """Raised when Step 1 cannot establish a trustworthy input timeline."""


@dataclass(frozen=True)
class _VideoProbe:
    width: int
    height: int
    display_rotation_degrees_clockwise: int
    fps: float
    duration_s: float
    frame_count: int
    timestamps_s: tuple[float, ...]
    timeline_source: TimelineSource
    codec: str | None
    backend: str
    tool_versions: tuple[ToolVersion, ...]


@dataclass(frozen=True)
class _PreparedVideo:
    video_id: str
    path: Path
    sha256: str
    byte_size: int
    probe: _VideoProbe


@dataclass(frozen=True)
class Step1Result:
    """Programmatic result; the persisted boundary is ``InitBundle``."""

    bundle: InitBundle
    manifests: tuple[VideoManifest, ...]
    run_root: Path
    bundle_path: Path


def _parse_rate(value: object) -> float:
    text = str(value or "").strip()
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        try:
            return float(text)
        except ValueError:
            return 0.0


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@lru_cache(maxsize=8)
def _ffprobe_version(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "-version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        first_line = completed.stdout.splitlines()[0]
        marker = "ffprobe version "
        if marker in first_line:
            return first_line.split(marker, 1)[1].split()[0]
        return first_line.strip() or "unknown"
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def _ffprobe_video(path: Path, executable: str) -> _VideoProbe:
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,"
            "nb_frames,duration:stream_side_data=rotation:"
            "frame=best_effort_timestamp_time,pts_time"
        ),
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise VideoValidationError(f"ffprobe found no video stream: {path}")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    frames = payload.get("frames") or []
    timestamps: list[float] = []
    for index, frame in enumerate(frames):
        timestamp = _finite_float(
            frame.get("best_effort_timestamp_time", frame.get("pts_time"))
        )
        if timestamp is None and fps > 0.0:
            timestamp = index / fps
        if timestamp is None:
            raise VideoValidationError(f"frame {index} has no usable timestamp: {path}")
        timestamps.append(timestamp)
    declared_count = int(stream.get("nb_frames") or 0)
    frame_count = len(timestamps) or declared_count
    duration = _finite_float(stream.get("duration")) or 0.0
    display_rotation = 0
    for side_data in stream.get("side_data_list") or []:
        rotation = _finite_float(side_data.get("rotation"))
        if rotation is not None:
            # FFmpeg's display-matrix convention is counter-clockwise; the
            # canonical contract records the clockwise transform applied to
            # decoded pixels.
            display_rotation = int(round(-rotation)) % 360
            break
    if fps <= 0.0 and frame_count > 0 and duration > 0.0:
        fps = frame_count / duration
    if not timestamps and frame_count > 0 and fps > 0.0:
        timestamps = [index / fps for index in range(frame_count)]
    if duration <= 0.0 and timestamps and fps > 0.0:
        duration = timestamps[-1] - timestamps[0] + 1.0 / fps
    if width <= 0 or height <= 0 or fps <= 0.0 or frame_count <= 0 or duration <= 0.0:
        raise VideoValidationError(f"incomplete video metadata from ffprobe: {path}")
    timeline_source = TimelineSource.CONTAINER_PTS
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        timestamps = [index / fps for index in range(frame_count)]
        timeline_source = TimelineSource.ASSUMED_CONSTANT_FRAME_RATE
    return _VideoProbe(
        width=width,
        height=height,
        display_rotation_degrees_clockwise=display_rotation,
        fps=fps,
        duration_s=duration,
        frame_count=frame_count,
        timestamps_s=tuple(timestamps),
        timeline_source=timeline_source,
        codec=str(stream.get("codec_name")) if stream.get("codec_name") else None,
        backend="ffprobe",
        tool_versions=(ToolVersion(name="ffprobe", version=_ffprobe_version(executable)),),
    )


def _opencv_video(path: Path) -> _VideoProbe:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoValidationError(f"OpenCV could not open video: {path}")
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        display_rotation = int(round(capture.get(cv2.CAP_PROP_ORIENTATION_META))) % 360
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0.0 or frame_count <= 0:
        raise VideoValidationError(f"OpenCV found incomplete video metadata: {path}")
    return _VideoProbe(
        width=width,
        height=height,
        display_rotation_degrees_clockwise=display_rotation,
        fps=fps,
        duration_s=frame_count / fps,
        frame_count=frame_count,
        timestamps_s=tuple(index / fps for index in range(frame_count)),
        timeline_source=TimelineSource.ASSUMED_CONSTANT_FRAME_RATE,
        codec=None,
        backend="opencv_cfr_fallback",
        tool_versions=(ToolVersion(name="opencv", version=cv2.__version__),),
    )


def probe_video(path: Path, *, ffprobe_executable: str = "ffprobe") -> _VideoProbe:
    executable = shutil.which(ffprobe_executable)
    if executable:
        try:
            return _ffprobe_video(path, executable)
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
            pass
    return _opencv_video(path)


def _video_directory(dataset_root: Path) -> Path:
    nested = dataset_root / "videos"
    return nested if nested.is_dir() else dataset_root


def resolve_video_inputs(
    *,
    dataset_root: Path,
    video_paths: Sequence[Path | str] | None = None,
    video_ids: Sequence[str] | None = None,
    video_count: int | None = None,
) -> tuple[tuple[str, Path], ...]:
    """Resolve raw videos only; labels and extracted-frame folders are ignored."""

    if video_paths and video_ids:
        raise ValueError("provide video_paths or video_ids, not both")
    if video_count is not None and video_count <= 0:
        raise ValueError("video_count must be positive")
    video_dir = _video_directory(dataset_root.expanduser().resolve())
    if video_paths:
        selected = [(Path(value).stem, Path(value).expanduser().resolve()) for value in video_paths]
    else:
        available: dict[str, Path] = {}
        duplicates: set[str] = set()
        for path in sorted(video_dir.iterdir()) if video_dir.is_dir() else ():
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            if path.stem in available:
                duplicates.add(path.stem)
            available[path.stem] = path.resolve()
        if duplicates:
            raise VideoValidationError(
                "ambiguous video IDs with multiple containers: " + ", ".join(sorted(duplicates))
            )
        requested = list(video_ids) if video_ids else sorted(available)
        selected = []
        for video_id in requested:
            if video_id not in available:
                raise FileNotFoundError(f"video ID not found in {video_dir}: {video_id}")
            selected.append((video_id, available[video_id]))
    if video_count is not None:
        selected = selected[:video_count]
    if not selected:
        raise VideoValidationError("Step 1 resolved no raw video inputs")
    seen: set[str] = set()
    normalized: list[tuple[str, Path]] = []
    for video_id, path in selected:
        if video_id in seen:
            raise VideoValidationError(f"duplicate video ID: {video_id}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"video input is missing or empty: {path}")
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise VideoValidationError(f"unsupported video extension: {path.suffix}")
        seen.add(video_id)
        normalized.append((video_id, path))
    return tuple(normalized)


def _normalized_frame_mapping(
    probe: _VideoProbe,
    canonical_fps: float,
) -> tuple[tuple[int, float, float, float], ...]:
    if canonical_fps <= 0.0:
        raise ValueError("canonical_fps must be positive")
    if canonical_fps > probe.fps + 1e-6:
        raise VideoValidationError(
            f"canonical FPS {canonical_fps:g} exceeds source FPS {probe.fps:g}; "
            "Step 1 currently supports downsampling only"
        )
    source_times = probe.timestamps_s
    if not source_times:
        raise VideoValidationError("video probe produced no frame timestamps")
    origin = source_times[0]
    normalized_source = tuple(value - origin for value in source_times)
    if any(right <= left for left, right in zip(normalized_source, normalized_source[1:])):
        raise VideoValidationError("source timestamps must be strictly increasing")
    final_supported_time = normalized_source[-1] + 0.5 / probe.fps
    canonical_count = max(1, int(math.floor(final_supported_time * canonical_fps + 1e-9)) + 1)
    mapping: list[tuple[int, float, float, float]] = []
    previous_source_index = -1
    for canonical_index in range(canonical_count):
        target_time = canonical_index / canonical_fps
        insertion = bisect.bisect_left(normalized_source, target_time, lo=previous_source_index + 1)
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if previous_source_index < index < len(normalized_source)
        ]
        if not candidates:
            break
        source_index = min(
            candidates,
            key=lambda index: (abs(normalized_source[index] - target_time), index),
        )
        source_timestamp = source_times[source_index]
        error = abs(normalized_source[source_index] - target_time)
        mapping.append((source_index, target_time, source_timestamp, error))
        previous_source_index = source_index
    if not mapping:
        raise VideoValidationError("timeline normalization produced no canonical frames")
    return tuple(mapping)


def _sample_indices(frame_count: int, sample_count: int) -> tuple[int, ...]:
    if sample_count == 1:
        return (0,)
    if frame_count <= sample_count:
        return tuple(range(frame_count))
    values = {
        int(round(position * (frame_count - 1) / (sample_count - 1)))
        for position in range(sample_count)
    }
    return tuple(sorted(values))


def _validate_decode(
    path: Path,
    mapping: Sequence[tuple[int, float, float, float]],
    mode: DecodeValidationMode,
    sample_count: int,
) -> tuple[tuple[int, ...], ImageSize]:
    if mode == DecodeValidationMode.NONE:
        return (), ImageSize(width=1, height=1)
    canonical_indices = (
        tuple(range(len(mapping)))
        if mode == DecodeValidationMode.FULL
        else _sample_indices(len(mapping), sample_count)
    )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoValidationError(f"OpenCV could not decode video: {path}")
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
    decoded_size: tuple[int, int] | None = None
    try:
        if mode == DecodeValidationMode.FULL:
            wanted = {mapping[index][0]: index for index in canonical_indices}
            final_source_index = max(wanted)
            found: set[int] = set()
            for source_index in range(final_source_index + 1):
                ok, frame = capture.read()
                if not ok:
                    raise VideoValidationError(
                        f"decode failed at source frame {source_index}: {path}"
                    )
                if source_index not in wanted:
                    continue
                size = (int(frame.shape[1]), int(frame.shape[0]))
                if decoded_size is not None and size != decoded_size:
                    raise VideoValidationError(f"decoded frame size changed in {path}")
                decoded_size = size
                found.add(wanted[source_index])
            if found != set(canonical_indices):
                raise VideoValidationError(f"not all canonical frames were decoded: {path}")
        else:
            for canonical_index in canonical_indices:
                source_index = mapping[canonical_index][0]
                capture.set(cv2.CAP_PROP_POS_FRAMES, float(source_index))
                ok, frame = capture.read()
                if not ok:
                    raise VideoValidationError(
                        f"decode failed at source frame {source_index}: {path}"
                    )
                size = (int(frame.shape[1]), int(frame.shape[0]))
                if decoded_size is not None and size != decoded_size:
                    raise VideoValidationError(f"decoded frame size changed in {path}")
                decoded_size = size
    finally:
        capture.release()
    if decoded_size is None:
        raise VideoValidationError(f"decode validation produced no frames: {path}")
    return canonical_indices, ImageSize(width=decoded_size[0], height=decoded_size[1])


def _build_manifest(
    prepared: _PreparedVideo,
    *,
    run_id: str,
    dataset_name: str,
    config: Step1ConfigSnapshot,
    config_sha256: str,
) -> VideoManifest:
    mapping = _normalized_frame_mapping(prepared.probe, config.canonical_fps)
    checked, decoded_size = _validate_decode(
        prepared.path,
        mapping,
        config.decode_validation_mode,
        config.decode_sample_count,
    )
    if config.decode_validation_mode == DecodeValidationMode.NONE:
        swap_axes = prepared.probe.display_rotation_degrees_clockwise in {90, 270}
        image_size = ImageSize(
            width=prepared.probe.height if swap_axes else prepared.probe.width,
            height=prepared.probe.width if swap_axes else prepared.probe.height,
        )
    else:
        image_size = decoded_size
    checked_set = set(checked)
    frames = tuple(
        FrameRecord(
            frame_index=canonical_index,
            timestamp_s=target_timestamp,
            source_frame_index=source_index,
            source_timestamp_s=source_timestamp,
            sampling_error_s=sampling_error,
            decode_status=(
                DecodeStatus.DECODED
                if canonical_index in checked_set
                else DecodeStatus.NOT_CHECKED
            ),
        )
        for canonical_index, (source_index, target_timestamp, source_timestamp, sampling_error)
        in enumerate(mapping)
    )
    errors = tuple(row[3] for row in mapping)
    tool_versions = list(prepared.probe.tool_versions)
    if (
        config.decode_validation_mode != DecodeValidationMode.NONE
        and not any(tool.name == "opencv" for tool in tool_versions)
    ):
        tool_versions.append(ToolVersion(name="opencv", version=cv2.__version__))
    return VideoManifest(
        run_id=run_id,
        video_id=prepared.video_id,
        dataset_name=dataset_name,
        source_path=str(prepared.path),
        input_sha256=prepared.sha256,
        input_byte_size=prepared.byte_size,
        container=prepared.path.suffix.lower().lstrip("."),
        codec=prepared.probe.codec,
        encoded_image_size=ImageSize(
            width=prepared.probe.width,
            height=prepared.probe.height,
        ),
        image_size=image_size,
        display_rotation_degrees_clockwise=(
            prepared.probe.display_rotation_degrees_clockwise
        ),
        orientation_applied=(prepared.probe.display_rotation_degrees_clockwise != 0),
        source_fps=prepared.probe.fps,
        canonical_fps=config.canonical_fps,
        duration_s=prepared.probe.duration_s,
        source_frame_count=prepared.probe.frame_count,
        canonical_frame_count=len(frames),
        frames=frames,
        timeline_transform=TimelineTransform(
            source=prepared.probe.timeline_source,
            source_time_origin_s=prepared.probe.timestamps_s[0],
            canonical_fps=config.canonical_fps,
            mean_sampling_error_s=sum(errors) / len(errors),
            max_sampling_error_s=max(errors),
        ),
        decode_validation=DecodeValidation(
            mode=config.decode_validation_mode,
            checked_frame_indices=checked,
        ),
        probe_backend=prepared.probe.backend,
        config_sha256=config_sha256,
        random_seed=config.random_seed,
        tool_versions=tuple(tool_versions),
    )


def run_step1(
    *,
    output_root: Path | str,
    dataset_root: Path | str,
    dataset_name: str = "driving_mini",
    video_paths: Sequence[Path | str] | None = None,
    video_ids: Sequence[str] | None = None,
    video_count: int | None = None,
    canonical_fps: float = 10.0,
    decode_validation_mode: DecodeValidationMode | str = DecodeValidationMode.SAMPLE,
    decode_sample_count: int = 7,
    random_seed: int = 726381,
    ffprobe_executable: str = "ffprobe",
) -> Step1Result:
    """Validate raw videos and persist a canonical Step 1 contract bundle."""

    if isinstance(decode_validation_mode, str):
        decode_validation_mode = DecodeValidationMode(decode_validation_mode)
    config = Step1ConfigSnapshot(
        dataset_name=dataset_name,
        canonical_fps=float(canonical_fps),
        decode_validation_mode=decode_validation_mode,
        decode_sample_count=int(decode_sample_count),
        random_seed=int(random_seed),
    )
    config_sha256 = hash_payload(config)
    selected = resolve_video_inputs(
        dataset_root=Path(dataset_root),
        video_paths=video_paths,
        video_ids=video_ids,
        video_count=video_count,
    )
    prepared: list[_PreparedVideo] = []
    for video_id, path in selected:
        prepared.append(
            _PreparedVideo(
                video_id=video_id,
                path=path,
                sha256=sha256_file(path),
                byte_size=path.stat().st_size,
                probe=probe_video(path, ffprobe_executable=ffprobe_executable),
            )
        )
    run_digest = hash_payload(
        {
            "config_sha256": config_sha256,
            "tool_versions": sorted(
                {
                    (tool.name, tool.version)
                    for item in prepared
                    for tool in item.probe.tool_versions
                }
                | (
                    {("opencv", cv2.__version__)}
                    if config.decode_validation_mode != DecodeValidationMode.NONE
                    else set()
                )
            ),
            "inputs": [
                {"video_id": item.video_id, "input_sha256": item.sha256}
                for item in prepared
            ],
        }
    )
    run_id = f"august-{run_digest[:16]}"
    run_root = Path(output_root).expanduser().resolve() / run_id
    stage_root = run_root / "01_init"
    manifests: list[VideoManifest] = []
    references: list[ArtifactRef] = []
    for item in prepared:
        manifest = _build_manifest(
            item,
            run_id=run_id,
            dataset_name=dataset_name,
            config=config,
            config_sha256=config_sha256,
        )
        relative_path = Path("videos") / f"{item.video_id}.manifest.json"
        manifest_path = stage_root / relative_path
        artifact_sha256, byte_size = write_contract(manifest_path, manifest)
        manifests.append(manifest)
        references.append(
            ArtifactRef(
                artifact_id=f"video_manifest:{item.video_id}",
                relative_path=relative_path.as_posix(),
                sha256=artifact_sha256,
                byte_size=byte_size,
                media_type="application/vnd.cauvid.video-manifest+json",
            )
        )
    bundle = InitBundle(
        run_id=run_id,
        config=config,
        config_sha256=config_sha256,
        video_ids=tuple(item.video_id for item in prepared),
        video_manifests=tuple(references),
    )
    bundle_path = stage_root / "init_bundle.json"
    write_contract(bundle_path, bundle)
    return Step1Result(
        bundle=bundle,
        manifests=tuple(manifests),
        run_root=run_root,
        bundle_path=bundle_path,
    )
