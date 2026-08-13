"""Concrete-frame visualizations for the target Step 3 tracking package."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.exp_august.contracts import (
    ArtifactOwner,
    MaskCandidateSource,
    TrackingStore,
    VideoManifest,
    VideoTrackingManifest,
)
from src.exp_august.contracts.codec import read_contract, sha256_file
from src.exp_august.inference.frames import CanonicalFrameProvider


def _color(track_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(track_id.encode("utf-8")).digest()
    return tuple(int(80 + value % 176) for value in digest[:3])


def _read_mask(
    *,
    link,
    step2_root: Path,
    step3_root: Path,
) -> np.ndarray | None:
    if link is None:
        return None
    root = (
        step2_root
        if link.owner == ArtifactOwner.STEP2_NEURAL_EVIDENCE
        else step3_root
    )
    image = cv2.imread(
        str(root / link.artifact.relative_path),
        cv2.IMREAD_GRAYSCALE,
    )
    return None if image is None else image > 0


def _label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.55,
) -> None:
    x, y = origin
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
    )
    cv2.rectangle(
        image,
        (x, max(0, y - height - baseline - 5)),
        (min(image.shape[1] - 1, x + width + 8), y + 3),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def _overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    image[mask] = np.clip(
        image[mask].astype(np.float32) * (1.0 - alpha)
        + np.asarray(color, dtype=np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def _render_tracking_frame(
    *,
    image: np.ndarray,
    frame_index: int,
    timestamp_s: float,
    package: VideoTrackingManifest,
    step2_root: Path,
    step3_root: Path,
) -> np.ndarray:
    canvas = image.copy()
    candidates = {item.candidate_id: item for item in package.mask_candidate_bank}
    ledger = {row.ledger_id: row for row in package.association_ledger}
    observations = []
    for track in package.tracks:
        for observation in track.observations:
            if observation.frame_index == frame_index:
                observations.append((track, observation))
    for track, observation in observations:
        color = _color(track.track_id)
        candidate = candidates.get(observation.selected_mask_candidate_id)
        mask = (
            _read_mask(
                link=candidate.mask,
                step2_root=step2_root,
                step3_root=step3_root,
            )
            if candidate is not None
            else None
        )
        if mask is not None:
            _overlay_mask(canvas, mask, color, 0.38)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, color, 2)
        box = observation.bbox
        x1, y1, x2, y2 = map(
            int, (round(box.x1), round(box.y1), round(box.x2), round(box.y2))
        )
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        short_id = track.track_id.rsplit(":", 1)[-1].lstrip("0") or "0"
        score = None
        if observation.association_ledger_id:
            row = ledger.get(observation.association_ledger_id)
            score = row.total_score if row else None
        text = f"ID {short_id} | {observation.class_name} | det {observation.confidence:.2f}"
        if score is not None:
            text += f" | assoc {score:.2f}"
        _label(canvas, text, (x1, max(20, y1)), color)

    header_height = max(42, image.shape[0] // 18)
    cv2.rectangle(canvas, (0, 0), (image.shape[1], header_height), (18, 24, 34), -1)
    marker_counts = defaultdict(int)
    for track in package.tracks:
        for marker in track.state_markers:
            if marker.frame_index == frame_index:
                marker_counts[marker.marker_type.value] += 1
    marker_text = ", ".join(f"{key}={value}" for key, value in sorted(marker_counts.items()))
    title = (
        f"Step 3 | {package.video_id} | frame {frame_index:04d}/{package.frame_count - 1:04d} "
        f"| t={timestamp_s:.2f}s | observed tracks={len(observations)}"
    )
    cv2.putText(
        canvas,
        title,
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    if marker_text:
        cv2.putText(
            canvas,
            marker_text,
            (14, header_height - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 215, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _render_candidate_panel(
    *,
    base: np.ndarray,
    frame_index: int,
    package: VideoTrackingManifest,
    step2_root: Path,
    step3_root: Path,
) -> np.ndarray | None:
    candidates = [
        item
        for item in package.mask_candidate_bank
        if item.frame_index == frame_index
        and item.source
        in {
            MaskCandidateSource.FLOW_FORWARD,
            MaskCandidateSource.FLOW_BACKWARD,
            MaskCandidateSource.UNASSIGNED_INSTANCE,
            MaskCandidateSource.EXPLICIT_UNOBSERVABLE,
            MaskCandidateSource.EMPTY_OR_OUTSIDE,
        }
    ]
    if not candidates:
        return None
    source_priority = {
        MaskCandidateSource.FLOW_FORWARD: 0,
        MaskCandidateSource.FLOW_BACKWARD: 1,
        MaskCandidateSource.UNASSIGNED_INSTANCE: 2,
        MaskCandidateSource.EMPTY_OR_OUTSIDE: 3,
        MaskCandidateSource.EXPLICIT_UNOBSERVABLE: 4,
    }
    candidates = sorted(
        candidates,
        key=lambda item: (
            source_priority[item.source],
            item.track_id,
            item.candidate_id,
        ),
    )[:6]
    tiles = []
    for candidate in candidates:
        tile = base.copy()
        color = _color(candidate.track_id)
        mask = _read_mask(
            link=candidate.mask,
            step2_root=step2_root,
            step3_root=step3_root,
        )
        if mask is not None:
            _overlay_mask(tile, mask, color, 0.50)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 54), (18, 24, 34), -1)
        short_id = candidate.track_id.rsplit(":", 1)[-1].lstrip("0") or "0"
        cv2.putText(
            tile,
            f"ID {short_id}: {candidate.source.value}",
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            candidate.observability.value,
            (12, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 215, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(cv2.resize(tile, (480, 270), interpolation=cv2.INTER_AREA))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[index : index + 3]) for index in range(0, len(tiles), 3)]
    panel = np.vstack(rows)
    heading = np.full((68, panel.shape[1], 3), (245, 245, 245), dtype=np.uint8)
    cv2.putText(
        heading,
        f"Step 3 candidate archive | {package.video_id} | frame {frame_index}",
        (18, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (25, 30, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        heading,
        "These are separate evidence hypotheses, not fused or claimed as observed masks.",
        (18, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (55, 65, 80),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((heading, panel))


def _contact_sheet(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    tiles = []
    for image, label in zip(images, labels):
        tile = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 326), (640, 360), (18, 24, 34), -1)
        cv2.putText(
            tile,
            label,
            (12, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    while len(tiles) < 4:
        tiles.append(np.zeros((360, 640, 3), dtype=np.uint8))
    return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:4])))


def render_step3_visualizations(
    *,
    tracking_store_path: Path | str,
    example_frame_count: int = 4,
    render_video: bool = True,
) -> Path:
    store_path = Path(tracking_store_path).expanduser().resolve()
    store = read_contract(store_path, TrackingStore)
    stage_root = store_path.parent
    run_root = stage_root.parent.parent
    output_root = stage_root / "visualizations"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for package_reference in store.video_tracking:
        package_path = stage_root / package_reference.relative_path
        package = read_contract(package_path, VideoTrackingManifest)
        if sha256_file(package_path) != package_reference.sha256:
            raise RuntimeError(f"tracking package hash mismatch: {package_path}")
        step2_root = run_root / package.input_snapshot.source_step2_relative_root
        source_ref = package.input_snapshot.source_video_manifest.artifact
        source_manifest_path = run_root / "01_init" / source_ref.relative_path
        source_manifest = read_contract(source_manifest_path, VideoManifest)
        provider = CanonicalFrameProvider(source_manifest, verify_source_hash=True)
        video_root = output_root / package.video_id
        frame_root = video_root / "frames"
        candidate_root = video_root / "candidate_examples"
        frame_root.mkdir(parents=True, exist_ok=True)
        candidate_root.mkdir(parents=True, exist_ok=True)
        writer = None
        video_path = video_root / f"{package.video_id}_step3_tracks.mp4"
        rendered_frames = []
        rendered_paths = []
        candidate_paths = []
        try:
            for canonical in provider.iter_frames():
                rendered = _render_tracking_frame(
                    image=canonical.image_bgr,
                    frame_index=canonical.frame_index,
                    timestamp_s=canonical.timestamp_s,
                    package=package,
                    step2_root=step2_root,
                    step3_root=stage_root,
                )
                frame_path = frame_root / f"frame_{canonical.frame_index:06d}.png"
                if not cv2.imwrite(str(frame_path), rendered):
                    raise RuntimeError(f"could not write visualization: {frame_path}")
                rendered_frames.append(rendered)
                rendered_paths.append(frame_path)
                panel = _render_candidate_panel(
                    base=canonical.image_bgr,
                    frame_index=canonical.frame_index,
                    package=package,
                    step2_root=step2_root,
                    step3_root=stage_root,
                )
                if panel is not None:
                    panel_path = candidate_root / f"frame_{canonical.frame_index:06d}_candidates.png"
                    if not cv2.imwrite(str(panel_path), panel):
                        raise RuntimeError(f"could not write candidate panel: {panel_path}")
                    candidate_paths.append(panel_path)
                if render_video:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            max(0.1, package.canonical_fps),
                            (rendered.shape[1], rendered.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"could not open visualization video: {video_path}")
                    writer.write(rendered)
        finally:
            if writer is not None:
                writer.release()
        indices = np.linspace(
            0,
            len(rendered_frames) - 1,
            min(example_frame_count, len(rendered_frames)),
            dtype=int,
        )
        sheet = _contact_sheet(
            [rendered_frames[index] for index in indices],
            [f"frame {index} | t={package.canonical_fps and index / package.canonical_fps:.2f}s" for index in indices],
        )
        sheet_path = video_root / f"{package.video_id}_step3_examples.png"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise RuntimeError(f"could not write contact sheet: {sheet_path}")
        manifest_rows.append(
            {
                "video_id": package.video_id,
                "frame_count": package.frame_count,
                "track_count": len(package.tracks),
                "observation_count": sum(len(track.observations) for track in package.tracks),
                "gap_count": len(package.gap_records),
                "frame_paths": [path.relative_to(output_root).as_posix() for path in rendered_paths],
                "contact_sheet": sheet_path.relative_to(output_root).as_posix(),
                "candidate_examples": [path.relative_to(output_root).as_posix() for path in candidate_paths],
                "video": video_path.relative_to(output_root).as_posix() if render_video else None,
            }
        )
    manifest_path = output_root / "step3_visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_name": "step3_visualization_manifest",
                "schema_version": 1,
                "run_id": store.run_id,
                "tracking_store_sha256": sha256_file(store_path),
                "videos": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path
