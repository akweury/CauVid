"""
Prepare the raw driving-video-with-object-tracking dataset for percept2matrix.

Input raw dataset:
    raw_root/
      versions/1/mot_labels.csv
      .../*.mov

Output dataset expected by exp_driving_utils.load_driving_mini_inputs:
    output_root/
      labels.csv
      videos/<video_id>.mov
      frames/<video_id>/frame_00000.jpg
      depth_maps/<video_id>/frame_00000_depth.npz

The label frameIndex values are remapped to the extracted frame sequence. This
matters when target_fps downsamples the videos.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import config


DEFAULT_RAW_ROOT = Path("/storage-02/ml-jsha/driving-video-with-object-tracking")


def _resolve_raw_root(raw_root: str | None) -> Path:
    if raw_root:
        return Path(raw_root).expanduser().resolve()
    env_path = os.environ.get("CAUVID_RAW_DRIVING_DATASET")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return config.DATASET_PATHS.get("driving_raw", DEFAULT_RAW_ROOT).resolve()


def _resolve_output_root(output_root: str | None) -> Path:
    if output_root:
        return Path(output_root).expanduser().resolve()
    env_path = os.environ.get("CAUVID_DRIVING_MINI_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return config.DATASET_PATHS["driving_mini"].resolve()


def _find_labels_csv(raw_root: Path, labels_csv: str | None) -> Path:
    candidates = []
    if labels_csv:
        candidates.append(Path(labels_csv).expanduser())
    candidates.extend([
        raw_root / "versions" / "1" / "mot_labels.csv",
        raw_root / "mot_labels.csv",
        raw_root / "labels.csv",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find labels CSV. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _find_videos(raw_root: Path, video_dir: str | None) -> dict[str, Path]:
    roots = [Path(video_dir).expanduser().resolve()] if video_dir else [raw_root]
    videos = {}
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Video directory does not exist: {root}")
        for path in root.rglob("*"):
            if path.suffix.lower() == ".mov":
                videos.setdefault(path.stem, path.resolve())
    return videos


def _parse_video_ids(value: str | None, file_path: str | None) -> list[str] | None:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    if file_path:
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return None


def _select_video_ids(
    labels: pd.DataFrame,
    available_videos: dict[str, Path],
    explicit_video_ids: list[str] | None,
    limit: int | None,
) -> list[str]:
    label_video_ids = list(labels["videoName"].dropna().astype(str).unique())
    if explicit_video_ids:
        selected = explicit_video_ids
    else:
        selected = [video_id for video_id in label_video_ids if video_id in available_videos]
        if limit is not None:
            selected = selected[:limit]

    missing = [video_id for video_id in selected if video_id not in available_videos]
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} selected videos have no .mov file. First missing: {preview}")

    return selected


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        raise ValueError(f"Unsupported video materialization mode: {mode}")


def extract_video_frames(
    video_path: Path,
    frame_output_dir: Path,
    target_fps: float,
    max_frames: int | None,
    skip_existing: bool,
) -> dict[int, int]:
    import cv2
    from src.exp_driving_videos.modules.data_preprocessing import (
        _disable_opencv_autorotation,
        get_video_rotation,
        rotate_frame,
    )

    existing = sorted(frame_output_dir.glob("frame_*.jpg"))
    if skip_existing and existing:
        return {}

    frame_output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    _disable_opencv_autorotation(cap)

    original_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frame_skip = max(1, int(round(original_fps / target_fps)))
    rotation = get_video_rotation(video_path)

    index_map = {}
    original_index = 0
    extracted_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if original_index % frame_skip == 0:
            if max_frames is not None and extracted_index >= max_frames:
                break
            if rotation:
                frame = rotate_frame(frame, rotation)
            frame_path = frame_output_dir / f"frame_{extracted_index:05d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Failed to write frame: {frame_path}")
            index_map[original_index] = extracted_index
            extracted_index += 1

        original_index += 1

    cap.release()
    return index_map


def _reconstruct_existing_index_map(video_path: Path, frame_output_dir: Path, target_fps: float) -> dict[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for existing-frame remap: {video_path}")
    original_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_skip = max(1, int(round(original_fps / target_fps)))
    extracted_count = len(sorted(frame_output_dir.glob("frame_*.jpg")))
    index_map = {}
    extracted_index = 0
    for original_index in range(total_frames):
        if original_index % frame_skip == 0:
            if extracted_index >= extracted_count:
                break
            index_map[original_index] = extracted_index
            extracted_index += 1
    return index_map


def remap_labels(labels: pd.DataFrame, video_id: str, index_map: dict[int, int]) -> pd.DataFrame:
    video_labels = labels[labels["videoName"].astype(str) == video_id].copy()
    video_labels["frameIndex"] = video_labels["frameIndex"].astype(int)
    video_labels = video_labels[video_labels["frameIndex"].isin(index_map)].copy()
    video_labels["frameIndex"] = video_labels["frameIndex"].map(index_map).astype(int)
    if "name" in video_labels.columns:
        video_labels["name"] = video_labels["frameIndex"].map(
            lambda idx: f"{video_id}-frame_{idx:05d}.jpg"
        )
    return video_labels


def generate_depth_for_videos(
    frames_root: Path,
    depth_root: Path,
    video_ids: list[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> None:
    from src.exp_driving_videos.modules.data_preprocessing import generate_frame_depth_maps

    for video_id in video_ids:
        video_frame_dir = frames_root / video_id
        video_depth_dir = depth_root / video_id
        if list(video_depth_dir.glob("*.npz")):
            print(f"[depth] Skipping existing depth maps: {video_id}")
            continue
        ok = generate_frame_depth_maps(
            frame_folder=video_frame_dir,
            depth_output_folder=video_depth_dir,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
        )
        if not ok:
            raise RuntimeError(f"Depth generation failed for video {video_id}")


def prepare_dataset(args: argparse.Namespace) -> Path:
    import pandas as pd

    raw_root = _resolve_raw_root(args.raw_root)
    output_root = _resolve_output_root(args.output_root)
    labels_csv = _find_labels_csv(raw_root, args.labels_csv)
    available_videos = _find_videos(raw_root, args.video_dir)

    labels = pd.read_csv(labels_csv, low_memory=False)
    video_ids = _select_video_ids(
        labels=labels,
        available_videos=available_videos,
        explicit_video_ids=_parse_video_ids(args.video_ids, args.video_ids_file),
        limit=args.limit,
    )
    if not video_ids:
        raise RuntimeError("No videos selected for preparation.")

    frames_root = output_root / "frames"
    depth_root = output_root / "depth_maps"
    videos_root = output_root / "videos"
    output_root.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)
    depth_root.mkdir(parents=True, exist_ok=True)
    videos_root.mkdir(parents=True, exist_ok=True)

    all_labels = []
    for position, video_id in enumerate(video_ids, start=1):
        video_path = available_videos[video_id]
        print(f"[{position}/{len(video_ids)}] Preparing {video_id}")
        _link_or_copy(video_path, videos_root / video_path.name, args.video_mode)

        frame_output_dir = frames_root / video_id
        index_map = extract_video_frames(
            video_path=video_path,
            frame_output_dir=frame_output_dir,
            target_fps=args.target_fps,
            max_frames=args.max_frames_per_video,
            skip_existing=args.skip_existing,
        )
        if not index_map:
            index_map = _reconstruct_existing_index_map(video_path, frame_output_dir, args.target_fps)

        remapped = remap_labels(labels, video_id, index_map)
        all_labels.append(remapped)
        print(f"  frames={len(index_map)} labels={len(remapped)}")

    labels_out = pd.concat(all_labels, ignore_index=True) if all_labels else labels.iloc[0:0].copy()
    labels_out.to_csv(output_root / "labels.csv", index=False)
    print(f"Saved prepared labels: {output_root / 'labels.csv'} ({len(labels_out)} rows)")

    if args.generate_depth:
        generate_depth_for_videos(
            frames_root=frames_root,
            depth_root=depth_root,
            video_ids=video_ids,
            model_name=args.depth_model,
            batch_size=args.depth_batch_size,
            device=args.depth_device,
        )
    else:
        print(
            "Depth generation skipped. Run again with --generate-depth before "
            "running percept2matrix unless depth_maps/ already exists."
        )

    return output_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare raw driving videos for percept2matrix.")
    parser.add_argument("--raw-root", default=None, help="Raw dataset root containing .mov files and labels CSV.")
    parser.add_argument("--output-root", default=None, help="Prepared dataset output root.")
    parser.add_argument("--labels-csv", default=None, help="Path to mot_labels.csv. Auto-detected by default.")
    parser.add_argument("--video-dir", default=None, help="Directory to search for .mov files. Defaults to raw root recursively.")
    parser.add_argument("--video-ids", default=None, help="Comma-separated video IDs to prepare.")
    parser.add_argument("--video-ids-file", default=None, help="File containing one video ID per line.")
    parser.add_argument("--limit", type=int, default=None, help="Prepare the first N labeled videos that have .mov files.")
    parser.add_argument("--target-fps", type=float, default=5.0, help="Frame extraction FPS.")
    parser.add_argument("--max-frames-per-video", type=int, default=None, help="Optional frame limit per video.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing extracted frames.")
    parser.add_argument("--video-mode", choices=["symlink", "copy"], default="symlink", help="How to materialize videos/.")
    parser.add_argument("--generate-depth", action="store_true", help="Generate depth maps after extracting frames.")
    parser.add_argument("--depth-model", default="depth-anything/DA3-Large", help="Depth Anything model name.")
    parser.add_argument("--depth-batch-size", type=int, default=4, help="Depth generation batch size.")
    parser.add_argument("--depth-device", default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Depth generation device.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_root = prepare_dataset(args)
    print(f"Prepared driving dataset at: {output_root}")


if __name__ == "__main__":
    main()
