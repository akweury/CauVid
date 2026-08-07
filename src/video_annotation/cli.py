from __future__ import annotations

import argparse
import getpass
from pathlib import Path

import config as project_config

from .config import load_config
from .validation import validate_annotation_set
from .videos import discover_videos
from .subset import select_manifest_videos


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "video_annotation" / "labels.yaml"
DEFAULT_DATASET = project_config.get_dataset_path("driving_mini")
DEFAULT_OUTPUT = PROJECT_ROOT / "annotations" / "video_segmentation"
DEFAULT_SUBSET = PROJECT_ROOT / "configs" / "video_annotation" / "video_subset.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local keyframe-based video segmentation annotator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("annotate", "validate"):
        command = subparsers.add_parser(name)
        command.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Root scanned recursively for videos")
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML or JSON label/shortcut configuration")
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Annotation JSON directory")
        command.add_argument("--subset", type=Path, default=DEFAULT_SUBSET, help="Fixed ordered video-subset manifest")
        if name == "annotate":
            command.add_argument("--annotator", default=getpass.getuser(), help="Annotator stored in each export")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    dataset = args.dataset.resolve()
    videos = discover_videos(dataset, config.supported_extensions)
    if not videos:
        print(f"No supported videos found under {dataset}")
        return 1
    try:
        videos = select_manifest_videos(videos, dataset, args.subset)
    except (OSError, ValueError) as error:
        print(f"Invalid video subset: {error}")
        return 2
    if args.command == "validate":
        issues = validate_annotation_set(videos, dataset, args.output.resolve(), set(config.labels_by_id))
        if issues:
            for issue in issues:
                print(f"[{issue.code}] {issue.video_id}: {issue.message}")
            print(f"Validation failed: {len(issues)} issue(s) across {len(videos)} discovered video(s).")
            return 1
        print(f"Validation passed: {len(videos)} video(s) are fully labeled with valid, gap-free segments.")
        return 0
    from .gui import VideoAnnotator

    VideoAnnotator(videos, dataset, args.output.resolve(), config, args.annotator).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
