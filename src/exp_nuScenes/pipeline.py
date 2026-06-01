"""
nuScenes experiment pipeline skeleton.

This mirrors the high-level flow used by the driving-video experiment:

1. Load local nuScenes trainval data.
2. Treat each nuScenes scene as one video.
3. Iterate through scenes one by one.
4. Leave perception/primitive/reasoning stages as placeholders for now.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import config
from config import PROJECT_ROOT

from src.exp_driving_videos.modules.pipe_utils.exp_driving_utils import load_pattern_cfg_file
from src.exp_nuScenes.ego_motion import (
    extract_ego_speed_3d,
    render_ego_motion_video,
    summarize_ego_speed,
)
from src.exp_nuScenes.load_dataset import (
    CAMERA_CHANNELS,
    default_dataroot,
    load_nuscenes_data,
)
from src.exp_nuScenes.motion_segmentation import (
    segment_ego_motion_signals,
    summarize_motion_segments,
)
from src.exp_nuScenes.object_motion import (
    extract_object_motion_3d,
    summarize_object_motion,
)
from src.exp_nuScenes.segment_objects import (
    extract_objects_in_ego_segments,
    summarize_segment_objects,
)
from src.exp_nuScenes.segment_predicates import (
    extract_segment_predicates,
    summarize_segment_predicates,
)


PIPELINE_DATA_FILENAME = "pipeline_data.pkl"
PIPELINE_DATA_VERSION = 1

DEFAULT_PIPELINE_CONFIG = {
    "dataroot": "dataset/nuScenes",
    "version": "v1.0-trainval_meta/v1.0-trainval",
    "media_root": None,
    "scene_names": [],
    "max_samples": None,
    "camera": "CAM_FRONT",
    "ego_channel": None,
    "visualize_ego_motion": False,
    "ego_motion_fps": 2.0,
    "include_sweeps": False,
    "output_root": None,
    "force_recompute": False,
    "motion_segment_params": {
        "forward_stop_threshold_mps": 0.25,
        "forward_accel_threshold_mps2": 0.5,
        "lateral_speed_threshold_mps": 0.1,
        "min_segment_frames": 2,
        "smoothing_window": 3,
    },
}


def _deep_merge_dicts(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_pipeline_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return dict(DEFAULT_PIPELINE_CONFIG)
    loaded = load_pattern_cfg_file(config_path) or {}
    if "scene" in loaded and "scene_names" not in loaded:
        loaded["scene_names"] = loaded["scene"]
    return _deep_merge_dicts(DEFAULT_PIPELINE_CONFIG, loaded)


def resolve_run_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = getattr(args, "config", None)
    run_cfg = load_pipeline_config(config_path)

    if getattr(args, "dataroot", None) is not None:
        run_cfg["dataroot"] = str(args.dataroot)
    if getattr(args, "version", None) is not None:
        run_cfg["version"] = args.version
    if getattr(args, "scene", None):
        run_cfg["scene_names"] = list(args.scene)
    if getattr(args, "max_samples", None) is not None:
        run_cfg["max_samples"] = args.max_samples
    if getattr(args, "camera", None) is not None:
        run_cfg["camera"] = args.camera
    if getattr(args, "ego_channel", None) is not None:
        run_cfg["ego_channel"] = args.ego_channel
    if getattr(args, "visualize_ego_motion", None) is not None:
        run_cfg["visualize_ego_motion"] = args.visualize_ego_motion
    if getattr(args, "ego_motion_fps", None) is not None:
        run_cfg["ego_motion_fps"] = args.ego_motion_fps
    if getattr(args, "include_sweeps", None) is not None:
        run_cfg["include_sweeps"] = args.include_sweeps
    if getattr(args, "output_root", None) is not None:
        run_cfg["output_root"] = str(args.output_root)
    if getattr(args, "media_root", None) is not None:
        run_cfg["media_root"] = str(args.media_root)
    if getattr(args, "force_recompute", None) is not None:
        run_cfg["force_recompute"] = args.force_recompute

    motion_cfg = dict(run_cfg.get("motion_segment_params", {}))
    if getattr(args, "forward_stop_threshold", None) is not None:
        motion_cfg["forward_stop_threshold_mps"] = args.forward_stop_threshold
    if getattr(args, "forward_accel_threshold", None) is not None:
        motion_cfg["forward_accel_threshold_mps2"] = args.forward_accel_threshold
    if getattr(args, "lateral_speed_threshold", None) is not None:
        motion_cfg["lateral_speed_threshold_mps"] = args.lateral_speed_threshold
    if getattr(args, "motion_min_segment_frames", None) is not None:
        motion_cfg["min_segment_frames"] = args.motion_min_segment_frames
    if getattr(args, "motion_smoothing_window", None) is not None:
        motion_cfg["smoothing_window"] = args.motion_smoothing_window
    run_cfg["motion_segment_params"] = motion_cfg

    return run_cfg


def _resolve_project_path(path_value: Optional[str]) -> Optional[Path]:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_output_root(output_root: Optional[Path] = None) -> Path:
    if output_root is not None:
        root = Path(output_root)
    else:
        root = config.get_output_path("pipeline_output") / "nuScenes"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_scene_output_path(scene_name: str, output_root: Optional[Path] = None) -> Path:
    out_path = get_output_root(output_root) / scene_name
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def get_pipeline_data_file(out_path: Path) -> Path:
    return Path(out_path) / PIPELINE_DATA_FILENAME


def load_pipeline_data(out_path: Path, scene_name: str) -> Dict[str, Any]:
    pipeline_file = get_pipeline_data_file(out_path)
    if pipeline_file.exists():
        with pipeline_file.open("rb") as file:
            data = pickle.load(file)
        data.setdefault("schema_version", PIPELINE_DATA_VERSION)
        data.setdefault("scene_name", scene_name)
        data.setdefault("stages", {})
        return data

    return {
        "schema_version": PIPELINE_DATA_VERSION,
        "scene_name": scene_name,
        "stages": {},
    }


def save_pipeline_data(out_path: Path, pipeline_data: Dict[str, Any]) -> Path:
    pipeline_file = get_pipeline_data_file(out_path)
    pipeline_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = pipeline_file.with_suffix(f"{pipeline_file.suffix}.tmp")
    with tmp_file.open("wb") as file:
        pickle.dump(pipeline_data, file)
    tmp_file.replace(pipeline_file)
    return pipeline_file


def set_pipeline_stage(
    pipeline_data: Dict[str, Any],
    out_path: Path,
    stage_name: str,
    data: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Path:
    pipeline_data.setdefault("stages", {})[stage_name] = {
        "_stage_cache": True,
        "params": params or {},
        "data": data,
    }
    return save_pipeline_data(out_path, pipeline_data)


def get_pipeline_stage(
    pipeline_data: Dict[str, Any],
    stage_name: str,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    entry = pipeline_data.setdefault("stages", {}).get(stage_name)
    if not entry:
        return None
    if isinstance(entry, dict) and entry.get("_stage_cache"):
        if params is None or entry.get("params", {}) == (params or {}):
            return entry.get("data")
        return None
    return entry


def group_samples_by_scene(samples: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(sample["scene_name"], []).append(sample)
    for scene_samples in grouped.values():
        scene_samples.sort(key=lambda sample: sample["timestamp"])
    return dict(sorted(grouped.items()))


def summarize_scene(scene_name: str, scene_samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    categories = set()
    annotation_count = 0
    media_channels = set()
    for sample in scene_samples:
        media_channels.update(sample.get("media", {}).keys())
        for annotation in sample.get("annotations", []):
            categories.add(annotation["category_name"])
            annotation_count += 1

    return {
        "scene_name": scene_name,
        "num_samples": len(scene_samples),
        "num_annotations": annotation_count,
        "media_channels": sorted(media_channels),
        "classes": sorted(categories),
        "first_timestamp": scene_samples[0]["timestamp"] if scene_samples else None,
        "last_timestamp": scene_samples[-1]["timestamp"] if scene_samples else None,
    }


def extract_scene_ego_motion(
    scene_samples: Sequence[Dict[str, Any]],
    channel: Optional[str] = None,
) -> Dict[str, Any]:
    """Pipeline-facing interface for 3D ego speed extraction."""
    return extract_ego_speed_3d(scene_samples, channel=channel)


def segment_scene_ego_motion(
    ego_motion: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pipeline-facing interface for forward/lateral ego motion masks."""
    return segment_ego_motion_signals(ego_motion, **(params or {}))


def extract_scene_object_motion(
    scene_samples: Sequence[Dict[str, Any]],
    ego_motion: Optional[Dict[str, Any]] = None,
    channel: Optional[str] = None,
) -> Dict[str, Any]:
    """Pipeline-facing interface for object-centric 3D motion tracks."""
    return extract_object_motion_3d(scene_samples, ego_motion=ego_motion, channel=channel)


def extract_segment_objects(
    ego_motion_segments: Dict[str, Any],
    object_motion: Dict[str, Any],
) -> Dict[str, Any]:
    """Pipeline-facing interface for segment-level object extraction."""
    return extract_objects_in_ego_segments(ego_motion_segments, object_motion)


def extract_scene_segment_predicates(
    segment_objects: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pipeline-facing interface for segment-level predicate extraction."""
    return extract_segment_predicates(segment_objects, **(params or {}))


def load_dataset(
    dataroot: Optional[Path] = None,
    version: str = "v1.0-trainval_meta",
    scene_names: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    camera: Optional[str] = "CAM_FRONT",
    include_sweeps: bool = False,
    media_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load nuScenes data for the pipeline.

    A nuScenes scene is treated as one video. By default this loads only
    CAM_FRONT media references, which keeps downstream work compact while still
    preserving all 3D annotations for each sample.
    """
    return load_nuscenes_data(
        dataroot=dataroot or default_dataroot(),
        version=version,
        scene_names=scene_names,
        max_samples=max_samples,
        camera=camera,
        include_sweeps=include_sweeps,
        include_annotations=True,
        media_root=media_root,
    )


def process_scene(
    scene_name: str,
    scene_samples: Sequence[Dict[str, Any]],
    ego_channel: Optional[str] = None,
    visualize_ego_motion: bool = True,
    ego_motion_fps: float = 2.0,
    motion_segment_params: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    Process one nuScenes scene/video.

    Logic reasoning is intentionally left empty for now. This function only
    saves the loaded scene data and a summary so future stages can plug in
    primitive extraction and pattern mining.
    """
    print(f"\n=== Processing nuScenes scene/video: {scene_name} ===")
    out_path = get_scene_output_path(scene_name, output_root)
    pipeline_data = load_pipeline_data(out_path, scene_name)

    summary_params = {"stage": "load_scene", "ego_channel": ego_channel}
    cached_summary = None if force_recompute else get_pipeline_stage(pipeline_data, "scene_summary", summary_params)
    # if cached_summary is not None:
    #     print(f"Loading cached scene summary: {get_pipeline_data_file(out_path)}")
    #     return cached_summary

    summary = summarize_scene(scene_name, scene_samples)
    ego_motion_params = {"stage": "ego_motion_3d", "ego_channel": ego_channel}
    ego_motion = extract_scene_ego_motion(scene_samples, channel=ego_channel)
    ego_speed_summary = summarize_ego_speed(ego_motion)
    summary["ego_speed"] = ego_speed_summary
    motion_segment_params = motion_segment_params or {}
    ego_motion_segments_params = {
        "stage": "ego_motion_segments",
        "ego_channel": ego_channel,
        **motion_segment_params,
    }
    ego_motion_segments = segment_scene_ego_motion(ego_motion, motion_segment_params)
    summary["ego_motion_segments"] = summarize_motion_segments(ego_motion_segments)
    object_motion_params = {"stage": "object_motion_3d", "ego_channel": ego_channel}
    object_motion = extract_scene_object_motion(scene_samples, ego_motion=ego_motion, channel=ego_channel)
    object_motion_summary = summarize_object_motion(object_motion)
    summary["object_motion"] = object_motion_summary
    segment_objects_params = {"stage": "objects_in_ego_segments", "ego_channel": ego_channel}
    segment_objects = extract_segment_objects(ego_motion_segments, object_motion)
    segment_objects_summary = summarize_segment_objects(segment_objects)
    summary["segment_objects"] = segment_objects_summary
    predicate_extract_params = {
        "object_moving_speed_threshold_mps": 0.5,
        "object_turn_heading_change_threshold_rad": 0.12,
        "object_relative_distance_change_threshold_m": 0.25,
        "object_lateral_velocity_threshold_mps": 0.2,
        "object_lateral_position_change_threshold_m": 0.3,
    }
    predicate_params = {
        "stage": "segment_predicates",
        "ego_channel": ego_channel,
        **predicate_extract_params,
    }
    segment_predicates = extract_scene_segment_predicates(segment_objects, predicate_extract_params)
    segment_predicates_summary = summarize_segment_predicates(segment_predicates)
    summary["segment_predicates"] = segment_predicates_summary

    scene_payload = {
        "scene_name": scene_name,
        "samples": list(scene_samples),
        "summary": summary,
    }

    set_pipeline_stage(pipeline_data, out_path, "loaded_scene", scene_payload, {"stage": "load_scene"})
    set_pipeline_stage(pipeline_data, out_path, "ego_motion_3d", ego_motion, ego_motion_params)
    set_pipeline_stage(pipeline_data, out_path, "ego_motion_segments", ego_motion_segments, ego_motion_segments_params)
    set_pipeline_stage(pipeline_data, out_path, "object_motion_3d", object_motion, object_motion_params)
    set_pipeline_stage(pipeline_data, out_path, "objects_in_ego_segments", segment_objects, segment_objects_params)
    set_pipeline_stage(pipeline_data, out_path, "segment_predicates", segment_predicates, predicate_params)
    set_pipeline_stage(pipeline_data, out_path, "scene_summary", summary, summary_params)

    summary_file = out_path / "scene_summary.json"
    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    ego_motion_file = out_path / "ego_motion_3d.json"
    with ego_motion_file.open("w", encoding="utf-8") as file:
        json.dump(ego_motion, file, indent=2)

    ego_motion_segments_file = out_path / "ego_motion_segments.json"
    with ego_motion_segments_file.open("w", encoding="utf-8") as file:
        json.dump(ego_motion_segments, file, indent=2)

    object_motion_file = out_path / "object_motion_3d.json"
    with object_motion_file.open("w", encoding="utf-8") as file:
        json.dump(object_motion, file, indent=2)

    segment_objects_file = out_path / "objects_in_ego_segments.json"
    with segment_objects_file.open("w", encoding="utf-8") as file:
        json.dump(segment_objects, file, indent=2)

    segment_predicates_file = out_path / "segment_predicates.json"
    with segment_predicates_file.open("w", encoding="utf-8") as file:
        json.dump(segment_predicates, file, indent=2)

    if visualize_ego_motion:
        if ego_channel is None or ego_channel not in CAMERA_CHANNELS:
            print("Skipping ego motion visualization because no camera channel is selected.")
        else:
            ego_vis_path = out_path / f"{scene_name}_{ego_channel}_ego_motion.mp4"
            render_ego_motion_video(
                scene_samples,
                ego_motion,
                ego_vis_path,
                channel=ego_channel,
                fps=ego_motion_fps,
                ego_motion_segments=ego_motion_segments,
                segment_objects=segment_objects,
                segment_predicates=segment_predicates,
            )
            summary["ego_motion_video"] = str(ego_vis_path)
            set_pipeline_stage(pipeline_data, out_path, "scene_summary", summary, summary_params)
            with summary_file.open("w", encoding="utf-8") as file:
                json.dump(summary, file, indent=2)
            print(f"Saved ego motion visualization to {ego_vis_path}")

    # TODO: Convert loaded nuScenes samples to object/ego primitive matrices.
    # TODO: Segment primitive signals.
    # TODO: Run logic reasoning / rule mining.

    print(
        f"Loaded {summary['num_samples']} samples, "
        f"{summary['num_annotations']} annotations, "
        f"{len(summary['classes'])} classes."
    )
    print(
        f"Ego speed: mean={ego_speed_summary['mean_speed_mps']:.2f} m/s, "
        f"max={ego_speed_summary['max_speed_mps']:.2f} m/s."
    )
    print(
        "Ego motion segments: "
        f"forward={summary['ego_motion_segments']['forward']['num_segments']}, "
        f"lateral={summary['ego_motion_segments']['lateral']['num_segments']}, "
        f"timeline={summary['ego_motion_segments']['timeline']['num_segments']}."
    )
    print(
        "Object motion: "
        f"tracks={object_motion_summary['num_tracks']}, "
        f"categories={object_motion_summary['num_categories']}, "
        f"full-span={object_motion_summary['full_span_track_count']}."
    )
    print(
        "Objects per ego segment: "
        f"segments={segment_objects_summary['num_segments']}, "
        f"mean_objects={segment_objects_summary['mean_objects_per_segment']:.1f}, "
        f"max_objects={segment_objects_summary['max_objects_per_segment']}."
    )
    print(
        "Segment predicates: "
        f"segments={segment_predicates_summary['num_segments']}, "
        f"object_predicates={segment_predicates_summary['num_object_predicates']}."
    )
    print(f"Saved scene cache to {get_pipeline_data_file(out_path)}")
    return summary


def run_pipeline(
    dataroot: Optional[Path] = None,
    version: str = "v1.0-trainval_meta/v1.0-trainval",
    scene_names: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    camera: Optional[str] = "CAM_FRONT",
    ego_channel: Optional[str] = None,
    visualize_ego_motion: bool = False,
    ego_motion_fps: float = 2.0,
    motion_segment_params: Optional[Dict[str, Any]] = None,
    include_sweeps: bool = False,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
    media_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Load nuScenes and process each scene/video one by one.
    """
    data = load_dataset(
        dataroot=dataroot,
        version=version,
        scene_names=scene_names,
        max_samples=max_samples,
        camera=camera,
        include_sweeps=include_sweeps,
        media_root=media_root,
    )

    scene_samples = group_samples_by_scene(data["samples"])
    print(f"Loaded nuScenes dataset from {data['dataroot']}")
    if data.get("media_root") and data["media_root"] != data["dataroot"]:
        print(f"Media files resolved from {data['media_root']}")
    print(f"Scenes/videos to process: {len(scene_samples)}")
    print(f"Samples loaded: {data['num_samples']}")
    print(f"Annotations loaded: {data['num_annotations']}")

    summaries = []
    for scene_name, samples in scene_samples.items():
        summary = process_scene(
            scene_name,
            samples,
            ego_channel=ego_channel or camera,
            visualize_ego_motion=visualize_ego_motion,
            ego_motion_fps=ego_motion_fps,
            motion_segment_params=motion_segment_params,
            output_root=output_root,
            force_recompute=force_recompute,
        )
        summaries.append(summary)

    manifest = {
        "dataroot": data["dataroot"],
        "media_root": data.get("media_root", data["dataroot"]),
        "version": version,
        "camera": camera,
        "ego_channel": ego_channel or camera,
        "visualize_ego_motion": visualize_ego_motion,
        "ego_motion_fps": ego_motion_fps,
        "motion_segment_params": motion_segment_params or {},
        "include_sweeps": include_sweeps,
        "num_scenes": len(summaries),
        "num_samples": data["num_samples"],
        "num_annotations": data["num_annotations"],
        "scenes": summaries,
    }
    manifest_path = get_output_root(output_root) / "pipeline_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"\nSaved pipeline manifest to {manifest_path}")

    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nuScenes experiment pipeline skeleton.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file for the nuScenes pipeline.")
    parser.add_argument("--dataroot", type=Path, default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--camera", default=None, help="Camera channel to load, or 'all' for all keyframe sensors.")
    parser.add_argument("--ego-channel", default=None, help="Sample_data channel for ego speed extraction. Defaults to --camera.")
    parser.add_argument("--visualize-ego-motion", action="store_true", default=None, help="Save an MP4 showing video plus x/y/z ego velocity charts.")
    parser.add_argument("--ego-motion-fps", type=float, default=None)
    parser.add_argument("--forward-stop-threshold", type=float, default=None, help="Forward speed below this m/s is labelled stop.")
    parser.add_argument("--forward-accel-threshold", type=float, default=None, help="Forward acceleration threshold in m/s^2 for speed_up/slow_down.")
    parser.add_argument("--lateral-speed-threshold", type=float, default=None, help="Lateral speed threshold in m/s for left/right labels.")
    parser.add_argument("--motion-min-segment-frames", type=int, default=None, help="Shorter label runs are merged into neighbors.")
    parser.add_argument("--motion-smoothing-window", type=int, default=None, help="Odd moving-average window used before labelling.")
    parser.add_argument("--include-sweeps", action="store_true", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--media-root", type=Path, default=None, help="Root for resolving camera image files when blobs are in a separate folder from metadata.")
    parser.add_argument("--force-recompute", action="store_true", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cfg = resolve_run_config(args)

    dataroot = _resolve_project_path(run_cfg["dataroot"])
    version = str(run_cfg["version"])
    scene_names = list(run_cfg.get("scene_names", []))
    max_samples = run_cfg.get("max_samples")
    camera_raw = run_cfg.get("camera")
    camera = None if camera_raw is None or str(camera_raw).lower() == "all" else str(camera_raw).upper()
    if camera is not None and camera not in CAMERA_CHANNELS:
        raise ValueError(f"Unknown camera channel {camera!r}. Expected one of {sorted(CAMERA_CHANNELS)} or 'all'.")

    ego_channel_raw = run_cfg.get("ego_channel")
    ego_channel = str(ego_channel_raw).upper() if ego_channel_raw else camera
    if ego_channel is not None and ego_channel not in CAMERA_CHANNELS and ego_channel != "LIDAR_TOP":
        raise ValueError(
            f"Unknown ego channel {ego_channel!r}. Expected one of {sorted(CAMERA_CHANNELS)} or 'LIDAR_TOP'."
        )
    motion_segment_params = dict(run_cfg["motion_segment_params"])
    output_root = _resolve_project_path(run_cfg.get("output_root"))
    media_root = _resolve_project_path(run_cfg.get("media_root"))

    run_pipeline(
        dataroot=dataroot,
        version=version,
        scene_names=scene_names,
        max_samples=max_samples,
        camera=camera,
        ego_channel=ego_channel,
        visualize_ego_motion=bool(run_cfg["visualize_ego_motion"]),
        ego_motion_fps=float(run_cfg["ego_motion_fps"]),
        motion_segment_params=motion_segment_params,
        include_sweeps=bool(run_cfg["include_sweeps"]),
        output_root=output_root,
        force_recompute=bool(run_cfg["force_recompute"]),
        media_root=media_root,
    )


if __name__ == "__main__":
    main()
