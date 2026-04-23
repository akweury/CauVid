"""
Utilities for loading the local nuScenes dataset.

The functions here read the raw nuScenes JSON tables directly, so the official
nuScenes devkit is not required. The default path targets the local mini subset:

    dataset/nuScenes/v1.0-mini

Main entry point:

    data = load_nuscenes_data()
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


CAMERA_CHANNELS = {
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
}


@dataclass(frozen=True)
class NuScenesTables:
    scenes: List[Dict[str, Any]]
    samples: List[Dict[str, Any]]
    sample_annotations: List[Dict[str, Any]]
    sample_data: List[Dict[str, Any]]
    instances: List[Dict[str, Any]]
    categories: List[Dict[str, Any]]
    attributes: List[Dict[str, Any]]
    visibility: List[Dict[str, Any]]
    calibrated_sensors: List[Dict[str, Any]]
    sensors: List[Dict[str, Any]]
    ego_poses: List[Dict[str, Any]]


def default_dataroot() -> Path:
    """Return the repo-local default nuScenes mini dataroot."""
    project_root = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("CAUVID_NUSCENES_PATH", project_root / "dataset" / "nuScenes" / "v1.0-mini"))


def load_table(meta_dir: Path, table_name: str) -> List[Dict[str, Any]]:
    """Load one nuScenes JSON table by name."""
    table_path = meta_dir / f"{table_name}.json"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing nuScenes table: {table_path}")
    with table_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_nuscenes_tables(dataroot: Optional[Path] = None, version: str = "v1.0-mini") -> NuScenesTables:
    """Load the raw nuScenes metadata tables."""
    dataroot = Path(dataroot) if dataroot is not None else default_dataroot()
    meta_dir = dataroot / version
    if not meta_dir.exists():
        raise FileNotFoundError(
            f"nuScenes metadata folder not found: {meta_dir}. "
            "Expected a folder like dataset/nuScenes/v1.0-mini/v1.0-mini."
        )

    return NuScenesTables(
        scenes=load_table(meta_dir, "scene"),
        samples=load_table(meta_dir, "sample"),
        sample_annotations=load_table(meta_dir, "sample_annotation"),
        sample_data=load_table(meta_dir, "sample_data"),
        instances=load_table(meta_dir, "instance"),
        categories=load_table(meta_dir, "category"),
        attributes=load_table(meta_dir, "attribute"),
        visibility=load_table(meta_dir, "visibility"),
        calibrated_sensors=load_table(meta_dir, "calibrated_sensor"),
        sensors=load_table(meta_dir, "sensor"),
        ego_poses=load_table(meta_dir, "ego_pose"),
    )


def index_by_token(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["token"]: row for row in rows}


def group_by(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


def resolve_path(dataroot: Path, filename: str) -> Path:
    """Resolve a nuScenes relative filename to a local OS path."""
    return dataroot.joinpath(*filename.split("/"))


def top_category(category_name: str) -> str:
    parts = category_name.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return category_name


def sample_sequence_for_scene(scene: Dict[str, Any], sample_by_token: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the ordered sample/keyframe sequence for one scene."""
    sequence = []
    token = scene["first_sample_token"]
    while token:
        sample = sample_by_token[token]
        sequence.append(sample)
        token = sample.get("next", "")
    return sequence


def select_samples(
    scenes: Sequence[Dict[str, Any]],
    sample_by_token: Dict[str, Dict[str, Any]],
    scene_names: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Select ordered samples from requested scenes."""
    requested = set(scene_names or [])
    selected = []
    for scene in sorted(scenes, key=lambda row: row["name"]):
        if requested and scene["name"] not in requested:
            continue
        selected.extend(sample_sequence_for_scene(scene, sample_by_token))
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def build_annotation_record(
    annotation: Dict[str, Any],
    instance_by_token: Dict[str, Dict[str, Any]],
    category_by_token: Dict[str, Dict[str, Any]],
    attribute_by_token: Dict[str, Dict[str, Any]],
    visibility_by_token: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Join a sample_annotation row with category, attribute, and visibility details."""
    instance = instance_by_token[annotation["instance_token"]]
    category = category_by_token[instance["category_token"]]
    category_name = category["name"]
    attributes = [
        attribute_by_token[token]["name"]
        for token in annotation.get("attribute_tokens", [])
        if token in attribute_by_token
    ]
    visibility = visibility_by_token.get(annotation.get("visibility_token", ""), {})
    translation = annotation["translation"]
    size = annotation["size"]

    return {
        "annotation_token": annotation["token"],
        "sample_token": annotation["sample_token"],
        "instance_token": annotation["instance_token"],
        "category_name": category_name,
        "top_category": top_category(category_name),
        "attribute_names": attributes,
        "visibility_token": annotation.get("visibility_token", ""),
        "visibility_level": visibility.get("level", ""),
        "visibility_description": visibility.get("description", ""),
        "translation": translation,
        "translation_x": translation[0],
        "translation_y": translation[1],
        "translation_z": translation[2],
        "size": size,
        "size_width": size[0],
        "size_length": size[1],
        "size_height": size[2],
        "rotation_wxyz": annotation["rotation"],
        "num_lidar_pts": annotation["num_lidar_pts"],
        "num_radar_pts": annotation["num_radar_pts"],
        "prev": annotation.get("prev", ""),
        "next": annotation.get("next", ""),
    }


def build_media_record(
    sample_data: Dict[str, Any],
    dataroot: Path,
    calibrated_sensor_by_token: Dict[str, Dict[str, Any]],
    sensor_by_token: Dict[str, Dict[str, Any]],
    ego_pose_by_token: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Join a sample_data row with sensor calibration and ego-pose details."""
    calibrated_sensor = calibrated_sensor_by_token[sample_data["calibrated_sensor_token"]]
    sensor = sensor_by_token[calibrated_sensor["sensor_token"]]
    ego_pose = ego_pose_by_token[sample_data["ego_pose_token"]]
    path = resolve_path(dataroot, sample_data["filename"])

    return {
        "sample_data_token": sample_data["token"],
        "sample_token": sample_data["sample_token"],
        "channel": sensor["channel"],
        "modality": sensor["modality"],
        "filename": sample_data["filename"],
        "path": str(path),
        "exists": path.exists(),
        "timestamp": sample_data["timestamp"],
        "fileformat": sample_data["fileformat"],
        "is_key_frame": sample_data["is_key_frame"],
        "width": sample_data["width"],
        "height": sample_data["height"],
        "calibrated_sensor": {
            "translation": calibrated_sensor["translation"],
            "rotation": calibrated_sensor["rotation"],
            "camera_intrinsic": calibrated_sensor.get("camera_intrinsic", []),
        },
        "ego_pose": {
            "translation": ego_pose["translation"],
            "rotation": ego_pose["rotation"],
        },
    }


def make_class_summary(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Summarize category coverage across loaded samples."""
    annotation_counts: Counter[str] = Counter()
    instance_sets: Dict[str, set] = defaultdict(set)
    sample_sets: Dict[str, set] = defaultdict(set)
    scene_sets: Dict[str, set] = defaultdict(set)

    for sample in samples:
        for annotation in sample["annotations"]:
            category = annotation["category_name"]
            annotation_counts[category] += 1
            instance_sets[category].add(annotation["instance_token"])
            sample_sets[category].add(sample["sample_token"])
            scene_sets[category].add(sample["scene_name"])

    return [
        {
            "category_name": category,
            "top_category": top_category(category),
            "annotation_count": count,
            "instance_count": len(instance_sets[category]),
            "sample_count": len(sample_sets[category]),
            "scene_count": len(scene_sets[category]),
        }
        for category, count in sorted(annotation_counts.items())
    ]


def load_nuscenes_data(
    dataroot: Optional[Path] = None,
    version: str = "v1.0-mini",
    scene_names: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    camera: Optional[str] = None,
    include_sweeps: bool = False,
    include_annotations: bool = True,
) -> Dict[str, Any]:
    """
    Load local nuScenes data into a simple Python dictionary.

    Args:
        dataroot: Path containing samples/, sweeps/, maps/, and v1.0-mini/.
        version: Metadata table folder, e.g. "v1.0-mini".
        scene_names: Optional scene names such as ["scene-0061"].
        max_samples: Optional sample/keyframe limit after scene filtering.
        camera: Optional camera channel filter, e.g. "CAM_FRONT".
        include_sweeps: Include non-keyframe sample_data rows if True.
        include_annotations: Include joined sample annotations if True.

    Returns:
        Dict with dataroot, version, scenes, samples, classes, class_summary.
    """
    dataroot = Path(dataroot) if dataroot is not None else default_dataroot()
    dataroot = dataroot.resolve()
    tables = load_nuscenes_tables(dataroot, version)

    sample_by_token = index_by_token(tables.samples)
    scene_by_token = index_by_token(tables.scenes)
    instance_by_token = index_by_token(tables.instances)
    category_by_token = index_by_token(tables.categories)
    attribute_by_token = index_by_token(tables.attributes)
    visibility_by_token = {row["token"]: row for row in tables.visibility}
    calibrated_sensor_by_token = index_by_token(tables.calibrated_sensors)
    sensor_by_token = index_by_token(tables.sensors)
    ego_pose_by_token = index_by_token(tables.ego_poses)
    annotations_by_sample = group_by(tables.sample_annotations, "sample_token")
    sample_data_by_sample = group_by(tables.sample_data, "sample_token")

    if camera is not None:
        camera = camera.upper()
        if camera not in CAMERA_CHANNELS:
            raise ValueError(f"Unknown camera channel {camera!r}. Expected one of {sorted(CAMERA_CHANNELS)}")

    selected = select_samples(tables.scenes, sample_by_token, scene_names, max_samples)
    loaded_samples = []

    for sample in selected:
        scene = scene_by_token[sample["scene_token"]]

        media = {}
        for row in sample_data_by_sample.get(sample["token"], []):
            if not include_sweeps and not row["is_key_frame"]:
                continue
            media_record = build_media_record(
                row,
                dataroot,
                calibrated_sensor_by_token,
                sensor_by_token,
                ego_pose_by_token,
            )
            if camera is not None and media_record["channel"] != camera:
                continue
            media[media_record["channel"]] = media_record

        annotations = []
        if include_annotations:
            for annotation in annotations_by_sample.get(sample["token"], []):
                annotations.append(
                    build_annotation_record(
                        annotation,
                        instance_by_token,
                        category_by_token,
                        attribute_by_token,
                        visibility_by_token,
                    )
                )

        loaded_samples.append(
            {
                "sample_token": sample["token"],
                "timestamp": sample["timestamp"],
                "scene_token": sample["scene_token"],
                "scene_name": scene["name"],
                "scene_description": scene["description"],
                "prev": sample["prev"],
                "next": sample["next"],
                "media": media,
                "annotations": annotations,
            }
        )

    selected_scene_names = sorted({sample["scene_name"] for sample in loaded_samples})
    selected_scene_rows = [
        {
            "scene_token": scene["token"],
            "scene_name": scene["name"],
            "description": scene["description"],
            "nbr_samples": scene["nbr_samples"],
            "first_sample_token": scene["first_sample_token"],
            "last_sample_token": scene["last_sample_token"],
        }
        for scene in sorted(tables.scenes, key=lambda row: row["name"])
        if scene["name"] in selected_scene_names
    ]
    class_summary = make_class_summary(loaded_samples) if include_annotations else []

    return {
        "dataroot": str(dataroot),
        "version": version,
        "num_scenes": len(selected_scene_rows),
        "num_samples": len(loaded_samples),
        "num_annotations": sum(len(sample["annotations"]) for sample in loaded_samples),
        "classes": [row["category_name"] for row in class_summary],
        "class_summary": class_summary,
        "scenes": selected_scene_rows,
        "samples": loaded_samples,
    }


def get_scene_names(dataroot: Optional[Path] = None, version: str = "v1.0-mini") -> List[str]:
    """Return available scene names."""
    tables = load_nuscenes_tables(dataroot, version)
    return [scene["name"] for scene in sorted(tables.scenes, key=lambda row: row["name"])]


def get_camera_channels() -> List[str]:
    """Return supported nuScenes camera channels."""
    return sorted(CAMERA_CHANNELS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load local nuScenes data and print a summary.")
    parser.add_argument("--dataroot", type=Path, default=default_dataroot())
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--include-sweeps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data = load_nuscenes_data(
        dataroot=args.dataroot,
        version=args.version,
        scene_names=args.scene,
        max_samples=args.max_samples,
        camera=args.camera,
        include_sweeps=args.include_sweeps,
    )
    print(f"Loaded nuScenes from {data['dataroot']}")
    print(f"Scenes: {data['num_scenes']}")
    print(f"Samples: {data['num_samples']}")
    print(f"Annotations: {data['num_annotations']}")
    print(f"Classes ({len(data['classes'])}): {', '.join(data['classes'])}")


if __name__ == "__main__":
    main()
