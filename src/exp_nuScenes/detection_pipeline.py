"""
Object detection pipeline for nuScenes scenes.

This pipeline supplements the annotation-based pipeline.py with open-vocabulary
detections from a pretrained object detector, enabling discovery of object classes
beyond those in the nuScenes annotation taxonomy.

High-level flow:
    1. Load nuScenes dataset (reuses load_dataset.py).
    2. Iterate scenes → keyframe samples → camera frames.
    3. Pass each frame image path to a pluggable ObjectDetector.
    4. Collect per-frame detections and aggregate per-scene.
    5. Save results as JSON for downstream use.

The detector is intentionally decoupled: implement the ObjectDetector ABC and
pass an instance to run_detection_pipeline() or via --detector-class on the CLI.

Usage example:
    python src/exp_nuScenes/detection_pipeline.py \\
        --config configs/exp_nuScenes/default.yaml \\
        --dataroot dataset/nuScenes \\
        --version v1.0-trainval_meta/v1.0-trainval \\
        --media-root dataset/nuScenes/v1.0-trainval01_blobs_camera \\
        --camera CAM_FRONT
"""

from __future__ import annotations

import argparse
import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.exp_driving_videos.modules.pipe_utils.exp_driving_utils import load_pattern_cfg_file
from src.exp_nuScenes.load_dataset import (
    CAMERA_CHANNELS,
    default_dataroot,
    load_nuscenes_data,
)
from src.exp_nuScenes.pipeline import (
    _deep_merge_dicts,
    _resolve_project_path,
    group_samples_by_scene,
)


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_DETECTION_CONFIG: Dict[str, Any] = {
    "dataroot": "dataset/nuScenes",
    "version": "v1.0-trainval_meta/v1.0-trainval",
    "media_root": None,
    "scene_names": [],
    "max_samples": None,
    "camera": "CAM_FRONT",
    "include_sweeps": False,
    "output_root": None,
    "force_recompute": False,
    "detector": {
        "name": "yolo_world",
        "model": "yolov8s-worldv2.pt",
        "confidence_threshold": 0.3,
        "nms_iou_threshold": 0.5,
        # Custom text classes for YOLO-World's open-vocabulary mode.
        # If empty, the model's built-in COCO-80 classes are used.
        "classes": [],
    },
}

# Default broad class vocabulary for open-vocabulary detection.
# These are in addition to the model's built-in classes; override via
# --classes on the CLI or detector.classes in the YAML config.
YOLO_WORLD_DEFAULT_CLASSES: List[str] = [
    # Vehicles
    "car", "truck", "bus", "motorcycle", "bicycle", "van", "trailer",
    "emergency vehicle", "police car", "ambulance", "fire truck",
    "construction vehicle", "forklift", "tractor", "wheel loader",
    # Micro-mobility
    "scooter", "skateboard", "wheelchair",
    # People
    "person", "pedestrian", "cyclist", "motorcyclist", "child",
    "construction worker",
    # Outdoor infrastructure
    "traffic cone", "traffic barrier", "bollard", "guardrail",
    "traffic light", "stop sign", "speed bump",
    # Animals
    "dog", "cat", "horse", "cow", "bird",
    # Debris / movable objects
    "debris", "garbage bag", "shopping cart", "stroller", "suitcase",
    "box", "pallet",
    # Buildings and structures
    "building", "house", "store", "gas station", "parking structure",
    "bridge", "overpass", "underpass", "tunnel", "wall",
    # Vegetation
    "tree", "bush", "hedge", "plant", "grass", "shrub",
    # Street furniture
    "bench", "lamp post", "street light", "telephone pole", "power line",
    "mailbox", "fire hydrant", "street sign", "parking meter", "railing",
    # Sidewalk and road features
    "sidewalk", "curb", "crosswalk", "parking lot", "road marking",
]

# ---------------------------------------------------------------------------
# Detector interface
# ---------------------------------------------------------------------------

class DetectionResult:
    """Holds detection output for one image frame."""

    def __init__(
        self,
        image_path: str,
        boxes: List[List[float]],
        scores: List[float],
        labels: List[str],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            image_path: Absolute path to the source image.
            boxes: List of [x1, y1, x2, y2] bounding boxes in pixel coordinates.
            scores: Confidence score per box (same length as boxes).
            labels: Class label string per box (same length as boxes).
            extra: Optional dict of additional detector-specific metadata.
        """
        self.image_path = image_path
        self.boxes = boxes
        self.scores = scores
        self.labels = labels
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "num_detections": len(self.boxes),
            "boxes": self.boxes,
            "scores": self.scores,
            "labels": self.labels,
            **self.extra,
        }


class ObjectDetector(ABC):
    """Abstract base class for pluggable object detectors."""

    @abstractmethod
    def detect(self, image_path: str) -> DetectionResult:
        """Run detection on a single image file and return a DetectionResult."""

    def detect_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """Run detection on a list of image paths. Default: calls detect() per image."""
        return [self.detect(path) for path in image_paths]

    def warmup(self) -> None:
        """Optional: called once before processing starts."""

    def teardown(self) -> None:
        """Optional: called once after all processing is complete."""


class StubDetector(ObjectDetector):
    """
    No-op detector that returns empty results.
    Used as a placeholder until a real detector is plugged in.
    """

    def detect(self, image_path: str) -> DetectionResult:
        return DetectionResult(image_path=image_path, boxes=[], scores=[], labels=[])


# ---------------------------------------------------------------------------
# YOLO-World detector
# ---------------------------------------------------------------------------

class YOLOWorldDetector(ObjectDetector):
    """
    Object detector backed by YOLO-World (ultralytics).

    YOLO-World supports open-vocabulary detection: call set_classes() with any
    list of text prompts and the model will detect matching objects.

    Args:
        model_name: Ultralytics YOLO-World checkpoint, e.g. "yolov8s-worldv2.pt"
            or "yolov8l-worldv2.pt" for higher accuracy at the cost of speed.
        classes: List of text class names to detect.  If empty, the model's
            built-in COCO-80 vocabulary is used unchanged.
        confidence_threshold: Minimum score to keep a detection.
        nms_iou_threshold: IoU threshold for NMS suppression.
        device: Inference device override, e.g. "cpu", "cuda:0", "mps".
            If None, ultralytics auto-selects.
    """

    def __init__(
        self,
        model_name: str = "yolov8s-worldv2.pt",
        classes: Optional[List[str]] = None,
        confidence_threshold: float = 0.3,
        nms_iou_threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.classes = list(classes) if classes else []
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self._model: Any = None  # loaded lazily in warmup()

    def warmup(self) -> None:
        """Load model weights and set vocabulary once before processing starts."""
        from ultralytics import YOLOWorld  # deferred import

        print(f"[YOLOWorldDetector] Loading model: {self.model_name}")
        kwargs: Dict[str, Any] = {}
        if self.device is not None:
            kwargs["device"] = self.device
        self._model = YOLOWorld(self.model_name, **kwargs)

        if self.classes:
            self._model.set_classes(self.classes)
            print(f"[YOLOWorldDetector] Custom vocabulary ({len(self.classes)} classes): "
                  f"{self.classes[:10]}{'...' if len(self.classes) > 10 else ''}")
        else:
            print("[YOLOWorldDetector] Using built-in COCO-80 vocabulary.")

    def teardown(self) -> None:
        self._model = None

    def detect(self, image_path: str) -> DetectionResult:
        if self._model is None:
            self.warmup()

        results = self._model.predict(
            source=image_path,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            verbose=False,
        )

        boxes: List[List[float]] = []
        scores: List[float] = []
        labels: List[str] = []

        for r in results:
            if r.boxes is None:
                continue
            names: Dict[int, str] = r.names  # {class_id: class_name}
            xyxy = r.boxes.xyxy.cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            cls_ids = r.boxes.cls.cpu().tolist()
            for box, conf, cls_id in zip(xyxy, confs, cls_ids):
                boxes.append([round(v, 2) for v in box])
                scores.append(round(float(conf), 4))
                labels.append(names.get(int(cls_id), str(int(cls_id))))

        return DetectionResult(
            image_path=image_path,
            boxes=boxes,
            scores=scores,
            labels=labels,
            extra={"model": self.model_name},
        )

    def detect_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """Run all images in a single model.predict() call for efficiency."""
        if self._model is None:
            self.warmup()

        all_results = self._model.predict(
            source=image_paths,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            verbose=False,
        )

        detection_results: List[DetectionResult] = []
        for image_path, r in zip(image_paths, all_results):
            boxes: List[List[float]] = []
            scores: List[float] = []
            labels: List[str] = []
            if r.boxes is not None:
                names: Dict[int, str] = r.names
                for box, conf, cls_id in zip(
                    r.boxes.xyxy.cpu().tolist(),
                    r.boxes.conf.cpu().tolist(),
                    r.boxes.cls.cpu().tolist(),
                ):
                    boxes.append([round(v, 2) for v in box])
                    scores.append(round(float(conf), 4))
                    labels.append(names.get(int(cls_id), str(int(cls_id))))
            detection_results.append(
                DetectionResult(
                    image_path=image_path,
                    boxes=boxes,
                    scores=scores,
                    labels=labels,
                    extra={"model": self.model_name},
                )
            )
        return detection_results

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def get_detection_output_root(output_root: Optional[Path] = None) -> Path:
    if output_root is not None:
        root = Path(output_root)
    else:
        root = config.get_output_path("pipeline_output") / "nuScenes_detection"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_scene_detection_path(scene_name: str, output_root: Optional[Path] = None) -> Path:
    out = get_detection_output_root(output_root) / scene_name
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------

def process_scene_detections(
    scene_name: str,
    scene_samples: Sequence[Dict[str, Any]],
    detector: ObjectDetector,
    camera: str = "CAM_FRONT",
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    Run the detector on every keyframe of one scene for the given camera channel.

    Returns a dict with scene-level detection results and a per-frame list.
    """
    out_path = get_scene_detection_path(scene_name, output_root)
    detections_file = out_path / f"detections_{camera}.json"

    if not force_recompute and detections_file.exists():
        print(f"  [cache] Loading existing detections from {detections_file}")
        with detections_file.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    print(f"\n=== Detection: {scene_name} | {camera} ({len(scene_samples)} frames) ===")

    frame_results: List[Dict[str, Any]] = []
    all_labels: List[str] = []
    missing_frames = 0

    for sample in sorted(scene_samples, key=lambda s: s["timestamp"]):
        media = sample.get("media", {})
        if camera not in media:
            missing_frames += 1
            continue

        image_path = media[camera].get("path", "")
        if not image_path or not Path(image_path).exists():
            missing_frames += 1
            continue

        result = detector.detect(image_path)
        all_labels.extend(result.labels)
        frame_results.append({
            "sample_token": sample["sample_token"],
            "timestamp": sample["timestamp"],
            "scene_name": scene_name,
            "camera": camera,
            **result.to_dict(),
        })

    from collections import Counter
    label_counts = dict(Counter(all_labels).most_common())

    scene_result: Dict[str, Any] = {
        "scene_name": scene_name,
        "camera": camera,
        "num_frames_processed": len(frame_results),
        "num_frames_missing": missing_frames,
        "num_total_detections": sum(len(f["boxes"]) for f in frame_results),
        "detected_classes": label_counts,
        "frames": frame_results,
    }

    with detections_file.open("w", encoding="utf-8") as fh:
        json.dump(scene_result, fh, indent=2)
    print(
        f"  Processed {len(frame_results)} frames, "
        f"{scene_result['num_total_detections']} detections, "
        f"{len(label_counts)} unique classes."
    )
    if missing_frames:
        print(f"  Skipped {missing_frames} frames (image file not found).")
    return scene_result


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_detection_pipeline(
    dataroot: Optional[Path] = None,
    version: str = "v1.0-trainval_meta/v1.0-trainval",
    scene_names: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    camera: str = "CAM_FRONT",
    include_sweeps: bool = False,
    media_root: Optional[Path] = None,
    detector: Optional[ObjectDetector] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load nuScenes data and run object detection over all scenes.

    Args:
        dataroot: nuScenes dataset root (contains version sub-folder).
        version: Metadata version folder, e.g. "v1.0-trainval_meta/v1.0-trainval".
        scene_names: Optional list of scene names to process. All scenes if empty.
        max_samples: Optional cap on number of samples loaded (for debugging).
        camera: Camera channel to use for detection, e.g. "CAM_FRONT".
        include_sweeps: If True, also process non-keyframe sample_data rows.
        media_root: Root for resolving image file paths when blobs are in a
            separate folder from metadata.
        detector: ObjectDetector instance. Defaults to StubDetector if None.
        output_root: Root folder for detection output JSON files.
        force_recompute: Re-run detection even if cached output exists.

    Returns:
        List of per-scene result dicts (one per scene).
    """
    if detector is None:
        detector = StubDetector()
        print("[detection_pipeline] No detector provided — using StubDetector (empty results).")

    effective_dataroot = Path(dataroot).resolve() if dataroot else default_dataroot()
    print(f"Loading nuScenes data from {effective_dataroot} / {version}")
    data = load_nuscenes_data(
        dataroot=effective_dataroot,
        version=version,
        scene_names=list(scene_names or []),
        max_samples=max_samples,
        camera=camera,
        include_sweeps=include_sweeps,
        include_annotations=False,
        media_root=media_root,
    )
    print(
        f"Loaded {data['num_scenes']} scenes, "
        f"{data['num_samples']} samples."
    )

    scenes_by_name = group_samples_by_scene(data["samples"])

    detector.warmup()
    scene_results: List[Dict[str, Any]] = []
    try:
        for scene_name, scene_samples in scenes_by_name.items():
            result = process_scene_detections(
                scene_name=scene_name,
                scene_samples=scene_samples,
                detector=detector,
                camera=camera,
                output_root=output_root,
                force_recompute=force_recompute,
            )
            scene_results.append(result)
    finally:
        detector.teardown()

    # Write a pipeline-level manifest
    out_root = get_detection_output_root(output_root)
    manifest = {
        "dataroot": str(effective_dataroot),
        "version": version,
        "camera": camera,
        "num_scenes": len(scene_results),
        "scenes": [
            {
                "scene_name": r["scene_name"],
                "num_frames_processed": r["num_frames_processed"],
                "num_total_detections": r["num_total_detections"],
                "num_detected_classes": len(r["detected_classes"]),
            }
            for r in scene_results
        ],
    }
    manifest_path = out_root / "detection_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nSaved detection manifest to {manifest_path}")

    return scene_results


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def load_detection_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return dict(DEFAULT_DETECTION_CONFIG)
    loaded = load_pattern_cfg_file(config_path) or {}
    if "scene" in loaded and "scene_names" not in loaded:
        loaded["scene_names"] = loaded["scene"]
    return _deep_merge_dicts(DEFAULT_DETECTION_CONFIG, loaded)


def resolve_detection_run_config(args: argparse.Namespace) -> Dict[str, Any]:
    run_cfg = load_detection_config(getattr(args, "config", None))

    for attr, key in [
        ("dataroot", "dataroot"),
        ("version", "version"),
        ("media_root", "media_root"),
        ("output_root", "output_root"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            run_cfg[key] = str(val)

    if getattr(args, "scene", None):
        run_cfg["scene_names"] = list(args.scene)
    if getattr(args, "max_samples", None) is not None:
        run_cfg["max_samples"] = args.max_samples
    if getattr(args, "camera", None) is not None:
        run_cfg["camera"] = args.camera
    if getattr(args, "include_sweeps", None) is not None:
        run_cfg["include_sweeps"] = args.include_sweeps
    if getattr(args, "force_recompute", None) is not None:
        run_cfg["force_recompute"] = args.force_recompute
    if getattr(args, "confidence_threshold", None) is not None:
        run_cfg["detector"]["confidence_threshold"] = args.confidence_threshold
    if getattr(args, "yolo_model", None) is not None:
        run_cfg["detector"]["model"] = args.yolo_model
    if getattr(args, "classes", None):
        run_cfg["detector"]["classes"] = args.classes
    if getattr(args, "device", None) is not None:
        run_cfg["detector"]["device"] = args.device

    return run_cfg


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run object detection over nuScenes scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config file.")
    parser.add_argument("--dataroot", type=Path, default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--media-root", type=Path, default=None, dest="media_root")
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--include-sweeps", action="store_true", default=None)
    parser.add_argument("--output-root", type=Path, default=None, dest="output_root")
    parser.add_argument("--force-recompute", action="store_true", default=None, dest="force_recompute")
    parser.add_argument("--confidence-threshold", type=float, default=None, dest="confidence_threshold")
    # YOLO-World specific
    parser.add_argument(
        "--yolo-model",
        default=None,
        dest="yolo_model",
        help="YOLO-World checkpoint, e.g. yolov8s-worldv2.pt (small/fast) or yolov8l-worldv2.pt (large/accurate).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Custom class list for open-vocabulary detection. Overrides the default vocabulary.",
    )
    parser.add_argument(
        "--use-default-classes",
        action="store_true",
        dest="use_default_classes",
        help="Use the built-in YOLO_WORLD_DEFAULT_CLASSES vocabulary (~50 driving-relevant classes).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device override, e.g. cpu, cuda:0, mps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cfg = resolve_detection_run_config(args)

    dataroot = _resolve_project_path(run_cfg.get("dataroot"))
    version = str(run_cfg["version"])
    media_root = _resolve_project_path(run_cfg.get("media_root"))
    output_root = _resolve_project_path(run_cfg.get("output_root"))
    scene_names = list(run_cfg.get("scene_names", []))
    max_samples = run_cfg.get("max_samples")
    camera_raw = run_cfg.get("camera")
    camera = str(camera_raw).upper() if camera_raw else "CAM_FRONT"
    if camera not in CAMERA_CHANNELS:
        raise ValueError(f"Unknown camera channel {camera!r}. Expected one of {sorted(CAMERA_CHANNELS)}.")

    det_cfg = run_cfg.get("detector", {})

    # Resolve class vocabulary
    classes: List[str] = list(det_cfg.get("classes", []))
    if not classes and getattr(args, "use_default_classes", False):
        classes = list(YOLO_WORLD_DEFAULT_CLASSES)
    if not classes:
        # Default: use the curated driving vocabulary
        classes = list(YOLO_WORLD_DEFAULT_CLASSES)

    detector: ObjectDetector = YOLOWorldDetector(
        model_name=str(det_cfg.get("model", "yolov8s-worldv2.pt")),
        classes=classes,
        confidence_threshold=float(det_cfg.get("confidence_threshold", 0.3)),
        nms_iou_threshold=float(det_cfg.get("nms_iou_threshold", 0.5)),
        device=det_cfg.get("device") or getattr(args, "device", None),
    )

    run_detection_pipeline(
        dataroot=dataroot,
        version=version,
        scene_names=scene_names,
        max_samples=max_samples,
        camera=camera,
        include_sweeps=bool(run_cfg.get("include_sweeps", False)),
        media_root=media_root,
        detector=detector,
        output_root=output_root,
        force_recompute=bool(run_cfg.get("force_recompute", False)),
    )


if __name__ == "__main__":
    main()
