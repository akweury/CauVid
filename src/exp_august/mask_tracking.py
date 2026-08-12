"""Detector-guided SAM 2 mask tracking for the August pipeline.

The July tracker remains the bootstrap and compatibility fallback.  When a
local SAM 2 checkpoint is available, its box tracks seed independent SAM 2
video propagations.  Per-frame masklets are then reconciled with detector
observations by a Hungarian assignment whose available cues can include mask
overlap, box overlap, semantic class, RAFT flow, and depth.

This module deliberately never turns a bounding box into a claimed SAM mask.
If the checkpoint or runtime is unavailable, the result is explicitly marked
as a ByteTrack fallback in its provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional rendering/runtime dependency
    cv2 = None

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - guarded when the hybrid backend runs
    linear_sum_assignment = None


State = Dict[str, Any]
MASK_TRACKING_SCHEMA_VERSION = 1
MASK_TRACKING_METHOD = "detector_guided_sam2_video_with_multicue_hungarian"


@dataclass(frozen=True)
class HybridMaskTrackingConfig:
    """Configuration kept JSON-serializable for cache/provenance hashing."""

    backend: str = "auto"
    sam2_model: str = "weights/sam2/sam2_t.pt"
    device: str = "cuda:0"
    allow_model_download: bool = False
    strict: bool = False
    min_assignment_score: float = 0.30
    minimum_mask_area: int = 16
    occlusion_fill: bool = True
    maximum_tracks_per_video: int = 0
    mask_iou_weight: float = 0.40
    flow_iou_weight: float = 0.20
    box_iou_weight: float = 0.20
    class_weight: float = 0.10
    depth_weight: float = 0.10

    @property
    def config_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class MaskProposal:
    track_id: int
    mask: np.ndarray
    bbox: list[float]
    label: str
    confidence: float
    mask_path: str
    flow_warped_mask: Optional[np.ndarray] = None
    depth: Optional[float] = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def resolve_config(
    tracking_args: Mapping[str, Any], detection_args: Optional[Mapping[str, Any]] = None
) -> HybridMaskTrackingConfig:
    """Resolve explicit args first and environment overrides second."""

    nested = dict(tracking_args.get("mask_tracking", {}))
    detection_device = str(dict(detection_args or {}).get("device", "cuda:0"))
    return HybridMaskTrackingConfig(
        backend=os.environ.get(
            "CAUVID_TRACKING_BACKEND", str(nested.get("backend", "auto"))
        ).strip().lower(),
        sam2_model=os.environ.get(
            "CAUVID_SAM2_MODEL", str(nested.get("sam2_model", "weights/sam2/sam2_t.pt"))
        ),
        device=os.environ.get(
            "CAUVID_SAM2_DEVICE", str(nested.get("device", detection_device))
        ),
        allow_model_download=_env_bool(
            "CAUVID_SAM2_ALLOW_DOWNLOAD", bool(nested.get("allow_model_download", False))
        ),
        strict=_env_bool("CAUVID_MASK_TRACKING_STRICT", bool(nested.get("strict", False))),
        min_assignment_score=_env_float(
            "CAUVID_MASK_TRACKING_MIN_SCORE",
            float(nested.get("min_assignment_score", 0.30)),
        ),
        minimum_mask_area=_env_int(
            "CAUVID_MASK_TRACKING_MIN_AREA", int(nested.get("minimum_mask_area", 16))
        ),
        occlusion_fill=_env_bool(
            "CAUVID_MASK_TRACKING_OCCLUSION_FILL", bool(nested.get("occlusion_fill", True))
        ),
        maximum_tracks_per_video=_env_int(
            "CAUVID_SAM2_MAX_TRACKS", int(nested.get("maximum_tracks_per_video", 0))
        ),
        mask_iou_weight=float(nested.get("mask_iou_weight", 0.40)),
        flow_iou_weight=float(nested.get("flow_iou_weight", 0.20)),
        box_iou_weight=float(nested.get("box_iou_weight", 0.20)),
        class_weight=float(nested.get("class_weight", 0.10)),
        depth_weight=float(nested.get("depth_weight", 0.10)),
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 4 or len(right) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(value) for value in left[:4])
    bx1, by1, bx2, by2 = (float(value) for value in right[:4])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        if cv2 is None:
            return 0.0
        right = cv2.resize(
            right.astype(np.uint8), (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    left_bool = left.astype(bool)
    right_bool = right.astype(bool)
    union = np.logical_or(left_bool, right_bool).sum()
    return float(np.logical_and(left_bool, right_bool).sum() / union) if union else 0.0


def _mask_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _box_mask(box: Sequence[float], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    result = np.zeros((height, width), dtype=bool)
    if len(box) < 4:
        return result
    x1 = max(0, min(width, int(math.floor(float(box[0])))))
    y1 = max(0, min(height, int(math.floor(float(box[1])))))
    x2 = max(0, min(width, int(math.ceil(float(box[2])))))
    y2 = max(0, min(height, int(math.ceil(float(box[3])))))
    if x2 > x1 and y2 > y1:
        result[y1:y2, x1:x2] = True
    return result


def _load_array(path_value: Any) -> Optional[np.ndarray]:
    path = Path(str(path_value or ""))
    if not path.is_file():
        return None
    try:
        if path.suffix.lower() == ".npy":
            return np.load(path, allow_pickle=False)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                for key in ("flow", "mask", "depth", "arr_0"):
                    if key in payload:
                        return np.asarray(payload[key])
        if cv2 is not None:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            return None if image is None else np.asarray(image)
    except (OSError, ValueError):
        return None
    return None


def _detection_mask(detection: Mapping[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    for key in ("mask", "segmentation_mask"):
        value = detection.get(key)
        if value is not None:
            array = np.asarray(value)
            if array.ndim >= 2:
                if array.ndim > 2:
                    array = array.squeeze()
                if array.shape != shape and cv2 is not None:
                    array = cv2.resize(
                        array.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
                    )
                return array.astype(bool), "neural_mask"
    for key in ("mask_path", "segmentation_path", "sam2_mask_path"):
        array = _load_array(detection.get(key))
        if array is not None and array.ndim >= 2:
            array = array.squeeze()
            if array.shape != shape and cv2 is not None:
                array = cv2.resize(
                    array.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
                )
            return array.astype(bool), "neural_mask"
    return _box_mask(detection.get("bbox", []), shape), "bbox_support"


def _depth_value(record: Mapping[str, Any]) -> Optional[float]:
    for key in ("median_depth", "depth_median", "depth", "z"):
        if key in record and record.get(key) is not None:
            value = _safe_float(record.get(key), float("nan"))
            if math.isfinite(value):
                return value
    evidence = record.get("evidence")
    if isinstance(evidence, Mapping):
        return _depth_value(evidence)
    return None


def _depth_similarity(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    scale = max(abs(left), abs(right), 1e-6)
    return float(math.exp(-abs(left - right) / scale))


def _warp_mask(mask: np.ndarray, flow: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Forward-warp a binary mask with dense flow in pixel units."""

    if flow is None or flow.ndim != 3 or flow.shape[-1] < 2:
        return None
    if flow.shape[:2] != mask.shape:
        if cv2 is None:
            return None
        old_h, old_w = flow.shape[:2]
        flow = cv2.resize(flow, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= mask.shape[1] / max(1, old_w)
        flow[..., 1] *= mask.shape[0] / max(1, old_h)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros_like(mask, dtype=bool)
    target_x = np.rint(xs + flow[ys, xs, 0]).astype(int)
    target_y = np.rint(ys + flow[ys, xs, 1]).astype(int)
    valid = (
        (target_x >= 0)
        & (target_x < mask.shape[1])
        & (target_y >= 0)
        & (target_y < mask.shape[0])
    )
    warped = np.zeros_like(mask, dtype=bool)
    warped[target_y[valid], target_x[valid]] = True
    if cv2 is not None and warped.any():
        warped = cv2.morphologyEx(
            warped.astype(np.uint8), np.ones((3, 3), np.uint8), cv2.MORPH_CLOSE
        ).astype(bool)
    return warped


def association_score(
    proposal: MaskProposal,
    detection: Mapping[str, Any],
    config: HybridMaskTrackingConfig,
) -> tuple[float, Dict[str, Any]]:
    """Return a normalized score using only cues that are actually available."""

    detection_mask, mask_support = _detection_mask(detection, proposal.mask.shape)
    cues: Dict[str, Optional[float]] = {
        "mask_iou": _mask_iou(proposal.mask, detection_mask),
        "flow_iou": (
            _mask_iou(proposal.flow_warped_mask, detection_mask)
            if proposal.flow_warped_mask is not None
            else None
        ),
        "box_iou": _box_iou(proposal.bbox, detection.get("bbox", [])),
        "class_consistency": (
            0.5
            if not proposal.label
            or proposal.label == "unknown"
            or str(detection.get("class", "unknown")) == "unknown"
            else float(proposal.label == str(detection.get("class", "unknown")))
        ),
        "depth_consistency": _depth_similarity(proposal.depth, _depth_value(detection)),
    }
    weights = {
        "mask_iou": config.mask_iou_weight,
        "flow_iou": config.flow_iou_weight,
        "box_iou": config.box_iou_weight,
        "class_consistency": config.class_weight,
        "depth_consistency": config.depth_weight,
    }
    available = [(name, value) for name, value in cues.items() if value is not None and weights[name] > 0]
    denominator = sum(weights[name] for name, _value in available)
    score = (
        sum(weights[name] * float(value) for name, value in available) / denominator
        if denominator > 0
        else 0.0
    )
    return float(score), {
        "score": float(score),
        "mask_support": mask_support,
        "cues": {name: value for name, value in cues.items() if value is not None},
        "normalized_weights": {
            name: float(weights[name] / denominator) for name, _value in available
        }
        if denominator > 0
        else {},
    }


def associate_proposals(
    proposals: Sequence[MaskProposal],
    detections: Sequence[Mapping[str, Any]],
    config: HybridMaskTrackingConfig,
) -> tuple[list[tuple[int, int, float, Dict[str, Any]]], list[int], list[int]]:
    """Perform one-to-one multi-cue assignment with explicit rejection."""

    if not proposals or not detections:
        return [], list(range(len(proposals))), list(range(len(detections)))
    if linear_sum_assignment is None:
        raise RuntimeError("scipy is required for multi-cue Hungarian association")

    score_matrix = np.zeros((len(proposals), len(detections)), dtype=np.float64)
    evidence: Dict[tuple[int, int], Dict[str, Any]] = {}
    for proposal_index, proposal in enumerate(proposals):
        for detection_index, detection in enumerate(detections):
            score, details = association_score(proposal, detection, config)
            score_matrix[proposal_index, detection_index] = score
            evidence[(proposal_index, detection_index)] = details

    rows, columns = linear_sum_assignment(1.0 - score_matrix)
    matches: list[tuple[int, int, float, Dict[str, Any]]] = []
    matched_proposals: set[int] = set()
    matched_detections: set[int] = set()
    for proposal_index, detection_index in zip(rows.tolist(), columns.tolist()):
        score = float(score_matrix[proposal_index, detection_index])
        if score < config.min_assignment_score:
            continue
        matches.append(
            (
                proposal_index,
                detection_index,
                score,
                evidence[(proposal_index, detection_index)],
            )
        )
        matched_proposals.add(proposal_index)
        matched_detections.add(detection_index)
    return (
        matches,
        [index for index in range(len(proposals)) if index not in matched_proposals],
        [index for index in range(len(detections)) if index not in matched_detections],
    )


def _accepted_detections(frame: Mapping[str, Any]) -> list[Dict[str, Any]]:
    records = list(frame.get("accepted_detections", []))
    if records:
        return [dict(record) for record in records]
    boxes = list(frame.get("boxes", []))
    scores = list(frame.get("scores", []))
    labels = list(frame.get("labels", []))
    return [
        {
            "bbox": list(box),
            "score": _safe_float(scores[index] if index < len(scores) else 0.0),
            "class": str(labels[index] if index < len(labels) else "unknown"),
            "detection_id": f"{int(frame.get('frame_index', -1)):06d}:accepted:{index:04d}",
        }
        for index, box in enumerate(boxes)
    ]


def _track_seeds(tracked_frames: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    tracks: Dict[int, Dict[str, Any]] = {}
    for position, frame in enumerate(tracked_frames):
        for index, track_id_value in enumerate(frame.get("track_ids", [])):
            track_id = int(track_id_value)
            boxes = list(frame.get("boxes", []))
            labels = list(frame.get("labels", []))
            scores = list(frame.get("scores", []))
            if index >= len(boxes):
                continue
            bucket = tracks.setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "first_position": position,
                    "last_position": position,
                    "first_bbox": list(boxes[index]),
                    "labels": Counter(),
                    "scores": [],
                },
            )
            bucket["last_position"] = position
            bucket["labels"].update([str(labels[index] if index < len(labels) else "unknown")])
            bucket["scores"].append(_safe_float(scores[index] if index < len(scores) else 0.0))
    seeds = []
    for track_id in sorted(tracks):
        bucket = tracks[track_id]
        scores = list(bucket.pop("scores"))
        labels = bucket.pop("labels")
        bucket["label"] = labels.most_common(1)[0][0] if labels else "unknown"
        bucket["confidence"] = float(sum(scores) / max(1, len(scores)))
        bucket["length"] = int(bucket["last_position"] - bucket["first_position"] + 1)
        seeds.append(bucket)
    return seeds


def _model_reference(config: HybridMaskTrackingConfig, project_root: Path) -> tuple[str, Optional[str]]:
    raw = config.sam2_model
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root / path
    if path.is_file():
        return str(path), None
    if config.allow_model_download and not any(separator in raw for separator in ("/", "\\")):
        return raw, None
    return str(path), f"SAM 2 checkpoint not found: {path}"


def _extract_result_mask(result: Any, minimum_area: int) -> Optional[np.ndarray]:
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None)
    if data is None or len(data) == 0:
        return None
    array = data[0].detach().cpu().numpy() if hasattr(data[0], "detach") else np.asarray(data[0])
    array = np.asarray(array).squeeze() > 0.5
    return array if int(array.sum()) >= int(minimum_area) else None


def _save_mask(mask: np.ndarray, path: Path) -> str:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to persist SAM 2 masklets")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise OSError(f"Could not write mask: {path}")
    return str(path)


def _write_video_clip(image_paths: Sequence[str], output_path: Path, frame_rate: float) -> Path:
    """Create the video-mode input required by Ultralytics SAM2VideoPredictor."""

    if cv2 is None:
        raise RuntimeError("OpenCV is required to build SAM 2 video clips")
    first = cv2.imread(str(image_paths[0])) if image_paths else None
    if first is None:
        raise FileNotFoundError(f"Could not read first SAM 2 input frame: {image_paths[:1]}")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(frame_rate)),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create SAM 2 input clip: {output_path}")
    try:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise FileNotFoundError(f"Could not read SAM 2 input frame: {image_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
    finally:
        writer.release()
    return output_path


def _propagate_sam2_masklets(
    tracked_video: Mapping[str, Any],
    detection_video: Mapping[str, Any],
    output_root: Path,
    model: Any,
    predictor_class: Any,
    config: HybridMaskTrackingConfig,
    frame_rate: float,
) -> tuple[Dict[int, list[MaskProposal]], Dict[str, Any]]:
    frames = list(tracked_video.get("frames", []))
    seeds = _track_seeds(frames)
    if config.maximum_tracks_per_video > 0:
        seeds = sorted(seeds, key=lambda row: (-int(row["length"]), -float(row["confidence"])))[: config.maximum_tracks_per_video]
    proposals: Dict[int, list[MaskProposal]] = defaultdict(list)
    failures: list[Dict[str, Any]] = []
    video_id = str(tracked_video.get("video_id", "video"))
    evidence_by_index = {
        int(frame.get("frame_index", -1)): frame for frame in detection_video.get("frames", [])
    }

    for seed in seeds:
        first = int(seed["first_position"])
        last = int(seed["last_position"])
        source_paths = [str(frame.get("image_path", "")) for frame in frames[first : last + 1]]
        if not source_paths or any(not Path(path).is_file() for path in source_paths):
            failures.append({"track_id": int(seed["track_id"]), "reason": "missing_frame_path"})
            continue
        clip_path = (
            output_root
            / video_id
            / "sam2_inputs"
            / f"track_{int(seed['track_id']):06d}.mp4"
        )
        try:
            _write_video_clip(source_paths, clip_path, frame_rate)
            results = model.predict(
                source=str(clip_path),
                predictor=predictor_class,
                bboxes=[list(seed["first_bbox"])],
                device=config.device,
                retina_masks=True,
                save=False,
                verbose=False,
                stream=False,
            )
            previous_mask: Optional[np.ndarray] = None
            for offset, result in enumerate(results):
                position = first + offset
                if position >= len(frames):
                    break
                mask = _extract_result_mask(result, config.minimum_mask_area)
                if mask is None:
                    previous_mask = None
                    continue
                frame = frames[position]
                frame_index = int(frame.get("frame_index", position))
                mask_path = _save_mask(
                    mask,
                    output_root
                    / video_id
                    / "masks"
                    / f"track_{int(seed['track_id']):06d}"
                    / f"frame_{frame_index:06d}.png",
                )
                flow = None
                evidence_frame = evidence_by_index.get(frame_index, frame)
                for key in ("optical_flow_path", "flow_path", "raft_flow_path"):
                    if evidence_frame.get(key):
                        flow = _load_array(evidence_frame.get(key))
                        if flow is not None:
                            break
                proposal = MaskProposal(
                    track_id=int(seed["track_id"]),
                    mask=mask,
                    bbox=_mask_bbox(mask),
                    label=str(seed["label"]),
                    confidence=float(seed["confidence"]),
                    mask_path=mask_path,
                    flow_warped_mask=_warp_mask(previous_mask, flow) if previous_mask is not None else None,
                )
                proposals[frame_index].append(proposal)
                previous_mask = mask
        except Exception as exc:  # one failed object must not erase successful masklets
            failures.append(
                {
                    "track_id": int(seed["track_id"]),
                    "reason": f"{exc.__class__.__name__}: {exc}",
                }
            )
        finally:
            try:
                clip_path.unlink(missing_ok=True)
            except OSError:
                pass

    return dict(proposals), {
        "num_bootstrap_tracks": len(_track_seeds(frames)),
        "num_prompted_tracks": len(seeds),
        "num_successful_tracks": len({p.track_id for rows in proposals.values() for p in rows}),
        "num_masklet_observations": sum(len(rows) for rows in proposals.values()),
        "track_failures": failures,
    }


def _baseline_track_id(frame: Mapping[str, Any], detection: Mapping[str, Any]) -> Optional[int]:
    detection_id = str(detection.get("detection_id", ""))
    detection_ids = [str(value) for value in frame.get("detection_ids", [])]
    if detection_id and detection_id in detection_ids:
        index = detection_ids.index(detection_id)
        track_ids = list(frame.get("track_ids", []))
        return int(track_ids[index]) if index < len(track_ids) else None
    best: tuple[float, Optional[int]] = (0.0, None)
    for index, box in enumerate(frame.get("boxes", [])):
        labels = list(frame.get("labels", []))
        if index < len(labels) and str(labels[index]) != str(detection.get("class", "unknown")):
            continue
        score = _box_iou(box, detection.get("bbox", []))
        track_ids = list(frame.get("track_ids", []))
        if score > best[0] and index < len(track_ids):
            best = (score, int(track_ids[index]))
    return best[1]


def _append_observation(
    frame: Dict[str, Any],
    *,
    bbox: Sequence[float],
    score: float,
    label: str,
    track_id: int,
    detection_id: str,
    mask_path: str,
    mask_source: str,
    association_score_value: float,
    visibility_state: str,
    evidence: Mapping[str, Any],
) -> None:
    aligned = {
        "boxes": list(bbox),
        "scores": float(score),
        "labels": str(label),
        "track_ids": int(track_id),
        "detection_ids": str(detection_id),
        "mask_paths": str(mask_path),
        "mask_sources": str(mask_source),
        "association_scores": float(association_score_value),
        "tracking_confidences": float(score) * max(0.0, min(1.0, float(association_score_value))),
        "visibility_states": str(visibility_state),
        "association_evidence": dict(evidence),
    }
    for key, value in aligned.items():
        frame.setdefault(key, []).append(value)


def _fuse_video(
    detection_video: Mapping[str, Any],
    tracked_video: Mapping[str, Any],
    proposals_by_frame: Mapping[int, Sequence[MaskProposal]],
    config: HybridMaskTrackingConfig,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    baseline_by_index = {
        int(frame.get("frame_index", -1)): dict(frame) for frame in tracked_video.get("frames", [])
    }
    detection_by_index = {
        int(frame.get("frame_index", -1)): dict(frame) for frame in detection_video.get("frames", [])
    }
    frame_indices = sorted(set(baseline_by_index) | set(detection_by_index) | set(proposals_by_frame))
    output_frames: list[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    last_depth_by_track: Dict[int, float] = {}

    for frame_index in frame_indices:
        baseline_frame = baseline_by_index.get(frame_index, {})
        detection_frame = detection_by_index.get(frame_index, {})
        detections = _accepted_detections(detection_frame)
        proposals = list(proposals_by_frame.get(frame_index, []))
        for proposal in proposals:
            proposal.depth = last_depth_by_track.get(proposal.track_id)
        matches, unmatched_proposals, unmatched_detections = associate_proposals(
            proposals, detections, config
        )
        output_frame: Dict[str, Any] = {
            "frame": detection_frame.get("frame") or baseline_frame.get("frame", ""),
            "frame_index": frame_index,
            "image_path": detection_frame.get("image_path") or baseline_frame.get("image_path", ""),
        }
        used_track_ids: set[int] = set()

        for proposal_index, detection_index, score, evidence in matches:
            proposal = proposals[proposal_index]
            detection = detections[detection_index]
            used_track_ids.add(proposal.track_id)
            matched_depth = _depth_value(detection)
            if matched_depth is not None:
                last_depth_by_track[proposal.track_id] = matched_depth
            _append_observation(
                output_frame,
                bbox=detection.get("bbox", proposal.bbox),
                score=_safe_float(detection.get("score", proposal.confidence)),
                label=str(detection.get("class", proposal.label)),
                track_id=proposal.track_id,
                detection_id=str(detection.get("detection_id", "")),
                mask_path=proposal.mask_path,
                mask_source="sam2_video",
                association_score_value=score,
                visibility_state="detected_and_propagated",
                evidence=evidence,
            )
            counts["matched"] += 1
            counts[f"mask_support_{evidence.get('mask_support', 'unknown')}"] += 1
            for cue_name in dict(evidence.get("cues", {})):
                counts[f"cue_{cue_name}_used"] += 1

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            track_id = _baseline_track_id(baseline_frame, detection)
            if track_id is None or track_id in used_track_ids:
                counts["unassigned_detections"] += 1
                continue
            used_track_ids.add(track_id)
            _append_observation(
                output_frame,
                bbox=detection.get("bbox", []),
                score=_safe_float(detection.get("score", 0.0)),
                label=str(detection.get("class", "unknown")),
                track_id=track_id,
                detection_id=str(detection.get("detection_id", "")),
                mask_path="",
                mask_source="none",
                association_score_value=1.0,
                visibility_state="bytetrack_fallback",
                evidence={"fallback": "no_accepted_mask_assignment"},
            )
            counts["bytetrack_fallback"] += 1

        if config.occlusion_fill:
            for proposal_index in unmatched_proposals:
                proposal = proposals[proposal_index]
                if proposal.track_id in used_track_ids or not proposal.bbox:
                    continue
                used_track_ids.add(proposal.track_id)
                _append_observation(
                    output_frame,
                    bbox=proposal.bbox,
                    score=proposal.confidence * 0.5,
                    label=proposal.label,
                    track_id=proposal.track_id,
                    detection_id="",
                    mask_path=proposal.mask_path,
                    mask_source="sam2_video",
                    association_score_value=0.5,
                    visibility_state="propagated_gap",
                    evidence={"detector_support": False},
                )
                counts["propagated_gaps"] += 1

        order = sorted(range(len(output_frame.get("track_ids", []))), key=lambda i: output_frame["track_ids"][i])
        for key in (
            "boxes",
            "scores",
            "labels",
            "track_ids",
            "detection_ids",
            "mask_paths",
            "mask_sources",
            "association_scores",
            "tracking_confidences",
            "visibility_states",
            "association_evidence",
        ):
            values = output_frame.get(key, [])
            output_frame[key] = [values[index] for index in order]
        output_frames.append(output_frame)

    result = dict(tracked_video)
    result["schema_version"] = max(int(result.get("schema_version", 1)), 7)
    result["frames"] = output_frames
    result["num_frames"] = len(output_frames)
    result["num_tracks"] = len(
        {int(track_id) for frame in output_frames for track_id in frame.get("track_ids", [])}
    )
    result["accepted_tracks"] = {
        **dict(result.get("accepted_tracks", {})),
        "num_tracks": result["num_tracks"],
        "frames": output_frames,
        "track_summaries": _track_summaries(output_frames),
    }
    return result, dict(counts)


def _track_summaries(frames: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    for frame in frames:
        frame_index = int(frame.get("frame_index", -1))
        for position, track_id_value in enumerate(frame.get("track_ids", [])):
            track_id = int(track_id_value)
            bucket = rows.setdefault(
                track_id,
                {"labels": Counter(), "scores": [], "frames": [], "mask_frames": 0, "gap_frames": 0},
            )
            labels = list(frame.get("labels", []))
            scores = list(frame.get("scores", []))
            mask_paths = list(frame.get("mask_paths", []))
            visibility = list(frame.get("visibility_states", []))
            bucket["labels"].update([str(labels[position] if position < len(labels) else "unknown")])
            bucket["scores"].append(_safe_float(scores[position] if position < len(scores) else 0.0))
            bucket["frames"].append(frame_index)
            bucket["mask_frames"] += int(position < len(mask_paths) and bool(mask_paths[position]))
            bucket["gap_frames"] += int(
                position < len(visibility) and visibility[position] == "propagated_gap"
            )
    summaries = []
    for track_id, bucket in sorted(rows.items()):
        scores = list(bucket["scores"])
        indices = list(bucket["frames"])
        summaries.append(
            {
                "track_id": track_id,
                "label": bucket["labels"].most_common(1)[0][0] if bucket["labels"] else "unknown",
                "mean_score": float(sum(scores) / max(1, len(scores))),
                "max_score": float(max(scores) if scores else 0.0),
                "track_length": len(scores),
                "first_frame_index": min(indices) if indices else -1,
                "last_frame_index": max(indices) if indices else -1,
                "num_mask_frames": int(bucket["mask_frames"]),
                "num_propagated_gap_frames": int(bucket["gap_frames"]),
            }
        )
    return summaries


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _annotate_fallback(
    tracking_state: State,
    config: HybridMaskTrackingConfig,
    reason: str,
) -> State:
    tracks = []
    for tracked_video in tracking_state.get("tracks", []):
        updated = dict(tracked_video)
        metadata = {
            "schema_version": MASK_TRACKING_SCHEMA_VERSION,
            "config_id": config.config_id,
            "requested_backend": config.backend,
            "effective_backend": "bytetrack",
            "status": "fallback",
            "fallback_reason": reason,
            "mask_semantics": "none",
        }
        updated["mask_tracking"] = metadata
        tracks_path = Path(str(dict(updated.get("output_paths", {})).get("tracks_json", "")))
        if tracks_path.parent.is_dir():
            _write_json_atomic(tracks_path, updated)
        tracks.append(updated)
    return {
        **tracking_state,
        "tracks": tracks,
        "tracking_backend_requested": config.backend,
        "tracking_backend_effective": "bytetrack",
        "mask_tracking_status": "fallback",
        "mask_tracking_fallback_reason": reason,
    }


def run(
    detection_state: State,
    tracking_state: State,
    tracking_args: Mapping[str, Any],
    *,
    project_root: Path,
) -> State:
    """Enhance ByteTrack bootstrap tracks with SAM 2 masklets when possible."""

    config = resolve_config(tracking_args, detection_state.get("detection_args"))
    if config.backend not in {"auto", "hybrid_mask", "bytetrack"}:
        raise ValueError("tracking backend must be one of: auto, hybrid_mask, bytetrack")
    if config.backend == "bytetrack":
        return _annotate_fallback(tracking_state, config, "ByteTrack explicitly selected")

    model_reference, unavailable_reason = _model_reference(config, project_root)
    if unavailable_reason:
        if config.strict or config.backend == "hybrid_mask":
            raise RuntimeError(unavailable_reason)
        return _annotate_fallback(tracking_state, config, unavailable_reason)
    if cv2 is None:
        reason = "OpenCV is required for SAM 2 mask persistence"
        if config.strict or config.backend == "hybrid_mask":
            raise RuntimeError(reason)
        return _annotate_fallback(tracking_state, config, reason)

    try:
        from ultralytics import SAM
        from ultralytics.models.sam import SAM2VideoPredictor
    except Exception as exc:
        reason = f"Ultralytics SAM 2 runtime unavailable: {exc.__class__.__name__}: {exc}"
        if config.strict or config.backend == "hybrid_mask":
            raise RuntimeError(reason) from exc
        return _annotate_fallback(tracking_state, config, reason)

    output_root = Path(str(tracking_args["output_root"]))
    model = SAM(model_reference)
    detections_by_video = {
        str(video.get("video_id", "")): video for video in detection_state.get("detections", [])
    }
    enhanced_tracks: list[Dict[str, Any]] = []
    manifest_rows: list[Dict[str, Any]] = []

    for tracked_video in tracking_state.get("tracks", []):
        video_id = str(tracked_video.get("video_id", ""))
        current_metadata = dict(tracked_video.get("mask_tracking", {}))
        if (
            current_metadata.get("status") == "completed"
            and current_metadata.get("config_id") == config.config_id
            and not bool(tracking_args.get("force_recompute", False))
        ):
            enhanced_tracks.append(dict(tracked_video))
            manifest_rows.append(current_metadata)
            continue
        detection_video = detections_by_video.get(video_id)
        if detection_video is None:
            reason = f"detection payload missing for video {video_id}"
            if config.strict:
                raise RuntimeError(reason)
            updated = _annotate_fallback({"tracks": [tracked_video]}, config, reason)["tracks"][0]
            enhanced_tracks.append(updated)
            manifest_rows.append(dict(updated["mask_tracking"]))
            continue

        proposals, propagation_stats = _propagate_sam2_masklets(
            tracked_video,
            detection_video,
            output_root,
            model,
            SAM2VideoPredictor,
            config,
            float(tracking_args.get("frame_rate", 10)),
        )
        if not propagation_stats["num_successful_tracks"]:
            reason = "SAM 2 produced no valid masklets"
            if propagation_stats.get("track_failures"):
                reason += f"; first failure: {propagation_stats['track_failures'][0]['reason']}"
            if config.strict:
                raise RuntimeError(reason)
            updated = _annotate_fallback({"tracks": [tracked_video]}, config, reason)["tracks"][0]
            enhanced_tracks.append(updated)
            manifest_rows.append(dict(updated["mask_tracking"]))
            continue

        fused, fusion_counts = _fuse_video(detection_video, tracked_video, proposals, config)
        metadata = {
            "schema_version": MASK_TRACKING_SCHEMA_VERSION,
            "config_id": config.config_id,
            "requested_backend": config.backend,
            "effective_backend": "hybrid_mask",
            "status": "completed",
            "method": MASK_TRACKING_METHOD,
            "bootstrap_tracker": "ByteTrack",
            "mask_propagator": "SAM2VideoPredictor",
            "association_solver": "Hungarian",
            "model": model_reference,
            "configured_cues": ["mask_iou", "raft_flow_iou", "box_iou", "class", "depth"],
            "mask_semantics": "binary_sam2_video_mask",
            "propagation": propagation_stats,
            "fusion": fusion_counts,
        }
        fused["mask_tracking"] = metadata
        tracks_path = Path(str(dict(fused.get("output_paths", {})).get("tracks_json", "")))
        if not tracks_path.parent.is_dir():
            tracks_path = output_root / video_id / "tracks.json"
            fused.setdefault("output_paths", {})["tracks_json"] = str(tracks_path)
        _write_json_atomic(tracks_path, fused)
        enhanced_tracks.append(fused)
        manifest_rows.append(metadata)

    manifest = {
        "schema_version": MASK_TRACKING_SCHEMA_VERSION,
        "method": MASK_TRACKING_METHOD,
        "config": asdict(config),
        "config_id": config.config_id,
        "num_videos": len(enhanced_tracks),
        "videos": [
            {"video_id": str(video.get("video_id", "")), **dict(video.get("mask_tracking", {}))}
            for video in enhanced_tracks
        ],
    }
    manifest_path = output_root / "mask_tracking_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    effective = "hybrid_mask" if any(
        str(row.get("effective_backend")) == "hybrid_mask" for row in manifest_rows
    ) else "bytetrack"
    return {
        **tracking_state,
        "tracks": enhanced_tracks,
        "tracking_backend_requested": config.backend,
        "tracking_backend_effective": effective,
        "mask_tracking_status": "completed" if effective == "hybrid_mask" else "fallback",
        "mask_tracking_manifest_path": str(manifest_path),
    }
