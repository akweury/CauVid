"""Reusable deterministic association for frame-to-frame ID linking.

This module exposes two layers:
1. a low-level one-frame assignment helper for already-scored pairs; and
2. a higher-level video tracker that accepts per-frame detections/masks and
    computes the pairwise matching internally.

Both layers are intentionally model-free. They rely only on geometry and the
provided observation tensors, making the code easy to copy into another
project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


PairKey = tuple[str, str]


@dataclass(frozen=True)
class FrameDetection:
    """A single detection observed in one frame."""

    detection_id: str
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    class_name: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class FrameDetections:
    """All detections for one video frame."""

    frame_index: int
    detections: tuple[FrameDetection, ...]


@dataclass(frozen=True)
class TrackObservation:
    """A matched detection attached to a persistent track."""

    frame_index: int
    detection_id: str
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    class_name: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class TrackedObject:
    """A single persistent object track across multiple frames."""

    track_id: str
    first_frame_index: int
    last_frame_index: int
    observations: tuple[TrackObservation, ...]


@dataclass
class _TrackState:
    track_id: str
    first_frame_index: int
    last_frame_index: int
    last_bbox_xyxy: tuple[float, float, float, float]
    last_mask: np.ndarray | None
    missed_count: int = 0
    observations: list[TrackObservation] = field(default_factory=list)


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _mask_iou(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    if left.shape != right.shape:
        raise ValueError("mask IoU requires masks with the same shape")
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(left, right).sum() / union)


def _center_distance_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_center = np.array(((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0))
    right_center = np.array(((right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0))
    diagonal = float(np.hypot(max(left[2], right[2]) - min(left[0], right[0]), max(left[3], right[3]) - min(left[1], right[1])))
    if diagonal <= 1e-12:
        return 0.0
    return float(np.linalg.norm(left_center - right_center) / diagonal)


@dataclass(frozen=True)
class AssociationOutcome:
    """Result of a one-to-one assignment pass."""

    selected_pairs: frozenset[PairKey]
    rank_by_pair: dict[PairKey, int]


def assign_one_to_one_matches(
    *,
    track_ids: Sequence[str],
    detection_ids: Sequence[str],
    score_by_pair: Mapping[PairKey, float],
    feasible_by_pair: Mapping[PairKey, bool],
    minimum_score: float,
) -> AssociationOutcome:
    """Select the best track/detection links for a single frame.

    The function performs two steps:
    1. build a cost matrix from the provided pair scores and deterministic gates;
    2. run Hungarian assignment to get one detection per track and one track per detection.

    Parameters
    ----------
    track_ids:
        Active track IDs for the current frame.
    detection_ids:
        Detection IDs for the current frame.
    score_by_pair:
        Association score for every evaluated track/detection pair.
    feasible_by_pair:
        Gate result for every evaluated pair. Infeasible pairs are excluded from
        assignment.
    minimum_score:
        Final acceptance threshold after one-to-one assignment.
    """

    selected_pairs: set[PairKey] = set()
    rank_by_pair: dict[PairKey, int] = {}

    if not track_ids or not detection_ids:
        return AssociationOutcome(frozenset(), rank_by_pair)

    cost = np.full((len(track_ids), len(detection_ids)), 1e6, dtype=np.float64)
    for track_index, track_id in enumerate(track_ids):
        for detection_index, detection_id in enumerate(detection_ids):
            pair = (track_id, detection_id)
            if feasible_by_pair.get(pair, False):
                cost[track_index, detection_index] = 1.0 - float(score_by_pair[pair])

    rows, columns = linear_sum_assignment(cost)
    for row_index, column_index in zip(rows, columns):
        pair = (track_ids[row_index], detection_ids[column_index])
        if feasible_by_pair.get(pair, False) and score_by_pair[pair] >= minimum_score:
            selected_pairs.add(pair)

    for track_id in track_ids:
        ordered = sorted(
            ((detection_id, score_by_pair[(track_id, detection_id)]) for detection_id in detection_ids),
            key=lambda item: (-item[1], item[0]),
        )
        for rank, (detection_id, _) in enumerate(ordered, 1):
            rank_by_pair[(track_id, detection_id)] = rank

    return AssociationOutcome(frozenset(selected_pairs), rank_by_pair)


def track_objects_across_frames(
    frames: Sequence[FrameDetections],
    *,
    minimum_score: float = 0.30,
    max_age_frames: int = 2,
    box_weight: float = 0.60,
    mask_weight: float = 0.40,
    max_center_distance_ratio: float = 0.25,
) -> tuple[TrackedObject, ...]:
    """Link detections into object tracks across an entire video.

    The function expects one frame at a time, each frame containing the
    detections already extracted from the video. It builds the candidate pairs
    internally, scores them with bbox IoU and optional mask IoU, then applies
    one-to-one Hungarian assignment per frame.

    Parameters
    ----------
    frames:
        Ordered video frames with detections and masks.
    minimum_score:
        Minimum accepted match score after assignment.
    max_age_frames:
        Number of missed frames allowed before a track is retired.
    box_weight:
        Relative weight for bbox IoU.
    mask_weight:
        Relative weight for mask IoU when both masks are available.
    max_center_distance_ratio:
        Hard gate on normalized center distance.
    """

    if not frames:
        return ()

    active_tracks: list[_TrackState] = []
    finished_tracks: list[_TrackState] = []
    next_track_index = 1

    for frame in frames:
        detections = tuple(frame.detections)
        if not detections:
            for track in active_tracks:
                track.missed_count += 1
            still_active: list[_TrackState] = []
            for track in active_tracks:
                if track.missed_count > max_age_frames:
                    finished_tracks.append(track)
                else:
                    still_active.append(track)
            active_tracks = still_active
            continue

        score_by_pair: dict[PairKey, float] = {}
        feasible_by_pair: dict[PairKey, bool] = {}
        track_ids = [track.track_id for track in active_tracks]
        detection_ids = [detection.detection_id for detection in detections]

        for track in active_tracks:
            for detection in detections:
                pair = (track.track_id, detection.detection_id)
                center_gate = _center_distance_ratio(track.last_bbox_xyxy, detection.bbox_xyxy)
                feasible = center_gate <= max_center_distance_ratio
                feasible_by_pair[pair] = feasible
                if not feasible:
                    score_by_pair[pair] = 0.0
                    continue
                box_score = _box_iou(track.last_bbox_xyxy, detection.bbox_xyxy)
                mask_score = (
                    _mask_iou(track.last_mask, detection.mask)
                    if track.last_mask is not None and detection.mask is not None
                    else None
                )
                if mask_score is None:
                    score = box_score
                else:
                    total_weight = box_weight + mask_weight
                    score = (
                        box_weight * box_score + mask_weight * mask_score
                    ) / total_weight if total_weight > 0 else box_score
                score_by_pair[pair] = float(np.clip(score, 0.0, 1.0))

        assignment = assign_one_to_one_matches(
            track_ids=track_ids,
            detection_ids=detection_ids,
            score_by_pair=score_by_pair,
            feasible_by_pair=feasible_by_pair,
            minimum_score=minimum_score,
        )

        detection_by_id = {detection.detection_id: detection for detection in detections}
        matched_detection_ids = {detection_id for _, detection_id in assignment.selected_pairs}
        still_active = []

        for track in active_tracks:
            matched = next(
                (detection_id for track_id, detection_id in assignment.selected_pairs if track_id == track.track_id),
                None,
            )
            if matched is None:
                track.missed_count += 1
                if track.missed_count > max_age_frames:
                    finished_tracks.append(track)
                else:
                    still_active.append(track)
                continue

            detection = detection_by_id[matched]
            track.last_frame_index = frame.frame_index
            track.last_bbox_xyxy = detection.bbox_xyxy
            track.last_mask = detection.mask
            track.missed_count = 0
            track.observations.append(
                TrackObservation(
                    frame_index=frame.frame_index,
                    detection_id=detection.detection_id,
                    bbox_xyxy=detection.bbox_xyxy,
                    mask=detection.mask,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                )
            )
            still_active.append(track)

        active_tracks = still_active

        for detection in detections:
            if detection.detection_id in matched_detection_ids:
                continue
            track_id = f"track:{next_track_index:06d}"
            next_track_index += 1
            track = _TrackState(
                track_id=track_id,
                first_frame_index=frame.frame_index,
                last_frame_index=frame.frame_index,
                last_bbox_xyxy=detection.bbox_xyxy,
                last_mask=detection.mask,
                observations=[
                    TrackObservation(
                        frame_index=frame.frame_index,
                        detection_id=detection.detection_id,
                        bbox_xyxy=detection.bbox_xyxy,
                        mask=detection.mask,
                        class_name=detection.class_name,
                        confidence=detection.confidence,
                    )
                ],
            )
            active_tracks.append(track)

    finished_tracks.extend(active_tracks)
    return tuple(
        TrackedObject(
            track_id=track.track_id,
            first_frame_index=track.first_frame_index,
            last_frame_index=track.last_frame_index,
            observations=tuple(track.observations),
        )
        for track in finished_tracks
    )
