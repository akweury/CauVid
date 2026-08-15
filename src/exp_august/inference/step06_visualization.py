"""Diagnostic plots for Step 6 residual packets."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.exp_august.contracts import (
    ArtifactOwner,
    ResidualStore,
    VideoManifest,
    VideoResidualManifest,
    VideoTrackingManifest,
)
from src.exp_august.contracts.codec import read_contract, sha256_file
from src.exp_august.inference.artifact_io import read_image_artifact
from src.exp_august.inference.frames import CanonicalFrameProvider


_FAMILY_COLORS = {
    "observation": "#1f77b4",
    "ego_background": "#9467bd",
    "object_identity": "#ff7f0e",
    "physics": "#d62728",
    "semantic": "#2ca02c",
}

_FAMILY_BGR = {
    "observation": (180, 105, 35),
    "ego_background": (170, 95, 145),
    "object_identity": (35, 135, 235),
    "physics": (55, 55, 210),
    "semantic": (80, 155, 55),
}

_KEY_RED = (55, 65, 205)
_KEY_BLUE = (205, 115, 35)
_KEY_GREEN = (70, 145, 75)
_KEY_ORANGE = (35, 135, 220)
_TEXT_DARK = (30, 35, 43)
_TEXT_MUTED = (76, 82, 92)
_PANEL_WIDTH = 1920
_PANEL_HEIGHT = 1080
_HEADER_HEIGHT = 130
_FRAME_REGION_WIDTH = 1280
_SIDE_LEFT = 1304
_SIDE_RIGHT = 1894


def _run_root(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "06_predict_verify":
            return parent.parent
    raise RuntimeError("Step 6 store must live below 06_predict_verify")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "conflict"


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write Step 6 visualization: {path}")


def _put_lines(
    image: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int],
    scale: float = 0.56,
    color: tuple[int, int, int] = (35, 40, 48),
    line_height: int = 30,
    thickness: int = 1,
) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_height


def _format_values(values) -> str:
    if not values:
        return "n/a"
    return "[" + ", ".join(f"{float(value):.3g}" for value in values) + "]"


def _load_tracking_and_frames(*, run_root: Path, manifest: VideoResidualManifest):
    tracking_link = manifest.input_snapshot.video_tracking_manifest
    tracking_root = run_root / manifest.input_snapshot.source_step3_relative_root
    tracking_path = tracking_root / tracking_link.artifact.relative_path
    if sha256_file(tracking_path) != tracking_link.artifact.sha256:
        raise RuntimeError(f"Step 6 tracking manifest hash mismatch: {tracking_path}")
    tracking = read_contract(tracking_path, VideoTrackingManifest)
    source_link = tracking.input_snapshot.source_video_manifest
    source_path = run_root / "01_init" / source_link.artifact.relative_path
    if sha256_file(source_path) != source_link.artifact.sha256:
        raise RuntimeError(f"Step 6 source manifest hash mismatch: {source_path}")
    source = read_contract(source_path, VideoManifest)
    provider = CanonicalFrameProvider(source, verify_source_hash=True)
    step2_root = run_root / manifest.input_snapshot.source_step2_relative_root
    return tracking, tracking_root, step2_root, provider


def _track_observation(tracking, track_id: str | None, frame_index: int):
    if track_id is None:
        return None, None
    track = next((row for row in tracking.tracks if row.track_id == track_id), None)
    if track is None:
        return None, None
    observation = next(
        (row for row in track.observations if row.frame_index == frame_index), None
    )
    return track, observation


def _selected_mask(*, tracking, observation, step2_root: Path, step3_root: Path):
    if observation is None or observation.selected_mask_candidate_id is None:
        return None
    candidate = next(
        (
            row
            for row in tracking.mask_candidate_bank
            if row.candidate_id == observation.selected_mask_candidate_id
        ),
        None,
    )
    if candidate is None or candidate.mask is None:
        return None
    root = step2_root if candidate.mask.owner == ArtifactOwner.STEP2_NEURAL_EVIDENCE else step3_root
    image = read_image_artifact(
        root / candidate.mask.artifact.relative_path, cv2.IMREAD_GRAYSCALE
    )
    return None if image is None else image > 0


def _draw_track_support(
    image: np.ndarray,
    *,
    tracking,
    residual,
    step2_root: Path,
    step3_root: Path,
) -> tuple[tuple[float, float] | None, object | None, object | None]:
    track, observation = _track_observation(
        tracking, residual.track_id, residual.start_frame_index
    )
    if observation is None:
        return None, track, None
    mask = _selected_mask(
        tracking=tracking,
        observation=observation,
        step2_root=step2_root,
        step3_root=step3_root,
    )
    color = _FAMILY_BGR[residual.family.value]
    if mask is not None and mask.shape == image.shape[:2]:
        image[mask] = np.clip(
            image[mask].astype(np.float32) * 0.68
            + np.asarray(color, dtype=np.float32) * 0.32,
            0,
            255,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, color, 2)
    box = observation.bbox
    x1, y1, x2, y2 = map(
        int, (round(box.x1), round(box.y1), round(box.x2), round(box.y2))
    )
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    return (0.5 * (box.x1 + box.x2), 0.5 * (box.y1 + box.y2)), track, observation


def _draw_prediction_observation(image: np.ndarray, residual, origin) -> None:
    predicted = tuple(float(value) for value in residual.predicted_values)
    observed = tuple(float(value) for value in residual.observed_values)
    if residual.metric_name == "pixel_reprojection_error" and len(predicted) >= 2 and len(observed) >= 2:
        pred = tuple(int(round(value)) for value in predicted[:2])
        obs = tuple(int(round(value)) for value in observed[:2])
        cv2.line(image, pred, obs, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.drawMarker(image, pred, (255, 205, 40), cv2.MARKER_CROSS, 22, 3)
        cv2.circle(image, obs, 10, (40, 170, 255), 3, cv2.LINE_AA)
        return
    if "flow" in residual.metric_name and len(predicted) >= 2 and len(observed) >= 2:
        if origin is None:
            origin = (image.shape[1] / 2.0, image.shape[0] / 2.0)
        start = tuple(int(round(value)) for value in origin)
        vectors = (np.asarray(predicted[:2]), np.asarray(observed[:2]))
        maximum_norm = max(float(np.linalg.norm(vector)) for vector in vectors)
        display_scale = min(1.0, 180.0 / max(maximum_norm, 1e-6))
        margin = 8.0
        for vector in vectors:
            for axis, limit in ((0, image.shape[1]), (1, image.shape[0])):
                component = float(vector[axis])
                if abs(component) < 1e-8:
                    continue
                available = (
                    limit - margin - origin[axis]
                    if component > 0
                    else origin[axis] - margin
                )
                display_scale = min(display_scale, max(0.0, available / abs(component)))
        for vector, color in ((vectors[0], _KEY_RED), (vectors[1], _KEY_BLUE)):
            end = (
                int(round(origin[0] + display_scale * vector[0])),
                int(round(origin[1] + display_scale * vector[1])),
            )
            cv2.arrowedLine(image, start, end, color, 5, cv2.LINE_AA, tipLength=0.18)
            cv2.circle(image, end, 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"Shared arrow scale: {display_scale:.3g}x",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(image, "Predicted motion", (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _KEY_RED, 2, cv2.LINE_AA)
        cv2.putText(image, "RAFT flow", (245, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _KEY_BLUE, 2, cv2.LINE_AA)


def _flow_diagnosis(residual) -> tuple[str, float, float] | None:
    if "flow" not in residual.metric_name:
        return None
    predicted = np.asarray(residual.predicted_values[:2], dtype=np.float64)
    observed = np.asarray(residual.observed_values[:2], dtype=np.float64)
    if predicted.size != 2 or observed.size != 2:
        return None
    predicted_norm = float(np.linalg.norm(predicted))
    observed_norm = float(np.linalg.norm(observed))
    epsilon = 1e-6
    direction = residual.flow_direction_error_deg
    if direction is None:
        cosine = float(np.dot(predicted, observed) / max(predicted_norm * observed_norm, epsilon))
        direction = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    ratio = residual.flow_magnitude_ratio
    if ratio is None:
        ratio = max(predicted_norm, observed_norm, epsilon) / max(
            min(predicted_norm, observed_norm), epsilon
        )
    direction_conflict = direction > 30.0
    magnitude_conflict = ratio > 1.5
    if direction_conflict and magnitude_conflict:
        label = "direction + magnitude conflict"
    elif direction_conflict:
        label = "direction conflict"
    elif magnitude_conflict:
        label = "magnitude conflict"
    else:
        label = "endpoint mismatch without dominant component"
    return label, float(direction), float(ratio)


def _section_label(
    image: np.ndarray,
    *,
    y: int,
    label: str,
    accent: tuple[int, int, int],
) -> int:
    del accent
    cv2.putText(
        image,
        label,
        (_SIDE_LEFT, y + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        _TEXT_DARK,
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        (_SIDE_LEFT, y + 38),
        (_SIDE_RIGHT, y + 38),
        (205, 210, 216),
        2,
    )
    return y + 58


def _bullet_text(
    image: np.ndarray,
    text: str,
    *,
    y: int,
    level: int = 0,
    important: bool = False,
    accent: tuple[int, int, int] = (120, 120, 120),
) -> int:
    x = _SIDE_LEFT + 10 + 25 * level
    text_x = x + 20
    wrap_width = 61 - 5 * level
    lines = textwrap.wrap(text, width=max(28, wrap_width)) or [""]
    line_height = 31 if important else 28
    block_height = line_height * len(lines) + 8
    radius = 6 if level == 0 else 4
    cv2.circle(image, (x + 4, y - 6), radius, accent if important else (105, 112, 122), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (text_x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64 if important else 0.57,
            accent if important else _TEXT_MUTED,
            2 if important else 1,
            cv2.LINE_AA,
        )
    return y + block_height


def _pale_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(0.88 * 247 + 0.12 * value) for value in color)


def _rich_bullet(
    image: np.ndarray,
    segments: list[tuple[str, tuple[int, int, int], bool]],
    *,
    y: int,
    level: int = 0,
) -> int:
    bullet_x = _SIDE_LEFT + 10 + 25 * level
    start_x = bullet_x + 20
    x = start_x
    baseline = y
    scale = 0.64 if level == 0 else 0.57
    thickness = 2 if level == 0 else 1
    line_height = 32 if level == 0 else 29
    cv2.circle(
        image,
        (bullet_x + 4, baseline - 6),
        6 if level == 0 else 4,
        segments[0][1] if segments else _TEXT_MUTED,
        -1,
    )
    for text, color, highlighted in segments:
        (width, height), baseline_pad = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        if x + width > _SIDE_RIGHT and x > start_x:
            x = start_x
            baseline += line_height
        if highlighted:
            cv2.rectangle(
                image,
                (x - 4, baseline - height - 5),
                (x + width + 4, baseline + baseline_pad + 4),
                _pale_color(color),
                -1,
            )
        cv2.putText(
            image,
            text,
            (x, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        x += width + 4
    return baseline + 32


def _conflict_relation(residual) -> str:
    descriptions = {
        "object_reprojection": "Predicted image position vs observed object centroid",
        "heldout_object_depth": "Predicted object depth vs held-out depth evidence",
        "heldout_object_backward_flow": "Predicted object motion vs held-out RAFT backward flow",
        "heldout_background_backward_flow": "Ego-predicted background flow vs held-out RAFT flow",
        "trajectory_temporal_gap": "Expected continuous trajectory vs missing temporal support",
        "ego_acceleration_bound": "Estimated ego acceleration vs physical plausibility bound",
        "object_acceleration_bound": "Estimated object acceleration vs physical plausibility bound",
        "ego_speed_bound": "Estimated ego speed vs physical plausibility bound",
        "object_speed_bound": "Estimated object speed vs physical plausibility bound",
        "semantic_static_motion": "Predicted object motion vs semantic-static prior",
        "track_endpoint_accounting": "Track endpoint vs required lifecycle explanation",
    }
    return descriptions.get(
        residual.constraint_id,
        f"Predicted state vs {residual.metric_name.replace('_', ' ')} evidence",
    )


def _subject_segments(*, residual, track):
    if residual.track_id is not None:
        short_id = residual.track_id.rsplit(":", 1)[-1].lstrip("0") or "0"
        class_name = track.primary_class if track is not None else "object"
        return [
            ("Object: ", _TEXT_DARK, False),
            (class_name, _KEY_RED, True),
            ("  |  Track ID ", _TEXT_DARK, False),
            (short_id, _KEY_BLUE, True),
        ]
    if residual.family.value == "ego_background":
        return [
            ("Ego vehicle", _KEY_RED, True),
            (" and ", _TEXT_DARK, False),
            ("background", _KEY_BLUE, True),
        ]
    return [("Ego/world state", _KEY_RED, True)]


def _relation_segments(residual):
    descriptions = {
        "object_reprojection": ("Predicted image position", "observed object centroid"),
        "heldout_object_depth": ("Predicted object depth", "held-out depth evidence"),
        "heldout_object_backward_flow": (
            "Predicted object motion",
            "held-out RAFT backward flow",
        ),
        "heldout_background_backward_flow": (
            "Ego-predicted background flow",
            "held-out RAFT flow",
        ),
        "trajectory_temporal_gap": (
            "Expected continuous trajectory",
            "missing temporal support",
        ),
        "ego_acceleration_bound": (
            "Estimated ego acceleration",
            "physical plausibility bound",
        ),
        "object_acceleration_bound": (
            "Estimated object acceleration",
            "physical plausibility bound",
        ),
        "ego_speed_bound": ("Estimated ego speed", "physical plausibility bound"),
        "object_speed_bound": (
            "Estimated object speed",
            "physical plausibility bound",
        ),
        "semantic_static_motion": ("Predicted object motion", "semantic-static prior"),
        "track_endpoint_accounting": (
            "Track endpoint",
            "required lifecycle explanation",
        ),
    }
    predicted, observed = descriptions.get(
        residual.constraint_id,
        ("Predicted state", residual.metric_name.replace("_", " ") + " evidence"),
    )
    return [
        (predicted, _KEY_RED, True),
        ("  vs  ", _TEXT_DARK, False),
        (observed, _KEY_BLUE, True),
    ]


def _conflict_panel(
    *,
    frame: np.ndarray,
    packet,
    conflict,
    residual,
    tracking,
    step2_root: Path,
    step3_root: Path,
) -> np.ndarray:
    annotated = frame.copy()
    origin, track, _ = _draw_track_support(
        annotated,
        tracking=tracking,
        residual=residual,
        step2_root=step2_root,
        step3_root=step3_root,
    )
    _draw_prediction_observation(annotated, residual, origin)
    view = cv2.resize(annotated, (1260, 709), interpolation=cv2.INTER_AREA)
    canvas = np.full((_PANEL_HEIGHT, _PANEL_WIDTH, 3), 247, dtype=np.uint8)
    canvas[_HEADER_HEIGHT:, :_FRAME_REGION_WIDTH] = (22, 26, 33)
    canvas[250:959, 10:1270] = view
    canvas[:_HEADER_HEIGHT] = (24, 29, 38)
    family_color = _FAMILY_BGR[residual.family.value]
    cv2.rectangle(
        canvas,
        (0, _HEADER_HEIGHT - 6),
        (_PANEL_WIDTH, _HEADER_HEIGHT),
        family_color,
        -1,
    )
    cv2.line(
        canvas,
        (_FRAME_REGION_WIDTH, _HEADER_HEIGHT),
        (_FRAME_REGION_WIDTH, _PANEL_HEIGHT),
        (208, 212, 218),
        2,
    )
    title = f"Step 6 Conflict | Hypothesis Rank {packet.hypothesis_rank:02d} | Frame {residual.start_frame_index:06d}"
    cv2.putText(
        canvas,
        title,
        (28, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.88,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    subtitle = (
        f"{residual.family.value.replace('_', ' ').upper()}  /  "
        f"{residual.constraint_id.replace('_', ' ').upper()}  /  "
        f"{residual.severity.value.replace('_', ' ').upper()}"
    )
    cv2.putText(
        canvas,
        subtitle,
        (28, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    y = 154
    y = _section_label(canvas, y=y, label="CONFLICT SUBJECT", accent=family_color)
    y = _rich_bullet(
        canvas,
        _subject_segments(residual=residual, track=track),
        y=y,
    )
    component = residual.component_id.rsplit(":", 1)[-1] if residual.component_id else "none"
    y = _bullet_text(
        canvas,
        f"Component {component}; frame {residual.start_frame_index}; time {residual.start_timestamp_s:.2f} s",
        y=y,
        level=1,
    )
    y += 10
    y = _section_label(canvas, y=y, label="WHAT CONFLICTS", accent=_KEY_ORANGE)
    y = _rich_bullet(
        canvas,
        _relation_segments(residual),
        y=y,
    )
    y = _rich_bullet(
        canvas,
        [
            ("Predicted", _KEY_RED, False),
            (f": {_format_values(residual.predicted_values)}", _TEXT_MUTED, False),
        ],
        y=y,
        level=1,
    )
    y = _rich_bullet(
        canvas,
        [
            ("Observed", _KEY_BLUE, False),
            (f": {_format_values(residual.observed_values)}", _TEXT_MUTED, False),
        ],
        y=y,
        level=1,
    )
    flow_diagnosis = _flow_diagnosis(residual)
    if flow_diagnosis is not None:
        diagnosis, direction_error, magnitude_ratio = flow_diagnosis
        y = _rich_bullet(
            canvas,
            [
                (diagnosis, _KEY_ORANGE, True),
                (
                    f"  |  direction {direction_error:.1f} deg; magnitude {magnitude_ratio:.2f}x",
                    _TEXT_MUTED,
                    False,
                ),
            ],
            y=y,
            level=1,
        )
        if residual.constraint_id == "heldout_background_backward_flow":
            y = _bullet_text(
                canvas,
                "Arrows are spatial-median summaries; residual is median per-point endpoint error",
                y=y,
                level=1,
            )
    y += 10
    y = _section_label(canvas, y=y, label="WHY IT IS FLAGGED", accent=_KEY_RED)
    y = _bullet_text(
        canvas,
        "Residual exceeds the configured conflict threshold",
        y=y,
    )
    y = _bullet_text(
        canvas,
        f"Residual {float(residual.raw_residual):.3f} {residual.metric_unit}; threshold {float(residual.threshold):.3f}",
        y=y,
        level=1,
    )
    y = _bullet_text(
        canvas,
        f"Normalized z {float(residual.normalized_residual):.3f}; uncertainty {float(residual.uncertainty):.3f}",
        y=y,
        level=1,
    )
    y += 10
    y = _section_label(canvas, y=y, label="EVIDENCE", accent=_KEY_GREEN)
    evidence_role = residual.evidence_role.value if residual.evidence_role else "none"
    basis = residual.evaluation_basis.value.replace("_", " ")
    y = _rich_bullet(
        canvas,
        [
            (basis, _KEY_GREEN, True),
            ("  |  role: ", _TEXT_DARK, False),
            (evidence_role, _KEY_BLUE, True),
        ],
        y=y,
    )
    _bullet_text(
        canvas,
        f"Cue: {residual.cue_family.value if residual.cue_family else 'none'}; check-supported window: {str(conflict.check_evidence_supported).lower()}",
        y=y,
        level=1,
    )
    return canvas


def _conflict_audit(packet) -> dict:
    residual_by_id = {row.residual_id: row for row in packet.residuals}
    return {
        "schema_name": "step6_conflict_audit",
        "schema_version": 1,
        "hypothesis_id": packet.hypothesis_id,
        "hypothesis_rank": packet.hypothesis_rank,
        "selection_applied": False,
        "repair_applied": False,
        "conflicts": [
            {
                **window.model_dump(mode="json"),
                "residuals": [
                    residual_by_id[residual_id].model_dump(mode="json")
                    for residual_id in window.residual_ids
                ],
            }
            for window in packet.conflict_windows
        ],
    }


def _comparison(packets, path: Path) -> None:
    ranks = np.asarray([packet.hypothesis_rank for packet in packets])
    conflicts = np.asarray([len(packet.conflict_windows) for packet in packets])
    check_supported = np.asarray(
        [packet.check_supported_conflict_count for packet in packets]
    )
    hard = np.asarray([sum(row.hard_violation_count for row in packet.family_summaries) for packet in packets])
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=150)
    width = 0.24
    axes[0].bar(ranks - width, conflicts, width=width, label="all conflicts", color="#9ecae1")
    axes[0].bar(ranks, check_supported, width=width, label="check-supported", color="#3182bd")
    axes[0].bar(ranks + width, hard, width=width, label="hard violations", color="#de2d26")
    axes[0].set_xlabel("Step 5 hypothesis rank")
    axes[0].set_ylabel("Conflict count")
    axes[0].set_xticks(ranks)
    axes[0].set_title("Conflict accounting across the beam")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].plot(
        ranks,
        [packet.evaluable_fraction for packet in packets],
        "o-",
        color="#31a354",
        linewidth=2,
    )
    axes[1].set_xlabel("Step 5 hypothesis rank")
    axes[1].set_ylabel("Evaluable residual fraction")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(ranks)
    axes[1].set_title("Evidence coverage, not a selection score")
    axes[1].grid(alpha=0.2)
    figure.suptitle("Step 6 hypothesis comparison - no ranking or selection applied")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path)
    plt.close(figure)


def _timeline(packet, path: Path, *, conflict_threshold: float, hard_threshold: float):
    figure, axis = plt.subplots(figsize=(12.8, 5.4), dpi=150)
    evaluable = [row for row in packet.residuals if row.evaluable]
    for family, color in _FAMILY_COLORS.items():
        rows = [row for row in evaluable if row.family.value == family]
        if not rows:
            continue
        axis.scatter(
            [row.start_frame_index for row in rows],
            [max(float(row.normalized_residual), 1e-3) for row in rows],
            s=[42 if row.evaluation_basis.value == "check_evidence" else 20 for row in rows],
            marker="o",
            color=color,
            alpha=0.78,
            label=family.replace("_", " "),
        )
    axis.axhline(conflict_threshold, color="#b22222", linestyle="--", linewidth=1.4,
                 label=f"conflict threshold (z={conflict_threshold:g})")
    axis.axhline(hard_threshold, color="#5a0000", linestyle=":", linewidth=1.4,
                 label=f"hard threshold (z={hard_threshold:g})")
    axis.set_yscale("log")
    axis.set_xlabel("Canonical frame index")
    axis.set_ylabel("Normalized residual")
    axis.set_title(
        f"Hypothesis rank {packet.hypothesis_rank}: residual timeline | "
        f"status={packet.status} | check-only residuals={packet.check_evidence_residual_count}"
    )
    axis.grid(alpha=0.22)
    axis.legend(loc="upper right", ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _summary(packet, path: Path):
    families = [row.family.value.replace("_", "\n") for row in packet.family_summaries]
    evaluable = np.asarray([row.evaluable_count for row in packet.family_summaries])
    check = np.asarray([row.check_evidence_count for row in packet.family_summaries])
    violations = np.asarray([row.violation_count for row in packet.family_summaries])
    peaks = np.asarray([
        row.peak_normalized_residual if row.peak_normalized_residual is not None else 0.0
        for row in packet.family_summaries
    ])
    x = np.arange(len(families))
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=150)
    axes[0].bar(x, evaluable, color="#b8cbe5", label="evaluable")
    axes[0].bar(x, check, color="#386cb0", label="check-only")
    axes[0].bar(x, violations, color="#ef3b2c", width=0.45, label="violations")
    axes[0].set_xticks(x, families)
    axes[0].set_ylabel("Residual count")
    axes[0].set_title("Evidence and violation accounting")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(x, peaks, color=[_FAMILY_COLORS[row.family.value] for row in packet.family_summaries])
    axes[1].set_xticks(x, families)
    axes[1].set_ylabel("Peak normalized residual")
    axes[1].set_title("Peak residual by family")
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle(
        f"Step 6 verification packet - rank {packet.hypothesis_rank} | "
        f"evaluable={packet.evaluable_fraction:.1%} | conflicts={len(packet.conflict_windows)} | "
        f"check-supported={packet.check_supported_conflict_count}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path)
    plt.close(figure)


def render_step6_visualizations(
    *,
    residual_store_path: Path | str,
    maximum_hypotheses: int = 5,
    maximum_conflict_panels: int = 8,
) -> Path:
    """Render beam accounting and concrete conflict evidence without selection."""
    if maximum_hypotheses <= 0:
        raise ValueError("maximum_hypotheses must be positive")
    if maximum_conflict_panels < 0:
        raise ValueError("maximum_conflict_panels cannot be negative")
    store_path = Path(residual_store_path).expanduser().resolve()
    store = read_contract(store_path, ResidualStore)
    stage_root = store_path.parent
    run_root = _run_root(store_path)
    output_root = stage_root / "visualizations"
    output_root.mkdir(parents=True, exist_ok=True)
    videos = []
    for video_id, reference in zip(store.video_ids, store.video_residuals):
        source_path = stage_root / reference.relative_path
        if not source_path.is_file() or source_path.stat().st_size != reference.byte_size:
            raise RuntimeError(f"Step 6 residual manifest is missing or truncated: {source_path}")
        if sha256_file(source_path) != reference.sha256:
            raise RuntimeError(f"Step 6 residual manifest failed integrity check: {source_path}")
        manifest = read_contract(source_path, VideoResidualManifest)
        tracking, tracking_root, step2_root, provider = _load_tracking_and_frames(
            run_root=run_root, manifest=manifest
        )
        video_root = output_root / video_id
        video_root.mkdir(parents=True, exist_ok=True)
        selected_packets = manifest.packets[:maximum_hypotheses]
        comparison = video_root / "hypothesis_comparison.png"
        _comparison(selected_packets, comparison)
        packets = []
        frame_cache: dict[int, np.ndarray] = {}
        for packet in selected_packets:
            timeline = video_root / f"rank_{packet.hypothesis_rank:02d}_residual_timeline.png"
            summary = video_root / f"rank_{packet.hypothesis_rank:02d}_family_summary.png"
            conflict_root = video_root / f"rank_{packet.hypothesis_rank:02d}_conflicts"
            conflict_root.mkdir(parents=True, exist_ok=True)
            # This directory is owned by the renderer. Remove panels from an older
            # render so changed ordering or limits cannot leave mixed templates.
            for stale_panel in conflict_root.glob("*.png"):
                stale_panel.unlink()
            conflict_audit = video_root / f"rank_{packet.hypothesis_rank:02d}_conflicts.json"
            conflict_audit.write_text(
                json.dumps(
                    _conflict_audit(packet),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            _timeline(
                packet,
                timeline,
                conflict_threshold=store.config.conflict_z_threshold,
                hard_threshold=store.config.hard_z_threshold,
            )
            _summary(packet, summary)
            residual_by_id = {row.residual_id: row for row in packet.residuals}
            prioritized = sorted(
                packet.conflict_windows,
                key=lambda row: (
                    not row.check_evidence_supported,
                    row.severity.value != "hard_violation",
                    -row.peak_normalized_residual,
                    row.start_frame_index,
                ),
            )[:maximum_conflict_panels]
            representative = {
                conflict.conflict_id: max(
                    (residual_by_id[residual_id] for residual_id in conflict.residual_ids),
                    key=lambda row: float(row.normalized_residual or 0.0),
                )
                for conflict in prioritized
            }
            requested_frames = sorted(
                {row.start_frame_index for row in representative.values()}
            )
            for frame_index in requested_frames:
                if frame_index not in frame_cache:
                    frame_cache[frame_index] = provider.get_frame(frame_index).image_bgr
            decoded = {frame_index: frame_cache[frame_index] for frame_index in requested_frames}
            conflict_paths = []
            panel_images = []
            for index, conflict in enumerate(prioritized, start=1):
                residual = representative[conflict.conflict_id]
                panel = _conflict_panel(
                    frame=decoded[residual.start_frame_index],
                    packet=packet,
                    conflict=conflict,
                    residual=residual,
                    tracking=tracking,
                    step2_root=step2_root,
                    step3_root=tracking_root,
                )
                panel_path = conflict_root / (
                    f"conflict_{index:02d}_frame_{residual.start_frame_index:06d}_"
                    f"{_safe_name(conflict.constraint_id)}.png"
                )
                _write_image(panel_path, panel)
                conflict_paths.append(panel_path)
                panel_images.append(panel)
            contact_sheet = None
            if panel_images:
                tiles = panel_images[:4]
                while len(tiles) < 4:
                    tiles.append(
                        np.full((_PANEL_HEIGHT, _PANEL_WIDTH, 3), 247, dtype=np.uint8)
                    )
                sheet = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:4])))
                contact_sheet = (
                    video_root
                    / f"rank_{packet.hypothesis_rank:02d}_conflict_overview.png"
                )
                _write_image(contact_sheet, sheet)
            packets.append({
                "hypothesis_id": packet.hypothesis_id,
                "hypothesis_rank": packet.hypothesis_rank,
                "status": packet.status,
                "evaluable_fraction": packet.evaluable_fraction,
                "check_evidence_residual_count": packet.check_evidence_residual_count,
                "conflict_window_count": len(packet.conflict_windows),
                "rendered_conflict_panel_count": len(conflict_paths),
                "check_supported_conflict_count": packet.check_supported_conflict_count,
                "timeline": timeline.relative_to(output_root).as_posix(),
                "family_summary": summary.relative_to(output_root).as_posix(),
                "conflict_audit": conflict_audit.relative_to(output_root).as_posix(),
                "conflict_panels": [
                    path.relative_to(output_root).as_posix() for path in conflict_paths
                ],
                "conflict_overview": (
                    contact_sheet.relative_to(output_root).as_posix()
                    if contact_sheet is not None
                    else None
                ),
                "conflict_panel_resolution": [_PANEL_WIDTH, _PANEL_HEIGHT],
                "conflict_overview_resolution": [
                    2 * _PANEL_WIDTH,
                    2 * _PANEL_HEIGHT,
                ],
            })
        videos.append({
            "video_id": video_id,
            "hypothesis_comparison": comparison.relative_to(output_root).as_posix(),
            "packets": packets,
        })
    manifest_path = output_root / "step6_visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_name": "step6_visualization_manifest",
                "schema_version": 2,
                "source_residual_store_sha256": sha256_file(store_path),
                "selection_applied": False,
                "repair_applied": False,
                "videos": videos,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return manifest_path


__all__ = ["render_step6_visualizations"]
