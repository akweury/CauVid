"""Auditable Step 8 parent/child and objective visualizations."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from src.exp_august.contracts import (
    LocalReestimationStore,
    VideoLocalReestimationManifest,
    VideoManifest,
    VideoRepairProposalManifest,
    VideoResidualManifest,
    VideoTrackingManifest,
    VideoWorldStateManifest,
)
from src.exp_august.contracts.codec import read_contract, sha256_file
from src.exp_august.inference.frames import CanonicalFrameProvider


_WIDTH = 1920
_HEIGHT = 1220
_FRAME_HEIGHT = 448
_FRAME_GAP = 24
_FRAME_WIDTH = (_WIDTH - 84 - _FRAME_GAP) // 2

_WHITE = (250, 250, 250)
_DARK = (34, 38, 45)
_MUTED = (87, 93, 102)
_GRID = (219, 222, 226)
_PARENT = (196, 115, 35)
# High-contrast Okabe-Ito-inspired colors (stored as OpenCV BGR).  Candidate
# identity should remain legible without relying on several shades of green.
_CHILDREN = (
    (115, 158, 0),    # bluish green
    (0, 94, 213),     # vermillion
    (167, 121, 204),  # reddish purple
    (0, 159, 230),    # orange
    (178, 114, 0),    # blue
    (233, 180, 86),   # sky blue
)
_WARNING = (37, 126, 214)
_ERROR = (58, 67, 196)
_REASON_COLORS = {
    "identity_error": (115, 158, 0),
    "mask_error": (0, 94, 213),
    "depth_jump": (167, 121, 204),
    "pose_drift": (178, 114, 0),
    "scale_ambiguity": (0, 159, 230),
    "invalid_static_background_assumption": (233, 180, 86),
    "dynamics_mismatch": (42, 181, 211),
    "true_acute_maneuver": (164, 82, 219),
    "unobservable_evidence": (112, 112, 112),
    "semantic_prior_mismatch": (148, 133, 0),
    "unresolved_conflict": (58, 67, 196),
}


def _run_root(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "08_local_reestimation":
            return parent.parent
    raise RuntimeError("Step 8 store must live below 08_local_reestimation")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "proposal"


def _verified_contract(path: Path, reference, model, *, label: str):
    if not path.is_file() or path.stat().st_size != reference.byte_size:
        raise RuntimeError(
            f"Step 8 visualization {label} is missing or truncated: {path}"
        )
    if sha256_file(path) != reference.sha256:
        raise RuntimeError(
            f"Step 8 visualization {label} failed integrity check: {path}"
        )
    return read_contract(path, model)


def _load_context(*, run_root: Path, manifest: VideoLocalReestimationManifest):
    snapshot = manifest.input_snapshot
    repair_path = (
        run_root
        / snapshot.source_step7_relative_root
        / snapshot.video_repair_proposal_manifest.artifact.relative_path
    )
    repair = _verified_contract(
        repair_path,
        snapshot.video_repair_proposal_manifest.artifact,
        VideoRepairProposalManifest,
        label="repair proposal manifest",
    )
    residual_path = (
        run_root
        / snapshot.source_step6_relative_root
        / snapshot.video_residual_manifest.artifact.relative_path
    )
    residual = _verified_contract(
        residual_path,
        snapshot.video_residual_manifest.artifact,
        VideoResidualManifest,
        label="residual manifest",
    )
    world_path = (
        run_root
        / snapshot.source_step5_relative_root
        / snapshot.video_world_state_manifest.artifact.relative_path
    )
    world = _verified_contract(
        world_path,
        snapshot.video_world_state_manifest.artifact,
        VideoWorldStateManifest,
        label="world-state manifest",
    )
    tracking_path = (
        run_root
        / snapshot.source_step3_relative_root
        / snapshot.video_tracking_manifest.artifact.relative_path
    )
    tracking = _verified_contract(
        tracking_path,
        snapshot.video_tracking_manifest.artifact,
        VideoTrackingManifest,
        label="tracking manifest",
    )
    source_link = tracking.input_snapshot.source_video_manifest
    source_path = run_root / "01_init" / source_link.artifact.relative_path
    source = _verified_contract(
        source_path,
        source_link.artifact,
        VideoManifest,
        label="source video manifest",
    )
    return repair, residual, world, tracking, CanonicalFrameProvider(
        source, verify_source_hash=True
    )


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write Step 8 visualization: {path}")


def _put(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.72,
    color: tuple[int, int, int] = _DARK,
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        str(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _fit_frame(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    canvas = np.full((_FRAME_HEIGHT, _FRAME_WIDTH, 3), 24, dtype=np.uint8)
    source_height, source_width = image.shape[:2]
    scale = min(_FRAME_WIDTH / source_width, _FRAME_HEIGHT / source_height)
    rendered = cv2.resize(
        image,
        (
            max(1, int(round(source_width * scale))),
            max(1, int(round(source_height * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )
    top = (_FRAME_HEIGHT - rendered.shape[0]) // 2
    left = (_FRAME_WIDTH - rendered.shape[1]) // 2
    canvas[top : top + rendered.shape[0], left : left + rendered.shape[1]] = rendered
    return canvas, scale, left, top


def _subjects(proposal, residual_packet) -> tuple[tuple[str, ...], tuple[str, ...]]:
    residuals = {row.residual_id: row for row in residual_packet.residuals}
    rows = [residuals[value] for value in proposal.target_residual_ids if value in residuals]
    components = tuple(
        dict.fromkeys(row.component_id for row in rows if row.component_id is not None)
    )
    tracks = tuple(dict.fromkeys(row.track_id for row in rows if row.track_id is not None))
    return components, tracks


def _target_scope(
    proposal, component_ids: tuple[str, ...], track_ids: tuple[str, ...]
) -> str:
    affects_ego = any(
        value.startswith("ego_components") for value in proposal.affected_variables
    )
    affects_objects = any(
        value.startswith("object_trajectories") for value in proposal.affected_variables
    )
    if affects_ego and affects_objects:
        subject = "ego/camera + tracked-object motion"
    elif affects_ego:
        subject = "ego/camera motion"
    elif affects_objects:
        subject = "tracked-object motion"
    else:
        subject = "world-state variables"
    identifiers = []
    if component_ids:
        identifiers.append("component " + ",".join(component_ids))
    if track_ids:
        identifiers.append("track " + ",".join(track_ids))
    return subject + (" | " + " | ".join(identifiers) if identifiers else "")


def _target_kind(
    proposal,
    component_ids: tuple[str, ...] = (),
    track_ids: tuple[str, ...] = (),
) -> str:
    affects_ego = any(
        value.startswith("ego_components") for value in proposal.affected_variables
    )
    affects_objects = any(
        value.startswith("object_trajectories") for value in proposal.affected_variables
    )
    if affects_ego and affects_objects:
        return "ego_and_object"
    if affects_ego:
        return "ego"
    if affects_objects:
        return "object"
    if track_ids:
        return "object"
    if component_ids:
        return "ego"
    return "state"


def _draw_target_frame(
    canvas: np.ndarray,
    *,
    frame,
    tracking,
    track_ids: tuple[str, ...],
    target_kind: str,
) -> None:
    image, scale, pad_x, pad_y = _fit_frame(frame.image_bgr)
    if target_kind in {"object", "ego_and_object"} and track_ids:
        track = next(
            (row for row in tracking.tracks if row.track_id == track_ids[0]), None
        )
        observation = _track_observation(tracking, track_ids, frame.frame_index)
        if observation is not None:
            _, _, box = _anchor(
                observation=observation,
                scale=scale,
                left=pad_x,
                top=pad_y,
                frame_shape=image.shape,
            )
            if box is not None:
                color = _candidate_color(0)
                cv2.rectangle(image, box[:2], box[2:], color, 5)
                class_name = track.primary_class if track is not None else "OBJECT"
                label = f"{class_name} | {track_ids[0]}"
                label_width = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2
                )[0][0]
                label_y = max(32, box[1])
                cv2.rectangle(
                    image,
                    (box[0], label_y - 30),
                    (min(_FRAME_WIDTH - 1, box[0] + label_width + 18), label_y + 4),
                    color,
                    -1,
                )
                _put(
                    image,
                    label,
                    (box[0] + 8, label_y - 7),
                    scale=0.62,
                    color=(255, 255, 255),
                )
    if target_kind in {"ego", "ego_and_object"}:
        cv2.rectangle(image, (18, 16), (116, 62), _PARENT, -1)
        _put(
            image,
            "EGO",
            (34, 49),
            scale=0.88,
            color=(255, 255, 255),
            thickness=2,
        )
    canvas[100 : 100 + _FRAME_HEIGHT, 30 : 30 + _FRAME_WIDTH] = image


def _series_rows(proposal, residual_packet):
    by_id = {row.residual_id: row for row in residual_packet.residuals}
    targets = [
        by_id[value]
        for value in proposal.target_residual_ids
        if value in by_id and by_id[value].evaluable
    ]
    if not targets:
        return ()
    metric = max(
        {row.metric_name for row in targets},
        key=lambda name: (
            sum(row.metric_name == name for row in targets),
            max(
                (
                    float(row.normalized_residual or 0.0)
                    for row in targets
                    if row.metric_name == name
                ),
                default=0.0,
            ),
        ),
    )
    component_ids = {row.component_id for row in targets if row.component_id is not None}
    track_ids = {row.track_id for row in targets if row.track_id is not None}
    rows = [
        row
        for row in residual_packet.residuals
        if row.evaluable
        and row.metric_name == metric
        and proposal.start_frame_index <= row.start_frame_index <= proposal.end_frame_index
        and (not component_ids or row.component_id in component_ids)
        and (not track_ids or row.track_id in track_ids)
    ]
    return tuple(sorted(rows or targets, key=lambda row: row.start_frame_index))


def _value_magnitude(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.reshape(-1)[0])
        if array.size == 1
        else float(np.linalg.norm(array))
    )


def _aggregate_series(rows, attribute: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.start_frame_index, []).append(
            _value_magnitude(getattr(row, attribute))
        )
    return {
        frame: float(np.median(values)) for frame, values in sorted(grouped.items())
    }


def _draw_cause_chart(
    canvas: np.ndarray, *, proposal, residual_packet, reason_color
) -> None:
    panel_left, panel_right = 990, 1880
    panel_top, panel_bottom = 100, 548
    rows = _series_rows(proposal, residual_packet)
    cv2.rectangle(
        canvas, (panel_left, panel_top), (panel_right, panel_bottom), _GRID, 2
    )
    cv2.rectangle(
        canvas, (panel_left, panel_top), (panel_left + 8, panel_bottom), reason_color, -1
    )
    if not rows:
        _put(
            canvas,
            "NO COMPARABLE RESIDUAL SERIES",
            (panel_left + 190, panel_top + 230),
            scale=0.78,
            color=_MUTED,
        )
        return
    current = _aggregate_series(rows, "raw_residual")
    acceptable = _aggregate_series(rows, "threshold")
    frames = sorted(set(current) & set(acceptable))
    if not frames:
        return
    values = [current[frame] for frame in frames] + [acceptable[frame] for frame in frames]
    low, high = min(values), max(values)
    padding = max((high - low) * 0.12, abs(high) * 0.05, 1e-6)
    low, high = low - padding, high + padding
    plot_left, plot_right = panel_left + 78, panel_right - 28
    plot_top, plot_bottom = panel_top + 92, panel_bottom - 54

    def to_x(frame_index: int) -> int:
        return int(
            round(
                plot_left
                + (plot_right - plot_left)
                * (frame_index - proposal.start_frame_index)
                / max(1, proposal.end_frame_index - proposal.start_frame_index)
            )
        )

    def to_y(value: float) -> int:
        return int(round(plot_bottom - (plot_bottom - plot_top) * (value - low) / (high - low)))

    for fraction in (0.0, 0.5, 1.0):
        y = int(round(plot_bottom - fraction * (plot_bottom - plot_top)))
        cv2.line(canvas, (plot_left, y), (plot_right, y), _GRID, 1)
        _put(
            canvas,
            f"{low + fraction * (high - low):.3g}",
            (panel_left + 8, y + 6),
            scale=0.48,
            color=_MUTED,
            thickness=1,
        )
    cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), _DARK, 2)
    for values_by_frame, color in ((current, reason_color), (acceptable, _DARK)):
        points = np.asarray(
            [(to_x(frame), to_y(values_by_frame[frame])) for frame in frames],
            dtype=np.int32,
        )
        if len(points) > 1:
            cv2.polylines(canvas, [points], False, color, 5, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, tuple(point), 7, color, -1, cv2.LINE_AA)
    metric = rows[0].metric_name.replace("_", " ").upper()
    unit = rows[0].metric_unit.replace("_", " ")
    _put(
        canvas,
        f"CAUSE | {metric} ({unit})",
        (panel_left + 24, panel_top + 37),
        scale=0.72,
        color=reason_color,
    )
    _put(canvas, "CURRENT ERROR", (panel_right - 340, panel_top + 72), scale=0.55, color=reason_color)
    _put(
        canvas,
        "ACCEPTABLE LIMIT",
        (panel_right - 165, panel_top + 72),
        scale=0.55,
        color=_DARK,
    )
    _put(canvas, str(proposal.start_frame_index), (plot_left - 8, plot_bottom + 34), scale=0.48, color=_MUTED, thickness=1)
    _put(canvas, str(proposal.end_frame_index), (plot_right - 24, plot_bottom + 34), scale=0.48, color=_MUTED, thickness=1)


def _draw_effect_chart(canvas: np.ndarray, *, result) -> None:
    left, right = 30, 1880
    top, bottom = 580, 895
    candidates = _instantiated(result)[:6]
    cv2.rectangle(canvas, (left, top), (right, bottom), _GRID, 2)
    _put(canvas, "EFFECT | OBJECTIVE ERROR", (left + 20, top + 38), scale=0.74)
    _put(canvas, "LOWER", (left + 405, top + 38), scale=0.56, color=_candidate_color(0))
    if not candidates:
        _put(canvas, result.status.replace("_", " ").upper(), (left + 720, top + 170), scale=0.9, color=_WARNING)
        return
    maximum = max(
        1e-12,
        max(float(row.objective_before.total) for row in candidates),
        max(float(row.objective_after.total) for row in candidates),
    )
    chart_left, chart_right = left + 235, right - 72
    row_height = min(42, 220 // len(candidates))
    for index, candidate in enumerate(candidates):
        y = top + 78 + index * row_height
        before = float(candidate.objective_before.total)
        after = float(candidate.objective_after.total)
        before_x = chart_left + int((chart_right - chart_left) * before / maximum)
        after_x = chart_left + int((chart_right - chart_left) * after / maximum)
        color = _candidate_color(index)
        _put(canvas, f"C{index + 1:02d}", (left + 24, y + 7), scale=0.56, color=color)
        cv2.line(canvas, (chart_left, y), (before_x, y), _PARENT, 8, cv2.LINE_AA)
        cv2.circle(canvas, (before_x, y), 8, _PARENT, -1, cv2.LINE_AA)
        cv2.arrowedLine(
            canvas,
            (before_x, y),
            (after_x, y),
            color,
            5,
            cv2.LINE_AA,
            tipLength=0.025,
        )
        cv2.circle(canvas, (after_x, y), 9, color, -1, cv2.LINE_AA)
        _put(canvas, f"{before:.3g}", (max(chart_left, before_x - 54), y - 11), scale=0.44, color=_PARENT, thickness=1)
        _put(canvas, f"{after:.3g}", (max(chart_left, after_x - 24), y + 23), scale=0.44, color=color, thickness=1)


def _draw_audit_table(
    canvas: np.ndarray, *, proposal, result, target_kind: str
) -> None:
    top = 925
    columns = (30, 190, 600, 790, 1080, 1235, 1410, 1600, 1885)
    headers = ("CANDIDATE", "CHANGED TARGET", "MAX DELTA", "OBJECTIVE", "FRAMES", "BOUND", "BOUNDARY", "STATUS")
    table_bottom = 1138
    cv2.rectangle(canvas, (30, top), (1885, table_bottom), _GRID, 2)
    for index, header in enumerate(headers):
        _put(canvas, header, (columns[index] + 8, top + 27), scale=0.48, color=_MUTED, thickness=1)
    cv2.line(canvas, (30, top + 40), (1885, top + 40), _GRID, 2)
    changed_target = {
        "ego": "ego position -> velocity/speed",
        "object": "object position -> velocity/speed",
        "ego_and_object": "ego + object dynamics",
        "state": "world state",
    }[target_kind]
    candidates = _instantiated(result)[:6]
    rows = candidates or result.candidates[:1]
    bound = _maximum_sigma(proposal)
    for index, candidate in enumerate(rows):
        y0 = top + 40 + index * 29
        y = y0 + 21
        if y0 + 29 > table_bottom:
            break
        if index:
            cv2.line(canvas, (30, y0), (1885, y0), _GRID, 1)
        color = _candidate_color(index) if candidate.status == "instantiated" else _WARNING
        sigma = max(
            (
                float(change.maximum_standardized_delta)
                for change in candidate.numerical_changes
                if change.maximum_standardized_delta is not None
            ),
            default=0.0,
        )
        changed_frames = len({row.frame_index for row in candidate.numerical_changes})
        objective = (
            f"{candidate.objective_before.total:.3g} -> {candidate.objective_after.total:.3g}"
            if candidate.objective_before is not None and candidate.objective_after is not None
            else "-"
        )
        values = (
            f"C{index + 1:02d}",
            changed_target,
            f"{sigma:.2f} sigma",
            objective,
            str(changed_frames),
            f"<= {bound:g} sigma" if bound is not None else "-",
            "yes" if candidate.boundary_preserved else "no",
            "not selected" if candidate.status == "instantiated" else candidate.status.replace("_", " "),
        )
        for column, value in enumerate(values):
            _put(
                canvas,
                value,
                (columns[column] + 8, y),
                scale=0.46,
                color=color if column in {0, 2, 3} else _DARK,
                thickness=1 if column not in {0, 2, 3} else 2,
            )


def _draw_reason_row(canvas: np.ndarray, *, active_diagnosis, diagnoses) -> None:
    top, bottom = 1152, 1202
    categories = _reason_categories(active_diagnosis, diagnoses)
    cv2.rectangle(canvas, (30, top), (1885, bottom), _GRID, 2)
    _put(canvas, "CAUSES", (44, top + 32), scale=0.54, color=_DARK)
    labels = [category.value.replace("_", " ").upper() for category in categories]
    available = 1885 - 160
    scale = 0.50
    widths = [
        cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
        for label in labels
    ]
    required = sum(width + 42 for width in widths)
    if required > available:
        scale = max(0.28, scale * available / required)
        widths = [
            cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
            for label in labels
        ]
    x = 154
    for category, label, width in zip(categories, labels, widths):
        color = _reason_color(category)
        active = category == active_diagnosis.category
        cv2.rectangle(
            canvas,
            (x, top + 15),
            (x + (22 if active else 16), top + 35),
            color,
            -1,
        )
        text_x = x + (30 if active else 24)
        _put(
            canvas,
            label,
            (text_x, top + 33),
            scale=scale,
            color=color,
            thickness=2 if active else 1,
        )
        x = text_x + width + 22


def _track_observation(tracking, track_ids: tuple[str, ...], frame_index: int):
    if not track_ids:
        return None
    track = next((row for row in tracking.tracks if row.track_id == track_ids[0]), None)
    if track is None or not track.observations:
        return None
    return min(
        track.observations,
        key=lambda row: (abs(row.frame_index - frame_index), row.frame_index),
    )


def _candidate_color(index: int) -> tuple[int, int, int]:
    return _CHILDREN[index % len(_CHILDREN)]


def _reason_color(category) -> tuple[int, int, int]:
    value = category.value if hasattr(category, "value") else str(category)
    return _REASON_COLORS.get(value, _MUTED)


def _reason_categories(active_diagnosis, diagnoses) -> tuple[object, ...]:
    ordered = [active_diagnosis.category]
    ordered.extend(row.category for row in diagnoses)
    ordered.extend(
        alternative
        for row in diagnoses
        for alternative in row.alternative_categories
    )
    return tuple(dict.fromkeys(ordered))


def _instantiated(result):
    return tuple(row for row in result.candidates if row.status == "instantiated")


def _representative_candidate(result):
    candidates = _instantiated(result)
    if not candidates:
        return result.candidates[0]

    def magnitude(candidate) -> float:
        values = []
        for change in candidate.numerical_changes:
            before = np.asarray(change.before_values, dtype=np.float64)
            after = np.asarray(change.after_values, dtype=np.float64)
            values.append(float(np.linalg.norm(after - before)))
        return max(values, default=float(len(candidate.discrete_changes)))

    return max(candidates, key=magnitude)


def _display_change(candidate, proposal) -> tuple[int, object | None]:
    position_changes = [
        row
        for row in candidate.numerical_changes
        if row.field_path.endswith(".position")
        and proposal.start_frame_index <= row.frame_index <= proposal.end_frame_index
    ]
    changes = position_changes or list(candidate.numerical_changes)
    if not changes:
        return (proposal.start_frame_index + proposal.end_frame_index) // 2, None
    change = max(
        changes,
        key=lambda row: float(
            np.linalg.norm(
                np.asarray(row.after_values, dtype=np.float64)
                - np.asarray(row.before_values, dtype=np.float64)
            )
        ),
    )
    return change.frame_index, change


def _anchor(
    *, observation, scale: float, left: int, top: int, frame_shape
) -> tuple[int, int, tuple[int, int, int, int] | None]:
    if observation is None:
        return _FRAME_WIDTH // 2, _FRAME_HEIGHT // 2 + 24, None
    box = observation.bbox
    x1 = int(round(left + box.x1 * scale))
    y1 = int(round(top + box.y1 * scale))
    x2 = int(round(left + box.x2 * scale))
    y2 = int(round(top + box.y2 * scale))
    x1, x2 = sorted((max(0, x1), min(frame_shape[1] - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(frame_shape[0] - 1, y2)))
    return (x1 + x2) // 2, (y1 + y2) // 2, (x1, y1, x2, y2)


def _maximum_sigma(proposal) -> float | None:
    for bound in proposal.parameter_bounds:
        if bound.parameter_name in {
            "maximum_pose_delta_sigma",
            "maximum_state_delta_sigma",
        }:
            return float(bound.upper_bound)
    return None


def _proposal_panel(
    *,
    proposal_number: int,
    proposal,
    result,
    residual_packet,
    tracking,
    frame,
    parent_rank: int,
    diagnosis,
    diagnoses,
) -> np.ndarray:
    canvas = np.full((_HEIGHT, _WIDTH, 3), _WHITE, dtype=np.uint8)
    component_ids, track_ids = _subjects(proposal, residual_packet)
    target_kind = _target_kind(proposal, component_ids, track_ids)
    status_color = (
        _candidate_color(0)
        if result.status == "candidates_generated"
        else _ERROR if result.status == "unsupported" else _WARNING
    )
    cv2.rectangle(canvas, (0, 0), (_WIDTH, 74), _DARK, -1)
    _put(
        canvas,
        f"STEP 8  |  {proposal.operator.value.replace('_', ' ').upper()}  |  PROPOSAL {proposal_number:02d}",
        (30, 47),
        scale=0.82,
        color=(255, 255, 255),
        thickness=2,
    )
    _put(
        canvas,
        f"RANK {parent_rank:02d}  |  FRAMES {proposal.start_frame_index}-{proposal.end_frame_index}  |  {result.status.replace('_', ' ').upper()}",
        (1290, 47),
        scale=0.55,
        color=status_color,
        thickness=2,
    )
    _draw_target_frame(
        canvas,
        frame=frame,
        tracking=tracking,
        track_ids=track_ids,
        target_kind=target_kind,
    )
    _draw_cause_chart(
        canvas,
        proposal=proposal,
        residual_packet=residual_packet,
        reason_color=_reason_color(diagnosis.category),
    )
    _draw_effect_chart(canvas, result=result)
    _draw_audit_table(
        canvas, proposal=proposal, result=result, target_kind=target_kind
    )
    _draw_reason_row(
        canvas, active_diagnosis=diagnosis, diagnoses=diagnoses
    )
    return canvas


def render_step8_visualizations(
    *,
    local_reestimation_store_path: Path | str,
    maximum_hypotheses: int = 5,
    maximum_proposal_panels: int = 8,
) -> Path:
    """Render one compact, quantitative panel for each Step 8 proposal."""

    if maximum_hypotheses <= 0:
        raise ValueError("maximum_hypotheses must be positive")
    if maximum_proposal_panels < 0:
        raise ValueError("maximum_proposal_panels cannot be negative")
    store_path = Path(local_reestimation_store_path).expanduser().resolve()
    store = read_contract(store_path, LocalReestimationStore)
    stage_root = store_path.parent
    run_root = _run_root(store_path)
    output_root = stage_root / "visualizations"
    output_root.mkdir(parents=True, exist_ok=True)
    videos = []
    for video_id, reference in zip(
        store.video_ids, store.video_local_reestimations
    ):
        manifest_path = stage_root / reference.relative_path
        manifest = _verified_contract(
            manifest_path,
            reference,
            VideoLocalReestimationManifest,
            label="local re-estimation manifest",
        )
        repair, residual, world, tracking, provider = _load_context(
            run_root=run_root, manifest=manifest
        )
        repair_packets = {row.hypothesis_id: row for row in repair.packets}
        residual_packets = {row.hypothesis_id: row for row in residual.packets}
        parents = {row.hypothesis_id: row for row in world.initial_beam.hypotheses}
        video_root = output_root / video_id
        video_root.mkdir(parents=True, exist_ok=True)
        packet_rows = []
        frame_cache = {}
        for packet in manifest.packets[:maximum_hypotheses]:
            proposal_root = video_root / f"rank_{packet.parent_hypothesis_rank:02d}_proposals"
            proposal_root.mkdir(parents=True, exist_ok=True)
            for stale in proposal_root.glob("*.png"):
                stale.unlink()
            repair_packet = repair_packets[packet.parent_hypothesis_id]
            proposal_by_id = {row.proposal_id: row for row in repair_packet.proposals}
            diagnosis_by_id = {
                row.diagnosis_id: row for row in repair_packet.diagnoses
            }
            residual_packet = residual_packets[packet.parent_hypothesis_id]
            rendered = []
            for proposal_number, result in enumerate(
                packet.proposal_results[:maximum_proposal_panels], start=1
            ):
                proposal = proposal_by_id[result.proposal_id]
                representative = _representative_candidate(result)
                frame_index, _ = _display_change(representative, proposal)
                frame_index = int(np.clip(frame_index, 0, manifest.frame_count - 1))
                if frame_index not in frame_cache:
                    frame_cache[frame_index] = provider.get_frame(frame_index)
                panel = _proposal_panel(
                    proposal_number=proposal_number,
                    proposal=proposal,
                    result=result,
                    residual_packet=residual_packet,
                    tracking=tracking,
                    frame=frame_cache[frame_index],
                    parent_rank=parents[packet.parent_hypothesis_id].rank,
                    diagnosis=diagnosis_by_id[proposal.diagnosis_id],
                    diagnoses=repair_packet.diagnoses,
                )
                panel_path = proposal_root / (
                    f"proposal_{proposal_number:02d}_frame_{frame_index:06d}_"
                    f"{_safe_name(proposal.operator.value)}.png"
                )
                _write_image(panel_path, panel)
                rendered.append(
                    {
                        "proposal_id": proposal.proposal_id,
                        "operator": proposal.operator.value,
                        "diagnosis_category": diagnosis_by_id[
                            proposal.diagnosis_id
                        ].category.value,
                        "diagnosis_rationale": diagnosis_by_id[
                            proposal.diagnosis_id
                        ].rationale,
                        "affected_variables": list(proposal.affected_variables),
                        "target_scope": _target_scope(
                            proposal,
                            *_subjects(proposal, residual_packet),
                        ),
                        "result_status": result.status,
                        "child_candidate_count": sum(
                            row.status == "instantiated" for row in result.candidates
                        ),
                        "source_frame_index": frame_index,
                        "panel": panel_path.relative_to(output_root).as_posix(),
                    }
                )
            packet_rows.append(
                {
                    "parent_hypothesis_id": packet.parent_hypothesis_id,
                    "parent_hypothesis_rank": packet.parent_hypothesis_rank,
                    "proposal_count": len(packet.proposal_results),
                    "rendered_proposal_count": len(rendered),
                    "proposals": rendered,
                    "proposal_overview": None,
                    "proposal_panel_resolution": [_WIDTH, _HEIGHT],
                    "proposal_overview_resolution": [0, 0],
                }
            )
        videos.append({"video_id": video_id, "packets": packet_rows})
    output_path = output_root / "step8_visualization_manifest.json"
    output_path.write_text(
        json.dumps(
            {
                "schema_name": "step8_visualization_manifest",
                "schema_version": 1,
                "source_local_reestimation_store_sha256": sha256_file(store_path),
                "parent_state_mutated": False,
                "raw_evidence_mutated": False,
                "selection_applied": False,
                "check_evidence_optimized": False,
                "videos": videos,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


__all__ = ["render_step8_visualizations"]
