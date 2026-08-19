"""Auditable Step 8 parent/child and objective visualizations."""

from __future__ import annotations

import json
import re
import textwrap
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
_HEADER_BOTTOM = 104
_FRAME_TOP = 124
_FRAME_HEIGHT = 448
_FRAME_GAP = 24
_FRAME_WIDTH = (_WIDTH - 84 - _FRAME_GAP) // 2
_CHART_TOP = 632
_CHART_BOTTOM = 956
_TEXT_TOP = 988

_WHITE = (250, 250, 250)
_DARK = (34, 38, 45)
_MUTED = (87, 93, 102)
_GRID = (219, 222, 226)
_PARENT = (196, 115, 35)
_CHILDREN = (
    (79, 154, 77),
    (47, 139, 92),
    (39, 122, 109),
    (48, 151, 139),
    (67, 165, 154),
    (83, 178, 166),
)
_WARNING = (37, 126, 214)
_ERROR = (58, 67, 196)


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


def _label_value(
    image: np.ndarray,
    *,
    label: str,
    value: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.72,
) -> None:
    label_text = f"{label}: "
    _put(image, label_text, origin, scale=scale, color=_DARK)
    width = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2
    )[0][0]
    _put(image, value, (origin[0] + width, origin[1]), scale=scale, color=color)


def _wrapped(value: str, width: int) -> list[str]:
    return textwrap.wrap(str(value), width=width, break_long_words=False) or [""]


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


def _frame_banner(image: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 56), color, -1)
    cv2.addWeighted(overlay, 0.90, image, 0.10, 0.0, image)
    _put(image, text, (20, 39), scale=0.75, color=(255, 255, 255), thickness=2)


def _subjects(proposal, residual_packet) -> tuple[tuple[str, ...], tuple[str, ...]]:
    residuals = {row.residual_id: row for row in residual_packet.residuals}
    rows = [residuals[value] for value in proposal.target_residual_ids if value in residuals]
    components = tuple(
        dict.fromkeys(row.component_id for row in rows if row.component_id is not None)
    )
    tracks = tuple(dict.fromkeys(row.track_id for row in rows if row.track_id is not None))
    return components, tracks


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


def _arrow_delta(change) -> tuple[int, int]:
    if change is None:
        return 92, -50
    before = np.asarray(change.before_values, dtype=np.float64)
    after = np.asarray(change.after_values, dtype=np.float64)
    delta = after - before
    dx = float(delta[0]) if delta.size else 0.0
    dy = float(-(delta[2] if delta.size >= 3 else delta[1] if delta.size >= 2 else 0.0))
    vector = np.asarray((dx, dy), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        vector = np.asarray((1.0, -0.55), dtype=np.float64)
        norm = float(np.linalg.norm(vector))
    sigma = float(change.maximum_standardized_delta or 1.0)
    length = float(np.clip(54.0 + 34.0 * sigma, 64.0, 152.0))
    vector = vector / norm * length
    return int(round(vector[0])), int(round(vector[1]))


def _draw_frame_pair(
    canvas: np.ndarray,
    *,
    frame,
    tracking,
    track_ids,
    candidate,
    change,
) -> None:
    left_image, scale, pad_x, pad_y = _fit_frame(frame.image_bgr)
    right_image = left_image.copy()
    observation = _track_observation(tracking, track_ids, frame.frame_index)
    anchor_x, anchor_y, box = _anchor(
        observation=observation,
        scale=scale,
        left=pad_x,
        top=pad_y,
        frame_shape=left_image.shape,
    )
    if box is not None:
        cv2.rectangle(left_image, box[:2], box[2:], _PARENT, 5)
        cv2.rectangle(right_image, box[:2], box[2:], _PARENT, 4)
    cv2.circle(left_image, (anchor_x, anchor_y), 13, _PARENT, -1, cv2.LINE_AA)
    cv2.circle(left_image, (anchor_x, anchor_y), 20, (255, 255, 255), 3, cv2.LINE_AA)
    _frame_banner(left_image, "parent state", _PARENT)

    if candidate.status == "instantiated":
        color = _candidate_color(0)
        delta_x, delta_y = _arrow_delta(change)
        end_x = int(np.clip(anchor_x + delta_x, 24, _FRAME_WIDTH - 24))
        end_y = int(np.clip(anchor_y + delta_y, 80, _FRAME_HEIGHT - 24))
        cv2.circle(right_image, (anchor_x, anchor_y), 11, _PARENT, -1, cv2.LINE_AA)
        cv2.arrowedLine(
            right_image,
            (anchor_x, anchor_y),
            (end_x, end_y),
            color,
            9,
            cv2.LINE_AA,
            tipLength=0.20,
        )
        cv2.circle(right_image, (end_x, end_y), 14, color, -1, cv2.LINE_AA)
        label = "child candidate | not selected"
        if candidate.discrete_changes and not candidate.numerical_changes:
            _put(
                right_image,
                candidate.discrete_changes[0].after_value.replace("_", " "),
                (max(24, anchor_x - 150), min(_FRAME_HEIGHT - 26, anchor_y + 74)),
                scale=0.68,
                color=color,
            )
    else:
        color = _ERROR if candidate.status == "unsupported" else _WARNING
        cv2.rectangle(
            right_image,
            (anchor_x - 128, anchor_y - 38),
            (anchor_x + 128, anchor_y + 38),
            color,
            5,
        )
        _put(
            right_image,
            candidate.status.replace("_", " "),
            (anchor_x - 108, anchor_y + 10),
            scale=0.72,
            color=color,
        )
        label = f"{candidate.status.replace('_', ' ')} | no child"
    _frame_banner(right_image, label, color)
    _put(
        right_image,
        "state-space arrow; source frame is unchanged",
        (20, _FRAME_HEIGHT - 18),
        scale=0.56,
        color=(238, 238, 238),
        thickness=1,
    )
    canvas[
        _FRAME_TOP : _FRAME_TOP + _FRAME_HEIGHT,
        30 : 30 + _FRAME_WIDTH,
    ] = left_image
    right_x = 30 + _FRAME_WIDTH + _FRAME_GAP
    canvas[
        _FRAME_TOP : _FRAME_TOP + _FRAME_HEIGHT,
        right_x : right_x + _FRAME_WIDTH,
    ] = right_image


def _maximum_sigma(proposal) -> float | None:
    for bound in proposal.parameter_bounds:
        if bound.parameter_name in {
            "maximum_pose_delta_sigma",
            "maximum_state_delta_sigma",
        }:
            return float(bound.upper_bound)
    return None


def _candidate_sigma_series(candidate) -> dict[int, float]:
    values: dict[int, float] = {}
    for change in candidate.numerical_changes:
        if change.maximum_standardized_delta is None:
            continue
        values[change.frame_index] = max(
            values.get(change.frame_index, 0.0),
            float(change.maximum_standardized_delta),
        )
    return values


def _draw_delta_chart(canvas: np.ndarray, *, proposal, result) -> None:
    x0, x1 = 72, 1160
    y0, y1 = _CHART_TOP + 58, _CHART_BOTTOM - 42
    _put(canvas, "bounded state change", (52, _CHART_TOP + 28), scale=0.82)
    candidates = _instantiated(result)
    series = [_candidate_sigma_series(row) for row in candidates]
    bound = _maximum_sigma(proposal)
    ymax = max(
        1.0,
        float(bound or 0.0),
        max((max(row.values(), default=0.0) for row in series), default=0.0),
    )
    cv2.rectangle(canvas, (x0, y0), (x1, y1), _DARK, 2)
    for fraction in (0.0, 0.5, 1.0):
        y = int(round(y1 - fraction * (y1 - y0)))
        cv2.line(canvas, (x0, y), (x1, y), _GRID, 1)
        _put(
            canvas,
            f"{fraction * ymax:.2g}",
            (x0 - 58, y + 6),
            scale=0.53,
            color=_MUTED,
            thickness=1,
        )
    start, end = proposal.start_frame_index, proposal.end_frame_index

    def to_x(frame_index: int) -> int:
        return int(round(x0 + (x1 - x0) * (frame_index - start) / max(1, end - start)))

    def to_y(value: float) -> int:
        return int(round(y1 - (y1 - y0) * value / ymax))

    if bound is not None:
        y = to_y(bound)
        cv2.line(canvas, (x0, y), (x1, y), _ERROR, 3, cv2.LINE_AA)
        _put(canvas, f"bound {bound:g} sigma", (x1 - 190, y - 10), scale=0.54, color=_ERROR)
    plotted = False
    for index, (candidate, values) in enumerate(zip(candidates[:6], series[:6])):
        if not values:
            continue
        plotted = True
        color = _candidate_color(index)
        points = [(to_x(frame), to_y(value)) for frame, value in sorted(values.items())]
        if len(points) == 1:
            cv2.circle(canvas, points[0], 9, color, -1, cv2.LINE_AA)
        else:
            cv2.polylines(
                canvas, [np.asarray(points, dtype=np.int32)], False, color, 5, cv2.LINE_AA
            )
            for point in points:
                cv2.circle(canvas, point, 7, color, -1, cv2.LINE_AA)
        _put(
            canvas,
            f"candidate {index + 1:02d}",
            (x0 + 16 + index * 168, y0 + 28),
            scale=0.52,
            color=color,
            thickness=2,
        )
    if not plotted:
        if candidates and any(row.discrete_changes for row in candidates):
            change = next(row.discrete_changes[0] for row in candidates if row.discrete_changes)
            _label_value(
                canvas,
                label="before",
                value=change.before_value.replace("_", " "),
                origin=(x0 + 70, y0 + 104),
                color=_PARENT,
                scale=0.77,
            )
            cv2.arrowedLine(
                canvas,
                (x0 + 300, y0 + 142),
                (x0 + 780, y0 + 142),
                _candidate_color(0),
                9,
                cv2.LINE_AA,
                tipLength=0.06,
            )
            _label_value(
                canvas,
                label="after",
                value=change.after_value.replace("_", " "),
                origin=(x0 + 800, y0 + 150),
                color=_candidate_color(0),
                scale=0.77,
            )
        else:
            _put(
                canvas,
                "no numerical child state was emitted",
                (x0 + 250, y0 + 145),
                scale=0.79,
                color=_MUTED,
            )
    _put(canvas, str(start), (x0 - 8, y1 + 31), scale=0.56, color=_MUTED, thickness=1)
    _put(canvas, str(end), (x1 - 24, y1 + 31), scale=0.56, color=_MUTED, thickness=1)
    _put(canvas, "frame", (x0 + (x1 - x0) // 2 - 28, y1 + 34), scale=0.56, color=_MUTED)
    _put(canvas, "position update (sigma)", (x0, y0 - 12), scale=0.55, color=_MUTED)


def _draw_objective_chart(canvas: np.ndarray, *, result) -> None:
    x0, x1 = 1240, 1870
    _put(canvas, "objective total | lower is better", (x0, _CHART_TOP + 28), scale=0.82)
    candidates = _instantiated(result)[:6]
    if not candidates:
        audit = result.candidates[0]
        _label_value(
            canvas,
            label="result",
            value=audit.status.replace("_", " "),
            origin=(x0 + 24, _CHART_TOP + 112),
            color=_ERROR if audit.status == "unsupported" else _WARNING,
            scale=0.82,
        )
        lines = _wrapped("; ".join(audit.limitations), 52)[:3]
        for index, line in enumerate(lines):
            _put(
                canvas,
                line,
                (x0 + 24, _CHART_TOP + 166 + index * 38),
                scale=0.63,
                color=_MUTED,
                thickness=1,
            )
        return
    values = [float(row.objective_before.total) for row in candidates]
    values += [float(row.objective_after.total) for row in candidates]
    maximum = max(max(values, default=0.0), 1e-12)
    chart_left = x0 + 164
    chart_right = x1 - 12
    row_height = min(68, max(42, 250 // len(candidates)))
    for index, candidate in enumerate(candidates):
        y = _CHART_TOP + 78 + index * row_height
        before = float(candidate.objective_before.total)
        after = float(candidate.objective_after.total)
        _put(canvas, f"candidate {index + 1:02d}", (x0, y + 13), scale=0.57, color=_DARK)
        before_x = chart_left + int((chart_right - chart_left) * before / maximum)
        after_x = chart_left + int((chart_right - chart_left) * after / maximum)
        cv2.line(canvas, (chart_left, y - 8), (before_x, y - 8), _PARENT, 10, cv2.LINE_AA)
        cv2.line(
            canvas,
            (chart_left, y + 13),
            (after_x, y + 13),
            _candidate_color(index),
            10,
            cv2.LINE_AA,
        )
        _put(canvas, f"{before:.3g}", (min(before_x + 8, chart_right - 52), y - 2), scale=0.49, color=_PARENT)
        _put(canvas, f"{after:.3g}", (min(after_x + 8, chart_right - 52), y + 24), scale=0.49, color=_candidate_color(index))
    _put(canvas, "parent", (chart_left, _CHART_BOTTOM - 16), scale=0.55, color=_PARENT)
    _put(canvas, "child", (chart_left + 112, _CHART_BOTTOM - 16), scale=0.55, color=_candidate_color(0))
    _put(canvas, "Step 9 decides acceptance", (x1 - 244, _CHART_BOTTOM - 16), scale=0.55, color=_MUTED)


def _change_accounting(result) -> tuple[int, int, int]:
    candidates = _instantiated(result)
    numeric = sum(len(row.numerical_changes) for row in candidates)
    discrete = sum(len(row.discrete_changes) for row in candidates)
    frames = {
        change.frame_index
        for row in candidates
        for change in row.numerical_changes
    }
    return numeric, discrete, len(frames)


def _draw_text_region(canvas: np.ndarray, *, proposal, result, store) -> None:
    left_x, right_x = 52, 1010
    _put(canvas, "Quantitative change", (left_x, _TEXT_TOP), scale=0.86)
    numeric, discrete, frames = _change_accounting(result)
    candidates = _instantiated(result)
    rows = (
        ("child candidates", str(len(candidates)), _candidate_color(0)),
        ("changed states", f"{numeric} numeric | {discrete} discrete", _candidate_color(0)),
        ("changed frames", str(frames), _candidate_color(0)),
        (
            "boundary",
            "preserved" if all(row.boundary_preserved for row in result.candidates) else "failed",
            _candidate_color(0),
        ),
    )
    for index, (label, value, color) in enumerate(rows):
        _label_value(
            canvas,
            label=label,
            value=value,
            origin=(left_x, _TEXT_TOP + 48 + index * 43),
            color=color,
            scale=0.72,
        )
    _put(canvas, "Optimization safety", (right_x, _TEXT_TOP), scale=0.86)
    all_candidates = result.candidates
    optimized = len({value for row in all_candidates for value in row.optimized_residual_ids})
    excluded = len({value for row in all_candidates for value in row.excluded_check_residual_ids})
    safety = (
        ("objective", f"fit {store.config.fit_objective_weight:g} + physics {store.config.physics_objective_weight:g}", _candidate_color(0)),
        ("optimized residuals", str(optimized), _candidate_color(0)),
        ("check-only excluded", str(excluded), _WARNING),
        ("selection", "none | parent retained", _PARENT),
    )
    for index, (label, value, color) in enumerate(safety):
        _label_value(
            canvas,
            label=label,
            value=value,
            origin=(right_x, _TEXT_TOP + 48 + index * 43),
            color=color,
            scale=0.72,
        )
    cv2.line(canvas, (960, _TEXT_TOP - 18), (960, _HEIGHT - 28), _GRID, 2)


def _proposal_panel(
    *,
    proposal_number: int,
    proposal,
    result,
    residual_packet,
    tracking,
    frame,
    parent_rank: int,
    store,
) -> np.ndarray:
    canvas = np.full((_HEIGHT, _WIDTH, 3), _WHITE, dtype=np.uint8)
    status_color = (
        _candidate_color(0)
        if result.status == "candidates_generated"
        else _ERROR if result.status == "unsupported" else _WARNING
    )
    cv2.rectangle(canvas, (0, 0), (18, _HEADER_BOTTOM), status_color, -1)
    title = f"proposal {proposal_number:02d} | {proposal.operator.value.replace('_', ' ')}"
    _put(canvas, title, (42, 51), scale=1.02, color=_DARK, thickness=2)
    _put(
        canvas,
        f"parent rank {parent_rank:02d} | frames {proposal.start_frame_index}-{proposal.end_frame_index} | {result.status.replace('_', ' ')}",
        (43, 86),
        scale=0.67,
        color=_MUTED,
        thickness=1,
    )
    _label_value(
        canvas,
        label="result",
        value=(
            f"{len(_instantiated(result))} child candidates | none selected"
            if _instantiated(result)
            else "no child emitted"
        ),
        origin=(1350, 64),
        color=status_color,
        scale=0.66,
    )
    representative = _representative_candidate(result)
    _, track_ids = _subjects(proposal, residual_packet)
    _, change = _display_change(representative, proposal)
    _draw_frame_pair(
        canvas,
        frame=frame,
        tracking=tracking,
        track_ids=track_ids,
        candidate=representative,
        change=change,
    )
    _draw_delta_chart(canvas, proposal=proposal, result=result)
    _draw_objective_chart(canvas, result=result)
    _draw_text_region(canvas, proposal=proposal, result=result, store=store)
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
                    store=store,
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
