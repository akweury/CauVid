"""Diagnostic visualizations for Step 7 diagnoses and repair proposals."""

from __future__ import annotations

import json
import re
import textwrap
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.exp_august.contracts import (
    ArtifactOwner,
    RepairProposalStore,
    VideoManifest,
    VideoRepairProposalManifest,
    VideoResidualManifest,
    VideoTrackingManifest,
)
from src.exp_august.contracts.codec import read_contract, sha256_file
from src.exp_august.inference.artifact_io import read_image_artifact
from src.exp_august.inference.frames import CanonicalFrameProvider


_OPERATOR_COLORS = {
    "relink_track": "#e67e22",
    "split_track": "#d35400",
    "switch_mask_candidate": "#16a085",
    "switch_pose_candidate": "#8e44ad",
    "switch_scale_candidate": "#6c5ce7",
    "invalidate_or_downweight_cue": "#7f8c8d",
    "refit_local_dynamics": "#2980b9",
    "adjust_process_noise": "#27ae60",
    "mark_occluded": "#95a5a6",
    "mark_unobservable": "#34495e",
    "leave_unresolved": "#c0392b",
}

_OPERATOR_BGR = {
    key: tuple(int(color[index : index + 2], 16) for index in (5, 3, 1))
    for key, color in _OPERATOR_COLORS.items()
}

_PANEL_WIDTH = 1920
_PANEL_HEIGHT = 1080
_HEADER_HEIGHT = 120
_IMAGE_LEFT = 42
_IMAGE_TOP = 150
_IMAGE_WIDTH = 1120
_IMAGE_HEIGHT = 700
_SIDE_LEFT = 1205
_SIDE_RIGHT = 1878
_TIMELINE_Y = 994


def _run_root(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "07_diagnose_propose":
            return parent.parent
    raise RuntimeError("Step 7 store must live below 07_diagnose_propose")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "proposal"


def _verified_contract(path: Path, reference, model, *, label: str):
    if not path.is_file() or path.stat().st_size != reference.byte_size:
        raise RuntimeError(f"Step 7 visualization {label} is missing or truncated: {path}")
    if sha256_file(path) != reference.sha256:
        raise RuntimeError(f"Step 7 visualization {label} failed integrity check: {path}")
    return read_contract(path, model)


def _load_context(*, run_root: Path, manifest: VideoRepairProposalManifest):
    snapshot = manifest.input_snapshot
    step6_root = run_root / snapshot.source_step6_relative_root
    residual_path = step6_root / snapshot.video_residual_manifest.artifact.relative_path
    residual = _verified_contract(
        residual_path,
        snapshot.video_residual_manifest.artifact,
        VideoResidualManifest,
        label="residual manifest",
    )
    step3_root = run_root / snapshot.source_step3_relative_root
    tracking_path = step3_root / snapshot.video_tracking_manifest.artifact.relative_path
    tracking = _verified_contract(
        tracking_path,
        snapshot.video_tracking_manifest.artifact,
        VideoTrackingManifest,
        label="tracking manifest",
    )
    source_link = snapshot.source_video_manifest
    source_path = run_root / "01_init" / source_link.artifact.relative_path
    source = _verified_contract(
        source_path,
        source_link.artifact,
        VideoManifest,
        label="source video manifest",
    )
    provider = CanonicalFrameProvider(source, verify_source_hash=True)
    step2_root = run_root / residual.input_snapshot.source_step2_relative_root
    return residual, tracking, provider, step2_root, step3_root


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write Step 7 visualization: {path}")


def _put_lines(
    image: np.ndarray,
    lines,
    *,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (42, 47, 56),
    line_height: int = 28,
    thickness: int = 1,
) -> int:
    x, y = origin
    for line in lines:
        cv2.putText(
            image,
            str(line),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_height
    return y


def _wrapped(value: str, width: int = 58) -> list[str]:
    return textwrap.wrap(str(value), width=width, break_long_words=False) or [""]


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


def _selected_mask(
    *,
    tracking,
    observation,
    step2_root: Path,
    step3_root: Path,
):
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
    root = (
        step2_root
        if candidate.mask.owner == ArtifactOwner.STEP2_NEURAL_EVIDENCE
        else step3_root
    )
    path = root / candidate.mask.artifact.relative_path
    if not path.is_file() or sha256_file(path) != candidate.mask.artifact.sha256:
        raise RuntimeError(f"Step 7 visualization mask integrity failure: {path}")
    image = read_image_artifact(path, cv2.IMREAD_GRAYSCALE)
    return None if image is None else image > 0


def _draw_track_support(
    image: np.ndarray,
    *,
    tracking,
    residual,
    frame_index: int,
    color,
    step2_root: Path,
    step3_root: Path,
):
    track_id = residual.track_id if residual is not None else None
    track, observation = _track_observation(tracking, track_id, frame_index)
    if observation is None:
        return None, track
    mask = _selected_mask(
        tracking=tracking,
        observation=observation,
        step2_root=step2_root,
        step3_root=step3_root,
    )
    if mask is not None and mask.shape == image.shape[:2]:
        image[mask] = np.clip(
            image[mask].astype(np.float32) * 0.72
            + np.asarray(color, dtype=np.float32) * 0.28,
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
    label = f"{track.primary_class} | {track.track_id}"
    cv2.putText(
        image,
        label,
        (max(4, x1), max(22, y1 - 9)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    return (0.5 * (box.x1 + box.x2), 0.5 * (box.y1 + box.y2)), track


def _draw_residual_geometry(image: np.ndarray, residual, origin, color) -> None:
    if residual is None:
        return
    predicted = tuple(float(value) for value in residual.predicted_values)
    observed = tuple(float(value) for value in residual.observed_values)
    if (
        residual.metric_name == "pixel_reprojection_error"
        and len(predicted) >= 2
        and len(observed) >= 2
    ):
        pred = tuple(int(round(value)) for value in predicted[:2])
        obs = tuple(int(round(value)) for value in observed[:2])
        cv2.line(image, pred, obs, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.drawMarker(image, pred, color, cv2.MARKER_CROSS, 24, 3)
        cv2.circle(image, obs, 10, (210, 125, 45), 3, cv2.LINE_AA)
    elif "flow" in residual.metric_name and len(predicted) >= 2 and len(observed) >= 2:
        origin = origin or (image.shape[1] / 2.0, image.shape[0] / 2.0)
        start = tuple(int(round(value)) for value in origin)
        vectors = (np.asarray(predicted[:2]), np.asarray(observed[:2]))
        maximum_norm = max(float(np.linalg.norm(vector)) for vector in vectors)
        display_scale = min(1.0, 180.0 / max(maximum_norm, 1e-6))
        for vector, arrow_color in zip(vectors, (color, (210, 125, 45))):
            end = tuple(
                int(round(start[index] + display_scale * vector[index]))
                for index in range(2)
            )
            cv2.arrowedLine(image, start, end, arrow_color, 4, cv2.LINE_AA, tipLength=0.2)


def _comparison(packets, path: Path) -> None:
    ranks = np.asarray([row.hypothesis_rank for row in packets])
    diagnoses = np.asarray([len(row.diagnoses) for row in packets])
    ready = np.asarray(
        [sum(item.status == "ready" for item in row.proposals) for row in packets]
    )
    unresolved = np.asarray(
        [sum(item.status == "leave_unresolved" for item in row.proposals) for row in packets]
    )
    deferred = np.asarray([len(row.deferred_conflict_ids) for row in packets])
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=150)
    width = 0.20
    axes[0].bar(ranks - 1.5 * width, diagnoses, width=width, label="diagnoses", color="#9ecae1")
    axes[0].bar(ranks - 0.5 * width, ready, width=width, label="ready proposals", color="#31a354")
    axes[0].bar(ranks + 0.5 * width, unresolved, width=width, label="unresolved", color="#de2d26")
    axes[0].bar(
        ranks + 1.5 * width,
        deferred,
        width=width,
        label="deferred conflicts",
        color="#969696",
    )
    axes[0].set_xlabel("Step 5 hypothesis rank")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(ranks)
    axes[0].set_title("Diagnosis and proposal accounting")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    if not np.any(diagnoses + ready + unresolved + deferred):
        axes[0].set_ylim(0.0, 1.0)
        axes[0].text(
            0.5,
            0.5,
            "No diagnoses or repair proposals",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
    operators = sorted(
        {proposal.operator.value for packet in packets for proposal in packet.proposals}
    )
    bottom = np.zeros(len(ranks), dtype=float)
    for operator in operators:
        values = np.asarray(
            [sum(row.operator.value == operator for row in packet.proposals) for packet in packets]
        )
        axes[1].bar(
            ranks,
            values,
            bottom=bottom,
            color=_OPERATOR_COLORS[operator],
            label=operator.replace("_", " "),
        )
        bottom += values
    axes[1].set_xlabel("Step 5 hypothesis rank")
    axes[1].set_ylabel("Proposal count")
    axes[1].set_xticks(ranks)
    axes[1].set_title("Allow-listed operator mix - not a ranking score")
    axes[1].grid(axis="y", alpha=0.2)
    if operators:
        axes[1].legend(fontsize=7, loc="upper right")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No repairs proposed",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    figure.suptitle("Step 7 diagnosis comparison - world hypotheses remain immutable")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path)
    plt.close(figure)


def _timeline(packet, frame_count: int, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12.8, 5.4), dpi=150)
    proposals = list(packet.proposals)
    if proposals:
        for index, proposal in enumerate(proposals):
            color = _OPERATOR_COLORS[proposal.operator.value]
            axis.barh(
                index,
                proposal.end_frame_index - proposal.start_frame_index + 1,
                left=proposal.start_frame_index,
                height=0.58,
                color=color,
                alpha=0.78,
            )
            diagnosis = next(
                row for row in packet.diagnoses if row.diagnosis_id == proposal.diagnosis_id
            )
            axis.text(
                proposal.start_frame_index,
                index + 0.34,
                diagnosis.category.value.replace("_", " "),
                fontsize=7,
                color="#263238",
            )
        axis.set_yticks(
            range(len(proposals)),
            [row.operator.value.replace("_", "\n") for row in proposals],
            fontsize=7,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No conflict and no repair proposal",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_yticks([])
    for keyframe in packet.evidence.keyframes:
        axis.axvline(keyframe.frame_index, color="#455a64", linewidth=0.8, alpha=0.32)
    axis.set_xlim(-0.5, max(0.5, frame_count - 0.5))
    axis.set_xlabel("Canonical frame index")
    axis.set_title(
        f"Hypothesis rank {packet.hypothesis_rank}: diagnosis/proposal timeline | "
        f"status={packet.status} | proposals={len(packet.proposals)}"
    )
    axis.grid(axis="x", alpha=0.18)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _proposal_audit(packet) -> dict:
    return {
        "schema_name": "step7_proposal_audit",
        "schema_version": 1,
        "hypothesis_id": packet.hypothesis_id,
        "hypothesis_rank": packet.hypothesis_rank,
        "world_state_mutated": False,
        "selection_applied": False,
        "evidence": packet.evidence.model_dump(mode="json"),
        "diagnoses": [row.model_dump(mode="json") for row in packet.diagnoses],
        "proposals": [row.model_dump(mode="json") for row in packet.proposals],
        "deferred_conflict_ids": list(packet.deferred_conflict_ids),
    }


def _parameter_lines(proposal) -> list[str]:
    lines = []
    for bound in proposal.parameter_bounds:
        if bound.allowed_values:
            value = ", ".join(bound.allowed_values)
        else:
            value = f"[{bound.lower_bound:g}, {bound.upper_bound:g}] {bound.unit}"
        lines.extend(_wrapped(f"{bound.parameter_name}: {value}", 55))
    return lines


def _proposal_panel(
    *,
    frame,
    packet,
    diagnosis,
    proposal,
    residual,
    tracking,
    frame_count: int,
    step2_root: Path,
    step3_root: Path,
) -> np.ndarray:
    color = _OPERATOR_BGR[proposal.operator.value]
    canvas = np.full((_PANEL_HEIGHT, _PANEL_WIDTH, 3), 248, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (_PANEL_WIDTH, _HEADER_HEIGHT), color, -1)
    cv2.putText(
        canvas,
        f"STEP 7 REPAIR PROPOSAL | HYPOTHESIS RANK {packet.hypothesis_rank:02d}",
        (42, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.10,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{diagnosis.category.value.upper().replace('_', ' ')}  ->  "
        f"{proposal.operator.value.upper().replace('_', ' ')}",
        (42, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        (245, 248, 252),
        2,
        cv2.LINE_AA,
    )
    annotated = frame.image_bgr.copy()
    origin, _ = _draw_track_support(
        annotated,
        tracking=tracking,
        residual=residual,
        frame_index=frame.frame_index,
        color=color,
        step2_root=step2_root,
        step3_root=step3_root,
    )
    _draw_residual_geometry(annotated, residual, origin, color)
    image_region = canvas[
        _IMAGE_TOP : _IMAGE_TOP + _IMAGE_HEIGHT,
        _IMAGE_LEFT : _IMAGE_LEFT + _IMAGE_WIDTH,
    ]
    source_height, source_width = annotated.shape[:2]
    scale = min(_IMAGE_WIDTH / source_width, _IMAGE_HEIGHT / source_height)
    rendered = cv2.resize(
        annotated,
        (max(1, int(round(source_width * scale))), max(1, int(round(source_height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    top = (_IMAGE_HEIGHT - rendered.shape[0]) // 2
    left = (_IMAGE_WIDTH - rendered.shape[1]) // 2
    image_region[:] = 25
    image_region[top : top + rendered.shape[0], left : left + rendered.shape[1]] = rendered
    cv2.rectangle(
        canvas,
        (_IMAGE_LEFT, _IMAGE_TOP),
        (_IMAGE_LEFT + _IMAGE_WIDTH, _IMAGE_TOP + _IMAGE_HEIGHT),
        (75, 82, 92),
        2,
    )
    cv2.putText(
        canvas,
        f"canonical frame {frame.frame_index:06d} | time {frame.timestamp_s:.2f} s",
        (_IMAGE_LEFT, _IMAGE_TOP + _IMAGE_HEIGHT + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (70, 76, 86),
        1,
        cv2.LINE_AA,
    )

    y = _put_lines(
        canvas,
        [
            f"Diagnosis confidence: {diagnosis.confidence:.0%}",
            f"Proposal status: {proposal.status}",
            f"Window: frames {proposal.start_frame_index}-{proposal.end_frame_index}",
        ],
        origin=(_SIDE_LEFT, 176),
        scale=0.62,
        line_height=32,
        thickness=2,
    )
    y += 12
    cv2.putText(
        canvas,
        "WHY",
        (_SIDE_LEFT, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        color,
        2,
        cv2.LINE_AA,
    )
    y += 30
    y = _put_lines(canvas, _wrapped(diagnosis.rationale), origin=(_SIDE_LEFT, y), line_height=25)
    y += 12
    cv2.putText(
        canvas,
        "AFFECTED VARIABLES",
        (_SIDE_LEFT, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        color,
        2,
        cv2.LINE_AA,
    )
    y += 30
    y = _put_lines(
        canvas,
        [f"- {value}" for value in proposal.affected_variables],
        origin=(_SIDE_LEFT, y),
        line_height=25,
    )
    y += 12
    cv2.putText(
        canvas,
        "BOUNDS",
        (_SIDE_LEFT, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        color,
        2,
        cv2.LINE_AA,
    )
    y += 30
    y = _put_lines(canvas, _parameter_lines(proposal), origin=(_SIDE_LEFT, y), line_height=24)
    y += 10
    cv2.putText(
        canvas,
        "SAFETY",
        (_SIDE_LEFT, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        color,
        2,
        cv2.LINE_AA,
    )
    y += 30
    safety = [
        "Parent hypothesis: immutable",
        "Raw evidence mutation: forbidden",
        "Diagnoser numeric state values: forbidden",
    ]
    if residual is not None and residual.evaluation_basis.value == "check_evidence":
        safety.append("CHECK-ONLY: diagnosis/acceptance; never optimized")
    _put_lines(canvas, safety, origin=(_SIDE_LEFT, y), line_height=25)

    x0, x1 = 80, _PANEL_WIDTH - 80
    cv2.line(canvas, (x0, _TIMELINE_Y), (x1, _TIMELINE_Y), (95, 103, 114), 2)
    denominator = max(1, frame_count - 1)
    to_x = lambda value: int(round(x0 + (x1 - x0) * value / denominator))
    start_x = to_x(proposal.start_frame_index)
    end_x = to_x(proposal.end_frame_index)
    cv2.rectangle(
        canvas,
        (start_x, _TIMELINE_Y - 16),
        (max(start_x + 3, end_x), _TIMELINE_Y + 16),
        color,
        -1,
    )
    for keyframe in packet.evidence.keyframes:
        key_x = to_x(keyframe.frame_index)
        cv2.line(canvas, (key_x, _TIMELINE_Y - 25), (key_x, _TIMELINE_Y + 25), (70, 78, 88), 2)
    current_x = to_x(frame.frame_index)
    cv2.circle(canvas, (current_x, _TIMELINE_Y), 9, (245, 245, 245), -1)
    cv2.circle(canvas, (current_x, _TIMELINE_Y), 9, color, 2)
    cv2.putText(
        canvas,
        "0",
        (x0 - 8, _TIMELINE_Y + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (80, 87, 97),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        str(frame_count - 1),
        (x1 - 28, _TIMELINE_Y + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (80, 87, 97),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "proposal window / evidence keyframes",
        (x0, _TIMELINE_Y - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (70, 77, 87),
        1,
        cv2.LINE_AA,
    )
    return canvas


def render_step7_visualizations(
    *,
    repair_proposal_store_path: Path | str,
    maximum_hypotheses: int = 5,
    maximum_proposal_panels: int = 8,
) -> Path:
    """Render proposal accounting, timelines and evidence-grounded panels."""

    if maximum_hypotheses <= 0:
        raise ValueError("maximum_hypotheses must be positive")
    if maximum_proposal_panels < 0:
        raise ValueError("maximum_proposal_panels cannot be negative")
    store_path = Path(repair_proposal_store_path).expanduser().resolve()
    store = read_contract(store_path, RepairProposalStore)
    stage_root = store_path.parent
    run_root = _run_root(store_path)
    output_root = stage_root / "visualizations"
    output_root.mkdir(parents=True, exist_ok=True)
    videos = []
    for video_id, reference in zip(store.video_ids, store.video_repair_proposals):
        source_path = stage_root / reference.relative_path
        manifest = _verified_contract(
            source_path,
            reference,
            VideoRepairProposalManifest,
            label="repair proposal manifest",
        )
        residual_manifest, tracking, provider, step2_root, step3_root = _load_context(
            run_root=run_root, manifest=manifest
        )
        residual_packets = {
            row.hypothesis_id: row for row in residual_manifest.packets
        }
        video_root = output_root / video_id
        video_root.mkdir(parents=True, exist_ok=True)
        selected_packets = manifest.packets[:maximum_hypotheses]
        comparison = video_root / "diagnosis_operator_summary.png"
        _comparison(selected_packets, comparison)
        packet_rows = []
        frame_cache: dict[int, object] = {}
        for packet in selected_packets:
            timeline = video_root / f"rank_{packet.hypothesis_rank:02d}_diagnosis_timeline.png"
            _timeline(packet, manifest.frame_count, timeline)
            audit_path = video_root / f"rank_{packet.hypothesis_rank:02d}_proposals.json"
            audit_path.write_text(
                json.dumps(
                    _proposal_audit(packet),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            proposal_root = video_root / f"rank_{packet.hypothesis_rank:02d}_proposals"
            proposal_root.mkdir(parents=True, exist_ok=True)
            for stale_panel in proposal_root.glob("*.png"):
                stale_panel.unlink()
            overview_path = (
                video_root
                / f"rank_{packet.hypothesis_rank:02d}_proposal_overview.png"
            )
            if overview_path.is_file():
                overview_path.unlink()
            residual_packet = residual_packets[packet.hypothesis_id]
            residual_by_id = {row.residual_id: row for row in residual_packet.residuals}
            diagnosis_by_id = {row.diagnosis_id: row for row in packet.diagnoses}
            prioritized = sorted(
                packet.proposals,
                key=lambda row: (
                    row.status != "ready",
                    not any(
                        effect.evaluation_basis.value == "check_evidence"
                        for effect in row.expected_residual_effects
                    ),
                    row.start_frame_index,
                    row.proposal_id,
                ),
            )[:maximum_proposal_panels]
            panel_paths = []
            panel_images = []
            for index, proposal in enumerate(prioritized, start=1):
                target_residuals = [
                    residual_by_id[value]
                    for value in proposal.target_residual_ids
                    if value in residual_by_id
                ]
                residual = (
                    max(
                        target_residuals,
                        key=lambda row: float(row.normalized_residual or 0.0),
                    )
                    if target_residuals
                    else None
                )
                frame_index = (
                    residual.start_frame_index
                    if residual is not None
                    else proposal.start_frame_index
                )
                if frame_index not in frame_cache:
                    frame_cache[frame_index] = provider.get_frame(frame_index)
                panel = _proposal_panel(
                    frame=frame_cache[frame_index],
                    packet=packet,
                    diagnosis=diagnosis_by_id[proposal.diagnosis_id],
                    proposal=proposal,
                    residual=residual,
                    tracking=tracking,
                    frame_count=manifest.frame_count,
                    step2_root=step2_root,
                    step3_root=step3_root,
                )
                panel_path = proposal_root / (
                    f"proposal_{index:02d}_frame_{frame_index:06d}_"
                    f"{_safe_name(proposal.operator.value)}.png"
                )
                _write_image(panel_path, panel)
                panel_paths.append(panel_path)
                panel_images.append(panel)
            overview = None
            if panel_images:
                tiles = panel_images[:4]
                while len(tiles) < 4:
                    tiles.append(np.full((_PANEL_HEIGHT, _PANEL_WIDTH, 3), 247, dtype=np.uint8))
                overview_image = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:4])))
                overview = overview_path
                _write_image(overview, overview_image)
            packet_rows.append(
                {
                    "hypothesis_id": packet.hypothesis_id,
                    "hypothesis_rank": packet.hypothesis_rank,
                    "status": packet.status,
                    "diagnosis_count": len(packet.diagnoses),
                    "proposal_count": len(packet.proposals),
                    "rendered_proposal_panel_count": len(panel_paths),
                    "deferred_conflict_count": len(packet.deferred_conflict_ids),
                    "timeline": timeline.relative_to(output_root).as_posix(),
                    "proposal_audit": audit_path.relative_to(output_root).as_posix(),
                    "proposal_panels": [
                        path.relative_to(output_root).as_posix() for path in panel_paths
                    ],
                    "proposal_overview": (
                        overview.relative_to(output_root).as_posix()
                        if overview is not None
                        else None
                    ),
                    "proposal_panel_resolution": [_PANEL_WIDTH, _PANEL_HEIGHT],
                    "proposal_overview_resolution": [
                        2 * _PANEL_WIDTH,
                        2 * _PANEL_HEIGHT,
                    ],
                }
            )
        videos.append(
            {
                "video_id": video_id,
                "diagnosis_operator_summary": comparison.relative_to(output_root).as_posix(),
                "packets": packet_rows,
            }
        )
    manifest_path = output_root / "step7_visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_name": "step7_visualization_manifest",
                "schema_version": 1,
                "source_repair_proposal_store_sha256": sha256_file(store_path),
                "world_state_mutated": False,
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
    return manifest_path


__all__ = ["render_step7_visualizations"]
