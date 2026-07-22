"""Image diagnostics for the Step 8C trajectory-pattern closed loop."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


PATTERNS = (
    "stationary", "same_direction", "opposite_direction", "approaching",
    "receding", "crossing", "turning", "lane_entry", "overtaking", "unknown",
)


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(cv2, image, text, x, y, scale=0.46, color=(225, 225, 225), thickness=1):
    cv2.putText(
        image, str(text), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )


def _fit_text(cv2, text, width, scale=0.40, thickness=1):
    words = str(text).split()
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )[0][0] > width:
            return current + " ..."
        current = candidate
    return current


def _render_track(record, output_path):
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError:
        return False, "missing_cv2_or_numpy"

    image = np.full((1500, 1800, 3), (22, 24, 28), dtype=np.uint8)
    white = (242, 242, 242)
    muted = (170, 175, 185)
    green = (70, 220, 100)
    red = (75, 90, 235)
    amber = (60, 190, 245)
    cyan = (225, 190, 70)

    video_id = str(record.get("video_id", ""))
    track_id = int(record.get("track_id", -1))
    _text(
        cv2, image,
        f"STEP 8C | TRAJECTORY PATTERN CLOSED LOOP | video {video_id} | track {track_id}",
        28, 48, 0.78, white, 2,
    )

    flow = (
        "symbolic abstraction", "all-pattern residuals", "LLM interpretation",
        "multi-repair", "symbolic validation", "final selection",
    )
    x = 28
    for index, label in enumerate(flow):
        box_width = 260 if index < 5 else 250
        cv2.rectangle(image, (x, 72), (x + box_width, 122), (55, 60, 70), -1)
        _text(cv2, image, label, x + 12, 104, 0.45, white)
        if index < len(flow) - 1:
            cv2.arrowedLine(
                image, (x + box_width + 5, 97), (x + box_width + 33, 97),
                amber, 2, tipLength=0.25,
            )
        x += box_width + 38

    track = dict(record.get("symbolic_track", {}))
    source_validation = dict(track.get("source_validation", {})).get(
        "validation_status", track.get("source_decision", "unknown")
    )
    facts = (
        f"class={track.get('object_class', 'unknown')}  "
        f"direction={track.get('direction', 'unknown')}  "
        f"persistence={_number(track.get('persistence')):.3f}  "
        f"confidence={_number(track.get('confidence')):.3f}  "
        f"source_validation={source_validation}"
    )
    _text(cv2, image, "1. SYMBOLIC TRACK", 28, 162, 0.58, cyan, 2)
    _text(cv2, image, facts, 48, 198, 0.50, white)

    interpretations = {
        str(row.get("pattern_id", "")): row
        for row in record.get("llm_residual_interpretation", [])
    }
    candidates = list(record.get("pattern_candidates", []))
    _text(
        cv2, image,
        "2-3. EVERY PATTERN: NUMERICAL RESIDUAL VECTOR + LLM INTERPRETATION",
        28, 248, 0.58, cyan, 2,
    )
    for heading, hx in (
        ("pattern", 48), ("plausibility", 285), ("residual sum", 470),
        ("largest residual components", 650), ("LLM conflicts / explanation", 1120),
    ):
        _text(cv2, image, heading, hx, 282, 0.43, muted)

    row_y = 318
    for candidate in candidates:
        pattern_id = str(candidate.get("pattern_id", "unknown"))
        residuals = dict(candidate.get("residual_vector", {}))
        interpretation = interpretations.get(pattern_id, {})
        plausibility = _number(interpretation.get("plausibility"))
        residual_sum = sum(_number(value) for value in residuals.values())
        largest = sorted(
            residuals.items(), key=lambda item: _number(item[1]), reverse=True
        )[:3]
        residual_text = " | ".join(
            f"{name}={_number(value):.3g}" for name, value in largest
        )
        conflicts = ", ".join(
            map(str, interpretation.get("structural_conflicts", []))
        ) or str(interpretation.get("explanation", "no conflict reported"))
        final = pattern_id == str(record.get("final_pattern", ""))
        cv2.rectangle(
            image, (32, row_y - 23), (1760, row_y + 17),
            (42, 46, 54) if final else (29, 32, 38), -1,
        )
        _text(
            cv2, image, ("* " if final else "  ") + pattern_id,
            48, row_y, 0.47, green if final else white, 2 if final else 1,
        )
        _text(
            cv2, image, f"{plausibility:.3f}", 305, row_y, 0.47,
            green if plausibility >= 0.5 else amber,
        )
        _text(cv2, image, f"{residual_sum:.3g}", 490, row_y, 0.47, white)
        _text(cv2, image, residual_text, 650, row_y, 0.42, white)
        _text(cv2, image, _fit_text(cv2, conflicts, 610), 1120, row_y, 0.40, muted)
        row_y += 43

    repairs = sorted(
        record.get("candidate_repairs", []),
        key=lambda row: _number(row.get("final_score"), -1e9), reverse=True,
    )
    y = max(790, row_y + 30)
    _text(
        cv2, image,
        "4-5. DETERMINISTIC REPAIR CANDIDATES + SYMBOLIC VALIDATION",
        28, y, 0.58, cyan, 2,
    )
    y += 38
    for heading, hx in (
        ("candidate", 48), ("decision", 400), ("score", 565),
        ("residual improvement", 690), ("retention", 955),
        ("new anomalies / modified frames", 1120),
    ):
        _text(cv2, image, heading, hx, y, 0.43, muted)
    y += 34

    selected_id = str(record.get("selected_candidate", {}).get("candidate_id", ""))
    shown = repairs[:10]
    for repair in shown:
        decision = str(repair.get("symbolic_verdict", repair.get("decision", "unknown")))
        candidate_id = str(repair.get("candidate_id", ""))
        selected = candidate_id == selected_id
        decision_color = green if decision in {"accept", "pass"} else amber if decision == "uncertain" else red
        cv2.rectangle(
            image, (32, y - 23), (1760, y + 17),
            (42, 55, 45) if selected else (29, 32, 38), -1,
        )
        _text(
            cv2, image, ("SELECTED  " if selected else "") + candidate_id,
            48, y, 0.43, green if selected else white, 2 if selected else 1,
        )
        _text(cv2, image, decision.upper(), 400, y, 0.43, decision_color, 2)
        _text(cv2, image, f"{_number(repair.get('final_score')):.3f}", 565, y, 0.43, white)
        _text(cv2, image, f"{_number(repair.get('residual_improvement')):+.3f}", 740, y, 0.43, white)
        _text(cv2, image, f"{_number(repair.get('observation_retention')):.3f}", 970, y, 0.43, white)
        details = (
            (", ".join(map(str, repair.get("new_anomalies", []))) or "none")
            + f" | frames={repair.get('modified_frame_ids', [])}"
        )
        _text(cv2, image, _fit_text(cv2, details, 620, 0.38), 1120, y, 0.38, muted)
        y += 42
    if len(repairs) > len(shown):
        _text(
            cv2, image,
            f"+ {len(repairs) - len(shown)} additional candidates in the per-track JSON",
            48, y, 0.40, muted,
        )
        y += 34

    y = min(1300, max(y + 18, 1240))
    cv2.rectangle(image, (28, y), (1770, 1472), (38, 44, 54), -1)
    _text(cv2, image, "6. FINAL RESULT", 48, y + 40, 0.62, (230, 135, 75), 2)
    repair_applied = bool(record.get("repair_applied", False))
    llm_preferred = str(record.get("LLM_preferred_pattern", "unknown"))
    validated = str(record.get("validated_pattern", record.get("final_pattern", "unknown")))
    mismatch = llm_preferred != validated
    selected_score = min((sum(_number(v) for v in row.get("post_repair_pattern_scores", {}).get(row.get("pattern_id"), {}).values()) for row in repairs if row.get("candidate_id") == selected_id), default=float("inf"))
    lower_rejected = any(row.get("symbolic_verdict") == "reject" and sum(_number(v) for v in row.get("post_repair_pattern_scores", {}).get(row.get("pattern_id"), {}).values()) < selected_score for row in repairs)
    result = (
        f"pattern={record.get('final_pattern', 'unknown')} | "
        f"repair_applied={repair_applied} | "
        f"validation={record.get('final_validation_status', 'unknown')} | "
        f"selected={selected_id or 'none'}"
    )
    _text(cv2, image, result, 48, y + 82, 0.58, green if repair_applied else amber, 2)
    alert = f"LLM preferred={llm_preferred} | validated={validated}"
    if lower_rejected:
        alert += " | LOWER-RESIDUAL CANDIDATE REJECTED BY HARD CONSTRAINTS"
    _text(cv2, image, alert, 48, y + 120, 0.46, red if mismatch or lower_rejected else green, 2)
    _text(cv2, image, f"reason: {record.get('final_selection_reason', '')}", 48, y + 150, 0.43, white)
    provenance = dict(record.get("provenance", {}))
    provenance_text = (
        f"provenance: LLM={provenance.get('llm_role', '')} | "
        f"repair={provenance.get('numeric_repair_role', '')} | "
        f"original preserved={provenance.get('original_observations_preserved', False)}"
    )
    _text(cv2, image, provenance_text, 900, y + 150, 0.38, muted)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        return False, "image_write_failed"
    return True, "rendered"


def render_trajectory_pattern_visualizations(state, output_root):
    """Render per-track process sheets and per-video Step 8C result summaries."""
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError:
        return {
            **state,
            "trajectory_pattern_visualizations": [],
            "trajectory_pattern_video_summaries": [],
            "trajectory_pattern_visualization_skipped": [{"reason": "missing_cv2_or_numpy"}],
            "trajectory_pattern_visualization_output_root": None,
        }

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rendered = []
    skipped = []
    by_video = defaultdict(list)
    for record in state.get("trajectory_pattern_records", []):
        video_id = str(record.get("video_id", ""))
        track_id = int(record.get("track_id", -1))
        by_video[video_id].append(record)
        path = output_root / video_id / f"track_{track_id:04d}_pattern_process.png"
        success, status = _render_track(record, path)
        row = {"video_id": video_id, "track_id": track_id, "status": status}
        if success:
            row["visualization_path"] = str(path)
            rendered.append(row)
        else:
            skipped.append(row)

    summaries = []
    for video_id, records in sorted(by_video.items()):
        image = np.full((720, 1400, 3), (22, 24, 28), dtype=np.uint8)
        white = (242, 242, 242)
        green = (70, 220, 100)
        amber = (60, 190, 245)
        _text(cv2, image, f"STEP 8C VIDEO SUMMARY | {video_id}", 30, 50, 0.82, white, 2)
        repaired = sum(bool(row.get("repair_applied")) for row in records)
        disagreements = sum(str(row.get("LLM_preferred_pattern")) != str(row.get("validated_pattern")) for row in records)
        non_invalid = sum(
            str(row.get("final_validation_status", "")) != "invalid" for row in records
        )
        _text(
            cv2, image,
            f"tracks={len(records)}  repairs_applied={repaired}  final_non_invalid={non_invalid}  LLM_validated_disagreements={disagreements}",
            30, 95, 0.58, green,
        )
        counts = defaultdict(int)
        for record in records:
            counts[str(record.get("final_pattern", "unknown"))] += 1
        _text(cv2, image, "Final pattern distribution", 30, 150, 0.58, amber, 2)
        y = 190
        for pattern_id in PATTERNS:
            count = counts[pattern_id]
            bar_width = int(700 * count / max(1, len(records)))
            _text(cv2, image, pattern_id, 50, y, 0.46, white)
            cv2.rectangle(image, (270, y - 19), (270 + bar_width, y + 4), (70, 150, 220), -1)
            _text(cv2, image, count, 990, y, 0.46, white)
            y += 42
        promotion = dict(state.get("trajectory_pattern_statistics_promotion", {}))
        _text(
            cv2, image,
            f"statistics table: {promotion.get('decision', 'unknown')} | {promotion.get('reason', '')}",
            30, 650, 0.50, green if promotion.get("decision") == "accept" else red if promotion.get("reason") == "validation_regression" else amber, 2,
        )
        path = output_root / video_id / "video_pattern_summary.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(path), image):
            summaries.append({"video_id": video_id, "visualization_path": str(path)})

    manifest = {
        "version": 2,
        "num_track_images": len(rendered),
        "num_summary_images": len(summaries),
        "num_skipped": len(skipped),
        "track_images": rendered,
        "summary_images": summaries,
        "skipped": skipped,
    }
    (output_root / "trajectory_pattern_visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        **state,
        "trajectory_pattern_visualizations": rendered,
        "trajectory_pattern_video_summaries": summaries,
        "trajectory_pattern_visualization_skipped": skipped,
        "trajectory_pattern_visualization_output_root": output_root,
    }
