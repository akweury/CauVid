"""Final ego-symbol materialization and offline audit artifacts for Step 7F."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
from pathlib import Path


VERSION = 4
VISUALIZATION_VERSION = 8
CUE_NAMES = (
    "ego_static", "ego_driving_forward", "ego_driving_backward",
    "ego_turning_left", "ego_turning_right", "ego_straight",
    "ego_accelerating", "ego_decelerating", "ego_motion_uncertain",
)


def _cues(action, signal, acceleration_threshold, validated):
    cues = {name: 0.0 for name in CUE_NAMES}
    if action == "unknown":
        cues["ego_motion_uncertain"] = 1.0
        return cues
    if not validated:
        # Preserve the best numerical state and expose uncertainty separately.
        cues["ego_motion_uncertain"] = 1.0
    if action == "static":
        cues["ego_static"] = 1.0
    elif action == "backward":
        cues["ego_driving_backward"] = 1.0
        cues["ego_straight"] = 1.0
    else:
        cues["ego_driving_forward"] = 1.0
        if action in {"left", "turning_left"}:
            cues["ego_turning_left"] = 1.0
        elif action in {"right", "turning_right"}:
            cues["ego_turning_right"] = 1.0
        else:
            cues["ego_straight"] = 1.0
    delta = signal.get("ego_speed_delta")
    if delta is not None:
        if float(delta) > acceleration_threshold:
            cues["ego_accelerating"] = 1.0
        elif float(delta) < -acceleration_threshold:
            cues["ego_decelerating"] = 1.0
    return cues



def _segment_signal_confidence(action, signals, static_threshold, lateral_threshold):
    """Score support from continuous signals without turning uncertainty into unknown."""
    finite = lambda key: [
        float(row[key]) for row in signals
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    speeds, vz_values, vx_values = finite("ego_speed"), finite("ego_vz"), finite("ego_vx")
    if not signals or action == "unknown":
        return 0.0
    if action == "static":
        if not speeds:
            return 0.0
        median_speed = sorted(speeds)[len(speeds) // 2]
        return float(max(0.0, min(1.0, 1.0 - median_speed / max(1e-6, 2.0 * static_threshold))))
    if action in {"forward", "backward"}:
        if not vz_values:
            return 0.0
        signed = sorted(vz_values)[len(vz_values) // 2] * (1.0 if action == "forward" else -1.0)
        return float(max(0.0, min(1.0, signed / max(1e-6, 3.0 * static_threshold))))
    if action in {"left", "turning_left", "right", "turning_right"}:
        if not vx_values:
            return 0.0
        magnitude = sorted(abs(value) for value in vx_values)[len(vx_values) // 2]
        return float(max(0.0, min(1.0, magnitude / max(1e-6, 3.0 * lateral_threshold))))
    return 0.0


def finalize_video(refinement, provisional):
    signal_by_frame = {
        int(row.get("frame_index", offset)): row
        for offset, row in enumerate(provisional.get("continuous_signals", []))
    }
    rule_by_segment = {
        int(row.get("segment_id", -1)): row
        for row in refinement.get("selected_global_rule_evaluation", {}).get("segment_evaluations", [])
    }
    evidence_by_segment = {
        int(row.get("segment_id", -1)): row
        for row in refinement.get("selected_normalized_evidence", {}).get("normalized_segment_evidence", [])
    }
    acceleration_threshold = float(
        refinement.get("selected_thresholds", {}).get(
            "acceleration_threshold",
            provisional.get("configuration", {}).get("acceleration_threshold", 0.0),
        )
    )
    frames = []
    final_segments = []
    corrections = {int(row.get("segment_id", -1)): row for row in refinement.get("corrections", [])}
    provisional_frames = {
        int(row.get("frame_index", offset)): row
        for offset, row in enumerate(provisional.get("frames", []))
    }
    for segment in refinement.get("refined_segments", []):
        segment_id = int(segment.get("segment_id", len(final_segments)))
        validated = segment.get("validation_status") == "validated"
        action = str(segment.get("action", "unknown"))
        supported_actions = {"static", "forward", "backward", "left", "right", "turning_left", "turning_right"}
        published_action = action if action in supported_actions else "unknown"
        rules = rule_by_segment.get(segment_id, {})
        evidence = evidence_by_segment.get(segment_id, {})
        hypotheses = dict(rules.get("hypothesis_scores", {}))
        target_family = (
            "left" if action in {"left", "turning_left"}
            else "right" if action in {"right", "turning_right"}
            else action
        )
        rule_confidence = float(hypotheses.get(target_family, max(hypotheses.values(), default=0.0)))
        segment_signals = [
            signal_by_frame.get(frame_index, {})
            for frame_index in range(int(segment.get("start_frame", 0)), int(segment.get("end_frame", 0)) + 1)
        ]
        selected_thresholds = dict(refinement.get("selected_thresholds", provisional.get("selected_thresholds", {})))
        signal_confidence = _segment_signal_confidence(
            published_action,
            segment_signals,
            float(selected_thresholds.get("static_speed_threshold", 0.2)),
            float(selected_thresholds.get("lateral_threshold", 0.2)),
        )
        evidence_confidence = 1.0 - float(evidence.get("uncertainty", 1.0))
        combined_confidence = 0.55 * signal_confidence + 0.25 * rule_confidence + 0.20 * evidence_confidence
        if validated:
            confidence = max(combined_confidence, 0.5 * rule_confidence + 0.5 * evidence_confidence)
        else:
            confidence = min(0.79, combined_confidence)
        if published_action != "unknown":
            confidence = max(0.05, confidence)
        confidence = float(max(0.0, min(1.0, confidence)))
        correction = corrections.get(segment_id)
        explanation = (
            f"Corrected provisional {','.join(correction.get('provisional_actions', []))} to {action} because {correction.get('reason')}; best evidence hypothesis={correction.get('best_rule_hypothesis')}."
            if correction else
            f"Retained {action}; validated by shared rules with confidence {confidence:.3f}."
        )
        if not validated:
            explanation = f"Segment retained as uncertain: {', '.join(segment.get('uncertainty_reasons', [])) or 'insufficient rule support'}."
        final_segments.append({
            **segment,
            "action_before_finalization": action,
            "action": published_action,
            "validated_action": action if validated else None,
            "confidence": confidence,
            "signal_confidence": signal_confidence,
            "semantic_confidence": rule_confidence,
            "prediction_status": "validated" if validated else "soft_uncertain",
            "correction": correction,
            "correction_reason": explanation,
            "fired_rule_ids": [row.get("rule_id") for row in rules.get("fired_rules", [])],
            "violated_rule_ids": [row.get("rule_id") for row in rules.get("violated_rules", [])],
            "hypothesis_scores": hypotheses,
            "supporting_evidence_values": evidence,
        })
        for frame_index in range(int(segment.get("start_frame", 0)), int(segment.get("end_frame", 0)) + 1):
            signal = signal_by_frame.get(frame_index, {"frame_index": frame_index})
            provisional_frame = provisional_frames.get(frame_index, {})
            frames.append({
                "frame_index": frame_index,
                "action": published_action,
                "validated_action": action if validated else None,
                "provisional_action": provisional_frame.get("action", "unknown"),
                "observable_cues": _cues(published_action, signal, acceleration_threshold, validated),
                "signal_evidence": {key: signal.get(key) for key in ("ego_vx", "ego_vz", "ego_yaw_rate", "ego_speed", "ego_speed_delta")},
                "segment_id": segment_id,
                "validation_status": segment.get("validation_status"),
                "confidence": confidence,
                "signal_confidence": signal_confidence,
                "semantic_confidence": rule_confidence,
                "prediction_status": "validated" if validated else "soft_uncertain",
                "correction_reason": explanation,
            })
    frames.sort(key=lambda row: row["frame_index"])
    aggregate = {
        cue: float(sum(frame["observable_cues"][cue] for frame in frames) / max(1, len(frames)))
        for cue in CUE_NAMES
    }
    return {
        "version": VERSION,
        "video_id": str(refinement.get("video_id", "")),
        "status": "completed",
        "role": "final_validated_ego_symbols",
        "label_status": "final",
        "downstream_usable_as_final": True,
        "source_steps": ["7a", "7b", "7c", "7d", "7e"],
        "selected_threshold": dict(refinement.get("selected_thresholds", {})),
        "selected_thresholds": dict(refinement.get("selected_thresholds", {})),
        "provisional_thresholds": dict(refinement.get("provisional_thresholds", {})),
        "threshold_changes": dict(refinement.get("threshold_changes", {})),
        "threshold_status": "final_validated_and_frozen",
        "provisional_segments": list(refinement.get("provisional_segments", [])),
        "provisional_frames": copy.deepcopy(provisional.get("frames", [])),
        "final_action_segments": final_segments,
        "frames": frames,
        "num_frames": len(frames),
        "aggregate_cues": aggregate,
        "corrections": list(refinement.get("corrections", [])),
        "uncertain_segments": list(refinement.get("uncertain_segments", [])),
        "candidate_rankings": list(refinement.get("candidate_rankings", [])),
        "rule_evaluation": dict(refinement.get("selected_global_rule_evaluation", {})),
        "normalized_evidence": dict(refinement.get("selected_normalized_evidence", {})),
        "refinement_iterations": list(refinement.get("iterations", [])),
        "refinement_stop_reason": refinement.get("stop_reason"),
        "provenance": {
            "provisional_preserved": True,
            "only_validated_segments_materialized": True,
            "uncertain_segments_publish_motion_uncertain": True,
            "deterministic": True,
        },
    }


_HTML = """<!doctype html><html><head><meta charset='utf-8'><title>Step 7F Ego Symbol Audit</title><style>
body{font-family:system-ui;background:#11151b;color:#e8edf3;margin:20px}select,button{font-size:16px;padding:6px;background:#202733;color:#fff}.card{background:#1a2029;padding:14px;margin:12px 0;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #394351;padding:6px;text-align:left}.good{color:#70dc92}.warn{color:#ffd166}pre{white-space:pre-wrap;max-height:420px;overflow:auto}.bar{height:18px;background:#29313d;margin:2px}.seg{display:inline-block;height:18px}h2{margin-bottom:6px}</style></head><body>
<h1>Step 7F — Final Ego Symbol Audit</h1><select id='video'></select><div id='view'></div><script>const D=__DATA__;const sel=document.getElementById('video');D.forEach((v,i)=>{let o=document.createElement('option');o.value=i;o.textContent=v.video_id;sel.appendChild(o)});function esc(x){return String(x??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}function render(){const v=D[+sel.value||0], seg=v.final_action_segments||[];let rows=seg.map(s=>`<tr><td>${s.segment_id}</td><td>${s.start_frame}-${s.end_frame}</td><td>${esc((s.correction||{}).provisional_actions||s.action_before_finalization)}</td><td class='${s.validation_status==='validated'?'good':'warn'}'>${esc(s.action)}</td><td>${(+s.confidence).toFixed(3)}</td><td>${esc(s.fired_rule_ids)}</td><td>${esc(s.correction_reason)}</td></tr>`).join('');document.getElementById('view').innerHTML=`<div class=card><h2>Thresholds</h2><pre>${esc(JSON.stringify({provisional:v.provisional_thresholds,final:v.selected_thresholds,changes:v.threshold_changes},null,2))}</pre></div><div class=card><h2>Initial → Final segments</h2><table><tr><th>ID</th><th>Frames</th><th>Initial</th><th>Final</th><th>Confidence</th><th>Fired rules</th><th>Explanation</th></tr>${rows}</table></div><div class=card><h2>Continuous signals, patch evidence and candidates</h2><pre>${esc(JSON.stringify({frames:v.frames,normalized_evidence:v.normalized_evidence,candidate_rankings:v.candidate_rankings,rule_evaluation:v.rule_evaluation,iterations:v.refinement_iterations},null,2))}</pre></div>`}sel.onchange=render;render();</script></body></html>"""


def build_html(final_videos, output_root):
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "index.html"
    encoded = json.dumps(final_videos, separators=(",", ":"), ensure_ascii=False).replace("</", "<\/")
    path.write_text(_HTML.replace("__DATA__", encoded), encoding="utf-8")
    return str(path)


def _draw_signal_chart(cv2, canvas, values, current, box, color, label):
    x1, y1, x2, y2 = box
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (50, 57, 68), 1)
    cv2.putText(canvas, label, (x1 + 5, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    bound = max(1e-9, max([abs(v) for v in finite] or [1.0])); mid=(y1+y2)//2
    # Bright dashed zero reference remains visible against all signal colors.
    dash_length, dash_gap = 14, 8
    dash_x = x1
    while dash_x < x2:
        cv2.line(
            canvas, (dash_x, mid), (min(x2, dash_x + dash_length), mid),
            (80, 245, 255), 2, cv2.LINE_AA,
        )
        dash_x += dash_length + dash_gap
    points=[]
    for i,value in enumerate(values):
        if value is None: continue
        x=x1+int(i*(x2-x1-1)/max(1,len(values)-1)); y=mid-int(float(value)/bound*(y2-y1-20)/2); points.append((x,y))
    for a,b in zip(points,points[1:]): cv2.line(canvas,a,b,color,2,cv2.LINE_AA)
    mx=x1+int(current*(x2-x1-1)/max(1,len(values)-1));cv2.line(canvas,(mx,y1),(mx,y2),(255,255,255),2,cv2.LINE_AA)


def _segment_for_frame(rows, frame_index):
    return next(
        (
            row for row in rows
            if int(row.get("start_frame", 0)) <= frame_index
            <= int(row.get("end_frame", -1))
        ),
        {},
    )


_EGO_STATE_COLORS = {
    "forward": (70, 220, 110),
    "backward": (80, 120, 245),
    "static": (180, 180, 180),
    "left": (245, 175, 70),
    "right": (220, 90, 225),
    "straight": (90, 210, 230),
    "accelerating": (70, 230, 150),
    "decelerating": (70, 145, 245),
    "steady": (190, 190, 105),
    "uncertain": (80, 84, 94),
    "unknown": (80, 84, 94),
}


def _ego_timeline_states(frame):
    cues = dict(frame.get("observable_cues", {}))
    action = str(frame.get("action", "")).lower()
    if float(cues.get("ego_static", 0.0)) > 0.5:
        longitudinal = "static"
    elif float(cues.get("ego_driving_backward", 0.0)) > 0.5:
        longitudinal = "backward"
    elif float(cues.get("ego_driving_forward", 0.0)) > 0.5:
        longitudinal = "forward"
    elif action == "static":
        longitudinal = "static"
    elif action == "backward":
        longitudinal = "backward"
    elif action == "forward":
        longitudinal = "forward"
    else:
        longitudinal = "uncertain"
    if longitudinal == "static":
        return "static", "static", "static"
    if float(cues.get("ego_turning_left", 0.0)) > 0.5:
        lateral = "left"
    elif float(cues.get("ego_turning_right", 0.0)) > 0.5:
        lateral = "right"
    elif float(cues.get("ego_straight", 0.0)) > 0.5:
        lateral = "straight"
    else:
        # A valid non-turning longitudinal state is laterally stable, not unknown.
        lateral = "straight" if longitudinal != "uncertain" else "uncertain"
    if float(cues.get("ego_accelerating", 0.0)) > 0.5:
        speed_change = "accelerating"
    elif float(cues.get("ego_decelerating", 0.0)) > 0.5:
        speed_change = "decelerating"
    else:
        speed_change = "steady" if longitudinal != "uncertain" else "uncertain"
    return longitudinal, lateral, speed_change


def _draw_ego_state_timelines(cv2, canvas, frames, current_frame, box):
    """Draw three synchronized final-decision bars under the source video."""
    x1, y1, x2, y2 = box
    rows = sorted(frames, key=lambda item: int(item.get("frame_index", 0)))
    if not rows:
        return
    first = int(rows[0].get("frame_index", 0))
    last = int(rows[-1].get("frame_index", first))
    span = max(1, last - first)
    label_width = 265
    bar_left = x1 + label_width
    lane_height = 32
    lane_gap = 10
    lane_names = ("LONGITUDINAL", "LATERAL", "SPEED CHANGE")
    current_row = min(
        rows,
        key=lambda item: abs(int(item.get("frame_index", first)) - current_frame),
    )
    current_states = _ego_timeline_states(current_row)
    cv2.putText(
        canvas, "FINAL EGO STATE TIMELINE [7F]", (x1, y1 - 9),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 247, 250), 2, cv2.LINE_AA,
    )
    for lane, (name, active_state) in enumerate(zip(lane_names, current_states)):
        top = y1 + lane * (lane_height + lane_gap)
        bottom = top + lane_height
        cv2.rectangle(canvas, (x1, top), (x2, bottom), (25, 30, 38), -1)
        cv2.putText(
            canvas, f"{name}: {active_state}", (x1 + 8, top + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50,
            _EGO_STATE_COLORS.get(active_state, _EGO_STATE_COLORS["unknown"]),
            2, cv2.LINE_AA,
        )
        for index, frame in enumerate(rows):
            frame_id = int(frame.get("frame_index", first))
            next_id = (
                int(rows[index + 1].get("frame_index", frame_id + 1))
                if index + 1 < len(rows) else last + 1
            )
            left = bar_left + int((frame_id - first) * (x2 - bar_left) / span)
            right = bar_left + int((next_id - first) * (x2 - bar_left) / span)
            right = max(bar_left, min(x2, right))
            state = _ego_timeline_states(frame)[lane]
            cv2.rectangle(
                canvas, (left, top + 3), (max(left + 1, right), bottom - 3),
                _EGO_STATE_COLORS.get(state, _EGO_STATE_COLORS["unknown"]), -1,
            )
        cv2.rectangle(canvas, (bar_left, top + 3), (x2, bottom - 3), (105, 112, 124), 1)
        marker_x = bar_left + int((current_frame - first) * (x2 - bar_left) / span)
        marker_x = max(bar_left, min(x2, marker_x))
        cv2.line(canvas, (marker_x, top - 2), (marker_x, bottom + 2), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(canvas, (marker_x, top + 2), 5, (255, 255, 255), -1, cv2.LINE_AA)


def _panel_text(cv2, canvas, text, x, y, color, scale=0.60, thickness=2, max_chars=54):
    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    for offset, line in enumerate(lines[:3]):
        cv2.putText(
            canvas, line, (int(x), int(y + offset * 27)),
            cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA,
        )
    return y + max(1, min(3, len(lines))) * 27


def _summary_table_cells(summary):
    cells = []
    plain_index = 0
    for token in (part.strip() for part in str(summary).split("|")):
        if not token:
            continue
        if "=" in token:
            name, value = token.split("=", 1)
        elif "->" in token:
            name, value = "correction", token
        else:
            name = "label" if plain_index == 0 else "state"
            value = token
            plain_index += 1
        cells.append((name.strip(), value.strip()))
    return cells


def _draw_step_block(cv2, canvas, top, step, title, summary, color):
    """Draw a title plus a two-row key/value table for one Step 7 stage."""
    left, right = 1140, 1900
    height = 146
    cv2.rectangle(canvas, (left, top), (right, top + height), (27, 32, 41), -1)
    cv2.rectangle(canvas, (left, top), (left + 10, top + height), color, -1)
    cv2.putText(
        canvas, f"{step}  {title}", (left + 25, top + 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 3, cv2.LINE_AA,
    )
    cells = _summary_table_cells(summary)
    table_left, table_right = left + 20, right - 12
    table_top, table_mid, table_bottom = top + 50, top + 91, top + 137
    cv2.rectangle(
        canvas, (table_left, table_top), (table_right, table_bottom),
        (74, 82, 94), 2,
    )
    cv2.line(
        canvas, (table_left, table_mid), (table_right, table_mid),
        (74, 82, 94), 2, cv2.LINE_AA,
    )
    count = max(1, len(cells))
    for index, (name, value) in enumerate(cells):
        cell_left = table_left + int(index * (table_right - table_left) / count)
        cell_right = table_left + int((index + 1) * (table_right - table_left) / count)
        if index:
            cv2.line(
                canvas, (cell_left, table_top), (cell_left, table_bottom),
                (74, 82, 94), 2, cv2.LINE_AA,
            )
        available = max(20, cell_right - cell_left - 10)
        header_scale = 0.62
        value_scale = 0.60
        while cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, header_scale, 2)[0][0] > available and header_scale > 0.38:
            header_scale -= 0.03
        while cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, value_scale, 2)[0][0] > available and value_scale > 0.38:
            value_scale -= 0.03
        header_width = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, header_scale, 2)[0][0]
        value_width = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, value_scale, 2)[0][0]
        cv2.putText(
            canvas, name, (cell_left + max(5, (cell_right - cell_left - header_width) // 2), table_top + 28),
            cv2.FONT_HERSHEY_SIMPLEX, header_scale, color, 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, value, (cell_left + max(5, (cell_right - cell_left - value_width) // 2), table_mid + 31),
            cv2.FONT_HERSHEY_SIMPLEX, value_scale, (245, 247, 250), 2, cv2.LINE_AA,
        )


def _rule_atom_text(atom):
    field = str(atom.get("field", "unknown"))
    operator = str(atom.get("operator", "eq"))
    expected = atom.get("expected")
    operator_text = {"eq": "=", "ge": ">=", "le": "<=", "abs_ge": "abs>=", "abs_le": "abs<="}.get(operator, operator)
    if isinstance(expected, float):
        expected_text = f"{expected:.3f}".rstrip("0").rstrip(".")
    else:
        expected_text = str(expected)
    if operator.startswith("abs_"):
        return f"abs({field}(S)) {operator_text[3:]} {expected_text}"
    return f"{field}(S) {operator_text} {expected_text}"


def _draw_complete_rule(cv2, canvas, rule, left, top, width, color):
    hypothesis = str(rule.get("hypothesis", "unknown"))
    cv2.putText(
        canvas, f"{hypothesis}(S) :-", (left, top + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA,
    )
    atoms = list(rule.get("atoms", []))
    for atom_index, atom in enumerate(atoms[:4]):
        suffix = "," if atom_index < len(atoms) - 1 else "."
        text = _rule_atom_text(atom) + suffix
        scale = 0.48
        while cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width and scale > 0.32:
            scale -= 0.02
        cv2.putText(
            canvas, text, (left + 13, top + 43 + atom_index * 22),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 240, 245), 2, cv2.LINE_AA,
        )


_EGO_LABEL_COLORS = {
    "forward": (70, 220, 110),
    "backward": (80, 120, 245),
    "static": (180, 180, 180),
    "left": (245, 175, 70),
    "turning_left": (245, 175, 70),
    "right": (220, 90, 225),
    "turning_right": (220, 90, 225),
    "unknown": (80, 84, 94),
}


def _label_segments_for_rule_stage(final_segments, rule_segments):
    rules_by_id = {
        int(row.get("segment_id", -1)): row for row in rule_segments
    }
    rows = []
    for segment in final_segments:
        candidate = dict(segment)
        rule = rules_by_id.get(int(segment.get("segment_id", -1)), {})
        scores = dict(rule.get("hypothesis_scores", {}))
        candidate["action"] = (
            max(scores.items(), key=lambda item: (float(item[1]), item[0]))[0]
            if scores else segment.get("action_before_finalization", "unknown")
        )
        rows.append(candidate)
    return rows


def _label_segments_for_refined_stage(final_segments):
    rows = []
    for segment in final_segments:
        candidate = dict(segment)
        candidate["action"] = segment.get(
            "action_before_finalization", segment.get("action", "unknown")
        )
        rows.append(candidate)
    return rows


def _segment_label_states(segment):
    action = str(segment.get("action", "unknown"))
    if action == "static":
        return ("static", "static", "static")
    if action == "backward":
        return ("backward", "straight", "steady")
    if action in {"left", "turning_left"}:
        return ("forward", "left", "steady")
    if action in {"right", "turning_right"}:
        return ("forward", "right", "steady")
    if action == "forward":
        return ("forward", "straight", "steady")
    return ("uncertain", "uncertain", "uncertain")



def _compare_segment_label_sequences(reference, candidate, first, last):
    changed_frames = []
    for frame_index in range(first, last + 1):
        reference_action = str(_segment_for_frame(reference, frame_index).get("action", "unknown"))
        candidate_action = str(_segment_for_frame(candidate, frame_index).get("action", "unknown"))
        if reference_action != candidate_action:
            changed_frames.append(frame_index)
    return {
        "identical": not changed_frames,
        "changed_frame_count": len(changed_frames),
        "total_frame_count": max(0, last - first + 1),
        "changed_frames": changed_frames,
    }


def _draw_label_version_panel(
    cv2,
    canvas,
    provisional_segments,
    provisional_frames,
    final_segments,
    rule_segments,
    final_frames,
    current_frame,
):
    """Show three synchronized ego-state lanes for every Step 7 version."""
    left, right = 1140, 1900
    cv2.rectangle(canvas, (1122, 0), (1919, 1079), (15, 19, 25), -1)
    cv2.putText(
        canvas, "EGO LABEL UPDATES: 3 STATE BARS", (left, 39),
        cv2.FONT_HERSHEY_SIMPLEX, 0.76, (248, 249, 252), 3, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "longitudinal | lateral | speed change", (left, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (165, 175, 188), 1, cv2.LINE_AA,
    )
    rule_labels = _label_segments_for_rule_stage(final_segments, rule_segments)
    refined_labels = _label_segments_for_refined_stage(final_segments)
    versions = (
        ("7A", "ORIGINAL", provisional_frames, True, "original frame-level prediction"),
        ("7D", "RULE UPDATE", rule_labels, False, "updated by 7D"),
        ("7E", "REFINED", refined_labels, False, "compare with 7A"),
        ("7F", "FINAL", final_frames, True, "validated by 7F"),
    )
    all_segments = list(provisional_segments) + list(final_segments)
    first = min((int(row.get("start_frame", 0)) for row in all_segments), default=0)
    last = max((int(row.get("end_frame", first)) for row in all_segments), default=first)
    span = max(1, last - first)
    refinement_comparison = _compare_segment_label_sequences(
        provisional_segments, refined_labels, first, last
    )
    lane_names = ("LONG", "LAT", "SPEED")
    row_top, row_height, row_gap = 82, 220, 20
    for version_index, (step, title, records, frame_mode, provenance) in enumerate(versions):
        top = row_top + version_index * (row_height + row_gap)
        bottom = top + row_height
        accent = (70, 180, 245) if version_index == 0 else (80, 225, 120)
        if step == "7E":
            current_7a = str(_segment_for_frame(provisional_segments, current_frame).get("action", "unknown"))
            current_7e = str(_segment_for_frame(refined_labels, current_frame).get("action", "unknown"))
            current_relation = "current changed" if current_7a != current_7e else "current identical"
            if refinement_comparison["identical"]:
                provenance = f"IDENTICAL TO 7A | {current_relation}"
                accent = (80, 225, 120)
            else:
                provenance = (
                    f"UPDATED FROM 7A | {current_relation} | "
                    f"changed={refinement_comparison['changed_frame_count']}/"
                    f"{refinement_comparison['total_frame_count']}"
                )
                accent = (70, 180, 245)
        cv2.rectangle(canvas, (left, top), (right, bottom), (27, 33, 42), -1)
        cv2.rectangle(canvas, (left, top), (left + 9, bottom), accent, -1)
        cv2.putText(
            canvas, f"{step} {title}", (left + 20, top + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.61, accent, 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, provenance, (left + 230, top + 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (205, 212, 222), 1, cv2.LINE_AA,
        )
        if frame_mode:
            current_record = min(
                records or [{}],
                key=lambda item: abs(int(item.get("frame_index", first)) - current_frame),
            )
            current_states = _ego_timeline_states(current_record)
        else:
            current_record = _segment_for_frame(records, current_frame)
            current_states = _segment_label_states(current_record)
        label_width = 215
        bar_left, bar_right = left + label_width, right - 15
        for lane, (lane_name, active_state) in enumerate(zip(lane_names, current_states)):
            lane_top = top + 58 + lane * 49
            lane_bottom = lane_top + 36
            color = _EGO_LABEL_COLORS.get(
                active_state, _EGO_STATE_COLORS.get(active_state, (80, 84, 94))
            )
            confidence_suffix = (
                f" c={float(current_record.get('confidence', 0.0)):.2f}"
                if frame_mode and lane == 0 else ""
            )
            cv2.putText(
                canvas, f"{lane_name}: {active_state}{confidence_suffix}", (left + 18, lane_top + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 2, cv2.LINE_AA,
            )
            cv2.rectangle(canvas, (bar_left, lane_top), (bar_right, lane_bottom), (58, 64, 75), -1)
            if frame_mode:
                ordered = sorted(records, key=lambda item: int(item.get("frame_index", first)))
                for record_index, record in enumerate(ordered):
                    frame_id = int(record.get("frame_index", first))
                    next_id = (
                        int(ordered[record_index + 1].get("frame_index", frame_id + 1))
                        if record_index + 1 < len(ordered) else last + 1
                    )
                    state = _ego_timeline_states(record)[lane]
                    x_start = bar_left + int((frame_id - first) * (bar_right - bar_left) / span)
                    x_end = bar_left + int((next_id - first) * (bar_right - bar_left) / span)
                    x_end = max(x_start + 1, min(bar_right, x_end))
                    cv2.rectangle(
                        canvas, (x_start, lane_top + 2), (x_end, lane_bottom - 2),
                        _EGO_STATE_COLORS.get(state, _EGO_STATE_COLORS["unknown"]), -1,
                    )
            else:
                for segment in records:
                    segment_start = int(segment.get("start_frame", first))
                    segment_end = int(segment.get("end_frame", segment_start))
                    state = _segment_label_states(segment)[lane]
                    x_start = bar_left + int((segment_start - first) * (bar_right - bar_left) / span)
                    x_end = bar_left + int((segment_end + 1 - first) * (bar_right - bar_left) / span)
                    x_start = max(bar_left, min(bar_right, x_start))
                    x_end = max(x_start + 1, min(bar_right, x_end))
                    cv2.rectangle(
                        canvas, (x_start, lane_top + 2), (x_end, lane_bottom - 2),
                        _EGO_STATE_COLORS.get(state, _EGO_STATE_COLORS["unknown"]), -1,
                    )
            marker = bar_left + int((current_frame - first) * (bar_right - bar_left) / span)
            marker = max(bar_left, min(bar_right, marker))
            cv2.line(
                canvas, (marker, lane_top - 2), (marker, lane_bottom + 2),
                (255, 255, 255), 2, cv2.LINE_AA,
            )


def render_mp4s(final_videos, position_by_video, output_root, fps=10.0, limit=5):
    """Render a clean source-video column and a large Step 7A–7F audit column."""
    import cv2
    import numpy as np

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rendered, skipped = [], []
    colors = {
        "7A": (70, 180, 245),
        "7B": (225, 200, 70),
        "7C": (245, 150, 70),
        "7D": (210, 100, 225),
        "7E": (70, 220, 245),
        "7F": (80, 225, 120),
    }
    for final in sorted(final_videos, key=lambda row: str(row.get("video_id", "")))[:max(0, int(limit))]:
        video_id = str(final.get("video_id", ""))
        position = position_by_video.get(video_id, {})
        source = {
            int(frame.get("frame_index", index)): frame
            for index, frame in enumerate(position.get("frames", []))
        }
        frames = list(final.get("frames", []))
        available = [
            (row, source.get(int(row.get("frame_index", -1)), {}))
            for row in frames
        ]
        available = [
            pair for pair in available
            if str(pair[1].get("image_path", ""))
            and Path(str(pair[1].get("image_path", ""))).exists()
        ]
        if not available:
            skipped.append({"video_id": video_id, "reason": "missing_images"})
            continue
        path = output_root / f"{video_id}_ego_symbol_audit.mp4"
        cache_metadata_path = path.with_suffix(".visual_cache.json")
        source_metadata = []
        for _, source_frame in available:
            image_path = Path(str(source_frame.get("image_path", "")))
            stat = image_path.stat()
            source_metadata.append({
                "frame_index": int(source_frame.get("frame_index", len(source_metadata))),
                "image_path": str(image_path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
        visual_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "visualization_version": VISUALIZATION_VERSION,
                    "final_video": final,
                    "source_metadata": source_metadata,
                    "fps": float(fps),
                    "resolution": [1920, 1080],
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()
        cached_visual = None
        if path.exists() and path.stat().st_size > 0 and cache_metadata_path.exists():
            try:
                cached_visual = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                cached_visual = None
        if cached_visual and cached_visual.get("visual_fingerprint") == visual_fingerprint:
            rendered.append({
                "video_id": video_id,
                "path": str(path),
                "cache_hit": True,
                "visual_fingerprint": visual_fingerprint,
            })
            continue
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(0.1, float(fps)),
            (1920, 1080),
        )
        provisional_segments = list(final.get("provisional_segments", []))
        final_segments = list(final.get("final_action_segments", []))
        normalized_segments = list(
            final.get("normalized_evidence", {}).get("normalized_segment_evidence", [])
        )
        raw_segments = list(
            final.get("normalized_evidence", {}).get("audit", {}).get(
                "raw_evidence_preserved", {}
            ).get("segments", [])
        )
        rule_segments = list(
            final.get("rule_evaluation", {}).get("segment_evaluations", [])
        )
        threshold_changes = dict(final.get("threshold_changes", {}))
        selected_candidate = str(
            (final.get("candidate_rankings") or [{}])[0].get("candidate_id", "unknown")
        )
        for row, source_frame in available:
            frame_index = int(row.get("frame_index", 0))
            image = cv2.imread(str(source_frame.get("image_path")))
            if image is None:
                continue
            canvas = np.full((1080, 1920, 3), (12, 15, 20), np.uint8)
            # Left column: intentionally clean source image, no boxes or flow overlays.
            source_left, source_top, source_right, source_bottom = 0, 0, 1120, 405
            scale = min(
                (source_right - source_left) / image.shape[1],
                (source_bottom - source_top) / image.shape[0],
            )
            view = cv2.resize(
                image,
                (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
            )
            x0 = source_left + (source_right - source_left - view.shape[1]) // 2
            y0 = source_top + (source_bottom - source_top - view.shape[0]) // 2
            canvas[y0:y0 + view.shape[0], x0:x0 + view.shape[1]] = view
            cv2.line(canvas, (1120, 0), (1120, 1080), (95, 103, 116), 4, cv2.LINE_AA)
            cv2.putText(
                canvas, f"ORIGINAL VIDEO   frame {frame_index}", (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.92, (255, 255, 255), 2, cv2.LINE_AA,
            )
            current_signal_index = min(
                range(len(frames)),
                key=lambda index: abs(
                    int(frames[index].get("frame_index", frame_index)) - frame_index
                ),
            )
            ego_vx = [
                item.get("signal_evidence", {}).get("ego_vx") for item in frames
            ]
            ego_vz = [
                item.get("signal_evidence", {}).get("ego_vz") for item in frames
            ]
            _draw_signal_chart(
                cv2, canvas, ego_vx, current_signal_index,
                (24, 430, 550, 552), (90, 225, 255), "EGO vx [7F]",
            )
            _draw_signal_chart(
                cv2, canvas, ego_vz, current_signal_index,
                (569, 430, 1095, 552), (100, 245, 145), "EGO vz [7F]",
            )
            _draw_ego_state_timelines(
                cv2, canvas, frames, frame_index, (24, 580, 1095, 716)
            )

            provisional = _segment_for_frame(provisional_segments, frame_index)
            final_segment = _segment_for_frame(final_segments, frame_index)
            segment_id = int(final_segment.get("segment_id", row.get("segment_id", -1)))
            raw = next((item for item in raw_segments if int(item.get("segment_id", -2)) == segment_id), {})
            normalized = next((item for item in normalized_segments if int(item.get("segment_id", -2)) == segment_id), {})
            rules = next((item for item in rule_segments if int(item.get("segment_id", -2)) == segment_id), {})
            scores = sorted(
                dict(rules.get("hypothesis_scores", final_segment.get("hypothesis_scores", {}))).items(),
                key=lambda item: (-float(item[1]), item[0]),
            )[:3]
            fired_rows = list(rules.get("fired_rules", []))
            fired = [str(item.get("rule_id")) for item in fired_rows]
            violated = [str(item.get("rule_id")) for item in rules.get("violated_rules", [])]
            correction = final_segment.get("correction")

            if fired_rows:
                rule_left, rule_top, rule_right, rule_bottom = 24, 735, 1095, 1058
                cv2.rectangle(
                    canvas, (rule_left, rule_top), (rule_right, rule_bottom),
                    (24, 30, 38), -1,
                )
                cv2.rectangle(
                    canvas, (rule_left, rule_top), (rule_left + 9, rule_bottom),
                    colors["7D"], -1,
                )
                cv2.putText(
                    canvas, "FIRED RULES [7D]  HEAD + BODY", (rule_left + 25, rule_top + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.88, colors["7D"], 3, cv2.LINE_AA,
                )
                rule_cell_width = 510
                rule_cell_height = 128
                for rule_index, fired_rule in enumerate(fired_rows[:4]):
                    column = rule_index % 2
                    row_index = rule_index // 2
                    cell_left = rule_left + 28 + column * 520
                    cell_top = rule_top + 55 + row_index * rule_cell_height
                    if column:
                        cv2.line(
                            canvas, (cell_left - 13, cell_top),
                            (cell_left - 13, cell_top + rule_cell_height - 7),
                            (62, 70, 82), 1, cv2.LINE_AA,
                        )
                    if row_index:
                        cv2.line(
                            canvas, (cell_left, cell_top - 5),
                            (cell_left + rule_cell_width - 15, cell_top - 5),
                            (62, 70, 82), 1, cv2.LINE_AA,
                        )
                    _draw_complete_rule(
                        cv2, canvas, fired_rule, cell_left, cell_top,
                        rule_cell_width - 24, (80, 225, 120),
                    )

            initial_action = provisional.get("action", row.get("provisional_action", "unknown"))
            provisional_thresholds = dict(final.get("provisional_thresholds", {}))
            summary_7a = (
                f"{initial_action} | static_th={float(provisional_thresholds.get('static_speed_threshold', 0)):.2f} "
                f"| lat_th={float(provisional_thresholds.get('lateral_threshold', 0)):.2f} "
                f"| yaw_th={float(provisional_thresholds.get('yaw_threshold', 0)):.3f}"
            )
            summary_7b = (
                f"patch={int(raw.get('num_accepted_vectors', 0))} | reg={len(raw.get('covered_regions', []))} "
                f"| exp={float(raw.get('radial_expansion_support', 0)):.2f} "
                f"| con={float(raw.get('radial_contraction_support', 0)):.2f} "
                f"| pers={float(raw.get('temporal_persistence', 0)):.2f}"
            )
            direction = str(normalized.get("dominant_radial_direction", "unknown"))
            direction_short = {"expansion": "exp", "contraction": "con", "neutral": "neutral"}.get(direction, direction)
            summary_7c = (
                f"mag={float(normalized.get('normalized_motion_magnitude', 0)):.2f} "
                f"| z={float(normalized.get('motion_magnitude_robust_z', 0)):+.2f} "
                f"| dir={direction_short} | sup={float(normalized.get('direction_support_ratio', 0)):.2f} "
                f"| agr={float(normalized.get('estimator_agreement', 0)):.2f} "
                f"| unc={float(normalized.get('uncertainty', 1)):.2f}"
            )
            score_text = " | ".join(f"{name[:3]}={float(score):.2f}" for name, score in scores)
            summary_7d = (
                f"{score_text} | fired={len(fired)} | viol={len(violated)} "
                f"| conflict={len(rules.get('conflicts', []))}"
            )
            changed_thresholds = [
                f"d_{key.replace('_threshold', '').replace('static_speed', 'static').replace('lateral', 'lat')}={float(value):+.2f}"
                for key, value in threshold_changes.items()
                if abs(float(value)) > 1e-12
            ]
            change_text = ",".join(changed_thresholds[:2]) or "d_th=0"
            refined_action = final_segment.get("action_before_finalization", final_segment.get("action", "unknown"))
            candidate_short = selected_candidate.rsplit("_", 1)[-1]
            summary_7e = (
                f"c={candidate_short} | {str(final.get('refinement_stop_reason', 'unknown')).replace('thresholds_and_labels_', '')} "
                f"| {change_text} | {initial_action}->{refined_action}"
            )
            final_action = row.get("action", "unknown")
            final_status = str(final_segment.get("validation_status", "uncertain"))
            correction_text = f"{initial_action}->{final_action}" if initial_action != final_action else "unchanged"
            summary_7f = (
                f"{final_action} | {final_status} | conf={float(row.get('confidence', 0)):.2f} "
                f"| {correction_text}"
            )
            _draw_label_version_panel(
                cv2,
                canvas,
                provisional_segments,
                list(final.get("provisional_frames", [])),
                final_segments,
                rule_segments,
                frames,
                frame_index,
            )
            writer.write(canvas)
        writer.release()
        cache_metadata_path.write_text(
            json.dumps(
                {
                    "visualization_version": VISUALIZATION_VERSION,
                    "visual_fingerprint": visual_fingerprint,
                    "video_id": video_id,
                    "path": str(path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        rendered.append({
            "video_id": video_id,
            "path": str(path),
            "cache_hit": False,
            "visual_fingerprint": visual_fingerprint,
        })
    return {
        "version": VISUALIZATION_VERSION,
        "layout": "original_video_signals_states_rules_left_three_lane_ego_state_versions_right",
        "resolution": [1920, 1080],
        "rendered": rendered,
        "skipped": skipped,
        "limit": limit,
    }
