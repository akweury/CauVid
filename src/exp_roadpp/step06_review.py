"""Step 06r: review videos, the test videos with the predicted events drawn in, for a human review.

For a random sample of the test videos (REVIEW["num_videos"], seed REVIEW["seed"], only videos where
at least one scorer alarms), one MP4 per video and scorer of REVIEW["scorers"] (the rule net, the MLP
and the calibrated own-label lookup), in <video>/<scorer>.mp4, each scorer at the threshold where its
episode recall is closest to REVIEW["target_recall"] (`choose_thresholds`), so the same videos can be
compared at a matched recall. Each shows the camera frames with
Layout: the camera frame on the left, a side column on the right with the current list (top) and
the timeline (below), a header line over both.
  - every agent's GT box (thin grey) with its id "#n"; an agent with an active alarm or a shown
    event gets a thick box in the colour below, and an item in the current list with the alarm
    (head, score, outcome) or the event, and the agent's class and current GT labels;
  - the alarms of the scorer (`step06_episodes.alarm_episodes`: the frames whose window-W score of an
    agent and head is >= the threshold, merged into episodes, >= min_length frames) as a thick box
    around the alarmed agent ("ALARM <head> p=<score>" in the list), coloured by the outcome of the alarm
    under the step 06 episode rules (tolerance REVIEW["tolerance_s"]):
      correct (green): the event of that head follows on that agent within the alarm (+ tolerance);
      group (cyan): it follows on an adjacent agent of the same class (`grid.agent_groups`) only;
      false (red): no such event, the outcome seen; unobservable (grey): the agent left too soon;
  - the GT events (exact onsets of the counted heads) as "EVENT <head>" on the agent's box for 1 s
    from the onset, "MISSED" (yellow) if no alarm of the agent and head hit it, "MISSED (group hit)"
    (cyan) if only an alarm of an adjacent same-class agent did;
  - the timeline of the whole video: one row per agent and head with an alarm or an event (alarm
    bars in the alarm colours over the alarm's frames, above a thin bar over each event's frames from
    its onset to its end with a tick at the onset, event ticks blue, missed ones yellow, group hits cyan), a seconds axis, the current frame as a cursor.
Next to the videos, review.csv: one line per alarm and per event and scorer (video, scorer, agent,
class, head, frames, seconds, peak score, outcome, matched event / lead), with empty columns for the
reviewer's verdict and comment; review_summary.csv: one line per video and scorer (alarm outcomes,
precision, events caught / missed); README.txt explaining the colours, thresholds and verdicts.

Frames: from frame_dir/<video>/%05d.jpg if it exists, else decoded from video_dir/<video>.mp4.
The atom tensors are indexed by the ROAD++ frame id, which starts at 1 (the frames are extracted as
00001.jpg, ...): tensor frame t is image t (%05d) and decoded frame t - FRAME_OFFSET.
"""
import csv
import os
import random
import torch
from tqdm import tqdm
from src.exp_roadpp import step05, step06, step06_episodes, utils_data

SCORER_NAMES = {"model": "rule net", "mlp": "MLP", "lookup": "own-label lookup",
                "lookup_cal": "own-label lookup (calibrated)", "model_cal": "rule net (calibrated)",
                "mlp_cal": "MLP (calibrated)"}
FRAME_OFFSET = 1                            # tensor frame t = decoded video frame t - FRAME_OFFSET
REVIEW = {
    "num_videos": 100,
    "seed": 0,
    # Scorers to compare on the same videos ("_cal": its probabilities calibrated on val), each at the
    # threshold where its episode recall (headline window, tolerance_s) is closest to target_recall.
    "scorers": ("model", "mlp", "lookup_cal"),
    "target_recall": 0.4,
    "fallback_thresholds": {"model": 0.5, "mlp": 0.5, "lookup_cal": 0.05, "lookup": 0.05, "model_cal": 0.05,
                            "mlp_cal": 0.05},
    "tolerance_s": 1.0,
    "width": 1280,                          # width of the camera frame (left), pixels (scaled down)
    "side_width": 640,                      # width of the side column (right): current list and timeline
    "event_show_s": 1.0,                    # an event label stays this long after the onset
    "timeline_rows": 40,                    # at most (and as many as fit under the list)
    "panel_lines": 10,                      # items of the current list at most (active alarms and shown events)
    "version": "review_v9_three_scorers",
}
# One clearly different colour per category: correct alarm (green), false alarm (red), GT event caught
# by an alarm (blue), GT event missed (yellow); a hit at the group level only, an alarm whose event
# happens to an adjacent agent of the same class or an event caught only by such an agent's alarm
# (cyan); an alarm whose outcome cannot be seen (grey).
COLORS = {                                  # BGR
    "correct": (0, 210, 0), "false": (0, 0, 255), "event": (255, 150, 40), "missed": (0, 230, 255),
    "group": (255, 255, 0), "missed_group": (255, 255, 0), "unobservable": (150, 150, 150),
    "box": (190, 190, 190), "text": (255, 255, 255), "panel": (30, 30, 30),
}


def _event_key(e):
    """The colour key of a GT event: caught by an alarm of its agent, by one of its group only, or missed."""
    return "event" if e["detected"] else ("missed_group" if e["detected_group"] else "missed")


# -------------- Scores, alarms and events of one video --------------
def _scorer(name, model, mlp, baselines, calibration, device):
    """f(inputs, stage1) -> (B, W, A) scores of a scorer name ("model", "mlp", "lookup", + "_cal")."""
    base = name[:-4] if name.endswith("_cal") else name
    f = step06._scorers(model, baselines, mlp, None, device, names=(base,)).get(base)
    if f is None:
        raise ValueError(f"Review: scorer {base} is not available.")
    if name.endswith("_cal"):
        if not calibration or base not in calibration:
            raise ValueError(f"Review: no calibration of {base}.")
        cal = calibration[base]
        return lambda inputs, stage1: step06.calibrate(f(inputs, stage1), cal)
    return f


def analyse_video(model, video, A, scorer, heads, cfg, threshold, k, tol):
    """Alarms and events of one test video for one scorer, window index k, threshold and tolerance.
    Returns {"X", "score" (T, N, H), "alarms": [...], "events": [...]} or None if too short."""
    device, m, min_context = cfg["device"], cfg["min_off_frames"], cfg["min_context"]
    X = step05._video_from_cells(video, A, device)
    roles = step05.video_roles(video, device)
    T, N, _ = X.shape
    if T <= min_context:
        return None
    H, w = len(heads), model.windows[k]
    is_row = torch.zeros((T, N), dtype=torch.bool, device=device)
    at_risk = torch.zeros((T, N, H), dtype=torch.bool, device=device)
    score = torch.zeros((T, N, H), device=device)
    with torch.no_grad():
        for begin in range(min_context, T, cfg["frames_per_video"]):
            frames = torch.arange(begin, min(begin + cfg["frames_per_video"], T), device=device)
            rows = step05.video_rows(model, X, frames, m, None, roles)
            if len(rows["rows"]) == 0:
                continue
            t, n = rows["rows"].T
            is_row[t, n] = True
            at_risk[t, n] = rows["at_risk"][:, heads]
            stage1 = model.window_probs(model(rows["inputs"])[0])
            score[t, n] = scorer(rows["inputs"], stage1)[:, k][:, heads]
    onsets = step05.atom_onsets(X[..., heads], m)
    onsets[0] = False
    onsets[1:] &= X[:-1].any(dim=2, keepdim=True)
    onsets[:min_context + 1] = False
    e_t, e_n, e_h = torch.nonzero(onsets, as_tuple=True)
    e_end = step06_episodes._event_ends(X, e_t, e_n, heads[e_h], m).to(device)
    groups = video["groups"].to(device).long() if "groups" in video else None
    next_gone = torch.cat([step06_episodes._next_index(~is_row), torch.full((1, N), T, dtype=torch.long, device=device)])
    pair, start, end = step06_episodes.alarm_episodes(
        score.reshape(T, N * H), is_row[:, :, None].expand(T, N, H).reshape(T, N * H), at_risk.reshape(T, N * H),
        threshold, cfg["fill_gap"], cfg["min_length"])
    s_n, s_h = pair // H, pair % H
    same = (e_n[:, None] == s_n[None]) & (e_h[:, None] == s_h[None])                  # (E, S)
    timing = (start[None] <= e_t[:, None]) & (e_t[:, None] <= end[None] + tol)
    hit = same & timing
    if groups is not None and len(e_t):
        g_e = groups[e_t - 1, e_n][:, None]                                           # (E, 1)
        g_s = groups[(e_t - 1)[:, None], s_n[None]]                                   # (E, S)
        hit_group = (e_h[:, None] == s_h[None]) & (g_e >= 0) & (g_e == g_s) & timing & ~same
    else:
        hit_group = torch.zeros_like(hit)
    seen = ((next_gone[(end + 1).clamp(max=T), s_n] > end + max(tol, 1))
            | (next_gone[start, s_n] > start + w - 1))
    alarms = []
    for i in range(len(pair)):
        a, b, n, h = int(start[i]), int(end[i]), int(s_n[i]), int(s_h[i])
        if hit[:, i].any():
            status, j = "correct", int(torch.nonzero(hit[:, i])[0])
        elif hit_group[:, i].any():
            status, j = "group", int(torch.nonzero(hit_group[:, i])[0])
        else:
            status, j = ("false" if bool(seen[i]) else "unobservable"), None
        alarms.append({"n": n, "h": h, "start": a, "end": b, "status": status,
                       "peak": float(score[a:b + 1, n, h].max()),
                       "event": None if j is None else {"n": int(e_n[j]), "t": int(e_t[j])}})
    events = []
    for j in range(len(e_t)):
        by = torch.nonzero(hit[j]).flatten().tolist()
        events.append({"n": int(e_n[j]), "h": int(e_h[j]), "t": int(e_t[j]), "end": int(e_end[j]),
                       "detected": bool(by), "detected_group": bool(hit_group[j].any()),
                       "lead": (int(e_t[j]) - min(int(start[i]) for i in by)) if by else None})
    return {"X": X.cpu(), "score": score.cpu(), "alarms": alarms, "events": events, "T": T, "N": N}


# -------------- Frames --------------
class FrameReader:
    """The camera frame of each tensor frame t of one video: frame_dir/<video>/%05d.jpg, else the
    decoded video_dir/<video>.mp4 (read in order); None where neither has it."""

    def __init__(self, video_id, frame_dir, video_dir):
        import cv2
        self.cv2 = cv2
        self.dir = os.path.join(frame_dir, video_id) if frame_dir else None
        if self.dir and not os.path.isdir(self.dir):
            self.dir = None
        self.cap, self.next = None, 0
        if self.dir is None and video_dir:
            for ext in (".mp4", ".avi", ".mov"):
                path = os.path.join(video_dir, video_id + ext)
                if os.path.exists(path):
                    self.cap = cv2.VideoCapture(path)
                    break

    @property
    def available(self):
        return self.dir is not None or self.cap is not None

    def read(self, t):
        if self.dir is not None:
            img = self.cv2.imread(os.path.join(self.dir, f"{t:05d}.jpg"))
            return img
        i = t - FRAME_OFFSET
        if self.cap is None or i < 0:
            return None
        while self.next <= i:
            ok, img = self.cap.read()
            self.next += 1
            if not ok:
                self.cap = None
                return None
            if self.next - 1 == i:
                return img
        return None

    def close(self):
        if self.cap is not None:
            self.cap.release()


# -------------- Drawing --------------
def _text(cv2, img, s, x, y, color, scale=0.45, bg=True):
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(max(0, min(x, img.shape[1] - w - 2))), int(max(h + 2, min(y, img.shape[0] - 2)))
    if bg:
        cv2.rectangle(img, (x - 1, y - h - 3), (x + w + 1, y + base - 1), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_video(result, video_id, boxes, classes, label_names, head_names, out_path, reader, scorer_name,
                 threshold, window_s, fps, review):
    """Write one review MP4 of `analyse_video` result: a header line over the camera frame (left) and
    a side column (right) with the current alarms / events and the GT labels of their agents (top)
    and the timeline of the whole video (below)."""
    import cv2
    import numpy as np
    X, score, T, N = result["X"], result["score"], result["T"], result["N"]
    W_img = review["width"]
    H_img = int(round(W_img * 1280 / 1920))
    W_side, header_h = review["side_width"], 26
    show = int(round(review["event_show_s"] * fps))
    L = len(label_names)

    # The list (top of the side column): one line per alarm active at t and per event shown at t,
    # the informative ones first, so a full list cuts the false and unobservable alarms.
    rank = {"correct": 1, "group": 2, "false": 3, "unobservable": 4}

    def active(t):
        items = [("alarm", a) for a in result["alarms"] if a["start"] <= t <= a["end"]]
        items += [("event", e) for e in result["events"] if e["t"] <= t < e["t"] + show]
        return sorted(items, key=lambda it: 0 if it[0] == "event" else rank[it[1]["status"]])
    list_lines = review["panel_lines"]                                  # a fixed list height
    line_h = 34                                                         # two text lines per item
    list_h = 28 + line_h * list_lines

    # The timeline (below the list): one row per agent and head with an alarm or an event, alarm bars
    # in the alarm colours, event ticks, a seconds axis; the cursor is drawn per frame.
    pairs = []
    for a in sorted(result["alarms"], key=lambda a: a["start"]):
        if (a["n"], a["h"]) not in pairs:
            pairs.append((a["n"], a["h"]))
    for e in sorted(result["events"], key=lambda e: e["t"]):
        if (e["n"], e["h"]) not in pairs:
            pairs.append((e["n"], e["h"]))
    # The timeline fills the column down to the bottom of the frame: its rows share the height left
    # under the list (between MIN_ROW_H and MAX_ROW_H pixels each; rows that do not fit are left out).
    min_row_h, max_row_h, axis_h, title_h = 14, 28, 26, 24
    tl_h = H_img - list_h
    room = max((tl_h - title_h - axis_h) // min_row_h, 1)
    shown = min(len(pairs), room, review["timeline_rows"])
    hidden = len(pairs) - shown
    pairs = pairs[:shown]
    row_h = min(max((tl_h - title_h - axis_h) // max(shown, 1), min_row_h), max_row_h)
    timeline = np.full((tl_h, W_side, 3), COLORS["panel"], dtype=np.uint8)
    x0, x1 = 150, W_side - 10

    def tx(t):
        return int(x0 + (x1 - x0) * t / max(T - 1, 1))
    _text(cv2, timeline, "Timeline (whole video)" + (f", {hidden} more rows not shown" if hidden else ""),
          6, 17, COLORS["text"], 0.45, bg=False)
    for r, (n, h) in enumerate(pairs):
        y = title_h + r * row_h
        _text(cv2, timeline, f"#{n} {head_names[h]}"[:22], 6, y + row_h // 2 + 5, COLORS["text"], 0.4, bg=False)
        cv2.line(timeline, (x0, y + row_h // 2), (x1, y + row_h // 2), (70, 70, 70), 1)
        # Alarms: a thick bar in the upper part of the row, from the alarm start to its end.
        # Events: a thin bar in the lower part, from the onset to the end of the event (the label off
        # for min_off_frames frames, or the agent gone), and a tick over the whole row at the onset.
        split = y + (row_h * 3) // 5
        for a in result["alarms"]:
            if (a["n"], a["h"]) == (n, h):
                cv2.rectangle(timeline, (tx(a["start"]), y + 2), (max(tx(a["end"]), tx(a["start"]) + 1), split - 1),
                              COLORS[a["status"]], -1)
        for e in result["events"]:
            if (e["n"], e["h"]) == (n, h):
                c = COLORS[_event_key(e)]
                cv2.rectangle(timeline, (tx(e["t"]), split + 2), (max(tx(e["end"]), tx(e["t"]) + 1), split + 4), c, -1)
                cv2.line(timeline, (tx(e["t"]), y), (tx(e["t"]), y + row_h - 1), c, 2)
    axis_y = tl_h - axis_h + 4                                          # the axis at the bottom of the column
    cv2.line(timeline, (x0, axis_y), (x1, axis_y), (120, 120, 120), 1)
    step = 5 if T / fps <= 60 else 10
    for sec in range(0, int(T / fps) + 1, step):
        x = tx(int(sec * fps))
        cv2.line(timeline, (x, axis_y), (x, axis_y + 4), (120, 120, 120), 1)
        _text(cv2, timeline, f"{sec}s", x - 8, axis_y + 16, (170, 170, 170), 0.35, bg=False)
    cursor_bottom = axis_y + 4

    W_out, H_out = W_img + W_side, header_h + H_img
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_out, H_out))
    sx, sy = W_img, H_img

    def box_of(t, n):
        b = boxes[t, n]
        if torch.isnan(b).any():
            return None
        return (int(b[0] * sx), int(b[1] * sy)), (int(b[2] * sx), int(b[3] * sy))
    for t in range(T):
        img = reader.read(t)
        if img is None:
            frame = np.zeros((H_img, W_img, 3), dtype=np.uint8)
            _text(cv2, frame, "no camera frame", 10, 30, COLORS["text"])
        else:
            frame = cv2.resize(img, (W_img, H_img), interpolation=cv2.INTER_AREA)
        items = active(t)
        marked = {}
        for kind, x in items:                                   # the colour and box width of each marked agent
            key = x["status"] if kind == "alarm" else _event_key(x)
            marked.setdefault(x["n"], COLORS[key])
        # Every agent: a thin grey box and its id; the marked ones a thick coloured box.
        for n in range(N):
            pb = box_of(t, n)
            if pb is None:
                continue
            p1, p2 = pb
            if n in marked:
                color = marked[n]
                cv2.rectangle(frame, p1, p2, color, 3)
                _text(cv2, frame, f"#{n}", p1[0], p1[1] - 3, color, 0.5)
            else:
                cv2.rectangle(frame, p1, p2, COLORS["box"], 1)
                _text(cv2, frame, f"#{n}", p1[0], p1[1] - 2, COLORS["box"], 0.33, bg=False)
        # Side column, top: the list of what is on now (agent and its GT labels, then the alarm / event).
        listing = np.full((list_h, W_side, 3), (22, 22, 22), dtype=np.uint8)
        _text(cv2, listing, f"Now: {len([i for i in items if i[0] == 'alarm'])} alarms, "
              f"{len([i for i in items if i[0] == 'event'])} events", 6, 18, COLORS["text"], 0.45, bg=False)
        for i, (kind, x) in enumerate(items[:list_lines]):
            n, y = x["n"], 28 + line_h * i
            labels = ",".join(label_names[a].split(":", 1)[-1] for a in torch.nonzero(X[t, n, :L]).flatten().tolist())
            if kind == "alarm":
                c = COLORS[x["status"]]
                what = f"ALARM {head_names[x['h']]}  p={float(score[t, n, x['h']]):.2f}  [{x['status']}]"
            else:
                c = COLORS[_event_key(x)]
                what = ("EVENT " if x["detected"] else ("MISSED (group hit) " if x["detected_group"] else "MISSED ")) \
                    + head_names[x["h"]]
            _text(cv2, listing, what, 6, y + 13, c, 0.45, bg=False)
            _text(cv2, listing, f"  #{n} {classes[n] or ''}  GT: {labels or '-'}"[:70], 6, y + 29, (200, 200, 200), 0.4,
                  bg=False)
        if len(items) > list_lines:
            _text(cv2, listing, f"+{len(items) - list_lines} more", W_side - 80, 18, COLORS["text"], 0.4, bg=False)
        tl = timeline.copy()
        cv2.line(tl, (tx(t), title_h - 2), (tx(t), cursor_bottom), (255, 255, 255), 1)
        side = np.vstack([listing, tl])
        header = np.full((header_h, W_out, 3), COLORS["panel"], dtype=np.uint8)
        _text(cv2, header, f"{video_id}  frame {t}  {t / fps:5.1f}s  |  {scorer_name} >= {threshold}  window {window_s:g}s"
              f"  |  green correct   red false   blue event caught   yellow event missed   cyan group hit   grey unobservable",
              4, 17, COLORS["text"], 0.45, bg=False)
        writer.write(np.vstack([header, np.hstack([frame, side])]))
    writer.release()


# -------------- Main --------------
def choose_thresholds(episode_file, window, tolerance_s):
    """{scorer: (threshold, its episode recall or None)} of REVIEW["scorers"]: the threshold of the
    scorer whose object-level episode recall (`step06_episodes`, window and tolerance) is closest to
    REVIEW["target_recall"], so the scorers alarm at a matched recall; REVIEW["fallback_thresholds"]
    if the episode metrics lack the scorer."""
    metrics = utils_data.load_json(episode_file) if episode_file and os.path.exists(episode_file) else {}
    results = metrics.get("results") or {}
    out = {}
    for name in REVIEW["scorers"]:
        by_th = (results.get(name) or {}).get(str(window)) or {}
        points = [(float(th), by_tol[str(float(tolerance_s))]["recall"]) for th, by_tol in by_th.items()
                  if str(float(tolerance_s)) in by_tol]
        if points:
            out[name] = min(points, key=lambda p: (abs(p[1] - REVIEW["target_recall"]), -p[0]))
        else:
            out[name] = (REVIEW["fallback_thresholds"][name], None)
    return out


def _lines(name, video_id, result, classes, head_names, fps):
    """review.csv lines of one video and scorer: one per alarm and per GT event."""
    lines = []
    for a in result["alarms"]:
        ev = a["event"]
        lines.append({"video": video_id, "scorer": name, "kind": "alarm", "agent": f"#{a['n']}",
                      "class": classes[a["n"]], "head": head_names[a["h"]], "start_frame": a["start"],
                      "end_frame": a["end"], "start_s": round(a["start"] / fps, 1), "end_s": round(a["end"] / fps, 1),
                      "peak_score": round(a["peak"], 3), "outcome": a["status"],
                      "matched": "" if ev is None else f"#{ev['n']} onset {ev['t']} ({ev['t'] / fps:.1f}s)",
                      "lead_s": "" if ev is None else round((ev["t"] - a["start"]) / fps, 1),
                      "verdict": "", "comment": ""})
    for e in result["events"]:
        lines.append({"video": video_id, "scorer": name, "kind": "event", "agent": f"#{e['n']}",
                      "class": classes[e["n"]], "head": head_names[e["h"]], "start_frame": e["t"],
                      "end_frame": e["end"], "start_s": round(e["t"] / fps, 1), "end_s": round(e["end"] / fps, 1),
                      "peak_score": "", "outcome": "detected" if e["detected"] else
                      ("missed (group hit)" if e["detected_group"] else "missed"),
                      "matched": "", "lead_s": "" if e["lead"] is None else round(e["lead"] / fps, 1),
                      "verdict": "", "comment": ""})
    return lines


def _summary(name, video_id, threshold, result):
    """review_summary.csv line of one video and scorer: the alarm outcomes and the events caught."""
    alarms = [a["status"] for a in result["alarms"]]
    events = result["events"]
    seen = sum(s in ("correct", "group", "false") for s in alarms)
    return {"video": video_id, "scorer": name, "threshold": threshold, "alarms": len(alarms),
            "correct": alarms.count("correct"), "group": alarms.count("group"), "false": alarms.count("false"),
            "unobservable": alarms.count("unobservable"),
            "precision": round(alarms.count("correct") / seen, 3) if seen else "",
            "events": len(events), "detected": sum(e["detected"] for e in events),
            "missed_group_hit": sum((not e["detected"]) and e["detected_group"] for e in events),
            "missed": sum((not e["detected"]) and (not e["detected_group"]) for e in events)}


def main(model, mlp, baselines, calibration, test_data, step05_cfg, cfg, output_dir, frame_dir, video_dir,
         episode_file=None):
    """The review videos of REVIEW in output_dir: for each sampled test video, one MP4 per scorer of
    REVIEW["scorers"] in output_dir/<video>/<scorer>.mp4, each at its matched-recall threshold
    (`choose_thresholds`, from the episode metrics of episode_file); review.csv (one line per alarm and
    event), review_summary.csv (one line per video and scorer) and README.txt. Skipped with a message
    if no camera frame source exists, or if the videos of these settings and models are already there."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[Review]\t skipped: opencv (cv2) is not installed")
        return
    if not ((frame_dir and os.path.isdir(frame_dir)) or (video_dir and os.path.isdir(video_dir))):
        print(f"[Review]\t skipped: neither the frame dir {frame_dir} nor the video dir {video_dir} exists")
        return
    os.makedirs(output_dir, exist_ok=True)
    done = output_dir / "review.csv"
    sources = [step05_cfg["model_file"], step05_cfg["mlp_model_file"], cfg["test_data_file"]]
    if episode_file:
        sources.append(episode_file)
    fps, window = cfg["frame_rate"], cfg["headline_window"]
    thresholds = choose_thresholds(episode_file, window, REVIEW["tolerance_s"])
    info_file = output_dir / "review_settings.json"
    settings = {**REVIEW, "frame_offset": FRAME_OFFSET, "thresholds": {n: list(t) for n, t in thresholds.items()}}
    if (step05.is_fresh(done, *sources) and os.path.exists(info_file)
            and utils_data.load_json(info_file) == settings):
        print(f"[Review]\t {output_dir}: up to date")
        return
    for name, (th, recall) in thresholds.items():
        print(f"[Review]\t {name}: threshold {th}" + (f" (episode recall {recall:.3f}, target {REVIEW['target_recall']})"
                                                     if recall is not None else " (fallback, no episode metrics)"))
    device = cfg["device"]
    ep = step06_episodes._default_cfg(fps)
    rcfg = {**cfg, "min_off_frames": test_data["min_off_frames"], "fill_gap": ep["fill_gap"],
            "min_length": ep["min_length"]}
    head_type = cfg["head_types"].to(device)
    counted = [step06.EVENT_TYPES.index(t) for t in step06.COUNTED_TYPES]
    heads = torch.nonzero(torch.isin(head_type, torch.tensor(counted, device=device))).flatten()
    vocab = step05.load_vocabulary(cfg)
    L = test_data.get("num_label_atoms", test_data["num_atoms"])
    label_names = [step05._atom_name(s) for s in vocab[:L]]
    head_names = [step05._atom_name(vocab[int(h)]) for h in heads]
    k = model.windows.index(window)
    tol = int(round(REVIEW["tolerance_s"] * fps))
    scorers = {name: _scorer(name, model, mlp, baselines, calibration, device) for name in thresholds}
    atoms = utils_data.load_json(cfg["test_atoms_file"])
    boxes_all = torch.load(cfg["test_box_file"], map_location="cpu") if os.path.exists(cfg["test_box_file"]) else {}
    order = sorted(test_data["videos"])
    random.Random(REVIEW["seed"]).shuffle(order)
    lines, summary, chosen = [], [], 0
    for video_id in tqdm(order, desc="Review videos"):
        if chosen >= REVIEW["num_videos"]:
            break
        if not FrameReader(video_id, frame_dir, video_dir).available:
            continue
        results = {name: analyse_video(model, test_data["videos"][video_id], test_data["num_atoms"], scorers[name],
                                       heads, rcfg, thresholds[name][0], k, tol) for name in scorers}
        if any(r is None for r in results.values()) or not any(r["alarms"] for r in results.values()):
            continue                                            # too short, or no scorer alarms in it
        out_dir = output_dir / video_id
        os.makedirs(out_dir, exist_ok=True)
        any_result = next(iter(results.values()))
        vb, classes, _has_box = step05._agent_boxes(boxes_all, video_id, atoms[video_id], any_result["X"][..., :L])
        for name, result in results.items():
            reader = FrameReader(video_id, frame_dir, video_dir)        # read in order: one per video
            render_video(result, video_id, vb, classes, label_names, head_names, out_dir / f"{name}.mp4", reader,
                         SCORER_NAMES.get(name, name), thresholds[name][0], window / fps, fps, REVIEW)
            reader.close()
            lines += _lines(name, video_id, result, classes, head_names, fps)
            summary.append(_summary(name, video_id, thresholds[name][0], result))
        chosen += 1
    for path, rows in ((done, lines), (output_dir / "review_summary.csv", summary)):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["video"])
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "README.txt").write_text(_readme(cfg, fps, thresholds))
    utils_data.save_json(settings, info_file)
    totals = {name: {key: sum(s[key] for s in summary if s["scorer"] == name)
                     for key in ("alarms", "correct", "false", "events", "detected")} for name in thresholds}
    for name, t in totals.items():
        seen = t["correct"] + t["false"]
        print(f"[Review]\t {name}: {t['alarms']} alarms ({t['correct']} correct, {t['false']} false"
              + (f", precision {t['correct'] / seen:.3f}" if seen else "") + f"), {t['detected']}/{t['events']} events caught")
    print(f"[Review]\t {chosen} videos x {len(thresholds)} scorers in {output_dir}; {len(lines)} lines in {done}")


def _readme(cfg, fps, thresholds):
    scorers = "\n".join(f"  {name}.mp4: {SCORER_NAMES.get(name, name)}, alarm threshold {th}"
                        + (f" (episode recall {recall:.2f})" if recall is not None else "")
                        for name, (th, recall) in thresholds.items())
    return f"""Review videos (step06_review.py), frame rate {fps}: one folder per test video, one MP4 per scorer,
each scorer at the threshold where its episode recall over the whole test split is closest to
{REVIEW['target_recall']} (so the three catch about as many events and their alarms can be compared):
{scorers}
review_summary.csv: per video and scorer, the alarms (correct / group / false / unobservable, precision)
and the GT events (caught / missed).

Window: "the event starts within {cfg['headline_window'] / fps:g} s". Alarm: the frames where the score of an agent
and head is >= the threshold, merged into one alarm (gaps of 1 frame bridged, >= 0.5 s long).
Box colours: green = correct alarm (the event follows on this agent within the alarm + {REVIEW['tolerance_s']} s),
cyan = the event follows on an adjacent agent of the same class only, red = false alarm, grey = the
agent left too soon to tell. EVENT (blue) = a GT event that an alarm caught, MISSED (yellow) = one
that none did, MISSED (group hit) (cyan) = one caught only by an adjacent same-class agent's alarm.
Thin grey boxes: every agent with its id #n. Right column, top: what is on now (alarm / event,
the agent's class and current GT labels); below: the timeline,
one row per agent and head with an alarm or event over the whole video: the thick bar is the alarm
(start to end), the thin bar below it the GT event (onset to end, a tick at the onset); the white
line is now.

review.csv: one line per alarm and per GT event, per scorer. Fill in "verdict" and "comment". Suggested verdicts:
  alarm:  reasonable (the situation makes the event likely; it just did not happen) /
          wrong_agent (the event happened to another agent) / early_or_late (right event, timing off) /
          label_error (the event did happen, the GT misses it) / no_reason (nothing suggests it);
  event:  predictable (the video shows signs before the onset) / unpredictable / label_error.
"""
