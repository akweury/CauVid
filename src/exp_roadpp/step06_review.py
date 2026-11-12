"""Step 06r: review videos, the test videos with the predicted events drawn in, for a human review.

For a random sample of the test videos (REVIEW["num_videos"], seed REVIEW["seed"], only videos where
some scorer alarms on a reviewed agent), three agents per video, each in videos of its own
(`focus_agents`): the ego and the REVIEW["focus_others"] other agents seen in the most frames. Per
agent, <video>/<agent>/<scorer>.mp4 for each scorer of REVIEW["scorers"] (the rule net, the MLP and
the calibrated own-label lookup), each at the threshold where its episode recall is closest to
REVIEW["target_recall"] (`choose_thresholds`), so the scorers can be compared at a matched recall, and
<video>/<agent>/gt.mp4 with the ground truth only (`gt_only`).
Layout (`render_video`): a header line over the camera frame (left) and a side column (right) about
the reviewed agent only, in large type (REVIEW["font_scale"]):
  - the frame: the reviewed agent in a thick box (the ego, without a box: a border while it has an
    alarm or event), every other agent in a thin grey box;
  - who it is and its current GT labels;
  - its alarms (`step06_episodes.alarm_episodes`: the frames whose window-W score of the agent and a
    head is >= the threshold, merged into episodes, >= min_length frames, "ALARM <head> p=<score>")
    and GT events now, coloured by the outcome under the step 06 episode rules (tolerance
    REVIEW["tolerance_s"]):
      correct (green): the event of that head follows on that agent within the alarm (+ tolerance);
      group (cyan): it follows on an adjacent agent of the same class (`grid.agent_groups`) only;
      false (red): no such event, the outcome seen; unobservable (grey): the agent left too soon;
    a GT event (exact onset of a counted head) shows for 1 s from the onset: "EVENT" (blue) if an
    alarm of the agent and head hit it, "MISSED" (yellow) if none, "MISSED (group)" (cyan) if only an
    alarm of an adjacent same-class agent did;
  - its scores now for the heads of its class (`agent_heads`), the REVIEW["pred_lines"] highest, as
    bars with the threshold (not in gt.mp4);
  - its timeline over the whole video: one row per head with an alarm or an event (alarm bars in the
    alarm colours, below them a thin bar over each event from its onset to its end with a tick at the
    onset), a seconds axis, the current frame as a cursor.
Next to the videos, review.csv: one line per alarm and per event of the reviewed agents and scorer
(video, scorer, agent, class, head, frames, seconds, peak score, outcome, matched event / lead), with
empty columns for the reviewer's verdict and comment; review_summary.csv: one line per video, agent
and scorer (alarm outcomes, precision, events caught / missed); README.txt explaining the colours,
thresholds and verdicts.

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
    "num_videos": 10,
    "seed": 0,
    # Scorers to compare on the same videos ("_cal": its probabilities calibrated on val), each at the
    # threshold where its episode recall (headline window, tolerance_s) is closest to target_recall.
    "scorers": ("model", "mlp", "lookup_cal"),
    "target_recall": 0.4,
    "fallback_thresholds": {"model": 0.5, "mlp": 0.5, "lookup_cal": 0.05, "lookup": 0.05, "model_cal": 0.05,
                            "mlp_cal": 0.05},
    "tolerance_s": 1.0,
    "width": 1280,                          # width of the camera frame (left), pixels (scaled down)
    "side_width": 720,                      # width of the side column (right): the focus agent only
    "font_scale": 0.75,                     # type size of the side column
    "focus_others": 2,                      # agents per video besides the ego: the ones seen longest
    "event_show_s": 1.0,                    # an event label stays this long after the onset
    "panel_lines": 3,                       # items of the now-list at most (the agent's alarms and events)
    "pred_lines": 6,                        # heads of the prediction bars (the highest scores now)
    "version": "review_v12_focus_agents",
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
def _text(cv2, img, s, x, y, color, scale=0.45, bg=True, thick=1):
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(max(0, min(x, img.shape[1] - w - 2))), int(max(h + 2, min(y, img.shape[0] - 2)))
    if bg:
        cv2.rectangle(img, (x - 1, y - h - 3), (x + w + 1, y + base - 1), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _wrap(s, width):
    """s cut into lines of at most width characters, at the commas where possible."""
    lines, cur = [], ""
    for part in s.split(","):
        nxt = part if not cur else cur + "," + part
        if len(nxt) <= width:
            cur = nxt
        else:
            if cur:
                lines.append(cur + ",")
            cur = part[:width]
    return lines + ([cur] if cur else [])


def focus_agents(X_labels, ego, num_other):
    """The agents of one video to review, each in a video of its own: the ego (if any), then the
    num_other other agents seen in the most frames (holding some label atom)."""
    frames = X_labels.any(dim=2).sum(dim=0)                                     # (N,)
    others = [n for n in torch.argsort(frames, descending=True).tolist() if n != ego and frames[n] > 0]
    return ([ego] if ego is not None else []) + others[:num_other]


def agent_heads(X_labels, n, label_names, head_names):
    """Indices into head_names of the heads of agent n's class (the class prefix of its label atoms:
    "av" for the ego, "Car", "Ped", ...)."""
    held = torch.nonzero(X_labels[:, n].any(dim=0)).flatten().tolist()
    prefix = label_names[held[0]].split(":", 1)[0] if held else None
    return [i for i, h in enumerate(head_names) if h.split(":", 1)[0] == prefix]


def of_agent(result, n):
    """The `analyse_video` result restricted to agent n's alarms and events."""
    return {**result, "alarms": [a for a in result["alarms"] if a["n"] == n],
            "events": [e for e in result["events"] if e["n"] == n]}


def render_video(result, video_id, boxes, classes, label_names, head_names, out_path, reader, scorer_name,
                 threshold, window_s, fps, review, ego=None, focus=None, heads=()):
    """Write one review MP4 of the focus agent of `analyse_video` result (`of_agent`): a header line
    over the camera frame (left; the focus agent's box marked) and a side column (right) about the
    focus agent only, in large type: who it is and its current GT labels; its alarms and GT events
    now; its scores now for the heads of its class (heads), the highest first, as bars with the
    alarm threshold (not in the ground-truth video, threshold None); its timeline over the whole video."""
    import cv2
    import numpy as np
    X, score, T, N = result["X"], result["score"], result["T"], result["N"]
    n0 = focus
    W_img = review["width"]
    H_img = int(round(W_img * 1280 / 1920))
    W_side, header_h = review["side_width"], 34
    show = int(round(review["event_show_s"] * fps))
    L = len(label_names)
    fs = review["font_scale"]                                   # the side column's type size
    line_h = int(round(44 * fs))
    small = fs * 0.8
    is_ego = n0 == ego
    who = "EGO (camera car)" if is_ego else f"#{n0} {classes[n0] or ''}"

    def short(h):
        return head_names[h].split(":", 1)[-1]

    # Side column, top to bottom: identity + GT labels, now-list, predictions, timeline.
    info_h = int(line_h * 3.4)
    now_lines = review["panel_lines"]
    now_h = line_h * (now_lines + 1)
    pred_lines = review["pred_lines"] if threshold is not None else 0
    pred_h = line_h * (pred_lines + 1) if pred_lines else 0
    tl_h = max(H_img - info_h - now_h - pred_h, 4 * line_h)
    side_h = info_h + now_h + pred_h + tl_h
    rank = {"correct": 1, "group": 2, "false": 3, "unobservable": 4}

    def active(t):
        items = [("alarm", a) for a in result["alarms"] if a["start"] <= t <= a["end"]]
        items += [("event", e) for e in result["events"] if e["t"] <= t < e["t"] + show]
        return sorted(items, key=lambda it: 0 if it[0] == "event" else rank[it[1]["status"]])

    # The timeline of the focus agent: one row per head with an alarm or an event.
    rows = []
    for x in sorted(result["alarms"], key=lambda a: a["start"]) + sorted(result["events"], key=lambda e: e["t"]):
        if x["h"] not in rows:
            rows.append(x["h"])
    title_h, axis_h = line_h, int(line_h * 0.9)
    room = max((tl_h - title_h - axis_h) // max(int(line_h * 0.7), 1), 1)
    hidden = max(len(rows) - room, 0)
    rows = rows[:room]
    row_h = min(max((tl_h - title_h - axis_h) // max(len(rows), 1), int(line_h * 0.7)), line_h)
    timeline = np.full((tl_h, W_side, 3), COLORS["panel"], dtype=np.uint8)
    x0, x1 = int(W_side * 0.32), W_side - 12

    def tx(t):
        return int(x0 + (x1 - x0) * t / max(T - 1, 1))
    _text(cv2, timeline, "Timeline" + (f" (+{hidden} rows)" if hidden else "") + ("" if rows else ": no alarm or event"),
          8, int(title_h * 0.75), COLORS["text"], fs, bg=False, thick=2)
    for r, h in enumerate(rows):
        y = title_h + r * row_h
        _text(cv2, timeline, short(h)[:14], 8, y + row_h // 2 + int(8 * small), COLORS["text"], small, bg=False)
        cv2.line(timeline, (x0, y + row_h // 2), (x1, y + row_h // 2), (70, 70, 70), 1)
        split = y + (row_h * 3) // 5
        for a in result["alarms"]:
            if a["h"] == h:
                cv2.rectangle(timeline, (tx(a["start"]), y + 2), (max(tx(a["end"]), tx(a["start"]) + 2), split - 1),
                              COLORS[a["status"]], -1)
        for e in result["events"]:
            if e["h"] == h:
                c = COLORS[_event_key(e)]
                cv2.rectangle(timeline, (tx(e["t"]), split + 2), (max(tx(e["end"]), tx(e["t"]) + 2), split + 5), c, -1)
                cv2.line(timeline, (tx(e["t"]), y), (tx(e["t"]), y + row_h - 1), c, 3)
    axis_y = tl_h - axis_h + 4
    cv2.line(timeline, (x0, axis_y), (x1, axis_y), (120, 120, 120), 1)
    step = 5 if T / fps <= 60 else 10
    for sec in range(0, int(T / fps) + 1, step):
        x = tx(int(sec * fps))
        cv2.line(timeline, (x, axis_y), (x, axis_y + 5), (120, 120, 120), 1)
        _text(cv2, timeline, f"{sec}s", x - 10, axis_y + int(22 * small), (170, 170, 170), small * 0.8, bg=False)
    cursor_bottom = axis_y + 5

    # The prediction bars: 0 .. bar_max (the threshold in the middle at most, so small calibrated
    # thresholds stay visible), a tick at the threshold.
    bar_max = min(1.0, max(0.1, 2 * threshold)) if threshold is not None else 1.0
    bx0, bx1 = int(W_side * 0.42), W_side - 90

    def bx(v):
        return int(bx0 + (bx1 - bx0) * min(max(v / bar_max, 0.0), 1.0))

    W_out, H_out = W_img + W_side, header_h + max(H_img, side_h)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_out, H_out))
    sx, sy = W_img, H_img
    focus_color = (255, 255, 255)

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
        color = COLORS[items[0][1]["status"] if items[0][0] == "alarm" else _event_key(items[0][1])] if items else focus_color
        # Every other agent: a thin grey box; the focus agent: a thick box (white, or the colour of
        # its first item now); the ego (no box): a border around the whole frame.
        for n in range(N):
            pb = box_of(t, n)
            if pb is None or n == n0:
                continue
            cv2.rectangle(frame, pb[0], pb[1], COLORS["box"], 1)
        if is_ego:
            if items:
                cv2.rectangle(frame, (3, 3), (W_img - 4, H_img - 4), color, 8)
            _text(cv2, frame, "EGO", W_img // 2 - 30, H_img - 14, color, 1.0, thick=2)
        else:
            pb = box_of(t, n0)
            if pb is not None:
                cv2.rectangle(frame, pb[0], pb[1], color, 4)
                _text(cv2, frame, f"#{n0}", pb[0][0], pb[0][1] - 4, color, 0.8, thick=2)

        # Side column.
        info = np.full((info_h, W_side, 3), (40, 40, 40), dtype=np.uint8)
        _text(cv2, info, who, 8, int(line_h * 0.85), (255, 255, 255), fs * 1.15, bg=False, thick=2)
        labels = ",".join(label_names[a].split(":", 1)[-1] for a in torch.nonzero(X[t, n0, :L]).flatten().tolist())
        gt_lines = _wrap("GT: " + (labels or ("-" if is_ego or box_of(t, n0) else "(not in view)")),
                         int(W_side / (19 * small)))
        for i, s in enumerate(gt_lines[:2]):
            _text(cv2, info, s, 8, int(line_h * (1.75 + 0.8 * i)), (210, 210, 210), small, bg=False)

        now = np.full((now_h, W_side, 3), (22, 22, 22), dtype=np.uint8)
        n_alarms, n_events = sum(i[0] == "alarm" for i in items), sum(i[0] == "event" for i in items)
        _text(cv2, now, f"Now: {n_events} GT events" if threshold is None else f"Now: {n_alarms} alarms, {n_events} GT events",
              8, int(line_h * 0.75), COLORS["text"], fs, bg=False, thick=2)
        for i, (kind, x) in enumerate(items[:now_lines]):
            y = int(line_h * (1.75 + i))
            if kind == "alarm":
                c = COLORS[x["status"]]
                what = f"ALARM {short(x['h'])} p={float(score[t, n0, x['h']]):.2f} [{x['status']}]"
            else:
                c = COLORS[_event_key(x)]
                what = ("EVENT " if x["detected"] else ("MISSED (group) " if x["detected_group"] else "MISSED ")) + short(x["h"])
            _text(cv2, now, what, 8, y, c, fs, bg=False, thick=2)
        if len(items) > now_lines:
            _text(cv2, now, f"+{len(items) - now_lines}", W_side - 60, int(line_h * 0.75), COLORS["text"], small, bg=False)
        parts = [info, now]

        if pred_lines:
            pred = np.full((pred_h, W_side, 3), (30, 30, 30), dtype=np.uint8)
            _text(cv2, pred, f"Predicted now (event within {window_s:g} s)", 8, int(line_h * 0.75), COLORS["text"],
                  fs, bg=False, thick=2)
            top = sorted(heads, key=lambda h: -float(score[t, n0, h]))[:pred_lines]
            for i, h in enumerate(top):
                v = float(score[t, n0, h])
                y = int(line_h * (1 + i))
                on = v >= threshold
                c = (0, 165, 255) if on else (150, 150, 150)
                _text(cv2, pred, short(h)[:14], 8, y + int(line_h * 0.7), (255, 255, 255) if on else (190, 190, 190),
                      small, bg=False, thick=2 if on else 1)
                cv2.rectangle(pred, (bx0, y + int(line_h * 0.2)), (bx1, y + int(line_h * 0.8)), (55, 55, 55), -1)
                cv2.rectangle(pred, (bx0, y + int(line_h * 0.2)), (max(bx(v), bx0 + 1), y + int(line_h * 0.8)), c, -1)
                cv2.line(pred, (bx(threshold), y + int(line_h * 0.1)), (bx(threshold), y + int(line_h * 0.9)),
                         (255, 255, 255), 2)
                _text(cv2, pred, f"{v:.2f}", bx1 + 8, y + int(line_h * 0.7), c, small, bg=False, thick=2 if on else 1)
            parts.append(pred)

        tl = timeline.copy()
        cv2.line(tl, (tx(t), title_h - 2), (tx(t), cursor_bottom), (255, 255, 255), 2)
        parts.append(tl)
        side = np.vstack(parts)
        body = np.zeros((H_out - header_h, W_out, 3), dtype=np.uint8)
        body[:H_img, :W_img] = frame
        body[:side.shape[0], W_img:] = side
        header = np.full((header_h, W_out, 3), COLORS["panel"], dtype=np.uint8)
        if threshold is None:                                   # the ground truth only (`gt_only`)
            title = f"{video_id}  {t / fps:5.1f}s  |  {who}  |  ground truth only"
        else:
            title = (f"{video_id}  {t / fps:5.1f}s  |  {who}  |  {scorer_name} >= {threshold}  |  green correct  "
                     f"red false  blue caught  yellow missed  cyan group  grey unobservable")
        _text(cv2, header, title, 6, 24, COLORS["text"], 0.62, bg=False, thick=1)
        writer.write(np.vstack([header, body]))
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


def gt_only(result):
    """The `analyse_video` result reduced to the ground truth: no alarms, every GT event drawn as
    "EVENT" in blue (none is caught or missed without a scorer)."""
    return {**result, "score": torch.zeros_like(result["score"]), "alarms": [],
            "events": [{**e, "detected": True, "detected_group": False, "lead": None} for e in result["events"]]}


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


def _summary(name, video_id, threshold, result, agent=""):
    """review_summary.csv line of one video, focus agent and scorer: the alarm outcomes and the events caught."""
    alarms = [a["status"] for a in result["alarms"]]
    events = result["events"]
    seen = sum(s in ("correct", "group", "false") for s in alarms)
    return {"video": video_id, "agent": agent, "scorer": name, "threshold": threshold, "alarms": len(alarms),
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
        if any(r is None for r in results.values()):
            continue                                            # too short
        any_result = next(iter(results.values()))
        X_labels = any_result["X"][..., :L]
        ego = step05.video_agent_ids(atoms[video_id]).get(None)          # the camera car: no box
        focus = focus_agents(X_labels, ego, REVIEW["focus_others"])
        if not any(a["n"] in focus for r in results.values() for a in r["alarms"]):
            continue                                            # no scorer alarms on a focus agent
        vb, classes, _has_box = step05._agent_boxes(boxes_all, video_id, atoms[video_id], X_labels)
        classes = list(classes)
        if ego is not None:
            classes[ego] = "EGO"
        for n in focus:
            tag = "ego" if n == ego else f"agent{n}_{classes[n] or 'unknown'}"
            out_dir = output_dir / video_id / tag
            os.makedirs(out_dir, exist_ok=True)
            hs = agent_heads(X_labels, n, label_names, head_names)
            # The ground truth alone (gt.mp4): the agent's GT labels and events, no alarm.
            reader = FrameReader(video_id, frame_dir, video_dir)
            render_video(of_agent(gt_only(any_result), n), video_id, vb, classes, label_names, head_names,
                         out_dir / "gt.mp4", reader, "ground truth", None, window / fps, fps, REVIEW, ego, n, hs)
            reader.close()
            for name, result in results.items():
                agent_result = of_agent(result, n)
                reader = FrameReader(video_id, frame_dir, video_dir)    # read in order: one per video
                render_video(agent_result, video_id, vb, classes, label_names, head_names, out_dir / f"{name}.mp4",
                             reader, SCORER_NAMES.get(name, name), thresholds[name][0], window / fps, fps, REVIEW,
                             ego, n, hs)
                reader.close()
                lines += _lines(name, video_id, agent_result, classes, head_names, fps)
                summary.append(_summary(name, video_id, thresholds[name][0], agent_result, tag))
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
    print(f"[Review]\t {chosen} videos x (ego + {REVIEW['focus_others']} agents) x {len(thresholds)} scorers in "
          f"{output_dir}; {len(lines)} lines in {done}")


def _readme(cfg, fps, thresholds):
    scorers = "\n".join(f"  {name}.mp4: {SCORER_NAMES.get(name, name)}, alarm threshold {th}"
                        + (f" (episode recall {recall:.2f})" if recall is not None else "")
                        for name, (th, recall) in thresholds.items())
    return f"""Review videos (step06_review.py), frame rate {fps}: one folder per test video, in it one folder per
reviewed agent: ego/ (the camera car) and agent<n>_<class>/ for the {REVIEW['focus_others']} other agents seen in the
most frames. Each agent folder holds one MP4 per scorer, each scorer at the threshold where its episode
recall over the whole test split is closest to {REVIEW['target_recall']} (so the scorers catch about as many
events and their alarms can be compared):
{scorers}
  gt.mp4: the ground truth only (no model): the agent's GT labels and events.
review_summary.csv: per video, agent and scorer, the alarms (correct / group / false / unobservable,
precision) and the GT events (caught / missed).

Window: "the event starts within {cfg['headline_window'] / fps:g} s". Alarm: the frames where the score of an agent
and head is >= the threshold, merged into one alarm (gaps of 1 frame bridged, >= 0.5 s long).
Left: the camera frame; the reviewed agent in a thick box (white, or the colour of its current alarm /
event), the others thin grey; for the ego (no box) a border around the frame while it has an alarm or event.
Colours: green = correct alarm (the event follows on this agent within the alarm + {REVIEW['tolerance_s']} s),
cyan = the event follows on an adjacent agent of the same class only, red = false alarm, grey = the
agent left too soon to tell. EVENT (blue) = a GT event that an alarm caught, MISSED (yellow) = one
that none did, MISSED (group) (cyan) = one caught only by an adjacent same-class agent's alarm.
Right, only about the reviewed agent: who it is and its current GT labels; its alarms and GT events now;
its scores now for the heads of its class, the {REVIEW['pred_lines']} highest, as bars (orange: at or above the
threshold, the white tick); its timeline over the whole video, one row per head with an alarm or event:
the thick bar is the alarm (start to end), the thin bar below it the GT event (onset to end, a tick at
the onset); the white line is now.

review.csv: one line per alarm and per GT event of the reviewed agents, per scorer. Fill in "verdict" and "comment". Suggested verdicts:
  alarm:  reasonable (the situation makes the event likely; it just did not happen) /
          wrong_agent (the event happened to another agent) / early_or_late (right event, timing off) /
          label_error (the event did happen, the GT misses it) / no_reason (nothing suggests it);
  event:  predictable (the video shows signs before the onset) / unpredictable / label_error.
"""
