"""Step 08: event clips, short videos around single test events, for a human review of predictability.

Question of the review: is an event foreseeable from what the camera shows 2 s before it (and are
the models right or wrong about it)? Per head with >= CLIPS["min_test_events"] test events (every
counted head of step 07, the "unpredictable" ones too), CLIPS["per_head"] of its test events, half
caught by at least one scorer at the headline false-alarm budget and half missed by all of them, each
from CLIPS["pre_s"] before its onset to CLIPS["post_s"] after it; and the CLIPS["false_per_scorer"]
strongest false alarms of the rule net and of the MLP on that head (an alarm with no event of the head
on the agent), from 2 s before the alarm to 2 s after it.
The scorers and thresholds are those of the paper tables (step 07, paper_metrics.json): the
calibrated rule net, MLP and own-label lookup at CLIPS["budget"] false alarms per minute (threshold on
val); caught / missed and the lead are read from paper_events.csv, so they match the tables.

Clip layout: the camera frame (left) with the reviewed agent in a thick box (white, orange in the
2 s before the onset, blue from the onset on; the ego, without a box: a border) and the other agents
in thin grey boxes; the side column (right):
  - the head, the kind of clip (caught / missed / false alarm), the agent, each scorer's outcome;
  - a countdown to the onset (orange inside the 2 s prediction window);
  - the agent's GT labels now and the ego's;
  - the window-2 s scores of the three scorers over the clip (one colour each, its threshold dashed,
    its alarms as a bar under the plot), the 2 s window before the onset shaded, the onset as a blue line;
  - the GT label timeline of the agent (and of the ego) over the clip: what the models see.
Played at CLIPS["playback"] x real time.

Output (step06_output/clips/): <head>/<kind>_<video>_f<frame>_a<agent>.mp4, clips.csv (one line per
clip, with empty columns for the reviewer: predictable, cue, label_ok, comment), README.txt; and
<exp>/packs/clips_<run>.tar.gz of the whole folder to download.
"""
import csv
import heapq
import os
import random
import re
import tarfile
from collections import defaultdict

import torch
from tqdm import tqdm

from src.exp_roadpp import step05, step06, step06_episodes, step06_review, step07_paper, utils_data

CLIPS = {
    "budget": 5,                              # false alarms per minute: the paper tables' headline budget
    "scorers": ("model_cal", "mlp_cal", "lookup_cal"),
    "false_from": ("model_cal", "mlp_cal"),   # false-alarm clips of these scorers
    "min_test_events": 5,                     # heads with at least this many test events
    "per_head": 10,                           # event clips per head: half caught (by any scorer), half missed
    "false_per_scorer": 2,                    # false-alarm clips per head and scorer (highest peak first)
    "pre_s": 4.0,                             # an event clip starts this long before the onset
    "post_s": 1.5,                            # and ends this long after it
    "false_pre_s": 2.0, "false_post_s": 2.0,  # a false-alarm clip: from before the alarm to after its end
    "false_max_s": 4.0,                       # (at most this much of a long alarm)
    "playback": 0.5,                          # playback speed (0.5: half real time)
    "width": 960,                             # camera frame width, pixels
    "side_width": 640,
    "label_rows": 8,                          # label timeline rows of the agent (+ 4 of the ego)
    "seed": 0,
    "version": "clips_v1",
}
NAMES = {"model_cal": "rule net", "mlp_cal": "MLP", "lookup_cal": "lookup"}
COLOR = {"model_cal": (0, 140, 255), "mlp_cal": (255, 80, 255), "lookup_cal": (255, 210, 0)}     # BGR
C_WINDOW, C_EVENT, C_MISS, C_FALSE = (0, 165, 255), (255, 150, 40), (0, 230, 255), (0, 0, 255)
WHITE, GREY = (255, 255, 255), (160, 160, 160)


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s)


def _caught_by(row, budget):
    """{scorer: lead s or None} of one paper_events.csv row."""
    out = {}
    for s in CLIPS["scorers"]:
        v = row.get(f"{s}@{budget}")
        out[s] = float(v) if v not in (None, "") else None
    return out


def select_events(rows, budget, rng):
    """The event clips: per head with >= min_test_events test events, per_head events, half caught by
    some scorer and half missed by all (the other kind fills up if one is short)."""
    by_head = defaultdict(list)
    for r in rows:
        by_head[r["label"]].append(r)
    chosen = []
    for label in sorted(by_head):
        rs = by_head[label]
        if len(rs) < CLIPS["min_test_events"]:
            continue
        caught = [r for r in rs if any(v is not None for v in _caught_by(r, budget).values())]
        missed = [r for r in rs if not any(v is not None for v in _caught_by(r, budget).values())]
        rng.shuffle(caught)
        rng.shuffle(missed)
        n = CLIPS["per_head"]
        nc = min(len(caught), max(n // 2, n - len(missed)))
        nm = min(len(missed), n - nc)
        chosen += [{**r, "kind": "caught"} for r in caught[:nc]] + [{**r, "kind": "missed"} for r in missed[:nm]]
    return chosen, {label: len(rs) for label, rs in by_head.items()}


def _alarms(v, name, th, H, tol, ep):
    """Alarm episodes of one scorer in one video: pair (n * H + h), start, end, hit (an event of the
    pair follows), seen (the outcome is observable)."""
    T, N = v["T"], v["N"]
    score = v["dense"][name].reshape(T, N * H)
    row_p = v["is_row"][:, :, None].expand(T, N, H).reshape(T, N * H)
    risk_p = v["at_risk"].reshape(T, N * H)
    pair, start, end = step06_episodes.alarm_episodes(score, row_p, risk_p, th, ep["fill_gap"], ep["min_length"])
    e_t, e_n, e_h = v["events"]
    if len(start) == 0:
        return pair, start, end, torch.zeros(0, dtype=torch.bool), torch.zeros(0, dtype=torch.bool)
    hit = (((e_n * H + e_h)[:, None] == pair[None]) & (start[None] <= e_t[:, None])
           & (e_t[:, None] <= end[None] + tol)).any(dim=0)
    s_n = pair // H
    nxt = v["next_gone"]
    seen = (nxt[(end + 1).clamp(max=T), s_n] > end + max(tol, 1)) | (nxt[start, s_n] > start + v["w"] - 1)
    return pair, start, end, hit, seen


def _clip_payload(v, alarms, n, h, H, t0, t1):
    """The scores (per scorer, frames t0..t1) of agent n and head h, and its alarm episodes in the clip."""
    out = {"scores": {s: v["dense"][s][t0:t1 + 1, n, h].cpu() for s in CLIPS["scorers"]}, "alarms": {}}
    for s, (pair, start, end, _hit, _seen) in alarms.items():
        sel = (pair == n * H + h) & (end >= t0) & (start <= t1)
        out["alarms"][s] = [(int(a), int(b)) for a, b in zip(start[sel].tolist(), end[sel].tolist())]
    return out


# -------------- Drawing --------------
def render_clip(spec, X, L, label_names, boxes, classes, ego, reader, fps, window, out_path):
    """Write one clip MP4 (layout: module docstring)."""
    import cv2
    import numpy as np
    _text = step06_review._text
    t0, t1, n0, onset = spec["t0"], spec["t1"], spec["n"], spec.get("onset")
    F = t1 - t0 + 1
    W_img = CLIPS["width"]
    H_img = int(round(W_img * 1280 / 1920))
    W_side, header_h = CLIPS["side_width"], 30
    is_ego = n0 == ego
    who = "EGO (camera car)" if is_ego else f"agent #{n0} {classes[n0] or ''}"
    short = (lambda a: label_names[a].split(":", 1)[-1])

    # Label timeline rows: the agent's labels held in the clip (the head first), then the ego's.
    def held(n, limit):
        if n is None:
            return []
        on = X[t0:t1 + 1, n, :L].any(dim=0)
        idx = torch.nonzero(on).flatten().tolist()
        first = {a: int(torch.nonzero(X[t0:t1 + 1, n, a])[0]) for a in idx}
        idx.sort(key=lambda a: (a != spec["head_atom"], first[a]))
        return [(n, a) for a in idx[:limit]]
    rows = held(n0, CLIPS["label_rows"]) + ([] if is_ego else [(ego, a) for _n, a in held(ego, 4)])

    info_h, cd_h, gt_h, plot_h = 140, 50, 64, 210
    tl_h = max(H_img - info_h - cd_h - gt_h - plot_h, 22 * (len(rows) + 2))
    side_h = info_h + cd_h + gt_h + plot_h + tl_h
    W_out, H_out = W_img + W_side, header_h + max(H_img, side_h)
    out_fps = max(fps * CLIPS["playback"], 1)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W_out, H_out))
    px0, px1 = 112, W_side - 14

    def px(t):
        return int(px0 + (px1 - px0) * (t - t0) / max(F - 1, 1))

    # The static score plot.
    ths = {s: spec["thresholds"][s] for s in CLIPS["scorers"]}
    vmax = max([float(spec["scores"][s].max()) for s in CLIPS["scorers"]] + list(ths.values()) + [0.02])
    ymax = min(1.0, vmax * 1.15)
    p_top, p_bot = 34, plot_h - 20 - 6 * len(CLIPS["scorers"])

    def py(val):
        return int(p_bot - (p_bot - p_top) * min(max(val / ymax, 0.0), 1.0))
    plot = np.full((plot_h, W_side, 3), (26, 26, 26), dtype=np.uint8)
    if onset is not None:
        cv2.rectangle(plot, (px(max(onset - window, t0)), p_top), (px(onset), p_bot), (20, 60, 100), -1)
    if spec.get("false_alarm"):
        a, b = spec["false_alarm"]
        cv2.rectangle(plot, (px(max(a, t0)), p_top), (px(min(b, t1)), p_bot), (30, 30, 90), -1)
    cv2.line(plot, (px0, p_bot), (px1, p_bot), GREY, 1)
    for frac in (0.5, 1.0):
        _text(cv2, plot, f"{ymax * frac:.2f}", px0 - 44, py(ymax * frac) + 5, GREY, 0.4, bg=False)
        cv2.line(plot, (px0 - 3, py(ymax * frac)), (px1, py(ymax * frac)), (55, 55, 55), 1)
    for i, s in enumerate(CLIPS["scorers"]):
        y = py(ths[s])
        for x in range(px0, px1, 12):
            cv2.line(plot, (x, y), (min(x + 6, px1), y), COLOR[s], 1)
        vals = spec["scores"][s].tolist()
        pts = np.array([[px(t0 + j), py(val)] for j, val in enumerate(vals)], dtype=np.int32)
        cv2.polylines(plot, [pts], False, COLOR[s], 2, cv2.LINE_AA)
        yb = p_bot + 6 + 6 * i
        for a, b in spec["alarms"].get(s, []):
            cv2.rectangle(plot, (px(max(a, t0)), yb), (px(min(b, t1)), yb + 4), COLOR[s], -1)
    if onset is not None:
        cv2.line(plot, (px(onset), p_top - 4), (px(onset), p_bot), C_EVENT, 2)
    _text(cv2, plot, f"P({spec['label'].split(':', 1)[-1]} within {window / fps:g} s)", 8, 20, WHITE, 0.55,
          bg=False, thick=1)
    x = 290
    for s in CLIPS["scorers"]:
        _text(cv2, plot, NAMES[s], x, 20, COLOR[s], 0.55, bg=False, thick=2)
        x += 20 + 11 * len(NAMES[s])
    _text(cv2, plot, "dashed: threshold, bars: alarms", 8, plot_h - 2, GREY, 0.38, bg=False)

    # The static label timeline.
    tl = np.full((tl_h, W_side, 3), (34, 34, 34), dtype=np.uint8)
    _text(cv2, tl, "GT labels over the clip (what the models see)", 8, 18, WHITE, 0.5, bg=False)
    row_h = 22
    for r, (n, a) in enumerate(rows):
        y = 30 + r * row_h
        name = ("ego " if (n == ego and not is_ego) else "") + short(a)
        c = C_EVENT if (n == n0 and a == spec["head_atom"]) else (210, 210, 210)
        _text(cv2, tl, name[:13], 6, y + 14, c, 0.45, bg=False)
        on = X[t0:t1 + 1, n, a].tolist()
        for j, o in enumerate(on):
            if o:
                cv2.rectangle(tl, (px(t0 + j), y + 4), (px(t0 + j) + max((px1 - px0) // max(F - 1, 1), 1), y + 16),
                              c if c == C_EVENT else (140, 140, 140), -1)
    if onset is not None:
        cv2.line(tl, (px(onset), 24), (px(onset), tl_h - 4), C_EVENT, 1)

    # Info block: the head, the clip kind, the outcomes.
    info = np.full((info_h, W_side, 3), (44, 44, 44), dtype=np.uint8)
    _text(cv2, info, spec["label"], 8, 30, WHITE, 0.9, bg=False, thick=2)
    kind_text = {"caught": ("EVENT, caught by a model", C_EVENT), "missed": ("EVENT, missed by every model", C_MISS),
                 "false": (f"FALSE ALARM of the {NAMES.get(spec.get('scorer'), '')} (no event follows)", C_FALSE)}
    s, c = kind_text[spec["kind"]]
    _text(cv2, info, s, 8, 58, c, 0.6, bg=False, thick=2)
    _text(cv2, info, f"{who}  |  {spec['video']}  frame {spec.get('onset') or spec['false_alarm'][0]}",
          8, 84, (210, 210, 210), 0.48, bg=False)
    x = 8
    for s in CLIPS["scorers"]:
        lead = spec["outcome"].get(s)
        txt = f"{NAMES[s]}: " + ("-" if spec["kind"] == "false" else
                                  (f"{lead:.1f}s ahead" if lead is not None else "missed"))
        _text(cv2, info, txt, x, 112, COLOR[s], 0.5, bg=False, thick=1)
        x += 24 + 9 * len(txt)
    _text(cv2, info, f"thresholds @ {CLIPS['budget']} FA/min: " + ", ".join(
        f"{NAMES[s]} {ths[s]:.3g}" for s in CLIPS["scorers"]), 8, 132, GREY, 0.42, bg=False)

    for t in range(t0, t1 + 1):
        img = reader.read(t)
        frame = (np.zeros((H_img, W_img, 3), dtype=np.uint8) if img is None
                 else cv2.resize(img, (W_img, H_img), interpolation=cv2.INTER_AREA))
        if img is None:
            _text(cv2, frame, "no camera frame", 10, 30, WHITE)
        if onset is None:
            a, b = spec["false_alarm"]
            color = C_FALSE if a <= t <= b else WHITE
        else:
            color = C_EVENT if t >= onset else (C_WINDOW if t >= onset - window else WHITE)
        for n in range(boxes.shape[1]):
            bb = boxes[t, n]
            if n == n0 or torch.isnan(bb).any():
                continue
            cv2.rectangle(frame, (int(bb[0] * W_img), int(bb[1] * H_img)), (int(bb[2] * W_img), int(bb[3] * H_img)),
                          (190, 190, 190), 1)
        if is_ego:
            cv2.rectangle(frame, (3, 3), (W_img - 4, H_img - 4), color, 6)
        else:
            bb = boxes[t, n0]
            if not torch.isnan(bb).any():
                p0, p1 = (int(bb[0] * W_img), int(bb[1] * H_img)), (int(bb[2] * W_img), int(bb[3] * H_img))
                cv2.rectangle(frame, p0, p1, color, 4)
                _text(cv2, frame, f"#{n0}", p0[0], p0[1] - 4, color, 0.7, thick=2)
            else:
                _text(cv2, frame, f"#{n0} not in view", 10, H_img - 14, color, 0.7, thick=2)

        cd = np.full((cd_h, W_side, 3), (20, 20, 20), dtype=np.uint8)
        if onset is not None:
            dt = (t - onset) / fps
            if dt < 0:
                inside = t >= onset - window
                _text(cv2, cd, f"onset in {-dt:.1f} s" + ("   (2 s prediction window)" if inside else ""), 8, 34,
                      C_WINDOW if inside else WHITE, 0.8, bg=False, thick=2)
            else:
                _text(cv2, cd, f"EVENT {spec['label'].split(':', 1)[-1]}  +{dt:.1f} s", 8, 34, C_EVENT, 0.8,
                      bg=False, thick=2)
        else:
            a, b = spec["false_alarm"]
            _text(cv2, cd, f"alarm {'ON' if a <= t <= b else 'off'}   {(t - a) / fps:+.1f} s from its start", 8, 34,
                  C_FALSE if a <= t <= b else WHITE, 0.8, bg=False, thick=2)

        gt = np.full((gt_h, W_side, 3), (36, 36, 36), dtype=np.uint8)
        labels = ",".join(short(a) for a in torch.nonzero(X[t, n0, :L]).flatten().tolist()) or "-"
        _text(cv2, gt, ("GT ego: " if is_ego else "GT agent: ") + labels[:70], 8, 24, (220, 220, 220), 0.5, bg=False)
        if not is_ego and ego is not None:
            e_labels = ",".join(short(a) for a in torch.nonzero(X[t, ego, :L]).flatten().tolist()) or "-"
            _text(cv2, gt, "GT ego: " + e_labels[:70], 8, 50, (180, 180, 180), 0.5, bg=False)

        p, tlc = plot.copy(), tl.copy()
        for img_, top in ((p, p_top - 4), (tlc, 24)):
            cv2.line(img_, (px(t), top), (px(t), img_.shape[0] - 4 if img_ is tlc else p_bot), WHITE, 1)
        for s in CLIPS["scorers"]:
            val = float(spec["scores"][s][t - t0])
            cv2.circle(p, (px(t), py(val)), 4, COLOR[s], -1)
        side = np.vstack([info, cd, gt, p, tlc])
        body = np.zeros((H_out - header_h, W_out, 3), dtype=np.uint8)
        body[:H_img, :W_img] = frame
        body[:side.shape[0], W_img:] = side
        header = np.full((header_h, W_out, 3), (30, 30, 30), dtype=np.uint8)
        _text(cv2, header, f"{spec['video']}  {t / fps:6.1f}s  |  {spec['label']}  |  {spec['kind']}  |  "
                           f"playback {CLIPS['playback']:g}x", 6, 21, WHITE, 0.55, bg=False)
        writer.write(np.vstack([header, body]))
    writer.release()


# -------------- Main --------------
def main(input_data):
    print("\n--------- Step 08: event clips for the predictability review ----------------------\n")
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[Clips]\t skipped: opencv (cv2) is not installed")
        return
    out_dir = input_data["output_dir"]
    clip_dir = out_dir / "clips"
    metrics_file, events_file = out_dir / "paper_metrics.json", out_dir / "paper_events.csv"
    if not (os.path.exists(metrics_file) and os.path.exists(events_file)):
        print("[Clips]\t skipped: run step 07 first (paper_metrics.json, paper_events.csv)")
        return
    frame_dir, video_dir = input_data.get("frame_dir"), input_data.get("video_dir")
    if not ((frame_dir and os.path.isdir(frame_dir)) or (video_dir and os.path.isdir(video_dir))):
        print(f"[Clips]\t skipped: neither the frame dir {frame_dir} nor the video dir {video_dir} exists")
        return
    settings_file = clip_dir / "clips_settings.json"
    if (os.path.exists(clip_dir / "clips.csv") and os.path.exists(settings_file)
            and utils_data.load_json(settings_file) == CLIPS
            and step05.is_fresh(clip_dir / "clips.csv", metrics_file, events_file)):
        print(f"[Clips]\t up to date: {clip_dir}")
        return
    os.makedirs(clip_dir, exist_ok=True)
    step05_cfg = step05.get_cfg(input_data["step05_input"])
    device = step05_cfg["device"] = input_data["device"]
    fps = input_data["frame_rate"]
    paper = utils_data.load_json(metrics_file)
    budget = str(CLIPS["budget"])
    thresholds = {s: paper["scorers"][s]["budgets"][budget]["threshold"] for s in CLIPS["scorers"]}
    if any(th is None for th in thresholds.values()):
        raise ValueError(f"Clips: no threshold at {budget} FA/min for {thresholds}")
    print("[Clips]\t thresholds @ %s FA/min: %s" % (budget, thresholds))

    # The counted heads of step 07 (its per-head AP keys) as atom indices.
    vocab = step05.load_vocabulary({**step05_cfg, "unique_atoms_file": input_data["unique_atoms_file"]})
    names_all = [step05._atom_name(s) for s in vocab]
    head_labels = list(paper["scorers"]["model"]["ap"]["per_head"])
    heads = sorted(names_all.index(lb) for lb in head_labels)
    head_names = [names_all[h] for h in heads]
    H = len(heads)
    heads_d = torch.tensor(heads, device=device)

    with open(events_file) as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(CLIPS["seed"])
    chosen, counts = select_events(rows, budget, rng)
    picked_heads = {r["label"] for r in chosen}
    print(f"[Clips]\t {len(chosen)} event clips of {len(picked_heads)} heads with >= {CLIPS['min_test_events']} "
          f"test events")
    by_video = defaultdict(list)
    for r in chosen:
        by_video[r["video"]].append(r)

    test_file = step05.shared_data_file(out_dir / "test_data.pt", step05_cfg, "test")
    test_data = torch.load(test_file, map_location="cpu")
    model = step05.load_temporal_rule_net(test_data, step05_cfg)
    mlp = step05.load_onset_mlp(test_data, step05_cfg)
    baselines = torch.load(out_dir / "baselines.pt", map_location="cpu", weights_only=False)
    calibration = torch.load(out_dir / "calibration.pt", map_location="cpu")
    bases = [s[:-4] for s in CLIPS["scorers"]]
    scorers = step06._scorers(model, baselines, mlp, None, device, names=bases)
    calibration = {b: calibration[b] for b in bases}
    episodes = out_dir / "episode_alarms.json"
    min_context = (paper.get("min_context") or utils_data.load_json(episodes)["min_context"])
    w = paper["window"]
    k = model.windows.index(w)
    cfg = {"device": device, "frame_rate": fps, "min_context": min_context,
           "frames_per_video": step05_cfg["frames_per_video"], "window": w}
    ep = step06_episodes._default_cfg(fps)
    tol = int(round(step07_paper.PAPER["tolerance_s"] * fps))
    pre, post = int(round(CLIPS["pre_s"] * fps)), int(round(CLIPS["post_s"] * fps))

    # 1. Score every test video: the event clips' scores and the false-alarm candidates.
    specs = []
    false_heap = defaultdict(list)                     # (head, scorer) -> heap of (peak, i, spec)
    counter = 0
    for video_id in tqdm(sorted(test_data["videos"]), desc="Clips (scoring)"):
        v = step07_paper.video_scores(model, scorers, calibration, test_data["videos"][video_id], test_data,
                                      heads_d, k, cfg)
        if v is None:
            continue
        v["w"] = w
        alarms = {s: _alarms(v, s, thresholds[s], H, tol, ep) for s in CLIPS["scorers"]}
        T = v["T"]
        for r in by_video.get(video_id, []):
            n, onset, atom = int(r["agent"]), int(r["frame"]), int(r["head"])
            h = heads.index(atom)
            t0, t1 = max(0, onset - pre), min(T - 1, onset + post)
            specs.append({"video": video_id, "n": n, "label": r["label"], "head_atom": atom, "kind": r["kind"],
                          "onset": onset, "t0": t0, "t1": t1, "type": r["type"],
                          "outcome": _caught_by(r, budget), "thresholds": thresholds,
                          **_clip_payload(v, alarms, n, h, H, t0, t1)})
        for s in CLIPS["false_from"]:
            pair, start, end, hit, seen = alarms[s]
            for i in torch.nonzero(~hit & seen).flatten().tolist():
                n, h = int(pair[i]) // H, int(pair[i]) % H
                if head_names[h] not in picked_heads:
                    continue
                a, b = int(start[i]), int(end[i])
                peak = float(v["dense"][s][a:b + 1, n, h].max())
                heap = false_heap[(head_names[h], s)]
                if len(heap) >= CLIPS["false_per_scorer"] and peak <= heap[0][0]:
                    continue
                t0 = max(0, a - int(round(CLIPS["false_pre_s"] * fps)))
                t1 = min(T - 1, min(b, a + int(round(CLIPS["false_max_s"] * fps))) + int(round(CLIPS["false_post_s"] * fps)))
                spec = {"video": video_id, "n": n, "label": head_names[h], "head_atom": heads[h], "kind": "false",
                        "scorer": s, "onset": None, "false_alarm": (a, b), "t0": t0, "t1": t1, "peak": peak,
                        "outcome": {}, "thresholds": thresholds, **_clip_payload(v, alarms, n, h, H, t0, t1)}
                counter += 1
                item = (peak, counter, spec)
                if len(heap) < CLIPS["false_per_scorer"]:
                    heapq.heappush(heap, item)
                else:
                    heapq.heapreplace(heap, item)
        del v
    specs += [spec for heap in false_heap.values() for _p, _c, spec in sorted(heap, reverse=True)]
    print(f"[Clips]\t {len(specs)} clips ({sum(s['kind'] == 'false' for s in specs)} false alarms)")

    # 2. Render, video by video (the frames read in order).
    atoms = utils_data.load_json(input_data["test_atoms_file"])
    box_file = input_data.get("test_box_file")
    boxes_all = torch.load(box_file, map_location="cpu") if box_file and os.path.exists(box_file) else {}
    L = test_data.get("num_label_atoms", test_data["num_atoms"])
    label_names = names_all[:L]
    lines = []
    specs_by_video = defaultdict(list)
    for s in specs:
        specs_by_video[s["video"]].append(s)
    for video_id in tqdm(sorted(specs_by_video), desc="Clips (rendering)"):
        X = step05._video_from_cells(test_data["videos"][video_id], test_data["num_atoms"], "cpu")
        X_labels = X[..., :L]
        ego = step05.video_agent_ids(atoms[video_id]).get(None)
        vb, classes, _has = step05._agent_boxes(boxes_all, video_id, atoms[video_id], X_labels)
        classes = list(classes)
        if ego is not None:
            classes[ego] = "EGO"
        reader, last = None, None
        for s in sorted(specs_by_video[video_id], key=lambda s: s["t0"]):
            if reader is None or last is None or s["t0"] <= last:
                if reader is not None:
                    reader.close()
                reader = step06_review.FrameReader(video_id, frame_dir, video_dir)
            head_dir = clip_dir / _safe(s["label"])
            os.makedirs(head_dir, exist_ok=True)
            frame = s["onset"] if s["onset"] is not None else s["false_alarm"][0]
            kind = s["kind"] if s["kind"] != "false" else f"false_{NAMES[s['scorer']].replace(' ', '')}"
            name = f"{kind}_{video_id}_f{frame:05d}_a{s['n']}.mp4"
            render_clip(s, X, L, label_names, vb, classes, ego, reader, fps, w, head_dir / name)
            last = s["t1"]
            lines.append({"clip": f"{_safe(s['label'])}/{name}", "head": s["label"],
                          "head_test_events": counts.get(s["label"], ""), "kind": kind, "video": video_id,
                          "agent": s["n"], "class": classes[s["n"]] or "", "frame": frame,
                          "time_s": round(frame / fps, 1),
                          **{f"lead_{NAMES[c].replace(' ', '_')}": ("" if s["outcome"].get(c) is None
                                                                    else s["outcome"][c]) for c in CLIPS["scorers"]},
                          **{f"max_p_{NAMES[c].replace(' ', '_')}": round(float(s["scores"][c].max()), 3)
                             for c in CLIPS["scorers"]},
                          "predictable": "", "cue": "", "label_ok": "", "comment": ""})
        if reader is not None:
            reader.close()
    lines.sort(key=lambda r: (r["head"], r["kind"], r["video"], r["frame"]))
    with open(clip_dir / "clips.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(lines[0]) if lines else ["clip"])
        writer.writeheader()
        writer.writerows(lines)
    (clip_dir / "README.txt").write_text(_readme(fps, thresholds))
    utils_data.save_json(CLIPS, settings_file)
    print(f"[Clips]\t {len(lines)} clips in {clip_dir}, review sheet {clip_dir / 'clips.csv'}")

    # 3. One archive to download: <exp>/packs/clips_<run>.tar.gz.
    run_dir = out_dir.parent
    pack_dir = run_dir.parent.parent / "packs"
    os.makedirs(pack_dir, exist_ok=True)
    pack = pack_dir / f"clips_{run_dir.name}.tar.gz"
    with tarfile.open(pack, "w:gz") as tar:
        tar.add(clip_dir, arcname=f"clips_{run_dir.name}")
    print(f"[Clips]\t packed: {pack} ({os.path.getsize(pack) / 1e6:.0f} MB)")
    print("\n--------- Step 08 Done ---------------\n")


def _readme(fps, thresholds):
    ths = ", ".join(f"{NAMES[s]} {th:.3g}" for s, th in thresholds.items())
    return f"""Event clips (step08_clips.py) for a human review: is the event foreseeable from the video 2 s ahead?

Folders: one per head (every head with >= {CLIPS['min_test_events']} test events). In each:
  caught_*.mp4   a test event that at least one model caught ({CLIPS['budget']} false alarms / min, thresholds on val: {ths})
  missed_*.mp4   a test event that every model missed
  false_rulenet_*.mp4 / false_MLP_*.mp4   the strongest false alarms of that model on the head (no event follows)
Up to {CLIPS['per_head']} event clips per head (half caught, half missed) and {CLIPS['false_per_scorer']} false alarms per model.
Event clips: {CLIPS['pre_s']:g} s before the onset to {CLIPS['post_s']:g} s after. Playback {CLIPS['playback']:g}x real time ({fps} fps data).

Left: the camera frame. The reviewed agent in a thick box: white, ORANGE in the 2 s before the onset (the
prediction window: an alarm here catches the event), BLUE from the onset on; for a false alarm RED while
the alarm is on. The ego (camera car) has no box: a border instead. Other agents: thin grey boxes.
Right: head and clip kind; each model's outcome (caught n s ahead / missed); a countdown to the onset; the
GT labels now of the agent and the ego; the predicted probability of the event within 2 s of the three
models over the clip (dashed: each model's threshold; the bars below: its alarms; shaded: the 2 s window
before the onset; blue line: the onset); the GT label timeline of the agent (and the ego) over the clip:
the symbolic input the models see.

clips.csv: one line per clip. Fill in:
  predictable: yes  (the video shows clear signs before the onset, e.g. brake lights, slowing, a red light, a car ahead stopping)
               weak (some hint, but ambiguous)
               no   (nothing foreseeable 2 s ahead: a decision of the driver, a sudden event)
  cue:         what foreshadows it, if anything (free text), and whether the cue is in the GT labels (the timeline) or only in the image
  label_ok:    yes / no (the onset is wrong, too early / late, or the event did not happen) / unsure
  comment
For false alarms: predictable = was an event of this kind plausible here (yes: the model's guess is reasonable).
"""
