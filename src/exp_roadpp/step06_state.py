"""Step 06 (state): evaluate the StateMLP (`step05_state`), the label state of every agent h frames ahead.

Every frame of every test video (from min_context on, so the memory is full) and every agent visible
now gets a prediction for every GT label atom: is it on h frames after now? Scored where the agent is
still visible then. The (row, head) cells fall in four kinds, by the state now and h frames later:
  stay-on, stay-off (persistence gets these right), onset (off -> on), offset (on -> off).
Since most labels keep their state, the overall state recall / precision look high even for
persistence; the onset and offset cells measure the forecast.

Scorers: "state_mlp"; "markov" (the train rate of on in t + h given the state now, per head: knows
only the state now); "persist" (the state now copied forward: never predicts a change).
Operating points, tuned on the val videos (micro best F1 over the counted heads): an agent's label
that is off now is predicted on if p >= onset threshold; one that is on now is predicted off if
p <= offset threshold. The persistence never predicts a change. Reported per scorer and horizon
(micro over the counted heads, per event type and per head):
  state: AP of p against the state in t + h, recall / precision / F1 of the predicted state;
  onset: AP of p over the off-now cells, recall / precision at the onset threshold;
  offset: AP of 1 - p over the on-now cells, recall / precision at the offset threshold;
  stay_on_kept / stay_off_kept: share of the unchanged cells predicted unchanged;
  exact: share of the rows (agent, frame) whose every counted label is predicted right, of all rows
      and of the rows with some change.
Output: state_metrics.json, a printed summary, and review videos in state_review/ (`render_video`):
the review layout of `step06_review` with the predicted state h frames ahead instead of alarms.
"""
import csv
import os
import random
import torch
from tqdm import tqdm

from src.exp_roadpp import step05, step05_state, step06, utils_data
from src.exp_roadpp.step06_review import COLORS, FrameReader, _text, _wrap, agent_heads, focus_agents

NB = 1000                                   # score bins of [0, 1]
SCORERS = ("state_mlp", "markov", "persist")
STATE = {
    "headline_horizon": 20,
    "min_changes": 5,                       # per-head metrics only for heads with >= this many onsets or offsets
    # Review videos.
    "num_videos": 10,
    "seed": 0,
    "focus_others": 2,
    "width": 1280,
    "side_width": 720,
    "font_scale": 0.75,
    "panel_lines": 4,
    "pred_lines": 6,
    "version": "state_review_v1",
}
STATE_COLORS = {"correct": (0, 210, 0), "false": (0, 0, 255), "missed": (0, 230, 255), "caught": (255, 150, 40),
                "unknown": (150, 150, 150), "pred_on": (255, 0, 255)}


# -------------- Scores --------------
def _scores(model, rows, markov):
    """{scorer: (B, K, L) probability of on in t - 1 + h} of one rows dict."""
    now = rows["now"]
    m = markov.to(now.device)                                                  # (K, 2, L)
    return {"state_mlp": torch.sigmoid(model(rows["inputs"], now)),
            "markov": torch.where(now[:, None, :] > 0.5, m[None, :, 1], m[None, :, 0]),
            "persist": now[:, None, :].expand(-1, m.shape[0], -1)}


@torch.no_grad()
def video_rows(model, video, num_atoms, num_labels, min_context, chunk, device):
    """All rows of one video (now = min_context - 1 .. T - 1), concatenated, or None."""
    X = step05._video_from_cells(video, num_atoms, device)
    T = X.shape[0]
    roles = step05.video_roles(video, device)
    parts = []
    for begin in range(min_context, T + 1, chunk):
        frames = torch.arange(begin, min(begin + chunk, T + 1), device=device)
        rows = step05_state.state_rows(model, X, frames, num_labels, roles)
        if rows is not None:
            parts.append(rows)
    return (step05.concat_rows(parts), X) if parts else (None, X)


def _bins(p):
    return (p * NB).long().clamp(0, NB - 1)


def predict_on(p, now, th):
    """(B, K, L) bool predicted state in t + h: off now -> on if bin >= onset bin; on now -> stays on
    unless bin <= the offset bin (th: {"onset": (K,) long, "offset": (K,) long} bins)."""
    b = _bins(p)
    on_now = now[:, None, :] > 0.5
    return torch.where(on_now, b > th["offset"].to(b.device)[None, :, None], b >= th["onset"].to(b.device)[None, :, None])


@torch.no_grad()
def evaluate_pass(model, data, video_ids, ckpt, cfg, counted, thresholds=None):
    """Score histograms {scorer: (K, 2 now, 2 target, L, NB)} over the videos and, with thresholds,
    the exact-match counts {scorer: (K, 4) rows, rows right, changed rows, changed rows right}."""
    device, K, L = cfg["device"], len(ckpt["horizons"]), ckpt["num_label_atoms"]
    hist = {s: torch.zeros((K, 2, 2, L, NB), dtype=torch.float64, device=device) for s in SCORERS}
    exact = {s: torch.zeros((K, 4), dtype=torch.float64) for s in SCORERS}
    head_index = torch.arange(L, device=device)[None, :] * NB
    counted = counted.to(device)
    for video_id in tqdm(video_ids, desc=cfg.get("desc", "State forecast")):
        rows, _X = video_rows(model, data["videos"][video_id], data["num_atoms"], L, cfg["min_context"],
                              cfg["frames_per_video"], device)
        if rows is None:
            continue
        now = rows["now"] > 0.5
        for s, p in _scores(model, rows, ckpt["markov"]).items():
            b = _bins(p)
            for k in range(K):
                known = rows["known"][:, k]
                for c in (0, 1):
                    for y in (0, 1):
                        cell = known[:, None] & (now == bool(c)) & (rows["targets"][:, k] == bool(y))
                        idx = (head_index + b[:, k])[cell]
                        hist[s][k, c, y] += torch.bincount(idx, minlength=L * NB).view(L, NB).double()
            if thresholds is not None:
                pred = predict_on(p, rows["now"], thresholds[s])
                for k in range(K):
                    known = rows["known"][:, k]
                    tgt = rows["targets"][:, k]
                    right = ((pred[:, k] == tgt) | ~counted[None]).all(dim=1)
                    changed = ((tgt != now) & counted[None]).any(dim=1)
                    exact[s][k] += torch.tensor([known.sum().item(), (known & right).sum().item(),
                                                 (known & changed).sum().item(),
                                                 (known & changed & right).sum().item()], dtype=torch.float64)
    return {s: h.cpu() for s, h in hist.items()}, exact


# -------------- Metrics of the histograms --------------
def _ap(pos, neg):
    """AP of score histograms pos, neg (..., NB), higher bins first; one bin is one tie block."""
    pos, neg = pos.flip(-1), neg.flip(-1)
    tp, fp = pos.cumsum(-1), neg.cumsum(-1)
    P = pos.sum(-1)
    prec = tp / (tp + fp).clamp(min=1e-12)
    return ((prec * pos).sum(-1) / P.clamp(min=1e-12)).where(P > 0, torch.full_like(P, float("nan")))


def _at(pos, neg, b):
    """(tp, fp, fn) of predicting the bins >= b."""
    tp, fp = pos[..., b:].sum(-1), neg[..., b:].sum(-1)
    return tp, fp, pos.sum(-1) - tp


def _prf(tp, fp, fn):
    r = tp / max(tp + fn, 1e-12) if tp + fn > 0 else None
    p = tp / max(tp + fp, 1e-12) if tp + fp > 0 else None
    f = 2 * p * r / (p + r) if p and r else 0.0
    return {"recall": r, "precision": p, "f1": f}


def _best_bin(pos, neg):
    """Bin threshold of the best F1 of predicting bins >= b (pos, neg (NB,) summed over heads)."""
    tp = pos.flip(0).cumsum(0).flip(0)
    fp = neg.flip(0).cumsum(0).flip(0)
    f1 = 2 * tp / (2 * tp + fp + (pos.sum() - tp)).clamp(min=1e-12)
    return int(torch.argmax(f1))


def _tasks(h):
    """The (pos, neg) score histograms (L, NB) of one horizon h (2, 2, L, NB):
    onset (off now; p), offset (on now; 1 - p, the bins flipped), state (all; p)."""
    return {"onset": (h[0, 1], h[0, 0]), "offset": (h[1, 0].flip(-1), h[1, 1].flip(-1)),
            "state": (h[0, 1] + h[1, 1], h[0, 0] + h[1, 0])}


def tune_thresholds(hist, counted, scorer):
    """{"onset": (K,), "offset": (K,)} bins of the best micro F1 over the counted heads (val);
    persistence: never a change."""
    K = hist.shape[0]
    if scorer == "persist":
        return {"onset": torch.full((K,), NB, dtype=torch.long), "offset": torch.full((K,), -1, dtype=torch.long)}
    on, off = [], []
    for k in range(K):
        t = _tasks(hist[k])
        on.append(_best_bin(t["onset"][0][counted].sum(0), t["onset"][1][counted].sum(0)))
        b = _best_bin(t["offset"][0][counted].sum(0), t["offset"][1][counted].sum(0))
        off.append(NB - 1 - b)                  # flipped bin >= b  <=>  bin of p <= NB - 1 - b
    return {"onset": torch.tensor(on), "offset": torch.tensor(off)}


def _block(h, th_on, th_off):
    """Metrics of one histogram block (2, 2, NB), summed over its heads; th_on / th_off bins."""
    t = _tasks(h)
    out = {}
    ons_tp, ons_fp, ons_fn = _at(*t["onset"], max(th_on, 0)) if th_on < NB else (0.0, 0.0, float(t["onset"][0].sum()))
    b_flip = NB - 1 - th_off
    off_tp, off_fp, off_fn = (_at(*t["offset"], b_flip) if th_off >= 0
                              else (0.0, 0.0, float(t["offset"][0].sum())))
    ons_tp, ons_fp, ons_fn, off_tp, off_fp, off_fn = (float(x) for x in (ons_tp, ons_fp, ons_fn, off_tp, off_fp, off_fn))
    stay_on, stay_off = float(h[1, 1].sum()), float(h[0, 0].sum())
    for name, (tp, fp, fn) in (("onset", (ons_tp, ons_fp, ons_fn)), ("offset", (off_tp, off_fp, off_fn))):
        out[name] = {"count": tp + fn, "ap": _nan(_ap(*t[name])), **_prf(tp, fp, fn)}
    # The predicted state: on = predicted onsets + on-now cells not predicted off.
    kept_on = stay_on - off_fp                  # stay-on cells not predicted off
    s_tp, s_fp = ons_tp + kept_on, ons_fp + off_fn
    s_fn = ons_fn + off_fp
    out["state"] = {"positives": s_tp + s_fn, "ap": _nan(_ap(*t["state"])), **_prf(s_tp, s_fp, s_fn)}
    out["stay_on_kept"] = kept_on / stay_on if stay_on else None
    out["stay_off_kept"] = (stay_off - ons_fp) / stay_off if stay_off else None
    out["cells"] = float(h.sum())
    return out


def _nan(x):
    x = float(x)
    return None if x != x else x


def metrics_of(hist, exact, thresholds, counted, head_types, label_names, horizons):
    """{scorer: {horizon: {thresholds, micro, by_type, per_head, exact}}} of the test histograms."""
    out = {}
    types = step05.EVENT_TYPES
    for s, H in hist.items():
        out[s] = {}
        for k, h in enumerate(horizons):
            on, off = int(thresholds[s]["onset"][k]), int(thresholds[s]["offset"][k])
            hk = H[k]                                                       # (2, 2, L, NB)
            res = {"thresholds": {"onset_p_at_least": on / NB if on < NB else None,
                                  "offset_p_at_most": (off + 1) / NB if off >= 0 else None},
                   "micro": _block(hk[:, :, counted].sum(2), on, off), "by_type": {}, "per_head": {}}
            for g, name in enumerate(types):
                sel = counted & (head_types == g)
                if sel.any():
                    res["by_type"][name] = _block(hk[:, :, sel].sum(2), on, off)
            for l in torch.nonzero(counted).flatten().tolist():
                changes = float(hk[0, 1, l].sum() + hk[1, 0, l].sum())
                if changes >= STATE["min_changes"]:
                    res["per_head"][label_names[l]] = {"type": types[int(head_types[l])],
                                                       **_block(hk[:, :, l], on, off)}
            e = exact[s][k].tolist()
            res["exact"] = {"rows": e[0], "all_right": e[1] / e[0] if e[0] else None,
                            "changed_rows": e[2], "changed_all_right": e[3] / e[2] if e[2] else None}
            out[s][str(h)] = res
    return out


def print_summary(metrics, horizons, fps):
    def f(x):
        return "  -  " if x is None else f"{x:.3f}"
    for h in horizons:
        print(f"\n[State] Label state {h / fps:g} s ahead (micro over the counted heads, test):")
        print(f"  {'scorer':<10} {'state AP':>8} {'st R':>6} {'st P':>6} | {'onsets':>7} {'on AP':>6} {'on R':>6} "
              f"{'on P':>6} | {'offsets':>7} {'off AP':>6} {'off R':>6} {'off P':>6} | {'exact':>6} {'ex chg':>6}")
        for s, by_h in metrics.items():
            m = by_h[str(h)]["micro"]
            e = by_h[str(h)]["exact"]
            print(f"  {s:<10} {f(m['state']['ap']):>8} {f(m['state']['recall']):>6} {f(m['state']['precision']):>6} | "
                  f"{m['onset']['count']:>7.0f} {f(m['onset']['ap']):>6} {f(m['onset']['recall']):>6} "
                  f"{f(m['onset']['precision']):>6} | {m['offset']['count']:>7.0f} {f(m['offset']['ap']):>6} "
                  f"{f(m['offset']['recall']):>6} {f(m['offset']['precision']):>6} | {f(e['all_right']):>6} "
                  f"{f(e['changed_all_right']):>6}")
        heads = metrics["state_mlp"][str(h)]["per_head"]
        top = sorted(heads.items(), key=lambda kv: -(kv[1]["onset"]["count"] + kv[1]["offset"]["count"]))[:12]
        print(f"  state_mlp per head (most changes first): onset AP / R / P, offset AP / R / P")
        for name, m in top:
            print(f"    {name:<24} {m['type']:<8} on n={m['onset']['count']:<6.0f} {f(m['onset']['ap'])} "
                  f"{f(m['onset']['recall'])} {f(m['onset']['precision'])}   off n={m['offset']['count']:<6.0f} "
                  f"{f(m['offset']['ap'])} {f(m['offset']['recall'])} {f(m['offset']['precision'])}")


# -------------- Review videos --------------
def _runs(values):
    """[(start, end, value)] of the runs of equal values (list), value None skipped."""
    out, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            if values[start] is not None:
                out.append((start, i - 1, values[start]))
            start = i
    return out


@torch.no_grad()
def dense_video(model, video, data, ckpt, cfg, k, thresholds):
    """Per now-frame (T, N, L) tensors of one video for horizon index k: p (nan where no row),
    pred (predicted on in now + h), known (T, N); X labels (T, N, L); or None.
    thresholds: {"onset": (K,), "offset": (K,)} bins of all horizons (`tune_thresholds`)."""
    L = ckpt["num_label_atoms"]
    rows, X = video_rows(model, video, data["num_atoms"], L, cfg["min_context"], cfg["frames_per_video"],
                         cfg["device"])
    if rows is None:
        return None
    T, N = X.shape[:2]
    p_rows = torch.sigmoid(model(rows["inputs"], rows["now"]))
    pred_rows = predict_on(p_rows, rows["now"], thresholds)
    t, n = rows["rows"].T
    now = t - 1
    P = torch.full((T, N, L), float("nan"))
    pred = torch.zeros((T, N, L), dtype=torch.bool)
    known = torch.zeros((T, N), dtype=torch.bool)
    P[now.cpu(), n.cpu()] = p_rows[:, k].cpu()
    pred[now.cpu(), n.cpu()] = pred_rows[:, k].cpu()
    known[now.cpu(), n.cpu()] = rows["known"][:, k].cpu()
    return {"P": P, "pred": pred, "known": known, "S": X[..., :L].cpu(), "X": X.cpu(), "T": T, "N": N}


def transitions(d, t, n, h, heads):
    """[(kind, head, p, outcome)] at now-frame t for agent n: kind "start" / "stop" predicted or
    "missed start" / "missed stop"; outcome correct / false / missed / unknown."""
    S, P, pred, known = d["S"], d["P"], d["pred"], d["known"]
    T = d["T"]
    if torch.isnan(P[t, n]).all():
        return []
    out = []
    for l in heads:
        now_on = bool(S[t, n, l])
        pr = bool(pred[t, n, l])
        kn = bool(known[t, n])
        fut = bool(S[t + h, n, l]) if kn and t + h < T else None
        p = float(P[t, n, l])
        if not now_on and pr:
            out.append(("START", l, p, "unknown" if fut is None else ("correct" if fut else "false")))
        elif now_on and not pr:
            out.append(("STOP", l, p, "unknown" if fut is None else ("correct" if not fut else "false")))
        elif fut is not None and not now_on and fut:
            out.append(("MISSED START", l, p, "missed"))
        elif fut is not None and now_on and not fut:
            out.append(("MISSED STOP", l, p, "missed"))
    rank = {"missed": 0, "false": 1, "correct": 2, "unknown": 3}
    return sorted(out, key=lambda x: rank[x[3]])


def render_video(d, video_id, n0, boxes, classes, label_names, heads, th, h, fps, out_path, reader, ego):
    """One review MP4 of agent n0: the camera frame (left) and the side column (right): who it is and
    its GT labels now; the predicted changes in h frames (start / stop) and the missed ones; its
    predicted probabilities of the labels h frames ahead as bars (the cut of each: onset threshold for
    a label off now, offset threshold for one on now); its timeline, one row per label, the predicted
    state placed at the time it is about (frame now + h: green right, red false) over the GT state
    (blue predicted, yellow missed); the white line is now, the grey one now + h."""
    import cv2
    import numpy as np
    S, P, pred, known, T, N = d["S"], d["P"], d["pred"], d["known"], d["T"], d["N"]
    W_img = STATE["width"]
    H_img = int(round(W_img * 1280 / 1920))
    W_side, header_h = STATE["side_width"], 34
    fs = STATE["font_scale"]
    line_h = int(round(44 * fs))
    small = fs * 0.8
    is_ego = n0 == ego
    who = "EGO (camera car)" if is_ego else f"#{n0} {classes[n0] or ''}"
    th_on, th_off = int(th["onset"]), int(th["offset"])

    def short(l):
        return label_names[l].split(":", 1)[-1]

    # Timeline rows: labels the agent holds or is predicted to hold (at time f = now + h).
    pred_at = torch.zeros((T, len(heads)), dtype=torch.bool)       # predicted state about frame f
    has_pred = torch.zeros(T, dtype=torch.bool)
    for t in range(T - h):
        if known[t, n0]:
            pred_at[t + h] = pred[t, n0, heads]
            has_pred[t + h] = True
    gt = S[:, n0, heads]
    used = [i for i in range(len(heads)) if gt[:, i].any() or pred_at[:, i].any()]
    changes = [int((gt[1:, i] != gt[:-1, i]).sum()) for i in used]
    rows = [used[j] for j in sorted(range(len(used)), key=lambda j: -changes[j])]
    info_h = int(line_h * 3.4)
    now_h = line_h * (STATE["panel_lines"] + 1)
    pred_h = line_h * (STATE["pred_lines"] + 1)
    tl_h = max(H_img - info_h - now_h - pred_h, 4 * line_h)
    side_h = info_h + now_h + pred_h + tl_h
    title_h, axis_h = line_h, int(line_h * 0.9)
    room = max((tl_h - title_h - axis_h) // max(int(line_h * 0.7), 1), 1)
    hidden = max(len(rows) - room, 0)
    rows = rows[:room]
    row_h = min(max((tl_h - title_h - axis_h) // max(len(rows), 1), int(line_h * 0.7)), line_h)
    timeline = np.full((tl_h, W_side, 3), COLORS["panel"], dtype=np.uint8)
    x0, x1 = int(W_side * 0.32), W_side - 12

    def tx(t):
        return int(x0 + (x1 - x0) * t / max(T - 1, 1))
    _text(cv2, timeline, f"Timeline: predicted state ({h / fps:g} s ahead) over GT" + (f" (+{hidden})" if hidden else ""),
          8, int(title_h * 0.75), COLORS["text"], fs * 0.85, bg=False, thick=2)
    for r, i in enumerate(rows):
        y = title_h + r * row_h
        _text(cv2, timeline, short(heads[i])[:14], 8, y + row_h // 2 + int(8 * small), COLORS["text"], small, bg=False)
        cv2.line(timeline, (x0, y + row_h // 2), (x1, y + row_h // 2), (70, 70, 70), 1)
        split = y + (row_h * 3) // 5
        top = [("correct" if gt[f, i] else "false") if (has_pred[f] and pred_at[f, i]) else None for f in range(T)]
        bottom = [(("caught" if pred_at[f, i] else "missed") if has_pred[f] else "unknown") if gt[f, i] else None
                  for f in range(T)]
        for a, b, v in _runs(top):
            cv2.rectangle(timeline, (tx(a), y + 2), (max(tx(b), tx(a) + 2), split - 1), STATE_COLORS[v], -1)
        for a, b, v in _runs(bottom):
            cv2.rectangle(timeline, (tx(a), split + 2), (max(tx(b), tx(a) + 2), split + 5), STATE_COLORS[v], -1)
    axis_y = tl_h - axis_h + 4
    cv2.line(timeline, (x0, axis_y), (x1, axis_y), (120, 120, 120), 1)
    step = 5 if T / fps <= 60 else 10
    for sec in range(0, int(T / fps) + 1, step):
        x = tx(int(sec * fps))
        cv2.line(timeline, (x, axis_y), (x, axis_y + 5), (120, 120, 120), 1)
        _text(cv2, timeline, f"{sec}s", x - 10, axis_y + int(22 * small), (170, 170, 170), small * 0.8, bg=False)

    bx0, bx1 = int(W_side * 0.42), W_side - 170

    def bx(v):
        return int(bx0 + (bx1 - bx0) * min(max(v, 0.0), 1.0))

    W_out, H_out = W_img + W_side, header_h + max(H_img, side_h)
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
        items = transitions(d, t, n0, h, heads)
        color = STATE_COLORS[items[0][3]] if items else (255, 255, 255)
        for n in range(N):
            pb = box_of(t, n)
            if pb is not None and n != n0:
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

        info = np.full((info_h, W_side, 3), (40, 40, 40), dtype=np.uint8)
        _text(cv2, info, who, 8, int(line_h * 0.85), (255, 255, 255), fs * 1.15, bg=False, thick=2)
        labels = ",".join(short(l) for l in torch.nonzero(S[t, n0]).flatten().tolist())
        gt_lines = _wrap("GT now: " + (labels or ("-" if is_ego or box_of(t, n0) else "(not in view)")),
                         int(W_side / (19 * small)))
        for i, s in enumerate(gt_lines[:2]):
            _text(cv2, info, s, 8, int(line_h * (1.75 + 0.8 * i)), (210, 210, 210), small, bg=False)

        now = np.full((now_h, W_side, 3), (22, 22, 22), dtype=np.uint8)
        n_pred = sum(not k.startswith("MISSED") for k, *_ in items)
        n_miss = len(items) - n_pred
        _text(cv2, now, f"Changes in {h / fps:g} s: {n_pred} predicted, {n_miss} missed", 8, int(line_h * 0.75),
              COLORS["text"], fs, bg=False, thick=2)
        for i, (kind, l, p, outcome) in enumerate(items[:STATE["panel_lines"]]):
            _text(cv2, now, f"{kind} {short(l)} p={p:.2f} [{outcome}]", 8, int(line_h * (1.75 + i)),
                  STATE_COLORS[outcome], fs, bg=False, thick=2)
        if len(items) > STATE["panel_lines"]:
            _text(cv2, now, f"+{len(items) - STATE['panel_lines']}", W_side - 60, int(line_h * 0.75), COLORS["text"],
                  small, bg=False)

        pr = np.full((pred_h, W_side, 3), (30, 30, 30), dtype=np.uint8)
        _text(cv2, pr, f"Predicted in {h / fps:g} s (p on; GT then)", 8, int(line_h * 0.75), COLORS["text"],
              fs, bg=False, thick=2)
        if not torch.isnan(P[t, n0]).all():
            top = sorted(heads, key=lambda l: -float(P[t, n0, l]))[:STATE["pred_lines"]]
            for i, l in enumerate(top):
                v = float(P[t, n0, l])
                on_now = bool(S[t, n0, l])
                cut = (th_off + 1) / NB if on_now else th_on / NB
                on = bool(pred[t, n0, l])
                y = int(line_h * (1 + i))
                c = STATE_COLORS["pred_on"] if on else (150, 150, 150)
                name = short(l)[:12] + ("*" if on_now else "")
                _text(cv2, pr, name, 8, y + int(line_h * 0.7), (255, 255, 255) if on else (190, 190, 190),
                      small, bg=False, thick=2 if on else 1)
                cv2.rectangle(pr, (bx0, y + int(line_h * 0.2)), (bx1, y + int(line_h * 0.8)), (55, 55, 55), -1)
                cv2.rectangle(pr, (bx0, y + int(line_h * 0.2)), (max(bx(v), bx0 + 1), y + int(line_h * 0.8)), c, -1)
                cv2.line(pr, (bx(cut), y + int(line_h * 0.1)), (bx(cut), y + int(line_h * 0.9)), (255, 255, 255), 2)
                if known[t, n0] and t + h < T:
                    fut = bool(S[t + h, n0, l])
                    gc = STATE_COLORS["correct"] if fut == on else (STATE_COLORS["false"] if on else STATE_COLORS["missed"])
                    gtxt = "GT on" if fut else "GT off"
                else:
                    gc, gtxt = STATE_COLORS["unknown"], "GT ?"
                _text(cv2, pr, f"{v:.2f} {gtxt}", bx1 + 8, y + int(line_h * 0.7), gc, small, bg=False, thick=2)

        tl = timeline.copy()
        if t + h < T:
            cv2.line(tl, (tx(t + h), title_h - 2), (tx(t + h), axis_y + 5), (140, 140, 140), 2)
        cv2.line(tl, (tx(t), title_h - 2), (tx(t), axis_y + 5), (255, 255, 255), 2)
        side = np.vstack([info, now, pr, tl])
        body = np.zeros((H_out - header_h, W_out, 3), dtype=np.uint8)
        body[:H_img, :W_img] = frame
        body[:side.shape[0], W_img:] = side
        header = np.full((header_h, W_out, 3), COLORS["panel"], dtype=np.uint8)
        title = (f"{video_id}  {t / fps:5.1f}s  |  {who}  |  state MLP, labels {h / fps:g} s ahead  |  green correct  "
                 f"red false  yellow missed  blue GT predicted  * on now")
        _text(cv2, header, title, 6, 24, COLORS["text"], 0.62, bg=False, thick=1)
        writer.write(np.vstack([header, body]))
    writer.release()


def _video_counts(d, n, h, heads):
    """Cell counts (now-frames) of agent n: predicted starts / stops right and false, missed ones."""
    c = {"start_correct": 0, "start_false": 0, "start_missed": 0, "stop_correct": 0, "stop_false": 0, "stop_missed": 0}
    for t in range(d["T"]):
        for kind, _l, _p, outcome in transitions(d, t, n, h, heads):
            if outcome == "unknown":
                continue
            key = ("start" if "START" in kind else "stop") + "_" + ("missed" if kind.startswith("MISSED") else outcome)
            c[key] += 1
    return c


def review_videos(model, test_data, ckpt, cfg, thresholds, label_names, counted, output_dir, frame_dir, video_dir,
                  test_atoms_file, test_box_file):
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[State review]\t skipped: opencv (cv2) is not installed")
        return
    if not ((frame_dir and os.path.isdir(frame_dir)) or (video_dir and os.path.isdir(video_dir))):
        print(f"[State review]\t skipped: neither the frame dir {frame_dir} nor the video dir {video_dir} exists")
        return
    os.makedirs(output_dir, exist_ok=True)
    fps = cfg["frame_rate"]
    k = ckpt["horizons"].index(STATE["headline_horizon"]) if STATE["headline_horizon"] in ckpt["horizons"] else 0
    h = ckpt["horizons"][k]
    th = {"onset": thresholds["onset"][k], "offset": thresholds["offset"][k]}
    heads_all = torch.nonzero(counted).flatten().tolist()
    head_names = [label_names[l] for l in heads_all]
    atoms = utils_data.load_json(test_atoms_file)
    boxes_all = torch.load(test_box_file, map_location="cpu") if os.path.exists(test_box_file) else {}
    order = sorted(test_data["videos"])
    random.Random(STATE["seed"]).shuffle(order)
    summary, chosen = [], 0
    for video_id in tqdm(order, desc="State review videos"):
        if chosen >= STATE["num_videos"]:
            break
        if not FrameReader(video_id, frame_dir, video_dir).available:
            continue
        d = dense_video(model, test_data["videos"][video_id], test_data, ckpt, cfg, k, thresholds)
        if d is None:
            continue
        X_labels = d["S"]
        ego = step05.video_agent_ids(atoms[video_id]).get(None)
        focus = focus_agents(X_labels, ego, STATE["focus_others"])
        if not any((X_labels[1:, n][:, heads_all] != X_labels[:-1, n][:, heads_all]).any() for n in focus):
            continue                                            # no label change on a focus agent
        vb, classes, _ = step05._agent_boxes(boxes_all, video_id, atoms[video_id], X_labels)
        classes = list(classes)
        if ego is not None:
            classes[ego] = "EGO"
        for n in focus:
            tag = "ego" if n == ego else f"agent{n}_{classes[n] or 'unknown'}"
            out_dir = output_dir / video_id / tag
            os.makedirs(out_dir, exist_ok=True)
            heads = [heads_all[i] for i in agent_heads(X_labels, n, label_names, head_names)]
            reader = FrameReader(video_id, frame_dir, video_dir)
            render_video(d, video_id, n, vb, classes, label_names, heads, th, h, fps, out_dir / "state_mlp.mp4",
                         reader, ego)
            reader.close()
            summary.append({"video": video_id, "agent": tag, **_video_counts(d, n, h, heads)})
        chosen += 1
    with open(output_dir / "state_review.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]) if summary else ["video"])
        writer.writeheader()
        writer.writerows(summary)
    (output_dir / "README.txt").write_text(_readme(h, fps, th))
    print(f"[State review]\t {chosen} videos x (ego + {STATE['focus_others']} agents) in {output_dir}")


def _readme(h, fps, th):
    on = th["onset"] / NB if th["onset"] < NB else None
    off = (th["offset"] + 1) / NB if th["offset"] >= 0 else None
    return f"""State review videos (step06_state.py), frame rate {fps}: one folder per test video, in it one folder per
reviewed agent (ego/ and the {STATE['focus_others']} other agents seen longest), each with state_mlp.mp4.
In every frame ("now") the StateMLP predicts which GT labels the agent holds {h / fps:g} s later.
A label off now is predicted on if p >= {on}; a label on now is predicted off if p <= {off}
(thresholds of the best micro F1 on the val videos).
Right column: the agent and its GT labels now; the predicted changes ({h / fps:g} s ahead):
  START / STOP (green: it happens, red: it does not, grey: the agent is gone by then),
  MISSED START / MISSED STOP (yellow: the GT changes, the model keeps the state);
the highest p as bars (magenta: predicted on; white tick: the cut that applies; * the label is on now)
with the GT {h / fps:g} s later; the timeline, one row per label: top bar the predicted state placed at the
time it is about (green on and right, red on but GT off), the thin bar below the GT state (blue predicted,
yellow missed, grey no prediction); white line now, grey line now + {h / fps:g} s.
state_review.csv: per video and agent, the now-frames with a predicted / missed start or stop.
"""


# -------------- Main --------------
def main(input_data):
    print("\n--------- Step 06 (state): the label state h frames ahead ----------------------\n")
    step05_cfg = step05.get_cfg(input_data["step05_input"])
    step05_cfg["device"] = input_data["device"]
    scfg = step05_state.get_cfg(step05_cfg)
    out = input_data["output_dir"]
    test_file = out / "test_data.pt"
    if not os.path.exists(test_file):
        print(f"[State]\t skipped: no {test_file} (run step 06 first)")
        return
    test_data = torch.load(test_file, map_location="cpu")
    model, ckpt = step05_state.load_state_mlp(test_data, step05_cfg, scfg)
    if model is None:
        print(f"[State]\t skipped: no StateMLP {scfg['model_file']} (step05_state)")
        return
    cfg = {"device": input_data["device"], "frame_rate": input_data["frame_rate"],
           "min_context": step05_cfg["slot_size"] * step05_cfg["num_slots"],
           "frames_per_video": step05_cfg["frames_per_video"]}
    L, horizons = ckpt["num_label_atoms"], ckpt["horizons"]
    head_types = ckpt["head_types"]
    counted = torch.isin(head_types, torch.tensor([step05.EVENT_TYPES.index(t) for t in step06.COUNTED_TYPES]))
    vocab = step05.load_vocabulary({**step05_cfg, "unique_atoms_file": input_data["unique_atoms_file"]})
    label_names = [step05._atom_name(s) for s in vocab[:L]]

    # 1. Thresholds on the val videos.
    training_data = step05.load_training_data(step05_cfg)
    val_hist, _ = evaluate_pass(model, training_data, training_data["val_videos"], ckpt,
                                {**cfg, "desc": "State forecast (val)"}, counted)
    del training_data
    thresholds = {s: tune_thresholds(val_hist[s], counted, s) for s in SCORERS}
    for s, th in thresholds.items():
        print(f"[State] {s}: onset bins {th['onset'].tolist()}, offset bins {th['offset'].tolist()} (of {NB})")

    # 2. Test.
    hist, exact = evaluate_pass(model, test_data, sorted(test_data["videos"]), ckpt,
                                {**cfg, "desc": "State forecast (test)"}, counted, thresholds)
    metrics = metrics_of(hist, exact, thresholds, counted, head_types, label_names, horizons)
    utils_data.save_json({"horizons": horizons, "frame_rate": cfg["frame_rate"], "min_context": cfg["min_context"],
                          "model_file": str(scfg["model_file"]), "scorers": metrics}, out / "state_metrics.json")
    print(f"[State]\t {out / 'state_metrics.json'}")
    print_summary(metrics, horizons, cfg["frame_rate"])

    # 3. Review videos of the StateMLP.
    review_videos(model, test_data, ckpt, cfg, thresholds["state_mlp"], label_names, counted, out / "state_review",
                  input_data.get("frame_dir"), input_data.get("video_dir"), input_data["test_atoms_file"],
                  input_data["test_box_file"])
    print("\n--------- Step 06 (state) Done ---------------\n")
