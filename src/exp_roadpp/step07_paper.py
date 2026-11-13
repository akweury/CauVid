"""Step 07: the paper numbers of one run, at a fixed false-alarm budget.

Every onset model is compared at the same operating point: the score threshold that keeps its
false alarms per minute on the val videos at or below a budget (the lowest such threshold, so the
most recall), applied to the test videos. Alarms and events are those of `step06_episodes` (the
review videos' alarm logic): per agent and head, a stretch of high window-W scores on the at-risk
frames is one alarm (>= min_length frames, gaps of fill_gap smoothed); an alarm detects an event of
its agent and head if it starts at or before the onset and ends at most tol before it; an alarm that
detects nothing is false if its outcome is seen. Events: the onsets of the counted heads
(`step06.COUNTED_TYPES`: rare, regular, trivial; the noise heads excluded), window W = headline_window.

Scorers: "model" (the TemporalRuleNet), "mlp" (OnsetMLP, same inputs), "lookup" (own-label lookup),
"anchor" (the LLM anchor clauses alone), "prior" (base rate per head), and with the val calibration
of step 06 (`calibration.pt`, per-head Platt scaling) "model_cal", "mlp_cal", "lookup_cal": one
probability threshold for every head instead of one raw score threshold.

Output (step06_output): paper_metrics.json
  scorers: {name: {
    ap: cell AP of the window-W onset (at-risk and observed rows): micro, macro (mean over the
        heads with >= macro_min_onsets test onsets), by event type, ego heads ("av:");
    val_curve / test_curve: per threshold of the grid: false alarms per minute, recall, precision;
    budgets: {budget: {threshold, val_fa_per_min, test: {events, detected, recall, false_alarms,
        fa_per_min, precision, median_lead_s, mean_lead_s, by_type, ego}}}}}
  videos: per budget and scorer, the per-video counts (events, detected, false alarms, minutes, by type)
      for the bootstrap of `paper_tables.py`;
and paper_events.csv: one row per test event (video, frame, agent, head, type, ego) with the lead
(s) of every scorer at every budget, empty if missed: the event count and overlap table.
"""
import csv
import os
import torch
from tqdm import tqdm

from src.exp_roadpp import step05, step06, step06_episodes, utils_data

VERSION = "paper_v1"
PAPER = {
    "budgets": [1, 2, 5, 10, 20],               # false alarms per minute (all agents and counted heads)
    "headline_budget": 5,
    "tolerance_s": 1.0,
    "macro_min_onsets": 5,
    "scorers": ("model", "mlp", "lookup", "anchor", "prior"),
    "calibrated": ("model", "mlp", "lookup"),
}
NB = 2001                                       # score bins of logit(p) in [-14, 14] (AP histograms)


def threshold_grid():
    """Score thresholds: log-spaced 1e-5..0.5 (the lookup and prior scores are small rates) and
    linear 0.5..0.995."""
    low = torch.logspace(-5, torch.log10(torch.tensor(0.5)).item(), 80)
    high = torch.linspace(0.5, 0.995, 41)[1:]
    return [round(float(x), 6) for x in torch.cat([low, high])]


def _logit_bins(p):
    q = p.clamp(1e-7, 1 - 1e-7)
    z = torch.log(q) - torch.log1p(-q)
    return ((z.clamp(-14, 14) + 14) / 28 * (NB - 1)).round().long()


def _ap(pos, neg):
    """AP of histograms pos, neg (..., NB), higher bins first; nan without positives."""
    pos, neg = pos.flip(-1).double(), neg.flip(-1).double()
    tp, fp = pos.cumsum(-1), neg.cumsum(-1)
    P = pos.sum(-1)
    prec = tp / (tp + fp).clamp(min=1e-12)
    return ((prec * pos).sum(-1) / P.clamp(min=1e-12)).where(P > 0, torch.full_like(P, float("nan")))


# -------------- One video: scores, events, observability --------------
@torch.no_grad()
def video_scores(model, scorers, calibration, video, data, heads, k, cfg):
    """Dense window-k scores {name: (T, N, H)}, is_row (T, N), at_risk (T, N, H), the AP rows
    (onset target, scored mask) and the events of one video, or None (too short)."""
    device, m = cfg["device"], data["min_off_frames"]
    X = step05._video_from_cells(video, data["num_atoms"], device)
    roles = step05.video_roles(video, device)
    T, N, _A = X.shape
    if T <= cfg["min_context"]:
        return None
    H = len(heads)
    names = list(scorers) + [f"{n}_cal" for n in calibration]
    dense = {name: torch.zeros((T, N, H), device=device) for name in names}
    is_row = torch.zeros((T, N), dtype=torch.bool, device=device)
    at_risk = torch.zeros((T, N, H), dtype=torch.bool, device=device)
    ap_rows = {name: [] for name in names}
    for begin in range(cfg["min_context"], T, cfg["frames_per_video"]):
        frames = torch.arange(begin, min(begin + cfg["frames_per_video"], T), device=device)
        rows = step05.video_rows(model, X, frames, m, None, roles)
        if len(rows["rows"]) == 0:
            continue
        t, n = rows["rows"].T
        is_row[t, n] = True
        at_risk[t, n] = rows["at_risk"][:, heads]
        stage1 = model.window_probs(model(rows["inputs"])[0])
        scored = rows["at_risk"][:, heads] & rows["known"][:, k, heads]                 # (B, H)
        target = rows["targets"][:, k, heads]
        for name, score in scorers.items():
            p = score(rows["inputs"], stage1)                                              # (B, W, A)
            outs = {name: p[:, k, heads]}
            if name in calibration:
                outs[f"{name}_cal"] = step06.calibrate(p[:, :, heads], calibration[name], heads)[:, k]
            for nm, q in outs.items():
                dense[nm][t, n] = q
                ap_rows[nm].append((q, target, scored))
    onsets = step05.atom_onsets(X[..., heads], m)
    onsets[0] = False
    onsets[1:] &= X[:-1].any(dim=2, keepdim=True)
    onsets[:cfg["min_context"] + 1] = False
    e_t, e_n, e_h = torch.nonzero(onsets, as_tuple=True)
    next_gone = torch.cat([step06_episodes._next_index(~is_row),
                           torch.full((1, N), T, dtype=torch.long, device=device)])
    return {"dense": dense, "is_row": is_row, "at_risk": at_risk, "ap_rows": ap_rows, "T": T, "N": N,
            "events": (e_t, e_n, e_h), "next_gone": next_gone, "minutes": (T - cfg["min_context"]) / cfg["frame_rate"] / 60}


def episode_outcome(v, name, th, H, w, tol, ep):
    """Alarms of one scorer and threshold in one video: (detected (E,) bool, lead (E,) frames,
    correct alarms, false alarms)."""
    T, N = v["T"], v["N"]
    score = v["dense"][name].reshape(T, N * H)
    row_p = v["is_row"][:, :, None].expand(T, N, H).reshape(T, N * H)
    risk_p = v["at_risk"].reshape(T, N * H)
    pair, start, end = step06_episodes.alarm_episodes(score, row_p, risk_p, th, ep["fill_gap"], ep["min_length"])
    e_t, e_n, e_h = v["events"]
    if len(start) == 0:
        return torch.zeros_like(e_t, dtype=torch.bool), torch.full_like(e_t, -1), 0, 0
    s_n = pair // H
    hit = ((e_n * H + e_h)[:, None] == pair[None]) & (start[None] <= e_t[:, None]) & (e_t[:, None] <= end[None] + tol)
    correct = hit.any(dim=0)
    nxt = v["next_gone"]
    seen = (nxt[(end + 1).clamp(max=T), s_n] > end + max(tol, 1)) | (nxt[start, s_n] > start + w - 1)
    counted = correct | seen
    detected = hit.any(dim=1)
    lead = torch.where(detected, e_t - torch.where(hit, start[None], T).amin(dim=1), -1)
    return detected, lead, int(correct.sum()), int((counted & ~correct).sum())


# -------------- Passes --------------
def _new_count(types):
    return {"events": 0, "detected": 0, "false": 0, "correct": 0, "minutes": 0.0, "lead": [],
            "by_type": {t: {"events": 0, "detected": 0} for t in types}, "ego": {"events": 0, "detected": 0}}


@torch.no_grad()
def run_pass(model, scorers, calibration, data, video_ids, heads, head_type, is_ego, cfg, grid, chosen=None):
    """Over the videos: per scorer the curve over the threshold grid {th: counts}, the AP
    histograms (H, 2, NB) and, with chosen {name: {budget: th}}, the per-video counts and the
    per-event detections at those thresholds."""
    ep = step06_episodes._default_cfg(cfg["frame_rate"])
    k = model.windows.index(cfg["window"])
    w, tol, H = cfg["window"], int(round(PAPER["tolerance_s"] * cfg["frame_rate"])), len(heads)
    types = list(step06.COUNTED_TYPES)
    type_names = [step06.EVENT_TYPES.index(t) for t in types]
    names = list(scorers) + [f"{n}_cal" for n in calibration]
    curve = {nm: {th: _new_count(types) for th in grid} for nm in names}
    hist = {nm: torch.zeros((H, 2, NB), dtype=torch.float64) for nm in names}
    per_video = {nm: {b: [] for b in (chosen or {}).get(nm, {})} for nm in names}
    events = []
    for video_id in tqdm(video_ids, desc=cfg.get("desc", "Paper metrics")):
        v = video_scores(model, scorers, calibration, data["videos"][video_id], data, heads, k, cfg)
        if v is None:
            continue
        e_t, e_n, e_h = (x.cpu() for x in v["events"])
        e_type = head_type[e_h]
        e_ego = is_ego[e_h]
        record = [{"video": video_id, "frame": int(t), "agent": int(n), "head": int(heads[h]),
                   "type": types[type_names.index(int(ty))] if int(ty) in type_names else str(int(ty)),
                   "ego": bool(eg)} for t, n, h, ty, eg in zip(e_t, e_n, e_h, e_type, e_ego)]
        for nm in names:
            for q, target, scored in v["ap_rows"][nm]:
                b = _logit_bins(q)
                for y in (0, 1):
                    cell = scored & (target == bool(y))
                    idx = (torch.arange(H, device=b.device)[None, :] * NB + b)[cell]
                    hist[nm][:, y] += torch.bincount(idx.cpu(), minlength=H * NB).view(H, NB).double()
            budgets_at = {th: [bd for bd, t in (chosen or {}).get(nm, {}).items() if t == th] for th in grid}
            for th in grid:
                det, lead, n_correct, n_false = episode_outcome(v, nm, th, H, w, tol, ep)
                det, lead = det.cpu(), lead.cpu()
                c = curve[nm][th]
                c["events"] += len(det)
                c["detected"] += int(det.sum())
                c["false"] += n_false
                c["correct"] += n_correct
                c["minutes"] += v["minutes"]
                for bd in budgets_at[th]:
                    vc = {"video": video_id, "events": len(det), "detected": int(det.sum()), "false": n_false,
                          "correct": n_correct, "minutes": v["minutes"],
                          "by_type": {t: {"events": int((e_type == g).sum()), "detected": int((det & (e_type == g)).sum())}
                                      for t, g in zip(types, type_names)},
                          "ego": {"events": int(e_ego.sum()), "detected": int((det & e_ego).sum())},
                          "lead": [int(x) for x in lead[det].tolist()]}
                    per_video[nm][bd].append(vc)
                    for i, r in enumerate(record):
                        r[f"{nm}@{bd}"] = round(int(lead[i]) / cfg["frame_rate"], 2) if bool(det[i]) else None
        events += record
    return curve, hist, per_video, events


def _curve_json(curve):
    out = {}
    for th, c in curve.items():
        out[str(th)] = {"fa_per_min": c["false"] / c["minutes"] if c["minutes"] else None,
                        "recall": c["detected"] / c["events"] if c["events"] else None,
                        "precision": c["correct"] / (c["correct"] + c["false"]) if c["correct"] + c["false"] else None,
                        "detected": c["detected"], "events": c["events"], "false_alarms": c["false"]}
    return out


def choose_thresholds(val_curve, budgets):
    """{budget: threshold} on val: going down from the highest grid threshold, the last one before
    the false alarms per minute first exceed the budget (None if already the highest does).
    Not the lowest threshold under the budget anywhere: at very low thresholds the alarms merge into
    one long episode per agent and head (always on), few alarms that hit nearly every event."""
    out = {}
    ths = sorted(val_curve, reverse=True)
    for b in budgets:
        best = None
        for th in ths:
            c = val_curve[th]
            if c["minutes"] and c["false"] / c["minutes"] > b:
                break
            best = th
        out[b] = best
    return out


def summarize(videos, fps):
    """Totals of per-video counts."""
    ev = sum(v["events"] for v in videos)
    det = sum(v["detected"] for v in videos)
    false = sum(v["false"] for v in videos)
    correct = sum(v["correct"] for v in videos)
    minutes = sum(v["minutes"] for v in videos)
    lead = sorted(x for v in videos for x in v["lead"])
    out = {"events": ev, "detected": det, "recall": det / ev if ev else None, "false_alarms": false,
           "fa_per_min": false / minutes if minutes else None,
           "precision": correct / (correct + false) if correct + false else None,
           "median_lead_s": lead[len(lead) // 2] / fps if lead else None,
           "mean_lead_s": sum(lead) / len(lead) / fps if lead else None, "minutes": minutes}
    out["by_type"] = {}
    for t in step06.COUNTED_TYPES:
        e = sum(v["by_type"][t]["events"] for v in videos)
        d = sum(v["by_type"][t]["detected"] for v in videos)
        out["by_type"][t] = {"events": e, "detected": d, "recall": d / e if e else None}
    e = sum(v["ego"]["events"] for v in videos)
    d = sum(v["ego"]["detected"] for v in videos)
    out["ego"] = {"events": e, "detected": d, "recall": d / e if e else None}
    return out


def ap_summary(hist, head_type, is_ego, label_names):
    """Micro / macro / by-type / ego AP of the (H, 2, NB) histograms."""
    neg, pos = hist[:, 0], hist[:, 1]
    onsets = pos.sum(-1)
    per_head = _ap(pos, neg)
    macro = per_head[onsets >= PAPER["macro_min_onsets"]]
    out = {"micro": float(_ap(pos.sum(0), neg.sum(0))),
           "macro": float(macro.mean()) if len(macro) else None,
           "macro_heads": int(len(macro)), "onset_rows": float(onsets.sum()), "by_type": {}}
    for t in step06.COUNTED_TYPES:
        sel = head_type == step06.EVENT_TYPES.index(t)
        out["by_type"][t] = float(_ap(pos[sel].sum(0), neg[sel].sum(0))) if sel.any() else None
    out["ego"] = float(_ap(pos[is_ego].sum(0), neg[is_ego].sum(0))) if is_ego.any() else None
    out["per_head"] = {label_names[i]: {"ap": None if per_head[i] != per_head[i] else float(per_head[i]),
                                        "onset_rows": float(onsets[i])} for i in range(len(label_names))}
    return out


# -------------- Main --------------
def main(input_data):
    print("\n--------- Step 07: paper metrics at fixed false-alarm budgets ----------------------\n")
    out_dir = input_data["output_dir"]
    step05_cfg = step05.get_cfg(input_data["step05_input"])
    step05_cfg["device"] = input_data["device"]
    test_file, baseline_file, cal_file = out_dir / "test_data.pt", out_dir / "baselines.pt", out_dir / "calibration.pt"
    metrics_file, events_file = out_dir / "paper_metrics.json", out_dir / "paper_events.csv"
    if not (os.path.exists(test_file) and os.path.exists(baseline_file) and os.path.exists(step05_cfg["model_file"])):
        print("[Paper]\t skipped: run step 06 first (test_data.pt, baselines.pt)")
        return
    sources = [step05_cfg["model_file"], step05_cfg["mlp_model_file"], test_file, baseline_file, cal_file,
               step05_cfg["training_data_file"]]
    old = utils_data.load_json(metrics_file) if os.path.exists(metrics_file) else {}
    if (step05.is_fresh(metrics_file, *sources) and os.path.exists(events_file)
            and old.get("version") == VERSION and old.get("settings") == _settings()):
        print(f"[Paper]\t cached: {metrics_file}")
        return
    test_data = torch.load(test_file, map_location="cpu")
    model = step05.load_temporal_rule_net(test_data, step05_cfg)
    mlp = step05.load_onset_mlp(test_data, step05_cfg)
    baselines = torch.load(baseline_file, map_location="cpu", weights_only=False)
    calibration = torch.load(cal_file, map_location="cpu") if os.path.exists(cal_file) else {}
    names = [n for n in PAPER["scorers"] if n != "mlp" or mlp is not None]
    scorers = step06._scorers(model, baselines, mlp, None, step05_cfg["device"], names=names)
    calibration = {n: calibration[n] for n in PAPER["calibrated"] if n in calibration and n in scorers}
    training_data = step05.load_training_data(step05_cfg)
    # The thresholds are chosen on val: it must not be smaller than test (step_dataset: 15% / 15%).
    if len(training_data["val_videos"]) < len(test_data["videos"]):
        raise ValueError(f"{len(training_data['val_videos'])} val videos < {len(test_data['videos'])} test videos: "
                         "the val split must be at least as large as test for the thresholds chosen on it.")
    head_types_all = training_data["event_types"]["types"]
    counted = [step06.EVENT_TYPES.index(t) for t in step06.COUNTED_TYPES]
    heads = torch.nonzero(torch.isin(head_types_all, torch.tensor(counted))).flatten()
    vocab = step05.load_vocabulary({**step05_cfg, "unique_atoms_file": input_data["unique_atoms_file"]})
    label_names = [step05._atom_name(vocab[int(h)]) for h in heads]
    head_type = head_types_all[heads].cpu()
    is_ego = torch.tensor([name.startswith("av:") for name in label_names])
    # min_context as in step 06 (the motif net's longer memory, if it was evaluated there).
    episodes = out_dir / "episode_alarms.json"
    min_context = (utils_data.load_json(episodes)["min_context"] if os.path.exists(episodes)
                   else max(step05_cfg["slot_size"] * step05_cfg["num_slots"], input_data.get("eval_min_context", 0)))
    cfg = {"device": step05_cfg["device"], "frame_rate": input_data["frame_rate"], "min_context": min_context,
           "frames_per_video": step05_cfg["frames_per_video"], "window": 20}
    heads_d = heads.to(cfg["device"])
    grid = threshold_grid()

    # 1. Val: the false alarms per minute of every threshold; the threshold of each budget.
    val_data = step06._subset(training_data, training_data["val_videos"])
    del training_data
    val_curve, _, _, _ = run_pass(model, scorers, calibration, val_data, sorted(val_data["videos"]), heads_d,
                                  head_type, is_ego, {**cfg, "desc": "Paper metrics (val)"}, grid)
    del val_data
    chosen = {nm: choose_thresholds(c, PAPER["budgets"]) for nm, c in val_curve.items()}

    # 2. Test at those thresholds (and the whole test curve).
    test_curve, hist, per_video, events = run_pass(model, scorers, calibration, test_data,
                                                   sorted(test_data["videos"]), heads_d, head_type, is_ego,
                                                   {**cfg, "desc": "Paper metrics (test)"}, grid,
                                                   {nm: {b: th for b, th in c.items() if th is not None}
                                                    for nm, c in chosen.items()})
    fps = cfg["frame_rate"]
    out = {"version": VERSION, "settings": _settings(), "window": cfg["window"], "frame_rate": fps,
           "min_context": min_context, "counted_types": list(step06.COUNTED_TYPES),
           "num_test_videos": len(test_data["videos"]), "scorers": {}, "videos": {}}
    for nm in val_curve:
        vc = val_curve[nm]
        out["scorers"][nm] = {
            "ap": ap_summary(hist[nm], head_type, is_ego, label_names),
            "val_curve": _curve_json(vc), "test_curve": _curve_json(test_curve[nm]),
            "budgets": {str(b): {"threshold": th,
                                 "val_fa_per_min": vc[th]["false"] / vc[th]["minutes"] if th is not None else None,
                                 "test": summarize(per_video[nm][b], fps) if th is not None else None}
                        for b, th in chosen[nm].items()}}
        out["videos"][nm] = {str(b): per_video[nm][b] for b in per_video[nm]}
    utils_data.save_json(out, metrics_file)
    columns = ["video", "frame", "agent", "head", "label", "type", "ego"] + [
        f"{nm}@{b}" for nm in val_curve for b in PAPER["budgets"]]
    with open(events_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in events:
            writer.writerow({**{c: r.get(c) for c in columns},
                             "label": label_names[int((heads == r["head"]).nonzero()[0])]})
    print_summary(out)
    print(f"[Paper]\t {metrics_file}\n[Paper]\t {events_file}")
    print("\n--------- Step 07 Done ---------------\n")


def _settings():
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in PAPER.items()}


def print_summary(out):
    def f(x, p="{:.3f}"):
        return "  -  " if x is None else p.format(x)
    b = str(PAPER["headline_budget"])
    print(f"\n[Paper] W={out['window']} frames, budget {b} false alarms / min (threshold chosen on val):")
    print(f"  {'scorer':<11} {'AP':>6} {'mAP':>6} | {'th':>7} {'FA/min':>6} {'R':>6} {'P':>6} {'lead':>5} | "
          + " ".join(f"{t:>8}" for t in step06.COUNTED_TYPES) + f" {'ego':>8}")
    for nm, s in out["scorers"].items():
        t = s["budgets"][b]["test"] or {}
        types = " ".join(f"{t['by_type'][x]['detected']:>3}/{t['by_type'][x]['events']:<4}" if t else f"{'-':>8}"
                         for x in step06.COUNTED_TYPES)
        ego = f"{t['ego']['detected']:>3}/{t['ego']['events']:<4}" if t else f"{'-':>8}"
        print(f"  {nm:<11} {f(s['ap']['micro'])} {f(s['ap']['macro'])} | {f(s['budgets'][b]['threshold'], '{:.4f}'):>7} "
              f"{f(t.get('fa_per_min'), '{:.1f}'):>6} {f(t.get('recall'))} {f(t.get('precision'))} "
              f"{f(t.get('median_lead_s'), '{:.1f}'):>5} | {types} {ego}")
