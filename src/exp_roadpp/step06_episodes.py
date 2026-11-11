"""Step 06e: episode alarms, one alarm per stretch of high scores.

Per agent and head, the frames whose window-W score is >= a threshold form alarm
episodes instead of one alarm per upward crossing (`step06`):
  - frames: the rows of `step06` (t >= min_context, the agent visible in t - 1 and t).
    A row whose atom is at risk (off in the min_off_frames frames before) is high
    (score >= threshold) or low. A row whose atom is not at risk (on, or on within
    the last min_off_frames frames: an event in progress) is neutral: it neither
    starts, breaks nor ends an episode, so the high frames before and after an
    event join into one episode. A frame without a row (the agent left the scene or
    its track dropped out, the video start) ends the episode.
  - smoothing: a run of at most fill_gap low frames between two high frames is high.
  - episode: a maximal run of high frames (the neutral frames inside included); it
    is an alarm if it spans at least min_length frames, from its first high frame s
    to its last e.
  - events: the exact onsets of `step06` (on in o, off in the min_off_frames
    before, the agent visible in o - 1, o > min_context), each lasting until its
    last on frame `end` (the episode of the label ends at min_off_frames off frames
    in a row or when the agent leaves).
  - matching, with a tolerance of tol frames: an alarm hits an event of its agent
    and head if s <= o <= e + tol. An alarm hitting several events is one correct
    alarm; an event is detected if some alarm hits it. Prediction time (lead) =
    o - s of the earliest alarm hitting it; lag = e - end of the latest one
    (> 0: the alarm goes on after the event, < 0: it stops before the event ends).
  - an alarm that hits no event is a false alarm only if its outcome is seen: the
    agent stays in the scene (rows) for max(tol, 1) frames after e. Alarms cut off by
    the agent leaving or the video end are left out.
Computed for every window of the model, every score threshold and tolerance, on the
counted heads (`step06.COUNTED_TYPES`), for the rule net and the OnsetMLP.
"""
import os
import torch
from tqdm import tqdm
from src.exp_roadpp import step05, step06, utils_data

SCORERS = ("model", "mlp")


def _default_cfg(frame_rate):
    return {
        # Score thresholds of the alarm episodes.
        "thresholds": [round(0.1 * i, 1) for i in range(1, 10)],
        # Matching tolerance: an alarm ending at most this long before the onset still hits it.
        "tolerances_s": [0.5 * i for i in range(11)],                           # 0 .. 5 s
        # An alarm episode spans at least this many frames (0.5 s).
        "min_length": int(round(0.5 * frame_rate)),
        # Runs of at most this many low frames between high frames are smoothed away.
        "fill_gap": 1,
        # The operating point of the printed tables.
        "print_threshold": 0.5,
        "print_tolerance_s": 1.0,
        # Version of this metric; a cached file of another one is redone.
        "metric": "episode_alarms_v1",
    }


# -------------- Episodes --------------
def _last_index(mask, dim=0):
    """Index of the last True at or before each position along dim (-1 if none)."""
    idx = torch.arange(mask.shape[dim], device=mask.device).view(-1, *([1] * (mask.dim() - 1)))
    return torch.where(mask, idx, -1).cummax(dim=dim).values


def _next_index(mask):
    """Index of the first True at or after each position along dim 0 (len if none)."""
    L = mask.shape[0]
    return L - 1 - _last_index(mask.flip(0)).flip(0)


def alarm_episodes(score, is_row, at_risk, threshold, fill_gap, min_length):
    """Alarm episodes of one video.

    score (T, P) scores of P agent-head pairs, is_row (T, P) bool, at_risk (T, P) bool.
    Returns pair (S,), start (S,), end (S,) long: the alarms (first and last high frame).
    """
    T, P = score.shape
    device = score.device
    risk = at_risk & is_row
    keep = risk | ~is_row                                       # neutral frames (row, not at risk) are skipped
    state = torch.where(risk, torch.where(score >= threshold, 2, 1), 0)   # 2 high, 1 low, 0 no row
    # Compress every column to its kept frames: neutral frames neither break nor end an episode.
    pos = keep.long().cumsum(dim=0) - 1
    L = int(keep.sum(dim=0).max()) if T else 0
    if L == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, empty
    col = torch.arange(P, device=device).expand(T, P)
    frame = torch.arange(T, device=device)[:, None].expand(T, P)
    C = torch.zeros((L, P), dtype=torch.long, device=device)
    O = torch.full((L, P), T, dtype=torch.long, device=device)
    C[pos[keep], col[keep]] = state[keep]
    O[pos[keep], col[keep]] = frame[keep]
    high, gone = C == 2, C == 0
    if fill_gap > 0:
        prev_high, prev_gone = _last_index(high), _last_index(gone)
        next_high, next_gone = _next_index(high), _next_index(gone)
        fill = ((C == 1) & (prev_high >= 0) & (prev_high > prev_gone) & (next_high < next_gone)
                & (next_high - prev_high - 1 <= fill_gap))
        high = high | fill
    before = torch.cat([torch.zeros((1, P), dtype=torch.bool, device=device), high[:-1]])
    after = torch.cat([high[1:], torch.zeros((1, P), dtype=torch.bool, device=device)])
    p_start, i_start = torch.nonzero((high & ~before).T, as_tuple=True)      # ordered by pair, then frame
    p_end, i_end = torch.nonzero((high & ~after).T, as_tuple=True)
    start, end = O[i_start, p_start], O[i_end, p_end]
    long_enough = end - start + 1 >= min_length
    return p_start[long_enough], start[long_enough], end[long_enough]


def _event_ends(X, e_t, e_n, e_a, min_off_frames):
    """(E,) last on frame of each event's label episode: until min_off_frames off frames
    in a row, or the agent leaving the scene."""
    Xc = X.cpu().numpy()
    visible = Xc.any(axis=2)
    T = X.shape[0]
    ends = []
    for t, n, a in zip(e_t.tolist(), e_n.tolist(), e_a.tolist()):
        last, off = t, 0
        for f in range(t, T):
            if not visible[f, n]:
                break
            if Xc[f, n, a]:
                last, off = f, 0
            else:
                off += 1
                if off >= min_off_frames:
                    break
        ends.append(last)
    return torch.tensor(ends, dtype=torch.long)


# -------------- Scoring --------------
def _empty_stats():
    return {"events": 0, "detected": 0, "alarms": 0, "correct": 0, "excluded": 0, "lead": [], "lag": []}


@torch.no_grad()
def score_episodes(model, mlp, baselines, data, cfg, ep):
    """{scorer: {window: {threshold: {tol_frames: stats}}}} and the scored frames.
    stats: events, detected, alarms (counted), correct, excluded (unobservable alarms),
    lead / lag (frames, per detected event), by_type {type: the same}."""
    device, fps = cfg["device"], cfg["frame_rate"]
    A, m, min_context = data["num_atoms"], data["min_off_frames"], cfg["min_context"]
    head_type = cfg["head_types"].to(device)
    counted_types = [step06.EVENT_TYPES.index(t) for t in step06.COUNTED_TYPES]
    heads = torch.nonzero(torch.isin(head_type, torch.tensor(counted_types, device=device))).flatten()
    H = len(heads)
    type_of_head = head_type[heads]                                             # (H,)
    windows = list(model.windows)
    tolerances = [int(round(s * fps)) for s in ep["tolerances_s"]]
    scorers = step06._scorers(model, baselines, mlp, None, device, names=SCORERS)
    stats = {name: {w: {th: {tol: {**_empty_stats(), "by_type": {t: _empty_stats() for t in step06.COUNTED_TYPES}}
                              for tol in tolerances} for th in ep["thresholds"]} for w in windows}
             for name in scorers}
    num_frames = 0
    for video_id in tqdm(data["videos"], desc="Scoring alarm episodes"):
        video = data["videos"][video_id]
        X = step05._video_from_cells(video, A, device)
        roles = step05.video_roles(video, device)
        T, N, _A = X.shape
        if T <= min_context:
            continue
        num_frames += T - min_context
        is_row = torch.zeros((T, N), dtype=torch.bool, device=device)
        at_risk = torch.zeros((T, N, H), dtype=torch.bool, device=device)
        dense = {name: torch.zeros((len(windows), T, N, H), device=device) for name in scorers}
        for begin in range(min_context, T, cfg["frames_per_video"]):
            frames = torch.arange(begin, min(begin + cfg["frames_per_video"], T), device=device)
            rows = step05.video_rows(model, X, frames, m, None, roles)
            if len(rows["rows"]) == 0:
                continue
            t, n = rows["rows"].T
            is_row[t, n] = True
            at_risk[t, n] = rows["at_risk"][:, heads]
            stage1 = model.window_probs(model(rows["inputs"])[0])
            for name, score in scorers.items():
                p = score(rows["inputs"], stage1)                                  # (B, W, A)
                dense[name][:, t, n] = p[:, :, heads].transpose(0, 1)
        # Events: the exact onsets of the counted heads (as `step06._score_videos`).
        onsets = step05.atom_onsets(X[..., heads], m)
        onsets[0] = False
        onsets[1:] &= X[:-1].any(dim=2, keepdim=True)
        onsets[:min_context + 1] = False
        e_t, e_n, e_h = torch.nonzero(onsets, as_tuple=True)
        e_end = _event_ends(X, e_t, e_n, heads[e_h], m).to(device)
        e_pair, e_type = e_n * H + e_h, type_of_head[e_h]
        # First frame >= t without a row of the agent (T past the video end): alarm observability.
        next_gone = torch.cat([_next_index(~is_row), torch.full((1, N), T, dtype=torch.long, device=device)])
        row_p = is_row[:, :, None].expand(T, N, H).reshape(T, N * H)
        risk_p = at_risk.reshape(T, N * H)
        for name in scorers:
            for k, w in enumerate(windows):
                score = dense[name][k].reshape(T, N * H)
                for th in ep["thresholds"]:
                    pair, start, end = alarm_episodes(score, row_p, risk_p, th, ep["fill_gap"], ep["min_length"])
                    s_n, s_type = pair // H, type_of_head[pair % H]
                    for tol in tolerances:
                        # hit (E, S): the alarm covers the onset or ends at most tol frames before it.
                        hit = ((e_pair[:, None] == pair[None]) & (start[None] <= e_t[:, None])
                               & (e_t[:, None] <= end[None] + tol))
                        correct = hit.any(dim=0)
                        seen = next_gone[(end + 1).clamp(max=T), s_n] > end + max(tol, 1)
                        counted = correct | seen
                        detected = hit.any(dim=1)
                        lead = e_t - torch.where(hit, start[None], T).amin(dim=1) if len(start) else None
                        lag = torch.where(hit, end[None], -1).amax(dim=1) - e_end if len(start) else None
                        groups = [(stats[name][w][th][tol], None)] + \
                                 [(stats[name][w][th][tol]["by_type"][t], step06.EVENT_TYPES.index(t))
                                  for t in step06.COUNTED_TYPES]
                        for st, g in groups:
                            ev = torch.ones_like(e_t, dtype=torch.bool) if g is None else e_type == g
                            al = torch.ones_like(start, dtype=torch.bool) if g is None else s_type == g
                            st["events"] += int(ev.sum())
                            st["detected"] += int((detected & ev).sum())
                            st["alarms"] += int((counted & al).sum())
                            st["correct"] += int((correct & al).sum())
                            st["excluded"] += int((~counted & al).sum())
                            if lead is not None:
                                st["lead"] += lead[detected & ev].tolist()
                                st["lag"] += lag[detected & ev].tolist()
    return stats, num_frames


def _summary(st, minutes, fps):
    def mean(v):
        return sum(v) / len(v) / fps if v else None

    def median(v):
        return sorted(v)[len(v) // 2] / fps if v else None
    false = st["alarms"] - st["correct"]
    return {
        "events": st["events"], "detected": st["detected"], "missed": st["events"] - st["detected"],
        "recall": st["detected"] / st["events"] if st["events"] else 0.0,
        "alarms": st["alarms"], "correct_alarms": st["correct"], "false_alarms": false,
        "excluded_alarms": st["excluded"],
        "precision": st["correct"] / st["alarms"] if st["alarms"] else 0.0,
        "false_alarms_per_minute": false / minutes if minutes else None,
        "events_per_correct_alarm": st["detected"] / st["correct"] if st["correct"] else None,
        "mean_lead_s": mean(st["lead"]), "median_lead_s": median(st["lead"]),
        "mean_lag_s": mean(st["lag"]), "median_lag_s": median(st["lag"]),
    }


def evaluate(model, mlp, baselines, data, cfg, ep):
    stats, num_frames = score_episodes(model, mlp, baselines, data, cfg, ep)
    fps = cfg["frame_rate"]
    minutes = num_frames / fps / 60
    out = {**{key: ep[key] for key in ep}, "frame_rate": fps, "minutes": minutes,
           "min_context": cfg["min_context"], "counted_types": list(step06.COUNTED_TYPES), "results": {}}
    for name, by_w in stats.items():
        out["results"][name] = {
            str(w): {str(th): {str(tol / fps): {**_summary(st, minutes, fps),
                                                "by_type": {t: _summary(s, minutes, fps)
                                                            for t, s in st["by_type"].items()}}
                               for tol, st in by_tol.items()}
                     for th, by_tol in by_th.items()}
            for w, by_th in by_w.items()}
    return out


# -------------- Printing --------------
def _fmt(v, f="{:.2f}"):
    return "-" if v is None else f.format(v)


def print_summary(metrics, headline_window):
    w, th, tol = str(headline_window), str(metrics["print_threshold"]), str(float(metrics["print_tolerance_s"]))
    print(f"[Episode alarms] window {w} frames; high frames merged into one alarm per agent and head "
          f"(gaps <= {metrics['fill_gap']} frames smoothed, >= {metrics['min_length']} frames long); "
          f"{metrics['minutes']:.1f} min")
    header = (f"  {'':<8}{'events':>7}{'detect':>7}{'recall':>7}{'alarms':>7}{'false':>7}{'prec':>7}"
              f"{'FA/min':>7}{'lead s':>7}{'lag s':>7}")
    for name, by_w in metrics["results"].items():
        by_th = by_w[w]
        print(f"  {name}: threshold {th}, by tolerance (s)")
        print(header)
        for t, r in by_th[th].items():
            print(f"  {t:<8}{r['events']:>7}{r['detected']:>7}{r['recall']:>7.3f}{r['alarms']:>7}"
                  f"{r['false_alarms']:>7}{r['precision']:>7.3f}{_fmt(r['false_alarms_per_minute'], '{:.1f}'):>7}"
                  f"{_fmt(r['mean_lead_s']):>7}{_fmt(r['mean_lag_s']):>7}")
        print(f"  {name}: tolerance {tol} s, by threshold")
        print(header)
        for t, by_tol in by_th.items():
            r = by_tol[tol]
            print(f"  {t:<8}{r['events']:>7}{r['detected']:>7}{r['recall']:>7.3f}{r['alarms']:>7}"
                  f"{r['false_alarms']:>7}{r['precision']:>7.3f}{_fmt(r['false_alarms_per_minute'], '{:.1f}'):>7}"
                  f"{_fmt(r['mean_lead_s']):>7}{_fmt(r['mean_lag_s']):>7}")
    print()


def main(model, mlp, baselines, test_data, step05_cfg, cfg, output_file):
    """Episode alarm metrics of the test videos (cached in output_file, redone when the
    models, the test data or the settings change)."""
    ep = _default_cfg(cfg["frame_rate"])
    metrics = utils_data.load_json(output_file) if os.path.exists(output_file) else {}
    sources = [step05_cfg["model_file"], step05_cfg["mlp_model_file"], cfg["test_data_file"], cfg["baseline_file"]]
    if (not step05.is_fresh(output_file, *sources) or {k: metrics.get(k) for k in ep} != ep
            or metrics.get("min_context") != cfg["min_context"]):
        if "head_types" not in cfg:
            cfg["head_types"] = step05.load_training_data(step05_cfg)["event_types"]["types"]
        metrics = evaluate(model, mlp, baselines, test_data, cfg, ep)
        utils_data.save_json(metrics, output_file)
    print_summary(metrics, cfg["headline_window"])
    print(f"[Results]\t {output_file}")
    return metrics
