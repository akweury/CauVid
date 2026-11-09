"""Step 06: evaluate the step 05 window predictions on the test videos.

For every agent visible in frame t (with the model's full history before t) and
every atom that is off, the model says how likely the atom switches on within
the next W frames, for each window W of step 05 (10, 20, ..., 100). Per window:
  rows: precision-recall / AP of these probabilities against "it did switch on
      within W" (rows whose window runs past the video end without an onset are
      censored), and their calibration;
  events: an onset is detected if, in some frame of its at-risk stretch (the
      frames before the onset in which the agent is scored and the atom is off,
      back to the first gap), the window-W score of its agent and head reaches
      detected_th; one frame is enough. Prediction time: the frames from the
      earliest detected frame to the onset (None if missed). Coverage: the
      detected frames / the prediction time (0 if missed);
  alarms: an alarm is raised when the score crosses the threshold upwards (one
      per episode, not one per frame); it is correct if the onset follows within W.
The threshold is the fixed cfg["detected_th"] (the rule set and the candidates use
FIXED_THRESHOLDS); every metric is also stored as a curve over all thresholds.
Every metric is also reported per event type: the heads are split by their
frequency in the train + val videos into noise, regular and trivial events
(`step05.head_event_types`). The noise heads are no training targets (step 05
exclude_noise_heads), so the overall metrics count only the regular and trivial
events and heads; the noise events are reported on their own.
"""
import os
import torch
from tqdm import tqdm
from src.exp_roadpp import step05, step05_causal, step05_motif, step05_rules, utils_data


def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "iou": iou, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def _short_num(n):
    """Format a number compactly, e.g. 143280 -> '143k', 1000000 -> '1M'."""
    n = float(n)
    for unit in ["", "k", "M", "B"]:
        s = f"{n:.0f}" if unit == "" or abs(n) >= 99.95 else f"{n:.1f}".rstrip("0").rstrip(".")
        if abs(float(s)) < 1000:
            return s + unit
        n /= 1000
    return f"{n:.1f}T"


def score_edges():
    """(S,) lower edges of the score bins: 0, then 40 log-spaced bins per decade
    from 1e-8 to 1e-2 (onsets are rare, so most scores are tiny), then steps of 0.01."""
    return torch.cat([torch.zeros(1), torch.logspace(-8, -2, 241)[:-1], torch.linspace(0.01, 0.99, 99)])


def _score_bins(scores, edges):
    return (torch.searchsorted(edges, scores.contiguous(), right=True) - 1).clamp(min=0)


# -------------- Baselines --------------
@torch.no_grad()
def fit_baselines(model, training_data, step05_cfg):
    """Scores of two baselines, fitted on the step 05 train videos (full history):
      prior (W, A): the share of at-risk rows whose atom switches on within window
          k, the same for every row;
      anchor_confidence (R, W, A): for each seeded rule r (anchor body in one slot)
          and each head of its anchor clause, the confidence of the anchor input
          alone (seen in the memory) for an onset within window k; -1 for other
          heads, free rules and anchors that never fire.
    The anchor baseline is the anchor clauses without the learned extra atoms, so
    the model beating it shows the extensions help."""
    device = step05_cfg["device"]
    R, W, A = model.log_alpha.shape[0], len(model.windows), model.num_atoms
    is_anchor_head = torch.zeros((R, A), dtype=torch.bool)
    for i, (_body, heads) in enumerate(step05.anchor_rules(step05_cfg)):
        is_anchor_head[i * model.num_bins:(i + 1) * model.num_bins, heads] = True
    anchor_body = model.fixed_inputs.float()
    anchor_size = anchor_body.sum(dim=1)
    hits, fires = torch.zeros((R, W * A), device=device), torch.zeros((R, W * A), device=device)
    onsets, at_risk = torch.zeros(W * A, device=device), torch.zeros(W * A, device=device)
    for video_id in tqdm(training_data["train_videos"], desc="Fitting the baselines (train)"):
        for frames in training_data["frames"][video_id].split(step05_cfg["frames_per_video"]):
            rows = step05._video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            x, _k = model.split_inputs(rows["inputs"])
            risk = (rows["at_risk"][:, None, :] & rows["known"]).flatten(1).float()
            onset = rows["targets"].flatten(1).float()
            fired = ((x @ anchor_body.T >= anchor_size) & (anchor_size > 0)).float()
            fires += fired.T @ risk
            hits += fired.T @ onset
            onsets += onset.sum(dim=0)
            at_risk += risk.sum(dim=0)
    fires, hits = fires.view(R, W, A).cpu(), hits.view(R, W, A).cpu()
    valid = is_anchor_head[:, None, :] & (fires > 0)
    return {
        "prior": (onsets / at_risk.clamp(min=1)).view(W, A).cpu(),
        "anchor_confidence": torch.where(valid, hits / fires.clamp(min=1), -1.0),
    }


def _anchor_scores(model, inputs, baselines, chunk_size=8):
    """(B, W, A) anchor baseline score of each row: the highest confidence of the
    anchors seen in the memory for the head and window, or the prior if none."""
    x, _k = model.split_inputs(inputs)
    confidence = baselines["anchor_confidence"]
    used = (confidence >= 0).flatten(1).any(dim=1)
    confidence = confidence[used]
    anchor_body = model.fixed_inputs[used].float()
    anchor_size = anchor_body.sum(dim=1)
    fired = (x @ anchor_body.T >= anchor_size) & (anchor_size > 0)                 # (B, R')
    best = torch.full((len(inputs), *baselines["prior"].shape), -1.0, device=inputs.device)
    if len(confidence):
        best = torch.cat([torch.where(chunk[:, :, None, None], confidence[None], -1.0).max(dim=1).values
                          for chunk in fired.split(chunk_size)])
    return torch.where(best >= 0, best, baselines["prior"][None])


# -------------- Scorers --------------
# The scorers whose detections are counted (the prior flags all rows of a head alike);
# "mlp" only when the OnsetMLP baseline was trained; "candidate" (the stage-1 flags
# at the cascade thresholds, a fixed 0 / 1 score) and "cascade" (the verifier's
# probabilities on those candidates, 0 elsewhere) only when the step 05 cascade was trained.
# "net_pruned": the rule net after the step 05b causal pruning, if it was made; "rule_set":
# the step 05c rule set (each rule precise, the set covering the events), if it was made.
# "motif_net": the step 05d two-level MotifRuleNet, if it was trained.
DETECTION_SCORERS = ("model", "net_pruned", "rule_set", "motif_net", "mlp", "anchor", "candidate", "cascade")
# Scorers detected at a fixed threshold instead of cfg["detected_th"]: their scores are no
# probabilities. The candidates are 0 / 1; the rule set detects whenever one of its rules
# fires (its scores are their precisions, >= the level, mostly below 0.5).
FIXED_THRESHOLDS = {"candidate": 0.5, "rule_set": 1e-6}
# The scorers that read the memory, compared under a truncated history.
HISTORY_SCORERS = ("model", "net_pruned", "rule_set", "motif_net", "mlp")
# Plot color of each scorer.
SCORER_COLORS = {"model": "tab:blue", "mlp": "tab:green", "anchor": "tab:orange", "prior": "tab:gray",
                 "candidate": "tab:purple", "cascade": "tab:brown", "net_pruned": "tab:cyan",
                 "rule_set": "tab:olive", "motif_net": "tab:pink"}
SCORER_LABELS = {"model": "rule net", "mlp": "MLP", "anchor": "anchor", "prior": "prior",
                 "candidate": "candidates", "cascade": "cascade", "net_pruned": "net pruned",
                 "rule_set": "rule set", "motif_net": "motif net"}
# Event recall levels of the operating points read off the test curves.
RECALL_LEVELS = (0.3, 0.5, 0.7)
# Event types by the frequency of their head (`step05.head_event_types`). The noise heads
# are no training targets (step05 exclude_noise_heads): the overall metrics count only the
# COUNTED_TYPES, the noise events are reported on their own (by_type).
EVENT_TYPES = step05.EVENT_TYPES
COUNTED_TYPES = ("rare", "regular", "trivial")
TYPE_COLORS = {"noise": "tab:gray", "rare": "tab:purple", "regular": "tab:blue", "trivial": "tab:orange"}


def _subset(data, video_ids):
    """`prepare_videos` data of only the given videos."""
    return {**data, "videos": {v: data["videos"][v] for v in video_ids}}


def _own_inputs(other, context, num_rows):
    """The inputs of a model with its own input layers (own_inputs) for the rows of
    the current chunk (context: X, frames, known, min_off_frames); same row
    selection as `step05.video_rows`, so the rows match the main model's."""
    inputs = step05.video_rows(other, context["X"], context["frames"], context["min_off_frames"],
                               context["known"])["inputs"]
    assert len(inputs) == num_rows, "the rows of the two models differ"
    return inputs


def _scorers(model, baselines, mlp=None, cascade=None, device="cpu", names=None, extra=None, context=None):
    """{name: f(inputs, stage1) -> (B, W, A) probabilities}; stage1 is the model's own;
    extra {name: model} adds models of the same inputs (e.g. net_pruned), scorers of
    the inputs (the rule set) or models with their own inputs (the motif net, built
    from context, the current chunk set by `_score_videos`)."""
    baselines = {name: value.to(device) for name, value in baselines.items()}
    scorers = {
        "model": lambda inputs, stage1: stage1,
        "anchor": lambda inputs, stage1: _anchor_scores(model, inputs, baselines),
        "prior": lambda inputs, stage1: baselines["prior"].expand(len(inputs), -1, -1),
    }
    if mlp is not None:
        # Same input layers as the TemporalRuleNet, so it scores the same rows.
        scorers["mlp"] = lambda inputs, stage1: mlp.window_probs(mlp(inputs)[0])
    if cascade is not None:
        verifier, thresholds = cascade["verifier"], cascade["thresholds"].to(device)
        scorers["candidate"] = lambda inputs, stage1: (stage1[:, -1:] >= thresholds).float().expand_as(stage1)
        scorers["cascade"] = lambda inputs, stage1: torch.where(
            stage1[:, -1:] >= thresholds, verifier.window_probs(verifier(inputs)[0]), 0.0)
    for name, other in (extra or {}).items():
        if getattr(other, "own_inputs", False):
            scorers[name] = lambda inputs, stage1, other=other: other.window_probs(
                other(_own_inputs(other, context, len(inputs)))[0])
        elif hasattr(other, "window_probs"):
            scorers[name] = lambda inputs, stage1, other=other: other.window_probs(other(inputs)[0])
        else:                                                    # a scorer of the inputs, e.g. the rule set
            scorers[name] = lambda inputs, stage1, other=other: other(inputs)
    return {name: f for name, f in scorers.items() if names is None or name in names}


@torch.no_grad()
def _score_videos(model, data, baselines, cfg, mlp=None, cascade=None, history=None, names=None, extra=None,
                  alarm_filter=None):
    """Window scores of the model and of the baselines on every video of the split.

    alarm_filter (k, g) (default cfg["alarm_filter"], else (1, 1): every frame on its own):
    the alarms and the event detection read the alarm score, the lowest score of the
    last k frames (a threshold is reached only after k frames in a row above it; a
    frame of the k that is not a row or not at risk gives no alarm score), and an alarm
    starts only if no frame of the g before scored as high (alarms g frames apart merge).
    The row histograms (AP, calibration) and the trajectory read the raw scores.

    Rows are the frames t >= cfg["min_context"]; history: if given, every row
    knows only that many frames before t (the older memory is unknown), else all
    of them. Returns
      num_frames: the scored frames of all videos;
      rows {scorer: {"pos", "neg": (W, A, S), "score_sum": (W, S), "pos_sum",
          "neg_sum": (W,)}}: histograms (S bins of `score_edges`) of the scores of
          the at-risk rows with window k observed, split by whether the atom
          switched on within W_k; score_sum the sum of the scores per bin
          (calibration); pos_sum / neg_sum the summed scores of the two groups;
      starts {scorer: {"pos", "neg": (W, S), "by_type": {type: {"pos", "neg"}}}}:
          alarms at each threshold bin b, rows whose alarm score is in bin >= b while
          the g frames before were below b (or not scored), split by whether the onset
          followed within W_k, on the counted heads (COUNTED_TYPES); by_type the
          alarms on the heads of each event type (cfg["head_types"]); per_head
          {"pos", "neg": (A, W, S)} those of each head;
      events {scorer: {...}}: the event detection (`_event_stats`) of every exact
          onset (on in t_e, off in the min_off_frames frames before, the agent
          visible in t_e - 1, t_e - 1 >= min_context), from the window-k scores of
          its agent and head over its at-risk stretch: the frames t_e - 1,
          t_e - 2, ... back to the first one that is not a row or not at risk.
          A threshold bin b detects the event if the alarm score of some frame of
          the stretch is in bin >= b; its prediction time is the lead t_e - t of the earliest such
          frame t and its coverage the share of the frames t..t_e - 1 that do:
            num_events: int; lookback_sum (W,): the summed stretch lengths;
            detected, lead_sum, coverage_sum (W, S): per threshold bin, the
                detected events and the summed prediction times (detected events)
                and coverages (0 for the missed ones);
            lead_at, count_at (E, W) long: per event, the prediction time (0 if
                missed) and the number of detected frames at the scorer's
                detection threshold (`detection_threshold`);
            by_type {type: the same of the events of that type};
            per_head {"num_events": (A,), "detected", "lead_sum", "coverage_sum":
                (A, W, S)}: the same sums per head (every event);
      trajectory {scorer: {"sum", "count": (top, W)}}: the summed score of the
          onset's own head and agent j frames before the onset, over all onsets
          (the confidence approaching an event).
    """
    device = cfg["device"]
    edges = score_edges().to(device)
    S, A = len(edges), data["num_atoms"]
    W, top = len(model.windows), max(model.windows)
    min_context = cfg["min_context"]
    context = {}
    scorers = _scorers(model, baselines, mlp, cascade, device, names, extra, context)
    persist, gap = alarm_filter or cfg.get("alarm_filter") or (1, 1)
    threshold_bins = {name: _threshold_bin(detection_threshold(name, cfg)) for name in scorers}
    head_type = cfg["head_types"].to(device)                                         # (A,) EVENT_TYPES index
    counted_index = [EVENT_TYPES.index(t) for t in COUNTED_TYPES]
    counted = torch.isin(head_type, torch.tensor(counted_index, device=device))       # (A,) heads of the overall metrics
    heads = torch.arange(A, device=device)
    window_index = torch.arange(W, device=device)[None, :, None]

    def zeros(*shape):
        return torch.zeros(shape, device=device)

    rows_hist = {name: {"pos": zeros(W * A * S), "neg": zeros(W * A * S), "score_sum": zeros(W * S),
                        "pos_sum": zeros(W), "neg_sum": zeros(W)} for name in scorers}
    starts = {name: {"pos": zeros(A * W * (S + 1)), "neg": zeros(A * W * (S + 1))} for name in scorers}
    trajectory = {name: {"sum": zeros(top, W), "count": zeros(top, W)} for name in scorers}

    def event_stats():
        return {"num_events": 0, "lookback_sum": zeros(W), "detected": zeros(W, S), "lead_sum": zeros(W, S),
                "coverage_sum": zeros(W, S), "lead_at": [], "count_at": []}

    def head_stats():
        return {"num_events": zeros(A), "detected": zeros(A, W, S), "lead_sum": zeros(A, W, S),
                "coverage_sum": zeros(A, W, S)}

    events = {name: {**event_stats(), "by_type": {t: event_stats() for t in EVENT_TYPES}, "per_head": head_stats()}
              for name in scorers}
    num_frames = 0
    for video_id in tqdm(data["videos"], desc="Scoring videos" + ("" if history is None else f" (history {history})")):
        X = step05._video_from_cells(data["videos"][video_id], A, device)
        T, N, _A = X.shape
        num_frames += max(T - min_context, 0)
        # Exact onsets and, per onset, the frames t_e - j (j = 0..T - 1) before it.
        onsets = step05.atom_onsets(X, data["min_off_frames"])
        onsets[0] = False
        onsets[1:] &= X[:-1].any(dim=2, keepdim=True)
        onsets[:min_context + 1] = False                                             # t_e - 1 must be a row
        e_t, e_n, e_a = torch.nonzero(onsets, as_tuple=True)
        e_lead = e_t[:, None] - torch.arange(T, device=device)                       # (E, T)
        e_raw = {name: torch.full((len(e_t), T, W), -1.0, device=device) for name in scorers}
        e_alarm = ({name: torch.full((len(e_t), T, W), -1.0, device=device) for name in scorers}
                   if persist > 1 else e_raw)
        # The raw scores of the persist - 1 frames before the current chunk and the alarm scores
        # of the gap frames before it (-1: not a row), for the alarm scores and starts.
        raw_before = {name: torch.full((persist - 1, N, W, A), -1.0, device=device) for name in scorers}
        alarm_before = {name: torch.full((gap, N, W, A), -1.0, device=device) for name in scorers}
        for begin in range(min_context, T, cfg["frames_per_video"]):
            frames = torch.arange(begin, min(begin + cfg["frames_per_video"], T), device=device)
            F = len(frames)
            known = None if history is None else frames.clamp(max=history)
            rows = step05.video_rows(model, X, frames, data["min_off_frames"], known)
            context.update(X=X, frames=frames, known=known, min_off_frames=data["min_off_frames"])
            t, n = rows["rows"].T
            valid = rows["at_risk"][:, None, :] & rows["known"]                     # (B, W, A)
            targets = rows["targets"]
            row_of = torch.full((F, N), -1, dtype=torch.long, device=device)
            row_of[t - begin, n] = torch.arange(len(t), device=device)
            e_in = (e_lead >= begin) & (e_lead < begin + F)
            e_row = torch.where(e_in, row_of[(e_lead - begin).clamp(0, F - 1), e_n[:, None]], -1)
            e_valid = (e_row >= 0) & (rows["at_risk"][e_row.clamp(min=0), e_a[:, None]] if len(t) else e_row >= 0)
            stage1 = model.window_probs(model(rows["inputs"])[0]) if len(t) else None
            for name, score in scorers.items():
                dense = torch.full((F, N, W, A), -1.0, device=device)
                p = None
                if len(t):
                    p = score(rows["inputs"], stage1)
                    dense[t - begin, n] = torch.where(valid, p, -1.0)
                # Alarm scores: the lowest raw score of the last `persist` frames.
                raw_window = torch.cat([raw_before[name], dense])                    # (persist - 1 + F, N, W, A)
                raw_before[name] = raw_window[len(raw_window) - (persist - 1):]
                alarm = raw_window.unfold(0, persist, 1).amin(dim=-1) if persist > 1 else dense
                window = torch.cat([alarm_before[name], alarm])                      # (gap + F, N, W, A)
                alarm_before[name] = window[-gap:]
                if p is None:
                    continue
                bins = _score_bins(p, edges)                                         # (B, W, A)
                cell = (window_index * A + heads) * S + bins
                h = rows_hist[name]
                h["pos"] += torch.bincount(cell[valid & targets], minlength=W * A * S)
                h["neg"] += torch.bincount(cell[valid & ~targets], minlength=W * A * S)
                counted_valid = valid & counted                                      # calibration: counted heads
                h["score_sum"] += torch.bincount((window_index * S + bins)[counted_valid].flatten(),
                                                 weights=p[counted_valid], minlength=W * S)
                h["pos_sum"] += torch.where(counted_valid & targets, p, 0.0).sum(dim=(0, 2))
                h["neg_sum"] += torch.where(counted_valid & ~targets, p, 0.0).sum(dim=(0, 2))
                # Alarms: the threshold bins b with previous bin < b <= current bin (a difference array).
                now = alarm[t - begin, n]
                before = window[:-1].unfold(0, gap, 1).amax(dim=-1)[t - begin, n]   # best of the gap frames before
                high = torch.where(valid & (now >= 0), _score_bins(now.clamp(min=0), edges), -1)
                low = torch.where(before >= 0, _score_bins(before.clamp(min=0), edges), -1)
                up = high > low
                base = ((heads[None, None, :] * W + window_index) * (S + 1)).expand_as(bins)
                for key, mask in (("pos", up & targets), ("neg", up & ~targets)):
                    starts[name][key] += (torch.bincount(base[mask] + low[mask] + 1, minlength=A * W * (S + 1))
                                          - torch.bincount(base[mask] + high[mask] + 1, minlength=A * W * (S + 1)))
                index = (e_row.clamp(min=0), slice(None), e_a[:, None])
                e_raw[name] = torch.where(e_valid[..., None], p[index], e_raw[name])
                if persist > 1:
                    e_alarm[name] = torch.where(e_valid[..., None], now[index], e_alarm[name])
        e_type = head_type[e_a]
        e_counted = counted[e_a]
        for name in scorers:
            raw = e_raw[name]
            seen = (raw[:, :top] >= 0) & e_counted[:, None, None]
            lead = seen.shape[1]
            trajectory[name]["sum"][:lead] += torch.where(seen, raw[:, :top], 0.0).sum(dim=0)
            trajectory[name]["count"][:lead] += seen.sum(dim=0)
            detections = _detect(e_alarm[name][:, 1:], edges)
            _event_stats(events[name], detections, e_counted, threshold_bins[name])
            for g, event_type in enumerate(EVENT_TYPES):
                _event_stats(events[name]["by_type"][event_type], detections, e_type == g, threshold_bins[name])
            _head_event_stats(events[name]["per_head"], detections, e_a)

    for e in (stats for scorer in events.values() for stats in (scorer, *scorer["by_type"].values())):
        e["lead_at"] = torch.cat(e["lead_at"]).cpu() if e["lead_at"] else torch.zeros((0, W), dtype=torch.long)
        e["count_at"] = torch.cat(e["count_at"]).cpu() if e["count_at"] else torch.zeros((0, W), dtype=torch.long)
        for key in ("lookback_sum", "detected", "lead_sum", "coverage_sum"):
            e[key] = e[key].cpu()
    for scorer in events.values():
        scorer["per_head"] = {key: v.cpu() for key, v in scorer["per_head"].items()}

    head_type_cpu, counted_cpu = head_type.cpu(), counted.cpu()

    def alarm_starts(v):                                                             # (A * W * (S + 1),) -> (A, W, S)
        return v.view(A, W, S + 1).cumsum(dim=2)[:, :, :S].cpu()
    return {
        "num_frames": num_frames,
        "rows": {name: {"pos": h["pos"].view(W, A, S).cpu(), "neg": h["neg"].view(W, A, S).cpu(),
                        "score_sum": h["score_sum"].view(W, S).cpu(), "pos_sum": h["pos_sum"].cpu(),
                        "neg_sum": h["neg_sum"].cpu()} for name, h in rows_hist.items()},
        "starts": {name: {**{key: alarm_starts(v)[counted_cpu].sum(dim=0) for key, v in s.items()},
                          "by_type": {t: {key: alarm_starts(v)[head_type_cpu == g].sum(dim=0) for key, v in s.items()}
                                      for g, t in enumerate(EVENT_TYPES)},
                          "per_head": {key: alarm_starts(v) for key, v in s.items()}}
                   for name, s in starts.items()},
        "events": events,
        "trajectory": {name: {key: v.cpu() for key, v in tr.items()} for name, tr in trajectory.items()},
    }


def _detect(scores, edges):
    """Detection of one video's events at every threshold bin.

    scores (E, L, W): the window scores of each event's agent and head j = 1..L
    frames before its onset, -1 where the frame is not a row or the head is not
    at risk; the at-risk stretch is the frames before the first such gap.
    Returns earliest (E, W, S) long: the prediction time at threshold bin b (the
    largest j of the stretch scoring in bin >= b, 0 if missed), count (E, W, S):
    the frames of the stretch scoring in bin >= b, lookback (E, W): the stretch length."""
    E, L, W = scores.shape
    S = len(edges)
    device = scores.device
    if E == 0 or L == 0:
        return (torch.zeros((E, W, S), dtype=torch.long, device=device),
                torch.zeros((E, W, S), dtype=torch.long, device=device), torch.zeros((E, W), device=device))
    stretch = (scores >= 0).long().cumprod(dim=1).bool()                             # (E, L, W)
    bins = _score_bins(scores.clamp(min=0), edges)
    cell = ((torch.arange(E, device=device)[:, None, None] * W + torch.arange(W, device=device)) * S + bins)[stretch]
    leads = torch.arange(1, L + 1, device=device)[None, :, None].expand(E, L, W)[stretch]
    # Per event and window, the detected frames and the earliest one at every threshold bin b (score bin >= b).
    count = torch.bincount(cell, minlength=E * W * S).view(E, W, S).flip(2).cumsum(2).flip(2)
    earliest = torch.zeros(E * W * S, dtype=torch.long, device=device).scatter_reduce(0, cell, leads, "amax")
    earliest = earliest.view(E, W, S).flip(2).cummax(dim=2).values.flip(2)          # 0: not detected
    return earliest, count, stretch.sum(dim=1).float()


def _event_stats(stats, detections, keep, threshold_bin):
    """Add the `_detect` detections of one video's events where keep (E,) bool to
    stats (see `_score_videos`)."""
    earliest, count, lookback = (d[keep] for d in detections)
    stats["num_events"] += len(earliest)
    if len(earliest) == 0:
        return
    stats["lookback_sum"] += lookback.sum(dim=0)
    stats["detected"] += (earliest > 0).sum(dim=0)
    stats["lead_sum"] += earliest.sum(dim=0)
    stats["coverage_sum"] += (count / earliest.clamp(min=1)).sum(dim=0)
    stats["lead_at"].append(earliest[:, :, threshold_bin])
    stats["count_at"].append(count[:, :, threshold_bin])


def _head_event_stats(stats, detections, event_heads):
    """Add the `_detect` detections of one video's events to the per-head sums of
    stats, by the head (E,) of each event."""
    earliest, count, _lookback = detections
    if len(earliest) == 0:
        return
    stats["num_events"].index_add_(0, event_heads, torch.ones(len(event_heads), device=event_heads.device))
    stats["detected"].index_add_(0, event_heads, (earliest > 0).float())
    stats["lead_sum"].index_add_(0, event_heads, earliest.float())
    stats["coverage_sum"].index_add_(0, event_heads, count / earliest.clamp(min=1))


def _histogram_metrics(positives, negatives, threshold, with_curve=False):
    """Onset metrics of one head (or all heads summed) from its score histograms
    over the `score_edges` bins: average precision, base rate, metrics at the
    probability threshold and at the threshold with the best F1; with_curve adds
    the precision-recall curve, one point per score bin from the highest threshold down."""
    edges = score_edges()
    num_bins = len(positives)
    # Predicting "on" for every score bin >= b, for b from the highest bin down.
    tp = positives.flip(0).cumsum(0)
    fp = negatives.flip(0).cumsum(0)
    num_positives, num_rows = positives.sum().item(), (positives + negatives).sum().item()
    precision = tp / (tp + fp).clamp(min=1)
    average_precision = (precision * positives.flip(0)).sum().item() / max(num_positives, 1)

    def at_bin(b):
        tp_b, fp_b = positives[b:].sum().item(), negatives[b:].sum().item()
        return compute_metrics(tp_b, fp_b, num_positives - tp_b)

    # F1 of flagging every bin >= b, for each b (first bin of the best F1 on ties, as max() did).
    tp_at = positives.flip(0).cumsum(0).flip(0)
    fp_at = negatives.flip(0).cumsum(0).flip(0)
    f1 = 2 * tp_at / (2 * tp_at + fp_at + (num_positives - tp_at)).clamp(min=1e-12)
    best_bin = int(f1.argmax()) if num_bins else 0
    curve = {}
    if with_curve:
        curve = {"curve": {
            "threshold": edges.flip(0).tolist(),
            "precision": precision.tolist(),
            "recall": (tp / max(num_positives, 1)).tolist(),
        }}
    return {
        "onsets": int(num_positives),
        "at_risk": int(num_rows),
        "base_rate": num_positives / num_rows if num_rows else 0.0,
        "average_precision": average_precision,
        "at_threshold": at_bin(int(torch.searchsorted(edges, torch.tensor(threshold), right=True)) - 1),
        "best_f1": {**at_bin(best_bin), "threshold": edges[best_bin].item()},
        **curve,
    }


def _detection(positives, negatives, threshold, minutes):
    """Event counts of (A, S) histograms flagged at the threshold: events (onsets),
    detected, missed, false_alarms, recall, precision, false_alarms_per_minute
    (used by step06_layer, whose false alarms are rows)."""
    b = int(torch.searchsorted(score_edges(), torch.tensor(threshold), right=True)) - 1
    events, detected, false_alarms = int(positives.sum()), int(positives[:, b:].sum()), int(negatives[:, b:].sum())
    return {
        "threshold": threshold,
        "events": events,
        "detected": detected,
        "missed": events - detected,
        "false_alarms": false_alarms,
        "recall": detected / events if events else 0.0,
        "precision": detected / (detected + false_alarms) if detected + false_alarms else 0.0,
        "false_alarms_per_minute": false_alarms / minutes if minutes else None,
    }


def _detection_curve(events, k, starts_pos, starts_neg):
    """Per threshold bin b, window k: detected events (some frame of the at-risk
    stretch before the onset scores in bin >= b), their recall, the mean
    prediction time of the detected events (frames), the mean coverage of all
    events (0 if missed) and of the detected ones; the alarms, correct alarms and
    alarm precision."""
    num_events = events["num_events"]
    detected = events["detected"][k]
    alarms = starts_pos + starts_neg
    return {"events": num_events, "detected": detected, "recall": detected / max(num_events, 1),
            "prediction_time": torch.where(detected > 0, events["lead_sum"][k] / detected.clamp(min=1), float("nan")),
            "coverage": events["coverage_sum"][k] / max(num_events, 1),
            "coverage_detected": torch.where(detected > 0, events["coverage_sum"][k] / detected.clamp(min=1),
                                             float("nan")),
            "alarms": alarms, "correct_alarms": starts_pos, "precision": starts_pos / alarms.clamp(min=1)}


def _nan_to_none(value):
    return None if value != value else value


def _detection_at(curve, b, threshold, minutes, frame_rate):
    alarms, correct = curve["alarms"][b].item(), curve["correct_alarms"][b].item()
    prediction_time = _nan_to_none(curve["prediction_time"][b].item())
    return {
        "threshold": threshold,
        "events": int(curve["events"]),
        "detected": int(curve["detected"][b]),
        "missed": int(curve["events"] - curve["detected"][b]),
        "recall": curve["recall"][b].item(),
        "mean_prediction_time": prediction_time,
        "mean_prediction_seconds": None if prediction_time is None else prediction_time / frame_rate,
        "coverage": curve["coverage"][b].item(),
        "coverage_detected": _nan_to_none(curve["coverage_detected"][b].item()),
        "alarms": int(alarms),
        "false_alarms": int(alarms - correct),
        "precision": curve["precision"][b].item(),
        "alarms_per_minute": alarms / minutes if minutes else None,
        "false_alarms_per_minute": (alarms - correct) / minutes if minutes else None,
    }


def _event_values(events, k):
    """Per event of window k at the scorer's detection threshold: prediction time
    (frames, None if missed) and coverage (0 if missed)."""
    lead, count = events["lead_at"][:, k], events["count_at"][:, k]
    return {"prediction_time": [int(x) if x > 0 else None for x in lead.tolist()],
            "coverage": (count / lead.clamp(min=1)).tolist()}


HEAD_THRESHOLD_SOURCES = ("head", "type", "global")


def select_head_thresholds(scores, cfg):
    """Per-head detection thresholds of every scorer and window, tuned on the
    `_score_videos` scores of the validation videos: the threshold bin with the
    best F1 of the head's event recall and alarm precision, if the head has at
    least cfg["per_head_min_events"] val events (and some F1 > 0); else the one of
    the pooled events and alarms of its event type, with the same condition; else
    cfg["detected_th"]. The scorers with FIXED_THRESHOLDS keep theirs.
    Returns {scorer: {"bins", "source": (W, A) long (HEAD_THRESHOLD_SOURCES index)}}."""
    head_type = cfg["head_types"]
    min_events = cfg["per_head_min_events"]
    global_bin = _threshold_bin(cfg["detected_th"])

    def best(detected, num_events, pos, neg):
        """Best-F1 bin and its F1 of (..., S) histograms, (...,) events."""
        recall = detected / num_events[..., None].clamp(min=1)
        precision = pos / (pos + neg).clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
        return f1.argmax(dim=-1), f1.amax(dim=-1)

    out = {}
    for name in (n for n in DETECTION_SCORERS if n in scores["events"] and n not in FIXED_THRESHOLDS):
        events, starts = scores["events"][name]["per_head"], scores["starts"][name]["per_head"]
        W, A = events["detected"].shape[1], events["detected"].shape[0]
        bins = torch.full((W, A), global_bin, dtype=torch.long)
        source = torch.full((W, A), HEAD_THRESHOLD_SOURCES.index("global"), dtype=torch.long)
        for k in range(W):
            for g in range(len(EVENT_TYPES)):
                of_type = head_type == g
                num_events = events["num_events"][of_type].sum()
                b, f1 = best(events["detected"][of_type, k].sum(dim=0), num_events,
                             starts["pos"][of_type, k].sum(dim=0), starts["neg"][of_type, k].sum(dim=0))
                if num_events >= min_events and f1 > 0:
                    bins[k, of_type] = b
                    source[k, of_type] = HEAD_THRESHOLD_SOURCES.index("type")
            b, f1 = best(events["detected"][:, k], events["num_events"], starts["pos"][:, k], starts["neg"][:, k])
            own = (events["num_events"] >= min_events) & (f1 > 0)
            bins[k, own] = b[own]
            source[k, own] = HEAD_THRESHOLD_SOURCES.index("head")
        out[name] = {"bins": bins, "source": source}
    return out


def _head_operating_point(events, starts, k, bins, heads, minutes, fps):
    """Detections of window k with a threshold bin per head (bins (A,)), summed over
    the heads where heads (A,) bool: event recall, prediction time, coverage and alarms."""
    index = torch.arange(len(bins))
    num_events = events["num_events"][heads].sum().item()
    detected = events["detected"][index, k, bins][heads].sum().item()
    lead = events["lead_sum"][index, k, bins][heads].sum().item()
    coverage = events["coverage_sum"][index, k, bins][heads].sum().item()
    correct = starts["pos"][index, k, bins][heads].sum().item()
    alarms = correct + starts["neg"][index, k, bins][heads].sum().item()
    prediction_time = lead / detected if detected else None
    return {
        "events": int(num_events),
        "detected": int(detected),
        "missed": int(num_events - detected),
        "recall": detected / num_events if num_events else 0.0,
        "mean_prediction_time": prediction_time,
        "mean_prediction_seconds": None if prediction_time is None else prediction_time / fps,
        "coverage": coverage / num_events if num_events else 0.0,
        "coverage_detected": coverage / detected if detected else None,
        "alarms": int(alarms),
        "false_alarms": int(alarms - correct),
        "precision": correct / alarms if alarms else 0.0,
        "alarms_per_minute": alarms / minutes if minutes else None,
        "false_alarms_per_minute": (alarms - correct) / minutes if minutes else None,
    }


def _per_head_threshold_point(events, starts, k, thresholds, counted, cfg, minutes, fps):
    """The per-head-threshold operating point of window k: overall (the counted
    heads), by_type, and how many heads of each type use their own, their type's
    or the global threshold."""
    bins = thresholds["bins"][k]
    point = _head_operating_point(events, starts, k, bins, counted, minutes, fps)
    point["by_type"] = {}
    for g, event_type in enumerate(EVENT_TYPES):
        of_type = cfg["head_types"] == g
        point["by_type"][event_type] = {
            **_head_operating_point(events, starts, k, bins, of_type, minutes, fps),
            "threshold_sources": {source: int(((thresholds["source"][k] == i) & of_type).sum())
                                  for i, source in enumerate(HEAD_THRESHOLD_SOURCES)}}
    return point


# -------------- Alarm filter (persistence, merged alarms) --------------
def _event_f1(curve):
    """(S,) F1 of event recall and alarm precision per threshold bin of a `_detection_curve`."""
    recall, precision = curve["recall"], curve["precision"]
    return 2 * recall * precision / (recall + precision).clamp(min=1e-12)


def select_alarm_filter(model, val_data, baselines, cfg):
    """The alarm filter (k, g) of `_score_videos` with the best rule-net event F1 on the
    validation videos: for each (k, g) of cfg["alarm_filter_grid"], the best F1 over the
    thresholds of the counted heads' event recall and alarm precision, averaged over the
    windows. Returns (chosen (k, g), grid [{"persist", "gap", "mean_best_f1", "per_window"}])."""
    grid = []
    for persist in cfg["alarm_filter_grid"]["persist"]:
        for gap in cfg["alarm_filter_grid"]["gap"]:
            scores = _score_videos(model, val_data, baselines, cfg, names=("model",), alarm_filter=(persist, gap))
            per_window = []
            for k in range(len(model.windows)):
                curve = _detection_curve(scores["events"]["model"], k, scores["starts"]["model"]["pos"][k],
                                         scores["starts"]["model"]["neg"][k])
                f1 = _event_f1(curve)
                b = int(f1.argmax())
                per_window.append({"best_f1": f1[b].item(), "threshold": score_edges()[b].item(),
                                   "recall": curve["recall"][b].item(), "precision": curve["precision"][b].item()})
            grid.append({"persist": persist, "gap": gap, "per_window": per_window,
                         "mean_best_f1": sum(p["best_f1"] for p in per_window) / len(per_window)})
            print(f"[Alarm filter] val k={persist} g={gap}: mean best event F1 {grid[-1]['mean_best_f1']:.4f}")
    best = max(grid, key=lambda g: (g["mean_best_f1"], -g["persist"], -g["gap"]))
    return (best["persist"], best["gap"]), grid


def _alarm_filter_comparison(model, data, baselines, cfg, minutes, fps):
    """Rule-net detection on the test videos without (k = g = 1) and with the chosen alarm
    filter, per window: at the detection threshold and at the best event F1, overall and by type."""
    out = {}
    for label, alarm_filter in (("raw", (1, 1)), ("filtered", tuple(cfg["alarm_filter"]))):
        scores = _score_videos(model, data, baselines, cfg, names=("model",), alarm_filter=alarm_filter)
        events, starts = scores["events"]["model"], scores["starts"]["model"]
        out[label] = {"alarm_filter": list(alarm_filter), "windows": {}}
        for k, w in enumerate(model.windows):
            curve = _detection_curve(events, k, starts["pos"][k], starts["neg"][k])
            b = int(_event_f1(curve).argmax())
            th = detection_threshold("model", cfg)
            entry = {"at_threshold": _detection_at(curve, _threshold_bin(th), th, minutes, fps),
                     "best_f1": _detection_at(curve, b, score_edges()[b].item(), minutes, fps), "by_type": {}}
            for event_type in EVENT_TYPES:
                typed = _detection_curve(events["by_type"][event_type], k, starts["by_type"][event_type]["pos"][k],
                                         starts["by_type"][event_type]["neg"][k])
                entry["by_type"][event_type] = _detection_at(typed, _threshold_bin(th), th, minutes, fps)
            out[label]["windows"][str(w)] = entry
    return out


# -------------- Per-head calibration (Platt scaling on val) --------------
# The scorers whose scores are probabilities, calibrated per window and head.
CALIBRATED_SCORERS = ("model", "net_pruned", "motif_net", "mlp", "cascade")


def _bin_scores():
    """(S,) a representative score of each `score_edges` bin (its middle)."""
    edges = score_edges()
    return (edges + torch.cat([edges[1:], torch.ones(1)])) / 2


def _logit(p):
    p = p.clamp(1e-9, 1 - 1e-9)
    return torch.log(p) - torch.log1p(-p)


def _fit_platt(pos, neg, iterations=50, ridge=1e-2):
    """Platt scaling sigmoid(a * logit(score) + c) of every (..., S) pair of score
    histograms (the bin middles as the scores), by Newton steps on the count-weighted
    log loss with a small ridge toward the identity (a = 1, c = 0). Returns a, c (...)."""
    z = _logit(_bin_scores())
    shape = pos.shape[:-1]
    a, c = torch.ones(shape), torch.zeros(shape)
    n = pos + neg
    for _ in range(iterations):
        q = torch.sigmoid(a[..., None] * z + c[..., None])
        r = n * q - pos                                                              # dloss / dlogit per bin
        g_a, g_c = (r * z).sum(-1) + ridge * (a - 1), r.sum(-1) + ridge * c
        h = n * q * (1 - q)
        h_aa, h_ac, h_cc = (h * z * z).sum(-1) + ridge, (h * z).sum(-1), h.sum(-1) + ridge
        det = (h_aa * h_cc - h_ac ** 2).clamp(min=1e-12)
        a = a - (h_cc * g_a - h_ac * g_c) / det
        c = c - (h_aa * g_c - h_ac * g_a) / det
    return a, c


def fit_calibration(val_rows, cfg, windows):
    """Per scorer of CALIBRATED_SCORERS, window and head, the Platt scaling of its
    validation rows (`_score_videos` rows histograms): the head's own fit if it has
    at least cfg["calibration_min_events"] * W onset rows in window W (an event gives
    about W of them), else the fit of its event type's pooled rows (the same
    condition), else the identity; a fit with a <= 0 (not
    monotone) is replaced by the identity too.
    Returns {scorer: {"a", "c": (W, A), "source": (W, A) long, CALIBRATION_SOURCES index}}."""
    head_type = cfg["head_types"]
    out = {}
    for name in (n for n in CALIBRATED_SCORERS if n in val_rows):
        pos, neg = val_rows[name]["pos"].float(), val_rows[name]["neg"].float()          # (W, A, S)
        W, A, _S = pos.shape
        min_rows = cfg["calibration_min_events"] * torch.tensor(windows, dtype=torch.float)       # (W,)
        a, c = torch.ones(W, A), torch.zeros(W, A)
        source = torch.full((W, A), CALIBRATION_SOURCES.index("identity"), dtype=torch.long)
        for g in range(len(EVENT_TYPES)):
            of_type = head_type == g
            if not of_type.any():
                continue
            type_pos, type_neg = pos[:, of_type].sum(dim=1), neg[:, of_type].sum(dim=1)      # (W, S)
            ta, tc = _fit_platt(type_pos, type_neg)
            enough = type_pos.sum(-1) >= min_rows                                          # (W,)
            use = enough[:, None] & of_type[None, :]
            a = torch.where(use, ta[:, None], a)
            c = torch.where(use, tc[:, None], c)
            source[use] = CALIBRATION_SOURCES.index("type")
        ha, hc = _fit_platt(pos, neg)
        own = pos.sum(-1) >= min_rows[:, None]
        a, c = torch.where(own, ha, a), torch.where(own, hc, c)
        source[own] = CALIBRATION_SOURCES.index("head")
        bad = ~(a > 0) | ~torch.isfinite(a) | ~torch.isfinite(c)
        a, c = torch.where(bad, 1.0, a), torch.where(bad, 0.0, c)
        source[bad] = CALIBRATION_SOURCES.index("identity")
        out[name] = {"a": a, "c": c, "source": source}
    return out


CALIBRATION_SOURCES = ("head", "type", "identity")


def _calibrated_histograms(pos, neg, a, c):
    """The (A, S) score histograms of one window after the calibration a, c (A,):
    each bin's rows move to the bin of its calibrated middle score. Returns pos, neg,
    score_sum (S,) (the summed calibrated scores per new bin)."""
    calibrated = torch.sigmoid(a[:, None] * _logit(_bin_scores())[None, :] + c[:, None])      # (A, S)
    new_bin = _score_bins(calibrated, score_edges())
    S = pos.shape[1]
    pos_c = torch.zeros_like(pos).scatter_add_(1, new_bin, pos)
    neg_c = torch.zeros_like(neg).scatter_add_(1, new_bin, neg)
    score_sum = torch.zeros(S).index_add_(0, new_bin.flatten(), ((pos + neg) * calibrated).flatten())
    return pos_c, neg_c, score_sum


def _calibrated_threshold_bins(a, c, level):
    """(A,) the raw threshold bin of each head at which its calibrated score reaches level:
    the lowest bin whose lower edge calibrates to >= level (the last bin + 1 if none)."""
    calibrated = torch.sigmoid(a[:, None] * _logit(score_edges())[None, :] + c[:, None])      # (A, S)
    reach = calibrated >= level
    S = reach.shape[1]
    return torch.where(reach.any(dim=1), reach.float().argmax(dim=1), S - 1)


def evaluate_calibration(scores, calibration, cfg, counted, windows, minutes, fps):
    """Per calibrated scorer and window on the test videos: the pooled AP and ECE of the
    counted heads before and after the per-head calibration, and the event detection
    when alarming at the calibrated probability levels cfg["calibrated_levels"]
    (overall, i.e. the counted heads, and by type)."""
    out = {}
    for name, cal in calibration.items():
        if name not in scores["rows"]:
            continue
        rows = scores["rows"][name]
        events, starts = scores["events"][name]["per_head"], scores["starts"][name]["per_head"]
        out[name] = {}
        for k, w in enumerate(windows):
            pos, neg = rows["pos"][k][counted], rows["neg"][k][counted]
            pos_c, neg_c, score_sum = _calibrated_histograms(pos, neg, cal["a"][k][counted], cal["c"][k][counted])
            raw = _histogram_metrics(pos.sum(dim=0), neg.sum(dim=0), cfg["prob_threshold"])
            calibrated = _histogram_metrics(pos_c.sum(dim=0), neg_c.sum(dim=0), cfg["prob_threshold"])
            entry = {
                "average_precision_raw": raw["average_precision"],
                "average_precision_calibrated": calibrated["average_precision"],
                "ece_raw": _calibration({"pos": pos, "neg": neg, "score_sum": rows["score_sum"][k]})["ece"],
                "ece_calibrated": _calibration({"pos": pos_c, "neg": neg_c, "score_sum": score_sum})["ece"],
                "head_sources": {src: int(((cal["source"][k] == i) & counted).sum())
                                 for i, src in enumerate(CALIBRATION_SOURCES)},
                "levels": {},
            }
            for level in cfg["calibrated_levels"]:
                bins = _calibrated_threshold_bins(cal["a"][k], cal["c"][k], level)
                point = _head_operating_point(events, starts, k, bins, counted, minutes, fps)
                point["by_type"] = {t: _head_operating_point(events, starts, k, bins, cfg["head_types"] == g,
                                                             minutes, fps) for g, t in enumerate(EVENT_TYPES)}
                entry["levels"][str(level)] = point
            out[name][str(w)] = entry
    return out


def detection_threshold(name, cfg):
    """The detection threshold of a scorer: cfg["detected_th"], or the fixed one of
    the scorers whose scores are no probabilities (FIXED_THRESHOLDS)."""
    return FIXED_THRESHOLDS.get(name, cfg["detected_th"])


def _threshold_bin(threshold):
    return int(torch.searchsorted(score_edges(), torch.tensor(threshold), right=True)) - 1


def _calibration(rows, num_buckets=10):
    """Reliability of the scores of all heads of one window: per equal-width score
    bucket the mean predicted probability, the observed onset share and the row
    count; ece the count-weighted mean gap."""
    edges = score_edges()
    bucket = (edges * num_buckets).long().clamp(max=num_buckets - 1)
    pos, neg = rows["pos"].sum(dim=0), rows["neg"].sum(dim=0)
    count = torch.zeros(num_buckets).index_add_(0, bucket, pos + neg)
    observed = torch.zeros(num_buckets).index_add_(0, bucket, pos)
    score_sum = torch.zeros(num_buckets).index_add_(0, bucket, rows["score_sum"])
    mean_predicted = score_sum / count.clamp(min=1)
    observed = observed / count.clamp(min=1)
    ece = ((count / count.sum().clamp(min=1)) * (observed - mean_predicted).abs()).sum().item()
    return {"mean_predicted": mean_predicted.tolist(), "observed": observed.tolist(), "count": count.tolist(),
            "ece": ece}


def evaluate_model(model, data, baselines, cfg, mlp=None, cascade=None, extra=None, head_thresholds=None,
                   calibration=None):
    """Window metrics of the model and of the baselines on the test split.

    Returns {"windows": {w: {"rows": {scorer: AP, macro AP, base rate, best F1,
    calibration, the precision-recall curve}, "detection": {scorer: event recall,
    prediction time, coverage and alarms at the detection threshold
    (`detection_threshold`), "events": the per-event prediction times and
    coverages there, "at_recall": the best alarm precision and alarm rates at
    event recall >= RECALL_LEVELS, "curve": the same per threshold bin}}},
    "per_head": AP of each head at the headline window, against the baselines,
    "trajectory": the mean score of the onset's head j frames before it, per
    window, "history": per known history in cfg["history_levels"] (and the full
    memory) and window, AP, calibration, the mean score of the onset and the other
    rows and the detections of the scorers reading the memory}, plus the settings.
    head_thresholds (`select_head_thresholds`): adds "per_head_threshold" to the
    detection of those scorers, the operating point with a threshold per head
    (overall and by_type), and "head_thresholds": the thresholds and their source per
    scorer, window and head. calibration (`fit_calibration`): adds "calibration"
    (`evaluate_calibration`). Every detection uses the alarm filter cfg["alarm_filter"];
    "alarm_filter" compares the rule net without and with it. "per_head_detection":
    each head's event detection per scorer and window ("reliable": at least
    cfg["per_head_min_test_events"] test events)."""
    scores = _score_videos(model, data, baselines, cfg, mlp, cascade, extra=extra)
    minutes = scores["num_frames"] / cfg["frame_rate"] / 60
    fps = cfg["frame_rate"]
    edges = score_edges()
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    out = {"num_videos": len(data["videos"]), "minutes": minutes, "min_off_frames": data["min_off_frames"],
           "min_context": cfg["min_context"], "windows": model.windows, "headline_window": cfg["headline_window"],
           "frame_rate": fps, "prob_threshold": cfg["prob_threshold"], "cascade": cfg["cascade"],
           "detected_th": cfg["detected_th"], "detection_metric": cfg["detection_metric"],
           "noise_below": cfg["event_types"]["noise_below"], "rare_below": cfg["event_types"].get("rare_below"),
           "trivial_above": cfg["event_types"]["trivial_above"],
           "exclude_noise_heads": cfg["exclude_noise_heads"], "counted_types": list(COUNTED_TYPES),
           "per_head_min_events": cfg["per_head_min_events"],
           "alarm_filter_grid": cfg["alarm_filter_grid"], "calibration_min_events": cfg["calibration_min_events"],
           "calibrated_levels": cfg["calibrated_levels"],
           "per_head_min_test_events": cfg["per_head_min_test_events"],
           "history_levels": cfg["history_levels"],
           "net_pruned": cfg.get("net_pruned", False), "rule_set": cfg.get("rule_set", False),
           "motif_net": cfg.get("motif_net", False),
           "threshold_from": f"detected_th {cfg['detected_th']} (fixed: {FIXED_THRESHOLDS})",
           "event_types": {key: v for key, v in cfg["event_types"].items() if key != "types"}, "results": {}}
    # The heads of the overall metrics: the regular and trivial ones (the noise heads are no training targets).
    counted = torch.isin(cfg["head_types"], torch.tensor([EVENT_TYPES.index(t) for t in COUNTED_TYPES]))

    def counted_rows(rows, k):
        return {"pos": rows["pos"][k][counted], "neg": rows["neg"][k][counted], "score_sum": rows["score_sum"][k]}

    for k, w in enumerate(model.windows):
        result = {"rows": {}, "detection": {}}
        for name, rows in scores["rows"].items():
            pos, neg = rows["pos"][k], rows["neg"][k]
            heads = torch.nonzero((pos.sum(dim=1) > 0) & counted).flatten().tolist()
            per_head = [_histogram_metrics(pos[h], neg[h], cfg["prob_threshold"])["average_precision"] for h in heads]
            result["rows"][name] = {
                **_histogram_metrics(pos[counted].sum(dim=0), neg[counted].sum(dim=0), cfg["prob_threshold"],
                                     with_curve=True),
                "macro_average_precision": sum(per_head) / len(per_head) if per_head else 0.0,
                "calibration": _calibration(counted_rows(rows, k)),
                "by_type": {},
            }
            for g, event_type in enumerate(EVENT_TYPES):
                of_type = cfg["head_types"] == g
                m = _histogram_metrics(pos[of_type].sum(dim=0), neg[of_type].sum(dim=0), cfg["prob_threshold"])
                result["rows"][name]["by_type"][event_type] = {
                    key: m[key] for key in ("onsets", "at_risk", "base_rate", "average_precision")}
        for name in (n for n in DETECTION_SCORERS if n in scores["events"]):
            events = scores["events"][name]
            curve = _detection_curve(events, k, scores["starts"][name]["pos"][k], scores["starts"][name]["neg"][k])
            threshold = detection_threshold(name, cfg)
            by_type = {}
            for event_type in EVENT_TYPES:
                typed = events["by_type"][event_type]
                starts = scores["starts"][name]["by_type"][event_type]
                typed_curve = _detection_curve(typed, k, starts["pos"][k], starts["neg"][k])
                times = sorted(t for t in _event_values(typed, k)["prediction_time"] if t is not None)
                by_type[event_type] = {
                    **_detection_at(typed_curve, _threshold_bin(threshold), threshold, minutes, fps),
                    "median_prediction_time": times[len(times) // 2] if times else None,
                    "events_at_threshold": _event_values(typed, k),
                    "curve": {"recall": typed_curve["recall"].tolist(),
                              "prediction_time": [_nan_to_none(x) for x in typed_curve["prediction_time"].tolist()],
                              "coverage": typed_curve["coverage"].tolist(),
                              "precision": typed_curve["precision"].tolist()},
                }
            at_recall = {}
            for level in RECALL_LEVELS:
                reach = torch.nonzero(curve["recall"] >= level).flatten()
                if len(reach):
                    b = int(reach[curve["precision"][reach].argmax()])
                    at_recall[str(level)] = _detection_at(curve, b, edges[b].item(), minutes, fps)
            values = _event_values(events, k)
            times = sorted(t for t in values["prediction_time"] if t is not None)
            result["detection"][name] = {
                **_detection_at(curve, _threshold_bin(threshold), threshold, minutes, fps),
                "median_prediction_time": times[len(times) // 2] if times else None,
                "mean_lookback": events["lookback_sum"][k].item() / max(events["num_events"], 1),
                "events_at_threshold": values,
                "by_type": by_type,
                **({"per_head_threshold": _per_head_threshold_point(
                    scores["events"][name]["per_head"], scores["starts"][name]["per_head"], k,
                    head_thresholds[name], counted, cfg, minutes, fps)} if name in (head_thresholds or {}) else {}),
                "at_recall": at_recall,
                "curve": {"threshold": edges.tolist(), "recall": curve["recall"].tolist(),
                          "prediction_time": [_nan_to_none(x) for x in curve["prediction_time"].tolist()],
                          "coverage": curve["coverage"].tolist(),
                          "precision": curve["precision"].tolist(),
                          "alarms_per_minute": (curve["alarms"] / max(minutes, 1e-9)).tolist()},
            }
        out["results"][str(w)] = result

    if head_thresholds:
        edges_list = edges.tolist()
        out["head_thresholds"] = {
            "min_events": cfg["per_head_min_events"],
            "labels": [step05._atom_name(a) for a in unique_atoms],
            "event_types": [EVENT_TYPES[int(g)] for g in cfg["head_types"]],
            "scorers": {name: {str(w): {"threshold": [edges_list[b] for b in t["bins"][k].tolist()],
                                        "source": [HEAD_THRESHOLD_SOURCES[x] for x in t["source"][k].tolist()]}
                               for k, w in enumerate(model.windows)}
                        for name, t in head_thresholds.items()},
        }

    k = model.windows.index(cfg["headline_window"])
    pos, neg = scores["rows"]["model"]["pos"][k], scores["rows"]["model"]["neg"][k]
    out["per_head"] = {}
    for h in torch.nonzero(pos.sum(dim=1)).flatten().tolist():
        m = _histogram_metrics(pos[h], neg[h], cfg["prob_threshold"])
        m["label"] = step05._atom_name(unique_atoms[h])
        m["event_type"] = EVENT_TYPES[int(cfg["head_types"][h])]
        m["baseline_average_precision"] = {
            name: _histogram_metrics(r["pos"][k][h], r["neg"][k][h], cfg["prob_threshold"])["average_precision"]
            for name, r in scores["rows"].items() if name != "model"}
        out["per_head"][str(h)] = m

    # Event detection of each head (with test events or alarms), per scorer and window, at the
    # detection threshold: event recall, mean prediction time, coverage, alarms and alarm precision.
    out["per_head_detection"] = {}
    for name in (n for n in DETECTION_SCORERS if n in scores["events"]):
        events, starts = scores["events"][name]["per_head"], scores["starts"][name]["per_head"]
        A = events["num_events"].shape[0]
        bins = torch.full((A,), _threshold_bin(detection_threshold(name, cfg)), dtype=torch.long)
        out["per_head_detection"][name] = {}
        for k, w in enumerate(model.windows):
            heads = {}
            for h in range(A):
                one = torch.zeros(A, dtype=torch.bool)
                one[h] = True
                point = _head_operating_point(events, starts, k, bins, one, minutes, fps)
                if point["events"] or point["alarms"]:
                    heads[str(h)] = {"label": step05._atom_name(unique_atoms[h]),
                                     "event_type": EVENT_TYPES[int(cfg["head_types"][h])],
                                     "reliable": point["events"] >= cfg["per_head_min_test_events"], **point}
            out["per_head_detection"][name][str(w)] = heads

    if calibration:
        out["calibration"] = evaluate_calibration(scores, calibration, cfg, counted, model.windows, minutes, fps)
    out["alarm_filter"] = {"chosen": list(cfg["alarm_filter"]), "val_grid": cfg.get("alarm_filter_val_grid"),
                           "test": _alarm_filter_comparison(model, data, baselines, cfg, minutes, fps)}

    out["trajectory"] = {name: {"mean": (tr["sum"] / tr["count"].clamp(min=1)).tolist(), "count": tr["count"].tolist()}
                         for name, tr in scores["trajectory"].items()
                         if name in ("model", "net_pruned", "rule_set", "motif_net", "mlp", "cascade")}

    # The same rows knowing only part of the memory: how the confidence grows with the known facts.
    out["history"] = {}
    for level in [*cfg["history_levels"], None]:
        partial = scores if level is None else _score_videos(model, data, baselines, cfg, mlp, cascade, level,
                                                             HISTORY_SCORERS, extra)
        key = str(cfg["min_context"] if level is None else level)
        out["history"][key] = {}
        for k, w in enumerate(model.windows):
            out["history"][key][str(w)] = {}
            for name in (n for n in HISTORY_SCORERS if n in partial["rows"]):
                rows = partial["rows"][name]
                pos, neg = rows["pos"][k][counted].sum(dim=0), rows["neg"][k][counted].sum(dim=0)
                curve = _detection_curve(partial["events"][name], k, partial["starts"][name]["pos"][k],
                                         partial["starts"][name]["neg"][k])
                threshold = detection_threshold(name, cfg)
                detection = _detection_at(curve, _threshold_bin(threshold), threshold, minutes, fps)
                out["history"][key][str(w)][name] = {
                    "average_precision": _histogram_metrics(pos, neg, cfg["prob_threshold"])["average_precision"],
                    "ece": _calibration(counted_rows(rows, k))["ece"],
                    "mean_onset_score": rows["pos_sum"][k].item() / max(pos.sum().item(), 1),
                    "mean_other_score": rows["neg_sum"][k].item() / max(neg.sum().item(), 1),
                    "recall": detection["recall"],
                    "mean_prediction_time": detection["mean_prediction_time"],
                    "coverage": detection["coverage"],
                    "precision": detection["precision"],
                    "alarms_per_minute": detection["alarms_per_minute"],
                    "recall_by_type": {
                        t: _detection_curve(partial["events"][name]["by_type"][t], k,
                                            partial["starts"][name]["by_type"][t]["pos"][k],
                                            partial["starts"][name]["by_type"][t]["neg"][k])["recall"][
                            _threshold_bin(threshold)].item() for t in EVENT_TYPES},
                }
    return out


# -------------- Clause evaluation --------------
def evaluate_clauses(model, data, clauses, step05_cfg):
    """The step 05 clauses with their test_ counts added (every body seen, head in
    the next slot): test_support (hits), test_fires, test_confidence,
    test_anchor_confidence (the anchor alone), test_recall (share of the head's
    onsets hit), test_base_rate; and in each "partial" entry test_confidence, the
    confidence knowing only the bodies more than unseen_slots slots before the
    head. Sorted by test confidence."""
    def ratio(a, b):
        return float(a) / float(b) if b > 0 else None

    counts = step05._rule_head_counts(model, data, list(data["videos"]), step05_cfg,
                                      desc="Counting rule firings (test)")
    evaluated = []
    for clause in clauses:
        r, h = clause["rule"], clause["head"][0]
        partial = [{**p, "test_confidence": ratio(counts["hits"][r, p["unseen_slots"], h],
                                                  counts["fires"][r, p["unseen_slots"], h]),
                    "test_base_rate": ratio(counts["onsets"][p["unseen_slots"], h],
                                            counts["at_risk"][p["unseen_slots"], h])}
                   for p in clause.get("partial", [])]
        evaluated.append({
            **clause,
            "partial": partial,
            "test_support": int(counts["hits"][r, 0, h]),
            "test_fires": int(counts["fires"][r, 0, h]),
            "test_confidence": ratio(counts["hits"][r, 0, h], counts["fires"][r, 0, h]),
            "test_anchor_confidence": ratio(counts["anchor_hits"][r, 0, h], counts["anchor_fires"][r, 0, h]),
            "test_recall": ratio(counts["hits"][r, 0, h], counts["onsets"][0, h]),
            "test_base_rate": ratio(counts["onsets"][0, h], counts["at_risk"][0, h]),
        })
    evaluated.sort(key=lambda c: -(c["test_confidence"] or 0))
    return evaluated


# -------------- Results --------------
def _strip_detection(d):
    """A detection entry without its curves and per-event values (also in by_type)."""
    out = {key: value for key, value in d.items() if key not in ("curve", "events_at_threshold", "by_type")}
    if "by_type" in d:
        out["by_type"] = {t: _strip_detection(v) for t, v in d["by_type"].items()}
    return out


def build_results(metrics, clauses):
    """The headline test numbers per window: AP of every scorer, the detections at
    the detection thresholds and the operating points at fixed event recall. The
    curves and per-head details are in test_metrics.json."""
    firing = [c for c in clauses if c["test_confidence"] is not None]
    extended = [c for c in firing if len(c["bodies"]) > 1 and c["test_anchor_confidence"] is not None]
    windows = {}
    for w, result in metrics["results"].items():
        windows[w] = {
            "base_rate": result["rows"]["model"]["base_rate"],
            "average_precision": {name: r["average_precision"] for name, r in result["rows"].items()},
            "average_precision_by_type": {name: {t: v["average_precision"] for t, v in r["by_type"].items()}
                                          for name, r in result["rows"].items()},
            "ece": {name: r["calibration"]["ece"] for name, r in result["rows"].items()},
            "detection": {name: _strip_detection(d) for name, d in result["detection"].items()},
        }
    return {
        "test_videos": metrics["num_videos"],
        "test_minutes": metrics["minutes"],
        "min_context": metrics["min_context"],
        "headline_window": metrics["headline_window"],
        "threshold_from": metrics["threshold_from"],
        "detection_metric": metrics["detection_metric"],
        "event_types": {t: [h["label"] for h in heads] for t, heads in metrics["event_types"]["heads"].items()},
        "windows": windows,
        "alarm_filter": metrics.get("alarm_filter"),
        "calibration": {name: {w: {key: v for key, v in e.items() if key != "levels"}
                               | {"levels": {q: {key: p[key] for key in ("recall", "precision", "alarms_per_minute",
                                                                          "mean_prediction_seconds")}
                                             for q, p in e["levels"].items()}}
                               for w, e in per_window.items()}
                        for name, per_window in metrics.get("calibration", {}).items()},
        "per_head_detection": {name: {w: {h: d for h, d in heads.items() if d["reliable"]}
                                      for w, heads in per_window.items()}
                               for name, per_window in metrics.get("per_head_detection", {}).items()},
        "history": metrics["history"],
        "clauses": {
            "num_clauses": len(clauses),
            "num_firing_on_test": len(firing),
            "num_extended_beating_anchor": f"{sum(c['test_confidence'] > c['test_anchor_confidence'] for c in extended)}"
                                           f"/{len(extended)}",
        },
    }


def plot_results(metrics, clauses, cfg, num_heads=30):
    """Save the step 06 figures, one PDF each, to cfg["figure_dir"]:
      ap_vs_window.pdf: AP of every scorer per window, against the base rate;
      detection_vs_window.pdf: event recall, mean prediction time, coverage (all
          events and the detected ones), alarm precision and false alarms per
          minute per window at the detection thresholds;
      detection_curve.pdf: event recall vs alarms per minute of every scorer
          (all thresholds), at the headline window;
      detection_threshold.pdf: event recall, mean prediction time and coverage
          over the threshold, at the headline window;
      detection_events.pdf: the distributions (cumulative) of the prediction time
          and coverage of the detected events at the headline window;
      event_types.pdf, detection_by_type.pdf, detection_by_type_vs_window.pdf,
          detection_threshold_by_type.pdf: per event type (`_plot_event_types`);
      per_head_threshold.pdf, per_head_threshold_vs_window.pdf, head_thresholds.pdf:
          the per-head thresholds against detected_th (`_plot_head_thresholds`);
      precision_recall_curve.pdf: row precision-recall at the headline window;
      calibration.pdf: predicted vs observed onset share of the rule net, per window;
      history.pdf: AP and mean onset confidence per known history (partial facts), per window;
      trajectory.pdf: mean confidence of an onset's head in the frames before it;
      ap_per_head.pdf: AP of each head at the headline window, against the baselines;
      compare_ap_per_head.pdf: rule net vs MLP AP per head;
      clause_train_vs_test.pdf, clause_extension_gain.pdf, clause_precision_recall.pdf:
          the step 05 clauses on the test videos.
    A figure is skipped (with a message) when its data is missing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = cfg["figure_dir"]
    os.makedirs(figure_dir, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figure_dir / name)
        plt.close(fig)
        print(f"[Figure]\t {figure_dir / name}")

    windows = metrics["windows"]
    results = metrics["results"]
    headline = str(metrics["headline_window"])
    seconds = [w / metrics["frame_rate"] for w in windows]
    row_names = list(results[str(windows[0])]["rows"])
    detection_names = [n for n in DETECTION_SCORERS if n in results[str(windows[0])]["detection"]]

    # AP per window.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name in row_names:
        ax.plot(seconds, [results[str(w)]["rows"][name]["average_precision"] for w in windows], "o-",
                color=SCORER_COLORS[name], label=SCORER_LABELS[name])
    ax.plot(seconds, [results[str(w)]["rows"]["model"]["base_rate"] for w in windows], "--", color="tab:red",
            label="base rate (random)")
    ax.set(title="Average precision of 'onset within W' per window", xlabel="window W (s)",
           ylabel="average precision (log)", yscale="log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    save(fig, "ap_vs_window.pdf")

    # Event detection and alarms at the detection thresholds per window.
    panels = (("recall", "event recall", 1), ("mean_prediction_seconds", "mean prediction time (s)", 1),
              ("coverage", "coverage (missed = 0)", 1), ("coverage_detected", "coverage of detected events", 1),
              ("precision", "alarm precision", 1), ("false_alarms_per_minute", "false alarms / min (log)", 1))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (key, label, _unit) in zip(axes.flat, panels):
        for name in detection_names:
            values = [results[str(w)]["detection"][name][key] for w in windows]
            ax.plot(seconds, [float("nan") if v is None else v for v in values], "o-", markersize=4,
                    color=SCORER_COLORS[name], label=SCORER_LABELS[name])
        ax.set(title=label, xlabel="window W (s)")
        if key == "false_alarms_per_minute":
            ax.set_yscale("log")
        elif key != "mean_prediction_seconds":
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Event detection at detected_th = {metrics['detected_th']} "
                 f"({metrics['minutes']:.1f} min of test video)")
    save(fig, "detection_vs_window.pdf")

    # Event recall vs alarms per minute at the headline window, every threshold.
    swept = [n for n in detection_names if n not in FIXED_THRESHOLDS]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name in swept:
        curve = results[headline]["detection"][name]["curve"]
        points = sorted((a, r) for a, r in zip(curve["alarms_per_minute"], curve["recall"]) if a > 0)
        ax.plot([a for a, _r in points], [r for _a, r in points], color=SCORER_COLORS[name], label=SCORER_LABELS[name])
        d = results[headline]["detection"][name]
        ax.plot(d["alarms_per_minute"] or 0, d["recall"], "o", color=SCORER_COLORS[name])
    ax.set(title=f"Event recall vs alarms, window {headline} frames (dots: detected_th)",
           xlabel="alarms per minute (log)", ylabel="event recall", xscale="log", ylim=(0, 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    save(fig, "detection_curve.pdf")

    # Recall, prediction time and coverage over the detection threshold at the headline window.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fps = metrics["frame_rate"]
    for name in swept:
        curve = results[headline]["detection"][name]["curve"]
        rows = [(t, r, pt, c) for t, r, pt, c in zip(curve["threshold"], curve["recall"], curve["prediction_time"],
                                                     curve["coverage"]) if t > 0]
        thresholds = [t for t, *_ in rows]
        style = dict(color=SCORER_COLORS[name], label=SCORER_LABELS[name])
        axes[0].plot(thresholds, [r for _t, r, _pt, _c in rows], **style)
        axes[1].plot(thresholds, [float("nan") if pt is None else pt / fps for _t, _r, pt, _c in rows], **style)
        axes[2].plot(thresholds, [c for _t, _r, _pt, c in rows], **style)
    for ax, label in zip(axes, ("event recall", "mean prediction time of detected events (s)",
                                "coverage (missed = 0)")):
        ax.axvline(metrics["detected_th"], color="k", linestyle="--", linewidth=0.8)
        ax.set(title=label, xlabel="detection threshold (log)", xscale="log", xlim=(1e-4, 1))
        ax.grid(alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Event detection over the threshold, window {headline} frames (dashed: detected_th)")
    save(fig, "detection_threshold.pdf")

    # Distributions over the events at the detection threshold, headline window.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in detection_names:
        d = results[headline]["detection"][name]
        values = d["events_at_threshold"]
        times = sorted(t / fps for t in values["prediction_time"] if t is not None)
        covered = sorted(c for t, c in zip(values["prediction_time"], values["coverage"]) if t is not None)
        if not times:
            continue
        label = f"{SCORER_LABELS[name]} ({len(times)}/{d['events']} detected)"
        share = [(i + 1) / len(times) for i in range(len(times))]
        axes[0].step(times, share, where="post", color=SCORER_COLORS[name], label=label)
        axes[1].step(covered, share, where="post", color=SCORER_COLORS[name], label=label)
    axes[0].set(title="Prediction time of the detected events", xlabel="seconds before the onset",
                ylabel="share of detected events (cumulative)", ylim=(0, 1))
    axes[1].set(title="Coverage of the detected events", xlabel="detected frames / prediction time",
                ylabel="share of detected events (cumulative)", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Detected events at detected_th = {metrics['detected_th']}, window {headline} frames")
    save(fig, "detection_events.pdf")

    _plot_event_types(metrics, plt, save, detection_names, swept, seconds, headline)
    _plot_head_thresholds(metrics, plt, save, swept, seconds, headline)

    # Row precision-recall at the headline window.
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    lows = []
    for name in row_names:
        m = results[headline]["rows"][name]
        points = [(r, p) for r, p in zip(m["curve"]["recall"], m["curve"]["precision"]) if r > 0]
        ax.plot([r for r, _p in points], [p for _r, p in points], color=SCORER_COLORS[name],
                label=f"{SCORER_LABELS[name]} (AP {m['average_precision']:.3f})")
        lows += [p for _r, p in points]
    base = results[headline]["rows"]["model"]["base_rate"]
    ax.axhline(base, color="tab:red", linestyle="--", label="base rate (random)")
    ax.set(title=f"Onset within {headline} frames: precision-recall of the rows", xlabel="recall",
           ylabel="precision (log)", xlim=(0, 1), ylim=(step05.log_floor(lows + [base]), 1), yscale="log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    save(fig, "precision_recall_curve.pdf")

    # Calibration of the rule net per window.
    fig, ax = plt.subplots(figsize=(5.5, 5))
    cmap = plt.get_cmap("viridis")
    for i, w in enumerate(windows):
        c = results[str(w)]["rows"]["model"]["calibration"]
        points = [(p, o) for p, o, n in zip(c["mean_predicted"], c["observed"], c["count"]) if n >= 20]
        ax.plot([p for p, _o in points], [o for _p, o in points], "o-", markersize=3, color=cmap(i / max(len(windows) - 1, 1)),
                label=f"{w} frames (ECE {c['ece']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set(title="Calibration of the rule net (buckets with >= 20 rows)", xlabel="predicted probability",
           ylabel="observed onset share", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=6)
    save(fig, "calibration.pdf")

    # Partial facts: AP and the mean confidence of the onsets by the known history.
    levels = sorted(int(h) for h in metrics["history"])
    shown = [w for w in windows if w in (windows[0], int(headline), windows[len(windows) // 2 - 1], windows[-1])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cmap = plt.get_cmap("viridis")
    for i, w in enumerate(shown):
        for name, style in (("model", "o-"), ("net_pruned", "^-."), ("rule_set", "d:"), ("motif_net", "*-"),
                            ("mlp", "s--")):
            if name not in metrics["history"][str(levels[0])][str(w)]:
                continue
            values = [metrics["history"][str(h)][str(w)][name] for h in levels]
            color = cmap(i / max(len(shown) - 1, 1))
            label = f"{SCORER_LABELS[name]}, within {w}"
            axes[0].plot(levels, [v["average_precision"] for v in values], style, color=color, label=label)
            axes[1].plot(levels, [v["mean_onset_score"] for v in values], style, color=color, label=label)
    axes[0].set(title="AP by the known history", xlabel="frames of history known", ylabel="average precision (log)",
                yscale="log")
    axes[1].set(title="Mean confidence of the true onsets", xlabel="frames of history known",
                ylabel="mean predicted probability")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=6)
    save(fig, "history.pdf")

    # The confidence approaching an onset: the mean score of its own head j frames before it.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, style in (("model", "-"), ("net_pruned", "-."), ("rule_set", (0, (1, 1))), ("motif_net", (0, (5, 1))),
                        ("mlp", "--"), ("cascade", ":")):
        if name not in metrics["trajectory"]:
            continue
        mean = torch.tensor(metrics["trajectory"][name]["mean"])                     # (top, W)
        count = torch.tensor(metrics["trajectory"][name]["count"])
        for i, w in enumerate(shown):
            k = windows.index(w)
            lead = [j for j in range(len(mean)) if count[j, k] >= 5]
            ax.plot([-j for j in lead], [mean[j, k].item() for j in lead], linestyle=style,
                    color=cmap(i / max(len(shown) - 1, 1)), label=f"{SCORER_LABELS[name]}, within {w}")
    ax.axvline(0, color="tab:red", linewidth=1)
    ax.set(title="Mean confidence of an event's head before it happens", xlabel="frames relative to the onset",
           ylabel="mean predicted probability")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    save(fig, "trajectory.pdf")

    # Per head at the headline window.
    learned = [m for m in metrics["per_head"].values() if m.get("event_type") != "noise"]   # noise heads: not trained
    heads = sorted(learned, key=lambda m: -m["average_precision"])[:num_heads]
    if heads:
        labels = [f"{m['label']} ({_short_num(m['onsets'])})" for m in heads]
        fig, ax = plt.subplots(figsize=(7, 0.25 * len(heads) + 1.5))
        step05.head_dot_plot(ax, labels,
                             [("rule net AP", [m["average_precision"] for m in heads], SCORER_COLORS["model"], "o")]
                             + [(f"{SCORER_LABELS[name]} AP", [m["baseline_average_precision"][name] for m in heads],
                                 SCORER_COLORS[name], marker)
                                for name, marker in (("net_pruned", "v"), ("rule_set", "P"), ("motif_net", "*"),
                                                     ("mlp", "s"), ("cascade", "^"), ("anchor", "D"))
                                if name in heads[0]["baseline_average_precision"]]
                             + [("base rate", [m["base_rate"] for m in heads], "tab:red", "|")],
                             reference=[m["base_rate"] for m in heads])
        ax.set(title=f"AP per head, window {headline} frames (top {len(heads)}, onset rows in brackets)",
               xlabel="average precision (log)")
        save(fig, "ap_per_head.pdf")

    heads = [m for m in learned if "mlp" in m["baseline_average_precision"]]
    if heads:
        xs = [m["baseline_average_precision"]["mlp"] for m in heads]
        ys = [m["average_precision"] for m in heads]
        low = step05.log_floor(xs + ys)
        largest = max(m["onsets"] for m in heads)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(step05.clip_low(xs, low), step05.clip_low(ys, low), s=[8 + 120 * m["onsets"] / largest for m in heads],
                   alpha=0.5, color=SCORER_COLORS["model"], clip_on=False)
        ax.plot([low, 1], [low, 1], "k--", linewidth=0.8)
        for m, x, y in sorted(zip(heads, xs, ys), key=lambda t: -t[0]["onsets"])[:8]:
            ax.annotate(m["label"], (max(x, low), max(y, low)), fontsize=6, xytext=(3, 3), textcoords="offset points")
        better = sum(y > x for x, y in zip(xs, ys))
        ax.set(title=f"AP per head, window {headline}: rule net better on {better}/{len(heads)}",
               xlabel="MLP AP (log)", ylabel="rule net AP (log)", xscale="log", yscale="log",
               xlim=(low, 1), ylim=(low, 1))
        save(fig, "compare_ap_per_head.pdf")

    # Clauses on the test videos.
    rows = [c for c in clauses if c["test_confidence"] is not None]
    if not rows:
        print("[Figure]\t skipped the clause figures: no clause fires on test")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    step05.clause_scatter(ax, rows, "confidence", "test_confidence")
    ax.set(title="Clause confidence, train vs test", xlabel="train confidence", ylabel="test confidence")
    save(fig, "clause_train_vs_test.pdf")

    extended = [c for c in rows if len(c["bodies"]) > 1 and c["test_anchor_confidence"] is not None]
    if extended:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        step05.clause_scatter(ax, extended, "test_anchor_confidence", "test_confidence")
        better = sum(c["test_confidence"] > c["test_anchor_confidence"] for c in extended)
        ax.set(title=f"Extension gain on test ({better}/{len(extended)} above y = x)",
               xlabel="confidence of the anchor alone", ylabel="confidence of the extended clause")
        save(fig, "clause_extension_gain.pdf")
    else:
        print("[Figure]\t skipped clause_extension_gain.pdf: no extended clause fires on test")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    low = step05.log_floor([c["test_confidence"] for c in rows])
    ax.scatter([c["test_recall"] or 0 for c in rows], step05.clip_low([c["test_confidence"] for c in rows], low),
               s=[4 + 60 * c["test_support"] / max(r["test_support"] for r in rows) for c in rows], alpha=0.5,
               clip_on=False)
    ax.set(title="Clause confidence vs recall on test (size = support)", xlabel="recall (share of head onsets hit)",
           ylabel="confidence (precision, log)", xlim=(0, 1), ylim=(low, 1), yscale="log")
    save(fig, "clause_precision_recall.pdf")


def _plot_event_types(metrics, plt, save, detection_names, swept, seconds, headline):
    """The figures of the event types (noise / regular / trivial heads by frequency):
      event_types.pdf: onsets per head (train + val), coloured by type;
      detection_by_type.pdf: AP, event recall, prediction time, coverage, alarm
          precision and false alarms per minute per type at the headline window;
      detection_by_type_vs_window.pdf: AP, recall, coverage and alarm precision per
          type and window;
      detection_threshold_by_type.pdf: event recall over the threshold per type."""
    if "event_types" not in metrics:
        print("[Figure]\t skipped the event type figures: no event types in the metrics")
        return
    types = metrics["event_types"]
    windows, results, fps = metrics["windows"], metrics["results"], metrics["frame_rate"]
    total = max(types["num_onsets"], 1)

    # Onsets per head, most frequent first, coloured by type.
    heads = sorted(((h["onsets"], t) for t, hs in types["heads"].items() for h in hs if h["onsets"] > 0), reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for event_type in EVENT_TYPES:
        ranks = [i for i, (_n, t) in enumerate(heads) if t == event_type]
        share = sum(heads[i][0] for i in ranks) / total
        ax.bar(ranks, [heads[i][0] for i in ranks], width=0.8, color=TYPE_COLORS[event_type],
               label=f"{event_type}: {len(types['heads'][event_type])} heads, {share:.0%} of onsets")
    ax.set(title=f"Onsets per head ({types['count_source']}); noise: < {types['noise_below']}, "
                 f"rare: < {types.get('rare_below')}, trivial: > {types['trivial_above']}",
           xlabel="head rank (most frequent first)", ylabel="onsets (log)", yscale="log")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    save(fig, "event_types.pdf")

    # Grouped bars per type at the headline window.
    def value(name, event_type, key, w=headline):
        if key == "average_precision":
            return results[w]["rows"][name]["by_type"][event_type]["average_precision"]
        v = results[w]["detection"][name]["by_type"][event_type][key]
        if v is not None and key == "mean_prediction_time":
            v /= fps
        return float("nan") if v is None else v

    panels = (("average_precision", "row AP (log)"), ("recall", "event recall"),
              ("mean_prediction_time", "mean prediction time (s)"), ("coverage", "coverage (missed = 0)"),
              ("precision", "alarm precision"), ("false_alarms_per_minute", "false alarms / min (log)"))
    width = 0.8 / len(detection_names)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (key, label) in zip(axes.flat, panels):
        for i, name in enumerate(detection_names):
            xs = [g + (i - (len(detection_names) - 1) / 2) * width for g in range(len(EVENT_TYPES))]
            ax.bar(xs, [value(name, t, key) for t in EVENT_TYPES], width=width * 0.9,
                   color=SCORER_COLORS[name], label=SCORER_LABELS[name])
        events = [results[headline]["detection"][detection_names[0]]["by_type"][t]["events"] for t in EVENT_TYPES]
        ax.set_xticks(range(len(EVENT_TYPES)), [f"{t}\n({n} events)" for t, n in zip(EVENT_TYPES, events)])
        ax.set(title=label)
        if key in ("average_precision", "false_alarms_per_minute"):
            ax.set_yscale("log")
        elif key != "mean_prediction_time":
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, axis="y")
    axes.flat[1].legend(fontsize=7)
    fig.suptitle(f"Per event type, window {headline} frames, detected_th = {metrics['detected_th']}")
    save(fig, "detection_by_type.pdf")

    # Per type and window.
    panels = (("average_precision", "row AP (log)"), ("recall", "event recall"),
              ("coverage", "coverage (missed = 0)"), ("precision", "alarm precision"))
    fig, axes = plt.subplots(len(EVENT_TYPES), len(panels), figsize=(15, 9.5), sharex=True)
    for row, event_type in enumerate(EVENT_TYPES):
        for col, (key, label) in enumerate(panels):
            ax = axes[row, col]
            for name in detection_names:
                ax.plot(seconds, [value(name, event_type, key, str(w)) for w in windows], "o-", markersize=3,
                        color=SCORER_COLORS[name], label=SCORER_LABELS[name])
            ax.set(title=f"{event_type}: {label}")
            if key == "average_precision":
                ax.set_yscale("log")
            else:
                ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            if row == len(EVENT_TYPES) - 1:
                ax.set_xlabel("window W (s)")
    axes[0, 1].legend(fontsize=6)
    fig.suptitle(f"Per event type and window, detected_th = {metrics['detected_th']}")
    save(fig, "detection_by_type_vs_window.pdf")

    # Event recall over the threshold per type.
    fig, axes = plt.subplots(1, len(EVENT_TYPES), figsize=(14, 4.2), sharey=True)
    for ax, event_type in zip(axes, EVENT_TYPES):
        for name in swept:
            d = results[headline]["detection"][name]
            points = [(t, r) for t, r in zip(d["curve"]["threshold"], d["by_type"][event_type]["curve"]["recall"])
                      if t > 0]
            ax.plot([t for t, _r in points], [r for _t, r in points], color=SCORER_COLORS[name],
                    label=SCORER_LABELS[name])
        ax.axvline(metrics["detected_th"], color="k", linestyle="--", linewidth=0.8)
        n = results[headline]["detection"][swept[0]]["by_type"][event_type]["events"] if swept else 0
        ax.set(title=f"{event_type} ({n} events)", xlabel="detection threshold (log)", xscale="log",
               xlim=(1e-4, 1), ylim=(0, 1))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("event recall")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Event recall over the threshold per event type, window {headline} frames (dashed: detected_th)")
    save(fig, "detection_threshold_by_type.pdf")


def _plot_head_thresholds(metrics, plt, save, swept, seconds, headline):
    """The figures of the per-head thresholds (tuned on val) against detected_th:
      per_head_threshold.pdf: event recall, alarm precision and false alarms per
          minute of both operating points, overall and per counted event type, at
          the headline window;
      per_head_threshold_vs_window.pdf: the same overall per window;
      head_thresholds.pdf: the rule net's threshold of every head vs its onsets."""
    if "head_thresholds" not in metrics:
        print("[Figure]\t skipped the per-head threshold figures: no per-head thresholds in the metrics")
        return
    windows, results = metrics["windows"], metrics["results"]
    names = [n for n in swept if "per_head_threshold" in results[headline]["detection"][n]]
    groups = ["overall", *metrics.get("counted_types", EVENT_TYPES)]
    points = (("detected_th", "tab:gray"), ("per-head", "tab:blue"))

    def point(name, group, which, w=headline):
        d = results[w]["detection"][name]
        if which == "per-head":
            d = d["per_head_threshold"]
        return d if group == "overall" else d["by_type"][group]

    panels = (("recall", "event recall"), ("precision", "alarm precision"),
              ("false_alarms_per_minute", "false alarms / min (log)"))
    fig, axes = plt.subplots(len(groups), len(panels), figsize=(14, 3.2 * len(groups)), squeeze=False)
    xs = range(len(names))
    for row, group in enumerate(groups):
        for col, (key, label) in enumerate(panels):
            ax = axes[row, col]
            for i, (which, color) in enumerate(points):
                ax.bar([x + (i - 0.5) * 0.4 for x in xs], [point(n, group, which)[key] or 0 for n in names],
                       width=0.38, color=color, label=which if (row, col) == (0, 0) else None)
            ax.set_xticks(list(xs), [SCORER_LABELS[n] for n in names], fontsize=7)
            events = point(names[0], group, "per-head")["events"] if names else 0
            ax.set(title=f"{group} ({events} events): {label}")
            if key == "false_alarms_per_minute":
                ax.set_yscale("log")
            else:
                ax.set_ylim(0, 1)
            ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"detected_th = {metrics['detected_th']} vs per-head thresholds tuned on val, "
                 f"window {headline} frames")
    save(fig, "per_head_threshold.pdf")

    fig, axes = plt.subplots(1, len(panels), figsize=(14, 4))
    for ax, (key, label) in zip(axes, panels):
        for name in names:
            for which, style in (("detected_th", "--"), ("per-head", "o-")):
                ax.plot(seconds, [point(name, "overall", which, str(w))[key] or float("nan") for w in windows], style,
                        markersize=3, color=SCORER_COLORS[name],
                        label=f"{SCORER_LABELS[name]}, {which}" if which == "per-head" else None)
        ax.set(title=label, xlabel="window W (s)")
        if key == "false_alarms_per_minute":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.suptitle(f"Overall ({' + '.join(metrics.get('counted_types', EVENT_TYPES))} events): per-head thresholds "
                 f"(solid) vs detected_th = {metrics['detected_th']} (dashed)")
    save(fig, "per_head_threshold_vs_window.pdf")

    if "model" not in metrics["head_thresholds"]["scorers"]:
        return
    info = metrics["head_thresholds"]
    onsets = {h["head"]: h["onsets"] for heads in metrics["event_types"]["heads"].values() for h in heads}
    chosen = info["scorers"]["model"][headline]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for source, marker in (("head", "o"), ("type", "^"), ("global", "x")):
        for event_type in metrics.get("counted_types", EVENT_TYPES):
            heads = [a for a, (s_, t) in enumerate(zip(chosen["source"], info["event_types"]))
                     if s_ == source and t == event_type and onsets.get(a, 0) > 0]
            if heads:
                ax.scatter([onsets[a] for a in heads], [max(chosen["threshold"][a], 1e-4) for a in heads],
                           marker=marker, color=TYPE_COLORS[event_type], alpha=0.7,
                           label=f"{event_type}, {source} threshold ({len(heads)})")
    ax.axhline(metrics["detected_th"], color="k", linestyle="--", linewidth=0.8, label="detected_th")
    ax.set(title=f"Rule net threshold per head, window {headline} frames\n"
                 f"(tuned on val; its own with >= {info['min_events']} val events)", xlabel="train + val onsets of the head (log)", ylabel="threshold (log)",
           xscale="log", yscale="log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    save(fig, "head_thresholds.pdf")


# -------------- Summary --------------
def _print_alarm_filter_calibration(metrics, sec):
    """The alarm filter (val grid, test without / with it) and the calibrated operating points."""
    windows = metrics["windows"]
    af = metrics.get("alarm_filter")
    if af:
        k, g = af["chosen"]
        print(f"[Alarm filter] k={k} frames in a row above the threshold, alarms < g={g} frames apart merge "
              f"(chosen on val by the rule net's mean best event F1)")
        for entry in af.get("val_grid") or []:
            print(f"  val k={entry['persist']:>2} g={entry['gap']:>2}: mean best F1 {entry['mean_best_f1']:.4f}")
        print(f"  test, rule net, counted heads: raw (k=g=1) -> filtered; at the detection threshold | at the best F1")
        print(f"  {'W':>5}{'recall':>16}{'precision':>18}{'alarms/min':>18}{'PT mean':>16}"
              f"{'best-F1 recall':>20}{'precision':>18}")
        for w in windows:
            raw, fil = af["test"]["raw"]["windows"][str(w)], af["test"]["filtered"]["windows"][str(w)]
            r, f = raw["at_threshold"], fil["at_threshold"]
            rb, fb = raw["best_f1"], fil["best_f1"]
            print(f"  {w:>5}{r['recall']:>8.3f}->{f['recall']:<6.3f}{r['precision']:>9.4f}->{f['precision']:<7.4f}"
                  f"{r['alarms_per_minute']:>9.1f}->{f['alarms_per_minute']:<7.1f}"
                  f"{sec(r['mean_prediction_time']):>8.2f}->{sec(f['mean_prediction_time']):<6.2f}"
                  f"{rb['recall']:>12.3f}->{fb['recall']:<6.3f}{rb['precision']:>9.4f}->{fb['precision']:<7.4f}")
        print()
    cal = metrics.get("calibration")
    if cal:
        print(f"[Calibration] per-head Platt scaling fitted on val; counted heads; AP / ECE raw -> calibrated; "
              f"detection when alarming at calibrated probability q (recall / alarm precision / alarms per min)")
        for name, per_window in cal.items():
            for w in windows:
                e = per_window[str(w)]
                levels = "  ".join(f"q={q}: {p['recall']:.2f}/{p['precision']:.3f}/{p['alarms_per_minute'] or 0:.1f}"
                                   for q, p in e["levels"].items())
                print(f"  {SCORER_LABELS[name]:<11}{w:>4}  AP {e['average_precision_raw']:.4f}->"
                      f"{e['average_precision_calibrated']:.4f}  ECE {e['ece_raw']:.4f}->{e['ece_calibrated']:.4f}  "
                      f"{levels}")
        print()


def _print_per_head_detection(metrics, sec, headline):
    """Event detection of every head with enough test events (`reliable`), per scorer, at the
    headline window and the largest one, at the detection threshold."""
    per_head = metrics.get("per_head_detection")
    if not per_head:
        return
    print(f"[Per head event detection] heads with >= {metrics['per_head_min_test_events']} test events; "
          f"recall / alarm precision / mean prediction time (s), at the detection threshold")
    names = [n for n in ("model", "motif_net", "mlp", "rule_set", "cascade") if n in per_head]
    for w in (headline, str(max(metrics["windows"]))):
        heads = {h: d for h, d in per_head["model"][w].items() if d["reliable"]}
        print(f"  W={w}  {'head':<20}{'type':<9}{'events':>7}" + "".join(f"{SCORER_LABELS[n]:>24}" for n in names))
        for h, d in sorted(heads.items(), key=lambda x: (x[1]["event_type"], -x[1]["events"])):
            cells = []
            for n in names:
                e = per_head[n][w].get(h)
                cells.append("—" if e is None else
                             f"{e['recall']:.2f}/{e['precision']:.3f}/{sec(e['mean_prediction_time']):.1f}")
            print(f"        {d['label']:<20}{d['event_type']:<9}{d['events']:>7}" + "".join(f"{c:>24}" for c in cells))
    print()


def print_summary(metrics, clauses, cfg, num_examples=15):
    windows, results = metrics["windows"], metrics["results"]
    headline = str(metrics["headline_window"])
    fps = metrics["frame_rate"]
    print(f"[Evaluation Summary] 'atom switches on within W frames' on the test split, {metrics['num_videos']} videos, "
          f"{metrics['minutes']:.1f} min of fully buffered video (>= {metrics['min_context']} frames of history)")
    row_names = list(results[str(windows[0])]["rows"])
    print(f"[AP per window] rows = (agent, frame, off atom); random = base rate")
    print(f"  {'W':>9}{'base rate':>11}" + "".join(f"{SCORER_LABELS[n][:10]:>12}" for n in row_names) + f"{'ECE net':>9}")
    for w in windows:
        r = results[str(w)]["rows"]
        print(f"  {w:>4} {w / fps:>3.0f}s{r['model']['base_rate']:>11.2e}"
              + "".join(f"{r[n]['average_precision']:>12.4f}" for n in row_names)
              + f"{r['model']['calibration']['ece']:>9.4f}")
    print()
    print(f"[Detection] threshold {metrics['threshold_from']}; an event is detected if its score reaches the "
          f"threshold in any frame of the at-risk stretch before the onset; PT: prediction time (s) from the "
          f"earliest detected frame, mean / median over the detected events; cov: detected frames / prediction "
          f"time (missed = 0); an alarm is an upward crossing, correct if the onset follows within W")
    print(f"  {'W':>5} {'scorer':<14}{'events':>7}{'detected':>9}{'recall':>8}{'PT mean':>9}{'PT med':>8}"
          f"{'cov':>7}{'cov det':>8}{'lookback':>9}{'alarms':>8}{'precision':>10}{'FA/min':>9}{'thr':>9}")

    def sec(frames):
        return float("nan") if frames is None else frames / fps

    for w in windows:
        for name, d in results[str(w)]["detection"].items():
            cov_det = float("nan") if d["coverage_detected"] is None else d["coverage_detected"]
            print(f"  {w:>5} {SCORER_LABELS[name]:<14}{d['events']:>7}{d['detected']:>9}{d['recall']:>8.3f}"
                  f"{sec(d['mean_prediction_time']):>9.2f}{sec(d['median_prediction_time']):>8.2f}"
                  f"{d['coverage']:>7.3f}{cov_det:>8.3f}{sec(d['mean_lookback']):>9.2f}"
                  f"{d['alarms']:>8}{d['precision']:>10.4f}{d['false_alarms_per_minute'] or 0:>9.2f}"
                  f"{d['threshold']:>9.3g}")
    print(f"  (lookback: mean length (s) of the at-risk stretch searched before the onset; overall rows count only "
          f"the {' + '.join(metrics.get('counted_types', EVENT_TYPES))} events, noise events are reported per type)")
    print()

    if "event_types" in metrics:
        types = metrics["event_types"]
        print(f"[Event types] heads by {types['count_source']}: noise < {types['noise_below']} onsets, "
              f"rare < {types.get('rare_below')}, trivial > {types['trivial_above']} onsets, regular in between")
        for event_type in EVENT_TYPES:
            heads = types["heads"][event_type]
            seen = [h for h in heads if h["onsets"] > 0]
            names = ", ".join(f"{h['label']} ({h['onsets']})" for h in seen[:8]) + (", ..." if len(seen) > 8 else "")
            print(f"  {event_type:<8} {len(heads):>3} heads ({len(seen)} seen), "
                  f"{sum(h['onsets'] for h in heads) / max(types['num_onsets'], 1):.0%} of onsets: {names}")
        print(f"[Per event type, window {headline}] AP of the rows; detections at the detection thresholds")
        print(f"  {'type':<9}{'scorer':<14}{'events':>7}{'AP':>8}{'recall':>8}{'PT mean':>9}{'PT med':>8}"
              f"{'cov':>7}{'alarms':>8}{'precision':>10}{'FA/min':>9}")
        for event_type in EVENT_TYPES:
            for name, d in results[headline]["detection"].items():
                t = d["by_type"][event_type]
                ap = results[headline]["rows"][name]["by_type"][event_type]["average_precision"]
                print(f"  {event_type:<9}{SCORER_LABELS[name]:<14}{t['events']:>7}{ap:>8.4f}{t['recall']:>8.3f}"
                      f"{sec(t['mean_prediction_time']):>9.2f}{sec(t['median_prediction_time']):>8.2f}"
                      f"{t['coverage']:>7.3f}{t['alarms']:>8}{t['precision']:>10.4f}"
                      f"{t['false_alarms_per_minute'] or 0:>9.2f}")
        if any("per_head_threshold" in d for d in results[headline]["detection"].values()):
            print(f"[Per-head thresholds, window {headline}] tuned on val (own threshold with >= "
                  f"{metrics['per_head_min_events']} val events, else the type's, else detected_th); "
                  f"in brackets: at detected_th")
            print(f"  {'scorer':<14}{'group':<9}{'events':>7}{'recall':>15}{'PT mean':>9}{'cov':>7}"
                  f"{'precision':>17}{'FA/min':>17}{'own/type/global':>17}")
            for name, d in results[headline]["detection"].items():
                if "per_head_threshold" not in d:
                    continue
                for group in ("overall", *metrics.get("counted_types", EVENT_TYPES), "noise"):
                    h = d["per_head_threshold"] if group == "overall" else d["per_head_threshold"]["by_type"][group]
                    g = d if group == "overall" else d["by_type"][group]
                    src = ("" if group == "overall" else
                           "/".join(str(h["threshold_sources"][x]) for x in HEAD_THRESHOLD_SOURCES))
                    print(f"  {SCORER_LABELS[name]:<14}{group:<9}{h['events']:>7}"
                          f"{h['recall']:>8.3f} ({g['recall']:.3f})"
                          f"{sec(h['mean_prediction_time']):>9.2f}{h['coverage']:>7.3f}"
                          f"{h['precision']:>9.4f} ({g['precision']:.3f})"
                          f"{h['false_alarms_per_minute'] or 0:>9.1f} ({g['false_alarms_per_minute'] or 0:.1f})"
                          f"{src:>17}")
        print(f"[Event recall per type and window]")
        print(f"  {'type':<9}{'scorer':<14}" + "".join(f"{w:>7}" for w in windows))
        for event_type in EVENT_TYPES:
            for name in results[headline]["detection"]:
                print(f"  {event_type:<9}{SCORER_LABELS[name]:<14}"
                      + "".join(f"{results[str(w)]['detection'][name]['by_type'][event_type]['recall']:>7.3f}"
                                for w in windows))
        print()
    print(f"[Operating points] window {headline} frames, best alarm precision at event recall >= R (test curves)")
    for name, d in results[headline]["detection"].items():
        if name in FIXED_THRESHOLDS:
            continue
        points = ", ".join(f"R>={level}: P {p['precision']:.3f}, {p['alarms_per_minute']:.1f} alarms/min"
                           for level, p in d["at_recall"].items())
        print(f"  {SCORER_LABELS[name]:<20}{points or 'recall levels not reached'}")
    print()

    print(f"[Partial facts] the same rows knowing only the last H frames of history (older memory unknown); "
          f"detections at the detection thresholds")
    print(f"  {'W':>5} {'scorer':<10}" + "".join(f"{'H=' + h:>33}" for h in sorted(metrics["history"], key=int)))
    for w in windows:
        for name in HISTORY_SCORERS:
            if name not in metrics["history"][str(metrics["min_context"])][str(w)]:
                continue
            cells = []
            for h in sorted(metrics["history"], key=int):
                v = metrics["history"][h][str(w)][name]
                cells.append(f"AP {v['average_precision']:.3f} c {v['mean_onset_score']:.3f} R {v['recall']:.2f} "
                             f"cov {v['coverage']:.2f}")
            print(f"  {w:>5} {SCORER_LABELS[name]:<10}" + "".join(f"{c:>33}" for c in cells))
    print(f"  (c: mean confidence of the true onsets; R: event recall; cov: coverage)")
    print()
    if "model" in metrics["trajectory"]:
        k = windows.index(int(headline))
        mean = metrics["trajectory"]["model"]["mean"]
        leads = [j for j in (50, 30, 20, 10, 5, 0) if j < len(mean)]
        print(f"[Trajectory] rule net, mean confidence of 'onset within {headline}' for the event's own head: "
              + ", ".join(f"{j} frames before {mean[j][k]:.3f}" for j in leads))
        print()

    if metrics.get("rule_levels"):
        print("[Rule sets on test] each rule at least p precise (train); event recall / row precision / rules, "
              "* = the level chosen on val")
        levels = list(metrics["rule_levels"])
        print(f"  {'W':>5}" + "".join(f"{'p>=' + level:>22}" for level in levels))
        for w in windows:
            cells = []
            for level in levels:
                m = metrics["rule_levels"][level][str(w)]
                star = "*" if metrics["rule_levels_chosen"].get(str(w)) == level else " "
                cells.append(f"{star}R {m['recall']:.2f} P {m['precision']:.3f} n {m['rules']:>3}")
            print(f"  {w:>5}" + "".join(f"{c:>22}" for c in cells))
        print()

    print(f"[Per head, window {headline}, by AP] onset rows  base rate  AP  MLP AP  anchor AP  best F1")
    rows = sorted((m for m in metrics["per_head"].values() if m.get("event_type") != "noise"),
                  key=lambda m: -m["average_precision"])
    for m in rows[:num_examples]:
        b = m["baseline_average_precision"]
        print(f"  {_short_num(m['onsets']):>7}  {m['base_rate']:.4f}  {m['average_precision']:.3f}  "
              f"{b.get('mlp', 0):.3f}  {b.get('anchor', 0):.3f}  {m['best_f1']['f1']:.3f}  [{m['label']}]")
    print()

    _print_alarm_filter_calibration(metrics, sec)
    _print_per_head_detection(metrics, sec, headline)

    print(f"[Clauses]\t {len(clauses)} from step 05")
    firing = [c for c in clauses if c["test_confidence"] is not None]
    if firing:
        confidences = sorted(c["test_confidence"] for c in firing)
        print(f"[Clauses]\t {len(firing)} fire on test, median confidence {confidences[len(confidences) // 2]:.3f}")
        extended = [c for c in firing if len(c["bodies"]) > 1 and c["test_anchor_confidence"] is not None]
        if extended:
            better = sum(c["test_confidence"] > c["test_anchor_confidence"] for c in extended)
            print(f"[Extension]\t {better}/{len(extended)} extended clauses beat their anchor on test")
        unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
        print(f"[Top {min(num_examples, len(firing))} clauses by test confidence]")
        for c in firing[:num_examples]:
            print(f"  conf {c['test_confidence']:.2f} (train {c['confidence']:.2f})  "
                  f"support {c['test_support']:>5}  {step05._temporal_clause_text(c, unique_atoms)}")
            steps = [f"{p['known_bodies']} known: {p['test_confidence']:.3f}" for p in reversed(c.get("partial", []))
                     if p.get("test_confidence") is not None and p["unseen_slots"] > 0]
            if steps:
                print(f"      earlier, fewer bodies known: " + ", ".join(steps))
    print()


def main(input_data):
    print("\n--------- Step 06 Evaluation ----------------------\n")
    step05_cfg = step05.get_cfg(input_data["step05_input"])
    cfg = {
        "device": input_data["device"],
        "unique_atoms_file": input_data["unique_atoms_file"],
        "test_atoms_file": input_data["test_atoms_file"],
        "frames_per_video": step05_cfg["frames_per_video"],
        # Only fully buffered predictions are evaluated: frames with the model's whole
        # lag range before them, so no rule misses its body for lack of history.
        "min_context": step05_cfg["slot_size"] * step05_cfg["num_slots"],
        # Event detection: an event is detected if the score of its agent and head reaches
        # detected_th in any frame of the at-risk stretch before the onset (one frame is
        # enough); prediction time = frames from the earliest such frame to the onset;
        # coverage = detected frames / prediction time (0 if missed). The rule set and the
        # candidates use FIXED_THRESHOLDS. The curves sweep every threshold.
        "detected_th": 0.5,
        # Version of the detection metric; a cached test_metrics.json of another one is redone.
        "detection_metric": "stretch_first_detection_coverage_by_type_head_thresholds_alarm_filter_calibration",
        # Alarm filter (`_score_videos`), chosen on the val videos (`select_alarm_filter`): an
        # alarm needs k frames in a row above the threshold (persist) and alarms closer than
        # g frames merge (gap). k = g = 1 is the plain per-frame alarm.
        "alarm_filter_grid": {"persist": [1, 3, 5, 10], "gap": [1, 10, 30]},
        # Per-head calibration (Platt scaling on val) of the probability scorers: own fit for
        # heads with >= calibration_min_events val events, else their event type's; the test
        # detection is reported when alarming at these calibrated probabilities.
        "calibration_min_events": 5,
        "calibrated_levels": [0.05, 0.1, 0.2, 0.3, 0.5],
        # Per-head results are "reliable" with at least this many test events.
        "per_head_min_test_events": 5,
        # Second operating point: one threshold per head, tuned on the val videos (best F1 of
        # the head's event recall and alarm precision) for heads with at least this many val
        # events, else the threshold of the head's event type, else detected_th.
        "per_head_min_events": 10,
        # Partial facts: the rows are scored again knowing only this many frames of history.
        "history_levels": [0, 10, 20, 30, 40],
        # The window of the per-head metrics and of the single-window figures.
        "headline_window": 20,
        # Probability threshold of the "at_threshold" row metrics.
        "prob_threshold": 0.5,
        "frame_rate": input_data["frame_rate"],
        "baseline_file": input_data["output_dir"] / "baselines.pt",
        "test_data_file": input_data["output_dir"] / "test_data.pt",
        "metrics_file": input_data["output_dir"] / "test_metrics.json",
        "test_clause_file": input_data["output_dir"] / "test_clauses.json",
        "results_file": input_data["output_dir"] / "test_results.json",
        "figure_dir": input_data["output_dir"] / "figures",
    }
    step05_cfg["device"] = cfg["device"]
    assert cfg["headline_window"] in step05_cfg["windows"]

    # 1. Test videos in the format of the step 05 training data, with the same min_off_frames.
    test_data = torch.load(cfg["test_data_file"], map_location="cpu") if os.path.exists(cfg["test_data_file"]) else {}
    if test_data.get("min_off_frames") != step05_cfg["min_off_frames"]:
        num_atoms = len(utils_data.load_json(cfg["unique_atoms_file"]))
        test_data = step05.prepare_videos(cfg["test_atoms_file"], num_atoms, cfg["device"],
                                          step05_cfg["min_off_frames"])
        torch.save(test_data, cfg["test_data_file"])
    model = step05.load_temporal_rule_net(test_data, step05_cfg)
    mlp = step05.load_onset_mlp(test_data, step05_cfg)      # None if the baseline was not trained
    # The step 05 cascade (stage-1 thresholds and the verifier), None if it was not trained.
    cascade = None
    if step05_cfg["cascade"] and os.path.exists(step05_cfg["verifier_model_file"]):
        cascade = {"verifier": step05.load_temporal_rule_net(test_data, step05_cfg, step05_cfg["verifier_model_file"]),
                   "thresholds": torch.load(step05_cfg["candidate_threshold_file"])["thresholds"]}
    cfg["cascade"] = cascade is not None
    # The step 05b causally pruned rule net, scored like the rule net, if it was made.
    pruned = step05_causal.load_pruned_rule_net(test_data, step05_cfg)
    extra = {"net_pruned": pruned} if pruned is not None else {}
    cfg["net_pruned"] = pruned is not None
    # The step 05c rule set, scored by the precision of its rules that fire, if it was made.
    selection_file = step05_rules.selection_file_of(step05_cfg)
    selection = utils_data.load_json(selection_file) if os.path.exists(selection_file) else None
    if selection is not None:
        extra["rule_set"] = step05_rules.RuleSetScorer(selection, model.num_inputs, model.num_atoms, cfg["device"])
    cfg["rule_set"] = selection is not None
    # The step 05d MotifRuleNet (its own 2-frame inputs, memory 60 frames), if it was trained.
    motif_file, motif_cfg = step05_motif.model_file_of(step05_cfg)
    motif = step05_motif.load_motif_rule_net(test_data, {**motif_cfg, "device": cfg["device"]})
    if motif is not None:
        extra["motif_net"] = motif
        # Every model is evaluated with its full memory: the rows need the longest one before them.
        cfg["min_context"] = max(cfg["min_context"], motif.num_lags)
    cfg["motif_net"] = motif is not None

    # 2. Baselines fitted on the step 05 training videos: the window prior and the anchor rules alone.
    if not step05.is_fresh(cfg["baseline_file"], step05_cfg["model_file"], step05_cfg["training_data_file"]):
        training_data = step05.load_training_data(step05_cfg)
        torch.save(fit_baselines(model, training_data, step05_cfg), cfg["baseline_file"])
    baselines = torch.load(cfg["baseline_file"], map_location="cpu")

    # 3. Window metrics of the model and the baselines; redone whenever any input or setting changes.
    metrics = utils_data.load_json(cfg["metrics_file"]) if os.path.exists(cfg["metrics_file"]) else {}
    settings = {"windows": step05_cfg["windows"], "min_context": cfg["min_context"],
                "headline_window": cfg["headline_window"], "cascade": cfg["cascade"],
                "detected_th": cfg["detected_th"], "detection_metric": cfg["detection_metric"],
                "noise_below": step05_cfg["noise_below"], "rare_below": step05_cfg["rare_below"],
                "trivial_above": step05_cfg["trivial_above"],
                "exclude_noise_heads": step05_cfg["exclude_noise_heads"],
                "per_head_min_events": cfg["per_head_min_events"],
                "alarm_filter_grid": cfg["alarm_filter_grid"], "calibration_min_events": cfg["calibration_min_events"],
                "calibrated_levels": cfg["calibrated_levels"],
                "per_head_min_test_events": cfg["per_head_min_test_events"],
                "history_levels": cfg["history_levels"],
                "net_pruned": cfg["net_pruned"], "rule_set": cfg["rule_set"], "motif_net": cfg["motif_net"]}
    if (not step05.is_fresh(cfg["metrics_file"], step05_cfg["model_file"], cfg["test_data_file"],
                            cfg["baseline_file"], step05_cfg["training_data_file"], step05_cfg["mlp_model_file"],
                            step05_cfg["candidate_threshold_file"], step05_cfg["verifier_model_file"],
                            step05_causal.get_cfg_from_step05(step05_cfg)["pruned_model_file"], selection_file,
                            motif_file)
            or settings != {key: metrics.get(key) for key in settings}):
        training_data = step05.load_training_data(step05_cfg)
        cfg["event_types"] = training_data["event_types"]
        cfg["head_types"] = cfg["event_types"]["types"]
        cfg["exclude_noise_heads"] = step05_cfg["exclude_noise_heads"]
        # Everything tuned on the validation videos (never on test): the alarm filter (rule net),
        # then with it the per-head thresholds and the per-head calibration of every scorer.
        val_data = _subset(training_data, training_data["val_videos"])
        cfg["alarm_filter"], cfg["alarm_filter_val_grid"] = select_alarm_filter(model, val_data, baselines, cfg)
        print(f"[Alarm filter] chosen on val: k={cfg['alarm_filter'][0]}, g={cfg['alarm_filter'][1]}")
        val_scores = _score_videos(model, val_data, baselines, cfg, mlp, cascade, extra=extra)
        head_thresholds = select_head_thresholds(val_scores, cfg)
        calibration = fit_calibration(val_scores["rows"], cfg, model.windows)
        del val_scores, val_data
        del training_data
        metrics = evaluate_model(model, test_data, baselines, cfg, mlp, cascade, extra, head_thresholds, calibration)
        if selection is not None:
            # Every precision level's rule set on the test videos: recall vs precision of precise rules.
            metrics["rule_levels"] = step05_rules.level_tradeoff(model, selection, test_data,
                                                                 {"device": cfg["device"], "step05_cfg": step05_cfg})
            metrics["rule_levels_chosen"] = selection["chosen_level"]
        utils_data.save_json(metrics, cfg["metrics_file"])
    metrics = utils_data.load_json(cfg["metrics_file"])

    # 4. The step 05 clauses on the test videos; redone whenever they change.
    if not step05.is_fresh(cfg["test_clause_file"], step05_cfg["extended_clause_file"], cfg["test_data_file"]):
        clauses = utils_data.load_json(step05_cfg["extended_clause_file"])
        utils_data.save_json(evaluate_clauses(model, test_data, clauses, step05_cfg), cfg["test_clause_file"])
    test_clauses = utils_data.load_json(cfg["test_clause_file"])

    # 5. The main results (JSON) and the figures (one PDF each), on every run.
    utils_data.save_json(build_results(metrics, test_clauses), cfg["results_file"])
    print(f"[Results]\t {cfg['results_file']}")
    plot_results(metrics, test_clauses, cfg)
    print_summary(metrics, test_clauses, cfg)
    print("\n--------- Step 06 Done ---------------\n")
