"""Step 06: evaluate the step 05 window predictions on the test videos.

For every agent visible in frame t (with the model's full history before t) and
every atom that is off, the model says how likely the atom switches on within
the next W frames, for each window W of step 05 (10, 20, ..., 100). Per window:
  rows: precision-recall / AP of these probabilities against "it did switch on
      within W" (rows whose window runs past the video end without an onset are
      censored), and their calibration;
  events: an onset is detected for window W if the score of some frame in the W
      frames up to it reaches the threshold (the model said "within W" in time);
  alarms: an alarm is raised when the score crosses the threshold upwards (one
      per episode, not one per frame); it is correct if the onset follows within W.
Thresholds are picked on the step 05 validation videos (best F1 of event recall
and alarm precision).
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
# Scorers counted at a fixed threshold instead of the best validation F1.
# The rule set alarms whenever one of its rules fires (its scores are their precisions, >= the level).
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
def _score_videos(model, data, baselines, cfg, mlp=None, cascade=None, history=None, names=None, extra=None):
    """Window scores of the model and of the baselines on every video of the split.

    Rows are the frames t >= cfg["min_context"]; history: if given, every row
    knows only that many frames before t (the older memory is unknown), else all
    of them. An alarm needs the score at or above the threshold for
    cfg["min_alarm_frames"] frames in a row: the sustained score of a frame is the
    lowest score of the last min_alarm_frames frames. Returns
      num_frames: the scored frames of all videos;
      rows {scorer: {"pos", "neg": (W, A, S), "score_sum": (W, S), "pos_sum",
          "neg_sum": (W,)}}: histograms (S bins of `score_edges`) of the scores of
          the at-risk rows with window k observed, split by whether the atom
          switched on within W_k; score_sum the sum of the scores per bin
          (calibration); pos_sum / neg_sum the summed scores of the two groups;
      starts {scorer: {"pos", "neg": (W, S)}}: alarms at each threshold bin b,
          rows whose sustained score is in bin >= b while the frame before was
          below b (or not scored), split by whether the onset followed within W_k;
      events {scorer: (W, A, S)}: per exact onset (on in t_e, off in the
          min_off_frames frames before, the agent visible in t_e - 1) whose W_k
          frames up to it are all rows, the histogram of the highest sustained
          score of window k over the frames t_e - W_k + 1..t_e;
      trajectory {scorer: {"sum", "count": (top, W)}}: the summed score of the
          onset's own head and agent j frames before the onset, over all onsets
          (the confidence approaching an event).
    """
    device = cfg["device"]
    edges = score_edges().to(device)
    S, A = len(edges), data["num_atoms"]
    W, top = len(model.windows), max(model.windows)
    min_context, D = cfg["min_context"], cfg["min_alarm_frames"]
    context = {}
    scorers = _scorers(model, baselines, mlp, cascade, device, names, extra, context)
    heads = torch.arange(A, device=device)
    window_index = torch.arange(W, device=device)[None, :, None]

    def zeros(*shape):
        return torch.zeros(shape, device=device)

    rows_hist = {name: {"pos": zeros(W * A * S), "neg": zeros(W * A * S), "score_sum": zeros(W * S),
                        "pos_sum": zeros(W), "neg_sum": zeros(W)} for name in scorers}
    starts = {name: {"pos": zeros(W * (S + 1)), "neg": zeros(W * (S + 1))} for name in scorers}
    trajectory = {name: {"sum": zeros(top, W), "count": zeros(top, W)} for name in scorers}
    event_heads, event_frames, event_scores = [], [], {name: [] for name in scorers}
    num_frames = 0
    for video_id in tqdm(data["videos"], desc="Scoring videos" + ("" if history is None else f" (history {history})")):
        X = step05._video_from_cells(data["videos"][video_id], A, device)
        T, N, _A = X.shape
        num_frames += max(T - min_context, 0)
        # Exact onsets and, per onset, the frames t_e - j (j = 0..top - 1) whose window scores can detect it.
        onsets = step05.atom_onsets(X, data["min_off_frames"])
        onsets[0] = False
        onsets[1:] &= X[:-1].any(dim=2, keepdim=True)
        e_t, e_n, e_a = torch.nonzero(onsets, as_tuple=True)
        e_lead = e_t[:, None] - torch.arange(top, device=device)                     # (E, top)
        e_scores = {name: torch.full((len(e_t), top, W), -1.0, device=device) for name in scorers}
        e_raw = {name: torch.full((len(e_t), top, W), -1.0, device=device) for name in scorers}
        # The scores of the last D frames (-1: not a row), for the sustained scores.
        recent = {name: torch.full((D, N, W, A), -1.0, device=device) for name in scorers}
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
                # Sustained scores: sustained[s] is the lowest score of the D frames up to frame begin + s - 1.
                window = torch.cat([recent[name], dense])
                sustained = window.unfold(0, D, 1).min(dim=-1).values                  # (F + 1, N, W, A)
                recent[name] = window[-D:]
                if p is None:
                    continue
                bins = _score_bins(p, edges)                                         # (B, W, A)
                cell = (window_index * A + heads) * S + bins
                h = rows_hist[name]
                h["pos"] += torch.bincount(cell[valid & targets], minlength=W * A * S)
                h["neg"] += torch.bincount(cell[valid & ~targets], minlength=W * A * S)
                h["score_sum"] += torch.bincount((window_index * S + bins)[valid].flatten(), weights=p[valid],
                                                 minlength=W * S)
                h["pos_sum"] += torch.where(valid & targets, p, 0.0).sum(dim=(0, 2))
                h["neg_sum"] += torch.where(valid & ~targets, p, 0.0).sum(dim=(0, 2))
                # Alarms: the threshold bins b with previous bin < b <= current bin (a difference array).
                now, before = sustained[1:][t - begin, n], sustained[:-1][t - begin, n]
                high = torch.where(valid & (now >= 0), _score_bins(now.clamp(min=0), edges), -1)
                low = torch.where(before >= 0, _score_bins(before.clamp(min=0), edges), -1)
                up = high > low
                base = (window_index * (S + 1)).expand_as(bins)
                for key, mask in (("pos", up & targets), ("neg", up & ~targets)):
                    starts[name][key] += (torch.bincount(base[mask] + low[mask] + 1, minlength=W * (S + 1))
                                          - torch.bincount(base[mask] + high[mask] + 1, minlength=W * (S + 1)))
                index = (e_row.clamp(min=0), slice(None), e_a[:, None])
                e_scores[name] = torch.where(e_valid[..., None], now[index], e_scores[name])
                e_raw[name] = torch.where(e_valid[..., None], p[index], e_raw[name])
        event_heads.append(e_a)
        event_frames.append(e_t)
        for name in scorers:
            event_scores[name].append(e_scores[name])
            seen = e_raw[name] >= 0
            trajectory[name]["sum"] += torch.where(seen, e_raw[name], 0.0).sum(dim=0)
            trajectory[name]["count"] += seen.sum(dim=0)

    e_head, e_frame = torch.cat(event_heads).cpu(), torch.cat(event_frames).cpu()
    events = {}
    for name in scorers:
        scores = torch.cat(event_scores[name]).cpu()                                 # (E, top, W)
        per_window = []
        for k, w in enumerate(model.windows):
            kept = e_frame - w + 1 >= min_context
            best = scores[kept, :w, k].max(dim=1).values if kept.any() else torch.zeros(0)
            bins = _score_bins(best, score_edges())
            per_window.append(torch.bincount(e_head[kept] * S + bins, minlength=A * S).view(A, S))
        events[name] = torch.stack(per_window).float()
    return {
        "num_frames": num_frames,
        "rows": {name: {"pos": h["pos"].view(W, A, S).cpu(), "neg": h["neg"].view(W, A, S).cpu(),
                        "score_sum": h["score_sum"].view(W, S).cpu(), "pos_sum": h["pos_sum"].cpu(),
                        "neg_sum": h["neg_sum"].cpu()} for name, h in rows_hist.items()},
        "starts": {name: {key: v.view(W, S + 1).cumsum(dim=1)[:, :S].cpu() for key, v in s.items()}
                   for name, s in starts.items()},
        "events": events,
        "trajectory": {name: {key: v.cpu() for key, v in tr.items()} for name, tr in trajectory.items()},
    }


# -------------- Metrics --------------
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


def _detection_curve(events, starts_pos, starts_neg):
    """Per threshold bin b: detected events (event histogram (A, S) at bins >= b),
    their recall, the alarms, correct alarms and alarm precision; F1 of the two."""
    detected = events.sum(dim=0).flip(0).cumsum(0).flip(0)
    num_events = events.sum().item()
    alarms = starts_pos + starts_neg
    recall = detected / max(num_events, 1)
    precision = starts_pos / alarms.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {"events": num_events, "detected": detected, "recall": recall, "alarms": alarms,
            "correct_alarms": starts_pos, "precision": precision, "f1": f1}


def _detection_at(curve, b, threshold, minutes):
    alarms, correct = curve["alarms"][b].item(), curve["correct_alarms"][b].item()
    return {
        "threshold": threshold,
        "events": int(curve["events"]),
        "detected": int(curve["detected"][b]),
        "missed": int(curve["events"] - curve["detected"][b]),
        "recall": curve["recall"][b].item(),
        "alarms": int(alarms),
        "false_alarms": int(alarms - correct),
        "precision": curve["precision"][b].item(),
        "alarms_per_minute": alarms / minutes if minutes else None,
        "false_alarms_per_minute": (alarms - correct) / minutes if minutes else None,
    }


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


def select_thresholds(model, training_data, baselines, cfg, mlp=None, cascade=None, extra=None):
    """Alarm threshold of each window and scorer: the best F1 of event recall and
    alarm precision on the step 05 validation videos (fixed for FIXED_THRESHOLDS),
    so the test counts are not tuned on the test set. Returns {w: {scorer: threshold}}."""
    scores = _score_videos(model, _subset(training_data, training_data["val_videos"]), baselines, cfg, mlp, cascade,
                           extra=extra)
    edges = score_edges()
    thresholds = {}
    for k, w in enumerate(model.windows):
        thresholds[w] = {}
        for name in (n for n in DETECTION_SCORERS if n in scores["events"]):
            if name in FIXED_THRESHOLDS:
                thresholds[w][name] = FIXED_THRESHOLDS[name]
                continue
            curve = _detection_curve(scores["events"][name][k], scores["starts"][name]["pos"][k],
                                     scores["starts"][name]["neg"][k])
            thresholds[w][name] = edges[int(curve["f1"].argmax())].item()
    return thresholds


def evaluate_model(model, data, baselines, thresholds, cfg, mlp=None, cascade=None, extra=None):
    """Window metrics of the model and of the baselines on the test split.

    Returns {"windows": {w: {"rows": {scorer: AP, macro AP, base rate, best F1,
    calibration, the precision-recall curve}, "detection": {scorer: the counts
    at the validation threshold, "at_recall": the best alarm precision and alarm
    rates at event recall >= RECALL_LEVELS, "curve": recall / precision / alarms
    per minute per threshold bin}}}, "per_head": AP of each head at the headline
    window, against the baselines, "trajectory": the mean score of the onset's
    head j frames before it, per window, "history": per known history in
    cfg["history_levels"] (and the full memory) and window, AP, calibration, the
    mean score of the onset and the other rows and the detections at the
    full-history thresholds of the scorers reading the memory}, plus the settings."""
    scores = _score_videos(model, data, baselines, cfg, mlp, cascade, extra=extra)
    minutes = scores["num_frames"] / cfg["frame_rate"] / 60
    edges = score_edges()
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    out = {"num_videos": len(data["videos"]), "minutes": minutes, "min_off_frames": data["min_off_frames"],
           "min_context": cfg["min_context"], "windows": model.windows, "headline_window": cfg["headline_window"],
           "frame_rate": cfg["frame_rate"], "prob_threshold": cfg["prob_threshold"], "cascade": cfg["cascade"],
           "min_alarm_frames": cfg["min_alarm_frames"], "history_levels": cfg["history_levels"],
           "net_pruned": cfg.get("net_pruned", False), "rule_set": cfg.get("rule_set", False),
           "motif_net": cfg.get("motif_net", False),
           "threshold_from": "val (best F1 of event recall and alarm precision)", "results": {}}
    for k, w in enumerate(model.windows):
        result = {"rows": {}, "detection": {}}
        for name, rows in scores["rows"].items():
            pos, neg = rows["pos"][k], rows["neg"][k]
            heads = torch.nonzero(pos.sum(dim=1)).flatten().tolist()
            per_head = [_histogram_metrics(pos[h], neg[h], cfg["prob_threshold"])["average_precision"] for h in heads]
            result["rows"][name] = {
                **_histogram_metrics(pos.sum(dim=0), neg.sum(dim=0), cfg["prob_threshold"], with_curve=True),
                "macro_average_precision": sum(per_head) / len(per_head) if per_head else 0.0,
                "calibration": _calibration({key: v[k] for key, v in rows.items()}),
            }
        for name in (n for n in DETECTION_SCORERS if n in scores["events"]):
            curve = _detection_curve(scores["events"][name][k], scores["starts"][name]["pos"][k],
                                     scores["starts"][name]["neg"][k])
            threshold = thresholds[w][name]
            at_recall = {}
            for level in RECALL_LEVELS:
                reach = torch.nonzero(curve["recall"] >= level).flatten()
                if len(reach):
                    b = int(reach[curve["precision"][reach].argmax()])
                    at_recall[str(level)] = _detection_at(curve, b, edges[b].item(), minutes)
            result["detection"][name] = {
                **_detection_at(curve, _threshold_bin(threshold), threshold, minutes),
                "at_recall": at_recall,
                "curve": {"threshold": edges.tolist(), "recall": curve["recall"].tolist(),
                          "precision": curve["precision"].tolist(),
                          "alarms_per_minute": (curve["alarms"] / max(minutes, 1e-9)).tolist()},
            }
        out["results"][str(w)] = result

    k = model.windows.index(cfg["headline_window"])
    pos, neg = scores["rows"]["model"]["pos"][k], scores["rows"]["model"]["neg"][k]
    out["per_head"] = {}
    for h in torch.nonzero(pos.sum(dim=1)).flatten().tolist():
        m = _histogram_metrics(pos[h], neg[h], cfg["prob_threshold"])
        m["label"] = step05._atom_name(unique_atoms[h])
        m["baseline_average_precision"] = {
            name: _histogram_metrics(r["pos"][k][h], r["neg"][k][h], cfg["prob_threshold"])["average_precision"]
            for name, r in scores["rows"].items() if name != "model"}
        out["per_head"][str(h)] = m

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
                pos, neg = rows["pos"][k].sum(dim=0), rows["neg"][k].sum(dim=0)
                curve = _detection_curve(partial["events"][name][k], partial["starts"][name]["pos"][k],
                                         partial["starts"][name]["neg"][k])
                detection = _detection_at(curve, _threshold_bin(thresholds[w][name]), thresholds[w][name], minutes)
                out["history"][key][str(w)][name] = {
                    "average_precision": _histogram_metrics(pos, neg, cfg["prob_threshold"])["average_precision"],
                    "ece": _calibration({key_: v[k] for key_, v in rows.items() if key_ in ("pos", "neg", "score_sum")})["ece"],
                    "mean_onset_score": rows["pos_sum"][k].item() / max(pos.sum().item(), 1),
                    "mean_other_score": rows["neg_sum"][k].item() / max(neg.sum().item(), 1),
                    "recall": detection["recall"],
                    "precision": detection["precision"],
                    "alarms_per_minute": detection["alarms_per_minute"],
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
def build_results(metrics, clauses):
    """The headline test numbers per window: AP of every scorer, the detections at
    the validation thresholds and the operating points at fixed event recall. The
    curves and per-head details are in test_metrics.json."""
    firing = [c for c in clauses if c["test_confidence"] is not None]
    extended = [c for c in firing if len(c["bodies"]) > 1 and c["test_anchor_confidence"] is not None]
    windows = {}
    for w, result in metrics["results"].items():
        windows[w] = {
            "base_rate": result["rows"]["model"]["base_rate"],
            "average_precision": {name: r["average_precision"] for name, r in result["rows"].items()},
            "ece": {name: r["calibration"]["ece"] for name, r in result["rows"].items()},
            "detection": {name: {key: value for key, value in d.items() if key != "curve"}
                          for name, d in result["detection"].items()},
        }
    return {
        "test_videos": metrics["num_videos"],
        "test_minutes": metrics["minutes"],
        "min_context": metrics["min_context"],
        "headline_window": metrics["headline_window"],
        "threshold_from": metrics["threshold_from"],
        "min_alarm_frames": metrics["min_alarm_frames"],
        "windows": windows,
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
      detection_vs_window.pdf: event recall, alarm precision and alarms per minute
          per window at the validation thresholds;
      detection_curve.pdf: event recall vs alarms per minute of every scorer
          (all thresholds), at the headline window;
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

    # Detection at the validation thresholds per window.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, label in zip(axes, ("recall", "precision", "alarms_per_minute"),
                              ("event recall", "alarm precision", "alarms / min")):
        for name in detection_names:
            ax.plot(seconds, [results[str(w)]["detection"][name][key] or 0 for w in windows], "o-",
                    color=SCORER_COLORS[name], label=SCORER_LABELS[name])
        ax.set(title=label, xlabel="window W (s)", ylabel=label)
        if key == "alarms_per_minute":
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Detections at the validation thresholds ({metrics['minutes']:.1f} min of test video)")
    save(fig, "detection_vs_window.pdf")

    # Event recall vs alarms per minute at the headline window, every threshold.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name in detection_names:
        if name in FIXED_THRESHOLDS:
            continue
        curve = results[headline]["detection"][name]["curve"]
        points = sorted((a, r) for a, r in zip(curve["alarms_per_minute"], curve["recall"]) if a > 0)
        ax.plot([a for a, _r in points], [r for _a, r in points], color=SCORER_COLORS[name], label=SCORER_LABELS[name])
        d = results[headline]["detection"][name]
        ax.plot(d["alarms_per_minute"] or 0, d["recall"], "o", color=SCORER_COLORS[name])
    ax.set(title=f"Event recall vs alarms, window {headline} frames (dots: val thresholds)",
           xlabel="alarms per minute (log)", ylabel="event recall", xscale="log", ylim=(0, 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    save(fig, "detection_curve.pdf")

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
    heads = sorted(metrics["per_head"].values(), key=lambda m: -m["average_precision"])[:num_heads]
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

    heads = [m for m in metrics["per_head"].values() if "mlp" in m["baseline_average_precision"]]
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


# -------------- Summary --------------
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
    print(f"[Detection] thresholds from {metrics['threshold_from']}; an alarm is raised when the score has stayed at "
          f"or above the threshold for {metrics['min_alarm_frames']} frames, correct if the onset follows within W; "
          f"an event is detected if an alarm was up in the W frames before it")
    print(f"  {'W':>5} {'scorer':<20}{'events':>7}{'detected':>9}{'recall':>8}{'alarms':>8}{'precision':>10}"
          f"{'alarms/min':>11}{'FA/min':>9}{'thr':>10}")
    for w in windows:
        for name, d in results[str(w)]["detection"].items():
            print(f"  {w:>5} {SCORER_LABELS[name]:<20}{d['events']:>7}{d['detected']:>9}{d['recall']:>8.3f}"
                  f"{d['alarms']:>8}{d['precision']:>10.4f}{d['alarms_per_minute'] or 0:>11.2f}"
                  f"{d['false_alarms_per_minute'] or 0:>9.2f}{d['threshold']:>10.3g}")
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
          f"detections at the full-history thresholds")
    print(f"  {'W':>5} {'scorer':<10}" + "".join(f"{'H=' + h:>24}" for h in sorted(metrics["history"], key=int)))
    for w in windows:
        for name in HISTORY_SCORERS:
            if name not in metrics["history"][str(metrics["min_context"])][str(w)]:
                continue
            cells = []
            for h in sorted(metrics["history"], key=int):
                v = metrics["history"][h][str(w)][name]
                cells.append(f"AP {v['average_precision']:.3f} c {v['mean_onset_score']:.3f} R {v['recall']:.2f}")
            print(f"  {w:>5} {SCORER_LABELS[name]:<10}" + "".join(f"{c:>24}" for c in cells))
    print(f"  (c: mean confidence of the true onsets; R: event recall)")
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
    rows = sorted(metrics["per_head"].values(), key=lambda m: -m["average_precision"])
    for m in rows[:num_examples]:
        b = m["baseline_average_precision"]
        print(f"  {_short_num(m['onsets']):>7}  {m['base_rate']:.4f}  {m['average_precision']:.3f}  "
              f"{b.get('mlp', 0):.3f}  {b.get('anchor', 0):.3f}  {m['best_f1']['f1']:.3f}  [{m['label']}]")
    print()

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
        # An alarm needs the score at or above the threshold for this many frames in a row.
        "min_alarm_frames": 5,
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
        training_data = torch.load(step05_cfg["training_data_file"], map_location="cpu")
        torch.save(fit_baselines(model, training_data, step05_cfg), cfg["baseline_file"])
    baselines = torch.load(cfg["baseline_file"], map_location="cpu")

    # 3. Window metrics of the model and the baselines; redone whenever any input or setting changes.
    #    The alarm thresholds are picked on the step 05 validation videos.
    metrics = utils_data.load_json(cfg["metrics_file"]) if os.path.exists(cfg["metrics_file"]) else {}
    settings = {"windows": step05_cfg["windows"], "min_context": cfg["min_context"],
                "headline_window": cfg["headline_window"], "cascade": cfg["cascade"],
                "min_alarm_frames": cfg["min_alarm_frames"], "history_levels": cfg["history_levels"],
                "net_pruned": cfg["net_pruned"], "rule_set": cfg["rule_set"], "motif_net": cfg["motif_net"]}
    if (not step05.is_fresh(cfg["metrics_file"], step05_cfg["model_file"], cfg["test_data_file"],
                            cfg["baseline_file"], step05_cfg["training_data_file"], step05_cfg["mlp_model_file"],
                            step05_cfg["candidate_threshold_file"], step05_cfg["verifier_model_file"],
                            step05_causal.get_cfg_from_step05(step05_cfg)["pruned_model_file"], selection_file,
                            motif_file)
            or settings != {key: metrics.get(key) for key in settings}):
        training_data = torch.load(step05_cfg["training_data_file"], map_location="cpu")
        thresholds = select_thresholds(model, training_data, baselines, cfg, mlp, cascade, extra)
        metrics = evaluate_model(model, test_data, baselines, thresholds, cfg, mlp, cascade, extra)
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
