"""Ego step sequences: what does the ego do next, regardless of when.

Why: the steps of a maneuver (move right, drive slowly, turn right, go straight) last more or less long,
so a rule over 1 s time slots splits one sequence into many. Here time only orders the steps: the
ego's timeline is cut into steps (a step = a stretch of one ego state), their durations are dropped,
and the rules read the order and the combination of the steps only. The task: at each change point
(the ego enters a new state), predict the next state (the ego atoms that switch on at the next change),
not when it comes.

Ego state per frame: the ego's own label atoms (AV-Mov, AV-TurRht, ...) and its ego flow levels
(`ego_flow`: yaw_* and speed_*; a frame without a flow fit keeps the previous level). A new state is a
step only if it holds MIN_STEP frames (shorter flickers stay in the step before). Each step also gets
its scene: the label atoms of the other agents during the step; the current step (the one just entered)
only its first MIN_STEP frames, which is what is known at the change point.

Rows: one per change point, step t - 1 just entered, target step t. Memory (`TemporalRuleNet` with
slot_size 1, the slots are steps): the last NUM_STEPS steps one by one, then coarse bins of steps
(`memory_bins`: 4-6, 7-12, > 12 steps back), with slot ranges, so a rule says "moved right 2-3 steps
before, slow now => turn right next". Scopes: X0 the ego's own state, X1 the scene.
Targets: the ego atoms on in step t and off in step t - 1 (at risk: off in t - 1).

Models: the rule net (free rules, no LLM anchors), the OnsetMLP on the same inputs, the base rate
(prior), first-order Markov (next given the current state) and second-order Markov (the last two
states, backing off to first order). A model that learns order rules beats the Markov baselines.
Metrics on the test change points: top-1 / top-3 (a top-ranked at-risk ego atom switches on next),
label top-1 (rows whose next step changes the ego's action label: the top label atom is that label),
AP per head, mAP; a bootstrap over the test videos for top-1. Significance: a paired bootstrap (the same
resamples of the test videos for both methods) of the difference of top-1, label top-1, mAP and micro AP,
per seed and for the seed-averaged models (`--seeds`, default 0,1,2).

Rules (`rule_stats`): rule bodies that hold at the same rows are one group; each (group, ego head) pair
with |weight| >= 0.1 gets a leave-one-out importance on val (`loo_importance`: the rise of the head's
val loss when the pair's weight is 0 in the rule net) and, as a description, its precision and lift
(precision / base rate; a pair below the base rate can still help: it corrects the other rules of its
head). Kept pairs (importance > 1e-4, fire on train) form the scorecard: per head, P(onset next) =
sigmoid(bias + the weights of its kept rules whose body holds), hard 0/1 bodies. Two scorecards: the
rule net's weights, and weights refit by a logistic regression on train (`refit_scorecard`); a reader
recomputes every score from rules.md. Checks: the rule net with the kept pairs only (if it holds, the
others were not needed); the rule list (`rule_list_scores`, reference: the best single rule above the
base rate, >= --min_rule_hits train onsets) shows what one rule alone predicts.

Re-evaluate saved models without training: add --eval_only (loads <tag>_seed<s>/rule_net.pt, mlp.pt).

Run (server; GPU optional, decodes the videos once for the ego flow cache if missing):
  ./run_roadpp_docker.sh --gpu 6 --seq -- --exp full_sn
Output: <exp>/seq/<tag>_seed<s>/results.md, results.json, rules.md, rules_all.md, models;
<exp>/seq/<tag>_seeds<0-1-2>/summary.md (mean ± sd, pooled paired tests, rules learned in >= 2 seeds).
"""
import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm import tqdm

from src.exp_roadpp import config, ego_flow, rtpt_utils, step05, step_dataset, utils_data, utils_pipe
from src.exp_roadpp.baselines.onset_mlp import OnsetMLP
from src.exp_roadpp.cauvid import TemporalRuleNet, memory_bins

MIN_STEP = 5            # frames a new ego state must hold to be a step (0.5 s at 10 fps)
NUM_STEPS = 3           # memory: the last 3 steps one by one, then bins of steps
SPLITS = ("train", "val", "test")
MARKOV_PRIOR = 5.0      # pseudo rows of the prior in the Markov estimates
MIN_HEAD_EVENTS = 5     # AP per head with at least these test onsets


# -------------- Steps --------------
def ego_steps(own, others, min_step=MIN_STEP):
    """Steps of one video. own (T, A) bool the ego's state per frame (frames where the ego is seen),
    others (T, A) bool the other agents' atoms. Returns (own (K, A), scene (K, A), scene_early (K, A),
    starts (K,)): a new state starts a step if it holds min_step frames; scene = the other agents'
    atoms over the step, scene_early over its first min_step frames."""
    T = own.shape[0]
    if T == 0:
        return None
    keys = [own[t].tobytes() for t in range(T)]
    starts, cur = [0], keys[0]
    t = 1
    while t < T:
        if keys[t] != cur and t + min_step <= T and all(keys[u] == keys[t] for u in range(t, t + min_step)):
            starts.append(t)
            cur = keys[t]
            t += min_step
            continue
        t += 1
    bounds = starts[1:] + [T]
    own_s = np.stack([own[s] for s in starts])
    scene = np.stack([others[s:e].any(axis=0) for s, e in zip(starts, bounds)])
    early = np.stack([others[s:min(s + min_step, e)].any(axis=0) for s, e in zip(starts, bounds)])
    return own_s, scene, early, np.array(starts)


def _fill_levels(F):
    """(T, 9) ego flow atoms: a frame without a level of a group keeps the previous frame's one."""
    F = F.copy()
    groups = [range(0, len(ego_flow.YAW_ATOMS)), range(len(ego_flow.YAW_ATOMS), len(ego_flow.ATOMS))]
    for g in groups:
        g = list(g)
        for t in range(1, len(F)):
            if not F[t, g].any():
                F[t, g] = F[t - 1, g]
    return F


def split_steps(atoms, num_label_atoms, flow_dir, min_step=MIN_STEP):
    """{video: ego_steps} of one split: the ego's label atoms plus its flow levels (atom index
    num_label_atoms + i of ego_flow.ATOMS), the other agents' label atoms as the scene."""
    out, skipped = {}, Counter()
    A = num_label_atoms + len(ego_flow.ATOMS)
    for vid, va in atoms.items():
        X = step05.build_video_tensor(va, num_label_atoms).numpy()
        ids = step05.video_agent_ids(va)
        ego = ids.get(None)
        raw = ego_flow.load_cache(flow_dir, vid)
        if ego is None or X.shape[0] < 2 * min_step:
            skipped["no ego / short"] += 1
            continue
        if raw is None:
            skipped["no flow cache"] += 1
            continue
        seen = X[:, ego].any(axis=1)
        F = _fill_levels(ego_flow.flow_atoms(raw, X.shape[0]))
        own = np.zeros((X.shape[0], A), dtype=bool)
        own[:, :num_label_atoms] = X[:, ego]
        own[:, num_label_atoms:] = F
        others = np.zeros((X.shape[0], A), dtype=bool)
        if X.shape[1] > 1:
            others[:, :num_label_atoms] = np.delete(X, ego, axis=1).any(axis=1)
        steps = ego_steps(own[seen], others[seen], min_step)
        if steps is not None and len(steps[0]) >= 2:
            out[vid] = steps
        else:
            skipped["fewer than 2 steps"] += 1
    return out, skipped


def step_rows(steps, num_steps=NUM_STEPS):
    """Packed inputs (B, 2 * 2 * bins * A) in `TemporalRuleNet.layer_inputs` layout (truth then known;
    scope, bin, atom), targets (B, A), at_risk (B, A), video index (B,) of every change point of
    {video: ego_steps}."""
    bins = memory_bins(num_steps, True)
    xs, ks, tg, rk, vids = [], [], [], [], []
    for v, (vid, (own, scene, early, _starts)) in enumerate(steps.items()):
        K, A = own.shape
        for t in range(1, K):
            x = np.zeros((2, len(bins), A), dtype=bool)
            k = np.zeros((2, len(bins), A), dtype=bool)
            for i, (first, last) in enumerate(bins):
                lo = 0 if last is None else t - last           # steps t - last .. t - first
                hi = t - first
                if hi < 0:
                    continue                                    # every step of the bin before the video
                known = last is None or lo >= 0
                lo = max(lo, 0)
                idx = list(range(lo, hi + 1))
                x[0, i] = own[idx].any(axis=0)
                sc = [scene[j] if j != t - 1 else early[j] for j in idx]   # the current step: its start only
                x[1, i] = np.any(sc, axis=0)
                k[:, i] = known
            k |= x
            xs.append(x.reshape(-1))
            ks.append(k.reshape(-1))
            tg.append(own[t] & ~own[t - 1])
            rk.append(~own[t - 1])
            vids.append(v)
    f = lambda a: torch.from_numpy(np.stack(a))
    return {"inputs": torch.cat([f(xs).float(), f(ks).float()], dim=1), "targets": f(tg), "at_risk": f(rk),
            "video": torch.tensor(vids), "videos": list(steps)}


# -------------- Training --------------
def _batches(n, size, shuffle, gen):
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    return [order[i:i + size] for i in range(0, n, size)]


def _loss(model, rows, idx, heads, device, pos_weight):
    hz, _ = model(rows["inputs"][idx].to(device))
    at_risk = rows["at_risk"][idx].to(device) & heads
    known = torch.ones((len(idx), 1, hz.shape[2]), dtype=torch.bool, device=device)
    return model.loss(hz, rows["targets"][idx].to(device)[:, None, :], at_risk, known, pos_weight)


def fit(model, train, val, heads, cfg, rule_net):
    """Adam (the rule net: its gates with their own lr, the L1 of the mask after warm-up), early stopping
    on the val loss (from the warm-up on for the rule net); the best val epoch is kept."""
    device, gen = cfg["device"], torch.Generator().manual_seed(cfg["seed"])
    if rule_net:
        others = [p for n, p in model.named_parameters() if n != "log_alpha"]
        opt = torch.optim.Adam([{"params": [model.log_alpha], "lr": 0.05, "eps": 1e-20},
                                {"params": others, "lr": 0.01}])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, wait, start = float("inf"), None, 0, (cfg["warmup"] if rule_net else 0)
    for epoch in range(cfg["epochs"]):
        model.train()
        for idx in _batches(len(train["targets"]), cfg["batch"], True, gen):
            loss = _loss(model, train, idx, heads, device, cfg["pos_weight"])
            if rule_net and epoch >= cfg["warmup"]:
                loss = loss + cfg["l0_lambda"] * model.sparsity_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(float(_loss(model, val, idx, heads, device, cfg["pos_weight"])) * len(idx)
                     for idx in _batches(len(val["targets"]), cfg["batch"], False, gen)) / max(len(val["targets"]), 1)
        if epoch >= start and vl < best - 1e-5:
            best, best_state, wait = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        elif epoch >= start:
            wait += 1
            if wait >= cfg["patience"]:
                break
        if epoch % 10 == 0:
            print(f"  epoch {epoch}: val loss {vl:.4f} (best {best:.4f})")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def model_scores(model, rows, device, batch=4096):
    out = [model.window_probs(model(rows["inputs"][i:i + batch].to(device))[0])[:, 0].cpu()
           for i in range(0, len(rows["targets"]), batch)]
    return torch.cat(out) if out else torch.zeros((0, rows["targets"].shape[1]))


# -------------- Baselines --------------
def markov_scores(train_steps, rows_steps, A, order):
    """(B, A) next-step onset rates of each row given its last `order` own states (train counts, smoothed
    toward the lower order; order 0 = the base rate)."""
    def keys(own, t, o):
        return tuple(own[t - j].tobytes() for j in range(1, o + 1) if t - j >= 0)
    count = [defaultdict(lambda: np.zeros(A)) for _ in range(order + 1)]
    risk = [defaultdict(lambda: np.zeros(A)) for _ in range(order + 1)]
    for own, *_ in train_steps.values():
        for t in range(1, len(own)):
            for o in range(order + 1):
                key = keys(own, t, o)
                count[o][key] += own[t] & ~own[t - 1]
                risk[o][key] += ~own[t - 1]
    out = []
    for own, *_ in rows_steps.values():
        for t in range(1, len(own)):
            p = count[0][()] / np.maximum(risk[0][()], 1)
            for o in range(1, order + 1):
                key = keys(own, t, o)
                if key in risk[o]:
                    p = (count[o][key] + MARKOV_PRIOR * p) / (risk[o][key] + MARKOV_PRIOR)
            out.append(p)
    return torch.tensor(np.stack(out), dtype=torch.float32)



# -------------- Metrics --------------
def _ap(y, s):
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    if y.sum() == 0:
        return None
    hits = np.cumsum(y)
    return float((hits[y > 0] / (np.nonzero(y)[0] + 1)).mean())


def _row_hits(scores, rows, heads, label_heads):
    """Per test row: at risk / target masks of the ego heads, top-1 / top-3 hits, label top-1 hits."""
    s = scores.clone()
    risk = rows["at_risk"] & heads[None]
    tgt = rows["targets"] & risk
    s[~risk] = -1.0
    has = tgt.any(dim=1)
    top = s.topk(3, dim=1).indices
    lab = label_heads[None] & risk
    lab_rows = (tgt & label_heads[None]).any(dim=1)
    sl = torch.where(lab, scores, torch.full_like(scores, -1.0))
    return {"risk": risk.numpy(), "tgt": tgt.numpy(), "scores": scores.numpy(), "has": has.numpy(),
            "hit1": (tgt.gather(1, top[:, :1]).any(dim=1) & has).numpy(),
            "hit3": (tgt.gather(1, top).any(dim=1) & has).numpy(),
            "lab_rows": lab_rows.numpy(),
            "lab_hit": (tgt.gather(1, sl.argmax(dim=1, keepdim=True))[:, 0] & lab_rows).numpy()}


def _metrics(R, sel, map_heads):
    """top-1, label top-1, mAP (over map_heads, those with an onset in sel) and micro AP of the rows sel
    (indices, repeats allowed: a bootstrap resample)."""
    has, lab = R["has"][sel], R["lab_rows"][sel]
    risk, tgt, sc = R["risk"][sel], R["tgt"][sel], R["scores"][sel]
    aps = [a for a in (_ap(tgt[risk[:, h], h], sc[risk[:, h], h]) for h in map_heads) if a is not None]
    return {"top1": float(R["hit1"][sel][has].mean()) if has.any() else np.nan,
            "label_top1": float(R["lab_hit"][sel][lab].mean()) if lab.any() else np.nan,
            "map": float(np.mean(aps)) if aps else np.nan,
            "micro_ap": _ap(tgt[risk], sc[risk]) or np.nan}


PAIRED_METRICS = ("top1", "label_top1", "map", "micro_ap")


def _resamples(rows, n, seed):
    """n bootstrap resamples of the test rows by video (the rows of a video stay together)."""
    v = rows["video"].numpy()
    by_vid = {u: np.nonzero(v == u)[0] for u in np.unique(v)}
    vids, rng = list(by_vid), np.random.default_rng(seed)
    return [np.concatenate([by_vid[vids[i]] for i in rng.integers(0, len(vids), len(vids))]) for _ in range(n)]


def _map_heads(rows, heads, names):
    tgt = (rows["targets"] & rows["at_risk"] & heads[None]).sum(dim=0)
    return [h for h in torch.nonzero(heads).flatten().tolist() if tgt[h] >= MIN_HEAD_EVENTS]


def evaluate(scores, rows, heads, label_heads, names, resamples):
    """top-1 (bootstrap CI over the test videos) / top-3, label top-1, AP per head, mAP and micro AP of
    (B, A) scores on the test rows."""
    R = _row_hits(scores, rows, heads, label_heads)
    has, lab = R["has"], R["lab_rows"]
    boot = [R["hit1"][sel][has[sel]].mean() for sel in resamples if has[sel].any()]
    per_head = {}
    for h in _map_heads(rows, heads, names):
        r = R["risk"][:, h]
        per_head[names[h]] = {"onsets": int(R["tgt"][r, h].sum()), "ap": _ap(R["tgt"][r, h], R["scores"][r, h])}
    aps = [d["ap"] for d in per_head.values() if d["ap"] is not None]
    return {"rows": int(has.sum()), "top1": float(R["hit1"][has].mean()) if has.any() else None,
            "top1_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None,
            "top3": float(R["hit3"][has].mean()) if has.any() else None,
            "label_rows": int(lab.sum()), "label_top1": float(R["lab_hit"][lab].mean()) if lab.any() else None,
            "map": float(np.mean(aps)) if aps else None, "micro_ap": _ap(R["tgt"][R["risk"]], R["scores"][R["risk"]]),
            "per_head": per_head}


def paired_bootstrap(scores_a, scores_b, rows, heads, label_heads, names, resamples):
    """Difference A - B of each paired metric, A averaged over its seeds (a list of (B, A) scores), on the
    same video resamples for both sides: the observed difference, its 95% CI and a two-sided bootstrap p
    (2 x the share of resamples on the other side of 0)."""
    hm = _map_heads(rows, heads, names)
    RA = [_row_hits(s, rows, heads, label_heads) for s in scores_a]
    RB = _row_hits(scores_b, rows, heads, label_heads)
    full = np.arange(len(rows["targets"]))

    def diff(sel):
        ma = [_metrics(R, sel, hm) for R in RA]
        mb = _metrics(RB, sel, hm)
        return {k: float(np.nanmean([m[k] for m in ma]) - mb[k]) for k in PAIRED_METRICS}
    obs, boot = diff(full), [diff(sel) for sel in resamples]
    out = {}
    for k in PAIRED_METRICS:
        d = np.array([b[k] for b in boot])
        d = d[~np.isnan(d)]
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else None
        out[k] = {"diff": obs[k], "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else None,
                  "p": p, "n": len(d)}
    return out


def _paired_lines(tests):
    """Markdown rows of {comparison: paired_bootstrap}."""
    lines = ["| Comparison | " + " | ".join(f"Δ {k}" for k in PAIRED_METRICS) + " |",
             "|---|" + "---|" * len(PAIRED_METRICS)]
    for name, t in tests.items():
        cells = [f"{t[k]['diff']:+.3f} [{t[k]['ci'][0]:+.3f}, {t[k]['ci'][1]:+.3f}] "
                 + (f"p={t[k]['p']:.3f}" if t[k]["p"] > 0 else f"p<{1 / t[k]['n']:.3f}")
                 if t[k]["ci"] else "–" for k in PAIRED_METRICS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


# -------------- Rules --------------
TAUTOLOGY = 0.9         # a body that fires on >= 90% of a head's at-risk train rows is a bias, not a rule
MIN_WEIGHT = 0.1        # (rule, head) pairs with |weight| below this are left out of the rules and the scorecards
MIN_IMPORTANCE = 1e-4   # a pair is kept if removing it raises its head's mean val loss by more than this


def _span_text(model, m):
    """The steps a body reads: "now" (the step just entered), "k back", "a..b back", "a+ back" (any earlier)."""
    a, b = model.decode_span(m)
    first, last = model.bins[a - 1][0], model.bins[b - 1][1]
    if (first, last) == (1, 1):
        return "now"
    if last is None:
        return f"{first - 1}+ back" if first > 1 else "now or any earlier"
    lo, hi = first - 1, last - 1                      # steps before the current one
    return (f"{hi} back" if lo == hi else f"{lo}..{hi} back") if lo > 0 else f"now..{hi} back"


@torch.no_grad()
def rule_groups(model, names):
    """The distinct rule bodies (inputs with mask > 0.5) of the rule net: [{"rules": [r, ...], "text"}].
    Rules with the same body are one group (their head weights add up); a body without an own-ego input is
    left out (the self gate of `require_self` mutes it)."""
    body = model.rule_mask() > 0.5
    groups = {}
    for r in range(body.shape[0]):
        inputs = torch.nonzero(body[r]).flatten().tolist()
        if not inputs or (model.require_self and not model.self_inputs[inputs].any()):
            continue
        groups.setdefault(tuple(inputs), []).append(r)
    out = []
    for inputs, rules in groups.items():
        inputs = sorted(inputs, key=lambda m: (-model.decode_span(m)[1], -model.decode_span(m)[0]))
        text = " ∧ ".join(f"{names[model.decode_input(m)[0]]}({'ego' if model.decode_input(m)[1] == 0 else 'scene'}"
                          f", {_span_text(model, m)})" for m in inputs)
        out.append({"rules": rules, "inputs": list(inputs), "text": text})
    return out


@torch.no_grad()
def group_fires(model, groups, rows, device):
    """(B, G) bool: the body of each group holds (every input true) at each row."""
    if not groups:
        return torch.zeros((len(rows["targets"]), 0), dtype=torch.bool)
    b = torch.zeros((len(groups), model.num_inputs), device=device)
    for g, grp in enumerate(groups):
        b[g, grp["inputs"]] = 1.0
    out = []
    for i in range(0, len(rows["targets"]), 2048):
        x, _k = model.split_inputs(rows["inputs"][i:i + 2048].to(device))
        out.append((x @ b.T >= b.sum(dim=1)).cpu())
    return torch.cat(out)


def _membership(groups, R):
    M = torch.zeros((R, len(groups)))
    for g, grp in enumerate(groups):
        M[grp["rules"], g] = 1.0
    return M


def _onset_bce(z, y, pos_weight):
    """Per-row BCE of the onset probability 1 - exp(-softplus(z)) (`TemporalRuleNet.loss`, one window)."""
    log_none = -torch.nn.functional.softplus(z)
    log_onset = torch.log(-torch.expm1(log_none.clamp(max=-1e-7)))
    return -torch.where(y, pos_weight * log_onset, log_none)


@torch.no_grad()
def loo_importance(model, groups, val, heads, cfg):
    """(A, G) leave-one-out importance on the val rows: the increase of a head's mean val loss (over its
    at-risk rows) when the head weights of the group are set to 0 in the rule net, the rest unchanged.
    > 0: the group helps the head; <= 0: the head does as well (or better) without it."""
    device = cfg["device"]
    acts = torch.cat([model(val["inputs"][i:i + 2048].to(device))[1].cpu()
                      for i in range(0, len(val["targets"]), 2048)])                       # (B, R) soft
    W, b = model.head_weight.detach().cpu(), model.head_bias[0].detach().cpu()
    M = _membership(groups, W.shape[1])
    hz = acts @ W.T + b
    out = torch.zeros((W.shape[0], len(groups)))
    for h in torch.nonzero(heads).flatten().tolist():
        risk = val["at_risk"][:, h]
        if not risk.any():
            continue
        y = val["targets"][risk, h]
        contrib = (acts[risk] * W[h][None]) @ M                                             # (n, G)
        base = _onset_bce(hz[risk, h], y, cfg["pos_weight"]).mean()
        alt = _onset_bce(hz[risk, h][:, None] - contrib, y[:, None], cfg["pos_weight"]).mean(dim=0)
        out[h] = alt - base
    return out


def rule_stats(model, rows, heads, names, cfg):
    """Every (group, ego head) pair with |summed head weight| >= MIN_WEIGHT: its body, weight, val
    leave-one-out importance (`loo_importance`), and on train / test its fires (the body holds at an
    at-risk row of the head), hits, precision, the head's base rate, coverage; lift = train precision /
    base rate. keep: importance > MIN_IMPORTANCE and a train fire (the scorecard rules). kind (a description, not a filter):
    "rule" lift > 1, "below" lift <= 1 or no train fire, "always" fires on >= TAUTOLOGY of the at-risk rows."""
    groups = rule_groups(model, names)
    fires = {sp: group_fires(model, groups, rows[sp], cfg["device"]) for sp in SPLITS}
    # Bodies that hold at the same rows of every split (e.g. a(now) and a(now..1 back) ∧ a(now)) are one
    # group, named by the shortest body.
    merged = {}
    for g in range(len(groups)):
        key = b"".join(fires[sp][:, g].numpy().tobytes() for sp in SPLITS)
        merged.setdefault(key, []).append(g)
    keep_cols = []
    for gs in merged.values():
        main = min(gs, key=lambda g: (len(groups[g]["inputs"]), len(groups[g]["text"])))
        groups[main]["rules"] = sorted(r for g in gs for r in groups[g]["rules"])
        keep_cols.append(main)
    keep_cols.sort()
    groups = [groups[g] for g in keep_cols]
    fires = {sp: f[:, keep_cols] for sp, f in fires.items()}
    W = model.head_weight.detach().cpu() @ _membership(groups, model.head_weight.shape[1])     # (A, G)
    imp = loo_importance(model, groups, rows["val"], heads, cfg)
    out = []
    for h in torch.nonzero(heads).flatten().tolist():
        for g, grp in enumerate(groups):
            w = float(W[h, g])
            if abs(w) < MIN_WEIGHT:
                continue
            e = {"group": g, "head_index": h, "head": names[h], "weight": w, "body": grp["text"],
                 "copies": len(grp["rules"]), "importance": float(imp[h, g])}
            for split in ("train", "test"):
                r = rows[split]
                risk = r["at_risk"][:, h]
                f = fires[split][:, g] & risk
                e[f"{split}_fires"] = int(f.sum())
                e[f"{split}_hits"] = int(r["targets"][f, h].sum())
                e[f"{split}_precision"] = float(r["targets"][f, h].float().mean()) if f.any() else None
                e[f"{split}_base"] = float(r["targets"][risk, h].float().mean()) if risk.any() else None
                e[f"{split}_coverage"] = float(f.sum()) / max(int(risk.sum()), 1)
            p, b0 = e["train_precision"], e["train_base"]
            e["lift"] = p / b0 if p is not None and b0 else None
            e["kind"] = ("always" if e["train_coverage"] >= TAUTOLOGY
                         else "rule" if e["lift"] is not None and e["lift"] > 1 else "below")
            e["keep"] = e["importance"] > MIN_IMPORTANCE and e["train_fires"] > 0
            out.append(e)
    return out, groups, fires


def _is_clean(e, cfg):
    """A rule of the rule list: above the head's base rate, train support >= min_rule_support and >=
    min_rule_hits train onsets it predicts."""
    return (e["kind"] == "rule" and e["train_fires"] >= cfg["min_rule_support"]
            and e["train_hits"] >= cfg["min_rule_hits"])


def rule_list_scores(stats, fires, train, cfg):
    """Scores of single rules (reference): per row and ego head, the largest train precision of the
    `_is_clean` rules of the head whose body holds, else the head's train base rate."""
    risk, tgt = train["at_risk"], train["targets"] & train["at_risk"]
    base = tgt.sum(dim=0).float() / risk.sum(dim=0).clamp(min=1).float()
    scores = base[None].repeat(len(fires), 1)
    for e in stats:
        if _is_clean(e, cfg):
            h, f = e["head_index"], fires[:, e["group"]]
            scores[f, h] = torch.maximum(scores[f, h], torch.tensor(e["train_precision"]))
    return scores


def scorecard_net(model, stats, fires):
    """The scorecard with the rule net's weights: hazard of a head = its bias + the weights of its kept
    groups whose body holds (hard 0/1, no soft masks); the onset probability as in the rule net."""
    A = model.head_weight.shape[0]
    W = torch.zeros((A, fires.shape[1]))
    for e in stats:
        if e["keep"]:
            W[e["head_index"], e["group"]] = e["weight"]
    hz = fires.float() @ W.T + model.head_bias[0].detach().cpu()
    return TemporalRuleNet.window_probs(hz[:, None])[:, 0]


def refit_scorecard(stats, fires_train, train, heads, l2=1e-3, steps=200):
    """Per ego head a logistic regression (bias + one weight per kept group, L2) on the hard fires of the
    train at-risk rows: the readable model, the rule net's bodies with relearned weights. Returns
    {head: (bias, {group: weight}, base rate)}; a head without kept groups gets the logit of its base rate."""
    card = {}
    for h in torch.nonzero(heads).flatten().tolist():
        risk = train["at_risk"][:, h]
        if not risk.any():
            continue
        y = train["targets"][risk, h].float()
        gs = sorted({e["group"] for e in stats if e["keep"] and e["head_index"] == h})
        p0 = float(y.mean().clamp(1e-4, 1 - 1e-4))
        bias = torch.tensor([np.log(p0 / (1 - p0))], dtype=torch.float32, requires_grad=True)
        w = torch.zeros(len(gs), requires_grad=True)
        if gs:
            X = fires_train[risk][:, gs].float()
            opt = torch.optim.LBFGS([bias, w], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + bias, y) + l2 * (w ** 2).sum()
                loss.backward()
                return loss
            opt.step(closure)
        card[h] = (float(bias), {g: float(v) for g, v in zip(gs, w.detach())}, float(y.mean()))
    return card


def scorecard_scores(card, fires, A):
    """(B, A) probabilities of a refit scorecard; heads without a card get 0."""
    out = torch.zeros((len(fires), A))
    for h, (bias, ws, _p0) in card.items():
        z = torch.full((len(fires),), bias)
        for g, v in ws.items():
            z = z + v * fires[:, g].float()
        out[:, h] = torch.sigmoid(z)
    return out


@torch.no_grad()
def kept_net_scores(model, stats, groups, rows, device):
    """Test scores of the (soft) rule net with only the kept (group, head) pairs: every other head weight
    set to 0. If the metrics hold, the dropped pairs were not needed."""
    saved = model.head_weight.data.clone()
    keep = torch.zeros_like(saved, dtype=torch.bool)
    for e in stats:
        if e["keep"]:
            keep[e["head_index"], groups[e["group"]]["rules"]] = True
    model.head_weight.data[~keep] = 0.0
    try:
        return model_scores(model, rows, device)
    finally:
        model.head_weight.data.copy_(saved)


def rule_report(stats, card, names, cfg):
    """Markdown of the rules: the scorecard (kept pairs per head, by importance, with the net weight and
    the refit weight, and the refit bias of the head) and the full table of every pair."""
    fmt = lambda v, d=2: "–" if v is None else f"{v:.{d}f}"
    head = ["| Head | Rule (body ⇒ next step) | Net weight | Refit weight | Δ val loss | Train fires | Train hits "
            "| Train prec. | Lift | Test fires | Test prec. |", "|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(e):
        copies = f" (×{e['copies']})" if e["copies"] > 1 else ""
        refit = card[e["head_index"]][1].get(e["group"]) if e["keep"] and e["head_index"] in card else None
        return (f"| {e['head']} | {e['body']}{copies} | {e['weight']:+.2f} | {fmt(refit)} | {e['importance']:+.4f} | "
                f"{e['train_fires']} | {e['train_hits']} | {fmt(e['train_precision'])} | {fmt(e['lift'])} | "
                f"{e['test_fires']} | {fmt(e['test_precision'])} |")
    kept = sorted((e for e in stats if e["keep"]), key=lambda e: (e["head"], -e["importance"]))
    n_heads = len({e["head"] for e in kept})
    summary = (f"{len(stats)} (rule body, ego head) pairs with |weight| >= {MIN_WEIGHT}; {len(kept)} help their head "
               f"on val (leave-one-out Δ val loss > {MIN_IMPORTANCE}) and fire on train; they form the scorecard of {n_heads} heads; "
               f"{len(stats) - len(kept)} do not.")
    bias_lines = ["| Head | Refit bias | Base rate (train) | Rules |", "|---|---|---|---|"]
    for h, (bias, ws, p0) in sorted(card.items(), key=lambda kv: names[kv[0]]):
        bias_lines.append(f"| {names[h]} | {bias:+.2f} | {p0:.3f} | {len(ws)} |")
    clean_lines = [summary, "",
                   "Scorecard: P(head switches on next) = sigmoid(refit bias + sum of the refit weights of its rules "
                   "whose body holds).", ""] + head + [row(e) for e in kept] + ["", "Refit bias per head:", ""] + bias_lines
    full = sorted(stats, key=lambda e: (e["head"], not e["keep"], -e["importance"]))
    full_lines = [summary, "", head[0].replace("| Head |", "| Head | Kept | Kind |"), "|---|---" + head[1]]
    full_lines += [row(e).replace(f"| {e['head']} |", f"| {e['head']} | {'yes' if e['keep'] else 'no'} | {e['kind']} |", 1)
                   for e in full]
    return kept, clean_lines, full_lines


# -------------- Main --------------
def ensure_dataset(exp, machine, device):
    """Run step_dataset for the exp (its exp_config: split, far-vehicle filter, split_from) if its atoms
    are missing, e.g. an exp whose pipeline has not run yet; it skips the files it already has."""
    pargs = utils_pipe.get_parser().parse_args(["--exp", exp, "--machine", machine, "--device", device])
    for key, value in utils_data.load_json(config.exp_config_path / f"{exp}.json").items():
        setattr(pargs, key, value)
    pargs = config.machine_setup(pargs)
    step_dataset.main(config.get_step_dataset_input(pargs))


def _fmt(v):
    return "–" if v is None else f"{v:.3f}"


def _model_table(results):
    md = ["| Method | Top-1 [95% CI] | Top-3 | Label top-1 | mAP | Micro AP |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        ci = f" [{r['top1_ci'][0]:.3f}, {r['top1_ci'][1]:.3f}]" if r["top1_ci"] else ""
        md.append(f"| {name} | {_fmt(r['top1'])}{ci} | {_fmt(r['top3'])} | {_fmt(r['label_top1'])} | "
                  f"{_fmt(r['map'])} | {_fmt(r['micro_ap'])} |")
    return md


RULE_NET, MLP, MARKOV1, MARKOV2 = "Rule net (ours)", "OnsetMLP", "Markov-1 (current state)", "Markov-2 (last two states)"
RULE_LIST = "Rule list (best single rule)"
CARD_NET = "Scorecard (net weights)"
CARD_FIT = "Scorecard (refit weights)"


def run_seed(seed, args, rows, A, heads, label_heads, names, base_scores, resamples, out_dir, device, rtpt):
    """Train the rule net and the MLP with one seed, evaluate them with the baselines, paired tests, rules."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    heads_d = heads.to(device)
    cfg = {"device": device, "seed": seed, "epochs": args.epochs, "warmup": 10, "patience": 20, "batch": 512,
           "l0_lambda": 1e-3, "pos_weight": 1.0, "min_rule_support": 10, "min_rule_hits": args.min_rule_hits}
    rule_net = TemporalRuleNet(A, 1, args.num_steps, args.num_rules, -11.0, "sigmoid", (1,), require_self=True,
                               ranges=True).to(device)
    mlp = OnsetMLP(A, 1, args.num_steps, args.num_rules, 0.2, (1,), open_memory=True).to(device)
    saved = (out_dir / "rule_net.pt").exists() and (out_dir / "mlp.pt").exists()
    if args.eval_only and saved:
        print(f"[Seq] seed {seed}: loading the trained models of {out_dir}")
        rule_net.load_state_dict(torch.load(out_dir / "rule_net.pt", map_location=device))
        mlp.load_state_dict(torch.load(out_dir / "mlp.pt", map_location=device))
        rule_net.eval(), mlp.eval()
    else:
        if args.eval_only:
            print(f"[Seq] seed {seed}: no saved models in {out_dir}, training")
        print(f"[Seq] seed {seed}: rule net")
        rule_net = fit(rule_net, rows["train"], rows["val"], heads_d, cfg, True)
        rtpt_utils.step(rtpt, f"seed {seed} rule net")
        print(f"[Seq] seed {seed}: MLP")
        mlp = fit(mlp, rows["train"], rows["val"], heads_d, cfg, False)
        rtpt_utils.step(rtpt, f"seed {seed} mlp")
        torch.save(rule_net.state_dict(), out_dir / "rule_net.pt")
        torch.save(mlp.state_dict(), out_dir / "mlp.pt")

    test = rows["test"]
    stats, groups, fires = rule_stats(rule_net, rows, heads, names, cfg)
    card = refit_scorecard(stats, fires["train"], rows["train"], heads)
    kept, card_lines, full_lines = rule_report(stats, card, names, cfg)
    scores = dict(base_scores)
    scores[MLP] = model_scores(mlp, test, device)
    scores[RULE_NET] = model_scores(rule_net, test, device)
    scores[CARD_FIT] = scorecard_scores(card, fires["test"], A)
    scores[CARD_NET] = scorecard_net(rule_net, stats, fires["test"])
    scores[RULE_LIST] = rule_list_scores(stats, fires["test"], rows["train"], cfg)
    ev = lambda s: evaluate(s, test, heads, label_heads, names, resamples)
    results = {name: ev(s) for name, s in scores.items()}
    pb = lambda a, b: paired_bootstrap([scores[a]], scores[b], test, heads, label_heads, names, resamples)
    paired = {f"{RULE_NET} − {MARKOV2}": pb(RULE_NET, MARKOV2), f"{RULE_NET} − {MARKOV1}": pb(RULE_NET, MARKOV1),
              f"{RULE_NET} − {MLP}": pb(RULE_NET, MLP), f"{MLP} − {MARKOV2}": pb(MLP, MARKOV2),
              f"{CARD_FIT} − {MARKOV2}": pb(CARD_FIT, MARKOV2), f"{CARD_FIT} − {RULE_NET}": pb(CARD_FIT, RULE_NET),
              f"{CARD_NET} − {MARKOV2}": pb(CARD_NET, MARKOV2)}
    kept_net = ev(kept_net_scores(rule_net, stats, groups, test, device))
    rtpt_utils.step(rtpt, f"seed {seed} eval")

    md = ["## 1. Models (test change points)\n"] + _model_table(results)
    md.append(f"\nRows: {results[RULE_NET]['rows']}; label rows (the next step changes the action label): "
              f"{results[RULE_NET]['label_rows']}. Scorecards: the kept rules (section 4) with hard 0/1 bodies; "
              "rule list: the best single rule above the base rate (reference).\n")
    md += ["## 2. Paired bootstrap (same test videos resampled for both sides)\n",
           f"Difference of the first minus the second method, 95% CI over {len(resamples)} resamples of the test "
           "videos, two-sided bootstrap p.\n"] + _paired_lines(paired) + [""]
    md += ["## 3. AP per head (test, >= 5 onsets)\n", "| Head | Onsets | " + " | ".join(results) + " |",
           "|---|---|" + "---|" * len(results)]
    for h in sorted(results[RULE_NET]["per_head"], key=lambda h: -results[RULE_NET]["per_head"][h]["onsets"]):
        cells = [_fmt(results[n]["per_head"].get(h, {}).get("ap")) for n in results]
        md.append(f"| {h} | {results[RULE_NET]['per_head'][h]['onsets']} | " + " | ".join(cells) + " |")
    md += ["", "## 4. Leave-one-out pruning: which rules the rule net needs\n",
           "Each (rule body, head) pair is removed alone on the val rows (head weight 0); kept if the head's mean val "
           f"loss rises by > {MIN_IMPORTANCE} and the body fires on train. The rule net with only the kept pairs (soft, as trained):\n",
           "| Rule net | Pairs | Top-1 | Label top-1 | mAP | Micro AP |", "|---|---|---|---|---|---|"]
    for k, r, n in (("all pairs", results[RULE_NET], len(stats)), ("kept pairs only", kept_net, len(kept))):
        md.append(f"| {k} | {n} | {_fmt(r['top1'])} | {_fmt(r['label_top1'])} | {_fmt(r['map'])} | {_fmt(r['micro_ap'])} |")
    md += ["", "## 5. Scorecard rules (kept pairs, by val importance)\n"] + card_lines + ["", "All pairs: rules_all.md."]
    (out_dir / "rules.md").write_text("\n".join(card_lines) + "\n")
    (out_dir / "rules_all.md").write_text("\n".join(full_lines) + "\n")
    (out_dir / "results.json").write_text(json.dumps(
        {"args": vars(args), "seed": seed, "results": results, "paired": paired, "kept_net": kept_net,
         "rules": stats, "scorecard": {names[h]: {"bias": c[0], "base_rate": c[2],
                                                  "rules": {groups[g]["text"]: v for g, v in c[1].items()}}
                                       for h, c in card.items()}}, indent=1, default=str))
    return {"md": md, "results": results, "clean": kept,
            "scores": {k: scores[k] for k in (RULE_NET, MLP, CARD_FIT, CARD_NET, RULE_LIST)}}


def summary_md(per_seed, base_scores, rows, heads, label_heads, names, resamples, seeds):
    """Across the seeds: mean ± sd per method, paired tests of the seed-averaged models against the
    baselines, and the rules (head, body) learned in several seeds."""
    test = rows["test"]
    md = [f"## Across seeds {', '.join(map(str, seeds))}\n",
          "| Method | Top-1 | Top-3 | Label top-1 | mAP | Micro AP |", "|---|---|---|---|---|---|"]
    for name in per_seed[0]["results"]:
        cells = []
        for k in ("top1", "top3", "label_top1", "map", "micro_ap"):
            v = [r["results"][name][k] for r in per_seed if r["results"][name][k] is not None]
            cells.append(f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "–")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    pooled = lambda a, b: paired_bootstrap([r["scores"][a] for r in per_seed], base_scores[b], test, heads,
                                           label_heads, names, resamples)
    paired = {f"{RULE_NET} (seed mean) − {MARKOV2}": pooled(RULE_NET, MARKOV2),
              f"{RULE_NET} (seed mean) − {MARKOV1}": pooled(RULE_NET, MARKOV1),
              f"{MLP} (seed mean) − {MARKOV2}": pooled(MLP, MARKOV2),
              f"{CARD_FIT} (seed mean) − {MARKOV2}": pooled(CARD_FIT, MARKOV2),
              f"{CARD_FIT} (seed mean) − {MARKOV1}": pooled(CARD_FIT, MARKOV1),
              f"{CARD_NET} (seed mean) − {MARKOV2}": pooled(CARD_NET, MARKOV2)}
    md += ["", "Paired bootstrap of the seed-averaged metric (the same video resamples for every seed and the "
           "baseline):\n"] + _paired_lines(paired)
    seen = defaultdict(list)
    for s, r in zip(seeds, per_seed):
        for e in r["clean"]:
            seen[(e["head"], e["body"])].append(e)
    stable = sorted(((k, v) for k, v in seen.items() if len(v) >= 2),
                    key=lambda kv: (-len(kv[1]), kv[0][0], -np.mean([x["importance"] for x in kv[1]])))
    md += ["", f"### Rules learned in >= 2 of {len(seeds)} seeds ({len(stable)} of {len(seen)} distinct scorecard rules)\n",
           "| Head | Rule | Seeds | Mean Δ val loss | Train prec. | Lift | Test fires | Test prec. |",
           "|---|---|---|---|---|---|---|---|"]
    for (h, b), es in stable:
        e = es[0]          # same body and head: the same fires and precisions in every seed
        imp = np.mean([x["importance"] for x in es])
        md.append(f"| {h} | {b} | {len(es)} | {imp:+.4f} | {_fmt(e['train_precision'])} | {_fmt(e['lift'])} | "
                  f"{e['test_fires']} | {_fmt(e['test_precision'])} |")
    return md, paired


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", default="medium_sn")
    ap.add_argument("--machine", default="dgx-jing")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_steps", type=int, default=NUM_STEPS)
    ap.add_argument("--min_step", type=int, default=MIN_STEP)
    ap.add_argument("--num_rules", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated training seeds")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples of the test videos")
    ap.add_argument("--ego_flow_workers", type=int, default=16)
    ap.add_argument("--min_rule_hits", type=int, default=5, help="train onsets a readable rule must predict")
    ap.add_argument("--eval_only", action="store_true",
                    help="load the saved rule_net.pt / mlp.pt of each seed instead of training")
    args, _unknown = ap.parse_known_args()
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    setup = config.machine_setup(args)
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    rtpt = rtpt_utils.start(f"roadpp-{args.exp}-seq", 1 + 3 * len(seeds))
    atoms_dir = setup.output_dir / "step_dataset_output" / "atoms"
    flow_dir = setup.output_dir.parent / "ego_flow_cache" / ego_flow.VERSION
    base_tag = f"s{args.num_steps}m{args.min_step}r{args.num_rules}_{ego_flow.tag().lstrip('_')}"
    if not all(os.path.exists(atoms_dir / f) for f in ["unique_atoms.json"]
               + [f"{sp}_atoms_compressed.json" for sp in SPLITS] + [f"{sp}_boxes.pt" for sp in SPLITS]):
        print(f"[Seq] the step_dataset output of {args.exp} is missing: preparing it")
        ensure_dataset(args.exp, args.machine, args.device)
    unique_atoms = utils_data.load_json(atoms_dir / "unique_atoms.json")
    L = len(unique_atoms)
    names = [step05._atom_name(sig) for sig in unique_atoms] + [f"av:{a}" for a in ego_flow.ATOMS]
    A = len(names)
    heads = torch.tensor([n.startswith("av:") for n in names])
    label_heads = heads.clone()
    label_heads[L:] = False

    steps, info = {}, {}
    spec = {"cache_dir": str(flow_dir), "frame_dir": str(setup.dataset_path / "frames"),
            "video_dir": str(setup.dataset_path / "videos"), "fps": 10, "workers": args.ego_flow_workers}
    for sp in SPLITS:
        atoms = utils_data.load_json(atoms_dir / f"{sp}_atoms_compressed.json")
        boxes = torch.load(atoms_dir / f"{sp}_boxes.pt", map_location="cpu")
        step05._ensure_ego_flow(atoms, boxes, spec)
        steps[sp], skipped = split_steps(atoms, L, flow_dir, args.min_step)
        n_steps = [len(s[0]) for s in steps[sp].values()]
        info[sp] = {"videos": len(steps[sp]), "skipped": dict(skipped), "change_points": int(sum(n - 1 for n in n_steps)),
                    "median_steps": float(np.median(n_steps)) if n_steps else 0}
        print(f"[Steps] {sp}: {info[sp]}")
        del atoms, boxes
    rtpt_utils.step(rtpt, "steps")
    rows = {sp: step_rows(steps[sp], args.num_steps) for sp in SPLITS}
    base_scores = {"Prior (base rate)": markov_scores(steps["train"], steps["test"], A, 0),
                   MARKOV1: markov_scores(steps["train"], steps["test"], A, 1),
                   MARKOV2: markov_scores(steps["train"], steps["test"], A, 2)}
    resamples = _resamples(rows["test"], args.boot, 0)      # one set of resamples for every seed and test

    intro = [f"At each change point (the ego enters a new state: its action label, yaw level, speed level), "
             "which ego atoms switch on at the next step; no time window. Steps hold >= "
             f"{args.min_step} frames; memory: the last {args.num_steps} steps, then bins of steps.\n",
             "| Split | Videos | Change points | Median steps / video |", "|---|---|---|---|"]
    intro += [f"| {sp} | {info[sp]['videos']} | {info[sp]['change_points']} | {info[sp]['median_steps']:.0f} |"
              for sp in SPLITS]
    per_seed = []
    for seed in seeds:
        out_dir = setup.output_dir / "seq" / f"{base_tag}_seed{seed}"
        r = run_seed(seed, args, rows, A, heads, label_heads, names, base_scores, resamples, out_dir, device, rtpt)
        md = [f"# Ego next-step prediction ({args.exp}, {base_tag}, seed {seed})\n"] + intro + [""] + r["md"]
        (out_dir / "results.md").write_text("\n".join(md) + "\n")
        print("\n".join(md))
        print(f"\n[Seq] {out_dir}")
        per_seed.append(r)
    if len(seeds) > 1:
        sum_dir = setup.output_dir / "seq" / f"{base_tag}_seeds{'-'.join(map(str, seeds))}"
        os.makedirs(sum_dir, exist_ok=True)
        md, paired = summary_md(per_seed, base_scores, rows, heads, label_heads, names, resamples, seeds)
        md = [f"# Ego next-step prediction ({args.exp}, {base_tag})\n"] + intro + [""] + md
        (sum_dir / "summary.md").write_text("\n".join(md) + "\n")
        (sum_dir / "summary.json").write_text(json.dumps(
            {"args": vars(args), "info": info, "paired": paired,
             "results": {s: r["results"] for s, r in zip(seeds, per_seed)}}, indent=1, default=str))
        print("\n".join(md))
        print(f"\n[Seq] {sum_dir}")


if __name__ == "__main__":
    main()
