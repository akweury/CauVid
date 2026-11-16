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

Rules (`rule_stats`): each (rule, ego head) pair gets its lift (train precision / the head's base rate) and
a kind: "rule" (lift > 1), "below" (lift <= 1: it only reorders inside the head) or "always" (fires on
>= 90% of the at-risk rows: a bias). rules.md keeps the "rule" pairs with train support >= 10, duplicate
bodies merged; rules_all.md lists all. A pruning check zeroes the head weights of the "below" (and
"always") pairs and re-evaluates: if the metrics hold, those pairs were noise. Readable rules (rules.md)
also predict >= --min_rule_hits train onsets. The rule list (`rule_list_scores`) predicts with them alone
(the largest train precision of the firing rules of a head, else its base rate), so its metrics are
those of the readable rules, apart from the rule net that found them.

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
def rule_stats(model, train, test, heads, names, cfg):
    """Every (rule, ego head) of the rule net: its body, weight, and on train / test its fires (the body
    holds at an at-risk row of the head), precision (the head switches on next), the head's base rate,
    coverage (fires / at-risk rows); lift = train precision / train base rate; kind:
      "rule"   lift > 1 (the body makes the head more likely than its base rate),
      "below"  lift <= 1 or no train fire (the rule ranks inside the head at best),
      "always" the body fires on >= TAUTOLOGY of the at-risk train rows (a bias term)."""
    body = model.rule_mask() > 0.5
    fired = {}
    for split, rows in (("train", train), ("test", test)):
        b = body.float()
        f = []
        for i in range(0, len(rows["targets"]), 2048):
            x, _k = model.split_inputs(rows["inputs"][i:i + 2048].to(cfg["device"]))
            f.append(((x @ b.T >= b.sum(dim=1)) & (b.sum(dim=1) > 0)).cpu())   # (B, R)
        fired[split] = torch.cat(f) if f else torch.zeros((0, len(b)), dtype=torch.bool)
    out = []
    for rule in model.extract_rules(0.5, 0.1):
        r = rule["rule"]
        inputs = torch.nonzero(body[r]).flatten().tolist()
        inputs.sort(key=lambda m: (-model.decode_span(m)[1], -model.decode_span(m)[0]))
        text = " ∧ ".join(f"{names[model.decode_input(m)[0]]}({'ego' if model.decode_input(m)[1] == 0 else 'scene'}"
                          f", {_span_text(model, m)})" for m in inputs)
        for h, w in rule["heads"]:
            if not heads[h]:
                continue
            e = {"rule": r, "head_index": h, "head": names[h], "weight": w, "body": text}
            for split, rows in (("train", train), ("test", test)):
                risk = rows["at_risk"][:, h]
                f = fired[split][:, r] & risk
                e[f"{split}_fires"] = int(f.sum())
                e[f"{split}_hits"] = int(rows["targets"][f, h].sum())
                e[f"{split}_precision"] = float(rows["targets"][f, h].float().mean()) if f.any() else None
                e[f"{split}_base"] = float(rows["targets"][risk, h].float().mean()) if risk.any() else None
                e[f"{split}_coverage"] = float(f.sum()) / max(int(risk.sum()), 1)
            p, b0 = e["train_precision"], e["train_base"]
            e["lift"] = p / b0 if p is not None and b0 else None
            e["kind"] = ("always" if e["train_coverage"] >= TAUTOLOGY
                         else "rule" if e["lift"] is not None and e["lift"] > 1 else "below")
            out.append(e)
    return out, fired


def _merge(entries):
    """One row per (head, body): rules with the same body (duplicates) get their weights summed."""
    merged = {}
    for e in entries:
        key = (e["head"], e["body"])
        if key in merged:
            merged[key]["weight"] += e["weight"]
            merged[key]["copies"] += 1
        else:
            merged[key] = dict(e, copies=1)
    return list(merged.values())


def _is_clean(e, cfg):
    """A readable rule: above the head's base rate, train support >= min_rule_support and >= min_rule_hits
    train onsets it predicts (a rare head's lift is large on a handful of hits)."""
    return (e["kind"] == "rule" and e["train_fires"] >= cfg["min_rule_support"]
            and e["train_hits"] >= cfg["min_rule_hits"])


def rule_report(stats, cfg):
    """Markdown of the rules: the clean table (`_is_clean`, duplicates merged, sorted by head then lift)
    and the full table with every kind."""
    fmt = lambda v: "–" if v is None else f"{v:.2f}"
    rows = _merge(stats)
    clean = sorted((e for e in rows if _is_clean(e, cfg)), key=lambda e: (e["head"], -e["lift"]))
    head = ["| Head | Rule (body ⇒ next step) | Weight | Train fires | Train hits | Train prec. | Base | Lift | Test fires "
            "| Test prec. |", "|---|---|---|---|---|---|---|---|---|---|"]
    def row(e):
        copies = f" (×{e['copies']})" if e["copies"] > 1 else ""
        return (f"| {e['head']} | {e['body']}{copies} | {e['weight']:.2f} | {e['train_fires']} | "
                f"{e['train_hits']} | {fmt(e['train_precision'])} | {fmt(e['train_base'])} | {fmt(e['lift'])} | {e['test_fires']} | "
                f"{fmt(e['test_precision'])} |")
    count = Counter(e["kind"] for e in stats)
    dup = len(stats) - len(rows)
    low = sum(1 for e in rows if e["kind"] == "rule" and not _is_clean(e, cfg))
    summary = (f"{len(stats)} (rule, ego head) pairs with weight > 0.1: {count['rule']} above the base rate, "
               f"{count['below']} at or below it (or never firing on train), {count['always']} always on "
               f"(a bias: the body fires on >= {TAUTOLOGY:.0%} of the head's at-risk rows); {dup} duplicates merged; "
               f"{low} above-base rules with train support < {cfg['min_rule_support']} or < {cfg['min_rule_hits']} "
               f"train hits left out. Kept: {len(clean)}.")
    clean_lines = [summary, ""] + head + [row(e) for e in clean]
    full = sorted(rows, key=lambda e: (e["head"], e["kind"], -(e["lift"] or 0)))
    full_lines = [summary, "", head[0].replace("| Head |", "| Head | Kind |"), "|---" + head[1]]
    full_lines += [row(e).replace(f"| {e['head']} |", f"| {e['head']} | {e['kind']} |", 1) for e in full]
    return clean, clean_lines, full_lines


def rule_list_scores(stats, fired, train, cfg):
    """Scores of the readable rules alone (`_is_clean`): per row and ego head, the largest train precision
    of the clean rules of the head whose body holds, else the head's train base rate. Every score is a
    rule of rules.md (or the base rate), so the metrics of this predictor are those of the rule list."""
    risk, tgt = train["at_risk"], train["targets"] & train["at_risk"]
    base = tgt.sum(dim=0).float() / risk.sum(dim=0).clamp(min=1).float()
    scores = base[None].repeat(len(fired), 1)
    for e in stats:
        if _is_clean(e, cfg):
            h, f = e["head_index"], fired[:, e["rule"]]
            scores[f, h] = torch.maximum(scores[f, h], torch.tensor(e["train_precision"]))
    return scores


@torch.no_grad()
def pruned_scores(model, stats, kinds, rows, device):
    """Test scores of the rule net with the head weights of the (rule, head) pairs of the given kinds set
    to 0: if the metrics hold, those pairs were noise."""
    saved = model.head_weight.data.clone()
    for e in stats:
        if e["kind"] in kinds:
            model.head_weight.data[e["head_index"], e["rule"]] = 0.0
    try:
        return model_scores(model, rows, device)
    finally:
        model.head_weight.data.copy_(saved)


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
RULE_LIST = "Rule list (readable rules only)"


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
    stats, fired = rule_stats(rule_net, rows["train"], test, heads, names, cfg)
    clean, clean_lines, full_lines = rule_report(stats, cfg)
    scores = dict(base_scores)
    scores[MLP] = model_scores(mlp, test, device)
    scores[RULE_NET] = model_scores(rule_net, test, device)
    scores[RULE_LIST] = rule_list_scores(stats, fired["test"], rows["train"], cfg)
    ev = lambda s: evaluate(s, test, heads, label_heads, names, resamples)
    results = {name: ev(s) for name, s in scores.items()}
    pb = lambda a, b: paired_bootstrap([scores[a]], scores[b], test, heads, label_heads, names, resamples)
    paired = {f"{RULE_NET} − {MARKOV2}": pb(RULE_NET, MARKOV2), f"{RULE_NET} − {MARKOV1}": pb(RULE_NET, MARKOV1),
              f"{RULE_NET} − {MLP}": pb(RULE_NET, MLP), f"{MLP} − {MARKOV2}": pb(MLP, MARKOV2),
              f"{RULE_LIST} − {MARKOV2}": pb(RULE_LIST, MARKOV2), f"{RULE_LIST} − {RULE_NET}": pb(RULE_LIST, RULE_NET)}

    pruned = {"all rules": results[RULE_NET],
              "without below-base": ev(pruned_scores(rule_net, stats, {"below"}, test, device)),
              "without below-base and always-on": ev(pruned_scores(rule_net, stats, {"below", "always"}, test, device))}
    kinds = Counter(e["kind"] for e in stats)
    rtpt_utils.step(rtpt, f"seed {seed} eval")

    md = ["## 1. Models (test change points)\n"] + _model_table(results)
    md.append(f"\nRows: {results[RULE_NET]['rows']}; label rows (the next step changes the action label): "
              f"{results[RULE_NET]['label_rows']}.\n")
    md += ["## 2. Paired bootstrap (same test videos resampled for both sides)\n",
           f"Difference of the first minus the second method, 95% CI over {len(resamples)} resamples of the test "
           "videos, two-sided bootstrap p.\n"] + _paired_lines(paired) + [""]
    md += ["## 3. AP per head (test, >= 5 onsets)\n", "| Head | Onsets | " + " | ".join(results) + " |",
           "|---|---|" + "---|" * len(results)]
    for h in sorted(results[RULE_NET]["per_head"], key=lambda h: -results[RULE_NET]["per_head"][h]["onsets"]):
        cells = [_fmt(results[n]["per_head"].get(h, {}).get("ap")) for n in results]
        md.append(f"| {h} | {results[RULE_NET]['per_head'][h]['onsets']} | " + " | ".join(cells) + " |")
    md += ["", "## 4. Pruning check: the rule net without the noise rules\n",
           "Head weights of the (rule, head) pairs of a kind set to 0 (`rule_stats`): if the metrics hold, those "
           f"pairs were noise. Pairs: {kinds['rule']} above base, {kinds['below']} below base, "
           f"{kinds['always']} always on.\n",
           "| Rule net | Top-1 | Label top-1 | mAP | Micro AP |", "|---|---|---|---|---|"]
    md += [f"| {k} | {_fmt(r['top1'])} | {_fmt(r['label_top1'])} | {_fmt(r['map'])} | {_fmt(r['micro_ap'])} |"
           for k, r in pruned.items()]
    md += ["", f"## 5. Rules of the rule net (above the base rate, train support >= {cfg['min_rule_support']}, "
           f">= {cfg['min_rule_hits']} train hits; the rules of the rule list)\n"]
    md += clean_lines + ["", "All pairs with their kind: rules_all.md."]
    (out_dir / "rules.md").write_text("\n".join(clean_lines) + "\n")
    (out_dir / "rules_all.md").write_text("\n".join(full_lines) + "\n")
    (out_dir / "results.json").write_text(json.dumps({"args": vars(args), "seed": seed, "results": results,
                                                      "paired": paired, "pruned": pruned, "rules": stats},
                                                     indent=1, default=str))
    return {"md": md, "results": results,
            "scores": {RULE_NET: scores[RULE_NET], MLP: scores[MLP], RULE_LIST: scores[RULE_LIST]},
            "clean": clean}


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
              f"{RULE_LIST} (seed mean) − {MARKOV2}": pooled(RULE_LIST, MARKOV2),
              f"{RULE_LIST} (seed mean) − {MARKOV1}": pooled(RULE_LIST, MARKOV1)}
    md += ["", "Paired bootstrap of the seed-averaged metric (the same video resamples for every seed and the "
           "baseline):\n"] + _paired_lines(paired)
    seen = defaultdict(list)
    for s, r in zip(seeds, per_seed):
        for e in r["clean"]:
            seen[(e["head"], e["body"])].append(e)
    stable = sorted(((k, v) for k, v in seen.items() if len(v) >= 2), key=lambda kv: (-len(kv[1]), kv[0]))
    md += ["", f"### Rules learned in >= 2 of {len(seeds)} seeds ({len(stable)} of {len(seen)} distinct clean rules)\n",
           "| Head | Rule | Seeds | Train prec. | Lift | Test fires | Test prec. |", "|---|---|---|---|---|---|---|"]
    for (h, b), es in stable:
        e = es[0]          # same body and head: the same fires and precisions in every seed
        md.append(f"| {h} | {b} | {len(es)} | {_fmt(e['train_precision'])} | {e['lift']:.2f} | {e['test_fires']} | "
                  f"{_fmt(e['test_precision'])} |")
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
