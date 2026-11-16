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
AP per head, mAP; a bootstrap over the test videos for top-1.

Run (server; GPU optional, decodes the videos once for the ego flow cache if missing):
  ./run_roadpp_docker.sh --gpu 6 --seq -- --exp full_sn
Output: <exp>/seq/<tag>/results.md, results.json, rules.md, models.
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


def evaluate(scores, rows, heads, label_heads, names, rng):
    """top-1 / top-3, label top-1, AP per head, mAP and micro AP of (B, A) scores on the test rows."""
    s = scores.clone()
    risk = rows["at_risk"] & heads[None]
    tgt = rows["targets"] & risk
    s[~risk] = -1.0
    has = tgt.any(dim=1)
    top = s.topk(3, dim=1).indices
    hit1 = tgt.gather(1, top[:, :1]).any(dim=1) & has
    hit3 = tgt.gather(1, top).any(dim=1) & has
    lab = label_heads[None] & risk
    lab_rows = (tgt & label_heads[None]).any(dim=1)
    sl = torch.where(lab, scores, torch.full_like(scores, -1.0))
    lab_hit = tgt.gather(1, sl.argmax(dim=1, keepdim=True))[:, 0] & lab_rows
    v = rows["video"].numpy()
    vids = np.unique(v)
    boot = []
    for _ in range(200):
        pick = rng.choice(vids, len(vids))
        sel = np.concatenate([np.nonzero(v == p)[0] for p in pick])
        sel = sel[has.numpy()[sel]]
        if len(sel):
            boot.append(hit1.numpy()[sel].mean())
    per_head = {}
    for h in torch.nonzero(heads).flatten().tolist():
        r = risk[:, h]
        y = tgt[r, h].numpy()
        if y.sum() >= MIN_HEAD_EVENTS:
            per_head[names[h]] = {"onsets": int(y.sum()), "ap": _ap(y, scores[r, h].numpy())}
    aps = [d["ap"] for d in per_head.values() if d["ap"] is not None]
    yy, ss = tgt[risk].numpy(), scores[risk].numpy()
    return {"rows": int(has.sum()), "top1": float(hit1[has].float().mean()) if has.any() else None,
            "top1_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None,
            "top3": float(hit3[has].float().mean()) if has.any() else None,
            "label_rows": int(lab_rows.sum()),
            "label_top1": float(lab_hit[lab_rows].float().mean()) if lab_rows.any() else None,
            "map": float(np.mean(aps)) if aps else None, "micro_ap": _ap(yy, ss), "per_head": per_head}


# -------------- Rules --------------
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
def rule_report(model, train, test, heads, names, cfg):
    """Rules of the rule net per ego head with their precision (next step holds the head when every body
    holds) and support on train and test."""
    lines = []
    body = model.rule_mask() > 0.5
    stats = {}
    for name, rows in (("train", train), ("test", test)):
        b = body.float()
        fired = []
        for i in range(0, len(rows["targets"]), 2048):
            x, _k = model.split_inputs(rows["inputs"][i:i + 2048].to(cfg["device"]))
            fired.append(((x @ b.T >= b.sum(dim=1)) & (b.sum(dim=1) > 0)).cpu())   # (B, R)
        stats[name] = (torch.cat(fired).float() if fired else torch.zeros((0, len(b))), rows)
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
            entry = {"head": names[h], "weight": w, "body": text}
            for split, (fired, rows) in stats.items():
                f = fired[:, r].bool() & rows["at_risk"][:, h]
                entry[f"{split}_fires"] = int(f.sum())
                entry[f"{split}_precision"] = float(rows["targets"][f, h].float().mean()) if f.any() else None
            if entry["train_fires"] >= cfg["min_rule_support"]:
                out.append(entry)
    out.sort(key=lambda e: (e["head"], -(e["train_precision"] or 0)))
    lines.append("| Head | Rule (body ⇒ next step) | Weight | Train fires | Train prec. | Test fires | Test prec. |")
    lines.append("|---|---|---|---|---|---|---|")
    for e in out:
        fmt = lambda v: "–" if v is None else f"{v:.2f}"
        lines.append(f"| {e['head']} | {e['body']} | {e['weight']:.2f} | {e['train_fires']} | {fmt(e['train_precision'])} "
                     f"| {e['test_fires']} | {fmt(e['test_precision'])} |")
    return out, lines


# -------------- Main --------------
def ensure_dataset(exp, machine, device):
    """Run step_dataset for the exp (its exp_config: split, far-vehicle filter, split_from) if its atoms
    are missing, e.g. an exp whose pipeline has not run yet; it skips the files it already has."""
    pargs = utils_pipe.get_parser().parse_args(["--exp", exp, "--machine", machine, "--device", device])
    for key, value in utils_data.load_json(config.exp_config_path / f"{exp}.json").items():
        setattr(pargs, key, value)
    pargs = config.machine_setup(pargs)
    step_dataset.main(config.get_step_dataset_input(pargs))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", default="medium_sn")
    ap.add_argument("--machine", default="dgx-jing")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_steps", type=int, default=NUM_STEPS)
    ap.add_argument("--min_step", type=int, default=MIN_STEP)
    ap.add_argument("--num_rules", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ego_flow_workers", type=int, default=16)
    args, _unknown = ap.parse_known_args()
    setup = config.machine_setup(args)
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    torch.manual_seed(args.seed)
    rtpt = rtpt_utils.start(f"roadpp-{args.exp}-seq", 4)
    atoms_dir = setup.output_dir / "step_dataset_output" / "atoms"
    flow_dir = setup.output_dir.parent / "ego_flow_cache" / ego_flow.VERSION
    tag = f"s{args.num_steps}m{args.min_step}r{args.num_rules}_{ego_flow.tag().lstrip('_')}_seed{args.seed}"
    out_dir = setup.output_dir / "seq" / tag
    os.makedirs(out_dir, exist_ok=True)
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
    heads_d = heads.to(device)
    cfg = {"device": device, "seed": args.seed, "epochs": args.epochs, "warmup": 10, "patience": 20, "batch": 512,
           "l0_lambda": 1e-3, "pos_weight": 1.0, "min_rule_support": 10}

    print("[Seq] rule net")
    rule_net = TemporalRuleNet(A, 1, args.num_steps, args.num_rules, -11.0, "sigmoid", (1,), require_self=True,
                               ranges=True).to(device)
    rule_net = fit(rule_net, rows["train"], rows["val"], heads_d, cfg, True)
    rtpt_utils.step(rtpt, "rule net")
    print("[Seq] MLP")
    mlp = OnsetMLP(A, 1, args.num_steps, args.num_rules, 0.2, (1,), open_memory=True).to(device)
    mlp = fit(mlp, rows["train"], rows["val"], heads_d, cfg, False)
    rtpt_utils.step(rtpt, "mlp")
    torch.save(rule_net.state_dict(), out_dir / "rule_net.pt")
    torch.save(mlp.state_dict(), out_dir / "mlp.pt")

    test = rows["test"]
    scores = {"Prior (base rate)": markov_scores(steps["train"], steps["test"], A, 0),
              "Markov-1 (current state)": markov_scores(steps["train"], steps["test"], A, 1),
              "Markov-2 (last two states)": markov_scores(steps["train"], steps["test"], A, 2),
              "OnsetMLP": model_scores(mlp, test, device),
              "Rule net (ours)": model_scores(rule_net, test, device)}
    rng = np.random.default_rng(args.seed)
    results = {name: evaluate(s, test, heads, label_heads, names, rng) for name, s in scores.items()}
    rules, rule_lines = rule_report(rule_net, rows["train"], test, heads, names, cfg)
    rtpt_utils.step(rtpt, "eval")

    fmt = lambda v: "–" if v is None else f"{v:.3f}"
    md = [f"# Ego next-step prediction ({args.exp}, {tag})\n",
          "At each change point (the ego enters a new state: its action label, yaw level, speed level), "
          "which ego atoms switch on at the next step; no time window. Steps hold >= "
          f"{args.min_step} frames; memory: the last {args.num_steps} steps, then bins of steps.\n",
          "| Split | Videos | Change points | Median steps / video |", "|---|---|---|---|"]
    md += [f"| {sp} | {info[sp]['videos']} | {info[sp]['change_points']} | {info[sp]['median_steps']:.0f} |" for sp in SPLITS]
    md += ["", "## 1. Models (test change points)\n",
           "| Method | Top-1 [95% CI] | Top-3 | Label top-1 | mAP | Micro AP |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        ci = f" [{r['top1_ci'][0]:.3f}, {r['top1_ci'][1]:.3f}]" if r["top1_ci"] else ""
        md.append(f"| {name} | {fmt(r['top1'])}{ci} | {fmt(r['top3'])} | {fmt(r['label_top1'])} | "
                  f"{fmt(r['map'])} | {fmt(r['micro_ap'])} |")
    md.append(f"\nRows: {results['Rule net (ours)']['rows']}; label rows (the next step changes the action label): "
              f"{results['Rule net (ours)']['label_rows']}.\n")
    md += ["## 2. AP per head (test, >= 5 onsets)\n", "| Head | Onsets | " + " | ".join(results) + " |",
           "|---|---|" + "---|" * len(results)]
    for h in sorted(results["Rule net (ours)"]["per_head"], key=lambda h: -results["Rule net (ours)"]["per_head"][h]["onsets"]):
        cells = [fmt(results[n]["per_head"].get(h, {}).get("ap")) for n in results]
        md.append(f"| {h} | {results['Rule net (ours)']['per_head'][h]['onsets']} | " + " | ".join(cells) + " |")
    md += ["", f"## 3. Rules of the rule net (train support >= {cfg['min_rule_support']})\n"] + rule_lines
    (out_dir / "results.md").write_text("\n".join(md) + "\n")
    (out_dir / "rules.md").write_text("\n".join(rule_lines) + "\n")
    (out_dir / "results.json").write_text(json.dumps({"args": vars(args), "info": info, "results": results,
                                                      "rules": rules}, indent=1, default=str))
    print("\n".join(md))
    print(f"\n[Seq] {out_dir}")


if __name__ == "__main__":
    main()
