"""Mining sweep: how the mining settings change the mined rule list, without training the rule net.

Questions (2026-11-21):
  1. Is mine_min_lift 2 (a rule's precision >= 2x its root's) too strict? Compare 1.2 / 1.5 / 2.0.
  2. Heads with < mine_min_events onsets get no mined rules (only their root): how much do they weigh in the
     AP? Compare mine_min_events 20 / 10 / 5 and report the AP by head size.
  3. The trained net's free rules are a separate question (`--free_rules_per_head 0` run).

Rows: the `step05_mine.collect` sample (hits within clause_slots, weighted back to the full data). The train
rows mine; the val videos are split in two halves by video: "check" for the miner's val filter
(mine_val_lift = val_ratio x mine_min_lift), "eval" for the scores. The test split is not touched.

Scorer of a rule list (as step06's anchor baseline): a row's score for a head is the highest confidence of the
head's rules (mined rules and roots) that hold on it, else the head's train base rate. The confidence is the
train precision shrunk to the root's with k onsets (`step05_mine.shrunk_confidence`), for each k in --priors
(k0 = the raw precision); the rules are mined once per setting, the priors only change the scores. Reported per
setting: rules, literals per rule, heads with mined rules, the weighted AP per head on the eval half (macro
over heads, by head size in train onsets), and the micro AP (all heads' rows pooled), all heads and without
the small ones. "roots" is the list of the roots alone (no mined rules) for reference.

Run (server): ./run_roadpp_docker.sh --gpu 0 --mine_sweep -- --exp sumo_maps_v7 --machine dgx-jing
Writes <output_dir>/mine_sweep/mine_sweep_<tag>.json; the sampled rows are cached next to it.
"""
import math
import os
import re
from collections import defaultdict

import torch
from tqdm import tqdm

from src.exp_roadpp import config, step05, step05_mine, utils_data, utils_pipe

SIZE_GROUPS = (("<20", 0, 20), ("20-99", 20, 100), ("100-499", 100, 500), (">=500", 500, math.inf))


def weighted_ap(score, y, w):
    """Average precision of the rows ranked by score (ties ranked together), rows weighted by w; None
    without a hit."""
    if not bool(y.any()):
        return None
    order = torch.argsort(score, descending=True)
    s, y, w = score[order], y[order].double(), w[order].double()
    tp, fp = torch.cumsum(w * y, 0), torch.cumsum(w * (1 - y), 0)
    _, counts = torch.unique_consecutive(s, return_counts=True)
    ends = torch.cumsum(counts, 0) - 1
    tp_e, fp_e = tp[ends], fp[ends]
    precision = tp_e / (tp_e + fp_e).clamp(min=1e-12)
    gained = torch.diff(tp_e, prepend=tp_e.new_zeros(1))
    return float((precision * gained).sum() / tp_e[-1])


def subset(data, keep_video):
    """The rows of data (a `collect` sample) whose video index is in keep_video (a set)."""
    pairs = data["pairs"]
    sel = torch.tensor([int(v) in keep_video for v in pairs["video"].tolist()], dtype=torch.bool)
    return {"packed": data["packed"], "pairs": {k: v[sel] for k, v in pairs.items()}, "rate": data["rate"]}


def rule_score(rule, prior):
    """A rule's score: its train precision shrunk to its root's with prior onsets (0: the raw precision)."""
    if rule.get("root"):
        return float(rule["precision"])
    return step05_mine.shrunk_confidence(rule["precision"], rule["events"], rule["root_precision"], prior)


def score_rows(pool, rules, base, prior=0.0):
    """(n,) score of the rows of one head's pool: max confidence (`rule_score`) of the rules holding, else
    base."""
    score = torch.full((len(pool["y"]),), float(base), device=pool["y"].device)
    for rule in rules:
        holds = step05_mine._holds(pool, rule["literals"])
        score = torch.where(holds, torch.clamp(score, min=rule_score(rule, prior)), score)
    return score


def evaluate(model, seeds, heads, train_events, evaluation, rate, device, small_below=100, prior=0.0):
    """AP of the rule list seeds on the evaluation sample: per head, macro by size group, micro."""
    L = model.num_literals
    by_head = defaultdict(list)
    for sd in seeds:
        by_head[int(sd["head"])].append(sd)
    per_head, pooled = {}, []
    for head in heads:
        pool = step05_mine._pool_of(evaluation, head, L, device)
        if len(pool["y"]) == 0:
            continue
        score = score_rows(pool, by_head.get(head, []), rate[head], prior)
        ap = weighted_ap(score, pool["y"], pool["w"])
        if ap is not None:
            per_head[head] = ap
        pooled.append((score, pool["y"], pool["w"], train_events[head]))
    out = {"macro": sum(per_head.values()) / max(len(per_head), 1), "heads": len(per_head), "by_size": {}}
    for name, lo, hi in SIZE_GROUPS:
        aps = [ap for h, ap in per_head.items() if lo <= train_events[h] < hi]
        out["by_size"][name] = {"heads": len(aps), "macro": sum(aps) / len(aps) if aps else None}
    for name, keep in (("micro", lambda e: True), (f"micro_ge{small_below}", lambda e: e >= small_below)):
        parts = [p for p in pooled if keep(p[3])]
        out[name] = weighted_ap(torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]),
                                torch.cat([p[2] for p in parts])) if parts else None
    out["per_head"] = per_head
    return out


def load_rows(cfg, training_data, model, head_mask, possible, out_dir):
    """The `step05_mine.collect` samples (train, val), cached in out_dir. The row sample does not depend on the
    mining version or the rule settings (only on the caps, which stay in the tag): the cache name leaves the
    version out, so a new version reuses it."""
    tag = os.path.basename(str(cfg["mined_seed_file"])).replace("mined_seeds_", "").replace(".json", "")
    rows_file = out_dir / f"rows_{re.sub(r'_m[0-9]+(?=_|$)', '', tag)}{cfg.get('target_tag', '')}_s{cfg['seed']}.pt"
    if os.path.exists(rows_file) and step05.is_fresh(rows_file, cfg["training_data_file"]):
        return step05.load_data(rows_file)
    train = step05_mine.collect(model, training_data, training_data["train_videos"], head_mask, possible, cfg,
                                "Rows (train)")
    val = step05_mine.collect(model, training_data, training_data["val_videos"], head_mask, possible, cfg,
                              "Rows (val)")
    os.makedirs(out_dir, exist_ok=True)                 # a new exp has no mine_sweep folder yet
    torch.save((train, val), rows_file)
    return train, val


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--lifts", type=str, default="1.2,1.5,2.0", help="mine_min_lift values")
    parser.add_argument("--min_events_list", type=str, default="20,10,5", help="mine_min_events values")
    parser.add_argument("--priors", type=str, default="0,10,20,50",
                        help="rule confidence shrunk to the root's with this many onsets (0 = raw precision)")
    parser.add_argument("--val_ratio", type=float, default=0.75,
                        help="mine_val_lift = val_ratio x mine_min_lift (default 0.75: 2.0 -> 1.5, as the main run)")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    # The step 05 files (ego motion / grid specs, ...) of a run with --run_id, else of the sweep folder.
    args.run_dir = args.output_dir / "runs" / args.run_id if getattr(args, "run_id", None) \
        else args.output_dir / "mine_sweep"
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    out_dir = args.output_dir / "mine_sweep"
    os.makedirs(out_dir, exist_ok=True)

    training_data = step05.load_training_data(cfg)
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    head_mask = training_data["target_heads"].to(device)
    heads = torch.nonzero(head_mask).flatten().tolist()
    possible = step05.class_possible(training_data, vocab, device)

    tag = os.path.basename(str(cfg["mined_seed_file"])).replace("mined_seeds_", "").replace(".json", "")
    train, val = load_rows(cfg, training_data, model, head_mask, possible, out_dir)
    num_val = len(training_data["val_videos"])
    check = subset(val, set(range(0, num_val, 2)))
    evaluation = subset(val, set(range(1, num_val, 2)))

    # Train onsets per head (the size groups) and the train base rate (the score where no rule holds).
    L = model.num_literals
    train_events = {h: step05_mine._stats(p, torch.ones_like(p["y"]))["events"]
                    for h in heads for p in [step05_mine._pool_of(train, h, L, device)]}
    rate = train["rate"].tolist()

    settings = [("roots", None, None)] + [(f"lift{lift:g}_me{me}", lift, me)
                                          for me in [int(x) for x in args.min_events_list.split(",")]
                                          for lift in [float(x) for x in args.lifts.split(",")]]
    results, root_seeds = {}, None
    priors = [float(x) for x in args.priors.split(",")]
    for name, lift, me in settings:
        if name == "roots":
            continue
        run_cfg = {**cfg, "mine_min_lift": lift, "mine_val_lift": args.val_ratio * lift, "mine_min_events": me}
        seeds = []
        for head in tqdm(heads, desc=f"Mining {name}"):
            s, _report = step05_mine.mine_head(model, names, head, train, check, run_cfg, device)
            seeds += s
        if root_seeds is None:
            root_seeds = [sd for sd in seeds if sd.get("root")]
        mined = [sd for sd in seeds if not sd.get("root")]
        for prior in priors:
            res = evaluate(model, seeds, heads, train_events, evaluation, rate, device, prior=prior)
            res.update({"rules": len(mined), "heads_mined": len({sd["head"] for sd in mined}),
                        "literals_per_rule": sum(len(sd["literals"]) for sd in mined) / max(len(mined), 1),
                        "lift": lift, "val_lift": args.val_ratio * lift, "min_events": me, "prior": prior,
                        "rules_text": [sd["text"] for sd in mined]})
            results[f"{name}_k{prior:g}"] = res
    res = evaluate(model, root_seeds, heads, train_events, evaluation, rate, device)
    res.update({"rules": 0, "heads_mined": 0, "literals_per_rule": 0.0})
    results = {"roots": res, **results}

    groups = [g[0] for g in SIZE_GROUPS]
    print(f"\n[Sweep] eval = odd val videos ({len(range(1, num_val, 2))}); train onsets per head group: "
          + ", ".join(f"{g} {sum(1 for h in heads if lo <= train_events[h] < hi)} heads"
                      for g, lo, hi in SIZE_GROUPS))
    print(f"{'setting':22s} {'rules':>5s} {'heads':>5s} {'lits':>5s} {'micro':>6s} {'mic100':>6s} {'macro':>6s} "
          + " ".join(f"{g:>8s}" for g in groups))
    for name, r in results.items():
        print(f"{name:22s} {r['rules']:5d} {r['heads_mined']:5d} {r['literals_per_rule']:5.2f} {r['micro']:6.3f} "
              f"{r['micro_ge100']:6.3f} {r['macro']:6.3f} "
              + " ".join(f"{r['by_size'][g]['macro']:8.3f}" if r['by_size'][g]['macro'] is not None else f"{'-':>8s}"
                         for g in groups))
    path = out_dir / f"mine_sweep_{tag}_s{cfg['seed']}.json"
    utils_data.save_json({"settings": {"lifts": args.lifts, "min_events": args.min_events_list, "priors": args.priors,
                                       "val_ratio": args.val_ratio, "seed": cfg["seed"]},
                          "names": {h: names[h] for h in heads}, "train_events": train_events,
                          "results": {k: {**v, "per_head": {names[h]: ap for h, ap in v["per_head"].items()}}
                                      for k, v in results.items()}}, path)
    print(f"[Sweep] saved {path}")


if __name__ == "__main__":
    main()
