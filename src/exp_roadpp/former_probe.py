"""What does the AtomFormer catch that the rules miss? (2026-11-27, v2)

v1 ranked literals by gradient x input: the top was "!X" of atoms that are (almost) never on for the head's class
(e.g. !Ped:Wait2X(own) for LarVeh:Jun), the same on hit and other rows, so uninformative. v2 compares the two
models' hits directly, on eval videos, per head:
  1. Scores of the AtomFormer (P(onset within the head's window)) and of the rule model (the rule-lab .rules.pt
     files of the run, `rule_paper.RuleScorer`) on every at-risk row of the head.
  2. Each model's threshold: as many rows flagged as there are hit rows (precision = recall). The hit rows split
     into both / former only / rules only / neither.
  3. Per literal (X: the atom held in the slot; !X: the slot known and the atom absent; the rule literals without
     class groups and derived literals): its share among the former-only hits, among the hits found by both, among
     the non-hit rows (sampled, weighted), its precision P(hit | literal) and its enrichment = share(former-only) /
     share(non-hit). The top literals: share(former-only) >= --probe_min_cover, by share x log2(enrichment).
  4. Pairs of the top literals: the same numbers for "A & B" and whether a kept rule of the head (the rule-lab
     selection, the fit --probe_mode, w > 0) contains both.

    ./run_roadpp_docker.sh --gpu 3 --former_probe -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --num_epochs 100 --lab_former factor
Writes <output_dir>/rule_lab/former_probe_<attention>.json and .md.
"""
import glob
import json
import math
import re

import torch
from tqdm import tqdm

from src.exp_roadpp import config, rule_lab, rule_paper, step05, utils_data, utils_pipe
from src.exp_roadpp.baselines import atom_former

DEFAULT_HEADS = ("Car:Jun,MedVeh:Jun,LarVeh:Jun,Bus:Jun,Mobike:Jun,Cyc:Jun,Car:XingFmLft,Mobike:XingFmLft,"
                 "MedVeh:IncomLane,Bus:IncomLane,Ped:LftPav,av:AV-TurRht,av:AV-Brake,Car:Brake")
GROUPS = ("both", "former_only", "rules_only", "neither")


def literal_names(model, names):
    """Names of the packed inputs m = (scope * J + j) * A + a (the rule literal names without "!")."""
    scopes = ["own", "other", *model.roles]
    J, A = model.num_memory_slots, model.num_atoms
    return [f"{names[a]}({scopes[s]})@{j + 1}" for s in range(len(scopes)) for j in range(J) for a in range(A)]


def kept_rules(pattern, mode):
    """{head: [set of literals]} of the positive kept rules (selected variant, w >= 0.05) of the result jsons."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        sel = d.get("selected") or {}
        for h in d["per_head"]:
            v = sel.get(h) or next(iter(d["rules"]))
            out[h] = [set(r["literals"]) for r in d["rules"][v]
                      if r["head"] == h and mode in r.get("soft", {}) and r["soft"][mode]["weight"] >= 0.05]
    return out


def in_rule(lit, rule_lits):
    """lit "X(own)@1" / "!X(own)@1" among a rule's literals, also via a class group ("veh:Brake(ahead)@1")."""
    if lit in rule_lits:
        return True
    m = re.match(r"^(!?)([^:]+):(.+)$", lit)
    return bool(m) and f"{m.group(1)}veh:{m.group(3)}" in rule_lits


def threshold(scores, k):
    """The score of the k-th highest row (k >= 1): rows >= it are flagged."""
    k = max(1, min(int(k), len(scores)))
    return float(torch.topk(scores, k).values[-1])


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--lab_former", type=str, default="factor", choices=atom_former.ATTENTIONS)
    parser.add_argument("--former_dim", type=int, default=128)
    parser.add_argument("--former_layers", type=int, default=2)
    parser.add_argument("--former_heads", type=int, default=4)
    parser.add_argument("--former_dropout", type=float, default=0.1)
    parser.add_argument("--former_lr", type=float, default=5e-4)
    parser.add_argument("--probe_heads", type=str, default=DEFAULT_HEADS)
    parser.add_argument("--probe_videos", type=int, default=600, help="eval videos (odd val, then test) at most")
    parser.add_argument("--probe_top", type=int, default=20, help="top literals per head")
    parser.add_argument("--probe_pairs", type=int, default=15, help="pairs among this many top literals")
    parser.add_argument("--probe_min_cover", type=float, default=0.15,
                        help="a top literal holds in at least this share of the former-only hits")
    parser.add_argument("--probe_neg", type=int, default=16, help="non-hit rows kept per head and frame chunk")
    parser.add_argument("--probe_rules", type=str, default="*_vt_p*", help="rule-lab result stem (rule_lab dir)")
    parser.add_argument("--probe_mode", type=str, default="mdls_0.1", help="the rules' soft fit")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    for key in ("dim", "layers", "heads", "dropout", "lr"):
        cfg[f"former_{key}"] = getattr(args, f"former_{key}")
    cfg["former_attention"] = args.lab_former
    device = cfg["device"]
    lab_dir = args.output_dir / "rule_lab"
    training_data = step05.load_training_data(cfg)
    former = atom_former.load_former(training_data, cfg)
    if former is None:
        raise FileNotFoundError(f"no AtomFormer {atom_former.former_files(cfg)[0]}: train it with rule_lab --lab_former")
    rule_net = step05.build_rule_net(training_data, cfg, 1).to(device)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    derived = rule_lab.Derived(rule_net, names)
    rule_files = sorted(glob.glob(str(lab_dir / f"{args.probe_rules}.rules.pt")))
    if not rule_files:
        raise FileNotFoundError(f"no rule files {lab_dir}/{args.probe_rules}.rules.pt")
    scorer = rule_paper.RuleScorer(rule_files, rule_net, derived, 1, device)
    lit_names = literal_names(former, names)
    lits = lit_names + ["!" + n for n in lit_names]
    M = len(lit_names)
    if M != rule_net.num_inputs:
        raise ValueError(f"packed inputs {M} != rule net inputs {rule_net.num_inputs}")
    head_ids = {n: i for i, n in enumerate(names)}
    heads = [h for h in args.probe_heads.split(",") if h in head_ids]
    hidx = torch.tensor([head_ids[h] for h in heads], device=device)
    H = len(heads)
    S, w = former.slot_size, int(cfg.get("clause_slots", 1))
    k_win = min(range(len(former.windows)), key=lambda i: abs(former.windows[i] - w * S))
    possible = step05.class_possible(training_data, vocab, device)
    val_ids = list(training_data["val_videos"])
    videos = (val_ids[1::2] + rule_lab.add_test_videos(training_data, cfg))[:args.probe_videos]
    print(f"[Former probe] {args.lab_former} vs rules ({len(rule_files)} files), {H} heads, {len(videos)} videos")

    scores = {i: [] for i in range(H)}       # all at-risk rows: (pF, pR, y)
    stored = {i: [] for i in range(H)}       # hit rows + sampled non-hits: (literal bits, pF, pR, y, weight)
    gen = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for video_id in tqdm(videos, desc="Scoring"):
            can = possible(video_id)
            for frames in training_data["frames"][video_id].split(128):
                rows = step05._video_rows(former, training_data, video_id, frames, device)
                if len(rows["inputs"]) == 0:
                    continue
                keep = (rows["at_risk"][:, hidx] & can[rows["rows"][:, 1]][:, hidx]
                        & rows["known"][:, w - 1][:, hidx])
                if not keep.any():
                    continue
                y = keep & (rows["first"][:, hidx] < w * S)
                pF = former.window_probs(former.forward_rows(rows))[:, k_win][:, hidx]
                pR = scorer(rows["inputs"])[:, 0][:, hidx]
                x = rows["inputs"][:, :M] > 0.5
                bits = torch.cat([x, rows["inputs"][:, M:2 * M].bool() & ~x], dim=1)        # [X | !X]
                for i in range(H):
                    ki = keep[:, i]
                    if not ki.any():
                        continue
                    scores[i].append(torch.stack([pF[ki, i], pR[ki, i], y[ki, i].float()], 1).cpu())
                    hit = torch.nonzero(y[:, i]).flatten()
                    neg = torch.nonzero(ki & ~y[:, i]).flatten().cpu()
                    neg = neg[torch.randperm(len(neg), generator=gen)[:args.probe_neg]].to(device)
                    wn = float((ki & ~y[:, i]).sum()) / max(len(neg), 1)
                    sel = torch.cat([hit, neg])
                    wt = torch.cat([torch.ones(len(hit), device=device), torch.full((len(neg),), wn, device=device)])
                    stored[i].append((bits[sel].cpu(), pF[sel, i].cpu(), pR[sel, i].cpu(), y[sel, i].cpu(), wt.cpu()))

    rules = kept_rules(str(lab_dir / f"{args.probe_rules}.json"), args.probe_mode)
    out, md = {"attention": args.lab_former, "videos": len(videos), "heads": {}}, []
    for i, head in enumerate(heads):
        if not scores[i]:
            continue
        sc = torch.cat(scores[i])
        n_hit = int(sc[:, 2].sum())
        if n_hit < 20:
            continue
        tF, tR = threshold(sc[:, 0], n_hit), threshold(sc[:, 1], n_hit)
        B = torch.cat([s[0] for s in stored[i]])
        pF = torch.cat([s[1] for s in stored[i]]); pR = torch.cat([s[2] for s in stored[i]])
        Y = torch.cat([s[3] for s in stored[i]]); W = torch.cat([s[4] for s in stored[i]])
        fF, fR = pF >= tF, pR >= tR
        grp = {"both": Y & fF & fR, "former_only": Y & fF & ~fR, "rules_only": Y & ~fF & fR, "neither": Y & ~fF & ~fR}
        counts = {g: int(m.sum()) for g, m in grp.items()}
        Bf = B.float()
        neg = ~Y
        share = lambda m: (Bf[m].sum(0) / max(int(m.sum()), 1))
        s_fo, s_both, s_hit = share(grp["former_only"]), share(grp["both"]), share(Y)
        s_neg = (Bf[neg] * W[neg, None]).sum(0) / W[neg].sum().clamp(min=1e-9)
        hits_with = Bf[Y].sum(0)
        prec = hits_with / (hits_with + (Bf[neg] * W[neg, None]).sum(0)).clamp(min=1e-9)
        enrich = s_fo / s_neg.clamp(min=1e-4)
        base = n_hit / len(sc)
        ok = (s_fo >= args.probe_min_cover) & (enrich > 1)
        score = torch.where(ok, s_fo * torch.log2(enrich.clamp(min=1)), torch.tensor(-1.0))
        top = [m for m in torch.argsort(score, descending=True)[:args.probe_top].tolist() if score[m] > 0]
        hr = rules.get(head, [])
        used = set().union(*hr) if hr else set()
        lit_rows = [{"literal": lits[m], "share_former_only": float(s_fo[m]), "share_both": float(s_both[m]),
                     "share_nonhit": float(s_neg[m]), "enrichment": float(enrich[m]), "precision": float(prec[m]),
                     "lift": float(prec[m]) / max(base, 1e-9), "in_rules": in_rule(lits[m], used)} for m in top]
        pairs = []
        P = top[:args.probe_pairs]
        if len(P) >= 2:
            T = Bf[:, P]
            fo = grp["former_only"].float()
            both_fo = (T * fo[:, None]).T @ T / max(counts["former_only"], 1)
            both_neg = (T * (W * neg.float())[:, None]).T @ T / W[neg].sum().clamp(min=1e-9)
            both_hit = (T * Y.float()[:, None]).T @ T
            both_all = both_hit + (T * (W * neg.float())[:, None]).T @ T
            for a in range(len(P)):
                for b in range(a + 1, len(P)):
                    e = float(both_fo[a, b] / both_neg[a, b].clamp(min=1e-4))
                    if both_fo[a, b] < args.probe_min_cover / 2 or e <= max(enrich[P[a]], enrich[P[b]]):
                        continue                                 # the pair must beat both of its literals
                    pr = float(both_hit[a, b] / both_all[a, b].clamp(min=1e-9))
                    pairs.append({"pair": [lits[P[a]], lits[P[b]]], "share_former_only": float(both_fo[a, b]),
                                  "enrichment": e, "precision": pr, "lift": pr / max(base, 1e-9),
                                  "in_one_rule": any(in_rule(lits[P[a]], r) and in_rule(lits[P[b]], r) for r in hr)})
            pairs.sort(key=lambda r: -r["share_former_only"] * math.log2(max(r["enrichment"], 1)))
        out["heads"][head] = {"hit_rows": n_hit, "at_risk_rows": len(sc), "base_rate": base, "groups": counts,
                              "kept_rules": len(hr), "literals": lit_rows, "pairs": pairs[:20]}
        md += [f"## {head}", "",
               f"hit rows {n_hit}, base rate {base:.3f}, kept rules {len(hr)}; hits found by both {counts['both']}, "
               f"**former only {counts['former_only']}**, rules only {counts['rules_only']}, neither {counts['neither']} "
               f"(stored rows; thresholds: as many flags as hits)", "",
               "| literal | share former-only | share both | share non-hit | enrichment | precision | lift | in rules |",
               "|---|---|---|---|---|---|---|---|"]
        md += [f"| `{r['literal']}` | {r['share_former_only']:.2f} | {r['share_both']:.2f} | {r['share_nonhit']:.3f} | "
               f"{r['enrichment']:.1f} | {r['precision']:.3f} | {r['lift']:.1f} | {'yes' if r['in_rules'] else '**no**'} |"
               for r in lit_rows]
        md += ["", "| pair (beats both literals) | share former-only | enrichment | precision | lift | in one rule |",
               "|---|---|---|---|---|---|"]
        md += [f"| `{r['pair'][0]}` & `{r['pair'][1]}` | {r['share_former_only']:.2f} | {r['enrichment']:.1f} | "
               f"{r['precision']:.3f} | {r['lift']:.1f} | {'yes' if r['in_one_rule'] else '**no**'} |" for r in pairs[:12]]
        md.append("")
        print(f"[Former probe] {head}: hits {n_hit}: both {counts['both']}, former only {counts['former_only']}, "
              f"rules only {counts['rules_only']}, neither {counts['neither']}; top: "
              + ", ".join(f"{r['literal']} (x{r['enrichment']:.1f})" for r in lit_rows[:4]))
    path = lab_dir / f"former_probe_{args.lab_former}"
    utils_data.save_json(out, f"{path}.json")
    open(f"{path}.md", "w").write("\n".join(md) + "\n")
    print(f"[Former probe] {path}.json / .md")


if __name__ == "__main__":
    main()
