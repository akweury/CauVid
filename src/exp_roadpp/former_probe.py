"""Which inputs does the AtomFormer use where it beats the rules? (2026-11-27)

For the heads where the Transformer leads the rule model most (Jun, Xing, IncomLane, av turns), on eval videos:
  1. Attribution, per input literal of the row's own token set (the OnsetMLP / rule literals "X(scope)@j"): the
     gradient of the logit of P(onset within the head's window) times the input, averaged over the head's onset
     rows (hits) and its other at-risk rows. A present atom X contributes grad * 1 ("X"); an absent known atom
     contributes -grad (first-order effect of its absence, "!X"). The rows of one frame share the scene, so in the
     factor / joint modes an input's gradient also holds its (small) effect on the other agents' rows.
  2. Per literal: hit rows / at-risk rows where it holds, precision (P(hit | literal)) and lift over the base rate.
  3. Pairs of the top literals: the precision, lift and hit coverage of "A & B", and whether some kept rule of the
     head (the rule-lab selection, the fit --probe_mode, |w| >= 0.05) contains both, one or none.
  4. Per head: the share of the top literals' attribution mass the kept rules use at all.

    ./run_roadpp_docker.sh --gpu 5 --former_probe -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --lab_former factor --num_epochs 100
Writes <output_dir>/rule_lab/former_probe_<attention>.json and .md.
"""
import glob
import json
import os
import re

import torch
from tqdm import tqdm

from src.exp_roadpp import config, rule_lab, step05, utils_data, utils_pipe
from src.exp_roadpp.baselines import atom_former

DEFAULT_HEADS = ("Car:Jun,MedVeh:Jun,LarVeh:Jun,Bus:Jun,Mobike:Jun,Cyc:Jun,Car:XingFmLft,Mobike:XingFmLft,"
                 "MedVeh:IncomLane,Bus:IncomLane,Ped:LftPav,av:AV-TurRht,av:AV-Brake,Car:Brake")


def literal_names(model, names):
    """Names of the packed inputs m = (scope * J + j) * A + a (the rule literal names without "!")."""
    scopes = ["own", "other", *model.roles]
    J, A = model.num_memory_slots, model.num_atoms
    return [f"{names[a]}({scopes[s]})@{j + 1}" for s in range(len(scopes)) for j in range(J) for a in range(A)]


def kept_rules(pattern, mode):
    """{head: [(set of literals, weight)]} of the rule-lab result jsons (selected variant, kept |w| >= 0.05)."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        sel = d.get("selected") or {}
        for h in d["per_head"]:
            v = sel.get(h) or next(iter(d["rules"]))
            out[h] = [(set(r["literals"]), r["soft"][mode]["weight"]) for r in d["rules"][v]
                      if r["head"] == h and mode in r.get("soft", {}) and abs(r["soft"][mode]["weight"]) >= 0.05]
    return out


def _rule_literals_match(lit, rule_lits):
    """lit "X(own)@1" / "!X(own)@1" in a rule's literals, also via a class group ("veh:Brake(ahead)@1")."""
    if lit in rule_lits:
        return True
    m = re.match(r"^(!?)([^:]+):(.+)$", lit)
    return bool(m) and f"{m.group(1)}veh:{m.group(3)}" in rule_lits


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
    parser.add_argument("--probe_top", type=int, default=20, help="top literals per head (and pairs among them)")
    parser.add_argument("--probe_rules", type=str, default="*_vt_p*.json", help="rule-lab result jsons (rule_lab dir)")
    parser.add_argument("--probe_mode", type=str, default="mdls_0.1", help="the rules' soft fit")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    for key in ("dim", "layers", "heads", "dropout", "lr"):
        cfg[f"former_{key}"] = getattr(args, f"former_{key}")
    cfg["former_attention"] = args.lab_former
    device = cfg["device"]
    training_data = step05.load_training_data(cfg)
    former = atom_former.load_former(training_data, cfg)
    if former is None:
        raise FileNotFoundError(f"no AtomFormer {atom_former.former_files(cfg)[0]}: train it with rule_lab --lab_former")
    for p in former.parameters():
        p.requires_grad_(False)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    lit_names = literal_names(former, names)
    M = len(lit_names)
    head_ids = {n: i for i, n in enumerate(names)}
    heads = [h for h in args.probe_heads.split(",") if h in head_ids]
    hidx = torch.tensor([head_ids[h] for h in heads], device=device)
    H = len(heads)
    S = former.slot_size
    w = int(cfg.get("clause_slots", 1))
    k_win = min(range(len(former.windows)), key=lambda i: abs(former.windows[i] - w * S))
    possible = step05.class_possible(training_data, vocab, device)
    val_ids = list(training_data["val_videos"])
    videos = (val_ids[1::2] + rule_lab.add_test_videos(training_data, cfg))[:args.probe_videos]
    print(f"[Former probe] {args.lab_former}, {H} heads, {len(videos)} videos, window {former.windows[k_win]} frames")

    z = lambda: torch.zeros((H, M), device=device)
    att_pos, att_neg, attn_pos, attn_neg = z(), z(), z(), z()            # X: grad * x; !X: -grad * absent
    cnt_hit, cnt_all, cntn_hit, cntn_all = z(), z(), z(), z()            # X held / !X held (known, absent)
    n_hit, n_all = torch.zeros(H, device=device), torch.zeros(H, device=device)
    row_store = {h: [] for h in range(H)}                                # pass 2: (x, absent, y) of sampled rows

    def head_rows(rows):
        can = possible(video_id)
        keep = rows["at_risk"][:, hidx] & can[rows["rows"][:, 1]][:, hidx] & rows["known"][:, w - 1][:, hidx]
        y = keep & (rows["first"][:, hidx] < w * S)
        return keep, y

    for video_id in tqdm(videos, desc="Attribution"):
        for frames in training_data["frames"][video_id].split(128):
            rows = step05._video_rows(former, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            keep, y = head_rows(rows)
            if not keep.any():
                continue
            scene = rows["_scene"].float().requires_grad_(True)
            h = former.forward_scene(scene, rows["_present"], rows["_partner"])[rows["fidx"], rows["slot"].clamp(min=0)]
            p = former.window_probs(h)[:, k_win][:, hidx].clamp(1e-6, 1 - 1e-6)            # (B, H)
            logit = torch.logit(p)
            x = rows["inputs"][:, :M]
            known = rows["inputs"][:, M:2 * M].bool()
            absent = (known & (x < 0.5)).float()
            for i in range(H):
                if not keep[:, i].any():
                    continue
                g, = torch.autograd.grad(logit[keep[:, i], i].sum(), scene, retain_graph=i < H - 1)
                g = g[rows["fidx"], rows["slot"].clamp(min=0), :M]                          # (B, M) own token set
                yi, ki = y[:, i], keep[:, i] & ~y[:, i]
                att_pos[i] += (g * x)[yi].sum(0); att_neg[i] += (g * x)[ki].sum(0)
                attn_pos[i] += (-g * absent)[yi].sum(0); attn_neg[i] += (-g * absent)[ki].sum(0)
                cnt_hit[i] += x[yi].sum(0); cnt_all[i] += x[keep[:, i]].sum(0)
                cntn_hit[i] += absent[yi].sum(0); cntn_all[i] += absent[keep[:, i]].sum(0)
                n_hit[i] += yi.sum(); n_all[i] += keep[:, i].sum()
                # rows for the pair statistics: every hit row, the other rows sampled (<= 20 per frame chunk)
                neg = torch.nonzero(ki).flatten()
                neg = neg[torch.randperm(len(neg), device=device)[:20]]
                sel = torch.cat([torch.nonzero(yi).flatten(), neg])
                wgt = torch.cat([torch.ones(int(yi.sum()), device=device),
                                 torch.full((len(neg),), float(ki.sum()) / max(len(neg), 1), device=device)])
                row_store[i].append((x[sel].bool().cpu(), absent[sel].bool().cpu(), yi[sel].cpu(), wgt.cpu()))

    rules = kept_rules(str(args.output_dir / "rule_lab" / args.probe_rules), args.probe_mode)
    out, md = {"attention": args.lab_former, "videos": len(videos), "heads": {}}, []
    for i, head in enumerate(heads):
        base = float(n_hit[i] / n_all[i].clamp(min=1))
        # literal table: X and !X side by side
        att = torch.cat([att_pos[i] / n_hit[i].clamp(min=1), attn_pos[i] / n_hit[i].clamp(min=1)])     # (2M,)
        att_n = torch.cat([att_neg[i], attn_neg[i]]) / (n_all[i] - n_hit[i]).clamp(min=1)
        hit = torch.cat([cnt_hit[i], cntn_hit[i]]); alln = torch.cat([cnt_all[i], cntn_all[i]])
        lits = lit_names + ["!" + n for n in lit_names]
        top = torch.argsort(att, descending=True)[:args.probe_top].tolist()
        hr = rules.get(head, [])
        used = {l for ls, wt in hr if wt > 0 for l in ls}
        rows_t = []
        for m in top:
            prec = float(hit[m] / alln[m].clamp(min=1))
            rows_t.append({"literal": lits[m], "attr_hit": float(att[m]), "attr_other": float(att_n[m]),
                           "hit_cover": float(hit[m] / n_hit[i].clamp(min=1)), "precision": prec,
                           "lift": prec / max(base, 1e-9), "in_rules": _rule_literals_match(lits[m], used)})
        # pairs among the top literals
        X = torch.cat([torch.cat([a, b], dim=1) for a, b, _, _ in row_store[i]]) if row_store[i] else None
        pairs = []
        if X is not None:
            Y = torch.cat([y for _, _, y, _ in row_store[i]]).float()
            W = torch.cat([wt for _, _, _, wt in row_store[i]])
            T = X[:, top].float()                                                        # (R, K)
            hits = (T * Y[:, None]).T @ T                                                # (K, K) hit rows with both
            alls = (T * W[:, None]).T @ T                                                # weighted at-risk rows
            for a in range(len(top)):
                for b in range(a + 1, len(top)):
                    if hits[a, b] < 5:
                        continue
                    prec = float(hits[a, b] / alls[a, b].clamp(min=1e-9))
                    both = any(_rule_literals_match(lits[top[a]], ls) and _rule_literals_match(lits[top[b]], ls)
                               for ls, wt in hr if wt > 0)
                    pairs.append({"pair": [lits[top[a]], lits[top[b]]], "hit_cover": float(hits[a, b] / Y.sum()),
                                  "precision": prec, "lift": prec / max(base, 1e-9), "in_one_rule": both})
            pairs.sort(key=lambda r: -r["precision"] * r["hit_cover"] ** 0.5)
        mass = sum(max(r["attr_hit"], 0) for r in rows_t)
        mass_used = sum(max(r["attr_hit"], 0) for r in rows_t if r["in_rules"])
        out["heads"][head] = {"hit_rows": int(n_hit[i]), "at_risk_rows": int(n_all[i]), "base_rate": base,
                              "kept_rules": len(hr), "top_mass_in_rules": mass_used / max(mass, 1e-9),
                              "literals": rows_t, "pairs": pairs[:25]}
        md += [f"## {head}  (hit rows {int(n_hit[i])}, base rate {base:.3f}, kept rules {len(hr)}, "
               f"top-{args.probe_top} attribution in rules {100 * mass_used / max(mass, 1e-9):.0f}%)", "",
               "| literal | attr hit | attr other | hit cover | precision | lift | in rules |", "|---|---|---|---|---|---|---|"]
        md += [f"| `{r['literal']}` | {r['attr_hit']:+.2f} | {r['attr_other']:+.2f} | {r['hit_cover']:.2f} | "
               f"{r['precision']:.3f} | {r['lift']:.1f} | {'yes' if r['in_rules'] else '**no**'} |" for r in rows_t]
        md += ["", "| pair | hit cover | precision | lift | in one rule |", "|---|---|---|---|---|"]
        md += [f"| `{r['pair'][0]}` & `{r['pair'][1]}` | {r['hit_cover']:.2f} | {r['precision']:.3f} | "
               f"{r['lift']:.1f} | {'yes' if r['in_one_rule'] else '**no**'} |" for r in pairs[:12]]
        md.append("")
        print(f"[Former probe] {head}: base {base:.3f}, top attribution in rules "
              f"{100 * mass_used / max(mass, 1e-9):.0f}%, top: " + ", ".join(r["literal"] for r in rows_t[:5]))
    path = args.output_dir / "rule_lab" / f"former_probe_{args.lab_former}"
    utils_data.save_json(out, f"{path}.json")
    open(f"{path}.md", "w").write("\n".join(md) + "\n")
    print(f"[Former probe] {path}.json / .md")


if __name__ == "__main__":
    main()
