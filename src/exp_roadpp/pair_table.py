"""Length-1 and length-2 rules of one head by tables (2026-11-27, the user's search principle).

Principle (the user): a literal is added to a rule if the precision rises, by any amount (no x1.5 gain); the recall
falling is expected. No literal may be missed: every combination is tested. Only the length-1 rules are tested on
the rows; the length-2 rules follow from tables of the length-1 results:
  rows(a & b) = (F^T F)[a, b], hit rows(a & b) = (F_hit^T F_hit)[a, b]   (F: the rows x literals 0 / 1 matrix),
so every pair of literals is counted at once, from the same pass over the rows.

Rows: every at-risk row of the head (`step05_mine._head_rows`, no sample) of --pt_train_videos train videos and
--pt_val_videos val videos; a hit row has the onset within clause_slots slots; onsets: distinct (video, agent,
onset frame) among the hit rows. Literals: the rule net's literals the miner may use (`input_mask`, not the head
itself; as the main model's "_nn" variants: no negated "other" literals, the shape atoms in the newest slot only).
  length 1  every literal: rows, hit rows, precision, recall (hit rows / all hit rows), onsets; a candidate if it
            has >= --pt_min_onsets train onsets;
  length 2  every pair of candidates: the same counts from the tables; kept if its train precision is above both
            of its literals' and it has >= --pt_min_onsets train onsets; on val: whether it is above both there too.

    ./run_roadpp_docker.sh --gpu 3 --pair_table -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --pt_head Car:Jun --pt_train_videos 1000 --pt_val_videos 500
Writes <output_dir>/rule_lab/pairs_<head>.md / .json.
"""
import torch

from src.exp_roadpp import config, rule_lab, step05, step05_mine, utils_data, utils_pipe
from src.exp_roadpp.rule_direct import head_rows

MM = 256                # rows per bf16 product: counts <= 256 are exact in bf16, summed in float32


def tables(gen, model, cols, device, desc, keep_hits=True):
    """One pass over the rows: n1 / h1 (C,) rows / hit rows per literal, N2 / H2 (C, C) per pair, the hit rows'
    literal bits (on the CPU) and their event index, the total rows and hit rows."""
    C = len(cols)
    n1, h1 = torch.zeros(C, device=device), torch.zeros(C, device=device)
    N2, H2 = torch.zeros((C, C), device=device), torch.zeros((C, C), device=device)
    hits, events, ev_ids = [], [], {}
    total, total_hit = 0, 0
    with torch.no_grad():
        for vid, inp, y, rk, first in gen:
            F = model.literal_values(inp)[:, cols] > 0.5                          # (B, C)
            Fb = F.to(torch.bfloat16)
            n1 += F.sum(0)
            h1 += F[y].sum(0)
            for i in range(0, len(Fb), MM):
                f = Fb[i:i + MM]
                N2 += (f.T @ f).float()
            Fh = Fb[y]
            for i in range(0, len(Fh), MM):
                f = Fh[i:i + MM]
                H2 += (f.T @ f).float()
            total += len(F)
            total_hit += int(y.sum())
            if keep_hits and y.any():
                hits.append(F[y].cpu())
                for i in torch.nonzero(y).flatten().tolist():
                    events.append(ev_ids.setdefault((vid, int(rk[i, 1]), int(rk[i, 0]) + int(first[i])), len(ev_ids)))
    return {"n1": n1, "h1": h1, "N2": N2, "H2": H2, "total": total, "total_hit": total_hit,
            "hits": torch.cat(hits) if hits else torch.zeros((0, C), dtype=torch.bool),
            "events": torch.tensor(events, dtype=torch.long), "num_events": len(ev_ids)}


def onsets(t, device, pairs=True, batch=32):
    """Distinct onsets per literal (C,) and per pair (C, C): an onset counts if one of its hit rows holds them."""
    Fh, ev, E = t["hits"].to(device), t["events"].to(device), t["num_events"]
    C = Fh.shape[1]
    per = torch.zeros((E, C), device=device).index_add_(0, ev, Fh.float()) > 0
    o1 = per.sum(0).float()
    if not pairs:
        return o1, None
    o2 = torch.zeros((C, C), device=device)
    order = torch.argsort(ev)
    Fh, ev = Fh[order], ev[order]
    starts = torch.searchsorted(ev, torch.arange(E + 1, device=device)).tolist()
    for e0 in range(0, E, batch):
        e1 = min(E, e0 + batch)
        n = max(starts[e + 1] - starts[e] for e in range(e0, e1))
        X = torch.zeros((e1 - e0, n, C), dtype=torch.bfloat16, device=device)
        for k, e in enumerate(range(e0, e1)):
            X[k, :starts[e + 1] - starts[e]] = Fh[starts[e]:starts[e + 1]].to(torch.bfloat16)
        o2 += ((X.transpose(1, 2) @ X).float() > 0).sum(0).float()
    return o1, o2


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--pt_head", type=str, default="Car:Jun")
    parser.add_argument("--pt_train_videos", type=int, default=1000, help="train videos (the first; 0: all)")
    parser.add_argument("--pt_val_videos", type=int, default=500, help="val videos (the first; 0: all)")
    parser.add_argument("--pt_min_onsets", type=int, default=5, help="train onsets of a kept rule (length 1 and 2)")
    parser.add_argument("--pt_top", type=int, default=30)
    parser.add_argument("--pt_show", type=str, default="Car:MovTow(own)@1 & !Car:IncomLane(own)@1",
                        help="pairs to report whatever their result (';' separated, literals ' & ')")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    lab_dir = args.output_dir / "rule_lab"
    training_data = step05.load_training_data(cfg)
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    h = names.index(args.pt_head)
    A, L = model.num_atoms, model.num_literals
    base = model.literal_base.to(device)
    scope = base // (model.num_spans * A)
    allowed = model.input_mask.to(device) & ~((base % A == h) & (scope == 0))
    allowed &= ~rule_lab.negated_others(model).to(device)                   # as the "_nn" variants
    layout = training_data.get("grid_layout", {})
    for key in ("shape", "gestalt"):                                        # shape atoms: newest slot only
        if key in layout:
            allowed &= ~rule_lab.atom_literals(model, *layout[key], older_only=True).to(device)
    cols = torch.nonzero(allowed).flatten()
    text = lambda c: step05_mine.literal_text(model, names, int(cols[c]))
    print(f"[Pair table] {args.pt_head}: {len(cols)} of {L} literals allowed")

    head_mask = training_data["target_heads"].to(device)
    possible = step05.class_possible(training_data, vocab, device)
    S, w = model.slot_size, int(cfg.get("clause_slots", 1))
    tr_ids = sorted(training_data["train_videos"])[:args.pt_train_videos or None]
    va_ids = sorted(training_data["val_videos"])[:args.pt_val_videos or None]
    gen = lambda ids, d: head_rows(model, training_data, ids, h, possible, head_mask, S, w, device, d)

    # Length 1 (and the pair tables) on train
    tr = tables(gen(tr_ids, "train"), model, cols, device, "train")
    o1, _ = onsets(tr, device, pairs=False)
    base_p = tr["total_hit"] / max(tr["total"], 1)
    p1 = tr["h1"] / tr["n1"].clamp(min=1)
    cand = torch.nonzero((o1 >= args.pt_min_onsets) & (tr["n1"] > 0)).flatten()
    print(f"[Pair table] train: {len(tr_ids)} videos, {tr['total']} rows, {tr['total_hit']} hit rows, "
          f"{tr['num_events']} onsets, base precision {base_p:.4f}; {len(cand)} literals with >= "
          f"{args.pt_min_onsets} onsets")

    # Length 2 on train: every pair of candidates from the tables
    sub = lambda M: M[cand][:, cand]
    trc = {**tr, "hits": tr["hits"][:, cand.cpu()]}
    _o1c, o2 = onsets(trc, device)
    N2, H2 = sub(tr["N2"]), sub(tr["H2"])
    p2 = H2 / N2.clamp(min=1)
    pc = p1[cand]
    upper = torch.triu(torch.ones_like(N2, dtype=torch.bool), diagonal=1)
    parent = torch.maximum(pc[:, None], pc[None, :])
    has = upper & (N2 > 0)
    enough = has & (o2 >= args.pt_min_onsets)
    better = enough & (p2 > parent)

    # Val: the same counts
    va = tables(gen(va_ids, "val"), model, cols[cand], device, "val")
    vo1, vo2 = onsets(va, device)
    vp1 = va["h1"] / va["n1"].clamp(min=1)
    vp2 = va["H2"] / va["N2"].clamp(min=1)
    v_base = va["total_hit"] / max(va["total"], 1)
    v_better = better & (vp2 > torch.maximum(vp1[:, None], vp1[None, :])) & (va["N2"] > 0)

    n_c = len(cand)
    stats = {"literals_allowed": len(cols), "candidates": n_c, "pairs_tested": n_c * (n_c - 1) // 2,
             "pairs_with_rows": int(has.sum()), "pairs_min_onsets": int(enough.sum()),
             "pairs_better_train": int(better.sum()), "pairs_better_train_and_val": int(v_better.sum())}
    print(f"[Pair table] {stats}")

    def single(c):
        c = int(c)
        return {"rule": text(int(cand[c])), "train_precision": float(pc[c]), "train_rows": int(tr["n1"][cand[c]]),
                "train_recall": float(tr["h1"][cand[c]]) / max(tr["total_hit"], 1), "train_onsets": int(o1[cand[c]]),
                "val_precision": float(vp1[c]), "val_rows": int(va["n1"][c]), "val_onsets": int(vo1[c])}

    def pair(a, b):
        a, b = int(a), int(b)
        return {"rule": f"{text(int(cand[a]))} & {text(int(cand[b]))}",
                "train_precision": float(p2[a, b]), "parents_train": [float(pc[a]), float(pc[b])],
                "train_rows": int(N2[a, b]), "train_recall": float(H2[a, b]) / max(tr["total_hit"], 1),
                "train_onsets": int(o2[a, b]), "val_precision": float(vp2[a, b]),
                "parents_val": [float(vp1[a]), float(vp1[b])], "val_rows": int(va["N2"][a, b]),
                "val_onsets": int(vo2[a, b]), "better_on_val": bool(v_better[a, b])}

    singles = sorted((single(c) for c in range(n_c)), key=lambda x: -x["train_precision"])
    ia, ib = torch.nonzero(better, as_tuple=True)
    order = torch.argsort(p2[ia, ib], descending=True)
    kept = [pair(ia[k], ib[k]) for k in order.tolist()]
    by_onsets = sorted(kept, key=lambda x: -x["train_onsets"])
    # buckets of the kept pairs' train precision (and how many also improve on val)
    edges = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.01]
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = [x for x in kept if lo <= x["train_precision"] < hi]
        buckets.append({"train_precision": f"[{lo:g}, {min(hi, 1):g}{')' if hi < 1 else ']'}", "pairs": len(m),
                        "better_on_val": sum(x["better_on_val"] for x in m)})
    # the pairs asked for
    idx = {text(int(cand[c])): c for c in range(n_c)}
    shown = []
    for spec in (s for s in args.pt_show.split(";") if s.strip()):
        a, _, b = (x.strip() for x in spec.partition("&"))
        if a in idx and b in idx:
            x, y = sorted((idx[a], idx[b]))
            shown.append({**pair(x, y), "kept": bool(better[x, y])})
        else:
            shown.append({"rule": spec, "missing": [t for t in (a, b) if t not in idx]})

    md = [f"# Length-1 and length-2 rules of {args.pt_head}", "",
          f"Train: {len(tr_ids)} videos, {tr['total']} rows, {tr['total_hit']} hit rows, {tr['num_events']} onsets, "
          f"base precision {base_p:.4f}. Val: {len(va_ids)} videos, {va['total']} rows, {va['num_events']} onsets, "
          f"base precision {v_base:.4f}.", "",
          f"- literals allowed: {len(cols)}; length-1 rules with >= {args.pt_min_onsets} train onsets: {n_c}",
          f"- pairs tested: {stats['pairs_tested']}; with rows: {stats['pairs_with_rows']}; with >= "
          f"{args.pt_min_onsets} onsets: {stats['pairs_min_onsets']}",
          f"- pairs above both literals on train: {stats['pairs_better_train']}; also on val: "
          f"{stats['pairs_better_train_and_val']}", "",
          "| train precision | kept pairs | also better on val |", "|---|---|---|"]
    md += [f"| {b['train_precision']} | {b['pairs']} | {b['better_on_val']} |" for b in buckets]
    row1 = lambda x: (f"| {x['train_precision']:.3f} | {x['train_onsets']} | {x['train_recall']:.3f} | "
                      f"{x['val_precision']:.3f} | {x['val_onsets']} | {x['rule']} |")
    md += ["", f"## Length 1: the {args.pt_top} most precise (>= {args.pt_min_onsets} onsets)", "",
           "| train prec | onsets | recall | val prec | val onsets | rule |", "|---|---|---|---|---|---|"]
    md += [row1(x) for x in singles[:args.pt_top]]
    row2 = lambda x: (f"| {x['train_precision']:.3f} ({x['parents_train'][0]:.3f}, {x['parents_train'][1]:.3f}) | "
                      f"{x['train_onsets']} | {x['train_recall']:.3f} | {x['val_precision']:.3f} "
                      f"({x['parents_val'][0]:.3f}, {x['parents_val'][1]:.3f}) | {x['val_onsets']} | "
                      f"{'yes' if x['better_on_val'] else 'no'} | {x['rule']} |")
    hdr = ["| train prec (literals) | onsets | recall | val prec (literals) | val onsets | better on val | rule |",
           "|---|---|---|---|---|---|---|"]
    md += ["", f"## Length 2: the {args.pt_top} most precise kept pairs", ""] + hdr + [row2(x) for x in kept[:args.pt_top]]
    md += ["", f"## Length 2: the {args.pt_top} kept pairs with the most onsets", ""] + hdr
    md += [row2(x) for x in by_onsets[:args.pt_top]]
    md += ["", "## Pairs asked for (--pt_show)", ""]
    for x in shown:
        md.append(f"- {x}")
    stem = lab_dir / f"pairs_{args.pt_head.replace(':', '_')}"
    open(f"{stem}.md", "w").write("\n".join(md) + "\n")
    utils_data.save_json({"head": args.pt_head, "train_videos": len(tr_ids), "val_videos": len(va_ids),
                          "train": {"rows": tr["total"], "hit_rows": tr["total_hit"], "onsets": tr["num_events"]},
                          "val": {"rows": va["total"], "hit_rows": va["total_hit"], "onsets": va["num_events"]},
                          "stats": stats, "buckets": buckets, "singles": singles, "pairs": kept[:20000],
                          "shown": shown}, f"{stem}.json")
    print("\n".join(md[:30]))
    print(f"\n[Pair table] {stem}.md / .json")


if __name__ == "__main__":
    main()
