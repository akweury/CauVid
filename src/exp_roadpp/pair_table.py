"""Length-1, -2 and -3 rules of one head by tables (2026-11-27, the user's search principle).

Principle (the user): a literal is added to a rule if the precision rises, by any amount (no x1.5 gain); the recall
falling is expected. No literal may be missed: every combination is tested. The longer rules follow from tables:
  rows(a & b) = (F^T F)[a, b], hit rows(a & b) = (F_hit^T F_hit)[a, b]   (F: the rows x literals 0 / 1 matrix),
  rows(a & b & c) = (P^T F)[ab, c] with P the rows x seed-pairs matrix F_a & F_b (a second pass over the rows).

Rows: every at-risk row of the head (`step05_mine._head_rows`, no sample) of --pt_train_videos train videos and
--pt_val_videos val videos; a hit row has the onset within clause_slots slots; onsets: distinct (video, agent,
onset frame) among the hit rows. Literals: the rule net's literals the miner may use (`input_mask`, not the head
itself; as the main model's "_nn" variants: no negated "other" literals, the shape atoms in the newest slot only).
Metrics: precision, recall (hit rows / all hit rows), precision x recall, onsets; on train and val.
  length 1  every literal; a candidate if it has >= --pt_min_onsets train onsets;
  length 2  every pair of candidates; kept if (a) one of its literals is of the head's own agent ("(own)"),
            (b) its train precision is above both literals' and (c) it has >= --pt_min_onsets train onsets.
            A kept pair is extended (a seed of length 3) only if its val precision >= --pt_val_ratio (0.5) x its
            train precision (2026-11-27, the user);
  length 3  every seed x every candidate; kept if its train precision is above every sub-rule's (the 3 pairs and
            3 literals) and it has >= --pt_min_onsets train onsets ((a) holds by the seed). "extend": the same
            val test (for a later length 4).
"Better on val": also above every sub-rule on val.

    ./run_roadpp_docker.sh --gpu 3 --pair_table -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --pt_head Car:Jun --pt_train_videos 1000 --pt_val_videos 500
Writes <output_dir>/rule_lab/pairs_<head>.md / .json.
"""
import torch

from src.exp_roadpp import config, rule_lab, step05, step05_mine, utils_data, utils_pipe
from src.exp_roadpp.rule_direct import head_rows

MM = 256                # rows per bf16 product: counts <= 256 are exact in bf16, summed in float32


def cross(Lm, Rm, out=None):
    """(Lm^T Rm) of two (rows, K) / (rows, C) bool matrices, exact: bf16 products of MM rows, summed in float32."""
    out = torch.zeros((Lm.shape[1], Rm.shape[1]), device=Lm.device) if out is None else out
    for i in range(0, len(Lm), MM):
        out += (Lm[i:i + MM].to(torch.bfloat16).T @ Rm[i:i + MM].to(torch.bfloat16)).float()
    return out


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
            n1 += F.sum(0)
            h1 += F[y].sum(0)
            cross(F, F, N2)
            cross(F[y], F[y], H2)
            total += len(F)
            total_hit += int(y.sum())
            if keep_hits and y.any():
                hits.append(F[y].cpu())
                for i in torch.nonzero(y).flatten().tolist():
                    events.append(ev_ids.setdefault((vid, int(rk[i, 1]), int(rk[i, 0]) + int(first[i])), len(ev_ids)))
    return {"n1": n1, "h1": h1, "N2": N2, "H2": H2, "total": total, "total_hit": total_hit,
            "hits": torch.cat(hits) if hits else torch.zeros((0, C), dtype=torch.bool),
            "events": torch.tensor(events, dtype=torch.long), "num_events": len(ev_ids)}


def triple_counts(gen, model, cols, pa, pb, device):
    """A second pass: (K, C) rows where the seed pair k (literals pa[k], pb[k]) and the literal c hold."""
    N3 = torch.zeros((len(pa), len(cols)), device=device)
    with torch.no_grad():
        for _vid, inp, _y, _rk, _first in gen:
            F = model.literal_values(inp)[:, cols] > 0.5
            cross(F[:, pa] & F[:, pb], F, N3)
    return N3


def onsets_lr(Lh, Rh, ev, E, budget=2e8):
    """(K, C) distinct onsets where one hit row holds the left column k and the right column c."""
    K, C = Lh.shape[1], Rh.shape[1]
    out = torch.zeros((K, C), device=Lh.device)
    if not E:
        return out
    order = torch.argsort(ev)
    Lh, Rh, ev = Lh[order], Rh[order], ev[order]
    starts = torch.searchsorted(ev, torch.arange(E + 1, device=ev.device)).tolist()
    batch = max(1, int(budget // (K * C)))
    for e0 in range(0, E, batch):
        e1 = min(E, e0 + batch)
        n = max(starts[e + 1] - starts[e] for e in range(e0, e1))
        XL = torch.zeros((e1 - e0, n, K), dtype=torch.bfloat16, device=Lh.device)
        XR = torch.zeros((e1 - e0, n, C), dtype=torch.bfloat16, device=Lh.device)
        for k, e in enumerate(range(e0, e1)):
            XL[k, :starts[e + 1] - starts[e]] = Lh[starts[e]:starts[e + 1]].to(torch.bfloat16)
            XR[k, :starts[e + 1] - starts[e]] = Rh[starts[e]:starts[e + 1]].to(torch.bfloat16)
        out += ((XL.transpose(1, 2) @ XR).float() > 0).sum(0).float()
    return out


def onsets(t, device, pairs=True):
    """Distinct onsets per literal (C,) and per pair (C, C): an onset counts if one of its hit rows holds them."""
    Fh, ev, E = t["hits"].to(device), t["events"].to(device), t["num_events"]
    C = Fh.shape[1]
    o1 = (torch.zeros((E, C), device=device).index_add_(0, ev, Fh.float()) > 0).sum(0).float()
    return o1, (onsets_lr(Fh, Fh, ev, E) if pairs else None)


def triples(t, N3, pa, pb, device):
    """Hit rows (K, C) and onsets (K, C) of the seed pairs x literals, from the stored hit rows."""
    Fh, ev = t["hits"].to(device), t["events"].to(device)
    P = Fh[:, pa] & Fh[:, pb]
    return {"N3": N3, "H3": cross(P, Fh), "o3": onsets_lr(P, Fh, ev, t["num_events"])}


def buckets_of(rules):
    edges = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.01]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = [x for x in rules if lo <= x["train_precision"] < hi]
        out.append({"train_precision": f"[{lo:g}, {min(hi, 1):g}{')' if hi < 1 else ']'}", "rules": len(m),
                    "better_on_val": sum(x["better_on_val"] for x in m), "extend": sum(x["extend"] for x in m)})
    return out


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--pt_head", type=str, default="Car:Jun")
    parser.add_argument("--pt_train_videos", type=int, default=1000, help="train videos (the first; 0: all)")
    parser.add_argument("--pt_val_videos", type=int, default=500, help="val videos (the first; 0: all)")
    parser.add_argument("--pt_min_onsets", type=int, default=5, help="train onsets of a kept rule (every length)")
    parser.add_argument("--pt_val_ratio", type=float, default=0.5,
                        help="a kept rule of length >= 2 is extended only if val precision >= this x train precision")
    parser.add_argument("--pt_max_seeds", type=int, default=5000,
                        help="seed pairs of length 3 (the most train precision x recall; 0: all)")
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

    # ---- Length 1 (and the pair tables) on train
    tr = tables(gen(tr_ids, "train"), model, cols, device, "train")
    o1, _ = onsets(tr, device, pairs=False)
    base_p = tr["total_hit"] / max(tr["total"], 1)
    p1 = tr["h1"] / tr["n1"].clamp(min=1)
    cand = torch.nonzero((o1 >= args.pt_min_onsets) & (tr["n1"] > 0)).flatten()
    n_c = len(cand)
    ccols = cols[cand]
    ctext = lambda c: text(int(cand[c]))
    own = torch.tensor([model.literal_info(int(m))["scope"] == 0 for m in ccols.tolist()], device=device)
    print(f"[Pair table] train: {len(tr_ids)} videos, {tr['total']} rows, {tr['total_hit']} hit rows, "
          f"{tr['num_events']} onsets, base precision {base_p:.4f}; {n_c} literals with >= "
          f"{args.pt_min_onsets} onsets ({int(own.sum())} of the own agent)")
    T, T_hit = tr["total_hit"], max(tr["total_hit"], 1)

    # ---- Length 2 on train: every pair of candidates from the tables
    sub = lambda M: M[cand][:, cand]
    trc = {**tr, "hits": tr["hits"][:, cand.cpu()]}
    _o1c, o2 = onsets(trc, device)
    N2, H2 = sub(tr["N2"]), sub(tr["H2"])
    p2 = H2 / N2.clamp(min=1)
    pc = p1[cand]
    upper = torch.triu(torch.ones_like(N2, dtype=torch.bool), diagonal=1)
    has = upper & (N2 > 0)
    with_own = has & (own[:, None] | own[None, :])
    enough = with_own & (o2 >= args.pt_min_onsets)
    better = enough & (p2 > torch.maximum(pc[:, None], pc[None, :]))

    # ---- Val (length 1, 2): the same counts
    va = tables(gen(va_ids, "val"), model, ccols, device, "val")
    vo1, vo2 = onsets(va, device)
    V_hit = max(va["total_hit"], 1)
    vp1 = va["h1"] / va["n1"].clamp(min=1)
    vp2 = va["H2"] / va["N2"].clamp(min=1)
    v_base = va["total_hit"] / max(va["total"], 1)
    v_better = better & (va["N2"] > 0) & (vp2 > torch.maximum(vp1[:, None], vp1[None, :]))
    seed = better & (va["N2"] > 0) & (vp2 >= args.pt_val_ratio * p2)

    # ---- Length 3: the seeds x every candidate (a second pass over the rows)
    sa, sb = torch.nonzero(seed, as_tuple=True)
    pr2 = p2 * H2 / T_hit
    order = torch.argsort(pr2[sa, sb], descending=True)
    n_seeds = len(order)
    if args.pt_max_seeds and n_seeds > args.pt_max_seeds:
        order = order[:args.pt_max_seeds]
    pa, pb = sa[order], sb[order]
    K = len(pa)
    print(f"[Pair table] length 3: {K} of {n_seeds} seed pairs x {n_c} literals")
    t3 = triples(trc, triple_counts(gen(tr_ids, "train 3"), model, ccols, pa, pb, device), pa, pb, device)
    v3 = triples(va, triple_counts(gen(va_ids, "val 3"), model, ccols, pa, pb, device), pa, pb, device)
    p3 = t3["H3"] / t3["N3"].clamp(min=1)
    vp3 = v3["H3"] / v3["N3"].clamp(min=1)

    def above_subrules(P1, P2):
        """(K, C) the best precision of the 3 pairs and 3 literals inside each seed + literal."""
        P2f = torch.maximum(P2, P2.T)                                       # p2 is filled on both sides
        m = torch.maximum(P2f[pa, pb][:, None], torch.maximum(P2f[pa], P2f[pb]))
        return torch.maximum(m, torch.maximum(torch.maximum(P1[pa], P1[pb])[:, None], P1[None, :]))

    ar = torch.arange(K, device=device)
    valid = torch.ones((K, n_c), dtype=torch.bool, device=device)
    valid[ar, pa] = False
    valid[ar, pb] = False
    keep3 = valid & (t3["o3"] >= args.pt_min_onsets) & (p3 > above_subrules(pc, p2))
    v_better3 = keep3 & (v3["N3"] > 0) & (vp3 > above_subrules(vp1, vp2))
    extend3 = keep3 & (v3["N3"] > 0) & (vp3 >= args.pt_val_ratio * p3)
    # the same triple from several seeds has the same counts: keep one
    ik, ic = torch.nonzero(keep3, as_tuple=True)
    trip = torch.sort(torch.stack([pa[ik], pb[ik], ic], dim=1), dim=1).values
    key = (trip[:, 0] * n_c + trip[:, 1]) * n_c + trip[:, 2]
    uniq, inv = torch.unique(key, return_inverse=True)
    first = torch.full((len(uniq),), len(key), device=device).scatter_reduce_(
        0, inv, torch.arange(len(key), device=device), "amin")
    ik, ic = ik[first], ic[first]

    stats = {"literals_allowed": len(cols), "candidates": n_c, "candidates_own": int(own.sum()),
             "pairs_tested": n_c * (n_c - 1) // 2, "pairs_with_rows": int(has.sum()),
             "pairs_with_own": int(with_own.sum()), "pairs_min_onsets": int(enough.sum()),
             "pairs_better_train": int(better.sum()), "pairs_better_train_and_val": int(v_better.sum()),
             "pairs_extend": n_seeds, "seeds_used": K, "triples_tested": int(valid.sum()),
             "triples_kept": len(ik), "triples_better_train_and_val": int(v_better3[ik, ic].sum()),
             "triples_extend": int(extend3[ik, ic].sum())}
    print(f"[Pair table] {stats}")

    def metrics(prec, hit, ons, vprec, vhit, vons, vrows, rows):
        r, vr = hit / T_hit, vhit / V_hit
        return {"train_precision": prec, "train_recall": r, "train_pr": prec * r, "train_onsets": ons,
                "train_rows": rows, "val_precision": vprec, "val_recall": vr, "val_pr": vprec * vr,
                "val_onsets": vons, "val_rows": vrows}

    def single(c):
        c = int(c)
        g = int(cand[c])
        return {"rule": ctext(c), **metrics(float(pc[c]), float(tr["h1"][g]), int(o1[g]), float(vp1[c]),
                                            float(va["h1"][c]), int(vo1[c]), int(va["n1"][c]), int(tr["n1"][g]))}

    def pair(a, b):
        a, b = int(a), int(b)
        return {"rule": f"{ctext(a)} & {ctext(b)}",
                **metrics(float(p2[a, b]), float(H2[a, b]), int(o2[a, b]), float(vp2[a, b]), float(va["H2"][a, b]),
                          int(vo2[a, b]), int(va["N2"][a, b]), int(N2[a, b])),
                "parents_train": [float(pc[a]), float(pc[b])], "parents_val": [float(vp1[a]), float(vp1[b])],
                "better_on_val": bool(v_better[a, b]), "extend": bool(seed[a, b])}

    def triple(k, c):
        k, c = int(k), int(c)
        a, b = int(pa[k]), int(pb[k])
        return {"rule": f"{ctext(a)} & {ctext(b)} & {ctext(c)}",
                **metrics(float(p3[k, c]), float(t3["H3"][k, c]), int(t3["o3"][k, c]), float(vp3[k, c]),
                          float(v3["H3"][k, c]), int(v3["o3"][k, c]), int(v3["N3"][k, c]), int(t3["N3"][k, c])),
                "seed_train": float(p2[a, b]), "seed_val": float(vp2[a, b]),
                "better_on_val": bool(v_better3[k, c]), "extend": bool(extend3[k, c])}

    singles = [single(c) for c in range(n_c)]
    ia, ib = torch.nonzero(better, as_tuple=True)
    kept2 = [pair(ia[k], ib[k]) for k in torch.argsort(p2[ia, ib], descending=True).tolist()]
    # length 3: the dicts of the top rules only (and up to 20000 in the json)
    lists3 = {}
    for name, score in (("precision", p3[ik, ic]), ("pr", p3[ik, ic] * t3["H3"][ik, ic]),
                        ("onsets", t3["o3"][ik, ic])):
        lists3[name] = torch.argsort(score, descending=True)
    kept3_json = [triple(ik[j], ic[j]) for j in lists3["precision"][:20000].tolist()]
    top3 = {n: [triple(ik[j], ic[j]) for j in o[:args.pt_top].tolist()] for n, o in lists3.items()}

    idx = {ctext(c): c for c in range(n_c)}
    shown = []
    for spec in (s for s in args.pt_show.split(";") if s.strip()):
        a, _, b = (x.strip() for x in spec.partition("&"))
        if a in idx and b in idx:
            x, y = sorted((idx[a], idx[b]))
            shown.append({**pair(x, y), "kept": bool(better[x, y])})
        else:
            shown.append({"rule": spec, "missing": [t for t in (a, b) if t not in idx]})

    top = lambda rules, k: sorted(rules, key=lambda x: -x[k])[:args.pt_top]
    cells = lambda x: (f"{x['train_precision']:.3f} | {x['train_recall']:.3f} | {x['train_pr']:.4f} | "
                       f"{x['train_onsets']} | {x['val_precision']:.3f} | {x['val_recall']:.3f} | {x['val_pr']:.4f} | "
                       f"{x['val_onsets']}")
    head = "train P | train R | train PxR | onsets | val P | val R | val PxR | val onsets"
    sep = lambda n: "|" + "---|" * n
    row1 = lambda x: f"| {cells(x)} | {x['rule']} |"
    row2 = lambda x: (f"| {cells(x)} | {x['parents_train'][0]:.3f}, {x['parents_train'][1]:.3f} | "
                      f"{x['parents_val'][0]:.3f}, {x['parents_val'][1]:.3f} | {'yes' if x['better_on_val'] else 'no'}"
                      f" | {'yes' if x['extend'] else 'no'} | {x['rule']} |")
    row3 = lambda x: (f"| {cells(x)} | {x['seed_train']:.3f} | {x['seed_val']:.3f} | "
                      f"{'yes' if x['better_on_val'] else 'no'} | {'yes' if x['extend'] else 'no'} | {x['rule']} |")
    hdr1 = [f"| {head} | rule |", sep(9)]
    hdr2 = [f"| {head} | literals train P | literals val P | better on val | extend | rule |", sep(13)]
    hdr3 = [f"| {head} | seed train P | seed val P | better on val | extend | rule |", sep(13)]
    bucket_md = lambda bs: (["| train precision | kept | also better on val | extend |", sep(4)] +
                            [f"| {b['train_precision']} | {b['rules']} | {b['better_on_val']} | {b['extend']} |"
                             for b in bs])
    b2, b3 = buckets_of(kept2), buckets_of(kept3_json)
    md = [f"# Length-1, -2 and -3 rules of {args.pt_head}", "",
          f"Train: {len(tr_ids)} videos, {tr['total']} rows, {T} hit rows, {tr['num_events']} onsets, "
          f"base precision {base_p:.4f}. Val: {len(va_ids)} videos, {va['total']} rows, {va['total_hit']} hit rows, "
          f"{va['num_events']} onsets, base precision {v_base:.4f}.",
          "P precision, R recall (hit rows / all hit rows), PxR their product. Length >= 2 needs an (own) literal; "
          f"extend: val P >= {args.pt_val_ratio:g} x train P.", "",
          f"- literals allowed: {len(cols)}; length 1 with >= {args.pt_min_onsets} train onsets: {n_c} "
          f"({stats['candidates_own']} own)",
          f"- pairs: tested {stats['pairs_tested']}, with rows {stats['pairs_with_rows']}, with an own literal "
          f"{stats['pairs_with_own']}, >= {args.pt_min_onsets} onsets {stats['pairs_min_onsets']}, above both "
          f"literals on train {stats['pairs_better_train']}, also on val {stats['pairs_better_train_and_val']}, "
          f"extended {stats['pairs_extend']} (used {K})",
          f"- triples: tested {stats['triples_tested']}, kept (above every sub-rule on train) {stats['triples_kept']}, "
          f"also on val {stats['triples_better_train_and_val']}, extend {stats['triples_extend']}", "",
          "## Kept pairs by train precision", ""] + bucket_md(b2) + [
          "", "## Kept triples by train precision" + (" (the 20000 most precise)" if len(ik) > 20000 else ""), ""
          ] + bucket_md(b3)
    for k, what in (("train_precision", "train precision"), ("train_pr", "train PxR")):
        md += ["", f"## Length 1: top {args.pt_top} by {what}", ""] + hdr1 + [row1(x) for x in top(singles, k)]
    for k, what in (("train_precision", "train precision"), ("train_pr", "train PxR"), ("train_onsets", "onsets")):
        md += ["", f"## Length 2: top {args.pt_top} kept pairs by {what}", ""] + hdr2 + [row2(x) for x in top(kept2, k)]
    for k, what in (("precision", "train precision"), ("pr", "train PxR"), ("onsets", "onsets")):
        md += ["", f"## Length 3: top {args.pt_top} kept triples by {what}", ""] + hdr3 + [row3(x) for x in top3[k]]
    md += ["", "## Pairs asked for (--pt_show)", ""] + [f"- {x}" for x in shown]
    stem = lab_dir / f"pairs_{args.pt_head.replace(':', '_')}"
    open(f"{stem}.md", "w").write("\n".join(md) + "\n")
    utils_data.save_json({"head": args.pt_head, "train_videos": len(tr_ids), "val_videos": len(va_ids),
                          "train": {"rows": tr["total"], "hit_rows": T, "onsets": tr["num_events"]},
                          "val": {"rows": va["total"], "hit_rows": va["total_hit"], "onsets": va["num_events"]},
                          "stats": stats, "buckets_pairs": b2, "buckets_triples": b3, "singles": singles,
                          "pairs": kept2[:20000], "triples": kept3_json, "shown": shown}, f"{stem}.json")
    print("\n".join(md[:40]))
    print(f"\n[Pair table] {stem}.md / .json")


if __name__ == "__main__":
    main()
