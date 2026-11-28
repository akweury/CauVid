"""Length-1, -2 and -3 rules of one head by tables (2026-11-27, the user's search principle).

Principle (the user): a literal is added to a rule if the precision rises, by any amount (no x1.5 gain), on train
AND on val (val precision is the final one); the recall falling is expected. No literal may be missed: every
combination is tested, from tables of the length-1 results:
  rows(a & b) = (F^T F)[a, b], hit rows(a & b) = (F_hit^T F_hit)[a, b]   (F: the rows x literals 0 / 1 matrix),
  rows(a & b & c) = (P^T F)[ab, c] with P the rows x seed-pairs matrix F_a & F_b.
The literal bits of every row are read once and kept (train and val); the seeds go through in blocks.

Rows: every at-risk row of the head (`step05_mine._head_rows`, no sample) of --pt_train_videos train videos and
--pt_val_videos val videos; a hit row has the onset within clause_slots slots; onsets: distinct (video, agent,
onset frame) among the hit rows. Literals: the rule net's literals the miner may use (`input_mask`, not the head
itself; as the main model's "_nn" variants: no negated "other" literals, the shape atoms in the newest slot only).
Metrics: precision P, recall R (hit rows / all hit rows), P x R, onsets; on train and val.
  length 1  every literal; reported: how many remain if the literals with <= k train onsets are dropped (k = 0..5);
            a seed (candidate) if it has >= --pt_min_onsets1 (2) train onsets. Literals true on exactly the same
            train and val rows are merged (one column, "(= ...)" lists the others).
  length 2  every pair of seeds; kept if (a) one of its literals is of the head's own agent ("(own)"),
            (b) its precision is above both literals' on train and on val, (c) >= --pt_min_onsets (5) train onsets.
            A kept pair is extended (a seed of length 3) only if val P >= --pt_val_ratio (0.5) x train P;
  length 3  every seed pair x every seed literal; kept if its precision is above every sub-rule's (the 3 pairs
            and 3 literals) on train and on val and it has >= --pt_min_onsets train onsets ((a) holds by the
            seed). "extend": the same val ratio test (for a later length 4).

    ./run_roadpp_docker.sh --gpu 3 --pair_table -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --pt_head Car:Jun --pt_train_videos 1000 --pt_val_videos 500
Writes <output_dir>/rule_lab/pairs_<head>.md / .json.
"""
import math

import torch

from src.exp_roadpp import config, rule_lab, step05, step05_mine, utils_data, utils_pipe
from src.exp_roadpp.rule_direct import head_rows

# Rows per half-precision product. Every partial sum is a count <= MM, exact in float16 (11 bits) up to 2048,
# so the result is exact whatever the reduction order; summed in float32.
MM = 2048


def _dt(device):
    return torch.float16 if torch.device(device).type == "cuda" else torch.float32


def cross(Lm, Rm, out=None):
    """(Lm^T Rm) of two (rows, K) / (rows, C) bool matrices, exact."""
    out = torch.zeros((Lm.shape[1], Rm.shape[1]), device=Lm.device) if out is None else out
    dt = _dt(Lm.device)
    for i in range(0, len(Lm), MM):
        out += (Lm[i:i + MM].to(dt).T @ Rm[i:i + MM].to(dt)).float()
    return out


def cross_pairs(F, pa, pb):
    """(K, C) rows where the pair k (columns pa[k], pb[k]) and the column c hold, without the (rows, K) matrix."""
    out = torch.zeros((len(pa), F.shape[1]), device=F.device)
    dt = _dt(F.device)
    for i in range(0, len(F), MM):
        f = F[i:i + MM]
        out += ((f[:, pa] & f[:, pb]).to(dt).T @ f.to(dt)).float()
    return out


def collect(gen, model, cols):
    """One pass: the literal bits (rows, C) of every row (on the CPU), the hit flags, the hit rows' event index and
    per row (video index, agent n, frame t, frames to the agent's next onset; capped at the largest window)."""
    Fs, ys, metas, events, ev_ids, vid_ids = [], [], [], [], {}, {}
    with torch.no_grad():
        for vid, inp, y, rk, first in gen:
            Fs.append((model.literal_values(inp)[:, cols] > 0.5).cpu())
            ys.append(y.cpu())
            v = torch.full((len(y),), vid_ids.setdefault(vid, len(vid_ids)), dtype=torch.long)
            metas.append(torch.stack([v, rk[:, 1].long().cpu(), rk[:, 0].long().cpu(), first.long().cpu()], 1))
            for i in torch.nonzero(y).flatten().tolist():
                events.append(ev_ids.setdefault((vid, int(rk[i, 1]), int(rk[i, 0]) + int(first[i])), len(ev_ids)))
    C = len(cols)
    return {"F": torch.cat(Fs) if Fs else torch.zeros((0, C), dtype=torch.bool),
            "y": torch.cat(ys) if ys else torch.zeros(0, dtype=torch.bool),
            "meta": torch.cat(metas) if metas else torch.zeros((0, 4), dtype=torch.long), "vids": list(vid_ids),
            "ev": torch.tensor(events, dtype=torch.long), "E": len(ev_ids)}


def onsets_lr(Lh, Rh, ev, E, budget=2e8):
    """(K, C) distinct onsets where one hit row holds the left column k and the right column c."""
    K, C = Lh.shape[1], Rh.shape[1]
    out = torch.zeros((K, C), device=Lh.device)
    if not E:
        return out
    order = torch.argsort(ev)
    Lh, Rh, ev = Lh[order], Rh[order], ev[order]
    starts = torch.searchsorted(ev, torch.arange(E + 1, device=ev.device)).tolist()
    batch = max(1, int(budget // max(K * C, 1)))
    dt = torch.bfloat16 if Lh.device.type == "cuda" else torch.float32     # only "> 0" is used
    for e0 in range(0, E, batch):
        e1 = min(E, e0 + batch)
        n = max(starts[e + 1] - starts[e] for e in range(e0, e1))
        XL = torch.zeros((e1 - e0, n, K), dtype=dt, device=Lh.device)
        XR = torch.zeros((e1 - e0, n, C), dtype=dt, device=Lh.device)
        for k, e in enumerate(range(e0, e1)):
            XL[k, :starts[e + 1] - starts[e]] = Lh[starts[e]:starts[e + 1]].to(dt)
            XR[k, :starts[e + 1] - starts[e]] = Rh[starts[e]:starts[e + 1]].to(dt)
        out += ((XL.transpose(1, 2) @ XR).float() > 0).sum(0).float()
    return out


def onsets1(Fh, ev, E):
    """(C,) distinct onsets per column."""
    return (torch.zeros((E, Fh.shape[1]), device=Fh.device).index_add_(0, ev, Fh.float()) > 0).sum(0).float()


def split_tables(d, idx, device):
    """Length 1 / 2 counts of the columns idx of one split, plus its rows on the device for length 3."""
    F = d["F"][:, idx.cpu()].to(device)
    y, ev = d["y"].to(device), d["ev"].to(device)
    Fh = F[y]
    ev_row = torch.full((len(F),), -1, dtype=torch.long, device=device)
    ev_row[y] = ev
    return {"F": F, "Fh": Fh, "y": y, "ev": ev, "ev_row": ev_row, "meta": d.get("meta", torch.zeros((len(F), 4), dtype=torch.long)).to(device), "E": d["E"], "total": len(F), "total_hit": int(y.sum()),
            "n1": F.sum(0).float(), "h1": Fh.sum(0).float(), "o1": onsets1(Fh, ev, d["E"]),
            "N2": cross(F, F), "H2": cross(Fh, Fh), "o2": onsets_lr(Fh, Fh, ev, d["E"])}


def merge_identical(Ftr, Fva, device):
    """Representative columns: the columns true on exactly the same train and val rows share one. Returns
    (reps (R,), members: list per rep of the column indices, the rep first)."""
    C = Ftr.shape[1]
    if not C:
        return torch.zeros(0, dtype=torch.long), []
    g = torch.Generator().manual_seed(0)
    hsh = torch.zeros(C, dtype=torch.float64, device=device)
    for F in (Ftr, Fva):
        r = torch.randn(len(F), generator=g, dtype=torch.float64).to(device)
        for i in range(0, len(F), 65536):
            hsh += r[i:i + 65536] @ F[i:i + 65536].to(device).double()
    _u, inv = torch.unique(hsh, return_inverse=True)
    first = torch.full((int(inv.max()) + 1,), C, device=device).scatter_reduce_(
        0, inv, torch.arange(C, device=device), "amin")
    rep = first[inv]                                                    # (C,) the hash group's first column
    diff = torch.zeros(C, device=device)
    for F in (Ftr, Fva):                                                # exact check (a hash collision: kept apart)
        for i in range(0, len(F), 65536):
            f = F[i:i + 65536].to(device)
            diff += (f ^ f[:, rep]).sum(0).float()
    rep = torch.where(diff > 0, torch.arange(C, device=device), rep).tolist()
    groups = {}
    for c, r in enumerate(rep):
        groups.setdefault(r, []).append(c)
    reps = sorted(groups)
    return torch.tensor(reps, dtype=torch.long), [groups[r] for r in reps]


class Top:
    """The best rows (n, 10: a, b, c, rows, hit rows, onsets, val rows, val hit rows, val onsets, extend) by a
    score, kept to cap while blocks of rows are added."""

    def __init__(self, score, cap):
        self.score, self.cap, self.rows = score, cap, torch.zeros((0, 10), dtype=torch.float64)

    def add(self, rows):
        x = torch.cat([self.rows, rows])
        self.rows = x[torch.argsort(self.score(x), descending=True)[:self.cap]]


EDGES = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.01]


def bucket_counts(prec, extend):
    """Per train precision bucket: [kept rules, how many of them extend]."""
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (prec >= lo) & (prec < hi)
        out.append([int(m.sum()), int((m & extend).sum())])
    return out


def above_subrules(P1, P2, pa, pb):
    """(K, C) the best precision of the 3 pairs and 3 literals inside each seed pair (pa, pb) + literal c."""
    m = torch.maximum(P2[pa, pb][:, None], torch.maximum(P2[pa], P2[pb]))
    return torch.maximum(m, torch.maximum(torch.maximum(P1[pa], P1[pb])[:, None], P1[None, :]))


def length3(tr, va, sa, sb, sid, min_onsets, ratio, block, tops):
    """Every seed pair (sa, sb) x every literal, in blocks of seeds. Kept: >= min_onsets train onsets and the
    precision above every sub-rule on train and on val; extend: val P >= ratio x train P. A triple is counted once,
    at the first of its seed pairs (the same counts from each). Returns the counts, the per bucket [kept, extend]
    and every kept triple (n, 7 float32: a, b, c, rows, hit rows, val rows, val hit rows; on the CPU); adds the
    kept triples to the Top lists."""
    device = tr["F"].device
    P = lambda t: t["H2"] / t["N2"].clamp(min=1)
    p1 = lambda t: t["h1"] / t["n1"].clamp(min=1)
    pc, p2, vp1, vp2 = p1(tr), P(tr), p1(va), P(va)
    n3 = {"tested": 0, "better_train": 0, "kept": 0, "extend": 0}
    b3 = [[0, 0] for _ in EDGES[:-1]]
    kept = []
    n_seeds = len(sa)
    for k0 in range(0, n_seeds, block):
        pa, pb = sa[k0:k0 + block], sb[k0:k0 + block]
        K = len(pa)
        ar = torch.arange(K, device=device)
        kk = (k0 + ar)[:, None]
        s1, s2 = sid[pa], sid[pb]                                           # seed ids of (a, c), (b, c)
        valid = ((s1 < 0) | (s1 > kk)) & ((s2 < 0) | (s2 > kk))
        valid[ar, pa] = False
        valid[ar, pb] = False
        N3, H3 = cross_pairs(tr["F"], pa, pb), cross_pairs(tr["Fh"], pa, pb)
        o3 = onsets_lr(tr["Fh"][:, pa] & tr["Fh"][:, pb], tr["Fh"], tr["ev"], tr["E"])
        p3 = H3 / N3.clamp(min=1)
        bt = valid & (o3 >= min_onsets) & (p3 > above_subrules(pc, p2, pa, pb))
        vN3, vH3 = cross_pairs(va["F"], pa, pb), cross_pairs(va["Fh"], pa, pb)
        vo3 = onsets_lr(va["Fh"][:, pa] & va["Fh"][:, pb], va["Fh"], va["ev"], va["E"])
        vp3 = vH3 / vN3.clamp(min=1)
        keep = bt & (vN3 > 0) & (vp3 > above_subrules(vp1, vp2, pa, pb))
        ext = keep & (vp3 >= ratio * p3)
        n3["tested"] += int(valid.sum())
        n3["better_train"] += int(bt.sum())
        n3["kept"] += int(keep.sum())
        n3["extend"] += int(ext.sum())
        ik, ic = torch.nonzero(keep, as_tuple=True)
        for j, (a, e) in enumerate(bucket_counts(p3[ik, ic], ext[ik, ic])):
            b3[j][0] += a
            b3[j][1] += e
        rows = torch.stack([pa[ik].double(), pb[ik].double(), ic.double()] +
                           [M[ik, ic].double() for M in (N3, H3, o3, vN3, vH3, vo3, ext)], dim=1).cpu()
        for t in tops.values():
            t.add(rows)
        kept.append(rows[:, [0, 1, 2, 3, 4, 6, 7]].float())
        print(f"[Pair table]   seeds {k0 + K}/{n_seeds}: triples kept so far {n3['kept']}")
    return n3, b3, (torch.cat(kept) if kept else torch.zeros((0, 7)))


CHECK_FRACS = (0.5, 0.8, 0.9, 0.95, 0.99, 1.0)


def coverage(va, rules, order, budget=5e7):
    """Val coverage of the rules (n, 3 column indices; the index C = always true) taken in the given order: per
    prefix k (1..n) the val rows, hit rows and onsets where at least one of the first k rules holds."""
    C = va["F"].shape[1]
    FT = torch.cat([va["F"], torch.ones((len(va["F"]), 1), dtype=torch.bool, device=va["F"].device)], 1).T
    FT = FT.contiguous()
    y, ev, E = va["y"], va["ev"], va["E"]
    union = torch.zeros(FT.shape[1], dtype=torch.bool, device=FT.device)
    out = {"rows": [], "hit_rows": [], "onsets": []}
    B = max(1, int(budget // max(FT.shape[1], 1)))
    rules = rules.to(FT.device).clamp(max=C)
    order = order.to(FT.device)
    for i in range(0, len(order), B):
        r = rules[order[i:i + B]]
        m = FT[r[:, 0]] & FT[r[:, 1]] & FT[r[:, 2]]                          # (b, rows)
        cum = (m.int().cumsum(0) > 0) | union
        ch = cum[:, y]
        out["rows"].append(cum.sum(1).cpu())
        out["hit_rows"].append(ch.sum(1).cpu())
        out["onsets"].append(((torch.zeros((len(ch), E), device=FT.device).index_add_(1, ev, ch.float()) > 0)
                              .sum(1)).cpu())
        union = cum[-1]
    return {k: torch.cat(v) if v else torch.zeros(0, dtype=torch.long) for k, v in out.items()}


def coverage_report(cov, score, E, V_hit):
    """How many rules (taken in order) cover a fraction of the most val onsets the set covers, and a curve."""
    ons = cov["onsets"]
    n = len(ons)
    best = int(ons[-1]) if n else 0
    point = lambda k: {"rules": k, "onsets": int(ons[k - 1]), "onset_recall": int(ons[k - 1]) / max(E, 1),
                       "hit_rows": int(cov["hit_rows"][k - 1]), "row_recall": int(cov["hit_rows"][k - 1]) / V_hit,
                       "rows": int(cov["rows"][k - 1]),
                       "precision": int(cov["hit_rows"][k - 1]) / max(int(cov["rows"][k - 1]), 1),
                       "score_of_last": float(score[k - 1])}
    checks = []
    for f in CHECK_FRACS:
        k = int(torch.searchsorted(ons, torch.tensor(math.ceil(f * best - 1e-9), dtype=ons.dtype))) + 1 if n else 0
        checks.append({"frac_of_max": f, **(point(min(k, n)) if n else {"rules": 0})})
    ks = sorted({k for k in [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000, 20000, 50000,
                             100000, 200000, 500000, 1000000] if k <= n} | ({n} if n else set()))
    return {"rules_total": n, "max_onsets": best, "checks": checks, "curve": [point(k) for k in ks]}


def rule_mask(t, r):
    """(rows,) where the rule r (column indices; >= C: the always-true padding) holds."""
    m = torch.ones(len(t["F"]), dtype=torch.bool, device=t["F"].device)
    for c in r:
        if c < t["F"].shape[1]:
            m &= t["F"][:, c]
    return m


def summary(t, m):
    """Rows, hit rows, onsets, precision and onset recall of the rows m."""
    hit = m & t["y"]
    ons = int((torch.bincount(t["ev_row"][hit], minlength=t["E"]) > 0).sum()) if t["E"] else 0
    rows, hits = int(m.sum()), int(hit.sum())
    return {"rows": rows, "hit_rows": hits, "negatives": rows - hits, "onsets": ons,
            "precision": hits / max(rows, 1), "onset_recall": ons / max(t["E"], 1)}


def split_counts(t, m):
    """Per literal c over the rows m: rows and hit rows with c, and the onsets of m lost if the rows with c are
    removed (every hit row of the onset in m has c)."""
    Fm, ym = t["F"][m], t["y"][m]
    Fh, evm = Fm[ym], t["ev_row"][m][ym]
    tot = torch.bincount(evm, minlength=t["E"]).float()
    cnt = torch.zeros((t["E"], Fm.shape[1]), device=Fm.device).index_add_(0, evm, Fh.float())
    lost = ((cnt == tot[:, None]) & (tot > 0)[:, None]).sum(0)
    return Fm.sum(0).float(), Fh.sum(0).float(), lost


def exceptions(tr, va, r, max_steps, min_share, ratio, val_ratio):
    """The exception literals of the rule r (2026-11-27, the user): literals that hold where the body holds but the
    head does not. Greedy: at each step the literal c that removes the largest share of r's train negatives (rows
    where r holds and the head not) from the rows left, such that
      train: it removes >= min_share of r's negatives and P(head | rows left & c) <= ratio x P(r);
      val:   the same purity (<= ratio x val P(r)) and it removes >= val_ratio x its train share of the negatives;
    then r := r & !c. Returns the start and every step (train / val after the step)."""
    m, vm = rule_mask(tr, r), rule_mask(va, r)
    s0, v0 = summary(tr, m), summary(va, vm)
    P, vP, N0, vN0 = s0["precision"], v0["precision"], max(s0["negatives"], 1), max(v0["negatives"], 1)
    banned = torch.zeros(tr["F"].shape[1], dtype=torch.bool, device=m.device)
    banned[[c for c in r if c < len(banned)]] = True
    steps, fail_val = [], None
    for _ in range(max_steps):
        n, h, lost = split_counts(tr, m)
        vn, vh, vlost = split_counts(va, vm)
        share, vshare = (n - h) / N0, (vn - vh) / vN0
        ok_tr = ~banned & (share >= min_share) & (h / n.clamp(min=1) <= ratio * P)
        ok = ok_tr & (vn > 0) & (vh / vn.clamp(min=1) <= ratio * vP) & (vshare >= val_ratio * share)
        if fail_val is None:
            fail_val = int((ok_tr & ~ok).sum())
        if not ok.any():
            break
        c = int(torch.argmax(torch.where(ok, share, torch.full_like(share, -1))))
        banned[c] = True
        step = {"literal": c, "share": float(share[c]), "val_share": float(vshare[c]),
                "precision_with": float(h[c] / n[c].clamp(min=1)), "val_precision_with": float(vh[c] / vn[c].clamp(min=1)),
                "onsets_lost": int(lost[c]), "val_onsets_lost": int(vlost[c]), "candidates": int(ok.sum())}
        m &= ~tr["F"][:, c]
        vm &= ~va["F"][:, c]
        step["train"], step["val"] = summary(tr, m), summary(va, vm)
        steps.append(step)
    return {"start": {"train": s0, "val": v0}, "steps": steps, "first_step_failed_val": fail_val}


def near_duplicates(tr, va, kept, score, tau):
    """Kept pairs (kept (C, C) upper triangle) that are near-identical to a literal or to a better pair: Jaccard of
    their rows >= tau on train and on val. A pair (a, b) and the literal a: |a & b| / |a|; pairs (a, b), (a, c)
    sharing a literal: |a & b & c| / |(a & b) | (a & c)|, from the cross table of the rows where a holds. Pairs
    not sharing a literal are not compared. Greedy: a pair near-identical to one of its literals goes to the
    literal; the others by score (high first), each unassigned pair absorbs its unassigned near-identical pairs.
    Returns (rep (P,) long: -1 the pair stays, >= 0 the absorbing pair's index, < -1: the literal -2 - a;
    the pairs' (ia, ib))."""
    C = kept.shape[0]
    device = kept.device
    ia, ib = torch.nonzero(kept, as_tuple=True)
    P = len(ia)
    pid = torch.full((C, C), -1, dtype=torch.long, device=device)
    pid[ia, ib] = torch.arange(P, device=device)
    pid[ib, ia] = torch.arange(P, device=device)
    to_lit = torch.full((P,), -1, dtype=torch.long, device=device)
    edges = []
    sym = kept | kept.T
    for a in range(C):
        B = torch.nonzero(sym[a]).flatten()
        if not len(B):
            continue
        Jp, Jl = None, None
        for t in (tr, va):
            G = t["F"][t["F"][:, a]][:, B]
            M = cross(G, G)
            d = torch.diagonal(M)
            J = M / (d[:, None] + d[None, :] - M).clamp(min=1)
            jl = d / t["n1"][a].clamp(min=1)
            Jp = J if Jp is None else torch.minimum(Jp, J)
            Jl = jl if Jl is None else torch.minimum(Jl, jl)
        ids = pid[a, B]
        near_lit = Jl >= tau
        to_lit[ids[near_lit & (to_lit[ids] < 0)]] = a
        i, j = torch.nonzero(torch.triu(Jp >= tau, diagonal=1), as_tuple=True)
        if len(i):
            edges.append(torch.stack([ids[i], ids[j]], 1))
    rep = torch.full((P,), -1, dtype=torch.long)
    lit = to_lit.cpu()
    rep[lit >= 0] = -2 - lit[lit >= 0]
    adj = {}
    if edges:
        for x, y in torch.cat(edges).cpu().tolist():
            adj.setdefault(x, []).append(y)
            adj.setdefault(y, []).append(x)
    taken = lit >= 0
    for r in torch.argsort(score.cpu(), descending=True).tolist():
        if taken[r]:
            continue
        taken[r] = True
        for q in adj.get(r, []):
            if not taken[q]:
                taken[q] = True
                rep[q] = r
    return rep, (ia, ib)


def fp_classes(t, m, H, S, masks=False):
    """The negative rows of the rows m (rule holds, no onset within the hit window) by what the agent does:
      early   the agent's onset is later within H frames (the rule fired too early), by slots ahead (first // S);
      later   the agent has an onset later than H frames;
      after   the agent's onsets are all before t (the rule fires after the action);
      never   the agent has no onset (among the split's at-risk rows).
    Agents: distinct (video, agent) per class. masks: also "_masks" {class: (rows,) bool over all rows of t}."""
    meta, y = t["meta"], t["y"]
    key = meta[:, 0] * 1_000_000 + meta[:, 1]
    on_key = key[y]
    on_frame = (meta[:, 2] + meta[:, 3])[y]
    uk, inv = torch.unique(on_key, return_inverse=True)
    last = torch.full((len(uk),), -1, dtype=torch.long, device=y.device).scatter_reduce_(0, inv, on_frame, "amax")
    neg = m & ~y
    k, tt, first = key[neg], meta[neg, 2], meta[neg, 3]
    pos = torch.searchsorted(uk, k).clamp(max=max(len(uk) - 1, 0))
    has = (uk[pos] == k) if len(uk) else torch.zeros_like(k, dtype=torch.bool)
    lastf = torch.where(has, last[pos] if len(uk) else k, torch.full_like(k, -1))
    early = first < H
    later = ~early & has & (lastf > tt)
    after = ~early & ~later & has
    never = ~early & ~has
    out = {"negatives": int(neg.sum())}
    for name, msk in (("early", early), ("later", later), ("after", after), ("never", never)):
        out[name] = {"rows": int(msk.sum()), "share": float(msk.sum()) / max(int(neg.sum()), 1),
                     "agents": int(torch.unique(k[msk]).numel())}
    slots = torch.bincount(first[early] // S, minlength=H // S + 1).tolist() if early.any() else []
    out["early_by_slots_ahead"] = {int(i): int(c) for i, c in enumerate(slots) if c}
    if masks:
        idx = torch.nonzero(neg).flatten()
        full = lambda x: torch.zeros_like(neg).index_fill_(0, idx[x], True)
        out["_masks"] = {"early": full(early), "later": full(later), "after": full(after), "never": full(never)}
    return out


def clip_tables(md, n):
    """Every markdown table cut to its header and first n rows (a note gives how many were cut)."""
    out, rows = [], -2
    for line in md + [""]:
        if line.startswith("|"):
            rows += 1
            if rows <= n:
                out.append(line)
            continue
        if rows > n:
            out.append(f"({rows - n} more rows in the json)")
        rows = -2
        out.append(line)
    return out[:-1]


def choose_sizes(ex):
    """Sizes k of the exception set (the first k greedy exceptions; 0: none), each judged on val:
      rises       the largest k with the val precision rising at every step up to k (the search principle);
      best_pxr    the k with the most val precision x val onset recall;
      gain90      the smallest k reaching 90% of the largest val precision gain;
      onsets90    the largest k keeping >= 90% of the rule's val onsets.
    Returns {name: {"k", "val_precision", "val_onset_recall", "train_precision", "train_onset_recall"}}."""
    pts = [ex["start"]] + [{"train": st["train"], "val": st["val"]} for st in ex["steps"]]
    vp = [x["val"]["precision"] for x in pts]
    vr = [x["val"]["onset_recall"] for x in pts]
    k_rise = 0
    while k_rise + 1 < len(pts) and vp[k_rise + 1] > vp[k_rise]:
        k_rise += 1
    k_pxr = max(range(len(pts)), key=lambda k: vp[k] * vr[k])
    top = max(vp)
    k_gain = next(k for k in range(len(pts)) if vp[k] >= vp[0] + 0.9 * (top - vp[0]) - 1e-12)
    k_ons = max(k for k in range(len(pts)) if vr[k] >= 0.9 * vr[0] - 1e-12)
    at = lambda k: {"k": k, "val_precision": vp[k], "val_onset_recall": vr[k],
                    "train_precision": pts[k]["train"]["precision"], "train_onset_recall": pts[k]["train"]["onset_recall"]}
    return {"rises": at(k_rise), "best_pxr": at(k_pxr), "gain90": at(k_gain), "onsets90": at(k_ons)}


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--pt_head", type=str, default="Car:Jun")
    parser.add_argument("--pt_train_videos", type=int, default=1000, help="train videos (the first; 0: all)")
    parser.add_argument("--pt_val_videos", type=int, default=500, help="val videos (the first; 0: all)")
    parser.add_argument("--pt_min_onsets1", type=int, default=2, help="train onsets of a length-1 seed")
    parser.add_argument("--pt_min_onsets", type=int, default=5, help="train onsets of a kept rule of length >= 2")
    parser.add_argument("--pt_val_ratio", type=float, default=0.5,
                        help="a kept rule of length >= 2 is extended only if val precision >= this x train precision")
    parser.add_argument("--pt_block", type=int, default=2000, help="seed pairs per block of length 3")
    parser.add_argument("--pt_cover_sort", choices=["train", "val"], default="train",
                        help="coverage: the rules sorted by their train (or val) precision x recall")
    parser.add_argument("--pt_max_len", type=int, default=3, choices=[2, 3], help="2: no length-3 rules")
    parser.add_argument("--pt_except", type=int, default=20, help="exception sets of this many high-recall rules")
    parser.add_argument("--pt_except_recall", type=float, default=0.2, help="high recall: train row recall >= this")
    parser.add_argument("--pt_except_steps", type=int, default=300, help="exception literals per rule (at most; the search runs until no literal passes)")
    parser.add_argument("--pt_except_share", type=float, default=0.002,
                        help="an exception removes >= this share of the rule's train negatives")
    parser.add_argument("--pt_except_ratio", type=float, default=0.25,
                        help="an exception: P(head | rule & exception) <= this x P(rule), on train and on val")
    parser.add_argument("--pt_dup_jaccard", type=float, default=0.99,
                        help="kept pairs with rows this similar (Jaccard, train and val) to a literal or a better pair "
                             "are merged into it (1.01: off)")
    parser.add_argument("--pt_md_rows", type=int, default=5, help="rows per table in the .md (the json has all)")
    parser.add_argument("--pt_top", type=int, default=30)
    parser.add_argument("--pt_json", type=int, default=20000, help="rules per length in the json (most precise)")
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
    print(f"[Pair table] {args.pt_head}: {len(cols)} of {L} literals allowed")

    head_mask = training_data["target_heads"].to(device)
    possible = step05.class_possible(training_data, vocab, device)
    S, w = model.slot_size, int(cfg.get("clause_slots", 1))
    tr_ids = sorted(training_data["train_videos"])[:args.pt_train_videos or None]
    va_ids = sorted(training_data["val_videos"])[:args.pt_val_videos or None]
    gen = lambda ids, d: head_rows(model, training_data, ids, h, possible, head_mask, S, w, device, d)
    tr_raw = collect(gen(tr_ids, "train"), model, cols)
    va_raw = collect(gen(va_ids, "val"), model, cols)

    # ---- Length 1: every allowed literal; the onset thresholds; the seeds, identical ones merged
    n1_all = tr_raw["F"].sum(0)
    o1_all = onsets1(tr_raw["F"][tr_raw["y"]].to(device), tr_raw["ev"].to(device), tr_raw["E"]).cpu()
    present = n1_all > 0
    drop_table = [{"drop_onsets_le": k, "remaining": int((present & (o1_all > k)).sum())} for k in range(0, 6)]
    print(f"[Pair table] length 1: literals true on a train row {int(present.sum())}; "
          + ", ".join(f"onsets > {x['drop_onsets_le']}: {x['remaining']}" for x in drop_table))
    cand0 = torch.nonzero(present & (o1_all >= args.pt_min_onsets1)).flatten()
    rep_idx, members = merge_identical(tr_raw["F"][:, cand0], va_raw["F"][:, cand0], device)
    cand = cand0[rep_idx]                                                   # indices into cols
    n_c = len(cand)
    lit_text = lambda i: step05_mine.literal_text(model, names, int(cols[i]))
    ctexts = [lit_text(int(c)) for c in cand.tolist()]
    alias = [[lit_text(int(cand0[m])) for m in grp[1:]] for grp in members]
    ctext = lambda c: ctexts[c] + ((f" (= {', '.join(alias[c][:3])}" +
                                    (f", ... {len(alias[c])} in all)" if len(alias[c]) > 3 else ")")) if alias[c] else "")
    own = torch.tensor([any(model.literal_info(int(cols[cand0[m]]))["scope"] == 0 for m in grp) for grp in members],
                       dtype=torch.bool, device=device)
    print(f"[Pair table] seeds: {len(cand0)} literals with >= {args.pt_min_onsets1} train onsets, "
          f"{n_c} after merging identical ones ({int(own.sum())} own)")

    tr = split_tables(tr_raw, cand, device)
    va = split_tables(va_raw, cand, device)
    del tr_raw, va_raw
    T_hit, V_hit = max(tr["total_hit"], 1), max(va["total_hit"], 1)
    base_p, v_base = tr["total_hit"] / max(tr["total"], 1), va["total_hit"] / max(va["total"], 1)
    pc, vp1 = tr["h1"] / tr["n1"].clamp(min=1), va["h1"] / va["n1"].clamp(min=1)

    # ---- Length 2: every pair of seeds
    p2, vp2 = tr["H2"] / tr["N2"].clamp(min=1), va["H2"] / va["N2"].clamp(min=1)
    upper = torch.triu(torch.ones((n_c, n_c), dtype=torch.bool, device=device), diagonal=1)
    has = upper & (tr["N2"] > 0)
    with_own = has & (own[:, None] | own[None, :])
    enough = with_own & (tr["o2"] >= args.pt_min_onsets)
    better_tr = enough & (p2 > torch.maximum(pc[:, None], pc[None, :]))
    kept2_all = better_tr & (va["N2"] > 0) & (vp2 > torch.maximum(vp1[:, None], vp1[None, :]))
    # near-identical kept pairs: merged into a literal or the pair with the most train PxR
    dup_rep, (da, db) = near_duplicates(tr, va, kept2_all, p2[kept2_all] * tr["H2"][kept2_all], args.pt_dup_jaccard)
    kept2 = kept2_all.clone()
    gone = dup_rep != -1
    kept2[da[gone.to(device)], db[gone.to(device)]] = False
    same_rows = torch.zeros((n_c, n_c), dtype=torch.long, device=device)    # pairs merged into each kept pair
    into_pair = dup_rep >= 0
    if into_pair.any():
        tgt = dup_rep[into_pair].to(device)
        same_rows.index_put_((da[tgt], db[tgt]), torch.ones_like(tgt), accumulate=True)
    merged_into_lit = int((dup_rep < -1).sum())
    print(f"[Pair table] length 2: {int(kept2_all.sum())} kept pairs, {int(gone.sum())} near-identical merged "
          f"({merged_into_lit} into a literal), {int(kept2.sum())} left")
    seed = kept2 & (vp2 >= args.pt_val_ratio * p2)

    # ---- Length 3: every seed pair x every seed literal, in blocks
    sa, sb = torch.nonzero(seed, as_tuple=True)
    n_seeds = len(sa)
    sid = torch.full((n_c, n_c), -1, dtype=torch.long, device=device)
    sid[sa, sb] = torch.arange(n_seeds, device=device)
    sid[sb, sa] = torch.arange(n_seeds, device=device)

    score_p = lambda x: x[:, 4] / x[:, 3].clamp(min=1)
    tops = {"precision": Top(score_p, max(args.pt_json, args.pt_top)),
            "pr": Top(lambda x: score_p(x) * x[:, 4], args.pt_top), "onsets": Top(lambda x: x[:, 5], args.pt_top)}
    print(f"[Pair table] length 3: {n_seeds} seed pairs x {n_c} literals, blocks of {args.pt_block}")
    if args.pt_max_len >= 3:
        n3, b3, kept3 = length3(tr, va, sa, sb, sid, args.pt_min_onsets, args.pt_val_ratio, args.pt_block, tops)
    else:
        n3, b3 = {"tested": 0, "better_train": 0, "kept": 0, "extend": 0}, [[0, 0] for _ in EDGES[:-1]]
        kept3 = torch.zeros((0, 7))

    # ---- Coverage of val: every kept rule of each length (length 1: the seeds) sorted by precision x recall;
    # how many rules (in that order) cover the most val onsets
    ia, ib = torch.nonzero(kept2, as_tuple=True)
    pad = lambda x: torch.full_like(x, n_c)
    ar1 = torch.arange(n_c, device=device)
    sets = {
        "length 1": (torch.stack([ar1, pad(ar1), pad(ar1)], 1), pc * tr["h1"] / T_hit, vp1 * va["h1"] / V_hit),
        "length 2": (torch.stack([ia, ib, pad(ia)], 1), p2[ia, ib] * tr["H2"][ia, ib] / T_hit,
                     vp2[ia, ib] * va["H2"][ia, ib] / V_hit),
        "length 3": (kept3[:, :3].long().to(device), (kept3[:, 4] ** 2 / kept3[:, 3].clamp(min=1) / T_hit).to(device),
                     (kept3[:, 6] ** 2 / kept3[:, 5].clamp(min=1) / V_hit).to(device))}
    sets["all"] = tuple(torch.cat([v[i] for v in sets.values()]) for i in range(3))
    cover = {}
    for name, (rules, s_tr, s_va) in sets.items():
        sc = s_tr if args.pt_cover_sort == "train" else s_va
        order = torch.argsort(sc, descending=True)
        cover[name] = coverage_report(coverage(va, rules, order), sc[order].cpu(), va["E"], V_hit)
        c = cover[name]
        print(f"[Pair table] coverage {name}: {c['rules_total']} rules cover {c['max_onsets']} of {va['E']} val "
              f"onsets; " + ", ".join(f"{x['frac_of_max']:g}: {x['rules']} rules" for x in c["checks"]))

    # ---- Exception sets of the high-recall rules (every kept rule of each length, length 1: the seeds; train row
    # recall >= --pt_except_recall; the most train PxR first; rules on the same rows counted once)
    pool = torch.cat([sets["length 1"][0], sets["length 2"][0], sets["length 3"][0]])
    cnt = torch.cat([torch.stack([tr["n1"], tr["h1"], va["n1"], va["h1"]], 1),
                     torch.stack([tr["N2"][ia, ib], tr["H2"][ia, ib], va["N2"][ia, ib], va["H2"][ia, ib]], 1),
                     kept3[:, 3:7].to(device)])
    rec = cnt[:, 1] / T_hit
    hi = torch.nonzero(rec >= args.pt_except_recall).flatten()
    hi = hi[torch.argsort((cnt[hi, 1] / cnt[hi, 0].clamp(min=1) * rec[hi]), descending=True)]
    seen, picked = set(), []
    for i in hi.tolist():
        key = tuple(int(v) for v in cnt[i].tolist())
        if key not in seen:
            seen.add(key)
            picked.append(i)
        if len(picked) >= args.pt_except:
            break
    rule_text = lambda r: " & ".join(ctext(c) for c in r if c < n_c)
    H = int(max(model.windows))                                             # `first` is capped at H
    all_rows = lambda t: torch.ones(len(t["F"]), dtype=torch.bool, device=device)
    fp_base = {"train": fp_classes(tr, all_rows(tr), H, S), "val": fp_classes(va, all_rows(va), H, S)}
    excepts = []
    print(f"[Pair table] exceptions: {len(hi)} rules with train recall >= {args.pt_except_recall}, "
          f"{len(picked)} distinct ones used")
    for i in picked:
        r = [int(c) for c in pool[i].tolist()]
        ex = exceptions(tr, va, r, args.pt_except_steps, args.pt_except_share, args.pt_except_ratio, args.pt_val_ratio)
        ex["rule"] = rule_text(r)
        ex["sizes"] = choose_sizes(ex)
        ex["fp"] = {"train": fp_classes(tr, rule_mask(tr, r), H, S), "val": fp_classes(va, rule_mask(va, r), H, S)}
        for st in ex["steps"]:
            st["unless"] = ctext(st["literal"])
        excepts.append(ex)
        a, z = ex["start"], (ex["steps"][-1] if ex["steps"] else ex["start"])
        print(f"[Pair table]   {ex['rule']}: {len(ex['steps'])} exceptions; val P {a['val']['precision']:.3f} -> "
              f"{z['val']['precision']:.3f}, val onset recall {a['val']['onset_recall']:.3f} -> "
              f"{z['val']['onset_recall']:.3f}; sizes " + ", ".join(f"{n} {v['k']}" for n, v in ex["sizes"].items()))

    # ---- Reports
    stats = {"literals_allowed": len(cols), "literals_true": int(present.sum()), "drop_onsets": drop_table,
             "seeds_literals": len(cand0), "seeds_merged": n_c, "seeds_own": int(own.sum()),
             "pairs_tested": n_c * (n_c - 1) // 2, "pairs_with_rows": int(has.sum()),
             "pairs_with_own": int(with_own.sum()), "pairs_min_onsets": int(enough.sum()),
             "pairs_better_train": int(better_tr.sum()), "pairs_kept_before_merge": int(kept2_all.sum()),
             "pairs_merged_into_literal": merged_into_lit, "pairs_merged_into_pair": int(into_pair.sum()),
             "pairs_kept": int(kept2.sum()), "pairs_extend": n_seeds,
             "triples_tested": n3["tested"], "triples_better_train": n3["better_train"],
             "triples_kept": n3["kept"], "triples_extend": n3["extend"], "cover_sort": args.pt_cover_sort}
    print(f"[Pair table] {stats}")

    def metrics(prec, hit, ons, rows, vprec, vhit, vons, vrows):
        r, vr = hit / T_hit, vhit / V_hit
        return {"train_precision": prec, "train_recall": r, "train_pr": prec * r, "train_onsets": int(ons),
                "train_rows": int(rows), "val_precision": vprec, "val_recall": vr, "val_pr": vprec * vr,
                "val_onsets": int(vons), "val_rows": int(vrows)}

    def single(c):
        return {"rule": ctext(c), **metrics(float(pc[c]), float(tr["h1"][c]), tr["o1"][c], tr["n1"][c],
                                            float(vp1[c]), float(va["h1"][c]), va["o1"][c], va["n1"][c])}

    def pair(a, b):
        a, b = int(a), int(b)
        n_same = int(same_rows[a, b])
        return {"rule": f"{ctext(a)} & {ctext(b)}" + (f" [+{n_same} near-identical pairs]" if n_same else ""),
                "near_identical": n_same,
                **metrics(float(p2[a, b]), float(tr["H2"][a, b]), tr["o2"][a, b], tr["N2"][a, b],
                          float(vp2[a, b]), float(va["H2"][a, b]), va["o2"][a, b], va["N2"][a, b]),
                "parents_train": [float(pc[a]), float(pc[b])], "parents_val": [float(vp1[a]), float(vp1[b])],
                "kept": bool(kept2[a, b]), "extend": bool(seed[a, b])}

    def triple(x):
        a, b, c = (int(v) for v in x[:3])
        n, hh, o, vn, vh, vo, e = x[3:].tolist()
        return {"rule": f"{ctext(a)} & {ctext(b)} & {ctext(c)}",
                **metrics(hh / max(n, 1), hh, o, n, vh / max(vn, 1), vh, vo, vn),
                "seed_train": float(p2[a, b]), "seed_val": float(vp2[a, b]), "extend": bool(e)}

    singles = [single(c) for c in range(n_c)]
    pairs = {}
    for name, score in (("precision", p2[ia, ib]), ("pr", p2[ia, ib] * tr["H2"][ia, ib]), ("onsets", tr["o2"][ia, ib])):
        n = max(args.pt_json, args.pt_top) if name == "precision" else args.pt_top
        pairs[name] = [pair(ia[j], ib[j]) for j in torch.argsort(score, descending=True)[:n].tolist()]
    b2 = bucket_counts(p2[ia, ib], seed[ia, ib])
    trip = {name: [triple(x) for x in t.rows] for name, t in tops.items()}

    idx = {}
    for c in range(n_c):
        for t in [ctexts[c]] + alias[c]:
            idx[t] = c
    shown = []
    for spec in (s for s in args.pt_show.split(";") if s.strip()):
        a, _, b = (x.strip() for x in spec.partition("&"))
        if a in idx and b in idx:
            shown.append(pair(*sorted((idx[a], idx[b]))))
        else:
            shown.append({"rule": spec, "missing": [t for t in (a, b) if t not in idx]})

    top1 = lambda k: sorted(singles, key=lambda x: -x[k])[:args.pt_top]
    cells = lambda x: (f"{x['train_precision']:.3f} | {x['train_recall']:.3f} | {x['train_pr']:.4f} | "
                       f"{x['train_onsets']} | {x['val_precision']:.3f} | {x['val_recall']:.3f} | {x['val_pr']:.4f} | "
                       f"{x['val_onsets']}")
    head = "train P | train R | train PxR | onsets | val P | val R | val PxR | val onsets"
    sep = lambda n: "|" + "---|" * n
    yn = lambda v: "yes" if v else "no"
    row1 = lambda x: f"| {cells(x)} | {x['rule']} |"
    row2 = lambda x: (f"| {cells(x)} | {x['parents_train'][0]:.3f}, {x['parents_train'][1]:.3f} | "
                      f"{x['parents_val'][0]:.3f}, {x['parents_val'][1]:.3f} | {yn(x['extend'])} | {x['rule']} |")
    row3 = lambda x: f"| {cells(x)} | {x['seed_train']:.3f} | {x['seed_val']:.3f} | {yn(x['extend'])} | {x['rule']} |"
    hdr1 = [f"| {head} | rule |", sep(9)]
    hdr2 = [f"| {head} | literals train P | literals val P | extend | rule |", sep(12)]
    hdr3 = [f"| {head} | seed train P | seed val P | extend | rule |", sep(12)]
    labels = [f"[{lo:g}, {min(hi, 1):g}{')' if hi < 1 else ']'}" for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    bucket_md = lambda bs: (["| train precision | kept | extend |", sep(3)] +
                            [f"| {lab} | {k} | {e} |" for lab, (k, e) in zip(labels, bs)])
    md = [f"# Length-1, -2 and -3 rules of {args.pt_head}", "",
          f"Train: {len(tr_ids)} videos, {tr['total']} rows, {tr['total_hit']} hit rows, {tr['E']} onsets, "
          f"base precision {base_p:.4f}. Val: {len(va_ids)} videos, {va['total']} rows, {va['total_hit']} hit rows, "
          f"{va['E']} onsets, base precision {v_base:.4f}.",
          "P precision, R recall (hit rows / all hit rows), PxR their product. Kept (length >= 2): an (own) literal, "
          f">= {args.pt_min_onsets} train onsets, P above every sub-rule on train and on val. "
          f"Extend: val P >= {args.pt_val_ratio:g} x train P.", "",
          "## Length 1: literals left if those with <= k train onsets are dropped", "",
          "| dropped | remaining |", sep(2)]
    md += [f"| onsets <= {x['drop_onsets_le']} | {x['remaining']} |" for x in drop_table]
    md += ["", f"- literals allowed {len(cols)}, true on a train row {int(present.sum())}; seeds (>= "
               f"{args.pt_min_onsets1} onsets) {len(cand0)}, {n_c} after merging identical ones ({int(own.sum())} own)",
           f"- pairs: tested {stats['pairs_tested']}, with rows {stats['pairs_with_rows']}, with an own literal "
           f"{stats['pairs_with_own']}, >= {args.pt_min_onsets} onsets {stats['pairs_min_onsets']}, above both "
           f"literals on train {stats['pairs_better_train']}, also on val {stats['pairs_kept_before_merge']}, "
           f"near-identical (Jaccard >= {args.pt_dup_jaccard:g} on train and val) merged into a literal "
           f"{stats['pairs_merged_into_literal']} / into a better pair {stats['pairs_merged_into_pair']}, kept "
           f"{stats['pairs_kept']}, "
           f"extend {stats['pairs_extend']}",
           f"- triples: tested {stats['triples_tested']}, above every sub-rule on train "
           f"{stats['triples_better_train']}, also on val (kept) {stats['triples_kept']}, extend "
           f"{stats['triples_extend']}", "", "## Kept pairs by train precision", ""]
    md += bucket_md(b2) + ["", "## Kept triples by train precision", ""] + bucket_md(b3)
    for k, what in (("train_precision", "train precision"), ("train_pr", "train PxR")):
        md += ["", f"## Length 1: top {args.pt_top} by {what}", ""] + hdr1 + [row1(x) for x in top1(k)]
    for k, what in (("precision", "train precision"), ("pr", "train PxR"), ("onsets", "onsets")):
        md += ["", f"## Length 2: top {args.pt_top} kept pairs by {what}", ""] + hdr2
        md += [row2(x) for x in pairs[k][:args.pt_top]]
    for k, what in (("precision", "train precision"), ("pr", "train PxR"), ("onsets", "onsets")):
        md += ["", f"## Length 3: top {args.pt_top} kept triples by {what}", ""] + hdr3
        md += [row3(x) for x in trip[k][:args.pt_top]]
    cov_row = lambda x: (f"| {x['rules']} | {x['onsets']} | {x['onset_recall']:.3f} | {x['row_recall']:.3f} | "
                         f"{x['precision']:.3f} | {x['score_of_last']:.4f} |")
    cov_hdr = ["| rules | val onsets | onset recall | row recall | precision of the union | PxR of the last |", sep(6)]
    md += ["", f"## Val coverage: the kept rules sorted by {args.pt_cover_sort} PxR", "",
           f"A row is covered if at least one of the first k rules holds; precision of the union = covered hit rows / "
           f"covered rows (predict the head when any rule holds). Val: {va['E']} onsets. Length 1: the seeds.", ""]
    for name, c in cover.items():
        md += [f"### {name}: {c['rules_total']} rules, at most {c['max_onsets']} val onsets", "",
               "Rules needed for a fraction of the most onsets:", ""]
        md += ["| fraction " + cov_hdr[0], "|---" + cov_hdr[1]]
        md += [f"| {x['frac_of_max']:g} " + cov_row(x) for x in c["checks"] if "onsets" in x]
        md += ["", "Curve:", ""] + cov_hdr + [cov_row(x) for x in c["curve"]] + [""]
    fp_names = ("early", "later", "after", "never")
    fp_cells = lambda f: " | ".join(f"{f[n]['share']:.3f} ({f[n]['agents']})" for n in fp_names)
    md += ["", "## Negative rows of the high-recall rules by what the agent does", "",
           f"Negative row: the rule holds, no onset within the hit window. early: the agent's onset comes later "
           f"within {H} frames (the rule fires too early); later: an onset more than {H} frames later; after: the "
           f"agent's onsets are all before; never: the agent has no onset. Share of the negative rows (distinct "
           f"agents). all rows: every at-risk row.", "",
           "| rule | split | negatives | early | later | after | never | early rows by slots ahead |", sep(8)]
    for name, f2 in [("all rows", fp_base)] + [(ex["rule"], ex["fp"]) for ex in excepts]:
        for sp in ("train", "val"):
            f = f2[sp]
            md.append(f"| {name if sp == 'train' else ''} | {sp} | {f['negatives']} | {fp_cells(f)} | "
                      f"{', '.join(f'{k}: {v}' for k, v in f['early_by_slots_ahead'].items())} |")
    md += ["", f"## Exception sets of the high-recall rules (train recall >= {args.pt_except_recall})", "",
           f"Exception E of rule R: holds where R holds and the head not. Greedy: the E removing the largest share of "
           f"R's train negatives, with P(head | R & E) <= {args.pt_except_ratio:g} x P(R) on train and val, >= "
           f"{args.pt_except_share:g} of R's train negatives and a val share >= {args.pt_val_ratio:g} x the train "
           f"share; then R := R & !E. Share: of R's original negatives. Lost: onsets of R removed by E.", "",
           "| rule | exceptions | train P | train onset R | val P | val onset R | val negatives removed |", sep(7)]
    for ex in excepts:
        a, z = ex["start"], (ex["steps"][-1] if ex["steps"] else ex["start"])
        md.append(f"| {ex['rule']} | {len(ex['steps'])} | {a['train']['precision']:.3f} -> {z['train']['precision']:.3f}"
                  f" | {a['train']['onset_recall']:.3f} -> {z['train']['onset_recall']:.3f} | "
                  f"{a['val']['precision']:.3f} -> {z['val']['precision']:.3f} | {a['val']['onset_recall']:.3f} -> "
                  f"{z['val']['onset_recall']:.3f} | "
                  f"{1 - z['val']['negatives'] / max(a['val']['negatives'], 1):.3f} |")
    cell = lambda v: f"{v['k']} ({v['val_precision']:.3f} / {v['val_onset_recall']:.3f})"
    md += ["", "### Exception set sizes (k exceptions; in brackets val P / val onset recall after them)", "",
           "rises: val P rises at every step up to k; best_pxr: the most val P x onset recall; gain90: 90% of the "
           "largest val P gain; onsets90: the most exceptions keeping >= 90% of the val onsets. all: until no "
           "literal passes.", "",
           "| rule | start (val P / onset R) | all | rises | best_pxr | gain90 | onsets90 |", sep(7)]
    for ex in excepts:
        a, z = ex["start"]["val"], ex["sizes"]
        md.append(f"| {ex['rule']} | {a['precision']:.3f} / {a['onset_recall']:.3f} | {len(ex['steps'])} | "
                  + " | ".join(cell(z[n]) for n in ("rises", "best_pxr", "gain90", "onsets90")) + " |")
    if excepts:
        import statistics
        for n in ("rises", "best_pxr", "gain90", "onsets90"):
            ks = [ex["sizes"][n]["k"] for ex in excepts]
            md.append(f"- {n}: k median {statistics.median(ks):g}, max {max(ks)}")
        alls = [len(ex["steps"]) for ex in excepts]
        md.append(f"- all: k median {statistics.median(alls):g}, max {max(alls)}")
    for ex in excepts[:args.pt_md_rows]:                                   # the step tables of the first rules
        a = ex["start"]
        md += ["", f"### {ex['rule']}", "",
               f"Start: train P {a['train']['precision']:.3f}, {a['train']['onsets']} onsets, {a['train']['negatives']} "
               f"negative rows; val P {a['val']['precision']:.3f}, {a['val']['onsets']} onsets, "
               f"{a['val']['negatives']} negative rows. First step: {ex['first_step_failed_val']} train candidates "
               f"failed on val.", "",
               "| step | unless | share train | share val | P with train | P with val | lost train | lost val | "
               "after: train P | train onsets | val P | val onsets |", sep(12)]
        shown_k = set(range(1, args.pt_top + 1)) | {v["k"] for v in ex["sizes"].values()} | \
            {k for k in (50, 100, 200, 300, 500) if k <= len(ex["steps"])} | {len(ex["steps"])}
        for k, st in enumerate(ex["steps"], 1):
            if k not in shown_k:
                continue
            md.append(f"| {k} | {st['unless']} | {st['share']:.3f} | {st['val_share']:.3f} | "
                      f"{st['precision_with']:.3f} | {st['val_precision_with']:.3f} | {st['onsets_lost']} | "
                      f"{st['val_onsets_lost']} | {st['train']['precision']:.3f} | {st['train']['onsets']} | "
                      f"{st['val']['precision']:.3f} | {st['val']['onsets']} |")
    md += ["", "## Pairs asked for (--pt_show)", ""] + [f"- {x}" for x in shown]
    stem = lab_dir / f"pairs_{args.pt_head.replace(':', '_')}"
    md = clip_tables(md, args.pt_md_rows)
    open(f"{stem}.md", "w").write("\n".join(md) + "\n")
    utils_data.save_json({"head": args.pt_head, "train_videos": len(tr_ids), "val_videos": len(va_ids),
                          "train": {"rows": tr["total"], "hit_rows": tr["total_hit"], "onsets": tr["E"]},
                          "val": {"rows": va["total"], "hit_rows": va["total_hit"], "onsets": va["E"]},
                          "stats": stats, "bucket_labels": labels, "buckets_pairs": b2, "buckets_triples": b3,
                          "singles": singles, "pairs": pairs["precision"], "triples": trip["precision"],
                          "coverage": cover, "exceptions": excepts, "fp_all_rows": fp_base, "shown": shown}, f"{stem}.json")
    print("\n".join(md[:45]))
    print(f"\n[Pair table] {stem}.md / .json")


if __name__ == "__main__":
    main()
