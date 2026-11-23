"""Rule lab: short tests of the mined-rule model ("mined rules", step 06's "anchor") against the OnsetMLP,
per head, on val, without training the rule net (2026-11-22).

Goal: most heads close to or above the MLP with rules alone. Run 1ca554 (SUMO v7): the mined rules matched
the MLP's recall at 5 FA/min (0.109 vs 0.107), but on 13 heads only; macro AP 0.173 vs 0.267.

Principle (the user, 2026-11-22): a rule that fired before only two onsets and was right both times counts,
whatever its length; every precise rule, and every rule that may be extended into a precise one, is
considered.

Rule sets ("variants"), all mined on the train sample of `step05_mine.collect` (cached by `mine_sweep`):
  m4       the step 05 mining (`step05_mine.mine_head`, the run's settings);
  relaxed  the same miner with the limits opened: 32 rules per root, 4 roots, depth 6, 2 onsets in 1 video,
           no val filter;
  precise  beam search for precise conjunctions (`precise_rules`): any literal (no own root needed), any
           length up to --lab_max_len. A rule is kept if its precision is >= --lab_precise_min with
           >= --lab_min_events distinct onsets. A rule is extended while its precision can still rise;
  a+b      the union of miners a and b (e.g. relaxed+precise);
  ..._d    the same with the derived literals (`Derived`: X started / stopped / held, own-box motion).
Lab 2 (2026-11-22): the derived literals gave nothing (relaxed_d -0.011, relaxed+precise_d +0.000, lane heads
worse): dropped from the defaults. relaxed+precise / soft_logit 0.255 vs MLP 0.277 (88 heads); the precise rules
helped the lane heads (IncomLane 0.25 -> 0.40, MovRht 0.02 -> 0.18). Lab 3: sparse gates (SPARSE) for readable
rules; the plain soft gates relaxed only 2.6% of the literals, so the rules kept noise negations.
Model since 2026-11-22 (the user): relaxed / soft_logit (1ca554 lab: macro AP 0.245 vs MLP 0.283 on 86 heads,
46 heads >= 0.9 x MLP; the m4 mined rules: 0.159, 13). Default variants: relaxed, relaxed+precise and both
with the derived literals; default scorers max_k0, nor_k0, logit, soft_logit.
Every rule is then recounted exactly on all train rows (`recount`). The sample's negatives carry weights of
~100; a rule with no sampled negative can still fire on many unsampled rows. A rule whose extension does
not raise the exact precision over its sub-rule is dropped.

Scorers of a rule set, per head:
  max_k{k}  the highest confidence of the holding rules, else the head's base rate. The confidence is the exact
            train precision shrunk to the base rate with k onsets: (e + k b) / (e / p + k). k = 0 is the raw
            precision.
  nor_k{k}  noisy-OR of the same confidences: rules that hold together add up.
  logit     per-head logistic regression on the rules that hold (fit on the train sample, weighted); a
            readable weight per rule.
  soft_logit / soft_nor  soft AND and soft OR (`fit_soft`, the user's request): every literal of a rule has a gate
            g in [0, 1] (start ~1, the hard rule). A rule's activation is prod (1 - g (1 - x)), so a rule with a
            literal missing still counts in part. The rules are combined by a logistic sum or a noisy-OR.
            The share of the literals relaxed (g < 0.5) is printed per variant.

Evaluation: every at-risk row of the odd val videos, not a sample. The even ones are the m4 miner's val check.
  - Label: the head switches on within clause_slots slots, as in mining.
  - MLP: its window of clause_slots x slot_size frames, on the same rows.
  - Per head: the AP of every (variant, scorer) and of the MLP, from score histograms.
  - Reported:
    - macro AP over the learnable heads (some scorer, MLP included, reaches LEARNABLE_MIN_AP);
    - the number of heads with >= 0.9 x the MLP's AP and with more than the MLP's AP;
    - micro AP;
    - rules and literals per rule;
    - per action, for the best scorer.
  The test split is never read.

Run (server, with the run's MLP and training data):
  ./run_roadpp_docker.sh --gpu 0 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing \\
      --run_id 20261121-173934_dgx-jing_1ca554
Writes <output_dir>/rule_lab/rule_lab_<tag>.json.
"""
import math
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from src.exp_roadpp import config, mine_sweep, step05, step05_mine, step07_paper, utils_data, utils_pipe

LEARNABLE_MIN_AP = 0.05
CLOSE = 0.9                     # "close to the MLP": AP >= CLOSE x the MLP's
MIN_EVAL_POS = 20               # heads with fewer positive eval rows are not compared
RULE_CHUNK = 4096               # rules per gather in `_holds_all`
SOFT_PRIOR = 5                  # confidence shrink (onsets) of the soft noisy-OR's starting q
# Sparse soft rules (2026-11-22, readability): the gates and the rule weights are pushed toward 0, so a rule
# keeps only the literals and the heads keep only the rules the data pays for. lambda x (the head's base loss)
# x (mean gate + mean |weight|) is added to the loss. A literal with gate < DROP_GATE and a rule with
# |weight| < DROP_WEIGHT count as dropped.
SPARSE = {"sparse_a": 0.001, "sparse_b": 0.005, "sparse_c": 0.02}   # fake-data check: 0.5+ drops every rule
DROP_GATE, DROP_WEIGHT = 0.1, 0.05
RELAXED = {"mine_max_rules": 32, "mine_roots": 4, "mine_root_min_cover": 0.05, "mine_depth": 6,
           "mine_min_events": 2, "min_videos": 1, "mine_val_lift": 0.0}


# -------------- Rules on literal rows --------------
def _pad(rule_lits, true_lit):
    """(R, K) long literal indices, padded with a literal that always holds."""
    K = max((len(x) for x in rule_lits), default=1)
    return torch.tensor([list(x) + [true_lit] * (K - len(x)) for x in rule_lits], dtype=torch.long)


def _holds_all(lits, body):
    """(B, R) bool: every literal of each rule holds. lits (B, L + 1) bool (last column: always true),
    body (R, K) long."""
    out = []
    for i in range(0, len(body), RULE_CHUNK):
        out.append(lits[:, body[i:i + RULE_CHUNK]].all(dim=2))
    return torch.cat(out, dim=1) if out else torch.zeros((len(lits), 0), dtype=torch.bool, device=lits.device)


def _with_true(lits):
    return torch.cat([lits, torch.ones((len(lits), 1), dtype=torch.bool, device=lits.device)], dim=1)


# -------------- Derived literals (second batch of facts) --------------
class Derived:
    """Literals derived from the rule net's literals of a row (2026-11-22). They are not stored in the data;
    they are computed from the literal bits.
      change  per positive input X of a single slot (any scope):
                "X started" = X@1 & !X@2;
                "X stopped" = X@2 & !X@1;
                "X held"    = X@1 & X@2 (& X@3).
              Each is one literal, so the miner does not need two tree levels to find an onset.
      motion  per corner (TL / BR) of the agent's own box: its mean grid row / column between slot 1 and slot 2
              (and slot 3) moved up / down / left / right by > MOVE cells. The box (BR - TL) grew / shrank
              in height / width. The grid is 3x3, so only coarse moves show; the lane and direction heads
              (VehLane, IncomLane, MovRht) are where the rules trail the MLP most.
    A derived literal is allowed if all its parts are, and it has the base input of its first part, so the
    head's own atom stays out of its rules."""
    MOVE = 0.4

    def __init__(self, model, names):
        L, M, A = model.num_literals, model.num_inputs, model.num_atoms
        self.L = L
        mask = model.input_mask.cpu()
        base = model.literal_base.cpu()

        def lit(kind, a, sc, j):
            try:
                m = model.literal_of(kind, a, sc, j)
            except (ValueError, KeyError, IndexError):
                return None
            return m if m < L else None
        conj, text, bases, allowed = [], [], [], []
        for m in range(M):
            a, sc, j = model.decode_input(m)
            if model.spans[(m // A) % model.num_spans] != (1, 1):
                continue
            x2, x3 = lit("X", a, sc, 2), lit("X", a, sc, 3)
            if x2 is None:
                continue
            nx1, nx2 = (lit("!X", a, sc, 1), lit("!X", a, sc, 2)) if model.negation else (None, None)
            src = step05_mine.literal_text(model, names, m).rsplit("@", 1)[0]
            for kind, parts in (("started", [m, nx2]), ("stopped", [x2, nx1]),
                                ("held", [m, x2] + ([x3] if x3 is not None else []))):
                if any(x is None for x in parts):
                    continue
                conj.append(parts)
                text.append(f"{kind}:{src}")
                bases.append(int(base[m]))
                allowed.append(bool(mask[parts].all()))
        self.conj = _pad(conj, L) if conj else torch.zeros((0, 1), dtype=torch.long)
        # Motion of the own box: the position literals pos:<corner>_r<i>c<j> in the own scope, per slot.
        self.cells = {}                                  # (corner, slot) -> (literals, rows, cols)
        for corner in ("TL", "BR"):
            for j in (1, 2, 3):
                lits, rows, cols = [], [], []
                for a, nm in enumerate(names):
                    if nm.startswith(f"pos:{corner}_r"):
                        m = lit("X", a, 0, j)
                        if m is not None:
                            r, c = nm.split("_r")[1].split("c")
                            lits.append(m)
                            rows.append(float(r))
                            cols.append(float(c))
                if lits:
                    self.cells[(corner, j)] = (torch.tensor(lits), torch.tensor(rows), torch.tensor(cols))
        self.motion = []                                 # (kind, corner or None, j)
        for j in (2, 3):
            for corner in ("TL", "BR"):
                if (corner, 1) in self.cells and (corner, j) in self.cells:
                    for kind in ("up", "down", "left", "right"):
                        self.motion.append((kind, corner, j))
            if all((c, jj) in self.cells for c in ("TL", "BR") for jj in (1, j)):
                for kind in ("grow_h", "shrink_h", "grow_w", "shrink_w"):
                    self.motion.append((kind, None, j))
        pos_lit = int(self.cells[("TL", 1)][0][0]) if ("TL", 1) in self.cells else 0
        for kind, corner, j in self.motion:
            text.append(f"box_{kind}{'_' + corner if corner else ''}(own)@1vs{j}")
            bases.append(int(base[pos_lit]))
            allowed.append(bool(mask[pos_lit]))
        self.text, self.base, self.allowed = text, torch.tensor(bases, dtype=torch.long), torch.tensor(allowed)
        self.size = len(text)
        print(f"[Rule lab] derived literals: {len(conj)} change (started / stopped / held), "
              f"{len(self.motion)} own-box motion")

    def __call__(self, lits):
        """(B, D) bool derived literals of literal rows lits (B, L) bool."""
        dev = lits.device
        parts = []
        if len(self.conj):
            parts.append(_holds_all(_with_true(lits), self.conj.to(dev)))
        if self.motion:
            mean = {}
            for (corner, j), (idx, rows, cols) in self.cells.items():
                x = lits[:, idx.to(dev)].float()
                n = x.sum(dim=1)
                mean[(corner, j)] = (n > 0, (x @ rows.to(dev)) / n.clamp(min=1), (x @ cols.to(dev)) / n.clamp(min=1))
            out = []
            for kind, corner, j in self.motion:
                if corner is not None:
                    (s1, r1, c1), (s2, r2, c2) = mean[(corner, 1)], mean[(corner, j)]
                    d = {"up": r2 - r1, "down": r1 - r2, "left": c2 - c1, "right": c1 - c2}[kind]
                    out.append(s1 & s2 & (d > self.MOVE))
                else:
                    (a1, tr1, tc1), (b1, br1, bc1) = mean[("TL", 1)], mean[("BR", 1)]
                    (a2, tr2, tc2), (b2, br2, bc2) = mean[("TL", j)], mean[("BR", j)]
                    h1, h2, w1, w2 = br1 - tr1, br2 - tr2, bc1 - tc1, bc2 - tc2
                    d = {"grow_h": h1 - h2, "shrink_h": h2 - h1, "grow_w": w1 - w2, "shrink_w": w2 - w1}[kind]
                    out.append(a1 & b1 & a2 & b2 & (d > self.MOVE))
            parts.append(torch.stack(out, dim=1))
        return torch.cat(parts, dim=1) if parts else torch.zeros((len(lits), 0), dtype=torch.bool, device=dev)

    def extend(self, lits):
        """(B, L + D) bool: the literals and the derived ones."""
        return torch.cat([lits, self(lits)], dim=1)


class ExtModel:
    """The rule net seen with its derived literals appended (`Derived`): what `step05_mine.mine_head` and
    `literal_text` read. use_derived False: the derived literals are not allowed (same indices)."""

    def __init__(self, model, derived, use_derived, forbid=None):
        """forbid (L,) bool: rule net literals the miners may not use (e.g. `negated_others`)."""
        self._model, self._d = model, derived
        dev = model.input_mask.device
        self.num_literals = model.num_literals + derived.size
        self.literal_kind = torch.cat([model.literal_kind, torch.full((derived.size,), 4, device=dev,
                                                                      dtype=model.literal_kind.dtype)])
        self.literal_base = torch.cat([model.literal_base, derived.base.to(dev).to(model.literal_base.dtype)])
        own = model.input_mask if forbid is None else model.input_mask & ~forbid.to(dev)
        self.input_mask = torch.cat([own, derived.allowed.to(dev) & bool(use_derived)])

    def __getattr__(self, key):
        return getattr(self._model, key)

    def text_of(self, m):
        return self._d.text[m - self._model.num_literals] if m >= self._model.num_literals else None


def extend_rows(data, derived, L, device, chunk=50000):
    """A `collect` sample with the derived literal bits appended to every row."""
    packed = []
    for i in range(0, len(data["packed"]), chunk):
        bits = step05_mine._unpack(data["packed"][i:i + chunk], L, device) > 0.5
        packed.append(step05_mine._pack(derived.extend(bits)).cpu())
    return {**data, "packed": torch.cat(packed) if packed else data["packed"]}


def literal_rows(model, inputs, derived):
    """(B, L + D + 1) bool: a row's literals, the derived ones and the always-true one."""
    return _with_true(derived.extend(model.literal_values(inputs) > 0.5))


# -------------- Precise rule search --------------
def precise_rules(pool, allowed, cfg):
    """Beam search for precise conjunctions on one head's train sample pool (`step05_mine._pool_of`).

    Every rule grows one literal at a time. A child is ranked by its pessimistic precision: weighted hits over
    weighted rows plus the weight of one more negative row, so a rule seen on few rows ranks low until it has
    support. Per node, the --lab_branch best children with >= min_events distinct onsets (in >= min_videos
    videos) whose precision rises over the parent's go on. A child is kept as a rule if its precision is
    >= precise_min, and grows further while it is below 1. The --lab_beam best children form the next level.
    Returns [(literals tuple, sample precision, onsets)], best first, at most --lab_max_rules."""
    F, y, w = pool["F"], pool["y"], pool["w"]
    n = len(y)
    if n == 0 or not bool(y.any()):
        return []
    dev = F.device
    yf = y.float()
    wy = w * yf
    neg_w = w[~y]
    lam = float(neg_w.median()) if len(neg_w) else 1.0        # weight of one more (sampled) negative row
    beam = [((), torch.ones(n, dtype=torch.bool, device=dev), 0.0)]
    kept, seen = {}, set()
    for _depth in range(cfg["lab_max_len"]):
        if not beam:
            break
        M = torch.stack([m for _, m, _ in beam]).float()        # (B, n)
        n1 = (M * w) @ F                                        # (B, L) weighted rows where the child holds
        h1 = (M * wy) @ F
        c1 = (M * yf) @ F                                       # hit rows (unweighted)
        prec = h1 / n1.clamp(min=1e-12)
        score = h1 / (n1 + lam)
        parent_rows = M.sum(dim=1, keepdim=True)
        ok = (allowed[None] & (c1 >= cfg["lab_min_events"]) & (n1 > 0)
              & (prec > torch.tensor([p for _, _, p in beam], device=dev)[:, None] + 1e-9)
              & (((M @ F) < parent_rows)))                      # the literal must cut some row
        score = torch.where(ok, score, torch.full_like(score, -1.0))
        top = score.topk(min(cfg["lab_branch"], score.shape[1]), dim=1)
        children = []
        for b, (lits, mask, _p) in enumerate(beam):
            for s, m in zip(top.values[b].tolist(), top.indices[b].tolist()):
                if s <= 0:
                    break
                key = tuple(sorted(lits + (m,)))
                if key in seen:
                    continue
                seen.add(key)
                child = mask & (F[:, m] > 0.5)
                hit = child & y
                events = int(pool["event"][hit].unique().numel())
                if events < cfg["lab_min_events"] or int(pool["video"][hit].unique().numel()) < cfg["lab_min_videos"]:
                    continue
                p = float(prec[b, m])
                if p >= cfg["lab_precise_min"]:
                    kept[key] = (p, events, s)
                if p < 1.0 - 1e-9:
                    children.append((s, key, child, p))
        children.sort(key=lambda x: -x[0])
        beam = [(key, child, p) for _s, key, child, p in children[:cfg["lab_beam"]]]
    best = sorted(kept.items(), key=lambda x: -x[1][2])[:cfg["lab_max_rules"]]
    return [(key, p, e) for key, (p, e, _s) in best]


# -------------- Exact counts on all train rows --------------
@torch.no_grad()
def recount(model, training_data, video_ids, rules, head_mask, possible, cfg, derived):
    """Exact fires (rows), hits (rows) and distinct onsets of every rule {"head", "literals"} on all at-risk
    rows of video_ids, and per head the rows and hit rows (the base rate)."""
    device, S, w = cfg["device"], model.slot_size, int(cfg.get("clause_slots", 1))
    A, L = model.num_atoms, model.num_literals + derived.size
    R = len(rules)
    body = _pad([r["literals"] for r in rules], L).to(device)
    head = torch.tensor([r["head"] for r in rules], dtype=torch.long, device=device)
    fires, hits = torch.zeros(R, device=device), torch.zeros(R, device=device)
    rows_h, pos_h = torch.zeros(A, device=device), torch.zeros(A, device=device)
    pairs = []
    for vi, video_id in enumerate(tqdm(video_ids, desc="Exact counts (train)")):
        can = possible(video_id)
        for frames in training_data["frames"][video_id].split(int(cfg.get("mine_chunk", 256))):
            rows = step05._video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            keep, y = step05_mine._head_rows(rows, can, head_mask, S, w)
            rows_h += keep.sum(dim=0)
            pos_h += y.sum(dim=0)
            sel = keep.any(dim=1)
            if not sel.any() or R == 0:
                continue
            lits = literal_rows(model, rows["inputs"][sel], derived)
            holds = _holds_all(lits, body)                                      # (B, R)
            k_r, y_r = keep[sel][:, head], y[sel][:, head]                      # (B, R)
            fire = holds & k_r
            hit = fire & y_r
            fires += fire.sum(dim=0)
            hits += hit.sum(dim=0)
            b, r = torch.nonzero(hit, as_tuple=True)
            if len(b):
                t, n = rows["rows"][sel][b].T
                onset = t + rows["first"][sel][b, head[r]]
                ev = (torch.full_like(onset, vi) << 40) | (n.long() << 20) | onset.long()
                pairs.append(torch.unique(torch.stack([r, ev], dim=1), dim=0).cpu())
    events = torch.zeros(R, dtype=torch.long)
    if pairs:
        u = torch.unique(torch.cat(pairs), dim=0)
        events.index_add_(0, u[:, 0], torch.ones(len(u), dtype=torch.long))
    return {"fires": fires.cpu(), "hits": hits.cpu(), "events": events,
            "base": (pos_h / rows_h.clamp(min=1)).cpu()}


def finalize(rules, counts, min_events):
    """The rules with exact precision and onsets: >= min_events onsets; a rule is dropped if one of its
    sub-rules (fewer literals, same head) is at least as precise (the extension adds nothing)."""
    out = []
    for i, r in enumerate(rules):
        f, h, e = float(counts["fires"][i]), float(counts["hits"][i]), int(counts["events"][i])
        if f <= 0 or e < min_events:
            continue
        out.append({**r, "precision": h / f, "events": e, "fires": f})
    by_head = defaultdict(dict)
    for r in out:
        by_head[r["head"]][frozenset(r["literals"])] = r
    keep = []
    for r in out:
        lits = frozenset(r["literals"])
        same = by_head[r["head"]]
        dominated = any(len(k) < len(lits) and k <= lits and s["precision"] >= r["precision"]
                        for k, s in same.items())
        if not dominated or r.get("root"):
            keep.append(r)
    return keep


# -------------- Merging close direction sectors --------------
def direction_literals(model, names):
    """{literal: (kind "dir" | "path", θ, scope, j)} of the 360-degree half circle literals (`shape.py`), and
    {(kind, θ, scope, j): literal}."""
    import re
    of, back = {}, {}
    for m in range(model.num_literals):
        info = model.literal_info(m)
        if info["kind"] != "X":
            continue
        mt = re.fullmatch(r"shp:(dir|path)_(\d{3})", names[info["atom"]])
        if mt:
            key = (mt.group(1), int(mt.group(2)), info["scope"], info["j"])
            of[m], back[key] = key, m
    return of, back


def _sector(thetas):
    """(start, width) in degrees of the intersection of half circles [θ, θ + 180), None if empty."""
    grid = torch.arange(360, dtype=torch.float)
    inside = torch.ones(360, dtype=torch.bool)
    for t in thetas:
        inside &= ((grid - t) % 360) < 180
    if not inside.any() or inside.all():
        return None
    return int(torch.nonzero(inside & ~inside.roll(1)).flatten()[0]), int(inside.sum())


def refine_sectors(rules, model, names, pool_of, min_events=2, min_lift=1.2, max_width=180):
    """Narrow each rule's direction to where its onsets concentrate (the user, 2026-11-23: find the dense region
    inside the interval and shrink it). Per rule and direction kind (dir, path; own scope, slot 1):
      - every row's angle bin (ANGLE_STEP degrees) is read from the half circle literals (the last θ whose half
        circle holds before the next one does not);
      - on the rows where the rule holds, every arc inside the rule's sector (or anywhere, for a rule without
        that kind) is scored by FOIL gain: weighted hits x (log precision of the arc - log precision of the rule);
      - the best arc with >= min_events onsets and precision >= min_lift x the rule's becomes a new rule: the
        rule's other literals and the arc as two half circles.
    The new rules are added; the soft fit (`fit_soft`) weighs them against the old ones. Returns the rules and
    the number added."""
    from src.exp_roadpp import shape as agent_shape
    dir_of, lit_of = direction_literals(model, names)
    if not dir_of:
        return rules, 0
    step = agent_shape.ANGLE_STEP
    nb = 360 // step
    added, seen = [], {(r["head"], tuple(sorted(r["literals"]))) for r in rules}
    by_head = defaultdict(list)
    for r in rules:
        by_head[r["head"]].append(r)
    for head, rs in by_head.items():
        pool = pool_of(head)
        F = pool["F"] > 0.5
        w, y = pool["w"], pool["y"]
        for kind in ("dir", "path"):
            cols = [lit_of.get((kind, t * step, 0, 1)) for t in range(nb)]
            if any(c is None for c in cols):
                continue
            B = F[:, cols]                                                     # (n, nb)
            last = B & ~B.roll(-1, dims=1)
            has = last.any(dim=1)
            bin_ = last.float().argmax(dim=1)
            for r in rs:
                d = [m for m in r["literals"] if m in dir_of and dir_of[m][0] == kind]
                other = [m for m in r["literals"] if m not in d]
                holds = F[:, other].all(dim=1) if other else torch.ones(len(y), dtype=torch.bool, device=F.device)
                if d:
                    holds &= F[:, d].all(dim=1)
                rows = holds & has
                if int((rows & y).sum()) < min_events:
                    continue
                fires_r = float(w[holds].sum())
                p_rule = float(w[holds & y].sum()) / max(fires_r, 1e-12)
                if p_rule <= 0:
                    continue
                hw = torch.zeros(nb, device=F.device).index_add_(0, bin_[rows], w[rows])
                hy = torch.zeros(nb, device=F.device).index_add_(0, bin_[rows], (w * y.float())[rows])
                allowed = torch.ones(nb, dtype=torch.bool, device=F.device)
                if d:
                    sec = _sector([dir_of[m][1] for m in d])
                    if sec is None:
                        continue
                    allowed = torch.zeros(nb, dtype=torch.bool, device=F.device)
                    for k in range(sec[1] // step):
                        allowed[(sec[0] // step + k) % nb] = True
                # Every arc [start, start + width) of bins inside the allowed bins: cumulative sums on the circle.
                cw, cy = torch.cat([hw, hw]).cumsum(0), torch.cat([hy, hy]).cumsum(0)
                ca = torch.cat([allowed, allowed]).long().cumsum(0)
                best = None
                for width in range(1, max_width // step + 1):
                    if d and width * step >= sec[1]:
                        break
                    starts = torch.arange(nb, device=F.device)
                    end = starts + width - 1
                    fw = cw[end] - torch.where(starts > 0, cw[starts - 1], torch.zeros_like(cw[0]))
                    fy = cy[end] - torch.where(starts > 0, cy[starts - 1], torch.zeros_like(cy[0]))
                    inside = (ca[end] - torch.where(starts > 0, ca[starts - 1], torch.zeros_like(ca[0]))) == width
                    p = fy / fw.clamp(min=1e-12)
                    gain = fy * (torch.log(p.clamp(min=1e-12)) - math.log(p_rule))
                    ok = inside & (fy > 0) & (p >= min_lift * p_rule)
                    gain = torch.where(ok, gain, torch.full_like(gain, -1.0))
                    i = int(gain.argmax())
                    if gain[i] > 0 and (best is None or float(gain[i]) > best[0]):
                        best = (float(gain[i]), i, width, float(p[i]))
                if best is None:
                    continue
                _g, start, width, _p = best
                a, b = start * step % 360, (start + width) * step
                lits = other + [lit_of[(kind, a, 0, 1)], lit_of[(kind, (b - 180) % 360, 0, 1)]]
                mask = F[:, lits].all(dim=1)
                events = int(pool["event"][mask & y].unique().numel())
                key = (head, tuple(sorted(lits)))
                if events < min_events or key in seen:
                    continue
                seen.add(key)
                fires = float(w[mask].sum())
                added.append({"head": head, "literals": lits, "refined": True, "events": events,
                              "precision": float(w[mask & y].sum()) / fires if fires > 0 else 0.0})
    return rules + added, len(added)


def linear_literals(model, names):
    """{literal: (quantity, "ge" | "lt", threshold index, scope, j)}, {key: literal} and {quantity: thresholds} of
    the fine position / size / distance literals (`shape.THRESHOLDS`)."""
    import re
    from src.exp_roadpp import shape as agent_shape
    of, back = {}, {}
    for m in range(model.num_literals):
        info = model.literal_info(m)
        if info["kind"] != "X":
            continue
        mt = re.fullmatch(r"shp:(cx|by|app_cx|app_by|size|move|travel)_(ge|lt)_([0-9.]+)", names[info["atom"]])
        if mt:
            q = mt.group(1)
            k = agent_shape.THRESHOLDS[q].index(float(mt.group(3)))
            key = (q, mt.group(2), k, info["scope"], info["j"])
            of[m], back[key] = key, m
    return of, back, agent_shape.THRESHOLDS


def _bins(lits_of_q, of):
    """[lo, hi) in bins (bin b: thresholds b - 1 <= value < threshold b; n + 1 bins) of a rule's ">=" / "<" literals."""
    lo, hi = 0, None
    for m in lits_of_q:
        q, op, k = of[m][:3]
        if op == "ge":
            lo = max(lo, k + 1)
        else:
            hi = k + 1 if hi is None else min(hi, k + 1)
    return lo, hi


def _interval_literals(q, lo, hi, nbins, back, scope=0, j=1):
    """The ">=" / "<" literals of the bins [lo, hi) of quantity q, None for the whole range."""
    lits = []
    if lo > 0:
        lits.append(back[(q, "ge", lo - 1, scope, j)])
    if hi < nbins:
        lits.append(back[(q, "lt", hi - 1, scope, j)])
    return lits or None


def refine_intervals(rules, model, names, pool_of, min_events=2, min_lift=1.2):
    """`refine_sectors` for the fine position / size / distance quantities: per rule and quantity, the interval of
    bins inside the rule's interval (or anywhere) with the largest FOIL gain, precision >= min_lift x the rule's and
    >= min_events onsets, added as a new rule. Returns the rules and the number added."""
    of, back, thresholds = linear_literals(model, names)
    if not of:
        return rules, 0
    added, seen = [], {(r["head"], tuple(sorted(r["literals"]))) for r in rules}
    by_head = defaultdict(list)
    for r in rules:
        by_head[r["head"]].append(r)
    for head, rs in by_head.items():
        pool = pool_of(head)
        F = pool["F"] > 0.5
        w, y = pool["w"], pool["y"]
        for q, ts in thresholds.items():
            n = len(ts)
            ge = [back.get((q, "ge", k, 0, 1)) for k in range(n)]
            lt0 = back.get((q, "lt", 0, 0, 1))
            if any(c is None for c in ge) or lt0 is None:
                continue
            has = F[:, ge[0]] | F[:, lt0]                                      # the quantity has a value
            bin_ = F[:, ge].long().sum(dim=1)                                  # 0 .. n
            for r in rs:
                mine = [m for m in r["literals"] if m in of and of[m][0] == q]
                other = [m for m in r["literals"] if m not in mine]
                holds = F[:, other].all(dim=1) if other else torch.ones(len(y), dtype=torch.bool, device=F.device)
                if mine:
                    holds &= F[:, mine].all(dim=1)
                rows = holds & has
                if int((rows & y).sum()) < min_events:
                    continue
                fires_r = float(w[holds].sum())
                p_rule = float(w[holds & y].sum()) / max(fires_r, 1e-12)
                if p_rule <= 0:
                    continue
                lo0, hi0 = _bins(mine, of) if mine else (0, n + 1)
                hi0 = n + 1 if hi0 is None else hi0
                hw = torch.zeros(n + 1, device=F.device).index_add_(0, bin_[rows], w[rows])
                hy = torch.zeros(n + 1, device=F.device).index_add_(0, bin_[rows], (w * y.float())[rows])
                cw, cy = torch.cat([torch.zeros(1, device=F.device), hw.cumsum(0)]), \
                    torch.cat([torch.zeros(1, device=F.device), hy.cumsum(0)])
                best = None
                for lo in range(lo0, hi0):
                    for hi in range(lo + 1, hi0 + 1):
                        if (lo, hi) == (lo0, hi0) or (lo == 0 and hi == n + 1):
                            continue
                        fw, fy = float(cw[hi] - cw[lo]), float(cy[hi] - cy[lo])
                        if fy <= 0 or fw <= 0:
                            continue
                        p = fy / fw
                        if p < min_lift * p_rule:
                            continue
                        gain = fy * (math.log(p) - math.log(p_rule))
                        if gain > 0 and (best is None or gain > best[0]):
                            best = (gain, lo, hi)
                if best is None:
                    continue
                lits = other + _interval_literals(q, best[1], best[2], n + 1, back)
                key = (head, tuple(sorted(lits)))
                mask = F[:, lits].all(dim=1)
                events = int(pool["event"][mask & y].unique().numel())
                if events < min_events or key in seen:
                    continue
                seen.add(key)
                fires = float(w[mask].sum())
                added.append({"head": head, "literals": lits, "refined": True, "events": events,
                              "precision": float(w[mask & y].sum()) / fires if fires > 0 else 0.0})
    return rules + added, len(added)


def merge_intervals(rules, model, names, pool_of, ratio):
    """`merge_sectors` for the fine position / size / distance quantities: rules equal but for the interval of one
    quantity, intervals next to each other, merged into one covering both and the gap if the merged rule keeps
    >= ratio x the precision of the two together. Returns the rules and the number of merges."""
    of, back, thresholds = linear_literals(model, names)
    if not of:
        return rules, 0
    groups, out = defaultdict(list), []
    for r in rules:
        qs = {of[m][0] for m in r["literals"] if m in of}
        if len(qs) != 1 or r.get("root"):
            out.append(r)
            continue
        q = qs.pop()
        mine = [m for m in r["literals"] if m in of and of[m][0] == q]
        if any(of[m][3:] != (0, 1) for m in mine):
            out.append(r)
            continue
        lo, hi = _bins(mine, of)
        hi = len(thresholds[q]) + 1 if hi is None else hi
        other = tuple(sorted(m for m in r["literals"] if m not in mine))
        groups[(r["head"], other, q)].append([lo, hi, r])
    merges = 0
    for (head, other, q), items in groups.items():
        nb = len(thresholds[q]) + 1
        if len(items) < 2:
            out += [it[2] for it in items]
            continue
        pool = pool_of(head)
        F = pool["F"] > 0.5

        def holds(lo, hi):
            lits = list(other) + (_interval_literals(q, lo, hi, nb, back) or [])
            return F[:, lits].all(dim=1)

        def precision(mask):
            fires = float(pool["w"][mask].sum())
            return float(pool["w"][mask & pool["y"]].sum()) / fires if fires > 0 else 0.0
        changed = True
        while changed and len(items) > 1:
            changed = False
            items.sort(key=lambda it: it[0])
            for a, b in zip(items, items[1:]):
                lo, hi = a[0], max(a[1], b[1])
                if (lo == 0 and hi == nb) or b[1] <= a[1]:
                    continue
                merged = holds(lo, hi)
                if bool(merged.any()) and precision(merged) >= ratio * precision(holds(a[0], a[1]) | holds(b[0], b[1])):
                    lits = list(other) + _interval_literals(q, lo, hi, nb, back)
                    items = [it for it in items if it is not a and it is not b] + \
                        [[lo, hi, {"head": head, "literals": lits, "merged": True}]]
                    merges += 1
                    changed = True
                    break
        out += [it[2] for it in items]
    for r in out:
        if r.get("merged") and "precision" not in r:
            pool = pool_of(r["head"])
            m = (pool["F"][:, r["literals"]] > 0.5).all(dim=1)
            fires = float(pool["w"][m].sum())
            r["precision"] = float(pool["w"][m & pool["y"]].sum()) / fires if fires > 0 else 0.0
            r["events"] = int(pool["event"][m & pool["y"]].unique().numel())
    return out, merges


def merge_sectors(rules, model, names, pool_of, ratio, max_width=180):
    """Rules equal but for the sector of one direction kind (dir or path, same scope and slot), with sectors
    next to each other on the circle, merged into one rule with the sector covering both and the gap between
    (the user, 2026-11-23: (35, 37) and (40, 43) -> (35, 43); how far to merge is for the data to say). A merge
    is kept if the merged rule's precision on the train sample (pool_of(head)) is >= ratio x the precision of the
    two rules together; merged until no pair passes. Returns the rules and the number of merges."""
    dir_of, lit_of = direction_literals(model, names)
    if not dir_of:
        return rules, 0
    groups = defaultdict(list)                  # (head, other literals, kind, scope, j) -> [(start, width, rule)]
    rest = []
    for r in rules:
        d = [m for m in r["literals"] if m in dir_of]
        kinds = {dir_of[m][0::2] for m in d}
        if len(kinds) != 1 or r.get("root"):
            rest.append(r)                      # no direction, or both kinds: left as it is
            continue
        kind, scope, j = dir_of[d[0]][0], dir_of[d[0]][2], dir_of[d[0]][3]
        sec = _sector([dir_of[m][1] for m in d])
        if sec is None:
            rest.append(r)
            continue
        other = tuple(sorted(m for m in r["literals"] if m not in dir_of))
        groups[(r["head"], other, kind, scope, j)].append([sec[0], sec[1], r])
    merges, out = 0, list(rest)
    for (head, other, kind, scope, j), items in groups.items():
        if len(items) < 2:
            out += [it[2] for it in items]
            continue
        pool = pool_of(head)
        F = pool["F"] > 0.5

        def holds(start, width):
            lits = list(other) + [lit_of[(kind, start % 360, scope, j)],
                                  lit_of[(kind, (start + width - 180) % 360, scope, j)]]
            return F[:, lits].all(dim=1)

        def precision(mask):
            fires = float(pool["w"][mask].sum())
            return float(pool["w"][mask & pool["y"]].sum()) / fires if fires > 0 else 0.0
        changed = True
        while changed and len(items) > 1:
            changed = False
            items.sort(key=lambda it: it[0])
            for i in range(len(items)):
                a, b = items[i], items[(i + 1) % len(items)]
                if a is b:
                    continue
                width = (b[0] - a[0]) % 360 + b[1]            # from a's start to b's end
                if width > max_width or width <= max(a[1], b[1]):
                    continue
                if ((kind, a[0] % 360, scope, j) not in lit_of
                        or (kind, (a[0] + width - 180) % 360, scope, j) not in lit_of):
                    continue
                both = holds(a[0], a[1]) | holds(b[0], b[1])
                merged = holds(a[0], width)
                if precision(merged) >= ratio * precision(both) and bool(merged.any()):
                    lits = list(other) + [lit_of[(kind, a[0] % 360, scope, j)],
                                          lit_of[(kind, (a[0] + width - 180) % 360, scope, j)]]
                    items = [it for it in items if it is not a and it is not b] + \
                        [[a[0], width, {"head": head, "literals": lits, "merged": True}]]
                    merges += 1
                    changed = True
                    break
        out += [it[2] for it in items]
    # Precision and onsets of the merged rules on the train sample (the others keep their exact counts).
    for r in out:
        if r.get("merged"):
            pool = pool_of(r["head"])
            m = (pool["F"][:, r["literals"]] > 0.5).all(dim=1)
            fires = float(pool["w"][m].sum())
            r["precision"] = float(pool["w"][m & pool["y"]].sum()) / fires if fires > 0 else 0.0
            r["events"] = int(pool["event"][m & pool["y"]].unique().numel())
    return out, merges


# -------------- Scorers --------------
def confidence(rules, base, k):
    p = torch.tensor([r["precision"] for r in rules], dtype=torch.float)
    e = torch.tensor([r["events"] for r in rules], dtype=torch.float)
    b = base[torch.tensor([r["head"] for r in rules], dtype=torch.long)] if rules else torch.zeros(0)
    if k <= 0:
        return p
    return (e + k * b) / (e / p.clamp(min=1e-9) + k)


def fit_logit(rules, train, model, heads, device, steps=500, l2=1e-5):
    """Per head, a logistic regression on the rules that hold (weighted train sample): (R,) weights, (A,) bias."""
    L = model.num_literals
    weight = torch.zeros(len(rules))
    bias = torch.full((model.num_atoms,), -10.0)
    by = defaultdict(list)
    for i, r in enumerate(rules):
        by[r["head"]].append(i)
    for h in heads:
        if not bool((train["pairs"]["head"] == h).any()):
            continue
        pool = step05_mine._pool_of(train, h, L, device)
        if not bool(pool["y"].any()):
            continue
        idx = by.get(h, [])
        X = (_holds_all(_with_true(pool["F"] > 0.5), _pad([rules[i]["literals"] for i in idx], L).to(device)).float()
             if idx else torch.zeros((len(pool["y"]), 0), device=device))
        y, w = pool["y"].float(), pool["w"] / pool["w"].sum()
        p0 = max(float((w * y).sum()), 1e-6)
        wt = torch.zeros(X.shape[1], device=device, requires_grad=True)
        b0 = torch.tensor(math.log(p0 / max(1 - p0, 1e-6)), device=device, requires_grad=True)
        opt = torch.optim.Adam([wt, b0], lr=0.1)
        for _ in range(steps):
            opt.zero_grad()
            z = X @ wt + b0
            loss = (w * torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")).sum() \
                + l2 * (wt ** 2).sum()
            loss.backward()
            opt.step()
        weight[idx] = wt.detach().cpu()
        bias[h] = float(b0)
    return weight, bias


def _soft_act(lits_f, body, gate):
    """(B, R) soft rule activations: the product over a rule's literals of 1 - g (1 - x), x the literal (0 / 1),
    g its gate in [0, 1] (1: required, the hard AND; 0: ignored). lits_f (B, L + 1) float, body / gate (R, K)."""
    out = []
    for i in range(0, len(body), RULE_CHUNK):
        x = lits_f[:, body[i:i + RULE_CHUNK]]                                   # (B, Rc, K)
        out.append((1 - gate[i:i + RULE_CHUNK][None] * (1 - x)).prod(dim=2))
    return torch.cat(out, dim=1) if out else torch.zeros((len(lits_f), 0), device=lits_f.device)


def fit_soft(rules, conf, base, train, model, heads, device, mode, steps=500, l2=1e-5, gate_reg=1e-4):
    """Soft AND and soft OR per head, fit on the weighted train sample (2026-11-22, the user: the AND within a
    rule and the OR between rules should be soft, not hard).
    Soft AND: a rule's activation a_r = prod_i (1 - g_ri (1 - x_i)); every gate g_ri starts at ~0.95 (the
    mined hard rule) and may fall, so a rule with a literal missing still gives part of its evidence;
    gate_reg x sum(1 - g) keeps the gates near the hard rule unless the data pays for it.
    Soft OR, mode "logit": p = sigmoid(b + sum_r w_r a_r); mode "nor": p = 1 - (1 - base) prod_r (1 - q_r a_r),
    q_r from the rule's confidence. Modes of SPARSE: "logit" with the gates and weights pushed toward 0
    instead (readable rules: the literals and rules that do not pay drop). Returns gate (R, K) and
    (w (R,), b (A,)) or q (R,)."""
    logistic = mode != "nor"
    L = model.num_literals
    K = max((len(r["literals"]) for r in rules), default=1)
    gate = torch.ones((len(rules), K))
    weight, bias = torch.zeros(len(rules)), torch.full((model.num_atoms,), -10.0)
    q = conf.clone()
    by = defaultdict(list)
    for i, r in enumerate(rules):
        by[r["head"]].append(i)
    for h in heads:
        idx = by.get(h, [])
        if not idx or not bool((train["pairs"]["head"] == h).any()):
            continue
        pool = step05_mine._pool_of(train, h, L, device)
        if not bool(pool["y"].any()):
            continue
        body = _pad([rules[i]["literals"] for i in idx], L).to(device)
        if body.shape[1] < K:
            body = torch.cat([body, torch.full((len(idx), K - body.shape[1]), L, device=device)], dim=1)
        X = torch.cat([pool["F"], torch.ones((len(pool["y"]), 1), device=device)], dim=1)[:, body]   # (n, R, K)
        pad = body == L
        y, w = pool["y"].float(), pool["w"] / pool["w"].sum()
        theta = torch.full(body.shape, 3.0, device=device, requires_grad=True)
        params = [theta]
        p0 = min(max(float((w * y).sum()), 1e-6), 1 - 1e-6)
        base_loss = -(p0 * math.log(p0) + (1 - p0) * math.log(1 - p0))       # BCE of the constant base rate
        n_real = max(int((~pad).sum()), 1)
        if logistic:
            wt = torch.zeros(len(idx), device=device, requires_grad=True)
            b0 = torch.tensor(math.log(p0 / max(1 - p0, 1e-6)), device=device, requires_grad=True)
            params += [wt, b0]
        else:
            c = conf[idx].to(device).clamp(1e-4, 1 - 1e-4)
            phi = torch.log(c / (1 - c)).clone().requires_grad_(True)
            params += [phi]
            b = float(base[h])
        opt = torch.optim.Adam(params, lr=0.1)
        for _ in range(steps):
            opt.zero_grad()
            g = torch.sigmoid(theta).masked_fill(pad, 0.0)
            a = (1 - g[None] * (1 - X)).prod(dim=2)                           # (n, R)
            if logistic:
                loss = (w * torch.nn.functional.binary_cross_entropy_with_logits(a @ wt + b0, y, reduction="none")).sum()
            else:
                miss = torch.log1p(-(torch.sigmoid(phi)[None] * a).clamp(max=1 - 1e-6)).sum(dim=1)
                p = (1 - (1 - b) * torch.exp(miss)).clamp(1e-6, 1 - 1e-6)
                loss = -(w * (y * p.log() + (1 - y) * (1 - p).log())).sum()
            if mode in SPARSE:
                loss = loss + SPARSE[mode] * base_loss * (g.sum() / n_real + wt.abs().mean())
            else:
                loss = loss + gate_reg * (1 - g).masked_fill(pad, 0.0).sum()
                if logistic:
                    loss = loss + l2 * (wt ** 2).sum()
            loss.backward()
            opt.step()
        gi = torch.sigmoid(theta.detach()).masked_fill(pad, 1.0).cpu()
        gate[idx] = gi
        if logistic:
            weight[idx] = wt.detach().cpu()
            bias[h] = float(b0)
        else:
            q[idx] = torch.sigmoid(phi.detach()).cpu()
    return gate, ((weight, bias) if logistic else q)


def effective(rules, gate, weight):
    """Rules and literals left after dropping (DROP_GATE, DROP_WEIGHT): (rules, literals per rule, keep mask)."""
    lens = torch.tensor([len(r["literals"]) for r in rules])
    real = torch.arange(gate.shape[1])[None] < lens[:, None]
    kept_lit = (gate >= DROP_GATE) & real
    keep = (weight.abs() >= DROP_WEIGHT) & kept_lit.any(dim=1)
    n = int(keep.sum())
    return n, float(kept_lit[keep].sum()) / max(n, 1), keep


class RuleSet:
    """One variant's rules on device, with the scorers of the module doc."""

    def __init__(self, rules, base, priors, logit, L, A, device, soft=None):
        self.rules, self.A = rules, A
        self.body = _pad([r["literals"] for r in rules], L).to(device)
        self.head = torch.tensor([r["head"] for r in rules], dtype=torch.long, device=device)
        self.base = base.to(device)
        self.conf = {k: confidence(rules, base, k).to(device) for k in priors}
        self.weight, self.bias = (logit[0].to(device), logit[1].to(device)) if logit else (None, None)
        # soft {"logit": (gate, (w, b)), "nor": (gate, q)} (`fit_soft`)
        self.soft = {m: (g.to(device), tuple(x.to(device) for x in p) if isinstance(p, tuple) else p.to(device))
                     for m, (g, p) in (soft or {}).items()}

    def scores(self, lits):
        """{scorer: (B, A) probabilities} of literal rows lits (B, L + 1) bool."""
        B = len(lits)
        holds = _holds_all(lits, self.body).float() if len(self.rules) else None
        out = {}
        idx = self.head[None].expand(B, -1)
        for k, conf in self.conf.items():
            s = self.base[None].expand(B, -1).clone()
            if holds is not None:
                s = s.scatter_reduce(1, idx, torch.where(holds > 0, conf[None], -1.0), "amax", include_self=True)
            out[f"max_k{k:g}"] = s
            log_miss = torch.zeros((B, self.A), device=lits.device)
            if holds is not None:
                log_miss = log_miss.index_add(1, self.head, holds * torch.log1p(-conf.clamp(max=1 - 1e-6))[None])
            out[f"nor_k{k:g}"] = 1 - (1 - self.base[None]) * torch.exp(log_miss)
        if self.weight is not None:
            z = self.bias[None].expand(B, -1).clone()
            if holds is not None:
                z = z.index_add(1, self.head, holds * self.weight[None])
            out["logit"] = torch.sigmoid(z)
        if self.soft and len(self.rules):
            lits_f = lits.float()
            for mode, (gate, par) in self.soft.items():
                a = _soft_act(lits_f, self.body, gate)                           # (B, R)
                if mode != "nor":
                    w, b = par
                    z = b[None].expand(B, -1).clone().index_add(1, self.head, a * w[None])
                    out["soft_logit" if mode == "logit" else mode] = torch.sigmoid(z)
                else:
                    miss = torch.zeros((B, self.A), device=lits.device).index_add(
                        1, self.head, torch.log1p(-(par[None] * a).clamp(max=1 - 1e-6)))
                    out["soft_nor"] = 1 - (1 - self.base[None]) * torch.exp(miss)
        return out


# -------------- Evaluation --------------
@torch.no_grad()
def evaluate(model, mlp, training_data, video_ids, rule_sets, head_mask, possible, cfg, k_mlp, derived):
    """Score histograms {(variant, scorer) | "mlp": (pos, neg) (A, NB)} on all at-risk rows of video_ids."""
    device, S, w = cfg["device"], model.slot_size, int(cfg.get("clause_slots", 1))
    A, NB = model.num_atoms, step07_paper.NB
    hist = {}

    def add(name, p, keep, y):
        if name not in hist:
            hist[name] = (torch.zeros(A * NB, device=device), torch.zeros(A * NB, device=device))
        cell = torch.arange(A, device=device)[None] * NB + step07_paper._logit_bins(p)
        hist[name][0].add_(torch.bincount(cell[keep & y], minlength=A * NB).float())
        hist[name][1].add_(torch.bincount(cell[keep & ~y], minlength=A * NB).float())

    for video_id in tqdm(video_ids, desc="Evaluating (val)"):
        can = possible(video_id)
        for frames in training_data["frames"][video_id].split(int(cfg.get("mine_chunk", 256))):
            rows = step05._video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            keep, y = step05_mine._head_rows(rows, can, head_mask, S, w)
            if not keep.any():
                continue
            lits = literal_rows(model, rows["inputs"], derived)
            if mlp is not None:
                add("mlp", mlp.window_probs(mlp(rows["inputs"])[0])[:, k_mlp], keep, y)
            for variant, rs in rule_sets.items():
                for scorer, p in rs.scores(lits).items():
                    add(f"{variant}/{scorer}", p, keep, y)
    return {name: (pos.view(A, NB).cpu(), neg.view(A, NB).cpu()) for name, (pos, neg) in hist.items()}


def report(hist, names, heads, rule_sets):
    """Per head AP of every scorer; macro / close / wins over the learnable heads; per action."""
    ap = {name: step07_paper._ap(pos, neg) for name, (pos, neg) in hist.items()}
    npos = hist["mlp"][0].sum(dim=1)
    compared = [h for h in heads if npos[h] >= MIN_EVAL_POS]
    scorers = [n for n in ap if n != "mlp"]

    def val(name, h):
        x = float(ap[name][h])
        return 0.0 if x != x else x
    learnable = [h for h in compared if max(val(n, h) for n in ap) >= LEARNABLE_MIN_AP]
    rows = {}
    for name in scorers:
        a = [val(name, h) for h in learnable]
        m = [val("mlp", h) for h in learnable]
        variant, scorer = name.split("/")
        rs = rule_sets[variant].rules
        size = getattr(rule_sets[variant], "size", {}).get(scorer)          # after the drops (sparse modes)
        rows[name] = {"macro": sum(a) / max(len(a), 1), "mlp_macro": sum(m) / max(len(m), 1),
                      "close": sum(x >= CLOSE * y for x, y in zip(a, m)), "wins": sum(x > y for x, y in zip(a, m)),
                      "rules": size[0] if size else len(rs),
                      "literals": size[1] if size else sum(len(r["literals"]) for r in rs) / max(len(rs), 1),
                      "max_literals": size[2] if size else max((len(r["literals"]) for r in rs), default=0)}
    best = max(rows, key=lambda n: rows[n]["macro"]) if rows else None
    print(f"\n[Rule lab] eval = odd val videos, every at-risk row; {len(compared)} heads with >= {MIN_EVAL_POS} "
          f"positive rows, {len(learnable)} learnable (some scorer >= AP {LEARNABLE_MIN_AP}); MLP macro "
          f"{rows[best]['mlp_macro'] if best else float('nan'):.3f}")
    print(f"  {'variant/scorer':<22}{'rules':>7}{'lits':>6}{'max':>5}{'macro':>8}{'close':>7}{'wins':>6}")
    for name, r in sorted(rows.items(), key=lambda x: -x[1]["macro"]):
        print(f"  {name:<22}{r['rules']:>7}{r['literals']:>6.2f}{r['max_literals']:>5}{r['macro']:>8.3f}"
              f"{r['close']:>5}/{len(learnable):<3}{r['wins']:>4}")
    if best:
        by_action = defaultdict(list)
        for h in learnable:
            by_action[names[h].split(":")[-1]].append(h)
        print(f"\n  per action, {best} vs MLP (macro AP, heads):")
        for action, hs in sorted(by_action.items(), key=lambda x: sum(val(best, h) - val("mlp", h) for h in x[1])):
            print(f"    {action:<12}{sum(val(best, h) for h in hs) / len(hs):>7.3f}{sum(val('mlp', h) for h in hs) / len(hs):>7.3f}"
                  f"{len(hs):>4}")
    per_head = {names[h]: {"learnable": h in learnable, "pos_rows": int(npos[h]),
                           **{n: _none(float(ap[n][h])) for n in ap}} for h in compared}
    return {"summary": rows, "best": best, "learnable": [names[h] for h in learnable], "per_head": per_head}


def _none(x):
    return None if x != x else x


# -------------- Main --------------
LINEAR_RE = r"shp:(cx|by|app_cx|app_by|size|move|travel)_(ge|lt)_([0-9.]+)\((\w+)\)@(\d+)"


def _merge_linear(items):
    """[(text, gate)] with the threshold literals of one quantity, scope and slot merged into an interval. The soft
    AND gives a value outside a bound of gate g the membership 1 - g: the core (membership >= 0.95) keeps the bounds
    with g > 0.05, "~" the extent (>= 0.5) those with g > 0.5, if it differs (`shape.interval_text`)."""
    from src.exp_roadpp import shape as agent_shape
    import re
    groups, out = {}, []
    for t, g in items:
        m = re.fullmatch(LINEAR_RE, t)
        if m:
            groups.setdefault((m.group(1), m.group(4), m.group(5)), []).append((m.group(2), float(m.group(3)), g))
        else:
            out.append((t, g))
    for (q, scope, j), lits in groups.items():
        def text(min_g):
            ge = [v for op, v, g in lits if op == "ge" and g >= min_g]
            lt = [v for op, v, g in lits if op == "lt" and g >= min_g]
            return agent_shape.interval_text(q, ge, lt)
        core, ext = text(0.05), text(0.5)
        out.append((f"shp:{q}{core if core == ext else core + ' ~' + ext}({scope})@{j}", max(g for _, _, g in lits)))
    return out


def pretty_soft(texts, gates, min_gate=None):
    """[(text, gate)] of a soft rule with the 360-degree half circles of one kind and slot merged into one soft
    sector ("shp:dir[200, 240) ~[190, 245)(own)@1", `shape.soft_sector_text`); literals below min_gate left out."""
    from src.exp_roadpp import shape as agent_shape
    import re
    groups, out = {}, []
    for t, g in zip(texts, gates):
        if min_gate is not None and g < min_gate:
            continue
        m = re.fullmatch(r"shp:(dir|path)_(\d{3})\((\w+)\)@(\d+)", t)
        if m:
            groups.setdefault((m.group(1), m.group(3), m.group(4)), []).append((int(m.group(2)), g))
        else:
            out.append((t, g))
    for (kind, scope, j), tg in groups.items():
        out.append((f"shp:{kind}{agent_shape.soft_sector_text([t for t, _ in tg], [g for _, g in tg])}({scope})@{j}",
                    max(g for _, g in tg)))
    return _merge_linear(out)


def pretty_literals(texts):
    """Rule literal texts with the 360-degree half circles of one kind and slot merged into one sector
    ("shp:dir_200(own)@1", "shp:dir_060(own)@1" -> "shp:dir[200, 240)(own)@1", `shape.sector_text`)."""
    from src.exp_roadpp import shape as agent_shape
    import re
    groups, out = {}, []
    for t in texts:
        m = re.fullmatch(r"shp:(dir|path)_(\d{3})\((\w+)\)@(\d+)", t)
        if m:
            groups.setdefault((m.group(1), m.group(3), m.group(4)), []).append(int(m.group(2)))
        else:
            out.append(t)
    for (kind, scope, j), thetas in groups.items():
        out.append(f"shp:{kind}{agent_shape.sector_text(thetas) or '[empty]'}({scope})@{j}")
    return [t for t, _ in _merge_linear([(t, 1.0) for t in out])]


def parse_variant(name):
    """"relaxed+precise_d_nn" -> (["relaxed", "precise"], (True, True, False)): the rule sets united; flags:
    with the derived literals (_d), without the negated literals of the other agents (_nn), without the
    shape-phase atoms (_ns, `shape.py`)."""
    flags = {"_d": False, "_nn": False, "_ns": False, "_nd": False, "_np": False}
    while any(name.endswith(f) for f in flags):
        f = next(f for f in flags if name.endswith(f))
        flags[f], name = True, name[:-len(f)]
    return name.split("+"), (flags["_d"], flags["_nn"], flags["_ns"], flags["_nd"], flags["_np"])


def atom_literals(model, lo, hi, older_only=False):
    """(L,) bool: the literals (any kind and scope) of the atoms lo..hi - 1, e.g. the shape atoms; older_only:
    only those of the slots before the newest (j > 1)."""
    out = torch.zeros(model.num_literals, dtype=torch.bool)
    for m in range(model.num_literals):
        info = model.literal_info(m)
        out[m] = info["kind"] in ("X", "!X") and lo <= info["atom"] < hi and (not older_only or info["j"] > 1)
    return out


def ensure_mlp(training_data, cfg):
    """The run's OnsetMLP, trained (and saved) if missing: the lab needs no rule net, so a new run with new
    atoms (e.g. --shape_phases True) only builds the data and the MLP here."""
    mlp = step05.load_onset_mlp(training_data, cfg)
    if mlp is None:
        print(f"[Rule lab] no OnsetMLP at {cfg['mlp_model_file']}: training it on the same inputs")
        torch.save(step05.train_onset_mlp(training_data, cfg).state_dict(), cfg["mlp_model_file"])
        mlp = step05.load_onset_mlp(training_data, cfg)
    return mlp


def negated_others(model):
    """(L,) bool: the negated literals of the other agents ("!X(other)", "!veh:X(other)"). On v7 they held on
    almost every row ("no bus is moving left"), so the relaxed miner's deep trees added them as noise
    (lab 3, 2026-11-22: e.g. Car:TurRht w -3.33 "!Bus:MovLft(other)@2 & ...")."""
    out = torch.zeros(model.num_literals, dtype=torch.bool)
    for m in range(model.num_literals):
        info = model.literal_info(m)
        out[m] = info["kind"] in ("!X", "!G") and info["scope"] == 1
    return out


def crisp_rules(rules, gate, weight, pool_of, heads):
    """The rules of a sparse soft fit made crisp: the literals with gate >= 0.5 (a hard AND), the rules with
    |weight| >= DROP_WEIGHT, duplicates merged; precision and onsets on the train sample (pool_of(head))."""
    out, seen = [], set()
    for i, r in enumerate(rules):
        if abs(float(weight[i])) < DROP_WEIGHT:
            continue
        lits = tuple(sorted(m for m, g in zip(r["literals"], gate[i].tolist()) if g >= 0.5))
        if lits and (r["head"], lits) not in seen:
            seen.add((r["head"], lits))
            out.append({"head": r["head"], "literals": list(lits)})
    by = defaultdict(list)
    for r in out:
        by[r["head"]].append(r)
    for h, rs in by.items():
        pool = pool_of(h)
        F = _with_true(pool["F"] > 0.5)
        for r in rs:
            m = F[:, r["literals"]].all(dim=1)
            fires, hits = float(pool["w"][m].sum()), float(pool["w"][m & pool["y"]].sum())
            r["precision"] = hits / fires if fires > 0 else 0.0
            r["events"] = int(pool["event"][m & pool["y"]].unique().numel())
    return out


def mine_component(component, use_derived, ext, names, heads, train, check, cfg, device):
    """The rules {"head", "literals"[, "root"]} of one miner ("m4", "relaxed", "precise") on the train sample."""
    L, A = ext.num_literals, ext.num_atoms
    rules = []
    if component in ("m4", "relaxed"):
        over = RELAXED if component == "relaxed" else {}
        for h in tqdm(heads, desc=f"Mining {component}{' + derived' if use_derived else ''}"):
            seeds, _ = step05_mine.mine_head(ext, names, h, train, check, {**cfg, **over}, device)
            rules += [{"head": int(sd["head"]), "literals": list(sd["literals"]), "root": bool(sd.get("root"))}
                      for sd in seeds]
        return rules
    base = ext.literal_base.to(device)
    scope = base // (ext.num_spans * A)
    for h in tqdm(heads, desc=f"Precise search{' + derived' if use_derived else ''}"):
        if not bool((train["pairs"]["head"] == h).any()):
            continue
        allowed = ext.input_mask.to(device) & ~((base % A == h) & (scope == 0))
        pool = step05_mine._pool_of(train, h, L, device)
        rules += [{"head": int(h), "literals": list(key)} for key, _p, _e in precise_rules(pool, allowed, cfg)]
    return rules


def show_rules(texts, heads, mode, top=8):
    """Print the strongest rules (soft logistic weight of mode) of some heads, with each literal's gate; in a
    sparse mode the dropped literals and rules (DROP_GATE, DROP_WEIGHT) are left out."""
    sparse = mode in SPARSE or mode == "crisp"
    for head in heads:
        rs = [r for r in texts if r["head"] == head and mode in r["soft"]]
        kept = [r for r in rs if not sparse or abs(r["soft"][mode]["weight"]) >= DROP_WEIGHT]
        kept = sorted(kept, key=lambda r: -abs(r["soft"][mode]["weight"]))
        if not kept:
            continue
        print(f"  {head}: {len(kept)} of {len(rs)} rules")
        for r in kept[:top]:
            s = r["soft"][mode]
            lits = " & ".join(f"{t}[{g:.2f}]" for t, g in pretty_soft(r["literals"], s["gates"],
                                                                      DROP_GATE if sparse else None))
            print(f"    w {s['weight']:+6.2f}  p {r['precision']:.2f}  {r['events']:5d} onsets  {lits}")


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--lab_variants", type=str, default="relaxed,relaxed_nn,relaxed+precise_nn",
                        help="rule sets: miners m4 / relaxed / precise joined by +, suffix _d: with the derived literals")
    parser.add_argument("--lab_priors", type=str, default="0", help="confidence shrink k (onsets)")
    parser.add_argument("--lab_crisp", type=str, default="sparse_c",
                        help="the sparse soft mode whose rounded gates give the crisp rules (variant ~crisp)")
    parser.add_argument("--lab_soft", type=str, default="logit,sparse_c",
                        help="soft modes: logit, nor, and the sparse ones of SPARSE")
    parser.add_argument("--lab_show", type=str,
                        default="Cyc:VehLane,Car:IncomLane,Cyc:MovRht,av:AV-TurLft,Car:MovAway,av:AV-Stop",
                        help="heads whose strongest rules are printed")
    parser.add_argument("--lab_dir_negated", type=str, default="False",
                        help="allow the negated 360-degree direction literals (they also hold without a move)")
    parser.add_argument("--lab_refine", type=str, default="True",
                        help="add rules whose direction sector is narrowed to where their onsets concentrate")
    parser.add_argument("--lab_merge_ratio", type=float, default=0.9,
                        help="merge two rules that differ only in a close direction sector if the covering sector "
                             "keeps >= this share of their precision (0: no merging)")
    parser.add_argument("--lab_shape_slot1", type=str, default="True",
                        help="the shape atoms only in the newest slot (their history is in the phases)")
    parser.add_argument("--lab_root_exclude", type=str, default="shape",
                        help="atom groups of the data layout (e.g. shape, position) that may not be mining roots")
    parser.add_argument("--lab_precise_min", type=float, default=0.5, help="precise search: kept rule precision")
    parser.add_argument("--lab_min_events", type=int, default=2, help="precise search: distinct onsets of a rule")
    parser.add_argument("--lab_min_videos", type=int, default=1)
    parser.add_argument("--lab_max_len", type=int, default=8, help="precise search: literals per rule")
    parser.add_argument("--lab_beam", type=int, default=100, help="precise search: nodes per level")
    parser.add_argument("--lab_branch", type=int, default=10, help="precise search: children per node")
    parser.add_argument("--lab_max_rules", type=int, default=100, help="precise search: rules per head")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    if not getattr(args, "run_id", None):
        raise ValueError("--run_id is needed: the lab compares with that run's OnsetMLP")
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    for key in ("lab_precise_min", "lab_min_events", "lab_min_videos", "lab_max_len", "lab_beam", "lab_branch",
                "lab_max_rules"):
        cfg[key] = getattr(args, key)
    device = cfg["device"]
    out_dir = args.output_dir / "rule_lab"
    os.makedirs(out_dir, exist_ok=True)

    training_data = step05.load_training_data(cfg)
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    mlp = ensure_mlp(training_data, cfg)
    window = int(cfg.get("clause_slots", 1)) * model.slot_size
    k_mlp = min(range(len(model.windows)), key=lambda i: abs(model.windows[i] - window))
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    head_mask = training_data["target_heads"].to(device)
    heads = torch.nonzero(head_mask).flatten().tolist()
    possible = step05.class_possible(training_data, vocab, device)
    A = model.num_atoms

    # Literals: the rule net's and the derived ones (same indices in every variant; the variants without "_d"
    # may not use the derived ones).
    derived = Derived(model, names)
    forbid_nn = negated_others(model)
    layout = training_data.get("grid_layout", {})
    cfg["mine_root_exclude"] = [layout[k] for k in args.lab_root_exclude.split(",") if k in layout]
    if cfg["mine_root_exclude"]:
        print(f"[Rule lab] no mining roots from {[k for k in args.lab_root_exclude.split(',') if k in layout]} "
              f"(atoms {cfg['mine_root_exclude']})")
    forbid_ns = (atom_literals(model, *layout["shape"]) if "shape" in layout
                 else torch.zeros(model.num_literals, dtype=torch.bool))
    if "shape" in layout:
        print(f"[Rule lab] shape-phase atoms {layout['shape']}: {int(forbid_ns.sum())} literals (off in the _ns variants)")
    # The shape atoms carry their own history (the phases p2..p4, the order-free rec_*): a rule reads them in the
    # newest slot only (2026-11-23; "p1_x_left@3", the current phase 2-3 s ago, doubled the phases).
    forbid_old = (atom_literals(model, *layout["shape"], older_only=True)
                  if "shape" in layout and args.lab_shape_slot1 == "True" else torch.zeros_like(forbid_ns))

    # The 360-degree direction literals: "_nd" variants go without them; their negations ("not in this half
    # circle", which also holds without a move) are off unless --lab_dir_negated True.
    dir_of, _ = direction_literals(model, names)
    forbid_dir = torch.zeros_like(forbid_nn)
    forbid_dir_neg = torch.zeros_like(forbid_nn)
    for m in dir_of:
        forbid_dir[m] = True
        if model.negation:
            forbid_dir[model.num_inputs + m] = True
            forbid_dir_neg[model.num_inputs + m] = args.lab_dir_negated != "True"

    lin_of, _, _ = linear_literals(model, names)
    forbid_lin = torch.zeros_like(forbid_nn)
    for m in lin_of:
        forbid_lin[m] = True
        if model.negation:
            forbid_lin[model.num_inputs + m] = True       # "not >= τ" is "< τ" or no value: the "<" atoms cover it

    def make_ext(flags):
        d, nn, ns, nd, np_ = flags
        forbid = ((forbid_nn if nn else torch.zeros_like(forbid_nn)) | (forbid_ns if ns else torch.zeros_like(forbid_ns))
                  | forbid_old | forbid_dir_neg | (forbid_dir if nd else torch.zeros_like(forbid_dir))
                  | (forbid_lin if np_ else (forbid_lin & (torch.arange(len(forbid_lin)) >= model.num_inputs))))
        return ExtModel(model, derived, d, forbid if bool(forbid.any()) else None)
    ext = {}
    L = make_ext((True, False, False, False, False)).num_literals
    train, val = mine_sweep.load_rows(cfg, training_data, model, head_mask, possible, args.output_dir / "mine_sweep")
    val_ids = list(training_data["val_videos"])
    train = extend_rows(train, derived, model.num_literals, device)
    check = extend_rows(mine_sweep.subset(val, set(range(0, len(val_ids), 2))), derived, model.num_literals, device)
    del val
    eval_ids = val_ids[1::2]

    # 1. Rule sets on the train sample, each miner once per literal set.
    variants = args.lab_variants.split(",")
    mined = {}
    for name in variants:
        components, flags = parse_variant(name)
        for c in components:
            if (c, flags) not in mined:
                if flags not in ext:
                    ext[flags] = make_ext(flags)
                mined[(c, flags)] = mine_component(c, flags[0], ext[flags], names, heads, train, check, cfg, device)

    # 2. Exact counts on all train rows (each distinct rule once), then the final rule sets.
    unique = {}
    for rules in mined.values():
        for r in rules:
            unique.setdefault((r["head"], tuple(sorted(r["literals"]))), r)
    keys = list(unique)
    counts = recount(model, training_data, training_data["train_videos"], [unique[k] for k in keys], head_mask,
                     possible, cfg, derived)
    exact = {k: i for i, k in enumerate(keys)}
    priors = [float(x) for x in args.lab_priors.split(",")]
    soft_modes = [m for m in args.lab_soft.split(",") if m]
    rule_sets, texts = {}, {}
    for name in variants:
        components, flags = parse_variant(name)
        seen, rs = set(), []
        for c in components:
            for r in mined[(c, flags)]:
                key = (r["head"], tuple(sorted(r["literals"])))
                if key not in seen:
                    seen.add(key)
                    rs.append(r)
        idx = [exact[(r["head"], tuple(sorted(r["literals"])))] for r in rs]
        sub = {k: v[idx] for k, v in counts.items() if k != "base"}
        final = finalize(rs, sub, 1 if components == ["m4"] else args.lab_min_events)
        full = make_ext((True, False, False, False, False))
        if args.lab_merge_ratio > 0:
            pools = {}

            def pool_of(h):
                if h not in pools:
                    pools[h] = step05_mine._pool_of(train, h, L, device)
                return pools[h]
            n0 = len(final)
            if args.lab_refine == "True" and not flags[3]:
                final, n_ref = refine_sectors(final, model, names, pool_of, args.lab_min_events)
                print(f"[Rule lab] {name}: {n_ref} rules with a direction narrowed to where their onsets concentrate")
            if args.lab_refine == "True" and not flags[4]:
                final, n_ref = refine_intervals(final, model, names, pool_of, args.lab_min_events)
                print(f"[Rule lab] {name}: {n_ref} rules with a position / size / distance interval narrowed")
            final, n_merges = merge_sectors(final, model, names, pool_of, args.lab_merge_ratio)
            final, n_lin = merge_intervals(final, model, names, pool_of, args.lab_merge_ratio)
            n_merges += n_lin
            if n_merges:
                print(f"[Rule lab] {name}: {n_merges} merges of close direction sectors, {n0} -> {len(final)} rules")
            pools.clear()
        logit = fit_logit(final, train, full, heads, device)
        soft = {m: fit_soft(final, confidence(final, counts["base"], SOFT_PRIOR), counts["base"], train, full,
                            heads, device, m) for m in soft_modes}
        for m, (g, _p) in soft.items():
            real = g[_pad([r["literals"] for r in final], L) != L] if final else g.flatten()
            print(f"[Rule lab] {name} soft {m}: {float((real < 0.5).float().mean()) if len(real) else 0:.1%} of the "
                  f"literals relaxed (gate < 0.5)")
        rule_sets[name] = RuleSet(final, counts["base"], priors, logit, L, A, device, soft)
        rule_sets[name].size = {}
        for m, (g, par) in soft.items():
            if m in SPARSE and final:
                n_eff, lits_eff, keep = effective(final, g, par[0])
                lens = (g >= DROP_GATE).sum(dim=1)[keep]
                rule_sets[name].size[m] = (n_eff, lits_eff, int(lens.max()) if len(lens) else 0)
                print(f"[Rule lab] {name} {m}: {n_eff} of {len(final)} rules kept, {lits_eff:.2f} literals per rule")
        texts[name] = sorted(({"head": names[r["head"]], "precision": r["precision"], "events": r["events"],
                               "literals": [step05_mine.literal_text(full, names, m) for m in r["literals"]],
                               "derived": sum(m >= model.num_literals for m in r["literals"]),
                               "weight": round(float(logit[0][i]), 3),
                               "soft": {m: {"weight": round(float(par[0][i]), 3),
                                            "gates": [round(float(x), 3) for x in g[i][:len(r["literals"])]]}
                                        for m, (g, par) in soft.items() if m != "nor"}}
                              for i, r in enumerate(final)), key=lambda r: r["head"])
        # Crisp rules: the sparse fit's gates rounded (>= 0.5: the literal is required, else dropped), then only
        # the rule weights refit (a hard AND per rule, a logistic OR per head): readable rules, no partial gates.
        if args.lab_crisp in soft and final:
            g, par = soft[args.lab_crisp]
            crisp = crisp_rules(final, g, par[0], lambda h: step05_mine._pool_of(train, h, L, device), heads)
            c_logit = fit_logit(crisp, train, full, heads, device)
            key = f"{name}~crisp"
            rule_sets[key] = RuleSet(crisp, counts["base"], [], c_logit, L, A, device)
            kept = c_logit[0].abs() >= DROP_WEIGHT
            lens = torch.tensor([len(r["literals"]) for r in crisp])[kept] if crisp else torch.zeros(0)
            rule_sets[key].size = {"logit": (int(kept.sum()), float(lens.float().mean()) if len(lens) else 0.0,
                                             int(lens.max()) if len(lens) else 0)}
            texts[key] = sorted(({"head": names[r["head"]], "precision": r["precision"], "events": r["events"],
                                  "literals": pretty_literals([step05_mine.literal_text(full, names, m)
                                                               for m in r["literals"]]),
                                  "derived": sum(m >= model.num_literals for m in r["literals"]),
                                  "weight": round(float(c_logit[0][i]), 3),
                                  "soft": {"crisp": {"weight": round(float(c_logit[0][i]), 3),
                                                     "gates": [1.0] * len(pretty_literals(
                                                         [step05_mine.literal_text(full, names, m)
                                                          for m in r["literals"]]))}}}
                                 for i, r in enumerate(crisp)), key=lambda r: r["head"])
            print(f"[Rule lab] {key}: {int(kept.sum())} rules (|w| >= {DROP_WEIGHT}) from {args.lab_crisp}, "
                  f"{float(lens.float().mean()) if len(lens) else 0:.2f} literals per rule")
        n_d = sum(1 for r in texts[name] if r["derived"])
        print(f"[Rule lab] {name}: {len(final)} rules for {len({r['head'] for r in final})} heads "
              f"(literals per rule {sum(len(r['literals']) for r in final) / max(len(final), 1):.2f}; "
              f"{n_d} with a derived literal)")

    # 3. Every at-risk row of the odd val videos.
    hist = evaluate(model, mlp, training_data, eval_ids, rule_sets, head_mask, possible, cfg, k_mlp, derived)
    out = report(hist, names, heads, rule_sets)
    if out["best"]:
        best_variant = out["best"].split("/")[0]
        for m in soft_modes:
            if m == "nor":
                continue
            print(f"\n[Rule lab] strongest rules of {best_variant} / {m} (soft logistic weight; [gate] per literal, "
                  f"1 = required, 0 = ignored{'; dropped literals and rules left out' if m in SPARSE else ''}):")
            show_rules(texts[best_variant], [h for h in args.lab_show.split(",") if h], m)
    for key in (k for k in rule_sets if k.endswith("~crisp")):
        print(f"\n[Rule lab] crisp rules of {key} (hard AND, logistic weight):")
        show_rules(texts[key], [h for h in args.lab_show.split(",") if h], "crisp")
    tag = os.path.basename(str(cfg["mined_seed_file"])).replace("mined_seeds_", "").replace(".json", "")
    rx = "-".join(k for k in args.lab_root_exclude.split(",") if k in layout)
    path = out_dir / f"rule_lab_{tag}_{'-'.join(variants)}{'_rx' + rx if rx else ''}.json"
    utils_data.save_json({"settings": {k: getattr(args, k) for k in vars(args) if k.startswith("lab_")},
                          "mlp_window": model.windows[k_mlp], "derived": derived.text, **out, "rules": texts}, path)
    print(f"[Rule lab] saved {path}")


if __name__ == "__main__":
    main()
