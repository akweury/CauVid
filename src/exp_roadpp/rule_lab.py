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
import time
import os
import re
from collections import Counter, defaultdict

import torch
from tqdm import tqdm

from src.exp_roadpp import config, eventual, mine_sweep, step05, step05_mine, step07_paper, utils_data, utils_pipe

LEARNABLE_MIN_AP = 0.05
CLOSE = 0.9                     # "close to the MLP": AP >= CLOSE x the MLP's
MIN_EVAL_POS = 20               # heads with fewer positive eval rows are not compared
EVAL_MIN_ONSETS = 100           # the user (2026-11-25): a head's comparison needs >= 100 eval onsets
RULE_CHUNK = 4096               # rules per gather in `_holds_all`
SOFT_PRIOR = 5                  # confidence shrink (onsets) of the soft noisy-OR's starting q
# Sparse soft rules (2026-11-22, readability): the gates and the rule weights are pushed toward 0, so a rule
# keeps only the literals and the heads keep only the rules the data pays for. lambda x (the head's base loss)
# x (mean gate + mean |weight|) is added to the loss. A literal with gate < DROP_GATE and a rule with
# |weight| < DROP_WEIGHT count as dropped.
SPARSE = {"sparse_a": 0.001, "sparse_b": 0.005, "sparse_c": 0.02}   # fake-data check: 0.5+ drops every rule
DROP_GATE, DROP_WEIGHT = 0.1, 0.05
# Description length (MDL, 2026-11-23, the user: fewer and shorter rules, e.g. one group rule instead of several
# single-agent ones). Mode "mdl_λ": the loss adds λ x (the head's base loss) / MDL_SCALE x (RULE_COST x the rules
# kept + the literals kept). A rule is kept to the degree tanh(|w| / RULE_TAU), a literal to its gate times that,
# so a literal of a dropped rule costs nothing. Unlike "sparse_*" (means over the rules and literals), the cost is
# per rule and per literal: a group rule that covers the onsets of several single-agent rules is cheaper.
RULE_COST, RULE_TAU, MDL_SCALE = 2.0, 0.1, 100.0
# Mode "mdls_λ" (2026-11-25): λ scaled by the head's size, λ x min(1, MDL_SIZE_REF / train onsets of the head). The fit
# loss is a mean over rows, so the plain penalty weighs the same for a head with 50 or 3000 onsets; in a real
# description length the data term grows with the onsets and the model term does not. λ 0.1 cost the big heads
# (Car:Brake 0.322 -> 0.236, Car:IncomLane 0.539 -> 0.383 AP on v7); small heads keep λ. Set by --lab_mdl_ref.
MDL_SIZE_REF = 200.0


def mdl_lambda(mode, onsets):
    """λ of a "mdl_λ" / "mdls_λ" / "mdlt_λ" / "mdlts_λ" mode for a head with this many train (sample) onsets."""
    lam = float(mode.split("_", 1)[1])
    return lam * min(1.0, MDL_SIZE_REF / max(onsets, 1)) if mode.startswith(("mdls_", "mdlts_")) else lam


def is_temporal(mode):
    """The MDL modes over temporal rules ("mdlt_λ", "mdlts_λ", `temporal_groups`)."""
    return mode.startswith(("mdlt_", "mdlts_"))


# Temporal rules (2026-11-25, the user: the rules of one event at different times, e.g. Car:MovTow@1 +0.81,
# @2 -0.65, @3 -0.76 = "just started to come toward the ego", are one rule with a weight per time, and the weights
# may differ between situations). A temporal rule: a context C (the other literals) and an event X; its members are
# the rules C & X@k (one per slot k), each with its own weight = the time profile of X under C. Another context
# gives another rule, so another profile. The members share C's gates; the MDL fit ("mdlt_λ") pays RULE_COST and
# C's literals once per temporal rule, plus one literal per slot weight kept.
SLOT_RE = re.compile(r"^(!?)(.+)@(\d+)$")


def temporal_groups(rules, text_of):
    """Per rule (group id, index of its slot literal in r["literals"] or -1). Rules equal but for the slot of one
    literal (same sign, scope and atom) form a group; a rule joins at most one group (largest first); the others
    are groups of one (plain rules, no slot literal)."""
    cand = defaultdict(list)
    for i, r in enumerate(rules):
        lits = list(r["literals"])
        for j, m in enumerate(lits):
            hit = SLOT_RE.match(text_of(m))
            if hit:
                rest = tuple(sorted(lits[:j] + lits[j + 1:]))
                cand[(r["head"], rest, hit[1] + hit[2])].append((i, j, int(hit[3])))
    group, slot_pos = [-1] * len(rules), [-1] * len(rules)
    n = 0
    for key, members in sorted(cand.items(), key=lambda kv: -len(kv[1])):
        free = [(i, j, k) for i, j, k in members if group[i] < 0]
        if len({k for _i, _j, k in free}) < 2:
            continue
        for i, j, _k in free:
            group[i], slot_pos[i] = n, j
        n += 1
    for i in range(len(rules)):
        if group[i] < 0:
            group[i], n = n, n + 1
    return group, slot_pos


def is_compact(mode):
    """The fits that drop rules and literals (their size is counted after DROP_GATE / DROP_WEIGHT)."""
    return mode in SPARSE or mode.startswith("mdl")
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


def literal_table(model):
    """[literal_info(m) for every literal of the rule net], computed once per model (2026-11-24: eight scans of
    every literal through `literal_info` made the lab's set-up minutes long)."""
    tab = getattr(model, "_lab_literal_table", None)
    if tab is None:
        tab = [model.literal_info(m) for m in range(model.num_literals)]
        model._lab_literal_table = tab
    return tab


def compact_rows(data, heads=None):
    """A `collect` sample cut to the pairs of some heads (None: all) and to the packed rows they use (2026-11-24:
    `subset` keeps every packed row, so the derived literals were computed for rows no pair reads)."""
    pairs = data["pairs"]
    sel = torch.ones(len(pairs["head"]), dtype=torch.bool) if heads is None else \
        torch.isin(pairs["head"].long(), torch.tensor(sorted(heads), dtype=torch.long))
    pairs = {k: v[sel] for k, v in pairs.items()}
    used, inv = pairs["row"].long().unique(return_inverse=True)
    return {**data, "packed": data["packed"][used], "pairs": {**pairs, "row": inv.to(pairs["row"].dtype)}}


def extend_rows(data, derived, L, device, chunk=50000, desc="Derived literals"):
    """A `collect` sample with the derived literal bits appended to every row."""
    packed = []
    for i in tqdm(range(0, len(data["packed"]), chunk), desc=desc):
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


# -------------- Exhaustive short rules and per-seed extension ("exh", 2026-11-24) --------------
# The user: a good rule must stay whatever atoms are added. The relaxed miner is a greedy tree (4 roots, depth 6):
# an added atom can change one split and the path to a good rule is gone (v4-v6 A/B on the same data:
# "Car:OutgoLane & av:AV-TurLft(other) -> Car:VehLane", p 0.40, 71 onsets, in 2 of 6 runs; a Ped:RhtPav rule with
# p 0.93, 273 onsets, lost in v6). The "exh" miner keeps a rule by its own counts, never by a rank among others:
#   1. every rule of 1 or 2 literals (all pairs at once: two matrix products on the head's train sample) with
#      >= lab_min_events distinct onsets in >= lab_min_videos videos and precision >= lab_exh_lift x the head's
#      sample rate; a pair only if its precision is >= lab_exh_gain x that of each of its literals (else the pair
#      says nothing more). Adding literals adds candidates and never changes a pair's counts: these rules stay;
#   2. every such rule, and every pair with lift >= lab_exh_seed_lift (extendable), is a seed grown on its own:
#      one literal at a time, the child of best pessimistic precision (hits / (rows + one negative)) among those
#      keeping >= lab_min_events onsets and raising the precision by lab_exh_gain; each step that qualifies (lift)
#      is kept. Seeds do not compete for a beam;
#   3. a per-head cap lab_exh_max (by pessimistic precision) only as a safeguard; reported when it binds. The
#      extension (2) starts from the lab_exh_seeds best pairs (best effort); the short rules of 1 hold always.
EXH_CHUNK = 4096


def _pair_events(pool, a, b, hit_rows):
    """(P,) distinct onsets of the hit rows where literals a[p] and b[p] both hold."""
    ev = pool["event"][hit_rows]
    uniq, inv = ev.unique(return_inverse=True)
    Fh = pool["F"][hit_rows] > 0.5                                             # (H, L)
    out = []
    for i in range(0, len(a), EXH_CHUNK):
        both = (Fh[:, a[i:i + EXH_CHUNK]] & Fh[:, b[i:i + EXH_CHUNK]]).float()  # (H, P)
        per = torch.zeros((len(uniq), both.shape[1]), device=both.device).index_add_(0, inv, both)
        out.append((per > 0).sum(dim=0))
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long, device=Fh.device)


def _check_ok(ev_train, rows_c, hits_c, lift_c):
    """`check_filter`'s test on tensors: >= 1 check hit row and a check lift >= 1."""
    return (rows_c >= 1) & (hits_c >= 1) & (lift_c >= 1.0)


WILSON_Z = 1.96            # 95% (two-sided) Wilson bound of the significance tests below


def wilson_lb(p, n_eff, z=WILSON_Z):
    """Lower Wilson bound of a precision p seen on n_eff (effective) rows; tensors."""
    n = n_eff.clamp(min=1e-9)
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    return (centre - z * torch.sqrt((p * (1 - p) / n + z * z / (4 * n * n)).clamp(min=0))) / den


def exhaustive_rules(pool, allowed, cfg, check=None):
    """The "exh" rules of one head (see above): [(literals tuple, sample precision, onsets)] and the number of
    rules over the cap. check: the head's pool of the check videos (`--lab_check`): the short rules must pass
    `check_filter`'s test there already (2026-11-24: without it 8 heads gave 263k short rules, mostly chance
    pairs of 2-4 onsets)."""
    F, y, w = pool["F"], pool["y"], pool["w"]
    n = len(y)
    if n == 0 or not bool(y.any()):
        return [], 0
    dev = F.device
    yf = y.float()
    wy = w * yf
    rate = float(wy.sum() / w.sum().clamp(min=1e-12))
    lift, seed_lift, gain = cfg["lab_exh_lift"], cfg["lab_exh_seed_lift"], cfg["lab_exh_gain"]
    min_ev, min_vid = cfg["lab_min_events"], cfg["lab_min_videos"]
    lam = float(w[~y].median()) if bool((~y).any()) else 1.0
    hit_rows = torch.nonzero(y).flatten()
    ev1 = step05_mine._distinct(pool["event"][hit_rows], F[hit_rows])         # (L,) onsets per literal
    vid1 = step05_mine._distinct(pool["video"][hit_rows], F[hit_rows])
    n1, h1, s1 = w @ F, wy @ F, (w * w) @ F
    p1 = h1 / n1.clamp(min=1e-12)
    # Significance (2026-11-24, 118k rules of >= 5 onsets for 8 heads ran the fit out of memory): the weighted
    # sample's effective rows n_eff = (sum w)^2 / sum w^2; a literal must beat the head's rate, a pair both its
    # literals, an extension its parent, at the lower 95% Wilson bound. Each test reads the rule's own counts only.
    lb1 = wilson_lb(p1, n1 * n1 / s1.clamp(min=1e-12))
    cand = torch.nonzero(allowed & (ev1 >= min_ev) & (vid1 >= min_vid) & (n1 > 0)).flatten()
    kept = {}
    if check is not None and len(check["y"]):
        Fk, yk, wk = check["F"], check["y"].float(), check["w"]
        rate_c = float((wk * yk).sum() / wk.sum().clamp(min=1e-12))
        n1c, h1c, r1c, c1c = wk @ Fk, (wk * yk) @ Fk, Fk.sum(dim=0), yk @ Fk
        lift1c = (h1c / n1c.clamp(min=1e-12)) / max(rate_c, 1e-12)
        ok1 = _check_ok(ev1, r1c, c1c, lift1c)
    else:
        Fk = None
        ok1 = torch.ones_like(ev1, dtype=torch.bool)
    for m in cand[(p1[cand] >= lift * rate) & ok1[cand] & (lb1[cand] > rate)].tolist():
        kept[(m,)] = (float(p1[m]), int(ev1[m]), float(h1[m] / (n1[m] + lam)))
    # 1. All pairs of candidate literals.
    Fc = F[:, cand]
    n2 = (Fc * w[:, None]).T @ Fc                                              # (c, c) weighted rows
    h2 = (Fc * wy[:, None]).T @ Fc
    r2 = (Fc * yf[:, None]).T @ Fc                                             # hit rows (unweighted)
    p2 = h2 / n2.clamp(min=1e-12)
    s2 = (Fc * (w * w)[:, None]).T @ Fc
    lb2 = wilson_lb(p2, n2 * n2 / s2.clamp(min=1e-12))
    pc = p1[cand]
    upper = torch.triu(torch.ones_like(p2, dtype=torch.bool), diagonal=1)
    best_part = torch.maximum(pc[:, None], pc[None, :])
    useful = (p2 >= gain * best_part) & (lb2 > best_part) & (r2 >= min_ev) & (n2 > 0) & upper
    del s2, lb2
    seedable = useful & (p2 >= seed_lift * rate)
    ia, ib = torch.nonzero(seedable, as_tuple=True)
    score = h2[ia, ib] / (n2[ia, ib] + lam)
    order = torch.argsort(score, descending=True)                              # deterministic: by own counts
    ia, ib, score = ia[order], ib[order], score[order]
    ev2 = _pair_events(pool, cand[ia], cand[ib], hit_rows)
    ok = ev2 >= min_ev
    if Fk is not None:                                                         # the pairs on the check videos
        Kc = Fk[:, cand]
        n2c = ((Kc * wk[:, None]).T @ Kc)[ia, ib]
        h2c = ((Kc * (wk * yk)[:, None]).T @ Kc)[ia, ib]
        r2c = (Kc.T @ Kc)[ia, ib]
        c2c = ((Kc * yk[:, None]).T @ Kc)[ia, ib]
        ok &= _check_ok(ev2, r2c, c2c, (h2c / n2c.clamp(min=1e-12)) / max(rate_c, 1e-12))
    ia, ib, score, ev2 = ia[ok], ib[ok], score[ok], ev2[ok]
    pp = p2[ia, ib]
    a, b = cand[ia].tolist(), cand[ib].tolist()
    qual = (pp >= lift * rate).tolist()
    for k, (x, z, q) in enumerate(zip(a, b, qual)):
        if q:
            kept[(min(x, z), max(x, z))] = (float(pp[k]), int(ev2[k]), float(score[k]))
    # Seeds: the pairs not yet at precision 1, best first (the extension is best effort; the pairs above stay).
    open_ = (pp < 1.0 - 1e-9).tolist()
    seeds = [(min(x, z), max(x, z)) for x, z, o in zip(a, b, open_) if o][:cfg["lab_exh_seeds"]]
    seeds += [key for key in kept if len(key) == 1 and kept[key][0] < 1.0 - 1e-9]
    # 2. Every seed grown on its own (no shared beam), batched.
    for i in range(0, len(seeds), 256):
        batch = seeds[i:i + 256]
        masks = torch.stack([F[:, list(key)].gt(0.5).all(dim=1) for key in batch])          # (S, n)
        precs = torch.tensor([float(((masks[s].float() * wy).sum()) / (masks[s].float() * w).sum().clamp(min=1e-12))
                              for s in range(len(batch))], device=dev)
        keys = list(batch)
        for _depth in range(max(0, cfg["lab_max_len"] - min(len(k) for k in keys))):
            M = masks.float()
            nn_ = (M * w) @ F
            hh = (M * wy) @ F
            cc = (M * yf) @ F
            pr = hh / nn_.clamp(min=1e-12)
            sc = hh / (nn_ + lam)
            lbc = wilson_lb(pr, nn_ * nn_ / ((M * (w * w)) @ F).clamp(min=1e-12))
            okc = (allowed[None] & (cc >= min_ev) & (pr >= gain * precs[:, None]) & (lbc > precs[:, None]) & (nn_ > 0)
                   & ((M @ F) < M.sum(dim=1, keepdim=True)))
            for s, key in enumerate(keys):
                okc[s, list(key)] = False
            sc = torch.where(okc, sc, torch.full_like(sc, -1.0))
            best_s, best_m = sc.max(dim=1)
            alive = []
            for s in range(len(keys)):
                if float(best_s[s]) <= 0 or len(keys[s]) >= cfg["lab_max_len"]:
                    continue
                m = int(best_m[s])
                child = masks[s] & (F[:, m] > 0.5)
                hit = child & y
                ev = int(pool["event"][hit].unique().numel())
                if ev < min_ev or int(pool["video"][hit].unique().numel()) < min_vid:
                    continue
                key = tuple(sorted(keys[s] + (m,)))
                p = float(pr[s, m])
                if p >= lift * rate and key not in kept:
                    kept[key] = (p, ev, float(sc[s, m]))
                if p < 1.0 - 1e-9:
                    alive.append((s, key, child, p))
            if not alive:
                break
            masks = torch.stack([c for _, _, c, _ in alive])
            precs = torch.tensor([p for _, _, _, p in alive], device=dev)
            keys = [k for _, k, _, _ in alive]
    # The cap binds the extended rules only: the rules of 1 - 2 literals all stay (the guarantee; the A/B of
    # 2026-11-24 lost 23 short rules to a cap over all rules).
    short = [(k, v) for k, v in kept.items() if len(k) <= 2]
    longer = sorted(((k, v) for k, v in kept.items() if len(k) > 2), key=lambda x: -x[1][2])
    room = max(0, cfg["lab_exh_max"] - len(short))
    over = max(0, len(longer) - room)
    ranked = sorted(short + longer[:room], key=lambda x: -x[1][2])
    return [(key, p, e) for key, (p, e, _s) in ranked], over


# -------------- Transfer from data-rich classes and adaptation ("tr", 2026-11-24) --------------
# The user: heads with few samples (Bus:VehLane, 24 train onsets) should borrow the rules of the same action of
# other classes (Car:VehLane, 596 onset rows) and adapt them to their own samples (a bus box is larger, ...).
#   1. targets: heads with < lab_tr_max_onsets train onsets; sources: the same action of the other vehicle classes
#      with >= lab_tr_min_source onsets (the two largest);
#   2. the source's "exh" rules (by its own train sample and check videos), the lab_tr_rules best;
#   3. mapped: the source class's own label literals ("Car:OutgoLane(own)") become the target's ("Bus:OutgoLane");
#      shape atoms are per-class quantiles already ("shp:tall" of a bus is tall for a bus), the rest stays;
#   4. adapted on the target's train videos of even index: a local search that keeps the change of best
#      pessimistic precision (>= lab_tr_min_events onsets): a literal in the slot before / after, the neighbouring
#      position cell or direction sector (+-5, +-10 deg), a literal dropped, one significant literal added;
#   5. confirmed on the target's odd train videos (>= 1 onset, lift >= 1); --lab_check then asks the check videos.
# The source's train onsets carry the rule (>= lab_min_events there); the target needs lab_tr_min_events.
TR_STEPS = 4               # improvements of the local search per rule
TR_ANGLES = (-10, -5, 5, 10)


def mirror_text(nm):
    """Lft <-> Rht, left <-> right in a name (the head "Car:TurLft" -> "Car:TurRht"; `LiteralMap.mirror_name`)."""
    import re
    swap = {"Lft": "Rht", "Rht": "Lft", "left": "right", "right": "left", "Left": "Right", "Right": "Left"}
    return re.sub(r"Lft|Rht|left|right|Left|Right", lambda x: swap[x.group(0)], nm)


def _class_actions(names, heads):
    """{head: (class, action)} of the agent label heads ("Car:VehLane" -> ("Car", "VehLane"))."""
    out = {}
    for h in heads:
        cls, _, act = names[h].partition(":")
        if act and cls not in ("av", "TL", "OthTL", "shp", "gst", "pos"):
            out[h] = (cls, act)
    return out


def transfer_sources(names, heads, all_heads, train, cfg):
    """{target head: [source heads]}: targets with few train onsets, sources the same action of other vehicle
    classes with many (`--lab_tr_*`)."""
    ca = _class_actions(names, all_heads)
    onsets = {}
    for h in ca:
        sel = (train["pairs"]["head"] == h) & train["pairs"]["y"].bool()
        onsets[h] = int(train["pairs"]["event"][sel].unique().numel())
    out, mirror = {}, {}
    for h in heads:
        if h not in ca or onsets.get(h, 0) >= cfg["lab_tr_max_onsets"]:
            continue
        cls, act = ca[h]
        if cls == "Ped":
            continue
        # sources: other classes with more onsets than the target (2026-11-25: with --lab_tr_max_onsets raised,
        # mid-size heads such as MedVeh:OutgoLane are targets too, and must not borrow from smaller classes)
        src = [g for g, (c, a) in ca.items() if a == act and c not in (cls, "Ped") and onsets[g] >= cfg["lab_tr_min_source"]
               and onsets[g] > onsets[h]]
        src = sorted(src, key=lambda g: -onsets[g])[:2]
        m_act = mirror_text(act)
        if m_act != act and cfg.get("lab_tr_mirror"):
            # A directional action (TurRht): its mirror is the mirrored action's rules (TurLft) of any vehicle class,
            # the target's own class too, mirrored (2026-11-24: mirroring a TurRht rule for TurRht was wrong).
            ms = [g for g, (c, a) in ca.items() if a == m_act and c != "Ped" and onsets[g] >= cfg["lab_tr_min_source"]]
            ms = sorted(ms, key=lambda g: -onsets[g])[:2]
            if ms:
                mirror[h] = ms
        if src or h in mirror:
            out[h] = src
    transfer_sources.mirror = mirror
    return out, onsets


class LiteralMap:
    """Literal arithmetic on the rule net's own literals (`literal_table`): the class swap of own label literals,
    and the siblings of a literal (other slot, neighbouring position cell, close direction sector)."""

    def __init__(self, model, names):
        import re
        self.re = re
        self.model = model
        self.tab = literal_table(model)
        self.names = names
        self.atom_of = {nm: a for a, nm in enumerate(names)}
        self.index, self.gindex = {}, {}
        for m, info in enumerate(self.tab):
            if info["kind"] in ("X", "!X"):
                self.index[(info["kind"], info["atom"], info["scope"], info["j"])] = m
            else:
                self.gindex[(info["kind"], info["group"], info["scope"], info["j"])] = m

    def _lit(self, kind, atom, scope, j):
        return self.index.get((kind, atom, scope, j))

    def swap_class(self, m, src, dst):
        """The literal with the own label atom "src:act" made "dst:act"; m itself if not such; None if missing."""
        if m >= len(self.tab):
            return None                                                     # derived literals: not mapped
        info = self.tab[m]
        if info["kind"] not in ("X", "!X") or info["scope"] != 0:
            return m
        nm = self.names[info["atom"]]
        cls, _, act = nm.partition(":")
        if cls != src:
            return m
        a = self.atom_of.get(f"{dst}:{act}")
        return None if a is None else self._lit(info["kind"], a, 0, info["j"])

    def mirror_name(self, nm):
        """The left / right mirror of an atom or group name (2026-11-24, the user: a cut-in from the left is the
        mirror of one from the right): Lft <-> Rht, left <-> right; the position grid column c -> 2 - c (the
        corners keep their names: an approximation for a box); a half circle dir_θ / path_θ ([θ, θ + 180) of
        the image heading, 0 = right, counter-clockwise) -> its mirror [-θ, -θ + 180)."""
        re_ = self.re
        mt = re_.fullmatch(r"(pos:(?:TL|BR)_r\d)c(\d)", nm)
        if mt:
            return f"{mt.group(1)}c{2 - int(mt.group(2))}"
        mt = re_.fullmatch(r"(.*(?:dir|path))_(\d{3})", nm)
        if mt:
            return f"{mt.group(1)}_{(-int(mt.group(2))) % 360:03d}"
        return mirror_text(nm)

    def mirror(self, m):
        """The mirrored literal of m (`mirror_name`), m itself if nothing to mirror, None if the mirror is missing."""
        if m >= len(self.tab):
            return None
        info = self.tab[m]
        if info["kind"] in ("X", "!X"):
            nm = self.names[info["atom"]]
            mn = self.mirror_name(nm)
            if mn == nm:
                return m
            a = self.atom_of.get(mn)
            return None if a is None else self._lit(info["kind"], a, info["scope"], info["j"])
        g = info["group"]
        mg = self.mirror_name(g) if isinstance(g, str) else g
        return m if mg == g else self.gindex.get((info["kind"], mg, info["scope"], info["j"]))

    def mirror_rule(self, lits):
        """The mirrored rule, None if a literal has no mirror or the rule has nothing to mirror."""
        out = [self.mirror(m) for m in lits]
        if any(x is None for x in out) or sorted(out) == sorted(lits):
            return None
        return out

    def siblings(self, m):
        if m >= len(self.tab):
            return []
        info = self.tab[m]
        if info["kind"] not in ("X", "!X"):
            return []
        k, a, sc, j = info["kind"], info["atom"], info["scope"], info["j"]
        out = [self._lit(k, a, sc, jj) for jj in (j - 1, j + 1)]
        nm = self.names[a]
        mt = self.re.fullmatch(r"(pos:(?:TL|BR))_r(\d)c(\d)", nm)
        if mt:
            r, c = int(mt.group(2)), int(mt.group(3))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                b = self.atom_of.get(f"{mt.group(1)}_r{r + dr}c{c + dc}")
                if b is not None:
                    out.append(self._lit(k, b, sc, j))
        mt = self.re.fullmatch(r"(.*(?:dir|path))_(\d{3})", nm)
        if mt:
            for d in TR_ANGLES:
                b = self.atom_of.get(f"{mt.group(1)}_{(int(mt.group(2)) + d) % 360:03d}")
                if b is not None:
                    out.append(self._lit(k, b, sc, j))
        return [x for x in out if x is not None]


def _rule_stats(pool, rows, lits, rate=None):
    """(weighted precision, score, onsets) of a rule on the pool rows `rows` (bool). score: the FOIL gain
    hits x log(precision / rate) (coverage and precision traded off: a rule twice as wide at a slightly lower
    precision wins; 2026-11-24 the pessimistic precision kept a needless literal), or the pessimistic precision
    without a rate."""
    hold = rows & (pool["F"][:, list(lits)] > 0.5).all(dim=1)
    w, y = pool["w"], pool["y"]
    f = float(w[hold].sum())
    h = float(w[hold & y].sum())
    p = h / f if f > 0 else 0.0
    if rate is not None:
        score = h * math.log(max(p, 1e-9) / max(rate, 1e-9)) if h > 0 else -1e9
    else:
        lam = float(w[rows & ~y].median()) if bool((rows & ~y).any()) else 1.0
        score = h / (f + lam)
    return p, score, int(pool["event"][hold & y].unique().numel())


def adapt_rule(pool, rows, lits, lmap, allowed, cfg):
    """The local search of the transfer (see above) on the pool rows `rows`, by FOIL gain over the rows' rate:
    returns the adapted literals."""
    best = tuple(sorted(set(lits)))
    w_r, y_r = pool["w"][rows], pool["y"][rows]
    rate = float((w_r * y_r).sum() / w_r.sum().clamp(min=1e-12))
    p0, s0, _e0 = _rule_stats(pool, rows, best, rate)
    min_ev = cfg["lab_tr_min_events"]
    F, w, y = pool["F"], pool["w"], pool["y"]
    for _ in range(TR_STEPS):
        cands = []
        for k, m in enumerate(best):
            rest = best[:k] + best[k + 1:]
            if rest:
                cands.append(rest)                                         # drop a literal
            for s in lmap.siblings(m):
                if bool(allowed[s]):
                    cands.append(tuple(sorted(set(rest + (s,)))))          # a sibling instead
        # one literal added: the significant best (as the exh extension)
        hold = rows & (F[:, list(best)] > 0.5).all(dim=1)
        M = hold.float()
        nn_ = (M * w) @ F
        hh = (M * w * y.float()) @ F
        cc = (M * y.float()) @ F
        pr = hh / nn_.clamp(min=1e-12)
        lb = wilson_lb(pr, nn_ * nn_ / ((M * w * w) @ F).clamp(min=1e-12))
        ok = allowed & (cc >= min_ev) & (lb > p0)
        ok[list(best)] = False
        if bool(ok.any()):
            foil = hh * torch.log(pr.clamp(min=1e-9) / max(rate, 1e-9))
            m = int(torch.where(ok, foil, torch.full_like(hh, -1e9)).argmax())
            cands.append(tuple(sorted(best + (m,))))
        top = None
        for c in set(cands):
            p, sc, e = _rule_stats(pool, rows, c, rate)
            if e >= min_ev and p > rate and sc > s0 + 1e-6 and (top is None or sc > top[1]):
                top = (c, sc, p, e)
        if top is None:
            break
        best, s0, p0 = top[0], top[1], top[2]
    return best


def _map_index(lmap, L, src, dst):
    """(L,) long: per literal of the target class dst the source column (-1: always false, -2: always true for a
    dst agent), so a source pool's rows read as rows of the target class ("pooled" transfer)."""
    idx = torch.arange(L)
    for t in range(L):
        if t >= len(lmap.tab):
            idx[t] = -1                                                     # derived literals: not mapped
            continue
        info = lmap.tab[t]
        if info["kind"] in ("X", "!X") and info["scope"] == 0:
            cls, _, act = lmap.names[info["atom"]].partition(":")
            if cls == src:
                idx[t] = -1 if info["kind"] == "X" else -2                  # src's own atoms never hold for dst
            elif cls == dst:
                s = lmap.swap_class(t, dst, src)
                idx[t] = -1 if s is None else s
    return idx


def _mapped_pool(pool, idx, offset):
    """A source pool in the target's literal space (`_map_index`), its onsets and videos offset."""
    dev = pool["F"].device
    i = idx.to(dev)
    F = pool["F"][:, i.clamp(min=0)]
    F[:, i == -1] = 0.0
    F[:, i == -2] = 1.0                                                     # a negated src-own literal: always true
    return {"F": F, "y": pool["y"], "w": pool["w"], "event": pool["event"] + offset, "video": pool["video"] + offset}


def _cat_pools(pools):
    return {k: torch.cat([p[k] for p in pools]) for k in ("F", "y", "w", "event", "video")}


def _foil_rank(found, pool):
    """found [(literals, precision, onsets)] best by FOIL gain (onsets x log(precision / rate)): wide rules first,
    which transfer better than narrow ones (2026-11-24)."""
    rate = float((pool["w"] * pool["y"]).sum() / pool["w"].sum().clamp(min=1e-12))
    return sorted(found, key=lambda x: -(x[2] * math.log(max(x[1], 1e-9) / max(rate, 1e-9))))


def _confirm(tgt, odd, tck, lits, rate, need):
    """Held-out onsets of a rule on the target (odd train videos + check videos) and whether it is kept: >= need
    onsets (2026-11-24, the user: with few samples a hit counts; two hits are enough) and a held-out precision >=
    the target's rate."""
    p_o, _s, e_o = _rule_stats(tgt, odd, lits)
    hold_o = odd & (tgt["F"][:, list(lits)] > 0.5).all(dim=1)
    f, h = float(tgt["w"][hold_o].sum()), float(tgt["w"][hold_o & tgt["y"]].sum())
    e = e_o
    if tck is not None and len(tck["y"]):
        hold_c = (tck["F"][:, list(lits)] > 0.5).all(dim=1)
        e += int(tck["event"][hold_c & tck["y"]].unique().numel())
        rate_c = float((tck["w"] * tck["y"]).sum() / tck["w"].sum().clamp(min=1e-12))
        # check weights differ from train weights: scale the check rows to the train rate
        sc = rate / max(rate_c, 1e-12)
        f += float(tck["w"][hold_c].sum())
        h += float(tck["w"][hold_c & tck["y"]].sum()) * sc
    return e, e >= need and f > 0 and h / f >= rate


def transfer_rules(ext, names, heads, all_heads, train, check, cfg, device, pooled=False):
    """The "tr" (per source) or "trp" (pooled) rules of the target heads (see above)."""
    base_model = getattr(ext, "_model", ext)
    lmap = LiteralMap(base_model, names)
    L, A = ext.num_literals, ext.num_atoms
    base = ext.literal_base.to(device)
    scope = base // (ext.num_spans * A)
    tag = "trp" if pooled else "tr"
    targets, onsets = transfer_sources(names, heads, all_heads, train, cfg)
    print(f"[Rule lab] {tag}: {len(targets)} target heads: "
          + ", ".join(f"{names[h]} ({onsets.get(h, 0)}) <- {'/'.join(names[g] for g in src)}" for h, src in targets.items()))
    rules = []
    for h, sources in tqdm(targets.items(), desc="Transfer" + (" (pooled)" if pooled else "")):
        allowed = ext.input_mask.to(device) & ~((base % A == h) & (scope == 0))
        tgt = step05_mine._pool_of(train, h, L, device)
        if not bool(tgt["y"].any()):
            continue
        rate = float((tgt["w"] * tgt["y"]).sum() / tgt["w"].sum().clamp(min=1e-12))
        even = tgt["video"] % 2 == 0
        tck = step05_mine._pool_of(check, h, L, device) if bool((check["pairs"]["head"] == h).any()) else None
        dst = names[h].partition(":")[0]
        seen, tally = set(), Counter()
        batches = []                                       # [(source name, [(literals, src class)])]
        directional = mirror_text(names[h]) != names[h]
        m_sources = transfer_sources.mirror.get(h, []) if directional else []
        if pooled and sources:
            parts = [{k: v[even] for k, v in tgt.items()}]
            for k, g in enumerate(sources):
                src_pool = step05_mine._pool_of(train, g, L, device)
                parts.append(_mapped_pool(src_pool, _map_index(lmap, L, names[g].partition(":")[0], dst),
                                          (k + 1) * 10 ** 9))
                del src_pool
            pool = _cat_pools(parts)
            del parts
            found, _ = exhaustive_rules(pool, allowed, cfg, None)
            tally["pooled rows"] = len(pool["y"])
            ranked = _foil_rank(found, pool)[:cfg["lab_tr_rules"]]
            del pool
            batches.append(("+".join(names[g] for g in sources), [(lits, None) for lits, _p, _e in ranked], False))
            src_back = names[sources[0]].partition(":")[0]          # the main source class, for "source_literals"
            tally["pooled rules"] += len(found)
        for g in ([] if pooled else sources) + m_sources:
            src_pool = step05_mine._pool_of(train, g, L, device)
            ck = step05_mine._pool_of(check, g, L, device) if cfg.get("lab_check") and \
                bool((check["pairs"]["head"] == g).any()) else None
            src_allowed = ext.input_mask.to(device) & ~((base % A == g) & (scope == 0))
            found, _ = exhaustive_rules(src_pool, src_allowed, cfg, ck)
            ranked = _foil_rank(found, src_pool)[:cfg["lab_tr_rules"]]
            del src_pool, ck
            tally["source rules"] += len(found)
            batches.append((names[g], [(lits, names[g].partition(":")[0]) for lits, _p, _e in ranked],
                            g in m_sources))
        for src_name, items, from_mirror in batches:
            for lits, src_cls in items:
                tally["tried"] += 1
                mapped = list(lits) if src_cls is None else [lmap.swap_class(m, src_cls, dst) for m in lits]
                if any(m is None or not bool(allowed[m]) for m in mapped):
                    tally["not mappable"] += 1
                    continue
                # A directional head takes mirrored rules only from the mirrored action (from_mirror), and those only
                # mirrored; a direction-free head (VehLane, Jun) also tries each rule mirrored.
                cands = [] if from_mirror else [(mapped, False)]
                if cfg.get("lab_tr_mirror") and (from_mirror or not directional):
                    mir = lmap.mirror_rule(mapped)
                    if from_mirror and mir is None and all(lmap.mirror(m) == m for m in mapped):
                        mir = mapped                     # a symmetric body predicts the mirrored action as well
                    if mir is not None and all(bool(allowed[m]) for m in mir):
                        cands.append((mir, True))
                        tally["mirrored tried"] += 1
                for body, mirrored in cands:
                    _p_m, _s, e_m = _rule_stats(tgt, even, body)
                    tally["with >= 1 target onset"] += e_m >= 1
                    adapted = adapt_rule(tgt, even, body, lmap, allowed, cfg)
                    tally["changed by adaptation"] += tuple(sorted(set(body))) != adapted
                    if adapted in seen:
                        continue
                    seen.add(adapted)
                    e_c, ok = _confirm(tgt, ~even, tck, adapted, rate, cfg["lab_tr_confirm"])
                    if not ok:
                        tally["not confirmed"] += 1
                        continue
                    tally["kept mirrored" if mirrored else "kept"] += 1
                    if src_cls is None:                              # pooled: the rule read back in the main source
                        src_lits = [lmap.swap_class(m, dst, src_back) for m in lits]
                        src_head, src_lits = names[sources[0]], [m for m in src_lits if m is not None]
                    else:
                        src_head, src_lits = src_name, list(lits)
                    rules.append({"head": int(h), "literals": list(adapted),
                                  "transfer": f"{tag}{'-mirror' if mirrored else ''}:{src_name}",
                                  "min_events": cfg["lab_tr_min_events"], "heldout_events": e_c,
                                  "source_head": src_head, "source_literals": src_lits, "mirrored": mirrored})
        print(f"    {names[h]}: rate {rate:.3f}, train onsets even/odd videos "
              f"{int(tgt['event'][tgt['y'] & even].unique().numel())}/{int(tgt['event'][tgt['y'] & ~even].unique().numel())}"
              f", check onsets {int(tck['event'][tck['y']].unique().numel()) if tck is not None else 0}; "
              + ", ".join(f"{k} {v}" for k, v in tally.items()))
        del tgt, tck
    print(f"[Rule lab] {tag}: {len(rules)} rules (confirmed: >= {cfg['lab_tr_confirm']} held-out onsets on the odd "
          f"train and check videos)")
    return rules


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


def sample_counts(rules, train, L, device, A, chunk=512):
    """`recount` from the train sample (`--lab_fast`): per rule the weighted rows and hits where it holds, its
    onsets in the sample and its (unweighted) rows; per head the weighted base rate. Vectorized per head over
    chunks of rules (2026-11-24: a loop over 42k rules was the slowest step of the exh runs)."""
    fires, hits = torch.zeros(len(rules)), torch.zeros(len(rules))
    events, rows = torch.zeros(len(rules), dtype=torch.long), torch.zeros(len(rules), dtype=torch.long)
    base = torch.zeros(A)
    by = defaultdict(list)
    for i, r in enumerate(rules):
        by[r["head"]].append(i)
    for h, idx in sorted(by.items()):
        if not bool((train["pairs"]["head"] == h).any()):
            continue
        pool = step05_mine._pool_of(train, h, L, device)
        F = torch.cat([pool["F"] > 0.5, torch.ones((len(pool["y"]), 1), dtype=torch.bool, device=device)], dim=1)
        w, y = pool["w"], pool["y"]
        base[h] = float((w * y).sum() / w.sum().clamp(min=1e-9))
        hit_rows = torch.nonzero(y).flatten()
        uniq, inv = pool["event"][hit_rows].unique(return_inverse=True)
        for c in range(0, len(idx), chunk):
            ids = idx[c:c + chunk]
            body = _pad([rules[i]["literals"] for i in ids], L).to(device)        # (R, K), pad = column L (true)
            hold = _holds_all(F, body)                                            # (n, R)
            hf = hold.float()
            t = torch.tensor(ids)
            fires[t] = (w @ hf).cpu()
            hits[t] = ((w * y) @ hf).cpu()
            rows[t] = hold.sum(dim=0).cpu()
            per = torch.zeros((len(uniq), len(ids)), device=device).index_add_(0, inv, hf[hit_rows])
            events[t] = (per > 0).sum(dim=0).cpu()
        del pool, F
    return {"fires": fires, "hits": hits, "events": events, "rows": rows, "base": base}


# -------------- Check on held-out videos and merging (2026-11-24) --------------
# The user: a rule that does not hold on the validation split should go before the fit; rules that differ only in
# a parameter should merge. relaxed+exh had 42k rules, 32k of them with < 5 train onsets (22k at precision 1):
# chance co-occurrences that the soft fit weighed high (small heads lost up to 0.47 AP).
# 2026-11-24, the user: a rule must stand on the train split by itself (--lab_min_events, default 5 onsets) and
# show up on the validation split too (and on test, which is never used to select): it holds on the check videos
# with >= 1 hit row and a lift >= 1. A rule that never fires there, or fires without beating the base rate, goes.


def check_filter(rules, ck, index):
    """The rules that hold on the check videos (the even val videos; the odd ones stay for the evaluation): >= 1
    check onset and a check precision >= the head's check rate (lift >= 1). ck: `sample_counts` of the check
    sample; index: (head, sorted literals) -> its row in ck. Each rule kept gets "check_events" and
    "check_lift"."""
    out = []
    for r in rules:
        if "heldout_events" in r:                         # transferred: confirmed on odd train + check videos
            out.append(r)
            continue
        i = index[(r["head"], tuple(sorted(r["literals"])))]
        f, h, e, n = float(ck["fires"][i]), float(ck["hits"][i]), int(ck["events"][i]), int(ck["rows"][i])
        rate = float(ck["base"][r["head"]])
        lift = (h / f) / rate if f > 0 and rate > 0 else None
        if e < 1 or lift is None or lift < 1.0 or n < 1:
            continue
        out.append({**r, "check_events": e, "check_lift": round(lift, 3) if lift is not None else None})
    return out


def merge_classes(rules, model, names, pool_of, ratio):
    """Rules on another agent's class-specific label ("Car:TurLft(other)@1") made class-free with the group
    literal of the same scope and slot ("veh:TurLft(other)@1", `TemporalRuleNet` groups) if the general rule keeps
    >= ratio x the precision and at least the onsets (train sample of the head). All such literals at once, else the
    one replacement with the most onsets. Duplicates merge. Returns the rules and the number generalized."""
    gmap = {}
    tab = literal_table(model)
    for m, info in enumerate(tab):
        if info["kind"] == "G":
            gmap[(info["scope"], info["j"], info["group"])] = m
    if not gmap:
        return rules, 0

    def general(m):
        if m >= model.num_literals:
            return None
        info = tab[m]
        if info["kind"] != "X" or info["scope"] == 0:
            return None
        cls, _, act = names[info["atom"]].partition(":")
        if not act or cls in ("av", "TL", "OthTL", "Ped", "shp", "gst", "pos"):
            return None
        return gmap.get((info["scope"], info["j"], act))

    def stats(pool, lits):
        hold = (pool["F"][:, lits] > 0.5).all(dim=1)
        f = float(pool["w"][hold].sum())
        hit = hold & pool["y"]
        return (float(pool["w"][hit].sum()) / f if f > 0 else 0.0), int(pool["event"][hit].unique().numel()), f
    out, seen, n_gen = [], set(), 0
    by_head = defaultdict(list)
    for r in rules:
        by_head[r["head"]].append(r)
    for h, rs in sorted(by_head.items()):
        pool = None
        for r in rs:
            subs = [(k, general(m)) for k, m in enumerate(r["literals"])]
            subs = [(k, g) for k, g in subs if g is not None]
            best = None
            if subs:
                pool = pool if pool is not None else pool_of(h)
                p0, e0, _ = stats(pool, r["literals"])
                trials = [subs] + ([[x] for x in subs] if len(subs) > 1 else [])
                for t in trials:
                    lits = list(r["literals"])
                    for k, g in t:
                        lits[k] = g
                    lits = sorted(set(lits))
                    p, e, f = stats(pool, lits)
                    if p >= ratio * p0 and e >= e0 and (best is None or e > best[1]):
                        best = (lits, e, p, f)
            r2 = r if best is None else {**r, "literals": best[0], "events": best[1], "precision": best[2],
                                         "fires": best[3], "merged_from": "class"}
            key = (r2["head"], tuple(sorted(r2["literals"])))
            if key in seen:
                continue
            seen.add(key)
            n_gen += best is not None
            out.append(r2)
    return out, n_gen


def finalize(rules, counts, min_events):
    """The rules with exact precision and onsets: >= min_events onsets; a rule is dropped if one of its
    sub-rules (fewer literals, same head) is at least as precise (the extension adds nothing)."""
    out = []
    for i, r in enumerate(rules):
        f, h, e = float(counts["fires"][i]), float(counts["hits"][i]), int(counts["events"][i])
        if f <= 0 or e < r.get("min_events", min_events):          # "tr" rules: their own minimum (target)
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
# The 360-degree half circle atoms: the own box's move / path (`shape.py`) and those of the agent's own / the
# nearest group's box (`gestalt.py` grp_ / ngrp_); group 1 the kind ("shp:dir", "gst:grp_path", ..), group 2 θ.
DIRECTION_RE = r"((?:shp|gst):(?:(?:grp_|ngrp_)?(?:dir|path)))_(\d{3})"

def direction_literals(model, names):
    """{literal: (kind "dir" | "path", θ, scope, j)} of the 360-degree half circle literals (`shape.py`), and
    {(kind, θ, scope, j): literal}."""
    import re
    of, back = {}, {}
    for m, info in enumerate(literal_table(model)):
        if info["kind"] != "X":
            continue
        mt = re.fullmatch(DIRECTION_RE, names[info["atom"]])
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


def refine_sectors(rules, model, names, pool_of, min_events=2, min_lift=1.2, max_width=180, forbid=None):
    """Narrow each rule's direction to where its onsets concentrate (the user, 2026-11-23: find the dense region
    inside the interval and shrink it). Per rule and direction kind (dir, path; own scope, slot 1):
      - every row's angle bin (ANGLE_STEP degrees) is read from the half circle literals (the last θ whose half
        circle holds before the next one does not);
      - on the rows where the rule holds, every arc inside the rule's sector (or anywhere, for a rule without
        that kind) is scored by FOIL gain: weighted hits x (log precision of the arc - log precision of the rule);
      - the best arc with >= min_events onsets and precision >= min_lift x the rule's becomes a new rule: the
        rule's other literals and the arc as two half circles.
    The new rules are added; the soft fit (`fit_soft`) weighs them against the old ones. Returns the rules and
    the number added. forbid (L,) bool: the variant's forbidden literals; a kind with any of them is left out (bug
    fix 2026-11-23: the _ng variants got "gst:grp_dir" arcs here)."""
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
    for head, rs in sorted(by_head.items()):
        pool = pool_of(head)
        F = pool["F"] > 0.5
        w, y = pool["w"], pool["y"]
        for kind in sorted({key[0] for key in lit_of}):
            cols = [lit_of.get((kind, t * step, 0, 1)) for t in range(nb)]
            if any(c is None for c in cols) or (forbid is not None and bool(forbid[cols].any())):
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
    for m, info in enumerate(literal_table(model)):
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


def refine_intervals(rules, model, names, pool_of, min_events=2, min_lift=1.2, forbid=None):
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
    for head, rs in sorted(by_head.items()):
        pool = pool_of(head)
        F = pool["F"] > 0.5
        w, y = pool["w"], pool["y"]
        for q, ts in thresholds.items():
            n = len(ts)
            ge = [back.get((q, "ge", k, 0, 1)) for k in range(n)]
            lt0 = back.get((q, "lt", 0, 0, 1))
            if any(c is None for c in ge) or lt0 is None or (forbid is not None and bool(forbid[ge + [lt0]].any())):
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
    for (head, other, q), items in sorted(groups.items(), key=lambda kv: kv[0][0]):   # head by head (pool_of)
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
    for r in sorted((r for r in out if r.get("merged")), key=lambda r: r["head"]):
        if "precision" not in r:
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
    for (head, other, kind, scope, j), items in sorted(groups.items(), key=lambda kv: kv[0][0]):
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
    for r in sorted((r for r in out if r.get("merged")), key=lambda r: r["head"]):
        if True:
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
        bias[h] = float(b0.detach())
    return weight, bias


def _soft_act(lits_f, body, gate):
    """(B, R) soft rule activations: the product over a rule's literals of 1 - g (1 - x), x the literal (0 / 1),
    g its gate in [0, 1] (1: required, the hard AND; 0: ignored). lits_f (B, L + 1) float, body / gate (R, K)."""
    out = []
    for i in range(0, len(body), RULE_CHUNK):
        x = lits_f[:, body[i:i + RULE_CHUNK]]                                   # (B, Rc, K)
        out.append((1 - gate[i:i + RULE_CHUNK][None] * (1 - x)).prod(dim=2))
    return torch.cat(out, dim=1) if out else torch.zeros((len(lits_f), 0), device=lits_f.device)


FIT_CELLS = 4e8             # rows x rules x literals of one head's soft fit (activations + gradients ~ 10 GB)


def fit_soft(rules, conf, base, train, model, heads, device, mode, steps=500, l2=1e-5, gate_reg=1e-4, temporal=None):
    """Soft AND and soft OR per head, fit on the weighted train sample (2026-11-22, the user: the AND within a
    rule and the OR between rules should be soft, not hard).
    Soft AND: a rule's activation a_r = prod_i (1 - g_ri (1 - x_i)); every gate g_ri starts at ~0.95 (the
    mined hard rule) and may fall, so a rule with a literal missing still gives part of its evidence;
    gate_reg x sum(1 - g) keeps the gates near the hard rule unless the data pays for it.
    Soft OR, mode "logit": p = sigmoid(b + sum_r w_r a_r); mode "nor": p = 1 - (1 - base) prod_r (1 - q_r a_r),
    q_r from the rule's confidence. Modes of SPARSE: "logit" with the gates and weights pushed toward 0
    instead (readable rules: the literals and rules that do not pay drop). Modes "mdlt_*" need temporal
    (`temporal_groups`): the members of a temporal rule share the gates of their context literals, and the
    description length counts each temporal rule once. Returns gate (R, K) and (w (R,), b (A,)) or q (R,)."""
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
        if mode.startswith("mdl"):
            lam = mdl_lambda(mode, int(pool["event"][pool["y"]].unique().numel()))
        cap = max(1, int(FIT_CELLS // max(len(pool["y"]) * K, 1)))
        if len(idx) > cap:
            # Safeguard (activations (rows, rules, literals) must fit the GPU): the cap best rules by their own
            # pessimistic precision are fit, the others get weight 0 (reported).
            order = sorted(idx, key=lambda i: -float(conf[i]))
            print(f"[Rule lab] fit_soft: head {h}: {len(idx)} rules, only the {cap} most confident fit (memory)")
            idx = order[:cap]
        body = _pad([rules[i]["literals"] for i in idx], L).to(device)
        if body.shape[1] < K:
            body = torch.cat([body, torch.full((len(idx), K - body.shape[1]), L, device=device)], dim=1)
        X = torch.cat([pool["F"], torch.ones((len(pool["y"]), 1), device=device)], dim=1)[:, body]   # (n, R, K)
        pad = body == L
        y, w = pool["y"].float(), pool["w"] / pool["w"].sum()
        tied = is_temporal(mode) and temporal is not None
        if tied:
            # One gate parameter per (temporal rule, context literal) and per (rule, slot literal); owner: the
            # temporal rule (context literal, paid while any member is kept) or the member (its slot literal).
            grp_of, spos = temporal[0], temporal[1]
            local = {g: n for n, g in enumerate(dict.fromkeys(grp_of[i] for i in idx))}
            gid = torch.tensor([local[grp_of[i]] for i in idx], device=device)
            keys, pid = {}, torch.zeros(body.shape, dtype=torch.long)
            own_rule, owner = [], []
            for a, i in enumerate(idx):
                for k, m in enumerate(rules[i]["literals"]):
                    key = ("r", a, k) if k == spos[i] else ("g", local[grp_of[i]], m)
                    if key not in keys:
                        keys[key] = len(keys)
                        own_rule.append(key[0] == "r")
                        owner.append(a if key[0] == "r" else local[grp_of[i]])
                    pid[a, k] = keys[key]
            pid = pid.to(device)
            own_rule = torch.tensor(own_rule, dtype=torch.bool, device=device)
            owner = torch.tensor(owner, dtype=torch.long, device=device)
            theta_p = torch.full((len(keys),), 3.0, device=device, requires_grad=True)
            params = [theta_p]
        else:
            theta = torch.full(body.shape, 3.0, device=device, requires_grad=True)
            params = [theta]
        # "hier" small cues: their main cue's literals are pinned (gate 1, not paid; with the temporal fit too)
        pinned = torch.zeros(body.shape, dtype=torch.bool)
        for a, i in enumerate(idx):
            for k, m in enumerate(rules[i]["literals"]):
                pinned[a, k] = m in rules[i].get("main", ())
        pinned = pinned.to(device)
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
            if tied:
                theta = theta_p[pid]
            g = torch.sigmoid(theta).masked_fill(pad, 0.0).masked_fill(pinned, 1.0)
            a = (1 - g[None] * (1 - X)).prod(dim=2)                           # (n, R)
            if logistic:
                loss = (w * torch.nn.functional.binary_cross_entropy_with_logits(a @ wt + b0, y, reduction="none")).sum()
            else:
                miss = torch.log1p(-(torch.sigmoid(phi)[None] * a).clamp(max=1 - 1e-6)).sum(dim=1)
                p = (1 - (1 - b) * torch.exp(miss)).clamp(1e-6, 1 - 1e-6)
                loss = -(w * (y * p.log() + (1 - y) * (1 - p).log())).sum()
            if mode in SPARSE:
                loss = loss + SPARSE[mode] * base_loss * (g.sum() / n_real + wt.abs().mean())
            elif tied:
                kept = torch.tanh(wt.abs() / RULE_TAU)                             # (R,) a member kept
                norm = torch.zeros(len(local), device=device).index_add(0, gid, wt ** 2).add(1e-12).sqrt()
                kept_g = torch.tanh(norm / RULE_TAU)                               # a temporal rule kept
                pay = torch.where(own_rule, kept[owner.clamp(max=len(idx) - 1)], kept_g[owner.clamp(max=len(local) - 1)])
                loss = loss + lam * base_loss / MDL_SCALE * (RULE_COST * kept_g.sum()
                                                             + (torch.sigmoid(theta_p) * pay).sum())
            elif mode.startswith("mdl"):
                kept = torch.tanh(wt.abs() / RULE_TAU)                             # (R,) a rule kept, 0..1
                lits = (g.masked_fill(pinned, 0.0) * kept[:, None]).sum()                                    # literals of the kept rules
                loss = loss + lam * base_loss / MDL_SCALE * (RULE_COST * kept.sum() + lits)
            else:
                loss = loss + gate_reg * (1 - g).masked_fill(pad, 0.0).sum()
                if logistic:
                    loss = loss + l2 * (wt ** 2).sum()
            loss.backward()
            opt.step()
        if tied:
            theta = theta_p[pid]
        gi = torch.sigmoid(theta.detach()).masked_fill(pad, 1.0).masked_fill(pinned, 1.0).cpu()
        gate[idx] = gi
        if logistic:
            weight[idx] = wt.detach().cpu()
            bias[h] = float(b0.detach())
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
BOOT_BINS = 10                  # score bins merged for the per-video histograms of the bootstrap (2001 -> 201)
BOOT_SCORERS = ("mlp", "soft_logit", "mdl")      # the scorers with per-video histograms (prefixes)
BOOT_SAMPLES = 1000


def _coarse(x):
    """(..., NB) -> (..., ceil(NB / BOOT_BINS)) summed bins."""
    pad = (-x.shape[-1]) % BOOT_BINS
    x = torch.nn.functional.pad(x, (0, pad))
    return x.view(*x.shape[:-1], -1, BOOT_BINS).sum(-1)


@torch.no_grad()
def evaluate(model, mlp, training_data, video_ids, rule_sets, head_mask, possible, cfg, k_mlp, derived, stats=None):
    """Score histograms {(variant, scorer) | "mlp": (pos, neg) (A, NB)} on all at-risk rows of video_ids.
    stats (dict, 2026-11-25): filled with "onsets" (A,) the distinct eval onsets per head and, for stats["heads"],
    "videos": per video {name: (pos, neg) (H, NB / BOOT_BINS)} of the BOOT_SCORERS (the video bootstrap)."""
    device, S, w = cfg["device"], model.slot_size, int(cfg.get("clause_slots", 1))
    A, NB = model.num_atoms, step07_paper.NB
    hist = {}
    cur = {}
    if stats is not None:
        stats["onsets"] = torch.zeros(A, dtype=torch.long)
        stats["videos"] = []
        sel = torch.tensor(stats["heads"], dtype=torch.long, device=device)

    def add(name, p, keep, y):
        if name not in cur:
            cur[name] = (torch.zeros(A * NB, device=device), torch.zeros(A * NB, device=device))
        cell = torch.arange(A, device=device)[None] * NB + step07_paper._logit_bins(p)
        cur[name][0].add_(torch.bincount(cell[keep & y], minlength=A * NB).float())
        cur[name][1].add_(torch.bincount(cell[keep & ~y], minlength=A * NB).float())

    for video_id in tqdm(video_ids, desc="Evaluating"):
        cur.clear()
        keys = []
        can = possible(video_id)
        for frames in training_data["frames"][video_id].split(int(cfg.get("mine_chunk", 256))):
            rows = step05._video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            keep, y = step05_mine._head_rows(rows, can, head_mask, S, w)
            if not keep.any():
                continue
            if stats is not None:
                r_i, h_i = torch.nonzero(y, as_tuple=True)
                t_i, n_i = rows["rows"][r_i].T
                onset = t_i.long() + rows["first"][r_i, h_i].long()
                keys.append((h_i.long() * 100000 + n_i.long()) * 1000000 + onset)
            lits = literal_rows(model, rows["inputs"], derived)
            if mlp is not None:
                add("mlp", mlp.window_probs(mlp(rows["inputs"])[0])[:, k_mlp], keep, y)
            for variant, rs in rule_sets.items():
                for scorer, p in rs.scores(lits).items():
                    add(f"{variant}/{scorer}", p, keep, y)
        for name, (pos, neg) in cur.items():
            if name not in hist:
                hist[name] = (pos.clone(), neg.clone())
            else:
                hist[name][0].add_(pos)
                hist[name][1].add_(neg)
        if stats is not None:
            if keys:
                u = torch.unique(torch.cat(keys))
                stats["onsets"] += torch.bincount((u // 100000 // 1000000).cpu(), minlength=A)[:A]
            stats["videos"].append({name: (_coarse(pos.view(A, NB)[sel]).cpu(), _coarse(neg.view(A, NB)[sel]).cpu())
                                    for name, (pos, neg) in cur.items()
                                    if name.split("/")[-1].startswith(BOOT_SCORERS)})
    return {name: (pos.view(A, NB).cpu(), neg.view(A, NB).cpu()) for name, (pos, neg) in hist.items()}


@torch.no_grad()
def bootstrap_ci(videos, names_h, device, ref=None, samples=BOOT_SAMPLES, seed=0):
    """Per head the 95% video-bootstrap interval of each scorer's AP and, with ref (e.g. "mlp"), of AP - AP(ref):
    {head name: {scorer: [lo, hi], scorer + " - " + ref: [lo, hi]}} (`evaluate` stats["videos"])."""
    if not videos:
        return {}
    keys = [k for k in videos[0] if all(k in v for v in videos)]
    P = {k: torch.stack([v[k][0] for v in videos]).to(device) for k in keys}       # (V, H, B)
    N = {k: torch.stack([v[k][1] for v in videos]).to(device) for k in keys}
    g = torch.Generator(device="cpu").manual_seed(seed)
    V = len(videos)
    aps = {k: [] for k in keys}
    for _ in range(samples):
        wv = torch.bincount(torch.randint(V, (V,), generator=g), minlength=V).float().to(device)
        for k in keys:
            aps[k].append(step07_paper._ap(torch.einsum("v,vhb->hb", wv, P[k]), torch.einsum("v,vhb->hb", wv, N[k])))
    aps = {k: torch.stack(v).cpu() for k, v in aps.items()}                       # (samples, H)
    out = {}
    for j, h in enumerate(names_h):
        row = {}
        for k, a in aps.items():
            x = a[:, j][~torch.isnan(a[:, j])]
            if len(x):
                row[k] = [round(float(x.quantile(0.025)), 3), round(float(x.quantile(0.975)), 3)]
            if ref and ref in aps and k != ref:
                d = (a[:, j] - aps[ref][:, j])
                d = d[~torch.isnan(d)]
                if len(d):
                    row[f"{k} - {ref}"] = [round(float(d.quantile(0.025)), 3), round(float(d.quantile(0.975)), 3)]
        out[h] = row
    return out


class _Selected:
    """The per-head choice as a rule set for `report` (its rules: the chosen variant's per head)."""

    def __init__(self, rules):
        self.rules = rules


def select_per_head(hist_eval, hist_check, rule_sets, heads, scorer, default, small=None, small_variant=None,
                    min_pos=20):
    """scorer: one scorer or a comma list (e.g. "soft_logit,mdl_0.1,mdl_0.3": the fit strength chosen per head too;
    the first is the one of the small heads and of the default)."""
    scorers = [x for x in scorer.split(",") if x]
    return _select(hist_eval, hist_check, rule_sets, heads, scorers, default, small or set(), small_variant, min_pos)


def _select(hist_eval, hist_check, rule_sets, heads, scorers, default, small, small_variant, min_pos):
    """Per head the variant of best AP on the check videos (the even val videos; the evaluation stays on the odd
    ones), scorer fixed (2026-11-24, the user: choose the variant per head; relaxed alone beat relaxed+exh on
    Bus:TurLft / LarVeh:TurLft, tr helped Cyc:VehLane). Rules of the choice (2026-11-24 quick test, 12 heads: the
    check videos chose right on the well sampled heads, at random on the small ones, Cyc:VehLane had no check
    positive and fell to a variant without rules, 0.610 -> 0.025):
      - a head in `small` (a transfer target, few train onsets) takes `small_variant` (the transfer);
      - a head with >= min_pos positive check rows the variant of best check AP;
      - else `default`.
    Returns the histograms of the choice on the evaluation videos, {head: variant} and the rule set."""
    scorer = scorers[0]
    names = [n for n in hist_check if n != "mlp" and n.split("/")[1] in scorers]
    ap = {n: step07_paper._ap(*hist_check[n]) for n in names}
    npos = next(iter(hist_check.values()))[0].sum(dim=1)
    pos, neg = (torch.zeros_like(x) for x in hist_eval[f"{default}/{scorer}"])
    choice, rules = {}, []
    for h in range(pos.shape[0]):
        best = f"{default}/{scorer}"
        if h in heads and h in small and small_variant:
            best = f"{small_variant}/{scorer}"
        elif h in heads and float(npos[h]) >= min_pos:
            vals = {n: float(ap[n][h]) for n in names}
            vals = {n: (v if v == v else -1.0) for n, v in vals.items()}
            best = max(names, key=lambda n: (vals[n], -names.index(n)))     # ties: the variant listed first
        pos[h], neg[h] = hist_eval[best][0][h], hist_eval[best][1][h]
        if h in heads:
            v = best.split("/")[0]
            choice[h] = best if len(scorers) > 1 else v
            rules += [r for r in rule_sets[v].rules if r["head"] == h]
    return (pos, neg), choice, _Selected(rules)


def report(hist, names, heads, rule_sets, onsets=None, ci=None, eval_desc="odd val videos"):
    """Per head AP of every scorer; macro / close / wins over the learnable heads; per action. onsets (A,): the
    distinct eval onsets per head; ci: `bootstrap_ci` per head name."""
    ap = {name: step07_paper._ap(pos, neg) for name, (pos, neg) in hist.items()}
    has_mlp = "mlp" in hist
    npos = next(iter(hist.values()))[0].sum(dim=1)            # the same rows and labels for every scorer
    compared = [h for h in heads if npos[h] >= MIN_EVAL_POS]
    scorers = [n for n in ap if n != "mlp"]

    def val(name, h):
        x = float(ap[name][h])
        return 0.0 if x != x else x
    learnable = [h for h in compared if max(val(n, h) for n in ap) >= LEARNABLE_MIN_AP]
    rows = {}
    for name in scorers:
        a = [val(name, h) for h in learnable]
        m = [val("mlp", h) for h in learnable] if has_mlp else [0.0] * len(a)
        variant, scorer = name.split("/")
        rs = rule_sets[variant].rules
        size = getattr(rule_sets[variant], "size", {}).get(scorer)          # after the drops (sparse modes)
        rows[name] = {"macro": sum(a) / max(len(a), 1), "mlp_macro": sum(m) / max(len(m), 1) if has_mlp else None,
                      "close": sum(x >= CLOSE * y for x, y in zip(a, m)) if has_mlp else None,
                      "wins": sum(x > y for x, y in zip(a, m)) if has_mlp else None,
                      "rules": size[0] if size else len(rs),
                      "literals": size[1] if size else sum(len(r["literals"]) for r in rs) / max(len(rs), 1),
                      "max_literals": size[2] if size else max((len(r["literals"]) for r in rs), default=0)}
        rows[name]["total_literals"] = round(rows[name]["rules"] * rows[name]["literals"])
    best = max(rows, key=lambda n: rows[n]["macro"]) if rows else None
    print(f"\n[Rule lab] eval = {eval_desc}, every at-risk row; {len(compared)} heads with >= {MIN_EVAL_POS} "
          f"positive rows, {len(learnable)} learnable (some scorer >= AP {LEARNABLE_MIN_AP})"
          + (f"; MLP macro {rows[best]['mlp_macro']:.3f}" if best and has_mlp else "; no MLP (--lab_mlp False)"))
    print(f"  {'variant/scorer':<22}{'rules':>7}{'lits':>6}{'max':>5}{'macro':>8}{'close':>7}{'wins':>6}")
    for name, r in sorted(rows.items(), key=lambda x: -x[1]["macro"]):
        mlp_cols = f"{r['close']:>5}/{len(learnable):<3}{r['wins']:>4}" if has_mlp else f"{'-':>9}{'-':>4}"
        print(f"  {name:<22}{r['rules']:>7}{r['literals']:>6.2f}{r['max_literals']:>5}{r['macro']:>8.3f}{mlp_cols}")
    # Accuracy vs rule-set size: per variant, its compact fits (MDL / sparse) and soft_logit, by size.
    by_variant = defaultdict(list)
    for name, r in rows.items():
        variant, scorer = name.split("/")
        if is_compact(scorer) or scorer == "soft_logit":
            by_variant[variant].append((r["total_literals"], r["rules"], r["macro"], scorer))
    if by_variant:
        print("\n  accuracy vs size (macro AP over the learnable heads; rules and literals kept):")
        for variant, pts in sorted(by_variant.items()):
            print(f"    {variant}: " + "  ".join(f"{sc} {m:.3f} @ {nr} rules / {nl} lits"
                                                for nl, nr, m, sc in sorted(pts, reverse=True)))
    if best:
        by_action = defaultdict(list)
        for h in learnable:
            by_action[names[h].split(":")[-1]].append(h)
        ref = "mlp" if has_mlp else None
        print(f"\n  per action, {best}" + (" vs MLP" if ref else "") + " (macro AP, heads):")
        for action, hs in sorted(by_action.items(),
                                 key=lambda x: sum(val(best, h) - (val(ref, h) if ref else 0) for h in x[1])):
            ref_txt = f"{sum(val(ref, h) for h in hs) / len(hs):>7.3f}" if ref else ""
            print(f"    {action:<12}{sum(val(best, h) for h in hs) / len(hs):>7.3f}{ref_txt}{len(hs):>4}")
    per_head = {names[h]: {"learnable": h in learnable, "pos_rows": int(npos[h]),
                           **({"onsets": int(onsets[h])} if onsets is not None else {}),
                           **{n: _none(float(ap[n][h])) for n in ap},
                           **({"ci": ci[names[h]]} if ci and names[h] in ci else {})} for h in compared}
    if onsets is not None:
        few = sorted((int(onsets[h]), names[h]) for h in heads if int(onsets[h]) < EVAL_MIN_ONSETS)
        print(f"\n  eval onsets per head: {len(heads) - len(few)} of {len(heads)} heads with >= {EVAL_MIN_ONSETS}; "
              f"fewer (more data needed, e.g. v8): " + ", ".join(f"{n} {k}" for k, n in few))
    if ci:
        print("\n  per head AP with 95% video-bootstrap intervals:")
        for h in compared:
            row = ci.get(names[h], {})
            ons = f"{int(onsets[h])} onsets" if onsets is not None else ""
            print(f"    {names[h]} ({ons}): " + "; ".join(f"{k} {_none(float(ap[k][h])) or 0:.3f} {v}" if k in ap
                                                       else f"{k} {v}" for k, v in row.items()))
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
        m = re.fullmatch(DIRECTION_RE + r"\((\w+)\)@(\d+)", t)
        if m:
            groups.setdefault((m.group(1), m.group(3), m.group(4)), []).append((int(m.group(2)), g))
        else:
            out.append((t, g))
    for (kind, scope, j), tg in groups.items():
        out.append((f"{kind}{agent_shape.soft_sector_text([t for t, _ in tg], [g for _, g in tg])}({scope})@{j}",
                    max(g for _, g in tg)))
    return _merge_linear(out)


def pretty_literals(texts):
    """Rule literal texts with the 360-degree half circles of one kind and slot merged into one sector
    ("shp:dir_200(own)@1", "shp:dir_060(own)@1" -> "shp:dir[200, 240)(own)@1", `shape.sector_text`)."""
    from src.exp_roadpp import shape as agent_shape
    import re
    groups, out = {}, []
    for t in texts:
        m = re.fullmatch(DIRECTION_RE + r"\((\w+)\)@(\d+)", t)
        if m:
            groups.setdefault((m.group(1), m.group(3), m.group(4)), []).append(int(m.group(2)))
        else:
            out.append(t)
    for (kind, scope, j), thetas in groups.items():
        out.append(f"{kind}{agent_shape.sector_text(thetas) or '[empty]'}({scope})@{j}")
    return [t for t, _ in _merge_linear([(t, 1.0) for t in out])]


def parse_variant(name):
    """"relaxed+precise_d_nn" -> (["relaxed", "precise"], (True, True, False)): the rule sets united; flags:
    with the derived literals (_d), without the negated literals of the other agents (_nn), without the
    shape-phase atoms (_ns, `shape.py`)."""
    flags = {"_d": False, "_nn": False, "_ns": False, "_nd": False, "_np": False, "_ng": False}
    while any(name.endswith(f) for f in flags):
        f = next(f for f in flags if name.endswith(f))
        flags[f], name = True, name[:-len(f)]
    return name.split("+"), (flags["_d"], flags["_nn"], flags["_ns"], flags["_nd"], flags["_np"], flags["_ng"])


def atom_literals(model, lo, hi, older_only=False):
    """(L,) bool: the literals (any kind and scope) of the atoms lo..hi - 1, e.g. the shape atoms; older_only:
    only those of the slots before the newest (j > 1)."""
    return torch.tensor([info["kind"] in ("X", "!X") and lo <= info["atom"] < hi and (not older_only or info["j"] > 1)
                         for info in literal_table(model)], dtype=torch.bool)


def add_test_videos(training_data, cfg):
    """The dataset's test split (SUMO multi-map exps: the test maps, seen neither in training nor by the rule check
    or the selection), prepared like the train / val videos (as step06 does) and merged into training_data. Cached
    next to the training data. Returns the sorted test video ids (2026-11-25: unseen-map evaluation on test)."""
    atom_file = str(cfg["val_atom_file"]).replace("val_atoms_compressed", "test_atoms_compressed")
    box_file = cfg.get("val_box_file")
    box_file = str(box_file).replace("val_boxes", "test_boxes") if box_file else None
    if not os.path.exists(atom_file):
        raise FileNotFoundError(f"--lab_eval_split test: no test atoms at {atom_file}")
    cache = os.path.join(os.path.dirname(str(cfg["training_data_file"])), "rule_lab_test_data.pt")
    data = step05.load_data(cache) if step05.is_fresh(cache, cfg["training_data_file"], atom_file) else {}
    if (data.get("min_off_frames") != cfg["min_off_frames"] or data.get("roles", []) != list(cfg["roles"])
            or data.get("geometry", []) != list(cfg["geometry"]) or data.get("grid_tag", "") != step05.grid_tag(cfg)):
        unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
        data = step05.prepare_videos(atom_file, len(unique_atoms), cfg["device"], cfg["min_off_frames"], box_file,
                                     cfg["roles"], cfg["geometry"], unique_atoms, step05.load_grid(cfg),
                                     step05.load_motion(cfg), step05.load_ego_motion(cfg), step05.load_ego_flow(cfg),
                                     step05.load_durations(cfg), step05.load_predict(cfg), step05.load_shape(cfg),
                                     step05.load_gestalt(cfg))
        data.update(min_off_frames=cfg["min_off_frames"], roles=list(cfg["roles"]), geometry=list(cfg["geometry"]),
                    grid_tag=step05.grid_tag(cfg))
        step05.save_data(data, cache)
    test_ids = sorted(data["videos"])
    shared = set(test_ids) & (set(training_data["train_videos"]) | set(training_data["val_videos"]))
    if shared:
        raise ValueError(f"{len(shared)} test videos are also train / val videos, e.g. {sorted(shared)[:3]}")
    if data.get("num_atoms", training_data["num_atoms"]) != training_data["num_atoms"]:
        raise ValueError(f"test data has {data.get('num_atoms')} atoms, the training data {training_data['num_atoms']}")
    for key in ("videos", "frames", "positive_frames"):
        training_data[key].update(data[key])
    print(f"[Rule lab] evaluation on the test split: {len(test_ids)} videos ({cache})")
    return test_ids


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
    return torch.tensor([info["kind"] in ("!X", "!G") and info["scope"] == 1 for info in literal_table(model)],
                        dtype=torch.bool)


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


# -------------- Main cues and small cues ("hier", 2026-11-25) --------------
# The user: a head should rest on a main cue that is common and important but not precise (it is there before most
# onsets), and on small cues for the different situations, rare but precise. Per head:
#   1. main cues: up to --lab_hier_mains rules of 1-2 literals (>= 1 positive literal), each the one of most
#      information gain on the rows earlier main cues leave (`main_cues`), lift >= --lab_hier_lift (and >= 1 on the
#      check videos), until they cover --lab_hier_cover of the onset rows;
#   2. small cues: the "exh" rules mined within the rows of each main cue (they must beat the main cue's precision
#      there); a small-cue rule is main & small, the main literals pinned (gate 1, not paid in the MDL fit), so a
#      small cue only acts inside its main cue;
#   3. exceptions: the "exh" rules of the rows no main cue covers.
# A head may have no main cue (the user: it is not required): then all its rules are of step 3, plain short rules
# (2026-11-25: LarVeh:Jun, no cue with lift >= 2 covers more than 13% of its onset rows).
# Small cues and exceptions: the --lab_hier_small most precise each (2026-11-25: 3-5k small cues per main cue made
# the fit slow and the heads flat again).
def _subpool(pool, mask):
    return {k: (v[mask] if torch.is_tensor(v) and len(v) == len(mask) else v) for k, v in pool.items()}


def _entropy_sum(n, h):
    """Weighted binary entropy x weight of a row set with weight n and hit weight h (elementwise)."""
    p = (h / n.clamp(min=1e-12)).clamp(1e-9, 1 - 1e-9)
    return -n * (p * p.log() + (1 - p) * (1 - p).log())


def main_cues(pool, allowed, positive, cfg, check=None, top=150, text=None):
    """[(literals tuple, onset recall, precision, lift)] of one head (see above). Each next main cue: the 1-2
    literal rule of most information gain about the onset among the rows no earlier main cue covers (a decision
    list's first steps), that covers >= --lab_hier_min_cover of those rows' onset rows, with lift >= --lab_hier_lift
    (2026-11-25: new rows x log(lift) chose "shp:app_middle & !p1_h_shrink" for Car:IncomLane, recall 0.93 at
    lift 1.7, a cue of almost every row)."""
    F, y, w = pool["F"], pool["y"], pool["w"]
    if not bool(y.any()):
        return []
    yf = y.float()
    rate = float((w * yf).sum() / w.sum().clamp(min=1e-12))
    n1, h1 = w @ F, (w * yf) @ F
    lift1 = (h1 / n1.clamp(min=1e-12)) / max(rate, 1e-12)
    rec1 = F[y].mean(dim=0)
    cand = torch.nonzero(allowed & (lift1 > 1.0) & (n1 > 0)).flatten()
    if len(cand) == 0:
        return []
    # Candidates: the most common literals and those of most gain (with the lift) on their own; by recall alone the
    # near-universal literals filled the list (2026-11-25).
    N, H = float(w.sum()), float((w * yf).sum())
    g1 = _entropy_sum(torch.tensor(N), torch.tensor(H)).to(F.device) - _entropy_sum(n1, h1) \
        - _entropy_sum((N - n1).clamp(min=0), (H - h1).clamp(min=0))
    by_rec = cand[torch.argsort(rec1[cand], descending=True)[:top // 2]]
    lifted = cand[lift1[cand] >= cfg["lab_hier_lift"]]
    by_gain = lifted[torch.argsort(g1[lifted], descending=True)[:top // 2]]
    cand = torch.unique(torch.cat([by_rec, by_gain]))
    Fc = F[:, cand]
    pos = positive[cand]
    ok = (pos[:, None] | pos[None, :]) & torch.triu(torch.ones((len(cand), len(cand)), dtype=torch.bool, device=F.device))
    if check is not None and len(check["y"]):
        Kc, wk, yk = check["F"][:, cand], check["w"], check["y"].float()
        rate_c = float((wk * yk).sum() / wk.sum().clamp(min=1e-12))
        n2c, h2c = (Kc * wk[:, None]).T @ Kc, (Kc * (wk * yk)[:, None]).T @ Kc
        ok &= (h2c > 0) & ((h2c / n2c.clamp(min=1e-12)) >= rate_c)
    free = torch.ones(len(y), dtype=torch.bool, device=F.device)                 # rows no main cue covers yet
    ev_all = pool["event"][y]
    n_ev = max(int(ev_all.unique().numel()), 1)
    out = []
    for _ in range(int(cfg["lab_hier_mains"])):
        Ff, wf, yff = Fc[free], w[free], yf[free]
        if not bool(yff.any()):
            break
        n2, h2 = (Ff * wf[:, None]).T @ Ff, (Ff * (wf * yff)[:, None]).T @ Ff
        r2 = (Ff * yff[:, None]).T @ Ff                                          # onset rows covered (unweighted)
        N, H = float(wf.sum()), float((wf * yff).sum())
        gain = _entropy_sum(torch.tensor(N), torch.tensor(H)).to(F.device) - _entropy_sum(n2, h2) \
            - _entropy_sum((N - n2).clamp(min=0), (H - h2).clamp(min=0))
        lift = (h2 / n2.clamp(min=1e-12)) / max(rate, 1e-12)
        good = ok & (lift >= cfg["lab_hier_lift"]) & (r2 >= cfg["lab_hier_min_cover"] * float(yff.sum()))
        single = torch.eye(len(cand), dtype=torch.bool, device=F.device)
        gain = gain * torch.where(single, 1.0, 0.99)                             # a pair must add > 1% over one literal
        score = torch.where(good, gain, torch.full_like(gain, -1.0))
        best = int(torch.argmax(score))
        if float(score.flatten()[best]) <= 0:
            if text is not None:
                # Why no (further) main cue: the best candidates by gain, then those with the lift by gain and by
                # cover (2026-11-25: the gain alone listed only near-universal cues such as shp:app_middle)
                cov = r2 / max(float(yff.sum()), 1.0)
                lifted = ok & (lift >= cfg["lab_hier_lift"])
                print(f"      no main cue {len(out)} (need lift >= {cfg['lab_hier_lift']}, cover >= "
                      f"{cfg['lab_hier_min_cover']} of the {int(yff.sum())} onset rows left):")
                for title, mask, key in (("by gain", ok, gain), ("lift ok, by gain", lifted, gain),
                                         ("lift ok, by cover", lifted, cov)):
                    order = torch.argsort(torch.where(mask, key, torch.full_like(key, -1e9)).flatten(),
                                          descending=True)[:5]
                    print(f"        {title}:")
                    for o in order.tolist():
                        a, b = divmod(o, len(cand))
                        if not bool(mask[a, b]):
                            break
                        lits = [int(cand[a])] if a == b else [int(cand[a]), int(cand[b])]
                        print(f"          {' & '.join(text(m) for m in lits)}: lift {float(lift[a, b]):.1f}, "
                              f"cover {float(cov[a, b]):.2f}")
            break
        a, b = divmod(best, len(cand))
        lits = (int(cand[a]),) if a == b else (int(cand[a]), int(cand[b]))
        hold = F[:, list(lits)].all(dim=1).bool()
        recall = float(pool["event"][y & hold].unique().numel()) / n_ev
        prec = float((w * yf)[hold].sum() / w[hold].sum().clamp(min=1e-12))
        out.append((lits, recall, prec, prec / max(rate, 1e-12)))
        ok[a, b] = False
        free &= ~hold
        if 1 - float(yf[free].sum() / yf.sum()) >= cfg["lab_hier_cover"]:
            break
    return out


def _cap(found, k):
    """The k most precise rules of an exhaustive_rules list (ties: more onsets first)."""
    return sorted(found, key=lambda x: (-x[1], -x[2]))[:k] if k > 0 else found


def hier_rules(ext, h, train, check, allowed, positive, cfg, device, names):
    """The "hier" rules of one head: main cues (role "main", root), main & small cues ("sub", "main" = the main
    literals), exceptions ("exc")."""
    L = ext.num_literals
    pool = step05_mine._pool_of(train, h, L, device)
    ck = step05_mine._pool_of(check, h, L, device) if cfg.get("lab_check") and check is not None and \
        bool((check["pairs"]["head"] == h).any()) else None
    text = lambda m: step05_mine.literal_text(ext, names, m)
    mains = main_cues(pool, allowed, positive, cfg, ck, text=text)
    out, seen = [], set()

    def add(r):
        key = frozenset(r["literals"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    anymain = torch.zeros(len(pool["y"]), dtype=torch.bool, device=device)
    anymain_c = torch.zeros(len(ck["y"]), dtype=torch.bool, device=device) if ck is not None else None
    for k, (lits, recall, prec, lift) in enumerate(mains):
        add({"head": int(h), "literals": list(lits), "root": True, "role": "main", "main_id": k,
             "main_recall": round(recall, 3)})
        hold = pool["F"][:, list(lits)].all(dim=1).bool()
        anymain |= hold
        hold_c = ck["F"][:, list(lits)].all(dim=1).bool() if ck is not None else None
        if hold_c is not None:
            anymain_c |= hold_c
        sub_allowed = allowed.clone()
        sub_allowed[list(lits)] = False
        found, _ = exhaustive_rules(_subpool(pool, hold), sub_allowed, cfg,
                                    _subpool(ck, hold_c) if ck is not None else None)
        n_found, found = len(found), _cap(found, cfg["lab_hier_small"])
        for key, _p, _e in found:
            add({"head": int(h), "literals": list(lits) + [m for m in key if m not in lits], "role": "sub",
                 "main_id": k, "main": list(lits)})
        print(f"    {names[h]} main {k}: {' & '.join(text(m) for m in lits)}  onset recall {recall:.2f}, precision "
              f"{prec:.3f} (lift {lift:.1f}); {len(found)} small cues (of {n_found})")
    found, _ = exhaustive_rules(_subpool(pool, ~anymain), allowed, cfg,
                                _subpool(ck, ~anymain_c) if ck is not None else None)
    # without a main cue the exceptions are the head's whole rule set: not capped (as the plain "exh" rules)
    n_found, found = len(found), (_cap(found, cfg["lab_hier_small"]) if mains else found)
    for key, _p, _e in found:
        add({"head": int(h), "literals": list(key), "role": "exc"})
    print(f"    {names[h]}: {len(mains)} main cues, {len(found)} exception rules (of {n_found}; rows no main cue covers)")
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
    if component in ("tr", "trp"):
        return transfer_rules(ext, names, heads, cfg.get("lab_all_heads", heads), train, check, cfg, device,
                              pooled=component == "trp")
    if component == "hier":
        base_tab = literal_table(ext._model if hasattr(ext, "_model") else ext)
        positive = torch.tensor([not info["kind"].startswith("!") for info in base_tab]
                                + [True] * (L - len(base_tab)), dtype=torch.bool, device=device)
        rules = []
        for h in tqdm(heads, desc="Main and small cues"):
            if not bool((train["pairs"]["head"] == h).any()):
                continue
            allowed = ext.input_mask.to(device) & ~((base % A == h) & (scope == 0))
            rules += hier_rules(ext, h, train, check, allowed, positive, cfg, device, names)
        print(f"[Rule lab] hier: {len(rules)} rules ({sum(r['role'] == 'main' for r in rules)} main cues, "
              f"{sum(r['role'] == 'sub' for r in rules)} small cues, {sum(r['role'] == 'exc' for r in rules)} exceptions)")
        return rules
    if component == "exh":
        capped = 0
        for h in tqdm(heads, desc=f"Exhaustive short rules{' + derived' if use_derived else ''}"):
            if not bool((train["pairs"]["head"] == h).any()):
                continue
            allowed = ext.input_mask.to(device) & ~((base % A == h) & (scope == 0))
            pool = step05_mine._pool_of(train, h, L, device)
            ck = step05_mine._pool_of(check, h, L, device) if cfg.get("lab_check") and \
                bool((check["pairs"]["head"] == h).any()) else None
            found, over = exhaustive_rules(pool, allowed, cfg, ck)
            capped += over > 0
            rules += [{"head": int(h), "literals": list(key)} for key, _p, _e in found]
            del pool, ck
        print(f"[Rule lab] exh: {len(rules)} rules{' (checked on the check videos)' if cfg.get('lab_check') else ''}; the cap --lab_exh_max {cfg['lab_exh_max']} bound on {capped} heads")
        return rules
    for h in tqdm(heads, desc=f"Precise search{' + derived' if use_derived else ''}"):
        if not bool((train["pairs"]["head"] == h).any()):
            continue
        allowed = ext.input_mask.to(device) & ~((base % A == h) & (scope == 0))
        pool = step05_mine._pool_of(train, h, L, device)
        rules += [{"head": int(h), "literals": list(key)} for key, _p, _e in precise_rules(pool, allowed, cfg)]
    return rules


def effective_temporal(rules, gate, keep, temporal):
    """Size of a temporal fit: (temporal rules kept, literals per kept temporal rule, (kept,) literal counts); a
    temporal rule is kept if a member is, its literals: the kept context literals + 1 for its event."""
    group, spos = temporal
    lits = {}
    for i, r in enumerate(rules):
        if not bool(keep[i]):
            continue
        n = sum(1 for k in range(len(r["literals"])) if k != spos[i] and float(gate[i, k]) >= DROP_GATE)
        lits[group[i]] = max(lits.get(group[i], 0), n + (1 if spos[i] >= 0 else 0))
    lens = torch.tensor(list(lits.values()), dtype=torch.long)
    return len(lits), float(lens.float().mean()) if len(lens) else 0.0, lens


def temporal_profiles(texts, mode):
    """The temporal rules of one variant's texts in a temporal mode: [{head, context, event, profile {slot:
    weight}, onsets (max over members), weight (L2 of the profile)}], strongest first; kept members only."""
    by = defaultdict(list)
    for r in texts:
        if "temporal" in r and mode in r["soft"]:
            by[r["temporal"][0]].append(r)
    out = []
    for rs in by.values():
        kept = [r for r in rs if abs(r["soft"][mode]["weight"]) >= DROP_WEIGHT]
        if not kept:
            continue
        j = rs[0]["temporal"][1]
        if j < 0:
            context, event, profile = rs[0]["literals"], None, {}
        else:
            hit = SLOT_RE.match(rs[0]["literals"][j])
            event = hit[1] + hit[2]
            context = [x for k, x in enumerate(rs[0]["literals"]) if k != j]
            profile = {int(SLOT_RE.match(r["literals"][r["temporal"][1]])[3]): r["soft"][mode]["weight"] for r in rs}
        w = (sum(r["soft"][mode]["weight"] ** 2 for r in kept)) ** 0.5
        out.append({"head": rs[0]["head"], "context": context, "event": event,
                    "profile": dict(sorted(profile.items())), "onsets": max(r["events"] for r in rs),
                    "weight": round(w, 3)})
    return sorted(out, key=lambda x: -x["weight"])


def show_temporal(texts, heads, mode, top=8):
    """Print the strongest temporal rules of some heads: context & event[@1 w1 | @2 w2 | ..]."""
    prof = temporal_profiles(texts, mode)
    for head in heads:
        rs = [p for p in prof if p["head"] == head]
        if not rs:
            continue
        print(f"  {head}: {len(rs)} temporal rules kept ({sum(1 for p in rs if p['event'])} with a time profile)")
        for p in rs[:top]:
            ev = (f"{p['event']}[" + " | ".join(f"@{k} {w:+.2f}" for k, w in p["profile"].items()) + "]"
                  if p["event"] else "")
            print(f"    |w| {p['weight']:5.2f}  {p['onsets']:5d} onsets  " + " & ".join(p["context"] + ([ev] if ev else [])))


def hierarchy(texts, mode, top=8):
    """{head: [{main, recall, precision, onsets, weight, small: [..], }], "exceptions": {head: [..]}} of the "hier"
    rules in one fit mode: each main cue with its strongest small cues (kept, by |weight|)."""
    out, exc = {}, {}
    w = lambda r: r["soft"][mode]["weight"] if mode in r["soft"] else r["weight"]
    for r in texts:
        if r.get("role") == "main":
            out.setdefault(r["head"], []).append({"main_id": r["main_id"], "main": r["literals"],
                                                  "recall": r.get("main_recall"), "precision": round(r["precision"], 3),
                                                  "onsets": r["events"], "weight": round(w(r), 3), "small": []})
    for r in texts:
        role = r.get("role")
        if role in ("sub", "exc") and abs(w(r)) >= DROP_WEIGHT:
            item = {"literals": r["literals"], "precision": round(r["precision"], 3), "onsets": r["events"],
                    "weight": round(w(r), 3)}
            if role == "sub":
                for m in out.get(r["head"], []):
                    if m["main_id"] == r["main_id"]:
                        item["literals"] = [x for x in r["literals"] if x not in m["main"]]
                        m["small"].append(item)
            else:
                exc.setdefault(r["head"], []).append(item)
    for ms in out.values():
        for m in ms:
            m["n_small"] = len(m["small"])
            m["small"] = sorted(m["small"], key=lambda x: -abs(x["weight"]))[:top]
    return {"mains": out, "exceptions": {h: sorted(v, key=lambda x: -abs(x["weight"]))[:top] for h, v in exc.items()},
            "n_exceptions": {h: len(v) for h, v in exc.items()}}


def show_hierarchy(hier, heads, top=5):
    for head in heads:
        ms = hier["mains"].get(head, [])
        if not ms and head not in hier["exceptions"]:
            continue
        print(f"  {head}:")
        for m in ms:
            print(f"    MAIN {' & '.join(m['main'])}: onset recall {m['recall']}, precision {m['precision']}, "
                  f"w {m['weight']:+.2f}; {m['n_small']} small cues kept")
            for s in m["small"][:top]:
                print(f"       + {' & '.join(s['literals'])}  (w {s['weight']:+.2f}, p {s['precision']:.2f}, {s['onsets']} onsets)")
        print(f"    {'exceptions (rows no main cue covers)' if ms else 'no main cue (none is common enough): rules'}: "
              f"{hier['n_exceptions'].get(head, 0)} kept")
        for s in hier["exceptions"].get(head, [])[:top]:
            print(f"       * {' & '.join(s['literals'])}  (w {s['weight']:+.2f}, p {s['precision']:.2f}, {s['onsets']} onsets)")


def show_rules(texts, heads, mode, top=8):
    """Print the strongest rules (soft logistic weight of mode) of some heads, with each literal's gate; in a
    sparse mode the dropped literals and rules (DROP_GATE, DROP_WEIGHT) are left out."""
    sparse = is_compact(mode) or mode == "crisp"
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
    parser.add_argument("--lab_gst_negated", type=str, default="False",
                        help="allow the negated gestalt literals (v1: they were filler, '!gst:near_1')")
    parser.add_argument("--lab_mdl", type=str, default="",
                        help="MDL strengths λ for the accuracy vs rule-set size curve, e.g. 0.03,0.1,0.3,1,3")
    parser.add_argument("--lab_fast", type=str, default="False",
                        help="quick test: train-sample counts (no pass over every train row), soft_logit only, no "
                             "sector / interval refinement or merging")
    parser.add_argument("--lab_heads", type=str, default="",
                        help="quick test: only these heads (names like Car:IncomLane or actions like IncomLane)")
    parser.add_argument("--lab_mlp", type=str, default="True",
                        help="train / score the OnsetMLP as the reference (False: rule variants only, no MLP)")
    parser.add_argument("--lab_target", type=str, default="window", choices=("window", "eventual"),
                        help="window: an onset within the rule window (when); eventual: an onset at any later frame "
                             "before the agent leaves the scene (whether, `eventual.py`; no MLP)")
    parser.add_argument("--lab_prepare_only", type=str, default="False",
                        help="build the training data, the OnsetMLP and the row sample, then stop (run once before "
                             "several variants in parallel)")
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
    parser.add_argument("--lab_forbid", type=str, default="",
                        help="atom name prefixes (comma separated) no rule may use, e.g. av:turn_ (the A/B of new "
                             "atoms on the same data)")
    parser.add_argument("--lab_root_exclude", type=str, default="shape,gestalt",
                        help="atom groups of the data layout (e.g. shape, position) that may not be mining roots")
    parser.add_argument("--lab_precise_min", type=float, default=0.5, help="precise search: kept rule precision")
    parser.add_argument("--lab_min_events", type=int, default=5,
                        help="distinct train onsets a rule needs (2026-11-24, the user: a rule must stand on train by "
                             "itself; was 2)")
    parser.add_argument("--lab_min_videos", type=int, default=1)
    parser.add_argument("--lab_max_len", type=int, default=8, help="precise search: literals per rule")
    parser.add_argument("--lab_beam", type=int, default=100, help="precise search: nodes per level")
    parser.add_argument("--lab_branch", type=int, default=10, help="precise search: children per node")
    parser.add_argument("--lab_max_rules", type=int, default=100, help="precise search: rules per head")
    parser.add_argument("--lab_mdl_ref", type=float, default=200.0,
                        help="mdls_λ (--lab_mdl s0.1): λ x min(1, ref / train onsets of the head)")
    parser.add_argument("--lab_eval_split", type=str, default="val", choices=("val", "test", "valtest"),
                        help="evaluate on the odd val videos (default) or on every test video (SUMO _unseen: the test "
                             "maps, untouched by the rule check and the selection); mining and selection are the same")
    parser.add_argument("--lab_head_part", type=str, default="",
                        help="i/k: only every k-th learned head from i (0-based), for parallel runs; use with --lab_tag")
    parser.add_argument("--lab_tag", type=str, default="", help="suffix of the result file (parallel runs on head batches)")
    parser.add_argument("--lab_select", type=str, default="True",
                        help="choose the variant per head on the check videos; reported as the variant 'selected'")
    parser.add_argument("--lab_select_scorer", type=str, default="soft_logit", help="the scorer of the selection")
    parser.add_argument("--lab_select_split", type=str, default="True",
                        help="the selection on val videos apart from the check videos of the rule filter (0,4,.. / 2,6,..)")
    parser.add_argument("--lab_select_min_pos", type=int, default=20,
                        help="positive check rows a head needs to be chosen by its check AP (else the exh variant)")
    parser.add_argument("--lab_check", type=str, default="False",
                        help="drop the rules that do not hold on the check videos (even val videos), `check_filter`")
    parser.add_argument("--lab_merge", type=str, default="False",
                        help="merge the rules that differ in a parameter: class -> veh: group, close sectors and "
                             "intervals (also with --lab_fast)")
    parser.add_argument("--lab_tr_max_onsets", type=int, default=60, help="tr: target heads below these train onsets")
    parser.add_argument("--lab_tr_min_source", type=int, default=100, help="tr: source heads with these train onsets")
    parser.add_argument("--lab_tr_rules", type=int, default=200, help="tr: source rules tried per source head")
    parser.add_argument("--lab_tr_min_events", type=int, default=1, help="tr: train onsets a rule needs on the target")
    parser.add_argument("--lab_tr_mirror", type=str, default="True",
                        help="tr: also try every transferred rule mirrored left / right (`LiteralMap.mirror`)")
    parser.add_argument("--lab_tr_confirm", type=int, default=2,
                        help="tr: held-out onsets (odd train + check videos of the target) a rule needs")
    parser.add_argument("--lab_exh_lift", type=float, default=2.0, help="exh: kept rule precision / head rate")
    parser.add_argument("--lab_exh_seed_lift", type=float, default=1.5, help="exh: pairs extended (lift)")
    parser.add_argument("--lab_exh_gain", type=float, default=1.1,
                        help="exh: a literal added must raise the precision by this factor")
    parser.add_argument("--lab_exh_max", type=int, default=400, help="exh: safeguard cap of rules per head")
    parser.add_argument("--lab_exh_seeds", type=int, default=300, help="exh: pairs grown further per head")
    parser.add_argument("--lab_ci", type=str, default="True",
                        help="per head 95%% video-bootstrap intervals of the AP (and of AP - MLP)")
    parser.add_argument("--lab_ci_ref", type=str, default="",
                        help="the reference of the AP differences (default mlp), e.g. relaxed+exh_nn/mdls_0.1")
    parser.add_argument("--lab_mlp_only", type=str, default="False",
                        help="only evaluate the OnsetMLP (e.g. a stronger one, --mlp_hidden_size) on the eval videos")
    parser.add_argument("--lab_mine_only", type=str, default="False",
                        help="stop after the mining step (step 5): a quick look, e.g. at the hier main cues")
    parser.add_argument("--lab_hier_mains", type=int, default=3, help="hier: main cues per head at most")
    parser.add_argument("--lab_hier_lift", type=float, default=2.0, help="hier: a main cue's lift at least")
    parser.add_argument("--lab_hier_min_cover", type=float, default=0.2,
                        help="hier: a main cue covers at least this share of the onset rows still uncovered")
    parser.add_argument("--lab_hier_small", type=int, default=300,
                        help="hier: small cues per main cue (and exceptions per head) at most, the most precise")
    parser.add_argument("--lab_hier_cover", type=float, default=0.7,
                        help="hier: main cues are added until they cover this share of the onset rows")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    if not getattr(args, "run_id", None):
        raise ValueError("--run_id is needed: the lab compares with that run's OnsetMLP")
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    for key in ("lab_precise_min", "lab_min_events", "lab_min_videos", "lab_max_len", "lab_beam", "lab_branch",
                "lab_max_rules", "lab_exh_lift", "lab_exh_seed_lift", "lab_exh_gain", "lab_exh_max",
                "lab_exh_seeds", "lab_tr_max_onsets", "lab_tr_min_source", "lab_tr_rules", "lab_tr_min_events", "lab_tr_confirm",
                "lab_hier_mains", "lab_hier_lift", "lab_hier_cover",
                "lab_hier_min_cover", "lab_hier_small"):
        cfg[key] = getattr(args, key)
    cfg["lab_check"] = args.lab_check == "True"          # exh: the short rules pass the check test while mining
    cfg["lab_tr_mirror"] = args.lab_tr_mirror == "True"
    device = cfg["device"]
    out_dir = args.output_dir / "rule_lab"
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()

    def step(k, msg):
        print(f"\n[Rule lab] step {k}/9 ({time.time() - t0:.0f} s): {msg}", flush=True)
    step(1, "training data and rule net")
    if os.path.exists(cfg["training_data_file"]):
        print(f"[Rule lab] loading {cfg['training_data_file']} ({os.path.getsize(cfg['training_data_file']) / 2 ** 30:.1f} GB)",
              flush=True)
    training_data = step05.load_training_data(cfg)
    print(f"[Rule lab] training data loaded ({time.time() - t0:.0f} s)", flush=True)
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    print(f"[Rule lab] rule net built ({time.time() - t0:.0f} s)", flush=True)
    if args.lab_target == "eventual":
        eventual.install()
        cfg["target_tag"] = "_ev"
        args.lab_mlp = "False"                  # the OnsetMLP is trained on the window target
        print("[Rule lab] target: eventual (an onset at any later frame before the agent leaves the scene); no MLP")
    mlp = ensure_mlp(training_data, cfg) if args.lab_mlp == "True" else None
    window = int(cfg.get("clause_slots", 1)) * model.slot_size
    k_mlp = min(range(len(model.windows)), key=lambda i: abs(model.windows[i] - window))
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    head_mask = training_data["target_heads"].to(device)
    heads = torch.nonzero(head_mask).flatten().tolist()
    cfg["lab_all_heads"] = list(heads)                     # "tr": its sources may lie outside --lab_heads
    if args.lab_heads:
        # Only these heads (names "Car:IncomLane" or actions "IncomLane"): a quick test. The others keep their
        # base rate in the evaluation and are not mined.
        wanted = {x for x in args.lab_heads.split(",") if x}
        heads = [h for h in heads if names[h] in wanted or names[h].split(":")[-1] in wanted]
        print(f"[Rule lab] --lab_heads: {len(heads)} heads: {', '.join(names[h] for h in heads)}")
    if args.lab_head_part:
        # One part of the heads (parallel runs on several GPUs, 2026-11-25): "i/k" keeps every k-th head from i
        i_p, k_p = (int(x) for x in args.lab_head_part.split("/"))
        heads = heads[i_p::k_p]
        print(f"[Rule lab] --lab_head_part {args.lab_head_part}: {len(heads)} heads")
    possible = step05.class_possible(training_data, vocab, device)
    A = model.num_atoms

    # Literals: the rule net's and the derived ones (same indices in every variant; the variants without "_d"
    # may not use the derived ones).
    derived = Derived(model, names)
    if args.lab_mlp_only == "True":
        # Only the OnsetMLP on the evaluation videos (2026-11-26: a stronger MLP, --mlp_hidden_size / --mlp_dropout,
        # against the rules of an earlier run on the same videos), no mining.
        val_ids = list(training_data["val_videos"])
        eval_ids = val_ids[1::2]
        if args.lab_eval_split == "test":
            eval_ids = add_test_videos(training_data, cfg)
        elif args.lab_eval_split == "valtest":
            eval_ids = val_ids[1::2] + add_test_videos(training_data, cfg)
        step(8, f"OnsetMLP only on {len(eval_ids)} {args.lab_eval_split} videos")
        stats = {"heads": list(heads)}
        hist = evaluate(model, mlp, training_data, eval_ids, {}, head_mask, possible, cfg, k_mlp, derived, stats)
        ci = bootstrap_ci(stats["videos"], [names[h] for h in heads], device) if args.lab_ci == "True" else None
        ap = step07_paper._ap(*hist["mlp"])
        per_head = {names[h]: {"mlp": _none(float(ap[h])), "onsets": int(stats["onsets"][h]),
                               "pos_rows": int(hist["mlp"][0][h].sum()),
                               **({"ci": ci.get(names[h], {})} if ci else {})} for h in heads}
        mtag = os.path.basename(str(cfg["mlp_model_file"])).replace(".pt", "")
        path = out_dir / f"mlp_only_{mtag}_{args.lab_eval_split}.json"
        utils_data.save_json({"mlp_model_file": str(cfg["mlp_model_file"]), "eval_videos": len(eval_ids),
                              "per_head": per_head}, path)
        print(f"[Rule lab] --lab_mlp_only: {len(heads)} heads, macro AP "
              f"{sum(v['mlp'] or 0 for v in per_head.values()) / max(len(per_head), 1):.3f}; saved {path}")
        return
    step(2, "literal masks (forbidden literals per variant)")
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
    forbid_old = torch.zeros_like(forbid_ns)
    if args.lab_shape_slot1 == "True":
        for key in ("shape", "gestalt"):            # gestalt v4: the group tracks' shape atoms carry their history too
            if key in layout:
                forbid_old |= atom_literals(model, *layout[key], older_only=True)

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

    # The gestalt atoms (`gestalt.py`): "_ng" variants go without them.
    forbid_ng = (atom_literals(model, *layout["gestalt"]) if "gestalt" in layout
                 else torch.zeros(model.num_literals, dtype=torch.bool))
    # Their negations ("!gst:near_1") were filler in gestalt v1 (rule lab 2026-11-23): off unless --lab_gst_negated.
    forbid_ng_neg = forbid_ng & (torch.arange(model.num_literals) >= model.num_inputs) \
        if args.lab_gst_negated != "True" else torch.zeros_like(forbid_ng)
    if "gestalt" in layout:
        print(f"[Rule lab] gestalt atoms {layout['gestalt']}: {int(forbid_ng.sum())} literals (off in the _ng variants)")

    prefixes = tuple(x for x in args.lab_forbid.split(",") if x)
    forbid_user = torch.zeros_like(forbid_nn)
    if prefixes:
        hit = {a for a, nm in enumerate(names) if nm.startswith(prefixes)}
        forbid_user = torch.tensor([info["kind"] in ("X", "!X") and info["atom"] in hit
                                    for info in literal_table(model)], dtype=torch.bool)
        print(f"[Rule lab] --lab_forbid {prefixes}: {len(hit)} atoms, {int(forbid_user.sum())} literals left out")

    def forbid_of(flags):
        d, nn, ns, nd, np_, ng = flags
        return forbid_user | ((forbid_nn if nn else torch.zeros_like(forbid_nn)) | (forbid_ns if ns else torch.zeros_like(forbid_ns))
                  | forbid_old | forbid_dir_neg | (forbid_dir if nd else torch.zeros_like(forbid_dir))
                  | (forbid_lin if np_ else (forbid_lin & (torch.arange(len(forbid_lin)) >= model.num_inputs)))
                  | (forbid_ng if ng else forbid_ng_neg))

    def make_ext(flags):
        forbid = forbid_of(flags)
        return ExtModel(model, derived, flags[0], forbid if bool(forbid.any()) else None)
    ext = {}
    L = make_ext((True, False, False, False, False, False)).num_literals
    step(3, "row sample (train, val)")
    train, val = mine_sweep.load_rows(cfg, training_data, model, head_mask, possible, args.output_dir / "mine_sweep")
    if args.lab_prepare_only == "True":
        if args.lab_eval_split in ("test", "valtest"):
            add_test_videos(training_data, cfg)         # built once here, not by each parallel head batch
        print("[Rule lab] --lab_prepare_only: the training data, the OnsetMLP and the row sample are cached; stop")
        return
    val_ids = list(training_data["val_videos"])
    step(4, "derived literals of the sampled rows")
    only = set(heads) if args.lab_heads else None
    if only is not None and any({"tr", "trp"} & set(parse_variant(v)[0]) for v in args.lab_variants.split(",")):
        srcs, _ = transfer_sources(names, heads, cfg["lab_all_heads"], train, cfg)
        only |= {g for v in srcs.values() for g in v}                 # keep the source heads' sample rows
        only |= {g for v in transfer_sources.mirror.values() for g in v}
    train = extend_rows(compact_rows(train, only), derived, model.num_literals, device, desc="Derived literals (train)")
    # The check videos: the even val videos; with the per-head selection only every other of them (0, 4, 8, ..): the
    # selection reads the rest (2, 6, 10, ..), never the videos the rules were filtered or confirmed on (2026-11-24:
    # the exh rules had to hold on the check videos, so they looked better there and the selection preferred them).
    check_step = 4 if args.lab_select == "True" and args.lab_select_split == "True" else 2
    check = extend_rows(compact_rows(mine_sweep.subset(val, set(range(0, len(val_ids), check_step))), only), derived,
                        model.num_literals, device, desc="Derived literals (check)")
    del val
    eval_ids = val_ids[1::2]
    if args.lab_eval_split == "test":
        eval_ids = add_test_videos(training_data, cfg)
    elif args.lab_eval_split == "valtest":                 # more eval onsets (2026-11-25): odd val + all test videos
        eval_ids = val_ids[1::2] + add_test_videos(training_data, cfg)

    # 1. Rule sets on the train sample, each miner once per literal set.
    step(5, f"mining ({args.lab_variants})")
    variants = args.lab_variants.split(",")
    mined = {}
    for name in variants:
        components, flags = parse_variant(name)
        for c in components:
            if (c, flags) not in mined:
                if flags not in ext:
                    ext[flags] = make_ext(flags)
                mined[(c, flags)] = mine_component(c, flags[0], ext[flags], names, heads, train, check, cfg, device)
    if args.lab_mine_only == "True":
        # A look at the mining before the slow part (counts, fits, evaluation), e.g. the main cues of "hier".
        print("[Rule lab] --lab_mine_only: stop after mining ("
              + ", ".join(f"{c}: {len(rs)} rules" for (c, _f), rs in mined.items()) + ")")
        return

    # 2. Exact counts on all train rows (each distinct rule once), then the final rule sets.
    step(6, "rule counts" + (" (train sample)" if args.lab_fast == "True" else " (all train rows)")
         + (" and check videos" if args.lab_check == "True" else ""))
    unique = {}
    for rules in mined.values():
        for r in rules:
            unique.setdefault((r["head"], tuple(sorted(r["literals"]))), r)
    keys = list(unique)
    if args.lab_fast == "True":
        # Quick test: the counts of the train sample (weighted rows, its onsets) instead of every train row.
        counts = sample_counts([unique[k] for k in keys], train, L, device, model.num_atoms)
    else:
        counts = recount(model, training_data, training_data["train_videos"], [unique[k] for k in keys], head_mask,
                         possible, cfg, derived)
    exact = {k: i for i, k in enumerate(keys)}
    if args.lab_check == "True":
        # The rules on the check videos (the even val videos, `check`): seen again, not contradicted.
        check_counts = sample_counts([unique[k] for k in keys], check, L, device, model.num_atoms)
    priors = [float(x) for x in args.lab_priors.split(",")]
    soft_modes = [m for m in args.lab_soft.split(",") if m] if args.lab_fast != "True" else ["logit"]
    # the size curve (MDL strengths); "s0.1": λ 0.1 scaled by the head's size (mode mdls_0.1)
    # "t0.1" / "ts0.1": over temporal rules (mdlt_0.1 / mdlts_0.1, `temporal_groups`)
    prefix = {"ts": "mdlts_", "t": "mdlt_", "s": "mdls_"}
    soft_modes += [next((v + x[len(k):] for k, v in prefix.items() if x.startswith(k)), f"mdl_{x}")
                   for x in args.lab_mdl.split(",") if x]
    global MDL_SIZE_REF
    MDL_SIZE_REF = args.lab_mdl_ref
    rule_sets, texts = {}, {}
    step(7, "per variant: finalize" + (", check filter" if args.lab_check == "True" else "")
         + (", merges" if args.lab_merge == "True" else "") + ", fits")
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
        full = make_ext((True, False, False, False, False, False))
        if args.lab_check == "True":
            n0 = len(final)
            final = check_filter(final, check_counts, exact)
            print(f"[Rule lab] {name}: check videos: {n0} -> {len(final)} rules (>= 1 check onset, lift >= 1)")
        if args.lab_merge_ratio > 0 and (args.lab_fast != "True" or args.lab_merge == "True"):
            pools = {}

            def pool_of(h):
                # One head's unpacked sample at a time: all 111 heads with ~4.4k literals took > 30 GB (v4 OOM).
                if h not in pools:
                    pools.clear()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    pools[h] = step05_mine._pool_of(train, h, L, device)
                return pools[h]
            n0 = len(final)
            if args.lab_merge == "True":
                final, n_cls = merge_classes(final, model, names, pool_of, args.lab_merge_ratio)
                print(f"[Rule lab] {name}: {n_cls} rules made class-free (veh:), {n0} -> {len(final)} rules")
            if args.lab_refine == "True" and not flags[3] and args.lab_fast != "True":
                final, n_ref = refine_sectors(final, model, names, pool_of, args.lab_min_events,
                                              forbid=forbid_of(flags).cpu())
                print(f"[Rule lab] {name}: {n_ref} rules with a direction narrowed to where their onsets concentrate")
            if args.lab_refine == "True" and not flags[4] and args.lab_fast != "True":
                final, n_ref = refine_intervals(final, model, names, pool_of, args.lab_min_events,
                                                forbid=forbid_of(flags).cpu())
                print(f"[Rule lab] {name}: {n_ref} rules with a position / size / distance interval narrowed")
            final, n_merges = merge_sectors(final, model, names, pool_of, args.lab_merge_ratio)
            final, n_lin = merge_intervals(final, model, names, pool_of, args.lab_merge_ratio)
            n_merges += n_lin
            if n_merges:
                print(f"[Rule lab] {name}: {n_merges} merges of close direction sectors, {n0} -> {len(final)} rules")
            pools.clear()
        logit = fit_logit(final, train, full, heads, device)
        tgroups = temporal_groups(final, lambda m: step05_mine.literal_text(full, names, m)) \
            if any(is_temporal(m) for m in soft_modes) else None
        if tgroups is not None:
            sizes = Counter(tgroups[0])
            multi = [g for g, c in sizes.items() if c > 1]
            print(f"[Rule lab] {name}: temporal rules: {len(multi)} of {len(sizes)} rules have >= 2 time slots "
                  f"({sum(sizes[g] for g in multi)} of {len(final)} mined rules)")
        soft = {m: fit_soft(final, confidence(final, counts["base"], SOFT_PRIOR), counts["base"], train, full,
                            heads, device, m, temporal=tgroups if is_temporal(m) else None) for m in soft_modes}
        for m, (g, _p) in soft.items():
            real = g[_pad([r["literals"] for r in final], L) != L] if final else g.flatten()
            print(f"[Rule lab] {name} soft {m}: {float((real < 0.5).float().mean()) if len(real) else 0:.1%} of the "
                  f"literals relaxed (gate < 0.5)")
        rule_sets[name] = RuleSet(final, counts["base"], priors, logit, L, A, device, soft)
        rule_sets[name].size = {}
        for m, (g, par) in soft.items():
            if is_compact(m) and final:
                n_eff, lits_eff, keep = effective(final, g, par[0])
                lens = (g >= DROP_GATE).sum(dim=1)[keep]
                if is_temporal(m):
                    n_eff, lits_eff, lens = effective_temporal(final, g, keep, tgroups)
                rule_sets[name].size[m] = (n_eff, lits_eff, int(lens.max()) if len(lens) else 0)
                print(f"[Rule lab] {name} {m}: {n_eff} of {len(final)} rules kept, {lits_eff:.2f} literals per rule")
        texts[name] = sorted(({"head": names[r["head"]], "precision": r["precision"], "events": r["events"],
                               "literals": [step05_mine.literal_text(full, names, m) for m in r["literals"]],
                               "derived": sum(m >= model.num_literals for m in r["literals"]),
                               **({"transfer": r["transfer"]} if r.get("transfer") else {}),
                               **({"check_events": r["check_events"], "check_lift": r["check_lift"]}
                                  if "check_events" in r else {}),
                               "weight": round(float(logit[0][i]), 3),
                               "soft": {m: {"weight": round(float(par[0][i]), 3),
                                            "gates": [round(float(x), 3) for x in g[i][:len(r["literals"])]]}
                                        for m, (g, par) in soft.items() if m != "nor"},
                               **({"temporal": [tgroups[0][i], tgroups[1][i]]} if tgroups is not None else {}),
                               **({k: r[k] for k in ("role", "main_id", "main_recall") if k in r})}
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
    step(8, "evaluation on the " + ("test videos" if args.lab_eval_split == "test" else "odd val videos"))
    stats = {"heads": list(heads)}
    hist = evaluate(model, mlp, training_data, eval_ids, rule_sets, head_mask, possible, cfg, k_mlp, derived, stats)
    # eval onsets of every learned head (not only --lab_heads): which heads need more data (v8)
    ons_all = {names[h]: int(stats["onsets"][h]) for h in cfg["lab_all_heads"]}
    few_all = sorted((k, n) for n, k in ons_all.items() if k < EVAL_MIN_ONSETS)
    print(f"[Rule lab] eval onsets: {len(ons_all) - len(few_all)} of {len(ons_all)} learned heads have >= "
          f"{EVAL_MIN_ONSETS}; fewer: " + ", ".join(f"{n} {k}" for k, n in few_all))
    choice = {}
    if args.lab_select == "True" and len(rule_sets) > 1:
        # Per-head variant selection on the check videos (even val), applied to the evaluation videos (odd val).
        check_ids = val_ids[2::4] if check_step == 4 else val_ids[0::2]   # the selection videos
        hist_c = evaluate(model, None, training_data, check_ids, rule_sets, head_mask, possible, cfg, k_mlp, derived)
        comps = {v: set(parse_variant(v)[0]) for v in variants}
        default = next((v for v in variants if "exh" in comps[v]), variants[0])
        small_variant = next((v for v in variants if {"tr", "trp"} & comps[v]), None)
        small = set()
        if small_variant:
            pairs = train["pairs"]
            for h in heads:
                sel = (pairs["head"] == h) & pairs["y"].bool()
                if int(pairs["event"][sel].unique().numel()) < cfg["lab_tr_max_onsets"]:
                    small.add(h)
        h_sel, choice, rs_sel = select_per_head(hist, hist_c, rule_sets, heads, args.lab_select_scorer, default,
                                                small, small_variant, args.lab_select_min_pos)
        print(f"[Rule lab] selection: transfer targets ({len(small)}) -> {small_variant}; others by the check AP with "
              f">= {args.lab_select_min_pos} positive check rows, else {default}")
        hist[f"selected/{args.lab_select_scorer.split(',')[0]}"] = h_sel
        rule_sets["selected"] = rs_sel
        print(f"[Rule lab] per-head selection ({args.lab_select_scorer}, on the check videos): "
              + ", ".join(f"{v} {n}" for v, n in Counter(choice.values()).most_common())
              + f" (chosen on {len(check_ids)} val videos the rules were not checked on)")
    if choice:
        # the per-head choice per video too (its bootstrap interval): each head's row from its chosen variant
        sc = args.lab_select_scorer.split(",")[0]
        for v in stats["videos"]:
            parts = [v.get(choice.get(h, "") if "/" in choice.get(h, "") else f"{choice.get(h, '')}/{sc}")
                     for h in heads]
            if all(p is not None for p in parts):
                v[f"selected/{sc}"] = tuple(torch.stack([p[k][j] for j, p in enumerate(parts)]) for k in (0, 1))
    ci_ref = args.lab_ci_ref or ("mlp" if mlp is not None else None)
    ci = bootstrap_ci(stats["videos"], [names[h] for h in heads], device, ref=ci_ref) \
        if args.lab_ci == "True" else None
    out = report(hist, names, heads, rule_sets, stats["onsets"], ci,
                 {"val": "odd val videos", "test": "test videos", "valtest": "odd val + test videos"}[args.lab_eval_split]
                 + f" ({len(eval_ids)})")
    if choice:
        out["selected"] = {names[h]: v for h, v in choice.items()}
    if out["best"]:
        best_variant = out["best"].split("/")[0]
        if best_variant not in texts:                       # "selected": show the best real variant's rules
            real = {k: v for k, v in out["summary"].items() if k.split("/")[0] in texts}
            best_variant = max(real, key=lambda k: real[k]["macro"]).split("/")[0] if real else variants[0]
        for m in soft_modes:
            if m == "nor":
                continue
            print(f"\n[Rule lab] strongest rules of {best_variant} / {m} (soft logistic weight; [gate] per literal, "
                  f"1 = required, 0 = ignored{'; dropped literals and rules left out' if is_compact(m) else ''}):")
            show_rules(texts[best_variant], [h for h in args.lab_show.split(",") if h], m)
    hier_out = {}
    for name_t in texts:
        if any(r.get("role") == "main" for r in texts[name_t]):
            for m in [m for m in soft_modes if m != "nor"]:
                hier_out[f"{name_t}/{m}"] = hierarchy(texts[name_t], m)
                print(f"\n[Rule lab] main and small cues of {name_t} / {m} (small cues act inside their main cue):")
                show_hierarchy(hier_out[f"{name_t}/{m}"], [names[h] for h in heads] if args.lab_heads
                               else [h for h in args.lab_show.split(",") if h])
        for m in (m for m in soft_modes if is_temporal(m)):
            if any("temporal" in r for r in texts[name_t]):
                print(f"\n[Rule lab] temporal rules of {name_t} / {m} (context & event[weight per time slot]):")
                show_temporal(texts[name_t], [names[h] for h in heads] if args.lab_heads
                              else [h for h in args.lab_show.split(",") if h], m)
    for key in (k for k in rule_sets if k.endswith("~crisp")):
        print(f"\n[Rule lab] crisp rules of {key} (hard AND, logistic weight):")
        show_rules(texts[key], [h for h in args.lab_show.split(",") if h], "crisp")
    step(9, "report and save")
    tag = os.path.basename(str(cfg["mined_seed_file"])).replace("mined_seeds_", "").replace(".json", "")
    rx = "-".join(k for k in args.lab_root_exclude.split(",") if k in layout)
    fb = "_fb" + "-".join(re.sub(r"[^A-Za-z0-9]", "", x) for x in args.lab_forbid.split(",") if x) if args.lab_forbid else ""
    fb += ("_cv" if args.lab_check == "True" else "") + ("_mg" if args.lab_merge == "True" else "")
    fb += {"test": "_test", "valtest": "_vt"}.get(args.lab_eval_split, "")
    fb += f"_{args.lab_tag}" if args.lab_tag else ""
    fb += "_nomir" if args.lab_tr_mirror != "True" and any({"tr", "trp"} & set(parse_variant(v)[0]) for v in variants) else ""
    path = out_dir / f"rule_lab_{tag}{cfg.get('target_tag', '')}_{'-'.join(variants)}{'_rx' + rx if rx else ''}{fb}.json"
    utils_data.save_json({"settings": {k: getattr(args, k) for k in vars(args) if k.startswith("lab_")},
                          "mlp_window": model.windows[k_mlp], "derived": derived.text, **out, "rules": texts,
                          "temporal": {f"{v}/{m}": temporal_profiles(texts[v], m) for v in texts for m in soft_modes
                                       if is_temporal(m) and any("temporal" in r for r in texts[v])},
                          "hierarchy": hier_out, "eval_onsets": ons_all, "eval_videos": len(eval_ids)}, path)
    print(f"[Rule lab] saved {path}")
    # The scorer of the paper protocol (`rule_paper.py`, 2026-11-26): per head the chosen variant's rules in the
    # selection's fit (body literal indices, soft gates, weights, the head's bias), all heads of this part.
    mode = args.lab_select_scorer.split(",")[0]
    pick = {}
    for h in heads:
        c = choice.get(h, variants[0])
        v, m = (c.split("/") + [mode])[:2] if "/" in c else (c, mode)
        if v in rule_sets and m in getattr(rule_sets[v], "soft", {}):
            pick[h] = (v, m)
    bodies, gates, weights, hs, bias = [], [], [], [], {}
    for h, (v, m) in pick.items():
        rs = rule_sets[v]
        gate, (w, b) = rs.soft[m]
        sel = torch.nonzero(rs.head == h).flatten()
        bias[int(h)] = float(b[h])
        if len(sel):
            bodies.append(rs.body[sel].cpu())
            gates.append(gate[sel].cpu())
            weights.append(w[sel].cpu())
            hs += [int(h)] * len(sel)
    K = max((x.shape[1] for x in bodies), default=1)
    padb = lambda x: torch.nn.functional.pad(x, (0, K - x.shape[1]), value=L)
    padg = lambda x: torch.nn.functional.pad(x, (0, K - x.shape[1]), value=1.0)
    torch.save({"num_literals": L, "num_atoms": A, "mode": mode, "heads": [int(h) for h in pick],
                "variant": {int(h): v for h, (v, _m) in pick.items()}, "bias": bias,
                "head": torch.tensor(hs, dtype=torch.long),
                "body": torch.cat([padb(x) for x in bodies]) if bodies else torch.zeros((0, K), dtype=torch.long),
                "gate": torch.cat([padg(x) for x in gates]) if gates else torch.zeros((0, K)),
                "weight": torch.cat(weights) if weights else torch.zeros(0)},
               str(path).replace(".json", ".rules.pt"))
    print(f"[Rule lab] rule scorer for the paper protocol: {str(path).replace('.json', '.rules.pt')} "
          f"({len(hs)} rules, {len(pick)} heads, {mode})")


if __name__ == "__main__":
    main()
