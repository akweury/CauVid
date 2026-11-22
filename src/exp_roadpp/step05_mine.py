"""Step 05 rule mining: short rules per head, mined from the train data, to seed the TemporalRuleNet
(cfg["anchor_source"] "mined", `TemporalRuleNet.seed_rules_mined`).

Why: on SUMO v7 (`rule_tree.py`, 2026-11-20) a rule list mined per head recovered the injected rules R3
("the car ahead just started braking and has not stopped -> the ego brakes", precision 0.68-0.84 val) and
R1 ("a pedestrian started crossing -> the ego brakes", 0.63) with <= 3 literals each, and with the ego motion
atoms 6-8 rules gave the ego's AV-Brake a val AP of 0.44 and AV-Stop 0.46 (the LLM / causal anchors seeded
~15k rule nodes and gave 0.38 / 0.22). The mined rules are short, readable and hold on val; the rule net
starts from them, learns their weights and may add literals.

Rows and labels as the clauses (`step05._rule_head_counts` at f = 0): per head, the at-risk rows of the
agents that can hold it (`step05.class_possible`); a hit if the head switches on within clause_slots slots.
The literals are those of the rule net (`TemporalRuleNet.literal_values`: inputs, negated inputs "!X",
groups "any motor vehicle X" and their negations), so a mined rule is a rule node of the net.

Sample: per head at most mine_pos_cap hit rows and mine_neg_cap other rows of the train videos, drawn at
random and weighted back (weight 1 / sampling rate), so the precisions are unbiased; the same for val. The
draw of a row is a hash of (seed, video, frame, agent, head) (`_row_uniform`), so it is reproducible.
The rows are kept bit-packed (one bit per literal).

Mining per head:
  roots: up to mine_roots own-agent positive literals (the agent itself holds X in some slot), chosen
      greedily by the share of the head's hits they cover (>= mine_root_min_cover of the hits not yet
      covered): every rule holds an own literal, so the rule net's self gate (require_self) keeps it;
  rule list per root (as `rule_tree` --mode list): on the root's rows not yet covered, grow a tree of depth
      mine_depth (weighted entropy gain >= mine_min_gain, the literal's branch with >= mine_min_events
      onsets in >= min_videos videos); turn each node (inner nodes too) into a conjunction of literals (a "not literal" branch
      becomes the complementary literal: X <-> !X, G <-> !G; unknown rows drop out); take the leaf of the
      largest FOIL gain h * (log p - log p_rest) among those with precision >= mine_min_lift x the root's and
      enough onsets; remove the rows it covers; repeat up to mine_max_rules times;
  val check: a rule is kept as a seed only if its val precision is >= mine_val_lift x the root's val
      precision (it holds on unseen videos), else its rows are still removed (the search moves on).
  root seeds: every root is also a seed of its own ("root": True), the head's default rule; the first
      root of a head is taken if it covers any hit, and a head with < mine_min_events onsets gets its
      roots but no mined rules (so every head with train onsets starts from at least one rule).
Each seed: {"head", "literals": [root, ...], "text", train / val precision and onsets, "confidence"
(the train precision shrunk to the root's with mine_prior onsets, `shrunk_confidence`)[, "root"]}.
"""
import math
import os

import torch
from tqdm import tqdm

from src.exp_roadpp import step05, utils_data


def _pack(bits):
    """(B, L) bool -> (B, ceil(L / 8)) uint8."""
    B, L = bits.shape
    pad = (-L) % 8
    if pad:
        bits = torch.cat([bits, bits.new_zeros((B, pad))], dim=1)
    weights = (2 ** torch.arange(8, device=bits.device)).view(1, 1, 8)
    return (bits.view(B, -1, 8).long() * weights).sum(dim=2).to(torch.uint8)


def _unpack(packed, L, device):
    """(n, ceil(L / 8)) uint8 -> (n, L) float on device."""
    shifts = torch.arange(8, device=device)
    bits = (packed.to(device).long()[..., None] >> shifts) & 1
    return bits.view(len(packed), -1)[:, :L].float()


def _head_rows(rows, can, head_mask, S, w):
    """keep (B, A) the at-risk rows of agents of the head's class, y (B, A) their hits (onset within w slots)."""
    first, at_risk = rows["first"], rows["at_risk"]
    keep = at_risk & can[rows["rows"][:, 1]] & head_mask[None] & rows["known"][:, w - 1]
    return keep, keep & (first < w * S)


@torch.no_grad()
def collect(model, training_data, video_ids, head_mask, possible, cfg, desc):
    """Bit-packed literal rows sampled per head: {"packed": (R, L/8) uint8, "pairs": {head, row, y, weight,
    video, event} (P,) tensors, "rate": (A,) hit rows / at-risk rows (all rows, the base rate)}."""
    device, S, w = cfg["device"], model.slot_size, int(cfg.get("clause_slots", 1))
    A = model.num_atoms
    elig, pos = torch.zeros(A, device=device), torch.zeros(A, device=device)
    chunk = int(cfg.get("mine_chunk", 256))         # frames per row batch (larger: fewer, bigger GPU calls)
    for video_id in tqdm(video_ids, desc=f"{desc}: counting"):
        can = possible(video_id)
        for frames in training_data["frames"][video_id].split(chunk):
            rows = step05._video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            keep, y = _head_rows(rows, can, head_mask, S, w)
            elig += keep.sum(dim=0)
            pos += y.sum(dim=0)
    p_pos = (cfg["mine_pos_cap"] / pos.clamp(min=1)).clamp(max=1.0)
    p_neg = (cfg["mine_neg_cap"] / (elig - pos).clamp(min=1)).clamp(max=1.0)
    packed, parts, offset = [], {k: [] for k in ("head", "row", "y", "weight", "video", "event")}, 0
    for vi, video_id in enumerate(tqdm(video_ids, desc=f"{desc}: sampling")):
        can = possible(video_id)
        for frames in training_data["frames"][video_id].split(chunk):
            rows = step05._video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            keep, y = _head_rows(rows, can, head_mask, S, w)
            t, n = rows["rows"].T
            r = _row_uniform(vi, t, n, A, int(cfg["seed"]))
            take = keep & torch.where(y, r < p_pos[None], r < p_neg[None])
            sel = take.any(dim=1)
            if not sel.any():
                continue
            packed.append(_pack(model.literal_values(rows["inputs"][sel]) > 0.5).cpu())
            local, h = torch.nonzero(take[sel], as_tuple=True)
            t, n = t[sel][local], n[sel][local]
            hit = y[sel][local, h]
            onset = t + rows["first"][sel][local, h]
            parts["head"].append(h.cpu())
            parts["row"].append((local + offset).cpu())
            parts["y"].append(hit.cpu())
            parts["weight"].append(torch.where(hit, 1 / p_pos[h], 1 / p_neg[h]).cpu())
            parts["video"].append(torch.full_like(h, vi).cpu())
            parts["event"].append(torch.where(hit, (vi << 40) | (n.long() << 20) | onset.long(),
                                              torch.full_like(onset, -1)).cpu())
            offset += int(sel.sum())
    width = (model.num_literals + 7) // 8
    return {"packed": torch.cat(packed) if packed else torch.zeros((0, width), dtype=torch.uint8),
            "pairs": {k: torch.cat(v) if v else torch.zeros(0) for k, v in parts.items()},
            "rate": (pos / elig.clamp(min=1)).cpu()}


_MASK32 = 0xFFFFFFFF


def _mix32(x):
    """lowbias32 integer hash of (..) int64 values in [0, 2^32) (the low 32 bits of a product are exact
    even if the int64 product wraps)."""
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & _MASK32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & _MASK32
    return x ^ (x >> 16)


def _row_uniform(video, t, n, A, seed):
    """(B, A) numbers in [0, 1) fixed by (seed, video, frame t, agent n, head): the row sample no longer
    depends on the batch size, the order of the GPU calls or the chunk size (the generator stream did:
    the m1 and m2 runs mined from different samples, 4 vs 1 rules for AV-Brake)."""
    h = torch.arange(A, device=t.device)[None, :]
    x = _mix32(torch.full((len(t), 1), seed & _MASK32, dtype=torch.long, device=t.device))
    for v in (torch.full_like(t, video)[:, None], t.long()[:, None], n.long()[:, None], h):
        x = _mix32(x ^ (v & _MASK32))
    return x.double() / 2 ** 32


def _pool_of(data, head, L, device):
    """The sampled rows of one head: F (n, L) float, y (n,) bool, w (n,) float, video, event (n,) long."""
    pairs = data["pairs"]
    sel = pairs["head"] == head
    return {"F": _unpack(data["packed"][pairs["row"][sel]], L, device), "y": pairs["y"][sel].bool().to(device),
            "w": pairs["weight"][sel].float().to(device), "video": pairs["video"][sel].long().to(device),
            "event": pairs["event"][sel].long().to(device)}


def _distinct(ids, F_hit):
    """(L,) number of distinct ids among the hit rows where each literal holds."""
    if len(ids) == 0:
        return torch.zeros(F_hit.shape[1], device=F_hit.device)
    unique, inverse = ids.unique(return_inverse=True)
    per_id = torch.zeros((len(unique), F_hit.shape[1]), device=F_hit.device).index_add_(0, inverse, F_hit)
    return (per_id > 0).sum(dim=0)


def _stats(pool, mask):
    """Weighted rows, weighted hits, onsets and videos of the rows mask."""
    y = pool["y"] & mask
    return {"rows": float(pool["w"][mask].sum()), "hits": float(pool["w"][y].sum()),
            "events": int(pool["event"][y].unique().numel()), "videos": int(pool["video"][y].unique().numel())}


def _entropy(h, n):
    p = (h / n.clamp(min=1e-9)).clamp(1e-12, 1 - 1e-12)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def _grow(pool, idx, depth, allowed, cfg, path=()):
    """Nodes [path of (literal, value)] of a weighted entropy tree on the rows idx of pool: every node, not
    only the leaves, so a short rule is not passed over for a longer one that only adds a noise split."""
    st = _stats(pool, _mask_of(pool, idx))
    if depth >= cfg["mine_depth"] or st["events"] < 2 * cfg["mine_min_events"] or st["hits"] >= st["rows"]:
        return [path]
    F, w, y = pool["F"][idx], pool["w"][idx], pool["y"][idx]
    wy = w * y
    n, h = w.sum(), wy.sum()
    n1, h1 = w @ F, wy @ F
    n0, h0 = n - n1, h - h1
    gain = _entropy(h, n) - (n1 / n) * _entropy(h1, n1) - (n0 / n) * _entropy(h0, n0)
    F_hit = F[y]
    ok = (allowed & (_distinct(pool["event"][idx][y], F_hit) >= cfg["mine_min_events"])
          & (_distinct(pool["video"][idx][y], F_hit) >= cfg["min_videos"]) & (n1 > 0) & (n0 > 0))
    gain = torch.where(ok, gain, torch.full_like(gain, -1.0))
    best = int(gain.argmax())
    if gain[best] < cfg["mine_min_gain"]:
        return [path]
    yes = F[:, best] > 0.5
    return ([path] + _grow(pool, idx[yes], depth + 1, allowed, cfg, path + ((best, True),))
            + _grow(pool, idx[~yes], depth + 1, allowed, cfg, path + ((best, False),)))


def _mask_of(pool, idx):
    mask = torch.zeros(len(pool["y"]), dtype=torch.bool, device=pool["y"].device)
    mask[idx] = True
    return mask


def _holds(pool, literals):
    """(n,) bool: every literal holds (known true)."""
    if not literals:
        return torch.ones(len(pool["y"]), dtype=torch.bool, device=pool["y"].device)
    return (pool["F"][:, literals] > 0.5).all(dim=1)


def _complement(model, m):
    """The literal of "not m" (X <-> !X, G <-> !G), None if the model has no such literal."""
    M, G = model.num_inputs, model.num_groups
    kind = int(model.literal_kind[m])
    if kind > 3:                    # a derived literal (`rule_lab.Derived`): no complement
        return None
    if kind == 0:
        return M + m if model.negation else None
    if kind == 1:
        return m - M
    return m + G if kind == 2 else m - G


def literal_text(model, names, m):
    text_of = getattr(model, "text_of", None)          # derived literals (`rule_lab.ExtModel`)
    if text_of is not None and text_of(m) is not None:
        return text_of(m)
    info = model.literal_info(m)
    scope = ["own", "other", *model.roles][info["scope"]]
    if info["kind"] in ("G", "!G"):
        return f"{'!' if info['kind'] == '!G' else ''}veh:{info['group']}({scope})@{info['j']}"
    return f"{'!' if info['kind'] == '!X' else ''}{names[info['atom']]}({scope})@{info['j']}"


def shrunk_confidence(precision, events, root_precision, prior):
    """A rule's confidence: its train precision shrunk to its root's with `prior` onsets of weight.

    The rule fired on about events / precision independent episodes (its distinct onsets over its precision),
    so (events + prior x root) / (events / precision + prior). A rule of 5 onsets at 0.9 over a root at 0.1
    gets 0.27, one of 370 onsets at 0.80 gets 0.77. Why: with mine_min_events 5 the raw precision of the
    small rules was too high (luck), they ranked above every head's rows and the micro AP fell 0.265 -> 0.233
    (`mine_sweep`, SUMO v7, 2026-11-21). prior 0 gives the raw precision."""
    if prior <= 0 or precision <= 0:
        return float(precision)
    return float((events + prior * root_precision) / (events / precision + prior))


def mine_head(model, names, head, train, val, cfg, device):
    """Seeds [{"head", "literals", "text", ...}] of one head (module doc) and a report."""
    L, A, S_ = model.num_literals, model.num_atoms, model.num_spans
    tr, va = _pool_of(train, head, L, device), _pool_of(val, head, L, device)
    report = {"train_rows": len(tr["y"]), "train_events": _stats(tr, torch.ones_like(tr["y"]))["events"], "roots": []}
    mine_rules = report["train_events"] >= cfg["mine_min_events"]
    base = model.literal_base.to(device)
    scope = base // (S_ * A)
    allowed = model.input_mask.to(device) & ~((base % A == head) & (scope == 0))     # not the head itself
    own_pos = allowed & (model.literal_kind.to(device) == 0) & (scope == 0)
    # Atoms that may not be roots ([lo, hi) atom ranges, e.g. the shape atoms): an input that holds for almost
    # every agent ("shp:app_middle") covers the most hits and became the root of 442 of 882 rules (rule lab,
    # 2026-11-22), so every rule had to hold it. They stay usable inside the rules.
    for lo, hi in cfg.get("mine_root_exclude") or ():
        atom = base % A
        own_pos &= ~((atom >= lo) & (atom < hi))
    wy = tr["w"] * tr["y"]
    total = float(wy.sum())
    seeds, remaining = [], tr["y"].clone()
    for _ in range(cfg["mine_roots"]):
        cover = ((tr["w"] * remaining) @ tr["F"]) / max(total, 1e-9)
        cover = torch.where(own_pos, cover, torch.full_like(cover, -1.0))
        root = int(cover.argmax())
        # The first root only needs to cover some hits (it is the head's default rule); further roots
        # need mine_root_min_cover of the hits not yet covered.
        if float(cover[root]) <= 0 or (report["roots"] and float(cover[root]) < cfg["mine_root_min_cover"]):
            break
        remaining &= tr["F"][:, root] <= 0.5
        root_tr, root_va = _holds(tr, [root]), _holds(va, [root])
        st_root, sv_root = _stats(tr, root_tr), _stats(va, root_va)
        p_root = st_root["hits"] / max(st_root["rows"], 1e-9)
        p_root_val = sv_root["hits"] / max(sv_root["rows"], 1e-9)
        root_report = {"root": literal_text(model, names, root), "literal": root, "cover": float(cover[root]),
                       "precision": p_root, "val_precision": p_root_val, "rules": 0, "rejected_val": 0}
        # The root alone is a seed too (the head's default rule, "root": True): it keeps the recall of the
        # rows no mined rule covers, and a head too rare to mine still starts from a rule (an unseeded rule
        # starts with every gate closed and its self gate ~0, so it gets almost no gradient).
        seeds.append({"head": int(head), "literals": [int(root)], "root": True,
                      "text": f"{literal_text(model, names, root)} => {names[head]}",
                      "precision": p_root, "events": st_root["events"], "videos": st_root["videos"],
                      "val_precision": p_root_val, "val_events": sv_root["events"], "root_precision": p_root,
                      "root_val_precision": p_root_val, "confidence": p_root})
        residual = torch.nonzero(root_tr).flatten()
        for _k in range(cfg["mine_max_rules"] if mine_rules else 0):
            left = _stats(tr, _mask_of(tr, residual))
            if left["events"] < cfg["mine_min_events"] or left["hits"] <= 0:
                break
            p_left = left["hits"] / max(left["rows"], 1e-9)
            best = None
            for path in _grow(tr, residual, 0, allowed, cfg):
                lits = []
                for m, value in path:
                    c = m if value else _complement(model, m)
                    if c is not None and c not in lits:
                        lits.append(c)
                if not lits:
                    continue
                mask = _mask_of(tr, residual) & _holds(tr, lits)
                st = _stats(tr, mask)
                if st["events"] < cfg["mine_min_events"] or st["videos"] < cfg["min_videos"] or st["rows"] <= 0:
                    continue
                p = st["hits"] / st["rows"]
                if p < cfg["mine_min_lift"] * p_root:
                    continue
                foil = st["hits"] * (math.log(max(p, 1e-12)) - math.log(max(p_left, 1e-12)))
                if best is None or foil > best[0]:
                    best = (foil, lits, mask, st, p)
            if best is None:
                break
            _foil, lits, mask, st, p = best
            residual = residual[~mask[residual]]
            sv = _stats(va, root_va & _holds(va, lits))
            p_val = sv["hits"] / sv["rows"] if sv["rows"] > 0 else None
            if p_val is None or p_val < cfg["mine_val_lift"] * p_root_val:
                root_report["rejected_val"] += 1
                continue
            literals = [root] + lits
            seeds.append({"head": int(head), "literals": [int(m) for m in literals],
                          "text": " & ".join(literal_text(model, names, m) for m in literals) + f" => {names[head]}",
                          "precision": p, "events": st["events"], "videos": st["videos"],
                          "val_precision": p_val, "val_events": sv["events"], "root_precision": p_root,
                          "root_val_precision": p_root_val,
                          "confidence": shrunk_confidence(p, st["events"], p_root, cfg.get("mine_prior", 0))})
            root_report["rules"] += 1
        report["roots"].append(root_report)
    return seeds, report


@torch.no_grad()
def mine(training_data, cfg):
    """{"seeds": [...], "heads": {name: report}, ...} of every learned head (module doc)."""
    device = cfg["device"]
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    head_mask = training_data["target_heads"].to(device)
    possible = step05.class_possible(training_data, vocab, device)
    train = collect(model, training_data, training_data["train_videos"], head_mask, possible, cfg, "Mining (train)")
    val = collect(model, training_data, training_data["val_videos"], head_mask, possible, cfg, "Mining (val)")
    seeds, reports = [], {}
    for head in tqdm(torch.nonzero(head_mask).flatten().tolist(), desc="Mining rules"):
        s, rep = mine_head(model, names, head, train, val, cfg, device)
        seeds += s
        reports[names[head]] = {**rep, "seeds": len(s), "mined": sum(1 for sd in s if not sd.get("root"))}
    covered = sum(1 for r in reports.values() if r["mined"])
    num_roots = sum(1 for sd in seeds if sd.get("root"))
    print(f"[Mining] {len(seeds) - num_roots} rules for {covered} of {len(reports)} heads, plus {num_roots} root "
          f"rules for {sum(1 for r in reports.values() if r['roots'])} heads ({model.num_literals} literals: "
          f"{model.num_inputs} inputs, negation {model.negation}, {model.num_groups} groups; hits within "
          f"{cfg.get('clause_slots', 1)} slot(s))")
    for sd in sorted((sd for sd in seeds if not sd.get("root")), key=lambda x: -x["events"])[:30]:
        print(f"  {sd['precision']:.2f} / val {sd['val_precision']:.2f} / conf {sd['confidence']:.2f}  "
              f"{sd['events']:5d} onsets  {sd['text']}")
    return {"version": step05.MINE_VERSION, "num_literals": model.num_literals, "seeds": seeds, "heads": reports,
            "settings": {k: cfg[k] for k in cfg if k.startswith("mine_")} | {"clause_slots": cfg.get("clause_slots")}}


def load_or_mine(training_data, cfg):
    """The seeds of cfg["mined_seed_file"], mined (`mine`) if it is missing or older than the training data."""
    path = cfg["mined_seed_file"]
    if not (os.path.exists(path) and step05.is_fresh(path, cfg["training_data_file"])):
        utils_data.save_json(mine(training_data, cfg), path)
    out = utils_data.load_json(path)
    model_literals = step05.build_rule_net(training_data, cfg, 1).num_literals
    if out.get("num_literals") != model_literals:
        raise ValueError(f"{path} was mined for {out.get('num_literals')} literals, the model has {model_literals}: "
                         f"delete it to mine again")
    return out["seeds"]
