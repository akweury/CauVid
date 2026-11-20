"""Refine one high-recall root rule into sub-rules. Count only, no training.

Given a head and a root body R (e.g. "the ego moves & some vehicle in a VehLane brakes" for av:AV-Brake),
find sub-rules R & literals with a higher precision, explaining as many of R's hits as possible. A literal is
true ("X@j") or false and known ("!X@j"), on one input (atom, scope, slot j) of the rule net's language
(`TemporalRuleNet` scopes own / other / roles, `step05.rule_input_mask`) or on a group "veh:X" (X of any motor
vehicle class, in one scope and slot: true if one of them is, false if all are false and known).

Rows and labels as `step05._rule_head_counts` at f = 0: the at-risk rows of the agents that can hold the head
(`step05.class_possible`), positive (a hit) if the head switches on within the next slot. One onset gives
up to slot_size hit rows, so the support of a sub-rule is counted in events (distinct onsets, >= --min_events)
and videos (>= min_videos), not in rows. Everything is fitted on the train videos; the val videos are only
routed and scored.

Two modes (--mode, default both):
  tree   a greedy decision tree on the root rows (entropy gain). The leaves partition the root rows: their hits
         add up to the root's; a split raises one branch's precision and lowers the other's.
  cover  sequential covering, a rule list aimed at the widest coverage: grow a sub-rule on the rows not yet
         covered (add the literal of the largest FOIL gain h1 * (log p1 - log p) until its precision reaches
         --target_prec or --max_len literals), keep it if its precision is >= --min_lift x that of the rows
         not yet covered, remove the rows it covers, repeat until none is kept or fewer than --min_events
         events are left. What is left at the end is the residual nothing explains. The coverage curve gives,
         per precision level p, the share of the root hits the sub-rules of train precision >= p cover.

Report: per leaf / sub-rule train and val rows, hits, events, precision, share of the root hits; cover: the
cumulative share of the root hits explained (val); the val AP of scoring the rows by their leaf / sub-rule's
train precision vs by the root alone (within the root rows, and over all at-risk rows).

    ./run_roadpp_docker.sh --gpu 0 --rule_tree -- --exp sumo_maps_v7 --machine dgx-jing --roles ahead
    options: --head av:AV-Brake --root "av:AV-Mov(own)@1 & veh:Brake(other)@1 & veh:VehLane(other)@1"
             --mode both --max_depth 3 --min_gain 0.001 --min_events 20
             --target_prec (default 2 x the root precision) --max_len 5 --max_rules 200 --min_lift 1.2
Root syntax: literals joined by "&", each "[!]Name1|Name2(scope)@j" (OR over the names; "!": all false and
known); "veh:X" stands for X of every motor vehicle class; scope own, other or a role of --roles.
Writes <output_dir>/rule_tree/<head>__<root tag>[_r<roles>].json and prints the tree and the rule list.
"""
import math
import re
from collections import defaultdict

import torch
from tqdm import tqdm

from src.exp_roadpp import config, step05, utils_data, utils_pipe
from src.exp_roadpp.cauvid import TemporalRuleNet

MOTOR = ("Car", "MedVeh", "LarVeh", "Bus", "Mobike")
DEFAULT_ROOTS = {"av:AV-Brake": "av:AV-Mov(own)@1 & veh:Brake(other)@1 & veh:VehLane(other)@1"}
LITERAL = re.compile(r"^(!?)(.+)\((\w+)\)@(\d+)$")


def parse_root(text, index, scopes):
    """[(atoms, scope, j, positive)] of the root syntax (module doc)."""
    literals = []
    for part in text.split("&"):
        m = LITERAL.match(part.strip())
        if not m:
            raise SystemExit(f"cannot parse root literal {part.strip()!r} (expected [!]Name|Name(scope)@j)")
        neg, names, scope, j = m.groups()
        expanded = []
        for name in names.split("|"):
            expanded += [f"{c}:{name[4:]}" for c in MOTOR] if name.startswith("veh:") else [name]
        atoms = [i for n in expanded for i in index.get(n, [])]
        if not atoms:
            raise SystemExit(f"no atom of {expanded} in the vocabulary")
        if scope not in scopes:
            raise SystemExit(f"scope {scope} not in {list(scopes)} (add --roles for a role scope)")
        literals.append((atoms, scopes[scope], int(j), not neg))
    return literals


def vehicle_groups(model, names, scope_names):
    """[(name, input indices)]: "veh:X(scope)@j" for every label X held by >= 2 motor vehicle classes."""
    by_label = defaultdict(list)
    for i, n in enumerate(names):
        cls, _, label = n.partition(":")
        if cls in MOTOR:
            by_label[label].append(i)
    groups = []
    for label, atoms in sorted(by_label.items()):
        if len(atoms) < 2:
            continue
        for scope, scope_name in enumerate(scope_names):
            for j in range(1, model.num_slots + 1):
                idx = [model.input_index(a, scope, j) for a in atoms]
                if model.input_mask[idx].all():
                    groups.append((f"veh:{label}({scope_name})@{j}", idx))
    return groups


@torch.no_grad()
def collect(model, training_data, video_ids, head, root, possible, groups, cfg, desc):
    """Rows where the root holds (on the at-risk rows of agents that can hold head): F (R, 2M + 2G) uint8
    literal values (inputs true, inputs false and known, groups true, groups false and known), y (R,) hit,
    v (R,) video index, e (R,) event id of a hit (-1 otherwise); and the totals of all at-risk rows."""
    device, S = cfg["device"], model.slot_size
    w = cfg.get("hit_slots", 1)                     # a hit: the head switches on within the next w slots
    index_of = [torch.tensor([model.input_index(a, s, j) for a in atoms], device=device)
                for atoms, s, j, _p in root]
    group_idx = [torch.tensor(idx, device=device) for _name, idx in groups]
    F, Y, V, E = [], [], [], []
    total_rows = total_hits = 0
    events = set()
    for vi, video_id in enumerate(tqdm(video_ids, desc=desc)):
        can = possible(video_id)[:, head]
        for frames in training_data["frames"][video_id].split(cfg["frames_per_video"]):
            rows = step05._video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            x, k = model.split_inputs(rows["inputs"])
            first, at_risk = rows["first"][:, head], rows["at_risk"][:, head]
            keep = at_risk & can[rows["rows"][:, 1]] & (first >= 0) & rows["known"][:, w - 1, head]
            y = keep & (first < w * S)
            total_rows += int(keep.sum())
            total_hits += int(y.sum())
            t, n = rows["rows"].T
            event = (vi << 40) | (n.long() << 20) | (t.long() + first.long())
            events.update(event[y].tolist())
            holds = keep.clone()
            for idx, (_atoms, _s, _j, positive) in zip(index_of, root):
                holds &= (x[:, idx].amax(dim=1) > 0) if positive else (((1 - x[:, idx]) * k[:, idx]).amin(dim=1) > 0)
            if not holds.any():
                continue
            xh, kh = x[holds], k[holds]
            neg = (1 - xh) * kh
            gpos = [xh[:, g].amax(dim=1) for g in group_idx]
            gneg = [neg[:, g].amin(dim=1) for g in group_idx]
            parts = [xh, neg] + ([torch.stack(gpos, dim=1), torch.stack(gneg, dim=1)] if groups else [])
            F.append(torch.cat(parts, dim=1).to(torch.uint8).cpu())
            yh = y[holds]
            Y.append(yh.cpu())
            V.append(torch.full((int(holds.sum()),), vi, dtype=torch.long))
            E.append(torch.where(yh, event[holds], torch.full_like(event[holds], -1)).cpu())
    width = 2 * model.num_inputs + 2 * len(groups)
    cat = lambda parts, empty: torch.cat(parts) if parts else empty
    return {"F": cat(F, torch.zeros((0, width), dtype=torch.uint8)), "y": cat(Y, torch.zeros(0, dtype=torch.bool)),
            "v": cat(V, torch.zeros(0, dtype=torch.long)), "e": cat(E, torch.zeros(0, dtype=torch.long)),
            "total_rows": total_rows, "total_hits": total_hits, "total_events": len(events)}


def distinct(ids, F_hit, device):
    """(features,) number of distinct ids (videos or events) among the hit rows where each literal holds."""
    if len(ids) == 0:
        return torch.zeros(F_hit.shape[1], device=device)
    unique, inverse = ids.unique(return_inverse=True)
    per_id = torch.zeros((len(unique), F_hit.shape[1]), device=device).index_add_(0, inverse.to(device), F_hit)
    return (per_id > 0).sum(dim=0)


def literal_stats(data, idx, device):
    """F (n, L) float of the rows idx and per literal: rows, hits, events, videos where it holds."""
    F = data["F"][idx].to(device).float()
    y = data["y"][idx]
    yd = y.to(device)
    F_hit = F[yd]
    return F, {"rows": F.sum(dim=0), "hits": yd.float() @ F,
               "events": distinct(data["e"][idx][y], F_hit, device),
               "videos": distinct(data["v"][idx][y], F_hit, device)}


def node_stats(data, idx):
    y = data["y"][idx]
    return {"rows": len(idx), "hits": int(y.sum()), "events": int(data["e"][idx][y].unique().numel()),
            "videos": int(data["v"][idx][y].unique().numel())}


def entropy(h, n):
    p = (h / n.clamp(min=1)).clamp(1e-12, 1 - 1e-12)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def supported(s, allowed, cfg, args):
    return allowed & (s["events"] >= args.min_events) & (s["videos"] >= cfg["min_videos"]) & (s["hits"] > 0)


def grow(data, idx, depth, allowed, cfg, args, device):
    """Tree mode: a node {stats, split: {feature, yes, no} | None} of the train rows idx (long tensor)."""
    node = {**node_stats(data, idx), "split": None}
    n, h = node["rows"], node["hits"]
    if depth >= args.max_depth or node["events"] < 2 * args.min_events or h == n:
        return node
    F, s = literal_stats(data, idx, device)
    n1, h1 = s["rows"], s["hits"]
    n0, h0 = n - n1, h - h1
    nn, hh = torch.tensor(float(n), device=device), torch.tensor(float(h), device=device)
    gain = entropy(hh, nn) - (n1 / n) * entropy(h1, n1) - (n0 / n) * entropy(h0, n0)
    ok = supported(s, allowed, cfg, args) & (n0 > 0)
    gain = torch.where(ok, gain, torch.full_like(gain, -1.0))
    best = int(gain.argmax())
    if gain[best] < args.min_gain:
        return node
    mask = F[:, best].bool().cpu()
    node["split"] = {"feature": best, "gain": float(gain[best]),
                     "yes": grow(data, idx[mask], depth + 1, allowed, cfg, args, device),
                     "no": grow(data, idx[~mask], depth + 1, allowed, cfg, args, device)}
    return node


def tree_leaves(node, out, path=()):
    """[(path, train node)] of the tree."""
    if node["split"] is None:
        out.append((path, node))
        return out
    f = node["split"]["feature"]
    tree_leaves(node["split"]["yes"], out, path + ((f, True),))
    tree_leaves(node["split"]["no"], out, path + ((f, False),))
    return out


def holds_path(data, idx, path):
    """(len(idx),) bool: every (literal, value) of path holds on the rows idx."""
    mask = torch.ones(len(idx), dtype=torch.bool)
    for f, value in path:
        mask &= data["F"][idx, f].bool() == value
    return mask


def cover(data, allowed, cfg, args, target, device):
    """Cover mode: [(literals, train stats on the rows it covers in list order)] and the residual stats.
    A sub-rule is kept if its precision is >= min_lift x the precision of the rows not yet covered (it explains
    some of them better than they are explained now); the list ends when none is, or fewer than min_events
    events are left."""
    residual = torch.arange(len(data["y"]))
    rules = []
    for _ in tqdm(range(args.max_rules), desc="Covering"):
        left = node_stats(data, residual)
        if left["events"] < args.min_events:
            break
        p_left = left["hits"] / max(left["rows"], 1)
        cur, lits = residual, []
        while len(lits) < args.max_len:
            st = node_stats(data, cur)
            if st["hits"] == 0:
                break
            p = st["hits"] / st["rows"]
            F, s = literal_stats(data, cur, device)
            p1 = s["hits"] / s["rows"].clamp(min=1)
            foil = s["hits"] * (torch.log(p1.clamp(min=1e-12)) - math.log(p))
            ok = supported(s, allowed, cfg, args) & (p1 > p)
            foil = torch.where(ok, foil, torch.full_like(foil, -1.0))
            best = int(foil.argmax())
            if foil[best] <= 0:
                break
            cur = cur[F[:, best].bool().cpu()]
            lits.append(best)
            if p1[best] >= target:
                break
        if not lits:
            break
        st = node_stats(data, cur)
        if st["hits"] / max(st["rows"], 1) < args.min_lift * p_left:
            break
        rules.append((lits, st))
        covered = torch.zeros(len(data["y"]), dtype=torch.bool)
        covered[cur] = True
        residual = residual[~covered[residual]]
    return rules, node_stats(data, residual)


def average_precision(scores, labels):
    """AP with tied scores grouped (a tie block counts as one threshold)."""
    if labels.sum() == 0:
        return None
    order = torch.argsort(scores, descending=True)
    s, l = scores[order], labels[order].float()
    _keys, counts = torch.unique_consecutive(s, return_counts=True)
    ends = counts.cumsum(0) - 1
    tp = l.cumsum(0)[ends]
    precision, recall = tp / (ends + 1), tp / l.sum()
    prev = torch.cat([torch.zeros(1), recall[:-1]])
    return float(((recall - prev) * precision).sum())


def ap_report(val, score, root_rate, rest_rate):
    """Val AP of the root rows scored by score vs by the root alone, within the root and over all rows."""
    y = val["y"]
    out_rows, out_hits = val["total_rows"] - len(y), val["total_hits"] - int(y.sum())
    rest = torch.cat([torch.ones(out_hits, dtype=torch.bool), torch.zeros(out_rows - out_hits, dtype=torch.bool)])
    labels = torch.cat([y, rest])
    rest_score = torch.full((out_rows,), rest_rate)
    return {"within_root": {"root": average_precision(torch.zeros(len(y)), y), "refined": average_precision(score, y)},
            "all_rows": {"root": average_precision(torch.cat([torch.full((len(y),), root_rate), rest_score]), labels),
                         "refined": average_precision(torch.cat([score, rest_score]), labels)}}


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--head", type=str, default="av:AV-Brake", help="head atom name")
    parser.add_argument("--root", type=str, default=None, help="root body (module doc); default per head")
    parser.add_argument("--mode", type=str, default="both", choices=("tree", "cover", "both"))
    parser.add_argument("--max_depth", type=int, default=3, help="tree: depth")
    parser.add_argument("--min_gain", type=float, default=1e-3, help="tree: entropy gain (nats per row) to split")
    parser.add_argument("--min_events", type=int, default=20, help="distinct onsets a literal's branch / rule hits")
    parser.add_argument("--target_prec", type=float, default=None,
                        help="cover: precision a sub-rule must reach (default 2 x the root's train precision)")
    parser.add_argument("--hit_slots", type=int, default=1,
                        help="a hit: the head switches on within the next hit_slots slots (1: as the clause counts)")
    parser.add_argument("--max_len", type=int, default=5, help="cover: literals per sub-rule")
    parser.add_argument("--max_rules", type=int, default=200, help="cover: sub-rules")
    parser.add_argument("--min_lift", type=float, default=1.2,
                        help="cover: keep a sub-rule if its precision >= min_lift x that of the rows not yet covered")
    parser.add_argument("--curve", type=str, default="0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8",
                        help="cover: precision levels of the coverage curve")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id if getattr(args, "run_id", None) \
        else args.output_dir / "rule_tree"
    cfg = step05.get_cfg(config.get_step_05_input(args))
    cfg["hit_slots"] = args.hit_slots
    device = cfg["device"]
    training_data = step05.load_training_data(cfg)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    index = defaultdict(list)
    for i, n in enumerate(names):
        index[n].append(i)
    if args.head not in index:
        raise SystemExit(f"head {args.head} not in the vocabulary")
    head = index[args.head][0]
    root_text = args.root or DEFAULT_ROOTS.get(args.head)
    if not root_text:
        raise SystemExit(f"no default root for {args.head}: pass --root")

    num_scopes = 2 + len(cfg["roles"])
    model = TemporalRuleNet(training_data["num_atoms"], cfg["slot_size"], cfg["num_slots"], 1, cfg["init_log_alpha"],
                            cfg["mask_type"], cfg["windows"], roles=cfg["roles"],
                            input_mask=step05.rule_input_mask(training_data, cfg, num_scopes),
                            ranges=cfg.get("slot_ranges", False)).to(device)
    scope_names = ["own", "other", *cfg["roles"]]
    root = parse_root(root_text, index, {s: i for i, s in enumerate(scope_names)})
    groups = vehicle_groups(model, names, scope_names)
    possible = step05.class_possible(training_data, vocab, device)
    M, G = model.num_inputs, len(groups)

    def feature_name(f):
        if f >= 2 * M:
            g = f - 2 * M
            return f"{'!' if g >= G else ''}{groups[g % G][0]}"
        neg, m = f >= M, f % M
        atom, scope, j = model.decode_input(m)
        return f"{'!' if neg else ''}{names[atom]}({scope_names[scope]})@{j}"

    path_text = lambda path: " & ".join(feature_name(f) if v else f"NOT {feature_name(f)}" for f, v in path)
    mask = model.input_mask.cpu()
    allowed = torch.cat([mask, mask, torch.ones(2 * G, dtype=torch.bool)]).to(device)

    train = collect(model, training_data, training_data["train_videos"], head, root, possible, groups, cfg,
                    "Rows (train)")
    val = collect(model, training_data, training_data["val_videos"], head, root, possible, groups, cfg, "Rows (val)")
    if len(train["y"]) == 0:
        raise SystemExit("the root never holds on the train rows")
    root_train, root_val = node_stats(train, torch.arange(len(train["y"]))), node_stats(val, torch.arange(len(val["y"])))
    root_rate = root_train["hits"] / root_train["rows"]
    rest_rate = (train["total_hits"] - root_train["hits"]) / max(train["total_rows"] - root_train["rows"], 1)
    target = args.target_prec if args.target_prec is not None else min(2 * root_rate, 0.95)
    prec = lambda h, n: h / n if n else None
    share = lambda st, r: prec(st["hits"], r["hits"])

    result = {"exp": args.exp, "head": args.head, "root": root_text, "roles": list(cfg["roles"]),
              "hit_slots": args.hit_slots,
              "min_events": args.min_events, "min_videos": cfg["min_videos"], "num_groups": G,
              "train": {"root": root_train, "total_rows": train["total_rows"], "total_hits": train["total_hits"],
                        "total_events": train["total_events"]},
              "val": {"root": root_val, "total_rows": val["total_rows"], "total_hits": val["total_hits"],
                      "total_events": val["total_events"]}}
    f = lambda v, p=3: "  -  " if v is None else f"{v:.{p}f}"
    print(f"\n[rule_tree] {args.exp} head {args.head}, roles {list(cfg['roles']) or 'none'}, {G} vehicle groups")
    print(f"  root: {root_text}")
    for split, r, d in (("train", root_train, train), ("val", root_val, val)):
        print(f"  {split}: root rows {r['rows']} hits {r['hits']} events {r['events']} precision "
              f"{f(prec(r['hits'], r['rows']))} recall {f(prec(r['hits'], d['total_hits']))} "
              f"(all at-risk: {d['total_rows']} rows, {d['total_hits']} hits, {d['total_events']} events)")

    def row(st, r):
        return {**st, "precision": prec(st["hits"], st["rows"]), "root_share": share(st, r)}

    if args.mode in ("tree", "both"):
        tree = grow(train, torch.arange(len(train["y"])), 0, allowed, cfg, args, device)
        leaves = []
        score = torch.zeros(len(val["y"]))
        for path, node in tree_leaves(tree, []):
            vm = holds_path(val, torch.arange(len(val["y"])), path)
            vst = node_stats(val, torch.arange(len(val["y"]))[vm])
            score[vm] = node["hits"] / max(node["rows"], 1)
            leaves.append({"path": path_text(path), "literals": [[a, b] for a, b in path],
                           "train": row({k: node[k] for k in ("rows", "hits", "events", "videos")}, root_train),
                           "val": row(vst, root_val)})
        leaves.sort(key=lambda l: -(l["train"]["precision"] or 0))
        result["tree"] = {"leaves": leaves, "val_ap": ap_report(val, score, root_rate, rest_rate)}

        def show(node, depth, label):
            print(f"  {'  ' * depth}{label} rows {node['rows']} hits {node['hits']} events {node['events']} "
                  f"prec {node['hits'] / max(node['rows'], 1):.3f}")
            if node["split"]:
                name = feature_name(node["split"]["feature"])
                show(node["split"]["yes"], depth + 1, f"[{name}]")
                show(node["split"]["no"], depth + 1, f"[NOT {name}]")
        print(f"\n  == tree (train; min {args.min_events} events per literal branch) ==")
        show(tree, 0, "[root]")
        print(f"  {'leaf':80s} {'tr prec':>7s} {'tr share':>8s} | {'va rows':>7s} {'va prec':>7s} {'va share':>8s}")
        for l in leaves:
            t, v = l["train"], l["val"]
            print(f"  {l['path'] or '(root)':80s} {f(t['precision']):>7s} {f(t['root_share']):>8s} | "
                  f"{v['rows']:7d} {f(v['precision']):>7s} {f(v['root_share']):>8s}")
        ap = result["tree"]["val_ap"]
        print(f"  val AP within the root: {f(ap['within_root']['root'])} -> {f(ap['within_root']['refined'])}; "
              f"over all at-risk rows: {f(ap['all_rows']['root'])} -> {f(ap['all_rows']['refined'])}")

    if args.mode in ("cover", "both"):
        rules, residual = cover(train, allowed, cfg, args, target, device)
        out, score = [], torch.zeros(len(val["y"]))
        left = torch.arange(len(val["y"]))
        explained_train = explained_val = 0
        for lits, st in rules:
            path = [(l, True) for l in lits]
            m = holds_path(val, left, path)
            vst = node_stats(val, left[m])
            alone = node_stats(val, torch.arange(len(val["y"]))[holds_path(val, torch.arange(len(val["y"])), path)])
            score[left[m]] = st["hits"] / max(st["rows"], 1)
            left = left[~m]
            explained_train += st["hits"]
            explained_val += vst["hits"]
            out.append({"rule": path_text(path), "literals": lits, "train": row(st, root_train),
                        "val": row(vst, root_val), "val_alone": row(alone, root_val),
                        "cum_train_share": explained_train / root_train["hits"],
                        "cum_val_share": explained_val / max(root_val["hits"], 1)})
        rest_val = node_stats(val, left)
        score[left] = residual["hits"] / max(residual["rows"], 1)
        result["cover"] = {"target_prec": target, "rules": out, "residual": {"train": row(residual, root_train),
                                                                             "val": row(rest_val, root_val)},
                           "val_ap": ap_report(val, score, root_rate, rest_rate)}
        print(f"\n  == cover (rule list on the rows not yet covered; target precision {target:.3f}, "
              f"min {args.min_events} events, <= {args.max_len} literals) ==")
        print(f"  {'#':>2s} {'sub-rule (root & ...)':70s} {'tr prec':>7s} {'tr ev':>5s} | {'va prec':>7s} "
              f"{'va share':>8s} {'cum va':>6s} | {'alone va prec':>13s}")
        for i, r in enumerate(out):
            print(f"  {i + 1:2d} {r['rule']:70s} {f(r['train']['precision']):>7s} {r['train']['events']:5d} | "
                  f"{f(r['val']['precision']):>7s} {f(r['val']['root_share']):>8s} {f(r['cum_val_share']):>6s} | "
                  f"{f(r['val_alone']['precision']):>13s}")
        rt, rv = result["cover"]["residual"]["train"], result["cover"]["residual"]["val"]
        print(f"     {'(residual: no sub-rule)':70s} {f(rt['precision']):>7s} {rt['events']:5d} | "
              f"{f(rv['precision']):>7s} {f(rv['root_share']):>8s}")
        # Coverage curve: the sub-rules with train precision >= p (selected on train), their share of the
        # root hits and their precision; val counts on the rows each covers in list order (a partition).
        curve = []
        for p in [float(x) for x in args.curve.split(",") if x.strip()]:
            chosen = [r for r in out if (r["train"]["precision"] or 0) >= p]
            th, tr = sum(r["train"]["hits"] for r in chosen), sum(r["train"]["rows"] for r in chosen)
            vh, vr = sum(r["val"]["hits"] for r in chosen), sum(r["val"]["rows"] for r in chosen)
            curve.append({"train_precision_at_least": p, "rules": len(chosen),
                          "train_share": th / root_train["hits"], "train_precision": prec(th, tr),
                          "val_share": vh / max(root_val["hits"], 1), "val_precision": prec(vh, vr),
                          "val_recall_all_onsets": vh / max(val["total_hits"], 1)})
        result["cover"]["curve"] = curve
        print(f"\n  coverage curve (sub-rules with train precision >= p):")
        print(f"  {'p':>5s} {'rules':>5s} | {'tr share':>8s} {'tr prec':>7s} | {'va share':>8s} {'va prec':>7s} "
              f"{'va recall (all onsets)':>22s}")
        for c in curve:
            print(f"  {c['train_precision_at_least']:5.2f} {c['rules']:5d} | {f(c['train_share']):>8s} "
                  f"{f(c['train_precision']):>7s} | {f(c['val_share']):>8s} {f(c['val_precision']):>7s} "
                  f"{f(c['val_recall_all_onsets']):>22s}")
        ap = result["cover"]["val_ap"]
        print(f"  val AP within the root: {f(ap['within_root']['root'])} -> {f(ap['within_root']['refined'])}; "
              f"over all at-risk rows: {f(ap['all_rows']['root'])} -> {f(ap['all_rows']['refined'])}")

    out_dir = args.output_dir / "rule_tree"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", root_text).strip("_")[:80]
    out_file = out_dir / (f"{args.head.replace(':', '_')}__{tag}{'_r' + '-'.join(cfg['roles']) if cfg['roles'] else ''}"
                          f"{f'_w{args.hit_slots}' if args.hit_slots != 1 else ''}_{args.mode}.json")
    utils_data.save_json(result, out_file)
    print(f"  -> {out_file}")


if __name__ == "__main__":
    main()
