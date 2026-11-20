"""Refine one high-recall root rule into sub-rules with a shallow decision tree. Count only, no training.

Given a head and a root body R (e.g. "the ego moves & some vehicle in a VehLane brakes" for av:AV-Brake), a
greedy tree splits the rows where R holds by one literal at a time. A literal is (atom, scope, slot j)
true ("X@j") or false and known ("!X@j"), in the rule net's input language (`TemporalRuleNet` scopes own /
other / roles, `step05.rule_input_mask`). Each root-to-leaf path is a sub-rule R & path. The leaves
partition the root rows, so the union recall of the leaves equals the root's recall exactly, and the gain is
the spread of leaf precisions: the high-precision leaves are the useful sub-rules, the rest is the residual.

Rows and labels as `step05._rule_head_counts` at f = 0: the at-risk rows of the agents that can hold the head
(`step05.class_possible`), positive if the head switches on within the next slot. The tree is grown on the
train videos only. Each split keeps the branch whose literal holds above min_support hits in min_videos
videos (the extraction rule of `step05.extract_temporal_clauses`), chosen by entropy gain; the other branch
("not literal": the literal false or unknown) may be any size. The val videos are only routed and scored.

Report (train and val per leaf): rows, hits, precision, share of the root hits and of all head onsets; the
union recall of the leaves with train precision >= each --prec_levels value, on val; and the val AP of
scoring the rows by their leaf's train precision vs by the root alone (within the root rows, and over all
at-risk rows with the rows outside the root at their train rate).

    ./run_roadpp_docker.sh --gpu 0 --rule_tree -- --exp sumo_maps_v7 --machine dgx-jing [--roles ahead]
    options: --head av:AV-Brake --root "av:AV-Mov(own)@1 & veh:Brake(other)@1 & veh:VehLane(other)@1"
             --max_depth 3 --min_gain 0.001 --prec_levels 0.3,0.5,0.7
Root syntax: literals joined by "&", each "[!]Name1|Name2(scope)@j" (OR over the names; "!": all false and
known); "veh:X" stands for X of every motor vehicle class; scope own, other or a role of --roles.
Writes <output_dir>/rule_tree/<head>__<root tag>.json and prints the tree.
"""
import json
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


@torch.no_grad()
def collect(model, training_data, video_ids, head, root, possible, cfg, desc):
    """The at-risk rows of the head: X, K (R, M) uint8 truth / known of the rows where the root holds, y (R,)
    their labels, v (R,) their video index; and the totals (rows, hits) of all at-risk rows."""
    device, S = cfg["device"], model.slot_size
    index_of = [torch.tensor([model.input_index(a, s, j) for a in atoms], device=device)
                for atoms, s, j, _p in root]
    X, K, Y, V = [], [], [], []
    total_rows = total_hits = 0
    for vi, video_id in enumerate(tqdm(video_ids, desc=desc)):
        can = possible(video_id)[:, head]
        for frames in training_data["frames"][video_id].split(cfg["frames_per_video"]):
            rows = step05._video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            x, k = model.split_inputs(rows["inputs"])
            first, at_risk = rows["first"][:, head], rows["at_risk"][:, head]
            keep = at_risk & can[rows["rows"][:, 1]] & (first >= 0) & rows["known"][:, 0, head]
            y = keep & (first < S)
            total_rows += int(keep.sum())
            total_hits += int(y.sum())
            holds = keep.clone()
            for idx, (_atoms, _s, _j, positive) in zip(index_of, root):
                holds &= (x[:, idx].amax(dim=1) > 0) if positive else (((1 - x[:, idx]) * k[:, idx]).amin(dim=1) > 0)
            if holds.any():
                X.append(x[holds].to(torch.uint8).cpu())
                K.append(k[holds].to(torch.uint8).cpu())
                Y.append(y[holds].cpu())
                V.append(torch.full((int(holds.sum()),), vi, dtype=torch.long))
    M = model.num_inputs
    cat = lambda parts, empty: torch.cat(parts) if parts else empty
    return {"X": cat(X, torch.zeros((0, M), dtype=torch.uint8)), "K": cat(K, torch.zeros((0, M), dtype=torch.uint8)),
            "y": cat(Y, torch.zeros(0, dtype=torch.bool)), "v": cat(V, torch.zeros(0, dtype=torch.long)),
            "total_rows": total_rows, "total_hits": total_hits}


def features(data, idx, device):
    """(n, 2M) float literal values of the rows idx: X@j (true) then !X@j (false and known)."""
    x = data["X"][idx].to(device).float()
    k = data["K"][idx].to(device).float()
    return torch.cat([x, (1 - x) * k], dim=1)


def entropy(h, n):
    p = (h / n.clamp(min=1)).clamp(1e-12, 1 - 1e-12)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def grow(data, idx, depth, allowed, cfg, args, device):
    """A node {rows, hits, videos, split: {feature, yes, no} | None} of the train rows idx (long tensor)."""
    y = data["y"][idx]
    n, h = len(idx), int(y.sum())
    node = {"rows": n, "hits": h, "videos": int(data["v"][idx][y].unique().numel()), "idx": idx, "split": None}
    if depth >= args.max_depth or h < 2 * cfg["min_support"] or h == n:
        return node
    F = features(data, idx, device)                                         # (n, 2M)
    yf = y.to(device).float()
    n1, h1 = F.sum(dim=0), yf @ F
    n0, h0 = n - n1, h - h1
    nn, hh = torch.tensor(float(n), device=device), torch.tensor(float(h), device=device)
    gain = entropy(hh, nn) - (n1 / n) * entropy(h1, n1) - (n0 / n) * entropy(h0, n0)
    # videos with a hit in the literal's branch
    hit_videos = data["v"][idx][y]
    vids, inverse = hit_videos.unique(return_inverse=True)
    onehot = torch.zeros((len(hit_videos), len(vids)), device=device)
    onehot[torch.arange(len(hit_videos), device=device), inverse.to(device)] = 1.0
    videos1 = ((F[y.to(device)].T @ onehot) > 0).sum(dim=1)
    ok = allowed & (h1 >= cfg["min_support"]) & (videos1 >= cfg["min_videos"]) & (n0 > 0) & (n1 > 0)
    gain = torch.where(ok, gain, torch.full_like(gain, -1.0))
    best = int(gain.argmax())
    if gain[best] < args.min_gain:
        return node
    mask = F[:, best].bool().cpu()
    node["split"] = {"feature": best, "gain": float(gain[best]),
                     "yes": grow(data, idx[mask], depth + 1, allowed, cfg, args, device),
                     "no": grow(data, idx[~mask], depth + 1, allowed, cfg, args, device)}
    return node


def route(node, data, idx, device, out, path=()):
    """Leaves [(path, train node, val idx)] with the val rows idx routed down the tree."""
    if node["split"] is None:
        out.append((path, node, idx))
        return out
    f = node["split"]["feature"]
    mask = features(data, idx, device)[:, f].bool().cpu() if len(idx) else torch.zeros(0, dtype=torch.bool)
    route(node["split"]["yes"], data, idx[mask], device, out, path + ((f, True),))
    route(node["split"]["no"], data, idx[~mask], device, out, path + ((f, False),))
    return out


def average_precision(scores, labels):
    """AP with tied scores grouped (a tie block counts as one threshold)."""
    if labels.sum() == 0:
        return None
    order = torch.argsort(scores, descending=True)
    s, l = scores[order], labels[order].float()
    keys, counts = torch.unique_consecutive(s, return_counts=True)
    ends = counts.cumsum(0) - 1
    tp = l.cumsum(0)[ends]
    seen = ends + 1
    precision, recall = tp / seen, tp / l.sum()
    prev = torch.cat([torch.zeros(1), recall[:-1]])
    return float(((recall - prev) * precision).sum())


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--head", type=str, default="av:AV-Brake", help="head atom name")
    parser.add_argument("--root", type=str, default=None, help="root body (module doc); default per head")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--min_gain", type=float, default=1e-3, help="entropy gain (nats per row) to split")
    parser.add_argument("--prec_levels", type=str, default="0.3,0.5,0.7")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id if getattr(args, "run_id", None) \
        else args.output_dir / "rule_tree"
    cfg = step05.get_cfg(config.get_step_05_input(args))
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
    scopes = {s: i for i, s in enumerate(scope_names)}
    root = parse_root(root_text, index, scopes)
    possible = step05.class_possible(training_data, vocab, device)

    def feature_name(f):
        neg, m = f >= model.num_inputs, f % model.num_inputs
        atom, scope, j = model.decode_input(m)
        return f"{'!' if neg else ''}{names[atom]}({scope_names[scope]})@{j}"

    # The literals a split may use: the rule net's inputs (input_mask), both polarities.
    allowed = torch.cat([model.input_mask, model.input_mask]).to(device)

    train = collect(model, training_data, training_data["train_videos"], head, root, possible, cfg, "Rows (train)")
    val = collect(model, training_data, training_data["val_videos"], head, root, possible, cfg, "Rows (val)")
    if len(train["y"]) == 0:
        raise SystemExit("the root never holds on the train rows")

    tree = grow(train, torch.arange(len(train["y"])), 0, allowed, cfg, args, device)
    leaves = route(tree, val, torch.arange(len(val["y"])), device, [])

    root_train = {"rows": len(train["y"]), "hits": int(train["y"].sum())}
    root_val = {"rows": len(val["y"]), "hits": int(val["y"].sum())}
    prec = lambda h, n: h / n if n else None
    out_leaves = []
    for path, node, vidx in leaves:
        vy = val["y"][vidx]
        out_leaves.append({
            "path": [feature_name(f) if yes else f"NOT {feature_name(f)}" for f, yes in path],
            "literals": [[f, yes] for f, yes in path],
            "train": {"rows": node["rows"], "hits": node["hits"], "videos": node["videos"],
                      "precision": prec(node["hits"], node["rows"]),
                      "root_recall": prec(node["hits"], root_train["hits"]),
                      "recall": prec(node["hits"], train["total_hits"])},
            "val": {"rows": len(vidx), "hits": int(vy.sum()), "precision": prec(int(vy.sum()), len(vidx)),
                    "root_recall": prec(int(vy.sum()), root_val["hits"]),
                    "recall": prec(int(vy.sum()), val["total_hits"])},
        })
    out_leaves.sort(key=lambda l: -(l["train"]["precision"] or 0))

    # Union recall (val) of the leaves selected by their train precision.
    levels = [float(p) for p in args.prec_levels.split(",") if p.strip()]
    union = []
    for p in levels:
        chosen = [l for l in out_leaves if (l["train"]["precision"] or 0) >= p]
        rows = sum(l["val"]["rows"] for l in chosen)
        hits = sum(l["val"]["hits"] for l in chosen)
        union.append({"train_precision_at_least": p, "leaves": len(chosen), "val_rows": rows, "val_hits": hits,
                      "val_precision": prec(hits, rows), "val_root_recall": prec(hits, root_val["hits"]),
                      "val_recall": prec(hits, val["total_hits"])})

    # AP on val: rows scored by their leaf's train precision vs by the root alone.
    leaf_score = torch.zeros(len(val["y"]))
    for path, node, vidx in leaves:
        leaf_score[vidx] = node["hits"] / max(node["rows"], 1)
    y = val["y"]
    ap_within = {"root": average_precision(torch.zeros(len(y)), y), "tree": average_precision(leaf_score, y)}
    # All at-risk val rows: the rows outside the root get the train rate outside the root.
    out_rows, out_hits = val["total_rows"] - len(y), val["total_hits"] - int(y.sum())
    rest_rate = (train["total_hits"] - root_train["hits"]) / max(train["total_rows"] - root_train["rows"], 1)
    root_rate = root_train["hits"] / root_train["rows"]
    rest_labels = torch.cat([torch.ones(out_hits, dtype=torch.bool), torch.zeros(out_rows - out_hits, dtype=torch.bool)])
    all_labels = torch.cat([y, rest_labels])
    ap_all = {"root": average_precision(torch.cat([torch.full((len(y),), root_rate),
                                                   torch.full((out_rows,), rest_rate)]), all_labels),
              "tree": average_precision(torch.cat([leaf_score, torch.full((out_rows,), rest_rate)]), all_labels)}

    result = {"exp": args.exp, "head": args.head, "root": root_text, "roles": list(cfg["roles"]),
              "max_depth": args.max_depth, "min_gain": args.min_gain,
              "min_support": cfg["min_support"], "min_videos": cfg["min_videos"],
              "train": {"root": root_train, "total_rows": train["total_rows"], "total_hits": train["total_hits"]},
              "val": {"root": root_val, "total_rows": val["total_rows"], "total_hits": val["total_hits"]},
              "leaves": out_leaves, "union": union, "val_ap_within_root": ap_within, "val_ap_all_rows": ap_all}
    out_dir = args.output_dir / "rule_tree"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", root_text).strip("_")[:80]
    out_file = out_dir / f"{args.head.replace(':', '_')}__{tag}{'_r' + '-'.join(cfg['roles']) if cfg['roles'] else ''}.json"
    utils_data.save_json(result, out_file)

    f = lambda v, p=3: "  -  " if v is None else f"{v:.{p}f}"
    print(f"\n[rule_tree] {args.exp} head {args.head}, roles {list(cfg['roles']) or 'none'}")
    print(f"  root: {root_text}")
    for split, r, d in (("train", root_train, train), ("val", root_val, val)):
        print(f"  {split}: root rows {r['rows']} hits {r['hits']} precision {f(prec(r['hits'], r['rows']))} "
              f"recall {f(prec(r['hits'], d['total_hits']))} (all at-risk: {d['total_rows']} rows, "
              f"{d['total_hits']} hits, rate {f(prec(d['total_hits'], d['total_rows']))})")

    def show(node, depth, label):
        p = node["hits"] / max(node["rows"], 1)
        print(f"  {'  ' * depth}{label} rows {node['rows']} hits {node['hits']} prec {p:.3f}")
        if node["split"]:
            name = feature_name(node["split"]["feature"])
            show(node["split"]["yes"], depth + 1, f"[{name}]")
            show(node["split"]["no"], depth + 1, f"[NOT {name}]")
    print("  tree (train):")
    show(tree, 0, "[root]")
    print(f"\n  {'leaf (sorted by train precision)':70s} {'tr prec':>7s} {'tr rec':>6s} | {'va rows':>7s} "
          f"{'va prec':>7s} {'va rootrec':>10s} {'va rec':>6s}")
    for l in out_leaves:
        t, v = l["train"], l["val"]
        print(f"  {' & '.join(l['path']) or '(root)':70s} {f(t['precision']):>7s} {f(t['recall']):>6s} | "
              f"{v['rows']:7d} {f(v['precision']):>7s} {f(v['root_recall']):>10s} {f(v['recall']):>6s}")
    print("\n  union of the leaves with train precision >= p (val):")
    for u in union:
        print(f"    p >= {u['train_precision_at_least']:.2f}: {u['leaves']} leaves, precision "
              f"{f(u['val_precision'])}, share of root hits {f(u['val_root_recall'])}, recall {f(u['val_recall'])}")
    print(f"  val AP within the root rows: root {f(ap_within['root'])} -> tree {f(ap_within['tree'])}")
    print(f"  val AP over all at-risk rows: root {f(ap_all['root'])} -> tree {f(ap_all['tree'])}")
    print(f"  -> {out_file}")


if __name__ == "__main__":
    main()
