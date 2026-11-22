"""Boosted rule heads in the main pipeline: the `brake_factors` design (2026-11-21) as a step 06 scorer.

brake_factors (AV-Brake, SUMO v7, settings chosen on val): boosted rules on image features 0.667 test AP vs
SetNet 0.654 and the rule list + logit 0.609 (`docs/pipeline/BOOSTED_RULE_HEADS.md`). Here the same rules
replace one ego head of the main rule net, scored on the step 06 rows and events:

  scorer "boost" = the TemporalRuleNet ("model") with the column of each boosted head replaced, on the ego
  agent's rows, by the boosted rules' probability. Every other head and agent keeps the rule net's score, so
  the per-head metrics of a boosted head (AP, event detection, head rank) compare it with the rule net and
  the OnsetMLP on the same rows, events and thresholds (chosen on val as for every scorer).

Fit (once per data, cached in <output_dir>/boost):
  rows: `brake_factors.collect(port=True)`: every 2nd frame t where the head was off in the min_off_frames
      before t (step 05 `at_risk`; the ego need not move), with an observed outcome; y = onset in
      t .. t + WINDOW - 1 (WINDOW 20 = the headline window). Image features and labels only (gt=False: no
      SUMO world, as on real video).
  literals: `brake_factors.search_literals` (sources "img") + `make_banks`, chosen on train.
  rules: `brake_factors.boosted_rules`, rounds stopped by the val AP.
Score: every frame (stride 1) of the val and test videos, cached per split.

One probability per row for every window (the rules are fit to the 20-frame window): read the boosted heads
at the headline window (20); the other windows only rank the same score.

Agent heads ("Car:Brake", since 2026-11-22, AGENT_VERSION): rows are (t, agent n) of the head's class, at
risk as in step 05 (the head off on n in the min_off_frames before t, n seen in t - 1). The groups of a row:
  own (n itself: image features and labels), near_ped (n's nearest pedestrian, role near_ped), ahead (the
  ego's lead), veh / ped / tl (some other agent of the group), ego (scene scale and the ego's labels);
  every feature and label also 1, 2, 3, 5 s before t - 1 (`brake_factors.LAGS_S`).
Train / val rows: every 2nd frame; all positive rows up to AGENT_MAX_POS, negatives sampled to
AGENT_NEG_PER_POS x the positives (at least AGENT_MIN_NEG), each row weighted by 1 / its sampling rate (the
boosting, its base score and the val AP use the weights: the probabilities stay on the true onset rate).
Literals: the bank (labels, intervals) and the 1-feature regions of every group, pairs and 2-feature regions
on own (no triples: the boosting's depth-3 rules combine own literals on the same agent).
Heads with fewer than AGENT_MIN_POS positive train rows are skipped (the rule net's score stays).

Fair neural reference, scorer "boost_net" (2026-11-22): per boosted head a SetNet (`brake_factors.SetNet`:
a per-agent MLP for each group, max-pooled, then a row MLP with the ego features) on exactly the same rows,
sample weights, features and labels (every lag) as the boosted rules, early-stopped on the same weighted
val AP, and plugged into the rule net the same way. "boost" vs "boost_net" isolates rules vs network on equal
inputs; both vs "mlp" shows what the new features add over the step 05 inputs.
"""
import math
import os
from pathlib import Path

import torch
from tqdm import tqdm

from src.exp_roadpp import brake_factors as bf
from src.exp_roadpp import ego_motion, roles as agent_roles, step05, utils_data

VERSION = "boost_v1"              # ego heads (the brake_factors search, ego rows)
AGENT_VERSION = "boost_a1"        # agent heads
NET_VERSION = "net_v1"            # the SetNet on the ego heads' rows
AGENT_NET_VERSION = "net_a1"      # ... on the agent heads' rows
METHODS = ("rules", "net")
SCORER_OF = {"rules": "boost", "net": "boost_net"}
NET_MAX_STEPS = 2000              # full-batch Adam steps
NET_EVAL = 25                     # steps between val AP checks
NET_PATIENCE = 8                  # checks without a better val AP
NET_LR = 1e-3
TRAIN_STRIDE = 2
AGENT_GROUPS = ("ego", "own", "near_ped", "ahead", "veh", "ped", "tl")
AGENT_MAX_POS = 30_000
AGENT_NEG_PER_POS = 4
AGENT_MIN_NEG = 20_000
AGENT_MIN_POS = 50
SCORE_VIDEO_CHUNK = 32


def _tag(head):
    return head.replace(":", "_")


def _rows(step05_cfg, split, atom_file, box_file, head, stride):
    cfg = {**step05_cfg, f"{split}_atom_file": atom_file, f"{split}_box_file": box_file}
    return bf.collect(cfg, split, Path("."), head, port=True, stride=stride, use_gt=False)


def _version(head, method):
    if head.startswith("av:"):
        return VERSION if method == "rules" else NET_VERSION
    return AGENT_VERSION if method == "rules" else AGENT_NET_VERSION


def fit_rules(train, val, device, head, version, use_gt_filter=False):
    """The boosted rules of one head on its rows (`brake_factors.search_literals` + `boosted_rules`)."""
    agent = "groups" in train
    singles, pairs, triples, regions = bf.search_literals(
        train, val, device, sources=("img",), **({"same_agent": ("own",), "triples": False} if agent else {}))
    _bank, big = bf.make_banks(train, val, val, singles, pairs, triples, regions)
    use = [i for i, d in enumerate(big[0]) if "gt:" not in bf.literal_text(d)] if use_gt_filter else None
    res = bf.boosted_rules(train, val, val, big, device, use=use)
    descs = res["_descs"]
    n = len(descs)
    rules = [{"weight": r["weight"], "literals": [(descs[k % n]["spec"], k >= n) for k in r["literals"]],
              "text": r["text"], "rows": r["rows"], "precision": r["precision"],
              "val_precision": r["val_precision"], "rounds": r["rounds"]} for r in res["rules"]]
    model = {"version": version, "method": "rules", "head": head, "window": bf.WINDOW, "f0": res["f0"],
             "rules": rules, "best": res["best"], "curve": res["curve"], "literals": res["literals"]}
    b = res["best"]
    print(f"[Boost] {head}: {len(rules)} rules over {res['literals']} literals, {b['rounds']} rounds; "
          f"train AP {b['train_ap']:.3f}, val AP {b['val_ap']:.3f}")
    for r in sorted(rules, key=lambda r: -abs(r["weight"]) * r["rows"])[:10]:
        print(f"    w {r['weight']:+.2f} rows {r['rows']:6d} p_tr {r['precision']:.2f} p_val {r['val_precision']:.2f}: "
              f"{r['text'][:170]}")
    return model


def _net_spec(data):
    f = data["features"]
    groups = [g for g in bf.groups_of(data) if g != "ego"]
    feats_agent = [x for x in f["agent_cont"] + f["agent_label"] if not x.startswith("gt:")]
    feats_ego = [x for x in f["ego_cont"] + f["ego_label"] if not x.startswith("gt:")]
    return groups, feats_agent, feats_ego


def fit_net(train, val, device, head, version):
    """The SetNet of one head on the same rows, weights and features as its boosted rules: full-batch Adam on
    the weighted BCE, the step of the best weighted val AP kept (checked every NET_EVAL steps)."""
    groups, feats_agent, feats_ego = _net_spec(train)
    stats = bf._stats_of(train, feats_agent, feats_ego, groups)
    x_tr = bf._net_inputs(train, feats_agent, feats_ego, stats, device, groups)
    x_va = bf._net_inputs(val, feats_agent, feats_ego, stats, device, groups)
    y_tr = train["y"].float().to(device)
    w_tr = train["weight"].float().to(device) if "weight" in train else torch.ones_like(y_tr)
    w_va = val.get("weight")
    torch.manual_seed(0)
    net = bf.SetNet(len(feats_agent), len(feats_ego), groups=groups).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=NET_LR)
    best, curve, fails = {"step": 0, "val_ap": -1.0}, [], 0
    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    for step in range(NET_MAX_STEPS):
        net.train()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x_tr), y_tr, weight=w_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % NET_EVAL == 0:
            net.eval()
            with torch.no_grad():
                point = {"step": step + 1, "loss": float(loss),
                         "train_ap": bf.average_precision(net(x_tr).cpu(), train["y"], train.get("weight")),
                         "val_ap": bf.average_precision(net(x_va).cpu(), val["y"], w_va)}
            curve.append(point)
            if point["val_ap"] is not None and point["val_ap"] > best["val_ap"]:
                best, fails = dict(point), 0
                state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                fails += 1
                if fails >= NET_PATIENCE:
                    break
    model = {"version": version, "method": "net", "head": head, "window": bf.WINDOW, "groups": groups,
             "feats_agent": feats_agent, "feats_ego": feats_ego, "stats": stats,
             "state": {k: v.cpu() for k, v in state.items()}, "best": best, "curve": curve}
    print(f"[Boost net] {head}: step {best['step']}; train AP {best.get('train_ap') or 0:.3f}, "
          f"val AP {best['val_ap']:.3f}")
    return model


def _save(model, out_file, train, val):
    model.update({"train_rows": len(train["y"]), "val_rows": len(val["y"])})
    torch.save(model, out_file)
    return model


def fit(step05_cfg, head, out_files, device):
    """Fit the stale methods ({method: file}) of one ego head on the step 05 train videos (val: the rounds /
    steps) and save them."""
    rows = {}
    for split in ("train", "val"):
        data = _rows(step05_cfg, split, step05_cfg[f"{split}_atom_file"], step05_cfg[f"{split}_box_file"], head,
                     TRAIN_STRIDE)
        rows[split] = bf.subset_rows(data, data["known"])
    train, val = rows["train"], rows["val"]
    print(f"[Boost] {head}: rows train {len(train['y'])}, val {len(val['y'])}, base {float(train['y'].float().mean()):.3f}")
    for method, out_file in out_files.items():
        if method == "rules":
            _save(fit_rules(train, val, device, head, VERSION, use_gt_filter=True), out_file, train, val)
        else:
            _save(fit_net(train, val, device, head, NET_VERSION), out_file, train, val)


def score(model, data, device="cpu"):
    """(R,) probability of a model (boosted rules or SetNet) on the rows of data (`collect` / `build_rows`)."""
    if model.get("method") == "net":
        net = bf.SetNet(len(model["feats_agent"]), len(model["feats_ego"]), groups=model["groups"]).to(device)
        net.load_state_dict(model["state"])
        net.eval()
        x = bf._net_inputs(data, model["feats_agent"], model["feats_ego"], model["stats"], device, model["groups"])
        with torch.no_grad():
            return torch.sigmoid(net(x)).cpu()
    F = torch.full((len(data["y"]),), float(model["f0"]))
    cache = {}
    for r in model["rules"]:
        f = torch.ones(len(data["y"]), dtype=torch.bool)
        for spec, neg in r["literals"]:
            key = id(spec)
            if key not in cache:
                cache[key] = bf.apply_spec(data, spec)
            f &= ~cache[key] if neg else cache[key]
        F += r["weight"] * f.float()
    return torch.sigmoid(F)


def score_split(models, step05_cfg, split, atom_file, box_file, device="cpu"):
    """Per model: {video: (T,) probability per frame t (NaN: no row: the head on, or t = 0)}, ego index and
    head track. models: the models of one ego head (their rows are the same)."""
    data = _rows(step05_cfg, split, atom_file, box_file, models[0]["head"], 1)
    out = []
    for model in models:
        p = score(model, data, device)
        scores = {}
        for vi, vid in enumerate(data["video_ids"]):
            sel = data["video"] == vi
            if vid not in data["head_track"]:
                continue
            arr = torch.full((len(data["head_track"][vid]),), float("nan"))
            arr[data["t"][sel]] = p[sel]
            scores[vid] = arr
        out.append({"scores": scores, "ego": data["ego_index"], "track": data["head_track"]})
    return out


# -------------- Agent heads --------------
def _class_of(X, names, classes):
    """[N] class of each agent: its box class, else the class prefix most of its atoms carry ("av": the ego)."""
    out = list(classes)
    for n in range(X.shape[1]):
        if out[n] is None:
            held = X[:, n].any(dim=0).nonzero().flatten().tolist()
            prefixes = [names[a].split(":", 1)[0] for a in held if ":" in names[a]]
            out[n] = max(set(prefixes), key=prefixes.count) if prefixes else None
    return out


def video_tables(step05_cfg, atom_file, box_file, names):
    """Per video of a split, the inputs of the agent rows (everything at frame level, the lags taken later):
    X (T, N, A) bool, img (T, N, F) image features, lab (T, N, P) class-free labels, has_box / present
    (T, N), cls [N], near (T, N) n's nearest pedestrian, lead (T,) the ego's lead, ego, scene (T,),
    ego_lab (T, n_av). Returns (tables, parts, av_atoms)."""
    unique_atoms = utils_data.load_json(step05_cfg["unique_atoms_file"])
    A = len(unique_atoms)
    parts = sorted({n.split(":", 1)[1] for n in names if ":" in n and not n.startswith("av:")})
    onehot = torch.zeros((A, len(parts)))
    for a, n in enumerate(names):
        if ":" in n and not n.startswith("av:"):
            onehot[a, parts.index(n.split(":", 1)[1])] = 1.0
    av_atoms = [a for a, n in enumerate(names) if n.startswith("av:")]
    atoms = utils_data.load_json(atom_file)
    boxes = torch.load(box_file, map_location="cpu")
    tables = []
    for vid, video_atoms in tqdm(sorted(atoms.items()), desc="Boost tables"):
        X = step05.build_video_tensor(video_atoms, A).cpu()
        T, N = X.shape[:2]
        if T < 2:
            continue
        ego = step05.video_agent_ids(video_atoms).get(None)
        vb, classes, has_box = step05._agent_boxes(boxes, vid, video_atoms, X)
        partners = agent_roles.role_partners(vb.clone(), classes, ego, ("ahead", "near_ped"),
                                             agent_roles.ego_lane(X, unique_atoms))
        if ego is not None:
            scene = ego_motion.signals(vb, has_box, classes, ego, X, unique_atoms)["scene_scale"].float()
            lead, ego_lab = partners[:, ego, 0], X[:, ego][:, av_atoms]
        else:
            scene = torch.full((T,), float("nan"))
            lead, ego_lab = torch.full((T,), -1, dtype=torch.long), torch.zeros((T, len(av_atoms)), dtype=torch.bool)
        img = bf.image_features(vb, has_box)
        tables.append({"vid": vid, "X": X, "img": torch.stack([img[f] for f in bf.IMG_AGENT], dim=2),
                       "lab": (X.float() @ onehot) > 0, "has_box": has_box, "present": X.any(dim=2) | has_box,
                       "cls": _class_of(X, names, classes), "near": partners[:, :, 1], "lead": lead, "ego": ego,
                       "scene": scene, "ego_lab": ego_lab})
    return tables, parts, av_atoms


def _held(col, m_off):
    """(T, k) bool: col (T, k) was on in some frame of t - m_off .. t - 1."""
    cs = torch.cat([torch.zeros_like(col[:1], dtype=torch.long), col.long().cumsum(0)])
    t = torch.arange(col.shape[0])
    return (cs[t] - cs[(t - m_off).clamp(min=0)]) > 0


def candidates(tables, a, cls, stride, m_off, window=bf.WINDOW):
    """The at-risk rows of head a (class cls) per video: [(video index, t, n, y, event, known)] tensors."""
    out = []
    for vi, tb in enumerate(tables):
        agents = [n for n, c in enumerate(tb["cls"]) if c == cls and n != tb["ego"]]
        if not agents:
            continue
        idx = torch.tensor(agents)
        col = tb["X"][:, idx, a]                                                   # (T, k)
        T = col.shape[0]
        onset = step05.atom_onsets(col[:, :, None], m_off)[:, :, 0]
        ok = ~_held(col, m_off) & torch.cat([torch.zeros_like(col[:1]), tb["present"][:-1, idx]])   # seen in t - 1
        t_rows = torch.arange(1, T, stride)
        ok = ok[t_rows]
        r, k = ok.nonzero(as_tuple=True)
        t = t_rows[r]
        # the first onset at or after t (T: none)
        frame = torch.where(onset, torch.arange(T)[:, None], T)
        nxt = frame.flip(0).cummin(dim=0).values.flip(0)                         # (T, k)
        first = nxt[t, k]
        y = first < (t + window).clamp(max=T)
        event = torch.where(y, (vi * 1000 + idx[k]) * 100000 + first, torch.full_like(first, -1))
        known = (t + window <= T) | y
        out.append((torch.full_like(t, vi), t, idx[k], y, event, known))
    if not out:
        return None
    return [torch.cat(c) for c in zip(*out)]


def _sample(cand, seed, max_pos, neg_per_pos, min_neg):
    """Known rows of cand: positives (up to max_pos) and sampled negatives, with weights 1 / rate."""
    v, t, n, y, ev, known = cand
    keep = known
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(len(y), generator=g)
    n_pos, n_neg = int((y & keep).sum()), int((~y & keep).sum())
    q_pos = min(1.0, max_pos / max(n_pos, 1))
    q_neg = min(1.0, max(neg_per_pos * q_pos * n_pos, min_neg) / max(n_neg, 1))
    q = torch.where(y, torch.tensor(q_pos), torch.tensor(q_neg))
    keep = keep & (u < q)
    return [c[keep] for c in (v, t, n, y, ev)] + [1.0 / q[keep]], n_pos, n_neg


def _lagged(Fr, frames, agents=None):
    """(r, F * len(LAGS_S)) values of Fr ((T, F) or (T, N, F)) at frames - lag, NaN before frame 0."""
    outs = []
    for sec in bf.LAGS_S:
        f = frames - sec * bf.FPS
        valid = f >= 0
        fc = f.clamp(min=0)
        x = (Fr[fc] if agents is None else Fr[fc, agents]).float()
        outs.append(torch.where(valid[:, None], x, torch.full_like(x, float("nan"))))
    return torch.cat(outs, dim=1)


def build_rows(tables, parts, av_atoms, names, rows):
    """`brake_factors.collect`-style rows (groups AGENT_GROUPS) of the given (video index, t, n, y, event[,
    weight]) tensors: each row's features at the last seen frame t - 1 and LAGS_S before it."""
    v, t, n, y, ev = rows[:5]
    feats = {"agent_cont": [bf.lag_name(f, s) for s in bf.LAGS_S for f in bf.IMG_AGENT],
             "agent_label": [bf.lag_name(f"lab:{p}", s) for s in bf.LAGS_S for p in parts],
             "ego_cont": [bf.lag_name("img:scene_scale", s) for s in bf.LAGS_S],
             "ego_label": [bf.lag_name(f"ego:{names[a].split(':', 1)[1]}", s) for s in bf.LAGS_S for a in av_atoms]}
    ego_parts, chunks = [], {g: {"row": [], "cont": [], "label": []} for g in AGENT_GROUPS[1:]}
    for vi in v.unique().tolist():
        sel = (v == vi).nonzero().flatten()
        tb = tables[vi]
        o, nn = t[sel] - 1, n[sel]
        ego_parts.append((sel, torch.cat([_lagged(tb["scene"][:, None], o), _lagged(tb["ego_lab"], o)], dim=1)))

        def add(g, rows_, frames, agents):
            if len(rows_):
                chunks[g]["row"].append(rows_)
                chunks[g]["cont"].append(_lagged(tb["img"], frames, agents))
                chunks[g]["label"].append(_lagged(tb["lab"], frames, agents).nan_to_num(0.0) > 0.5)
        add("own", sel, o, nn)
        m = tb["near"][o, nn]
        add("near_ped", sel[m >= 0], o[m >= 0], m[m >= 0])
        m = tb["lead"][o]
        ok = (m >= 0) & (m != nn)
        add("ahead", sel[ok], o[ok], m[ok])
        N = tb["X"].shape[1]
        member = {"veh": [c in bf.VEH_CLASSES for c in tb["cls"]], "ped": [c == "Ped" for c in tb["cls"]],
                  "tl": [c in bf.TL_GROUP_CLASSES for c in tb["cls"]]}
        for g, mm in member.items():
            M = tb["has_box"][o] & torch.tensor(mm)[None] & (torch.arange(N)[None] != nn[:, None])   # (r, N) others
            r, a = M.nonzero(as_tuple=True)
            add(g, sel[r], o[r], a)
    R = len(y)
    out = {"y": y, "event": ev, "video": v, "t": t, "agent": n, "known": torch.ones(R, dtype=torch.bool),
           "features": feats, "groups": AGENT_GROUPS}
    if len(rows) > 5:
        out["weight"] = rows[5].float()
    E = torch.full((R, len(feats["ego_cont"]) + len(feats["ego_label"])), float("nan"))
    for sel, e in ego_parts:
        E[sel] = e
    out["ego"] = {k: E[:, i].contiguous() for i, k in enumerate(feats["ego_cont"] + feats["ego_label"])}
    for g, d in chunks.items():
        rows_ = torch.cat(d["row"]) if d["row"] else torch.zeros(0, dtype=torch.long)
        C = torch.cat(d["cont"]) if d["cont"] else torch.zeros((0, len(feats["agent_cont"])))
        L = torch.cat(d["label"]) if d["label"] else torch.zeros((0, len(feats["agent_label"])), dtype=torch.bool)
        out[g] = {"row": rows_}
        out[g].update({k: C[:, i].contiguous() for i, k in enumerate(feats["agent_cont"])})
        out[g].update({k: L[:, i].contiguous() for i, k in enumerate(feats["agent_label"])})
    return out


def fit_agent(head, a, tables, parts, av_atoms, names, m_off, out_files, device):
    """Fit the stale methods ({method: file}) of one agent head on sampled train rows (val rows: the rounds /
    steps) and save them."""
    cls = head.split(":", 1)[0]
    data, info = {}, {}
    for split in ("train", "val"):
        cand = candidates(tables[split], a, cls, TRAIN_STRIDE, m_off)
        if cand is None:
            info[split] = (0, 0)
            continue
        rows, n_pos, n_neg = _sample(cand, 0 if split == "train" else 1, AGENT_MAX_POS, AGENT_NEG_PER_POS, AGENT_MIN_NEG)
        info[split] = (n_pos, n_neg)
        data[split] = rows
    if info["train"][0] < AGENT_MIN_POS or "val" not in data or int(data["val"][3].sum()) == 0:
        for method, out_file in out_files.items():
            torch.save({"version": _version(head, method), "method": method, "head": head, "skip": True,
                        "train_pos": info["train"][0]}, out_file)
        print(f"[Boost] {head}: skipped ({info['train'][0]} positive train rows < {AGENT_MIN_POS}, or none on val)")
        return
    train = build_rows(tables["train"], parts, av_atoms, names, data["train"])
    val = build_rows(tables["val"], parts, av_atoms, names, data["val"])
    print(f"[Boost] {head}: sampled rows train {len(train['y'])} of {sum(info['train'])} ({info['train'][0]} "
          f"positive), val {len(val['y'])}")
    extra = {"kind": "agent", "class": cls, "train_pos": info["train"][0], "train_neg": info["train"][1],
             "train_base": info["train"][0] / max(sum(info["train"]), 1)}
    for method, out_file in out_files.items():
        fitter = fit_rules if method == "rules" else fit_net
        model = fitter(train, val, device, head, _version(head, method))
        model.update(extra)
        _save(model, out_file, train, val)


def score_agents(models, tables, parts, av_atoms, names, m_off, device="cpu"):
    """{(head, method): {video: (agents (k,), (T, k) probability, NaN: not at risk)}} of the agent heads'
    models, every frame of every video (in chunks of videos, one row build per class)."""
    out = {(m["head"], m["method"]): {} for m in models}
    by_class = {}
    for m in models:
        by_class.setdefault(m["class"], []).append(m)
    for c0 in tqdm(range(0, len(tables), SCORE_VIDEO_CHUNK), desc="Boost scores"):
        chunk = list(range(c0, min(c0 + SCORE_VIDEO_CHUNK, len(tables))))
        for cls, ms in by_class.items():
            rows = []
            for vi in chunk:
                tb = tables[vi]
                agents = [n for n, c in enumerate(tb["cls"]) if c == cls and n != tb["ego"]]
                if not agents:
                    continue
                T = tb["X"].shape[0]
                seen = tb["present"][:-1, agents]                                     # (T - 1, k): seen in t - 1
                r, k = seen.nonzero(as_tuple=True)
                rows.append((torch.full_like(r, vi), r + 1, torch.tensor(agents)[k]))
            if not rows:
                continue
            v, t, n = [torch.cat(x) for x in zip(*rows)]
            zero = torch.zeros(len(v), dtype=torch.bool)
            data = build_rows(tables, parts, av_atoms, names, (v, t, n, zero, torch.full_like(v, -1)))
            v, t, n = data["video"], data["t"], data["agent"]
            for m in ms:
                a = names.index(m["head"])
                p = score(m, data, device)
                for vi in v.unique().tolist():
                    tb = tables[vi]
                    sel = v == vi
                    agents = sorted(set(n[sel].tolist()))
                    col = {x: i for i, x in enumerate(agents)}
                    arr = torch.full((tb["X"].shape[0], len(agents)), float("nan"))
                    arr[t[sel], torch.tensor([col[x] for x in n[sel].tolist()])] = p[sel]
                    idx = torch.tensor(agents)
                    track = tb["X"][:, idx, a]
                    arr[_held(track, m_off)] = float("nan")                           # not at risk
                    out[(m["head"], m["method"])][tb["vid"]] = (idx, arr)
    return out


class BoostScorer:
    """step06 scorer (`step06._scorers`, video_scorer): the rule net's probabilities with the boosted heads'
    column replaced on the rows of the current video (context: video_id, rows, X): ego heads on the ego's
    rows, agent heads on the rows of their class's agents."""
    video_scorer = True

    def __init__(self, heads, device):
        self.heads = heads          # [(head index, kind "ego" | "agent", scores of val + test)]
        self.device = device
        self.checked, self.missing = set(), set()
        self.replaced = {}

    def __call__(self, context, stage1):
        out = stage1.clone()
        vid = context["video_id"]
        t, n = context["rows"].T
        X = context["X"]
        for a, kind, s in self.heads:
            if kind == "ego":
                if vid not in s["scores"]:
                    self.missing.add(vid)
                    continue
                ego = s["ego"][vid]
                if (vid, a) not in self.checked:          # the same agent order and head track as the boosted rows
                    track = X[:, ego, a].cpu()
                    if not torch.equal(track, s["track"][vid].to(track.dtype)):
                        raise RuntimeError(f"[Boost] {vid}: the ego's head track differs between step 06 and the boosted rows")
                    self.checked.add((vid, a))
                full = s["scores"][vid].to(self.device)[:, None].expand(-1, X.shape[1]).clone()
                full[:, torch.arange(X.shape[1], device=self.device) != ego] = float("nan")
            else:
                if vid not in s["scores"]:
                    continue                              # no agent of the head's class
                idx, arr = s["scores"][vid]
                if (vid, a) not in self.checked:
                    if not torch.equal(X[:, idx.to(X.device), a].cpu().bool(), s["track"][vid]):
                        raise RuntimeError(f"[Boost] {vid}: the head track of {a} differs between step 06 and the boosted rows")
                    self.checked.add((vid, a))
                full = torch.full((arr.shape[0], X.shape[1]), float("nan"), device=self.device)
                full[:, idx.to(self.device)] = arr.to(self.device)
            m = t < full.shape[0]
            idx_rows = torch.nonzero(m).flatten()
            pv = full[t[idx_rows], n[idx_rows]]
            good = ~torch.isnan(pv)
            idx_rows, pv = idx_rows[good], pv[good]
            out[idx_rows, :, a] = pv[:, None].to(out.dtype)
            self.replaced[a] = self.replaced.get(a, 0) + len(idx_rows)
        return out

    def report(self):
        print(f"[Boost] replaced rows of {len(self.replaced)} heads ({sum(self.replaced.values())} in all); "
              f"videos without ego scores: {len(self.missing)}")


SELECT_CANDIDATES = ("anchor", "boost", "model")   # rule scorers of "rule_sel"; ties go to the earlier one


class SelectScorer:
    """step06 / step07 scorer "rule_sel": per head, the rule scorer with the highest val AP (`step06.select_rules`):
    "anchor" (the mined rules alone), "boost" (the boosted rules, boosted heads only) or "model" (the rule net).
    choice: {head index: candidate}; the heads not in it keep the rule net. anchor: f(inputs) -> (B, W, A)."""
    video_scorer = True

    def __init__(self, choice, anchor, boost, labels):
        self.choice, self.anchor, self.boost, self.labels = choice, anchor, boost, labels
        self.heads = {c: torch.tensor(sorted(a for a, x in choice.items() if x == c), dtype=torch.long)
                      for c in SELECT_CANDIDATES}

    def __call__(self, context, stage1):
        out = stage1.clone()
        for c, f in (("anchor", lambda: self.anchor(context["inputs"])), ("boost", lambda: self.boost(context, stage1))):
            idx = self.heads[c]
            if len(idx):
                idx = idx.to(out.device)
                out[:, :, idx] = f()[:, :, idx].to(out.dtype)
        return out

    def report(self):
        counts = {c: len(v) for c, v in self.heads.items()}
        print(f"[Rule select] heads per rule scorer (highest val AP): {counts}")


def boosted_heads(scorer):
    """The head indices a BoostScorer replaces (fitted heads only)."""
    return sorted({a for a, _, _ in scorer.heads})


def expand_heads(heads, step05_cfg):
    """The head names of --boost_heads: a comma list, or "all" (every label head), "agents" (every non-ego
    head), "ego" (every av: head); mixed lists allowed."""
    names = [step05._atom_name(sig) for sig in utils_data.load_json(step05_cfg["unique_atoms_file"])]
    out = []
    for h in heads:
        if h == "all":
            out += names
        elif h == "agents":
            out += [x for x in names if not x.startswith("av:")]
        elif h == "ego":
            out += [x for x in names if x.startswith("av:")]
        else:
            if h not in names:
                raise ValueError(f"--boost_heads: unknown head {h}")
            out.append(h)
    return list(dict.fromkeys(out))


def settings_tag(heads):
    """The step 06 settings entry of --boost_heads (a new version recomputes the metrics)."""
    return f"{','.join(heads)}:{VERSION}+{AGENT_VERSION}+{NET_VERSION}+{AGENT_NET_VERSION}"


def _model_file(out_dir, head, method="rules"):
    return Path(out_dir) / f"boost_{_tag(head)}_{_version(head, method)}.pt"


def _score_file(out_dir, head, method, split):
    return Path(out_dir) / f"boost_scores_{_tag(head)}_{_version(head, method)}_{split}.pt"


def load_scorer(heads, step05_cfg, input_data, device):
    """{"boost": BoostScorer of the boosted rules, "boost_net": BoostScorer of the SetNets} of the given heads
    (`expand_heads`): fit (cached in input_data["boost_dir"]) and the per-frame scores of the val and test
    videos (cached per head, method and split)."""
    heads = expand_heads(heads, step05_cfg)
    out_dir = Path(input_data["boost_dir"])
    os.makedirs(out_dir, exist_ok=True)
    names = [step05._atom_name(sig) for sig in utils_data.load_json(step05_cfg["unique_atoms_file"])]
    files = {"train": (step05_cfg["train_atom_file"], step05_cfg["train_box_file"]),
             "val": (step05_cfg["val_atom_file"], step05_cfg["val_box_file"]),
             "test": (input_data["test_atoms_file"], input_data["test_box_file"])}
    m_off = max(int(step05_cfg["min_off_frames"]), 1)
    parts = {method: [] for method in METHODS}

    def stale(head, method):
        return not step05.is_fresh(_model_file(out_dir, head, method), step05_cfg["train_atom_file"],
                                   step05_cfg["val_atom_file"])
    # Ego heads: the brake_factors rows of the ego.
    for head in [h for h in heads if h.startswith("av:")]:
        todo = {m: _model_file(out_dir, head, m) for m in METHODS if stale(head, m)}
        if todo:
            fit(step05_cfg, head, todo, device)
        models = [torch.load(_model_file(out_dir, head, m), map_location="cpu", weights_only=False) for m in METHODS]
        merged = {m: {"scores": {}, "ego": {}, "track": {}} for m in METHODS}
        for split in ("val", "test"):
            atom_file, box_file = files[split]
            caches = [_score_file(out_dir, head, m, split) for m in METHODS]
            if all(step05.is_fresh(c, _model_file(out_dir, head, m), atom_file) for c, m in zip(caches, METHODS)):
                res = [torch.load(c, map_location="cpu", weights_only=False) for c in caches]
            else:
                res = score_split(models, step05_cfg, split, atom_file, box_file, device)
                for r, c in zip(res, caches):
                    torch.save(r, c)
            for m, r in zip(METHODS, res):
                for key in merged[m]:
                    merged[m][key].update(r[key])
        for m, model in zip(METHODS, models):
            parts[m].append((names.index(head), "ego", merged[m]))
            print(f"[Boost {m}] {head}: fit val AP {model['best']['val_ap']:.3f}; "
                  f"scores for {len(merged[m]['scores'])} val + test videos")
    # Agent heads: fit the missing models (train + val tables), then score val and test.
    agent_heads = [h for h in heads if not h.startswith("av:")]
    tables, vocab = {}, None

    def get_tables(split):
        nonlocal vocab
        if split not in tables:
            tables[split], p, av = video_tables(step05_cfg, *files[split], names)
            vocab = (p, av)
        return tables[split]
    todo = {h: {m: _model_file(out_dir, h, m) for m in METHODS if stale(h, m)} for h in agent_heads}
    todo = {h: t for h, t in todo.items() if t}
    if todo:
        tb = {"train": get_tables("train"), "val": get_tables("val")}
        for head, out_files in todo.items():
            fit_agent(head, names.index(head), tb, *vocab, names, m_off, out_files, device)
        tables.pop("train", None)
    models = [torch.load(_model_file(out_dir, h, m), map_location="cpu", weights_only=False)
              for h in agent_heads for m in METHODS]
    active = [m for m in models if not m.get("skip")]
    scores = {(m["head"], m["method"]): {"scores": {}, "track": {}} for m in active}
    for split in ("val", "test"):
        atom_file = files[split][0]
        caches = {(m["head"], m["method"]): _score_file(out_dir, m["head"], m["method"], split) for m in active}
        todo_s = [m for m in active if not step05.is_fresh(caches[(m["head"], m["method"])],
                                                           _model_file(out_dir, m["head"], m["method"]), atom_file)]
        if todo_s:
            fresh = score_agents(todo_s, get_tables(split), *vocab, names, m_off, device)
            for m in todo_s:
                key = (m["head"], m["method"])
                track = {x["vid"]: x["X"][:, fresh[key][x["vid"]][0], names.index(m["head"])]
                         for x in tables[split] if x["vid"] in fresh[key]}
                torch.save({"scores": fresh[key], "track": track}, caches[key])
            tables.pop(split, None)
        for m in active:
            key = (m["head"], m["method"])
            s = torch.load(caches[key], map_location="cpu", weights_only=False)
            scores[key]["scores"].update(s["scores"])
            scores[key]["track"].update(s["track"])
    for m in active:
        parts[m["method"]].append((names.index(m["head"]), "agent", scores[(m["head"], m["method"])]))
    if agent_heads:
        by = {(m["head"], m["method"]): m for m in active}
        print(f"[Boost] agent heads: {len({m['head'] for m in active})} fitted, "
              f"{len(agent_heads) - len({m['head'] for m in active})} skipped (too few onsets); fit val AP:")
        for h in agent_heads:
            if (h, "rules") in by:
                r, n = by[(h, "rules")], by.get((h, "net"))
                print(f"    {h:24s} rules {r['best']['val_ap']:.3f} ({len(r['rules'])} rules)  "
                      f"net {n['best']['val_ap'] if n else float('nan'):.3f}  (train pos {r['train_pos']})")
    return {SCORER_OF[m]: BoostScorer(parts[m], device) for m in METHODS}


def model_files(heads, input_data, step05_cfg=None):
    """The boosted model files (freshness of the step 06 metrics)."""
    if step05_cfg is not None:
        heads = expand_heads(heads, step05_cfg)
    return [_model_file(input_data["boost_dir"], h, m) for h in heads for m in METHODS]
