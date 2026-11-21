"""Which facts of the scene go with the ego braking: the size, the position or the speed of the agents, or
all of them together? And over which ranges (a, b), learned from the data, not fixed bins?

Rows: frames t of the train (val) videos where the ego was moving and not braking in the last seen frame
t - 1; y = the ego starts braking (AV-Brake onset) in t .. t + W - 1 (W = 20 frames, 2 s, as clause_slots 2).
Every STRIDE-th frame is a row (neighbouring frames are near copies).

Agent features (per visible agent in frame t - 1; from its 2D box, normalized image coordinates):
  size:     img:log_area (log px^2), img:height (image fraction)
  position: img:bottom (y of the box bottom, 1 = image bottom: near), img:offset (box centre x minus the
            image centre, in box widths: < 0 left), img:abs_offset
  speed:    img:scale (log area growth per frame over motion.K frames: > 0 closing in), img:accel (its
            change), img:lateral (centre x drift, box widths per frame), img:toward_centre (lateral drift
            towards the image centre: > 0 moving in front of the ego, e.g. merging), img:vertical
SUMO ground truth of the same agent (the upper bound of what the boxes could tell; meta/world):
  size: gt:area (footprint m^2); position: gt:fwd, gt:left, gt:abs_left, gt:dist (m, ego camera frame);
  speed: gt:closing (m/s, distance shrinking), gt:lat_closing (|left| shrinking, m/s), gt:speed (m/s),
  lane: gt:same_lane (1 = the ego's lane).
Groups of agents: "ahead" (the ego's lead agent, `roles` ahead: one per row), "veh" (vehicles and
cyclists), "ped". "ego": the row's own features img:scene_scale (`ego_motion`, ~ the ego's speed) and
gt:ego_speed.

Literal: "some agent of the group has feature f in (a, b)" (for "ahead" / "ego": that one agent / the
ego). The bins are the train quantiles of f in the group (Q bins); every interval of consecutive bins is
a candidate. Score: FOIL gain hits x (log p - log base) on train (hits = positive rows covered, p their
precision, base the rows' onset rate), >= MIN_POS positive rows. The best interval per (group, feature) is
then counted on val (precision, lift, events). Pairs: "the same agent has f1 in (a1, b1) and f2 in
(a2, b2)" on Q2 bins per feature; synergy = the pair's gain minus the better of its two single gains
(both on the pair's coarser bins): > 0 when size and speed (say) matter together.

Triples (`triple_search`): "the same agent has a position, a size and a speed feature in their intervals"
(Q3 bins each); synergy = the triple's gain minus the best box with one feature over its whole range (the
best pair / single on the same bins): > 0 when all three matter together.

Regions (`region_search`): the most expressive literal, "the same agent has (f1, .., fk) in a union of
cells" (k = 1, 2, 3 features; 32 / 16 / 8 train-quantile bins per feature): several disjoint ranges per feature
and any shape over the features, grown greedily cell by cell by the FOIL gain. Compared with the single best
interval / box of the same features; synergy = gain minus the best region over fewer features.

Round 2 (`refine`): the best roots (singles / pairs with val lift >= 1.5) get up to two more conditions,
each from a bank of literals "some agent of a group (ahead / veh / ped / the ego) has f in (a, b]" (Q_BANK
train-quantile bins) or its negation "no agent of the group has f in (a, b]" (e.g. "a large vehicle merges
in front AND no vehicle on the right"); the condition of the largest FOIL gain over the root's precision,
kept only if the val precision rises by >= 5% too (>= 10 val onsets). The added agent may be the root's own agent
(no "another agent" binding); a negation excludes it anyway.

Run (server): ./run_roadpp_docker.sh --gpu 0 --brake_factors -- --exp sumo_maps_v7 --machine dgx-jing
Writes <output_dir>/brake_factors/brake_factors_<head>.json and prints the tables.
"""
import gzip
import itertools
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from src.exp_roadpp import config, ego_motion, motion, roles as agent_roles, step05, utils_data, utils_pipe
from src.exp_roadpp.synth.sumo import observe

STRIDE = 2
WINDOW = 20
Q, Q2 = 16, 8
MIN_POS = 100
GROUPS = ("ego", "ahead", "veh", "ped")
VEH_CLASSES = set(step05.MOTOR_CLASSES) | {"Cyc"}
CATEGORY = {
    "img:log_area": "size", "img:height": "size", "gt:area": "size",
    "img:bottom": "position", "img:offset": "position", "img:abs_offset": "position",
    "gt:fwd": "position", "gt:left": "position", "gt:abs_left": "position", "gt:dist": "position",
    "img:scale": "speed", "img:accel": "speed", "img:lateral": "speed", "img:toward_centre": "speed",
    "img:vertical": "speed", "gt:closing": "speed", "gt:lat_closing": "speed", "gt:speed": "speed",
    "gt:same_lane": "lane", "img:scene_scale": "ego speed", "gt:ego_speed": "ego speed",
}
IMG_AGENT = ("img:log_area", "img:height", "img:bottom", "img:offset", "img:abs_offset", "img:scale",
             "img:accel", "img:lateral", "img:toward_centre", "img:vertical")
GT_AGENT = ("gt:area", "gt:fwd", "gt:left", "gt:abs_left", "gt:dist", "gt:closing", "gt:lat_closing",
            "gt:speed", "gt:same_lane")
RATE_FRAMES = 5                        # gt rates over 0.5 s, as motion.K


# -------------- Rows --------------
def image_features(boxes, has_box):
    """{name: (T, N)} of the IMG_AGENT features (NaN where undefined)."""
    W, H = motion.FRAME_SIZE_PX
    b = boxes.float()
    x1, y1, x2, y2 = b.unbind(dim=2)
    w = (x2 - x1).clamp(min=1e-4)
    sig = motion.signals(boxes, has_box)                                          # (T, N, 4)
    cx = (x1 + x2) / 2
    nan = torch.tensor(float("nan"))
    ok = has_box & ~torch.isnan(b).any(dim=2)
    out = {"img:log_area": torch.log(((x2 - x1) * W).clamp(min=1.0) * ((y2 - y1) * H).clamp(min=1.0)),
           "img:height": y2 - y1, "img:bottom": y2, "img:offset": (cx - 0.5) / w,
           "img:abs_offset": (cx - 0.5).abs() / w,
           "img:scale": sig[..., 0], "img:accel": sig[..., 1], "img:lateral": sig[..., 2],
           "img:toward_centre": -sig[..., 2] * torch.sign(cx - 0.5), "img:vertical": sig[..., 3]}
    return {k: torch.where(ok, v, nan) for k, v in out.items()}


def _agent_of(uid, vid, world_frame):
    """The SUMO id of a tube uid "<vid>-<agent id>" (maybe with a "-NNN" tube suffix)."""
    aid = uid[len(vid) + 1:] if uid.startswith(f"{vid}-") else uid
    if aid in world_frame["veh"] or aid in world_frame["ped"]:
        return aid
    return re.sub(r"-[0-9]{3}$", "", aid)


def gt_features(world, w, aid):
    """{name: value} of the GT_AGENT features of agent aid in world index w, or None."""
    def state(k):
        if k < 0 or k >= len(world) or "ego" not in world[k]["veh"]:
            return None
        frame = world[k]
        a = frame["veh"].get(aid) or frame["ped"].get(aid)
        if a is None:
            return None
        cam = observe.Camera(frame["veh"]["ego"])
        fwd, left = cam.to_cam(a["x"], a["y"])
        return a, fwd, left, frame["veh"]["ego"]
    now, before = state(w), state(w - RATE_FRAMES)
    if now is None:
        return None
    a, fwd, left, ego = now
    dt = RATE_FRAMES * world[w].get("dt", 0.1)
    dist = math.hypot(fwd, left)
    out = {"gt:area": a.get("length", 0.5) * a.get("width", 0.5), "gt:fwd": fwd, "gt:left": left,
           "gt:abs_left": abs(left), "gt:dist": dist, "gt:speed": a.get("speed", float("nan")),
           "gt:same_lane": float(a.get("lane") is not None and a.get("lane") == ego.get("lane")),
           "gt:closing": float("nan"), "gt:lat_closing": float("nan")}
    if before is not None:
        _a0, fwd0, left0, _e0 = before
        out["gt:closing"] = (math.hypot(fwd0, left0) - dist) / dt
        out["gt:lat_closing"] = (abs(left0) - abs(left)) / dt
    return out


def collect(cfg, split, dataset, head_name):
    """Rows of one split: {"y": (R,) bool, "event": (R,) long, "ego": {feature: (R,)},
    group: {"row": (n,) long, feature: (n,)} for ahead / veh / ped}."""
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    names = [step05._atom_name(sig) for sig in unique_atoms]
    i_head, i_mov = names.index(head_name), names.index("av:AV-Mov")
    atoms = utils_data.load_json(cfg[f"{split}_atom_file"])
    boxes = torch.load(cfg[f"{split}_box_file"], map_location="cpu")
    A = len(unique_atoms)
    y_all, ev_all, ego_f = [], [], defaultdict(list)
    obs = {g: defaultdict(list) for g in GROUPS[1:]}
    num_rows, gt_videos = 0, 0
    for vi, (vid, video_atoms) in enumerate(tqdm(sorted(atoms.items()), desc=f"Rows ({split})")):
        X = step05.build_video_tensor(video_atoms, A)
        T = X.shape[0]
        agent_ids = step05.video_agent_ids(video_atoms)
        ego = agent_ids.get(None)
        if ego is None or T <= WINDOW + motion.K + 2:
            continue
        vb, classes, has_box = step05._agent_boxes(boxes, vid, video_atoms, X)
        X = X.cpu()
        moving, braking = X[:, ego, i_mov], X[:, ego, i_head]
        onset = step05.atom_onsets(X[:, ego:ego + 1, i_head:i_head + 1], cfg["min_off_frames"])[:, 0, 0]
        img = {k: v.numpy() for k, v in image_features(vb, has_box).items()}
        partner = agent_roles.role_partners(vb.clone(), classes, ego, ("ahead",),
                                            agent_roles.ego_lane(X, unique_atoms))[:, ego, 0]
        scene = ego_motion.signals(vb, has_box, classes, ego, X, unique_atoms)["scene_scale"]
        world_file = dataset / "meta" / "world" / f"{vid}.json.gz"
        world = None
        if world_file.exists():
            with gzip.open(world_file, "rt") as f:
                world = json.load(f)
            gt_videos += 1
        has_box_np = has_box.numpy()
        uid_of = {n: uid for uid, n in agent_ids.items() if uid is not None}
        group_of = ["veh" if c in VEH_CLASSES else ("ped" if c == "Ped" else None) for c in classes]
        for t in range(motion.K + 2, T - WINDOW + 1, STRIDE):
            o = t - 1                                             # the last seen frame
            if not bool(moving[o]) or bool(braking[o]):
                continue
            hit = onset[t:t + WINDOW]
            r = num_rows
            num_rows += 1
            y_all.append(bool(hit.any()))
            ev_all.append(vi * 100000 + t + int(hit.nonzero()[0]) if bool(hit.any()) else -1)
            w = o - 1                                             # world index of atom frame o
            ego_f["img:scene_scale"].append(float(scene[o]))
            ego_speed = float("nan")
            if world is not None and 0 <= w < len(world) and "ego" in world[w]["veh"]:
                ego_speed = world[w]["veh"]["ego"]["speed"]
            ego_f["gt:ego_speed"].append(ego_speed)
            lead = int(partner[o])
            for n in has_box_np[o].nonzero()[0].tolist():
                groups = [g for g in (group_of[n], "ahead" if n == lead else None) if g]
                if not groups:
                    continue
                values = {k: float(v[o, n]) for k, v in img.items()}          # numpy: fast scalar reads
                gt = None
                if world is not None and 0 <= w < len(world) and n in uid_of:
                    gt = gt_features(world, w, _agent_of(uid_of[n], vid, world[w]))
                for k in GT_AGENT:
                    values[k] = gt[k] if gt else float("nan")
                for g in groups:
                    obs[g]["row"].append(r)
                    for k, v in values.items():
                        obs[g][k].append(v)
    out = {"y": torch.tensor(y_all), "event": torch.tensor(ev_all, dtype=torch.long),
           "ego": {k: torch.tensor(v) for k, v in ego_f.items()}}
    for g, d in obs.items():
        out[g] = {k: torch.tensor(v, dtype=torch.long if k == "row" else torch.float) for k, v in d.items()}
    print(f"[Rows] {split}: {num_rows} rows, {int(out['y'].sum())} positive "
          f"({len(set(e for e in ev_all if e >= 0))} onsets), SUMO world for {gt_videos} videos")
    return out


# -------------- Interval search --------------
def _obs(data, group, feature):
    """(rows, values) of one feature in one group, NaN dropped."""
    if group == "ego":
        v = data["ego"][feature]
        rows = torch.arange(len(v))
    else:
        rows, v = data[group]["row"], data[group][feature]
    keep = ~torch.isnan(v)
    return rows[keep], v[keep]


def _edges(values, bins):
    if len(values) == 0:
        return torch.zeros(0)
    q = torch.quantile(values[torch.randperm(len(values), generator=torch.Generator().manual_seed(0))[:1_000_000]],
                       torch.linspace(0, 1, bins + 1)[1:-1])
    return torch.unique(q)


def _presence(rows, bins, R, B, device):
    """(R, B) float: some observation of the row falls into bin b."""
    P = torch.zeros((R, B), device=device)
    P[rows.to(device), bins.to(device)] = 1.0
    return P


def _stats(member, y, base):
    """hits, rows, precision, gain of (R, K) bool memberships."""
    yf = y.float()
    hits = yf @ member.float()
    rows = member.float().sum(dim=0)
    p = hits / rows.clamp(min=1)
    gain = hits * (torch.log(p.clamp(min=1e-12)) - math.log(base))
    gain = torch.where((hits >= MIN_POS) & (p > base), gain, torch.full_like(gain, -1.0))
    return hits, rows, p, gain


def _interval_index(B):
    i, j = torch.triu_indices(B, B)                                  # i <= j: bins i..j
    return i, j


def single_search(train, val, group, feature, device, bins=Q):
    rows, v = _obs(train, group, feature)
    edges = _edges(v, bins)
    B = len(edges) + 1
    R = len(train["y"])
    y = train["y"].to(device)
    base = float(train["y"].float().mean())
    P = _presence(rows, torch.bucketize(v, edges), R, B, device)
    C = torch.cat([torch.zeros((R, 1), device=device), P.cumsum(dim=1)], dim=1)       # (R, B + 1)
    i, j = _interval_index(B)
    member = (C[:, j + 1] - C[:, i]) > 0
    hits, cov, p, gain = _stats(member, y, base)
    k = int(gain.argmax())
    if gain[k] <= 0:
        return None
    lo = float(edges[i[k] - 1]) if i[k] > 0 else -math.inf
    hi = float(edges[j[k]]) if j[k] < B - 1 else math.inf
    res = {"group": group, "feature": feature, "category": CATEGORY.get(feature, "?"), "a": lo, "b": hi,
           "train_precision": float(p[k]), "train_hits": int(hits[k]), "train_rows": int(cov[k]),
           "gain": float(gain[k]), "train_base": base}
    res.update(_val_of(val, group, [(feature, lo, hi)], device))
    return res


def _val_of(val, group, conditions, device):
    """Val precision / lift / events of "some agent of group has every feature in its (a, b)"."""
    R = len(val["y"])
    base = float(val["y"].float().mean())
    if group == "ego":
        ok = torch.ones(R, dtype=torch.bool)
        for f, lo, hi in conditions:
            v = val["ego"][f]
            ok &= (v > lo) & (v <= hi)
        member = ok
    else:
        d = val[group]
        ok = torch.ones(len(d["row"]), dtype=torch.bool)
        for f, lo, hi in conditions:
            ok &= (d[f] > lo) & (d[f] <= hi)
        member = torch.zeros(R, dtype=torch.bool)
        member[d["row"][ok]] = True
    hits = int((member & val["y"]).sum())
    rows = int(member.sum())
    events = int(val["event"][member & val["y"]].unique().numel())
    p = hits / max(rows, 1)
    return {"val_precision": p, "val_lift": p / max(base, 1e-12), "val_rows": rows, "val_hits": hits,
            "val_events": events, "val_base": base, "val_recall": hits / max(int(val["y"].sum()), 1)}


def pair_search(train, val, group, f1, f2, device):
    """Best rectangle (f1 in bins i1..j1, f2 in bins i2..j2 on the same agent) and the best single interval
    of f1 and of f2 on the same coarse bins; None without enough data."""
    d = train[group]
    keep = ~torch.isnan(d[f1]) & ~torch.isnan(d[f2])
    if int(keep.sum()) < 1000:
        return None
    rows, v1, v2 = d["row"][keep], d[f1][keep], d[f2][keep]
    e1, e2 = _edges(v1, Q2), _edges(v2, Q2)
    B1, B2 = len(e1) + 1, len(e2) + 1
    R = len(train["y"])
    y = train["y"].to(device)
    base = float(train["y"].float().mean())
    b1, b2 = torch.bucketize(v1, e1), torch.bucketize(v2, e2)
    P = torch.zeros((R, B1, B2), device=device)
    P[rows.to(device), b1.to(device), b2.to(device)] = 1.0
    C = torch.zeros((R, B1 + 1, B2 + 1), device=device)
    C[:, 1:, 1:] = P.cumsum(dim=1).cumsum(dim=2)
    i1, j1 = _interval_index(B1)
    i2, j2 = _interval_index(B2)
    I1, I2 = torch.meshgrid(torch.arange(len(i1)), torch.arange(len(i2)), indexing="ij")
    I1, I2 = I1.flatten(), I2.flatten()
    a1, c1, a2, c2 = i1[I1], j1[I1] + 1, i2[I2], j2[I2] + 1
    best = (-1.0, None)
    for s in range(0, len(I1), 256):
        sl = slice(s, s + 256)
        total = (C[:, c1[sl], c2[sl]] - C[:, a1[sl], c2[sl]] - C[:, c1[sl], a2[sl]] + C[:, a1[sl], a2[sl]])
        _h, _r, _p, gain = _stats(total > 0, y, base)
        k = int(gain.argmax())
        if float(gain[k]) > best[0]:
            best = (float(gain[k]), s + k, float(_p[k]), int(_h[k]))
    singles = []
    for e, f in ((e1, f1), (e2, f2)):
        bins = torch.bucketize(d[f][~torch.isnan(d[f])], e)
        Pf = _presence(d["row"][~torch.isnan(d[f])], bins, R, len(e) + 1, device)
        Cf = torch.cat([torch.zeros((R, 1), device=device), Pf.cumsum(dim=1)], dim=1)
        i, j = _interval_index(len(e) + 1)
        singles.append(float(_stats((Cf[:, j + 1] - Cf[:, i]) > 0, y, base)[3].max()))
    if best[1] is None or best[0] <= 0:
        return None
    k = best[1]

    def bounds(e, a, c, B):
        return (float(e[a - 1]) if a > 0 else -math.inf, float(e[c - 1]) if c < B else math.inf)
    lo1, hi1 = bounds(e1, int(a1[k]), int(c1[k]), B1)
    lo2, hi2 = bounds(e2, int(a2[k]), int(c2[k]), B2)
    res = {"group": group, "f1": f1, "f2": f2, "a1": lo1, "b1": hi1, "a2": lo2, "b2": hi2, "gain": best[0],
           "train_precision": best[2], "train_hits": best[3], "single_gain_f1": singles[0],
           "single_gain_f2": singles[1], "synergy": best[0] - max(singles)}
    res.update(_val_of(val, group, [(f1, lo1, hi1), (f2, lo2, hi2)], device))
    return res




# -------------- Triples: position x size x speed on the same agent --------------
Q3 = 8                     # bins per feature of a triple (36 intervals each, 46656 boxes; 6 bins were too coarse in the planted test)
TRIPLE_FEATURES = {
    "img": {"position": ("img:bottom", "img:offset", "img:abs_offset"), "size": ("img:log_area", "img:height"),
            "speed": ("img:scale", "img:accel", "img:lateral", "img:toward_centre", "img:vertical")},
    "gt": {"position": ("gt:fwd", "gt:left", "gt:abs_left", "gt:dist"), "size": ("gt:area",),
           "speed": ("gt:closing", "gt:lat_closing", "gt:speed")},
}


def triple_search(train, val, group, feats, device, bins=Q3):
    """Best box "the same agent has f1 in (a1, b1], f2 in (a2, b2] and f3 in (a3, b3]" (bins of each feature
    from its train quantiles) by train FOIL gain, and the best box with at least one feature over its whole
    range (= the best pair or single on the same bins): synergy = the triple's gain minus that one."""
    d = train[group]
    keep = torch.ones(len(d["row"]), dtype=torch.bool)
    for f in feats:
        keep &= ~torch.isnan(d[f])
    if int(keep.sum()) < 1000:
        return None
    rows = d["row"][keep]
    values = [d[f][keep] for f in feats]
    edges = [_edges(v, bins) for v in values]
    Bs = [len(e) + 1 for e in edges]
    R = len(train["y"])
    y = train["y"].to(device)
    base = float(train["y"].float().mean())
    P = torch.zeros((R, *Bs), device=device)
    idx = [torch.bucketize(v, e).to(device) for v, e in zip(values, edges)]
    P[rows.to(device), idx[0], idx[1], idx[2]] = 1.0
    C = torch.zeros((R, Bs[0] + 1, Bs[1] + 1, Bs[2] + 1), device=device)
    C[:, 1:, 1:, 1:] = P.cumsum(dim=1).cumsum(dim=2).cumsum(dim=3)
    del P
    ivs = [_interval_index(B) for B in Bs]
    n = [len(iv[0]) for iv in ivs]
    grid = torch.cartesian_prod(torch.arange(n[0]), torch.arange(n[1]), torch.arange(n[2]))
    lo = [ivs[k][0][grid[:, k]] for k in range(3)]                    # first bin
    hi = [ivs[k][1][grid[:, k]] + 1 for k in range(3)]                # one past the last bin
    full = torch.zeros(len(grid), dtype=torch.bool)
    for k in range(3):
        full |= (lo[k] == 0) & (hi[k] == Bs[k])
    best = {True: (-1.0, None, 0.0, 0), False: (-1.0, None, 0.0, 0)}  # key: some feature over its full range
    for s in range(0, len(grid), 128):
        sl = slice(s, s + 128)
        a0, a1, a2 = lo[0][sl], lo[1][sl], lo[2][sl]
        c0, c1, c2 = hi[0][sl], hi[1][sl], hi[2][sl]
        total = (C[:, c0, c1, c2] - C[:, a0, c1, c2] - C[:, c0, a1, c2] - C[:, c0, c1, a2]
                 + C[:, a0, a1, c2] + C[:, a0, c1, a2] + C[:, c0, a1, a2] - C[:, a0, a1, a2])
        hits, _rows, p, gain = _stats(total > 0, y, base)
        f_sl = full[sl].to(device)
        for is_full in (True, False):
            g = torch.where(f_sl == is_full, gain, torch.full_like(gain, -1.0))
            k = int(g.argmax())
            if float(g[k]) > best[is_full][0]:
                best[is_full] = (float(g[k]), s + k, float(p[k]), int(hits[k]))
    gain, k, p_tr, hits_tr = best[False]
    if k is None or gain <= 0:
        return None

    def bounds(e, a, c, B):
        return (float(e[a - 1]) if a > 0 else -math.inf, float(e[c - 1]) if c < B else math.inf)
    cond = [(feats[m], *bounds(edges[m], int(lo[m][k]), int(hi[m][k]), Bs[m])) for m in range(3)]
    res = {"group": group, "features": list(feats), "conditions": [[f, a, b] for f, a, b in cond], "gain": gain,
           "train_precision": p_tr, "train_hits": hits_tr, "best_sub_gain": best[True][0],
           "synergy": gain - best[True][0]}
    res.update(_val_of(val, group, cond, device))
    return res


# -------------- Regions: any union of cells (several disjoint ranges per feature) --------------
Q_REGION = {1: 32, 2: 16, 3: 8}   # bins per feature of a region over 1 / 2 / 3 features (32 / 256 / 512 cells)
REGION_MIN_NEW_HITS = 20          # a cell joins the region only if it adds >= this many positive train rows
REGION_MARGIN_LIFT = 1.3          # ... at a precision (of the rows it adds) >= this x the base rate
REGION_MAX_STEPS = 300


def _cells(data, group, feats, edges):
    """(rows, cell index) of the observations of group with every feature known (cells of the edges grid)."""
    if group == "ego":
        R = len(data["y"])
        rows, cols = torch.arange(R), [data["ego"][f] for f in feats]
    else:
        rows, cols = data[group]["row"], [data[group][f] for f in feats]
    keep = torch.ones(len(rows), dtype=torch.bool)
    for c in cols:
        keep &= ~torch.isnan(c)
    cell = torch.zeros(int(keep.sum()), dtype=torch.long)
    for c, e in zip(cols, edges):
        cell = cell * (len(e) + 1) + torch.bucketize(c[keep], e)
    return rows[keep], cell


def _region_text(feats, edges, selected):
    """A region as a union of boxes: per feature its bins merged into ranges along the last feature."""
    sizes = [len(e) + 1 for e in edges]

    def rng(k, a, b):                      # bins a..b of feature k
        lo = float(edges[k][a - 1]) if a > 0 else -math.inf
        hi = float(edges[k][b]) if b < sizes[k] - 1 else math.inf
        return f"({_fmt(lo)}, {_fmt(hi)}]"
    cells = sorted(selected)
    idx = []
    for c in cells:
        t = []
        for size in reversed(sizes):
            t.append(c % size)
            c //= size
        idx.append(tuple(reversed(t)))
    boxes = []                                 # merge runs along the last feature with equal other bins
    for t in idx:
        if boxes and boxes[-1][0] == t[:-1] and boxes[-1][2] == t[-1] - 1:
            boxes[-1][2] = t[-1]
        else:
            boxes.append([t[:-1], t[-1], t[-1]])
    more = f" ... (+{len(boxes) - 60} boxes)" if len(boxes) > 60 else ""
    boxes = boxes[:60]
    if len(feats) == 1:
        return " U ".join(rng(0, b[1], b[2]) for b in boxes) + more
    parts = []
    for prefix, a, b in boxes:
        parts.append(" & ".join([f"{feats[k]} {rng(k, prefix[k], prefix[k])}" for k in range(len(prefix))]
                                + [f"{feats[-1]} {rng(len(feats) - 1, a, b)}"]))
    return " | ".join(f"[{x}]" for x in parts) + more


def average_precision(score, y):
    """AP of the rows ranked by score (ties ranked together); None without a positive."""
    y = y.float().to(score.device)
    if float(y.sum()) == 0:
        return None
    order = torch.argsort(score, descending=True)
    s, yy = score[order], y[order]
    tp = torch.cumsum(yy, 0)
    _, counts = torch.unique_consecutive(s, return_counts=True)
    ends = torch.cumsum(counts, 0) - 1
    tp_e = tp[ends]
    precision = tp_e / (ends + 1).float()
    gained = torch.diff(tp_e, prepend=tp_e.new_zeros(1))
    return float((precision * gained).sum() / tp_e[-1])


def _presence_matrix(data, group, feats, edges, n_cells, device):
    rows, cell = _cells(data, group, feats, edges)
    P = torch.zeros((len(data["y"]), n_cells), device=device)
    P[rows.to(device), cell.to(device)] = 1.0
    return P


def region_search(train, val, group, feats, device):
    """Greedy region "the same agent has (f1, .., fk) in a union of cells": cells of Q_REGION[k] train-quantile
    bins per feature; from the empty region, add the cell of the largest FOIL gain of the region (hits x
    (log p - log base), exact: one matmul per step over all cells), if it adds >= REGION_MIN_NEW_HITS positive
    rows at a precision >= REGION_MARGIN_LIFT x base; stop when the gain no longer rises. Several disjoint
    ranges per feature and any shape over the features (not only one interval / box)."""
    bins = Q_REGION[len(feats)]
    rows_tr = train["ego"][feats[0]] if group == "ego" else train[group][feats[0]]
    if len(rows_tr) == 0:
        return None
    if group == "ego":
        values = [train["ego"][f][~torch.isnan(train["ego"][f])] for f in feats]
    else:
        d = train[group]
        keep = torch.ones(len(d["row"]), dtype=torch.bool)
        for f in feats:
            keep &= ~torch.isnan(d[f])
        if int(keep.sum()) < 1000:
            return None
        values = [d[f][keep] for f in feats]
    edges = [_edges(v, bins) for v in values]
    n_cells = 1
    for e in edges:
        n_cells *= len(e) + 1
    R = len(train["y"])
    P = _presence_matrix(train, group, feats, edges, n_cells, device)
    y = train["y"].to(device).float()
    base = float(y.mean())
    # Capacity of these features: every cell scored by its train onset rate, a row by its best agent's cell
    # (the nested regions of all rate thresholds at once); train AP = how well they can fit, val AP = hold.
    rate = (y @ P) / P.sum(dim=0).clamp(min=1)
    P_va = _presence_matrix(val, group, feats, edges, n_cells, device)
    lookup = {}
    for name, M, yy in (("train", P, train["y"]), ("val", P_va, val["y"])):
        score = (M * rate[None]).amax(dim=1)
        score = torch.where(M.sum(dim=1) > 0, score, torch.full_like(score, base))
        lookup[name] = average_precision(score, yy)
    del P_va
    covered = torch.zeros(R, device=device)
    hits, cov, gain = 0.0, 0.0, 0.0
    selected = []
    for _step in range(REGION_MAX_STEPS):
        free = 1.0 - covered
        new_hits = (y * free) @ P
        new_rows = free @ P
        h, n = hits + new_hits, cov + new_rows
        g = h * (torch.log((h / n.clamp(min=1)).clamp(min=1e-12)) - math.log(base))
        ok = (new_hits >= REGION_MIN_NEW_HITS) & (new_hits / new_rows.clamp(min=1) >= REGION_MARGIN_LIFT * base)
        if selected:
            ok[torch.tensor(selected, device=device)] = False
        g = torch.where(ok, g, torch.full_like(g, -1.0))
        k = int(g.argmax())
        if float(g[k]) <= gain:
            break
        selected.append(k)
        hits, cov, gain = float(h[k]), float(n[k]), float(g[k])
        covered = torch.clamp(covered + P[:, k], max=1.0)
    if not selected:
        return None
    in_region = torch.zeros(n_cells, dtype=torch.bool)
    in_region[torch.tensor(selected)] = True
    member_tr = covered.cpu() > 0
    rows_va, cell_va = _cells(val, group, feats, edges)
    member_va = torch.zeros(len(val["y"]), dtype=torch.bool)
    member_va[rows_va[in_region[cell_va]]] = True
    y_va = val["y"]
    hits_va, rows_n = int((member_va & y_va).sum()), int(member_va.sum())
    p_va, base_va = hits_va / max(rows_n, 1), float(y_va.float().mean())
    text = _region_text(feats, edges, selected)
    return {"group": group, "features": list(feats), "kind": "region", "cells": len(selected), "of_cells": n_cells,
            "text": text, "gain": gain, "train_precision": hits / max(cov, 1), "train_hits": int(hits),
            "val_precision": p_va, "val_lift": p_va / max(base_va, 1e-12), "val_rows": rows_n, "val_hits": hits_va,
            "val_events": int(val["event"][member_va & y_va].unique().numel()),
            "val_recall": hits_va / max(int(y_va.sum()), 1), "train_recall": hits / max(float(y.sum()), 1),
            "lookup_train_ap": lookup["train"], "lookup_val_ap": lookup["val"], "_member": (member_tr, member_va)}

# -------------- Refinement: conditions on other agents, negated ones too --------------
Q_BANK = 8                 # bins per feature of the literal bank
REFINE_ROOTS = 6           # roots: the best singles / pairs (val lift >= ROOT_MIN_LIFT, >= 20 val events)
ROOT_MIN_LIFT = 1.5
REFINE_DEPTH = 2           # conditions added to a root
REFINE_MIN_POS = 50        # positive rows a refined rule keeps (train)
REFINE_MIN_VAL_EVENTS = 10 # ... and keep >= this many val onsets
ROOT_MIN_EVENTS = 20       # val onsets of a root
REFINE_VAL_RISE = 1.05     # an added condition must raise the val precision by >= 5% (relative): noise
                           # conditions pass a bare "rises" check (planted test: a random 3rd condition did)


def member_of(data, group, conditions):
    """(R,) bool: some agent of group (the ego for "ego") has every feature in its (a, b]."""
    R = len(data["y"])
    if group == "ego":
        ok = torch.ones(R, dtype=torch.bool)
        for f, lo, hi in conditions:
            v = data["ego"][f]
            ok &= (v > lo) & (v <= hi)
        return ok
    d = data[group]
    ok = torch.ones(len(d["row"]), dtype=torch.bool)
    for f, lo, hi in conditions:
        ok &= (d[f] > lo) & (d[f] <= hi)
    member = torch.zeros(R, dtype=torch.bool)
    member[d["row"][ok]] = True
    return member


def literal_bank(train, val):
    """Literals "some agent of a group has f in (a, b]" for every group, feature and interval of Q_BANK
    train-quantile bins: [description], (R_train, n) bool, (R_val, n) bool. The negation of a literal
    ("no agent of the group has f in (a, b]") is its complement, taken in `refine`."""
    descs, cols_tr, cols_va = [], [], []
    for group in GROUPS:
        feats = ("img:scene_scale", "gt:ego_speed") if group == "ego" else IMG_AGENT + GT_AGENT
        for f in feats:
            rows, v = _obs(train, group, f)
            if len(v) < 1000:
                continue
            edges = _edges(v, Q_BANK)
            B = len(edges) + 1
            i, j = _interval_index(B)
            per_split = []
            for data in (train, val):
                r, x = _obs(data, group, f)
                P = _presence(r, torch.bucketize(x, edges), len(data["y"]), B, "cpu")
                C = torch.cat([torch.zeros((len(data["y"]), 1)), P.cumsum(dim=1)], dim=1)
                per_split.append((C[:, j + 1] - C[:, i]) > 0)
            for k in range(len(i)):
                if i[k] == 0 and j[k] == B - 1:
                    continue                                    # the whole range: "the agent exists"
                lo = float(edges[i[k] - 1]) if i[k] > 0 else -math.inf
                hi = float(edges[j[k]]) if j[k] < B - 1 else math.inf
                descs.append({"group": group, "feature": f, "a": lo, "b": hi})
            keep = ~((i == 0) & (j == B - 1))
            cols_tr.append(per_split[0][:, keep])
            cols_va.append(per_split[1][:, keep])
    return descs, torch.cat(cols_tr, dim=1), torch.cat(cols_va, dim=1)


def literal_text(desc, negated=False):
    scope = {"ego": "ego", "ahead": "ahead"}.get(desc["group"], f"some {desc['group']}")
    body = desc.get("body") or f"{desc['feature']} in ({_fmt(desc['a'])}, {_fmt(desc['b'])}]"
    if negated:
        return f"NOT [{scope}: {body}]" if desc["group"] in ("ego", "ahead") else f"no {desc['group']}: {body}"
    return f"{scope}: {body}"


def refine(root, train, val, bank, device):
    """Greedy refinement of a root (a single or pair result): up to REFINE_DEPTH literals of the bank or
    their negations, each the largest FOIL gain hits x (log p - log p_now) on the root's train rows still
    covered, kept only if the val precision rises by REFINE_VAL_RISE too. Returns the steps with train / val stats."""
    descs, M_tr, M_va = bank
    if "_member" in root:
        S_tr, S_va = root["_member"]
        text = literal_text({"group": root["group"], "body": " , ".join(root["features"]) + " in " + root["text"]})
    elif "conditions" in root:
        cond = [tuple(c) for c in root["conditions"]]
        text = literal_text({"group": root["group"], "body": " & same agent ".join(
            f"{f} in ({_fmt(lo)}, {_fmt(hi)}]" for f, lo, hi in cond)})
    elif "f1" in root:
        cond = [(root["f1"], root["a1"], root["b1"]), (root["f2"], root["a2"], root["b2"])]
        text = (f"{literal_text({'group': root['group'], 'feature': root['f1'], 'a': root['a1'], 'b': root['b1']})}"
                f" & same agent {root['f2']} in ({_fmt(root['a2'])}, {_fmt(root['b2'])}]")
    else:
        cond = [(root["feature"], root["a"], root["b"])]
        text = literal_text({"group": root["group"], "feature": root["feature"], "a": root["a"], "b": root["b"]})
    if "_member" not in root:
        S_tr, S_va = member_of(train, root["group"], cond), member_of(val, root["group"], cond)
    y_tr, y_va = train["y"], val["y"]

    def stats(S, y, ev):
        rows, hits = int(S.sum()), int((S & y).sum())
        return {"rows": rows, "hits": hits, "precision": hits / max(rows, 1),
                "events": int(ev[S & y].unique().numel()), "recall": hits / max(int(y.sum()), 1)}
    steps = [{"add": text, "train": stats(S_tr, y_tr, train["event"]), "val": stats(S_va, y_va, val["event"])}]
    Mf = M_tr.to(device)
    for _depth in range(REFINE_DEPTH):
        now = steps[-1]
        p_now = now["train"]["precision"]
        s = S_tr.to(device).float()
        ys = (S_tr & y_tr).to(device).float()
        hits_pos = torch.cat([ys @ Mf[:, c:c + 2048].float() for c in range(0, Mf.shape[1], 2048)])
        rows_pos = torch.cat([s @ Mf[:, c:c + 2048].float() for c in range(0, Mf.shape[1], 2048)])
        hits = torch.cat([hits_pos, ys.sum() - hits_pos])                 # literals, then their negations
        rows = torch.cat([rows_pos, s.sum() - rows_pos])
        p = hits / rows.clamp(min=1)
        gain = hits * (torch.log(p.clamp(min=1e-12)) - math.log(max(p_now, 1e-12)))
        gain = torch.where((hits >= REFINE_MIN_POS) & (p > p_now), gain, torch.full_like(gain, -1.0))
        order = torch.argsort(gain, descending=True)[:20].tolist()
        chosen = None
        for k in order:                                               # the best one that also holds on val
            if gain[k] <= 0:
                break
            neg, d = k >= len(descs), k % len(descs)
            col_tr, col_va = M_tr[:, d], M_va[:, d]
            if neg:
                col_tr, col_va = ~col_tr, ~col_va
            nxt_tr, nxt_va = S_tr & col_tr, S_va & col_va
            st_va = stats(nxt_va, y_va, val["event"])
            if (st_va["precision"] >= REFINE_VAL_RISE * now["val"]["precision"]
                    and st_va["events"] >= REFINE_MIN_VAL_EVENTS):
                chosen = (literal_text(descs[d], neg), nxt_tr, nxt_va, st_va, float(gain[k]))
                break
        if chosen is None:
            break
        add, S_tr, S_va, st_va, g = chosen
        steps.append({"add": add, "gain": g, "train": stats(S_tr, y_tr, train["event"]), "val": st_va})
    return steps


# -------------- Neural reference on the same features --------------
NET_EPOCHS = 400           # full-batch steps
NET_HIDDEN = 64


def _net_inputs(data, feats_agent, feats_ego, stats, device):
    """Standardized features with NaN -> 0 plus a known flag per feature: ego (R, 2E), ahead (R, 2F + 1),
    veh / ped observations (n, 2F) with their rows."""
    def prep(x, mu, sd):
        known = ~torch.isnan(x)
        z = torch.where(known, (x - mu) / sd, torch.zeros_like(x))
        return torch.cat([z, known.float()], dim=1)
    R = len(data["y"])
    out = {"R": R}
    E = torch.stack([data["ego"][f] for f in feats_ego], dim=1)
    out["ego"] = prep(E, *stats["ego"]).to(device)
    for g in ("ahead", "veh", "ped"):
        X = torch.stack([data[g][f] for f in feats_agent], dim=1)
        out[g] = (data[g]["row"].to(device), prep(X, *stats[g]).to(device))
    return out


def _stats_of(data, feats_agent, feats_ego):
    def ms(X):
        mu = torch.nanmean(X, dim=0)
        sd = torch.sqrt(torch.nanmean((X - mu) ** 2, dim=0)).nan_to_num(1.0).clamp(min=1e-6)
        return mu.nan_to_num(0.0), sd
    st = {"ego": ms(torch.stack([data["ego"][f] for f in feats_ego], dim=1))}
    for g in ("ahead", "veh", "ped"):
        st[g] = ms(torch.stack([data[g][f] for f in feats_agent], dim=1))
    return st


class SetNet(torch.nn.Module):
    """Per-agent MLP of each group (ahead, veh, ped), max-pooled over the row's agents, then a row MLP with the
    ego features: the neural upper reference of what these features can fit."""
    def __init__(self, f_agent, f_ego, h=NET_HIDDEN):
        super().__init__()
        mlp = lambda i: torch.nn.Sequential(torch.nn.Linear(i, h), torch.nn.GELU(), torch.nn.Linear(h, h),
                                            torch.nn.GELU())
        self.phi = torch.nn.ModuleDict({g: mlp(2 * f_agent) for g in ("ahead", "veh", "ped")})
        self.head = torch.nn.Sequential(torch.nn.Linear(2 * f_ego + 3 * (h + 1), h), torch.nn.GELU(),
                                        torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.Linear(h, 1))
        self.h = h

    def forward(self, x):
        parts = [x["ego"]]
        for g in ("ahead", "veh", "ped"):
            rows, feats = x[g]
            z = self.phi[g](feats)
            pooled = torch.full((x["R"], self.h), -1e4, device=z.device)
            pooled = pooled.scatter_reduce(0, rows[:, None].expand(-1, self.h), z, reduce="amax")
            has = torch.zeros(x["R"], device=z.device).index_fill_(0, rows, 1.0)
            parts += [torch.where(has[:, None] > 0, pooled, torch.zeros_like(pooled)), has[:, None]]
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


def neural_reference(train, val, source, device):
    """Train / val AP of a SetNet on the img (or img + gt) features, trained to fit train (no early stop);
    also its best val AP over the steps."""
    feats_agent = IMG_AGENT if source == "img" else IMG_AGENT + GT_AGENT
    feats_ego = ("img:scene_scale",) if source == "img" else ("img:scene_scale", "gt:ego_speed")
    st = _stats_of(train, feats_agent, feats_ego)
    x_tr = _net_inputs(train, feats_agent, feats_ego, st, device)
    x_va = _net_inputs(val, feats_agent, feats_ego, st, device)
    y_tr = train["y"].float().to(device)
    torch.manual_seed(0)
    net = SetNet(len(feats_agent), len(feats_ego)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best_val, history = 0.0, []
    for step in range(NET_EPOCHS):
        net.train()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x_tr), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            net.eval()
            with torch.no_grad():
                ap_tr, ap_va = average_precision(net(x_tr), train["y"]), average_precision(net(x_va), val["y"])
            best_val = max(best_val, ap_va)
            history.append({"step": step + 1, "loss": float(loss), "train_ap": ap_tr, "val_ap": ap_va})
    return {"source": source, "train_ap": history[-1]["train_ap"], "val_ap": history[-1]["val_ap"],
            "best_val_ap": best_val, "history": history}

# -------------- Main --------------
def _fmt(x):
    return "-inf" if x == -math.inf else ("inf" if x == math.inf else f"{x:.3g}")


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--head", type=str, default="av:AV-Brake", help="the ego head")
    parser.add_argument("--capacity", action="store_true",
                        help="capacity mode: can the interval / region literals fit train at all? No noise guards "
                             "(min hits, margin lift, val checks), finer region bins (64 / 32 / 16), deeper "
                             "refinement chosen on train only; writes brake_factors_<head>_capacity.json")
    args = utils_pipe.parse_args(parser)
    global MIN_POS, Q_REGION, REGION_MIN_NEW_HITS, REGION_MARGIN_LIFT, REGION_MAX_STEPS, REFINE_DEPTH, \
        REFINE_MIN_POS, REFINE_VAL_RISE, REFINE_MIN_VAL_EVENTS, ROOT_MIN_LIFT, ROOT_MIN_EVENTS
    if args.capacity:
        MIN_POS, REGION_MIN_NEW_HITS, REGION_MARGIN_LIFT = 10, 1, 0.0
        Q_REGION, REGION_MAX_STEPS = {1: 64, 2: 32, 3: 16}, 5000
        REFINE_DEPTH, REFINE_MIN_POS, REFINE_VAL_RISE, REFINE_MIN_VAL_EVENTS = 4, 10, 0.0, 0
        ROOT_MIN_LIFT, ROOT_MIN_EVENTS = 0.0, 0
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id if getattr(args, "run_id", None) \
        else args.output_dir / "brake_factors"
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    out_dir = args.output_dir / "brake_factors"
    os.makedirs(out_dir, exist_ok=True)
    dataset = Path(args.dataset_path)
    tag = args.head.replace(":", "_")
    cache = out_dir / f"rows_{tag}_s{STRIDE}w{WINDOW}.pt"
    if os.path.exists(cache):
        train, val = torch.load(cache, map_location="cpu")
    else:
        train = collect(cfg, "train", dataset, args.head)
        val = collect(cfg, "val", dataset, args.head)
        torch.save((train, val), cache)

    singles = []
    for group in GROUPS:
        feats = ("img:scene_scale", "gt:ego_speed") if group == "ego" else IMG_AGENT + GT_AGENT
        for f in tqdm(feats, desc=f"Intervals ({group})"):
            r = single_search(train, val, group, f, device)
            if r is not None:
                singles.append(r)
    pairs = []
    for group in ("ahead", "veh"):
        for feats in (IMG_AGENT, GT_AGENT):
            for a in range(len(feats)):
                for b in range(a + 1, len(feats)):
                    r = pair_search(train, val, group, feats[a], feats[b], device)
                    if r is not None:
                        pairs.append(r)

    triples = []
    for group in ("ahead", "veh"):
        for source, cats in TRIPLE_FEATURES.items():
            combos = [(p, z, v) for p in cats["position"] for z in cats["size"] for v in cats["speed"]]
            for feats in tqdm(combos, desc=f"Triples ({group}, {source})"):
                r = triple_search(train, val, group, feats, device)
                if r is not None:
                    triples.append(r)

    # Regions: any union of cells, over 1 / 2 / 3 features of the same agent.
    regions = {1: [], 2: [], 3: []}
    for group in GROUPS:
        feats = ("img:scene_scale", "gt:ego_speed") if group == "ego" else IMG_AGENT + GT_AGENT
        for f in tqdm(feats, desc=f"Regions 1 ({group})"):
            r = region_search(train, val, group, (f,), device)
            if r is not None:
                regions[1].append(r)
    for group in ("ahead", "veh"):
        for feats in (IMG_AGENT, GT_AGENT):
            combos = [(feats[a], feats[b]) for a in range(len(feats)) for b in range(a + 1, len(feats))]
            for fs in tqdm(combos, desc=f"Regions 2 ({group})"):
                r = region_search(train, val, group, fs, device)
                if r is not None:
                    regions[2].append(r)
        for source, cats in TRIPLE_FEATURES.items():
            combos = [(p, z, v) for p in cats["position"] for z in cats["size"] for v in cats["speed"]]
            for fs in tqdm(combos, desc=f"Regions 3 ({group}, {source})"):
                r = region_search(train, val, group, fs, device)
                if r is not None:
                    regions[3].append(r)
    # Synergy of a region: its gain minus the best region over a strict subset of its features.
    best_of = {(r["group"], tuple(sorted(r["features"]))): r["gain"] for k in regions for r in regions[k]}
    for k in (2, 3):
        for r in regions[k]:
            fs = r["features"]
            subs = [tuple(sorted(c)) for m in range(1, k) for c in itertools.combinations(fs, m)]
            r["best_sub_gain"] = max([best_of.get((r["group"], c), 0.0) for c in subs] + [0.0])
            r["synergy"] = r["gain"] - r["best_sub_gain"]
    box_of = {(r["group"], r["feature"]): r for r in singles}

    # Round 2: refine the best roots (boxes and regions) with conditions on any agent (other agents, negations).
    candidates = [r for r in singles + pairs + triples + regions[1] + regions[2] + regions[3]
                  if r["val_lift"] >= ROOT_MIN_LIFT and r["val_events"] >= ROOT_MIN_EVENTS]
    roots = sorted(candidates, key=lambda r: -r["gain"])[:REFINE_ROOTS]
    bank = literal_bank(train, val)
    # The single-feature regions join the bank as literals (several ranges of one feature, and their negation).
    descs, M_tr, M_va = bank
    region_cols = regions[1]
    bank = (descs + [{"group": r["group"], "body": f"{r['features'][0]} in {r['text']}"} for r in region_cols],
            torch.cat([M_tr] + [r["_member"][0][:, None] for r in region_cols], dim=1),
            torch.cat([M_va] + [r["_member"][1][:, None] for r in region_cols], dim=1))
    print(f"[Refine] {len(bank[0])} literals (+ their negations), {len(roots)} roots")
    refined = [{"root": r, "steps": refine(r, train, val, bank, device)} for r in tqdm(roots, desc="Refining")]

    nets = [neural_reference(train, val, source, device) for source in ("img", "all")]

    base_tr, base_va = float(train["y"].float().mean()), float(val["y"].float().mean())
    print(f"\n[Brake factors] {args.head}: train {len(train['y'])} rows, base {base_tr:.3f}; "
          f"val {len(val['y'])} rows, base {base_va:.3f}")
    print("\n== Best single interval per (group, feature), by train gain")
    print(f"{'group':6s} {'feature':18s} {'cat':9s} {'(a, b]':>20s} {'gain':>7s} {'p_tr':>5s} {'p_val':>5s} "
          f"{'lift':>5s} {'recall':>6s} {'events':>6s}")
    for r in sorted(singles, key=lambda r: -r["gain"]):
        print(f"{r['group']:6s} {r['feature']:18s} {r['category']:9s} "
              f"{'(' + _fmt(r['a']) + ', ' + _fmt(r['b']) + ']':>20s} {r['gain']:7.1f} {r['train_precision']:5.2f} "
              f"{r['val_precision']:5.2f} {r['val_lift']:5.2f} {r['val_recall']:6.2f} {r['val_events']:6d}")
    print("\n== Factor summary: best val lift (with >= 20 val events) and best train gain per category")
    summary = defaultdict(lambda: {"gain": 0.0, "lift": 0.0, "feature": None})
    for r in singles:
        key = (r["group"], r["feature"].split(":")[0], r["category"])
        s = summary[key]
        if r["gain"] > s["gain"]:
            s["gain"], s["feature"] = r["gain"], r["feature"]
        if r["val_events"] >= 20:
            s["lift"] = max(s["lift"], r["val_lift"])
    for (g, src, cat), s in sorted(summary.items()):
        print(f"  {g:6s} {src:4s} {cat:9s} gain {s['gain']:7.1f} ({s['feature']}), best val lift {s['lift']:.2f}")
    print("\n== Pairs on the same agent, by synergy (pair gain - better single gain on the same bins)")
    for r in sorted(pairs, key=lambda r: -r["synergy"])[:25]:
        print(f"  {r['group']:5s} {r['f1']} in ({_fmt(r['a1'])}, {_fmt(r['b1'])}] & {r['f2']} in "
              f"({_fmt(r['a2'])}, {_fmt(r['b2'])}]: gain {r['gain']:.1f} (singles {r['single_gain_f1']:.1f} / "
              f"{r['single_gain_f2']:.1f}, synergy {r['synergy']:+.1f}), p_tr {r['train_precision']:.2f}, "
              f"p_val {r['val_precision']:.2f}, lift {r['val_lift']:.2f}, events {r['val_events']}")
    print("\n== Triples position x size x speed on the same agent, by synergy (triple gain - best box with a "
          "feature over its whole range, i.e. the best pair / single on the same bins)")
    for r in sorted(triples, key=lambda r: -r["synergy"])[:20]:
        text = " & ".join(f"{f} in ({_fmt(a)}, {_fmt(b)}]" for f, a, b in r["conditions"])
        print(f"  {r['group']:5s} {text}: gain {r['gain']:.1f} (best sub {r['best_sub_gain']:.1f}, synergy "
              f"{r['synergy']:+.1f}), p_tr {r['train_precision']:.2f}, p_val {r['val_precision']:.2f}, "
              f"lift {r['val_lift']:.2f}, events {r['val_events']}, recall {r['val_recall']:.2f}")
    print("\n== Single features: one interval vs a region (union of ranges), val lift and train gain")
    for r in sorted(regions[1], key=lambda r: -r["gain"]):
        box = box_of.get((r["group"], r["features"][0]))
        bx = f"box gain {box['gain']:7.1f} lift {box['val_lift']:.2f}" if box else "box -"
        print(f"  {r['group']:6s} {r['features'][0]:18s} {bx} | region gain {r['gain']:7.1f} lift {r['val_lift']:.2f} "
              f"recall {r['val_recall']:.2f} ({r['cells']}/{r['of_cells']} cells): {r['text']}")
    for k in (2, 3):
        print(f"\n== Regions over {k} features of the same agent, by synergy (gain - best region over fewer features)")
        for r in sorted(regions[k], key=lambda r: -r["synergy"])[:20]:
            print(f"  {r['group']:5s} {' x '.join(r['features'])}: gain {r['gain']:.1f} (sub {r['best_sub_gain']:.1f}, "
                  f"synergy {r['synergy']:+.1f}), {r['cells']}/{r['of_cells']} cells, p_tr {r['train_precision']:.2f}, "
                  f"p_val {r['val_precision']:.2f}, lift {r['val_lift']:.2f}, events {r['val_events']}, "
                  f"recall {r['val_recall']:.2f}")
        top = sorted([r for r in regions[k] if r["val_events"] >= 30], key=lambda r: -r["val_lift"])[:5]
        print(f"   best val lift (>= 30 val events):")
        for r in top:
            print(f"    {r['group']:5s} {' x '.join(r['features'])}: p_val {r['val_precision']:.2f}, lift "
                  f"{r['val_lift']:.2f}, events {r['val_events']}, recall {r['val_recall']:.2f}, {r['cells']} cells")
    print("\n== Fit: train AP (can it fit?) and val AP (does it hold?); prior AP = base rate "
          f"{base_tr:.3f} / {base_va:.3f}")
    for k in (1, 2, 3):
        for src in ("img", "gt"):
            rs = [r for r in regions[k] if r["features"][0].startswith(src) and r["lookup_train_ap"] is not None]
            if not rs:
                continue
            r = max(rs, key=lambda r: r["lookup_train_ap"])
            rv = max(rs, key=lambda r: r["lookup_val_ap"] or 0)
            print(f"  cell lookup over {k} feature(s), {src}: best train AP {r['lookup_train_ap']:.3f} "
                  f"({r['group']} {' x '.join(r['features'])}, val {r['lookup_val_ap']:.3f}); best val AP "
                  f"{rv['lookup_val_ap']:.3f} ({rv['group']} {' x '.join(rv['features'])})")
    for n in nets:
        print(f"  SetNet ({n['source']} features, all agents): train AP {n['train_ap']:.3f}, val AP {n['val_ap']:.3f} "
              f"(best over steps {n['best_val_ap']:.3f})")
    print("\n== Refined rules: root, then each added condition (train / val precision, val events, val recall)")
    for item in refined:
        for k, st in enumerate(item["steps"]):
            print(f"  {'root ' if k == 0 else '   & '}{st['add']}: p_tr {st['train']['precision']:.2f}, "
                  f"p_val {st['val']['precision']:.2f}, events {st['val']['events']}, recall {st['val']['recall']:.2f}")
        print()
    path = out_dir / f"brake_factors_{tag}{'_capacity' if args.capacity else ''}.json"
    utils_data.save_json({"head": args.head, "stride": STRIDE, "window": WINDOW, "bins": Q, "pair_bins": Q2,
                          "min_pos": MIN_POS, "train_base": base_tr, "val_base": base_va,
                          "capacity": bool(args.capacity), "nets": nets,
                          "singles": singles, "pairs": pairs, "triples": triples,
                          "regions": {str(k): [{x: v for x, v in r.items() if x != "_member"} for r in regions[k]]
                                      for k in regions},
                          "refined": [{"root": {x: v for x, v in i["root"].items() if x != "_member"},
                                       "steps": i["steps"]} for i in refined]}, path)
    print(f"[Brake factors] saved {path}")


if __name__ == "__main__":
    main()
