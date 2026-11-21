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
Rows v2 (`ROWS_VERSION`): every continuous feature also at LAGS_S = 1, 2, 3, 5 s before the last seen frame
("img:scale@-2s"); the agents' labels, class-free ("lab:Brake", "lab:Red", also lagged: "lab:Red@-3s"); the
ego's labels ("ego:AV-Mov@-1s"); the growth rate over 1 s and 2 s (img:scale_1s / _2s: a small, far box
changes little in 0.5 s); the traffic light group "tl". The labels enter the literal bank (round 2, the rule
list) and the SetNet; the continuous features with their lags the intervals / one-feature regions; the
pairs / triples / multi-feature regions stay on the current frame (lag 0).
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

Rule list (`rule_list`): sequential covering over every literal found (intervals, boxes, regions, their
negations), no cap on the number of rules: phase 1 grows rules by FOIL gain (coverage first), phase 2 collects
the small precise rules on the rows left (Wilson lower bound of the precision); AP of the list (decision list
and unordered max) on train and val by number of rules and without the large / mid / small rules, against the
SetNet (can rules fit as well? do the many small rules add up?).

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
GROUPS = ("ego", "ahead", "veh", "ped", "tl")
VEH_CLASSES = set(step05.MOTOR_CLASSES) | {"Cyc"}
TL_GROUP_CLASSES = set(agent_roles.NON_GROUND_CLASSES)      # traffic lights (labels: their signal)
LAGS_S = (0, 1, 2, 3, 5)       # every feature and label also 1, 2, 3 and 5 s before the last seen frame
FPS = 10
ROWS_VERSION = "v2"            # v2: labels, lags, 1 s / 2 s growth rates, the traffic light group
CATEGORY = {
    "img:log_area": "size", "img:height": "size", "gt:area": "size",
    "img:bottom": "position", "img:offset": "position", "img:abs_offset": "position",
    "gt:fwd": "position", "gt:left": "position", "gt:abs_left": "position", "gt:dist": "position",
    "img:scale": "speed", "img:scale_1s": "speed", "img:scale_2s": "speed", "img:accel": "speed", "img:lateral": "speed", "img:toward_centre": "speed",
    "img:vertical": "speed", "gt:closing": "speed", "gt:lat_closing": "speed", "gt:speed": "speed",
    "gt:same_lane": "lane", "img:scene_scale": "ego speed", "gt:ego_speed": "ego speed",
}
IMG_AGENT = ("img:log_area", "img:height", "img:bottom", "img:offset", "img:abs_offset", "img:scale",
             "img:scale_1s", "img:scale_2s", "img:accel", "img:lateral", "img:toward_centre", "img:vertical")
EGO_CONT = ("img:scene_scale", "gt:ego_speed")


def lag_name(feature, s):
    """The name of a feature s seconds before the last seen frame ("" suffix for s = 0)."""
    return feature if s == 0 else f"{feature}@-{s}s"


def features_of(data, group, kind="cont"):
    """Feature names of a group in the rows: kind "cont" (continuous, all lags), "label" (0/1, all lags),
    "cont0" (continuous, no lag: the pairs / triples / multi-feature regions)."""
    f = data["features"]
    if group == "ego":
        return {"cont": f["ego_cont"], "label": f["ego_label"], "cont0": list(EGO_CONT)}[kind]
    return {"cont": f["agent_cont"], "label": f["agent_label"], "cont0": list(IMG_AGENT + GT_AGENT)}[kind]
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
    log_area = torch.log(((x2 - x1) * W).clamp(min=1.0) * ((y2 - y1) * H).clamp(min=1.0))

    def rate(k):                     # log area growth per frame over k frames (a small, far box needs long k)
        r = torch.full_like(log_area, float("nan"))
        if len(log_area) > k:
            pair = ok[k:] & ok[:-k]
            r[k:] = torch.where(pair, (log_area[k:] - log_area[:-k]) / k, float("nan"))
        return r
    out = {"img:log_area": log_area, "img:scale_1s": rate(FPS), "img:scale_2s": rate(2 * FPS),
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


def collect(cfg, split, dataset, head_name, port=False, stride=STRIDE, use_gt=True):
    """Rows of one split: {"y": (R,) bool, "event": (R,) long, "features": {kind: [names]},
    "ego": {feature: (R,)}, group: {"row": (n,) long, feature: (n,)} for ahead / veh / ped / tl,
    "video": (R,) long index into "video_ids", "t": (R,) long frame, "known": (R,) bool}.
    port (the main model's rows, `step06_boost`): every frame t >= 1 (each stride-th) where the head was off
    in the min_off_frames frames before t (step05 `at_risk`; the ego need not move), up to the video end
    ("known": the window t .. t + WINDOW - 1 fits in the video or the onset was seen). use_gt False: no SUMO
    world (the gt:* features stay NaN; what the model has on real video).
    Every continuous feature and every label (class-free label of the agent, e.g. "lab:Brake"; the ego's
    "ego:AV-Mov") at the last seen frame and LAGS_S seconds before it (NaN before the video start; a label of
    an agent not in view then is 0). Per video as tensors."""
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    names = [step05._atom_name(sig) for sig in unique_atoms]
    i_head, i_mov = names.index(head_name), names.index("av:AV-Mov")
    A = len(unique_atoms)
    parts = sorted({n.split(":", 1)[1] for n in names if ":" in n and not n.startswith("av:")})
    onehot = torch.zeros((A, len(parts)))
    for a, n in enumerate(names):
        if ":" in n and not n.startswith("av:"):
            onehot[a, parts.index(n.split(":", 1)[1])] = 1.0
    av_atoms = [a for a, n in enumerate(names) if n.startswith("av:")]
    cont_base = list(IMG_AGENT + GT_AGENT)
    feats = {"agent_cont": [lag_name(f, s) for s in LAGS_S for f in cont_base],
             "agent_label": [lag_name(f"lab:{p}", s) for s in LAGS_S for p in parts],
             "ego_cont": [lag_name(f, s) for s in LAGS_S for f in EGO_CONT],
             "ego_label": [lag_name(f"ego:{names[a].split(':', 1)[1]}", s) for s in LAGS_S for a in av_atoms]}
    atoms = utils_data.load_json(cfg[f"{split}_atom_file"])
    boxes = torch.load(cfg[f"{split}_box_file"], map_location="cpu")
    y_all, ev_all, ego_chunks = [], [], []
    chunks = {g: {"row": [], "cont": [], "label": []} for g in GROUPS[1:]}
    num_rows, gt_videos = 0, 0
    video_of, t_of, known_of, video_ids = [], [], [], sorted(atoms)
    ego_index, head_track = {}, {}
    nan = float("nan")
    for vi, (vid, video_atoms) in enumerate(tqdm(sorted(atoms.items()), desc=f"Rows ({split})")):
        X = step05.build_video_tensor(video_atoms, A)
        T = X.shape[0]
        agent_ids = step05.video_agent_ids(video_atoms)
        ego = agent_ids.get(None)
        if ego is None or T <= (1 if port else WINDOW + motion.K + 2):
            continue
        vb, classes, has_box = step05._agent_boxes(boxes, vid, video_atoms, X)
        X = X.cpu()
        N = X.shape[1]
        moving, braking = X[:, ego, i_mov], X[:, ego, i_head]
        ego_index[vid], head_track[vid] = ego, braking.clone()
        onset = step05.atom_onsets(X[:, ego:ego + 1, i_head:i_head + 1], cfg["min_off_frames"])[:, 0, 0]
        img = image_features(vb, has_box)
        partner = agent_roles.role_partners(vb.clone(), classes, ego, ("ahead",),
                                            agent_roles.ego_lane(X, unique_atoms))[:, ego, 0]
        scene = ego_motion.signals(vb, has_box, classes, ego, X, unique_atoms)["scene_scale"]
        world_file = dataset / "meta" / "world" / f"{vid}.json.gz"
        gt = {k: torch.full((T, N), nan) for k in GT_AGENT}
        ego_speed = torch.full((T,), nan)
        if use_gt and world_file.exists():
            with gzip.open(world_file, "rt") as f:
                world = json.load(f)
            gt_videos += 1
            uid_of = {n: uid for uid, n in agent_ids.items() if uid is not None}
            for t in range(1, min(T, len(world) + 1)):
                w = t - 1                                         # world index of atom frame t
                if "ego" not in world[w]["veh"]:
                    continue
                ego_speed[t] = world[w]["veh"]["ego"]["speed"]
                for n in has_box[t].nonzero().flatten().tolist():
                    if n not in uid_of:
                        continue
                    g = gt_features(world, w, _agent_of(uid_of[n], vid, world[w]))
                    if g:
                        for k in GT_AGENT:
                            gt[k][t, n] = g[k]
        cont = torch.stack([img[f] for f in IMG_AGENT] + [gt[f] for f in GT_AGENT], dim=2)    # (T, N, Fc)
        labels = (X.float() @ onehot) > 0                                                     # (T, N, P)
        ego_cont = torch.stack([scene.float(), ego_speed], dim=1)                             # (T, 2)
        ego_lab = X[:, ego][:, av_atoms].float()                                              # (T, nav)
        os_, ys, evs = [], [], []
        m_off = max(int(cfg["min_off_frames"]), 1)
        frames_t = range(1, T, stride) if port else range(motion.K + 2, T - WINDOW + 1, stride)
        for t in frames_t:
            o = t - 1                                             # the last seen frame
            ok = (not bool(braking[max(t - m_off, 0):t].any())) if port else (bool(moving[o]) and not bool(braking[o]))
            if ok:
                hit = onset[t:t + WINDOW]
                os_.append(o)
                ys.append(bool(hit.any()))
                evs.append(vi * 100000 + t + int(hit.nonzero()[0]) if bool(hit.any()) else -1)
                video_of.append(vi)
                t_of.append(t)
                known_of.append(t + WINDOW <= T or bool(hit.any()))
        if not os_:
            continue
        o_t = torch.tensor(os_)
        R0 = num_rows
        num_rows += len(os_)
        y_all += ys
        ev_all += evs

        def lagged(Fr, idx_rows, idx_agents=None):
            """(n, F * len(LAGS_S)) values of Fr ((T, F) or (T, N, F)) at each lag, NaN before frame 0."""
            outs = []
            for sec in LAGS_S:
                frame = idx_rows - sec * FPS
                valid = frame >= 0
                fc = frame.clamp(min=0)
                v = Fr[fc] if idx_agents is None else Fr[fc, idx_agents]
                outs.append(torch.where(valid[:, None], v.float(), torch.full_like(v.float(), nan)))
            return torch.cat(outs, dim=1)
        ego_chunks.append(torch.cat([lagged(ego_cont, o_t), lagged(ego_lab, o_t)], dim=1))
        ri, n = has_box[o_t].nonzero(as_tuple=True)
        if len(ri) == 0:
            continue
        frames = o_t[ri]
        c_obs = lagged(cont, frames, n)
        l_obs = lagged(labels, frames, n).nan_to_num(0.0)
        cls = [classes[k] for k in n.tolist()]
        member = {"veh": torch.tensor([c in VEH_CLASSES for c in cls]),
                  "ped": torch.tensor([c == "Ped" for c in cls]),
                  "tl": torch.tensor([c in TL_GROUP_CLASSES for c in cls]),
                  "ahead": n == partner[frames]}
        for g, m in member.items():
            if bool(m.any()):
                chunks[g]["row"].append(ri[m] + R0)
                chunks[g]["cont"].append(c_obs[m])
                chunks[g]["label"].append(l_obs[m])
    out = {"y": torch.tensor(y_all), "event": torch.tensor(ev_all, dtype=torch.long), "features": feats,
           "video": torch.tensor(video_of, dtype=torch.long), "t": torch.tensor(t_of, dtype=torch.long),
           "known": torch.tensor(known_of, dtype=torch.bool), "video_ids": video_ids,
           "ego_index": ego_index, "head_track": head_track}
    E = torch.cat(ego_chunks)
    names_ego = feats["ego_cont"] + feats["ego_label"]
    out["ego"] = {k: E[:, i].contiguous() for i, k in enumerate(names_ego)}
    for g, d in chunks.items():
        rows = torch.cat(d["row"]) if d["row"] else torch.zeros(0, dtype=torch.long)
        C = torch.cat(d["cont"]) if d["cont"] else torch.zeros((0, len(feats["agent_cont"])))
        L = torch.cat(d["label"]) if d["label"] else torch.zeros((0, len(feats["agent_label"])))
        out[g] = {"row": rows}
        out[g].update({k: C[:, i].contiguous() for i, k in enumerate(feats["agent_cont"])})
        out[g].update({k: L[:, i].contiguous() for i, k in enumerate(feats["agent_label"])})
    print(f"[Rows] {split}: {num_rows} rows, {int(out['y'].sum())} positive "
          f"({len(set(e for e in ev_all if e >= 0))} onsets), SUMO world for {gt_videos} videos; "
          + ", ".join(f"{g} {len(out[g]['row'])} obs" for g in GROUPS[1:]) + f"; {len(parts)} agent labels, "
          f"{len(av_atoms)} ego labels, lags {LAGS_S} s")
    return out


def subset_rows(data, keep):
    """The rows of data (`collect`) with keep (R,) bool, the agent observations re-indexed."""
    idx = torch.nonzero(keep).flatten()
    remap = torch.full((len(keep),), -1, dtype=torch.long)
    remap[idx] = torch.arange(len(idx))
    out = {k: v for k, v in data.items() if not torch.is_tensor(v) and k not in GROUPS}
    out.update({k: data[k][keep] for k in ("y", "event", "video", "t", "known")})
    out["ego"] = {f: v[keep] for f, v in data["ego"].items()}
    for g in GROUPS[1:]:
        m = keep[data[g]["row"]]
        out[g] = {f: (remap[v[m]] if f == "row" else v[m]) for f, v in data[g].items()}
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
    if len(v) < 1000:
        return None
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
    res = {"group": group, "feature": feature, "category": CATEGORY.get(feature.split("@")[0], "?"), "a": lo, "b": hi,
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
            "speed": ("img:scale", "img:scale_1s", "img:scale_2s", "img:accel", "img:lateral", "img:toward_centre",
                      "img:vertical")},
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
            "lookup_train_ap": lookup["train"], "lookup_val_ap": lookup["val"], "_member": (member_tr, member_va),
            "_edges": edges, "_in_region": in_region}


def region_spec(r):
    """The spec of a region literal (`apply_spec`): its group, features, cell edges and cells."""
    return ("region", {"group": r["group"], "features": list(r["features"]), "_edges": r["_edges"],
                       "_in_region": r["_in_region"]})


def apply_spec(data, spec):
    """(R,) bool membership of a literal spec ("box", group, [(f, lo, hi)]) or ("region", r) on any rows."""
    if spec[0] == "box":
        return member_of(data, spec[1], spec[2])
    return region_member(data, spec[1])


def region_member(data, r):
    """(R,) bool: the rows of data (any split) where some agent of r's group lies in the region r."""
    rows, cell = _cells(data, r["group"], r["features"], r["_edges"])
    member = torch.zeros(len(data["y"]), dtype=torch.bool)
    member[rows[r["_in_region"][cell]]] = True
    return member

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


def literal_bank(train, val, test):
    """Literals "some agent of a group has f in (a, b]" for every group, feature and interval of Q_BANK
    train-quantile bins: [description], (R_train, n), (R_val, n), (R_test, n) bool. The negation of a literal
    ("no agent of the group has f in (a, b]") is its complement, taken in `refine`."""
    descs, cols_tr, cols_va, cols_te = [], [], [], []
    for group in GROUPS:
        # The labels (0/1): "some agent of the group holds label X (s seconds ago)", one literal each.
        for f in features_of(train, group, "label"):
            per_split = [member_of(data, group, [(f, 0.5, math.inf)]) for data in (train, val, test)]
            if int(per_split[0].sum()) < 10:
                continue
            descs.append({"group": group, "feature": f, "body": f"{f}", "spec": ("box", group, [(f, 0.5, math.inf)])})
            cols_tr.append(per_split[0][:, None])
            cols_va.append(per_split[1][:, None])
            cols_te.append(per_split[2][:, None])
        for f in features_of(train, group, "cont"):
            rows, v = _obs(train, group, f)
            if len(v) < 1000:
                continue
            edges = _edges(v, Q_BANK)
            B = len(edges) + 1
            i, j = _interval_index(B)
            per_split = []
            for data in (train, val, test):
                r, x = _obs(data, group, f)
                P = _presence(r, torch.bucketize(x, edges), len(data["y"]), B, "cpu")
                C = torch.cat([torch.zeros((len(data["y"]), 1)), P.cumsum(dim=1)], dim=1)
                per_split.append((C[:, j + 1] - C[:, i]) > 0)
            for k in range(len(i)):
                if i[k] == 0 and j[k] == B - 1:
                    continue                                    # the whole range: "the agent exists"
                lo = float(edges[i[k] - 1]) if i[k] > 0 else -math.inf
                hi = float(edges[j[k]]) if j[k] < B - 1 else math.inf
                descs.append({"group": group, "feature": f, "a": lo, "b": hi, "spec": ("box", group, [(f, lo, hi)])})
            keep = ~((i == 0) & (j == B - 1))
            cols_tr.append(per_split[0][:, keep])
            cols_va.append(per_split[1][:, keep])
            cols_te.append(per_split[2][:, keep])
    return descs, torch.cat(cols_tr, dim=1), torch.cat(cols_va, dim=1), torch.cat(cols_te, dim=1)


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
    descs, M_tr, M_va = bank[:3]
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



# -------------- Rule list by sequential covering --------------
LIST_MAX_RULES = 100000    # no cap: the FOIL phase runs until no rule keeps LIST_MIN_HITS positive rows
LIST_DEPTH = 3             # literals per rule (capacity mode: 5)
LIST_MIN_HITS = 30         # positive rows a FOIL rule keeps on the rows still uncovered (capacity mode: 5)
# Phase 2, small precise rules on the rows the FOIL rules leave: each grows by the Wilson lower bound of its
# precision (z = SMALL_Z) instead of the FOIL gain (which prefers coverage), and is kept if that bound is
# >= SMALL_LB_LIFT x the train base rate with >= SMALL_MIN_HITS positive rows in >= SMALL_MIN_EVENTS onsets.
# No cap: until no rule passes. "Many small rules add up" (user, 2026-11-21).
SMALL_Z = 1.96
SMALL_LB_LIFT = 1.5        # also the bar of a phase-1 rule: without it the uncapped FOIL phase covers every
                           # positive with ever broader rules (planted test: precision 0.48 at the end, 98% covered)
                           # and leaves nothing to phase 2
SMALL_MIN_HITS = 5
SMALL_MIN_EVENTS = 3
SMALL_MAX_FAILS = 50       # phase 2 stops after this many rules in a row failed SMALL_MIN_EVENTS
RECALL_GROUPS = (("large", 0.05, 2.0), ("mid", 0.01, 0.05), ("small", 0.0, 0.01))
SHRINK_ROWS = 50           # list_shrunk: a rule's precision shrunk to the base rate with this many rows
LOGIT_L2 = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)   # 0 / 1e-5 since the test run (smaller L2 was better)
LOGIT_STEPS = 500
LIST_CURVE = (1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000)


def full_bank(train, val, test, bank, singles, pairs, triples, regions):
    """Every literal found so far as membership columns (train / val / test): the interval bank (and
    single-feature regions) of round 2, plus each single / pair / triple box and each 1 / 2 / 3-feature region
    on the same agent. All of them are chosen on train only."""
    descs, M_tr, M_va, M_te = bank
    descs = list(descs)
    extra = ([], [], [])

    def add(text, group, spec, *cols):
        descs.append({"group": group, "body": text, "spec": spec})
        for e, c in zip(extra, cols):
            e.append(c[:, None])

    def add_box(text, group, cond):
        add(text, group, ("box", group, cond), *[member_of(d, group, cond) for d in (train, val, test)])
    for r in singles:
        add_box(f"{r['feature']} in ({_fmt(r['a'])}, {_fmt(r['b'])}]", r["group"], [(r["feature"], r["a"], r["b"])])
    for r in pairs:
        cond = [(r["f1"], r["a1"], r["b1"]), (r["f2"], r["a2"], r["b2"])]
        add_box(" & same agent ".join(f"{f} in ({_fmt(a)}, {_fmt(b)}]" for f, a, b in cond), r["group"], cond)
    for r in triples:
        cond = [tuple(c) for c in r["conditions"]]
        add_box(" & same agent ".join(f"{f} in ({_fmt(a)}, {_fmt(b)}]" for f, a, b in cond), r["group"], cond)
    for k in (2, 3):
        for r in regions[k]:
            add(f"{' x '.join(r['features'])} in region ({r['cells']} cells)", r["group"], region_spec(r),
                *r["_member"], region_member(test, r))
    return (descs,) + tuple(torch.cat([M] + e, dim=1) for M, e in zip((M_tr, M_va, M_te), extra))


def _best_literal(S, y, M, p_now, min_hits, device, used):
    """The literal (or its negation: index >= n) of the largest FOIL gain hits x (log p - log p_now) on the
    rows S, with >= min_hits hits and p > p_now; (index, gain, hits, rows) or None."""
    s, ys = S.to(device).float(), (S & y).to(device).float()
    hits_pos = torch.cat([ys @ M[:, c:c + 2048].float() for c in range(0, M.shape[1], 2048)])
    rows_pos = torch.cat([s @ M[:, c:c + 2048].float() for c in range(0, M.shape[1], 2048)])
    hits = torch.cat([hits_pos, ys.sum() - hits_pos])
    rows = torch.cat([rows_pos, s.sum() - rows_pos])
    p = hits / rows.clamp(min=1)
    gain = hits * (torch.log(p.clamp(min=1e-12)) - math.log(max(p_now, 1e-12)))
    ok = (hits >= min_hits) & (p > p_now)
    if used:
        ok[torch.tensor(used, device=device)] = False
    gain = torch.where(ok, gain, torch.full_like(gain, -1.0))
    k = int(gain.argmax())
    if float(gain[k]) <= 0:
        return None
    return k, float(gain[k]), int(hits[k]), int(rows[k])


def _wilson_lb(hits, rows, z=SMALL_Z):
    n = rows.clamp(min=1)
    p = hits / n
    return (p + z * z / (2 * n) - z * torch.sqrt((p * (1 - p) / n + z * z / (4 * n * n)).clamp(min=0))) / (1 + z * z / n)


def _best_literal_lb(S, y, M, lb_now, min_hits, device, used):
    """The literal (or negation) of the largest Wilson lower bound of the precision on the rows S."""
    s, ys = S.to(device).float(), (S & y).to(device).float()
    hits_pos = torch.cat([ys @ M[:, c:c + 2048].float() for c in range(0, M.shape[1], 2048)])
    rows_pos = torch.cat([s @ M[:, c:c + 2048].float() for c in range(0, M.shape[1], 2048)])
    hits = torch.cat([hits_pos, ys.sum() - hits_pos])
    rows = torch.cat([rows_pos, s.sum() - rows_pos])
    lb = _wilson_lb(hits, rows)
    ok = hits >= min_hits
    if used:
        ok[torch.tensor(used, device=device)] = False
    lb = torch.where(ok, lb, torch.full_like(lb, -1.0))
    k = int(lb.argmax())
    if float(lb[k]) <= lb_now:
        return None
    return k, float(lb[k]), int(hits[k]), int(rows[k])


def rule_list(train, val, test, bank, device):
    """Rule list without a cap on the number of rules, in three phases on the train rows no earlier rule covers
    (a rule's rows are removed once it is learned):
      1 + 2, rule by rule: a FOIL rule (grown by the FOIL gain, up to LIST_DEPTH literals, >= LIST_MIN_HITS
        positive rows) if the Wilson lower bound of its precision is >= SMALL_LB_LIFT x base; else a small
        precise rule (grown by that lower bound, >= SMALL_MIN_HITS hits) if it clears the same bar, kept with
        >= SMALL_MIN_EVENTS onsets (its rows removed either way); stop when neither clears the bar;
      3: broad FOIL rules without the bar (low precision, still above the rows left), until none keeps
        LIST_MIN_HITS positive rows.
    Scores: decision list (the first rule that fires gives its train precision; none: the onset rate of the
    rows left) and unordered (the highest precision of the rules that fire). Train / val AP by number of rules,
    and by recall group (large / mid / small rules): their count, coverage and the AP without them."""
    descs, M_tr, M_va, M_te = bank
    n = len(descs)
    M_dev = M_tr.to(device)
    y_tr, y_va, y_te = train["y"], val["y"], test["y"]
    base = float(y_tr.float().mean())
    total_pos = max(int(y_tr.sum()), 1)
    left = torch.ones(len(y_tr), dtype=torch.bool)
    rules = []

    def add_rule(lits, S, phase):
        hits, rows = int((S & y_tr).sum()), int(S.sum())
        events = int(train["event"][S & y_tr].unique().numel())
        rules.append({"literals": lits, "phase": phase, "precision": hits / max(rows, 1), "hits": hits,
                      "rows": rows, "events": events, "train_recall": hits / total_pos,
                      "text": " & ".join(literal_text(descs[k % n], k >= n) for k in lits)})

    def grow_foil(min_hits):
        S = left.clone()
        p_now = float((S & y_tr).sum()) / max(int(S.sum()), 1)
        lits, used = [], []
        for _d in range(LIST_DEPTH):
            best = _best_literal(S, y_tr, M_dev, p_now, min_hits, device, used)
            if best is None:
                break
            k, _gain, hits, rows = best
            S = S & (M_tr[:, k % n] if k < n else ~M_tr[:, k % n])
            p_now = hits / max(rows, 1)
            lits.append(k)
            used += [k % n, k % n + n]
        return lits, S

    def grow_lb():
        S = left.clone()
        lb_now, lits, used = -1.0, [], []
        for _d in range(LIST_DEPTH + 1):
            best = _best_literal_lb(S, y_tr, M_dev, lb_now, SMALL_MIN_HITS, device, used)
            if best is None:
                break
            k, lb_now, _hits, _rows = best
            S = S & (M_tr[:, k % n] if k < n else ~M_tr[:, k % n])
            lits.append(k)
            used += [k % n, k % n + n]
        return lits, S, lb_now

    def lb_of(S):
        return float(_wilson_lb(torch.tensor(float((S & y_tr).sum())), torch.tensor(float(S.sum()))))
    bar_lb = SMALL_LB_LIFT * base
    fails = 0
    # Phases 1 + 2, rule by rule: a FOIL rule if it is clearly above base, else the most precise small rule.
    with tqdm(desc="Rule list (FOIL / small precise)") as bar:
        while int((left & y_tr).sum()) >= SMALL_MIN_HITS and fails < SMALL_MAX_FAILS and len(rules) < LIST_MAX_RULES:
            lits, S = grow_foil(LIST_MIN_HITS) if int((left & y_tr).sum()) >= LIST_MIN_HITS else ([], None)
            if lits and lb_of(S) >= bar_lb:
                add_rule(lits, S, "foil")
                left &= ~S
                bar.update(1)
                continue
            lits, S, lb = grow_lb()
            if not lits or lb < bar_lb:
                break
            events = int(train["event"][S & y_tr].unique().numel())
            left &= ~S                                           # its rows are removed either way
            if events < SMALL_MIN_EVENTS:
                fails += 1
                continue
            fails = 0
            add_rule(lits, S, "small")
            bar.update(1)
    # Phase 3: the broad rules (FOIL, no bar) on the rows left, until none keeps LIST_MIN_HITS positive rows:
    # low precision, but still above the rows left, so they rank them above the default.
    with tqdm(desc="Rule list (broad)") as bar:
        while int((left & y_tr).sum()) >= LIST_MIN_HITS and len(rules) < LIST_MAX_RULES:
            lits, S = grow_foil(LIST_MIN_HITS)
            if not lits:
                break
            add_rule(lits, S, "broad")
            left &= ~S
            bar.update(1)

    def fires(M, rule):
        f = torch.ones(M.shape[0], dtype=torch.bool)
        for k in rule["literals"]:
            f &= M[:, k % n] if k < n else ~M[:, k % n]
        return f
    F_tr = [fires(M_tr, r) for r in rules]
    F_va = [fires(M_va, r) for r in rules]
    F_te = [fires(M_te, r) for r in rules]
    for r, f, g in zip(rules, F_va, F_te):
        hits = int((f & y_va).sum())
        r["val_precision"] = hits / max(int(f.sum()), 1)
        r["val_hits"], r["val_rows"] = hits, int(f.sum())
        r["test_precision"] = int((g & y_te).sum()) / max(int(g.sum()), 1)
        r["test_rows"] = int(g.sum())

    for r in rules:
        r["lb"] = float(_wilson_lb(torch.tensor(float(r["hits"])), torch.tensor(float(r["rows"]))))
        r["shrunk"] = (r["hits"] + SHRINK_ROWS * base) / (r["rows"] + SHRINK_ROWS)
    X_tr_all = torch.stack(F_tr, dim=1).float().to(device) if rules else None
    X_va_all = torch.stack(F_va, dim=1).float().to(device) if rules else None
    X_te_all = torch.stack(F_te, dim=1).float().to(device) if rules else None

    def logistic(idx, l2):
        """Train / val / test scores of an L2 logistic regression on the firings of the rules idx (fit on train)."""
        X_tr, X_va, X_te = X_tr_all[:, idx], X_va_all[:, idx], X_te_all[:, idx]
        w = torch.zeros(len(idx), device=device, requires_grad=True)
        b = torch.tensor([math.log(base / (1 - base))], device=device, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        yt = y_tr.float().to(device)
        for _ in range(LOGIT_STEPS):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X_tr @ w + b, yt) + l2 * (w * w).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return (X_tr @ w + b).cpu(), (X_va @ w + b).cpu(), (X_te @ w + b).cpu()

    def evaluate(keep, logit=True):
        """AP of the rules with keep[i] (in list order), on train and val, per scorer:
          list / max:        decision list / best firing rule, by the raw train precision of each rule;
          list_lb / max_lb:  the same by the Wilson lower bound of the precision (small rules not overrated);
          list_shrunk:       by the precision shrunk to the base rate with SHRINK_ROWS rows;
          logit_<l2>:        L2 logistic regression on the rule firings (evidence of several rules adds up)."""
        idx = [i for i in range(len(rules)) if keep[i]]
        left_m = torch.ones(len(y_tr), dtype=torch.bool)
        for i in idx:
            left_m &= ~F_tr[i]
        default = float(y_tr[left_m].float().mean()) if bool(left_m.any()) else 0.0
        out = {}
        for name, F, y in (("train", F_tr, y_tr), ("val", F_va, y_va), ("test", F_te, y_te)):
            for key, value in (("", "precision"), ("_lb", "lb"), ("_shrunk", "shrunk")):
                first = torch.full((len(y),), -1.0)
                best = torch.full((len(y),), -1.0)
                for i in idx:
                    p = torch.tensor(rules[i][value])
                    first = torch.where((first < 0) & F[i], p, first)
                    best = torch.where(F[i], torch.clamp(best, min=p), best)
                covered = first >= 0
                d = torch.tensor(default if key == "" else min(default, base))
                out[f"{name}_ap_list{key}"] = average_precision(torch.where(covered, first, d), y)
                if key != "_shrunk":
                    out[f"{name}_ap_max{key}"] = average_precision(torch.where(covered, best, d), y)
                if key == "":
                    out[f"{name}_coverage"] = float((covered & y).sum()) / max(int(y.sum()), 1)
        if logit and idx:
            for l2 in LOGIT_L2:
                s_tr, s_va, s_te = logistic(idx, l2)
                out[f"train_ap_logit{l2:g}"] = average_precision(s_tr, y_tr)
                out[f"val_ap_logit{l2:g}"] = average_precision(s_va, y_va)
                out[f"test_ap_logit{l2:g}"] = average_precision(s_te, y_te)
        return out
    curve = []
    for m in sorted(set([c for c in LIST_CURVE if c <= len(rules)] + [len(rules)])):
        curve.append({"rules": m, **evaluate([i < m for i in range(len(rules))])})
    groups = {}
    full = evaluate([True] * len(rules))
    sets = [(name, [lo <= r["train_recall"] < hi for r in rules]) for name, lo, hi in RECALL_GROUPS]
    sets += [(f"phase {ph}", [r["phase"] == ph for r in rules]) for ph in ("foil", "small", "broad")]
    for name, member in sets:
        g = [r for r, m in zip(rules, member) if m]
        cov_tr = torch.zeros(len(y_tr), dtype=torch.bool)
        cov_va = torch.zeros(len(y_va), dtype=torch.bool)
        for f_tr, f_va, m in zip(F_tr, F_va, member):
            if m:
                cov_tr |= f_tr
                cov_va |= f_va
        without = evaluate([not m for m in member]) if g else full
        groups[name] = {"rules": len(g), "phase_small": sum(r["phase"] == "small" for r in g),
                        "phase_broad": sum(r["phase"] == "broad" for r in g),
                        "train_coverage": float((cov_tr & y_tr).sum()) / total_pos,
                        "val_coverage": float((cov_va & y_va).sum()) / max(int(y_va.sum()), 1),
                        "train_precision": float((cov_tr & y_tr).sum()) / max(int(cov_tr.sum()), 1),
                        "val_precision": float((cov_va & y_va).sum()) / max(int(cov_va.sum()), 1),
                        "without": without,
                        "ap_drop_val_list": full["val_ap_list"] - without["val_ap_list"],
                        "ap_drop_train_list": full["train_ap_list"] - without["train_ap_list"]}
    return {"rules": rules, "curve": curve, "full": full, "groups": groups,
            "phases": {ph: sum(r["phase"] == ph for r in rules) for ph in ("foil", "small", "broad")}}


# -------------- Boosted rules: each rule learned on all rows, on the residual of the rules so far --------------
# Sequential covering removes a rule's rows, so no later rule can rank the rows inside it (test 2026-11-21:
# 10 -> 200 rules raised the test AP only 0.506 -> 0.530; rule list + logit 0.609 vs SetNet 0.664). Boosting
# (as RuleFit / SLIPPER): round by round, on all train rows, the rule (<= BOOST_DEPTH literals or negations of
# the same bank) of the largest Newton gain G^2 / (H + lambda) on the current logistic score, G / H the sums of
# the gradient y - p and of p (1 - p) over its rows; its weight is BOOST_SHRINK x G / (H + lambda) (negative
# weights too: a rule can lower the score inside an earlier rule). The rounds stop by the val AP.
BOOST_ROUNDS = 2000
BOOST_SHRINK = 0.1
BOOST_LAMBDA = 5.0
BOOST_DEPTH = 3
BOOST_MIN_ROWS = 30         # train rows a rule holds on
BOOST_MIN_RISE = 1.01       # a further literal must raise the rule's gain by >= 1%
BOOST_EVAL = 50             # rounds between AP checks
BOOST_PATIENCE = 10         # checks without a better val AP before stopping


def _col_sums(V, M):
    """(k, L) = V (k, R) float @ M (R, L) bool, in column chunks."""
    return torch.cat([V @ M[:, c:c + 2048].float() for c in range(0, M.shape[1], 2048)], dim=1)


def boosted_rules(train, val, test, bank, device, use=None):
    """Boosted rule ensemble on the literal bank (the columns use only, if given). Returns the rules (with
    weights), the AP curve by round, the train / val / test AP at the best val round, and a refit: an L2
    logistic regression on the firings of the distinct rules up to that round (L2 chosen on val)."""
    descs, M_tr, M_va, M_te = bank
    if use is not None:
        idx = torch.tensor(use, dtype=torch.long)
        descs, M_tr, M_va, M_te = [descs[i] for i in use], M_tr[:, idx], M_va[:, idx], M_te[:, idx]
    n = len(descs)
    M_dev = M_tr.to(device)
    ys = {"train": train["y"], "val": val["y"], "test": test["y"]}
    y = ys["train"].float().to(device)
    base = float(y.mean())
    f0 = math.log(base / (1 - base))
    F = {"train": torch.full((len(y),), f0, device=device), "val": torch.full((len(ys["val"]),), f0),
         "test": torch.full((len(ys["test"]),), f0)}
    Ms = {"val": M_va, "test": M_te}

    def fires(M, lits):
        f = torch.ones(M.shape[0], dtype=torch.bool, device=M.device)
        for k in lits:
            f &= M[:, k % n] if k < n else ~M[:, k % n]
        return f
    rules, curve = [], []
    best = {"round": 0, "val_ap": -1.0}
    fails = 0
    for rnd in tqdm(range(BOOST_ROUNDS), desc="Boosted rules"):
        p = torch.sigmoid(F["train"])
        g, h = y - p, p * (1 - p)
        S = torch.ones(len(y), dtype=torch.bool, device=device)
        lits, used, now = [], [], 0.0
        for _d in range(BOOST_DEPTH):
            Sf = S.float()
            tot = torch.stack([(g * Sf).sum(), (h * Sf).sum(), Sf.sum()])
            pos = _col_sums(torch.stack([g * Sf, h * Sf, Sf]), M_dev)
            sums = torch.cat([pos, tot[:, None] - pos], dim=1)                 # literals, then negations
            gain = sums[0] ** 2 / (sums[1] + BOOST_LAMBDA)
            ok = sums[2] >= BOOST_MIN_ROWS
            if used:
                ok[torch.tensor(used, device=device)] = False
            gain = torch.where(ok, gain, torch.full_like(gain, -1.0))
            k = int(gain.argmax())
            if float(gain[k]) <= now * BOOST_MIN_RISE:
                break
            S = S & (M_dev[:, k % n] if k < n else ~M_dev[:, k % n])
            lits.append(k)
            used += [k % n, k % n + n]
            now = float(gain[k])
        if not lits:
            break
        Sf = S.float()
        G, H = float((g * Sf).sum()), float((h * Sf).sum())
        w = BOOST_SHRINK * G / (H + BOOST_LAMBDA)
        F["train"] += w * Sf
        for split in ("val", "test"):
            F[split] += w * fires(Ms[split], lits).float()
        hits, rows = int((S & ys["train"].to(device)).sum()), int(S.sum())
        rules.append({"literals": lits, "weight": w, "rows": rows, "hits": hits, "precision": hits / max(rows, 1),
                      "text": " & ".join(literal_text(descs[k % n], k >= n) for k in lits)})
        if (rnd + 1) % BOOST_EVAL == 0:
            point = {"rounds": rnd + 1, **{f"{s}_ap": average_precision(F[s].cpu(), ys[s]) for s in F}}
            curve.append(point)
            if point["val_ap"] > best["val_ap"]:
                best, fails = dict(point), 0
            else:
                fails += 1
                if fails >= BOOST_PATIENCE:
                    break
    if not curve or curve[-1]["rounds"] != len(rules):                        # the last partial check
        point = {"rounds": len(rules), **{f"{s}_ap": average_precision(F[s].cpu(), ys[s]) for s in F}}
        curve.append(point)
        if point["val_ap"] > best["val_ap"]:
            best = dict(point)
    kept = rules[:best["rounds"]]

    # Distinct rules up to the best round (a rule can be chosen in several rounds): summed weight; the refit.
    distinct = {}
    for r in kept:
        key = tuple(sorted(r["literals"]))
        if key in distinct:
            distinct[key]["weight"] += r["weight"]
            distinct[key]["rounds"] += 1
        else:
            distinct[key] = {**r, "rounds": 1}
    uniq = list(distinct.values())
    for r in uniq:
        for split in ("val", "test"):
            f = fires(Ms[split], r["literals"])
            r[f"{split}_rows"] = int(f.sum())
            r[f"{split}_precision"] = int((f & ys[split]).sum()) / max(int(f.sum()), 1)
    refit = {}
    if uniq:
        X = {"train": torch.stack([fires(M_dev, r["literals"]) for r in uniq], dim=1).float()}
        for split in ("val", "test"):
            X[split] = torch.stack([fires(Ms[split], r["literals"]) for r in uniq], dim=1).float().to(device)
        for l2 in LOGIT_L2:
            w = torch.tensor([r["weight"] for r in uniq], device=device, requires_grad=True)   # boosted init
            b = torch.tensor([f0], device=device, requires_grad=True)
            opt = torch.optim.Adam([w, b], lr=0.02)
            for _ in range(LOGIT_STEPS):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(X["train"] @ w + b, y) + l2 * (w * w).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                refit[f"{l2:g}"] = {f"{s}_ap": average_precision((X[s] @ w + b).cpu(), ys[s]) for s in X}
    uniq.sort(key=lambda r: -abs(r["weight"]) * r["rows"])
    return {"literals": n, "rounds": len(rules), "best": best, "curve": curve, "refit": refit, "f0": f0,
            "_descs": descs,
            "distinct_rules": len(uniq), "positive_rules": sum(r["weight"] > 0 for r in uniq),
            "rules": uniq}


# -------------- Neural reference on the same features --------------
NET_EPOCHS = 400           # full-batch steps
NET_HIDDEN = 64
NET_GROUPS = ("ahead", "veh", "ped", "tl")


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
    for g in NET_GROUPS:
        X = torch.stack([data[g][f] for f in feats_agent], dim=1)
        out[g] = (data[g]["row"].to(device), prep(X, *stats[g]).to(device))
    return out


def _stats_of(data, feats_agent, feats_ego):
    def ms(X):
        mu = torch.nanmean(X, dim=0)
        sd = torch.sqrt(torch.nanmean((X - mu) ** 2, dim=0)).nan_to_num(1.0).clamp(min=1e-6)
        return mu.nan_to_num(0.0), sd
    st = {"ego": ms(torch.stack([data["ego"][f] for f in feats_ego], dim=1))}
    for g in NET_GROUPS:
        st[g] = ms(torch.stack([data[g][f] for f in feats_agent], dim=1))
    return st


class SetNet(torch.nn.Module):
    """Per-agent MLP of each group (ahead, veh, ped), max-pooled over the row's agents, then a row MLP with the
    ego features: the neural upper reference of what these features can fit."""
    def __init__(self, f_agent, f_ego, h=NET_HIDDEN):
        super().__init__()
        mlp = lambda i: torch.nn.Sequential(torch.nn.Linear(i, h), torch.nn.GELU(), torch.nn.Linear(h, h),
                                            torch.nn.GELU())
        self.phi = torch.nn.ModuleDict({g: mlp(2 * f_agent) for g in NET_GROUPS})
        self.head = torch.nn.Sequential(torch.nn.Linear(2 * f_ego + len(NET_GROUPS) * (h + 1), h), torch.nn.GELU(),
                                        torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.Linear(h, 1))
        self.h = h

    def forward(self, x):
        parts = [x["ego"]]
        for g in NET_GROUPS:
            rows, feats = x[g]
            z = self.phi[g](feats)
            pooled = torch.full((x["R"], self.h), -1e4, device=z.device)
            pooled = pooled.scatter_reduce(0, rows[:, None].expand(-1, self.h), z, reduce="amax")
            has = torch.zeros(x["R"], device=z.device).index_fill_(0, rows, 1.0)
            parts += [torch.where(has[:, None] > 0, pooled, torch.zeros_like(pooled)), has[:, None]]
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


def neural_reference(train, val, test, source, device):
    """Train / val / test AP of a SetNet on the img (or img + gt) features, trained to fit train (no early
    stop); also its best val AP over the steps and the test AP at that step (early stop chosen on val)."""
    f = train["features"]
    keep = (lambda name: not name.startswith("gt:")) if source == "img" else (lambda name: True)
    feats_agent = [x for x in f["agent_cont"] + f["agent_label"] if keep(x)]
    feats_ego = [x for x in f["ego_cont"] + f["ego_label"] if keep(x)]
    st = _stats_of(train, feats_agent, feats_ego)
    x_tr = _net_inputs(train, feats_agent, feats_ego, st, device)
    x_va = _net_inputs(val, feats_agent, feats_ego, st, device)
    x_te = _net_inputs(test, feats_agent, feats_ego, st, device)
    y_tr = train["y"].float().to(device)
    torch.manual_seed(0)
    net = SetNet(len(feats_agent), len(feats_ego)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best_val, best_test, history = -1.0, None, []
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
                ap_te = average_precision(net(x_te), test["y"])
            if ap_va > best_val:
                best_val, best_test = ap_va, ap_te
            history.append({"step": step + 1, "loss": float(loss), "train_ap": ap_tr, "val_ap": ap_va,
                            "test_ap": ap_te})
    return {"source": source, "train_ap": history[-1]["train_ap"], "val_ap": history[-1]["val_ap"],
            "test_ap": history[-1]["test_ap"], "best_val_ap": best_val, "test_ap_at_best_val": best_test,
            "history": history}

def search_literals(train, val, device, sources=("img", "gt")):
    """The literals of round 1, all chosen on train (val only reported): the best interval per (group,
    feature), pairs and triples on the same agent, regions over 1 / 2 / 3 features. sources: the feature
    families of the pairs / triples / multi-feature regions ("img", "gt")."""
    singles = []
    for group in GROUPS:
        for f in tqdm(features_of(train, group, "cont"), desc=f"Intervals ({group})"):
            r = single_search(train, val, group, f, device)
            if r is not None:
                singles.append(r)
    pairs = []
    for group in ("ahead", "veh"):
        for feats in [f for src, f in (("img", IMG_AGENT), ("gt", GT_AGENT)) if src in sources]:
            for a in range(len(feats)):
                for b in range(a + 1, len(feats)):
                    r = pair_search(train, val, group, feats[a], feats[b], device)
                    if r is not None:
                        pairs.append(r)

    triples = []
    for group in ("ahead", "veh"):
        for source, cats in [(k, v) for k, v in TRIPLE_FEATURES.items() if k in sources]:
            combos = [(p, z, v) for p in cats["position"] for z in cats["size"] for v in cats["speed"]]
            for feats in tqdm(combos, desc=f"Triples ({group}, {source})"):
                r = triple_search(train, val, group, feats, device)
                if r is not None:
                    triples.append(r)

    # Regions: any union of cells, over 1 / 2 / 3 features of the same agent.
    regions = {1: [], 2: [], 3: []}
    for group in GROUPS:
        for f in tqdm(features_of(train, group, "cont"), desc=f"Regions 1 ({group})"):
            r = region_search(train, val, group, (f,), device)
            if r is not None:
                regions[1].append(r)
    for group in ("ahead", "veh"):
        for feats in [f for src, f in (("img", IMG_AGENT), ("gt", GT_AGENT)) if src in sources]:
            combos = [(feats[a], feats[b]) for a in range(len(feats)) for b in range(a + 1, len(feats))]
            for fs in tqdm(combos, desc=f"Regions 2 ({group})"):
                r = region_search(train, val, group, fs, device)
                if r is not None:
                    regions[2].append(r)
        for source, cats in [(k, v) for k, v in TRIPLE_FEATURES.items() if k in sources]:
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
    return singles, pairs, triples, regions


def make_banks(train, val, test, singles, pairs, triples, regions):
    """bank: the interval / label literals and the one-feature regions; big bank: plus every single, pair,
    triple and region (`full_bank`). Each a tuple (descs with "spec", M_train, M_val, M_test)."""
    bank = literal_bank(train, val, test)
    # The single-feature regions join the bank as literals (several ranges of one feature, and their negation).
    descs, M_tr, M_va, M_te = bank
    region_cols = regions[1]
    bank = (descs + [{"group": r["group"], "body": f"{r['features'][0]} in {r['text']}", "spec": region_spec(r)}
                     for r in region_cols],
            torch.cat([M_tr] + [r["_member"][0][:, None] for r in region_cols], dim=1),
            torch.cat([M_va] + [r["_member"][1][:, None] for r in region_cols], dim=1),
            torch.cat([M_te] + [region_member(test, r)[:, None] for r in region_cols], dim=1))
    return bank, full_bank(train, val, test, bank, singles, pairs, triples, regions)


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
    global LIST_DEPTH, LIST_MIN_HITS
    global MIN_POS, Q_REGION, REGION_MIN_NEW_HITS, REGION_MARGIN_LIFT, REGION_MAX_STEPS, REFINE_DEPTH, \
        REFINE_MIN_POS, REFINE_VAL_RISE, REFINE_MIN_VAL_EVENTS, ROOT_MIN_LIFT, ROOT_MIN_EVENTS
    if args.capacity:
        MIN_POS, REGION_MIN_NEW_HITS, REGION_MARGIN_LIFT = 10, 1, 0.0
        Q_REGION, REGION_MAX_STEPS = {1: 64, 2: 32, 3: 16}, 5000
        REFINE_DEPTH, REFINE_MIN_POS, REFINE_VAL_RISE, REFINE_MIN_VAL_EVENTS = 4, 10, 0.0, 0
        ROOT_MIN_LIFT, ROOT_MIN_EVENTS = 0.0, 0
        LIST_DEPTH, LIST_MIN_HITS = 5, 5
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id if getattr(args, "run_id", None) \
        else args.output_dir / "brake_factors"
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    out_dir = args.output_dir / "brake_factors"
    os.makedirs(out_dir, exist_ok=True)
    dataset = Path(args.dataset_path)
    tag = args.head.replace(":", "_")
    cache = out_dir / f"rows_{tag}_s{STRIDE}w{WINDOW}_{ROWS_VERSION}.pt"
    if os.path.exists(cache):
        train, val = torch.load(cache, map_location="cpu")
    else:
        train = collect(cfg, "train", dataset, args.head)
        val = collect(cfg, "val", dataset, args.head)
        torch.save((train, val), cache)
    # The test split: only for the final comparison (rule list + logit vs SetNet, each with its settings chosen
    # on val). Its own cache, so the train / val cache stays valid.
    test_cache = out_dir / f"rows_{tag}_s{STRIDE}w{WINDOW}_{ROWS_VERSION}_test.pt"
    if os.path.exists(test_cache):
        test = torch.load(test_cache, map_location="cpu")
    else:
        atoms_dir = args.output_dir / "step_dataset_output" / "atoms"
        test = collect({**cfg, "test_atom_file": atoms_dir / "test_atoms_compressed.json",
                        "test_box_file": atoms_dir / "test_boxes.pt"}, "test", dataset, args.head)
        torch.save(test, test_cache)

    singles, pairs, triples, regions = search_literals(train, val, device)
    box_of = {(r["group"], r["feature"]): r for r in singles}

    # Round 2: refine the best roots (boxes and regions) with conditions on any agent (other agents, negations).
    candidates = [r for r in singles + pairs + triples + regions[1] + regions[2] + regions[3]
                  if r["val_lift"] >= ROOT_MIN_LIFT and r["val_events"] >= ROOT_MIN_EVENTS]
    roots = sorted(candidates, key=lambda r: -r["gain"])[:REFINE_ROOTS]
    bank, big_bank = make_banks(train, val, test, singles, pairs, triples, regions)
    print(f"[Refine] {len(bank[0])} literals (+ their negations), {len(roots)} roots")
    refined = [{"root": r, "steps": refine(r, train, val, bank, device)} for r in tqdm(roots, desc="Refining")]

    # The rule list: can a set of such rules fit train as the SetNet does?
    print(f"[Rule list] {len(big_bank[0])} literals (+ their negations)")
    rlist = rule_list(train, val, test, big_bank, device)

    # Boosted rules on the same bank: all literals, and the image ones only (no SUMO ground truth: what the
    # main model has on real video; compare with SetNet img).
    img_cols = [i for i, d_ in enumerate(big_bank[0]) if "gt:" not in literal_text(d_)]
    boosted = {"all": boosted_rules(train, val, test, big_bank, device),
               "img": boosted_rules(train, val, test, big_bank, device, use=img_cols)}

    nets = [neural_reference(train, val, test, source, device) for source in ("img", "all")]

    # Final comparison on test: every model's settings chosen on val (rule scorer / L2; SetNet step).
    full = rlist["full"]
    scorer_keys = ["list", "max", "list_lb", "max_lb", "list_shrunk"] + [f"logit{l:g}" for l in LOGIT_L2]
    logit_keys = [f"logit{l:g}" for l in LOGIT_L2]
    pick = lambda keys: max(keys, key=lambda k: full.get(f"val_ap_{k}") or -1.0)
    comparison = []
    for label, k in (("rule list, best scorer on val", pick(scorer_keys)), ("rule list + logit, L2 on val", pick(logit_keys))):
        comparison.append({"model": label, "setting": k, "train_ap": full.get(f"train_ap_{k}"),
                           "val_ap": full.get(f"val_ap_{k}"), "test_ap": full.get(f"test_ap_{k}")})
    for src, bo in boosted.items():
        b_ = bo["best"]
        comparison.append({"model": f"boosted rules {src}, rounds on val", "setting": f"{b_['rounds']} rounds",
                           "train_ap": b_.get("train_ap"), "val_ap": b_["val_ap"], "test_ap": b_.get("test_ap")})
        if bo["refit"]:
            k = max(bo["refit"], key=lambda k: bo["refit"][k]["val_ap"] or -1.0)
            r_ = bo["refit"][k]
            comparison.append({"model": f"boosted rules {src} + refit", "setting": f"L2 {k}",
                               "train_ap": r_["train_ap"], "val_ap": r_["val_ap"], "test_ap": r_["test_ap"]})
    for n_ in nets:
        comparison.append({"model": f"SetNet {n_['source']}, step on val", "setting": "early stop",
                           "train_ap": None, "val_ap": n_["best_val_ap"], "test_ap": n_["test_ap_at_best_val"]})
        comparison.append({"model": f"SetNet {n_['source']}, last step", "setting": f"{NET_EPOCHS} steps",
                           "train_ap": n_["train_ap"], "val_ap": n_["val_ap"], "test_ap": n_["test_ap"]})

    base_tr, base_va = float(train["y"].float().mean()), float(val["y"].float().mean())
    base_te = float(test["y"].float().mean())
    print(f"\n[Brake factors] {args.head}: train {len(train['y'])} rows, base {base_tr:.3f}; "
          f"val {len(val['y'])} rows, base {base_va:.3f}; test {len(test['y'])} rows, base {base_te:.3f}")
    print("\n== Best single interval per (group, feature), by train gain")
    print(f"{'group':6s} {'feature':18s} {'cat':9s} {'(a, b]':>20s} {'gain':>7s} {'p_tr':>5s} {'p_val':>5s} "
          f"{'lift':>5s} {'recall':>6s} {'events':>6s}")
    for r in sorted(singles, key=lambda r: -r["gain"])[:60]:
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
    for r in sorted(regions[1], key=lambda r: -r["gain"])[:60]:
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
    print(f"\n== Rule list (sequential covering, <= {LIST_DEPTH} literals per rule, >= {LIST_MIN_HITS} positive rows): "
          f"AP by number of rules; list = first rule that fires, max = best rule that fires; coverage = share of "
          f"the positive rows some rule covers")
    print(f"  rules: {rlist['phases']['foil']} FOIL + {rlist['phases']['small']} small precise + "
          f"{rlist['phases']['broad']} broad (no cap)")
    scorers = [("list", "list"), ("max", "max"), ("list_lb", "list LB"), ("max_lb", "max LB"),
               ("list_shrunk", "list shrunk")] + [(f"logit{l:g}", f"logit {l:g}") for l in LOGIT_L2]
    print("  AP train / val by number of rules and scorer (list: first rule fires; max: best rule fires; LB: by the "
          "Wilson lower bound of the precision; shrunk: precision shrunk to base; logit: logistic regression on the "
          "rule firings):")
    print("  " + f"{'rules':>5s} {'cov':>5s} " + " ".join(f"{lab:>13s}" for _k, lab in scorers))
    for c in rlist["curve"]:
        print("  " + f"{c['rules']:5d} {c['val_coverage']:5.2f} " + " ".join(
            f"{c.get('train_ap_' + k, float('nan')):6.3f}/{c.get('val_ap_' + k, float('nan')):.3f}" for k, _l in scorers))
    for n_ in nets:
        print(f"  SetNet ({n_['source']}): train AP {n_['train_ap']:.3f}, val AP {n_['val_ap']:.3f}")
    print("  without a group of rules (val AP per scorer; full list first):")
    full = rlist["full"]
    print("   " + f"{'all rules':22s} " + " ".join(f"{lab}={full.get('val_ap_' + k, float('nan')):.3f}" for k, lab in scorers))
    for name, g in rlist["groups"].items():
        print(f"   {name:12s} {g['rules']:4d} rules (val prec {g['val_precision']:.2f}, val cov {g['val_coverage']:.2f}) "
              + " ".join(f"{lab}={g['without'].get('val_ap_' + k, float('nan')):.3f}" for k, lab in scorers))
    small = [r for r in rlist["rules"] if r["phase"] == "small"]
    if small:
        print("  phase-2 rules (first 15):")
        for r in small[:15]:
            print(f"    p_tr {r['precision']:.2f} ({r['hits']} hits, {r['events']} onsets) p_val {r['val_precision']:.2f} "
                  f"({r['val_hits']}/{r['val_rows']}): {r['text'][:200]}")
    print("  first rules:")
    for i, r in enumerate(rlist["rules"][:20]):
        print(f"   {i + 1:3d}. [{r['phase']}] p_tr {r['precision']:.2f} ({r['hits']} hits) p_val {r['val_precision']:.2f}: {r['text'][:220]}")
    print("\n== Refined rules: root, then each added condition (train / val precision, val events, val recall)")
    for item in refined:
        for k, st in enumerate(item["steps"]):
            print(f"  {'root ' if k == 0 else '   & '}{st['add']}: p_tr {st['train']['precision']:.2f}, "
                  f"p_val {st['val']['precision']:.2f}, events {st['val']['events']}, recall {st['val']['recall']:.2f}")
        print()
    for src, bo in boosted.items():
        print(f"\n== Boosted rules ({src}: {bo['literals']} literals): {bo['rounds']} rounds, best val at "
              f"{bo['best']['rounds']}, {bo['distinct_rules']} distinct rules ({bo['positive_rules']} raise the score)")
        print("  rounds  train / val / test AP")
        for c in bo["curve"][::2]:
            print(f"  {c['rounds']:6d}  {c['train_ap']:.3f} / {c['val_ap']:.3f} / {c['test_ap']:.3f}")
        print("  refit (logistic on the distinct rules): " + ", ".join(
            f"L2 {k}: {v['train_ap']:.3f} / {v['val_ap']:.3f} / {v['test_ap']:.3f}" for k, v in bo["refit"].items()))
        print("  rules of the largest |weight| x rows:")
        for r in bo["rules"][:25]:
            print(f"    w {r['weight']:+.2f} x{r['rounds']:3d} rows {r['rows']:6d} p_tr {r['precision']:.2f} "
                  f"p_val {r['val_precision']:.2f} p_test {r['test_precision']:.2f}: {r['text'][:180]}")
    print(f"\n== Test comparison (settings chosen on val; test untouched until here; prior AP ~ {base_te:.3f})")
    for c in comparison:
        f3 = lambda x: f"{x:.3f}" if x is not None else "  -  "
        print(f"  {c['model']:34s} {c['setting']:14s} train {f3(c['train_ap'])}  val {f3(c['val_ap'])}  "
              f"test {f3(c['test_ap'])}")
    path = out_dir / f"brake_factors_{tag}{'_capacity' if args.capacity else ''}.json"
    utils_data.save_json({"head": args.head, "stride": STRIDE, "window": WINDOW, "bins": Q, "pair_bins": Q2,
                          "min_pos": MIN_POS, "train_base": base_tr, "val_base": base_va, "test_base": base_te,
                          "comparison": comparison,
                          "boosted": {k: {**{x: y for x, y in v.items() if not x.startswith("_")},
                                          "rules": v["rules"][:500]} for k, v in boosted.items()},
                          "capacity": bool(args.capacity), "nets": nets, "rule_list": rlist,
                          "singles": singles, "pairs": pairs, "triples": triples,
                          "regions": {str(k): [{x: v for x, v in r.items() if not x.startswith("_")} for r in regions[k]]
                                      for k in regions},
                          "refined": [{"root": {x: v for x, v in i["root"].items() if not x.startswith("_")},
                                       "steps": i["steps"]} for i in refined]}, path)
    print(f"[Brake factors] saved {path}")


if __name__ == "__main__":
    main()
