"""Gestalt atoms: groups of boxes that are close and move the same way (common fate), described with the same
concepts as a single box, as inputs of the rules (2026-11-23, the user's request).

Why: several boxes moving the same way at once (a queue starting at a green light, a group of pedestrians
crossing, the traffic flowing right while one car stays) says what an agent will do, beyond its own box.

History:
  - v1: agent-relative counts (near_1..3, co_move_1/2, scene counts 2 / 4). No clear gain (rule lab 2026-11-23,
    83 heads: soft +0.006, CI [-0.005, +0.018]); the rules used mostly the negations ("!gst:near_1") as filler.
  - v2: a group is >= 2 agents, close and moving the same way, whatever its size (the user: two or three cars
    moving right together are the same fact).
  - v3: + the group's 360-degree direction.
  - v4 (the user: every concept of a single box has its counterpart for a group): a group is described as a
    box of its own, with every shape atom (`shape.py`), and by its members' labels.

Groups, per frame t (pixels of the image; the ego has no box and is never a member):
  move     the centre's move over K frames (both boxes uncut by the image border): a direction if >= MOVE_PX;
  near     two centres closer than NEAR x the mean of the two boxes' larger sides;
  link     two agents of the same kind (KINDS: vehicles incl. two-wheelers, pedestrians; traffic lights none),
           near, both moving, directions differing by < SAME degrees, speeds (box sizes per frame over K_SPEED
           frames) differing by a factor < SPEED_SIM (v5);
  group    a connected component of the links with >= 2 agents; its quadrant is that of the members' mean
           direction: right [315, 45), up [45, 135), left [135, 225), down [225, 315) degrees (0 right, 90 up).
Two group tracks per agent i:
  grp      i's own group, while i is in one;
  ngrp     the group of the nearest member of another group near i.
The track's box is the union of the members' boxes, frame by frame. Members join and leave, so the box can grow or
shrink with the membership too (the group spreads, gathers).

Atoms (inputs only, own agent scope):
  "gst:{in_group,near_group,group}_{kind}_{q}"  i is in / is near / the scene has a group of that kind moving to
      q (debounced over DEBOUNCE frames);
  "gst:grp_{a}" / "gst:ngrp_{a}"  every shape atom a of `shape.py` (phases, order-free "rec", the box when the
      group formed, its aspect, "young", 360-degree "dir" / "path") of the group track's box: e.g.
      "gst:grp_rec_w_grow+h_grow" (the group comes closer), "gst:ngrp_dir_200" (the nearby group's move lies in
      [200, 380));
  "gst:{grp,ngrp}_any_{action}"  some other member holds the action label (Brake, Stop, TurRht, IncomLane, ..);
  "gst:{grp,ngrp}_all_{action}"  every member holds it (debounced);
  "gst:mate_new_{action}"  (v7) an agent that was in the agent's group within the last MATE_MEMORY frames started
      the action within the last NEW_WINDOW frames: the leader of the queue entered the junction (Jun).
v7: still agents form groups too (near, same kind, both still, similar size; never with moving ones).
The actions are the label names after "Class:" of the agent classes, the ego's (av:) left out.
"""
import torch

from src.exp_roadpp import shape as agent_shape

K = 2                                   # frames of the move (as `shape.K`)
MOVE_PX = 3.0                           # a move below this over K frames gives no direction
NEAR = 2.0                              # centre distance < NEAR x the mean of the two boxes' larger sides
SAME = 45.0                             # degrees: linked if the directions differ by less
# v5 (the user: a box clearly faster than the others is not in their group): linked only if their speeds, in box
# sizes per frame over K_SPEED frames (the same for near and far boxes), differ by a factor < SPEED_SIM.
K_SPEED = 5
SPEED_SIM = 1.5
STILL_SIMILAR = 1.5                     # v7: still agents link if their heights differ by a factor below this
# v7 (the user: a queue on the oncoming lane, one car enters the junction, the others will follow): "mate_new_a",
# a group mate within the last MATE_MEMORY frames (it may have left the group by now) started action a within the
# last NEW_WINDOW frames (off in the NEW_OFF frames before).
MATE_MEMORY, NEW_WINDOW, NEW_OFF = 30, 20, 5
DEBOUNCE = 3
EDGE = 0.005
FRAME_SIZE_PX = (1920, 1280)
QUADRANTS = ("right", "up", "left", "down")
KINDS = ("veh", "ped")
PED_CLASSES = ("Ped",)
NO_KIND = ("TL",)                       # traffic lights: no motion of their own, never in a group
VERSION = "v7"                          # v4: group = a box with every shape atom; v5: + similar speed; v6: tracked;
                                        # v7: still groups and "mate_new" (a recent group mate just started a)


def actions(label_names):
    """The action names of the label atoms ("Car:Brake" -> "Brake"), the ego's and the traffic lights' left out,
    in first-seen order."""
    out = []
    for nm in label_names:
        cls, _, act = nm.partition(":")
        if act and cls not in ("av", "TL") and act not in out:
            out.append(act)
    return out


def atom_names(label_names):
    names = [f"{rel}_{k}_{q}" for rel in ("in_group", "near_group", "group") for k in KINDS for q in QUADRANTS]
    names += [f"{g}_{a}" for g in ("grp", "ngrp") for a in agent_shape.ATOMS]
    names += [f"{g}_{m}_{a}" for g in ("grp", "ngrp") for m in ("any", "all") for a in actions(label_names)]
    names += [f"mate_new_{a}" for a in actions(label_names)]
    return names


def tag():
    return f"_gst{K}n{NEAR:g}a{SAME:g}v{SPEED_SIM:g}{VERSION}"


def vocabulary(label_names):
    """Signatures of the gestalt atoms; `step05._atom_name` reads them as "gst:grp_p1_x_right". label_names: the
    names of the label atoms (`step05._atom_name`)."""
    return [["pred", "gestalt", "gestalt_label", a, "agent_class_label", "gst"] for a in atom_names(label_names)]


def default_shape_spec():
    """Shape thresholds without a fit (the FLOORS), for the group tracks when the shape atoms are off."""
    return agent_shape.fit({})


def _quadrant(angle):
    q = (((angle + 45) % 360) // 90).long()
    return torch.where(torch.isnan(angle), torch.full_like(q, -1), q)


def _debounce(cond):
    out = cond.clone()
    for d in range(1, DEBOUNCE):
        shifted = torch.zeros_like(cond)
        shifted[d:] = cond[:-d]
        out &= shifted
    return out


def _components(adj, valid):
    """(T, N) long component label (the smallest member index; N for agents not in valid) of (T, N, N) links."""
    T, N, _ = adj.shape
    adj = adj | (torch.eye(N, dtype=torch.bool)[None] & valid[:, :, None])
    label = torch.where(valid, torch.arange(N)[None].expand(T, N), N)
    while True:
        new = torch.where(valid, torch.where(adj, label[:, None, :], N).amin(dim=2), N)
        if torch.equal(new, label):
            return label
        label = new


def groups(boxes, has_box, classes):
    """The groups of one video (module doc), as tensors: label (T, N) the component of each agent (its smallest
    member index; N: in none), in_group (T, N), size / is_group / gq (quadrant) / gkind / gdir (degrees) per
    component (T, g), angle (T, N) each agent's move direction (NaN: no move), kind (N,), and the pair tensors
    near / dist (T, N, N). Shared by `gestalt_atoms` and `gestalt_viz`."""
    boxes, has_box = boxes.cpu().float(), has_box.cpu()
    T, N, _ = boxes.shape
    W, H = FRAME_SIZE_PX
    b = boxes.nan_to_num(0.0)
    x1, y1, x2, y2 = (boxes[..., i] for i in range(4))
    cx, cy = (b[..., 0] + b[..., 2]) / 2 * W, (b[..., 1] + b[..., 3]) / 2 * H
    side = torch.maximum(((b[..., 2] - b[..., 0]) * W).clamp(min=1.0), ((b[..., 3] - b[..., 1]) * H).clamp(min=1.0))
    uncut = has_box & ~((x1 < EDGE) | (y1 < EDGE) | (x2 > 1 - EDGE) | (y2 > 1 - EDGE))
    angle = torch.full((T, N), float("nan"))
    if T > K:
        dx, dy = cx[K:] - cx[:-K], cy[K:] - cy[:-K]
        ok = uncut[K:] & uncut[:-K] & (torch.hypot(dx, dy) >= MOVE_PX)
        angle[K:] = torch.where(ok, torch.rad2deg(torch.atan2(-dy, dx)) % 360, torch.full_like(dx, float("nan")))
    moving = ~torch.isnan(angle)
    # Still (v7): two uncut boxes K frames apart and a move below MOVE_PX (unknown without them).
    still = torch.zeros((T, N), dtype=torch.bool)
    if T > K:
        still[K:] = uncut[K:] & uncut[:-K] & (torch.hypot(cx[K:] - cx[:-K], cy[K:] - cy[:-K]) < MOVE_PX)
    # Speed: the centre's move over K_SPEED frames (K frames where the track is shorter), in larger box sides.
    speed = torch.full((T, N), float("nan"))
    for k in (K, K_SPEED):
        if T > k:
            d = torch.hypot(cx[k:] - cx[:-k], cy[k:] - cy[:-k]) / (k * side[k:])
            ok_k = uncut[k:] & uncut[:-k]
            speed[k:] = torch.where(ok_k, d, speed[k:])
    speed = torch.where(moving, speed, torch.full_like(speed, float("nan")))
    kind = torch.tensor([-1 if c is None or str(c) in NO_KIND else (1 if str(c) in PED_CLASSES else 0)
                         for c in classes], dtype=torch.long)
    eye = torch.eye(N, dtype=torch.bool)[None]
    pair = has_box[:, :, None] & has_box[:, None, :] & ~eye
    dist = torch.hypot(cx[:, :, None] - cx[:, None, :], cy[:, :, None] - cy[:, None, :])
    near = pair & (dist < NEAR * (side[:, :, None] + side[:, None, :]) / 2)
    dang = ((angle[:, :, None] - angle[:, None, :] + 180) % 360 - 180).abs()           # NaN compares False
    same_kind = (kind[:, None] == kind[None, :]) & (kind[:, None] >= 0)
    ratio = (torch.log(speed[:, :, None].clamp(min=1e-6)) - torch.log(speed[:, None, :].clamp(min=1e-6))).abs()
    link = near & same_kind[None] & moving[:, :, None] & moving[:, None, :] & (dang < SAME) \
        & (ratio < torch.log(torch.tensor(SPEED_SIM)))                                  # NaN compares False
    # v7: still agents group too (a queue waiting at the junction, pedestrians waiting to cross): near, the same
    # kind, both still, of a similar size. A still agent and a moving one never link.
    hgt = (b[..., 3] - b[..., 1]).clamp(min=1e-6)
    similar = (torch.log(hgt[:, :, None] / hgt[:, None, :])).abs() < torch.log(torch.tensor(STILL_SIMILAR))
    link = link | (near & same_kind[None] & still[:, :, None] & still[:, None, :] & similar)
    member_ok = (moving | still) & (kind >= 0)[None]
    label = _components(link, member_ok)                                                # (T, N), N: none
    lab = label.clamp(max=N - 1)
    member = torch.nn.functional.one_hot(label, N + 1)[..., :N].bool() & member_ok[..., None]   # (T, i, g)
    size = member.sum(dim=1)                                                            # (T, g)
    is_group = size >= 2
    in_group = (label < N) & is_group.gather(1, lab)
    rad = torch.deg2rad(angle.nan_to_num(0.0))
    gx = (member.float() * torch.cos(rad)[..., None]).sum(dim=1)
    gy = (member.float() * torch.sin(rad)[..., None]).sum(dim=1)
    movers = (member & moving[..., None]).sum(dim=1) > 0                               # a moving group (T, g)
    gq = torch.where(is_group & movers, _quadrant(torch.rad2deg(torch.atan2(gy, gx)) % 360), torch.full_like(size, -1))
    gkind = torch.where(is_group, kind[None].expand(T, N), torch.full_like(size, -1))
    gdir = torch.where(is_group & movers, torch.rad2deg(torch.atan2(gy, gx)) % 360, torch.full_like(gx, float("nan")))
    return {"b": b, "angle": angle, "speed": speed, "still": still, "kind": kind, "eye": eye, "dist": dist, "near": near, "label": label, "lab": lab,
            "member": member, "size": size, "is_group": is_group, "in_group": in_group, "gq": gq, "gkind": gkind,
            "gdir": gdir}


# v6 (the user, 2026-11-23: the shown group jumped from two turning cars to two pedestrians on the crossing and
# back): groups are tracked over the frames. A group keeps its id while it shares members with the group of a frame
# up to GAP frames before. A gap of <= GAP frames is filled with the members on both sides. The agent's near group
# sticks to the one it had while that one is still near; a new one is the nearest of the agent's kind first.
GAP = 5


def track_groups(g, has_box, gap=GAP):
    """Persistent groups of one video from the per-frame groups of `groups`. Returns
      M      (T, N, P) bool: agent n is a member of track p in frame t (gaps of <= gap frames filled);
      own    (T, N) long: the track agent n is in (-1: none);
      near   (T, N) long: agent n's near group (-1: none), the same track while it stays near;
      kind   (P,) long (`KINDS` index), present (T, P) bool, dir (T, P) degrees (NaN: no mover), q (T, P) quadrant."""
    label, in_group, near_pair, dist = g["label"], g["in_group"], g["near"], g["dist"]
    T, N = label.shape
    tracks, frame = [], [dict() for _ in range(T)]                     # frame[t]: {track: set of members}
    for t in range(T):
        comps = {}
        for n in torch.nonzero(in_group[t]).flatten().tolist():
            comps.setdefault(int(label[t, n]), set()).add(n)
        cands = [p for p, tr in enumerate(tracks) if t - tr["last"] <= gap + 1]
        pairs = sorted(((len(S & tracks[p]["members"]) / len(S | tracks[p]["members"]), k, p)
                        for k, S in comps.items() for p in cands if S & tracks[p]["members"]), reverse=True)
        assign, used = {}, set()
        for _j, k, p in pairs:
            if k not in assign and p not in used:
                assign[k] = p
                used.add(p)
        for k, S in comps.items():
            p = assign.get(k)
            if p is None:
                p = len(tracks)
                tracks.append({"members": S, "last": t, "kind": int(g["kind"][next(iter(S))])})
            else:
                tracks[p].update(members=S, last=t)
            frame[t][p] = S
    # Gaps of <= gap frames: the members on both sides that still have a box.
    for p in range(len(tracks)):
        ts = [t for t in range(T) if p in frame[t]]
        for a, b in zip(ts, ts[1:]):
            if 1 < b - a <= gap + 1:
                keep = frame[a][p] & frame[b][p]
                for t in range(a + 1, b):
                    S = {n for n in keep if bool(has_box[t, n])}
                    if S:
                        frame[t][p] = S
    P = len(tracks)
    M = torch.zeros((T, N, max(P, 1)), dtype=torch.bool)
    for t in range(T):
        for p, S in frame[t].items():
            M[t, list(S), p] = True
    own = torch.full((T, N), -1, dtype=torch.long)
    for t in range(T):
        for p, S in frame[t].items():
            for n in S:
                if own[t, n] < 0 or bool(in_group[t, n]) and int(label[t, n]) in {int(label[t, m]) for m in S}:
                    own[t, n] = p
    kind = torch.tensor([tr["kind"] for tr in tracks] or [-1], dtype=torch.long)
    present = M.any(dim=1)
    rad = torch.deg2rad(g["angle"].nan_to_num(0.0))
    moving = ~torch.isnan(g["angle"])
    mm = (M & moving[..., None]).float()
    sx = (mm * torch.cos(rad)[..., None]).sum(dim=1)
    sy = (mm * torch.sin(rad)[..., None]).sum(dim=1)
    direction = torch.where(mm.sum(dim=1) > 0, torch.rad2deg(torch.atan2(sy, sx)) % 360, torch.full_like(sx, float("nan")))
    q = _quadrant(direction)
    # The near group, sticky: the track of the agent's previous near group while one of its members is near.
    near = torch.full((T, N), -1, dtype=torch.long)
    for n in range(N):
        prev, last_seen = -1, -10 ** 9
        my_kind = int(g["kind"][n])
        for t in range(T):
            if not bool(has_box[t, n]):
                prev = -1
                continue
            cands = {}
            for p, S in frame[t].items():
                if p == int(own[t, n]) or n in S:
                    continue
                close = [m for m in S if bool(near_pair[t, n, m])]
                if close:
                    cands[p] = min(float(dist[t, n, m]) for m in close)
            if prev in cands:
                choice = prev
            elif prev >= 0 and t - last_seen <= gap and p_exists(frame, t, prev):
                choice = prev                                          # briefly out of reach: keep it
            elif cands:
                choice = min(cands, key=lambda p: (int(kind[p]) != my_kind, cands[p]))
            else:
                choice = -1
            if choice >= 0 and choice in cands:
                last_seen = t
            near[t, n] = choice
            prev = choice
    return {"M": M, "own": own, "near": near, "kind": kind, "present": present, "dir": direction, "q": q,
            "num": P}


def p_exists(frame, t, p):
    return p in frame[t]


def gestalt_atoms(boxes, has_box, classes, X_labels, label_names, shape_spec=None):
    """(T, N, len(atom_names(label_names))) bool gestalt atoms of one video. boxes (T, N, 4) normalized (NaN
    without a box), has_box (T, N) bool (False for the ego), classes [N] the agents' class labels, X_labels
    (T, N, A_labels) bool the label atoms, label_names their names, shape_spec the shape thresholds (`shape.fit`;
    None: `default_shape_spec`). The groups are tracked over the frames (`track_groups`)."""
    spec = shape_spec if shape_spec is not None else default_shape_spec()
    boxes, has_box, X_labels = boxes.cpu().float(), has_box.cpu(), X_labels.cpu().bool()
    T, N, _ = boxes.shape
    acts = actions(label_names)
    n_total = len(atom_names(label_names))
    if N == 0:
        return torch.zeros((T, N, n_total), dtype=torch.bool)
    g = groups(boxes, has_box, classes)
    tr = track_groups(g, has_box)
    b, eye = g["b"], g["eye"]
    M, own, near_g, kind, present, q = tr["M"], tr["own"], tr["near"], tr["kind"], tr["present"], tr["q"]
    P = M.shape[2]
    in_group, has_near = own >= 0, near_g >= 0
    my_p, near_p = own.clamp(min=0), near_g.clamp(min=0)
    out = []
    # Quadrants (debounced).
    quad = []
    for rel in ("in_group", "near_group", "group"):
        for k in range(len(KINDS)):
            for qq in range(4):
                sel = (q == qq) & (kind[None, :] == k) & present                       # (T, P)
                if rel == "in_group":
                    quad.append(in_group & sel.gather(1, my_p))
                elif rel == "near_group":
                    quad.append(has_near & sel.gather(1, near_p))
                else:
                    quad.append(sel.any(dim=1)[:, None].expand(T, N))
    out.append(_debounce(torch.stack(quad, dim=2)))
    # The group tracks' boxes: the union of the members' boxes (normalized), and every shape atom of them.
    big = torch.full_like(b[..., 0], float("inf"))[..., None]
    mins = [torch.where(M, b[..., c][..., None], big).amin(dim=1) for c in (0, 1)]
    maxs = [torch.where(M, b[..., c][..., None], -big).amax(dim=1) for c in (2, 3)]
    gbox = torch.stack(mins + maxs, dim=2)                                              # (T, P, 4)

    def track(p_index, valid):
        bx = gbox.gather(1, p_index[..., None].expand(T, N, 4))
        return torch.where(valid[..., None], bx, torch.full_like(bx, float("nan")))
    out.append(agent_shape.shape_atoms(track(my_p, in_group), in_group, classes, spec))
    out.append(agent_shape.shape_atoms(track(near_p, has_near), has_near, classes, spec))
    # Labels of the members: some other member / every member holds the action.
    A_lab = X_labels.shape[2]
    act_of = torch.zeros((A_lab, max(len(acts), 1)), dtype=torch.float)
    for a, nm in enumerate(label_names[:A_lab]):
        cls, _, act = nm.partition(":")
        if act in acts and cls not in ("av", "TL"):
            act_of[a, acts.index(act)] = 1.0
    holds = ((X_labels.float() @ act_of) > 0)[..., :len(acts)]                          # (T, N, n_act)
    labs = []
    for p_index, valid, is_own in ((my_p, in_group, True), (near_p, has_near, False)):
        mem = M.gather(2, p_index[:, None, :].expand(T, N, N)).transpose(1, 2) & valid[..., None]   # (T, i, j)
        others = mem & ~eye if is_own else mem
        cnt = mem.sum(dim=2, keepdim=True)
        any_ = torch.einsum("tij,tja->tia", others.float(), holds.float()) > 0
        all_ = (torch.einsum("tij,tja->tia", mem.float(), holds.float()) >= cnt) & (cnt > 0)
        labs += [any_ & valid[..., None], all_ & valid[..., None]]
    out.append(_debounce(torch.cat(labs, dim=2)))
    # mate_new: a group mate of the last MATE_MEMORY frames started the action within the last NEW_WINDOW frames.
    Mf = M.float()
    co = (torch.einsum("tip,tjp->tij", Mf, Mf) > 0) & ~eye                              # (T, i, j) mates now
    cs = torch.cat([torch.zeros((1, N, N)), co.float().cumsum(0)])
    t_idx = torch.arange(T)
    mate = (cs[t_idx + 1] - cs[(t_idx - MATE_MEMORY).clamp(min=0)]) > 0                  # mates in the window
    hc = torch.cat([torch.zeros((1, N, holds.shape[2])), holds.float().cumsum(0)])
    off_before = (hc[t_idx] - hc[(t_idx - NEW_OFF).clamp(min=0)]) == 0                   # off in the NEW_OFF before
    new_on = holds & off_before & (t_idx >= NEW_OFF)[:, None, None]
    nc = torch.cat([torch.zeros((1, N, holds.shape[2])), new_on.float().cumsum(0)])
    recent = (nc[t_idx + 1] - nc[(t_idx - NEW_WINDOW).clamp(min=0)]) > 0                 # (T, N, n_act)
    out.append(torch.einsum("tij,tja->tia", mate.float(), recent.float()) > 0)
    del P
    return torch.cat(out, dim=2) & has_box[..., None]
