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
           near, both moving, directions differing by < SAME degrees;
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
  "gst:{grp,ngrp}_all_{action}"  every member holds it (debounced).
The actions are the label names after "Class:" of the agent classes, the ego's (av:) left out.
"""
import torch

from src.exp_roadpp import shape as agent_shape

K = 2                                   # frames of the move (as `shape.K`)
MOVE_PX = 3.0                           # a move below this over K frames gives no direction
NEAR = 2.0                              # centre distance < NEAR x the mean of the two boxes' larger sides
SAME = 45.0                             # degrees: linked if the directions differ by less
DEBOUNCE = 3
EDGE = 0.005
FRAME_SIZE_PX = (1920, 1280)
QUADRANTS = ("right", "up", "left", "down")
KINDS = ("veh", "ped")
PED_CLASSES = ("Ped",)
NO_KIND = ("TL",)                       # traffic lights: no motion of their own, never in a group
VERSION = "v4"


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
    return names


def tag():
    return f"_gst{K}n{NEAR:g}a{SAME:g}{VERSION}"


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


def gestalt_atoms(boxes, has_box, classes, X_labels, label_names, shape_spec=None):
    """(T, N, len(atom_names(label_names))) bool gestalt atoms of one video. boxes (T, N, 4) normalized (NaN
    without a box), has_box (T, N) bool (False for the ego), classes [N] the agents' class labels, X_labels
    (T, N, A_labels) bool the label atoms, label_names their names, shape_spec the shape thresholds (`shape.fit`;
    None: `default_shape_spec`)."""
    spec = shape_spec if shape_spec is not None else default_shape_spec()
    boxes, has_box, X_labels = boxes.cpu().float(), has_box.cpu(), X_labels.cpu().bool()
    T, N, _ = boxes.shape
    acts = actions(label_names)
    n_total = len(atom_names(label_names))
    if N == 0:
        return torch.zeros((T, N, n_total), dtype=torch.bool)
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
    kind = torch.tensor([-1 if c is None or str(c) in NO_KIND else (1 if str(c) in PED_CLASSES else 0)
                         for c in classes], dtype=torch.long)
    eye = torch.eye(N, dtype=torch.bool)[None]
    pair = has_box[:, :, None] & has_box[:, None, :] & ~eye
    dist = torch.hypot(cx[:, :, None] - cx[:, None, :], cy[:, :, None] - cy[:, None, :])
    near = pair & (dist < NEAR * (side[:, :, None] + side[:, None, :]) / 2)
    dang = ((angle[:, :, None] - angle[:, None, :] + 180) % 360 - 180).abs()           # NaN compares False
    same_kind = (kind[:, None] == kind[None, :]) & (kind[:, None] >= 0)
    link = near & same_kind[None] & moving[:, :, None] & moving[:, None, :] & (dang < SAME)
    member_ok = moving & (kind >= 0)[None]
    label = _components(link, member_ok)                                                # (T, N), N: none
    lab = label.clamp(max=N - 1)
    member = torch.nn.functional.one_hot(label, N + 1)[..., :N].bool() & member_ok[..., None]   # (T, i, g)
    size = member.sum(dim=1)                                                            # (T, g)
    is_group = size >= 2
    in_group = (label < N) & is_group.gather(1, lab)
    rad = torch.deg2rad(angle.nan_to_num(0.0))
    gx = (member.float() * torch.cos(rad)[..., None]).sum(dim=1)
    gy = (member.float() * torch.sin(rad)[..., None]).sum(dim=1)
    gq = torch.where(is_group, _quadrant(torch.rad2deg(torch.atan2(gy, gx)) % 360), torch.full_like(size, -1))
    gkind = torch.where(is_group, kind[None].expand(T, N), torch.full_like(size, -1))
    # The nearest member of another group near i: its group is i's "near group".
    other = near & (label < N)[:, None, :] & is_group.gather(1, lab)[:, None, :] \
        & ((label[:, None, :] != label[:, :, None]) | ~in_group[..., None])
    nearest = torch.where(other, dist, torch.full_like(dist, float("inf"))).argmin(dim=2)
    has_near = other.any(dim=2)
    near_g = label.gather(1, nearest).clamp(max=N - 1)                                  # (T, N) the near group
    my_g = lab
    out = []
    # Quadrants (debounced).
    quad = []
    for rel in ("in_group", "near_group", "group"):
        for k in range(len(KINDS)):
            for q in range(4):
                sel = (gq == q) & (gkind == k)                                          # (T, g)
                if rel == "in_group":
                    quad.append(in_group & sel.gather(1, my_g))
                elif rel == "near_group":
                    quad.append(has_near & sel.gather(1, near_g))
                else:
                    quad.append(sel.any(dim=1)[:, None].expand(T, N))
    out.append(_debounce(torch.stack(quad, dim=2)))
    # The group tracks' boxes: the union of the members' boxes (normalized), and every shape atom of them.
    big = torch.full_like(b[..., 0], float("inf"))[..., None]
    mins = [torch.where(member, b[..., c][..., None], big).amin(dim=1) for c in (0, 1)]
    maxs = [torch.where(member, b[..., c][..., None], -big).amax(dim=1) for c in (2, 3)]
    gbox = torch.stack(mins + maxs, dim=2)                                              # (T, g, 4)

    def track(g_index, valid):
        bx = gbox.gather(1, g_index[..., None].expand(T, N, 4))
        return torch.where(valid[..., None], bx, torch.full_like(bx, float("nan")))
    out.append(agent_shape.shape_atoms(track(my_g, in_group), in_group, classes, spec))
    out.append(agent_shape.shape_atoms(track(near_g, has_near), has_near, classes, spec))
    # Labels of the members: some other member / every member holds the action.
    A_lab = X_labels.shape[2]
    act_of = torch.zeros((A_lab, max(len(acts), 1)), dtype=torch.float)
    for a, nm in enumerate(label_names[:A_lab]):
        cls, _, act = nm.partition(":")
        if act in acts and cls not in ("av", "TL"):
            act_of[a, acts.index(act)] = 1.0
    holds = ((X_labels.float() @ act_of) > 0)[..., :len(acts)]                          # (T, N, n_act)
    labs = []
    for g_index, valid, own in ((my_g, in_group, True), (near_g, has_near, False)):
        mem = (label[:, None, :] == g_index[:, :, None]) & (label < N)[:, None, :] & valid[..., None]   # (T, i, j)
        others = mem & ~eye if own else mem
        cnt = mem.sum(dim=2, keepdim=True)
        any_ = torch.einsum("tij,tja->tia", others.float(), holds.float()) > 0
        all_ = (torch.einsum("tij,tja->tia", mem.float(), holds.float()) >= cnt) & (cnt > 0)
        labs += [any_ & valid[..., None], all_ & valid[..., None]]
    out.append(_debounce(torch.cat(labs, dim=2)))
    return torch.cat(out, dim=2) & has_box[..., None]
