"""Gestalt atoms: an agent's relation to the boxes around it, by proximity, similarity and common fate, as inputs
of the rules (2026-11-23, the user's request).

Why: several boxes moving the same way at once (a queue starting at a green light, a group of pedestrians
crossing, the traffic flowing right while one car stays) says what an agent will do, beyond its own box. The
other-agent scope of the rules pools a label over every agent of the scene. The other agents' motion is not
read there at all (`step05.rule_input_mask`), because pooled over the whole scene it says little. These atoms
keep only the agents near the agent or similar to it.

Per frame t and agent i with a box (pixels of the image; the ego has no box and is never a partner):
  move     the centre's move over K frames (both boxes uncut by the image border): a direction if >= MOVE_PX,
           "still" if below;
  near     another agent j whose centre is closer than NEAR x the mean of the two boxes' larger sides;
  similar  |log(height_i / height_j)| < log(SIMILAR);
  same / opposite / cross  both move and their directions differ by < SAME / > 180 - SAME / in between.
Atoms (inputs only, own agent scope, debounced over DEBOUNCE frames):
  "gst:near_{1,2,3}"              >= 1 / 2 / 3 near agents;
  "gst:near_same_cls"             a near agent of the same class;
  "gst:near_similar"              a near agent of a similar size;
  "gst:co_move_{1,2}"             >= 1 / 2 near agents moving the same way as i (common fate);
  "gst:co_move_similar"           a near agent of a similar size moving the same way;
  "gst:counter_move"              a near agent moving the opposite way;
  "gst:cross_move"                a near agent moving across i's direction;
  "gst:others_move_i_still_{1,2}" i is still, >= 1 / 2 near agents move (the group starts, i not yet);
  "gst:i_move_others_still"       i moves, near agents are there and none of them moves;
  "gst:near_flow_{right,up,left,down}"  the near movers (excluding i) agree on a direction (mean unit vector
                                  >= FLOW_AGREE long) that lies in that quadrant;
  "gst:scene_{right,up,left,down}_{2,4}"  >= 2 / 4 other agents anywhere move in that quadrant;
  "gst:scene_co_move_{2,4}"       >= 2 / 4 other agents anywhere move the same way as i.
Quadrants: right [315, 45), up [45, 135), left [135, 225), down [225, 315) degrees (0 right, 90 up).
"""
import math

import torch

K = 2                                   # frames of the move (as `shape.K`)
MOVE_PX = 3.0                           # a move below this over K frames is "still"
NEAR = 2.0                              # centre distance < NEAR x the mean of the two boxes' larger sides
SIMILAR = 1.5                           # height ratio below this: similar size
SAME = 45.0                             # degrees: same direction below, opposite above 180 - SAME
FLOW_AGREE = 0.5                        # the near movers' mean unit vector must be at least this long
DEBOUNCE = 3
EDGE = 0.005
FRAME_SIZE_PX = (1920, 1280)
QUADRANTS = ("right", "up", "left", "down")
VERSION = "v1"


def atom_names():
    names = ["near_1", "near_2", "near_3", "near_same_cls", "near_similar", "co_move_1", "co_move_2",
             "co_move_similar", "counter_move", "cross_move", "others_move_i_still_1", "others_move_i_still_2",
             "i_move_others_still"]
    names += [f"near_flow_{q}" for q in QUADRANTS]
    names += [f"scene_{q}_{k}" for q in QUADRANTS for k in (2, 4)]
    names += ["scene_co_move_2", "scene_co_move_4"]
    return names


ATOMS = tuple(atom_names())


def tag():
    return f"_gst{K}n{NEAR:g}s{SIMILAR:g}{VERSION}"


def vocabulary():
    """Signatures of the gestalt atoms; `step05._atom_name` reads them as "gst:co_move_1"."""
    return [["pred", "gestalt", "gestalt_label", a, "agent_class_label", "gst"] for a in ATOMS]


def _quadrant(angle):
    """(...,) long 0..3 (QUADRANTS) of degrees, -1 for NaN."""
    q = (((angle + 45) % 360) // 90).long()
    return torch.where(torch.isnan(angle), torch.full_like(q, -1), q)


def _debounce(cond):
    out = cond.clone()
    for d in range(1, DEBOUNCE):
        shifted = torch.zeros_like(cond)
        shifted[d:] = cond[:-d]
        out &= shifted
    return out


def gestalt_atoms(boxes, has_box, classes):
    """(T, N, len(ATOMS)) bool gestalt atoms of one video. boxes (T, N, 4) normalized (NaN without a box),
    has_box (T, N) bool (False for the ego), classes [N] the agents' class labels."""
    boxes, has_box = boxes.cpu().float(), has_box.cpu()
    T, N, _ = boxes.shape
    W, H = FRAME_SIZE_PX
    b = boxes.nan_to_num(0.0)
    x1, y1, x2, y2 = (boxes[..., i] for i in range(4))
    cx, cy = (b[..., 0] + b[..., 2]) / 2 * W, (b[..., 1] + b[..., 3]) / 2 * H
    w, h = ((b[..., 2] - b[..., 0]) * W).clamp(min=1.0), ((b[..., 3] - b[..., 1]) * H).clamp(min=1.0)
    side = torch.maximum(w, h)
    uncut = has_box & ~((x1 < EDGE) | (y1 < EDGE) | (x2 > 1 - EDGE) | (y2 > 1 - EDGE))
    # Own move: a direction (moving) or still, unknown without two uncut boxes K frames apart.
    angle = torch.full((T, N), float("nan"))
    still = torch.zeros((T, N), dtype=torch.bool)
    if T > K:
        dx, dy = cx[K:] - cx[:-K], cy[K:] - cy[:-K]
        ok = uncut[K:] & uncut[:-K]
        dist = torch.hypot(dx, dy)
        angle[K:] = torch.where(ok & (dist >= MOVE_PX), torch.rad2deg(torch.atan2(-dy, dx)) % 360,
                                torch.full_like(dx, float("nan")))
        still[K:] = ok & (dist < MOVE_PX)
    moving = ~torch.isnan(angle)
    # Pairs (T, N, N): i the agent, j the partner.
    eye = torch.eye(N, dtype=torch.bool)[None]
    pair = has_box[:, :, None] & has_box[:, None, :] & ~eye
    dist = torch.hypot(cx[:, :, None] - cx[:, None, :], cy[:, :, None] - cy[:, None, :])
    near = pair & (dist < NEAR * (side[:, :, None] + side[:, None, :]) / 2)
    similar = (torch.log(h[:, :, None] / h[:, None, :])).abs() < math.log(SIMILAR)
    cls = {c: k for k, c in enumerate(sorted({str(c) for c in classes}))}
    cid = torch.tensor([cls[str(c)] for c in classes], dtype=torch.long) if N else torch.zeros(0, dtype=torch.long)
    same_cls = (cid[:, None] == cid[None, :])[None]
    both = moving[:, :, None] & moving[:, None, :]
    dang = ((angle[:, :, None] - angle[:, None, :] + 180) % 360 - 180).abs()               # NaN compares False
    same = both & (dang < SAME)
    opposite = both & (dang > 180 - SAME)
    cross = both & ~same & ~opposite
    n_near = near.sum(dim=2)
    co = (near & same).sum(dim=2)
    near_moving = (near & moving[:, None, :]).sum(dim=2)
    near_still = (near & still[:, None, :]).sum(dim=2)
    # Mean direction of the near movers (unit vectors, i excluded).
    rad = torch.deg2rad(angle.nan_to_num(0.0))
    ux, uy = torch.where(moving, torch.cos(rad), 0.0), torch.where(moving, torch.sin(rad), 0.0)
    nm = (near & moving[:, None, :]).float()
    sx, sy = (nm * ux[:, None, :]).sum(2), (nm * uy[:, None, :]).sum(2)
    flow = torch.rad2deg(torch.atan2(sy, sx)) % 360
    agree = torch.hypot(sx, sy) >= FLOW_AGREE * near_moving                 # the near movers agree on a direction
    flow = torch.where((near_moving > 0) & agree, flow, torch.full_like(flow, float("nan")))
    flow_q = _quadrant(flow)
    # Scene: every other agent with a box, by the quadrant of its move.
    q_all = _quadrant(angle)                                                                 # (T, N)
    scene_q = torch.stack([((q_all == k) & has_box).sum(dim=1) for k in range(4)], dim=1)  # (T, 4)
    own_q = torch.nn.functional.one_hot(q_all.clamp(min=0), 4).bool() & (q_all >= 0)[..., None]
    scene_others = scene_q[:, None, :] - own_q.long()                                        # (T, N, 4)
    scene_co = (pair & same).sum(dim=2)
    cond = [n_near >= 1, n_near >= 2, n_near >= 3, (near & same_cls).any(dim=2), (near & similar).any(dim=2),
            co >= 1, co >= 2, (near & similar & same).any(dim=2), (near & opposite).any(dim=2),
            (near & cross).any(dim=2), still & (near_moving >= 1), still & (near_moving >= 2),
            moving & (n_near >= 1) & (near_moving == 0) & (near_still >= 1)]
    cond += [flow_q == k for k in range(4)]
    cond += [scene_others[..., k] >= n for k in range(4) for n in (2, 4)]
    cond += [scene_co >= 2, scene_co >= 4]
    out = torch.stack(cond, dim=2)
    return _debounce(out) & has_box[..., None]
