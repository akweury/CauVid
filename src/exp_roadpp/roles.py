"""Agent roles: for every agent and frame, which other agent fills a role relative to it.

The TemporalRuleNet reads, for the agent of a row, its own atoms and the OR over
all other agents ("some other agent"). Agents of the same class are therefore
interchangeable: "a car brakes" cannot say which car. Roles name one specific
related agent per frame, from the boxes of the GT tracks, and the model gets
one more input channel per role holding that agent's atoms
(`TemporalRuleNet.layer_inputs`), e.g. "the agent ahead of me braked 1 s ago".

Boxes are normalized image coordinates (x1, y1, x2, y2) of the front camera,
y down; the bottom edge y2 is where the agent touches the ground, so a smaller
y2 is further away. The ego vehicle has no box: it stands at the bottom centre
of the image (EGO_BOX), so its roles use the same definitions.

ROLES (the order of the role channels is the order of cfg["roles"]):
  ahead:    the nearest agent in front of it in its lane column: further away
            (y2 smaller by > AHEAD_MIN_GAP) and horizontally inside its extent
            (>= AHEAD_MIN_OVERLAP of the candidate's width overlaps it); for the
            ego, the lead agent in the ego path.
  near_ped: the nearest pedestrian (agent class "Ped") whose ground point (bottom
            centre) is within NEAR_PED_MAX_HEIGHTS times the larger box height of
            the two (scale-aware: a fixed image distance would span metres near
            the camera and tens of metres far away), the agent itself excluded.
Agents off the ground (traffic lights, NON_GROUND_CLASSES) neither fill nor hold roles:
their boxes hang above the road, so "further up the image" says nothing about them.
The ego vehicle is never a role partner: its atoms (av:*) already stand apart
in the "others" channel, as no other agent holds them.
"""
import torch

ROLES = ("ahead", "near_ped")
EGO_BOX = (0.35, 1.0, 0.65, 1.0)          # the ego path: centre third of the image, at the bottom
AHEAD_MIN_GAP = 0.01
AHEAD_MIN_OVERLAP = 0.5
NEAR_PED_MAX_HEIGHTS = 1.0
PEDESTRIAN_CLASSES = ("Ped",)
NON_GROUND_CLASSES = ("TL", "OthTL")


def role_partners(boxes, classes, ego, roles=ROLES):
    """(T, N, K) long: the agent filling role k of agent n in frame t, -1 if none.

    boxes (T, N, 4) float, NaN where the agent has no box in that frame; classes
    [N] agent class labels (None if unknown); ego: the ego agent's index or None.
    An agent without a box in frame t (other than the ego) has no partners then,
    and is no partner of anyone."""
    T, N, _ = boxes.shape
    boxes = boxes.clone().float()
    has_box = ~torch.isnan(boxes).any(dim=2)                                           # (T, N)
    if ego is not None:
        boxes[:, ego] = torch.tensor(EGO_BOX)
        has_box[:, ego] = True
    out = torch.full((T, N, len(roles)), -1, dtype=torch.long)
    if N < 2:
        return out
    x1, y1, x2, y2 = boxes.unbind(dim=2)                                               # (T, N) each
    eye = torch.eye(N, dtype=torch.bool)[None]
    ground = torch.tensor([c not in NON_GROUND_CLASSES for c in classes], dtype=torch.bool)
    on_ground = has_box & ground[None, :]
    # Candidate partners m of agent n: both on the ground with a box, m != n, m is not the ego.
    candidate = on_ground[:, :, None] & on_ground[:, None, :] & ~eye                   # (T, n, m)
    if ego is not None:
        candidate[:, :, ego] = False
    for k, role in enumerate(roles):
        if role == "ahead":
            further = y2[:, None, :] < y2[:, :, None] - AHEAD_MIN_GAP                   # m further away than n
            overlap = (torch.minimum(x2[:, :, None], x2[:, None, :])
                       - torch.maximum(x1[:, :, None], x1[:, None, :])).clamp(min=0)
            width = (x2 - x1).clamp(min=1e-6)[:, None, :]
            ok = candidate & further & (overlap / width >= AHEAD_MIN_OVERLAP)
            # The nearest of them: the largest y2 (closest to n).
            score = torch.where(ok, y2[:, None, :], -float("inf"))
        elif role == "near_ped":
            ped = torch.tensor([c in PEDESTRIAN_CLASSES for c in classes], dtype=torch.bool)
            gx, gy = (x1 + x2) / 2, y2                                                  # ground points
            dist = torch.sqrt((gx[:, :, None] - gx[:, None, :]) ** 2 + (gy[:, :, None] - gy[:, None, :]) ** 2)
            height = (y2 - y1).clamp(min=0)
            scale = torch.maximum(height[:, :, None], height[:, None, :])                 # (T, n, m)
            ok = candidate & ped[None, None, :] & (dist <= NEAR_PED_MAX_HEIGHTS * scale)
            score = torch.where(ok, -dist, -float("inf"))
        else:
            raise ValueError(f"Unknown role {role!r}; known: {ROLES}.")
        best = score.argmax(dim=2)                                                     # (T, n)
        out[:, :, k] = torch.where(ok.any(dim=2), best, -1)
    return out


def video_boxes(video_boxes_entry, agent_ids, num_frames):
    """(T, N, 4) float boxes (NaN where missing) and [N] classes of one video, in
    the agent order of `step05.video_agent_ids` (agent_ids {tube_uid: n}), from
    its entry of the step_dataset box file ({tube_uid: {"label", "frames", "boxes"}})."""
    N = len(agent_ids)
    boxes = torch.full((num_frames, N, 4), float("nan"))
    classes = [None] * N
    for tube_uid, tube in (video_boxes_entry or {}).items():
        n = agent_ids.get(tube_uid)
        if n is None:
            continue
        classes[n] = tube["label"]
        frames = torch.as_tensor(tube["frames"], dtype=torch.long)
        keep = (frames >= 0) & (frames < num_frames)
        boxes[frames[keep], n] = torch.as_tensor(tube["boxes"], dtype=torch.float)[keep]
    return boxes, classes
