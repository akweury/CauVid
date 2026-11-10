"""Geometry atoms: facts read from the GT boxes that hold without depth.

A front-camera box gives the bearing of an agent (the angle between the camera
axis and the line of sight to it) from its column alone; its distance, speed and
heading need depth and are left out. Each fact is an atom per agent and frame
like the GT labels, with the agent's class in it ("Car:bearing_left"), so it
fits every input scope of the TemporalRuleNet (own, others, roles). The geometry
atoms are inputs only: no head learns or is evaluated on them
(`step05.head_event_types`, event type "input").

FACTS (cfg["geometry"] picks the groups):
  bearing: bearing_far_left / bearing_left / bearing_centre / bearing_right /
           bearing_far_right, exactly one per frame with a box: the bearing
           theta = atan((x_centre - 0.5) * IMAGE_WIDTH / FOCAL_PX) of the box
           centre in the sectors BEARING_EDGES_DEG (|theta| < 5 deg centre, 5..15
           side, > 15 far side). A sector changes only once theta is
           HYSTERESIS_DEG past its edge, so an agent near an edge does not flicker
           (every flicker would be a fake onset).
  edge:    at_left_edge / at_right_edge: the box touches the image border
           (entering or leaving the view; its centre is then biased inwards).
  drift:   drifting_left / drifting_right: theta changed by more than
           DRIFT_MIN_DEG over the last DRIFT_FRAMES frames. The ego's own yaw turns
           every bearing, so drift is never asserted while the ego turns
           (EGO_TURN_LABELS) or with a box at the image border (its centre moves
           as the box is cut).
The ego vehicle has no box and no geometry atoms; the bearings are relative to it.
A geometry atom is only set where the agent holds some GT label atom, so the
agents and frames of the rows stay those of the labels.
"""
import math
import torch

FACTS = {
    "bearing": ("bearing_far_left", "bearing_left", "bearing_centre", "bearing_right", "bearing_far_right"),
    "edge": ("at_left_edge", "at_right_edge"),
    "drift": ("drifting_left", "drifting_right"),
}
GEOMETRY = ("bearing", "edge", "drift")
IMAGE_WIDTH = 1920                       # px, ROAD++ (Waymo front camera)
FOCAL_PX = 2055.0                        # px, Waymo front camera f_u (~50 deg horizontal field of view)
BEARING_EDGES_DEG = (-15.0, -5.0, 5.0, 15.0)
HYSTERESIS_DEG = 1.0
EDGE_MARGIN = 0.005                      # normalized x
DRIFT_FRAMES = 5
DRIFT_MIN_DEG = 1.0
EGO_TURN_LABELS = ("AV-TurLft", "AV-TurRht")


def fact_names(geometry):
    """The geometry facts of the groups `geometry`, in a fixed order."""
    unknown = [g for g in geometry if g not in FACTS]
    if unknown:
        raise ValueError(f"Unknown geometry groups {unknown}; known: {tuple(FACTS)}.")
    return [name for group in GEOMETRY if group in geometry for name in FACTS[group]]


def agent_classes(unique_atoms):
    """[(agent_class, agent_class_label)] of the agents (not the ego) in the label vocabulary, by class id."""
    classes = set()
    for signature in unique_atoms:
        fields = dict(zip(signature[0::2], signature[1::2]))
        if fields.get("agent_class") not in (None, "av"):
            classes.add((fields["agent_class"], fields["agent_class_label"]))
    return sorted(classes)


def vocabulary(unique_atoms, geometry):
    """Signatures of the geometry atoms, appended after the label vocabulary: one per
    agent class and fact, in the format of the label atoms (`step05._atom_name`
    reads them as "<class>:<fact>")."""
    return [["pred", "geometry", "geometry_id", f, "geometry_label", name,
             "agent_class", cls, "agent_class_label", label]
            for cls, label in agent_classes(unique_atoms)
            for f, name in enumerate(fact_names(geometry))]


def bearing_deg(boxes):
    """Bearing (degrees, positive to the right) of the box centres, (...,) of (..., 4)."""
    x_centre = (boxes[..., 0] + boxes[..., 2]) / 2
    return torch.rad2deg(torch.atan((x_centre - 0.5) * IMAGE_WIDTH / FOCAL_PX))


def _sectors(theta, has_box):
    """(T, N) long bearing sector 0..4 (-1 without a box), with hysteresis: a run of
    frames with a box starts in the plain sector of its first frame, and leaves its
    sector only once theta is HYSTERESIS_DEG past one of its edges."""
    edges = torch.tensor(BEARING_EDGES_DEG)
    plain = torch.bucketize(theta.nan_to_num(0.0), edges)              # 0..4
    T, N = theta.shape
    out = torch.full((T, N), -1, dtype=torch.long)
    current = torch.full((N,), -1, dtype=torch.long)
    lower = torch.cat([torch.tensor([-math.inf]), edges])              # sector s spans [lower[s], upper[s])
    upper = torch.cat([edges, torch.tensor([math.inf])])
    for t in range(T):
        box = has_box[t]
        start = box & (current < 0)
        held = box & (current >= 0)
        s = current.clamp(min=0)
        inside = (theta[t] >= lower[s] - HYSTERESIS_DEG) & (theta[t] < upper[s] + HYSTERESIS_DEG)
        current = torch.where(start | (held & ~inside), plain[t], current)
        current = torch.where(box, current, -1)
        out[t] = current
    return out


def geometry_atoms(boxes, labelled, geometry, ego_turning=None):
    """(T, N, G) bool geometry facts (`fact_names(geometry)` order) of one video.

    boxes (T, N, 4) float, NaN where the agent has no box (`roles.video_boxes`);
    labelled (T, N) bool: the agent holds some GT label atom in frame t (a fact
    is only set there); ego_turning (T,) bool: the ego turns in frame t (no drift then)."""
    names = fact_names(geometry)
    T, N, _ = boxes.shape
    out = torch.zeros((T, N, len(names)), dtype=torch.bool)
    has_box = ~torch.isnan(boxes).any(dim=2) & labelled
    theta = torch.where(has_box, bearing_deg(boxes), torch.nan)
    at_left = has_box & (boxes[..., 0] <= EDGE_MARGIN)
    at_right = has_box & (boxes[..., 2] >= 1 - EDGE_MARGIN)
    facts = {}
    if "bearing" in geometry:
        sector = _sectors(theta, has_box)
        for s, name in enumerate(FACTS["bearing"]):
            facts[name] = sector == s
    if "edge" in geometry:
        facts["at_left_edge"], facts["at_right_edge"] = at_left, at_right
    if "drift" in geometry:
        k = DRIFT_FRAMES
        change = torch.full((T, N), torch.nan)
        clean = has_box & ~at_left & ~at_right
        if T > k:
            ok = clean[k:] & clean[:-k]
            change[k:] = torch.where(ok, theta[k:] - theta[:-k], torch.nan)
        steady = torch.ones(T, dtype=torch.bool) if ego_turning is None else ~ego_turning.cpu().bool()
        facts["drifting_left"] = (change < -DRIFT_MIN_DEG) & steady[:, None]
        facts["drifting_right"] = (change > DRIFT_MIN_DEG) & steady[:, None]
    for g, name in enumerate(names):
        out[:, :, g] = facts[name]
    return out


def video_geometry(boxes, classes, X_labels, unique_atoms, geometry, ego):
    """(T, N, C * G) bool geometry columns of one video, in the order of
    `vocabulary(unique_atoms, geometry)` (C agent classes, each with its G facts):
    the facts of `geometry_atoms` of each agent in the columns of its class (agents
    of an unknown class and the ego get none). boxes, classes: `roles.video_boxes`;
    X_labels (T, N, A) bool label atoms (unique_atoms); ego: the ego's agent index or None."""
    names = fact_names(geometry)
    class_list = agent_classes(unique_atoms)
    column = {label: c for c, (_cls, label) in enumerate(class_list)}
    T, N, _A = X_labels.shape
    X_labels = X_labels.cpu()
    turn = [a for a, sig in enumerate(unique_atoms)
            if dict(zip(sig[0::2], sig[1::2])).get("action_id_label") in EGO_TURN_LABELS]
    ego_turning = (X_labels[:, ego][:, turn].any(dim=1) if ego is not None and turn
                   else torch.zeros(T, dtype=torch.bool))
    facts = geometry_atoms(boxes, X_labels.any(dim=2), geometry, ego_turning)    # (T, N, G)
    out = torch.zeros((T, N, len(class_list) * len(names)), dtype=torch.bool)
    for n, label in enumerate(classes):
        c = column.get(label)
        if c is None or n == ego:
            continue
        out[:, n, c * len(names):(c + 1) * len(names)] = facts[:, n]
    return out
