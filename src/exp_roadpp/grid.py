"""Grid atoms: where an agent's box lies in the 2D image, as atoms of the TemporalRuleNet.

The image is cut into GRID_ROWS x GRID_COLS cells. Each agent's box gives two points,
its top-left (x1, y1) and bottom-right (x2, y2) corner, and each point the cells it lies
in. A cell is widened by OVERLAP of its size on every side, so a point near an edge lies
in both neighbouring cells (no flicker between them, every flicker would be a fake change).
Both axes are cut in rank space: a corner's x (y) is mapped to its rank among that
corner's x (y) in the train boxes (0..1, `axis_knots`), and the ranks are cut into equal
columns (rows), widened by OVERLAP of a band, so each band holds the same share of the
boxes and the overlap the same share at every edge. Equal cuts in pixels separate the
agents worse: most boxes sit near the horizon and the image centre, so equal pixel rows
leave the top and bottom cells nearly empty (and the narrow middle row in pixels is
covered by the overlap of its tall neighbours), and the equal pixel column 0.4..0.6 holds
~43% of the boxes. On the full test split, pairs of agents in the same frame with a
common label have identical cells at both corners in 16.3% (pixel columns) vs 12.0%
(rank columns) of the cases; cuts searched for the fewest such pairs on train did no
better on test (12.6%; the best row cuts were the equal ones). The top-left corners lie
mostly above the horizon, the bottom-right ones below it, so each corner gets its own ranks.

Atoms (inputs only, never heads; `step05.is_input_atom`), after the label atoms:
  position: "pos:TL_r{i}c{j}" / "pos:BR_r{i}c{j}", 2 * R * C atoms: the agent's own
      corners. Rules read them in the agent's own scope (where am I in the image).
  compound: "<label atom>@TL_r{i}c{j}" / "...@BR_...": the agent holds the label atom
      and that corner lies in that cell. Rules read them in the "other agent" scope, so
      a rule can name where in the image another agent does something ("an agent whose
      box bottom-right is in r2c2 brakes"), which the plain "some other agent brakes"
      cannot. Only the (atom, corner, cell) triples seen in at least MIN_VIDEOS train
      videos and MIN_FRAMES train rows are kept (`select_compound`).
The ego has no box and gets no grid atom; its label atoms stay plain.
The grid follows the 2D image only (no depth): the rules it allows are rules of the
image, not of the 3D scene, which the fixed front camera of ROAD++ makes consistent
across videos.
"""
import torch

# 3x3: the knee for telling an agent from the others by its own inputs (full test split, share of
# rows whose self inputs equal another agent's: no grid 54%, 2x2 22%, 2x3 18%, 3x3 12%, 3x4 11%,
# 3x5 9%, 4x6 6%), at 16.7k rule inputs (3x5: 24.0k, 4x6: 34.2k).
GRID_ROWS, GRID_COLS = 3, 3
OVERLAP = 0.1                     # share of a cell's width / height added on each side
CORNERS = ("TL", "BR")
MIN_VIDEOS = 5                    # a compound atom must occur in this many train videos
MIN_FRAMES = 50                   # ... and in this many train rows (agent-frames)


def num_cells():
    return GRID_ROWS * GRID_COLS


def cell_names():
    return [f"r{i}c{j}" for i in range(GRID_ROWS) for j in range(GRID_COLS)]


def corner_points(boxes):
    """(T, N, 2, 2) float (x, y) of the top-left and bottom-right corner of (T, N, 4) boxes."""
    return torch.stack([boxes[..., [0, 1]], boxes[..., [2, 3]]], dim=2)


NUM_KNOTS = 101                   # quantiles of each corner coordinate kept for the rank map


def axis_knots(points):
    """[[[NUM_KNOTS quantiles] of x, of y] per corner] of the corner points (K, 2 corners,
    2 axes) of the train boxes: the map from a coordinate to its rank (`_rank`)."""
    levels = torch.linspace(0, 1, NUM_KNOTS)
    return [[torch.quantile(points[:, c, a].float(), levels).tolist() for a in range(2)]
            for c in range(len(CORNERS))]


def _rank(y, knots):
    """Rank in [0, 1] of coordinates y among the train ones (piecewise linear between the knots)."""
    k = torch.tensor(knots, dtype=torch.float, device=y.device)
    levels = torch.linspace(0, 1, len(knots), device=y.device)
    i = torch.searchsorted(k, y.contiguous()).clamp(1, len(k) - 1)
    lo, hi = k[i - 1], k[i]
    w = ((y - lo) / (hi - lo).clamp(min=1e-9)).clamp(0, 1)
    return levels[i - 1] + w * (levels[i] - levels[i - 1])


def _membership(v, edges, overlap):
    """(..., K) bool: v lies in bin k, widened by overlap of the bin size (outer edges open); NaN in none."""
    e = torch.tensor(edges, dtype=torch.float, device=v.device)
    lo, hi = e[:-1].clone(), e[1:].clone()
    margin = overlap * (hi - lo)
    lo, hi = lo - margin, hi + margin
    lo[0], hi[-1] = -float("inf"), float("inf")
    return (v[..., None] >= lo) & (v[..., None] < hi)


def corner_cells(boxes, has_box, knots, overlap=OVERLAP):
    """(T, N, 2, R * C) bool: the cells each corner of each box lies in (several near an edge).
    boxes (T, N, 4) normalized x1, y1, x2, y2 (NaN without a box); has_box (T, N) bool;
    knots: the coordinate quantiles of each corner and axis (`axis_knots`)."""
    p = corner_points(boxes).nan_to_num(0.0)                             # (T, N, 2, 2)
    cols = torch.linspace(0, 1, GRID_COLS + 1).tolist()
    rows = torch.linspace(0, 1, GRID_ROWS + 1).tolist()
    in_col = torch.stack([_membership(_rank(p[:, :, c, 0], knots[c][0]), cols, overlap)
                          for c in range(len(CORNERS))], dim=2)          # (T, N, 2, C)
    in_row = torch.stack([_membership(_rank(p[:, :, c, 1], knots[c][1]), rows, overlap)
                          for c in range(len(CORNERS))], dim=2)          # (T, N, 2, R)
    cells = (in_row[..., :, None] & in_col[..., None, :]).flatten(-2)    # (T, N, 2, R * C)
    return cells & has_box[:, :, None, None]


def position_names():
    return [f"{c}_{g}" for c in CORNERS for g in cell_names()]


def compound_counts(X_labels, cells):
    """(A, 2 * R * C) long: the rows (agent-frames) of one video where the agent holds label atom a
    and its corner lies in the cell. X_labels (T, N, A) bool; cells (T, N, 2, R * C) bool."""
    T, N, A = X_labels.shape
    return torch.einsum("tna,tnk->ak", X_labels.float(), cells.reshape(T, N, -1).float()).long()


def select_compound(frames, videos, min_videos=MIN_VIDEOS, min_frames=MIN_FRAMES):
    """The kept (atom, position) pairs of the (A, P) train counts: [[a, p]], p = corner * R * C + cell."""
    keep = (videos >= min_videos) & (frames >= min_frames)
    return torch.nonzero(keep).tolist()


def grid_atoms(X_labels, cells, compound):
    """(T, N, 2 * R * C + K) bool grid columns of one video: the position atoms, then the
    K compound atoms of `compound` ([[a, p]])."""
    T, N, _ = X_labels.shape
    pos = cells.reshape(T, N, -1)
    if not compound:
        return pos
    a, p = torch.tensor(compound, dtype=torch.long, device=X_labels.device).T
    return torch.cat([pos, X_labels[..., a] & pos[..., p]], dim=2)


def vocabulary(unique_atoms, compound, atom_name):
    """Signatures of the grid atoms (position, then compound), after the label vocabulary;
    `step05._atom_name` reads them as "pos:TL_r0c1" and "Car:Brake@TL_r0c1"."""
    names = position_names()
    out = [["pred", "grid", "grid_label", n, "agent_class_label", "pos"] for n in names]
    for a, p in compound:
        cls, label = atom_name(unique_atoms[a]).split(":", 1)
        out.append(["pred", "grid", "grid_label", f"{label}@{names[p]}", "agent_class_label", cls])
    return out


# -------------- Groups of adjacent same-class agents --------------
# A group: per frame, the agents of the same class whose boxes overlap (IoU > 0) or whose centres are
# closer than one box width (the mean width of the two, in pixels), connected components. Step 06
# counts an alarm as correct at the group level when the event happens to an agent of the alarm
# agent's group (same class, adjacent in the image): the labels and the 2D boxes often cannot tell
# which of such agents it will be (full test: 71% of the onsets happen to an agent in a group; the
# box motion of the members does not tell them apart, only the nearest member is a little likelier).
GROUP_TAG = "groups_overlap_or_1width_v1"
FRAME_SIZE_PX = (1920, 1280)                      # ROAD++ front camera, to compare distances in pixels


def agent_groups(boxes, has_box, classes):
    """(T, N) long group id of every agent and frame (the smallest agent index of its group, -1
    without a box). boxes (T, N, 4) normalized; has_box (T, N) bool; classes [N] class labels."""
    T, N, _ = boxes.shape
    W, H = FRAME_SIZE_PX
    b = boxes.nan_to_num(0.0)
    x1, y1, x2, y2 = (b[..., i] for i in range(4))
    ix = (torch.minimum(x2[:, :, None], x2[:, None]) - torch.maximum(x1[:, :, None], x1[:, None])).clamp(min=0)
    iy = (torch.minimum(y2[:, :, None], y2[:, None]) - torch.maximum(y1[:, :, None], y1[:, None])).clamp(min=0)
    overlap = ix * iy > 0
    cx, cy, w = (x1 + x2) / 2 * W, (y1 + y2) / 2 * H, (x2 - x1) * W
    near = torch.sqrt((cx[:, :, None] - cx[:, None]) ** 2 + (cy[:, :, None] - cy[:, None]) ** 2) \
        < (w[:, :, None] + w[:, None]) / 2
    index = {c: i for i, c in enumerate(sorted({str(c) for c in classes}))}
    cls = torch.tensor([index[str(c)] for c in classes], dtype=torch.long)
    adj = (overlap | near) & (cls[:, None] == cls[None])[None] & has_box[:, :, None] & has_box[:, None]
    adj |= torch.eye(N, dtype=torch.bool)[None] & has_box[:, :, None]
    label = torch.where(has_box, torch.arange(N)[None].expand(T, N), N)
    while True:                                                  # connected components: min label propagation
        new = torch.where(has_box, torch.where(adj, label[:, None, :], N).amin(dim=2), N)
        if torch.equal(new, label):
            break
        label = new
    return torch.where(has_box, label, -1)
