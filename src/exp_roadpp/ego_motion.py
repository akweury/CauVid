"""Ego motion atoms: how fast the ego drives and how it closes in on the agents in front, read from
the boxes of the scene, as inputs of the TemporalRuleNet.

Why: the rule net and the lookup alarm AV-Stop many seconds early. The timing diagnostic
(`step05_timing_diag`, train, rows 0-2 s vs 4-10 s before the onset, AUC late vs early) found no
timing in the ego's label atoms (AV-Mov etc.: 0.49 on medium and full) but a clear one in the scene:
the median box growth of the ground agents (the scene expands while the ego drives forward) 0.15 /
0.17, falling steadily from the background level 10 s before the stop to 0; the lead agent's box
size 0.68 / 0.63 and growth 0.32 / 0.33; the change of the distance to the nearest agent 0.66 / 0.70.
The ego has no box, so `motion.py` gives it no motion atom; these are its motion atoms.

Signals per frame (`signals`; NaN where undefined):
  scene_scale:  median over the ground agents with a box of their growth rate (`motion.signals`
                scale): ~ the ego's forward speed (static agents grow by it, the others too on average);
  scene_outflow: median outward lateral flow (lateral * side of the image), its second witness;
  ahead_log_area, ahead_scale: the ego's lead agent (`roles.role_partners` "ahead"): its box size
                (near / far) and growth rate (closing in);
  near_rate:    the change per frame of the ground distance (in box heights) from the ego (at
                roles.EGO_BOX) to its nearest ground agent, to that same agent K frames earlier;
  tl_log_area, tl_scale, tl_lateral, tl_vertical: the traffic light in view (class in TL_CLASSES; with
                several, the largest box): its box size (near / far), growth rate (> 0: the ego comes
                closer), horizontal drift (box widths per frame, > 0 right) and vertical drift of its box
                bottom (box heights per frame, > 0 down) (`motion.signals`). Traffic lights are not
                ground agents, so no other signal sees them. A box cut by the image border (`motion.EDGE`)
                has no rates there (its size and centre are cut), only a size.
The rate signals are averaged over the last SMOOTH frames.
Atoms (only the ego holds them; inputs only, never heads, `step05.is_input_atom`), each a level of
one signal, the levels cut at the train tertiles (`fit`):
  "speed_still": |scene_scale| and |scene_outflow| below STILL; else "speed_slow" / "speed_medium" /
      "speed_fast" by the tertiles of scene_scale over the train frames that are not still;
  "ahead_near" / "ahead_mid" / "ahead_far": the lead agent's box size;
  "ahead_closing_fast" / "ahead_closing_slow" / "ahead_steady": its growth rate (high: closing in);
  "near_closing_fast" / "near_closing_slow" / "near_steady": the distance change (low: closing in);
  "tl_far" / "tl_mid" / "tl_near": the traffic light's box size (train tertiles);
Signed atoms of the traffic light's rates (SIGNED, a dead band SIGNED_BAND around 0, not tertiles: the
direction is the fact): "tl_shrink" / "tl_size_steady" / "tl_grow", "tl_left" / "tl_lateral_steady" /
"tl_right", "tl_up" / "tl_vertical_steady" / "tl_down" (in the image).
A level holds at t only if it held in the motion.DEBOUNCE frames up to t (no flicker at a cut).
The rules read the levels per memory slot, so a change ("fast 3 s ago, slow now") is a rule of
two atoms in two slots. The ego's atoms are not pooled away in the other-agent scope (only the ego
holds them), so the other agents' rules may read them too ("the ego slows down").
"""
import torch

from src.exp_roadpp import motion as agent_motion, roles as agent_roles

SMOOTH = 5                                  # frames of the moving average of the rate signals
STILL = {"scene_scale": 0.002, "scene_outflow": 0.004}  # the median of many boxes jitters less than one
QUANTILES = (1 / 3, 2 / 3)
MAX_SAMPLES = 1_000_000
VERSION = "v2"                              # v2: the traffic light atoms
TL_CLASSES = ("TL",)                        # the ego's traffic light (ROAD++ "OthTL": the other directions)
SIGNALS = ("scene_scale", "scene_outflow", "ahead_log_area", "ahead_scale", "near_rate",
           "tl_log_area", "tl_scale", "tl_lateral", "tl_vertical")
LEVELS = {                                  # signal: atom names from low to high values
    "scene_scale": ("speed_slow", "speed_medium", "speed_fast"),
    "ahead_log_area": ("ahead_far", "ahead_mid", "ahead_near"),
    "ahead_scale": ("ahead_steady", "ahead_closing_slow", "ahead_closing_fast"),
    "near_rate": ("near_closing_fast", "near_closing_slow", "near_steady"),
    "tl_log_area": ("tl_far", "tl_mid", "tl_near"),
}
SIGNED = {                                  # signal: atom names (below -band, within the band, above +band)
    "tl_scale": ("tl_shrink", "tl_size_steady", "tl_grow"),
    "tl_lateral": ("tl_left", "tl_lateral_steady", "tl_right"),
    "tl_vertical": ("tl_up", "tl_vertical_steady", "tl_down"),
}
# Dead bands of the smoothed rates, per frame: motion.STILL (the jitter of a GT box that stands still).
SIGNED_BAND = {"tl_scale": agent_motion.STILL["scale"], "tl_lateral": agent_motion.STILL["lateral"],
               "tl_vertical": agent_motion.STILL["vertical"]}
ATOMS = (("speed_still",) + tuple(a for levels in LEVELS.values() for a in levels)
         + tuple(a for levels in SIGNED.values() for a in levels))


def tag():
    return f"_ego{SMOOTH}s{VERSION}"


def names():
    return [f"av:{a}" for a in ATOMS]


def vocabulary():
    """Signatures of the ego motion atoms, after the motion atoms; `step05._atom_name` reads them as
    "av:speed_still"."""
    return [["pred", "ego_motion", "ego_motion_label", a, "agent_class_label", "av"] for a in ATOMS]


# -------------- Signals --------------
def scene_median(values, mask):
    """(T,) median over the agents of (T, N) values where mask, NaN where none."""
    return torch.nanmedian(torch.where(mask, values, float("nan")), dim=1).values


def smooth(v, n=SMOOTH):
    """(T,) mean of the non-NaN values of v over the frames t - n + 1 .. t, NaN where none."""
    T = len(v)
    ok = ~torch.isnan(v)
    sums = torch.cat([v.new_zeros(1), torch.where(ok, v, 0.0).cumsum(dim=0)])
    counts = torch.cat([v.new_zeros(1), ok.float().cumsum(dim=0)])
    hi = torch.arange(1, T + 1)
    lo = (hi - n).clamp(min=0)
    c = counts[hi] - counts[lo]
    return torch.where(c > 0, (sums[hi] - sums[lo]) / c.clamp(min=1), float("nan"))


def nearest_rate(boxes, ok, ground, ego):
    """(T,) the change per frame of the ego's ground distance (in box heights of the other agent)
    to its nearest ground agent, to the same agent motion.K frames earlier."""
    T, N, _ = boxes.shape
    K = agent_motion.K
    ex, ey = (agent_roles.EGO_BOX[0] + agent_roles.EGO_BOX[2]) / 2, agent_roles.EGO_BOX[3]
    x1, y1, x2, y2 = boxes.float().unbind(dim=2)
    cand = ok & ground[None]
    cand[:, ego] = False
    d = torch.sqrt(((x1 + x2) / 2 - ex) ** 2 + (y2 - ey) ** 2) / (y2 - y1).clamp(min=0.02)
    d = torch.where(cand, d, float("inf"))                                         # (T, N)
    best, m = d.min(dim=1)
    rate = torch.full((T,), float("nan"))
    if T > K:
        prev = d[torch.arange(K, T) - K, m[K:]]
        ok_pair = torch.isfinite(best[K:]) & torch.isfinite(prev)
        rate[K:] = torch.where(ok_pair, (best[K:] - prev) / K, float("nan"))
    return rate


def signals(boxes, has_box, classes, ego, X_labels, unique_atoms):
    """{signal: (T,) float} of SIGNALS of one video (boxes (T, N, 4) normalized, NaN without a box;
    has_box (T, N); classes [N]; ego the ego's index; X_labels (T, N, A) the label atoms)."""
    T, N, _ = boxes.shape
    boxes, has_box = boxes.cpu().float(), has_box.cpu()
    sig = agent_motion.signals(boxes, has_box)                                      # (T, N, 4)
    scale, lateral = sig[..., 0], sig[..., 2]
    W, H = agent_motion.FRAME_SIZE_PX
    x1, y1, x2, y2 = boxes.unbind(dim=2)
    ok = has_box & ~torch.isnan(boxes).any(dim=2)
    ground = torch.tensor([c not in agent_roles.NON_GROUND_CLASSES for c in classes], dtype=torch.bool)
    valid = ~torch.isnan(scale) & ground[None]
    out = {"scene_scale": smooth(scene_median(scale, valid)),
           "scene_outflow": smooth(scene_median(lateral * torch.sign((x1 + x2) / 2 - 0.5), valid))}
    lane = agent_roles.ego_lane(X_labels, unique_atoms)
    partner = agent_roles.role_partners(boxes, classes, ego, ("ahead",), lane)[:, ego, 0]       # (T,)
    has = partner >= 0
    idx, frames = partner.clamp(min=0), torch.arange(T)
    log_area = torch.log(((x2 - x1) * W).clamp(min=1.0) * ((y2 - y1) * H).clamp(min=1.0))
    out["ahead_log_area"] = torch.where(has & ok[frames, idx], log_area[frames, idx], float("nan"))
    out["ahead_scale"] = smooth(torch.where(has, scale[frames, idx], float("nan")))
    out["near_rate"] = smooth(nearest_rate(boxes, ok, ground, ego))
    # The traffic light in view: the largest box of a TL_CLASSES agent per frame.
    is_tl = torch.tensor([c in TL_CLASSES for c in classes], dtype=torch.bool)
    tl_area = torch.where(ok & is_tl[None], log_area, float("-inf"))
    best, m = tl_area.max(dim=1)
    has_tl = torch.isfinite(best)
    out["tl_log_area"] = torch.where(has_tl, best, float("nan"))
    for name, i in (("tl_scale", 0), ("tl_lateral", 2), ("tl_vertical", 3)):
        out[name] = smooth(torch.where(has_tl, sig[frames, m, i], float("nan")))
    return out


# -------------- Thresholds and atoms --------------
def _still(sig):
    return (sig["scene_scale"].abs() < STILL["scene_scale"]) & (sig["scene_outflow"].abs() < STILL["scene_outflow"])


def fit(samples):
    """{"cuts": {signal: [low cut, high cut]}, "counts": {signal: train frames}} of the train signals
    {signal: (M,) float} (the ego frames of every train video, `signals`); the speed cuts over the
    frames that are not still."""
    spec = {"cuts": {}, "counts": {}}
    still = _still(samples)
    generator = torch.Generator().manual_seed(0)
    for s in LEVELS:
        v = samples[s]
        keep = ~torch.isnan(v) & (~still if s == "scene_scale" else torch.ones_like(still))
        v = v[keep]
        if len(v) > MAX_SAMPLES:
            v = v[torch.randperm(len(v), generator=generator)[:MAX_SAMPLES]]
        spec["counts"][s] = len(v)
        spec["cuts"][s] = torch.quantile(v, torch.tensor(QUANTILES)).tolist() if len(v) else [0.0, 0.0]
    return spec


def ego_atoms(sig, spec):
    """(T, len(ATOMS)) bool atoms of the ego of one video from its `signals` and the `fit` spec."""
    still = _still(sig)
    cond = [still]
    for s, levels in LEVELS.items():
        v, (lo, hi) = sig[s], spec["cuts"][s]
        moving = ~still if s == "scene_scale" else torch.ones_like(still)
        cond += [moving & (v < lo), moving & (v >= lo) & (v < hi), moving & (v >= hi)]   # NaN: all False
    for s in SIGNED:
        v, band = sig[s], SIGNED_BAND[s]
        cond += [v < -band, v.abs() <= band, v > band]                             # NaN: all False
    cond = torch.stack(cond, dim=1)[:, None]                                        # (T, 1, M)
    return agent_motion._debounce(cond)[:, 0]
