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
                closer), horizontal and vertical drift of its box centre (> 0 right / down). A traffic
                light box is small: over motion.K = 0.5 s it barely changes and a few pixels of jitter
                divided by its width look like a large move. So its rates are taken over TL_K frames
                (2 s), the drifts in image fractions (not box widths), per frame, then smoothed over SMOOTH
                frames. Traffic lights are not ground agents, so no other signal sees them. A box cut by
                the image border (`motion.EDGE`) has no rates there, only a size.
The rate signals are averaged over the last SMOOTH frames.
Atoms (only the ego holds them; inputs only, never heads, `step05.is_input_atom`), each a level of
one signal, the levels cut at the train tertiles (`fit`):
  "speed_still": |scene_scale| and |scene_outflow| below STILL; else "speed_slow" / "speed_medium" /
      "speed_fast" by the tertiles of scene_scale over the train frames that are not still;
  "ahead_near" / "ahead_mid" / "ahead_far": the lead agent's box size;
  "ahead_closing_fast" / "ahead_closing_slow" / "ahead_steady": its growth rate (high: closing in);
  "near_closing_fast" / "near_closing_slow" / "near_steady": the distance change (low: closing in);
  "tl_far" / "tl_mid" / "tl_near": the traffic light's box size (train tertiles);
Signed atoms of the traffic light's rates (SIGNED: the direction is the fact; the dead band around 0 is the
SIGNED_Q quantile of |rate| over the train frames with a traffic light, `fit`, so it follows the size of
these small changes instead of a fixed jitter band): "tl_shrink" / "tl_size_steady" / "tl_grow", "tl_left" / "tl_lateral_steady" /
"tl_right", "tl_up" / "tl_vertical_steady" / "tl_down" (in the image).
Turn atoms (v4, 2026-11-23; `turn_probe.py` on sumo_maps_v7, val AV-TurLft: 20% of the onsets end a brake during
which the ego already turned for ~2 s, yaw ~25 deg/s, but no input said so): "scene_drift", the median over the
agents' boxes (uncut by the image border, all classes but the ego) of the horizontal move of the box centre over
DRIFT_K frames, per frame, in image widths. A yaw moves every point of the image the same way (the forward motion
moves them outward, which cancels in the median of both sides), so a left turn moves the scene right. Signed atoms
"turn_right" / "turn_straight" / "turn_left" (the scene moves left / little / right), the dead band the
SIGNED_Q_OF["scene_drift"] quantile of |drift| over the train frames (a turn is the exception).
v5 (2026-11-23, the user on the failure clips: before a left turn from a stop the ego moves left a few frames
before the label, and many objects move right together at a similar speed; `turn_fail_clips`: the v4 median was
pulled by the crossing traffic, often to the wrong side, and the 5-frame mean plus the debounce made turn_left hold
~4 frames late, at or after the onset):
  - per box the horizontal move over DRIFT_K frames minus the forward motion's outflow ((cx - 0.5) x the change
    of log box width: a static box moves away from the image centre as it grows), per frame;
  - the drift is the common fate of the boxes: the largest set of boxes whose moves agree (within DRIFT_TOL_ABS
    or DRIFT_TOL_REL of each other), its mean, if it has >= DRIFT_MIN boxes and >= DRIFT_SHARE of the boxes;
    NaN else. Crossing traffic and single movers fall out;
  - no moving average and no debounce for the turn atoms (NO_DEBOUNCE).
Layout atoms (v6, 2026-11-23, the user: the strongest cue of a left turn is a left-turn lane: no vehicle of the
same direction on the ego's left, the oncoming traffic right next to it on the left, more agents on the right
and close together, maybe vehicles turning right on the right). From the boxes and the other agents' labels
(SUMO: lanes relative to the ego, OutgoLane = same direction, IncomLane = oncoming; blinkers and turns), per frame,
a side being the box centre left of 0.5 - SIDE_MARGIN or right of 0.5 + SIDE_MARGIN of the image width:
  "lay_left_same" / "lay_right_same"   a vehicle of the same direction (OutgoLane) on that side;
  "lay_left_oncoming"                  an oncoming vehicle (IncomLane) on the left;
  "lay_left_heavy" / "lay_right_heavy" >= HEAVY_MIN ground agents in view and >= HEAVY_SHARE of them on that side;
  "lay_left_turner" / "lay_right_turner"  a vehicle turning or blinking (TurLft / IncatLft, TurRht / IncatRht)
                                       on that side.
A left-turn lane reads "!lay_left_same & lay_left_oncoming & lay_right_heavy".
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
VERSION = "v6"                              # v2: the traffic light atoms; v3: their 2 s rates, fitted bands;
                                            # v4: the turn atoms (scene drift); v5: drift = common fate, no lag;
                                            # v6: the layout atoms (lay_*)
TL_CLASSES = ("TL",)                        # the ego's traffic light (ROAD++ "OthTL": the other directions)
SIGNALS = ("scene_scale", "scene_outflow", "ahead_log_area", "ahead_scale", "near_rate",
           "tl_log_area", "tl_scale", "tl_lateral", "tl_vertical", "scene_drift")
LAYOUT = ("lay_left_same", "lay_right_same", "lay_left_oncoming", "lay_left_heavy", "lay_right_heavy",
          "lay_left_turner", "lay_right_turner")                                     # v6: 0 / 1 signals
SIGNALS = SIGNALS + LAYOUT
SIDE_MARGIN = 0.05                          # v6: a box centre within 0.5 +- this is on neither side
HEAVY_MIN, HEAVY_SHARE = 3, 0.7             # v6: agents in view and the share on one side
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
    "scene_drift": ("turn_right", "turn_straight", "turn_left"),   # v4: the scene moves left / little / right
}
DRIFT_K = 2                                 # frames of the box moves of scene_drift (as the shape atoms)
DRIFT_TOL_ABS = 0.002                       # v5: two moves agree within this (image widths / frame) ..
DRIFT_TOL_REL = 0.25                        # .. or this share of the larger one
DRIFT_MIN, DRIFT_SHARE = 2, 1 / 3           # v5: the common set needs this many boxes and this share of them
NO_DEBOUNCE = ("scene_drift",)              # v5: the turn atoms hold from the first frame (no lag)
TL_K = 20                                   # frames (2 s) of the traffic light's rate differences
SIGNED_Q = 1 / 3                            # dead band: this quantile of |rate| (train frames with a light)
SIGNED_Q_OF = {"scene_drift": 0.8}          # v4: a turn moves the scene in few frames, the band is wider
ATOMS = (("speed_still",) + tuple(a for levels in LEVELS.values() for a in levels)
         + tuple(a for levels in SIGNED.values() for a in levels) + LAYOUT)


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
    out.update(tl_rates(boxes, ok, best, m, has_tl))
    out["scene_drift"] = scene_drift(boxes, ok, ego)                               # v5: no moving average
    out.update(layout(boxes, ok, classes, ego, X_labels, unique_atoms))
    return out


def layout(boxes, ok, classes, ego, X_labels, unique_atoms):
    """{LAYOUT name: (T,) float 0 / 1} v6: where the other agents are around the ego (module doc)."""
    T, N, _ = boxes.shape
    x1, _y1, x2, _y2 = boxes.unbind(dim=2)
    cx = (x1 + x2) / 2
    seen = ok.clone()
    if 0 <= ego < N:
        seen[:, ego] = False
    ground = torch.tensor([c not in agent_roles.NON_GROUND_CLASSES for c in classes], dtype=torch.bool)
    vehicle = ground & torch.tensor([c not in ("Ped",) for c in classes], dtype=torch.bool)
    left = seen & (cx < 0.5 - SIDE_MARGIN)
    right = seen & (cx > 0.5 + SIDE_MARGIN)

    def holds(labels, keys):
        return agent_roles._holds_any(X_labels, unique_atoms, labels, keys).to(seen.device)

    outgo = holds(("OutgoLane",), ("location_name_label",))
    incom = holds(("IncomLane",), ("location_name_label",))
    turn_l = holds(("TurLft", "IncatLft"), ("action_id_label",))
    turn_r = holds(("TurRht", "IncatRht"), ("action_id_label",))
    v = vehicle[None]
    n_ground = (seen & ground[None]).sum(dim=1)
    n_left, n_right = (left & ground[None]).sum(dim=1), (right & ground[None]).sum(dim=1)
    many = n_ground >= HEAVY_MIN
    out = {
        "lay_left_same": (left & v & outgo).any(dim=1),
        "lay_right_same": (right & v & outgo).any(dim=1),
        "lay_left_oncoming": (left & v & incom).any(dim=1),
        "lay_left_heavy": many & (n_left >= HEAVY_SHARE * n_ground),
        "lay_right_heavy": many & (n_right >= HEAVY_SHARE * n_ground),
        "lay_left_turner": (left & v & turn_l).any(dim=1),
        "lay_right_turner": (right & v & turn_r).any(dim=1),
    }
    return {k: x.float() for k, x in out.items()}


def common_move(move, valid):
    """(F,) v5: per frame the mean move of the largest set of agents whose moves agree (|m_i - m_j| <=
    max(DRIFT_TOL_ABS, DRIFT_TOL_REL x max(|m_i|, |m_j|))), if it has >= DRIFT_MIN agents and >= DRIFT_SHARE of
    the valid ones; NaN else. move, valid (F, N)."""
    m = torch.where(valid, move, torch.zeros_like(move))
    diff = (m[:, :, None] - m[:, None, :]).abs()
    tol = torch.maximum(torch.full_like(diff, DRIFT_TOL_ABS),
                        DRIFT_TOL_REL * torch.maximum(m.abs()[:, :, None], m.abs()[:, None, :]))
    agree = (diff <= tol) & valid[:, :, None] & valid[:, None, :]                  # (F, i, j)
    count = agree.sum(dim=2)                                                       # (F, i): agents agreeing with i
    best, i = count.max(dim=1)
    members = agree[torch.arange(len(m)), i]                                       # (F, j)
    mean = (m * members).sum(dim=1) / members.sum(dim=1).clamp(min=1)
    n = valid.sum(dim=1)
    ok = (best >= DRIFT_MIN) & (best.float() >= DRIFT_SHARE * n.float())
    return torch.where(ok, mean, torch.full_like(mean, float("nan")))


def scene_drift(boxes, ok, ego, k=DRIFT_K):
    """(T,) v5: the common horizontal move of the agents' boxes (the ego left out, both boxes uncut by the image
    border) from t - k to t, per frame, in image widths, the forward motion's outflow taken out (`common_move`);
    NaN where no common move."""
    T, N, _ = boxes.shape
    E = agent_motion.EDGE
    x1, y1, x2, y2 = boxes.unbind(dim=2)
    uncut = ok & (x1 >= E) & (y1 >= E) & (x2 <= 1 - E) & (y2 <= 1 - E)
    if 0 <= ego < N:
        uncut = uncut.clone()
        uncut[:, ego] = False
    cx = (x1 + x2) / 2
    log_w = torch.log((x2 - x1).clamp(min=1e-4))
    out = torch.full((T,), float("nan"))
    if T > k:
        valid = uncut[k:] & uncut[:-k]
        outflow = (cx[:-k] - 0.5) * (log_w[k:] - log_w[:-k])                     # a static box under forward motion
        move = (cx[k:] - cx[:-k] - outflow) / k
        out[k:] = common_move(move.nan_to_num(0.0), valid)
    return out


def tl_rates(boxes, ok, log_area_best, m, has_tl, k=TL_K):
    """{tl_scale, tl_lateral, tl_vertical: (T,)} of the traffic light m[t] (the largest at t): the change of
    its log box area and of its box centre (image fractions) from t - k to t, per frame, if the same agent had
    an uncut box then (and has one now); smoothed over SMOOTH frames."""
    T = len(m)
    E = agent_motion.EDGE
    x1, y1, x2, y2 = boxes.unbind(dim=2)
    uncut = ok & (x1 >= E) & (y1 >= E) & (x2 <= 1 - E) & (y2 <= 1 - E)
    W, H = agent_motion.FRAME_SIZE_PX
    log_area = torch.log(((x2 - x1) * W).clamp(min=1.0) * ((y2 - y1) * H).clamp(min=1.0))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    out = {n: torch.full((T,), float("nan")) for n in ("tl_scale", "tl_lateral", "tl_vertical")}
    if T > k:
        t = torch.arange(k, T)
        mm = m[k:]
        valid = has_tl[k:] & uncut[t, mm] & uncut[t - k, mm]
        for name, v in (("tl_scale", log_area), ("tl_lateral", cx), ("tl_vertical", cy)):
            out[name][k:] = torch.where(valid, (v[t, mm] - v[t - k, mm]) / k, float("nan"))
    return {n: smooth(v) for n, v in out.items()}


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
    spec["bands"] = {}
    for s in SIGNED:
        v = samples[s][~torch.isnan(samples[s])].abs()
        if len(v) > MAX_SAMPLES:
            v = v[torch.randperm(len(v), generator=generator)[:MAX_SAMPLES]]
        spec["counts"][s] = len(v)
        spec["bands"][s] = float(torch.quantile(v, SIGNED_Q_OF.get(s, SIGNED_Q))) if len(v) else 0.0
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
        v, band = sig[s], spec["bands"][s]
        cond += [v < -band, v.abs() <= band, v > band]                             # NaN: all False
    cond += [sig[a] > 0.5 for a in LAYOUT]                                         # v6: 0 / 1 signals
    cond = torch.stack(cond, dim=1)                                                 # (T, M)
    out = agent_motion._debounce(cond[:, None])[:, 0]
    i = 1 + 3 * len(LEVELS)
    for s in SIGNED:                                                                # v5: no lag for these
        if s in NO_DEBOUNCE:
            out[:, i:i + 3] = cond[:, i:i + 3]
        i += 3
    return out
