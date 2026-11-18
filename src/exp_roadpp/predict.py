"""Prediction atoms: when will the ego reach a pedestrian (cyclist), and will their paths cross then,
from the 2D GT boxes of the front camera, as inputs of the TemporalRuleNet.

Why: a rule "the ego brakes near a pedestrian it saw at the junction ahead" (SUMO R5') needs memory when
the pedestrian is hidden at the brake. Without negation the rule language cannot say "seen, now gone";
an atom holding at the sighting that already says WHEN the ego gets there ("reach in 4-8 s") turns it
into a plain lagged rule: Ped:reach_4-8s (other agent, 3-6 s before) -> AV-Brake. Seen once is enough.

Estimation per agent and frame t (causal: frames <= t only; numpy, so it runs without torch):
  distance Z (m, along the camera axis), two monocular estimates that must agree:
    ground plane:  Z_g = f * h_cam / (y_bottom - y_horizon)       (flat road, known camera height)
    known height:  Z_h = f * H_class / (y_bottom - y_top)         (pedestrian ~1.75 m)
    Z = sqrt(Z_g * Z_h) if they agree within AGREE, Z_g alone if the top is cut, else unknown
    (a disagreement means a cut box, a slope or a wrong height: no atom rather than a wrong one);
  lateral X (m, right of the camera axis) = Z * (u_centre - c_x) / f (the bearing is precise; X inherits
    the error of Z);
  velocities vZ, vX (m/s, relative to the camera: the ego's motion is included, no ego speed needed) and
    the smoothed Z, X at t: least squares over the last WINDOW frames (>= MIN_VALID with an estimate);
  reach = (Z - Z_FRONT) / closing speed (-vZ), for agents within REACH_LATERAL of the axis that close in;
  conflict: at the reach time the agent is in the ego's corridor (|X + vX * reach| <= CORRIDOR_HALF,
    widened by |vX| * TOL_S for the timing error) under constant relative velocity.
Unknown (no atom) without a box, for a box at the image border or smaller than MIN_BOX_H_PX, beyond
MAX_DIST, with fewer than MIN_VALID estimates in the window, and while the ego turns (AV-TurLft /
AV-TurRht in the last TURN_LOOKBACK frames: the relative motion is not straight then).
Atoms (inputs only, never heads; every scope, so the ego's rules read them of the other agents):
  "<class>:reach_<a>-<b>s" and "<class>:conflict_<a>-<b>s" for the BINS, with hysteresis (HYST) and a
  debounce of DEBOUNCE frames, like the motion atoms (every flicker would be a fake onset).
The camera constants are the Waymo front camera's, which the SUMO camera (`synth/sumo/observe.py`)
copies; on roadpp they are approximate (mounting height, pitch). `predict_check` measures the errors
against the SUMO world states."""
import math

import numpy as np

VERSION = "v1"
CLASSES = ("Ped", "Cyc")
HEIGHT_M = {"Ped": 1.75, "Cyc": 1.7}            # SUMO: traffic.PED_SIZE height; a cyclist about as tall
CAMERA = {"focal_px": 2055.0, "cam_height_m": 2.0, "horizon_y": 0.5, "image_px": (1920, 1280), "fps": 10}
BINS = ((0.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0))
KINDS = ("reach", "conflict")
WINDOW = 10                 # frames of the velocity fit (1 s)
MIN_VALID = 7               # estimates the fit needs in its window
MAX_DIST = 50.0             # m
MIN_BOX_H_PX = 15.0
EDGE = 0.005                # normalized: a box this close to the border is cut
AGREE = 0.35                # |Z_g / Z_h - 1| allowed (log ratio <= log(1 + AGREE))
Z_FRONT = 2.0               # m from the camera to the ego's front (the camera is 1.5 m behind it)
CLOSING_MIN = 0.5           # m/s: slower closing has no reach time
REACH_LATERAL = 10.0        # m off the axis: farther agents get no reach atom
CORRIDOR_HALF = 2.0         # m: the ego's lane, half its width plus a margin
TOL_S = 1.0                 # s of timing error the corridor test allows
HYST = 0.1                  # a bin is left only 10% past its edges
DEBOUNCE = 3                # frames a bin must hold in a row
TURN_LOOKBACK = 10          # frames
EGO_TURN_LABELS = ("AV-TurLft", "AV-TurRht")


def tag():
    return f"_pred{VERSION}"


def atom_names():
    return [f"{k}_{a:g}-{b:g}s" for k in KINDS for a, b in BINS]


def agent_classes(unique_atoms):
    """[(agent_class, label)] of the CLASSES in the label vocabulary, by class id."""
    out = set()
    for signature in unique_atoms:
        fields = dict(zip(signature[0::2], signature[1::2]))
        if fields.get("agent_class_label") in CLASSES:
            out.add((fields["agent_class"], fields["agent_class_label"]))
    return sorted(out)


def vocabulary(unique_atoms):
    """Signatures of the prediction atoms, appended after the other atoms; `step05._atom_name` reads them
    as "Ped:reach_2-4s"."""
    return [["pred", "predict", "predict_label", name, "agent_class", cls, "agent_class_label", label]
            for cls, label in agent_classes(unique_atoms) for name in atom_names()]


def ego_turn_atoms(unique_atoms):
    return [a for a, sig in enumerate(unique_atoms)
            if dict(zip(sig[0::2], sig[1::2])).get("action_id_label") in EGO_TURN_LABELS]


def _distance(boxes, heights, cam):
    """Z, X (T, N) in m from (T, N, 4) normalized boxes (NaN: none); heights (N,) m (NaN: no class)."""
    W, H = cam["image_px"]
    f, cx, cy = cam["focal_px"], W / 2, cam["horizon_y"] * H
    x1, y1, x2, y2 = (boxes[..., i] for i in range(4))
    with np.errstate(invalid="ignore", divide="ignore"):
        u1, v1, u2, v2 = x1 * W, y1 * H, x2 * W, y2 * H
        ok = (np.isfinite(boxes).all(axis=-1) & np.isfinite(heights)[None, :] & (v2 - v1 >= MIN_BOX_H_PX)
              & (x1 > EDGE) & (x2 < 1 - EDGE))
        bottom = ok & (y2 < 1 - EDGE) & (v2 - cy > 2.0)
        top = y1 > EDGE
        z_g = np.where(bottom, f * cam["cam_height_m"] / (v2 - cy), np.nan)
        z_h = np.where(bottom & top, f * heights[None, :] / (v2 - v1), np.nan)
        agree = np.abs(np.log(z_g / z_h)) <= math.log(1 + AGREE)
        Z = np.where(bottom & top, np.where(agree, np.sqrt(z_g * z_h), np.nan), z_g)
        Z = np.where(Z <= MAX_DIST, Z, np.nan)
        X = Z * ((u1 + u2) / 2 - cx) / f
    return Z, X


def _fit(values):
    """Least squares over the last WINDOW frames of (T, N) values (NaN: none): (value at t, slope per frame,
    fitted) where fitted needs >= MIN_VALID values and one at t itself."""
    T, N = values.shape
    padded = np.concatenate([np.full((WINDOW - 1, N), np.nan), values])
    win = np.lib.stride_tricks.sliding_window_view(padded, WINDOW, axis=0)          # (T, N, WINDOW)
    m = np.isfinite(win)
    y = np.where(m, win, 0.0)
    x = np.arange(WINDOW, dtype=float) - (WINDOW - 1)                               # frames before t: -9..0
    c = m.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = (m * x).sum(axis=-1) / c
        ym = y.sum(axis=-1) / c
        dx = np.where(m, x - xm[..., None], 0.0)
        sxx = (dx * dx).sum(axis=-1)
        slope = (dx * (y - ym[..., None])).sum(axis=-1) / sxx
        now = ym - slope * xm
    fitted = (c >= MIN_VALID) & (sxx > 0) & m[..., -1]
    return np.where(fitted, now, np.nan), np.where(fitted, slope, np.nan), fitted


def estimate(boxes, classes, ego_turning=None, cam=CAMERA):
    """{"Z", "X", "vZ", "vX", "reach" (s), "conflict" (bool), "known" (bool)}, each (T, N), of (T, N, 4)
    normalized boxes (NaN where none; the ego's column NaN), classes [N] class labels, ego_turning (T,)."""
    boxes = np.asarray(boxes, dtype=float)
    T, N, _ = boxes.shape
    heights = np.array([HEIGHT_M.get(c, np.nan) for c in classes], dtype=float)
    Z, X = _distance(boxes, heights, cam)
    Zs, dZ, fz = _fit(Z)
    Xs, dX, fx = _fit(X)
    known = fz & fx
    if ego_turning is not None:
        turning = np.asarray(ego_turning, dtype=bool)
        recent = np.zeros(T, dtype=bool)
        for k in range(TURN_LOOKBACK):
            recent[k:] |= turning[:T - k]
        known &= ~recent[:, None]
    vZ, vX = dZ * cam["fps"], dX * cam["fps"]
    closing = -vZ
    with np.errstate(invalid="ignore", divide="ignore"):
        reachable = known & (closing > CLOSING_MIN) & (np.abs(Xs) <= REACH_LATERAL)
        reach = np.where(reachable, np.maximum(Zs - Z_FRONT, 0.0) / closing, np.nan)
        conflict = reachable & (np.abs(Xs + vX * reach) <= CORRIDOR_HALF + np.abs(vX) * TOL_S)
    return {"Z": np.where(known, Zs, np.nan), "X": np.where(known, Xs, np.nan), "vZ": np.where(known, vZ, np.nan),
            "vX": np.where(known, vX, np.nan), "reach": reach, "conflict": conflict, "known": known}


def bin_index(values):
    """(T, N) int bin of BINS per frame, -1 for none, with hysteresis (HYST) along t."""
    lo = np.array([a for a, _b in BINS])
    hi = np.array([b for _a, b in BINS])
    T, N = values.shape
    out = np.full((T, N), -1, dtype=int)
    prev = np.full(N, -1, dtype=int)
    for t in range(T):
        v = values[t]
        finite = np.isfinite(v)
        raw = np.full(N, -1, dtype=int)
        for b in range(len(BINS)):
            raw[finite & (v >= lo[b]) & (v < hi[b])] = b
        p = np.maximum(prev, 0)
        keep = (prev >= 0) & finite & (v >= lo[p] * (1 - HYST)) & (v < hi[p] * (1 + HYST))
        cur = np.where(keep, prev, raw)
        out[t] = cur
        prev = cur
    return out


def _debounced(bins):
    """(T, N, len(BINS)) bool: bin b holds at t if it held in the DEBOUNCE frames t - DEBOUNCE + 1 .. t."""
    T, N = bins.shape
    out = np.zeros((T, N, len(BINS)), dtype=bool)
    for b in range(len(BINS)):
        on = bins == b
        held = on.copy()
        for k in range(1, DEBOUNCE):
            shifted = np.zeros_like(on)
            shifted[k:] = on[:-k]
            held &= shifted
        out[..., b] = held
    return out


def atoms(est):
    """(T, N, len(atom_names())) bool atoms of one `estimate` (KINDS x BINS)."""
    reach = _debounced(bin_index(est["reach"]))
    conflict = _debounced(bin_index(np.where(est["conflict"], est["reach"], np.nan)))
    return np.concatenate([reach, conflict], axis=-1)


def predict_atoms(boxes, has_box, classes, ego_turning, unique_atoms, ego=None, cam=CAMERA):
    """(T, N, C * K) bool columns of one video, in the order of `vocabulary(unique_atoms)`: the atoms of
    each agent of a CLASSES class in its class's columns; others and the ego get none.
    boxes (T, N, 4) normalized, has_box (T, N) bool (`step05._agent_boxes`), classes [N] labels."""
    boxes = np.array(boxes, dtype=float)
    boxes[~np.asarray(has_box, dtype=bool)] = np.nan
    if ego is not None:
        boxes[:, ego] = np.nan
    T, N, _ = boxes.shape
    names = atom_names()
    class_list = agent_classes(unique_atoms)
    column = {label: c for c, (_cls, label) in enumerate(class_list)}
    out = np.zeros((T, N, len(class_list) * len(names)), dtype=bool)
    if not class_list or T == 0:
        return out
    est = atoms(estimate(boxes, classes, ego_turning, cam))
    for n, label in enumerate(classes):
        c = column.get(label)
        if c is None or n == ego:
            continue
        out[:, n, c * len(names):(c + 1) * len(names)] = est[:, n]
    return out
