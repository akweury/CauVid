"""Motion atoms: how an agent's own 2D box moves, as inputs of the TemporalRuleNet.

Why: on full (run 78e420) a lookup of the agent's own current labels alone matched the rule net
and the MLP (row AP W20: lookup 0.046, rule net 0.036, MLP 0.048): the label atoms say what an
agent does, not how it moves, and the events (Brake, Stop, MovAway, TurRht, ...) are preceded by
motion: a braking car ahead closes in faster, a car about to move away starts to shrink. The old
geometry atoms (`geometry.py`: bearing sector, image edge, bearing drift) said where an agent is
and how it turns; these say how fast it approaches and whether that changes.

Signals per agent and frame t (pixels of the ROAD++ frame, over K frames; NaN without a box in
all the frames used, or with a box at the image border, whose size and centre are cut):
  scale:    (log area(t) - log area(t - K)) / K, the box growth rate per frame: > 0 the agent
            comes closer (or the ego closes in on it), < 0 it moves away;
  accel:    scale(t) - scale(t - K): the growth rate rises (closing in faster: the agent ahead
            brakes, or comes at us) or falls (closing in slower);
  lateral:  (centre x(t) - centre x(t - K)) / (K * width(t)), box widths per frame;
  vertical: (bottom y(t) - bottom y(t - K)) / (K * height(t)), box heights per frame (> 0: the
            box's ground contact moves down the image, towards the camera).
Atoms (inputs only, never heads; own agent scope only, `step05.rule_input_mask`):
  "mot:approach" / "mot:approach_fast" / "mot:recede": scale above / far above / below its
      thresholds; "mot:closing_faster" / "mot:closing_slower": accel above / below;
  "mot:left" / "mot:right": lateral; "mot:down" / "mot:up": vertical;
  "mot:still": every signal within the STILL band (the box stands still in the image).
The thresholds are per agent class (`fit`), quantiles of the train signals, but never closer to
0 than FLOORS (the GT boxes jitter by a few pixels: a quantile of a mostly still class would make
an atom of that jitter); classes with fewer than MIN_CLASS_SAMPLES train signals use the
thresholds of all classes. An atom holds at t only if its condition held in the DEBOUNCE frames
t - DEBOUNCE + 1 .. t, so a signal near its threshold does not flicker (every flicker would be a
fake change). The ego has no box and no motion atom; its own motion is in its labels (AV-*),
which the rules read in the other-agent scope, so a rule can tell "it grows while we drive"
from "it grows while we stand".
"""
import torch

K = 5                                      # frames of each difference (0.5 s at 10 fps)
DEBOUNCE = 3                               # frames a condition must hold in a row
EDGE = 0.005                               # normalized: a box this close to the border is cut
FRAME_SIZE_PX = (1920, 1280)
SIGNALS = ("scale", "accel", "lateral", "vertical")
ATOMS = ("approach", "approach_fast", "recede", "closing_faster", "closing_slower",
         "left", "right", "down", "up", "still")
# Train quantiles of each signal: (low, high[, very high]).
QUANTILES = {"scale": (0.2, 0.8, 0.95), "accel": (0.1, 0.9), "lateral": (0.1, 0.9), "vertical": (0.1, 0.9)}
# Smallest |threshold| per signal (per frame): above the jitter of the GT boxes (2 px on a 100 px
# box over K = 5 frames: ~0.008 in scale).
FLOORS = {"scale": 0.01, "accel": 0.01, "lateral": 0.02, "vertical": 0.01}
# "still": |scale|, |lateral| and |vertical| below these.
STILL = {"scale": 0.006, "lateral": 0.01, "vertical": 0.008}
MIN_CLASS_SAMPLES = 5000
MAX_SAMPLES = 1_000_000                    # per class, for the quantiles
VERSION = "v1"


def tag():
    return f"_mot{K}d{DEBOUNCE}{VERSION}"


def names():
    return [f"mot:{a}" for a in ATOMS]


def vocabulary():
    """Signatures of the motion atoms, after the label (and grid) vocabulary; `step05._atom_name`
    reads them as "mot:approach"."""
    return [["pred", "motion", "motion_label", a, "agent_class_label", "mot"] for a in ATOMS]


def signals(boxes, has_box):
    """(T, N, 4) float SIGNALS of (T, N, 4) normalized boxes, NaN where undefined."""
    T, N, _ = boxes.shape
    W, H = FRAME_SIZE_PX
    b = boxes.float()
    x1, y1, x2, y2 = (b[..., i] for i in range(4))
    cut = (x1 < EDGE) | (y1 < EDGE) | (x2 > 1 - EDGE) | (y2 > 1 - EDGE)
    ok = has_box & ~cut & ~torch.isnan(b).any(dim=2)
    w, h = ((x2 - x1) * W).clamp(min=1.0), ((y2 - y1) * H).clamp(min=1.0)
    log_area, cx, bottom = torch.log(w * h), (x1 + x2) / 2 * W, y2 * H
    nan = torch.full((T, N), float("nan"))
    out = torch.full((T, N, len(SIGNALS)), float("nan"))
    if T <= K:
        return out
    pair = ok[K:] & ok[:-K]                                                 # (T - K, N)

    def diff(v):
        d = nan.clone()
        d[K:] = torch.where(pair, v[K:] - v[:-K], float("nan"))
        return d
    scale = diff(log_area) / K
    out[..., 0] = scale
    out[K:, :, 1] = scale[K:] - scale[:-K]                                  # NaN propagates
    out[..., 2] = diff(cx) / (K * w)
    out[..., 3] = diff(bottom) / (K * h)
    return out


def _thresholds(values):
    """{signal: [low, high(, very high)]} of (M, 4) signals (NaN-free per column), with the FLOORS."""
    out = {}
    for i, s in enumerate(SIGNALS):
        v = values[:, i]
        v = v[~torch.isnan(v)]
        if len(v) > MAX_SAMPLES:
            v = v[torch.randperm(len(v), generator=torch.Generator().manual_seed(0))[:MAX_SAMPLES]]
        q = torch.quantile(v, torch.tensor(QUANTILES[s])).tolist() if len(v) else [0.0] * len(QUANTILES[s])
        floor = FLOORS[s]
        t = [min(q[0], -floor), max(q[1], floor)]
        if len(q) > 2:
            t.append(max(q[2], 2 * floor, t[1]))
        out[s] = t
    return out


def fit(samples):
    """The thresholds of the train signals {class: (M, 4) float}: {"classes": {class: thresholds},
    "all": thresholds, "counts": {class: M}}; a class with < MIN_CLASS_SAMPLES rows gets "all"."""
    every = torch.cat(list(samples.values())) if samples else torch.zeros((0, len(SIGNALS)))
    spec = {"all": _thresholds(every), "classes": {}, "counts": {}}
    for cls, values in samples.items():
        n = int((~torch.isnan(values[:, 0])).sum())
        spec["counts"][cls] = n
        if n >= MIN_CLASS_SAMPLES:
            spec["classes"][cls] = _thresholds(values)
    return spec


def _debounce(cond):
    """(T, N, M) bool: cond held in each of the DEBOUNCE frames up to t."""
    out = cond.clone()
    for d in range(1, DEBOUNCE):
        shifted = torch.zeros_like(cond)
        shifted[d:] = cond[:-d]
        out &= shifted
    return out


def motion_atoms(boxes, has_box, classes, spec):
    """(T, N, len(ATOMS)) bool motion atoms of one video. boxes (T, N, 4) normalized (NaN without a
    box), has_box (T, N) bool, classes [N] the agents' class labels, spec (`fit`)."""
    sig = signals(boxes.cpu(), has_box.cpu())                              # (T, N, 4)
    T, N, _ = sig.shape
    th = torch.zeros((N, 9))
    for n, cls in enumerate(classes):
        t = spec["classes"].get(str(cls), spec["all"])
        th[n] = torch.tensor(t["scale"] + t["accel"] + t["lateral"] + t["vertical"])
    scale, accel, lateral, vertical = (sig[..., i] for i in range(4))     # NaN compares False
    cond = torch.stack([
        scale > th[:, 1], scale > th[:, 2], scale < th[:, 0],
        accel > th[:, 4], accel < th[:, 3],
        lateral < th[:, 5], lateral > th[:, 6],
        vertical > th[:, 8], vertical < th[:, 7],
        (scale.abs() < STILL["scale"]) & (lateral.abs() < STILL["lateral"]) & (vertical.abs() < STILL["vertical"]),
    ], dim=2)
    return _debounce(cond) & has_box.cpu()[:, :, None]
