"""Shape-phase atoms: how an agent's own 2D box moves and changes its shape, and in which order, as inputs of
the rules (2026-11-22, the user's request).

Why: lane and turn onsets show in the box before the label changes:
  - an oncoming car crossing the junction straight: its box moves left / down-left and grows;
  - a truck from the left turning right towards the ego: a tall box on the left moves right, gets wider,
    then taller, then narrower while it keeps getting taller.
The second needs the order of several changes, of varying durations (speed, length). The grid (3x3)
and the motion atoms (`motion.py`: area, lateral, vertical) do not separate width from height and keep no
order. Rule lab 2026-11-22: the rules trail the MLP most on VehLane / IncomLane / OutgoLane / MovRht.

Primitives per agent and frame t (own box, over K frames, pixels of the image):
  x   dx = (centre x(t) - centre x(t - K)) / (K x width)    -> x_left / x_right
  y   dy = (centre y(t) - centre y(t - K)) / (K x height)   -> y_up / y_down (down: towards the camera)
  w   dw = log(width(t) / width(t - K)) / K                  -> w_grow / w_shrink
  h   dh = log(height(t) / height(t - K)) / K                -> h_grow / h_shrink
The thresholds are per class (`fit`): the train quantiles LOW / HIGH of each signal, never closer to 0 than
FLOORS (the GT boxes jitter). A primitive holds only if its condition held DEBOUNCE frames in a row. Boxes cut
by the image border give no signal.

Phases: the frames of an agent's track where the primitive state (x, y, w, h each -1 / 0 / +1) stays the same.
A run shorter than MIN_PHASE frames is not a phase of its own: a change must last to count. Phase 1 is the
current one, phase 2 the one before, ... up to PHASES. The phases carry no duration, so a slow and a fast
truck give the same sequence (the order and the combination only, as `step_seq.py`).

Atoms (inputs only, own agent scope, `step05.rule_input_mask`):
  "shp:p{k}_{prim}"   in phase k the primitive held (k = 1..PHASES, prim in PRIMS), "shp:p{k}_still": none did;
  "shp:app_{tall,square,wide}", "shp:app_{left,centre,right}", "shp:app_{top,middle,bottom}": the box when the
      track appeared (its aspect tertile of the class; the image third of its centre);
  "shp:{tall,square,wide}": the box now;
  "shp:young": the track appeared less than YOUNG frames ago (the agent was not seen before).
The truck turning right towards the ego then reads, e.g.:
  shp:app_tall & shp:app_left & shp:p3_x_right & shp:p3_w_grow & shp:p2_h_grow & shp:p1_w_shrink & shp:p1_h_grow
"""
import torch

K = 5                                   # frames of each difference (0.5 s at 10 fps)
DEBOUNCE = 3                            # frames a primitive's condition must hold in a row
MIN_PHASE = 3                           # frames of a run to count as a phase
PHASES = 4                              # phases kept: the current one and the 3 before
YOUNG = 20                              # frames: a track younger than this is "young"
EDGE = 0.005                            # normalized: a box this close to the border is cut
FRAME_SIZE_PX = (1920, 1280)
SIGNALS = ("dx", "dy", "dw", "dh", "aspect")
PRIMS = ("x_left", "x_right", "y_up", "y_down", "w_grow", "w_shrink", "h_grow", "h_shrink")
LOW, HIGH = 0.15, 0.85                  # train quantiles of dx, dy, dw, dh
FLOORS = {"dx": 0.02, "dy": 0.01, "dw": 0.008, "dh": 0.008}
MIN_CLASS_SAMPLES = 5000
MAX_SAMPLES = 1_000_000
VERSION = "v1"


def atom_names():
    names = [f"p{k}_{p}" for k in range(1, PHASES + 1) for p in PRIMS + ("still",)]
    names += [f"app_{a}" for a in ("tall", "square", "wide", "left", "centre", "right", "top", "middle", "bottom")]
    names += ["tall", "square", "wide", "young"]
    return names


ATOMS = tuple(atom_names())


def tag():
    return f"_shp{K}d{DEBOUNCE}m{MIN_PHASE}p{PHASES}{VERSION}"


def vocabulary():
    """Signatures of the shape atoms; `step05._atom_name` reads them as "shp:p1_w_grow"."""
    return [["pred", "shape", "shape_label", a, "agent_class_label", "shp"] for a in ATOMS]


# -------------- Signals and thresholds --------------
def signals(boxes, has_box):
    """(T, N, 5) float SIGNALS of (T, N, 4) normalized boxes, NaN where undefined (aspect: where a box is)."""
    T, N, _ = boxes.shape
    W, H = FRAME_SIZE_PX
    b = boxes.float()
    x1, y1, x2, y2 = (b[..., i] for i in range(4))
    cut = (x1 < EDGE) | (y1 < EDGE) | (x2 > 1 - EDGE) | (y2 > 1 - EDGE)
    box = has_box & ~torch.isnan(b).any(dim=2)
    ok = box & ~cut
    w, h = ((x2 - x1) * W).clamp(min=1.0), ((y2 - y1) * H).clamp(min=1.0)
    cx, cy = (x1 + x2) / 2 * W, (y1 + y2) / 2 * H
    out = torch.full((T, N, len(SIGNALS)), float("nan"))
    out[..., 4] = torch.where(box, torch.log(w / h), torch.full_like(w, float("nan")))
    if T <= K:
        return out
    pair = ok[K:] & ok[:-K]

    def diff(v):
        d = torch.full((T, N), float("nan"))
        d[K:] = torch.where(pair, v[K:] - v[:-K], torch.full_like(v[K:], float("nan")))
        return d
    out[..., 0] = diff(cx) / (K * w)
    out[..., 1] = diff(cy) / (K * h)
    out[..., 2] = diff(torch.log(w)) / K
    out[..., 3] = diff(torch.log(h)) / K
    return out


def _thresholds(values):
    """{signal: [low, high]} of (M, 5) signals: the quantiles, the change signals with the FLOORS."""
    out = {}
    for i, s in enumerate(SIGNALS):
        v = values[:, i]
        v = v[~torch.isnan(v)]
        if len(v) > MAX_SAMPLES:
            v = v[torch.randperm(len(v), generator=torch.Generator().manual_seed(0))[:MAX_SAMPLES]]
        if s == "aspect":
            q = torch.quantile(v, torch.tensor([1 / 3, 2 / 3])).tolist() if len(v) else [-0.3, 0.3]
            out[s] = q
            continue
        q = torch.quantile(v, torch.tensor([LOW, HIGH])).tolist() if len(v) else [0.0, 0.0]
        out[s] = [min(q[0], -FLOORS[s]), max(q[1], FLOORS[s])]
    return out


def fit(samples):
    """The thresholds of the train signals {class: (M, 5)}: {"classes": {class: thresholds}, "all": ...,
    "counts": {class: M}}; a class with < MIN_CLASS_SAMPLES change signals gets "all"."""
    every = torch.cat(list(samples.values())) if samples else torch.zeros((0, len(SIGNALS)))
    spec = {"all": _thresholds(every), "classes": {}, "counts": {}}
    for cls, values in samples.items():
        n = int((~torch.isnan(values[:, 0])).sum())
        spec["counts"][cls] = n
        if n >= MIN_CLASS_SAMPLES:
            spec["classes"][cls] = _thresholds(values)
    return spec


def _debounce(cond):
    out = cond.clone()
    for d in range(1, DEBOUNCE):
        shifted = torch.zeros_like(cond)
        shifted[d:] = cond[:-d]
        out &= shifted
    return out


# -------------- Atoms --------------
def primitives(sig, classes, spec):
    """(T, N, 8) bool debounced PRIMS, (T, N) bool where the change signals are defined, (T, N) long aspect bin
    (0 tall, 1 square, 2 wide; -1 without a box)."""
    T, N, _ = sig.shape
    th = torch.zeros((N, 10))
    for n, cls in enumerate(classes):
        t = spec["classes"].get(str(cls), spec["all"])
        th[n] = torch.tensor(t["dx"] + t["dy"] + t["dw"] + t["dh"] + t["aspect"])
    dx, dy, dw, dh, asp = (sig[..., i] for i in range(5))       # NaN compares False
    cond = torch.stack([dx < th[:, 0], dx > th[:, 1], dy < th[:, 2], dy > th[:, 3],
                        dw > th[:, 5], dw < th[:, 4], dh > th[:, 7], dh < th[:, 6]], dim=2)
    defined = ~torch.isnan(dx)
    aspect = torch.where(torch.isnan(asp), torch.full((T, N), -1, dtype=torch.long),
                         (asp > th[:, 8]).long() + (asp > th[:, 9]).long())
    return _debounce(cond) & defined[..., None], defined, aspect


def _state_code(prim):
    """(T, N) long: the primitive state (x, y, w, h each -1 / 0 / +1) as one code 0..80."""
    x = prim[..., 1].long() - prim[..., 0].long()
    y = prim[..., 3].long() - prim[..., 2].long()
    w = prim[..., 4].long() - prim[..., 5].long()
    h = prim[..., 6].long() - prim[..., 7].long()
    return (x + 1) * 27 + (y + 1) * 9 + (w + 1) * 3 + (h + 1)


def _code_prims(code):
    """(8,) bool PRIMS of a state code and whether it is still (all 0)."""
    x, y, w, h = code // 27 - 1, (code // 9) % 3 - 1, (code // 3) % 3 - 1, code % 3 - 1
    p = [x < 0, x > 0, y < 0, y > 0, w > 0, w < 0, h > 0, h < 0]
    return p, not any(p)


def phase_history(code, defined, has_box):
    """(T, N, PHASES) long: the state code of phase 1 (current) .. PHASES at every frame, -1 if none. A track
    restarts after a frame without a box; a run shorter than MIN_PHASE is not a phase (but the current run is
    phase 1 from its first frame)."""
    T, N = code.shape
    out = torch.full((T, N, PHASES), -1, dtype=torch.long)
    code_l, defined_l, box_l = code.tolist(), defined.tolist(), has_box.tolist()
    for n in range(N):
        history, run_code, run_len = [], None, 0
        for t in range(T):
            if not box_l[t][n]:
                history, run_code, run_len = [], None, 0
                continue
            if not defined_l[t][n]:
                continue
            c = code_l[t][n]
            if c != run_code:
                if run_code is not None and run_len >= MIN_PHASE and (not history or history[-1] != run_code):
                    history.append(run_code)
                run_code, run_len = c, 0
            run_len += 1
            # A run back to the last phase after a short interruption continues that phase.
            before = history[:-1] if history and history[-1] == run_code else history
            for k, p in enumerate([run_code] + before[::-1][:PHASES - 1]):
                out[t, n, k] = p
    return out


def shape_atoms(boxes, has_box, classes, spec):
    """(T, N, len(ATOMS)) bool shape atoms of one video. boxes (T, N, 4) normalized (NaN without a box),
    has_box (T, N) bool, classes [N] the agents' class labels, spec (`fit`)."""
    boxes, has_box = boxes.cpu(), has_box.cpu()
    sig = signals(boxes, has_box)
    T, N, _ = sig.shape
    prim, defined, aspect = primitives(sig, classes, spec)
    hist = phase_history(_state_code(prim), defined, has_box)
    table = torch.zeros((81, len(PRIMS) + 1), dtype=torch.bool)               # code -> PRIMS + still
    for c in range(81):
        p, still = _code_prims(c)
        table[c] = torch.tensor(p + [still])
    out = torch.zeros((T, N, len(ATOMS)), dtype=torch.bool)
    per = len(PRIMS) + 1
    for k in range(PHASES):
        h = hist[..., k]
        out[..., k * per:(k + 1) * per] = table[h.clamp(min=0)] & (h >= 0)[..., None]
    # The box when its track appeared, the box now, the track age.
    base = PHASES * per
    W, H = FRAME_SIZE_PX
    b = boxes.nan_to_num(0.0)
    cx, cy = (b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2
    for n in range(N):
        start = -1
        for t in range(T):
            if not bool(has_box[t, n]):
                start = -1
                continue
            if start < 0:
                start = t
            if aspect[start, n] >= 0:
                out[t, n, base + int(aspect[start, n])] = True
            out[t, n, base + 3 + min(int(cx[start, n] * 3), 2)] = True
            out[t, n, base + 6 + min(int(cy[start, n] * 3), 2)] = True
            if aspect[t, n] >= 0:
                out[t, n, base + 9 + int(aspect[t, n])] = True
            out[t, n, base + 12] = t - start < YOUNG
    return out & has_box[..., None]
