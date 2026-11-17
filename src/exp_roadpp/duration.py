"""Duration atoms: how long a label has held, in log-scale age bins, as rule inputs.

For a label atom L of LABELS (persistent states: braking, standing, a hazard light, waiting to cross, the
signal colours, moving away, the ego's own states) and every range of adjacent age bins a..b, the atom
"L[a-b)" holds for an agent in frame t if L holds and its current run started a..b frames before t:

    BINS (frames): 0-5, 5-10, 10-20, 20-40, 40-80, 80+     (log scale)
    ranges: every contiguous union of them, but the whole range (that is L itself): 20 per label

A run is the frames of L joined over gaps of at most MIN_OFF_FRAMES (as `step05.atom_onsets`: a new run is
a new onset). Its age counts from its first frame; for a run that is on when the agent appears, from the
frame the agent is first seen (the start of the observation, as for every other atom), so "Car:Brake[0-5)"
is either a brake that just began or a braking car that just came into view.

The atoms are inputs only (`step05.is_input_atom`, never heads), after the other input atoms, one column
per (label atom, range); like the label atoms they are read in every scope (the agent itself, some other
agent, the roles), so a rule can say "the agent ahead has been braking for 0.5-2 s -> the ego brakes" or
"the ego has stood for more than 8 s -> it starts". Bodies over spans of slots (`slot_ranges`) still give
when, the duration atom how long.

Turned on by --durations True; the labels by --duration_labels (default LABELS)."""
import torch

VERSION = "dur_v1"
MIN_OFF_FRAMES = 10                 # as step05 min_off_frames (`atom_onsets`)
BINS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, None)]
LABELS = ("Brake", "Stop", "HazLit", "Wait2X", "Red", "Green", "Amber", "MovAway",
          "AV-Stop", "AV-Mov", "AV-Brake")


def ranges():
    """[(first bin, last bin)] of every contiguous union of the bins but the whole range."""
    n = len(BINS)
    return [(i, j) for i in range(n) for j in range(i, n) if not (i == 0 and j == n - 1)]


def range_name(i, j):
    a, b = BINS[i][0], BINS[j][1]
    return f"[{a}-{b})" if b is not None else f"[{a}+)"


def _fields(signature):
    return dict(zip(signature[0::2], signature[1::2]))


def base_atoms(unique_atoms, labels=LABELS):
    """[(label atom index, its class label, its label)] of the label vocabulary that get duration atoms."""
    out = []
    for a, sig in enumerate(unique_atoms):
        f = _fields(sig)
        label = f.get("action_id_label")
        if f.get("pred") == "action" and label in labels:
            out.append((a, str(f.get("agent_class_label")), label))
    return out


def tag(labels=LABELS):
    return f"_{VERSION}" + ("" if tuple(labels) == LABELS else "-" + "-".join(labels))


def vocabulary(unique_atoms, labels=LABELS):
    """Signatures of the duration atoms, base atom by base atom, range by range; `step05._atom_name`
    reads them as "<class>:<label>[a-b)" (the ego's "av:AV-Stop[80+)")."""
    out = []
    for _a, cls, label in base_atoms(unique_atoms, labels):
        for i, j in ranges():
            out.append(["pred", "duration", "duration_label", f"{label}{range_name(i, j)}",
                        "agent_class_label", cls])
    return out


def duration_atoms(X, unique_atoms, labels=LABELS):
    """(T, N, D) bool duration atoms of one video from its (T, N, A) bool label atoms X (A = the label
    vocabulary, in its order), D = len(base atoms) * len(ranges)."""
    base = base_atoms(unique_atoms, labels)
    T, N = X.shape[:2]
    R = ranges()
    out = torch.zeros((T, N, len(base), len(R)), dtype=torch.bool)
    if not base or T == 0:
        return out.view(T, N, -1)
    on = X[:, :, [a for a, _c, _l in base]].cpu()                      # (T, N, B)
    edges = torch.tensor([b for _a, b in BINS[:-1]])                   # 5, 10, 20, 40, 80
    start = torch.zeros((N, len(base)), dtype=torch.long)
    last_on = torch.full((N, len(base)), -10 ** 6, dtype=torch.long)
    lo = torch.tensor([i for i, _j in R])
    hi = torch.tensor([j for _i, j in R])
    for t in range(T):
        x = on[t]
        new = x & (t - last_on > MIN_OFF_FRAMES)                       # a new run (or the first frame seen)
        start = torch.where(new, torch.full_like(start, t), start)
        last_on = torch.where(x, torch.full_like(last_on, t), last_on)
        k = torch.bucketize(t - start, edges, right=True)              # (N, B) bin index of the age
        out[t] = x[:, :, None] & (k[:, :, None] >= lo) & (k[:, :, None] <= hi)
    return out.view(T, N, -1)
