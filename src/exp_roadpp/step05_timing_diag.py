"""Step 05 diagnostic: which signals tell "the onset comes in ~2 s" from "the onset comes in ~10 s"?

Why: the rule net and the lookup alarm many seconds early (the ego is predicted to stop 10 s before
it stops). The label atoms are states that hold for long stretches (AV-Mov, a red light in view,
VehLane), so they say whether an event comes, not when. Before new atoms are built (ego motion read
from the scene, the motion of role partners, time to contact), this script checks on the train split,
with no model, which candidate signals change in the last seconds before an onset.

For every head of --heads (or AV-Stop, the other ego heads and the most frequent agent heads) and every
row agent that can hold it, each frame t at risk (the head atom off in the MIN_OFF_FRAMES before t)
gets its time to the next onset tau = onset - t (s):
  - pre-event rows: the next onset of the head on this agent is within MAX_LEAD_S (tau = 0 is the
    onset frame itself), binned by 1 s;
  - background rows: no onset within BG_GAP_S (or none at all), a random share (--bg_rate_ego for the
    ego, --bg_rate_agent for the other agents).
Signals per row (continuous, NaN where undefined; frame differences over motion.K frames):
  own_*:    the agent's own box (none for the ego): scale (log-area growth per frame, > 0 closer;
            2 / (scale * fps) is the time to contact, so its AUC is that of the TTC), accel (change of
            scale), lateral, vertical (`motion.signals`), log_area (size), bottom (y2, lower = nearer),
            centre_off (|centre x - 0.5|);
  scene_*:  all ground agents with a box in the frame (the ego's own motion moves all of them):
            median scale (the scene expands while the ego drives forward), its change over K frames
            (< 0: the expansion slows, the ego brakes), median outward lateral flow, median vertical
            flow, median scale of the static agents (traffic lights), the number of boxes;
  ahead_*:  the ego's lead agent (`roles.role_partners` "ahead"), ego rows only: exists, scale, accel,
            log_area, bottom;
  near_*:   the nearest other ground agent (the ego stands at roles.EGO_BOX): its ground distance in
            box heights and the change of the distance to that same agent over K frames, per frame;
  label:*:  the row agent's own label atoms (binary), the current state the rules read now.
Outputs (out_dir):
  timing_bins.csv: head, signal, bin (0-1s .. 9-10s, bg), rows, coverage (non-NaN share), q25, median,
                   q75, mean;
  timing_auc.csv:  head, signal, rows per group, AUC late vs early (late = LATE_S, early = EARLY_S),
                   AUC late vs background, AUC early vs background;
  timing_<head>.pdf: per head, the median and IQR of the most timing-informative signals over the
                   time to the onset, the background as a band;
  and a terminal summary per head.
Reading: |AUC late vs early - 0.5| is the timing information of a signal (>= 0.15 is clear; AUC < 0.5:
lower in the last seconds). A signal with a high AUC late vs background but late vs early near 0.5
says that an event comes, not when (what the label atoms are expected to show).

Run on the server (no model needed, only the step_dataset files of --exp):
  ./run_roadpp_docker.sh --gpu 2 -- --exp medium --timing_diag [--heads av:AV-Stop] [--max_videos 20]
or from the repo root: python -m src.exp_roadpp.step05_timing_diag --exp medium --machine dgx-jing
"""
import argparse
import csv
import math
import os
from pathlib import Path

import torch
from tqdm import tqdm

from src.exp_roadpp import step05, utils_data, roles as agent_roles, motion as agent_motion

FPS = 10
MIN_OFF_FRAMES = 10                 # step05 get_cfg: an onset needs the atom off this long before
MAX_LEAD_S = 10                     # pre-event rows: the onset at most this far ahead
LATE_S = (0, 2)                     # the headline window (W20 at 10 fps)
EARLY_S = (4, 10)                   # the onset still far: an alarm here is the early alarm to avoid
BG_GAP_S = 10                       # background rows: no onset within this
STATIC_CLASSES = ("TL", "OthTL")    # do not move: their image motion is the ego's
NEAR_CHUNK = 256                    # frames per chunk of the (F, N, N) distance tensor
MIN_ROWS = 20                       # per group, for an AUC
MAX_AUC_ROWS = 200_000              # per group, subsampled for the AUC
MIN_LABEL_SHARE = 0.01              # a label atom held in this share of the late or early rows
NUM_PLOTTED = 12
SIGNALS = ("own_scale", "own_accel", "own_lateral", "own_vertical", "own_log_area", "own_bottom",
           "own_centre_off",
           "scene_scale", "scene_scale_change", "scene_outflow", "scene_vertical", "scene_static_scale",
           "scene_boxes",
           "ahead_exists", "ahead_scale", "ahead_accel", "ahead_log_area", "ahead_bottom",
           "near_dist", "near_dist_rate")


# -------------- Signals of one video --------------
def _median(values, mask):
    """(T,) median over the agents of (T, N) values where mask, NaN where none."""
    return torch.nanmedian(torch.where(mask, values, float("nan")), dim=1).values


def _shift_diff(v, k):
    """v(t) - v(t - k), NaN for t < k (NaN propagates)."""
    out = torch.full_like(v, float("nan"))
    out[k:] = v[k:] - v[:-k]
    return out


def _nearest(gx, gy, h, cand):
    """(T, N) normalized ground distance to the nearest candidate agent and its change over K frames
    to that same agent. gx, gy, h (T, N) ground points and box heights; cand (T, N) the agents that
    may be (and have) a nearest agent."""
    T, N = gx.shape
    K = agent_motion.K
    eye = torch.eye(N, dtype=torch.bool)

    def dist(frames):
        dx = gx[frames, :, None] - gx[frames, None, :]
        dy = gy[frames, :, None] - gy[frames, None, :]
        scale = torch.maximum(h[frames, :, None], h[frames, None, :]).clamp(min=0.02)
        d = torch.sqrt(dx ** 2 + dy ** 2) / scale
        ok = cand[frames, :, None] & cand[frames, None, :] & ~eye
        return torch.where(ok, d, float("inf"))                                     # (F, N, N)

    near = torch.full((T, N), float("nan"))
    rate = torch.full((T, N), float("nan"))
    for begin in range(0, T, NEAR_CHUNK):
        frames = torch.arange(begin, min(begin + NEAR_CHUNK, T))
        d, m = dist(frames).min(dim=2)                                              # (F, N)
        found = torch.isfinite(d)
        near[frames] = torch.where(found, d, float("nan"))
        has_prev = frames >= K
        if has_prev.any():
            f = frames[has_prev]
            prev = dist(f - K).gather(2, m[has_prev][:, :, None]).squeeze(2)        # same pair, K frames earlier
            r = (d[has_prev] - prev) / K
            rate[f] = torch.where(found[has_prev] & torch.isfinite(prev), r, float("nan"))
    return near, rate


def video_signals(X, boxes, has_box, classes, ego, unique_atoms):
    """(T, N, len(SIGNALS)) float signals of one video (NaN where undefined)."""
    T, N, _ = X.shape
    K = agent_motion.K
    nan = float("nan")
    out = torch.full((T, N, len(SIGNALS)), nan)
    col = {s: i for i, s in enumerate(SIGNALS)}
    sig = agent_motion.signals(boxes, has_box)                                      # (T, N, 4)
    scale, accel, lateral, vertical = sig.unbind(dim=2)
    W, H = agent_motion.FRAME_SIZE_PX
    b = boxes.float()
    x1, y1, x2, y2 = b.unbind(dim=2)
    ok = has_box & ~torch.isnan(b).any(dim=2)
    cx = (x1 + x2) / 2
    log_area = torch.log(((x2 - x1) * W).clamp(min=1.0) * ((y2 - y1) * H).clamp(min=1.0))
    own = {"own_scale": scale, "own_accel": accel, "own_lateral": lateral, "own_vertical": vertical,
           "own_log_area": torch.where(ok, log_area, nan), "own_bottom": torch.where(ok, y2, nan),
           "own_centre_off": torch.where(ok, (cx - 0.5).abs(), nan)}
    for s, v in own.items():
        out[..., col[s]] = v

    ground = torch.tensor([c not in agent_roles.NON_GROUND_CLASSES for c in classes], dtype=torch.bool)
    static = torch.tensor([c in STATIC_CLASSES for c in classes], dtype=torch.bool)
    valid = ~torch.isnan(scale)
    scene_scale = _median(scale, valid & ground[None])
    scene = {"scene_scale": scene_scale,
             "scene_scale_change": _shift_diff(scene_scale, K),
             "scene_outflow": _median(lateral * torch.sign(cx - 0.5), valid & ground[None]),
             "scene_vertical": _median(vertical, valid & ground[None]),
             "scene_static_scale": _median(scale, valid & static[None]),
             "scene_boxes": ok.sum(dim=1).float()}
    for s, v in scene.items():
        out[..., col[s]] = v[:, None]

    if ego is not None:
        lane = agent_roles.ego_lane(X, unique_atoms)
        partner = agent_roles.role_partners(boxes, classes, ego, ("ahead",), lane)[:, ego, 0]   # (T,)
        has = partner >= 0
        idx = partner.clamp(min=0)
        frames = torch.arange(T)
        out[:, ego, col["ahead_exists"]] = has.float()
        for s, v in (("ahead_scale", scale), ("ahead_accel", accel), ("ahead_log_area", own["own_log_area"]),
                     ("ahead_bottom", own["own_bottom"])):
            out[:, ego, col[s]] = torch.where(has, v[frames, idx], nan)

    # Nearest other ground agent; the ego (no box) stands at EGO_BOX and is nobody's nearest agent.
    gb = torch.where(ok[..., None], b, nan)
    cand = ok & ground[None]
    if ego is not None:
        gb[:, ego] = torch.tensor(agent_roles.EGO_BOX)
    gx, gy = (gb[..., 0] + gb[..., 2]) / 2, gb[..., 3]
    h = (gb[..., 3] - gb[..., 1]).clamp(min=0)
    near, rate = _nearest(torch.nan_to_num(gx), torch.nan_to_num(gy), torch.nan_to_num(h), cand)
    if ego is not None:
        # The ego has a nearest agent (at EGO_BOX) but is none: rerun with the ego as a candidate
        # and keep its row only.
        cand_ego = cand.clone()
        cand_ego[:, ego] = True
        near_e, rate_e = _nearest(torch.nan_to_num(gx), torch.nan_to_num(gy), torch.nan_to_num(h), cand_ego)
        near[:, ego], rate[:, ego] = near_e[:, ego], rate_e[:, ego]
    out[..., col["near_dist"]] = near
    out[..., col["near_dist_rate"]] = rate
    return out


# -------------- Rows of one head --------------
def head_rows(X, h, agents, num_labels, signals, bg_rate, generator, fps):
    """(tau (R,) float s, -1 for background; signals (R, S); labels (R, num_labels) bool) of the rows
    of head atom h and the agents (long), at risk and visible."""
    T = X.shape[0]
    x = X[:, agents, h]                                                             # (T, n)
    visible = X[:, agents].any(dim=2)
    cum = torch.cat([torch.zeros((1, len(agents)), dtype=torch.long), x.long().cumsum(dim=0)])
    t = torch.arange(T)
    lo = (t - MIN_OFF_FRAMES).clamp(min=0)
    held = (cum[t] - cum[lo]) > 0                                                   # on in t - m .. t - 1
    seen_before = torch.zeros_like(visible)
    seen_before[1:] = visible[:-1]
    onset = x & ~held & seen_before
    big = T + 10 ** 6
    on_at = torch.where(x, t[:, None], big)
    next_on = on_at.flip(0).cummin(dim=0).values.flip(0)                            # first on frame >= t
    tau = next_on - t[:, None]
    is_onset_next = torch.zeros_like(x)
    has_next = next_on < T
    is_onset_next[has_next] = onset[next_on.clamp(max=T - 1), torch.arange(len(agents))[None].expand(T, -1)][has_next]
    base = visible & ~held
    pre = base & has_next & is_onset_next & (tau < MAX_LEAD_S * fps)
    bg = base & ~x & (tau > BG_GAP_S * fps)
    bg &= torch.rand(bg.shape, generator=generator) < bg_rate
    rows_t, rows_a = torch.nonzero(pre | bg, as_tuple=True)
    n = agents[rows_a]
    taus = torch.where(pre[rows_t, rows_a], tau[rows_t, rows_a].float() / fps, torch.tensor(-1.0))
    return taus, signals[rows_t, n], X[rows_t, n, :num_labels]


# -------------- Statistics --------------
def auc(pos, neg, generator):
    """P(pos > neg) + P(=) / 2 of two value sets (NaN dropped), NaN with too few values."""
    pos, neg = pos[~torch.isnan(pos)].double(), neg[~torch.isnan(neg)].double()
    if len(pos) < MIN_ROWS or len(neg) < MIN_ROWS:
        return float("nan")
    if len(pos) > MAX_AUC_ROWS:
        pos = pos[torch.randperm(len(pos), generator=generator)[:MAX_AUC_ROWS]]
    if len(neg) > MAX_AUC_ROWS:
        neg = neg[torch.randperm(len(neg), generator=generator)[:MAX_AUC_ROWS]]
    v = torch.cat([pos, neg])
    order = torch.argsort(v)
    _, inverse, counts = torch.unique(v[order], return_inverse=True, return_counts=True)
    ends = counts.cumsum(dim=0).double()
    average = ends - (counts.double() - 1) / 2                                      # 1-based mean rank of ties
    ranks = torch.empty_like(v)
    ranks[order] = average[inverse]
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bin_stats(values):
    v = values[~torch.isnan(values)]
    if len(v) == 0:
        return {"coverage": 0.0, "q25": "", "median": "", "q75": "", "mean": ""}
    q = torch.quantile(v[:MAX_AUC_ROWS].float(), torch.tensor([0.25, 0.5, 0.75])).tolist()
    return {"coverage": round(len(v) / len(values), 4), "q25": q[0], "median": q[1], "q75": q[2],
            "mean": float(v.float().mean())}


def bins():
    return [(f"{s}-{s + 1}s", s, s + 1) for s in range(MAX_LEAD_S)]


def _groups(tau):
    late = (tau >= LATE_S[0]) & (tau < LATE_S[1])
    early = (tau >= EARLY_S[0]) & (tau < EARLY_S[1])
    return late, early, tau < 0


# -------------- Main --------------
def select_heads(atoms, unique_atoms, names, wanted, num_agent_heads, min_onsets):
    """The head atom indices: --heads by name, else every ego head (av:*) and the num_agent_heads most
    frequent agent heads with >= min_onsets train onsets; with the onset counts."""
    A = len(unique_atoms)
    counts = torch.zeros(A, dtype=torch.long)
    for video_atoms in tqdm(atoms.values(), desc="Counting onsets"):
        X = step05.build_video_tensor(video_atoms, A)
        if X.shape[0] < 2:
            continue
        visible = X.any(dim=2)
        onset = step05.atom_onsets(X, MIN_OFF_FRAMES)
        onset[0] = False
        onset[1:] &= visible[:-1, :, None]
        counts += onset.sum(dim=(0, 1))
    if wanted:
        missing = [w for w in wanted if w not in names]
        if missing:
            raise ValueError(f"Unknown heads {missing}; e.g. {[n for n in names if n.startswith('av:')][:10]}")
        return [names.index(w) for w in wanted], counts
    ego = [a for a in range(A) if names[a].startswith("av:") and counts[a] >= min_onsets]
    agent = [a for a in torch.argsort(counts, descending=True).tolist()
             if not names[a].startswith("av:") and counts[a] >= min_onsets][:num_agent_heads]
    return ego + agent, counts


def collect(atoms, boxes, unique_atoms, names, heads, args):
    """{head: (tau, signals, labels)} over the train videos."""
    A = len(unique_atoms)
    generator = torch.Generator().manual_seed(args.seed)
    class_of = [n.split(":", 1)[0] for n in names]
    same_class = {h: torch.tensor([c == class_of[h] for c in class_of], dtype=torch.bool) for h in heads}
    parts = {h: [] for h in heads}
    for video_id, video_atoms in tqdm(atoms.items(), desc="Collecting rows"):
        X = step05.build_video_tensor(video_atoms, A)
        if X.shape[0] <= agent_motion.K:
            continue
        video_boxes, classes, has_box = step05._agent_boxes(boxes, video_id, video_atoms, X)
        ego = step05.video_agent_ids(video_atoms).get(None)
        S = video_signals(X, video_boxes, has_box, classes, ego, unique_atoms)
        # The agents of each atom class: those that ever hold an atom of that class.
        holds = X.any(dim=0)                                                        # (N, A)
        for h in heads:
            agents = torch.nonzero(holds[:, same_class[h]].any(dim=1)).flatten()
            if len(agents) == 0:
                continue
            is_ego = class_of[h] == "av"
            rate = args.bg_rate_ego if is_ego else args.bg_rate_agent
            parts[h].append(head_rows(X, h, agents, A, S, rate, generator, args.fps))
    out = {}
    for h, p in parts.items():
        if p:
            out[h] = tuple(torch.cat([q[i] for q in p]) for i in range(3))
    return out


def analyse(rows, names, heads, counts, out_dir, seed):
    generator = torch.Generator().manual_seed(seed)
    bin_lines, auc_lines, summary = [], [], {}
    for h in heads:
        if h not in rows:
            continue
        tau, S, L = rows[h]
        late, early, bg = _groups(tau)
        entries = []
        for i, s in enumerate(SIGNALS):
            entries.append((s, S[:, i]))
        # The agent's own label atoms held in enough of the late or early rows (the head itself is off).
        share = torch.maximum(L[late].float().mean(dim=0) if late.any() else torch.zeros(L.shape[1]),
                              L[early].float().mean(dim=0) if early.any() else torch.zeros(L.shape[1]))
        for a in torch.nonzero(share >= MIN_LABEL_SHARE).flatten().tolist():
            if a != h:
                entries.append((f"label:{names[a]}", L[:, a].float()))
        results = []
        for s, v in entries:
            a_le, a_lb, a_eb = auc(v[late], v[early], generator), auc(v[late], v[bg], generator), \
                auc(v[early], v[bg], generator)
            auc_lines.append({"head": names[h], "signal": s, "late_rows": int(late.sum()),
                              "early_rows": int(early.sum()), "bg_rows": int(bg.sum()),
                              "auc_late_vs_early": a_le, "auc_late_vs_bg": a_lb, "auc_early_vs_bg": a_eb})
            results.append((s, a_le, a_lb, a_eb))
            if s.startswith("label:"):
                continue
            for label, lo, hi in bins() + [("bg", None, None)]:
                m = bg if lo is None else (tau >= lo) & (tau < hi)
                bin_lines.append({"head": names[h], "signal": s, "bin": label, "rows": int(m.sum()),
                                  **bin_stats(v[m])})
        summary[h] = results
        plot_head(names[h], tau, S, results, out_dir)
        print_head(names[h], int(counts[h]), late, early, bg, results)
    _write_csv(out_dir / "timing_bins.csv", bin_lines)
    _write_csv(out_dir / "timing_auc.csv", auc_lines)
    return summary


def _separation(a):
    return -1.0 if math.isnan(a) else abs(a - 0.5)


def print_head(name, num_onsets, late, early, bg, results, top=10):
    print(f"\n[Timing]\t {name}: {num_onsets} train onsets; rows late {int(late.sum())} "
          f"({LATE_S[0]}-{LATE_S[1]} s before), early {int(early.sum())} ({EARLY_S[0]}-{EARLY_S[1]} s), "
          f"background {int(bg.sum())}")
    print(f"  {'signal':<34}{'late/early':>11}{'late/bg':>9}{'early/bg':>10}")
    for title, keep in (("signals", lambda s: not s.startswith("label:")), ("label atoms", lambda s: s.startswith("label:"))):
        ranked = sorted((r for r in results if keep(r[0])), key=lambda r: -_separation(r[1]))[:top]
        print(f"  -- {title}, by timing information |AUC late vs early - 0.5|")
        for s, a_le, a_lb, a_eb in ranked:
            print(f"  {s[:34]:<34}{a_le:>11.3f}{a_lb:>9.3f}{a_eb:>10.3f}")


def plot_head(name, tau, S, results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ranked = [r[0] for r in sorted(results, key=lambda r: -_separation(r[1])) if not r[0].startswith("label:")]
    ranked = [s for s in ranked if not math.isnan(next(r[1] for r in results if r[0] == s))][:NUM_PLOTTED]
    if not ranked:
        return
    cols = 4
    rows_n = math.ceil(len(ranked) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 2.8 * rows_n), squeeze=False)
    bg = tau < 0
    for ax, s in zip(axes.flat, ranked):
        v = S[:, SIGNALS.index(s)]
        xs, med, q25, q75 = [], [], [], []
        for _label, lo, hi in bins():
            st = bin_stats(v[(tau >= lo) & (tau < hi)])
            if st["median"] == "":
                continue
            xs.append(-(lo + hi) / 2)
            med.append(st["median"]); q25.append(st["q25"]); q75.append(st["q75"])
        b = bin_stats(v[bg])
        if b["median"] != "":
            ax.axhspan(b["q25"], b["q75"], color="0.85", label="background IQR")
            ax.axhline(b["median"], color="0.5", lw=1)
        ax.fill_between(xs, q25, q75, alpha=0.3)
        ax.plot(xs, med, marker="o", ms=3)
        ax.axvspan(-LATE_S[1], -LATE_S[0], color="tab:green", alpha=0.08)
        a = next(r[1] for r in results if r[0] == s)
        ax.set_title(f"{s}  AUC late/early {a:.2f}", fontsize=9)
        ax.set_xlabel("seconds before the onset", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(ranked):]:
        ax.axis("off")
    fig.suptitle(f"{name}: median and IQR per second before the onset (grey: background)", fontsize=10)
    fig.tight_layout()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    fig.savefig(out_dir / f"timing_{safe}.pdf")
    plt.close(fig)


def _write_csv(path, lines):
    if not lines:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(lines[0]))
        writer.writeheader()
        writer.writerows(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--exp", type=str, default="debug", help="experiment: the atoms of its step_dataset output")
    parser.add_argument("--machine", type=str, default="dgx-jing", help="config.machine_setup: the output paths")
    parser.add_argument("--device", type=str, default="cpu", help="unused (the script runs on the CPU)")
    parser.add_argument("--atoms_dir", type=Path, default=None,
                        help="step_dataset_output/atoms: train_atoms_compressed.json, train_boxes.pt, unique_atoms.json; "
                             "default: that of --exp on --machine")
    parser.add_argument("--out_dir", type=Path, default=None, help="default: <atoms_dir>/../timing_diag")
    parser.add_argument("--heads", nargs="*", default=None, help='head atom names, e.g. "av:AV-Stop" "Car:Brake"')
    parser.add_argument("--num_agent_heads", type=int, default=8)
    parser.add_argument("--min_onsets", type=int, default=50)
    parser.add_argument("--bg_rate_ego", type=float, default=0.3)
    parser.add_argument("--bg_rate_agent", type=float, default=0.03)
    parser.add_argument("--max_videos", type=int, default=None, help="the first videos only (a quick check)")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.atoms_dir is None:
        from src.exp_roadpp import config
        args.atoms_dir = config.machine_setup(args).output_dir / "step_dataset_output" / "atoms"
    out_dir = args.out_dir or args.atoms_dir.parent / "timing_diag"
    os.makedirs(out_dir, exist_ok=True)

    unique_atoms = utils_data.load_json(args.atoms_dir / "unique_atoms.json")
    names = [step05._atom_name(sig) for sig in unique_atoms]
    atoms = utils_data.load_json(args.atoms_dir / "train_atoms_compressed.json")
    if args.max_videos:
        atoms = dict(list(atoms.items())[:args.max_videos])
    boxes = torch.load(args.atoms_dir / "train_boxes.pt", map_location="cpu")
    heads, counts = select_heads(atoms, unique_atoms, names, args.heads, args.num_agent_heads, args.min_onsets)
    print(f"[Timing]\t {len(atoms)} train videos; heads: " + ", ".join(f"{names[h]} ({int(counts[h])})" for h in heads))
    rows = collect(atoms, boxes, unique_atoms, names, heads, args)
    analyse(rows, names, heads, counts, out_dir, args.seed)
    print(f"\n[Timing]\t wrote {out_dir}/timing_bins.csv, timing_auc.csv, timing_<head>.pdf")


if __name__ == "__main__":
    main()
