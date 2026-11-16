"""Diagnostic: does the image ego motion (`ego_flow.py`) announce the ego's turns and stops before
their labels switch on?

Observation: before a right turn the ego already drifts / steers right while the label is still
AV-Mov. This script checks it on the train + val split of an exp, with no model, before ego flow
atoms are built.

Anchors (ego onsets, the rule of `step_dataset.head_onsets`: the label off MIN_OFF_FRAMES before,
not the ego's first frame), of the heads --heads; controls: up to --controls_per_video random frames
per video where the ego is AV-Mov and no ego onset of any head lies within CONTROL_GAP_S.
For each anchor the flow is computed from PRE_S + WARM_S before to POST_S after (WARM_S only warms
up the causal filters), on the sampled frames of `ego_flow` (every STEP frames).

Outputs (<exp>/ego_flow_diag/):
  ego_flow_diag.md / .json:
    1. per head and signal (yaw, speed; raw and denoised): AUC of the anchors' samples in each
       second before / after the onset vs the control samples (0.5: no information; Rht / Stop
       expect > 0.5 for yaw / < 0.5 for speed);
    2. lead: share of the onsets whose denoised signal passes the control q90 (q10) in the expected
       direction (DIRECTION) in each second before the onset;
    3. quantiles of the denoised yaw and speed over the control samples and over all samples:
       the cut points of the future symbolic levels;
    4. coverage: share of samples with a valid flow fit.
  flow_<head>.png: median and IQR of yaw and speed over the time to the onset, the controls as a band;
  samples.npz: every sample (head, rel time, raw and denoised signals), for later checks.

Run (server; CPU only, decodes the videos):
  ./run_roadpp_docker.sh --flow_diag -- --exp medium_s [--max_videos 50] [--workers 16]
"""
import argparse
import json
import os
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.exp_roadpp import config, ego_flow, utils_data

FPS = 10
MIN_OFF_FRAMES = 10
PRE_S, POST_S, WARM_S = 6.0, 3.0, 2.0
CONTROL_GAP_S = 10.0
HEADS = ("av:AV-TurRht", "av:AV-TurLft", "av:AV-MovRht", "av:AV-MovLft", "av:AV-Stop", "av:AV-Mov")
# head: (signal, +1 the onset expects high values / -1 low values), for the lead table.
DIRECTION = {"av:AV-TurRht": ("yaw", 1), "av:AV-MovRht": ("yaw", 1), "av:AV-TurLft": ("yaw", -1),
             "av:AV-MovLft": ("yaw", -1), "av:AV-Stop": ("speed", -1), "av:AV-Mov": ("speed", 1)}
AUC_BINS = [(-6, -5), (-5, -4), (-4, -3), (-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2)]
LEAD_BINS = [(-3, -2), (-2, -1), (-1, 0)]
QS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def ego_onsets(interval_atoms):
    """[(head, onset frame)] of the ego, and the ego's label per frame {t: label}."""
    ego = [a for a in interval_atoms if a.get("tube_uid") is None and str(a.get("agent_class_label")) == "av"]
    if not ego:
        return [], {}
    first = min(int(a["start_frame"]) for a in ego)
    spans, label_at = defaultdict(list), {}
    for a in ego:
        lab = a.get("action_id_label")
        s, e = int(a["start_frame"]), int(a["end_frame"])
        spans[f"av:{lab}"].append((s, e))
        for t in range(s, e):
            label_at[t] = lab
    out = []
    for head, ss in spans.items():
        last_end = None
        for s, e in sorted(ss):
            if s > first and (last_end is None or s - last_end >= MIN_OFF_FRAMES):
                out.append((head, s))
            last_end = e if last_end is None else max(last_end, e)
    return out, label_at


def _video_job(job):
    """Samples of one video: [(head or 'control', onset frame, rel time s, raw yaw, yaw, raw speed, speed)]."""
    import cv2
    cv2.setNumThreads(1)
    vid, anchors, boxes_by_frame, frame_dir, video_dir, fps = job
    pre, post, warm = int(PRE_S * fps), int(POST_S * fps), int(WARM_S * fps)
    sample = set()
    for _head, t in anchors:
        a = max(1, t - pre - warm)
        sample.update(range(a + (-a) % ego_flow.STEP, t + post + 1, ego_flow.STEP))
    raw = ego_flow.raw_series(cv2, vid, frame_dir, video_dir, sample,
                              lambda t: boxes_by_frame.get(t, ()), fps)
    if not raw:
        return vid, []
    ser = ego_flow.denoised_series(raw)
    grid = ser["yaw"][0]
    index = {int(t): i for i, t in enumerate(grid)}
    rows = []
    for head, t in anchors:
        for s in range(t - pre, t + post + 1):
            i = index.get(s)
            if i is None:
                continue
            rows.append((head, t, (s - t) / fps, ser["yaw"][1][i], ser["yaw"][2][i],
                         ser["speed"][1][i], ser["speed"][2][i]))
    return vid, rows


def _auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) < 5 or len(neg) < 5:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks of ties
    _u, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _plot(out_dir, head, S, control, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (sig, col) in zip(axes, (("yaw", 2), ("speed", 4))):
        c = control[:, col] if control.size else np.zeros(0)
        c = c[~np.isnan(c)]
        if len(c):
            lo, md, hi = np.percentile(c, [10, 50, 90])
            ax.axhspan(lo, hi, color="#bbbbbb", alpha=0.35, lw=0, label="control q10–q90")
            ax.axhline(md, color="#777777", lw=1)
        rel = S[:, 0]
        centers, med, q1, q3 = [], [], [], []
        for b in np.arange(-PRE_S, POST_S + 1e-9, ego_flow.STEP / FPS):
            v = S[np.isclose(rel, b), col]
            v = v[~np.isnan(v)]
            if len(v) >= 3:
                centers.append(b)
                q1.append(np.percentile(v, 25)); med.append(np.median(v)); q3.append(np.percentile(v, 75))
        if centers:
            ax.fill_between(centers, q1, q3, color="#2a6fdb", alpha=0.25, lw=0, label="onsets IQR")
            ax.plot(centers, med, color="#2a6fdb", lw=2, label="onsets median")
        ax.axvline(0, color="#d0453a", lw=1, ls="--", label="label onset")
        ax.set_xlabel("time to the label onset (s)")
        ax.set_title(f"{sig} (denoised){' — > 0: right' if sig == 'yaw' else ' — expansion / s'}", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{head}  ({n} onsets; control = AV-Mov frames far from any ego onset)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"flow_{head.replace(':', '_')}.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--exp", type=str, default="medium_s")
    parser.add_argument("--machine", type=str, default="dgx-jing")
    parser.add_argument("--device", type=str, default="cpu", help="unused")
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    parser.add_argument("--heads", nargs="*", default=list(HEADS))
    parser.add_argument("--controls_per_video", type=int, default=2)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--seed", type=int, default=0)
    args, _unknown = parser.parse_known_args()
    setup = config.machine_setup(args)
    atoms_dir = setup.output_dir / "step_dataset_output" / "atoms"
    frame_dir, video_dir = setup.dataset_path / "frames", setup.dataset_path / "videos"
    out_dir = setup.output_dir / "ego_flow_diag"
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    fps = args.fps
    gap = int(CONTROL_GAP_S * fps)
    pre_need = int((PRE_S + WARM_S) * fps)

    jobs, n_onsets = [], defaultdict(int)
    for sp in args.splits:
        atoms = utils_data.load_json(atoms_dir / f"{sp}_atoms.json")
        boxes = torch.load(atoms_dir / f"{sp}_boxes.pt", map_location="cpu")
        vids = sorted(atoms)
        if args.max_videos:
            vids = vids[:args.max_videos]
        for vid in vids:
            onsets, label_at = ego_onsets((atoms[vid] or {}).get("interval_atoms", []))
            if not label_at:
                continue
            anchors = [(h, t) for h, t in onsets if h in args.heads]
            all_on = [t for _h, t in onsets]
            cand = [t for t, lab in label_at.items() if lab == "AV-Mov" and t > pre_need
                    and t + int(POST_S * fps) < max(label_at) and all(abs(t - o) > gap for o in all_on)]
            anchors += [("control", t) for t in rng.sample(cand, min(len(cand), args.controls_per_video))]
            if not anchors:
                continue
            for h, _t in anchors:
                n_onsets[h] += 1
            by_frame = defaultdict(list)
            for tube in (boxes.get(vid) or {}).values():
                for f, b in zip(tube["frames"].tolist(), tube["boxes"].tolist()):
                    by_frame[int(f)].append(tuple(b))
            jobs.append((vid, anchors, dict(by_frame), str(frame_dir), str(video_dir), fps))
        del atoms, boxes
    print(f"[Flow diag] {len(jobs)} videos, anchors: " + ", ".join(f"{h} {n}" for h, n in sorted(n_onsets.items())))

    rows = []
    with Pool(args.workers) as pool:
        for vid, r in tqdm(pool.imap_unordered(_video_job, jobs), total=len(jobs), desc="Flow"):
            rows += [(vid, *x) for x in r]

    heads = [h for h in args.heads if n_onsets.get(h)]
    # columns: rel, raw yaw, yaw, raw speed, speed
    def arr(h):
        a = np.array([x[3:] for x in rows if x[1] == h], dtype=np.float64)
        return a.reshape(-1, 5)
    control = arr("control")
    md, js = [], {"settings": {"step": ego_flow.STEP, "hampel_win": ego_flow.HAMPEL_WIN,
                               "smooth_win": ego_flow.SMOOTH_WIN, "version": ego_flow.VERSION,
                               "exp": args.exp, "splits": args.splits}, "onsets": dict(n_onsets)}

    def emit(s=""):
        print(s)
        md.append(s)

    emit(f"# Ego flow diagnostic ({args.exp}, {'+'.join(args.splits)})\n")
    emit("Anchors: " + ", ".join(f"{h} {n_onsets[h]}" for h in heads + ["control"]) + "\n")
    valid = {s: float(np.mean(~np.isnan(np.array([x[5 if s == 'yaw' else 7] for x in rows], float))))
             for s in ("yaw", "speed")} if rows else {}
    emit(f"Valid flow fit (coverage): yaw {valid.get('yaw', 0):.1%}, speed {valid.get('speed', 0):.1%}\n")
    js["coverage"] = valid

    emit("## 1. AUC per second around the onset vs control (raw → denoised)\n")
    emit("| Head | Signal | " + " | ".join(f"{a}..{b} s" for a, b in AUC_BINS) + " |")
    emit("|---|---|" + "---|" * len(AUC_BINS))
    js["auc"] = {}
    for h in heads:
        S = arr(h)
        for sig, (cr, cd) in (("yaw", (1, 2)), ("speed", (3, 4))):
            cells = []
            for a, b in AUC_BINS:
                sel = (S[:, 0] >= a) & (S[:, 0] < b)
                r, d = _auc(S[sel, cr], control[:, cr]), _auc(S[sel, cd], control[:, cd])
                js["auc"][f"{h}|{sig}|{a}"] = {"raw": r, "denoised": d}
                cells.append("–" if d is None else f"{r:.2f}→**{d:.2f}**")
            emit(f"| {h} | {sig} | " + " | ".join(cells) + " |")

    emit("\n## 2. Lead: share of onsets past the control q90 (q10) in the expected direction, per second before\n")
    emit("Same test on the control anchors (any sample of a 1 s bin passes): the false-positive base rate.\n")
    emit("| Head | Signal | " + " | ".join(f"{a}..{b} s" for a, b in LEAD_BINS) + " | control |")
    emit("|---|---|" + "---|" * (len(LEAD_BINS) + 1))
    js["lead"] = {}

    def passes(head, a, b, col, th, sign):
        per = defaultdict(bool)                     # (video, onset) -> any sample of the bin passes
        for vid, _h, t, rel, *vals in rows:
            if _h != head or not a <= rel < b:
                continue
            v = vals[col - 1]
            if not np.isnan(v):
                per[(vid, t)] |= bool(v > th if sign > 0 else v < th)
        return per
    for h in heads:
        if h not in DIRECTION:
            continue
        sig, sign = DIRECTION[h]
        col = 2 if sig == "yaw" else 4
        c = control[:, col][~np.isnan(control[:, col])]
        if not len(c):
            continue
        th = np.percentile(c, 90 if sign > 0 else 10)
        cells = []
        for a, b in LEAD_BINS:
            per = passes(h, a, b, col, th, sign)
            share = float(np.mean(list(per.values()))) if per else None
            js["lead"][f"{h}|{a}"] = share
            cells.append("–" if share is None else f"{share:.0%} (n {len(per)})")
        base = [np.mean(list(p.values())) for a, b in LEAD_BINS if (p := passes("control", a, b, col, th, sign))]
        js["lead"][f"{h}|control"] = float(np.mean(base)) if base else None
        cells.append(f"{np.mean(base):.0%}" if base else "–")
        emit(f"| {h} | {sig} {'>' if sign > 0 else '<'} {th:.3f} | " + " | ".join(cells) + " |")

    emit("\n## 3. Quantiles of the denoised signals (cut points of symbolic levels)\n")
    emit("| Signal | Samples | " + " | ".join(f"q{int(q * 100)}" for q in QS) + " |")
    emit("|---|---|" + "---|" * len(QS))
    js["quantiles"] = {}
    allv = np.array([x[3:] for x in rows], float).reshape(-1, 5)
    for sig, col in (("yaw", 2), ("speed", 4)):
        for name, A in (("control", control), ("all", allv)):
            v = A[:, col][~np.isnan(A[:, col])] if A.size else np.zeros(0)
            qs = np.percentile(v, [100 * q for q in QS]).tolist() if len(v) else [None] * len(QS)
            js["quantiles"][f"{sig}|{name}"] = qs
            emit(f"| {sig} ({name}) | {len(v)} | " + " | ".join("–" if q is None else f"{q:.3f}" for q in qs) + " |")

    for h in heads:
        S = arr(h)
        if len(S):
            _plot(out_dir, h, S, control, n_onsets[h])
    np.savez_compressed(out_dir / "samples.npz",
                        video=np.array([x[0] for x in rows]), head=np.array([x[1] for x in rows]),
                        onset=np.array([x[2] for x in rows]),
                        values=np.array([x[3:] for x in rows], float).reshape(-1, 5))
    (out_dir / "ego_flow_diag.md").write_text("\n".join(md) + "\n")
    (out_dir / "ego_flow_diag.json").write_text(json.dumps(js, indent=1, default=str))
    print(f"\n[Flow diag] {out_dir}")



if __name__ == "__main__":
    main()
