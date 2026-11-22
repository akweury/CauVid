"""Box motion angles per label: do the agents before an onset of a label move their 2D box in a typical
direction, and does it depend on the box size and the place it appeared? (SUMO / ROAD++, 2026-11-22)

Why: in the lane clips (`lane_clips`), an oncoming car that crosses the junction straight shows no label (it is
not yet seen), then Jun, then IncomLane. Its box moves left or down-left and grows, by an amount set by its
speed and size, close to the direction of the cars crossing the junction sideways but not the same. It tends to
appear left of centre in the lower half. The rule model has no fact for this (3x3 grid only, motion atoms
off) and scores these onsets low; the MLP scores them higher.

Per agent and frame t with a box (the box centre in pixels, image W x H):
  ang_k    direction of the move over the last K frames, c(t) - c(t - K);
  ang_app  direction of the move since the agent appeared, c(t) - c(t0), t0 the first frame of its current
           track segment (smooth: the whole path so far);
  Angles are in degrees: 0 right, 90 up, 180 left, 270 down. A move below MIN_MOVE_PX gives no angle ("still").
  speed    |c(t) - c(t - K)| / (K x box height): box heights per frame;
  scale    log(area(t) / area(t - K)) / K: > 0 the box grows;
  cell     the 3x3 image cell of the segment's first centre (where it appeared);
  size     the box height tertile of its class (small / mid / large).
Per label L of the agent's class (e.g. Car:IncomLane):
  pre      the at-risk frames (L off in the min_off_frames before t) whose onset of L comes within W frames;
  risk     every at-risk frame of the class (the base).
Reported per label (train):
  - the circular mean and R (1: all the same direction) of ang_k and ang_app;
  - the shortest arc holding 80% of the pre angles;
  - the same within each (cell, size) group with >= MIN_GROUP pre frames, and how much the groups concentrate
    the angles (mean R within groups vs R overall).
The learned ranges (arcs on train) are then checked on val as a fact "the angle lies in L's arc", at every
at-risk frame of the class:
  - precision, recall and lift (precision / base rate) of the onset within W frames;
  - per label, for ang_k and ang_app, with one arc per label or one per (cell, size) group.

    ./run_roadpp_docker.sh --gpu 0 --box_angles -- --exp sumo_maps_v7 --machine dgx-jing \\
        --run_id 20261121-173934_dgx-jing_1ca554
Writes <output_dir>/box_angles/box_angles.json and box_angles.pdf (per label: polar histograms of the pre vs the
at-risk angles, and ang_app by the cell where the agent appeared).
"""
import math
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from src.exp_roadpp import config, step05, utils_data, utils_pipe

K = 5                         # frames of ang_k (0.5 s at 10 fps)
WINDOW = 20                   # frames: an onset within this many frames after t (the 2 s of the rules)
MIN_MOVE_PX = 4.0             # a move below this (over K frames / since appearance) gives no angle
ARC_SHARE = 0.8               # the learned range holds this share of the pre angles
MIN_GROUP = 30                # pre frames of a (cell, size) group to learn its own arc
MIN_PRE = 50                  # pre frames of a label to report it
FRAME_PX = (1920, 1280)
SIZES = ("small", "mid", "large")


# -------------- Per video: angles and label targets --------------
def video_samples(X, boxes, has_box, classes, m_off, class_index):
    """Per (t, n) with a box at t and t - 1: the motion values and, per label atom, hold / at_risk / onset
    within WINDOW (bool (S, A) each)."""
    T, N, A = X.shape
    W, H = FRAME_PX
    b = boxes.nan_to_num(0.0)
    cx, cy = (b[..., 0] + b[..., 2]) / 2 * W, (b[..., 1] + b[..., 3]) / 2 * H
    h = (b[..., 3] - b[..., 1]).clamp(min=1e-6) * H
    area = ((b[..., 2] - b[..., 0]).clamp(min=1e-6) * W) * h
    # The first frame of each agent's current track segment (a frame without a box starts a new one).
    start = torch.zeros((T, N), dtype=torch.long)
    for t in range(1, T):
        start[t] = torch.where(has_box[t] & has_box[t - 1], start[t - 1], torch.full((N,), t))
    # At risk at t: off in t - m_off .. t - 1 (and seen at t - 1); onset within WINDOW: on at some u in t..t+W-1.
    Xi = X.long()
    csum = torch.cat([torch.zeros((1, N, A), dtype=torch.long), Xi.cumsum(0)])           # (T + 1, N, A)
    t_idx = torch.arange(T)
    lo = (t_idx - m_off).clamp(min=0)
    off_before = (csum[t_idx] - csum[lo]) == 0                                            # (T, N, A)
    seen_before = torch.zeros((T, N), dtype=torch.bool)
    seen_before[1:] = has_box[:-1]
    at_risk = off_before & seen_before[..., None] & (t_idx >= m_off)[:, None, None]
    big = T + WINDOW + 1
    idx = torch.where(X, t_idx[:, None, None].expand(T, N, A), torch.full((T, N, A), big))
    next_on = idx.flip(0).cummin(0).values.flip(0)                                        # first on frame >= t
    soon = (next_on - t_idx[:, None, None]) < WINDOW

    valid = has_box & seen_before
    t_s, n_s = torch.nonzero(valid, as_tuple=True)
    tk = (t_s - K).clamp(min=0)
    has_k = has_box[tk, n_s] & (t_s >= K) & (start[t_s, n_s] <= tk)

    def angle(dx, dy, ok):
        a = torch.rad2deg(torch.atan2(-dy, dx)) % 360
        return torch.where(ok & (torch.hypot(dx, dy) >= MIN_MOVE_PX), a, torch.full_like(a, float("nan")))
    dxk, dyk = cx[t_s, n_s] - cx[tk, n_s], cy[t_s, n_s] - cy[tk, n_s]
    t0 = start[t_s, n_s]
    dxa, dya = cx[t_s, n_s] - cx[t0, n_s], cy[t_s, n_s] - cy[t0, n_s]
    col = (cx[t0, n_s] / W * 3).clamp(0, 2.999).long()
    row = (cy[t0, n_s] / H * 3).clamp(0, 2.999).long()
    return {
        "cls": torch.tensor([class_index.get(classes[n], -1) for n in n_s.tolist()], dtype=torch.long),
        "ang_k": angle(dxk, dyk, has_k), "ang_app": angle(dxa, dya, t_s > t0),
        "speed": torch.where(has_k, torch.hypot(dxk, dyk) / (K * h[t_s, n_s]), torch.full_like(dxk, float("nan"))),
        "scale": torch.where(has_k, torch.log(area[t_s, n_s] / area[tk, n_s]) / K, torch.full_like(dxk, float("nan"))),
        "height": h[t_s, n_s], "cell": row * 3 + col, "age": t_s - t0,
        "hold": X[t_s, n_s], "risk": at_risk[t_s, n_s], "pre": at_risk[t_s, n_s] & soon[t_s, n_s],
    }


def collect(cfg, atom_file, box_file, A, names, desc):
    atoms = utils_data.load_json(atom_file)
    boxes = torch.load(box_file, map_location="cpu")
    classes_all = sorted({nm.split(":")[0] for nm in names})
    class_index = {c: i for i, c in enumerate(classes_all)}
    parts = defaultdict(list)
    for video_id, video_atoms in tqdm(atoms.items(), desc=desc):
        X = step05.build_video_tensor(video_atoms, A)
        if X.shape[0] <= K + 1:
            continue
        video_boxes, classes, has_box = step05._agent_boxes(boxes, video_id, video_atoms, X)
        s = video_samples(X, video_boxes, has_box, classes, int(cfg["min_off_frames"]), class_index)
        for k, v in s.items():
            parts[k].append(v)
    return {k: torch.cat(v) for k, v in parts.items()}, classes_all


# -------------- Circular statistics --------------
def circ(angles):
    """(mean direction, R) of degrees; (nan, 0) if empty."""
    a = angles[~torch.isnan(angles)]
    if not len(a):
        return float("nan"), 0.0
    r = torch.deg2rad(a)
    c, s = torch.cos(r).mean(), torch.sin(r).mean()
    return float(torch.rad2deg(torch.atan2(s, c)) % 360), float(torch.hypot(c, s))


def arc(angles, share=ARC_SHARE):
    """(start, width) in degrees of the shortest arc holding share of the angles; None if too few."""
    a = torch.sort(angles[~torch.isnan(angles)]).values
    n = len(a)
    if n < 5:
        return None
    k = max(int(math.ceil(share * n)), 1)
    ext = torch.cat([a, a + 360])
    widths = ext[k - 1:k - 1 + n] - ext[:n]
    i = int(widths.argmin())
    return float(a[i]), float(widths[i])


def in_arc(angles, a):
    if a is None:
        return torch.zeros_like(angles, dtype=torch.bool)
    start, width = a
    d = (angles - start) % 360
    return ~torch.isnan(angles) & (d <= width)


def size_bins(samples, num_classes):
    """(S,) 0 / 1 / 2: the box height tertile of the sample's class (train tertiles)."""
    edges = torch.zeros((num_classes, 2))
    for c in range(num_classes):
        hc = samples["height"][samples["cls"] == c]
        if len(hc) > 10:
            edges[c] = torch.quantile(hc[:1_000_000], torch.tensor([1 / 3, 2 / 3]))
    return edges


def size_of(samples, edges):
    e = edges[samples["cls"].clamp(min=0)]
    return (samples["height"] > e[:, 0]).long() + (samples["height"] > e[:, 1]).long()


# -------------- Analysis --------------
def analyse(train, val, names, classes_all, label_ids, edges):
    out = {}
    tr_size, va_size = size_of(train, edges), size_of(val, edges)
    for a in label_ids:
        cls = names[a].split(":")[0]
        c = classes_all.index(cls)
        tr_c, va_c = train["cls"] == c, val["cls"] == c
        pre = tr_c & train["pre"][:, a]
        if int(pre.sum()) < MIN_PRE:
            continue
        risk = tr_c & train["risk"][:, a]
        entry = {"class": cls, "pre_frames": int(pre.sum()), "risk_frames": int(risk.sum()),
                 "base_rate": float(pre.sum()) / max(float(risk.sum()), 1)}
        for key in ("ang_k", "ang_app"):
            ang = train[key]
            mean_p, r_p = circ(ang[pre])
            mean_r, r_r = circ(ang[risk])
            moving = float((~torch.isnan(ang[pre])).float().mean())
            entry[key] = {"pre_mean": mean_p, "pre_R": r_p, "risk_mean": mean_r, "risk_R": r_r,
                          "pre_moving": moving, "pre_arc": arc(ang[pre])}
            # Groups by where it appeared and its size.
            groups, rs, ws = {}, [], []
            for cell in range(9):
                for sz in range(3):
                    g = pre & (train["cell"] == cell) & (tr_size == sz)
                    if int(g.sum()) >= MIN_GROUP:
                        m, r = circ(ang[g])
                        groups[f"r{cell // 3}c{cell % 3}_{SIZES[sz]}"] = {"n": int(g.sum()), "mean": m, "R": r,
                                                                         "arc": arc(ang[g])}
                        rs.append(r)
                        ws.append(int(g.sum()))
            entry[key]["groups"] = groups
            entry[key]["group_R"] = sum(r * w for r, w in zip(rs, ws)) / max(sum(ws), 1) if ws else None
            # The learned ranges as a fact on val: one arc per label, or per (cell, size) group (else the label's).
            v_risk = va_c & val["risk"][:, a]
            v_y = val["pre"][:, a][v_risk]
            facts = {"label_arc": in_arc(val[key][v_risk], entry[key]["pre_arc"])}
            grp = torch.zeros(int(v_risk.sum()), dtype=torch.bool)
            v_cell, v_size, v_ang = val["cell"][v_risk], va_size[v_risk], val[key][v_risk]
            for cell in range(9):
                for sz in range(3):
                    m = (v_cell == cell) & (v_size == sz)
                    gk = f"r{cell // 3}c{cell % 3}_{SIZES[sz]}"
                    a_g = groups[gk]["arc"] if gk in groups else entry[key]["pre_arc"]
                    grp[m] = in_arc(v_ang[m], a_g)
            facts["group_arc"] = grp
            base = float(v_y.float().mean()) if len(v_y) else 0.0
            entry[key]["val"] = {}
            for fk, f in facts.items():
                n_f, hit = int(f.sum()), int((f & v_y).sum())
                prec = hit / n_f if n_f else 0.0
                entry[key]["val"][fk] = {"fires": n_f, "precision": prec, "recall": hit / max(int(v_y.sum()), 1),
                                         "lift": prec / base if base > 0 else None, "base": base}
        # Scale and speed of the pre frames vs the at-risk frames (median).
        for key in ("scale", "speed"):
            vp, vr = train[key][pre], train[key][risk]
            vp, vr = vp[~torch.isnan(vp)], vr[~torch.isnan(vr)]
            entry[key] = {"pre_median": float(vp.median()) if len(vp) else None,
                          "risk_median": float(vr.median()) if len(vr) else None}
        entry["pre_cells"] = torch.bincount(train["cell"][pre], minlength=9).tolist()
        out[names[a]] = entry
    return out


def plot(result, train, names, classes_all, edges, label_ids, path, top=40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    bins = torch.linspace(0, 360, 37)
    centres = torch.deg2rad((bins[:-1] + bins[1:]) / 2).numpy()
    width = math.radians(10)

    def rose(ax, ang, color, label, norm=True):
        a = ang[~torch.isnan(ang)]
        if not len(a):
            return
        hcount = torch.histc(a, bins=36, min=0, max=360)
        if norm:
            hcount = hcount / hcount.sum()
        ax.bar(centres, hcount.numpy(), width=width, color=color, alpha=0.55, label=label, edgecolor="none")

    order = sorted(result, key=lambda k: -result[k]["pre_frames"])[:top]
    size = size_of(train, edges)
    with PdfPages(path) as pdf:
        for label in order:
            e = result[label]
            a = names.index(label)
            c = classes_all.index(e["class"])
            pre = (train["cls"] == c) & train["pre"][:, a]
            risk = (train["cls"] == c) & train["risk"][:, a]
            fig = plt.figure(figsize=(18, 8))
            gs = fig.add_gridspec(3, 7)
            for i, key in enumerate(("ang_k", "ang_app")):
                ax = fig.add_subplot(gs[0:3, 2 * i:2 * i + 2], projection="polar")
                rose(ax, train[key][risk], "#999999", "at risk")
                rose(ax, train[key][pre], "#d62728", "before onset")
                s = e[key]
                arc_txt = f"arc80 {s['pre_arc'][0]:.0f}+{s['pre_arc'][1]:.0f}" if s["pre_arc"] else "arc80 -"
                v = s["val"]
                ax.set_title(f"{key}: R pre {s['pre_R']:.2f} / risk {s['risk_R']:.2f}; groups R "
                             f"{s['group_R'] if s['group_R'] is not None else float('nan'):.2f}\n{arc_txt}; moving "
                             f"{s['pre_moving']:.0%}; val lift {v['label_arc']['lift'] or 0:.1f} / groups "
                             f"{v['group_arc']['lift'] or 0:.1f}", fontsize=8)
                ax.set_yticklabels([])
                if i == 0:
                    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.3, -0.25))
            # ang_app before the onset by where the agent appeared (3x3), all sizes; the size as colour.
            for cell in range(9):
                ax = fig.add_subplot(gs[cell // 3, 4 + cell % 3], projection="polar")
                for sz, colr in zip(range(3), ("#1f77b4", "#ff7f0e", "#2ca02c")):
                    g = pre & (train["cell"] == cell) & (size == sz)
                    rose(ax, train["ang_app"][g], colr, SIZES[sz], norm=False)
                ax.set_title(f"appeared r{cell // 3}c{cell % 3}: {int((pre & (train['cell'] == cell)).sum())}",
                             fontsize=7)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                if cell == 0:
                    ax.legend(fontsize=6, loc="lower left", bbox_to_anchor=(-0.4, -0.3))
            fig.suptitle(f"{label}: {e['pre_frames']} frames before an onset (within {WINDOW} f), base rate "
                         f"{e['base_rate']:.3f}; angles 0 right, 90 up, 180 left, 270 down; scale median pre "
                         f"{e['scale']['pre_median'] or 0:+.3f} vs risk {e['scale']['risk_median'] or 0:+.3f}",
                         fontsize=11)
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--angle_labels", type=str, default="",
                        help="label atoms to analyse (comma list); default every label with >= MIN_PRE pre frames")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id if getattr(args, "run_id", None) \
        else args.output_dir / "box_angles"
    cfg = step05.get_cfg(config.get_step_05_input(args))
    out_dir = args.output_dir / "box_angles"
    os.makedirs(out_dir, exist_ok=True)
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    A = len(unique_atoms)
    names = [step05._atom_name(sig) for sig in unique_atoms]
    train, classes_all = collect(cfg, cfg["train_atom_file"], cfg["train_box_file"], A, names, "Angles (train)")
    val, _ = collect(cfg, cfg["val_atom_file"], cfg["val_box_file"], A, names, "Angles (val)")
    edges = size_bins(train, len(classes_all))
    wanted = [x for x in args.angle_labels.split(",") if x]
    label_ids = [names.index(x) for x in wanted] if wanted else [
        a for a, nm in enumerate(names) if ":" in nm and not nm.startswith(("av:", "TL:", "pos:", "mot:"))]
    result = analyse(train, val, names, classes_all, label_ids, edges)

    print(f"\n[box_angles] {len(result)} labels; angles 0 right, 90 up, 180 left, 270 down; R: 1 = one direction. "
          f"val lift: precision of 'angle in the train arc' / base rate")
    print(f"  {'label':<20}{'pre':>7}{'mean_k':>8}{'R_k':>6}{'R_app':>7}{'R_grp':>7}{'arc_app':>12}"
          f"{'moving':>8}{'lift_k':>8}{'lift_app':>9}{'lift_grp':>9}{'rec_grp':>8}")
    for label, e in sorted(result.items(), key=lambda x: -x[1]["pre_frames"]):
        k, ap = e["ang_k"], e["ang_app"]
        arc_txt = f"{ap['pre_arc'][0]:.0f}+{ap['pre_arc'][1]:.0f}" if ap["pre_arc"] else "-"
        print(f"  {label:<20}{e['pre_frames']:>7}{k['pre_mean']:>8.0f}{k['pre_R']:>6.2f}{ap['pre_R']:>7.2f}"
              f"{ap['group_R'] or 0:>7.2f}{arc_txt:>12}{ap['pre_moving']:>8.0%}"
              f"{k['val']['label_arc']['lift'] or 0:>8.1f}{ap['val']['label_arc']['lift'] or 0:>9.1f}"
              f"{ap['val']['group_arc']['lift'] or 0:>9.1f}{ap['val']['group_arc']['recall']:>8.2f}")
    utils_data.save_json({"settings": {"K": K, "window": WINDOW, "min_move_px": MIN_MOVE_PX, "arc_share": ARC_SHARE,
                                       "min_group": MIN_GROUP, "frame_px": FRAME_PX,
                                       "size_edges_px": {c: edges[i].tolist() for i, c in enumerate(classes_all)}},
                          "labels": result}, out_dir / "box_angles.json")
    plot(result, train, names, classes_all, edges, label_ids, out_dir / "box_angles.pdf")
    print(f"[box_angles] -> {out_dir}")


if __name__ == "__main__":
    main()
