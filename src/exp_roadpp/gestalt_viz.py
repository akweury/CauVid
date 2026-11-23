"""Gestalt group pictures: which boxes `gestalt.groups` puts together, drawn in the camera view (2026-11-23).

On frames of the val videos, every box is drawn in the colour of its group, grey if in none. Each drawing has:
  - an arrow along the box's move over gestalt.K frames;
  - the dashed union box and the mean direction of each group;
  - the box's class and labels;
  - a caption per group: kind, members and direction.
The frames are picked from the videos per kind of scene:
  veh_group    a vehicle group of >= 3 members;
  ped_group    a pedestrian group;
  two_groups   >= 2 groups at once;
  still_near   a still agent next to a moving group (the queue starts, one waits);
  single       moving agents close to each other but in no group (to see what does not group).
Plus a few clips (--viz_clips) around such frames, and counts over the videos read: groups per frame, sizes,
kinds, the share of moving agents in a group.

    ./run_roadpp_docker.sh --gpu 0 --gestalt_viz -- --exp sumo_maps_v7 --machine dgx-jing \\
        --run_id 20261123-gestalt4_lab --shape_phases True --gestalt True
Writes <output_dir>/gestalt_viz/: frames.pdf (one page per frame), frame_<kind>_<video>_f<t>.png,
clip_<video>_f<t>.mp4, stats.json.
"""
import math
import os
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from src.exp_roadpp import config, gestalt, step05, utils_data, utils_pipe

COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2", "#8c564b", "#bcbd22",
          "#7f7f7f"]
QUAD = gestalt.QUADRANTS


def frame_kinds(g, t, has_box):
    """The picking kinds a frame t of a video's `gestalt.groups` falls under."""
    out = []
    groups_t = [k for k in range(g["size"].shape[1]) if g["is_group"][t, k]]
    kinds = [int(g["gkind"][t, k]) for k in groups_t]
    sizes = [int(g["size"][t, k]) for k in groups_t]
    if any(kd == 0 and sz >= 3 for kd, sz in zip(kinds, sizes)):
        out.append("veh_group")
    if any(kd == 1 for kd in kinds):
        out.append("ped_group")
    if len(groups_t) >= 2:
        out.append("two_groups")
    moving = ~torch.isnan(g["angle"][t])
    still = has_box[t] & ~moving
    member = g["in_group"][t]
    if bool((g["near"][t] & still[:, None] & member[None, :]).any()):
        out.append("still_near")
    loose = moving & ~member
    if bool((g["near"][t] & loose[:, None] & loose[None, :]).any()):
        out.append("single")
    return out


def draw(ax, boxes_t, has_t, classes, labels_t, g, t, title):
    """One camera view: the boxes by group, arrows, group union boxes, a caption."""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    W, H = gestalt.FRAME_SIZE_PX
    ax.add_patch(Rectangle((0, 0), W, H / 2, fc="#e8f1fa", zorder=0))
    ax.add_patch(Rectangle((0, H / 2), W, H / 2, fc="#d9d9d9", zorder=0))
    N = boxes_t.shape[0]
    label = g["label"][t]
    group_ids = [k for k in range(g["size"].shape[1]) if g["is_group"][t, k]]
    color_of = {k: COLORS[i % len(COLORS)] for i, k in enumerate(group_ids)}
    for n in range(N):
        if not has_t[n]:
            continue
        x1, y1, x2, y2 = (float(v) for v in boxes_t[n])
        k = int(label[n])
        col = color_of.get(k, "#9e9e9e") if bool(g["in_group"][t, n]) else "#9e9e9e"
        lw = 2.2 if col != "#9e9e9e" else 1.0
        ax.add_patch(Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H, fill=False, ec=col, lw=lw, zorder=3))
        text = f"{classes[n]}" + (f" {labels_t[n]}" if labels_t[n] else "")
        ax.text(x1 * W, y1 * H - 4, text, fontsize=5.5, color=col if col != "#9e9e9e" else "#555555", zorder=4)
        a = float(g["angle"][t, n])
        if a == a:
            cx, cy = (x1 + x2) / 2 * W, (y1 + y2) / 2 * H
            L = 45
            ax.add_patch(FancyArrowPatch((cx, cy), (cx + L * math.cos(math.radians(a)), cy - L * math.sin(math.radians(a))),
                                         arrowstyle="-|>", mutation_scale=7, color=col, lw=1.2, zorder=5))
    lines = []
    for i, k in enumerate(group_ids):
        members = [n for n in range(N) if int(label[n]) == k and bool(g["in_group"][t, n])]
        bx = boxes_t[members]
        gx1, gy1 = float(bx[:, 0].min()) * W, float(bx[:, 1].min()) * H
        gx2, gy2 = float(bx[:, 2].max()) * W, float(bx[:, 3].max()) * H
        col = color_of[k]
        ax.add_patch(Rectangle((gx1 - 6, gy1 - 6), gx2 - gx1 + 12, gy2 - gy1 + 12, fill=False, ec=col, lw=1.0,
                               ls="--", zorder=2))
        d = float(g["gdir"][t, k])
        kind = gestalt.KINDS[int(g["gkind"][t, k])]
        q = QUAD[int(g["gq"][t, k])] if int(g["gq"][t, k]) >= 0 else "-"
        lines.append((col, f"G{i + 1} {kind}: {len(members)} members ({', '.join(sorted(Counter(classes[n] for n in members)))}), "
                           f"direction {d:.0f} deg ({q})"))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    for i, (col, line) in enumerate(lines or [("#555555", "no group")]):
        ax.text(0.01, -0.04 - 0.045 * i, line, transform=ax.transAxes, fontsize=7.5, color=col, va="top")


def short_labels(X_t, names):
    """Per agent the action labels it holds ("Stop,Jun")."""
    out = []
    for n in range(X_t.shape[0]):
        acts = [names[a].split(":", 1)[1] for a in torch.nonzero(X_t[n]).flatten().tolist() if ":" in names[a]]
        out.append(",".join(acts[:3]))
    return out


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--viz_split", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--viz_videos", type=int, default=200, help="videos to read (random)")
    parser.add_argument("--viz_per_kind", type=int, default=6, help="frames per picking kind")
    parser.add_argument("--viz_clips", type=int, default=3, help="clips (4 s) around picked frames")
    parser.add_argument("--seed", type=int, default=0)
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id if getattr(args, "run_id", None) \
        else args.output_dir / "gestalt_viz"
    cfg = step05.get_cfg(config.get_step_05_input(args))
    out_dir = args.output_dir / "gestalt_viz"
    os.makedirs(out_dir, exist_ok=True)
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    names = [step05._atom_name(sig) for sig in unique_atoms]
    A = len(unique_atoms)
    atoms = utils_data.load_json(cfg[f"{args.viz_split}_atom_file"])
    boxes_all = torch.load(cfg[f"{args.viz_split}_box_file"], map_location="cpu")
    rng = random.Random(args.seed)
    vids = sorted(atoms)
    rng.shuffle(vids)
    vids = vids[:args.viz_videos]

    stats = Counter()
    sizes, groups_per_frame = Counter(), Counter()
    candidates = defaultdict(list)
    cache = {}
    for vid in tqdm(vids, desc="Groups"):
        X = step05.build_video_tensor(atoms[vid], A)
        if X.shape[0] <= gestalt.K + 1:
            continue
        boxes, classes, has_box = step05._agent_boxes(boxes_all, vid, atoms[vid], X)
        g = gestalt.groups(boxes, has_box, classes)
        cache[vid] = (X, boxes, classes, has_box, g)
        moving = ~torch.isnan(g["angle"]) & has_box
        stats["moving_agent_frames"] += int(moving.sum())
        stats["grouped_agent_frames"] += int(g["in_group"].sum())
        for t in range(X.shape[0]):
            ks = torch.nonzero(g["is_group"][t]).flatten().tolist()
            groups_per_frame[len(ks)] += 1
            for k in ks:
                sizes[int(g["size"][t, k])] += 1
                stats[f"groups_{gestalt.KINDS[int(g['gkind'][t, k])]}"] += 1
            if t % 5 == 0:                                  # every 5th frame as a picking candidate
                for kind in frame_kinds(g, t, has_box):
                    candidates[kind].append((vid, t))
    picked = []
    for kind in ("veh_group", "ped_group", "two_groups", "still_near", "single"):
        pool = candidates[kind]
        rng.shuffle(pool)
        seen_v = set()
        for vid, t in pool:
            if vid in seen_v:
                continue                                    # one frame per video and kind
            seen_v.add(vid)
            picked.append((kind, vid, t))
            if len([p for p in picked if p[0] == kind]) >= args.viz_per_kind:
                break

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out_dir / "frames.pdf") as pdf:
        for kind, vid, t in tqdm(picked, desc="Frames"):
            X, boxes, classes, has_box, g = cache[vid]
            fig, ax = plt.subplots(figsize=(12, 9.5))
            fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.2)
            draw(ax, boxes[t].nan_to_num(0.0), has_box[t], [str(c) for c in classes], short_labels(X[t], names),
                 g, t, f"[{kind}] {vid} frame {t}")
            fig.savefig(out_dir / f"frame_{kind}_{vid}_f{t}.png", dpi=110)
            pdf.savefig(fig)
            plt.close(fig)
    # Clips: frames t - 20 .. t + 20 around the first picked frames.
    ffmpeg = shutil.which("ffmpeg")
    for kind, vid, t in picked[:args.viz_clips] if ffmpeg else []:
        X, boxes, classes, has_box, g = cache[vid]
        lo, hi = max(gestalt.K, t - 20), min(X.shape[0] - 1, t + 20)
        with tempfile.TemporaryDirectory() as tmp:
            for i, f in enumerate(range(lo, hi + 1)):
                fig, ax = plt.subplots(figsize=(12, 9.5))
                fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.2)
                draw(ax, boxes[f].nan_to_num(0.0), has_box[f], [str(c) for c in classes], short_labels(X[f], names),
                     g, f, f"[{kind}] {vid} frame {f}")
                fig.savefig(Path(tmp) / f"{i:05d}.png", dpi=80)
                plt.close(fig)
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", "10", "-i", str(Path(tmp) / "%05d.png"),
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            str(out_dir / f"clip_{vid}_f{t}.mp4")], check=True)
    frames = sum(groups_per_frame.values())
    summary = {"videos": len(cache), "frames": frames,
               "share_of_moving_agent_frames_in_a_group": stats["grouped_agent_frames"] / max(stats["moving_agent_frames"], 1),
               "groups_per_frame": dict(sorted(groups_per_frame.items())), "group_sizes": dict(sorted(sizes.items())),
               "groups_by_kind": {k: stats[f"groups_{k}"] for k in gestalt.KINDS},
               "picked": [{"kind": k, "video": v, "frame": t} for k, v, t in picked]}
    utils_data.save_json(summary, out_dir / "stats.json")
    print(f"[gestalt_viz] {len(cache)} videos, {frames} frames; moving agent-frames in a group: "
          f"{summary['share_of_moving_agent_frames_in_a_group']:.1%}; groups per frame {summary['groups_per_frame']}; "
          f"sizes {summary['group_sizes']}; by kind {summary['groups_by_kind']}")
    print(f"[gestalt_viz] {len(picked)} frames -> {out_dir}")


if __name__ == "__main__":
    main()
