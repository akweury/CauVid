"""Schematic images of a generated SUMO video, for checking labels and rules by eye (no realistic
rendering): left the top view (roads, every agent, the ego's field of view; agents the camera sees in
colour, the others grey), right the camera view with exactly the boxes and labels of {vid}_gt.json.

    python -m src.exp_roadpp.synth.sumo.render --dataset <dataset dir> --num 4 [--vid sumo_00003 --frames 50 120] [--maps]
    python -m src.exp_roadpp.synth.sumo.render --dataset <dataset dir> --video [--vid sumo_00003] [--fps 10]
--video: every frame of one video (default: the first one with an executed rule the ego sees) as an mp4
(ffmpeg), <dataset>/render/<vid>.mp4.
    ./run_roadpp_docker.sh --gpu none --sumo_render -- --dataset /storage-01/ml-jsha/CauVid_Data/synthetic/sumo_junction_v0
Without --frames, frames with something to see are picked: R1 / R2 executions the ego sees, turns,
red-light stops. Images: <dataset>/render/<vid>_f<frame>.png"""
import argparse
import functools
import gzip
import json
import math
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Wedge

from src.exp_roadpp.synth.sumo import observe, traffic
from src.exp_roadpp.synth.sumo.observe import heading

COLORS = {"Car": "#1f77b4", "MedVeh": "#17becf", "LarVeh": "#9467bd", "Bus": "#8c564b", "Cyc": "#2ca02c",
          "Mobike": "#bcbd22", "Ped": "#d62728", "TL": "#ff7f0e"}


def _offset(shape, d):
    """The polyline shifted d to the left."""
    out = []
    for i, (x, y) in enumerate(shape):
        a, b = shape[max(0, i - 1)], shape[min(len(shape) - 1, i + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy) or 1.0
        out.append((x - dy / n * d, y + dx / n * d))
    return out


@functools.lru_cache(maxsize=4)
def _read_net(net_file):
    import sumolib
    return sumolib.net.readNet(str(net_file), withInternal=True, withPedestrianConnections=True)


@functools.lru_cache(maxsize=2)
def _video(dataset, vid):
    """(labels, gt data, world states, firings) of one video, read once for all its frames."""
    labels = json.loads((dataset / "gt" / "label.json").read_text())
    data = json.loads((dataset / "gt" / f"{vid}_gt.json").read_text())["data"]
    with gzip.open(dataset / "meta" / "world" / f"{vid}.json.gz", "rt") as f:
        world = json.load(f)
    firings = json.loads((dataset / "meta" / "firings" / f"{vid}.json").read_text())
    return labels, data, world, firings


def draw_network(ax, net_file):
    """Draws the network; returns the position of junction C, or the centre of a grid (netconvert shifts
    the coordinates)."""
    net = _read_net(str(net_file))
    for edge in net.getEdges(withInternal=True):
        func = edge.getFunction()
        for lane in edge.getLanes():
            shape = lane.getShape()
            if func == "walkingarea":
                ax.add_patch(Polygon(shape, closed=True, fc="#d9d9d9", ec="none", zorder=1))
                continue
            w = lane.getWidth()
            poly = _offset(shape, w / 2) + _offset(shape, -w / 2)[::-1]
            if func == "crossing":
                ax.add_patch(Polygon(poly, closed=True, fc="white", ec="#999999", hatch="||", lw=0.3, zorder=2))
            elif lane.allows("pedestrian") and not lane.allows("passenger"):
                ax.add_patch(Polygon(poly, closed=True, fc="#d9d9d9", ec="none", zorder=1))
            else:
                ax.add_patch(Polygon(poly, closed=True, fc="#555555", ec="#777777", lw=0.2, zorder=1))
    if net.hasNode("C"):
        return net.getNode("C").getCoord()
    (x0, y0), (x1, y1) = net.getBBoxXY()
    return (x0 + x1) / 2, (y0 + y1) / 2


def network_of(dataset, vid):
    """(net file, map name or None) of a video: its map in a multi-map dataset, else the one junction."""
    index = dataset / "meta" / "maps.json"
    if index.exists():
        maps = json.loads(index.read_text())
        m = maps["videos"][vid]
        return dataset / "sumo" / maps["maps"][m]["net"], f"{m} {maps['maps'][m]['name']}"
    return dataset / "sumo" / "junction.net.xml", None


def render_maps(dataset, out_file, cols=8):
    """All junctions of a multi-map dataset in one image (top views, 2 x 120 m)."""
    maps = json.loads((dataset / "meta" / "maps.json").read_text())["maps"]
    rows = math.ceil(len(maps) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.8 * rows))
    for ax, (m, info) in zip(axes.flat, maps.items()):
        cx, cy = draw_network(ax, dataset / "sumo" / info["net"])
        half = 520 if info["spec"].get("layout") == "grid" else 120
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{m}\n{info['name']} {info['spec']['speed']} m/s", fontsize=7)
    for ax in list(axes.flat)[len(maps):]:
        ax.axis("off")
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=90)
    plt.close(fig)


def _short(labels, ids, kind):
    return [labels[kind][i] for i in ids]


def render(dataset, vid, frame, out_file, note=None, highlight=None):
    """One image of frame (1-based) of video vid; note: an extra line for the title of the top view;
    highlight: the tube_uid of an agent drawn with a thick red outline in both views (`lane_clips`)."""
    hl = highlight.split("-", 1)[1] if highlight and "-" in highlight else highlight
    labels, data, worlds, firings = _video(dataset, vid)
    world = worlds[frame - 1]
    annos = data["frames"][str(frame)]["annos"]
    av = labels["av_action_labels"][data["frames"][str(frame)]["av_action_ids"][0]]
    seen = {a["tube_uid"].split("-", 1)[1] for a in annos.values()}

    fig, (top, cam_ax) = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1, 1.5]})
    # -------------- top view --------------
    net_file, map_name = network_of(dataset, vid)
    draw_network(top, net_file)
    ego = world["veh"]["ego"]
    cam = observe.Camera(ego)
    yaw = 90 - ego["angle"]                                     # SUMO angle -> matplotlib degrees
    top.add_patch(Wedge((cam.x, cam.y), observe.MAX_RANGE, yaw - observe.HFOV_DEG / 2, yaw + observe.HFOV_DEG / 2,
                        fc="#ffe680", alpha=0.35, ec="#e6b800", zorder=3))

    def box(x, y, angle, length, width, color, alpha, z, mark=False):
        corners = observe.footprint(x, y, angle, length, width)
        top.add_patch(Polygon([corners[0], corners[1], corners[3], corners[2]], closed=True, fc=color,
                              ec="red" if mark else "black", lw=2.5 if mark else 0.4, alpha=alpha, zorder=z))

    for vid_, v in world["veh"].items():
        if vid_ == "ego":
            box(v["x"], v["y"], v["angle"], v["length"], v["width"], "#000000", 1.0, 6)
            continue
        on = vid_ in seen
        box(v["x"], v["y"], v["angle"], v["length"], v["width"], COLORS[v["cls"]] if on else "#bbbbbb",
            1.0 if on else 0.6, 5, mark=vid_ == hl)
        if v["signals"] & 4:
            top.plot(v["x"], v["y"], marker="*", color="orange", ms=10, zorder=7)
    for pid, p in world["ped"].items():
        on = pid in seen
        top.add_patch(plt.Circle((p["x"], p["y"]), 0.6, fc=COLORS["Ped"] if on else "#bbbbbb",
                                 ec="red" if pid == hl else "black", lw=2.5 if pid == hl else 0.3, zorder=5))
    if world.get("tl"):
        tl = world["tl"]
        c = {"r": "red", "y": "gold", "g": "green"}.get(tl["state"][0].lower(), "grey")
        top.plot(tl.get("far_x", tl["x"]), tl.get("far_y", tl["y"]), marker="s", color=c, ms=7, mec="black", zorder=7)
    hx, hy = heading(ego["angle"])
    half = 45
    top.set_xlim(ego["x"] + hx * 25 - half, ego["x"] + hx * 25 + half)
    top.set_ylim(ego["y"] + hy * 25 - half, ego["y"] + hy * 25 + half)
    top.set_aspect("equal")
    top.set_xticks([]); top.set_yticks([])
    top.set_title(f"{vid} frame {frame} (t={world['t']:.1f} s)  ego: {av}, {ego['speed']:.1f} m/s"
                  + (f"\n{map_name}" if map_name else "") + (f"\n{note}" if note else ""), fontsize=10)

    # -------------- camera view (the GT boxes) --------------
    W, H = observe.W_IMG, observe.H_IMG
    cam_ax.add_patch(Rectangle((0, 0), W, H / 2, fc="#cfe8ff", zorder=0))
    cam_ax.add_patch(Rectangle((0, H / 2), W, H / 2, fc="#9e9e9e", zorder=0))
    items = sorted(annos.values(), key=lambda a: (a["box"][3] - a["box"][1]))           # far (small) first
    for a in items:
        x1, y1, x2, y2 = a["box"]
        cls = labels["agent_labels"][a["agent_ids"][0]]
        col = COLORS.get(cls, "black")
        mark = highlight is not None and a.get("tube_uid") == highlight
        cam_ax.add_patch(Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H, fill=False,
                                   ec="red" if mark else col, lw=4.0 if mark else 1.6))
        text = cls + " " + ",".join(_short(labels, a["action_ids"], "action_labels"))
        locs = _short(labels, a["loc_ids"], "loc_labels")
        if locs:
            text += " | " + ",".join(locs)
        cam_ax.text(x1 * W, y1 * H - 6, text, fontsize=7, color="white",
                    bbox={"fc": col, "ec": "none", "pad": 1, "alpha": 0.85})
    cam_ax.set_xlim(0, W); cam_ax.set_ylim(H, 0)
    cam_ax.set_aspect("equal")
    cam_ax.set_xticks([]); cam_ax.set_yticks([])
    # The executed rules of the ego / visible agents firing within +-1 s, the ego's first, at most 4.
    near = sorted((f for f in firings if f.get("frame_fire") and abs(f["frame_fire"] - frame) <= 10
                   and f.get("executed") and (f["agent_is_ego"] or f["agent_visible"])),
                  key=lambda f: (not f["agent_is_ego"], abs(f["frame_fire"] - frame)))[:4]
    note = "; ".join(f"{f['rule']} {f['agent']} fires f{f['frame_fire']} (body {f['body_agent']} f{f['frame_body']})"
                     for f in near)
    cam_ax.set_title(f"camera view: {len(annos)} GT boxes\n" + (textwrap.fill(note, 110) if note else ""), fontsize=9,
                     loc="left")
    # A fixed layout (not tight_layout): every frame of a video has the same geometry.
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.86, wspace=0.04)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=110)
    plt.close(fig)


def interesting(dataset, num):
    """[(vid, frame)]: frames a few frames after an executed R1 / R2 the ego sees, turns, red stops."""
    labels = json.loads((dataset / "gt" / "label.json").read_text())
    picks = {"R1": [], "R2": [], "R3": [], "turn": [], "red": []}
    for gt in sorted((dataset / "gt").glob("*_gt.json")):
        vid = gt.name[:-len("_gt.json")]
        for f in json.loads((dataset / "meta" / "firings" / f"{vid}.json").read_text()):
            if f.get("executed") and f["body_visible"] and f["agent_visible"] and f["frame_fire"] <= 195:
                picks.setdefault(f["rule"], []).append((vid, f["frame_fire"] + 3))
        frames = json.loads(gt.read_text())["data"]["frames"]
        av = [labels["av_action_labels"][frames[str(k)]["av_action_ids"][0]] for k in range(1, len(frames) + 1)]
        turns = [k + 1 for k, a in enumerate(av) if a in ("AV-TurLft", "AV-TurRht")]
        if turns:
            picks["turn"].append((vid, turns[len(turns) // 2]))
        tl = labels["agent_labels"].index("TL")
        red = labels["action_labels"].index("Red")
        for k, a in enumerate(av):
            if a == "AV-Stop" and any(x["agent_ids"] == [tl] and red in x["action_ids"]
                                      for x in frames[str(k + 1)]["annos"].values()):
                picks["red"].append((vid, k + 1))
                break
    out = []
    while len(out) < num and any(picks.values()):
        for key in picks:
            if picks[key] and len(out) < num:
                out.append(picks[key].pop(0))
    return out


def render_video(dataset, vid, out_file, fps=10):
    """Every frame of video vid (`render`) as an mp4 (H.264, ffmpeg)."""
    num_frames = len(_video(dataset, vid)[1]["frames"])
    with tempfile.TemporaryDirectory() as tmp:
        for frame in range(1, num_frames + 1):
            render(dataset, vid, frame, Path(tmp) / f"{frame:05d}.png")
            if frame % 25 == 0 or frame == num_frames:
                print(f"[Render] {vid}: {frame}/{num_frames} frames")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(Path(tmp) / "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_file)], check=True)
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Schematic images of SUMO synthetic videos")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--vid", default=None)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--maps", action="store_true", help="also one image of all junctions (multi-map datasets)")
    parser.add_argument("--video", action="store_true", help="every frame of one video as an mp4 (--vid)")
    parser.add_argument("--fps", type=int, default=10)
    args, _unknown = parser.parse_known_args()             # the docker script adds --device / --machine
    dataset = Path(args.dataset)
    if args.video:
        vid = args.vid or interesting(dataset, 1)[0][0]
        print(render_video(dataset, vid, dataset / "render" / f"{vid}.mp4", args.fps))
        return
    if args.maps and (dataset / "meta" / "maps.json").exists():
        out = dataset / "render" / "maps.png"
        render_maps(dataset, out)
        print(out)
    jobs = [(args.vid, f) for f in args.frames] if args.vid and args.frames else interesting(dataset, args.num)
    for vid, frame in jobs:
        out = dataset / "render" / f"{vid}_f{frame:03d}.png"
        render(dataset, vid, frame, out)
        print(out)


if __name__ == "__main__":
    main()
