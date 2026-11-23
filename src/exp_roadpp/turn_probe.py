"""Turn probe: what happens before the ego's turn onsets (AV-TurLft / AV-TurRht), read from the SUMO world states
(2026-11-23, the user: before a lateral move starts there should be a short time to read the key facts; does the
ego already move a little, or does the turn label start the moment the move starts?).

The rule model lags the OnsetMLP most on av:AV-TurLft (rule lab, 11 runs: 0.248 vs 0.399). The label
(`labels.av_label`) is AV-TurLft in a frame when:
  - the ego is on a junction-internal edge (road ":..."),
  - its yaw rate, the mean over +-YAW_SMOOTH frames, is above TURN_RATE (8 deg/s) to the left,
  - and it is neither AV-Stop nor AV-Brake (they come first).
The ego labels are recomputed here from the world states with the same code (`labels.agent_streams`, `av_label`),
so they are the ground truth's.

Per onset (the head in frame k, not in the min_off frames before), reported:
  1. path, by the label in k - 1 and the ego's road:
       after_brake / after_stop  the ego braked or stood and was already turning (the turn hidden by Brake/Stop);
       entered_junction          AV-Mov* in k - 1 off the junction (turning before the junction edge);
       yaw_rise                  AV-Mov* in k - 1 on the junction (the yaw rate crossed TURN_RATE);
  2. precursor: the frames before k in which the yaw rate toward the turn already exceeds 2 / 4 deg/s (a run
     ending in k - 1), and the yaw rate at k - 1, k - 3, k - 5, k - 10 (the yaw is a centred mean, so it reads up
     to YAW_SMOOTH frames ahead, as the label does);
  3. cues of the LOOKBACK frames before: the ego braked or stood; a lane change to the turn's side (AV-MovLft for a
     left turn) within LC_LOOKBACK frames; the speed drop; the frames on the junction; a traffic light in view in
     k - 1 (its box visible, `observe.visible`) and its colour.

Ground-truth mode (datasets without meta/world, e.g. sumo_maps_v7; automatic): the yaw is not in the labels, so
the probe reads what the model can see, the boxes of gt/{vid}_gt.json:
  - path by the ego label in k - 1 only (after_brake / after_stop / from_mov / from_lane_change);
  - scene drift: per frame the median horizontal move of the visible boxes (centres, over K_DRIFT frames, per
    frame, image widths; boxes cut by the image border left out), signed toward the turn (a left turn moves the
    scene right). Reported at k - 1 .. k - 20, the frames before k above the drift's q90 of the AV-Mov frames
    far (> FAR frames) from any turn (the background), and that background;
  - the cues: braked / stood, lane change to the turn's side, the traffic light's box in k - 1 and its colour.

    ./run_roadpp_docker.sh --gpu 0 --turn_probe -- --exp sumo_maps_v7 --machine dgx-jing
The dataset is the exp's (`config.machine_setup`; --sumo_dir overrides it). Writes
<output_dir>/turn_probe/turn_probe_<head>_<split>[_gt].json (the folder is created).
"""
import gzip
import json
from collections import Counter
from pathlib import Path

from src.exp_roadpp import config, utils_pipe
from src.exp_roadpp.synth.sumo import labels, observe

LOOKBACK = 30                # frames (3 s at 10 fps) for the cues before the onset
LC_LOOKBACK = 100            # frames (10 s) for a lane change to the turn's side
YAW_LEVELS = (2.0, 4.0)      # deg/s: the precursor levels (the label needs labels.TURN_RATE = 8)
LEAD_BINS = (1, 3, 5, 10, 20, 30)
K_DRIFT = 2                  # frames of the box move (as the shape atoms)
EDGE = 0.005                 # a box this close to the image border is cut
FAR = 30                     # frames from any turn label: the background frames of the drift
DRIFT_AT = (1, 3, 5, 10, 20)


def _videos(dataset, split, gt_mode=False):
    if gt_mode:
        world = sorted(p.name[:-len("_gt.json")] for p in (dataset / "gt").glob("*_gt.json"))
    else:
        world = sorted(p.name[:-len(".json.gz")] for p in (dataset / "meta" / "world").glob("*.json.gz"))
    if split == "all":
        return world
    f = dataset / "meta" / "video_split.json"
    if not f.exists():
        print(f"[turn_probe] no {f}: all {len(world)} videos")
        return world
    keep = set(json.loads(f.read_text())["splits"][split])
    return [v for v in world if v in keep]


def _run_back(flags, k):
    """Frames of the run of True ending in k - 1 (0 if flags[k - 1] is False)."""
    n, j = 0, k - 1
    while j >= 0 and flags[j]:
        n, j = n + 1, j - 1
    return n


def probe_video(frames, head, min_off):
    T = next((k for k, w in enumerate(frames) if "ego" not in w.get("veh", {})), len(frames))
    frames = frames[:T]                                             # the ego's frames (it may leave at the end)
    streams = labels.agent_streams(frames)
    ego_s = streams["ego"]
    T = len(frames)
    egos = [w["veh"]["ego"] for w in frames]
    lab = [labels.av_label(egos[k], ego_s[k]) for k in range(T)]
    side = -1.0 if head.endswith("Lft") else 1.0                   # SUMO: yaw > 0 turns right
    toward = [side * ego_s[k]["yaw"] for k in range(T)]
    lc_side = "AV-Mov" + head[-3:]
    out = []
    for k in range(1, T):
        if lab[k] != head or any(lab[j] == head for j in range(max(0, k - min_off), k)):
            continue
        prev = lab[k - 1]
        on_junction = egos[k - 1]["road"].startswith(":")
        if prev == "AV-Brake":
            path = "after_brake"
        elif prev == "AV-Stop":
            path = "after_stop"
        elif on_junction:
            path = "yaw_rise"
        else:
            path = "entered_junction"
        lo = max(0, k - LOOKBACK)
        speeds = [egos[j]["speed"] for j in range(lo, k + 1)]
        w = frames[k - 1]
        cam = observe.Camera(egos[k - 1])
        agents = labels._scene_agents(w, cam)
        tl_view = labels.TL_ID in observe.visible(cam, agents)
        tl = w.get("tl")
        rec = {
            "onset": k, "path": path, "prev": prev,
            "lead": {f"{y:g}": _run_back([t > y for t in toward], k) for y in YAW_LEVELS},
            "yaw_before": {str(d): (round(toward[k - d], 2) if k - d >= 0 else None) for d in (1, 3, 5, 10)},
            "yaw_at": round(toward[k], 2),
            "speed": round(egos[k]["speed"], 2), "speed_max_before": round(max(speeds), 2),
            "braked": any(lab[j] in ("AV-Brake", "AV-Stop") for j in range(lo, k)),
            "lane_change": any(lab[j] == lc_side for j in range(max(0, k - LC_LOOKBACK), k)),
            "on_junction_frames": _run_back([e["road"].startswith(":") for e in egos], k),
            "tl_in_view": tl_view,
            "tl_colour": labels.tl_label(tl["state"]) if tl else None,
            "prev_run": _run_back([x == prev for x in lab], k),
        }
        out.append(rec)
    return out


def _q(xs):
    if not xs:
        return "-"
    s = sorted(xs)
    return f"10% {s[len(s) // 10]}, 25% {s[len(s) // 4]}, median {s[len(s) // 2]}, 75% {s[3 * len(s) // 4]}, 90% {s[9 * len(s) // 10]}"


def report(events, title):
    n = max(len(events), 1)
    print(f"\n  === {title}: {len(events)} onsets ===")
    if not events:
        return
    for y in YAW_LEVELS:
        lead = [e["lead"][f"{y:g}"] for e in events]
        bins = ", ".join(f">={b} {sum(1 for x in lead if x >= b) / n:.0%}" for b in LEAD_BINS)
        print(f"    frames before the onset with yaw > {y:g} deg/s toward the turn: {_q(lead)}  ({bins})")
    for d in ("1", "3", "5", "10"):
        v = [e["yaw_before"][d] for e in events if e["yaw_before"][d] is not None]
        print(f"    yaw rate toward the turn at k-{d:>2s} (deg/s): {_q([round(x, 1) for x in v])}")
    print(f"    speed at the onset (m/s): {_q([e['speed'] for e in events])}; "
          f"max of the {LOOKBACK} frames before: {_q([e['speed_max_before'] for e in events])}")
    print(f"    on the junction before the onset (frames): {_q([e['on_junction_frames'] for e in events])}")
    print(f"    braked or stood in the {LOOKBACK} frames before: {sum(e['braked'] for e in events) / n:.0%}; "
          f"lane change to the turn's side in the {LC_LOOKBACK} frames before: "
          f"{sum(e['lane_change'] for e in events) / n:.0%}")
    print(f"    traffic light in view in k-1: {sum(e['tl_in_view'] for e in events) / n:.0%}; colour "
          f"{dict(Counter(e['tl_colour'] for e in events).most_common())}")
    print(f"    the previous label's run (frames): {_q([e['prev_run'] for e in events])}")


def gt_video(video, label_json):
    """(ego labels per frame, tubes {uid: {frame: box}}, TL {frame: colour}) of one ground-truth video."""
    av_names, agent_names = label_json["av_action_labels"], label_json["agent_labels"]
    action_names = label_json["action_labels"]
    data = video["data"]
    T = int(data["numf"])
    lab = [None] * T
    tubes, tl = {}, {}
    for fk, fr in data["frames"].items():
        k = int(fk) - 1
        if not 0 <= k < T:
            continue
        ids = fr.get("av_action_ids") or []
        lab[k] = av_names[ids[0]] if ids else None
        for a in fr.get("annos", {}).values():
            cls = agent_names[a["agent_ids"][0]] if a.get("agent_ids") else None
            if cls == "TL":
                acts = [action_names[i] for i in a.get("action_ids", [])]
                tl[k] = acts[0] if acts else "?"
                continue
            tubes.setdefault(a["tube_uid"], {})[k] = a["box"]
    return lab, tubes, tl, T


def scene_drift(tubes, T):
    """(T,) list: the median horizontal move per frame (image widths) of the visible uncut boxes over K_DRIFT
    frames; None without boxes."""
    moves = [[] for _ in range(T)]
    for boxes in tubes.values():
        for k, b in boxes.items():
            a = boxes.get(k - K_DRIFT)
            if a is None or min(a[0], b[0]) < EDGE or max(a[2], b[2]) > 1 - EDGE:
                continue
            moves[k].append(((b[0] + b[2]) - (a[0] + a[2])) / 2 / K_DRIFT)
    out = []
    for m in moves:
        m = sorted(m)
        out.append(m[len(m) // 2] if m else None)
    return out


def probe_gt(video, label_json, head, min_off):
    lab, tubes, tl, T = gt_video(video, label_json)
    side = 1.0 if head.endswith("Lft") else -1.0                   # a left turn moves the scene right
    drift = [None if d is None else side * d for d in scene_drift(tubes, T)]
    turn = [x in ("AV-TurLft", "AV-TurRht") for x in lab]
    lc_side = "AV-Mov" + head[-3:]
    background = [drift[k] for k in range(T) if drift[k] is not None and lab[k] == "AV-Mov"
                  and not any(turn[max(0, k - FAR):k + FAR + 1])]
    out = []
    for k in range(1, T):
        if lab[k] != head or any(lab[j] == head for j in range(max(0, k - min_off), k)):
            continue
        prev = lab[k - 1]
        path = {"AV-Brake": "after_brake", "AV-Stop": "after_stop"}.get(prev) or (
            "from_lane_change" if prev and prev.startswith("AV-Mov") and prev != "AV-Mov" else "from_mov")
        lo = max(0, k - LOOKBACK)
        out.append({
            "onset": k, "path": path, "prev": prev,
            "drift_before": {str(d): (round(drift[k - d] * 1000, 3) if k - d >= 0 and drift[k - d] is not None
                                      else None) for d in DRIFT_AT},
            "drift_series": [None if k - d < 0 or drift[k - d] is None else drift[k - d] for d in range(1, LOOKBACK + 1)],
            "braked": any(lab[j] in ("AV-Brake", "AV-Stop") for j in range(lo, k)),
            "lane_change": any(lab[j] == lc_side for j in range(max(0, k - LC_LOOKBACK), k)),
            "tl_in_view": (k - 1) in tl, "tl_colour": tl.get(k - 1),
            "prev_run": _run_back([x == prev for x in lab], k),
        })
    return out, background


def report_gt(events, title, thr):
    n = max(len(events), 1)
    print(f"\n  === {title}: {len(events)} onsets ===")
    if not events:
        return
    for d in DRIFT_AT:
        v = [e["drift_before"][str(d)] for e in events if e["drift_before"][str(d)] is not None]
        above = sum(1 for x in v if x / 1000 > thr) / max(len(v), 1)
        print(f"    scene drift toward the turn at k-{d:<2d} (1e-3 image widths / frame): {_q([round(x, 2) for x in v])}; "
              f"above the background q90: {above:.0%}")
    lead = []
    for e in events:
        m = 0
        for x in e["drift_series"]:                                  # k-1, k-2, ..
            if x is None or x <= thr:
                break
            m += 1
        lead.append(m)
    bins = ", ".join(f">={b} {sum(1 for x in lead if x >= b) / n:.0%}" for b in LEAD_BINS)
    print(f"    frames before the onset with the drift above the background q90: {_q(lead)}  ({bins})")
    print(f"    braked or stood in the {LOOKBACK} frames before: {sum(e['braked'] for e in events) / n:.0%}; "
          f"lane change to the turn's side in the {LC_LOOKBACK} frames before: "
          f"{sum(e['lane_change'] for e in events) / n:.0%}")
    print(f"    traffic light in view in k-1: {sum(e['tl_in_view'] for e in events) / n:.0%}; colour "
          f"{dict(Counter(e['tl_colour'] for e in events).most_common())}")
    print(f"    the previous label's run (frames): {_q([e['prev_run'] for e in events])}")


def main_gt(args, dataset):
    label_json = json.loads((dataset / "gt" / "label.json").read_text())
    vids = _videos(dataset, args.probe_split, gt_mode=True)
    if args.max_videos:
        vids = vids[:args.max_videos]
    print(f"[turn_probe] ground-truth mode (no meta/world): {args.head}, {len(vids)} {args.probe_split} videos of {dataset}")
    events, background = [], []
    for i, vid in enumerate(vids):
        video = json.loads((dataset / "gt" / f"{vid}_gt.json").read_text())
        ev, bg = probe_gt(video, label_json, args.head, args.min_off)
        events += [{"video": vid, **e} for e in ev]
        background += bg
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} / {len(vids)} videos, {len(events)} onsets")
    bg = sorted(background)
    thr = bg[9 * len(bg) // 10] if bg else 0.0
    if bg:
        print(f"\n[turn_probe] background drift toward the turn (AV-Mov frames > {FAR} frames from a turn, "
              f"{len(bg)} frames, 1e-3 image widths / frame): {_q([round(x * 1000, 2) for x in bg])}; q90 {thr * 1000:.2f}")
    paths = Counter(e["path"] for e in events)
    n = max(len(events), 1)
    print(f"[turn_probe] {len(events)} onsets of {args.head}; path: "
          + ", ".join(f"{k} {v / n:.0%}" for k, v in paths.most_common()))
    report_gt(events, "all", thr)
    for p, _ in paths.most_common():
        report_gt([e for e in events if e["path"] == p], p, thr)
    out = args.probe_dir / f"turn_probe_{args.head}_{args.probe_split}_gt.json"
    for e in events:
        e.pop("drift_series", None)
    out.write_text(json.dumps({"head": args.head, "split": args.probe_split, "mode": "gt", "videos": len(vids),
                               "background_q90": thr, "paths": dict(paths), "events": events}, indent=1))
    print(f"\n[turn_probe] -> {out}")


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--sumo_dir", default="", help="the SUMO dataset dir (default: the exp's)")
    parser.add_argument("--head", default="AV-TurLft", choices=("AV-TurLft", "AV-TurRht"))
    parser.add_argument("--probe_split", default="val", choices=("train", "val", "test", "all"))
    parser.add_argument("--min_off", type=int, default=5, help="frames without the label before an onset")
    parser.add_argument("--max_videos", type=int, default=0, help="0: all")
    args = config.machine_setup(utils_pipe.parse_args(parser))
    dataset = Path(args.sumo_dir) if args.sumo_dir else Path(args.dataset_path)
    args.probe_dir = Path(args.output_dir) / "turn_probe"
    args.probe_dir.mkdir(parents=True, exist_ok=True)
    print(f"[turn_probe] dataset {dataset}; results -> {args.probe_dir}")
    if not (dataset / "gt").exists():
        raise FileNotFoundError(f"no {dataset / 'gt'}: pass the SUMO dataset dir with --sumo_dir")
    if not (dataset / "meta" / "world").exists():
        return main_gt(args, dataset)
    vids = _videos(dataset, args.probe_split)
    if args.max_videos:
        vids = vids[:args.max_videos]
    print(f"[turn_probe] {args.head}, {len(vids)} {args.probe_split} videos of {dataset}")
    events = []
    for i, vid in enumerate(vids):
        with gzip.open(dataset / "meta" / "world" / f"{vid}.json.gz", "rt") as f:
            frames = json.load(f)
        for e in probe_video(frames, args.head, args.min_off):
            events.append({"video": vid, **e})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} / {len(vids)} videos, {len(events)} onsets")
    paths = Counter(e["path"] for e in events)
    n = max(len(events), 1)
    print(f"\n[turn_probe] {len(events)} onsets of {args.head}; path: "
          + ", ".join(f"{k} {v / n:.0%}" for k, v in paths.most_common()))
    report(events, "all")
    for p, _ in paths.most_common():
        report([e for e in events if e["path"] == p], p)
    out = args.probe_dir / f"turn_probe_{args.head}_{args.probe_split}.json"
    out.write_text(json.dumps({"head": args.head, "split": args.probe_split, "videos": len(vids), "paths": dict(paths),
                               "events": events}, indent=1))
    print(f"\n[turn_probe] -> {out}")


if __name__ == "__main__":
    main()
