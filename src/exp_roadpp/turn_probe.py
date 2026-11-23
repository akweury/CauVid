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

    ./run_roadpp_docker.sh --gpu none --turn_probe -- --dataset /storage-01/ml-jsha/CauVid_Data/synthetic/sumo_maps_v7
Writes <dataset>/meta/probe/turn_probe_<head>_<split>.json.
"""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from src.exp_roadpp.synth.sumo import labels, observe

LOOKBACK = 30                # frames (3 s at 10 fps) for the cues before the onset
LC_LOOKBACK = 100            # frames (10 s) for a lane change to the turn's side
YAW_LEVELS = (2.0, 4.0)      # deg/s: the precursor levels (the label needs labels.TURN_RATE = 8)
LEAD_BINS = (1, 3, 5, 10, 20, 30)


def _videos(dataset, split):
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


def main():
    parser = argparse.ArgumentParser(description="What happens before the ego's turn onsets (SUMO world states)")
    parser.add_argument("--dataset", required=True, help="the SUMO dataset dir (with meta/world)")
    parser.add_argument("--head", default="AV-TurLft", choices=("AV-TurLft", "AV-TurRht"))
    parser.add_argument("--split", default="val", choices=("train", "val", "test", "all"))
    parser.add_argument("--min_off", type=int, default=5, help="frames without the label before an onset")
    parser.add_argument("--max_videos", type=int, default=0, help="0: all")
    args, _unknown = parser.parse_known_args()                  # the docker script adds --device / --machine
    dataset = Path(args.dataset)
    vids = _videos(dataset, args.split)
    if args.max_videos:
        vids = vids[:args.max_videos]
    print(f"[turn_probe] {args.head}, {len(vids)} {args.split} videos of {dataset}")
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
    out = dataset / "meta" / "probe" / f"turn_probe_{args.head}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"head": args.head, "split": args.split, "videos": len(vids), "paths": dict(paths),
                               "events": events}, indent=1))
    print(f"\n[turn_probe] -> {out}")


if __name__ == "__main__":
    main()
