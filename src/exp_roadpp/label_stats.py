"""Are the labels of two datasets alike? A check before transferring rules (SUMO -> roadpp).

For every label "<class>:<label>" (the ego's "av:AV-*"), from the step_dataset atoms of each exp:
    agents       agents of the class that ever hold it, and their share of the class's agents
    frame_share  frames holding it / frames of the class's agents (each agent from its first to its last frame)
    onsets_per_1k  onsets per 1000 agent frames of the class (as step_dataset counts them: not on the
                 agent's first frame, runs joined over gaps < MIN_OFF_FRAMES)
    run_median   median run length in frames (p10, p90)
and per agent class the agents per video, the track length and, from the box files, the quantiles of the
box centre x, bottom y2 and height (normalized image coordinates: where the grid atoms come from).
A shared label is flagged when its onset rate differs more than RATE_FLAG x or its median run length
more than RUN_FLAG x: rules on it are not expected to transfer.

    ./run_roadpp_docker.sh --gpu none --label_stats -- --exps sumo_maps_v7 full_sn
    python -m src.exp_roadpp.label_stats --exps sumo_maps_v5 full_sn --machine local --splits test
Writes <output root>/roadpp/compare/<a>_vs_<b>_labels.json."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

MIN_OFF_FRAMES = 10         # as step_dataset.MIN_OFF_FRAMES
RATE_FLAG, RUN_FLAG = 3.0, 2.0
QUANTILES = (0.1, 0.5, 0.9)


def _runs(spans):
    out = []
    for s, e in sorted(spans):
        if out and s - out[-1][1] < MIN_OFF_FRAMES:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _q(values):
    if not values:
        return None
    v = sorted(values)
    return [round(v[min(len(v) - 1, int(q * len(v)))], 3) for q in QUANTILES]


def atom_stats(videos):
    """{"labels": {label: stats}, "classes": {class: stats}} over the videos' interval atoms."""
    agents_of = defaultdict(set)                    # class -> agents
    agent_frames = defaultdict(int)                 # class -> frames of its agents
    track_len = defaultdict(list)
    holders = defaultdict(set)
    label_frames, onsets, run_len = defaultdict(int), defaultdict(int), defaultdict(list)
    num_videos = 0
    for vid, interval_atoms in videos:
        num_videos += 1
        spans = defaultdict(lambda: defaultdict(list))
        cls_of, first, last = {}, {}, {}
        for a in interval_atoms:
            agent = (vid, a.get("tube_uid") or "ego")
            cls = str(a.get("agent_class_label"))
            label = a.get("action_id_label") or a.get("location_name_label")
            s, e = int(a["start_frame"]), int(a["end_frame"])
            cls_of[agent] = cls
            first[agent], last[agent] = min(first.get(agent, s), s), max(last.get(agent, e), e)
            spans[agent][f"{cls}:{label}"].append((s, e))
        for agent, per in spans.items():
            cls = cls_of[agent]
            agents_of[cls].add(agent)
            n = last[agent] - first[agent] + 1
            agent_frames[cls] += n
            track_len[cls].append(n)
            for label, sp in per.items():
                holders[label].add(agent)
                for s, e in _runs(sp):
                    label_frames[label] += e - s + 1
                    run_len[label].append(e - s + 1)
                    onsets[label] += s > first[agent]
    labels = {}
    for label in holders:
        cls = label.split(":", 1)[0]
        frames = max(agent_frames[cls], 1)
        labels[label] = {"agents": len(holders[label]), "agent_share": round(len(holders[label]) / max(len(agents_of[cls]), 1), 3),
                         "frame_share": round(label_frames[label] / frames, 4),
                         "onsets": onsets[label], "onsets_per_1k": round(1000 * onsets[label] / frames, 3),
                         "run_median": statistics.median(run_len[label]), "run_q": _q(run_len[label])}
    classes = {c: {"agents": len(a), "agents_per_video": round(len(a) / max(num_videos, 1), 2),
                   "track_len_q": _q(track_len[c])} for c, a in agents_of.items()}
    return {"videos": num_videos, "labels": labels, "classes": classes}


def box_stats(box_files):
    """{class: {"x_centre", "y2", "height": quantiles}} of every box in the box files."""
    import torch
    vals = defaultdict(lambda: defaultdict(list))
    for f in box_files:
        if not f.exists():
            continue
        for tubes in torch.load(f, map_location="cpu").values():
            for tube in (tubes or {}).values():
                b = torch.as_tensor(tube["boxes"], dtype=torch.float).view(-1, 4)
                if not len(b):
                    continue
                step = max(1, len(b) // 20)                    # up to ~20 boxes per tube
                b = b[::step]
                cls = str(tube["label"])
                vals[cls]["x_centre"] += ((b[:, 0] + b[:, 2]) / 2).tolist()
                vals[cls]["y2"] += b[:, 3].tolist()
                vals[cls]["height"] += (b[:, 3] - b[:, 1]).tolist()
    return {c: {k: _q(v) for k, v in d.items()} for c, d in vals.items()}


def compare(a, b):
    """Shared labels with both datasets' stats, ratios (a / b) and flags; and the labels of one only."""
    shared, only_a, only_b = {}, [], []
    for label in sorted(set(a["labels"]) | set(b["labels"])):
        x, y = a["labels"].get(label), b["labels"].get(label)
        if x is None or y is None:
            (only_b if x is None else only_a).append((label, (x or y)["onsets"]))
            continue
        rate = x["onsets_per_1k"] / y["onsets_per_1k"] if y["onsets_per_1k"] else None
        run = x["run_median"] / y["run_median"] if y["run_median"] else None
        flags = []
        if rate is not None and not (1 / RATE_FLAG <= rate <= RATE_FLAG):
            flags.append("rate")
        if run is not None and not (1 / RUN_FLAG <= run <= RUN_FLAG):
            flags.append("duration")
        shared[label] = {"a": x, "b": y, "rate_ratio": None if rate is None else round(rate, 2),
                         "run_ratio": None if run is None else round(run, 2), "flags": flags}
    return {"shared": shared, "only_a": sorted(only_a, key=lambda t: -t[1]), "only_b": sorted(only_b, key=lambda t: -t[1])}


def print_report(names, result, min_onsets=20):
    a, b = names
    print(f"[Label stats] {a} vs {b}: onsets per 1000 agent frames, median run (frames), ratio {a}/{b}; "
          f"flag: rate off by > {RATE_FLAG}x, run by > {RUN_FLAG}x; labels with >= {min_onsets} onsets in both")
    print(f"  {'label':24s} {'rate ' + a[:10]:>16s} {'rate ' + b[:10]:>16s} {'ratio':>6s} {'run':>11s} {'ratio':>6s}  flags")
    rows = [(l, r) for l, r in compare_rows(result) if r["a"]["onsets"] >= min_onsets and r["b"]["onsets"] >= min_onsets]
    for label, r in rows:
        print(f"  {label:24s} {r['a']['onsets_per_1k']:16.2f} {r['b']['onsets_per_1k']:16.2f} {r['rate_ratio'] or 0:6.2f} "
              f"{r['a']['run_median']:5.0f}/{r['b']['run_median']:<5.0f} {r['run_ratio'] or 0:6.2f}  {','.join(r['flags'])}")
    flagged = sum(bool(r["flags"]) for _l, r in rows)
    print(f"  {flagged} of {len(rows)} shared labels flagged; only in {a}: {len(result['labels']['only_a'])}, "
          f"only in {b}: {len(result['labels']['only_b'])} (top: {', '.join(l for l, _n in result['labels']['only_b'][:10])})")
    print(f"  classes (agents per video, track length p50; box y2 p10/p50/p90, height p50):")
    for cls in sorted(set(result["classes"]["a"]) & set(result["classes"]["b"])):
        ca, cb = result["classes"]["a"][cls], result["classes"]["b"][cls]
        ba, bb = result["boxes"]["a"].get(cls, {}), result["boxes"]["b"].get(cls, {})
        def box(bx):
            return (f"y2 {bx['y2']} h {bx['height'][1]}" if bx.get("y2") else "-")
        print(f"   {cls:8s} {a[:10]}: {ca['agents_per_video']:6.2f}/video, {ca['track_len_q'][1]:4.0f} f, {box(ba)}  |  "
              f"{b[:10]}: {cb['agents_per_video']:6.2f}/video, {cb['track_len_q'][1]:4.0f} f, {box(bb)}")


def compare_rows(result):
    return sorted(result["labels"]["shared"].items(), key=lambda kv: (kv[0].split(":")[0] != "av", kv[0]))


def load_videos(atoms_dir, splits):
    for sp in splits:
        for vid, v in json.loads((atoms_dir / f"{sp}_atoms.json").read_text()).items():
            yield vid, v["interval_atoms"]


def main():
    parser = argparse.ArgumentParser(description="Compare the labels of two exps (step_dataset atoms)")
    parser.add_argument("--exps", nargs=2, required=True, help="e.g. sumo_maps_v7 full_sn")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], help="atom files (the train files are large)")
    parser.add_argument("--machine", default="dgx-jing", help="dgx-jing: the server output; else the local output")
    args, _unknown = parser.parse_known_args()             # the docker script adds --device
    root = (Path("/storage-01/ml-jsha/CauVid_output/roadpp") if args.machine == "dgx-jing"
            else Path(__file__).resolve().parent / "output" / "roadpp")
    stats, boxes = {}, {}
    for key, exp in zip("ab", args.exps):
        atoms_dir = root / exp / "step_dataset_output" / "atoms"
        stats[key] = atom_stats(load_videos(atoms_dir, args.splits))
        boxes[key] = box_stats([atoms_dir / f"{sp}_boxes.pt" for sp in args.splits])
        print(f"[Label stats] {exp}: {stats[key]['videos']} videos ({', '.join(args.splits)})")
    result = {"exps": args.exps, "splits": args.splits, "rate_flag": RATE_FLAG, "run_flag": RUN_FLAG,
              "videos": {k: s["videos"] for k, s in stats.items()},
              "labels": compare(stats["a"], stats["b"]),
              "classes": {k: s["classes"] for k, s in stats.items()}, "boxes": boxes}
    print_report(args.exps, result)
    out = root / "compare" / f"{args.exps[0]}_vs_{args.exps[1]}_labels.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"[Label stats] -> {out}")


if __name__ == "__main__":
    main()
