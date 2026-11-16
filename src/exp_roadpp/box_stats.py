"""Box size statistics: how many vehicle frames, onsets and test events a minimum box height would drop.

Before far (small) vehicles are dropped from the dataset (planned: an agent of VEHICLES counts only in
the frames where its GT box is at least min_height of the frame height; pedestrians, cyclists,
motorbikes and traffic lights stay), this reads the existing step_dataset output of an exp and prints:
  1. per agent class: the box height (share of the frame height) over all frames and at the onsets;
  2. per threshold of THRESHOLDS: the share of vehicle frames, vehicle tubes (entirely) and vehicle
     onsets dropped, per split, and the onsets left of the main vehicle heads (train / val / test);
  3. with --run_id <run> (a run with step 07's paper_events.csv): the test events of the vehicle heads
     by box height bin and the recall of every model (5 false alarms / min): do the models fail on far cars?
No training, no model, no GPU. Output: printed and <exp>/box_stats/box_stats.md (+ .json).

Run (server): ./run_roadpp_docker.sh --exp full_s --box_stats --run_id 20261115-2131_main
"""
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch

from src.exp_roadpp import config, step05, utils_data, utils_pipe

VEHICLES = ("Car", "MedVeh", "LarVeh", "Bus", "EmVeh", "SmalVeh")   # 4-wheel vehicles: filtered by size
THRESHOLDS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]             # box height / frame height
BINS = [0.0, 0.03, 0.05, 0.08, 0.15, 1.01]
MIN_OFF_FRAMES = 10
SPLITS = ("train", "val", "test")
SCORERS = ("model_cal", "mlp_cal", "lookup_cal")
BUDGET = 5


def _height_at(tube, frame, near=2):
    """Box height of a tube at frame (the nearest box within `near` frames), None without one."""
    if tube is None:
        return None
    d = (tube["frames"] - frame).abs()
    i = int(torch.argmin(d))
    if int(d[i]) > near:
        return None
    b = tube["boxes"][i]
    return float(b[3] - b[1])


def _onsets(interval_atoms):
    """[(tube_uid, class, head, onset frame)] of one video, the rule of `step_dataset.head_onsets`."""
    first, by_atom, cls = {}, defaultdict(list), {}
    for atom in interval_atoms:
        tube = atom.get("tube_uid") or "ego"
        start, end = int(atom["start_frame"]), int(atom["end_frame"])
        first[tube] = min(first.get(tube, start), start)
        label = atom.get("action_id_label") or atom.get("location_name_label")
        c = str(atom.get("agent_class_label"))
        cls[tube] = c
        by_atom[(tube, f"{c}:{label}")].append((start, end))
    out = []
    for (tube, head), spans in by_atom.items():
        last_end = None
        for start, end in sorted(spans):
            if start > first[tube] and (last_end is None or start - last_end >= MIN_OFF_FRAMES):
                out.append((tube, cls[tube], head, start))
            last_end = end if last_end is None else max(last_end, end)
    return out


def _pct(values, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if not values:
        return ["–"] * len(qs)
    v = np.percentile(np.asarray(values, dtype=np.float32), [100 * q for q in qs])
    return [f"{float(x):.3f}" for x in v]


def main():
    args = config.machine_setup(utils_pipe.parse_args())
    atoms_dir = args.output_dir / "step_dataset_output" / "atoms"
    out_dir = args.output_dir / "box_stats"
    os.makedirs(out_dir, exist_ok=True)
    md, js = [], {"vehicles": VEHICLES, "thresholds": THRESHOLDS}

    def emit(s=""):
        print(s)
        md.append(s)

    frame_h = defaultdict(list)                 # class -> heights over all frames
    onset_h = defaultdict(list)                 # class -> heights at onsets
    tube_max = defaultdict(list)                # class -> max height per tube
    veh_onsets = {sp: [] for sp in SPLITS}      # split -> [(head, height)]
    for sp in SPLITS:
        boxes = torch.load(atoms_dir / f"{sp}_boxes.pt", map_location="cpu")
        atoms = utils_data.load_json(atoms_dir / f"{sp}_atoms.json")
        for vid, tubes in boxes.items():
            for uid, tube in tubes.items():
                h = (tube["boxes"][:, 3] - tube["boxes"][:, 1]).tolist()
                frame_h[tube["label"]] += h
                tube_max[tube["label"]].append(max(h))
            for tube, c, head, t in _onsets((atoms.get(vid) or {}).get("interval_atoms", [])):
                if tube == "ego":
                    continue
                h = _height_at(tubes.get(tube), t)
                if h is None:
                    continue
                onset_h[c].append(h)
                if c in VEHICLES:
                    veh_onsets[sp].append((head, h))
        del boxes, atoms

    emit(f"# Box size statistics ({args.exp})\n")
    emit("Height = box height / frame height. Vehicles (filtered): " + ", ".join(VEHICLES) + "\n")
    emit("## 1. Box height per class (quantiles 10 / 25 / 50 / 75 / 90 %)\n")
    emit("| Class | Frames | Height (all frames) | Onsets | Height (at onsets) |")
    emit("|---|---|---|---|---|")
    for c in sorted(frame_h, key=lambda c: -len(frame_h[c])):
        emit(f"| {c}{' (filtered)' if c in VEHICLES else ''} | {len(frame_h[c])} | {' / '.join(_pct(frame_h[c]))} | "
             f"{len(onset_h[c])} | {' / '.join(_pct(onset_h[c]))} |")
    js["quantiles"] = {c: {"frames": _pct(frame_h[c]), "onsets": _pct(onset_h[c])} for c in frame_h}

    emit("\n## 2. Dropped by a minimum height (vehicles only; all splits)\n")
    veh_frames = [h for c in VEHICLES for h in frame_h.get(c, [])]
    veh_tubes = [h for c in VEHICLES for h in tube_max.get(c, [])]
    all_onsets = sum(len(v) for v in onset_h.values())
    emit("| Min height | Vehicle frames dropped | Vehicle tubes dropped entirely | Vehicle onsets dropped | "
         "All agent onsets dropped | Test vehicle onsets left |")
    emit("|---|---|---|---|---|---|")
    js["dropped"] = {}
    for th in THRESHOLDS:
        f = sum(h < th for h in veh_frames) / max(len(veh_frames), 1)
        t = sum(h < th for h in veh_tubes) / max(len(veh_tubes), 1)
        o = sum(h < th for sp in SPLITS for _hd, h in veh_onsets[sp])
        n_o = sum(len(veh_onsets[sp]) for sp in SPLITS)
        left_test = sum(h >= th for _hd, h in veh_onsets["test"])
        emit(f"| {th:.2f} | {f:.1%} | {t:.1%} | {o / max(n_o, 1):.1%} ({o}) | {o / max(all_onsets, 1):.1%} | "
             f"{left_test} of {len(veh_onsets['test'])} |")
        js["dropped"][th] = {"frames": f, "tubes": t, "vehicle_onsets": o, "test_left": left_test}

    heads = Counter(hd for sp in SPLITS for hd, _h in veh_onsets[sp])
    top = [hd for hd, _n in heads.most_common(14)]
    emit("\n## 2b. Onsets left per vehicle head (train / val / test) by minimum height\n")
    emit("| Head | " + " | ".join(["none"] + [f"{th:.2f}" for th in THRESHOLDS]) + " |")
    emit("|---|" + "---|" * (len(THRESHOLDS) + 1))
    for hd in top:
        cells = []
        for th in [0.0] + THRESHOLDS:
            cells.append("/".join(str(sum(1 for h2, h in veh_onsets[sp] if h2 == hd and h >= th)) for sp in SPLITS))
        emit(f"| {hd} | " + " | ".join(cells) + " |")

    # 3. The test events of a run by box height: the models' recall per bin.
    run_id = getattr(args, "run_id", None)
    events_file = args.output_dir / "runs" / str(run_id) / "step06_output" / "paper_events.csv"
    if run_id and os.path.exists(events_file):
        boxes = torch.load(atoms_dir / "test_boxes.pt", map_location="cpu")
        comp = utils_data.load_json(atoms_dir / "test_atoms_compressed.json")
        inv = {vid: {n: uid for uid, n in step05.video_agent_ids(comp[vid]).items()} for vid in comp}
        rows = list(csv.DictReader(open(events_file)))
        bins = [(BINS[i], BINS[i + 1]) for i in range(len(BINS) - 1)]
        stat = {b: Counter() for b in bins}
        no_box = Counter()
        for r in rows:
            if r["label"].split(":", 1)[0] not in VEHICLES:
                continue
            uid = inv.get(r["video"], {}).get(int(r["agent"]))
            h = _height_at((boxes.get(r["video"]) or {}).get(uid), int(r["frame"]))
            caught = {s: r.get(f"{s}@{BUDGET}") not in (None, "") for s in SCORERS}
            target = no_box if h is None else next(stat[b] for b in bins if b[0] <= h < b[1])
            target["events"] += 1
            for s in SCORERS:
                target[s] += caught[s]
        emit(f"\n## 3. Test events of the vehicle heads by box height at the onset, recall at {BUDGET} FA/min "
             f"(run {run_id})\n")
        emit("| Box height | Events | " + " | ".join(f"Recall {s.replace('_cal', '')}" for s in SCORERS) + " |")
        emit("|---|---|" + "---|" * len(SCORERS))
        js["recall_by_height"] = {}
        for b in bins + ["no box"]:
            c = no_box if b == "no box" else stat[b]
            name = b if b == "no box" else f"{b[0]:.2f}–{min(b[1], 1):.2f}"
            emit(f"| {name} | {c['events']} | " + " | ".join(
                f"{c[s] / c['events']:.3f}" if c["events"] else "–" for s in SCORERS) + " |")
            js["recall_by_height"][name] = dict(c)
    elif run_id:
        emit(f"\n(no {events_file}: part 3 skipped)")
    else:
        emit("\n(part 3 skipped: pass --run_id <run> with step 07 results)")

    (out_dir / "box_stats.md").write_text("\n".join(md) + "\n")
    (out_dir / "box_stats.json").write_text(json.dumps(js, indent=1, default=str))
    print(f"\n[Box stats] {out_dir / 'box_stats.md'}")


if __name__ == "__main__":
    main()
