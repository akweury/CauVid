"""Does the time an atom has held predict the next onset? A check in the labels before duration atoms.

For a body label B and a head label H of the same agent (the ego: AV-* labels), at every frame t where B
holds, its age is t - (start of its current run) (runs of the same atom joined over gaps shorter than
MIN_OFF_FRAMES, as `step_dataset.head_onsets`). Per age bin (log scale, frames):

    P(an onset of H in (t, t + WINDOW] | B holds at t with age in the bin)

A curve that rises (or falls) with the age means a duration atom "B held a..b frames" carries information
that the plain atom B lacks. A run that starts at the agent's first frame (the ego: frame 1) began before
it was seen: its age is unknown (left-censored), counted as "cens" and not in the bins.

    python -m src.exp_roadpp.duration_check --exp sumo_junction_v3 --machine local
    ./run_roadpp_docker.sh --gpu none --duration_check -- --exp full_sn
Writes <exp>/step_dataset_output/duration_check.json."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

MIN_OFF_FRAMES = 10         # as step_dataset.MIN_OFF_FRAMES (not imported: that module loads torch)
WINDOW = 10                 # frames ahead in which the head onset counts (1 s)
BINS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, None)]     # age in frames, [a, b)

# (body label, head labels): the same agent; labels across all agent classes. The ego's are AV-*.
PAIRS = [
    ("AV-Brake", ("AV-Stop",)),                       # SUMO only (roadpp has no AV-Brake)
    ("AV-Stop", ("AV-Mov",)),                         # waiting at a red light / behind a stopped car
    ("AV-Mov", ("AV-Stop",)),
    ("AV-Mov", ("AV-Brake",)),                        # SUMO only
    ("Brake", ("Stop",)),
    ("Stop", ("MovAway", "MovTow", "Mov")),
    ("HazLit", ("Brake",)),                           # SUMO R2
    ("Wait2X", ("Xing", "XingFmLft", "XingFmRht")),
    ("Red", ("Green",)),
    ("Green", ("Amber", "Red")),
    ("IncatLft", ("TurLft", "MovLft")),
    ("IncatRht", ("TurRht", "MovRht")),
]


def bin_of(age):
    for k, (a, b) in enumerate(BINS):
        if age >= a and (b is None or age < b):
            return k
    raise ValueError(age)


def bin_name(k):
    a, b = BINS[k]
    return f"{a}-{b}" if b is not None else f"{a}+"


def runs(spans):
    """Sorted (start, end) runs of an atom's intervals, joined over gaps shorter than MIN_OFF_FRAMES."""
    out = []
    for s, e in sorted(spans):
        if out and s - out[-1][1] < MIN_OFF_FRAMES:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def video_labels(interval_atoms):
    """{agent: {label: runs}} and {agent: first frame} of one video (the ego as "ego")."""
    spans = defaultdict(lambda: defaultdict(list))
    first = {}
    for atom in interval_atoms:
        agent = atom.get("tube_uid") or "ego"
        s, e = int(atom["start_frame"]), int(atom["end_frame"])
        first[agent] = min(first.get(agent, s), s)
        label = atom.get("action_id_label") or atom.get("location_name_label")
        spans[agent][label].append((s, e))
    return {a: {lab: runs(v) for lab, v in per.items()} for a, per in spans.items()}, first


def head_onsets(per, heads, first):
    """Onset frames of the union of the head labels (as step_dataset.head_onsets: not on the agent's first frame)."""
    starts = []
    for s, _e in runs([se for h in heads for se in per.get(h, [])]):
        if s > first:
            starts.append(s)
    return starts


def check(videos):
    """{pair: {"bins": [{bin, frames, hits, p}], "cens": {...}, "all": {...}, "runs": n}}."""
    stats = {f"{b}>{'|'.join(h)}": {"frames": [0] * len(BINS), "hits": [0] * len(BINS), "cens": [0, 0], "runs": 0}
             for b, h in PAIRS}
    for interval_atoms in videos:
        labels, first = video_labels(interval_atoms)
        for agent, per in labels.items():
            for body, heads in PAIRS:
                if body not in per:
                    continue
                st = stats[f"{body}>{'|'.join(heads)}"]
                onsets = head_onsets(per, heads, first[agent])
                for s, e in per[body]:
                    st["runs"] += 1
                    censored = s <= first[agent]
                    for t in range(s, e + 1):
                        hit = any(t < x <= t + WINDOW for x in onsets)
                        if censored:
                            st["cens"][0] += 1
                            st["cens"][1] += hit
                        else:
                            k = bin_of(t - s)
                            st["frames"][k] += 1
                            st["hits"][k] += hit
    report = {}
    for pair, st in stats.items():
        n, h = sum(st["frames"]), sum(st["hits"])
        p = [st["hits"][k] / st["frames"][k] if st["frames"][k] else None for k in range(len(BINS))]
        valid = [x for k, x in enumerate(p) if x is not None and st["frames"][k] >= 50]
        report[pair] = {"runs": st["runs"], "frames": n, "p_all": round(h / n, 4) if n else None,
                        "bins": [{"age": bin_name(k), "frames": st["frames"][k], "hits": st["hits"][k],
                                  "p": None if p[k] is None else round(p[k], 4)} for k in range(len(BINS))],
                        "censored": {"frames": st["cens"][0], "p": round(st["cens"][1] / st["cens"][0], 4) if st["cens"][0] else None},
                        "max_over_min": round(max(valid) / min(valid), 2) if len(valid) >= 2 and min(valid) > 0 else None}
    return report


def print_report(report):
    print(f"[Duration check] P(head onset within {WINDOW} frames | body holds, age bin); bins with < 50 frames in ()")
    print(f"  {'body > head':32s} {'runs':>6s} {'p_all':>6s} " + " ".join(f"{bin_name(k):>8s}" for k in range(len(BINS)))
          + f" {'cens':>6s} {'max/min':>7s}")
    for pair, r in report.items():
        if not r["frames"]:
            continue
        cells = []
        for b in r["bins"]:
            if b["p"] is None:
                cells.append(f"{'-':>8s}")
            elif b["frames"] < 50:
                cells.append(f"{'(' + format(b['p'], '.3f') + ')':>8s}")
            else:
                cells.append(f"{b['p']:8.3f}")
        cens = "-" if r["censored"]["p"] is None else format(r["censored"]["p"], ".3f")
        print(f"  {pair:32s} {r['runs']:6d} {r['p_all']:6.3f} " + " ".join(cells)
              + f" {cens:>6s} {str(r['max_over_min'] or '-'):>7s}")


def load_videos(atoms_dir, splits):
    for sp in splits:
        data = json.loads((atoms_dir / f"{sp}_atoms.json").read_text())
        for v in data.values():
            yield v["interval_atoms"]


def main():
    parser = argparse.ArgumentParser(description="Does an atom's age predict the next onset?")
    parser.add_argument("--exp", required=True)
    parser.add_argument("--machine", default="dgx-jing", help="dgx-jing: the server output; else the local output")
    parser.add_argument("--splits", nargs="+", default=["train"], help="atom files: train val test (or all)")
    args, _unknown = parser.parse_known_args()             # the docker script adds --device
    if args.machine == "dgx-jing":
        out = Path(f"/storage-01/ml-jsha/CauVid_output/roadpp/{args.exp}/step_dataset_output")
    else:
        out = Path(__file__).resolve().parent / "output" / "roadpp" / args.exp / "step_dataset_output"
    report = check(load_videos(out / "atoms", args.splits))
    print_report(report)
    (out / "duration_check.json").write_text(json.dumps(
        {"window": WINDOW, "bins": [bin_name(k) for k in range(len(BINS))], "splits": args.splits, "pairs": report}, indent=1))
    print(f"[Duration check] -> {out / 'duration_check.json'}")


if __name__ == "__main__":
    main()
