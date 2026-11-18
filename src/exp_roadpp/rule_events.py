"""Which injected rule caused each test event, and how well each scorer detects the events of each rule
(SUMO datasets: the rule log meta/firings says when a rule fired; the ego's events are matched to it).

Groups of the ego's test events (`paper_events.csv` of step 07, at fixed false alarm budgets):
  R1, R3:                AV-Brake onsets after an executed R1 / R3 firing (short lag: the control);
  R5:visible             R5' brakes whose pedestrian was in view at the trigger (no memory needed);
  R5:hidden<3s / >=3s    ... out of view at the trigger, for less / at least MEMORY_S (memory needed);
  AV-Brake:other         AV-Brake onsets without an injected rule (SUMO's own braking, red lights);
  mixed                  two different rules fired near the onset;
  R4:gap<=17s / >17s     AV-TurLft onsets whose previous turn was a right turn (R4), by the gap between
                         the two turn onsets (> 17 s: does not fit into a 20 s clip's evaluated part);
  TurLft:after-left, TurLft:first   the other left turns (no R4);
  other ego / other agents          every other event (reference).
An onset belongs to a firing if it comes MATCH frames around its fire time (the brake label starts a few
frames after the command: acceleration smoothing and hysteresis).

Paired short dataset (cuts of a long one, `run.py --cut_from`): every cut event is mapped to its long video
(meta/results: parent, start_s) and gets the group of the long video's event there, so both runs report
the same instances; body_in_clip says whether the cause (the R5 sighting, the R4 right turn) lies inside
the cut before the event (else the short model cannot know it).

    ./run_roadpp_docker.sh --gpu none --rule_events -- --run_dir <run> --dataset <dataset dir>
    ./run_roadpp_docker.sh --gpu none --rule_events -- --compare <long run>/step06_output/rule_events.csv \\
        <short run>/step06_output/rule_events.csv
Writes <run>/step06_output/rule_events.csv (one row per event: group, long-video key, detections) and
rule_events.json (recall per group, scorer and budget)."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

MEMORY_S = 3.0
MATCH = (-5, 20)            # frames of the AV-Brake onset after the fire time
R4_GAP_S = 17.0             # s: the evaluated part of a 20 s clip (min_context 3 s)
TURN_MERGE = 50             # frames: turn onsets of the same kind closer than this are one turn
MIN_OFF = 10                # frames, as step_dataset.MIN_OFF_FRAMES
SCORERS = ("model_cal", "mlp_cal", "lookup_cal", "model", "mlp", "lookup")
GROUP_ORDER = ("R1", "R3", "R5:visible", "R5:hidden<3s", "R5:hidden>=3s", "mixed", "AV-Brake:other",
               "R4:gap<=17s", "R4:gap>17s", "TurLft:after-left", "TurLft:first", "other ego", "other agents")


class Dataset:
    """A SUMO dataset folder: its firings, ego turn onsets and (cuts) the parent of each video."""
    def __init__(self, path):
        self.path = Path(path)
        self.labels = json.loads((self.path / "gt" / "label.json").read_text())
        split = self.path / "meta" / "video_split.json"
        parent = json.loads(split.read_text()).get("from") if split.exists() else None
        self.parent = Dataset(self.path.parent / parent) if parent else None
        self._cache = {}

    def result(self, vid):
        return json.loads((self.path / "meta" / "results" / f"{vid}.json").read_text())

    def firings(self, vid):
        key = ("firings", vid)
        if key not in self._cache:
            self._cache[key] = json.loads((self.path / "meta" / "firings" / f"{vid}.json").read_text())
        return self._cache[key]

    def turns(self, vid):
        """Sorted [(onset frame, "R" / "L")] of the ego's turns (1-based frames)."""
        key = ("turns", vid)
        if key not in self._cache:
            data = json.loads((self.path / "gt" / f"{vid}_gt.json").read_text())["data"]
            av = self.labels["av_action_labels"]
            frames = {"R": [], "L": []}
            for f, fr in data["frames"].items():
                for k in fr.get("av_action_ids") or []:
                    if av[k] == "AV-TurRht":
                        frames["R"].append(int(f))
                    elif av[k] == "AV-TurLft":
                        frames["L"].append(int(f))
            onsets = []
            for kind, fs in frames.items():
                last = None
                for f in sorted(fs):
                    if last is None or f - last > MIN_OFF:
                        onsets.append((f, kind))
                    last = f
            merged = []
            for f, kind in sorted(onsets):
                if merged and merged[-1][1] == kind and f - merged[-1][0] < TURN_MERGE:
                    continue
                merged.append((f, kind))
            self._cache[key] = merged
        return self._cache[key]


def classify(ds, vid, frame, label, ego):
    """(group, cause frame or None) of one event of video vid at (1-based) frame."""
    if not ego:
        return "other agents", None
    if label == "av:AV-Brake":
        hits = []
        for e in ds.firings(vid):
            if e["rule"] in ("R1", "R3", "R5") and e.get("agent_is_ego") and e.get("executed") and e.get("frame_fire"):
                if MATCH[0] <= frame - e["frame_fire"] <= MATCH[1]:
                    hits.append(e)
        rules = {e["rule"] for e in hits}
        if not hits:
            return "AV-Brake:other", None
        if len(rules) > 1:
            return "mixed", None
        e = min(hits, key=lambda x: abs(frame - x["frame_fire"] - 5))
        if e["rule"] != "R5":
            return e["rule"], e.get("frame_body")
        if e.get("visible_at_trigger"):
            return "R5:visible", e.get("frame_body")
        return ("R5:hidden>=3s" if e.get("hidden_s", 0.0) >= MEMORY_S else "R5:hidden<3s"), e.get("frame_body")
    if label == "av:AV-TurLft":
        before = [(f, k) for f, k in ds.turns(vid) if f < frame - MIN_OFF]
        if not before:
            return "TurLft:first", None
        f, k = before[-1]
        if k == "L":
            return "TurLft:after-left", f
        return ("R4:gap>17s" if (frame - f) / 10 > R4_GAP_S else "R4:gap<=17s"), f
    return "other ego", None


def analyse(run_dir, dataset):
    run_dir, ds = Path(run_dir), Dataset(dataset)
    rows = list(csv.DictReader(open(run_dir / "step06_output" / "paper_events.csv")))
    cols = [c for c in rows[0] if "@" in c] if rows else []
    out = []
    for r in rows:
        if r["type"] == "noise":
            continue
        vid, frame, ego = r["video"], int(r["frame"]), r["ego"] == "True"
        if ds.parent is not None:                           # a cut: the long video's event
            res = ds.result(vid)
            start = int(round(res["start_s"] * 10))
            key_vid, key_frame = res["parent"], start + frame
            group, cause = classify(ds.parent, key_vid, key_frame, r["label"], ego)
            body_in_clip = None if cause is None else cause > start
        else:
            key_vid, key_frame = vid, frame
            group, cause = classify(ds, vid, frame, r["label"], ego)
            body_in_clip = None if cause is None else True
        out.append({"video": vid, "frame": frame, "label": r["label"], "group": group, "long_video": key_vid,
                    "long_frame": key_frame, "body_in_clip": body_in_clip, **{c: r[c] for c in cols}})
    with open(run_dir / "step06_output" / "rule_events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]) if out else ["video"])
        w.writeheader()
        w.writerows(out)
    report = summarize(out, cols)
    (run_dir / "step06_output" / "rule_events.json").write_text(json.dumps(report, indent=1))
    return report


def summarize(events, cols):
    """{group: {"events", "body_in_clip", scorer@budget: recall}}."""
    by = defaultdict(list)
    for e in events:
        by[e["group"]].append(e)
    report = {}
    for g in [g for g in GROUP_ORDER if g in by] + sorted(set(by) - set(GROUP_ORDER)):
        es = by[g]
        inside = [e for e in es if e["body_in_clip"] in (True, "True")]
        report[g] = {"events": len(es), "body_in_clip": len(inside),
                     **{c: round(sum(bool(e[c]) for e in es) / len(es), 3) for c in cols}}
    return report


def print_report(report, budget="5", scorers=("model_cal", "mlp_cal", "lookup_cal")):
    print(f"[Rule events] recall at {budget} false alarms / min")
    print(f"  {'group':18s} {'events':>6s} {'cause in clip':>13s} " + " ".join(f"{s:>10s}" for s in scorers))
    for g, x in report.items():
        print(f"  {g:18s} {x['events']:6d} {x['body_in_clip']:13d} "
              + " ".join(f"{x.get(f'{s}@{budget}', float('nan')):10.3f}" for s in scorers))


def compare(long_csv, short_csv, budget="5", scorers=("model_cal", "mlp_cal", "lookup_cal")):
    """The same events in a long run and its cut run (matched by long video, frame within 2, label): recall
    per group in both, and how many events only one of them detects."""
    def load(p):
        return list(csv.DictReader(open(p)))
    long_ev, short_ev = load(long_csv), load(short_csv)
    index = defaultdict(list)
    for e in short_ev:
        index[(e["long_video"], e["label"])].append(e)
    pairs = []
    for e in long_ev:
        cands = [s for s in index[(e["long_video"], e["label"])] if abs(int(s["long_frame"]) - int(e["long_frame"])) <= 2]
        if cands:
            pairs.append((e, cands[0]))
    print(f"[Compare] {len(pairs)} paired events ({len(long_ev)} long, {len(short_ev)} short), "
          f"recall at {budget} false alarms / min: long / short (only long, only short)")
    by = defaultdict(list)
    for a, b in pairs:
        by[a["group"]].append((a, b))
    out = {}
    for g in [g for g in GROUP_ORDER if g in by] + sorted(set(by) - set(GROUP_ORDER)):
        ps = by[g]
        inside = sum(1 for _a, b in ps if b["body_in_clip"] == "True")
        row = {"events": len(ps), "cause_in_short_clip": inside}
        cells = []
        for s in scorers:
            c = f"{s}@{budget}"
            la = sum(bool(a[c]) for a, _b in ps)
            sb = sum(bool(b[c]) for _a, b in ps)
            only_l = sum(bool(a[c]) and not b[c] for a, b in ps)
            only_s = sum(bool(b[c]) and not a[c] for a, b in ps)
            row[s] = {"long": round(la / len(ps), 3), "short": round(sb / len(ps), 3), "only_long": only_l,
                      "only_short": only_s}
            cells.append(f"{la / len(ps):.2f}/{sb / len(ps):.2f} ({only_l},{only_s})")
        out[g] = row
        print(f"  {g:18s} n {len(ps):4d} cause in cut {inside:4d}  " + "  ".join(f"{s} {c}" for s, c in zip(scorers, cells)))
    return out


def main():
    parser = argparse.ArgumentParser(description="Recall per injected rule (SUMO)")
    parser.add_argument("--run_dir", help="a run folder with step06_output/paper_events.csv")
    parser.add_argument("--dataset", help="the run's SUMO dataset folder")
    parser.add_argument("--compare", nargs=2, metavar=("LONG_CSV", "SHORT_CSV"),
                        help="rule_events.csv of a long run and of its cut run")
    parser.add_argument("--budget", default="5")
    args, _unknown = parser.parse_known_args()             # the docker script adds --device / --machine
    if args.compare:
        out = compare(*args.compare, budget=args.budget)
        Path(args.compare[1]).with_name("rule_events_compare.json").write_text(json.dumps(out, indent=1))
        return
    if not (args.run_dir and args.dataset):
        parser.error("--run_dir and --dataset, or --compare")
    print_report(analyse(args.run_dir, args.dataset), args.budget)


if __name__ == "__main__":
    main()
