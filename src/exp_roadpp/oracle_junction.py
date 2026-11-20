"""Does a model rule out what cannot happen? The ego on the 20 s single-junction SUMO maps (sumo_maps_v*),
against the simulator (oracle). Counterpart of `oracle_turns` (grids, R4).

Per test frame t of the ego (the row predicting an onset in frames t..t + W - 1; it has seen the frames
before t, world index t - 2 is the last one) and ego head, the frames fall into categories:
  red_hold   AV-Mov    the ego stands (< STOP_ON m/s) first at the stop line (<= FIRST_M m) and its own
                       signal stays red over the last seen frame and the whole window: it cannot start.
                       Visible cause: TL:Red (the signal head past the junction, sumo_v6 on).
  stopped_other AV-Mov the ego stands, not red_hold (green, a light to come, priority / stop control,
                       or not first in the queue): it may start; stopped_go: those where it does.
  exit       AV-TurLft / AV-TurRht   the ego is on its outgoing edge (past the only junction): no turn.
                       Visible cause: the turn (or the crossing) it just made.
  approach_<turn>  AV-TurLft / AV-TurRht   the window holds the ego's entry into the junction, split by
                       the route's turn (from the arm angles): straight -> no turn label can start
                       (impossible, but the route is hidden; a lane change at most hints at it); the head
                       of the route's own turn comes (free).
Per category and scorer: the share of frames excluded (score <= the head's exclusion threshold step 07
chose on val: the excluded cells' onset rate <= the headline rho x the head's base rate), the mean score, and the oracle's own
violations (frames where the head did start: should be 0 for red_hold / exit; for approach_straight it
shows bends the turn labels count as turns). AUCs: does the score drop where the head is impossible
(stopped_go vs red_hold for AV-Mov; approach_right vs approach_straight for AV-TurRht, left likewise)?

Needs step 07 paper_v5 (ego_scores.pt with the targets; older ones: no violations / stopped_go).
    ./run_roadpp_docker.sh --gpu none --oracle_junction -- --run_dir <run> --dataset <dataset dir>
Writes <run>/step06_output/oracle_junction.json."""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import torch

from src.exp_roadpp.oracle_turns import SCORERS, auc, excluded_share, thresholds

STOP_ON = 0.3          # m/s, as synth/sumo/labels.py (Stop below it)
FIRST_M = 5.0          # m to the stop line: first in the queue (the next car waits ~7.5 m back)
STRAIGHT_DEG = 20.0    # route bend below this: straight (labels turn at a yaw rate > 8 deg/s in the junction)
JUNCTION = ":C"


def route_turn(spec, route):
    """"straight", "right" or "left" of the ego route (in edge "<arm>2C", out edge "C2<arm>") from the
    arm angles (deg from +x, ccw, as `scenario.random_spec`)."""
    a_in, a_out = route[0][:-2], route[1][2:]
    heading_in = spec["arms"][a_in]["angle"] + 180.0
    bend = (spec["arms"][a_out]["angle"] - heading_in + 540.0) % 360.0 - 180.0    # ccw positive
    if abs(bend) < STRAIGHT_DEG:
        return "straight"
    return "left" if bend > 0 else "right"


def is_red(tl):
    return tl is not None and tl.get("state", "")[:1] in ("r", "R")


def evaluate(run_dir, dataset):
    run_dir, dataset = Path(run_dir), Path(dataset)
    out_dir = run_dir / "step06_output"
    ego = torch.load(out_dir / "ego_scores.pt", map_location="cpu")
    metrics = json.loads((out_dir / "paper_metrics.json").read_text())
    budget, th = thresholds(metrics)
    index = json.loads((dataset / "meta" / "maps.json").read_text())
    maps, video_map = index["maps"], index.get("videos", {})
    heads, W = ego["heads"], ego["window"]
    idx = {h: heads.index(h) for h in ("av:AV-Mov", "av:AV-TurLft", "av:AV-TurRht") if h in heads}
    if not idx:
        raise SystemExit(f"[Oracle] none of AV-Mov / AV-TurLft / AV-TurRht among the ego heads {heads}")
    scores = defaultdict(lambda: defaultdict(list))     # (category, head) -> scorer -> [scores]
    hits = defaultdict(lambda: [0, 0])                  # (category, head) -> [frames, head started]
    routes = defaultdict(int)
    for vid, v in ego["videos"].items():
        res = json.loads((dataset / "meta" / "results" / f"{vid}.json").read_text())
        entry = maps.get(res.get("map") or video_map.get(vid))
        spec = None if entry is None else entry.get("spec", entry)     # maps.json: {map id: {name, net, spec}}
        if spec is None or "grid" in (entry.get("layout"), spec.get("layout")) or "arms" not in spec:
            continue
        with gzip.open(dataset / "meta" / "world" / f"{vid}.json.gz", "rt") as f:
            world = json.load(f)
        turn = route_turn(spec, res["ego_route"])
        routes[turn] += 1
        out_edge = res["ego_route"][1]
        enter = next((k for k, w in enumerate(world)
                      if w["veh"].get("ego", {}).get("road", "").startswith(JUNCTION)), None)
        known, target = v["known"], v.get("target")
        T = known.shape[0]

        def add(category, head, t):
            h = idx.get(head)
            if h is None or not bool(known[t, h]):
                return
            for nm in SCORERS:
                if nm in v:
                    scores[(category, head)][nm].append(float(v[nm][t, h]))
            if target is not None:
                hits[(category, head)][0] += 1
                hits[(category, head)][1] += int(bool(target[t, h]))

        for t in range(2, T):
            last = t - 2                                       # world index of the last seen frame
            span = range(last, min(t + W - 1, len(world)))     # last seen frame .. end of the window
            if len(span) < W + 1:
                continue
            e = world[last]["veh"].get("ego")
            if e is None:
                continue
            if e["speed"] < STOP_ON:
                tl = world[last].get("tl")
                red_hold = (tl is not None and tl["dist"] <= FIRST_M
                            and all(is_red(world[k].get("tl")) for k in span))
                if red_hold:
                    add("red_hold", "av:AV-Mov", t)
                else:
                    add("stopped_other", "av:AV-Mov", t)
                    if target is not None and "av:AV-Mov" in idx and bool(target[t, idx["av:AV-Mov"]]):
                        add("stopped_go", "av:AV-Mov", t)
            if e["road"] == out_edge:
                add("exit", "av:AV-TurLft", t)
                add("exit", "av:AV-TurRht", t)
            elif enter is not None and last < enter <= t + W - 2:
                add(f"approach_{turn}", "av:AV-TurLft", t)
                add(f"approach_{turn}", "av:AV-TurRht", t)
    report = {"budget": budget, "thresholds": th, "window": W, "routes": dict(routes), "categories": {}}
    for (cat, head), per in sorted(scores.items()):
        n, started = hits[(cat, head)]
        row = {"frames": len(next(iter(per.values()), [])),
               "oracle_violation": round(started / n, 4) if n else None}
        for nm, xs in per.items():
            ex = excluded_share(xs, th.get(nm, {}).get(head))
            row[nm] = {"mean_score": round(sum(xs) / len(xs), 4) if xs else None,
                       "excluded": round(ex, 3) if ex is not None else None}
        report["categories"][f"{cat}:{head}"] = row

    def pair(pos, neg, head):
        return {nm: auc(scores[(pos, head)].get(nm, []), scores[(neg, head)].get(nm, [])) for nm in SCORERS}

    report["auc"] = {"AV-Mov stopped_go vs red_hold": pair("stopped_go", "red_hold", "av:AV-Mov"),
                     "AV-Mov stopped_other vs red_hold": pair("stopped_other", "red_hold", "av:AV-Mov"),
                     "AV-TurRht approach_right vs approach_straight":
                         pair("approach_right", "approach_straight", "av:AV-TurRht"),
                     "AV-TurLft approach_left vs approach_straight":
                         pair("approach_left", "approach_straight", "av:AV-TurLft")}
    (out_dir / "oracle_junction.json").write_text(json.dumps(report, indent=1))
    return report


def print_report(r):
    print(f"[Oracle junction] W {r['window']} frames, exclusion at rho {r['budget']} x the base rate (per-head "
          f"thresholds from val); routes {r['routes']}. Per category: frames, the oracle's own violations (should be 0 "
          f"where impossible), share excluded (mean score) per scorer")
    for cat, row in r["categories"].items():
        cells = "  ".join(f"{nm} {x['excluded']} ({x['mean_score']})" for nm, x in row.items()
                          if isinstance(x, dict))
        print(f"  {cat:38s} n {row['frames']:6d}  viol {row['oracle_violation']}  {cells}")
    print("  AUC (higher: the score drops where the head is impossible):")
    for name, per in r["auc"].items():
        cells = ", ".join(f"{nm} {v:.3f}" for nm, v in per.items() if v is not None)
        print(f"    {name:46s} {cells or 'n/a'}")


def main():
    parser = argparse.ArgumentParser(description="Exclusion of impossible ego actions (SUMO single-junction maps)")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset", required=True)
    args, _unknown = parser.parse_known_args()
    print_report(evaluate(args.run_dir, args.dataset))


if __name__ == "__main__":
    main()
