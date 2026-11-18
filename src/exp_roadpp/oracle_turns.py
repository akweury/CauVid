"""Does a model rule out what cannot happen? The ego's turns on the SUMO grids against the routes (oracle):
after a right turn the next junction is a left turn (R4), so AV-TurRht cannot start there; on the last edge
(to the network boundary) no turn can.

From the routes (meta/results: ego_route, ego_turns), the world states (meta/world: the frame the ego enters
each junction) and step 07's per-frame ego scores (ego_scores.pt), in the last `window` frames before the
ego enters each junction (where a turn would start within the window):
  forced    the previous turn was right (R4): AV-TurRht impossible -> share excluded (impossibility recall);
  free_right / free_other   no R4: the ego does / does not turn right there (excluding a right turn that
            comes is a violation);
  exit      the edges after the last junction: AV-TurRht and AV-TurLft impossible.
Excluded: score <= the exclusion threshold step 07 chose on val (group ego_nontrivial, else nontrivial, at
the headline violation budget). Also the mean score and the AUC forced vs free_right (does the score drop
after a right turn?). Cuts of a long dataset (`run.py --cut_from`) are mapped to their long video: the R4
status comes from the full route, also when the right turn lies before the cut (the short model cannot
know it then: cause_in_clip).

    ./run_roadpp_docker.sh --gpu none --oracle_turns -- --run_dir <run> --dataset <dataset dir>
Writes <run>/step06_output/oracle_turns.json."""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import torch

SCORERS = ("model_cal", "mlp_cal", "lookup_cal", "prior")


def junction_entries(world):
    """[(world index, junction id)] of the frames the ego enters a junction (an internal edge ":J..")."""
    out, inside = [], None
    for k, w in enumerate(world):
        road = w["veh"].get("ego", {}).get("road", "")
        node = road[1:].rsplit("_", 1)[0] if road.startswith(":J") else None
        if node is not None and node != inside:
            out.append((k, node))
        inside = node
    return out


def thresholds(metrics, budget):
    ex = metrics.get("exclusion", {}).get("scorers", {})
    out = {}
    for nm, groups in ex.items():
        for g in ("ego_nontrivial", "nontrivial"):
            x = groups.get(g, {}).get(str(budget))
            if x and x.get("threshold") is not None:
                out[nm] = x["threshold"]
                break
    return out


def auc(pos, neg):
    """P(score of a pos > score of a neg) (ties half): here pos = free_right, neg = forced."""
    if not pos or not neg:
        return None
    s = torch.tensor(pos)[:, None] - torch.tensor(neg)[None, :]
    return float(((s > 0).float() + 0.5 * (s == 0).float()).mean())


def evaluate(run_dir, dataset):
    run_dir, dataset = Path(run_dir), Path(dataset)
    out_dir = run_dir / "step06_output"
    ego = torch.load(out_dir / "ego_scores.pt", map_location="cpu")
    metrics = json.loads((out_dir / "paper_metrics.json").read_text())
    budget = metrics.get("exclusion", {}).get("settings", {}).get("headline", 0.01)
    th = thresholds(metrics, budget)
    split = dataset / "meta" / "video_split.json"
    parent_name = json.loads(split.read_text()).get("from") if split.exists() else None
    long_ds = dataset.parent / parent_name if parent_name else dataset
    heads = ego["heads"]
    if "av:AV-TurRht" not in heads:
        raise SystemExit("[Oracle] no av:AV-TurRht head among the ego heads")
    i_r, i_l = heads.index("av:AV-TurRht"), heads.index("av:AV-TurLft") if "av:AV-TurLft" in heads else None
    W = ego["window"]
    scores = defaultdict(lambda: defaultdict(list))     # (category, head) -> scorer -> [scores]
    cause = defaultdict(int)
    worlds = {}
    for vid, v in ego["videos"].items():
        if parent_name:
            res = json.loads((dataset / "meta" / "results" / f"{vid}.json").read_text())
            parent, start = res["parent"], int(round(res["start_s"] * 10))
        else:
            parent, start = vid, 0
        res = json.loads((long_ds / "meta" / "results" / f"{parent}.json").read_text())
        turns = res.get("ego_turns") or []
        if parent not in worlds:
            with gzip.open(long_ds / "meta" / "world" / f"{parent}.json.gz", "rt") as f:
                worlds = {parent: json.load(f)}             # one long video at a time
        entries = junction_entries(worlds[parent])
        known = v["known"]
        T = known.shape[0]

        def collect(category, head, k_from, k_to, cause_frame=None):
            for k in range(k_from, k_to):
                t = k - start + 1                           # world index k -> frame k + 1 of the long video
                if not (0 <= t < T) or not bool(known[t, head]):
                    continue
                for nm in SCORERS:
                    if nm in v:
                        scores[(category, head)][nm].append(float(v[nm][t, head]))
                if cause_frame is not None:
                    cause[(category, cause_frame - start + 1 >= 1)] += 1

        for i, (k_in, _node) in enumerate(entries):
            if i >= len(turns):
                break
            prev = turns[i - 1] if i >= 1 else None
            seg = (max(0, k_in - W), k_in)
            if prev == "right":
                cause_k = entries[i - 1][0] if i >= 1 else None
                collect("forced", i_r, *seg, cause_k)
                if i_l is not None:
                    collect("forced_left_comes", i_l, *seg)
            else:
                collect("free_right" if turns[i] == "right" else "free_other", i_r, *seg)
        if entries and len(entries) >= len(turns) and turns:
            last_out = entries[len(turns) - 1][0] + 30        # past the last junction
            for head in [h for h in (i_r, i_l) if h is not None]:
                collect("exit", head, last_out, len(worlds[parent]))
    report = {"budget": budget, "thresholds": th, "categories": {}}
    for (cat, head), per in sorted(scores.items()):
        row = {"head": heads[head], "frames": len(next(iter(per.values()), []))}
        for nm, xs in per.items():
            ex = sum(x <= th[nm] for x in xs) / len(xs) if nm in th and xs else None
            row[nm] = {"mean_score": round(sum(xs) / len(xs), 4) if xs else None,
                       "excluded": round(ex, 3) if ex is not None else None}
        report["categories"][f"{cat}:{heads[head]}"] = row
    report["auc_free_right_vs_forced"] = {
        nm: auc(scores[("free_right", i_r)].get(nm, []), scores[("forced", i_r)].get(nm, [])) for nm in SCORERS}
    report["forced_cause_in_clip"] = {"in": cause[("forced", True)], "before": cause[("forced", False)]}
    (out_dir / "oracle_turns.json").write_text(json.dumps(report, indent=1))
    return report


def print_report(r):
    print(f"[Oracle turns] exclusion at violation budget {r['budget']} (thresholds from val): share of frames "
          f"excluded (mean score)")
    for cat, row in r["categories"].items():
        cells = "  ".join(f"{nm} {x['excluded']} ({x['mean_score']})" for nm, x in row.items()
                          if isinstance(x, dict))
        print(f"  {cat:32s} n {row['frames']:6d}  {cells}")
    print("  AUC free_right vs forced (higher: the score drops after a right turn): "
          + ", ".join(f"{nm} {v:.3f}" for nm, v in r["auc_free_right_vs_forced"].items() if v is not None))
    c = r["forced_cause_in_clip"]
    print(f"  forced frames whose right turn lies in the clip: {c['in']}, before it: {c['before']}")


def main():
    parser = argparse.ArgumentParser(description="Exclusion of impossible ego turns (SUMO grids, R4)")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset", required=True)
    args, _unknown = parser.parse_known_args()
    print_report(evaluate(args.run_dir, args.dataset))


if __name__ == "__main__":
    main()
