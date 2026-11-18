"""How accurate are the prediction atoms (`predict.py`)? On a SUMO dataset, the estimates from the GT boxes
against the world states (meta/world): distance and lateral offset, relative velocities, and the reach
time and path conflict against what really happens next (constant velocity is part of what is tested).

Truth per labelled pedestrian / cyclist frame t, in the ego's camera frame at t (`observe.Camera`):
  Z, X: the agent's position (forward, right); vZ, vX: their change over the last second;
  reach: the first later frame (within the last bin's end) with Z <= predict.Z_FRONT ("none" if never,
  unknown if the agent leaves the world first); conflict: |X| <= predict.CORRIDOR_HALF at that frame.
Reported: coverage (share of the labelled frames with an estimate), errors by true distance, the bin
confusion of the reach time (est vs true, raw bins without hysteresis), conflict precision / recall.

    ./run_roadpp_docker.sh --gpu none --predict_check -- --dataset <data root>/synthetic/sumo_long_v3 --num 100
Writes <dataset>/meta/predict_check.json."""
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.exp_roadpp import predict
from src.exp_roadpp.synth.sumo import observe

DIST_BINS = ((0, 10), (10, 20), (20, 30), (30, 50))
HORIZON = int(predict.BINS[-1][1] * 10)         # frames looked ahead for the true reach


def video_arrays(data, labels):
    """(boxes (T, N, 4), classes [N], agent ids [N], ego_turning (T,)) of one gt video's "data"."""
    T = data["numf"]
    uids = sorted({a["tube_uid"] for fr in data["frames"].values() for a in fr["annos"].values()})
    col = {u: n for n, u in enumerate(uids)}
    boxes = np.full((T, len(uids), 4), np.nan)
    classes = [None] * len(uids)
    turning = np.zeros(T, dtype=bool)
    for f, fr in data["frames"].items():
        t = int(f) - 1
        av = [labels["av_action_labels"][k] for k in fr.get("av_action_ids") or []]
        turning[t] = any(a in predict.EGO_TURN_LABELS for a in av)
        for a in fr["annos"].values():
            n = col[a["tube_uid"]]
            boxes[t, n] = a["box"]
            classes[n] = labels["agent_labels"][a["agent_ids"][0]]
    return boxes, classes, [u.split("-", 1)[1] for u in uids], turning


def truth(world, aid):
    """(T,) Z, X of agent aid in the ego camera frame of each frame (NaN where it or the ego is missing)."""
    T = len(world)
    Z, X = np.full(T, np.nan), np.full(T, np.nan)
    for t, w in enumerate(world):
        ego = w["veh"].get("ego")
        a = w["ped"].get(aid) or w["veh"].get(aid)
        if ego is None or a is None:
            continue
        f, l = observe.Camera(ego).to_cam(a["x"], a["y"])
        Z[t], X[t] = f, -l
    return Z, X


def true_reach(Z, X, t):
    """(reach s or inf, conflict bool) after frame t, or None if unknown (the agent leaves first)."""
    for k in range(t + 1, min(len(Z), t + HORIZON + 1)):
        if not np.isfinite(Z[k]):
            return None
        if Z[k] <= predict.Z_FRONT:
            return (k - t) / 10, abs(X[k]) <= predict.CORRIDOR_HALF
    return (float("inf"), False) if t + HORIZON < len(Z) else None


def bin_of(v):
    if v is None or not np.isfinite(v):
        return "none"
    for a, b in predict.BINS:
        if a <= v < b:
            return f"{a:g}-{b:g}s"
    return "none"


def check(dataset, num):
    dataset = Path(dataset)
    labels = json.loads((dataset / "gt" / "label.json").read_text())
    gts = sorted((dataset / "gt").glob("*_gt.json"))[:num]
    err = defaultdict(list)
    cover = Counter()
    confusion = defaultdict(Counter)
    conflict = Counter()
    for gt in gts:
        vid = gt.name[:-len("_gt.json")]
        data = json.loads(gt.read_text())["data"]
        with gzip.open(dataset / "meta" / "world" / f"{vid}.json.gz", "rt") as f:
            world = json.load(f)[:data["numf"]]
        boxes, classes, aids, turning = video_arrays(data, labels)
        est = predict.estimate(boxes, classes, turning)
        for n, (cls, aid) in enumerate(zip(classes, aids)):
            if cls not in predict.CLASSES:
                continue
            Zt, Xt = truth(world, aid)
            for t in np.nonzero(np.isfinite(boxes[:, n, 0]))[0]:
                cover["frames"] += 1
                if not est["known"][t, n] or not np.isfinite(Zt[t]):
                    continue
                cover["known"] += 1
                d = next((f"{a}-{b}m" for a, b in DIST_BINS if a <= Zt[t] < b), None)
                if d:
                    err[f"Z_rel {d}"].append(abs(est["Z"][t, n] / Zt[t] - 1))
                    err[f"X_abs {d}"].append(abs(est["X"][t, n] - Xt[t]))
                if t >= 10 and np.isfinite(Zt[t - 10]):
                    err["vZ_abs"].append(abs(est["vZ"][t, n] - (Zt[t] - Zt[t - 10])))
                    err["vX_abs"].append(abs(est["vX"][t, n] - (Xt[t] - Xt[t - 10])))
                tr = true_reach(Zt, Xt, t)
                if tr is None:
                    continue
                r_est = est["reach"][t, n]
                confusion[bin_of(tr[0])][bin_of(r_est)] += 1
                if np.isfinite(r_est):
                    err["reach_rel (both finite)"].append(abs(r_est / tr[0] - 1) if np.isfinite(tr[0]) and tr[0] > 0 else np.nan)
                conflict[("est" if est["conflict"][t, n] else "no") + "/" + ("true" if tr[1] else "false")] += 1
    q = lambda v, p: round(float(np.nanpercentile(v, p)), 3) if len(v) else None
    report = {"videos": len(gts), "coverage": round(cover["known"] / max(1, cover["frames"]), 3),
              "labelled_frames": cover["frames"],
              "errors": {k: {"n": len(v), "median": q(v, 50), "p90": q(v, 90)} for k, v in sorted(err.items())},
              "reach_confusion_true_to_est": {k: dict(v) for k, v in sorted(confusion.items())},
              "conflict": dict(conflict)}
    tp, fp, fn = conflict["est/true"], conflict["est/false"], conflict["no/true"]
    report["conflict_precision"] = round(tp / max(1, tp + fp), 3)
    report["conflict_recall"] = round(tp / max(1, tp + fn), 3)
    bins = [f"{a:g}-{b:g}s" for a, b in predict.BINS] + ["none"]
    right = sum(confusion[b][b] for b in bins)
    total = sum(sum(c.values()) for c in confusion.values())
    report["reach_bin_accuracy"] = round(right / max(1, total), 3)
    (dataset / "meta" / "predict_check.json").write_text(json.dumps(report, indent=1))
    return report


def print_report(r):
    print(f"[Predict check] {r['videos']} videos, {r['labelled_frames']} labelled Ped/Cyc frames, "
          f"coverage {r['coverage']:.0%} (frames with an estimate)")
    for k, v in r["errors"].items():
        print(f"  {k:26s} median {v['median']}  p90 {v['p90']}  (n {v['n']})")
    print(f"  reach bin accuracy {r['reach_bin_accuracy']:.0%}; confusion (true -> est):")
    for t, row in r["reach_confusion_true_to_est"].items():
        print(f"    {t:7s} " + ", ".join(f"{e} {n}" for e, n in sorted(row.items())))
    print(f"  conflict precision {r['conflict_precision']:.2f}, recall {r['conflict_recall']:.2f}  {r['conflict']}")


def main():
    parser = argparse.ArgumentParser(description="Accuracy of the prediction atoms on a SUMO dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num", type=int, default=100, help="videos (sorted by name)")
    args, _unknown = parser.parse_known_args()             # the docker script adds --device / --machine
    print_report(check(args.dataset, args.num))


if __name__ == "__main__":
    main()
