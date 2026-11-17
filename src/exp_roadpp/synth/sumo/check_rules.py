"""Are the injected rules in the labels? For each exact rule, from the generated ground truth only (what the
pipeline sees), with the onsets as `step_dataset.head_onsets` counts them (a label run starting at least
MIN_OFF frames after the previous one, not on the agent's first frame):

    P(head onset in the lag window | body onset)  vs  P(head onset in a window as long | control moment)

A rule whose lift is small cannot be learned from the labels (sumo_junction_v0: R1 lift 1.0). The rule
log adds why instances are lost (not fired, not executed, not visible, head not seen).

    python -m src.exp_roadpp.synth.sumo.check_rules --dataset <dataset dir>
    ./run_roadpp_docker.sh --gpu none --sumo_check -- --dataset /storage-01/ml-jsha/CauVid_Data/synthetic/<name>
Writes <dataset>/meta/rule_check.json; `run.py` runs it after the generation."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.exp_roadpp.synth.sumo import rules

MIN_OFF = 10                # frames, as step_dataset.MIN_OFF_FRAMES
SLACK = 5                   # frames added on both sides of the lag window
CONTROL_STEP = 10           # frames between control moments
MIN_LIFT, MIN_P = 2.0, 0.4  # a rule is "ok" above both


def label_frames(video, labels):
    """{agent: {label: sorted frames}} of one ROAD++ video (agents by tube_uid, the ego as "ego"), and
    {agent: its first frame}."""
    out = defaultdict(lambda: defaultdict(list))
    first = {"ego": 1}
    for f, fr in video["frames"].items():
        f = int(f)
        for k in fr.get("av_action_ids") or []:
            out["ego"][labels["av_action_labels"][k]].append(f)
        for a in fr["annos"].values():
            uid = a["tube_uid"]
            first[uid] = min(first.get(uid, f), f)
            for k in a["action_ids"]:
                out[uid][labels["action_labels"][k]].append(f)
    for per in out.values():
        for v in per.values():
            v.sort()
    return out, first


def onsets(frames, first):
    """Onset frames of a sorted frame list (runs at least MIN_OFF frames apart, not on the first frame)."""
    out, last = [], None
    for f in frames:
        if (last is None or f - last > MIN_OFF) and f > first:    # a shorter gap: the same run
            out.append(f)
        last = f
    return out


def _hit(onset_list, start, lag):
    lo, hi = int(lag[0] * 10) - SLACK, int(lag[1] * 10) + SLACK
    return any(start + lo <= x <= start + hi for x in onset_list)


def check(dataset):
    dataset = Path(dataset)
    labels = json.loads((dataset / "gt" / "label.json").read_text())
    r1_lag, r2_lag = rules.R1["lag_s"], rules.R2["lag_s"]
    width_r1, width_r2 = int(r1_lag[1] * 10) - int(r1_lag[0] * 10), int(r2_lag[1] * 10) - int(r2_lag[0] * 10)
    stats = Counter()
    log = defaultdict(Counter)
    for gt in sorted((dataset / "gt").glob("*_gt.json")):
        vid = gt.name[:-len("_gt.json")]
        video = json.loads(gt.read_text())["data"]
        lf, first = label_frames(video, labels)
        ego = lf["ego"]
        mov = set(ego.get("AV-Mov", []))
        av_brake = onsets(ego.get("AV-Brake", []), 1)

        # R1 (ego): the first frame of each visible Wait2X run of a pedestrian at which the ego is AV-Mov
        for uid, per in lf.items():
            if uid == "ego" or "Wait2X" not in per:
                continue
            run = []
            for f in per["Wait2X"] + [None]:
                if f is not None and (not run or f - run[-1] <= MIN_OFF):
                    run.append(f)
                    continue
                start = next((x for x in run if x in mov), None)
                if start is not None:
                    stats["r1_events"] += 1
                    stats["r1_hits"] += _hit(av_brake, start, r1_lag)
                run = [f] if f is not None else []
        for t in sorted(mov)[::CONTROL_STEP]:
            stats["r1_control"] += 1
            stats["r1_control_hits"] += any(t + 1 <= x <= t + 1 + width_r1 + 2 * SLACK for x in av_brake)

        # R2: the HazLit onset of a vehicle -> its own Brake onset
        for uid, per in lf.items():
            if uid == "ego":
                continue
            brake = onsets(per.get("Brake", []), first[uid])
            for e in onsets(per.get("HazLit", []), first[uid]):
                stats["r2_events"] += 1
                stats["r2_hits"] += _hit(brake, e, r2_lag)
            moving = sorted(set(per.get("MovAway", [])) | set(per.get("MovTow", [])))
            for t in moving[::CONTROL_STEP]:
                stats["r2_control"] += 1
                stats["r2_control_hits"] += any(t + 1 <= x <= t + 1 + width_r2 + 2 * SLACK for x in brake)

        # the rule log: where instances are lost
        for e in json.loads((dataset / "meta" / "firings" / f"{vid}.json").read_text()):
            key = e["rule"] + (":ego" if e["agent_is_ego"] else ":other")
            c = log[key]
            c["triggers"] += 1
            c["fired"] += bool(e["fired"])
            c["executed"] += bool(e.get("executed"))
            seen = e["body_visible"] and e["agent_visible"]
            if e.get("executed") and seen:
                c["executed_and_seen"] += 1
                agent = "ego" if e["agent_is_ego"] else f"{vid}-{e['agent']}"
                head = av_brake if agent == "ego" else onsets(lf.get(agent, {}).get("Brake", []), first.get(agent, 0))
                # in the lag window after the body onset, as the label test counts it: a head onset just
                # before the firing belongs to another cause (e.g. a red light) but is still in the labels
                lag = rules.R1["lag_s"] if e["rule"] == "R1" else rules.R2["lag_s"]
                in_window = _hit(head, e["frame_body"], lag)
                c["head_onset_seen"] += in_window
                c["head_onset_before_firing"] += in_window and not any(
                    e["frame_fire"] - 2 <= x <= e["frame_fire"] + 10 for x in head)

    report = {}
    for r in ("r1", "r2"):
        p = stats[f"{r}_hits"] / max(1, stats[f"{r}_events"])
        base = stats[f"{r}_control_hits"] / max(1, stats[f"{r}_control"])
        lift = p / base if base else float("inf")
        report[r.upper()] = {"events": stats[f"{r}_events"], "p_head_after_body": round(p, 3),
                             "p_head_control": round(base, 3), "lift": round(lift, 2),
                             "ok": bool(lift >= MIN_LIFT and p >= MIN_P)}
    report["R1"]["what"] = "ego: visible Ped:Wait2X run starts while AV-Mov -> AV-Brake onset"
    report["R2"]["what"] = "vehicle: HazLit onset -> its own Brake onset"
    report["rule_log"] = {k: dict(v) for k, v in sorted(log.items())}
    (dataset / "meta" / "rule_check.json").write_text(json.dumps(report, indent=1))
    return report


def print_report(report):
    print("[Rule check] P(head in the lag window | body onset) vs control moments")
    for r in ("R1", "R2"):
        x = report[r]
        print(f"  {r} {'ok  ' if x['ok'] else 'WEAK'} lift {x['lift']:5.2f}  p {x['p_head_after_body']:.2f} "
              f"vs {x['p_head_control']:.2f}  ({x['events']} events)  {x['what']}")
    for k, c in report["rule_log"].items():
        print(f"  log {k:9s} " + ", ".join(f"{n} {c.get(n, 0)}" for n in
                                           ("triggers", "fired", "executed", "executed_and_seen", "head_onset_seen",
                                            "head_onset_before_firing")))


def main():
    parser = argparse.ArgumentParser(description="Check that the injected rules are in the labels")
    parser.add_argument("--dataset", required=True)
    args, _unknown = parser.parse_known_args()             # the docker script adds --device / --machine
    print_report(check(args.dataset))


if __name__ == "__main__":
    main()
