"""Are the rules in the labels? For each rule, from the generated ground truth only (what the pipeline
sees), with the onsets as `step_dataset.head_onsets` counts them (a label run starting more than MIN_OFF
frames after the previous one, not on the agent's first frame):

    P(head onset in the rule's window | body onset)  vs  P(head onset in a window as long | control moment)

The injected rules (R1-R3) must pass (lift >= MIN_LIFT and p >= MIN_P); a rule that does not cannot be
learned from the labels (sumo_junction_v0: R1 lift 1.0). The natural SUMO rules (N1-N3) are reported. The
rule log adds why instances of the injected rules are lost (not fired, not executed, not seen, no head).

    python -m src.exp_roadpp.synth.sumo.check_rules --dataset <dataset dir>
    ./run_roadpp_docker.sh --gpu none --sumo_check -- --dataset /storage-01/ml-jsha/CauVid_Data/synthetic/<name>
Writes <dataset>/meta/rule_check.json; `run.py` runs it after the generation."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.exp_roadpp.synth.sumo import rules

MIN_OFF = 10                # frames, as step_dataset.MIN_OFF_FRAMES
SLACK = 5                   # frames added on both sides of a lag window
CONTROL_STEP = 10           # frames between control moments
MIN_LIFT, MIN_P = 2.0, 0.4  # an injected rule passes above both
MIN_LIFT_OF = {"R4": 1.5}   # an order rule between two turns: its control is ~0.5 (left or right), lift <= ~2
TURN_MERGE = 50             # frames: turn onsets of the same kind closer than this are one turn (flicker)
VEHICLES = ("Car", "MedVeh", "LarVeh", "Bus", "Mobike", "Cyc")
CROSSING = ("Xing", "XingFmLft", "XingFmRht")


def _lag_window(spec):
    return int(spec["lag_s"][0] * 10) - SLACK, int(spec["lag_s"][1] * 10) + SLACK


# rule: (what is tested, window after the body onset in frames, injected)
TESTS = {
    "R1": ("ego: a pedestrian starts crossing in view while AV-Mov -> AV-Brake", _lag_window(rules.R1), True),
    "R2": ("vehicle: HazLit onset -> its own Brake", _lag_window(rules.R2), True),
    "R3": ("ego: the vehicle ahead in VehLane starts braking while AV-Mov -> AV-Brake", _lag_window(rules.R3), True),
    "N1": ("ego: the vehicle ahead in VehLane starts (MovAway) while AV-Stop -> AV-Mov", (5, 30), False),
    "N2": ("ego and vehicles: MovRht -> TurRht within 10 s", (1, 100), False),
    "N3": ("ego: its TL turns Red in view while AV-Mov -> AV-Brake within 8 s", (1, 80), False),
    "N4": ("ego: its TL turns Green in view while AV-Stop -> AV-Mov within 5 s", (1, 50), False),
    # grids (long videos): the window is unused, the test is the next turn label (`_r4`)
    "R4": ("ego: AV-TurRht -> the next turn label is AV-TurLft (control: the turn after an AV-TurLft)", (1, 600), True),
}


def label_frames(video, labels):
    """{agent: {label: sorted frames}} of one ROAD++ video (agents by tube_uid, the ego as "ego"; the
    class as "cls:<name>", locations as "loc:<name>"), and {agent: its first frame}."""
    out = defaultdict(lambda: defaultdict(list))
    first = {"ego": 1}
    for f, fr in video["frames"].items():
        f = int(f)
        for k in fr.get("av_action_ids") or []:
            out["ego"][labels["av_action_labels"][k]].append(f)
        for a in fr["annos"].values():
            uid = a["tube_uid"]
            first[uid] = min(first.get(uid, f), f)
            out[uid]["cls:" + labels["agent_labels"][a["agent_ids"][0]]].append(f)
            for k in a["action_ids"]:
                out[uid][labels["action_labels"][k]].append(f)
            for k in a["loc_ids"]:
                out[uid]["loc:" + labels["loc_labels"][k]].append(f)
    for per in out.values():
        for v in per.values():
            v.sort()
    return out, first


def onsets(frames, first):
    """Onset frames of a sorted frame list (runs more than MIN_OFF frames apart, not on the first frame)."""
    out, last = [], None
    for f in frames:
        if (last is None or f - last > MIN_OFF) and f > first:    # a shorter gap: the same run
            out.append(f)
        last = f
    return out


def run_starts(frames):
    """First frames of the runs of a sorted frame list (gaps up to MIN_OFF frames bridged)."""
    out, last = [], None
    for f in frames:
        if last is None or f - last > MIN_OFF:
            out.append(f)
        last = f
    return out


def _hit(head_onsets, start, window):
    return any(start + window[0] <= x <= start + window[1] for x in head_onsets)


def _event(stats, rule, hit):
    stats[rule]["events"] += 1
    stats[rule]["hits"] += bool(hit)


def _control(stats, rule, moments, head_onsets):
    width = TESTS[rule][1][1] - TESTS[rule][1][0]
    for t in moments[::CONTROL_STEP]:
        stats[rule]["control"] += 1
        stats[rule]["control_hits"] += _hit(head_onsets, t, (1, 1 + width))


def _r4(av, stats):
    """R4 on the ego's turn sequence: after a right turn the next turn is left; control: the turn after a
    left turn (free: left or right). Only pairs of turns inside the video count."""
    turns = []
    for t, kind in sorted([(t, "R") for t in av["AV-TurRht"]] + [(t, "L") for t in av["AV-TurLft"]]):
        if turns and turns[-1][1] == kind and t - turns[-1][0] < TURN_MERGE:
            continue
        turns.append((t, kind))
    for (_t0, k0), (_t1, k1) in zip(turns, turns[1:]):
        if k0 == "R":
            _event(stats, "R4", k1 == "L")
        else:
            stats["R4"]["control"] += 1
            stats["R4"]["control_hits"] += k1 == "L"


def check_video(lf, first, stats):
    win = {r: TESTS[r][1] for r in TESTS}
    ego = lf["ego"]
    mov, stop = set(ego.get("AV-Mov", [])), set(ego.get("AV-Stop", []))
    av = {k: onsets(ego.get(k, []), 1) for k in ("AV-Brake", "AV-Mov", "AV-TurRht", "AV-TurLft", "AV-MovRht")}
    _r4(av, stats)
    for r in ("R1", "R3", "N3"):
        _control(stats, r, sorted(mov), av["AV-Brake"])
    _control(stats, "N1", sorted(stop), av["AV-Mov"])
    _control(stats, "N4", sorted(stop), av["AV-Mov"])
    _control(stats, "N2", sorted(mov), av["AV-TurRht"])
    for t in av["AV-MovRht"]:
        _event(stats, "N2", _hit(av["AV-TurRht"], t, win["N2"]))

    for uid, per in lf.items():
        if uid == "ego":
            continue
        # R1: a pedestrian's crossing starts in view (its first frame there while the ego is AV-Mov)
        if per.get("cls:Ped"):
            crossing = sorted(set().union(*(per.get(x, []) for x in CROSSING)))
            for s in run_starts(crossing):
                start = next((f for f in crossing if s <= f <= s + 50 and f in mov), None)
                if start is not None:
                    _event(stats, "R1", _hit(av["AV-Brake"], start, win["R1"]))
        # N3: the ego's signal turns Red in view while the ego moves; N4: Green while the ego stands
        if per.get("cls:TL"):
            for s in run_starts(per.get("Red", [])):
                if s in mov:
                    _event(stats, "N3", _hit(av["AV-Brake"], s, win["N3"]))
            for s in run_starts(per.get("Green", [])):
                if s > first[uid] and s in stop:              # a change in view, not the light coming into view
                    _event(stats, "N4", _hit(av["AV-Mov"], s, win["N4"]))
        if not any(per.get("cls:" + c) for c in VEHICLES):
            continue
        vehlane = set(per.get("loc:VehLane", []))
        brake = onsets(per.get("Brake", []), first[uid])
        # R2: HazLit onset -> its own Brake; control: moments the vehicle moves
        for e in onsets(per.get("HazLit", []), first[uid]):
            _event(stats, "R2", _hit(brake, e, win["R2"]))
        _control(stats, "R2", sorted(set(per.get("MovAway", [])) | set(per.get("MovTow", []))), brake)
        # R3: the vehicle ahead in VehLane starts braking while the ego is AV-Mov
        for t in brake:
            if t in vehlane and t in mov:
                _event(stats, "R3", _hit(av["AV-Brake"], t, win["R3"]))
        # N1: the vehicle ahead in VehLane starts while the ego stands
        for t in onsets(per.get("MovAway", []), first[uid]):
            if t in vehlane and t in stop:
                _event(stats, "N1", _hit(av["AV-Mov"], t, win["N1"]))
        # N2 for the other vehicles
        tur = onsets(per.get("TurRht", []), first[uid])
        for t in onsets(per.get("MovRht", []), first[uid]):
            _event(stats, "N2", _hit(tur, t, win["N2"]))


def video_events(data, labels):
    """{rule: number of body events} of one ROAD++ video (its "data"): the instances of each tested rule
    that the labels hold, hit or not (`run.py` samples videos until each rule has enough)."""
    lf, first = label_frames(data, labels)
    stats = defaultdict(Counter)
    check_video(lf, first, stats)
    return {r: stats[r]["events"] for r in TESTS}


def check(dataset):
    dataset = Path(dataset)
    labels = json.loads((dataset / "gt" / "label.json").read_text())
    stats = defaultdict(Counter)
    log = defaultdict(Counter)
    for gt in sorted((dataset / "gt").glob("*_gt.json")):
        vid = gt.name[:-len("_gt.json")]
        lf, first = label_frames(json.loads(gt.read_text())["data"], labels)
        check_video(lf, first, stats)
        av_brake = onsets(lf["ego"].get("AV-Brake", []), 1)
        # the rule log: where instances of the injected rules are lost
        for e in json.loads((dataset / "meta" / "firings" / f"{vid}.json").read_text()):
            c = log[e["rule"] + (":ego" if e["agent_is_ego"] else ":other")]
            c["triggers"] += 1
            c["fired"] += bool(e["fired"])
            c["executed"] += bool(e.get("executed"))
            if e.get("executed") and e["body_visible"] and e["agent_visible"]:
                c["executed_and_seen"] += 1
                agent = "ego" if e["agent_is_ego"] else f"{vid}-{e['agent']}"
                head = av_brake if agent == "ego" else onsets(lf.get(agent, {}).get("Brake", []), first.get(agent, 0))
                c["head_onset_seen"] += _hit(head, e["frame_body"], TESTS[e["rule"]][1])

    report = {}
    for r, (what, _window, injected) in TESTS.items():
        s = stats[r]
        p = s["hits"] / max(1, s["events"])
        base = s["control_hits"] / max(1, s["control"])
        lift = p / base if base else float("inf")
        report[r] = {"what": what, "injected": injected, "events": s["events"], "p_head_after_body": round(p, 3),
                     "p_head_control": round(base, 3), "lift": round(lift, 2),
                     "ok": bool(lift >= MIN_LIFT_OF.get(r, MIN_LIFT) and p >= MIN_P)
                     if injected and s["events"] else None}
    report["rule_log"] = {k: dict(v) for k, v in sorted(log.items())}
    (dataset / "meta" / "rule_check.json").write_text(json.dumps(report, indent=1))
    return report


def print_report(report):
    print("[Rule check] P(head in the rule's window | body onset) vs control moments")
    for r in TESTS:
        x = report[r]
        flag = "n/a " if x["injected"] and not x["events"] else ("ok  " if x["ok"] else "WEAK") if x["injected"] else "info"
        print(f"  {r} {flag} lift {x['lift']:5.2f}  p {x['p_head_after_body']:.2f} vs {x['p_head_control']:.2f}"
              f"  ({x['events']} events)  {x['what']}")
    for k, c in report["rule_log"].items():
        print(f"  log {k:9s} " + ", ".join(f"{n} {c.get(n, 0)}" for n in
                                           ("triggers", "fired", "executed", "executed_and_seen", "head_onset_seen")))


def main():
    parser = argparse.ArgumentParser(description="Check that the rules are in the labels")
    parser.add_argument("--dataset", required=True)
    args, _unknown = parser.parse_known_args()             # the docker script adds --device / --machine
    print_report(check(args.dataset))


if __name__ == "__main__":
    main()
