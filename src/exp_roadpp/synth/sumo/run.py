"""Generates a SUMO synthetic dataset in the roadpp layout: one simulation per video (its own traffic,
drivers, signal offset and ego route), 20 s of the ego as the video, cut by a clip mode chosen per video
(`simulate.CLIP_MODES`: from the departure, up to the junction, or anywhere in between).

Videos are sampled in rounds of --batch until every event of --targets has enough instances (rule tests
of `check_rules`, e.g. R1 = a pedestrian starts crossing while the ego moves, or head onsets such as
av:AV-TurLft), at most --num. Each round weights the clip modes by how many of the still missing
instances they yield per video (`clip_weights`), every mode keeping at least MIN_SHARE: N3 (a light
turning red while the ego drives) comes from depart clips, N4 (green while the ego waits) from junction
clips, and the random clips keep the clip ends from being tied to events.

    <data root>/synthetic/<name>/
        gt/label.json                   the roadpp label.json (same vocabulary)
        gt/{vid}_gt.json                ROAD++ ground truth (`labels.build_video`)
        meta/gt_rules.json              injected rules (exact) and SUMO's own rules (approximate)
        meta/firings/{vid}.json         every rule trigger of the clip: time, frames, fired, executed, visible
        meta/world/{vid}.json.gz        world state per frame (oracle, debugging)
        meta/generation.json            settings, counts per video, summary (rewritten after every round)
        meta/results/{vid}.json         the video's result (clip mode, events, onsets): a resumed run counts it
        meta/rule_check.json            the rules' lift in the labels (`check_rules`)
        meta/maps.json                  multi-map datasets: {"maps": {map id: spec}, "videos": {vid: map id}}
        meta/map_split.json             multi-map datasets: the videos split by map (unseen test maps),
                                        for step_dataset "split_from": "meta/map_split.json"
        sumo/                           network(s) and route files (reproducible inputs)

--maps 0: the one junction of sumo_junction_v0..v3; --maps K: K random junctions (`scenario.random_spec`),
video i on map i % K.

Server: ./run_roadpp_docker.sh --gpu none --sumo_gen -- --name sumo_maps_v7 --maps 40 --workers 16
Then the pipeline, if the rule check at the end says ok: ./run_roadpp_docker.sh --gpu N -- --exp sumo_maps_v7
(stratified split over all maps) and --exp sumo_maps_v7_unseen (test on unseen maps)."""
import argparse
import gzip
import json
import random
import shutil
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from src.exp_roadpp.synth.sumo import check_rules, labels, rules, scenario, simulate, traffic

MIN_FRAMES = 100            # shorter ego clips (e.g. the ego never departed) are dropped
VERSION = "sumo_v8"             # v2: R1 on the driver's camera view; v3: p = 1, no refractory;
                                # v4: E1 rule set (R1 crossing pedestrian, R2, R3 lead brake; random depart lane)
                                # v5: multi-map (--maps), rules as v4
                                # v6: the ego's signal head past the far side of the junction (in view while waiting)
                                # v7: the clip ends 1.5-5 s after the ego enters the junction (`simulate.clip_window`),
                                #     so it holds the wait at a red light and the change to green
                                # v8: clip modes mixed per video (depart, junction, random); videos are sampled
                                #     until every target event has enough instances (--targets)


def _clip_firings(log, frames, num_frames, visible):
    """The triggers of the rule log inside the clip, with their frames (1-based) and visibility."""
    t0 = frames[0]["t"]
    out = []
    for e in log:
        k_body = int(round((e["t_body"] - t0) / simulate.DT))
        k_fire = int(round((e["t_fire"] - t0) / simulate.DT)) if e.get("t_fire") is not None else None
        if not (0 <= k_body < num_frames or (k_fire is not None and 0 <= k_fire < num_frames)):
            continue
        agent, body = e["agent"], e["body_agent"]
        out.append({**e, "frame_body": k_body + 1, "frame_fire": None if k_fire is None else k_fire + 1,
                    "agent_is_ego": agent == "ego",
                    "body_visible": k_body in visible.get(body, ()),
                    "agent_visible": agent == "ego" or (k_fire is not None and k_fire in visible.get(agent, ()))})
    return out


def _onsets(video, labels_json):
    """Onsets per head "<class>:<label>" of a generated video, as step_dataset counts them."""
    from src.exp_roadpp.step_dataset import head_onsets
    from src.exp_roadpp.step_dataset_language import Language
    from src.exp_roadpp import utils_data
    data = video["data"]
    lang = Language("cpu")
    atoms = lang.video2atoms("av", data["av_action_tubes"], data["frames"])
    atoms += lang.video2atoms("agents", data["agent_tubes"], data["frames"])
    return head_onsets([utils_data.add_labels_to_atom(a, labels_json) for a in atoms])


def generate_one(job):
    i, out, net_file, spec, map_id, labels_json, seed, clip_mode = job
    vid = f"sumo_{i:05d}"
    gt_file = out / "gt" / f"{vid}_gt.json"
    result_file = out / "meta" / "results" / f"{vid}.json"
    if gt_file.exists():
        if result_file.exists():                            # resumed: the video's earlier result
            return {**json.loads(result_file.read_text()), "resumed": True}
        return {"vid": vid, "status": "exists"}
    rng = random.Random(seed)
    route_file = out / "sumo" / f"{vid}.rou.xml"
    info = traffic.write_routes(route_file, rng, spec=spec)
    sim = simulate.simulate(net_file, route_file, seed, rng, info["ego_depart"], clip_mode)
    frames = sim["frames"]
    num_frames = min(simulate.CLIP_FRAMES, len(frames))
    if num_frames < MIN_FRAMES:
        return {"vid": vid, "status": "short", "frames": num_frames, "clip": sim["clip"]}
    video, visible = labels.build_video(vid, frames, labels_json, num_frames)
    firings = _clip_firings(sim["firings"], frames, num_frames, visible)
    (out / "meta" / "firings" / f"{vid}.json").write_text(json.dumps(firings))
    with gzip.open(out / "meta" / "world" / f"{vid}.json.gz", "wt") as f:
        json.dump(frames, f)
    agents_per_frame = sum(len(fr["annos"]) for fr in video["data"]["frames"].values()) / num_frames
    result = {"vid": vid, "status": "ok", "seed": seed, "map": map_id, "frames": num_frames,
              "ego_route": info["ego_route"], "clip": sim["clip"], "agents_per_frame": round(agents_per_frame, 2),
              "events": check_rules.video_events(video["data"], labels_json),
              "onsets": dict(_onsets(video, labels_json)),
              "firings": Counter(f"{f['rule']}:{'executed' if f.get('executed') else 'fired' if f['fired'] else 'not_fired'}"
                                 + (":visible" if f["body_visible"] and f["agent_visible"] else "") for f in firings)}
    result_file.write_text(json.dumps(result))
    tmp = gt_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(video))
    tmp.rename(gt_file)                         # last: a video exists only when complete
    return result


def summary(results):
    ok = [r for r in results if r["status"] == "ok"]
    onsets, firings = Counter(), Counter()
    for r in ok:
        onsets.update(r["onsets"])
        firings.update(r["firings"])
    return {"videos": len(ok), "dropped": Counter(r["status"] for r in results if r["status"] != "ok"),
            "mean_frames": round(sum(r["frames"] for r in ok) / max(1, len(ok)), 1),
            "mean_agents_per_frame": round(sum(r["agents_per_frame"] for r in ok) / max(1, len(ok)), 2),
            "clip": {"modes": Counter(r.get("clip", {}).get("mode") for r in ok),
                     "entered_junction": sum(r.get("clip", {}).get("enter_s") is not None for r in ok),
                     "mean_start_s": round(sum(r.get("clip", {}).get("start_s", 0) for r in ok) / max(1, len(ok)), 1)},
            "events": dict(event_counts(ok)), "events_by_mode": _events_by_mode(ok),
            "onsets": dict(onsets.most_common()), "firings": dict(sorted(firings.items()))}


# --- sampling until every target event has enough instances (--targets) ---
DEFAULT_TARGETS = "R1=400,R2=400,R3=400,N1=200,N3=150,N4=200"
BATCH = 64                  # videos per sampling round (then the counts and the clip mode weights are updated)
MIN_SHARE = 0.15            # every clip mode keeps this share of a round (no clip type disappears)
MIN_RATE_VIDEOS = 10        # videos of a mode before its event rates steer the weights


def parse_targets(text):
    """"R1=400,av:AV-TurLft=300" -> {event: count}; events are rule tests (`check_rules.TESTS`) or heads
    "<class>:<label>" (onsets as step_dataset counts them)."""
    out = {}
    for item in filter(None, (x.strip() for x in (text or "").split(","))):
        name, _, n = item.rpartition("=")
        if not name or not n.isdigit():
            raise ValueError(f"--targets item {item!r}: expected <event>=<count>")
        out[name] = int(n)
    return out


def _count(r, event):
    return (r.get("events") or {}).get(event, 0) or (r.get("onsets") or {}).get(event, 0)


def event_counts(ok, events=None):
    """Counter {event: instances} over the ok results, for the rule tests and the head onsets (or events)."""
    c = Counter()
    for r in ok:
        for e in (events if events is not None else list(r.get("events") or {}) + list(r.get("onsets") or {})):
            c[e] += _count(r, e)
    return c


def _events_by_mode(ok):
    return {m: {"videos": sum(1 for r in ok if r.get("clip", {}).get("mode") == m),
                **{e: n for e, n in sorted(event_counts([r for r in ok if r.get("clip", {}).get("mode") == m],
                                                        check_rules.TESTS).items())}}
            for m in simulate.CLIP_MODES}


def clip_weights(ok, targets, counts):
    """Weights of simulate.CLIP_MODES for the next round: each mode by the share of the missing instances
    it yields per video (from the videos made so far), at least MIN_SHARE each; equal until every mode has
    MIN_RATE_VIDEOS videos."""
    modes = simulate.CLIP_MODES
    need = {e: t - counts.get(e, 0) for e, t in targets.items() if counts.get(e, 0) < t}
    by_mode = {m: [r for r in ok if r.get("clip", {}).get("mode") == m] for m in modes}
    if not need or any(len(v) < MIN_RATE_VIDEOS for v in by_mode.values()):
        return [1.0 / len(modes)] * len(modes)
    score = {m: sum(min(sum(_count(r, e) for r in rs) / len(rs), need[e]) / targets[e] for e in need)
             for m, rs in by_mode.items()}
    total = sum(score.values())
    if total <= 0:
        return [1.0 / len(modes)] * len(modes)
    return [MIN_SHARE + (1 - len(modes) * MIN_SHARE) * score[m] / total for m in modes]


def _check_version(out):
    """A dataset folder is only resumed (videos that exist are skipped) by the generator version that made
    it: sumo_maps_v4 was made by sumo_v5, then "resumed" by sumo_v6, which skipped all 492 videos and only
    rewrote the metadata as sumo_v6. meta/version.txt holds the version of the videos."""
    if not (out / "gt").exists() or not any((out / "gt").glob("sumo_*_gt.json")):
        return                                      # a new dataset
    version_file = out / "meta" / "version.txt"
    made_by = version_file.read_text().strip() if version_file.exists() else "unknown (before sumo_v6)"
    if made_by != VERSION:
        raise SystemExit(f"[SUMO] {out} has videos of generator {made_by}, this is {VERSION}: generate into a new "
                         f"dataset (--name), or delete the folder first.")


def _earlier_results(out):
    """{vid: result} of an earlier run of the same version (meta/generation.json), for videos that exist
    but have no meta/results/<vid>.json."""
    gen_file = out / "meta" / "generation.json"
    if not gen_file.exists():
        return {}
    return {r["vid"]: r for r in json.loads(gen_file.read_text()).get("videos", []) if r.get("status") == "ok"}


def _write_generation(gen_file, args, targets, results, stop):
    """meta/generation.json (rewritten after every round); returns the summary."""
    report = {"stop": stop, "targets": targets, **summary(results)}
    gen = {"version": VERSION, "args": {k: v for k, v in vars(args).items() if k != "device"},
           "settings": {"traffic": traffic.DEFAULTS, "maps": args.maps,
                        "map_family": {"p_three_arms": scenario.P_THREE_ARMS, "angle_jitter": scenario.ANGLE_JITTER,
                                       "lanes": scenario.LANES, "speeds": scenario.SPEEDS,
                                       "approach_s": scenario.APPROACH_S, "controls": scenario.CONTROLS},
                        "R1": rules.R1, "R2": rules.R2, "R3": rules.R3,
                        "clip_frames": simulate.CLIP_FRAMES, "dt": simulate.DT,
                        "clip_modes": simulate.CLIP_MODES, "post_junction_s": simulate.POST_JUNCTION_S,
                        "sampling": {"batch": args.batch, "min_share": MIN_SHARE, "min_rate_videos": MIN_RATE_VIDEOS}},
           "summary": report, "videos": sorted(results, key=lambda r: r["vid"])}
    gen_file.write_text(json.dumps(gen, indent=1, default=list))
    return report


def _write_map_split(out, maps, assign):
    """meta/map_split.json: the generated videos (also those of an earlier, resumed run) split by map
    (`scenario.map_split`), in the layout of step_dataset's split.json."""
    by_map = scenario.map_split(maps)
    split_of = {m: sp for sp, ms in by_map.items() for m in ms}
    splits = {"train": [], "val": [], "test": []}
    for vid, m in assign.items():
        if (out / "gt" / f"{vid}_gt.json").exists():
            splits[split_of[m]].append(vid)
    if len(splits["val"]) < len(splits["test"]):
        # step07 chooses the thresholds on val and needs val >= test videos (maps have 12 or 13 videos,
        # minus the dropped ones); val and test maps are dealt symmetrically, so they may swap
        by_map["val"], by_map["test"] = by_map["test"], by_map["val"]
        splits["val"], splits["test"] = splits["test"], splits["val"]
    (out / "meta" / "map_split.json").write_text(json.dumps({"maps": by_map, "splits": splits}, indent=1))
    print("[SUMO] map split: " + ", ".join(f"{sp} {len(by_map[sp])} maps / {len(v)} videos" for sp, v in splits.items()))


def main():
    from src.exp_roadpp import config
    parser = argparse.ArgumentParser(description="SUMO synthetic dataset in the roadpp layout")
    parser.add_argument("--name", default="sumo_maps_v7")
    parser.add_argument("--maps", type=int, default=40, help="random junctions; 0: the v0-v3 junction")
    parser.add_argument("--num", type=int, default=3000, help="simulations at most (with --targets: stop earlier "
                        "once every target is reached)")
    parser.add_argument("--targets", default=DEFAULT_TARGETS,
                        help="instances per event to reach, e.g. R1=400,N4=200,av:AV-TurLft=300 (rule tests of "
                             "check_rules or heads); empty: exactly --num simulations")
    parser.add_argument("--batch", type=int, default=BATCH, help="simulations per sampling round")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=None, help="dataset folder (default: <data root>/synthetic/<name>)")
    parser.add_argument("--label_json", default=None, help="default: the roadpp gt/label.json of the machine")
    parser.add_argument("--machine", default="dgx-jing")
    parser.add_argument("--device", default="cpu", help="unused (docker script)")
    parser.add_argument("--split_only", action="store_true",
                        help="only rewrite meta/map_split.json of an existing multi-map dataset (no simulation)")
    args = parser.parse_args()

    out = Path(args.out) if args.out else config.synthetic_dataset_path(args.machine, args.name)
    if args.split_only:
        index = json.loads((out / "meta" / "maps.json").read_text())
        _write_map_split(out, index["maps"], index["videos"])
        return
    label_src = Path(args.label_json) if args.label_json else config.roadpp_dataset_path(args.machine) / "gt" / "label.json"
    _check_version(out)
    for d in ("gt", "sumo", "meta/firings", "meta/world", "meta/results"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / "meta" / "version.txt").write_text(VERSION)
    if not (out / "gt" / "label.json").exists():
        shutil.copy(label_src, out / "gt" / "label.json")
    labels_json = json.loads((out / "gt" / "label.json").read_text())
    (out / "meta" / "gt_rules.json").write_text(json.dumps(
        {"version": VERSION, "rules": rules.GT_RULES, "natural_rules": rules.NATURAL_RULES}, indent=2))
    targets = parse_targets(args.targets)
    if args.maps:
        maps = scenario.build_maps(out / "sumo", args.maps, args.seed)
        kinds = Counter(maps[m]["spec"]["control"] + f"/{len(maps[m]['spec']['arms'])}arm" for m in maps)
        print(f"[SUMO] {args.maps} maps: {dict(kinds)}")
    else:
        net_file = scenario.build_network(out / "sumo")
    print(f"[SUMO] up to {args.num} simulations -> {out}; targets: {targets or 'none (all ' + str(args.num) + ')'}")

    def job(i, mode):
        seed = args.seed * 1_000_003 + i
        if args.maps:
            m = f"map_{i % args.maps:03d}"
            return (i, out, out / "sumo" / maps[m]["net"], maps[m]["spec"], m, labels_json, seed, mode)
        return (i, out, net_file, None, None, labels_json, seed, mode)

    gen_file = out / "meta" / "generation.json"
    earlier = _earlier_results(out)
    start, results, i, rounds = time.time(), [], 0, 0
    with Pool(args.workers) as pool:
        while True:
            ok = [r for r in results if r["status"] == "ok"]
            counts = event_counts(ok, targets)
            missing = {e: t - counts[e] for e, t in targets.items() if counts[e] < t}
            if targets and not missing:
                stop = "targets reached"
                break
            if i >= args.num:
                stop = f"--num {args.num} reached" + (f", missing {missing}" if missing else "")
                break
            weights = clip_weights(ok, targets, counts)
            n = min(args.batch, args.num - i)
            modes = random.Random(args.seed * 7_919 + i).choices(simulate.CLIP_MODES, weights, k=n)
            batch = list(pool.imap_unordered(generate_one, [job(i + k, modes[k]) for k in range(n)]))
            results += [earlier.get(r["vid"], r) if r["status"] == "exists" else r for r in batch]
            i, rounds = i + n, rounds + 1
            ok = [r for r in results if r["status"] == "ok"]
            counts = event_counts(ok, targets)
            print(f"[SUMO] round {rounds}: {i} simulations, {len(ok)} videos ({time.time() - start:.0f} s); "
                  f"modes {dict(zip(simulate.CLIP_MODES, (round(w, 2) for w in weights)))}; "
                  + ", ".join(f"{e} {counts[e]}/{t}" for e, t in targets.items()))
            _write_generation(gen_file, args, targets, results, f"running, round {rounds}")
    results.sort(key=lambda r: r["vid"])
    report = _write_generation(gen_file, args, targets, results, stop)
    print(f"[SUMO] stop: {stop}")
    if args.maps:
        assign = {f"sumo_{k:05d}": f"map_{k % args.maps:03d}" for k in range(i)
                  if (out / "gt" / f"sumo_{k:05d}_gt.json").exists()}
        (out / "meta" / "maps.json").write_text(json.dumps({"maps": maps, "videos": assign}, indent=1))
        _write_map_split(out, maps, assign)
    print(json.dumps(report, indent=1, default=list))
    print(f"[SUMO] done: {gen_file}")
    check_rules.print_report(check_rules.check(out))


if __name__ == "__main__":
    main()
