"""Generates a SUMO synthetic dataset in the roadpp layout: one simulation per video (its own traffic,
drivers, signal offset and ego route), 20 s of the ego as the video, ending shortly after it enters the
junction (`simulate.clip_window`).

    <data root>/synthetic/<name>/
        gt/label.json                   the roadpp label.json (same vocabulary)
        gt/{vid}_gt.json                ROAD++ ground truth (`labels.build_video`)
        meta/gt_rules.json              injected rules (exact) and SUMO's own rules (approximate)
        meta/firings/{vid}.json         every rule trigger of the clip: time, frames, fired, executed, visible
        meta/world/{vid}.json.gz        world state per frame (oracle, debugging)
        meta/generation.json            settings, counts per video, summary
        meta/rule_check.json            the rules' lift in the labels (`check_rules`)
        meta/maps.json                  multi-map datasets: {"maps": {map id: spec}, "videos": {vid: map id}}
        meta/map_split.json             multi-map datasets: the videos split by map (unseen test maps),
                                        for step_dataset "split_from": "meta/map_split.json"
        sumo/                           network(s) and route files (reproducible inputs)

--maps 0: the one junction of sumo_junction_v0..v3; --maps K: K random junctions (`scenario.random_spec`),
video i on map i % K.

Server: ./run_roadpp_docker.sh --gpu none --sumo_gen -- --name sumo_maps_v6 --maps 40 --num 500 --workers 16
Then the pipeline, if the rule check at the end says ok: ./run_roadpp_docker.sh --gpu N -- --exp sumo_maps_v6
(stratified split over all maps) and --exp sumo_maps_v6_unseen (test on unseen maps)."""
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
VERSION = "sumo_v7"             # v2: R1 on the driver's camera view; v3: p = 1, no refractory;
                                # v4: E1 rule set (R1 crossing pedestrian, R2, R3 lead brake; random depart lane)
                                # v5: multi-map (--maps), rules as v4
                                # v6: the ego's signal head past the far side of the junction (in view while waiting)
                                # v7: the clip ends 1.5-5 s after the ego enters the junction (`simulate.clip_window`),
                                #     so it holds the wait at a red light and the change to green


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
    i, out, net_file, spec, map_id, labels_json, seed = job
    vid = f"sumo_{i:05d}"
    gt_file = out / "gt" / f"{vid}_gt.json"
    if gt_file.exists():
        return {"vid": vid, "status": "exists"}
    rng = random.Random(seed)
    route_file = out / "sumo" / f"{vid}.rou.xml"
    info = traffic.write_routes(route_file, rng, spec=spec)
    sim = simulate.simulate(net_file, route_file, seed, rng, info["ego_depart"])
    frames = sim["frames"]
    num_frames = min(simulate.CLIP_FRAMES, len(frames))
    if num_frames < MIN_FRAMES:
        return {"vid": vid, "status": "short", "frames": num_frames}
    video, visible = labels.build_video(vid, frames, labels_json, num_frames)
    firings = _clip_firings(sim["firings"], frames, num_frames, visible)
    (out / "meta" / "firings" / f"{vid}.json").write_text(json.dumps(firings))
    with gzip.open(out / "meta" / "world" / f"{vid}.json.gz", "wt") as f:
        json.dump(frames, f)
    tmp = gt_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(video))
    tmp.rename(gt_file)                         # last: a video exists only when complete
    agents_per_frame = sum(len(fr["annos"]) for fr in video["data"]["frames"].values()) / num_frames
    return {"vid": vid, "status": "ok", "seed": seed, "map": map_id, "frames": num_frames, "ego_route": info["ego_route"],
            "clip": sim["clip"],
            "agents_per_frame": round(agents_per_frame, 2), "onsets": dict(_onsets(video, labels_json)),
            "firings": Counter(f"{f['rule']}:{'executed' if f.get('executed') else 'fired' if f['fired'] else 'not_fired'}"
                               + (":visible" if f["body_visible"] and f["agent_visible"] else "") for f in firings)}


def summary(results):
    ok = [r for r in results if r["status"] == "ok"]
    onsets, firings = Counter(), Counter()
    for r in ok:
        onsets.update(r["onsets"])
        firings.update(r["firings"])
    return {"videos": len(ok), "dropped": Counter(r["status"] for r in results if r["status"] != "ok"),
            "mean_frames": round(sum(r["frames"] for r in ok) / max(1, len(ok)), 1),
            "mean_agents_per_frame": round(sum(r["agents_per_frame"] for r in ok) / max(1, len(ok)), 2),
            "clip": {"entered_junction": sum(r.get("clip", {}).get("enter_s") is not None for r in ok),
                     "mean_start_s": round(sum(r.get("clip", {}).get("start_s", 0) for r in ok) / max(1, len(ok)), 1)},
            "onsets": dict(onsets.most_common()), "firings": dict(sorted(firings.items()))}


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


def _merge_earlier(out, results):
    """The per-video results of an earlier run of the same version for the videos this run skipped."""
    gen_file = out / "meta" / "generation.json"
    if not gen_file.exists():
        return results
    earlier = {r["vid"]: r for r in json.loads(gen_file.read_text()).get("videos", []) if r.get("status") == "ok"}
    return [earlier.get(r["vid"], r) if r["status"] == "exists" else r for r in results]


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
    parser.add_argument("--name", default="sumo_maps_v6")
    parser.add_argument("--maps", type=int, default=40, help="random junctions; 0: the v0-v3 junction")
    parser.add_argument("--num", type=int, default=500, help="simulations (videos)")
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
    for d in ("gt", "sumo", "meta/firings", "meta/world"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / "meta" / "version.txt").write_text(VERSION)
    if not (out / "gt" / "label.json").exists():
        shutil.copy(label_src, out / "gt" / "label.json")
    labels_json = json.loads((out / "gt" / "label.json").read_text())
    (out / "meta" / "gt_rules.json").write_text(json.dumps(
        {"version": VERSION, "rules": rules.GT_RULES, "natural_rules": rules.NATURAL_RULES}, indent=2))
    if args.maps:
        maps = scenario.build_maps(out / "sumo", args.maps, args.seed)
        assign = {f"sumo_{i:05d}": f"map_{i % args.maps:03d}" for i in range(args.num)}
        (out / "meta" / "maps.json").write_text(json.dumps({"maps": maps, "videos": assign}, indent=1))
        jobs = [(i, out, out / "sumo" / maps[m]["net"], maps[m]["spec"], m, labels_json, args.seed * 1_000_003 + i)
                for i, m in enumerate(assign.values())]
        kinds = Counter(maps[m]["spec"]["control"] + f"/{len(maps[m]['spec']['arms'])}arm" for m in maps)
        print(f"[SUMO] {args.maps} maps: {dict(kinds)}")
    else:
        net_file = scenario.build_network(out / "sumo")
        jobs = [(i, out, net_file, None, None, labels_json, args.seed * 1_000_003 + i) for i in range(args.num)]
    print(f"[SUMO] {args.num} simulations -> {out}")

    start, results = time.time(), []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(generate_one, jobs):
            results.append(r)
            if len(results) % 20 == 0 or len(results) == len(jobs):
                print(f"[SUMO] {len(results)}/{len(jobs)} ({time.time() - start:.0f} s)")
    results = _merge_earlier(out, results)
    results.sort(key=lambda r: r["vid"])
    report = summary(results)
    gen_file = out / "meta" / "generation.json"
    gen = {"version": VERSION, "args": {k: v for k, v in vars(args).items() if k != "device"},
           "settings": {"traffic": traffic.DEFAULTS, "maps": args.maps,
                        "map_family": {"p_three_arms": scenario.P_THREE_ARMS, "angle_jitter": scenario.ANGLE_JITTER,
                                       "lanes": scenario.LANES, "speeds": scenario.SPEEDS,
                                       "approach_s": scenario.APPROACH_S, "controls": scenario.CONTROLS}, "R1": rules.R1, "R2": rules.R2, "R3": rules.R3,
                        "clip_frames": simulate.CLIP_FRAMES, "dt": simulate.DT},
           "summary": report, "videos": results}
    gen_file.write_text(json.dumps(gen, indent=1, default=list))
    if args.maps:
        _write_map_split(out, maps, assign)
    print(json.dumps(report, indent=1, default=list))
    print(f"[SUMO] done: {gen_file}")
    check_rules.print_report(check_rules.check(out))


if __name__ == "__main__":
    main()
