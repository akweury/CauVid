"""Generates a SUMO synthetic dataset in the roadpp layout: one simulation per video (its own traffic,
drivers, signal offset and ego route), the ego's first 20 s as the video.

    <data root>/synthetic/<name>/
        gt/label.json                   the roadpp label.json (same vocabulary)
        gt/{vid}_gt.json                ROAD++ ground truth (`labels.build_video`)
        meta/gt_rules.json              injected rules (exact) and SUMO's own rules (approximate)
        meta/firings/{vid}.json         every rule trigger of the clip: time, frames, fired, executed, visible
        meta/world/{vid}.json.gz        world state per frame (oracle, debugging)
        meta/generation.json            settings, counts per video, summary
        meta/rule_check.json            the rules' lift in the labels (`check_rules`)
        sumo/                           network and route files (reproducible inputs)

Server: ./run_roadpp_docker.sh --gpu none --sumo_gen -- --name sumo_junction_v2 --num 500 --workers 16
Then the pipeline, if the rule check at the end says ok: ./run_roadpp_docker.sh --gpu N -- --exp sumo_junction_v2"""
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
VERSION = "sumo_v3"             # v2: R1 on the driver's camera view; v3: p = 1, no R1 refractory, Mov in the junction


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
    i, out, net_file, labels_json, seed = job
    vid = f"sumo_{i:05d}"
    gt_file = out / "gt" / f"{vid}_gt.json"
    if gt_file.exists():
        return {"vid": vid, "status": "exists"}
    rng = random.Random(seed)
    route_file = out / "sumo" / f"{vid}.rou.xml"
    info = traffic.write_routes(route_file, rng)
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
    return {"vid": vid, "status": "ok", "seed": seed, "frames": num_frames, "ego_route": info["ego_route"],
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
            "onsets": dict(onsets.most_common()), "firings": dict(sorted(firings.items()))}


def main():
    from src.exp_roadpp import config
    parser = argparse.ArgumentParser(description="SUMO synthetic dataset in the roadpp layout")
    parser.add_argument("--name", default="sumo_junction_v2")
    parser.add_argument("--num", type=int, default=500, help="simulations (videos)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default=None, help="dataset folder (default: <data root>/synthetic/<name>)")
    parser.add_argument("--label_json", default=None, help="default: the roadpp gt/label.json of the machine")
    parser.add_argument("--machine", default="dgx-jing")
    parser.add_argument("--device", default="cpu", help="unused (docker script)")
    args = parser.parse_args()

    out = Path(args.out) if args.out else config.synthetic_dataset_path(args.machine, args.name)
    label_src = Path(args.label_json) if args.label_json else config.roadpp_dataset_path(args.machine) / "gt" / "label.json"
    for d in ("gt", "sumo", "meta/firings", "meta/world"):
        (out / d).mkdir(parents=True, exist_ok=True)
    if not (out / "gt" / "label.json").exists():
        shutil.copy(label_src, out / "gt" / "label.json")
    labels_json = json.loads((out / "gt" / "label.json").read_text())
    (out / "meta" / "gt_rules.json").write_text(json.dumps(
        {"version": VERSION, "rules": rules.GT_RULES, "approx_rules": rules.APPROX_RULES}, indent=2))
    net_file = scenario.build_network(out / "sumo")
    print(f"[SUMO] {args.num} simulations -> {out}")

    jobs = [(i, out, net_file, labels_json, args.seed * 1_000_003 + i) for i in range(args.num)]
    start, results = time.time(), []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(generate_one, jobs):
            results.append(r)
            if len(results) % 20 == 0 or len(results) == len(jobs):
                print(f"[SUMO] {len(results)}/{len(jobs)} ({time.time() - start:.0f} s)")
    results.sort(key=lambda r: r["vid"])
    report = summary(results)
    gen_file = out / "meta" / "generation.json"
    gen = {"version": VERSION, "args": {k: v for k, v in vars(args).items() if k != "device"},
           "settings": {"traffic": traffic.DEFAULTS, "R1": rules.R1, "R2": rules.R2,
                        "clip_frames": simulate.CLIP_FRAMES, "dt": simulate.DT},
           "summary": report, "videos": results}
    gen_file.write_text(json.dumps(gen, indent=1, default=list))
    print(json.dumps(report, indent=1, default=list))
    print(f"[SUMO] done: {gen_file}")
    check_rules.print_report(check_rules.check(out))


if __name__ == "__main__":
    main()
