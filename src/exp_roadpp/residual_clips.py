"""Clips of the ego brakes no found rule explains, for checking by eye (SUMO datasets).

Takes a rule list of `rule_tree --mode list` (its JSON: the root and the rules r1, r2, ... as literal paths)
and, on the videos of one split, every onset event of the head (default av:AV-Brake of the ego). The hit
rows of an event are the at-risk rows of the agent whose onset comes within hit_slots slots (as rule_tree).
An event is
  explained    some hit row has the root and one of the rules holding,
  root_only    the root holds on some hit row, no rule does,
  no_root      the root holds on none of its hit rows.
Prints the counts (the event-level share the rules explain) and renders a random sample of root_only events
(and a few explained ones as controls) with `synth/sumo/render.render`: frames onset - before .. onset + after,
top view and camera view with the GT labels, the title noting the time to the onset. Frames: the event frame is
the tensor frame index, used as the 1-based GT frame (as `rule_events`).

    ./run_roadpp_docker.sh --gpu 0 --residual_clips -- --exp sumo_maps_v7 --machine dgx-jing --roles ahead \\
        --rule_list <output_dir>/rule_tree/av_AV-Brake__av_AV_Mov_own_1_rahead_w2_list.json
    options: --clip_split val --num 12 --controls 3 --seed 0 --before_s 5 --after_s 2
Writes <output_dir>/residual_clips/<split>/<group>_<vid>_f<frame>.mp4 and events.json (every event, its group
and the rules that held).
"""
import json
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from src.exp_roadpp import config, step05, utils_data, utils_pipe
from src.exp_roadpp.cauvid import TemporalRuleNet
from src.exp_roadpp.oracle_junction import route_turn
from src.exp_roadpp.rule_tree import parse_root, vehicle_groups
from src.exp_roadpp.synth.sumo import render as sumo_render


@torch.no_grad()
def literal_table(x, k, groups_idx):
    """(B, 2M + 2G) float literal values, the column layout of `rule_tree.collect`."""
    neg = (1 - x) * k
    parts = [x, neg]
    if groups_idx:
        parts += [torch.stack([x[:, g].amax(dim=1) for g in groups_idx], dim=1),
                  torch.stack([neg[:, g].amin(dim=1) for g in groups_idx], dim=1)]
    return torch.cat(parts, dim=1)


@torch.no_grad()
def classify_events(model, training_data, video_ids, head, root, rules, groups, possible, cfg, hit_slots):
    """{(video, agent, onset frame): {"group", "rules": [rule numbers that held on a hit row]}}."""
    device, S = cfg["device"], model.slot_size
    root_idx = [torch.tensor([model.input_index(a, s, j) for a in atoms], device=device) for atoms, s, j, _p in root]
    groups_idx = [torch.tensor(idx, device=device) for _n, idx in groups]
    events = {}
    for video_id in tqdm(video_ids, desc="Events"):
        can = possible(video_id)[:, head]
        for frames in training_data["frames"][video_id].split(cfg["frames_per_video"]):
            rows = step05._video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            x, k = model.split_inputs(rows["inputs"])
            first, at_risk = rows["first"][:, head], rows["at_risk"][:, head]
            hit = at_risk & can[rows["rows"][:, 1]] & (first >= 0) & (first < hit_slots * S)
            if not hit.any():
                continue
            xh, kh = x[hit], k[hit]
            t, n = rows["rows"][hit].T
            onset = (t + first[hit]).long()
            holds_root = torch.ones(len(xh), dtype=torch.bool, device=device)
            for idx, (_a, _s, _j, positive) in zip(root_idx, root):
                holds_root &= (xh[:, idx].amax(dim=1) > 0) if positive else (((1 - xh[:, idx]) * kh[:, idx]).amin(dim=1) > 0)
            F = literal_table(xh, kh, groups_idx)
            fired = []
            for path in rules:
                h = holds_root.clone()
                for f, value in path:
                    h &= (F[:, f] > 0) == bool(value)
                fired.append(h)
            for i in range(len(xh)):
                key = (video_id, int(n[i]), int(onset[i]))
                e = events.setdefault(key, {"root": False, "rules": set()})
                e["root"] |= bool(holds_root[i])
                for r, h in enumerate(fired):
                    if h[i]:
                        e["rules"].add(r + 1)
    out = {}
    for key, e in events.items():
        group = "explained" if e["rules"] else ("root_only" if e["root"] else "no_root")
        out[key] = {"group": group, "rules": sorted(e["rules"])}
    return out


NEAR_M = 30.0              # m to the junction (stop line): "near"
CONTEXT_S = 2.0            # s before the onset the scene is checked over
DIST_BINS = ((0.0, 15.0), (15.0, 30.0), (30.0, 60.0), (60.0, float("inf")))


def junction_distance(e):
    """m from the ego's front to the end of its lane (the stop line on an incoming edge), 0 inside the
    junction (an internal lane ":..."); the SUMO lane position is the front."""
    if e["road"].startswith(":"):
        return 0.0
    return max(e["lane_len"] - e["lane_pos"], 0.0)


def leader_gap(world_frame, e):
    """m of free gap to the nearest vehicle ahead on the ego's own lane (lane_pos ahead), None if none."""
    best = None
    for vid, v in world_frame["veh"].items():
        if vid == "ego" or v["lane"] != e["lane"] or v["lane_pos"] <= e["lane_pos"]:
            continue
        gap = v["lane_pos"] - v["length"] - e["lane_pos"]
        best = gap if best is None else min(best, gap)
    return best


def event_context(dataset, vid, onset, maps):
    """What the simulator knows around an ego onset at (1-based) frame onset: the junction distance, inside /
    after the junction, the route's turn, a traffic light ahead, the agents the camera shows (GT boxes, not
    traffic lights) over the CONTEXT_S before, the leader on the ego's lane, the ego's speed 1 s before, and
    the injected rules the ego executed within 2 s before .. 0.5 s after (`meta/firings`)."""
    labels, data, world, firings = sumo_render._video(dataset, vid)
    res = json.loads((dataset / "meta" / "results" / f"{vid}.json").read_text())
    spec = maps["maps"].get(res.get("map") or maps.get("videos", {}).get(vid))
    k = int(CONTEXT_S * 10)
    w_on = min(max(onset - 1, 0), len(world) - 1)                 # world index of the onset frame
    w_before = max(w_on - 10, 0)                                  # 1 s before
    e = world[w_before]["veh"].get("ego")
    if e is None:
        return None
    turn = route_turn(spec, res["ego_route"]) if spec and spec.get("layout") != "grid" and res.get("ego_route") else None
    after = bool(res.get("ego_route")) and e["road"] == res["ego_route"][-1]
    tl_names = {i for i, n in enumerate(labels["agent_labels"]) if n in ("TL", "OthTL")}
    shown, shown_veh, shown_ped = set(), set(), set()
    for frame in range(max(1, onset - k), onset):
        for a in data["frames"].get(str(frame), {}).get("annos", {}).values():
            cls = a["agent_ids"][0]
            if cls in tl_names:
                continue
            shown.add(a["tube_uid"])
            name = labels["agent_labels"][cls]
            (shown_ped if name == "Ped" else shown_veh).add(a["tube_uid"])
    tl = any(world[i].get("tl") for i in range(max(0, w_on - k), w_on + 1))
    rules = sorted({f["rule"] for f in firings if f.get("agent_is_ego") and f.get("executed") and f.get("frame_fire")
                    and -5 <= onset - f["frame_fire"] <= 20})
    return {"dist_m": round(junction_distance(e), 1), "inside": e["road"].startswith(":"), "after": after,
            "turn": turn, "tl": tl, "agents_shown": len(shown), "vehicles_shown": len(shown_veh),
            "peds_shown": len(shown_ped), "leader_gap_m": leader_gap(world[w_before], e),
            "speed_1s_before": e["speed"], "rules_fired": rules}


def context_summary(events, contexts):
    """Per event group: counts of the scene categories, incl. "empty junction approach" (near or inside the
    junction, before leaving it, no agent shown, no traffic light, no leader within NEAR_M, no injected rule)."""
    out = {}
    for group in ("explained", "root_only", "no_root", "all"):
        keys = [k for k, e in events.items() if group == "all" or e["group"] == group]
        cs = [contexts[k] for k in keys if contexts.get(k) is not None]
        n = len(cs)
        if not n:
            continue
        near = [c for c in cs if not c["after"] and c["dist_m"] <= NEAR_M]
        empty = lambda c: c["agents_shown"] == 0
        no_lead = lambda c: c["leader_gap_m"] is None or c["leader_gap_m"] > NEAR_M
        row = {"events": n, "with_context": n,
               "dist_bins": {f"{lo:g}-{hi:g} m": 0 for lo, hi in DIST_BINS},
               "inside_junction": sum(c["inside"] for c in cs),
               "after_junction": sum(c["after"] for c in cs),
               "near_junction": len(near),
               "no_agent_shown": sum(empty(c) for c in cs),
               "traffic_light": sum(c["tl"] for c in cs),
               "rule_fired": sum(bool(c["rules_fired"]) for c in cs),
               "near_and_no_agent": sum(empty(c) for c in near),
               "near_no_agent_no_tl": sum(empty(c) and not c["tl"] for c in near),
               "empty_junction_approach": sum(empty(c) and not c["tl"] and no_lead(c) and not c["rules_fired"]
                                              for c in near),
               "turn_of_empty": {}, "leader_within_near_m": sum(not no_lead(c) for c in cs)}
        for c in cs:
            if c["after"]:
                continue
            for lo, hi in DIST_BINS:
                if lo <= c["dist_m"] < hi or (hi == float("inf") and c["dist_m"] >= lo):
                    row["dist_bins"][f"{lo:g}-{hi:g} m"] += 1
                    break
        for c in near:
            if empty(c) and not c["tl"] and no_lead(c) and not c["rules_fired"]:
                row["turn_of_empty"][str(c["turn"])] = row["turn_of_empty"].get(str(c["turn"]), 0) + 1
        out[group] = row
    return out


def render_clip(dataset, vid, onset, first_frame, last_frame, out_file, label, fps=10):
    """Frames first_frame..last_frame of vid as an mp4, the title noting the time to the onset."""
    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(range(first_frame, last_frame + 1)):
            dt = (frame - onset) / 10
            note = f"{label} onset at f{onset}: " + ("NOW" if frame == onset else f"{dt:+.1f} s")
            sumo_render.render(dataset, vid, frame, Path(tmp) / f"{i:05d}.png", note=note)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(Path(tmp) / "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_file)], check=True)


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--rule_list", type=Path, required=True, help="JSON of rule_tree --mode list")
    parser.add_argument("--clip_split", type=str, default="val", choices=("train", "val"),
                        help="videos to take the events from (--split is the pipeline's data split)")
    parser.add_argument("--num", type=int, default=12, help="root_only events to render")
    parser.add_argument("--controls", type=int, default=3, help="explained events to render for comparison")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--before_s", type=float, default=5.0)
    parser.add_argument("--after_s", type=float, default=2.0)
    parser.add_argument("--no_render", action="store_true", help="only the event counts and the scene statistics")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id if getattr(args, "run_id", None) \
        else args.output_dir / "residual_clips"
    rl = utils_data.load_json(args.rule_list)
    cfg = step05.get_cfg(config.get_step_05_input(args))
    if list(cfg["roles"]) != list(rl["roles"]):
        raise SystemExit(f"--roles {list(cfg['roles'])} differ from the rule list's {rl['roles']}: pass the same")
    device = cfg["device"]
    training_data = step05.load_training_data(cfg)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    index = defaultdict(list)
    for i, n in enumerate(names):
        index[n].append(i)
    head = index[rl["head"]][0]
    model = TemporalRuleNet(training_data["num_atoms"], cfg["slot_size"], cfg["num_slots"], 1, cfg["init_log_alpha"],
                            cfg["mask_type"], cfg["windows"], roles=cfg["roles"],
                            input_mask=step05.rule_input_mask(training_data, cfg, 2 + len(cfg["roles"])),
                            ranges=cfg.get("slot_ranges", False)).to(device)
    scope_names = ["own", "other", *cfg["roles"]]
    root = parse_root(rl["root"], index, {s: i for i, s in enumerate(scope_names)})
    groups = vehicle_groups(model, names, scope_names)
    if len(groups) != rl["num_groups"]:
        raise SystemExit(f"{len(groups)} vehicle groups, the rule list has {rl['num_groups']}: other data or code")
    rules = [[(int(f), bool(v)) for f, v in r["literals"]] for r in rl["list"]["rules"]]
    texts = [r["rule"] for r in rl["list"]["rules"]]
    possible = step05.class_possible(training_data, vocab, device)
    hit_slots = int(rl.get("hit_slots", 1))

    events = classify_events(model, training_data, training_data[f"{args.clip_split}_videos"], head, root, rules, groups,
                             possible, cfg, hit_slots)
    counts = defaultdict(int)
    for e in events.values():
        counts[e["group"]] += 1
    per_rule = defaultdict(int)
    for e in events.values():
        for r in e["rules"]:
            per_rule[r] += 1
    total = max(len(events), 1)
    print(f"\n[residual_clips] {args.exp} {args.clip_split}: {len(events)} {rl['head']} onset events "
          f"(hit within {hit_slots} slot(s)), root {rl['root']}")
    for g in ("explained", "root_only", "no_root"):
        print(f"  {g:10s} {counts[g]:5d} ({counts[g] / total:.0%})")
    for r in sorted(per_rule):
        print(f"  r{r} holds on a hit row of {per_rule[r]} events: {texts[r - 1]}")

    dataset = Path(args.dataset_path)                       # config.machine_setup: the synthetic dataset folder
    maps = json.loads((dataset / "meta" / "maps.json").read_text())
    contexts = {k: event_context(dataset, k[0], k[2], maps) for k in tqdm(sorted(events), desc="Scene context")}
    summary = context_summary(events, contexts)
    print(f"\n  scene of the ego's onsets ({CONTEXT_S:g} s before; near = <= {NEAR_M:g} m to the stop line or "
          f"inside the junction, before leaving it; agents = GT boxes shown, traffic lights not counted):")
    for group, r in summary.items():
        n = r["events"]
        pct = lambda x: f"{x} ({x / n:.0%})"
        print(f"  [{group}] {n} events: near junction {pct(r['near_junction'])} (inside {r['inside_junction']}, "
              f"after {r['after_junction']}), no agent shown {pct(r['no_agent_shown'])}, traffic light "
              f"{pct(r['traffic_light'])}, leader <= {NEAR_M:g} m {pct(r['leader_within_near_m'])}, "
              f"injected rule fired {pct(r['rule_fired'])}")
        print(f"      near & no agent {pct(r['near_and_no_agent'])}, & no light {pct(r['near_no_agent_no_tl'])}, "
              f"EMPTY JUNCTION APPROACH (also no leader, no rule) {pct(r['empty_junction_approach'])}, its turns "
              f"{r['turn_of_empty']}; distance bins {r['dist_bins']}")
    out_dir = args.output_dir / "residual_clips" / args.clip_split
    rng = random.Random(args.seed)
    pick = lambda g, num: rng.sample(sorted(k for k, e in events.items() if e["group"] == g),
                                     min(num, counts[g]))
    chosen = [("root_only", k) for k in pick("root_only", args.num)] + \
             [("explained", k) for k in pick("explained", args.controls)]
    rendered = []
    if args.no_render:
        chosen = []
    for group, (vid, agent, onset) in chosen:
        num_frames = len(sumo_render._video(dataset, vid)[1]["frames"])
        lo, hi = max(1, onset - int(args.before_s * 10)), min(num_frames, onset + int(args.after_s * 10))
        out_file = out_dir / f"{group}_{vid}_f{onset}.mp4"
        print(f"  rendering {out_file.name} (frames {lo}-{hi})")
        render_clip(dataset, vid, onset, lo, hi, out_file, rl["head"])
        rendered.append({"group": group, "video": vid, "agent": agent, "onset_frame": onset,
                         "rules": events[(vid, agent, onset)]["rules"], "file": out_file.name})
    utils_data.save_json({"exp": args.exp, "split": args.clip_split, "rule_list": str(args.rule_list), "root": rl["root"],
                          "rules": texts, "hit_slots": hit_slots, "counts": dict(counts),
                          "per_rule_events": {f"r{r}": c for r, c in per_rule.items()}, "rendered": rendered,
                          "scene_summary": summary,
                          "events": [{"video": v, "agent": a, "onset_frame": f, **e, "scene": contexts.get((v, a, f))}
                                     for (v, a, f), e in events.items()]},
                         out_dir / "events.json")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
