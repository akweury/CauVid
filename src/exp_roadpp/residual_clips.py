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
    out_dir = args.output_dir / "residual_clips" / args.clip_split
    rng = random.Random(args.seed)
    pick = lambda g, num: rng.sample(sorted(k for k, e in events.items() if e["group"] == g),
                                     min(num, counts[g]))
    chosen = [("root_only", k) for k in pick("root_only", args.num)] + \
             [("explained", k) for k in pick("explained", args.controls)]
    rendered = []
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
                          "events": [{"video": v, "agent": a, "onset_frame": f, **e} for (v, a, f), e in events.items()]},
                         out_dir / "events.json")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
