"""Turn failure clips: the ego's turn onsets the rule model misses, rendered for checking by eye, with the ego's
labels and the turn atoms (`ego_motion` v4, scene drift) frame by frame (2026-11-23).

Why: the turn atoms (av:turn_left / turn_straight / turn_right) gave no gain in the rule lab (20261123-egoturn_lab,
soft_logit -0.005 over 77 heads), and the AV-TurLft rules used only "turn_straight 3 slots ago", never turn_left.
`turn_probe.py` found that 76% of the AV-TurLft onsets follow a yaw rise of ~3 frames and 20% end a brake during
which the ego already turned for ~2 s. The clips show what the atoms say before the onset.

Rule model (no MLP): as the rule lab's, on the heads of --clip_heads:
  - mining: the relaxed miner (shape / gestalt atoms as literals, not as roots);
  - scoring: the soft fit --clip_mode (default logit: the lab's soft_logit), on the train sample.
Events: every onset of those heads on the odd val videos (as `lane_clips`). An event's score is the model's highest
on its hit rows; missed if below the head's threshold (only --clip_fpr of the non-onset rows above it).
Path of the onset, by the ego's label in the frame before: after_brake, after_stop, from_mov (AV-Mov*), other.
Clips: --num missed events per path (after_brake and from_mov first) and --controls detected ones, --before_s before
the onset to --after_s after it, top view (left) and camera view (right). The text gives per frame:
  - the head, the score vs the threshold, the path, the rule that holds at the best row or the nearest miss;
  - the ego's labels in that frame;
  - the scene drift (1e-3 image widths / frame, > 0: the scene moves right, a left turn) against the dead band, and
    the turn atom that holds (as the model reads it: smoothed and debounced).

    ./run_roadpp_docker.sh --gpu 0 --turn_fail_clips -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True \\
        --run_id 20261123-egoturn_lab
Writes <output_dir>/turn_fail_clips/<state>_<path>/<head>_<vid>_f<onset>.mp4 and events.json (every event: score,
missed or not, path, explanation; the drift band; counts per path).
"""
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from src.exp_roadpp import (config, ego_motion, gestalt_fail_clips, lane_clips, mine_sweep, residual_clips, rule_lab,
                            step05, step07_paper, utils_data, utils_pipe)

PATH_ORDER = ("after_brake", "from_mov", "after_stop", "other")


def path_of(prev):
    if prev == "av:AV-Brake":
        return "after_brake"
    if prev == "av:AV-Stop":
        return "after_stop"
    if prev and prev.startswith("av:AV-Mov"):
        return "from_mov"
    return "other"


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--clip_heads", type=str, default="av:AV-TurLft", help="comma separated head names")
    parser.add_argument("--clip_mode", type=str, default="logit", help="the rule_lab soft fit of the rule model")
    parser.add_argument("--clip_fpr", type=float, default=0.02, help="share of non-onset rows above the threshold")
    parser.add_argument("--num", type=int, default=6, help="missed clips per path")
    parser.add_argument("--controls", type=int, default=3, help="detected clips")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--before_s", type=float, default=4.0)
    parser.add_argument("--after_s", type=float, default=1.0)
    parser.add_argument("--font_scale", type=float, default=1.7, help="text size in the clips")
    parser.add_argument("--no_render", action="store_true")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    if not getattr(args, "run_id", None):
        raise ValueError("--run_id is needed (the rule lab's run, e.g. 20261123-egoturn_lab)")
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    training_data = step05.load_training_data(cfg)
    layout = training_data.get("grid_layout", {})
    cfg["mine_root_exclude"] = [layout[k] for k in ("shape", "gestalt") if k in layout]
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    head_mask = training_data["target_heads"].to(device)
    wanted = [h for h in args.clip_heads.split(",") if h]
    missing = [h for h in wanted if h not in names]
    if missing:
        raise ValueError(f"unknown heads {missing}")
    heads = [names.index(h) for h in wanted]
    possible = step05.class_possible(training_data, vocab, device)
    L, A = model.num_literals, model.num_atoms
    print(f"[turn_fail_clips] heads: {', '.join(wanted)}")

    # The rule model (rule lab: relaxed miner, soft fit), on the train sample.
    train, val = mine_sweep.load_rows(cfg, training_data, model, head_mask, possible, args.output_dir / "mine_sweep")
    val_ids = list(training_data["val_videos"])
    check = mine_sweep.subset(val, set(range(0, len(val_ids), 2)))
    del val
    rules = rule_lab.mine_component("relaxed", False, model, names, heads, train, check, cfg, device)
    base = train["rate"].float()
    soft = rule_lab.fit_soft(rules, torch.full((len(rules),), 0.1), base, train, model, heads, device, args.clip_mode)
    rs = rule_lab.RuleSet(rules, base, [], None, L, A, device, {args.clip_mode: soft})
    gate, weight = soft[0], soft[1][0]
    by_head = defaultdict(list)
    for i, r in enumerate(rules):
        by_head[r["head"]].append(i)
    kept = [i for i in range(len(rules)) if abs(float(weight[i])) >= rule_lab.DROP_WEIGHT]
    head_rules = {names[h]: sorted(({"weight": round(float(weight[i]), 2),
                                     "rule": lane_clips.rule_text(model, names, rules[i], gate[i].tolist())}
                                    for i in kept if rules[i]["head"] == h), key=lambda x: -abs(x["weight"]))
                  for h in heads}
    for h, rr in head_rules.items():
        print(f"\n[turn_fail_clips] {h}: {len(rr)} rules (|w| >= {rule_lab.DROP_WEIGHT}), strongest:")
        for x in rr[:8]:
            print(f"    w {x['weight']:+.2f}  {x['rule']}")

    # Events and thresholds (odd val videos, as the lab).
    eval_ids = val_ids[1::2]
    neg, events = gestalt_fail_clips.collect_events(model, rs, args.clip_mode, training_data, eval_ids, heads,
                                                    head_mask, possible, cfg)
    th = lane_clips.thresholds(neg, args.clip_fpr)

    # Per video: the ego's labels per frame and its scene drift / turn atoms (`ego_motion`).
    atoms = utils_data.load_json(cfg["val_atom_file"])
    boxes_all = torch.load(cfg["val_box_file"], map_location="cpu")
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    label_names = [step05._atom_name(sig) for sig in unique_atoms]
    av_idx = [a for a, nm in enumerate(label_names) if nm.startswith("av:")]
    spec = step05.load_ego_motion(cfg)
    band = spec["bands"]["scene_drift"] if spec and "scene_drift" in spec.get("bands", {}) else None
    turn_names = ego_motion.SIGNED["scene_drift"]
    i_turn = [ego_motion.ATOMS.index(a) for a in turn_names]
    cache = {}

    def video(vid):
        if vid not in cache:
            X = step05.build_video_tensor(atoms[vid], len(unique_atoms))
            ego = step05.video_agent_ids(atoms[vid]).get(None)
            boxes, classes, has_box = step05._agent_boxes(boxes_all, vid, atoms[vid], X)
            labs = [[label_names[a] for a in av_idx if bool(X[t, ego, a])] if ego is not None else []
                    for t in range(X.shape[0])]
            drift, turn = None, None
            if ego is not None and spec is not None and "scene_drift" in ego_motion.SIGNALS:
                sig = ego_motion.signals(boxes, has_box, classes, ego, X, unique_atoms)
                drift = sig["scene_drift"]
                turn = ego_motion.ego_atoms(sig, spec)[:, i_turn]
            cache[vid] = (labs, drift, turn)
        return cache[vid]

    groups = defaultdict(list)
    counts = defaultdict(Counter)
    out_events = []
    for key, e in tqdm(events.items(), desc="Paths"):
        vid, agent, onset, h = key
        labs, drift, turn = video(vid)
        prev = labs[onset - 1] if 0 < onset <= len(labs) else []
        prev_main = next((x for x in prev if x != names[h]), None)
        path = path_of(prev_main)
        missed = step07_paper._logit_bins(torch.tensor([e["score"]]))[0] < th[h]
        state = "missed" if bool(missed) else "detected"
        counts[names[h]][f"{state}_{path}"] += 1
        groups[f"{state}_{path}"].append(key)
        ex = lane_clips.explain(model, names, rules, gate, weight, by_head[h], e["act"], e["lits"]) if "act" in e else {}
        e.update(state=state, path=path, explain=ex)
        before = {}
        if drift is not None:
            for d in (1, 5, 10, 20):
                t = onset - d
                if t >= 0:
                    before[str(d)] = {"drift_1e3": round(float(drift[t]) * 1000, 3) if drift[t] == drift[t] else None,
                                      "turn": [n for n, on in zip(turn_names, turn[t].tolist()) if on]}
        out_events.append({"video": vid, "onset_frame": onset, "head": names[h], "state": state, "path": path,
                           "prev_labels": prev, "score": round(e["score"], 4),
                           "threshold": lane_clips.bin_score(int(th[h])), "before": before, **ex})
    for h, c in counts.items():
        print(f"\n[turn_fail_clips] {h}: events by state and path (label in the frame before): {dict(c)}")
        for p in PATH_ORDER:
            n = c[f"missed_{p}"] + c[f"detected_{p}"]
            if n:
                print(f"    {p:12s}: {n:5d} events, missed {c[f'missed_{p}'] / n:.0%}")
    if band is not None:
        print(f"  scene drift dead band +-{band * 1000:.3f} (1e-3 image widths / frame)")

    # Clips.
    dataset = Path(args.dataset_path)
    out_dir = args.output_dir / "turn_fail_clips"
    rng = random.Random(args.seed)
    rendered = []
    plan = [(f"missed_{p}", args.num) for p in PATH_ORDER] + [("detected_from_mov", args.controls),
                                                              ("detected_after_brake", args.controls)]
    for group, num in plan:
        keys = sorted(groups[group])
        for key in rng.sample(keys, min(num, len(keys))):
            if args.no_render:
                break
            vid, agent, onset, h = key
            labs, drift, turn = video(vid)
            e = events[key]
            ex = e["explain"]
            line = (f"rules hold: {ex['holding'][0]['rule']} (w {ex['holding'][0]['weight']})" if ex.get("holding")
                    else f"near miss: {ex['near_miss'][0]['rule']}" if ex.get("near_miss") else "no rule of the head")
            label = (f"[{group}] {names[h]}; rules {e['score']:.3f} (thr {lane_clips.bin_score(int(th[h])):.3f})\n"
                     f"{line[:150]}")
            num_frames = len(residual_clips.sumo_render._video(dataset, vid)[1]["frames"])
            lo, hi = max(1, onset - int(args.before_s * 10)), min(num_frames, onset + int(args.after_s * 10))
            out_file = out_dir / group / f"{names[h].replace(':', '_')}_{vid}_f{onset}.mp4"
            print(f"  rendering {out_file.relative_to(out_dir)} (frames {lo}-{hi})")
            with tempfile.TemporaryDirectory() as tmp:
                for i, f in enumerate(range(lo, hi + 1)):
                    dt = (f - onset) / 10
                    tf = min(f, len(labs) - 1)
                    ego_txt = ",".join(x.replace("av:", "") for x in labs[tf]) or "-"
                    if drift is not None:
                        dv = float(drift[tf])
                        on = [n for n, o in zip(turn_names, turn[tf].tolist()) if o]
                        turn_txt = (f"scene drift {dv * 1000:+.2f} (band +-{band * 1000:.2f}) -> "
                                    f"{on[0] if on else 'no turn atom'}") if dv == dv else "scene drift -"
                    else:
                        turn_txt = "no turn atoms in this data"
                    note = (f"{label}\nonset at f{onset}: " + ("NOW" if f == onset else f"{dt:+.1f} s")
                            + f"  |  ego: {ego_txt}  |  {turn_txt}")
                    residual_clips.sumo_render.render(dataset, vid, f, Path(tmp) / f"{i:05d}.png", note=note,
                                                      font_scale=args.font_scale)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-framerate", "10",
                                "-i", str(Path(tmp) / "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_file)], check=True)
            rendered.append({"file": str(out_file.relative_to(out_dir)), "video": vid, "onset_frame": onset,
                             "head": names[h], "group": group})
    out_dir.mkdir(parents=True, exist_ok=True)
    utils_data.save_json({"mode": args.clip_mode, "fpr": args.clip_fpr, "heads": wanted,
                          "drift_band": band, "rules": head_rules, "counts": {h: dict(c) for h, c in counts.items()},
                          "rendered": rendered, "events": out_events}, out_dir / "events.json")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
