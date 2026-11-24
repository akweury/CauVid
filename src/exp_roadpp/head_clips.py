"""Head clips: the onsets of any head under one rule lab variant, rendered with the agent outlined, its labels,
the score vs the threshold and the rules (with the transfer source of transferred ones) (2026-11-24).

Why: the transfer (`rule_lab` "tr" / "trp") made Bus:MovRht learnable (soft_logit 0.002 -> 0.146 with 3 rules);
the user wants to see what the rules catch and miss.

Rule model as the rule lab's (no MLP), on --clip_heads:
  - components of --clip_variant ("relaxed+tr", "relaxed+exh+trp", ..; the negated literals of the other agents
    off, as the "_nn" variants), on the train sample; the transfer sources are mined on their own sample rows;
  - counts on the train sample, the check filter of --lab_check, the soft fit --clip_mode (logit = soft_logit).
Events: every onset of those heads on the odd val videos (as the rule lab evaluates). An event's score is the
model's highest on its hit rows; detected if at or above the head's threshold (only --clip_fpr of the non-onset
rows above it). Clips: --num missed and --num detected events per head, --before_s before the onset to
--after_s after it. Per frame the text gives the agent's labels, the time to the onset, the score vs the
threshold, and the rule that holds at the best row (its transfer source) or the nearest miss (literals + / -).

    ./run_roadpp_docker.sh --gpu 0 --head_clips -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True \\
        --run_id 20261123-egolay_lab --clip_heads Bus:MovRht --clip_variant relaxed+tr
Writes <output_dir>/head_clips/<head>/<missed|detected>_<vid>_f<onset>.mp4 and events.json (every event, the rules
with weights, gates and transfer sources).
"""
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

from src.exp_roadpp import (config, gestalt_fail_clips, lane_clips, mine_sweep, residual_clips, rule_lab, step05,
                            step07_paper, utils_data, utils_pipe)

LAB_DEFAULTS = {"lab_precise_min": 0.5, "lab_min_events": 5, "lab_min_videos": 1, "lab_max_len": 8, "lab_beam": 100,
                "lab_branch": 10, "lab_max_rules": 100, "lab_exh_lift": 2.0, "lab_exh_seed_lift": 1.5,
                "lab_exh_gain": 1.1, "lab_exh_max": 400, "lab_exh_seeds": 300, "lab_tr_max_onsets": 60,
                "lab_tr_min_source": 100, "lab_tr_rules": 200, "lab_tr_min_events": 1, "lab_tr_confirm": 2}


def agent_age(X, n, t):
    """Frames agent n was seen (any label atom) without a break right before frame t (2026-11-24, the user: an
    event right after the agent appears, e.g. a new state in its second frame, is not worth predicting or showing)."""
    seen = X[:t, n].any(dim=1)
    gone = torch.nonzero(~seen).flatten()
    return int(t - (int(gone.max()) + 1 if len(gone) else 0))


class NoNegOthers:
    """The rule net with the negated literals of the other agents off (the rule lab's "_nn")."""

    def __init__(self, model):
        self._model = model
        self.input_mask = model.input_mask & ~rule_lab.negated_others(model).to(model.input_mask.device)

    def __getattr__(self, key):
        return getattr(self._model, key)


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--clip_heads", type=str, default="Bus:MovRht")
    parser.add_argument("--clip_variant", type=str, default="relaxed+tr", help="rule lab miners joined by +")
    parser.add_argument("--clip_mode", type=str, default="logit", help="soft fit (logit = the lab's soft_logit)")
    parser.add_argument("--clip_fpr", type=float, default=0.02)
    parser.add_argument("--lab_check", type=str, default="True")
    parser.add_argument("--num", type=int, default=6, help="missed and detected clips per head (large: all)")
    parser.add_argument("--clip_videos", type=str, default="eval", choices=("eval", "val", "all"),
                        help="onsets of the odd val videos (the lab's evaluation), all val videos, or val + train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--before_s", type=float, default=4.0)
    parser.add_argument("--after_s", type=float, default=1.0)
    parser.add_argument("--font_scale", type=float, default=1.7)
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--min_age", type=int, default=5,
                        help="skip onsets whose agent was seen for fewer frames right before (the user, 2026-11-24)")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    if not getattr(args, "run_id", None):
        raise ValueError("--run_id is needed (the rule lab's run)")
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    cfg.update(LAB_DEFAULTS)
    cfg["lab_check"] = args.lab_check == "True"
    device = cfg["device"]
    training_data = step05.load_training_data(cfg)
    layout = training_data.get("grid_layout", {})
    cfg["mine_root_exclude"] = [layout[k] for k in ("shape", "gestalt") if k in layout]
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    ext = NoNegOthers(model)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    head_mask = training_data["target_heads"].to(device)
    all_heads = torch.nonzero(head_mask).flatten().tolist()
    cfg["lab_all_heads"] = all_heads
    wanted = [h for h in args.clip_heads.split(",") if h]
    missing = [h for h in wanted if h not in names]
    if missing:
        raise ValueError(f"unknown heads {missing}")
    heads = [names.index(h) for h in wanted]
    possible = step05.class_possible(training_data, vocab, device)
    L, A = model.num_literals, model.num_atoms
    print(f"[head_clips] {', '.join(wanted)}; variant {args.clip_variant}")

    # The rule model (the lab's components, counts, check filter and soft fit).
    train, val = mine_sweep.load_rows(cfg, training_data, model, head_mask, possible, args.output_dir / "mine_sweep")
    val_ids = list(training_data["val_videos"])
    check = mine_sweep.subset(val, set(range(0, len(val_ids), 2)))
    del val
    rules, seen = [], set()
    for c in args.clip_variant.split("+"):
        for r in rule_lab.mine_component(c, False, ext, names, heads, train, check, cfg, device):
            key = (r["head"], tuple(sorted(r["literals"])))
            if key not in seen:
                seen.add(key)
                rules.append(r)
    counts = rule_lab.sample_counts(rules, train, L, device, A)
    final = rule_lab.finalize(rules, counts, cfg["lab_min_events"])
    if cfg["lab_check"]:
        index = {(r["head"], tuple(sorted(r["literals"]))): i for i, r in enumerate(rules)}
        final = rule_lab.check_filter(final, rule_lab.sample_counts(rules, check, L, device, A), index)
    base = counts["base"]
    soft = rule_lab.fit_soft(final, rule_lab.confidence(final, base, rule_lab.SOFT_PRIOR), base, train, model, heads,
                             device, args.clip_mode)
    rs = rule_lab.RuleSet(final, base, [], None, L, A, device, {args.clip_mode: soft})
    gate, weight = soft[0], soft[1][0]
    by_head = defaultdict(list)
    for i, r in enumerate(final):
        by_head[r["head"]].append(i)
    rule_info = {names[h]: sorted(({"weight": round(float(weight[i]), 2),
                                    "rule": lane_clips.rule_text(model, names, final[i], gate[i].tolist()),
                                    "transfer": final[i].get("transfer"), "precision": round(final[i]["precision"], 3),
                                    "events": final[i]["events"]}
                                   for i in by_head[h]), key=lambda x: -x["weight"]) for h in heads}
    for h, rr in rule_info.items():
        print(f"\n[head_clips] {h}: {len(rr)} rules")
        for x in rr[:10]:
            print(f"    w {x['weight']:+.2f}  p {x['precision']:.2f}  {x['events']:4d} onsets  "
                  f"{('[' + x['transfer'] + '] ') if x['transfer'] else ''}{x['rule']}")

    # Events on the odd val videos, thresholds.
    eval_ids = val_ids[1::2]
    neg, events = gestalt_fail_clips.collect_events(model, rs, args.clip_mode, training_data, eval_ids, heads,
                                                    head_mask, possible, cfg)
    th = lane_clips.thresholds(neg, args.clip_fpr)              # the threshold of the evaluation videos, always
    split_of = {v: "eval" for v in eval_ids}
    more = []
    if args.clip_videos in ("val", "all"):
        more += [(v, "check") for v in val_ids[0::2]]
    if args.clip_videos == "all":
        more += [(v, "train") for v in training_data["train_videos"]]
    if more:
        _neg2, ev2 = gestalt_fail_clips.collect_events(model, rs, args.clip_mode, training_data, [v for v, _ in more],
                                                       heads, head_mask, possible, cfg)
        events.update(ev2)
        split_of.update(dict(more))
    rule_tr = {i: final[i].get("transfer") for i in range(len(final))}
    atoms = utils_data.load_json(cfg["val_atom_file"])
    if args.clip_videos == "all":
        atoms.update(utils_data.load_json(cfg["train_atom_file"]))
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    label_names = [step05._atom_name(sig) for sig in unique_atoms]
    groups, out_events = defaultdict(list), []
    young = 0
    X_of = {}
    for key, e in events.items():
        vid, agent, onset, h = key
        if vid not in X_of:
            X_of[vid] = step05.build_video_tensor(atoms[vid], len(unique_atoms))
        if agent_age(X_of[vid], agent, onset) < args.min_age:
            young += 1
            continue
        detected = bool(step07_paper._logit_bins(torch.tensor([e["score"]]))[0] >= th[h])
        ex = lane_clips.explain(model, names, final, gate, weight, by_head[h], e["act"], e["lits"]) if "act" in e else {}
        hold = [i for i in by_head[h] if float(e["act"][i]) >= 0.5 and float(weight[i]) > rule_lab.DROP_WEIGHT] \
            if "act" in e else []
        e.update(detected=detected, explain=ex, transfer=sorted({rule_tr[i] for i in hold if rule_tr[i]}))
        groups[(h, "detected" if detected else "missed")].append(key)
        out_events.append({"video": vid, "split": split_of.get(vid), "onset_frame": onset, "head": names[h],
                           "detected": detected,
                           "score": round(e["score"], 4), "threshold": lane_clips.bin_score(int(th[h])),
                           "transfer_rules_holding": e["transfer"], **ex})
    print(f"[head_clips] {young} onsets skipped: their agent was seen < {args.min_age} frames before")
    X_of.clear()
    for h in heads:
        n_d, n_m = len(groups[(h, "detected")]), len(groups[(h, "missed")])
        n_tr = sum(1 for k in groups[(h, "detected")] if events[k]["transfer"])
        print(f"\n[head_clips] {names[h]}: {n_d + n_m} onsets ({args.clip_videos} videos), detected {n_d} "
              f"({n_tr} with a transferred rule holding), missed {n_m}; threshold "
              f"{lane_clips.bin_score(int(th[h])):.3f} (fpr {args.clip_fpr})")

    # Clips.
    dataset = Path(args.dataset_path)
    out_dir = args.output_dir / "head_clips"
    rng = random.Random(args.seed)
    rendered = []
    cache = {}
    for h in heads:
        for state in ("missed", "detected"):
            keys = sorted(groups[(h, state)])
            for key in rng.sample(keys, min(args.num, len(keys))):
                if args.no_render:
                    break
                vid, agent, onset, _h = key
                if vid not in cache:
                    X = step05.build_video_tensor(atoms[vid], len(unique_atoms))
                    tube_of = {n: uid for uid, n in step05.video_agent_ids(atoms[vid]).items()}
                    cache[vid] = (X, tube_of)
                X, tube_of = cache[vid]
                e = events[key]
                ex = e["explain"]
                line = (f"rules hold: {ex['holding'][0]['rule']} (w {ex['holding'][0]['weight']})"
                        + (f" [{', '.join(e['transfer'])}]" if e["transfer"] else "") if ex.get("holding")
                        else f"near miss: {ex['near_miss'][0]['rule']}" if ex.get("near_miss") else "no rule of the head")
                label = (f"[{state}, {split_of.get(vid)}] {names[h]}; score {e['score']:.3f} (thr {lane_clips.bin_score(int(th[h])):.3f})\n"
                         f"{line[:170]}")
                num_frames = len(residual_clips.sumo_render._video(dataset, vid)[1]["frames"])
                lo, hi = max(1, onset - int(args.before_s * 10)), min(num_frames, onset + int(args.after_s * 10))
                out_file = out_dir / names[h].replace(":", "_") / f"{state}_{split_of.get(vid)}_{vid}_f{onset}.mp4"
                print(f"  rendering {out_file.relative_to(out_dir)} (frames {lo}-{hi})")
                with tempfile.TemporaryDirectory() as tmp:
                    for i, f in enumerate(range(lo, hi + 1)):
                        tf = min(f, X.shape[0] - 1)
                        labs = [label_names[a].partition(":")[2] for a in torch.nonzero(X[tf, agent]).flatten().tolist()]
                        dt = (f - onset) / 10
                        note = (f"{label}\nonset at f{onset}: " + ("NOW" if f == onset else f"{dt:+.1f} s")
                                + f"  |  agent: {', '.join(labs) or '-'}")
                        residual_clips.sumo_render.render(dataset, vid, f, Path(tmp) / f"{i:05d}.png", note=note,
                                                          highlight=tube_of.get(agent), font_scale=args.font_scale)
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-framerate", "10",
                                    "-i", str(Path(tmp) / "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_file)], check=True)
                rendered.append({"file": str(out_file.relative_to(out_dir)), "video": vid, "onset_frame": onset,
                                 "head": names[h], "state": state})
    out_dir.mkdir(parents=True, exist_ok=True)
    utils_data.save_json({"variant": args.clip_variant, "mode": args.clip_mode, "fpr": args.clip_fpr,
                          "rules": rule_info, "rendered": rendered, "events": out_events}, out_dir / "events.json")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
