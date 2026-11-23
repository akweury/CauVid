"""Gestalt failure clips: onsets the rule model with the gestalt atoms misses although the agent is in a group or
next to one, drawn with the groups in the SUMO top view (left) and the camera view (right) (2026-11-23).

Why: the gestalt atoms gave no clear gain in the rule lab (v1, v4, v5; with MDL fits the rules kept used them in
15 of 541 rules). Gestalt stays a direction to pursue (the user), so the misses are looked at by eye: is the group
there, is it the right group, would a group fact tell the onset, or is the group beside the point?

Rule model: as the rule lab's, on the heads of --clip_actions:
  - mining: the relaxed miner, with the shape and gestalt atoms as literals but not as roots;
  - scoring: the soft fit --clip_mode (default mdl_0.03, the compact one), on the train sample.
Events: every onset of those heads on the odd val videos (as `lane_clips`).
  - An event's score is the rule model's highest on its hit rows.
  - The event is missed if that score stays below the head's threshold (only --clip_fpr of the non-onset rows
    above it).
Group context: in the frame before the onset (`gestalt.groups` of the val boxes):
  in_group     the agent is a member of a group;
  near_group   a member of another group is near the agent;
  no_group     neither.
Situations rendered (--num each): missed_in_group, missed_near_group, and detected_in_group as a control.
Every clip spans --before_s before the onset to --after_s after it:
  - one group per frame: the agent's own group, else the nearest group near it (none if neither), in its colour
    in both views, with its union box (dashed) in the camera view and a caption (kind, members, direction, speed);
  - text --font_scale times the default size;
  - the event's agent outlined in red;
  - the title gives the head, the score vs the threshold, the agent's group context, and the rules that hold at the
    frame of the highest score or the strongest near miss.

    ./run_roadpp_docker.sh --gpu 2 --gestalt_fail_clips -- --exp sumo_maps_v7 --machine dgx-jing \\
        --shape_phases True --gestalt True --run_id 20261123-gestalt5_lab
Writes <output_dir>/gestalt_fail_clips/<situation>/<head>_<vid>_f<onset>.mp4 and events.json (every event: its
score, missed or not, group context; per head the counts and its rules that use a gestalt atom).
"""
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from src.exp_roadpp import (config, gestalt, lane_clips, mine_sweep, residual_clips, rule_lab, step05, step05_mine,
                            step07_paper, utils_data, utils_pipe)

# Not a class colour of the SUMO renderer (Car is #1f77b4: groups drawn in it looked like single cars, 2026-11-23).
COLORS = ["#ff00ff", "#00c853", "#ff6d00"]


@torch.no_grad()
def collect_events(model, rs, mode, training_data, video_ids, heads, head_mask, possible, cfg):
    """Non-onset score histograms (A, NB) and the events {(vid, agent, onset, head): {"score", "t", "act",
    "lits"}} of the rule model (`lane_clips.collect_events` without the MLP)."""
    device, S, w = cfg["device"], model.slot_size, int(cfg.get("clause_slots", 1))
    A, NB = model.num_atoms, step07_paper.NB
    gate = rs.soft[mode][0]
    neg = torch.zeros(A * NB, device=device)
    events = {}
    hd = torch.tensor(heads, device=device)
    for video_id in tqdm(video_ids, desc="Events (val)"):
        can = possible(video_id)
        for frames in training_data["frames"][video_id].split(int(cfg.get("mine_chunk", 256))):
            rows = step05._video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            keep, y = step05_mine._head_rows(rows, can, head_mask, S, w)
            keep, y = keep[:, hd], y[:, hd]
            sel = keep.any(dim=1)
            if not sel.any():
                continue
            lits = rule_lab._with_true(model.literal_values(rows["inputs"][sel]) > 0.5)
            p = rs.scores(lits)[mode][:, hd]
            keep, y = keep[sel], y[sel]
            b = hd[None].expand_as(p) * NB + step07_paper._logit_bins(p)
            neg.add_(torch.bincount(b[keep & ~y], minlength=A * NB).float())
            if not y.any():
                continue
            b_idx, j_idx = torch.nonzero(y, as_tuple=True)
            act = rule_lab._soft_act(lits[b_idx].float(), rs.body, gate)
            t, n = rows["rows"][sel][b_idx].T
            onset = t + rows["first"][sel][b_idx, hd[j_idx]]
            for i in range(len(b_idx)):
                key = (video_id, int(n[i]), int(onset[i]), heads[int(j_idx[i])])
                s = float(p[b_idx[i], j_idx[i]])
                e = events.setdefault(key, {"score": -1.0})
                if s > e["score"]:
                    e.update(score=s, t=int(t[i]), act=act[i].cpu(), lits=lits[b_idx[i]].cpu())
    return neg.view(A, NB).cpu(), events


def agent_group(g, t, agent):
    """The one group to show for the agent in frame t: its own group if it is in one, else the group of the nearest
    member of another group near it, else None."""
    if bool(g["in_group"][t, agent]):
        return int(g["label"][t, agent])
    other = g["near"][t, agent] & g["in_group"][t]
    if bool(other.any()):
        d = torch.where(other, g["dist"][t, agent], torch.full_like(g["dist"][t, agent], float("inf")))
        return int(g["label"][t, int(d.argmin())])
    return None


def group_captions(g, t, classes, tube_of, only=None):
    """[(colour, [tube_uids], caption)] of the groups in frame t (`gestalt.groups`); only: just that group."""
    out = []
    ids = [k for k in range(g["size"].shape[1]) if g["is_group"][t, k] and (only is None or k == only)]
    for i, k in enumerate(ids):
        members = [n for n in range(g["label"].shape[1]) if int(g["label"][t, n]) == k and bool(g["in_group"][t, n])]
        uids = [tube_of[n] for n in members if n in tube_of]
        speed = sum(float(g["speed"][t, n]) for n in members) / max(len(members), 1)
        kind = gestalt.KINDS[int(g["gkind"][t, k])]
        cls = ",".join(f"{c}x{v}" if v > 1 else c for c, v in sorted(Counter(str(classes[n]) for n in members).items()))
        out.append((COLORS[i % len(COLORS)], uids,
                    f"G{i + 1} {kind}: {len(members)} ({cls}), direction {float(g['gdir'][t, k]):.0f} deg, "
                    f"speed {speed:.3f} box/frame"))
    return out


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--clip_actions", type=str,
                        default="Stop,Mov,MovAway,MovTow,Brake,Jun,Xing,XingFmLft,XingFmRht,Wait2X,TurLft,TurRht")
    parser.add_argument("--clip_mode", type=str, default="mdl_0.03", help="the rule_lab soft fit of the rule model")
    parser.add_argument("--clip_fpr", type=float, default=0.02, help="share of non-onset rows above the threshold")
    parser.add_argument("--num", type=int, default=6, help="clips per missed situation")
    parser.add_argument("--controls", type=int, default=3, help="detected_in_group clips")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--before_s", type=float, default=4.0)
    parser.add_argument("--after_s", type=float, default=1.0)
    parser.add_argument("--font_scale", type=float, default=1.7, help="text size in the clips")
    parser.add_argument("--no_render", action="store_true")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    if not getattr(args, "run_id", None):
        raise ValueError("--run_id is needed (a run with the gestalt atoms, e.g. the rule lab's)")
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    training_data = step05.load_training_data(cfg)
    layout = training_data.get("grid_layout", {})
    if "gestalt" not in layout:
        raise ValueError("the training data has no gestalt atoms: pass --gestalt True")
    cfg["mine_root_exclude"] = [layout[k] for k in ("shape", "gestalt") if k in layout]
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    head_mask = training_data["target_heads"].to(device)
    actions = set(args.clip_actions.split(","))
    heads = [h for h in torch.nonzero(head_mask).flatten().tolist() if names[h].split(":")[-1] in actions]
    possible = step05.class_possible(training_data, vocab, device)
    L, A = model.num_literals, model.num_atoms
    print(f"[gestalt_fail_clips] {len(heads)} heads: {', '.join(names[h] for h in heads)}")

    # The rule model (rule lab: relaxed miner, compact soft fit).
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
    gst_rules = defaultdict(list)
    for i in kept:
        text = lane_clips.rule_text(model, names, rules[i], gate[i].tolist())
        if "gst:" in text:
            gst_rules[names[rules[i]["head"]]].append({"weight": round(float(weight[i]), 2), "rule": text})

    # Events, thresholds, group context.
    eval_ids = val_ids[1::2]
    neg, events = collect_events(model, rs, args.clip_mode, training_data, eval_ids, heads, head_mask, possible, cfg)
    th = lane_clips.thresholds(neg, args.clip_fpr)
    atoms = utils_data.load_json(cfg["val_atom_file"])
    boxes_all = torch.load(cfg["val_box_file"], map_location="cpu")
    A_lab = len(utils_data.load_json(cfg["unique_atoms_file"]))
    video_cache = {}

    def video(vid):
        if vid not in video_cache:
            X = step05.build_video_tensor(atoms[vid], A_lab)
            boxes, classes, has_box = step05._agent_boxes(boxes_all, vid, atoms[vid], X)
            tube_of = {n: uid for uid, n in step05.video_agent_ids(atoms[vid]).items()}
            video_cache[vid] = (gestalt.groups(boxes, has_box, classes), classes, tube_of)
        return video_cache[vid]
    situations = defaultdict(list)
    counts = defaultdict(Counter)
    out_events = []
    for key, e in tqdm(events.items(), desc="Group context"):
        vid, agent, onset, h = key
        g, _classes, _tube = video(vid)
        t = max(onset - 1, 0)
        if t >= g["in_group"].shape[0] or agent >= g["in_group"].shape[1]:
            continue
        ctx = ("in_group" if bool(g["in_group"][t, agent]) else
               "near_group" if bool((g["near"][t, agent] & g["in_group"][t]).any()) else "no_group")
        missed = step07_paper._logit_bins(torch.tensor([e["score"]]))[0] < th[h]
        state = "missed" if bool(missed) else "detected"
        counts[names[h]][f"{state}_{ctx}"] += 1
        situations[f"{state}_{ctx}"].append(key)
        ex = lane_clips.explain(model, names, rules, gate, weight, by_head[h], e["act"], e["lits"]) if "act" in e else {}
        e["state"], e["ctx"], e["explain"] = state, ctx, ex
        out_events.append({"video": vid, "agent": agent, "onset_frame": onset, "head": names[h], "state": state,
                           "context": ctx, "score": round(e["score"], 4),
                           "threshold": lane_clips.bin_score(int(th[h])), **ex})
    total = Counter()
    for c in counts.values():
        total.update(c)
    print(f"\n[gestalt_fail_clips] events by state and group context (frame before the onset), all heads: "
          f"{dict(total)}")
    for state in ("missed", "detected"):
        for ctx in ("in_group", "near_group", "no_group"):
            n_ctx = total[f"missed_{ctx}"] + total[f"detected_{ctx}"]
            if state == "missed" and n_ctx:
                print(f"  {ctx:11s}: {n_ctx:6d} events, missed {total[f'missed_{ctx}'] / n_ctx:.0%}")
    print(f"  rules (|w| >= {rule_lab.DROP_WEIGHT}) using a gestalt atom: "
          f"{sum(len(v) for v in gst_rules.values())} of {len(kept)}")

    # Clips.
    dataset = Path(args.dataset_path)
    out_dir = args.output_dir / "gestalt_fail_clips"
    rng = random.Random(args.seed)
    rendered = []
    plan = [("missed_in_group", args.num), ("missed_near_group", args.num), ("detected_in_group", args.controls)]
    for situation, num in plan:
        keys = sorted(situations[situation])
        for key in rng.sample(keys, min(num, len(keys))):
            if args.no_render:
                break
            vid, agent, onset, h = key
            g, classes, tube_of = video(vid)
            e = events[key]
            ex = e["explain"]
            line = (f"rules hold: {ex['holding'][0]['rule']} (w {ex['holding'][0]['weight']})" if ex.get("holding")
                    else f"near miss: {ex['near_miss'][0]['rule']}" if ex.get("near_miss") else "no rule of the head")
            label = (f"[{situation}] {names[h]} agent {tube_of.get(agent)}; rules {e['score']:.3f} "
                     f"(thr {lane_clips.bin_score(int(th[h])):.3f})\n{line[:150]}")
            num_frames = len(residual_clips.sumo_render._video(dataset, vid)[1]["frames"])
            lo, hi = max(1, onset - int(args.before_s * 10)), min(num_frames, onset + int(args.after_s * 10))
            out_file = out_dir / situation / f"{names[h].replace(':', '_')}_{vid}_f{onset}.mp4"
            print(f"  rendering {out_file.relative_to(out_dir)} (frames {lo}-{hi})")
            with tempfile.TemporaryDirectory() as tmp:
                for i, f in enumerate(range(lo, hi + 1)):
                    dt = (f - onset) / 10
                    note = f"{label}\nonset at f{onset}: " + ("NOW" if f == onset else f"{dt:+.1f} s")
                    tf = min(f, g["in_group"].shape[0] - 1)
                    k = agent_group(g, tf, agent)                   # one group: the agent's own or the nearest
                    shown = group_captions(g, tf, classes, tube_of, only=k) if k is not None else []
                    state = ("agent IN the group" if k is not None and bool(g["in_group"][tf, agent])
                             else "agent NEXT TO the group" if k is not None else "no group this frame")
                    note = f"{note}  |  {state}"
                    residual_clips.sumo_render.render(dataset, vid, f, Path(tmp) / f"{i:05d}.png", note=note,
                                                      highlight=tube_of.get(agent), groups=shown,
                                                      font_scale=args.font_scale)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-framerate", "10",
                                "-i", str(Path(tmp) / "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_file)], check=True)
            rendered.append({"file": str(out_file.relative_to(out_dir)), "video": vid, "agent": agent,
                             "onset_frame": onset, "head": names[h], "situation": situation})
    out_dir.mkdir(parents=True, exist_ok=True)
    utils_data.save_json({"mode": args.clip_mode, "fpr": args.clip_fpr, "heads": [names[h] for h in heads],
                          "counts": {h: dict(c) for h, c in counts.items()}, "total": dict(total),
                          "gestalt_rules": gst_rules, "rendered": rendered, "events": out_events},
                         out_dir / "events.json")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
