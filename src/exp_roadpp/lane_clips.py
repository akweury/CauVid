"""Lane clips: the lane onsets the rule model misses and the OnsetMLP catches, rendered for checking by eye
(SUMO, 2026-11-22).

The heads are those of --clip_actions (default VehLane, IncomLane, OutgoLane, MovRht), where the rules trail
the MLP most (rule lab, 1ca554 val: VehLane 0.38 vs 0.61, MovRht 0.18 vs 0.33, IncomLane 0.40 vs 0.50).

Rule model: the lab's relaxed miner (`rule_lab.mine_component`) on those heads, scored by the sparse soft
logistic fit (`rule_lab.fit_soft`, mode --clip_mode, default sparse_c). Both are fit on the train sample.

Events: every onset of those heads on the odd val videos, the videos the lab evaluates on (the even ones are
the miner's check). An event's hit rows are the at-risk rows of its agent within clause_slots slots before the
onset, as in mining. Per scorer, the event's score is its highest score on those rows. The event counts as
detected if that score reaches the head's threshold: the score above which only --clip_fpr of the head's
non-onset rows lie. Groups:
  mlp_only  the MLP detects it, the rules do not (what to look at);
  both / rules_only / neither  for comparison.

Clips: per action, --num mlp_only events, --controls "both" events and one "neither" event, chosen at random.
Each clip spans --before_s before the onset to --after_s after it, with the top and camera views of
`synth/sumo/render` and the event's agent outlined in red. The title gives:
  - the MLP and rule scores with their thresholds;
  - the rules that hold at the frame of the MLP's highest score, or the strongest near miss with each
    literal true (+) or false (-).

    ./run_roadpp_docker.sh --gpu 0 --lane_clips -- --exp sumo_maps_v7 --machine dgx-jing \\
        --run_id 20261121-173934_dgx-jing_1ca554
Writes <output_dir>/lane_clips/<action>/<group>_<head>_<vid>_f<onset>.mp4 and events.json (every event: its
scores, group, frame and rule explanation; the thresholds; the rules with their weights and gates).
"""
import random
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from src.exp_roadpp import (config, mine_sweep, residual_clips, rule_lab, step05, step05_mine, step07_paper,
                            utils_data, utils_pipe)


def rule_text(model, names, rule, gates=None, truth=None, min_gate=rule_lab.DROP_GATE):
    """'A & B & C' of the literals kept (gate >= min_gate), each marked + / - by truth (a row's literal values)."""
    parts = []
    for i, m in enumerate(rule["literals"]):
        if gates is not None and gates[i] < min_gate:
            continue
        t = step05_mine.literal_text(model, names, m)
        parts.append(t if truth is None else f"{'+' if truth[m] else '-'}{t}")
    return " & ".join(parts)


def explain(model, names, rules, gate, weight, head_rules, act, lits_row, top=2):
    """The rules of the head holding at a row (activation >= 0.5, by weight x activation), else the strongest
    near misses (positive weight, by weight x activation) with each literal marked + / -."""
    idx = torch.tensor(head_rules, dtype=torch.long)
    if not len(idx):
        return {"holding": [], "near_miss": []}
    contrib = weight[idx] * act[idx]
    order = idx[torch.argsort(contrib, descending=True)].tolist()
    holding = [r for r in order if act[r] >= 0.5 and weight[r] > rule_lab.DROP_WEIGHT][:top]
    near = [r for r in order if act[r] < 0.5 and weight[r] > rule_lab.DROP_WEIGHT][:top]
    fmt = lambda r, truth=None: {"rule": rule_text(model, names, rules[r], gate[r].tolist(), truth),
                                 "weight": round(float(weight[r]), 2), "activation": round(float(act[r]), 2)}
    return {"holding": [fmt(r) for r in holding], "near_miss": [fmt(r, lits_row) for r in near]}


@torch.no_grad()
def collect_events(model, mlp, rs, mode, training_data, video_ids, heads, head_mask, possible, cfg, k_mlp):
    """Per head the non-onset score histograms {"mlp" | "rules": (A, NB)} and the events {(vid, agent, onset, head):
    {"mlp", "rules": max score, "t": frame of the MLP's max, "act": rule activations there, "lits": literals}}."""
    device, S, w = cfg["device"], model.slot_size, int(cfg.get("clause_slots", 1))
    A, NB = model.num_atoms, step07_paper.NB
    gate = rs.soft[mode][0]
    neg = {k: torch.zeros(A * NB, device=device) for k in ("mlp", "rules")}
    events = {}
    hd = torch.tensor(heads, device=device)
    for video_id in tqdm(video_ids, desc="Lane events (val)"):
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
            inputs = rows["inputs"][sel]
            lits = rule_lab._with_true(model.literal_values(inputs) > 0.5)
            p = {"mlp": mlp.window_probs(mlp(inputs)[0])[:, k_mlp][:, hd],
                 "rules": rs.scores(lits)[mode][:, hd]}
            keep, y = keep[sel], y[sel]
            for k, v in p.items():
                b = hd[None].expand_as(v) * NB + step07_paper._logit_bins(v)
                neg[k].add_(torch.bincount(b[keep & ~y], minlength=A * NB).float())
            if not y.any():
                continue
            b_idx, j_idx = torch.nonzero(y, as_tuple=True)
            act = rule_lab._soft_act(lits[b_idx].float(), rs.body, gate)            # (E_rows, R)
            t, n = rows["rows"][sel][b_idx].T
            onset = t + rows["first"][sel][b_idx, hd[j_idx]]
            for i in range(len(b_idx)):
                key = (video_id, int(n[i]), int(onset[i]), heads[int(j_idx[i])])
                pm, pr = float(p["mlp"][b_idx[i], j_idx[i]]), float(p["rules"][b_idx[i], j_idx[i]])
                e = events.get(key)
                if e is None:
                    e = events[key] = {"mlp": -1.0, "rules": -1.0}
                e["rules"] = max(e["rules"], pr)
                if pm > e["mlp"]:
                    e.update(mlp=pm, t=int(t[i]), act=act[i].cpu(), lits=lits[b_idx[i]].cpu())
    return {k: v.view(A, NB).cpu() for k, v in neg.items()}, events


def thresholds(neg, fpr):
    """(A,) lowest score bin per head whose share of non-onset rows at or above it is <= fpr."""
    tail = neg.flip(1).cumsum(1).flip(1) / neg.sum(1, keepdim=True).clamp(min=1)
    ok = tail <= fpr
    first = torch.where(ok.any(1), ok.float().argmax(1), torch.full((neg.shape[0],), neg.shape[1] - 1))
    return first


def bin_score(b):
    """The score at the low edge of a `step07_paper._logit_bins` bin."""
    z = b / (step07_paper.NB - 1) * 28 - 14
    return float(torch.sigmoid(torch.tensor(z)))


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--clip_actions", type=str, default="VehLane,IncomLane,OutgoLane,MovRht")
    parser.add_argument("--clip_mode", type=str, default="sparse_c", help="the rule_lab soft mode of the rule model")
    parser.add_argument("--clip_fpr", type=float, default=0.02, help="share of non-onset rows above the threshold")
    parser.add_argument("--num", type=int, default=6, help="mlp_only clips per action")
    parser.add_argument("--controls", type=int, default=2, help="'both' clips per action")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--before_s", type=float, default=4.0)
    parser.add_argument("--after_s", type=float, default=1.0)
    parser.add_argument("--no_render", action="store_true", help="only the events and their explanations")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    if not getattr(args, "run_id", None):
        raise ValueError("--run_id is needed (the run's OnsetMLP)")
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    training_data = step05.load_training_data(cfg)
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    mlp = step05.load_onset_mlp(training_data, cfg)
    window = int(cfg.get("clause_slots", 1)) * model.slot_size
    k_mlp = min(range(len(model.windows)), key=lambda i: abs(model.windows[i] - window))
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    head_mask = training_data["target_heads"].to(device)
    actions = set(args.clip_actions.split(","))
    heads = [h for h in torch.nonzero(head_mask).flatten().tolist() if names[h].split(":")[-1] in actions]
    possible = step05.class_possible(training_data, vocab, device)
    L, A = model.num_literals, model.num_atoms
    print(f"[lane_clips] {len(heads)} heads: {', '.join(names[h] for h in heads)}")

    # The rule model of those heads (rule lab: relaxed miner, sparse soft logistic fit) on the train sample.
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

    # Events on the odd val videos, thresholds at --clip_fpr of the non-onset rows.
    eval_ids = val_ids[1::2]
    neg, events = collect_events(model, mlp, rs, args.clip_mode, training_data, eval_ids, heads, head_mask, possible,
                                 cfg, k_mlp)
    th = {k: thresholds(v, args.clip_fpr) for k, v in neg.items()}
    groups = defaultdict(list)
    out_events = []
    for key, e in events.items():
        vid, agent, onset, h = key
        hit_m = step07_paper._logit_bins(torch.tensor([e["mlp"]]))[0] >= th["mlp"][h]
        hit_r = step07_paper._logit_bins(torch.tensor([e["rules"]]))[0] >= th["rules"][h]
        g = {(True, False): "mlp_only", (True, True): "both", (False, True): "rules_only"}.get(
            (bool(hit_m), bool(hit_r)), "neither")
        groups[(names[h].split(":")[-1], g)].append(key)
        ex = explain(model, names, rules, gate, weight, by_head[h], e["act"], e["lits"]) if "act" in e else {}
        e["group"], e["explain"] = g, ex
        out_events.append({"video": vid, "agent": agent, "onset_frame": onset, "head": names[h], "group": g,
                           "mlp": round(e["mlp"], 4), "rules": round(e["rules"], 4), "frame_mlp_max": e.get("t"),
                           **ex})
    print(f"\n[lane_clips] odd val videos, threshold per head at {args.clip_fpr:.0%} of the non-onset rows:")
    print(f"  {'head':<20}{'events':>7}{'both':>6}{'mlp_only':>9}{'rules_only':>11}{'neither':>8}"
          f"{'thr mlp':>9}{'thr rules':>10}")
    for h in heads:
        a = names[h].split(":")[-1]
        cnt = {g: sum(1 for k in groups[(a, g)] if k[3] == h) for g in ("both", "mlp_only", "rules_only", "neither")}
        total = sum(cnt.values())
        if total:
            print(f"  {names[h]:<20}{total:>7}{cnt['both']:>6}{cnt['mlp_only']:>9}{cnt['rules_only']:>11}"
                  f"{cnt['neither']:>8}{bin_score(int(th['mlp'][h])):>9.3f}{bin_score(int(th['rules'][h])):>10.3f}")

    # Clips.
    dataset = Path(args.dataset_path)
    atoms = utils_data.load_json(cfg["val_atom_file"])
    rng = random.Random(args.seed)
    out_dir = args.output_dir / "lane_clips"
    rendered = []
    for a in sorted(actions):
        chosen = []
        for g, num in (("mlp_only", args.num), ("both", args.controls), ("neither", 1)):
            keys = sorted(groups[(a, g)])
            chosen += [(g, k) for k in rng.sample(keys, min(num, len(keys)))]
        for g, (vid, agent, onset, h) in chosen:
            if args.no_render:
                break
            tube = {n: uid for uid, n in step05.video_agent_ids(atoms[vid]).items()}.get(agent)
            e = events[(vid, agent, onset, h)]
            ex = e["explain"]
            line = (f"rules hold: {ex['holding'][0]['rule']} (w {ex['holding'][0]['weight']})" if ex.get("holding")
                    else f"near miss: {ex['near_miss'][0]['rule']}" if ex.get("near_miss") else "no rule of the head")
            label = (f"[{g}] {names[h]} agent {tube}; MLP {e['mlp']:.2f} (thr {bin_score(int(th['mlp'][h])):.2f}), "
                     f"rules {e['rules']:.2f} (thr {bin_score(int(th['rules'][h])):.2f})\n{line[:150]}")
            num_frames = len(residual_clips.sumo_render._video(dataset, vid)[1]["frames"])
            lo, hi = max(1, onset - int(args.before_s * 10)), min(num_frames, onset + int(args.after_s * 10))
            out_file = out_dir / a / f"{g}_{names[h].replace(':', '_')}_{vid}_f{onset}.mp4"
            print(f"  rendering {out_file.relative_to(out_dir)} (frames {lo}-{hi})")
            render_clip(dataset, vid, onset, lo, hi, out_file, label, tube)
            rendered.append({"file": str(out_file.relative_to(out_dir)), "video": vid, "agent": agent, "tube": tube,
                             "onset_frame": onset, "head": names[h], "group": g})
    rule_list = [{"head": names[r["head"]], "weight": round(float(weight[i]), 3),
                  "rule": rule_text(model, names, r, gate[i].tolist())}
                 for i, r in enumerate(rules) if abs(float(weight[i])) >= rule_lab.DROP_WEIGHT]
    out_dir.mkdir(parents=True, exist_ok=True)
    utils_data.save_json({"heads": [names[h] for h in heads], "fpr": args.clip_fpr, "mode": args.clip_mode,
                          "thresholds": {names[h]: {k: bin_score(int(th[k][h])) for k in th} for h in heads},
                          "rules": sorted(rule_list, key=lambda r: (r["head"], -r["weight"])),
                          "rendered": rendered, "events": out_events}, out_dir / "events.json")
    print(f"  -> {out_dir}")


def render_clip(dataset, vid, onset, first_frame, last_frame, out_file, label, tube, fps=10):
    """`residual_clips.render_clip` with the event's agent outlined (`render(..., highlight=tube)`)."""
    import shutil
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(range(first_frame, last_frame + 1)):
            dt = (frame - onset) / 10
            note = f"{label}\nonset at f{onset}: " + ("NOW" if frame == onset else f"{dt:+.1f} s")
            residual_clips.sumo_render.render(dataset, vid, frame, Path(tmp) / f"{i:05d}.png", note=note,
                                              highlight=tube)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(Path(tmp) / "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_file)], check=True)


if __name__ == "__main__":
    main()
