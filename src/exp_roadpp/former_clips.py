"""Former clips: onsets the AtomFormer catches and the rule model misses, rendered for checking by eye (SUMO,
2026-11-27). Default heads: the Jun family, where the rules trail the Transformer most (v8: 0.22 vs 0.43).

Models: the AtomFormer of --lab_former (factor by default, trained by `rule_lab --lab_former`) and the rule model
of the run (the rule-lab .rules.pt files, `rule_paper.RuleScorer`, the selection's soft fit).

Events: every onset of the heads on the odd val videos. An event's hit rows are the at-risk rows of its agent
within clause_slots slots before the onset; per model the event's score is its highest score on them. Detected:
that score reaches the head's threshold, the score above which only --clip_fpr of the head's non-onset rows lie.
Groups: former_only (what to look at), both, rules_only, neither.

Per chosen event (--num former_only, --controls both, one neither per head):
  - <group>_<head>_<vid>_f<onset>.mp4: --before_s before the onset to --after_s after, the top and camera views of
    `synth/sumo/render`, the agent outlined in red; per frame the two models' scores and the time to the onset;
  - the same name .png: the timeline of both scores for the agent, the thresholds, the onset, and the rules'
    explanation at the frame of the Transformer's highest score: the head's rules that hold (by w x activation)
    or the strongest near misses with each literal + (true) / - (false), and the agent's own atoms then.
    ./run_roadpp_docker.sh --gpu 3 --former_clips -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --num_epochs 100 --lab_former factor
Writes <output_dir>/former_clips/<head>/... and events.json (every event: scores, group, explanation).
"""
import glob
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from src.exp_roadpp import (config, lane_clips, residual_clips, rule_lab, rule_paper, step05, step05_mine,
                            step07_paper, utils_data, utils_pipe)
from src.exp_roadpp.baselines import atom_former

JUN = "Car:Jun,MedVeh:Jun,LarVeh:Jun,Bus:Jun,Mobike:Jun,Cyc:Jun"


class HeadRules:
    """The rules of some heads from the merged rule files: per-row scores and activations."""

    def __init__(self, scorer, heads):
        sel = torch.isin(scorer.head, torch.tensor(heads, device=scorer.head.device))
        self.idx = torch.nonzero(sel).flatten()
        self.body, self.gate = scorer.body[sel], scorer.gate[sel]
        self.weight, self.head = scorer.weight[sel], scorer.head[sel]
        self.bias = scorer.bias
        self.model, self.derived = scorer.model, scorer.derived

    def __call__(self, inputs):
        """(lits (B, L+1) float, act (B, R), z (B, A) logits)."""
        lits = rule_lab.literal_rows(self.model, inputs, self.derived).float()
        act = rule_lab._soft_act(lits, self.body, self.gate)
        z = self.bias[None].expand(len(inputs), -1).clone().index_add(1, self.head, act * self.weight[None])
        return lits, act, z


def explain(hr, text, h, lits, act, top=3):
    """Rules of head h holding at the row (act >= 0.5, by w x act), else near misses (w > 0) with + / - literals."""
    r_h = torch.nonzero(hr.head == h).flatten()
    if not len(r_h):
        return {"holding": [], "near_miss": []}
    contrib = hr.weight[r_h] * act[r_h]
    order = r_h[torch.argsort(contrib, descending=True)].tolist()
    pad = lits.shape[0] - 1

    def fmt(r, marks):
        parts = []
        for m, g in zip(hr.body[r].tolist(), hr.gate[r].tolist()):
            if m >= pad or g < rule_lab.DROP_GATE:                  # the always-true padding literal
                continue
            parts.append((("+" if lits[m] > 0.5 else "-") if marks else "") + text(m))
        return {"rule": " & ".join(parts), "weight": round(float(hr.weight[r]), 2),
                "activation": round(float(act[r]), 2)}
    holding = [fmt(r, False) for r in order if act[r] >= 0.5 and hr.weight[r] > rule_lab.DROP_WEIGHT][:top]
    near = [r for r in r_h.tolist() if act[r] < 0.5 and hr.weight[r] > rule_lab.DROP_WEIGHT]
    near.sort(key=lambda r: -float(hr.weight[r] * act[r]))
    # The score's decomposition: logit = bias + positive rules + negative rules (w x activation summed).
    c = (hr.weight[r_h] * act[r_h])
    against = [fmt(r, False) for r in order[::-1] if float(hr.weight[r] * act[r]) <= -0.05][:top]
    return {"holding": holding, "near_miss": [fmt(r, True) for r in near[:top]], "against": against,
            "logit": {"bias": round(float(hr.bias[h]), 2), "positive": round(float(c[c > 0].sum()), 2),
                      "negative": round(float(c[c < 0].sum()), 2)}}


def own_atoms(inputs_row, model, names, slot=0):
    """The atoms the agent itself holds in a memory slot (0: the last second)."""
    A, J = model.num_atoms, model.num_memory_slots
    x = inputs_row[:model.num_scopes * J * A].view(model.num_scopes, J, A)[0, slot]
    return [names[a] for a in torch.nonzero(x > 0.5).flatten().tolist()]


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--lab_former", type=str, default="factor", choices=atom_former.ATTENTIONS)
    parser.add_argument("--former_dim", type=int, default=128)
    parser.add_argument("--former_layers", type=int, default=2)
    parser.add_argument("--former_heads", type=int, default=4)
    parser.add_argument("--former_dropout", type=float, default=0.1)
    parser.add_argument("--former_lr", type=float, default=5e-4)
    parser.add_argument("--clip_heads", type=str, default=JUN)
    parser.add_argument("--clip_rules", type=str, default="*_vt_p*", help="rule-lab result stem (rule_lab dir)")
    parser.add_argument("--clip_fpr", type=float, default=0.02, help="share of non-onset rows above the threshold")
    parser.add_argument("--clip_videos", type=int, default=0, help="odd val videos at most (0: all)")
    parser.add_argument("--num", type=int, default=4, help="former_only clips per head")
    parser.add_argument("--controls", type=int, default=1, help="'both' clips per head")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--before_s", type=float, default=4.0)
    parser.add_argument("--after_s", type=float, default=1.0)
    parser.add_argument("--no_render", action="store_true", help="only events.json and the timelines")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    for key in ("dim", "layers", "heads", "dropout", "lr"):
        cfg[f"former_{key}"] = getattr(args, f"former_{key}")
    cfg["former_attention"] = args.lab_former
    device = cfg["device"]
    lab_dir = args.output_dir / "rule_lab"
    training_data = step05.load_training_data(cfg)
    former = atom_former.load_former(training_data, cfg)
    if former is None:
        raise FileNotFoundError(f"no AtomFormer {atom_former.former_files(cfg)[0]}")
    rule_net = step05.build_rule_net(training_data, cfg, 1).to(device)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    derived = rule_lab.Derived(rule_net, names)
    ext = rule_lab.ExtModel(rule_net, derived, True)
    text = lambda m: step05_mine.literal_text(ext, names, m)
    files = sorted(glob.glob(str(lab_dir / f"{args.clip_rules}.rules.pt")))
    if not files:
        raise FileNotFoundError(f"no rule files {lab_dir}/{args.clip_rules}.rules.pt")
    head_ids = {n: i for i, n in enumerate(names)}
    heads = [head_ids[h] for h in args.clip_heads.split(",") if h in head_ids]
    hr = HeadRules(rule_paper.RuleScorer(files, rule_net, derived, 1, device), heads)
    print(f"[Former clips] {args.lab_former} vs rules: {len(heads)} heads, {len(hr.idx)} rules")
    hd = torch.tensor(heads, device=device)
    head_mask = training_data["target_heads"].to(device)
    possible = step05.class_possible(training_data, vocab, device)
    S, w = former.slot_size, int(cfg.get("clause_slots", 1))
    k_win = min(range(len(former.windows)), key=lambda i: abs(former.windows[i] - w * S))
    A, NB = former.num_atoms, step07_paper.NB
    val_ids = list(training_data["val_videos"])[1::2]
    if args.clip_videos:
        val_ids = val_ids[:args.clip_videos]

    def scores(rows):
        pF = former.window_probs(former.forward_rows(rows))[:, k_win]
        lits, act, z = hr(rows["inputs"])
        return pF, torch.sigmoid(z), lits, act

    # Events and the non-onset score histograms.
    neg = {k: torch.zeros(A * NB, device=device) for k in ("former", "rules")}
    events = {}
    with torch.no_grad():
        for vid in tqdm(val_ids, desc="Events (odd val)"):
            can = possible(vid)
            for frames in training_data["frames"][vid].split(128):
                rows = step05._video_rows(former, training_data, vid, frames, device)
                if len(rows["inputs"]) == 0:
                    continue
                keep, y = step05_mine._head_rows(rows, can, head_mask, S, w)
                keep, y = keep[:, hd], y[:, hd]
                if not keep.any():
                    continue
                pF, pR, lits, act = scores(rows)
                p = {"former": pF[:, hd], "rules": pR[:, hd]}
                for k, v in p.items():
                    b = hd[None].expand_as(v) * NB + step07_paper._logit_bins(v)
                    neg[k].add_(torch.bincount(b[keep & ~y], minlength=A * NB).float())
                if not y.any():
                    continue
                b_i, j_i = torch.nonzero(y, as_tuple=True)
                t, n = rows["rows"][b_i].T
                onset = t + rows["first"][b_i, hd[j_i]]
                for i in range(len(b_i)):
                    key = (vid, int(n[i]), int(onset[i]), heads[int(j_i[i])])
                    pf, pr = float(p["former"][b_i[i], j_i[i]]), float(p["rules"][b_i[i], j_i[i]])
                    e = events.setdefault(key, {"former": -1.0, "rules": -1.0})
                    e["rules"] = max(e["rules"], pr)
                    if pf > e["former"]:
                        e.update(former=pf, t=int(t[i]),
                                 explain=explain(hr, text, heads[int(j_i[i])], lits[b_i[i]], act[b_i[i]]),
                                 own=own_atoms(rows["inputs"][b_i[i]], former, names))
    neg = {k: v.view(A, NB).cpu() for k, v in neg.items()}
    th = {k: lane_clips.thresholds(v, args.clip_fpr) for k, v in neg.items()}
    thr = {h: {k: lane_clips.bin_score(int(th[k][h])) for k in th} for h in heads}
    groups = defaultdict(list)
    for key, e in events.items():
        h = key[3]
        g = {(True, False): "former_only", (True, True): "both", (False, True): "rules_only"}.get(
            (e["former"] >= thr[h]["former"], e["rules"] >= thr[h]["rules"]), "neither")
        e["group"] = g
        groups[(h, g)].append(key)
    print(f"\n[Former clips] odd val videos ({len(val_ids)}), thresholds at {args.clip_fpr:.0%} of non-onset rows:")
    print(f"  {'head':<14}{'events':>7}{'both':>6}{'former_only':>12}{'rules_only':>11}{'neither':>8}"
          f"{'thr former':>11}{'thr rules':>10}")
    for h in heads:
        c = {g: len(groups[(h, g)]) for g in ("both", "former_only", "rules_only", "neither")}
        print(f"  {names[h]:<14}{sum(c.values()):>7}{c['both']:>6}{c['former_only']:>12}{c['rules_only']:>11}"
              f"{c['neither']:>8}{thr[h]['former']:>11.3f}{thr[h]['rules']:>10.3f}")

    # Clips and timelines.
    dataset = Path(args.dataset_path)
    atoms = utils_data.load_json(cfg["val_atom_file"])
    rng = random.Random(args.seed)
    out_dir = args.output_dir / "former_clips"
    rendered = []
    for h in heads:
        chosen = []
        for g, num in (("former_only", args.num), ("both", args.controls), ("neither", 1)):
            keys = sorted(groups[(h, g)])
            chosen += [(g, k) for k in rng.sample(keys, min(num, len(keys)))]
        for g, key in chosen:
            vid, agent, onset, _ = key
            e = events[key]
            n_frames = len(residual_clips.sumo_render._video(dataset, vid)[1]["frames"])
            lo, hi = max(1, onset - int(args.before_s * 10)), min(n_frames, onset + int(args.after_s * 10))
            curve = agent_curve(former, hr, training_data, vid, agent, h, lo, hi, scores, device)
            stem = out_dir / names[h].replace(":", "_") / f"{g}_{vid}_f{onset}"
            tube = {n: uid for uid, n in step05.video_agent_ids(atoms[vid]).items()}.get(agent)
            plot_timeline(curve, onset, thr[h], e, names[h], g, f"{stem}.png")
            if not args.no_render:
                print(f"  rendering {stem.relative_to(out_dir)}.mp4 (frames {lo}-{hi})")
                ex = e["explain"]
                line = (f"rule holds: {ex['holding'][0]['rule']}" if ex.get("holding")
                        else f"near miss: {ex['near_miss'][0]['rule']}" if ex.get("near_miss") else "no rule")
                label = f"[{g}] {names[h]} agent {tube}\n{line[:140]}"
                render(dataset, vid, onset, lo, hi, f"{stem}.mp4", label, tube, curve, thr[h])
            rendered.append({"file": f"{stem.relative_to(out_dir)}", "video": vid, "agent": agent, "tube": tube,
                             "onset_frame": onset, "head": names[h], "group": g})
    out_dir.mkdir(parents=True, exist_ok=True)
    utils_data.save_json({"attention": args.lab_former, "fpr": args.clip_fpr, "rule_files": [Path(f).name for f in files],
                          "thresholds": {names[h]: thr[h] for h in heads}, "rendered": rendered,
                          "events": [{"video": k[0], "agent": k[1], "onset_frame": k[2], "head": names[k[3]],
                                      "group": e["group"], "former": round(e["former"], 4),
                                      "rules": round(e["rules"], 4), "frame_former_max": e.get("t"),
                                      "own_atoms_last_s": e.get("own"), **(e.get("explain") or {})}
                                     for k, e in events.items()]}, out_dir / "events.json")
    print(f"  -> {out_dir}")


@torch.no_grad()
def agent_curve(former, hr, training_data, vid, agent, h, lo, hi, scores, device):
    """{frame: (former, rules)} of the agent's at-risk-or-not rows in frames lo..hi (frames it is not seen: none)."""
    frames = torch.arange(lo, hi + 1)
    rows = step05._video_rows(former, training_data, vid, frames, device)
    out = {}
    if len(rows["inputs"]) == 0:
        return out
    pF, pR, _l, _a = scores(rows)
    sel = torch.nonzero(rows["rows"][:, 1] == agent).flatten()
    for i in sel.tolist():
        out[int(rows["rows"][i, 0])] = (float(pF[i, h]), float(pR[i, h]))
    return out


def plot_timeline(curve, onset, thr, e, head, group, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fs = sorted(curve)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if fs:
        t = [(f - onset) / 10 for f in fs]
        ax.plot(t, [curve[f][0] for f in fs], color="#2a78d6", lw=2, label="Transformer")
        ax.plot(t, [curve[f][1] for f in fs], color="#eb6834", lw=2, label="rules")
    ax.axhline(thr["former"], color="#2a78d6", ls="--", lw=1, label=f"thr Transformer {thr['former']:.2f}")
    ax.axhline(thr["rules"], color="#eb6834", ls="--", lw=1, label=f"thr rules {thr['rules']:.2f}")
    ax.axvline(0, color="#0b0b0b", lw=1)
    ax.axvspan(-2, 0, color="#e6e5e0", alpha=0.6, lw=0)
    ax.set_xlabel("seconds to the onset (grey: the 2 s window)"); ax.set_ylabel(f"P({head} within 2 s)")
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"[{group}] {head}", loc="left", fontsize=10)
    ex = e.get("explain") or {}
    lines = [f"at f{e.get('t')} (Transformer max {e['former']:.2f}; rules max {e['rules']:.2f}):"]
    lines += [f"  holds  w {r['weight']:+.2f}: {r['rule']}" for r in ex.get("holding", [])]
    lines += [f"  near   w {r['weight']:+.2f} act {r['activation']:.2f}: {r['rule']}" for r in ex.get("near_miss", [])]
    lines += [f"  against w {r['weight']:+.2f} act {r['activation']:.2f}: {r['rule']}" for r in ex.get("against", [])]
    lg = ex.get("logit")
    if lg:
        lines.append(f"rules' logit = bias {lg['bias']:+.2f} + positive rules {lg['positive']:+.2f} "
                     f"+ negative rules {lg['negative']:+.2f} = {lg['bias'] + lg['positive'] + lg['negative']:+.2f}")
    own = e.get("own") or []
    lines.append("own atoms (last s): " + ", ".join(own[:18]) + (" ..." if len(own) > 18 else ""))
    fig.text(0.01, -0.02, "\n".join(l[:170] for l in lines), fontsize=7, va="top", family="monospace")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def render(dataset, vid, onset, first, last, out_file, label, tube, curve, thr, fps=10):
    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(range(first, last + 1)):
            dt = (frame - onset) / 10
            sc = curve.get(frame)
            s = (f"Transformer {sc[0]:.2f} (thr {thr['former']:.2f}) | rules {sc[1]:.2f} (thr {thr['rules']:.2f})"
                 if sc else "agent not scored in this frame")
            note = f"{label}\n{s}\nonset at f{onset}: " + ("NOW" if frame == onset else f"{dt:+.1f} s")
            residual_clips.sumo_render.render(dataset, vid, frame, Path(tmp) / f"{i:05d}.png", note=note,
                                              highlight=tube)
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(Path(tmp) / "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_file)], check=True)


if __name__ == "__main__":
    main()
