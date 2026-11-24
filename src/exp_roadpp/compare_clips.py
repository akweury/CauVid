"""Transfer comparison clips: a target onset where a transferred rule holds (top) above a source onset where the
rule it came from holds (bottom), both aligned on the onset (2026-11-24, the user: to compare the transferred
rule with its origin, any pairing will do).

Rules: the rule lab's transfer component (--clip_variant tr or trp) on --clip_heads, as `head_clips` mines them
(the negated literals of the other agents off). Each transferred rule carries its source head and its literals
there ("source_literals"; for the pooled transfer the rule read back in the main source class).
Pairs: per target head, the transferred rules best first (held-out onsets, then train onsets); per rule up to
--per_rule pairs of
  - a target onset (val videos first, then train) at which the rule holds on one of its hit rows, and
  - a source onset (same) at which the source rule holds;
at most --pairs per head. Both clips span --before_s before the onset to --after_s after it, the agent outlined;
the text gives the rule, the agent's labels and the time to the onset.

    ./run_roadpp_docker.sh --gpu 0 --compare_clips -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True \\
        --run_id 20261123-egolay_lab --clip_heads Cyc:VehLane,Cyc:MovAway,Bus:MovRht
Writes <output_dir>/compare_clips/<head>/<k>_<target vid>_f<onset>__<source vid>_f<onset>.mp4 and pairs.json.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.exp_roadpp import (config, head_clips, mine_sweep, residual_clips, rule_lab, step05, step05_mine,  # noqa: E402
                            utils_data, utils_pipe)


@torch.no_grad()
def rule_onsets(model, training_data, video_ids, head, lits, head_mask, possible, cfg, limit):
    """[(video, agent, onset)] of the head's onsets at which the rule (literal list) holds on a hit row."""
    device, S, w = cfg["device"], model.slot_size, int(cfg.get("clause_slots", 1))
    out, seen = [], set()
    for vid in video_ids:
        can = possible(vid)
        for frames in training_data["frames"][vid].split(int(cfg.get("mine_chunk", 256))):
            rows = step05._video_rows(model, training_data, vid, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            keep, y = step05_mine._head_rows(rows, can, head_mask, S, w)
            hit = y[:, head]
            if not bool(hit.any()):
                continue
            lv = model.literal_values(rows["inputs"][hit]) > 0.5
            ok = lv[:, list(lits)].all(dim=1)
            t, n = rows["rows"][hit].T
            onset = t + rows["first"][hit][:, head]
            for i in torch.nonzero(ok).flatten().tolist():
                key = (vid, int(n[i]), int(onset[i]))
                if key not in seen:
                    seen.add(key)
                    out.append(key)
                    if len(out) >= limit:
                        return out
    return out


def stack(top, bottom, out):
    a, b = mpimg.imread(top), mpimg.imread(bottom)
    wd = max(a.shape[1], b.shape[1])
    pad = lambda x: np.pad(x, ((0, 0), (0, wd - x.shape[1]), (0, 0)), constant_values=1.0)
    sep = np.zeros((12, wd, a.shape[2]), dtype=a.dtype)
    mpimg.imsave(out, np.concatenate([pad(a), sep, pad(b)], axis=0))


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--clip_heads", type=str, default="Cyc:VehLane,Cyc:MovAway,Bus:MovRht")
    parser.add_argument("--clip_variant", type=str, default="tr", choices=("tr", "trp"))
    parser.add_argument("--lab_check", type=str, default="True")
    parser.add_argument("--pairs", type=int, default=8, help="comparison clips per target head")
    parser.add_argument("--per_rule", type=int, default=2, help="pairs per transferred rule")
    parser.add_argument("--before_s", type=float, default=4.0)
    parser.add_argument("--after_s", type=float, default=1.0)
    parser.add_argument("--font_scale", type=float, default=1.5)
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    if not getattr(args, "run_id", None):
        raise ValueError("--run_id is needed (the rule lab's run)")
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    cfg.update(head_clips.LAB_DEFAULTS)
    cfg["lab_check"] = args.lab_check == "True"
    device = cfg["device"]
    training_data = step05.load_training_data(cfg)
    layout = training_data.get("grid_layout", {})
    cfg["mine_root_exclude"] = [layout[k] for k in ("shape", "gestalt") if k in layout]
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    ext = head_clips.NoNegOthers(model)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    head_mask = training_data["target_heads"].to(device)
    cfg["lab_all_heads"] = torch.nonzero(head_mask).flatten().tolist()
    heads = [names.index(h) for h in args.clip_heads.split(",") if h]
    possible = step05.class_possible(training_data, vocab, device)
    train, val = mine_sweep.load_rows(cfg, training_data, model, head_mask, possible, args.output_dir / "mine_sweep")
    val_ids = list(training_data["val_videos"])
    check = mine_sweep.subset(val, set(range(0, len(val_ids), 2)))
    del val
    rules = rule_lab.mine_component(args.clip_variant, False, ext, names, heads, train, check, cfg, device)
    videos = val_ids + list(training_data["train_videos"])
    atoms = utils_data.load_json(cfg["val_atom_file"])
    atoms.update(utils_data.load_json(cfg["train_atom_file"]))
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    label_names = [step05._atom_name(sig) for sig in unique_atoms]
    split = {v: "val" for v in val_ids}
    dataset = Path(args.dataset_path)
    out_dir = args.output_dir / "compare_clips"
    cache, pairs = {}, []

    def text(lits):
        return " & ".join(step05_mine.literal_text(model, names, m) for m in lits)

    def frames_of(vid, agent, onset, title, tmp, tag):
        if vid not in cache:
            X = step05.build_video_tensor(atoms[vid], len(unique_atoms))
            tube_of = {n: uid for uid, n in step05.video_agent_ids(atoms[vid]).items()}
            cache[vid] = (X, tube_of, len(residual_clips.sumo_render._video(dataset, vid)[1]["frames"]))
        X, tube_of, num_frames = cache[vid]
        files = []
        for d in range(-int(args.before_s * 10), int(args.after_s * 10) + 1):
            f = min(max(1, onset + d), num_frames)
            tf = min(f, X.shape[0] - 1)
            labs = [label_names[a].partition(":")[2] for a in torch.nonzero(X[tf, agent]).flatten().tolist()]
            note = (f"{title}\nonset at f{onset} ({split.get(vid, 'train')} {vid}): "
                    + ("NOW" if d == 0 else f"{d / 10:+.1f} s") + f"  |  agent: {', '.join(labs) or '-'}")
            p = Path(tmp) / f"{tag}_{len(files):05d}.png"
            residual_clips.sumo_render.render(dataset, vid, f, p, note=note, highlight=tube_of.get(agent),
                                              font_scale=args.font_scale)
            files.append(p)
        return files

    for h in heads:
        rs = [r for r in rules if r["head"] == h and r.get("source_literals")]
        rs.sort(key=lambda r: -r.get("heldout_events", 0))
        print(f"\n[compare_clips] {names[h]}: {len(rs)} transferred rules")
        done = 0
        for r in tqdm(rs, desc=names[h]):
            if done >= args.pairs:
                break
            src_head = names.index(r["source_head"])
            tgt_on = rule_onsets(model, training_data, videos, h, r["literals"], head_mask, possible, cfg, args.per_rule)
            if not tgt_on:
                continue
            src_on = rule_onsets(model, training_data, videos, src_head, r["source_literals"], head_mask, possible, cfg,
                                 len(tgt_on))
            for (tv, ta, to), (sv, sa, so) in zip(tgt_on, src_on):
                if done >= args.pairs:
                    break
                out_file = out_dir / names[h].replace(":", "_") / f"{done:02d}_{tv}_f{to}__{sv}_f{so}.mp4"
                print(f"  rendering {out_file.relative_to(out_dir)}")
                with tempfile.TemporaryDirectory() as tmp:
                    top = frames_of(tv, ta, to, f"[TARGET, transferred: {r['transfer']}] {names[h]} <- {text(r['literals'])}",
                                    tmp, "t")
                    bot = frames_of(sv, sa, so, f"[SOURCE] {r['source_head']} <- {text(r['source_literals'])}", tmp, "s")
                    for i, (a, b) in enumerate(zip(top, bot)):
                        stack(a, b, Path(tmp) / f"c_{i:05d}.png")
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-framerate", "10",
                                    "-i", str(Path(tmp) / "c_%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_file)], check=True)
                pairs.append({"file": str(out_file.relative_to(out_dir)), "head": names[h], "rule": text(r["literals"]),
                              "source_head": r["source_head"], "source_rule": text(r["source_literals"]),
                              "transfer": r["transfer"], "target": [tv, ta, to], "source": [sv, sa, so]})
                done += 1
        print(f"[compare_clips] {names[h]}: {done} comparison clips")
    out_dir.mkdir(parents=True, exist_ok=True)
    utils_data.save_json({"variant": args.clip_variant, "pairs": pairs}, out_dir / "pairs.json")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
