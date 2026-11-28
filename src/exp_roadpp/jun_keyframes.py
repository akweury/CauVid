"""Keyframes of a head's onsets and of a rule's "never" negatives: before / at / after (2026-11-28).

Why: the high-recall rules of Car:Jun fire mostly on cars that never enter the junction (`pair_table`
fp_classes: ~3/4 of the negatives); the user wants to see what tells them from the cars that do.
Two groups on --kf_val_videos val videos:
  onset   onsets of the head (the agent enters the junction); reference frame = the onset;
  never   agents with no onset of the head on which --kf_rule holds; reference = the middle frame of the rule.
Per case one image: three frames (reference - --kf_dt s, reference, reference + --kf_dt s) stacked, the agent
outlined in red. The text says only what matters (group, time, the agent's labels, whether the rule holds);
the video, agent and frame are in the file name.

    ./run_roadpp_docker.sh --gpu 3 --jun_keyframes -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --kf_head Car:Jun
Writes <output_dir>/keyframes/<head>/<group>_<vid>_a<agent>_f<frame>.png.
"""
import random
import tempfile
from pathlib import Path

import torch
from PIL import Image

from src.exp_roadpp import config, residual_clips, step05, step05_mine, utils_data, utils_pipe
from src.exp_roadpp.pair_table import collect
from src.exp_roadpp.rule_direct import head_rows


def stack(files, out_file):
    """The images one under the other."""
    ims = [Image.open(f).convert("RGB") for f in files]
    w = max(i.width for i in ims)
    canvas = Image.new("RGB", (w, sum(i.height for i in ims)), "white")
    y = 0
    for im in ims:
        canvas.paste(im, (0, y))
        y += im.height
    out_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_file)


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--kf_head", type=str, default="Car:Jun")
    parser.add_argument("--kf_rule", type=str, default="!Car:Stop(own)@1 & veh:Jun(other)@1",
                        help="the rule whose 'never' negatives are shown (literals ' & ')")
    parser.add_argument("--kf_num", type=int, default=4, help="cases per group")
    parser.add_argument("--kf_dt", type=float, default=2.0, help="s before / after the reference frame")
    parser.add_argument("--kf_val_videos", type=int, default=200)
    parser.add_argument("--kf_seed", type=int, default=0)
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    training_data = step05.load_training_data(cfg)
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    h = names.index(args.kf_head)
    lit_of = {step05_mine.literal_text(model, names, m): m for m in range(model.num_literals)}
    rule = [x.strip() for x in args.kf_rule.split("&")]
    missing = [t for t in rule if t not in lit_of]
    if missing:
        raise ValueError(f"unknown literals {missing}")
    cols = torch.tensor([lit_of[t] for t in rule], device=device)

    head_mask = training_data["target_heads"].to(device)
    possible = step05.class_possible(training_data, vocab, device)
    S, w = model.slot_size, int(cfg.get("clause_slots", 1))
    ids = sorted(training_data["val_videos"])[:args.kf_val_videos or None]
    d = collect(head_rows(model, training_data, ids, h, possible, head_mask, S, w, device, "val"), model, cols)
    holds = d["F"].all(dim=1)
    meta, y = d["meta"], d["y"]

    # onsets: (vid index, agent, onset frame); never: agents without an onset where the rule holds
    onsets = sorted({(int(v), int(n), int(t + f)) for v, n, t, f in meta[y].tolist()})
    with_onset = {(v, n) for v, n, _ in onsets}
    never = {}
    for (v, n, t, _f), hd in zip(meta.tolist(), holds.tolist()):
        if hd and (v, n) not in with_onset:
            never.setdefault((v, n), []).append(t)
    rule_at = {}
    for (v, n, t, _f), hd in zip(meta.tolist(), holds.tolist()):
        if hd:
            rule_at.setdefault((v, n), set()).add(t)
    rng = random.Random(args.kf_seed)
    cases = [("onset", v, n, t) for v, n, t in rng.sample(onsets, min(args.kf_num, len(onsets)))]
    nv = sorted(never)
    cases += [("never", v, n, sorted(never[k])[len(never[k]) // 2])
              for k in rng.sample(nv, min(args.kf_num, len(nv))) for v, n in [k]]
    print(f"[Keyframes] {len(onsets)} onsets, {len(never)} 'never' agents of the rule; {len(cases)} cases")

    dataset = Path(args.dataset_path)
    atoms = utils_data.load_json(cfg["val_atom_file"])
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    label_names = [step05._atom_name(sig) for sig in unique_atoms]
    out_dir = args.output_dir / "keyframes" / args.kf_head.replace(":", "_")
    dt = int(round(args.kf_dt * 10))
    for group, v, n, ref in cases:
        vid = d["vids"][v]
        X = step05.build_video_tensor(atoms[vid], len(unique_atoms))
        tube_of = {k: uid for uid, k in step05.video_agent_ids(atoms[vid]).items()}
        num_frames = len(residual_clips.sumo_render._video(dataset, vid)[1]["frames"])
        files = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, (when, f) in enumerate((("before", ref - dt), ("at", ref), ("after", ref + dt))):
                f = min(max(f, 1), num_frames)
                labs = [label_names[a].partition(":")[2] for a in torch.nonzero(X[min(f, X.shape[0] - 1), n]).flatten().tolist()]
                head_txt = f"{args.kf_head} " + ("onset" if group == "onset" else "never happens")
                time_txt = {"before": f"{-args.kf_dt:+.1f} s", "at": "onset" if group == "onset" else "rule holds",
                            "after": f"{args.kf_dt:+.1f} s"}[when]
                rule_txt = "rule holds" if f in rule_at.get((v, n), ()) else "rule off"
                title = f"{head_txt} | {time_txt} | {rule_txt}\ncar: {', '.join(labs) or 'not in view'}"
                files.append(Path(tmp) / f"{i}.png")
                residual_clips.sumo_render.render(dataset, vid, f, files[-1], highlight=tube_of.get(n), title=title)
            out_file = out_dir / f"{group}_{vid}_a{n}_f{ref}.png"
            stack(files, out_file)
        print(f"  {out_file.relative_to(args.output_dir)}")
    print(f"[Keyframes] {out_dir}")


if __name__ == "__main__":
    main()
