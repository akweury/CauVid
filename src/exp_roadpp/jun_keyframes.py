"""Keyframes around a head's onsets: frames before, at and after the onset, one file per frame (2026-11-28).

Why: the user wants to see what happens before a car enters the junction (Car:Jun). Cases: --kf_num random
onsets of --kf_head on the first --kf_val_videos val videos (an onset: the head's first frame on the agent).
Per case one folder, one image per time in --kf_times (s from the onset), the agent outlined in red. The text
says only what matters (the head, the time to the onset, the agent's labels); the video, agent and onset frame
are in the folder name.

    ./run_roadpp_docker.sh --gpu 3 --jun_keyframes -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --kf_head Car:Jun
Writes <output_dir>/keyframes/<head>/<vid>_a<agent>_f<onset>/<k>_<time>s.png.
"""
import random
from pathlib import Path

import torch

from src.exp_roadpp import config, residual_clips, step05, utils_data, utils_pipe
from src.exp_roadpp.pair_table import collect
from src.exp_roadpp.rule_direct import head_rows


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--kf_head", type=str, default="Car:Jun")
    parser.add_argument("--kf_times", type=str, default="-4,-3,-2,-1,0,1,2",
                        help="s from the onset of the frames shown (at least 5)")
    parser.add_argument("--kf_num", type=int, default=5, help="onsets shown")
    parser.add_argument("--kf_val_videos", type=int, default=200)
    parser.add_argument("--kf_seed", type=int, default=0)
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id
    times = [float(x) for x in args.kf_times.split(",")]
    if len(times) < 5:
        raise ValueError("--kf_times: at least 5 frames")
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    training_data = step05.load_training_data(cfg)
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    h = names.index(args.kf_head)

    # the onsets: (video, agent, onset frame) of the hit rows (no literal needed: one column read)
    head_mask = training_data["target_heads"].to(device)
    possible = step05.class_possible(training_data, vocab, device)
    S, w = model.slot_size, int(cfg.get("clause_slots", 1))
    ids = sorted(training_data["val_videos"])[:args.kf_val_videos or None]
    d = collect(head_rows(model, training_data, ids, h, possible, head_mask, S, w, device, "val"), model,
                torch.zeros(1, dtype=torch.long, device=device))
    onsets = sorted({(int(v), int(n), int(t + f)) for v, n, t, f in d["meta"][d["y"]].tolist()})
    cases = random.Random(args.kf_seed).sample(onsets, min(args.kf_num, len(onsets)))
    print(f"[Keyframes] {len(onsets)} onsets of {args.kf_head} in {len(ids)} val videos; {len(cases)} shown")

    dataset = Path(args.dataset_path)
    atoms = utils_data.load_json(cfg["val_atom_file"])
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    label_names = [step05._atom_name(sig) for sig in unique_atoms]
    out_dir = args.output_dir / "keyframes" / args.kf_head.replace(":", "_")
    for v, n, onset in cases:
        vid = d["vids"][v]
        X = step05.build_video_tensor(atoms[vid], len(unique_atoms))
        tube_of = {k: uid for uid, k in step05.video_agent_ids(atoms[vid]).items()}
        num_frames = len(residual_clips.sumo_render._video(dataset, vid)[1]["frames"])
        case_dir = out_dir / f"{vid}_a{n}_f{onset}"
        for k, s in enumerate(times):
            f = onset + int(round(s * 10))
            if not 1 <= f <= num_frames:
                continue
            labs = [label_names[a].partition(":")[2] for a in torch.nonzero(X[min(f, X.shape[0] - 1), n]).flatten().tolist()]
            when = "onset" if s == 0 else f"{s:+.0f} s" if s == int(s) else f"{s:+.1f} s"
            title = f"{args.kf_head} {when}\ncar: {', '.join(labs) or 'not in view'}"
            residual_clips.sumo_render.render(dataset, vid, f, case_dir / f"{k}_{s:+g}s.png", highlight=tube_of.get(n),
                                              title=title)
        print(f"  {case_dir.relative_to(args.output_dir)}")
    print(f"[Keyframes] {out_dir}")


if __name__ == "__main__":
    main()
