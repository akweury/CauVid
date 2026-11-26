"""The rule lab's rules in the paper protocol (2026-11-26): step 07's numbers at fixed false-alarm budgets, for the
rule lab scorer and the OnsetMLP(s), without a trained TemporalRuleNet.

Rules: the `.rules.pt` files rule_lab writes next to its result json (one per head part, `--lab_head_part`): per head
the chosen variant's rules in the selection's soft fit (body literals, gates, weights, the head's bias). The parts
are merged into one scorer p = sigmoid(b_h + sum_r w_r a_r) over every head (heads without rules: 0).
MLPs: the run's OnsetMLP and, with --paper_mlps "2048:0.2,2048:0" (hidden size : dropout), the stronger ones
trained by `rule_lab --lab_mlp_only` (`--mlp_hidden_size`, `--mlp_dropout`), if their model files exist.
Protocol (`step07_paper`): the counted heads (rare, regular, trivial; no noise heads), window W = 20 frames, per
scorer the threshold of each false-alarm budget (alarms per minute, all agents and heads) chosen on all val videos,
then alarms, detected events, recall and lead on all test videos; plus the cell AP.

    ./run_roadpp_docker.sh --gpu 7 --rule_paper -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --paper_rules "*_vt_p*.rules.pt" --paper_mlps 2048:0.2,2048:0
Writes <output_dir>/rule_lab/paper_<tag>.json (as step 07's paper_metrics.json) and prints step 07's summary.
"""
import glob
import os

import torch

from src.exp_roadpp import config, rule_lab, step05, step06, step07_paper, utils_data, utils_pipe


class RuleScorer:
    """(B, W, A) scores of packed inputs from the merged rule files (the same score in every window)."""

    def __init__(self, files, model, derived, num_windows, device):
        parts = [torch.load(f, map_location="cpu") for f in files]
        L, A = parts[0]["num_literals"], parts[0]["num_atoms"]
        if any(p["num_literals"] != L or p["num_atoms"] != A for p in parts):
            raise ValueError("the rule files were made with different literal sets")
        K = max(p["body"].shape[1] for p in parts)
        pad = lambda x, v: torch.nn.functional.pad(x, (0, K - x.shape[1]), value=v)
        self.body = torch.cat([pad(p["body"], L) for p in parts]).to(device)
        self.gate = torch.cat([pad(p["gate"], 1.0) for p in parts]).to(device)
        self.weight = torch.cat([p["weight"] for p in parts]).to(device)
        self.head = torch.cat([p["head"] for p in parts]).to(device)
        self.bias = torch.full((A,), -20.0, device=device)
        self.heads = sorted({h for p in parts for h in p["heads"]})
        for p in parts:
            for h, b in p["bias"].items():
                self.bias[int(h)] = b
        self.model, self.derived, self.W = model, derived, num_windows
        self.modes = sorted({p["mode"] for p in parts})
        print(f"[Rule paper] rules: {len(files)} files, {len(self.head)} rules, {len(self.heads)} heads, "
              f"fit {self.modes}")

    @torch.no_grad()
    def __call__(self, inputs, stage1=None):
        lits = rule_lab.literal_rows(self.model, inputs, self.derived).float()
        a = rule_lab._soft_act(lits, self.body, self.gate)
        z = self.bias[None].expand(len(inputs), -1).clone().index_add(1, self.head, a * self.weight[None])
        return torch.sigmoid(z)[:, None, :].expand(-1, self.W, -1)


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--paper_rules", type=str, default="*_vt_p*.rules.pt",
                        help="glob of the rule files in <output_dir>/rule_lab (one per head part)")
    parser.add_argument("--paper_mlps", type=str, default="",
                        help="more OnsetMLPs, hidden:dropout comma list, e.g. 2048:0.2,2048:0 (the run's is always in)")
    parser.add_argument("--paper_tag", type=str, default="", help="suffix of the output file")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    if not getattr(args, "run_id", None):
        raise ValueError("--run_id is needed (the rule lab's run)")
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    lab_dir = args.output_dir / "rule_lab"
    files = sorted(glob.glob(str(lab_dir / args.paper_rules)))
    if not files:
        raise FileNotFoundError(f"no rule files {lab_dir / args.paper_rules}: run rule_lab (it writes .rules.pt)")

    training_data = step05.load_training_data(cfg)
    model = step05.build_rule_net(training_data, cfg, 1).to(device)     # the row inputs only (not trained)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    derived = rule_lab.Derived(model, names)
    W = len(model.windows)
    scorers = {"rules": RuleScorer(files, model, derived, W, device)}
    mlp = step05.load_onset_mlp(training_data, cfg)
    if mlp is not None:
        scorers["mlp"] = lambda inputs, stage1, m=mlp: m.window_probs(m(inputs)[0])
    for spec in (x for x in args.paper_mlps.split(",") if x):
        hid, _, drop = spec.partition(":")
        args.mlp_hidden_size, args.mlp_dropout = int(hid), float(drop or 0.2)
        cfg_m = step05.get_cfg(config.get_step_05_input(args))
        m = step05.load_onset_mlp(training_data, cfg_m) if os.path.exists(cfg_m["mlp_model_file"]) else None
        if m is None:
            print(f"[Rule paper] no OnsetMLP {cfg_m['mlp_model_file']}: left out")
            continue
        scorers[f"mlp_h{hid}_do{drop or 0.2}"] = lambda inputs, stage1, m=m: m.window_probs(m(inputs)[0])

    # Val (thresholds) and test (the numbers): all videos of each split.
    val_ids = sorted(training_data["val_videos"])
    test_ids = sorted(rule_lab.add_test_videos(training_data, cfg))
    head_types_all = training_data["event_types"]["types"]
    counted = [step06.EVENT_TYPES.index(t) for t in step06.COUNTED_TYPES]
    heads = torch.nonzero(torch.isin(head_types_all, torch.tensor(counted))).flatten()
    missing = [names[int(h)] for h in heads if int(h) not in scorers["rules"].heads]
    if missing:
        print(f"[Rule paper] {len(missing)} counted heads without rules (score 0): {', '.join(missing)}")
    label_names = [names[int(h)] for h in heads]
    head_type = head_types_all[heads].cpu()
    is_ego = torch.tensor([n.startswith("av:") for n in label_names])
    fps = int(getattr(args, "frame_rate", 10) or 10)
    run_cfg = {"device": device, "frame_rate": fps, "min_context": cfg["slot_size"] * cfg["num_slots"],
               "frames_per_video": cfg["frames_per_video"], "window": 20, "context": None}
    heads_d = heads.to(device)
    run_cfg["ego_heads"] = heads_d[is_ego.to(device)]
    atom_class, class_names = step05.atom_classes(vocab, step05.label_atom_count(training_data),
                                                  training_data["num_atoms"])
    run_cfg["atom_class"], run_cfg["num_classes"] = atom_class.to(device), len(class_names)
    grid = step07_paper.threshold_grid()

    val_curve, _h, _pv, _e, _hv = step07_paper.run_pass(model, scorers, {}, training_data, val_ids, heads_d, head_type,
                                                        is_ego, {**run_cfg, "desc": "Paper protocol (val)"}, grid)
    chosen = {nm: step07_paper.choose_thresholds(c, step07_paper.PAPER["budgets"]) for nm, c in val_curve.items()}
    test_curve, hist, per_video, events, _hx = step07_paper.run_pass(
        model, scorers, {}, training_data, test_ids, heads_d, head_type, is_ego,
        {**run_cfg, "desc": "Paper protocol (test)"}, grid,
        {nm: {b: th for b, th in c.items() if th is not None} for nm, c in chosen.items()})
    out = {"version": step07_paper.VERSION, "settings": step07_paper._settings(), "window": 20, "frame_rate": fps,
           "rule_files": [os.path.basename(f) for f in files], "fit": scorers["rules"].modes,
           "num_val_videos": len(val_ids), "num_test_videos": len(test_ids), "heads": label_names,
           "heads_without_rules": missing, "scorers": {}, "videos": {}}
    for nm, vc in val_curve.items():
        out["scorers"][nm] = {
            "ap": step07_paper.ap_summary(hist[nm], head_type, is_ego, label_names),
            "val_curve": step07_paper._curve_json(vc), "test_curve": step07_paper._curve_json(test_curve[nm]),
            "budgets": {str(b): {"threshold": th,
                                 "val_fa_per_min": vc[th]["false"] / vc[th]["minutes"] if th is not None else None,
                                 "test": step07_paper.summarize(per_video[nm][b], fps) if th is not None else None}
                        for b, th in chosen[nm].items()}}
        out["videos"][nm] = {str(b): per_video[nm][b] for b in per_video[nm]}
    # per test event: the lead of every scorer at every budget (the event count / overlap table)
    out["events"] = events
    path = lab_dir / f"paper_{args.run_id}{'_' + args.paper_tag if args.paper_tag else ''}.json"
    utils_data.save_json(out, path)
    step07_paper.print_summary(out)
    print(f"[Rule paper] {path}")


if __name__ == "__main__":
    main()
