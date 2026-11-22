"""Do the rules learned on one dataset hold on another? Clause-level transfer (SUMO -> roadpp).

The trained TemporalRuleNet and the clauses of a source run (e.g. sumo_maps_v7) are evaluated on the
videos of a target exp (e.g. full_sn), without training on them:

1. Vocabulary alignment (`align`): the target videos are prepared with the SOURCE run's input settings
   (roles, grid with the source's fitted grid, durations) on the target's own label vocabulary, then
   every atom is mapped to the source model's atom index by its signature (both datasets use the roadpp
   label.json, so a label atom has the same signature; grid and duration atoms by their names). Target
   atoms the source never saw are dropped (counted in the report); source atoms the target never holds
   stay off, so a clause with such a body never fires there.
2. Per clause (`step06.evaluate_clauses` on the aligned target videos): target_support, target_fires,
   target_confidence, target_recall, target_base_rate, next to its source val and test values.
   A clause "transfers" if its target lift (confidence / base rate) is >= MIN_LIFT with >= MIN_SUPPORT
   hits; "silent" if it never fires on the target.

    ./run_roadpp_docker.sh --gpu 0 --transfer -- --exp sumo_maps_v7 --run_id <source run> --target_exp full_sn
    (add the source run's model flags, e.g. --roles ahead,near_ped,xing_ped; --target_splits test val train:
    a model trained on SUMO has seen no roadpp video, so every split is a test)
Writes <source run>/transfer/<target_exp>_clauses.json and <target_exp>_summary.json."""
import json
from collections import Counter

import torch

from src.exp_roadpp import config, step05, step06, utils_data, utils_pipe

MIN_LIFT = 2.0              # a clause holds on a dataset with confidence >= MIN_LIFT x the head's base rate
MIN_SUPPORT = 5             # ... and at least this many hits
REPORT_HEADS = ("av:AV-Stop", "av:AV-Mov", "av:AV-TurLft", "av:AV-TurRht", "Car:Brake", "Car:Stop",
                "Car:MovAway", "Car:TurLft", "Car:TurRht", "Ped:Wait2X", "Ped:Xing")


def _key(signature):
    return json.dumps(signature)


def _lift(conf, base):
    return conf / base if conf is not None and base else None


def holds(conf, base, support):
    lift = _lift(conf, base)
    return lift is not None and lift >= MIN_LIFT and (support or 0) >= MIN_SUPPORT


def align(source_vocab, target_vocab):
    """(A_target,) long: the source atom index of every target atom, -1 if the source has none; matched
    by signature, else (grid / duration atoms, other ids) by `step05._atom_name`."""
    by_key = {_key(s): i for i, s in enumerate(source_vocab)}
    by_name = {}
    for i, s in enumerate(source_vocab):
        by_name.setdefault(step05._atom_name(s), i)
    out = torch.full((len(target_vocab),), -1, dtype=torch.long)
    for a, s in enumerate(target_vocab):
        out[a] = by_key.get(_key(s), by_name.get(step05._atom_name(s), -1))
    return out


def remap_videos(prepared, mapping, num_source_atoms):
    """The prepared target videos with their atom cells moved to the source indices (unmapped ones
    dropped); returns the data dict for the source model and Counter {target atom: dropped cells}."""
    dropped = Counter()
    videos = {}
    for vid, video in prepared["videos"].items():
        cells = video["cells"].long()
        src = mapping[cells[:, 2]]
        keep = src >= 0
        dropped.update(cells[~keep, 2].tolist())
        new = cells[keep].clone()
        new[:, 2] = src[keep]
        videos[vid] = {**video, "cells": new.to(torch.int32)}
    data = {"num_atoms": num_source_atoms, "min_off_frames": prepared["min_off_frames"],
            "videos": videos, "frames": prepared["frames"]}
    return data, dropped


def prepare_target(target_atoms_dir, splits, cfg):
    """`step05.prepare_videos` of the target splits with the source run's input settings (cfg),
    on the target's label vocabulary; merged over the splits."""
    for name in ("motion", "ego_motion", "ego_flow"):
        if cfg.get(name):
            raise NotImplementedError(f"transfer of a model with {name} atoms (fitted per dataset)")
    unique_atoms = utils_data.load_json(target_atoms_dir / "unique_atoms.json")
    target_cfg = {**cfg, "unique_atoms_file": target_atoms_dir / "unique_atoms.json"}
    grid_spec = step05.load_grid(cfg)                       # the SOURCE grid: the same cell definitions
    duration_spec = step05.load_durations(target_cfg)
    merged = {"videos": {}, "frames": {}, "min_off_frames": cfg["min_off_frames"]}
    for sp in splits:
        d = step05.prepare_videos(target_atoms_dir / f"{sp}_atoms_compressed.json", len(unique_atoms), "cpu",
                                  cfg["min_off_frames"], target_atoms_dir / f"{sp}_boxes.pt", cfg["roles"],
                                  cfg["geometry"], unique_atoms, grid_spec, None, None, None, duration_spec,
                                  step05.load_predict(cfg),     # fixed camera constants: nothing fitted per dataset
                                  step05.load_shape(cfg))       # the SOURCE thresholds, as the grid
        merged["videos"].update(d["videos"])
        merged["frames"].update(d["frames"])
    return merged, step05.load_vocabulary(target_cfg)


def summarize(clauses, names):
    """Counts of clauses that hold on the source (val), fire on the target and hold there; per head."""
    def entry(cs):
        valid = [c for c in cs if holds(c.get("val_confidence"), c.get("val_base_rate"), c.get("val_support"))]
        fire = [c for c in valid if (c.get("target_fires") or 0) > 0]
        ok = [c for c in valid if c["transfers"]]
        return {"clauses": len(cs), "hold_on_source": len(valid), "fire_on_target": len(fire),
                "hold_on_target": len(ok), "share_transferred": round(len(ok) / len(valid), 3) if valid else None}
    by_head = {}
    for c in clauses:
        by_head.setdefault(names[c["head"][0]], []).append(c)
    return {"all": entry(clauses), "per_head": {h: entry(cs) for h, cs in sorted(by_head.items())}}


def _text(c, names):
    """"av:AV-Mov(X0)[0-10) & Car:Brake(ahead)[0-10) -> av:AV-Brake": bodies (atom, scope, slot), frames before the head."""
    scopes = c.get("scope_names") or ["X0", "X1"]
    frames = c.get("span_frames") or [[None, None]] * len(c["bodies"])
    body = " & ".join(f"{names[b[0]]}({scopes[b[1]]})[{f[0]}-{f[1]})" for b, f in zip(c["bodies"], frames))
    return f"{body} -> {names[c['head'][0]]}"


def print_report(clauses, summary, names, coverage):
    print(f"[Transfer] target rows: {coverage['target_atom_cells']} atom cells, "
          f"{coverage['dropped_share']:.1%} of them on atoms the source never saw "
          f"(top: {', '.join(f'{n} {k}' for n, k in coverage['dropped_top'][:8])})")
    a = summary["all"]
    print(f"[Transfer] clauses holding on the source val (lift >= {MIN_LIFT}, >= {MIN_SUPPORT} hits): "
          f"{a['hold_on_source']}; firing on the target: {a['fire_on_target']}; holding there: {a['hold_on_target']}")
    for h in REPORT_HEADS:
        cs = [c for c in clauses if names[c["head"][0]] == h
              and holds(c.get("val_confidence"), c.get("val_base_rate"), c.get("val_support"))]
        if not cs:
            continue
        print(f"  {h}: {summary['per_head'][h]['hold_on_target']}/{len(cs)} transfer")
        for c in sorted(cs, key=lambda c: -(c.get("val_support") or 0) * (c.get("val_confidence") or 0))[:5]:
            sl, tl = _lift(c.get("val_confidence"), c.get("val_base_rate")), _lift(c["target_confidence"], c["target_base_rate"])
            print(f"     {'ok ' if c['transfers'] else 'off' if not c.get('target_fires') else '-- '} "
                  f"lift src {sl or 0:5.1f} tgt {tl or 0:5.1f} (tgt hits {c['target_support']}/{c['target_fires']})  "
                  f"{_text(c, names)}")


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--target_exp", required=True, help="exp whose step_dataset videos are the target")
    parser.add_argument("--target_splits", nargs="+", default=["test"], help="target splits (all: train val test)")
    args = utils_pipe.parse_args(parser)                    # the SOURCE exp's config and flags
    if not getattr(args, "run_id", None):
        raise SystemExit("--run_id <source run> is needed (the trained model and its clauses)")
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    training_data = step05.load_training_data(cfg)
    model = step05.load_temporal_rule_net(training_data, cfg)
    clauses = utils_data.load_json(args.run_dir / "step06_output" / "test_clauses.json")
    source_vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(s) for s in source_vocab]
    if len(source_vocab) != training_data["num_atoms"]:
        raise ValueError(f"source vocabulary {len(source_vocab)} != model atoms {training_data['num_atoms']}")

    target_atoms_dir = args.output_dir.parent / args.target_exp / "step_dataset_output" / "atoms"
    prepared, target_vocab = prepare_target(target_atoms_dir, args.target_splits, cfg)
    mapping = align(source_vocab, target_vocab)
    data, dropped = remap_videos(prepared, mapping, len(source_vocab))
    data["target_heads"] = training_data["target_heads"]
    total = sum(len(v["cells"]) for v in prepared["videos"].values())
    target_names = [step05._atom_name(s) for s in target_vocab]
    coverage = {"target_atoms": len(target_vocab), "mapped_atoms": int((mapping >= 0).sum()),
                "source_atoms": len(source_vocab),
                "source_atoms_in_target": int(torch.unique(mapping[mapping >= 0]).numel()),
                "target_atom_cells": total, "dropped_share": sum(dropped.values()) / max(total, 1),
                "dropped_top": [(target_names[a], n) for a, n in dropped.most_common(30)],
                "videos": len(data["videos"]), "splits": args.target_splits}

    # evaluate_clauses writes its counts as test_*: the source's own test_* are kept from the clause file
    source = {(c["rule"], c["head"][0]): c for c in clauses}
    present = set(mapping[mapping >= 0].tolist())
    out = []
    for c in step06.evaluate_clauses(model, data, clauses, cfg):
        t = {k.replace("test_", "target_", 1): c[k] for k in ("test_support", "test_fires", "test_confidence",
                                                               "test_anchor_confidence", "test_recall", "test_base_rate")}
        src = source[(c["rule"], c["head"][0])]
        silent = [names[b[0]] for b in c["bodies"] if b[0] not in present]
        out.append({**src, **t, "text": _text(c, names),
                    "target_lift": _lift(t["target_confidence"], t["target_base_rate"]),
                    "source_val_lift": _lift(c.get("val_confidence"), c.get("val_base_rate")),
                    "bodies_absent_on_target": silent,
                    "transfers": holds(t["target_confidence"], t["target_base_rate"], t["target_support"])})
    summary = {"source": {"exp": args.exp, "run": args.run_id}, "target": args.target_exp,
               "min_lift": MIN_LIFT, "min_support": MIN_SUPPORT, "coverage": coverage, **summarize(out, names)}
    out_dir = args.run_dir / "transfer"
    out_dir.mkdir(exist_ok=True)
    utils_data.save_json(out, out_dir / f"{args.target_exp}_clauses.json")
    utils_data.save_json(summary, out_dir / f"{args.target_exp}_summary.json")
    print_report(out, summary, names, coverage)
    print(f"[Transfer] -> {out_dir}")


if __name__ == "__main__":
    main()
