"""The whole experiment in one command: the main model and every ablation variant, then the paper tables.

  ./run_roadpp_docker.sh --gpu 6 --exp medium --ablation
  ./run_roadpp_docker.sh --gpu 6 --exp full --ablation                      # the paper preset (see --list)
  ./run_roadpp_docker.sh --gpu 6 --exp medium --ablation --variants main,no_grid
  ./run_roadpp_docker.sh --gpu 6 --exp medium --ablation --suite 20261114-0900   # resume a suite

Each variant is one pipeline run (`main.py`) with a fixed run id <suite>_<variant>, so a stopped
suite resumes where it was: a variant whose paper_metrics.json exists is skipped, a half-done one
continues from its cached steps. The shared steps (dataset, step 03 anchors, step 04 causal filter)
run once in the first variant and are cached for the others. Every variant is scored from the same
first frame (--eval_min_context = the longest memory of the suite), so all see the same test events.
The main variant also trains the layered rule net and the state forecast; the others only what
the tables need. A variant that fails is logged and the suite goes on.

Output: output/roadpp/<exp>/suites/<suite>/ with suite.json (variants, run ids, status, times),
<variant>.log (the pipeline output of each) and the tables (`paper_tables.py`, markdown + LaTeX); and
output/roadpp/<exp>/packs/results_<exp>_<suite>.tar.gz, the suite and the small result files of its runs
to download (no models, data or videos; the same as pack_results.sh).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.exp_roadpp import config, paper_tables, rtpt_utils

# name: (table label, group, main.py args). The groups order Table 2.
VARIANTS = {
    "main": ("Full model", "main", []),
    # The causal step: what the causal filter of the LLM anchors adds, and against chance.
    "no_causal_filter": ("− causal filter (all LLM anchors)", "causal", ["--anchor_source", "all"]),
    "random_filter_0": ("random filter, same size (draw 0)", "causal", ["--anchor_source", "random", "--anchor_subset_seed", "0"]),
    "random_filter_1": ("random filter, same size (draw 1)", "causal", ["--anchor_source", "random", "--anchor_subset_seed", "1"]),
    "random_filter_2": ("random filter, same size (draw 2)", "causal", ["--anchor_source", "random", "--anchor_subset_seed", "2"]),
    "no_anchors": ("− LLM anchors (free rules, same capacity)", "causal", ["--anchor_source", "none"]),
    # Inputs.
    "no_grid": ("− grid atoms", "inputs", ["--grid", "False"]),
    "motion": ("+ motion atoms", "inputs", ["--motion", "True"]),
    "ego_motion": ("+ ego motion atoms", "inputs", ["--ego_motion", "True"]),
    "ego_flow": ("+ ego flow atoms (image yaw / speed)", "inputs", ["--ego_flow", "True"]),
    # Sequences of steps with variable durations (slow down, move right, turn right): bodies over slot
    # ranges a..b and the whole known memory (open spans "a s ago or any time before").
    "slot_ranges": ("+ slot ranges, all known memory", "structure", ["--slot_ranges", "True"]),
    "ego_flow_ranges": ("+ ego flow, slot ranges, all known memory", "structure",
                        ["--ego_flow", "True", "--slot_ranges", "True"]),
    "roles": ("+ agent roles (ahead, near ped.)", "inputs", ["--roles", "ahead,near_ped"]),
    "geometry": ("+ geometry atoms", "inputs", ["--geometry", "True"]),
    # Structure and training.
    "no_require_self": ("− own-agent atom per rule", "structure", ["--require_self", "False"]),
    "no_partial": ("− partial facts (no truncation)", "structure", ["--truncate_prob", "0"]),
    "memory_1": ("memory 1 s (1 slot)", "structure", ["--num_slots", "1"]),
    "memory_5": ("memory 5 s (5 slots)", "structure", ["--num_slots", "5"]),
    "type_balance": ("+ event type balance", "structure", ["--type_balance", "True"]),
    # Learning curve: nested shares of the train videos (val, test, heads unchanged; `step05.subsample_train_videos`).
    "train_5": ("5% of the train videos", "data", ["--train_fraction", "0.05"]),
    "train_10": ("10% of the train videos", "data", ["--train_fraction", "0.1"]),
    "train_25": ("25% of the train videos", "data", ["--train_fraction", "0.25"]),
    "train_50": ("50% of the train videos", "data", ["--train_fraction", "0.5"]),
    # Fit diagnostics (preset fit): why the rule net trails the MLP on the same inputs. Full 20261115-2131 / medium
    # 20261116-1340: the rule net's best-val epoch has a higher train AND val loss than the MLP's (full 0.041 / 0.051
    # vs 0.031 / 0.044), and the sparsity loss raises its train loss after the warm-up (0.037 -> 0.041).
    "rule_lr_1e-3": ("rule net lr 1e-3 (the MLP's)", "fit", ["--rule_lr", "0.001"]),
    "l0_1e-4": ("sparsity weight 1e-4", "fit", ["--l0_lambda", "0.0001"]),
    "no_l0": ("no sparsity loss", "fit", ["--l0_lambda", "0"]),
    "init_m6": ("mask gates start at -6 (stronger gradients)", "fit", ["--init_log_alpha", "-6"]),
    "free_512": ("512 free rules (MLP hidden size alike)", "fit", ["--num_free_rules", "512"]),
    "seed_1": ("train seed 1 (noise of both models)", "fit", ["--train_seed", "1"]),
}
# Variants not run by default: the existing results already settle them (medium suite 20261113-1635,
# full suite 20261113-1636 and the earlier full ablations); --preset all or --variants runs them anyway.
SKIPPED = {
    "motion": "off in the main model since 2026-11-15: medium suite 20261113-1635 no change in any metric without it; "
              "full 78e420: the own-label lookup matched the models without it",
    "ego_motion": "full a2765e vs f45174: no clear gain (AP@W20 0.071 vs 0.069, more false alarms); medium: AP up, recall down",
    "roles": "earlier: val loss unchanged; medium: AP +0.005, recall unchanged",
    "geometry": "earlier: worse; medium: recall -0.044",
    "type_balance": "earlier: inflates regular / rare scores into false alarms; medium: mixed",
    "memory_1": "medium: slightly worse on every metric; medium 6b02b3 / full 78e420: 1 s keeps >= 90% of the AP",
    "memory_5": "medium: AP up, recall down; medium 6b02b3 / full 78e420: no gain beyond 3 s",
}
PRESETS = {
    # The paper ablation: the causal step (with 3 random draws for its noise), and the modules whose medium
    # ablation moved the results (grid: recall; own-agent atom and partial facts: every metric).
    "paper": ["main", "no_causal_filter", "random_filter_0", "random_filter_1", "random_filter_2", "no_anchors",
              "no_grid", "no_require_self", "no_partial"],
    # The learning curve: how much train data the models need (main = 100%).
    "data": ["main", "train_5", "train_10", "train_25", "train_50"],
    # Fit diagnostics: optimisation and capacity of the rule net (fit.md: train / val loss of both models).
    "fit": ["main", "rule_lr_1e-3", "l0_1e-4", "no_l0", "init_m6", "free_512", "seed_1"],
    "all": list(VARIANTS),
}
# Only the main variant trains what the tables do not need (the layered rule net).
LEAN = ["--layered_rules", "False"]
# No variant renders the review videos (~2.8 GB per run on full; off by default in main.py too) unless
# --review_videos is given; they can be rendered later for one run: --run_id <suite>_main --review_videos True.
NO_VIDEOS = ["--review_videos", "False"]


PACK_FILES = ("step06_output/paper_metrics.json", "step06_output/paper_events.csv", "step06_output/test_results.json",
              "step06_output/state_metrics.json", "step06_output/state_compare.png")


def pack(out_root, exp, suite, run_ids):
    """<out_root>/packs/results_<exp>_<suite>.tar.gz: the suite folder and the small result files of its runs
    (the same as pack_results.sh), paths relative to <out_root> so it unpacks into the local layout."""
    import tarfile
    out_root = Path(out_root)
    os.makedirs(out_root / "packs", exist_ok=True)
    out = out_root / "packs" / f"results_{exp}_{suite}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(out_root / "suites" / suite, arcname=f"suites/{suite}")
        for run_id in run_ids:
            run = out_root / "runs" / run_id
            history = [f for g in HISTORY_GLOBS for f in sorted(run.glob(g))]
            for f in sorted(run.glob("run_info*.json")) + [run / x for x in PACK_FILES] + history:
                if f.exists():
                    tar.add(f, arcname=str(f.relative_to(out_root)))
    print(f"[Suite] results packed: {out} ({out.stat().st_size / 1e6:.1f} MB); "
          f"unpack locally with tar -xzf {out.name} -C <CauVid>/src/exp_roadpp/output/roadpp/{exp}/")
    return out


# The per-epoch losses of both models (`step05.fit_onset_model`), for fit.md.
HISTORY_GLOBS = ("step05_output/training_history_*.json", "step05_output/onset_mlp_history_*.json")


def _best(history, select_from=0):
    """The kept epoch of a loss history: the lowest val loss after select_from epochs (`fit_onset_model`)."""
    rows = [e for e in history if e["epoch"] > select_from] or history
    return min(rows, key=lambda e: e["val_loss"])


def fit_table(out_root, state, names, warmup=10):
    """fit.md: per variant, the kept epoch and its train / val loss of the rule net and the MLP, the same
    loss on the same val frames per variant (a variant with another --train_seed samples other val frames:
    compare the two models within a row). Rule net val / MLP val > 1: the rule net fits the val data worse."""
    lines = ["# Fit: train / val onset loss at the kept epoch\n",
             "Same loss and val frames for both models within a row. Gap = val - train; ratio = rule val / MLP val.\n",
             "| Variant | Rule epochs (best) | Rule train | Rule val | Rule gap | MLP epochs (best) | MLP train | MLP val "
             "| MLP gap | Ratio | Rule val at epoch 1 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        run = Path(out_root) / "runs" / state["variants"].get(n, {}).get("run_id", "-")
        files = {k: sorted(run.glob(g)) for k, g in zip(("rule", "mlp"), HISTORY_GLOBS)}
        files["rule"] = [f for f in files["rule"] if "verifier" not in f.name]
        if not files["rule"] or not files["mlp"]:
            continue
        rule, mlp = json.load(open(files["rule"][-1])), json.load(open(files["mlp"][-1]))
        r, m = _best(rule, warmup), _best(mlp)
        lines.append(f"| {VARIANTS[n][0]} | {len(rule)} ({r['epoch']}) | {r['train_loss']:.4f} | {r['val_loss']:.4f} | "
                     f"{r['val_loss'] - r['train_loss']:+.4f} | {len(mlp)} ({m['epoch']}) | {m['train_loss']:.4f} | "
                     f"{m['val_loss']:.4f} | {m['val_loss'] - m['train_loss']:+.4f} | {r['val_loss'] / m['val_loss']:.2f} | "
                     f"{rule[0]['val_loss']:.4f} |")
    out = Path(out_root) / "suites" / state["suite"] / "fit.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[Suite] {out}")
    return out


def _slots(args_list):
    return int(args_list[args_list.index("--num_slots") + 1]) if "--num_slots" in args_list else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="medium")
    ap.add_argument("--machine", default="dgx-jing")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--suite", default=None, help="suite id to resume (default: a new one)")
    ap.add_argument("--preset", default="paper",
                    help="variant set(s), comma separated: paper (default; the variants the existing results do not "
                         "settle), data (the learning curve), all; e.g. paper,data")
    ap.add_argument("--variants", default=None, help="comma separated variant names (overrides --preset)")
    ap.add_argument("--budget", type=float, default=5, help="false alarms / min of the tables")
    ap.add_argument("--variant_scores", choices=("cal", "raw"), default="cal")
    ap.add_argument("--list", action="store_true", help="list the variants and exit")
    ap.add_argument("--review_videos", action="store_true", help="render the review videos of the main variant")
    args, passthrough = ap.parse_known_args()     # anything else (e.g. --num_epochs 50) goes to every run
    if args.list:
        for name, (label, group, a) in VARIANTS.items():
            mark = ("" if name in PRESETS["paper"] else "  [preset data]" if name in PRESETS["data"]
                    else "  [preset fit]" if name in PRESETS["fit"] else "  [skipped by default]")
            print(f"{name:<18} {group:<10} {label:<45} {' '.join(a)}{mark}")
        print("\nSkipped by default (run with --preset all or --variants):")
        for name, why in SKIPPED.items():
            print(f"  {name:<16} {why}")
        return
    if args.variants:
        names = args.variants.split(",")
    else:
        unknown_presets = [p for p in args.preset.split(",") if p not in PRESETS]
        if unknown_presets:
            raise SystemExit(f"Unknown presets {unknown_presets}; one of {list(PRESETS)}")
        names = list(dict.fromkeys(n for p in args.preset.split(",") for n in PRESETS[p]))
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants {unknown}; see --list")
    if "main" not in names:
        names = ["main"] + names                  # the tables compare every variant to it
    suite = args.suite or time.strftime("%Y%m%d-%H%M")
    out_root = config.machine_setup(argparse.Namespace(machine=args.machine, exp=args.exp, device=args.device)).output_dir
    suite_dir = Path(out_root) / "suites" / suite
    os.makedirs(suite_dir, exist_ok=True)
    state_file = suite_dir / "suite.json"
    state = json.load(open(state_file)) if state_file.exists() else {"suite": suite, "exp": args.exp, "variants": {}}
    eval_min_context = 10 * max(_slots(VARIANTS[n][2]) for n in names)
    state.update({"names": names, "eval_min_context": eval_min_context, "passthrough": passthrough})

    def save():
        json.dump(state, open(state_file, "w"), indent=2)
    print(f"[Suite] {suite}: {len(names)} variants ({', '.join(names)}), scored from frame {eval_min_context}")
    # RTPT title of the suite process (one iteration per variant); every variant run names itself too (main.py).
    rtpt = rtpt_utils.start(f"roadpp-{args.exp}-suite", len(names))
    for i, name in enumerate(names):
        label, group, extra = VARIANTS[name]
        run_id = f"{suite}_{name}"
        metrics = Path(out_root) / "runs" / run_id / "step06_output" / "paper_metrics.json"
        entry = state["variants"].setdefault(name, {"run_id": run_id, "label": label, "group": group, "args": extra})
        if metrics.exists():
            print(f"[Suite] ({i + 1}/{len(names)}) {name}: done, skipped")
            entry["status"] = "done"
            save()
            rtpt_utils.step(rtpt, name)
            continue
        cmd = [sys.executable, "-m", "src.exp_roadpp.main", "--exp", args.exp, "--machine", args.machine,
               "--device", args.device, "--run_id", run_id, "--eval_min_context", str(eval_min_context),
               *extra, *([] if name == "main" else LEAN),
               *(["--review_videos", "True"] if name == "main" and args.review_videos else NO_VIDEOS), *passthrough]
        print(f"[Suite] ({i + 1}/{len(names)}) {name}: {' '.join(cmd[2:])}")
        entry.update({"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "command": cmd[2:]})
        save()
        t0 = time.time()
        with open(suite_dir / f"{name}.log", "a") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = proc.wait()
        entry.update({"status": "done" if code == 0 and metrics.exists() else f"failed (exit {code})",
                      "minutes": round((time.time() - t0) / 60, 1)})
        save()
        print(f"[Suite] {name}: {entry['status']}, {entry['minutes']} min")
        rtpt_utils.step(rtpt, name)

    # Tables of every finished variant.
    done = [n for n in names if state["variants"].get(n, {}).get("status") == "done"]
    if "main" not in done:
        print("[Suite] the main variant did not finish: no tables")
        pack(out_root, args.exp, suite, [state["variants"][n]["run_id"] for n in done])
        return
    try:
        fit_table(out_root, state, done)
    except Exception as e:
        print(f"[Suite] fit.md failed ({type(e).__name__}: {e})")
    budget = int(args.budget) if float(args.budget).is_integer() else args.budget
    table_args = ["--exp", args.exp, "--root", str(Path(out_root).parent), "--main", state["variants"]["main"]["run_id"],
                  "--budget", str(budget), "--variant", args.variant_scores, "--out", str(suite_dir)]
    for n in done:
        if n == "main":
            continue
        table_args += ["--ablation", f"{VARIANTS[n][0]}={state['variants'][n]['run_id']}"]
    random_runs = [state["variants"][n]["run_id"] for n in done if n.startswith("random_filter_")]
    if len(random_runs) > 1:                       # the random draws also as one row (mean over the draws)
        table_args += ["--ablation", f"random filter, same size (mean of {len(random_runs)})={','.join(random_runs)}"]
    try:
        paper_tables.main(table_args)
    except Exception as e:                         # the tables can be redone locally from the pack
        print(f"[Suite] the tables failed ({type(e).__name__}: {e}); the results are packed anyway")
    pack(out_root, args.exp, suite, [state["variants"][n]["run_id"] for n in done])
    failed = [n for n in names if n not in done]
    print(f"[Suite] {len(done)}/{len(names)} variants done" + (f"; not done: {failed}" if failed else ""))
    print(f"[Suite] {suite_dir}")


if __name__ == "__main__":
    main()
