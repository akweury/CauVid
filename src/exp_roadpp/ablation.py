"""The whole experiment in one command: the main model and every ablation variant, then the paper tables.

  ./run_roadpp_docker.sh --gpu 6 --exp medium --ablation
  ./run_roadpp_docker.sh --gpu 6 --exp full --ablation                      # the full preset (core variants)
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
<variant>.log (the pipeline output of each) and the tables (`paper_tables.py`, markdown + LaTeX).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.exp_roadpp import config, paper_tables

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
    "no_motion": ("− motion atoms", "inputs", ["--motion", "False"]),
    "ego_motion": ("+ ego motion atoms", "inputs", ["--ego_motion", "True"]),
    "roles": ("+ agent roles (ahead, near ped.)", "inputs", ["--roles", "ahead,near_ped"]),
    "geometry": ("+ geometry atoms", "inputs", ["--geometry", "True"]),
    # Structure and training.
    "no_require_self": ("− own-agent atom per rule", "structure", ["--require_self", "False"]),
    "no_partial": ("− partial facts (no truncation)", "structure", ["--truncate_prob", "0"]),
    "memory_1": ("memory 1 s (1 slot)", "structure", ["--num_slots", "1"]),
    "memory_5": ("memory 5 s (5 slots)", "structure", ["--num_slots", "5"]),
    "type_balance": ("+ event type balance", "structure", ["--type_balance", "True"]),
}
PRESETS = {
    "all": list(VARIANTS),
    "core": ["main", "no_causal_filter", "random_filter_0", "random_filter_1", "random_filter_2", "no_anchors",
             "no_grid", "no_motion"],
}
# Only the main variant trains what the tables do not need (layered rule net, state forecast).
LEAN = ["--layered_rules", "False", "--state_forecast", "False"]


def _slots(args_list):
    return int(args_list[args_list.index("--num_slots") + 1]) if "--num_slots" in args_list else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="medium")
    ap.add_argument("--machine", default="dgx-jing")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--suite", default=None, help="suite id to resume (default: a new one)")
    ap.add_argument("--preset", choices=list(PRESETS), default=None,
                    help="variant set: all (default for medium), core (default for full)")
    ap.add_argument("--variants", default=None, help="comma separated variant names (overrides --preset)")
    ap.add_argument("--budget", type=float, default=5, help="false alarms / min of the tables")
    ap.add_argument("--variant_scores", choices=("cal", "raw"), default="cal")
    ap.add_argument("--list", action="store_true", help="list the variants and exit")
    args, passthrough = ap.parse_known_args()     # anything else (e.g. --num_epochs 50) goes to every run
    if args.list:
        for name, (label, group, a) in VARIANTS.items():
            print(f"{name:<18} {group:<10} {label:<45} {' '.join(a)}")
        return
    names = (args.variants.split(",") if args.variants
             else PRESETS[args.preset or ("core" if args.exp == "full" else "all")])
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
    for i, name in enumerate(names):
        label, group, extra = VARIANTS[name]
        run_id = f"{suite}_{name}"
        metrics = Path(out_root) / "runs" / run_id / "step06_output" / "paper_metrics.json"
        entry = state["variants"].setdefault(name, {"run_id": run_id, "label": label, "group": group, "args": extra})
        if metrics.exists():
            print(f"[Suite] ({i + 1}/{len(names)}) {name}: done, skipped")
            entry["status"] = "done"
            save()
            continue
        cmd = [sys.executable, "-m", "src.exp_roadpp.main", "--exp", args.exp, "--machine", args.machine,
               "--device", args.device, "--run_id", run_id, "--eval_min_context", str(eval_min_context),
               *extra, *([] if name == "main" else LEAN), *passthrough]
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

    # Tables of every finished variant.
    done = [n for n in names if state["variants"].get(n, {}).get("status") == "done"]
    if "main" not in done:
        print("[Suite] the main variant did not finish: no tables")
        return
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
    paper_tables.main(table_args)
    failed = [n for n in names if n not in done]
    print(f"[Suite] {len(done)}/{len(names)} variants done" + (f"; not done: {failed}" if failed else ""))
    print(f"[Suite] {suite_dir}")


if __name__ == "__main__":
    main()
