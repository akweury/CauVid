"""Tune step 05: sweep its hyperparameters one at a time around the base
setting of `step05.get_cfg`, score every run on the val split with
`step06.evaluate_clauses`, and plot all sweeps on one chart (PDF).

Usage: python -m src.exp_roadpp.step05_tune --exp debug --machine ml-pulsar
"""
import json
import copy

from src.exp_roadpp import config, step05, step06, utils_data, utils_pipe


SWEEPS = {
    "l0_lambda": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "min_confidence_gain": [0.0, 0.02, 0.05, 0.1, 0.2],
    "lr": [0.03, 0.1, 0.3],
    "num_epochs": [100, 200, 400],
    # Last: every new window set rebuilds the cached training examples.
    "delay_windows": [[1, 5, 15, 50], [3, 5, 15, 50], [3, 10, 25, 50], [5, 15, 30, 50]],
}


def _run_key(name, value):
    return f"{name}={json.dumps(value)}"


def run_sweeps(base_cfg, tune_dir, atoms, val_atoms, anchor_clauses, labels):
    """Run every sweep value not yet in tune_dir/tune_results.json (resumable)."""
    results_file = tune_dir / "tune_results.json"
    results = utils_data.load_json(results_file) if results_file.exists() else {}
    head_atoms = sorted({c["anchor"]["head"][0] for c in anchor_clauses.values()})
    for name, values in SWEEPS.items():
        for value in values:
            key = _run_key(name, value)
            if key in results:
                print(f"[Tune] {key} already done")
                continue
            print(f"\n[Tune] {key}")
            cfg = copy.deepcopy(base_cfg)
            cfg[name] = value
            # Shared cache of the training examples; rebuilt when the windows change.
            cfg["output_dir"] = tune_dir
            clauses = step05.extend_clauses(anchor_clauses, labels, atoms, cfg)
            summary = step06.evaluate_clauses(
                clauses, atoms["unique_atoms"], val_atoms, head_atoms,
                cfg["delay_windows"], device=cfg["device"],
            )
            results[key] = {
                "f1": [m["f1"] for m in summary["overall"]],
                "precision": [m["precision"] for m in summary["overall"]],
                "recall": [m["recall"] for m in summary["overall"]],
                "bin_accuracy": summary["bin_accuracy"],
                "num_clauses": summary["num_clauses"],
            }
            print(f"[Tune] {key}: f1={[round(f, 3) for f in results[key]['f1']]}, "
                  f"bin_accuracy={results[key]['bin_accuracy']:.3f}")
            utils_data.save_json(results, results_file)
    return results


def plot_sweeps(results, base_cfg, plot_file):
    import matplotlib.pyplot as plt
    names = list(SWEEPS)
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6), sharey=True)
    num_windows = len(base_cfg["delay_windows"])
    colors = plt.cm.viridis([k / max(num_windows - 1, 1) for k in range(num_windows)])
    for ax, name in zip(axes, names):
        values = [v for v in SWEEPS[name] if _run_key(name, v) in results]
        runs = [results[_run_key(name, v)] for v in values]
        positions = range(len(values))
        for k in range(num_windows):
            ax.plot(positions, [run["f1"][k] for run in runs], marker="o", color=colors[k],
                    label=f"F1, window {k + 1}")
        ax.plot(positions, [run["bin_accuracy"] for run in runs], marker="s", linestyle="--",
                color="black", label="bin accuracy")
        if base_cfg[name] in values:
            ax.axvline(values.index(base_cfg[name]), color="gray", linestyle=":", linewidth=1)
        ax.set_xticks(list(positions))
        ax.set_xticklabels([str(v) for v in values], rotation=30, ha="right", fontsize=8)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("val score")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Step 05 tuning on val (one hyperparameter at a time, dotted = base; "
        f"base windows {base_cfg['delay_windows']})", fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(plot_file)
    plt.close(fig)
    print(f"[Tune] Saved {plot_file}")


def main():
    args = utils_pipe.parse_args()
    args = config.step_0_setup(args)
    step05_input = config.get_step_05_input(args)
    val_atoms_file = config.get_step_06_input(args)["val_atoms_file"]
    base_cfg = step05.get_cfg(step05_input)
    tune_dir = step05_input["output_dir"] / "tune"
    tune_dir.mkdir(parents=True, exist_ok=True)

    labels = utils_data.load_json(step05_input["dataset_label_file"])
    atoms = utils_data.load_atoms(step05_input["train_atom_file"])
    val_atoms = utils_data.load_atoms(val_atoms_file)
    anchor_clauses = utils_data.load_json(step05_input["causal_anchor_clause_file"])

    results = run_sweeps(base_cfg, tune_dir, atoms, val_atoms, anchor_clauses, labels)
    plot_sweeps(results, base_cfg, tune_dir / "step05_tune.pdf")


if __name__ == "__main__":
    main()
