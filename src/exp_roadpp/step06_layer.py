"""Step 06 (layered): evaluate the step05_layer LayeredRuleNet on the test split.

From the true layers up to some layer, the model rolls out the next horizons
layers (its own predictions fed back). For each horizon h (h layers = h *
block_size frames after the last observed layer) it is scored on:
  onset / offset: an atom off (on) in the last observed layer that is on (off)
      h layers later; AP (micro and macro over heads), best F1, and the detected
      / missed events at the threshold with the best F1 on the step 05 val videos;
      against the prior baseline, the training base rate of each head and horizon;
  state: F1 of the predicted on atoms (probability > 0.5) against the true
      state, against persistence (the last observed layer copied forward).
Also the step05_layer clauses get their test counts, and `compare_models`
scores the LayeredRuleNet, the step 05 TemporalRuleNet and OnsetMLP and the
baselines on one shared onset task, the same rows and events for all.
"""
import os
import torch
from tqdm import tqdm

from src.exp_roadpp import step05, step05_layer, step06, utils_data

KINDS = ("onset", "offset")


@torch.no_grad()
def layer_priors(training_data, cfg):
    """(H, 2, A) base rate of each horizon h (index h - 1), kind (onset, offset)
    and head over the training videos: atoms off (on) in a layer and on (off) h
    layers later, for the agents visible in both."""
    H, A = cfg["rollout_steps"], step05.label_atom_count(training_data)
    events, at_risk = torch.zeros((H, 2, A)), torch.zeros((H, 2, A))
    for video_id in tqdm(training_data["train_videos"], desc="Counting layer base rates (train)"):
        layers = step05_layer._video_layers(training_data, video_id, cfg).cpu() > 0.5
        visible = layers.any(dim=2)
        for h in range(1, min(H, layers.shape[0] - 1) + 1):
            last, target = layers[:-h], layers[h:]
            keep = (visible[:-h] & visible[h:])[..., None]
            for i, (risk, event) in enumerate(((~last, ~last & target), (last, last & ~target))):
                at_risk[h - 1, i] += (risk & keep).sum(dim=(0, 1))
                events[h - 1, i] += (event & keep).sum(dim=(0, 1))
    return events / at_risk.clamp(min=1)


@torch.no_grad()
def score_videos(model, data, priors, cfg):
    """Rollout scores of every layer of the split, started from each layer.

    Returns {
      "histograms": {kind: {scorer: (positives, negatives)}}, (H, A, S) score
          histograms (`step06.score_edges` bins) of the at-risk rows of each
          horizon, split into the rows where the event happens and the others;
          scorers "model" (onset: p, offset: 1 - p) and "prior";
      "state": {scorer: (H, 3)} tp, fp, fn of the on atoms, scorers "model"
          (p > 0.5) and "persistence" (the last observed layer);
      "num_frames": the frames of all videos,
    }.
    """
    model.eval()
    device, H, K, A = cfg["device"], cfg["rollout_steps"], model.num_layers, step05.label_atom_count(data)
    edges = step06.score_edges().to(device)
    S = len(edges)
    priors = priors.to(device)
    heads = torch.arange(A, device=device)
    histograms = {kind: {name: (torch.zeros((H, A * S), device=device), torch.zeros((H, A * S), device=device))
                         for name in ("model", "prior")} for kind in KINDS}
    state = {name: torch.zeros((H, 3), device=device) for name in ("model", "persistence")}
    num_frames = 0

    def cells(scores):
        return heads * S + (torch.searchsorted(edges, scores.contiguous(), right=True) - 1).clamp(min=0)

    for video_id in tqdm(data["videos"], desc="Scoring rollouts"):
        layers = step05_layer._video_layers(data, video_id, cfg)
        num_frames += data["videos"][video_id]["shape"][0]
        # Empty layers after the video: no agent is visible there, so no rows.
        padded = torch.cat([layers, layers.new_zeros((H, *layers.shape[1:]))])
        for starts in torch.arange(1, layers.shape[0]).split(cfg["windows_per_video"]):
            windows = step05_layer._windows(padded, starts, K, H)
            last_all = windows[:, K - 1].reshape(-1, A) > 0.5
            for h, out in enumerate(step05_layer.unroll(model, windows, H, feed_prob=1.0)):
                if len(out["rows"]) == 0:
                    continue
                last, target, pred = last_all[out["rows"]], out["target"] > 0.5, out["pred"]
                for kind, risk, event, score in (("onset", ~last, ~last & target, pred),
                                                 ("offset", last, last & ~target, 1 - pred)):
                    i = KINDS.index(kind)
                    for name, scores in (("model", score), ("prior", priors[h, i].expand_as(score))):
                        positives, negatives = histograms[kind][name]
                        row_cells = cells(scores)
                        positives[h] += torch.bincount(row_cells[event], minlength=A * S)
                        negatives[h] += torch.bincount(row_cells[risk & ~event], minlength=A * S)
                for name, on in (("model", pred > 0.5), ("persistence", last)):
                    state[name][h] += torch.stack([(on & target).sum(), (on & ~target).sum(), (~on & target).sum()])

    return {
        "histograms": {kind: {name: (p.view(H, A, S).cpu(), n.view(H, A, S).cpu()) for name, (p, n) in v.items()}
                       for kind, v in histograms.items()},
        "state": {name: v.cpu() for name, v in state.items()},
        "num_frames": num_frames,
    }


def select_thresholds(scores):
    """{kind: {scorer: [best-F1 threshold of each horizon]}} of the `score_videos` scores (of the val videos)."""
    return {kind: {name: [step06._histogram_metrics(p[h].sum(dim=0), n[h].sum(dim=0), 0.5)["best_f1"]["threshold"]
                          for h in range(len(p))]
                   for name, (p, n) in by_scorer.items()}
            for kind, by_scorer in scores["histograms"].items()}


def evaluate(scores, thresholds, cfg):
    """Metrics per horizon of the `score_videos` scores of the test videos (see the module doc)."""
    minutes = scores["num_frames"] / cfg["frame_rate"] / 60
    H = cfg["rollout_steps"]
    metrics = {"block_size": cfg["block_size"], "num_layers": cfg["num_layers"], "minutes": minutes,
               "frames_ahead": [h * cfg["block_size"] for h in range(1, H + 1)], "threshold_from": "val (best F1)"}
    for kind, by_scorer in scores["histograms"].items():
        metrics[kind] = {}
        for h in range(H):
            metrics[kind][str(h + 1)] = {}
            for name, (positives, negatives) in by_scorer.items():
                with_events = torch.nonzero(positives[h].sum(dim=1)).flatten().tolist()
                per_head = [step06._histogram_metrics(positives[h, a], negatives[h, a], 0.5)["average_precision"]
                            for a in with_events]
                metrics[kind][str(h + 1)][name] = {
                    **step06._histogram_metrics(positives[h].sum(dim=0), negatives[h].sum(dim=0), 0.5),
                    "macro_average_precision": sum(per_head) / len(per_head) if per_head else 0.0,
                    "detection": step06._detection(positives[h], negatives[h], thresholds[kind][name][h], minutes),
                }
    metrics["state"] = {str(h + 1): {name: step06.compute_metrics(*(int(x) for x in counts[h]))
                                     for name, counts in scores["state"].items()}
                        for h in range(H)}
    return metrics


# -------------- Shared task: LayeredRuleNet vs TemporalRuleNet vs OnsetMLP --------------
@torch.no_grad()
def score_shared_onsets(model, temporal, mlp, baselines, data, priors, cfg):
    """Onset scores of every model on one shared task, the same rows for all.

    Row: an agent visible in layers b - 1 and b and an atom off in layer b - 1;
    event: the atom is on in layer b. Every model sees the frames before layer b
    (up to frame b * block_size - 1): "layered" the h = 1 prediction of the
    LayeredRuleNet; "temporal" and "mlp" the onset probability of the step 05
    TemporalRuleNet and OnsetMLP for target frame b * block_size (an onset within
    their first window (10 frames), which covers layer b when >= block_size);
    "anchor" the anchor rules alone (`step06.fit_baselines`); "prior" the layer
    base rate of `layer_priors`. mlp / baselines may be None (left out).

    Returns ({scorer: (positives, negatives)} (A, S) `step06.score_edges`
    histograms of the rows with and without the event, the frames of all videos).
    """
    model.eval()
    device, s, K, A = cfg["device"], cfg["block_size"], model.num_layers, step05.label_atom_count(data)
    edges = step06.score_edges().to(device)
    S = len(edges)
    heads = torch.arange(A, device=device)
    prior = priors[0, KINDS.index("onset")].to(device)
    baselines = {name: value.to(device) for name, value in baselines.items()} if baselines else None
    names = ["layered", "temporal"] + (["mlp"] if mlp is not None else []) + \
            (["anchor"] if baselines else []) + ["prior"]
    histograms = {name: (torch.zeros(A * S, device=device), torch.zeros(A * S, device=device)) for name in names}
    num_frames = 0
    for video_id in tqdm(data["videos"], desc="Scoring the shared onset task"):
        X = step05._video_from_cells(data["videos"][video_id], A, device)
        num_frames += X.shape[0]
        layers = step05.coarsen_video(X, s).float()
        for starts in torch.arange(1, layers.shape[0], device=device).split(cfg["windows_per_video"]):
            out = step05_layer.unroll(model, step05_layer._windows(layers, starts, K, 1), 1)[0]
            if len(out["rows"]) == 0:
                continue
            # Row w * N + n of both: agent n before layer starts[w], i.e. before frame starts[w] * s.
            roles = step05.video_roles(data["videos"][video_id], device)
            inputs = (temporal.layer_inputs(X, starts * s) if not temporal.roles
                      else temporal.layer_inputs(X, starts * s, roles=roles))[out["rows"]]
            # The step 05 models give one probability per window; the first (10 frames) covers layer b.
            scores = {"layered": out["pred"], "temporal": temporal.window_probs(temporal(inputs)[0])[:, 0],
                      "prior": prior.expand_as(out["pred"])}
            if mlp is not None:
                scores["mlp"] = mlp.window_probs(mlp(inputs)[0])[:, 0]
            if baselines:
                scores["anchor"] = step06._anchor_scores(temporal, inputs, baselines)[:, 0]
            last, target = out["prev"] > 0.5, out["target"] > 0.5
            risk, event = ~last, ~last & target
            for name, score in scores.items():
                positives, negatives = histograms[name]
                row_cells = heads * S + (torch.searchsorted(edges, score.contiguous(), right=True) - 1).clamp(min=0)
                positives += torch.bincount(row_cells[event], minlength=A * S)
                negatives += torch.bincount(row_cells[risk & ~event], minlength=A * S)
    return {name: (p.view(A, S).cpu(), n.view(A, S).cpu()) for name, (p, n) in histograms.items()}, num_frames


def compare_models(val_scores, test_scores, cfg, temporal_horizon):
    """Test metrics of each scorer of `score_shared_onsets` on the shared task:
    AP (micro, macro over the heads with events), best F1, and the detected /
    missed events at its best-F1 threshold on the val videos."""
    (val_histograms, _), (histograms, num_frames) = val_scores, test_scores
    minutes = num_frames / cfg["frame_rate"] / 60
    results = {"task": f"onset in the next layer of {cfg['block_size']} frames, from the frames before it",
               "block_size": cfg["block_size"], "temporal_target_horizon": temporal_horizon,
               "minutes": minutes, "threshold_from": "val (best F1)", "scorers": {}}
    for name, (positives, negatives) in histograms.items():
        val_p, val_n = val_histograms[name]
        threshold = step06._histogram_metrics(val_p.sum(dim=0), val_n.sum(dim=0), 0.5)["best_f1"]["threshold"]
        per_head = [step06._histogram_metrics(positives[a], negatives[a], 0.5)["average_precision"]
                    for a in torch.nonzero(positives.sum(dim=1)).flatten().tolist()]
        results["scorers"][name] = {
            **step06._histogram_metrics(positives.sum(dim=0), negatives.sum(dim=0), 0.5),
            "macro_average_precision": sum(per_head) / len(per_head) if per_head else 0.0,
            "detection": step06._detection(positives, negatives, threshold, minutes),
        }
    return results


def print_comparison(comparison):
    print(f"[Shared task] {comparison['task']} (TemporalRuleNet / MLP horizon "
          f"{comparison['temporal_target_horizon']} frames); thresholds from {comparison['threshold_from']}")
    first = next(iter(comparison["scorers"].values()))
    print(f"  {step06._short_num(first['at_risk'])} at-risk rows, {first['onsets']} onsets, base rate {first['base_rate']:.4f}")
    print(f"  {'scorer':<10}{'AP':>7}{'macro':>7}{'best F1':>9}{'events':>8}{'detected':>9}{'false al.':>10}"
          f"{'recall':>8}{'precision':>10}")
    for name, m in comparison["scorers"].items():
        d = m["detection"]
        print(f"  {name:<10}{m['average_precision']:>7.3f}{m['macro_average_precision']:>7.3f}"
              f"{m['best_f1']['f1']:>9.3f}{d['events']:>8}{d['detected']:>9}{d['false_alarms']:>10}"
              f"{d['recall']:>8.3f}{d['precision']:>10.3f}")
    print()


def evaluate_clauses(model, data, clauses, cfg):
    """The step05_layer clauses with their test_ counts added (as `step06.evaluate_clauses`), sorted by test confidence."""
    def ratio(a, b):
        return float(a) / float(b) if b > 0 else None

    counts = step05_layer._rule_head_counts(model, data, list(data["videos"]), cfg,
                                            desc="Counting rule firings (test)")
    evaluated = []
    for clause in clauses:
        c, r, h = counts[clause["kind"]], clause["rule"], clause["head"][0]
        evaluated.append({
            **clause,
            "test_support": int(c["hits"][r, h]),
            "test_fires": int(c["fires"][r, h]),
            "test_confidence": ratio(c["hits"][r, h], c["fires"][r, h]),
            "test_anchor_confidence": ratio(c["anchor_hits"][r, h], c["anchor_fires"][r, h]),
            "test_recall": ratio(c["hits"][r, h], c["events"][h]),
            "test_base_rate": ratio(c["events"][h], c["at_risk"][h]),
        })
    evaluated.sort(key=lambda c: -(c["test_confidence"] or 0))
    return evaluated


def plot_results(metrics, clauses, cfg):
    """Save to cfg["figure_dir"]: layered_horizons_*.pdf, AP (onset, offset) and
    state F1 per horizon against the baselines; layered_clause_train_vs_test_*.pdf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(cfg["figure_dir"], exist_ok=True)
    suffix = f"b{cfg['block_size']}_k{cfg['num_layers']}"
    frames = metrics["frames_ahead"]
    horizons = [str(h) for h in range(1, len(frames) + 1)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, kind in zip(axes, KINDS):
        for name, color in (("model", "tab:blue"), ("prior", "tab:gray")):
            ax.plot(frames, [metrics[kind][h][name]["average_precision"] for h in horizons], marker="o",
                    color=color, label=f"{name} micro")
            ax.plot(frames, [metrics[kind][h][name]["macro_average_precision"] for h in horizons], marker="s",
                    linestyle="--", color=color, label=f"{name} macro")
        ax.set(title=f"{kind} AP", xlabel="frames after the last observed layer", ylabel="AP", ylim=(0, None))
        ax.legend(fontsize=7)
    for name, color in (("model", "tab:blue"), ("persistence", "tab:gray")):
        axes[2].plot(frames, [metrics["state"][h][name]["f1"] for h in horizons], marker="o", color=color, label=name)
    axes[2].set(title="state F1 (on atoms)", xlabel="frames after the last observed layer", ylabel="F1")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(cfg["figure_dir"] / f"layered_horizons_{suffix}.pdf")
    plt.close(fig)

    rows = [c for c in clauses if c["test_confidence"] is not None]
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        step05.clause_scatter(ax, rows, "confidence", "test_confidence")
        ax.set(title="Layered clause confidence, train vs test", xlabel="train confidence",
               ylabel="test confidence")
        fig.tight_layout()
        fig.savefig(cfg["figure_dir"] / f"layered_clause_train_vs_test_{suffix}.pdf")
        plt.close(fig)
    print(f"[Figures]\t {cfg['figure_dir']}")


def print_summary(metrics, clauses, cfg, num_examples=15):
    print(f"[Evaluation Summary] layered rollout on the test split, layers of {metrics['block_size']} frames, "
          f"{metrics['num_layers']} layers of context, {metrics['minutes']:.1f} min of video")
    print(f"  {'ahead':>6}  {'kind':<7}{'AP model':>9}{'AP prior':>9}{'macro':>7}{'events':>8}{'detected':>9}"
          f"{'false al.':>10}{'precision':>10}")
    for h, frames in enumerate(metrics["frames_ahead"], start=1):
        for kind in KINDS:
            m, prior = metrics[kind][str(h)]["model"], metrics[kind][str(h)]["prior"]
            d = m["detection"]
            print(f"  {frames:>5}f  {kind:<7}{m['average_precision']:>9.3f}{prior['average_precision']:>9.3f}"
                  f"{m['macro_average_precision']:>7.3f}{d['events']:>8}{d['detected']:>9}{d['false_alarms']:>10}"
                  f"{d['precision']:>10.3f}")
        state = metrics["state"][str(h)]
        print(f"  {frames:>5}f  state  F1 model {state['model']['f1']:.3f}, persistence {state['persistence']['f1']:.3f}")
    print()

    firing = [c for c in clauses if c["test_confidence"] is not None]
    print(f"[Clauses]\t {len(clauses)} from step05_layer, {len(firing)} fire on test")
    extended = [c for c in firing if len(c["bodies"]) > 1 and c["test_anchor_confidence"] is not None]
    if extended:
        better = sum(c["test_confidence"] > c["test_anchor_confidence"] for c in extended)
        print(f"[Extension]\t {better}/{len(extended)} extended clauses beat their anchor on test")
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    for c in firing[:num_examples]:
        print(f"  conf {c['test_confidence']:.2f} (train {c['confidence']:.2f})  support {c['test_support']:>5}  "
              f"{step05_layer._layered_clause_text(c, unique_atoms)}")
    print()


def main(input_data):
    print("\n--------- Step 06 (layered) Evaluation ----------------------\n")
    cfg = step05_layer.get_cfg(input_data["step05_input"])
    suffix = f"b{cfg['block_size']}_k{cfg['num_layers']}"
    cfg.update({
        "device": input_data["device"],
        "frame_rate": input_data["frame_rate"],
        "test_data_file": input_data["output_dir"] / "test_data.pt",
        "prior_file": input_data["output_dir"] / f"layered_priors_{suffix}.pt",
        "metrics_file": input_data["output_dir"] / f"layered_test_metrics_{suffix}.json",
        "test_clause_file": input_data["output_dir"] / f"layered_test_clauses_{suffix}.json",
        "comparison_file": input_data["output_dir"] / f"layered_comparison_{suffix}.json",
        # The anchor baseline of step 06 (fitted there), used if present.
        "baseline_file": input_data["output_dir"] / "baselines.pt",
        "figure_dir": input_data["output_dir"] / "figures",
    })
    if not os.path.exists(cfg["model_file"]):
        print(f"[Skip]\t\t no layered model {cfg['model_file']}; run step05_layer first")
        return
    step05_cfg = cfg["step05_cfg"]

    # 1. Test videos, the same file as step 06 (same min_off_frames, so step 06 keeps it).
    test_data = torch.load(cfg["test_data_file"], map_location="cpu") if os.path.exists(cfg["test_data_file"]) else {}
    if (test_data.get("min_off_frames") != step05_cfg["min_off_frames"]
            or test_data.get("roles", []) != list(step05_cfg["roles"])
            or test_data.get("geometry", []) != list(step05_cfg["geometry"])):
        unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
        test_data = step05.prepare_videos(input_data["test_atoms_file"], len(unique_atoms), cfg["device"],
                                          step05_cfg["min_off_frames"], input_data.get("test_box_file"),
                                          step05_cfg["roles"], step05_cfg["geometry"], unique_atoms)
        torch.save(test_data, cfg["test_data_file"])
    model = step05_layer.load_layered_rule_net(test_data, cfg)

    # 2. Rollout metrics per horizon; thresholds and priors from the step 05 training data.
    training_data = None
    if not step05.is_fresh(cfg["prior_file"], cfg["training_data_file"]):
        training_data = torch.load(cfg["training_data_file"], map_location="cpu")
        torch.save(layer_priors(training_data, cfg), cfg["prior_file"])
    priors = torch.load(cfg["prior_file"], map_location="cpu")
    if not step05.is_fresh(cfg["metrics_file"], cfg["model_file"], cfg["test_data_file"], cfg["prior_file"]):
        training_data = training_data or torch.load(cfg["training_data_file"], map_location="cpu")
        val_scores = score_videos(model, step06._subset(training_data, training_data["val_videos"]), priors, cfg)
        test_scores = score_videos(model, test_data, priors, cfg)
        utils_data.save_json(evaluate(test_scores, select_thresholds(val_scores), cfg), cfg["metrics_file"])
    metrics = utils_data.load_json(cfg["metrics_file"])

    # 2b. The shared onset task with the step 05 TemporalRuleNet, OnsetMLP and anchor baseline.
    comparison = None
    step05_cfg["device"] = cfg["device"]
    if os.path.exists(step05_cfg["model_file"]):
        if not step05.is_fresh(cfg["comparison_file"], cfg["model_file"], step05_cfg["model_file"],
                               step05_cfg["mlp_model_file"], cfg["baseline_file"], cfg["test_data_file"],
                               cfg["prior_file"]):
            training_data = training_data or torch.load(cfg["training_data_file"], map_location="cpu")
            temporal = step05.load_temporal_rule_net(test_data, step05_cfg)
            mlp = step05.load_onset_mlp(test_data, step05_cfg)
            baselines = (torch.load(cfg["baseline_file"], map_location="cpu")
                         if os.path.exists(cfg["baseline_file"]) else None)
            val = step06._subset(training_data, training_data["val_videos"])
            utils_data.save_json(compare_models(
                score_shared_onsets(model, temporal, mlp, baselines, val, priors, cfg),
                score_shared_onsets(model, temporal, mlp, baselines, test_data, priors, cfg),
                cfg, step05_cfg["windows"][0]), cfg["comparison_file"])
        comparison = utils_data.load_json(cfg["comparison_file"])
    else:
        print(f"[Skip]\t\t shared task: no step 05 model {step05_cfg['model_file']}")

    # 3. The step05_layer clauses on the test videos.
    if not step05.is_fresh(cfg["test_clause_file"], cfg["clause_file"], cfg["test_data_file"]):
        clauses = utils_data.load_json(cfg["clause_file"])
        utils_data.save_json(evaluate_clauses(model, test_data, clauses, cfg), cfg["test_clause_file"])
    test_clauses = utils_data.load_json(cfg["test_clause_file"])

    print(f"[Results]\t {cfg['metrics_file']}")
    plot_results(metrics, test_clauses, cfg)
    print_summary(metrics, test_clauses, cfg)
    if comparison is not None:
        print(f"[Results]\t {cfg['comparison_file']}")
        print_comparison(comparison)
    print("\n--------- Step 06 (layered) Done ---------------\n")
