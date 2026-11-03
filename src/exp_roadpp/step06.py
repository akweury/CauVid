import os
import torch
from tqdm import tqdm
from src.exp_roadpp import step05, utils_data


def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "iou": iou, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def _short_num(n):
    """Format a number compactly, e.g. 143280 -> '143k', 1000000 -> '1M'."""
    n = float(n)
    for unit in ["", "k", "M", "B"]:
        s = f"{n:.0f}" if unit == "" or abs(n) >= 99.95 else f"{n:.1f}".rstrip("0").rstrip(".")
        if abs(float(s)) < 1000:
            return s + unit
        n /= 1000
    return f"{n:.1f}T"


def score_edges():
    """(S,) lower edges of the score bins: 0, then 40 log-spaced bins per decade
    from 1e-8 to 1e-2 (onsets are rare, so most scores are tiny), then steps of 0.01."""
    return torch.cat([torch.zeros(1), torch.logspace(-8, -2, 241)[:-1], torch.linspace(0.01, 0.99, 99)])


# -------------- Baselines --------------
@torch.no_grad()
def fit_baselines(model, training_data, step05_cfg):
    """Scores of two baselines, fitted on all step 05 training videos:
      prior (A,): onsets / at-risk rows of each head, the same for every row;
      anchor_confidence (R, A): for each seeded rule r (anchor body in one lag
          bin) and each head of its anchor clause, the confidence of the anchor
          input alone; -1 for other heads, free rules and anchors that never fire.
    The anchor baseline is the anchor clauses with lag bins but without the
    learned extra atoms, so the model beating it shows the extensions help."""
    counts = step05._rule_head_counts(model, training_data, list(training_data["videos"]), step05_cfg,
                                      desc="Counting anchor firings (train)")
    num_bins = model.num_bins
    is_anchor_head = torch.zeros_like(counts["anchor_fires"], dtype=torch.bool)
    for i, (_body, heads) in enumerate(step05.anchor_rules(step05_cfg)):
        is_anchor_head[i * num_bins:(i + 1) * num_bins, heads] = True
    valid = is_anchor_head & (counts["anchor_fires"] > 0)
    return {
        "prior": counts["onsets"] / counts["at_risk"].clamp(min=1),
        "anchor_confidence": torch.where(valid, counts["anchor_hits"] / counts["anchor_fires"].clamp(min=1), -1.0),
    }


def _anchor_scores(model, inputs, baselines, chunk_size=64):
    """(B, A) anchor baseline score of each row: the highest confidence of the
    anchors firing for the head, or the head's prior if none fires."""
    confidence = baselines["anchor_confidence"]
    used = (confidence >= 0).any(dim=1)
    confidence = confidence[used]
    anchor_body = model.fixed_inputs[used].float()
    anchor_size = anchor_body.sum(dim=1)
    fired = (inputs @ anchor_body.T >= anchor_size) & (anchor_size > 0)            # (B, R')
    best = torch.full((len(inputs), model.num_atoms), -1.0, device=inputs.device)
    if len(confidence):
        best = torch.cat([torch.where(chunk[:, :, None], confidence[None], -1.0).max(dim=1).values
                          for chunk in fired.split(chunk_size)])
    return torch.where(best >= 0, best, baselines["prior"])


# -------------- Model evaluation --------------
@torch.no_grad()
def _score_histograms(model, data, baselines, cfg):
    """{scorer: (positives, negatives)}, the (A, S) histograms of the onset
    scores of the model and of the baselines over all at-risk (agent, frame,
    atom) rows of the split, split into the rows where the atom switches on
    (positives) and the others (negatives); S score bins of `score_edges`."""
    device = cfg["device"]
    edges = score_edges().to(device)
    num_atoms, num_bins = data["num_atoms"], len(edges)
    baselines = {name: value.to(device) for name, value in baselines.items()}
    scorers = {
        "model": lambda inputs: torch.sigmoid(model(inputs)[0]),
        "anchor": lambda inputs: _anchor_scores(model, inputs, baselines),
        "prior": lambda inputs: baselines["prior"].expand(len(inputs), -1),
    }
    histograms = {name: (torch.zeros(num_atoms * num_bins, device=device),
                         torch.zeros(num_atoms * num_bins, device=device)) for name in scorers}
    heads = torch.arange(num_atoms, device=device) * num_bins
    for video_id in tqdm(data["videos"], desc="Scoring test videos"):
        for frames in data["frames"][video_id].split(cfg["frames_per_video"]):
            inputs, targets, at_risk = step05._video_rows(model, data, video_id, frames, device)
            if len(inputs) == 0:
                continue
            for name, score in scorers.items():
                bins = torch.searchsorted(edges, score(inputs).contiguous(), right=True) - 1
                cells = heads + bins.clamp(min=0)
                positives, negatives = histograms[name]
                positives += torch.bincount(cells[targets], minlength=len(positives))
                negatives += torch.bincount(cells[at_risk & ~targets], minlength=len(negatives))
    return {name: (positives.view(num_atoms, num_bins).cpu(), negatives.view(num_atoms, num_bins).cpu())
            for name, (positives, negatives) in histograms.items()}


def _histogram_metrics(positives, negatives, threshold, with_curve=False):
    """Onset metrics of one head (or all heads summed) from its score histograms
    over the `score_edges` bins: average precision, base rate, metrics at the
    probability threshold and at the threshold with the best F1; with_curve adds
    the precision-recall curve, one point per score bin from the highest threshold down."""
    edges = score_edges()
    num_bins = len(positives)
    # Predicting "on" for every score bin >= b, for b from the highest bin down.
    tp = positives.flip(0).cumsum(0)
    fp = negatives.flip(0).cumsum(0)
    num_positives, num_rows = positives.sum().item(), (positives + negatives).sum().item()
    precision = tp / (tp + fp).clamp(min=1)
    average_precision = (precision * positives.flip(0)).sum().item() / max(num_positives, 1)

    def at_bin(b):
        tp_b, fp_b = positives[b:].sum().item(), negatives[b:].sum().item()
        return compute_metrics(tp_b, fp_b, num_positives - tp_b)

    best_bin = max(range(num_bins), key=lambda b: at_bin(b)["f1"])
    curve = {}
    if with_curve:
        curve = {"curve": {
            "threshold": edges.flip(0).tolist(),
            "precision": precision.tolist(),
            "recall": (tp / max(num_positives, 1)).tolist(),
        }}
    return {
        "onsets": int(num_positives),
        "at_risk": int(num_rows),
        "base_rate": num_positives / num_rows if num_rows else 0.0,
        "average_precision": average_precision,
        "at_threshold": at_bin(int(torch.searchsorted(edges, torch.tensor(threshold), right=True)) - 1),
        "best_f1": {**at_bin(best_bin), "threshold": edges[best_bin].item()},
        **curve,
    }


def evaluate_model(model, data, baselines, cfg):
    """Next-frame onset prediction of the TemporalRuleNet on the test split, over
    all heads together (micro) and per head atom with at least one onset; the
    same for the baselines of `fit_baselines` under "baselines", on the same rows.
    Each per-head entry also has the baselines' AP under "baseline_average_precision"."""
    histograms = _score_histograms(model, data, baselines, cfg)
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    onset_heads = torch.nonzero(histograms["model"][0].sum(dim=1)).flatten().tolist()

    def scorer_metrics(positives, negatives):
        per_head = {h: _histogram_metrics(positives[h], negatives[h], cfg["prob_threshold"]) for h in onset_heads}
        return {
            "overall": _histogram_metrics(positives.sum(dim=0), negatives.sum(dim=0), cfg["prob_threshold"],
                                          with_curve=True),
            # Macro: every head with onsets counts the same.
            "macro_average_precision": (sum(m["average_precision"] for m in per_head.values()) / len(per_head)
                                        if per_head else 0.0),
            "per_head": per_head,
        }

    results = {name: scorer_metrics(*histograms[name]) for name in histograms}
    model_results = results.pop("model")
    for h, m in model_results["per_head"].items():
        m["label"] = step05._atom_name(unique_atoms[h])
        m["baseline_average_precision"] = {name: r["per_head"][h]["average_precision"] for name, r in results.items()}
    return {
        "num_videos": len(data["videos"]),
        "min_off_frames": data["min_off_frames"],
        "prob_threshold": cfg["prob_threshold"],
        **model_results,
        "baselines": {name: {k: r[k] for k in ("overall", "macro_average_precision")} for name, r in results.items()},
    }


# -------------- Clause evaluation --------------
def evaluate_clauses(model, data, clauses, step05_cfg):
    """The step 05 clauses with their test_ counts added: test_support (hits),
    test_fires, test_confidence, test_anchor_confidence (the anchor alone),
    test_recall (share of the head's onsets hit), test_base_rate. Sorted by
    test confidence."""
    def ratio(a, b):
        return float(a) / float(b) if b > 0 else None

    counts = step05._rule_head_counts(model, data, list(data["videos"]), step05_cfg,
                                      desc="Counting rule firings (test)")
    evaluated = []
    for clause in clauses:
        r, h = clause["rule"], clause["head"][0]
        evaluated.append({
            **clause,
            "test_support": int(counts["hits"][r, h]),
            "test_fires": int(counts["fires"][r, h]),
            "test_confidence": ratio(counts["hits"][r, h], counts["fires"][r, h]),
            "test_anchor_confidence": ratio(counts["anchor_hits"][r, h], counts["anchor_fires"][r, h]),
            "test_recall": ratio(counts["hits"][r, h], counts["onsets"][h]),
            "test_base_rate": ratio(counts["onsets"][h], counts["at_risk"][h]),
        })
    evaluated.sort(key=lambda c: -(c["test_confidence"] or 0))
    return evaluated


# -------------- Results --------------
def build_results(metrics, clauses):
    """The main test performance, without the per-head and per-clause details."""
    overall = metrics["overall"]
    firing = [c for c in clauses if c["test_confidence"] is not None]
    extended = [c for c in firing if len(c["bodies"]) > 1 and c["test_anchor_confidence"] is not None]
    confidences = sorted(c["test_confidence"] for c in firing)
    heads = sorted(metrics["per_head"].values(), key=lambda m: -m["average_precision"])
    def summary(m):
        return {"micro_average_precision": m["overall"]["average_precision"],
                "macro_average_precision": m["macro_average_precision"],
                "best_f1": {k: m["overall"]["best_f1"][k] for k in ("threshold", "precision", "recall", "f1")}}

    anchor_ap = metrics["baselines"]["anchor"]["overall"]["average_precision"]
    return {
        # The main check: the model's micro AP against the anchor rules alone (> 1 = extensions help).
        "micro_ap_gain_over_anchor": overall["average_precision"] / anchor_ap if anchor_ap > 0 else None,
        "baselines": {name: summary(m) for name, m in metrics["baselines"].items()},
        "model": {
            "num_videos": metrics["num_videos"],
            "min_off_frames": metrics["min_off_frames"],
            "at_risk_rows": overall["at_risk"],
            "onsets": overall["onsets"],
            "base_rate": overall["base_rate"],
            "micro_average_precision": overall["average_precision"],
            "macro_average_precision": metrics["macro_average_precision"],
            "prob_threshold": metrics["prob_threshold"],
            "at_threshold": {k: overall["at_threshold"][k] for k in ("precision", "recall", "f1")},
            "best_f1": {k: overall["best_f1"][k] for k in ("threshold", "precision", "recall", "f1")},
            "num_heads": len(heads),
            "top_heads": [{"label": m["label"], "average_precision": m["average_precision"],
                           "anchor_average_precision": m["baseline_average_precision"]["anchor"],
                           "base_rate": m["base_rate"], "onsets": m["onsets"]} for m in heads[:10]],
        },
        "clauses": {
            "num_clauses": len(clauses),
            "num_firing_on_test": len(firing),
            "median_test_confidence": confidences[len(confidences) // 2] if confidences else None,
            "num_extended_firing": len(extended),
            "num_extended_beating_anchor": sum(c["test_confidence"] > c["test_anchor_confidence"] for c in extended),
        },
    }


def plot_results(metrics, clauses, cfg, num_heads=30):
    """Save the step 06 figures, one PDF each, to cfg["figure_dir"]:
      precision_recall_curve.pdf: all heads together, against the base rate;
      ap_per_head.pdf: average precision of each head against its base rate;
      f1_per_head.pdf: best F1 of each head and its precision / recall there;
      clause_train_vs_test.pdf: train vs test confidence of every clause;
      clause_extension_gain.pdf: test confidence of the anchor alone vs the extended clause;
      clause_precision_recall.pdf: test confidence vs recall of every clause.
    A figure is skipped (with a message) when its data is missing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = cfg["figure_dir"]
    os.makedirs(figure_dir, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figure_dir / name)
        plt.close(fig)
        print(f"[Figure]\t {figure_dir / name}")

    def length_scatter(fig, ax, rows, x_key, y_key):
        # One dot per clause, colored by body length; dashed y = x.
        points = ax.scatter([c[x_key] for c in rows], [c[y_key] for c in rows],
                            c=[len(c["bodies"]) for c in rows], cmap="viridis", s=14, alpha=0.7)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        fig.colorbar(points, ax=ax, label="body length (atoms)")

    # Precision-recall curve of all heads together.
    overall = metrics["overall"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(overall["curve"]["recall"], overall["curve"]["precision"],
            label=f"TemporalRuleNet (AP {overall['average_precision']:.3f})")
    for name, style in (("anchor", "tab:orange"), ("prior", "tab:gray")):
        baseline = metrics["baselines"][name]["overall"]
        ax.plot(baseline["curve"]["recall"], baseline["curve"]["precision"], color=style,
                label=f"{name} baseline (AP {baseline['average_precision']:.3f})")
    ax.axhline(overall["base_rate"], color="tab:red", linestyle="--", label="base rate (random)")
    best = overall["best_f1"]
    ax.scatter([best["recall"]], [best["precision"]], color="black", zorder=3,
               label=f"best F1 {best['f1']:.3f} @ {best['threshold']:.2f}")
    ax.set(title=f"Test onsets (off >= {metrics['min_off_frames']} frames before)", xlabel="recall", ylabel="precision",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    save(fig, "precision_recall_curve.pdf")

    # Per head: AP and best F1, the heads with the highest AP.
    heads = sorted(metrics["per_head"].values(), key=lambda m: -m["average_precision"])[:num_heads]
    if heads:
        y = list(range(len(heads)))
        labels = [f"{m['label']} ({_short_num(m['onsets'])})" for m in heads]

        fig, ax = plt.subplots(figsize=(7, 0.25 * len(heads) + 1.5))
        ax.barh(y, [m["average_precision"] for m in heads], color="tab:blue", label="AP")
        ax.scatter([m["baseline_average_precision"]["anchor"] for m in heads], y, color="tab:orange",
                   marker="D", s=14, label="anchor baseline AP", zorder=3)
        ax.scatter([m["base_rate"] for m in heads], y, color="tab:red", marker="|", s=120,
                   label="base rate", zorder=3)
        ax.set_yticks(y, labels, fontsize=7)
        ax.invert_yaxis()
        ax.set(title=f"Average precision per head (top {len(heads)}, onsets in brackets)",
               xlabel="average precision", xlim=(0, 1))
        ax.legend(loc="lower right")
        save(fig, "ap_per_head.pdf")

        fig, ax = plt.subplots(figsize=(7, 0.25 * len(heads) + 1.5))
        ax.barh(y, [m["best_f1"]["f1"] for m in heads], color="tab:green", label="best F1")
        ax.scatter([m["best_f1"]["precision"] for m in heads], y, marker="o", s=12, color="tab:blue",
                   label="precision", zorder=3)
        ax.scatter([m["best_f1"]["recall"] for m in heads], y, marker="x", s=14, color="tab:orange",
                   label="recall", zorder=3)
        ax.set_yticks(y, labels, fontsize=7)
        ax.invert_yaxis()
        ax.set(title="Best F1 per head, with its precision and recall", xlabel="score", xlim=(0, 1))
        ax.legend(loc="lower right")
        save(fig, "f1_per_head.pdf")

    # Clauses on the test videos.
    rows = [c for c in clauses if c["test_confidence"] is not None]
    if not rows:
        print("[Figure]\t skipped the clause figures: no clause fires on test")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    length_scatter(fig, ax, rows, "confidence", "test_confidence")
    ax.set(title="Clause confidence, train vs test", xlabel="train confidence", ylabel="test confidence")
    save(fig, "clause_train_vs_test.pdf")

    extended = [c for c in rows if len(c["bodies"]) > 1 and c["test_anchor_confidence"] is not None]
    if extended:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        length_scatter(fig, ax, extended, "test_anchor_confidence", "test_confidence")
        better = sum(c["test_confidence"] > c["test_anchor_confidence"] for c in extended)
        ax.set(title=f"Extension gain on test ({better}/{len(extended)} above y = x)",
               xlabel="confidence of the anchor alone", ylabel="confidence of the extended clause")
        save(fig, "clause_extension_gain.pdf")
    else:
        print("[Figure]\t skipped clause_extension_gain.pdf: no extended clause fires on test")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    largest = max(c["test_support"] for c in rows) or 1
    ax.scatter([c["test_recall"] or 0 for c in rows], [c["test_confidence"] for c in rows],
               s=[4 + 60 * c["test_support"] / largest for c in rows], alpha=0.5)
    ax.set(title="Clause confidence vs recall on test (size = test support)",
           xlabel="recall (share of head onsets hit)", ylabel="confidence (precision)", xlim=(0, None), ylim=(0, 1))
    save(fig, "clause_precision_recall.pdf")


# -------------- Summary --------------
def print_summary(metrics, clauses, cfg, num_examples=15):
    print("[Evaluation Summary] next-frame onset prediction on the test split")
    overall = metrics["overall"]
    print(f"[Videos]\t {metrics['num_videos']}")
    print(f"[Rows]\t\t {_short_num(overall['at_risk'])} at-risk (agent, frame, atom), "
          f"{_short_num(overall['onsets'])} onsets, base rate {overall['base_rate']:.4f}")
    print(f"[AP]\t\t micro {overall['average_precision']:.3f}, macro {metrics['macro_average_precision']:.3f} "
          f"(random = base rate)")
    for name, baseline in metrics["baselines"].items():
        print(f"[AP {name}]\t micro {baseline['overall']['average_precision']:.3f}, "
              f"macro {baseline['macro_average_precision']:.3f} (baseline)")
    for name, m in (("@" + str(metrics["prob_threshold"]), overall["at_threshold"]),
                    (f"best F1 @{overall['best_f1']['threshold']:.2f}", overall["best_f1"])):
        print(f"[{name}]\t precision {m['precision']:.3f}, recall {m['recall']:.3f}, f1 {m['f1']:.3f}")
    print()

    print(f"[Per head, by AP] onsets  base rate  AP  anchor AP  best F1")
    rows = sorted(metrics["per_head"].values(), key=lambda m: -m["average_precision"])
    for m in rows[:num_examples]:
        print(f"{_short_num(m['onsets']):>7}  {m['base_rate']:.4f}  {m['average_precision']:.3f}  "
              f"{m['baseline_average_precision']['anchor']:.3f}  {m['best_f1']['f1']:.3f}  [{m['label']}]")
    print()

    print(f"[Clauses]\t {len(clauses)} from step 05")
    firing = [c for c in clauses if c["test_confidence"] is not None]
    if firing:
        confidences = sorted(c["test_confidence"] for c in firing)
        print(f"[Clauses]\t {len(firing)} fire on test, median confidence {confidences[len(confidences) // 2]:.3f}")
        extended = [c for c in firing if len(c["bodies"]) > 1 and c["test_anchor_confidence"] is not None]
        if extended:
            better = sum(c["test_confidence"] > c["test_anchor_confidence"] for c in extended)
            print(f"[Extension]\t {better}/{len(extended)} extended clauses beat their anchor on test")
        unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
        print(f"[Top {min(num_examples, len(firing))} clauses by test confidence]")
        for c in firing[:num_examples]:
            print(f"  conf {c['test_confidence']:.2f} (train {c['confidence']:.2f})  "
                  f"support {c['test_support']:>5}  {step05._temporal_clause_text(c, unique_atoms)}")
    print()


def main(input_data):
    print("\n--------- Step 06 Evaluation ----------------------\n")
    step05_cfg = step05.get_cfg(input_data["step05_input"])
    cfg = {
        "device": input_data["device"],
        "unique_atoms_file": input_data["unique_atoms_file"],
        "test_atoms_file": input_data["test_atoms_file"],
        "frames_per_video": step05_cfg["frames_per_video"],
        # A head is predicted to switch on in the next frame above this probability.
        "prob_threshold": 0.5,
        "baseline_file": input_data["output_dir"] / "baselines.pt",
        "test_data_file": input_data["output_dir"] / "test_data.pt",
        "metrics_file": input_data["output_dir"] / "test_metrics.json",
        "test_clause_file": input_data["output_dir"] / "test_clauses.json",
        "results_file": input_data["output_dir"] / "test_results.json",
        "figure_dir": input_data["output_dir"] / "figures",
    }
    step05_cfg["device"] = cfg["device"]

    # 1. Test videos in the format of the step 05 training data, with the same min_off_frames.
    test_data = torch.load(cfg["test_data_file"], map_location="cpu") if os.path.exists(cfg["test_data_file"]) else {}
    if test_data.get("min_off_frames") != step05_cfg["min_off_frames"]:
        num_atoms = len(utils_data.load_json(cfg["unique_atoms_file"]))
        test_data = step05.prepare_videos(cfg["test_atoms_file"], num_atoms, cfg["device"],
                                          step05_cfg["min_off_frames"])
        torch.save(test_data, cfg["test_data_file"])
    model = step05.load_temporal_rule_net(test_data, step05_cfg)

    # 2. Baselines fitted on the step 05 training videos: the head prior and the anchor rules alone.
    if not step05.is_fresh(cfg["baseline_file"], step05_cfg["model_file"], step05_cfg["training_data_file"]):
        training_data = torch.load(step05_cfg["training_data_file"], map_location="cpu")
        torch.save(fit_baselines(model, training_data, step05_cfg), cfg["baseline_file"])
    baselines = torch.load(cfg["baseline_file"], map_location="cpu")

    # 3. Next-frame onset prediction of the model and the baselines; redone whenever any input changes.
    metrics = utils_data.load_json(cfg["metrics_file"]) if os.path.exists(cfg["metrics_file"]) else {}
    if (not step05.is_fresh(cfg["metrics_file"], step05_cfg["model_file"], cfg["test_data_file"],
                            cfg["baseline_file"])
            or "baselines" not in metrics):
        utils_data.save_json(evaluate_model(model, test_data, baselines, cfg), cfg["metrics_file"])
    metrics = utils_data.load_json(cfg["metrics_file"])

    # 4. The step 05 clauses on the test videos; redone whenever they change.
    if not step05.is_fresh(cfg["test_clause_file"], step05_cfg["extended_clause_file"], cfg["test_data_file"]):
        clauses = utils_data.load_json(step05_cfg["extended_clause_file"])
        utils_data.save_json(evaluate_clauses(model, test_data, clauses, step05_cfg), cfg["test_clause_file"])
    test_clauses = utils_data.load_json(cfg["test_clause_file"])

    # 5. The main results (JSON) and the figures (one PDF each), on every run.
    utils_data.save_json(build_results(metrics, test_clauses), cfg["results_file"])
    print(f"[Results]\t {cfg['results_file']}")
    plot_results(metrics, test_clauses, cfg)
    print_summary(metrics, test_clauses, cfg)
    print("\n--------- Step 06 Done ---------------\n")
