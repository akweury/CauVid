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


# -------------- Detection --------------
# The scorers whose detections are counted (the prior flags all rows of a head alike).
DETECTION_SCORERS = ("model", "anchor")


def _subset(data, video_ids):
    """`prepare_videos` data of only the given videos."""
    return {**data, "videos": {v: data["videos"][v] for v in video_ids}}


def select_thresholds(model, training_data, baselines, cfg):
    """Detection threshold of each scorer: the best-F1 threshold on the step 05
    validation videos, so the test counts are not tuned on the test set.
    Returns {"frame": {scorer: threshold}, "block": {scorer: threshold}}, the
    block level with tolerance cfg["detection_block_tolerance"]."""
    frame, blocks = _score_videos(model, _subset(training_data, training_data["val_videos"]), baselines, cfg)
    thresholds = {"frame": {}, "block": {}}
    for name in DETECTION_SCORERS:
        for level, (positives, negatives) in (
                ("frame", frame[name]),
                ("block", _block_histograms(blocks, name, cfg["detection_block_tolerance"],
                                            training_data["num_atoms"]))):
            best = _histogram_metrics(positives.sum(dim=0), negatives.sum(dim=0), cfg["prob_threshold"])["best_f1"]
            thresholds[level][name] = best["threshold"]
    return thresholds


def _detection(positives, negatives, threshold, minutes):
    """Event counts of (A, S) histograms flagged at the threshold: events (onsets),
    detected, missed, false_alarms, recall, precision, false_alarms_per_minute."""
    b = int(torch.searchsorted(score_edges(), torch.tensor(threshold), right=True)) - 1
    events, detected, false_alarms = int(positives.sum()), int(positives[:, b:].sum()), int(negatives[:, b:].sum())
    return {
        "threshold": threshold,
        "events": events,
        "detected": detected,
        "missed": events - detected,
        "false_alarms": false_alarms,
        "recall": detected / events if events else 0.0,
        "precision": detected / (detected + false_alarms) if detected + false_alarms else 0.0,
        "false_alarms_per_minute": false_alarms / minutes if minutes else None,
    }


# -------------- Model evaluation --------------
@torch.no_grad()
def _score_videos(model, data, baselines, cfg):
    """Onset scores of the model and of the baselines on every video of the split.

    Returns (frame, blocks):
      frame {scorer: (positives, negatives)}: (A, S) histograms of the scores over
          all at-risk (agent, frame, atom) rows, split into the rows where the
          atom switches on (positives) and the others; S bins of `score_edges`.
      blocks: the coarse level, frames merged into blocks of s = block_size:
          "events": one per block onset (the atom held in block b, not in b - 1,
              the agent visible in b - 1): "head" (E,), "true_frame" (E,) its first
              frame in b, and per scorer "scores" (E, K + 1), the highest score of
              the at-risk frames of blocks b, b - 1, ..., b - K (-1 if none), and
              for the model "frames" (E, K + 1), the frame of each of them;
          "num_frames": the frames of all videos;
          "negatives" {scorer: {k: (A, S)}}: histograms of the block scores of the
              other at-risk blocks, without the k blocks before an onset (those
              are the tolerance window of the onset, not false alarms).
    K is the largest of cfg["block_tolerances"].
    """
    device = cfg["device"]
    edges = score_edges().to(device)
    num_atoms, num_bins = data["num_atoms"], len(edges)
    s, K = cfg["block_size"], max(cfg["block_tolerances"])
    baselines = {name: value.to(device) for name, value in baselines.items()}
    scorers = {
        "model": lambda inputs: torch.sigmoid(model(inputs)[0]),
        "anchor": lambda inputs: _anchor_scores(model, inputs, baselines),
        "prior": lambda inputs: baselines["prior"].expand(len(inputs), -1),
    }

    def zeros():
        return torch.zeros(num_atoms * num_bins, device=device)

    def cells(heads, scores):
        return heads * num_bins + (torch.searchsorted(edges, scores.contiguous(), right=True) - 1).clamp(min=0)

    frame = {name: (zeros(), zeros()) for name in scorers}
    negatives = {name: {k: zeros() for k in cfg["block_tolerances"]} for name in scorers}
    events = {"head": [], "true_frame": [], "frames": [], "scores": {name: [] for name in scorers}}
    all_heads = torch.arange(num_atoms, device=device)
    num_frames = 0
    chunk = s * max(1, cfg["frames_per_video"] // s)            # whole blocks per chunk
    for video_id in tqdm(data["videos"], desc="Scoring test videos"):
        X = step05._video_from_cells(data["videos"][video_id], num_atoms, device)
        T, N, A = X.shape
        num_frames += T
        num_blocks = -(-T // s)
        # Highest score of the at-risk frames of each (block, agent, atom), -1 if none, and its frame offset.
        block_max = {name: torch.full((num_blocks, N, A), -1.0, device=device) for name in scorers}
        block_arg = torch.zeros((num_blocks, N, A), dtype=torch.long, device=device)
        for begin in range(0, T, chunk):
            frames = torch.arange(begin, min(begin + chunk, T), device=device)
            inputs, targets, at_risk, rows = step05.video_rows(model, X, frames, data["min_off_frames"])
            if len(inputs) == 0:
                continue
            t, n = rows.T
            b0, nb = begin // s, -(-len(frames) // s)
            for name, score in scorers.items():
                scores = score(inputs)
                positives_f, negatives_f = frame[name]
                row_cells = cells(all_heads, scores)
                positives_f += torch.bincount(row_cells[targets], minlength=len(positives_f))
                negatives_f += torch.bincount(row_cells[at_risk & ~targets], minlength=len(negatives_f))
                dense = torch.full((nb * s, N, A), -1.0, device=device)
                dense[t - begin, n] = torch.where(at_risk, scores, -1.0)
                values, offsets = dense.view(nb, s, N, A).max(dim=1)
                block_max[name][b0:b0 + nb] = values
                if name == "model":
                    block_arg[b0:b0 + nb] = offsets

        # Block onsets: held in b, not in b - 1, the agent visible in b - 1, some frame of b scored.
        Xc = step05.coarsen_video(X, s)
        onset = torch.zeros_like(Xc)
        onset[1:] = Xc[1:] & ~Xc[:-1] & Xc[:-1].any(dim=2, keepdim=True)
        at_risk_c = torch.zeros_like(Xc)
        at_risk_c[1:] = ~Xc[:-1] & Xc[:-1].any(dim=2, keepdim=True)
        at_risk_c &= block_max["model"] >= 0
        onset &= at_risk_c
        b, n, a = torch.nonzero(onset, as_tuple=True)
        padded = torch.cat([X, X.new_zeros(((-T) % s, N, A))]).view(num_blocks, s, N, A)
        events["head"].append(a)
        events["true_frame"].append(b * s + padded[b, :, n, a].int().argmax(dim=1))
        window = [b - j for j in range(K + 1)]                  # blocks b, b - 1, ..., b - K
        events["frames"].append(torch.stack([bj * s + block_arg[bj.clamp(min=0), n, a] for bj in window], dim=1))
        for name in scorers:
            events["scores"][name].append(torch.stack(
                [torch.where(bj >= 0, block_max[name][bj.clamp(min=0), n, a], -1.0) for bj in window], dim=1))
        # Negatives: at-risk blocks with no onset in them or in the next k blocks.
        near = onset.clone()
        for k in range(K + 1):
            if k > 0:
                near[:-k] |= onset[k:]
            if k not in cfg["block_tolerances"]:
                continue
            b_neg, _n, a_neg = torch.nonzero(at_risk_c & ~near, as_tuple=True)
            for name in scorers:
                negatives[name][k] += torch.bincount(cells(a_neg, block_max[name][at_risk_c & ~near]),
                                                     minlength=num_atoms * num_bins)

    def cpu(histogram):
        return histogram.view(num_atoms, num_bins).cpu()

    blocks = {
        "num_frames": num_frames,
        "head": torch.cat(events["head"]).cpu(),
        "true_frame": torch.cat(events["true_frame"]).cpu(),
        "frames": torch.cat(events["frames"]).cpu(),
        "scores": {name: torch.cat(v).cpu() for name, v in events["scores"].items()},
        "negatives": {name: {k: cpu(h) for k, h in v.items()} for name, v in negatives.items()},
    }
    return {name: (cpu(p), cpu(n)) for name, (p, n) in frame.items()}, blocks


def _block_histogram(heads, scores, num_atoms):
    """(A, S) histogram of the event scores of each head."""
    edges = score_edges()
    bins = (torch.searchsorted(edges, scores.contiguous(), right=True) - 1).clamp(min=0)
    return torch.bincount(heads * len(edges) + bins, minlength=num_atoms * len(edges)).view(num_atoms, len(edges))


def _block_histograms(blocks, name, k, num_atoms):
    """(A, S) positives and negatives of the block onsets of one scorer, an onset
    hit by the highest score of its block and the k blocks before it."""
    positives = _block_histogram(blocks["head"], blocks["scores"][name][:, :k + 1].max(dim=1).values, num_atoms)
    return positives, blocks["negatives"][name][k]


def evaluate_blocks(blocks, num_atoms, cfg):
    """Coarse-fine-focus metrics of the model and the baselines, per block tolerance k.

    coarse {k: {scorer: metrics}}: block onset detection; an onset in block b is
        hit by the highest score of blocks b - k..b (k = 0 the block itself,
        k = 1 also one block early). Metrics as `_histogram_metrics`, plus macro AP.
    fine {k: ...}: for the model, the frame error of each onset = true frame -
        frame of that highest score (> 0: predicted early), as mean / median of
        its absolute value and the share within each of cfg["frame_tolerances"].
    focus {k: ...}: the whole coarse -> fine pipeline of the model at its best-F1
        block threshold: the share of onsets whose block is flagged and whose
        frame is within each frame tolerance; work_share, the share of at-risk
        blocks flagged (where the fine level has to look).
    """
    threshold = cfg["prob_threshold"]
    coarse, fine, focus, frame_errors = {}, {}, {}, {}
    for k in cfg["block_tolerances"]:
        key = str(k)                                            # JSON keys are strings
        coarse[key] = {}
        for name in blocks["scores"]:
            positives, negatives = _block_histograms(blocks, name, k, num_atoms)
            per_head = [_histogram_metrics(positives[h], negatives[h], threshold)["average_precision"]
                        for h in torch.nonzero(positives.sum(dim=1)).flatten().tolist()]
            coarse[key][name] = {
                **_histogram_metrics(positives.sum(dim=0), negatives.sum(dim=0), threshold, with_curve=True),
                "macro_average_precision": sum(per_head) / len(per_head) if per_head else 0.0,
            }

        score, best = blocks["scores"]["model"][:, :k + 1].max(dim=1)
        error = blocks["true_frame"] - blocks["frames"][torch.arange(len(best)), best]
        scored = score >= 0
        abs_error = error[scored].abs().float()
        frame_errors[key] = error[scored].tolist()
        fine[key] = {
            "num_onsets": len(score),
            "mean_abs_error": abs_error.mean().item() if len(abs_error) else None,
            "median_abs_error": abs_error.median().item() if len(abs_error) else None,
            "within": {str(f): (abs_error <= f).float().mean().item() if len(abs_error) else None
                       for f in cfg["frame_tolerances"]},
        }

        model = coarse[key]["model"]
        flag = model["best_f1"]["threshold"]
        caught = scored & (score >= flag)
        num_flagged = model["best_f1"]["tp"] + model["best_f1"]["fp"]
        focus[key] = {
            "threshold": flag,
            "block_precision": model["best_f1"]["precision"],
            "block_recall": model["best_f1"]["recall"],
            "work_share": num_flagged / model["at_risk"] if model["at_risk"] else 0.0,
            "recall_within": {str(f): (caught & (error.abs() <= f)).float().mean().item() if len(score) else None
                              for f in cfg["frame_tolerances"]},
        }
    return {"block_size": cfg["block_size"], "block_tolerances": cfg["block_tolerances"], "coarse": coarse, "fine": fine, "focus": focus,
            "frame_errors": frame_errors}


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


def evaluate_model(model, data, baselines, thresholds, cfg):
    """Next-frame onset prediction of the TemporalRuleNet on the test split, over
    all heads together (micro) and per head atom with at least one onset; the
    same for the baselines of `fit_baselines` under "baselines", on the same rows.
    Each per-head entry also has the baselines' AP under "baseline_average_precision";
    "blocks" holds the coarse-fine-focus metrics of `evaluate_blocks`; "detection"
    the event counts at the `select_thresholds` thresholds, frame exact and per
    block (with tolerance cfg["detection_block_tolerance"])."""
    histograms, blocks = _score_videos(model, data, baselines, cfg)
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
    k = cfg["detection_block_tolerance"]
    minutes = blocks["num_frames"] / cfg["frame_rate"] / 60
    detection = {
        "threshold_from": "val (best F1)",
        "minutes": minutes,
        "block_size": cfg["block_size"],
        "block_tolerance": k,
        "frame_exact": {name: _detection(*histograms[name], thresholds["frame"][name], minutes)
                        for name in DETECTION_SCORERS},
        "block": {name: _detection(*_block_histograms(blocks, name, k, data["num_atoms"]),
                                   thresholds["block"][name], minutes)
                  for name in DETECTION_SCORERS},
    }
    return {
        "num_videos": len(data["videos"]),
        "min_off_frames": data["min_off_frames"],
        "detection": detection,
        "prob_threshold": cfg["prob_threshold"],
        **model_results,
        "baselines": {name: {k: r[k] for k in ("overall", "macro_average_precision")} for name, r in results.items()},
        "blocks": evaluate_blocks(blocks, data["num_atoms"], cfg),
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
    """The headline test numbers: detected / missed events and AP of the model
    and the baselines. The details (curves, per head, all block tolerances,
    frame errors) are in test_metrics.json."""
    detection = metrics["detection"]
    firing = [c for c in clauses if c["test_confidence"] is not None]
    extended = [c for c in firing if len(c["bodies"]) > 1 and c["test_anchor_confidence"] is not None]
    k = str(detection["block_tolerance"])
    return {
        "test_videos": metrics["num_videos"],
        "test_minutes": detection["minutes"],
        "min_off_frames": metrics["min_off_frames"],
        "detection": detection,
        "average_precision": {
            "frame_exact": {"model": metrics["overall"]["average_precision"],
                            **{name: m["overall"]["average_precision"] for name, m in metrics["baselines"].items()}},
            "block": {name: m["average_precision"] for name, m in metrics["blocks"]["coarse"][k].items()},
        },
        "clauses": {
            "num_clauses": len(clauses),
            "num_firing_on_test": len(firing),
            "num_extended_beating_anchor": f"{sum(c['test_confidence'] > c['test_anchor_confidence'] for c in extended)}"
                                           f"/{len(extended)}",
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

    def pr_curves(ax, curves, base_rate):
        # Precision on a log axis (it spans the base rate ~1e-5 up to 1); the
        # points of empty score bins (nothing predicted yet) are left out.
        lows = [base_rate]
        for label, m, color in curves:
            points = [(r, p) for r, p in zip(m["curve"]["recall"], m["curve"]["precision"]) if r > 0]
            ax.plot([r for r, _p in points], [p for _r, p in points], color=color, label=label)
            lows += [p for _r, p in points]
        ax.axhline(base_rate, color="tab:red", linestyle="--", label="base rate (random)")
        ax.set(xlabel="recall", ylabel="precision (log)", xlim=(0, 1), ylim=(step05.log_floor(lows), 1), yscale="log")
        ax.grid(which="major", alpha=0.3)

    # Precision-recall curve of all heads together.
    overall = metrics["overall"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    pr_curves(ax, [(f"TemporalRuleNet (AP {overall['average_precision']:.3f})", overall, "tab:blue")]
              + [(f"{name} baseline (AP {metrics['baselines'][name]['overall']['average_precision']:.3f})",
                  metrics["baselines"][name]["overall"], color)
                 for name, color in (("anchor", "tab:orange"), ("prior", "tab:gray"))],
              overall["base_rate"])
    best = overall["best_f1"]
    ax.scatter([best["recall"]], [best["precision"]], color="black", zorder=3,
               label=f"best F1 {best['f1']:.3f} @ {best['threshold']:.2g}")
    ax.set(title=f"Test onsets (off >= {metrics['min_off_frames']} frames before)")
    ax.legend(fontsize=7)
    save(fig, "precision_recall_curve.pdf")

    # Coarse level: block onset precision-recall per tolerance, model against the baselines.
    blocks = metrics["blocks"]
    for k, by_scorer in blocks["coarse"].items():
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        pr_curves(ax, [(f"{name} (AP {by_scorer[name]['average_precision']:.3f})", by_scorer[name], color)
                       for name, color in (("model", "tab:blue"), ("anchor", "tab:orange"), ("prior", "tab:gray"))],
                  by_scorer["model"]["base_rate"])
        ax.set(title=f"Block onsets ({blocks['block_size']} frames, up to {k} block(s) early)")
        ax.legend(fontsize=7)
        save(fig, f"coarse_precision_recall_k{k}.pdf")

    # Fine level: frame error of the highest-scoring frame in each onset's window.
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for k, errors in blocks["frame_errors"].items():
        if errors:
            ax.hist(errors, bins=range(min(errors), max(errors) + 2), histtype="step", linewidth=1.5,
                    label=f"up to {k} block(s) early (median |err| {blocks['fine'][k]['median_abs_error']:.0f})")
    ax.set(title="Frame error of block onsets (> 0: predicted early)", xlabel="true frame - predicted frame",
           ylabel="onsets")
    ax.legend()
    save(fig, "fine_frame_error.pdf")

    # Per head: AP and best F1, the heads with the highest AP.
    heads = sorted(metrics["per_head"].values(), key=lambda m: -m["average_precision"])[:num_heads]
    if heads:
        y = list(range(len(heads)))
        labels = [f"{m['label']} ({_short_num(m['onsets'])})" for m in heads]

        fig, ax = plt.subplots(figsize=(7, 0.25 * len(heads) + 1.5))
        step05.head_dot_plot(ax, labels,
                             [("model AP", [m["average_precision"] for m in heads], "tab:blue", "o"),
                              ("anchor baseline AP", [m["baseline_average_precision"]["anchor"] for m in heads],
                               "tab:orange", "D"),
                              ("base rate", [m["base_rate"] for m in heads], "tab:red", "|")],
                             reference=[m["base_rate"] for m in heads])
        ax.set(title=f"AP per head (top {len(heads)}, onsets in brackets), gray = lift over base rate",
               xlabel="average precision (log)")
        save(fig, "ap_per_head.pdf")

        fig, ax = plt.subplots(figsize=(7, 0.25 * len(heads) + 1.5))
        step05.head_dot_plot(ax, labels,
                             [("best F1", [m["best_f1"]["f1"] for m in heads], "tab:green", "o"),
                              ("precision", [m["best_f1"]["precision"] for m in heads], "tab:blue", "^"),
                              ("recall", [m["best_f1"]["recall"] for m in heads], "tab:orange", "x")])
        ax.set(title="Best F1 per head, with its precision and recall", xlabel="score (log)")
        save(fig, "f1_per_head.pdf")

    # Clauses on the test videos.
    rows = [c for c in clauses if c["test_confidence"] is not None]
    if not rows:
        print("[Figure]\t skipped the clause figures: no clause fires on test")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    step05.clause_scatter(ax, rows, "confidence", "test_confidence")
    ax.set(title="Clause confidence, train vs test", xlabel="train confidence", ylabel="test confidence")
    save(fig, "clause_train_vs_test.pdf")

    extended = [c for c in rows if len(c["bodies"]) > 1 and c["test_anchor_confidence"] is not None]
    if extended:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        step05.clause_scatter(ax, extended, "test_anchor_confidence", "test_confidence")
        better = sum(c["test_confidence"] > c["test_anchor_confidence"] for c in extended)
        ax.set(title=f"Extension gain on test ({better}/{len(extended)} above y = x)",
               xlabel="confidence of the anchor alone", ylabel="confidence of the extended clause")
        save(fig, "clause_extension_gain.pdf")
    else:
        print("[Figure]\t skipped clause_extension_gain.pdf: no extended clause fires on test")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    largest = max(c["test_support"] for c in rows) or 1
    low = step05.log_floor([c["test_confidence"] for c in rows])
    ax.scatter([c["test_recall"] or 0 for c in rows], step05.clip_low([c["test_confidence"] for c in rows], low),
               s=[4 + 60 * c["test_support"] / largest for c in rows], alpha=0.5, clip_on=False)
    ax.set(title="Clause confidence vs recall on test (size = test support)",
           xlabel="recall (share of head onsets hit)", ylabel="confidence (precision, log)",
           xlim=(0, 1), ylim=(low, 1), yscale="log")
    save(fig, "clause_precision_recall.pdf")


# -------------- Summary --------------
def print_summary(metrics, clauses, cfg, num_examples=15):
    print("[Evaluation Summary] next-frame onset prediction on the test split")
    detection = metrics["detection"]
    print(f"[Detection] events = onsets in {detection['minutes']:.1f} min of test video; "
          f"thresholds picked on {detection['threshold_from']}")
    print(f"  {'level':<15}{'scorer':<8}{'events':>8}{'detected':>10}{'missed':>8}{'false alarms':>14}"
          f"{'recall':>8}{'precision':>11}{'FA/min':>8}")
    levels = (("frame exact", detection["frame_exact"]),
              (f"block {detection['block_size']}f k{detection['block_tolerance']}", detection["block"]))
    for level, by_scorer in levels:
        for name, d in by_scorer.items():
            print(f"  {level:<15}{name:<8}{d['events']:>8}{d['detected']:>10}{d['missed']:>8}{d['false_alarms']:>14}"
                  f"{d['recall']:>8.3f}{d['precision']:>11.3f}{d['false_alarms_per_minute'] or 0:>8.2f}")
    print()
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

    blocks = metrics["blocks"]
    print(f"[Coarse-fine-focus] blocks of {blocks['block_size']} frames, k = blocks early allowed")
    for k, by_scorer in blocks["coarse"].items():
        aps = ", ".join(f"{name} {m['average_precision']:.3f}" for name, m in by_scorer.items())
        model, fine, focus = by_scorer["model"], blocks["fine"][k], blocks["focus"][k]
        print(f"[k={k} coarse]\t AP {aps}; best F1 {model['best_f1']['f1']:.3f} "
              f"(P {model['best_f1']['precision']:.3f}, R {model['best_f1']['recall']:.3f})")
        within = ", ".join(f"<={f}: {v:.2f}" for f, v in fine["within"].items() if v is not None)
        print(f"[k={k} fine]\t {fine['num_onsets']} onsets, median |frame error| {fine['median_abs_error']}, "
              f"share {within}")
        recall = ", ".join(f"<={f}: {v:.3f}" for f, v in focus["recall_within"].items() if v is not None)
        print(f"[k={k} focus]\t flags {focus['work_share']:.4f} of blocks; onsets found and placed {recall}")
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
        # Coarse-fine-focus: frames merged into blocks of block_size; an onset is hit
        # at the coarse level by a high score in its block or up to k blocks before
        # it (k in block_tolerances); frame errors are counted within frame_tolerances.
        "block_size": 10,
        "block_tolerances": [0, 1],
        "frame_tolerances": [0, 2, 5],
        # Detected / missed events are counted per block with this tolerance (one of block_tolerances).
        "detection_block_tolerance": 1,
        "frame_rate": input_data["frame_rate"],
        "baseline_file": input_data["output_dir"] / "baselines.pt",
        "test_data_file": input_data["output_dir"] / "test_data.pt",
        "metrics_file": input_data["output_dir"] / "test_metrics.json",
        "test_clause_file": input_data["output_dir"] / "test_clauses.json",
        "results_file": input_data["output_dir"] / "test_results.json",
        "figure_dir": input_data["output_dir"] / "figures",
    }
    step05_cfg["device"] = cfg["device"]
    assert cfg["detection_block_tolerance"] in cfg["block_tolerances"]

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
    #    The detection thresholds are picked on the step 05 validation videos.
    settings = {"block_size": metrics.get("blocks", {}).get("block_size"),
                "block_tolerances": metrics.get("blocks", {}).get("block_tolerances"),
                "detection_block_tolerance": metrics.get("detection", {}).get("block_tolerance")}
    if (not step05.is_fresh(cfg["metrics_file"], step05_cfg["model_file"], cfg["test_data_file"],
                            cfg["baseline_file"], step05_cfg["training_data_file"])
            or settings != {k: cfg[k] for k in settings}):
        training_data = torch.load(step05_cfg["training_data_file"], map_location="cpu")
        thresholds = select_thresholds(model, training_data, baselines, cfg)
        utils_data.save_json(evaluate_model(model, test_data, baselines, thresholds, cfg), cfg["metrics_file"])
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
