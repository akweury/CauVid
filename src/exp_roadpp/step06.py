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


def _is_fresh(output_file, *source_files):
    """True if output_file exists and is newer than every source file."""
    if not os.path.exists(output_file):
        return False
    return all(os.path.getmtime(output_file) >= os.path.getmtime(f) for f in source_files if os.path.exists(f))


# -------------- Model evaluation --------------
@torch.no_grad()
def _score_histograms(model, data, cfg):
    """(A, S) histograms of the predicted onset probabilities over all at-risk
    (agent, frame, atom) rows of the split, split into the rows where the atom
    switches on (positives) and the others (negatives); S score bins over [0, 1]."""
    num_atoms, num_bins = data["num_atoms"], cfg["num_score_bins"]
    device = cfg["device"]
    positives = torch.zeros(num_atoms * num_bins, device=device)
    negatives = torch.zeros(num_atoms * num_bins, device=device)
    heads = torch.arange(num_atoms, device=device) * num_bins
    for video_id in tqdm(data["videos"], desc="Scoring test videos"):
        for frames in data["frames"][video_id].split(cfg["frames_per_video"]):
            inputs, targets, at_risk = step05._video_rows(model, data, video_id, frames, device)
            if len(inputs) == 0:
                continue
            probabilities = torch.sigmoid(model(inputs)[0])
            cells = heads + (probabilities * num_bins).long().clamp(max=num_bins - 1)
            positives += torch.bincount(cells[targets], minlength=len(positives))
            negatives += torch.bincount(cells[at_risk & ~targets], minlength=len(negatives))
    return positives.view(num_atoms, num_bins).cpu(), negatives.view(num_atoms, num_bins).cpu()


def _histogram_metrics(positives, negatives, threshold):
    """Onset metrics of one head (or all heads summed) from its score histograms:
    average precision, base rate, metrics at the probability threshold and at
    the threshold with the best F1."""
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
    return {
        "onsets": int(num_positives),
        "at_risk": int(num_rows),
        "base_rate": num_positives / num_rows if num_rows else 0.0,
        "average_precision": average_precision,
        "at_threshold": at_bin(min(int(threshold * num_bins), num_bins - 1)),
        "best_f1": {**at_bin(best_bin), "threshold": best_bin / num_bins},
    }


def evaluate_model(model, data, cfg):
    """Next-frame onset prediction of the TemporalRuleNet on the test split, over
    all heads together (micro) and per head atom with at least one onset."""
    positives, negatives = _score_histograms(model, data, cfg)
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    per_head = {}
    for h in torch.nonzero(positives.sum(dim=1)).flatten().tolist():
        per_head[h] = {"label": step05._atom_name(unique_atoms[h]),
                       **_histogram_metrics(positives[h], negatives[h], cfg["prob_threshold"])}
    return {
        "num_videos": len(data["videos"]),
        "prob_threshold": cfg["prob_threshold"],
        "overall": _histogram_metrics(positives.sum(dim=0), negatives.sum(dim=0), cfg["prob_threshold"]),
        # Macro: every head with onsets counts the same.
        "macro_average_precision": (sum(m["average_precision"] for m in per_head.values()) / len(per_head)
                                    if per_head else 0.0),
        "per_head": per_head,
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


# -------------- Summary --------------
def print_summary(metrics, clauses, cfg, num_examples=15):
    print("[Evaluation Summary] next-frame onset prediction on the test split")
    overall = metrics["overall"]
    print(f"[Videos]\t {metrics['num_videos']}")
    print(f"[Rows]\t\t {_short_num(overall['at_risk'])} at-risk (agent, frame, atom), "
          f"{_short_num(overall['onsets'])} onsets, base rate {overall['base_rate']:.4f}")
    print(f"[AP]\t\t micro {overall['average_precision']:.3f}, macro {metrics['macro_average_precision']:.3f} "
          f"(random = base rate)")
    for name, m in (("@" + str(metrics["prob_threshold"]), overall["at_threshold"]),
                    (f"best F1 @{overall['best_f1']['threshold']:.2f}", overall["best_f1"])):
        print(f"[{name}]\t precision {m['precision']:.3f}, recall {m['recall']:.3f}, f1 {m['f1']:.3f}")
    print()

    print(f"[Per head, by AP] onsets  base rate  AP  best F1")
    rows = sorted(metrics["per_head"].values(), key=lambda m: -m["average_precision"])
    for m in rows[:num_examples]:
        print(f"{_short_num(m['onsets']):>7}  {m['base_rate']:.4f}  {m['average_precision']:.3f}  "
              f"{m['best_f1']['f1']:.3f}  [{m['label']}]")
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
        "num_score_bins": 100,
        "test_data_file": input_data["output_dir"] / "test_data.pt",
        "metrics_file": input_data["output_dir"] / "test_metrics.json",
        "test_clause_file": input_data["output_dir"] / "test_clauses.json",
    }
    step05_cfg["device"] = cfg["device"]

    # 1. Test videos in the format of the step 05 training data.
    if not os.path.exists(cfg["test_data_file"]):
        num_atoms = len(utils_data.load_json(cfg["unique_atoms_file"]))
        torch.save(step05.prepare_videos(cfg["test_atoms_file"], num_atoms, cfg["device"]), cfg["test_data_file"])
    test_data = torch.load(cfg["test_data_file"], map_location="cpu")
    model = step05.load_temporal_rule_net(test_data, step05_cfg)

    # 2. The model's next-frame onset prediction; redone whenever the model is retrained.
    if not _is_fresh(cfg["metrics_file"], step05_cfg["model_file"], cfg["test_data_file"]):
        utils_data.save_json(evaluate_model(model, test_data, cfg), cfg["metrics_file"])
    metrics = utils_data.load_json(cfg["metrics_file"])

    # 3. The step 05 clauses on the test videos; redone whenever they change.
    if not _is_fresh(cfg["test_clause_file"], step05_cfg["extended_clause_file"], cfg["test_data_file"]):
        clauses = utils_data.load_json(step05_cfg["extended_clause_file"])
        utils_data.save_json(evaluate_clauses(model, test_data, clauses, step05_cfg), cfg["test_clause_file"])
    test_clauses = utils_data.load_json(cfg["test_clause_file"])

    print_summary(metrics, test_clauses, cfg)
    print("\n--------- Step 06 Done ---------------\n")
