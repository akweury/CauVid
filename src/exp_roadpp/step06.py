from pathlib import Path
import hashlib
import json
import torch

from src.exp_roadpp import utils_data
from src.exp_roadpp import step05



def _atom_label(atom):
    """Readable name of a flat (key, value, ...) atom signature, e.g. "Car Stop"."""
    atom = dict(zip(atom[::2], atom[1::2]))
    return f"{atom['agent_class_label']} {atom.get('action_id_label', atom.get('location_name_label'))}"


def _clause_text(clause, train_unique_atoms):
    """e.g. "Car MovAway[0] within 5 <- Car Stop[0] & Ped MovTow[1]"; [slot] is the object slot
    of step05's `build_features` (0: head object, body slot: body object, body slot + 1: some
    other object)."""
    bodies = " & ".join(
        f"{_atom_label(train_unique_atoms[atom_index])}[{slot}]"
        for atom_index, _tag, slot in clause["bodies"]
    )
    head = _atom_label(train_unique_atoms[clause["head"][0]])
    return f"{head}[0] within {clause['delay_window']} <- {bodies}"


def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "iou": iou, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def predict_delays(clauses, train_unique_atoms, eval_atoms, delay_windows, device):
    """Ground the timed clauses on the eval split, as step05 grounds them for training.

    Returns (pred, clause_of, delay, node_keys, to_eval):
      pred[n, a]: smallest delay_window of a clause firing on node n with head
          atom a (max(delay_windows) + 1 if none fires);
      clause_of[n, a]: index in `clauses` of that clause (-1 if none);
      delay, node_keys: ground truth tables of `step05._node_tables`;
      to_eval[i]: eval atom id of train atom i (-1 if unseen in eval).
    """
    eval_vocab = {tuple(atom): i for i, atom in enumerate(eval_atoms["unique_atoms"])}
    to_eval = [eval_vocab.get(tuple(atom), -1) for atom in train_unique_atoms]

    # Group the clauses by anchor body in eval atom ids; a clause with an atom
    # unseen in the eval split never fires there.
    body_groups, group_clauses = {}, {}
    for clause_id, clause in enumerate(clauses):
        if min(to_eval[atom[0]] for atom in [*clause["bodies"], clause["head"]]) < 0:
            continue
        anchor = clause["bodies"][0]
        body = (to_eval[anchor[0]], "oid", anchor[2])
        head = (to_eval[clause["head"][0]], "oid", 0)
        body_groups.setdefault(body, {"candidate_heads": set()})["candidate_heads"].add(head)
        extras = [(atom[2], to_eval[atom[0]]) for atom in clause["bodies"][1:]]
        group_clauses.setdefault(body, []).append(
            (clause["delay_window"], clause["confidence"], clause_id, extras, head[0])
        )
    for body_group in body_groups.values():
        body_group["candidate_heads"] = tuple(sorted(body_group["candidate_heads"], key=repr))

    examples = step05.build_features(
        eval_atoms, None, body_groups, step05.build_head_vocab(body_groups, delay_windows),
        delay_windows, device=device,
    )
    _node_frame, _current, delay, node_keys = step05._node_tables(
        eval_atoms, max(delay_windows), device
    )
    pred = torch.full(delay.shape, max(delay_windows) + 1, dtype=torch.int16, device=device)
    clause_of = torch.full(delay.shape, -1, dtype=torch.long, device=device)
    # Clauses ordered by decreasing window, then increasing confidence: later
    # writes win, so each cell keeps the smallest window and, among those,
    # the most confident clause.
    fired = []
    for gid, body in enumerate(body_groups):
        column_ids = {
            (object_slot, atom_index): j
            for j, (_gid, object_slot, atom_index) in enumerate(examples["feature_vocab"][gid])
        }
        dense_features = step05.sparse_rows_to_dense(examples["features"][gid]).to(device)
        row_heads = examples["row_heads"][gid].to(device)
        for window, confidence, clause_id, extras, head in group_clauses[body]:
            columns = [column_ids.get(extra) for extra in extras]
            if None in columns:
                continue
            nodes = torch.unique(row_heads[dense_features[:, columns].all(dim=1)])
            fired.append((-window, confidence, clause_id, head, nodes))
        del dense_features
    for _neg_window, _confidence, clause_id, head, nodes in sorted(fired, key=lambda f: f[:3]):
        pred[nodes, head] = clauses[clause_id]["delay_window"]
        clause_of[nodes, head] = clause_id
    return pred, clause_of, delay, node_keys, to_eval


def evaluate_clauses(clauses, train_unique_atoms, eval_atoms, head_atoms, delay_windows,
                     device="cpu", num_explanations=0):
    """Score timed clauses per (frame t, agent observed at t, head atom).

    A clause predicts that its head starts for the head agent within its
    delay_window, hence also within every larger window. For window w, a cell
    is predicted if some clause with delay_window <= w fires on it, and is
    positive if the head is not active at t and starts within (t, t + w]
    (the targets of step05's `build_features`).
    head_atoms: train atom ids of the scored heads; keep it fixed across runs
    so that recall is comparable.
    Returns per-window metrics (overall and per video), bin_accuracy (among
    events predicted and starting within the largest window, the share whose
    predicted window is the smallest window containing the onset), the bin
    confusion matrix and up to num_explanations correct predictions with the
    clause that made them.
    """
    if any("delay_window" not in clause for clause in clauses):
        raise ValueError("Clauses have no delay_window: they come from a step05 run "
                         "before timed heads; rerun step05.")
    pred, clause_of, delay, node_keys, to_eval = predict_delays(
        clauses, train_unique_atoms, eval_atoms, delay_windows, device
    )
    heads = sorted({to_eval[atom] for atom in head_atoms if to_eval[atom] >= 0})
    heads = torch.tensor(heads, dtype=torch.long, device=device)
    pred, clause_of, delay = pred[:, heads], clause_of[:, heads], delay[:, heads]

    videos = sorted({video_id for video_id, _t, _tube in node_keys})
    video_index = {video_id: i for i, video_id in enumerate(videos)}
    node_video = torch.tensor([video_index[key[0]] for key in node_keys],
                              dtype=torch.long, device=device)

    def per_video_counts(cells):
        return torch.bincount(node_video, weights=cells.sum(dim=1).double(),
                              minlength=len(videos)).long().tolist()

    overall, per_video = [], {video_id: [] for video_id in videos}
    for window in delay_windows:
        predicted = pred <= window
        positive = (delay > 0) & (delay <= window)
        tp = per_video_counts(predicted & positive)
        fp = per_video_counts(predicted & ~positive)
        fn = per_video_counts(~predicted & positive)
        overall.append({"window": window, **compute_metrics(sum(tp), sum(fp), sum(fn))})
        for i, video_id in enumerate(videos):
            per_video[video_id].append({"window": window, **compute_metrics(tp[i], fp[i], fn[i])})

    # Bins 0..K-1: smallest window containing the (predicted) onset; K: none.
    num_bins = len(delay_windows)
    windows = torch.tensor(delay_windows, dtype=torch.long, device=device)
    pred_bin = torch.bucketize(pred.long(), windows)
    true_bin = torch.where(delay > 0, torch.bucketize(delay.long(), windows), num_bins)
    confusion = torch.bincount((true_bin * (num_bins + 1) + pred_bin).flatten(),
                               minlength=(num_bins + 1) ** 2).reshape(num_bins + 1, num_bins + 1)
    hits = confusion[:num_bins, :num_bins]
    bin_accuracy = float(hits.diag().sum() / hits.sum()) if hits.sum() > 0 else 0.0

    # The first correct prediction of each clause, so the sample spans clauses.
    explanations = []
    explained = set()
    correct = torch.nonzero((pred_bin == true_bin) & (true_bin < num_bins))
    for node, column in correct.tolist():
        if len(explanations) >= num_explanations:
            break
        clause_id = int(clause_of[node, column])
        if clause_id in explained:
            continue
        explained.add(clause_id)
        video_id, t, tube_uid = node_keys[node]
        clause = clauses[clause_id]
        explanations.append({
            "video": video_id, "frame": t, "tube_uid": tube_uid,
            "true_delay": int(delay[node, column]),
            "predicted_window": clause["delay_window"],
            "clause": _clause_text(clause, train_unique_atoms),
            "clause_confidence": clause["confidence"],
        })

    return {
        "delay_windows": list(delay_windows),
        "num_clauses": len(clauses),
        "num_scored_heads": len(heads),
        "overall": overall,
        "bin_accuracy": bin_accuracy,
        # rows: true bin, columns: predicted bin; the last label is "none".
        "bin_confusion": {"labels": [f"<={w}" for w in delay_windows] + ["none"],
                          "matrix": confusion.tolist()},
        "videos": per_video,
        "explanations": explanations,
    }


def plot_summary(summary, plot_file):
    """One chart of the timed result, from the bin confusion matrix: a 100% stacked
    bar per true delay bin (correct bin / predicted earlier / predicted later /
    missed), and dots for the precision of the predictions made in each bin."""
    import numpy as np
    import matplotlib.pyplot as plt

    surface, ink, ink_secondary, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    # Ordinal one-hue ramp (validated light->dark) + neutral gray for "missed".
    colors = {"correct": "#184f95", "earlier": "#3987e5", "later": "#86b6ef", "missed": "#c3c2b7"}

    windows = summary["delay_windows"]
    num_bins = len(windows)
    confusion = np.array(summary["bin_confusion"]["matrix"], dtype=float)
    events = confusion[:num_bins].sum(axis=1)
    shares = {name: np.zeros(num_bins) for name in colors}
    for i in range(num_bins):
        row = confusion[i] / events[i] if events[i] else np.zeros(num_bins + 1)
        shares["correct"][i] = row[i]
        shares["earlier"][i] = row[:i].sum()
        shares["later"][i] = row[i + 1:num_bins].sum()
        shares["missed"][i] = row[num_bins]
    predictions = confusion[:, :num_bins].sum(axis=0)
    precision = np.divide(confusion.diagonal()[:num_bins], predictions,
                          out=np.full(num_bins, np.nan), where=predictions > 0)

    lows = [1] + [w + 1 for w in windows[:-1]]
    tick_labels = [f"{lo}–{hi} frames\nn = {int(n)}" if lo < hi else f"{hi} frame\nn = {int(n)}"
                   for lo, hi, n in zip(lows, windows, events)]

    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=surface)
    ax.set_facecolor(surface)
    x = np.arange(num_bins)
    bottom = np.zeros(num_bins)
    labels = {"correct": "correct bin", "earlier": "predicted earlier bin",
              "later": "predicted later bin", "missed": "missed (no prediction)"}
    for name, color in colors.items():
        ax.bar(x, shares[name], 0.6, bottom=bottom, color=color, edgecolor=surface,
               linewidth=2, label=labels[name], zorder=2)
        bottom += shares[name]
    for i in range(num_bins):
        # Correct share at the top of its segment, clear of the precision dot.
        if shares["correct"][i] >= 0.12:
            ax.text(x[i], shares["correct"][i] - 0.015, f"{shares['correct'][i]:.0%}",
                    ha="center", va="top", color="white", fontsize=9, zorder=3)
        if not np.isnan(precision[i]):
            ax.text(x[i] + 0.06, precision[i], f"{precision[i]:.0%}", ha="left", va="center",
                    color=ink, fontsize=8, zorder=5,
                    bbox={"facecolor": surface, "edgecolor": "none", "pad": 1})
    ax.plot(x - 0.06, precision, linestyle="none", marker="o", markersize=8, color=ink,
            markeredgecolor=surface, markeredgewidth=2, label="precision of predictions in bin",
            zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9, color=ink_secondary)
    ax.set_xlabel("true time until the event starts", color=ink_secondary)
    ax.set_ylabel("share", color=ink_secondary)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: f"{v:.0%}"))
    ax.tick_params(colors=ink_secondary, length=0)
    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)

    largest = summary["overall"][-1]
    ax.set_title(
        f"Event + time prediction on val: F1 (within {largest['window']} frames) = "
        f"{largest['f1']:.2f}, bin accuracy = {summary['bin_accuracy']:.2f}, "
        f"{summary['num_clauses']} rules",
        fontsize=10, color=ink, loc="left",
    )
    # Legend top to bottom as the bars stack, then the precision dots.
    handles = dict(zip(*ax.get_legend_handles_labels()[::-1]))
    legend_labels = [labels[name] for name in reversed(colors)] + ["precision of predictions in bin"]
    ax.legend([handles[label] for label in legend_labels], legend_labels,
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9,
              labelcolor=ink_secondary)
    fig.tight_layout()
    fig.savefig(plot_file, facecolor=surface)
    plt.close(fig)
    print(f"[Plot] Saved {plot_file}")


def main(input_data):
    print("\n--------- Step 06 Evaluation ----------------------\n")
    cfg = {
        "output_dir": Path(input_data["output_dir"]),
        "device": input_data.get("device", "cpu"),
        "num_explanations": 200,
    }
    atoms = utils_data.load_atoms(input_data["val_atoms_file"])
    extended_clauses = utils_data.load_json(input_data["extended_clauses_file"])
    # Clause atom ids index the train split's unique atoms (saved by step03).
    train_unique_atoms = utils_data.load_json(input_data["anchor_clause_file"])["unique_atoms"]
    causal_anchor_clauses = utils_data.load_json(input_data["causal_anchor_clause_file"])
    head_atoms = sorted({c["anchor"]["head"][0] for c in causal_anchor_clauses.values()})
    print(f"Extended Clauses Num: {len(extended_clauses)}")
    if not extended_clauses:
        raise ValueError(f"No extended clauses in {input_data['extended_clauses_file']}")
    delay_windows = extended_clauses[0]["delay_windows"]

    # The summary is reused only if it scored exactly these clauses.
    clauses_hash = hashlib.sha256(
        json.dumps(extended_clauses, sort_keys=True).encode()
    ).hexdigest()
    summary_file = cfg["output_dir"] / "extended_clauses_evaluation.json"
    summary = utils_data.load_json(summary_file) if summary_file.exists() else None
    if summary is None or summary.get("clauses_hash") != clauses_hash:
        summary = evaluate_clauses(
            extended_clauses, train_unique_atoms, atoms, head_atoms, delay_windows,
            device=cfg["device"], num_explanations=cfg["num_explanations"],
        )
        summary["clauses_hash"] = clauses_hash
        utils_data.save_json(summary, summary_file)
    plot_summary(summary, cfg["output_dir"] / "step06_result.pdf")

    for m in summary["overall"]:
        print(f"[Overall] within {m['window']:>3} frames: recall={m['recall']:.4f}, "
              f"precision={m['precision']:.4f}, iou={m['iou']:.4f}, f1={m['f1']:.4f}")
    print(f"[Overall] bin accuracy={summary['bin_accuracy']:.4f}")
    for explanation in summary["explanations"][:5]:
        print(f"[Explanation] {explanation['video']} frame {explanation['frame']}: "
              f"{explanation['clause']} (true delay {explanation['true_delay']})")

    print("\n--------- Step 06 Done ---------------\n")
    return summary
