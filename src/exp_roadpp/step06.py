from pathlib import Path
import hashlib
import json
import torch
from tqdm import tqdm
from src.exp_roadpp.cauvid import CauVid
from src.exp_roadpp import utils_data



def _atom_label(atom):
    """Readable name of a flat (key, value, ...) atom signature, e.g. "Car Stop"."""
    atom = dict(zip(atom[::2], atom[1::2]))
    return f"{atom['agent_class_label']} {atom.get('action_id_label', atom.get('location_name_label'))}"

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


def _print_row(correct, total, label):
    print(f"{_short_num(correct):>6} / {_short_num(total):<6} [{label}]")





def tasks_evaluation(cfg):
    """Raw lead-time results of the clauses on the val split: {video_id: (events,
    predictions) of `evaluate_video`}. Leads are eval_frame_step, 2 * eval_frame_step,
    ... up to the largest delay window."""
    
    test_tasks = torch.load(cfg["test_task_file"])
    model = CauVid(cfg)
    
    raw_results = {}
    for video_id, video_tasks in tqdm(test_tasks["tasks"].items(), desc="Evaluating Videos"):
        raw_results[video_id] = []
        for task in video_tasks:
            if len(task["input"]) == 0:
                continue
            predictions = model.predict(task["input"])
            raw_results[video_id].append({"gt": task["target"], "predictions": predictions})
    
    print(f"[Task Evaluation] Completed!")
    print()
    
    return raw_results


def _compute_metrics(gt, predictions):
    tp = sum(1 for g, p in zip(gt, predictions) if g == p == 1)
    fp = sum(1 for g, p in zip(gt, predictions) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gt, predictions) if g == 1 and p == 0)
    return tp, fp, fn

def result_analysis(cfg):
    raw_results = utils_data.load_json(cfg["raw_result_file"])
    num_videos = len(set(raw_results.keys()))
    num_events = 0
    summary = {"videos": []}
    for video_id, video_data in raw_results.items():
        print(f"Video ID: {video_id}")
        num_events += len(video_data)
        video_metrics = {"tp": 0, "fp": 0, "fn": 0}
        for event in video_data:
            gt = event["gt"]
            predictions = event["predictions"]
            tp, fp, fn = _compute_metrics(gt, predictions)
            video_metrics["tp"] += tp
            video_metrics["fp"] += fp
            video_metrics["fn"] += fn
        summary["videos"].append({"video_id": video_id, "metrics": video_metrics})   
    summary["num_videos"] = num_videos
    summary["num_events"] = num_events
    
    return summary
    
def print_summary(summary):
    print("[Evaluation Summary]")
    print(f"[Videos] \t {summary['num_videos']}")
    print(f"[Events] \t {summary['num_events']}")
    print(f"[Detected] \t {summary['detected_events']} events, mean earliest lead "
          f"{summary['mean_earliest_lead']:.1f} frames")
    print()

    # Timed: a clause whose lead bin contains L fires L frames before the onset;
    # any: a clause with the right head fires, whatever its bin.
    print(f"[Recall per lead]")
    overall = summary["overall"]
    for i, lead in enumerate(summary["leads"]):
        print(f"{lead:>4}f  timed {overall['recall_timed'][i]:.2f}  any {overall['recall_any'][i]:.2f}  "
              f"({_short_num(overall['timed'][i])} / {_short_num(overall['events'][i])})")
    print()

    print(f"[Precision per lead bin]")
    for row in summary["precision_per_bin"]:
        lo, hi = row["lead_bin"]
        _print_row(row["correct"], row["fired"], f"{lo + 1}-{hi}f  {row['precision']:.2f}")
    print()

    print(f"[Per atom: timed hits / events at each lead {summary['leads']}; precision]")
    for row in summary["per_atom"].values():
        hits = "  ".join(f"{_short_num(h)}/{_short_num(n)}" for h, n in zip(row["timed"], row["events"]))
        precision = f"{row['correct'] / row['fired']:.2f}" if row["fired"] else "-"
        print(f"{hits}  {precision:>5}  [{row['label']}]")
    print()
    
def main(input_data):
    print("\n--------- Step 06 Evaluation ----------------------\n")
    cfg = {
        "output_dir": input_data["output_dir"],
        "device": input_data["device"],
        "delay_windows": input_data["delay_windows"],
        'extended_clauses_file': input_data["extended_clauses_file"],
        "unique_atoms_file": input_data["unique_atoms_file"],
        "val_atoms_file": input_data["val_atoms_file"],
        # Clauses below this confidence (on train) are not used.
        "threshold": input_data["threshold"],
        "eval_frame_step": input_data["eval_frame_step"],
        'label_file': input_data["label_file"],
        "summary_file": input_data["output_dir"] / f"summary_{input_data['threshold']}.json",
        "raw_result_file": input_data["output_dir"] / f"raw_results_{input_data['threshold']}.json",
        "test_task_file": input_data["test_task_file"]
    }
    if not cfg["raw_result_file"].exists():
        raw_results = tasks_evaluation(cfg)
        utils_data.save_json(raw_results, cfg["raw_result_file"])
    if not cfg["summary_file"].exists():
        summary = result_analysis(cfg)
        utils_data.save_json(summary, cfg["summary_file"])
    summary = utils_data.load_json(cfg["summary_file"])
    print_summary(summary)
    print("\n--------- Step 06 Done ---------------\n")
