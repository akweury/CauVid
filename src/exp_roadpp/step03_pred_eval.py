import os

from src.exp_roadpp.utils_data import atom_to_signature
from src.exp_roadpp.step03_visual import visual_line_plot

def get_video_data(video_id, video_atoms, dataset_labels):
    data = {}
    data["video_id"] = video_id
    atoms_per_start_frame = {}
    for atom in video_atoms:
        if int(atom["start_frame"]) not in atoms_per_start_frame:
            atoms_per_start_frame[int(atom["start_frame"])] = []
        atoms_per_start_frame[int(atom["start_frame"])].append(atom)

    end_frame = max([int(atom["end_frame"]) for atom in video_atoms]) if video_atoms else 0
    # for each frame, we need a list to include the atoms that occur in that frame
    frame_data = {i: [] for i in range(end_frame)}
    for start_frame, atoms in atoms_per_start_frame.items():
        for atom in atoms:
            for frame_id in range(start_frame, int(atom["end_frame"])):
                frame_as = atom_to_signature(atom, dataset_labels)
                frame_data[frame_id].append(frame_as)

    data["frame_data"] = frame_data
    data["video_length"] = end_frame
    # connect the neighboring frames with the atoms that occur in them
    two_frames_data = {i: (frame_data[i], frame_data[i + 1]) for i in frame_data.keys() if i + 1 in frame_data}    
    data["two_frames_data"] = two_frames_data
    return data


def predict_next_n_steps(model, test_atoms, output_dir, dataset_labels, step=1):
    output_dir = output_dir/"predictions"
    os.makedirs(output_dir, exist_ok=True)
    # start predicting the next steps for each video in the test set
    print(f"Predicting next {step} steps for {len(test_atoms)} videos")
    start_frame_id = step
    all_video_scores = []
    for vid, video_atoms in test_atoms.items():
        print(f"Predicting for video: {vid}")
        # predict the actions frame by frame
        data = get_video_data(vid, video_atoms, dataset_labels)
        frame_results = []
        summary_counts = {
            event_type: {"tp": 0, "fp": 0, "fn": 0}
            for event_type in ("added", "removed", "unchanged")
        }
        rule_summary = {
            "target_rule_count": 0,
            "covered_rule_count": 0,
            "uncovered_rule_count": 0,
        }
        for frame_id, (siss, soss_gt) in data["two_frames_data"].items():
            if frame_id < start_frame_id:
                continue
            # given the current frame, predict the action/location of the atoms in the next frame
            meta_data = {"video_id": vid, "frame_id": frame_id, "output_dir": output_dir}
            soss = model.predict(siss, meta_data=None)
            # evaluate the prediction against the ground truth labels
            res = model.eval(siss, soss, soss_gt, meta_data=None)

            frame_results.append({
                "frame_id": frame_id,
                **res["events"],
                "rule_matches": res["rule_matches"],
            })
            for event_type, counts in res["events"].items():
                for count_name in ("tp", "fp", "fn"):
                    summary_counts[event_type][count_name] += counts[count_name]
            for count_name in rule_summary:
                rule_summary[count_name] += res["rule_matches"][count_name]

        summary = {}
        for event_type, counts in summary_counts.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall
                else None
            )
            summary[event_type] = {
                **counts,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        rule_summary["coverage"] = (
            rule_summary["covered_rule_count"] / rule_summary["target_rule_count"]
            if rule_summary["target_rule_count"]
            else None
        )

        all_video_scores.append({
            "video_id": vid,
            "frames": frame_results,
            "summary": summary,
            "rule_matches": rule_summary,
        })
        # draw performance over time for the current video

        # visual added tp, fp, fn counts
        visual_line_plot(
            [
                [
                    frame["added"]["recall"]
                    if frame["added"]["recall"] is not None
                    else 1
                    for frame in frame_results
                ]
            ],
            ["Added Recall"],
            output_dir=output_dir,
            title=f"Video {vid} Added Event Counts",
            filename=f"{vid}_prediction_added_counts"
        )

        visual_line_plot(
            [
                [
                    frame[event_type]["f1"]
                    if frame[event_type]["f1"] is not None
                    else float("nan")
                    for frame in frame_results
                ]
                for event_type in ("added", "removed", "unchanged")
            ],
            ["Added F1", "Removed F1", "Unchanged F1"],
            output_dir=output_dir,
            title=f"Video {vid} Event F1 Scores",
            filename=f"{vid}_prediction_scores"
        )
        visual_line_plot(
            [[
                frame["rule_matches"]["coverage"]
                if frame["rule_matches"]["coverage"] is not None
                else float("nan")
                for frame in frame_results
            ]],
            ["Target Rule Coverage"],
            output_dir=output_dir,
            title=f"Video {vid} Target Rule Coverage",
            filename=f"{vid}_target_rule_coverage"
        )

        print(
            f"Scores for video {vid}: Events={summary}, "
            f"Target rule coverage={rule_summary}"
        )
        

    print(f"Prediction data for video {vid}: {data}")
    return all_video_scores