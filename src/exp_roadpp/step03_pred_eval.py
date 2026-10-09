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
        recalls = []
        precisions = []
        f1s = []
        unchanged_as_nums = [] 
        pred_unchanged_as_percents = [] 
        removed_as_nums = [] 
        pred_removed_as_percents = [] 
        new_as_nums = []

        for frame_id, (siss, soss_gt) in data["two_frames_data"].items():
            if frame_id < start_frame_id:
                continue
            # given the current frame, predict the action/location of the atoms in the next frame
            meta_data = {"video_id": vid, "frame_id": frame_id, "output_dir": output_dir}
            soss = model.predict(siss, meta_data=None)
            # evaluate the prediction against the ground truth labels
            res = model.eval(siss, soss, soss_gt, meta_data=None)
            recalls.append(res["recall"])
            removed_as_nums.append(res["removed_atom_num"])
            pred_removed_as_percents.append(res["pred_removed_atom_num"])
            pred_unchanged_as_percents.append(res["pred_unchanged_atom_num"])
            unchanged_as_nums.append(res["unchanged_atom_num"])
            new_as_nums.append(res["new_atom_num"])
        
        all_video_scores.append({ "video_id": vid, "recalls": recalls})
        # draw performance over time for the current video
        visual_line_plot(
            [new_as_nums],
            ["New AS Num"],
            output_dir=output_dir,
            title=f"Video {vid} New Atoms in Next Frame",
            filename=f"{vid}_new_atoms_next_frame"
        )
        visual_line_plot(
            [recalls],
            ["Recall"],
            output_dir=output_dir,
            title=f"Video {vid} Prediction Scores",
            filename=f"{vid}_prediction_scores"
        )
        visual_line_plot(
            [unchanged_as_nums, pred_unchanged_as_percents],
            ["Unchanged AS Num", "Predicted Unchanged AS Num"],
            output_dir=output_dir,
            title=f"Video {vid} Atoms in Current Frame",
            filename=f"{vid}_atom_change_percentages_current_frame"
        )
        visual_line_plot(
            [removed_as_nums, pred_removed_as_percents],
            ["Removed AS Num", "Predicted Removed AS Num"],
            output_dir=output_dir,
            title=f"Video {vid} Atoms in Current Frame",
            filename=f"{vid}_removed_atom_change_percentages_current_frame"
        )
        print(f"Scores for video {vid}: Recalls={recalls}, Precisions={precisions}, F1s={f1s}")
        

    print(f"Prediction data for video {vid}: {data}")
    return all_video_scores