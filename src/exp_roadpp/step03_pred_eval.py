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
        unchanged_as_percents = [] 
        changed_as_percents = [] 
        for frame_id, (siss, soss_gt) in data["two_frames_data"].items():
            if frame_id < start_frame_id:
                continue
            # given the current frame, predict the action/location of the atoms in the next frame
            meta_data = {"video_id": vid, "frame_id": frame_id, "output_dir": output_dir}
            soss = model.predict(siss, meta_data=None)
            # evaluate the prediction against the ground truth labels
            res = model.eval(soss, soss_gt, meta_data=None)
            recalls.append(res["recall"])
            precisions.append(res["precision"])
            f1s.append(res["f1"]) 
            unchanged_as_percents.append(res["frame_atom_unchanged_as_percent"])
            changed_as_percents.append(res["frame_atom_removed_as_percent"])
        
        all_video_scores.append({ "video_id": vid, "recalls": recalls, "precisions": precisions, "f1s": f1s, "unchanged_as_percents": unchanged_as_percents, "changed_as_percents": changed_as_percents })
        # draw performance over time for the current video
        visual_line_plot(
            [recalls, precisions, f1s, unchanged_as_percents, changed_as_percents],
            ["Recall", "Precision", "F1", "Unchanged AS Percent", "Changed AS Percent"],
            output_dir=output_dir,
            title=f"Video {vid} Prediction Scores",
            filename=f"{vid}_prediction_scores"
        )
        print(f"Scores for video {vid}: Recalls={recalls}, Precisions={precisions}, F1s={f1s}")
        

    print(f"Prediction data for video {vid}: {data}")
    return all_video_scores