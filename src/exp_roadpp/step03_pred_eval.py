from src.exp_roadpp.utils_data import atom_to_signature
from src.exp_roadpp.step03_visual import visual_2d_heatmap

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
    frame_data = {frame_id: [] for frame_id in range(end_frame)}
    for start_frame, atoms in atoms_per_start_frame.items():
        for atom in atoms:
            for frame_id in range(start_frame, int(atom["end_frame"])):
                frame_as = atom_to_signature(atom, dataset_labels)
                frame_data[frame_id].append(frame_as)

    data["frame_data"] = frame_data
    data["video_length"] = end_frame
    # connect the neighboring frames with the atoms that occur in them
    two_frames_data = {frame_id: {"current_frame": frame_data[frame_id], 
                                  "next_frame": frame_data[frame_id + 1]} for frame_id in frame_data.keys() if frame_id + 1 in frame_data}    
    data["two_frames_data"] = two_frames_data
    return data


def predict_next_n_steps(model, test_atoms, output_dir, dataset_labels, step=1):
    # start predicting the next steps for each video in the test set
    print(f"Predicting next {step} steps for {len(test_atoms)} videos")
    start_frame_id = step
    all_video_scores = []
    for vid, video_atoms in test_atoms.items():
        print(f"Predicting for video: {vid}")
        # predict the actions frame by frame
        data = get_video_data(vid, video_atoms, dataset_labels)
        video_scores = []
        for frame_id, two_frames in data["two_frames_data"].items():
            if frame_id < start_frame_id:
                continue
            # given the current frame, predict the action/location of the atoms in the next frame
            soss, fim = model.predict(two_frames["current_frame"])
            visual_2d_heatmap(fim.mean(dim=0), output_dir=output_dir, filename=f"{vid}_frame_{frame_id}_fim_sum_{fim.sum().item()}")

            # evaluate the prediction against the ground truth labels
            as_out_gt = two_frames["next_frame"]
            frame_scores = model.eval(soss, as_out_gt)
            video_scores.append(frame_scores)
        all_video_scores.append({ "video_id": vid, "video_scores": video_scores })

    print(f"Prediction data for video {vid}: {data}")
    return all_video_scores