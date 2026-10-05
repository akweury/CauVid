from src.exp_roadpp.utils_data import atom_to_signature, add_labels_to_atom

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
                frame_data[frame_id].append(atom)

    data["frame_data"] = frame_data
    data["video_length"] = end_frame
    # connect the neighboring frames with the atoms that occur in them
    two_frames_data = {frame_id: {"current_frame": frame_data[frame_id], "next_frame": frame_data[frame_id + 1]} for frame_id in frame_data.keys() if frame_id + 1 in frame_data}
    
    for frame_id, two_frames in two_frames_data.items():
        frame_signatures = []
        for atom in two_frames["next_frame"]:
            atom_signature = atom_to_signature(add_labels_to_atom(atom, dataset_labels))
            frame_signatures.append(atom_signature)
        two_frames["labels"] = frame_signatures
    data["two_frames_data"] = two_frames_data
    return data


def predict_next_n_steps(model, test_atoms, output_dir, dataset_labels, step=1):
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
            # Here you can add any preprocessing or additional steps before prediction
            pred = model.predict(two_frames["current_frame"])
            labels = two_frames["labels"]
            frame_scores = model.eval(pred, labels)
            video_scores.append(frame_scores)
        all_video_scores.append({ "video_id": vid, "video_scores": video_scores })

    print(f"Prediction data for video {vid}: {data}")
    return all_video_scores