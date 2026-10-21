import os
import torch 

from src.exp_roadpp.utils_data import atom_to_signature
from src.exp_roadpp.step03_visual import visual_line_plot, visual_2d_heatmap

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


def predict_next_n_steps(test_atoms, output_dir, dataset_labels, step=1):
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
            "fired_rule_count": 0,
            "correct_fired_rule_count": 0,
            "incorrect_fired_rule_count": 0,
            "transition_count": 0,
            "predicted_transition_count": 0,
            "unpredicted_transition_count": 0,
        }
        for frame_id, (siss, soss_gt) in data["two_frames_data"].items():
            if frame_id < start_frame_id:
                continue
            # given the current frame, predict the action/location of the atoms in the next frame
            meta_data = {"video_id": vid, "frame_id": frame_id, "output_dir": output_dir}
            soss = model.predict(siss, meta_data=None)
            # evaluate the prediction against the ground truth labels
            res = model.eval(siss, soss, soss_gt, meta_data=None)
            print(f"Evaluation results for frame {frame_id}: {res}")
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
        rule_summary["rule_accuracy"] = (
            rule_summary["correct_fired_rule_count"] / rule_summary["fired_rule_count"]
            if rule_summary["fired_rule_count"]
            else None
        )
        rule_summary["transition_recall"] = (
            rule_summary["predicted_transition_count"] / rule_summary["transition_count"]
            if rule_summary["transition_count"]
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
                frame["rule_matches"]["transition_recall"]
                if frame["rule_matches"]["transition_recall"] is not None
                else float("nan")
                for frame in frame_results
            ]],
            ["Transition Recall"],
            output_dir=output_dir,
            title=f"Video {vid} Transition Recall",
            filename=f"{vid}_transition_recall"
        )

        print(
            f"Scores for video {vid}: Events={summary}, "
            f"Fired-rule and transition scores={rule_summary}"
        )
        

    print(f"Prediction data for video {vid}: {data}")
    return all_video_scores


def build_frame_proposals(frame_atoms, clause_matrix, unique_bodies, obj_num, label_num):
    proposals = torch.zeros((obj_num, label_num), dtype=torch.bool)
    body_matrix = torch.zeros(len(unique_bodies), dtype=torch.bool)
    for atom in frame_atoms:
        if atom[:-2] in unique_bodies:
            body_matrix[unique_bodies.index(atom[:-2])] = True 
    proposals = (~clause_matrix | body_matrix).all(dim=1).nonzero(as_tuple=False).squeeze(-1)

    return proposals
            

def build_video_matrix(apf, vm, ocm_mask, tube_ids, bk):
    # vm: video matrix of shape (num_objects, num_frames, num_labels)
    # clause_matrix: matrix of shape (num_clauses, num_features)
    # ocm_present: matrix of shape (num_objects, num_frames)
    # calculate vm
    for frame_i in range(vm.shape[1]):
        if apf[frame_i] is None or len(apf[frame_i]) == 0:
            continue
        # objs in this frame
        # estimate active clauses for this frame based on object presence
        frame_clause_proposals = build_frame_proposals(apf[frame_i],bk["clause_matrix"], 
                                                       bk["unique_bodies"], 
                                                       vm.shape[0], vm.shape[2])
        
        obj_matrix = torch.zeros((len(apf[frame_i]), 2))
        for i, atom in enumerate(apf[frame_i]):
            obj_matrix[i, 0] = tube_ids.index(atom[-1])
            obj_matrix[i, 1] = bk["unique_bodies"].index(atom[:-2])

        if frame_clause_proposals.sum() == 0:
            print("No clause proposals were made.")

        # now we need to determine the labels of each objects, either iterate over the atoms or the clause proposals   
        for c_index in frame_clause_proposals:
            c_bodies = bk["clause_matrix"][c_index].nonzero(as_tuple=False).squeeze(-1)
            c_heads = bk["clause_heads"][c_index]
            # for each body, determine which objects satisfy it
            obj_proposals = obj_matrix[[i for i, body_id in enumerate(obj_matrix[:, 1]) if body_id in c_bodies],:]
            # filter out the not possible objects
            
            # get the objects labels 
            labels = None 
            for obj in c_objs:
                vm[obj, frame_i, labels] = True


        vm[:, frame_i] = frame_clause_proposals
    
    return vm 

def apply_clause_proposals(proposals, heads):
    # proposals includes active clauses for each frame
    # events are the ranges of frames where each clause is active
    events = []

    return events


def detect_events(bk,atoms, cfg):
    output_dir = cfg["output_dir"]
    clause_matrix = bk["clause_matrix"]
    clause_heads = bk["clause_heads"]
    feature_num = bk["feature_num"]
    dataset_labels = bk["dataset_labels"]
    # 
    
    ocm = apply_clause_proposals(vm, clause_heads)

    visual_2d_heatmap(
        vm.cpu().numpy(),
        output_dir=output_dir,
        title="Clause Proposals",
        filename="clause_proposals",
        x_label = "Clauses",
        y_label = "Frames"
        
    )
    return events

def build_ocm_gt(all_tube_uids, apsf, frame_num,obj_num, feature_num, dataset_labels):
    ocm_gt = torch.zeros((obj_num, frame_num, feature_num), dtype=torch.bool)
    ocm_presents = torch.zeros((obj_num, frame_num), dtype=torch.bool)
    
    for frame_id, atoms_in_frame in apsf.items():
        for atom in atoms_in_frame:
            tube_uid = atom["tube_uid"]
            obj_id = all_tube_uids.index(tube_uid)
            start_frame = int(atom["start_frame"])
            end_frame = int(atom["end_frame"])
            if atom["agent_class"]=="av":
                feature_index = 0 + atom["action_id"]
            elif atom["pred"]=="action":
                feature_index = len(dataset_labels["av_action_labels"]) + atom["action_id"]
            elif atom["pred"]=="location":
                feature_index = len(dataset_labels["av_action_labels"]) + len(dataset_labels["action_labels"]) + atom["location_name"]
            ocm_gt[obj_id, start_frame:end_frame+1, feature_index] = True
            ocm_presents[obj_id, start_frame:end_frame+1] = True

    return ocm_gt, ocm_presents

def extract_atoms_per_frame(interval_atoms, dataset_labels, frame_num):
    atoms_per_start_frame = {}
    
    for atom in interval_atoms:
        if int(atom["start_frame"]) not in atoms_per_start_frame:
            atoms_per_start_frame[int(atom["start_frame"])] = []
        atoms_per_start_frame[int(atom["start_frame"])].append(atom)
    # for each frame, we need a list to include the atoms that occur in that frame
    atoms_per_frame = {i: [] for i in range(frame_num)}
    for start_frame, atoms in atoms_per_start_frame.items():
        for atom in atoms:
            for frame_id in range(start_frame, int(atom["end_frame"])):
                frame_as = atom_to_signature(atom, dataset_labels)
                atoms_per_frame[frame_id].append(frame_as)
    return atoms_per_frame, atoms_per_start_frame




def evaluate_video_events(events, events_gt):
    # Implement the logic to evaluate the detected events for a video
    # For example, you can calculate precision, recall, F1 score, etc.
    
    evaluation = {}
    return evaluation




def clauses_evaluation(bk, atoms, cfg):
    summary = {}
    label_num = bk["label_num"]
    labels = bk["dataset_labels"]
    for vid, interval_atoms in atoms.items():
        tube_uids = list(set([atom["tube_uid"] for atom in interval_atoms]))
        frame_num = max([int(atom["end_frame"]) for atom in interval_atoms]) if interval_atoms else 0
        obj_num = len(tube_uids)
        vm = torch.zeros((obj_num, frame_num, label_num), dtype=torch.bool)
        
        apf, apsf = extract_atoms_per_frame(interval_atoms,labels, frame_num)
        ocm_gt, ocm_mask = build_ocm_gt(tube_uids, 
                                            apsf, 
                                            frame_num, 
                                            obj_num, 
                                            label_num, 
                                            labels)
        vm = build_video_matrix(apf, vm, ocm_mask, tube_uids, bk)
        ocm = detect_events(bk, apf, cfg) 
        summary[vid] = {
            "ocm": ocm,
            "ocm_gt": ocm_gt,
            "vm": vm,
            "recall": None,
            "precision": None,
            "iou":None,
            "f1": None,
        }
    return summary


    