import os
import torch 

from src.exp_roadpp.utils_data import atom_to_signature
from src.exp_roadpp.step03_visual import visual_2d_heatmap

def get_video_data(video_atoms, dataset_labels):
    data = []
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

    data = [(frame_data[i], frame_data[i + 1]) for i in frame_data.keys() if i + 1 in frame_data]
    return data



def build_frame_proposals(frame_atoms, clause_matrix, unique_bodies, obj_num, label_num):
    proposals = torch.zeros((obj_num, label_num), dtype=torch.bool)
    body_matrix = torch.zeros(len(unique_bodies), dtype=torch.bool)
    for atom in frame_atoms:
        if atom[:-2] in unique_bodies:
            body_matrix[unique_bodies.index(atom[:-2])] = True 
    proposals = (~clause_matrix | body_matrix).all(dim=1).nonzero(as_tuple=False).squeeze(-1)

    return proposals
            

def build_ocm(apf, ocm_mask, tube_ids, bk):
    # vm: video matrix of shape (num_objects, num_frames, num_labels)
    # clause_matrix: matrix of shape (num_clauses, num_features)
    # ocm_present: matrix of shape (num_objects, num_frames)
    # calculate vm
    ocm = torch.zeros((ocm_mask.shape[0], ocm_mask.shape[1], bk["label_num"]), dtype=torch.bool)
    for frame_i in range(ocm.shape[1]):
        if apf[frame_i] is None or len(apf[frame_i]) == 0:
            continue
        # objs in this frame
        # estimate active clauses for this frame based on object presence
        frame_clause_proposals = build_frame_proposals(apf[frame_i],bk["clause_matrix"], 
                                                       bk["unique_bodies"], 
                                                       ocm.shape[0], ocm.shape[2])
        
        # Only atoms that are the body of some clause can trigger a clause;
        # the rest of the frame's atoms are not in unique_bodies.
        body_atoms = [atom for atom in apf[frame_i] if atom[:-2] in bk["unique_bodies"]]
        obj_matrix = torch.zeros((len(body_atoms), 2))
        for i, atom in enumerate(body_atoms):
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
                ocm[obj, frame_i, labels] = True


        ocm[:, frame_i] = frame_clause_proposals
    
    return ocm 

def apply_clause_proposals(proposals, heads):
    # proposals includes active clauses for each frame
    # events are the ranges of frames where each clause is active
    events = []

    return events


def detect_events(bk, atoms, cfg):
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

def build_ocm_gt(all_tube_uids, apsf, frame_num, label_num, dataset_labels):
    obj_num = len(all_tube_uids)
    ocm_gt = torch.zeros((obj_num, frame_num, label_num), dtype=torch.bool)
    ocm_mask = torch.zeros((obj_num, frame_num), dtype=torch.bool)
    
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
            ocm_mask[obj_id, start_frame:end_frame+1] = True

    return ocm_gt, ocm_mask

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
        apf, apsf = extract_atoms_per_frame(interval_atoms,labels, frame_num)
        ocm_gt, ocm_mask = build_ocm_gt(tube_uids, 
                                        apsf, 
                                        frame_num, 
                                        label_num, 
                                        labels)
        ocm = build_ocm(apf, ocm_mask, tube_uids, bk)
        summary[vid] = {
            "ocm_gt": ocm_gt,
            "ocm": ocm,
            "recall": None,
            "precision": None,
            "iou":None,
            "f1": None,
        }
    return summary


    