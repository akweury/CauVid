import os
import torch 

from src.exp_roadpp.utils_data import atom_to_signature
from src.exp_roadpp.step03_visual import visual_2d_heatmap
from src.exp_roadpp import utils_data


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



def head_signature_to_label_index(head_signature, dataset_labels):
    """Map a clause head signature (see `atom_to_signature`/body form, i.e.
    the atom tuple without its trailing `tube_uid` pair) to the flat label
    index used by `ocm`'s last dimension, matching the scheme in
    `build_ocm_gt`: av actions first, then non-av actions, then locations.
    """
    predicate = head_signature[1]
    if predicate == "action":
        action_id = head_signature[3]
        agent_class = head_signature[7]
        if agent_class == "av":
            return action_id
        return len(dataset_labels["av_action_labels"]) + action_id
    elif predicate == "location":
        location_name = head_signature[3]
        return (
            len(dataset_labels["av_action_labels"])
            + len(dataset_labels["action_labels"])
            + location_name
        )
    else:
        raise ValueError(f"Unknown predicate type in head signature: {head_signature}")


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
        obj_matrix = torch.zeros((len(body_atoms), 2), dtype=torch.long)
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

            # bk["clause_scopes"][c_index, body_id] tells us which object_slot
            # each body is bound to for this clause (0 = anchor object; equal
            # nonzero slots = the same other object; different slots = distinct
            # objects). Bodies that share a slot must all be satisfied by the
            # *same* object, so the candidates for a slot are the intersection
            # of the per-body candidate sets, not their union.
            bodies_by_slot = {}
            objs_by_body = {}
            for body_id in c_bodies.tolist():
                slot = int(bk["clause_scopes"][c_index, body_id].item())
                objs_for_body = set(obj_proposals[obj_proposals[:, 1] == body_id, 0].tolist())
                objs_by_body[body_id] = objs_for_body
                bodies_by_slot.setdefault(slot, []).append(body_id)

            slot_candidates = {
                slot: set.intersection(*(objs_by_body[b] for b in body_ids))
                for slot, body_ids in bodies_by_slot.items()
            }

            # body_to_objs[body_id] = sorted list of objects that could fill
            # this body's slot (shared by every body bound to the same slot),
            # i.e. a body -> possible-object(s) mapping consistent with
            # clause_scopes.
            body_to_objs = {
                body_id: sorted(slot_candidates[slot])
                for slot, body_ids in bodies_by_slot.items()
                for body_id in body_ids
            }
            # Candidate anchor objects (slot 0) for this clause.
            c_objs = slot_candidates.get(0, set())

            # The head is always evaluated against the clause's own (anchor)
            # entity (see `build_features`'s `next_entity_facts` lookup, keyed
            # by the body atom's own entity_key) -- i.e. the head is implicitly
            # bound to slot 0, the same object as the anchor body. So the
            # objects that should receive the head's label are exactly the
            # anchor-slot candidates.
            head_objs = c_objs
            if not head_objs:
                continue

            label = head_signature_to_label_index(c_heads, bk["dataset_labels"])
            for obj in head_objs:
                # Estimated label of `obj` in the next frame, based on this
                # active clause's head.
                ocm[obj, frame_i, label] = True

    return ocm

def apply_clause_proposals(proposals, heads):
    # proposals includes active clauses for each frame
    # events are the ranges of frames where each clause is active
    events = []

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




def evaluate_ocm(ocm, ocm_gt, ocm_mask):
    """Compare predicted object-label estimates `ocm` against ground truth
    `ocm_gt`, restricted to the (object, frame) positions where `ocm_mask`
    is True (i.e. where the object actually exists in the ground truth).

    ocm, ocm_gt: (obj_num, frame_num, label_num) bool tensors.
    ocm_mask: (obj_num, frame_num) bool tensor.

    Returns (recall, precision, iou, f1) as plain floats, computed by
    micro-averaging True Positives/False Positives/False Negatives over all
    masked (object, frame, label) entries.
    """
    mask = ocm_mask.unsqueeze(-1).expand_as(ocm_gt)
    pred = ocm & mask
    gt = ocm_gt & mask

    tp = (pred & gt).sum().item()
    fp = (pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return recall, precision, iou, f1


def clauses_evaluation(bk, atoms, cfg):
    summary = {}
    label_num = bk["label_num"]
    labels = bk["dataset_labels"]
    summary_file = cfg["output_dir"] / "clauses_evaluation_summary.json"
    if summary_file.exists():
        summary = utils_data.load_json(summary_file)
        return summary
    
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
        recall, precision, iou, f1 = evaluate_ocm(ocm, ocm_gt, ocm_mask)
        summary[vid] = {
            "recall": recall,
            "precision": precision,
            "iou": iou,
            "f1": f1,
        }
        print(f"Evaluation for video {vid}: recall={recall}, precision={precision}, iou={iou}, f1={f1}")
    # Save the evaluation summary to a JSON file for future reference. The
    # per-video ocm/ocm_gt tensors are intentionally excluded (JSON can't
    # serialize torch.Tensor); only the scalar metrics are persisted.
    utils_data.save_json(summary, summary_file)

    return summary


    