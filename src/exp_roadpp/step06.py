import os
from pathlib import Path
import torch

from src.exp_roadpp import utils_data



def build_clause_matrix(clauses):
    # get all unique bodies
    unique_bodies = []
    for clause in clauses:
        bodies = clause["bodies"]
        for body in bodies:
            if body not in unique_bodies:
                unique_bodies.append(body)

    clause_matrix = torch.zeros((len(clauses), len(unique_bodies)), dtype=torch.bool)
    # clause_scopes[i, j] = the object_slot that body j binds to in clause i's
    # body_scopes (0 = anchor object; equal nonzero values = co-referenced to
    # the same other object; -1 = body j is not part of clause i).
    clause_scopes = torch.full((len(clauses), len(unique_bodies)), -1, dtype=torch.long)
    clause_heads = []
    for i, clause in enumerate(clauses):
        bodies = clause["bodies"]
        body_scopes = clause.get("body_scopes")
        clause_heads.append(clause["head"])
        for j, body in enumerate(unique_bodies):
            if body in bodies:
                clause_matrix[i, j] = True
                if body_scopes is not None:
                    clause_scopes[i, j] = body_scopes[bodies.index(body)]
    print(f"[Clause Matrix]: {clause_matrix.shape}")

    return clause_matrix, clause_heads, unique_bodies, clause_scopes


def build_background_knowledge(clauses, dataset_labels):
    cm, heads, bodies, scopes = build_clause_matrix(clauses)
    label_num = len(dataset_labels["av_action_labels"]) + len(dataset_labels["action_labels"]) + len(dataset_labels["loc_labels"])
    bk = {
        "clause_matrix": cm,
        "clause_heads": heads,
        "unique_bodies": bodies,
        "clause_scopes": scopes,
        "label_num": label_num,
        "dataset_labels": dataset_labels
    }
    return bk




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



def main(input_data):
    print("\n--------- Step 06 Evaluation ----------------------\n")    
    cfg = {
        "output_dir": Path(input_data["output_dir"]),
        "device": input_data.get("device", "cpu")
    }
    atoms = utils_data.load_json(input_data["val_atoms_file"])
    dataset_labels = utils_data.load_json(input_data["dataset_labels_file"])
    extended_clauses = utils_data.load_json(input_data["extended_clauses_file"])

    bk = build_background_knowledge(extended_clauses, dataset_labels)
    # clause evaluation
    summary = clauses_evaluation(bk, atoms, cfg)    

    print("\n--------- Step 06 Done ---------------\n")
    