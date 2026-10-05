
import torch 


from src.exp_roadpp.utils_data import atom_to_signature, add_labels_to_atom, same_signature_no_tube_uid, build_signature


class ActionPredictor:
    def __init__(self, coarse_pruned_clauses,dataset_labels):
        self.device = "cpu"
        self.coarse_pruned_clauses = coarse_pruned_clauses
        self.head_signatures = [(atom_to_signature(clause[0]["clause"]["head"]), clause[1]) for clause in coarse_pruned_clauses]
        self.body_signatures = [(atom_to_signature(clause[0]["clause"]["body"][0]), clause[1]) for clause in coarse_pruned_clauses]
        self.dataset_labels = dataset_labels
        self.agent_action_num = len(self.dataset_labels["action_labels"])
        self.av_action_num = len(self.dataset_labels["av_action_labels"])
        self.location_name_num = len(self.dataset_labels["loc_labels"])
        self.action_num = self.agent_action_num + self.av_action_num + self.location_name_num
        

    def _atom_to_signature(self, atom):
        if "action_id" in atom:
            prediction = ("action_id_label", atom["action_id_label"],
                          "agent_class_label", atom["agent_class_label"]
                          )
        elif "location_name" in atom:
            prediction = ("location_name_label", atom["location_name_label"],
                          "agent_class_label", atom["agent_class_label"]
                          )
        else:
            raise ValueError("Unknown head type in clause")
        return prediction

    
    def find_matching_bodies(self, atom, clause_body_signatures):
        matching_indices = []
        matching_scores = []
        for i, body_signature in enumerate(clause_body_signatures):
            if atom == body_signature[0]:
                matching_indices.append(i)
                matching_scores.append(body_signature[1])
        return matching_indices

    def find_matching_head_signatures(self, matching_body_signature_indices, clause_head_signatures, th = 0.9):
        all_matches = [clause_head_signatures[i] for i in matching_body_signature_indices]
        filtered_matches = [match for match in all_matches if match[1] >= th]
        return filtered_matches

    
    def predict_atom(self, as_in, as_forzen):
        """
        as_in: list of atom signatures in the current frame
        as_forzen: list of atom signatures from the rule head to match against

        # for each atom in as_in, we need to give it a
        """
        pred_actions = torch.zeros(self.action_num).to(self.device)
        
        for atom_signature in as_in:
            for frozen_signature, score in as_forzen:
                if not same_signature_no_tube_uid(atom_signature, frozen_signature):
                    continue
                if "location_name_label" in atom_signature:
                    index = self.agent_action_num + self.av_action_num + self.dataset_labels["loc_labels"].index(atom_signature[5])
                elif "action_id_label" in atom_signature and "av" in atom_signature:
                    index = self.agent_action_num + self.dataset_labels["av_action_labels"].index(atom_signature[5])
                elif "action_id_label" in atom_signature and "av" not in atom_signature:
                    index = self.dataset_labels["action_labels"].index(atom_signature[5])
                else:
                    raise ValueError("Unknown atom signature type")
                pred_actions[index] = score
        
        return pred_actions
    def connect_pred_to_tube_uid(self, as_in, atom_prediction_matrix):
        # Implement the logic to connect predictions to tube UIDs
        # This is a placeholder implementation
        # tube_uid, pred, location_name, agent_class
        preds = []
        for atom_i in range(atom_prediction_matrix.shape[0]):
            if atom_prediction_matrix[atom_i].sum()>0:
                atom_signature = as_in[atom_i]
                col_index = (atom_prediction_matrix[atom_i].argmax().item())
                predicate, label, label_id = self.index_to_label(col_index)
                raise ValueError("Unknown predicate type in connect_pred_to_tube_uid")
                preds.append(build_signature(predicate, label, label_id, atom_signature[7], atom_signature[9], atom_signature[-1]))
        return preds
    def predict(self, atoms):
        # given the atoms in the current frame, predict the actions for the next frame
        # for each atom, determine the possible actions based on the coarse pruned clauses
        
        as_in = [atom_to_signature(add_labels_to_atom(atom, self.dataset_labels)) for atom in atoms]
        as_frozen_bodies = self.body_signatures
        as_frozen_heads = self.head_signatures
        # each atom need at least one predictions as the action/location in the next frame
        # we need an funciton, given an atom, return one predicted action/location for the next frame
        # we can construct a dictionary mapping each atom to its predicted action/location for the next frame
        # find the closest matching head of each atom

        # N actions x M atoms
        atom_prediction_matrix = torch.zeros(len(as_in), self.action_num)
        for atom_i,atom_signature in enumerate(as_in):
            as_frozen_bodies_matched_indices = self.find_matching_bodies(atom_signature, as_frozen_bodies)
            as_frozen_heads_matched = self.find_matching_head_signatures(as_frozen_bodies_matched_indices, as_frozen_heads)
            atom_prediction_matrix[atom_i] = self.predict_atom(as_in, as_frozen_heads_matched)    
        pred = self.connect_pred_to_tube_uid(as_in, atom_prediction_matrix)
        return pred





    def eval(self, pred, labels):
        # Implement the evaluation logic for the predictions
        pass