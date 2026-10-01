
import torch 


from src.exp_roadpp import utils_data


class ActionPredictor:
    def __init__(self, coarse_pruned_clauses,dataset_labels):
        self.coarse_pruned_clauses = coarse_pruned_clauses
        self.head_signatures = [(self._atom_to_signature(clause[0]["clause"]["head"]), clause[1]) for clause in coarse_pruned_clauses]
        self.body_signatures = [(self._atom_to_signature(clause[0]["clause"]["body"][0]), clause[1]) for clause in coarse_pruned_clauses]
        self.dataset_labels = dataset_labels
        

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
    def update_atom_continue_table(self, atom_signatures, matching_head_signatures, atom_continue_table):
        for i, atom_signature in enumerate(atom_signatures):
            if atom_signature in matching_head_signatures:
                atom_continue_table[i] = True
        return atom_continue_table
    def find_matching_head_signatures(self, matching_body_signature_indices, clause_head_signatures, th = 0.9):
        all_matches = [clause_head_signatures[i] for i in matching_body_signature_indices]
        filtered_matches = [match[0] for match in all_matches if match[1] >= th]
        return filtered_matches


    def predict(self, atoms):
        # given the atoms in the current frame, predict the actions for the next frame
        # for each atom, determine the possible actions based on the coarse pruned clauses
        predictions = {}
        atom_signatures = [self._atom_to_signature(utils_data.add_labels_to_atom(atom, self.dataset_labels)) for atom in atoms]
        clause_body_signatures = self.body_signatures
        clause_head_signatures = self.head_signatures
        # each atom need at least one predictions as the action/location in the next frame
        # we need an funciton, given an atom, return one predicted action/location for the next frame
        # we can construct a dictionary mapping each atom to its predicted action/location for the next frame
        # find the closest matching head of each atom
        action_num = len(self.dataset_labels["action_labels"])
        av_action_num = len(self.dataset_labels["av_action_labels"])
        # N actions x M atoms
        atom_prediction_matrix = torch.zeros(len(atom_signatures), action_num + av_action_num)
        for atom_i,atom_signature in enumerate(atom_signatures):
            matching_body_signature_indices = self.find_matching_bodies(atom_signature, clause_body_signatures)
            matching_head_signatures = self.find_matching_head_signatures(matching_body_signature_indices, clause_head_signatures)
            matching_atom_indices = self.find_matching_atoms(atom_signatures, matching_head_signatures)
            
        return atom_prediction_matrix





    def eval(self, pred, labels):
        # Implement the evaluation logic for the predictions
        pass