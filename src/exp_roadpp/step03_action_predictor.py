




class ActionPredictor:
    def __init__(self, coarse_pruned_clauses):
        self.coarse_pruned_clauses = coarse_pruned_clauses
    def atom_in_clause_body(self, atom, clause):
        body = clause["body"]
        pred = False
        for atom_in_body in body:
            same_pred = atom["pred"] == atom_in_body["pred"]
            if "location_name" in atom and "location_name" in atom_in_body:
                same_pred_label = atom["location_name"] == atom_in_body["location_name"]
            elif "action_id" in atom and "action_id" in atom_in_body:
                same_pred_label = atom["action_id"] == atom_in_body["action_id"]
            else:
                raise ValueError("Atom does not have a recognized identifier for comparison")
            same_agent_class = atom["agent_class"] == atom_in_body["agent_class"]

            if same_pred and same_pred_label and same_agent_class:
                pred = True
                break

        return pred


    def predict(self, atoms):
        # given the atoms in the current frame, predict the actions for the next frame
        # for each atom, determine the possible actions based on the coarse pruned clauses
        predicted_actions = {}
        for atom in atoms:
            possible_heads = []
            for clause in self.coarse_pruned_clauses:
                if self.atom_in_clause_body(atom, clause["clause"]):
                    possible_heads.extend(clause["clause"]["head"])
            predicted_actions[atom] = possible_heads
        
        return predicted_actions

    def eval(self, pred, labels):
        # Implement the evaluation logic for the predictions
        pass