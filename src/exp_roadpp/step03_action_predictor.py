




class ActionPredictor:
    def __init__(self, coarse_pruned_clauses):
        self.coarse_pruned_clauses = coarse_pruned_clauses

    def predict(self, atoms):
        # given the atoms in the current frame, predict the actions for the next frame
        pass

    def eval(self, pred, labels):
        # Implement the evaluation logic for the predictions
        pass