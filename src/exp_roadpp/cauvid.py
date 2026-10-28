import torch 

from src.exp_roadpp import utils_data


class CauVid:
    def __init__(self, cfg):
        self.clauses = utils_data.load_json(cfg["extended_clauses_file"])
        self.unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
        self.delay_windows = self.clauses[0]["delay_windows"]
    
    
    
    
    
    
    
    def predict(self, task_input):
        # based on a given set of frames, 
        # predict if any event will happen in the upcoming frames

        # -------- substep functions ----------------------------------
        def _find_matching_clause():
            clause_matches ={}
            for c_i, clause in enumerate(self.clauses):
                body_match = True 
                scores = []
                for body in clause["bodies"]:
                    if body not in task_input:
                        body_match = False
                        scores.append(0)
                    else:
                        scores.append(1)
                if body_match:
                    clause_matches[c_i] = torch.mean(torch.tensor(scores))
            return clause_matches

        def _select_matching_clauses(clause_matches):
            if len(clause_matches) == 0:
                return None
            max_score = max(clause_matches.values())
            selected_clause_i = [c_i for c_i, score in clause_matches.items() if score == max_score][0]

            selected_clause = self.clauses[selected_clause_i]
            return selected_clause
        
        # -------- main prediction steps ----------------------------------
        clause_matches = _find_matching_clause()
        selected_clause = _select_matching_clauses(clause_matches)
        return selected_clause

        
    