import torch
from scipy.optimize import linear_sum_assignment

from src.exp_roadpp import utils_data


class CauVid:
    def __init__(self, cfg):
        self.clauses = utils_data.load_json(cfg["extended_clauses_file"])
        self.unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
        self.delay_windows = cfg["delay_windows"]
        self.clause_matrices = self._build_clause_matrix()



    def _reorder_clauses(self, clauses):
        reordered_clauses = []
        for c in clauses:
            reordered_body = []
            for b in c["bodies"]:
                assert len(b) == 3, "Each body should have exactly 3 elements"
                if b[2] == "now":
                    window = 0
                elif "held" in b[2]:
                    window = int(b[2].split("_")[1])
                elif "ended" in b[2]:
                    window = int(b[2].split("_")[2])
                elif "started" in b[2]:
                    window = int(b[2].split("_")[2])
                else:
                    raise ValueError(f"Unknown window type: {b[2]}")
                reordered_body.append([b[1],window,b[0]])
            reordered_c = {
                "head": c["head"],
                "body_scopes": c["body_scopes"],
                "bodies": reordered_body
            }
            reordered_clauses.append(reordered_c)

        return reordered_clauses
    def _build_clause_matrix(self):
        self.clauses = self._reorder_clauses(self.clauses)
        clause_matrices = []
        for c in self.clauses:
            scope_size = max(c["body_scopes"])+1
            window_size = max([b[1] for b in c["bodies"]])+1
            matrix = torch.zeros((scope_size, window_size, len(self.unique_atoms)))
            for b_i, body in enumerate(c["bodies"]):
                matrix[body[0], body[1], body[2]] = True
            clause_matrices.append(matrix)
        return clause_matrices
    
    def predict(self, task_input):
        # -------- substep functions ----------------------------------
        def _compress_task_matrix(task_matrix):
            return task_matrix.sum(dim=1)
        
        def _pad_rows(m, obj):
            if m.shape[0] < obj:
                padding = torch.zeros((obj - m.shape[0], m.shape[1]), dtype=m.dtype)
                m = torch.cat([m, padding], dim=0)
            return m

        def _fit_clause_to_task(c_compressed, task_compressed):
            # Best object grounding = max-weight assignment between task objects and clause scopes
            # (equivalent to trying all permutations, but O(obj^3) instead of O(obj!))
            obj = max(c_compressed.shape[0], task_compressed.shape[0])
            c = _pad_rows(c_compressed, obj)
            t = _pad_rows(task_compressed, obj).to(c.dtype)
            # match[i, j] = number of equal entries between task row i and clause row j
            match = (t[:, None, :] == c[None, :, :]).sum(dim=-1)
            rows, cols = linear_sum_assignment(match.numpy(), maximize=True)
            score = match[rows, cols].sum().item()
            return score / c.numel()
        
        def _build_task_matrix(task_input):
            task_window = len(task_input)
            obj_ids = list(set(atom[-1] for frame in task_input for atom in frame))
            print(f"Object Number: {len(obj_ids)}")
            task_obj_num = len(obj_ids)
            atom_num = len(self.unique_atoms)
            task_matrix = torch.zeros((task_obj_num, task_window,atom_num), dtype=torch.bool) 
            for f_i, frame in enumerate(task_input):
                for atom in frame:
                    atom_idx = atom[0]
                    obj_idx = obj_ids.index(atom[-1])
                    task_matrix[obj_idx, f_i, atom_idx] = True
            return task_matrix
        
        def _compute_clause_score(clause_matrices, task_matrix):
            clause_matches = {}
            task_compressed = _compress_task_matrix(task_matrix)
            for c_i, c_matrix in enumerate(clause_matrices):
                best_score = _fit_clause_to_task(c_matrix.sum(dim=1), task_compressed)
                clause_matches[c_i] = best_score
            return clause_matches

        def _select_matching_clauses(clause_matches):
            if len(clause_matches) == 0:
                return None
            max_score = max(clause_matches.values())
            selected_clause_i = [c_i for c_i, score in clause_matches.items() if score == max_score][0]
            selected_clause = self.clauses[selected_clause_i]
            return selected_clause
        
        # -------- main prediction steps ----------------------------------
        task_matrix = _build_task_matrix(task_input)
        clause_matches = _compute_clause_score(self.clause_matrices, task_matrix)
        selected_clause = _select_matching_clauses(clause_matches)
        return selected_clause

        
    