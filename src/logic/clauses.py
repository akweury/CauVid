
from typing import Callable, Dict, List, Tuple
from collections import defaultdict


class Clause:
    def __init__(self, head: str, body: str):
        self.head = head
        self.body = body
        self.stats = {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0
        }
        
    def update(self, body_true: bool, head_true: bool):
        if body_true and head_true:
            self.stats["tp"] += 1
        elif body_true and not head_true:
            self.stats["fp"] += 1
        elif not body_true and head_true:
            self.stats["fn"] += 1
        else:
            self.stats["tn"] += 1

    def confidence(self) -> float:
        tp, fp = self.stats["tp"], self.stats["fp"]
        return tp/(tp+fp+1e-6)

    def coverage(self) -> float:
        tp, fp, fn, tn = self.stats.values()
        total = tp + fp + fn + tn
        return (tp + fp) / (total + 1e-6)

    def __repr__(self):
        return (f"{self.head}(O):-{self.body}(O) "
                f"[conf={self.confidence():.3f}, cov={self.coverage():.3f}]"
                )


class ClauseLearner:
    def __init__(
        self,
        body_predicates: Dict[str, Callable],
        head_predicates: Dict[str, Callable],
        conf_threshold: float = 0.95,
        cov_threshold: float = 0.1
    ):
        self.body_predicates = body_predicates
        self.head_predicates = head_predicates
        self.conf_threshold = conf_threshold
        self.cov_threshold = cov_threshold
        self.clauses = self._enumerate_clauses()

    def _enumerate_clauses(self) -> List[Clause]:
        clauses = []
        for head in self.head_predicates:
            for body in self.body_predicates:
                clauses.append(Clause(head, body))
        return clauses

    def observe_transition(self, scene_t, scene_t1):
        
        for obj_id, obj_t in scene_t.items():
            if obj_id not in scene_t1:
                continue
            obj_t1 = scene_t1[obj_id]
            for clause in self.clauses:
                body_fn = self.body_predicates[clause.body]
                head_fn = self.head_predicates[clause.head]
                body_true = body_fn(obj_t, scene_t)
                head_true = head_fn(obj_t, obj_t1)
                clause.update(body_true, head_true)

    def get_accepted_clauses(self) -> List[Clause]:
        return [
            c for c in self.clauses
            if c.confidence() >= self.conf_threshold
            and c.coverage() >= self.cov_threshold
        ]


def learn_rules(scene_dataset):
    from logic.predicates import left, right, moves_down, moves_up    
    frame_t_t1 =[]
    for i in range(len(scene_dataset)-1):
        frame_t_t1.append(scene_dataset[i:i+2])

    body_preds = {
        "left": left,
        "right": right}

    head_preds = {
        "moves_up": moves_up,
        "moves_down": moves_down}

    learner = ClauseLearner(body_preds, head_preds)

    for frame_now, frame_next in frame_t_t1: 
        learner.observe_transition(frame_now, frame_next)
    rules = learner.get_accepted_clauses()
    for r in rules:
        print(r)

    return rules
