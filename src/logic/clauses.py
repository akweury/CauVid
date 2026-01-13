
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

    def str(self) -> str:
        return f"{self.head}(O):-{self.body}(O)."

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

    def observe_transition(self, scene_t, scene_t1, frame_data):

        for obj_id, obj_t in scene_t.items():
            if obj_id not in scene_t1:
                continue
            obj_t1 = scene_t1[obj_id]
            for clause in self.clauses:
                body_fn = self.body_predicates[clause.body]
                head_fn = self.head_predicates[clause.head]
                body_true = body_fn(obj_t, scene_t, frame_data)
                head_true = head_fn(obj_t, obj_t1)
                clause.update(body_true, head_true)

    def get_accepted_clauses(self) -> List[Clause]:
        return [
            c for c in self.clauses
            if c.confidence() >= self.conf_threshold
            and c.coverage() >= self.cov_threshold
        ]


class ClauseEvaluator:

    def __init__(self, clauses, body_predicates, head_predicates):
        self.clauses = clauses
        self.body_predicates = body_predicates
        self.head_predicates = head_predicates

    def evaluate_transition(self, scene_t, scene_t1):
        violations = []

        for obj_id, obj_t in scene_t.items():
            if obj_id not in scene_t1:
                continue

            obj_t1 = scene_t1[obj_id]

            for clause in self.clauses:
                body_fn = self.body_predicates[clause.body]
                head_fn = self.head_predicates[clause.head]
                body_true = body_fn(obj_t, scene_t)
                head_true = head_fn(obj_t, obj_t1)

                if body_true and not head_true:
                    violations.append({
                        "clause": str(clause),
                        "object_id": obj_id
                    })
        return violations

    def evaluate_video(self, scene_sequence, frame_data):
        report = {
            clause.str(): {
                "applications": 0,
                "violations": 0,
                "violation_rate": 0.0,
            } for clause in self.clauses
        }
        active_violations = {}
        violation_events = {clause.str(): [] for clause in self.clauses}

        for i in range(len(scene_sequence)):
            scene_t, scene_t1 = scene_sequence[i]

            for obj_id, obj_t in scene_t.items():
                if obj_id not in scene_t1:
                    continue

                obj_t1 = scene_t1[obj_id]

                for clause in self.clauses:
                    key = (obj_id, clause.str())
                    body_fn = self.body_predicates[clause.body]
                    head_fn = self.head_predicates[clause.head]
                    body_true = body_fn(obj_t, scene_t, frame_data)
                    head_true = head_fn(obj_t, obj_t1)

                    violating = body_true and not head_true

                    # ------------------------
                    # CASE 1: Violation Starts
                    # ------------------------
                    if violating and key not in active_violations:
                        active_violations[key] = {
                            "start": i,
                            "last": i
                        }
                    # ------------------------
                    # CASE 2: Violation Continues
                    # ------------------------
                    elif violating and key in active_violations:
                        active_violations[key]["last"] = i

                    # ------------------------
                    # CASE 3: Violation Ends
                    # ------------------------

                    elif not violating and key in active_violations:
                        v = active_violations.pop(key)
                        violation_events[clause.str()].append({
                            "object_id": obj_id,
                            "start_frame": v["start"],
                            "end_frame": v["last"],
                            "duration": v["last"] - v["start"] + 1
                        })

                    if body_true:
                        report[clause.str()]["applications"] += 1
                        if not head_true:
                            report[clause.str()]["violations"] += 1

        # Handle any ongoing violations at the end of the video
        for (obj_id, clause_str), v in active_violations.items():
            violation_events[clause_str].append({
                "object_id": obj_id,
                "start_frame": v["start"],
                "end_frame": v["last"],
                "duration": v["last"] - v["start"] + 1
            })

        for clause_str, data in report.items():
            applications = data["applications"]
            violations = data["violations"]
            violation_rate = violations / applications if applications > 0 else 0.0
            data["violation_rate"] = violation_rate

        return report, violation_events


def learn_rules(frames_data, scene_dataset):
    from logic.predicates import left, right, moves_down, moves_up, red, blue
    frame_t_t1 = []
    frame_data = {
        "width": frames_data[0].frame_size[0],
        "height": frames_data[0].frame_size[1]
    }

    for i in range(len(scene_dataset)-1):
        frame_t_t1.append((scene_dataset[i], scene_dataset[i+1]))

    body_preds = {
        "left": left,
        "right": right,
        "red": red,
        "blue": blue,}

    head_preds = {
        "moves_up": moves_up,
        "moves_down": moves_down}

    learner = ClauseLearner(body_preds, head_preds)

    for frame_now, frame_next in frame_t_t1:
        learner.observe_transition(frame_now, frame_next, frame_data)
    rules = learner.get_accepted_clauses()
    for r in rules:
        print(r)

    return rules


def make_predictions(observed_rules, processed_ts_matrix, frame_data):
    from logic.predicates import left, right, moves_down, moves_up, red, blue
    frame_t_t1 = []
    for i in range(len(processed_ts_matrix)-1):
        frame_t_t1.append(processed_ts_matrix[i:i+2])

    body_preds = {
        "left": left,
        "right": right,
        "red": red,
        "blue": blue,}

    head_preds = {
        "moves_up": moves_up,
        "moves_down": moves_down}

    evaluator = ClauseEvaluator(
        observed_rules,
        body_preds,
        head_preds
    )
    report, violation_events = evaluator.evaluate_video(frame_t_t1, frame_data)

    for rule, stats in report.items():
        print(f"Learned Rule: {rule}")
        print(f"  Applications: {stats['applications']}")
        print(f"  Violations: {stats['violations']}")
        print(f"  Violation Rate: {stats['violation_rate']:.3f}")

    # also print violation events
    for rule, events in violation_events.items():
        print(f"Violation Events for Rule: {rule}")
        for event in events:
            print(f"  Object ID: {event['object_id']}, "
                  f"Start Frame: {event['start_frame']}, "
                  f"End Frame: {event['end_frame']}, "
                  f"Duration: {event['duration']} frames")

    return report, violation_events
