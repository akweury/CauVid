import os
import pickle

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, f1_score

from src.exp_roadpp import utils_data

"""
The ILP (Inductive Logic Programming style) baseline model is implemented in this file.
The model takes the facts of the objects of each video segment as input and predicts the
av_action_id, i.e. the action of the autonomous vehicle in the segment.

The input includes the object class, action_ids, and loc_ids of the objects in the segment.
The model is trained on the training set and evaluated on the test set.

Each fact (segment) has the shape:
    {
        "start_frame": int,
        "end_frame": int,
        "av_action_id": int,
        "agents": [
            {
                "class": int,
                "frame-action-location": [
                    {"frame": int, "action_ids": [...], "loc_ids": [...]},
                    ...
                ],
            },
            ...
        ],
    }

A full ILP solver (e.g. Aleph/Popper/Metagol over Prolog) is not available in this
environment, so this baseline follows the standard practical stand-in: every segment is
represented as a fixed-size vector of ground background-predicate indicators (is there any
agent of class c / any observed action a / any observed location l in this segment?), and a
shallow decision tree is induced over these atoms. A decision tree over binary atoms is
exactly a disjunction of conjunctive if-then rules, which is what an ILP system would learn,
and the induced rules can be dumped to a human-readable text file for inspection.
"""


def _flatten_ids(values):
    flattened = []
    for value in values or []:
        if isinstance(value, (list, tuple, set)):
            flattened.extend(value)
        else:
            flattened.append(value)
    return flattened


def _agent_action_loc_ids(agent):
    action_ids = set()
    loc_ids = set()
    for pair in agent.get("frame-action-location", []) or []:
        action_ids.update(_flatten_ids(pair.get("action_ids", [])))
        loc_ids.update(_flatten_ids(pair.get("loc_ids", [])))
    return action_ids, loc_ids


def _scan_vocab_sizes(*fact_lists):
    max_agent_class = -1
    max_action_id = -1
    max_loc_id = -1
    max_av_action_id = -1
    for facts in fact_lists:
        for fact in facts or []:
            if "av_action_id" in fact:
                max_av_action_id = max(max_av_action_id, int(fact["av_action_id"]))
            for agent in fact.get("agents", []) or []:
                if agent.get("class") is not None:
                    max_agent_class = max(max_agent_class, int(agent["class"]))
                action_ids, loc_ids = _agent_action_loc_ids(agent)
                if action_ids:
                    max_action_id = max(max_action_id, max(int(a) for a in action_ids))
                if loc_ids:
                    max_loc_id = max(max_loc_id, max(int(l) for l in loc_ids))
    return {
        "num_agent_classes": max(1, max_agent_class + 1),
        "num_action_classes": max(1, max_action_id + 1),
        "num_loc_classes": max(1, max_loc_id + 1),
        "num_av_actions": max(1, max_av_action_id + 1),
    }


def _fact_to_feature_vector(fact, vocab):
    num_agent_classes = vocab["num_agent_classes"]
    num_action_classes = vocab["num_action_classes"]
    num_loc_classes = vocab["num_loc_classes"]
    feature = np.zeros(num_agent_classes + num_action_classes + num_loc_classes, dtype=np.float32)
    for agent in fact.get("agents", []) or []:
        agent_class = agent.get("class")
        if agent_class is not None:
            agent_class = min(max(0, int(agent_class)), num_agent_classes - 1)
            feature[agent_class] = 1.0
        action_ids, loc_ids = _agent_action_loc_ids(agent)
        for action_id in action_ids:
            action_id = int(action_id)
            if 0 <= action_id < num_action_classes:
                feature[num_agent_classes + action_id] = 1.0
        for loc_id in loc_ids:
            loc_id = int(loc_id)
            if 0 <= loc_id < num_loc_classes:
                feature[num_agent_classes + num_action_classes + loc_id] = 1.0
    return feature


def _feature_names(vocab):
    names = [f"has_agent_class_{i}" for i in range(vocab["num_agent_classes"])]
    names += [f"has_action_{i}" for i in range(vocab["num_action_classes"])]
    names += [f"has_loc_{i}" for i in range(vocab["num_loc_classes"])]
    return names


def _facts_to_dataset(facts, vocab):
    num_features = vocab["num_agent_classes"] + vocab["num_action_classes"] + vocab["num_loc_classes"]
    if not facts:
        return np.zeros((0, num_features), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    features = np.stack([_fact_to_feature_vector(fact, vocab) for fact in facts])
    labels = np.array([int(fact["av_action_id"]) for fact in facts], dtype=np.int64)
    return features, labels


class ILPModel:

    def __init__(self, vocab, param_grid=None, seed=7):
        self.vocab = vocab
        self.seed = seed
        self.param_grid = param_grid or [
            {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf}
            for max_depth in (3, 5, 8, None)
            for min_samples_leaf in (1, 5, 10)
        ]
        self.model = None
        self.best_params = None

    def train(self, train_facts, val_facts):
        """
        Fit a decision tree over the background-predicate atoms on the training facts,
        selecting the tree depth / leaf size that generalizes best to the validation facts.

        Args:
            train_facts (list): List of training facts.
            val_facts (list): List of validation facts.
        """
        X_train, y_train = _facts_to_dataset(train_facts, self.vocab)
        if len(y_train) == 0:
            print("No training facts provided; skipping ILP training.")
            return

        X_val, y_val = _facts_to_dataset(val_facts, self.vocab)
        has_val = len(y_val) > 0

        best_score = -1.0
        best_params = self.param_grid[0]
        best_model = None
        for params in self.param_grid:
            candidate = DecisionTreeClassifier(random_state=self.seed, **params)
            candidate.fit(X_train, y_train)
            if has_val:
                score = accuracy_score(y_val, candidate.predict(X_val))
            else:
                score = accuracy_score(y_train, candidate.predict(X_train))
            print(f"[ILP] params={params} {'val' if has_val else 'train'}_accuracy={score:.4f}")
            if score > best_score:
                best_score = score
                best_params = params
                best_model = candidate

        self.best_params = best_params
        self.model = best_model
        print(f"[ILP] Selected params={best_params} with accuracy={best_score:.4f}")

    def evaluate(self, test_facts, dataset_labels):
        """
        Evaluate the ILP model on the test facts.

        Args:
            test_facts (list): List of test facts.
            dataset_labels (dict): Dictionary containing dataset labels.

        Returns:
            dict: Evaluation results.
        """
        X_test, y_test = _facts_to_dataset(test_facts, self.vocab)
        if self.model is None or len(y_test) == 0:
            return {
                "test_label_count": 0,
                "test_accuracy": 0.0,
                "test_f1_macro": 0.0,
                "test_accuracy_per_class": {},
            }

        av_action_label_map = dataset_labels["av_action_labels"]
        preds = self.model.predict(X_test)

        test_accuracy = float(accuracy_score(y_test, preds))
        test_f1_macro = float(f1_score(y_test, preds, average="macro"))

        test_accuracy_per_class = {}
        for class_label in sorted(set(y_test.tolist())):
            class_indices = [i for i, label in enumerate(y_test) if label == class_label]
            class_correct = sum(1 for i in class_indices if preds[i] == class_label)
            action_name = av_action_label_map[int(class_label)]
            test_accuracy_per_class[action_name] = float(class_correct) / len(class_indices)
            test_accuracy_per_class[f"{action_name}_Num"] = len(class_indices)
            test_accuracy_per_class[f"{action_name}_Correct"] = class_correct
        print(f"[ILP] Test accuracy: {test_accuracy:.4f}, F1 macro: {test_f1_macro:.4f}")
        for action_name, acc in test_accuracy_per_class.items():
            if not action_name.endswith("_Num") and not action_name.endswith("_Correct"):
                print(f"  {action_name}: accuracy={acc:.4f}, Num={test_accuracy_per_class[action_name + '_Num']}, "
                      f"Correct={test_accuracy_per_class[action_name + '_Correct']}")
        return {
            "test_label_count": len(set(y_test.tolist())),
            "test_accuracy": test_accuracy,
            "test_f1_macro": test_f1_macro,
            "test_accuracy_per_class": test_accuracy_per_class,
        }

    def export_rules(self, dataset_labels):
        """Render the induced decision tree as human-readable if-then rules."""
        if self.model is None:
            return ""
        av_action_label_map = dataset_labels["av_action_labels"]
        feature_names = _feature_names(self.vocab)
        class_names = [av_action_label_map[int(class_id)] for class_id in self.model.classes_]
        return export_text(self.model, feature_names=feature_names, class_names=class_names)

    def save(self, model_path):
        with open(model_path, "wb") as f:
            pickle.dump({"model": self.model, "vocab": self.vocab, "best_params": self.best_params}, f)


def prepare_dataset(train_facts, val_facts, test_facts):
    """
    Prepare the dataset for the ILP model, i.e. compute the vocabulary sizes
    (agent classes, action ids, location ids and av_action ids) shared across the
    train/val/test splits so they can all be encoded into the same background-predicate
    feature space.

    Args:
        train_facts (list): List of training facts.
        val_facts (list): List of validation facts.
        test_facts (list): List of test facts.

    Returns:
        dict: Vocabulary sizes (num_agent_classes, num_action_classes, num_loc_classes, num_av_actions).
    """
    return _scan_vocab_sizes(train_facts, val_facts, test_facts)


def run(all_facts, dataset_labels, output_dir, result_summary, device=None):
    """
    Run the ILP baseline on the given train, validation, and test sets.

    Args:
        all_facts (dict): Dictionary containing 'train', 'val', and 'test' facts.
        dataset_labels (dict): Dictionary containing dataset labels.
        output_dir (Path): Directory to save the results.
        result_summary (dict): Dictionary to store the results summary.
        device: Unused, kept for interface parity with the neural baselines (ILP runs on CPU).
    """
    print("\n--------- Running ILP Baseline ----------------------\n")
    model_output_dir = output_dir / "ilp"
    os.makedirs(model_output_dir, exist_ok=True)

    train_facts, val_facts, test_facts = all_facts["train"], all_facts["val"], all_facts["test"]
    ilp_results = {}
    ilp_model_file = model_output_dir / "ilp_model.pkl"  # Path to save the trained model
    ilp_result_file = model_output_dir / "ilp_results.json"  # Path to save the results
    ilp_rules_file = model_output_dir / "ilp_rules.txt"  # Path to save the induced rules
    if ilp_model_file.exists() and ilp_result_file.exists():
        print("ILP model and results already exist. Loading them...")
        # Load the model and results if they already exist
        ilp_results = utils_data.load_json(ilp_result_file)
    else:
        vocab = prepare_dataset(train_facts, val_facts, test_facts)
        ilp_model = ILPModel(vocab)  # Initialize the ILP model
        ilp_model.train(train_facts, val_facts)  # Train the model
        ilp_model.save(ilp_model_file)  # Save the trained model
        ilp_results = ilp_model.evaluate(test_facts, dataset_labels)  # Evaluate on test set
        utils_data.save_json(ilp_results, ilp_result_file)  # Save the results
        with open(ilp_rules_file, "w") as f:
            f.write(ilp_model.export_rules(dataset_labels))  # Save the induced rules
    result_summary["ilp"] = ilp_results
    print("\n--------- ILP Baseline Done! ------------------------\n")