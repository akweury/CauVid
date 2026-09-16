import os
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import sparse
import random

from src.exp_roadpp import utils_data
from src.exp_roadpp.step03_visual import visualize_rule_aggregation_results, visualize_baseline_results, visual_bar


def save_dataset(data, dataset_file):
    np.savez(dataset_file, **data)


def _unwrap_cached_npz_value(value):
    if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype == object:
        return value.item()
    return value




def _flatten_ids(values):
    flattened = []
    for value in values or []:
        if isinstance(value, (list, tuple, set)):
            flattened.extend(value)
        else:
            flattened.append(value)
    return tuple(flattened)


def _rule_body_signature(rule):
    body = rule["body"]
    return (
        int(body["agent_class"]),
        tuple(body.get("action", [])),
        tuple(body.get("location", [])),
    )


def _build_body_signature_index(rules):
    """Map each rule's body signature (agent_class, action_ids, location_ids) to
    the list of rule column indices sharing that body, so a fact's agent
    behaviors can be matched against all candidate rules with a single dict
    lookup instead of scanning every rule."""
    index = {}
    for rule_index, rule in enumerate(rules):
        index.setdefault(_rule_body_signature(rule), []).append(rule_index)
    return index


def _fact_matching_rule_indices(fact, body_signature_index):
    matches = []
    for agent in fact.get("agents", []) or []:
        agent_class = agent.get("class")
        if agent_class is None:
            continue
        for pair in agent.get("frame-action-location", []) or []:
            action_ids = _flatten_ids(pair.get("action_ids", []))
            loc_ids = _flatten_ids(pair.get("loc_ids", []))
            if not action_ids or not loc_ids:
                continue
            rule_indices = body_signature_index.get((int(agent_class), action_ids, loc_ids))
            if rule_indices:
                matches.extend(rule_indices)
    return matches



def _atom_body_signature(atom):
    if "action_id" in atom:
        return (atom["pred"], atom["agent_class"], "action_id", atom["action_id"])
    elif "location_name" in atom:
        return (atom["pred"], atom["agent_class"], "location_name", atom["location_name"])
    else:
        raise ValueError(f"Unsupported atom format: {atom}")

def _clause_body_signature(clause):
    return _atom_body_signature(clause["body"][0])

def _build_body_index(init_clauses):
    signatures = {_clause_body_signature(c['clause']) for c in init_clauses.values()}
    return {sig: i for i, sig in enumerate(signatures)}, signatures

HEAD_PRED, HEAD_AGENT_CLASS = "action", "av"
def _head_examples(atoms):
    head_atoms = [a for a in atoms if a["pred"] == HEAD_PRED and a["agent_class"] == HEAD_AGENT_CLASS]
    body_atoms = [a for a in atoms if not (a["pred"] == HEAD_PRED and a["agent_class"] == HEAD_AGENT_CLASS)]
    examples = []
    for head in head_atoms:
        present = {
            _atom_body_signature(a) for a in body_atoms
            if int(a["end_frame"]) <= int(head["start_frame"]) 
        }
        examples.append((int(head["action_id"]),present))
    return examples 


def _build_clause_feature_matrix(atoms_by_video, body_index, desc):
    row_indices, col_indices, data, labels = [], [], [], []
    row = 0
    for vid, atoms in tqdm(atoms_by_video.items(), desc=desc):
        for label, present in _head_examples(atoms):
            labels.append(label)
            for sig in present:
                col = body_index.get(sig)
                if col is not None:
                    row_indices.append(row)
                    col_indices.append(col)
                    data.append(1)
            row += 1
    feature_matrix = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(row, len(body_index)),
        dtype=np.float32,
    )
    return feature_matrix, np.asarray(labels, dtype=np.int64)


def build_rule_learning_dataset(atoms_by_video, init_clauses, output_dir, split, model_type):
    dataset_file = Path(output_dir) / f"rule_learning_dataset_{model_type}_{split}.npz"
    if dataset_file.exists():
        npz =  np.load(dataset_file, allow_pickle=True)
        return {
            "body_signatures": _unwrap_cached_npz_value(npz["body_signatures"]),
            "feature_matrix": _unwrap_cached_npz_value(npz["feature_matrix"]),
            "labels": np.asarray(npz["labels"], dtype=np.int64),
        }

    body_index, body_signatures = _build_body_index(init_clauses)
    feature_matrix, labels = _build_clause_feature_matrix(atoms_by_video, 
                                                          body_index,
                                                          desc=f"Building {split} dataset")
    data = {"feature_matrix": feature_matrix, "labels": labels, "body_signatures": body_signatures}
    save_dataset(data, dataset_file)

    return data


def build_rule_learning_test_dataset(facts, rules, output_dir, test_indices):

    dataset_file = Path(output_dir) / "rule_learning_test_dataset.npz"
    if dataset_file.exists():
        npz =  np.load(dataset_file, allow_pickle=True)
        return {
            "rules": _unwrap_cached_npz_value(npz["rules"]),
            "feature_matrix": _unwrap_cached_npz_value(npz["feature_matrix"]),
            "labels": np.asarray(npz["labels"], dtype=np.int64),
        }

    feature_matrix, labels = _build_clause_feature_matrix(atoms_by_video, body_index, desc="Building rule learning test dataset")

    data = {
        'feature_matrix': feature_matrix,
        'labels': labels,
        'rules': rules,
    }
    save_dataset(data, dataset_file)

    return data



def _rule_signature(rule):
    head = dict(rule.get("head", {}))
    body = dict(rule.get("body", {}))
    return (
        int(head.get("av_action_id", -1)),
        int(body.get("agent_class", -1)),
        tuple(body.get("action", [])),
        tuple(body.get("location", [])),
    )

def _rule_id(rule):
    head_id, agent_class, action_ids, location_ids = _rule_signature(rule)
    action_text = "-".join(str(value) for value in action_ids)
    location_text = "-".join(str(value) for value in location_ids)
    return f"{head_id}:{agent_class}:{action_text}:{location_text}"


def _fit_rule_aggregation_lr(train_matrix, train_labels, val_matrix, val_labels, seed=7):
    if LogisticRegression is None or accuracy_score is None or f1_score is None:
        raise RuntimeError(
            "Rule aggregation baseline requires scipy and scikit-learn in the runtime environment."
        )
    # Evaluated C=0.05: {'c_value': 0.05, 'validation_accuracy': 0.9111842105263158, 'validation_f1_macro': 0.8888809701980068, 'nonzero_rule_count': 3244}
    # Evaluated C=0.1: {'c_value': 0.1, 'validation_accuracy': 0.9013157894736842, 'validation_f1_macro': 0.8832540231082783, 'nonzero_rule_count': 756}
    # Evaluated C=0.5: {'c_value': 0.5, 'validation_accuracy': 0.9523026315789473, 'validation_f1_macro': 0.9212468542512752, 'nonzero_rule_count': 1163}
    # Evaluated C=1.0: {'c_value': 1.0, 'validation_accuracy': 0.9671052631578947, 'validation_f1_macro': 0.9346958647854106, 'nonzero_rule_count': 1295}
    # Evaluated C=5.0: {'c_value': 5.0, 'validation_accuracy': 0.975328947368421, 'validation_f1_macro': 0.9437663887993266, 'nonzero_rule_count': 1812}
    # Evaluated C=10.0: {'c_value': 10.0, 'validation_accuracy': 0.9819078947368421, 'validation_f1_macro': 0.951205349022739, 'nonzero_rule_count': 2193}
    c_values = [0.05, 0.1,0.5,1.0,5.0,10.0]
    best_model = None
    best_key = None
    best_summary = None

    for c_value in c_values:
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=float(c_value),
            max_iter=5000,
            tol=1e-3,
            class_weight="balanced",
            random_state=int(seed),
        )
        # With very few training examples relative to the number of candidate
        # rule features, weakly-regularized (large C) fits are close to
        # separable and saga will not fully converge no matter how large
        # max_iter is. That non-convergence is already reflected in the
        # validation metrics below, so we silence the noisy per-fit warning
        # instead of chasing convergence with ever more iterations.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(train_matrix, train_labels)

        val_pred = model.predict(val_matrix)
        val_accuracy = float(accuracy_score(val_labels, val_pred))
        val_f1 = float(f1_score(val_labels, val_pred, average="macro"))
        nonzero_rule_count = int(np.count_nonzero(np.abs(model.coef_) > 1e-12))

        candidate_key = (val_f1, val_accuracy, -nonzero_rule_count, -float(c_value))
        summary = {
            "c_value": float(c_value),
            "validation_accuracy": val_accuracy,
            "validation_f1_macro": val_f1,
            "nonzero_rule_count": nonzero_rule_count,
        }
        print(f"[Val] c_value={c_value}, acc: {val_accuracy}, f1: {val_f1}, nonzero_rule_count: {nonzero_rule_count}")
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_model = model
            best_summary = summary

    return best_model, best_summary


def _rank_rules_with_model(rules, model):
    class_labels = [int(label) for label in list(model.classes_)]
    coefficients = np.asarray(model.coef_, dtype=np.float64)
    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)

    ranked_rules = []
    for rule_index, rule in enumerate(rules):
        rule = dict(rule)
        column = coefficients[:, rule_index] if rule_index < coefficients.shape[1] else np.zeros(len(class_labels))
        abs_column = np.abs(column)
        best_class_index = int(np.argmax(abs_column)) if len(abs_column) else 0
        best_class_label = class_labels[best_class_index] if class_labels else -1
        best_weight = float(column[best_class_index]) if len(column) else 0.0

        rule.update(
            {
                "rule_id": _rule_id(rule),
                "lr_weight": best_weight,
                "lr_abs_weight": float(abs_column[best_class_index]) if len(abs_column) else 0.0,
                "lr_predicted_class": best_class_label,
                "lr_class_weights": {
                    str(class_label): float(column[class_index])
                    for class_index, class_label in enumerate(class_labels)
                },
            }
        )
        ranked_rules.append(rule)

    ranked_rules.sort(
        key=lambda row: (
            -float(row.get("lr_abs_weight", 0.0)),
            -float(row.get("support", 0)),
            tuple(row.get("rank_key", [])),
            str(row.get("rule_id", "")),
        )
    )
    return ranked_rules

def test_global_rules(model,dataset, output_dir, suffix):
    
    os.makedirs(output_dir, exist_ok=True)

    test_matrix = dataset["feature_matrix"]
    test_labels = dataset["labels"]

    if isinstance(test_matrix, np.ndarray) and test_matrix.ndim == 0:
        test_matrix = test_matrix.item()    

    test_pred = model.predict(test_matrix)

    test_accuracy = float(accuracy_score(test_labels, test_pred)) if len(test_labels) else 0.0
    test_f1_macro = float(f1_score(test_labels, test_pred, average="macro")) if len(test_labels) else 0.0

    # accuracy on each class
    test_accuracy_per_class = {}
    if len(test_labels):
        for class_label in set(test_labels):
            class_indices = [i for i, label in enumerate(test_labels) if label == class_label]
            class_correct = sum(1 for i in class_indices if test_pred[i] == class_label)
            test_accuracy_per_class[int(class_label)] = float(class_correct) / len(class_indices) if class_indices else 0.0

    dataset_summary = {
        "test_label_count": len(set(test_labels)),
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1_macro,
        "test_accuracy_per_class": test_accuracy_per_class,
    }
    utils_data.save_json(dataset_summary, output_dir / f"rule_aggregation_summary_{suffix}.json")
    print(f"[Test] Acc: {test_accuracy}, F1 Macro: {test_f1_macro}")
    return dataset_summary



def learn_rule_aggregation(atoms, clauses, output_dir, dataset_path, model_type):
    print(f"\n[Learn Rule Aggregation] Model type: {model_type}, Clause count: {len(clauses)}")
    train_dataset = build_rule_learning_dataset(atoms["train"], clauses, output_dir, "train", model_type)
    val_dataset = build_rule_learning_dataset(atoms["val"], clauses, output_dir, "val", model_type)
    test_dataset = build_rule_learning_dataset(atoms["test"], clauses, output_dir, "test", model_type)

    train_matrix = train_dataset["feature_matrix"]
    val_matrix = val_dataset["feature_matrix"]
    val_labels = val_dataset["labels"]
    labels = train_dataset["labels"]
    

    model, selection_summary = _fit_rule_aggregation_lr(train_matrix, labels, val_matrix, val_labels, seed=7)
    # ranked_rules = _rank_rules_with_model(train_dataset["rules"], model)
    # save model and ranked rules
    # np.save("ranked_rules.npy", ranked_rules)
    model_file = f"model_{model_type}.npy"
    np.save(model_file, model)

    dataset_summary = test_global_rules(model, test_dataset, output_dir, model_type)
    visualize_rule_aggregation_results(dataset_path, dataset_summary, output_dir, model_type)    
    






    