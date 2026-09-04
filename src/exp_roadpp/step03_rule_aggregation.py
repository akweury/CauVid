import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import sparse
import random
import torch





def save_dataset(data, dataset_file):
    np.savez(dataset_file, **data)


def _unwrap_cached_npz_value(value):
    if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype == object:
        return value.item()
    return value




def _rule_fires_on_fact(rule, fact):
    if int(fact["av_action_id"]) != int(rule["head"]["av_action_id"]):
        return False

    agent_class = int(rule["body"]["agent_class"])
    action_ids = tuple(rule["body"].get("action", []))
    location_ids = tuple(rule["body"].get("location", []))
    for agent in fact.get("agents", []):
        if int(agent["class"]) != agent_class:
            continue
        for pair in agent.get("frame-action-location", []) or []:
            if tuple(pair.get("action_ids", [])) == action_ids and tuple(pair.get("loc_ids", [])) == location_ids:
                return True
    return False


def build_rule_learning_dataset(facts, rules, output_dir,  all_rule_supports, all_head_supports):
    dataset_file = Path(output_dir) / "rule_learning_dataset.npz"

    if dataset_file.exists():
        npz =  np.load(dataset_file, allow_pickle=True)
        return {
            "rules": _unwrap_cached_npz_value(npz["rules"]),
            "feature_matrix": _unwrap_cached_npz_value(npz["feature_matrix"]),
            "labels": np.asarray(npz["labels"], dtype=np.int64),
        }
    
    row_indices = []
    col_indices = []
    data = []
    labels = []
    matrix = torch.zeros((len(all_rule_supports), len(rules)), dtype=torch.float32)

    rules_dict = {}
    for rule_index, rule in enumerate(rules):
        body_signature = (rule['body']['agent_class'], 
                          tuple(rule["body"]["action"]), 
                          tuple(rule["body"]["location"]))
        action_label = int(rule["head"]["av_action_id"])
        rules_dict[(action_label, body_signature)] = rule_index

    for row_index, body_signature in tqdm(enumerate(all_rule_supports), total=len(all_rule_supports), desc="Building rule learning dataset"):
        for action_label in all_rule_supports[body_signature]:
            labels.append(action_label)
            rule_signature = (int(action_label),body_signature)
            rule_index = rules_dict[rule_signature]
            matrix[row_index, rule_index] = all_rule_supports[body_signature][action_label]
            


        # labels.append(fact["av_action_id"])
        # for col_index, rule in enumerate(rules):
        #     if _rule_fires_on_fact(rule, fact):
        #         row_indices.append(row_index)
        #         col_indices.append(col_index)
        #         data.append(1)

    feature_matrix = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(len(facts), len(rules)), dtype=np.float32)

    if isinstance(feature_matrix, np.ndarray) and feature_matrix.ndim == 0:
        feature_matrix = feature_matrix.item()    

    data = {
        'feature_matrix': feature_matrix,
        'labels': labels,
        'rules': rules,
    }
    save_dataset(data, dataset_file)

    return data


def build_rule_learning_test_dataset(facts, rules, output_dir, test_indices):

    dataset_file = Path(output_dir) / "rule_learning_test_dataset.npz"
    if dataset_file.exists():
        npz =  np.load(dataset_file, allow_pickle=True)
        return {
            "rules": _unwrap_cached_npz_value(npz["rules"]),
            "test_matrix": _unwrap_cached_npz_value(npz["test_matrix"]),
            "test_labels": np.asarray(npz["test_labels"], dtype=np.int64),
        }
    normalized_rules = [_normalize_rule(rule) for rule in rules]
    row_indices = []
    col_indices = []
    data = []
    examples = []
    labels = []

    for row_index, fact in tqdm(enumerate(facts), total=len(facts), desc="Building rule learning test dataset"):
        label = int(fact.get("av_action_id", -1))
        labels.append(label)
        examples.append(
            {
                "example_id": row_index,
                "av_action_id": label,
                "start_frame": int(fact.get("start_frame", -1)),
                "end_frame": None if fact.get("end_frame", None) in {None, float("inf")} else int(fact.get("end_frame", -1)),
                "num_agents": len(fact.get("agents", [])),
            }
        )

        for col_index, rule in enumerate(normalized_rules):
            if _rule_fires_on_fact(rule, fact):
                row_indices.append(row_index)
                col_indices.append(col_index)
                data.append(1)

    feature_matrix = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(facts), len(normalized_rules)),
        dtype=np.float32,
    )
    if isinstance(feature_matrix, np.ndarray) and feature_matrix.ndim == 0:
        feature_matrix = feature_matrix.item()    

    test_matrix = feature_matrix[test_indices]
    labels = np.array(labels)
    test_labels = labels[test_indices]

    data = {
        'test_matrix': test_matrix,
        'test_labels': test_labels,
        'rules': normalized_rules,
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
    c_values = [0.05,0.1,0.5,1.0,5.0,10.0]
    best_model = None
    best_key = None
    best_summary = None

    for c_value in c_values:
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=float(c_value),
            max_iter=3000,
            class_weight="balanced",
            random_state=int(seed),
        )
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
        print(f"Evaluated C={c_value}: {summary}")
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

def learn_rule_aggregation(train_dataset, val_dataset):

    train_matrix = train_dataset["feature_matrix"]
    val_matrix = val_dataset["feature_matrix"]
    val_labels = val_dataset["labels"]
    labels = train_dataset["labels"]
    

    model, selection_summary = _fit_rule_aggregation_lr(train_matrix, labels, val_matrix, val_labels, seed=7)
    ranked_rules = _rank_rules_with_model(train_dataset["rules"], model)
    return ranked_rules, model






    