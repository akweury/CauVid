from pathlib import Path
import random

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


from src.exp_roadpp import utils_data
from src.exp_roadpp.step03_language import Language
from src.exp_roadpp.step03_beam_search import BeamSearch




def _tracks_to_atoms(track_files, lang, output_dir):
    atom_files = []
    for track_file in track_files:
        track_data = utils_data.load_json(track_file)
        vid = track_data["vid"]
        output_file = Path(output_dir) / f"{vid}_atoms.json"
        atom_files.append(output_file)

        if output_file.exists():
            print(f"Atoms file already exists: {output_file}")
            continue

        agent_tubes = track_data["data"]["agent_tubes"]
        segments_by_ego_actions = track_data["data"]["av_action_tubes"]
        frames = track_data["data"]["frames"]
        atoms = []
        atoms.extend(lang.segs2atoms("av", segments_by_ego_actions))
        atoms.extend(lang.segs2atoms("agents", agent_tubes, frames))
        utils_data.save_json(atoms, output_file)

    return atom_files


def _atoms_to_facts(atom_files, lang, output_dir):
    fact_files = []
    for atom_file in atom_files:
        print(f"Converting atoms to rules for {atom_file}")
        atom_data = utils_data.load_json(atom_file)
        vid = Path(atom_file).stem.replace("_atoms", "")
        output_file = Path(output_dir) / f"{vid}_facts.json"
        fact_files.append(output_file)

        if output_file.exists():
            print(f"Facts file already exists: {output_file}")
            continue

        facts = lang.atoms2facts(atom_data)
        utils_data.save_json(facts, output_file)
    return fact_files


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


def _normalize_rule(rule):
    normalized = dict(rule)
    head = dict(normalized.get("head", {}))
    body = dict(normalized.get("body", {}))
    support = int(normalized.get("support", getattr(rule, "support", 0)) or 0)
    total_support = int(normalized.get("total_support", getattr(rule, "total_support", 1)) or 1)
    confidence = float(normalized.get("confidence", getattr(rule, "confidence", 0.0)) or 0.0)
    coverage = float(normalized.get("coverage", support / max(1, total_support)))
    rank_key = list(normalized.get("rank_key", []))
    if not rank_key:
        rank_key = [
            -support,
            -round(coverage, 12),
            -round(confidence, 12),
            int(head.get("av_action_id", -1)),
            int(body.get("agent_class", -1)),
            list(body.get("action", [])),
            list(body.get("location", [])),
        ]
    return {
        "head": {"av_action_id": int(head.get("av_action_id", -1))},
        "body": {
            "agent_class": int(body.get("agent_class", -1)),
            "action": list(body.get("action", [])),
            "location": list(body.get("location", [])),
        },
        "support": support,
        "total_support": max(1, total_support),
        "coverage": coverage,
        "confidence": confidence,
        "rank_key": rank_key,
    }


def _rule_fires_on_fact(rule, fact):
    if int(fact.get("av_action_id", -1)) != int(rule["head"]["av_action_id"]):
        return False

    agent_class = int(rule["body"]["agent_class"])
    action_ids = tuple(rule["body"].get("action", []))
    location_ids = tuple(rule["body"].get("location", []))
    for agent in fact.get("agents", []):
        if int(agent.get("class", -1)) != agent_class:
            continue
        for pair in agent.get("frame-action-location", []) or []:
            if tuple(pair.get("action_ids", [])) == action_ids and tuple(pair.get("loc_ids", [])) == location_ids:
                return True
    return False


def _split_example_indices(num_examples, train_fraction=0.7, val_fraction=0.15, seed=7):
    indices = list(range(num_examples))
    if num_examples <= 2:
        return indices, indices, indices

    rng = random.Random(int(seed))
    rng.shuffle(indices)

    train_end = max(1, int(round(num_examples * float(train_fraction))))
    val_end = max(train_end + 1, int(round(num_examples * float(train_fraction + val_fraction))))
    val_end = min(num_examples - 1, val_end)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    if not val_indices:
        val_indices = indices[-1:]
    if not test_indices:
        test_indices = indices[-1:]

    return train_indices, val_indices, test_indices


def _build_rule_learning_dataset(facts, rules):
    if sparse is None:
        raise RuntimeError(
            "Rule aggregation baseline requires scipy and scikit-learn in the runtime environment."
        )

    normalized_rules = [_normalize_rule(rule) for rule in rules]
    row_indices = []
    col_indices = []
    data = []
    examples = []
    labels = []

    for row_index, fact in enumerate(facts):
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

    return {
        "examples": examples,
        "labels": labels,
        "feature_matrix": feature_matrix,
        "rules": normalized_rules,
    }


def _fit_rule_aggregation_lr(train_matrix, train_labels, val_matrix, val_labels, seed=7):
    if LogisticRegression is None or accuracy_score is None or f1_score is None:
        raise RuntimeError(
            "Rule aggregation baseline requires scipy and scikit-learn in the runtime environment."
        )

    c_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    best_model = None
    best_key = None
    best_summary = None

    for c_value in c_values:
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            multi_class="ovr",
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


def _learn_rule_aggregation(rules, facts, output_dir):
    
    dataset = _build_rule_learning_dataset(facts, rules)
    train_indices, val_indices, test_indices = _split_example_indices(len(dataset["examples"]))
    feature_matrix = dataset["feature_matrix"]
    labels = np.asarray(dataset["labels"], dtype=np.int64)

    train_matrix = feature_matrix[train_indices]
    val_matrix = feature_matrix[val_indices]
    test_matrix = feature_matrix[test_indices]
    train_labels = labels[train_indices].tolist()
    val_labels = labels[val_indices].tolist()
    test_labels = labels[test_indices].tolist()

    model, selection_summary = _fit_rule_aggregation_lr(
        train_matrix,
        train_labels,
        val_matrix,
        val_labels,
    )

    ranked_rules = _rank_rules_with_model(dataset["rules"], model)
    selected_rules = ranked_rules[: min(50, len(ranked_rules))]

    test_pred = model.predict(test_matrix)
    test_accuracy = float(accuracy_score(test_labels, test_pred)) if len(test_labels) else 0.0
    test_f1_macro = float(f1_score(test_labels, test_pred, average="macro")) if len(test_labels) else 0.0

    dataset_summary = {
        "num_examples": len(dataset["examples"]),
        "num_rules": len(dataset["rules"]),
        "num_train_examples": len(train_indices),
        "num_val_examples": len(val_indices),
        "num_test_examples": len(test_indices),
        "train_label_count": len(set(train_labels)),
        "val_label_count": len(set(val_labels)),
        "test_label_count": len(set(test_labels)),
        "selection_summary": selection_summary,
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1_macro,
    }

    output_dir = Path(output_dir)
    utils_data.save_json(dataset_summary, output_dir / "rule_aggregation_summary.json")
    utils_data.save_json(dataset["examples"], output_dir / "rule_aggregation_examples.json")
    utils_data.save_json(selected_rules, output_dir / "rule_aggregation_rules.json")
    return selected_rules

def _rules_to_global(rule_files, fact_files, bs_model, output_dir):
    r_0 = []
    for rule_file in rule_files:
        r_0.extend(utils_data.load_json(rule_file))
    global_facts = []
    for fact_file in fact_files:
        global_facts.extend(utils_data.load_json(fact_file))

    
    r_k = bs_model.search(r_0)
    rules = _learn_rule_aggregation(r_k, global_facts, output_dir)
    
    

def _facts_to_rules(fact_files, lang, output_dir):
    rule_files = []
    
    for fact_file in fact_files:
        print(f"Converting facts to rules for {fact_file}")
        fact_data = utils_data.load_json(fact_file)
        vid = Path(fact_file).stem.replace("_facts", "")
        output_file = Path(output_dir) / f"{vid}_rules.json"
        rule_files.append(output_file)

        if output_file.exists():
            print(f"Rules file already exists: {output_file}")
            continue
        r_0 = lang.facts2rules(fact_data)
        utils_data.save_json(r_0, output_file)
    return rule_files


def main(input_data):
    print("\n------- Step 03 -------\n")
    output_dir = input_data["output_dir"]
    step01_output_dir = input_data["step01_output_dir"]
    track_dir = step01_output_dir / "gt"
    track_files = list(track_dir.glob("*_gt.json"))

    language_model = Language()
    beam_search_model = BeamSearch()
    atom_files = _tracks_to_atoms(track_files, language_model, output_dir)
    fact_files = _atoms_to_facts(atom_files, language_model, output_dir)
    rule_files = _facts_to_rules(fact_files, language_model, output_dir)
    _rules_to_global(rule_files,fact_files, beam_search_model, output_dir)
    print("\n--------- Step 03 Done ---------------\n")