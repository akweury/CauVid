from pathlib import Path

from tqdm import tqdm
import os
import random
import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score


from src.exp_roadpp import utils_data
from src.exp_roadpp.step03_language import Language
from src.exp_roadpp.step03_beam_search import BeamSearch
from src.exp_roadpp.step03_visual import visualize_rule_aggregation_results
from src.exp_roadpp.step03_rule_aggregation import build_rule_learning_test_dataset, learn_rule_aggregation, build_rule_learning_dataset


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


def _tracks_to_atoms(track_dir, video_ids, lang, output_dir):
    atom_files = []
    for vid in tqdm(video_ids, desc="Tracks to Atoms"):
        output_file = Path(output_dir) / f"{vid}_atoms.json"
        if output_file.exists():
            atom_files.append(output_file)
            continue
        track_file = Path(track_dir) / f"{vid}_gt.json"
        if not track_file.exists():
            continue
        track_data = utils_data.load_json(track_file)       
        agent_tubes = track_data["data"]["agent_tubes"]
        segments_by_ego_actions = track_data["data"]["av_action_tubes"]
        frames = track_data["data"]["frames"]
        atoms = []
        atoms.extend(lang.segs2atoms("av", segments_by_ego_actions))
        atoms.extend(lang.segs2atoms("agents", agent_tubes, frames))
        utils_data.save_json(atoms, output_file)
        atom_files.append(output_file)
    return atom_files


def _atoms_to_facts(train_ids, lang, output_dir):
    fact_files = []
    all_facts = []
    for vid in tqdm(train_ids, desc="Atoms to Facts"):
        output_file = Path(output_dir) / f"{vid}_facts.json"
        if output_file.exists():
            facts = utils_data.load_json(output_file)
            all_facts.extend(facts)
        else:
            atom_file = Path(output_dir) / f"{vid}_atoms.json"
            atom_data = utils_data.load_json(atom_file)
            vid = Path(atom_file).stem.replace("_atoms", "")
            fact_files.append(output_file)
            facts = lang.atoms2facts(atom_data)
            all_facts.extend(facts)
            utils_data.save_json(facts, output_file)
    return fact_files, all_facts



def test_global_rules(model, rules, language_model, output_dir, track_dir, test_indices):
    all_track_files =[os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.endswith("_gt.json")]
    test_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in test_indices]

    _tracks_to_atoms(track_dir, test_ids, language_model, output_dir)
    fact_files, facts = _atoms_to_facts(test_ids, language_model, output_dir)
    
    dataset = build_rule_learning_test_dataset(facts, rules, output_dir, test_indices)
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

    return dataset_summary

def _rules_to_global(track_dir, all_rules, train_indices, val_indices, test_indices, all_facts,val_facts, lang_model, output_dir, test_output_dir):
    train_dataset = build_rule_learning_dataset(all_facts, all_rules, output_dir, train_indices)
    val_dataset = build_rule_learning_dataset(val_facts, all_rules, output_dir, val_indices)
    ranked_rules, model = learn_rule_aggregation(train_dataset,val_dataset)
    dataset_summary = test_global_rules(model, ranked_rules, lang_model, test_output_dir, track_dir, test_indices)
    utils_data.save_json(dataset_summary, test_output_dir / "rule_aggregation_summary.json")
    return ranked_rules, model, dataset_summary

def _facts_to_rules(train_ids, lang, output_dir):
    all_rules = []
    all_rule_file =Path(output_dir) / f"all_rules.json"
    if all_rule_file.exists():
        all_rules = utils_data.load_json(all_rule_file)
        return all_rules 
    for vid in tqdm(train_ids, desc="Facts to Rules"):
        fact_data = utils_data.load_json(Path(output_dir) / f"{vid}_facts.json")
        output_file = Path(output_dir) / f"{vid}_rules.json"
        if output_file.exists():
            all_rules.extend(utils_data.load_json(output_file))
        else:
            r_0 = lang.facts2rules(fact_data)
            utils_data.save_json(r_0, output_file)
            all_rules.extend(r_0)

    utils_data.save_json(all_rules, all_rule_file)
    return all_rules


def main(input_data):
    print("\n------- Step 03 -------\n")
    output_dir = input_data["output_dir"]
    test_output_dir = output_dir / "test"
    os.makedirs(test_output_dir, exist_ok=True)
    track_dir = input_data["dataset_path"] / "gt"
    
    language_model = Language()
    beam_search_model = BeamSearch()

    all_track_files =[os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.endswith("_gt.json")]
    train_indices, val_indices, test_indices = _split_example_indices(len(all_track_files))
    train_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in train_indices]
    val_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in val_indices]
    test_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in test_indices]

    _tracks_to_atoms(track_dir, train_ids, language_model, output_dir)
    fact_files, all_facts = _atoms_to_facts(train_ids, language_model, output_dir)
    all_rules = _facts_to_rules(train_ids, language_model, output_dir)

    _tracks_to_atoms(track_dir, val_ids, language_model, output_dir)
    _, val_facts = _atoms_to_facts(val_ids, language_model, output_dir)

    ranked_rules, rule_model, dataset_summary = _rules_to_global(track_dir, all_rules, train_indices, val_indices, test_indices, all_facts,val_facts, language_model, output_dir, test_output_dir)
    dataset_path = input_data["dataset_path"]
    visualize_rule_aggregation_results(dataset_path, dataset_summary, test_output_dir)
    
    print("\n--------- Step 03 Done ---------------\n")