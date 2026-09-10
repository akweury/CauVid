from pathlib import Path
import json 
from tqdm import tqdm
import os
import random
import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

from src.exp_roadpp import utils_data
from src.exp_roadpp.step03_language import Language
from src.exp_roadpp.step03_beam_search import BeamSearch
from src.exp_roadpp.step03_visual import visualize_rule_aggregation_results, visualize_baseline_results
from src.exp_roadpp.step03_rule_aggregation import build_rule_learning_test_dataset, learn_rule_aggregation, build_rule_learning_dataset
from src.exp_roadpp.baselines import transformer_baseline, ilp_baseline, gnn_baseline, lstm_baseline

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
    atom_dir = Path(output_dir) / "atoms"
    os.makedirs(atom_dir, exist_ok=True)
    atom_file = atom_dir / "all_atoms.json"

    atoms_by_video = utils_data.load_json(atom_file) if atom_file.exists() else {}

    missing_ids = [vid for vid in video_ids if vid not in atoms_by_video]
    if missing_ids:
        for vid in tqdm(missing_ids, desc="Tracks to Atoms"):
            track_file = Path(track_dir) / f"{vid}_gt.json"
            if not track_file.exists():
                continue

            track_data = utils_data.load_json(track_file)       

            agent_tubes = track_data["data"]["agent_tubes"]
            segments_by_ego_actions = track_data["data"]["av_action_tubes"]
            frames = track_data["data"]["frames"]

            atoms = []
            atoms.extend(lang.video2atoms("av", segments_by_ego_actions))
            atoms.extend(lang.video2atoms("agents", agent_tubes, frames))

            atoms_by_video[vid] = atoms

        utils_data.save_json(atoms_by_video, atom_file)

    return {vid: atoms_by_video[vid] for vid in video_ids if vid in atoms_by_video}




def _atoms_to_init_clauses(train_ids, lang, output_dir, split):
    facts_dir = Path(output_dir) / "clauses"
    os.makedirs(facts_dir, exist_ok=True)
    all_init_clauses_file = facts_dir / f"all_init_clauses_{split}.json"
    if all_init_clauses_file.exists():
        return utils_data.load_json(all_init_clauses_file)
    all_atoms = utils_data.load_json(Path(output_dir) / "atoms" / "all_atoms.json")
    all_init_clauses = lang.atoms2atom_clauses(all_atoms, head_ungrounded_atoms=[])
    utils_data.save_json(all_init_clauses, all_init_clauses_file)
    return all_init_clauses



def test_global_rules(model, init_clauses, atoms_by_video, output_dir, track_dir, test_indices):
    all_track_files =[os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.endswith("_gt.json")]
    test_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in test_indices]
    dataset = build_rule_learning_dataset(atoms_by_video, init_clauses, output_dir, split="test")
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


def _to_hashable(value):
    if isinstance(value, list):
        return tuple(_to_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _to_hashable(item)) for key, item in value.items())
    return value


def _encode_supports(supports):
    return {json.dumps(key): value for key, value in supports.items()}


def _decode_rule_supports(supports):
    return Counter({_to_hashable(json.loads(key)): value for key, value in supports.items()})


def _decode_head_supports(supports):
    return Counter({json.loads(key): value for key, value in supports.items()})



def load_rule_files(rule_file):
    data = utils_data.load_json(rule_file)
    rules = data['rules']
    rule_supports = _decode_rule_supports(data["rule_supports"])
    head_supports = _decode_head_supports(data["head_supports"])
    return rules, rule_supports, head_supports

def save_rule_files(rules, rule_supports, head_supports, rule_file):
    utils_data.save_json(
        {'rules': rules,
         'rule_supports': _encode_supports(rule_supports),
         'head_supports': _encode_supports(head_supports)
         }, rule_file)

def _facts_to_rules(facts, lang, output_dir):
    rules_dir = output_dir/ 'rules'
    facts_dir = output_dir / 'facts'
    os.makedirs(rules_dir, exist_ok=True)
    os.makedirs(facts_dir, exist_ok=True)
    all_rules = []
    all_rule_file =Path(rules_dir) / f"all_rules.json"
    all_rule_supports =  Counter()
    all_head_supports = Counter()

    if all_rule_file.exists():
        all_rules, all_rule_supports, all_head_supports = load_rule_files(all_rule_file)
        return all_rules, all_rule_supports, all_head_supports
    
    for fact in tqdm(facts, desc="Facts to Rules"):
        r_0, rule_supports, head_supports = lang.facts2rules(fact)
        all_rules = merge_rules(all_rules, r_0)
        all_rule_supports = merge_rule_supports(all_rule_supports, rule_supports)
        all_head_supports = merge_head_supports(all_head_supports, head_supports)
        
    save_rule_files(all_rules, all_rule_supports, all_head_supports, all_rule_file)
    return all_rules, all_rule_supports, all_head_supports

def merge_head_supports(target, source):
    for key, value in source.items():
        if key in target:
            target[key] += source[key]
        else:
            target[key] = value
    return target


def merge_rules(target, source):
    for rule in source:
        if rule not in target:
            target.append(rule)
    return target

def merge_rule_supports(target, source):
    for key, value in source.items():
        if key in target:
            for key2 in source[key]:
                if key2 in target[key]:
                    target[key][key2] += source[key][key2]
                else:
                    target[key][key2] = source[key][key2]
        else:
            target[key] = value
    return target

def main(input_data):
    print("\n--------- Step 03 ----------------------\n")
    output_dir = input_data["output_dir"]
    test_output_dir = output_dir / "test"
    os.makedirs(test_output_dir, exist_ok=True)
    track_dir = input_data["dataset_path"] / "gt"
    dataset_path = input_data["dataset_path"]
    

    language_model = Language(input_data["device"])
    beam_search_model = BeamSearch()

    all_track_files =[os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.endswith("_gt.json")]
    if input_data['data_num'] != 'full':
        all_track_files = all_track_files[:int(input_data['data_num'])]
    train_indices, val_indices, test_indices = _split_example_indices(len(all_track_files))
    
    # train data
    train_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in train_indices]
    atoms_by_train_video = _tracks_to_atoms(track_dir, train_ids, language_model, output_dir)
    init_clauses = _atoms_to_init_clauses(train_ids, language_model, output_dir, 'train')

    if input_data["skip_lr"] == 'True':
        return

    train_dataset = build_rule_learning_dataset(atoms_by_train_video, init_clauses, output_dir, "train")

    # val data
    val_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in val_indices]
    atoms_by_val_video = _tracks_to_atoms(track_dir, val_ids, language_model, output_dir)
    val_dataset = build_rule_learning_dataset(atoms_by_val_video, init_clauses, output_dir, "val")

    # test data
    test_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in test_indices]
    atoms_by_test_video = _tracks_to_atoms(track_dir, test_ids, language_model, output_dir)   

    
    # learn rule aggregation
    ranked_rules, model = learn_rule_aggregation(train_dataset,val_dataset)
    # test data
    dataset_summary = test_global_rules(model, init_clauses, atoms_by_test_video, test_output_dir, track_dir, test_indices)
    
    utils_data.save_json(dataset_summary, test_output_dir / "rule_aggregation_summary.json")
    
    visualize_rule_aggregation_results(dataset_path, dataset_summary, test_output_dir)
    
    print("\n--------- Step 03 Done ---------------\n")


def baselines(input_data):
    print("\n--------- Step 03 Baselines ----------------------\n")
    if input_data["skip_baselines_03"] == 'True':
        return
    
    output_dir = input_data["output_dir"]
    test_output_dir = input_data["test_output_dir"]
    track_dir = input_data["dataset_path"] / "gt"
    dataset_path = input_data["dataset_path"]
    dataset_labels = input_data["dataset_labels"]
    language_model = Language(input_data["device"])
    device = input_data["device"]
    all_track_files =[os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.endswith("_gt.json")]
    if input_data['data_num'] != 'full':
        all_track_files = all_track_files[:int(input_data['data_num'])]
    train_indices, val_indices, test_indices = _split_example_indices(len(all_track_files))
    
    # train data
    train_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in train_indices]
    _tracks_to_atoms(track_dir, train_ids, language_model, output_dir)
    train_facts = _atoms_to_init_clauses(train_ids, language_model, output_dir, 'train')
    # all_rules, all_rule_supports, all_head_supports = _facts_to_rules(train_facts, train_ids, language_model, output_dir)

    # val data
    val_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in val_indices]
    val_facts = _atoms_to_init_clauses(val_ids, language_model, output_dir, 'val')
    # test data
    test_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in test_indices]
    _tracks_to_atoms(track_dir, test_ids, language_model, output_dir)
    test_facts = _atoms_to_init_clauses(test_ids, language_model, output_dir, 'test')

    # train and test baselines
    result_summary = {}
    all_facts = {"train": train_facts, "val": val_facts, "test": test_facts}
    ilp_baseline.run(all_facts, dataset_labels, output_dir, result_summary, device)
    gnn_baseline.run(all_facts, dataset_labels, output_dir, result_summary, device)
    transformer_baseline.run(all_facts, dataset_labels, output_dir, result_summary, device)
    lstm_baseline.run(all_facts, dataset_labels, output_dir, result_summary, device)
    utils_data.save_json(result_summary, test_output_dir / "baselines_summary.json")
    visualize_baseline_results(result_summary, test_output_dir)
    
    print("\n--------- Step 03 Baseline Done! ---------------\n")
