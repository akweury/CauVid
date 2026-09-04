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
    atom_dir = Path(output_dir) / "atoms"
    os.makedirs(atom_dir, exist_ok=True)
    for vid in tqdm(video_ids, desc="Tracks to Atoms"):
        output_file = atom_dir / f"{vid}_atoms.json"
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
    facts_dir = Path(output_dir) / "facts"
    os.makedirs(facts_dir, exist_ok=True)
    fact_files = []
    all_facts = []
    for vid in tqdm(train_ids, desc="Atoms to Facts"):
        output_file = facts_dir / f"{vid}_facts.json"
        if output_file.exists():
            facts = utils_data.load_json(output_file)
            all_facts.extend(facts)
        else:
            atom_file = Path(output_dir) / "atoms" / f"{vid}_atoms.json"
            atom_data = utils_data.load_json(atom_file)
            vid = Path(atom_file).stem.replace("_atoms", "")
            fact_files.append(output_file)
            facts = lang.atoms2facts(atom_data)
            all_facts.extend(facts)
            utils_data.save_json(facts, output_file)
    return fact_files, all_facts



def test_global_rules(model, rules, facts, output_dir, track_dir, test_indices):
    all_track_files =[os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.endswith("_gt.json")]
    test_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in test_indices]

    # _tracks_to_atoms(track_dir, test_ids, language_model, output_dir)
    # fact_files, facts = _atoms_to_facts(test_ids, language_model, output_dir)
    
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
    return Counter({int(key): value for key, value in supports.items()})



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

def _facts_to_rules(facts, train_ids, lang, output_dir):
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

    for vid in tqdm(train_ids, desc="Facts to Rules"):
        output_file = Path(rules_dir) / f"{vid}_rules.json"
        if output_file.exists():
            rules, rule_supports, head_supports = load_rule_files(output_file)
            all_rules.extend(rules)
            all_rule_supports = merge_rule_supports(all_rule_supports, rule_supports)
            all_head_supports = merge_head_supports(all_head_supports, head_supports)
        else:
            fact_data = utils_data.load_json(Path(facts_dir) / f"{vid}_facts.json")
            r_0, rule_supports, head_supports = lang.facts2rules(fact_data)
            save_rule_files(r_0, rule_supports, head_supports, output_file)

            all_rules.extend(r_0)
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
    print("\n------- Step 03 -------\n")
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
    _tracks_to_atoms(track_dir, train_ids, language_model, output_dir)
    fact_files, train_facts = _atoms_to_facts(train_ids, language_model, output_dir)
    all_rules, all_rule_supports, all_head_supports = _facts_to_rules(train_facts,train_ids, language_model, output_dir)

    if input_data["skip_lr"]:
        return

    
    train_dataset = build_rule_learning_dataset(train_facts, all_rules, output_dir, all_rule_supports, all_head_supports)

    # val data
    val_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in val_indices]
    _tracks_to_atoms(track_dir, val_ids, language_model, output_dir)
    _, val_facts = _atoms_to_facts(val_ids, language_model, output_dir)
    val_dataset = build_rule_learning_dataset(val_facts, all_rules, output_dir, all_rule_supports, all_head_supports)

    # test data
    test_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in test_indices]
    _tracks_to_atoms(track_dir, test_ids, language_model, output_dir)
    test_fact_files, test_facts = _atoms_to_facts(test_ids, language_model, output_dir)

    
    # learn rule aggregation
    ranked_rules, model = learn_rule_aggregation(train_dataset,val_dataset)
    # test data
    dataset_summary = test_global_rules(model, ranked_rules, test_facts, test_output_dir, track_dir, test_indices)
    
    utils_data.save_json(dataset_summary, test_output_dir / "rule_aggregation_summary.json")
    
    visualize_rule_aggregation_results(dataset_path, dataset_summary, test_output_dir)
    
    print("\n--------- Step 03 Done ---------------\n")