from pathlib import Path

from tqdm import tqdm
import os 
import numpy as np
from scipy import sparse

from src.exp_roadpp import utils_data
from src.exp_roadpp.step03_language import Language
from src.exp_roadpp.step03_beam_search import BeamSearch
from src.exp_roadpp.step03_visual import visualize_rule_aggregation_results
from src.exp_roadpp.step03_rule_aggregation import learn_rule_aggregation, build_rule_learning_dataset, test_global_rules


def _tracks_to_atoms(track_dir,train_ids, lang, output_dir):
    atom_files = []
    for vid in tqdm(train_ids, desc="Tracks to Atoms"):
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


def _atoms_to_facts(atom_files, lang, output_dir):
    fact_files = []
    all_facts = []
    for atom_file in tqdm(atom_files, desc="Atoms to Facts"):
        atom_data = utils_data.load_json(atom_file)
        vid = Path(atom_file).stem.replace("_atoms", "")
        output_file = Path(output_dir) / f"{vid}_facts.json"
        fact_files.append(output_file)

        if output_file.exists():
            facts = utils_data.load_json(output_file)
            all_facts.extend(facts)
            continue
        facts = lang.atoms2facts(atom_data)
        all_facts.extend(facts)
        utils_data.save_json(facts, output_file)
    return fact_files, all_facts

def _rules_to_global(all_rules, all_facts, lang_model, output_dir, test_output_dir):
    dataset = build_rule_learning_dataset(all_facts, all_rules, output_dir)
    ranked_rules, model = learn_rule_aggregation(dataset)
    dataset_summary = test_global_rules(model, ranked_rules, lang_model, dataset, test_output_dir)
    utils_data.save_json(dataset_summary, test_output_dir / "rule_aggregation_summary.json")
    return ranked_rules, model, dataset_summary

def _facts_to_rules(fact_files, lang, output_dir):
    all_rules = []
    all_rule_file =Path(output_dir) / f"all_rules.json"
    if all_rule_file.exists():
        all_rules = utils_data.load_json(all_rule_file)
        return all_rules 
    for fact_file in tqdm(fact_files, desc="Facts to Rules"):
        fact_data = utils_data.load_json(fact_file)
        vid = Path(fact_file).stem.replace("_facts", "")
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
    train_ids = input_data["train_ids"]
    track_dir = input_data["dataset_path"] / "gt"
    
    language_model = Language()
    beam_search_model = BeamSearch()

    train_atom_files = _tracks_to_atoms(track_dir, train_ids, language_model, output_dir)
    train_fact_files, all_facts = _atoms_to_facts(train_atom_files, language_model, output_dir)
    all_rules = _facts_to_rules(train_fact_files, language_model, output_dir)
    ranked_rules, rule_model, dataset_summary = _rules_to_global(all_rules, all_facts, language_model, output_dir, test_output_dir)

    visualize_rule_aggregation_results(dataset_summary, test_output_dir)
    
    print("\n--------- Step 03 Done ---------------\n")