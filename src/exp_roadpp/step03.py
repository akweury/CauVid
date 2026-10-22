from pathlib import Path
import json 
from tqdm import tqdm
import os
import time

from collections import Counter
from collections import defaultdict

from src.exp_roadpp.logic import Clause
from src.exp_roadpp import utils_data
from src.exp_roadpp.step_dataset_language import Language
from src.exp_roadpp.step03_visual import visual_bar




def atoms2atom_clauses(self, atoms_by_videos, head_ungrounded_atoms):
    clauses = {}

    body_opportunities_by_video = defaultdict(Counter)
    rule_support_by_video = defaultdict(Counter)

    for video_id, atoms in tqdm(atoms_by_videos.items()):
        atoms_by_time = defaultdict(lambda: defaultdict(set))
        for atom in atoms:
            start_frame = int(atom["start_frame"])
            end_frame = int(atom["end_frame"])
            tube_uid = atom.get("tube_uid")
            agent_class = atom.get("agent_class")

            if tube_uid is not None:
                tube_key = ("tube_uid", tube_uid)
            elif agent_class == "av":
                tube_key = ("agent_class", "av")
            else:
                continue

            ungrounded_atom = self._to_ungrounded_atom(atom)
            for frame_id in range(start_frame, end_frame):
                atoms_by_time[frame_id][tube_key].add(ungrounded_atom)

        for frame_id, current_by_tube in atoms_by_time.items():
            next_by_tube = atoms_by_time.get(frame_id + 1)
            if not next_by_tube:
                continue

            for tube_key, current_atoms in current_by_tube.items():
                next_atoms = next_by_tube.get(tube_key)
                if not next_atoms:
                    continue

                for body_atom in current_atoms:
                    body_opportunities_by_video[video_id][body_atom] += 1
                    for head_atom in next_atoms:
                        if body_atom == head_atom:
                            continue

                        clause = Clause(head_atom, [body_atom])
                        clauses.setdefault(clause, {})
                        rule_support_by_video[video_id][clause] += 1

    for clause in clauses:
        body_atom = clause.body[0]
        for video_id, body_opportunities in body_opportunities_by_video.items():
            coverage = body_opportunities[body_atom]
            if coverage:
                clauses[clause][video_id] = {
                    "support": rule_support_by_video[video_id][clause],
                    "coverage": coverage,
                }

    _clauses = {}
    clause_id = 0
    for clause, c_dict in clauses.items():
        _clauses[clause_id] = {
            "clause": clause.to_dict(),
            "support_coverage": c_dict
        }
        clause_id += 1

    
    return _clauses
















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




def load_atoms(atom_file):
    if not atom_file.exists():
        raise FileNotFoundError(f"Atom file {atom_file} does not exist.")
    atoms = utils_data.load_json(atom_file)
    print(f"[Load Atoms] Loaded atoms from {len(atoms)} videos from {atom_file}")
    return atoms



def build_anchor_clauses(all_atoms, output_dir, split):
    anchor_clause_file = output_dir / f"anchor_clauses_{split}.json"
    if anchor_clause_file.exists():
        return utils_data.load_json(anchor_clause_file)
    
    anchor_clauses = []
    seen_clauses = set()
    unique_atoms = all_atoms["unique_atoms"]
    for vid, atoms in all_atoms.items():
        if vid == "unique_atoms":
            continue
        per_frame_atoms = atoms["per_frame_atoms"]
        for current_frame, next_frame in tqdm(per_frame_atoms):
            if not (len(current_frame) > 0 and len(next_frame) > 0):
                continue
            current_signatures = list(set([tuple(atom) for atom in current_frame]))
            next_signatures = list(set([tuple(atom) for atom in next_frame]))            
            for body in current_signatures:
                for head in next_signatures:
                    if body[-1] != head[-1]:
                        clause = Clause(head, [body], [1])
                        if clause not in seen_clauses:
                            seen_clauses.add(clause)
                            anchor_clauses.append(clause)
    
    # add self-loop clauses for each unique atom
    for i in range(len(unique_atoms)):
        anchor_clauses.append(
            Clause([i], [[i]], [0])
        )
    data = {
        "anchor_clauses": anchor_clauses,
        "unique_atoms": unique_atoms
    }
    utils_data.save_json(data, anchor_clause_file)
    return data



def main(input_data):
    print("\n--------- Step 03 ----------------------\n")
    cfg_legacy = {
        "device": input_data["device"], 
        "output_dir": input_data["output_dir"],
        "test_output_dir": input_data["output_dir"] / "test",
        "track_dir": input_data["dataset_path"] / "gt",
        "dataset_path": input_data["dataset_path"],
        "cd_method": input_data["cd_method"],
        "cd_stat": input_data["cd_stat"],
        "cd_alpha": input_data["cd_alpha"],
        "cd_depth": input_data["cd_depth"],
        "graph_strategy": input_data["graph_strategy"],
        "min_support_features": input_data["min_support_features"],
        "exp": input_data["exp"],
    }
    
    cfg = {
        "train_atom_file": input_data["train_atom_file"],
        "output_dir": input_data["output_dir"]
        
    }
    anchor_clause_file = cfg["output_dir"] / "anchor_clauses_train.json"

    if anchor_clause_file.exists():
        return utils_data.load_json(anchor_clause_file)

    # Prepare dataset by converting tracks to atoms for train, val, and test splits
    atoms = load_atoms(cfg["train_atom_file"])
    # Convert atoms to anchor clauses
    anchor_data = build_anchor_clauses(atoms, cfg["output_dir"], 'train')
    raise NotImplementedError("Step 3 output handling not implemented yet.")    
    return anchor_data

