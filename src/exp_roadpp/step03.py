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
from src.exp_roadpp.step03_pruner import Pruner
from src.exp_roadpp.step03_visual import visualize_baseline_results, visual_bar
from src.exp_roadpp.step03_rule_aggregation import learn_rule_aggregation
from src.exp_roadpp.step03_pred_eval import get_video_data, predict_next_n_steps
from src.exp_roadpp.step03_action_predictor import ActionPredictor
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




def _atoms_to_init_clauses(all_atoms, lang, output_dir, split):

    facts_dir = Path(output_dir) / "clauses"
    os.makedirs(facts_dir, exist_ok=True)
    all_init_clauses_file = facts_dir / f"all_init_clauses_adjacent_transitions_{split}.json"
    if all_init_clauses_file.exists():
        return utils_data.load_json(all_init_clauses_file)
    
    all_init_clauses = lang.atoms2atom_clauses(all_atoms, head_ungrounded_atoms=[])
    # init_clauses = lang.atoms2clauses(all_atoms, head_ungrounded_atoms=[])
    utils_data.save_json(all_init_clauses, all_init_clauses_file)
    return all_init_clauses


def _coarse_prune(init_clauses, pruner):
    pruned_clauses = pruner.coarse_prune(init_clauses)
    # rank clauses with their support coverage ratio
    rank_sc_ratio = [(clause, utils_data.support_coverage_ratio(clause)) for clause in pruned_clauses.values()]
    rank_sc_ratio.sort(key=lambda x: x[1], reverse=True)
    return rank_sc_ratio



def group_clauses_by_body(clauses, dataset_labels):
    clauses_by_body = {}
    for c in clauses:
        clause = c[0]["clause"]
        if len(clause["body"]) != 1:
            raise ValueError("Body grouping currently supports single-body clauses only.")
        body_signature = utils_data.atom_to_signature(
            clause["body"][0], dataset_labels
        )[:-2]
        head_signature = utils_data.atom_to_signature(
            clause["head"], dataset_labels
        )[:-2]
        body_group = clauses_by_body.setdefault(
            body_signature, {"candidate_heads": set(), "clauses": []}
        )
        body_group["candidate_heads"].add(head_signature)
        body_group["clauses"].append(c)

    for body_group in clauses_by_body.values():
        body_group["candidate_heads"] = tuple(
            sorted(body_group["candidate_heads"], key=repr)
        )

    print(f"[Grouping] Clauses grouped by body: {len(clauses)} -> {len(clauses_by_body)}")
    return clauses_by_body


def _build_body_conditioned_training_examples(
    atoms, clauses, dataset_labels, output_dir
):
    """Load or build sparse feature and target matrices for each body group."""
    body_groups = group_clauses_by_body(clauses, dataset_labels)
    cache_dir = Path(output_dir) / "training_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = cache_dir / "body_conditioned_examples_v2_train.json"
    features_file = cache_dir / "body_conditioned_features_v2_train.npz"
    targets_file = cache_dir / "body_conditioned_targets_v2_train.npz"
    group_ids_file = cache_dir / "body_conditioned_group_ids_v2_train.npz"

    body_signatures = tuple(body_groups)
    body_ids = {
        body_signature: index
        for index, body_signature in enumerate(body_signatures)
    }
    metadata_groups = [
        {
            "body_signature": list(body_signature),
            "candidate_heads": [
                list(head_signature)
                for head_signature in body_groups[body_signature]["candidate_heads"]
            ],
        }
        for body_signature in body_signatures
    ]

    if all(path.exists() for path in (
        metadata_file, features_file, targets_file, group_ids_file
    )):
        print(f"[Training data] Loading sparse training cache from {metadata_file}")
        cache_metadata = utils_data.load_json(metadata_file)
        if cache_metadata.get("schema_version") != 2:
            raise ValueError("Unsupported sparse training cache version.")
        cached_groups = cache_metadata["groups"]
        if cached_groups != metadata_groups:
            raise ValueError(
                "Cached body groups or candidate heads do not match current rules."
            )
        features = sparse.load_npz(features_file).tocsr()
        targets = sparse.load_npz(targets_file).tocsr()
        with np.load(group_ids_file) as group_data:
            group_ids = group_data["group_ids"]
        if features.shape[0] != targets.shape[0] or features.shape[0] != len(group_ids):
            raise ValueError("Cached training matrices have inconsistent row counts.")
        return {
            "body_groups": body_groups,
            "features": features,
            "targets": targets,
            "group_ids": group_ids,
            "feature_vocab": tuple(
                (int(item["group_id"]), item["scope"], tuple(item["signature"]))
                for item in cache_metadata["feature_vocab"]
            ),
            "head_vocab": tuple(
                (int(item["group_id"]), tuple(item["signature"]))
                for item in cache_metadata["head_vocab"]
            ),
        }

    feature_ids = {}
    head_ids = {}
    feature_vocab = []
    head_vocab = []
    feature_rows = []
    feature_columns = []
    target_rows = []
    target_columns = []
    group_ids = []
    row_count = 0

    for body_signature, body_group in body_groups.items():
        group_id = body_ids[body_signature]
        head_ids[body_signature] = {}
        for head_signature in body_group["candidate_heads"]:
            head_ids[body_signature][head_signature] = len(head_vocab)
            head_vocab.append((group_id, head_signature))

    for video_id, video_atoms in tqdm(atoms.items(), desc="Processing videos"):
        video_data = get_video_data(video_id, video_atoms, dataset_labels)
        frame_data = video_data["frame_data"]

        for frame_id, current_atoms in frame_data.items():
            # Skip the last frame since it has no next frame to predict.
            next_atoms = frame_data.get(frame_id + 1)
            if next_atoms is None:
                continue
            # Organize next frame atoms by their entity key for efficient lookup.
            next_atoms_by_entity = {}
            for atom_signature in next_atoms:
                entity_key = _atom_entity_key(atom_signature)
                if entity_key is not None:
                    next_atoms_by_entity.setdefault(entity_key, set()).add(
                        atom_signature[:-2]
                    )

            for body_atom in current_atoms:
                body_signature = body_atom[:-2]
                if body_signature not in body_groups:
                    continue
                entity_key = _atom_entity_key(body_atom)
                next_entity_facts = next_atoms_by_entity.get(entity_key)
                if not next_entity_facts:
                    continue

                group_id = body_ids[body_signature]
                active_features = set()
                for atom in current_atoms:
                    atom_fact = atom[:-2]
                    if atom_fact == body_signature:
                        continue
                    scope = (
                        "entity"
                        if _atom_entity_key(atom) == entity_key
                        else "scene"
                    )
                    feature_key = (group_id, scope, atom_fact)
                    feature_id = feature_ids.get(feature_key)
                    if feature_id is None:
                        feature_id = len(feature_vocab)
                        feature_ids[feature_key] = feature_id
                        feature_vocab.append(feature_key)
                    active_features.add(feature_id)

                for feature_id in active_features:
                    feature_rows.append(row_count)
                    feature_columns.append(feature_id)

                for head_signature, target_id in head_ids[body_signature].items():
                    if head_signature in next_entity_facts:
                        target_rows.append(row_count)
                        target_columns.append(target_id)

                group_ids.append(group_id)
                row_count += 1

    features = sparse.csr_matrix(
        (
            np.ones(len(feature_rows), dtype=np.uint8),
            (feature_rows, feature_columns),
        ),
        shape=(row_count, len(feature_vocab)),
        dtype=np.uint8,
    )
    targets = sparse.csr_matrix(
        (
            np.ones(len(target_rows), dtype=np.uint8),
            (target_rows, target_columns),
        ),
        shape=(row_count, len(head_vocab)),
        dtype=np.uint8,
    )
    group_ids_array = np.asarray(group_ids, dtype=np.int32)

    utils_data.save_json(
        {
            "schema_version": 2,
            "groups": metadata_groups,
            "feature_vocab": [
                {
                    "group_id": group_id,
                    "scope": scope,
                    "signature": list(signature),
                }
                for group_id, scope, signature in feature_vocab
            ],
            "head_vocab": [
                {
                    "group_id": group_id,
                    "signature": list(signature),
                }
                for group_id, signature in head_vocab
            ],
        },
        metadata_file,
    )
    sparse.save_npz(features_file, features, compressed=True)
    sparse.save_npz(targets_file, targets, compressed=True)
    np.savez_compressed(group_ids_file, group_ids=group_ids_array)

    print(
        f"[Training data] Prepared sparse matrices: "
        f"{features.shape[0]} examples, {features.shape[1]} features, "
        f"{targets.shape[1]} candidate heads"
    )
    print(f"[Training data] Cached sparse examples under {cache_dir}")
    return {
        "body_groups": body_groups,
        "features": features,
        "targets": targets,
        "group_ids": group_ids_array,
        "feature_vocab": tuple(feature_vocab),
        "head_vocab": tuple(head_vocab),
    }


def _atom_entity_key(atom_signature):
    tube_uid = atom_signature[-1]
    agent_class = atom_signature[7]
    if tube_uid is not None:
        return ("tube_uid", tube_uid)
    if agent_class == "av":
        return ("agent_class", "av")
    return None



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

def prepare_dataset(data_num, track_dir, language_model, output_dir):
    all_track_files =[os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.endswith("_gt.json")]
    if data_num != 'full':
        all_track_files = all_track_files[:int(data_num)]
    train_indices, val_indices, test_indices = _split_example_indices(len(all_track_files))
    # train data
    train_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in train_indices]
    atoms_by_train_video = _tracks_to_atoms(track_dir, train_ids, language_model, output_dir)
    # val data
    val_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in val_indices]
    atoms_by_val_video = _tracks_to_atoms(track_dir, val_ids, language_model, output_dir)
    # test data
    test_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in test_indices]
    atoms_by_test_video = _tracks_to_atoms(track_dir, test_ids, language_model, output_dir)   

    return {"train": atoms_by_train_video, "val": atoms_by_val_video, "test": atoms_by_test_video}


def main(input_data):
    print("\n--------- Step 03 ----------------------\n")

    # Prepare output directories and dataset paths
    output_dir = input_data["output_dir"]
    test_output_dir = output_dir / "test"
    os.makedirs(test_output_dir, exist_ok=True)
    track_dir = input_data["dataset_path"] / "gt"
    dataset_path = input_data["dataset_path"]
    device = input_data["device"]
    dataset_labels = input_data["dataset_labels"]
    # Prepare language model and beam search model
    language_model = Language(input_data["device"])
    beam_search_model = BeamSearch()
    pruner = Pruner(input_data["prune_args"], dataset_labels)
    
    # Prepare dataset by converting tracks to atoms for train, val, and test splits
    atoms = prepare_dataset(input_data['data_num'], track_dir, language_model, output_dir)

    # C_0: Convert atoms to initial clauses
    init_clauses = _atoms_to_init_clauses(atoms["train"], language_model, output_dir, 'train')
    visual_bar(init_clauses, output_dir, "init_clauses_distribution")
    # C_1: Coarse Prune, filter out the low frequent clauses
    coarse_pruned_clauses = _coarse_prune(init_clauses, pruner)
    # replace id with label names
    coarse_pruned_clauses = utils_data.replace_ids_with_labels(coarse_pruned_clauses, dataset_labels)
    # predict and evaluation
    model = ActionPredictor(coarse_pruned_clauses, dataset_labels)
    # C_2
    training_examples_by_body = _build_body_conditioned_training_examples(
        atoms["train"], coarse_pruned_clauses, dataset_labels, output_dir
    )

    # predict the next action/position of each atom
    predict_next_n_steps(model, atoms["val"], output_dir, dataset_labels, step=1)    

    


    if input_data["learning"] == 'True':
        learn_rule_aggregation(atoms, init_clauses, output_dir, dataset_path, "vanilla")
        learn_rule_aggregation(atoms, coarse_pruned_clauses, output_dir,dataset_path, "coarse_prune")
    if input_data["baselines"] == 'True':
        # train and test baselines
        result_summary = {}
        
        transformer_baseline.run(atoms, dataset_labels, output_dir, result_summary, device)
        ilp_baseline.run(atoms, dataset_labels, output_dir, result_summary, device)
        gnn_baseline.run(atoms, dataset_labels, output_dir, result_summary, device)
        lstm_baseline.run(atoms, dataset_labels, output_dir, result_summary, device)
        utils_data.save_json(result_summary, test_output_dir / "baselines_summary.json")
        visualize_baseline_results(result_summary, test_output_dir)
    
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
    train_facts = _atoms_to_init_clauses(language_model, output_dir, 'train')
    # all_rules, all_rule_supports, all_head_supports = _facts_to_rules(train_facts, train_ids, language_model, output_dir)
    # val data
    val_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in val_indices]
    # test data
    test_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in test_indices]
    _tracks_to_atoms(track_dir, test_ids, language_model, output_dir)
    

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
