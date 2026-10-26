import bisect
import json
import numpy as np
from pathlib import Path

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_atoms(compressed_file):
    """Load a compressed atom file and attach the shared vocabulary saved next to it
    (unique_atoms.json) under the "unique_atoms" key."""
    compressed_file = Path(compressed_file)
    atoms = load_json(compressed_file)
    atoms["unique_atoms"] = load_json(compressed_file.parent / "unique_atoms.json")
    return atoms


def load_npz_dict(npz_path):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}

def load_json_list(json_path):
    json_path = Path(json_path)
    if not json_path.exists():
        raise ValueError(f"JSON file does not exist: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_agent_frame_action_pairs(agent_tubes, frames):
    """
    Returns:
      {
        agent_tube_id: [
          {"frame": 12, "action_ids": [3, 7]},
          {"frame": 13, "action_ids": [7]},
          ...
        ],
        ...
      }
    """
    result = {}

    for agent_tube_id, agent_tube in agent_tubes.items():
        annos_map = agent_tube.get("annos", {})  # frame_id -> b_id
        pairs = []

        for frame_id_str, b_id in annos_map.items():
            frame_obj = frames.get(str(frame_id_str), {})
            anno_obj = frame_obj.get("annos", {}).get(b_id, {})
            action_ids = anno_obj.get("action_ids", [])
            pairs.append({
                "frame": int(frame_id_str),
                "action_ids": action_ids
            })

        pairs.sort(key=lambda x: x["frame"])
        result[agent_tube_id] = pairs

    return result



def build_agent_frame_action_loc_pairs(agent_tubes, frames):
    result = {}
    for agent_tube_id, agent_tube in agent_tubes.items():
        pairs = []
        for frame_id_str, b_id in agent_tube.get("annos", {}).items():
            anno = frames.get(str(frame_id_str), {}).get("annos", {}).get(b_id, {})
            pairs.append({
                "frame": int(frame_id_str),
                "action_ids": anno.get("action_ids", []),
                "loc_ids": anno.get("loc_ids", [])
            })
        pairs.sort(key=lambda x: x["frame"])
        result[agent_tube_id] = pairs
    return result

def get_start_end_frame(segment, frames):
    seg_frames = sorted(segment["annos"].keys(), key=int)

    tube_uid = None
    for frame_id, box_id in segment["annos"].items():
        box = (frames or {}).get(str(frame_id), {}).get("annos", {}).get(box_id, {})
        if box.get("tube_uid"):
            tube_uid = box["tube_uid"]
            break 
    start_frame = seg_frames[0]
    end_frame = seg_frames[-1]
    return start_frame, end_frame, tube_uid


def support_coverage_ratio(clause):
    if not clause:
        return 0.0
    sc_ratios = [sc["support"] / sc["coverage"] for sc in clause['support_coverage'].values()]
    return sum(sc_ratios) / len(sc_ratios) if sc_ratios else 0.0


def build_signature(predicate, label, label_id, agent_class, agent_class_label, tube_uid):
    if predicate == "action":
        return (
            "pred", predicate,
            "action_id", label_id,
            "action_id_label", label,
            "agent_class", agent_class,
            "agent_class_label", agent_class_label,
            "tube_uid", tube_uid
        )
    elif predicate == "location":
        return (
            "pred", predicate,
            "location_name", label_id,
            "location_name_label", label,
            "agent_class", agent_class,
            "agent_class_label", agent_class_label,
            "tube_uid", tube_uid
        )
    else:
        raise ValueError("Unknown predicate type in build_signature")


def atom_to_signature(atom, dataset_labels=None):
    # the atom is a dict, if it is a tuple already return it as is
    if isinstance(atom, tuple):
        return atom
    if dataset_labels is not None:
        atom = add_labels_to_atom(atom, dataset_labels)
        
    if "location_name" in atom:
        return ("pred", atom["pred"], 
                "location_name", atom["location_name"], 
                "location_name_label", atom["location_name_label"],
                "agent_class", atom["agent_class"], 
                "agent_class_label", atom["agent_class_label"],
                "tube_uid", atom.get("tube_uid"))
    elif "action_id" in atom:
        return ("pred", atom["pred"], 
                "action_id", atom["action_id"], 
                "action_id_label", atom["action_id_label"],
                "agent_class", atom["agent_class"], 
                "agent_class_label", atom["agent_class_label"],
                "tube_uid", atom.get("tube_uid"))
    else:
        raise ValueError("Unknown atom type in atom_to_signature")

def build_onset_index(interval_atoms, atom_vocab, dataset_labels=None):
    """Sorted [(start_frame, atom_idx)] of a video's interval atoms.
    atom_vocab maps an atom signature without tube_uid (sig[:-2]) to its index in unique_atoms,
    i.e. the same key used in the compressed atom file. Atoms not in the vocab are skipped."""
    onsets = []
    for atom in interval_atoms:
        if int(atom["end_frame"]) <= int(atom["start_frame"]):
            continue
        key = tuple(atom_to_signature(atom, dataset_labels)[:-2])
        if key in atom_vocab:
            onsets.append((int(atom["start_frame"]), atom_vocab[key]))
    onsets.sort()
    return onsets


def build_entity_onset_index(interval_atoms, atom_vocab, dataset_labels=None):
    """{tube_uid: sorted [(start_frame, atom_idx)]}, as `build_onset_index` but per entity
    (tube_uid is None for the ego vehicle)."""
    onsets = {}
    for atom in interval_atoms:
        if int(atom["end_frame"]) <= int(atom["start_frame"]):
            continue
        signature = atom_to_signature(atom, dataset_labels)
        key = tuple(signature[:-2])
        if key in atom_vocab:
            onsets.setdefault(signature[-1], []).append((int(atom["start_frame"]), atom_vocab[key]))
    for entity_onsets in onsets.values():
        entity_onsets.sort()
    return onsets


def future_onsets(onsets, t, horizon):
    """[(atom_idx, delta_t)] for the first onset of each atom index within frames (t, t + horizon]."""
    first = {}
    for start_frame, atom_idx in onsets[bisect.bisect_right(onsets, (t, float("inf"))):]:
        if start_frame > t + horizon:
            break
        first.setdefault(atom_idx, start_frame - t)
    return list(first.items())


def same_signature_no_tube_uid(sig1, sig2):
    if not sig1 or not sig2:
        return False
    return sig1[:-2] == sig2[:-2]  # Compare all elements except the last two ("tube_uid")

def body_to_signature(body):
    signature = []
    for atom in body:
        atom_signature = atom_to_signature(atom)
        if atom_signature:
            signature.append(list(atom_signature))
    signature = tuple(tuple(item) for item in signature)
    return signature


def add_labels_to_atom(atom, dataset_labels):
    if "action_id" in atom and atom["agent_class"] != "av":
        atom["action_id_label"] = dataset_labels["action_labels"][atom["action_id"]]
    if "action_id" in atom and atom["agent_class"] == "av":
        atom["action_id_label"] = dataset_labels["all_av_action_labels"][atom["action_id"]]

    if "location_name" in atom:
        atom["location_name_label"] = dataset_labels["loc_labels"][atom["location_name"]]
    if "agent_class" in atom and atom["agent_class"] != "av":
        atom["agent_class_label"] = dataset_labels["agent_labels"][atom["agent_class"]]
    if "agent_class" in atom and atom["agent_class"] == "av":
        atom["agent_class_label"] = atom["agent_class"]
    return atom 


def replace_ids_with_labels(clauses, dataset_labels):
    for clause,score in clauses:
        add_labels_to_atom(clause['clause']["head"], dataset_labels)
        for atom in clause['clause']["body"]:
            add_labels_to_atom(atom, dataset_labels)
    return clauses
