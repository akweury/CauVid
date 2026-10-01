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



def atom_to_signature(atom):
    if "location_name" in atom:
        return ("pred", atom["pred"], "location_name", atom["location_name"], "agent_class", atom["agent_class"])
    elif "action_id" in atom:
        return ("pred", atom["pred"], "action_id", atom["action_id"], "agent_class", atom["agent_class"])
    return None

def body_to_signature(body):
    signature = []
    for atom in body:
        atom_signature = atom_to_signature(atom)
        if atom_signature:
            signature.append(list(atom_signature))
    signature = tuple(tuple(item) for item in signature)
    return signature


def add_labels_to_atom(atom, dataset_labels):
    if "action_id" in atom:
        atom["action_id_label"] = dataset_labels["action_labels"][atom["action_id"]]
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