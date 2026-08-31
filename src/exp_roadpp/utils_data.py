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