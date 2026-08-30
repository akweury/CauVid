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