import os 
from pathlib import Path
import random
from tqdm import tqdm
import json

from src.exp_roadpp import utils_data
from src.exp_roadpp.step_dataset_language import Language


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


def get_frame_pairs(interval_atoms, dataset_labels):
    frame_pairs = []
    atoms_per_start_frame = {}
    for atom in interval_atoms:
        if int(atom["start_frame"]) not in atoms_per_start_frame:
            atoms_per_start_frame[int(atom["start_frame"])] = []
        atoms_per_start_frame[int(atom["start_frame"])].append(atom)

    end_frame = max([int(atom["end_frame"]) for atom in interval_atoms]) if interval_atoms else 0
    # for each frame, we need a list to include the atoms that occur in that frame
    frame_data = {i: [] for i in range(end_frame)}
    for start_frame, atoms in atoms_per_start_frame.items():
        for atom in atoms:
            for frame_id in range(start_frame, int(atom["end_frame"])):
                frame_as = utils_data.atom_to_signature(atom, dataset_labels)
                frame_data[frame_id].append(frame_as)

    frame_pairs = [(frame_data[i], frame_data[i + 1]) for i in frame_data.keys() if i + 1 in frame_data]
    return frame_pairs



def _tracks_to_atoms(track_dir, video_ids, lang, atom_file, dataset_labels):
    video_atoms = utils_data.load_json(atom_file) if atom_file.exists() else {}

    missing_ids = [vid for vid in video_ids if vid not in video_atoms]
    
    if missing_ids:
        for vid in tqdm(missing_ids, desc="Tracks to Atoms"):
            track_file = Path(track_dir) / f"{vid}_gt.json"
            if not track_file.exists():
                continue

            track_data = utils_data.load_json(track_file)       

            agent_tubes = track_data["data"]["agent_tubes"]
            segments_by_ego_actions = track_data["data"]["av_action_tubes"]
            frames = track_data["data"]["frames"]

            interval_atoms = []
            interval_atoms.extend(lang.video2atoms("av", segments_by_ego_actions))
            interval_atoms.extend(lang.video2atoms("agents", agent_tubes, frames))


            interval_atoms = [utils_data.add_labels_to_atom(atom, dataset_labels) for atom in interval_atoms]
            per_frame_atoms = get_frame_pairs(interval_atoms, dataset_labels)
            video_atoms[vid] = {
                "interval_atoms": interval_atoms,
                "per_frame_atoms": per_frame_atoms
            }

        utils_data.save_json(video_atoms, atom_file)

    return video_atoms

def _compress_atom_file(atom_file, compressed_file):
    atoms = utils_data.load_json(atom_file)
    compressed_atoms = {}
    unique_atoms = []
    for vid, vid_atoms in tqdm(atoms.items(), desc="Compressing Atoms"):
        compressed_atoms[vid] = {}
        compressed_atoms[vid]["interval_atoms"] = vid_atoms["interval_atoms"]
        
        compressed_per_frame_atoms = []
        for current_frame, next_frame in vid_atoms["per_frame_atoms"]:
            cf_compressed = []
            nf_compressed = []
            for atom in current_frame:
                if tuple(atom[:-2]) not in unique_atoms:
                    unique_atoms.append(tuple(atom[:-2]))
                index = unique_atoms.index(tuple(atom[:-2]))
                cf_compressed.append((index,atom[-2],atom[-1]))
            for atom in next_frame:
                if tuple(atom[:-2]) not in unique_atoms:
                    unique_atoms.append(tuple(atom[:-2]))
                index = unique_atoms.index(tuple(atom[:-2]))
                nf_compressed.append((index,atom[-2],atom[-1]))
            compressed_per_frame_atoms.append((cf_compressed, nf_compressed))
        compressed_atoms[vid]["per_frame_atoms"] = compressed_per_frame_atoms
    compressed_atoms["unique_atoms"] = unique_atoms
    utils_data.save_json(compressed_atoms, compressed_file)


def prepare_dataset(language_model, cfg):
    data_num = cfg["data_num"]
    track_dir = cfg["track_dir"]
    output_dir = cfg["output_dir"]
    all_track_files =[os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.endswith("_gt.json")]
    if data_num != 'full':
        all_track_files = all_track_files[:int(data_num)]
    train_indices, val_indices, test_indices = _split_example_indices(len(all_track_files))

    atoms_folder= Path(output_dir) / "atoms" 
    os.makedirs(atoms_folder, exist_ok=True)
    # train data
    train_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in train_indices]
    train_file = atoms_folder / "train_atoms.json"
    if not os.path.exists(train_file):
        _tracks_to_atoms(track_dir, train_ids, language_model, train_file, cfg["dataset_labels"])
    else:
        print(f"[Train Atoms] {train_file} already exists.")
    train_compressed_file = atoms_folder / "train_atoms_compressed.json"
    if not os.path.exists(train_compressed_file):
        _compress_atom_file(train_file, train_compressed_file)
    else:
        print(f"[Train Atoms Compressed] {train_compressed_file} already exists.")

    # val data
    val_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in val_indices]
    val_file = atoms_folder / "val_atoms.json"
    if not os.path.exists(val_file):
        _tracks_to_atoms(track_dir, val_ids, language_model, val_file, cfg["dataset_labels"])
    else:
        print(f"[Val Atoms] {val_file} already exists.")
    val_compressed_file = atoms_folder / "val_atoms_compressed.json"
    if not os.path.exists(val_compressed_file):
        _compress_atom_file(val_file, val_compressed_file)
    else:
        print(f"[Val Atoms Compressed] {val_compressed_file} already exists.")

    # test data
    test_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in test_indices]
    test_file = atoms_folder / "test_atoms.json"
    if not os.path.exists(test_file):
        _tracks_to_atoms(track_dir, test_ids, language_model, test_file, cfg["dataset_labels"])
    else:
        print(f"[Test Atoms] {test_file} already exists.")
    test_compressed_file = atoms_folder / "test_atoms_compressed.json"
    if not os.path.exists(test_compressed_file):
        _compress_atom_file(test_file, test_compressed_file)
    else:
        print(f"[Test Atoms Compressed] {test_compressed_file} already exists.")




def main(input_data):
    print("\n--------- Step Dataset ----------------------\n")
    
    cfg = {
        "track_dir": input_data["dataset_path"] / "gt",
        "output_dir": input_data["output_dir"],
        "data_num": input_data["data_num"],
        "device": input_data["device"],
    }
    cfg["dataset_labels"] = json.load(open(input_data["dataset_path"] / "gt" / "label.json", 'r'))
    
    print(f"[Data Num] {cfg['data_num']}")

    # Prepare language model and beam search model
    language_model = Language(cfg["device"])

    # Prepare dataset by converting tracks to atoms for train, val, and test splits
    prepare_dataset(language_model, cfg)
    print("\n--------- Step Dataset Done------------------\n")