import os 
from pathlib import Path
import random
from tqdm import tqdm
import json
import numpy as np
import torch
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


def _onset_statistics(compressed_file, stats_dir, horizon):
    """Histogram of delta_t (frames until an atom's next onset) over all anchor frames, per atom."""
    stats_file = stats_dir / "onset_stats.json"
    if stats_file.exists():
        print(f"[Onset Stats] {stats_file} already exists.")
        return
    os.makedirs(stats_dir, exist_ok=True)

    atoms = utils_data.load_json(compressed_file)
    unique_atoms = atoms["unique_atoms"]
    atom_vocab = {tuple(atom): i for i, atom in enumerate(unique_atoms)}
    # counts[i, dt - 1]: number of anchor frames whose next onset of atom i is dt frames ahead
    counts = np.zeros((len(unique_atoms), horizon), dtype=np.int64)
    num_anchor_frames = 0
    for vid, vid_atoms in tqdm(atoms.items(), desc="Onset Stats"):
        if vid == "unique_atoms":
            continue
        onsets = utils_data.build_onset_index(vid_atoms["interval_atoms"], atom_vocab)
        num_frames = len(vid_atoms["per_frame_atoms"])
        for t in range(num_frames):
            for atom_idx, delta_t in utils_data.future_onsets(onsets, t, horizon):
                counts[atom_idx, delta_t - 1] += 1
        num_anchor_frames += num_frames

    per_atom = []
    for i, atom in enumerate(unique_atoms):
        total = int(counts[i].sum())
        per_atom.append({
            "atom_idx": i,
            "atom": atom,
            "num_onsets": total,
            "num_no_onset": num_anchor_frames - total,
            "mean_delta_t": float((counts[i] * np.arange(1, horizon + 1)).sum() / total) if total else None,
            "counts": counts[i].tolist(),
        })
    utils_data.save_json({
        "horizon": horizon,
        "num_anchor_frames": num_anchor_frames,
        "overall_counts": counts.sum(axis=0).tolist(),
        "per_atom": per_atom,
    }, stats_file)
    _plot_onset_statistics(counts, unique_atoms, stats_dir / "onset_hist.png")
    print(f"[Onset Stats] saved to {stats_file}")


def _plot_onset_statistics(counts, unique_atoms, plot_file, top_k=15):
    import matplotlib.pyplot as plt
    horizon = counts.shape[1]
    delta_ts = np.arange(1, horizon + 1)
    top = np.argsort(-counts.sum(axis=1))[:top_k]
    num_cols = 4
    num_rows = 1 + int(np.ceil(len(top) / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    overall = fig.add_subplot(num_rows, 1, 1)
    overall.bar(delta_ts, counts.sum(axis=0), width=1.0)
    overall.set_title("All atoms")
    overall.set_xlabel("delta_t (frames)")
    overall.set_ylabel("count")

    for ax, atom_idx in zip(axes.flat[num_cols:], top):
        atom = dict(zip(unique_atoms[atom_idx][::2], unique_atoms[atom_idx][1::2]))
        label = atom.get("action_id_label", atom.get("location_name_label"))
        ax.axis("on")
        ax.bar(delta_ts, counts[atom_idx], width=1.0)
        ax.set_title(f"[{atom_idx}] {atom['agent_class_label']} {label}", fontsize=9)
        ax.set_xlabel("delta_t (frames)", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)

def prepare_per_frame_events(cfg):

    def _prepare_per_split(split_name):
        output_file = cfg["output_dir"] / f"{split_name}_events.pt"
        if output_file.exists():
            return
        
        atoms = utils_data.load_json(cfg["output_dir"] / "atoms" /f"{split_name}_atoms_compressed.json")
        interval_atoms = [data["interval_atoms"] for data in atoms.values() if isinstance(data, dict) and "interval_atoms" in data]
        events = []
        count_downs = [] 
        unique_atoms = [tuple(atom) for atom in atoms["unique_atoms"]]

        for video_intervals in tqdm(interval_atoms, desc="Events Preparing"):
            min_start = min(int(interval["start_frame"]) for interval in video_intervals) if video_intervals else None
            max_end = max(int(interval["end_frame"]) for interval in video_intervals) if video_intervals else None
            video_event_matrix = torch.zeros((max_end, len(unique_atoms)),dtype = torch.bool)
            for interval in video_intervals:
                start, end = int(interval["start_frame"]), int(interval["end_frame"])
                atom = utils_data.atom_to_signature(interval, cfg["dataset_labels"])
                atom_idx = unique_atoms.index(atom[:-2])
                video_event_matrix[start:end, atom_idx] = True
            events.append(video_event_matrix)
            video_count_down_matrix = torch.zeros_like(video_event_matrix, dtype=torch.int) - 1
            for frame_idx in range(max_end-1):
                for atom_idx in range(len(unique_atoms)):
                    # the frame left until the sign change for this atom
                    counts = (video_event_matrix[frame_idx:, atom_idx] != video_event_matrix[frame_idx, atom_idx]).nonzero(as_tuple=True)[0]
                    if len(counts)>0:
                        video_count_down_matrix[frame_idx, atom_idx] = counts[0]
            count_downs.append(video_count_down_matrix)
            # Ensure the last frame has a countdown of 0 for all atoms
            video_count_down_matrix[max_end-1, :] = 0
        torch.save({
            "events": events,
            "count_downs": count_downs
        }, output_file)

    _prepare_per_split("train")
    _prepare_per_split("val")
    _prepare_per_split("test")
    
    return
            
    




def prepare_per_frame_atoms(language_model, cfg):
    data_num = cfg["data_num"]
    track_dir = cfg["track_dir"]
    output_dir = cfg["output_dir"]
    all_track_files =[os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.endswith("_gt.json")]
    if data_num != 'full':
        all_track_files = all_track_files[:int(data_num)]

    train_indices, val_indices, test_indices = _split_example_indices(len(all_track_files))
    atoms_folder= Path(output_dir) / "atoms" 
    os.makedirs(atoms_folder, exist_ok=True)

    def _prepare_split(split_name, indices):
        # train data
        split_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in indices]
        split_file = atoms_folder / f"{split_name}_atoms.json"
        if not os.path.exists(split_file):
            _tracks_to_atoms(track_dir, split_ids, language_model, split_file, cfg["dataset_labels"])
        else:
            print(f"[Per Frame Atoms] {split_file} already exists.")
        compressed_file = atoms_folder / f"{split_name}_atoms_compressed.json"
        if not os.path.exists(compressed_file):
            _compress_atom_file(split_file, compressed_file)
        else:
            print(f"[Per Frame Atoms Compressed] {compressed_file} already exists.")

    _prepare_split("train", train_indices)
    _prepare_split("val", val_indices)
    _prepare_split("test", test_indices)




def main(input_data):
    print("\n--------- Step Dataset ----------------------\n")
    
    cfg = {
        "track_dir": input_data["dataset_path"] / "gt",
        "output_dir": input_data["output_dir"],
        "data_num": input_data["data_num"],
        "device": input_data["device"],
        "onset_horizon": input_data["onset_horizon"],
        "dataset_labels": json.load(open(input_data["dataset_path"] / "gt" / "label.json", 'r'))
    }
    print(f"[Data Num] {cfg['data_num']}")

    # Prepare language model and beam search model
    language_model = Language(cfg["device"])
    # Prepare dataset by converting tracks to atoms for train, val, and test splits
    prepare_per_frame_atoms(language_model, cfg)
    prepare_per_frame_events(cfg)
    print("\n--------- Step Dataset Done------------------\n")