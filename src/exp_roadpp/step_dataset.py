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


# -------------- Stratified split --------------
# Splits by video (the atoms of one video never cross splits), but so that the onsets of every head
# fall into train / val / test in the split fractions as far as possible: iterative stratification
# (Sechidis et al. 2011, rarest head first), with the onset counts as weights. A plain random split of
# videos gives the rare heads 0 or 1 test events by chance.
SPLIT_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}
STRATIFIED_VERSION = "strat_v1"
MIN_OFF_FRAMES = 10                 # as step 05: an atom back on after a shorter gap is no new onset


def head_onsets(interval_atoms):
    """{head "<class>:<label>": onsets} of one video's interval atoms: an interval of an agent's atom that
    starts after the agent is seen (not its first frame: entering the scene is no event) and at least
    MIN_OFF_FRAMES after the previous interval of the same atom ended."""
    from collections import Counter, defaultdict
    first = {}
    by_atom = defaultdict(list)
    for atom in interval_atoms:
        tube = atom.get("tube_uid") or "ego"
        start, end = int(atom["start_frame"]), int(atom["end_frame"])
        first[tube] = min(first.get(tube, start), start)
        label = atom.get("action_id_label") or atom.get("location_name_label")
        by_atom[(tube, str(atom.get("agent_class_label")) + ":" + str(label))].append((start, end))
    counts = Counter()
    for (tube, head), spans in by_atom.items():
        last_end = None
        for start, end in sorted(spans):
            if start > first[tube] and (last_end is None or start - last_end >= MIN_OFF_FRAMES):
                counts[head] += 1
            last_end = end if last_end is None else max(last_end, end)
    return counts


def stratified_split(video_counts, fractions=None, seed=7):
    """{split: [video ids]} by iterative stratification of the per-video head onset counts
    video_counts {video: Counter}: the rarest head with unassigned videos first, each of its videos
    (randomly ordered) to the split that still needs most of that head's onsets (ties: the split that
    needs most videos, then random); videos without onsets fill the splits up to their sizes."""
    fractions = fractions or SPLIT_FRACTIONS
    rng = random.Random(int(seed))
    videos = sorted(video_counts)
    total = {}
    for v in videos:
        for h, c in video_counts[v].items():
            total[h] = total.get(h, 0) + c
    need = {sp: {h: f * n for h, n in total.items()} for sp, f in fractions.items()}
    need_n = {sp: f * len(videos) for sp, f in fractions.items()}
    out = {sp: [] for sp in fractions}
    left = set(videos)

    def assign(v, sp):
        out[sp].append(v)
        left.discard(v)
        need_n[sp] -= 1
        for h, c in video_counts[v].items():
            need[sp][h] -= c
    while True:
        remaining = {}
        for v in left:
            for h, c in video_counts[v].items():
                remaining[h] = remaining.get(h, 0) + c
        if not remaining:
            break
        head = min(remaining, key=lambda h: (remaining[h], h))
        cands = sorted(v for v in left if video_counts[v].get(head))
        rng.shuffle(cands)
        for v in cands:
            best = max(fractions, key=lambda sp: (need[sp][head], need_n[sp], rng.random()))
            assign(v, best)
    rest = sorted(left)
    rng.shuffle(rest)
    for v in rest:
        assign(v, max(fractions, key=lambda sp: (need_n[sp], rng.random())))
    # The thresholds are chosen on val (step 07): val never smaller than test (both have the same share).
    if "val" in out and "test" in out and len(out["val"]) < len(out["test"]):
        out["val"], out["test"] = out["test"], out["val"]
    return {sp: sorted(vs) for sp, vs in out.items()}


def split_report(video_counts, splits):
    """{head: {split: onsets}}."""
    report = {}
    for sp, vs in splits.items():
        for v in vs:
            for h, c in video_counts[v].items():
                report.setdefault(h, {s: 0 for s in splits})[sp] += c
    return report


def _stratified_split_files(track_dir, all_track_files, language_model, atoms_folder, dataset_labels,
                            min_height=0.0, split_from=None):
    """The split atom files of the stratified split: the atoms of every video (all_atoms.json), the head
    onsets per video, `stratified_split`, then each split's file cut out of all_atoms.json; split.json
    keeps the assignment and the onsets per head and split."""
    split_files = {sp: atoms_folder / f"{sp}_atoms.json" for sp in SPLIT_FRACTIONS}
    if all(f.exists() for f in split_files.values()) and (atoms_folder / "split.json").exists():
        print("[Split] stratified split files already exist.")
        return
    ids = [Path(f).stem.replace("_gt", "") for f in all_track_files]
    all_atoms = _tracks_to_atoms(track_dir, ids, language_model, atoms_folder / "all_atoms.json", dataset_labels,
                                 min_height)
    counts = {v: head_onsets(all_atoms[v]["interval_atoms"]) for v in ids if v in all_atoms}
    if split_from:
        # The video split of another exp (e.g. full_s for full_sn): the same videos per split, so the
        # results differ only by the dataset change.
        splits = {sp: [v for v in vs if v in all_atoms] for sp, vs in utils_data.load_json(split_from)["splits"].items()}
        print(f"[Split] the video split of {split_from}")
    else:
        splits = stratified_split(counts)
    for sp, vs in splits.items():
        utils_data.save_json({v: all_atoms[v] for v in vs}, split_files[sp])
    report = split_report(counts, splits)
    utils_data.save_json({"version": STRATIFIED_VERSION, "fractions": SPLIT_FRACTIONS, "splits": splits,
                          "onsets": report}, atoms_folder / "split.json")
    n = {sp: len(vs) for sp, vs in splits.items()}
    total = {sp: sum(r[sp] for r in report.values()) for sp in splits}
    print(f"[Split] stratified by the onsets of {len(report)} heads: videos {n}, onsets {total}")
    missing = sorted(h for h, r in report.items() if sum(r.values()) >= 5 and min(r["val"], r["test"]) == 0)
    if missing:
        print(f"[Split] heads with >= 5 onsets but none in val or test: {missing}")


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



# -------------- Near vehicles --------------
# Far vehicles (small GT boxes) are dropped from the dataset (--min_vehicle_height h > 0, exps *_sn): a
# vehicle of VEHICLE_CLASSES exists only in the frames where its box is at least h of the frame height,
# with hysteresis (it appears at >= h and leaves below NEAR_HYSTERESIS h; frames without a readable box
# keep the state) and gaps of at most NEAR_FILL_GAP dropped frames between kept ones filled. Dropped
# entirely: neither a prediction target nor the context of other agents. Each run of kept frames becomes a
# tube of its own with the same tube_uid (so the same agent): it disappears while far, and a re-entry
# is no onset (step 05 needs the agent seen in the frame before). Pedestrians, cyclists, motorbikes and
# traffic lights are small when near too: kept.
VEHICLE_CLASSES = ("Car", "MedVeh", "LarVeh", "Bus", "EmVeh", "SmalVeh")
NEAR_HYSTERESIS = 0.8
NEAR_FILL_GAP = 5


def near_runs(tube, frames, dataset_labels, min_height):
    """The annos ({frame_id: box_id}) of one agent tube that are kept, as a list of runs (one dict per
    run of kept frames); [all annos] for a class that is not filtered or without min_height."""
    annos = tube.get("annos", {})
    if not min_height or dataset_labels["agent_labels"][tube["label_id"]] not in VEHICLE_CLASSES:
        return [annos]
    items = sorted(annos.items(), key=lambda kv: int(kv[0]))
    keep, near = [], False
    for frame_id, box_id in items:
        frame = frames.get(str(frame_id), {})
        box = _normalized_box(frame.get("annos", {}).get(box_id, {}).get("box"), frame)
        if box is not None:
            h = box[3] - box[1]
            near = h >= (NEAR_HYSTERESIS * min_height if near else min_height)
        keep.append(near)
    kept = [i for i, k in enumerate(keep) if k]
    for a, b in zip(kept, kept[1:]):
        if 1 < b - a <= NEAR_FILL_GAP + 1:
            for i in range(a + 1, b):
                keep[i] = True
    runs, cur = [], {}
    for (frame_id, box_id), k in zip(items, keep):
        if k:
            cur[frame_id] = box_id
        elif cur:
            runs.append(cur)
            cur = {}
    if cur:
        runs.append(cur)
    return runs


def near_tubes(agent_tubes, frames, dataset_labels, min_height):
    """agent_tubes with the far frames of the vehicles dropped (`near_runs`): a tube split by dropped
    frames becomes one tube per run ("<id>~<j>", same tube_uid); a tube that is never near is gone."""
    if not min_height:
        return agent_tubes
    out = {}
    for tube_id, tube in agent_tubes.items():
        runs = near_runs(tube, frames, dataset_labels, min_height)
        for j, annos in enumerate(runs):
            out[tube_id if len(runs) == 1 else f"{tube_id}~{j}"] = {**tube, "annos": annos}
    return out


def _tracks_to_atoms(track_dir, video_ids, lang, atom_file, dataset_labels, min_height=0.0):
    video_atoms = utils_data.load_json(atom_file) if atom_file.exists() else {}

    missing_ids = [vid for vid in video_ids if vid not in video_atoms]
    
    if missing_ids:
        for vid in tqdm(missing_ids, desc="Tracks to Atoms"):
            track_file = Path(track_dir) / f"{vid}_gt.json"
            if not track_file.exists():
                continue

            track_data = utils_data.load_json(track_file)       

            frames = track_data["data"]["frames"]
            agent_tubes = near_tubes(track_data["data"]["agent_tubes"], frames, dataset_labels, min_height)
            segments_by_ego_actions = track_data["data"]["av_action_tubes"]

            interval_atoms = []
            interval_atoms.extend(lang.video2atoms("av", segments_by_ego_actions, frames))
            interval_atoms.extend(lang.video2atoms("agents", agent_tubes, frames))


            interval_atoms = [utils_data.add_labels_to_atom(atom, dataset_labels) for atom in interval_atoms]
            per_frame_atoms = get_frame_pairs(interval_atoms, dataset_labels)
            video_atoms[vid] = {
                "interval_atoms": interval_atoms,
                "per_frame_atoms": per_frame_atoms
            }

        utils_data.save_json(video_atoms, atom_file)

    return video_atoms

# Frame size of the ROAD++ (Waymo front camera) videos, to normalize boxes given in pixels.
DEFAULT_FRAME_SIZE = (1920, 1280)


def _normalized_box(box, frame):
    """(x1, y1, x2, y2) in [0, 1] of a GT box: a list [x1, y1, x2, y2] or a dict
    {"x1", "y1", "x2", "y2"}, normalized already or in pixels (divided by the
    frame's width / height, else DEFAULT_FRAME_SIZE). None if it cannot be read."""
    if isinstance(box, dict):
        box = [box.get(key) for key in ("x1", "y1", "x2", "y2")]
    if not isinstance(box, (list, tuple)) or len(box) != 4 or any(v is None for v in box):
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    if max(x1, y1, x2, y2) > 1.5:
        width = float(frame.get("width") or DEFAULT_FRAME_SIZE[0])
        height = float(frame.get("height") or DEFAULT_FRAME_SIZE[1])
        x1, x2, y1, y2 = x1 / width, x2 / width, y1 / height, y2 / height
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _tracks_to_boxes(track_dir, video_ids, box_file, dataset_labels, min_height=0.0):
    """Per-frame boxes of every agent tube, for the agent roles of step 05 (`roles.py`).

    Saves {video_id: {tube_uid: {"label": agent class label, "frames": (K,) long,
    "boxes": (K, 4) float normalized x1, y1, x2, y2}}} to box_file (torch). The
    tube_uid is the one of the tube's atoms (`utils_data.get_start_end_frame`),
    so step 05 maps the boxes onto its agents; tubes without one are left out
    (their atoms share the ego's None)."""
    out, unread, read = {}, 0, 0
    for vid in tqdm(video_ids, desc="Tracks to Boxes"):
        track_file = Path(track_dir) / f"{vid}_gt.json"
        if not track_file.exists():
            continue
        data = utils_data.load_json(track_file)["data"]
        frames = data["frames"]
        tubes = {}
        for tube in near_tubes(data["agent_tubes"], frames, dataset_labels, min_height).values():
            _start, _end, tube_uid = utils_data.get_start_end_frame(tube, frames)
            if tube_uid is None:
                continue
            frame_ids, boxes = [], []
            if tube_uid in tubes:                       # a near-vehicle run of a split tube: append
                frame_ids = tubes[tube_uid]["frames"].tolist()
                boxes = tubes[tube_uid]["boxes"].tolist()
            for frame_id, box_id in tube.get("annos", {}).items():
                frame = frames.get(str(frame_id), {})
                box = _normalized_box(frame.get("annos", {}).get(box_id, {}).get("box"), frame)
                if box is None:
                    unread += 1
                    continue
                read += 1
                frame_ids.append(int(frame_id))
                boxes.append(box)
            if frame_ids:
                tubes[tube_uid] = {"label": dataset_labels["agent_labels"][tube["label_id"]],
                                   "frames": torch.tensor(frame_ids, dtype=torch.long),
                                   "boxes": torch.tensor(boxes, dtype=torch.float)}
        out[vid] = tubes
    torch.save(out, box_file)
    print(f"[Boxes] {read} boxes of {sum(len(t) for t in out.values())} tubes in {len(out)} videos"
          f" ({unread} unreadable) saved to {box_file}")
    if read == 0:
        raise ValueError(f"No readable box in the tracks of {track_dir}: check the box format (`_normalized_box`).")


def _build_unique_atoms(atom_files, unique_atoms_file):
    """Vocabulary of atom signatures without tube_uid, shared by all splits.
    Built from the interval atoms, since per_frame_atoms miss atoms whose intervals
    are all zero-length (end_frame <= start_frame)."""
    unique_atoms = {}
    for atom_file in atom_files:
        for vid_atoms in tqdm(utils_data.load_json(atom_file).values(), desc="Unique Atoms"):
            for interval in vid_atoms["interval_atoms"]:
                unique_atoms.setdefault(utils_data.atom_to_signature(interval)[:-2], len(unique_atoms))
    utils_data.save_json(list(unique_atoms), unique_atoms_file)
    print(f"[Unique Atoms] {len(unique_atoms)} atoms saved to {unique_atoms_file}")


def _compress_atom_file(atom_file, compressed_file, atom_vocab):
    atoms = utils_data.load_json(atom_file)
    compressed_atoms = {}
    for vid, vid_atoms in tqdm(atoms.items(), desc="Compressing Atoms"):
        compressed_atoms[vid] = {}
        compressed_atoms[vid]["interval_atoms"] = vid_atoms["interval_atoms"]

        compressed_per_frame_atoms = []
        for current_frame, next_frame in vid_atoms["per_frame_atoms"]:
            cf_compressed = [(atom_vocab[tuple(atom[:-2])], atom[-2], atom[-1]) for atom in current_frame]
            nf_compressed = [(atom_vocab[tuple(atom[:-2])], atom[-2], atom[-1]) for atom in next_frame]
            compressed_per_frame_atoms.append((cf_compressed, nf_compressed))
        compressed_atoms[vid]["per_frame_atoms"] = compressed_per_frame_atoms
    utils_data.save_json(compressed_atoms, compressed_file)


def _onset_statistics(compressed_file, stats_dir, horizon):
    """Histogram of delta_t (frames until an atom's next onset) over all anchor frames, per atom."""
    stats_file = stats_dir / "onset_stats.json"
    if stats_file.exists():
        print(f"[Onset Stats] {stats_file} already exists.")
        return
    os.makedirs(stats_dir, exist_ok=True)

    atoms = utils_data.load_atoms(compressed_file)
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


            




# Version of the atom files: files of another version (or of none) are rebuilt, with the vocabulary
# and the compressed files (the boxes are kept). "av_frames_v1": the ego's atoms from the per-frame
# av_action_ids instead of the av_action_tubes (`video2atoms`). The vocabulary may change with it, and
# steps 03 / 04 store atom indices: they rebuild when unique_atoms.json is newer than their outputs.
ATOMS_VERSION = "av_frames_v1"


def prepare_per_frame_atoms(language_model, cfg):
    data_num = cfg["data_num"]
    track_dir = cfg["track_dir"]
    output_dir = cfg["output_dir"]
    split_mode = cfg.get("split", "random")
    all_track_files =[os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.endswith("_gt.json")]
    if split_mode == "stratified":
        all_track_files = sorted(all_track_files)       # listdir order differs between file systems
    if data_num != 'full':
        all_track_files = all_track_files[:int(data_num)]

    train_indices, val_indices, test_indices = _split_example_indices(len(all_track_files))
    atoms_folder= Path(output_dir) / "atoms" 
    os.makedirs(atoms_folder, exist_ok=True)
    version_file = atoms_folder / "atoms_version.txt"
    version = version_file.read_text().strip() if version_file.exists() else None
    expected = ATOMS_VERSION if split_mode == "random" else f"{ATOMS_VERSION}_{STRATIFIED_VERSION}"
    min_height = float(cfg.get("min_vehicle_height") or 0.0)
    split_from = None
    if cfg.get("split_from"):
        if split_mode != "stratified":
            raise ValueError("split_from needs split stratified")
        split_from = Path(output_dir).parent.parent / cfg["split_from"] / "step_dataset_output" / "atoms" / "split.json"
        if not split_from.exists():
            raise FileNotFoundError(f"split_from {cfg['split_from']}: no {split_from} (run that exp's step_dataset first)")
    if min_height:
        expected += f"_near{min_height:g}"
        if split_mode != "stratified":
            raise ValueError("min_vehicle_height is implemented for split stratified (exps *_sn)")
    if version != expected:
        stale = [atoms_folder / f"{split}_atoms{suffix}.json" for split in ("train", "val", "test")
                 for suffix in ("", "_compressed")] + [atoms_folder / "unique_atoms.json"]
        stale = [f for f in stale if f.exists()]
        if stale:
            print(f"[Per Frame Atoms] atom files of version {version} (now {expected}): rebuilding "
                  + ", ".join(f.name for f in stale))
            for f in stale:
                f.unlink()

    splits = {"train": train_indices, "val": val_indices, "test": test_indices}
    if split_mode == "stratified":
        _stratified_split_files(track_dir, all_track_files, language_model, atoms_folder, cfg["dataset_labels"],
                                min_height, split_from)
    else:
        for split_name, indices in splits.items():
            split_ids = [Path(all_track_files[i]).stem.replace("_gt", "") for i in indices]
            split_file = atoms_folder / f"{split_name}_atoms.json"
            if not os.path.exists(split_file):
                _tracks_to_atoms(track_dir, split_ids, language_model, split_file, cfg["dataset_labels"])
            else:
                print(f"[Per Frame Atoms] {split_file} already exists.")

    # One vocabulary for all splits; compressed files index into it, so rebuild them with it.
    unique_atoms_file = atoms_folder / "unique_atoms.json"
    rebuild_compressed = not unique_atoms_file.exists()
    if rebuild_compressed:
        _build_unique_atoms([atoms_folder / f"{split_name}_atoms.json" for split_name in splits], unique_atoms_file)
    else:
        print(f"[Unique Atoms] {unique_atoms_file} already exists.")
    atom_vocab = {tuple(atom): i for i, atom in enumerate(utils_data.load_json(unique_atoms_file))}

    for split_name in splits:
        compressed_file = atoms_folder / f"{split_name}_atoms_compressed.json"
        if rebuild_compressed or not os.path.exists(compressed_file):
            _compress_atom_file(atoms_folder / f"{split_name}_atoms.json", compressed_file, atom_vocab)
        else:
            print(f"[Per Frame Atoms Compressed] {compressed_file} already exists.")

    version_file.write_text(expected + "\n")

    # The agents' boxes per frame (step 05 agent roles), for the videos of each split's atom file.
    for split_name in splits:
        box_file = atoms_folder / f"{split_name}_boxes.pt"
        if not os.path.exists(box_file):
            video_ids = list(utils_data.load_json(atoms_folder / f"{split_name}_atoms_compressed.json"))
            _tracks_to_boxes(track_dir, video_ids, box_file, cfg["dataset_labels"], min_height)
        else:
            print(f"[Boxes] {box_file} already exists.")

# -------------- Task Preparation Helper Functions --------------
def _prepare_task_target(interval_atom):
    start_frame=interval_atom["start_frame"]
    end_frame=interval_atom["end_frame"]
    if "location_name" in interval_atom:
        task_target = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "pred": interval_atom["pred"],
            "location_name": interval_atom["location_name"],
            "agent_class": interval_atom["agent_class"],
        }
    elif "action_id" in interval_atom:
        task_target = {
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "pred": interval_atom["pred"],
            "action_id": interval_atom["action_id"],
            "agent_class": interval_atom["agent_class"],
        }
    else:
        raise ValueError(f"Unknown interval atom type: {interval_atom}")
    return task_target 

def _prepare_task_input(interval_atom, per_frame_atoms, min_window):
    start_frame = int(interval_atom["start_frame"])
    task_input = per_frame_atoms[:start_frame]
    task_input = [i[0] for i in task_input if len(i[0])>0]
    return task_input


def prepare_per_split(cfg,split_name):
    output_file = cfg["output_dir"] / f"tasks_{split_name}.pt"
    if output_file.exists():
        print(f"[{split_name} Tasks] already exists.")
        return
    atoms = utils_data.load_json(cfg["output_dir"] / "atoms" /f"{split_name}_atoms_compressed.json")
    min_window = cfg["delay_windows"][0]
    tasks = {}
    for vid, video_atoms in atoms.items():
        video_tasks = []
        interval_atoms = video_atoms["interval_atoms"]
        per_frame_atoms = video_atoms["per_frame_atoms"]
        for interval_atom in interval_atoms:
            if int(interval_atom["start_frame"])<=min_window:
                continue
            task = {}
            task["input"] = _prepare_task_input(interval_atom, per_frame_atoms, min_window)
            task["target"] = _prepare_task_target(interval_atom)
            video_tasks.append(task)
        tasks[vid] = video_tasks
    torch.save({"tasks": tasks}, output_file)
    print(f"[{split_name} Tasks] saved to {output_file}")



def main(input_data):
    print("\n--------- Step Dataset ----------------------\n")
    
    cfg = {
        "track_dir": input_data["dataset_path"] / "gt",
        "output_dir": input_data["output_dir"],
        "data_num": input_data["data_num"],
        "device": input_data["device"],
        "delay_windows": input_data["delay_windows"],
        "split": input_data.get("split", "random"),
        "min_vehicle_height": input_data.get("min_vehicle_height", 0.0),
        "split_from": input_data.get("split_from"),
        "dataset_labels": json.load(open(input_data["dataset_path"] / "gt" / "label.json", 'r'))
        
    }
    print(f"[Data Num] {cfg['data_num']}")

    # Prepare language model and beam search model
    language_model = Language(cfg["device"])
    # Prepare dataset by converting tracks to atoms for train, val, and test splits
    prepare_per_frame_atoms(language_model, cfg)
    # prepare_per_frame_events(cfg)
    for split_name in ["train", "val", "test"]:
        prepare_per_split(cfg, split_name)

    print("\n--------- Step Dataset Done------------------\n")