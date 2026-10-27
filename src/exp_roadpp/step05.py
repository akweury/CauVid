

from pathlib import Path
import os
import sys
import getpass
import json
import time
import torch
import numpy as np
from tqdm import tqdm

from src.exp_roadpp import utils_data



def _atom_entity_key(atom_signature):
    tube_uid = atom_signature[-1]
    if tube_uid is not None:
        return ("tube_uid", tube_uid)
    else:
        return ("agent_class", "av")


def _beta_const_log_term(beta=2 / 3, gamma=-0.1, zeta=1.1):
    # Closed-form constant from Louizos et al. (2017), "Learning Sparse Neural
    # Networks through L0 Regularization": P(gate != 0) = sigmoid(log_alpha - beta*log(-gamma/zeta)).
    return beta * np.log(-gamma / zeta)


def hard_concrete_mask(log_alpha, beta=2 / 3, gamma=-0.1, zeta=1.1, training=True):
    if training:
        u = torch.rand_like(log_alpha).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    else:
        s = torch.sigmoid(log_alpha)
    s_bar = s * (zeta - gamma) + gamma
    return s_bar.clamp(0, 1)



def _auto_batch_size(num_features, device, safety_factor=0.25, max_batch_size=1 << 20):
    """Pick a row batch size that should fit in the current GPU's free memory.

    Scales with whatever card is attached at runtime (e.g. 8GB GTX 1080 vs an
    80GB A100) instead of assuming a fixed size.
    """
    if device is None or str(device) == 'cpu' or not torch.cuda.is_available():
        return max_batch_size
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    usable_bytes = free_bytes * safety_factor
    # Heuristic: forward/backward intermediates for `hard_concrete_mask` and the
    # `torch.prod` term need a handful of (batch, M) float32 buffers per row.
    bytes_per_row = max(num_features, 1) * 4 * 6
    batch_size = max(1, int(usable_bytes // bytes_per_row))
    return min(batch_size, max_batch_size)


def sparse_rows_to_dense(sparse, dtype=torch.bool):
    """Inverse of `_to_sparse_rows`: return the dense (N, M) tensor."""
    num_rows, num_cols = sparse["shape"]
    crow = sparse["crow"].long()
    rows = torch.repeat_interleave(torch.arange(num_rows), crow[1:] - crow[:-1])
    dense = torch.zeros((num_rows, num_cols), dtype=dtype)
    dense[rows, sparse["col"].long()] = 1
    return dense





def _csr_from_cells(rows, columns, shape):
    """Tensor version of `_to_sparse_rows`; `rows`/`columns` are long tensors."""
    num_rows, num_cols = shape
    row_counts = torch.bincount(rows, minlength=num_rows)
    crow = torch.zeros(num_rows + 1, dtype=torch.long)
    torch.cumsum(row_counts, dim=0, out=crow[1:])
    index_dtype = torch.int32 if len(columns) < 2**31 else torch.long
    return {
        "crow": crow.to(index_dtype),
        "col": columns.to(torch.int32),
        "shape": (num_rows, num_cols),
    }



    
def _to_sparse_rows(rows, columns, shape):
    """Pack the True cells of a 0/1 (N, M) matrix in CSR form.

    `rows` must be non-decreasing (cells are appended row by row). Only the
    True cells are stored: `col` holds their column ids and row i's columns
    are col[crow[i]:crow[i + 1]]. This replaces a dense (N, M) matrix, which
    for sparse features costs one byte per cell, mostly False.
    """
    return _csr_from_cells(
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(columns, dtype=torch.long),
        shape,
    )

def _index_frame_entities(atoms, horizon):
    """Give every (frame, entity) observed in frame t a node id and list its atoms.

    Node ids are contiguous per frame, so nodes of one frame form a
    consecutive range. Entities not observed in frame t get no node.
    Returns long tensors:
      node_frame[n]: frame of node n;
      current_cells: (node, atom_index) of every atom in frame t;
      onset_cells: (node, atom_index, delta_t) of the first onset of each atom
          of the node's entity within frames (t, t + horizon].
    and the list node_keys[n] = (video_id, t, tube_uid) of node n.
    """
    atom_vocab = {tuple(atom): i for i, atom in enumerate(atoms["unique_atoms"])}
    node_frame = []
    node_keys = []
    current_nodes, current_atom_ids = [], []
    onset_nodes, onset_atom_ids, onset_delays = [], [], []
    frame_id = 0
    for video_id, video_atoms in tqdm(atoms.items(), desc="Indexing frames"):
        if video_id == "unique_atoms":
            continue
        entity_onsets = utils_data.build_entity_onset_index(video_atoms["interval_atoms"], atom_vocab)
        for t, (current_frame, _next_frame) in enumerate(video_atoms["per_frame_atoms"]):
            if len(current_frame) == 0:
                continue
            base = len(node_frame)
            entity_nodes = {}
            node_tube_uids = {}
            for atom in current_frame:
                node = entity_nodes.setdefault(
                    _atom_entity_key(atom), base + len(entity_nodes)
                )
                node_tube_uids[node] = atom[-1]
                current_nodes.append(node)
                current_atom_ids.append(atom[0])
            for node, tube_uid in node_tube_uids.items():
                node_keys.append((video_id, t, tube_uid))
                for atom_index, delta_t in utils_data.future_onsets(
                    entity_onsets.get(tube_uid, []), t, horizon
                ):
                    onset_nodes.append(node)
                    onset_atom_ids.append(atom_index)
                    onset_delays.append(delta_t)
            node_frame.extend([frame_id] * len(entity_nodes))
            frame_id += 1
    return (
        torch.tensor(node_frame, dtype=torch.long),
        torch.tensor([current_nodes, current_atom_ids], dtype=torch.long),
        torch.tensor([onset_nodes, onset_atom_ids, onset_delays], dtype=torch.long),
        node_keys,
    )


def _node_tables(atoms, horizon, device):
    """Dense per-node tables of `_index_frame_entities`. Returns
      node_frame[n]: frame of node n;
      current[n, a]: atom a holds for node n in frame t;
      delay[n, a]: frames until atom a next starts for node n (0: not within
          horizon, or already active in frame t);
      node_keys[n]: (video_id, t, tube_uid) of node n.
    """
    num_atoms = len(atoms["unique_atoms"])
    node_frame, current_cells, onset_cells, node_keys = _index_frame_entities(atoms, horizon)
    num_nodes = len(node_frame)
    node_frame = node_frame.to(device)
    current = torch.zeros((num_nodes, num_atoms), dtype=torch.bool, device=device)
    current[current_cells[0].to(device), current_cells[1].to(device)] = True
    delay = torch.zeros((num_nodes, num_atoms), dtype=torch.int16, device=device)
    delay[onset_cells[0].to(device), onset_cells[1].to(device)] = onset_cells[2].to(
        device=device, dtype=torch.int16
    )
    delay[current] = 0
    return node_frame, current, delay, node_keys


def build_video_tensor(video_atoms, num_atoms, device="cpu"):
    """(T, N, A) bool tensor of one video: X[t, n, a] is True if agent n holds atom a in frame t.

    video_atoms is one video of the compressed atom file; its per_frame_atoms[t]
    is the (frame t, frame t + 1) pair of [atom_index, "tube_uid", tube_uid]
    atoms, so T = len(per_frame_atoms) + 1. Agents are the tube_uids (the ego
    vehicle has tube_uid None), numbered 0..N-1 in order of first appearance.
    num_atoms is A, the size of the shared atom vocabulary.
    """
    frames = _video_frames(video_atoms)
    agent_ids = video_agent_ids(video_atoms)
    cells = [
        (t, agent_ids[tube_uid], atom_index)
        for t, frame_atoms in enumerate(frames)
        for atom_index, _tag, tube_uid in frame_atoms
    ]
    X = torch.zeros((len(frames), len(agent_ids), num_atoms), dtype=torch.bool, device=device)
    if cells:
        t, n, a = torch.tensor(cells, dtype=torch.long, device=device).T
        X[t, n, a] = True
    return X


def _video_frames(video_atoms):
    # The T frames of `build_video_tensor`: every current frame and the last next frame.
    per_frame_atoms = video_atoms["per_frame_atoms"]
    frames = [current_frame for current_frame, _next_frame in per_frame_atoms]
    if per_frame_atoms:
        frames.append(per_frame_atoms[-1][1])
    return frames


def video_agent_ids(video_atoms):
    """{tube_uid: agent index n} of `build_video_tensor`, in order of first appearance."""
    agent_ids = {}
    for frame_atoms in _video_frames(video_atoms):
        for _atom_index, _tag, tube_uid in frame_atoms:
            agent_ids.setdefault(tube_uid, len(agent_ids))
    return agent_ids

def build_video_delay_tensor(X, horizon):
    """(T, N, A) int16 tensor of one video: D[t, n, a] is the number of frames until
    agent n next starts atom a after frame t, i.e. the head targets of step 05.

    X is the (T, N, A) tensor of `build_video_tensor`. Atom a starts for agent n
    in frame s if X[s, n, a] and not X[s - 1, n, a]. D[t, n, a] is 0 if a does
    not start for n within frames (t, t + horizon], or if n already holds a in
    frame t (only new events count).
    """
    num_frames, num_agents, num_atoms = X.shape
    starts = X.clone()
    starts[1:] &= ~X[:-1]
    D = torch.zeros((num_frames, num_agents, num_atoms), dtype=torch.int16, device=X.device)
    # next_start[n, a]: first frame after t in which a starts for n (scanning t backwards).
    no_start = num_frames + horizon + 1
    next_start = torch.full((num_agents, num_atoms), no_start, dtype=torch.long, device=X.device)
    for t in range(num_frames - 2, -1, -1):
        next_start = torch.where(starts[t + 1], t + 1, next_start)
        delta = next_start - t
        D[t] = torch.where((delta <= horizon) & ~X[t], delta, 0).to(torch.int16)
    return D


def build_body_features(X, D, body, head_atoms, windows):
    """Training rows of one body (b, s) in one video.

    X, D: (T, N, A) atom and delay tensors of the video (`build_video_tensor`,
    `build_video_delay_tensor`). body = (b, s): agent slot s holds atom b.
    head_atoms: (H,) atom ids of the body's candidate heads; windows: (K,).

    A row is one grounding in a frame t where some agent n_body holds b: the
    head agent n_head is n_body itself if s == 0, otherwise every other agent
    visible in frame t. Returns
      features: (R, s + 2, A) bool, the atoms in frame t of each role:
          slot 0 the head agent, slot s the body agent (without b itself, which
          every row has) and slot s + 1 "some other agent" (any agent but those two);
      targets: (R, H, K) bool, head h starts for the head agent within windows[k]
          frames after t (and is not already held by it in frame t);
      rows: (R, 3) long, (t, n_body, n_head) of each row.
    """
    b, s = body
    # t is the frame indices where the body atom b is held by some agent.
    # n_body is the agent indices holding the body atom b in those frames.
    t, n_body = torch.nonzero(X[:, :, b], as_tuple=True)
    if s == 0:
        n_head = n_body
    else:
        # Pair each body agent with every other agent visible in the same frame.
        candidates = X[t].any(dim=2)                        # (R_body, N)
        candidates[torch.arange(len(t), device=X.device), n_body] = False
        pair, n_head = torch.nonzero(candidates, as_tuple=True)
        t, n_body = t[pair], n_body[pair]

    head = X[t, n_head]                                     # (R, A)
    body_agent = X[t, n_body]                               # (R, A)
    others = X.sum(dim=1)[t] - body_agent.int()             # (R, A) agents holding a, minus the roles
    if s != 0:
        others -= head.int()
    features = torch.zeros((len(t), s + 2, X.shape[2]), dtype=torch.bool, device=X.device)
    features[:, 0] = head
    features[:, s] |= body_agent
    features[:, s, b] = False
    features[:, s + 1] = others > 0

    delay = D[t, n_head][:, head_atoms].unsqueeze(-1)       # (R, H, 1)
    targets = (delay > 0) & (delay <= windows)              # (R, H, K)
    rows = torch.stack([t, n_body, n_head], dim=1)
    return features, targets, rows


def build_features(atoms, caw_clauses, unique_atoms, device):
    bodies = list(set([c[0] for c in caw_clauses]))
    # Sorted so the window axis K of the targets has a fixed order.
    delay_windows = sorted(set([c[-1] for c in caw_clauses]))
    num_atom_type = len(unique_atoms)
    windows = torch.tensor(delay_windows, dtype=torch.int16, device=device)
    features = []
    targets = []
    features = {body:[] for body in bodies}
    targets = {body:[] for body in bodies}
    rows = {body:[] for body in bodies}
    heads_by_body = {}
    for body, head, _window in caw_clauses:
        heads_by_body.setdefault(body, {})[head] = None
    head_atoms = {
        body: torch.tensor([head[0] for head in heads], dtype=torch.long, device=device)
        for body, heads in heads_by_body.items()
    }
    for video_atoms in atoms.values():
        X = build_video_tensor(video_atoms, num_atom_type, device=device)
        D = build_video_delay_tensor(X, max(delay_windows))
        # print X and D memory usage
        print(f"X memory usage: {X.element_size() * X.nelement() / 1024 ** 2:.2f} MB")
        print(f"D memory usage: {D.element_size() * D.nelement() / 1024 ** 2:.2f} MB")
        for body in bodies:
            F, T, R = build_body_features(X,D, body,head_atoms[body], windows)
            if len(F)>0:
                features[body].append(F)
                targets[body].append(T)
                rows[body].append(R)
            
    return features, targets, rows
        




def _encode_examples(examples):
    """Replace every atom signature in `examples` with an integer id.

    Returns (compact, signatures): `compact` holds only ids and tensors,
    `signatures[i]` is the atom signature with id i.
    """
    signature_ids = {}
    signatures = []

    def to_id(signature):
        signature_id = signature_ids.get(signature)
        if signature_id is None:
            signature_id = len(signatures)
            signature_ids[signature] = signature_id
            signatures.append(signature)
        return signature_id

    body_ids = []
    candidate_head_ids = []
    for body_signature, body_group in examples["body_groups"].items():
        body_ids.append(to_id(body_signature))
        candidate_head_ids.append(torch.tensor(
            [to_id(h) for h in body_group["candidate_heads"]], dtype=torch.long
        ))
    # feature_vocab[g][j] = (g, object_slot, atom_fact); g is implied by the
    # list position, so store only (object_slot, atom_id) as an (M_g, 2) tensor.
    feature_vocab = [
        torch.tensor(
            [(object_slot, to_id(atom_fact)) for _gid, object_slot, atom_fact in vocab],
            dtype=torch.long,
        ).reshape(-1, 2)
        for vocab in examples["feature_vocab"]
    ]
    # head_vocab is rebuilt from body_groups on load (see build_head_vocab).
    compact = {
        "features": examples["features"],
        "targets": examples["targets"],
        "feature_vocab": feature_vocab,
        "body_ids": torch.tensor(body_ids, dtype=torch.long),
        "candidate_head_ids": candidate_head_ids,
        "delay_windows": torch.tensor(examples["delay_windows"], dtype=torch.long),
    }
    # Pruned examples only: feature_aliases[g][j] is a list of vocab entries.
    # Flatten each group to (K_g, 2) (object_slot, atom_id) rows; column j's
    # aliases are rows alias_offsets[g][j]:alias_offsets[g][j + 1].
    if "feature_aliases" in examples:
        compact["alias_atoms"] = []
        compact["alias_offsets"] = []
        for group_aliases in examples["feature_aliases"]:
            flat = [
                (object_slot, to_id(atom_fact))
                for aliases in group_aliases
                for _gid, object_slot, atom_fact in aliases
            ]
            offsets = [0]
            for aliases in group_aliases:
                offsets.append(offsets[-1] + len(aliases))
            compact["alias_atoms"].append(
                torch.tensor(flat, dtype=torch.long).reshape(-1, 2)
            )
            compact["alias_offsets"].append(torch.tensor(offsets, dtype=torch.long))
    return compact, signatures


def build_head_vocab(body_groups, delay_windows):
    # One cumulative head "starts within w" per (candidate head, window).
    head_vocab = []
    for body, heads in body_groups.items():
        for head in heads:
            for window in delay_windows:
                head_vocab.append((body, head, window))
    return head_vocab

    
def _decode_examples(compact, signatures):
    """Inverse of `_encode_examples`."""
    body_groups = {}
    for body_id, head_ids in zip(compact["body_ids"].tolist(), compact["candidate_head_ids"]):
        body_groups[signatures[body_id]] = {
            "candidate_heads": tuple(signatures[h] for h in head_ids.tolist())
        }
    delay_windows = compact["delay_windows"].tolist()
    head_vocab = build_head_vocab(body_groups, delay_windows)
    group_ids = {body_signature: gid for gid, body_signature in enumerate(body_groups)}
    group_head_vocab = [[] for _ in range(len(body_groups))]
    for body_signature, head_signature, window in head_vocab:
        gid = group_ids[body_signature]
        group_head_vocab[gid].append((gid, head_signature, window))
    feature_vocab = [
        [(gid, object_slot, signatures[atom_id]) for object_slot, atom_id in vocab.tolist()]
        for gid, vocab in enumerate(compact["feature_vocab"])
    ]
    examples = {
        "features": compact["features"],
        "targets": compact["targets"],
        "feature_vocab": feature_vocab,
        "head_vocab": group_head_vocab,
        "body_groups": body_groups,
        "delay_windows": delay_windows,
    }
    if "alias_atoms" in compact:
        examples["feature_aliases"] = []
        for gid, (flat, offsets) in enumerate(
            zip(compact["alias_atoms"], compact["alias_offsets"])
        ):
            flat = [(gid, object_slot, signatures[atom_id]) for object_slot, atom_id in flat.tolist()]
            offsets = offsets.tolist()
            examples["feature_aliases"].append(
                [flat[start:end] for start, end in zip(offsets[:-1], offsets[1:])]
            )
    return examples



def _save_examples(examples, cache_file, signature_file):
    # Features/targets/vocab ids live in the .pt file; the id -> atom signature
    # mapping lives in a separate JSON file so signatures are stored only once.
    compact, signatures = _encode_examples(examples)
    torch.save(compact, cache_file)
    utils_data.save_json(signatures, signature_file)
    print(f"[Cache] Saved {cache_file} and {signature_file}")


def _load_examples(cache_file, signature_file):
    """Return the examples cached by `_save_examples`, or None if unavailable."""
    if not (cache_file.exists() and signature_file.exists()):
        return None
    compact = torch.load(cache_file, weights_only=False)
    # Older caches stored dense bool tensors; rebuild them in sparse form.
    if any(torch.is_tensor(X) for X in compact["features"]):
        print(f"[Cache] {cache_file} uses the old dense format, rebuilding")
        return None
    with open(signature_file, "r") as f:
        signatures = [tuple(s) for s in json.load(f)]
    print(f"[Cache] Loaded {cache_file} and {signature_file}")
    return _decode_examples(compact, signatures)




def group_clauses_by_body(clauses):
    clauses_by_body = {}
    for c in clauses.values():
        anchor_body = tuple(c['anchor']["body"][0])
        clauses_by_body[anchor_body] = clauses_by_body.get(anchor_body, [])
        clauses_by_body[anchor_body].append(tuple(c['anchor']["head"]))
    
    for body_group in clauses_by_body.values():
        body_group[:] = tuple(sorted(body_group, key=repr))

    print(f"[Grouping] Clauses grouped by body: {len(clauses)} -> {len(clauses_by_body)}")
    return clauses_by_body




def _prune_group_features(sparse, num_head_positives, min_support):
    """Choose which feature columns of one group to keep.

    Any extension requires all of its selected features, so its support is at
    most the row count of its rarest feature. The following can therefore
    never yield an extension that passes `min_support`, or never change one:
      - the whole group, if it has fewer than `min_support` rows or every head
        is degenerate (fires in no row or in every row);
      - features True in fewer than `min_support` rows;
      - features True in every row (adding them changes nothing).
    Columns True in exactly the same rows are merged: the first is kept and
    the rest are returned as its aliases.

    Returns (kept, aliases): `kept` is a list of old column ids in their new
    order, `aliases[i]` lists the old column ids that duplicate `kept[i]`.
    """
    num_rows, num_cols = sparse["shape"]
    head_learnable = (num_head_positives > 0) & (num_head_positives < num_rows)
    if num_rows < min_support or not bool(head_learnable.any()):
        return [], []

    crow = sparse["crow"].long()
    col = sparse["col"].long()
    rows = torch.repeat_interleave(torch.arange(num_rows), crow[1:] - crow[:-1])
    counts = torch.bincount(col, minlength=num_cols)
    candidates = torch.nonzero(
        (counts >= min_support) & (counts < num_rows), as_tuple=True
    )[0]
    if len(candidates) == 0:
        return [], []

    # Bucket columns by (count, two random row-hashes); equal row sets always
    # share a bucket. Hash values stay below 2**31, so sums cannot overflow.
    generator = torch.Generator().manual_seed(0)
    row_hashes = torch.randint(0, 2**31, (2, num_rows), generator=generator)
    column_hashes = torch.zeros((2, num_cols), dtype=torch.long)
    column_hashes[0].index_add_(0, col, row_hashes[0][rows])
    column_hashes[1].index_add_(0, col, row_hashes[1][rows])
    keys = torch.stack(
        [counts[candidates], column_hashes[0][candidates], column_hashes[1][candidates]],
        dim=1,
    )
    _unique_keys, bucket_ids = torch.unique(keys, dim=0, return_inverse=True)

    # Row ids of each column (CSC view) to confirm duplicates exactly.
    order = torch.argsort(col, stable=True)
    rows_by_column = rows[order]
    column_start = torch.zeros(num_cols + 1, dtype=torch.long)
    torch.cumsum(counts, dim=0, out=column_start[1:])

    def column_rows(j):
        return rows_by_column[column_start[j]:column_start[j + 1]]

    kept = []
    aliases = []
    # bucket id -> list of (position in kept, row ids) of that bucket's columns.
    representatives = {}
    for j, bucket in zip(candidates.tolist(), bucket_ids.tolist()):
        j_rows = column_rows(j)
        for kept_pos, rep_rows in representatives.get(bucket, []):
            if torch.equal(j_rows, rep_rows):
                aliases[kept_pos].append(j)
                break
        else:
            representatives.setdefault(bucket, []).append((len(kept), j_rows))
            kept.append(j)
            aliases.append([])
    return kept, aliases




def prune_examples(examples, min_support):
    """Drop feature columns that cannot affect any accepted extension.

    See `_prune_group_features` for the rules. Rows and groups are kept so
    group ids stay stable; a pruned group just ends up with zero features and
    is skipped by `extend_clauses`. Adds `feature_aliases[g][j]`: the vocab
    entries that duplicate `feature_vocab[g][j]`.
    """
    features = []
    feature_vocab = []
    feature_aliases = []
    num_before = sum(X["shape"][1] for X in examples["features"])
    for gid, sparse in enumerate(examples["features"]):
        num_rows, num_cols = sparse["shape"]
        target = examples["targets"][gid]
        num_head_positives = torch.bincount(
            target["col"].long(), minlength=target["shape"][1]
        )
        kept, aliases = _prune_group_features(sparse, num_head_positives, min_support)

        old_to_new = torch.full((num_cols,), -1, dtype=torch.long)
        old_to_new[torch.tensor(kept, dtype=torch.long)] = torch.arange(len(kept))
        crow = sparse["crow"].long()
        rows = torch.repeat_interleave(torch.arange(num_rows), crow[1:] - crow[:-1])
        new_col = old_to_new[sparse["col"].long()]
        keep_cell = new_col >= 0
        features.append(_csr_from_cells(
            rows[keep_cell], new_col[keep_cell], (num_rows, len(kept))
        ))

        vocab = examples["feature_vocab"][gid]
        feature_vocab.append([vocab[j] for j in kept])
        feature_aliases.append([[vocab[j] for j in dup] for dup in aliases])

    num_after = sum(X["shape"][1] for X in features)
    num_empty = sum(X["shape"][1] == 0 for X in features)
    print(
        f"[Pruning] min_support={min_support}: features {num_before} -> {num_after}, "
        f"{num_empty}/{len(features)} groups left with no features"
    )
    return {
        **examples,
        "features": features,
        "feature_vocab": feature_vocab,
        "feature_aliases": feature_aliases,
    }

def build_training_examples(atoms, clauses, labels, unique_atoms, cfg):
    body_groups = group_clauses_by_body(clauses)
    head_vocab = build_head_vocab(body_groups, cfg["delay_windows"])
    features, targets, rows = build_features(atoms, head_vocab , unique_atoms, cfg["device"])
    examples = {
        "body_groups": body_groups,
        "delay_windows": list(cfg["delay_windows"]),
        # Axes H and K of targets[body]: the body's distinct heads in clause
        # order and the sorted windows, as in `build_features`.
        "heads": {body: list(dict.fromkeys(heads)) for body, heads in body_groups.items()},
        "windows": sorted(set(cfg["delay_windows"])),
        "features": features,
        "targets": targets,
        "rows": rows,
    }
    # examples = prune_examples(examples, cfg["min_support"])
    
    # print summary of the training data
    # features[body] / targets[body]: per-video lists of (R, s + 2, A) / (R, H, K) tensors.
    num_bodies = sum(1 for F in features.values() if F)
    num_rows = sum(len(F) for Fs in features.values() for F in Fs)
    num_features = sum(Fs[0][0].numel() for Fs in features.values() if Fs)
    num_heads = sum(Ts[0][0].numel() for Ts in targets.values() if Ts)

    print(f"[Videos] \t {len(atoms)}")
    print(f"[Bodies]\t {num_bodies}")
    print(f"[Heads]\t\t {num_heads}")
    print(f"[Examples]\t {num_rows}")
    print(f"[Features]\t {num_features}")

    return examples

def learn_extension_masks(
    X, Y, num_epochs=200, lr=0.1, l0_lambda=1e-3, device='cpu', batch_size=None,
    mask_patience=20, check_every=5, gate_tol=1e-3, init_log_alpha=-5.0,
):
    # X: (N, M) 0/1 tensor, Y: (N, H) 0/1 tensor; one rule (gate vector) per
    # head column of Y, all trained together. Heads are independent: the loss
    # is a sum over heads, so each head's gradient is the same as alone.
    # The rule fires with s = prod_j (1 - w_j (1 - x_j)), w = sigmoid(log_alpha).
    # Since x_j is 0/1, log(1 - w_j (1 - x_j)) = (1 - x_j) log(1 - w_j), so
    # log s = (1 - X) @ log(1 - w): one (N, M) x (M, H) matmul for all heads.
    # Training avoids the ways all gates collapse to "off":
    #   - gates are deterministic and start nearly off (`init_log_alpha`), so
    #     the empty rule gives s ~= 1 and each feature is switched on only if
    #     it helps. With w ~= 0.5, or with stochastic hard-concrete gates (which
    #     randomly turn on ~20% of M gates per step), the product over hundreds
    #     of absent features underflows to 0 for every row;
    #   - the loss is computed from log s, never clamping s, so the gradient
    #     survives even when s is tiny.
    # Positives are up-weighted per head to balance rare heads, and `l0_lambda`
    # is the cost of each selected atom (penalty = sum of w), independent of M.
    # (1 - X) is uploaded to `device` once if it fits (`batch_size=None`
    # auto-sizes from free GPU memory), otherwise moved in row batches.
    # The output is the hard mask w > 0.5, so training stops once that mask has
    # not changed for `mask_patience` epochs and no gate moved by more than
    # `gate_tol` since the last check (gates start off and rise slowly, so an
    # unchanged mask alone would stop too early). Checked every `check_every`
    # epochs to avoid a GPU sync per step.
    N, M = X.shape
    H = Y.shape[1]
    if batch_size is None:
        batch_size = _auto_batch_size(M + H, device)
    X_off = 1 - X.detach().to('cpu', torch.float32)
    Y = Y.detach().to('cpu', torch.float32)
    num_positives = Y.sum(dim=0)
    pos_weight = torch.where(
        num_positives > 0, (N - num_positives) / num_positives.clamp(min=1), torch.ones_like(num_positives)
    )
    weights = 1 + (pos_weight - 1) * Y

    whole_dataset_fits = batch_size >= N
    if whole_dataset_fits:
        X_off = X_off.to(device)
        Y = Y.to(device)
        weights = weights.to(device)

    log_alpha = torch.full((M, H), init_log_alpha, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([log_alpha], lr=lr)
    previous_w = None
    epochs_unchanged = 0
    for epoch in range(num_epochs):
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            if whole_dataset_fits:
                X_batch, Y_batch, W_batch = X_off, Y, weights
            else:
                X_batch = X_off[start:end].to(device, non_blocking=True)
                Y_batch = Y[start:end].to(device, non_blocking=True)
                W_batch = weights[start:end].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            w = torch.sigmoid(log_alpha)
            # The factor is capped below 1 so a fully-on gate stays finite.
            log_s = X_batch @ torch.log1p(-w.clamp(max=1 - 1e-6))  # (batch, H)
            # log(1 - s); the epsilon keeps it finite (and its gradient
            # alive) at s = 1, which is where training starts.
            log_not_s = torch.log(-torch.expm1(log_s) + 1e-6)
            bce = -(W_batch * (Y_batch * log_s + (1 - Y_batch) * log_not_s)).sum(dim=0) / W_batch.sum(dim=0)
            loss = bce.sum() + l0_lambda * w.sum()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % check_every == 0:
            with torch.no_grad():
                w = torch.sigmoid(log_alpha)
                settled = previous_w is not None and bool(
                    torch.equal(w > 0.5, previous_w > 0.5)
                    and (w - previous_w).abs().max() < gate_tol
                )
            if settled:
                epochs_unchanged += check_every
                if epochs_unchanged >= mask_patience:
                    break
            else:
                epochs_unchanged = 0
            previous_w = w

    final_log_alpha = log_alpha.detach().to('cpu')
    final_mask = torch.sigmoid(final_log_alpha) > 0.5
    del log_alpha, optimizer, X_off, Y, weights
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_mask, final_log_alpha




def _evaluate_extension(feature_block, target_col, selected_feature_positions):
    """Compute hard support/coverage/confidence/recall for a candidate extension.

    feature_block: dense (N, M) 0/1 array for this group's rows/features.
    target_col: dense (N,) 0/1 array for the candidate head within this group.
    selected_feature_positions: indices (into the group's local feature axis)
        selected by the learned mask.
    """
    base_rows = feature_block.shape[0]
    base_head_count = int(target_col.sum())
    base_confidence = base_head_count / base_rows if base_rows else 0.0

    if len(selected_feature_positions) == 0:
        extended_mask = np.ones(base_rows, dtype=bool)
    else:
        extended_mask = np.all(
            feature_block[:, selected_feature_positions] == 1, axis=1
        )

    support = int(extended_mask.sum())
    if support == 0:
        return {
            "support": 0, "coverage": 0.0, "confidence": 0.0,
            "recall": 0.0, "confidence_gain": -base_confidence, "f1": 0.0,
        }

    head_hits = int(target_col[extended_mask].sum())
    confidence = head_hits / support
    coverage = support / base_rows if base_rows else 0.0
    recall = head_hits / base_head_count if base_head_count else 0.0
    f1 = (
        2 * confidence * recall / (confidence + recall)
        if (confidence + recall) > 0 else 0.0
    )
    return {
        "support": support,
        "coverage": coverage,
        "confidence": confidence,
        "recall": recall,
        "confidence_gain": confidence - base_confidence,
        "f1": f1,
    }

def _atom_name(signature):
    # ["pred", "action", "action_id", 3, "action_id_label", "MovAway",
    #  "agent_class", 1, "agent_class_label", "Car"] -> "Car:MovAway"
    fields = dict(zip(signature[2::2], signature[3::2]))
    labels = [str(v) for k, v in fields.items() if k.endswith("_label") and k != "agent_class_label"]
    return f"{fields.get('agent_class_label', '?')}:{'/'.join(labels) or signature[1]}"


def _clause_text(clause, unique_atoms):
    # Object slot s is the variable X{s}; slot 0 is the head object.
    def atom(atom_index, slot):
        name = _atom_name(unique_atoms[atom_index]) if unique_atoms else f"a{atom_index}"
        return f"{name}(X{slot})"
    body = " ∧ ".join(atom(a, s) for a, s in clause["bodies"])
    return f"{body} ⇒ starts {atom(*clause['head'])} within {clause['delay_window']}f"


def _summarize_extended_clauses(extended_clauses, unique_atoms, output_dir, num_examples=15):
    """Print statistics of the extended clauses and save them as a PDF figure."""
    print(f"[Extended clauses] {len(extended_clauses)} clauses")
    if not extended_clauses:
        return
    body_lengths = np.array([len(c["bodies"]) for c in extended_clauses])
    windows = np.array([c["delay_window"] for c in extended_clauses])
    metrics = {k: np.array([c[k] for c in extended_clauses]) for k in ("support", "confidence", "recall", "confidence_gain", "f1")}
    num_bodies = len({tuple(c["bodies"][0]) for c in extended_clauses})
    num_heads = len({tuple(c["head"]) for c in extended_clauses})
    print(f"[Extended clauses] {num_bodies} anchor bodies, {num_heads} heads")
    lengths, length_counts = np.unique(body_lengths, return_counts=True)
    print("[Extended clauses] body length: " + ", ".join(f"{l}: {n}" for l, n in zip(lengths, length_counts)))
    window_values, window_counts = np.unique(windows, return_counts=True)
    print("[Extended clauses] delay window: " + ", ".join(f"{w}: {n}" for w, n in zip(window_values, window_counts)))
    for k, v in metrics.items():
        print(f"[Extended clauses] {k}: mean {v.mean():.3f}, median {np.median(v):.3f}, max {v.max():.3f}")
    top = sorted(extended_clauses, key=lambda c: c["f1"], reverse=True)[:num_examples]
    print(f"[Extended clauses] top {len(top)} by f1:")
    for c in top:
        print(f"  f1={c['f1']:.3f} conf={c['confidence']:.3f} sup={c['support']}  {_clause_text(c, unique_atoms)}")

    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_file = Path(output_dir) / "extended_clauses_summary.pdf"
    with PdfPages(pdf_file) as pdf:
        # Page 1: distributions.
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"{len(extended_clauses)} extended clauses ({num_bodies} anchor bodies, {num_heads} heads)")
        axes[0, 0].bar(lengths, length_counts)
        axes[0, 0].set(title="Body length (anchor + extra atoms)", xlabel="atoms", ylabel="clauses", xticks=lengths)
        axes[0, 1].bar([str(w) for w in window_values], window_counts)
        axes[0, 1].set(title="Delay window", xlabel="frames", ylabel="clauses")
        axes[0, 2].hist(metrics["support"], bins=30, log=True)
        axes[0, 2].set(title="Support", xlabel="rows", ylabel="clauses (log)")
        for ax, k in zip(axes[1], ("confidence", "confidence_gain", "f1")):
            ax.hist(metrics[k], bins=30)
            ax.set(title=k.replace("_", " ").capitalize(), ylabel="clauses")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: confidence vs. recall, coloured by body length.
        fig, ax = plt.subplots(figsize=(8, 6))
        points = ax.scatter(metrics["recall"], metrics["confidence"], c=body_lengths, s=12, cmap="viridis")
        fig.colorbar(points, ax=ax, label="body length")
        ax.set(title="Confidence vs. recall", xlabel="recall", ylabel="confidence")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: example clauses.
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        ax.set_title(f"Top {len(top)} clauses by f1", loc="left")
        lines = []
        for i, c in enumerate(top, 1):
            header = f"{i:>2}. f1={c['f1']:.2f} conf={c['confidence']:.2f} recall={c['recall']:.2f} gain={c['confidence_gain']:+.2f} support={c['support']}"
            lines.append(header)
            lines.extend("      " + line for line in textwrap.wrap(_clause_text(c, unique_atoms), 120))
        ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=7, transform=ax.transAxes)
        pdf.savefig(fig)
        plt.close(fig)
    print(f"[Extended clauses] Summary figure saved to {pdf_file}")


def _to_extended_clauses(extended_data, delay_windows, unique_atoms=None, output_dir=None):
    extended_clauses = []
    for item in extended_data:
        # Bodies are ungrounded (atom_index, object_slot) atoms, the anchor
        # body first; body_scopes lists each body's object slot.
        bodies = [tuple(item["body"])] + [(atom, slot) for slot, atom in item["extra_atoms"]]
        extended_clauses.append({
            "bodies": bodies,
            "body_scopes": [slot for _atom, slot in bodies],
            "head": tuple(item["head"]),
            # The head starts within delay_window frames; delay_windows are all
            # windows learned, needed to decode the cumulative heads.
            "delay_window": item["delay_window"],
            "delay_windows": list(delay_windows),
            **item["metrics"],
        })
    if output_dir is not None:
        _summarize_extended_clauses(extended_clauses, unique_atoms, output_dir)
    return extended_clauses


def _learnable_heads(targets, body):
    # Skip degenerate targets: nothing to learn if the head never fires
    # or always fires for this body.
    num_rows, num_head_cols = targets.shape
    head_positives = targets.sum(dim=0)
    learnable_heads = torch.nonzero(
        (head_positives > 0) & (head_positives < num_rows), as_tuple=True
    )[0].tolist()
    return learnable_heads


def _accepted_metrics(feature_block, target_col, selected_positions, body, head_col, cfg):
    # Hard metrics of the learned extension, or None if it is rejected.
    if not selected_positions:
        return None

    metrics = _evaluate_extension(feature_block, target_col, selected_positions)
    if metrics["support"] < cfg["min_support"]:
        return None
    if metrics["confidence_gain"] < cfg["min_confidence_gain"]:
        return None
    return metrics


def _extend_body(body, features, targets, heads, windows, cfg):
    """Differentiable conjunctive rule learning for one body; returns the
    accepted extensions of its (head, window) pairs.

    features, targets: per-video lists of (R, s + 2, A) / (R, H, K) bool tensors
    (`build_body_features`); heads, windows: the H heads and K windows.
    """
    if not features:
        return []

    # Flatten the role slots and the (head, window) pairs into columns, on CPU;
    # `learn_extension_masks` moves them to the device. Feature column
    # slot * A + atom is "agent in role slot holds atom"; target column
    # h * K + k is "heads[h] starts within windows[k] frames".
    F = torch.cat(features).cpu()
    num_rows, num_slots, num_atoms = F.shape
    X = F.reshape(num_rows, num_slots * num_atoms)
    Y = torch.cat(targets).cpu().reshape(num_rows, -1)
    feature_block = X.numpy()

    learnable_heads = _learnable_heads(Y, body)
    if not learnable_heads:
        return []

    # All learnable heads of the body are trained together.
    final_masks, _log_alpha = learn_extension_masks(
        X, Y[:, learnable_heads],
        num_epochs=cfg["num_epochs"], lr=cfg["lr"], l0_lambda=cfg["l0_lambda"],
        device=cfg["device"],
    )

    body_data = []
    for mask_col, head_col in enumerate(learnable_heads):
        target_col = Y[:, head_col].numpy().astype(np.float32)
        selected_positions = torch.nonzero(final_masks[:, mask_col], as_tuple=True)[0].tolist()
        metrics = _accepted_metrics(
            feature_block, target_col, selected_positions, body, head_col, cfg
        )
        if metrics is None:
            continue

        h, k = divmod(head_col, len(windows))
        body_data.append({
            "body": body,
            "head": heads[h],
            "delay_window": windows[k],
            # (object_slot, atom_index) of each extra atom.
            "extra_atoms": [divmod(pos, num_atoms) for pos in selected_positions],
            "metrics": metrics,
        })
    return body_data


_WANDB_CLOUD_HOST = "https://api.wandb.ai"


def _wandb_login(wandb):
    """Log in to the hosted W&B cloud; returns False if the user skips tracking.

    Asks the user to pick a login method: the saved credentials (~/.netrc or
    WANDB_API_KEY) or an API key typed in (hidden), from
    https://wandb.ai/authorize. Without a terminal, only saved credentials are tried.
    """
    if not sys.stdin.isatty():
        return wandb.login(host=_WANDB_CLOUD_HOST)
    print(
        "[wandb] Choose a login method for https://wandb.ai:\n"
        "  1) use saved credentials (~/.netrc or WANDB_API_KEY)\n"
        "  2) enter an API key (from https://wandb.ai/authorize)\n"
        "  3) skip wandb tracking"
    )
    choice = input("[wandb] Select 1/2/3 [1]: ").strip() or "1"
    if choice == "3":
        return False
    if choice == "1" and wandb.login(host=_WANDB_CLOUD_HOST, relogin=False, verify=True):
        return True
    if choice == "1":
        print("[wandb] No valid saved credentials, enter an API key instead.")
    key = getpass.getpass("[wandb] API key (input hidden): ").strip()
    if not key:
        return False
    # The key is verified against the server and saved to ~/.netrc for later runs.
    return wandb.login(key=key, host=_WANDB_CLOUD_HOST, relogin=True, verify=True)


def _init_wandb(cfg, num_bodies):
    """Online run on the hosted W&B cloud, or None if tracking is off or fails.

    Tracking is fail-open: a missing package, skipped or failed login, or
    network error only prints a warning.
    """
    if not cfg.get("wandb", True):
        return None
    try:
        import wandb
        # Always the hosted cloud, never a stale WANDB_BASE_URL of a local server.
        os.environ["WANDB_BASE_URL"] = _WANDB_CLOUD_HOST
        if not _wandb_login(wandb):
            print("[wandb] Tracking disabled: not logged in.")
            return None
        hparams = ("num_epochs", "lr", "l0_lambda", "min_support", "min_confidence_gain", "delay_windows", "device")
        run = wandb.init(
            project=cfg.get("wandb_project", "cauvid-roadpp"),
            name=cfg.get("wandb_run_name"),
            job_type="step05_extension",
            mode="online",
            config={**{k: cfg[k] for k in hparams if k in cfg}, "num_bodies": num_bodies},
        )
        print(f"[wandb] Run: {run.url}")
        return run
    except Exception as exc:
        print(f"[wandb] Tracking disabled: {type(exc).__name__}: {exc}")
        return None


def _learn_extensions(examples, cfg):
    num_bodies = len(examples["features"])
    run = _init_wandb(cfg, num_bodies)
    extended_data = []
    start_time = time.perf_counter()
    try:
        for i, (body, features) in enumerate(tqdm(examples["features"].items(), desc="Learning rule extensions")):
            body_start = time.perf_counter()
            body_data = _extend_body(
                body, features, examples["targets"][body],
                examples["heads"][body], examples["windows"], cfg,
            )
            extended_data.extend(body_data)
            # The body's dense blocks are freed on return.
            if cfg["device"] != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if run is not None:
                run.log({
                    "progress/bodies_done": i + 1,
                    "progress/fraction": (i + 1) / num_bodies,
                    "progress/elapsed_s": time.perf_counter() - start_time,
                    "body/rows": sum(len(F) for F in features),
                    "body/head_cols": len(examples["heads"][body]) * len(examples["windows"]),
                    "body/accepted": len(body_data),
                    "body/seconds": time.perf_counter() - body_start,
                    "total/accepted": len(extended_data),
                }, step=i)
        if run is not None:
            import wandb
            table = wandb.Table(columns=[
                "body", "head", "delay_window", "extra_atoms",
                "support", "confidence", "recall", "confidence_gain", "f1",
            ])
            for item in extended_data:
                m = item["metrics"]
                table.add_data(
                    str(item["body"]), str(item["head"]), item["delay_window"], str(item["extra_atoms"]),
                    m["support"], m["confidence"], m["recall"], m["confidence_gain"], m["f1"],
                )
            run.log({"extensions": table})
            run.summary["total/accepted"] = len(extended_data)
            run.summary["total/seconds"] = time.perf_counter() - start_time
    finally:
        if run is not None:
            run.finish()
    return extended_data


def extend_clauses(cfg):
    # load necessary data files
    labels = utils_data.load_json(cfg['dataset_label_file'])
    atoms = utils_data.load_json(cfg['train_atom_file'])
    causal_anchor_clauses = utils_data.load_json(cfg['causal_anchor_clause_file'])
    unique_atoms = utils_data.load_json(cfg['unique_atoms_file'])
    events = torch.load(cfg['events_file'])
    print(f"Loaded causal anchor clauses: {len(causal_anchor_clauses)}")
    
    # Build training examples from the loaded anchor clauses, labels, and atoms.
    if not os.path.exists(cfg["extension_training_example_file"]):
        examples = build_training_examples(atoms, causal_anchor_clauses, labels, unique_atoms, cfg)
        torch.save(examples, cfg["extension_training_example_file"])
    else:
        examples = torch.load(cfg["extension_training_example_file"])

    extended_data = _learn_extensions(examples, cfg)
    extended_clauses = _to_extended_clauses(extended_data, cfg["delay_windows"], unique_atoms, cfg["output_dir"])
    return extended_clauses





def main(input_data):
    print(f"----------------- Step 05: Rule Extension -----------------")
    cfg = {
            'output_dir': input_data['output_dir'],
            'device': input_data['device'],
            'min_support': 10,
            'min_confidence_gain': 0.05,
            'num_epochs': 200,
            'lr': 0.1,
            'l0_lambda': 1e-3,
            'delay_windows': list(input_data['delay_windows']),
            'dataset_label_file': input_data['dataset_label_file'],
            'train_atom_file': input_data['train_atom_file'],
            'causal_anchor_clause_file': input_data['causal_anchor_clause_file'],
            'anchor_clause_file': input_data['anchor_clause_file'],
            'extension_training_example_file': input_data["output_dir"] / "training_examples.pt",
            "unique_atoms_file": input_data["unique_atoms_file"],
            "events_file": input_data["events_file"]
    }
    extend_clause_file = cfg['output_dir'] / "extended_clauses.json"
    if not extend_clause_file.exists():
        extended_clauses = extend_clauses(cfg)
        utils_data.save_json(extended_clauses, extend_clause_file)
        print(f"[Rule extension] Learned {len(extended_clauses)} extended clauses")