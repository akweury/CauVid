

from pathlib import Path
import os
import json
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

def _index_frame_entities(atoms):
    """Give every (frame pair, entity) a node id and list its atoms.

    Node ids are contiguous per frame pair, so nodes of one frame pair form a
    consecutive range. Returns long tensors:
      node_frame[n]: frame pair of node n;
      current_cells / next_cells: (node, atom_index) of every atom in frame
          t / t+1.
    """
    node_frame = []
    current_nodes, current_atom_ids = [], []
    next_nodes, next_atom_ids = [], []
    frame_id = 0
    for video_id, video_atoms in tqdm(atoms.items(), desc="Indexing frames"):
        if video_id == "unique_atoms":
            continue
        for current_frame, next_frame in video_atoms["per_frame_atoms"]:
            if len(current_frame) == 0 or len(next_frame) == 0:
                continue
            base = len(node_frame)
            entity_nodes = {}
            for frame_atoms, nodes, atom_ids in (
                (current_frame, current_nodes, current_atom_ids),
                (next_frame, next_nodes, next_atom_ids),
            ):
                for atom in frame_atoms:
                    node = entity_nodes.setdefault(
                        _atom_entity_key(atom), base + len(entity_nodes)
                    )
                    nodes.append(node)
                    atom_ids.append(atom[0])
            node_frame.extend([frame_id] * len(entity_nodes))
            frame_id += 1
    return (
        torch.tensor(node_frame, dtype=torch.long),
        torch.tensor([current_nodes, current_atom_ids], dtype=torch.long),
        torch.tensor([next_nodes, next_atom_ids], dtype=torch.long),
    )


def build_features(atoms, dataset_labels, body_groups, head_vocab,
                   device="cpu", chunk_rows=1 << 18):
    """Build sparse 0/1 feature/target matrices, one pair per body group.

    A group is keyed by its ungrounded anchor body (atom_index, "oid", oid);
    heads are always (atom_index, "oid", 0) (see `Clause`). A row is one
    grounding of the clause's oids to entities in frame t:
      - the body oid is bound to the entity of a matching atom in frame t;
      - oid 0 (the head object) is that same entity if the body oid is 0,
        otherwise each other entity present in frame t+1 (distinct oids bind
        distinct objects, as in step03's anchor clauses).
      features[g]: (N_g, M_g); column j = feature_vocab[g][j], a
          co-occurring atom (group_id, object_slot, atom_index) in frame t.
          object_slot is 0 for the head object, the body oid for the body
          object, and body oid + 1 for "some other object in the frame"
          (all unbound entities share this one existential slot).
      targets[g]:  (N_g, H_g); column k = head_vocab[g][k], True if that
          head atom holds for the oid-0 entity in frame t+1.
    Both are stored in the compact CSR form of `_to_sparse_rows`; use
    `sparse_rows_to_dense` to get a dense tensor for one group.
    Columns are local to each group (only columns True in some row are kept),
    so no matrix carries other groups' zeros.

    Every entity's atoms are held as one dense bool row of a (nodes, atoms)
    table on `device`, so each group's rows are built with tensor indexing in
    chunks of `chunk_rows`.
    """
    # body_signature -> group_id, same order as build_head_vocab's enumerate.
    num_groups = len(body_groups)
    group_ids = {body_signature: gid for gid, body_signature in enumerate(body_groups)}
    group_head_vocab = [[] for _ in range(num_groups)]
    for body_signature, head_signature in head_vocab:
        gid = group_ids[body_signature]
        group_head_vocab[gid].append((gid, head_signature))

    num_atoms = len(atoms["unique_atoms"])
    node_frame, current_cells, next_cells = _index_frame_entities(atoms)
    num_nodes = len(node_frame)
    num_frames = int(node_frame[-1]) + 1 if num_nodes else 0

    # current[n, a] / upcoming[n, a]: atom a holds for node n in frame t / t+1.
    node_frame = node_frame.to(device)
    current = torch.zeros((num_nodes, num_atoms), dtype=torch.bool, device=device)
    current[current_cells[0].to(device), current_cells[1].to(device)] = True
    upcoming = torch.zeros((num_nodes, num_atoms), dtype=torch.bool, device=device)
    upcoming[next_cells[0].to(device), next_cells[1].to(device)] = True
    # frame_counts[f, a]: number of entities in frame pair f with atom a at t.
    cell_nodes, cell_atoms = torch.nonzero(current, as_tuple=True)
    frame_counts = torch.zeros((num_frames, num_atoms), dtype=torch.int32, device=device)
    frame_counts.index_put_(
        (node_frame[cell_nodes], cell_atoms),
        torch.ones(len(cell_nodes), dtype=torch.int32, device=device),
        accumulate=True,
    )
    del cell_nodes, cell_atoms
    # Candidate head objects: nodes with some atom at t+1, grouped by frame.
    has_next = upcoming.any(dim=1)
    next_nodes = torch.nonzero(has_next, as_tuple=True)[0]
    next_counts = torch.bincount(node_frame[next_nodes], minlength=num_frames)
    next_starts = torch.cumsum(next_counts, dim=0) - next_counts

    features = []
    targets = []
    group_feature_vocab = []
    for body_signature, group_id in tqdm(group_ids.items(), desc="Building features"):
        body_index, _oid_tag, body_oid = body_signature
        other_slot = body_oid + 1
        num_columns = (other_slot + 1) * num_atoms
        head_atoms = torch.tensor(
            [head_signature[0] for _gid, head_signature in group_head_vocab[group_id]],
            dtype=torch.long, device=device,
        )

        # Rows: (body node, head node) groundings of this group.
        body_nodes = torch.nonzero(current[:, body_index], as_tuple=True)[0]
        if body_oid == 0:
            body_nodes = body_nodes[has_next[body_nodes]]
            row_body = row_head = body_nodes
        else:
            body_frames = node_frame[body_nodes]
            counts = next_counts[body_frames]
            row_body = torch.repeat_interleave(body_nodes, counts)
            within = torch.arange(len(row_body), device=device) - torch.repeat_interleave(
                torch.cumsum(counts, dim=0) - counts, counts
            )
            row_head = next_nodes[
                torch.repeat_interleave(next_starts[body_frames], counts) + within
            ]
            keep = row_head != row_body
            row_body, row_head = row_body[keep], row_head[keep]
        num_rows = len(row_body)

        feature_rows, feature_columns = [], []
        target_rows, target_columns = [], []
        for start in range(0, num_rows, chunk_rows):
            body = row_body[start:start + chunk_rows]
            head = row_head[start:start + chunk_rows]
            body_atoms = current[body]
            head_atoms_t = current[head]
            others = frame_counts[node_frame[body]] - body_atoms.int()
            if body_oid != 0:
                others -= head_atoms_t.int()
            block = torch.zeros(
                (len(body), other_slot + 1, num_atoms), dtype=torch.bool, device=device
            )
            block[:, 0] = head_atoms_t
            block[:, body_oid] |= body_atoms
            # Skip the grounded body itself; it is implied by the group.
            block[:, body_oid, body_index] = False
            block[:, other_slot] = others > 0
            rows, columns = torch.nonzero(block.reshape(len(body), -1), as_tuple=True)
            feature_rows.append((rows + start).cpu())
            feature_columns.append(columns.cpu())

            rows, columns = torch.nonzero(upcoming[head][:, head_atoms], as_tuple=True)
            target_rows.append((rows + start).cpu())
            target_columns.append(columns.cpu())

        feature_rows = torch.cat(feature_rows) if feature_rows else torch.zeros(0, dtype=torch.long)
        feature_columns = torch.cat(feature_columns) if feature_columns else torch.zeros(0, dtype=torch.long)
        target_rows = torch.cat(target_rows) if target_rows else torch.zeros(0, dtype=torch.long)
        target_columns = torch.cat(target_columns) if target_columns else torch.zeros(0, dtype=torch.long)

        # Keep only columns True in some row, renumbered in (slot, atom) order.
        used_columns = torch.unique(feature_columns)
        local_ids = torch.full((num_columns,), -1, dtype=torch.long)
        local_ids[used_columns] = torch.arange(len(used_columns))
        group_feature_vocab.append([
            (group_id, column // num_atoms, column % num_atoms)
            for column in used_columns.tolist()
        ])
        features.append(_csr_from_cells(
            feature_rows, local_ids[feature_columns], (num_rows, len(used_columns)),
        ))
        targets.append(_csr_from_cells(
            target_rows, target_columns, (num_rows, len(group_head_vocab[group_id])),
        ))

    del current, upcoming, frame_counts
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "features": features,
        "targets": targets,
        "feature_vocab": group_feature_vocab,
        "head_vocab": group_head_vocab,
    }



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


def build_head_vocab(body_groups):
    head_vocab = []
    for body_id, (body_signature, body_group) in enumerate(body_groups.items()):
        for head_signature in body_group["candidate_heads"]:
            head_vocab.append((body_signature, head_signature))
    return head_vocab

    
def _decode_examples(compact, signatures):
    """Inverse of `_encode_examples`."""
    body_groups = {}
    for body_id, head_ids in zip(compact["body_ids"].tolist(), compact["candidate_head_ids"]):
        body_groups[signatures[body_id]] = {
            "candidate_heads": tuple(signatures[h] for h in head_ids.tolist())
        }
    head_vocab = build_head_vocab(body_groups)
    group_head_vocab = [[] for _ in range(len(body_groups))]
    for gid, head_signature in head_vocab:
        group_head_vocab[gid].append((gid, head_signature))
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




def group_clauses_by_body(clauses, dataset_labels):
    clauses_by_body = {}
    for c in clauses.values():
        anchor_body = tuple(c['anchor']["body"][0])
        body_group = clauses_by_body.setdefault(
            anchor_body, {"candidate_heads": set()}
        )
        body_group["candidate_heads"].add(tuple(c['anchor']["head"]))
        

    for body_group in clauses_by_body.values():
        body_group["candidate_heads"] = tuple(
            sorted(body_group["candidate_heads"], key=repr)
        )

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

def build_training_examples(atoms, clauses, labels,cfg):   
    body_groups = group_clauses_by_body(clauses, labels)
    head_vocab = build_head_vocab(body_groups)
    examples = build_features(atoms, labels, body_groups, head_vocab, device=cfg["device"])
    examples["body_groups"] = body_groups
    # examples = prune_examples(examples, cfg["min_support"])
    
    # print summary of the training data
    num_rows = sum(X["shape"][0] for X in examples["features"])
    num_features = sum(X["shape"][1] for X in examples["features"])
    num_heads = sum(Y["shape"][1] for Y in examples["targets"])
    print(
        f"[Training data] {len(examples['body_groups'])} groups, {num_rows} examples, "
        f"{num_features} features, {num_heads} candidate heads"
    )

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

def _to_extended_clauses(extended_data, training_examples_by_body):
    body_groups = training_examples_by_body["body_groups"]
    extended_clauses = []
    group_bodies = list(body_groups.keys())
    for item in extended_data:
        # Bodies are ungrounded (atom_index, "oid", oid) atoms; a feature
        # (gid, object_slot, atom_index) is the body atom on oid object_slot
        # (see `build_features`). body_scopes lists each body's oid.
        group_body = tuple(group_bodies[item['group_id']])
        bodies = [group_body]
        body_scopes = [group_body[2]]
        for _gid, object_slot, atom_index in item['extra_atoms']:
            bodies.append((atom_index, "oid", object_slot))
            body_scopes.append(object_slot)
        head = tuple(item['head'][1])
        extended_clause = {
            "bodies": bodies,
            "body_scopes": body_scopes,
            "head": head,
            "support": item["metrics"]["support"],
            "coverage": item["metrics"]["coverage"],
            "confidence": item["metrics"]["confidence"],
            "recall": item["metrics"]["recall"],
            "confidence_gain": item["metrics"]["confidence_gain"],
            "f1": item["metrics"]["f1"],
        }
        extended_clauses.append(extended_clause)
    return extended_clauses




def extend_clauses(anchor_clauses, labels, atoms, cfg):
    
    # prepare the training examples for rule extension
    training_file = cfg["output_dir"] / "training_examples.pt"
    if training_file.exists():
        examples = torch.load(training_file)
    else:
        examples = build_training_examples(atoms, anchor_clauses, labels, cfg)
        torch.save(examples, training_file)

    features = examples["features"]
    targets = examples["targets"]
    feature_vocab = examples["feature_vocab"]
    head_vocab = examples["head_vocab"]
    # Only pruned examples carry aliases; otherwise each column is its own.
    feature_aliases = examples.get(
        "feature_aliases", [[[entry] for entry in vocab] for vocab in feature_vocab]
    )
    device = cfg["device"]
    num_epochs = cfg["num_epochs"]
    lr = cfg["lr"]
    l0_lambda = cfg["l0_lambda"]
    min_support = cfg["min_support"]
    min_confidence_gain = cfg["min_confidence_gain"]

    extended_data = []
    # extension in one shot: differentiable conjunctive rule learning, group by group
    for gid in tqdm(range(len(examples["features"])), desc="Learning rule extensions"):
        num_rows, num_feature_cols = examples["features"][gid]["shape"]
        num_head_cols = examples["targets"][gid]["shape"][1]
        if num_rows == 0 or num_feature_cols == 0 or num_head_cols == 0:
            print(
                f"[Rule extension] Skipping group {gid}: "
                f"rows={num_rows}, feature_cols={num_feature_cols}, "
                f"head_cols={num_head_cols} "
                "(nothing to learn for this body-group)."
            )
            continue

        # Densify only this group's (N, M) block, kept on CPU; it is moved to
        # the GPU once inside `learn_extension_masks`. Numpy view of the bool
        # block for hard evaluation.
        dense_features = sparse_rows_to_dense(features[gid])
        feature_block = dense_features.numpy()
        dense_targets = sparse_rows_to_dense(targets[gid])

        # Skip degenerate targets: nothing to learn if the head never fires
        # or always fires for this body-group.
        head_positives = dense_targets.sum(dim=0)
        learnable_heads = torch.nonzero(
            (head_positives > 0) & (head_positives < num_rows), as_tuple=True
        )[0].tolist()
        num_degenerate = num_head_cols - len(learnable_heads)
        if num_degenerate:
            print(
                f"[Rule extension] Skipping {num_degenerate}/{num_head_cols} heads of "
                f"group {gid}: degenerate target (head fires in no row or every row)."
            )
        if not learnable_heads:
            del dense_features, feature_block, dense_targets
            continue

        # All learnable heads of the group are trained together.
        final_masks, _log_alpha = learn_extension_masks(
            dense_features, dense_targets[:, learnable_heads],
            num_epochs=num_epochs, lr=lr, l0_lambda=l0_lambda, device=device,
        )

        for mask_col, head_col in enumerate(learnable_heads):
            target_col = dense_targets[:, head_col].numpy().astype(np.float32)
            selected_positions = torch.nonzero(final_masks[:, mask_col], as_tuple=True)[0].tolist()
            if not selected_positions:
                print(
                    f"[Rule extension] Skipping group {gid}, head {head_col}: "
                    "learned mask selected no extra atoms."
                )
                continue

            metrics = _evaluate_extension(feature_block, target_col, selected_positions)
            if metrics["support"] < min_support:
                print(
                    f"[Rule extension] Skipping group {gid}, head {head_col}: "
                    f"support {metrics['support']} < min_support {min_support}."
                )
                continue
            if metrics["confidence_gain"] < min_confidence_gain:
                print(
                    f"[Rule extension] Skipping group {gid}, head {head_col}: "
                    f"confidence_gain {metrics['confidence_gain']:.4f} < "
                    f"min_confidence_gain {min_confidence_gain}."
                )
                continue

            selected_atoms = [feature_vocab[gid][pos] for pos in selected_positions]
            print(f"Extra Atom Num: {len(selected_atoms)}")
            extended_data.append({
                "group_id": gid,
                "head": head_vocab[gid][head_col],
                "extra_atoms": selected_atoms,
                # Atoms true in exactly the same rows as each extra atom.
                "extra_atom_aliases": [feature_aliases[gid][pos] for pos in selected_positions],
                "metrics": metrics,
            })

        del dense_features, feature_block, dense_targets, final_masks
        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    extended_clauses = _to_extended_clauses(extended_data, examples)
    return extended_clauses



def main(input_data):
    cfg = {
        'output_dir': input_data['output_dir'],
        'device': input_data['device'],
        'min_support': 10,
        'min_confidence_gain': 0.05,
        'num_epochs': 200,
        'lr': 0.1,
        'l0_lambda': 1e-3
    }
    print(f"----------------- Step 05: Rule Extension -----------------")
    extend_clause_file = cfg['output_dir'] / "extended_clauses.json"
    if extend_clause_file.exists():
        extended_clauses = utils_data.load_json(extend_clause_file)
        print(f"Loaded extended clauses: {len(extended_clauses)}")
        return extended_clauses
    
    dataset_labels = utils_data.load_json(input_data['dataset_label_file'])
    atoms = utils_data.load_json(input_data['train_atom_file'])
    causal_anchor_clauses = utils_data.load_json(input_data['causal_anchor_clause_file'])
    anchor_clauses = utils_data.load_json(input_data['anchor_clause_file'])
    print(f"Loaded anchor clauses: {len(anchor_clauses['anchor_clauses'])}")
    print(f"Loaded causal anchor clauses: {len(causal_anchor_clauses)}")
    # replace id with label names
    # clause extension
    extended_clauses = extend_clauses(causal_anchor_clauses, dataset_labels, atoms, cfg)
    utils_data.save_json(extended_clauses, extend_clause_file)
    print(f"[Rule extension] Learned {len(extended_clauses)} extended clauses")
    return extended_clauses