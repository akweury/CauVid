
import torch

import json 
from tqdm import tqdm

import numpy as np

from src.exp_roadpp import utils_data


def _atom_entity_key(atom_signature):
    tube_uid = atom_signature[-1]
    agent_class = atom_signature[7]
    if tube_uid is not None:
        return ("tube_uid", tube_uid)
    if agent_class == "av":
        return ("agent_class", "av")
    raise ValueError()


def hard_concrete_mask(log_alpha, beta=2 / 3, gamma=-0.1, zeta=1.1, training=True):
    if training:
        u = torch.rand_like(log_alpha).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    else:
        s = torch.sigmoid(log_alpha)
    s_bar = s * (zeta - gamma) + gamma
    return s_bar.clamp(0, 1)





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
            body_signature, {"candidate_heads": set()}
        )
        body_group["candidate_heads"].add(head_signature)
        

    for body_group in clauses_by_body.values():
        body_group["candidate_heads"] = tuple(
            sorted(body_group["candidate_heads"], key=repr)
        )

    print(f"[Grouping] Clauses grouped by body: {len(clauses)} -> {len(clauses_by_body)}")
    return clauses_by_body

def build_head_vocab(body_groups):
    head_vocab = []
    for group_id, (body_signature, body_group) in enumerate(body_groups.items()):
        for head_signature in body_group["candidate_heads"]:
            head_vocab.append((group_id, head_signature))
    return head_vocab
    
    
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
    for item in extended_data:
        bodies = []
        # Slot 0 marks the clause's own (anchor) object; see object_slot comment
        # in `_build_body_conditioned_training_examples`.
        body_scopes = [0]
        group_body = list(body_groups.keys())[item['group_id']]
        bodies.append(group_body)
        for atom in item['extra_atoms']:
            body = tuple(atom[-1])
            bodies.append(body)
            body_scopes.append(atom[1])
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


