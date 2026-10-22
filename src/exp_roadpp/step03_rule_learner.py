
import torch

import json 
from tqdm import tqdm

import numpy as np

from src.exp_roadpp import utils_data
from src.exp_roadpp.step03_pred_eval import get_video_data

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


def _beta_const_log_term(beta=2 / 3, gamma=-0.1, zeta=1.1):
    # Closed-form constant from Louizos et al. (2017), "Learning Sparse Neural
    # Networks through L0 Regularization": P(gate != 0) = sigmoid(log_alpha - beta*log(-gamma/zeta)).
    return beta * np.log(-gamma / zeta)


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


def learn_extension_mask(
    X, y, num_epochs=200, lr=0.1, l0_lambda=1e-3, device='cpu', batch_size=None,
    early_stop_tol=1e-4, early_stop_patience=30, init_log_alpha=-5.0,
):
    # X: (N, M) float tensor (0/1), y: (N,) float tensor (0/1)
    # The rule fires with s = prod_j (1 - w_j (1 - x_j)), w = sigmoid(log_alpha).
    # Training avoids the ways all gates collapse to "off":
    #   - gates are deterministic and start nearly off (`init_log_alpha`), so
    #     the empty rule gives s ~= 1 and each feature is switched on only if
    #     it helps. With w ~= 0.5, or with stochastic hard-concrete gates (which
    #     randomly turn on ~20% of M gates per step), the product over hundreds
    #     of absent features underflows to 0 for every row;
    #   - the loss is computed from log s, never clamping s, so the gradient
    #     survives even when s is tiny.
    # Positives are up-weighted to balance rare heads, and `l0_lambda` is the
    # cost of each selected atom (penalty = sum of w), independent of M.
    # X/y are kept on CPU and moved to `device` in row batches so that only one
    # (batch_size, M) slice lives on the GPU at a time -- the full (N, M) dense
    # block plus its `torch.prod` intermediate can otherwise exceed GPU memory
    # (e.g. on an 8GB GTX 1080) when N or M is large. `batch_size=None` auto-sizes
    # the batch from the currently attached GPU's free memory, so the same code
    # adapts whether it runs on a GTX 1080 or an A100. If the whole dataset fits
    # in one batch (common on an A100), it is uploaded to the GPU once instead of
    # being re-copied from CPU every epoch, which otherwise dominates runtime.
    X_cpu = X.detach().to('cpu')
    y_cpu = y.detach().to('cpu')
    N, M = X_cpu.shape
    if batch_size is None:
        batch_size = _auto_batch_size(M, device)
    log_alpha = torch.full((M,), init_log_alpha, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([log_alpha], lr=lr)
    num_positives = float(y_cpu.sum())
    pos_weight = (N - num_positives) / num_positives if num_positives > 0 else 1.0
    best_loss = float("inf")
    epochs_without_improvement = 0

    whole_dataset_fits = batch_size >= N
    if whole_dataset_fits:
        try:
            X_resident = X_cpu.to(device)
            y_resident = y_cpu.to(device)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                "[Rule extension] CUDA out of memory while uploading training "
                f"data to device={device} (N={N}, M={M}). Re-run with a "
                "smaller --batch_size, a smaller --num_epochs, or pass "
                "--device cpu."
            )
            raise

    for _ in range(num_epochs):
        epoch_loss = 0.0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            try:
                if whole_dataset_fits:
                    X_batch = X_resident
                    y_batch = y_resident
                else:
                    X_batch = X_cpu[start:end].to(device)
                    y_batch = y_cpu[start:end].to(device)

                optimizer.zero_grad()
                w = torch.sigmoid(log_alpha)
                # log s = sum_j log(1 - w_j (1 - x_j)); the factor is capped
                # below 1 so a fully-on gate on an absent feature stays finite.
                off_factor = (w.unsqueeze(0) * (1 - X_batch)).clamp(max=1 - 1e-6)
                log_s = torch.log1p(-off_factor).sum(dim=1)  # (batch,)
                # log(1 - s); the epsilon keeps it finite (and its gradient
                # alive) at s = 1, which is where training starts.
                log_not_s = torch.log(-torch.expm1(log_s) + 1e-6)
                weights = 1 + (pos_weight - 1) * y_batch
                bce = -(weights * (y_batch * log_s + (1 - y_batch) * log_not_s)).sum() / weights.sum()
                l0_penalty = w.sum()  # soft count of selected atoms
                loss = bce + l0_lambda * l0_penalty
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if whole_dataset_fits:
                    del w, off_factor, log_s, log_not_s, weights, bce, loss
                else:
                    del X_batch, y_batch, w, off_factor, log_s, log_not_s, weights, bce, loss
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    "[Rule extension] CUDA out of memory while training on "
                    f"device={device} with batch_size={batch_size} "
                    f"(N={N}, M={M}). Re-run with a smaller --batch_size, a "
                    "smaller --num_epochs, or pass --device cpu."
                )
                raise

        # Early stopping: stop once the per-epoch loss stops improving by more
        # than `early_stop_tol` for `early_stop_patience` consecutive epochs.
        # This avoids wasting epochs after convergence, which otherwise
        # dominates runtime now that the whole dataset can live on the GPU.
        if best_loss - epoch_loss > early_stop_tol:
            best_loss = epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                break

    final_mask = (torch.sigmoid(log_alpha) > 0.5)
    final_mask = final_mask.detach().to('cpu')
    final_log_alpha = log_alpha.detach().to('cpu')
    del log_alpha, X_cpu, y_cpu
    if whole_dataset_fits:
        del X_resident, y_resident
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_mask, final_log_alpha


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


def sparse_rows_to_dense(sparse, dtype=torch.bool):
    """Inverse of `_to_sparse_rows`: return the dense (N, M) tensor."""
    num_rows, num_cols = sparse["shape"]
    crow = sparse["crow"].long()
    rows = torch.repeat_interleave(torch.arange(num_rows), crow[1:] - crow[:-1])
    dense = torch.zeros((num_rows, num_cols), dtype=dtype)
    dense[rows, sparse["col"].long()] = 1
    return dense


def build_features(atoms, dataset_labels, body_groups, head_vocab):
    """Build sparse 0/1 feature/target matrices, one pair per body group.

    For group g, one row = one occurrence of g's body atom in frame t.
      features[g]: (N_g, M_g); column j = feature_vocab[g][j], a
          co-occurring atom (group_id, object_slot, fact) in frame t.
      targets[g]:  (N_g, H_g); column k = head_vocab[g][k], True if that
          head fact holds for the same entity in frame t+1.
    Both are stored in the compact CSR form of `_to_sparse_rows`; use
    `sparse_rows_to_dense` to get a dense tensor for one group.
    Columns are local to each group, so no matrix carries other groups' zeros.
    """
    # body_signature -> group_id, same order as build_head_vocab's enumerate.
    body_ids = {body_signature: gid for gid, body_signature in enumerate(body_groups)}
    num_groups = len(body_groups)

    # Split the global head_vocab into per-group local columns.
    group_head_vocab = [[] for _ in range(num_groups)]
    for gid, head_signature in head_vocab:
        group_head_vocab[gid].append((gid, head_signature))

    # Per-group accumulators; vocab grows while scanning, so collect (row, col)
    # pairs first and allocate the dense tensors once sizes are known.
    feature_ids = [{} for _ in range(num_groups)]
    group_feature_vocab = [[] for _ in range(num_groups)]
    feature_rows = [[] for _ in range(num_groups)]
    feature_columns = [[] for _ in range(num_groups)]
    target_rows = [[] for _ in range(num_groups)]
    target_columns = [[] for _ in range(num_groups)]
    row_counts = [0] * num_groups

    for video_id, video_atoms in tqdm(atoms.items(), desc="Processing videos"):
        video_data = get_video_data(video_atoms, dataset_labels)
        for current_frame, next_frame in video_data:
            # Index next-frame facts by entity so a body atom can look up what
            # happens to *its own* entity at t+1.
            next_atoms_by_entity = {}
            for atom_signature in next_frame:
                next_atoms_by_entity.setdefault(
                    _atom_entity_key(atom_signature), set()
                ).add(atom_signature[:-2])

            for body_atom in current_frame:
                body_signature = body_atom[:-2]
                group_id = body_ids.get(body_signature)
                # Only atoms that are the body of some candidate clause become rows.
                if group_id is None:
                    continue
                entity_key = _atom_entity_key(body_atom)
                next_entity_facts = next_atoms_by_entity.get(entity_key)
                # Entity disappears in t+1: no supervision for this row.
                if not next_entity_facts:
                    continue

                row = row_counts[group_id]
                ids = feature_ids[group_id]
                vocab = group_feature_vocab[group_id]

                # object_slot: 0 = the body's own entity; other entities get
                # 1, 2, ... in order of first appearance, so atoms sharing a slot
                # refer to the same other object (variable co-reference).
                active_features = set()
                other_entity_slots = {}
                for atom in current_frame:
                    atom_fact = atom[:-2]
                    # Skip the body itself; it is implied by the group.
                    if atom_fact == body_signature:
                        continue
                    atom_entity_key = _atom_entity_key(atom)
                    if atom_entity_key == entity_key:
                        object_slot = 0
                    else:
                        object_slot = other_entity_slots.setdefault(
                            atom_entity_key, len(other_entity_slots) + 1
                        )
                    feature_key = (group_id, object_slot, atom_fact)
                    feature_id = ids.get(feature_key)
                    if feature_id is None:
                        feature_id = len(vocab)
                        ids[feature_key] = feature_id
                        vocab.append(feature_key)
                    active_features.add(feature_id)

                for feature_id in active_features:
                    feature_rows[group_id].append(row)
                    feature_columns[group_id].append(feature_id)
                # Positive target for each candidate head of this group that
                # holds for the same entity in the next frame.
                for head_col, (_gid, head_signature) in enumerate(group_head_vocab[group_id]):
                    if head_signature in next_entity_facts:
                        target_rows[group_id].append(row)
                        target_columns[group_id].append(head_col)
                row_counts[group_id] += 1

    features = []
    targets = []
    for gid in range(num_groups):
        features.append(_to_sparse_rows(
            feature_rows[gid], feature_columns[gid],
            (row_counts[gid], len(group_feature_vocab[gid])),
        ))
        targets.append(_to_sparse_rows(
            target_rows[gid], target_columns[gid],
            (row_counts[gid], len(group_head_vocab[gid])),
        ))

    return {
        "features": features,
        "targets": targets,
        "feature_vocab": group_feature_vocab,
        "head_vocab": group_head_vocab,
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


def _build_body_conditioned_training_examples(
    atoms, clauses, dataset_labels, output_dir, min_support=None
):
    """Load or build per-group feature and target matrices.

    With `min_support`, return the `prune_examples` result; it is cached in
    its own files per min_support value, next to the unpruned cache.
    """
    examples = None
    if min_support is not None:
        pruned_cache_file = output_dir / f"body_conditioned_examples_pruned_ms{min_support}.pt"
        pruned_signature_file = output_dir / f"atom_signatures_pruned_ms{min_support}.json"
        examples = _load_examples(pruned_cache_file, pruned_signature_file)

    if examples is None:
        cache_file = output_dir / "body_conditioned_examples.pt"
        signature_file = output_dir / "atom_signatures.json"
        examples = _load_examples(cache_file, signature_file)
        if examples is None:
            body_groups = group_clauses_by_body(clauses, dataset_labels)
            head_vocab = build_head_vocab(body_groups)
            examples = build_features(atoms, dataset_labels, body_groups, head_vocab)
            examples["body_groups"] = body_groups
            _save_examples(examples, cache_file, signature_file)
        if min_support is not None:
            examples = prune_examples(examples, min_support)
            _save_examples(examples, pruned_cache_file, pruned_signature_file)

    num_rows = sum(X["shape"][0] for X in examples["features"])
    num_features = sum(X["shape"][1] for X in examples["features"])
    num_heads = sum(Y["shape"][1] for Y in examples["targets"])
    print(
        f"[Training data] {len(examples['body_groups'])} groups, {num_rows} examples, "
        f"{num_features} features, {num_heads} candidate heads"
    )
    return examples


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

def extend_clauses(
    coarse_pruned_clauses,
    dataset_labels,
    atoms,
    cfg,
    min_support=10,
    min_confidence_gain=0.05,
    num_epochs=200,
    lr=0.1,
    l0_lambda=1e-3,
):
    output_dir = cfg["output_dir"]
    device = cfg["device"]
    # load the extended clauses if they exist, otherwise extend from coarse pruned clauses
    output_file = output_dir / "extended_clauses.json"
    if output_file.exists():
        with open(output_file, "r") as f:
            extended_data = json.load(f)
    else:
        training_examples_by_body = _build_body_conditioned_training_examples(
            atoms, coarse_pruned_clauses, dataset_labels, output_dir, min_support
        )
        features = training_examples_by_body["features"]
        targets = training_examples_by_body["targets"]
        feature_vocab = training_examples_by_body["feature_vocab"]
        feature_aliases = training_examples_by_body["feature_aliases"]
        head_vocab = training_examples_by_body["head_vocab"]

        extended_data = []

        # extension in one shot: differentiable conjunctive rule learning, group by group
        for gid in tqdm(range(len(features)), desc="Learning rule extensions"):
            num_rows, num_feature_cols = features[gid]["shape"]
            num_head_cols = targets[gid]["shape"][1]
            if num_rows == 0 or num_feature_cols == 0 or num_head_cols == 0:
                print(
                    f"[Rule extension] Skipping group {gid}: "
                    f"rows={num_rows}, feature_cols={num_feature_cols}, "
                    f"head_cols={num_head_cols} "
                    "(nothing to learn for this body-group)."
                )
                continue

            # Densify only this group's (N, M) block, kept on CPU; only
            # row-batches are moved to the GPU inside `learn_extension_mask`.
            # Float copy for training, numpy view of the bool block for hard
            # evaluation.
            dense_features = sparse_rows_to_dense(features[gid])
            X = dense_features.float()
            feature_block = dense_features.numpy()
            dense_targets = sparse_rows_to_dense(targets[gid])

            for head_col in range(num_head_cols):
                target_col = dense_targets[:, head_col].numpy().astype(np.float32)

                # Skip degenerate targets: nothing to learn if the head never fires
                # or always fires for this body-group.
                if target_col.sum() == 0 or target_col.sum() == len(target_col):
                    print(
                        f"[Rule extension] Skipping group {gid}, head {head_col}: "
                        f"degenerate target (head fires in {int(target_col.sum())}/"
                        f"{len(target_col)} rows)."
                    )
                    continue

                y = torch.from_numpy(target_col)
                final_mask, log_alpha = learn_extension_mask(
                    X, y, num_epochs=num_epochs, lr=lr, l0_lambda=l0_lambda, device=device,
                )
                selected_positions = torch.nonzero(final_mask, as_tuple=True)[0].tolist()
                del final_mask, log_alpha
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

            del X, dense_features, feature_block, dense_targets
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()
        # save extended clauses to a file
        output_file = output_dir / "extended_clauses.json"
        utils_data.save_json(extended_data, output_file)
        print(f"[Rule extension] Learned {len(extended_data)} extended clauses")
        raise NotImplementedError()
    training_examples_by_body = _build_body_conditioned_training_examples(
            atoms, coarse_pruned_clauses, dataset_labels, output_dir, min_support)
    extended_clauses = _to_extended_clauses(extended_data, training_examples_by_body)

    return extended_clauses


def build_clause_matrix(clauses):
    # get all unique bodies
    unique_bodies = []
    for clause in clauses:
        bodies = clause["bodies"]
        for body in bodies:
            if body not in unique_bodies:
                unique_bodies.append(body)

    clause_matrix = torch.zeros((len(clauses), len(unique_bodies)), dtype=torch.bool)
    # clause_scopes[i, j] = the object_slot that body j binds to in clause i's
    # body_scopes (0 = anchor object; equal nonzero values = co-referenced to
    # the same other object; -1 = body j is not part of clause i).
    clause_scopes = torch.full((len(clauses), len(unique_bodies)), -1, dtype=torch.long)
    clause_heads = []
    for i, clause in enumerate(clauses):
        bodies = clause["bodies"]
        body_scopes = clause.get("body_scopes")
        clause_heads.append(clause["head"])
        for j, body in enumerate(unique_bodies):
            if body in bodies:
                clause_matrix[i, j] = True
                if body_scopes is not None:
                    clause_scopes[i, j] = body_scopes[bodies.index(body)]
    print(f"[Clause Matrix]: {clause_matrix.shape}")

    return clause_matrix, clause_heads, unique_bodies, clause_scopes