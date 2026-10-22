

from pathlib import Path
import os
import json
import torch
import numpy as np
from tqdm import tqdm

from src.exp_roadpp import utils_data



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



def main(input_data):
    cfg = {
        'output_dir': input_data['output_dir'],
        'device': input_data['device']

    }
    dataset_labels = utils_data.load_json(input_data['dataset_label_file'])
    atoms = utils_data.load_json(input_data['train_atom_file'])
    causal_anchor_clauses = utils_data.load_json(input_data['causal_anchor_clause_file'])
    # replace id with label names
    # clause extension
    extended_clauses = extend_clauses(causal_anchor_clauses, dataset_labels, atoms, cfg)
    return extended_clauses