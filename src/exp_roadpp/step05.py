

from pathlib import Path
import os
import json
import torch
import numpy as np
from tqdm import tqdm

from src.exp_roadpp import utils_data




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
    # replace id with label names
    # clause extension
    extended_clauses = extend_clauses(coarse_pruned_clauses, dataset_labels, atoms['train'], cfg)
