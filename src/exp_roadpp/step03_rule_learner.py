


import torch
from pathlib import Path
import json 
from tqdm import tqdm
import os
import random
import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

from src.exp_roadpp import utils_data
from src.exp_roadpp.step03_pred_eval import get_video_data, predict_next_n_steps



def _atom_entity_key(atom_signature):
    tube_uid = atom_signature[-1]
    agent_class = atom_signature[7]
    if tube_uid is not None:
        return ("tube_uid", tube_uid)
    if agent_class == "av":
        return ("agent_class", "av")
    return None


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
    X, y, num_epochs=200, lr=0.1, l0_lambda=0.01, device='cpu', batch_size=None,
):
    # X: (N, M) float tensor (0/1), y: (N,) float tensor (0/1)
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
    log_alpha = torch.zeros(M, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([log_alpha], lr=lr)
    const_term = _beta_const_log_term()

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
                w = hard_concrete_mask(log_alpha, training=True)
                s = torch.prod(1 - w.unsqueeze(0) * (1 - X_batch), dim=1)  # (batch,)
                bce = torch.nn.functional.binary_cross_entropy(s.clamp(1e-6, 1 - 1e-6), y_batch)
                l0_penalty = torch.sigmoid(log_alpha - const_term).sum()  # expected L0 norm
                loss = bce + l0_lambda * l0_penalty
                loss.backward()
                optimizer.step()

                if whole_dataset_fits:
                    del w, s, bce, loss
                else:
                    del X_batch, y_batch, w, s, bce, loss
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

    final_mask = (hard_concrete_mask(log_alpha, training=False) > 0.5)
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
            body_signature, {"candidate_heads": set(), "clauses": []}
        )
        body_group["candidate_heads"].add(head_signature)
        body_group["clauses"].append(c)

    for body_group in clauses_by_body.values():
        body_group["candidate_heads"] = tuple(
            sorted(body_group["candidate_heads"], key=repr)
        )

    print(f"[Grouping] Clauses grouped by body: {len(clauses)} -> {len(clauses_by_body)}")
    return clauses_by_body

def _build_body_conditioned_training_examples(
    atoms, clauses, dataset_labels, output_dir
):
    """Load or build sparse feature and target matrices for each body group."""
    body_groups = group_clauses_by_body(clauses, dataset_labels)
    cache_dir = Path(output_dir) / "training_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = cache_dir / "body_conditioned_examples_v3_train.json"
    features_file = cache_dir / "body_conditioned_features_v3_train.npz"
    targets_file = cache_dir / "body_conditioned_targets_v3_train.npz"
    group_ids_file = cache_dir / "body_conditioned_group_ids_v3_train.npz"

    body_signatures = tuple(body_groups)
    body_ids = {
        body_signature: index
        for index, body_signature in enumerate(body_signatures)
    }
    metadata_groups = [
        {
            "body_signature": list(body_signature),
            "candidate_heads": [
                list(head_signature)
                for head_signature in body_groups[body_signature]["candidate_heads"]
            ],
        }
        for body_signature in body_signatures
    ]

    if all(path.exists() for path in (
        metadata_file, features_file, targets_file, group_ids_file
    )):
        print(f"[Training data] Loading sparse training cache from {metadata_file}")
        cache_metadata = utils_data.load_json(metadata_file)
        if cache_metadata.get("schema_version") != 3:
            raise ValueError("Unsupported sparse training cache version.")
        cached_groups = cache_metadata["groups"]
        if cached_groups != metadata_groups:
            raise ValueError(
                "Cached body groups or candidate heads do not match current rules."
            )
        features = sparse.load_npz(features_file).tocsr()
        targets = sparse.load_npz(targets_file).tocsr()
        with np.load(group_ids_file) as group_data:
            group_ids = group_data["group_ids"]
        if features.shape[0] != targets.shape[0] or features.shape[0] != len(group_ids):
            raise ValueError("Cached training matrices have inconsistent row counts.")
        return {
            "body_groups": body_groups,
            "features": features,
            "targets": targets,
            "group_ids": group_ids,
            "feature_vocab": tuple(
                (
                    int(item["group_id"]),
                    item["object_slot"] if item["object_slot"] is None else int(item["object_slot"]),
                    tuple(item["signature"]),
                )
                for item in cache_metadata["feature_vocab"]
            ),
            "head_vocab": tuple(
                (int(item["group_id"]), tuple(item["signature"]))
                for item in cache_metadata["head_vocab"]
            ),
        }

    feature_ids = {}
    head_ids = {}
    feature_vocab = []
    head_vocab = []
    feature_rows = []
    feature_columns = []
    target_rows = []
    target_columns = []
    group_ids = []
    row_count = 0

    for body_signature, body_group in body_groups.items():
        group_id = body_ids[body_signature]
        head_ids[body_signature] = {}
        for head_signature in body_group["candidate_heads"]:
            head_ids[body_signature][head_signature] = len(head_vocab)
            head_vocab.append((group_id, head_signature))

    for video_id, video_atoms in tqdm(atoms.items(), desc="Processing videos"):
        video_data = get_video_data(video_id, video_atoms, dataset_labels)
        frame_data = video_data["frame_data"]

        for frame_id, current_atoms in frame_data.items():
            # Skip the last frame since it has no next frame to predict.
            next_atoms = frame_data.get(frame_id + 1)
            if next_atoms is None:
                continue
            # Organize next frame atoms by their entity key for efficient lookup.
            next_atoms_by_entity = {}
            for atom_signature in next_atoms:
                entity_key = _atom_entity_key(atom_signature)
                if entity_key is not None:
                    next_atoms_by_entity.setdefault(entity_key, set()).add(
                        atom_signature[:-2]
                    )

            for body_atom in current_atoms:
                body_signature = body_atom[:-2]
                if body_signature not in body_groups:
                    continue
                entity_key = _atom_entity_key(body_atom)
                next_entity_facts = next_atoms_by_entity.get(entity_key)
                if not next_entity_facts:
                    continue

                group_id = body_ids[body_signature]
                active_features = set()
                # object_slot distinguishes which concrete object a co-occurring atom
                # refers to: 0 is always the body's own (anchor) object; atoms about
                # other objects get their own increasing slot number so that two
                # atoms sharing a slot are known to refer to the *same* other object
                # (variable co-reference); atoms with no object binding (e.g. purely
                # scene-level facts) get slot None.
                other_entity_slots = {}
                for atom in current_atoms:
                    atom_fact = atom[:-2]
                    if atom_fact == body_signature:
                        continue
                    atom_entity_key = _atom_entity_key(atom)
                    if atom_entity_key == entity_key:
                        object_slot = 0
                    elif atom_entity_key is None:
                        object_slot = None
                    else:
                        object_slot = other_entity_slots.setdefault(
                            atom_entity_key, len(other_entity_slots) + 1
                        )
                    feature_key = (group_id, object_slot, atom_fact)
                    feature_id = feature_ids.get(feature_key)
                    if feature_id is None:
                        feature_id = len(feature_vocab)
                        feature_ids[feature_key] = feature_id
                        feature_vocab.append(feature_key)
                    active_features.add(feature_id)

                for feature_id in active_features:
                    feature_rows.append(row_count)
                    feature_columns.append(feature_id)

                for head_signature, target_id in head_ids[body_signature].items():
                    if head_signature in next_entity_facts:
                        target_rows.append(row_count)
                        target_columns.append(target_id)

                group_ids.append(group_id)
                row_count += 1

    features = sparse.csr_matrix(
        (
            np.ones(len(feature_rows), dtype=np.uint8),
            (feature_rows, feature_columns),
        ),
        shape=(row_count, len(feature_vocab)),
        dtype=np.uint8,
    )
    targets = sparse.csr_matrix(
        (
            np.ones(len(target_rows), dtype=np.uint8),
            (target_rows, target_columns),
        ),
        shape=(row_count, len(head_vocab)),
        dtype=np.uint8,
    )
    group_ids_array = np.asarray(group_ids, dtype=np.int32)

    utils_data.save_json(
        {
            "schema_version": 3,
            "groups": metadata_groups,
            "feature_vocab": [
                {
                    "group_id": group_id,
                    "object_slot": object_slot,
                    "signature": list(signature),
                }
                for group_id, object_slot, signature in feature_vocab
            ],
            "head_vocab": [
                {
                    "group_id": group_id,
                    "signature": list(signature),
                }
                for group_id, signature in head_vocab
            ],
        },
        metadata_file,
    )
    sparse.save_npz(features_file, features, compressed=True)
    sparse.save_npz(targets_file, targets, compressed=True)
    np.savez_compressed(group_ids_file, group_ids=group_ids_array)

    print(
        f"[Training data] Prepared sparse matrices: "
        f"{features.shape[0]} examples, {features.shape[1]} features, "
        f"{targets.shape[1]} candidate heads"
    )
    print(f"[Training data] Cached sparse examples under {cache_dir}")
    return {
        "body_groups": body_groups,
        "features": features,
        "targets": targets,
        "group_ids": group_ids_array,
        "feature_vocab": tuple(feature_vocab),
        "head_vocab": tuple(head_vocab),
    }


def _build_group_indices(training_examples_by_body):
    # Implement the logic to build group indices based on the training examples by body
    group_ids = training_examples_by_body["group_ids"]
    feature_vocab = training_examples_by_body["feature_vocab"]
    head_vocab = training_examples_by_body["head_vocab"]

    rows_by_group = {}
    for gid in np.unique(group_ids):
        rows_by_group[int(gid)] = np.where(group_ids == gid)[0]

    feature_cols_by_group = {}
    for idx, (gid, _object_slot, _sig) in enumerate(feature_vocab):
        feature_cols_by_group.setdefault(gid, []).append(idx)

    head_cols_by_group = {}
    for idx, (gid, _sig) in enumerate(head_vocab):
        head_cols_by_group.setdefault(gid, []).append(idx)


    return rows_by_group, feature_cols_by_group, head_cols_by_group



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
    l0_lambda=0.01,
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
            atoms, coarse_pruned_clauses, dataset_labels, output_dir
        )
        features = training_examples_by_body["features"].tocsc()
        targets = training_examples_by_body["targets"].tocsc()
        feature_vocab = training_examples_by_body["feature_vocab"]
        head_vocab = training_examples_by_body["head_vocab"]

        rows_by_group, feature_cols_by_group, head_cols_by_group = _build_group_indices(
            training_examples_by_body
        )

        extended_data = []

        # extension in one shot: differentiable conjunctive rule learning, group by group
        for gid in tqdm(rows_by_group, desc="Learning rule extensions"):
            rows = rows_by_group.get(gid)
            feature_cols = feature_cols_by_group.get(gid, [])
            head_cols = head_cols_by_group.get(gid, [])
            if rows is None or len(rows) == 0 or len(feature_cols) == 0 or len(head_cols) == 0:
                continue

            # Dense slice of this group's features: (N, M) -- kept on CPU; only
            # row-batches are moved to the GPU inside `learn_extension_mask` to
            # avoid holding the full (N, M) block (plus its forward/backward
            # intermediates) in GPU memory at once.
            feature_block = np.asarray(
                features[rows][:, feature_cols].todense()
            ).astype(np.float32)
            X = torch.from_numpy(feature_block)

            for head_col in head_cols:
                target_col = np.asarray(
                    targets[rows][:, head_col].todense()
                ).ravel().astype(np.float32)

                # Skip degenerate targets: nothing to learn if the head never fires
                # or always fires for this body-group.
                if target_col.sum() == 0 or target_col.sum() == len(target_col):
                    continue

                y = torch.from_numpy(target_col)

                try:
                    final_mask, log_alpha = learn_extension_mask(
                        X, y, num_epochs=num_epochs, lr=lr, l0_lambda=l0_lambda, device=device,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(
                        f"[Rule extension] CUDA OOM for group {gid}, head {head_col} "
                        f"on device={device}. Re-run with a smaller --batch_size, "
                        "a smaller --num_epochs, or pass --device cpu."
                    )
                    raise
                selected_positions = torch.nonzero(final_mask, as_tuple=True)[0].tolist()
                del final_mask, log_alpha
                if not selected_positions:
                    continue

                metrics = _evaluate_extension(feature_block, target_col, selected_positions)
                if metrics["support"] < min_support:
                    continue
                if metrics["confidence_gain"] < min_confidence_gain:
                    continue

                selected_atoms = [
                    feature_vocab[feature_cols[pos]] for pos in selected_positions
                ]
                print(f"Extra Atom Num: {len(selected_atoms)}")
                extended_data.append({
                    "group_id": gid,
                    "head": head_vocab[head_col],
                    "extra_atoms": selected_atoms,
                    "metrics": metrics,
                })

            del X
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()
        # save extended clauses to a file
        output_file = output_dir / "extended_clauses.json"
        utils_data.save_json(extended_data, output_file)
        print(f"[Rule extension] Learned {len(extended_data)} extended clauses")
    training_examples_by_body = _build_body_conditioned_training_examples(
            atoms, coarse_pruned_clauses, dataset_labels, output_dir)
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
    clause_heads = []
    for i, clause in enumerate(clauses):
        bodies = clause["bodies"]
        clause_heads.append(clause["head"])
        for j, body in enumerate(unique_bodies):
            if body in bodies:
                clause_matrix[i, j] = True
    print(f"[Clause Matrix]: {clause_matrix.shape}")
    
    return clause_matrix, clause_heads, unique_bodies