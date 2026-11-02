

from pathlib import Path
import os
import sys
import getpass
import time
import torch
import numpy as np
from tqdm import tqdm

from src.exp_roadpp import utils_data
from src.exp_roadpp.cauvid import TemporalRuleNet



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


def past_bins(past_windows):
    """Disjoint past time bins (lo, hi) from the sorted window edges, e.g.
    [3, 5, 10] -> (0, 3), (3, 5), (5, 10). Bin (lo, hi) covers the frames
    lo..hi-1 frames before t, i.e. (t - hi, t - lo]."""
    edges = [0] + sorted(set(past_windows))
    return list(zip(edges[:-1], edges[1:]))


def lead_bins(delay_windows):
    """Disjoint future bins (lo, hi) of the head targets from the sorted window edges,
    e.g. [10, 20, 50] -> (0, 10), (10, 20), (20, 50). Bin (lo, hi) means the head
    starts in frames (t + lo, t + hi], i.e. lo < delay <= hi frames ahead."""
    edges = [0] + sorted(set(delay_windows))
    return list(zip(edges[:-1], edges[1:]))


def temporal_predicates(past_windows):
    """Names of the P predicates of `build_video_temporal_tensor`: "now", then
    "appeared_lo_hi" for each bin of `past_bins`."""
    return ["now"] + [f"appeared_{lo}_{hi}" for lo, hi in past_bins(past_windows)]


def build_video_temporal_tensor(X, past_windows):
    """(T, N, P, A) bool tensor of one video: Xt[t, n, p, a] is temporal predicate p
    (`temporal_predicates`) of atom a for agent n in frame t, over the past frames.

    p = 0 ("now") is X itself. For each disjoint bin (lo, hi) of `past_bins`,
    appeared: n holds a in at least one of the frames (t - hi, t - lo] (frames
    before the video count as not held).
    Bins are time ranges, not exact frame shifts, so rules can order events
    (A appeared 5-10 frames ago, B within 3 frames) and stay robust to jitter.
    """
    # counts[j] - counts[i]: frames i..j-1 in which n holds a.
    zero = torch.zeros_like(X[:1], dtype=torch.int32)
    counts = torch.cat([zero, X.int().cumsum(dim=0, dtype=torch.int32)])
    end = torch.arange(1, X.shape[0] + 1, device=X.device)
    predicates = [X]
    for lo, hi in past_bins(past_windows):
        stop = (end - lo).clamp(min=0)
        start = (end - hi).clamp(min=0)
        num_frames = counts[stop] - counts[start]
        predicates.append(num_frames > 0)
    return torch.stack(predicates, dim=2)


def _video_cells(X):
    # (T, N, A) bool tensor of `build_video_tensor` -> its compact form on CPU:
    # the (C, 3) int32 (t, n, a) of its True cells and its (T, N) shape.
    return {"cells": torch.nonzero(X).to(torch.int32).cpu(), "shape": tuple(X.shape[:2])}


def _video_from_cells(video, num_atoms, device):
    # Inverse of `_video_cells`.
    X = torch.zeros((*video["shape"], num_atoms), dtype=torch.bool, device=device)
    t, n, a = video["cells"].to(device).long().T
    X[t, n, a] = True
    return X


def build_body_features(Xt, D, body, head_atoms, bins):
    """Training rows of one body (b, s) in one video.

    Xt: (T, N, P, A) temporal atom tensor of the video (`build_video_temporal_tensor`;
    Xt[:, :, 0] is the atom tensor of `build_video_tensor`), D: (T, N, A) delay
    tensor (`build_video_delay_tensor`). body = (b, s): agent slot s holds atom b.
    head_atoms: (H,) atom ids of the body's candidate heads; bins: (K, 2) lead bins
    (lo, hi] of `lead_bins`.

    A row is one grounding in a frame t where some agent n_body holds b: the
    head agent n_head is n_body itself if s == 0, otherwise every other agent
    visible in frame t. Returns
      features: (R, s + 2, P, A) bool, the temporal atoms in frame t of each role:
          slot 0 the head agent, slot s the body agent (without b "now", which
          every row has) and slot s + 1 "some other agent" (any agent but those two);
      targets: (R, H, K) bool, head h next starts for the head agent in lead bin k,
          i.e. lo_k < delay <= hi_k frames after t (and is not already held by it in
          frame t). Every frame before an onset is a row, so the bins cover every
          lead time the evaluation asks for (step06), each learned separately;
      rows: (R, 3) long, (t, n_body, n_head) of each row.
    """
    b, s = body
    X = Xt[:, :, 0]
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

    head = Xt[t, n_head]                                    # (R, P, A)
    body_agent = Xt[t, n_body]                              # (R, P, A)
    others = Xt.sum(dim=1)[t] - body_agent.int()            # (R, P, A) agents with (p, a), minus the roles
    if s != 0:
        others -= head.int()
    features = torch.zeros((len(t), s + 2, *Xt.shape[2:]), dtype=torch.bool, device=Xt.device)
    features[:, 0] = head
    features[:, s] |= body_agent
    features[:, s, 0, b] = False
    features[:, s + 1] = others > 0

    delay = D[t, n_head][:, head_atoms].unsqueeze(-1)       # (R, H, 1)
    targets = (delay > bins[:, 0]) & (delay <= bins[:, 1])  # (R, H, K)
    rows = torch.stack([t, n_body, n_head], dim=1)
    return features, targets, rows


def build_features(atoms, caw_clauses, unique_atoms, device, past_windows=()):
    bodies = list(set([c[0] for c in caw_clauses]))
    # Sorted so the bin axis K of the targets has a fixed order.
    delay_windows = sorted(set([c[-1] for c in caw_clauses]))
    num_atom_type = len(unique_atoms)
    bins = torch.tensor(lead_bins(delay_windows), dtype=torch.int16, device=device)
    delay_dtype = torch.uint8 if max(delay_windows) <= 255 else torch.int16
    videos = []
    video_ids = {body:[] for body in bodies}
    delays = {body:[] for body in bodies}
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
        Xt = build_video_temporal_tensor(X, past_windows)
        # print X and D memory usage
        print(f"Xt memory usage: {Xt.element_size() * Xt.nelement() / 1024 ** 2:.2f} MB")
        print(f"D memory usage: {D.element_size() * D.nelement() / 1024 ** 2:.2f} MB")
        for body in bodies:
            _F, _T, R = build_body_features(Xt, D, body, head_atoms[body], bins)
            if len(R)>0:
                video_ids[body].append(len(videos))
                delays[body].append(D[R[:, 0], R[:, 2]][:, head_atoms[body]].to(delay_dtype).cpu())
                rows[body].append(R.to(torch.int32).cpu())
        videos.append(_video_cells(X))

    return videos, video_ids, delays, rows


def body_features(examples, body, device):
    """(R, M) bool features of one body on CPU, over the rows of
    `examples["rows"][body]`: `build_body_features` rebuilt from the atoms of
    each of its videos, flattened to M = (s + 2) * P * A columns."""
    blocks = []
    for v, stored_rows in zip(examples["video_ids"][body], examples["rows"][body]):
        X = _video_from_cells(examples["videos"][v], examples["num_atoms"], device)
        Xt = build_video_temporal_tensor(X, examples["past_windows"])
        # Only the features are needed: no heads, so the delays are never read.
        no_heads = torch.empty(0, dtype=torch.long, device=device)
        no_bins = torch.empty((0, 2), dtype=torch.int16, device=device)
        F, _T, R = build_body_features(Xt, X[:, :, :0], body, no_heads, no_bins)
        if not torch.equal(R.cpu().to(torch.int32), stored_rows):
            raise RuntimeError(f"Rebuilt rows of body {body} in video {v} differ from the stored ones.")
        blocks.append(F.reshape(len(F), -1).cpu())
        del X, Xt, F
    return torch.cat(blocks)


def delay_targets(delays, windows):
    """(R, H * K) bool targets of `build_body_features` from the (R, H) delays of
    `build_features`: column h * K + k is "head h next starts in lead bin k"."""
    delays = delays.long().unsqueeze(-1)
    bins = torch.tensor(lead_bins(windows), dtype=torch.long)
    return ((delays > bins[:, 0]) & (delays <= bins[:, 1])).reshape(len(delays), -1)
        



def build_head_vocab(body_groups, delay_windows):
    # One head "starts in the lead bin ending at w" per (candidate head, window).
    head_vocab = []
    for body, heads in body_groups.items():
        for head in heads:
            for window in delay_windows:
                head_vocab.append((body, head, window))
    return head_vocab




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


def learn_extension_masks(
    X, Y, num_epochs=200, lr=0.1, l0_lambda=1e-3, device='cpu', batch_size=None,
    mask_patience=20, check_every=5, gate_tol=1e-3, init_log_alpha=-5.0,
    positive_weights=None,
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
    # `positive_weights` (N, H) weights the positives relative to each other
    # (e.g. 1 / rows of their event, so each event counts once); the positives
    # of a head then get the same total weight as its negatives.
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
    if positive_weights is None:
        positive_weights = torch.ones_like(Y)
    positive_weights = positive_weights.detach().to('cpu', torch.float32) * Y
    total_positive = positive_weights.sum(dim=0)
    pos_scale = torch.where(
        total_positive > 0, (N - Y.sum(dim=0)) / total_positive.clamp(min=1e-12), torch.ones_like(total_positive)
    )
    weights = (1 - Y) + positive_weights * pos_scale
    del positive_weights

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




def _evaluate_extension(feature_block, target_col, selected_feature_positions, event_col):
    """Compute hard support/coverage/confidence/recall for a candidate extension.

    feature_block: dense (N, M) 0/1 array for this group's rows/features.
    target_col: dense (N,) 0/1 array for the candidate head within this group.
    selected_feature_positions: indices (into the group's local feature axis)
        selected by the learned mask.
    event_col: (N,) event id of each positive row (`_event_ids`), -1 for negatives.
    Support and confidence count rows (observation frames); recall counts
    events, as the rows before one onset are many correlated views of it.
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
            "support": 0, "coverage": 0.0, "confidence": 0.0, "recall": 0.0, "row_recall": 0.0,
            "events": 0, "confidence_gain": -base_confidence, "lift": 0.0, "f1": 0.0,
        }

    head_hits = int(target_col[extended_mask].sum())
    confidence = head_hits / support
    coverage = support / base_rows if base_rows else 0.0
    row_recall = head_hits / base_head_count if base_head_count else 0.0
    num_events = len(np.unique(event_col[event_col >= 0]))
    events_hit = len(np.unique(event_col[extended_mask & (event_col >= 0)]))
    recall = events_hit / num_events if num_events else 0.0
    f1 = (
        2 * confidence * recall / (confidence + recall)
        if (confidence + recall) > 0 else 0.0
    )
    return {
        "support": support,
        "coverage": coverage,
        "confidence": confidence,
        "recall": recall,
        "row_recall": row_recall,
        "events": events_hit,
        "confidence_gain": confidence - base_confidence,
        # confidence relative to the body alone; base_confidence > 0 for learnable heads.
        "lift": confidence / base_confidence if base_confidence else 0.0,
        "f1": f1,
    }

def _atom_name(signature):
    # ["pred", "action", "action_id", 3, "action_id_label", "MovAway",
    #  "agent_class", 1, "agent_class_label", "Car"] -> "Car:MovAway"
    fields = dict(zip(signature[2::2], signature[3::2]))
    labels = [str(v) for k, v in fields.items() if k.endswith("_label") and k != "agent_class_label"]
    return f"{fields.get('agent_class_label', '?')}:{'/'.join(labels) or signature[1]}"


def predicate_text(predicate):
    # "now" -> "", "held_0_3" -> "[held last 3f]", "started_0_3" -> "[started ≤3f ago]",
    # "ended_5_10" -> "[ended 5-10f ago]", "held_5_10" -> "[held 5-10f ago]".
    if predicate == "now":
        return ""
    kind, lo, hi = predicate.split("_")
    if lo == "0":
        return f"[held last {hi}f]" if kind == "held" else f"[{kind} ≤{hi}f ago]"
    return f"[{kind} {lo}-{hi}f ago]"


def _clause_text(clause, unique_atoms):
    # Object slot s is the variable X{s}; slot 0 is the head object.
    def atom(atom_index, slot, predicate="now"):
        name = _atom_name(unique_atoms[atom_index]) if unique_atoms else f"a{atom_index}"
        return f"{name}(X{slot}){predicate_text(predicate)}"
    body = " ∧ ".join(atom(*a) for a in clause["bodies"])
    lo, hi = clause["lead_bin"]
    return f"{body} ⇒ starts {atom(*clause['head'])} in {lo + 1}-{hi}f"


def _summarize_extended_clauses(extended_clauses, unique_atoms, output_dir, num_examples=15):
    """Print statistics of the extended clauses and save them as a PDF figure."""
    print(f"[Extended clauses] {len(extended_clauses)} clauses")
    if not extended_clauses:
        return
    body_lengths = np.array([len(c["bodies"]) for c in extended_clauses])
    windows = np.array([c["delay_window"] for c in extended_clauses])
    metrics = {k: np.array([c[k] for c in extended_clauses]) for k in ("support", "confidence", "recall", "lift", "f1")}
    num_bodies = len({tuple(c["bodies"][0]) for c in extended_clauses})
    num_heads = len({tuple(c["head"]) for c in extended_clauses})
    print(f"[Extended clauses] {num_bodies} anchor bodies, {num_heads} heads")
    lengths, length_counts = np.unique(body_lengths, return_counts=True)
    print("[Extended clauses] body length: " + ", ".join(f"{l}: {n}" for l, n in zip(lengths, length_counts)))
    window_values, window_counts = np.unique(windows, return_counts=True)
    print("[Extended clauses] delay window: " + ", ".join(f"{w}: {n}" for w, n in zip(window_values, window_counts)))
    # Extra atoms by temporal predicate kind ("now", "held", "started", "ended").
    kind_counts = {}
    for c in extended_clauses:
        for _atom, _slot, predicate in c["bodies"][1:]:
            kind = predicate.split("_")[0]
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
    num_temporal = sum(any(p != "now" for _a, _s, p in c["bodies"][1:]) for c in extended_clauses)
    print(f"[Extended clauses] {num_temporal}/{len(extended_clauses)} clauses with temporal atoms; extra atoms: "
          + ", ".join(f"{k}: {n}" for k, n in kind_counts.items()))
    videos = np.array([c.get("videos", 0) for c in extended_clauses])
    print(f"[Extended clauses] training videos a clause is correct in: mean {videos.mean():.1f}, min {videos.min()}")
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
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        fig.suptitle(f"{len(extended_clauses)} extended clauses ({num_bodies} anchor bodies, {num_heads} heads)")
        axes[0, 0].bar(lengths, length_counts)
        axes[0, 0].set(title="Body length (anchor + extra atoms)", xlabel="atoms", ylabel="clauses", xticks=lengths)
        axes[0, 1].bar([str(w) for w in window_values], window_counts)
        axes[0, 1].set(title="Delay window", xlabel="frames", ylabel="clauses")
        axes[0, 2].bar(list(kind_counts), list(kind_counts.values()))
        axes[0, 2].set(title="Extra atoms by temporal predicate", ylabel="atoms")
        axes[0, 3].hist(metrics["support"], bins=30, log=True)
        axes[0, 3].set(title="Support", xlabel="rows", ylabel="clauses (log)")
        video_values, video_counts = np.unique(videos, return_counts=True)
        axes[1, 3].bar([str(v) for v in video_values], video_counts)
        axes[1, 3].set(title="Training videos the clause is correct in", xlabel="videos", ylabel="clauses")
        for ax, k in zip(axes[1], ("confidence", "lift", "f1")):
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
            header = f"{i:>2}. f1={c['f1']:.2f} conf={c['confidence']:.2f} recall={c['recall']:.2f} lift={c['lift']:.1f} support={c['support']}"
            lines.append(header)
            lines.extend("      " + line for line in textwrap.wrap(_clause_text(c, unique_atoms), 120))
        ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=7, transform=ax.transAxes)
        pdf.savefig(fig)
        plt.close(fig)
    print(f"[Extended clauses] Summary figure saved to {pdf_file}")


def _to_extended_clauses(extended_data, unique_atoms, cfg):
    delay_windows = cfg["delay_windows"]
    past_windows = cfg["past_windows"]
    output_dir = cfg["output_dir"]
    extended_clauses = []
    for item in extended_data:
        # Bodies are ungrounded (atom_index, object_slot, temporal_predicate)
        # atoms, the anchor body ("now") first; body_scopes lists each body's
        # object slot. Predicates are named as in `temporal_predicates`.
        bodies = [(*item["body"], "now")] + [tuple(extra) for extra in item["extra_atoms"]]
        extended_clauses.append({
            "bodies": bodies,
            "body_scopes": [slot for _atom, slot, _predicate in bodies],
            "head": tuple(item["head"]),
            # The head starts lo < delay <= hi frames ahead (`lead_bins`);
            # delay_window is hi, delay_windows the edges of all bins learned.
            "lead_bin": list(item["lead_bin"]),
            "delay_window": item["delay_window"],
            "delay_windows": list(delay_windows),
            # Edges of the past bins of the temporal predicates, needed to ground them.
            "past_windows": list(past_windows),
            **item["metrics"],
        })
    if output_dir is not None:
        _summarize_extended_clauses(extended_clauses, unique_atoms, output_dir)
    return extended_clauses


def _learnable_heads(targets, video_of_row, min_videos):
    # Skip degenerate targets: nothing to learn if the head never fires or
    # always fires for this body, and nothing general if all its positives
    # come from fewer than min_videos videos.
    num_rows, num_head_cols = targets.shape
    head_positives = targets.sum(dim=0)
    num_videos = int(video_of_row.max()) + 1
    positives_per_video = torch.zeros((num_videos, num_head_cols), dtype=torch.long)
    positives_per_video.index_add_(0, video_of_row, targets.long())
    learnable_heads = torch.nonzero(
        (head_positives > 0) & (head_positives < num_rows)
        & ((positives_per_video > 0).sum(dim=0) >= min_videos),
        as_tuple=True,
    )[0].tolist()
    return learnable_heads


def _event_ids(Y, rows, delays, video_of_row, num_windows):
    """(N, C) long id of the event each positive target cell belongs to, -1 for
    negatives. An event is one onset (video, head agent, head, onset frame
    t + delay) seen in one lead bin: target column c = h * K + k of `_extend_body`."""
    r, c = torch.nonzero(Y, as_tuple=True)
    onsets = rows[r, 0].long() + delays[r, c // num_windows].long()
    keys = torch.stack([video_of_row[r], rows[r, 2].long(), onsets, c], dim=1)
    _keys, ids = torch.unique(keys, dim=0, return_inverse=True)
    event_id = torch.full(Y.shape, -1, dtype=torch.long)
    event_id[r, c] = ids
    return event_id


def _accepted_metrics(feature_block, target_col, selected_positions, video_of_row, event_col, cfg):
    # Hard metrics of the learned extension, or None if it is rejected.
    if not selected_positions:
        return None

    metrics = _evaluate_extension(feature_block, target_col, selected_positions, event_col)
    if metrics["support"] < cfg["min_support"]:
        return None
    # Relative, not absolute: head onsets are rare (base confidence ~0.5%), so a
    # rule 10x above the body alone still gains far less than 0.05 in absolute terms.
    if metrics["lift"] < cfg["min_lift"]:
        return None
    # A rule correct in a single video only memorizes that video.
    hits = feature_block[:, selected_positions].all(axis=1) & (target_col > 0)
    metrics["videos"] = len(np.unique(video_of_row[hits]))
    if metrics["videos"] < cfg["min_videos"]:
        return None
    return metrics


def _extend_body(body, examples, cfg):
    """Differentiable conjunctive rule learning for one body; returns the
    accepted extensions of its (head, window) pairs.

    examples: of `build_training_examples`; the body's (R, H) delays and (R, 3)
    rows per video, its H heads, the K windows and P temporal predicates. The
    features are rebuilt from the video atoms (`body_features`).
    """
    rows = examples["rows"][body]
    if not rows:
        return []
    heads, windows = examples["heads"][body], examples["windows"]
    predicates, num_atoms = examples["predicates"], examples["num_atoms"]

    # Flatten the role slots, temporal predicates and atoms, and the (head,
    # window) pairs into columns, on CPU; `learn_extension_masks` moves them to
    # the device. Feature column (slot * P + p) * A + atom is "agent in role
    # slot satisfies predicate p of atom"; target column h * K + k is
    # "heads[h] starts in lead bin k", the bin ending at windows[k].
    num_predicates = len(predicates)
    delays = torch.cat(examples["delays"][body])
    Y = delay_targets(delays, windows)
    video_of_row = torch.repeat_interleave(
        torch.arange(len(rows)), torch.tensor([len(R) for R in rows])
    )

    learnable_heads = _learnable_heads(Y, video_of_row, cfg["min_videos"])
    if not learnable_heads:
        return []
    event_id = _event_ids(Y, torch.cat(rows), delays, video_of_row, len(windows))
    del delays
    X = body_features(examples, body, cfg["device"])
    feature_block = X.numpy()
    video_of_row = video_of_row.numpy()

    # Each event's positive rows share a weight of 1, so an event observed over
    # many frames (or body agents) counts as much as a briefly observed one.
    event_id_learnable = event_id[:, learnable_heads]
    positive = event_id_learnable >= 0
    event_rows = torch.bincount(event_id_learnable[positive])
    positive_weights = torch.zeros(event_id_learnable.shape, dtype=torch.float32)
    positive_weights[positive] = 1.0 / event_rows[event_id_learnable[positive]]
    del event_id_learnable, positive, event_rows

    # All learnable heads of the body are trained together.
    final_masks, _log_alpha = learn_extension_masks(
        X, Y[:, learnable_heads],
        num_epochs=cfg["num_epochs"], lr=cfg["lr"], l0_lambda=cfg["l0_lambda"],
        device=cfg["device"], positive_weights=positive_weights,
    )
    del positive_weights

    body_data = []
    for mask_col, head_col in enumerate(learnable_heads):
        target_col = Y[:, head_col].numpy().astype(np.float32)
        selected_positions = torch.nonzero(final_masks[:, mask_col], as_tuple=True)[0].tolist()
        metrics = _accepted_metrics(
            feature_block, target_col, selected_positions, video_of_row,
            event_id[:, head_col].numpy(), cfg
        )
        if metrics is None:
            continue

        h, k = divmod(head_col, len(windows))
        extra_atoms = []
        for pos in selected_positions:
            slot_predicate, atom = divmod(pos, num_atoms)
            slot, p = divmod(slot_predicate, num_predicates)
            extra_atoms.append((atom, slot, predicates[p]))
        body_data.append({
            "body": body,
            "head": heads[h],
            "lead_bin": lead_bins(windows)[k],
            "delay_window": windows[k],
            # (atom_index, object_slot, temporal_predicate) of each extra atom.
            "extra_atoms": extra_atoms,
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


def _init_wandb(cfg, extra_config=None, job_type="step05_extension"):
    """Online run on the hosted W&B cloud, or None if tracking is off or fails.

    Its config holds the plain settings of cfg (numbers, strings, lists; not the
    file paths) and extra_config.

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
        settings = {k: v for k, v in cfg.items() if isinstance(v, (bool, int, float, str, list))}
        run = wandb.init(
            project=cfg.get("wandb_project", "cauvid-roadpp"),
            name=cfg.get("wandb_run_name"),
            job_type=job_type,
            mode="online",
            config={**settings, **(extra_config or {})},
        )
        print(f"[wandb] Run: {run.url}")
        return run
    except Exception as exc:
        print(f"[wandb] Tracking disabled: {type(exc).__name__}: {exc}")
        return None


def _learn_extensions(examples, cfg):
    num_bodies = len(examples["rows"])
    run = _init_wandb(cfg, {"num_bodies": num_bodies})
    extended_data = []
    start_time = time.perf_counter()
    try:
        for i, body in enumerate(tqdm(examples["rows"], desc="Learning rule extensions")):
            body_start = time.perf_counter()
            body_data = _extend_body(body, examples, cfg)
            extended_data.extend(body_data)
            # The body's dense blocks are freed on return.
            if cfg["device"] != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if run is not None:
                run.log({
                    "progress/bodies_done": i + 1,
                    "progress/fraction": (i + 1) / num_bodies,
                    "progress/elapsed_s": time.perf_counter() - start_time,
                    "body/rows": sum(len(R) for R in examples["rows"][body]),
                    "body/head_cols": len(examples["heads"][body]) * len(examples["windows"]),
                    "body/accepted": len(body_data),
                    "body/seconds": time.perf_counter() - body_start,
                    "total/accepted": len(extended_data),
                }, step=i)
        if run is not None:
            import wandb
            table = wandb.Table(columns=[
                "body", "head", "delay_window", "extra_atoms",
                "support", "events", "videos", "confidence", "recall", "row_recall", "confidence_gain", "lift", "f1",
            ])
            for item in extended_data:
                m = item["metrics"]
                table.add_data(
                    str(item["body"]), str(item["head"]), item["delay_window"], str(item["extra_atoms"]),
                    m["support"], m["events"], m["videos"], m["confidence"], m["recall"], m["row_recall"],
                    m["confidence_gain"], m["lift"], m["f1"],
                )
            run.log({"extensions": table})
            run.summary["total/accepted"] = len(extended_data)
            run.summary["total/seconds"] = time.perf_counter() - start_time
    finally:
        if run is not None:
            run.finish()
    return extended_data



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


def atom_onsets(X):
    """(T, N, A) bool: atom a switches on (false -> true) for agent n in frame t."""
    previous = torch.zeros_like(X)
    previous[1:] = X[:-1]
    return X & ~previous


def next_true_frame(mask):
    """(T + 1, N) long: first frame >= t where mask[:, n] is True, T if none.

    Row T is all T, so next_true_frame(mask)[t + 1] is the first frame after t."""
    T = mask.shape[0]
    frames = torch.arange(T, device=mask.device).unsqueeze(1).expand_as(mask)
    first = torch.where(mask, frames, torch.full_like(frames, T))
    first = torch.cat([first, torch.full_like(first[:1], T)])
    return first.flip(0).cummin(dim=0).values.flip(0)




def extend_clauses(cfg):
    unique_atoms = utils_data.load_json(cfg['unique_atoms_file'])
    examples = torch.load(cfg["extension_training_example_file"], map_location="cpu")
    # learn temporal clauses
    # the anchor atom happens at a certain time, and the head atom occurs after a delay
    # the facts those are invariant to the videos within the time window are considered as the other body of the extended clause
    # firstly, for each anchor clause, we need to identify its occurrences in the training examples and the corresponding frames where it appears
    extended_data = _learn_extensions(examples, cfg)
    extended_clauses = _to_extended_clauses(extended_data, unique_atoms, cfg)
    print(f"[Rule extension] Learned {len(extended_clauses)} extended clauses")
    return extended_clauses



def prepare_training_data(cfg):
    """Everything the TemporalRuleNet training samples its batches from.

    Returns {
      "num_atoms": A,
      "videos": {video_id: `_video_cells` of its (T, N, A) atom tensor},
      "frames": {video_id: (F,) long, frames with some visible agent},
      "positive_frames": {video_id: (P,) long, frames where some agent already
                          visible in the previous frame switches an atom on},
      "train_videos" / "val_videos": [video_id], a random split by video,
    }.
    An agent entering the scene switches on all of its atoms at once; those
    are appearances, not events, so they do not make a frame positive.
    """
    atoms = utils_data.load_json(cfg['train_atom_file'])
    num_atoms = len(utils_data.load_json(cfg['unique_atoms_file']))

    videos, frames, positive_frames = {}, {}, {}
    for video_id, video_atoms in tqdm(atoms.items(), desc="Preparing videos"):
        X = build_video_tensor(video_atoms, num_atoms, device=cfg["device"])
        if X.shape[0] < 2:
            continue
        visible = X.any(dim=2)                                      # (T, N)
        stayed = torch.zeros_like(visible)
        stayed[1:] = visible[1:] & visible[:-1]                     # visible in t and t - 1
        has_onset = (atom_onsets(X).any(dim=2) & stayed).any(dim=1)

        videos[video_id] = _video_cells(X)
        frames[video_id] = torch.nonzero(visible.any(dim=1)).flatten().cpu()
        positive_frames[video_id] = torch.nonzero(has_onset).flatten().cpu()

    video_ids = sorted(videos)
    order = torch.randperm(len(video_ids), generator=torch.Generator().manual_seed(cfg["seed"]))
    num_val = max(1, round(cfg["val_video_ratio"] * len(video_ids)))
    val_videos = sorted(video_ids[i] for i in order[:num_val].tolist())
    train_videos = sorted(video_ids[i] for i in order[num_val:].tolist())

    num_frames = sum(len(f) for f in frames.values())
    num_positive = sum(len(f) for f in positive_frames.values())
    print(f"[Videos]\t {len(videos)} ({len(train_videos)} train, {len(val_videos)} val)")
    print(f"[Frames]\t {num_frames} ({num_positive} with onsets)")
    return {
        "num_atoms": num_atoms,
        "videos": videos,
        "frames": frames,
        "positive_frames": positive_frames,
        "train_videos": train_videos,
        "val_videos": val_videos,
    }


def _sample_frames(training_data, video_id, num_frames, positive_ratio, generator):
    """(<= num_frames,) sorted target frames of one video, a share positive_ratio
    of them drawn from its frames with onsets and the rest from all its frames."""
    def pick(pool, k):
        if len(pool) == 0 or k <= 0:
            return pool[:0]
        return pool[torch.randint(len(pool), (k,), generator=generator)]

    positives = pick(training_data["positive_frames"][video_id], round(positive_ratio * num_frames))
    others = pick(training_data["frames"][video_id], num_frames - len(positives))
    return torch.unique(torch.cat([positives, others]))


def _video_rows(model, training_data, video_id, frames, device):
    """Inputs, targets and at-risk masks of `TemporalRuleNet.training_rows` for the
    target frames of one video, rebuilt from its stored atoms."""
    X = _video_from_cells(training_data["videos"][video_id], training_data["num_atoms"], device)
    inputs, targets, at_risk, rows = model.training_rows(X, frames.to(device))
    t, n = rows.T
    # An agent entering the scene in t switches on all of its atoms at once: not events.
    seen = (t > 0) & X[(t - 1).clamp(min=0), n].any(dim=1)
    return inputs[seen], targets[seen], at_risk[seen]


@torch.no_grad()
def _validation_loss(model, training_data, val_frames, cfg):
    """Mean onset loss over the fixed target frames of the validation videos."""
    model.eval()
    losses = []
    for video_id, frames in val_frames.items():
        inputs, targets, at_risk = _video_rows(model, training_data, video_id, frames, cfg["device"])
        if len(inputs) == 0:
            continue
        logits, _rules = model(inputs)
        losses.append(model.loss(logits, targets, at_risk, cfg["pos_weight"]).item())
    return sum(losses) / len(losses) if losses else float("nan")


def train_temporal_rule_net(training_data, cfg, run=None):
    """Train a TemporalRuleNet on the training videos; returns the model with the
    lowest validation loss.

    One epoch visits every training video once, in batches of videos_per_batch
    videos with frames_per_video sampled target frames each. The input rows are
    rebuilt from the stored atoms per batch, never saved. run: the W&B run the
    losses of each epoch are logged to, or None.
    """
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    device = cfg["device"]
    model = TemporalRuleNet(training_data["num_atoms"], cfg["num_lags"], cfg["num_rules"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Fixed validation frames, so the losses of the epochs are comparable.
    val_frames = {
        video_id: _sample_frames(training_data, video_id, cfg["frames_per_video"],
                                 cfg["positive_frame_ratio"], generator)
        for video_id in training_data["val_videos"]
    }
    train_videos = training_data["train_videos"]
    best_val_loss, best_state = float("inf"), None
    for epoch in range(cfg["num_epochs"]):
        model.train()
        order = torch.randperm(len(train_videos), generator=generator).tolist()
        epoch_losses = []
        for start in range(0, len(order), cfg["videos_per_batch"]):
            parts = [
                _video_rows(model, training_data, train_videos[i],
                            _sample_frames(training_data, train_videos[i], cfg["frames_per_video"],
                                           cfg["positive_frame_ratio"], generator),
                            device)
                for i in order[start:start + cfg["videos_per_batch"]]
            ]
            inputs, targets, at_risk = (torch.cat(part) for part in zip(*parts))
            if len(inputs) == 0:
                continue

            logits, _rules = model(inputs)
            onset_loss = model.loss(logits, targets, at_risk, cfg["pos_weight"])
            loss = onset_loss + cfg["l0_lambda"] * model.sparsity_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(onset_loss.item())

        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        val_loss = _validation_loss(model, training_data, val_frames, cfg)
        print(f"[Epoch {epoch + 1}/{cfg['num_epochs']}] train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        if run is not None:
            with torch.no_grad():
                active_inputs = (model.rule_mask() > cfg["mask_threshold"]).sum(dim=1).float()
            run.log({
                "train/onset_loss": train_loss,
                "val/onset_loss": val_loss,
                "model/sparsity_loss": model.sparsity_loss().item(),
                "model/active_inputs_per_rule": active_inputs.mean().item(),
                "model/empty_rules": (active_inputs == 0).sum().item(),
            }, step=epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[Training] Best val loss {best_val_loss:.4f}")
    if run is not None:
        run.summary["best_val_onset_loss"] = best_val_loss
    return model


def load_temporal_rule_net(training_data, cfg):
    """The trained TemporalRuleNet of cfg["model_file"], in eval mode on cfg["device"].

    The model is rebuilt with the same sizes it was trained with (num_atoms of
    the training data, num_lags and num_rules of cfg) before its weights are loaded.
    """
    model = TemporalRuleNet(training_data["num_atoms"], cfg["num_lags"], cfg["num_rules"])
    model.load_state_dict(torch.load(cfg["model_file"], map_location="cpu"))
    return model.to(cfg["device"]).eval()


@torch.no_grad()
def _rule_head_counts(model, training_data, cfg):
    """(R, A) counts of every rule r and head atom h over all frames of the
    training videos: fires, rows where every body atom of r holds and h is at
    risk (off in the previous frame); hits, the fires where h switches on;
    videos, the training videos with at least one hit."""
    num_rules, num_atoms = model.log_alpha.shape[0], model.num_atoms
    fires, hits, videos = (torch.zeros((num_rules, num_atoms), device=cfg["device"]) for _ in range(3))
    for video_id in tqdm(training_data["train_videos"], desc="Counting rule firings"):
        video_hits = torch.zeros_like(hits)
        for frames in training_data["frames"][video_id].split(cfg["frames_per_video"]):
            inputs, targets, at_risk = _video_rows(model, training_data, video_id, frames, cfg["device"])
            if len(inputs) == 0:
                continue
            fired = (model.partial_activation(inputs, cfg["mask_threshold"]) >= 1).float()
            fires += fired.T @ at_risk.float()
            video_hits += fired.T @ targets.float()
        hits += video_hits
        videos += video_hits > 0
    return fires.cpu(), hits.cpu(), videos.cpu()


def _temporal_clause_text(clause, unique_atoms):
    # Bodies earliest first: "Car:MovAway(X0)@-6f -> Ped:Wait(X1)@-2f => Car:Stop(X0)".
    def atom(atom_index, slot):
        return f"{_atom_name(unique_atoms[atom_index])}(X{slot})"
    body = " -> ".join(f"{atom(a, slot)}@-{lag}f" for a, slot, lag in clause["bodies"])
    return f"{body} => {atom(*clause['head'])}"


def extract_temporal_clauses(model, training_data, cfg, run=None):
    """Clauses of the trained model, one per (rule node, head) pair with a large
    head weight, kept if the rule hits the head in at least min_support training
    frames of at least min_videos videos.

    Each clause: bodies [(atom, slot, lag)] earliest first (lag = frames before
    the head; slot 0 the head agent, 1 some other agent), body_scopes their
    slots, head (atom, 0), weight, and on the training videos support (hits),
    fires and confidence = support / fires, num_videos. Sorted by confidence.
    """
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    rules = model.extract_rules(cfg["mask_threshold"], cfg["weight_threshold"])
    fires, hits, videos = _rule_head_counts(model, training_data, cfg)

    clauses = []
    for rule in rules:
        r = rule["rule"]
        for h, weight in rule["heads"]:
            support, num_videos = int(hits[r, h]), int(videos[r, h])
            if support < cfg["min_support"] or num_videos < cfg["min_videos"]:
                continue
            clauses.append({
                "bodies": [list(b) for b in rule["bodies"]],
                "body_scopes": [slot for _atom, slot, _lag in rule["bodies"]],
                "head": [h, 0],
                "rule": r,
                "weight": weight,
                "support": support,
                "fires": int(fires[r, h]),
                "confidence": support / max(int(fires[r, h]), 1),
                "num_videos": num_videos,
            })
    clauses.sort(key=lambda c: (-c["confidence"], -c["support"]))

    print(f"[Rules] {len(rules)} rule nodes with heads -> {len(clauses)} clauses kept")
    for clause in clauses[:15]:
        print(f"  conf {clause['confidence']:.2f}  support {clause['support']:>5}  "
              f"{_temporal_clause_text(clause, unique_atoms)}")
    if run is not None:
        import wandb
        table = wandb.Table(columns=["clause", "confidence", "support", "fires", "num_videos", "weight", "length"])
        for clause in clauses:
            table.add_data(_temporal_clause_text(clause, unique_atoms), clause["confidence"], clause["support"],
                           clause["fires"], clause["num_videos"], clause["weight"], len(clause["bodies"]))
        run.log({"clauses": table})
        run.summary["num_clauses"] = len(clauses)
        run.summary["num_rules_with_heads"] = len(rules)
    return clauses


def main(input_data):
    print(f"----------------- Step 05: Temporal Rule Learning -----------------")
    cfg = {
            'output_dir': input_data['output_dir'],
            'device': input_data['device'],
            'seed': 0,
            # Model: K past frames (input layers) per prediction, R rule nodes.
            'num_lags': 30,
            'num_rules': 256,
            # Training: batches of target frames sampled from a few videos at a time.
            'num_epochs': 50,
            'lr': 0.01,
            'videos_per_batch': 4,
            'frames_per_video': 32,
            # Share of sampled frames where some atom switches on (onsets are rare).
            'positive_frame_ratio': 0.5,
            'pos_weight': 20.0,
            'l0_lambda': 1e-3,
            'val_video_ratio': 0.1,
            # Rule extraction thresholds of the trained model.
            'mask_threshold': 0.5,
            'weight_threshold': 0.1,
            # A clause must hit its head in min_support training frames of min_videos videos.
            'min_support': 10,
            'min_videos': 2,
            'wandb': True,
            'wandb_project': 'cauvid-roadpp',
            'train_atom_file': input_data['train_atom_file'],
            'causal_anchor_clause_file': input_data['causal_anchor_clause_file'],
            "unique_atoms_file": input_data["unique_atoms_file"],
            'training_data_file': input_data["output_dir"] / "training_data.pt",
            'model_file': input_data["output_dir"] / "temporal_rule_net.pt",
            "extended_clause_file": input_data["output_dir"] / "extended_clauses.json"
    }

    # 1. Prepare once: compressed atoms of each video, the frames to sample
    #    (all / with onsets) and the train / validation split by video.
    if not os.path.exists(cfg["training_data_file"]):
        training_data = prepare_training_data(cfg)
        torch.save(training_data, cfg["training_data_file"])
    training_data = torch.load(cfg["training_data_file"], map_location="cpu")

    # One W&B run for the training and extraction below, only if any is not cached.
    run = None
    if not os.path.exists(cfg["model_file"]) or not os.path.exists(cfg["extended_clause_file"]):
        run = _init_wandb(cfg, {"num_train_videos": len(training_data["train_videos"]),
                                "num_val_videos": len(training_data["val_videos"]),
                                "num_atoms": training_data["num_atoms"]},
                          job_type="step05_temporal_rule_net")
    try:
        # 2. Train the TemporalRuleNet; its input rows are built from the atoms per batch.
        if not os.path.exists(cfg["model_file"]):
            model = train_temporal_rule_net(training_data, cfg, run)
            torch.save(model.state_dict(), cfg["model_file"])
        model = load_temporal_rule_net(training_data, cfg)

        # 3. Read the learned rules (ordered bodies with lags -> heads) out of the model.
        if not os.path.exists(cfg["extended_clause_file"]):
            extended_clauses = extract_temporal_clauses(model, training_data, cfg, run)
            utils_data.save_json(extended_clauses, cfg["extended_clause_file"])
    finally:
        if run is not None:
            run.finish()
