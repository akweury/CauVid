import math
import os
import sys
import getpass
import torch
from tqdm import tqdm

from src.exp_roadpp import utils_data
from src.exp_roadpp.cauvid import TemporalRuleNet
from src.exp_roadpp.baselines.onset_mlp import OnsetMLP


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


def _atom_name(signature):
    # ["pred", "action", "action_id", 3, "action_id_label", "MovAway",
    #  "agent_class", 1, "agent_class_label", "Car"] -> "Car:MovAway"
    fields = dict(zip(signature[2::2], signature[3::2]))
    labels = [str(v) for k, v in fields.items() if k.endswith("_label") and k != "agent_class_label"]
    return f"{fields.get('agent_class_label', '?')}:{'/'.join(labels) or signature[1]}"


_WANDB_CLOUD_HOST = "https://api.wandb.ai"


def _wandb_login(wandb):
    """Log in to the hosted W&B cloud; returns False if the user skips tracking.

    Asks the user to pick a login method: the saved credentials (~/.netrc or
    WANDB_API_KEY) or an API key typed in (hidden), from
    https://wandb.ai/authorize. Without a terminal, only saved credentials are tried.
    A WANDB_API_KEY in the environment (e.g. forwarded into Docker) is used without asking.
    """
    if not sys.stdin.isatty() or os.environ.get("WANDB_API_KEY"):
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


def atom_onsets(X, min_off_frames=1):
    """(T, N, A) bool: atom a switches on for agent n in frame t, after being off
    for at least the min_off_frames frames before t."""
    held = torch.zeros_like(X)
    for k in range(1, min_off_frames + 1):
        held[k:] |= X[:-k]
    return X & ~held


def is_fresh(output_file, *source_files):
    """True if output_file exists and is newer than every existing source file."""
    if not os.path.exists(output_file):
        return False
    return all(os.path.getmtime(output_file) >= os.path.getmtime(f) for f in source_files if os.path.exists(f))


def prepare_videos(atom_file, num_atoms, device, min_off_frames=1):
    """Compressed atoms and frame lists of every video of one split.

    Returns {
      "num_atoms": A,
      "min_off_frames": frames an atom must be off before it can switch on again,
      "videos": {video_id: `_video_cells` of its (T, N, A) atom tensor},
      "frames": {video_id: (F,) long, frames with some visible agent},
      "positive_frames": {video_id: (P,) long, frames where some agent already
                          visible in the previous frame switches an atom on},
    }; videos shorter than 2 frames are left out.
    An agent entering the scene switches on all of its atoms at once; those
    are appearances, not events, so they do not make a frame positive. Neither
    does an atom back on after a gap shorter than min_off_frames (label flicker).
    """
    atoms = utils_data.load_json(atom_file)
    videos, frames, positive_frames = {}, {}, {}
    for video_id, video_atoms in tqdm(atoms.items(), desc="Preparing videos"):
        X = build_video_tensor(video_atoms, num_atoms, device=device)
        if X.shape[0] < 2:
            continue
        visible = X.any(dim=2)                                      # (T, N)
        stayed = torch.zeros_like(visible)
        stayed[1:] = visible[1:] & visible[:-1]                     # visible in t and t - 1
        has_onset = (atom_onsets(X, min_off_frames).any(dim=2) & stayed).any(dim=1)

        videos[video_id] = _video_cells(X)
        frames[video_id] = torch.nonzero(visible.any(dim=1)).flatten().cpu()
        positive_frames[video_id] = torch.nonzero(has_onset).flatten().cpu()
    return {"num_atoms": num_atoms, "min_off_frames": min_off_frames, "videos": videos, "frames": frames,
            "positive_frames": positive_frames}


# Marks how training_data.pt was split, so a file made another way is rebuilt.
SPLIT_SOURCE = "dataset train / dataset val"


def prepare_training_data(cfg):
    """Everything the TemporalRuleNet training samples its batches from: the
    `prepare_videos` data of the dataset's train split ("train_videos", all of
    them trained on) and of its val split ("val_videos": model selection, the
    cascade thresholds and the step 06 alarm thresholds), in one dict."""
    num_atoms = len(utils_data.load_json(cfg['unique_atoms_file']))
    train = prepare_videos(cfg['train_atom_file'], num_atoms, cfg["device"], cfg["min_off_frames"])
    val = prepare_videos(cfg['val_atom_file'], num_atoms, cfg["device"], cfg["min_off_frames"])
    shared = set(train["videos"]) & set(val["videos"])
    if shared:
        raise ValueError(f"{len(shared)} videos are in both the train and the val split, e.g. {sorted(shared)[:3]}.")
    data = {"num_atoms": num_atoms, "min_off_frames": cfg["min_off_frames"], "split": SPLIT_SOURCE}
    for key in ("videos", "frames", "positive_frames"):
        data[key] = {**train[key], **val[key]}
    train_videos, val_videos = sorted(train["videos"]), sorted(val["videos"])

    num_frames = sum(len(f) for f in data["frames"].values())
    num_positive = sum(len(f) for f in data["positive_frames"].values())
    print(f"[Videos]\t {len(train_videos)} train, {len(val_videos)} val (the dataset splits)")
    print(f"[Frames]\t {num_frames} ({num_positive} with onsets)")
    return {**data, "train_videos": train_videos, "val_videos": val_videos}


def load_training_data(cfg):
    """training_data.pt, (re)built by `prepare_training_data` if it is missing,
    has another min_off_frames or was split another way. Adds the event types of
    the heads ("event_types", `head_event_types`) and "target_heads" (A,) bool,
    the heads the models learn: all but the noise heads if cfg["exclude_noise_heads"]."""
    data = torch.load(cfg["training_data_file"], map_location="cpu") if os.path.exists(cfg["training_data_file"]) else {}
    if data.get("min_off_frames") != cfg["min_off_frames"] or data.get("split") != SPLIT_SOURCE:
        data = prepare_training_data(cfg)
        torch.save(data, cfg["training_data_file"])
    data["event_types"] = head_event_types(data, utils_data.load_json(cfg["unique_atoms_file"]), cfg)
    noise = data["event_types"]["types"] == EVENT_TYPES.index("noise")
    data["target_heads"] = ~noise if cfg["exclude_noise_heads"] else torch.ones_like(noise)
    data["pos_weight_scale"] = head_pos_weight_scale(data["event_types"], cfg)
    print(f"[Heads]\t {int(data['target_heads'].sum())} of {data['num_atoms']} heads are learned"
          + (f" ({int(noise.sum())} noise heads, < {cfg['noise_below']} train + val onsets, left out)"
             if cfg["exclude_noise_heads"] else ""))
    return data


# Event types by the frequency of their head (onsets in the train + val videos), see
def head_pos_weight_scale(event_types, cfg):
    """(A,) factor of every head's onset weight (pos_weight): with cfg["type_balance"], an
    event type whose heads have fewer onsets each (mean train + val onsets per head) weighs
    its onsets up, (M_max / M_type) ** power capped at type_balance_max, M_max the mean of
    the type with the most onsets per head (trivial), so the frequent trivial heads no longer
    dominate the loss, the sparsity budget and the rules; else 1 for every head. The mean per
    head, not the type's total: a type of few heads (medium: 2 trivial heads, 278 onsets, vs
    17 regular heads, 485) is still the frequent one. The noise heads (no targets) keep 1."""
    types = event_types["types"]
    scale = torch.ones(len(types))
    if not cfg.get("type_balance"):
        return scale
    means = {t: sum(h["onsets"] for h in event_types["heads"][t]) / len(event_types["heads"][t])
             for t in EVENT_TYPES if t != "noise" and event_types["heads"][t]}
    reference = max(means.values())
    for g, t in enumerate(EVENT_TYPES):
        if t in means and means[t] > 0:
            scale[types == g] = min((reference / means[t]) ** cfg["type_balance_power"], cfg["type_balance_max"])
    print("[Type balance]\t onset weight x " + ", ".join(
        f"{t} {scale[types == EVENT_TYPES.index(t)][0]:.2f} ({means[t]:.1f} onsets per head)"
        for t in means if (types == EVENT_TYPES.index(t)).any()))
    return scale


def pos_weight_of(training_data, cfg):
    """The onset weight of the loss: cfg["pos_weight"], times the per-head type balance
    (`head_pos_weight_scale`, (A,) on cfg["device"]) only for the main rule net
    (cfg["balanced"], set by `train_temporal_rule_net`); every other model is unweighted."""
    scale = training_data.get("pos_weight_scale")
    if scale is None or not cfg.get("balanced"):
        return cfg["pos_weight"]
    return cfg["pos_weight"] * scale.to(cfg["device"])


def type_balance_tag(cfg):
    """File tag suffix of the type balance (per-head mean), "" without it."""
    return f"_tbh{cfg['type_balance_power']:g}m{cfg['type_balance_max']:g}" if cfg.get("type_balance") else ""


# `head_event_types`: heads with < noise_below onsets are "noise", < rare_below "rare",
# > trivial_above "trivial", the rest "regular". The rare heads are training targets like the
# regular ones; step 06 only reports them apart (too few onsets for reliable rules).
EVENT_TYPES = ("noise", "rare", "regular", "trivial")


def head_event_types(training_data, unique_atoms, cfg):
    """The event type of every head by its frequency: its onsets (on after
    min_off_frames off, the agent visible in the frame before) in the train + val
    videos: fewer than cfg["noise_below"] onsets (every head never seen included)
    is "noise", fewer than cfg["rare_below"] "rare", more than cfg["trivial_above"]
    "trivial", the rest "regular".
    Returns {"types": (A,) long EVENT_TYPES index, "heads": {type: [{"head",
    "label", "onsets"}], most frequent first}, the two cutoffs and the onset count}."""
    A = training_data["num_atoms"]
    counts = torch.zeros(A)
    for video in training_data["videos"].values():
        X = _video_from_cells(video, A, "cpu")
        onsets = atom_onsets(X, training_data["min_off_frames"])
        onsets[0] = False
        onsets[1:] &= X[:-1].any(dim=2, keepdim=True)
        counts += onsets.sum(dim=(0, 1)).float()
    types = torch.full((A,), EVENT_TYPES.index("regular"), dtype=torch.long)
    types[counts < cfg.get("rare_below", 0)] = EVENT_TYPES.index("rare")
    types[counts < cfg["noise_below"]] = EVENT_TYPES.index("noise")
    types[counts > cfg["trivial_above"]] = EVENT_TYPES.index("trivial")
    order = torch.argsort(counts, descending=True, stable=True)
    heads = {t: [{"head": int(a), "label": _atom_name(unique_atoms[a]), "onsets": int(counts[a])}
                 for a in order.tolist() if types[a] == g]
             for g, t in enumerate(EVENT_TYPES)}
    return {"types": types, "heads": heads, "noise_below": cfg["noise_below"], "rare_below": cfg.get("rare_below"),
            "trivial_above": cfg["trivial_above"],
            "count_source": "train + val onsets", "num_onsets": int(counts.sum())}


def _sample_frames(training_data, video_id, num_frames, positive_ratio, generator, horizon=1):
    """(<= num_frames,) sorted target frames of one video, a share positive_ratio
    of them drawn from its frames with onsets and the rest from all its frames.
    A positive is moved 0..horizon - 1 frames before its onset, where the onset
    is still ahead within the largest window."""
    def pick(pool, k):
        if len(pool) == 0 or k <= 0:
            return pool[:0]
        return pool[torch.randint(len(pool), (k,), generator=generator)]

    positives = pick(training_data["positive_frames"][video_id], round(positive_ratio * num_frames))
    positives = (positives - torch.randint(horizon, positives.shape, generator=generator)).clamp(min=0)
    others = pick(training_data["frames"][video_id], num_frames - len(positives))
    return torch.unique(torch.cat([positives, others]))


def _video_rows(model, training_data, video_id, frames, device, history=None):
    """`video_rows` of the target frames of one stored video, rebuilt from its atoms.
    Heads outside training_data["target_heads"] (the noise heads, see
    `load_training_data`) are never at risk and never targets, so no loss, count
    or clause covers them."""
    X = _video_from_cells(training_data["videos"][video_id], training_data["num_atoms"], device)
    rows = video_rows(model, X, frames.to(device), training_data.get("min_off_frames", 1),
                      None if history is None else history.to(device))
    if "target_heads" in training_data:
        target_heads = training_data["target_heads"].to(device)
        rows["at_risk"] = rows["at_risk"] & target_heads
        rows["targets"] = rows["targets"] & target_heads[None, None, :]
    return rows


def video_rows(model, X, frames, min_off_frames=1, history=None):
    """`TemporalRuleNet.training_rows` (a dict of inputs, rows, at_risk, first,
    targets, known) of the target frames of one (T, N, A) video tensor, without
    the agents entering the scene: their atoms all switch on at once in t, which
    are not events. history (F,): the frames known before each target frame
    (default all; older ones are unknown inputs)."""
    rows = model.training_rows(X, frames, min_off_frames, history)
    t, n = rows["rows"].T
    seen = (t > 0) & X[(t - 1).clamp(min=0), n].any(dim=1)
    return {key: value[seen] for key, value in rows.items()}


def sample_history(frames, cfg, generator):
    """(F,) known history of target frames for training: with probability
    truncate_prob a random 0..memory frames (the model learns to predict from
    partial facts, with lower confidence), else everything before the frame."""
    memory = cfg["slot_size"] * cfg["num_slots"]
    truncated = torch.rand(len(frames), generator=generator) < cfg["truncate_prob"]
    return torch.where(truncated, torch.randint(memory + 1, (len(frames),), generator=generator), frames.cpu())


def concat_rows(parts):
    """One rows dict of several `video_rows` dicts."""
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def coarsen_video(X, block_size):
    """(ceil(T / s), N, A) bool: the atom holds in some frame of each block of
    s = block_size frames (frames past the video end count as empty).
    Used by step05_layer / step06_layer."""
    pad = (-X.shape[0]) % block_size
    X = torch.cat([X, X.new_zeros((pad, *X.shape[1:]))])
    return X.view(-1, block_size, *X.shape[1:]).any(dim=1)


@torch.no_grad()
def _validation_loss(model, training_data, val_frames, cfg, candidates=None):
    """Mean window loss over the fixed target frames of the validation videos
    (only the candidate cells, if candidates is given; see `fit_onset_model`)."""
    model.eval()
    losses = []
    for video_id, (frames, history) in val_frames.items():
        rows = _video_rows(model, training_data, video_id, frames, cfg["device"], history)
        if len(rows["inputs"]) == 0:
            continue
        at_risk = rows["at_risk"] if candidates is None else rows["at_risk"] & candidates(rows["inputs"])
        hazards, _rules = model(rows["inputs"])
        losses.append(model.loss(hazards, rows["targets"], at_risk, rows["known"],
                                 pos_weight_of(training_data, cfg)).item())
    return sum(losses) / len(losses) if losses else float("nan")


def anchor_rules(cfg):
    """[((atom, slot), [head atom])] of the anchor clauses, one per body, in the
    order their rules are seeded: rules i * B..i * B + B - 1 belong to anchor i."""
    body_groups = group_clauses_by_body(utils_data.load_json(cfg["causal_anchor_clause_file"]))
    return [(body, [head[0] for head in heads]) for body, heads in body_groups.items()]


def fit_onset_model(model, training_data, cfg, optimizer, history_file, regularization=None, run=None,
                    model_stats=None, candidates=None):
    """Train an onset model (TemporalRuleNet or the OnsetMLP baseline) on the
    training videos; returns it with the weights of the lowest validation loss.

    One epoch visits every training video once, in batches of videos_per_batch
    videos with frames_per_video sampled target frames each. The input rows are
    rebuilt from the stored atoms per batch, never saved. regularization(epoch):
    the extra loss term of an epoch, or None; run: the W&B run the losses of each
    epoch are logged to, or None; model_stats(): extra values logged per epoch;
    candidates(inputs) -> (B, A) bool: if given, the loss only covers these
    (row, head) cells (the cascade verifier learns on the stage-1 candidates).
    The targets are the onsets within each of cfg["windows"] frames; the known
    history of each target frame is truncated at random (`sample_history`).
    """
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    device = cfg["device"]
    # Optional (the OnsetMLP sets them): gradient-norm clipping, halving the learning rate
    # after lr_patience epochs without a better val loss, and early stopping after patience.
    grad_clip, patience = cfg.get("grad_clip"), cfg.get("patience")
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=cfg["lr_patience"])
                 if cfg.get("lr_patience") else None)
    best_epoch = 0
    pos_weight = pos_weight_of(training_data, cfg)                                   # scalar or (A,)
    # Fixed validation frames and known histories, so the losses of the epochs are comparable.
    val_frames = {}
    for video_id in training_data["val_videos"]:
        frames = _sample_frames(training_data, video_id, cfg["frames_per_video"], cfg["positive_frame_ratio"],
                                generator, max(cfg["windows"]))
        val_frames[video_id] = (frames, sample_history(frames, cfg, generator))
    train_videos = training_data["train_videos"]
    best_val_loss, best_state = float("inf"), None
    history = []
    for epoch in range(cfg["num_epochs"]):
        model.train()
        order = torch.randperm(len(train_videos), generator=generator).tolist()
        epoch_losses = []
        for start in range(0, len(order), cfg["videos_per_batch"]):
            parts = []
            for i in order[start:start + cfg["videos_per_batch"]]:
                frames = _sample_frames(training_data, train_videos[i], cfg["frames_per_video"],
                                        cfg["positive_frame_ratio"], generator, max(cfg["windows"]))
                parts.append(_video_rows(model, training_data, train_videos[i], frames, device,
                                         sample_history(frames, cfg, generator)))
            rows = concat_rows(parts)
            if len(rows["inputs"]) == 0:
                continue
            at_risk = rows["at_risk"]
            if candidates is not None:
                at_risk = at_risk & candidates(rows["inputs"])

            hazards, _hidden = model(rows["inputs"])
            onset_loss = model.loss(hazards, rows["targets"], at_risk, rows["known"], pos_weight)
            loss = onset_loss if regularization is None else onset_loss + regularization(epoch)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(onset_loss.item())

        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        val_loss = _validation_loss(model, training_data, val_frames, cfg, candidates)
        print(f"[Epoch {epoch + 1}/{cfg['num_epochs']}] train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if run is not None:
            run.log({"train/onset_loss": train_loss, "val/onset_loss": val_loss,
                     **(model_stats() if model_stats else {})}, step=epoch)
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if scheduler is not None:
            scheduler.step(val_loss)
            history[-1]["lr"] = optimizer.param_groups[0]["lr"]
        if patience and epoch + 1 - best_epoch >= patience:
            print(f"[Training] Early stop: no better val loss for {patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[Training] Best val loss {best_val_loss:.4f} (epoch {best_epoch})")
    utils_data.save_json(history, history_file)
    if run is not None:
        run.summary["best_val_onset_loss"] = best_val_loss
    return model


def _seeded_rule_net(training_data, cfg):
    """A TemporalRuleNet with one rule node per (anchor body, slot offset), plus
    num_free_rules unseeded ones, on cfg["device"]."""
    anchors = anchor_rules(cfg)
    model = TemporalRuleNet(training_data["num_atoms"], cfg["slot_size"], cfg["num_slots"],
                            len(anchors) * cfg["num_slots"] + cfg["num_free_rules"], cfg["init_log_alpha"],
                            cfg["mask_type"], cfg["windows"])
    num_seeded = model.seed_rules(anchors, cfg["seed_head_weight"])
    print(f"[Model] {num_seeded} rules seeded from {len(anchors)} anchor bodies, {cfg['num_free_rules']} free, "
          f"{cfg['mask_type']} mask")
    return model.to(cfg["device"])


def _fit_rule_net(model, training_data, cfg, history_file, run=None, candidates=None):
    """`fit_onset_model` of a TemporalRuleNet, with its sparsity loss after
    l0_warmup_epochs; the sigmoid gates get their own learning rate (gate_lr,
    and a tiny gate_eps: their gradients start at ~1e-12, sigmoid' of -11)."""
    def regularization(epoch):
        # No sparsity pressure in the first epochs, so useful inputs can grow first.
        return (cfg["l0_lambda"] if epoch >= cfg["l0_warmup_epochs"] else 0.0) * model.sparsity_loss()

    @torch.no_grad()
    def model_stats():
        active = (model.rule_mask() > cfg["mask_threshold"]) & ~model.fixed_inputs
        extra = active.sum(dim=1).float()
        gates = model.log_alpha[~model.fixed_inputs]
        return {"model/sparsity_loss": model.sparsity_loss().item(),
                "model/extra_inputs_per_rule": extra.mean().item(),
                "model/max_extra_inputs": extra.max().item(),
                "model/share_rules_extended": (extra > 0).float().mean().item(),
                "gates/max_log_alpha": gates.max().item(),
                "gates/count_above_0": (gates > 0).sum().item()}

    if cfg["mask_type"] == "sigmoid":
        others = [p for name, p in model.named_parameters() if name != "log_alpha"]
        optimizer = torch.optim.Adam([{"params": [model.log_alpha], "lr": cfg["gate_lr"], "eps": cfg["gate_eps"]},
                                      {"params": others, "lr": cfg["lr"]}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    return fit_onset_model(model, training_data, cfg, optimizer, history_file, regularization, run, model_stats,
                           candidates)


def train_temporal_rule_net(training_data, cfg, run=None):
    """Stage 1 of the cascade: the recall model, a TemporalRuleNet seeded from the
    anchor clauses plus free rules for the heads without an anchor."""
    # Only the main rule net has the type-balanced onset weights (`pos_weight_of`).
    return _fit_rule_net(_seeded_rule_net(training_data, cfg), training_data, {**cfg, "balanced": True},
                         cfg["history_file"], run)


def load_temporal_rule_net(training_data, cfg, model_file=None):
    """The trained TemporalRuleNet of model_file (default cfg["model_file"]), in
    eval mode on cfg["device"].

    The model is rebuilt with the same sizes it was trained with (num_atoms of
    the training data, slots, mask_type and windows of cfg, the number of rules of the
    saved weights) before its weights, and its seeded anchor inputs, are loaded.
    """
    state = torch.load(model_file or cfg["model_file"], map_location="cpu")
    model = TemporalRuleNet(training_data["num_atoms"], cfg["slot_size"], cfg["num_slots"],
                            state["log_alpha"].shape[0], mask_type=cfg["mask_type"], windows=cfg["windows"])
    model.load_state_dict(state)
    return model.to(cfg["device"]).eval()


# -------------- Cascade: stage-1 candidates and the stage-2 verifier --------------
@torch.no_grad()
def candidate_thresholds(model, training_data, cfg):
    """Per-head stage-1 score thresholds that keep cascade_recall of the onset
    rows of the validation videos, on the largest window (rows with the head
    switching on within max(windows) frames); a head with fewer than
    cascade_min_positives such rows gets the threshold of all heads together.

    Returns {"thresholds": (A,) CPU, "recall": the target, "val_row_recall" and
    "val_candidate_share": the share of the onset rows and of all at-risk
    (row, head) cells with an observed largest window at or above the thresholds}.
    """
    model.eval()
    device, A = cfg["device"], training_data["num_atoms"]

    def rows(video_id):
        for frames in training_data["frames"][video_id].split(cfg["frames_per_video"]):
            r = _video_rows(model, training_data, video_id, frames, device)
            if len(r["inputs"]):
                scores = model.window_probs(model(r["inputs"])[0])[:, -1]
                yield scores, r["targets"][:, -1], r["at_risk"] & r["known"][:, -1]

    heads, values = [], []
    for video_id in tqdm(training_data["val_videos"], desc="Stage-1 scores of the val onsets"):
        for scores, targets, _cells in rows(video_id):
            r, h = torch.nonzero(targets, as_tuple=True)
            heads.append(h)
            values.append(scores[r, h])
    heads, values = (torch.cat(x).cpu() if x else torch.zeros(0) for x in (heads, values))
    q = 1 - cfg["cascade_recall"]
    thresholds = torch.full((A,), torch.quantile(values, q).item() if len(values) else 0.0)
    for h in heads.unique().tolist():
        if (heads == h).sum() >= cfg["cascade_min_positives"]:
            thresholds[h] = torch.quantile(values[heads == h], q)

    kept, onsets, candidates, cells = 0, 0, 0, 0
    on_device = thresholds.to(device)
    for video_id in tqdm(training_data["val_videos"], desc="Stage-1 candidates of val"):
        for scores, targets, at_risk in rows(video_id):
            flagged = scores >= on_device
            kept += (flagged & targets).sum().item()
            onsets += targets.sum().item()
            candidates += (flagged & at_risk).sum().item()
            cells += at_risk.sum().item()
    return {"thresholds": thresholds, "recall": cfg["cascade_recall"],
            "val_row_recall": kept / max(onsets, 1), "val_candidate_share": candidates / max(cells, 1)}


def make_candidates(model, thresholds):
    """candidates(inputs) -> (B, A) bool: the stage-1 model flags the head (its
    probability of an onset within the largest window is at or above the head's threshold)."""
    model.eval()

    @torch.no_grad()
    def candidates(inputs):
        return model.window_probs(model(inputs)[0])[:, -1] >= thresholds.to(inputs.device)
    return candidates


def train_verifier(training_data, cfg, candidates, run=None):
    """Stage 2 of the cascade: a TemporalRuleNet seeded like stage 1, trained only
    on the stage-1 candidate cells with verifier_pos_weight, so its rules (the
    anchors extended by the atoms that separate true from false candidates)
    learn to reject the stage-1 false alarms."""
    verifier_cfg = {**cfg, "pos_weight": cfg["verifier_pos_weight"]}
    return _fit_rule_net(_seeded_rule_net(training_data, cfg), training_data, verifier_cfg,
                         cfg["verifier_history_file"], run, candidates)


# -------------- Black-box baseline --------------
def train_onset_mlp(training_data, cfg, run=None):
    """Train the OnsetMLP baseline (`fit_onset_model`) on the same input rows,
    targets and batches as the TemporalRuleNet, with weight decay instead of
    the L0 loss and no anchor seeding. Its hidden size defaults to the number of
    rule nodes of the TemporalRuleNet, so both have about as many parameters."""
    hidden_size = cfg["mlp_hidden_size"] or len(anchor_rules(cfg)) * cfg["num_slots"] + cfg["num_free_rules"]
    model = OnsetMLP(training_data["num_atoms"], cfg["slot_size"], cfg["num_slots"], hidden_size, cfg["mlp_dropout"],
                     cfg["windows"], cfg["mlp_activation"]).to(cfg["device"])
    print(f"[Baseline] OnsetMLP with {hidden_size} {cfg['mlp_activation']} hidden units, lr {cfg['mlp_lr']}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["mlp_lr"], weight_decay=cfg["mlp_weight_decay"])
    mlp_cfg = {**cfg, "grad_clip": cfg["mlp_grad_clip"], "lr_patience": cfg["mlp_lr_patience"],
               "patience": cfg["mlp_patience"]}

    @torch.no_grad()
    def model_stats():
        # Hidden units that never vary would leave one constant score per head (the prior).
        return {"model/hidden_weight_norm": model.hidden.weight.norm().item()}
    return fit_onset_model(model, training_data, mlp_cfg, optimizer, cfg["mlp_history_file"], run=run,
                           model_stats=model_stats)


def load_onset_mlp(training_data, cfg):
    """The trained OnsetMLP of cfg["mlp_model_file"], in eval mode on cfg["device"],
    or None if it was not trained."""
    if not os.path.exists(cfg["mlp_model_file"]):
        return None
    state = torch.load(cfg["mlp_model_file"], map_location="cpu")
    model = OnsetMLP(training_data["num_atoms"], cfg["slot_size"], cfg["num_slots"], state["hidden.weight"].shape[0],
                     windows=cfg["windows"], activation=cfg["mlp_activation"])
    model.load_state_dict(state)
    return model.to(cfg["device"]).eval()


@torch.no_grad()
def _rule_head_counts(model, training_data, video_ids, cfg, desc="Counting rule firings", candidates=None):
    """Counts of every rule r and head atom h over all frames of the given videos
    (with their full history; only the cells candidates(inputs) flags, if given:
    the verifier's rules are judged on the stage-1 candidates they filter), for
    each alignment f = 0..J-1: the head in future slot f, so the bodies with
    j > f are in the memory and the f slots of bodies closest to the head have
    not happened yet. f = 0 is the whole clause (every body seen, head within the
    next slot); a larger f is the same clause known only up to its earlier bodies.

    Returns a dict of CPU tensors:
      fires (R, J, A): rows where every body of r with j > f holds (r has one) and
          h is at risk, has not switched on before slot f and slot f is observed;
      hits (R, J, A): the fires whose first onset of h falls in slot f;
      videos (R, J, A): the videos with a hit;
      anchor_fires / anchor_hits (R, J, A): the same for the anchor input of r
          alone (its fixed input, without the learned extra atoms; 0 for free rules);
      onsets (J, A) / at_risk (J, A): the hits and fires of an empty body (the base rate).
    """
    J, S = model.num_slots, model.slot_size
    num_rules, num_atoms = model.log_alpha.shape[0], model.num_atoms
    device = cfg["device"]
    shape = (num_rules, J * num_atoms)
    counts = {name: torch.zeros(shape, device=device)
              for name in ("fires", "hits", "videos", "anchor_fires", "anchor_hits")}
    counts["onsets"] = torch.zeros(J * num_atoms, device=device)
    counts["at_risk"] = torch.zeros(J * num_atoms, device=device)
    def in_memory(rule_inputs, f):
        # Bodies with j > f, moved to the past slot p = j - f they lie in for a head in slot f.
        memory = torch.zeros((num_rules, model.num_inputs + 1), device=device)
        memory[:, model.shift_index[f]] = (rule_inputs & (model.input_offset > f)).float()
        return memory[:, :-1]

    body = model.rule_mask() > cfg["mask_threshold"]
    known_bodies = [in_memory(body, f) for f in range(J)]
    known_anchors = [in_memory(model.fixed_inputs, f) for f in range(J)]
    for video_id in tqdm(video_ids, desc=desc):
        video_hits = torch.zeros(shape, device=device)
        for frames in training_data["frames"][video_id].split(cfg["frames_per_video"]):
            rows = _video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            at_risk = rows["at_risk"]
            if candidates is not None:
                at_risk = at_risk & candidates(rows["inputs"])
            x, _k = model.split_inputs(rows["inputs"])
            first = rows["first"]
            for f in range(J):
                risk = (at_risk & (first >= f * S) & rows["known"][:, f]).float()          # (B, A)
                onset = (at_risk & (first >= f * S) & (first < (f + 1) * S)).float()
                cols = slice(f * num_atoms, (f + 1) * num_atoms)
                for prefix, b in (("", known_bodies[f]), ("anchor_", known_anchors[f])):
                    size = b.sum(dim=1)
                    fired = ((x @ b.T >= size) & (size > 0)).float()                      # (B, R)
                    counts[prefix + "fires"][:, cols] += fired.T @ risk
                    if prefix:
                        counts["anchor_hits"][:, cols] += fired.T @ onset
                    else:
                        video_hits[:, cols] += fired.T @ onset
                counts["onsets"][cols] += onset.sum(dim=0)
                counts["at_risk"][cols] += risk.sum(dim=0)
        counts["hits"] += video_hits
        counts["videos"] += video_hits > 0
    out = {name: value.view(num_rules, J, num_atoms) for name, value in counts.items()
           if name not in ("onsets", "at_risk")}
    out["onsets"] = counts["onsets"].view(J, num_atoms)
    out["at_risk"] = counts["at_risk"].view(J, num_atoms)
    return {name: value.cpu() for name, value in out.items()}


def _temporal_clause_text(clause, unique_atoms):
    # Bodies earliest first, in frames before the head's 10-frame slot:
    # "Car:MovAway(X0)@-30..-20f -> Ped:Wait(X1)@-10..0f => Car:Stop(X0)".
    def atom(atom_index, slot):
        return f"{_atom_name(unique_atoms[atom_index])}(X{slot})"
    size = clause.get("slot_size", 10)
    body = " -> ".join(f"{atom(a, slot)}@-{size * j}..-{size * (j - 1)}f" for a, slot, j in clause["bodies"])
    return f"{body} => {atom(*clause['head'])}"


def extract_temporal_clauses(model, training_data, cfg, run=None, candidates=None):
    """Clauses of the trained model, one per (rule node, head) pair with a large
    head weight, kept if the rule hits the head in at least min_support training
    frames of at least min_videos videos (every body seen, head in the next slot).

    Each clause: bodies [(atom, slot, j)] earliest first (the atom holds in the
    slot j slots before the head's; slot 0 the head agent, 1 some other agent),
    body_scopes their slots, head (atom, 0), anchor (atom, slot, j) its seeded
    anchor input or None, weight, and on the training videos support (hits),
    fires, confidence = support / fires, num_videos, anchor_confidence (of the
    anchor input alone). The val_ fields are the same on the validation videos,
    plus val_recall (the share of the head's onsets the clause hits) and
    val_base_rate. partial (train and val): per alignment f, the confidence of
    the head f slots ahead knowing only the bodies with j > f (known_bodies of
    them), with the base rate: how the confidence grows as more bodies are seen.
    Sorted by confidence. With candidates (the verifier), all counts are over
    the stage-1 candidate cells.
    """
    def ratio(a, b):
        return float(a) / float(b) if b > 0 else None

    rules = model.extract_rules(cfg["mask_threshold"], cfg["weight_threshold"])
    train = _rule_head_counts(model, training_data, training_data["train_videos"], cfg,
                              desc="Counting rule firings (train)", candidates=candidates)
    val = _rule_head_counts(model, training_data, training_data["val_videos"], cfg,
                            desc="Counting rule firings (val)", candidates=candidates)

    clauses = []
    for rule in rules:
        r = rule["rule"]
        anchor_inputs = torch.nonzero(model.fixed_inputs[r]).flatten().tolist()
        anchor = list(model.decode_input(anchor_inputs[0])) if anchor_inputs else None
        offsets = [j for _atom, _slot, j in rule["bodies"]]
        for h, weight in rule["heads"]:
            support, num_videos = int(train["hits"][r, 0, h]), int(train["videos"][r, 0, h])
            if support < cfg["min_support"] or num_videos < cfg["min_videos"]:
                continue
            partial = []
            for f in range(max(offsets)):
                partial.append({
                    "unseen_slots": f,
                    "known_bodies": sum(j > f for j in offsets),
                    "confidence": ratio(train["hits"][r, f, h], train["fires"][r, f, h]),
                    "val_confidence": ratio(val["hits"][r, f, h], val["fires"][r, f, h]),
                    "base_rate": ratio(train["onsets"][f, h], train["at_risk"][f, h]),
                })
            clauses.append({
                "bodies": [list(b) for b in rule["bodies"]],
                "body_scopes": [slot for _atom, slot, _j in rule["bodies"]],
                "head": [h, 0],
                "slot_size": model.slot_size,
                "anchor": anchor,
                "rule": r,
                "weight": weight,
                "support": support,
                "fires": int(train["fires"][r, 0, h]),
                "confidence": support / max(int(train["fires"][r, 0, h]), 1),
                "num_videos": num_videos,
                "anchor_confidence": ratio(train["anchor_hits"][r, 0, h], train["anchor_fires"][r, 0, h]),
                "val_support": int(val["hits"][r, 0, h]),
                "val_fires": int(val["fires"][r, 0, h]),
                "val_confidence": ratio(val["hits"][r, 0, h], val["fires"][r, 0, h]),
                "val_anchor_confidence": ratio(val["anchor_hits"][r, 0, h], val["anchor_fires"][r, 0, h]),
                "val_recall": ratio(val["hits"][r, 0, h], val["onsets"][0, h]),
                "val_base_rate": ratio(val["onsets"][0, h], val["at_risk"][0, h]),
                "partial": partial,
            })
    clauses.sort(key=lambda c: (-c["confidence"], -c["support"]))

    print(f"[Rules] {len(rules)} rule nodes with heads -> {len(clauses)} clauses kept")
    if run is not None:
        import wandb
        table = wandb.Table(columns=["clause", "confidence", "support", "fires", "num_videos", "weight", "length"])
        unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
        for clause in clauses:
            table.add_data(_temporal_clause_text(clause, unique_atoms), clause["confidence"], clause["support"],
                           clause["fires"], clause["num_videos"], clause["weight"], len(clause["bodies"]))
        run.log({"clauses": table})
        run.summary["num_clauses"] = len(clauses)
        run.summary["num_rules_with_heads"] = len(rules)
    return clauses


# -------------- Plot helpers (shared with step 06) --------------
# Confidences, precisions and base rates span 1e-6..1, so they go on log axes;
# a 0 cannot be shown there and is drawn on the lower edge instead.
def log_floor(values, floor=1e-6):
    """Lower limit of a log axis: the decade at or below the smallest positive value."""
    positive = [v for v in values if v is not None and v > floor]
    return 10 ** math.floor(math.log10(min(positive))) if positive else floor


def clip_low(values, low):
    """Values with None / 0 / anything below low raised to low (drawn on the axis edge)."""
    return [max(v or 0.0, low) for v in values]


def clause_scatter(ax, rows, x_key, y_key):
    """Log-log scatter of two confidences of every clause, one color per body
    length, dashed y = x; 0 is drawn on the lower edges."""
    xs, ys = [c[x_key] for c in rows], [c[y_key] for c in rows]
    low = log_floor(xs + ys)
    xs, ys = clip_low(xs, low), clip_low(ys, low)
    lengths = [len(c["bodies"]) for c in rows]
    for length in sorted(set(lengths)):
        picked = [i for i, n in enumerate(lengths) if n == length]
        ax.scatter([xs[i] for i in picked], [ys[i] for i in picked], s=16, alpha=0.7,
                   label=f"{length} atom{'s' if length > 1 else ''}", clip_on=False)
    ax.plot([low, 1], [low, 1], "k--", linewidth=0.8)
    ax.set(xscale="log", yscale="log", xlim=(low, 1), ylim=(low, 1))
    ax.legend(title="body length", fontsize=7, title_fontsize=7)


def head_dot_plot(ax, labels, series, reference=None):
    """Per-row dots on a log x axis: series [(label, values, color, marker)], and
    a gray line from reference (e.g. the base rate) to the first series, the lift."""
    low = log_floor([v for _label, values, _c, _m in series for v in values] + list(reference or []))
    y = list(range(len(labels)))
    if reference is not None:
        ax.hlines(y, clip_low(reference, low), clip_low(series[0][1], low), color="lightgray", linewidth=2, zorder=1)
    for label, values, color, marker in series:
        ax.scatter(clip_low(values, low), y, color=color, marker=marker, s=22, label=label, zorder=3, clip_on=False)
    ax.set_yticks(y, labels, fontsize=7)
    ax.invert_yaxis()
    ax.set(xscale="log", xlim=(low, 1))
    ax.grid(axis="x", which="major", alpha=0.3)
    ax.legend(loc="lower right", fontsize=7)


def _plot_results(clauses, cfg, num_heads=30):
    """Save the step 05 figures, one PDF each, to cfg["figure_dir"]:
      training_curves.pdf: train / val onset loss per epoch;
      extension_gain.pdf: val confidence of the anchor alone vs the extended clause;
      train_vs_val_confidence.pdf: overfitting check of every clause;
      precision_recall_val.pdf: val confidence vs recall, size = train support;
      best_clause_per_head.pdf: best val confidence per head vs its base rate;
      delay_map.pdf: clauses per (head, slot of the anchor before the head);
      partial_evidence.pdf: clause confidence (val, over the base rate) by how many of its bodies are known.
    A figure is skipped (with a message) when its data is missing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    figure_dir = cfg["figure_dir"]
    os.makedirs(figure_dir, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figure_dir / name)
        plt.close(fig)
        print(f"[Figure]\t {figure_dir / name}")

    def has(clause, *keys):
        return all(clause.get(k) is not None for k in keys)

    # Training curves.
    if os.path.exists(cfg["history_file"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, file, color in (("rule net", cfg["history_file"], "tab:blue"),
                                  ("MLP baseline", cfg["mlp_history_file"], "tab:green")):
            if not os.path.exists(file):
                continue
            history = utils_data.load_json(file)
            ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], color=color,
                    linestyle="--", label=f"{name} train")
            ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], color=color,
                    label=f"{name} val")
        ax.set(title="Onset loss per epoch", xlabel="epoch", ylabel="weighted BCE")
        ax.legend()
        save(fig, "training_curves.pdf")
    else:
        print(f"[Figure]\t skipped training_curves.pdf: no {cfg['history_file']}")

    if not clauses:
        print("[Figure]\t skipped the clause figures: no clauses")
        return

    # Extension gain: the extended clauses only, above the diagonal = the extra atoms help.
    rows = [c for c in clauses if len(c["bodies"]) > 1 and has(c, "val_confidence", "val_anchor_confidence")]
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        clause_scatter(ax, rows, "val_anchor_confidence", "val_confidence")
        better = sum(c["val_confidence"] > c["val_anchor_confidence"] for c in rows)
        ax.set(title=f"Extension gain on val ({better}/{len(rows)} above y = x)",
               xlabel="confidence of the anchor alone", ylabel="confidence of the extended clause")
        save(fig, "extension_gain.pdf")
    else:
        print("[Figure]\t skipped extension_gain.pdf: no extended clause fires on val")

    # Train vs val confidence: far below the diagonal = overfitted.
    rows = [c for c in clauses if has(c, "val_confidence")]
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        clause_scatter(ax, rows, "confidence", "val_confidence")
        ax.set(title="Clause confidence, train vs val", xlabel="train confidence", ylabel="val confidence")
        save(fig, "train_vs_val_confidence.pdf")

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sizes = [4 + 60 * c["support"] / max(r["support"] for r in rows) for c in rows]
        low = log_floor([c["val_confidence"] for c in rows])
        ax.scatter([c["val_recall"] or 0 for c in rows], clip_low([c["val_confidence"] for c in rows], low),
                   s=sizes, alpha=0.5, clip_on=False)
        ax.set(title="Val confidence vs recall (size = train support)",
               xlabel="recall (share of head onsets hit)", ylabel="confidence (precision, log)",
               xlim=(0, 1), ylim=(low, 1), yscale="log")
        save(fig, "precision_recall_val.pdf")
    else:
        print("[Figure]\t skipped train_vs_val_confidence.pdf / precision_recall_val.pdf: no clause fires on val")

    # Best clause per head, against the head's base rate.
    best = {}
    for c in rows:
        h = c["head"][0]
        if h not in best or c["val_confidence"] > best[h]["val_confidence"]:
            best[h] = c
    if best:
        heads = sorted(best, key=lambda h: best[h]["val_confidence"], reverse=True)[:num_heads]
        fig, ax = plt.subplots(figsize=(7, 0.25 * len(heads) + 1.5))
        head_dot_plot(ax, [_atom_name(unique_atoms[h]) for h in heads],
                      [("best clause", [best[h]["val_confidence"] for h in heads], "tab:blue", "o"),
                       ("base rate", [best[h]["val_base_rate"] for h in heads], "tab:red", "|")],
                      reference=[best[h]["val_base_rate"] for h in heads])
        ax.set(title=f"Best val confidence per head (top {len(heads)}), gray = lift over base rate",
               xlabel="val confidence (log)")
        save(fig, "best_clause_per_head.pdf")

    # Delay map: how many slots before the head the anchor of the clauses of each head lies.
    size = cfg["slot_size"]
    bins = [(size * (j - 1), size * j) for j in range(1, cfg["num_slots"] + 1)]
    rows = [c for c in clauses if c.get("anchor")]
    if rows:
        heads = sorted({c["head"][0] for c in rows},
                       key=lambda h: -sum(c["head"][0] == h for c in rows))[:num_heads]
        grid = torch.zeros((len(heads), len(bins)))
        for c in rows:
            if c["head"][0] in heads:
                grid[heads.index(c["head"][0]), c["anchor"][2] - 1] += 1
        fig, ax = plt.subplots(figsize=(7, 0.25 * len(heads) + 1.5))
        image = ax.imshow(grid.numpy(), aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(bins)), [f"{lo}-{hi}" for lo, hi in bins], fontsize=7)
        ax.set_yticks(range(len(heads)), [_atom_name(unique_atoms[h]) for h in heads], fontsize=7)
        ax.set(title="Clauses per head and anchor slot", xlabel="anchor held this many frames before the head's slot")
        fig.colorbar(image, ax=ax, label="clauses")
        save(fig, "delay_map.pdf")

    # Partial evidence: confidence of the clauses with 2+ bodies by how many of their bodies are known.
    rows = [c for c in clauses if len(c["bodies"]) > 1 and c.get("partial")]
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        lift = {}
        for c in rows:
            points = [(p["known_bodies"], p["val_confidence"] / p["base_rate"]) for p in c["partial"]
                      if p["val_confidence"] is not None and p["base_rate"]]
            if len(points) > 1:
                ax.plot(*zip(*sorted(points)), color="tab:blue", alpha=0.15, linewidth=0.8)
            for known, value in points:
                lift.setdefault(known, []).append(value)
        known = sorted(lift)
        ax.plot(known, [sorted(lift[k])[len(lift[k]) // 2] for k in known], "o-", color="black", label="median")
        ax.axhline(1, color="tab:red", linestyle="--", label="base rate")
        ax.set(title="Clause confidence by bodies known (val)", xlabel="bodies of the clause known",
               ylabel="confidence / base rate (log)", yscale="log")
        ax.legend()
        save(fig, "partial_evidence.pdf")


def _print_summary(clauses, training_data, cfg, num_examples=15, title="Step 05 Result", clause_file=None):
    """Terminal summary of the step 05 result, printed on every run (cached or not)."""
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    print(f"----------------- {title} -----------------")
    print(f"[Videos]\t {len(training_data['train_videos'])} train, {len(training_data['val_videos'])} val")
    print(f"[Clauses]\t {len(clauses)} in {clause_file or cfg['extended_clause_file']}")
    if not clauses:
        print("[Clauses]\t none passed min_support / min_videos / weight_threshold")
        return
    lengths = {}
    for clause in clauses:
        lengths[len(clause["bodies"])] = lengths.get(len(clause["bodies"]), 0) + 1
    confidences = sorted(clause["confidence"] for clause in clauses)
    print(f"[Heads]\t\t {len({clause['head'][0] for clause in clauses})} distinct head atoms")
    print(f"[Body length]\t " + ", ".join(f"{n} atoms: {count}" for n, count in sorted(lengths.items())))
    print(f"[Confidence]\t median {confidences[len(confidences) // 2]:.3f}, max {confidences[-1]:.3f} (train)")
    val_confidences = sorted(c["val_confidence"] for c in clauses if c.get("val_confidence") is not None)
    if val_confidences:
        print(f"[Confidence]\t median {val_confidences[len(val_confidences) // 2]:.3f}, "
              f"max {val_confidences[-1]:.3f} (val, {len(val_confidences)} clauses firing on val)")
    gains = sorted(c["val_confidence"] - c["val_anchor_confidence"] for c in clauses
                   if len(c["bodies"]) > 1 and c.get("val_confidence") is not None
                   and c.get("val_anchor_confidence") is not None)
    if gains:
        print(f"[Extension]\t {sum(g > 0 for g in gains)}/{len(gains)} extended clauses beat their anchor on val, "
              f"median gain {gains[len(gains) // 2]:+.3f}")
    print(f"[Figures]\t {cfg['figure_dir']}")
    print(f"[Top {min(num_examples, len(clauses))}]")
    for clause in clauses[:num_examples]:
        print(f"  conf {clause['confidence']:.2f}  support {clause['support']:>5}  "
              f"{_temporal_clause_text(clause, unique_atoms)}")


def get_cfg(input_data):
    """Settings and files of step 05; step 06 uses them to load the trained model."""
    slot_size, num_slots = 10, 5
    windows = [slot_size * (f + 1) for f in range(10)]
    mask_type, num_free_rules = "sigmoid", 128
    # Files are named by the settings that change the model, so a model trained
    # for other settings is never reused.
    # Noise heads (fewer than noise_below onsets in the train + val videos) are no training
    # targets; step 06 reports their events separately.
    exclude_noise_heads, noise_below = True, 5
    # Type balance (`head_pos_weight_scale`), main rule net only: the onsets of the event types
    # with fewer onsets per head weigh (M_max / M_type) ** power, at most type_balance_max.
    type_balance = {"type_balance": True, "type_balance_power": 0.5, "type_balance_max": 4.0}
    base_tag = f"slots{slot_size}x{num_slots}_w{len(windows)}_{mask_type}_f{num_free_rules}_partial"
    if exclude_noise_heads:
        base_tag += f"_tn{noise_below}"
    # The main rule net (and what is derived from it: its clauses, cascade, pruning, rule set)
    # carries the type balance; the unweighted models (the MLP) keep base_tag.
    tag = base_tag + type_balance_tag(type_balance)
    # OnsetMLP baseline: lr 0.01 (the rule net's) with ReLU let every hidden unit die in the
    # first epochs (a constant score per head); a smaller lr, GELU, clipping and early stopping.
    mlp_lr, mlp_activation = 1e-3, "gelu"
    mlp_tag = f"{base_tag}_lr{mlp_lr:g}_{mlp_activation}"
    return {
            'device': input_data['device'],
            'seed': 0,
            # An atom switches on only after being off for this many frames for the
            # agent; shorter gaps are label flicker (re-onsets), not new events.
            'min_off_frames': 10,
            # Event types by head frequency (onsets in the train + val videos, `head_event_types`):
            # fewer than noise_below onsets are noise, more than trivial_above trivial, the rest
            # regular. With exclude_noise_heads the noise heads are no training targets.
            'exclude_noise_heads': exclude_noise_heads,
            'noise_below': noise_below,
            # Heads with noise_below..rare_below - 1 onsets are learned but reported as "rare".
            'rare_below': 10,
            'trivial_above': 100,
            **type_balance,
            # Train only the main rule net (training time): the MLP and the motif net are not
            # trained, their cached (unweighted) models are evaluated if they exist; the models
            # derived from the main rule net (cascade, causal pruning, rule set) are skipped.
            'main_only': True,
            # Targets: the atom switches on within the next W frames, for each window W
            # (10, 20, ..., 100 frames = 1..10 s at 10 fps); the model gives one
            # probability per window, never decreasing with W.
            'windows': windows,
            # Model: time in slots of slot_size frames; the memory holds num_slots past
            # slots (5 x 10 = 50 frames), a rule is a pattern over the num_slots slots
            # before its head, and each window is one more future slot. The videos are
            # ~200 frames, so 50 frames of memory leave the largest window (100 frames)
            # observable at t = 50..~100.
            'slot_size': slot_size,
            'num_slots': num_slots,
            # Partial facts: a share truncate_prob of the training frames know only a
            # random 0..50 frames of history (the older memory is unknown), so the
            # model learns calibrated, lower confidence from partial facts.
            'truncate_prob': 0.5,
            # Rules: one per (anchor body, lag bin) with the anchor as a fixed input
            # and seed_head_weight to its heads, plus num_free_rules unseeded ones,
            # so heads without an anchor (about a quarter of the onsets) can be recalled.
            'num_free_rules': num_free_rules,
            'seed_head_weight': 1.0,
            # Mask of the rule bodies: "sigmoid" (deterministic, init_log_alpha -11 so
            # every input gets a gradient; its gates learn at gate_lr) or
            # "hard_concrete" (the stochastic L0 mask; use init_log_alpha -6).
            'mask_type': mask_type,
            'init_log_alpha': -11.0,
            'gate_lr': 0.05,
            'gate_eps': 1e-20,
            # Training: batches of target frames sampled from a few videos at a time.
            'num_epochs': input_data['num_epochs'],
            'lr': 0.01,
            'videos_per_batch': 4,
            'frames_per_video': 32,
            # Share of sampled frames where some atom switches on (onsets are rare).
            'positive_frame_ratio': 0.5,
            'pos_weight': 20.0,
            'l0_lambda': 1e-3,
            'l0_warmup_epochs': 10,
            # Rule extraction thresholds of the trained model.
            'mask_threshold': 0.5,
            'weight_threshold': 0.1,
            # A clause must hit its head in min_support training frames of min_videos videos.
            'min_support': 10,
            'min_videos': 2,
            # Cascade. Stage 1 (the model above) is the recall model: per head, the
            # threshold keeping cascade_recall of the val onset rows flags the
            # candidates (heads with < cascade_min_positives val onset rows use the
            # threshold of all heads). Stage 2, the verifier, is a rule net trained
            # only on those candidates (pos_weight verifier_pos_weight) to reject
            # the false ones; step 06 scores the cascade.
            'cascade': True,
            'cascade_recall': 0.9,
            'cascade_min_positives': 20,
            'verifier_pos_weight': 5.0,
            # Black-box baseline: an OnsetMLP on the same inputs and targets, trained after the
            # rule net; hidden size None = the number of rule nodes (about as many parameters).
            'mlp_baseline': True,
            'mlp_hidden_size': None,
            'mlp_dropout': 0.2,
            'mlp_weight_decay': 1e-4,
            'mlp_lr': mlp_lr,
            'mlp_activation': mlp_activation,
            'mlp_grad_clip': 1.0,
            # Halve the lr after mlp_lr_patience epochs without a better val loss; stop after
            # mlp_patience (num_epochs is the maximum); the best val epoch is kept.
            'mlp_lr_patience': 5,
            'mlp_patience': 20,
            'wandb': True,
            'wandb_project': 'cauvid-roadpp',
            'train_atom_file': input_data['train_atom_file'],
            'val_atom_file': input_data['val_atom_file'],
            'causal_anchor_clause_file': input_data['causal_anchor_clause_file'],
            "unique_atoms_file": input_data["unique_atoms_file"],
            'training_data_file': input_data["output_dir"] / "training_data.pt",
            'model_file': input_data["output_dir"] / f"temporal_rule_net_{tag}.pt",
            "extended_clause_file": input_data["output_dir"] / f"extended_clauses_{tag}.json",
            'history_file': input_data["output_dir"] / f"training_history_{tag}.json",
            'mlp_model_file': input_data["output_dir"] / f"onset_mlp_{mlp_tag}.pt",
            'mlp_history_file': input_data["output_dir"] / f"onset_mlp_history_{mlp_tag}.json",
            'candidate_threshold_file': input_data["output_dir"] / f"cascade_thresholds_{tag}.pt",
            'verifier_model_file': input_data["output_dir"] / f"verifier_rule_net_{tag}.pt",
            'verifier_clause_file': input_data["output_dir"] / f"verifier_clauses_{tag}.json",
            'verifier_history_file': input_data["output_dir"] / f"verifier_history_{tag}.json",
            'figure_dir': input_data["output_dir"] / "figures",
    }


def main(input_data):
    print(f"----------------- Step 05: Temporal Rule Learning -----------------")
    cfg = get_cfg(input_data)

    # 1. Prepare once: compressed atoms of each video of the dataset's train and
    #    val splits and the frames to sample (all / with onsets); redone when
    #    min_off_frames or the split changes. The model and clauses are redone after it.
    training_data = load_training_data(cfg)
    model_fresh = is_fresh(cfg["model_file"], cfg["training_data_file"])
    clauses_fresh = model_fresh and is_fresh(cfg["extended_clause_file"], cfg["model_file"])

    # One W&B run for the training and extraction below, only if any is not cached.
    run = None
    if not clauses_fresh:
        run = _init_wandb(cfg, {"num_train_videos": len(training_data["train_videos"]),
                                "num_val_videos": len(training_data["val_videos"]),
                                "num_atoms": training_data["num_atoms"]},
                          job_type="step05_temporal_rule_net")
    try:
        # 2. Train the TemporalRuleNet; its input rows are built from the atoms per batch.
        if not model_fresh:
            model = train_temporal_rule_net(training_data, cfg, run)
            torch.save(model.state_dict(), cfg["model_file"])
        model = load_temporal_rule_net(training_data, cfg)

        # 3. Read the learned rules (ordered bodies with lags -> heads) out of the model.
        if not clauses_fresh:
            extended_clauses = extract_temporal_clauses(model, training_data, cfg, run)
            utils_data.save_json(extended_clauses, cfg["extended_clause_file"])
    finally:
        if run is not None:
            run.finish()

    # 4. The black-box baseline on the same data, its own W&B run; step 06 evaluates it.
    if cfg["mlp_baseline"] and not cfg["main_only"] and not is_fresh(cfg["mlp_model_file"], cfg["training_data_file"]):
        run = _init_wandb(cfg, {"num_atoms": training_data["num_atoms"]}, job_type="step05_onset_mlp")
        try:
            mlp = train_onset_mlp(training_data, cfg, run)
            torch.save(mlp.state_dict(), cfg["mlp_model_file"])
        finally:
            if run is not None:
                run.finish()

    # 5. Cascade: stage-1 candidate thresholds on val, then the verifier on the candidates, its own W&B run.
    if cfg["cascade"] and not cfg["main_only"]:
        run_cascade(model, training_data, cfg)

    # 6. Figures (one PDF each) and the terminal summary, on every run.
    extended_clauses = utils_data.load_json(cfg["extended_clause_file"])
    _plot_results(extended_clauses, cfg)
    _print_summary(extended_clauses, training_data, cfg)
    if cfg["cascade"] and not cfg["main_only"]:
        print_cascade_summary(training_data, cfg)


def run_cascade(model, training_data, cfg):
    """Stage-1 thresholds (`candidate_thresholds`), the verifier trained on the
    candidates (`train_verifier`) and its clauses, each redone when its inputs change."""
    if not is_fresh(cfg["candidate_threshold_file"], cfg["model_file"], cfg["training_data_file"]):
        torch.save(candidate_thresholds(model, training_data, cfg), cfg["candidate_threshold_file"])
    candidates = make_candidates(model, torch.load(cfg["candidate_threshold_file"])["thresholds"])

    verifier_fresh = is_fresh(cfg["verifier_model_file"], cfg["candidate_threshold_file"])
    clauses_fresh = verifier_fresh and is_fresh(cfg["verifier_clause_file"], cfg["verifier_model_file"])
    run = None
    if not clauses_fresh:
        run = _init_wandb(cfg, {"num_atoms": training_data["num_atoms"]}, job_type="step05_verifier")
    try:
        if not verifier_fresh:
            verifier = train_verifier(training_data, cfg, candidates, run)
            torch.save(verifier.state_dict(), cfg["verifier_model_file"])
        verifier = load_temporal_rule_net(training_data, cfg, cfg["verifier_model_file"])
        if not clauses_fresh:
            utils_data.save_json(extract_temporal_clauses(verifier, training_data, cfg, run, candidates),
                                 cfg["verifier_clause_file"])
    finally:
        if run is not None:
            run.finish()


def print_cascade_summary(training_data, cfg):
    stage1 = torch.load(cfg["candidate_threshold_file"])
    thresholds = stage1["thresholds"]
    print(f"----------------- Step 05 Cascade -----------------")
    print(f"[Stage 1]\t target recall {stage1['recall']:.2f}: val onset rows kept {stage1['val_row_recall']:.3f}, "
          f"candidates {stage1['val_candidate_share']:.4f} of the at-risk cells")
    print(f"[Stage 1]\t thresholds median {thresholds.median():.3g}, min {thresholds.min():.3g}, "
          f"max {thresholds.max():.3g}")
    _print_summary(utils_data.load_json(cfg["verifier_clause_file"]), training_data, cfg,
                   title="Step 05 Verifier (stage 2, counts over the candidates)",
                   clause_file=cfg["verifier_clause_file"])
