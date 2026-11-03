import os
import sys
import getpass
import torch
from tqdm import tqdm

from src.exp_roadpp import utils_data
from src.exp_roadpp.cauvid import TemporalRuleNet


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


def prepare_videos(atom_file, num_atoms, device):
    """Compressed atoms and frame lists of every video of one split.

    Returns {
      "num_atoms": A,
      "videos": {video_id: `_video_cells` of its (T, N, A) atom tensor},
      "frames": {video_id: (F,) long, frames with some visible agent},
      "positive_frames": {video_id: (P,) long, frames where some agent already
                          visible in the previous frame switches an atom on},
    }; videos shorter than 2 frames are left out.
    An agent entering the scene switches on all of its atoms at once; those
    are appearances, not events, so they do not make a frame positive.
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
        has_onset = (atom_onsets(X).any(dim=2) & stayed).any(dim=1)

        videos[video_id] = _video_cells(X)
        frames[video_id] = torch.nonzero(visible.any(dim=1)).flatten().cpu()
        positive_frames[video_id] = torch.nonzero(has_onset).flatten().cpu()
    return {"num_atoms": num_atoms, "videos": videos, "frames": frames, "positive_frames": positive_frames}


def prepare_training_data(cfg):
    """Everything the TemporalRuleNet training samples its batches from: the
    `prepare_videos` data of the training atoms, plus "train_videos" /
    "val_videos", a random split by video."""
    num_atoms = len(utils_data.load_json(cfg['unique_atoms_file']))
    data = prepare_videos(cfg['train_atom_file'], num_atoms, cfg["device"])

    video_ids = sorted(data["videos"])
    order = torch.randperm(len(video_ids), generator=torch.Generator().manual_seed(cfg["seed"]))
    num_val = max(1, round(cfg["val_video_ratio"] * len(video_ids)))
    val_videos = sorted(video_ids[i] for i in order[:num_val].tolist())
    train_videos = sorted(video_ids[i] for i in order[num_val:].tolist())

    num_frames = sum(len(f) for f in data["frames"].values())
    num_positive = sum(len(f) for f in data["positive_frames"].values())
    print(f"[Videos]\t {len(video_ids)} ({len(train_videos)} train, {len(val_videos)} val)")
    print(f"[Frames]\t {num_frames} ({num_positive} with onsets)")
    return {**data, "train_videos": train_videos, "val_videos": val_videos}


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
    # One rule node per (anchor body, lag bin), plus num_free_rules unseeded ones.
    body_groups = group_clauses_by_body(utils_data.load_json(cfg["causal_anchor_clause_file"]))
    anchors = [(body, [head[0] for head in heads]) for body, heads in body_groups.items()]
    num_bins = len(cfg["lag_edges"])
    model = TemporalRuleNet(training_data["num_atoms"], cfg["lag_edges"],
                            len(anchors) * num_bins + cfg["num_free_rules"])
    num_seeded = model.seed_rules(anchors, cfg["seed_head_weight"])
    model = model.to(device)
    print(f"[Model] {num_seeded} rules seeded from {len(anchors)} anchor bodies, {cfg['num_free_rules']} free")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Fixed validation frames, so the losses of the epochs are comparable.
    val_frames = {
        video_id: _sample_frames(training_data, video_id, cfg["frames_per_video"],
                                 cfg["positive_frame_ratio"], generator)
        for video_id in training_data["val_videos"]
    }
    train_videos = training_data["train_videos"]
    best_val_loss, best_state = float("inf"), None
    history = []
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
            # No sparsity pressure in the first epochs, so useful inputs can grow first.
            l0_lambda = cfg["l0_lambda"] if epoch >= cfg["l0_warmup_epochs"] else 0.0
            loss = onset_loss + l0_lambda * model.sparsity_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(onset_loss.item())

        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        val_loss = _validation_loss(model, training_data, val_frames, cfg)
        print(f"[Epoch {epoch + 1}/{cfg['num_epochs']}] train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
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
    utils_data.save_json(history, cfg["history_file"])
    if run is not None:
        run.summary["best_val_onset_loss"] = best_val_loss
    return model


def load_temporal_rule_net(training_data, cfg):
    """The trained TemporalRuleNet of cfg["model_file"], in eval mode on cfg["device"].

    The model is rebuilt with the same sizes it was trained with (num_atoms of
    the training data, lag_edges of cfg, the number of rules of the saved weights)
    before its weights, and its seeded anchor inputs, are loaded.
    """
    state = torch.load(cfg["model_file"], map_location="cpu")
    model = TemporalRuleNet(training_data["num_atoms"], cfg["lag_edges"], state["log_alpha"].shape[0])
    model.load_state_dict(state)
    return model.to(cfg["device"]).eval()


@torch.no_grad()
def _rule_head_counts(model, training_data, video_ids, cfg, desc="Counting rule firings"):
    """Counts of every rule r and head atom h over all frames of the given videos.

    Returns a dict of CPU tensors:
      fires (R, A): rows where every body atom of r holds and h is at risk (off
          in the previous frame); hits (R, A): the fires where h switches on;
      videos (R, A): the videos with at least one hit;
      anchor_fires / anchor_hits (R, A): the same for the anchor input of r alone
          (its fixed input, without the learned extra atoms; 0 for free rules);
      onsets (A,): all rows where h switches on; at_risk (A,): all rows where it could.
    """
    num_rules, num_atoms = model.log_alpha.shape[0], model.num_atoms
    device = cfg["device"]
    counts = {name: torch.zeros((num_rules, num_atoms), device=device)
              for name in ("fires", "hits", "videos", "anchor_fires", "anchor_hits")}
    counts["onsets"] = torch.zeros(num_atoms, device=device)
    counts["at_risk"] = torch.zeros(num_atoms, device=device)
    anchor_body = model.fixed_inputs.float()
    anchor_size = anchor_body.sum(dim=1)
    for video_id in tqdm(video_ids, desc=desc):
        video_hits = torch.zeros((num_rules, num_atoms), device=device)
        for frames in training_data["frames"][video_id].split(cfg["frames_per_video"]):
            inputs, targets, at_risk = _video_rows(model, training_data, video_id, frames, device)
            if len(inputs) == 0:
                continue
            targets, at_risk = targets.float(), at_risk.float()
            fired = (model.partial_activation(inputs, cfg["mask_threshold"]) >= 1).float()
            anchor_fired = ((inputs @ anchor_body.T >= anchor_size) & (anchor_size > 0)).float()
            counts["fires"] += fired.T @ at_risk
            video_hits += fired.T @ targets
            counts["anchor_fires"] += anchor_fired.T @ at_risk
            counts["anchor_hits"] += anchor_fired.T @ targets
            counts["onsets"] += targets.sum(dim=0)
            counts["at_risk"] += at_risk.sum(dim=0)
        counts["hits"] += video_hits
        counts["videos"] += video_hits > 0
    return {name: value.cpu() for name, value in counts.items()}


def _temporal_clause_text(clause, unique_atoms):
    # Bodies earliest first: "Car:MovAway(X0)@-5..8f -> Ped:Wait(X1)@-2..2f => Car:Stop(X0)".
    def atom(atom_index, slot):
        return f"{_atom_name(unique_atoms[atom_index])}(X{slot})"
    body = " -> ".join(f"{atom(a, slot)}@-{lo}..{hi}f" for a, slot, lo, hi in clause["bodies"])
    return f"{body} => {atom(*clause['head'])}"


def extract_temporal_clauses(model, training_data, cfg, run=None):
    """Clauses of the trained model, one per (rule node, head) pair with a large
    head weight, kept if the rule hits the head in at least min_support training
    frames of at least min_videos videos.

    Each clause: bodies [(atom, slot, lo, hi)] earliest first (the atom holds
    lo..hi frames before the head; slot 0 the head agent, 1 some other agent), body_scopes their
    slots, head (atom, 0), anchor (atom, slot, lo, hi) its seeded anchor input or
    None, weight, and on the training videos support (hits), fires, confidence =
    support / fires, num_videos, anchor_confidence (of the anchor input alone).
    The val_ fields are the same on the validation videos, plus val_recall (the
    share of the head's onsets the clause hits) and val_base_rate (the head's
    onsets / at-risk rows, the confidence of always predicting it). Sorted by confidence.
    """
    def ratio(a, b):
        return float(a) / float(b) if b > 0 else None

    rules = model.extract_rules(cfg["mask_threshold"], cfg["weight_threshold"])
    train = _rule_head_counts(model, training_data, training_data["train_videos"], cfg,
                              desc="Counting rule firings (train)")
    val = _rule_head_counts(model, training_data, training_data["val_videos"], cfg,
                            desc="Counting rule firings (val)")

    clauses = []
    for rule in rules:
        r = rule["rule"]
        anchor_inputs = torch.nonzero(model.fixed_inputs[r]).flatten().tolist()
        anchor = list(model.decode_input(anchor_inputs[0])) if anchor_inputs else None
        for h, weight in rule["heads"]:
            support, num_videos = int(train["hits"][r, h]), int(train["videos"][r, h])
            if support < cfg["min_support"] or num_videos < cfg["min_videos"]:
                continue
            clauses.append({
                "bodies": [list(b) for b in rule["bodies"]],
                "body_scopes": [slot for _atom, slot, _lo, _hi in rule["bodies"]],
                "head": [h, 0],
                "anchor": anchor,
                "rule": r,
                "weight": weight,
                "support": support,
                "fires": int(train["fires"][r, h]),
                "confidence": support / max(int(train["fires"][r, h]), 1),
                "num_videos": num_videos,
                "anchor_confidence": ratio(train["anchor_hits"][r, h], train["anchor_fires"][r, h]),
                "val_support": int(val["hits"][r, h]),
                "val_fires": int(val["fires"][r, h]),
                "val_confidence": ratio(val["hits"][r, h], val["fires"][r, h]),
                "val_anchor_confidence": ratio(val["anchor_hits"][r, h], val["anchor_fires"][r, h]),
                "val_recall": ratio(val["hits"][r, h], val["onsets"][h]),
                "val_base_rate": ratio(val["onsets"][h], val["at_risk"][h]),
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


def _plot_results(clauses, cfg, num_heads=30):
    """Save the step 05 figures, one PDF each, to cfg["figure_dir"]:
      training_curves.pdf: train / val onset loss per epoch;
      extension_gain.pdf: val confidence of the anchor alone vs the extended clause;
      train_vs_val_confidence.pdf: overfitting check of every clause;
      precision_recall_val.pdf: val confidence vs recall, size = train support;
      best_clause_per_head.pdf: best val confidence per head vs its base rate;
      delay_map.pdf: clauses per (head, lag bin of the anchor).
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

    def length_scatter(ax, rows, x_key, y_key):
        # One dot per clause, colored by body length; dashed y = x.
        lengths = [len(c["bodies"]) for c in rows]
        points = ax.scatter([c[x_key] for c in rows], [c[y_key] for c in rows],
                            c=lengths, cmap="viridis", s=14, alpha=0.7)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        fig.colorbar(points, ax=ax, label="body length (atoms)")

    # Training curves.
    if os.path.exists(cfg["history_file"]):
        history = utils_data.load_json(cfg["history_file"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train")
        ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="val")
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
        length_scatter(ax, rows, "val_anchor_confidence", "val_confidence")
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
        length_scatter(ax, rows, "confidence", "val_confidence")
        ax.set(title="Clause confidence, train vs val", xlabel="train confidence", ylabel="val confidence")
        save(fig, "train_vs_val_confidence.pdf")

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sizes = [4 + 60 * c["support"] / max(r["support"] for r in rows) for c in rows]
        ax.scatter([c["val_recall"] or 0 for c in rows], [c["val_confidence"] for c in rows],
                   s=sizes, alpha=0.5)
        ax.set(title="Val confidence vs recall (size = train support)",
               xlabel="recall (share of head onsets hit)", ylabel="confidence (precision)",
               xlim=(0, None), ylim=(0, 1))
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
        y = list(range(len(heads)))
        ax.barh(y, [best[h]["val_confidence"] for h in heads], color="tab:blue", label="best clause")
        ax.scatter([best[h]["val_base_rate"] or 0 for h in heads], y, color="tab:red", marker="|", s=120,
                   label="base rate", zorder=3)
        ax.set_yticks(y, [_atom_name(unique_atoms[h]) for h in heads], fontsize=7)
        ax.invert_yaxis()
        ax.set(title=f"Best val confidence per head (top {len(heads)})", xlabel="val confidence", xlim=(0, 1))
        ax.legend(loc="lower right")
        save(fig, "best_clause_per_head.pdf")

    # Delay map: in which lag bin the anchor of the clauses of each head lies.
    bins = [(lo + 1, hi) for lo, hi in zip((0, *cfg["lag_edges"][:-1]), cfg["lag_edges"])]
    rows = [c for c in clauses if c.get("anchor")]
    if rows:
        heads = sorted({c["head"][0] for c in rows},
                       key=lambda h: -sum(c["head"][0] == h for c in rows))[:num_heads]
        grid = torch.zeros((len(heads), len(bins)))
        for c in rows:
            if c["head"][0] in heads:
                grid[heads.index(c["head"][0]), bins.index((c["anchor"][2], c["anchor"][3]))] += 1
        fig, ax = plt.subplots(figsize=(7, 0.25 * len(heads) + 1.5))
        image = ax.imshow(grid.numpy(), aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(bins)), [f"{lo}-{hi}" if lo != hi else f"{lo}" for lo, hi in bins], fontsize=7)
        ax.set_yticks(range(len(heads)), [_atom_name(unique_atoms[h]) for h in heads], fontsize=7)
        ax.set(title="Clauses per head and anchor lag bin", xlabel="anchor held this many frames before the head")
        fig.colorbar(image, ax=ax, label="clauses")
        save(fig, "delay_map.pdf")


def _print_summary(clauses, training_data, cfg, num_examples=15):
    """Terminal summary of the step 05 result, printed on every run (cached or not)."""
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    print(f"----------------- Step 05 Result -----------------")
    print(f"[Videos]\t {len(training_data['train_videos'])} train, {len(training_data['val_videos'])} val")
    print(f"[Clauses]\t {len(clauses)} in {cfg['extended_clause_file']}")
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
    return {
            'device': input_data['device'],
            'seed': 0,
            # Model: lag bins (lo, hi] of the past frames, one input layer each, and
            # R rule nodes. Edges 1, 2, 4, ..., 100 -> bins 1, 2, 3-4, ..., 65-100 frames ago.
            'lag_edges': [1, 2, 4, 8, 16, 32, 64, 100],
            # Rules: one per (anchor body, lag bin) with the anchor as a fixed input
            # and seed_head_weight to its heads, plus num_free_rules unseeded ones.
            'num_free_rules': 0,
            'seed_head_weight': 1.0,
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
            "extended_clause_file": input_data["output_dir"] / "extended_clauses.json",
            'history_file': input_data["output_dir"] / "training_history.json",
            'figure_dir': input_data["output_dir"] / "figures",
    }


def main(input_data):
    print(f"----------------- Step 05: Temporal Rule Learning -----------------")
    cfg = get_cfg(input_data)

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

    # 4. Figures (one PDF each) and the terminal summary, on every run.
    extended_clauses = utils_data.load_json(cfg["extended_clause_file"])
    _plot_results(extended_clauses, cfg)
    _print_summary(extended_clauses, training_data, cfg)
