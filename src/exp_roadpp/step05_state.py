"""Step 05 (state): the StateMLP, which forecasts the GT label state of every agent h frames ahead.

Unlike the onset models (an atom switching on within a window, scored only on the at-risk cells),
every visible agent is predicted in every frame, for every GT label atom: is the label on h frames
after now (h = 20 frames = 2 s by default)? "Now" is the last observed frame t - 1 of the memory
inputs (`TemporalRuleNet.layer_inputs` of target frame t); the target is the label state in frame
t - 1 + h. A target counts only if the agent is still visible then (else the outcome is unknown).

Model: the OnsetMLP's inputs (the same memory slots and scopes), plus the agent's label state now,
a GELU hidden layer, and one logit per horizon and label atom. A learned persistence term per
horizon and atom, persist * (2 * now - 1), starts every atom at "it stays as it is", so the hidden
layer only learns the changes. Loss: binary cross-entropy over the known targets of the learned
heads (`load_training_data` "target_heads" restricted to the label atoms: no noise heads), the
changed cells (target != now: onsets and offsets) weighted by change_weight.

Also stores the per-head Markov rates of the train videos, P(on in t - 1 + h | on / off in t - 1),
the baseline that knows only the current state (`step06_state`).
"""
import os
import torch
from tqdm import tqdm

from src.exp_roadpp import step05, utils_data
from src.exp_roadpp.cauvid import TemporalRuleNet


class StateMLP(torch.nn.Module):
    """Input (B, 2M) memory inputs of `TemporalRuleNet.layer_inputs` and (B, L) label state now;
    output (B, K, L) logits of each label atom being on h frames after now, for each horizon h."""

    layer_inputs = TemporalRuleNet.layer_inputs

    def __init__(self, num_atoms, num_labels, slot_size, num_slots, hidden_size, dropout, horizons, roles=()):
        super().__init__()
        self.num_atoms, self.num_labels = num_atoms, num_labels
        self.slot_size, self.num_slots = slot_size, num_slots
        self.num_lags = slot_size * num_slots
        self.roles = tuple(roles)
        self.num_scopes = 2 + len(self.roles)
        self.num_inputs = self.num_scopes * num_slots * num_atoms
        self.horizons = list(horizons)
        self.hidden = torch.nn.Linear(2 * self.num_inputs + num_labels, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_size, len(self.horizons) * num_labels)
        torch.nn.init.normal_(self.head.weight, std=1e-3)
        torch.nn.init.constant_(self.head.bias, 0.0)
        # Persistence: logit +persist if the atom is on now, -persist if off (p ~ 0.95 / 0.05 at the start).
        self.persist = torch.nn.Parameter(torch.full((len(self.horizons), num_labels), 3.0))

    def forward(self, inputs, now):
        """(B, K, L) logits of inputs (B, 2M) and the label state now (B, L) float."""
        hidden = self.dropout(torch.nn.functional.gelu(self.hidden(torch.cat([inputs, now], dim=1))))
        logits = self.head(hidden).view(len(inputs), len(self.horizons), self.num_labels)
        return logits + self.persist[None] * (2 * now[:, None, :] - 1)


def state_rows(model, X, frames, num_labels, roles=None):
    """Rows of every agent visible now (frame t - 1) for the target frames t of one (T, N, A) video:
    {"inputs" (B, 2M), "now" (B, L) float, "targets" (B, K, L) bool (label state in t - 1 + h),
    "known" (B, K) bool (t - 1 + h inside the video and the agent visible then), "rows" (B, 2) (t, n)}.
    frames may hold T (now = the last frame: predictions past the video end, never known)."""
    T, N, _ = X.shape
    device = X.device
    frames = frames.to(device)
    frames = frames[frames >= 1]
    visible = X.any(dim=2)                                                     # (T, N)
    now_f = frames - 1
    keep = visible[now_f]                                                      # (F, N)
    f_i, n_i = torch.nonzero(keep, as_tuple=True)                              # the row order of layer_inputs(keep=)
    if len(f_i) == 0:
        return None
    inputs = model.layer_inputs(X, frames, None, roles if model.roles else None, keep=keep)
    t = frames[f_i]
    now = X[t - 1, n_i, :num_labels].float()
    horizons = torch.tensor(model.horizons, device=device)
    target_f = (t - 1)[:, None] + horizons[None]                              # (B, K)
    known = target_f < T
    tf = target_f.clamp(max=T - 1)
    known &= visible[tf, n_i[:, None]]
    targets = X[tf, n_i[:, None], :num_labels]                                 # (B, K, L)
    return {"inputs": inputs, "now": now, "targets": targets, "known": known, "rows": torch.stack([t, n_i], dim=1)}


def _video_state_rows(model, data, video_id, frames, device):
    video = data["videos"][video_id]
    X = step05._video_from_cells(video, data["num_atoms"], device)
    return state_rows(model, X, frames, step05.label_atom_count(data), step05.video_roles(video, device))


def state_loss(logits, rows, head_mask, change_weight):
    """Weighted BCE over the known targets of the heads in head_mask (L,) bool; the changed cells
    (target != now) weigh change_weight."""
    targets = rows["targets"].float()
    changed = rows["targets"] != (rows["now"][:, None, :] > 0.5)
    weight = (1.0 + (change_weight - 1.0) * changed.float()) * rows["known"][..., None] * head_mask[None, None]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (bce * weight).sum() / weight.sum().clamp(min=1)


def get_cfg(step05_cfg):
    """Settings and files of the StateMLP, next to the step 05 models (named by the MLP's tag)."""
    horizons = (20,)                            # frames ahead (10 fps): 2 s
    hidden_size, lr, change_weight = 512, 1e-3, 5.0
    mlp_tag = os.path.basename(str(step05_cfg["mlp_model_file"]))[len("onset_mlp_"):-len(".pt")]
    tag = f"{mlp_tag}_h{'-'.join(str(h) for h in horizons)}_hid{hidden_size}_cw{change_weight:g}"
    out = os.path.dirname(str(step05_cfg["mlp_model_file"]))
    return {
        "horizons": horizons,
        "hidden_size": hidden_size,
        "dropout": 0.2,
        "lr": lr,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        # Changed cells (onsets and offsets) are rare; they weigh change_weight in the loss.
        "change_weight": change_weight,
        "num_epochs": step05_cfg["num_epochs"],
        "videos_per_batch": step05_cfg["videos_per_batch"],
        "frames_per_video": step05_cfg["frames_per_video"],
        # Share of sampled frames placed 0..max(horizons) - 1 frames before an onset (the rest uniform).
        "positive_frame_ratio": 0.5,
        # Fixed val frames per val video for the val loss.
        "val_frames_per_video": 2 * step05_cfg["frames_per_video"],
        "lr_patience": 5,
        "patience": 20,
        "seed": step05_cfg["seed"],
        "device": step05_cfg["device"],
        "model_file": os.path.join(out, f"state_mlp_{tag}.pt"),
        "history_file": os.path.join(out, f"state_mlp_history_{tag}.json"),
    }


def _new_model(data, step05_cfg, cfg):
    return StateMLP(data["num_atoms"], step05.label_atom_count(data), step05_cfg["slot_size"],
                    step05_cfg["num_slots"], cfg["hidden_size"], cfg["dropout"], cfg["horizons"],
                    roles=step05_cfg["roles"]).to(cfg["device"])


@torch.no_grad()
def markov_rates(data, video_ids, horizons, device):
    """(K, 2, L) P(label on in t - 1 + h | off / on in t - 1) per head over the given videos, the
    agent visible in both frames."""
    L = step05.label_atom_count(data)
    on = torch.zeros((len(horizons), 2, L), device=device)
    total = torch.zeros((len(horizons), 2, L), device=device)
    for video_id in tqdm(video_ids, desc="Markov rates of the label states (train)"):
        X = step05._video_from_cells(data["videos"][video_id], data["num_atoms"], device)
        visible = X.any(dim=2)
        S = X[..., :L]
        for k, h in enumerate(horizons):
            if X.shape[0] <= h:
                continue
            keep = (visible[:-h] & visible[h:])[..., None]
            for c, now in enumerate((~S[:-h], S[:-h])):
                cell = now & keep
                total[k, c] += cell.sum(dim=(0, 1))
                on[k, c] += (cell & S[h:]).sum(dim=(0, 1))
    return (on / total.clamp(min=1)).cpu()


def _val_loss(model, data, val_frames, head_mask, cfg):
    model.eval()
    losses = []
    with torch.no_grad():
        for video_id, frames in val_frames.items():
            rows = _video_state_rows(model, data, video_id, frames, cfg["device"])
            if rows is None or not rows["known"].any():
                continue
            losses.append(state_loss(model(rows["inputs"], rows["now"]), rows, head_mask, cfg["change_weight"]).item())
    return sum(losses) / len(losses) if losses else float("nan")


def train_state_mlp(data, step05_cfg, cfg, run=None):
    """Train the StateMLP on the train videos, the best val-loss epoch kept (early stopping)."""
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    device = cfg["device"]
    model = _new_model(data, step05_cfg, cfg)
    L = step05.label_atom_count(data)
    head_mask = data["target_heads"][:L].to(device)
    print(f"[State MLP] {cfg['hidden_size']} GELU hidden units, horizons {cfg['horizons']} frames, "
          f"{int(head_mask.sum())} of {L} label heads learned")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=cfg["lr_patience"])
    horizon = max(cfg["horizons"])
    val_frames = {v: step05._sample_frames(data, v, cfg["val_frames_per_video"], cfg["positive_frame_ratio"],
                                           generator, horizon) for v in data["val_videos"]}
    train_videos = data["train_videos"]
    best_val, best_state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(cfg["num_epochs"]):
        model.train()
        order = torch.randperm(len(train_videos), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), cfg["videos_per_batch"]):
            parts = []
            for i in order[start:start + cfg["videos_per_batch"]]:
                frames = step05._sample_frames(data, train_videos[i], cfg["frames_per_video"],
                                               cfg["positive_frame_ratio"], generator, horizon)
                rows = _video_state_rows(model, data, train_videos[i], frames, device)
                if rows is not None:
                    parts.append(rows)
            if not parts:
                continue
            rows = step05.concat_rows(parts)
            if not rows["known"].any():
                continue
            loss = state_loss(model(rows["inputs"], rows["now"]), rows, head_mask, cfg["change_weight"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            losses.append(loss.item())
        train_loss = sum(losses) / max(len(losses), 1)
        val_loss = _val_loss(model, data, val_frames, head_mask, cfg)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"[State MLP epoch {epoch + 1}/{cfg['num_epochs']}] train loss {train_loss:.4f}, "
              f"val loss {val_loss:.4f}, lr {lr:.2g}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        if run is not None:
            run.log({"train/state_loss": train_loss, "val/state_loss": val_loss, "lr": lr}, step=epoch)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch + 1 - best_epoch >= cfg["patience"]:
            print(f"[State MLP] Early stop: no better val loss for {cfg['patience']} epochs")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[State MLP] Best val loss {best_val:.4f} (epoch {best_epoch})")
    utils_data.save_json(history, cfg["history_file"])
    if run is not None:
        run.summary["best_val_state_loss"] = best_val
    return model


def load_state_mlp(data, step05_cfg, cfg=None):
    """(model in eval mode, checkpoint dict) of the trained StateMLP, or (None, None) if missing."""
    cfg = cfg or get_cfg(step05_cfg)
    if not os.path.exists(cfg["model_file"]):
        return None, None
    ckpt = torch.load(cfg["model_file"], map_location="cpu", weights_only=False)
    model = _new_model(data, step05_cfg, {**cfg, "horizons": ckpt["horizons"]})
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def main(input_data):
    print("----------------- Step 05 (state): StateMLP, the label state h frames ahead -----------------")
    step05_cfg = step05.get_cfg(input_data)
    cfg = get_cfg(step05_cfg)
    if step05.is_fresh(cfg["model_file"], step05_cfg["training_data_file"]):
        print(f"[State MLP]\t {cfg['model_file']}: up to date")
        return
    data = step05.load_training_data(step05_cfg)
    run = step05._init_wandb({**step05_cfg, **cfg}, {"num_atoms": data["num_atoms"]}, job_type="step05_state_mlp")
    try:
        model = train_state_mlp(data, step05_cfg, cfg, run)
    finally:
        if run is not None:
            run.finish()
    L = step05.label_atom_count(data)
    torch.save({"state_dict": model.state_dict(), "horizons": list(cfg["horizons"]),
                "num_label_atoms": L, "head_types": data["event_types"]["types"][:L].clone(),
                "event_types": data["event_types"]["heads"],
                "markov": markov_rates(data, data["train_videos"], cfg["horizons"], cfg["device"])},
               cfg["model_file"])
    print(f"[State MLP]\t saved {cfg['model_file']}")
