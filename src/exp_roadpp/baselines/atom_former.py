"""AtomFormer: Transformer baselines of the rule model on the same symbolic inputs (2026-11-26).

Tokens: one per (agent, memory slot) of a target frame, built from the OnsetMLP's packed inputs
(`TemporalRuleNet.layer_inputs`): the slot's atoms of the agent itself, of some other agent and of each role
partner (e.g. "ahead"), plus a flag "the slot is known". So every token holds exactly what the MLP reads for
that slot; the models differ only in the structure on top. The scene of a target frame is every agent visible in
the frame before it (the agents a row may be predicted for).

attention:
  "time"   AVT-style (Girdhar & Grauman 2021): each agent attends over its own slots only; no agent interaction
           beyond the MLP's "other" / role scopes.
  "factor" ViViT / VPTR-style factorised attention: the blocks alternate between the agent's own slots (time)
           and the other agents' tokens of the same slot (scene).
  "joint"  AgentFormer-style (Yuan et al. 2021): every token attends to every (agent, slot) token of the scene,
           with agent-aware attention: a learned bias per head for "same agent", "other agent" and "the query
           agent's role-k partner" (e.g. the vehicle ahead).
Ablations without attention (2026-11-27: where does the Transformers' lead over the OnsetMLP come from?), both
two hidden layers (4 d, d) with LayerNorm on the agent's own tokens:
  "slotmlp" every slot through the same embedding (shared across slots, + a slot embedding), then concatenated;
  "deepmlp" the slots' raw tokens concatenated (no sharing), as the OnsetMLP but deeper.
  "mlp1"    the OnsetMLP's structure (raw tokens concatenated, ONE hidden layer of 4 d units, LayerNorm + GELU)
            with the AtomFormer's training (lr, weight decay, dropout, epochs): deepmlp vs mlp1 is the second layer,
            mlp1 vs the OnsetMLP the training recipe.
Read-out: the mean of the agent's slot tokens -> hazard logits (W, A), the same windows, targets and loss as the
OnsetMLP and the TemporalRuleNet (`window_probs`, `loss`).

Training and scoring go through the step 05 rows (`step05._video_rows`): `training_rows` adds the scene of the
target frames ("_scene", "_present", "_partner": per video, not per row) and per row its frame ("fidx") and
its place in the scene ("slot"); `forward_rows(rows)` gives the rows' hazards (B, W, A).
"""
import os

import torch
import torch.nn.functional as F

from src.exp_roadpp.cauvid import TemporalRuleNet, memory_bins

ATTENTIONS = ("time", "factor", "joint", "slotmlp", "deepmlp", "mlp1")
MLP_MODES = ("slotmlp", "deepmlp", "mlp1")


class Block(torch.nn.Module):
    """Pre-norm Transformer block; the attention takes an additive bias (B, H, L, L) (-inf: masked)."""

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.out = torch.nn.Linear(d_model, d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(d_model, 4 * d_model), torch.nn.GELU(),
                                       torch.nn.Dropout(dropout), torch.nn.Linear(4 * d_model, d_model))
        self.dropout = torch.nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x, bias):
        B, L, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.p if self.training else 0.0)
        x = x + self.dropout(self.out(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.dropout(self.ffn(self.norm2(x)))


class AtomFormer(torch.nn.Module):
    scene_model = True

    # The input layers, targets and loss are the TemporalRuleNet's (as for the OnsetMLP).
    layer_inputs = TemporalRuleNet.layer_inputs
    first_onsets = staticmethod(TemporalRuleNet.first_onsets)
    window_log_probs = staticmethod(TemporalRuleNet.window_log_probs)
    window_probs = staticmethod(TemporalRuleNet.window_probs)
    loss = classmethod(TemporalRuleNet.loss.__func__)

    def __init__(self, num_atoms, slot_size=10, num_slots=3, windows=(10,), roles=(), open_memory=False,
                 attention="joint", d_model=128, num_layers=2, num_heads=4, dropout=0.1, frame_chunk=64):
        super().__init__()
        if attention not in ATTENTIONS:
            raise ValueError(f"attention {attention} not in {ATTENTIONS}")
        self.config = {"attention": attention, "d_model": d_model, "num_layers": num_layers,
                       "num_heads": num_heads, "dropout": dropout}
        self.attention = attention
        self.num_atoms = num_atoms
        self.slot_size, self.num_slots = slot_size, num_slots
        self.num_lags = slot_size * num_slots
        self.open_memory = bool(open_memory)
        self.roles = tuple(roles)
        self.num_scopes = 2 + len(self.roles)
        self.num_memory_slots = len(memory_bins(num_slots, self.open_memory))
        self.num_inputs = self.num_scopes * self.num_memory_slots * num_atoms
        self.num_memory_inputs = self.num_inputs
        self.windows = list(windows)
        self.frame_chunk = frame_chunk
        d_in = self.num_scopes * num_atoms + 1
        J = self.num_memory_slots
        self.embed = torch.nn.Linear(d_in, d_model)
        if attention in MLP_MODES:
            width = J * (d_model if attention == "slotmlp" else d_in)
            if attention == "mlp1":
                self.mlp = torch.nn.Sequential(torch.nn.Linear(width, 4 * d_model), torch.nn.LayerNorm(4 * d_model),
                                               torch.nn.GELU(), torch.nn.Dropout(dropout))
            else:
                self.mlp = torch.nn.Sequential(torch.nn.Linear(width, 4 * d_model), torch.nn.LayerNorm(4 * d_model),
                                               torch.nn.GELU(), torch.nn.Dropout(dropout),
                                               torch.nn.Linear(4 * d_model, d_model), torch.nn.GELU(),
                                               torch.nn.Dropout(dropout))
            num_layers = 0
        self.slot_embed = torch.nn.Parameter(0.02 * torch.randn(self.num_memory_slots, d_model))
        self.blocks = torch.nn.ModuleList(Block(d_model, num_heads, dropout) for _ in range(num_layers))
        # Agent-aware attention bias per block and head: 0 same agent, 1 other agent, 2 + k role-k partner.
        self.rel_bias = torch.nn.Parameter(torch.zeros(num_layers, num_heads, 2 + len(self.roles)))
        self.norm = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(4 * d_model if attention == "mlp1" else d_model, len(self.windows) * num_atoms)
        torch.nn.init.constant_(self.head.bias, -6.0)          # onsets are rare (as the MLP / rule net)

    # ---------------- rows ----------------
    def training_rows(self, X, frames, min_off_frames=1, history=None, roles=None):
        """`TemporalRuleNet.training_rows` plus the scene of the target frames (see the module doc)."""
        if self.roles:
            rows = TemporalRuleNet.training_rows(self, X, frames, min_off_frames, history, roles=roles)
        else:
            rows = TemporalRuleNet.training_rows(self, X, frames, min_off_frames, history)
        Fn, N = len(frames), X.shape[1]
        last = frames - 1
        present = X[last.clamp(min=0)].any(dim=2) & (last >= 0)[:, None]                  # (F, N)
        if self.roles:
            full = self.layer_inputs(X, frames, history, roles=roles, keep=present)
        else:
            full = self.layer_inputs(X, frames, history, keep=present)                    # (P, 2M), row-major
        pos = torch.where(present, present.long().cumsum(dim=1) - 1, -1)                 # (F, N) place in scene
        Np = max(int(present.sum(dim=1).max()) if Fn else 0, 1)
        fi, ni = torch.nonzero(present, as_tuple=True)
        scene = torch.zeros((Fn, Np, full.shape[1]), dtype=torch.bool, device=X.device)
        scene[fi, pos[fi, ni]] = full.bool()
        mask = torch.zeros((Fn, Np), dtype=torch.bool, device=X.device)
        mask[fi, pos[fi, ni]] = True
        K = len(self.roles)
        partner = torch.full((Fn, Np, K), -1, dtype=torch.long, device=X.device)
        if K:
            p = roles.to(X.device)[last.clamp(min=0)].long()                               # (F, N, K) agent ids
            p_pos = torch.where(p >= 0, pos.gather(1, p.clamp(min=0).flatten(1)).view(Fn, N, K), -1)
            partner[fi, pos[fi, ni]] = p_pos[fi, ni]
        keep = X[frames.repeat_interleave(N), torch.arange(N, device=X.device).repeat(Fn)].any(dim=1)
        fidx = torch.arange(Fn, device=X.device).repeat_interleave(N)[keep]
        rows["fidx"] = fidx
        rows["slot"] = pos[fidx, rows["rows"][:, 1]]            # -1: not visible before t (`video_rows` drops it)
        rows["_scene"], rows["_present"], rows["_partner"] = scene, mask, partner
        return rows

    def forward_rows(self, rows):
        """(B, W, A) hazard logits of the rows of one video (`training_rows` / `step05._video_rows`)."""
        h = self.forward_scene(rows["_scene"], rows["_present"], rows["_partner"])
        return h[rows["fidx"], rows["slot"].clamp(min=0)]

    def forward(self, rows):
        return self.forward_rows(rows), None

    # ---------------- model ----------------
    def forward_scene(self, scene, present, partner):
        """(F, Np, W, A) hazard logits of every agent of the scenes, in chunks of frame_chunk frames."""
        return torch.cat([self._chunk(scene[s:s + self.frame_chunk], present[s:s + self.frame_chunk],
                                      partner[s:s + self.frame_chunk])
                          for s in range(0, len(scene), self.frame_chunk)])

    def _chunk(self, scene, present, partner):
        Fc, Np, _ = scene.shape
        S, J, A = self.num_scopes, self.num_memory_slots, self.num_atoms
        M = S * J * A
        # scene: bool, or float (gradients of the inputs, `former_probe`)
        x = scene[..., :M].float().view(Fc, Np, S, J, A).permute(0, 1, 3, 2, 4).reshape(Fc, Np, J, S * A)
        k = scene[..., M:].bool().view(Fc, Np, S, J, A).permute(0, 1, 3, 2, 4).reshape(Fc, Np, J, S * A)
        known = k.all(dim=-1, keepdim=True).float()             # the slot is known (absent atoms are known)
        tok = torch.cat([x, known], dim=-1)
        if self.attention == "mlp1":
            return self.head(self.mlp(tok.reshape(Fc, Np, -1))).view(Fc, Np, len(self.windows), A)
        if self.attention == "deepmlp":
            h = self.norm(self.mlp(tok.reshape(Fc, Np, -1)))
            return self.head(h).view(Fc, Np, len(self.windows), A)
        h = self.embed(tok) + self.slot_embed
        if self.attention == "slotmlp":
            h = self.norm(self.mlp(torch.nn.functional.gelu(h).reshape(Fc, Np, -1)))
            return self.head(h).view(Fc, Np, len(self.windows), A)
        L, dev = Np * J, scene.device
        h = h.reshape(Fc, L, -1)
        agent = torch.arange(Np, device=dev).repeat_interleave(J)
        slot = torch.arange(J, device=dev).repeat(Np)
        same_agent = agent[:, None] == agent[None, :]                                   # (L, L)
        same_slot = slot[:, None] == slot[None, :]
        allowed = present.repeat_interleave(J, dim=1)[:, None, :] | same_agent[None]    # (Fc, L, L)
        rel = torch.where(same_agent, 0, 1)[None].expand(Fc, L, L)
        if self.roles:
            pq = partner[:, agent, :]                                                   # (Fc, L, K)
            for kk in range(len(self.roles)):
                hit = (pq[:, :, kk, None] == agent[None, None, :]) & ~same_agent[None]
                rel = torch.where(hit, 2 + kk, rel)
        for i, block in enumerate(self.blocks):
            if self.attention == "time":
                ok = allowed & same_agent[None]
            elif self.attention == "factor":
                ok = allowed & (same_agent if i % 2 == 0 else same_slot)[None]
            else:
                ok = allowed
            bias = self.rel_bias[i][:, rel].permute(1, 0, 2, 3)                          # (Fc, H, L, L)
            bias = bias.masked_fill(~ok[:, None], float("-inf"))
            h = block(h, bias)
        h = self.norm(h).view(Fc, Np, J, -1).mean(dim=2)
        return self.head(h).view(Fc, Np, len(self.windows), A)


# ---------------- training / loading ----------------
def former_spec(cfg):
    """The model file's tag of the former settings in cfg (`former_*`)."""
    # A short debug run (--num_epochs) gets its own file, so a full run never loads it.
    return (f"{cfg['former_attention']}_d{cfg['former_dim']}_l{cfg['former_layers']}_h{cfg['former_heads']}"
            f"_do{cfg['former_dropout']:g}_lr{cfg['former_lr']:g}"
            + (f"_ep{cfg['num_epochs']}" if cfg.get("num_epochs", 200) != 200 else ""))


def former_files(cfg):
    """(model file, history file) next to the run's OnsetMLP files."""
    mlp = str(cfg["mlp_model_file"])
    d, base = os.path.dirname(mlp), os.path.basename(mlp).replace("onset_mlp_", "")
    return (os.path.join(d, f"atom_former_{former_spec(cfg)}_{base}"),
            os.path.join(d, f"atom_former_history_{former_spec(cfg)}_{base.replace('.pt', '.json')}"))


def build_former(training_data, cfg):
    return AtomFormer(training_data["num_atoms"], cfg["slot_size"], cfg["num_slots"], cfg["windows"],
                      roles=cfg["roles"], open_memory=cfg.get("slot_ranges", False),
                      attention=cfg["former_attention"], d_model=cfg["former_dim"], num_layers=cfg["former_layers"],
                      num_heads=cfg["former_heads"], dropout=cfg["former_dropout"]).to(cfg["device"])


def train_former(training_data, cfg):
    """Train an AtomFormer with `step05.fit_onset_model` (same videos, target frames, batches, loss, val-loss model
    selection and lr halving / early stopping as the OnsetMLP), AdamW lr former_lr, weight decay 0.01."""
    from src.exp_roadpp import step05
    model = build_former(training_data, cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"[Baseline] AtomFormer {former_spec(cfg)}: {n / 1e6:.2f} M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["former_lr"], weight_decay=0.01)
    fcfg = {**cfg, "grad_clip": 1.0, "lr_patience": cfg["mlp_lr_patience"], "patience": cfg["mlp_patience"]}
    return step05.fit_onset_model(model, training_data, fcfg, optimizer, former_files(cfg)[1])


def load_former(training_data, cfg):
    """The trained AtomFormer of the cfg's former settings, in eval mode, or None if it was not trained."""
    path = former_files(cfg)[0]
    if not os.path.exists(path):
        return None
    model = build_former(training_data, cfg)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(cfg["device"]).eval()


def ensure_former(training_data, cfg):
    model = load_former(training_data, cfg)
    if model is None:
        path = former_files(cfg)[0]
        print(f"[Baseline] no AtomFormer at {path}: training it on the same inputs")
        torch.save(train_former(training_data, cfg).state_dict(), path)
        model = load_former(training_data, cfg)
    return model
