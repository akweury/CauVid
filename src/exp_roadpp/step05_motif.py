"""Step 05d: train the two-level MotifRuleNet (2-frame motifs, 10-frame rules, noisy-OR clauses).

Same data, split, window targets, partial-fact training (random history
truncation) and step 06 evaluation as the step 05 TemporalRuleNet; the model is
`motif_net.MotifRuleNet`: motifs over 2-frame blocks of the last 60 frames
(hold / onset / offset of every atom, self and other agents), rules over motifs
and holds in 10-frame slots, and one confidence per clause (rule, head) in a
noisy-OR. After reg_warmup_epochs the loss adds sparsity_lambda * the mask and
confidence L1 and precision_lambda * the precision loss: every clause with a
non-negligible confidence is pushed to a precision of at least p_min (onset in
the next 10 frames when every body holds), so single clauses are precise and
the rules together cover the onsets.

Outputs: motif_rule_net_<tag>.pt, motif_clauses_<tag>.json, motif_history_<tag>.json.
"""
import os

import torch
from tqdm import tqdm

from src.exp_roadpp import step05, utils_data
from src.exp_roadpp.motif_net import MotifRuleNet


def get_cfg(input_data):
    """Settings and files of step 05d; the data, split and anchors are step 05's."""
    base = step05.get_cfg(input_data)
    fine_size, motif_span, slot_size, num_slots = 2, 5, 10, 6
    num_free_motifs, num_free_rules, p_min = 1024, 1024, 0.05
    tag = f"b{fine_size}x{motif_span}_s{slot_size}x{num_slots}_m{num_free_motifs}_r{num_free_rules}_p{p_min}"
    if base["exclude_noise_heads"]:
        tag += f"_tn{base['noise_below']}"                     # the noise heads are no training targets
    if base["geometry"]:
        tag += "_g" + "-".join(base["geometry"])              # the geometry atoms (inputs) widen every input
    tag += step05.grid_tag(base)                              # so do the grid atoms
    output_dir = input_data["output_dir"]
    return {
        "enabled": True,
        "device": base["device"],
        "seed": base["seed"],
        # Level 1: motifs over motif_span blocks of fine_size frames (10 frames).
        "fine_size": fine_size,
        "motif_span": motif_span,
        # Level 2: rules over num_slots slots of slot_size frames (memory 60 frames).
        "slot_size": slot_size,
        "num_slots": num_slots,
        "windows": base["windows"],
        # One motif per anchor body (its onset) and one rule per (anchor body, slot), plus free ones.
        "num_free_motifs": num_free_motifs,
        "num_free_rules": num_free_rules,
        "seed_confidence": 0.1,
        "init_log_alpha": -11.0,
        # Training (sampling, truncation and pos_weight as step 05).
        "num_epochs": base["num_epochs"],
        "lr": 0.01,
        "gate_lr": 0.05,
        "gate_eps": 1e-20,
        "videos_per_batch": base["videos_per_batch"],
        "frames_per_video": base["frames_per_video"],
        "positive_frame_ratio": base["positive_frame_ratio"],
        "pos_weight": base["pos_weight"],
        "truncate_prob": base["truncate_prob"],
        "min_off_frames": base["min_off_frames"],
        # Regularization after reg_warmup_epochs.
        "reg_warmup_epochs": 10,
        "sparsity_lambda": 1e-4,
        # Every clause at least p_min precise (onset within the next slot), m-estimate with k.
        "precision_lambda": 1.0,
        "p_min": p_min,
        "precision_k": 10.0,
        "precision_decay": 0.9,
        # Clause extraction.
        "mask_threshold": 0.5,
        "conf_threshold": 0.02,
        "min_support": 10,
        "min_videos": 2,
        "wandb": base["wandb"],
        "wandb_project": base["wandb_project"],
        "unique_atoms_file": base["unique_atoms_file"],
        "geometry": base["geometry"],
        "grid": base["grid"],
        "grid_file": base["grid_file"],
        "training_data_file": base["training_data_file"],
        "model_file": output_dir / f"motif_rule_net_{tag}.pt",
        "clause_file": output_dir / f"motif_clauses_{tag}.json",
        "history_file": output_dir / f"motif_history_{tag}.json",
        "step05_cfg": base,
    }


def build_model(num_atoms, cfg, num_motifs=None, num_rules=None):
    anchors = step05.anchor_rules(cfg["step05_cfg"])
    model = MotifRuleNet(num_atoms, cfg["fine_size"], cfg["motif_span"], cfg["slot_size"], cfg["num_slots"],
                         num_motifs or len(anchors) + cfg["num_free_motifs"],
                         num_rules or len(anchors) * cfg["num_slots"] + cfg["num_free_rules"],
                         cfg["windows"], cfg["init_log_alpha"])
    return model, anchors


def load_motif_rule_net(data, cfg, model_file=None):
    """The trained MotifRuleNet in eval mode on cfg["device"], or None if it was not trained."""
    model_file = model_file or cfg["model_file"]
    if not os.path.exists(model_file):
        return None
    state = torch.load(model_file, map_location="cpu")
    model, _anchors = build_model(data["num_atoms"], cfg, state["motif_alpha"].shape[0], state["rule_alpha"].shape[0])
    model.load_state_dict(state)
    return model.to(cfg["device"]).eval()


def model_file_of(step05_cfg):
    """The step 05d model file next to the given step 05 settings (for step 06), and its cfg."""
    cfg = get_cfg({"device": step05_cfg["device"], "num_epochs": step05_cfg["num_epochs"],
                   "output_dir": step05_cfg["model_file"].parent,
                   "train_atom_file": step05_cfg["train_atom_file"], "val_atom_file": step05_cfg["val_atom_file"],
                   "causal_anchor_clause_file": step05_cfg["causal_anchor_clause_file"],
                   "unique_atoms_file": step05_cfg["unique_atoms_file"],
                   "train_box_file": step05_cfg.get("train_box_file"),
                   "val_box_file": step05_cfg.get("val_box_file"),
                   "grid": step05_cfg.get("grid", True), "motion": step05_cfg.get("motion", True),
                   "ego_motion": step05_cfg.get("ego_motion", False),
                   "require_self": step05_cfg.get("require_self", True),
                   "num_slots": step05_cfg.get("num_slots", 3), "roles": step05_cfg.get("roles", ()),
                   "geometry": bool(step05_cfg.get("geometry")), "type_balance": step05_cfg.get("type_balance", False),
                   "truncate_prob": step05_cfg.get("truncate_prob", 0.5)})
    return cfg["model_file"], cfg


# -------------- Training --------------
def train_motif_rule_net(training_data, cfg, run=None):
    """Train a MotifRuleNet seeded from the anchors; returns it with the weights of the
    lowest val loss (the window loss on fixed val frames with fixed truncated histories)."""
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    device = cfg["device"]
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model, anchors = build_model(training_data["num_atoms"], cfg)
    seeded = model.seed(anchors, cfg["seed_confidence"])
    model = model.to(device)
    print(f"[Model] MotifRuleNet: {model.num_motifs} motifs ({seeded[0]} seeded), {model.rule_alpha.shape[0]} rules "
          f"({seeded[1]} seeded), {model.fine_channels * model.num_blocks} fine + "
          f"{model.coarse_channels * model.num_slots} coarse inputs, memory {model.num_lags} frames")
    gates = [model.motif_alpha, model.rule_alpha]
    others = [p for name, p in model.named_parameters() if name not in ("motif_alpha", "rule_alpha")]
    optimizer = torch.optim.Adam([{"params": gates, "lr": cfg["gate_lr"], "eps": cfg["gate_eps"]},
                                  {"params": others, "lr": cfg["lr"]}])

    val_frames = {}
    for video_id in training_data["val_videos"]:
        frames = step05._sample_frames(training_data, video_id, cfg["frames_per_video"], cfg["positive_frame_ratio"],
                                       generator, max(cfg["windows"]))
        val_frames[video_id] = (frames, step05.sample_history(frames, cfg, generator))
    train_videos = training_data["train_videos"]
    best_val, best_state, history, counts = float("inf"), None, [], {}
    for epoch in range(cfg["num_epochs"]):
        model.train()
        regularize = epoch >= cfg["reg_warmup_epochs"]
        order = torch.randperm(len(train_videos), generator=generator).tolist()
        losses, precision_losses = [], []
        for start in range(0, len(order), cfg["videos_per_batch"]):
            parts = []
            for i in order[start:start + cfg["videos_per_batch"]]:
                frames = step05._sample_frames(training_data, train_videos[i], cfg["frames_per_video"],
                                               cfg["positive_frame_ratio"], generator, max(cfg["windows"]))
                parts.append(step05._video_rows(model, training_data, train_videos[i], frames, device,
                                                step05.sample_history(frames, cfg, generator)))
            rows = step05.concat_rows(parts)
            if len(rows["inputs"]) == 0:
                continue
            hazards, rules_now = model(rows["inputs"])
            window_loss = model.loss(hazards, rows["targets"], rows["at_risk"], rows["known"],
                                     step05.pos_weight_of(training_data, cfg))
            loss = window_loss
            if regularize:
                precision = model.precision_loss(rules_now, rows, counts, cfg["p_min"], cfg["precision_k"],
                                                 cfg["precision_decay"])
                loss = loss + cfg["sparsity_lambda"] * model.sparsity_loss() + cfg["precision_lambda"] * precision
                precision_losses.append(precision.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(window_loss.item())

        train_loss = sum(losses) / max(len(losses), 1)
        val_loss = step05._validation_loss(model, training_data, val_frames, cfg)
        stats = model_stats(model, cfg)
        print(f"[Epoch {epoch + 1}/{cfg['num_epochs']}] train loss {train_loss:.4f}, val loss {val_loss:.4f}"
              + (f", precision loss {sum(precision_losses) / len(precision_losses):.4f}" if precision_losses else "")
              + f", clauses {stats['model/clauses']}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, **stats})
        if run is not None:
            run.log({"train/window_loss": train_loss, "val/window_loss": val_loss,
                     **({"train/precision_loss": sum(precision_losses) / len(precision_losses)} if precision_losses else {}),
                     **stats}, step=epoch)
        # Model selection only once the regularization is on, so the kept model is a regularized one.
        if (regularize or epoch == cfg["num_epochs"] - 1) and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[Training] Best val loss {best_val:.4f}")
    utils_data.save_json(history, cfg["history_file"])
    return model


@torch.no_grad()
def model_stats(model, cfg):
    conf = torch.sigmoid(model.conf_logit)
    motif_body = (model.motif_mask() > cfg["mask_threshold"]).flatten(1).sum(dim=1).float()
    rule_body = (model.rule_mask() > cfg["mask_threshold"]).sum(dim=1).float()
    return {"model/clauses": int((conf > cfg["conf_threshold"]).sum()),
            "model/mean_confidence": conf.mean().item(),
            "model/max_confidence": conf.max().item(),
            "model/motif_body_mean": motif_body.mean().item(),
            "model/rule_body_mean": rule_body.mean().item(),
            "model/rules_with_motifs": int(((model.rule_mask() > cfg["mask_threshold"])
                                            .view(len(rule_body), model.num_slots, model.num_nodes)[..., :model.num_motifs]
                                            .any(dim=(1, 2))).sum())}


# -------------- Clauses --------------
@torch.no_grad()
def clause_counts(model, data, video_ids, cfg, desc="Counting clause firings"):
    """Per rule and head (R, A): fires (every body holds, the head at risk, the next
    slot observed), hits (the head switches on in the next slot), videos with a
    hit; per head the base rate. Full history, every frame."""
    device = cfg["device"]
    R, A = model.rule_alpha.shape[0], model.num_atoms
    fires, hits, videos = (torch.zeros((R, A), device=device) for _ in range(3))
    onsets, at_risk = torch.zeros(A, device=device), torch.zeros(A, device=device)
    for video_id in tqdm(video_ids, desc=desc):
        video_hits = torch.zeros((R, A), device=device)
        for frames in data["frames"][video_id].split(cfg["frames_per_video"]):
            rows = step05._video_rows(model, data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            fired = model.hard_rules(rows["inputs"], cfg["mask_threshold"]).float()
            risk = (rows["at_risk"] & rows["known"][:, 0]).float()
            onset = rows["targets"][:, 0].float()
            fires += fired.T @ risk
            video_hits += fired.T @ onset
            onsets += onset.sum(dim=0)
            at_risk += risk.sum(dim=0)
        hits += video_hits
        videos += video_hits > 0
    return {"fires": fires.cpu(), "hits": hits.cpu(), "videos": videos.cpu(),
            "base_rate": (onsets / at_risk.clamp(min=1)).cpu()}


def motif_text(motif, unique_atoms):
    return "[" + ", ".join(f"{step05._atom_name(unique_atoms[atom])}(X{slot}) {feature}@-{frames}f"
                           for feature, slot, atom, frames in motif) + "]"


def clause_text(clause, unique_atoms):
    def body(b):
        if b[0] == "motif":
            return f"motif{b[1]}{motif_text(clause['motifs'][str(b[1])], unique_atoms)}@slot-{b[2]}"
        return f"{step05._atom_name(unique_atoms[b[2]])}(X{b[1]})@slot-{b[3]}"
    return " -> ".join(body(b) for b in clause["bodies"]) + f" => {step05._atom_name(unique_atoms[clause['head']])}(X0)"


def extract_clauses(model, training_data, cfg):
    """Clauses (rule, head) with confidence > conf_threshold and at least
    min_support train hits in min_videos videos: bodies (motifs with their own
    bodies, raw holds; slot j before the head's), head, confidence, and on train
    / val the hits, fires, precision and the head's base rate. Sorted by train precision."""
    rules, motifs = model.extract(cfg["mask_threshold"], cfg["conf_threshold"])
    train = clause_counts(model, training_data, training_data["train_videos"], cfg, "Counting clause firings (train)")
    val = clause_counts(model, training_data, training_data["val_videos"], cfg, "Counting clause firings (val)")

    def ratio(a, b):
        return float(a) / float(b) if b > 0 else None

    clauses = []
    for rule in rules:
        r = rule["rule"]
        used = {str(b[1]): motifs[b[1]] for b in rule["bodies"] if b[0] == "motif"}
        for h, confidence in rule["heads"]:
            support, num_videos = int(train["hits"][r, h]), int(train["videos"][r, h])
            if support < cfg["min_support"] or num_videos < cfg["min_videos"]:
                continue
            clauses.append({
                "rule": r, "head": h, "bodies": [list(b) for b in rule["bodies"]], "motifs": used,
                "confidence": confidence, "support": support, "fires": int(train["fires"][r, h]),
                "precision": ratio(train["hits"][r, h], train["fires"][r, h]), "num_videos": num_videos,
                "base_rate": train["base_rate"][h].item(),
                "val_support": int(val["hits"][r, h]), "val_fires": int(val["fires"][r, h]),
                "val_precision": ratio(val["hits"][r, h], val["fires"][r, h]),
            })
    clauses.sort(key=lambda c: (-(c["precision"] or 0), -c["support"]))
    print(f"[Clauses] {len(rules)} rules with heads -> {len(clauses)} clauses kept")
    return clauses


def print_summary(clauses, cfg, num_examples=15):
    unique_atoms = step05.load_vocabulary(cfg)
    print("----------------- Step 05d Result (MotifRuleNet) -----------------")
    print(f"[Clauses]\t {len(clauses)} in {cfg['clause_file']}")
    if not clauses:
        return
    with_motifs = sum(any(b[0] == "motif" for b in c["bodies"]) for c in clauses)
    precise = [c for c in clauses if (c["val_precision"] or 0) >= cfg["p_min"]]
    print(f"[Bodies]\t {with_motifs} clauses use a motif; {len(precise)} reach p_min = {cfg['p_min']} on val")
    vals = sorted(c["val_precision"] for c in clauses if c["val_precision"] is not None)
    if vals:
        print(f"[Precision]\t val median {vals[len(vals) // 2]:.3f}, max {vals[-1]:.3f}")
    print(f"[Top {min(num_examples, len(clauses))}]")
    for c in clauses[:num_examples]:
        print(f"  prec {c['precision']:.3f} (val {c['val_precision'] if c['val_precision'] is None else round(c['val_precision'], 3)}, "
              f"conf {c['confidence']:.2f}, base {c['base_rate']:.4f})  support {c['support']:>4}  {clause_text(c, unique_atoms)}")


def main(input_data):
    print("----------------- Step 05d: Two-Level Motif Rule Net -----------------")
    cfg = get_cfg(input_data)
    if not cfg["enabled"]:
        print("[Skip]\t\t disabled")
        return
    model_fresh = step05.is_fresh(cfg["model_file"], cfg["training_data_file"])
    if cfg["step05_cfg"]["main_only"] and not model_fresh:
        print(f"[Skip]\t\t main_only: no cached motif net {cfg['model_file'].name}, not trained")
        return
    training_data = step05.load_training_data(cfg["step05_cfg"])
    clauses_fresh = model_fresh and step05.is_fresh(cfg["clause_file"], cfg["model_file"])
    run = None
    if not clauses_fresh:
        run = step05._init_wandb({**cfg["step05_cfg"], **{k: v for k, v in cfg.items() if k != "step05_cfg"}},
                                 {"num_atoms": training_data["num_atoms"]}, job_type="step05_motif_rule_net")
    try:
        if not model_fresh:
            model = train_motif_rule_net(training_data, cfg, run)
            torch.save(model.state_dict(), cfg["model_file"])
        model = load_motif_rule_net(training_data, cfg)
        if not clauses_fresh:
            utils_data.save_json(extract_clauses(model, training_data, cfg), cfg["clause_file"])
    finally:
        if run is not None:
            run.finish()
    print_summary(utils_data.load_json(cfg["clause_file"]), cfg)
