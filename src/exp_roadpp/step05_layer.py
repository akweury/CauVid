"""Step 05 (layered): learn temporal rules with the LayeredRuleNet.

Each layer is one block of block_size frames (`step05.coarsen_video`). The model
predicts the full atom state of every agent in a layer from the K layers before
it, as on / off transitions, and can roll its own predictions forward: given
layers 1..20 it predicts 21, then 22 from 2..21, and so on. Training unrolls
rollout_steps layers per window with scheduled sampling (the predicted layer
replaces the true one with a growing probability), so rules learn to chain.

Shares the training videos and their train / val split with step 05
(`step05.prepare_training_data`) and its anchor clauses.
"""
import os
import torch
from tqdm import tqdm

from src.exp_roadpp import step05, utils_data
from src.exp_roadpp.cauvid import LayeredRuleNet


# -------------- Layers and windows --------------
def _video_layers(training_data, video_id, cfg):
    """(L, N, A) float layer states of one video, blocks of block_size frames."""
    X = step05._video_from_cells(training_data["videos"][video_id], training_data["num_atoms"], cfg["device"])
    return step05.coarsen_video(X, cfg["block_size"]).float()


def _change_layers(layers):
    """(P,) long, the layers t where some agent visible in t - 1 and t changes an atom."""
    visible = layers.amax(dim=2) > 0
    changed = (layers[1:] != layers[:-1]).any(dim=2) & visible[1:] & visible[:-1]
    return torch.nonzero(changed.any(dim=1)).flatten() + 1


def _sample_starts(layers, num_windows, positive_ratio, steps, generator):
    """(<= num_windows,) first target layers 1..L - steps of the windows; a share
    positive_ratio of them has a state change in one of its steps target layers."""
    last = layers.shape[0] - steps
    if last < 1:
        return torch.zeros(0, dtype=torch.long)
    num_positive = round(positive_ratio * num_windows)
    changes = _change_layers(layers).cpu()
    positives = changes[torch.randint(len(changes), (num_positive,), generator=generator)] if len(changes) else changes
    positives = (positives - torch.randint(steps, positives.shape, generator=generator)).clamp(1, last)
    others = torch.randint(1, last + 1, (num_windows - len(positives),), generator=generator)
    return torch.unique(torch.cat([positives, others]))


def _windows(layers, starts, num_layers, steps):
    """(W, K + steps, N, A) layers starts - K..starts + steps - 1 of each window;
    layers before the video are empty."""
    padded = torch.cat([layers.new_zeros((num_layers, *layers.shape[1:])), layers])
    index = starts.to(layers.device)[:, None] + torch.arange(num_layers + steps, device=layers.device)
    return padded[index]


def unroll(model, windows, steps, feed_prob=0.0, generator=None):
    """Predict the last steps layers of each window, each from the K layers before it.

    With probability feed_prob per window and step, the predicted layer replaces
    the true one in the context of the next steps (the gradient passes through
    it); otherwise the true layer is used (teacher forcing). Only the agents
    visible in both the previous and the target layer are predicted; agents
    entering or leaving keep their true states.

    Returns one dict per step: on, off (B, A) logits, rules (B, R), pred (B, A)
    next states, prev / target (B, A) true previous and target states, rows (B,)
    their rows w * N + n in the windows.
    """
    K = model.num_layers
    W, _L, N, A = windows.shape
    visible = windows.amax(dim=3) > 0                                  # (W, K + steps, N)
    layers = list(windows.unbind(1))
    outputs = []
    for j in range(steps):
        t = K + j
        valid = (visible[:, t - 1] & visible[:, t]).flatten()
        rows = torch.nonzero(valid).flatten()
        inputs = model.layer_inputs(torch.stack(layers[j:t], dim=1))[rows]
        on, off, rules = model(inputs)
        prev_used = layers[t - 1].reshape(W * N, A)[rows]
        pred = model.next_state(prev_used, on, off)
        outputs.append({"on": on, "off": off, "rules": rules, "pred": pred, "rows": rows,
                        "prev": windows[:, t - 1].reshape(W * N, A)[rows],
                        "target": windows[:, t].reshape(W * N, A)[rows]})
        if feed_prob > 0 and j + 1 < steps:
            feed = (torch.rand(W, generator=generator) < feed_prob).to(windows.device)
            feed = feed.repeat_interleave(N)[rows]
            truth = windows[:, t].reshape(W * N, A)
            layers[t] = truth.index_put((rows,), torch.where(feed[:, None], pred, truth[rows])).view(W, N, A)
    return outputs


# -------------- Training --------------
def _feed_prob(epoch, cfg):
    """Scheduled sampling: 0 in the teacher_forcing_epochs, then a linear ramp to
    max_feed_prob over feed_ramp_epochs."""
    ramp = (epoch - cfg["teacher_forcing_epochs"] + 1) / max(cfg["feed_ramp_epochs"], 1)
    return cfg["max_feed_prob"] * min(max(ramp, 0.0), 1.0)


def _window_losses(model, training_data, video_windows, cfg, feed_prob, generator=None):
    """State loss (weighted mean over the steps) and the outputs of all steps of
    the windows {video_id: starts}."""
    loss_sum, weight_sum, outputs = 0.0, 0.0, []
    for video_id, starts in video_windows.items():
        if len(starts) == 0:
            continue
        windows = _windows(_video_layers(training_data, video_id, cfg), starts, model.num_layers,
                           cfg["rollout_steps"])
        for out in unroll(model, windows, cfg["rollout_steps"], feed_prob, generator):
            if len(out["pred"]) == 0:
                continue
            s, w = model.state_loss(out["pred"], out["prev"], out["target"], cfg["pos_weight"])
            loss_sum, weight_sum = loss_sum + s, weight_sum + w
            outputs.append(out)
    if not outputs:
        return None, outputs
    return loss_sum / weight_sum.clamp(min=1), outputs


@torch.no_grad()
def _validation_losses(model, training_data, val_windows, cfg):
    """Teacher-forced and fully rolled-out (feed 1) state loss on the val windows."""
    model.eval()
    losses = {}
    for name, feed_prob in (("teacher_forced", 0.0), ("rollout", 1.0)):
        loss, _outputs = _window_losses(model, training_data, val_windows, cfg, feed_prob)
        losses[name] = loss.item() if loss is not None else float("nan")
    return losses


@torch.no_grad()
def _model_stats(model, cfg):
    """Learned body inputs (beyond the fixed anchor) of the rules that may learn any,
    and how far their gates have climbed from init_log_alpha toward 0 (the
    extraction threshold), so a short run shows whether bodies are growing."""
    trained = model.learnable.any(dim=1)
    extra = ((model.rule_mask() > cfg["mask_threshold"]) & model.learnable).sum(dim=1)[trained].float()
    if len(extra) == 0:
        return {}
    gates = model.log_alpha[model.learnable]
    return {"model/sparsity_loss": model.sparsity_loss().item(),
            "model/extra_inputs_per_rule": extra.mean().item(),
            "model/max_extra_inputs": extra.max().item(),
            "model/share_rules_extended": (extra > 0).float().mean().item(),
            "gates/max_log_alpha": gates.max().item(),
            "gates/mean_log_alpha": gates.mean().item(),
            "gates/count_above_-8": (gates > -8).sum().item(),
            "gates/count_above_-5": (gates > -5).sum().item(),
            "gates/count_above_-2": (gates > -2).sum().item(),
            "gates/count_above_0": (gates > 0).sum().item()}


def train_layered_rule_net(training_data, cfg, run=None):
    """Train a LayeredRuleNet seeded from the anchor clauses; returns it with the
    weights of the lowest val rollout loss.

    One epoch visits every training video once, in batches of videos_per_batch
    videos with windows_per_video windows each. After reg_warmup_epochs the loss
    adds sparsity_lambda * sparsity and precision_lambda * the m-estimate precision loss.
    """
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    anchors = step05.anchor_rules(cfg)
    copies = len(cfg["anchor_offsets"]) * (1 + cfg["num_specialized"])
    model = LayeredRuleNet(training_data["num_atoms"], cfg["num_layers"],
                           len(anchors) * copies + cfg["num_free_rules"], cfg["init_log_alpha"])
    num_seeded = model.seed_rules(anchors, cfg["anchor_offsets"], cfg["num_specialized"], cfg["seed_head_weight"])
    model = model.to(cfg["device"])
    print(f"[Model] {model.log_alpha.shape[0]} rules ({num_seeded} seeded from {len(anchors)} anchor bodies, "
          f"{cfg['num_free_rules']} free), {model.num_inputs} inputs ({cfg['num_layers']} layers "
          f"of {cfg['block_size']} frames)")

    optimizer = torch.optim.Adam([
        # The mask gradients are ~1e-12 (sigmoid' of -11): a tiny eps keeps Adam scale-free on them.
        {"params": [model.log_alpha], "lr": cfg["gate_lr"], "eps": cfg["gate_eps"]},
        {"params": [model.on_weight, model.off_weight, model.on_bias, model.off_bias], "lr": cfg["lr"]},
    ])

    def sample(video_ids):
        return {v: _sample_starts(_video_layers(training_data, v, cfg), cfg["windows_per_video"],
                                  cfg["positive_window_ratio"], cfg["rollout_steps"], generator)
                for v in video_ids}

    val_windows = sample(training_data["val_videos"])
    train_videos = training_data["train_videos"]
    best_val_loss, best_state, history = float("inf"), None, []
    precision_counts = {}
    for epoch in range(cfg["num_epochs"]):
        model.train()
        feed_prob = _feed_prob(epoch, cfg)
        regularize = epoch >= cfg["reg_warmup_epochs"]
        order = torch.randperm(len(train_videos), generator=generator).tolist()
        epoch_losses = []
        for start in range(0, len(order), cfg["videos_per_batch"]):
            batch = sample([train_videos[i] for i in order[start:start + cfg["videos_per_batch"]]])
            state_loss, outputs = _window_losses(model, training_data, batch, cfg, feed_prob, generator)
            if state_loss is None:
                continue
            loss = state_loss
            if regularize:
                rules, prev, target = (torch.cat([o[key] for o in outputs]) for key in ("rules", "prev", "target"))
                loss = (loss + cfg["sparsity_lambda"] * model.sparsity_loss()
                        + cfg["precision_lambda"] * model.precision_loss(rules, prev, target, precision_counts,
                                                                         cfg["precision_k"], cfg["precision_decay"]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(state_loss.item())

        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        val = _validation_losses(model, training_data, val_windows, cfg)
        print(f"[Epoch {epoch + 1}/{cfg['num_epochs']}] train loss {train_loss:.4f} (feed {feed_prob:.2f}), "
              f"val teacher-forced {val['teacher_forced']:.4f}, val rollout {val['rollout']:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "feed_prob": feed_prob,
                        "val_teacher_forced_loss": val["teacher_forced"], "val_rollout_loss": val["rollout"]})
        if run is not None:
            run.log({"train/state_loss": train_loss, "train/feed_prob": feed_prob,
                     "val/teacher_forced_loss": val["teacher_forced"], "val/rollout_loss": val["rollout"],
                     **_model_stats(model, cfg)}, step=epoch)
        if val["rollout"] < best_val_loss:
            best_val_loss = val["rollout"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[Training] Best val rollout loss {best_val_loss:.4f}")
    utils_data.save_json(history, cfg["history_file"])
    if run is not None:
        run.summary["best_val_rollout_loss"] = best_val_loss
    return model


def load_layered_rule_net(training_data, cfg):
    """The trained LayeredRuleNet of cfg["model_file"], in eval mode on cfg["device"]."""
    state = torch.load(cfg["model_file"], map_location="cpu")
    model = LayeredRuleNet(training_data["num_atoms"], cfg["num_layers"], state["log_alpha"].shape[0])
    model.load_state_dict(state)
    return model.to(cfg["device"]).eval()


# -------------- Clauses --------------
@torch.no_grad()
def _rule_head_counts(model, training_data, video_ids, cfg, desc="Counting rule firings"):
    """Teacher-forced counts over every layer of the given videos, for each rule r,
    head kind ("on", "off") and head atom h, as dicts of CPU (R, A) tensors:
      fires: rows where every body input of r holds and h can happen (off for
          "on", on for "off"); hits: the fires where it happens; videos: the
          videos with a hit; anchor_fires / anchor_hits: the same for the fixed
          anchor input of r alone (0 for free rules);
    plus events (A,) / at_risk (A,) of each kind over all rows.
    """
    R, A, device = model.log_alpha.shape[0], model.num_atoms, cfg["device"]
    anchor_body = model.fixed_inputs.float()
    anchor_size = anchor_body.sum(dim=1)
    counts = {kind: {name: torch.zeros((R, A), device=device)
                     for name in ("fires", "hits", "videos", "anchor_fires", "anchor_hits")}
              for kind in ("on", "off")}
    for kind in counts:
        counts[kind]["events"] = torch.zeros(A, device=device)
        counts[kind]["at_risk"] = torch.zeros(A, device=device)
    for video_id in tqdm(video_ids, desc=desc):
        layers = _video_layers(training_data, video_id, cfg)
        video_hits = {kind: torch.zeros((R, A), device=device) for kind in counts}
        for starts in torch.arange(1, layers.shape[0]).split(cfg["windows_per_video"]):
            windows = _windows(layers, starts, model.num_layers, 1)
            visible = windows.amax(dim=3) > 0
            rows = torch.nonzero((visible[:, -2] & visible[:, -1]).flatten()).flatten()
            if len(rows) == 0:
                continue
            inputs = model.layer_inputs(windows[:, :-1])[rows]
            prev = windows[:, -2].reshape(-1, A)[rows] > 0.5
            target = windows[:, -1].reshape(-1, A)[rows] > 0.5
            fired = (model.partial_activation(inputs, cfg["mask_threshold"]) >= 1).float()
            anchor_fired = ((inputs @ anchor_body.T >= anchor_size) & (anchor_size > 0)).float()
            for kind, risk, happens in (("on", ~prev, ~prev & target), ("off", prev, prev & ~target)):
                c, risk, happens = counts[kind], risk.float(), happens.float()
                c["fires"] += fired.T @ risk
                video_hits[kind] += fired.T @ happens
                c["anchor_fires"] += anchor_fired.T @ risk
                c["anchor_hits"] += anchor_fired.T @ happens
                c["events"] += happens.sum(dim=0)
                c["at_risk"] += risk.sum(dim=0)
        for kind in counts:
            counts[kind]["hits"] += video_hits[kind]
            counts[kind]["videos"] += video_hits[kind] > 0
    return {kind: {name: value.cpu() for name, value in c.items()} for kind, c in counts.items()}


def _layered_clause_text(clause, unique_atoms):
    # Bodies earliest first: "Car:MovAway(X1)@-3L -> Ped:Wait(X0)@-1L => +Car:Stop(X0)".
    def atom(atom_index, slot):
        return f"{step05._atom_name(unique_atoms[atom_index])}(X{slot})"
    body = " -> ".join(f"{atom(a, slot)}@-{offset}L" for a, slot, offset in clause["bodies"])
    sign = "+" if clause["kind"] == "on" else "-"
    return f"{body} => {sign}{atom(*clause['head'])}"


def extract_layered_clauses(model, training_data, cfg, run=None):
    """Clauses of the trained model, one per (rule node, head kind, head) with a
    large head weight, kept if the rule hits the head in at least min_support
    training rows of at least min_videos videos, and an extended rule (more atoms
    than its anchor) beats its anchor alone by min_gain_over_anchor on train.

    Each clause: bodies [(atom, slot, offset)] earliest first (the atom holds
    offset layers of block_size frames before the head; slot 0 the head agent,
    1 some other agent), head (atom, 0), kind "on" (switches on) or "off",
    anchor (atom, slot, offset) or None, weight, and on the training videos
    support (hits), fires, confidence, num_videos, anchor_confidence; the val_
    fields are the same on the val videos, plus val_recall and val_base_rate.
    Sorted by confidence.
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
        for h, kind, weight in rule["heads"]:
            tr, va = train[kind], val[kind]
            support, num_videos, fires = int(tr["hits"][r, h]), int(tr["videos"][r, h]), int(tr["fires"][r, h])
            if support < cfg["min_support"] or num_videos < cfg["min_videos"]:
                continue
            confidence = support / max(fires, 1)
            anchor_confidence = ratio(tr["anchor_hits"][r, h], tr["anchor_fires"][r, h])
            extended = anchor is not None and len(rule["bodies"]) > 1
            if extended and confidence < (anchor_confidence or 0.0) + cfg["min_gain_over_anchor"]:
                continue
            clauses.append({
                "bodies": [list(b) for b in rule["bodies"]],
                "body_scopes": [slot for _atom, slot, _offset in rule["bodies"]],
                "head": [h, 0],
                "kind": kind,
                "anchor": anchor,
                "rule": r,
                "weight": weight,
                "support": support,
                "fires": fires,
                "confidence": confidence,
                "num_videos": num_videos,
                "anchor_confidence": anchor_confidence,
                "val_support": int(va["hits"][r, h]),
                "val_fires": int(va["fires"][r, h]),
                "val_confidence": ratio(va["hits"][r, h], va["fires"][r, h]),
                "val_anchor_confidence": ratio(va["anchor_hits"][r, h], va["anchor_fires"][r, h]),
                "val_recall": ratio(va["hits"][r, h], va["events"][h]),
                "val_base_rate": ratio(va["events"][h], va["at_risk"][h]),
            })
    # A specialized copy that learned no atom repeats its anchor rule: keep one clause per body and head.
    clauses.sort(key=lambda c: (-c["confidence"], -c["support"], -c["weight"]))
    unique = {}
    for clause in clauses:
        unique.setdefault((str(clause["bodies"]), clause["head"][0], clause["kind"]), clause)
    clauses = list(unique.values())

    print(f"[Rules] {len(rules)} rule nodes with heads -> {len(clauses)} clauses kept")
    if run is not None:
        import wandb
        table = wandb.Table(columns=["clause", "kind", "confidence", "support", "fires", "num_videos",
                                     "weight", "length"])
        unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
        for clause in clauses:
            table.add_data(_layered_clause_text(clause, unique_atoms), clause["kind"], clause["confidence"],
                           clause["support"], clause["fires"], clause["num_videos"], clause["weight"],
                           len(clause["bodies"]))
        run.log({"clauses": table})
        run.summary["num_clauses"] = len(clauses)
        run.summary["num_rules_with_heads"] = len(rules)
    return clauses


def _print_summary(clauses, cfg, num_examples=15):
    """Terminal summary of the layered step 05 result, printed on every run."""
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    print(f"----------------- Step 05 (layered) Result -----------------")
    print(f"[Clauses]\t {len(clauses)} in {cfg['clause_file']}")
    if not clauses:
        print("[Clauses]\t none passed min_support / min_videos / weight_threshold / min_gain_over_anchor")
        return
    for kind in ("on", "off"):
        rows = [c for c in clauses if c["kind"] == kind]
        lengths = {}
        for clause in rows:
            lengths[len(clause["bodies"])] = lengths.get(len(clause["bodies"]), 0) + 1
        print(f"[{kind:>3} heads]\t {len(rows)} clauses, body length "
              + (", ".join(f"{n} atoms: {count}" for n, count in sorted(lengths.items())) or "-"))
    val_confidences = sorted(c["val_confidence"] for c in clauses if c.get("val_confidence") is not None)
    if val_confidences:
        print(f"[Confidence]\t median {val_confidences[len(val_confidences) // 2]:.3f}, "
              f"max {val_confidences[-1]:.3f} (val, {len(val_confidences)} clauses firing on val)")
    print(f"[Top {min(num_examples, len(clauses))}]")
    for clause in clauses[:num_examples]:
        print(f"  conf {clause['confidence']:.2f}  support {clause['support']:>5}  "
              f"{_layered_clause_text(clause, unique_atoms)}")


# -------------- Step --------------
def get_cfg(input_data):
    """Settings and files of the layered step 05; the data, split and anchors are step 05's."""
    base = step05.get_cfg(input_data)
    block_size, num_layers = 5, 20
    return {
            'device': base['device'],
            'seed': base['seed'],
            # Layers: blocks of block_size frames; each layer is predicted from the
            # num_layers layers before it (5 * 20 = 100 frames of context).
            'block_size': block_size,
            'num_layers': num_layers,
            # Rules: per anchor body and layer offset in anchor_offsets, one anchor-only
            # rule and num_specialized copies that may add atoms; plus num_free_rules
            # unseeded ones (e.g. for the "off" heads, which have no anchors).
            'anchor_offsets': [1, 2, 4],
            'num_specialized': 2,
            'num_free_rules': 128,
            'seed_head_weight': 1.0,
            # Mask sigmoid(-11) ~ 2e-5 per input: the rules start on (about 0.9 over the
            # 6320 mostly-0 inputs) and every input gets a gradient.
            'init_log_alpha': -11.0,
            # Training: windows of num_layers context + rollout_steps target layers.
            'num_epochs': base['num_epochs'],
            'lr': 0.01,
            'gate_lr': 0.05,
            'gate_eps': 1e-20,
            'videos_per_batch': 4,
            'windows_per_video': 8,
            'positive_window_ratio': 0.5,
            'rollout_steps': 5,
            # Scheduled sampling: teacher forcing first, then predicted layers fed back.
            'teacher_forcing_epochs': 20,
            'feed_ramp_epochs': 50,
            'max_feed_prob': 0.5,
            'pos_weight': 20.0,
            # Regularization after reg_warmup_epochs: L1 of the learned mask (drops
            # redundant atoms) and the m-estimate precision of each learnable rule,
            # (hits + precision_k * p0) / (fires + precision_k).
            'reg_warmup_epochs': 10,
            'sparsity_lambda': 1e-4,
            'precision_lambda': 0.1,
            'precision_k': 10.0,
            'precision_decay': 0.9,
            # Rule extraction.
            'mask_threshold': 0.5,
            'weight_threshold': 0.1,
            'min_support': 10,
            'min_videos': 2,
            'min_gain_over_anchor': 0.0,
            'wandb': base['wandb'],
            'wandb_project': base['wandb_project'],
            'causal_anchor_clause_file': base['causal_anchor_clause_file'],
            'unique_atoms_file': base['unique_atoms_file'],
            'training_data_file': base['training_data_file'],
            'model_file': input_data["output_dir"] / f"layered_rule_net_b{block_size}_k{num_layers}.pt",
            'clause_file': input_data["output_dir"] / f"layered_clauses_b{block_size}_k{num_layers}.json",
            'history_file': input_data["output_dir"] / f"layered_history_b{block_size}_k{num_layers}.json",
            'step05_cfg': base,
    }


def main(input_data):
    print(f"----------------- Step 05 (layered): Layered Rule Learning -----------------")
    cfg = get_cfg(input_data)

    # 1. The step 05 training videos and split (the dataset's train / val), prepared by step 05 if missing.
    training_data = step05.load_training_data(cfg["step05_cfg"])
    model_fresh = step05.is_fresh(cfg["model_file"], cfg["training_data_file"])
    clauses_fresh = model_fresh and step05.is_fresh(cfg["clause_file"], cfg["model_file"])

    run = None
    if not clauses_fresh:
        wandb_cfg = {k: v for k, v in cfg.items() if k != "step05_cfg"}
        run = step05._init_wandb(wandb_cfg, {"num_train_videos": len(training_data["train_videos"]),
                                             "num_val_videos": len(training_data["val_videos"]),
                                             "num_atoms": training_data["num_atoms"]},
                                 job_type="step05_layered_rule_net")
    try:
        # 2. Train the LayeredRuleNet.
        if not model_fresh:
            model = train_layered_rule_net(training_data, cfg, run)
            torch.save(model.state_dict(), cfg["model_file"])
        model = load_layered_rule_net(training_data, cfg)

        # 3. Read the learned on / off rules out of the model.
        if not clauses_fresh:
            utils_data.save_json(extract_layered_clauses(model, training_data, cfg, run), cfg["clause_file"])
    finally:
        if run is not None:
            run.finish()

    _print_summary(utils_data.load_json(cfg["clause_file"]), cfg)
