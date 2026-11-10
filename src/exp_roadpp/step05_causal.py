"""Step 05b: causal pruning of the trained step 05 rule net ("net_pruned").

The step 05 TemporalRuleNet learns rules (body atoms in the memory slots before
the head => head) that are associations: a body atom may only go along with the
head because of something else. This step tests every learned clause, atom by
atom, for conditional independence and prunes what does not hold:

1. A table per head h, from the train videos: one row per (agent, t) with the
   full memory, every slot_size frames (t, t + 10, ...: neighbouring frames are
   near copies), where h is at risk. Columns: the memory inputs (atom, self /
   other agent, slot), the confounders and the outcome Y = h switches on within
   the next slot (the clause's own claim, every body seen). All rows with Y = 1
   and a random sample of at most max_negatives others (case-control: the
   logistic slopes, which the tests use, are unbiased under it).
2. Test B: for each clause (rule r, head h) and each of its body atoms X, a
   likelihood-ratio test of a logistic regression of Y on X, the clause's other
   body atoms and the confounders, against the same without X: is X still
   associated with h given the rest of the clause and the confounders? The
   confounders: h itself held in the older memory (an atom that switches on
   again), the number of agents in the scene, and the num_confounders memory
   inputs most associated with Y that are no body atom of any clause of h.
   p-values over all tests are corrected with Benjamini-Hochberg; X is kept if
   q <= fdr and its effect is positive (the rule requires X to hold).
3. Pruning, both granularities: a body atom kept for none of its rule's heads is
   removed from the rule; a clause (r, h) none of whose atoms is kept loses its
   head weight; a rule with no clause left is off. Heads with fewer than
   min_positives onset rows cannot be tested and keep their clauses.
4. Optionally the kept clauses' head weights (and the hazard biases and the
   unknown-input priors) are fine-tuned for finetune_epochs epochs with every
   rule body frozen and every pruned weight held at 0, so the probabilities
   are calibrated again without the pruned evidence.

Outputs: causal_tests_<tag>.json (every test), pruned_rule_net_<tag>.pt (the
"net_pruned" step 06 evaluates), pruned_clauses_<tag>.json and its history.
The step 04 causal discovery is not used or changed.
"""
import copy
import os

import torch
from scipy.stats import chi2
from tqdm import tqdm

from src.exp_roadpp import step05, utils_data


def get_cfg(input_data):
    """Settings and files of step 05b; the model, data and split are step 05's."""
    base = step05.get_cfg(input_data)
    tag = base["model_file"].stem.replace("temporal_rule_net_", "")
    output_dir = input_data["output_dir"]
    return {
        "device": base["device"],
        "seed": base["seed"],
        # Table rows: one per agent and slot (every row_stride frames) with the full memory.
        "row_stride": base["slot_size"],
        # Case-control table per head: every onset row and at most this many others.
        "max_negatives": 50000,
        # Heads with fewer onset rows in the train videos keep their clauses untested.
        "min_positives": 20,
        # The memory inputs most associated with the head (no body atom of its clauses) as confounders.
        "num_confounders": 5,
        # Benjamini-Hochberg false discovery rate of the kept body atoms.
        "fdr": 0.05,
        # Ridge of the logistic regressions (not on the intercept), against perfect separation.
        "l2": 1e-3,
        # Fine-tuning of the kept clauses' weights after the pruning (0: none).
        "finetune_epochs": 5,
        "finetune_lr": 0.01,
        "wandb": base["wandb"],
        "wandb_project": base["wandb_project"],
        "test_file": output_dir / f"causal_tests_{tag}.json",
        "pruned_model_file": output_dir / f"pruned_rule_net_{tag}.pt",
        "pruned_clause_file": output_dir / f"pruned_clauses_{tag}.json",
        "pruned_history_file": output_dir / f"pruned_history_{tag}.json",
        "step05_cfg": base,
    }


# -------------- The table --------------
@torch.no_grad()
def collect_rows(model, training_data, video_ids, cfg):
    """Rows of every (agent, t) of the videos with the full memory, every
    row_stride frames: x (B, M) bool memory truth values, at_risk (B, A),
    onset (B, A) bool (the atom switches on within the next slot), agents (B,)
    the agents visible in t."""
    device = cfg["device"]
    memory = model.slot_size * model.num_slots
    parts = {"x": [], "at_risk": [], "onset": [], "agents": []}
    for video_id in tqdm(video_ids, desc="Building the causal table (train)"):
        X = step05._video_from_cells(training_data["videos"][video_id], training_data["num_atoms"], device)
        frames = torch.arange(memory, X.shape[0], cfg["row_stride"], device=device)
        if len(frames) == 0:
            continue
        rows = step05.video_rows(model, X, frames, training_data["min_off_frames"],
                                 roles=step05.video_roles(training_data["videos"][video_id], device))
        if len(rows["inputs"]) == 0:
            continue
        x, _k = model.split_inputs(rows["inputs"])
        t = rows["rows"][:, 0]
        parts["x"].append((x > 0.5).cpu())
        parts["at_risk"].append(rows["at_risk"].cpu())
        parts["onset"].append((rows["at_risk"] & (rows["first"] < model.slot_size)).cpu())
        parts["agents"].append(X[t].any(dim=2).sum(dim=1).float().cpu())
    return {key: torch.cat(value) for key, value in parts.items()}


def _logistic(design, y, l2, iterations=30):
    """Maximum (ridge-penalised, not on the intercept in column 0) log-likelihood
    and weights of a logistic regression of y on design (n, d), by Newton steps."""
    n, d = design.shape
    w = torch.zeros(d, dtype=design.dtype)
    penalty = torch.full((d,), l2, dtype=design.dtype)
    penalty[0] = 0.0
    for _ in range(iterations):
        z = design @ w
        p = torch.sigmoid(z)
        gradient = design.T @ (y - p) - penalty * w
        hessian = design.T @ (design * (p * (1 - p))[:, None]) + torch.diag(penalty) + 1e-9 * torch.eye(d, dtype=design.dtype)
        step = torch.linalg.solve(hessian, gradient)
        w = w + step
        if step.abs().max() < 1e-8:
            break
    z = design @ w
    log_likelihood = (y * torch.nn.functional.logsigmoid(z) + (1 - y) * torch.nn.functional.logsigmoid(-z)).sum()
    return log_likelihood.item(), w


def _benjamini_hochberg(p_values):
    """q-values of the p-values (same order)."""
    p = torch.tensor(p_values, dtype=torch.float64)
    order = torch.argsort(p)
    ranked = p[order] * len(p) / torch.arange(1, len(p) + 1, dtype=torch.float64)
    ranked = torch.flip(torch.cummin(torch.flip(ranked, [0]), 0).values, [0]).clamp(max=1.0)
    q = torch.empty_like(p)
    q[order] = ranked
    return q.tolist()


def run_tests(model, table, cfg):
    """Test B for every (rule, head, body atom) of the extracted clauses.

    Returns [{rule, head, input, atom, slot, offset, coef, statistic, p, q, kept,
    tested, num_positives, num_rows}] (untested entries: heads with too few
    onset rows; they are kept)."""
    generator = torch.Generator().manual_seed(cfg["seed"])
    step05_cfg = cfg["step05_cfg"]
    rules = model.extract_rules(step05_cfg["mask_threshold"], step05_cfg["weight_threshold"])
    body = model.rule_mask() > step05_cfg["mask_threshold"]
    by_head = {}
    for rule in rules:
        inputs = torch.nonzero(body[rule["rule"]]).flatten().tolist()
        for h, _weight in rule["heads"]:
            by_head.setdefault(h, []).append((rule["rule"], inputs))

    A, J = model.num_atoms, model.num_slots
    tests = []
    for h, clauses in tqdm(sorted(by_head.items()), desc="Conditional independence tests"):
        risk = table["at_risk"][:, h]
        positive = risk & table["onset"][:, h]
        num_positives = int(positive.sum())
        if num_positives < cfg["min_positives"]:
            for r, inputs in clauses:
                for m in inputs:
                    tests.append({"rule": r, "head": h, "input": m, "tested": False, "kept": True,
                                  "num_positives": num_positives})
            continue
        negatives = torch.nonzero(risk & ~positive).flatten()
        if len(negatives) > cfg["max_negatives"]:
            negatives = negatives[torch.randperm(len(negatives), generator=generator)[:cfg["max_negatives"]]]
        index = torch.cat([torch.nonzero(positive).flatten(), negatives])
        x = table["x"][index].double()
        y = positive[index].double()

        # Confounders: h held in the older memory (self, slots 2..J), the scene size, and the
        # inputs most associated with Y that are no body atom of any clause of h.
        held_before = x[:, [model.input_index(h, 0, j) for j in range(2, J + 1)]].amax(dim=1, keepdim=True)
        scene = torch.log1p(table["agents"][index]).double()[:, None]
        bodies = {m for _r, inputs in clauses for m in inputs}
        n1 = y.sum()
        a = y @ x                                         # rows with X = 1 and Y = 1
        b = x.sum(dim=0) - a                              # X = 1, Y = 0
        c = n1 - a
        d = len(y) - n1 - b
        log_odds = torch.log((a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5))).abs()
        log_odds[list(bodies)] = -1.0
        log_odds[x.sum(dim=0) < 5] = -1.0
        top = torch.argsort(log_odds, descending=True)[:cfg["num_confounders"]]
        top = [m for m in top.tolist() if log_odds[m] > 0]
        confounders = torch.cat([held_before, scene, x[:, top]], dim=1)
        intercept = torch.ones((len(y), 1), dtype=torch.float64)

        for r, inputs in clauses:
            for m in inputs:
                others = [o for o in inputs if o != m]
                reduced = torch.cat([intercept, x[:, others], confounders], dim=1)
                full = torch.cat([reduced, x[:, [m]]], dim=1)
                if x[:, m].sum() == 0 or x[:, m].sum() == len(y):
                    statistic, coef = 0.0, 0.0              # never (or always) true: no evidence
                else:
                    ll_reduced, _w = _logistic(reduced, y, cfg["l2"])
                    ll_full, w = _logistic(full, y, cfg["l2"])
                    statistic, coef = max(2 * (ll_full - ll_reduced), 0.0), w[-1].item()
                tests.append({"rule": r, "head": h, "input": m, "tested": True, "coef": coef,
                              "statistic": statistic, "p": float(chi2.sf(statistic, 1)),
                              "num_positives": num_positives, "num_rows": len(y),
                              "confounders": top})
    tested = [t for t in tests if t["tested"]]
    for t, q in zip(tested, _benjamini_hochberg([t["p"] for t in tested]) if tested else []):
        t["q"] = q
        t["kept"] = q <= cfg["fdr"] and t["coef"] > 0
    for t in tests:
        t["atom"], t["slot"], t["offset"] = model.decode_input(t["input"])
    return tests


# -------------- Pruning --------------
@torch.no_grad()
def prune(model, tests):
    """A copy of the model with the failed body atoms and clauses removed, and
    the (A, R) bool mask of the clause weights kept (trainable in the fine-tuning).

    A body atom kept for none of its rule's heads leaves the rule (its gate is
    closed and it is no fixed input any more); a clause (r, h) with no kept atom
    loses its head weight."""
    pruned = copy.deepcopy(model)
    clause_atoms = {}
    for t in tests:
        clause_atoms.setdefault((t["rule"], t["head"]), []).append(t)
    rule_keeps = {}
    keep = torch.zeros_like(pruned.head_weight, dtype=torch.bool)
    removed_clauses = 0
    for (r, h), atoms in clause_atoms.items():
        kept = {t["input"] for t in atoms if t["kept"]}
        rule_keeps.setdefault(r, set()).update(kept)
        if kept:
            keep[h, r] = True
        else:
            pruned.head_weight[h, r] = 0.0
            removed_clauses += 1
    removed_atoms = 0
    for r, kept in rule_keeps.items():
        body = {t["input"] for (rule, _h), atoms in clause_atoms.items() if rule == r for t in atoms}
        for m in body - kept:
            pruned.fixed_inputs[r, m] = False
            pruned.log_alpha[r, m] = -30.0
            removed_atoms += 1
    stats = {"clauses": len(clause_atoms), "removed_clauses": removed_clauses,
             "rules": len(rule_keeps), "removed_rules": sum(not kept for kept in rule_keeps.values()),
             "body_atoms": sum(len({t["input"] for (rule, _h), atoms in clause_atoms.items() if rule == r for t in atoms})
                               for r in rule_keeps),
             "removed_atoms": removed_atoms}
    return pruned, keep, stats


def finetune(pruned, keep, training_data, cfg, run=None):
    """Fine-tune the kept clause weights, the hazard biases and the unknown-input
    priors (`step05.fit_onset_model`, best val epoch), every rule body frozen and
    every other head weight held where it is."""
    pruned.log_alpha.requires_grad_(False)
    pruned.head_weight.register_hook(lambda grad: grad * keep.to(grad.device))
    params = [pruned.head_weight, pruned.head_bias, pruned.unknown_logit]
    optimizer = torch.optim.Adam(params, lr=cfg["finetune_lr"])
    finetune_cfg = {**cfg["step05_cfg"], "num_epochs": cfg["finetune_epochs"]}
    model = step05.fit_onset_model(pruned, training_data, finetune_cfg, optimizer, cfg["pruned_history_file"], run=run)
    model.log_alpha.requires_grad_(True)
    return model


def load_pruned_rule_net(data, step05_cfg, cfg=None):
    """The net_pruned of step 05b in eval mode, or None if it was not made."""
    cfg = cfg or get_cfg_from_step05(step05_cfg)
    if not os.path.exists(cfg["pruned_model_file"]):
        return None
    return step05.load_temporal_rule_net(data, step05_cfg, cfg["pruned_model_file"])


def get_cfg_from_step05(step05_cfg):
    """The step 05b files next to the given step 05 settings (for step 06)."""
    tag = step05_cfg["model_file"].stem.replace("temporal_rule_net_", "")
    output_dir = step05_cfg["model_file"].parent
    return {"pruned_model_file": output_dir / f"pruned_rule_net_{tag}.pt"}


def _print_summary(tests, stats, cfg):
    unique_atoms = step05.load_vocabulary(cfg["step05_cfg"])
    tested = [t for t in tests if t["tested"]]
    print("----------------- Step 05b Result (causal pruning) -----------------")
    print(f"[Tests]\t\t {len(tests)} (rule, head, body atom), {len(tested)} tested, "
          f"{len(tests) - len(tested)} untested (heads with < {cfg['min_positives']} onset rows, kept)")
    if tested:
        kept = sum(t["kept"] for t in tested)
        negative = sum(t["coef"] <= 0 for t in tested)
        print(f"[Kept]\t\t {kept}/{len(tested)} tested body atoms (q <= {cfg['fdr']} and positive effect); "
              f"{negative} with a non-positive effect")
    print(f"[Pruned]\t {stats['removed_atoms']}/{stats['body_atoms']} body atoms, "
          f"{stats['removed_clauses']}/{stats['clauses']} clauses, {stats['removed_rules']}/{stats['rules']} rules")
    by_head = {}
    for t in tested:
        by_head.setdefault(t["head"], []).append(t)
    print("[Per head]\t onset rows, tested atoms kept / tested")
    for h, items in sorted(by_head.items(), key=lambda kv: -kv[1][0]["num_positives"])[:15]:
        print(f"  {items[0]['num_positives']:>6}  {sum(t['kept'] for t in items):>4}/{len(items):<4} "
              f"{step05._atom_name(unique_atoms[h])}")


def main(input_data):
    print("----------------- Step 05b: Causal Pruning of the Rule Net -----------------")
    cfg = get_cfg(input_data)
    step05_cfg = cfg["step05_cfg"]
    if step05_cfg["main_only"]:
        print("[Skip]\t\t main_only: only the main rule net is trained")
        return
    if not os.path.exists(step05_cfg["model_file"]):
        print(f"[Skip]\t\t no step 05 model {step05_cfg['model_file']}")
        return
    training_data = step05.load_training_data(step05_cfg)
    model = step05.load_temporal_rule_net(training_data, step05_cfg)

    # 1-2. The table and the conditional independence tests, redone when the model changes.
    if not step05.is_fresh(cfg["test_file"], step05_cfg["model_file"], step05_cfg["training_data_file"]):
        table = collect_rows(model, training_data, training_data["train_videos"], cfg)
        print(f"[Table]\t\t {len(table['x'])} rows (agent, slot) x {table['x'].shape[1]} memory inputs")
        utils_data.save_json(run_tests(model, table, cfg), cfg["test_file"])
    tests = utils_data.load_json(cfg["test_file"])

    # 3-4. Pruning and fine-tuning, redone when the tests change.
    pruned, keep, stats = prune(model, tests)
    if not step05.is_fresh(cfg["pruned_model_file"], cfg["test_file"]):
        run = None
        if cfg["finetune_epochs"] > 0:
            run = step05._init_wandb({**step05_cfg, **{k: v for k, v in cfg.items() if k != "step05_cfg"}},
                                     {"num_atoms": training_data["num_atoms"], **stats}, job_type="step05_causal")
        try:
            if cfg["finetune_epochs"] > 0:
                pruned = finetune(pruned, keep, training_data, cfg, run)
            torch.save(pruned.state_dict(), cfg["pruned_model_file"])
            pruned = step05.load_temporal_rule_net(training_data, step05_cfg, cfg["pruned_model_file"])
            utils_data.save_json(step05.extract_temporal_clauses(pruned, training_data, step05_cfg, run),
                                 cfg["pruned_clause_file"])
        finally:
            if run is not None:
                run.finish()
    _print_summary(tests, stats, cfg)
    print(f"[Files]\t\t {cfg['test_file']}\n\t\t {cfg['pruned_model_file']}\n\t\t {cfg['pruned_clause_file']}")
