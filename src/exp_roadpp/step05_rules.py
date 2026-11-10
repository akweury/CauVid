"""Step 05c: select a rule set where every rule is precise and the set covers the events.

The step 05 net adds the evidence of all its rules (a weighted sum in logit
space), so a single rule need not be precise and nothing makes the rules share
out the events. Here the rules are used as a rule set instead:

  a rule (body atoms in the memory slots before the head => head) fires on a row
  (agent, t) when every body atom is seen in the memory; its precision for window
  W is P(the head switches on within W frames | it fires, the head is at risk),
  counted on the train videos; it covers an event (an onset) for window W when it
  fired on one of the W frames up to the onset.

For every precision level p_min of `precision_levels`, window W and head, a greedy
set cover keeps only the rules with precision >= p_min and at least min_support
hits, and adds the rule covering the most events not covered yet, until a rule
adds fewer than min_new_events. The level with the best F1 of event recall and
the rule set's row precision (any of its rules fires) on the val videos is the
one step 06 evaluates as "rule_set": the score of a row is the highest precision
of its selected rules that fire, 0 if none. Step 06 also reports, on the test
videos, the recall and precision of every level: how many events a set of rules,
each at least p_min precise, can cover.

The candidate rules are those of the step 05b pruned net, or of the step 05 net
if it was not made. Output: rule_selection_<tag>.json.
"""
import os

import torch
from tqdm import tqdm

from src.exp_roadpp import step05, step05_causal, utils_data


def get_cfg(input_data):
    """Settings and files of step 05c; the model, data and split are step 05's."""
    base = step05.get_cfg(input_data)
    tag = base["model_file"].stem.replace("temporal_rule_net_", "")
    return {
        "device": base["device"],
        # Minimum precision of every rule of a set (absolute, for the window of the set).
        "precision_levels": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
        # A rule must hit its head on at least this many train rows (for that window).
        "min_support": 10,
        # The greedy cover stops when the best rule adds fewer events than this.
        "min_new_events": 2,
        "selection_file": input_data["output_dir"] / f"rule_selection_{tag}.json",
        "step05_cfg": base,
    }


def selection_file_of(step05_cfg):
    """The step 05c file next to the given step 05 settings (for step 06)."""
    tag = step05_cfg["model_file"].stem.replace("temporal_rule_net_", "")
    return step05_cfg["model_file"].parent / f"rule_selection_{tag}.json"


def candidate_rules(model, step05_cfg):
    """[(rule, head, [body memory inputs])] of every extracted clause of the model."""
    body = model.rule_mask() > step05_cfg["mask_threshold"]
    pairs = []
    for rule in model.extract_rules(step05_cfg["mask_threshold"], step05_cfg["weight_threshold"]):
        inputs = torch.nonzero(body[rule["rule"]]).flatten().tolist()
        for h, _weight in rule["heads"]:
            pairs.append((rule["rule"], h, inputs))
    return pairs


def _body_matrix(pairs, num_inputs, device):
    body = torch.zeros((len(pairs), num_inputs), device=device)
    for p, (_r, _h, inputs) in enumerate(pairs):
        body[p, inputs] = 1.0
    return body, body.sum(dim=1)


@torch.no_grad()
def rule_statistics(model, pairs, data, video_ids, cfg, selections=None):
    """Counts of the candidate rules on the videos (rows t >= the memory).

    Returns {"fires", "hits": (W, P) per rule and window (at-risk rows with the
    window observed where the rule fires / and the head switches on within it),
    "base": (W, A) onset share of the at-risk rows, "events": [(video, t_e, head)]
    the exact onsets with t_e >= memory, "lead": {p: {event: j}} the fewest frames
    before the event the rule fired (it covers the event for every W > j),
    "union": {level: {"fires", "hits": (W,)}} the rows where any selected rule of
    the head fires, for each selection {level: {k: {h: [p]}}} given}.
    """
    device = cfg["device"]
    A, W, top = data["num_atoms"], len(model.windows), max(model.windows)
    memory = model.slot_size * model.num_slots
    body, size = _body_matrix(pairs, model.num_inputs, device)
    heads = torch.tensor([h for _r, h, _i in pairs], device=device)
    head_onehot = torch.nn.functional.one_hot(heads, A).float() if len(pairs) else torch.zeros((0, A), device=device)
    selected = {}
    for level, by_window in (selections or {}).items():
        mask = torch.zeros((W, len(pairs)), device=device)
        for k, by_head in by_window.items():
            for _h, ps in by_head.items():
                mask[int(k), ps] = 1.0
        selected[level] = mask
    fires, hits = torch.zeros((W, len(pairs)), device=device), torch.zeros((W, len(pairs)), device=device)
    base_on, base_risk = torch.zeros((W, A), device=device), torch.zeros((W, A), device=device)
    union = {level: {"fires": torch.zeros(W, device=device), "hits": torch.zeros(W, device=device)} for level in selected}
    events, lead = [], {p: {} for p in range(len(pairs))}
    for video_id in tqdm(video_ids, desc="Rule statistics"):
        X = step05._video_from_cells(data["videos"][video_id], A, device)
        T, N, _A = X.shape
        if T <= memory:
            continue
        fired_dense = torch.zeros((T, N, len(pairs)), dtype=torch.bool, device=device)
        risk_dense = torch.zeros((T, N, A), dtype=torch.bool, device=device)
        for frames in torch.arange(memory, T, device=device).split(cfg["step05_cfg"]["frames_per_video"]):
            rows = step05.video_rows(model, X, frames, data["min_off_frames"])
            if len(rows["inputs"]) == 0:
                continue
            x, _k = model.split_inputs(rows["inputs"])
            fired = (x @ body.T >= size) & (size > 0)                                   # (B, P)
            t, n = rows["rows"].T
            fired_dense[t, n] = fired
            risk_dense[t, n] = rows["at_risk"]
            risk = (rows["at_risk"][:, None, :] & rows["known"]).float()               # (B, W, A)
            target = rows["targets"].float()
            base_on += target.sum(dim=0)
            base_risk += risk.sum(dim=0)
            f = fired.float()[:, None, :]                                                # (B, 1, P)
            fires += (f * risk[:, :, heads]).sum(dim=0)
            hits += (f * target[:, :, heads]).sum(dim=0)
            for level, mask in selected.items():
                # (B, W, A): some selected rule of the head and window fires.
                any_rule = (f * mask[None]) @ head_onehot > 0
                union[level]["fires"] += (any_rule * risk).sum(dim=(0, 2))
                union[level]["hits"] += (any_rule * target).sum(dim=(0, 2))
        # Exact onsets and the lead of the rules of their head that fired before them.
        onsets = step05.atom_onsets(X, data["min_off_frames"])
        onsets[0] = False
        onsets[1:] &= X[:-1].any(dim=2, keepdim=True)
        onsets[:memory] = False
        for t_e, n, h in torch.nonzero(onsets).tolist():
            e = len(events)
            events.append((video_id, t_e, h))
            ps = torch.nonzero(heads == h).flatten()
            if len(ps) == 0:
                continue
            t = t_e - torch.arange(top, device=device)
            ok = t >= memory
            t = t[ok]
            seen = fired_dense[t, n][:, ps] & risk_dense[t, n, h][:, None]               # (L, P_h)
            for i, p in enumerate(ps.tolist()):
                j = torch.nonzero(seen[:, i]).flatten()
                if len(j):
                    lead[p][e] = int(j.min())
    return {"fires": fires.cpu(), "hits": hits.cpu(), "base": (base_on / base_risk.clamp(min=1)).cpu(),
            "events": events, "lead": lead,
            "union": {level: {key: v.cpu() for key, v in u.items()} for level, u in union.items()}}


def select_rules(stats, pairs, windows, memory, level, cfg):
    """Greedy set cover per window and head: {k: {h: [p]}} of the rules with
    precision >= level and >= min_support hits, each adding the most uncovered events."""
    selection = {}
    precision = stats["hits"] / stats["fires"].clamp(min=1)
    for k, w in enumerate(windows):
        eligible = {e for e, (_v, t_e, _h) in enumerate(stats["events"]) if t_e - w + 1 >= memory}
        selection[k] = {}
        by_head = {}
        for p, (_r, h, _i) in enumerate(pairs):
            if precision[k, p] >= level and stats["hits"][k, p] >= cfg["min_support"]:
                covered = {e for e, j in stats["lead"][p].items() if j < w and e in eligible}
                by_head.setdefault(h, []).append((p, covered))
        for h, candidates in by_head.items():
            chosen, covered = [], set()
            while candidates:
                p, gain = max(((p, len(c - covered)) for p, c in candidates),
                              key=lambda item: (item[1], precision[k, item[0]].item()))
                if gain < cfg["min_new_events"]:
                    break
                chosen.append(p)
                covered |= next(c for q, c in candidates if q == p)
                candidates = [(q, c) for q, c in candidates if q != p]
            if chosen:
                selection[k][h] = chosen
    return selection


def set_metrics(stats, selection, windows, memory, union):
    """Per window: event recall of the rule set (an event is covered when a
    selected rule of its head fired in the W frames up to it), the row precision
    of its union, the rows it fires on and the rules it holds."""
    out = {}
    for k, w in enumerate(windows):
        chosen = selection.get(k, {})
        eligible = [e for e, (_v, t_e, _h) in enumerate(stats["events"]) if t_e - w + 1 >= memory]
        covered = sum(any(stats["lead"][p].get(e, w) < w for p in chosen.get(stats["events"][e][2], []))
                      for e in eligible)
        recall = covered / max(len(eligible), 1)
        precision = union["hits"][k].item() / max(union["fires"][k].item(), 1)
        f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
        out[str(w)] = {"events": len(eligible), "covered": covered, "recall": recall, "precision": precision,
                       "f1": f1, "rows_fired": int(union["fires"][k].item()),
                       "rules": sum(len(ps) for ps in chosen.values()), "heads": len(chosen)}
    return out


def main(input_data):
    print("----------------- Step 05c: Rule Set Selection -----------------")
    cfg = get_cfg(input_data)
    step05_cfg = cfg["step05_cfg"]
    if step05_cfg["main_only"]:
        print("[Skip]\t\t main_only: only the main rule net is trained")
        return
    pruned_file = step05_causal.get_cfg_from_step05(step05_cfg)["pruned_model_file"]
    source = pruned_file if os.path.exists(pruned_file) else step05_cfg["model_file"]
    if not os.path.exists(source):
        print(f"[Skip]\t\t no step 05 model {source}")
        return
    if not step05.is_fresh(cfg["selection_file"], source, step05_cfg["training_data_file"]):
        training_data = step05.load_training_data(step05_cfg)
        model = step05.load_temporal_rule_net(training_data, step05_cfg, source)
        pairs = candidate_rules(model, step05_cfg)
        memory = model.slot_size * model.num_slots
        print(f"[Rules]\t\t {len(pairs)} candidate rules (clauses) of {source.name}")
        train = rule_statistics(model, pairs, training_data, training_data["train_videos"], cfg)
        selections = {str(level): select_rules(train, pairs, model.windows, memory, level, cfg)
                      for level in cfg["precision_levels"]}
        val = rule_statistics(model, pairs, training_data, training_data["val_videos"], cfg, selections)
        levels = {}
        for level, selection in selections.items():
            levels[level] = {"val": set_metrics(val, selection, model.windows, memory, val["union"][level]),
                             "selection": {str(k): {str(h): ps for h, ps in by_head.items()}
                                           for k, by_head in selection.items()}}
        # Per window, the level with the best val F1.
        chosen = {str(w): max(levels, key=lambda level: levels[level]["val"][str(w)]["f1"]) for w in model.windows}
        utils_data.save_json({
            "source_model": str(source), "windows": model.windows, "memory": memory,
            "pairs": [{"rule": r, "head": h, "inputs": inputs, "slot_size": model.slot_size,
                       "bodies": [list(model.decode_input(m)) for m in inputs]} for r, h, inputs in pairs],
            "precision": {"train": (train["hits"] / train["fires"].clamp(min=1)).tolist(),
                          "val": (val["hits"] / val["fires"].clamp(min=1)).tolist()},
            "levels": levels, "chosen_level": chosen, "settings": {k: v for k, v in cfg.items() if k != "step05_cfg"
                                                                   and not hasattr(v, "parent")},
        }, cfg["selection_file"])
    print_summary(utils_data.load_json(cfg["selection_file"]), cfg)


def print_summary(selection, cfg):
    print(f"[Rule sets]\t {cfg['selection_file']}")
    print("  val: event recall / row precision / rules of the set, per precision level (columns) and window")
    levels = list(selection["levels"])
    print(f"  {'W':>5}" + "".join(f"{'p>=' + level:>22}" for level in levels) + "   chosen")
    for w in selection["windows"]:
        cells = []
        for level in levels:
            m = selection["levels"][level]["val"][str(w)]
            cells.append(f"R {m['recall']:.2f} P {m['precision']:.3f} n {m['rules']:>3}")
        print(f"  {w:>5}" + "".join(f"{c:>22}" for c in cells) + f"   p>={selection['chosen_level'][str(w)]}")


# -------------- Scoring (step 06) --------------
class RuleSetScorer:
    """(B, W, A) scores of packed inputs: for each window, the highest train
    precision of the selected rules of the head that fire (every body atom seen in
    the memory), 0 if none. selection: the step 05c file; level: per window the
    precision level of the set (default the val-chosen one)."""

    def __init__(self, selection, num_inputs, num_atoms, device, levels=None):
        windows = selection["windows"]
        precision = torch.tensor(selection["precision"]["train"])                      # (W, P)
        items = []                                                                      # (pair, window, head, precision)
        for k, w in enumerate(windows):
            level = (levels or selection["chosen_level"])[str(w)]
            for h, ps in selection["levels"][level]["selection"].get(str(k), {}).items():
                items += [(p, k, int(h), precision[k, p].item()) for p in ps]
        self.num_inputs, self.shape = num_inputs, (len(windows), num_atoms)
        pairs = selection["pairs"]
        self.body = torch.zeros((len(items), num_inputs), device=device)
        for i, (p, _k, _h, _prec) in enumerate(items):
            self.body[i, pairs[p]["inputs"]] = 1.0
        self.size = self.body.sum(dim=1)
        self.index = torch.tensor([k * num_atoms + h for _p, k, h, _prec in items], dtype=torch.long, device=device)
        self.value = torch.tensor([prec for _p, _k, _h, prec in items], device=device)

    def __call__(self, inputs):
        B = len(inputs)
        out = torch.zeros((B, self.shape[0] * self.shape[1]), device=inputs.device)
        if len(self.index):
            x = inputs[:, :self.num_inputs]
            fired = ((x @ self.body.T) >= self.size) & (self.size > 0)                     # (B, S)
            values = fired.float() * self.value
            out.scatter_reduce_(1, self.index.expand(B, -1), values, reduce="amax")
        return out.view(B, *self.shape)


@torch.no_grad()
def level_tradeoff(model, selection, data, cfg):
    """{level: {w: event recall, row precision, rules}} of every precision level's
    rule set on the given videos (step 06: the test videos)."""
    pairs = [(p["rule"], p["head"], p["inputs"]) for p in selection["pairs"]]
    selections = {level: {int(k): {int(h): ps for h, ps in by_head.items()}
                          for k, by_head in entry["selection"].items()}
                  for level, entry in selection["levels"].items()}
    stats = rule_statistics(model, pairs, data, list(data["videos"]), cfg, selections)
    return {level: set_metrics(stats, sel, model.windows, selection["memory"], stats["union"][level])
            for level, sel in selections.items()}
