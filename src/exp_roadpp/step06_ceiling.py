"""Step 06c: situation rates: how often the onset follows a situation, and how plausible the false alarms are.

An onset that follows a situation only sometimes (a reversing car stops within 2 s in ~10% of the
cases) cannot be predicted for sure from that situation: an alarm there is a good prediction that
fails by chance. To tell such alarms from wrong ones, the test rows are grouped by their context,
the observation the prediction is made from, at four levels (finest first):
  "inputs": every model input (the agent's own and the other agents' atoms, all slots): what
      the rule net and the MLP see; rows of the same context get the same score from any such model;
  "self": the agent's own inputs, all slots (own labels and, with the grid, own position);
  "self_now": the agent's own inputs of the last slot (its last slot_size frames);
  "self_now_labels": the agent's own label atoms of the last slot (e.g. "a car reversing in the lane").
The context rate of a row is the share of the rows of its context in which the onset followed
within W, counted over the *other test videos* only (leave one video out): consecutive frames of one
agent share the context and the outcome, so the row's own video would leak its outcome. A context
not seen in another video gets the head's base rate. Where its contexts recur, the context rate is
the chance of the onset in that situation. It bounds what a model of these inputs can do only where
they recur: on medium (30 test videos) the full inputs never recur in another video and the own
inputs of all slots in 21% of the rows, so at these levels it is a weak lookup baseline, not a
ceiling; the coarse "self_now" levels recur and give the situation rates. Reported:
  - context AP: the pooled AP (counted heads, as `step06` rows) of the context rate, vs the models'
    (and on the "covered" rows, whose context has >= min_group rows in other videos);
  - at each calibrated probability level q: the model's alarm rows (calibrated score >= q) vs as
    many rows taken by the context rate: precision of both, and of the model's false alarm rows
    (no onset) the share whose context rate is >= plausible_rate (the situation often leads to the
    event: a plausible alarm) and their mean context rate.
Rows are as in `step06._score_videos` (at risk, the window observed, the agent not leaving); this
is row-level (one row per agent and frame), not episodes.
"""
import os
import torch
from tqdm import tqdm
from src.exp_roadpp import step05, step06, utils_data

LEVELS = ("inputs", "self", "self_now", "self_now_labels")


def _default_cfg():
    return {
        "windows": [10, 20, 50, 100],            # (those of the model) frames
        "min_group": 10,                         # a context is "covered" with this many rows in other videos
        "probability_levels": [0.05, 0.1, 0.2, 0.3],
        "plausible_rate": 0.05,                  # a false alarm row is plausible if its context rate is >= this
        "metric": "ceiling_v2_leave_one_video_out_self_now",
    }


def _context_keys(inputs, columns, weights):
    """(B, 2) long hash of the 0/1 inputs[:, columns]: two projections with integer weights < 2^26
    in float64 (exact sums, < 2^53), so equal rows get equal keys and distinct ones collide with
    chance ~2^-52."""
    x = inputs[:, columns].double()
    return (x @ weights.to(x.device)).long().cpu()


def collect(model, mlp, data, cfg, ce, calibration=None):
    """Per valid row (agent, frame) of the test videos: the context keys per level, the video,
    the outcome code (W, H) int8 (0 not valid, 1 no onset, 2 onset within W) of the ceiling windows
    and counted heads, and the model scores (H,) of the headline window (calibrated if given)."""
    device, A = cfg["device"], data["num_atoms"]
    head_type = cfg["head_types"].to(device)
    counted = [step06.EVENT_TYPES.index(t) for t in step06.COUNTED_TYPES]
    heads = torch.nonzero(torch.isin(head_type, torch.tensor(counted, device=device))).flatten()
    windows = [w for w in ce["windows"] if w in model.windows]
    w_index = torch.tensor([list(model.windows).index(w) for w in windows], device=device)
    k_head = list(model.windows).index(cfg["headline_window"])
    window_sizes = torch.tensor(model.windows, device=device)
    M = model.num_inputs
    own = model.num_slots * model.num_atoms                                     # scope 0 comes first
    # Inputs (B, 2M): the true part, then the known part; each (scopes, slots, atoms), the last slot first.
    labels = data.get("num_label_atoms", A)
    columns = {"inputs": torch.arange(2 * M, device=device),
               "self": torch.cat([torch.arange(own, device=device), M + torch.arange(own, device=device)]),
               "self_now": torch.cat([torch.arange(A, device=device), M + torch.arange(A, device=device)]),
               "self_now_labels": torch.cat([torch.arange(labels, device=device),
                                             M + torch.arange(labels, device=device)])}
    generator = torch.Generator().manual_seed(0)
    weights = {lv: torch.randint(0, 2 ** 26, (len(c), 2), generator=generator).double() for lv, c in columns.items()}
    scorers = {"model": model} | ({"mlp": mlp} if mlp is not None else {})
    out = {"keys": {lv: [] for lv in LEVELS}, "video": [], "code": [], "score": {name: [] for name in scorers}}
    with torch.no_grad():
        for v, video_id in enumerate(tqdm(data["videos"], desc="Situation rates: contexts")):
            video = data["videos"][video_id]
            X = step05._video_from_cells(video, A, device)
            roles = step05.video_roles(video, device)
            T, N, _ = X.shape
            gone_before = torch.cat([torch.zeros((1, N), dtype=torch.long, device=device),
                                     (~X.any(dim=2)).long().cumsum(dim=0)])
            for begin in range(cfg["min_context"], T, cfg["frames_per_video"]):
                frames = torch.arange(begin, min(begin + cfg["frames_per_video"], T), device=device)
                rows = step05.video_rows(model, X, frames, data["min_off_frames"], None, roles)
                if len(rows["rows"]) == 0:
                    continue
                t, n = rows["rows"].T
                valid = rows["at_risk"][:, None, :] & rows["known"]
                if cfg.get("exclude_agent_leaving"):
                    valid &= step06._agent_observed(gone_before, t, n, window_sizes, rows["targets"])
                valid, targets = valid[:, w_index][:, :, heads], rows["targets"][:, w_index][:, :, heads]
                keep = valid.any(dim=(1, 2))
                if not keep.any():
                    continue
                code = torch.where(valid, 1 + targets.to(torch.int8), 0).to(torch.int8)[keep]
                inputs = rows["inputs"][keep]
                for lv in LEVELS:
                    out["keys"][lv].append(_context_keys(inputs, columns[lv], weights[lv]))
                out["video"].append(torch.full((int(keep.sum()),), v, dtype=torch.long))
                out["code"].append(code.cpu())
                for name, net in scorers.items():
                    p = net.window_probs(net(inputs)[0])
                    if calibration is not None and name in calibration:
                        p = step06.calibrate(p, calibration[name])
                    out["score"][name].append(p[:, k_head, heads].half().cpu())
    return {"keys": {lv: torch.cat(k) for lv, k in out["keys"].items()}, "video": torch.cat(out["video"]),
            "code": torch.cat(out["code"]), "score": {name: torch.cat(s) for name, s in out["score"].items()},
            "windows": windows, "calibrated": calibration is not None}


def _average_precision(scores, labels):
    """AP of 1-D scores and bool labels, rows of equal score one tie group."""
    scores, order = scores.float().sort(descending=True)
    labels = labels[order].float()
    groups = torch.unique_consecutive(scores, return_inverse=True)[1]
    G = int(groups.max()) + 1 if len(groups) else 0
    pos = torch.zeros(G).index_add_(0, groups, labels)
    count = torch.zeros(G).index_add_(0, groups, torch.ones_like(labels))
    tp, seen = pos.cumsum(0), count.cumsum(0)
    return ((tp / seen.clamp(min=1)) * pos).sum().item() / max(pos.sum().item(), 1)


def _top_k_precision(scores, labels, k):
    """Precision of the k rows with the highest scores (the tie group at the cut counted pro rata)."""
    if k <= 0:
        return None
    scores, order = scores.float().sort(descending=True)
    labels = labels[order].float()
    cut = scores[k - 1]
    above = scores > cut
    tie = scores == cut
    rest = k - int(above.sum())
    tie_rate = labels[tie].mean().item() if tie.any() else 0.0
    return (labels[above].sum().item() + rest * tie_rate) / k


def context_rates(keys, video, code):
    """(R, H) leave-one-video-out context rate of every row and head (one window's (R, H) outcome
    code), and (R, H) the number of rows of its context in the other videos."""
    _, g = torch.unique(keys, dim=0, return_inverse=True)
    _, gv = torch.unique(torch.stack([g, video], dim=1), dim=0, return_inverse=True)
    valid, pos = (code > 0).float(), (code == 2).float()
    G, GV, H = int(g.max()) + 1, int(gv.max()) + 1, code.shape[1]
    n_g = torch.zeros(G, H).index_add_(0, g, valid)
    p_g = torch.zeros(G, H).index_add_(0, g, pos)
    n_gv = torch.zeros(GV, H).index_add_(0, gv, valid)
    p_gv = torch.zeros(GV, H).index_add_(0, gv, pos)
    n_other, p_other = n_g[g] - n_gv[gv], p_g[g] - p_gv[gv]
    base = pos.sum(0) / valid.sum(0).clamp(min=1)                                 # (H,)
    rate = torch.where(n_other > 0, p_other / n_other.clamp(min=1), base[None, :])
    return rate, n_other


def evaluate(collected, cfg, ce):
    windows, code_all = collected["windows"], collected["code"]
    headline = windows.index(cfg["headline_window"])
    out = {**ce, "calibrated": collected["calibrated"], "headline_window": cfg["headline_window"],
           "num_rows": len(code_all), "levels": {}}
    for lv in LEVELS:
        keys = collected["keys"][lv]
        res = {"num_contexts": int(torch.unique(keys, dim=0).shape[0]), "windows": {}}
        for k, w in enumerate(windows):
            code = code_all[:, k]
            valid, y = code > 0, code == 2
            rate, n_other = context_rates(keys, collected["video"], code)
            entry = {"cells": int(valid.sum()), "positive_rate": y[valid].float().mean().item(),
                     "seen_in_other_videos": (n_other[valid] > 0).float().mean().item(),
                     "covered": (n_other[valid] >= ce["min_group"]).float().mean().item(),
                     "ceiling_ap": _average_precision(rate[valid], y[valid])}
            covered = valid & (n_other >= ce["min_group"])
            entry["ceiling_ap_covered"] = _average_precision(rate[covered], y[covered]) if covered.any() else None
            if k == headline:
                entry["models"] = {}
                for name, score in collected["score"].items():
                    s = score.float()
                    m = {"ap": _average_precision(s[valid], y[valid]),
                         "ap_covered": _average_precision(s[covered], y[covered]) if covered.any() else None,
                         "levels": {}}
                    # Alarm rows at each probability level (meaningful with calibrated scores).
                    for q in ce["probability_levels"]:
                        alarm = valid & (s >= q)
                        K = int(alarm.sum())
                        false = alarm & ~y
                        fr = rate[false]
                        m["levels"][str(q)] = {
                            "alarm_rows": K,
                            "precision": y[alarm].float().mean().item() if K else None,
                            "ceiling_precision_same_rows": _top_k_precision(rate[valid], y[valid], K),
                            "mean_context_rate_of_alarms": rate[alarm].mean().item() if K else None,
                            "false_alarm_rows": int(false.sum()),
                            "false_plausible_share": (fr >= ce["plausible_rate"]).float().mean().item()
                            if len(fr) else None,
                            "false_mean_context_rate": fr.mean().item() if len(fr) else None,
                            "false_unseen_context_share": (n_other[false] == 0).float().mean().item()
                            if len(fr) else None,
                        }
                    entry["models"][name] = m
            res["windows"][str(w)] = entry
        out["levels"][lv] = res
    return out


def _fmt(v, f="{:.3f}"):
    return "-" if v is None else f.format(v)


def print_summary(metrics, frame_rate):
    print(f"[Situation rates] test rows grouped by context; context rate = onset share in the other videos' rows of the "
          f"same context (leave one video out); {metrics['num_rows']} rows; model scores "
          f"{'calibrated on val' if metrics['calibrated'] else 'raw'}")
    w = str(metrics["headline_window"])
    for lv, res in metrics["levels"].items():
        print(f"  context = {lv} ({res['num_contexts']} distinct)")
        print(f"  {'window':>8}{'cells':>11}{'pos rate':>10}{'seen':>7}{'covered':>9}{'ctx AP':>9}{'ctx AP cov':>12}")
        for win, e in res["windows"].items():
            print(f"  {int(win) / frame_rate:>7g}s{e['cells']:>11}{e['positive_rate']:>10.4f}"
                  f"{e['seen_in_other_videos']:>7.2f}{e['covered']:>9.2f}{e['ceiling_ap']:>9.4f}"
                  f"{_fmt(e['ceiling_ap_covered'], '{:.4f}'):>12}")
        e = res["windows"][w]
        for name, m in e.get("models", {}).items():
            print(f"    {name} (window {int(w) / frame_rate:g} s): AP {m['ap']:.4f} vs context {e['ceiling_ap']:.4f}; "
                  f"covered rows AP {_fmt(m['ap_covered'], '{:.4f}')} vs {_fmt(e['ceiling_ap_covered'], '{:.4f}')}")
            print(f"      {'p >=':>6}{'alarms':>9}{'prec':>7}{'ctx prec':>10}{'ctx rate':>9}"
                  f"{'false':>8}{'plausible':>10}{'false ctx':>10}{'unseen':>8}")
            for q, r in m["levels"].items():
                print(f"      {q:>6}{r['alarm_rows']:>9}{_fmt(r['precision']):>7}"
                      f"{_fmt(r['ceiling_precision_same_rows']):>10}{_fmt(r['mean_context_rate_of_alarms']):>9}"
                      f"{r['false_alarm_rows']:>8}{_fmt(r['false_plausible_share'], '{:.2f}'):>10}"
                      f"{_fmt(r['false_mean_context_rate']):>10}{_fmt(r['false_unseen_context_share'], '{:.2f}'):>8}")
    print(f"  (ctx prec: precision of as many rows taken by the context rate; plausible: false alarm rows whose "
          f"context rate >= {metrics['plausible_rate']}; unseen: their context not in another video)\n")


def main(model, mlp, test_data, step05_cfg, cfg, output_file, calibration=None):
    """The ceiling metrics of the test videos (cached in output_file, redone when the models,
    the test data, the calibration or the settings change)."""
    ce = _default_cfg()
    metrics = utils_data.load_json(output_file) if os.path.exists(output_file) else {}
    sources = [step05_cfg["model_file"], step05_cfg["mlp_model_file"], cfg["test_data_file"]]
    if calibration is not None:
        sources.append(cfg["calibration_file"])
    if (not step05.is_fresh(output_file, *sources) or {k: metrics.get(k) for k in ce} != ce
            or metrics.get("calibrated") != (calibration is not None)):
        collected = collect(model, mlp, test_data, cfg, ce, calibration)
        metrics = evaluate(collected, cfg, ce)
        utils_data.save_json(metrics, output_file)
    print_summary(metrics, cfg["frame_rate"])
    print(f"[Results]\t {output_file}")
    return metrics
