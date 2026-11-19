"""How many per-head rule nodes (`TemporalRuleNet.seed_rules_per_head`) can never become a clause?

Count only, no training. Plan: docs/pipeline/RULE_NET_PER_HEAD_PLAN.md, section 4.

A per-head rule node is (anchor body input m = (atom, scope, slot j), head h, copy c). Training only adds
body atoms (a soft AND) and the anchor input stays fixed, so the support of every rule grown from the node
is at most the support of the anchor input alone. `step05.extract_temporal_clauses` keeps a (rule, head)
clause only if support >= min_support training onsets in >= min_videos videos. So a node whose anchor
alone falls below that is discarded at extraction whatever it learns (as a positive rule; the inhibitors
"X -> not Y" are not covered by this bound).

The counts mirror `step05._rule_head_counts` at alignment f = 0 (every body in the memory, head within the
next slot) on the training videos, all their frames, with the at-risk / min_off_frames definition and the
learned heads of `step05.load_training_data`. For a body of one input m:
  fires[m, h] = rows where m holds and h is at risk (slot 0 observed),
  hits[m, h]  = the fires where h switches on in the next slot,
  videos[m, h] = videos with a hit.

Runs steps 03 and 04 first if their files are missing or stale (v2_self anchors), as the pipeline would.
    ./run_roadpp_docker.sh --gpu 0 --anchor_support -- --exp sumo_maps_v7 --machine dgx-jing
Optional: --run_id <run> (only names the step05_output folder the config makes; the counts use the shared
training data), --support_grid "5:2,10:2,20:3" (extra min_support:min_videos pairs to report).
Writes <output_dir>/anchor_support/anchor_support.json (summary + per-node counts) and prints a summary.
"""
import json
from collections import Counter, defaultdict

import torch
from tqdm import tqdm

from src.exp_roadpp import config, step03, step04, step05, utils_data, utils_pipe
from src.exp_roadpp.cauvid import TemporalRuleNet


@torch.no_grad()
def input_head_counts(model, training_data, video_ids, cfg):
    """(M, A) fires, hits and videos of every single rule input m as the whole body, head h, at f = 0."""
    device = cfg["device"]
    M, A, S = model.num_inputs, model.num_atoms, model.slot_size
    fires = torch.zeros((M, A), device=device)
    hits = torch.zeros((M, A), device=device)
    videos = torch.zeros((M, A), device=device)
    onsets = torch.zeros(A, device=device)
    at_risk_rows = torch.zeros(A, device=device)
    # The column each rule input reads for a head in slot f = 0 (`_rule_head_counts.in_memory`); M: unknown.
    read = model.shift_index[0].to(device)
    usable = (model.input_offset > 0).to(device)             # always true for f = 0; kept for exactness
    for video_id in tqdm(video_ids, desc="Counting anchor inputs (train)"):
        video_hits = torch.zeros((M, A), device=device)
        for frames in training_data["frames"][video_id].split(cfg["frames_per_video"]):
            rows = step05._video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            x, _k = model.split_inputs(rows["inputs"])
            x = torch.cat([x, x.new_zeros((len(x), 1))], dim=1)[:, read] * usable     # (B, M)
            first, at_risk = rows["first"], rows["at_risk"]
            risk = (at_risk & (first >= 0) & rows["known"][:, 0]).float()             # (B, A)
            onset = (at_risk & (first >= 0) & (first < S)).float()
            held = (x >= 1).float()
            fires += held.T @ risk
            video_hits += held.T @ onset
            onsets += onset.sum(dim=0)
            at_risk_rows += risk.sum(dim=0)
        hits += video_hits
        videos += video_hits > 0
    return {"fires": fires.cpu(), "hits": hits.cpu(), "videos": videos.cpu(),
            "onsets": onsets.cpu(), "at_risk": at_risk_rows.cpu()}


def per_head_nodes(model, anchors, heads):
    """[(body (atom, scope), anchor index, head, j, input m)] of `seed_rules_per_head`, in its order."""
    head_set = set(heads)
    nodes = []
    for i, ((atom, slot), anchor_heads) in enumerate(anchors):
        for h in sorted(head_set.intersection(int(x) for x in anchor_heads)):
            for j in range(1, model.num_slots + 1):
                nodes.append(((atom, min(slot, 1)), i, h, j, model.input_index(atom, min(slot, 1), j)))
    return nodes


def summarize(nodes, counts, heads, min_support, min_videos, copies, free_per_head):
    """Kept / removed node counts under one (min_support, min_videos)."""
    kept = [n for n in nodes
            if counts["hits"][n[4], n[2]] >= min_support and counts["videos"][n[4], n[2]] >= min_videos]
    kept_pairs = {(n[1], n[2]) for n in kept}
    all_pairs = {(n[1], n[2]) for n in nodes}
    heads_with_seed = {h for _i, h in kept_pairs}
    by_scope, by_slot = Counter(), Counter()
    by_scope_all, by_slot_all = Counter(), Counter()
    for n in nodes:
        by_scope_all[n[0][1]] += 1
        by_slot_all[n[3]] += 1
    for n in kept:
        by_scope[n[0][1]] += 1
        by_slot[n[3]] += 1
    free = len(heads) * free_per_head
    return {
        "min_support": min_support, "min_videos": min_videos,
        "seed_slots_total": len(nodes), "seed_slots_kept": len(kept),
        "rule_nodes_total": len(nodes) * copies + free, "rule_nodes_kept": len(kept) * copies + free,
        "pairs_total": len(all_pairs), "pairs_kept": len(kept_pairs),
        "heads_learned": len(heads), "heads_with_kept_seed": len(heads_with_seed),
        "heads_free_rules_only": sorted(set(heads) - heads_with_seed),
        "kept_by_scope": {("own" if s == 0 else "other"): f"{by_scope[s]} / {by_scope_all[s]}"
                          for s in sorted(by_scope_all)},
        "kept_by_slot": {f"j={j}": f"{by_slot[j]} / {by_slot_all[j]}" for j in sorted(by_slot_all)},
    }


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--support_grid", type=str, default="5:2,10:2,20:3,50:5",
                        help="extra min_support:min_videos pairs to report (the step05 one is always first)")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id if getattr(args, "run_id", None) \
        else args.output_dir / "anchor_support"
    # Anchors as the pipeline would build them (cached; rebuilt if stale or not v2_self).
    step03.main(config.get_step_03_input(args))
    step04.main_v2(config.get_step_04_input(args))

    cfg = step05.get_cfg(config.get_step_05_input(args))
    training_data = step05.load_training_data(cfg)
    vocab = step05.load_vocabulary(cfg)
    anchors = step05.anchor_rules(cfg)
    heads = torch.nonzero(training_data["target_heads"]).flatten().tolist()
    # A one-node model: only its input layout and `training_rows` are used.
    model = TemporalRuleNet(training_data["num_atoms"], cfg["slot_size"], cfg["num_slots"], 1, cfg["init_log_alpha"],
                            cfg["mask_type"], cfg["windows"], roles=cfg["roles"],
                            input_mask=step05.rule_input_mask(training_data, cfg, 2 + len(cfg["roles"])),
                            require_self=cfg.get("require_self", False),
                            ranges=cfg.get("slot_ranges", False)).to(cfg["device"])
    copies, free_per_head = cfg["rule_copies"], cfg["free_rules_per_head"]
    nodes = per_head_nodes(model, anchors, heads)
    expected = TemporalRuleNet.per_head_rule_count(anchors, heads, cfg["num_slots"], copies, free_per_head)
    if len(nodes) * copies + len(heads) * free_per_head != expected:
        raise RuntimeError(f"node count {len(nodes) * copies + len(heads) * free_per_head} != "
                           f"per_head_rule_count {expected}")

    counts = input_head_counts(model, training_data, training_data["train_videos"], cfg)

    grid = [(cfg["min_support"], cfg["min_videos"])]
    for item in args.support_grid.split(","):
        if item.strip():
            s, v = (int(x) for x in item.split(":"))
            if (s, v) not in grid:
                grid.append((s, v))
    summaries = [summarize(nodes, counts, heads, s, v, copies, free_per_head) for s, v in grid]

    # Per-node detail (one row per (anchor body, head, j); the copies share it).
    names = [step05._atom_name(sig) for sig in vocab]
    rows = []
    for (atom, scope), i, h, j, m in nodes:
        hits, fires = int(counts["hits"][m, h]), int(counts["fires"][m, h])
        rows.append({"anchor": i, "body": names[atom], "body_atom": atom, "scope": "own" if scope == 0 else "other",
                     "j": j, "head": names[h], "head_atom": h, "support": hits, "fires": fires,
                     "videos": int(counts["videos"][m, h]),
                     "confidence": hits / fires if fires else None,
                     "base_rate": (float(counts["onsets"][h] / counts["at_risk"][h])
                                   if counts["at_risk"][h] > 0 else None)})
    # Support distribution of the nodes (how many sit at 0, below the threshold, ...).
    bins = [0, 1, 5, 10, 20, 50, 100, 1000]
    hist = defaultdict(int)
    for r in rows:
        label = next((f"<{b}" for b in bins[1:] if r["support"] < b), f">={bins[-1]}")
        hist[label] += 1

    out_dir = args.output_dir / "anchor_support"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "anchor_support.json"
    result = {"exp": args.exp, "num_train_videos": len(training_data["train_videos"]),
              "num_anchor_bodies": len(anchors), "num_slots": cfg["num_slots"], "rule_copies": copies,
              "free_rules_per_head": free_per_head, "support_hist": dict(hist), "summaries": summaries,
              "nodes": rows}
    utils_data.save_json(result, out_file)

    print(f"\n[anchor_support] {args.exp}: {len(anchors)} anchor bodies, {len(heads)} learned heads, "
          f"{len(training_data['train_videos'])} train videos")
    print(f"  (body, head, slot) seeds: {len(nodes)}  x {copies} copies + {len(heads) * free_per_head} free "
          f"= {expected} rule nodes")
    print(f"  support of the anchor alone, per seed: {dict(hist)}")
    for s in summaries:
        print(f"  min_support {s['min_support']:>3} min_videos {s['min_videos']:>2}: "
              f"nodes {s['rule_nodes_kept']:>6} / {s['rule_nodes_total']}, seeds {s['seed_slots_kept']} / "
              f"{s['seed_slots_total']}, (body, head) pairs {s['pairs_kept']} / {s['pairs_total']}, "
              f"heads with a seed {s['heads_with_kept_seed']} / {s['heads_learned']}")
        print(f"      by scope {s['kept_by_scope']}, by slot {s['kept_by_slot']}")
    print(f"  -> {out_file}")


if __name__ == "__main__":
    main()
