"""Atom probe: why a fact does or does not make it into the rules (2026-11-23).

The gestalt v7 quick test found no Car:Jun rule with gst:mate_new_Jun, while its main rule was
"Car:IncomLane(own)@1" (precision 0.06, 1035 onsets). This probe answers three questions for one head and one
action of its group mates (--probe_head, default Car:Jun; the action is the head's, e.g. Jun).

1. Rules (--probe_rules, literal texts as the rule lab prints them, "&"-joined, ";"-separated). On every at-risk
   row of the head in the split's videos (no sample):
     - the rows where the rule holds and its hits (an onset within clause_slots slots, as in mining);
     - precision, the head's base rate and the lift;
     - the onsets the rule covers on some hit row, of all the head's onsets.
2. Timing. For every onset of the head (agent n, frame t):
     - its group mates of the MATE_LOOKBACK frames before (`gestalt.track_groups`: n and m in one tracked group);
     - the latest onset of the action among them before t.
   Reported:
     - the share of onsets with a mate onset within 20 / 50 / 100 / MATE_LOOKBACK frames;
     - the distribution of the lag;
     - the share with no mate at all.
   gst:mate_new_{a} needs a mate in the last gestalt.MATE_MEMORY frames and its onset in the last
   gestalt.NEW_WINDOW frames.
3. Group context at frame t - 1: in a group (moving or still), near one, or none; and for the onsets the mates
   had, their classes and whether they were still.

    ./run_roadpp_docker.sh --gpu 0 --atom_probe -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True \\
        --gestalt True --run_id 20261123-gestalt7_lab --probe_head Car:Jun \\
        --probe_rules "Car:IncomLane(own)@1;Car:IncomLane(own)@1 & gst:mate_new_Jun(own)@1"
Writes <output_dir>/atom_probe/probe_<head>.json.
"""
from collections import Counter

import torch
from tqdm import tqdm

from src.exp_roadpp import config, gestalt, rule_lab, step05, step05_mine, utils_data, utils_pipe

MATE_LOOKBACK = 150                     # frames before the onset to look for group mates and their onsets
LAG_BINS = (10, 20, 30, 50, 75, 100, 150)


@torch.no_grad()
def probe_rules(model, training_data, video_ids, head, rules, head_mask, possible, cfg):
    """Per rule (literal lists) the rows / hits where it holds and the onsets it covers; the head's rows, hits and
    onsets."""
    device, S, w = cfg["device"], model.slot_size, int(cfg.get("clause_slots", 1))
    L = model.num_literals
    body = rule_lab._pad(rules, L).to(device)
    fires, hits = torch.zeros(len(rules)), torch.zeros(len(rules))
    covered = [set() for _ in rules]
    rows_h, hits_h, onsets = 0, 0, set()
    for vi, video_id in enumerate(tqdm(video_ids, desc="Rules")):
        can = possible(video_id)
        for frames in training_data["frames"][video_id].split(int(cfg.get("mine_chunk", 256))):
            rows = step05._video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            keep, y = step05_mine._head_rows(rows, can, head_mask, S, w)
            k, yy = keep[:, head], y[:, head]
            if not k.any():
                continue
            rows_h += int(k.sum())
            hits_h += int(yy.sum())
            t, n = rows["rows"].T
            onset = t + rows["first"][:, head]
            ev = [(video_id, int(a), int(b)) for a, b in zip(n[yy].tolist(), onset[yy].tolist())]
            onsets.update(ev)
            lits = rule_lab._with_true(model.literal_values(rows["inputs"][k]) > 0.5)
            holds = rule_lab._holds_all(lits, body)                                 # (B_k, R)
            fires += holds.sum(dim=0).float().cpu()
            yk = yy[k]
            hits += (holds & yk[:, None]).sum(dim=0).float().cpu()
            ev_k = [(video_id, int(a), int(b)) for a, b in zip(n[k].tolist(), onset[k].tolist())]
            for r in range(len(rules)):
                idx = torch.nonzero(holds[:, r] & yk).flatten().tolist()
                covered[r].update(ev_k[i] for i in idx)
    return fires, hits, covered, rows_h, hits_h, onsets


def mate_timing(cfg, split, head_name, label_names, m_off):
    """Per onset of the head: the lag to the latest onset of the same action among its recent group mates, the
    group context in the frame before, and the mates' classes."""
    cls_head, _, act = head_name.partition(":")
    atoms = utils_data.load_json(cfg[f"{split}_atom_file"])
    boxes_all = torch.load(cfg[f"{split}_box_file"], map_location="cpu")
    A = len(label_names)
    a_head = label_names.index(head_name)
    a_act = [a for a, nm in enumerate(label_names) if nm.partition(":")[2] == act and not nm.startswith(("av:", "TL:"))]
    out = []
    for vid, video_atoms in tqdm(atoms.items(), desc="Timing"):
        X = step05.build_video_tensor(video_atoms, A)
        if X.shape[0] <= gestalt.K + 1:
            continue
        boxes, classes, has_box = step05._agent_boxes(boxes_all, vid, video_atoms, X)
        g = gestalt.groups(boxes, has_box, classes)
        tr = gestalt.track_groups(g, has_box)
        on = step05.atom_onsets(X, m_off)                                          # (T, N, A)
        on[0] = False
        act_on = on[..., a_act].any(dim=2)                                          # (T, N): an onset of the action
        T, N = X.shape[:2]
        Mf = tr["M"].float()
        co = torch.einsum("tip,tjp->tij", Mf, Mf) > 0                               # (T, i, j) in one track
        for t, n in torch.nonzero(on[:, :, a_head]).tolist():
            if t < 2 or str(classes[n]) != cls_head:
                continue
            lo = max(0, t - MATE_LOOKBACK)
            mates = co[lo:t, n].any(dim=0)
            mates[n] = False
            lag = None
            if bool(mates.any()):
                prior = act_on[lo:t][:, mates]                                        # (frames, mates)
                fr = torch.nonzero(prior.any(dim=1)).flatten()
                if len(fr):
                    lag = t - (lo + int(fr.max()))
            p_own, p_near = int(tr["own"][t - 1, n]), int(tr["near"][t - 1, n])
            if p_own >= 0:
                still = not bool((tr["M"][t - 1, :, p_own] & ~torch.isnan(g["angle"][t - 1])).any())
                ctx = "in_group_still" if still else "in_group_moving"
            else:
                ctx = "near_group" if p_near >= 0 else "no_group"
            out.append({"video": vid, "agent": n, "onset": t, "has_mate": bool(mates.any()), "lag": lag,
                        "context": ctx, "mate_classes": sorted({str(classes[j]) for j in torch.nonzero(mates).flatten().tolist()})})
    return out


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--probe_head", type=str, default="Car:Jun")
    parser.add_argument("--probe_rules", type=str,
                        default="Car:IncomLane(own)@1;Car:IncomLane(own)@1 & gst:mate_new_Jun(own)@1;"
                                "gst:mate_new_Jun(own)@1")
    parser.add_argument("--probe_split", type=str, default="val", choices=("train", "val"))
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    if not getattr(args, "run_id", None):
        raise ValueError("--run_id is needed (a run whose data has the probed atoms)")
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    training_data = step05.load_training_data(cfg)
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    head = names.index(args.probe_head)
    head_mask = training_data["target_heads"].to(device)
    possible = step05.class_possible(training_data, vocab, device)
    text_of = {step05_mine.literal_text(model, names, m): m for m in range(model.num_literals)}
    rule_texts = [r.strip() for r in args.probe_rules.split(";") if r.strip()]
    rules = []
    for rt in rule_texts:
        lits = [x.strip() for x in rt.split("&")]
        missing = [x for x in lits if x not in text_of]
        if missing:
            raise ValueError(f"unknown literal(s) {missing}; e.g. {list(text_of)[:5]}")
        rules.append([text_of[x] for x in lits])
    video_ids = list(training_data[f"{args.probe_split}_videos"])

    # 1. Rules on every at-risk row.
    fires, hits, covered, rows_h, hits_h, onsets = probe_rules(model, training_data, video_ids, head, rules,
                                                               head_mask, possible, cfg)
    base = hits_h / max(rows_h, 1)
    print(f"\n[atom_probe] {args.probe_head} on the {args.probe_split} videos: {rows_h} at-risk rows, {hits_h} hit rows "
          f"(base rate {base:.4f}), {len(onsets)} onsets")
    res_rules = []
    for rt, f, h, c in zip(rule_texts, fires.tolist(), hits.tolist(), covered):
        p = h / f if f else 0.0
        res_rules.append({"rule": rt, "rows": int(f), "hit_rows": int(h), "precision": p, "lift": p / base if base else None,
                          "onsets_covered": len(c), "onset_recall": len(c) / max(len(onsets), 1)})
        print(f"  {rt}\n      rows {int(f):8d}  hits {int(h):6d}  precision {p:.4f}  lift {p / base if base else 0:6.1f}  "
              f"onsets {len(c)} / {len(onsets)} ({len(c) / max(len(onsets), 1):.1%})")

    # 2. + 3. Timing and group context of the onsets.
    label_names = [step05._atom_name(sig) for sig in utils_data.load_json(cfg["unique_atoms_file"])]
    timing = mate_timing(cfg, args.probe_split, args.probe_head, label_names, int(cfg["min_off_frames"]))
    n = max(len(timing), 1)
    with_mate = [e for e in timing if e["has_mate"]]
    lags = [e["lag"] for e in timing if e["lag"] is not None]
    act = args.probe_head.partition(":")[2]
    print(f"\n  timing over {len(timing)} onsets of {args.probe_head} (group mates of the {MATE_LOOKBACK} frames before):")
    print(f"    with a group mate: {len(with_mate) / n:.1%}; a mate's {act} onset before it: {len(lags) / n:.1%}")
    for b in LAG_BINS:
        print(f"    mate's {act} onset within {b:3d} frames before: {sum(1 for x in lags if x <= b) / n:.1%}")
    if lags:
        q = sorted(lags)
        print(f"    lag (frames): median {q[len(q) // 2]}, 25% {q[len(q) // 4]}, 75% {q[3 * len(q) // 4]}; "
              f"mate_new_{act} covers <= {gestalt.NEW_WINDOW} frames after the mate's onset")
    ctx = Counter(e["context"] for e in timing)
    print("    group context in the frame before: " + ", ".join(f"{k} {v / n:.1%}" for k, v in ctx.most_common()))
    mc = Counter(c for e in with_mate for c in e["mate_classes"])
    print(f"    mates' classes: {dict(mc.most_common())}")
    out_dir = args.output_dir / "atom_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    utils_data.save_json({"head": args.probe_head, "split": args.probe_split, "rows": rows_h, "hit_rows": hits_h,
                          "base_rate": base, "onsets": len(onsets), "rules": res_rules,
                          "timing": {"onsets": len(timing), "with_mate": len(with_mate), "lags": lags,
                                     "context": dict(ctx), "mate_classes": dict(mc)},
                          "events": timing}, out_dir / f"probe_{args.probe_head.replace(':', '_')}.json")
    print(f"[atom_probe] -> {out_dir}")


if __name__ == "__main__":
    main()
