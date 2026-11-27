"""Direct prediction by precise rules (2026-11-27, the user: a precise rule that holds predicts the onset directly;
its precision is the one on val, the threshold is learned). No fit, no weights: the head's mined rules (train), the
full mined body as a hard AND (a rule that does not hold gives nothing).

Three splits:
  train  the rules were mined there (rule lab); their train precision is not used;
  val    every rule's precision on the val rows: hit rows / rows where it holds, as the Wilson lower bound (95%) over
         n_eff = holding rows / slot_size (consecutive frames of an agent are not independent); a rule needs
         >= --rd_min_val_onsets distinct val onsets among its hit rows. tau is chosen on val too:
           tau_prec  the lowest tau whose alarms on val are right at least --rd_target of the time;
           tau_f1    the tau of the best F1 of the alarm rows on val;
  test   the check: alarms, their precision, FA share, event recall at every tau; and the fitted rule model
         (rule-lab soft scorer, `rule_paper.RuleScorer`) at the same FA share.
Score of a row: the highest val precision (lower bound) of the qualifying rules that hold there (none: no alarm).
Rows: the at-risk rows of the head (`step05_mine._head_rows`); a hit row has the onset within clause_slots slots;
an event is detected if one of its hit rows is predicted. The given event (--rd_event video:agent:onset frame):
its score, rule, detected or not, in its split.

    ./run_roadpp_docker.sh --gpu 3 --rule_direct -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --rd_head Car:Jun --rd_event sumo_00473:19:57 --rd_videos 100
Writes <output_dir>/rule_lab/direct_<head>.json.
"""
import glob
import json

import torch
from tqdm import tqdm

from src.exp_roadpp import (config, lane_clips, rule_lab, rule_paper, step05, step05_mine, step07_paper, utils_data,
                            utils_pipe)

NBIN = 101                      # precision bins 0.00 .. 1.00 (bin 0 also: no rule holds)
TAUS = [round(0.05 * i, 2) for i in range(2, 21)]       # 0.10 .. 1.00


def rule_stats(lab_dir, stem, head):
    """{frozenset of literal texts: the rule's mining stats} of the head's chosen variant in the rule-lab json."""
    stats = {}
    for f in sorted(glob.glob(str(lab_dir / f"{stem}.json"))):
        d = json.load(open(f))
        if head not in d.get("per_head", {}):
            continue
        v = (d.get("selected") or {}).get(head) or next(iter(d["rules"]))
        for r in d["rules"][v]:
            if r["head"] == head:
                stats[frozenset(r["literals"])] = r
    return stats


def table(split, events, n_events):
    """Per tau: alarms, right alarms, FA share, event recall from the row histograms and the event scores."""
    pos, neg = split["pos"], split["neg"]
    out = {}
    for tau in TAUS:
        b = int(round(tau * (NBIN - 1)))
        tp, fp = float(pos[b:].sum()), float(neg[b:].sum())
        det = sum(1 for s in events.values() if s >= tau)
        rec_rows = tp / max(float(pos.sum()), 1)
        prec = tp / max(tp + fp, 1)
        out[tau] = {"alarms": int(tp + fp), "alarm_precision": prec, "fa_share": fp / max(float(neg.sum()), 1),
                    "row_recall": rec_rows, "f1": 2 * prec * rec_rows / max(prec + rec_rows, 1e-9),
                    "events_detected": det, "event_recall": det / max(n_events, 1)}
    return out


def head_rows(model, training_data, vids, h, possible, head_mask, S, w, device, desc):
    """Per frame chunk of the videos: (vid, inputs, y, rows (t, n), frames to the onset) of the head's at-risk rows."""
    for vid in tqdm(vids, desc=desc):
        can = possible(vid)
        for frames in training_data["frames"][vid].split(128):
            rows = step05._video_rows(model, training_data, vid, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            keep, y = step05_mine._head_rows(rows, can, head_mask, S, w)
            keep, y = keep[:, h], y[:, h]
            if keep.any():
                yield vid, rows["inputs"][keep], y[keep], rows["rows"][keep], rows["first"][keep][:, h]


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--rd_head", type=str, default="Car:Jun")
    parser.add_argument("--rd_rules", type=str, default="*_vt_p*", help="rule-lab result stem (rule_lab dir)")
    parser.add_argument("--rd_target", type=float, default=0.5, help="tau_prec: alarm precision on val")
    parser.add_argument("--rd_min_val_onsets", type=int, default=3, help="distinct val onsets a rule needs")
    parser.add_argument("--rd_event", type=str, default="sumo_00473:19:57", help="video:agent:onset frame")
    parser.add_argument("--rd_videos", type=int, default=0, help="videos per split at most (0: all)")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    lab_dir = args.output_dir / "rule_lab"
    training_data = step05.load_training_data(cfg)
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    derived = rule_lab.Derived(model, names)
    ext = rule_lab.ExtModel(model, derived, True)
    text = lambda m: step05_mine.literal_text(ext, names, m)
    files = sorted(glob.glob(str(lab_dir / f"{args.rd_rules}.rules.pt")))
    sc = rule_paper.RuleScorer(files, model, derived, 1, device)
    h = names.index(args.rd_head)
    LT = model.num_literals + derived.size                       # the always-true literal (padding)
    stats = rule_stats(lab_dir, args.rd_rules, args.rd_head)     # train stats, only reported
    sel = torch.nonzero(sc.head == h).flatten()
    body = sc.body[sel]
    R = len(sel)
    texts = [" & ".join(text(m) for m in body[r].tolist() if m < LT) for r in range(R)]

    head_mask = training_data["target_heads"].to(device)
    possible = step05.class_possible(training_data, vocab, device)
    S, w = model.slot_size, int(cfg.get("clause_slots", 1))
    ev_vid, ev_agent, ev_onset = args.rd_event.split(":")
    ev_key = (ev_vid, int(ev_agent), int(ev_onset))
    splits = {"val": sorted(training_data["val_videos"]), "test": sorted(rule_lab.add_test_videos(training_data, cfg))}
    if args.rd_videos:
        splits = {k: v[:args.rd_videos] + ([ev_vid] if ev_vid in v[args.rd_videos:] else [])
                  for k, v in splits.items()}
    it = lambda name: head_rows(model, training_data, splits[name], h, possible, head_mask, S, w, device,
                                f"{name}")

    # 1. Every rule's precision on val.
    n_hold, n_hit, pairs, ev_ids = torch.zeros(R, device=device), torch.zeros(R, device=device), [], {}
    with torch.no_grad():
        for vid, inp, y, rk, first in it("val"):
            holds = rule_lab._holds_all(rule_lab.literal_rows(model, inp, derived), body)     # (B, R)
            n_hold += holds.sum(0)
            n_hit += holds[y].sum(0)
            for i in torch.nonzero(y).flatten().tolist():
                key = (vid, int(rk[i, 1]), int(rk[i, 0]) + int(first[i]))
                e = ev_ids.setdefault(key, len(ev_ids))
                rs = torch.nonzero(holds[i]).flatten().cpu()
                pairs.append(torch.stack([rs, torch.full_like(rs, e)], 1))
    pr = torch.cat(pairs).unique(dim=0) if pairs else torch.zeros((0, 2), dtype=torch.long)
    val_onsets = torch.bincount(pr[:, 0], minlength=R).to(device)
    p_val = n_hit / n_hold.clamp(min=1)
    lb = rule_lab.wilson_lb(p_val, n_hold / S)
    ok = (val_onsets >= args.rd_min_val_onsets) & (n_hold > 0)
    score_of = torch.where(ok, lb, torch.full_like(lb, -1.0))
    info = []
    for r in range(R):
        st = stats.get(frozenset(t for t in texts[r].split(" & ") if t), {})
        info.append({"rule": texts[r], "val_precision": float(p_val[r]), "val_lb": float(lb[r]),
                     "val_rows": int(n_hold[r]), "val_hit_rows": int(n_hit[r]), "val_onsets": int(val_onsets[r]),
                     "train_precision": st.get("precision"), "train_onsets": st.get("events"),
                     "fitted_weight": float(sc.weight[sel[r]])})
    print(f"[Rule direct] {args.rd_head}: {R} rules; on val ({len(splits['val'])} videos, {len(ev_ids)} onsets): "
          f"{int(ok.sum())} with >= {args.rd_min_val_onsets} val onsets; val lower bound >= 0.3: "
          f"{int((score_of >= 0.3).sum())}, >= 0.5: {int((score_of >= 0.5).sum())}")

    # 2. Scores on val (tau) and test (the check).
    NB = step07_paper.NB
    res = {}
    with torch.no_grad():
        for name in splits:
            acc = {"pos": torch.zeros(NBIN), "neg": torch.zeros(NBIN), "soft_neg": torch.zeros(NB),
                   "none_pos": 0, "none_neg": 0}
            events, soft_ev, why = {}, {}, {}
            for vid, inp, y, rk, first in it(name):
                holds = rule_lab._holds_all(rule_lab.literal_rows(model, inp, derived), body)
                best, arg = torch.where(holds, score_of[None], torch.full_like(lb, -1.0)[None]).max(dim=1)
                score = best.clamp(min=0)
                yc = y.cpu()
                bins = (score * (NBIN - 1)).round().long().cpu()
                acc["pos"] += torch.bincount(bins[yc], minlength=NBIN).float()
                acc["neg"] += torch.bincount(bins[~yc], minlength=NBIN).float()
                acc["none_pos"] += int(((best < 0) & y).sum())
                acc["none_neg"] += int(((best < 0) & ~y).sum())
                soft = sc(inp)[:, 0, h]
                acc["soft_neg"] += torch.bincount(step07_paper._logit_bins(soft[~y]).cpu(), minlength=NB).float()
                for i in torch.nonzero(y).flatten().tolist():
                    key = (vid, int(rk[i, 1]), int(rk[i, 0]) + int(first[i]))
                    s = float(score[i])
                    if s > events.get(key, -1.0):
                        events[key] = s
                        why[key] = info[int(arg[i])] if float(best[i]) >= 0 else None
                    soft_ev[key] = max(soft_ev.get(key, -1.0), float(soft[i]))
            res[name] = {"videos": len(splits[name]), "events": len(events), "hit_rows": int(acc["pos"].sum()),
                         "non_onset_rows": int(acc["neg"].sum()), "hit_rows_no_rule": acc["none_pos"],
                         "non_onset_rows_no_rule": acc["none_neg"], "table": table(acc, events, len(events)),
                         "_acc": acc, "_events": events, "_soft_ev": soft_ev, "_why": why}

    vt = res["val"]["table"]
    good = [t for t in TAUS if vt[t]["alarm_precision"] >= args.rd_target and vt[t]["alarms"] > 0]
    taus = {"tau_prec": min(good) if good else None, "tau_f1": max(TAUS, key=lambda t: vt[t]["f1"])}
    out = {"head": args.rd_head, "rules": R, "qualifying_rules": int(ok.sum()), "target": args.rd_target,
           "min_val_onsets": args.rd_min_val_onsets, "taus": taus, "splits": {}, "event": None}
    print(f"\n[Rule direct] tau on val: tau_prec {taus['tau_prec']} (alarm precision >= {args.rd_target}), "
          f"tau_f1 {taus['tau_f1']:.2f}")
    for name, r in res.items():
        print(f"\n {name}: {r['videos']} videos, {r['events']} events, {r['hit_rows']} hit rows "
              f"({r['hit_rows_no_rule']} with no rule), {r['non_onset_rows']} non-onset rows "
              f"({r['non_onset_rows_no_rule']} with no rule)")
        print(f"   {'tau':>5}{'alarms':>9}{'alarm prec':>11}{'FA share':>10}{'row rec':>9}{'events':>8}{'ev rec':>8}")
        for t in TAUS:
            x = r["table"][t]
            mark = " <- " + ",".join(k for k, v in taus.items() if v == t) if t in taus.values() else ""
            print(f"   {t:>5.2f}{x['alarms']:>9}{x['alarm_precision']:>11.3f}{x['fa_share']:>10.4f}"
                  f"{x['row_recall']:>9.3f}{x['events_detected']:>8}{x['event_recall']:>8.3f}{mark}")
        cmp = {}
        for k, t in taus.items():
            if t is None:
                continue
            fa = r["table"][t]["fa_share"]
            thr = lane_clips.bin_score(int(lane_clips.thresholds(r["_acc"]["soft_neg"][None], fa)[0]))
            det = sum(1 for s in r["_soft_ev"].values() if s >= thr)
            cmp[k] = {"fa_share": fa, "fitted_threshold": thr, "fitted_events_detected": det,
                      "direct_events_detected": r["table"][t]["events_detected"]}
            print(f"   {k}: direct {r['table'][t]['events_detected']} events vs fitted rule model {det} events "
                  f"at the same FA share {fa:.4f} (its threshold {thr:.3f})")
        out["splits"][name] = {k: v for k, v in r.items() if not k.startswith("_")}
        out["splits"][name]["table"] = {str(t): v for t, v in r["table"].items()}
        out["splits"][name]["fitted_same_fa"] = cmp
        if ev_key in r["_events"]:
            s = r["_events"][ev_key]
            out["event"] = {"split": name, "score": s, "rule": r["_why"][ev_key],
                            "detected": {k: (t is not None and s >= t) for k, t in taus.items()},
                            "fitted_score": r["_soft_ev"][ev_key]}
    if out["event"]:
        e = out["event"]
        print(f"\n[Rule direct] event {args.rd_event} ({e['split']}): score {e['score']:.3f}, detected "
              f"{e['detected']}, fitted rule model {e['fitted_score']:.3f}; rule: {e['rule']}")
    else:
        print(f"\n[Rule direct] event {args.rd_event}: not among the hit rows")
    key_rule = "Car:MovTow(own)@1 & !Car:IncomLane(own)@1"
    out["rules_of_note"] = [x for x in info if x["rule"] == key_rule or (x["train_precision"] or 0) >= 0.5]
    out["precise_on_val"] = sorted([x for x, q in zip(info, ok.tolist()) if q and x["val_lb"] >= 0.3],
                                   key=lambda x: -x["val_lb"])
    print(f"\n[Rule direct] rules with val lower bound >= 0.3 ({len(out['precise_on_val'])}):")
    for x in out["precise_on_val"][:30]:
        print(f"   val {x['val_precision']:.3f} (lb {x['val_lb']:.3f}, {x['val_onsets']} onsets, {x['val_rows']} rows)"
              f"  train {x['train_precision']}  w {x['fitted_weight']:+.2f}  {x['rule']}")
    print("\n[Rule direct] train precision >= 0.5 (or the event's rule) on val:")
    for x in out["rules_of_note"]:
        print(f"   train {x['train_precision']} ({x['train_onsets']} onsets) -> val {x['val_precision']:.3f} "
              f"({x['val_hit_rows']}/{x['val_rows']} rows, {x['val_onsets']} onsets)  {x['rule']}")
    path = lab_dir / f"direct_{args.rd_head.replace(':', '_')}.json"
    utils_data.save_json(out, path)
    print(f"\n[Rule direct] {path}")


if __name__ == "__main__":
    main()
