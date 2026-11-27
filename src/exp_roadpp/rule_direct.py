"""Direct prediction by precise rules (2026-11-27, the user: a precise rule that holds predicts the onset directly;
the precision threshold is learned). No fit, no weights: the mined rules of one head with their own train precision
(rule-lab json), the full mined body as a hard AND (a rule that does not hold gives nothing).

Score of a row: the highest train precision of the head's rules that hold there (no rule holds: none).
Predict "onset" if the score >= tau. Rows: the at-risk rows of the head (`step05_mine._head_rows`); a hit row has the
onset within clause_slots slots; an event is detected if one of its hit rows is predicted.
tau is learned on the check videos (even val videos; the rules were mined on train), then applied to the odd val
videos and the test videos:
  tau_prec   the lowest tau whose alarms on the check videos are right at least --rd_target of the time;
  tau_f1     the tau of the best F1 of the alarm rows on the check videos.
Per split and tau: alarms, their precision, share of non-onset rows alarmed (FA share), event recall; for the chosen
taus also the fitted rule model (rule-lab soft scorer, `rule_paper.RuleScorer`) at the same FA share; and the given
event (--rd_event video:agent:onset frame): its score, the rule behind it, detected or not.

    ./run_roadpp_docker.sh --gpu 3 --rule_direct -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --rd_head Car:Jun --rd_event sumo_00473:19:57
Writes <output_dir>/rule_lab/direct_<head>.json.
"""
import glob
import json
from collections import defaultdict

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


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--rd_head", type=str, default="Car:Jun")
    parser.add_argument("--rd_rules", type=str, default="*_vt_p*", help="rule-lab result stem (rule_lab dir)")
    parser.add_argument("--rd_target", type=float, default=0.5, help="tau_prec: alarm precision on the check videos")
    parser.add_argument("--rd_min_onsets", type=int, default=5, help="rules with fewer train onsets are left out")
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

    # The head's mined rules with their own train precision.
    stats = rule_stats(lab_dir, args.rd_rules, args.rd_head)
    sel = torch.nonzero(sc.head == h).flatten()
    keep_r, prec, info = [], [], []
    for r in sel.tolist():
        lits = [text(m) for m in sc.body[r].tolist() if m < LT]
        st = stats.get(frozenset(lits))
        if st is None or (st.get("events") or 0) < args.rd_min_onsets:
            continue
        keep_r.append(r)
        prec.append(float(st["precision"]))
        info.append({"rule": " & ".join(lits), "train_precision": st["precision"], "train_onsets": st["events"],
                     "check_lift": st.get("check_lift"), "fitted_weight": float(sc.weight[r])})
    body = sc.body[torch.tensor(keep_r, dtype=torch.long, device=device)]
    prec = torch.tensor(prec, device=device)
    print(f"[Rule direct] {args.rd_head}: {len(keep_r)} of {len(sel)} rules with stats and >= {args.rd_min_onsets} "
          f"train onsets; precision >= 0.5: {int((prec >= 0.5).sum())}, >= 0.8: {int((prec >= 0.8).sum())}")

    head_mask = training_data["target_heads"].to(device)
    possible = step05.class_possible(training_data, vocab, device)
    S, w = model.slot_size, int(cfg.get("clause_slots", 1))
    val = sorted(training_data["val_videos"])
    splits = {"check (even val)": val[0::2], "odd val": val[1::2],
              "test": sorted(rule_lab.add_test_videos(training_data, cfg))}
    ev_vid, ev_agent, ev_onset = args.rd_event.split(":")
    if args.rd_videos:
        # the first videos of each split, plus the event's video in its own split
        splits = {k: v[:args.rd_videos] + ([ev_vid] if ev_vid in v[args.rd_videos:] else [])
                  for k, v in splits.items()}
    ev_key = (ev_vid, int(ev_agent), int(ev_onset))
    NB = step07_paper.NB
    res = {}
    with torch.no_grad():
        for name, vids in splits.items():
            acc = {"pos": torch.zeros(NBIN), "neg": torch.zeros(NBIN), "none_pos": 0, "none_neg": 0,
                   "soft_neg": torch.zeros(NB)}
            events, soft_ev, why = {}, {}, {}
            for vid in tqdm(vids, desc=f"Direct rules ({name})"):
                can = possible(vid)
                for frames in training_data["frames"][vid].split(128):
                    rows = step05._video_rows(model, training_data, vid, frames, device)
                    if len(rows["inputs"]) == 0:
                        continue
                    keep, y = step05_mine._head_rows(rows, can, head_mask, S, w)
                    keep, y = keep[:, h], y[:, h]
                    if not keep.any():
                        continue
                    inp = rows["inputs"][keep]
                    lits = rule_lab.literal_rows(model, inp, derived)
                    holds = rule_lab._holds_all(lits, body)                          # (B, R) hard AND
                    pr = torch.where(holds, prec[None], torch.full_like(holds, -1.0, dtype=torch.float))
                    best, arg = pr.max(dim=1) if len(keep_r) else (torch.full((len(inp),), -1.0, device=device), None)
                    score = best.clamp(min=0)
                    yk = y[keep]
                    bins = (score * (NBIN - 1)).round().long().cpu()
                    acc["pos"] += torch.bincount(bins[yk.cpu()], minlength=NBIN).float()
                    acc["neg"] += torch.bincount(bins[~yk.cpu()], minlength=NBIN).float()
                    acc["none_pos"] += int(((best < 0) & yk).sum())
                    acc["none_neg"] += int(((best < 0) & ~yk).sum())
                    soft = sc(inp)[:, 0, h]
                    acc["soft_neg"] += torch.bincount(step07_paper._logit_bins(soft[~yk]).cpu(), minlength=NB).float()
                    if yk.any():
                        rk = rows["rows"][keep]
                        for i in torch.nonzero(yk).flatten().tolist():
                            t, n = int(rk[i, 0]), int(rk[i, 1])
                            key = (vid, n, t + int(rows["first"][keep][i, h]))
                            s = float(score[i])
                            if s > events.get(key, -1.0):
                                events[key] = s
                                why[key] = info[int(arg[i])]["rule"] if float(best[i]) >= 0 else None
                            soft_ev[key] = max(soft_ev.get(key, -1.0), float(soft[i]))
            tab = table(acc, events, len(events))
            res[name] = {"videos": len(vids), "events": len(events), "hit_rows": int(acc["pos"].sum()),
                         "non_onset_rows": int(acc["neg"].sum()), "hit_rows_no_rule": acc["none_pos"],
                         "non_onset_rows_no_rule": acc["none_neg"], "table": tab,
                         "_acc": acc, "_events": events, "_soft_ev": soft_ev, "_why": why}

    # tau learned on the check videos
    ck = res["check (even val)"]["table"]
    ok = [t for t in TAUS if ck[t]["alarm_precision"] >= args.rd_target and ck[t]["alarms"] > 0]
    taus = {"tau_prec": min(ok) if ok else 1.0, "tau_f1": max(TAUS, key=lambda t: ck[t]["f1"])}
    out = {"head": args.rd_head, "rules": len(keep_r), "target": args.rd_target, "taus": taus, "splits": {},
           "event": None}
    print(f"\n[Rule direct] {args.rd_head}, tau learned on the check videos: tau_prec {taus['tau_prec']:.2f} "
          f"(alarm precision >= {args.rd_target}), tau_f1 {taus['tau_f1']:.2f}")
    for name, r in res.items():
        print(f"\n {name}: {r['videos']} videos, {r['events']} events, {r['hit_rows']} hit rows "
              f"({r['hit_rows_no_rule']} with no rule holding), {r['non_onset_rows']} non-onset rows")
        print(f"   {'tau':>5}{'alarms':>9}{'alarm prec':>11}{'FA share':>10}{'row rec':>9}{'events':>8}{'ev rec':>8}")
        for t in TAUS:
            x = r["table"][t]
            mark = " <- " + ",".join(k for k, v in taus.items() if v == t) if t in taus.values() else ""
            print(f"   {t:>5.2f}{x['alarms']:>9}{x['alarm_precision']:>11.3f}{x['fa_share']:>10.4f}"
                  f"{x['row_recall']:>9.3f}{x['events_detected']:>8}{x['event_recall']:>8.3f}{mark}")
        # the fitted rule model at the same FA share
        cmp = {}
        for k, t in taus.items():
            fa = r["table"][t]["fa_share"]
            thr_bin = int(lane_clips.thresholds(r["_acc"]["soft_neg"][None], fa)[0])
            thr = lane_clips.bin_score(thr_bin)
            det = sum(1 for s in r["_soft_ev"].values() if s >= thr)
            cmp[k] = {"fa_share": fa, "fitted_threshold": thr, "fitted_events_detected": det,
                      "direct_events_detected": r["table"][t]["events_detected"]}
            print(f"   {k}: direct {r['table'][t]['events_detected']} events vs fitted rule model "
                  f"{det} events at the same FA share {fa:.4f} (its threshold {thr:.3f})")
        out["splits"][name] = {k: v for k, v in r.items() if not k.startswith("_")}
        out["splits"][name]["table"] = {str(t): v for t, v in r["table"].items()}
        out["splits"][name]["fitted_same_fa"] = cmp
        if ev_key in r["_events"]:
            s = r["_events"][ev_key]
            out["event"] = {"split": name, "score": s, "rule": r["_why"][ev_key],
                            "detected": {k: s >= t for k, t in taus.items()}, "fitted_score": r["_soft_ev"][ev_key]}
    if out["event"]:
        e = out["event"]
        print(f"\n[Rule direct] event {args.rd_event} ({e['split']}): direct score {e['score']:.3f} by "
              f"'{e['rule']}'; detected {e['detected']}; fitted rule model {e['fitted_score']:.3f}")
    else:
        print(f"\n[Rule direct] event {args.rd_event}: not found among the hit rows")
    # the precise rules themselves (by precision)
    out["precise_rules"] = sorted([x for x in info if x["train_precision"] >= 0.5],
                                  key=lambda x: -x["train_precision"])
    print(f"\n[Rule direct] rules with train precision >= 0.5 ({len(out['precise_rules'])}):")
    for x in out["precise_rules"][:30]:
        print(f"   prec {x['train_precision']:.3f} onsets {x['train_onsets']:>4} lift {x['check_lift']} "
              f"w {x['fitted_weight']:+.2f}  {x['rule']}")
    path = lab_dir / f"direct_{args.rd_head.replace(':', '_')}.json"
    utils_data.save_json(out, path)
    print(f"\n[Rule direct] {path}")


if __name__ == "__main__":
    main()
