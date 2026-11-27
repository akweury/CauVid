"""Direct prediction by precise rules (2026-11-27, the user: a precise rule that holds predicts the onset directly;
its precision is the one on val, the threshold is learned). No fit, no weights: the head's mined rules (train), the
full mined body as a hard AND (a rule that does not hold gives nothing).

Three splits:
  train  the rules were mined there (rule lab, on a row sample); here counted on every row of --rd_train_videos
         train videos;
  val    counted the same way. A rule's precision: hit rows / rows where it holds; its independent units: episodes
         (distinct agents where it holds). The user's rule (2026-11-27): a rule must keep a similar precision on val.
           too_few       < --rd_min_episodes episodes on train or on val: dropped;
           inconsistent  enough episodes, the train and val precisions differ (two-proportion z over episodes,
                         |z| > 1.96): dropped, listed for the analysis (val lower / higher, videos, the share of
                         the val rows in one video);
           consistent    used; its precision is the val one (Wilson lower bound over the val episodes).
         tau is chosen on val too:
           tau_prec  the lowest tau whose alarms on val are right at least --rd_target of the time;
           tau_f1    the tau of the best F1 of the alarm rows on val;
  test   the check: alarms, their precision, FA share, event recall at every tau; and the fitted rule model
         (rule-lab soft scorer, `rule_paper.RuleScorer`) at the same FA share.
Score of a row: the highest val precision (lower bound) of the consistent rules that hold there (none: no alarm).
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


def count_rules(gen, body, R, model, derived, device):
    """Per rule over the rows of gen: rows and hit rows where it holds, distinct onsets among its hit rows, episodes
    (distinct (video, agent) where it holds: the independent units), videos, and the most rows from one video."""
    z = lambda: torch.zeros(R, device=device)
    out = {"rows": z(), "hit_rows": z(), "episodes": z(), "videos": z(), "max_video_rows": z()}
    pairs, ev_ids = [], {}
    cur, agents, vrows = None, None, None

    def flush():
        if agents is not None:
            out["episodes"] += agents.sum(0)
            out["videos"] += (vrows > 0).float()
            torch.maximum(out["max_video_rows"], vrows, out=out["max_video_rows"])

    for vid, inp, y, rk, first in gen:
        if vid != cur:
            flush()
            cur, agents, vrows = vid, torch.zeros((256, R), device=device), z()
        holds = rule_lab._holds_all(rule_lab.literal_rows(model, inp, derived), body)          # (B, R)
        hf = holds.float()
        out["rows"] += hf.sum(0)
        out["hit_rows"] += hf[y].sum(0)
        vrows += hf.sum(0)
        n = rk[:, 1].long()
        if int(n.max()) >= len(agents):
            agents = torch.cat([agents, torch.zeros((int(n.max()) + 1 - len(agents), R), device=device)])
        agents.index_add_(0, n, hf).clamp_(max=1)
        for i in torch.nonzero(y).flatten().tolist():
            e = ev_ids.setdefault((vid, int(rk[i, 1]), int(rk[i, 0]) + int(first[i])), len(ev_ids))
            rs = torch.nonzero(holds[i]).flatten().cpu()
            pairs.append(torch.stack([rs, torch.full_like(rs, e)], 1))
    flush()
    pr = torch.cat(pairs).unique(dim=0) if pairs else torch.zeros((0, 2), dtype=torch.long)
    out["onsets"] = torch.bincount(pr[:, 0], minlength=R).to(device)
    out["num_onsets"] = len(ev_ids)
    return out


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--rd_head", type=str, default="Car:Jun")
    parser.add_argument("--rd_rules", type=str, default="*_vt_p*", help="rule-lab result stem (rule_lab dir)")
    parser.add_argument("--rd_target", type=float, default=0.5, help="tau_prec: alarm precision on val")
    parser.add_argument("--rd_min_episodes", type=int, default=5,
                        help="a rule needs >= this many distinct agents holding it on train and on val (else too few)")
    parser.add_argument("--rd_train_videos", type=int, default=300, help="train videos counted (the first ones)")
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
    # test: every test video, or with --toy the toy test videos (`rule_lab.eval_videos`)
    splits = {"val": sorted(training_data["val_videos"]), "test": sorted(rule_lab.eval_videos(training_data, cfg, "test"))}
    if args.rd_videos:
        splits = {k: v[:args.rd_videos] + ([ev_vid] if ev_vid in v[args.rd_videos:] else [])
                  for k, v in splits.items()}
    splits["train"] = sorted(training_data["train_videos"])[:args.rd_train_videos or None]
    it = lambda name: head_rows(model, training_data, splits[name], h, possible, head_mask, S, w, device,
                                f"{name}")

    # 1. Every rule on train (all rows of --rd_train_videos videos, no sample) and on val.
    cnt = {name: count_rules(it(name), body, R, model, derived, device) for name in ("train", "val")}
    tr, va = cnt["train"], cnt["val"]
    p_tr = tr["hit_rows"] / tr["rows"].clamp(min=1)
    p_va = va["hit_rows"] / va["rows"].clamp(min=1)
    n_tr, n_va = tr["episodes"].float(), va["episodes"].float()
    few = (n_tr < args.rd_min_episodes) | (n_va < args.rd_min_episodes)
    pool = (p_tr * n_tr + p_va * n_va) / (n_tr + n_va).clamp(min=1)
    se = (pool * (1 - pool) * (1 / n_tr.clamp(min=1) + 1 / n_va.clamp(min=1))).sqrt().clamp(min=1e-9)
    z = (p_va - p_tr) / se
    differ = ~few & (z.abs() > rule_lab.WILSON_Z)
    ok = ~few & ~differ
    lb = rule_lab.wilson_lb(p_va, n_va)
    score_of = torch.where(ok, lb, torch.full_like(lb, -1.0))
    status = ["too_few" if bool(few[r]) else ("inconsistent" if bool(differ[r]) else "consistent") for r in range(R)]
    info = []
    for r in range(R):
        st = stats.get(frozenset(t for t in texts[r].split(" & ") if t), {})
        info.append({"rule": texts[r], "status": status[r], "z": float(z[r]),
                     "train_precision": float(p_tr[r]), "train_rows": int(tr["rows"][r]),
                     "train_episodes": int(n_tr[r]), "train_onsets": int(tr["onsets"][r]),
                     "train_videos": int(tr["videos"][r]),
                     "val_precision": float(p_va[r]), "val_lb": float(lb[r]), "val_rows": int(va["rows"][r]),
                     "val_episodes": int(n_va[r]), "val_onsets": int(va["onsets"][r]), "val_videos": int(va["videos"][r]),
                     "val_top_video_share": float(va["max_video_rows"][r] / va["rows"][r].clamp(min=1)),
                     "train_top_video_share": float(tr["max_video_rows"][r] / tr["rows"][r].clamp(min=1)),
                     "sample_precision": st.get("precision"), "sample_onsets": st.get("events"),
                     "fitted_weight": float(sc.weight[sel[r]])})
    n_st = {k: status.count(k) for k in ("consistent", "inconsistent", "too_few")}
    print(f"[Rule direct] {args.rd_head}: {R} rules on {len(splits['train'])} train / {len(splits['val'])} val videos "
          f"(episodes: distinct agents where the rule holds; >= {args.rd_min_episodes} on both needed): {n_st}")
    print(f"   consistent rules, val lower bound >= 0.3: {int((score_of >= 0.3).sum())}, >= 0.5: "
          f"{int((score_of >= 0.5).sum())}")

    # 2. Scores on val (tau) and test (the check).
    NB = step07_paper.NB
    res = {}
    with torch.no_grad():
        for name in ("val", "test"):
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
    out = {"head": args.rd_head, "rules": R, "status_counts": n_st, "target": args.rd_target,
           "min_episodes": args.rd_min_episodes, "train_videos": len(splits["train"]),
           "onsets": {k: cnt[k]["num_onsets"] for k in cnt}, "taus": taus, "splits": {}, "event": None}
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
    row = lambda x: (f"train {x['train_precision']:.3f} ({x['train_onsets']} onsets, {x['train_episodes']} agents, "
                     f"{x['train_videos']} videos) | val {x['val_precision']:.3f} ({x['val_onsets']} onsets, "
                     f"{x['val_episodes']} agents, {x['val_videos']} videos, top video {x['val_top_video_share']:.0%}) "
                     f"| z {x['z']:+.1f} | sample {x['sample_precision'] if x['sample_precision'] is None else round(x['sample_precision'], 3)} "
                     f"| w {x['fitted_weight']:+.2f} | {x['rule']}")
    out["rules_of_note"] = [x for x in info if x["rule"] == key_rule or (x["sample_precision"] or 0) >= 0.5
                            or x["train_precision"] >= 0.5]
    out["consistent_precise"] = sorted([x for x in info if x["status"] == "consistent" and x["val_lb"] >= 0.3],
                                       key=lambda x: -x["val_lb"])
    out["inconsistent"] = sorted([x for x in info if x["status"] == "inconsistent"], key=lambda x: -abs(x["z"]))
    print(f"\n[Rule direct] consistent rules with val lower bound >= 0.3 ({len(out['consistent_precise'])}):")
    for x in out["consistent_precise"][:20]:
        print("   " + row(x))
    lower = sum(1 for x in out["inconsistent"] if x["z"] < 0)
    print(f"\n[Rule direct] inconsistent rules ({len(out['inconsistent'])}: val lower {lower}, val higher "
          f"{len(out['inconsistent']) - lower}), the 20 strongest:")
    for x in out["inconsistent"][:20]:
        print("   " + row(x))
    print("\n[Rule direct] sample precision >= 0.5, full train precision >= 0.5, or the event's rule:")
    for x in out["rules_of_note"]:
        print(f"   [{x['status']}] " + row(x))
    path = lab_dir / f"direct_{args.rd_head.replace(':', '_')}.json"
    utils_data.save_json(out, path)
    print(f"\n[Rule direct] {path}")


if __name__ == "__main__":
    main()
