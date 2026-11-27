"""Single rules at one event: does any rule on its own predict (or rule out) the onset? (2026-11-27)

For one row (video, agent, target frame t) and one head, each rule is judged alone, by its own rate of the onset
(P(onset within the head's window | the rule holds), at-risk rows of the head, row level), not by the fitted sum:
  1. Mined rules: every rule of the head in the run's rule-lab .rules.pt files whose FULL mined body (hard AND,
     the soft gates ignored) holds at the row; per rule its rate on train and val rows, its hit rows and its fitted
     weight.
  2. Any single literal / pair of literals true at the row (not only mined ones), the candidates: the rule
     literals and the derived ones, minus the literals true only because the agent was not in view in a slot
     (own / role scope of a slot with no atom of the agent: "!Car:Stop(own)@2" for a car not there 2 s ago).
     Positive: the highest rates; negative: the lowest (lift < --er_neg_lift), each with >= --er_min_hits hit
     rows (positive) or >= --er_min_rows rows (negative) on train.
Rows: the head's at-risk rows (`step05_mine._head_rows`) of --er_train train videos and of all odd val videos.

    ./run_roadpp_docker.sh --gpu 3 --event_rules -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --ev_video sumo_00473 --ev_agent 19 --ev_frame 57 --ev_head Car:Jun
Prints the tables; writes <output_dir>/rule_lab/event_rules_<video>_a<agent>_f<frame>_<head>.json / .md.
"""
import glob
import random

import torch
from tqdm import tqdm

from src.exp_roadpp import config, rule_lab, rule_paper, step05, step05_mine, utils_data, utils_pipe
from src.exp_roadpp.event_whatif import literal_bits


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--ev_video", type=str, required=True)
    parser.add_argument("--ev_agent", type=int, required=True)
    parser.add_argument("--ev_frame", type=int, required=True)
    parser.add_argument("--ev_head", type=str, required=True)
    parser.add_argument("--ev_rules", type=str, default="*_vt_p*", help="rule-lab result stem (rule_lab dir)")
    parser.add_argument("--er_scan", type=str, default="False",
                        help="False (default): only this event; the rules' own rates are their mining stats in the "
                             "rule-lab json. True: also measure rates of rules / literal pairs on train + val rows")
    parser.add_argument("--er_train", type=int, default=1500, help="--er_scan: train videos (random) for the rates")
    parser.add_argument("--er_min_hits", type=int, default=30, help="positive candidates: hit rows on train")
    parser.add_argument("--er_min_rows", type=int, default=300, help="negative candidates: rows on train")
    parser.add_argument("--er_neg_lift", type=float, default=0.5)
    parser.add_argument("--er_top", type=int, default=15)
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
    files = sorted(glob.glob(str(lab_dir / f"{args.ev_rules}.rules.pt")))
    sc = rule_paper.RuleScorer(files, model, derived, 1, device)
    h = names.index(args.ev_head)
    sel = torch.nonzero(sc.head == h).flatten()
    body, weight = sc.body[sel], sc.weight[sel]
    LT = model.num_literals + derived.size                       # the always-true literal (padding)
    possible = step05.class_possible(training_data, vocab, device)
    head_mask = training_data["target_heads"].to(device)
    S, w = model.slot_size, int(cfg.get("clause_slots", 1))

    # The event's row.
    with torch.no_grad():
        rows = step05._video_rows(model, training_data, args.ev_video, torch.tensor([args.ev_frame]), device)
        i = torch.nonzero(rows["rows"][:, 1] == args.ev_agent).flatten()
        if not len(i):
            raise ValueError(f"no row of agent {args.ev_agent} at frame {args.ev_frame} of {args.ev_video}")
        inp = rows["inputs"][int(i[0]):int(i[0]) + 1]
        lits_ev = rule_lab.literal_rows(model, inp, derived)[0]                     # as trained
        real, _u = literal_bits(model, derived, inp, 0.0, absent_unknown=True)      # absent slots unknown -> 0
        real = real[0] > 0.5
    cand = torch.nonzero(lits_ev[:LT] & real[:LT]).flatten()
    absent_only = torch.nonzero(lits_ev[:LT] & ~real[:LT]).flatten()
    hard = torch.ones_like(sc.gate[sel])
    holds = rule_lab._soft_act(lits_ev[None].float(), body, hard)[0] > 0.5
    rules_ev = torch.nonzero(holds).flatten()
    print(f"[Event rules] {args.ev_video} agent {args.ev_agent} f{args.ev_frame} {args.ev_head}: {len(cand)} literals "
          f"true ({len(absent_only)} more only because the agent was not in view), {len(rules_ev)} of {len(sel)} "
          f"mined rules hold (full bodies)")
    K = len(cand)
    stem = lab_dir / f"event_rules_{args.ev_video}_a{args.ev_agent}_f{args.ev_frame}_{args.ev_head.replace(':', '_')}"
    if args.er_scan != "True":
        single_event(args, lab_dir, stem, text, lits_ev, real, body, sc.gate[sel], weight, float(sc.bias[h]),
                     LT, absent_only)
        return

    # Rates on train and val rows.
    val_ids = list(training_data["val_videos"])[1::2]
    train_ids = list(training_data["train_videos"])
    random.Random(0).shuffle(train_ids)
    stats = {}
    for split, vids in (("train", train_ids[:args.er_train]), ("val", val_ids)):
        n_all = torch.zeros(K, K, device=device); n_hit = torch.zeros(K, K, device=device)
        r_all = torch.zeros(len(rules_ev), device=device); r_hit = torch.zeros(len(rules_ev), device=device)
        tot, tot_hit = 0, 0
        with torch.no_grad():
            for vid in tqdm(vids, desc=f"Rates ({split})"):
                can = possible(vid)
                for frames in training_data["frames"][vid].split(256):
                    rw = step05._video_rows(model, training_data, vid, frames, device)
                    if len(rw["inputs"]) == 0:
                        continue
                    keep, y = step05_mine._head_rows(rw, can, head_mask, S, w)
                    keep, y = keep[:, h], y[:, h]
                    if not keep.any():
                        continue
                    L = rule_lab.literal_rows(model, rw["inputs"][keep], derived)
                    yk = y[keep].float()
                    T = L[:, cand].float()
                    n_all += T.T @ T
                    n_hit += (T * yk[:, None]).T @ T
                    if len(rules_ev):
                        a = (rule_lab._soft_act(L.float(), body[rules_ev], hard[rules_ev]) > 0.5).float()
                        r_all += a.sum(0); r_hit += (a * yk[:, None]).sum(0)
                    tot += int(keep.sum()); tot_hit += int(yk.sum())
        stats[split] = {"all": n_all.cpu(), "hit": n_hit.cpu(), "r_all": r_all.cpu(),
                        "r_hit": r_hit.cpu(), "base": tot_hit / max(tot, 1), "rows": tot}
        print(f"[Event rules] {split}: {tot} at-risk rows, base rate {stats[split]['base']:.4f}")

    tr, va = stats["train"], stats["val"]
    rate = lambda s, a, b: float(s["hit"][a, b] / s["all"][a, b]) if s["all"][a, b] > 0 else float("nan")
    lit = lambda m: text(int(cand[m]))
    out = {"video": args.ev_video, "agent": args.ev_agent, "frame": args.ev_frame, "head": args.ev_head,
           "base_train": tr["base"], "base_val": va["base"],
           "absent_only_literals": [text(int(m)) for m in absent_only][:200]}
    md = [f"# {args.ev_head} at {args.ev_video} agent {args.ev_agent} f{args.ev_frame}", "",
          f"base rate train {tr['base']:.4f}, val {va['base']:.4f}; rate = P(onset within the window | the rule "
          f"holds), at-risk rows; lift = rate / base", ""]

    # 1. mined rules holding (full bodies)
    rows_m = []
    for j, r in enumerate(rules_ev.tolist()):
        lits_r = [text(m) for m in body[r].tolist() if m < LT]
        rt = float(tr["r_hit"][j] / tr["r_all"][j]) if tr["r_all"][j] > 0 else float("nan")
        rv = float(va["r_hit"][j] / va["r_all"][j]) if va["r_all"][j] > 0 else float("nan")
        rows_m.append({"rule": " & ".join(lits_r), "weight": float(weight[r]), "rate_train": rt, "rate_val": rv,
                       "hits_train": int(tr["r_hit"][j]), "rows_train": int(tr["r_all"][j]),
                       "lift_val": rv / max(va["base"], 1e-9)})
    rows_m.sort(key=lambda x: -(x["rate_val"] if x["rate_val"] == x["rate_val"] else -1))
    out["mined_rules_holding"] = rows_m
    md += ["## 1. Mined rules of the head that hold here (full mined body, gates ignored)", "",
           "| rule | fitted w | rate train | rate val | lift val | hit rows train / rows |", "|---|---|---|---|---|---|"]
    md += [f"| `{x['rule']}` | {x['weight']:+.2f} | {x['rate_train']:.3f} | {x['rate_val']:.3f} | {x['lift_val']:.1f} | "
           f"{x['hits_train']} / {x['rows_train']} |" for x in rows_m]

    # 2. single literals and pairs among the true literals
    cands = []
    for a in range(K):
        for b in range(a, K):
            if tr["all"][a, b] == 0:
                continue
            if b != a and (tr["all"][a, b] == tr["all"][a, a] or tr["all"][a, b] == tr["all"][b, b]):
                continue                                         # the pair = one of its literals
            name = lit(a) if a == b else f"{lit(a)} & {lit(b)}"
            cands.append({"rule": name, "size": 1 if a == b else 2, "rate_train": rate(tr, a, b),
                          "rate_val": rate(va, a, b), "hits_train": int(tr["hit"][a, b]), "rows_train": int(tr["all"][a, b]),
                          "hits_val": int(va["hit"][a, b]), "rows_val": int(va["all"][a, b])})
    for c in cands:
        c["lift_train"] = c["rate_train"] / max(tr["base"], 1e-9)
        c["lift_val"] = c["rate_val"] / max(va["base"], 1e-9) if c["rate_val"] == c["rate_val"] else float("nan")
    pos = sorted([c for c in cands if c["hits_train"] >= args.er_min_hits], key=lambda c: -c["rate_train"])
    neg = sorted([c for c in cands if c["rows_train"] >= args.er_min_rows and c["lift_train"] < args.er_neg_lift],
                 key=lambda c: c["rate_train"])
    for title, lst, key in (("2. Strongest single literals / pairs FOR the onset (true here)", pos, "for"),
                            ("3. Literals / pairs AGAINST the onset (true here, rate << base)", neg, "against")):
        top = [c for c in lst if c["size"] == 1][:args.er_top // 2] + [c for c in lst if c["size"] == 2][:args.er_top]
        out[key] = top
        md += ["", f"## {title}", "", "| literal(s) | rate train | lift train | rate val | lift val | hit rows / rows train | "
               "hit rows / rows val |", "|---|---|---|---|---|---|---|"]
        md += [f"| `{c['rule']}` | {c['rate_train']:.3f} | {c['lift_train']:.1f} | {c['rate_val']:.3f} | {c['lift_val']:.1f} | "
               f"{c['hits_train']} / {c['rows_train']} | {c['hits_val']} / {c['rows_val']} |" for c in top]
    md += ["", f"Literals true only because the agent was not in view ({len(absent_only)}, not candidates): "
           + ", ".join(f"`{text(int(m))}`" for m in absent_only[:40]) + (" ..." if len(absent_only) > 40 else "")]
    utils_data.save_json(out, f"{stem}.json")
    open(f"{stem}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[Event rules] {stem}.json / .md")


def single_event(args, lab_dir, stem, text, lits_ev, real, body, gate, weight, bias, LT, absent_only):
    """Only the event's row: every rule of the head that holds there, hard (full mined body) or soft (the fitted
    gates, activation >= 0.5), with its own evidence from mining (rule-lab json: train precision, train onsets,
    lift on the check videos) and its fitted weight / contribution; split into rules for (w > 0) and against
    (w < 0) the onset."""
    stats = {}
    for f in sorted(glob.glob(str(lab_dir / f"{args.ev_rules}.json"))):
        d = __import__("json").load(open(f))
        if args.ev_head not in d["per_head"]:
            continue
        v = (d.get("selected") or {}).get(args.ev_head) or next(iter(d["rules"]))
        for r in d["rules"][v]:
            if r["head"] == args.ev_head:
                stats[frozenset(r["literals"])] = r
    soft = rule_lab._soft_act(lits_ev[None].float(), body, gate)[0]
    hard = rule_lab._soft_act(lits_ev[None].float(), body, torch.ones_like(gate))[0]
    rows, total = [], {"positive": 0.0, "negative": 0.0}
    for r in torch.nonzero((hard > 0.5) | (soft >= 0.5)).flatten().tolist():
        lits = [text(m) for m in body[r].tolist() if m < LT]
        kept = [text(m) for m, g in zip(body[r].tolist(), gate[r].tolist()) if m < LT and g >= rule_lab.DROP_GATE]
        st = stats.get(frozenset(lits), {})
        # a literal of the rule that is true here only because the agent was not in view
        art = [text(m) for m in body[r].tolist() if m < LT and bool(lits_ev[m]) and not bool(real[m])]
        c = float(weight[r] * soft[r])
        rows.append({"rule": " & ".join(lits), "kept_literals": " & ".join(kept), "holds_full_body": bool(hard[r] > 0.5),
                     "activation": float(soft[r]), "weight": float(weight[r]), "contribution": c,
                     "train_precision": st.get("precision"), "train_onsets": st.get("events"),
                     "check_lift": st.get("check_lift"), "absent_artifact_literals": art})
    for x in rows:
        if abs(x["contribution"]) >= 0.0:
            total["positive" if x["contribution"] > 0 else "negative"] += x["contribution"]
    key = lambda x: -(x["train_precision"] or 0)
    pos = sorted([x for x in rows if x["weight"] > 0], key=key)
    neg = sorted([x for x in rows if x["weight"] < 0], key=key)
    logit = bias + total["positive"] + total["negative"]
    md = [f"# {args.ev_head} at {args.ev_video} agent {args.ev_agent} f{args.ev_frame} (this event only)", "",
          f"rules that hold here: {len(rows)} (full mined body or fitted activation >= 0.5). Score of the fitted sum: "
          f"bias {bias:+.2f} + for {total['positive']:+.2f} + against {total['negative']:+.2f} = logit {logit:+.2f}, "
          f"P {1 / (1 + __import__('math').exp(-logit)):.3f}", "",
          "precision / onsets: the rule's own train precision P(onset | rule) and distinct train onsets when mined; "
          "check lift: its lift on the check videos. full: the whole mined body holds; act: the fitted (gated) "
          "activation. absent: literals true only because the agent was not in view.", ""]
    for title, lst in (("Rules FOR the onset (fitted w > 0)", pos), ("Rules AGAINST the onset (fitted w < 0)", neg)):
        md += [f"## {title}: {len(lst)}", "",
               "| rule (mined body) | precision | onsets | check lift | full | act | w | w x act | absent |",
               "|---|---|---|---|---|---|---|---|---|"]
        md += [f"| `{x['rule']}` | {x['train_precision'] if x['train_precision'] is None else round(x['train_precision'], 3)} "
               f"| {x['train_onsets']} | {x['check_lift']} | {'yes' if x['holds_full_body'] else 'no'} | "
               f"{x['activation']:.2f} | {x['weight']:+.2f} | {x['contribution']:+.2f} | "
               f"{', '.join(x['absent_artifact_literals']) or '-'} |" for x in lst]
        md.append("")
    md += [f"Literals true only because the agent was not in view: {len(absent_only)}"]
    utils_data.save_json({"video": args.ev_video, "agent": args.ev_agent, "frame": args.ev_frame, "head": args.ev_head,
                          "bias": bias, "logit": logit, "for": pos, "against": neg}, f"{stem}.json")
    open(f"{stem}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[Event rules] {stem}.json / .md")


if __name__ == "__main__":
    main()
