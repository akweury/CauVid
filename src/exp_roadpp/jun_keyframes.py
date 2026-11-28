"""Keyframes of a head: its onsets, or the failures of a hand rule; one file per frame (2026-11-28).

--kf_mode onsets    --kf_num random onsets of --kf_head; frames at --kf_times s from the onset.
--kf_mode failures  --kf_num random failures of --kf_rule (`hand_rules` syntax), the timing ignored (2026-11-28, the
                    user: does the predicted event happen at all?): rows where the rule holds and the agent never
                    does the head afterwards (`pair_table.fp_classes` "never" and "after": it did it only before),
                    one per agent; a row whose agent does it later at any time counts as right. Frames at
                    --kf_fail_times s from the predicting frame; the title gives the agent's next actions (its own
                    class's labels switching on within the horizon). The camera view (right) shows only
                    the agent's box and the traffic lights; the top view (left) as always.
--kf_mode missed    --kf_num random onsets of the head the rule misses (it holds on none of the onset's hit rows);
                    frames at --kf_times s from the onset; the title gives the rule's conditions that fail at the
                    onset's last hit row (or the exception that blocks it). Camera view as in failures.
--kf_mode errors    both (the rule computed once): <run>/fail/ and <run>/missed/, --kf_num cases each.
Cases from the first --kf_val_videos val videos; the agent outlined in red. The text says only what matters; the
video, agent and frame are in the folder name.

    ./run_roadpp_docker.sh --gpu 3 --jun_keyframes -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --kf_head Car:Jun --kf_mode failures
Writes <output_dir>/keyframes/<head>[_fail_<rule hash>]_<date-time>/<case>/<k>_<time>s.png (+ cases.json with the rule in
failures mode).
"""
import hashlib
import random
import time
from pathlib import Path

import torch

from src.exp_roadpp import config, hand_rules, residual_clips, step05, step05_mine, utils_data, utils_pipe
from src.exp_roadpp.pair_table import collect, fp_classes, split_tables
from src.exp_roadpp.rule_direct import head_rows

USER_RULE = "!shp:rec_w_shrink+h_shrink(own)@1 & av:tl_grow(other)@1"


def when_text(s, zero):
    return zero if s == 0 else f"{s:+.0f} s" if s == int(s) else f"{s:+.1f} s"


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--kf_head", type=str, default="Car:Jun")
    parser.add_argument("--kf_mode", choices=["onsets", "failures", "missed", "errors"], default="onsets")
    parser.add_argument("--kf_times", type=str, default="-4,-3,-2,-1,0,1,2", help="onsets: s from the onset")
    parser.add_argument("--kf_rule", type=str, default=USER_RULE, help="failures: the rule (hand_rules syntax)")
    parser.add_argument("--kf_fail_times", type=str, default="-1,0,1,2,4",
                        help="failures: s from the predicting frame")
    parser.add_argument("--kf_num", type=int, default=None, help="cases (default: onsets 5, failures 100)")
    parser.add_argument("--kf_val_videos", type=int, default=None, help="val videos (default: onsets 200, failures all)")
    parser.add_argument("--kf_seen_s", type=float, default=None,
                        help="failures: a failure only if the agent stays in view this many s after the prediction "
                             "(else the outcome is unknown; default: the horizon)")
    parser.add_argument("--kf_seed", type=int, default=0)
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id
    fail, missed = args.kf_mode in ("failures", "errors"), args.kf_mode in ("missed", "errors")
    ruled = fail or missed
    on_times = [float(x) for x in args.kf_times.split(",")]                 # onsets and missed: s from the onset
    fail_times = [float(x) for x in args.kf_fail_times.split(",")]          # failures: s from the prediction
    num = args.kf_num or (5 if args.kf_mode == "onsets" else 50)
    if len(on_times) < 5 or len(fail_times) < 5:
        raise ValueError("at least 5 frames per case")
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    training_data = step05.load_training_data(cfg)
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    h = names.index(args.kf_head)
    head_mask = training_data["target_heads"].to(device)
    possible = step05.class_possible(training_data, vocab, device)
    S, w = model.slot_size, int(cfg.get("clause_slots", 1))
    H = int(max(model.windows))
    nv = args.kf_val_videos if args.kf_val_videos is not None else (0 if ruled else 200)
    ids = sorted(training_data["val_videos"])[:nv or None]
    rng = random.Random(args.kf_seed)

    if not ruled:
        d = collect(head_rows(model, training_data, ids, h, possible, head_mask, S, w, device, "val"), model,
                    torch.zeros(1, dtype=torch.long, device=device))
        onsets = sorted({(int(v), int(n), int(t + f)) for v, n, t, f in d["meta"][d["y"]].tolist()})
        cases = [(v, n, t, f"{d['vids'][v]}_a{n}_f{t}", "", None) for v, n, t in rng.sample(onsets, min(num, len(onsets)))]
        print(f"[Keyframes] {len(onsets)} onsets of {args.kf_head} in {len(ids)} val videos; {len(cases)} shown")
    else:
        lit_of = {step05_mine.literal_text(model, names, m): m for m in range(model.num_literals)}
        tl_lits = hand_rules.tl_literals(args.kf_rule)
        extra = hand_rules.BOX_LITS + tl_lits
        (rule,), _ = hand_rules.parse_rules(args.kf_rule, {**lit_of, **{b: None for b in extra}})
        lits = set(hand_rules.rule_literals(rule))
        used = sorted({lit_of[x] for x in lits if x in lit_of})
        col = {x: used.index(lit_of[x]) if x in lit_of else len(used) + extra.index(x) for x in lits}
        cls = args.kf_head.split(":")[0] + ":"
        sel = torch.tensor([a for a, nm in enumerate(names) if nm.startswith(cls) and a != h], device=device)
        sel_names = [names[a].split(":", 1)[1] for a in sel.tolist()]
        store = []
        d = collect(hand_rules.rows_next(model, training_data, ids, h, possible, head_mask, S, w, device, "val",
                                         sel, store), model, torch.tensor(used or [0], dtype=torch.long, device=device))
        d["F"] = d["F"][:, :len(used)]
        bc, tl_h, leave = hand_rules.box_columns(d, cfg, "val", training_data["num_atoms"], 0.02, 5)
        d["F"] = torch.cat([d["F"], bc] + [hand_rules.tl_column(tl_h, x)[:, None] for x in tl_lits], 1)
        t = split_tables(d, torch.arange(d["F"].shape[1]), device)
        nxt = torch.cat(store) if store else torch.zeros((0, len(sel)), dtype=torch.int16)
        m = hand_rules.rule_holds(t, rule, col)
        cases = []
    if missed:
        # the onsets on none of whose hit rows the rule holds; why: the failing conditions at the last hit row
        ev = t["ev_row"]
        caught = set(torch.unique(ev[m & t["y"]]).tolist())
        last = {}
        for i in torch.nonzero(t["y"]).flatten().tolist():                 # rows in time order per agent
            last[int(ev[i])] = i
        miss = sorted(e for e in last if e not in caught)
        print(f"[Keyframes] {t['E']} onsets of {args.kf_head} in {len(ids)} val videos; the rule catches "
              f"{len(caught)}, misses {len(miss)}")
        conds = [(" | ".join(c), t["F"][:, [col[x] for x in c]].any(dim=1)) for c in rule[0]]
        blocks = [(" & ".join(" | ".join(c) for c in e), hand_rules._conj_holds(t, e, col)) for e in rule[1]]
        meta = d["meta"].tolist()
        for e in rng.sample(miss, min(num, len(miss))):
            i = last[e]
            v, n, tt, first = meta[i]
            off = [txt for txt, cm in conds if not bool(cm[i])]
            blk = [txt for txt, bm in blocks if bool(bm[i])]
            why = ("fails: " + ", ".join(off) if off else "") + ("; " if off and blk else "") + \
                ("blocked by: " + ", ".join(blk) if blk else "")
            cases.append((v, n, tt + first, f"missed_{d['vids'][v]}_a{n}_f{tt + first}", why, "missed"))
    if fail:
        fp = fp_classes(t, m, H, S, masks=True)
        kind = torch.full((len(m),), -1, dtype=torch.long, device=device)
        kinds = ("early", "later", "after", "never")
        for i, k in enumerate(kinds):
            kind[fp["_masks"][k]] = i
        happens = m & (t["y"] | fp["_masks"]["early"] | fp["_masks"]["later"])     # the agent does it later
        fa = fp_classes(t, torch.ones_like(m), H, S, masks=True)["_masks"]
        base_rows = float((t["y"] | fa["early"] | fa["later"]).float().mean())
        key = d["meta"][:, 0] * 1_000_000 + d["meta"][:, 1]
        ag_all, ag_ok = torch.unique(key[m.cpu()]), torch.unique(key[happens.cpu()])
        print(f"[Keyframes] timing ignored: the rule holds on {int(m.sum())} rows of {len(ag_all)} agents; the agent "
              f"does {args.kf_head} later on {float(happens.sum()) / max(int(m.sum()), 1):.3f} of the rows, "
              f"{len(ag_ok) / max(len(ag_all), 1):.3f} of the agents (base: {base_rows:.3f} of all rows)")
        # after / never: it does not happen; the agent leaving the view within --kf_seen_s: unknown, not shown
        no = (kind == 2) | (kind == 3)
        lv = leave.to(device)
        key_d = key.to(device)
        for sec in (1, 2, 4, 6, 8, H / 10):
            ok_s = no & (lv >= int(sec * 10))
            print(f"[Keyframes]   agent in view >= {sec:g} s after the prediction: {int(ok_s.sum())} rows, "
                  f"{len(torch.unique(key_d[ok_s]))} agents")
        need = int(round((args.kf_seen_s if args.kf_seen_s is not None else H / 10) * 10))
        seen = lv >= need
        print(f"[Keyframes] it does not happen on {int(no.sum())} rows; the agent leaves the view within {need / 10:g} s "
              f"on {int((no & ~seen).sum())} of them (unknown, dropped)")
        rows = torch.nonzero(no & seen).flatten().tolist()
        rng.shuffle(rows)
        meta, kind = d["meta"].tolist(), kind.cpu().tolist()
        per_agent = {}
        for i in rows:                                                   # one random failing row per agent
            per_agent.setdefault((meta[i][0], meta[i][1]), i)
        picked = rng.sample(sorted(per_agent.values()), min(num, len(per_agent)))
        for i in picked:
            v, n, tt, first = meta[i]
            k = kinds[kind[i]]
            acts = sorted((int(f), sel_names[j]) for j, f in enumerate(nxt[i].tolist()) if f < H)
            why = ("no Jun later (only before)" if k == "after" else "never Jun") + "; next: " + \
                (", ".join(f"{a} {f / 10:.1f} s" for f, a in acts[:3]) if acts else "nothing new")
            stay = int(leave[i])
            why += "; in view " + (f"{stay / 10:.1f} s" if stay < hand_rules.LEAVE_NEVER else "to the end")
            cases.append((v, n, tt, f"{k}_{d['vids'][v]}_a{n}_f{tt}", why, k))
        counts = {k: sum(c[5] == k for c in cases) for k in kinds + ("missed",)}
        print(f"[Keyframes] {len(rows)} rows where it never happens, {len(per_agent)} agents; {len(cases)} shown "
              f"{counts}")

    dataset = Path(args.dataset_path)
    atoms = utils_data.load_json(cfg["val_atom_file"])
    unique_atoms = utils_data.load_json(cfg["unique_atoms_file"])
    label_names = [step05._atom_name(sig) for sig in unique_atoms]
    # failures: one folder per rule (a short hash of its text), so runs of different rules do not mix
    mode_tag = {"failures": "fail", "missed": "missed", "errors": "errors"}.get(args.kf_mode)
    tag = f"_{mode_tag}_" + hashlib.md5(args.kf_rule.encode()).hexdigest()[:6] if ruled else ""
    # every run its own folder (the start time), so runs never mix
    out_dir = args.output_dir / "keyframes" / (args.kf_head.replace(":", "_") + tag + time.strftime("_%Y%m%d-%H%M%S"))
    done = []
    for v, n, ref, case, why, k in cases:
        vid = d["vids"][v]
        X = step05.build_video_tensor(atoms[vid], len(unique_atoms))
        tube = {i: uid for uid, i in step05.video_agent_ids(atoms[vid]).items()}.get(n)
        num_frames = len(residual_clips.sumo_render._video(dataset, vid)[1]["frames"])
        group = "missed" if k == "missed" else "fail" if k else None
        case_dir = out_dir / group / case if args.kf_mode == "errors" else out_dir / case
        for j, s in enumerate(fail_times if group == "fail" else on_times):
            f = ref + int(round(s * 10))
            if not 1 <= f <= num_frames:
                continue
            labs = [label_names[a].partition(":")[2] for a in torch.nonzero(X[min(f, X.shape[0] - 1), n]).flatten().tolist()]
            car = f"car: {', '.join(labs) or 'not in view'}"
            if group == "fail":
                title = f"predicted {args.kf_head}, {when_text(s, 'now')} | {why}\n{car}"
            elif group == "missed":
                title = f"{args.kf_head} {when_text(s, 'onset')}, missed | {why}\n{car}"
            else:
                title = f"{args.kf_head} {when_text(s, 'onset')}\n{car}"
            residual_clips.sumo_render.render(dataset, vid, f, case_dir / f"{j}_{s:+g}s.png", highlight=tube,
                                              title=title, cam_keep={tube} if ruled else None)
        done.append({"case": str(case_dir.relative_to(out_dir)), "why": why, "kind": k})
        print(f"  {case_dir.relative_to(out_dir.parent)}")
    if ruled:
        utils_data.save_json({"rule": args.kf_rule, "mode": args.kf_mode, "cases": done}, out_dir / "cases.json")
    print(f"[Keyframes] {out_dir}")


if __name__ == "__main__":
    main()
