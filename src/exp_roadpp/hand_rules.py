"""Hand-written rules of one head, evaluated on train and val (2026-11-28).

A rule is literals joined by " & ", as `step05_mine.literal_text` writes them ("Car:IncomLane(own)@1",
"!Car:Stop(own)@2", "veh:Jun(other)@1"); rules are separated by ";". Per rule on the head's at-risk rows (as
`pair_table`): rows, precision, row recall, onsets, onset recall, and its negative rows by what the agent does
(`pair_table.fp_classes`: early / later / after / never). Last: the union of the rules (the head predicted if any
holds). An unknown literal stops with the closest names.

    ./run_roadpp_docker.sh --gpu 3 --hand_rules -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --hr_head Car:Jun
Writes <output_dir>/rule_lab/hand_<head>.md / .json.
"""
import difflib

import torch

from src.exp_roadpp import config, step05, step05_mine, utils_data, utils_pipe
from src.exp_roadpp.pair_table import collect, fp_classes, rule_mask, split_tables, summary
from src.exp_roadpp.rule_direct import head_rows

# Car:Jun (the car enters the junction within the hit window): hypotheses from the length-1 table
CAR_JUN = "; ".join([
    "Car:IncomLane(own)@1 & !Car:Stop(own)@1",                         # approaching on the incoming lane, moving
    "Car:IncomLane(own)@2 & !Car:IncomLane(own)@1 & !Car:Stop(own)@1",  # just left the incoming lane
    "Car:Stop(own)@2 & !Car:Stop(own)@1",                              # moves off after a stop
    "Car:Brake(own)@2 & !Car:Brake(own)@1 & !Car:Stop(own)@1",         # releases the brake
    "Car:IncomLane(own)@1 & !Car:Stop(own)@1 & TL:Green(other)@1",     # approaching, green light
    "Car:IncomLane(own)@1 & TL:Red(other)@1",                          # approaching, red light (expected low)
    "Car:XingFmRht(own)@1", "Car:XingFmLft(own)@1",                    # crossing in front of the AV
    "Car:IncatLft(own)@1 & !Car:Stop(own)@1", "Car:IncatRht(own)@1 & !Car:Stop(own)@1",   # indicating, moving
])


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--hr_head", type=str, default="Car:Jun")
    parser.add_argument("--hr_rules", type=str, default=CAR_JUN, help="rules ';' separated, literals ' & '")
    parser.add_argument("--hr_train_videos", type=int, default=1000, help="train videos (the first; 0: all)")
    parser.add_argument("--hr_val_videos", type=int, default=500, help="val videos (the first; 0: all)")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id
    cfg = step05.get_cfg(config.get_step_05_input(args))
    device = cfg["device"]
    training_data = step05.load_training_data(cfg)
    model = step05.build_rule_net(training_data, cfg, 1).to(device)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    h = names.index(args.hr_head)

    # rule text -> literal indices
    lit_of = {step05_mine.literal_text(model, names, m): m for m in range(model.num_literals)}
    rules = [[x.strip() for x in r.split("&")] for r in args.hr_rules.split(";") if r.strip()]
    for r in rules:
        for t in r:
            if t not in lit_of:
                raise ValueError(f"unknown literal {t!r}; close: {difflib.get_close_matches(t, list(lit_of), 5)}")
    used = sorted({lit_of[t] for r in rules for t in r})
    col = {m: i for i, m in enumerate(used)}
    cols = torch.tensor(used, device=device)

    head_mask = training_data["target_heads"].to(device)
    possible = step05.class_possible(training_data, vocab, device)
    S, w = model.slot_size, int(cfg.get("clause_slots", 1))
    H = int(max(model.windows))
    tr_ids = sorted(training_data["train_videos"])[:args.hr_train_videos or None]
    va_ids = sorted(training_data["val_videos"])[:args.hr_val_videos or None]
    gen = lambda ids, d: head_rows(model, training_data, ids, h, possible, head_mask, S, w, device, d)
    idx = torch.arange(len(used))
    data = {sp: split_tables(collect(gen(ids, sp), model, cols), idx, device)
            for sp, ids in (("train", tr_ids), ("val", va_ids))}

    out = []
    union = {sp: torch.zeros(len(t["F"]), dtype=torch.bool, device=device) for sp, t in data.items()}
    for r in rules + [None]:
        res = {"rule": " & ".join(r) if r else "UNION of the rules above"}
        for sp, t in data.items():
            m = rule_mask(t, [col[lit_of[x]] for x in r]) if r else union[sp]
            if r:
                union[sp] |= m
            s = summary(t, m)
            res[sp] = {**s, "row_recall": s["hit_rows"] / max(t["total_hit"], 1), "fp": fp_classes(t, m, H, S)}
        out.append(res)
        tr, va = res["train"], res["val"]
        print(f"[Hand rules] train P {tr['precision']:.3f} onsets {tr['onsets']} | val P {va['precision']:.3f} "
              f"onset R {va['onset_recall']:.3f} never {va['fp']['never']['share']:.2f}  {res['rule']}")

    base = {sp: summary(t, torch.ones(len(t["F"]), dtype=torch.bool, device=device)) for sp, t in data.items()}
    md = [f"# Hand rules of {args.hr_head}", "",
          f"Train {len(tr_ids)} videos, base P {base['train']['precision']:.4f}, {base['train']['onsets']} onsets; "
          f"val {len(va_ids)} videos, base P {base['val']['precision']:.4f}, {base['val']['onsets']} onsets. "
          f"Negatives by what the agent does (share of the rule's negative rows): early = its onset comes later "
          f"within {H} frames, never = it has no onset.", "",
          "| train P | train onsets | val P | val onset R | val row R | val early | val never | rule |",
          "|---|---|---|---|---|---|---|---|"]
    for x in out:
        tr, va = x["train"], x["val"]
        md.append(f"| {tr['precision']:.3f} | {tr['onsets']} | {va['precision']:.3f} | {va['onset_recall']:.3f} | "
                  f"{va['row_recall']:.3f} | {va['fp']['early']['share']:.2f} | {va['fp']['never']['share']:.2f} | "
                  f"{x['rule']} |")
    stem = args.output_dir / "rule_lab" / f"hand_{args.hr_head.replace(':', '_')}"
    open(f"{stem}.md", "w").write("\n".join(md) + "\n")
    utils_data.save_json({"head": args.hr_head, "base": base, "rules": out}, f"{stem}.json")
    print("\n".join(md))
    print(f"\n[Hand rules] {stem}.md / .json")


if __name__ == "__main__":
    main()
