"""Hand-written rules of one head, evaluated on train and val (2026-11-28).

A rule is conditions joined by " & ", a condition one literal as `step05_mine.literal_text` writes it
("Car:IncomLane(own)@1", "!Car:Stop(own)@2", "veh:Jun(other)@1") or several joined by " | " (any of them; brackets
optional); rules are separated by ";", an optional "name:" before a rule labels it. Per rule on the head's at-risk rows (as
`pair_table`): rows, precision, row recall, onsets, onset recall, and its negative rows by what the agent does
(`pair_table.fp_classes`: early / later / after / never). Last: the union of the rules (the head predicted if any
holds). An unknown literal stops with the closest names.
Failures (negative rows, 2026-11-28, the user): is it the timing (the agent does the head, but earlier / later than the
hit window: early, later, after) or does the agent do something else (never)? For the "never" rows: the agent's
label atoms of its own class (not the head) that switch on within the hit window and within the horizon, as shares
of those rows ("nothing new": none switches on).

    ./run_roadpp_docker.sh --gpu 3 --hand_rules -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \\
        --run_id 20261125-v8_lab --hr_head Car:Jun
Writes <output_dir>/rule_lab/hand_<head>.md / .json.
"""
import difflib

import torch
from tqdm import tqdm

from src.exp_roadpp import config, step05, step05_mine, utils_data, utils_pipe
from src.exp_roadpp.pair_table import collect, fp_classes, split_tables, summary

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
    rules, labels = [], []
    for r in (r.strip() for r in args.hr_rules.split(";")):
        if not r:
            continue
        name, sep, body = r.partition(":=")
        labels.append(name.strip() if sep else None)
        body = body if sep else r
        rules.append([[x.strip(" ()") for x in c.split("|")] for c in body.split("&")])
    for r in rules:
        for t in (t for c in r for t in c):
            if t not in lit_of:
                raise ValueError(f"unknown literal {t!r}; close: {difflib.get_close_matches(t, list(lit_of), 5)}")
    used = sorted({lit_of[t] for r in rules for c in r for t in c})
    col = {m: i for i, m in enumerate(used)}
    cols = torch.tensor(used, device=device)

    head_mask = training_data["target_heads"].to(device)
    possible = step05.class_possible(training_data, vocab, device)
    S, w = model.slot_size, int(cfg.get("clause_slots", 1))
    H = int(max(model.windows))
    tr_ids = sorted(training_data["train_videos"])[:args.hr_train_videos or None]
    va_ids = sorted(training_data["val_videos"])[:args.hr_val_videos or None]
    # the agent's next actions: its own class's label atoms (not the head), frames to their next onset
    cls = args.hr_head.split(":")[0] + ":"
    sel = torch.tensor([a for a, nm in enumerate(names) if nm.startswith(cls) and a != h], device=device)
    sel_names = [names[a].split(":", 1)[1] for a in sel.tolist()]

    def gen(ids, desc, store):
        """`rule_direct.head_rows`, storing per row the frames to the next onset of the atoms sel."""
        for vid in tqdm(ids, desc=desc):
            can = possible(vid)
            for frames in training_data["frames"][vid].split(128):
                rows = step05._video_rows(model, training_data, vid, frames, device)
                if len(rows["inputs"]) == 0:
                    continue
                keep, y = step05_mine._head_rows(rows, can, head_mask, S, w)
                keep, y = keep[:, h], y[:, h]
                if keep.any():
                    store.append(rows["first"][keep][:, sel].to(torch.int16).cpu())
                    yield vid, rows["inputs"][keep], y[keep], rows["rows"][keep], rows["first"][keep][:, h]

    idx = torch.arange(len(used))
    data = {}
    for sp, ids in (("train", tr_ids), ("val", va_ids)):
        store = []
        data[sp] = split_tables(collect(gen(ids, sp, store), model, cols), idx, device)
        data[sp]["next"] = torch.cat(store).to(device) if store else torch.zeros((0, len(sel)), device=device)

    def next_actions(t, m, top=5):
        """Shares of the rows m with each next action within the hit window and within the horizon."""
        n = max(int(m.sum()), 1)
        out = {}
        for key, lim in (("hit_window", w * S), ("horizon", H)):
            on = t["next"][m] < lim                                         # (rows, sel)
            share = on.float().sum(0) / n
            order = torch.argsort(share, descending=True)[:top].tolist()
            out[key] = {"nothing_new": float((~on.any(dim=1)).float().sum()) / n,
                        "top": [[sel_names[i], float(share[i])] for i in order if share[i] > 0]}
        return out

    out = []
    union = {sp: torch.zeros(len(t["F"]), dtype=torch.bool, device=device) for sp, t in data.items()}
    def mask(t, r):
        m = torch.ones(len(t["F"]), dtype=torch.bool, device=device)
        for c in r:
            m &= t["F"][:, [col[lit_of[x]] for x in c]].any(dim=1)
        return m

    text = lambda r: " & ".join(c[0] if len(c) == 1 else "(" + " | ".join(c) + ")" for c in r)
    for r, lab in list(zip(rules, labels)) + [(None, None)]:
        res = {"rule": (f"[{lab}] " if lab else "") + text(r) if r else "UNION of the rules above"}
        for sp, t in data.items():
            m = mask(t, r) if r else union[sp]
            if r:
                union[sp] |= m
            s = summary(t, m)
            fp = fp_classes(t, m, H, S, masks=True)
            fm = fp.pop("_masks")
            fp["timing"] = sum(fp[k]["share"] for k in ("early", "later", "after"))
            fp["never_next"] = next_actions(t, fm["never"])
            res[sp] = {**s, "row_recall": s["hit_rows"] / max(t["total_hit"], 1), "fp": fp}
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
    acts = lambda x: ", ".join(f"{a} {v:.2f}" for a, v in x["top"]) + f"; nothing new {x['nothing_new']:.2f}"
    md += ["", "## Failures on val: timing or another action", "",
           f"timing: the agent does {args.hr_head} but outside the hit window (early / later / after); other: it never "
           f"does it. Next actions of the 'other' rows: its labels switching on within {w * S / 10:g} s (the hit "
           f"window) and {H / 10:g} s, as shares of those rows.", "",
           f"| rule | timing | other | other: next within {w * S / 10:g} s | other: next within {H / 10:g} s |",
           "|---|---|---|---|---|"]
    for x in out:
        f = x["val"]["fp"]
        md.append(f"| {x['rule']} | {f['timing']:.2f} | {f['never']['share']:.2f} | "
                  f"{acts(f['never_next']['hit_window'])} | {acts(f['never_next']['horizon'])} |")
    stem = args.output_dir / "rule_lab" / f"hand_{args.hr_head.replace(':', '_')}"
    open(f"{stem}.md", "w").write("\n".join(md) + "\n")
    utils_data.save_json({"head": args.hr_head, "base": base, "rules": out}, f"{stem}.json")
    print("\n".join(md))
    print(f"\n[Hand rules] {stem}.md / .json")


if __name__ == "__main__":
    main()
