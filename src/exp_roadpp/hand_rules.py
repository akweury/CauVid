"""Hand-written rules of one head, evaluated on train and val (2026-11-28).

A rule is conditions joined by " & ", a condition one literal as `step05_mine.literal_text` writes it
("Car:IncomLane(own)@1", "!Car:Stop(own)@2", "veh:Jun(other)@1") or several joined by " | " (any of them; brackets
optional); " unless " followed by such a conjunction drops the rows where that holds (several allowed: "A & B
unless C & D unless E" = A & B & !(C & D) & !E); rules are separated by ";", an optional "name:=" labels a rule.
Box literals (2026-11-28; no atom says this, read from the agent's GT box at the row's frame t, `box_columns`):
  box:edge_left / box:edge_right / box:edge_bottom  the box's side within --hr_edge of that image border;
  box:to_left / box:to_right / box:to_bottom        over the last --hr_move frames the box centre moved left / right,
                                                    its bottom moved down (by more than 0.002 of the image);
  box:leaving   (edge_left & to_left) | (edge_right & to_right) | (edge_bottom & to_bottom): leaving the image;
each also negated ("!box:leaving"; false where the agent has no box at t or t - --hr_move).
Traffic light size (2026-11-28): the height (share of the image height) of the largest "TL" box at the row's frame;
  literals "tl:h>=0.03", "tl:h<0.2" (any number; false without a traffic light in view) and "tl:none" / "tl:any".
  The table "traffic light size" gives the precision of the at-risk rows (and of the first rule's rows) per decile
  of that height (train deciles) and without a traffic light, to choose the range. Per rule on the head's at-risk rows (as
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
import re

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


def _conj(text):
    """A conjunction: a list of conditions, a condition a list of literal texts (any of them holds)."""
    return [[x.strip(" ()") for x in c.split("|")] for c in text.split("&")]


def parse_rules(spec, lit_of):
    """(rules, labels): a rule is (body, exceptions), the body a conjunction (`_conj`), the exceptions a list of
    conjunctions (" unless ")."""
    rules, labels = [], []
    for r in (r.strip() for r in spec.split(";")):
        if not r:
            continue
        name, sep, body = r.partition(":=")
        labels.append(name.strip() if sep else None)
        parts = (body if sep else r).split(" unless ")
        rules.append((_conj(parts[0]), [_conj(p) for p in parts[1:]]))
    for r in rules:
        for t in rule_literals(r):
            if t not in lit_of:
                raise ValueError(f"unknown literal {t!r}; close: {difflib.get_close_matches(t, list(lit_of), 5)}")
    return rules, labels


def rule_literals(r):
    return [t for conj in [r[0]] + r[1] for c in conj for t in c]


def rule_text(r):
    conj = lambda q: " & ".join(c[0] if len(c) == 1 else "(" + " | ".join(c) + ")" for c in q)
    return conj(r[0]) + "".join(f" unless {conj(e)}" for e in r[1])


def _conj_holds(t, conj, col):
    m = torch.ones(len(t["F"]), dtype=torch.bool, device=t["F"].device)
    for c in conj:
        m &= t["F"][:, [col[x] for x in c]].any(dim=1)
    return m


def rule_holds(t, r, col):
    """(rows,) where the rule r holds (its body and none of its exceptions); col: literal text -> column of t["F"]."""
    m = _conj_holds(t, r[0], col)
    for e in r[1]:
        m &= ~_conj_holds(t, e, col)
    return m


def rows_next(model, training_data, ids, h, possible, head_mask, S, w, device, desc, sel, store):
    """`rule_direct.head_rows`, storing per row (in store) the frames to the agent's next onset of the atoms sel."""
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


BOX_NAMES = ("edge_left", "edge_right", "edge_bottom", "to_left", "to_right", "to_bottom", "leaving")
BOX_LITS = [f"box:{x}" for x in BOX_NAMES] + [f"!box:{x}" for x in BOX_NAMES]
MOVE_EPS = 0.002


TL_LIT = re.compile(r"^tl:h(>=|<)([0-9.]+)$")


def tl_literals(spec):
    """The traffic light literals written in a rules spec."""
    return sorted({m.group(0) for m in re.finditer(r"tl:h(?:>=|<)[0-9.]+|tl:none|tl:any", spec)})


def tl_column(tl_h, lit):
    """(rows,) bool of one traffic light literal from the per-row height (NaN: none in view)."""
    if lit in ("tl:none", "tl:any"):
        return torch.isnan(tl_h) if lit == "tl:none" else ~torch.isnan(tl_h)
    op, v = TL_LIT.match(lit).groups()
    return (tl_h >= float(v)) if op == ">=" else (tl_h < float(v))          # NaN compares False


def box_columns(d, cfg, split, num_atoms, edge, k):
    """(rows, len(BOX_LITS)) bool of the rows of `pair_table.collect` d (meta: video index, agent, frame t), from
    the split's GT boxes (`step05._agent_boxes`, normalised x1, y1, x2, y2), and (rows,) float the height of the
    largest traffic light ("TL") box at the row's frame (NaN: none)."""
    atoms = utils_data.load_json(cfg[f"{split}_atom_file"])
    boxes = torch.load(cfg[f"{split}_box_file"], map_location="cpu")
    meta = d["meta"]
    out = torch.zeros((len(meta), len(BOX_LITS)), dtype=torch.bool)
    tl_h = torch.full((len(meta),), float("nan"))
    for v, vid in enumerate(tqdm(d["vids"], desc=f"{split}: boxes")):
        rows = torch.nonzero(meta[:, 0] == v).flatten()
        if not len(rows):
            continue
        X = step05.build_video_tensor(atoms[vid], num_atoms)
        vb, classes, has = step05._agent_boxes(boxes, vid, atoms[vid], X)
        n, t = meta[rows, 1], meta[rows, 2].clamp(max=vb.shape[0] - 1)
        tl = [i for i, c in enumerate(classes) if c == "TL"]
        if tl:
            hts = (vb[:, tl, 3] - vb[:, tl, 1]).nan_to_num(nan=-1.0).max(dim=1).values      # (T,)
            tl_h[rows] = torch.where(hts[t] >= 0, hts[t], torch.tensor(float("nan")))
        t0 = (t - k).clamp(min=0)
        b, b0 = vb[t, n], vb[t0, n]
        ok = has[t, n] & has[t0, n] & (t >= k)
        x1, y1, x2, y2 = b.unbind(1)
        cx, cx0 = (x1 + x2) / 2, (b0[:, 0] + b0[:, 2]) / 2
        f = {"edge_left": x1 <= edge, "edge_right": x2 >= 1 - edge, "edge_bottom": y2 >= 1 - edge,
             "to_left": cx < cx0 - MOVE_EPS, "to_right": cx > cx0 + MOVE_EPS, "to_bottom": y2 > b0[:, 3] + MOVE_EPS}
        f["leaving"] = (f["edge_left"] & f["to_left"]) | (f["edge_right"] & f["to_right"]) | \
            (f["edge_bottom"] & f["to_bottom"])
        for j, name in enumerate(BOX_NAMES):
            out[rows, j] = f[name] & ok
            out[rows, len(BOX_NAMES) + j] = ~f[name] & ok
    return out, tl_h


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--hr_head", type=str, default="Car:Jun")
    parser.add_argument("--hr_rules", type=str, default=CAR_JUN, help="rules ';' separated, literals ' & '")
    parser.add_argument("--hr_train_videos", type=int, default=1000, help="train videos (the first; 0: all)")
    parser.add_argument("--hr_val_videos", type=int, default=500, help="val videos (the first; 0: all)")
    parser.add_argument("--hr_edge", type=float, default=0.02, help="box literals: a side this close to the border")
    parser.add_argument("--hr_move", type=int, default=5, help="box literals: frames of the move")
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
    tl_lits = tl_literals(args.hr_rules)
    rules, labels = parse_rules(args.hr_rules, {**lit_of, **{b: None for b in BOX_LITS + tl_lits}})
    lits = {t for r in rules for t in rule_literals(r)}
    used = sorted({lit_of[t] for t in lits if t in lit_of})
    extra = BOX_LITS + tl_lits                                              # computed columns after the literals
    col = {t: used.index(lit_of[t]) if t in lit_of else len(used) + extra.index(t) for t in lits}
    cols = torch.tensor(used or [0], dtype=torch.long, device=device)

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

    gen = lambda ids, desc, store: rows_next(model, training_data, ids, h, possible, head_mask, S, w, device, desc,
                                             sel, store)

    data = {}
    for sp, ids in (("train", tr_ids), ("val", va_ids)):
        store = []
        d = collect(gen(ids, sp, store), model, cols)
        d["F"] = d["F"][:, :len(used)]
        bc, tl_h = box_columns(d, cfg, sp, training_data["num_atoms"], args.hr_edge, args.hr_move)
        d["F"] = torch.cat([d["F"], bc] + [tl_column(tl_h, x)[:, None] for x in tl_lits], dim=1)
        data[sp] = split_tables(d, torch.arange(d["F"].shape[1]), device)
        data[sp]["tl_h"] = tl_h.to(device)
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
    mask = lambda t, r: rule_holds(t, r, col)

    text = rule_text
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
    # Traffic light size: precision per decile of the TL height (train deciles), all rows and the first rule's rows
    trh = data["train"]["tl_h"]
    q = torch.quantile(trh[~torch.isnan(trh)].float().cpu(), torch.linspace(0.1, 0.9, 9)).tolist() if \
        (~torch.isnan(trh)).any() else []
    edges = [0.0] + q + [float("inf")]
    first_rule = rules[0] if rules else None
    eventual = {}                                                           # the agent does it later (any time)
    for sp, t in data.items():
        fa = fp_classes(t, torch.ones_like(t["y"]), H, S, masks=True)["_masks"]
        eventual[sp] = t["y"] | fa["early"] | fa["later"]
    tl_rows = []
    for i in range(len(edges)):
        row = {"bin": "none" if i == len(edges) - 1 else [edges[i], edges[i + 1]]}
        for sp, t in data.items():
            hh = t["tl_h"]
            b = torch.isnan(hh) if i == len(edges) - 1 else (hh >= edges[i]) & (hh < edges[i + 1])
            ev = eventual[sp]
            rm = b & mask(t, first_rule) if first_rule else b
            row[sp] = {"rows": int(b.sum()), "precision": float(t["y"][b].float().mean()) if b.any() else 0.0,
                       "eventual": float(ev[b].float().mean()) if b.any() else 0.0,
                       "rule_rows": int(rm.sum()), "rule_precision": float(t["y"][rm].float().mean()) if rm.any() else 0.0}
        tl_rows.append(row)
    lab0 = (f"[{labels[0]}]" if labels and labels[0] else "the first rule") if first_rule else "-"
    md += ["", "## Traffic light size (height of the largest TL box, share of the image height)", "",
           f"Per train decile of the height and without a traffic light: P = hit within the window; eventual = the "
           f"car does {args.hr_head} later at any time; rule: the rows of {lab0}. Base val P "
           f"{base['val']['precision']:.3f}.", "",
           "| TL height | train rows | train P | val rows | val P | val eventual | rule val rows | rule val P |",
           "|---|---|---|---|---|---|---|---|"]
    for r in tl_rows:
        bt = "no TL" if r["bin"] == "none" else f"[{r['bin'][0]:.3f}, {r['bin'][1]:.3f})"
        tr, va = r["train"], r["val"]
        md.append(f"| {bt} | {tr['rows']} | {tr['precision']:.3f} | {va['rows']} | {va['precision']:.3f} | "
                  f"{va['eventual']:.3f} | {va['rule_rows']} | {va['rule_precision']:.3f} |")
    stem = args.output_dir / "rule_lab" / f"hand_{args.hr_head.replace(':', '_')}"
    open(f"{stem}.md", "w").write("\n".join(md) + "\n")
    utils_data.save_json({"head": args.hr_head, "base": base, "rules": out, "tl_size": tl_rows}, f"{stem}.json")
    print("\n".join(md))
    print(f"\n[Hand rules] {stem}.md / .json")


if __name__ == "__main__":
    main()
