"""Cause A check: does "the cue just started" predict a head much better than "the cue holds"?

Count only, no training. The rule net's bodies are conjunctions of positive atoms ("X held in slot j"),
so it cannot say "X just started" (X in slot j, not in slot j + 1), which needs a negated literal. The
SUMO rules for the ego's brake fire on onsets: R1 a pedestrian starts to cross, R3 the vehicle ahead
starts to brake (`synth/sumo/rules.py`). If the onset form of a cue has a clearly higher confidence than
its state form at a similar recall, negated / "just started" literals are worth adding to the rule net.

Counts as `step05._rule_head_counts` at f = 0 (the head switches on within the next slot, at-risk and
min_off_frames as training), over the rows of the agents that can hold the head (`step05.class_possible`;
for av:* only the ego). A body holds on a row if every literal does:
  positive  "G@j":  some atom of group G is true in slot j (in scope s);
  negative  "!G@j": every atom of G is false in slot j and the slot is known (unknown is not false).
Scope "other" pools all other agents: "!Brake@2" means no other agent braked in slot 2, stricter than "this
vehicle did not brake". With --roles ahead the same cues are also counted in the "ahead" scope (one agent).

Variants per cue G and slot j (j = 1: 0-1 s before the head's slot, 2: 1-2 s):
  state   cond & G@j
  onset   cond & G@j & !G@(j+1)                 (started in slot j)
  onset2  cond & G@j & !G@(j+1) & !G@(j+2)      (and off for 2 s before)
cond is --cond (default "av:AV-Mov" own, slot 1: the ego was moving); references: the empty body, cond alone.

    ./run_roadpp_docker.sh --gpu 0 --onset_cue_diag -- --exp sumo_maps_v7 --machine dgx-jing
    (optional --head av:AV-Brake --cond av:AV-Mov; with --roles ahead for the ahead scope)
Writes <output_dir>/onset_cue_diag/<head>.json and prints a table (train and val).
"""
import json
from collections import defaultdict

import torch
from tqdm import tqdm

from src.exp_roadpp import config, step05, utils_data, utils_pipe
from src.exp_roadpp.cauvid import TemporalRuleNet

MOTOR = ("Car", "MedVeh", "LarVeh", "Bus", "Mobike")
# Cues of the ego's brake (R1, R3, N3); a cue is an OR over its atoms, "+" joins two cues that must hold in
# the same slot (the onset negates the first one only).
DEFAULT_CUES = {
    "av:AV-Brake": {
        "Car:Brake": ["Car:Brake"],
        "veh:Brake": [f"{c}:Brake" for c in MOTOR],
        "veh:Brake+VehLane": ([f"{c}:Brake" for c in MOTOR], [f"{c}:VehLane" for c in MOTOR]),
        "Ped:Xing*": ["Ped:Xing", "Ped:XingFmLft", "Ped:XingFmRht"],
        "Ped:xing(loc)": ["Ped:xing"],
        "TL:Red": ["TL:Red"],
        "TL:Amber": ["TL:Amber"],
    },
}


def build_model(training_data, cfg):
    """A one-node TemporalRuleNet: only its input layout and `training_rows` are used."""
    return TemporalRuleNet(training_data["num_atoms"], cfg["slot_size"], cfg["num_slots"], 1, cfg["init_log_alpha"],
                           cfg["mask_type"], cfg["windows"], roles=cfg["roles"],
                           ranges=cfg.get("slot_ranges", False)).to(cfg["device"])


def body_variants(cues, scopes, num_slots):
    """{name: [literal]}, literal (atoms, scope, j, positive), atoms a list of atom indices (OR);
    a cue joined with "+" adds its second group as a positive literal in the same slot."""
    variants = {}
    for cue, groups in cues.items():
        first, extra = (groups if isinstance(groups, tuple) else (groups, None))
        for scope_name, scope in scopes.items():
            for j in range(1, num_slots):
                base = [(first, scope, j, True)] + ([(extra, scope, j, True)] if extra else [])
                tag = f"{cue}({scope_name})@{j}"
                variants[f"{tag} state"] = base
                variants[f"{tag} onset"] = base + [(first, scope, j + 1, False)]
                if j + 2 <= num_slots:
                    variants[f"{tag} onset2"] = base + [(first, scope, j + 1, False), (first, scope, j + 2, False)]
    return variants


@torch.no_grad()
def count(model, training_data, video_ids, head, cond, variants, possible, cfg):
    """{variant: {fires, hits, videos}} at f = 0 over the at-risk rows of agents that can hold head."""
    device, S = cfg["device"], model.slot_size
    names = ["(empty)", "(cond)"] + list(variants)
    fires = torch.zeros(len(names), device=device)
    hits = torch.zeros(len(names), device=device)
    videos = torch.zeros(len(names), device=device)

    def literal(x, k, atoms, scope, j, positive):
        idx = torch.tensor([model.input_index(a, scope, j) for a in atoms], device=device)
        if positive:
            return x[:, idx].amax(dim=1)
        return ((1 - x[:, idx]) * k[:, idx]).amin(dim=1)

    for video_id in tqdm(video_ids, desc="Counting cues"):
        video_hits = torch.zeros(len(names), device=device)
        can = possible(video_id)[:, head]
        for frames in training_data["frames"][video_id].split(cfg["frames_per_video"]):
            rows = step05._video_rows(model, training_data, video_id, frames, device)
            if len(rows["inputs"]) == 0:
                continue
            x, k = model.split_inputs(rows["inputs"])
            first, at_risk = rows["first"][:, head], rows["at_risk"][:, head]
            keep = at_risk & can[rows["rows"][:, 1]]
            risk = (keep & (first >= 0) & rows["known"][:, 0, head]).float()
            onset = (keep & (first >= 0) & (first < S)).float()
            c = literal(x, k, *cond)
            holds = [torch.ones_like(c), c]
            for lits in variants.values():
                h = c.clone()
                for lit in lits:
                    h = h * literal(x, k, *lit)
                holds.append(h)
            H = torch.stack(holds, dim=1)                                   # (B, V)
            fires += H.T @ risk
            video_hits += H.T @ onset
        hits += video_hits
        videos += video_hits > 0
    return {n: {"fires": int(fires[i]), "hits": int(hits[i]), "videos": int(videos[i])} for i, n in enumerate(names)}


def table(counts):
    """Adds confidence, lift (over the empty body and over cond alone) and recall (share of the onsets)."""
    base, cond = counts["(empty)"], counts["(cond)"]
    rate = lambda c: c["hits"] / c["fires"] if c["fires"] else None
    out = {}
    for name, c in counts.items():
        conf = rate(c)
        out[name] = {**c, "confidence": conf,
                     "lift": conf / rate(base) if conf is not None and rate(base) else None,
                     "lift_over_cond": conf / rate(cond) if conf is not None and rate(cond) else None,
                     "recall": c["hits"] / base["hits"] if base["hits"] else None}
    return out


def main():
    parser = utils_pipe.get_parser()
    parser.add_argument("--head", type=str, default="av:AV-Brake", help="head atom name")
    parser.add_argument("--cond", type=str, default="av:AV-Mov",
                        help="own atom in slot 1 every body holds (empty string: none)")
    args = utils_pipe.parse_args(parser)
    args = config.machine_setup(args)
    args.run_dir = args.output_dir / "runs" / args.run_id if getattr(args, "run_id", None) \
        else args.output_dir / "onset_cue_diag"
    cfg = step05.get_cfg(config.get_step_05_input(args))
    training_data = step05.load_training_data(cfg)
    vocab = step05.load_vocabulary(cfg)
    names = [step05._atom_name(sig) for sig in vocab]
    index = defaultdict(list)
    for i, n in enumerate(names):
        index[n].append(i)

    def atoms_of(group):
        found = [i for n in group for i in index.get(n, [])]
        missing = [n for n in group if n not in index]
        if missing:
            print(f"[onset_cue_diag] not in the vocabulary, skipped: {missing}")
        return found

    if args.head not in index:
        raise SystemExit(f"head {args.head} not in the vocabulary")
    head = index[args.head][0]
    model = build_model(training_data, cfg)
    scopes = {"other": 1}
    if "ahead" in cfg["roles"]:
        scopes["ahead"] = 2 + list(cfg["roles"]).index("ahead")
    cues = {}
    for cue, groups in DEFAULT_CUES.get(args.head, {}).items():
        if isinstance(groups, tuple):
            a, b = atoms_of(groups[0]), atoms_of(groups[1])
            if a and b:
                cues[cue] = (a, b)
        else:
            found = atoms_of(groups)
            if found:
                cues[cue] = found
    if not cues:
        raise SystemExit(f"no cues for {args.head}: add them to DEFAULT_CUES")
    variants = body_variants(cues, scopes, cfg["num_slots"])
    cond = (atoms_of([args.cond]), 0, 1, True) if args.cond else ([head], 0, 1, False)   # no cond: head off
    possible = step05.class_possible(training_data, vocab, cfg["device"])

    result = {"exp": args.exp, "head": args.head, "cond": args.cond, "roles": list(cfg["roles"]),
              "num_slots": cfg["num_slots"], "slot_size": cfg["slot_size"]}
    for split in ("train", "val"):
        counts = count(model, training_data, training_data[f"{split}_videos"], head, cond, variants, possible, cfg)
        result[split] = table(counts)

    out_dir = args.output_dir / "onset_cue_diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.head.replace(':', '_')}.json"
    utils_data.save_json(result, out_file)

    tr, va = result["train"], result["val"]
    fmt = lambda v, p=3: "   -  " if v is None else f"{v:.{p}f}"
    print(f"\n[onset_cue_diag] {args.exp} head {args.head}, cond {args.cond or '(head off)'}(own)@1, "
          f"f = 0 (onset within the next {cfg['slot_size']} frames)")
    print(f"{'body':44s} {'fires':>8s} {'hits':>6s} {'conf':>6s} {'lift':>6s} {'xcond':>6s} {'recall':>6s}"
          f" | {'val conf':>8s} {'val rec':>7s}")
    for name in tr:
        t, v = tr[name], va[name]
        print(f"{name:44s} {t['fires']:8d} {t['hits']:6d} {fmt(t['confidence'])} {fmt(t['lift'], 1):>6s} "
              f"{fmt(t['lift_over_cond'], 2):>6s} {fmt(t['recall'])} | {fmt(v['confidence']):>8s} {fmt(v['recall']):>7s}")
    print(f"  -> {out_file}")


if __name__ == "__main__":
    main()
