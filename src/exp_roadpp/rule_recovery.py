"""Recovery of the SUMO ground-truth rules in the rule lab's learned rules (2026-11-26).

The injected rules (R1-R3, R5; `synth/sumo/rules.py`) and the natural ones of the traffic (N1, N3, N4) each name a
head and a cue. For every rule-lab result json (the head parts of one run), per GT rule: the learned rules of the
head (the per-head selection, the fit --mode, kept: |w| >= 0.05) that contain the cue, and the best of them:
its rank among the head's kept rules by weight and by evidence (w x sqrt(train onsets)), its weight, train precision
and onsets. A GT rule counts as recovered if a matching rule is among the head's top --top by either order.

    python -m src.exp_roadpp.rule_recovery --results "<output_dir>/rule_lab/*_vt_p*.json" [--out recovery.md]
Reads only the result jsons (no data, no GPU).
"""
import argparse
import glob
import json
import math
import re

# (id, description, heads (regex of the head name), cue (regex a literal must match), extra (regex any literal of the
# rule may match, optional context; None = none needed))
GT_RULES = [
    ("R1", "a pedestrian starts crossing in view while the ego moves -> AV-Brake",
     r"av:AV-Brake", r"^Ped:(xing|Xing|XingFmLft|XingFmRht)\(other\)", None),
    ("R2", "a vehicle's HazLit onset -> its own Brake",
     r"(Car|MedVeh|LarVeh|Bus|Mobike):Brake", r"^{cls}:HazLit\(own\)", None),
    ("R3", "the vehicle ahead in the ego's lane starts braking while the ego moves -> AV-Brake",
     r"av:AV-Brake", r"^(veh|Car|MedVeh|LarVeh|Bus|Mobike|Cyc):Brake\(ahead\)", None),
    ("R5", "a waiting / crossing pedestrian in view -> AV-Brake (within 15 s)",
     r"av:AV-Brake", r"^Ped:(Wait2X|xing|Xing)\(other\)", None),
    ("N1", "the vehicle ahead starts (MovAway) while the ego stands -> AV-Mov",
     r"av:AV-Mov", r"^(veh|Car|MedVeh|LarVeh|Bus|Mobike|Cyc):MovAway\(ahead\)", None),
    ("N3", "the ego's light turns Red while it moves -> AV-Brake",
     r"av:AV-Brake", r"^(started:)?TL:Red\(other\)", None),
    ("N4", "the ego's light turns Green while it stands -> AV-Mov",
     r"av:AV-Mov", r"^(started:)?TL:Green\(other\)", None),
]


def load(pattern):
    """{head: [rules of its selected variant]} over the result files of one run."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        sel = d.get("selected") or {}
        for h in d["per_head"]:
            v = sel.get(h) or next(iter(d["rules"]))
            out[h] = [r for r in d["rules"][v] if r["head"] == h]
    return out


def strip(lit):
    """The literal without its slot; None for a negated one (not a cue)."""
    return None if lit.startswith("!") else re.sub(r"@\d+$", "", lit)


def recovery(rules_of, mode, top):
    rows = []
    for gid, desc, head_re, cue_re, _extra in GT_RULES:
        for head, rules in sorted(rules_of.items()):
            m = re.fullmatch(head_re, head)
            if not m:
                continue
            cue = re.compile(cue_re.format(cls=head.split(":")[0]))
            kept = [r for r in rules if mode in r["soft"] and abs(r["soft"][mode]["weight"]) >= 0.05]
            pos = [r for r in kept if r["soft"][mode]["weight"] > 0]
            by_w = sorted(pos, key=lambda r: -r["soft"][mode]["weight"])
            by_e = sorted(pos, key=lambda r: -r["soft"][mode]["weight"] * math.sqrt(r["events"]))
            match = [r for r in pos if any(s and cue.search(s) for s in map(strip, r["literals"]))]
            if not match:
                rows.append({"gt": gid, "desc": desc, "head": head, "kept": len(kept), "found": False})
                continue
            best = min(match, key=lambda r: by_e.index(r))
            rows.append({"gt": gid, "desc": desc, "head": head, "kept": len(kept), "found": True,
                         "matches": len(match), "rank_w": by_w.index(best) + 1, "rank_e": by_e.index(best) + 1,
                         "recovered": min(by_w.index(best), by_e.index(best)) < top,
                         "rule": " & ".join(best["literals"]), "weight": best["soft"][mode]["weight"],
                         "precision": best["precision"], "onsets": best["events"]})
    return rows


def markdown(rows, title, top):
    lines = [f"### {title}", "", f"recovered = a matching rule in the head's top {top} (by weight or evidence)", "",
             "| GT | head | recovered | rank w / ev | rule | w | precision | onsets | matching rules / kept |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["found"]:
            lines.append(f"| {r['gt']} | {r['head']} | {'yes' if r['recovered'] else 'no'} | {r['rank_w']} / {r['rank_e']} "
                         f"| `{r['rule']}` | {r['weight']:+.2f} | {r['precision']:.2f} | {r['onsets']} "
                         f"| {r['matches']} / {r['kept']} |")
        else:
            lines.append(f"| {r['gt']} | {r['head']} | no (no rule with the cue) | - | - | - | - | - | 0 / {r['kept']} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, action="append", help="glob of one run's result jsons (repeatable)")
    p.add_argument("--mode", default="mdls_0.1", help="the fit whose kept rules and weights are read")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out", default=None, help="markdown file")
    args = p.parse_args()
    parts = []
    for pattern in args.results:
        rows = recovery(load(pattern), args.mode, args.top)
        parts.append(markdown(rows, pattern, args.top))
        n = len({r["gt"] for r in rows})
        rec = len({r["gt"] for r in rows if r.get("recovered")})
        print(f"{pattern}: {rec} of {n} GT rules recovered (in some head's top {args.top})")
    text = "\n\n".join(parts)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
