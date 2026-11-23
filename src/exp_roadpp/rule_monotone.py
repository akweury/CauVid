"""Monotonicity check of the rule lab's miners (2026-11-24, the user: a good rule must stay whatever atoms are
added).

Two rule lab results on the same data and the same row sample:
  - A: fewer atoms (e.g. --lab_forbid av:lay_);
  - B: all of them.
Every rule of A whose literals B may use should be in B too. Checked per variant (rule set) present in both:
  - the short rules (<= --max_len literals) of A that are "good" (precision >= --min_prec, >= --min_events onsets);
  - missing in B: listed per head with their precision and onsets.
A rule of the "exh" miner of <= 2 literals must never be missing; the tree miners ("relaxed", "m4") can lose rules.

    python -m src.exp_roadpp.rule_monotone A.json B.json [--variant relaxed_nn] [--max_len 2]
"""
import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Rules of A missing in B (rule lab results)")
    parser.add_argument("a", help="rule lab json with fewer atoms (e.g. --lab_forbid)")
    parser.add_argument("b", help="rule lab json with all atoms")
    parser.add_argument("--variant", default="", help="rule set name (default: every set in both)")
    parser.add_argument("--max_len", type=int, default=2)
    parser.add_argument("--min_prec", type=float, default=0.3)
    parser.add_argument("--min_events", type=int, default=20)
    parser.add_argument("--show", type=int, default=10)
    args, _ = parser.parse_known_args()
    A, B = json.load(open(args.a)), json.load(open(args.b))
    names = [args.variant] if args.variant else [v for v in A["rules"] if v in B["rules"]]
    bad = 0
    for v in names:
        key = lambda r: (r["head"], tuple(sorted(r["literals"])))
        in_b = {key(r) for r in B["rules"][v]}
        good = [r for r in A["rules"][v] if len(r["literals"]) <= args.max_len and r["precision"] >= args.min_prec
                and r["events"] >= args.min_events]
        miss = [r for r in good if key(r) not in in_b]
        bad += len(miss)
        print(f"[rule_monotone] {v}: {len(good)} good rules of <= {args.max_len} literals in A, "
              f"{len(miss)} missing in B")
        per = defaultdict(list)
        for r in miss:
            per[r["head"]].append(r)
        for h, rs in sorted(per.items(), key=lambda x: -max(r["events"] for r in x[1]))[:args.show]:
            for r in sorted(rs, key=lambda r: -r["events"])[:2]:
                print(f"    {h}: p {r['precision']:.2f}, {r['events']} onsets  {' & '.join(r['literals'])}")
    print("[rule_monotone] OK" if bad == 0 else f"[rule_monotone] {bad} good rules lost")


if __name__ == "__main__":
    main()
