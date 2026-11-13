"""Paper tables from the step 07 outputs (paper_metrics.json, paper_events.csv) of several runs.

  python -m src.exp_roadpp.paper_tables --exp medium --main 20261113-130159_dgx-jing_a62fe0 \
      --ablation "- grid=<run_id>" --ablation "+ ego motion=<run_id>" --budget 5

--main: the full model's run(s); several, comma separated, are seeds (mean +- std); one run gets
95% bootstrap intervals over the test videos instead. --ablation "label=run[,run...]": one variant
each. --variant cal (default): the scorers calibrated per head on val where step 06 calibrated
them (rule net, MLP, lookup; one probability threshold for all heads), raw otherwise; raw: the
raw scores.

Tables (markdown on stdout, markdown + LaTeX in --out, default output/roadpp/<exp>/paper_tables):
  1. main: every model at the false-alarm budget: cell AP (micro, macro over heads, per event type),
     event recall, false alarms / min reached on test, alarm precision, median lead, ego recall;
  2. ablation: the rule net of every variant and its difference to the main run;
  3. event counts: per event type (and the ego heads) the test events, how many each model detects,
     detected by any, by only one model, by none.
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "output" / "roadpp"
ROWS = [("prior", "Prior (base rate)"), ("lookup", "Own-label lookup"), ("anchor", "LLM anchor clauses"),
        ("mlp", "OnsetMLP"), ("model", "Rule net (ours)")]
TYPES = ("rare", "regular", "trivial")


def load_run(exp, run_id):
    d = ROOT / exp / "runs" / run_id / "step06_output"
    m = json.load(open(d / "paper_metrics.json"))
    with open(d / "paper_events.csv") as f:
        events = list(csv.DictReader(f))
    return m, events


def scorer_name(m, base, variant):
    cal = f"{base}_cal"
    return cal if variant == "cal" and cal in m["scorers"] else base


def _budget_entry(m, name, budget):
    s = m["scorers"].get(name)
    return None if s is None else s["budgets"].get(str(budget))


def row_values(m, name, budget):
    """Metric dict of one scorer in one run (None where missing)."""
    s = m["scorers"].get(name)
    if s is None:
        return None
    ap = s["ap"]
    b = _budget_entry(m, name, budget) or {}
    t = b.get("test") or {}
    return {"ap": ap["micro"], "map": ap["macro"], **{f"ap_{x}": ap["by_type"].get(x) for x in TYPES},
            "ap_ego": ap.get("ego"), "recall": t.get("recall"), "fa": t.get("fa_per_min"),
            "precision": t.get("precision"), "lead": t.get("median_lead_s"),
            "ego_recall": (t.get("ego") or {}).get("recall"), "threshold": b.get("threshold")}


def bootstrap(m, name, budget, n=1000, seed=0):
    """95% intervals of recall and false alarms / min, resampling the test videos."""
    videos = m["videos"].get(name, {}).get(str(budget))
    if not videos:
        return {}
    rng = random.Random(seed)
    rec, fa = [], []
    for _ in range(n):
        s = [rng.choice(videos) for _ in videos]
        ev, mins = sum(v["events"] for v in s), sum(v["minutes"] for v in s)
        if ev:
            rec.append(sum(v["detected"] for v in s) / ev)
        if mins:
            fa.append(sum(v["false"] for v in s) / mins)

    def ci(x):
        x = sorted(x)
        return (x[int(0.025 * len(x))], x[int(0.975 * len(x)) - 1]) if x else None
    return {"recall": ci(rec), "fa": ci(fa)}


def _mean_std(values):
    v = [x for x in values if x is not None]
    if not v:
        return None, None
    mu = sum(v) / len(v)
    return mu, (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else None


def fmt(mu, sd=None, ci=None, p=3):
    if mu is None:
        return "–"
    s = f"{mu:.{p}f}"
    if sd is not None:
        s += f" ± {sd:.{p}f}"
    elif ci is not None:
        s += f" [{ci[0]:.{p}f}, {ci[1]:.{p}f}]"
    return s


COLS = [("ap", "AP", 3), ("map", "mAP", 3), ("ap_rare", "AP rare", 3), ("ap_regular", "AP reg.", 3),
        ("ap_trivial", "AP triv.", 3), ("recall", "Recall", 3), ("fa", "FA/min", 1), ("precision", "Prec.", 3),
        ("lead", "Lead (s)", 1), ("ego_recall", "Ego recall", 3)]


def main_table(runs, budget, variant):
    header = ["Method"] + [c[1] for c in COLS]
    rows = []
    for base, label in ROWS:
        vals = [row_values(m, scorer_name(m, base, variant), budget) for m, _ in runs]
        if all(v is None for v in vals):
            continue
        vals = [v for v in vals if v is not None]
        cis = bootstrap(runs[0][0], scorer_name(runs[0][0], base, variant), budget) if len(runs) == 1 else {}
        row = [label]
        for key, _name, p in COLS:
            mu, sd = _mean_std([v[key] for v in vals])
            row.append(fmt(mu, sd, cis.get(key), p))
        rows.append(row)
    return header, rows


def ablation_table(main_runs, ablations, budget, variant):
    keys = [("ap", "AP", 3), ("map", "mAP", 3), ("ap_regular", "AP reg.", 3), ("recall", "Recall", 3),
            ("fa", "FA/min", 1), ("lead", "Lead (s)", 1)]

    def model_mean(runs):
        vals = [row_values(m, scorer_name(m, "model", variant), budget) for m, _ in runs]
        vals = [v for v in vals if v is not None]
        return {k: _mean_std([v[k] for v in vals])[0] for k, _n, _p in keys}
    ref = model_mean(main_runs)
    header = ["Variant"] + [n for _k, n, _p in keys] + [f"Δ {n}" for k, n, _p in keys if k not in ("fa",)]
    rows = [["Full model"] + [fmt(ref[k], p=p) for k, _n, p in keys] + ["" for k, _n, _p in keys if k != "fa"]]
    for label, runs in ablations:
        v = model_mean(runs)
        delta = [fmt(v[k] - ref[k], p=p) if v[k] is not None and ref[k] is not None else "–"
                 for k, _n, p in keys if k != "fa"]
        rows.append([label] + [fmt(v[k], p=p) for k, _n, p in keys] + [("+" + d if d[0] not in "-–" else d)
                                                                        for d in delta])
    return header, rows


def count_table(m, events, budget, variant):
    names = [(base, label, scorer_name(m, base, variant)) for base, label in ROWS if base != "prior"
             and scorer_name(m, base, variant) in m["scorers"]]
    groups = [(t, [e for e in events if e["type"] == t]) for t in TYPES]
    groups += [("all", events), ("ego (av:)", [e for e in events if e["ego"] == "True"])]

    def hit(e, nm):
        v = e.get(f"{nm}@{budget}")
        return v not in (None, "", "None")
    header = ["Events", "#"] + [label for _b, label, _n in names] + ["Any"] + [
        f"Only {label}" for _b, label, _n in names] + ["None"]
    rows = []
    for g, ev in groups:
        det = {nm: [hit(e, nm) for e in ev] for _b, _l, nm in names}
        any_ = [any(det[nm][i] for _b, _l, nm in names) for i in range(len(ev))]
        only = [sum(det[nm][i] and not any(det[o][i] for _b2, _l2, o in names if o != nm) for i in range(len(ev)))
                for _b, _l, nm in names]
        rows.append([g, str(len(ev))] + [str(sum(det[nm])) for _b, _l, nm in names] + [str(sum(any_))]
                    + [str(x) for x in only] + [str(len(ev) - sum(any_))])
    return header, rows


def per_head_counts(m, events, budget, variant, top=20):
    names = [(label, scorer_name(m, base, variant)) for base, label in ROWS if base != "prior"
             and scorer_name(m, base, variant) in m["scorers"]]
    by = {}
    for e in events:
        by.setdefault((e["label"], e["type"]), []).append(e)
    header = ["Head", "Type", "#"] + [label for label, _n in names]
    rows = []
    for (label, typ), ev in sorted(by.items(), key=lambda kv: -len(kv[1]))[:top]:
        rows.append([label, typ, str(len(ev))] + [
            str(sum(e.get(f"{nm}@{budget}") not in (None, "", "None") for e in ev)) for _l, nm in names])
    return header, rows


def markdown(title, header, rows):
    out = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def latex(title, header, rows):
    esc = lambda s: s.replace("±", r"$\pm$").replace("−", r"$-$").replace("Δ", r"$\Delta$").replace("%", r"\%").replace("_", r"\_")
    out = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{esc(title)}}}",
           r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}", r"\toprule", " & ".join(map(esc, header)) + r" \\",
           r"\midrule"]
    out += [" & ".join(map(esc, r)) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def main(argv=None):
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="full")
    ap.add_argument("--main", required=True, help="run id(s) of the full model, comma separated (seeds)")
    ap.add_argument("--ablation", action="append", default=[], help='"label=run_id[,run_id...]"')
    ap.add_argument("--budget", type=float, default=5)
    ap.add_argument("--variant", choices=("cal", "raw"), default="cal")
    ap.add_argument("--out", default=None)
    ap.add_argument("--root", default=str(ROOT), help="the output/roadpp folder holding <exp>/runs")
    args = ap.parse_args(argv)
    ROOT = Path(args.root)
    budget = int(args.budget) if float(args.budget).is_integer() else args.budget
    main_runs = [load_run(args.exp, r) for r in args.main.split(",")]
    ablations = []
    for a in args.ablation:
        label, ids = a.split("=", 1)
        ablations.append((label.strip(), [load_run(args.exp, r) for r in ids.split(",")]))
    m0 = main_runs[0][0]
    sub = (f"{args.exp}, W = {m0['window'] / m0['frame_rate']:g} s, {budget} false alarms / min (threshold on val), "
           f"{'calibrated' if args.variant == 'cal' else 'raw'} scores, noise heads excluded")
    tables = [(f"Table 1: models ({sub})", *main_table(main_runs, budget, args.variant))]
    if ablations:
        tables.append((f"Table 2: ablation of the rule net ({sub})",
                       *ablation_table(main_runs, ablations, budget, args.variant)))
    tables.append((f"Table 3: test events detected per model ({sub}; run {args.main.split(',')[0]})",
                   *count_table(m0, main_runs[0][1], budget, args.variant)))
    tables.append((f"Table 3b: events per head, top 20 ({sub})", *per_head_counts(m0, main_runs[0][1], budget,
                                                                                  args.variant)))
    out = Path(args.out) if args.out else ROOT / args.exp / "paper_tables"
    os.makedirs(out, exist_ok=True)
    md = "\n".join(markdown(*t) for t in tables)
    print(md)
    (out / f"tables_b{budget}_{args.variant}.md").write_text(md)
    (out / f"tables_b{budget}_{args.variant}.tex").write_text("\n".join(latex(*t) for t in tables))
    print(f"[Tables]\t {out}")


if __name__ == "__main__":
    main()
