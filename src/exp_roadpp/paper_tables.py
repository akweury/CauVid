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
ROWS = [("prior", "Prior (base rate)"), ("lookup", "Own-label lookup"), ("anchor", "Mined rules"),
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


# Driving scenarios of the focus table: label regex over "<class>:<action>" (every agent class).
FOCUS_GROUPS = [
    ("Braking", r":Brake$"),
    ("Stopping (other agents)", r"^(?!av:).+:Stop$"),
    ("Ego stops", r"^av:AV-Stop$"),
    ("Ego starts", r"^av:AV-Mov$"),
    ("Lane change / indicator", r":(MovLft|MovRht|IncatLft|IncatRht|Ovtak)$"),
    ("Turns (other agents)", r"^(?!av:).+:(TurLft|TurRht)$"),
    ("Ego turns", r"^av:AV-(TurLft|TurRht|MovLft|MovRht)$"),
    ("Pedestrian crossing", r"^Ped:(Xing|XingFmLft|XingFmRht|Wait2X)$"),
]


def _group_heads(m, scorer, pattern):
    import re
    per_head = m["scorers"][scorer]["ap"]["per_head"]
    return {h: v for h, v in per_head.items() if re.search(pattern, h)}


def focus_table(m, events, budget, variant):
    """Per scenario: test events, then per model the AP (mean over the scenario's heads weighted by
    their onset rows) and the events detected at the budget."""
    names = [(label, scorer_name(m, base, variant)) for base, label in ROWS if base != "prior"
             and scorer_name(m, base, variant) in m["scorers"]]
    header = ["Scenario", "Heads", "Events"] + [f"AP {l}" for l, _n in names] + [f"Det. {l}" for l, _n in names]
    rows = []
    for g, pattern in FOCUS_GROUPS:
        import re
        ev = [e for e in events if re.search(pattern, e["label"])]
        aps = []
        for _l, nm in names:
            heads = {h: v for h, v in _group_heads(m, nm, pattern).items() if v["ap"] is not None and v["onset_rows"] > 0}
            w = sum(v["onset_rows"] for v in heads.values())
            aps.append(fmt(sum(v["ap"] * v["onset_rows"] for v in heads.values()) / w) if w else "–")
        n_heads = len({e["label"] for e in ev})
        det = [str(sum(e.get(f"{nm}@{budget}") not in (None, "", "None") for e in ev)) for _l, nm in names]
        rows.append([g, str(n_heads), str(len(ev))] + aps + det)
    return header, rows


def focus_head_table(m, events, budget, variant, min_events=5):
    """The heads of the focus scenarios with >= min_events test events: AP and detections per model."""
    import re
    names = [(label, scorer_name(m, base, variant)) for base, label in ROWS if base != "prior"
             and scorer_name(m, base, variant) in m["scorers"]]
    base = m["scorers"][names[-1][1]]["ap"]["per_head"]
    header = ["Scenario", "Head", "Events"] + [f"AP {l}" for l, _n in names] + [f"Det. {l}" for l, _n in names]
    rows = []
    for g, pattern in FOCUS_GROUPS:
        for h in sorted(base):
            if not re.search(pattern, h):
                continue
            ev = [e for e in events if e["label"] == h]
            if len(ev) < min_events:
                continue
            rows.append([g, h, str(len(ev))] + [fmt(m["scorers"][nm]["ap"]["per_head"].get(h, {}).get("ap")) for _l, nm in names]
                        + [str(sum(e.get(f"{nm}@{budget}") not in (None, "", "None") for e in ev)) for _l, nm in names])
    return header, rows


def focus_ablation_table(main_runs, ablations, budget, variant):
    """The rule net's AP per focus scenario in every variant (Δ to the main run in brackets)."""
    def group_ap(m, pattern):
        nm = scorer_name(m, "model", variant)
        heads = {h: v for h, v in _group_heads(m, nm, pattern).items() if v["ap"] is not None and v["onset_rows"] > 0}
        w = sum(v["onset_rows"] for v in heads.values())
        return sum(v["ap"] * v["onset_rows"] for v in heads.values()) / w if w else None

    def mean_ap(runs, pattern):
        return _mean_std([group_ap(m, pattern) for m, _e in runs])[0]
    header = ["Variant"] + [g for g, _p in FOCUS_GROUPS]
    ref = [mean_ap(main_runs, p) for _g, p in FOCUS_GROUPS]
    rows = [["Full model"] + [fmt(r) for r in ref]]
    for label, runs in ablations:
        vals = [mean_ap(runs, p) for _g, p in FOCUS_GROUPS]
        rows.append([label] + [fmt(v) + (f" ({v - r:+.3f})" if v is not None and r is not None else "")
                               for v, r in zip(vals, ref)])
    return header, rows


# Head classes (`predictability_table`): a head is "near chance" if the best model's AP is below
# CHANCE_LIFT x its base rate (the prior's AP: a constant score's AP is the base rate); else a model
# "wins" if its AP is >= WIN_RATIO x the next best and >= WIN_ABS above it; else the models are tied.
# Also unpredictable if the best AP is below MIN_AP: for a very rare head a large lift over its tiny
# base rate (e.g. 0.007 vs 0.0003) is still useless.
CHANCE_LIFT, MIN_AP, WIN_RATIO, WIN_ABS, MIN_EVENTS, FEW_EVENTS = 5.0, 0.03, 1.25, 0.02, 5, 20
COMPARED = [("lookup", "lookup"), ("mlp", "MLP"), ("model", "rule net")]


def classify_heads(m, events, budget, variant):
    """[{head, scenario, events, base, ap {model}, det {model}, best, lift, cls}] of the heads with
    >= MIN_EVENTS test events."""
    import re
    names = [(short, scorer_name(m, base, variant)) for base, short in COMPARED if scorer_name(m, base, variant) in m["scorers"]]
    if "prior" not in m["scorers"]:
        return []                                  # no base rate: no classes
    prior = m["scorers"]["prior"]["ap"]["per_head"]
    by_head = {}
    for e in events:
        by_head.setdefault(e["label"], []).append(e)
    out = []
    for h, ev in by_head.items():
        if len(ev) < MIN_EVENTS or prior.get(h, {}).get("ap") is None:
            continue
        base = prior[h]["ap"]
        ap = {short: m["scorers"][nm]["ap"]["per_head"].get(h, {}).get("ap") or 0.0 for short, nm in names}
        det = {short: sum(e.get(f"{nm}@{budget}") not in (None, "", "None") for e in ev) for short, nm in names}
        order = sorted(ap, key=lambda k: -ap[k])
        best, second = ap[order[0]], ap[order[1]] if len(order) > 1 else 0.0
        lift = best / base if base else None
        if lift is None or lift < CHANCE_LIFT or best < MIN_AP:
            cls = "unpredictable"
        elif best >= WIN_RATIO * second and best - second >= WIN_ABS:
            cls = f"{order[0]} best"
        else:
            tied = [k for k in order if ap[k] * WIN_RATIO >= best or best - ap[k] < WIN_ABS]
            cls = ("tie incl. lookup" if "lookup" in tied else "tie MLP / rule net") + " (" + ", ".join(tied) + ")"
        scenario = next((g for g, pattern in FOCUS_GROUPS if re.search(pattern, h)), "other")
        out.append({"head": h, "scenario": scenario, "events": len(ev), "base": base, "ap": ap, "det": det,
                    "best": best, "lift": lift, "cls": cls})
    return sorted(out, key=lambda r: (r["cls"], -r["events"]))


def predictability_table(m, events, budget, variant):
    """Summary: per class the heads, their test events and the events each model detects at the budget."""
    rows_h = classify_heads(m, events, budget, variant)
    models = [short for _b, short in COMPARED]
    header = ["Class", "Heads", "Events"] + [f"Det. {k}" for k in models] + ["Heads (test events)"]
    groups = {}
    for r in rows_h:
        key = r["cls"].split(" (")[0]
        groups.setdefault(key, []).append(r)
    order = ["rule net best", "MLP best", "lookup best", "tie MLP / rule net", "tie incl. lookup", "unpredictable"]
    keys = sorted(groups, key=lambda k: order.index(k) if k in order else len(order))
    rows = []
    for k in keys:
        g = groups[k]
        rows.append([k, str(len(g)), str(sum(r["events"] for r in g))]
                    + [str(sum(r["det"].get(x, 0) for r in g)) for x in models]
                    + [", ".join(f"{r['head']} ({r['events']}{'*' if r['events'] < FEW_EVENTS else ''})" for r in g)])
    return header, rows


def predictability_head_table(m, events, budget, variant):
    rows_h = classify_heads(m, events, budget, variant)
    models = [short for _b, short in COMPARED]
    header = ["Head", "Scenario", "Events", "Base rate"] + [f"AP {k}" for k in models] + ["Best / base"] + [
        f"Det. {k}" for k in models] + ["Class"]
    rows = [[r["head"], r["scenario"], str(r["events"]) + ("*" if r["events"] < FEW_EVENTS else ""), f"{r['base']:.4f}"]
            + [fmt(r["ap"].get(k)) for k in models] + [f"{r['lift']:.0f}x" if r["lift"] else "–"]
            + [str(r["det"].get(k, 0)) for k in models] + [r["cls"]] for r in rows_h]
    return header, rows


def _head_ap(m, head):
    """The rule net's AP of one head (raw scores: the per-head calibration is monotone, the same AP)."""
    v = m["scorers"]["model"]["ap"]["per_head"].get(head, {}).get("ap")
    return v


def variant_head_tables(main_runs, ablations, budget, variant):
    """The ablation per head instead of averaged: the learnable heads (Table 5: not unpredictable, >=
    MIN_EVENTS test events in the main run), the rule net's AP in every variant. A head counts as better
    / worse in a variant if its AP changes by more than its noise (the spread of the AP over the runs of
    the multi-run variants, e.g. the random filter draws), WIN_ABS and WIN_RATIO. Returns the summary
    (per variant: heads better / same / worse and which) and the per-head matrix."""
    m0, ev0 = main_runs[0]
    heads = [r for r in classify_heads(m0, ev0, budget, variant) if r["cls"] != "unpredictable"]
    heads.sort(key=lambda r: (r["scenario"], -r["events"]))
    ref = {r["head"]: _mean_std([_head_ap(m, r["head"]) for m, _e in main_runs])[0] for r in heads}
    noise = {}
    for _label, runs in ablations:
        if len(runs) > 1:
            for r in heads:
                v = [x for x in (_head_ap(m, r["head"]) for m, _e in runs) if x is not None]
                if len(v) > 1:
                    noise[r["head"]] = max(noise.get(r["head"], 0.0), max(v) - min(v))
    singles = [(label, runs) for label, runs in ablations]

    def verdict(h, v):
        r0 = ref[h]
        if v is None or r0 is None:
            return 0
        margin = max(noise.get(h, 0.0), WIN_ABS)
        if v - r0 > margin and v >= WIN_RATIO * r0:
            return 1
        if r0 - v > margin and r0 >= WIN_RATIO * v:
            return -1
        return 0
    sum_header = ["Variant", "Better", "Same", "Worse", "Heads better (Δ AP)", "Heads worse (Δ AP)"]
    sum_rows = []
    for label, runs in singles:
        vals = {r["head"]: _mean_std([_head_ap(m, r["head"]) for m, _e in runs])[0] for r in heads}
        verd = {h: verdict(h, v) for h, v in vals.items()}
        better = [h for h, x in verd.items() if x > 0]
        worse = [h for h, x in verd.items() if x < 0]
        sum_rows.append([label, str(len(better)), str(len(heads) - len(better) - len(worse)), str(len(worse)),
                         ", ".join(f"{h} ({vals[h] - ref[h]:+.3f})" for h in better) or "–",
                         ", ".join(f"{h} ({vals[h] - ref[h]:+.3f})" for h in worse) or "–"])
    mat_header = ["Scenario", "Head", "Events", "Main", "Noise"] + [label for label, _r in singles]
    mat_rows = []
    for r in heads:
        h = r["head"]
        row = [r["scenario"], h, str(r["events"]) + ("*" if r["events"] < FEW_EVENTS else ""), fmt(ref[h]),
               fmt(noise.get(h)) if h in noise else "–"]
        for _label, runs in singles:
            v = _mean_std([_head_ap(m, h) for m, _e in runs])[0]
            x = verdict(h, v)
            row.append(fmt(v) + (" ▲" if x > 0 else " ▼" if x < 0 else ""))
        mat_rows.append(row)
    return (sum_header, sum_rows), (mat_header, mat_rows)


# -------------- Learning curve --------------
SATURATION = 0.9                        # a head is saturated at the smallest share reaching this share of its full-data AP


def curve_runs(main_runs, ablations):
    """[(train share, label, runs)] of the main run (share 1) and the variants trained on a share of the
    train videos (step 07 "train_fraction" < 1), smallest share first."""
    out = [(main_runs[0][0].get("train_fraction", 1.0), "100%", main_runs)]
    for label, runs in ablations:
        f = runs[0][0].get("train_fraction", 1.0)
        if f < 1:
            out.append((f, label, runs))
    return sorted(out, key=lambda x: x[0])


def _ap_of(m, base, variant, head=None):
    nm = scorer_name(m, base, variant)
    s = m["scorers"].get(nm)
    if s is None:
        return None
    return s["ap"]["micro"] if head is None else s["ap"]["per_head"].get(head, {}).get("ap")


def learning_curve_table(points, budget, variant):
    header = ["Train share", "Train videos", "AP rule net", "mAP rule net", "AP MLP", "mAP MLP", "AP lookup",
              "Recall rule net", "Recall MLP", "FA/min rule net"] + [f"AP {g}" for g, _p in FOCUS_GROUPS]
    rows = []
    for f, _label, runs in points:
        m = runs[0][0]
        rv = row_values(m, scorer_name(m, "model", variant), budget) or {}
        mv = row_values(m, scorer_name(m, "mlp", variant), budget) or {}
        groups = []
        for _g, pattern in FOCUS_GROUPS:
            heads = {h: v for h, v in _group_heads(m, scorer_name(m, "model", variant), pattern).items()
                     if v["ap"] is not None and v["onset_rows"] > 0}
            w = sum(v["onset_rows"] for v in heads.values())
            groups.append(fmt(sum(v["ap"] * v["onset_rows"] for v in heads.values()) / w) if w else "–")
        rows.append([f"{f:.0%}", str(m.get("num_train_videos", "–")), fmt(rv.get("ap")), fmt(rv.get("map")),
                     fmt(mv.get("ap")), fmt(mv.get("map")), fmt(_ap_of(m, "lookup", variant)), fmt(rv.get("recall")),
                     fmt(mv.get("recall")), fmt(rv.get("fa"), p=1)] + groups)
    return header, rows


def saturation_table(points, budget, variant):
    """Per learnable head (Table 5 of the full-data run): the rule net's AP at every train share (with the
    head's train onsets at that share), and the smallest share (and its train onsets) where the rule net
    and the MLP reach SATURATION of their full-data AP."""
    m_full, ev_full = points[-1][2][0]
    heads = [r for r in classify_heads(m_full, ev_full, budget, variant) if r["cls"] != "unpredictable"]
    heads.sort(key=lambda r: (r["scenario"], -r["events"]))
    header = ["Scenario", "Head", "Test events"] + [f"{f:.0%}" for f, _l, _r in points] + [
        f"{SATURATION:.0%} rule net at", f"{SATURATION:.0%} MLP at"]
    rows = []
    for r in heads:
        h = r["head"]
        cells, reach = [], {}
        for base in ("model", "mlp"):
            full = _ap_of(m_full, base, variant, h)
            reach[base] = "–"
            if full:
                for f, _l, runs in points:
                    v = _ap_of(runs[0][0], base, variant, h)
                    if v is not None and v >= SATURATION * full:
                        n = runs[0][0].get("train_onsets", {}).get(h)
                        reach[base] = f"{f:.0%}" + (f" ({n} onsets)" if n is not None else "")
                        break
        for f, _l, runs in points:
            m = runs[0][0]
            v = _ap_of(m, "model", variant, h)
            n = m.get("train_onsets", {}).get(h)
            cells.append(fmt(v) + (f" ({n})" if n is not None else ""))
        rows.append([r["scenario"], h, str(r["events"]) + ("*" if r["events"] < FEW_EVENTS else "")] + cells
                    + [reach["model"], reach["mlp"]])
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
    tables.append((f"Table 4: driving scenarios, AP (onset-weighted over the scenario's heads) and events detected ({sub})",
                   *focus_table(m0, main_runs[0][1], budget, args.variant)))
    tables.append((f"Table 4b: the scenario heads with >= 5 test events ({sub})",
                   *focus_head_table(m0, main_runs[0][1], budget, args.variant)))
    note = (f"unpredictable: best AP < {MIN_AP:g} or < {CHANCE_LIFT:g}x base rate; a model is best with AP >= {WIN_RATIO:g}x and "
            f">= {WIN_ABS:g} above the next; heads with >= {MIN_EVENTS} test events, * fewer than {FEW_EVENTS}")
    tables.append((f"Table 5: heads by predictability and the best model ({sub}; {note})",
                   *predictability_table(m0, main_runs[0][1], budget, args.variant)))
    tables.append((f"Table 5b: every head of Table 5 ({sub})",
                   *predictability_head_table(m0, main_runs[0][1], budget, args.variant)))
    if ablations:
        tables.append((f"Table 4c: rule net AP per scenario in every variant ({sub})",
                       *focus_ablation_table(main_runs, ablations, budget, args.variant)))
    if ablations:
        (h6, r6), (h6b, r6b) = variant_head_tables(main_runs, ablations, budget, args.variant)
        rule = (f"rule net AP per learnable head (Table 5: not unpredictable, >= {MIN_EVENTS} test events); better / "
                f"worse: |Δ| > max(noise, {WIN_ABS:g}) and >= {WIN_RATIO:g}x; noise: AP spread over the runs of the "
                f"multi-run variants (random filter draws)")
        tables.append((f"Table 6: ablation per head ({sub}; {rule})", h6, r6))
        tables.append((f"Table 6b: rule net AP per head and variant (▲ better, ▼ worse; * < {FEW_EVENTS} test events)",
                       h6b, r6b))
    points = curve_runs(main_runs, ablations)
    if len(points) > 1:
        tables.append((f"Table 7: learning curve, models trained on a share of the train videos ({sub})",
                       *learning_curve_table(points, budget, args.variant)))
        tables.append((f"Table 7b: rule net AP per learnable head and train share (train onsets of the head in "
                       f"brackets), and the share where the rule net / MLP reach {SATURATION:.0%} of their "
                       f"full-data AP ({sub})", *saturation_table(points, budget, args.variant)))
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
