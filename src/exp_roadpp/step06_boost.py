"""Boosted rule heads in the main pipeline: the `brake_factors` design (2026-11-21) as a step 06 scorer.

brake_factors (AV-Brake, SUMO v7, settings chosen on val): boosted rules on image features 0.667 test AP vs
SetNet 0.654 and the rule list + logit 0.609 (`docs/pipeline/BOOSTED_RULE_HEADS.md`). Here the same rules
replace one ego head of the main rule net, scored on the step 06 rows and events:

  scorer "boost" = the TemporalRuleNet ("model") with the column of each boosted head replaced, on the ego
  agent's rows, by the boosted rules' probability. Every other head and agent keeps the rule net's score, so
  the per-head metrics of a boosted head (AP, event detection, head rank) compare it with the rule net and
  the OnsetMLP on the same rows, events and thresholds (chosen on val as for every scorer).

Fit (once per data, cached in <output_dir>/boost):
  rows: `brake_factors.collect(port=True)`: every 2nd frame t where the head was off in the min_off_frames
      before t (step 05 `at_risk`; the ego need not move), with an observed outcome; y = onset in
      t .. t + WINDOW - 1 (WINDOW 20 = the headline window). Image features and labels only (gt=False: no
      SUMO world, as on real video).
  literals: `brake_factors.search_literals` (sources "img") + `make_banks`, chosen on train.
  rules: `brake_factors.boosted_rules`, rounds stopped by the val AP.
Score: every frame (stride 1) of the val and test videos, cached per split.

One probability per row for every window (the rules are fit to the 20-frame window): read the boosted heads
at the headline window (20); the other windows only rank the same score.
"""
import os
from pathlib import Path

import torch

from src.exp_roadpp import brake_factors as bf
from src.exp_roadpp import step05, utils_data

VERSION = "boost_v1"
TRAIN_STRIDE = 2


def _tag(head):
    return head.replace(":", "_")


def _rows(step05_cfg, split, atom_file, box_file, head, stride):
    cfg = {**step05_cfg, f"{split}_atom_file": atom_file, f"{split}_box_file": box_file}
    return bf.collect(cfg, split, Path("."), head, port=True, stride=stride, use_gt=False)


def fit(step05_cfg, head, out_file, device):
    """Fit the boosted rules of one ego head on the step 05 train videos (val: the rounds) and save them."""
    rows = {}
    for split in ("train", "val"):
        data = _rows(step05_cfg, split, step05_cfg[f"{split}_atom_file"], step05_cfg[f"{split}_box_file"], head,
                     TRAIN_STRIDE)
        rows[split] = bf.subset_rows(data, data["known"])
    train, val = rows["train"], rows["val"]
    singles, pairs, triples, regions = bf.search_literals(train, val, device, sources=("img",))
    _bank, big = bf.make_banks(train, val, val, singles, pairs, triples, regions)
    use = [i for i, d in enumerate(big[0]) if "gt:" not in bf.literal_text(d)]
    res = bf.boosted_rules(train, val, val, big, device, use=use)
    descs = res["_descs"]
    n = len(descs)
    rules = [{"weight": r["weight"], "literals": [(descs[k % n]["spec"], k >= n) for k in r["literals"]],
              "text": r["text"], "rows": r["rows"], "precision": r["precision"],
              "val_precision": r["val_precision"], "rounds": r["rounds"]} for r in res["rules"]]
    model = {"version": VERSION, "head": head, "window": bf.WINDOW, "f0": res["f0"], "rules": rules,
             "best": res["best"], "curve": res["curve"], "literals": res["literals"],
             "train_rows": len(train["y"]), "val_rows": len(val["y"]), "train_base": float(train["y"].float().mean())}
    torch.save(model, out_file)
    b = res["best"]
    print(f"[Boost] {head}: {len(rules)} rules over {res['literals']} literals, {b['rounds']} rounds; "
          f"train AP {b['train_ap']:.3f}, val AP {b['val_ap']:.3f} (rows: train {len(train['y'])}, val {len(val['y'])}, "
          f"base {model['train_base']:.3f})")
    for r in sorted(rules, key=lambda r: -abs(r["weight"]) * r["rows"])[:15]:
        print(f"    w {r['weight']:+.2f} rows {r['rows']:6d} p_tr {r['precision']:.2f} p_val {r['val_precision']:.2f}: "
              f"{r['text'][:170]}")
    return model


def score(model, data):
    """(R,) probability of the boosted rules on the rows of data (`collect`)."""
    F = torch.full((len(data["y"]),), float(model["f0"]))
    cache = {}
    for r in model["rules"]:
        f = torch.ones(len(data["y"]), dtype=torch.bool)
        for spec, neg in r["literals"]:
            key = id(spec)
            if key not in cache:
                cache[key] = bf.apply_spec(data, spec)
            f &= ~cache[key] if neg else cache[key]
        F += r["weight"] * f.float()
    return torch.sigmoid(F)


def score_split(model, step05_cfg, split, atom_file, box_file):
    """{video: (T,) probability per frame t (NaN: no row: the head on, or t = 0)}, ego index and head track."""
    data = _rows(step05_cfg, split, atom_file, box_file, model["head"], 1)
    p = score(model, data)
    scores = {}
    for vi, vid in enumerate(data["video_ids"]):
        sel = data["video"] == vi
        if vid not in data["head_track"]:
            continue
        arr = torch.full((len(data["head_track"][vid]),), float("nan"))
        arr[data["t"][sel]] = p[sel]
        scores[vid] = arr
    return {"scores": scores, "ego": data["ego_index"], "track": data["head_track"]}


class BoostScorer:
    """step06 scorer (`step06._scorers`, video_scorer): the rule net's probabilities with the boosted heads'
    column replaced on the ego rows of the current video (context: video_id, rows, X)."""
    video_scorer = True

    def __init__(self, heads, device):
        self.heads = heads          # [(head index, {"scores", "ego", "track"} of val + test)]
        self.device = device
        self.checked, self.replaced, self.missing = set(), 0, set()

    def __call__(self, context, stage1):
        out = stage1.clone()
        vid = context["video_id"]
        t, n = context["rows"].T
        for a, s in self.heads:
            if vid not in s["scores"]:
                self.missing.add(vid)
                continue
            ego = s["ego"][vid]
            if (vid, a) not in self.checked:          # the same agent order and head track as the boosted rows
                track = context["X"][:, ego, a].cpu()
                if not torch.equal(track, s["track"][vid].to(track.dtype)):
                    raise RuntimeError(f"[Boost] {vid}: the ego's head track differs between step 06 and the boosted rows")
                self.checked.add((vid, a))
            p = s["scores"][vid].to(self.device)
            m = (n == ego) & (t < len(p))
            idx = torch.nonzero(m).flatten()
            pv = p[t[idx]]
            good = ~torch.isnan(pv)
            idx, pv = idx[good], pv[good]
            out[idx, :, a] = pv[:, None].to(out.dtype)
            self.replaced += len(idx)
        return out

    def report(self):
        print(f"[Boost] replaced {self.replaced} ego rows; videos without boosted scores: {len(self.missing)}")


def load_scorer(heads, step05_cfg, input_data, device):
    """The BoostScorer of the given ego heads ("av:..."): fit (cached in input_data["boost_dir"]) and the
    per-frame scores of the val and test videos (cached)."""
    out_dir = Path(input_data["boost_dir"])
    os.makedirs(out_dir, exist_ok=True)
    names = [step05._atom_name(sig) for sig in utils_data.load_json(step05_cfg["unique_atoms_file"])]
    parts = []
    for head in heads:
        if not head.startswith("av:"):
            raise ValueError(f"--boost_heads: only ego heads (av:...), got {head}")
        model_file = out_dir / f"boost_{_tag(head)}_{VERSION}.pt"
        if not step05.is_fresh(model_file, step05_cfg["train_atom_file"], step05_cfg["val_atom_file"]):
            fit(step05_cfg, head, model_file, device)
        model = torch.load(model_file, map_location="cpu", weights_only=False)
        merged = {"scores": {}, "ego": {}, "track": {}}
        for split, atom_file, box_file in (("val", step05_cfg["val_atom_file"], step05_cfg["val_box_file"]),
                                           ("test", input_data["test_atoms_file"], input_data["test_box_file"])):
            cache = out_dir / f"boost_scores_{_tag(head)}_{VERSION}_{split}.pt"
            if step05.is_fresh(cache, model_file, atom_file):
                s = torch.load(cache, map_location="cpu", weights_only=False)
            else:
                s = score_split(model, step05_cfg, split, atom_file, box_file)
                torch.save(s, cache)
            for key in merged:
                merged[key].update(s[key])
        parts.append((names.index(head), merged))
        print(f"[Boost] {head}: {len(model['rules'])} rules, fit val AP {model['best']['val_ap']:.3f}; "
              f"scores for {len(merged['scores'])} val + test videos")
    return BoostScorer(parts, device)


def model_files(heads, input_data):
    """The boosted model files (freshness of the step 06 metrics)."""
    return [Path(input_data["boost_dir"]) / f"boost_{_tag(h)}_{VERSION}.pt" for h in heads]
