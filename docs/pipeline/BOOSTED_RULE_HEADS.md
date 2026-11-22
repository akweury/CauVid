# Boosted rule heads (step 06 scorer "boost", 2026-11-21)

## Why

`brake_factors.py` was the testbed for the rule design on one ego head: AV-Brake on SUMO v7. Rows were the
frames where the ego moves and does not brake; the label is an onset within 2 s. Each model's settings were chosen on val:

| Model | Test AP |
|---|---|
| Rule list (sequential covering) + logistic weights | 0.609 |
| Boosted rules, image features only | **0.667** |
| SetNet (per-agent MLP, max-pooled), image features, early-stopped | 0.654 |
| Boosted rules, image + SUMO ground truth | **0.676** |
| SetNet, image + SUMO ground truth | 0.664 |

What carries the signal:
- Label-onset literals:
  - "pedestrian crossing now & not 2 s ago" (R1);
  - "lead vehicle brakes now & not 1 s ago" (R3).

  Both have a test precision of about 0.85.
- Closing speed and lead distance.
- Interval and region literals learned on train (several ranges per feature).
- Many suppressor rules with negative weights: the ego is slow, or the lead is close and stopped.

Boosting beats sequential covering because a later rule can still re-rank the rows inside an earlier rule.

## What it does

- **`step06_boost.fit`**:
  - Rows: `brake_factors.collect(port=True)`. Every 2nd frame where the head was off in the min_off_frames before
    it (the step 05 at-risk rule), with an observed outcome. Image features and labels only, no SUMO world.
  - Literals: `search_literals(sources=("img",))` and `make_banks`. Every literal carries a `spec`, so it can be
    re-applied to any rows (`apply_spec`).
  - Rules: `boosted_rules`. The number of rounds is chosen by the val AP.
  - Output: `<output_dir>/boost/boost_<head>_boost_v1.pt`.
- **`step06_boost.score_split`**: one probability per frame of every val and test video (cached).
- **step06 scorer "boost"**:
  - The TemporalRuleNet's probabilities, except the boosted heads' column on the ego's rows, which is replaced.
  - Before replacing, it checks per video that the ego index and head track match step 06's tensor.
  - It is tuned on val like every other scorer: per-head thresholds and calibration.
  - Its per-head numbers for the boosted head compare it with the rule net and the OnsetMLP on the same rows and events.
- **Windows**: one probability for all windows. The rules are fit to the 20-frame window, so read the headline
  window (20).

## Run

```bash
./run_roadpp_docker.sh --gpu 0 -- --exp sumo_maps_v7 --machine dgx-jing --num_epochs 200 --boost_heads av:AV-Brake
```

- A new run retrains the step 05 models; `--run_id <run>` reuses them.
- Step 06 fits the boosted head once and scores the val and test videos.
- Step 06 then recomputes its metrics, because the settings now include `boost`.
- Without `--boost_heads`, nothing changes and the cached metrics stay valid.

## First result (run 1ca554, AV-Brake only, test, window 20)

| Scorer | AV-Brake AP |
|---|---|
| Rule net | 0.409 |
| OnsetMLP | 0.499 |
| Rule net + boosted AV-Brake | **0.661** |

Where the rule net trails the MLP (109 heads, rule net wins 18): heads with few mined seeds (no mined rule:
0.068 vs 0.145; 1-3: 0.196 vs 0.294; 10+: 0.347 vs 0.359). The rule net is only 0-0.03 above its seeds
(anchor), and its train loss (0.086) stays above its val loss (0.084) from epoch 9 on: it does not grow rules
beyond the mined seeds. Largest gaps: Mobike / Cyc heads, lane and direction actions (VehLane, OutgoLane,
MovTow, IncomLane).

## All heads (2026-11-22, `AGENT_VERSION` boost_a1)

`--boost_heads all` (or `agents`, `ego`, or names) boosts every head:
- Ego heads (`av:`): as above.
- Agent heads (e.g. `Car:Brake`):
  - Rows: (t, agent n) of the head's class, at risk as in step 05.
  - Groups of a row:
    - `own`: n itself;
    - `near_ped`: n's nearest pedestrian;
    - `ahead`: the ego's lead;
    - `veh` / `ped` / `tl`: some *other* agent;
    - `ego`: the scene scale and the ego's labels.
  - Every feature and label is also taken 1, 2, 3 and 5 s before.
  - Train / val rows are sampled:
    - all positives, up to 30k;
    - negatives: 4x the positives, at least 20k;
    - each row is weighted by 1 / its sampling rate, so the probabilities stay on the true onset rate.
  - Literals:
    - the bank and the 1-feature regions of every group;
    - pairs and 2-feature regions on `own`;
    - no triples: a depth-3 rule already combines `own` literals.
  - Heads with fewer than 50 positive train rows are skipped; they keep the rule net's score.
- Models and per-head val / test scores are cached in `<output_dir>/boost`.

```bash
./run_roadpp_docker.sh --gpu 0 -- --exp sumo_maps_v7 --machine dgx-jing --run_id 20261121-173934_dgx-jing_1ca554 --boost_heads all
```

## Fair neural reference: scorer "boost_net" (2026-11-22)

Each boosted head also gets a SetNet (`brake_factors.SetNet`, `NET_VERSION` / `AGENT_NET_VERSION`). It shares
these with the boosted rules:
- the same rows and sample weights;
- the same features and labels, with every lag;
- the same row groups;
- the same early-stopping criterion: the weighted val AP;
- the same step 06 plug-in: it replaces the head's column in the rule net.

The SetNet itself is a per-agent MLP for each group, max-pooled, then a row MLP. It is trained with full-batch
Adam on the weighted BCE, for up to 2000 steps, checking the val AP every 25 steps with a patience of 8 checks.

How to read the scorers:

| Comparison | What it shows |
|---|---|
| `boost` vs `boost_net` | rules vs network, on equal inputs |
| `boost_net` vs `mlp` | what the new features and rows add over the step 05 inputs |
| `boost` vs `model` | the full change to the rule net |

`--boost_heads` always produces both scorers.

## 33 heads with 0-2 rule-net clauses (run 1ca554, test, window 20)

| | rule net | MLP | boost | boost_net |
|---|---|---|---|---|
| 31 fitted heads (macro AP) | 0.097 | 0.207 | **0.219** | 0.129 |

Boost beats the rule net on 19 of the 31 heads and the MLP on 13. It wins big on Bus:IncomLane (0.64 vs MLP 0.32),
Mobike:TurLft (0.61 vs 0.18) and Cyc:XingFmRht (0.45 vs 0.21). It collapses on Cyc:IncomLane (0.00 vs rule net
0.42). 11 heads stay below AP 0.03 with every scorer.

## Per-head rule scorer on val, learnable heads (2026-11-22, `SELECTION_VERSION` sel_v2)

- **"anchor" is shown as "mined rules"** in figures, logs and paper tables. It scores a row with the
  mined seed rules alone (1-4 literals each) and each rule's train confidence, shrunk toward the head
  prior. The key stays `anchor` so the caches stay valid.
  - Run 1ca554, 5 FA/min, test recall: mined rules 0.109, MLP 0.107, rule net 0.054.
  - The direction since 2026-11-22: mine rules, then predict with the rules themselves.
- **Scorer "rule_sel"** (`step06.select_rules`, `step06_boost.SelectScorer`): for every counted head, the rule
  scorer with the highest val AP at the headline window.
  - Candidates: mined rules (`anchor`), boosted rules (`boost`, boosted heads only) and the rule net
    (`model`). Ties go to them in that order.
  - Heads without val onsets take the mined rules.
  - It is a pure rule model: the MLP and the SetNet are never candidates.
  - Its val scores are added for its thresholds and calibration.
  - The choice is saved in `step06_output/rule_sel.json` for step 07.
- **Learnable heads** (`learnable_heads`): some scorer reaches a val AP of at least `LEARNABLE_MIN_AP` (0.05).
  - The others are listed in the summary and not compared.
  - `per_head[h]` gets `val_average_precision` and `learnable`. Every window gets
    `macro_average_precision_learnable`.
- **`test_metrics.json["rule_heads"]`** is printed at the end of step 06 as `[Rule heads]`. It has:
  - for every head: the test AP of model / anchor / boost / rule_sel / boost_net / mlp, the chosen scorer,
    and whether the head is boosted and learnable;
  - the macro AP and the wins over the learnable heads, and over the learnable boosted heads.
- **Step 07** (`paper_metrics.json`): with `--boost_heads` and `rule_sel.json` present, "boost",
  "boost_net" and "rule_sel" are evaluated at the fixed FA/min budgets too.
- All of this needs `--boost_heads`, so the cached metrics of other runs stay valid.
- The step 06 metrics and step 07 are redone. The episode and ceiling stages are not, because their calibration
  is unchanged (`save_calibration`).

## Not yet

- The step 05 OnsetMLP still has only the step 05 inputs. `boost_net` is the like-for-like neural model.
- The boosted rules are not yet inside the TemporalRuleNet; for now they are a separate head scorer.
