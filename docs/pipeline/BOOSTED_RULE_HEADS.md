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

- The step 05 models are reused.
- Step 06 fits the boosted head once and scores the val and test videos.
- Step 06 then recomputes its metrics, because the settings now include `boost`.
- Without `--boost_heads`, nothing changes and the cached metrics stay valid.

## Not yet

- The OnsetMLP does not get the new features yet. A fair comparison needs both sides to have them.
- The step 07 paper metrics (fixed FA/min budget) do not include "boost" yet.
- The boosted rules are not yet inside the TemporalRuleNet; for now they are a separate head scorer.
