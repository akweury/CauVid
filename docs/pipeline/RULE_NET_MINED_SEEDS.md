# Rule net: mined seeds, negated / group literals, 2 s clauses (main model since 2026-11-20)

## Why

The SUMO v7 rule-mining study (`rule_tree.py`, `onset_cue_diag.py`, `residual_clips.py`) used the ego's
AV-Brake and AV-Stop heads. It found what the rule net needs to express the injected rules:

| Finding | Evidence (v7 val) |
|---|---|
| The lead vehicle must be the same agent (role `ahead`) | R3 "front brakes -> ego brakes": precision 0.23 -> 0.46 |
| Lead box size / closing speed and ego speed (`ego_motion`) | AV-Brake AP 0.34 -> 0.44; "front stopped and not far -> stop" 0.80 |
| Negated literals ("has not stopped", "no pedestrian crossed 2-3 s ago") | +0.1..0.3 precision on R3 / R1 |
| "Any motor vehicle X" | R3 coverage 46% -> 55% at the same precision |
| 2 s instead of 1 s for a clause hit | R3 precision 0.46 -> 0.61 (the rule's 0.5-1.5 s lag) |
| Short mined rules hold on val | R1 and R3 recovered with <= 3 literals; train ~= val |

Traffic light atoms (`ego_motion` v2/v3) showed no effect on AV-Brake / AV-Stop. They stay in the inputs.

## What changed

- **`cauvid.TemporalRuleNet`** (`negation`, `groups`): rule bodies are conjunctions of literals.
  - Literals: `X`, `!X` (known absent), `G` / `!G` (a group in one non-own scope and slot).
  - `literal_values(inputs, f)` gives the literal truth for counting; `literal_info` / `literal_of` decode and encode.
  - `seed_rules_mined` seeds one rule per mined seed.
  - Without negation and groups the model is unchanged (unit test: identical forward).
- **`step05_mine.py`**: per learned head, mines rules on a weighted sample of the train rows.
  - Roots are own-agent literals (so the self gate keeps every rule).
  - Rule list: depth-3 tree, every node a candidate, FOIL gain, a "not" branch becomes the complementary literal.
  - A rule is kept if precision >= 2x the root's and it holds on val (>= 1.5x the root's val precision).
  - Cached in `mined_seeds_*.json` in the run's `step05_output`; the log prints the top rules.
  - Since m2: every root is also a seed ("root": True), the head's default rule. A head too rare to mine still
    gets its root. (Run a4e02b, m1: the 52 heads without a mined rule had only free rules, which start with every
    gate closed and a self gate ~0, get no gradient, and ended at AP ~0; macro AP fell 0.199 -> 0.173.)
- **`seed_rules_mined`** (since m2):
  - The 2nd+ free rules of a head learn only a random half of the literals (`rule_inputs`); before, the
    free rules of a head grew the same body (3x `AV-Mov@2` on AV-Brake).
  - Training adds positive input literals only (`learn_negated=False`): negated and group literals come from
    the mining. In a4e02b, learned `!X` literals grew to ~20 per rule.
- **`step05`**:
  - `anchor_source "mined"` (default) seeds the net from the mined rules plus `free_rules_per_head` free rules.
  - Clause counts use literals and `clause_slots` (default 2).
  - Clauses carry `literals` (kind, group) and `seed`.
  - `build_rule_net` / `load_temporal_rule_net` handle old (inputs-only) checkpoints.
- **`step06`**:
  - The anchor baseline uses literals.
  - The seeded heads come from `head_mask` (it was wrong for per-head layouts).
  - The rule-set scorer uses literals.
- **`step05_rules`, `step05_causal`**: literal-aware.
- **Defaults**: `--roles ahead`, `--ego_motion True`, `--anchor_source mined`, `--negation True`,
  `--literal_groups True`, `--clause_slots 2`. Mining knobs: `--mine_min_lift`, `--mine_max_rules`,
  `--mine_roots`, `--mine_min_events`.
- **`ablation.py`**: new variants `anchors_causal`, `no_negation`, `no_groups`, `clause_1s`, `no_roles`,
  `no_ego_motion` (in preset `paper`). `slot_ranges` variants switch the literals off; they are not supported together.

## Run

```bash
./run_roadpp_docker.sh --gpu 0 -- --exp sumo_maps_v7 --machine dgx-jing --num_epochs 200
./run_roadpp_docker.sh --gpu 1 -- --exp sumo_maps_v7_unseen --machine dgx-jing --num_epochs 200
```

The first run rebuilds the training data (roles + ego motion v3), then mines (a few minutes), then trains.
Compare with the MLP on the same inputs (it gets the role and ego motion atoms too): AP / mAP and per-head wins,
then the paper table at the fixed false-alarm budget, then the recall of the injected rules (`rule_events`).

## m3 and the mining sweep (2026-11-21)

- **Reproducible sample.** Whether a row is drawn is a hash of (seed, video, frame, agent, head)
  (`step05_mine._row_uniform`). Before, the generator stream depended on the chunk and batch order: m1 and m2
  drew different rows and mined 4 vs 1 AV-Brake rules.
- **Mining settings in the names.** Non-default `mine_*` settings go into the seeds file and the model tag
  (`step05.mine_settings_tag`, e.g. `_me10_ml1.5_vl1.125`). Before, `--mine_min_lift 1.5` reused the cached
  2.0 seeds. New flag: `--mine_val_lift`.
- **`mine_sweep.py`** (`--mine_sweep`) compares mining settings without training the rule net.
  - It mines once per (`mine_min_lift`, `mine_min_events`) on the train sample. The val filter uses the even
    val videos; the rule list is scored on the odd ones.
  - It reports micro / macro AP and the macro AP by head size in train onsets.
  - The scorer is the same as the anchor baseline: max train precision of the holding rules, else the base rate.

```bash
./run_roadpp_docker.sh --gpu 0 --mine_sweep -- --exp sumo_maps_v7 --machine dgx-jing
```

## m4: sweep defaults and shrunk confidence (2026-11-21)

The first `mine_sweep` (v7, odd val videos, row-level weighted AP):

| setting | rules | heads mined | micro | macro |
|---|---|---|---|---|
| lift 2.0 / 20 onsets (m3) | 129 | 59 | 0.258 | 0.117 |
| lift 1.2 / 10 | 274 | 79 | 0.265 | 0.131 |
| lift 1.2 / 5 | 354 | 92 | 0.233 | 0.140 |

- **New defaults:** `mine_min_lift` 1.2, `mine_val_lift` 0.9, `mine_min_events` 10.
- **Shrunk confidence.** With 5 onsets the small rules' raw precision ranks them first across heads, and the micro AP falls.
  - Each seed now carries `"confidence"` = (events + k x root precision) / (events / precision + k), with k = `--mine_prior` (20).
  - The step06 anchor baseline shrinks toward the head prior with `ANCHOR_PRIOR_ROWS` = 200 rows (`BASELINE_VERSION` v2).
  - `mine_sweep --priors 0,10,20,50` scores every setting with each k.
- **Not yet used.** The rule net still seeds every rule with head weight `seed_head_weight`; it does not read the confidence yet.
