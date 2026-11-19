# Rule net: per-head rules, own-agent anchors, and the next step (pre-filter)

Status: 2026-11-19. The code changes below are implemented, not yet trained on real data.
The pre-filter (section 4) is an idea, not implemented.

## 1. Why: the audit of the sumo_maps_v7 runs

Goal of the paper: the rule net (NeSy) beats the MLP on the same inputs. Current runs say otherwise:

| run | AP micro model / mlp / lookup |
|---|---|
| v7 b4ebd2 | 0.223 / 0.285 / 0.314 |
| v7 ad9e0b (`--anchor_source all`) | 0.234 / 0.302 / 0.314 |
| v7_unseen 0fed34 | 0.215 / 0.288 / 0.309 |

Recall per injected SUMO rule (events matched to `meta/firings`, budget 5 FA/min, uncalibrated):
SUMO R2 (`my HazLit -> my Brake`, fully expressible): lookup 0.92, model 0.00, mlp 0.04. The rule net
did learn `self:Car:HazLit@1s -> Car:Brake` (w 2.19) but with confidence 0.16. So the bottleneck is
the rule structure and its learning, not the inputs.

Causes found in the code:

1. **One rule node per anchor body, shared by all its heads** (118 bodies, ~16 heads each). The
   body can grow only one set of extra atoms, so heads that need different extra atoms pull against
   each other and the rules stay at 1-2 atoms with low confidence.
2. **No own-agent anchors.** `step03.build_anchor_clauses` kept only pairs of different agents
   (`body[-1] != head[-1]`, the last signature field is the tube uid) plus the self-loops `X -> X`,
   which step 04 removes (an atom is never adjacent to itself in the causal graph). step 04 output
   on v7: 1908 clauses, all "other agent X -> my Y". R2, N5 (`IncatLft -> TurLft/MovLft`),
   `Brake -> Stop`, `AV-Brake -> AV-Stop` had no correct anchor.
3. With `require_self=True` (default) a rule fires only once it holds an own-agent atom, so every
   seeded rule (other-agent body only) starts silent: the anchors gave almost no head start.

## 2. Implemented (uncommitted)

### 2.1 Per-head rules: `--per_head_rules True` (default False)

- `cauvid.TemporalRuleNet`: new saved buffers `head_mask` (A, R), the head a rule may feed, and
  `rule_inputs` (R, M), the inputs a rule may learn. Default all True; old checkpoints load with the
  defaults (`_load_from_state_dict`). `head_weights()` = `head_weight * head_mask` is used in
  `forward`, `sparsity_loss` and `extract_rules`.
- `TemporalRuleNet.seed_rules_per_head`: one rule per (anchor body, learned head, slot j, copy c),
  feeding only that head, head weight `seed_head_weight / copies`; plus `free_rules_per_head` free
  rules per learned head. `per_head_rule_count` gives the count before building the model.
- Symmetry breaking: copy 0 may learn every input, copies > 0 only a random `copy_input_share` of
  them (`rule_inputs`). A small random init of `log_alpha` is not enough: the gates use Adam with
  `gate_eps=1e-20`, i.e. sign-like steps, so identical rules move identically.
- `step05._seeded_rule_net` switches on `cfg["per_head_rules"]`; the model tag gets
  `_ph{copies}c{free}f{share}s`.
- CLI (`utils_pipe.py`, `config.py`): `--per_head_rules`, `--rule_copies` (2),
  `--free_rules_per_head` (4), `--copy_input_share` (0.5).
- Unit test (scratch, not in the repo): one head per rule, no gradient outside `head_mask` or to
  disallowed inputs, save/load round trip, old checkpoint loads with identical outputs.

### 2.2 Own-agent anchors: step 03 `ANCHOR_VERSION = "v2_self"`

- New clauses "my X -> my Y": when Y switches on for an agent (off in t, on in t + 1, agent seen in
  t), every other atom X the same agent held in the last `SELF_WINDOW = 30` frames gives one clause
  (onset based, not co-occurrence). The other-agent clauses and self-loops are unchanged.
- `step03_output/anchor_clauses_version.txt`: an old anchor file is rebuilt automatically, and
  step 04 then rebuilds its causal clauses (its file is older).
- v7 train atoms: 13,699 clauses, 1,342 own-agent (all with body and head of the same agent class).
- step 04 filters by atom adjacency in its causal graph, blind to own vs other agent, so an
  own-agent clause survives iff the same atom pair survived as other-agent. Estimate on v7:
  1908 other + 732 own = 2640 clauses, 222 bodies.

### 2.3 MLP hidden size fixed at 830

`step05.get_cfg` `mlp_hidden_size: 830` (was None = shared-rule count; with the v2 anchors that
count is 222 x 3 + 128 = 794). Each run retrains the MLP (step05_output is per run), so old runs
used 482 hidden units.

### 2.4 Size on v7 (111 learned heads)

| mode | rule nodes |
|---|---|
| shared (old) | 794 |
| per-head, `--rule_copies 1` | 8,013 |
| per-head, `--rule_copies 2` (default) | 15,582 = 2,523 (body, head) x 3 slots x 2 + 444 free |
| per-head, `--rule_copies 3` | 23,151 |

About 32x the gates of the old 482-node net; still fine on one GPU, but slower.

## 3. Run and check

```bash
./run_roadpp_docker.sh --gpu 0 -- --exp sumo_maps_v7 --machine dgx-jing --per_head_rules True --num_epochs 200
./run_roadpp_docker.sh --gpu 0 -- --exp sumo_maps_v7_unseen --machine dgx-jing --per_head_rules True --num_epochs 200
```

Check after the run:

- Recall per injected rule, ego and non-ego (`rule_events.py` covers the ego; for non-ego Brake
  events map the agent index to the SUMO vehicle via `step05.video_agent_ids` on
  `test_atoms_compressed.json` and match `meta/firings` entries with `agent == vehicle id`,
  `executed`, onset within `rule_events.MATCH` of `frame_fire`). First target: R2 recall.
- Do the copies of one (anchor, head) grow different bodies? (`extended_clauses_*.json`, group by
  `anchor` and head.)
- Is the rule net, per head, still below the own-label lookup?

## 4. Next idea: pre-filter rule nodes before training (not implemented)

Many of the 15,582 nodes can never become a rule. A safe filter follows from the extraction:

- Extraction keeps a (rule, head) clause only with `support >= min_support` (10) training hits in
  `>= min_videos` (2) videos (`step05.extract_temporal_clauses`).
- Training only adds body atoms; the anchor atom stays fixed. Adding atoms can only lower the
  support. So **support(anchor atom alone at slot j, head) is an upper bound** of the support of
  every rule grown from that node.
- Hence: before training, for every (anchor body input at slot j, head), count the head's onsets in
  the train videos (same at-risk / `min_off_frames` definition as training) at which the body input
  holds, and the number of videos. Drop the node if the count `< min_support` or the videos
  `< min_videos`. Such a node is guaranteed to be discarded at extraction.

What it removes, e.g.:

- `self:Car:Stop@1s -> Car:Stop`: the head is off in the 10 frames before its onset (at-risk), so
  the body never holds there (support 0).
- A pair that is supported at slot 1 but not at slot 3: keep only the slot-1 nodes.
- Rare other-agent clauses: step 03 creates a clause from a single co-occurrence, so many never
  reach 10 onsets.

Implementation sketch (in `_seeded_rule_net`, before `seed_rules_per_head`):

1. For every train video, take the onset rows of the learned heads (`training_rows` gives
   `targets` / `first`; the onset at window 0 is the head switching on in the next slot).
2. `model.layer_inputs` / `split_inputs` on those rows give, per input (scope, slot, atom), whether
   it holds.
3. Accumulate support (A_head x M_input) and the set of videos per pair.
4. Pass the kept (body input, head) set to `seed_rules_per_head` (skip the others), and count the
   nodes with the same filter in `per_head_rule_count`.

Caveat: the filter is for positive rules only. Inhibitors (negative head weight, "X -> not Y") need
the opposite (X often on at-risk rows without an onset); leave them to the free rules per head.

First step: compute on v7 how many of the 15,582 nodes the filter removes, then decide.

## 5. Other open issues found in the audit

- `mlp_cal` is `None` in `paper_metrics.json` of b4ebd2 and 0fed34 (no calibrated MLP to compare).
- ad9e0b `model_cal`: ego recall 0.00 at budget 2 jumps to 1.00 at budget 5, median lead 4.8-7.5 s.
  Likely a long alarm that "detects" every ego event of a video (an alarm is false only if it detects
  nothing). Check the alarm logic of `step06_episodes` / `step07_paper` before trusting ego numbers.
- Uncalibrated thresholds are >= 0.98 (one threshold across heads; `pos_weight=20` inflates scores).
- The runs with `--roles ahead,near_ped,xing_ped` (a43a9c, fd76b0) never reached step 07, so the ego
  R1 / R3 with exact roles are untested.
- Non-ego agents have no own "ahead" role and motion atoms are masked in every non-own scope
  (`step05.rule_input_mask`), so R1 / R3 for non-ego vehicles are not expressible yet.
