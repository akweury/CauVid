# CauVid `exp_july` Pipeline Steps

This document describes the execution order currently implemented by
`src/exp_july/pipeline.py`. Pipeline artifacts are written below
`CAUVID_PIPELINE_OUTPUT_PATH` (or the configured `pipeline_output` directory).
One optional Weights & Biases run records step timing, compact metrics, selected
media, and audit artifacts.

## Execution overview

**High-level pipeline chart:** [PIPELINE_HIGH_LEVEL.pdf](./PIPELINE_HIGH_LEVEL.pdf)

```text
1 Init → 2 Detection → 3 Tracking → 6 3D Positions
→ 7 Compatibility Shell (empty) → 7 Train/Eval Split
→ 7A Enabled Axis Candidates → 7B Semantic-Corrected Consensus + Optimal N
→ 8 Track Repair → 8A Relative Motion → 8B Signal Evidence
→ 8C Cohort Clustering → 8D Closed-Loop Repair → 8E Validation
→ 8F Statistics → 8G Materialization → 8H Visualization
→ 8I Audit Dashboard → 8J Provenance → 8K Handoff
→ 9–18 Downstream Scaffolds
```

Steps 4 and 5 were removed. Downstream processing uses the object detections and
tracks from Steps 2 and 3 directly.

The implemented trajectory path currently runs through Step 8K. Steps 9–18
remain downstream interface scaffolds rather than production processing stages.

## Step descriptions

### Step 1 — Dataset initialization

Selects the requested driving videos, resolves the local dataset and output
roots, discovers frames/video files, and constructs the configuration passed to
detection, tracking, geometry, and ego-motion stages.

**Primary output:** selected video IDs, dataset metadata, and per-step arguments.

### Step 2 — Object detection

Runs object detection for each selected video and stores frame-level classes,
bounding boxes, and confidence scores. Valid copied caches are relocated to the
local dataset/output paths and reused.

**Output directory:** `01_driving_mini_detection/`

### Step 3 — Multi-object tracking

Associates detections across frames, assigns track IDs, and records track
summaries. It also writes track-length statistics and a histogram. Existing
per-video `tracks.json` caches are reused when valid.

**Output directory:** `02_driving_mini_tracking/`

### Steps 4–5 — Removed

These stages are intentionally absent from the current pipeline. Step 6 consumes
Step 3 tracking results.

### Step 6 — 3D object positions

Combines tracked objects with geometry/depth information to estimate per-frame
3D object positions. Valid per-video caches are relocated and loaded without
recomputation. When depth generation is required, Depth Anything is loaded once
per process/device and reused across videos; only missing frame depth maps are
inferred. Step 6 suppresses internal model/download progress while retaining its
video-level progress bar and actionable warnings.

**Output directory:** `06_driving_mini_3d_positions/`

### Step 7 — Compatibility shell and active substeps

```text
Step 6 geometry → empty Step 7 compatibility state → 4:1 video split
→ Step 7A enabled candidates → Step 7B consensus + optimal N → Step 8
```

The top-level Step 7 compatibility state remains empty when execution stops at
`max_step=7`. In longer runs, the deterministic train/evaluation split, Step 7A,
and Step 7B execute before Step 8. The former 7B–7F background-evidence and
rule-refinement branch remains disabled.

### Pre-7A — Train/evaluation split

```text
videos → deterministic SHA-256 ordering → train/eval split (4:1)
```

The split is performed at video level and saved to
`07_train_eval_split/train_eval_split.json`. Training videos fit the dense
plateau region; evaluation videos are used only for held-out scoring.

### Step 7A — Axis threshold segmentation

**High level:** Convert continuous ego `vx`/`vz` signals into cleaned motion-state candidates, evaluate plateau-middle thresholds, and return every enabled candidate with its confidence and audit evidence. Step 7A does not merge candidates.

```text
ego vx/vz → 100 N values → threshold labels → short-interruption filtering
→ temporal-segment counts → stable plateaus → candidate middle N
→ train confidence heat map → eval confidence → enabled candidate set
```

| Axis | Below `-N` | `[-N, N]` | Above `N` |
|---|---|---|---|
| `vz` | `backward` | `static` | `forward` |
| `vx` | `right` | `straight` | `left` |

Plateaus must span at least three sampled `N` values and produce more than one
temporal segment. Every retained plateau contributes its middle `N`. The 100
sampled thresholds remain audit evidence. Step 7A applies the configured
segment-count and maximum-`N` limits, then stores all enabled plateau-middle
candidates under `enabled_segmentation_candidates`; disabled candidates and
their reasons remain under `disabled_segmentation_candidates`. It marks the
final merge as `pending_step7b_consensus_merge` and emits no final ego symbols.

Before segment counting, a deterministic robust bridge filter joins equal-state
outer anchors across one or more noisy interior states. It handles simple
interruptions such as `forward → static → forward` and complex interruptions
such as `forward → backward → static → forward → backward → forward`. A bridge
is accepted only when both outer anchors meet `anchor_min_frames`, every inner
segment is within `noise_tolerance_frames`, the total interruption span is
within `bridge_total_max_frames`, the number of inner segments is within
`bridge_max_segments`, and the interruption-to-shorter-anchor ratio is within
`bridge_max_anchor_ratio`. The same operation is applied independently to
`right`, `straight`, and `left` on `vx`.

Every sampled threshold candidate and retained plateau-middle candidate stores a
frame-level label audit. Before filtering, every observed frame label has
confidence `1.0`. Unchanged post-filter labels remain at `1.0`. A relabeled
source segment receives a symmetric triangular confidence valley: confidence
decreases toward its middle and rises symmetrically toward its end. Valley
depth is `min(1, source_segment_length / minimum_long_segment_length)`, so a
segment whose length equals the minimum long length reaches zero at its middle.
Each row records original label, final label, confidence, source-segment bounds,
duration, and confidence method.

A second cleanup pass guarantees that no removable short segments remain. It
groups residual short segments into islands. Edge islands attach to their only
long neighbor. For an island between two long anchors, all possible monotonic
left/right split points are scored using duration-weighted distance between the
short-segment mean signals and the two anchor mean signals. The minimum-cost
split attaches the prefix to the left anchor and suffix to the right anchor.
When no long anchor exists, the entire sequence collapses to its dominant state;
a sequence whose total observed span is itself no more than the tolerance is
marked as unavoidably short. Assignment states, anchor means, signal distances,
selected sides, and methods are retained in the segment audit. The segment-count PNG
plots the original raw count as a dashed gray curve and the filtered count as a
solid axis-colored curve. Plateau annotations report the filtered count and the
raw count range; plateau detection uses only the filtered count.

| Configuration | Default |
|---|---:|
| `vx_seg_max_count` | 8 |
| `vz_seg_max_count` | 5 |
| `max_plateau_middle_th_vx` | 250 |
| `max_plateau_middle_th_vz` | 70 |
| `plateau_min_n_values` | 3 |
| `noise_tolerance_frames_vx` | 5 |
| `noise_tolerance_frames_vz` | 5 |
| `bridge_total_max_frames_vx` / `vz` | 15 |
| `anchor_min_frames_vx` / `vz` | 8 |
| `bridge_max_segments_vx` / `vz` | 5 |
| `bridge_max_anchor_ratio_vx` / `vz` | 0.75 |
| `filter_comparison_max_candidates` | 20 |
| `semantic_opposite_transition_penalty` | 0.5 |

Points exceeding either axis-specific limit remain visible but are disabled
and colored gray. Enabled training points fit the normalized Gaussian confidence
function `c(middle N, temporal segments)`. Evaluation points do not affect the
fit and are scored using `mean_eval_confidence`. Each confidence heat map uses
fixed axis ranges from zero to 1.2 times its axis-specific hyperparameter:
`max_plateau_middle_th_v*` on x and `v*_seg_max_count` on y.

| Output | Content |
|---|---|
| Per-video JSON | `train/<video_id>/` or `eval/<video_id>/`; all enabled and disabled `vx`/`vz` candidates, confidence, reasons, and a merge-pending marker |
| Per-eval-video PNG | `eval/<video_id>/`; 1×2 `vx`/`vz` charts showing raw pre-merge and filtered post-merge segment counts at every `N` |
| Per-eval-video signal PNG | k×2 matrix of all qualifying `vx`/`vz` thresholds, including enabled and disabled candidates, with status/reason, state-colored backgrounds, and dashed `±N` thresholds |
| Per-eval candidate filter PNGs | For each axis and its 20 smallest sampled candidate `N` values, a separate 4×1 chart: before segmentation, before confidence, after segmentation, after confidence |
| Overall PNG | 1×2 confidence heat maps with train, eval, and disabled points |
| Scatter audit | confidence surfaces, point confidence, limits, split, eval metric |

**MP4 audit visualization:** `eval/<video_id>/axis_segmentation_visualization.mp4`

**Evaluation signal chart:** `eval/<video_id>/axis_signal_segmentation.png`

**Candidate filter comparisons:**
`eval/<video_id>/candidate_filter_comparisons/{vx,vz}/candidate_*.png`

Each PNG uses four rows: before-filter segmentation, before-filter confidence,
after-filter segmentation, and after-filter confidence. Every segment in the
two segmentation rows is annotated with its state, duration, and length class.
Each confidence row contains a frame-aligned continuous Viridis field
for arbitrary decimal confidence values in `[0,1]` (`0 → 0.5 → 1`) plus the
numerical confidence curve. Bilinear color interpolation provides smooth
transitions between adjacent frames. The before-filter
confidence is uniformly `1`; the after-filter row visualizes symmetric
confidence valleys caused by label changes. `SHORT` means `duration_frames <= noise_tolerance_frames_v*`; `LONG`
means `duration_frames > noise_tolerance_frames_v*`. Short labels use a red
badge and long labels use a green badge. The same classifications are stored in
the per-chart JSON metadata as `raw_segments` and `filtered_segments`.

The signal chart uses `vx` and `vz` as its two columns. Its row count `k` is
the larger qualifying-candidate count across the two axes; if the counts differ,
unused cells are explicitly marked. Candidates are ordered by increasing `N`.
Enabled candidates have compact green status titles. Disabled candidates remain
visible with pale-red panels, compact red status titles, and a `DISABLED`
watermark. Full disabling reasons are wrapped inside the subplot instead of
being appended to its title.

The Step 7A MP4 is generated for at most the first three deterministic evaluation-split videos. The cap is configured by `step7a_axis_threshold_segmentation.visualization_max_eval_videos`. Its middle panel lists enabled candidate bars only; it does not show a final prediction. It uses a three-column
layout: original frames with synchronized ego `vx` and `vz` plots in the left
column (with bright dashed zero references), all enabled `vx` / `vz` threshold
candidates and their colored segmentation timelines stacked by increasing `N`
in the middle column, and vertically stacked all-video `vx` / `vz` plateau
scatter charts in the right column. The top of the right column lists the
current evaluation video's enabled `vx` and `vz` plateau-middle candidate
thresholds; it explicitly notes that Step 7A has not selected a final `N`.
The current evaluation video's scatter points are emphasized with cyan stars. Fixed legends above each middle-panel
axis group explain the `vx` colors (`right`, `straight`, `left`) and `vz` colors
(`backward`, `static`, `forward`). Every middle-column row identifies its
threshold and confidence, and a white marker follows the current frame. Because Step
7A retains multiple threshold plateaus, the timeline uses
the highest-confidence enabled plateau for display only; the selected display
threshold and confidence are recorded without changing pipeline decisions.

**Output directory:** `07a_ego_axis_threshold_segmentation/`

```text
07a_ego_axis_threshold_segmentation/
├── train/<video_id>/   # train JSON only
├── eval/<video_id>/    # eval JSON; visual files for at most 3 eval videos
│   └── candidate_filter_comparisons/{vx,vz}/  # 4×1 segmentation/confidence PNGs
├── all_videos_plateau_scatter.png
└── axis_threshold_segmentation_manifest.json
```

Artifacts: `<video_id>/axis_threshold_segment_counts.png` and
`all_videos_plateau_scatter.png`.

### Step 7B — Enabled-candidate consensus merge

```text
Step 7A enabled candidates → confidence-weighted frame/state evidence
→ minimum-length dynamic programming → one final vx/vz sequence
```

Step 7B is the only stage that merges candidates. Before aggregation, it applies
a deterministic semantic-confidence correction. The current rule forbids direct
`forward → backward` and `backward → forward` transitions in a `vz` candidate.
When violated, both adjacent segments receive the configured penalty; a segment
involved in multiple violations receives the compounded multiplier
`(1 - penalty)^incident_count`. Candidates satisfying every semantic rule keep
their original confidence. Original, multiplier, corrected confidence, affected
frames/segments, and rule violations are preserved for audit.

The merge multiplies each enabled candidate's train-fitted plateau confidence
by its semantic-corrected frame confidence, aggregates state evidence, and
applies deterministic minimum-segment-length dynamic programming. It outputs
final frame and segment `state`, `confidence`, `consensus`, `margin`, and
`candidate_disagreement`. If an axis has no enabled candidate, it returns
`unavailable_no_enabled_candidates` without using a disabled threshold.

After merging, Step 7B compares the final sequence with every enabled candidate.
The primary similarity is final-confidence-weighted frame-state agreement; raw
agreement, coverage, candidate confidence, lower `N`, and lower candidate ID are
deterministic tie breakers. The most similar candidate's midpoint threshold is
stored as that video's axis-specific `optimal_n`, together with every candidate
similarity and the selected candidate provenance.

Outputs are written to
`07b_ego_axis_consensus_segmentation/{train,eval}/<video_id>/final_axis_segmentation.json`.
For at most three eval videos, Step 7B writes all MP4s into one shared folder:
`07b_ego_axis_consensus_segmentation/eval_visualizations/`. Files are named
`<video_id>_final_consensus.mp4`. Each middle panel appends a yellow-bordered
`FINAL PREDICTION` bar below each axis's enabled candidates. The right panel
shows the train-fitted candidate-confidence heat map without train scatter
points and overlays only the current eval video's thresholds. A yellow star
marks the threshold whose segmentation is most similar to the Step 7B final
sequence, while a cyan X marks the eval threshold with the highest heat-map
confidence; both markers are retained when they identify the same threshold.
The corresponding middle-panel candidate bars carry large `FINAL BEST` and
`HEATMAP BEST` badges with thick lime/orange borders. A candidate selected by
both criteria receives both badges and a double border.

The dataset-level chart
`07b_ego_axis_consensus_segmentation/train_optimal_n_with_eval_scatter.png` uses
a 1×2 `vx`/`vz` layout. Each video contributes at most one point per axis at
`(optimal N, selected candidate segment count)`. The confidence heat-map
background is fitted exclusively from train-video optimal points and rendered
with a fully opaque, high-resolution Viridis map so low-confidence areas remain
colored rather than washed out to gray. Train points are not rendered as
scatter markers. Only held-out eval optimal points
are visible, overlaid as magenta diamonds without affecting the fit. Each
subplot adapts its x-range to its eval optimal-`N` range with padding; if no eval
point exists, it falls back to the train optimal-`N` range. The matching points,
train-fitted confidence model, plot limits, and held-out metrics are also saved
to `train_optimal_n_with_eval_scatter.json`.

### Step 8 — Track repair

```text
tracklets → split/clean tracks → canonical track IDs
```

### Step 8A — Relative motion

```text
canonical tracks + ego motion → relative position/velocity signals
```

### Archived Step 8 threshold epoch — Disabled

```text
pending threshold policy → activate → freeze for this run
```

This legacy activation point is retained in the code archive but is not invoked
by the current pipeline.

### Step 8B — Symbolize signals

```text
raw measurements → observable signal symbols → active/quarantined evidence
```

**Input signal fields**

| Group | Fields |
|---|---|
| Position and shape | `position_3d`, `bbox` |
| Relative motion | `rel_vx`, `rel_vz`, `rel_speed`, `has_rel_motion` |
| Detection quality | `score` |
| Provenance | `source`, `source_type` |

**Output symbols**

| Symbol group | Symbols |
|---|---|
| Identity | `track_id`, `primary_label` |
| Observable cues | `leftness`, `rightness`, `approach`, `recede`, `acceleration`, `deceleration`, `relative_static`, `relative_moving`, `relative_motion_uncertain` |
| Usefulness features | `num_observations`, `temporal_coverage_in_video`, `max_bbox_area_px`, `bbox_growth_ratio`, `min_depth`, `depth_change`, `min_abs_lateral_position`, `max_detection_score`, `max_relative_speed`, `max_observable_cue`, `approach` |
| Usefulness conditions | `short`, `tiny`, `far`, `low_detection_confidence`, `weak_cues`, `vehicle_category` |
| Usefulness decisions | `active`, `quarantine` |

### Step 8C — Trajectory clustering

```text
Step 8B symbols → symbolic tracks → cohort rules → cluster assignments
```

| Output group | Symbols |
|---|---|
| Track identity | `video_id`, `track_id`, `object_class` |
| Cluster assignment | `cohort_id`, `activated_cohort_rule` |
| Static metadata | `category`, `track_length`, `bbox_area`, `confidence`, `provenance` |
| Cluster audit | `compiled_rules`, `cohort_statistics`, `cohort_track_counts` |

Step 8C does not repair tracks.

### Step 8D — Closed-loop trajectory repair

```text
clustered tracks → pattern residuals → repair candidates → repaired tracks
```

### Step 8E — Repaired-trajectory validation

```text
original/repaired candidates → symbolic checks → validation outcomes
```

### Step 8F — Trajectory statistics

```text
validated outcomes → versioned statistics → promote or rollback
```

### Step 8G — Repaired-track materialization

```text
accepted repairs → downstream relative-motion tracks
```

### Step 8H — Repair visualization

```text
scene/cues + repair process + signal versions → MP4 + statistical PDFs
```

### Step 8I — Audit dashboard

```text
track audits + LLM records + repair results → offline dashboard
```

### Step 8J — Provenance audit

```text
Steps 8C–8I artifacts → cross-stage provenance record
```

### Step 8K — Final handoff

```text
repaired tracks + audit artifacts → downstream Step 9 input
```

**Archived and disabled:** the previous Step 8D–8I semantic-protection,
final-validation, ego-refinement, important-video, and threshold-calibration
execution path. Threshold-epoch activation is also disabled.

## Downstream rule-learning scaffold

The following stages exist in the execution interface but are currently
scaffolds or minimal placeholders. They preserve the intended data contracts but
do not yet implement the complete algorithms described by their names.

### Step 9 — Temporal segmentation

Intended to divide each video into temporally coherent scene segments.
Currently returns an empty `temporal_segments` collection.

### Step 10 — Segment-level object motion

Intended to summarize object motion within each temporal segment. Currently
returns an empty `segment_object_motion` collection.

### Step 11 — Important-object audit visualization

Important-object selection remains an empty scaffold. The active read-only audit
renders up to five MP4s from the Step 8K handoff: the left panel shows the scene,
object boxes, Step 7B ego labels, and synchronized `vx`/`vz` segmentation bars;
the right panel shows up to four current-frame objects with class, track ID,
relative-motion labels, positions, confidence, and numerical motion signals.

**Output directory:** `11_important_objects_visualization/`

### Step 12 — Logic-atom construction

Intended to convert scene and motion summaries into executable symbolic atoms.
Currently returns an empty `logic_atoms` collection.

### Step 13 — Target-head definition

Intended to define the target predicates used by rule learning. Currently
returns an empty `target_heads` collection.

### Step 14 — Temporal rule examples

Intended to construct positive/negative temporal examples from atoms and target
heads. Currently returns an empty `temporal_rule_examples` collection.

### Step 15 — Candidate-rule mining

Intended to generate candidate temporal/causal rules from the examples.
Currently returns an empty `candidate_rules` collection.

### Step 16 — Rule-pool merge and extension

Intended to deduplicate, merge, extend, and rank candidate rules. Currently
returns empty merged, extended, and ranked rule collections.

### Step 17 — Final rule selection

Intended to select the initial final rule set from the ranked pool. Currently
returns an empty `final_rules` collection with `top_k = 0`.

### Step 18 — Iterative causal refinement

Runs the configured number of refinement rounds over the selected rules. The
current scaffold records evaluation, masking, reselection, and refined-evaluation
structures but does not yet change the empty rule set.

## Caching and observability

- Steps 2, 3, 6, 7, and 8B support per-video cache reuse.
- Copied cache paths are relocated when matching local files are available.
- Step 8B cache validity includes the signal fingerprint and usefulness-policy
  version; the current usefulness policy is version 2.
- Step 8C stores signature caches, LLM audit records, runtime metrics, anomaly
  alerts, qualitative samples, and offline dashboards.
- W&B tracking is optional and fail-open. In online mode, the terminal prints
  the hosted run URL when authentication succeeds.
