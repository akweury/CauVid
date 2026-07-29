# CauVid `exp_july` Pipeline Steps

This document describes the execution order currently implemented by
`src/exp_july/pipeline.py`. Pipeline artifacts are written below
`CAUVID_PIPELINE_OUTPUT_PATH` (or the configured `pipeline_output` directory).
One optional Weights & Biases run records step timing, compact metrics, selected
media, and audit artifacts.

## Execution overview

```text
1 Init → 2 Detection → 3 Tracking → 6 3D Positions → 7 Ego Motion
→ 7 Empty → 7 Train/Eval Split → 7A Axis Threshold Segmentation
→ 8 Trajectory Repair → 8A Relative Motion → 8B Signal Evidence
→ 8C Pattern/Cohort Repair → 8D Validation
→ 8E Semantic Protection → 8F Final Validation → 8G Ego Refinement
→ 8H Visualization → 8I Threshold Calibration
→ 9 Segmentation → 10 Segment Motion → 11–18 Rule Learning/Refinement
```

Steps 4 and 5 were removed. Downstream processing uses the object detections and
tracks from Steps 2 and 3 directly.

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
recomputation.

**Output directory:** `06_driving_mini_3d_positions/`

### Step 7 — Empty

```text
Step 6 geometry → empty Step 7 → 4:1 video split → Step 7A → Step 8
```

Step 7 itself is empty. Step 7A is the only active Step 7 substep and obtains
the continuous ego signals internally. Step 7B is retained only as a documented
future transition-validation placeholder; the former executable 7B–7F pipeline
remains disabled.

### Pre-7A — Train/evaluation split

```text
videos → deterministic SHA-256 ordering → train/eval split (4:1)
```

The split is performed at video level and saved to
`07_train_eval_split/train_eval_split.json`. Training videos fit the dense
plateau region; evaluation videos are used only for held-out scoring.

### Step 7A — Axis threshold segmentation

```text
ego vx/vz → 100 N values → threshold labels → short-interruption filtering
→ temporal-segment counts → stable plateaus → candidate middle N
→ train confidence heat map → eval confidence
```

| Axis | Below `-N` | `[-N, N]` | Above `N` |
|---|---|---|---|
| `vz` | `backward` | `static` | `forward` |
| `vx` | `right` | `straight` | `left` |

Plateaus must span at least three sampled `N` values and produce more than one
temporal segment. Every retained plateau contributes its middle `N`; Step 7A
does not select one final threshold.

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

Points exceeding either axis-specific limit remain visible but are disabled
and colored gray. Enabled training points fit the normalized Gaussian confidence
function `c(middle N, temporal segments)`. Evaluation points do not affect the
fit and are scored using `mean_eval_confidence`. Each confidence heat map uses
fixed axis ranges from zero to 1.2 times its axis-specific hyperparameter:
`max_plateau_middle_th_v*` on x and `v*_seg_max_count` on y.

| Output | Content |
|---|---|
| Per-video JSON | `train/<video_id>/` or `eval/<video_id>/`; thresholds, segment counts, plateaus, middle `N`, candidate segments |
| Per-video PNG | `train/<video_id>/` or `eval/<video_id>/`; 1×2 `vx`/`vz` charts showing raw pre-merge and filtered post-merge segment counts at every `N` |
| Per-eval-video signal PNG | k×2 matrix of all qualifying `vx`/`vz` thresholds, including enabled and disabled candidates, with status/reason, state-colored backgrounds, and dashed `±N` thresholds |
| Per-eval candidate filter PNGs | For each axis and its 20 smallest sampled candidate `N` values, a separate 2×1 before/after short-merge comparison with state-colored backgrounds and per-segment `SHORT`/`LONG` labels |
| Overall PNG | 1×2 confidence heat maps with train, eval, and disabled points |
| Scatter audit | confidence surfaces, point confidence, limits, split, eval metric |

**MP4 audit visualization:** `eval/<video_id>/axis_segmentation_visualization.mp4`

**Evaluation signal chart:** `eval/<video_id>/axis_signal_segmentation.png`

**Candidate filter comparisons:**
`eval/<video_id>/candidate_filter_comparisons/{vx,vz}/candidate_*.png`

Every segment in both rows is annotated with its state, duration, and length
class. `SHORT` means `duration_frames <= noise_tolerance_frames_v*`; `LONG`
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

The MP4 is generated only for evaluation-split videos. It uses a three-column
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
├── train/<video_id>/   # train JSON and plateau PNG
├── eval/<video_id>/    # eval JSON, plateau PNG, signal PNG, MP4
│   └── candidate_filter_comparisons/{vx,vz}/  # 2×1 raw/filtered PNGs
├── all_videos_plateau_scatter.png
└── axis_threshold_segmentation_manifest.json
```

Artifacts: `<video_id>/axis_threshold_segment_counts.png` and
`all_videos_plateau_scatter.png`.

### Step 7B — Reserved transition-logic validation (empty)

```text
Step 7A filtered segments → transition validation (planned only)
```

Step 7B remains empty and is not called by the pipeline. It is reserved for a
future deterministic real-world transition validator that runs after the Step
7A short-segment filter. The planned legal transitions are:

| Axis | Legal adjacent-state transitions | Forbidden direct transition |
|---|---|---|
| `vz` | `forward ↔ static`, `static ↔ backward` | `forward ↔ backward` |
| `vx` | `left ↔ straight`, `straight ↔ right` | `left ↔ right` |

This is currently a design comment only: it does not modify segments, insert
neutral states, reject candidates, or change downstream results.

## Step 8 — High-level flow

### Step 8 — Track repair

```text
tracklets → split/clean tracks → canonical track IDs
```

### Step 8A — Relative motion

```text
canonical tracks + ego motion → relative position/velocity signals
```

### Step 8 threshold epoch — Freeze policy

```text
pending threshold policy → activate → freeze for this run
```

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

### Step 11 — Important-object selection

Intended to select the objects required for higher-level scene reasoning.
Currently returns an empty `important_objects` collection.

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
