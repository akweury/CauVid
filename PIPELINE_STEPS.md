# CauVid `exp_july` Pipeline Steps

This document describes the execution order currently implemented by
`src/exp_july/pipeline.py`. Pipeline artifacts are written below
`CAUVID_PIPELINE_OUTPUT_PATH` (or the configured `pipeline_output` directory).
One optional Weights & Biases run records step timing, compact metrics, selected
media, and audit artifacts.

## Execution overview

```text
1 Init → 2 Detection → 3 Tracking → 6 3D Positions → 7 Ego Motion
→ 8 Trajectory Repair → 8A Relative Motion → Threshold Epoch
→ 8B Signal Evidence → 8C Pattern/Cohort Repair → 8D Validation
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

### Step 7 — Ego-motion estimation

Estimates camera/ego motion over time from the Step 6 geometry and stores
frame-level ego-motion signals and uncertainty. Valid per-video caches are
reused.

**Output directory:** `07_driving_mini_ego_motion/`

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
| Observable cues | `leftness`, `rightness`, `approach`, `recede`, `acceleration`, `deceleration` |
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
original vs repaired tracks → MP4 + HTML + statistical PDFs
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
