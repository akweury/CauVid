# `exp_august` Pipeline Design Specification

**Status:** living design document  
**Last updated:** 2026-08-12  
**Diagram:** [`EXP_AUGUST_CLOSED_LOOP_FLOWCHART.pdf`](./EXP_AUGUST_CLOSED_LOOP_FLOWCHART.pdf)  
**Executable entry point:** [`src/exp_august/pipeline.py`](../../src/exp_august/pipeline.py)

## 1. Purpose and authority

This document preserves how every stage is intended to work. The flowchart is
the compact visual overview; this specification is the authoritative place for
implementation choices, input/output contracts, current repository status,
and unresolved design decisions.

The target system receives video only during inference. Human segmentation is
not available to perception, tracking, estimation, repair, scoring, stopping,
or model selection. It may be opened only by an independent evaluator after
the selected prediction and its manifest have been frozen.

The document deliberately distinguishes:

- **Target design:** the architecture represented by the flowchart.
- **Current repository:** what the executable pipeline actually does today.
- **Gap:** work required to make the repository match the target design.

## 2. Non-negotiable design rules

1. **Video-only inference.** No human segmentation enters Steps 1-11 or the
   closed-loop search.
2. **Independent evidence first.** Step 2 stores YOLO, SAM 2, RAFT, and depth
   outputs separately. It does not perform evidence fusion or assign persistent
   object IDs.
3. **Uncertainty is retained.** Raw observations, confidence, model versions,
   transforms, and missing-cue states remain available downstream.
4. **The LLM does not invent final physical values.** It may classify a
   failure, select constraints, and propose a bounded repair family. Numerical
   methods estimate the actual parameters.
5. **Search retains alternatives.** The loop operates on a Top-K beam and a
   best-ever register, rather than overwriting a single state in place.
6. **Evaluation cannot feed back.** Annotation-based scores are report-only.
7. **Every artifact is auditable.** Outputs carry input fingerprints, config
   hashes, model versions, parent hypothesis IDs, and evidence provenance.

## 3. Canonical data contracts

These names match the symbols in the flowchart. Exact JSON schemas should be
versioned when implementation begins.

### `VideoManifest` — $\mathcal{M}$

```text
run_id, video_id, source_path, input_hash
frame_index, timestamp_s, source_timestamp_s
fps_source, fps_canonical, width, height
decode_status, timeline_transform
config_hash, random_seed, model_versions
```

### `NeuralEvidenceStore` — $\mathcal{O}_t$

```text
frame_index, timestamp_s
detections[]: bbox, class, confidence, detection_id
masks[]: mask reference, prompt/detection reference, mask confidence
flow: forward/backward field references, validity, confidence
depth: relative/metric depth reference, validity, confidence
preprocessing transforms and model provenance
```

The records are frame-local. A detection or mask ID is not yet a persistent
track ID.

### `ObjectTrackSet` — $\mathcal{T}$

```text
track_id, primary_class
observations[]:
  frame_index, timestamp_s, detection_id
  bbox, mask reference, confidence, visibility state
  association score and per-cue evidence
gaps, birth/death state, split/merge audit, provenance
```

This is an image-space mask track. It is not yet a metric 3D trajectory.

### `GeometryHypothesis` — $\mathcal{G}$

```text
camera pose trajectory and covariance
ground/road geometry candidates
metric scale candidate and confidence interval
per-track 3D observations and covariance
calibration assumptions and residuals
```

### `WorldHypothesis` — $\mathcal{H}$

```text
hypothesis_id, parent_id, iteration
camera and metric-scale hypothesis
ego position, velocity, acceleration, yaw rate
object world trajectories and motion states
observation assignments
constraint residuals and hard-constraint status
score breakdown, repair history, provenance
```

### Loop records

```text
ResidualPacket      R_i: failed constraints, residual curves, conflict windows
EvidencePacket      E_i: keyframes, crops, plots, cue values, provenance
RepairProposal      Delta_i: bounded edits, affected window, expected effect
ScoredRanking       Q_i: hard status, score terms, rank, Top-K, delta-J
```

## 4. Target pipeline, step by step

### Step 1 — Init

**Purpose:** validate the video and create one deterministic canonical timeline.

**Inputs**

- Raw video $\mathcal{X}$.
- Frozen knowledge/configuration $\mathcal{K}$.

**Primary implementation**

- Decode-test the stream and validate duration, FPS, frame count, resolution,
  timestamps, and corrupt-frame behavior.
- Normalize timestamps onto a canonical timeline without discarding the mapping
  back to source frames.
- Freeze run configuration, random seed, input fingerprint, and model versions.
- Create the train/eval/test manifest without exposing labels to inference.

**Output:** `VideoManifest` $\mathcal{M}$ and canonical RGB frames.

**Current repository:** partially aligned. August reuses July initialization
and creates a persisted seeded split, but timeline validation/normalization is
not yet represented by a dedicated, versioned contract.

---

### Step 2 — Neural Perception

**Purpose:** extract independent neural evidence; do not track or fuse evidence.

**Inputs:** canonical RGB frame sequence and `VideoManifest`.

**Primary implementation**

- **Objects — YOLO:** per-frame boxes, classes, scores, and detection IDs.
- **Masks — SAM 2:** frame-local instance masks prompted by accepted detector
  boxes; masks retain their prompt/detection reference but receive no persistent
  object ID here.
- **Optical flow — RAFT:** adjacent-frame forward and backward dense flow,
  including forward/backward consistency and invalid regions.
- **Depth — Depth Anything 3:** per-frame depth and confidence/validity. Until a
  metric scale is resolved, depth must be explicitly marked relative.

All outputs must share a recorded coordinate transform. Model confidence must
not be collapsed into a single fused confidence in this step.

**Output:** `NeuralEvidenceStore` $\mathcal{O}_t=(D_t,S_t,F_t,Z_t,U_t)$.

**Current repository:** not yet aligned. The active Step 2 calls the July YOLO
detector. SAM 2 currently runs inside Step 3; RAFT and DA3 outputs are not
currently generated or attached to the detection payload.

---

### Step 3 — Object Tracking

**Purpose:** build ID-consistent mask tracks from frame-local evidence.

**Detailed subfigure:** [`STEP3_MULTI_EVIDENCE_TRACKING.pdf`](./STEP3_MULTI_EVIDENCE_TRACKING.pdf)
([TikZ source](./STEP3_MULTI_EVIDENCE_TRACKING.tex)).

**Inputs:** detections, masks, flow, depth, and their uncertainty from
`NeuralEvidenceStore`.

**Primary implementation**

1. **3.1 - Candidate Construction:** pair active tracks with current instances,
   retain explicit birth/gap/death alternatives, and compute a robust depth
   descriptor for every current instance from the eroded mask interior:
   $z_{j,t}=\operatorname{median}(Z_t[S_{j,t}^{\mathrm{eroded}}])$, together
   with dispersion/confidence. A marked inner-box fallback is allowed only when
   a valid mask is unavailable.
2. **3.2 - Multi-cue Association Score:** warp the preceding mask with RAFT,
   then score each feasible pair using available mask, flow, box, class, and
   depth cues. Depth consistency compares the current masked descriptor with
   the previous descriptor stored in the track history. Normalize weights over
   cues that are actually present.
3. **3.3 - Gating and Assignment:** remove infeasible pairs, run one-to-one
   Hungarian assignment, and reject matches below the configured threshold.
4. **3.4 - Track Lifecycle Update:** retain IDs for matches, create IDs for new
   instances, and preserve explicit occluded/lost states.
5. **3.5 - Object Track Set:** materialize ID-aligned masks, boxes, confidence,
   visibility state, association evidence, and provenance.

SAM 2 supplies masks; flow and depth help decide whether masks in different
frames belong to the same physical object. Step 3 owns persistent IDs. It does
not estimate metric speed or a world-space trajectory.

**Output:** `ObjectTrackSet` $\mathcal{T}$ — ID-aligned frame-wise masks,
boxes, confidence, visibility state, and association evidence.

**Current repository:** partially aligned. The implementation uses ByteTrack
bootstrap IDs, SAM 2 video propagation, and Hungarian association. Its scoring
function supports mask, flow, box, class, and depth cues with current default
weights `0.40/0.20/0.20/0.10/0.10`. Because Step 2 does not currently provide
RAFT or depth records, those two cues are normally absent and their weights are
renormalized away. In particular, the current tracker only reads an existing
scalar such as `median_depth`; it does not yet compute that scalar from the SAM
mask and DA3 map. This is a capability hook, not a completed integration.

---

### Step 4 — Geometry / Scale

**Purpose:** convert image-space observations into camera, scale, and 3D
geometry hypotheses with uncertainty.

**Inputs:** object tracks, depth, background flow/features, camera metadata, and
frozen physical priors.

**Primary implementation**

- Consolidate the frame-level masked depth descriptors already created in Step
  3 into track-aligned depth observations. Revisit the full mask-depth
  distribution, ground-contact point, confidence, and temporal outliers for
  geometric estimation; this is refinement, not the first extraction of object
  depth.
- Estimate camera motion from background support using visual odometry or SLAM,
  rejecting dynamic foreground masks.
- Estimate ground/road geometry and camera orientation.
- Generate metric-scale candidates from calibrated camera information when
  available; otherwise combine ground-plane, camera-height, and class-size
  priors without pretending the result is uniquely metric.
- Back-project object masks/anchor points into candidate 3D coordinates.
- Propagate depth, pose, mask, and scale uncertainty.

**Output:** one or more `GeometryHypothesis` records $\mathcal{G}$.

**Current repository:** partially aligned through July's `step6_positions_3d`.
Its camera-pose, scale observability, and uncertainty behavior require a
separate audit before it can satisfy this contract.

---

### Step 5 — Joint State

**Purpose:** combine geometry and tracks into initial alternative world models.

**Inputs:** $\mathcal{G}$, $\mathcal{T}$, observations, and frozen knowledge.

**Primary implementation**

- Instantiate ego and object state sequences for every viable scale/pose/data-
  association candidate.
- Estimate velocity, acceleration, heading, and yaw rate with uncertainty-aware
  smoothing; retain unsmoothed observations for residual computation.
- Keep mutually plausible alternatives in an initial Top-K beam.

**Output:** initial world hypotheses $\mathcal{H}_0$ and Top-K beam
$\mathcal{B}_0$.

**Current repository:** target stage not implemented. Current public Step 5 is
`ego_motion_abstraction`; it does not create the canonical joint
`WorldHypothesis` beam described here.

---

### Step 6 — Consistency

**Purpose:** measure whether each world hypothesis explains the entire video
and obeys physical/temporal constraints.

**Inputs:** hypothesis beam, raw neural observations, tracks, geometry, and
frozen knowledge.

**Primary implementation**

- Observation residuals: mask/box reprojection, depth agreement, and flow
  agreement.
- Ego/background checks: static ego versus background flow; forward motion,
  braking, and left/right turn signatures versus background evolution.
- Object checks: mask continuity, depth ordering, relative motion, occlusion,
  and track identity consistency.
- Physics checks: bounded speed, acceleration, jerk, curvature, and yaw-rate
  changes, using uncertainty-aware thresholds rather than single hard values.
- Semantic checks: road/vehicle/pedestrian relations that are observable from
  the video.

**Output:** `ResidualPacket` $\mathcal{R}_i$ per hypothesis.

**Current repository:** partially overlaps with trajectory refinement and its
validation logic, but there is no unified residual contract evaluated against
each `WorldHypothesis` in a Top-K beam.

---

### Step 7 — Evidence Check

**Purpose:** turn residual peaks into a compact, auditable explanation of what
failed and where.

**Inputs:** residual packets, tracks, curves, and source frames.

**Primary implementation**

- Select keyframes and temporal windows around residual peaks, state changes,
  occlusions, and cue disagreements.
- Package synchronized frames/crops, masks, flow, depth, trajectories, and
  numerical residual plots.
- Run deterministic logical checks before asking an LLM/VLM for diagnosis.
- Allow the LLM/VLM to return only a structured failure category, relevant
  constraints, and allowed repair families.

**Output:** `EvidencePacket` $\mathcal{E}_i$.

**Current repository:** target stage not implemented. Current public Step 7 is
only a handoff of relative-motion results produced inside current Step 6.

---

### Knowledge + Numerical Repair

**Purpose:** propose bounded, testable corrections without directly generating
the final trajectory.

**Primary implementation**

- LLM/VLM: diagnose likely causes such as scale error, ID switch, bad static
  background, excessive smoothing, or an incorrect motion-state transition.
- Rule engine: filter proposals against frozen allowed operations.
- Numerical solver: instantiate parameter ranges or candidates using robust
  fitting, constrained optimization, Kalman/RTS smoothing, splines, or factor
  graphs as appropriate.

**Output:** one or more `RepairProposal` records $\Delta_i$ with affected
windows, parameter bounds, expected evidence changes, and provenance.

**Current repository:** partially represented by the July 8C-8G repair stack
called from August trajectory refinement, but it is not yet connected to the
canonical evidence packet and hypothesis beam.

---

### Step 8 — Local Re-estimation

**Purpose:** apply each permitted repair only where needed and generate new
hypotheses.

**Inputs:** $\Delta_i$, parent hypotheses, affected evidence windows.

**Primary implementation**

- Re-run only the affected tracking, geometry, scale, filtering, or state
  windows while preserving boundary conditions.
- Produce multiple candidates when a repair remains ambiguous.
- Record parent ID, changed fields, changed frames, and computation budget.

**Output:** re-estimated candidates $\mathcal{H}_{i+1}^{1:n}$.

**Current repository:** target stage not implemented as an explicit public
module. Some local repair behavior exists inside current Step 6.

---

### Step 9 — Unified Score and Selection

**Purpose:** rank candidates consistently, retain the best-ever explanation,
and decide whether to continue the loop.

**Primary implementation**

1. Reject candidates that violate hard constraints or artifact contracts.
2. Rank survivors using a versioned score:

   ```text
   J(H) = w_obs*observation_error
        + w_reproj*reprojection_error
        + w_flow*background_flow_error
        + w_physics*physics_violation
        + w_semantic*semantic_violation
        + w_complexity*explanation_complexity
        + w_uncertainty*unresolved_uncertainty
   ```

3. Keep a diverse Top-K beam, not merely the K lowest near-duplicates.
4. Update the best-ever register across all iterations.
5. Stop when constraints pass, improvement is below $\epsilon$, no admissible
   repair remains, or the iteration/compute budget is exhausted.

**Outputs:** `ScoredRanking` $\mathcal{Q}_i$, next beam $\mathcal{B}_{i+1}$,
best explanation $\mathcal{H}^{*}$, and stop decision $C_i$.

**Current repository:** no canonical unified scorer, Top-K beam, best-ever
register, or common stop controller has been implemented.

---

### Freeze Best Explanation

**Purpose:** create an immutable boundary between inference and evaluation.

**Primary implementation**

- Select the best-ever hypothesis, not simply the last iteration.
- Write all physical states, tracks, uncertainty, score terms, and provenance.
- Create a frozen manifest containing code/config/model/input/output hashes.
- Close the inference process before any annotation path is opened.

**Output:** frozen world state $\mathcal{W}^{*}$.

**Current repository:** traceability artifacts exist, but there is not yet a
single enforced freeze boundary before annotation-based evaluation.

---

### Step 10 — Segmentation

**Purpose:** convert the frozen continuous state into temporally coherent motion
segments.

**Inputs:** frozen ego/object states, residual/change-point evidence, and
uncertainty.

**Primary implementation**

- Detect candidate boundaries from changes in longitudinal/lateral motion,
  yaw rate, relative object motion, and confidence.
- Merge short or redundant intervals under explicit temporal constraints.
- Assign segment labels and boundary confidence from frozen predictions only.

**Output:** final segmentation $\mathcal{Z}^{*}$.

**Current repository:** temporal segmentation is implemented as current public
Step 8, not Step 10. It also launches annotation-based test evaluation inside
the same module. Although that evaluator does not fit the prediction, the
target architecture requires evaluation to be process-isolated after freeze.

---

### Step 11 — Symbolic Scene

**Purpose:** materialize an interpretable scene representation from the frozen
segments and physical states.

**Inputs:** $\mathcal{Z}^{*}$, $\mathcal{W}^{*}$, tracks, and provenance.

**Primary implementation**

- Summarize segment-level ego and object motion.
- Select important objects using frozen, auditable criteria.
- Produce logic atoms, physical curves, confidence, and evidence links.

**Output:** symbolic scene $\mathcal{A}^{*}$ and frozen prediction package
$\mathcal{P}^{*}$.

**Current repository:** substantially aligned at public Step 11 through the
existing logic-atom materializer. Current public Steps 9 and 10 provide segment
motion abstraction and important-object selection and should become explicit
sub-stages of this target step.

---

### Independent Blind Evaluation

**Purpose:** measure frozen predictions against held-out human segmentation
without influencing inference.

**Inputs:** frozen predictions $\mathcal{P}^{*}$ and human labels $\mathcal{Y}$.

**Primary implementation**

- Verify the frozen manifest and output hashes before evaluation.
- Run in a separate command/process with read-only access to predictions.
- Report boundary F1 at declared tolerances, frame/segment F1, temporal IoU,
  confusion matrices, and per-video failure analysis.
- Never write parameters, thresholds, prompts, or states back into inference.

**Current repository:** evaluation metrics exist, but their invocation must be
moved out of current Step 8 to enforce the intended isolation boundary.

## 5. Target design versus current runner

The same number currently refers to different concepts after Step 4. Until the
runner is refactored, always use the module name as well as the step number.

| Target flowchart step | Target module | Current repository location | Alignment |
|---:|---|---|---|
| 1 | Init | Public Step 1 `dataset_initialization` | Partial |
| 2 | Neural Perception | Public Step 2 `object_detection` | YOLO only |
| 3 | Object Tracking | Public Step 3 `object_tracking` | Partial; flow/depth dormant |
| 4 | Geometry / Scale | Public Step 4 `trajectory_construction_3d` | Partial |
| 5 | Joint State | Public Step 5 is `ego_motion_abstraction` | Mismatch |
| 6 | Consistency | Public Step 6 `trajectory_refinement` | Partial overlap |
| 7 | Evidence Check | Public Step 7 is `relative_motion_representation` | Mismatch |
| 8 | Local Re-estimation | Internal portions of current Step 6 | Not explicit |
| 9 | Unified Score | No single corresponding module | Missing |
| 10 | Segmentation | Current public Step 8 | Numbering/isolation mismatch |
| 11 | Symbolic Scene | Current public Steps 9-11 | Substantially aligned |
| — | Blind Evaluation | Invoked inside current Step 8 | Must be isolated |

## 6. Current Step 3 association defaults

The present implementation calculates a normalized score over available cues:

```text
mask IoU          0.40
RAFT-flow IoU     0.20
box IoU           0.20
class consistency 0.10
depth consistency 0.10
```

These are implementation defaults, not research conclusions. Each experiment
must record both configured weights and which cues were actually present. An
experiment must not be described as flow/depth-guided when the corresponding
cue-use counters are zero.

## 7. Acceptance criteria by boundary

| Boundary | Minimum acceptance check |
|---|---|
| Step 1 → 2 | Every frame has a canonical timestamp and reversible source mapping |
| Step 2 → 3 | Evidence tensors share coordinates; missing cues are explicit |
| Step 3 → 4 | IDs, masks, confidence, gaps, and association evidence are aligned |
| Step 4 → 5 | Pose/scale alternatives include uncertainty and observability status |
| Step 5 → 6 | Every hypothesis is immutable, parented, and reproducible |
| Step 6 → 7 | Every failure identifies a constraint, time window, and evidence path |
| Repair → 8 | Every edit is bounded, permitted, and independently replayable |
| Step 8 → 9 | Parent/child diffs and compute cost are recorded |
| Step 9 → Freeze | Best-ever selection and stopping reason are auditable |
| Freeze → 10/11 | Frozen hashes verify; no annotation dependency exists |
| Predictions → Evaluation | Evaluator cannot mutate or feed back into inference |

## 8. Open design decisions

1. Decide whether SAM 2 should run strictly frame-locally in Step 2 or also
   provide a video-memory proposal to Step 3. In either case, Step 3 remains the
   owner of canonical persistent IDs.
2. Define the canonical resolution and transform policy shared by masks, RAFT,
   and depth.
3. Choose the camera-pose backend and a formal metric-scale observability test.
4. Define hard versus soft physical constraints by road context and uncertainty.
5. Define the structured LLM/VLM diagnosis schema and the complete allow-list
   of numerical repairs.
6. Define beam diversity, maximum loop budget, and score calibration without
   consulting test annotations.
7. Refactor annotation evaluation behind the frozen prediction boundary.

## 9. Decision log

| Date | Decision | Consequence |
|---|---|---|
| 2026-08-12 | Step 2 extracts independent evidence and performs no evidence fusion. | Tracking and all cue fusion begin at Step 3 or later. |
| 2026-08-12 | Step 3 is named **Object Tracking**. | Its concise responsibility is “Build ID consistent mask tracks.” |
| 2026-08-12 | Step 3 is not SAM-only. | The target matcher uses SAM masks plus available flow, box, class, and depth evidence. |
| 2026-08-12 | Object-level depth is first derived inside Step 3 from the Step 2 mask and depth map. | Step 3 uses masked depth for association; Step 4 consolidates/refines it for geometry and does not create it for the first time. |
| 2026-08-12 | Step 3 output is an image-space `ObjectTrackSet`. | Metric position, speed, and physical trajectories are deferred to Steps 4-5. |
| 2026-08-12 | TikZ is the authoritative diagram format. | Layout changes are made in the `.tex` source and exported to PDF/SVG. |

## 10. Maintenance procedure

For every architectural change:

1. Update the affected step in this document: purpose, input, method, output,
   and current repository status.
2. Update the decision log when the change resolves or reverses a design choice.
3. Update the canonical schema/version if an artifact contract changes.
4. Update the TikZ flowchart only when responsibilities, ordering, or major data
   edges change; implementation detail belongs here rather than in the diagram.
5. Update the target/current mapping when code is added or reordered.
6. Add or update tests that verify the relevant boundary acceptance check.
7. Record model/config versions and effective cue use in the run manifest.
