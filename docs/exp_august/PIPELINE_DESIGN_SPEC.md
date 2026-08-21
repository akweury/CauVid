# `exp_august` Pipeline Design Specification

**Status:** living design document
**Last updated:** 2026-08-18
**Diagram:** [`EXP_AUGUST_CLOSED_LOOP_FLOWCHART.pdf`](./EXP_AUGUST_CLOSED_LOOP_FLOWCHART.pdf)
**Current executable entry point:** [`src/exp_august/inference/runner.py`](../../src/exp_august/inference/runner.py)

**Archived legacy baseline:** [`src/exp_august/pipeline.py`](../../src/exp_august/pipeline.py)

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

The paper-facing method is not the sequence of pretrained components by itself.
Its proposed contribution is the typed, uncertainty-aware loop spanning Steps
5-9: construct alternative world states, predict their observable video
consequences, diagnose localized failures, apply bounded repairs, and retain the
best explanation under independent evidence checks.

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
8. **No cross-video learning.** Test-video inference may optimize that video's
   latent world state, but it must not update shared weights, priors, thresholds,
   prompts, or repair policies.
9. **Physical plausibility is not truth.** A smoother or dynamically feasible
   path is not accepted unless it also improves, or does not degrade within a
   frozen tolerance, evidence that the current repair was not allowed to fit.
10. **Unobservability is a valid result.** The pipeline emits a wide or
    multimodal interval, `relative`, `ambiguous`, or `unobservable` instead of
    inventing a precise metric state.

### 2.1 Paper-facing claim

The intended method is **training-free target-video inference**. YOLO, SAM 2,
RAFT, DA3 and any optional VLM are pretrained elsewhere and frozen here. For a
video $\mathcal X_j$, the pipeline infers a posterior-like set of latent world
states without changing global parameters:

$$
p(\mathcal W_j\mid\mathcal X_j,\Theta,\mathcal K),
\qquad
(\Theta,\mathcal K)\ \text{fixed for all test videos}.
$$

This is per-video inference/optimization, not supervised training. The main
scientific question is whether evidence-grounded, physics-constrained
analysis-by-synthesis improves externally measured world-state accuracy and
calibration over open-loop perception and ordinary trajectory smoothing.

### 2.2 Development and test protocol

- **Development videos:** may be inspected while designing the architecture,
  constraints, thresholds, prompts and operator allow-list. They are not a
  gradient-training set.
- **Optional calibration videos:** may calibrate uncertainty or fixed decision
  thresholds. Any labels used here remain isolated from the blind test set.
- **Blind test videos:** run independently after code, models, configuration,
  knowledge, prompts and budgets are frozen. No state is carried across videos.
- **Held-out references:** pose, trajectory, tracking or segmentation labels
  are opened only after prediction hashes are sealed and are report-only.

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

### `EvidenceUsePlan` — $\Pi$

`EvidenceUsePlan` is frozen before world-state construction and prevents a
repair from validating itself with exactly the evidence it was optimized to
fit:

```text
plan_id, policy_version, random_seed
evidence_ref, cue_family, frame/window
role: fit | check_only | report_only
allowed_consumers, prohibited_optimizers
selection_reason, hash, provenance
```

- `fit` evidence may be used by a numerical estimator.
- `check_only` evidence may score acceptance or non-degradation, but the
  current repair cannot optimize it.
- `report_only` references include human annotation and external ground truth;
  they remain unavailable until the prediction is frozen.

Backward flow, unselected mask candidates, unmatched detections, fixed
spatiotemporal validation samples, or a frozen cue-family holdout can serve as
check-only evidence. When no independent check is possible, the corresponding
result is labeled `self_consistency_only`; it cannot support a truth claim.

### `TrackingPackage` — $\mathcal{T}$

$\mathcal{T}$ is a package, not a single lossy track table. The
`ObjectTrackSet` is one derived view inside it:

```text
input_snapshot:
  immutable references to every Step 3 input record/artifact
association_ledger:
  every candidate pair, cue value, gate, rank, assignment and rejection
unassigned_evidence:
  every unmatched/invalid detection, mask and propagated proposal
mask_candidate_bank:
  per track/frame direct, forward, backward, flow-warped and unassigned masks
  mask/logit references, confidence, observability, parent references, provenance
track_view: ObjectTrackSet
  track_id, primary_class
  observations[]:
    frame_index, timestamp_s, detection_id, proposal_id
    bbox, mask reference, confidence, tracker visibility state
    source-frame/crop, forward/backward-flow, depth/confidence references
    lightweight cue descriptors and selected association evidence
  state_markers[]:
    first_observed | matched | missed | reobserved | retired | video_end
    frame_index, tracker state, operational trigger, evidence references
artifact_manifest, transform_registry, provenance
retention_report
```

The track view is an image-space mask track. It is not yet a metric 3D
trajectory. The input snapshot and ledger prevent that convenient view from
becoming an irreversible information bottleneck.
`first_observed` and `retired` are tracker facts, not explanations of physical
object birth or disappearance. Dense arrays remain in immutable,
content-addressed Step 2 artifacts; Step 3 stores stable references and aligned
indexes rather than duplicating them.

### `GeometryHypothesis` — $\mathcal{G}$

```text
camera pose trajectory and covariance
ground/road geometry candidates
metric scale candidate and confidence interval
scale observability: metric | relative | ambiguous | unobservable
per-track 3D observations and covariance
calibration assumptions, evidence references, residuals and rank
```

### `WorldHypothesis` — $\mathcal{H}$

```text
hypothesis_id, parent_id, iteration
camera and metric-scale hypothesis
ego position, velocity, acceleration, heading, yaw rate and covariance
object world trajectories, motion states and covariance
observation assignments and evidence-use roles
fit/check/physics residuals and hard-constraint status
observability, evaluability and uncertainty intervals
score breakdown, repair history, immutable parent diff, provenance
```

### Loop records

```text
ResidualPacket      R_i: hypothesis/constraint IDs, applicability/evaluability
                         fit/check role, normalized residual curves, uncertainty,
                         hard/soft status, conflict windows, suspected components,
                         evidence references and forward-prediction artifacts
EvidencePacket      E_i: keyframes, crops, plots, cue values, provenance
RepairProposal      Delta_i: allow-listed operator, bounded edits, affected
                         variables/window, expected fit/check effects, budget
ScoredRanking       Q_i: hard status, fit/check/physics/complexity terms,
                         acceptance decision, rank, diverse Top-K, delta-J
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
- Accept dataset membership from the external experiment controller when a
  development/calibration/blind-test protocol is active. Step 1 itself never
  discovers annotations or selects membership from label availability.

**Output:** versioned `VideoManifest` $\mathcal{M}$ records defining canonical
RGB frame access. Frames may be decoded on demand through the reversible source
mapping rather than duplicated during initialization.

**Current repository:** the target implementation is available in
`src/exp_august/inference/step01_init.py` with strict Pydantic contracts under
`src/exp_august/contracts/`. It hashes raw inputs, probes timestamps and stream
metadata, normalizes display orientation and a downsampled canonical timeline,
performs configurable sample/full decode validation, and writes content-addressed
per-video manifests plus an `InitBundle`. The legacy `src/exp_august/pipeline.py`
remains unchanged as a baseline and still uses July initialization/splitting.

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

**Current repository:** target Step 2 is implemented in
`src/exp_august/inference/step02_neural_evidence.py`. A canonical frame provider
verifies Step 1 hashes and decodes the recorded source-frame mapping with the
frozen display transform. YOLO-World runs on in-memory BGR batches and retains
primary plus lower-confidence candidates. Frame-local SAM 2 uses eligible YOLO
boxes as prompts without assigning persistent IDs. RAFT-Small writes adjacent
forward/backward flow together with domain validity, consistency masks and
forward/backward residuals. DA3 runs in single-frame context and stores depth,
validity and optional confidence while declaring the representation `relative`.
All dense products are restored to the canonical image coordinate space and
referenced by typed, hashed `ArtifactRef` records. The four models run as
sequential passes so they do not occupy GPU memory concurrently. Per-video
`VideoEvidenceManifest` records and the `NeuralEvidenceStore` are namespaced by
the frozen configuration hash. Backends may still be disabled individually; in
that case the cue is explicitly `unavailable` rather than reported as an empty
prediction. The legacy public Step 2 still calls the July detector and is
retained only as a baseline.

---

### Step 3 — Object Tracking

**Purpose:** build ID-consistent mask tracks and a replayable evidence archive
without performing deep semantic or physical reasoning.

**Detailed subfigure:** [`STEP3_MULTI_EVIDENCE_TRACKING.pdf`](./STEP3_MULTI_EVIDENCE_TRACKING.pdf)
([TikZ source](./STEP3_MULTI_EVIDENCE_TRACKING.tex)).

**Inputs:** detections, masks, flow, depth, and their uncertainty from
`NeuralEvidenceStore`.

**Primary implementation**

1. **3.1 - Candidate Tracks:** pair active tracks with current instances,
   retain explicit new/missed/retired alternatives, and compute only the
   lightweight deterministic descriptors needed for association. For example,
   compute a robust depth descriptor from the eroded mask interior:
   $z_{j,t}=\operatorname{median}(Z_t[S_{j,t}^{\mathrm{eroded}}])$, together
   with dispersion/confidence. A marked inner-box fallback is allowed only when
   a valid mask is unavailable. Preserve references to the full mask, depth,
   confidence, and flow artifacts so later stages are not restricted to this
   scalar summary.
2. **3.2 - Multi-cue Association Score:** warp the preceding mask with RAFT,
   then score each feasible pair using available mask, flow, box, class, and
   depth cues. Depth consistency compares the current masked descriptor with
   the previous descriptor stored in the track history. Normalize weights over
   cues that are actually present.
3. **3.3 - Gating and Assignment:** remove infeasible pairs, run one-to-one
   Hungarian assignment, and reject matches below the configured threshold.
4. **3.4 - Track State Logging:** retain IDs for matches, create IDs for new
   instances, maintain active/lost/retired tracker states, and log factual
   start/gap/reappearance/end markers. `max_age`, end-of-video, failed gate, or
   missing observation is recorded only as an operational trigger; Step 3 does
   not claim that the object exited, was occluded, or was missed by a model.
5. **3.5 - Tracking Package:** materialize ID-aligned masks, boxes, confidence,
   per-observation depth descriptors and state markers, plus the raw-evidence
   index, full candidate audit, unassigned evidence pool, uncertainty,
   transforms, and provenance.
   After manifest closure, apply the frozen, seed-controlled evidence-role
   policy to artifact references and write `EvidenceUsePlan` $\Pi$. This
   assignment does not inspect annotations or future evaluation scores.

The Step 3 diagnostic overlay exposes the object label, persistent ID, selected
mask, detection and association confidence, depth representation/support,
masked median, IQR, and valid fraction. These depth values remain relative DA3
descriptors; the visualization must not label them as metric distance.

**Evidence-retention contract**

Step 3 must preserve enough information for Steps 4-9 to test a new geometric,
identity, occlusion, or lifecycle hypothesis without rerunning Steps 2-3:

1. **Raw observation references:** canonical RGB frame/crop; YOLO detection;
   SAM 2 binary mask and mask confidence/logits when available; RAFT
   forward/backward flow, validity and confidence; DA3 depth and confidence;
   timestamps and reversible coordinate transforms.
2. **Track-aligned index:** for every track and frame, link the selected
   observation to all source artifacts and store mask geometry, boundary
   distance, robust masked-depth distribution, object-region flow statistics,
   confidence, and missing-cue flags.
3. **Association audit:** retain every tested candidate pair, all individual cue
   values, effective weights, gates, ranks and assignment results—not only the
   winning match. Store rejected and unmatched observations in a recoverable
   pool so a later hypothesis can relink track fragments.
4. **Operational state history:** record when the tracker first observed,
   missed, recovered or retired an ID and exactly which numerical rule fired.
   These are observations and software decisions, not semantic causes.
5. **Reproducibility:** content hashes, schemas, model/config versions, random
   seeds and artifact health checks make every reference independently
   verifiable. Dense tensors may be compressed or chunked but must not be
   discarded before the final prediction and blind evaluation are frozen.

**Mask recoverability when a track is lost**

The selected mask sequence is not sufficient for a lost-track interval. For
every active or recoverable-lost track $i$ and frame $t$, Step 3 maintains a
`mask_candidate_bank[i,t]` containing every available spatial-support
hypothesis, without deciding which explanation is physically correct:

1. `direct_instance`: the frame-local SAM 2 mask prompted by a current YOLO
   detection.
2. `sam_forward`: SAM 2 video-memory propagation from the last reliable mask,
   produced even when the current detector has no accepted match.
3. `flow_forward`: the preceding mask warped by RAFT forward flow, with
   forward/backward consistency and validity masks.
4. `unassigned_instance`: every current mask that failed identity assignment;
   it may later prove to be a fragmented continuation of the lost track.
5. `sam_backward` and `flow_backward`: after the offline video pass reaches the
   next reliable observation or a new fragment, propagate/warp backward across
   the gap and archive these candidates separately from the forward pass.
6. `empty_or_outside`: an explicit empty candidate when the predicted support
   lies wholly outside the canonical frame; this is a numerical support record,
   not an `exited_fov` explanation.

Each candidate stores `mask_ref`, optional `logit_ref`, confidence, source type,
prompt/anchor frame, parent mask/flow/detection references, transform ID,
generation direction, and whether its pixels are observed or latent. Forward
and backward candidates must not be overwritten by a fused mask.

The per-track/frame coverage invariant is:

$$
\operatorname{observed\_mask}(i,t)
\;\lor\;
|\mathcal{C}^{\mathrm{mask}}_{i,t}|>0
\;\lor\;
\operatorname{explicitly\_unobservable}(i,t).
$$

Thus a downstream stage can retrieve a real observed mask, compare several
stored candidates, or see an explicit absence. It is not promised a true mask
when the object is fully occluded: those pixels are not observable from the
video. A propagated mask in that case is marked `latent_support` and must not be
treated as a direct segmentation measurement.

For every gap, Step 3 also emits a `GapEvidenceRecord` containing the last and
next reliable mask anchors when available, all per-frame candidate references,
unassigned detections/masks, forward/backward flow, depth, RGB-frame references,
and tracker decision records. This allows Steps 6-8 to select a continuation,
declare the interval unobservable, or relink fragments without rerunning the
front-end models.

The archive is logically part of $\mathcal{T}$, while large arrays remain in the
immutable `NeuralEvidenceStore`. “No return to Step 3” means later stages may
read these artifacts and propose alternative associations; they do not need to
rerun neural extraction or reconstruct information that Step 3 discarded.

**Information-preservation guarantee**

“Sufficient” is defined here as **input-complete and traceable**, not as a claim
that the retained evidence will make every later quantity observable. Let
$I_3$ be the manifest of records and artifacts actually presented to Step 3 and
$R_3$ the evidence references emitted in the `TrackingPackage`. Step 3 must
satisfy the following invariants:

1. **Manifest closure:** every $x\in I_3$ has exactly one disposition in $R_3$:
   selected, unselected, invalid/quarantined, or explicitly unavailable at
   input. Nothing may disappear because it lost an assignment.
2. **Content fidelity:** for every available input artifact, its original bytes
   or a losslessly encoded content-addressed copy remain resolvable:
   $\operatorname{hash}(\operatorname{resolve}(r_x))=\operatorname{hash}(x)$.
   A lossy preview may be added but cannot replace the retained original.
3. **Lineage completeness:** every derived mask-track observation, descriptor,
   score and state marker stores `parent_refs`, `transform_id`, code/config hash
   and the numerical rule that produced it. Derived summaries never overwrite
   their parents.
4. **Decision completeness:** every candidate evaluated by the tracker has a
   ledger row, including candidates rejected by a gate, threshold, conflict, or
   Hungarian assignment. The winning association alone is insufficient.
5. **Explicit absence:** a cue that was never produced by Step 2 is represented
   as `input_missing`; a cue that existed but became unreadable is
   `archive_error`. Neither may be silently converted to zero or omitted.

For each frame and modality the accounting identity must hold:

$$
N_{m,t}^{\mathrm{input}} =
N_{m,t}^{\mathrm{selected}} +
N_{m,t}^{\mathrm{unselected}} +
N_{m,t}^{\mathrm{invalid}}.
$$

Step 3 writes a machine-checkable `retention_report` containing per-frame and
per-modality counts, hashes, schema/shape/dtype checks, coordinate-transform
checks, unresolved/orphan references, association-ledger coverage,
mask-candidate coverage for active/lost tracks, and an overall `pass` flag. Step
4 must refuse normal execution when this gate fails. A cue that was explicitly
absent at Step 3 input may reduce later observability but does not count as
information loss inside Step 3.

This guarantee deliberately stops at the Step 3 input boundary. For example,
if Step 2 supplied only binary SAM masks, Step 3 cannot preserve unavailable SAM
logits. Therefore the Step 2 output contract must declare which raw products
are required before a run begins.

**Deferred lifecycle reasoning (Steps 6-8)**

Step 6 uses the full video, geometry and the archived evidence to score possible
causes for each start/gap/end marker. Step 7 turns the relevant evidence window
into an auditable explanation such as `entered_fov`, `temporary_occlusion`,
`exited_fov`, `detector_or_mask_failure`, `track_fragmentation_or_id_handoff`,
`scene_cut_or_timeline_break`, or `unresolved`. Step 8 may create a revised track
hypothesis, such as relinking two fragments, without rerunning Step 3 or erasing
the original assignment audit.

SAM 2 supplies masks; flow and depth help decide whether masks in different
frames belong to the same physical object. Step 3 owns persistent IDs. It does
not estimate metric speed or a world-space trajectory.

**Output:** `TrackingPackage` $\mathcal{T}$ contains the immutable input
snapshot and association ledger, mask candidate bank, gap records, derived
ID-aligned `ObjectTrackSet`, unassigned pool, state markers, uncertainty,
transforms, provenance and a passing `retention_report`. `EvidenceUsePlan`
$\Pi$ assigns the archived references to frozen fit/check/report roles.

**Current repository:** the target implementation is
`src/exp_august/inference/step03_object_tracking.py`. It consumes the typed
`NeuralEvidenceStore`, computes robust eroded-mask depth descriptors, propagates
latent support through adjacent RAFT fields, scores all active-track/instance
pairs with normalized mask/flow/box/class/depth weights, applies deterministic
class and center-distance gates, and performs one-to-one Hungarian assignment.
Every tested pair is written to `association_ledger`, including gate, threshold,
and conflict rejections. The target package records selected observations,
unassigned candidate-tier evidence, factual state markers, forward gap masks,
backward proposals from the next reliable anchor, explicit unobservable support,
and `GapEvidenceRecord` objects. All original Step 2 artifacts remain immutable;
derived gap masks are separately content-addressed in Step 3. A seeded
`EvidenceUsePlan` freezes downstream fit/check-only roles after tracking and
before world-state construction. `retention_report` verifies input hashes and
shapes, manifest accounting, ledger completeness, evidence dispositions, and
every observed/lost track-frame coverage before publication.

The target Step 3 visualization module renders concrete canonical video frames
with stable IDs, selected masks, boxes, detection/association scores and factual
state counts. It separately renders candidate-bank panels so latent flow-warped
support is visually distinguishable from direct observations and explicit
unobservability. Still frames, contact sheets, MP4s and a visualization manifest
are stored inside the corresponding configuration namespace; visualization does
not modify tracking decisions or evidence dispositions.

The legacy public Step 3 still uses ByteTrack bootstrap IDs and an older SAM 2
video/Hungarian adapter. It remains available only for the linear baseline and
does not emit the target `TrackingPackage`. The target implementation currently
does not add a separate SAM 2 video-memory proposal; it preserves frame-local
SAM masks and bidirectional flow-warped alternatives instead.

---

### Step 4 — Relative Camera Geometry + 3D Observation Lift

**Purpose:** estimate relative camera geometry and lift image-space tracks into
camera-centric 3D observations with uncertainty. This stage provides geometric
initialization; it does not yet claim final ego or object physical trajectories.

**Detailed subfigure:** [`STEP4_GEOMETRY_SCALE.pdf`](./STEP4_GEOMETRY_SCALE.pdf)
([TikZ source](./STEP4_GEOMETRY_SCALE.tex)).

**Inputs:** the `TrackingPackage` (including its `ObjectTrackSet` view and
immutable evidence archive), camera metadata, and frozen physical priors.

**Primary implementation**

1. **4.1 - Resolve and Align Evidence:** require the Step 3 retention gate to
   pass; resolve immutable artifact references; align timestamps, coordinates,
   masks, flow, depth and uncertainty without discarding alternatives.
2. **4.2 - Relative Camera Motion:** exclude tracked foreground support and
   estimate pairwise camera motion using robust visual odometry/SLAM or
   equivalent geometry. Retain viable pose edges, disconnected pose components
   and covariance rather than forcing one global path.
3. **4.3 - Intrinsics and Ground:** validate available camera metadata or
   estimate intrinsics candidates; fit horizon/road-plane candidates and derive
   ground-contact anchors with uncertainty.
4. **4.4 - Scale Hypotheses:** combine validated metric-depth evidence when
   available with camera-height, ground-plane, object-size and road priors.
   Retain alternatives, confidence intervals and an explicit scale-observability
   state: `metric`, `relative`, `ambiguous`, or `unobservable`.
5. **4.5 - Lift to Candidate 3D:** consolidate the full masked-depth
   distribution and anchors, then back-project observations as
   $X_c=zK^{-1}p$. A pose component may express them in a component-local frame
   for diagnostics, but only Step 5 may promote them into a globally consistent
   world frame. Propagate mask, depth, calibration, pose and scale covariance.
6. **4.6 - Geometric Validation:** measure mask/box reprojection, background-flow
   agreement, depth consistency, ground contact and scale plausibility. Reject
   hard failures and rank the surviving candidates.
7. **4.7 - Relative Geometry Package:** emit pairwise camera poses and pose
   components, calibration and ground candidates, scale intervals,
   camera-centric per-track 3D observations, covariance, residuals and
   provenance.

Monocular scale is not assumed to be observable. Step 4 must run an explicit
observability test using cue availability, conditioning and posterior spread.
When evidence cannot support a unique metric scale, it emits multiple
scale-conditioned candidates, a wide interval, or `unobservable` rather than
inventing one metric value.

**Output:** a ranked relative geometry package $\mathcal{G}_{\mathrm{rel}}$
containing one or more `GeometryHypothesis` records. Its pose components and 3D
observations are inputs to Step 5, not final physical tracks.

**Current repository:** target Step 4 is implemented in
`src/exp_august/inference/step04_geometry_scale.py`, with immutable contracts in
`src/exp_august/contracts/geometry.py`. It verifies the Step 3 store and every
selected mask/depth/flow artifact before use. A provided pinhole calibration or
a frozen horizontal-FOV prior defines $K$. Available fit-role RAFT background
correspondences are filtered by tracked foreground masks and used to estimate
pairwise essential matrices, camera rotation, and translation direction; the
translation magnitude remains explicitly `up_to_scale`. For each usable track
observation, the selected mask is eroded (or a marked inner-box fallback is
used), valid depth pixels are back-projected, and robust camera-centric 3D
median/IQR/MAD statistics plus a centroid reprojection check are written to the
typed `GeometryStore`. Depth artifacts frozen as `check_only` by Step 3 are not
used for this fit.

The implementation intentionally emits `relative` scale with no
`scale_to_meters` for current DA3 evidence. Ground-plane estimation,
camera-height/object-size scale candidates, multi-hypothesis calibration,
covariance propagation, and world-coordinate pose accumulation remain open;
their fields are explicit `unobservable` states rather than placeholder metric
values. Thus the executable stage currently satisfies relative camera-centric
lifting and provenance, but not the full metric multi-hypothesis claim above.

The executable Step 4 visualization is implemented in
`src/exp_august/inference/step04_visualization.py`. It produces annotated
canonical frames and an optional MP4, depth/mask/back-projection example panels,
a camera-coordinate 3D point-sequence plot with Z-IQR whiskers, XYZ temporal
plots with interquartile bands, and camera-motion inlier/residual/translation-
direction diagnostics. The visualization manifest declares
`world_trajectory_claimed: false`: without validated pose accumulation and
metric scale, connecting camera-frame observations is a diagnostic temporal
sequence rather than a physical world trajectory. This guard prevents Step 4
debug output from pre-empting the Step 5 motion-estimation claim.

The same module now emits a first `RelativeStaticScene` diagnostic and a 3D
ego/static-object sandbox. Pairwise camera rotations are accumulated without
alteration. Translation-direction magnitudes are normalized from repeated
stationary-semantic tracks when their cross-frame residual is sufficiently
conditioned; low-motion tracks are an explicitly marked fallback only when no
stationary semantic anchor exists. Camera centers form the red ego path.
Repeated transformed landmark observations are summarized by a robust median,
axis IQR and radial spread, producing `supported` or `inconsistent` static
markers. Missing pose edges produce independent component-local origins and are
never interpolated. The JSON records that metric scale and a final physical
world trajectory are not claimed. A higher-rate canonical geometry timeline
(approximately 5--10 FPS) is required for useful RAFT/essential-matrix pose
continuity; the 0.2 FPS quick-debug setting is unsuitable for a coherent path.
Disconnected components are visualized in separate 3D subplots and separate
16:9 component figures; no shared placement is implied. Plot-box geometry
preserves an elongated forward dimension instead of forcing a cube, and only
the two temporal endpoints of each component receive frame labels.

---

### Step 5 — Joint Ego/Object World Reconstruction

**Purpose:** connect compatible ego pose components, separate ego/camera motion
from object motion, and instantiate complete uncertainty-aware alternative world
states. Do not collapse ambiguous geometry, scale or association into one
smoothed path.

**Inputs:** $\mathcal{G}_{\mathrm{rel}}$, $\mathcal{T}^{2D}$,
$\mathcal{O}$, $\Pi$, and frozen knowledge $\mathcal K$.

**Primary implementation**

1. **5.1 - Connect and branch alternatives:** join pose components only when
   cross-component evidence supports the transformation, then instantiate
   viable combinations of camera pose, ground, scale, mask candidate and
   identity/relink choices. Unsupported joins remain separate hypotheses. A
   branch stores only references and diffs, not duplicated dense artifacts.
2. **5.2 - Estimate continuous states:** jointly separate ego/camera motion
   from residual object motion and estimate
   $\mathbf s_t=[\mathbf p_t,\mathbf v_t,\mathbf a_t,\theta_t,\dot\theta_t]$
   for ego and each observable object. Use an uncertainty-aware smoother,
   factor graph or constrained spline, while preserving unsmoothed observations.
3. **5.3 - Propagate uncertainty:** carry mask, depth, pose, scale and temporal
   uncertainty into per-state covariance or samples. Never report a narrower
   metric interval than the scale hypothesis permits.
4. **5.4 - Mark observability:** record which state dimensions are metric,
   relative, ambiguous or unobservable for each time window and object.
5. **5.5 - Form a diverse beam:** prune exact/near duplicates while preserving
   distinct scale, pose and identity explanations. Each hypothesis is immutable,
   parented, replayable and assigned a construction score that is not yet the
   final closed-loop score.

The canonical state is

$$
\mathcal H_i = \{C_{1:T},s,A_{1:T},X_{1:T}^{1:N},U_{1:T}\},
$$

where $C$ is camera/ego pose, $s$ scale, $A$ association/lifecycle choices,
$X$ object physical states and $U$ uncertainty/observability. Step 5 estimates
a distribution or finite hypothesis set, not one forced answer.

**Output:** initial world hypotheses $\mathcal{H}_0^{1:n}$ and diverse Top-K
beam $\mathcal{B}_0$.

**Current repository:** the first target implementation is in
`src/exp_august/inference/step05_joint_world_reconstruction.py`, with immutable
contracts in `src/exp_august/contracts/world_state.py`. It accumulates supported
Step 4 pose edges without bridging failed links, estimates relative translation
magnitude from repeated static-semantic observations when conditioned, places
ego and object states in component-local frames, propagates position uncertainty
into speed intervals, classifies initial object motion, and emits an immutable
`HypothesisBeam` $\mathcal B_0$. Relative DA3 inputs remain `relative_unit`; no
m/s claim is made. The current beam branches over evidence-distinct Step 4 scale
hypotheses and adds bounded one-variable `static`/`moving` alternatives for
ambiguous object motion while retaining the ambiguous parent. It does not yet
implement the full factor graph, track-relink/mask branching, joint combinatorial
branching, or duplicate-pruning policy described above. Those refinements remain
Step 5 work rather than being silently delegated to Step 6.

`src/exp_august/inference/step05_visualization.py` renders component-local ego
and ego-compensated object trajectories plus speed intervals. It explicitly
marks the selected rank-1 state as an initial, not Step 6-verified, hypothesis.

---

### Step 6 — Forward Prediction and Consistency Verification

**Purpose:** use analysis-by-synthesis to measure whether each physical world
hypothesis predicts the available video evidence and obeys physical/temporal
constraints.

**Detailed subfigure:** [`STEP6_CONSISTENCY_CHECKS.pdf`](./STEP6_CONSISTENCY_CHECKS.pdf)
([TikZ source](./STEP6_CONSISTENCY_CHECKS.tex)).

**Inputs:** hypothesis beam, raw neural observations, tracks, geometry, frozen
knowledge and `EvidenceUsePlan` $\Pi$.

**Primary implementation**

1. **6.1 - Forward Prediction and Reprojection:** resolve references and
   forward-project every candidate 3D state into the image. Predict the mask,
   box, depth evolution, optical flow and background-motion signature that the
   hypothesis should generate. Propagate state, pose, scale and rendering
   uncertainty and mark non-evaluable intervals explicitly.
2. **6.2 - Observation Fit:** measure mask/box reprojection, depth agreement and
   optical-flow agreement. Record fit and check-only residuals separately
   according to $\Pi$.
3. **6.3 - Ego and Background:** test static, forward, braking and left/right
   turn signatures against background evolution and temporal coherence.
4. **6.4 - Object and Identity:** test mask/depth/relative-motion continuity and
   identity alternatives. For each factual start/gap/end marker, evaluate
   occlusion, exit, detector failure and fragment-relink hypotheses against
   later observations.
5. **6.5 - Physical Plausibility:** evaluate speed, acceleration, jerk,
   curvature and yaw-rate changes with uncertainty-aware limits rather than one
   universal threshold.
6. **6.6 - Semantic Logic:** test only road, vehicle and pedestrian relations
   that are observable in the available video evidence.
7. **6.7 - Constraint Aggregation:** keep cue families and fit/check roles
   separate; classify hard violations versus soft residuals; localize peaks,
   temporal windows and suspected components while retaining uncertainty and
   evidence lineage.
8. **6.8 - Residual Packet:** emit constraint identifiers, residual curves,
   conflict windows, suspected components, hard-constraint status, evidence
   references and provenance for each hypothesis.

For cue $c$ at time $t$, Step 6 records

$$
\hat y_{c,t}=g_c(\mathcal H_i),\qquad
z_{c,t}=\frac{d_c(y_{c,t},\hat y_{c,t})}
{\sqrt{\sigma^2_{\mathrm{obs},c,t}+\sigma^2_{\mathrm{pred},c,t}+\epsilon}}.
$$

Persistent high normalized residuals, hard violations, or contradictions across
independent cue families constitute an inconsistency. A single spike does not
automatically fail a hypothesis. Missing or unobservable evidence reduces the
applicable constraint set; it is not assigned zero and is not automatically a
violation.

Step 6 predicts and evaluates only. Repair belongs to Step 8, ranking belongs
to Step 9, and report-only annotations remain inaccessible. If no check-only
evidence supports a residual family, its result is tagged
`self_consistency_only`.

**Output:** `ResidualPacket` $\mathcal{R}_i$ per hypothesis.

**Current repository:** the target runner implements a typed baseline in
`inference/step06_predict_verify.py`. Every `WorldHypothesis` receives a strict
`HypothesisResidualPacket`; fit evidence, seeded check-only evidence,
non-evaluable checks, conflict windows, hard/soft status and artifact lineage
remain separate. Implemented forward checks include fitted centroid
reprojection, held-out object depth, held-out backward object flow, rigid
background flow from ego pose/depth, temporal gaps, speed/acceleration and a
soft semantic-static prior. `step06_visualization.py` renders beam comparison,
per-hypothesis residual timelines, family accounting, machine-readable conflict
audits, and concrete keyframe panels with mask/box support and predicted-versus-
observed image/flow marks, without choosing a winner. Dense mask
rendering, explicit lifecycle-cause hypotheses, road-context physical limits,
jerk/curvature/yaw and calibrated predictive uncertainty remain open.

---

### Step 7 — Failure Diagnosis and Repair Proposal

Detailed 16:9 subfigure: [`STEP7_DIAGNOSE_PROPOSE.pdf`](./STEP7_DIAGNOSE_PROPOSE.pdf)
for review/print, [`STEP7_DIAGNOSE_PROPOSE.svg`](./STEP7_DIAGNOSE_PROPOSE.svg)
for browser/Markdown, and [`STEP7_DIAGNOSE_PROPOSE.tex`](./STEP7_DIAGNOSE_PROPOSE.tex)
as the editable TikZ source.

**Purpose:** turn residual peaks into a compact, auditable diagnosis and choose
a bounded repair operator without changing the world state.

**Inputs:** residual packets, tracks, curves, and source frames.

**Primary implementation**

- Select keyframes and temporal windows around residual peaks, state changes,
  track births/terminations, occlusions, and cue disagreements.
- Package synchronized frames/crops, masks, flow, depth, trajectories, and
  numerical residual plots.
- Run deterministic model-based and logical checks first. Assign candidate
  causes such as identity error, mask error, depth jump, pose drift, scale
  ambiguity, invalid static-background assumption, dynamics mismatch, true
  acute maneuver, or unobservable evidence.
- Choose only from the versioned repair allow-list and predict which fit and
  check residuals should change if the diagnosis is correct.
- Treat an LLM/VLM as optional. It may rank structured failure categories,
  select relevant frozen constraints and recommend an allow-listed operator.
  It must not create continuous state values, edit evidence, or make the final
  acceptance decision.

**Output:** `EvidencePacket` $\mathcal{E}_i$ and zero or more bounded
`RepairProposal` records $\Delta_i$.

**Current repository:** the target runner implements a deterministic typed
baseline in `inference/step07_diagnose_propose.py`. It verifies Step 6 lineage,
clusters related conflict windows, selects a structured failure category and
an operator from `bounded_repair_operators_v1`, and emits `EvidencePacket` plus
bounded `RepairProposal` records. Proposals declare the immutable parent,
affected variables/window, discrete or numerical bounds, expected fit/check
effects and compute budget. Check-only evidence is hard-blocked from Step 8
optimization targets. Missing evidence produces `mark_unobservable`, and a
soft semantic conflict without sufficient support produces `leave_unresolved`
rather than invented physical values. The current baseline references source
frame mappings and dense evidence artifacts. `step07_visualization.py` now
renders hypothesis-level diagnosis/operator accounting, proposal timelines,
machine-readable audits and source-mapped 1920x1080 repair panels with
mask/box support, residual geometry, parameter bounds and explicit safety
flags. The optional LLM/VLM diagnosis path remains open.

---

### Repair Operator Library and Numerical Solver

**Purpose:** instantiate bounded, testable corrections proposed by Step 7
without modifying raw evidence or directly inventing the final trajectory.

**Primary implementation**

- Versioned allow-list:
  `relink_track`, `split_track`, `switch_mask_candidate`,
  `switch_pose_candidate`, `switch_scale_candidate`,
  `invalidate_or_downweight_cue`, `refit_local_dynamics`,
  `adjust_process_noise`, `mark_occluded`, `mark_unobservable`, and
  `leave_unresolved`.
- Rule engine: reject proposals outside their declared variables, windows,
  parameter bounds, evidence permissions or compute budget.
- Numerical solver: instantiate continuous candidates using robust fitting,
  constrained optimization, Kalman/RTS smoothing, splines or factor graphs.
- Every proposal declares its target residual, expected check-evidence effect,
  parent ID, affected fields and frames, and reversible diff.

**Output:** zero or more `RepairProposal` records $\Delta_i$ with affected
variables/windows, parameter bounds, expected fit/check changes, budget and
provenance. `leave_unresolved` is a valid output when evidence cannot identify
a safe repair.

**Current repository:** the versioned allow-list, typed bounds, expected
residual effects and per-proposal budgets are connected to the canonical Step 7
evidence and residual packets. Target Step 8 now instantiates the world-state
operators that the current boundary can represent directly. Operators that
require a mutable tracking or candidate-bank child are retained as explicit
`unsupported` audit records rather than being simulated with invented state.

---

### Step 8 — Local Re-estimation

Detailed 16:9 subfigure: [`STEP8_LOCAL_REESTIMATION.pdf`](./STEP8_LOCAL_REESTIMATION.pdf)
for review/print, [`STEP8_LOCAL_REESTIMATION.svg`](./STEP8_LOCAL_REESTIMATION.svg)
for browser/Markdown, and [`STEP8_LOCAL_REESTIMATION.tex`](./STEP8_LOCAL_REESTIMATION.tex)
for editing.

**Purpose:** apply each permitted repair only where needed and generate new
hypotheses.

**Inputs:** $\Delta_i$, parent hypotheses, affected evidence windows.

**Primary implementation**

- Re-estimate only the affected association, geometry, scale, filtering, or
  state variables and time windows while preserving boundary conditions. Use
  the archived candidate bank; do not rerun neural evidence extraction.
- Produce multiple candidates when a repair remains ambiguous.
- Optimize a declared objective over permitted variables only, for example:

  $$
  \min_{\delta\in\Omega(\Delta_i)}
  J_{\mathrm{fit}}(\mathcal H_i\oplus\delta)
  +\lambda_pJ_{\mathrm{phys}}(\mathcal H_i\oplus\delta),
  $$

  where check-only evidence is excluded from the optimizer.
- Record parent ID, changed fields, changed frames, boundary conditions,
  solver status, computation budget and reversible parent/child diff.
- Preserve the parent unchanged; an unsuccessful repair produces no accepted
  child and may return an explicit unresolved record.

**Output:** re-estimated candidates $\mathcal{H}_{i+1}^{1:n}$.

**Current repository:** implemented in
`inference/step08_local_reestimation.py` with typed contracts in
`contracts/local_reestimation.py`. The deterministic baseline supports bounded
local dynamics refits, process-noise scaling, explicit occlusion and
unobservability marking, and `leave_unresolved`. Each proposal produces its own
audited result and zero or more immutable child hypotheses. Persisted candidate
records contain objective terms, optimized and excluded residual IDs, changed
fields/frames, reversible before/after values, boundary/bounds/budget guards,
and explicit no-mutation/no-selection flags. Check-only residuals are never
passed to the objective; candidate acceptance remains Step 9 work.

For the baseline local-dynamics operator, the first and last state in the
proposal window are fixed. For every interior state, the solver forms the
boundary interpolation $l_t$, sweeps deterministic strengths
$\alpha_j=j/n$, and applies the axis-wise bounded update

$$
p'_t = p_t + \operatorname{clip}\!\left(
\alpha_j(\ell_t-p_t), -k\sigma_t, k\sigma_t
\right),
$$

where $k$ is the proposal's declared maximum state/pose delta in standard
deviations. Interior velocity and speed are then recomputed by centered finite
differences. Thus Step 7's arrow names an operator and its permitted direction;
Step 8 supplies concrete, bounded candidate values and records their measured
objective terms.

The Step 8 visualization emits one 1920x1220 panel per proposal. Its target
frame shows only `EGO` for ego-state changes or a labeled bounding box for an
object-state change. A cause plot compares the current residual error with its
acceptable limit; its color matches the active diagnosis in the bottom row of
all primary and alternative causes. Candidate objective movement is visualized
separately, while numerical changes, bounds, boundary preservation and status
remain in a compact audit table. No multi-proposal overview is produced and no
displayed candidate is labeled as selected.

---

### Step 9 — Feasibility, Selection and Retention

**Purpose:** rank candidates consistently, retain the best-ever explanation,
and decide whether to continue the loop.

**Primary implementation**

1. Reject candidates that violate hard physical constraints, artifact
   contracts, repair permissions or immutable-evidence requirements.
2. Compare each child with its parent. A child is admissible only when it adds
   no hard violation, improves its target fit residual, and improves or does not
   degrade check-only evidence beyond a frozen tolerance. It may not shrink
   uncertainty without new supporting evidence.
3. Rank admissible survivors using a versioned score:

   ```text
   J(H) = w_fit*fit_evidence_error
        + w_check*check_evidence_error
        + w_physics*physics_violation
        + w_semantic*semantic_violation
        + w_complexity*repair_and_model_complexity
        + w_uncertainty*unresolved_uncertainty
   ```

4. Keep a diverse Top-K beam, not merely the K lowest near-duplicates. Enforce
   diversity across scale, pose and identity explanations.
5. Update the best-ever register across all iterations; never assume the final
   iteration is the best.
6. Stop when constraints pass, improvement is below $\epsilon$, no admissible
   repair remains, or the iteration/compute budget is exhausted.

Score weights, normalization functions, check-evidence tolerances, beam size
and stopping budgets are selected on development/calibration videos and frozen
before blind testing. A candidate unsupported by check-only evidence can remain
in the beam but must carry `self_consistency_only`; it cannot displace an
externally supported candidate solely by becoming smoother.

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

**Purpose:** test whether refinement improves external accuracy, not merely
self-consistency, without influencing inference.

**Inputs:** frozen predictions $\mathcal{P}^{*}$ and held-out references
$\mathcal{Y}$, which may include pose/trajectory measurements, tracking labels
and human temporal segmentation.

**Primary implementation**

- Verify the frozen manifest and output hashes before evaluation.
- Run in a separate command/process with read-only access to predictions.
- Report world-state metrics when references exist: camera pose error,
  trajectory error, speed/acceleration error and scale error.
- Report tracking and mask metrics: association accuracy, ID switches,
  fragmentation and mask overlap.
- Report boundary F1 at declared tolerances, frame/segment F1, temporal IoU,
  confusion matrices and per-video failure analysis.
- Report uncertainty calibration/coverage and stratify results by
  `metric`, `relative`, `ambiguous` and `unobservable` status.
- Measure repair success as external-error change, not only internal-residual
  change. A physically smoother path that increases held-out error is a failed
  repair.
- Never write parameters, thresholds, prompts, or states back into inference.

If pose or trajectory ground truth is unavailable, the paper must limit its
claim to segmentation quality and evidence/physics self-consistency. It may not
claim verified recovery of true metric motion from segmentation alone.

**Current repository:** evaluation metrics exist, but their invocation must be
moved out of current Step 8 to enforce the intended isolation boundary.

## 5. Paper-facing evaluation contract

### 5.1 Falsifiable claims

The method is publishable only if experiments can falsify and support claims
stronger than "the final trajectory is smoother":

1. Closed-loop refinement reduces held-out world-state or downstream task error
   relative to the open-loop pipeline.
2. Forward prediction plus check-only evidence distinguishes plausible-but-wrong
   trajectories from explanations supported by the video.
3. Explicit observability prevents false metric precision and produces useful,
   calibrated uncertainty intervals.
4. A frozen training-free system generalizes across videos and, where possible,
   across datasets without test-time parameter learning.
5. The best-ever/beam mechanism improves robustness over overwriting one state
   with the last repair.

### 5.2 Required baselines and ablations

At minimum, evaluate:

| Variant | Purpose |
|---|---|
| Raw open-loop perception/geometry | Establish the starting error |
| Conventional smoothing only | Test whether the contribution is more than smoothness |
| Evidence-only optimization | Isolate physics/knowledge value |
| Physics-only refinement | Expose plausible-but-unsupported solutions |
| Evidence + physics without loop | Measure the value of iterative repair |
| Full loop without check-only evidence | Measure circular self-validation risk |
| Full loop without beam/best-ever | Measure search and retention value |
| Full deterministic method without LLM/VLM | Establish the reproducible core |
| Full method with optional LLM/VLM | Measure diagnosis efficiency or repair success only |

Report both internal residual change and external ground-truth change per repair.
Include failure cases, compute/iteration budgets and performance stratified by
cue availability and scale observability.

### 5.3 Primary method boundary

For the paper, Steps 1-3 are evidence infrastructure, Steps 4-5 construct
uncertain world hypotheses, Steps 6-9 are the primary method, and Steps 10-11
are downstream materialization. The paper should not present all eleven steps
as equally novel. The core algorithm must be expressible independently of the
particular YOLO/SAM/RAFT/depth backends.

## 6. Target design versus current runner

The target runner now implements Steps 1-8 with their flowchart meanings. The
legacy public runner still uses different concepts after Step 4, so always name
the execution path and module as well as the step number.

| Target flowchart step | Target module | Current repository location | Alignment |
|---:|---|---|---|
| 1 | Init | Target `inference/step01_init.py`; legacy public Step 1 remains | Typed boundary implemented and consumed by target Step 2 |
| 2 | Neural Perception | Target `inference/step02_neural_evidence.py`; legacy public Step 2 remains | Typed YOLO/SAM 2/RAFT/DA3 evidence store implemented |
| 3 | Object Tracking | Target `inference/step03_object_tracking.py`; legacy public Step 3 remains | Typed replayable TrackingPackage implemented |
| 4 | Relative Geometry + 3D Lift | Target `inference/step04_geometry_scale.py`; legacy public Step 4 remains | Typed relative geometry implemented; metric/ground alternatives remain partial |
| 5 | Joint Ego/Object World Reconstruction | Target `inference/step05_joint_world_reconstruction.py`; legacy public Step 5 remains | Typed component-local reconstruction and initial beam implemented; full branching/factor graph remain partial |
| 6 | Predict + Verify | Target `inference/step06_predict_verify.py`; legacy public Step 6 remains | Typed residual packets, frozen fit/check separation and baseline forward checks implemented; dense mask/lifecycle/context models remain partial |
| 7 | Diagnose + Propose | Target `inference/step07_diagnose_propose.py`; legacy public Step 7 remains | Typed evidence packets, deterministic diagnoses, bounded proposals and rendered evidence bundles implemented; optional LLM/VLM remains open |
| 8 | Local Re-estimation | Target `inference/step08_local_reestimation.py` | Typed bounded candidates, reversible diffs and fit/check isolation implemented; tracking/candidate-bank rewrites remain explicit unsupported records |
| 9 | Select + Retain | No single corresponding module | Missing |
| 10 | Segmentation | Current public Step 8 | Numbering/isolation mismatch |
| 11 | Symbolic Scene | Current public Steps 9-11 | Substantially aligned |
| — | Blind Evaluation | Invoked inside current Step 8 | Must be isolated |

## 7. Current Step 3 association defaults

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

## 8. Acceptance criteria by boundary

| Boundary | Minimum acceptance check |
|---|---|
| Step 1 → 2 | Every frame has a canonical timestamp and reversible source mapping |
| Step 2 → 3 | Evidence tensors share coordinates; missing cues are explicit |
| Step 3 → 4 | `retention_report.pass=true`; manifest accounting closes; all hashes/references resolve; every active/lost track-frame has an observed mask, a non-empty candidate bank, or an explicit unobservable marker; all decisions, unmatched observations, gap records, transforms and provenance are readable without rerunning Steps 2-3 |
| Step 4 → 5 | Pairwise poses, pose components, camera-centric 3D observations and scale alternatives include covariance and `metric/relative/ambiguous/unobservable` status; no final world trajectory is claimed |
| Step 5 → 6 | Every hypothesis is immutable, parented and reproducible; $\Pi$ fixes fit/check evidence roles |
| Step 6 → 7 | Every residual identifies predicted/observed values, uncertainty, role, constraint, time window and evidence path |
| Repair → 8 | Every edit uses an allow-listed operator and is bounded, permitted and independently replayable |
| Step 8 → 9 | Check-only evidence was excluded from optimization; parent/child diffs and compute cost are recorded |
| Step 9 → Freeze | No new hard failure; target fit improves; check evidence is non-degraded; best-ever selection and stopping reason are auditable |
| Freeze → 10/11 | Frozen hashes verify; no annotation dependency exists |
| Predictions → Evaluation | Evaluator cannot mutate or feed back into inference |

## 9. Open design decisions

1. Decide whether Step 3 should add an optional SAM 2 video-memory proposal in
   addition to the frame-local Step 2 masks. Step 3 remains the owner of
   canonical persistent IDs.
2. Choose the camera-pose backend and a formal metric-scale observability test.
3. Extend the implemented Step 5 baseline with a factor graph/state-space
   estimator, track-relink/mask discrete branches, calibrated uncertainty and a
   duplicate-pruning rule.
4. Extend the implemented Step 6 baseline with dense mask rendering, explicit
   lifecycle-cause tests, calibrated predictive uncertainty and coverage-aware
   check-only sampling. Flow/depth/background checks and the frozen
   `EvidenceUsePlan` boundary are already executable.
5. Define hard versus soft physical constraints by road context and uncertainty.
6. Freeze the deterministic diagnosis taxonomy and complete repair allow-list;
   define an optional structured LLM/VLM schema only after the deterministic
   core works.
7. Define beam diversity, maximum loop budget, check-evidence non-degradation
   tolerance, and score calibration without
   consulting test annotations.
8. Refactor annotation evaluation behind the frozen prediction boundary and
   identify a benchmark or subset with external pose/trajectory references.
9. Define the immutable evidence archive format, chunking/compression policy,
   retention period, and integrity checks for dense masks, flow, and depth.

## 10. Decision log

| Date | Decision | Consequence |
|---|---|---|
| 2026-08-12 | Step 2 extracts independent evidence and performs no evidence fusion. | Tracking and all cue fusion begin at Step 3 or later. |
| 2026-08-12 | Step 3 is named **Object Tracking**. | Its concise responsibility is “Build ID consistent mask tracks.” |
| 2026-08-12 | Step 3 is not SAM-only. | The target matcher uses SAM masks plus available flow, box, class, and depth evidence. |
| 2026-08-12 | Object-level depth is first derived inside Step 3 from the Step 2 mask and depth map. | Step 3 uses masked depth for association; Step 4 consolidates/refines it for geometry and does not create it for the first time. |
| 2026-08-12 | Step 3 output was initially described as an image-space `ObjectTrackSet`. | The track set remains a derived view, but the output contract is superseded by the 2026-08-13 `TrackingPackage`. |
| 2026-08-12 | Step 3.4 was initially assigned lifecycle explanations with delayed confirmation. | Superseded by the 2026-08-13 evidence-first boundary below. |
| 2026-08-12 | TikZ is the authoritative diagram format. | Layout changes are made in the `.tex` source and exported to PDF/SVG. |
| 2026-08-13 | Step 3 performs shallow identity association and evidence archiving, not deep lifecycle reasoning. | Step 3 records factual state markers and preserves all raw/candidate evidence; Steps 6-8 infer, validate and revise explanations without rerunning Steps 2-3. |
| 2026-08-13 | Step 3 sufficiency means input completeness and traceability, not guaranteed future observability. | A manifest-closure, hash-fidelity, lineage and decision-completeness gate produces `retention_report`; Step 4 fails closed if Step 3 silently loses available input. |
| 2026-08-13 | Lost-track intervals retain a multi-source mask candidate bank rather than one fused/imputed mask. | Direct, forward, backward, flow-warped and unassigned candidates remain distinguishable; fully occluded pixels are marked latent/unobservable, allowing later relinking or explanation without rerunning the front end. |
| 2026-08-13 | Step 4 emits a ranked relative-geometry package rather than forcing one metric reconstruction. | Pairwise camera pose components, camera-centric 3D observations, calibration/ground and scale alternatives retain covariance, residuals and an explicit metric/relative/ambiguous/unobservable state; Step 5 alone forms globally expressed ego/object motion hypotheses. |
| 2026-08-13 | Step 6 evaluates five residual families in parallel and only aggregates/localizes violations. | Observation, ego/background, object/identity, physics and semantic residuals remain separately auditable; repair, best-hypothesis selection and human labels stay outside Step 6. |
| 2026-08-13 | The pipeline remains training-free on target videos. | Per-video latent states may be optimized, but no shared parameter, prior, threshold or prompt is updated across blind-test videos. |
| 2026-08-13 | Steps 5-9 are the paper-facing method; Steps 1-3 are evidence infrastructure. | The paper will claim an uncertainty-aware analysis-by-synthesis loop, not novelty from chaining pretrained components. |
| 2026-08-13 | Step 6 begins with forward prediction and reprojection. | Every world hypothesis predicts mask/flow/depth/background signatures before independent residual families are evaluated. |
| 2026-08-13 | Fit and check-only evidence roles are frozen in $\Pi$. | A repair cannot validate itself solely using cues it was allowed to optimize; unsupported results are labeled self-consistency only. |
| 2026-08-13 | Step 9 accepts children conservatively and retains the best-ever beam. | No new hard violation is permitted; check evidence must be non-degraded; physical smoothness alone cannot justify replacement. |
| 2026-08-13 | Target Step 2 implements independent YOLO, frame-local SAM 2, bidirectional RAFT and single-frame DA3 passes. | Dense cues remain unfused, share canonical image coordinates, carry content hashes, and DA3 is explicitly relative until Step 4 resolves scale. |
| 2026-08-13 | Step 2 models run sequentially and canonical frames are decoded per pass. | Peak GPU residency is bounded without creating a permanent RGB/JPEG cache; the source hash and canonical timeline remain the common reference. |
| 2026-08-13 | Target Step 3 uses deterministic multi-cue Hungarian association and publishes only after retention closure. | Every candidate pair and disposition is auditable; gap masks remain separate forward/backward/explicitly-unobservable alternatives, while semantic lifecycle causes remain deferred. |
| 2026-08-14 | Target Step 5 emits the typed initial beam $\mathcal B_0$. | Supported pose edges form independent ego components; object observations are ego-motion compensated inside those components; relative units, uncertainty, unresolved evidence and the lack of Step 6 verification remain explicit. |
| 2026-08-14 | Target Step 6 emits immutable residual packets for every Step 5 hypothesis. | Fitted reprojection, held-out depth/backward-flow checks, ego/background rendering, physics/semantic diagnostics and conflict localization are auditable; missing evidence is `not_evaluable`, while repair and selection remain outside Step 6. |
| 2026-08-18 | Target Step 7 emits deterministic typed diagnoses, bounded allow-listed proposals and proposal visualizations. | The diagnosis-to-repair boundary is auditable while world state remains immutable and check-only evidence remains forbidden as a Step 8 optimization target. |
| 2026-08-19 | Target Step 8 instantiates bounded local child hypotheses without selection. | Parent and raw evidence remain immutable; fit/physics drive re-estimation, check-only residuals are excluded, every child carries a reversible diff, and Step 9 retains sole ownership of acceptance/ranking. |

## 11. Maintenance procedure

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
