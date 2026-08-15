# exp_august

There are currently two explicit execution paths:

- `src.exp_august.pipeline` is the frozen legacy linear baseline.
- `src.exp_august.inference.runner` is the target annotation-free world-state
  pipeline. Its implemented boundaries currently cover Steps 1-5.

## Target Step 1: Init

The target Step 1 reads raw videos only. It does not discover annotations,
labels, prepared frame folders, or train/test membership. It validates each
video, fingerprints the input, applies display orientation, and maps source
timestamps to a deterministic canonical timeline. Dense RGB frames are not
duplicated: the versioned manifest preserves the source-frame mapping needed to
decode them on demand.

```bash
python -m src.exp_august.inference.runner \
  --video-count 1 \
  --max-step 1 \
  --canonical-fps 10 \
  --decode-validation sample
```

Use `--video-ids ID ...` or `--video-paths PATH ...` for explicit inputs.
`--decode-validation full` checks every canonical frame; `sample` checks a
deterministic spread while retaining all frame mappings; `none` performs probe
validation only. Outputs are written below:

```text
<output-root>/<run-id>/01_init/
  init_bundle.json
  videos/<video-id>.manifest.json
```

The contract implementation is under `src/exp_august/contracts/`; Step 1 is
under `src/exp_august/inference/step01_init.py`. The output JSON is immutable,
schema-versioned, content-addressed, and reload-validated by Pydantic.

## Target Step 2: Neural Evidence

Step 2 uses the Step 1 source-frame mapping to decode canonical, orientation-
normalized frames on demand. It extracts independent evidence only: no tracking,
persistent object identity, evidence fusion, or physical inference occurs here.
YOLO-World retains primary detections and lower-confidence candidates with stable
frame-local IDs. Optional SAM 2, RAFT-Small, and Depth Anything 3 backends write
frame-local masks, adjacent-frame bidirectional flow, and single-frame relative
depth. The models run sequentially and release GPU memory between passes; frames
are decoded again on demand instead of being duplicated into a JPEG cache.

```bash
python -m src.exp_august.inference.runner \
  --video-count 1 \
  --max-step 2 \
  --canonical-fps 1 \
  --objects-backend yolo_world \
  --yolo-model weights/yolo/yolov8s-worldv2.pt \
  --masks-backend sam2 \
  --sam2-model weights/sam2/sam2_t.pt \
  --flow-backend raft \
  --depth-backend da3 \
  --depth-process-resolution 504 \
  --batch-size 4 \
  --device cuda:0
```

Local weights are required by default. Add `--allow-model-download` only when a
run is explicitly permitted to resolve missing YOLO, SAM 2, or RAFT weights.
DA3 follows the Hugging Face cache policy used by the repository depth module.
Torchvision RAFT requires both image dimensions, after padding to a multiple of
eight, to be at least 128 pixels; the backend fails early with this constraint.

Step 2 adds:

```text
<output-root>/<run-id>/02_neural_evidence/config_<hash>/
  neural_evidence_store.json
  videos/<video-id>.evidence.json
  artifacts/masks/<video-id>/frame_<index>/mask_<rank>.png
  artifacts/flow/<video-id>/frame_<source>_to_<target>_<direction>.npz
  artifacts/depth/<video-id>/frame_<index>.npz
```

Each dense artifact records shape, dtype, coordinate space, content hash, and
provenance. Flow artifacts retain the dense field, domain-valid mask,
forward/backward consistency mask, and residual error. Depth artifacts retain
depth, validity, and model confidence when supplied; DA3 output is deliberately
typed as `relative`, not meters. The configuration hash includes model identity,
vocabulary, thresholds, device declaration, and batch settings, so alternative
Step 2 configurations cannot overwrite one another.

The dense backends default to `disabled` because they are compute-intensive.
Use `--objects-backend disabled`, `--masks-backend disabled`,
`--flow-backend disabled`, or `--depth-backend disabled` to exercise a partial
pipeline. A disabled or missing cue is explicitly `unavailable`; an executed
model with no result is `empty`; a temporal endpoint or a SAM frame with no
eligible detector prompt is `not_applicable`.

## Target Step 3: Object Tracking

Step 3 reads the immutable `NeuralEvidenceStore` and builds ID-consistent,
image-space mask tracks. It forms every active-track/current-instance pair,
computes mask IoU, RAFT-warped mask IoU, box IoU, class consistency, and robust
masked-depth consistency, normalizes weights over cues that are actually
available, then applies deterministic gates and one-to-one Hungarian assignment.
This stage does not estimate 3D position, metric speed, or physical causes for
an object's appearance or disappearance.

```bash
python -m src.exp_august.inference.runner \
  --video-count 1 \
  --max-step 3 \
  --canonical-fps 0.2 \
  --objects-backend yolo_world \
  --yolo-model weights/yolo/yolov8s-worldv2.pt \
  --masks-backend sam2 \
  --sam2-model weights/sam2/sam2_t.pt \
  --flow-backend raft \
  --depth-backend da3 \
  --depth-process-resolution 224 \
  --tracking-max-age-frames 2 \
  --tracking-min-score 0.30 \
  --visualize-step3 \
  --device cuda:0
```

Step 3 adds:

```text
<output-root>/<run-id>/03_object_tracking/config_<hash>/
  tracking_store.json
  videos/<video-id>.tracking.json
  artifacts/mask_candidates/<video-id>/<track-id>/*.png
```

The per-video `TrackingPackage` contains the track view plus the complete
association ledger, input references, evidence dispositions, forward/backward
gap-mask candidates, unassigned evidence, factual state markers, deterministic
transforms, a downstream `EvidenceUsePlan`, and a machine-checkable retention
report. It is not published unless hashes/shapes, candidate-pair accounting,
evidence disposition, and observed/lost track-frame coverage all close.

With `--visualize-step3`, each video also receives annotated canonical-frame
PNGs, a four-frame contact sheet, candidate-archive panels, and an MP4 under
`visualizations/<video-id>/`. Selected observations show stable ID, class,
detection confidence, association score, box, mask fill, mask contour, and the
object-masked depth representation/support, median, IQR, and valid fraction.
Candidate panels deliberately keep flow-forward, flow-backward, unassigned, and
unobservable hypotheses separate. `step3_visualization_manifest.json` indexes
all outputs. Use `--no-step3-video` when only still images are needed.

## Target Step 4: Geometry and Scale

Step 4 consumes the immutable `TrackingStore`; it never reruns YOLO, SAM 2,
RAFT, or DA3. It verifies the Step 3 package and referenced mask/depth/flow
artifacts, forms a camera-intrinsics hypothesis, estimates relative camera
motion from background RAFT correspondences when sufficiently conditioned, and
back-projects each usable object-mask depth distribution into camera-centric
3D. Depth artifacts reserved by the Step 3 `EvidenceUsePlan` as `check_only`
are not used for geometry fitting.

```bash
python -m src.exp_august.inference.runner \
  --video-count 1 \
  --max-step 4 \
  --canonical-fps 1 \
  --objects-backend yolo_world \
  --masks-backend sam2 \
  --flow-backend raft \
  --depth-backend da3 \
  --horizontal-fov-degrees 90 \
  --visualize-step4 \
  --device cuda:0
```

When calibrated intrinsics are known, replace the FOV assumption with
`--camera-fx-px`, `--camera-fy-px`, and optional `--camera-cx-px` /
`--camera-cy-px`. Supplying these values records their provenance but does not
by itself mark the calibration externally validated.

Step 4 adds:

```text
<output-root>/<run-id>/04_geometry_scale/config_<hash>/
  geometry_store.json
  videos/<video-id>.geometry.json
  visualizations/step4_visualization_manifest.json
  visualizations/<video-id>/frames/*.png
  visualizations/<video-id>/*_step4_examples.png
  visualizations/<video-id>/*_camera_centric_points_3d.png
  visualizations/<video-id>/*_geometry_timeline.png
  visualizations/<video-id>/*_camera_motion_diagnostics.png
  visualizations/<video-id>/*_relative_static_scene.json
  visualizations/<video-id>/*_relative_static_sandbox_3d.png
  visualizations/<video-id>/relative_static_sandbox_components/*.png
  visualizations/<video-id>/depth_geometry_examples/*.png
  visualizations/<video-id>/*_step4_geometry.mp4
```

Each per-track observation stores the robust 3D median, IQR and MAD, pixel
support and valid-depth fraction, intrinsics/scale hypothesis IDs, reprojection
check, and source artifact links. With the current single-frame DA3 output,
these coordinates use `relative_unit`; `scale_to_meters` is deliberately absent.
Camera translation is reported only as a direction (`up_to_scale`). Ground
plane and metric scale remain explicitly `unobservable` until independent
evidence supports them.

With `--visualize-step4`, the canonical frames show the selected mask, stable
track ID, camera-centric XYZ median, Z interquartile range, valid-depth fraction,
and pixel centroid. Separate 16:9 plots show camera-frame 3D point sequences,
XYZ-versus-time uncertainty bands, and background-flow camera-motion residuals.
Depth example panels place the source frame beside the depth artifact actually
used for back-projection. Every output explicitly states its coordinate frame
and unit: current DA3 sequences are **not** labeled as metric world trajectories.
`step4_visualization_manifest.json` indexes all products and preserves this
semantic warning. Use `--no-step4-video` to omit the MP4 while retaining the
still frames and diagnostic plots.

The relative static-scene sandbox is a conservative first world-frame view.
It accumulates observable pairwise rotations and translation directions,
estimates normalized translation steps from repeated stationary-semantic tracks
when available, and places the resulting ego camera centers and static landmark
candidates in component-local coordinates. Traffic lights, signs, hydrants and
similar classes receive the static semantic prior; low-motion residual tracks
are used only as a marked fallback when no such class is available. Squares
denote candidates whose transformed observations cluster consistently, while
X markers expose inconsistent candidates. Failed pose links start a new local
component rather than being silently bridged. The overview gives every
component its own 3D subplot and each component is also exported as an
independent 16:9 figure. The forward axis is elongated according to the local
motion extent rather than forced into a cube; only the first and last frame of
each segment are labeled. The companion JSON stores every pose, scale cue,
transform, landmark observation, spread and limitation.

For a useful ego path, use a denser geometry timeline than the 0.2 FPS quick
debug default. A one-video reconstruction run is:

```bash
./d4.sh run --gpu 0 --step 4 --data 1 --seed 1 \
  --canonical-fps 5 --diagnostics
```

This is substantially more expensive because YOLO, SAM 2, RAFT and DA3 process
the denser canonical sequence. It remains normalized relative reconstruction,
not metric trajectory ground truth.

For remote Docker execution, use the independent D4 launcher:

```bash
./d4.sh run --gpu 0 --step 4 --scale debug --seed 1 --diagnostics
```

D4 writes under `CAUVID_OUTPUT_D4_HOST` (default
`pipeline_august_target`) and refuses a path overlapping D3's output tree.

## Target Step 5: Joint Ego/Object World Reconstruction

Step 5 consumes the immutable `GeometryStore` and creates the initial world-
state beam $\mathcal B_0$. It accumulates only supported pairwise camera-pose
edges, keeps failed-link components in independent local frames, transforms
camera-centric object observations into those component frames, subtracts ego
motion, and estimates initial ego/object velocity with propagated intervals.
Static, moving, ambiguous, and unobservable are explicit states; a semantic
static class is only a prior and never overrides contradictory motion evidence.

```bash
python -m src.exp_august.inference.runner \
  --video-count 1 \
  --max-step 5 \
  --canonical-fps 5 \
  --objects-backend yolo_world \
  --masks-backend sam2 \
  --flow-backend raft \
  --depth-backend da3 \
  --world-top-k 5 \
  --visualize-step5 \
  --device cuda:0
```

Step 5 adds:

```text
<output-root>/<run-id>/05_world_reconstruction/input_<geometry-hash>/config_<hash>/
  world_state_store.json
  videos/<video-id>.world_state.json
  visualizations/step5_visualization_manifest.json
  visualizations/<video-id>/initial_world_hypothesis_3d.png
  visualizations/<video-id>/initial_motion_intervals.png
  visualizations/<video-id>/components/*.png
  visualizations/<video-id>/step5_summary.json
```

`WorldHypothesis`, `EgoPoseComponent`, `ObjectTrajectoryHypothesis`, uncertainty
fields, unresolved observations, construction score, and `HypothesisBeam` are
strict immutable contracts. With current DA3 evidence, output remains in
`relative_unit`; velocities are relative units per second, not m/s. A missing
pose edge is never interpolated. `top_k` is a capacity rather than a required
count. The current implementation emits evidence-distinct scale branches and,
for ambiguous object motion, auditable one-variable `static`/`moving`
alternatives. It retains the unconstrained ambiguous parent and does not form
the combinatorial product merely to fill the beam.

Step 5 visualization shows the best initial hypothesis, not verified truth.
The component-local 3D plots contain the ego path and ego-compensated object
paths; motion plots show speed intervals. All summaries declare
`step6_verified: false`. Step 6 evaluates every beam member independently; it
does not retroactively turn the Step 5 rank-1 construction into verified truth.

For the remote Docker launcher:

```bash
./d4.sh run --gpu 0 --step 5 --data 1 --seed 1 \
  --canonical-fps 5 --diagnostics
```

## Target Step 6: Forward Prediction and Consistency Verification

Step 6 consumes the immutable Step 5 beam and the archived Step 2-4 evidence.
It forward-projects every available hypothesis, evaluates five separately
auditable residual families, and emits one `HypothesisResidualPacket` per beam
member. The stage never edits a trajectory and never selects a winner.

The implemented baseline checks:

- fitted object-centroid reprojection, explicitly labeled as fit/self-
  consistency rather than independent validation;
- seeded held-out object depth where interpolation and a pose make it
  observable;
- held-out backward RAFT flow for object trajectories;
- rigid-background flow predicted from ego pose and depth, checked against
  held-out backward RAFT flow;
- object temporal gaps, metric speed/acceleration bounds when metric scale is
  available, relative acceleration diagnostics otherwise, and a soft
semantic-static prior.

Flow verification retains endpoint error as the primary residual, augments the
configured pixel uncertainty with the median RAFT forward/backward consistency
error, and records direction error plus a symmetric magnitude ratio. This
separates direction conflicts from motion-scale conflicts without normalizing
the two vectors independently.

Missing poses, masks, depth, or temporal support produce `not_evaluable`
records, not zero residuals or violations. Every evaluable held-out result
retains its evidence key and content-addressed artifact reference. Current
limitations are also explicit: dense mask rendering, lifecycle-cause testing,
road-context-conditioned physical limits, jerk/curvature/yaw checks, and
calibrated predictive uncertainty remain to be implemented.

```bash
python -m src.exp_august.inference.runner \
  --video-count 1 \
  --max-step 6 \
  --canonical-fps 5 \
  --objects-backend yolo_world \
  --masks-backend sam2 \
  --flow-backend raft \
  --depth-backend da3 \
  --world-top-k 5 \
  --visualize-step6 \
  --device cuda:0
```

Step 6 adds:

```text
<output-root>/<run-id>/06_predict_verify/input_<world-hash>/config_<hash>/
  residual_store.json
  videos/<video-id>.residuals.json
  visualizations/step6_visualization_manifest.json
  visualizations/<video-id>/hypothesis_comparison.png
  visualizations/<video-id>/rank_<rank>_residual_timeline.png
  visualizations/<video-id>/rank_<rank>_family_summary.png
  visualizations/<video-id>/rank_<rank>_conflict_overview.png
  visualizations/<video-id>/rank_<rank>_conflicts.json
  visualizations/<video-id>/rank_<rank>_conflicts/*.png
```

The comparison plot reports conflict counts and evidence coverage across the
beam but is not a new ranking score. Each conflict panel displays the concrete
video frame, selected track mask/box when applicable, predicted-versus-observed
pixel or flow marks, residual magnitude, evidence role, cue family, component,
track, and whether the window is supported by held-out evidence. Flow arrows
use one shared display scale, with predicted motion in the same red
used by the right-hand `Predicted` label and RAFT evidence in the same blue used
by `Observed`. Flow panels also state direction error, magnitude ratio, and the
resulting conflict type. Background arrows are explicitly identified as
spatial-median summaries because their residual remains the more robust median
of per-point endpoint errors. By default at
most eight conflict panels are rendered per hypothesis; change this with
`--step6-maximum-conflict-panels`. Every individual conflict panel is rendered
at 1920x1080. The 2x2 conflict overview preserves each panel at that resolution
and is therefore rendered at 3840x2160 rather than downsampling its subfigures.

On the remote server:

```bash
./d4.sh run --gpu 0 --step 6 --data 1 --seed 1 \
  --canonical-fps 5 --diagnostics
```

## Legacy linear baseline

The legacy `exp_august` path is:

`Raw Video -> Object Tracks -> 3D Trajectories -> Trajectory Refinement -> Ego/Object Motion -> Temporal Segmentation -> Symbolic Representation`

Run the legacy baseline independently with:

```bash
python -m src.exp_august.pipeline --video-count 1 --max-step 11
```

## Legacy Step 3: object mask tracking

Step 3 now uses a detector-guided hybrid tracker. ByteTrack first supplies
bootstrap IDs and box prompts. When a local SAM 2 checkpoint is available,
`SAM2VideoPredictor` propagates one binary masklet per object and a Hungarian
assignment reconciles it with each frame's available mask, RAFT flow, box,
class, and depth evidence. Missing cues are omitted and the remaining weights
are renormalized; a box is never persisted or reported as a SAM mask.

The default `auto` backend remains runnable on machines without SAM 2 weights:
it falls back to ByteTrack and records `mask_semantics: none` plus the exact
reason in `tracks.json`. Select `hybrid_mask` or strict mode when an experiment
must fail rather than use the fallback.

```bash
# Local checkpoint, with an error if hybrid tracking cannot run
python -m src.exp_august.pipeline --video-count 1 --max-step 3 \
  --tracking-backend hybrid_mask \
  --sam2-model weights/sam2/sam2_t.pt \
  --sam2-device cuda:0 \
  --mask-tracking-strict

# Permit Ultralytics to obtain the named checkpoint when it is not local
python -m src.exp_august.pipeline --video-count 1 --max-step 3 \
  --tracking-backend auto --sam2-model sam2_t.pt --sam2-allow-download
```

The VS Code debug configuration selects `auto` and looks for
`weights/sam2/sam2_t.pt`. Step 3 writes `mask_tracking_manifest.json`; aligned
per-frame fields include `mask_paths`, `mask_sources`, `association_scores`,
`tracking_confidences`, `visibility_states`, and `association_evidence`.

## Docker usage (`d3.sh`)

Run these commands from the repository root. `d3.sh` uses the same command
style as the July `d2.sh` launcher, but runs only `exp_august` and writes to a
separate August output directory.

Standard runs are isolated by scale and seed:

```text
pipeline_august/
  debug/seed_726381/   # 10 videos
  debug/seed_184957/
  debug/seed_930241/
  small/seed_.../      # 100 videos per seed
  full/seed_.../       # 961 videos per seed
  custom_N/seed_.../   # explicit --data N runs
```

Each run directory contains that run's pipeline artifacts and its own
`evaluation/` directory. If `CAUVID_AUGUST_EVALUATION_HOST` is set, the same
`<scale>/seed_<value>/` hierarchy is created below that evaluation root.
On the first run, `data_split_manifest.json` records a deterministic 70/15/15
train/eval/test partition. The manifest is authoritative on reruns: the same
scale and seed reuse exactly the same video IDs, and missing videos or a
seed/count mismatch produce an error instead of silently changing the split.
Test videos are selected exclusively from valid segment annotations under
`annotations/` (override with `CAUVID_ANNOTATIONS_PATH`). If fewer annotated
videos exist than the nominal 15% test target, all available annotated videos
are used for test, eval remains 15%, and the remainder is assigned to train.
The manifest records both requested and realized counts.

Step 6 enforces the persisted split as a strict test holdout. Static metadata
bucket boundaries, cohort rules, cohort statistics, and statistical policy
updates are fitted on train only. Repair-parameter calibration and policy
validation use eval only. Test trajectories are assigned with the frozen
train-fitted transform and receive the frozen repair policy, but cannot update
any fitted policy, statistic, or threshold. Test inference artifacts may be
cached, but are never consumed by fitting or calibration. The audit is written to
`06_trajectory_refinement/strict_test_holdout_audit.json`.

Step 5 uses the canonical nested sequence
`05_ego_motion_abstraction/05a_ego_motion/`,
`05_ego_motion_abstraction/05b_ego_axis_threshold_segmentation/`, and
`05_ego_motion_abstraction/05c_ego_axis_consensus_segmentation/`.

Full-scale runs automatically enable Weights & Biases using project
`cauvid-exp-august`, group `full`, and run name `full-seed-<seed>`. Set
`WANDB_API_KEY` on the host before launching. Debug, small, and custom scales
do not enable W&B automatically.

```bash
# Show all options
./d3.sh --help

# Run the default debug scale (10 videos), seed 1, on all GPUs
./d3.sh

# Run the three standard scales
./d3.sh run --gpu 0 --scale debug # 10 videos
./d3.sh run --gpu 0 --scale small # 100 videos
./d3.sh run --gpu 0 --scale full  # 961 videos

# Each scale supports seed indexes 1-3 (726381, 184957, 930241)
./d3.sh run --gpu 0 --scale small --seed 2

# --data selects a custom scale; --video-count is an alias
./d3.sh run --gpu 0 --data 25 --seed 3

# Stop after temporal video segmentation
./d3.sh run --gpu 0 --step 8 --scale debug

# Run the pipeline and immediately evaluate its test split
./d3.sh run --gpu 0 --step 8 \
  --scale debug --evaluate --split test --seed 1 \
  --test-ratio 0.2 --tolerances 1,3,5,10

# Evaluate existing cached August predictions without rerunning the pipeline
./d3.sh evaluate --scale debug --split test --seed 1 \
  --test-ratio 0.2 --tolerances 1,3,5,10

# Generate optional trajectory visualizations, dashboards, and audits
./d3.sh run --gpu 0 --step 11 --data 10 --diagnostics

# Also render the optional ego candidate-filter comparisons
./d3.sh run --gpu 0 --step 11 --data 10 \
  --render-candidate-filter-comparisons

# Build the Docker image or open an interactive container shell
./d3.sh build
./d3.sh shell --gpu 0
```

### Public step numbers

| Step | Module |
| ---: | --- |
| 1 | Dataset Initialization |
| 2 | Object Detection |
| 3 | Object Mask Tracking |
| 4 | 3D Trajectory Construction |
| 5 | Ego Motion Abstraction |
| 6 | Trajectory Refinement |
| 7 | Relative Motion Representation |
| 8 | Temporal Video Segmentation |
| 9 | Segment Motion Abstraction |
| 10 | Important Object Selection |
| 11 | Symbolic Scene Representation |

`--step N` executes through the selected step. Valid values are 1–11; the
default is 11.

### Host paths and environment variables

By default, the launcher expects the prepared `driving_mini` dataset under
`/storage-02/ml-jsha/driving_mini` and writes August artifacts under
`/storage-01/ml-jsha/storage/CauVid_output/pipeline_august`. Override these
locations when needed:

```bash
export CAUVID_DRIVING_MINI_HOST=/path/to/driving_mini
export CAUVID_OUTPUT_AUGUST_HOST=/path/to/pipeline_august
export CAUVID_AUGUST_EVALUATION_HOST=/path/to/august_evaluation
./d3.sh run --gpu 0 --step 11 --data 10 --evaluate --split test
```

Other useful overrides are:

- `CAUVID_IMAGE_NAME` — Docker image name; default: `cauvid:latest`.
- `CAUVID_CONTAINER_NAME` — container name; default: `cauvid-exp-august`.
- `CAUVID_GPU_ID` — default GPU ID or `all`.
- `CAUVID_STORAGE_ROOT` — base host storage location.
- `CAUVID_OUTPUT_ROOT` — base host output location.
- `CAUVID_TORCH_CACHE_HOST` — host directory for the mounted Torch cache.
- `CAUVID_AUGUST_EVALUATION_HOST` — evaluation output root; by default,
  `<pipeline_august>/evaluation`.

If trajectory clustering or repair requires the configured model service,
export `OPENAI_API_KEY` and any applicable `OPENAI_BASE_URL` or `OPENAI_MODEL`
before invoking `d3.sh`. The launcher forwards these variables into the
container without embedding their values in the script.

Its default artifacts are written below the configured pipeline output at
`output/pipeline_august/<scale>/seed_<seed>/`. Step 1 creates the complete run
directory recursively when it does not exist. Pass `--output-root PATH` (or set
`CAUVID_AUGUST_OUTPUT_PATH`) to isolate a particular run. The runner scopes
July's `CAUVID_PIPELINE_OUTPUT_PATH` internally and restores it afterward.

The eleven public modules are listed in `PIPELINE_STEPS`. July's internal
8A-8K operations remain internal to trajectory refinement. Media reports and
dashboards run only with `--diagnostics`. The pipeline always stops after logic
atoms/symbolic scene representation; it has no rule-learning imports or CLI
options.

Evaluation consumers should use `temporal_segments` for boundary/label metrics
and `symbolic_scene_representation` (plus `ego_motion` and
`segment_object_motion`) for symbolic metrics. `exp_august_traceability.json`
records the cross-module lineage and preserved confidence/provenance fields.

Step 8 automatically evaluates its annotated test partition and writes a
single-page subplot report to
`08_temporal_video_segmentation/evaluation/test/step_08_test_evaluation_charts.pdf`.
Coverage and aggregate scalar results are presented as tables; boundary
performance, per-class F1, and the confusion matrix are presented as charts.
The same directory contains the JSON and CSV evaluation artifacts.

## Video-segmentation evaluation

The standalone evaluator reads the existing manual schema under
`annotations/video_segmentation/` and matches annotations to August outputs by
the source video ID. It does not fit thresholds or modify predictions.

```bash
python -m src.exp_august.evaluation \
  --predictions /path/to/pipeline_august \
  --annotations annotations/video_segmentation \
  --output /path/to/evaluation/test \
  --split test \
  --seed 20260809 \
  --test-ratio 0.2 \
  --tolerances 1 3 5 10
```

Use `--split eval` for parameter development (`dev` remains a compatibility
alias). The run-level deterministic manifest is reused by every evaluation;
its test IDs are explicitly evaluation-only. `--split all` is available for
descriptive whole-dataset reporting, not model selection.

The annotations use raw-video frames while August predictions use the prepared
lower-rate timeline. The adapter maps prediction intervals onto the annotated
full-duration timeline before scoring. Adjacent segments carrying the same
semantic label are treated as one segment, so redundant same-label keyframes do
not create artificial boundaries.

Outputs include:

- `evaluation_results.json` — configuration, matching audit, per-video metrics,
  aggregate metrics, confusion matrices, and optimal segment matches.
- `split_manifest.json` — reproducible dev/test membership.
- `per_video_metrics.csv` and `per_video_class_metrics.csv`.
- `aggregate_metrics.csv`, `boundary_metrics.csv`, `segment_matches.csv`, and
  `confusion_matrix.csv`.
- `confusion_matrix.svg` and `metric_summary.svg` — vector publication plots.

The CLI finishes by printing a one-line summary suitable for an experiment
log. The default annotation lookup accepts both the repository's correctly
spelled `annotations/` directory and the legacy `annotaions/` spelling.
