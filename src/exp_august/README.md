# exp_august

`exp_august` is the Paper-1-only pipeline:

`Raw Video -> Object Tracks -> 3D Trajectories -> Trajectory Refinement -> Ego/Object Motion -> Temporal Segmentation -> Symbolic Representation`

Run it independently with:

```bash
python -m src.exp_august.pipeline --video-count 1 --max-step 11
```

## Step 3: object mask tracking

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
