# exp_august

`exp_august` is the Paper-1-only pipeline:

`Raw Video -> Object Tracks -> 3D Trajectories -> Trajectory Refinement -> Ego/Object Motion -> Temporal Segmentation -> Symbolic Representation`

Run it independently with:

```bash
python -m src.exp_august.pipeline --video-count 1 --max-step 11
```

## Docker usage (`d3.sh`)

Run these commands from the repository root. `d3.sh` uses the same command
style as the July `d2.sh` launcher, but runs only `exp_august` and writes to a
separate August output directory.

```bash
# Show all options
./d3.sh --help

# Run all 11 stages with the default data selection and all GPUs
./d3.sh

# Run 10 videos through Step 11 on GPU 0
./d3.sh run --gpu 0 --step 11 --data 10

# --video-count is an alias for --data
./d3.sh run --gpu 0 --step 11 --video-count 10

# Stop after temporal video segmentation
./d3.sh run --gpu 0 --step 8 --data 10

# Run the pipeline and immediately evaluate its test split
./d3.sh run --gpu 0 --step 8 \
  --evaluate --split test --seed 20260809 \
  --test-ratio 0.2 --tolerances 1,3,5,10

# Evaluate existing cached August predictions without rerunning the pipeline
./d3.sh evaluate --split test --seed 20260809 \
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
| 3 | Object Tracking |
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
`exp_august/`. Pass `--output-root PATH` (or set
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

Use `--split dev` for parameter development. The deterministic split manifest
is saved with every evaluation; its test IDs are explicitly evaluation-only.
`--split all` is available for descriptive whole-dataset reporting, not model
selection.

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
