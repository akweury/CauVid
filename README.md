# CauVid
video reasoning by nesy and causal models

---
## 


## Setup
0. make sure using Python 3.10
1. setup depth anything v3: `python setup_depth_anything_v3.py` 
2. unzip two_parts.zip under the folder ./CauVid/
3. run `pip install -r requirements.txt`
4. On a remote server, set the data root for Docker/Python runs:
   ```bash
   source ./set_cauvid_data_root.sh /storage-01/ml-jsha/CauVid_Data
   ```
5. Prepare the driving dataset if needed:
   ```bash
   ./docker.sh prepare --gpu 2 --limit 961 --target-fps 5 --generate-depth --skip-existing
   ```
6. First-time run with no precomputed cache:
   ```bash
   ./docker.sh run --gpu 2 --recompute-preset full_pipeline 18
   ```
7. First-time 200-video run:
   ```bash
   ./docker.sh run --gpu 2 --data 200 --recompute-preset full_pipeline 18
   ```
8. Run downstream diagnostics after Step 18 artifacts exist:
   ```bash
   ./docker.sh run --gpu 2 --start-step 19 25
   ```

## exp_july Base Usage

Use `d2.sh` to run the lightweight `exp_july` pipeline in Docker on the remote server.

Steps 8C and 8E call an OpenAI-compatible LLM. Export the API key before
running through Step 8 or later; `d2.sh` forwards it into the container:

```bash
export OPENAI_API_KEY="..."
./d2.sh run --gpu 0 --step 8
```

For a compatible non-default endpoint, also set `OPENAI_BASE_URL` and either
`OPENAI_MODEL` or `CAUVID_STEP8_PATTERN_LLM_MODEL`.

The pipeline creates one Weights & Biases run per invocation. It records
per-step latency and compact state/manifest metrics, uploads a capped sample of
the important Step 8H videos, and stores generated manifests, review packages,
policy decisions, and threshold-conflict reports as a capped audit artifact.
With an API key it uses online mode; without one it runs offline and writes W&B
data below the pipeline output's `wandb` directory. Rebuild an existing Docker
image once to install the new dependency:

```bash
./d2.sh build
```

```bash
# Sign in to hosted W&B in a browser and create/copy an API key:
# https://wandb.ai/settings
export WANDB_API_KEY="..."
export CAUVID_WANDB_PROJECT="cauvid-exp-july"
export CAUVID_WANDB_RUN_NAME="step8-review"
./d2.sh run --gpu 0 --step 8 --data 50
```

W&B defaults to hosted web tracking in online mode at
`https://api.wandb.ai`; it does not default to a local W&B server. For a remote
run, authenticate in the browser at `https://wandb.ai/settings`, then export
the API key on the server before invoking `d2.sh`. A self-hosted endpoint is
used only when `CAUVID_WANDB_BASE_URL` is explicitly set.

Tracking is fail-open: a missing login, initialization problem, or upload
problem is reported but does not stop video processing. Use
`CAUVID_WANDB_ENABLED=0` to disable it, or set `CAUVID_WANDB_MODE=offline`
explicitly. `CAUVID_WANDB_MAX_VIDEOS` (default
4, one representative track per video) and
`CAUVID_WANDB_MAX_ARTIFACT_FILES` (default 200) control upload volume.
`CAUVID_WANDB_INIT_TIMEOUT_SECONDS` defaults to 30 seconds. A relative
`CAUVID_WANDB_DIR` is resolved below the pipeline output; an absolute override
used with Docker must be a container-visible path.

Step 8C processes each epoch deterministically with a frozen versioned policy.
It aggregates track outcomes and requests one candidate policy patch per review
interval (500 tracks by default). Configure the interval on the host with:

```bash
export CAUVID_STEP8C_REVIEW_INTERVAL_TRACKS=500
```

Candidate patches are compiled and compared with the active policy on a stable,
video-disjoint validation split. A successful candidate is staged as pending;
it is activated only when the next Step 8C epoch starts.

Step 8I also aggregates final Step 8F conflicts where a grounded semantic rule
protects a trajectory that the raw validator would discard. It deterministically
calibrates only bounded soft motion thresholds, validates the candidate on a
fixed video split, and activates successful patches at the next Step 8 boundary.
Semantic protection is treated as a weak retention label, not proof that a
trajectory is physically valid. Identity, continuity, malformed geometry, and
abrupt direction-change checks remain locked.

Optional calibration controls:

```bash
export CAUVID_STEP8_THRESHOLD_MIN_CONFLICTS=10
export CAUVID_STEP8_THRESHOLD_TARGET_QUANTILE=0.90
export CAUVID_STEP8_THRESHOLD_MAX_RELATIVE_CHANGE=0.10
export CAUVID_STEP8_THRESHOLD_MAX_UNPROTECTED_FLIP_RATE=0.02
export CAUVID_STEP8_THRESHOLD_VALIDATION_FRACTION=0.20
```

Example:
```bash
./d2.sh --gpu 1 --step 2 --data 20
```

Meaning:
- `--gpu 1`: run inside Docker on GPU 1
- `--step 2`: run the pipeline only up to Step 2
- `--data 20`: use the first 20 videos from `driving_mini`

More examples:
```bash
./d2.sh
./d2.sh run --gpu 0 --step 18 --data 50 --rounds 3
./d2.sh shell --gpu 0
```

Outputs for this pipeline go to:
```bash
/storage-01/ml-jsha/storage/CauVid_output/output_july
```
