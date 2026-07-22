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
