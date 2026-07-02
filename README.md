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
