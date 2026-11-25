#!/bin/bash
# Rule lab vs OnsetMLP on sumo_maps_v8 (2026-11-25): all learned heads split into one part per GPU
# (--lab_head_part i/k), each part in a detached screen session v8_p<i>. Evaluation on the odd val + all test videos
# (--lab_eval_split valtest) with per-head bootstrap intervals; the fixed policy: relaxed + exh with the size-scaled
# MDL fit (mdls_0.1), heads with < 60 train onsets relaxed + transfer.
# Before, once (from the repo root):
#   ./run_roadpp_docker.sh --gpu 0 -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True --dataset_only True
#   ./run_roadpp_docker.sh --gpu 0 --rule_lab -- --exp sumo_maps_v8 --machine dgx-jing --shape_phases True \
#       --run_id 20261125-v8_lab --lab_prepare_only True --lab_eval_split valtest
# Then: bash scripts/v8_batches.sh  (GPUS="4 5 6 7" bash scripts/v8_batches.sh for other GPUs; EXP / RUN_ID likewise)
# Watch: screen -ls; screen -r v8_p0. Results: <output_dir>/rule_lab/*_vt_p0.. .json
EXP=${EXP:-sumo_maps_v8}
RUN_ID=${RUN_ID:-20261125-v8_lab}
read -r -a G <<< "${GPUS:-0 1 2 3}"
K=${#G[@]}
COMMON="--exp $EXP --machine dgx-jing --shape_phases True --run_id $RUN_ID"
LAB="--lab_fast True --lab_variants relaxed_nn,relaxed+exh_nn,relaxed+tr_nn --lab_check True --lab_merge True \
--lab_exh_gain 1.5 --lab_mdl s0.1 --lab_select_scorer mdls_0.1 --lab_select_min_pos 1000000 --lab_eval_split valtest"
for ((i = 0; i < K; i++)); do
  screen -dmS "v8_p$i" bash -c "./run_roadpp_docker.sh --gpu ${G[$i]} --rule_lab -- $COMMON $LAB $* --lab_head_part $i/$K --lab_tag p$i 2>&1 | tee v8_p$i.log; exec bash"
  echo "[v8] part $i/$K on GPU ${G[$i]}: screen -r v8_p$i"
done
