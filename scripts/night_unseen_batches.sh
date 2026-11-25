#!/bin/bash
# Overnight, step 2 (2026-11-24): the rule lab's fixed model on unseen maps (sumo_maps_v7_unseen), 4 head batches in
# parallel on GPUs 0-3, each in a detached screen session (night_u1 .. night_u4). Run after the data / OnsetMLP
# preparation of run 20261124-unseen_lab has finished, from the repo root:
#   ./run_roadpp_docker.sh --gpu 0 --rule_lab -- --exp sumo_maps_v7_unseen --machine dgx-jing --shape_phases True \
#       --run_id 20261124-unseen_lab --lab_prepare_only True 2>&1 | tee night_prepare.log && bash scripts/night_unseen_batches.sh
# Watch: screen -ls; screen -r night_u1. Results: <output_dir>/rule_lab/*_cv_mg_u1..u4.json.
# Extra rule-lab args go after the script; TAG names the batches (default u). Test maps (2026-11-25):
#   TAG=t bash scripts/night_unseen_batches.sh --lab_eval_split test   -> screens night_t1..t4, *_cv_mg_test_t1..t4.json
TAG=${TAG:-u}
EXTRA="$*"
COMMON="--exp sumo_maps_v7_unseen --machine dgx-jing --shape_phases True --run_id 20261124-unseen_lab"
LAB="--lab_fast True --lab_variants relaxed_nn,relaxed+exh_nn,relaxed+tr_nn --lab_check True --lab_merge True --lab_exh_gain 1.5 --lab_mdl 0.1 --lab_select_scorer mdl_0.1 --lab_select_min_pos 1000000"
B[1]="Car:Brake,Car:TurLft,Ped:Mov,MedVeh:Jun,Car:XingFmRht,MedVeh:TurRht,LarVeh:Jun,Car:XingFmLft,Car:IncatLft,MedVeh:XingFmRht,Car:IncomLane,Bus:XingFmLft,LarVeh:MovTow,Bus:XingFmRht,Cyc:Jun,av:AV-MovLft,MedVeh:HazLit,Ped:Xing,Cyc:VehLane,Mobike:MovAway,Cyc:MovTow,Mobike:XingFmRht,Mobike:TurLft,Bus:IncatRht"
B[2]="Car:Stop,av:AV-Stop,MedVeh:Brake,Car:OutgoLane,Ped:RhtPav,av:AV-TurLft,av:AV-TurRht,Ped:LftPav,LarVeh:MovAway,Car:IncatRht,MedVeh:OutgoLane,Bus:TurLft,MedVeh:XingFmLft,MedVeh:VehLane,Bus:OutgoLane,Cyc:Stop,Cyc:TurRht,Mobike:Stop,Cyc:MovAway,Ped:XingFmLft,Mobike:TurRht,LarVeh:IncatLft,MedVeh:MovRht,Bus:MovRht"
B[3]="Car:Jun,Car:MovAway,MedVeh:Stop,Car:MovTow,LarVeh:Brake,Ped:MovAway,MedVeh:TurLft,Bus:Stop,Ped:MovTow,MedVeh:MovTow,LarVeh:TurLft,LarVeh:XingFmRht,LarVeh:OutgoLane,LarVeh:XingFmLft,Cyc:Brake,Car:MovLft,Cyc:OutgoLane,MedVeh:IncatLft,Cyc:TurLft,MedVeh:MovLft,LarVeh:IncomLane,Mobike:Jun,MedVeh:IncatRht,Bus:VehLane"
B[4]="av:AV-Brake,av:AV-Mov,Car:TurRht,Ped:Wait2X,LarVeh:Stop,MedVeh:MovAway,Car:VehLane,Car:HazLit,Bus:Brake,LarVeh:TurRht,Bus:Jun,Bus:MovAway,Bus:TurRht,Car:MovRht,Bus:MovTow,av:AV-MovRht,LarVeh:HazLit,Mobike:Brake,Ped:Stop,Cyc:XingFmRht,Cyc:IncomLane,Ped:XingFmRht,LarVeh:VehLane,Cyc:MovRht,LarVeh:MovLft"
GPU=(-1 0 1 2 3)
for i in 1 2 3 4; do
  screen -dmS "night_$TAG$i" bash -c "./run_roadpp_docker.sh --gpu ${GPU[$i]} --rule_lab -- $COMMON $LAB $EXTRA --lab_tag $TAG$i --lab_heads ${B[$i]} 2>&1 | tee night_$TAG$i.log; exec bash"
  echo "[night] batch $TAG$i on GPU ${GPU[$i]}: screen -r night_$TAG$i"
done
