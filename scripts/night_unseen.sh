#!/bin/bash
# Overnight run (2026-11-24): the rule lab's fixed model on unseen maps (sumo_maps_v7_unseen) vs an OnsetMLP
# trained on the same data. 1) data + OnsetMLP + row sample (GPU 4), 2) the 4 head batches in parallel (GPUs 4-7,
# tmux sessions night_u1..night_u4), each: relaxed / relaxed+exh / relaxed+tr, MDL 0.1, transfer targets -> tr.
# Run from the repo root:  bash scripts/night_unseen.sh
set -e
COMMON="--exp sumo_maps_v7_unseen --machine dgx-jing --shape_phases True --run_id 20261124-unseen_lab"
LAB="--lab_fast True --lab_variants relaxed_nn,relaxed+exh_nn,relaxed+tr_nn --lab_check True --lab_merge True \
--lab_exh_gain 1.5 --lab_mdl 0.1 --lab_select_scorer mdl_0.1 --lab_select_min_pos 1000000"
B[1]="Car:Brake,Car:TurLft,Ped:Mov,MedVeh:Jun,Car:XingFmRht,MedVeh:TurRht,LarVeh:Jun,Car:XingFmLft,Car:IncatLft,MedVeh:XingFmRht,Car:IncomLane,Bus:XingFmLft,LarVeh:MovTow,Bus:XingFmRht,Cyc:Jun,av:AV-MovLft,MedVeh:HazLit,Ped:Xing,Cyc:VehLane,Mobike:MovAway,Cyc:MovTow,Mobike:XingFmRht,Mobike:TurLft,Bus:IncatRht"
B[2]="Car:Stop,av:AV-Stop,MedVeh:Brake,Car:OutgoLane,Ped:RhtPav,av:AV-TurLft,av:AV-TurRht,Ped:LftPav,LarVeh:MovAway,Car:IncatRht,MedVeh:OutgoLane,Bus:TurLft,MedVeh:XingFmLft,MedVeh:VehLane,Bus:OutgoLane,Cyc:Stop,Cyc:TurRht,Mobike:Stop,Cyc:MovAway,Ped:XingFmLft,Mobike:TurRht,LarVeh:IncatLft,MedVeh:MovRht,Bus:MovRht"
B[3]="Car:Jun,Car:MovAway,MedVeh:Stop,Car:MovTow,LarVeh:Brake,Ped:MovAway,MedVeh:TurLft,Bus:Stop,Ped:MovTow,MedVeh:MovTow,LarVeh:TurLft,LarVeh:XingFmRht,LarVeh:OutgoLane,LarVeh:XingFmLft,Cyc:Brake,Car:MovLft,Cyc:OutgoLane,MedVeh:IncatLft,Cyc:TurLft,MedVeh:MovLft,LarVeh:IncomLane,Mobike:Jun,MedVeh:IncatRht,Bus:VehLane"
B[4]="av:AV-Brake,av:AV-Mov,Car:TurRht,Ped:Wait2X,LarVeh:Stop,MedVeh:MovAway,Car:VehLane,Car:HazLit,Bus:Brake,LarVeh:TurRht,Bus:Jun,Bus:MovAway,Bus:TurRht,Car:MovRht,Bus:MovTow,av:AV-MovRht,LarVeh:HazLit,Mobike:Brake,Ped:Stop,Cyc:XingFmRht,Cyc:IncomLane,Ped:XingFmRht,LarVeh:VehLane,Cyc:MovRht,LarVeh:MovLft"
GPU=(0 4 5 6 7)

echo "[night] 1/2 data, OnsetMLP and row sample ($(date))"
./run_roadpp_docker.sh --gpu 4 --rule_lab -- $COMMON --lab_prepare_only True 2>&1 | tee night_prepare.log

echo "[night] 2/2 four head batches on GPUs 4-7 ($(date)): tmux attach -t night_u1 .. night_u4"
for i in 1 2 3 4; do
  tmux new-session -d -s "night_u$i" \
    "./run_roadpp_docker.sh --gpu ${GPU[$i]} --rule_lab -- $COMMON $LAB --lab_tag u$i --lab_heads ${B[$i]} 2>&1 | tee night_u$i.log; read"
done
echo "[night] started; results: <output_dir>/rule_lab/*_cv_mg_u1..u4.json"
