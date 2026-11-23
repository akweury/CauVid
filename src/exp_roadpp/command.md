
./run_roadpp_docker.sh --gpu 4 --rule_tree -- --exp sumo_maps_v7 --machine dgx-jing --roles ahead --mode list --root "av:AV-Mov(own)@1" --hit_slots 2
./run_roadpp_docker.sh --gpu 1 --rule_tree -- --exp sumo_maps_v7 --machine dgx-jing --roles ahead --root "av:AV-Mov(own)@1 & veh:Brake(ahead)@1 & !veh:Stop(ahead)@1" --hit_slots 2 --invert --mode tree

./run_roadpp_docker.sh --gpu 1 --rule_tree -- --exp sumo_maps_v7 --machine dgx-jing --roles ahead --root 'av:AV-Mov(own)@1 & veh:Brake(ahead)@1 & !veh:Stop(ahead)@1' --hit_slots 2 --invert --mode list


./run_roadpp_docker.sh --gpu 2 --residual_clips -- --exp sumo_maps_v7 --machine dgx-jing --roles ahead --rule_list /storage-01/ml-jsha/CauVid_output/roadpp/sumo_maps_v7/rule_tree/av_AV-Brake__av_AV_Mov_own_1_rahead_w2_list.json
./run_roadpp_docker.sh --gpu 2 --residual_clips -- --exp sumo_maps_v7 --machine dgx-jing --roles ahead --rule_list /storage-01/ml-jsha/CauVid_output/roadpp/sumo_maps_v7/rule_tree/av_AV-Brake__av_AV_Mov_own_1_rahead_w2_list.json

./run_roadpp_docker.sh --gpu 7 --rule_tree -- --exp sumo_maps_v7 --machine dgx-jing --roles ahead --ego_motion True --mode list --root 'av:AV-Mov(own)@1' --hit_slots 2
./run_roadpp_docker.sh --gpu 7 --rule_tree -- --exp sumo_maps_v7 --machine dgx-jing --roles ahead --ego_motion True --mode list --root 'av:AV-Mov(own)@1' --hit_slots 2
./run_roadpp_docker.sh --gpu 7 --rule_tree -- --exp sumo_maps_v7 --machine dgx-jing --roles ahead --ego_motion True --head av:AV-Stop --mode list --root 'av:AV-Mov(own)@1' --hit_slots 2

./run_roadpp_docker.sh --gpu 7 --rule_tree -- --exp sumo_maps_v7 --machine dgx-jing --roles ahead --ego_motion True --mode tree --root 'av:AV-Mov(own)@1 & av:tl_grow(own)@1' --hit_slots 2 --max_depth 3
./run_roadpp_docker.sh --gpu 7 --rule_tree -- --exp sumo_maps_v7 --machine dgx-jing --roles ahead --ego_motion True --mode list --root 'av:AV-Mov(own)@1' --hit_slots 2 --min_prec 0.25
./run_roadpp_docker.sh --gpu 7 --residual_clips -- --exp sumo_maps_v7 --machine dgx-jing --roles ahead --ego_motion True --rule_list /storage-01/ml-jsha/CauVid_output/roadpp/sumo_maps_v7/rule_tree/av_AV-Brake__av_AV_Mov_own_1_rahead_w2_egov3_list.json --no_render

./run_roadpp_docker.sh --gpu 6 -- --exp sumo_maps_v7 --machine dgx-jing --num_epochs 50
./run_roadpp_docker.sh --gpu 6 -- --exp sumo_maps_v7 --machine dgx-jing --num_epochs 50 --run_id 20261120-132819_dgx-jing_a4e02b
./run_roadpp_docker.sh --gpu 4 -- --exp sumo_maps_v7 --machine dgx-jing --num_epochs 200
./run_roadpp_docker.sh --gpu 3 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --run_id 20261123-shape2_lab --lab_variants relaxed_nn_ns,relaxed_nn,relaxed+precise_nn_ns,relaxed+precise_nn



./run_roadpp_docker.sh --gpu 3 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True \
    --run_id 20261123-shape4_lab \
    --lab_variants relaxed+precise_nn_nd_np,relaxed+precise_nn_nd,relaxed+precise_nn_np,relaxed+precise_nn

./run_roadpp_docker.sh --gpu 3 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --gestalt True --run_id 20261123-gestalt7_lab --lab_mlp False --lab_fast True --lab_mdl 0.03,0.1 --lab_variants relaxed_nn_ng,relaxed_nn


### 跨类别共享模式
如果某条规则在某一个类别成立，那么测试其在其他类别上的适用性，如果适用率较高，则生成新的跨类别规则，无需从头重新挖掘规则。


### 运动轨迹RAW模式
对于同一个行为的运动轨迹，RAW模式记录每一帧的原始位置信息，而不进行任何离散化或特征提取。
将RAW格式的轨迹进行符号化处理，是否可以通过数学公式拟合？ 学出一个可以安全描述的运动范围？


./run_roadpp_docker.sh --gpu 6 --atom_probe -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --gestalt True --run_id 20261123-gestalt7_lab --probe_head Car:Jun


./run_roadpp_docker.sh --gpu 6 --atom_probe -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --gestalt True --run_id 20261123-gestalt7_lab --probe_head Car:Jun --probe_skip_rules True

./run_roadpp_docker.sh --gpu 6 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --gestalt True --run_id 20261123-gestalt8_lab --lab_prepare_only True

./run_roadpp_docker.sh --gpu 6 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --gestalt True --run_id 20261123-gestalt8_lab --lab_prepare_only True --lab_mlp False


./run_roadpp_docker.sh --gpu 3 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --gestalt True --run_id 20261123-gestalt8_lab --lab_mlp False --lab_variants relaxed_nn_ng --lab_root_exclude shape,gestalt 2>&1 | tee rule_lab_gst8_gpu2.log

./run_roadpp_docker.sh --gpu 1 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --gestalt True --run_id 20261123-gestalt8_lab --lab_mlp False --lab_variants relaxed_nn --lab_root_exclude shape,gestalt 2>&1 | tee rule_lab_gst8_gpu3.log


./run_roadpp_docker.sh --gpu 3 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --gestalt True --run_id 20261123-gestalt8_lab --lab_mlp False --lab_variants relaxed_nn --lab_root_exclude shape 2>&1 | tee rule_lab_gst8_gpu4.log

./run_roadpp_docker.sh --gpu 5 --atom_probe -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --gestalt True --run_id 20261123-gestalt8_lab --probe_head Car:Jun --probe_target eventual --probe_rules "Car:IncomLane(own)@1;Car:IncomLane(own)@1 & TL:Red(other)@1;Car:IncomLane(own)@1 & TL:Green(other)@1;Car:IncomLane(own)@1 & !Car:Jun(own)@1 & !Car:Jun(own)@2 & !Car:Jun(own)@3"

./run_roadpp_docker.sh --gpu 5 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --gestalt True --run_id 20261123-gestalt8_lab --lab_target eventual --lab_heads Car:Jun --lab_variants relaxed_nn_ng --lab_root_exclude shape,gestalt
