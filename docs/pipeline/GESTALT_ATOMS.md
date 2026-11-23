# Gestalt atoms (`gestalt.py`, `--gestalt`, 2026-11-23)

## Why

Several boxes moving the same way at once says what an agent will do, beyond its own box. Examples: a queue
starting at a green light, a group of pedestrians crossing, the traffic flowing right while one car stays.

The rules already read the other agents' labels, pooled over the whole scene (`veh:X(other)`). They do not read
the other agents' motion: pooled over the scene it says little (`step05.rule_input_mask`). The gestalt atoms
describe the agent by the boxes near it or similar to it. They follow three grouping principles: proximity,
similarity and common fate.

## v1 result (rule lab, 2026-11-23, 83 heads, no MLP)

The main model (relaxed+precise_nn, shape v5) with the v1 atoms vs without them:

| Scorer | Difference | Heads better / worse | 95% CI |
|---|---|---|---|
| soft_logit | +0.006 | 25 / 25 | [-0.005, +0.018] |
| sparse_c | -0.002 | 24 / 33 | |
| crisp | -0.001 | | |

No clear gain. The v1 atoms were agent-relative counts (near_1..3, co_move_1/2, scene counts 2 / 4). Their rules
used mostly the negations as filler (`!gst:near_1` 78x, `!gst:scene_left_4` 54x).

## Atoms (v2, the user: two or three cars moving right together are the same fact)

A **group** is a set of at least 2 agents of one kind that are close and move the same way. Its size does not
matter.
- **Kinds:** vehicles (two-wheelers included) and pedestrians. Traffic lights belong to no group.
- **Close and same way:** centres closer than 2 x the mean of the boxes' larger sides; both moving (at least
  3 px over 2 frames); directions differing by < 45 degrees.
- **The group:** a connected component of these links.
- **Its direction:** the members' mean direction, as a quadrant (right / up / left / down).

24 atoms, own scope, debounced over 3 frames, for kind in {veh, ped} and quadrant q:

| Atom | Holds when |
|---|---|
| `gst:in_group_{kind}_{q}` | the agent is a member of such a group |
| `gst:near_group_{kind}_{q}` | a member of such a group is near the agent, which is not in it (e.g. the car that stays while the queue starts) |
| `gst:group_{kind}_{q}` | the scene has such a group |

**360-degree group directions (v3).** As the shape atoms' `dir_θ`, θ = 0, 5, .., 355:
- `gst:in_group_dir_θ`: the agent's own group moves into the half circle [θ, θ + 180);
- `gst:near_group_dir_θ`: the group of the nearest member of another group does.

Two of them give any sector. The rule lab prints them merged (`gst:in_group_dir[345, 350)`), narrows them to
where the onsets concentrate and merges close sectors. That makes 24 + 144 = 168 gestalt atoms.

**v4 (the user: every concept of a single box has its counterpart for a group).** A group is described as a box
of its own: the union of its members' boxes, frame by frame. Two group tracks per agent:
- `grp_`: the agent's own group;
- `ngrp_`: the group of the nearest member of another group.

Each track gets every shape atom (`shape.py`) and the members' labels. The v3 `in_group_dir` / `near_group_dir`
atoms are `grp_dir` / `ngrp_dir` now.

| Single box | Group counterpart |
|---|---|
| primitives, phases p1-p4, order-free `rec_*` | `gst:grp_p1_x_right`, `gst:grp_rec_w_grow+h_grow`, `gst:ngrp_...` |
| 360-degree `dir_θ` / `path_θ` | `gst:grp_dir_θ`, `gst:grp_path_θ` (the path since the group formed) |
| `app_*`, aspect, `young` | where the group formed, the group box's aspect, the group formed recently |
| labels (`Car:Brake`) | `gst:grp_any_Brake` (another member brakes), `gst:grp_all_Stop` (every member stands) |

Members join and leave, so the group box also grows or shrinks with the membership (the group spreads or
gathers). On SUMO v7 there are 24 + 2 x 225 + 4 x 22 = 562 gestalt atoms. In the rule lab they are read in the
newest slot only, as the shape atoms.

The negated gestalt literals are off in the rule lab (`--lab_gst_negated False`).

## Shape atoms v5

The test runs with the shape atoms v5: v4 without the fine position, size and distance atoms (rule lab v4: no
gain, 3x the rules).

## Test (2x2: shape x gestalt, 4 GPUs)

```bash
# 1. Once: the training data (shape v5 + gestalt), the OnsetMLP and the row sample.
./run_roadpp_docker.sh --gpu 2 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --gestalt True \
    --run_id 20261123-gestalt4_lab --lab_prepare_only True
# 2. The four variants in parallel (tmux: docker -it needs a terminal).
for spec in "2 relaxed+precise_nn_ns_ng" "3 relaxed+precise_nn_ng" "4 relaxed+precise_nn_ns" "5 relaxed+precise_nn"; do
  set -- $spec
  tmux new-session -d -s gst_gpu$1 "./run_roadpp_docker.sh --gpu $1 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --gestalt True --run_id 20261123-gestalt4_lab --lab_variants $2 2>&1 | tee rule_lab_gst_gpu$1.log; read"
done
```
