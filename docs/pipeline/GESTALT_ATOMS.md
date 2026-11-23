# Gestalt atoms (`gestalt.py`, `--gestalt`, 2026-11-23)

## Why

Several boxes moving the same way at once says what an agent will do, beyond its own box. Examples: a queue
starting at a green light, a group of pedestrians crossing, the traffic flowing right while one car stays.

The rules already read the other agents' labels, pooled over the whole scene (`veh:X(other)`). They do not read
the other agents' motion: pooled over the scene it says little (`step05.rule_input_mask`). The gestalt atoms
describe the agent by the boxes near it or similar to it. They follow three grouping principles: proximity,
similarity and common fate.

## Atoms

There are 27 atoms. They are inputs only, read in the agent's own scope, and must hold for 3 frames in a row.
They are computed per frame from the GT boxes; the ego has no box.

**Definitions:**
- **Move:** the centre over K = 2 frames. It is a direction if it moved at least 3 px, "still" if less.
- **Near:** the two centres are closer than 2 x the mean of the boxes' larger sides.
- **Similar:** the height ratio is below 1.5.
- **Same / opposite / cross:** both move, and their directions differ by < 45 / > 135 degrees / in between.

**Atoms:**

| Atom | Holds when |
|---|---|
| `gst:near_{1,2,3}` | at least 1 / 2 / 3 near agents |
| `gst:near_same_cls`, `gst:near_similar` | a near agent of the same class / of a similar size |
| `gst:co_move_{1,2}`, `gst:co_move_similar` | near agents moving the same way as the agent (common fate) |
| `gst:counter_move`, `gst:cross_move` | a near agent moving the opposite way / across |
| `gst:others_move_i_still_{1,2}` | the agent is still while near agents move: the group starts, the agent not yet |
| `gst:i_move_others_still` | the agent moves, the near agents stand |
| `gst:near_flow_{right,up,left,down}` | the near movers agree on a direction (mean unit vector >= 0.5) in that quadrant |
| `gst:scene_{right,up,left,down}_{2,4}` | at least 2 / 4 other agents anywhere move in that quadrant |
| `gst:scene_co_move_{2,4}` | at least 2 / 4 other agents anywhere move the same way as the agent |

In the rule lab the gestalt atoms may not be mining roots (`--lab_root_exclude shape,gestalt`): `near_1` holds
almost always, as `shp:app_middle` did. The `_ng` variants go without them.

## Shape atoms v5

The test runs with the shape atoms v5: v4 without the fine position, size and distance atoms (rule lab v4: no
gain, 3x the rules).

## Test (2x2: shape x gestalt, 4 GPUs)

```bash
# 1. Once: the training data (shape v5 + gestalt), the OnsetMLP and the row sample.
./run_roadpp_docker.sh --gpu 2 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --gestalt True \
    --run_id 20261123-gestalt_lab --lab_prepare_only True
# 2. The four variants in parallel (tmux: docker -it needs a terminal).
for spec in "2 relaxed+precise_nn_ns_ng" "3 relaxed+precise_nn_ng" "4 relaxed+precise_nn_ns" "5 relaxed+precise_nn"; do
  set -- $spec
  tmux new-session -d -s gst_gpu$1 "./run_roadpp_docker.sh --gpu $1 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True --gestalt True --run_id 20261123-gestalt_lab --lab_variants $2 2>&1 | tee rule_lab_gst_gpu$1.log; read"
done
```
