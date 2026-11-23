# Shape-phase atoms (`shape.py`, `--shape_phases`, 2026-11-22)

## Why

The rule model trails the MLP most on lane heads. In the rule lab on 1ca554 (val), compared with the MLP:

| Head | Rules | MLP |
|---|---|---|
| VehLane | 0.38 | 0.61 |
| MovRht | 0.18 | 0.33 |
| IncomLane | 0.40 | 0.50 |

The lane clips (`lane_clips.py`) and the user's review show that these onsets are visible in the box before
the label changes. Two examples:
- An oncoming car crossing the junction straight: not seen, then `Jun`, then `IncomLane`. Its box moves left or
  down-left and grows.
- A truck from the left turning right towards the ego: a tall box on the left moves right and gets wider, then
  taller, then narrower while it keeps getting taller.

The 3x3 grid and the motion atoms (`motion.py`) cannot express these. They do not separate width from height,
and they keep no order. Grid moves derived in the rule lab gave nothing.

## Atoms

All atoms are inputs only and are read in the agent's own scope.

**Primitives.** Each is measured over K = 2 frames (v2; v1 used 5) and must hold for 3 frames in a row:
- `x_left` / `x_right`: the centre's x, in box widths;
- `y_up` / `y_down`: the centre's y, in box heights;
- `w_grow` / `w_shrink`: log width;
- `h_grow` / `h_shrink`: log height.

The thresholds are the train quantiles 0.15 / 0.85 per class (`fit_shape`, in `shape_spec*.json`). FLOORS keep
them above the jitter of the GT boxes. A box cut by the image border gives no signal.

**Phases.** A phase is a run of frames with the same (x, y, w, h) state. A run shorter than 3 frames does not
count as a phase. The atoms are:
- `shp:p{k}_{primitive}`: phase k held that primitive (1 = the current phase, up to 4);
- `shp:p{k}_still`: phase k held none.

Phases carry no duration: a slow and a fast vehicle give the same sequence.

**Track facts:**
- `shp:app_*`: the box when the track appeared: its aspect tertile of the class (tall / square / wide), and its
  image third in x and in y;
- `shp:{tall,square,wide}`: the box now;
- `shp:young`: the track appeared less than 2 s ago.

**Order-free atoms (v2, 2026-11-23).** The user's principle: a rule should keep the one step that tells a
true onset from a false one, not the whole sequence. The atoms are:
- `shp:rec_{primitive}`: some phase of the last 4 held the primitive;
- `shp:rec_{a}+{b}`: some phase held both primitives at once, e.g. `rec_w_shrink+h_grow`, the box gets narrower
  and taller (turning towards the ego).

This gives 8 + 24 atoms, 81 shape atoms in all.

**360-degree directions (v3, 2026-11-23, the user: 360 degrees, not 8 directions).**
- `shp:dir_θ`: the move now (over K = 2 frames, at least 3 px, debounced) points into the half circle
  [θ, θ + 180).
- `shp:path_θ`: the same for the path since the track appeared (at least 0.3 box heights).
- θ = 0, 5, .., 355: 72 atoms each. Angles: 0 right, 90 up, 180 left, 270 down, in image pixels.
- Two half circles make any sector up to 180 degrees at a 5 degree resolution, also across 0:
  `dir_200 & dir_060` = [200, 240), left and a little down. The rule lab prints them merged (`dir[200, 240)`).

That makes 225 shape atoms in all. In the rule lab the shape atoms are read in the newest slot
only (`--lab_shape_slot1`): their history is already in the phases.

**Results of v1 (rule lab, run 20261122-shape_lab, roots fixed, val, 89 heads).**
- Rules with the shape atoms vs without, relaxed+precise_nn / sparse_c: +0.012 macro AP, 38 heads better and
  26 worse.
- VehLane 0.27 -> 0.39, TurRht 0.34 -> 0.45 and Brake 0.24 -> 0.31 gained; MovAway 0.30 -> 0.22 and HazLit
  lost.
- The MLP did not gain: 0.270 vs 0.283 without the shape atoms.

**Example.** The truck turning right towards the ego, at the frame where it narrows:
`shp:app_tall & shp:app_left & shp:p3_x_right & shp:p3_w_grow & shp:p2_h_grow & shp:p1_w_shrink & shp:p1_h_grow`.

## Test (rule lab, new run)

```bash
./run_roadpp_docker.sh --gpu 0 --rule_lab -- --exp sumo_maps_v7 --machine dgx-jing --shape_phases True \
    --run_id 20261123-shape2_lab --lab_variants relaxed_nn_ns,relaxed_nn,relaxed+precise_nn_ns,relaxed+precise_nn
```

What the run does:
- It builds the training data with the shape atoms. This is a new data file, because the data tag has
  `_shp2d3m3p4v3` (v1: `_shp5d3m3p4v1`).
- It trains the OnsetMLP on the same inputs (`rule_lab.ensure_mlp`). No rule net is trained.
- It compares the rule sets with (`relaxed_nn`) and without (`relaxed_nn_ns`) the shape literals, against an MLP
  that has them too.

The module stays off by default until the lab shows a clear gain on the lane heads.
