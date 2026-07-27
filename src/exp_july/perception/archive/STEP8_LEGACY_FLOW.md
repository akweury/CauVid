# Archived Step 8 execution flow

The following calls are intentionally disabled in `src/exp_july/pipeline.py`:

- `step8_threshold_epoch_begin`
- legacy Step 8D pattern-refined validation
- legacy Step 8E semantic protection and its visualization
- legacy Step 8F final validation
- legacy Step 8G ego-motion refinement
- legacy Step 8H important-object video visualization
- legacy Step 8I threshold calibration

Their implementations remain importable for reproducibility of older runs, but
they are not part of the active pipeline. The active direction is the decoupled
Step 8C–8K trajectory clustering, repair, validation, statistics,
materialization, visualization, audit, and handoff flow.
