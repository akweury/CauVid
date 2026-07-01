"""Thin entrypoint for the driving_mini experiment pipeline.

The full orchestration, step handlers, summaries, and warm-start cache loading
live in `exp_driving_pipeline_runtime.py`. This file stays intentionally small
so the control surface is easy to read:

1. choose a start step
2. choose a stop step
3. optionally restrict videos
4. hand off to the runtime
"""

from __future__ import annotations

import argparse
from typing import Optional

from src.exp_driving_videos.exp_driving_pipeline_runtime import (
    main as _runtime_main,
    parse_args as _runtime_parse_args,
)

DEFAULT_START_STEP: int = 17
DEFAULT_MAX_STEP: int = 24

PIPELINE_STEP_OVERVIEW = [
    ("0-16", "upstream perception, symbolic abstraction, initial-rule pruning, rule extension"),
    ("17", "final rule selection"),
    ("17D", "optional rule-pool and selector diagnostics"),
    ("18", "selected-rule evaluation"),
    ("18B", "optional baseline comparison"),
    ("19", "optional OD calibration loop"),
    ("18D-24B", "optional downstream diagnostics and visualization"),
]


def parse_args() -> argparse.Namespace:
    """Delegate CLI parsing to the runtime module."""

    return _runtime_parse_args()


def main(
    start_step: int | str = DEFAULT_START_STEP,
    max_step: int | str = DEFAULT_MAX_STEP,
    video_ids: Optional[list[str]] = None,
    video_count: int | None = None,
    od_calibration_iterations: int | None = None,
    recompute_preset: str | None = None,
) -> None:
    """Run the pipeline through the shared runtime implementation."""

    _runtime_main(
        start_step=start_step,
        max_step=max_step,
        video_ids=video_ids,
        video_count=video_count,
        od_calibration_iterations=od_calibration_iterations,
        recompute_preset=recompute_preset,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        start_step=args.start_step,
        max_step=args.max_step,
        video_ids=args.video_ids,
        video_count=args.video_count,
        od_calibration_iterations=args.od_calibration_iterations,
        recompute_preset=args.recompute_preset,
    )
