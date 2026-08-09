"""Independent runner for the Paper-1 ``exp_august`` pipeline."""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.exp_august import modules


PIPELINE_STEPS = (
    "dataset_initialization",
    "object_detection",
    "object_tracking",
    "trajectory_construction_3d",
    "ego_motion_abstraction",
    "trajectory_refinement",
    "relative_motion_representation",
    "temporal_video_segmentation",
    "segment_motion_abstraction",
    "important_object_selection",
    "symbolic_scene_representation",
)


class _NullTracker:
    def log_state(self, _name, state, **_kwargs):
        return state

    def log_failure(self, _name, _error, **_kwargs):
        return None

    def finish(self, **_kwargs):
        return None


def _tracked(tracker, name: str, operation: Callable):
    started = time.perf_counter()
    try:
        state = operation()
        if not isinstance(state, dict):
            raise TypeError(f"{name} returned {type(state).__name__}; expected dict")
        if not state.get("videos"):
            raise RuntimeError(f"{name} produced no videos")
    except BaseException as exc:
        tracker.log_failure(name, exc, duration_seconds=time.perf_counter() - started)
        raise
    return tracker.log_state(name, state, duration_seconds=time.perf_counter() - started)


@contextmanager
def _august_output_environment(output_root: Path):
    """Scope July's environment-driven artifact paths to August only."""
    key = "CAUVID_PIPELINE_OUTPUT_PATH"
    previous = os.environ.get(key)
    os.environ[key] = str(output_root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def run_pipeline(
    *,
    video_ids=None,
    video_count=None,
    max_step: int = 11,
    output_root: Path | str | None = None,
    diagnostics: bool = False,
    render_candidate_filter_comparisons: bool = False,
    tracker=None,
):
    """Run August through ``max_step``; Step 11 is the hard final boundary."""
    max_step = int(max_step)
    if not 1 <= max_step <= len(PIPELINE_STEPS):
        raise ValueError("max_step must be between 1 and 11")
    if isinstance(video_ids, str):
        video_ids = [video_ids]
    elif video_ids is not None:
        video_ids = list(video_ids)

    root = (
        Path(output_root)
        if output_root is not None
        else Path(
            os.environ.get(
                "CAUVID_AUGUST_OUTPUT_PATH",
                config.get_output_path("pipeline_output") / "exp_august",
            )
        )
    )
    root = root.expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    tracker = tracker or _NullTracker()

    with _august_output_environment(root):
        state = _tracked(tracker, "01_dataset_initialization", lambda: modules.dataset_initialization(video_ids, video_count))
        if max_step == 1:
            return state
        state = _tracked(tracker, "02_object_detection", lambda: modules.object_detection(state))
        if max_step == 2:
            return state
        state = _tracked(tracker, "03_object_tracking", lambda: modules.object_tracking(state))
        if max_step == 3:
            return state
        state = _tracked(tracker, "04_3d_trajectory_construction", lambda: modules.trajectory_construction_3d(state))
        if max_step == 4:
            return state
        state = _tracked(
            tracker,
            "05_ego_motion_abstraction",
            lambda: modules.ego_motion_abstraction(
                state,
                render_candidate_filter_comparisons=render_candidate_filter_comparisons,
            ),
        )
        if max_step == 5:
            return state
        state = _tracked(tracker, "06_trajectory_refinement", lambda: modules.trajectory_refinement(state, diagnostics=diagnostics))
        if max_step == 6:
            return state
        state = _tracked(tracker, "07_relative_motion_representation", lambda: modules.relative_motion_representation(state))
        if max_step == 7:
            return state
        state = _tracked(tracker, "08_temporal_video_segmentation", lambda: modules.temporal_video_segmentation(state, root))
        if max_step == 8:
            return state
        state = _tracked(tracker, "09_segment_motion_abstraction", lambda: modules.segment_motion_abstraction(state, root))
        if max_step == 9:
            return state
        state = _tracked(tracker, "10_important_object_selection", lambda: modules.important_object_selection(state, root))
        if max_step == 10:
            return state
        return _tracked(tracker, "11_symbolic_scene_representation", lambda: modules.symbolic_scene_representation(state, root))


def main(**kwargs):
    """Programmatic entry point kept deliberately free of reasoning options."""
    tracker = kwargs.pop("tracker", None) or _NullTracker()
    try:
        result = run_pipeline(tracker=tracker, **kwargs)
    except BaseException as exc:
        tracker.finish(status="failed", error=exc)
        raise
    tracker.finish(status="completed")
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the exp_august Paper-1 pipeline")
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--video-count", type=int, default=None)
    parser.add_argument("--max-step", type=int, choices=range(1, 12), default=11)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--diagnostics", action="store_true", help="Run optional July visualization/dashboard/provenance audits")
    parser.add_argument("--render-candidate-filter-comparisons", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = main(
        video_ids=args.video_ids,
        video_count=args.video_count,
        max_step=args.max_step,
        output_root=args.output_root,
        diagnostics=args.diagnostics,
        render_candidate_filter_comparisons=args.render_candidate_filter_comparisons,
    )
    print(f"done videos={len(result['videos'])} final_step={min(args.max_step, 11)}")
