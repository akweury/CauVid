"""Independent runner for the Paper-1 ``exp_august`` pipeline."""

from __future__ import annotations

import argparse
import os
import platform
import re
import sys
import time
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

STEP_FUNCTION_NAMES = (
    "Initialization",
    "Object Detection",
    "Mask Tracking",
    "Trajectories",
    "Egomotion",
    "Refinement",
    "Relativity",
    "Segmentation",
    "Abstraction",
    "Selection",
    "Symbolization",
)


class _NullTracker:
    def log_state(self, _name, state, **_kwargs):
        return state

    def log_failure(self, _name, _error, **_kwargs):
        return None

    def finish(self, **_kwargs):
        return None


def _wandb_tracker(*, video_ids, video_count, seed, max_step):
    enabled = os.environ.get("CAUVID_WANDB_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return _NullTracker()
    from src.exp_july.wandb_tracking import WandbTracker

    return WandbTracker(
        enabled=True,
        project=os.environ.get("CAUVID_WANDB_PROJECT", "cauvid-exp-august"),
        run_name=os.environ.get("CAUVID_WANDB_RUN_NAME"),
        mode=os.environ.get("CAUVID_WANDB_MODE"),
        config={
            "pipeline": "exp_august",
            "video_ids": list(video_ids) if video_ids is not None else None,
            "video_count": video_count,
            "seed": int(seed),
            "max_step": int(max_step),
            "data_scale": modules.data_scale_name(video_count),
        },
    )


class _SelectedTqdmStream:
    """Forward selected real nested tqdm bars with canonical public labels."""

    def __init__(self, stream, specifications):
        self.stream = stream
        self.specifications = list(specifications)
        self.used = set()
        self.active = False
        self.active_index = None
        self.active_label = None
        self.saw_progress = False

    def write(self, value):
        if not self.active:
            if not re.search(r"\d+%\|.*\|\s*\d+/\d+", value):
                return len(value)
            description = re.split(r"\s*\d+%\|", value.lstrip("\r"), maxsplit=1)[0].strip()
            selected = next(
                (
                    (index, label)
                    for index, (pattern, label) in enumerate(self.specifications)
                    if index not in self.used and re.search(pattern, description, flags=re.IGNORECASE)
                ),
                None,
            )
            if selected is None:
                return len(value)
            self.active_index, self.active_label = selected
            self.active = True
            self.saw_progress = True
        value = re.sub(
            r"(^|\r)[^\r\n]*?:\s*(?=\d+%\|)",
            lambda match: f"{match.group(1)}{self.active_label}: ",
            value,
        )
        written = self.stream.write(value)
        if "\n" in value:
            self.used.add(self.active_index)
            self.active = False
            self.active_index = None
            self.active_label = None
        return written

    def flush(self):
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _short_device_model(value: str, fallback: str) -> str:
    model = " ".join(str(value or "").replace("NVIDIA", "").replace("(R)", "").split()).strip()
    model = re.sub(r"\s+Family\s+\d+\s+Model\s+\d+\s+Stepping\s+\d+.*$", "", model, flags=re.IGNORECASE)
    model = model.replace("AuthenticAMD", "").replace("GenuineIntel", "").strip(" ,")
    return (model or fallback)[:48]


def _detection_device_summary(state: dict) -> str:
    requested = str((state.get("detection_args") or {}).get("device", "cpu")).lower()
    if requested != "cpu":
        try:
            import torch

            if torch.cuda.is_available():
                return f"Device: GPU | {_short_device_model(torch.cuda.get_device_name(0), 'CUDA')}"
        except (ImportError, RuntimeError):
            pass
    cpu_model = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "CPU")
    return f"Device: CPU | {_short_device_model(cpu_model, 'CPU')}"


def _tracked(tracker, name: str, operation: Callable, detail: str | None = None):
    step_number = int(name.split("_", 1)[0])
    step_label = f"Step {step_number} {STEP_FUNCTION_NAMES[step_number - 1]}"
    print(f"\n\n{'-' * 72}", flush=True)
    print(step_label, flush=True)
    if detail:
        print(detail, flush=True)
    started = time.perf_counter()
    progress_specs = (
        (
            (r"\[step 7\]\s*ego_motion", "Step 5a Ego Motion"),
            (r"\[step 7a\]\s*axis_threshold_segmentation", "Step 5b Axis Threshold Segmentation"),
            (r"\[step 7b\]\s*consensus_merge", "Step 5c Axis Consensus Segmentation"),
        )
        if step_number == 5
        else ((r".*", step_label),)
    )
    progress_stream = _SelectedTqdmStream(sys.stderr, progress_specs)
    if step_number == 2:
        # Ultralytics configures Python logging during its first prediction.
        # Keep stderr native for that stage; wrapping it can trigger logging's
        # recursive emergency handler on some server/runtime combinations.
        progress_stream.saw_progress = True
    try:
        with open(os.devnull, "w", encoding="utf-8") as quiet:
            stderr_context = nullcontext() if step_number == 2 else redirect_stderr(progress_stream)
            with redirect_stdout(quiet), stderr_context:
                state = operation()
        if not isinstance(state, dict):
            raise TypeError(f"{name} returned {type(state).__name__}; expected dict")
        if not state.get("videos"):
            raise RuntimeError(f"{name} produced no videos")
        if not progress_stream.saw_progress:
            video_count = len(state.get("videos", []))
            with tqdm(total=video_count, desc=step_label, unit="video", dynamic_ncols=True, leave=True) as progress:
                progress.update(video_count)
    except BaseException as exc:
        tracker.log_failure(name, exc, duration_seconds=time.perf_counter() - started)
        print(f"FAILED: {exc}", flush=True)
        raise
    duration = time.perf_counter() - started
    return tracker.log_state(name, state, duration_seconds=duration)


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
    seed: int = modules.DEFAULT_DATA_SELECTION_SEED,
    max_step: int = 11,
    output_root: Path | str | None = None,
    diagnostics: bool = False,
    render_candidate_filter_comparisons: bool = False,
    tracking_backend: str | None = None,
    sam2_model: Path | str | None = None,
    sam2_device: str | None = None,
    sam2_allow_download: bool | None = None,
    mask_tracking_strict: bool | None = None,
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
        else modules.get_august_output_root(video_count, seed)
    )
    root = root.expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    tracker = tracker or _NullTracker()

    with _august_output_environment(root):
        state = _tracked(
            tracker,
            "01_dataset_initialization",
            lambda: modules.dataset_initialization(video_ids, video_count, seed),
        )
        if max_step == 1:
            return state
        state = _tracked(
            tracker,
            "02_object_detection",
            lambda: modules.object_detection(state),
            detail=_detection_device_summary(state),
        )
        if max_step == 2:
            return state
        tracking_overrides = {
            key: value
            for key, value in {
                "backend": tracking_backend,
                "sam2_model": str(sam2_model) if sam2_model is not None else None,
                "device": sam2_device,
                "allow_model_download": sam2_allow_download,
                "strict": mask_tracking_strict,
            }.items()
            if value is not None
        }
        state = _tracked(
            tracker,
            "03_object_tracking",
            lambda: modules.object_tracking(state, tracking_overrides=tracking_overrides),
        )
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
    tracker = kwargs.pop("tracker", None) or _wandb_tracker(
        video_ids=kwargs.get("video_ids"),
        video_count=kwargs.get("video_count"),
        seed=kwargs.get("seed", modules.DEFAULT_DATA_SELECTION_SEED),
        max_step=kwargs.get("max_step", 11),
    )
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
    parser.add_argument("--seed", type=int, default=modules.DEFAULT_DATA_SELECTION_SEED)
    parser.add_argument("--max-step", type=int, choices=range(1, 12), default=11)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--diagnostics", action="store_true", help="Run optional July visualization/dashboard/provenance audits")
    parser.add_argument("--render-candidate-filter-comparisons", action="store_true")
    parser.add_argument(
        "--tracking-backend",
        choices=("auto", "hybrid_mask", "bytetrack"),
        default=None,
        help="Step 3 backend; auto uses hybrid mask tracking only when a SAM 2 checkpoint is available",
    )
    parser.add_argument("--sam2-model", type=Path, default=None, help="Local SAM 2 checkpoint")
    parser.add_argument("--sam2-device", default=None, help="SAM 2 inference device, for example cuda:0 or cpu")
    parser.add_argument(
        "--sam2-allow-download",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow Ultralytics to download a checkpoint named by --sam2-model",
    )
    parser.add_argument(
        "--mask-tracking-strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail instead of falling back to ByteTrack when hybrid mask tracking cannot run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = main(
        video_ids=args.video_ids,
        video_count=args.video_count,
        seed=args.seed,
        max_step=args.max_step,
        output_root=args.output_root,
        diagnostics=args.diagnostics,
        render_candidate_filter_comparisons=args.render_candidate_filter_comparisons,
        tracking_backend=args.tracking_backend,
        sam2_model=args.sam2_model,
        sam2_device=args.sam2_device,
        sam2_allow_download=args.sam2_allow_download,
        mask_tracking_strict=args.mask_tracking_strict,
    )
    print(f"done videos={len(result['videos'])} final_step={min(args.max_step, 11)}")
