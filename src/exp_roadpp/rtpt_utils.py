"""RTPT process naming on the shared servers: the process title shows the owner's initials, the experiment
and the estimated remaining time (`rtpt.RTPT`). Optional: without the package the runs go on unnamed.

Usage:
    rtpt = rtpt_utils.start("roadpp-full_sn", max_iterations=12)
    ...
    rtpt_utils.step(rtpt, "step05")          # after each stage
"""
import os
import re

INITIALS = os.environ.get("RTPT_INITIALS", "JI")     # the initials of exp_driving_videos' RTPT
MAX_NAME = 22               # characters of the experiment name in the title
MAX_SUBTITLE = 10           # characters of the stage
# (pattern, replacement) shortening the names; the ETA ("#h:mm:ss") is appended by rtpt and kept
_SHORT = [(r"^roadpp-", ""), (r"sumo_maps_", "sm"), (r"sumo_junction_", "sj"), (r"_unseen", "_u"),
          (r"-[a-z0-9-]+?_([0-9a-f]{6})$", r"-\1"),          # "-dgx-jing_975fa2" -> "-975fa2" (the run token)
          (r"timing_diag", "tdiag"), (r"flow_diag", "fdiag"), (r"box_stats", "boxes")]


def short_name(name, limit=MAX_NAME):
    """The process title's experiment name, e.g. "roadpp-sumo_maps_v7-dgx-jing_975fa2" -> "smv7-975fa2"."""
    for pattern, repl in _SHORT:
        name = re.sub(pattern, repl, name)
    return name[:limit]


def short_stage(subtitle, limit=MAX_SUBTITLE):
    """The stage in the title, e.g. "step05_causal" -> "s5causal", "dataset" -> "data"."""
    subtitle = re.sub(r"step0*(\d+)_?", r"s\1", str(subtitle)).replace("dataset", "data")
    return subtitle.replace(" ", "")[:limit]


def start(experiment_name, max_iterations):
    """A started RTPT, None if rtpt is not installed or fails (never stops a run)."""
    try:
        from rtpt.rtpt import RTPT
    except Exception:
        print("[RTPT] rtpt unavailable; the process keeps its default name.")
        return None
    try:
        rtpt = RTPT(name_initials=INITIALS, experiment_name=short_name(str(experiment_name)),
                    max_iterations=max(1, int(max_iterations)))
        rtpt.start()
        return rtpt
    except Exception as e:
        print(f"[RTPT] not started ({type(e).__name__}: {e})")
        return None


def step(rtpt, subtitle=""):
    """One iteration done; subtitle names the next stage in the process title."""
    if rtpt is None:
        return
    try:
        rtpt.step(subtitle=short_stage(subtitle) or None)
    except Exception as e:
        print(f"[RTPT] step failed ({type(e).__name__}: {e})")
