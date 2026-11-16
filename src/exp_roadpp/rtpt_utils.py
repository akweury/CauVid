"""RTPT process naming on the shared servers: the process title shows the owner's initials, the experiment
and the estimated remaining time (`rtpt.RTPT`). Optional: without the package the runs go on unnamed.

Usage:
    rtpt = rtpt_utils.start("roadpp-full_sn", max_iterations=12)
    ...
    rtpt_utils.step(rtpt, "step05")          # after each stage
"""
import os

INITIALS = os.environ.get("RTPT_INITIALS", "JI")     # the initials of exp_driving_videos' RTPT


def start(experiment_name, max_iterations):
    """A started RTPT, None if rtpt is not installed or fails (never stops a run)."""
    try:
        from rtpt.rtpt import RTPT
    except Exception:
        print("[RTPT] rtpt unavailable; the process keeps its default name.")
        return None
    try:
        rtpt = RTPT(name_initials=INITIALS, experiment_name=str(experiment_name),
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
        rtpt.step(subtitle=str(subtitle))
    except Exception as e:
        print(f"[RTPT] step failed ({type(e).__name__}: {e})")
