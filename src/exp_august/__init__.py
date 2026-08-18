"""August experiments.

The current pipeline is :mod:`src.exp_august.inference.runner`.  The package-
level ``main`` and ``run_pipeline`` names below are legacy compatibility
exports for the archived 11-step linear baseline; new code must not use them.
"""

__all__ = ["main", "run_pipeline"]


def run_pipeline(*args, **kwargs):
    """Compatibility wrapper for the archived legacy linear baseline."""
    from src.exp_august.pipeline import run_pipeline as run

    return run(*args, **kwargs)


def main(*args, **kwargs):
    """Compatibility wrapper for the archived legacy linear baseline."""
    from src.exp_august.pipeline import main as run

    return run(*args, **kwargs)
