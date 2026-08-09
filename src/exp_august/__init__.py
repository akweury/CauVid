"""Paper-1 video segmentation and symbolic representation experiment."""

__all__ = ["main", "run_pipeline"]


def run_pipeline(*args, **kwargs):
    from src.exp_august.pipeline import run_pipeline as run

    return run(*args, **kwargs)


def main(*args, **kwargs):
    from src.exp_august.pipeline import main as run

    return run(*args, **kwargs)
