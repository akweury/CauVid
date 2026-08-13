"""Single-frame Depth Anything 3 evidence backend."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.exp_august.contracts import DepthRepresentation, ToolVersion
from src.exp_august.inference.frames import CanonicalFrame


@dataclass(frozen=True)
class DepthFrameOutput:
    depth: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray | None
    representation: DepthRepresentation


class DisabledDepthBackend:
    backend_name = "disabled"
    model_name = "none"
    model_id = "none"
    available = False
    unavailable_reason = "depth backend explicitly disabled"
    process_resolution = 504
    representation = DepthRepresentation.RELATIVE
    tool_versions: tuple[ToolVersion, ...] = ()

    def warmup(self) -> None:
        return None

    def predict_frame(self, frame: CanonicalFrame) -> DepthFrameOutput:
        raise RuntimeError("disabled depth backend cannot predict")

    def teardown(self) -> None:
        return None


class Da3DepthEvidenceBackend:
    """Run DA3 one frame at a time and retain relative depth plus confidence."""

    backend_name = "depth_anything_3_single_frame"
    available = True
    unavailable_reason = None
    representation = DepthRepresentation.RELATIVE

    def __init__(
        self,
        *,
        model_name: str = "depth-anything/DA3-Large",
        device: str = "auto",
        process_resolution: int = 504,
    ) -> None:
        import torch

        self.model_name = model_name
        self.device = (
            "cuda:0" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        self.process_resolution = int(process_resolution)
        if self.process_resolution <= 0:
            raise ValueError("DA3 process resolution must be positive")
        revision = "unresolved"
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(model_name, "config.json")
            if isinstance(cached, str):
                cached_path = Path(cached)
                if "snapshots" in cached_path.parts:
                    revision = cached_path.parts[cached_path.parts.index("snapshots") + 1]
        except Exception:
            pass
        self.model_id = f"{model_name}@{revision}"
        self.tool_versions = (ToolVersion(name="torch", version=torch.__version__),)
        self._model = None

    def warmup(self) -> None:
        import torch
        from src.external_depth_anything.depth_map_generator import (
            _get_cached_depth_model,
        )

        self._model, _ = _get_cached_depth_model(
            self.model_name,
            torch.device(self.device),
            use_fp16=False,
        )

    def predict_frame(self, frame: CanonicalFrame) -> DepthFrameOutput:
        if self._model is None:
            self.warmup()
        prediction = self._model.inference(
            [frame.image_rgb],
            process_res=self.process_resolution,
            process_res_method="upper_bound_resize",
        )
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        height, width = frame.image_bgr.shape[:2]
        if depth.shape != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        valid = np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid):
            raise RuntimeError(
                f"DA3 produced no valid depth at {frame.video_id} frame {frame.frame_index}"
            )
        depth = depth.astype(np.float32, copy=False)
        depth[~valid] = np.nan
        confidence = None
        if getattr(prediction, "conf", None) is not None:
            confidence = np.asarray(prediction.conf[0], dtype=np.float32)
            if confidence.shape != (height, width):
                confidence = cv2.resize(
                    confidence,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
            confidence = confidence.astype(np.float32, copy=False)
            confidence[~np.isfinite(confidence)] = np.nan
        return DepthFrameOutput(
            depth=depth,
            valid=valid,
            confidence=confidence,
            representation=self.representation,
        )

    def teardown(self) -> None:
        self._model = None
        try:
            from src.external_depth_anything.depth_map_generator import (
                clear_depth_model_cache,
            )

            clear_depth_model_cache()
        except Exception:
            pass
        gc.collect()

