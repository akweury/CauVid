"""Bidirectional RAFT evidence backend with consistency diagnostics."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from src.exp_august.contracts import ToolVersion
from src.exp_august.inference.frames import CanonicalFrame


@dataclass(frozen=True)
class DirectionalFlowOutput:
    flow: np.ndarray
    domain_valid: np.ndarray
    consistency_valid: np.ndarray
    fb_error: np.ndarray


@dataclass(frozen=True)
class FlowPairOutput:
    forward: DirectionalFlowOutput
    backward: DirectionalFlowOutput


class DisabledFlowBackend:
    backend_name = "disabled"
    model_name = "none"
    model_id = "none"
    available = False
    unavailable_reason = "flow backend explicitly disabled"
    consistency_threshold_px = 1.5
    tool_versions: tuple[ToolVersion, ...] = ()

    def warmup(self) -> None:
        return None

    def predict_pair(
        self, earlier: CanonicalFrame, later: CanonicalFrame
    ) -> FlowPairOutput:
        raise RuntimeError("disabled flow backend cannot predict")

    def teardown(self) -> None:
        return None


def _consistency(
    flow: np.ndarray,
    reverse_flow: np.ndarray,
    threshold_px: float,
) -> DirectionalFlowOutput:
    flow = np.asarray(flow, dtype=np.float32)
    reverse_flow = np.asarray(reverse_flow, dtype=np.float32)
    if flow.shape != reverse_flow.shape or flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("forward/backward flow must share HxWx2 shape")
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    domain_valid = (
        np.isfinite(flow).all(axis=2)
        & (map_x >= 0.0)
        & (map_x <= width - 1)
        & (map_y >= 0.0)
        & (map_y <= height - 1)
    )
    sampled_reverse = cv2.remap(
        reverse_flow,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    error = np.linalg.norm(flow + sampled_reverse, axis=2).astype(np.float32)
    domain_valid &= np.isfinite(sampled_reverse).all(axis=2) & np.isfinite(error)
    error[~domain_valid] = np.nan
    consistency_valid = domain_valid & (error <= float(threshold_px))
    return DirectionalFlowOutput(
        flow=flow,
        domain_valid=domain_valid,
        consistency_valid=consistency_valid,
        fb_error=error,
    )


class RaftFlowEvidenceBackend:
    backend_name = "raft_small_bidirectional"
    available = True
    unavailable_reason = None

    def __init__(
        self,
        *,
        device: str = "auto",
        consistency_threshold_px: float = 1.5,
        allow_model_download: bool = False,
    ) -> None:
        import torch
        import torchvision
        from torchvision.models.optical_flow import Raft_Small_Weights

        self.device = (
            "cuda:0" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        self.consistency_threshold_px = float(consistency_threshold_px)
        if self.consistency_threshold_px <= 0.0:
            raise ValueError("flow consistency threshold must be positive")
        weights = Raft_Small_Weights.DEFAULT
        weight_name = Path(urlparse(weights.url).path).name
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weight_name
        if not checkpoint.is_file() and not allow_model_download:
            raise FileNotFoundError(
                f"RAFT weights are not cached at {checkpoint}; enable model download explicitly"
            )
        self.model_name = "torchvision/raft_small:Raft_Small_Weights.DEFAULT"
        self.model_id = f"raft_small@{weight_name}"
        self.tool_versions = (
            ToolVersion(name="torchvision", version=torchvision.__version__),
            ToolVersion(name="torch", version=torch.__version__),
        )

    def warmup(self) -> None:
        from src.exp_driving_videos.modules.pipe_utils.exp_driving_utils import (
            _get_raft_small_runtime,
        )

        _get_raft_small_runtime(device=self.device)

    def predict_pair(
        self, earlier: CanonicalFrame, later: CanonicalFrame
    ) -> FlowPairOutput:
        from src.exp_driving_videos.modules.pipe_utils.exp_driving_utils import (
            compute_optical_flow,
        )

        earlier_shape = earlier.image_bgr.shape
        later_shape = later.image_bgr.shape
        if earlier_shape != later_shape:
            raise ValueError(
                "RAFT frame pairs must have identical image shapes; "
                f"got {earlier_shape} and {later_shape}"
            )
        height, width = earlier_shape[:2]
        padded_height = height + (8 - height % 8) % 8
        padded_width = width + (8 - width % 8) % 8
        if padded_height < 128 or padded_width < 128:
            raise ValueError(
                "torchvision RAFT requires each padded image dimension to be at "
                f"least 128 pixels; got {height}x{width} before padding"
            )

        forward = compute_optical_flow(
            earlier.image_rgb,
            later.image_rgb,
            device=self.device,
        )
        backward = compute_optical_flow(
            later.image_rgb,
            earlier.image_rgb,
            device=self.device,
        )
        return FlowPairOutput(
            forward=_consistency(forward, backward, self.consistency_threshold_px),
            backward=_consistency(backward, forward, self.consistency_threshold_px),
        )

    def teardown(self) -> None:
        try:
            from src.exp_driving_videos.modules.pipe_utils.exp_driving_utils import (
                _RAFT_SMALL_RUNTIME,
            )

            _RAFT_SMALL_RUNTIME.pop(str(self.device), None)
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
