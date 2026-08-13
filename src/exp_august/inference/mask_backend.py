"""Frame-local SAM 2 evidence backend for target Step 2."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.exp_august.contracts import DetectionObservation, DetectionTier, ToolVersion
from src.exp_august.contracts.codec import sha256_file
from src.exp_august.inference.frames import CanonicalFrame


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class MaskCandidateOutput:
    prompt_detection_id: str
    mask: np.ndarray
    confidence: float


class DisabledMaskBackend:
    backend_name = "disabled"
    model_name = "none"
    model_id = "none"
    available = False
    unavailable_reason = "mask backend explicitly disabled"
    prompt_candidates = False
    tool_versions: tuple[ToolVersion, ...] = ()

    def warmup(self) -> None:
        return None

    def predict_frame(
        self,
        frame: CanonicalFrame,
        detections: Sequence[DetectionObservation],
    ) -> tuple[MaskCandidateOutput, ...]:
        return ()

    def teardown(self) -> None:
        return None


class Sam2MaskEvidenceBackend:
    """Generate independent masks from frame-local YOLO box prompts."""

    backend_name = "sam2_frame_local"
    available = True
    unavailable_reason = None

    def __init__(
        self,
        *,
        model_name: str = "weights/sam2/sam2_t.pt",
        device: str = "auto",
        prompt_candidates: bool = False,
        allow_model_download: bool = False,
    ) -> None:
        import torch
        import ultralytics

        model_path = Path(model_name).expanduser()
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if model_path.is_file():
            self.model_source = str(model_path.resolve())
            self.model_name = str(model_name)
            self.model_id = f"{model_path.name}@sha256:{sha256_file(model_path)[:16]}"
        elif allow_model_download:
            self.model_source = model_name
            self.model_name = model_name
            self.model_id = f"{model_name}@download-resolved-at-runtime"
        else:
            raise FileNotFoundError(
                f"SAM 2 model is not local: {model_path}; enable model download explicitly"
            )
        self.device = (
            "cuda:0" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        self.prompt_candidates = bool(prompt_candidates)
        self.tool_versions = (
            ToolVersion(name="ultralytics", version=ultralytics.__version__),
            ToolVersion(name="torch", version=torch.__version__),
        )
        self._model = None

    def warmup(self) -> None:
        from ultralytics import SAM

        self._model = SAM(self.model_source)

    def predict_frame(
        self,
        frame: CanonicalFrame,
        detections: Sequence[DetectionObservation],
    ) -> tuple[MaskCandidateOutput, ...]:
        prompts = tuple(
            detection
            for detection in detections
            if self.prompt_candidates or detection.tier == DetectionTier.PRIMARY
        )
        if not prompts:
            return ()
        if self._model is None:
            self.warmup()
        boxes = [
            [
                prompt.bbox.x1,
                prompt.bbox.y1,
                prompt.bbox.x2,
                prompt.bbox.y2,
            ]
            for prompt in prompts
        ]
        results = self._model.predict(
            source=frame.image_bgr,
            bboxes=boxes,
            device=self.device,
            retina_masks=True,
            conf=0.0,
            verbose=False,
            stream=False,
        )
        if len(results) != 1 or results[0].masks is None:
            raise RuntimeError(
                f"SAM 2 returned no mask container for {frame.video_id} frame {frame.frame_index}"
            )
        result = results[0]
        masks = result.masks.data.detach().cpu().numpy()
        if len(masks) != len(prompts):
            raise RuntimeError(
                f"SAM 2 returned {len(masks)} masks for {len(prompts)} prompts "
                f"at {frame.video_id} frame {frame.frame_index}"
            )
        scores = (
            result.boxes.conf.detach().cpu().tolist()
            if result.boxes is not None and result.boxes.conf is not None
            else [1.0] * len(masks)
        )
        height, width = frame.image_bgr.shape[:2]
        outputs: list[MaskCandidateOutput] = []
        for prompt, mask, score in zip(prompts, masks, scores):
            binary = np.asarray(mask, dtype=bool)
            if binary.shape != (height, width):
                binary = cv2.resize(
                    binary.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if not np.any(binary):
                raise RuntimeError(
                    f"SAM 2 returned an empty mask for prompt {prompt.detection_id}"
                )
            outputs.append(
                MaskCandidateOutput(
                    prompt_detection_id=prompt.detection_id,
                    mask=binary,
                    confidence=max(0.0, min(1.0, float(score))),
                )
            )
        return tuple(outputs)

    def teardown(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

