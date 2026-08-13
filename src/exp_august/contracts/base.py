"""Shared primitives for versioned ``exp_august`` boundary contracts.

Contracts are intentionally stricter than the numerical data structures used
inside an algorithm.  They describe persisted stage boundaries, reject unknown
fields, and remain immutable after validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


ContractId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class ContractModel(BaseModel):
    """Base class for immutable, strict and forward-incompatible contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class CoordinateSpace(str, Enum):
    IMAGE_PIXELS = "image_pixels"
    CAMERA_3D = "camera_3d"
    EGO_3D = "ego_3d"
    COMPONENT_LOCAL_3D = "component_local_3d"
    WORLD_3D = "world_3d"


class DecodeStatus(str, Enum):
    NOT_CHECKED = "not_checked"
    DECODED = "decoded"


class TimelineSource(str, Enum):
    CONTAINER_PTS = "container_pts"
    ASSUMED_CONSTANT_FRAME_RATE = "assumed_constant_frame_rate"


class DecodeValidationMode(str, Enum):
    NONE = "none"
    SAMPLE = "sample"
    FULL = "full"


class EvidenceRole(str, Enum):
    FIT = "fit"
    CHECK_ONLY = "check_only"
    REPORT_ONLY = "report_only"


class CueFamily(str, Enum):
    OBJECTS = "objects"
    MASKS = "masks"
    FLOW_FORWARD = "flow_forward"
    FLOW_BACKWARD = "flow_backward"
    DEPTH = "depth"


class CueStatus(str, Enum):
    AVAILABLE = "available"
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class DetectionTier(str, Enum):
    PRIMARY = "primary"
    CANDIDATE = "candidate"


class FlowDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


class DepthRepresentation(str, Enum):
    RELATIVE = "relative"
    METRIC = "metric"


class DepthUnit(str, Enum):
    RELATIVE_UNIT = "relative_unit"
    METER = "meter"


class ArtifactOwner(str, Enum):
    STEP1_INIT = "step1_init"
    STEP2_NEURAL_EVIDENCE = "step2_neural_evidence"
    STEP3_OBJECT_TRACKING = "step3_object_tracking"
    STEP4_GEOMETRY_SCALE = "step4_geometry_scale"
    STEP5_WORLD_RECONSTRUCTION = "step5_world_reconstruction"


class MotionState(str, Enum):
    STATIC = "static"
    MOVING = "moving"
    AMBIGUOUS = "ambiguous"
    UNOBSERVABLE = "unobservable"


class TrackState(str, Enum):
    ACTIVE = "active"
    LOST = "lost"
    RETIRED = "retired"


class TrackMarkerType(str, Enum):
    FIRST_OBSERVED = "first_observed"
    MATCHED = "matched"
    MISSED = "missed"
    REOBSERVED = "reobserved"
    RETIRED = "retired"
    VIDEO_END = "video_end"


class AssociationDecision(str, Enum):
    MATCHED = "matched"
    REJECTED_GATE = "rejected_gate"
    REJECTED_THRESHOLD = "rejected_threshold"
    REJECTED_CONFLICT = "rejected_conflict"


class EvidenceDisposition(str, Enum):
    SELECTED = "selected"
    UNSELECTED = "unselected"
    INVALID = "invalid"


class MaskCandidateSource(str, Enum):
    DIRECT_INSTANCE = "direct_instance"
    FLOW_FORWARD = "flow_forward"
    FLOW_BACKWARD = "flow_backward"
    UNASSIGNED_INSTANCE = "unassigned_instance"
    EMPTY_OR_OUTSIDE = "empty_or_outside"
    EXPLICIT_UNOBSERVABLE = "explicit_unobservable"


class SupportObservability(str, Enum):
    OBSERVED = "observed"
    LATENT_SUPPORT = "latent_support"
    UNOBSERVABLE = "unobservable"


class GapStatus(str, Enum):
    REOBSERVED = "reobserved"
    RETIRED = "retired"
    VIDEO_END = "video_end"


class Observability(str, Enum):
    METRIC = "metric"
    RELATIVE = "relative"
    AMBIGUOUS = "ambiguous"
    UNOBSERVABLE = "unobservable"


class ImageSize(ContractModel):
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]


class ToolVersion(ContractModel):
    name: ContractId
    version: ContractId
