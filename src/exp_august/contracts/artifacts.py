"""Stable references to dense or persisted pipeline artifacts."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, field_validator

from .base import ContractId, ContractModel, CoordinateSpace, Sha256


class ArtifactRef(ContractModel):
    """Content-addressed reference relative to the owning stage directory."""

    schema_name: Literal["artifact_ref"] = "artifact_ref"
    schema_version: Literal[1] = 1

    artifact_id: ContractId
    relative_path: str
    sha256: Sha256
    byte_size: Annotated[int, Field(ge=0)]
    media_type: ContractId
    shape: tuple[Annotated[int, Field(gt=0)], ...] | None = None
    dtype: str | None = None
    coordinate_space: CoordinateSpace | None = None

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if not value or "\\" in value:
            raise ValueError("artifact paths must be non-empty POSIX relative paths")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact paths must stay inside their owning stage directory")
        if any(part in {"", "."} for part in path.parts):
            raise ValueError("artifact paths must be normalized")
        return path.as_posix()

