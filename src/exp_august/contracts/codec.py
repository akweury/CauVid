"""Canonical JSON persistence for typed contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

from .base import ContractModel


ContractT = TypeVar("ContractT", bound=ContractModel)


def canonical_json_bytes(value: ContractModel | Any, *, trailing_newline: bool = True) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, ContractModel) else value
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if trailing_newline:
        text += "\n"
    return text.encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_payload(value: ContractModel | Any) -> str:
    return sha256_bytes(canonical_json_bytes(value, trailing_newline=False))


def write_contract(path: Path, value: ContractModel) -> tuple[str, int]:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return sha256_bytes(payload), len(payload)


def read_contract(path: Path, model_type: type[ContractT]) -> ContractT:
    return model_type.model_validate_json(path.read_bytes())

