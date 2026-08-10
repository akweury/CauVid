"""Stable, persisted train/eval/test splits for August experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

SPLIT_RATIOS = {"train": 0.70, "eval": 0.15, "test": 0.15}
MANIFEST_FILENAME = "data_split_manifest.json"


def _ordered(video_ids: Iterable[str], seed: int, namespace: str) -> list[str]:
    return sorted(
        {str(value) for value in video_ids if str(value)},
        key=lambda value: (
            hashlib.sha256(f"{namespace}:{int(seed)}:{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )


def _split_counts(total: int) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    priority = {"train": 2, "eval": 1, "test": 0}
    ranked = sorted(raw, key=lambda name: (raw[name] - counts[name], priority[name]), reverse=True)
    for name in ranked[:remainder]:
        counts[name] += 1
    return counts


def discover_annotated_video_ids(annotations_root: Path | str) -> set[str]:
    annotated = set()
    for path in Path(annotations_root).rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        video = payload.get("video", {})
        video_path = video.get("path") if isinstance(video, dict) else None
        if video_path and payload.get("segments"):
            annotated.add(Path(str(video_path)).stem)
    return annotated


def create_split_manifest(
    video_ids: Iterable[str],
    video_count: int | None,
    seed: int,
    annotated_video_ids: Iterable[str] | None = None,
) -> dict:
    available = _ordered(video_ids, seed, "selection")
    if video_count is not None:
        if int(video_count) > len(available):
            raise ValueError(
                f"Requested {int(video_count)} videos, but only {len(available)} are available"
            )
    total = len(available) if video_count is None else int(video_count)
    requested_counts = _split_counts(total)
    annotated_pool = set(available) if annotated_video_ids is None else set(annotated_video_ids) & set(available)
    test_count = min(requested_counts["test"], len(annotated_pool))
    test_ids = _ordered(annotated_pool, seed, "annotated_test")[:test_count]
    remaining = _ordered(set(available) - set(test_ids), seed, "selection_non_test")[: total - test_count]
    eval_count = min(requested_counts["eval"], len(remaining))
    split_order = _ordered(remaining, seed, "train_eval_split")
    eval_ids = split_order[:eval_count]
    train_ids = split_order[eval_count:]
    assignments = {
        "train": sorted(train_ids),
        "eval": sorted(eval_ids),
        "test": sorted(test_ids),
    }
    counts = {name: len(values) for name, values in assignments.items()}
    return {
        "version": 2,
        "method": "seeded_scale_selection_with_annotated_test_only",
        "seed": int(seed),
        "requested_video_count": None if video_count is None else int(video_count),
        "ratios": dict(SPLIT_RATIOS),
        "num_videos": sum(counts.values()),
        "counts": counts,
        "requested_counts": requested_counts,
        "num_annotated_videos_available": len(annotated_pool),
        "test_annotation_policy": "all_test_video_ids_must_have_valid_segment_annotations",
        "train_video_ids": assignments["train"],
        "eval_video_ids": assignments["eval"],
        "test_video_ids": assignments["test"],
    }


def load_or_create_split_manifest(
    path: Path | str,
    video_ids: Iterable[str],
    video_count: int | None,
    seed: int,
    annotated_video_ids: Iterable[str] | None = None,
) -> dict:
    manifest_path = Path(path)
    available = {str(value) for value in video_ids if str(value)}
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) < 2:
            payload = create_split_manifest(available, video_count, seed, annotated_video_ids)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return payload
        if int(payload.get("seed", -1)) != int(seed):
            raise ValueError(f"Existing split manifest seed does not match {seed}: {manifest_path}")
        expected_count = None if video_count is None else int(video_count)
        if payload.get("requested_video_count") != expected_count:
            raise ValueError(f"Existing split manifest video count does not match {expected_count}: {manifest_path}")
        assigned = {
            str(value)
            for name in ("train", "eval", "test")
            for value in payload.get(f"{name}_video_ids", [])
        }
        missing = sorted(assigned - available)
        if missing:
            raise ValueError(f"Split manifest references {len(missing)} unavailable videos: {missing[:5]}")
        if annotated_video_ids is not None:
            unannotated_test = sorted(set(payload.get("test_video_ids", [])) - set(annotated_video_ids))
            if unannotated_test:
                raise ValueError(f"Split manifest contains unannotated test videos: {unannotated_test[:5]}")
        return payload

    payload = create_split_manifest(available, video_count, seed, annotated_video_ids)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def selected_video_ids(manifest: dict) -> list[str]:
    return [
        str(value)
        for name in ("train", "eval", "test")
        for value in manifest.get(f"{name}_video_ids", [])
    ]
