"""
Analyze and select important objects for each temporal segment.

Current status:
  - Structural placeholder only.
  - The actual importance/filtering strategy is intentionally not implemented yet.
  - For now, all objects are passed through as selected objects so downstream
    steps can be wired against the final schema.

Consumes:
  - Step 9 output: segment-level object motion summaries

Output layout:
    pipeline_output/10_driving_mini_important_objects/
        important_objects_manifest.json
        <video_id>/
            important_objects.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_IMPORTANT_OBJECTS_VERSION = 2


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "10_driving_mini_important_objects"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "selection_strategy",
        "passthrough_selected_objects",
    ]
    return {k: cfg.get(k) for k in keys}


def _print_video_summary(result: Dict[str, Any]) -> None:
    print(
        f"  {result.get('video_id', 'unknown')}: "
        f"objects={int(result.get('num_objects', 0))} | "
        f"candidate_objects={int(result.get('num_candidate_objects', 0))} | "
        f"selected_objects={int(result.get('num_selected_objects', 0))} | "
        f"selected_candidate_objects={int(result.get('num_selected_candidate_objects', 0))} | "
        f"strategy_applied={bool(result.get('selection_strategy_applied', False))}"
    )


def process_video(
    segment_object_motion_video_result: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    selection_strategy = str(cfg.get("selection_strategy", "not_implemented"))
    passthrough_selected_objects = bool(cfg.get("passthrough_selected_objects", True))

    video_id = str(segment_object_motion_video_result["video_id"])
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "important_objects.json"

    if not force_recompute and out_file.exists():
        with out_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        cache_cfg = cached.get("config", {}) if isinstance(cached, dict) else {}
        if (
            int(cached.get("version", 0)) == _IMPORTANT_OBJECTS_VERSION
            and _cfg_key_subset(cache_cfg) == _cfg_key_subset(
                {
                    "selection_strategy": selection_strategy,
                    "passthrough_selected_objects": passthrough_selected_objects,
                }
            )
        ):
            print(f"  [cache] {video_id} - loading {out_file.name}")
            _print_video_summary(cached)
            return cached

    segments_in = list(segment_object_motion_video_result.get("segments", []))
    segments_out: List[Dict[str, Any]] = []
    num_objects = 0
    num_candidate_objects = 0
    num_selected_objects = 0
    num_selected_candidate_objects = 0

    for segment in segments_in:
        objects = list(segment.get("objects", []))
        candidate_objects = list(segment.get("candidate_objects", []))
        selected_objects = list(objects) if passthrough_selected_objects else []
        selected_candidate_objects = list(candidate_objects) if passthrough_selected_objects else []
        filtered_objects: List[Dict[str, Any]] = []
        filtered_candidate_objects: List[Dict[str, Any]] = []

        num_objects += len(objects)
        num_candidate_objects += len(candidate_objects)
        num_selected_objects += len(selected_objects)
        num_selected_candidate_objects += len(selected_candidate_objects)

        segment_out = dict(segment)
        segment_out["objects"] = objects
        segment_out["candidate_objects"] = candidate_objects
        segment_out["selected_objects"] = selected_objects
        segment_out["selected_candidate_objects"] = selected_candidate_objects
        segment_out["filtered_objects"] = filtered_objects
        segment_out["filtered_candidate_objects"] = filtered_candidate_objects
        segment_out["selection_strategy"] = selection_strategy
        segment_out["selection_strategy_applied"] = False
        segment_out["num_objects"] = len(objects)
        segment_out["num_candidate_objects"] = len(candidate_objects)
        segment_out["num_selected_objects"] = len(selected_objects)
        segment_out["num_selected_candidate_objects"] = len(selected_candidate_objects)
        segments_out.append(segment_out)

    result: Dict[str, Any] = {
        "version": _IMPORTANT_OBJECTS_VERSION,
        "video_id": video_id,
        "selection_strategy_applied": False,
        "num_segments": len(segments_out),
        "num_objects": num_objects,
        "num_candidate_objects": num_candidate_objects,
        "num_selected_objects": num_selected_objects,
        "num_selected_candidate_objects": num_selected_candidate_objects,
        "config": {
            "selection_strategy": selection_strategy,
            "passthrough_selected_objects": passthrough_selected_objects,
        },
        "segments": segments_out,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    _print_video_summary(result)
    return result


def run(
    segment_object_motion_results: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    out_root = output_root or get_output_root()
    results: List[Dict[str, Any]] = []

    for segment_object_result in segment_object_motion_results:
        result = process_video(
            segment_object_motion_video_result=segment_object_result,
            cfg=cfg,
            output_root=out_root,
            force_recompute=force_recompute,
        )
        results.append(result)

    manifest = {
        "version": _IMPORTANT_OBJECTS_VERSION,
        "num_videos": len(results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_segments": r.get("num_segments", 0),
                "num_objects": r.get("num_objects", 0),
                "num_candidate_objects": r.get("num_candidate_objects", 0),
                "num_selected_objects": r.get("num_selected_objects", 0),
                "num_selected_candidate_objects": r.get("num_selected_candidate_objects", 0),
                "selection_strategy_applied": r.get("selection_strategy_applied", False),
            }
            for r in results
        ],
    }
    manifest_path = out_root / "important_objects_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Important objects manifest written to {manifest_path}")
    return results
