"""
Prepare per-object 3D positions for driving_mini videos.

Consumes the Step 4 merged annotations and enriches each frame/object with:
- depth_map_path
- positions_3d: [x, y, z] camera-frame centers inferred from Depth Anything

Output layout:
    pipeline_output/05_driving_mini_3d_positions/
        positions_3d_manifest.json
        <video_id>/
            positions_3d.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from config import PROJECT_ROOT
from src.exp_driving_videos.modules import data_preprocessing


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "05_driving_mini_3d_positions"
    out.mkdir(parents=True, exist_ok=True)
    return out


def process_video(
    video_result: Dict[str, Any],
    output_root: Optional[Path] = None,
    model_name: str = "depth-anything/DA3-Large",
    batch_size: int = 4,
    device: str = "auto",
    force_recompute: bool = False,
    force_recompute_depth: bool = False,
) -> Dict[str, Any]:
    video_id = video_result["video_id"]
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "positions_3d.json"

    if not force_recompute and out_file.exists():
        print(f"  [cache] {video_id} - loading {out_file.name}")
        with out_file.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    enriched = data_preprocessing.prepare_video_3d_positions(
        video_result=video_result,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        force_recompute_depth=force_recompute_depth,
    )

    num_objects = sum(len(frame.get("positions_3d", [])) for frame in enriched.get("frames", []))
    result: Dict[str, Any] = {
        **enriched,
        "num_frames": len(enriched.get("frames", [])),
        "num_objects_with_3d": num_objects,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(
        f"  {video_id}: {result['num_frames']} frames, "
        f"{result['num_objects_with_3d']} objects with 3D positions"
    )
    return result


def run(
    merged_results: List[Dict[str, Any]],
    output_root: Optional[Path] = None,
    model_name: str = "depth-anything/DA3-Large",
    batch_size: int = 4,
    device: str = "auto",
    force_recompute: bool = False,
    force_recompute_depth: bool = False,
) -> List[Dict[str, Any]]:
    effective_output_root = output_root or get_output_root()

    print(f"Videos to prepare 3D positions for: {[r['video_id'] for r in merged_results]}")

    results: List[Dict[str, Any]] = []
    for video_result in merged_results:
        result = process_video(
            video_result=video_result,
            output_root=effective_output_root,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            force_recompute=force_recompute,
            force_recompute_depth=force_recompute_depth,
        )
        results.append(result)

    manifest = {
        "model_name": model_name,
        "num_videos": len(results),
        "num_frames_total": sum(r.get("num_frames", 0) for r in results),
        "num_objects_with_3d_total": sum(r.get("num_objects_with_3d", 0) for r in results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_frames": r.get("num_frames", 0),
                "num_objects_with_3d": r.get("num_objects_with_3d", 0),
                "depth_dir": r.get("depth_dir", ""),
            }
            for r in results
        ],
    }
    manifest_path = effective_output_root / "positions_3d_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Saved 3D position manifest to {manifest_path}")
    return results
