"""
Derive temporal target/head atoms for rule learning from step-10 logic atoms.

Current use:
  - Create brake-next targets for each current segment S_t using the ego forward
    state of the next segment S_t+1.

Output layout:
    pipeline_output/driving_mini_target_head_atoms/
        target_head_atoms_manifest.json
        <video_id>/
            target_head_atoms.json
            target_head_atoms.csv
"""

from __future__ import annotations

import csv
import json
import re
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


_TARGET_HEAD_ATOMS_VERSION = 1


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "driving_mini_target_head_atoms"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _sym(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "unknown"
    if text[0].isdigit():
        text = f"n_{text}"
    return text


def _make_atom(predicate: str, *args: Any) -> str:
    return f"{_sym(predicate)}({','.join(_sym(arg) for arg in args)})."


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "target_predicate",
        "negative_target_predicate",
        "positive_forward_states",
        "include_negative_examples",
    ]
    return {k: cfg.get(k) for k in keys}


def _print_video_summary(result: Dict[str, Any]) -> None:
    video_id = str(result.get("video_id", "unknown"))
    num_examples = int(result.get("num_examples", 0))
    num_positive = int(result.get("num_positive_examples", 0))
    num_negative = int(result.get("num_negative_examples", 0))
    print(
        f"  {video_id}: examples={num_examples} | "
        f"brake_next={num_positive} | not_brake_next={num_negative}"
    )


def process_video(
    logic_atoms_video_result: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    target_predicate = str(cfg.get("target_predicate", "brake_next"))
    negative_target_predicate = str(cfg.get("negative_target_predicate", f"not_{target_predicate}"))
    positive_forward_states = [str(v) for v in cfg.get("positive_forward_states", ["forward_slowdown", "stopping"])]
    include_negative_examples = bool(cfg.get("include_negative_examples", True))

    video_id = str(logic_atoms_video_result["video_id"])
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "target_head_atoms.json"
    csv_file = out_dir / "target_head_atoms.csv"

    if not force_recompute and out_file.exists():
        with out_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        cache_cfg = cached.get("config", {}) if isinstance(cached, dict) else {}
        if (
            int(cached.get("version", 0)) == _TARGET_HEAD_ATOMS_VERSION
            and _cfg_key_subset(cache_cfg) == _cfg_key_subset(
                {
                    "target_predicate": target_predicate,
                    "negative_target_predicate": negative_target_predicate,
                    "positive_forward_states": positive_forward_states,
                    "include_negative_examples": include_negative_examples,
                }
            )
        ):
            print(f"  [cache] {video_id} - loading {out_file.name}")
            _print_video_summary(cached)
            return cached

    segments = list(logic_atoms_video_result.get("segments", []))
    examples: List[Dict[str, Any]] = []
    positive_count = 0
    negative_count = 0

    for idx in range(len(segments) - 1):
        current_seg = segments[idx]
        next_seg = segments[idx + 1]

        current_segment_id = str(current_seg.get("segment_id", "unknown"))
        next_segment_id = str(next_seg.get("segment_id", "unknown"))
        next_forward_label = str(next_seg.get("segment_forward_label", "unknown"))
        is_positive = next_forward_label in set(positive_forward_states)

        head_predicate = target_predicate if is_positive else negative_target_predicate
        if not is_positive and not include_negative_examples:
            continue

        head_atom = _make_atom(head_predicate, current_segment_id)
        body_atoms = list(current_seg.get("atoms", []))
        for obj in current_seg.get("objects", []):
            body_atoms.extend(obj.get("atoms", []))

        example = {
            "example_index": idx,
            "current_segment_id": current_segment_id,
            "next_segment_id": next_segment_id,
            "current_segment_index": int(current_seg.get("segment_index", idx)),
            "next_segment_index": int(next_seg.get("segment_index", idx + 1)),
            "current_segment_label": str(current_seg.get("segment_label", "unknown")),
            "next_forward_label": next_forward_label,
            "target_predicate": _sym(head_predicate),
            "label": bool(is_positive),
            "head_atom": head_atom,
            "body_atoms": body_atoms,
            "num_body_atoms": len(body_atoms),
        }
        examples.append(example)
        if is_positive:
            positive_count += 1
        else:
            negative_count += 1

    result: Dict[str, Any] = {
        "version": _TARGET_HEAD_ATOMS_VERSION,
        "video_id": video_id,
        "num_examples": len(examples),
        "num_positive_examples": positive_count,
        "num_negative_examples": negative_count,
        "config": {
            "target_predicate": target_predicate,
            "negative_target_predicate": negative_target_predicate,
            "positive_forward_states": positive_forward_states,
            "include_negative_examples": include_negative_examples,
        },
        "examples": examples,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    with csv_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "example_index",
                "current_segment_id",
                "next_segment_id",
                "current_segment_index",
                "next_segment_index",
                "current_segment_label",
                "next_forward_label",
                "target_predicate",
                "label",
                "head_atom",
                "num_body_atoms",
                "body_atoms",
            ],
        )
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "example_index": example.get("example_index", ""),
                    "current_segment_id": example.get("current_segment_id", ""),
                    "next_segment_id": example.get("next_segment_id", ""),
                    "current_segment_index": example.get("current_segment_index", ""),
                    "next_segment_index": example.get("next_segment_index", ""),
                    "current_segment_label": example.get("current_segment_label", ""),
                    "next_forward_label": example.get("next_forward_label", ""),
                    "target_predicate": example.get("target_predicate", ""),
                    "label": example.get("label", ""),
                    "head_atom": example.get("head_atom", ""),
                    "num_body_atoms": example.get("num_body_atoms", 0),
                    "body_atoms": json.dumps(example.get("body_atoms", [])),
                }
            )

    _print_video_summary(result)
    return result


def run(
    logic_atom_results: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    out_root = output_root or get_output_root()
    results: List[Dict[str, Any]] = []

    for logic_result in logic_atom_results:
        result = process_video(
            logic_atoms_video_result=logic_result,
            cfg=cfg,
            output_root=out_root,
            force_recompute=force_recompute,
        )
        results.append(result)

    manifest = {
        "num_videos": len(results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_examples": r.get("num_examples", 0),
                "num_positive_examples": r.get("num_positive_examples", 0),
                "num_negative_examples": r.get("num_negative_examples", 0),
            }
            for r in results
        ],
    }
    manifest_path = out_root / "target_head_atoms_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Target head atoms manifest written to {manifest_path}")
    return results
