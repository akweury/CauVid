"""
Build final temporal rule-learning examples from step-11 target/head atoms.

Consumes:
  - Step 11 output: head atoms + current-segment body atoms

Output layout:
    pipeline_output/13_driving_mini_temporal_rule_examples/
        temporal_rule_examples_manifest.json
        <video_id>/
            temporal_rule_examples.json
            temporal_rule_examples.csv
"""

from __future__ import annotations

import csv
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


_TEMPORAL_RULE_EXAMPLES_VERSION = 4


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "13_driving_mini_temporal_rule_examples"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "deduplicate_body_atoms",
        "sort_body_atoms",
        "include_clause_text",
        "include_negative_examples",
    ]
    return {k: cfg.get(k) for k in keys}


def _print_video_summary(result: Dict[str, Any]) -> None:
    video_id = str(result.get("video_id", "unknown"))
    num_examples = int(result.get("num_examples", 0))
    num_positive = int(result.get("num_positive_examples", 0))
    num_negative = int(result.get("num_negative_examples", 0))
    avg_body_atoms = float(result.get("avg_num_body_atoms", 0.0))
    candidate_examples = int(result.get("num_examples_with_candidate_atoms", 0))
    print(
        f"  {video_id}: examples={num_examples} | "
        f"positive={num_positive} | negative={num_negative} | "
        f"avg_body_atoms={avg_body_atoms:.1f} | "
        f"with_candidate_atoms={candidate_examples}"
    )


def _normalize_body_atoms(
    atoms: List[str],
    deduplicate: bool,
    sort_atoms: bool,
) -> List[str]:
    body_atoms = [str(atom) for atom in atoms]
    if deduplicate:
        seen = set()
        deduped: List[str] = []
        for atom in body_atoms:
            if atom in seen:
                continue
            seen.add(atom)
            deduped.append(atom)
        body_atoms = deduped
    if sort_atoms:
        body_atoms = sorted(body_atoms)
    return body_atoms


def _build_clause(head_atom: str, body_atoms: List[str]) -> str:
    head = str(head_atom).strip()
    if not body_atoms:
        return head
    if head.endswith("."):
        head = head[:-1]
    return f"{head} :- {', '.join(body_atoms)}."


def process_video(
    target_head_video_result: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    deduplicate_body_atoms = bool(cfg.get("deduplicate_body_atoms", True))
    sort_body_atoms = bool(cfg.get("sort_body_atoms", True))
    include_clause_text = bool(cfg.get("include_clause_text", True))
    include_negative_examples = bool(cfg.get("include_negative_examples", True))

    video_id = str(target_head_video_result["video_id"])
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "temporal_rule_examples.json"
    csv_file = out_dir / "temporal_rule_examples.csv"

    if not force_recompute and out_file.exists():
        with out_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        cache_cfg = cached.get("config", {}) if isinstance(cached, dict) else {}
        if (
            int(cached.get("version", 0)) == _TEMPORAL_RULE_EXAMPLES_VERSION
            and _cfg_key_subset(cache_cfg) == _cfg_key_subset(
                {
                    "deduplicate_body_atoms": deduplicate_body_atoms,
                    "sort_body_atoms": sort_body_atoms,
                    "include_clause_text": include_clause_text,
                    "include_negative_examples": include_negative_examples,
                }
            )
        ):
            print(f"  [cache] {video_id} - loading {out_file.name}")
            _print_video_summary(cached)
            return cached

    examples_in = list(target_head_video_result.get("examples", []))
    examples_out: List[Dict[str, Any]] = []
    positive_count = 0
    negative_count = 0
    total_body_atoms = 0
    total_candidate_body_atoms = 0
    examples_with_candidate_atoms = 0
    positive_examples_with_candidate_atoms = 0
    negative_examples_with_candidate_atoms = 0
    positive_candidate_atom_total = 0
    negative_candidate_atom_total = 0

    for example in examples_in:
        is_positive = bool(example.get("label", False))
        if not is_positive and not include_negative_examples:
            continue

        accepted_body_atoms = _normalize_body_atoms(
            atoms=list(example.get("accepted_body_atoms", [])),
            deduplicate=deduplicate_body_atoms,
            sort_atoms=sort_body_atoms,
        )
        candidate_body_atoms = _normalize_body_atoms(
            atoms=list(example.get("candidate_rule_search_body_atoms", example.get("candidate_body_atoms", []))),
            deduplicate=deduplicate_body_atoms,
            sort_atoms=sort_body_atoms,
        )
        candidate_quality_auxiliary_body_atoms = _normalize_body_atoms(
            atoms=list(example.get("candidate_quality_auxiliary_body_atoms", [])),
            deduplicate=deduplicate_body_atoms,
            sort_atoms=sort_body_atoms,
        )
        candidate_provenance_metadata_body_atoms = _normalize_body_atoms(
            atoms=list(example.get("candidate_provenance_metadata_body_atoms", [])),
            deduplicate=deduplicate_body_atoms,
            sort_atoms=sort_body_atoms,
        )
        body_atoms = _normalize_body_atoms(
            atoms=list(example.get("body_atoms", [])),
            deduplicate=deduplicate_body_atoms,
            sort_atoms=sort_body_atoms,
        )
        head_atom = str(example.get("head_atom", ""))
        rule_clause = _build_clause(head_atom, body_atoms) if include_clause_text else ""
        has_candidate_atoms = bool(candidate_body_atoms)

        out_example = {
            "example_index": int(example.get("example_index", len(examples_out))),
            "example_id": f"{video_id}_ex_{int(example.get('example_index', len(examples_out))):04d}",
            "current_segment_id": str(example.get("current_segment_id", "")),
            "next_segment_id": str(example.get("next_segment_id", "")),
            "current_segment_index": int(example.get("current_segment_index", -1)),
            "next_segment_index": int(example.get("next_segment_index", -1)),
            "current_segment_label": str(example.get("current_segment_label", "unknown")),
            "next_forward_label": str(example.get("next_forward_label", "unknown")),
            "target_predicate": str(example.get("target_predicate", "unknown")),
            "label": is_positive,
            "head_atom": head_atom,
            "accepted_body_atoms": accepted_body_atoms,
            "candidate_body_atoms": candidate_body_atoms,
            "candidate_rule_search_body_atoms": candidate_body_atoms,
            "candidate_quality_auxiliary_body_atoms": candidate_quality_auxiliary_body_atoms,
            "candidate_provenance_metadata_body_atoms": candidate_provenance_metadata_body_atoms,
            "body_atoms": body_atoms,
            "num_accepted_body_atoms": len(accepted_body_atoms),
            "num_candidate_body_atoms": len(candidate_body_atoms),
            "num_candidate_quality_auxiliary_body_atoms": len(candidate_quality_auxiliary_body_atoms),
            "num_candidate_provenance_metadata_body_atoms": len(candidate_provenance_metadata_body_atoms),
            "num_body_atoms": len(body_atoms),
            "has_candidate_atoms": has_candidate_atoms,
            "rule_clause": rule_clause,
        }
        examples_out.append(out_example)
        total_body_atoms += len(body_atoms)
        total_candidate_body_atoms += len(candidate_body_atoms)
        if has_candidate_atoms:
            examples_with_candidate_atoms += 1
        if is_positive:
            positive_count += 1
            positive_candidate_atom_total += len(candidate_body_atoms)
            if has_candidate_atoms:
                positive_examples_with_candidate_atoms += 1
        else:
            negative_count += 1
            negative_candidate_atom_total += len(candidate_body_atoms)
            if has_candidate_atoms:
                negative_examples_with_candidate_atoms += 1

    avg_num_body_atoms = float(total_body_atoms / max(1, len(examples_out)))
    avg_num_candidate_body_atoms = float(total_candidate_body_atoms / max(1, len(examples_out)))
    result: Dict[str, Any] = {
        "version": _TEMPORAL_RULE_EXAMPLES_VERSION,
        "video_id": video_id,
        "num_examples": len(examples_out),
        "num_positive_examples": positive_count,
        "num_negative_examples": negative_count,
        "avg_num_body_atoms": avg_num_body_atoms,
        "num_examples_with_candidate_atoms": examples_with_candidate_atoms,
        "num_positive_examples_with_candidate_atoms": positive_examples_with_candidate_atoms,
        "num_negative_examples_with_candidate_atoms": negative_examples_with_candidate_atoms,
        "avg_num_candidate_body_atoms": avg_num_candidate_body_atoms,
        "avg_num_candidate_body_atoms_positive": (
            float(positive_candidate_atom_total / max(1, positive_count)) if positive_count else 0.0
        ),
        "avg_num_candidate_body_atoms_negative": (
            float(negative_candidate_atom_total / max(1, negative_count)) if negative_count else 0.0
        ),
        "config": {
            "deduplicate_body_atoms": deduplicate_body_atoms,
            "sort_body_atoms": sort_body_atoms,
            "include_clause_text": include_clause_text,
            "include_negative_examples": include_negative_examples,
        },
        "examples": examples_out,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    with csv_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "example_index",
                "example_id",
                "current_segment_id",
                "next_segment_id",
                "current_segment_index",
                "next_segment_index",
                "current_segment_label",
                "next_forward_label",
                "target_predicate",
                "label",
                "head_atom",
                "num_accepted_body_atoms",
                "num_candidate_body_atoms",
                "num_body_atoms",
                "has_candidate_atoms",
                "body_atoms",
                "candidate_body_atoms",
                "candidate_quality_auxiliary_body_atoms",
                "candidate_provenance_metadata_body_atoms",
                "rule_clause",
            ],
        )
        writer.writeheader()
        for example in examples_out:
            writer.writerow(
                {
                    "example_index": example.get("example_index", ""),
                    "example_id": example.get("example_id", ""),
                    "current_segment_id": example.get("current_segment_id", ""),
                    "next_segment_id": example.get("next_segment_id", ""),
                    "current_segment_index": example.get("current_segment_index", ""),
                    "next_segment_index": example.get("next_segment_index", ""),
                    "current_segment_label": example.get("current_segment_label", ""),
                    "next_forward_label": example.get("next_forward_label", ""),
                    "target_predicate": example.get("target_predicate", ""),
                    "label": example.get("label", ""),
                    "head_atom": example.get("head_atom", ""),
                    "num_accepted_body_atoms": example.get("num_accepted_body_atoms", 0),
                    "num_candidate_body_atoms": example.get("num_candidate_body_atoms", 0),
                    "num_body_atoms": example.get("num_body_atoms", 0),
                    "has_candidate_atoms": example.get("has_candidate_atoms", False),
                    "body_atoms": json.dumps(example.get("body_atoms", [])),
                    "candidate_body_atoms": json.dumps(example.get("candidate_body_atoms", [])),
                    "candidate_quality_auxiliary_body_atoms": json.dumps(
                        example.get("candidate_quality_auxiliary_body_atoms", [])
                    ),
                    "candidate_provenance_metadata_body_atoms": json.dumps(
                        example.get("candidate_provenance_metadata_body_atoms", [])
                    ),
                    "rule_clause": example.get("rule_clause", ""),
                }
            )

    _print_video_summary(result)
    return result


def run(
    target_head_results: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    out_root = output_root or get_output_root()
    results: List[Dict[str, Any]] = []

    for target_head_result in target_head_results:
        result = process_video(
            target_head_video_result=target_head_result,
            cfg=cfg,
            output_root=out_root,
            force_recompute=force_recompute,
        )
        results.append(result)

    manifest = {
        "version": _TEMPORAL_RULE_EXAMPLES_VERSION,
        "num_videos": len(results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_examples": r.get("num_examples", 0),
                "num_positive_examples": r.get("num_positive_examples", 0),
                "num_negative_examples": r.get("num_negative_examples", 0),
                "avg_num_body_atoms": r.get("avg_num_body_atoms", 0.0),
                "num_examples_with_candidate_atoms": r.get("num_examples_with_candidate_atoms", 0),
                "num_positive_examples_with_candidate_atoms": r.get("num_positive_examples_with_candidate_atoms", 0),
                "num_negative_examples_with_candidate_atoms": r.get("num_negative_examples_with_candidate_atoms", 0),
                "avg_num_candidate_body_atoms": r.get("avg_num_candidate_body_atoms", 0.0),
                "avg_num_candidate_body_atoms_positive": r.get("avg_num_candidate_body_atoms_positive", 0.0),
                "avg_num_candidate_body_atoms_negative": r.get("avg_num_candidate_body_atoms_negative", 0.0),
            }
            for r in results
        ],
    }
    manifest_path = out_root / "temporal_rule_examples_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Temporal rule examples manifest written to {manifest_path}")
    return results
