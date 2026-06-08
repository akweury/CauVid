"""
Generate temporal initial rules from step-13 rule-learning examples.

Current scope:
  - Only unary-body temporal clauses are generated.
  - Rule head is the target predicate, currently brake_next.
  - Rules are intended as short seed clauses for later extension steps.

Output layout:
    pipeline_output/driving_mini_initial_rules/
        initial_rules_manifest.json
        <video_id>/
            initial_rules.json
            initial_rules.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_INITIAL_RULES_VERSION = 2


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "driving_mini_initial_rules"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "target_predicate",
        "use_only_positive_examples",
        "min_positive_support",
        "include_example_ids",
    ]
    return {k: cfg.get(k) for k in keys}


def _print_video_summary(result: Dict[str, Any]) -> None:
    print(
        f"  {result.get('video_id', 'unknown')}: "
        f"initial_rules={int(result.get('num_initial_rules', result.get('num_candidate_rules', 0)))} | "
        f"target={result.get('target_predicate', 'unknown')}"
    )


def _parse_atom(atom: str) -> Optional[Tuple[str, List[str]]]:
    text = str(atom).strip()
    m = re.match(r"^([a-z0-9_]+)\((.*)\)\.$", text)
    if not m:
        return None
    predicate = m.group(1)
    args_text = m.group(2).strip()
    if not args_text:
        return predicate, []
    args = [part.strip() for part in args_text.split(",")]
    return predicate, args


def _abstract_arg(arg: str) -> str:
    text = str(arg)
    if text.startswith("seg_"):
        return "S"
    if text.startswith("obj_"):
        return "O"
    if text.startswith("track_"):
        return "T"
    if text.startswith("frame_"):
        return "F"
    return text


def _abstract_atom(atom: str) -> Optional[str]:
    parsed = _parse_atom(atom)
    if parsed is None:
        return None
    predicate, args = parsed
    abstract_args = [_abstract_arg(arg) for arg in args]
    return f"{predicate}({','.join(abstract_args)})."


def _build_clause(head_predicate: str, body_atom_template: str) -> str:
    head = f"{head_predicate}(S)."
    head_no_dot = head[:-1]
    return f"{head_no_dot} :- {body_atom_template}"


def process_video(
    temporal_rule_examples_video_result: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    target_predicate = str(cfg.get("target_predicate", "brake_next"))
    use_only_positive_examples = bool(cfg.get("use_only_positive_examples", True))
    min_positive_support = int(cfg.get("min_positive_support", 1))
    include_example_ids = bool(cfg.get("include_example_ids", True))

    video_id = str(temporal_rule_examples_video_result["video_id"])
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "initial_rules.json"
    csv_file = out_dir / "initial_rules.csv"

    if not force_recompute and out_file.exists():
        with out_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        cache_cfg = cached.get("config", {}) if isinstance(cached, dict) else {}
        if (
            int(cached.get("version", 0)) == _INITIAL_RULES_VERSION
            and _cfg_key_subset(cache_cfg) == _cfg_key_subset(
                {
                    "target_predicate": target_predicate,
                    "use_only_positive_examples": use_only_positive_examples,
                    "min_positive_support": min_positive_support,
                    "include_example_ids": include_example_ids,
                }
            )
        ):
            print(f"  [cache] {video_id} - loading {out_file.name}")
            _print_video_summary(cached)
            return cached

    examples = list(temporal_rule_examples_video_result.get("examples", []))
    stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "positive_support": 0,
            "negative_support": 0,
            "total_support": 0,
            "positive_example_ids": [],
            "negative_example_ids": [],
        }
    )

    for example in examples:
        if str(example.get("target_predicate", "")) != target_predicate:
            continue
        is_positive = bool(example.get("label", False))
        if use_only_positive_examples and not is_positive:
            continue

        example_id = str(example.get("example_id", ""))
        body_atoms = list(example.get("body_atoms", []))
        seen_templates: set[str] = set()
        for body_atom in body_atoms:
            body_template = _abstract_atom(str(body_atom))
            if not body_template or body_template in seen_templates:
                continue
            seen_templates.add(body_template)

            stat = stats[body_template]
            stat["total_support"] += 1
            if is_positive:
                stat["positive_support"] += 1
                if include_example_ids:
                    stat["positive_example_ids"].append(example_id)
            else:
                stat["negative_support"] += 1
                if include_example_ids:
                    stat["negative_example_ids"].append(example_id)

    initial_rules: List[Dict[str, Any]] = []
    for idx, body_template in enumerate(sorted(stats.keys())):
        stat = stats[body_template]
        positive_support = int(stat["positive_support"])
        negative_support = int(stat["negative_support"])
        total_support = int(stat["total_support"])
        if positive_support < min_positive_support:
            continue

        confidence = float(positive_support / max(1, total_support))
        clause = _build_clause(target_predicate, body_template)
        initial_rules.append(
            {
                "rule_index": len(initial_rules),
                "rule_id": f"{video_id}_rule_{idx:04d}",
                "head_predicate": target_predicate,
                "head_atom_template": f"{target_predicate}(S).",
                "body_atom_template": body_template,
                "clause": clause,
                "positive_support": positive_support,
                "negative_support": negative_support,
                "total_support": total_support,
                "confidence": confidence,
                "positive_example_ids": stat["positive_example_ids"] if include_example_ids else [],
                "negative_example_ids": stat["negative_example_ids"] if include_example_ids else [],
            }
        )

    result: Dict[str, Any] = {
        "version": _INITIAL_RULES_VERSION,
        "video_id": video_id,
        "target_predicate": target_predicate,
        "num_initial_rules": len(initial_rules),
        "config": {
            "target_predicate": target_predicate,
            "use_only_positive_examples": use_only_positive_examples,
            "min_positive_support": min_positive_support,
            "include_example_ids": include_example_ids,
        },
        "initial_rules": initial_rules,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    with csv_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rule_index",
                "rule_id",
                "head_predicate",
                "head_atom_template",
                "body_atom_template",
                "clause",
                "positive_support",
                "negative_support",
                "total_support",
                "confidence",
                "positive_example_ids",
                "negative_example_ids",
            ],
        )
        writer.writeheader()
        for rule in initial_rules:
            writer.writerow(
                {
                    "rule_index": rule.get("rule_index", ""),
                    "rule_id": rule.get("rule_id", ""),
                    "head_predicate": rule.get("head_predicate", ""),
                    "head_atom_template": rule.get("head_atom_template", ""),
                    "body_atom_template": rule.get("body_atom_template", ""),
                    "clause": rule.get("clause", ""),
                    "positive_support": rule.get("positive_support", 0),
                    "negative_support": rule.get("negative_support", 0),
                    "total_support": rule.get("total_support", 0),
                    "confidence": rule.get("confidence", 0.0),
                    "positive_example_ids": json.dumps(rule.get("positive_example_ids", [])),
                    "negative_example_ids": json.dumps(rule.get("negative_example_ids", [])),
                }
            )

    _print_video_summary(result)
    return result


def run(
    temporal_rule_results: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    out_root = output_root or get_output_root()
    results: List[Dict[str, Any]] = []

    for temporal_rule_result in temporal_rule_results:
        result = process_video(
            temporal_rule_examples_video_result=temporal_rule_result,
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
                "target_predicate": r.get("target_predicate", ""),
                "num_initial_rules": r.get("num_initial_rules", r.get("num_candidate_rules", 0)),
            }
            for r in results
        ],
    }
    manifest_path = out_root / "initial_rules_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Initial rules manifest written to {manifest_path}")
    return results
