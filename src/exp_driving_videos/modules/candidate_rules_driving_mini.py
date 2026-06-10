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


_INITIAL_RULES_VERSION = 6


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "driving_mini_initial_rules"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "target_predicate",
        "negative_target_predicate",
        "use_only_positive_examples",
        "min_positive_support",
        "include_example_ids",
        "ignored_body_predicates",
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


def _extract_bindings(
    body_atom_template: str,
    concrete_atom: str,
) -> Optional[Dict[str, str]]:
    template_parsed = _parse_atom(body_atom_template)
    concrete_parsed = _parse_atom(concrete_atom)
    if template_parsed is None or concrete_parsed is None:
        return None

    template_predicate, template_args = template_parsed
    concrete_predicate, concrete_args = concrete_parsed
    if template_predicate != concrete_predicate or len(template_args) != len(concrete_args):
        return None

    bindings: Dict[str, str] = {}
    for template_arg, concrete_arg in zip(template_args, concrete_args):
        if template_arg in {"S", "O", "T", "F"}:
            existing = bindings.get(template_arg)
            if existing is not None and existing != concrete_arg:
                return None
            bindings[template_arg] = concrete_arg
            continue
        if template_arg != concrete_arg:
            return None
    return bindings


def _make_evidence_entry(
    video_id: str,
    example: Dict[str, Any],
    body_atom_template: str,
    concrete_atom: str,
) -> Optional[Dict[str, Any]]:
    bindings = _extract_bindings(body_atom_template, concrete_atom)
    if bindings is None:
        return None

    return {
        "video_id": video_id,
        "example_id": str(example.get("example_id", "")),
        "current_segment_id": str(example.get("current_segment_id", "")),
        "next_segment_id": str(example.get("next_segment_id", "")),
        "target_predicate": str(example.get("target_predicate", "")),
        "label": bool(example.get("label", False)),
        "body_atom_template": body_atom_template,
        "matched_atom": str(concrete_atom),
        "bindings": bindings,
    }


def _dedupe_evidence_entries(evidence_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, Tuple[Tuple[str, str], ...]]] = set()
    for entry in evidence_entries:
        key = (
            str(entry.get("example_id", "")),
            str(entry.get("matched_atom", "")),
            tuple(sorted((str(k), str(v)) for k, v in dict(entry.get("bindings", {})).items())),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _summarize_evidence(
    evidence_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    total_firings = len(evidence_entries)
    positive_firings = sum(1 for entry in evidence_entries if bool(entry.get("label", False)))
    negative_firings = total_firings - positive_firings

    positive_example_ids = sorted(
        {
            str(entry.get("example_id", ""))
            for entry in evidence_entries
            if bool(entry.get("label", False)) and str(entry.get("example_id", ""))
        }
    )
    negative_example_ids = sorted(
        {
            str(entry.get("example_id", ""))
            for entry in evidence_entries
            if not bool(entry.get("label", False)) and str(entry.get("example_id", ""))
        }
    )
    total_example_ids = sorted(set(positive_example_ids) | set(negative_example_ids))
    confidence = float(positive_firings / max(1, total_firings))

    return {
        "positive_support": len(positive_example_ids),
        "negative_support": len(negative_example_ids),
        "total_support": len(total_example_ids),
        "positive_firings": positive_firings,
        "negative_firings": negative_firings,
        "total_firings": total_firings,
        "confidence": confidence,
        "positive_example_ids": positive_example_ids,
        "negative_example_ids": negative_example_ids,
    }


def process_video(
    temporal_rule_examples_video_result: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    target_predicate = str(cfg.get("target_predicate", "brake_next"))
    negative_target_predicate = str(cfg.get("negative_target_predicate", f"not_{target_predicate}"))
    use_only_positive_examples = bool(cfg.get("use_only_positive_examples", True))
    min_positive_support = int(cfg.get("min_positive_support", 1))
    include_example_ids = bool(cfg.get("include_example_ids", True))
    ignored_body_predicates = {
        str(name).strip()
        for name in cfg.get(
            "ignored_body_predicates",
            [
                "segment",
                "segment_start_frame",
                "segment_end_frame",
                "object_in_segment",
                "object_track",
            ],
        )
        if str(name).strip()
    }

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
                    "negative_target_predicate": negative_target_predicate,
                    "use_only_positive_examples": use_only_positive_examples,
                    "min_positive_support": min_positive_support,
                    "include_example_ids": include_example_ids,
                    "ignored_body_predicates": sorted(ignored_body_predicates),
                }
            )
        ):
            print(f"  [cache] {video_id} - loading {out_file.name}")
            _print_video_summary(cached)
            return cached

    examples = list(temporal_rule_examples_video_result.get("examples", []))
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"evidence_set": []})
    candidate_body_templates: set[str] = set()
    total_positive_examples = sum(1 for example in examples if bool(example.get("label", False)))
    total_negative_examples = sum(1 for example in examples if not bool(example.get("label", False)))

    for example in examples:
        example_target_predicate = str(example.get("target_predicate", ""))
        if example_target_predicate not in {target_predicate, negative_target_predicate}:
            continue

        is_positive = bool(example.get("label", False))
        example_id = str(example.get("example_id", ""))
        body_atoms = list(example.get("body_atoms", []))
        seen_templates: set[str] = set()
        for body_atom in body_atoms:
            parsed_body_atom = _parse_atom(str(body_atom))
            if parsed_body_atom is None:
                continue
            body_predicate, _ = parsed_body_atom
            if body_predicate in ignored_body_predicates:
                continue
            body_template = _abstract_atom(str(body_atom))
            if not body_template or body_template in seen_templates:
                continue
            seen_templates.add(body_template)
            evidence_entry = _make_evidence_entry(
                video_id=video_id,
                example=example,
                body_atom_template=body_template,
                concrete_atom=str(body_atom),
            )
            if evidence_entry is not None:
                stats[body_template]["evidence_set"].append(evidence_entry)

        if is_positive or not use_only_positive_examples:
            candidate_body_templates.update(seen_templates)

    initial_rules: List[Dict[str, Any]] = []
    for idx, body_template in enumerate(sorted(stats.keys())):
        if body_template not in candidate_body_templates:
            continue

        evidence_set = _dedupe_evidence_entries(list(stats[body_template].get("evidence_set", [])))
        evidence_summary = _summarize_evidence(evidence_set)
        positive_support = int(evidence_summary["positive_support"])
        negative_support = int(evidence_summary["negative_support"])
        total_support = int(evidence_summary["total_support"])
        if positive_support < min_positive_support:
            continue

        confidence = float(evidence_summary["confidence"])
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
                "positive_firings": int(evidence_summary["positive_firings"]),
                "negative_firings": int(evidence_summary["negative_firings"]),
                "total_firings": int(evidence_summary["total_firings"]),
                "confidence": confidence,
                "positive_example_ids": (
                    evidence_summary["positive_example_ids"] if include_example_ids else []
                ),
                "negative_example_ids": (
                    evidence_summary["negative_example_ids"] if include_example_ids else []
                ),
                "evidence_set": evidence_set,
            }
        )

    result: Dict[str, Any] = {
        "version": _INITIAL_RULES_VERSION,
        "video_id": video_id,
        "target_predicate": target_predicate,
        "num_initial_rules": len(initial_rules),
        "num_examples": len(examples),
        "num_positive_examples": total_positive_examples,
        "num_negative_examples": total_negative_examples,
        "config": {
            "target_predicate": target_predicate,
            "negative_target_predicate": negative_target_predicate,
            "use_only_positive_examples": use_only_positive_examples,
            "min_positive_support": min_positive_support,
            "include_example_ids": include_example_ids,
            "ignored_body_predicates": sorted(ignored_body_predicates),
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
                "positive_firings",
                "negative_firings",
                "total_firings",
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
                    "positive_firings": rule.get("positive_firings", 0),
                    "negative_firings": rule.get("negative_firings", 0),
                    "total_firings": rule.get("total_firings", 0),
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
                "num_examples": r.get("num_examples", 0),
                "num_positive_examples": r.get("num_positive_examples", 0),
                "num_negative_examples": r.get("num_negative_examples", 0),
            }
            for r in results
        ],
    }
    manifest_path = out_root / "initial_rules_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Initial rules manifest written to {manifest_path}")
    return results
