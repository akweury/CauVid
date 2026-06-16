"""
Select a diverse final rule set by greedily maximizing new positive coverage
while penalizing overlap and repeated rule families.

Consumes:
  - Step 16 output: aggregated kept rules from initial + extended rounds

Output layout:
    pipeline_output/17b_driving_mini_diverse_final_rules/
        diverse_final_rules.json
        diverse_final_rules.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_DIVERSE_FINAL_RULES_VERSION = 1


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "17b_driving_mini_diverse_final_rules"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "top_k": int(cfg.get("top_k", 50)),
        "new_positive_weight": float(cfg.get("new_positive_weight", 1.0)),
        "confidence_weight": float(cfg.get("confidence_weight", 0.25)),
        "overlap_penalty": float(cfg.get("overlap_penalty", 0.35)),
        "family_penalty": float(cfg.get("family_penalty", 0.75)),
        "negative_support_penalty": float(cfg.get("negative_support_penalty", 0.1)),
    }


def _sort_rules(all_kept_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        all_kept_rules,
        key=lambda rule: (
            -float(rule.get("confidence", 0.0)),
            -int(rule.get("positive_support", 0)),
            int(rule.get("kept_round_index", 0)),
            str(rule.get("clause", "")),
        ),
    )


def _parse_atom(atom: str) -> Optional[Tuple[str, List[str]]]:
    text = str(atom).strip()
    match = re.match(r"^([a-z0-9_]+)\((.*)\)\.$", text)
    if not match:
        return None
    predicate = match.group(1)
    args_text = match.group(2).strip()
    if not args_text:
        return predicate, []
    return predicate, [part.strip() for part in args_text.split(",")]


def _get_rule_body_atom_templates(rule: Dict[str, Any]) -> List[str]:
    body_atom_templates = rule.get("body_atom_templates")
    if isinstance(body_atom_templates, list):
        return [str(atom).strip() for atom in body_atom_templates if str(atom).strip()]
    body_atom_template = str(rule.get("body_atom_template", "")).strip()
    return [body_atom_template] if body_atom_template else []


def _rule_family_signature(rule: Dict[str, Any]) -> str:
    predicates: Set[str] = set()
    for atom in _get_rule_body_atom_templates(rule):
        parsed = _parse_atom(atom)
        if parsed is None:
            continue
        predicates.add(str(parsed[0]))
    return "|".join(sorted(predicates))


def _positive_example_ids(rule: Dict[str, Any]) -> Set[str]:
    return {
        str(example_id)
        for example_id in list(rule.get("positive_example_ids", []))
        if str(example_id)
    }


def _rule_utility(
    rule: Dict[str, Any],
    covered_positive_ids: Set[str],
    family_counts: Dict[str, int],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    positive_ids = _positive_example_ids(rule)
    family_signature = _rule_family_signature(rule)
    new_positive_ids = positive_ids - covered_positive_ids
    overlap_positive_ids = positive_ids & covered_positive_ids
    family_reuse_count = int(family_counts.get(family_signature, 0))
    confidence = float(rule.get("confidence", 0.0))
    negative_support = int(rule.get("negative_support", 0))

    new_positive_gain = len(new_positive_ids)
    overlap_positive_count = len(overlap_positive_ids)
    utility = (
        float(cfg.get("new_positive_weight", 1.0)) * new_positive_gain
        + float(cfg.get("confidence_weight", 0.25)) * confidence
        - float(cfg.get("overlap_penalty", 0.35)) * overlap_positive_count
        - float(cfg.get("family_penalty", 0.75)) * family_reuse_count
        - float(cfg.get("negative_support_penalty", 0.1)) * negative_support
    )
    return {
        "utility": utility,
        "family_signature": family_signature,
        "new_positive_ids": new_positive_ids,
        "new_positive_gain": new_positive_gain,
        "overlap_positive_count": overlap_positive_count,
        "family_reuse_count": family_reuse_count,
    }


def process_rules(
    extended_rule_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    top_k = int(cfg.get("top_k", 50))

    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "diverse_final_rules.json"
    csv_path = out_root / "diverse_final_rules.csv"

    if not force_recompute and json_path.exists():
        with json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _DIVERSE_FINAL_RULES_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {json_path.name}")
            return cached

    candidate_rules = _sort_rules(list(extended_rule_results.get("all_kept_rules", [])))
    selected_rules: List[Dict[str, Any]] = []
    selection_trace: List[Dict[str, Any]] = []
    selected_rule_ids: Set[str] = set()
    covered_positive_ids: Set[str] = set()
    family_counts: Dict[str, int] = {}

    while len(selected_rules) < max(0, top_k):
        best_rule: Optional[Dict[str, Any]] = None
        best_trace: Optional[Dict[str, Any]] = None

        for rule in candidate_rules:
            rule_id = str(rule.get("rule_id", ""))
            if rule_id in selected_rule_ids:
                continue
            utility = _rule_utility(rule, covered_positive_ids, family_counts, cfg)
            trace = {
                "rule_id": rule_id,
                "clause": str(rule.get("clause", "")),
                "utility": float(utility["utility"]),
                "new_positive_gain": int(utility["new_positive_gain"]),
                "overlap_positive_count": int(utility["overlap_positive_count"]),
                "family_reuse_count": int(utility["family_reuse_count"]),
                "family_signature": str(utility["family_signature"]),
                "confidence": float(rule.get("confidence", 0.0)),
                "positive_support": int(rule.get("positive_support", 0)),
                "negative_support": int(rule.get("negative_support", 0)),
                "new_positive_ids": sorted(str(v) for v in utility["new_positive_ids"]),
            }
            if best_rule is None:
                best_rule = rule
                best_trace = trace
                continue
            assert best_trace is not None
            current_key = (
                float(trace["utility"]),
                int(trace["new_positive_gain"]),
                -int(trace["family_reuse_count"]),
                float(trace["confidence"]),
                int(trace["positive_support"]),
                str(trace["clause"]),
            )
            best_key = (
                float(best_trace["utility"]),
                int(best_trace["new_positive_gain"]),
                -int(best_trace["family_reuse_count"]),
                float(best_trace["confidence"]),
                int(best_trace["positive_support"]),
                str(best_trace["clause"]),
            )
            if current_key > best_key:
                best_rule = rule
                best_trace = trace

        if best_rule is None or best_trace is None:
            break

        selected_rule = dict(best_rule)
        selected_rule["selection_rank"] = len(selected_rules)
        selected_rule["selection_method"] = "greedy_diverse_positive_coverage"
        selected_rule["selection_utility"] = float(best_trace["utility"])
        selected_rule["selection_family_signature"] = str(best_trace["family_signature"])
        selected_rule["selection_new_positive_gain"] = int(best_trace["new_positive_gain"])
        selected_rule["selection_overlap_positive_count"] = int(best_trace["overlap_positive_count"])
        selected_rule["selection_family_reuse_count"] = int(best_trace["family_reuse_count"])
        selected_rules.append(selected_rule)

        selected_rule_ids.add(str(best_rule.get("rule_id", "")))
        covered_positive_ids.update(best_trace["new_positive_ids"])
        family_signature = str(best_trace["family_signature"])
        family_counts[family_signature] = family_counts.get(family_signature, 0) + 1

        selection_trace.append(
            {
                "selection_rank": len(selected_rules) - 1,
                "rule_id": str(best_rule.get("rule_id", "")),
                "clause": str(best_rule.get("clause", "")),
                "selection_utility": float(best_trace["utility"]),
                "new_positive_gain": int(best_trace["new_positive_gain"]),
                "overlap_positive_count": int(best_trace["overlap_positive_count"]),
                "family_reuse_count": int(best_trace["family_reuse_count"]),
                "family_signature": family_signature,
                "cumulative_positive_coverage": len(covered_positive_ids),
            }
        )

    result: Dict[str, Any] = {
        "version": _DIVERSE_FINAL_RULES_VERSION,
        "config": _cfg_key_subset(cfg),
        "selection_method": "greedy_diverse_positive_coverage",
        "num_input_rules": len(candidate_rules),
        "num_final_rules": len(selected_rules),
        "num_distinct_families": len(family_counts),
        "covered_training_positive_examples": len(covered_positive_ids),
        "final_rules": selected_rules,
        "selection_trace": selection_trace,
    }

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "selection_rank",
                "rule_id",
                "kept_stage",
                "kept_round_index",
                "clause",
                "confidence",
                "positive_support",
                "negative_support",
                "total_support",
                "selection_utility",
                "selection_family_signature",
                "selection_new_positive_gain",
                "selection_overlap_positive_count",
                "selection_family_reuse_count",
            ],
        )
        writer.writeheader()
        for rule in selected_rules:
            writer.writerow(
                {
                    "selection_rank": rule.get("selection_rank", ""),
                    "rule_id": rule.get("rule_id", ""),
                    "kept_stage": rule.get("kept_stage", ""),
                    "kept_round_index": rule.get("kept_round_index", ""),
                    "clause": rule.get("clause", ""),
                    "confidence": rule.get("confidence", 0.0),
                    "positive_support": rule.get("positive_support", 0),
                    "negative_support": rule.get("negative_support", 0),
                    "total_support": rule.get("total_support", 0),
                    "selection_utility": rule.get("selection_utility", 0.0),
                    "selection_family_signature": rule.get("selection_family_signature", ""),
                    "selection_new_positive_gain": rule.get("selection_new_positive_gain", 0),
                    "selection_overlap_positive_count": rule.get("selection_overlap_positive_count", 0),
                    "selection_family_reuse_count": rule.get("selection_family_reuse_count", 0),
                }
            )

    print(
        "  diverse_final_rules: "
        f"input={len(candidate_rules)} | "
        f"top_k={top_k} | "
        f"kept={len(selected_rules)} | "
        f"families={len(family_counts)} | "
        f"covered_training_positive_examples={len(covered_positive_ids)}"
    )
    print(f"Diverse final rules JSON written to {json_path}")
    print(f"Diverse final rules CSV written to {csv_path}")
    return result


def run(
    extended_rule_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_rules(
        extended_rule_results=extended_rule_results,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
