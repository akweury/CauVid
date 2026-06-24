"""
Select the final rule set by ranking all kept rules and retaining the top-k.

Consumes:
  - Step 16 output: aggregated kept rules from initial + extended rounds

Output layout:
    pipeline_output/17_driving_mini_final_rules/
        final_rules.json
        final_rules.csv
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


_FINAL_RULES_VERSION = 2


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "17_driving_mini_final_rules"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "top_k": int(cfg.get("top_k", 50)),
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


def _rule_candidate_category(rule: Dict[str, Any]) -> str:
    if bool(rule.get("mixes_accepted_and_candidate_atoms", False)):
        return "mixed_accepted_candidate"
    if bool(rule.get("uses_only_candidate_atoms", False)):
        return "candidate_only"
    if bool(rule.get("uses_candidate_atoms", False)):
        return "candidate_only"
    return "accepted_only"


def _summarize_candidate_rule_subset(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    matched_prior_id_counts: Dict[str, int] = {}
    total_candidate_body_atoms = 0
    candidate_body_atom_ratios: List[float] = []
    candidate_rule_count = 0
    for rule in rules:
        if bool(rule.get("uses_candidate_atoms", False)):
            candidate_rule_count += 1
        total_candidate_body_atoms += max(0, int(rule.get("num_candidate_body_atoms", 0)))
        candidate_body_atom_ratios.append(max(0.0, float(rule.get("candidate_body_atom_ratio", 0.0))))
        for prior_id in list(rule.get("matched_prior_ids_involved", [])):
            prior_text = str(prior_id).strip()
            if prior_text:
                matched_prior_id_counts[prior_text] = matched_prior_id_counts.get(prior_text, 0) + 1
    return {
        "num_rules": len(rules),
        "num_rules_using_candidate_atoms": candidate_rule_count,
        "total_candidate_body_atoms": total_candidate_body_atoms,
        "avg_candidate_body_atom_ratio": float(sum(candidate_body_atom_ratios) / max(1, len(candidate_body_atom_ratios))),
        "avg_num_candidate_body_atoms": float(total_candidate_body_atoms / max(1, len(rules))),
        "matched_prior_id_counts": {
            key: matched_prior_id_counts[key]
            for key in sorted(matched_prior_id_counts)
        },
    }


def _summarize_candidate_rule_selection(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    subset_rules = {
        "all_rules": list(rules),
        "accepted_only_rules": [rule for rule in rules if _rule_candidate_category(rule) == "accepted_only"],
        "candidate_only_rules": [rule for rule in rules if _rule_candidate_category(rule) == "candidate_only"],
        "mixed_accepted_candidate_rules": [
            rule for rule in rules if _rule_candidate_category(rule) == "mixed_accepted_candidate"
        ],
    }
    return {
        "category_counts": {
            "accepted_only_rules": len(subset_rules["accepted_only_rules"]),
            "candidate_only_rules": len(subset_rules["candidate_only_rules"]),
            "mixed_accepted_candidate_rules": len(subset_rules["mixed_accepted_candidate_rules"]),
            "all_rules": len(subset_rules["all_rules"]),
        },
        "subsets": {
            subset_name: _summarize_candidate_rule_subset(subset)
            for subset_name, subset in subset_rules.items()
        },
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
    json_path = out_root / "final_rules.json"
    csv_path = out_root / "final_rules.csv"

    if not force_recompute and json_path.exists():
        with json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _FINAL_RULES_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset({"top_k": top_k}):
            print(f"  [cache] loading {json_path.name}")
            return cached

    all_kept_rules = list(extended_rule_results.get("all_kept_rules", []))
    ranked_rules = _sort_rules(all_kept_rules)
    final_rules = ranked_rules[: max(0, top_k)]
    for rule in final_rules:
        rule["candidate_rule_category"] = _rule_candidate_category(rule)
    candidate_rule_diagnostics = _summarize_candidate_rule_selection(final_rules)

    result: Dict[str, Any] = {
        "version": _FINAL_RULES_VERSION,
        "config": {
            "top_k": top_k,
        },
        "selection_method": "score_top_k",
        "num_input_rules": len(all_kept_rules),
        "num_final_rules": len(final_rules),
        "candidate_rule_diagnostics": candidate_rule_diagnostics,
        "final_rules": final_rules,
    }

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rule_id",
                "kept_stage",
                "kept_round_index",
                "clause",
                "confidence",
                "positive_support",
                "negative_support",
                "total_support",
                "positive_firings",
                "negative_firings",
                "total_firings",
                "candidate_rule_category",
                "uses_candidate_atoms",
                "num_candidate_body_atoms",
                "candidate_body_atom_ratio",
                "mixes_accepted_and_candidate_atoms",
                "uses_only_candidate_atoms",
                "body_source_mix",
                "matched_prior_ids_involved",
            ],
        )
        writer.writeheader()
        for rule in final_rules:
            writer.writerow(
                {
                    "rule_id": rule.get("rule_id", ""),
                    "kept_stage": rule.get("kept_stage", ""),
                    "kept_round_index": rule.get("kept_round_index", ""),
                    "clause": rule.get("clause", ""),
                    "confidence": rule.get("confidence", 0.0),
                    "positive_support": rule.get("positive_support", 0),
                    "negative_support": rule.get("negative_support", 0),
                    "total_support": rule.get("total_support", 0),
                    "positive_firings": rule.get("positive_firings", 0),
                    "negative_firings": rule.get("negative_firings", 0),
                    "total_firings": rule.get("total_firings", 0),
                    "candidate_rule_category": rule.get("candidate_rule_category", "accepted_only"),
                    "uses_candidate_atoms": rule.get("uses_candidate_atoms", False),
                    "num_candidate_body_atoms": rule.get("num_candidate_body_atoms", 0),
                    "candidate_body_atom_ratio": rule.get("candidate_body_atom_ratio", 0.0),
                    "mixes_accepted_and_candidate_atoms": rule.get("mixes_accepted_and_candidate_atoms", False),
                    "uses_only_candidate_atoms": rule.get("uses_only_candidate_atoms", False),
                    "body_source_mix": rule.get("body_source_mix", ""),
                    "matched_prior_ids_involved": json.dumps(rule.get("matched_prior_ids_involved", [])),
                }
            )

    print(
        "  final_rules: "
        f"input={len(all_kept_rules)} | "
        f"top_k={top_k} | "
        f"kept={len(final_rules)}"
    )
    print(f"Final rules JSON written to {json_path}")
    print(f"Final rules CSV written to {csv_path}")
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
