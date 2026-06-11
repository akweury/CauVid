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


_FINAL_RULES_VERSION = 1


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

    result: Dict[str, Any] = {
        "version": _FINAL_RULES_VERSION,
        "config": {
            "top_k": top_k,
        },
        "num_input_rules": len(all_kept_rules),
        "num_final_rules": len(final_rules),
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
