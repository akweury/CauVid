"""
Categorize false negatives for each rule selector using step 16/18/19/20 outputs.

Outputs:
    pipeline_output/20b_rule_selection_fn_diagnostic/
        fn_categorization_original.csv
        fn_categorization_diverse.csv
        fn_categorization_coverage_family_aware.csv
        fn_category_summary.csv
        fn_categorization_manifest.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config
from src.exp_driving_videos.modules.evaluate_rules_driving_mini import (
    _find_rule_matches_for_example,
    _get_rule_body_atom_templates,
)
from src.exp_driving_videos.modules.vehicle_rule_diagnostic_driving_mini import (
    _example_vehicle_context,
)


_FN_DIAGNOSTIC_VERSION = 1
_SELECTOR_ORDER = ["original", "diverse", "coverage_family_aware"]
_CATEGORY_ORDER = [
    "covered_by_unselected_candidate_rule",
    "vehicle_context_but_no_selected_rule",
    "predicate_missing_or_weak_context",
    "likely_noisy_or_ambiguous",
]


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "20b_rule_selection_fn_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "vehicle_context_match_levels": sorted(str(v) for v in cfg.get("vehicle_context_match_levels", [])),
        "predicate_gap_levels": sorted(str(v) for v in cfg.get("predicate_gap_levels", [])),
        "noisy_levels": sorted(str(v) for v in cfg.get("noisy_levels", [])),
    }


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _iter_eval_examples(temporal_rule_results: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for video_result in temporal_rule_results:
        video_id = str(video_result.get("video_id", ""))
        for example in list(video_result.get("examples", [])):
            row = dict(example)
            row["video_id"] = video_id
            yield row


def _vehicle_match_level(vehicle_context: Dict[str, Any]) -> str:
    has_vehicle = bool(vehicle_context.get("has_vehicle", False))
    has_near = bool(vehicle_context.get("has_near", False))
    has_centered = bool(vehicle_context.get("has_centered", False))
    if has_vehicle and has_near and has_centered:
        return "exact_vehicle_near_centered"
    if has_vehicle and has_near:
        return "vehicle_near_partial"
    if has_vehicle and has_centered:
        return "vehicle_centered_partial"
    if has_near and has_centered:
        return "near_centered_partial"
    if has_vehicle:
        return "vehicle_only"
    if has_near:
        return "near_only"
    if has_centered:
        return "centered_only"
    return "no_match"


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


def _matching_rule_summaries(
    example: Dict[str, Any],
    candidate_rules: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    body_atoms = list(example.get("body_atoms", []))
    matches: List[Dict[str, Any]] = []
    for rule in candidate_rules:
        body_atom_templates = _get_rule_body_atom_templates(rule)
        match_states = _find_rule_matches_for_example(
            body_atom_templates=body_atom_templates,
            body_atoms=body_atoms,
        )
        if not match_states:
            continue
        matches.append(
            {
                "rule_id": str(rule.get("rule_id", "")),
                "clause": str(rule.get("clause", "")),
                "confidence": float(rule.get("confidence", 0.0)),
                "positive_support": int(rule.get("positive_support", 0)),
                "body_length": int(rule.get("body_length", len(body_atom_templates))),
                "match_count": len(match_states),
            }
        )
    return matches


def _categorize_fn(
    fn_row: Dict[str, Any],
    vehicle_context: Dict[str, Any],
    matching_unselected_rules: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> str:
    vehicle_context_match_levels = {
        str(v)
        for v in cfg.get(
            "vehicle_context_match_levels",
            ["exact_vehicle_near_centered", "vehicle_near_partial", "vehicle_centered_partial"],
        )
    }
    predicate_gap_levels = {
        str(v)
        for v in cfg.get(
            "predicate_gap_levels",
            [
                "missing_rule_or_predicate_dense_context",
                "missing_rule_or_predicate_sparse_context",
                "unexplained_noise_or_symbol_gap",
            ],
        )
    }
    noisy_levels = {
        str(v)
        for v in cfg.get(
            "noisy_levels",
            ["unexplained_noise_no_objects", "unexplained_noise_or_symbol_gap"],
        )
    }

    if matching_unselected_rules:
        return "covered_by_unselected_candidate_rule"

    match_level = _vehicle_match_level(vehicle_context)
    if match_level in vehicle_context_match_levels:
        return "vehicle_context_but_no_selected_rule"

    explainability_level = str(fn_row.get("explainability_level", ""))
    if explainability_level in predicate_gap_levels:
        return "predicate_missing_or_weak_context"

    if explainability_level in noisy_levels:
        return "likely_noisy_or_ambiguous"

    if not any(
        [
            bool(vehicle_context.get("has_vehicle", False)),
            bool(vehicle_context.get("has_near", False)),
            bool(vehicle_context.get("has_centered", False)),
        ]
    ):
        return "predicate_missing_or_weak_context"

    return "likely_noisy_or_ambiguous"


def process_diagnostic(
    extended_rule_results: Dict[str, Any],
    rule_results_by_name: Dict[str, Dict[str, Any]],
    evaluation_results_by_name: Dict[str, Dict[str, Any]],
    error_analysis_results_by_name: Dict[str, Dict[str, Any]],
    temporal_rule_results: List[Dict[str, Any]],
    vehicle_rule_diagnostic_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "fn_categorization_manifest.json"
    summary_path = out_root / "fn_category_summary.csv"

    if not force_recompute and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _FN_DIAGNOSTIC_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {manifest_path.name}")
            return cached

    examples_by_id: Dict[str, Dict[str, Any]] = {}
    for example in _iter_eval_examples(temporal_rule_results):
        example_id = str(example.get("example_id", ""))
        if example_id:
            examples_by_id[example_id] = example

    all_kept_rules = _sort_rules(list(extended_rule_results.get("all_kept_rules", [])))
    category_rows: List[Dict[str, Any]] = []
    selector_summary_rows: List[Dict[str, Any]] = []
    selector_output_paths: Dict[str, str] = {}

    step20_eval_context_path = Path(str(vehicle_rule_diagnostic_results.get("eval_context_summary_path", "")))
    step20_eval_context_rows = _read_csv(step20_eval_context_path) if step20_eval_context_path.exists() else []

    for selector_name in [name for name in _SELECTOR_ORDER if name in rule_results_by_name]:
        selected_rule_ids = {str(rule.get("rule_id", "")) for rule in list(rule_results_by_name[selector_name].get("final_rules", []))}
        unselected_candidate_rules = [
            rule for rule in all_kept_rules if str(rule.get("rule_id", "")) not in selected_rule_ids
        ]

        error_manifest = dict(error_analysis_results_by_name.get(selector_name, {}))
        fn_examples_path = Path(str(error_manifest.get("fn_examples_path", "")))
        fn_rows = _read_csv(fn_examples_path) if fn_examples_path.exists() else []
        recovered_by_other: Dict[str, List[str]] = {}
        for other_selector_name in [name for name in _SELECTOR_ORDER if name in evaluation_results_by_name and name != selector_name]:
            predicted_ids = set(evaluation_results_by_name.get(other_selector_name, {}).get("predicted_positive_example_ids", []))
            for row in fn_rows:
                example_id = str(row.get("example_id", ""))
                if example_id in predicted_ids:
                    recovered_by_other.setdefault(example_id, []).append(other_selector_name)

        selector_rows: List[Dict[str, Any]] = []
        category_counter: Counter[str] = Counter()
        for fn_row in fn_rows:
            example_id = str(fn_row.get("example_id", ""))
            example = examples_by_id.get(example_id)
            if example is None:
                continue
            vehicle_context = _example_vehicle_context(example, cfg)
            match_level = _vehicle_match_level(vehicle_context)
            matching_unselected_rules = _matching_rule_summaries(example, unselected_candidate_rules)
            category = _categorize_fn(fn_row, vehicle_context, matching_unselected_rules, cfg)
            category_counter[category] += 1

            top_unselected = matching_unselected_rules[0] if matching_unselected_rules else {}
            selector_row = {
                "selector_name": selector_name,
                "example_id": example_id,
                "video_id": str(fn_row.get("video_id", example.get("video_id", ""))),
                "category": category,
                "explainability_level": str(fn_row.get("explainability_level", "")),
                "ego_forward_state": str(fn_row.get("ego_forward_state", "")),
                "ego_lateral_state": str(fn_row.get("ego_lateral_state", "")),
                "ego_motion_state": str(fn_row.get("ego_motion_state", "")),
                "has_vehicle": bool(vehicle_context.get("has_vehicle", False)),
                "has_near": bool(vehicle_context.get("has_near", False)),
                "has_centered": bool(vehicle_context.get("has_centered", False)),
                "has_vehicle_near_centered": bool(vehicle_context.get("has_vehicle_near_centered", False)),
                "vehicle_match_level": match_level,
                "num_unselected_covering_rules": len(matching_unselected_rules),
                "top_unselected_rule_id": str(top_unselected.get("rule_id", "")),
                "top_unselected_rule_confidence": float(top_unselected.get("confidence", 0.0)),
                "top_unselected_rule_positive_support": int(top_unselected.get("positive_support", 0)),
                "top_unselected_rule_clause": str(top_unselected.get("clause", "")),
                "recovered_by_other_selectors": json.dumps(sorted(recovered_by_other.get(example_id, []))),
            }
            selector_rows.append(selector_row)
            category_rows.append(selector_row)

        selector_csv_path = out_root / f"fn_categorization_{selector_name}.csv"
        selector_output_paths[selector_name] = str(selector_csv_path)
        _write_csv(
            selector_csv_path,
            [
                "selector_name",
                "example_id",
                "video_id",
                "category",
                "explainability_level",
                "ego_forward_state",
                "ego_lateral_state",
                "ego_motion_state",
                "has_vehicle",
                "has_near",
                "has_centered",
                "has_vehicle_near_centered",
                "vehicle_match_level",
                "num_unselected_covering_rules",
                "top_unselected_rule_id",
                "top_unselected_rule_confidence",
                "top_unselected_rule_positive_support",
                "top_unselected_rule_clause",
                "recovered_by_other_selectors",
            ],
            selector_rows,
        )

        for category_name in _CATEGORY_ORDER:
            selector_summary_rows.append(
                {
                    "selector_name": selector_name,
                    "category": category_name,
                    "num_examples": int(category_counter.get(category_name, 0)),
                    "num_vehicle_context_examples": sum(
                        1
                        for row in selector_rows
                        if str(row.get("category", "")) == category_name and str(row.get("vehicle_match_level", "")) != "no_match"
                    ),
                }
            )

    _write_csv(
        summary_path,
        ["selector_name", "category", "num_examples", "num_vehicle_context_examples"],
        selector_summary_rows,
    )

    manifest = {
        "version": _FN_DIAGNOSTIC_VERSION,
        "config": _cfg_key_subset(cfg),
        "selectors": [name for name in _SELECTOR_ORDER if name in rule_results_by_name],
        "categories": list(_CATEGORY_ORDER),
        "num_all_kept_rules": len(all_kept_rules),
        "selector_output_paths": selector_output_paths,
        "summary_path": str(summary_path),
        "step20_eval_context_summary_path": str(step20_eval_context_path),
        "step20_eval_context_num_rows": len(step20_eval_context_rows),
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(
        "  fn_categorization_diagnostic: "
        f"selectors={len(manifest['selectors'])} | "
        f"all_kept_rules={len(all_kept_rules)} | "
        f"rows={len(category_rows)}"
    )
    print(f"FN category summary written to {summary_path}")
    print(f"FN categorization manifest written to {manifest_path}")
    for selector_name, selector_path in selector_output_paths.items():
        print(f"FN categorization CSV written to {selector_path}")
    return manifest


def run(
    extended_rule_results: Dict[str, Any],
    rule_results_by_name: Dict[str, Dict[str, Any]],
    evaluation_results_by_name: Dict[str, Dict[str, Any]],
    error_analysis_results_by_name: Dict[str, Dict[str, Any]],
    temporal_rule_results: List[Dict[str, Any]],
    vehicle_rule_diagnostic_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_diagnostic(
        extended_rule_results=extended_rule_results,
        rule_results_by_name=rule_results_by_name,
        evaluation_results_by_name=evaluation_results_by_name,
        error_analysis_results_by_name=error_analysis_results_by_name,
        temporal_rule_results=temporal_rule_results,
        vehicle_rule_diagnostic_results=vehicle_rule_diagnostic_results,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
