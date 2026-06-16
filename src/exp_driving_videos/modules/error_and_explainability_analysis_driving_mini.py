"""
Analyze held-out evaluation errors and generate explainability-oriented summaries.

Consumes:
  - Step 13 output: temporal rule-learning examples for evaluation videos
  - Step 17 output: final learned rules
  - Step 18 output in memory: evaluation summary with per-rule eval metrics

Output layout:
    pipeline_output/19_driving_mini_error_and_explainability_analysis/
        fn_examples.csv
        fp_examples.csv
        per_video_error_summary.csv
        rule_family_summary.csv
        uncovered_positive_summary.csv
        error_and_explainability_manifest.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    _parse_atom,
)


_ANALYSIS_VERSION = 1


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "19_driving_mini_error_and_explainability_analysis"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "vehicle_classes": sorted(str(v) for v in cfg.get("vehicle_classes", [])),
        "dense_context_min_objects": int(cfg.get("dense_context_min_objects", 2)),
        "overlap_rule_threshold": int(cfg.get("overlap_rule_threshold", 10)),
        "rule_set_name": str(cfg.get("rule_set_name", "")),
    }


def _body_predicates(rule: Dict[str, Any]) -> List[str]:
    predicates: List[str] = []
    for atom in _get_rule_body_atom_templates(rule):
        parsed = _parse_atom(str(atom))
        if parsed is None:
            continue
        predicates.append(str(parsed[0]))
    return sorted(set(predicates))


def _build_example_feature_row(
    example: Dict[str, Any],
    matching_rule_ids: List[str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    objects: Dict[str, Dict[str, str]] = {}
    ego_forward_state = "unknown"
    ego_lateral_state = "unknown"
    ego_motion_state = "unknown"

    for atom in list(example.get("body_atoms", [])):
        parsed = _parse_atom(str(atom))
        if parsed is None:
            continue
        predicate, args = parsed
        if predicate == "segment_forward_state" and len(args) >= 2:
            ego_forward_state = str(args[1])
        elif predicate == "segment_lateral_state" and len(args) >= 2:
            ego_lateral_state = str(args[1])
        elif predicate == "segment_motion_state" and len(args) >= 2:
            ego_motion_state = str(args[1])
        elif predicate == "object_in_segment" and len(args) >= 2:
            objects.setdefault(str(args[1]), {})
        elif predicate == "object_track" and len(args) >= 1:
            objects.setdefault(str(args[0]), {})
        elif predicate.startswith("object_") and len(args) >= 2:
            object_id = str(args[1])
            obj = objects.setdefault(object_id, {})
            if predicate == "object_class" and len(args) >= 3:
                obj["object_class"] = str(args[2])
            elif predicate == "object_distance_state" and len(args) >= 3:
                obj["distance_state"] = str(args[2])
            elif predicate == "object_x_position_state" and len(args) >= 3:
                obj["x_position_state"] = str(args[2])
            elif predicate == "object_visibility_state" and len(args) >= 3:
                obj["visibility_state"] = str(args[2])
            elif predicate == "object_vx_state" and len(args) >= 3:
                obj["vx_state"] = str(args[2])

    object_rows = list(objects.values())
    vehicle_classes = {
        str(name).strip()
        for name in cfg.get("vehicle_classes", ["car", "truck", "bus", "motorcycle", "bicycle"])
        if str(name).strip()
    }

    def _distance_rank(distance_state: str) -> int:
        return {"near": 0, "medium": 1, "far": 2}.get(str(distance_state), 3)

    def _x_rank(x_state: str) -> int:
        return {"centered": 0, "left_of_ego": 1, "right_of_ego": 1}.get(str(x_state), 2)

    def _visibility_rank(visibility_state: str) -> int:
        return {"persistent": 0, "intermittent": 1, "brief": 2}.get(str(visibility_state), 3)

    nearest_object = min(
        object_rows,
        key=lambda obj: (
            _distance_rank(obj.get("distance_state", "unknown")),
            _x_rank(obj.get("x_position_state", "unknown")),
            _visibility_rank(obj.get("visibility_state", "unknown")),
            str(obj.get("object_class", "unknown")),
        ),
        default={},
    )

    has_pedestrian = any(str(obj.get("object_class", "")) == "pedestrian" for obj in object_rows)
    has_vehicle = any(str(obj.get("object_class", "")) in vehicle_classes for obj in object_rows)
    has_traffic_light = any(str(obj.get("object_class", "")) == "traffic_light" for obj in object_rows)
    has_near_object = any(str(obj.get("distance_state", "")) == "near" for obj in object_rows)
    has_center_object = any(str(obj.get("x_position_state", "")) == "centered" for obj in object_rows)
    has_stable_visible_object = any(
        str(obj.get("vx_state", "")) == "vx_stable"
        and str(obj.get("visibility_state", "")) in {"persistent", "intermittent"}
        for obj in object_rows
    )

    predicted_positive = bool(matching_rule_ids)
    label = bool(example.get("label", False))
    explainability_level = _classify_explainability_level(
        label=label,
        predicted_positive=predicted_positive,
        num_matching_rules=len(matching_rule_ids),
        num_objects=len(objects),
        has_pedestrian=has_pedestrian,
        has_vehicle=has_vehicle,
        has_traffic_light=has_traffic_light,
        has_near_object=has_near_object,
        has_center_object=has_center_object,
        has_stable_visible_object=has_stable_visible_object,
        dense_context_min_objects=int(cfg.get("dense_context_min_objects", 2)),
        overlap_rule_threshold=int(cfg.get("overlap_rule_threshold", 10)),
    )

    return {
        "video_id": str(example.get("video_id", "")),
        "example_id": str(example.get("example_id", "")),
        "label": label,
        "predicted_positive": predicted_positive,
        "num_matching_rules": len(matching_rule_ids),
        "ego_forward_state": ego_forward_state,
        "ego_lateral_state": ego_lateral_state,
        "ego_motion_state": ego_motion_state,
        "num_objects": len(objects),
        "has_pedestrian": has_pedestrian,
        "has_vehicle": has_vehicle,
        "has_traffic_light": has_traffic_light,
        "has_near_object": has_near_object,
        "has_center_object": has_center_object,
        "has_stable_visible_object": has_stable_visible_object,
        "nearest_object_class": str(nearest_object.get("object_class", "unknown")),
        "nearest_object_distance_state": str(nearest_object.get("distance_state", "unknown")),
        "nearest_object_x_position_state": str(nearest_object.get("x_position_state", "unknown")),
        "explainability_level": explainability_level,
    }


def _classify_explainability_level(
    *,
    label: bool,
    predicted_positive: bool,
    num_matching_rules: int,
    num_objects: int,
    has_pedestrian: bool,
    has_vehicle: bool,
    has_traffic_light: bool,
    has_near_object: bool,
    has_center_object: bool,
    has_stable_visible_object: bool,
    dense_context_min_objects: int,
    overlap_rule_threshold: int,
) -> str:
    if label and not predicted_positive:
        context_signals = sum(
            1
            for flag in [
                has_pedestrian,
                has_vehicle,
                has_traffic_light,
                has_near_object,
                has_center_object,
                has_stable_visible_object,
            ]
            if flag
        )
        if num_objects == 0:
            return "unexplained_noise_no_objects"
        if num_objects >= dense_context_min_objects and context_signals >= 2:
            return "missing_rule_or_predicate_dense_context"
        if context_signals >= 1:
            return "missing_rule_or_predicate_sparse_context"
        return "unexplained_noise_or_symbol_gap"

    if (not label) and predicted_positive:
        if num_matching_rules >= overlap_rule_threshold:
            return "overgeneralized_rule_overlap"
        if has_pedestrian or has_traffic_light or has_near_object or has_center_object:
            return "ambiguous_context_false_alarm"
        return "noisy_rule_trigger"

    return "not_error"


def _iter_eval_examples(temporal_rule_results: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for video_result in temporal_rule_results:
        video_id = str(video_result.get("video_id", ""))
        for example in list(video_result.get("examples", [])):
            example_out = dict(example)
            example_out["video_id"] = video_id
            yield example_out


def _matching_rule_ids_for_example(
    example: Dict[str, Any],
    final_rules: List[Dict[str, Any]],
) -> List[str]:
    matching_rule_ids: List[str] = []
    body_atoms = list(example.get("body_atoms", []))
    for rule in final_rules:
        rule_id = str(rule.get("rule_id", ""))
        body_atom_templates = _get_rule_body_atom_templates(rule)
        match_states = _find_rule_matches_for_example(
            body_atom_templates=body_atom_templates,
            body_atoms=body_atoms,
        )
        if match_states:
            matching_rule_ids.append(rule_id)
    return matching_rule_ids


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def process_analysis(
    final_rule_results: Dict[str, Any],
    temporal_rule_results: List[Dict[str, Any]],
    evaluation_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "error_and_explainability_manifest.json"
    fn_path = out_root / "fn_examples.csv"
    fp_path = out_root / "fp_examples.csv"
    per_video_path = out_root / "per_video_error_summary.csv"
    rule_family_path = out_root / "rule_family_summary.csv"
    uncovered_path = out_root / "uncovered_positive_summary.csv"

    if not force_recompute and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _ANALYSIS_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {manifest_path.name}")
            return cached

    final_rules = list(final_rule_results.get("final_rules", []))
    per_video_metrics = {
        str(item.get("video_id", "")): dict(item)
        for item in list(evaluation_results.get("per_video_metrics", []))
    }

    example_rows: List[Dict[str, Any]] = []
    errors_by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    fn_rows: List[Dict[str, Any]] = []
    fp_rows: List[Dict[str, Any]] = []

    for example in _iter_eval_examples(temporal_rule_results):
        matching_rule_ids = _matching_rule_ids_for_example(example, final_rules)
        row = _build_example_feature_row(example=example, matching_rule_ids=matching_rule_ids, cfg=cfg)
        example_rows.append(row)
        is_error = bool(row["label"]) != bool(row["predicted_positive"])
        if is_error:
            errors_by_video[str(row["video_id"])].append(row)
        if bool(row["label"]) and not bool(row["predicted_positive"]):
            fn_rows.append(row)
        elif (not bool(row["label"])) and bool(row["predicted_positive"]):
            fp_rows.append(row)

    example_fieldnames = [
        "video_id",
        "example_id",
        "label",
        "predicted_positive",
        "num_matching_rules",
        "ego_forward_state",
        "ego_lateral_state",
        "ego_motion_state",
        "num_objects",
        "has_pedestrian",
        "has_vehicle",
        "has_traffic_light",
        "has_near_object",
        "has_center_object",
        "has_stable_visible_object",
        "nearest_object_class",
        "nearest_object_distance_state",
        "nearest_object_x_position_state",
        "explainability_level",
    ]
    _write_csv(fn_path, example_fieldnames, fn_rows)
    _write_csv(fp_path, example_fieldnames, fp_rows)

    per_video_rows: List[Dict[str, Any]] = []
    all_video_ids = sorted(
        set(per_video_metrics) | {str(row.get("video_id", "")) for row in example_rows if str(row.get("video_id", ""))}
    )
    for video_id in all_video_ids:
        metrics = per_video_metrics.get(video_id, {})
        error_rows = list(errors_by_video.get(video_id, []))
        explainability_counter = Counter(str(row.get("explainability_level", "unknown")) for row in error_rows)
        per_video_rows.append(
            {
                "video_id": video_id,
                "num_examples": int(metrics.get("num_examples", 0)),
                "true_positive": int(metrics.get("true_positive", 0)),
                "false_positive": int(metrics.get("false_positive", 0)),
                "false_negative": int(metrics.get("false_negative", 0)),
                "true_negative": int(metrics.get("true_negative", 0)),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "num_error_examples": len(error_rows),
                "num_fn_examples": sum(1 for row in error_rows if bool(row.get("label", False))),
                "num_fp_examples": sum(1 for row in error_rows if not bool(row.get("label", False))),
                "dominant_explainability_level": (
                    explainability_counter.most_common(1)[0][0] if explainability_counter else "no_errors"
                ),
            }
        )

    _write_csv(
        per_video_path,
        [
            "video_id",
            "num_examples",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "num_error_examples",
            "num_fn_examples",
            "num_fp_examples",
            "dominant_explainability_level",
        ],
        per_video_rows,
    )

    family_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for rule in list(evaluation_results.get("rule_evaluations", [])):
        predicates = _body_predicates(rule)
        key = (len(predicates), "|".join(predicates))
        family = family_map.setdefault(
            key,
            {
                "body_length": len(predicates),
                "predicate_signature": "|".join(predicates),
                "num_rules": 0,
                "sum_eval_precision": 0.0,
                "max_eval_precision": 0.0,
                "max_eval_recall": 0.0,
                "union_positive_example_ids": set(),
                "union_negative_example_ids": set(),
                "representative_clauses": [],
            },
        )
        family["num_rules"] += 1
        family["sum_eval_precision"] += float(rule.get("eval_precision", 0.0))
        family["max_eval_precision"] = max(family["max_eval_precision"], float(rule.get("eval_precision", 0.0)))
        family["max_eval_recall"] = max(family["max_eval_recall"], float(rule.get("eval_recall", 0.0)))
        family["union_positive_example_ids"].update(str(v) for v in rule.get("eval_positive_example_ids", []))
        family["union_negative_example_ids"].update(str(v) for v in rule.get("eval_negative_example_ids", []))
        if len(family["representative_clauses"]) < 3 and str(rule.get("clause", "")):
            family["representative_clauses"].append(str(rule.get("clause", "")))

    rule_family_rows: List[Dict[str, Any]] = []
    max_rules_in_family = 0
    for idx, family in enumerate(
        sorted(
            family_map.values(),
            key=lambda item: (
                -len(item["union_positive_example_ids"]),
                item["predicate_signature"],
            ),
        ),
        start=1,
    ):
        max_rules_in_family = max(max_rules_in_family, int(family["num_rules"]))
        rule_family_rows.append(
            {
                "family_id": f"family_{idx:03d}",
                "body_length": int(family["body_length"]),
                "predicate_signature": str(family["predicate_signature"]),
                "num_rules": int(family["num_rules"]),
                "avg_eval_precision": float(family["sum_eval_precision"] / max(1, family["num_rules"])),
                "max_eval_precision": float(family["max_eval_precision"]),
                "max_eval_recall": float(family["max_eval_recall"]),
                "covered_positive_examples": len(family["union_positive_example_ids"]),
                "covered_negative_examples": len(family["union_negative_example_ids"]),
                "representative_clauses": json.dumps(family["representative_clauses"]),
            }
        )

    _write_csv(
        rule_family_path,
        [
            "family_id",
            "body_length",
            "predicate_signature",
            "num_rules",
            "avg_eval_precision",
            "max_eval_precision",
            "max_eval_recall",
            "covered_positive_examples",
            "covered_negative_examples",
            "representative_clauses",
        ],
        rule_family_rows,
    )

    uncovered_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in fn_rows:
        key = (
            row["explainability_level"],
            row["ego_forward_state"],
            row["ego_lateral_state"],
            row["ego_motion_state"],
            row["has_pedestrian"],
            row["has_vehicle"],
            row["has_traffic_light"],
            row["has_near_object"],
            row["has_center_object"],
            row["has_stable_visible_object"],
            row["nearest_object_class"],
            row["nearest_object_distance_state"],
            row["nearest_object_x_position_state"],
        )
        group = uncovered_map.setdefault(
            key,
            {
                "num_examples": 0,
                "video_ids": set(),
                "example_ids": [],
                "num_objects_total": 0,
                "row": row,
            },
        )
        group["num_examples"] += 1
        group["video_ids"].add(str(row["video_id"]))
        group["example_ids"].append(str(row["example_id"]))
        group["num_objects_total"] += int(row["num_objects"])

    uncovered_rows: List[Dict[str, Any]] = []
    for idx, group in enumerate(
        sorted(uncovered_map.values(), key=lambda item: (-int(item["num_examples"]), str(item["row"]["ego_motion_state"]))),
        start=1,
    ):
        row = dict(group["row"])
        uncovered_rows.append(
            {
                "pattern_id": f"uncovered_{idx:03d}",
                "num_examples": int(group["num_examples"]),
                "num_videos": len(group["video_ids"]),
                "avg_num_objects": float(group["num_objects_total"] / max(1, group["num_examples"])),
                "ego_forward_state": row["ego_forward_state"],
                "ego_lateral_state": row["ego_lateral_state"],
                "ego_motion_state": row["ego_motion_state"],
                "has_pedestrian": row["has_pedestrian"],
                "has_vehicle": row["has_vehicle"],
                "has_traffic_light": row["has_traffic_light"],
                "has_near_object": row["has_near_object"],
                "has_center_object": row["has_center_object"],
                "has_stable_visible_object": row["has_stable_visible_object"],
                "nearest_object_class": row["nearest_object_class"],
                "nearest_object_distance_state": row["nearest_object_distance_state"],
                "nearest_object_x_position_state": row["nearest_object_x_position_state"],
                "explainability_level": row["explainability_level"],
                "example_ids": json.dumps(group["example_ids"][:50]),
            }
        )

    _write_csv(
        uncovered_path,
        [
            "pattern_id",
            "num_examples",
            "num_videos",
            "avg_num_objects",
            "ego_forward_state",
            "ego_lateral_state",
            "ego_motion_state",
            "has_pedestrian",
            "has_vehicle",
            "has_traffic_light",
            "has_near_object",
            "has_center_object",
            "has_stable_visible_object",
            "nearest_object_class",
            "nearest_object_distance_state",
            "nearest_object_x_position_state",
            "explainability_level",
            "example_ids",
        ],
        uncovered_rows,
    )

    manifest = {
        "version": _ANALYSIS_VERSION,
        "config": _cfg_key_subset(cfg),
        "rule_set_name": str(cfg.get("rule_set_name", "")),
        "num_rules": int(final_rule_results.get("num_final_rules", len(final_rules))),
        "num_fn_examples": len(fn_rows),
        "num_fp_examples": len(fp_rows),
        "num_videos": len(per_video_rows),
        "num_rule_families": len(rule_family_rows),
        "max_rules_in_family": int(max_rules_in_family),
        "redundancy_ratio": float(
            max(0.0, 1.0 - (len(rule_family_rows) / max(1, int(final_rule_results.get("num_final_rules", len(final_rules))))))
        ),
        "num_uncovered_positive_patterns": len(uncovered_rows),
        "fn_examples_path": str(fn_path),
        "fp_examples_path": str(fp_path),
        "per_video_error_summary_path": str(per_video_path),
        "rule_family_summary_path": str(rule_family_path),
        "uncovered_positive_summary_path": str(uncovered_path),
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(
        "  error_analysis: "
        f"fn={len(fn_rows)} | "
        f"fp={len(fp_rows)} | "
        f"videos={len(per_video_rows)} | "
        f"rule_families={len(rule_family_rows)} | "
        f"uncovered_patterns={len(uncovered_rows)}"
    )
    print(f"FN examples CSV written to {fn_path}")
    print(f"FP examples CSV written to {fp_path}")
    print(f"Per-video error summary CSV written to {per_video_path}")
    print(f"Rule family summary CSV written to {rule_family_path}")
    print(f"Uncovered positive summary CSV written to {uncovered_path}")
    print(f"Error analysis manifest written to {manifest_path}")
    return manifest


def run(
    final_rule_results: Dict[str, Any],
    temporal_rule_results: List[Dict[str, Any]],
    evaluation_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_analysis(
        final_rule_results=final_rule_results,
        temporal_rule_results=temporal_rule_results,
        evaluation_results=evaluation_results,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
