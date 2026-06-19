"""
Trace object classes through the driving_mini pipeline from detections to rules.

Outputs:
    pipeline_output/18d_driving_mini_object_to_atom_coverage_diagnostic/
        object_to_atom_coverage_summary.json
        object_to_atom_coverage_summary.csv
        object_to_atom_stage_counts.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_DIAGNOSTIC_VERSION = 1
_DEFAULT_CLASS_ALIASES = {
    "traffic_light": "traffic_light",
    "traffic_lights": "traffic_light",
    "traffic_signal": "traffic_light",
    "traffic_signals": "traffic_light",
}
_SELECTOR_ORDER: Tuple[str, ...] = (
    "original",
    "diverse",
    "semantic_constrained_diverse",
    "coverage_family_aware",
)


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "18d_driving_mini_object_to_atom_coverage_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _normalize_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _normalized_alias_map(cfg: Dict[str, Any]) -> Dict[str, str]:
    alias_map = dict(_DEFAULT_CLASS_ALIASES)
    for raw_key, raw_value in dict(cfg.get("class_aliases", {})).items():
        alias_map[_normalize_token(raw_key)] = _normalize_token(raw_value)
    return {key: alias_map.get(value, value) for key, value in alias_map.items()}


def _normalize_class_name(value: Any, alias_map: Dict[str, str]) -> str:
    normalized = _normalize_token(value)
    return alias_map.get(normalized, normalized)


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    alias_map = _normalized_alias_map(cfg)
    return {
        "class_aliases": {str(key): str(value) for key, value in sorted(alias_map.items())},
        "max_rule_examples_per_class": int(cfg.get("max_rule_examples_per_class", 5)),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _extract_rule_classes(rule: Dict[str, Any], alias_map: Dict[str, str]) -> Set[str]:
    classes: Set[str] = set()
    for atom in _get_rule_body_atom_templates(rule):
        parsed = _parse_atom(atom)
        if parsed is None:
            continue
        predicate, args = parsed
        if predicate == "object_class" and len(args) >= 3:
            classes.add(_normalize_class_name(args[2], alias_map))
    return classes


def _extract_rule_classes_from_clause(clause: str, alias_map: Dict[str, str]) -> Set[str]:
    classes: Set[str] = set()
    for match in re.finditer(r"object_class\(([^)]*)\)", str(clause)):
        args = [part.strip() for part in match.group(1).split(",")]
        if len(args) >= 3:
            classes.add(_normalize_class_name(args[2], alias_map))
    return classes


def _make_class_entry() -> Dict[str, Any]:
    return {
        "raw_labels": set(),
        "step01_detection_instances": 0,
        "step03_annotation_instances": 0,
        "step03_annotation_track_ids": set(),
        "step04_merged_instances": 0,
        "step04_merged_track_ids": set(),
        "step10_important_object_instances": 0,
        "step10_important_object_track_ids": set(),
        "step10_important_object_segments": set(),
        "step11_logic_atom_instances": 0,
        "step11_logic_atom_track_ids": set(),
        "step11_logic_atom_segments": set(),
        "step16_rule_pool_rule_ids": set(),
        "step16_rule_pool_examples": [],
        "step17_selected_rule_ids": set(),
        "step17_selected_rule_examples": [],
        "step17b_selected_rule_ids": set(),
        "step17b_selected_rule_examples": [],
        "step17b2_selected_rule_ids": set(),
        "step17b2_selected_rule_examples": [],
        "step17c_selected_rule_ids": set(),
        "step17c_selected_rule_examples": [],
        "step18c_top_weight_rule_ids": set(),
        "step18c_top_weight_positive_rule_ids": set(),
        "step18c_top_weight_negative_rule_ids": set(),
        "step18c_top_weight_examples": [],
    }


def _append_unique_rule_example(
    examples: List[Dict[str, Any]],
    example: Dict[str, Any],
    max_examples: int,
) -> None:
    if len(examples) >= max_examples:
        return
    example_key = (str(example.get("rule_id", "")), str(example.get("clause", "")))
    if any((str(row.get("rule_id", "")), str(row.get("clause", ""))) == example_key for row in examples):
        return
    examples.append(example)


def _ensure_entry(class_map: Dict[str, Dict[str, Any]], class_name: str) -> Dict[str, Any]:
    if class_name not in class_map:
        class_map[class_name] = _make_class_entry()
    return class_map[class_name]


def _collect_detection_stage(
    class_map: Dict[str, Dict[str, Any]],
    detection_results: Sequence[Dict[str, Any]],
    alias_map: Dict[str, str],
) -> None:
    for video_result in detection_results:
        for frame in list(video_result.get("frames", [])):
            for raw_label in list(frame.get("labels", [])):
                class_name = _normalize_class_name(raw_label, alias_map)
                entry = _ensure_entry(class_map, class_name)
                entry["raw_labels"].add(str(raw_label))
                entry["step01_detection_instances"] += 1


def _collect_annotation_stage(
    class_map: Dict[str, Dict[str, Any]],
    dataset_annotation_results: Sequence[Dict[str, Any]],
    alias_map: Dict[str, str],
) -> None:
    for video_result in dataset_annotation_results:
        video_id = str(video_result.get("video_id", ""))
        for frame in list(video_result.get("frames", [])):
            labels = list(frame.get("labels", []))
            track_ids = list(frame.get("track_ids", []))
            for index, label in enumerate(labels):
                track_id = track_ids[index] if index < len(track_ids) else -1
                class_name = _normalize_class_name(label, alias_map)
                entry = _ensure_entry(class_map, class_name)
                entry["raw_labels"].add(str(label))
                entry["step03_annotation_instances"] += 1
                normalized_track_id = _safe_int(track_id, -1)
                if normalized_track_id >= 0:
                    entry["step03_annotation_track_ids"].add((video_id, normalized_track_id))


def _collect_merged_stage(
    class_map: Dict[str, Dict[str, Any]],
    merged_results: Sequence[Dict[str, Any]],
    alias_map: Dict[str, str],
) -> None:
    for video_result in merged_results:
        video_id = str(video_result.get("video_id", ""))
        for frame in list(video_result.get("frames", [])):
            labels = list(frame.get("labels", []))
            track_ids = list(frame.get("track_ids", []))
            for index, label in enumerate(labels):
                track_id = track_ids[index] if index < len(track_ids) else -1
                class_name = _normalize_class_name(label, alias_map)
                entry = _ensure_entry(class_map, class_name)
                entry["raw_labels"].add(str(label))
                entry["step04_merged_instances"] += 1
                normalized_track_id = _safe_int(track_id, -1)
                if normalized_track_id >= 0:
                    entry["step04_merged_track_ids"].add((video_id, normalized_track_id))


def _collect_important_object_stage(
    class_map: Dict[str, Dict[str, Any]],
    important_object_results: Sequence[Dict[str, Any]],
    alias_map: Dict[str, str],
) -> None:
    for video_result in important_object_results:
        video_id = str(video_result.get("video_id", ""))
        for segment in list(video_result.get("segments", [])):
            segment_index = _safe_int(segment.get("segment_index", -1))
            for obj in list(segment.get("selected_objects", [])):
                raw_label = str(obj.get("object_class", "unknown"))
                class_name = _normalize_class_name(raw_label, alias_map)
                entry = _ensure_entry(class_map, class_name)
                entry["raw_labels"].add(raw_label)
                entry["step10_important_object_instances"] += 1
                normalized_track_id = _safe_int(obj.get("track_id", -1), -1)
                if normalized_track_id >= 0:
                    entry["step10_important_object_track_ids"].add((video_id, normalized_track_id))
                entry["step10_important_object_segments"].add((video_id, segment_index))


def _collect_logic_atom_stage(
    class_map: Dict[str, Dict[str, Any]],
    logic_atom_results: Sequence[Dict[str, Any]],
    alias_map: Dict[str, str],
) -> None:
    for video_result in logic_atom_results:
        video_id = str(video_result.get("video_id", ""))
        for segment in list(video_result.get("segments", [])):
            segment_index = _safe_int(segment.get("segment_index", -1))
            for obj in list(segment.get("objects", [])):
                raw_label = str(obj.get("object_class", "unknown"))
                class_name = _normalize_class_name(raw_label, alias_map)
                entry = _ensure_entry(class_map, class_name)
                entry["raw_labels"].add(raw_label)
                entry["step11_logic_atom_instances"] += 1
                object_id = str(obj.get("object_id", ""))
                if object_id:
                    entry["step11_logic_atom_track_ids"].add((video_id, object_id))
                entry["step11_logic_atom_segments"].add((video_id, segment_index))


def _collect_rule_stage(
    class_map: Dict[str, Dict[str, Any]],
    stage_rule_key: str,
    stage_example_key: str,
    rules: Sequence[Dict[str, Any]],
    alias_map: Dict[str, str],
    max_rule_examples_per_class: int,
) -> None:
    for rule in rules:
        matched_classes = _extract_rule_classes(rule, alias_map)
        if not matched_classes:
            continue
        example = {
            "rule_id": str(rule.get("rule_id", "")),
            "clause": str(rule.get("clause", "")),
            "confidence": _safe_float(rule.get("confidence", 0.0)),
            "positive_support": _safe_int(rule.get("positive_support", 0)),
            "negative_support": _safe_int(rule.get("negative_support", 0)),
        }
        for class_name in matched_classes:
            entry = _ensure_entry(class_map, class_name)
            entry[stage_rule_key].add(str(rule.get("rule_id", "")))
            _append_unique_rule_example(entry[stage_example_key], example, max_rule_examples_per_class)


def _collect_learned_aggregation_stage(
    class_map: Dict[str, Dict[str, Any]],
    extended_rule_results: Dict[str, Any],
    rule_aggregation_baseline_results: Dict[str, Any],
    alias_map: Dict[str, str],
    max_rule_examples_per_class: int,
) -> None:
    rule_lookup = {
        str(rule.get("rule_id", "")): dict(rule)
        for rule in list(extended_rule_results.get("all_kept_rules", []))
        if str(rule.get("rule_id", ""))
    }
    top_weighted_rules = list(rule_aggregation_baseline_results.get("top_weighted_rules", []))
    for row in top_weighted_rules:
        rule_id = str(row.get("rule_id", ""))
        matched_classes = _extract_rule_classes(rule_lookup.get(rule_id, {}), alias_map)
        if not matched_classes:
            matched_classes = _extract_rule_classes_from_clause(str(row.get("clause", "")), alias_map)
        if not matched_classes:
            continue
        example = {
            "rank": _safe_int(row.get("rank", 0)),
            "rule_id": rule_id,
            "clause": str(row.get("clause", "")),
            "weight": _safe_float(row.get("weight", 0.0)),
            "sign": str(row.get("sign", "")),
            "confidence": _safe_float(row.get("confidence", 0.0)),
            "semantic_family": str(row.get("semantic_family", "")),
        }
        for class_name in matched_classes:
            entry = _ensure_entry(class_map, class_name)
            entry["step18c_top_weight_rule_ids"].add(rule_id)
            if example["weight"] >= 0.0:
                entry["step18c_top_weight_positive_rule_ids"].add(rule_id)
            else:
                entry["step18c_top_weight_negative_rule_ids"].add(rule_id)
            _append_unique_rule_example(entry["step18c_top_weight_examples"], example, max_rule_examples_per_class)


def _finalize_row(class_name: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "normalized_class_name": class_name,
        "raw_class_names": " || ".join(sorted(str(value) for value in entry["raw_labels"])),
        "step01_detection_instances": int(entry["step01_detection_instances"]),
        "step03_annotation_instances": int(entry["step03_annotation_instances"]),
        "step03_annotation_tracks": len(entry["step03_annotation_track_ids"]),
        "step04_merged_instances": int(entry["step04_merged_instances"]),
        "step04_merged_tracks": len(entry["step04_merged_track_ids"]),
        "step10_important_object_instances": int(entry["step10_important_object_instances"]),
        "step10_important_object_tracks": len(entry["step10_important_object_track_ids"]),
        "step10_important_object_segments": len(entry["step10_important_object_segments"]),
        "step11_logic_atom_instances": int(entry["step11_logic_atom_instances"]),
        "step11_logic_atom_tracks": len(entry["step11_logic_atom_track_ids"]),
        "step11_logic_atom_segments": len(entry["step11_logic_atom_segments"]),
        "is_converted_to_logic_atoms": bool(entry["step11_logic_atom_instances"] > 0),
        "step16_rule_pool_rule_count": len(entry["step16_rule_pool_rule_ids"]),
        "appears_in_step16_rule_pool": bool(entry["step16_rule_pool_rule_ids"]),
        "step17_original_rule_count": len(entry["step17_selected_rule_ids"]),
        "appears_in_step17_original_rules": bool(entry["step17_selected_rule_ids"]),
        "step17b_diverse_rule_count": len(entry["step17b_selected_rule_ids"]),
        "appears_in_step17b_diverse_rules": bool(entry["step17b_selected_rule_ids"]),
        "step17b2_semantic_constrained_rule_count": len(entry["step17b2_selected_rule_ids"]),
        "appears_in_step17b2_semantic_constrained_rules": bool(entry["step17b2_selected_rule_ids"]),
        "step17c_coverage_family_aware_rule_count": len(entry["step17c_selected_rule_ids"]),
        "appears_in_step17c_coverage_family_aware_rules": bool(entry["step17c_selected_rule_ids"]),
        "appears_in_any_selected_hard_or_rules": bool(
            entry["step17_selected_rule_ids"]
            or entry["step17b_selected_rule_ids"]
            or entry["step17b2_selected_rule_ids"]
            or entry["step17c_selected_rule_ids"]
        ),
        "step18c_top_weight_rule_count": len(entry["step18c_top_weight_rule_ids"]),
        "step18c_top_weight_positive_rule_count": len(entry["step18c_top_weight_positive_rule_ids"]),
        "step18c_top_weight_negative_rule_count": len(entry["step18c_top_weight_negative_rule_ids"]),
        "appears_in_step18c_top_weight_rules": bool(entry["step18c_top_weight_rule_ids"]),
    }


def _stage_count_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    for row in rows:
        class_name = str(row.get("normalized_class_name", ""))
        for stage_name, count_type, value in [
            ("step01_detection", "instances", row.get("step01_detection_instances", 0)),
            ("step03_dataset_annotations", "instances", row.get("step03_annotation_instances", 0)),
            ("step03_dataset_annotations", "tracks", row.get("step03_annotation_tracks", 0)),
            ("step04_merged_objects", "instances", row.get("step04_merged_instances", 0)),
            ("step04_merged_objects", "tracks", row.get("step04_merged_tracks", 0)),
            ("step10_important_objects", "instances", row.get("step10_important_object_instances", 0)),
            ("step10_important_objects", "tracks", row.get("step10_important_object_tracks", 0)),
            ("step10_important_objects", "segments", row.get("step10_important_object_segments", 0)),
            ("step11_logic_atoms", "instances", row.get("step11_logic_atom_instances", 0)),
            ("step11_logic_atoms", "tracks", row.get("step11_logic_atom_tracks", 0)),
            ("step11_logic_atoms", "segments", row.get("step11_logic_atom_segments", 0)),
            ("step16_rule_pool", "rules", row.get("step16_rule_pool_rule_count", 0)),
            ("step17_original", "rules", row.get("step17_original_rule_count", 0)),
            ("step17b_diverse", "rules", row.get("step17b_diverse_rule_count", 0)),
            ("step17b2_semantic_constrained", "rules", row.get("step17b2_semantic_constrained_rule_count", 0)),
            ("step17c_coverage_family_aware", "rules", row.get("step17c_coverage_family_aware_rule_count", 0)),
            ("step18c_top_weighted_learned_rules", "rules", row.get("step18c_top_weight_rule_count", 0)),
        ]:
            output_rows.append(
                {
                    "normalized_class_name": class_name,
                    "stage_name": stage_name,
                    "count_type": count_type,
                    "count": int(value),
                }
            )
    return output_rows


def process_diagnostic(
    detection_results: Sequence[Dict[str, Any]],
    dataset_annotation_results: Sequence[Dict[str, Any]],
    merged_results: Sequence[Dict[str, Any]],
    important_object_results: Sequence[Dict[str, Any]],
    logic_atom_results: Sequence[Dict[str, Any]],
    extended_rule_results: Dict[str, Any],
    original_final_rule_results: Dict[str, Any],
    diverse_final_rule_results: Dict[str, Any],
    semantic_constrained_diverse_final_rule_results: Optional[Dict[str, Any]],
    coverage_family_aware_final_rule_results: Optional[Dict[str, Any]],
    rule_aggregation_baseline_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    alias_map = _normalized_alias_map(cfg)
    max_rule_examples_per_class = max(1, int(cfg.get("max_rule_examples_per_class", 5)))

    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_json_path = out_root / "object_to_atom_coverage_summary.json"
    summary_csv_path = out_root / "object_to_atom_coverage_summary.csv"
    stage_counts_csv_path = out_root / "object_to_atom_stage_counts.csv"

    if not force_recompute and summary_json_path.exists():
        with summary_json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _DIAGNOSTIC_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {summary_json_path.name}")
            return cached

    class_map: Dict[str, Dict[str, Any]] = {}
    _collect_detection_stage(class_map, detection_results, alias_map)
    _collect_annotation_stage(class_map, dataset_annotation_results, alias_map)
    _collect_merged_stage(class_map, merged_results, alias_map)
    _collect_important_object_stage(class_map, important_object_results, alias_map)
    _collect_logic_atom_stage(class_map, logic_atom_results, alias_map)
    _collect_rule_stage(
        class_map,
        stage_rule_key="step16_rule_pool_rule_ids",
        stage_example_key="step16_rule_pool_examples",
        rules=list(extended_rule_results.get("all_kept_rules", [])),
        alias_map=alias_map,
        max_rule_examples_per_class=max_rule_examples_per_class,
    )
    _collect_rule_stage(
        class_map,
        stage_rule_key="step17_selected_rule_ids",
        stage_example_key="step17_selected_rule_examples",
        rules=list(original_final_rule_results.get("final_rules", [])),
        alias_map=alias_map,
        max_rule_examples_per_class=max_rule_examples_per_class,
    )
    _collect_rule_stage(
        class_map,
        stage_rule_key="step17b_selected_rule_ids",
        stage_example_key="step17b_selected_rule_examples",
        rules=list(diverse_final_rule_results.get("final_rules", [])),
        alias_map=alias_map,
        max_rule_examples_per_class=max_rule_examples_per_class,
    )
    _collect_rule_stage(
        class_map,
        stage_rule_key="step17b2_selected_rule_ids",
        stage_example_key="step17b2_selected_rule_examples",
        rules=list((semantic_constrained_diverse_final_rule_results or {}).get("final_rules", [])),
        alias_map=alias_map,
        max_rule_examples_per_class=max_rule_examples_per_class,
    )
    _collect_rule_stage(
        class_map,
        stage_rule_key="step17c_selected_rule_ids",
        stage_example_key="step17c_selected_rule_examples",
        rules=list((coverage_family_aware_final_rule_results or {}).get("final_rules", [])),
        alias_map=alias_map,
        max_rule_examples_per_class=max_rule_examples_per_class,
    )
    _collect_learned_aggregation_stage(
        class_map=class_map,
        extended_rule_results=extended_rule_results,
        rule_aggregation_baseline_results=rule_aggregation_baseline_results,
        alias_map=alias_map,
        max_rule_examples_per_class=max_rule_examples_per_class,
    )

    summary_rows = [_finalize_row(class_name, class_map[class_name]) for class_name in sorted(class_map)]
    stage_count_rows = _stage_count_rows(summary_rows)
    detail_rows = {
        class_name: {
            "raw_labels": sorted(str(value) for value in entry["raw_labels"]),
            "step16_rule_pool_examples": list(entry["step16_rule_pool_examples"]),
            "step17_selected_rule_examples": list(entry["step17_selected_rule_examples"]),
            "step17b_selected_rule_examples": list(entry["step17b_selected_rule_examples"]),
            "step17b2_selected_rule_examples": list(entry["step17b2_selected_rule_examples"]),
            "step17c_selected_rule_examples": list(entry["step17c_selected_rule_examples"]),
            "step18c_top_weight_examples": list(entry["step18c_top_weight_examples"]),
        }
        for class_name, entry in sorted(class_map.items())
    }

    _write_csv(
        summary_csv_path,
        [
            "normalized_class_name",
            "raw_class_names",
            "step01_detection_instances",
            "step03_annotation_instances",
            "step03_annotation_tracks",
            "step04_merged_instances",
            "step04_merged_tracks",
            "step10_important_object_instances",
            "step10_important_object_tracks",
            "step10_important_object_segments",
            "step11_logic_atom_instances",
            "step11_logic_atom_tracks",
            "step11_logic_atom_segments",
            "is_converted_to_logic_atoms",
            "step16_rule_pool_rule_count",
            "appears_in_step16_rule_pool",
            "step17_original_rule_count",
            "appears_in_step17_original_rules",
            "step17b_diverse_rule_count",
            "appears_in_step17b_diverse_rules",
            "step17b2_semantic_constrained_rule_count",
            "appears_in_step17b2_semantic_constrained_rules",
            "step17c_coverage_family_aware_rule_count",
            "appears_in_step17c_coverage_family_aware_rules",
            "appears_in_any_selected_hard_or_rules",
            "step18c_top_weight_rule_count",
            "step18c_top_weight_positive_rule_count",
            "step18c_top_weight_negative_rule_count",
            "appears_in_step18c_top_weight_rules",
        ],
        summary_rows,
    )
    _write_csv(
        stage_counts_csv_path,
        ["normalized_class_name", "stage_name", "count_type", "count"],
        stage_count_rows,
    )

    summary = {
        "version": _DIAGNOSTIC_VERSION,
        "config": _cfg_key_subset(cfg),
        "num_classes": len(summary_rows),
        "rows": summary_rows,
        "details_by_class": detail_rows,
        "output_paths": {
            "summary_json": str(summary_json_path),
            "summary_csv": str(summary_csv_path),
            "stage_counts_csv": str(stage_counts_csv_path),
        },
    }
    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        "  object_to_atom_coverage_diagnostic: "
        f"classes={len(summary_rows)} | "
        f"logic_atom_classes={sum(1 for row in summary_rows if bool(row.get('is_converted_to_logic_atoms', False)))} | "
        f"rule_pool_classes={sum(1 for row in summary_rows if bool(row.get('appears_in_step16_rule_pool', False)))}"
    )
    print(f"Object-to-atom coverage summary JSON written to {summary_json_path}")
    print(f"Object-to-atom coverage summary CSV written to {summary_csv_path}")
    print(f"Object-to-atom stage-counts CSV written to {stage_counts_csv_path}")
    return summary


def run(
    detection_results: Sequence[Dict[str, Any]],
    dataset_annotation_results: Sequence[Dict[str, Any]],
    merged_results: Sequence[Dict[str, Any]],
    important_object_results: Sequence[Dict[str, Any]],
    logic_atom_results: Sequence[Dict[str, Any]],
    extended_rule_results: Dict[str, Any],
    original_final_rule_results: Dict[str, Any],
    diverse_final_rule_results: Dict[str, Any],
    semantic_constrained_diverse_final_rule_results: Optional[Dict[str, Any]],
    coverage_family_aware_final_rule_results: Optional[Dict[str, Any]],
    rule_aggregation_baseline_results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_diagnostic(
        detection_results=detection_results,
        dataset_annotation_results=dataset_annotation_results,
        merged_results=merged_results,
        important_object_results=important_object_results,
        logic_atom_results=logic_atom_results,
        extended_rule_results=extended_rule_results,
        original_final_rule_results=original_final_rule_results,
        diverse_final_rule_results=diverse_final_rule_results,
        semantic_constrained_diverse_final_rule_results=semantic_constrained_diverse_final_rule_results,
        coverage_family_aware_final_rule_results=coverage_family_aware_final_rule_results,
        rule_aggregation_baseline_results=rule_aggregation_baseline_results,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
