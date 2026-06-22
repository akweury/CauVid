"""
Define a fixed background causal prior for brake_next as a perception re-check aid.

Outputs:
    pipeline_output/23a_driving_mini_background_causal_prior/
        background_causal_prior_summary.json
        background_causal_prior_table.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_PRIOR_VERSION = 1


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "23a_driving_mini_background_causal_prior"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target_predicate": str(cfg.get("target_predicate", "brake_next")),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _default_prior_entries(target_predicate: str) -> List[Dict[str, Any]]:
    return [
        {
            "prior_id": "lead_vehicle",
            "display_name": "lead_vehicle_ahead",
            "target_predicate": target_predicate,
            "prior_scope": "causal_object",
            "entity_kind": "object",
            "candidate_classes": ["car", "truck", "bus", "motorcycle", "vehicle"],
            "candidate_states": ["near", "centered", "slowing", "stopped", "blocking_lane"],
            "perception_recheck_focus": [
                "front_center_vehicle_presence",
                "distance_estimate",
                "lane_blocking_or_queue",
            ],
            "rationale": "A close lead vehicle is a common direct cause of braking.",
            "priority": 1.0,
        },
        {
            "prior_id": "pedestrian",
            "display_name": "pedestrian_crossing_or_near_path",
            "target_predicate": target_predicate,
            "prior_scope": "causal_object",
            "entity_kind": "object",
            "candidate_classes": ["pedestrian", "person", "rider"],
            "candidate_states": ["near", "crossing", "approaching_path", "occluding_lane_edge"],
            "perception_recheck_focus": [
                "crossing_intent",
                "near_road_boundary_presence",
                "small_object_visibility",
            ],
            "rationale": "Pedestrians near the ego path can explain conservative braking.",
            "priority": 0.95,
        },
        {
            "prior_id": "cyclist",
            "display_name": "cyclist_or_bicycle_conflict",
            "target_predicate": target_predicate,
            "prior_scope": "causal_object",
            "entity_kind": "object",
            "candidate_classes": ["bicycle", "cyclist", "rider", "motorcycle"],
            "candidate_states": ["near", "merging", "crossing", "sharing_lane"],
            "perception_recheck_focus": [
                "thin_object_recall",
                "rider_visibility",
                "path_conflict",
            ],
            "rationale": "Cyclists and riders often trigger braking through path conflict or proximity.",
            "priority": 0.9,
        },
        {
            "prior_id": "traffic_light",
            "display_name": "traffic_light_control",
            "target_predicate": target_predicate,
            "prior_scope": "causal_control",
            "entity_kind": "object",
            "candidate_classes": ["traffic_light"],
            "candidate_states": ["red", "yellow", "relevant", "front_center"],
            "perception_recheck_focus": [
                "detection_recall",
                "state_classification",
                "control_relevance",
                "temporal_alignment",
            ],
            "rationale": "Traffic lights can cause braking, but may appear earlier than the immediate brake_next target.",
            "priority": 0.9,
        },
        {
            "prior_id": "stop_sign",
            "display_name": "stop_sign_control",
            "target_predicate": target_predicate,
            "prior_scope": "causal_control",
            "entity_kind": "object",
            "candidate_classes": ["stop_sign"],
            "candidate_states": ["relevant", "front_center", "approaching_stop_line"],
            "perception_recheck_focus": [
                "detection_recall",
                "control_relevance",
                "distance_to_stop",
            ],
            "rationale": "Stop signs may explain braking if the ego vehicle is approaching a stop-controlled entry.",
            "priority": 0.82,
        },
        {
            "prior_id": "obstacle",
            "display_name": "generic_obstacle_or_blockage",
            "target_predicate": target_predicate,
            "prior_scope": "causal_object",
            "entity_kind": "object",
            "candidate_classes": ["barrier", "cone", "debris", "unknown", "stopped_vehicle"],
            "candidate_states": ["near", "blocking_lane", "partially_occluded"],
            "perception_recheck_focus": [
                "unexpected_static_object",
                "unknown_object_salience",
                "small_obstacle_recall",
            ],
            "rationale": "Braking can be caused by obstacles that are weakly labeled or collapsed into unknown classes.",
            "priority": 0.78,
        },
        {
            "prior_id": "crosswalk",
            "display_name": "crosswalk_context",
            "target_predicate": target_predicate,
            "prior_scope": "contextual_scene",
            "entity_kind": "scene_context",
            "candidate_classes": ["crosswalk"],
            "candidate_states": ["approaching", "occupied", "near_intersection"],
            "perception_recheck_focus": [
                "scene_layout",
                "painted_marking_visibility",
                "pedestrian_yield_context",
            ],
            "rationale": "Crosswalk context can explain braking even when a pedestrian is small or partially missed.",
            "priority": 0.72,
        },
        {
            "prior_id": "intersection",
            "display_name": "intersection_context",
            "target_predicate": target_predicate,
            "prior_scope": "contextual_scene",
            "entity_kind": "scene_context",
            "candidate_classes": ["intersection", "junction"],
            "candidate_states": ["approaching", "entering", "conflict_zone"],
            "perception_recheck_focus": [
                "road_topology",
                "conflict_region_awareness",
                "control_device_context",
            ],
            "rationale": "Intersections often induce braking through latent conflict not captured by a single object fact.",
            "priority": 0.7,
        },
    ]


def process_prior(
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)

    summary_json_path = out_root / "background_causal_prior_summary.json"
    table_csv_path = out_root / "background_causal_prior_table.csv"

    if not force_recompute and summary_json_path.exists():
        with summary_json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _PRIOR_VERSION and _cfg_key_subset(cached.get("config", {})) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {summary_json_path.name}")
            return cached

    target_predicate = str(cfg.get("target_predicate", "brake_next"))
    entries = _default_prior_entries(target_predicate)

    csv_rows: List[Dict[str, Any]] = []
    for entry in entries:
        csv_rows.append(
            {
                "prior_id": str(entry.get("prior_id", "")),
                "display_name": str(entry.get("display_name", "")),
                "target_predicate": str(entry.get("target_predicate", "")),
                "prior_scope": str(entry.get("prior_scope", "")),
                "entity_kind": str(entry.get("entity_kind", "")),
                "candidate_classes_json": json.dumps(list(entry.get("candidate_classes", []))),
                "candidate_states_json": json.dumps(list(entry.get("candidate_states", []))),
                "perception_recheck_focus_json": json.dumps(list(entry.get("perception_recheck_focus", []))),
                "priority": float(entry.get("priority", 0.0)),
                "rationale": str(entry.get("rationale", "")),
            }
        )

    _write_csv(
        table_csv_path,
        [
            "prior_id",
            "display_name",
            "target_predicate",
            "prior_scope",
            "entity_kind",
            "candidate_classes_json",
            "candidate_states_json",
            "perception_recheck_focus_json",
            "priority",
            "rationale",
        ],
        csv_rows,
    )

    summary = {
        "version": _PRIOR_VERSION,
        "config": _cfg_key_subset(cfg),
        "usage_constraints": {
            "perception_recheck_prior_only": True,
            "not_direct_rule": True,
            "not_object_fact": True,
            "not_training_label": True,
        },
        "target_predicate": target_predicate,
        "num_prior_entries": len(entries),
        "entries": entries,
        "output_paths": {
            "summary_json": str(summary_json_path),
            "table_csv": str(table_csv_path),
        },
    }
    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        "  background_causal_prior: "
        f"target={target_predicate} | "
        f"entries={len(entries)}"
    )
    print(f"Background causal prior summary JSON written to {summary_json_path}")
    print(f"Background causal prior table CSV written to {table_csv_path}")
    return summary


def run(
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_prior(
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
