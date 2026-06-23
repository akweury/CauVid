"""
Define a manual task-level rule relevance prior for brake_next before detection.

Outputs:
    pipeline_output/00_driving_mini_background_rule_relevance_prior/
        background_rule_relevance_prior.json
"""

from __future__ import annotations

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


_PRIOR_VERSION = 1


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "00_driving_mini_background_rule_relevance_prior"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target_predicate": str(cfg.get("target_predicate", "brake_next")),
    }


def _default_prior_entries(target_predicate: str) -> List[Dict[str, Any]]:
    return [
        {
            "prior_id": "lead_vehicle",
            "display_name": "lead_vehicle_ahead",
            "target_predicate": target_predicate,
            "rule_relevant_object_classes": ["car", "truck", "bus", "motorcycle", "vehicle"],
            "rule_relevant_states": ["near", "centered", "slowing", "stopped", "lead_vehicle"],
            "rule_relevant_spatial_positions": ["front_center", "same_lane", "near_ahead"],
            "rule_relevant_temporal_patterns": ["persistent_near", "closing_in", "distance_decreasing", "queueing_ahead"],
            "tracking_priority": "highest",
            "candidate_ranking_priority": 1.0,
            "rationale": "A lead vehicle in the ego lane is the most common direct brake_next cause.",
        },
        {
            "prior_id": "pedestrian",
            "display_name": "pedestrian_near_path",
            "target_predicate": target_predicate,
            "rule_relevant_object_classes": ["pedestrian", "person", "rider"],
            "rule_relevant_states": ["near", "crossing", "approaching_path", "waiting_to_cross"],
            "rule_relevant_spatial_positions": ["front_left", "front_right", "crosswalk_edge", "ego_path_boundary"],
            "rule_relevant_temporal_patterns": ["entering_path", "persistent_near", "crossing_progress"],
            "tracking_priority": "high",
            "candidate_ranking_priority": 0.96,
            "rationale": "Pedestrians can trigger braking even when small, partially occluded, or briefly visible.",
        },
        {
            "prior_id": "cyclist",
            "display_name": "cyclist_or_rider_conflict",
            "target_predicate": target_predicate,
            "rule_relevant_object_classes": ["bicycle", "cyclist", "rider", "motorcycle"],
            "rule_relevant_states": ["near", "merging", "crossing", "sharing_lane"],
            "rule_relevant_spatial_positions": ["front_left", "front_right", "same_lane_margin", "bike_lane_edge"],
            "rule_relevant_temporal_patterns": ["merge_toward_lane", "crossing_progress", "persistent_near"],
            "tracking_priority": "high",
            "candidate_ranking_priority": 0.92,
            "rationale": "Cyclists and riders often explain braking through lateral conflict and thin-object visibility issues.",
        },
        {
            "prior_id": "traffic_light",
            "display_name": "traffic_light_control",
            "target_predicate": target_predicate,
            "rule_relevant_object_classes": ["traffic light"],
            "rule_relevant_states": ["red", "yellow", "relevant", "ego_lane_control"],
            "rule_relevant_spatial_positions": ["front_center", "overhead", "intersection_entry"],
            "rule_relevant_temporal_patterns": ["state_change_to_red", "persistent_relevant_control"],
            "tracking_priority": "high",
            "candidate_ranking_priority": 0.9,
            "rationale": "Traffic-light state and ego relevance matter for brake_next candidate ranking.",
        },
        {
            "prior_id": "stop_sign",
            "display_name": "stop_sign_control",
            "target_predicate": target_predicate,
            "rule_relevant_object_classes": ["stop sign"],
            "rule_relevant_states": ["relevant", "approaching_stop_line"],
            "rule_relevant_spatial_positions": ["front_right", "front_center", "intersection_entry"],
            "rule_relevant_temporal_patterns": ["persistent_relevant_control", "approaching_intersection"],
            "tracking_priority": "medium_high",
            "candidate_ranking_priority": 0.84,
            "rationale": "Stop signs are strong brake cues when spatially relevant to the ego path.",
        },
        {
            "prior_id": "obstacle",
            "display_name": "obstacle_or_blockage",
            "target_predicate": target_predicate,
            "rule_relevant_object_classes": ["barrier", "cone", "debris", "unknown", "stopped_vehicle"],
            "rule_relevant_states": ["near", "blocking_lane", "partially_occluded", "unexpected_static_object"],
            "rule_relevant_spatial_positions": ["front_center", "same_lane", "lane_boundary"],
            "rule_relevant_temporal_patterns": ["persistent_blockage", "sudden_visibility"],
            "tracking_priority": "medium_high",
            "candidate_ranking_priority": 0.8,
            "rationale": "Unexpected obstacles can explain braking even when detector confidence is weak or class labels are coarse.",
        },
        {
            "prior_id": "crosswalk",
            "display_name": "crosswalk_context",
            "target_predicate": target_predicate,
            "rule_relevant_object_classes": ["crosswalk"],
            "rule_relevant_states": ["approaching", "occupied", "yield_context"],
            "rule_relevant_spatial_positions": ["front_center", "intersection_entry", "ego_path"],
            "rule_relevant_temporal_patterns": ["approaching_crosswalk", "pedestrian_yield_context"],
            "tracking_priority": "medium",
            "candidate_ranking_priority": 0.74,
            "rationale": "Crosswalk context raises the ranking priority of nearby pedestrians and cyclists.",
        },
        {
            "prior_id": "intersection",
            "display_name": "intersection_context",
            "target_predicate": target_predicate,
            "rule_relevant_object_classes": ["intersection", "junction"],
            "rule_relevant_states": ["approaching", "entering", "conflict_zone"],
            "rule_relevant_spatial_positions": ["front_center", "multi_path_conflict_region"],
            "rule_relevant_temporal_patterns": ["approaching_intersection", "control_transition_zone"],
            "tracking_priority": "medium",
            "candidate_ranking_priority": 0.7,
            "rationale": "Intersection context helps prioritize rule-relevant objects and controls before direct evidence is complete.",
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

    json_path = out_root / "background_rule_relevance_prior.json"
    if not force_recompute and json_path.exists():
        with json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _PRIOR_VERSION and _cfg_key_subset(cached.get("config", {})) == _cfg_key_subset(cfg):
            print(f"Background rule relevance prior JSON written to {json_path}")
            return cached

    target_predicate = str(cfg.get("target_predicate", "brake_next"))
    entries = _default_prior_entries(target_predicate)
    summary = {
        "version": _PRIOR_VERSION,
        "config": _cfg_key_subset(cfg),
        "target_predicate": target_predicate,
        "prior_table_type": "manual_task_level_background_rule_relevance_prior",
        "usage_constraints": {
            "candidate_ranking_tracking_priority_prior_only": True,
            "not_object_fact": True,
            "not_logic_fact": True,
            "not_rule": True,
            "not_rule_label": True,
            "not_detection": True,
            "not_tracking_result": True,
            "not_training_label": True,
        },
        "intended_use": {
            "before_object_detection": True,
            "before_tracking": True,
            "rank_tracking_candidates": True,
            "surface_rule_relevant_entities": True,
        },
        "num_prior_entries": len(entries),
        "entries": entries,
        "output_paths": {
            "prior_json": str(json_path),
        },
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        "background_rule_relevance_prior: "
        f"target={target_predicate} | entries={len(entries)}"
    )
    print(f"Background rule relevance prior JSON written to {json_path}")
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
