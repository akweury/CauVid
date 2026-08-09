"""Coherent Paper-1 modules assembled from the existing July implementation.

This file intentionally contains orchestration adapters, not new perception
algorithms.  Keeping the imports here makes the provenance of every August
module explicit and leaves ``exp_july`` unchanged and reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

State = Dict[str, Any]


def _july():
    """Import the heavy perception runtime only when a stage is executed."""
    from src.exp_july import perception

    return perception


def _driving_config():
    from src.exp_driving_videos import pipeline_config

    return pipeline_config


def _stage(state: State, number: int, name: str, origins: Iterable[str]) -> State:
    """Attach non-destructive module lineage while retaining all prior data."""
    history = list(state.get("august_stage_lineage", []))
    history.append(
        {
            "august_step": number,
            "module": name,
            "origin": list(origins),
        }
    )
    return {**state, "august_stage_lineage": history}


def dataset_initialization(video_ids=None, video_count=None) -> State:
    # August Step 1 originates from July Step 1 and its pre-7A split helper.
    july = _july()
    state = july.step1_init(video_ids=video_ids, video_count=video_count)
    state = july.step7_train_eval_split(state)
    return _stage(state, 1, "dataset_initialization", ("july:01", "july:07_train_eval_split"))


def object_detection(state: State) -> State:
    # August Step 2 is July Step 2 unchanged.
    result = _july().step2_detection(state, state["detection_args"])
    return _stage({**state, **result}, 2, "object_detection", ("july:02",))


def object_tracking(state: State) -> State:
    # August Step 3 is July Step 3 unchanged.
    result = _july().step3_tracking(state)
    return _stage({**state, **result}, 3, "object_tracking", ("july:03",))


def trajectory_construction_3d(state: State) -> State:
    # August Step 4 is the depth/geometry implementation from July Step 6.
    result = _july().step6_positions_3d(state)
    return _stage({**state, **result}, 4, "trajectory_construction_3d", ("july:06",))


def ego_motion_abstraction(
    state: State, *, render_candidate_filter_comparisons: bool = False
) -> State:
    """Run July 7A and 7B as one active, explicit ego-motion module."""
    # August Step 5 originates from July Steps 7A and 7B.  There is no empty
    # Step 7 sentinel in this pipeline.
    july = _july()
    axis_state = july.step7a_axis_threshold_segmentation(
        state,
        render_candidate_filter_comparisons=render_candidate_filter_comparisons,
    )
    selected = july.step7b_optimal_segmentation_selection(axis_state)
    return _stage(
        {**state, **selected, "ego_motion_module_status": "completed"},
        5,
        "ego_motion_abstraction",
        ("july:07a_axis_threshold_segmentation", "july:07b_optimal_segmentation_selection"),
    )


def trajectory_refinement(state: State, *, diagnostics: bool = False) -> State:
    """Run the July repair loop as one coherent August refinement module.

    Relative motion is computed internally because July's uncertain-signal,
    clustering, and closed-loop repair algorithms consume it.  The final
    repaired relative representation is exposed by August Step 7.
    """
    # August Step 6 originates from July Step 8 and internal Steps 8A-8G/8K.
    july = _july()
    refined = july.step8_trajectory_repair(state, state)
    refined = july.step8a_relative_object_motion(state, refined)
    refined = july.step8b_signal_evidence(refined)
    refined = july.step8c_trajectory_clustering(refined)
    refined = july.step8d_closed_loop_trajectory_repair(refined)
    refined = july.step8e_repaired_trajectory_validation(refined)
    refined = july.step8f_trajectory_statistics(refined)
    refined = july.step8g_repaired_track_materialization(refined)

    # July 8H-8J are diagnostics, not public August model stages.
    if diagnostics:
        refined = july.step8h_trajectory_repair_visualization(refined)
        refined = july.step8i_trajectory_audit_dashboard(refined)
        refined = july.step8j_trajectory_provenance_audit(refined)

    refined = july.step8k_trajectory_handoff(refined)
    return _stage(
        {**state, **refined},
        6,
        "trajectory_refinement",
        (
            "july:08_trajectory_repair",
            "july:08b_uncertain_signal_evidence",
            "july:08c_trajectory_clustering",
            "july:08d_closed_loop_trajectory_repair",
            "july:08e_repaired_trajectory_validation",
            "july:08f_trajectory_statistics",
            "july:08g_repaired_track_materialization",
        ),
    )


def relative_motion_representation(state: State) -> State:
    # August Step 7 exposes July Step 8A's post-repair representation.  Step 8A
    # ran inside Step 6 because it is also an input to July's repair loop.
    if not state.get("relative_object_motion"):
        raise RuntimeError("trajectory refinement produced no relative object motion")
    return _stage(
        state,
        7,
        "relative_motion_representation",
        ("july:08a_relative_object_motion", "july:08k_trajectory_handoff"),
    )


def temporal_video_segmentation(state: State, output_root: Path) -> State:
    # August Step 8 fulfills the July Step 9 contract with the existing driving
    # video segmentation implementation (July's current wrapper is a stub).
    from src.exp_driving_videos.modules import temporal_segmentation_driving_mini

    cfg = dict(_driving_config().get_temporal_segmentation_cfg())
    cfg["render_videos"] = False
    results = temporal_segmentation_driving_mini.run(
        ego_motion_results=list(state.get("ego_motion", [])),
        relative_motion_results=list(state.get("relative_object_motion", [])),
        seg_cfg=cfg,
        output_root=output_root / "08_temporal_video_segmentation",
    )
    result = {**state, "temporal_segments": results, "temporal_segmentation_results": results}
    return _stage(result, 8, "temporal_video_segmentation", ("july:09_contract", "driving:temporal_segmentation",))


def segment_motion_abstraction(state: State, output_root: Path) -> State:
    # August Step 9 fulfills the July Step 10 contract using its existing source
    # implementation and keeps per-object confidence/provenance dictionaries.
    from src.exp_driving_videos.modules import segment_object_motion_driving_mini

    cfg = dict(_driving_config().get_segment_object_motion_cfg())
    cfg["render_videos"] = False
    results = segment_object_motion_driving_mini.run(
        relative_motion_results=list(state.get("relative_object_motion", [])),
        temporal_segmentation_results=list(state.get("temporal_segments", [])),
        cfg=cfg,
        output_root=output_root / "09_segment_motion_abstraction",
    )
    result = {**state, "segment_object_motion": results}
    return _stage(result, 9, "segment_motion_abstraction", ("july:10_contract", "driving:segment_object_motion",))


def important_object_selection(state: State, output_root: Path) -> State:
    # August Step 10 replaces July Step 11's visualization-only wrapper with the
    # existing selection implementation.  Its current configured behavior is
    # preserved (including pass-through selection when configured).
    from src.exp_driving_videos.modules import important_objects_driving_mini

    results = important_objects_driving_mini.run(
        segment_object_motion_results=list(state.get("segment_object_motion", [])),
        cfg=_driving_config().get_important_objects_cfg(),
        output_root=output_root / "10_important_object_selection",
    )
    result = {**state, "important_objects": results}
    return _stage(result, 10, "important_object_selection", ("july:11_contract", "driving:important_objects",))


def _collect_field_names(value: Any, predicate, names: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if predicate(str(key).lower()):
                names.add(str(key))
            _collect_field_names(child, predicate, names)
    elif isinstance(value, list):
        for child in value:
            _collect_field_names(child, predicate, names)


def _video_ids(rows: Any) -> list[str]:
    return sorted(
        {str(row.get("video_id")) for row in rows or [] if isinstance(row, dict) and row.get("video_id")}
    )


def _video_row(rows: Any, video_id: str) -> State:
    return next(
        (
            row
            for row in rows or []
            if isinstance(row, dict) and str(row.get("video_id")) == str(video_id)
        ),
        {},
    )


def _collect_values(value: Any, field_name: str, values: set[str]) -> None:
    if isinstance(value, dict):
        if field_name in value and value[field_name] is not None:
            values.add(str(value[field_name]))
        for child in value.values():
            _collect_values(child, field_name, values)
    elif isinstance(value, list):
        for child in value:
            _collect_values(child, field_name, values)


def _traceability_manifest(state: State) -> State:
    confidence_fields: set[str] = set()
    provenance_fields: set[str] = set()
    _collect_field_names(state, lambda key: "confidence" in key or "score" in key, confidence_fields)
    _collect_field_names(state, lambda key: "provenance" in key or key.startswith("source"), provenance_fields)
    symbolic = list(state.get("symbolic_scene_representation", []))
    videos = sorted(set(state.get("videos", [])) | set(_video_ids(symbolic)))
    per_video = []
    for video_id in videos:
        detection_video = _video_row(state.get("detections"), video_id)
        tracking_video = _video_row(state.get("tracks"), video_id)
        position_video = _video_row(state.get("positions_3d"), video_id)
        relative_video = _video_row(state.get("relative_object_motion"), video_id)
        segment_video = _video_row(state.get("temporal_segments"), video_id)
        symbolic_video = _video_row(symbolic, video_id)
        track_ids: set[str] = set()
        _collect_values(tracking_video, "track_id", track_ids)
        _collect_values(position_video, "track_id", track_ids)
        _collect_values(relative_video, "track_id", track_ids)
        segment_ids: set[str] = set()
        _collect_values(segment_video, "segment_id", segment_ids)
        _collect_values(symbolic_video, "segment_id", segment_ids)
        per_video.append(
            {
                "video_id": str(video_id),
                "chain": [
                    "video_observation",
                    "detection",
                    "track",
                    "trajectory_3d",
                    "repaired_trajectory",
                    "motion_state",
                    "temporal_segment",
                    "symbolic_representation",
                ],
                "join_keys": ["video_id", "frame_index", "track_id", "segment_id"],
                "artifacts": {
                    "detections": int(detection_video.get("num_detections", 0) or 0),
                    "tracks": int(tracking_video.get("num_tracks", len(track_ids)) or 0),
                    "trajectory_3d_observations": int(position_video.get("num_objects_with_3d", 0) or 0),
                    "repaired_relative_observations": int(relative_video.get("num_objects_total", 0) or 0),
                    "track_ids": sorted(track_ids),
                    "segment_ids": sorted(segment_ids),
                },
                "symbolic_segments": int(symbolic_video.get("num_segments", len(symbolic_video.get("segments", []))) or 0),
                "symbolic_atoms": int(symbolic_video.get("num_atoms", 0) or 0),
            }
        )
    return {
        "schema_version": 1,
        "pipeline": "exp_august",
        "stage_lineage": list(state.get("august_stage_lineage", [])),
        "preserved_confidence_fields": sorted(confidence_fields),
        "preserved_provenance_fields": sorted(provenance_fields),
        "artifact_graph": [
            {"node": "video_observation", "state_field": "videos", "next": "detection"},
            {"node": "detection", "state_field": "detections", "next": "track"},
            {"node": "track", "state_field": "tracks", "next": "trajectory_3d"},
            {"node": "trajectory_3d", "state_field": "positions_3d", "next": "repaired_trajectory"},
            {"node": "repaired_trajectory", "state_field": "trajectory_pattern_records", "next": "motion_state"},
            {"node": "motion_state", "state_field": "relative_object_motion", "next": "temporal_segment"},
            {"node": "temporal_segment", "state_field": "temporal_segments", "next": "symbolic_representation"},
            {"node": "symbolic_representation", "state_field": "symbolic_scene_representation", "next": None},
        ],
        "video_lineage": per_video,
        "evaluation_outputs": {
            "temporal_segmentation": "temporal_segments",
            "symbolic_representation": "symbolic_scene_representation",
            "ego_motion": "ego_motion",
            "object_motion": "segment_object_motion",
        },
    }


def symbolic_scene_representation(state: State, output_root: Path) -> State:
    # August Step 11 fulfills the July Step 12 contract with the existing logic
    # atom materializer.  No target heads or rule/causal modules are imported.
    from src.exp_driving_videos.modules import logic_atoms_driving_mini

    results = logic_atoms_driving_mini.run(
        segment_object_motion_results=list(state.get("important_objects", [])),
        cfg=_driving_config().get_logic_atoms_cfg(),
        output_root=output_root / "11_symbolic_scene_representation",
    )
    result = _stage(
        {**state, "logic_atoms": results, "symbolic_scene_representation": results},
        11,
        "symbolic_scene_representation",
        ("july:12_contract", "driving:logic_atoms"),
    )
    manifest = _traceability_manifest(result)
    path = output_root / "exp_august_traceability.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {**result, "traceability": manifest, "traceability_path": str(path)}
