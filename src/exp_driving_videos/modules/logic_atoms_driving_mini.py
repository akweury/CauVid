"""
Convert filtered segment-level symbolic facts into logic atoms for driving_mini videos.

Consumes:
  - Step 10 output: per-segment important-object selection results

Output layout:
    pipeline_output/11_driving_mini_logic_atoms/
        logic_atoms_manifest.json
        <video_id>/
            logic_atoms.json
"""

from __future__ import annotations

import csv
import json
import re
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


_LOGIC_ATOMS_VERSION = 6

_CANDIDATE_SEMANTIC_RULE_SEARCH_PREDICATES = {
    "object_class",
    "object_vz_state",
    "object_vx_state",
    "object_speed_state",
    "object_distance_state",
    "object_visibility_state",
    "object_x_position_state",
    "traffic_control_type",
    "traffic_control_relevance_state",
    "traffic_control_front_center_region",
    "traffic_control_relevant",
    "traffic_light_relevant",
    "stop_sign_relevant",
    "traffic_light_position_state",
    "traffic_light_state",
}
_CANDIDATE_QUALITY_AUXILIARY_PREDICATES = {
    "object_prior_relevance_state",
}
_CANDIDATE_PROVENANCE_METADATA_PREDICATES = {
    "object_is_candidate",
    "object_source_type",
    "object_candidate_score_state",
    "object_matched_prior",
}


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "11_driving_mini_logic_atoms"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _sym(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "unknown"
    if text[0].isdigit():
        text = f"n_{text}"
    return text


def _seg_id(segment_index: int) -> str:
    return f"seg_{int(segment_index):04d}"


def _obj_id(track_id: int) -> str:
    return f"obj_{int(track_id):04d}"


def _candidate_obj_id(candidate_object: Dict[str, Any]) -> str:
    candidate_object_id = str(candidate_object.get("candidate_object_id", "")).strip()
    if candidate_object_id:
        return f"obj_candidate_{_sym(candidate_object_id)}"
    candidate_track_id = int(candidate_object.get("candidate_track_id", candidate_object.get("track_id", -1)))
    return f"obj_candidate_track_{candidate_track_id:04d}"


def _score_state(value: float, high_threshold: float = 0.67, medium_threshold: float = 0.34) -> str:
    score = float(value)
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def _matched_prior_ids(obj: Dict[str, Any]) -> List[str]:
    prior_metadata = dict(obj.get("prior_metadata", {}))
    values = {
        str(value).strip()
        for value in list(prior_metadata.get("matched_prior_ids", []))
        + list(prior_metadata.get("track_matched_prior_ids", []))
        if str(value).strip()
    }
    return sorted(values)


def _prior_relevance_score(obj: Dict[str, Any]) -> float:
    prior_metadata = dict(obj.get("prior_metadata", {}))
    candidates = [
        prior_metadata.get("prior_relevance_score", 0.0),
        prior_metadata.get("prior_relevance_mean", 0.0),
        prior_metadata.get("prior_relevance_max", 0.0),
    ]
    return max(float(value) for value in candidates)


def _make_atom(predicate: str, *args: Any) -> str:
    rendered_args = ",".join(_sym(arg) for arg in args)
    return f"{_sym(predicate)}({rendered_args})."


def _make_record(predicate: str, args: List[Any], kind: str) -> Dict[str, Any]:
    return {
        "predicate": _sym(predicate),
        "args": [_sym(arg) for arg in args],
        "kind": kind,
        "atom": _make_atom(predicate, *args),
    }


def _candidate_atom_category(predicate: str) -> str:
    predicate_text = _sym(predicate)
    if predicate_text in _CANDIDATE_SEMANTIC_RULE_SEARCH_PREDICATES:
        return "semantic_rule_search"
    if predicate_text in _CANDIDATE_QUALITY_AUXILIARY_PREDICATES:
        return "quality_auxiliary"
    if predicate_text in _CANDIDATE_PROVENANCE_METADATA_PREDICATES:
        return "provenance_metadata"
    return "quality_auxiliary"


def _visibility_state(visibility_ratio: float, persistent_threshold: float, present_threshold: float) -> str:
    ratio = float(visibility_ratio)
    if ratio >= persistent_threshold:
        return "persistent"
    if ratio >= present_threshold:
        return "intermittent"
    return "brief"


def _position_x_state(mean_x: float, threshold: float) -> str:
    x = float(mean_x)
    if x < -threshold:
        return "left_of_ego"
    if x > threshold:
        return "right_of_ego"
    return "centered"


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "lateral_position_threshold",
        "visibility_persistent_threshold",
        "visibility_present_threshold",
        "include_segment_boundary_atoms",
        "include_object_identity_atoms",
        "include_traffic_control_atoms",
        "traffic_control_relevance_threshold",
        "include_unknown_traffic_light_state_atoms",
    ]
    return {k: cfg.get(k) for k in keys}


def _print_video_summary(result: Dict[str, Any]) -> None:
    video_id = str(result.get("video_id", "unknown"))
    num_objects = int(result.get("num_objects", 0))
    num_candidate_objects = int(result.get("num_candidate_objects", 0))
    object_types = [str(v) for v in result.get("object_types", [])]
    num_atoms = int(result.get("num_atoms", 0))
    object_types_text = ", ".join(object_types) if object_types else "none"
    print(
        f"  {video_id}: objects={num_objects} | "
        f"candidate_objects={num_candidate_objects} | "
        f"types=[{object_types_text}] | atoms={num_atoms}"
    )


def process_video(
    segment_object_motion_video_result: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    lateral_position_threshold = float(cfg.get("lateral_position_threshold", 2.0))
    visibility_persistent_threshold = float(cfg.get("visibility_persistent_threshold", 0.8))
    visibility_present_threshold = float(cfg.get("visibility_present_threshold", 0.3))
    include_segment_boundary_atoms = bool(cfg.get("include_segment_boundary_atoms", True))
    include_object_identity_atoms = bool(cfg.get("include_object_identity_atoms", True))
    include_traffic_control_atoms = bool(cfg.get("include_traffic_control_atoms", True))
    traffic_control_relevance_threshold = float(cfg.get("traffic_control_relevance_threshold", 0.4))
    include_unknown_traffic_light_state_atoms = bool(
        cfg.get("include_unknown_traffic_light_state_atoms", False)
    )

    video_id = str(segment_object_motion_video_result["video_id"])
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "logic_atoms.json"
    csv_file = out_dir / "logic_atoms.csv"

    if not force_recompute and out_file.exists():
        with out_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        cache_cfg = cached.get("config", {}) if isinstance(cached, dict) else {}
        if (
            int(cached.get("version", 0)) == _LOGIC_ATOMS_VERSION
            and _cfg_key_subset(cache_cfg) == _cfg_key_subset(
                {
                    "lateral_position_threshold": lateral_position_threshold,
                    "visibility_persistent_threshold": visibility_persistent_threshold,
                    "visibility_present_threshold": visibility_present_threshold,
                    "include_segment_boundary_atoms": include_segment_boundary_atoms,
                    "include_object_identity_atoms": include_object_identity_atoms,
                    "include_traffic_control_atoms": include_traffic_control_atoms,
                    "traffic_control_relevance_threshold": traffic_control_relevance_threshold,
                    "include_unknown_traffic_light_state_atoms": include_unknown_traffic_light_state_atoms,
                }
            )
        ):
            print(f"  [cache] {video_id} - loading {out_file.name}")
            _print_video_summary(cached)
            return cached

    segments_out: List[Dict[str, Any]] = []
    all_atom_records: List[Dict[str, Any]] = []
    all_atoms: List[str] = []

    for segment in segment_object_motion_video_result.get("segments", []):
        segment_index = int(segment.get("segment_index", 0))
        segment_id = _seg_id(segment_index)
        start_frame = int(segment.get("start_frame", 0))
        end_frame = int(segment.get("end_frame", start_frame))
        segment_label = str(segment.get("segment_label", "unknown"))
        forward_label = str(segment.get("segment_forward_label", "unknown"))
        lateral_label = str(segment.get("segment_lateral_label", "unknown"))

        segment_atom_records: List[Dict[str, Any]] = []
        segment_atoms: List[str] = []

        def _append_segment_atom(predicate: str, *args: Any) -> None:
            rec = _make_record(predicate, list(args), kind="segment")
            segment_atom_records.append(rec)
            segment_atoms.append(rec["atom"])
            all_atom_records.append(rec)
            all_atoms.append(rec["atom"])

        _append_segment_atom("segment", segment_id)
        _append_segment_atom("segment_forward_state", segment_id, forward_label)
        _append_segment_atom("segment_lateral_state", segment_id, lateral_label)
        _append_segment_atom("segment_motion_state", segment_id, segment_label)
        if include_segment_boundary_atoms:
            _append_segment_atom("segment_start_frame", segment_id, f"frame_{start_frame}")
            _append_segment_atom("segment_end_frame", segment_id, f"frame_{end_frame}")

        selected_objects = list(segment.get("selected_objects", segment.get("objects", [])))
        selected_candidate_objects = list(
            segment.get("selected_candidate_objects", segment.get("candidate_objects", []))
        )
        objects_out: List[Dict[str, Any]] = []
        candidate_objects_out: List[Dict[str, Any]] = []

        def _emit_object(obj: Dict[str, Any], *, is_candidate: bool) -> Dict[str, Any]:
            track_id = int(obj.get("track_id", -1))
            candidate_track_id = int(obj.get("candidate_track_id", track_id))
            object_id = _candidate_obj_id(obj) if is_candidate else _obj_id(track_id)
            object_class = str(obj.get("object_class", "unknown"))
            vz_state = str(obj.get("vz_state", "vz_unknown"))
            vx_state = str(obj.get("vx_state", "vx_unknown"))
            speed_state = str(obj.get("speed_state", "rel_static"))
            distance_state = str(obj.get("distance_state", "unknown"))
            visibility_state = _visibility_state(
                visibility_ratio=float(obj.get("visibility_ratio", 0.0)),
                persistent_threshold=visibility_persistent_threshold,
                present_threshold=visibility_present_threshold,
            )
            mean_position = obj.get("mean_position_3d", [0.0, 0.0, 0.0])
            x_position_state = _position_x_state(
                mean_x=float(mean_position[0]) if len(mean_position) > 0 else 0.0,
                threshold=lateral_position_threshold,
            )

            object_atom_records: List[Dict[str, Any]] = []
            object_atoms: List[str] = []
            candidate_semantic_rule_search_atoms: List[str] = []
            candidate_quality_auxiliary_atoms: List[str] = []
            candidate_provenance_metadata_atoms: List[str] = []

            def _append_object_atom(predicate: str, *args: Any) -> None:
                rec = _make_record(predicate, list(args), kind="object")
                if is_candidate:
                    atom_category = _candidate_atom_category(predicate)
                    rec["candidate_atom_category"] = atom_category
                else:
                    atom_category = ""
                object_atom_records.append(rec)
                object_atoms.append(rec["atom"])
                if is_candidate:
                    if atom_category == "semantic_rule_search":
                        candidate_semantic_rule_search_atoms.append(rec["atom"])
                    elif atom_category == "quality_auxiliary":
                        candidate_quality_auxiliary_atoms.append(rec["atom"])
                    elif atom_category == "provenance_metadata":
                        candidate_provenance_metadata_atoms.append(rec["atom"])
                all_atom_records.append(rec)
                all_atoms.append(rec["atom"])

            _append_object_atom("object_in_segment", segment_id, object_id)
            if include_object_identity_atoms:
                _append_object_atom("object_track", object_id, f"track_{track_id}")
            _append_object_atom("object_class", segment_id, object_id, object_class)
            _append_object_atom("object_vz_state", segment_id, object_id, vz_state)
            _append_object_atom("object_vx_state", segment_id, object_id, vx_state)
            _append_object_atom("object_speed_state", segment_id, object_id, speed_state)
            _append_object_atom("object_distance_state", segment_id, object_id, distance_state)
            _append_object_atom("object_visibility_state", segment_id, object_id, visibility_state)
            _append_object_atom("object_x_position_state", segment_id, object_id, x_position_state)

            if is_candidate:
                selection_score = float(obj.get("selection_score", 0.0))
                prior_relevance_score = _prior_relevance_score(obj)
                source_label = "candidate"
                _append_object_atom("object_source_type", object_id, source_label)
                _append_object_atom("object_is_candidate", object_id)
                _append_object_atom(
                    "object_candidate_score_state",
                    object_id,
                    _score_state(selection_score),
                )
                _append_object_atom(
                    "object_prior_relevance_state",
                    object_id,
                    _score_state(prior_relevance_score),
                )
                for prior_id in _matched_prior_ids(obj):
                    _append_object_atom("object_matched_prior", object_id, prior_id)

            traffic_control_attributes = obj.get("traffic_control_attributes", {})
            if include_traffic_control_atoms and isinstance(traffic_control_attributes, dict):
                traffic_control_type = str(
                    traffic_control_attributes.get("traffic_control_type", "")
                ).strip()
                if traffic_control_type:
                    _append_object_atom(
                        "traffic_control_type",
                        segment_id,
                        object_id,
                        traffic_control_type,
                    )
                    relevance_label = str(
                        traffic_control_attributes.get("traffic_control_relevance_label", "low")
                    )
                    relevance_score = float(
                        traffic_control_attributes.get("traffic_control_relevance_score", 0.0)
                    )
                    _append_object_atom(
                        "traffic_control_relevance_state",
                        segment_id,
                        object_id,
                        relevance_label,
                    )
                    if relevance_score >= traffic_control_relevance_threshold:
                        _append_object_atom("traffic_control_relevant", segment_id, object_id)
                        if traffic_control_type == "traffic_light":
                            _append_object_atom("traffic_light_relevant", segment_id, object_id)
                        elif traffic_control_type == "stop_sign":
                            _append_object_atom("stop_sign_relevant", segment_id, object_id)
                    if bool(traffic_control_attributes.get("is_front_center_region", False)):
                        _append_object_atom(
                            "traffic_control_front_center_region",
                            segment_id,
                            object_id,
                        )
                        if traffic_control_type == "traffic_light":
                            _append_object_atom(
                                "traffic_light_position_state",
                                segment_id,
                                object_id,
                                "front_center",
                            )

                    if traffic_control_type == "traffic_light":
                        traffic_light_state = str(
                            traffic_control_attributes.get("traffic_light_state", "unknown")
                        )
                        if (
                            include_unknown_traffic_light_state_atoms
                            or traffic_light_state != "unknown"
                        ):
                            _append_object_atom(
                                "traffic_light_state",
                                segment_id,
                                object_id,
                                traffic_light_state,
                            )

            return {
                "track_id": track_id,
                "candidate_track_id": candidate_track_id if is_candidate else track_id,
                "object_id": object_id,
                "candidate_object_id": str(obj.get("candidate_object_id", "")) if is_candidate else "",
                "object_class": object_class,
                "accepted": not is_candidate,
                "source_type": str(obj.get("source_type", "accepted")) if is_candidate else "accepted",
                "bbox": list(obj.get("bbox", [])),
                "frame_detection_id": str(obj.get("frame_detection_id", "")) if is_candidate else "",
                "source_detection_ids": list(obj.get("source_detection_ids", [])) if is_candidate else [],
                "candidate_source": str(obj.get("candidate_source", "")) if is_candidate else "accepted_high_confidence",
                "selection_score": float(obj.get("selection_score", 0.0)) if is_candidate else 0.0,
                "prior_metadata": dict(obj.get("prior_metadata", {})) if is_candidate else {},
                "track_quality": dict(obj.get("track_quality", {})) if is_candidate else {},
                "traffic_control_attributes": traffic_control_attributes
                if isinstance(traffic_control_attributes, dict)
                else {},
                "atoms": object_atoms,
                "semantic_rule_search_atoms": candidate_semantic_rule_search_atoms if is_candidate else object_atoms,
                "quality_auxiliary_atoms": candidate_quality_auxiliary_atoms if is_candidate else [],
                "provenance_metadata_atoms": candidate_provenance_metadata_atoms if is_candidate else [],
                "atom_records": object_atom_records,
            }

        for obj in selected_objects:
            objects_out.append(_emit_object(obj, is_candidate=False))
        for obj in selected_candidate_objects:
            candidate_objects_out.append(_emit_object(obj, is_candidate=True))

        segments_out.append(
            {
                "segment_index": segment_index,
                "segment_id": segment_id,
                "segment_label": segment_label,
                "segment_forward_label": forward_label,
                "segment_lateral_label": lateral_label,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "atoms": segment_atoms,
                "atom_records": segment_atom_records,
                "objects": objects_out,
                "candidate_objects": candidate_objects_out,
            }
        )

    unique_object_ids = sorted(
        {
            obj.get("object_id", "")
            for segment in segments_out
            for obj in list(segment.get("objects", [])) + list(segment.get("candidate_objects", []))
            if obj.get("object_id", "")
        }
    )
    unique_candidate_object_ids = sorted(
        {
            obj.get("object_id", "")
            for segment in segments_out
            for obj in segment.get("candidate_objects", [])
            if obj.get("object_id", "")
        }
    )
    object_types = sorted(
        {
            obj.get("object_class", "")
            for segment in segments_out
            for obj in list(segment.get("objects", [])) + list(segment.get("candidate_objects", []))
            if obj.get("object_class", "")
        }
    )

    result: Dict[str, Any] = {
        "version": _LOGIC_ATOMS_VERSION,
        "video_id": video_id,
        "num_segments": len(segments_out),
        "num_objects": len(unique_object_ids),
        "num_candidate_objects": len(unique_candidate_object_ids),
        "object_ids": unique_object_ids,
        "candidate_object_ids": unique_candidate_object_ids,
        "object_types": object_types,
        "num_atoms": len(all_atoms),
        "config": {
            "lateral_position_threshold": lateral_position_threshold,
            "visibility_persistent_threshold": visibility_persistent_threshold,
            "visibility_present_threshold": visibility_present_threshold,
            "include_segment_boundary_atoms": include_segment_boundary_atoms,
            "include_object_identity_atoms": include_object_identity_atoms,
            "include_traffic_control_atoms": include_traffic_control_atoms,
            "traffic_control_relevance_threshold": traffic_control_relevance_threshold,
            "include_unknown_traffic_light_state_atoms": include_unknown_traffic_light_state_atoms,
        },
        "segments": segments_out,
        "all_atoms": all_atoms,
        "all_atom_records": all_atom_records,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    csv_columns = [
        "kind",
        "predicate",
        "arg_1",
        "arg_2",
        "arg_3",
        "arg_4",
        "segment_id",
        "object_id",
        "atom",
    ]
    with csv_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_columns)
        writer.writeheader()
        for rec in all_atom_records:
            args = list(rec.get("args", []))
            row = {
                "kind": rec.get("kind", ""),
                "predicate": rec.get("predicate", ""),
                "arg_1": args[0] if len(args) > 0 else "",
                "arg_2": args[1] if len(args) > 1 else "",
                "arg_3": args[2] if len(args) > 2 else "",
                "arg_4": args[3] if len(args) > 3 else "",
                "segment_id": args[0] if len(args) > 0 and str(args[0]).startswith("seg_") else "",
                "object_id": next((arg for arg in args if str(arg).startswith("obj_")), ""),
                "atom": rec.get("atom", ""),
            }
            writer.writerow(row)

    _print_video_summary(result)
    return result


def run(
    segment_object_motion_results: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    out_root = output_root or get_output_root()
    results: List[Dict[str, Any]] = []

    for segment_object_result in segment_object_motion_results:
        result = process_video(
            segment_object_motion_video_result=segment_object_result,
            cfg=cfg,
            output_root=out_root,
            force_recompute=force_recompute,
        )
        results.append(result)

    manifest = {
        "version": _LOGIC_ATOMS_VERSION,
        "num_videos": len(results),
        "videos": [
            {
                "video_id": r["video_id"],
                "num_segments": r.get("num_segments", 0),
                "num_objects": r.get("num_objects", 0),
                "num_candidate_objects": r.get("num_candidate_objects", 0),
                "object_types": r.get("object_types", []),
                "num_atoms": r.get("num_atoms", 0),
            }
            for r in results
        ],
    }
    manifest_path = out_root / "logic_atoms_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Logic atoms manifest written to {manifest_path}")
    return results
