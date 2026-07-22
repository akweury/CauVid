"""Step 8A: grounded semantic protection for relative-motion tracks.

The LLM is constrained to exact, categorical atoms produced here. Its output is
validated before any rule can execute; unknown predicates, values, targets,
threshold expressions, or unsupported atoms are rejected.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


_VERSION = 1
_TARGET_PREDICATE = "important_obj"
_RULE_HEAD = "protected_object"
_ALLOWED_PREDICATES = (
    "object_class",
    "bbox_size",
    "screen_region",
    "relative_position",
    "temporal_persistence",
    "observation_source",
    "detection_confidence",
)
_BBOX_SIZE_THRESHOLDS = {"small_max_area_ratio": 0.01, "medium_max_area_ratio": 0.08}
_PERSISTENCE_THRESHOLDS = {"brief_max_ratio": 0.10, "persistent_min_ratio": 0.60}
_CONFIDENCE_THRESHOLDS = {"low_max": 0.35, "high_min": 0.70}
_MAX_RULES = 12
_MIN_CONDITIONS = 2
_MAX_CONDITIONS = 4
_MAX_RULE_COVERAGE_RATIO = 0.75
_FORBIDDEN_MOTION_PREDICATES = {
    "relative_motion",
    "vx",
    "vz",
    "object_vx_state",
    "object_vz_state",
    "trajectory_validity",
    "trajectory_decision",
    "keep",
    "discard",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sym(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).strip().lower()).strip("_")
    return text or "unknown"


def _mode(values: Iterable[Any], default: str = "unknown") -> str:
    cleaned = [_sym(value) for value in values if str(value).strip()]
    if not cleaned:
        return default
    counts = Counter(cleaned)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def _image_size(image_path: str) -> Tuple[int, int]:
    if not image_path:
        return 0, 0
    try:
        import cv2
    except ModuleNotFoundError:
        return 0, 0
    image = cv2.imread(str(image_path))
    if image is None:
        return 0, 0
    height, width = image.shape[:2]
    return int(width), int(height)


def _frame_size(relative_video: Dict[str, Any]) -> Tuple[int, int]:
    for frame in relative_video.get("frames", []):
        width, height = _image_size(str(frame.get("image_path", "")))
        if width > 0 and height > 0:
            return width, height
    boxes = [
        obj.get("bbox", obj.get("box", []))
        for frame in relative_video.get("frames", [])
        for obj in frame.get("objects", [])
    ]
    width = max((_safe_float(box[2]) for box in boxes if len(box) >= 4), default=1.0)
    height = max((_safe_float(box[3]) for box in boxes if len(box) >= 4), default=1.0)
    return max(1, int(round(width))), max(1, int(round(height)))


def _track_index(relative_video: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    tracks: Dict[int, List[Dict[str, Any]]] = {}
    for frame in relative_video.get("frames", []):
        frame_index = int(frame.get("frame_index", 0))
        for obj in frame.get("objects", []):
            try:
                track_id = int(obj.get("track_id", -1))
            except (TypeError, ValueError):
                continue
            if track_id < 0:
                continue
            tracks.setdefault(track_id, []).append({**dict(obj), "frame_index": frame_index})
    return tracks


def _bbox_size(observations: Sequence[Dict[str, Any]], width: int, height: int) -> str:
    ratios = []
    frame_area = max(1.0, float(width * height))
    for obs in observations:
        box = list(obs.get("bbox", obs.get("box", [])))
        if len(box) < 4:
            continue
        area = max(0.0, _safe_float(box[2]) - _safe_float(box[0]))
        area *= max(0.0, _safe_float(box[3]) - _safe_float(box[1]))
        ratios.append(area / frame_area)
    mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    if mean_ratio <= _BBOX_SIZE_THRESHOLDS["small_max_area_ratio"]:
        return "small"
    if mean_ratio <= _BBOX_SIZE_THRESHOLDS["medium_max_area_ratio"]:
        return "medium"
    return "large"


def _screen_region(observations: Sequence[Dict[str, Any]], width: int, height: int) -> str:
    regions = []
    for obs in observations:
        box = list(obs.get("bbox", obs.get("box", [])))
        if len(box) < 4:
            continue
        cx = (_safe_float(box[0]) + _safe_float(box[2])) / 2.0
        cy = (_safe_float(box[1]) + _safe_float(box[3])) / 2.0
        horizontal = "left" if cx < width / 3.0 else "right" if cx > 2.0 * width / 3.0 else "center"
        vertical = "upper" if cy < height / 3.0 else "lower" if cy > 2.0 * height / 3.0 else "middle"
        regions.append(f"{vertical}_{horizontal}")
    return _mode(regions)


def _relative_position(observations: Sequence[Dict[str, Any]]) -> str:
    lateral = _mode(obs.get("x_position_state", "unknown") for obs in observations)
    distance = _mode(obs.get("distance_state", "unknown") for obs in observations)
    return f"{distance}_{lateral}"


def _temporal_persistence(observations: Sequence[Dict[str, Any]], num_frames: int) -> str:
    ratio = len({int(obs.get("frame_index", -1)) for obs in observations}) / max(1, num_frames)
    if ratio <= _PERSISTENCE_THRESHOLDS["brief_max_ratio"]:
        return "brief"
    if ratio >= _PERSISTENCE_THRESHOLDS["persistent_min_ratio"]:
        return "persistent"
    return "intermittent"



def _observation_source(observations: Sequence[Dict[str, Any]]) -> str:
    observed = any(bool(obs.get("is_observed", False)) for obs in observations)
    repaired = any(bool(obs.get("is_repaired", False)) for obs in observations)
    if observed and repaired:
        return "mixed"
    return "repaired" if repaired else "observed"


def _detection_confidence(observations: Sequence[Dict[str, Any]]) -> str:
    values = [_safe_float(obs.get("score", 0.0)) for obs in observations]
    mean = sum(values) / len(values) if values else 0.0
    if mean <= _CONFIDENCE_THRESHOLDS["low_max"]:
        return "low"
    if mean >= _CONFIDENCE_THRESHOLDS["high_min"]:
        return "high"
    return "medium"


def _atom(predicate: str, video_id: str, track_id: int, value: str) -> str:
    return f"{predicate}({_sym(video_id)},track_{track_id},{_sym(value)})."


def build_grounded_tracks(relative_video: Dict[str, Any]) -> List[Dict[str, Any]]:
    video_id = str(relative_video.get("video_id", ""))
    num_frames = int(relative_video.get("num_frames", len(relative_video.get("frames", []))))
    width, height = _frame_size(relative_video)
    grounded = []
    for track_id, observations in sorted(_track_index(relative_video).items()):
        features = {
            "object_class": _mode(obs.get("frame_label", obs.get("label", "unknown")) for obs in observations),
            "bbox_size": _bbox_size(observations, width, height),
            "screen_region": _screen_region(observations, width, height),
            "relative_position": _relative_position(observations),
            "temporal_persistence": _temporal_persistence(observations, num_frames),
            "observation_source": _observation_source(observations),
            "detection_confidence": _detection_confidence(observations),
        }
        atoms = [_atom(predicate, video_id, track_id, features[predicate]) for predicate in _ALLOWED_PREDICATES]
        grounded.append(
            {
                "video_id": video_id,
                "track_id": track_id,
                "features": features,
                "atoms": atoms,
                "num_observations": len(observations),
            }
        )
    return grounded


def _prompt(video_id: str, tracks: Sequence[Dict[str, Any]]) -> str:
    payload = [{"track_id": row["track_id"], "atoms": row["atoms"]} for row in tracks]
    return (
        "You are a symbolic rule synthesizer. Generate only a compact, diverse set of executable "
        "semantic protection rules that identify potentially important objects independently of "
        "unreliable motion estimates. Never classify individual tracks and never emit important_obj "
        "judgments. Every rule head is protected_object(Video,Track). Each body must contain 2-4 "
        "robust semantic atoms. You may use ONLY these grounded body predicates: "
        + ", ".join(_ALLOWED_PREDICATES)
        + ". Never use relative_motion, vx, vz, trajectory validity, existing keep/discard decisions, "
        "numeric thresholds, comparisons, latent visual claims, unknown predicates, or constants absent "
        "from the supplied atoms. Prioritize object category, screen position, relative position, and "
        "bounding-box size. Prefer general rules preserving safety-relevant objects: vehicles near the "
        "ego path, large frontal vehicles, pedestrians or cyclists near the driving region, and visible "
        "traffic-control objects ahead. Do not generate near-duplicate rules differing only in bbox_size. "
        "Do not create isolated conjunctions matching only one supplied track. When proposing a specific "
        "rule, also consider a simpler semantically meaningful parent rule. Reject contradictory atoms "
        "and overly narrow conjunctions. Desired forms include protected_object(V,O) :- "
        "object_class(V,O,car), bbox_size(V,O,large), relative_position(V,O,near_centered); "
        "protected_object(V,O) :- object_class(V,O,traffic_light), "
        "screen_region(V,O,upper_center); and protected_object(V,O) :- "
        "object_class(V,O,pedestrian), relative_position(V,O,near_centered). Use example constants only "
        "when they occur in the supplied atoms. Every supporting_atom must be copied exactly from the "
        "input and every justification must be short and symbol-linked. Return JSON only with this schema: "
        '{"rules":[{"rule_id":"r1","target":"protected_object","conditions":'
        '[{"predicate":"object_class","value":"car"},'
        '{"predicate":"relative_position","value":"near_centered"}],'
        '"supporting_atoms":["exact input atom for each condition"],'
        '"justification":"short symbol-linked reason"}]}. '
        f"video_id={video_id}; grounded_tracks="
        + json.dumps(payload, separators=(",", ":"))
    )


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def _http_llm(prompt: str) -> Dict[str, Any]:
    response_path = os.environ.get("CAUVID_STEP8A_LLM_RESPONSE_PATH", "").strip()
    if response_path:
        return _extract_json(Path(response_path).read_text(encoding="utf-8"))
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    endpoint = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    model = os.environ.get("CAUVID_STEP8A_LLM_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return only grounded JSON that follows the user schema."},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    return _extract_json(payload["choices"][0]["message"]["content"])


def _condition_supported(condition: Dict[str, Any], domains: Dict[str, set]) -> bool:
    predicate = str(condition.get("predicate", ""))
    value = _sym(condition.get("value", ""))
    return predicate in _ALLOWED_PREDICATES and value in domains.get(predicate, set())


def _atom_supports_condition(atom: str, condition: Dict[str, Any]) -> bool:
    predicate = str(condition.get("predicate", ""))
    value = _sym(condition.get("value", ""))
    return str(atom).startswith(f"{predicate}(") and str(atom).endswith(f",{value}).")


def validate_response(raw: Dict[str, Any], tracks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    atom_set = {atom for track in tracks for atom in track["atoms"]}
    domains = {
        predicate: {str(track["features"][predicate]) for track in tracks}
        for predicate in _ALLOWED_PREDICATES
    }
    raw_rules = raw.get("rules", [])
    if not isinstance(raw_rules, list):
        return {
            "rules": [],
            "rejected_rules": [
                {
                    "rule_id": "invalid_rules_payload",
                    "target": _RULE_HEAD,
                    "conditions": [],
                    "supporting_atoms": [],
                    "justification": "",
                    "rejection_reason": "invalid_syntax",
                    "rejection_reasons": ["invalid_syntax"],
                }
            ],
        }

    accepted_rules = []
    rejected_rules = []
    seen_rule_ids = set()
    seen_body_signatures = set()
    seen_semantic_signatures = set()
    for index, candidate in enumerate(raw_rules):
        syntax_valid = isinstance(candidate, dict)
        rule = dict(candidate) if syntax_valid else {}
        reasons = []
        rule_id = _sym(rule.get("rule_id", f"rule_{index + 1}"))
        conditions_value = rule.get("conditions", [])
        conditions = list(conditions_value) if isinstance(conditions_value, list) else []
        supporting_value = rule.get("supporting_atoms", [])
        supporting_atoms = [str(atom) for atom in supporting_value] if isinstance(supporting_value, list) else []
        justification = str(rule.get("justification", "")).strip()
        if index >= _MAX_RULES:
            reasons.append("rule_budget_exceeded")
        if not syntax_valid or not isinstance(conditions_value, list) or not isinstance(supporting_value, list):
            reasons.append("invalid_syntax")
        if str(rule.get("target", "")) != _RULE_HEAD:
            reasons.append("invalid_target")
        if rule_id in seen_rule_ids:
            reasons.append("duplicate_rule_id")
        if len(conditions) < _MIN_CONDITIONS or len(conditions) > _MAX_CONDITIONS:
            reasons.append("invalid_condition_count")

        normalized_conditions = []
        condition_syntax_valid = True
        for condition in conditions:
            if (
                not isinstance(condition, dict)
                or set(condition) != {"predicate", "value"}
                or not isinstance(condition.get("predicate"), str)
                or not isinstance(condition.get("value"), str)
            ):
                condition_syntax_valid = False
                continue
            predicate = str(condition["predicate"])
            value = _sym(condition["value"])
            normalized_conditions.append({"predicate": predicate, "value": value})
            if predicate in _FORBIDDEN_MOTION_PREDICATES or any(
                token in predicate.lower()
                for token in ("relative_motion", "_vx", "_vz", "trajectory", "discard", "keep")
            ):
                reasons.append("motion_dependent_condition")
            elif predicate not in _ALLOWED_PREDICATES:
                reasons.append("unknown_predicate")
            elif value not in domains.get(predicate, set()):
                reasons.append("unsupported_constant")
        if not condition_syntax_valid:
            reasons.append("invalid_syntax")

        body_signature = tuple(
            sorted((condition["predicate"], condition["value"]) for condition in normalized_conditions)
        )
        if len(body_signature) != len(set(body_signature)) or body_signature in seen_body_signatures:
            reasons.append("redundant_body")
        values_by_predicate: Dict[str, set] = {}
        for predicate, value in body_signature:
            values_by_predicate.setdefault(predicate, set()).add(value)
        if any(len(values) > 1 for values in values_by_predicate.values()):
            reasons.append("contradictory_atoms")
        semantic_signature = tuple(
            literal for literal in body_signature if literal[0] != "bbox_size"
        )
        if semantic_signature in seen_semantic_signatures:
            reasons.append("near_duplicate_bbox_variant")

        if not supporting_atoms or any(atom not in atom_set for atom in supporting_atoms):
            reasons.append("ungrounded_atoms")
        elif any(
            not any(_atom_supports_condition(atom, condition) for atom in supporting_atoms)
            for condition in normalized_conditions
        ):
            reasons.append("ungrounded_atoms")
        if not justification:
            reasons.append("missing_justification")

        normalized = {
            "rule_id": rule_id,
            "target": _RULE_HEAD,
            "conditions": normalized_conditions,
            "supporting_atoms": supporting_atoms,
            "justification": justification[:400],
        }
        matched_track_ids = [
            int(track["track_id"]) for track in tracks if normalized_conditions and _rule_matches(normalized, track)
        ]
        coverage_ratio = len(matched_track_ids) / max(1, len(tracks))
        normalized["matched_track_ids"] = matched_track_ids
        normalized["coverage_ratio"] = float(coverage_ratio)
        if len(tracks) > 1 and len(matched_track_ids) < 2:
            reasons.append("overly_narrow_coverage")
        if tracks and coverage_ratio > _MAX_RULE_COVERAGE_RATIO:
            reasons.append("overly_general_coverage")

        reasons = list(dict.fromkeys(reasons))
        if reasons:
            rejected_rules.append(
                {
                    **normalized,
                    "rejection_reason": reasons[0],
                    "rejection_reasons": reasons,
                }
            )
        else:
            seen_rule_ids.add(rule_id)
            seen_body_signatures.add(body_signature)
            seen_semantic_signatures.add(semantic_signature)
            accepted_rules.append(normalized)

    return {
        "rules": accepted_rules,
        "rejected_rules": rejected_rules,
    }


def _rule_matches(rule: Dict[str, Any], track: Dict[str, Any]) -> bool:
    features = dict(track.get("features", {}))
    return all(features.get(condition["predicate"]) == condition["value"] for condition in rule.get("conditions", []))


def _rule_activation(rule: Dict[str, Any], track: Dict[str, Any]) -> Dict[str, Any]:
    track_atoms = list(track.get("atoms", []))
    matched_atoms = []
    missing_atoms = []
    missing_conditions = []
    for condition in rule.get("conditions", []):
        matching_atom = next(
            (atom for atom in track_atoms if _atom_supports_condition(atom, condition)),
            None,
        )
        if matching_atom is not None:
            matched_atoms.append(matching_atom)
        else:
            missing_atoms.append(
                _atom(
                    condition["predicate"],
                    str(track.get("video_id", "")),
                    int(track.get("track_id", -1)),
                    condition["value"],
                )
            )
            missing_conditions.append(
                f"{condition['predicate']}={condition['value']}"
            )
    active = not missing_atoms
    return {
        "rule_id": str(rule.get("rule_id", "")),
        "active": active,
        "matched_atoms": matched_atoms,
        "missing_atoms": missing_atoms,
        "activation_failure_reason": (
            ""
            if active
            else "missing_required_atoms: " + ", ".join(missing_conditions)
        ),
    }


def _execute_rules(
    validated: Dict[str, Any],
    tracks: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    important_objects = []
    protected_objects = []
    uncovered_track_ids = []
    for track in tracks:
        track_id = int(track["track_id"])
        activations = [
            _rule_activation(rule, track)
            for rule in validated.get("rules", [])
        ]
        track["rule_activations"] = activations
        active_rule_ids = {
            activation["rule_id"]
            for activation in activations
            if bool(activation.get("active", False))
        }
        matched = [
            rule
            for rule in validated.get("rules", [])
            if str(rule.get("rule_id", "")) in active_rule_ids
        ]
        if not matched:
            uncovered_track_ids.append(track_id)
            continue

        grounding_evidence = []
        for rule in matched:
            condition_evidence = []
            for condition in rule.get("conditions", []):
                ground_atom = next(
                    atom
                    for atom in track.get("atoms", [])
                    if _atom_supports_condition(atom, condition)
                )
                condition_evidence.append(
                    {
                        "predicate": condition["predicate"],
                        "value": condition["value"],
                        "ground_atom": ground_atom,
                    }
                )
            grounding_evidence.append(
                {
                    "rule_id": rule["rule_id"],
                    "conditions": condition_evidence,
                    "justification": rule["justification"],
                }
            )

        matched_rule_ids = [rule["rule_id"] for rule in matched]
        reasons = list(dict.fromkeys(rule["justification"] for rule in matched))
        protection_reason = " | ".join(reasons)
        protected = {
            "video_id": track["video_id"],
            "track_id": track_id,
            "atom": f"{_RULE_HEAD}({_sym(track['video_id'])},track_{track_id}).",
            "matched_rule_ids": matched_rule_ids,
            "grounding_evidence": grounding_evidence,
            "protection_reason": protection_reason,
            "original_decision_before_protection": None,
            "trajectory_decision": "pending_step8b",
            "final_decision_after_protection": "pending_step8b",
        }
        protected_objects.append(protected)
        important_objects.append(
            {
                "video_id": track["video_id"],
                "track_id": track_id,
                "atom": f"{_TARGET_PREDICATE}({_sym(track['video_id'])},track_{track_id}).",
                "derived_from": _RULE_HEAD,
                "matched_rule_ids": matched_rule_ids,
                "grounding_evidence": grounding_evidence,
                "protection_reason": protection_reason,
            }
        )
    return important_objects, protected_objects, uncovered_track_ids


def _annotate_protection_prior(
    relative_video: Dict[str, Any],
    protected_objects: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    protected_by_track = {int(row["track_id"]): dict(row) for row in protected_objects}
    frames = []
    for frame in relative_video.get("frames", []):
        objects = []
        for obj in frame.get("objects", []):
            enriched = dict(obj)
            protection = protected_by_track.get(int(obj.get("track_id", -1)))
            enriched["symbol_grounded_protected"] = bool(protection)
            if protection:
                enriched["symbol_grounded_protection"] = protection
            objects.append(enriched)
        frames.append({**dict(frame), "objects": objects})
    return {**dict(relative_video), "frames": frames}


def run_symbol_grounded_refinement(
    relative_motion_state: Dict[str, Any],
    output_root: Path,
    llm_generate: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    generator = llm_generate or _http_llm
    video_results = []
    all_important = []
    all_protected = []
    all_rules = []
    all_uncovered = []
    protection_annotated_relative_motion = []
    for relative_video in relative_motion_state.get("relative_object_motion", []):
        video_id = str(relative_video.get("video_id", ""))
        tracks = build_grounded_tracks(relative_video)
        status = "completed"
        error = ""
        raw: Dict[str, Any] = {}
        try:
            raw = generator(_prompt(video_id, tracks))
            if not isinstance(raw.get("rules"), list) or not raw.get("rules"):
                raise ValueError("LLM returned no semantic protection rules")
            validated = validate_response(raw, tracks)
        except Exception as exc:
            message = (
                f"[step 8a][error] LLM rule generation failed "
                f"video_id={video_id}: {type(exc).__name__}: {exc}"
            )
            print(message, file=sys.stderr, flush=True)
            raise RuntimeError(message) from exc
        important, protected, uncovered = _execute_rules(validated, tracks)
        protection_annotated_relative_motion.append(
            _annotate_protection_prior(relative_video, protected)
        )
        result = {
            "status": status,
            "rule_head_predicate": _RULE_HEAD,
            "grounded_tracks": tracks,
            "semantic_protection_rules": validated["rules"],
            "rejected_rules": validated["rejected_rules"],
            "important_objects": important,
            "protected_objects": protected,
            "uncovered_track_ids": uncovered,
        }
        video_dir = output_root / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        with (video_dir / "symbol_grounded_refinement.json").open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
        video_results.append({**result, "video_id": video_id, "error": error})
        all_important.extend(important)
        all_protected.extend(protected)
        all_uncovered.extend({"video_id": video_id, "track_id": track_id} for track_id in uncovered)
        all_rules.extend({**rule, "video_id": video_id} for rule in validated["rules"])

    manifest = {
        "version": _VERSION,
        "num_videos": len(video_results),
        "num_grounded_tracks": sum(len(row["grounded_tracks"]) for row in video_results),
        "num_rules": len(all_rules),
        "num_rejected_rules": sum(len(row["rejected_rules"]) for row in video_results),
        "num_protected_objects": len(all_protected),
        "num_uncovered_tracks": len(all_uncovered),
        "status_counts": dict(Counter(row["status"] for row in video_results)),
        "videos": [
            {
                "video_id": row["video_id"],
                "status": row["status"],
                "num_grounded_tracks": len(row["grounded_tracks"]),
                "num_rules": len(row["semantic_protection_rules"]),
                "num_rejected_rules": len(row["rejected_rules"]),
                "num_protected_objects": len(row["protected_objects"]),
                "num_uncovered_tracks": len(row["uncovered_track_ids"]),
                "error": row["error"],
            }
            for row in video_results
        ],
    }
    with (output_root / "symbol_grounded_refinement_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return {
        **relative_motion_state,
        "relative_object_motion": protection_annotated_relative_motion,
        "protection_annotated_relative_motion": protection_annotated_relative_motion,
        "symbol_grounded_refinement": video_results,
        "semantic_protection_rules": all_rules,
        "important_objects": all_important,
        "protected_objects": all_protected,
        "uncovered_tracks": all_uncovered,
        "symbol_grounded_refinement_output_root": output_root,
    }



def _representative_track_frame(
    relative_video: Dict[str, Any],
    track_id: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    candidates = []
    for frame in relative_video.get("frames", []):
        for obj in frame.get("objects", []):
            if int(obj.get("track_id", -1)) != int(track_id):
                continue
            box = list(obj.get("bbox", obj.get("box", [])))
            area = 0.0
            if len(box) >= 4:
                area = max(0.0, _safe_float(box[2]) - _safe_float(box[0]))
                area *= max(0.0, _safe_float(box[3]) - _safe_float(box[1]))
            candidates.append(
                (
                    int(bool(obj.get("is_observed", False))),
                    area,
                    int(frame.get("frame_index", 0)),
                    frame,
                    obj,
                )
            )
    if not candidates:
        return None, None
    _, _, _, frame, obj = max(candidates, key=lambda row: (row[0], row[1], row[2]))
    return frame, obj


def _wrap_visual_text(cv2: Any, text: str, width: int, scale: float, thickness: int) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _rule_summary(rule: Dict[str, Any]) -> str:
    body = ", ".join(
        f"{condition.get('predicate')}={condition.get('value')}"
        for condition in rule.get("conditions", [])
    )
    return f"{rule.get('rule_id')}: {body}"


def _render_track_grounding_image(
    relative_video: Dict[str, Any],
    video_result: Dict[str, Any],
    track: Dict[str, Any],
    output_path: Path,
) -> Tuple[Optional[str], str]:
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError:
        return None, "missing_cv2_or_numpy"

    track_id = int(track.get("track_id", -1))
    frame, obj = _representative_track_frame(relative_video, track_id)
    if frame is None or obj is None:
        return None, "track_not_found"
    image = cv2.imread(str(frame.get("image_path", "")))
    if image is None:
        return None, "missing_frame_image"

    frame_h, frame_w = image.shape[:2]
    protected_by_track = {
        int(row.get("track_id", -1)): row
        for row in video_result.get("protected_objects", [])
    }
    protection = protected_by_track.get(track_id)
    is_protected = protection is not None
    status_color = (70, 220, 70) if is_protected else (80, 180, 255)
    box = list(obj.get("bbox", obj.get("box", [])))
    if len(box) >= 4:
        x1, y1, x2, y2 = [int(round(_safe_float(value))) for value in box]
        x1, x2 = sorted((max(0, min(frame_w - 1, x1)), max(0, min(frame_w - 1, x2))))
        y1, y2 = sorted((max(0, min(frame_h - 1, y1)), max(0, min(frame_h - 1, y2))))
        thickness = max(3, int(round(min(frame_w, frame_h) / 180.0)))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), thickness + 3)
        cv2.rectangle(image, (x1, y1), (x2, y2), status_color, thickness)
        label = (
            f"track {track_id} | {track.get('features', {}).get('object_class', 'unknown')} | "
            f"{'PROTECTED' if is_protected else 'UNCOVERED'}"
        )
        text_y = max(32, y1 - 10)
        cv2.rectangle(image, (x1, text_y - 28), (min(frame_w - 1, x1 + 600), text_y + 7), status_color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 6, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    accepted_rules = list(video_result.get("semantic_protection_rules", []))
    rejected_rules = list(video_result.get("rejected_rules", []))
    accepted_atom_rows = sum(max(1, len(rule.get("conditions", []))) for rule in accepted_rules)
    rejected_atom_rows = sum(max(1, len(rule.get("conditions", []))) for rule in rejected_rules)
    panel_h = max(
        480,
        310
        + len(accepted_rules) * 155
        + accepted_atom_rows * 82
        + len(rejected_rules) * 125
        + rejected_atom_rows * 82,
    )
    panel = np.full((panel_h, frame_w, 3), (24, 24, 24), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 22
    right_x = margin
    right_w = max(1, frame_w - 2 * margin)

    cv2.putText(
        panel,
        "LLM semantic protection rules and per-track activations",
        (right_x, 38),
        font,
        0.76,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    result_y = 76
    result_lines = [
        f"status: {video_result.get('status', 'unknown')}",
        f"accepted rules: {len(video_result.get('semantic_protection_rules', []))}",
        f"rejected rules: {len(video_result.get('rejected_rules', []))}",
        f"track result: {'protected_object' if is_protected else 'uncovered'}",
    ]
    for line in result_lines:
        color = status_color if line.startswith("track result") else (215, 215, 215)
        cv2.putText(panel, line, (right_x, result_y), font, 0.60, color, 2 if line.startswith("track result") else 1, cv2.LINE_AA)
        result_y += 42

    matched_ids = set(protection.get("matched_rule_ids", [])) if protection else set()
    activation_by_rule = {
        str(row.get("rule_id", "")): row
        for row in track.get("rule_activations", [])
    }
    cv2.putText(
        panel,
        "Accepted rules: activation for this track",
        (right_x, result_y),
        font,
        0.54,
        (225, 225, 225),
        2,
        cv2.LINE_AA,
    )
    result_y += 36
    if not accepted_rules:
        cv2.putText(panel, "none", (right_x, result_y), font, 0.50, (150, 150, 150), 1, cv2.LINE_AA)
        result_y += 34
    track_features = dict(track.get("features", {}))
    atom_valid_color = (70, 230, 70)
    atom_invalid_color = (60, 80, 240)
    for rule in accepted_rules:
        rule_id = str(rule.get("rule_id", ""))
        active = rule_id in matched_ids
        activation = "ACTIVE" if active else "INACTIVE"
        rule_color = atom_valid_color if active else (190, 190, 190)
        header = f"[{activation}] {rule_id} -> protected_object"
        cv2.putText(
            panel,
            header,
            (right_x, result_y),
            font,
            0.70,
            rule_color,
            3 if active else 2,
            cv2.LINE_AA,
        )
        result_y += 40
        for atom_index, condition in enumerate(rule.get("conditions", []), start=1):
            predicate = str(condition.get("predicate", ""))
            value = str(condition.get("value", ""))
            atom_grounded = any(
                _atom_supports_condition(atom, condition)
                for atom in rule.get("supporting_atoms", [])
            )
            atom_matches = (
                predicate in _ALLOWED_PREDICATES
                and atom_grounded
                and track_features.get(predicate) == value
            )
            atom_color = atom_valid_color if atom_matches else atom_invalid_color
            atom_status = "VALID / MATCH" if atom_matches else "INVALID / NO MATCH"
            atom_text = f"  atom {atom_index}: {predicate}={value} [{atom_status}]"
            for line in _wrap_visual_text(cv2, atom_text, right_w - 12, 0.58, 2):
                cv2.putText(panel, line, (right_x + 12, result_y), font, 0.58, atom_color, 2, cv2.LINE_AA)
                result_y += 33
        justification = f"why: {str(rule.get('justification', ''))[:140]}"
        for line in _wrap_visual_text(cv2, justification, right_w, 0.43, 1)[:2]:
            cv2.putText(panel, line, (right_x, result_y), font, 0.43, rule_color, 1, cv2.LINE_AA)
            result_y += 25
        if not active:
            failure = str(
                activation_by_rule.get(rule_id, {}).get(
                    "activation_failure_reason",
                    "required semantic atoms did not match",
                )
            )
            for line in _wrap_visual_text(cv2, f"failure: {failure}", right_w, 0.42, 1)[:2]:
                cv2.putText(panel, line, (right_x, result_y), font, 0.42, atom_invalid_color, 1, cv2.LINE_AA)
                result_y += 24
        result_y += 12

    if rejected_rules:
        result_y += 4
        cv2.putText(
            panel,
            "Rejected LLM-generated rules",
            (right_x, result_y),
            font,
            0.54,
            (90, 110, 240),
            2,
            cv2.LINE_AA,
        )
        result_y += 36
        for rule in rejected_rules:
            reasons = ", ".join(rule.get("rejection_reasons", [])) or str(
                rule.get("rejection_reason", "validation_failed")
            )
            rule_id = str(rule.get("rule_id", ""))
            cv2.putText(
                panel,
                f"[REJECTED] {rule_id} -> protected_object",
                (right_x, result_y),
                font,
                0.65,
                atom_invalid_color,
                2,
                cv2.LINE_AA,
            )
            result_y += 38
            conditions = list(rule.get("conditions", []))
            if not conditions:
                cv2.putText(
                    panel,
                    "  atom list invalid or empty [INVALID]",
                    (right_x + 12, result_y),
                    font,
                    0.55,
                    atom_invalid_color,
                    2,
                    cv2.LINE_AA,
                )
                result_y += 33
            for atom_index, condition in enumerate(conditions, start=1):
                predicate = str(condition.get("predicate", ""))
                value = str(condition.get("value", ""))
                atom_matches = predicate in _ALLOWED_PREDICATES and track_features.get(predicate) == value
                atom_color = atom_valid_color if atom_matches else atom_invalid_color
                atom_status = "VALID / MATCH" if atom_matches else "INVALID / NO MATCH"
                atom_text = f"  atom {atom_index}: {predicate}={value} [{atom_status}]"
                for line in _wrap_visual_text(cv2, atom_text, right_w - 12, 0.58, 2):
                    cv2.putText(panel, line, (right_x + 12, result_y), font, 0.58, atom_color, 2, cv2.LINE_AA)
                    result_y += 33
            for line in _wrap_visual_text(cv2, f"rule rejection: {reasons}", right_w, 0.45, 1)[:3]:
                cv2.putText(panel, line, (right_x, result_y), font, 0.45, atom_invalid_color, 1, cv2.LINE_AA)
                result_y += 25
            result_y += 10

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = cv2.vconcat([image, panel])
    if not cv2.imwrite(str(output_path), combined):
        return None, "image_write_failed"
    return str(output_path), "rendered"


def render_symbol_grounded_visualizations(
    relative_motion_state: Dict[str, Any],
    output_root: Path,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    relative_by_video = {
        str(row.get("video_id", "")): row
        for row in relative_motion_state.get("relative_object_motion", [])
    }
    rendered = []
    skipped = []
    for video_result in relative_motion_state.get("symbol_grounded_refinement", []):
        video_id = str(video_result.get("video_id", ""))
        relative_video = relative_by_video.get(video_id, {})
        for track in video_result.get("grounded_tracks", []):
            track_id = int(track.get("track_id", -1))
            path, status = _render_track_grounding_image(
                relative_video,
                video_result,
                track,
                output_root / video_id / f"track_{track_id:04d}_symbol_grounded.png",
            )
            row = {"video_id": video_id, "track_id": track_id, "status": status}
            if path:
                row["visualization_path"] = path
                rendered.append(row)
            else:
                skipped.append(row)

    manifest = {
        "version": _VERSION,
        "num_images_rendered": len(rendered),
        "num_images_skipped": len(skipped),
        "rendered": rendered,
        "skipped": skipped,
    }
    with (output_root / "symbol_grounded_visualization_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return {
        **relative_motion_state,
        "symbol_grounded_visualizations": rendered,
        "symbol_grounded_visualization_skipped": skipped,
        "symbol_grounded_visualization_output_root": output_root,
    }

