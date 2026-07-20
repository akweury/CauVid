"""Step 8A: grounded semantic protection for relative-motion tracks.

The LLM is constrained to exact, categorical atoms produced here. Its output is
validated before any rule can execute; unknown predicates, values, targets,
threshold expressions, or unsupported atoms are rejected.
"""

from __future__ import annotations

import json
import os
import re
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
    "relative_motion",
    "observation_source",
    "detection_confidence",
)
_BBOX_SIZE_THRESHOLDS = {"small_max_area_ratio": 0.01, "medium_max_area_ratio": 0.08}
_PERSISTENCE_THRESHOLDS = {"brief_max_ratio": 0.10, "persistent_min_ratio": 0.60}
_CONFIDENCE_THRESHOLDS = {"low_max": 0.35, "high_min": 0.70}
_MAX_RULES = 12
_MAX_CONDITIONS = 4
_MAX_RULE_COVERAGE_RATIO = 0.75


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
        vertical = "top" if cy < height / 3.0 else "bottom" if cy > 2.0 * height / 3.0 else "middle"
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


def _relative_motion(observations: Sequence[Dict[str, Any]]) -> str:
    speed = _mode(obs.get("speed_state", "speed_unknown") for obs in observations)
    vx = _mode(obs.get("vx_state", "vx_unknown") for obs in observations)
    vz = _mode(obs.get("vz_state", "vz_unknown") for obs in observations)
    return f"{speed}_{vx}_{vz}"


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
            "relative_motion": _relative_motion(observations),
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
        "You are a symbolic rule synthesizer. Generate only a compact set of semantic protection "
        "rules. Never classify individual tracks and never emit important_obj judgments. You may "
        "reason ONLY from the supplied ground atoms. Do not introduce predicates, constants, numeric "
        "thresholds, comparisons, latent visual claims, or general traffic rules. Every rule head "
        f"must be exactly {_RULE_HEAD}. Allowed body predicates are: "
        + ", ".join(_ALLOWED_PREDICATES)
        + ". Conditions are exact categorical equality matches using values already present in the "
        "supplied atoms. Avoid duplicate conditions, duplicate rule bodies, and rules broad enough to "
        "protect most tracks. Every justification must cite available symbols, and every supporting_atom "
        "must be copied exactly from the input. Return JSON only with this schema: "
        '{"rules":[{"rule_id":"r1","target":"protected_object","conditions":'
        '[{"predicate":"object_class","value":"pedestrian"}],"supporting_atoms":["exact atom"],'
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
        if not conditions or len(conditions) > _MAX_CONDITIONS:
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
            if predicate not in _ALLOWED_PREDICATES:
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
            accepted_rules.append(normalized)

    return {
        "rules": accepted_rules,
        "rejected_rules": rejected_rules,
    }


def _rule_matches(rule: Dict[str, Any], track: Dict[str, Any]) -> bool:
    features = dict(track.get("features", {}))
    return all(features.get(condition["predicate"]) == condition["value"] for condition in rule.get("conditions", []))


def _execute_rules(
    validated: Dict[str, Any],
    tracks: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    important_objects = []
    protected_objects = []
    uncovered_track_ids = []
    for track in tracks:
        track_id = int(track["track_id"])
        matched = [rule for rule in validated.get("rules", []) if _rule_matches(rule, track)]
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
            validated = validate_response(raw, tracks)
        except Exception as exc:
            status = "llm_unavailable" if "not configured" in str(exc) else "llm_error"
            error = str(exc)
            validated = {"rules": [], "rejected_rules": []}
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

