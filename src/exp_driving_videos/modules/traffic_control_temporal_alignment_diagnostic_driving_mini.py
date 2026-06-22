"""
Diagnose whether traffic-control predicates align with immediate or delayed braking.

Outputs:
    pipeline_output/18f_driving_mini_traffic_control_temporal_alignment/
        traffic_control_temporal_alignment_summary.json
        traffic_control_horizon_rates.csv
        traffic_light_state_horizon_rates.csv
        traffic_control_horizon_coverage.csv
        traffic_control_alignment_examples.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config
from src.exp_driving_videos.modules.evaluate_rules_driving_mini import _parse_atom


_TRAFFIC_CONTROL_TEMPORAL_ALIGNMENT_VERSION = 3
_DEFAULT_HIGHLIGHT_PREDICATES: Tuple[str, ...] = (
    "traffic_light_state",
    "traffic_light_relevant",
    "traffic_control_relevant",
    "stop_sign_relevant",
)
_DEFAULT_STATE_VALUES: Tuple[str, ...] = ("red", "yellow", "green")
_DEFAULT_FUTURE_HORIZONS: Tuple[int, ...] = (1, 2, 3, 5)
_DEFAULT_POSITIVE_FORWARD_STATES: Tuple[str, ...] = ("forward_slowdown", "stopping")
_FOCUS_GROUP_KEYS: Tuple[Tuple[str, str, str, str], ...] = (
    ("group", "traffic_light", "", "all_traffic_lights"),
    ("group", "traffic_light_relevant", "", "relevant_traffic_lights"),
    ("group", "traffic_light_position_state", "front_center", "front_center_traffic_lights"),
    ("group", "traffic_light_state", "red", "traffic_light_state=red"),
    ("group", "traffic_light_state", "yellow", "traffic_light_state=yellow"),
    ("group", "traffic_light_state", "green", "traffic_light_state=green"),
    ("group", "stop_sign", "", "stop_signs"),
)


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "18f_driving_mini_traffic_control_temporal_alignment"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


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


def _safe_divide(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator / denominator)


def _horizon_name(horizon: int) -> str:
    return "brake_next" if int(horizon) == 1 else f"brake_within_{int(horizon)}_segments"


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "highlight_predicates": sorted(
            _normalize_text(value)
            for value in cfg.get("highlight_predicates", list(_DEFAULT_HIGHLIGHT_PREDICATES))
            if _normalize_text(value)
        ),
        "tracked_states": sorted(
            _normalize_text(value)
            for value in cfg.get("tracked_states", list(_DEFAULT_STATE_VALUES))
            if _normalize_text(value)
        ),
        "future_horizons": sorted(
            max(1, _safe_int(value, 1))
            for value in cfg.get("future_horizons", list(_DEFAULT_FUTURE_HORIZONS))
        ),
        "positive_forward_states": sorted(
            _normalize_text(value)
            for value in cfg.get("positive_forward_states", list(_DEFAULT_POSITIVE_FORWARD_STATES))
            if _normalize_text(value)
        ),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _is_traffic_control_predicate(predicate: str) -> bool:
    normalized = _normalize_text(predicate)
    return normalized.startswith("traffic_") or normalized == "stop_sign_relevant"


def _iter_eval_examples(eval_temporal_rule_results: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for video_result in eval_temporal_rule_results:
        video_id = str(video_result.get("video_id", ""))
        for example in list(video_result.get("examples", [])):
            row = dict(example)
            row["video_id"] = video_id
            yield row


def _collect_traffic_control_key_matches(
    body_atoms: Sequence[str],
) -> Dict[Tuple[str, str, str, str], List[str]]:
    key_matches: Dict[Tuple[str, str, str, str], List[str]] = {}
    traffic_light_atoms: List[str] = []
    relevant_traffic_light_atoms: List[str] = []
    front_center_traffic_light_atoms: List[str] = []
    stop_sign_atoms: List[str] = []
    state_atoms: Dict[str, List[str]] = {"red": [], "yellow": [], "green": []}

    for atom in body_atoms:
        parsed = _parse_atom(str(atom))
        if parsed is None:
            continue
        predicate_name, args = parsed
        normalized_predicate = _normalize_text(predicate_name)
        if not _is_traffic_control_predicate(normalized_predicate):
            continue

        predicate_key = ("predicate", normalized_predicate, "", normalized_predicate)
        key_matches.setdefault(predicate_key, []).append(str(atom))

        if normalized_predicate == "traffic_control_type" and len(args) >= 3:
            control_type = _normalize_text(args[2])
            if control_type == "traffic_light":
                traffic_light_atoms.append(str(atom))
            elif control_type == "stop_sign":
                stop_sign_atoms.append(str(atom))
        elif normalized_predicate == "traffic_light_relevant":
            relevant_traffic_light_atoms.append(str(atom))
            traffic_light_atoms.append(str(atom))
        elif normalized_predicate == "traffic_light_position_state" and len(args) >= 3:
            traffic_light_atoms.append(str(atom))
            if _normalize_text(args[2]) == "front_center":
                front_center_traffic_light_atoms.append(str(atom))
        elif normalized_predicate == "stop_sign_relevant":
            stop_sign_atoms.append(str(atom))

        if normalized_predicate == "traffic_light_state" and len(args) >= 3:
            state_name = _normalize_text(args[2]) or "unknown"
            key_matches.setdefault(
                ("state", normalized_predicate, state_name, f"{normalized_predicate}={state_name}"),
                [],
            ).append(str(atom))
            traffic_light_atoms.append(str(atom))
            if state_name in state_atoms:
                state_atoms[state_name].append(str(atom))

    def _dedupe(atoms: Sequence[str]) -> List[str]:
        return sorted({str(atom) for atom in atoms if str(atom)})

    for key_tuple, atoms in list(key_matches.items()):
        deduped = _dedupe(atoms)
        if deduped:
            key_matches[key_tuple] = deduped
        else:
            key_matches.pop(key_tuple, None)

    group_atoms = {
        ("group", "traffic_light", "", "all_traffic_lights"): traffic_light_atoms,
        ("group", "traffic_light_relevant", "", "relevant_traffic_lights"): relevant_traffic_light_atoms,
        (
            "group",
            "traffic_light_position_state",
            "front_center",
            "front_center_traffic_lights",
        ): front_center_traffic_light_atoms,
        ("group", "stop_sign", "", "stop_signs"): stop_sign_atoms,
    }
    for state_name, atoms in state_atoms.items():
        group_atoms[("group", "traffic_light_state", state_name, f"traffic_light_state={state_name}")] = atoms

    for key_tuple, atoms in group_atoms.items():
        deduped = _dedupe(atoms)
        if deduped:
            key_matches[key_tuple] = deduped
    return key_matches


def _segment_positions_by_video(
    logic_atom_results: Sequence[Dict[str, Any]],
    eval_video_ids: Set[str],
) -> Dict[str, Dict[str, Any]]:
    positions: Dict[str, Dict[str, Any]] = {}
    for video_result in logic_atom_results:
        video_id = str(video_result.get("video_id", ""))
        if video_id not in eval_video_ids:
            continue
        segments = sorted(
            list(video_result.get("segments", [])),
            key=lambda segment: (
                _safe_int(segment.get("segment_index", -1), -1),
                str(segment.get("segment_id", "")),
            ),
        )
        positions[video_id] = {
            "segments": segments,
            "position_by_segment_id": {
                str(segment.get("segment_id", "")): index for index, segment in enumerate(segments)
            },
        }
    return positions


def _future_brake_diagnostics(
    segments: Sequence[Dict[str, Any]],
    current_position: int,
    future_horizons: Sequence[int],
    positive_forward_states: Set[str],
) -> Dict[str, Any]:
    future_segments = list(segments[current_position + 1 :])
    future_labels = [_normalize_text(segment.get("segment_forward_label", "unknown")) for segment in future_segments]
    horizon_flags = {
        int(horizon): any(label in positive_forward_states for label in future_labels[: int(horizon)])
        for horizon in future_horizons
    }
    first_brake_horizon: Optional[int] = None
    for index, label in enumerate(future_labels, start=1):
        if label in positive_forward_states:
            first_brake_horizon = index
            break
    return {
        "future_labels": future_labels,
        "horizon_flags": horizon_flags,
        "first_brake_horizon": first_brake_horizon,
        "remaining_future_segments": len(future_labels),
    }


def _new_aggregate() -> Dict[str, Any]:
    return {
        "segment_count": 0,
        "video_ids": set(),
        "matched_atom_occurrences": 0,
        "brake_counts": {},
        "delayed_only_counts": {},
        "no_brake_within_max_horizon_count": 0,
        "no_future_brake_count": 0,
        "first_brake_horizon_sum": 0,
        "first_brake_horizon_count": 0,
    }


def _coverage_fieldnames(future_horizons: Sequence[int]) -> List[str]:
    fieldnames = [
        "entry_kind",
        "predicate_name",
        "state_name",
        "display_name",
        "num_firing_segments",
        "firing_segment_rate",
        "num_videos",
    ]
    for horizon in future_horizons:
        horizon_name = _horizon_name(horizon)
        fieldnames.extend(
            [
                f"{horizon_name}_precision_like",
                f"{horizon_name}_recall_coverage",
                f"{horizon_name}_positive_count",
            ]
        )
    return fieldnames


def _rate_fieldnames(future_horizons: Sequence[int]) -> List[str]:
    fieldnames = [
        "entry_kind",
        "predicate_name",
        "state_name",
        "display_name",
        "num_firing_segments",
        "firing_segment_rate",
        "num_videos",
        "matched_atom_occurrences",
        "mean_first_brake_horizon_any_future",
        "no_brake_within_max_horizon_count",
        "no_future_brake_count",
    ]
    previous_horizon = 0
    for horizon in future_horizons:
        horizon_name = _horizon_name(horizon)
        fieldnames.extend(
            [
                f"{horizon_name}_positive_count",
                f"{horizon_name}_positive_rate",
                f"{horizon_name}_dataset_base_rate",
                f"{horizon_name}_rate_lift_vs_base",
                f"{horizon_name}_rate_ratio_vs_base",
            ]
        )
        if previous_horizon:
            fieldnames.append(f"new_delayed_brake_count_h{previous_horizon}_to_h{horizon}")
        previous_horizon = horizon
    return fieldnames


def _row_lift(row: Dict[str, Any], horizon: int) -> float:
    return _safe_float(row.get(f"{_horizon_name(horizon)}_rate_lift_vs_base", 0.0))


def _row_positive_rate(row: Dict[str, Any], horizon: int) -> float:
    return _safe_float(row.get(f"{_horizon_name(horizon)}_positive_rate", 0.0))


def _diagnose_weakness(
    summary_rows: Sequence[Dict[str, Any]],
    dataset_base_rates: Dict[str, Dict[str, Any]],
    future_horizons: Sequence[int],
) -> Dict[str, Any]:
    rows_by_display = {str(row.get("display_name", "")): row for row in summary_rows}
    focus_rows = [row for row in summary_rows if str(row.get("entry_kind", "")) == "group"]
    focus_rows = [row for row in focus_rows if int(row.get("num_firing_segments", 0)) > 0]

    broad_candidates = [
        rows_by_display.get("relevant_traffic_lights", {}),
        rows_by_display.get("traffic_control_relevant", {}),
        rows_by_display.get("traffic_light_relevant", {}),
        rows_by_display.get("all_traffic_lights", {}),
    ]
    broad_row = max(
        broad_candidates,
        key=lambda row: int(row.get("num_firing_segments", 0)) if isinstance(row, dict) else 0,
        default={},
    )
    specific_rows = [
        rows_by_display.get("front_center_traffic_lights", {}),
        rows_by_display.get("traffic_light_state=red", {}),
        rows_by_display.get("traffic_light_state=yellow", {}),
        rows_by_display.get("traffic_light_state=green", {}),
        rows_by_display.get("stop_signs", {}),
    ]
    specific_rows = [row for row in specific_rows if int(row.get("num_firing_segments", 0)) > 0]

    broad_immediate_lift = _row_lift(broad_row, 1)
    broad_delayed_lift = max((_row_lift(broad_row, horizon) for horizon in future_horizons if horizon > 1), default=0.0)
    best_specific_immediate_lift = max((_row_lift(row, 1) for row in specific_rows), default=0.0)
    best_specific_delayed_lift = max(
        (_row_lift(row, horizon) for row in specific_rows for horizon in future_horizons if horizon > 1),
        default=0.0,
    )
    max_any_lift = max(
        (_row_lift(row, horizon) for row in focus_rows for horizon in future_horizons),
        default=0.0,
    )

    relevance_too_broad = (
        int(broad_row.get("num_firing_segments", 0)) > 0
        and broad_immediate_lift <= 0.03
        and best_specific_immediate_lift >= broad_immediate_lift + 0.08
    )
    temporally_earlier = (
        int(broad_row.get("num_firing_segments", 0)) > 0
        and broad_delayed_lift >= broad_immediate_lift + 0.08
        and broad_delayed_lift > 0.0
    )
    not_predictive = max_any_lift <= 0.05 and best_specific_delayed_lift <= 0.08

    if temporally_earlier:
        primary = "temporally_earlier_than_immediate_brake_next"
        answer = (
            "Traffic-control predicates look weak mainly because they fire earlier than the immediate "
            "brake_next target: delayed brake-within-horizon lift is stronger than immediate brake_next lift."
        )
    elif relevance_too_broad:
        primary = "relevance_too_broad"
        answer = (
            "Traffic-control predicates look weak mainly because the broad relevance signals are too coarse: "
            "specific traffic-light buckets are more predictive than the broad relevant-traffic-light bucket."
        )
    elif not_predictive:
        primary = "not_predictive"
        answer = (
            "Traffic-control predicates look weak mainly because they are not strongly predictive: their "
            "horizon rates stay close to the dataset base rate even in the delayed horizons."
        )
    else:
        primary = "mixed_or_moderate_signal"
        answer = (
            "Traffic-control predicates show mixed behavior: there is some signal, but it is not cleanly explained "
            "by only one of non-predictiveness, overly broad relevance, or purely early temporal offset."
        )

    return {
        "answer": answer,
        "primary_explanation": primary,
        "hypotheses": {
            "not_predictive": {
                "answer": bool(not_predictive),
                "evidence": {
                    "max_focus_group_lift_any_horizon": max_any_lift,
                    "best_specific_delayed_lift": best_specific_delayed_lift,
                },
            },
            "relevance_too_broad": {
                "answer": bool(relevance_too_broad),
                "evidence": {
                    "broad_bucket": str(broad_row.get("display_name", "")),
                    "broad_brake_next_lift": broad_immediate_lift,
                    "best_specific_brake_next_lift": best_specific_immediate_lift,
                },
            },
            "temporally_earlier_than_immediate_brake_next": {
                "answer": bool(temporally_earlier),
                "evidence": {
                    "broad_bucket": str(broad_row.get("display_name", "")),
                    "broad_brake_next_lift": broad_immediate_lift,
                    "best_broad_delayed_lift": broad_delayed_lift,
                },
            },
        },
        "evidence_snapshot": {
            "dataset_base_rates": dataset_base_rates,
            "broad_bucket_display_name": str(broad_row.get("display_name", "")),
            "broad_brake_next_positive_rate": _row_positive_rate(broad_row, 1),
            "broad_brake_next_lift": broad_immediate_lift,
            "broad_best_delayed_lift": broad_delayed_lift,
            "best_specific_brake_next_lift": best_specific_immediate_lift,
            "best_specific_delayed_lift": best_specific_delayed_lift,
        },
    }


def process_diagnostic(
    logic_atom_results: Sequence[Dict[str, Any]],
    eval_temporal_rule_results: Sequence[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_json_path = out_root / "traffic_control_temporal_alignment_summary.json"
    horizon_rates_csv_path = out_root / "traffic_control_horizon_rates.csv"
    state_horizon_rates_csv_path = out_root / "traffic_light_state_horizon_rates.csv"
    horizon_coverage_csv_path = out_root / "traffic_control_horizon_coverage.csv"
    examples_csv_path = out_root / "traffic_control_alignment_examples.csv"

    if not force_recompute and summary_json_path.exists():
        with summary_json_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _TRAFFIC_CONTROL_TEMPORAL_ALIGNMENT_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset(cfg):
            print(f"  [cache] loading {summary_json_path.name}")
            return cached

    future_horizons = sorted(
        {
            max(1, _safe_int(value, 1))
            for value in cfg.get("future_horizons", list(_DEFAULT_FUTURE_HORIZONS))
        }
    )
    if not future_horizons:
        future_horizons = list(_DEFAULT_FUTURE_HORIZONS)
    positive_forward_states = {
        _normalize_text(value)
        for value in cfg.get("positive_forward_states", list(_DEFAULT_POSITIVE_FORWARD_STATES))
        if _normalize_text(value)
    }
    if not positive_forward_states:
        positive_forward_states = set(_DEFAULT_POSITIVE_FORWARD_STATES)
    forced_predicates = {
        _normalize_text(value)
        for value in cfg.get("highlight_predicates", list(_DEFAULT_HIGHLIGHT_PREDICATES))
        if _normalize_text(value)
    }
    forced_states = {
        _normalize_text(value)
        for value in cfg.get("tracked_states", list(_DEFAULT_STATE_VALUES))
        if _normalize_text(value)
    }

    eval_examples = sorted(
        list(_iter_eval_examples(eval_temporal_rule_results)),
        key=lambda row: (
            str(row.get("video_id", "")),
            _safe_int(row.get("current_segment_index", -1), -1),
            str(row.get("example_id", "")),
        ),
    )
    if not eval_examples:
        raise RuntimeError("Traffic-control temporal alignment diagnostic found no evaluation examples to inspect.")

    eval_video_ids = {str(example.get("video_id", "")) for example in eval_examples if str(example.get("video_id", ""))}
    segment_positions = _segment_positions_by_video(logic_atom_results, eval_video_ids)
    missing_videos = sorted(video_id for video_id in eval_video_ids if video_id not in segment_positions)
    if missing_videos:
        raise RuntimeError(
            "Traffic-control temporal alignment diagnostic is missing logic-atom segments for "
            f"evaluation videos: {', '.join(missing_videos)}"
        )

    aggregate_by_key: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    example_rows: List[Dict[str, Any]] = []
    discovered_predicates: Set[str] = set(forced_predicates)
    discovered_states: Set[str] = set(forced_states)
    segments_with_any_traffic_control: Set[Tuple[str, str]] = set()
    brake_next_consistency_mismatches = 0
    max_horizon = max(future_horizons)
    dataset_positive_counts: Dict[int, int] = {int(horizon): 0 for horizon in future_horizons}

    for example in eval_examples:
        video_id = str(example.get("video_id", ""))
        current_segment_id = str(example.get("current_segment_id", ""))
        video_positions = segment_positions.get(video_id, {})
        positions_by_segment_id = dict(video_positions.get("position_by_segment_id", {}))
        segments = list(video_positions.get("segments", []))
        current_position = positions_by_segment_id.get(current_segment_id)
        if current_position is None:
            continue

        diagnostics = _future_brake_diagnostics(
            segments=segments,
            current_position=current_position,
            future_horizons=future_horizons,
            positive_forward_states=positive_forward_states,
        )
        for horizon in future_horizons:
            dataset_positive_counts[horizon] = dataset_positive_counts.get(horizon, 0) + int(
                bool(diagnostics["horizon_flags"].get(horizon, False))
            )

        brake_next_target = bool(example.get("label", False))
        brake_at_next_from_future = bool(diagnostics["horizon_flags"].get(1, False))
        if brake_next_target != brake_at_next_from_future:
            brake_next_consistency_mismatches += 1

        key_matches = _collect_traffic_control_key_matches(example.get("body_atoms", []))
        if not key_matches:
            continue

        segments_with_any_traffic_control.add((video_id, current_segment_id))
        for key_tuple, matched_atoms in sorted(key_matches.items()):
            if key_tuple[0] == "predicate":
                discovered_predicates.add(key_tuple[1])
            if key_tuple[1] == "traffic_light_state" and key_tuple[2]:
                discovered_states.add(key_tuple[2])

            aggregate = aggregate_by_key.setdefault(key_tuple, _new_aggregate())
            aggregate["segment_count"] += 1
            aggregate["video_ids"].add(video_id)
            aggregate["matched_atom_occurrences"] += len(matched_atoms)

            previous_horizon_flag = False
            for horizon in future_horizons:
                horizon_flag = bool(diagnostics["horizon_flags"].get(horizon, False))
                aggregate["brake_counts"][horizon] = aggregate["brake_counts"].get(horizon, 0) + int(horizon_flag)
                aggregate["delayed_only_counts"][horizon] = aggregate["delayed_only_counts"].get(horizon, 0) + int(
                    horizon_flag and not previous_horizon_flag
                )
                previous_horizon_flag = horizon_flag

            if not bool(diagnostics["horizon_flags"].get(max_horizon, False)):
                aggregate["no_brake_within_max_horizon_count"] += 1
            if diagnostics["first_brake_horizon"] is None:
                aggregate["no_future_brake_count"] += 1
            else:
                aggregate["first_brake_horizon_sum"] += int(diagnostics["first_brake_horizon"])
                aggregate["first_brake_horizon_count"] += 1

            example_row = {
                "video_id": video_id,
                "example_id": str(example.get("example_id", "")),
                "current_segment_id": current_segment_id,
                "current_segment_index": _safe_int(example.get("current_segment_index", -1), -1),
                "next_segment_id": str(example.get("next_segment_id", "")),
                "entry_kind": key_tuple[0],
                "predicate_name": key_tuple[1],
                "state_name": key_tuple[2],
                "display_name": key_tuple[3],
                "firing_segment": True,
                "matched_atom_count": len(matched_atoms),
                "matched_atoms_json": json.dumps(matched_atoms),
                "brake_next_target": brake_next_target,
                "brake_at_next_segment_from_future_labels": brake_at_next_from_future,
                "current_target_matches_future_labels": brake_next_target == brake_at_next_from_future,
                "remaining_future_segment_count": int(diagnostics["remaining_future_segments"]),
                "first_brake_horizon_any_future": (
                    "" if diagnostics["first_brake_horizon"] is None else int(diagnostics["first_brake_horizon"])
                ),
            }
            for horizon in future_horizons:
                example_row[_horizon_name(horizon)] = bool(diagnostics["horizon_flags"].get(horizon, False))
            example_rows.append(example_row)

    for predicate_name in sorted(discovered_predicates | forced_predicates):
        aggregate_by_key.setdefault(("predicate", predicate_name, "", predicate_name), _new_aggregate())
    for state_name in sorted(discovered_states | forced_states):
        aggregate_by_key.setdefault(
            ("state", "traffic_light_state", state_name, f"traffic_light_state={state_name}"),
            _new_aggregate(),
        )
    for key_tuple in _FOCUS_GROUP_KEYS:
        aggregate_by_key.setdefault(key_tuple, _new_aggregate())

    summary_rows: List[Dict[str, Any]] = []
    for key_tuple in sorted(aggregate_by_key, key=lambda item: (item[0], item[1], item[2], item[3])):
        aggregate = aggregate_by_key[key_tuple]
        segment_count = int(aggregate["segment_count"])
        row: Dict[str, Any] = {
            "entry_kind": key_tuple[0],
            "predicate_name": key_tuple[1],
            "state_name": key_tuple[2],
            "display_name": key_tuple[3],
            "num_firing_segments": segment_count,
            "firing_segment_rate": _safe_divide(segment_count, len(eval_examples)),
            "num_videos": len(set(aggregate["video_ids"])),
            "matched_atom_occurrences": int(aggregate["matched_atom_occurrences"]),
            "no_brake_within_max_horizon_count": int(aggregate["no_brake_within_max_horizon_count"]),
            "no_future_brake_count": int(aggregate["no_future_brake_count"]),
            "mean_first_brake_horizon_any_future": (
                0.0
                if int(aggregate["first_brake_horizon_count"]) == 0
                else float(aggregate["first_brake_horizon_sum"] / aggregate["first_brake_horizon_count"])
            ),
        }
        previous_horizon = 0
        for horizon in future_horizons:
            horizon_name = _horizon_name(horizon)
            positive_count = int(aggregate["brake_counts"].get(horizon, 0))
            dataset_positive_count = int(dataset_positive_counts.get(horizon, 0))
            dataset_base_rate = _safe_divide(dataset_positive_count, len(eval_examples))
            positive_rate = _safe_divide(positive_count, segment_count)
            row[f"{horizon_name}_positive_count"] = positive_count
            row[f"{horizon_name}_positive_rate"] = positive_rate
            row[f"{horizon_name}_dataset_base_rate"] = dataset_base_rate
            row[f"{horizon_name}_rate_lift_vs_base"] = positive_rate - dataset_base_rate
            row[f"{horizon_name}_rate_ratio_vs_base"] = _safe_divide(positive_rate, dataset_base_rate)
            row[f"{horizon_name}_precision_like"] = positive_rate
            row[f"{horizon_name}_recall_coverage"] = _safe_divide(positive_count, dataset_positive_count)
            if previous_horizon:
                row[f"new_delayed_brake_count_h{previous_horizon}_to_h{horizon}"] = int(
                    aggregate["delayed_only_counts"].get(horizon, 0)
                )
            previous_horizon = horizon
        summary_rows.append(row)

    rate_rows = [row for row in summary_rows if str(row.get("entry_kind", "")) != "state"]
    state_rows = [row for row in summary_rows if str(row.get("entry_kind", "")) == "state"]
    coverage_rows = [row for row in summary_rows if str(row.get("entry_kind", "")) in {"predicate", "group"}]

    example_rows = sorted(
        example_rows,
        key=lambda row: (
            str(row.get("video_id", "")),
            _safe_int(row.get("current_segment_index", -1), -1),
            str(row.get("display_name", "")),
        ),
    )

    _write_csv(horizon_rates_csv_path, _rate_fieldnames(future_horizons), rate_rows)
    _write_csv(state_horizon_rates_csv_path, _rate_fieldnames(future_horizons), state_rows)
    _write_csv(horizon_coverage_csv_path, _coverage_fieldnames(future_horizons), coverage_rows)

    example_fieldnames: List[str] = [
        "video_id",
        "example_id",
        "current_segment_id",
        "current_segment_index",
        "next_segment_id",
        "entry_kind",
        "predicate_name",
        "state_name",
        "display_name",
        "firing_segment",
        "matched_atom_count",
        "matched_atoms_json",
        "brake_next_target",
        "brake_at_next_segment_from_future_labels",
        "current_target_matches_future_labels",
        "remaining_future_segment_count",
        "first_brake_horizon_any_future",
    ]
    for horizon in future_horizons:
        example_fieldnames.append(_horizon_name(horizon))
    _write_csv(examples_csv_path, example_fieldnames, example_rows)

    dataset_base_rates = {
        _horizon_name(horizon): {
            "positive_count": int(dataset_positive_counts.get(horizon, 0)),
            "positive_rate": _safe_divide(int(dataset_positive_counts.get(horizon, 0)), len(eval_examples)),
        }
        for horizon in future_horizons
    }
    weakness_diagnosis = _diagnose_weakness(
        summary_rows=summary_rows,
        dataset_base_rates=dataset_base_rates,
        future_horizons=future_horizons,
    )

    summary = {
        "version": _TRAFFIC_CONTROL_TEMPORAL_ALIGNMENT_VERSION,
        "config": _cfg_key_subset(cfg),
        "diagnostic_only_future_label_usage": True,
        "scope": "evaluation_split_only",
        "num_eval_videos": len(eval_video_ids),
        "num_eval_examples": len(eval_examples),
        "dataset_base_rates": dataset_base_rates,
        "num_segments_with_any_traffic_control": len(segments_with_any_traffic_control),
        "num_keyed_segment_rows": len(example_rows),
        "brake_next_target_consistency_mismatch_count": brake_next_consistency_mismatches,
        "weakness_diagnosis": weakness_diagnosis,
        "focus_group_rows": {
            row["display_name"]: row
            for row in summary_rows
            if str(row.get("entry_kind", "")) == "group"
        },
        "rows": summary_rows,
        "output_paths": {
            "summary_json": str(summary_json_path),
            "traffic_control_horizon_rates_csv": str(horizon_rates_csv_path),
            "traffic_light_state_horizon_rates_csv": str(state_horizon_rates_csv_path),
            "traffic_control_horizon_coverage_csv": str(horizon_coverage_csv_path),
            "traffic_control_alignment_examples_csv": str(examples_csv_path),
        },
    }
    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        "  traffic_control_temporal_alignment_diagnostic: "
        f"segments={len(segments_with_any_traffic_control)} | "
        f"rows={len(summary_rows)} | "
        f"primary={weakness_diagnosis.get('primary_explanation', 'unknown')}"
    )
    print(f"Traffic-control temporal alignment summary JSON written to {summary_json_path}")
    print(f"Traffic-control horizon rates CSV written to {horizon_rates_csv_path}")
    print(f"Traffic-light state horizon rates CSV written to {state_horizon_rates_csv_path}")
    print(f"Traffic-control horizon coverage CSV written to {horizon_coverage_csv_path}")
    print(f"Traffic-control alignment examples CSV written to {examples_csv_path}")
    return summary


def run(
    logic_atom_results: Sequence[Dict[str, Any]],
    eval_temporal_rule_results: Sequence[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_diagnostic(
        logic_atom_results=logic_atom_results,
        eval_temporal_rule_results=eval_temporal_rule_results,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
