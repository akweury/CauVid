"""
Pattern mining scaffold for nuScenes segment predicates.

This module is intended to adapt the driving-video ``prims2patterns`` flow to
the nuScenes predicate schema:

1. Load per-scene ``segment_predicates.json`` files.
2. Convert each segment into symbolic atoms.
3. Mine lagged temporal rules across scene timelines.
4. Score, filter, and save discovered rules.

Implementation is intentionally left for a later step.
"""

from __future__ import annotations

import argparse
import json
import sys
import math
import csv
import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.exp_driving_videos.pipe_utils.exp_driving_utils import load_pattern_cfg_file


Atom = Tuple[str, Any]
Rule = Tuple[Atom, Atom, int]

DEFAULT_PATTERN_MINING_CONFIG = {
    "output_root": str(Path("pipeline_output") / "nuScenes_from_config"),
    "scene_names": [],
    "top_k_objects": 3,
    "max_lag": 3,
    # minimum absolute support (counts) required to keep a rule
    "min_support": 4,
    # minimum number of distinct scenes a rule must appear in
    "min_scene_support": 3,
    "save_atoms": None,
    "debug": False,
    "extend_iters": 0,
}


def _deep_merge_dicts(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_project_path(path_value: Optional[Any]) -> Optional[Path]:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_pattern_mining_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load pattern-mining options, accepting the nuScenes pipeline config shape."""
    run_cfg = dict(DEFAULT_PATTERN_MINING_CONFIG)
    if config_path is None:
        return run_cfg

    loaded = load_pattern_cfg_file(config_path) or {}
    pattern_cfg = loaded.get("pattern_mining", {})
    run_cfg = _deep_merge_dicts(run_cfg, loaded)
    run_cfg = _deep_merge_dicts(run_cfg, pattern_cfg)

    if "scene" in loaded and not loaded.get("scene_names"):
        run_cfg["scene_names"] = loaded["scene"]
    if isinstance(run_cfg.get("scene_names"), str):
        run_cfg["scene_names"] = [run_cfg["scene_names"]]
    return run_cfg


def resolve_run_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve CLI arguments over optional config file values."""
    run_cfg = load_pattern_mining_config(getattr(args, "config", None))

    if getattr(args, "output_root", None) is not None:
        run_cfg["output_root"] = str(args.output_root)
    if getattr(args, "scene", None):
        run_cfg["scene_names"] = list(args.scene)
    if getattr(args, "top_k_objects", None) is not None:
        run_cfg["top_k_objects"] = args.top_k_objects
    if getattr(args, "max_lag", None) is not None:
        run_cfg["max_lag"] = args.max_lag
    if getattr(args, "save_atoms", None) is not None:
        run_cfg["save_atoms"] = str(args.save_atoms)
    if getattr(args, "debug", None) is not None:
        run_cfg["debug"] = args.debug
    if getattr(args, "extend_iters", None) is not None:
        run_cfg["extend_iters"] = int(args.extend_iters)
    if getattr(args, "min_support", None) is not None:
        run_cfg["min_support"] = int(args.min_support)
    if getattr(args, "min_scene_support", None) is not None:
        run_cfg["min_scene_support"] = int(args.min_scene_support)

    return run_cfg


def load_segment_predicates(scene_path: Path) -> Dict[str, Any]:
    """Load one scene's segment predicate payload."""
    scene_path = Path(scene_path)
    predicate_file = scene_path
    if scene_path.is_dir():
        predicate_file = scene_path / "segment_predicates.json"

    if not predicate_file.exists():
        raise FileNotFoundError(f"Segment predicate file does not exist: {predicate_file}")

    with predicate_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"{predicate_file} is missing a list-valued 'segments' field.")

    return payload


def _state_label(state: Any) -> Optional[str]:
    if not isinstance(state, dict):
        return None
    label = state.get("label")
    if label is None:
        return None
    return str(label)


def _append_state_atom(atoms: List[Atom], key: str, state: Any) -> None:
    label = _state_label(state)
    if label is not None:
        atoms.append((key, label))


def _forward_label_from_segment(segment: Dict[str, Any]) -> Optional[str]:
    combined_label = segment.get("combined_label_name")
    if not isinstance(combined_label, str) or not combined_label:
        return None
    return combined_label.split("+", 1)[0]


def _ego_speed_change_label(forward_label: Optional[str]) -> Optional[str]:
    label_map = {
        "speed_up": "speeding_up",
        "slow_down": "slowing_down",
        "stable": "constant_speed",
        "stop": "stopped",
    }
    if forward_label is None:
        return None
    return label_map.get(forward_label, forward_label)


def _object_rank(obj: Dict[str, Any]) -> float:
    rank = obj.get("nearest_distance_rank")
    if rank is None:
        return float("inf")
    try:
        return float(rank)
    except (TypeError, ValueError):
        return float("inf")


def _select_focus_objects(objects: Iterable[Dict[str, Any]], top_k_objects: int) -> List[Dict[str, Any]]:
    sorted_objects = sorted(objects, key=_object_rank)
    if top_k_objects <= 0:
        return sorted_objects
    return sorted_objects[:top_k_objects]


def segment_to_atoms(segment: Dict[str, Any], top_k_objects: int = 3) -> List[Atom]:
    """Convert one nuScenes segment predicate record into symbolic atoms."""
    atoms: List[Atom] = []

    combined_label = segment.get("combined_label_name")
    if combined_label is not None:
        atoms.append(("ego_motion_segment", str(combined_label)))

    speed_change = _ego_speed_change_label(_forward_label_from_segment(segment))
    if speed_change is not None:
        atoms.append(("ego_speed_change", speed_change))

    ego_predicates = segment.get("ego", {})
    if isinstance(ego_predicates, dict):
        _append_state_atom(atoms, "ego_is_moving", ego_predicates.get("ego_is_moving"))
        _append_state_atom(atoms, "ego_is_turning", ego_predicates.get("ego_is_turning"))

    objects = segment.get("objects", [])
    if not isinstance(objects, list):
        return atoms

    focus_objects = _select_focus_objects(objects, top_k_objects)
    atoms.append(("num_focus_objects", len(focus_objects)))

    for obj in focus_objects:
        if not isinstance(obj, dict):
            continue
        category_name = obj.get("category_name")
        if category_name is not None:
            atoms.append(("obj_category", str(category_name)))
        _append_state_atom(atoms, "obj_is_moving", obj.get("is_moving"))
        _append_state_atom(atoms, "obj_is_turning", obj.get("is_turning"))
        _append_state_atom(atoms, "obj_relative_motion", obj.get("relative_motion"))
        _append_state_atom(atoms, "obj_lateral_motion", obj.get("lateral_motion"))

    return atoms


def predicates_to_timeline_atoms(
    segment_predicates: Dict[str, Any],
    top_k_objects: int = 3,
) -> List[List[Atom]]:
    """Convert a scene-level predicate payload into one atom list per segment."""
    segments = segment_predicates.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("segment_predicates must contain a list-valued 'segments' field.")
    return [
        segment_to_atoms(segment, top_k_objects=top_k_objects)
        for segment in segments
        if isinstance(segment, dict)
    ]


def discover_scene_paths(output_root: Path, scene_names: Optional[Sequence[str]] = None) -> List[Path]:
    """Find scene directories containing segment predicate files."""
    output_root = Path(output_root)
    if scene_names:
        return [output_root / scene_name for scene_name in scene_names]
    if (output_root / "segment_predicates.json").exists():
        return [output_root]
    return sorted(
        path
        for path in output_root.iterdir()
        if path.is_dir() and (path / "segment_predicates.json").exists()
    )


def build_timeline_atoms_for_scenes(
    output_root: Path,
    scene_names: Optional[Sequence[str]] = None,
    top_k_objects: int = 3,
) -> List[Dict[str, Any]]:
    """Load scene predicates and build timeline atoms for each scene."""
    scene_paths = discover_scene_paths(output_root, scene_names=scene_names)
    timelines = []
    for scene_path in scene_paths:
        segment_predicates = load_segment_predicates(scene_path)
        timeline_atoms = predicates_to_timeline_atoms(
            segment_predicates,
            top_k_objects=top_k_objects,
        )
        timelines.append(
            {
                "scene_name": segment_predicates.get("scene_name", scene_path.name),
                "scene_path": str(scene_path),
                "num_segments": len(timeline_atoms),
                "timeline_atoms": timeline_atoms,
            }
        )
    return timelines


def save_timeline_atoms(timelines: Sequence[Dict[str, Any]], output_file: Path) -> Path:
    """Save timeline atoms as JSON for debugging the predicate-to-atom step."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(list(timelines), file, indent=2)
    return output_file


def print_atoms_debug_summary(
    timelines: Sequence[Dict[str, Any]],
    preview_segments: int = 3,
    preview_atoms: int = 12,
) -> None:
    """Print a compact summary of timeline atoms for debugging."""
    all_atoms = [
        atom
        for scene in timelines
        for segment_atoms in scene.get("timeline_atoms", [])
        for atom in segment_atoms
    ]
    unique_atoms = sorted(set(tuple(atom) for atom in all_atoms))
    print("\nAtom debug summary")
    print(f"Total atoms: {len(all_atoms)}")
    print(f"Unique atoms: {len(unique_atoms)}")

    for scene in timelines:
        timeline_atoms = scene.get("timeline_atoms", [])
        segment_lengths = [len(segment_atoms) for segment_atoms in timeline_atoms]
        mean_atoms = sum(segment_lengths) / len(segment_lengths) if segment_lengths else 0.0
        print(
            f"\nScene {scene.get('scene_name')}: "
            f"{len(timeline_atoms)} segments, "
            f"mean {mean_atoms:.1f} atoms/segment"
        )
        for segment_index, segment_atoms in enumerate(timeline_atoms[:preview_segments]):
            shown_atoms = segment_atoms[:preview_atoms]
            suffix = " ..." if len(segment_atoms) > preview_atoms else ""
            print(f"  segment {segment_index}: {shown_atoms}{suffix}")

    print("\nTop unique atoms preview:")
    for atom in unique_atoms[:preview_atoms]:
        print(f"  {atom}")


def format_atomic_rule(rule: Rule, support: int) -> Dict[str, Any]:
    """Convert a counted rule tuple into a JSON/CSV-friendly record."""
    antecedent, consequent, lag = rule
    return {
        "antecedent_key": antecedent[0],
        "antecedent_value": antecedent[1],
        "consequent_key": consequent[0],
        "consequent_value": consequent[1],
        "lag": int(lag),
        "support": int(support),
    }


def format_atomic_rules(rule_counts: Dict[Rule, int]) -> List[Dict[str, Any]]:
    """Return counted rules sorted by support."""
    return [
        format_atomic_rule(rule, support)
        for rule, support in sorted(
            rule_counts.items(),
            key=lambda item: (-item[1], item[0][2], item[0][0], item[0][1]),
        )
    ]


def mine_atomic_rules_for_scenes(
    timelines: Sequence[Dict[str, Any]],
    max_lag: int = 3,
) -> List[Dict[str, Any]]:
    """Mine atomic temporal rules for each scene timeline."""
    scene_rules = []
    for scene in timelines:
        rule_counts = mine_atomic_rules(scene.get("timeline_atoms", []), max_lag=max_lag)
        scene_rules.append(
            {
                "scene_name": scene.get("scene_name"),
                "num_rules": len(rule_counts),
                "rules": format_atomic_rules(rule_counts),
            }
        )
    return scene_rules


def print_rules_debug_summary(
    scene_rules: Sequence[Dict[str, Any]],
    preview_rules: int = 12,
) -> None:
    """Print a compact summary of generated atomic temporal rules."""
    total_rules = sum(int(scene.get("num_rules", 0)) for scene in scene_rules)
    print("\nAtomic temporal rules debug summary")
    print(f"Total unique scene-local rules: {total_rules}")
    for scene in scene_rules:
        print(f"\nScene {scene.get('scene_name')}: {scene.get('num_rules', 0)} rules")
        for rule in scene.get("rules", [])[:preview_rules]:
            print(
                "  "
                f"({rule['antecedent_key']}={rule['antecedent_value']}) "
                f"-> ({rule['consequent_key']}={rule['consequent_value']}) "
                f"dt={rule['lag']} support={rule['support']}"
            )


def _dedupe_atoms(atoms: Sequence[Atom]) -> List[Atom]:
    deduped = []
    seen = set()
    for key, value in atoms:
        atom = (str(key), value)
        if atom in seen:
            continue
        seen.add(atom)
        deduped.append(atom)
    return deduped


def _is_trivial_rule(antecedent: Atom, consequent: Atom) -> bool:
    return antecedent == consequent


def _is_valid_consequent(atom: Atom) -> bool:
    key, value = atom
    del value
    return key in {"ego_speed_change", "ego_is_moving", "ego_is_turning"}


def _is_excluded_consequent(atom: Atom) -> bool:
    """Return True for consequents we want to exclude from learning.

    Currently we exclude rules that predict the ego being `moving` because
    these are too generic (e.g., P -> ego_is_moving=moving) and not useful.
    """
    if not isinstance(atom, (list, tuple)) or len(atom) < 2:
        return False
    key = str(atom[0])
    value = atom[1]
    return key == "ego_is_moving" and value == "moving"


def _is_informative_antecedent(atom: Atom) -> bool:
    key, value = atom
    if key == "num_focus_objects":
        return False
    if key == "ego_motion_segment":
        return False
    # Exclude atoms whose value is explicitly 'uncertain' from antecedents.
    # These uncertain labels are not informative for antecedent patterns.
    try:
        if isinstance(value, str) and value.strip().lower() == "uncertain":
            return False
    except Exception:
        pass
    if key == "obj_relative_motion" and value == "moving_away":
        return False
    return True


def mine_atomic_rules(
    timeline_atoms: Sequence[Sequence[Atom]],
    max_lag: int = 3,
) -> Dict[Rule, int]:
    """Mine lagged atomic rules from one scene timeline."""
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1.")

    rule_counts: Dict[Rule, int] = defaultdict(int)
    timeline = [_dedupe_atoms(segment_atoms) for segment_atoms in timeline_atoms]

    for time_index, antecedents in enumerate(timeline):
        for lag in range(1, max_lag + 1):
            consequent_index = time_index + lag
            if consequent_index >= len(timeline):
                continue

            consequents = timeline[consequent_index]
            for antecedent in antecedents:
                if not _is_informative_antecedent(antecedent):
                    continue
                for consequent in consequents:
                    if not _is_valid_consequent(consequent):
                        continue
                    if _is_excluded_consequent(consequent):
                        continue
                    if _is_trivial_rule(antecedent, consequent):
                        continue
                    rule_counts[(antecedent, consequent, lag)] += 1

    return dict(rule_counts)


def _canonical_antecedent(atoms: Iterable[Atom]) -> Tuple[Atom, ...]:
    normalized = [(str(k), v) for k, v in atoms]
    normalized.sort(key=lambda a: (a[0], str(a[1])))
    return tuple(normalized)


def _antecedent_to_str(antecedent: Sequence[Atom]) -> str:
    return " | ".join(f"{k}={v}" for k, v in antecedent)


def mine_extended_rules_from_candidates(
    timeline_dedup: Sequence[Sequence[Atom]],
    candidates: Iterable[Tuple[Atom, ...]],
    max_lag: int = 3,
) -> Dict[Tuple[Tuple[Atom, ...], Atom, int], int]:
    rule_counts: Dict[Tuple[Tuple[Atom, ...], Atom, int], int] = defaultdict(int)
    for time_index, seg_atoms in enumerate(timeline_dedup):
        seg_set = set(seg_atoms)
        for candidate in candidates:
            if not all(atom in seg_set for atom in candidate):
                continue
            for lag in range(1, max_lag + 1):
                consequent_index = time_index + lag
                if consequent_index >= len(timeline_dedup):
                    continue
                consequents = timeline_dedup[consequent_index]
                for consequent in consequents:
                    if not _is_valid_consequent(consequent):
                        continue
                    if _is_excluded_consequent(consequent):
                        continue
                    if consequent in candidate:
                        continue
                    rule_counts[(candidate, consequent, lag)] += 1
    return dict(rule_counts)


def compute_antecedent_valid_counts_for_candidates(
    timeline_dedup: Sequence[Sequence[Atom]],
    candidates: Iterable[Tuple[Atom, ...]],
    max_lag: int = 3,
) -> Dict[Tuple[Tuple[Atom, ...], int], int]:
    counts: Dict[Tuple[Tuple[Atom, ...], int], int] = defaultdict(int)
    for time_index, seg_atoms in enumerate(timeline_dedup):
        seg_set = set(seg_atoms)
        for candidate in candidates:
            if not all(atom in seg_set for atom in candidate):
                continue
            for lag in range(1, max_lag + 1):
                if time_index + lag < len(timeline_dedup):
                    counts[(candidate, lag)] += 1
    return dict(counts)


def format_extended_rule(rule: Tuple[Tuple[Atom, ...], Atom, int], support: int) -> Dict[str, Any]:
    antecedent, consequent, lag = rule
    return {
        "antecedent_atoms": [{"key": a[0], "value": a[1]} for a in antecedent],
        "antecedent_length": int(len(antecedent)),
        "antecedent_str": _antecedent_to_str(antecedent),
        "consequent_key": consequent[0],
        "consequent_value": consequent[1],
        "lag": int(lag),
        "support": int(support),
    }


def format_extended_rules(rule_counts: Dict[Tuple[Tuple[Atom, ...], Atom, int], int]) -> List[Dict[str, Any]]:
    return [
        format_extended_rule(rule, support)
        for rule, support in sorted(
            rule_counts.items(),
            key=lambda item: (-item[1], len(item[0][0]), item[0][0], item[0][1]),
        )
    ]


def _save_init_rules(scene_rule_counts: Dict[Rule, int], scene_out: Path):
    init_sorted = sorted(
        scene_rule_counts.items(),
        key=lambda item: (-item[1], item[0][2], item[0][0], item[0][1]),
    )
    init_formatted = [format_atomic_rule(rule, support) for rule, support in init_sorted]

    init_json = scene_out / "init_atomic_rules.json"
    with init_json.open("w", encoding="utf-8") as jf:
        json.dump(init_formatted, jf, indent=2)

    init_csv = scene_out / "init_atomic_rules.csv"
    with init_csv.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "antecedent_key",
            "antecedent_value",
            "consequent_key",
            "consequent_value",
            "lag",
            "support",
        ])
        for r in init_formatted:
            writer.writerow(
                [
                    r.get("antecedent_key"),
                    r.get("antecedent_value"),
                    r.get("consequent_key"),
                    r.get("consequent_value"),
                    r.get("lag"),
                    r.get("support"),
                ]
            )

    return init_formatted, init_json, init_csv


def _compute_antecedent_valid_counts(timeline_dedup: Sequence[Sequence[Atom]], max_lag: int) -> Dict[Tuple[Atom, int], int]:
    counts: Dict[Tuple[Atom, int], int] = defaultdict(int)
    for time_index, antecedents in enumerate(timeline_dedup):
        for lag_val in range(1, max_lag + 1):
            consequent_index = time_index + lag_val
            if consequent_index >= len(timeline_dedup):
                continue
            for antecedent in antecedents:
                if not _is_informative_antecedent(antecedent):
                    continue
                counts[(antecedent, lag_val)] += 1
    return counts


def _compute_p_without_q(scene_rule_counts: Dict[Rule, int], antecedent_valid_counts: Dict[Tuple[Atom, int], int]) -> Dict[Rule, int]:
    p_without_q_counts: Dict[Rule, int] = {}
    for rule, support in scene_rule_counts.items():
        antecedent, consequent, lag_val = rule
        valid_count = antecedent_valid_counts.get((antecedent, lag_val), 0)
        p_without_q = int(valid_count) - int(support)
        if p_without_q < 0:
            p_without_q = 0
        p_without_q_counts[rule] = p_without_q
    return p_without_q_counts


def _format_scene_rules(scene_rule_counts: Dict[Rule, int], antecedent_valid_counts: Dict[Tuple[Atom, int], int], p_without_q_counts: Dict[Rule, int]) -> List[Dict[str, Any]]:
    sorted_rules = sorted(
        scene_rule_counts.items(),
        key=lambda item: (-item[1], item[0][2], item[0][0], item[0][1]),
    )
    formatted_rules: List[Dict[str, Any]] = []
    for rule, support in sorted_rules:
        antecedent, consequent, lag_val = rule
        rec = format_atomic_rule(rule, support)
        valid_count = antecedent_valid_counts.get((antecedent, lag_val), 0)
        rec["valid_antecedent_count"] = int(valid_count)
        rec["p_without_q"] = int(p_without_q_counts.get(rule, 0))
        formatted_rules.append(rec)
    return formatted_rules


def _save_scene_rules(formatted_rules: List[Dict[str, Any]], scene_out: Path):
    scene_json = scene_out / "atomic_rules.json"
    with scene_json.open("w", encoding="utf-8") as jf:
        json.dump(formatted_rules, jf, indent=2)

    scene_csv = scene_out / "atomic_rules.csv"
    with scene_csv.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "antecedent_key",
            "antecedent_value",
            "consequent_key",
            "consequent_value",
            "lag",
            "support",
            "valid_antecedent_count",
            "p_without_q",
        ])
        for r in formatted_rules:
            writer.writerow(
                [
                    r.get("antecedent_key"),
                    r.get("antecedent_value"),
                    r.get("consequent_key"),
                    r.get("consequent_value"),
                    r.get("lag"),
                    r.get("support"),
                    r.get("valid_antecedent_count", 0),
                    r.get("p_without_q", 0),
                ]
            )
    return scene_json, scene_csv


def _build_atom_counts(timeline_atoms: Sequence[Sequence[Atom]]) -> Tuple[Dict[Atom, int], Dict[Atom, int]]:
    antecedent_counts: Dict[Atom, int] = defaultdict(int)
    consequent_counts: Dict[Atom, int] = defaultdict(int)
    for seg_atoms in timeline_atoms:
        deduped = _dedupe_atoms(seg_atoms)
        for atom in deduped:
            antecedent_counts[atom] += 1
            consequent_counts[atom] += 1
    return dict(antecedent_counts), dict(consequent_counts)


def _extend_rules_iterative(
    scene_rule_counts: Dict[Rule, int],
    timeline_dedup: Sequence[Sequence[Atom]],
    antecedent_valid_counts: Dict[Tuple[Atom, int], int],
    scene_out: Path,
    max_lag: int,
    extend_iters: int,
) -> Dict[int, List[Dict[str, Any]]]:
    """Iteratively extend antecedents by one atom per iteration.

    Parameters added for future filtering/pruning (kept disabled by default):
    - enable_filtering: if True, apply thresholds and optional beam pruning
      to select which antecedents survive to the next iteration.
    - min_total_support/min_confidence/beam_width: filtering params.

    The function currently performs no pruning unless `enable_filtering`
    is explicitly set to True (interface-ready but inert by default).
    """
    extended_rules_by_length: Dict[int, List[Dict[str, Any]]] = {}

    # init atoms: single-atom antecedents that appear in scene_rule_counts
    init_atom_set = set()
    for (ant, cons, lag), sup in scene_rule_counts.items():
        init_atom_set.add((str(ant[0]), ant[1]))

    # length-1 rules (mirror of atomic rules)
    current_antecedents = set()
    rule_counts_len1: Dict[Tuple[Tuple[Atom, ...], Atom, int], int] = {}
    for (ant, cons, lag), sup in scene_rule_counts.items():
        ant_tuple = _canonical_antecedent([ant])
        rule_counts_len1[(ant_tuple, cons, lag)] = int(sup)
        if int(sup) > 0:
            current_antecedents.add(ant_tuple)

    formatted_len1 = []
    for (ant_tuple, cons, lag), sup in sorted(rule_counts_len1.items(), key=lambda item: (-item[1], item[0][2], item[0][0], item[0][1])):
        rec = format_extended_rule((ant_tuple, cons, lag), sup)
        valid_count = antecedent_valid_counts.get((ant_tuple[0], lag), 0) if ant_tuple and len(ant_tuple) == 1 else 0
        rec["valid_antecedent_count"] = int(valid_count)
        rec["p_without_q"] = max(0, int(valid_count) - int(sup))
        formatted_len1.append(rec)
    extended_rules_by_length[1] = formatted_len1

    # iterate to produce lengths 2..(1+extend_iters)
    for length in range(2, 1 + int(extend_iters)):
        candidates: set[Tuple[Atom, ...]] = set()
        for ant_tuple in current_antecedents:
            for atom in init_atom_set:
                if atom in ant_tuple:
                    continue
                new_ant = _canonical_antecedent(list(ant_tuple) + [atom])
                candidates.add(new_ant)

        if not candidates:
            break

        # Report counts before extension: number of antecedents of the previous
        # length and how many candidate antecedents will be evaluated.
        prev_len_rules = len(extended_rules_by_length.get(length - 1, []))
        print(
            f"Extension iter (len={length}): previous_length_rules={prev_len_rules}, candidates={len(candidates)}"
        )

        ext_rule_counts = mine_extended_rules_from_candidates(timeline_dedup, candidates, max_lag=max_lag)
        antecedent_valid_counts_ext = compute_antecedent_valid_counts_for_candidates(timeline_dedup, candidates, max_lag=max_lag)

        formatted_ext = []
        sorted_items = sorted(ext_rule_counts.items(), key=lambda item: (-item[1], len(item[0][0]), item[0][0]))
        for (ant_tuple, cons, lag), sup in sorted_items:
            rec = format_extended_rule((ant_tuple, cons, lag), sup)
            valid_count = antecedent_valid_counts_ext.get((ant_tuple, lag), 0)
            rec["valid_antecedent_count"] = int(valid_count)
            rec["p_without_q"] = max(0, int(valid_count) - int(sup))
            formatted_ext.append(rec)

        ext_json = scene_out / f"extended_rules_len_{length}.json"
        with ext_json.open("w", encoding="utf-8") as jf:
            json.dump(formatted_ext, jf, indent=2)

        ext_csv = scene_out / f"extended_rules_len_{length}.csv"
        with ext_csv.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow([
                "antecedent_atoms",
                "antecedent_length",
                "consequent_key",
                "consequent_value",
                "lag",
                "support",
                "valid_antecedent_count",
                "p_without_q",
            ])
            for r in formatted_ext:
                writer.writerow([
                    r.get("antecedent_str"),
                    r.get("antecedent_length"),
                    r.get("consequent_key"),
                    r.get("consequent_value"),
                    r.get("lag"),
                    r.get("support"),
                    r.get("valid_antecedent_count", 0),
                    r.get("p_without_q", 0),
                ])

        extended_rules_by_length[length] = formatted_ext

        # Report how many extended rules were discovered for this iteration.
        after_count = len(formatted_ext)
        print(f"Extension iter (len={length}) completed: extended_rules_found={after_count}")

        # === Filtering interface (disabled by default) ===
        # By default we keep antecedents that produced any extended rules;
        # if filtering is enabled, apply support/confidence/beam filters.
        # NOTE: keep the interface inert unless callers opt-in by passing
        # enable_filtering=True and appropriate thresholds.
        enable_filtering = False
        min_total_support = 2
        min_confidence = 0.5
        beam_width: Optional[int] = None

        if enable_filtering:
            from collections import defaultdict

            # Aggregate total support per antecedent across consequents/lags
            total_support_by_ant = defaultdict(int)
            for (ant, cons, lag), sup in ext_rule_counts.items():
                total_support_by_ant[ant] += int(sup)

            # Aggregate valid counts per antecedent (summed across lags)
            valid_count_by_ant = defaultdict(int)
            for (ant, lag), cnt in antecedent_valid_counts_ext.items():
                valid_count_by_ant[ant] += int(cnt)

            # Select antecedents meeting thresholds
            selected = []
            for ant, sup in total_support_by_ant.items():
                valid = valid_count_by_ant.get(ant, 0)
                conf = (float(sup) / float(valid)) if valid > 0 else 0.0
                if sup >= int(min_total_support) and conf >= float(min_confidence):
                    selected.append((ant, sup))

            # Beam/top-k trimming
            if beam_width is not None:
                selected.sort(key=lambda it: -it[1])
                kept_ants = [it[0] for it in selected[: int(beam_width)]]
            else:
                kept_ants = [it[0] for it in selected]

            current_antecedents = set(kept_ants)
            print(f"Filtering enabled: kept antecedents={len(current_antecedents)} from {len(total_support_by_ant)}")
        else:
            current_antecedents = set(ant for (ant, _, _) in ext_rule_counts.keys())

    return extended_rules_by_length


def _process_scene(scene: Dict[str, Any], output_root: Path, max_lag: int, top_k_objects: int, extend_iters: int) -> Dict[str, Any]:
    """Process one scene: mine rules, compute diagnostics, save outputs.

    This helper performs the per-scene work for pattern mining and returns a
    compact summary dict that includes counts and paths to saved files.
    """

    # Resolve the scene name and ensure the scene output directory exists.
    scene_name = scene.get("scene_name") or Path(scene.get("scene_path", "")).name
    scene_out = Path(output_root) / scene_name
    scene_out.mkdir(parents=True, exist_ok=True)

    # Timeline atoms for this scene (list of segments -> list of atoms).
    timeline_atoms = scene.get("timeline_atoms", [])

    # Mine atomic rules (single-atom antecedents) for the current scene.
    scene_rule_counts = mine_atomic_rules(timeline_atoms, max_lag=max_lag)

    # Save the initial atomic rule set (rules that occur >= 1) for inspection.
    init_formatted, init_json, init_csv = _save_init_rules(scene_rule_counts, scene_out)

    # Deduplicate atoms per segment to avoid double-counting within a segment.
    timeline_dedup = [_dedupe_atoms(seg) for seg in timeline_atoms]

    # Compute how many antecedent occurrences have a valid consequent slot
    # (i.e., exclude antecedents that are too close to the timeline end).
    antecedent_valid_counts = _compute_antecedent_valid_counts(timeline_dedup, max_lag)

    # For diagnostics: how many times P occurred without Q following.
    p_without_q_counts = _compute_p_without_q(scene_rule_counts, antecedent_valid_counts)

    # Format rules with diagnostics and persist per-scene JSON/CSV files.
    formatted_rules = _format_scene_rules(scene_rule_counts, antecedent_valid_counts, p_without_q_counts)
    scene_json, scene_csv = _save_scene_rules(formatted_rules, scene_out)

    # Build per-scene atom counts (deduped per segment) for later aggregation/scoring.
    antecedent_counts, consequent_counts = _build_atom_counts(timeline_atoms)

    # Optionally run iterative antecedent extension to generate longer antecedent rules.
    extended_rules_by_length: Dict[int, List[Dict[str, Any]]] = {}
    if int(extend_iters or 0) > 0:
        extended_rules_by_length = _extend_rules_iterative(
            scene_rule_counts, timeline_dedup, antecedent_valid_counts, scene_out, max_lag, int(extend_iters)
        )

    # Log saved outputs for visibility.
    print(f"Saved scene-level atomic rules to {scene_json} and {scene_csv}")
    print(f"Saved initial atomic rule set to {init_json} and {init_csv}")
    if extended_rules_by_length:
        print(f"Saved extended rule sets for lengths: {list(extended_rules_by_length.keys())}")

    # Return a compact summary of results and saved file paths.
    return {
        "scene_name": scene_name,
        "num_segments": int(scene.get("num_segments", 0)),
        "rule_counts": scene_rule_counts,
        "antecedent_counts": dict(antecedent_counts),
        "consequent_counts": dict(consequent_counts),
        "json": str(scene_json),
        "csv": str(scene_csv),
        "init_rules": init_formatted,
        "init_json": str(init_json),
        "init_csv": str(init_csv),
        "extended_rules": extended_rules_by_length,
    }


def aggregate_global_rule_counts(
    timelines: Sequence[Dict[str, Any]],
    max_lag: int = 3,
):
    """Aggregate scene-local rule counts and atom counts into global totals.

    Returns: (global_rule_counts, global_P_counts, global_Q_counts, rule_video_support, total_segments, num_scenes)
    """
    global_rule_counts: Dict[Rule, int] = defaultdict(int)
    global_P_counts: Dict[Atom, int] = defaultdict(int)
    global_Q_counts: Dict[Atom, int] = defaultdict(int)
    rule_video_support: Dict[Rule, set] = defaultdict(set)

    total_segments = 0
    num_scenes = len(timelines)

    for scene in timelines:
        scene_name = scene.get("scene_name")
        timeline_atoms = scene.get("timeline_atoms", [])
        # scene-local rules
        scene_rule_counts = mine_atomic_rules(timeline_atoms, max_lag=max_lag)
        for rule, count in scene_rule_counts.items():
            global_rule_counts[rule] += int(count)
            rule_video_support[rule].add(scene_name)

        # atom counts (per-segment deduped)
        for seg_atoms in timeline_atoms:
            deduped = _dedupe_atoms(seg_atoms)
            for atom in deduped:
                global_P_counts[atom] += 1
                global_Q_counts[atom] += 1

        total_segments += len(timeline_atoms)

    return dict(global_rule_counts), dict(global_P_counts), dict(global_Q_counts), rule_video_support, int(total_segments), int(num_scenes)


def score_rules(
    rule_counts: Dict[Rule, int],
    antecedent_counts: Dict[Atom, int],
    consequent_counts: Dict[Atom, int],
    total_segments: int,
    num_scenes: int,
    rule_video_support: Optional[Dict[Rule, set]] = None,
) -> List[Dict[str, Any]]:
    """Score mined rules by multiple metrics and return scored records.

    New composite score is computed as:
      score = support_score * lift_score * stability_score * novelty_score

    where
      support_score = support / total_segments
      lift_score = lift (confidence / P(Q))
      stability_score = (#videos containing rule) / num_scenes
      novelty_score = P(Q|P) - P(Q) = confidence - p_q

    Returns a list of scored rule records sorted by `score` descending.
    """
    eps = 1e-9
    if total_segments == 0 or num_scenes == 0:
        return []

    scored_rules: List[Dict[str, Any]] = []
    for rule, support in rule_counts.items():
        antecedent, consequent, lag = rule
        antecedent_count = antecedent_counts.get(antecedent, 0)
        if antecedent_count == 0:
            continue

        # basic probabilities
        confidence = float(support) / float(antecedent_count)
        p_q = float(consequent_counts.get(consequent, 0)) / float(total_segments)

        # lift and stability
        lift = float(confidence) / (p_q + eps)
        log_lift = math.log(lift + eps)
        stability = (
            float(len(rule_video_support.get(rule, set()))) / float(num_scenes)
            if rule_video_support is not None
            else 0.0
        )

        # component scores
        support_score = float(support) / float(total_segments)
        lift_score = float(lift)
        stability_score = float(stability)
        novelty_score = float(confidence - p_q)

        # composite score (may be negative if novelty_score < 0)
        composite_score = support_score * lift_score * stability_score * novelty_score

        scored_rules.append(
            {
                "rule": (antecedent, consequent, lag),
                "antecedent_key": antecedent[0],
                "antecedent_value": antecedent[1],
                "consequent_key": consequent[0],
                "consequent_value": consequent[1],
                "lag": int(lag),
                "support": int(support),
                "confidence": float(confidence),
                "p_q": float(p_q),
                "lift": float(lift),
                "log_lift": float(log_lift),
                "stability": float(stability),
                "support_score": float(support_score),
                "lift_score": float(lift_score),
                "stability_score": float(stability_score),
                "novelty_score": float(novelty_score),
                "score": float(composite_score),
            }
        )

    scored_rules.sort(key=lambda r: (-r.get("score", 0.0), -r.get("support", 0)))
    return scored_rules


def filter_rules(
    scored_rules: Sequence[Dict[str, Any]],
    min_support: int = 4,
    min_confidence: float = 0.5,
    min_lift: float = 1.0,
    min_stability: float = 0.0,
) -> List[Dict[str, Any]]:
    """Filter scored rules using configurable thresholds."""
    filtered: List[Dict[str, Any]] = []
    for r in scored_rules:
        if (
            int(r.get("support", 0)) >= int(min_support)
            and float(r.get("confidence", 0.0)) >= float(min_confidence)
            and float(r.get("lift", 0.0)) >= float(min_lift)
            and float(r.get("stability", 0.0)) >= float(min_stability)
        ):
            filtered.append(r)
    return filtered


def aggregate_and_score(
    timelines: Sequence[Dict[str, Any]],
    max_lag: int = 3,
    output_root: Optional[Path] = None,
    min_support: int = 4,
    min_confidence: float = 0.5,
    min_lift: float = 1.0,
    min_stability: float = 0.0,
    save_prefix: str = "patterns_atomic_rules",
) -> Dict[str, Any]:
    """Aggregate scene rules globally, score them, filter, and optionally save outputs.

    Returns a dict with global counts, scored rules, filtered rules and metadata.
    """
    (
        global_rule_counts,
        global_P_counts,
        global_Q_counts,
        rule_video_support,
        total_segments,
        num_scenes,
    ) = aggregate_global_rule_counts(timelines, max_lag=max_lag)

    scored = score_rules(
        global_rule_counts,
        global_P_counts,
        global_Q_counts,
        total_segments,
        num_scenes,
        rule_video_support=rule_video_support,
    )

    filtered = filter_rules(
        scored, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift, min_stability=min_stability
    )

    results = {
        "global_rule_counts": global_rule_counts,
        "scored_rules": scored,
        "filtered_rules": filtered,
        "rule_video_support": {str(rule): list(scenes) for rule, scenes in rule_video_support.items()},
        "total_segments": total_segments,
        "num_scenes": num_scenes,
    }

    if output_root is not None:
        out_dir = Path(output_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        # JSON summary: ensure keys are JSON-serializable (format atomic rules)
        json_path = out_dir / f"{save_prefix}.json"
        json_results = dict(results)
        try:
            json_results["global_rule_counts"] = format_atomic_rules(global_rule_counts)
        except Exception:
            json_results["global_rule_counts"] = {}
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(json_results, jf, indent=2)
        # CSV of filtered rules
        csv_path = out_dir / f"{save_prefix}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow([
                "antecedent_key",
                "antecedent_value",
                "consequent_key",
                "consequent_value",
                "lag",
                "support",
                "confidence",
                "lift",
                "log_lift",
                "stability",
                "score",
            ])
            for r in filtered:
                writer.writerow(
                    [
                        r.get("antecedent_key"),
                        r.get("antecedent_value"),
                        r.get("consequent_key"),
                        r.get("consequent_value"),
                        r.get("lag"),
                        r.get("support"),
                        r.get("confidence"),
                        r.get("lift"),
                        r.get("log_lift"),
                        r.get("stability"),
                        r.get("score"),
                    ]
                )

    return results


def aggregate_mixed_length_rules(
    per_scene_results: Sequence[Dict[str, Any]],
    output_root: Optional[Path] = None,
    save_prefix: str = "patterns_all_rules",
) -> Dict[str, Any]:
    """Aggregate rules of varying antecedent lengths across scenes and save CSV/JSON.

    The function combines per-scene atomic rules and any per-scene extended
    rules (stored in `per_scene_results[*]["extended_rules"]`) into a single
    aggregated table keyed by (antecedent_tuple, consequent, lag). Aggregated
    fields include summed `support`, summed `valid_antecedent_count`, summed
    `p_without_q`, and the list of scenes in which the rule appears.
    """
    aggregated: Dict[Tuple[Tuple[Tuple[str, Any], ...], Tuple[str, Any], int], Dict[str, Any]] = {}

    for scene in per_scene_results:
        scene_name = scene.get("scene_name")

        # 1) Load per-scene atomic rules (if JSON available) to include length-1 rules
        scene_json_path = scene.get("json")
        if scene_json_path:
            try:
                with open(scene_json_path, "r", encoding="utf-8") as jf:
                    scene_rules = json.load(jf)
            except Exception:
                scene_rules = []
        else:
            scene_rules = []

        for rec in scene_rules:
            # skip excluded consequents like ego_is_moving=moving
            if rec.get("consequent_key") == "ego_is_moving" and rec.get("consequent_value") == "moving":
                continue
            ant_tuple = ((rec.get("antecedent_key"), rec.get("antecedent_value")),)
            cons = (rec.get("consequent_key"), rec.get("consequent_value"))
            lag = int(rec.get("lag", 0))
            key = (ant_tuple, cons, lag)
            entry = aggregated.setdefault(key, {
                "antecedent_atoms": [
                    {"key": ant_tuple[0][0], "value": ant_tuple[0][1]}
                ],
                "antecedent_length": 1,
                "antecedent_str": f"{ant_tuple[0][0]}={ant_tuple[0][1]}",
                "consequent_key": cons[0],
                "consequent_value": cons[1],
                "lag": int(lag),
                "support": 0,
                "valid_antecedent_count": 0,
                "p_without_q": 0,
                "scenes": set(),
            })
            entry["support"] += int(rec.get("support", 0))
            entry["valid_antecedent_count"] += int(rec.get("valid_antecedent_count", 0))
            entry["p_without_q"] += int(rec.get("p_without_q", 0))
            entry["scenes"].add(scene_name)

        # 2) Include any extended rules saved in per-scene results
        ext = scene.get("extended_rules") or {}
        for length, rules in (ext.items() if isinstance(ext, dict) else []):
            for rec in rules:
                # skip excluded consequents like ego_is_moving=moving
                if rec.get("consequent_key") == "ego_is_moving" and rec.get("consequent_value") == "moving":
                    continue
                ants = rec.get("antecedent_atoms", [])
                ant_tuple = tuple((a.get("key"), a.get("value")) for a in ants)
                cons = (rec.get("consequent_key"), rec.get("consequent_value"))
                lag = int(rec.get("lag", 0))
                key = (ant_tuple, cons, lag)
                entry = aggregated.setdefault(key, {
                    "antecedent_atoms": [{"key": k, "value": v} for k, v in ant_tuple],
                    "antecedent_length": int(rec.get("antecedent_length", len(ant_tuple))),
                    "antecedent_str": rec.get("antecedent_str", " | ".join(f"{k}={v}" for k, v in ant_tuple)),
                    "consequent_key": cons[0],
                    "consequent_value": cons[1],
                    "lag": int(lag),
                    "support": 0,
                    "valid_antecedent_count": 0,
                    "p_without_q": 0,
                    "scenes": set(),
                })
                entry["support"] += int(rec.get("support", 0))
                entry["valid_antecedent_count"] += int(rec.get("valid_antecedent_count", 0))
                entry["p_without_q"] += int(rec.get("p_without_q", 0))
                entry["scenes"].add(scene_name)

    # Prepare output list and sort by support desc then antecedent length asc
    out_list = []
    for key, v in aggregated.items():
        v_copy = dict(v)
        v_copy["scenes"] = sorted(list(v_copy["scenes"]))
        out_list.append(v_copy)

    out_list.sort(key=lambda x: (-int(x.get("support", 0)), int(x.get("antecedent_length", 0)), x.get("antecedent_str", "")))

    results = {"num_rules": len(out_list), "rules": out_list}

    if output_root is not None:
        out_dir = Path(output_root)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{save_prefix}.json"
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=2)

        csv_path = out_dir / f"{save_prefix}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow([
                "antecedent_length",
                "antecedent_str",
                "antecedent_atoms",
                "consequent_key",
                "consequent_value",
                "lag",
                "support",
                "valid_antecedent_count",
                "p_without_q",
                "scenes",
            ])
            for r in out_list:
                ants_str = r.get("antecedent_str", "")
                ants_atoms = " | ".join(f"{a['key']}={a['value']}" for a in r.get("antecedent_atoms", []))
                writer.writerow([
                    r.get("antecedent_length"),
                    ants_str,
                    ants_atoms,
                    r.get("consequent_key"),
                    r.get("consequent_value"),
                    r.get("lag"),
                    r.get("support"),
                    r.get("valid_antecedent_count", 0),
                    r.get("p_without_q", 0),
                    ";".join(r.get("scenes", [])),
                ])

    return results


def run_pattern_mining(
    output_root: Optional[Path] = None,
    scene_names: Optional[Sequence[str]] = None,
    max_lag: int = 3,
    top_k_objects: int = 3,
    extend_iters: int = 0,
) -> List[Dict[str, Any]]:
    """Entry point for nuScenes predicate-to-pattern mining."""
    if output_root is None:
        output_root = Path("pipeline_output") / "nuScenes_from_config"
    output_root = Path(output_root)

    timelines = build_timeline_atoms_for_scenes(
        Path(output_root),
        scene_names=scene_names,
        top_k_objects=top_k_objects,
    )

    per_scene_results: List[Dict[str, Any]] = []
    for scene in timelines:
        per_scene_results.append(
            _process_scene(scene, Path(output_root), max_lag=int(max_lag), top_k_objects=int(top_k_objects), extend_iters=int(extend_iters))
        )

    return timelines, per_scene_results


def _parse_cli(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Build nuScenes timeline atoms from segment predicate files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file. Compatible with configs/exp_nuScenes/default.yaml.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root containing scene folders, or a single scene folder.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Scene name to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--top-k-objects",
        type=int,
        default=None,
        help="Number of nearest objects to include as object atoms per segment. Use 0 to include all.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=None,
        help="Maximum segment lag dt used when generating atomic temporal rules.",
    )
    parser.add_argument(
        "--save-atoms",
        type=Path,
        default=None,
        help="Optional JSON path for saving built timeline atoms.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Print atom counts and previews after building timeline atoms.",
    )
    parser.add_argument(
        "--extend-iters",
        type=int,
        default=None,
        help="Number of extension iterations to run per scene (0=no extension).",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=None,
        help="Minimum support count for filtered rules (overrides config).",
    )
    parser.add_argument(
        "--min-scene-support",
        type=int,
        default=None,
        help="Minimum number of distinct scenes a rule must appear in (overrides config).",
    )
    args = parser.parse_args(argv)
    run_cfg = resolve_run_config(args)

    output_root = _resolve_project_path(run_cfg.get("output_root"))
    save_atoms = _resolve_project_path(run_cfg.get("save_atoms"))
    scene_names = run_cfg.get("scene_names") or None
    return args, run_cfg, output_root, save_atoms, scene_names


def _run_and_report(output_root: Optional[Path], scene_names: Optional[Sequence[str]], run_cfg: Dict[str, Any]):
    timelines, per_scene_results = run_pattern_mining(
        output_root=output_root,
        scene_names=scene_names,
        max_lag=int(run_cfg["max_lag"]),
        top_k_objects=int(run_cfg["top_k_objects"]),
        extend_iters=int(run_cfg.get("extend_iters", 0)),
    )

    # Build a scene_rules view for debug printing (same shape as mine_atomic_rules_for_scenes)
    scene_rules = [
        {"scene_name": r["scene_name"], "num_rules": len(r["rule_counts"]), "rules": format_atomic_rules(r["rule_counts"])}
        for r in per_scene_results
    ]

    total_segments = sum(int(r["num_segments"]) for r in per_scene_results)
    print(f"Built timeline atoms for {len(per_scene_results)} scene(s), {total_segments} segment(s).")
    for r in per_scene_results:
        print(f"- {r['scene_name']}: {r['num_segments']} segments")
    print(f"Generated atomic temporal rules with max_lag={int(run_cfg['max_lag'])}.")

    if run_cfg.get("debug"):
        print_atoms_debug_summary(timelines)
        print_rules_debug_summary(scene_rules)

    return timelines, per_scene_results, total_segments


def _maybe_save_atoms(save_atoms: Optional[Path], timelines: Sequence[Dict[str, Any]]):
    if save_atoms is not None:
        saved_path = save_timeline_atoms(timelines, save_atoms)
        print(f"Saved timeline atoms to {saved_path}")


def _aggregate_from_per_scene(per_scene_results: Sequence[Dict[str, Any]], total_segments: int, output_root: Optional[Path], run_cfg: Dict[str, Any]):
    global_rule_counts: Dict[Rule, int] = defaultdict(int)
    global_P_counts: Dict[Atom, int] = defaultdict(int)
    global_Q_counts: Dict[Atom, int] = defaultdict(int)
    rule_video_support: Dict[Rule, set] = defaultdict(set)

    for r in per_scene_results:
        scene_name = r["scene_name"]
        for rule, cnt in r.get("rule_counts", {}).items():
            global_rule_counts[rule] += int(cnt)
            rule_video_support[rule].add(scene_name)
        for atom, cnt in r.get("antecedent_counts", {}).items():
            key = tuple(atom) if isinstance(atom, (list, tuple)) else atom
            global_P_counts[key] += int(cnt)
            global_Q_counts[key] += int(cnt)

    num_scenes = len(per_scene_results)

    scored = score_rules(
        global_rule_counts,
        global_P_counts,
        global_Q_counts,
        total_segments,
        num_scenes,
        rule_video_support=rule_video_support,
    )

    # Read filtering hyperparameters from run config (defaults set in DEFAULT_PATTERN_MINING_CONFIG)
    min_support = int(run_cfg.get("min_support", 4))
    min_confidence = float(run_cfg.get("min_confidence", 0.5))
    min_lift = float(run_cfg.get("min_lift", 1.0))
    min_stability = float(run_cfg.get("min_stability", 0.0))
    min_scene_support = int(run_cfg.get("min_scene_support", 3))

    # Apply numeric threshold filtering first
    filtered = filter_rules(
        scored,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        min_stability=min_stability,
    )

    # Then require rule to appear in at least `min_scene_support` distinct scenes
    if int(min_scene_support) > 1 and rule_video_support:
        filtered_by_scene: List[Dict[str, Any]] = []
        for r in filtered:
            rule_key = r.get("rule")
            if rule_key is None:
                continue
            scenes = rule_video_support.get(rule_key, set())
            if len(scenes) >= int(min_scene_support):
                filtered_by_scene.append(r)
        filtered = filtered_by_scene

    # Save aggregated results under output_root
    if output_root is not None:
        out_dir = Path(output_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "patterns_atomic_rules.json"
        results = {
            "global_rule_counts": format_atomic_rules(global_rule_counts),
            "scored_rules": scored,
            "filtered_rules": filtered,
            "rule_video_support": {str(rule): list(scenes) for rule, scenes in rule_video_support.items()},
            "total_segments": total_segments,
            "num_scenes": num_scenes,
        }
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=2)

        csv_path = out_dir / "patterns_atomic_rules.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow([
                "antecedent_key",
                "antecedent_value",
                "consequent_key",
                "consequent_value",
                "lag",
                "support",
                "confidence",
                "lift",
                "log_lift",
                "stability",
                "score",
            ])
            for r in filtered:
                writer.writerow(
                    [
                        r.get("antecedent_key"),
                        r.get("antecedent_value"),
                        r.get("consequent_key"),
                        r.get("consequent_value"),
                        r.get("lag"),
                        r.get("support"),
                        r.get("confidence"),
                        r.get("lift"),
                        r.get("log_lift"),
                        r.get("stability"),
                        r.get("score"),
                    ]
                )

        # Additionally aggregate mixed-length extended rules across scenes and
        # save a unified CSV/JSON that contains rules of all antecedent lengths.
        try:
            mixed_results = aggregate_mixed_length_rules(per_scene_results, output_root=output_root, save_prefix="patterns_all_rules")

            # Print per-length rule counts and simple quality statistics
            length_buckets = defaultdict(list)
            for rec in mixed_results.get("rules", []):
                length = int(rec.get("antecedent_length", 1))
                support = int(rec.get("support", 0))
                valid = int(rec.get("valid_antecedent_count", 0))
                p_without_q = int(rec.get("p_without_q", 0))
                confidence = float(support) / float(valid) if valid > 0 else 0.0
                length_buckets[length].append({
                    "support": support,
                    "confidence": confidence,
                    "p_without_q": p_without_q,
                    "antecedent_str": rec.get("antecedent_str", ""),
                    "consequent": f"{rec.get('consequent_key')}={rec.get('consequent_value')}",
                })

            if length_buckets:
                print("\nMixed-length aggregation summary:")
                for length in sorted(length_buckets.keys()):
                    items = length_buckets[length]
                    count = len(items)
                    mean_support = sum(i["support"] for i in items) / float(count) if count else 0.0
                    mean_conf = sum(i["confidence"] for i in items) / float(count) if count else 0.0
                    mean_p_without_q = sum(i["p_without_q"] for i in items) / float(count) if count else 0.0
                    print(f"  length={length}: count={count}, mean_support={mean_support:.2f}, mean_confidence={mean_conf:.3f}, mean_p_without_q={mean_p_without_q:.2f}")

                    # show top-3 by support
                    topk = sorted(items, key=lambda x: -x["support"])[:3]
                    for idx, it in enumerate(topk, start=1):
                        print(f"    top{idx}: {it['antecedent_str']} -> {it['consequent']} (support={it['support']}, conf={it['confidence']:.3f})")
            else:
                print("No mixed-length rules produced.")

            # Confidence distribution with step=0.2 (overall and per-antecedent-length)
            try:
                rules = mixed_results.get("rules", []) if mixed_results else []
                if rules:
                    step = 0.2
                    bins = int(1.0 / step)
                    # build labels like [0.0-0.2), ..., [0.8-1.0]
                    labels = [f"[{i*step:.1f}-{(i+1)*step:.1f})" for i in range(bins - 1)] + [f"[{(bins-1)*step:.1f}-{bins*step:.1f}]"]

                    overall_hist = [0 for _ in range(bins)]
                    per_length_hist: Dict[int, List[int]] = defaultdict(lambda: [0 for _ in range(bins)])

                    for rec in rules:
                        valid = int(rec.get("valid_antecedent_count", 0))
                        support = int(rec.get("support", 0))
                        conf = float(support) / float(valid) if valid > 0 else 0.0
                        idx = int(math.floor(conf / step)) if step > 0 else 0
                        if idx >= bins:
                            idx = bins - 1
                        if idx < 0:
                            idx = 0

                        overall_hist[idx] += 1
                        length = int(rec.get("antecedent_length", 1))
                        per_length_hist[length][idx] += 1

                    print("\nConfidence distribution across all mixed-length rules (step=0.2):")
                    for i, label in enumerate(labels):
                        print(f"  {label}: {overall_hist[i]}")

                    print("\nPer-length confidence distribution:")
                    for length in sorted(per_length_hist.keys()):
                        counts = per_length_hist[length]
                        parts = [f"{labels[i]}:{counts[i]}" for i in range(len(labels))]
                        print(f"  length={length}: {', '.join(parts)}")
                    # Per-length composite score distribution (using the same composite
                    # score formula as `score_rules` but applied to aggregated mixed-length
                    # entries). We compute simple summary statistics per antecedent length.
                    try:
                        score_by_length: Dict[int, List[float]] = defaultdict(list)
                        eps = 1e-9
                        for rec in rules:
                            support = int(rec.get("support", 0))
                            valid = int(rec.get("valid_antecedent_count", 0))
                            confidence = float(support) / float(valid) if valid > 0 else 0.0
                            cons_key = (rec.get("consequent_key"), rec.get("consequent_value"))
                            p_q = float(global_Q_counts.get(cons_key, 0)) / float(total_segments) if total_segments > 0 else 0.0
                            lift = float(confidence) / (p_q + eps)
                            stability = float(len(rec.get("scenes", []))) / float(num_scenes) if num_scenes > 0 else 0.0
                            support_score = float(support) / float(total_segments) if total_segments > 0 else 0.0
                            novelty_score = float(confidence - p_q)
                            composite = support_score * float(lift) * float(stability) * float(novelty_score)
                            length = int(rec.get("antecedent_length", 1))
                            score_by_length[length].append(float(composite))

                        if score_by_length:
                            print("\nPer-length composite score distribution:")
                            for length in sorted(score_by_length.keys()):
                                scs = sorted(score_by_length[length])
                                cnt = len(scs)
                                mean = sum(scs) / cnt if cnt > 0 else 0.0
                                if cnt == 0:
                                    median = 0.0
                                elif cnt % 2 == 1:
                                    median = scs[cnt // 2]
                                else:
                                    median = 0.5 * (scs[cnt // 2 - 1] + scs[cnt // 2])
                                min_s = scs[0] if cnt > 0 else 0.0
                                max_s = scs[-1] if cnt > 0 else 0.0
                                var = sum((x - mean) ** 2 for x in scs) / cnt if cnt > 0 else 0.0
                                std = math.sqrt(var)
                                print(
                                    f"  length={length}: count={cnt}, mean={mean:.6g}, median={median:.6g}, "
                                    f"min={min_s:.6g}, max={max_s:.6g}, std={std:.6g}"
                                )
                            # Overall and per-length score-range histograms (bins across
                            # the observed score range). This prints counts of rules whose
                            # composite scores fall into each bin.
                            try:
                                all_scores = [s for vals in score_by_length.values() for s in vals]
                                if all_scores:
                                    bins = 10
                                    s_min = min(all_scores)
                                    s_max = max(all_scores)
                                    if s_min == s_max:
                                        labels = [f"[{s_min:.6g}-{s_max:.6g}]"]
                                        overall_counts = [len(all_scores)]
                                    else:
                                        step = (s_max - s_min) / float(bins)
                                        edges = [s_min + i * step for i in range(bins + 1)]
                                        labels = [f"[{edges[i]:.6g}-{edges[i+1]:.6g})" for i in range(bins)]
                                        labels[-1] = f"[{edges[-2]:.6g}-{edges[-1]:.6g}]"
                                        overall_counts = [0 for _ in range(bins)]
                                        for v in all_scores:
                                            if v <= s_min:
                                                idx = 0
                                            elif v >= s_max:
                                                idx = bins - 1
                                            else:
                                                idx = int(math.floor((v - s_min) / (s_max - s_min) * bins))
                                                if idx < 0:
                                                    idx = 0
                                                if idx >= bins:
                                                    idx = bins - 1
                                            overall_counts[idx] += 1

                                    print("\nScore distribution across all mixed-length rules (bins=10):")
                                    for i, label in enumerate(labels):
                                        print(f"  {label}: {overall_counts[i]}")

                                    # Per-length distribution using same bin edges
                                    print("\nPer-length score distribution (same bins):")
                                    for length in sorted(score_by_length.keys()):
                                        vals = score_by_length[length]
                                        counts_len = [0 for _ in range(len(labels))]
                                        if vals:
                                            if s_min == s_max:
                                                counts_len[0] = len(vals)
                                            else:
                                                for v in vals:
                                                    if v <= s_min:
                                                        idx = 0
                                                    elif v >= s_max:
                                                        idx = bins - 1
                                                    else:
                                                        idx = int(math.floor((v - s_min) / (s_max - s_min) * bins))
                                                        if idx < 0:
                                                            idx = 0
                                                        if idx >= bins:
                                                            idx = bins - 1
                                                    counts_len[idx] += 1
                                        parts = [f"{labels[i]}:{counts_len[i]}" for i in range(len(labels))]
                                        print(f"  length={length}: {', '.join(parts)}")
                                else:
                                    print("No scores available to produce histogram.")
                            except Exception as ex3:
                                print(f"Warning: failed to compute score histograms: {ex3}")
                        else:
                            print("No mixed-length rules to compute score distribution.")
                    except Exception as ex2:
                        print(f"Warning: failed to compute per-length score distribution: {ex2}")
                else:
                    print("No mixed-length rules to compute confidence distribution.")
            except Exception as ex:
                print(f"Warning: failed to compute confidence distribution: {ex}")

        except Exception as exc:
            print(f"Warning: mixed-length aggregation failed: {exc}")

    print(
        f"Aggregated {len(global_rule_counts)} unique rules across {num_scenes} scenes, {total_segments} segments."
    )
    print(f"Filtered rules: {len(filtered)}")
    if run_cfg.get("debug"):
        print("\nTop filtered rules:")
        for r in filtered[:10]:
            print(
                f"  ({r['antecedent_key']}={r['antecedent_value']}) -> ({r['consequent_key']}={r['consequent_value']}) "
                f"dt={r['lag']} support={r['support']} conf={r['confidence']:.2f} lift={r['lift']:.2f} stab={r['stability']:.2f}"
            )
        # Also print top rules by composite score (new ranking metric)
        print("\nTop scored rules (by composite score):")
        for r in scored[:10]:
            print(
                f"  ({r['antecedent_key']}={r['antecedent_value']}) -> ({r['consequent_key']}={r['consequent_value']}) "
                f"dt={r['lag']} support={r['support']} score={r.get('score', 0.0):.6g} conf={r.get('confidence', 0.0):.2f} novelty={r.get('novelty_score', 0.0):.3f} lift={r.get('lift', 0.0):.2f} stab={r.get('stability', 0.0):.2f}"
            )


def main() -> None:
    args, run_cfg, output_root, save_atoms, scene_names = _parse_cli()

    # Setup per-run logging: write a timestamped log file under output_root/logs
    log_fh = None
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        if output_root is None:
            log_dir = Path("logs")
        else:
            log_dir = Path(output_root) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"pattern_mining_{ts}.log"
        log_fh = log_path.open("a", encoding="utf-8")

        class Tee:
            def __init__(self, *files):
                self.files = files

            def write(self, data):
                for f in self.files:
                    try:
                        f.write(data)
                    except Exception:
                        pass

            def flush(self):
                for f in self.files:
                    try:
                        f.flush()
                    except Exception:
                        pass

        sys.stdout = Tee(old_stdout, log_fh)
        sys.stderr = Tee(old_stderr, log_fh)

        # Configure basic logging to also write to the same file
        logging.basicConfig(
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
            format="%(asctime)s %(levelname)s %(message)s",
        )

        print(f"Run log: {log_path}")
    except Exception as exc:
        # If logging setup fails, continue but warn
        try:
            print(f"Warning: failed to initialize run log: {exc}")
        except Exception:
            pass

    try:
        timelines, per_scene_results, total_segments = _run_and_report(output_root, scene_names, run_cfg)

        _maybe_save_atoms(save_atoms, timelines)

        try:
            _aggregate_from_per_scene(per_scene_results, total_segments, output_root, run_cfg)
        except Exception as exc:  # don't crash the whole CLI if scoring fails
            print(f"Warning: aggregation/scoring failed: {exc}")
    finally:
        # restore stdout/stderr and close the log file
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except Exception:
            pass
        try:
            if log_fh is not None:
                log_fh.close()
                print(f"Closed run log: {log_path}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
