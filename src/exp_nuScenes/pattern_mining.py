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
    "save_atoms": None,
    "debug": False,
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


def _is_informative_antecedent(atom: Atom) -> bool:
    key, value = atom
    if key == "num_focus_objects":
        return False
    if key == "ego_motion_segment":
        return False
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
                    if _is_trivial_rule(antecedent, consequent):
                        continue
                    rule_counts[(antecedent, consequent, lag)] += 1

    return dict(rule_counts)


def score_rules(
    rule_counts: Dict[Rule, int],
    antecedent_counts: Dict[Atom, int],
    consequent_counts: Dict[Atom, int],
    total_segments: int,
    num_scenes: int,
) -> List[Dict[str, Any]]:
    """Score mined rules by support, confidence, lift, and scene stability."""
    raise NotImplementedError


def filter_rules(
    scored_rules: Sequence[Dict[str, Any]],
    min_support: int = 2,
    min_confidence: float = 0.5,
    min_lift: float = 1.0,
    min_stability: float = 0.0,
) -> List[Dict[str, Any]]:
    """Filter scored rules using configurable thresholds."""
    raise NotImplementedError


def run_pattern_mining(
    output_root: Optional[Path] = None,
    scene_names: Optional[Sequence[str]] = None,
    max_lag: int = 3,
    top_k_objects: int = 3,
) -> List[Dict[str, Any]]:
    """Entry point for nuScenes predicate-to-pattern mining."""
    del max_lag
    if output_root is None:
        output_root = Path("pipeline_output") / "nuScenes_from_config"
    return build_timeline_atoms_for_scenes(
        Path(output_root),
        scene_names=scene_names,
        top_k_objects=top_k_objects,
    )


def main() -> None:
    """CLI entry point placeholder."""
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
    args = parser.parse_args()
    run_cfg = resolve_run_config(args)

    output_root = _resolve_project_path(run_cfg.get("output_root"))
    save_atoms = _resolve_project_path(run_cfg.get("save_atoms"))
    scene_names = run_cfg.get("scene_names") or None

    timelines = run_pattern_mining(
        output_root=output_root,
        scene_names=scene_names,
        max_lag=int(run_cfg["max_lag"]),
        top_k_objects=int(run_cfg["top_k_objects"]),
    )
    scene_rules = mine_atomic_rules_for_scenes(timelines, max_lag=int(run_cfg["max_lag"]))

    total_segments = sum(int(scene["num_segments"]) for scene in timelines)
    print(f"Built timeline atoms for {len(timelines)} scene(s), {total_segments} segment(s).")
    for scene in timelines:
        print(f"- {scene['scene_name']}: {scene['num_segments']} segments")
    print(f"Generated atomic temporal rules with max_lag={int(run_cfg['max_lag'])}.")

    if run_cfg.get("debug"):
        print_atoms_debug_summary(timelines)
        print_rules_debug_summary(scene_rules)

    if save_atoms is not None:
        saved_path = save_timeline_atoms(timelines, save_atoms)
        print(f"Saved timeline atoms to {saved_path}")


if __name__ == "__main__":
    main()
