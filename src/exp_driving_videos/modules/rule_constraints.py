"""
Background-knowledge constraints that keep impossible rules out of the search.

The step-16 extension loop pairs every parent rule with every extension atom and
runs a binding-aware evidence intersection on each combination. Some of those
combinations cannot be satisfied by any data, so the intersection is wasted work
and the resulting "no evidence" audit entry hides the real reason.

This module answers one question with no I/O and no pipeline dependencies:

    given a candidate rule body, does background knowledge already rule it out?

Two knowledge sources are supported.

Derived exclusions
    Every state predicate is emitted exactly once per entity with a single
    classifier value (see `logic_atoms_driving_mini._append_object_atom` calls and
    the `_classify_*` helpers in `segment_object_motion_driving_mini`). Such a
    predicate is *functional*: its leading `key_arity` arguments identify an
    entity and the trailing argument is the single value for that entity. A body
    holding two atoms with the same predicate and key but different values is
    unsatisfiable. No value enumeration is needed, so new classifier values are
    covered automatically.

Authored constraints
    Asserted domain knowledge: atoms that may never appear in a body, and atom
    sets that may not co-occur. These are claims about the world rather than facts
    about the code, so they ship empty and are only consulted in
    `derived_and_authored` mode.

Only *monotone* constraints belong here. A constraint is monotone when blocking a
body also invalidates every superset of that body, which is what makes it safe to
apply mid-search: the extension loop grows bodies one atom at a time, so a
non-monotone constraint would reject a partial body that was going to become
valid in a later round. All three kinds above are monotone. Implication-style
knowledge ("atom A requires atom B") is not, and is deliberately unsupported.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MODE_OFF = "off"
MODE_DERIVED = "derived"
MODE_DERIVED_AND_AUTHORED = "derived_and_authored"

_VALID_MODES = (MODE_OFF, MODE_DERIVED, MODE_DERIVED_AND_AUTHORED)

# predicate -> number of leading arguments that identify the entity. The argument
# directly after the key is the single value for that entity.
#
# Each entry is verifiable at its emission site:
#   key_arity 2, keyed on (segment, object) - logic_atoms_driving_mini.py:434-440, 504-518
#   key_arity 1, keyed on (segment)         - logic_atoms_driving_mini.py:360-362
#   key_arity 1, keyed on (object)          - logic_atoms_driving_mini.py:446-457
DEFAULT_FUNCTIONAL_PREDICATES: Dict[str, int] = {
    "object_class": 2,
    "object_vz_state": 2,
    "object_vx_state": 2,
    "object_speed_state": 2,
    "object_distance_state": 2,
    "object_visibility_state": 2,
    "object_x_position_state": 2,
    "traffic_light_state": 2,
    "traffic_control_type": 2,
    "segment_forward_state": 1,
    "segment_lateral_state": 1,
    "segment_motion_state": 1,
    "object_candidate_score_state": 1,
    "object_prior_relevance_state": 1,
    "object_source_type": 1,
}


def _strip_trailing_dot(atom_text: str) -> str:
    return str(atom_text).strip().rstrip(".").strip()


def _normalize_atom_template(atom_text: str) -> str:
    normalized = _strip_trailing_dot(atom_text)
    return f"{normalized}." if normalized else ""


def _parse_atom(atom_text: str) -> Optional[Tuple[str, List[str]]]:
    """Split an atom template into its predicate and argument list."""

    text = _strip_trailing_dot(atom_text)
    if not text or "(" not in text or not text.endswith(")"):
        return None
    predicate, args_text = text.split("(", 1)
    predicate = predicate.strip()
    args_text = args_text[:-1].strip()
    if not predicate:
        return None
    if not args_text:
        return predicate, []
    return predicate, [part.strip() for part in args_text.split(",")]


def normalize_mode(mode: Any) -> str:
    text = str(mode or "").strip().lower()
    if text in {"", "none", "false", "disabled"}:
        return MODE_OFF
    if text in {"true", "on", "enabled"}:
        return MODE_DERIVED
    if text not in _VALID_MODES:
        raise ValueError(
            f"Unsupported rule constraint mode: {mode!r}. Expected one of {', '.join(_VALID_MODES)}."
        )
    return text


def _normalized_functional_predicates(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return dict(DEFAULT_FUNCTIONAL_PREDICATES)

    normalized: Dict[str, int] = {}
    for predicate, key_arity in value.items():
        name = str(predicate).strip()
        if not name:
            continue
        try:
            arity = int(key_arity)
        except (TypeError, ValueError):
            continue
        if arity < 1:
            continue
        normalized[name] = arity
    return normalized


def _normalized_atom_list(value: Any) -> Tuple[str, ...]:
    atoms = {
        _normalize_atom_template(atom_text)
        for atom_text in list(value or [])
        if _normalize_atom_template(atom_text)
    }
    return tuple(sorted(atoms))


def _normalized_combinations(value: Any) -> Tuple[Tuple[str, ...], ...]:
    combinations: List[Tuple[str, ...]] = []
    for group in list(value or []):
        if isinstance(group, str):
            group = [group]
        atoms = _normalized_atom_list(group)
        # A one-atom "combination" is a forbidden atom; keep those in the
        # dedicated list so reasons stay accurate.
        if len(atoms) < 2:
            continue
        if atoms not in combinations:
            combinations.append(atoms)
    return tuple(sorted(combinations))


def normalize_constraints_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve a raw config section into the structure `body_conflict` expects.

    The result is JSON-serializable and stable under sorting so it can be folded
    into a step's cache key.
    """

    cfg = cfg or {}
    mode = normalize_mode(cfg.get("mode", MODE_DERIVED))
    authored_enabled = mode == MODE_DERIVED_AND_AUTHORED
    return {
        "mode": mode,
        "derived_enabled": mode in {MODE_DERIVED, MODE_DERIVED_AND_AUTHORED},
        "authored_enabled": authored_enabled,
        "functional_predicates": dict(
            sorted(_normalized_functional_predicates(cfg.get("functional_predicates")).items())
        ),
        "forbidden_atoms": list(_normalized_atom_list(cfg.get("forbidden_atoms"))),
        "forbidden_combinations": [list(group) for group in _normalized_combinations(cfg.get("forbidden_combinations"))],
    }


def is_enabled(constraints: Optional[Dict[str, Any]]) -> bool:
    constraints = constraints or {}
    return bool(constraints.get("derived_enabled", False) or constraints.get("authored_enabled", False))


def _derived_conflict(
    body_atom_templates: Sequence[str],
    functional_predicates: Dict[str, int],
) -> str:
    seen: Dict[Tuple[str, Tuple[str, ...]], str] = {}
    for atom_text in body_atom_templates:
        parsed = _parse_atom(atom_text)
        if parsed is None:
            continue
        predicate, args = parsed
        key_arity = functional_predicates.get(predicate)
        if key_arity is None or len(args) <= key_arity:
            continue

        key = (predicate, tuple(args[:key_arity]))
        value = args[key_arity]
        previous = seen.get(key)
        if previous is not None and previous != value:
            # Sort the two values so the reason is stable regardless of body order.
            low, high = sorted((previous, value))
            return f"functional_exclusion:{predicate}:{low}|{high}"
        seen[key] = value
    return ""


def _authored_conflict(
    body_atom_templates: Sequence[str],
    forbidden_atoms: Sequence[str],
    forbidden_combinations: Sequence[Sequence[str]],
) -> str:
    body = {_normalize_atom_template(atom_text) for atom_text in body_atom_templates}
    body.discard("")

    for atom_text in forbidden_atoms:
        if atom_text in body:
            return f"forbidden_atom:{atom_text}"

    for group in forbidden_combinations:
        group_atoms = tuple(group)
        if group_atoms and body.issuperset(group_atoms):
            return f"forbidden_combination:{'|'.join(group_atoms)}"

    return ""


def body_conflict(
    body_atom_templates: Iterable[str],
    constraints: Optional[Dict[str, Any]],
) -> str:
    """Return a reason string when background knowledge rules this body out.

    Returns an empty string when the body is permitted. `constraints` must come
    from `normalize_constraints_cfg`.
    """

    constraints = constraints or {}
    if not is_enabled(constraints):
        return ""

    atoms = [_normalize_atom_template(atom_text) for atom_text in body_atom_templates]
    atoms = [atom_text for atom_text in atoms if atom_text]
    if not atoms:
        return ""

    if bool(constraints.get("derived_enabled", False)):
        reason = _derived_conflict(
            atoms,
            dict(constraints.get("functional_predicates", {})),
        )
        if reason:
            return reason

    if bool(constraints.get("authored_enabled", False)):
        reason = _authored_conflict(
            atoms,
            list(constraints.get("forbidden_atoms", [])),
            list(constraints.get("forbidden_combinations", [])),
        )
        if reason:
            return reason

    return ""


def conflict_kind(reason: str) -> str:
    """Reduce a reason string to its constraint kind for counter breakdowns."""

    text = str(reason or "").strip()
    if not text:
        return ""
    return text.split(":", 1)[0]
