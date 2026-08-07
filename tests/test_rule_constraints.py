from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from src.exp_driving_videos import pipeline_data
from src.exp_driving_videos.modules import extended_rules_driving_mini
from src.exp_driving_videos.modules import rule_constraints


def _constraints(mode: str, **overrides: Any) -> Dict[str, Any]:
    cfg = {"mode": mode}
    cfg.update(overrides)
    return rule_constraints.normalize_constraints_cfg(cfg)


class BodyConflictTest(unittest.TestCase):
    """Unit coverage for the constraint checker itself."""

    def test_same_functional_predicate_different_value_conflicts(self) -> None:
        reason = rule_constraints.body_conflict(
            [
                "object_vz_state(S,O,vz_approaching).",
                "object_vz_state(S,O,vz_awaying).",
            ],
            _constraints(rule_constraints.MODE_DERIVED),
        )
        self.assertTrue(reason.startswith("functional_exclusion:object_vz_state:"))

    def test_reason_is_stable_regardless_of_atom_order(self) -> None:
        constraints = _constraints(rule_constraints.MODE_DERIVED)
        forward = rule_constraints.body_conflict(
            ["object_distance_state(S,O,near).", "object_distance_state(S,O,far)."],
            constraints,
        )
        backward = rule_constraints.body_conflict(
            ["object_distance_state(S,O,far).", "object_distance_state(S,O,near)."],
            constraints,
        )
        self.assertEqual(forward, backward)

    def test_same_predicate_same_value_is_allowed(self) -> None:
        self.assertEqual(
            rule_constraints.body_conflict(
                [
                    "object_distance_state(S,O,near).",
                    "object_distance_state(S,O,near).",
                ],
                _constraints(rule_constraints.MODE_DERIVED),
            ),
            "",
        )

    def test_different_objects_are_not_a_conflict(self) -> None:
        """Two objects may legitimately hold different values of one predicate."""

        self.assertEqual(
            rule_constraints.body_conflict(
                [
                    "object_distance_state(S,O1,near).",
                    "object_distance_state(S,O2,far).",
                ],
                _constraints(rule_constraints.MODE_DERIVED),
            ),
            "",
        )

    def test_different_predicates_are_not_a_conflict(self) -> None:
        self.assertEqual(
            rule_constraints.body_conflict(
                [
                    "object_distance_state(S,O,near).",
                    "object_vz_state(S,O,vz_approaching).",
                ],
                _constraints(rule_constraints.MODE_DERIVED),
            ),
            "",
        )

    def test_non_functional_predicate_is_ignored(self) -> None:
        self.assertEqual(
            rule_constraints.body_conflict(
                [
                    "object_matched_prior(O,lead_vehicle).",
                    "object_matched_prior(O,pedestrian).",
                ],
                _constraints(rule_constraints.MODE_DERIVED),
            ),
            "",
        )

    def test_segment_level_predicate_uses_key_arity_one(self) -> None:
        reason = rule_constraints.body_conflict(
            ["segment_motion_state(S,forward).", "segment_motion_state(S,stopping)."],
            _constraints(rule_constraints.MODE_DERIVED),
        )
        self.assertTrue(reason.startswith("functional_exclusion:segment_motion_state:"))

    def test_off_mode_permits_everything(self) -> None:
        self.assertEqual(
            rule_constraints.body_conflict(
                [
                    "object_vz_state(S,O,vz_approaching).",
                    "object_vz_state(S,O,vz_awaying).",
                ],
                _constraints(rule_constraints.MODE_OFF),
            ),
            "",
        )

    def test_authored_constraints_only_apply_in_authored_mode(self) -> None:
        forbidden = {"forbidden_atoms": ["object_distance_state(S,O,far)."]}
        body = ["object_distance_state(S,O,far)."]

        self.assertEqual(
            rule_constraints.body_conflict(
                body, _constraints(rule_constraints.MODE_DERIVED, **forbidden)
            ),
            "",
        )
        self.assertEqual(
            rule_constraints.body_conflict(
                body,
                _constraints(rule_constraints.MODE_DERIVED_AND_AUTHORED, **forbidden),
            ),
            "forbidden_atom:object_distance_state(S,O,far).",
        )

    def test_forbidden_combination_requires_every_atom(self) -> None:
        constraints = _constraints(
            rule_constraints.MODE_DERIVED_AND_AUTHORED,
            forbidden_combinations=[
                [
                    "object_distance_state(S,O,far).",
                    "object_vz_state(S,O,vz_awaying).",
                ]
            ],
        )
        self.assertEqual(
            rule_constraints.body_conflict(["object_distance_state(S,O,far)."], constraints),
            "",
        )
        self.assertTrue(
            rule_constraints.body_conflict(
                [
                    "object_distance_state(S,O,far).",
                    "object_vz_state(S,O,vz_awaying).",
                ],
                constraints,
            ).startswith("forbidden_combination:")
        )

    def test_atom_templates_normalize_trailing_dots_and_spacing(self) -> None:
        constraints = _constraints(
            rule_constraints.MODE_DERIVED_AND_AUTHORED,
            forbidden_atoms=["  object_distance_state(S,O,far)  "],
        )
        self.assertTrue(
            rule_constraints.body_conflict(
                ["object_distance_state(S,O,far)."], constraints
            ).startswith("forbidden_atom:")
        )

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rule_constraints.normalize_mode("aggressive")

    def test_functional_predicate_overrides_merge_onto_defaults(self) -> None:
        constraints = _constraints(
            rule_constraints.MODE_DERIVED,
            functional_predicates={"custom_state": 2},
        )
        # An explicit override replaces the table wholesale at this layer; the
        # merge onto defaults happens in pipeline_config.get_rule_constraints_cfg.
        self.assertEqual(constraints["functional_predicates"], {"custom_state": 2})


def _evidence(example_id: str, label: bool, atom_template: str, obj: str) -> Dict[str, Any]:
    """One evidence entry binding S to the example and O to `obj`."""

    concrete = atom_template.replace("(S,O,", f"({example_id},{obj},")
    return {
        "example_id": example_id,
        "current_segment_id": example_id,
        "next_segment_id": f"{example_id}_next",
        "label": label,
        "body_atom_template": atom_template,
        "matched_atom": concrete,
        "body_atom_source": "accepted",
        "bindings": {"S": example_id, "O": obj},
        "matched_atoms": {atom_template: concrete},
        "matched_atom_sources": {atom_template: "accepted"},
        "matched_atom_prior_ids": {},
    }


# (example_id, label, bound object) triples per atom template.
_Matches = List[tuple]


def _unary_rule(rule_id: str, atom_template: str, matches: _Matches) -> Dict[str, Any]:
    positive_ids = [example_id for example_id, label, _ in matches if label]
    negative_ids = [example_id for example_id, label, _ in matches if not label]
    total = len(positive_ids) + len(negative_ids)
    return {
        "rule_id": rule_id,
        "head_predicate": "brake_next",
        "head_atom_template": "brake_next(S).",
        "body_atom_template": atom_template,
        "body_atom_templates": [atom_template],
        "body_length": 1,
        "clause": f"brake_next(S) :- {atom_template}",
        "positive_support": len(positive_ids),
        "negative_support": len(negative_ids),
        "total_support": total,
        "positive_firings": len(positive_ids),
        "negative_firings": len(negative_ids),
        "total_firings": total,
        "confidence": (len(positive_ids) / total) if total else 0.0,
        "positive_example_ids": positive_ids,
        "negative_example_ids": negative_ids,
        "uses_candidate_atoms": False,
        "num_candidate_body_atoms": 0,
        "candidate_body_atom_ratio": 0.0,
        "mixes_accepted_and_candidate_atoms": False,
        "uses_only_candidate_atoms": False,
        "candidate_rule_category": "accepted_only",
        "initial_rule_pair_category": "accepted_only",
        "evidence_set": [
            _evidence(example_id, label, atom_template, obj) for example_id, label, obj in matches
        ],
    }


def _merged_pool() -> Dict[str, Any]:
    """A seed pool whose evidence respects the functional property.

    This matters: the pipeline's classifiers give each object exactly one distance
    state per segment, so no example may claim one object is both `near` and
    `far`. Fabricating such evidence would let a contradictory rule intersect to
    a non-empty set and make the equivalence test vacuous.

    Layout (obj_1 near, obj_2 far - never the same object twice):

        ex_a  +   obj_1 near, obj_1 approaching
        ex_b  +   obj_1 near, obj_1 approaching
        ex_c  +   obj_1 near, obj_2 far, obj_1 awaying
        ex_n1 -   obj_1 near, obj_1 awaying
        ex_n2 -   obj_2 far,  obj_2 approaching

    `near` + `approaching` narrows to {ex_a, ex_b} and drops the negative, so it
    survives extension. `near` + `far` can never hold of one object, so it is the
    combination the constraint is expected to avoid.
    """

    return {
        "version": 1,
        "num_rules": 4,
        "num_videos": 1,
        "rules": [
            _unary_rule(
                "near",
                "object_distance_state(S,O,near).",
                [
                    ("ex_a", True, "obj_1"),
                    ("ex_b", True, "obj_1"),
                    ("ex_c", True, "obj_1"),
                    ("ex_n1", False, "obj_1"),
                ],
            ),
            _unary_rule(
                "far",
                "object_distance_state(S,O,far).",
                [("ex_c", True, "obj_2"), ("ex_n2", False, "obj_2")],
            ),
            _unary_rule(
                "approaching",
                "object_vz_state(S,O,vz_approaching).",
                [
                    ("ex_a", True, "obj_1"),
                    ("ex_b", True, "obj_1"),
                    ("ex_n2", False, "obj_2"),
                ],
            ),
            _unary_rule(
                "awaying",
                "object_vz_state(S,O,vz_awaying).",
                [("ex_c", True, "obj_1"), ("ex_n1", False, "obj_1")],
            ),
        ],
        "num_positive_examples": 3,
        "num_negative_examples": 2,
        "candidate_rule_stage_stats": {},
    }


def _extension_cfg(mode: str, *, num_rounds: int = 2, **constraint_overrides: Any) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {"mode": mode}
    constraints.update(constraint_overrides)
    return {
        "num_rounds": num_rounds,
        "min_positive_support_to_extend": 1,
        # Mirror the shipped config so the comparison exercises the real path.
        "prune_strategies": [
            "low_evidence",
            "empty_evidence",
            "same_firings_as_parent",
            "same_confidence_smaller_evidence",
        ],
        "rule_constraints": constraints,
    }


def _run_extension(root: Path, mode: str, tag: str, **kwargs: Any) -> Dict[str, Any]:
    return extended_rules_driving_mini.run(
        merged_initial_rules=_merged_pool(),
        cfg=_extension_cfg(mode, **kwargs),
        output_root=root / tag,
        force_recompute=True,
    )


def _round_pruning(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(entry.get("pruning", {})) for entry in list(manifest.get("rounds", []))]


def _summed(manifest: Dict[str, Any], key: str) -> int:
    return sum(int(pruning.get(key, 0)) for pruning in _round_pruning(manifest))


def _summed_round_field(manifest: Dict[str, Any], key: str) -> int:
    return sum(int(entry.get(key, 0)) for entry in list(manifest.get("rounds", [])))


class ExtensionGateTest(unittest.TestCase):
    def test_derived_mode_produces_the_same_rules_as_the_naive_run(self) -> None:
        """The gate must only remove rules the naive path already discarded.

        This is the load-bearing test: if a functional-predicate declaration were
        wrong, the constrained run would drop a rule the naive run kept and the
        clause lists would diverge.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            naive = _run_extension(root, rule_constraints.MODE_OFF, "off")
            constrained = _run_extension(root, rule_constraints.MODE_DERIVED, "derived")

            self.assertEqual(
                naive["num_all_kept_rules"], constrained["num_all_kept_rules"]
            )
            self.assertEqual(
                sorted(rule["clause"] for rule in naive["all_kept_rules"]),
                sorted(rule["clause"] for rule in constrained["all_kept_rules"]),
            )

    def test_gate_avoids_work_rather_than_pruning_it_afterwards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            naive = _run_extension(root, rule_constraints.MODE_OFF, "off")
            constrained = _run_extension(root, rule_constraints.MODE_DERIVED, "derived")

            blocked = _summed(constrained, "pruned_background_knowledge_conflict")
            self.assertEqual(_summed(naive, "pruned_background_knowledge_conflict"), 0)
            self.assertGreater(blocked, 0)

            # The point of the change: blocked combinations never reach the
            # evidence intersection, so fewer candidates are evaluated while the
            # same total number of combinations is rejected.
            naive_evaluated = _summed_round_field(naive, "num_candidates_generated")
            constrained_evaluated = _summed_round_field(constrained, "num_candidates_generated")
            self.assertEqual(naive_evaluated - constrained_evaluated, blocked)
            self.assertEqual(_summed(naive, "pruned_num_rules"), _summed(constrained, "pruned_num_rules"))

            reasons = _round_pruning(constrained)[0].get("background_knowledge_conflict_reasons", {})
            self.assertEqual(set(reasons), {"functional_exclusion"})

    def test_authored_constraints_remove_rules_the_derived_run_keeps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            derived = _run_extension(
                root, rule_constraints.MODE_DERIVED, "derived", num_rounds=1
            )
            authored = _run_extension(
                root,
                rule_constraints.MODE_DERIVED_AND_AUTHORED,
                "authored",
                num_rounds=1,
                forbidden_combinations=[
                    [
                        "object_distance_state(S,O,near).",
                        "object_vz_state(S,O,vz_approaching).",
                    ]
                ],
            )

            def has_pair(manifest: Dict[str, Any]) -> bool:
                return any(
                    "object_distance_state(S,O,near)." in rule.get("body_atom_templates", [])
                    and "object_vz_state(S,O,vz_approaching)." in rule.get("body_atom_templates", [])
                    for rule in manifest["all_kept_rules"]
                )

            self.assertTrue(has_pair(derived))
            self.assertFalse(has_pair(authored))

    def test_mode_change_does_not_reuse_a_cached_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_root = Path(tmpdir) / "shared"
            first = extended_rules_driving_mini.run(
                merged_initial_rules=_merged_pool(),
                cfg=_extension_cfg(rule_constraints.MODE_OFF, num_rounds=1),
                output_root=shared_root,
                force_recompute=True,
            )
            second = extended_rules_driving_mini.run(
                merged_initial_rules=_merged_pool(),
                cfg=_extension_cfg(rule_constraints.MODE_DERIVED, num_rounds=1),
                output_root=shared_root,
                force_recompute=False,
            )

            self.assertEqual(first["rule_constraints"]["mode"], rule_constraints.MODE_OFF)
            self.assertEqual(second["rule_constraints"]["mode"], rule_constraints.MODE_DERIVED)
            self.assertGreater(_summed(second, "pruned_background_knowledge_conflict"), 0)


class InitialRulePruningGateTest(unittest.TestCase):
    def test_authored_constraints_drop_seed_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pruned = pipeline_data.prune_initial_rules(
                _merged_pool(),
                cfg={
                    "max_total_initial_rules": 10,
                    "category_budgets": {"accepted_only": 10},
                    "max_accepted_only_initial_rules": 10,
                    "rule_constraints": {
                        "mode": rule_constraints.MODE_DERIVED_AND_AUTHORED,
                        "forbidden_atoms": ["object_distance_state(S,O,far)."],
                    },
                },
                output_root=root / "15b",
            )

            summary = pruned["candidate_rule_stage_stats"]["initial_rule_pruning"]
            self.assertEqual(summary["background_knowledge_pruned_num_rules"], 1)
            self.assertEqual(
                summary["background_knowledge_pruned_reasons"], {"forbidden_atom": 1}
            )
            self.assertNotIn(
                "object_distance_state(S,O,far).",
                {rule.get("body_atom_template", "") for rule in pruned["rules"]},
            )

    def test_derived_mode_alone_drops_no_unary_seed_rules(self) -> None:
        """A single atom cannot contradict itself, so derived is a no-op here."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pruned = pipeline_data.prune_initial_rules(
                _merged_pool(),
                cfg={
                    "max_total_initial_rules": 10,
                    "category_budgets": {"accepted_only": 10},
                    "max_accepted_only_initial_rules": 10,
                    "rule_constraints": {"mode": rule_constraints.MODE_DERIVED},
                },
                output_root=root / "15b",
            )

            summary = pruned["candidate_rule_stage_stats"]["initial_rule_pruning"]
            self.assertEqual(summary["background_knowledge_pruned_num_rules"], 0)


if __name__ == "__main__":
    unittest.main()
