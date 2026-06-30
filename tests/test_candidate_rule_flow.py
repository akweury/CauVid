from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.exp_driving_videos import pipeline_data
from src.exp_driving_videos.modules import candidate_rules_driving_mini
from src.exp_driving_videos.modules import evaluate_rules_driving_mini
from src.exp_driving_videos.modules import extended_rules_driving_mini
from src.exp_driving_videos.modules import final_rules_driving_mini


class CandidateRuleFlowTest(unittest.TestCase):
    def test_merge_initial_rules_filters_non_unary_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate_rule_results = [
                {
                    "video_id": "video_merge_filter",
                    "target_predicate": "brake_next",
                    "num_examples": 2,
                    "num_positive_examples": 1,
                    "num_negative_examples": 1,
                    "num_rules_using_candidate_atoms": 1,
                    "num_candidate_only_rules": 1,
                    "num_mixed_source_rules": 1,
                    "initial_rules": [
                        {
                            "rule_id": "unary_rule",
                            "head_predicate": "brake_next",
                            "head_atom_template": "brake_next(S).",
                            "body_atom_template": "object_distance_state(S,O,near).",
                            "body_atom_templates": ["object_distance_state(S,O,near)."],
                            "body_length": 1,
                            "clause": "brake_next(S) :- object_distance_state(S,O,near).",
                            "positive_support": 1,
                            "negative_support": 0,
                            "total_support": 1,
                            "positive_firings": 1,
                            "negative_firings": 0,
                            "total_firings": 1,
                            "confidence": 1.0,
                            "positive_example_ids": ["ex_pos"],
                            "negative_example_ids": [],
                            "uses_candidate_atoms": False,
                            "num_candidate_body_atoms": 0,
                            "candidate_body_atom_ratio": 0.0,
                            "mixes_accepted_and_candidate_atoms": False,
                            "uses_only_candidate_atoms": False,
                            "candidate_rule_category": "accepted_only",
                            "initial_rule_pair_category": "accepted_only",
                            "evidence_set": [
                                {
                                    "example_id": "ex_pos",
                                    "label": True,
                                    "matched_atom": "object_distance_state(seg,obj,near).",
                                    "bindings": {"S": "seg", "O": "obj"},
                                    "matched_atoms": {"object_distance_state(S,O,near).": "object_distance_state(seg,obj,near)."},
                                }
                            ],
                        },
                        {
                            "rule_id": "binary_rule",
                            "head_predicate": "brake_next",
                            "head_atom_template": "brake_next(S).",
                            "body_atom_template": "",
                            "body_atom_templates": [
                                "object_distance_state(S,O,near).",
                                "object_class(S,C,pedestrian).",
                            ],
                            "body_length": 2,
                            "clause": "brake_next(S) :- object_distance_state(S,O,near), object_class(S,C,pedestrian).",
                            "positive_support": 1,
                            "negative_support": 0,
                            "total_support": 1,
                            "positive_firings": 1,
                            "negative_firings": 0,
                            "total_firings": 1,
                            "confidence": 1.0,
                            "positive_example_ids": ["ex_pos"],
                            "negative_example_ids": [],
                            "uses_candidate_atoms": True,
                            "num_candidate_body_atoms": 1,
                            "candidate_body_atom_ratio": 0.5,
                            "mixes_accepted_and_candidate_atoms": True,
                            "uses_only_candidate_atoms": False,
                            "candidate_rule_category": "mixed_accepted_candidate",
                            "initial_rule_pair_category": "accepted_candidate",
                            "evidence_set": [
                                {
                                    "example_id": "ex_pos",
                                    "label": True,
                                    "matched_atom": "",
                                    "bindings": {"S": "seg"},
                                    "matched_atoms": {
                                        "object_distance_state(S,O,near).": "object_distance_state(seg,obj,near).",
                                        "object_class(S,C,pedestrian).": "object_class(seg,cand,pedestrian).",
                                    },
                                }
                            ],
                        },
                    ],
                }
            ]

            with mock.patch(
                "src.exp_driving_videos.pipeline_config.get_merged_candidate_rules_output_root",
                return_value=root / "15",
            ):
                merged = pipeline_data.merge_candidate_rules(candidate_rule_results)

            self.assertEqual(merged["num_rules"], 1)
            self.assertEqual(merged["num_skipped_non_unary_input_rules"], 1)
            self.assertTrue(all(int(rule.get("body_length", 0)) == 1 for rule in merged["rules"]))
            self.assertEqual(
                merged["candidate_rule_stage_stats"]["skipped_non_unary_input_rules"],
                1,
            )

    def test_initial_rule_pruning_before_extension(self) -> None:
        def evidence(example_id: str, label: bool) -> dict:
            return {
                "example_id": example_id,
                "current_segment_id": f"{example_id}_cur",
                "next_segment_id": f"{example_id}_next",
                "label": label,
                "bindings": {"S": example_id},
                "matched_atoms": {"body": example_id},
            }

        def rule(
            rule_id: str,
            body_atoms: list[str],
            evidence_set: list[dict],
            category: str = "accepted_only",
            pair_category: str = "accepted_only",
        ) -> dict:
            positive_ids = sorted(entry["example_id"] for entry in evidence_set if entry["label"])
            negative_ids = sorted(entry["example_id"] for entry in evidence_set if not entry["label"])
            uses_candidate = category != "accepted_only"
            uses_only_candidate = category == "candidate_only"
            mixes = category == "mixed_accepted_candidate"
            candidate_count = sum(1 for atom in body_atoms if "(S,C)" in atom)
            return {
                "rule_id": rule_id,
                "head_predicate": "target_next",
                "head_atom_template": "target_next(S).",
                "body_atom_template": body_atoms[0] if len(body_atoms) == 1 else "",
                "body_atom_templates": body_atoms,
                "body_length": len(body_atoms),
                "clause": f"target_next(S) :- {', '.join(atom.rstrip('.') for atom in body_atoms)}.",
                "positive_support": len(set(positive_ids)),
                "negative_support": len(set(negative_ids)),
                "total_support": len(set(positive_ids) | set(negative_ids)),
                "positive_firings": len(positive_ids),
                "negative_firings": len(negative_ids),
                "total_firings": len(evidence_set),
                "confidence": len(positive_ids) / max(1, len(evidence_set)),
                "positive_example_ids": positive_ids,
                "negative_example_ids": negative_ids,
                "uses_candidate_atoms": uses_candidate,
                "num_candidate_body_atoms": candidate_count,
                "candidate_body_atom_ratio": candidate_count / max(1, len(body_atoms)),
                "mixes_accepted_and_candidate_atoms": mixes,
                "uses_only_candidate_atoms": uses_only_candidate,
                "candidate_rule_category": category,
                "initial_rule_pair_category": pair_category,
                "source_video_ids": ["video_a"],
                "source_rule_ids": [rule_id],
                "num_source_rules": 1,
                "evidence_set": evidence_set,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            merged = {
                "version": 3,
                "num_videos": 1,
                "num_examples": 5,
                "num_positive_examples": 4,
                "num_negative_examples": 1,
                "candidate_rule_stage_stats": {
                    "atom_availability": {
                        "candidate_semantic_body_atom_templates": ["candidate_semantic(S,C)."],
                        "candidate_provenance_only_body_atom_templates": ["candidate_source(S,C)."],
                    }
                },
                "candidate_rule_flow_summary": {},
                "rules": [
                    rule(
                        "dup_best",
                        ["accepted_feature(S,O)."],
                        [evidence("ex1", True), evidence("ex2", False)],
                    ),
                    rule(
                        "dup_worse",
                        ["accepted_feature(S,O).", "accepted_context(S)."],
                        [evidence("ex1", True), evidence("ex2", False)],
                    ),
                    rule(
                        "dominates_same_positive",
                        ["candidate_semantic(S,C)."],
                        [evidence("ex1", True)],
                        category="candidate_only",
                        pair_category="candidate_only",
                    ),
                    rule(
                        "mixed_a",
                        ["accepted_feature(S,O).", "candidate_semantic(S,C)."],
                        [evidence("ex3", True)],
                        category="mixed_accepted_candidate",
                        pair_category="accepted_candidate",
                    ),
                    rule(
                        "mixed_b",
                        ["accepted_context(S).", "candidate_source(S,C)."],
                        [evidence("ex4", True)],
                        category="mixed_accepted_candidate",
                        pair_category="accepted_candidate",
                    ),
                ],
            }

            pruned = pipeline_data.prune_initial_rules(
                merged,
                cfg={
                    "max_total_initial_rules": 10,
                    "max_accepted_only_initial_rules": 10,
                    "max_mixed_candidate_initial_rules": 1,
                    "max_candidate_only_initial_rules": 10,
                    "max_candidate_candidate_initial_rules": 10,
                    "diversity_key": "positive_coverage",
                },
                output_root=root / "15",
            )

            summary = pruned["candidate_rule_stage_stats"]["initial_rule_pruning"]
            kept_ids = {rule["rule_id"] for rule in pruned["rules"]}
            self.assertEqual(summary["firing_signature_deduplicated_num_rules"], 1)
            self.assertEqual(summary["dominance_pruned_num_rules"], 1)
            self.assertEqual(summary["budget_pruned_rule_counts"]["mixed_accepted_candidate_rules"], 1)
            self.assertIn("dominates_same_positive", kept_ids)
            self.assertNotIn("dup_best", kept_ids)
            self.assertNotIn("dup_worse", kept_ids)
            self.assertTrue((root / "15" / "pruned_initial_rules.json").exists())
            self.assertTrue((root / "15" / "pruned_initial_rules.csv").exists())
            self.assertTrue((root / "15" / "initial_rule_pruning_summary.json").exists())

            extended = extended_rules_driving_mini.run(
                merged_initial_rules=pruned,
                cfg={"num_rounds": 0},
                output_root=root / "16",
                force_recompute=True,
            )
            self.assertEqual(extended["input_num_initial_rules"], pruned["num_rules"])
            self.assertEqual(extended["input_num_unary_initial_rules_used"], 1)
            self.assertEqual(extended["input_num_skipped_non_unary_initial_rules"], 1)
            self.assertEqual(extended["num_all_kept_rules"], 1)

    def test_candidate_rules_flow_from_generation_to_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            temporal_rule_results = [
                {
                    "video_id": "video_candidate_flow",
                    "num_examples": 2,
                    "num_positive_examples": 1,
                    "num_negative_examples": 1,
                    "examples": [
                        {
                            "example_id": "ex_pos",
                            "target_predicate": "brake_next",
                            "label": True,
                            "current_segment_id": "seg_0001",
                            "next_segment_id": "seg_0002",
                            "accepted_body_atoms": [
                                "object_class(seg_0001,obj_0001,car).",
                                "object_distance_state(seg_0001,obj_0001,near).",
                            ],
                            "candidate_body_atoms": [
                                "object_class(seg_0001,obj_candidate_c1,pedestrian).",
                                "object_distance_state(seg_0001,obj_candidate_c1,near).",
                                "object_is_candidate(obj_candidate_c1).",
                                "object_source_type(obj_candidate_c1,candidate).",
                                "object_candidate_score_state(obj_candidate_c1,low).",
                                "object_matched_prior(obj_candidate_c1,prior_1).",
                            ],
                            "body_atoms": [
                                "segment_forward_state(seg_0001,forward_slowdown).",
                                "object_class(seg_0001,obj_0001,car).",
                                "object_distance_state(seg_0001,obj_0001,near).",
                                "object_class(seg_0001,obj_candidate_c1,pedestrian).",
                                "object_distance_state(seg_0001,obj_candidate_c1,near).",
                                "object_is_candidate(obj_candidate_c1).",
                                "object_source_type(obj_candidate_c1,candidate).",
                                "object_candidate_score_state(obj_candidate_c1,low).",
                                "object_matched_prior(obj_candidate_c1,prior_1).",
                            ],
                        },
                        {
                            "example_id": "ex_neg",
                            "target_predicate": "not_brake_next",
                            "label": False,
                            "current_segment_id": "seg_0003",
                            "next_segment_id": "seg_0004",
                            "accepted_body_atoms": [
                                "object_class(seg_0003,obj_0002,car).",
                            ],
                            "candidate_body_atoms": [
                                "object_class(seg_0003,obj_candidate_c2,pedestrian).",
                                "object_distance_state(seg_0003,obj_candidate_c2,far).",
                            ],
                            "body_atoms": [
                                "object_class(seg_0003,obj_0002,car).",
                                "object_class(seg_0003,obj_candidate_c2,pedestrian).",
                                "object_distance_state(seg_0003,obj_candidate_c2,far).",
                            ],
                        },
                    ],
                }
            ]

            step14_results = candidate_rules_driving_mini.run(
                temporal_rule_results=temporal_rule_results,
                cfg={
                    "target_predicate": "brake_next",
                    "min_positive_support": 1,
                    "use_only_positive_examples": True,
                    "include_example_ids": True,
                },
                output_root=root / "14",
                force_recompute=True,
            )
            step14_counts = step14_results[0]["candidate_rule_stage_stats"]["generated_rule_counts"]
            self.assertGreater(step14_counts["candidate_only_rules"], 0)
            self.assertEqual(step14_counts["mixed_accepted_candidate_rules"], 0)
            self.assertTrue(all(int(rule.get("body_length", 0)) == 1 for rule in step14_results[0]["initial_rules"]))
            self.assertFalse(
                any(
                    rule.get("body_atom_templates") == ["object_is_candidate(C)."]
                    for rule in step14_results[0]["initial_rules"]
                )
            )
            self.assertTrue(
                any("object_class(S,C,pedestrian)." in rule.get("body_atom_templates", []) for rule in step14_results[0]["initial_rules"])
            )

            with mock.patch(
                "src.exp_driving_videos.pipeline_config.get_merged_candidate_rules_output_root",
                return_value=root / "15",
            ):
                merged = pipeline_data.merge_candidate_rules(step14_results)
            step15_counts = merged["candidate_rule_stage_stats"]["merged_rule_counts"]
            self.assertGreater(step15_counts["candidate_only_rules"], 0)
            self.assertEqual(step15_counts["mixed_accepted_candidate_rules"], 0)
            self.assertTrue(all(int(rule.get("body_length", 0)) == 1 for rule in merged["rules"]))

            extended = extended_rules_driving_mini.run(
                merged_initial_rules=merged,
                cfg={
                    "num_rounds": 1,
                    "prune_strategies": ["empty_evidence"],
                    "min_positive_support_to_extend": 1,
                },
                output_root=root / "16",
                force_recompute=True,
            )
            step16_counts = extended["candidate_rule_stage_stats"]["all_kept_after_step16_rule_counts"]
            self.assertGreater(step16_counts["candidate_only_rules"], 0)
            self.assertGreater(extended["candidate_rule_stage_stats"]["extension_only_kept_rule_counts"]["all_rules"], 0)
            self.assertTrue(any(int(rule.get("body_length", 0)) > 1 for rule in extended["all_kept_rules"]))
            self.assertTrue(
                any(
                    bool(rule.get("uses_candidate_atoms", False))
                    and "object_distance_state(S,C,near)." in rule.get("body_atom_templates", [])
                    for rule in extended["all_kept_rules"]
                )
            )

            selected = final_rules_driving_mini.run(
                extended_rule_results=extended,
                cfg={
                    "top_k": 3,
                    "category_budgets": {
                        "accepted_only": 2,
                        "mixed_accepted_candidate": 0,
                        "candidate_only": 1,
                        "candidate_candidate": 0,
                    },
                },
                output_root=root / "17",
                force_recompute=True,
            )
            selected_counts = selected["candidate_rule_stage_stats"]["selected_rule_counts"]
            self.assertGreater(selected_counts["candidate_only_rules"], 0)
            self.assertGreater(selected_counts["accepted_only_rules"], 0)
            self.assertLessEqual(selected_counts["candidate_only_rules"], 1)
            self.assertEqual(selected["selection_input_stage"], "step16_post_pruned_kept_pool")
            self.assertGreater(selected_counts["all_rules"], 0)

            with mock.patch.object(evaluate_rules_driving_mini, "_save_evaluation_pdf"):
                evaluated = evaluate_rules_driving_mini.run(
                    final_rule_results=selected,
                    temporal_rule_results=temporal_rule_results,
                    cfg={"prediction_mode": "any_rule_positive"},
                    output_root=root / "18",
                    force_recompute=True,
                )
            evaluated_counts = evaluated["candidate_rule_stage_stats"]["evaluated_rule_counts"]
            self.assertGreater(evaluated_counts["candidate_only_rules"], 0)
            self.assertGreater(evaluated_counts["all_rules"], 0)
            self.assertIn("accepted_plus_mixed_rules", evaluated["rule_subset_metrics"])
            self.assertIn("accepted_plus_all_candidate_rules", evaluated["rule_subset_metrics"])
            self.assertIn("candidate_candidate_rules", evaluated["rule_subset_metrics"])
            self.assertIn("evaluation", evaluated["candidate_rule_flow_summary"])
            self.assertTrue((root / "18" / "candidate_rule_flow_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
