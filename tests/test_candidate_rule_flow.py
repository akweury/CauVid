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
            self.assertGreater(step14_counts["mixed_accepted_candidate_rules"], 0)
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
            self.assertGreater(step15_counts["mixed_accepted_candidate_rules"], 0)

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
            self.assertGreater(step16_counts["mixed_accepted_candidate_rules"], 0)
            self.assertTrue(
                any(
                    bool(rule.get("uses_candidate_atoms", False))
                    and "object_distance_state(S,C,near)." in rule.get("body_atom_templates", [])
                    for rule in extended["all_kept_rules"]
                )
            )

            selected = final_rules_driving_mini.run(
                extended_rule_results=extended,
                cfg={"top_k": 100},
                output_root=root / "17",
                force_recompute=True,
            )
            selected_counts = selected["candidate_rule_stage_stats"]["selected_rule_counts"]
            self.assertGreater(selected_counts["candidate_only_rules"], 0)
            self.assertGreater(selected_counts["mixed_accepted_candidate_rules"], 0)

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
            self.assertGreater(evaluated_counts["mixed_accepted_candidate_rules"], 0)
            self.assertIn("evaluation", evaluated["candidate_rule_flow_summary"])
            self.assertTrue((root / "18" / "candidate_rule_flow_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
