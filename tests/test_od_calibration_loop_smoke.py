from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from src.exp_driving_videos.modules import baseline_safe_calibration_gate_driving_mini
from src.exp_driving_videos.modules import detect_driving_mini
from src.exp_driving_videos.modules import od_calibration_policy_utils
from src.exp_driving_videos.modules import od_confidence_calibration_driving_mini
from src.exp_driving_videos.modules import reasoning_to_od_pseudo_labels_driving_mini


class ODCalibrationLoopSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self._tmpdir.name)
        self.loop_root = self.tmp_root / "od_calibration_loop"
        self.iterations_root = self.loop_root / "iterations"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _iteration_root(self, iteration_id: str) -> Path:
        path = self.iterations_root / str(iteration_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _patch_loop_paths(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            mock.patch.object(
                od_calibration_policy_utils,
                "get_od_calibration_loop_root",
                side_effect=lambda: self.loop_root,
            )
        )
        stack.enter_context(
            mock.patch.object(
                od_calibration_policy_utils,
                "get_od_calibration_iterations_root",
                side_effect=lambda: self.iterations_root,
            )
        )
        stack.enter_context(
            mock.patch.object(
                od_calibration_policy_utils,
                "get_iteration_root",
                side_effect=self._iteration_root,
            )
        )
        stack.enter_context(
            mock.patch.object(
                od_calibration_policy_utils,
                "get_active_od_calibration_policy_path",
                side_effect=lambda: self.loop_root / "active_policy.json",
            )
        )
        stack.enter_context(
            mock.patch.object(
                od_calibration_policy_utils,
                "get_active_od_calibration_state_path",
                side_effect=lambda: self.loop_root / "active_policy_state.json",
            )
        )
        return stack

    @staticmethod
    def _base_frame_record() -> Dict[str, Any]:
        return {
            "frame": "frame_000001.jpg",
            "frame_index": 1,
            "image_path": "/tmp/fake_frame.jpg",
            "accepted_detections": [
                {
                    "detection_id": "000001:accepted:0000",
                    "class": "car",
                    "score": 0.95,
                    "raw_score": 0.95,
                    "accepted": True,
                    "bbox": [0.0, 0.0, 40.0, 30.0],
                    "candidate_source": "accepted_high_confidence",
                    "prior_metadata": {
                        "matched_prior_ids": ["prior_accept"],
                        "prior_relevance_score": 0.8,
                    },
                }
            ],
            "candidate_detections": [
                {
                    "detection_id": "000001:candidate:0000",
                    "class": "pedestrian",
                    "score": 0.22,
                    "raw_score": 0.22,
                    "accepted": False,
                    "bbox": [10.0, 10.0, 24.0, 50.0],
                    "candidate_source": "low_confidence",
                    "prior_metadata": {
                        "matched_prior_ids": ["prior_pos"],
                        "prior_relevance_score": 0.9,
                    },
                },
                {
                    "detection_id": "000001:candidate:0001",
                    "class": "traffic_sign",
                    "score": 0.18,
                    "raw_score": 0.18,
                    "accepted": False,
                    "bbox": [30.0, 8.0, 42.0, 24.0],
                    "candidate_source": "position_provenance",
                    "prior_metadata": {
                        "matched_prior_ids": ["prior_neg"],
                        "prior_relevance_score": 0.1,
                    },
                },
            ],
        }

    def _detection_results(self) -> List[Dict[str, Any]]:
        frame = od_calibration_policy_utils.apply_policy_to_frame_record(self._base_frame_record(), None)
        frame["od_calibration"] = od_calibration_policy_utils.current_policy_marker({})
        return [
            {
                "video_id": "video_smoke",
                "frames": [frame],
                "od_calibration": od_calibration_policy_utils.current_policy_marker({}),
            }
        ]

    @staticmethod
    def _tracking_results() -> List[Dict[str, Any]]:
        return [
            {
                "video_id": "video_smoke",
                "candidate_tracks": {
                    "selected_candidate_tracks": {
                        "track_summaries": [
                            {
                                "track_id": 101,
                                "track_length": 3,
                                "temporal_consistency": 0.9,
                                "selection_score": 0.8,
                                "mean_score": 0.22,
                                "max_score": 0.25,
                                "prior_relevance_mean": 0.9,
                                "source_detection_ids": ["000001:candidate:0000"],
                            },
                            {
                                "track_id": 102,
                                "track_length": 2,
                                "temporal_consistency": 0.4,
                                "selection_score": 0.3,
                                "mean_score": 0.18,
                                "max_score": 0.18,
                                "prior_relevance_mean": 0.1,
                                "source_detection_ids": ["000001:candidate:0001"],
                            },
                        ]
                    }
                },
            }
        ]

    @staticmethod
    def _logic_atom_results() -> List[Dict[str, Any]]:
        return [
            {
                "video_id": "video_smoke",
                "segments": [
                    {
                        "segment_id": "seg_0001",
                        "candidate_objects": [
                            {
                                "object_id": "obj_candidate_001",
                                "candidate_track_id": 101,
                                "candidate_object_id": "cand_obj_001",
                                "object_class": "pedestrian",
                                "frame_detection_id": "000001:candidate:0000",
                                "source_detection_ids": ["000001:candidate:0000"],
                                "candidate_source": "low_confidence",
                                "selection_score": 0.8,
                                "prior_metadata": {
                                    "matched_prior_ids": ["prior_pos"],
                                    "prior_relevance_score": 0.9,
                                },
                                "track_quality": {"temporal_consistency": 0.9},
                            },
                            {
                                "object_id": "obj_candidate_002",
                                "candidate_track_id": 102,
                                "candidate_object_id": "cand_obj_002",
                                "object_class": "traffic_sign",
                                "frame_detection_id": "000001:candidate:0001",
                                "source_detection_ids": ["000001:candidate:0001"],
                                "candidate_source": "position_provenance",
                                "selection_score": 0.3,
                                "prior_metadata": {
                                    "matched_prior_ids": ["prior_neg"],
                                    "prior_relevance_score": 0.1,
                                },
                                "track_quality": {"temporal_consistency": 0.4},
                            },
                        ],
                    }
                ],
            }
        ]

    @staticmethod
    def _eval_temporal_rule_results() -> List[Dict[str, Any]]:
        return [
            {
                "video_id": "video_smoke",
                "examples": [
                    {
                        "example_id": "ex_pos",
                        "label": True,
                        "current_segment_id": "seg_0001",
                        "body_atoms": ["candidate_semantic(obj_candidate_001)."],
                    },
                    {
                        "example_id": "ex_neg",
                        "label": False,
                        "current_segment_id": "seg_0001",
                        "body_atoms": ["candidate_position(obj_candidate_002)."],
                    },
                ],
            }
        ]

    @staticmethod
    def _evaluation_results(f1_baseline: float = 0.50, f1_augmented: float = 0.60) -> Dict[str, Dict[str, Any]]:
        return {
            "selector_primary": {
                "candidate_rule_ablation": {
                    "baseline_metrics": {"precision": 0.55, "recall": 0.46, "f1": f1_baseline, "accuracy": 0.70},
                    "augmented_metrics": {"precision": 0.60, "recall": 0.60, "f1": f1_augmented, "accuracy": 0.78},
                },
                "rule_evaluations": [
                    {
                        "rule_id": "rule_positive",
                        "body_atom_templates": ["candidate_semantic(obj_candidate_001)."],
                        "uses_candidate_atoms": True,
                        "has_strong_candidate_semantic_atom": True,
                        "is_broad_candidate_rule": False,
                        "broad_candidate_rule_pattern": "",
                        "eval_fn_coverage_gain_count_vs_accepted_only": 1,
                        "eval_fp_contribution_count_vs_accepted_only": 0,
                        "eval_fn_coverage_gain_example_ids_vs_accepted_only": ["ex_pos"],
                        "eval_fp_contribution_example_ids_vs_accepted_only": [],
                        "eval_triggered_example_ids": ["ex_pos"],
                    },
                    {
                        "rule_id": "rule_negative",
                        "body_atom_templates": ["candidate_position(obj_candidate_002)."],
                        "uses_candidate_atoms": True,
                        "has_strong_candidate_semantic_atom": False,
                        "is_broad_candidate_rule": True,
                        "broad_candidate_rule_pattern": "object_x_position_state_only",
                        "eval_fn_coverage_gain_count_vs_accepted_only": 0,
                        "eval_fp_contribution_count_vs_accepted_only": 1,
                        "eval_fn_coverage_gain_example_ids_vs_accepted_only": [],
                        "eval_fp_contribution_example_ids_vs_accepted_only": ["ex_neg"],
                        "eval_triggered_example_ids": ["ex_neg"],
                    },
                ],
            }
        }

    @staticmethod
    def _rule_aggregation_results(f1_value: float = 0.65) -> Dict[str, Any]:
        return {
            "metrics_by_split": {
                "eval": {
                    "best_validation_threshold": {
                        "precision": 0.68,
                        "recall": 0.63,
                        "f1": f1_value,
                        "accuracy": 0.80,
                    }
                }
            }
        }

    def _run_accepting_iteration(self, iteration_id: str) -> Dict[str, Any]:
        detection_results = self._detection_results()
        tracking_results = self._tracking_results()
        pseudo_label_results = reasoning_to_od_pseudo_labels_driving_mini.run(
            detection_results=detection_results,
            tracking_results=tracking_results,
            logic_atom_results=self._logic_atom_results(),
            eval_temporal_rule_results=self._eval_temporal_rule_results(),
            evaluation_results_by_name=self._evaluation_results(),
            candidate_contribution_summary_results={"best_selector_by_delta_f1": "selector_primary"},
            primary_rule_set="selector_primary",
            iteration_id=iteration_id,
            output_root=self.iterations_root,
            force_recompute=True,
        )
        calibration_results = od_confidence_calibration_driving_mini.run(
            pseudo_label_results=pseudo_label_results,
            detection_results=detection_results,
            tracking_results=tracking_results,
            iteration_id=iteration_id,
            cfg={"min_labeled_detections": 99, "min_positive_detections": 2, "min_negative_detections": 2},
            output_root=self.iterations_root,
            force_recompute=True,
        )
        gate_results = baseline_safe_calibration_gate_driving_mini.run(
            evaluation_results_by_name=self._evaluation_results(),
            primary_rule_set="selector_primary",
            rule_aggregation_baseline_results=self._rule_aggregation_results(),
            calibration_results=calibration_results,
            iteration_id=iteration_id,
            output_root=self.iterations_root,
            force_recompute=True,
        )
        return {
            "detection_results": detection_results,
            "tracking_results": tracking_results,
            "pseudo_label_results": pseudo_label_results,
            "calibration_results": calibration_results,
            "gate_results": gate_results,
        }

    def test_calibration_loop_smoke_and_artifacts(self) -> None:
        with self._patch_loop_paths():
            identity_frame = od_calibration_policy_utils.apply_policy_to_frame_record(self._base_frame_record(), None)
            accepted_identity = identity_frame["accepted_detections"][0]
            candidate_identity = identity_frame["candidate_detections"][0]
            self.assertAlmostEqual(accepted_identity["raw_score"], accepted_identity["calibrated_score"])
            self.assertAlmostEqual(candidate_identity["raw_score"], candidate_identity["calibrated_score"])
            self.assertAlmostEqual(candidate_identity["feedback_bonus"], 0.0)
            self.assertEqual(od_calibration_policy_utils.load_active_od_calibration_policy(), {})

            outputs = self._run_accepting_iteration("iteration_0001")

            self.assertEqual(outputs["gate_results"]["decision"], "accept")
            self.assertEqual(outputs["gate_results"]["active_policy_after_id"], "od_calibration_iteration_0001")
            self.assertEqual(outputs["calibration_results"]["policy"]["policy_type"], "heuristic_additive")

            label_by_detection = {
                row["detection_id"]: row["pseudo_label"]
                for row in outputs["pseudo_label_results"]["detection_pseudo_labels"]
            }
            self.assertEqual(label_by_detection["000001:candidate:0000"], "positive")
            self.assertEqual(label_by_detection["000001:candidate:0001"], "negative")

            active_policy = od_calibration_policy_utils.load_active_od_calibration_policy()
            self.assertEqual(active_policy.get("policy_id"), "od_calibration_iteration_0001")
            calibrated_frame = od_calibration_policy_utils.apply_policy_to_frame_record(
                self._base_frame_record(),
                active_policy,
            )
            accepted_after = calibrated_frame["accepted_detections"][0]
            candidate_after = {row["detection_id"]: row for row in calibrated_frame["candidate_detections"]}
            self.assertAlmostEqual(accepted_after["raw_score"], accepted_after["calibrated_score"])
            self.assertAlmostEqual(accepted_after["feedback_bonus"], 0.0)
            self.assertTrue(abs(candidate_after["000001:candidate:0000"]["feedback_bonus"]) > 1e-9)
            self.assertTrue(abs(candidate_after["000001:candidate:0001"]["feedback_bonus"]) > 1e-9)
            self.assertEqual(
                candidate_after["000001:candidate:0000"]["od_calibration"]["calibration_branch"],
                "candidate_exploration",
            )

            artifact_check = od_calibration_policy_utils.validate_iteration_artifacts(
                "iteration_0001",
                output_root=self.iterations_root,
            )
            self.assertTrue(artifact_check["ok"], artifact_check["errors"])

    def test_gate_reject_keeps_previous_policy(self) -> None:
        with self._patch_loop_paths():
            self._run_accepting_iteration("iteration_0001")
            active_before = od_calibration_policy_utils.load_active_od_calibration_policy()

            reject_policy = dict(active_before)
            reject_policy["policy_id"] = "od_calibration_iteration_0002"
            reject_policy["source_iteration_id"] = "iteration_0002"
            reject_results = baseline_safe_calibration_gate_driving_mini.run(
                evaluation_results_by_name=self._evaluation_results(f1_baseline=0.40, f1_augmented=0.45),
                primary_rule_set="selector_primary",
                rule_aggregation_baseline_results=self._rule_aggregation_results(f1_value=0.50),
                calibration_results={"policy": reject_policy},
                iteration_id="iteration_0002",
                output_root=self.iterations_root,
                force_recompute=True,
            )

            self.assertEqual(reject_results["decision"], "reject")
            self.assertEqual(reject_results["active_policy_after_id"], "od_calibration_iteration_0001")
            active_after = od_calibration_policy_utils.load_active_od_calibration_policy()
            self.assertEqual(active_after.get("policy_id"), "od_calibration_iteration_0001")


class DetectDependencyContractTest(unittest.TestCase):
    def test_dependency_status_contract(self) -> None:
        status = detect_driving_mini.get_detector_dependency_status(render_video=True)
        self.assertIn("cv2_available", status)
        self.assertIn("detector_backend_available", status)
        self.assertIn("render_video_available", status)


if __name__ == "__main__":
    unittest.main()
