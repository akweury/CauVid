import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.exp_july.perception.adaptive_motion_repair import _recompute_motion
from src.exp_july.perception.pipeline import (
    _uncertain_signal_evidence_video,
    step8c_trajectory_clustering,
    step8d_closed_loop_trajectory_repair,
)
from src.exp_july.perception.trajectory_pattern_closed_loop import (
    PATTERNS,
    RESIDUALS,
    run_trajectory_pattern_closed_loop,
)
from src.exp_july.perception.trajectory_pattern_epoch import begin_epoch, default_policy


def observations():
    rows = []
    for frame_id in range(6):
        rows.append(
            {
                "frame_index": frame_id,
                "frame_label": "car",
                "bbox": [100 + frame_id, 80, 150 + frame_id, 130],
                "position_3d": [0.1 * frame_id, 0.0, 12.0 - 0.1 * frame_id],
                "motion": {"ego_vx": 0.0, "ego_vz": 0.0},
                "provenance": {"source": "observed", "is_observed": True},
                "uncertainty": {"score": 0.95, "source_uncertainty": 0.01},
            }
        )
    return _recompute_motion(rows, {})


def state():
    obs = observations()
    frames = []
    for row in obs:
        motion = dict(row["motion"])
        frames.append(
            {
                "frame_index": row["frame_index"],
                "image_path": "",
                "objects": [
                    {
                        "track_id": 7,
                        "label": "car",
                        "frame_label": "car",
                        "bbox": row["bbox"],
                        "box": row["bbox"],
                        "position_3d": row["position_3d"],
                        "relative_position_3d": row["position_3d"],
                        "score": 0.95,
                        **motion,
                    }
                ],
            }
        )
    relative_video = {
        "video_id": "demo",
        "num_frames": 6,
        "frames": frames,
    }
    return {
        "videos": ["demo"],
        "step8b_evidence_type": "uncertain_signal_evidence",
        "uncertain_signal_evidence": [
            _uncertain_signal_evidence_video(relative_video)
        ],
        "relative_object_motion": [relative_video],
        "ego_motion": [{"video_id": "demo", "frames": []}],
    }


def llm(kind, _prompt):
    if kind == "cohort_rule_generation":
        return {
            "rules": [
                {
                    "rule_id": "persistent_vehicle",
                    "description": "Persistent vehicle tracks",
                    "priority": 50,
                    "all": [
                        {
                            "attribute": "category",
                            "operator": "in",
                            "value": ["car", "truck", "bus"],
                        },
                        {
                            "attribute": "track_length_bucket",
                            "operator": "in",
                            "value": ["medium", "long"],
                        },
                    ],
                },
                {
                    "rule_id": "other_tracks",
                    "description": "Catch all",
                    "priority": 0,
                    "all": [],
                },
            ],
            "rationale": "Static metadata cohorts only",
        }
    if kind == "cohort_repair_selection":
        return {
            "plans": [
                {
                    "cohort_id": "persistent_vehicle",
                    "operator": "outlier_removal",
                    "initial_parameters": {
                        "median_radius": 2,
                        "mad_scale": 3.0,
                    },
                    "anomaly_types": [
                        "track_drift",
                        "bbox_jump",
                        "depth_jump",
                        "speed_abnormal_change",
                    ],
                    "rationale": "Robustly remove cohort outliers",
                },
                {
                    "cohort_id": "other_tracks",
                    "operator": "no_repair",
                    "initial_parameters": {},
                    "anomaly_types": [],
                    "rationale": "No systematic anomaly",
                },
            ]
        }
    if kind == "policy_interval_review":
        return {
            "policy_patch": {
                "residual_weights": {"depth_consistency": 1.1},
                "pattern_biases": {},
                "repair_preferences": {"approaching": ["motion_recomputation"]},
            },
            "rationale": "aggregated interval evidence supports a small candidate change",
            "critical_regressions": [],
        }
    if kind in {"batch_stage1", "stage1_individual"}:
        inputs = json.loads(_prompt.split("inputs=", 1)[1])
        return {"results": [
            {
                "track_uid": row["track_uid"],
                "assessments": [
                    {
                        "pattern_id": pattern,
                        "plausibility": 0.8 if pattern == "approaching" else 0.2,
                        "ignorable_errors": ["minor bbox jitter"],
                        "structural_conflicts": [],
                        "explanation": "independent residual interpretation",
                    }
                    for pattern in PATTERNS
                ],
                "requires_repair_planning": False,
                "batch_confidence": 0.9,
                "batch_conflicts": [],
            }
            for row in inputs
        ]}
    if kind in {"batch_stage2", "stage2_individual"}:
        inputs = json.loads(_prompt.split("inputs=", 1)[1])
        return {"results": [
            {
                "track_uid": row["track_uid"],
                "repair_recommendations": {
                    pattern: ["motion_recomputation"] for pattern in PATTERNS
                },
            }
            for row in inputs
        ]}
    if kind == "pattern_enumeration":
        return {
            "patterns": [
                {
                    "pattern_id": pattern,
                    "required_metrics": list(RESIDUALS),
                    "qualitative_constraints": ["grounded qualitative relation"],
                    "justification": "uses the supplied trajectory signals",
                }
                for pattern in PATTERNS
            ]
        }
    if kind == "residual_interpretation":
        return {
            "assessments": [
                {
                    "pattern_id": pattern,
                    "plausibility": 0.8 if pattern == "approaching" else 0.2,
                    "ignorable_errors": ["minor bbox jitter"],
                    "structural_conflicts": [],
                    "recommended_repairs": ["motion_recomputation", "kalman_smoothing"],
                    "explanation": "based on residuals and observed provenance",
                }
                for pattern in PATTERNS
            ]
        }
    if kind == "statistics_review":
        return {
            "candidate_update": {
                "rationale": "reviewed validation statistics",
                "residual_priority": ["depth_consistency", "continuity"],
                "pattern_hypotheses": ["approaching remains plausible"],
                "critical_regressions": [],
            }
        }
    raise AssertionError(kind)


class TrajectoryPatternClosedLoopTests(unittest.TestCase):
    def test_clustering_and_repair_are_separate_active_stages(self):
        source = state()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.exp_july.perception.pipeline.get_pipeline_output_root",
            return_value=Path(tmp),
        ):
            clustered = step8c_trajectory_clustering(source, llm_generate=llm)
            self.assertEqual(
                clustered["trajectory_clustering_manifest"]["repairs_performed"],
                0,
            )
            self.assertNotIn("trajectory_pattern_records", clustered)
            self.assertTrue(clustered["trajectory_clustered_tracks"])
            self.assertTrue(
                all(track.get("cohort_id") for track in clustered["trajectory_clustered_tracks"])
            )

            repaired = step8d_closed_loop_trajectory_repair(
                clustered,
                llm_generate=llm,
            )
            self.assertTrue(repaired["trajectory_pattern_records"])
            self.assertNotIn("trajectory_pattern_visualizations", repaired)
            self.assertEqual(
                repaired["trajectory_pattern_records"][0]["trajectory_cohort_id"],
                clustered["trajectory_clustered_tracks"][0]["cohort_id"],
            )
            self.assertEqual(
                repaired["trajectory_pattern_manifest"]["repair_cache_hits"],
                0,
            )
            cached = step8d_closed_loop_trajectory_repair(
                clustered,
                llm_generate=llm,
            )
            self.assertEqual(
                cached["trajectory_pattern_manifest"]["repair_cache_hits"],
                1,
            )
            self.assertTrue(
                cached["trajectory_pattern_records"][0]["llm_processing"][
                    "repair_cache_hit"
                ]
            )

    def test_pending_policy_activates_only_at_the_next_epoch_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            epoch_id, frozen, _ = begin_epoch(root)
            self.assertEqual(epoch_id, 1)
            self.assertEqual(frozen["version"], 1)
            pending = default_policy()
            pending.update({"version": 2, "parent_version": 1, "status": "pending"})
            pending["residual_weights"]["depth_consistency"] = 1.2
            (root / "pending_policy.json").write_text(json.dumps(pending), encoding="utf-8")
            # Epoch 1 remains frozen even after a candidate has been staged.
            self.assertEqual(frozen["residual_weights"]["depth_consistency"], 1.0)
            epoch_id, next_frozen, snapshot = begin_epoch(root)
            self.assertEqual(epoch_id, 2)
            self.assertTrue(snapshot["activated_pending_policy"])
            self.assertEqual(next_frozen["version"], 2)
            self.assertEqual(next_frozen["residual_weights"]["depth_consistency"], 1.2)

    def test_all_patterns_residuals_audit_statistics_and_original_branch_are_saved(self):
        source = state()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_trajectory_pattern_closed_loop(
                source,
                root,
                llm_generate=llm,
            )
            self.assertTrue((root / "trajectory_pattern_manifest.json").exists())
            self.assertFalse((root / "statistics" / "current_table.json").exists())
            self.assertTrue(list((root / "statistics").glob("candidate_table_v*.json")))
            self.assertEqual(
                result["trajectory_pattern_statistics_promotion"]["reason"],
                "independent_validation_split_unavailable",
            )
            self.assertTrue(list((root / "llm_audit" / "policy_interval_review").glob("*.json")))
            self.assertTrue(
                list((root / "llm_audit" / "cohort_rule_generation").glob("*.json"))
            )
            self.assertTrue(
                list((root / "llm_audit" / "cohort_repair_selection").glob("*.json"))
            )
            for filename in (
                "metadata_catalog.json",
                "compiled_rules.json",
                "cohort_statistics.json",
                "operator_library.json",
                "calibrated_operator_plans.json",
                "frozen_policy.json",
            ):
                self.assertTrue((root / "cohorts" / filename).exists())
            self.assertEqual(result["trajectory_pattern_manifest"]["interval_review_count"], 1)
            self.assertTrue(result["trajectory_pattern_manifest"]["policy_frozen"])
            self.assertTrue((root / "policies" / "active_policy.json").exists())
            self.assertTrue((root / "policies" / "epoch_0001.json").exists())
            self.assertTrue(list((root / "epoch_reviews").glob("*_package.json")))
            visualization_root = root / "visualizations"
            visualization_files = [
                path for path in visualization_root.rglob("*") if path.is_file()
            ]
            self.assertTrue(visualization_files)
            self.assertTrue(
                all(path.suffix.lower() in {".mp4", ".pdf"} for path in visualization_files)
            )
            self.assertEqual(
                len(list((visualization_root / "statistics_pdfs").glob("*.pdf"))),
                3,
            )
            self.assertFalse(list(visualization_root.rglob("*.html")))
            self.assertFalse(list(visualization_root.rglob("*.json")))
            dashboard = root / "dashboard" / "index.html"
            self.assertTrue(dashboard.exists())
            html = dashboard.read_text(encoding="utf-8")
            for marker in (
                "Step 8C Prior-Guided Statistical Repair Dashboard", "Raw trajectory playback",
                "Repaired trajectory playback", "Interactive signals",
                "Semantic cohort and calibrated repair policy",
                "Pattern and residual comparison", "Repair candidates",
                "Symbolic validation", "LLM audit records",
                "Dataset-level ablation summary", "READ ONLY",
            ):
                self.assertIn(marker, html)
            self.assertNotIn("https://", html)
            self.assertTrue((root / "dashboard" / "dashboard_manifest.json").exists())

        record = result["trajectory_pattern_records"][0]
        self.assertEqual(
            record["symbolic_track"]["source_evidence_type"],
            "uncertain_signal_evidence",
        )
        self.assertIn("observable_cues", record["symbolic_track"])
        self.assertNotIn("source_validation", record["symbolic_track"])
        self.assertNotIn("source_decision", record["symbolic_track"])
        self.assertEqual(
            result["trajectory_pattern_manifest"]["input_evidence_type"],
            "uncertain_signal_evidence",
        )
        self.assertEqual(
            result["trajectory_pattern_manifest"]["method"],
            "prior_guided_statistical_signal_repair",
        )
        self.assertFalse(
            result["trajectory_pattern_manifest"]["llm_direct_trajectory_repair"]
        )
        self.assertEqual(record["trajectory_cohort_id"], "persistent_vehicle")
        self.assertEqual(
            record["activated_rule"]["source"],
            "llm_static_metadata_rule",
        )
        self.assertEqual(
            record["cohort_operator_plan"]["operator"],
            "no_repair",
        )
        self.assertEqual(
            record["cohort_operator_plan"]["llm_requested_operator"],
            "outlier_removal",
        )
        self.assertEqual(record["resolution_status"], "validated_no_repair")
        self.assertEqual(
            record["record_status"],
            "completed_validated_original_preserved",
        )
        self.assertIn(
            "calibrated_parameters", record["cohort_operator_plan"]
        )
        self.assertEqual(len(record["pattern_candidates"]), len(PATTERNS))
        for candidate in record["pattern_candidates"]:
            self.assertEqual(set(candidate["residual_vector"]), set(RESIDUALS))
        if not record["candidate_repairs"]:
            self.assertTrue(record["llm_processing"]["repair_fast_path"])
            self.assertEqual(
                record["final_selection_reason"],
                "valid_no_repair_fast_path_original_preserved",
            )
        else:
            self.assertTrue(record["candidate_repairs"])
        required_candidate_fields = {
            "pre_pattern_scores", "post_repair_pattern_scores", "LLM_prior",
            "repair_hypothesis", "pattern_hypothesis", "symbolic_verdict",
            "hard_constraint_results", "final_score", "validated_pattern",
            "final_selection_reason",
        }
        for repair in record["candidate_repairs"]:
            self.assertTrue(required_candidate_fields.issubset(repair))
            if repair["symbolic_verdict"] == "reject":
                self.assertIsNone(repair["final_score"])
        sources = {
            source
            for repair in record["candidate_repairs"]
            for source in repair["pattern_hypothesis"]["selection_sources"]
        }
        if record["candidate_repairs"]:
            self.assertIn("mandatory_unknown_baseline", sources)
            self.assertIn("minimum_residual_baseline", sources)
        self.assertIn("LLM_preferred_pattern", record)
        self.assertEqual(record["provenance"]["frozen_policy_version"], 1)
        self.assertEqual(record["provenance"]["epoch_id"], 1)
        self.assertEqual(
            record["provenance"]["llm_role"],
            "static_cohort_rule_generation_and_cohort_operator_selection_only",
        )
        self.assertEqual(
            record["provenance"]["activated_rule_id"],
            "persistent_vehicle",
        )
        self.assertIn("validated_pattern", record)
        self.assertIn(record["final_selection_reason"], {
            "highest_ranked_after_hard_constraints",
            "no_candidate_passed_hard_constraints_original_preserved",
            "no_repair_required_original_preserved",
            "valid_no_repair_fast_path_original_preserved",
        })
        self.assertEqual(
            result["pre_pattern_relative_object_motion"],
            source["relative_object_motion"],
        )
        cohort_summary = result["trajectory_cohort_statistics"][
            "persistent_vehicle"
        ]
        self.assertEqual(cohort_summary["track_count"], 1)
        self.assertIn("motion_statistics", cohort_summary)
        self.assertIn("systematic_anomalies", cohort_summary)
        self.assertEqual(result["trajectory_pattern_visualizations"], [])
        self.assertEqual(result["trajectory_pattern_video_summaries"], [])
        self.assertEqual(
            len(result["trajectory_pattern_statistical_pdf_reports"]), 3
        )


    def test_statistics_update_and_promotion_splits_are_video_disjoint(self):
        source = state()
        source["videos"].append("demo_validation")
        for key in (
            "uncertain_signal_evidence",
            "relative_object_motion",
            "ego_motion",
        ):
            duplicate = copy.deepcopy(source[key][0])
            duplicate["video_id"] = "demo_validation"
            source[key].append(duplicate)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_trajectory_pattern_closed_loop(source, Path(tmp), llm_generate=llm)
        candidate = result["trajectory_pattern_statistics_candidate"]
        update_ids = set(candidate["update_video_ids"])
        validation_ids = set(candidate["validation_video_ids"])
        self.assertTrue(update_ids)
        self.assertTrue(validation_ids)
        self.assertFalse(update_ids & validation_ids)
        promotion = result["trajectory_pattern_statistics_promotion"]
        self.assertTrue(promotion["independent_split"])
        self.assertEqual(promotion["update_video_ids"], candidate["update_video_ids"])
        self.assertEqual(promotion["validation_video_ids"], candidate["validation_video_ids"])
if __name__ == "__main__":
    unittest.main()
